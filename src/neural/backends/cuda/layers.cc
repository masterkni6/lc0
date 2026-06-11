/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#include "layers.h"

#include <atomic>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cuda_common.h"
#include "gemm_tuner.h"
#include "kernels.h"
#include "neural/network.h"
#include "weight_arena.h"
#include "neural/tables/attention_policy_map.h"
#include "utils/fp16_utils.h"

#include <type_traits>

#if CUDART_VERSION >= 11010
#define LAYERS_EXTERNAL_EVENTS 1
#else
#define LAYERS_EXTERNAL_EVENTS 0
#endif

namespace lczero {

#if 0
// debug code to dump allocation in GPU memory
template <typename T>
void dumpTensor(T* memory, int elements, const char* message, bool only_summary = false) {
    const bool fp16 = std::is_same<half, T>::value;
    printf("\n%s\n", message);
    int elementSize = (int) (fp16 ? sizeof(half) : sizeof(float));
    int bytes = elements * elementSize;
    void *temp = malloc(bytes);
    cudaMemcpy(temp, memory, bytes, cudaMemcpyDeviceToHost);
    float maxval = -std::numeric_limits<float>::max();
    float minval = std::numeric_limits<float>::max();
    int nans = 0;
    int nanss[10] {};

    for (int i = 0; i < elements; i++)
    {
        float val;
        if (fp16) 
        {
            half *arr = (half*)temp;
            val = (float)arr[i];
        }
        else
        {
            float *arr = (float *)temp;
            val = arr[i];
        }
        maxval = std::max(maxval, val);
        minval = std::min(minval, val);

        if (std::isnan(val)) {
          if (nans < 10) nanss[nans] = i;
          nans++;
        }

        if (!only_summary || i < 2 || i == elements - 1) {
          // printf("%8.4f ", val);
          // if ((i % 8) == 7) printf("\n");
          printf("%i;%.6f\n", i, val);
        }
    }
    free(temp);
    if (maxval == -std::numeric_limits<float>::max())
       maxval = std::numeric_limits<double>::quiet_NaN();
    if (minval == std::numeric_limits<float>::max())
       minval = std::numeric_limits<double>::quiet_NaN();

    printf("Max: %.6f, Min: %.6f, NaNs: %i of %i", maxval, minval, nans, elements);
    printf("\nNaN indices: ");
    for (int i=0; i<nans && i<10; i++) printf("%i ", nanss[i]);
    if (nans > 10) printf("......");
    printf("\n");
}
#endif

namespace cudnn_backend {

// `tl_weight_arena` + `WeightArenaScope` are declared in weight_arena.h
// so both this TU and network_cuda.cc can use them.  `allocAndUpload`
// below consults `tl_weight_arena` to decide whether to allocate from
// the arena (Phase B.3 activation: arena lives, scope entered) or via
// cudaMalloc (default; arena dormant).
//
// Layer constructors that want to be arena-aware capture the arena
// pointer at construction time (member `arena_ = tl_weight_arena`)
// and use `WeightArena::Owns()` in their destructor to decide whether
// to call cudaFree on a given weight buffer.

// Helper for arena-aware weight free.  Drop-in replacement for
// `cudaFree(ptr)` calls in layer destructors that free weight buffers.
//
// Semantics:
//   - ptr == nullptr → no-op (matches cudaFree(NULL) being valid)
//   - arena != nullptr AND arena->Owns(ptr) → skip cudaFree (the arena
//     will free the underlying chunk when its destructor runs)
//   - otherwise → cudaFree(ptr) (the pointer came from a direct
//     cudaMalloc — runtime buffer, or weight from a non-arena path)
//
// Layers that don't use the arena have `arena_ == nullptr`, and
// `FreeWeight(ptr, nullptr)` degenerates to the original
// `cudaFree(ptr)` call.  This means non-arena layers' destructors
// keep their existing behavior after the cudaFree → FreeWeight swap.
inline void FreeWeight(void* ptr, WeightArena* arena) {
  if (ptr == nullptr) return;
  if (arena != nullptr && arena->Owns(ptr)) return;
  ReportCUDAErrors(cudaFree(ptr));
}

// Use Single kernel for entire SE operation.
// Right now supported only for fp16 with nhwc and it's quite a bit faster
// than using multiple passes. The flag can be set to false for debugging.
static constexpr bool kUseFusedSELayer = true;

template <typename DataType>
BaseLayer<DataType>::BaseLayer(int c, int h, int w, BaseLayer* ip, bool nhwc)
    : input_(ip), C(c), H(h), W(w), nhwc_(nhwc), use_gemm_ex_(false) {}

template <typename DataType>
BaseLayer<DataType>::BaseLayer(int c, int h, int w, BaseLayer* ip, bool nhwc,
                               bool gemm_ex)
    : input_(ip), C(c), H(h), W(w), nhwc_(nhwc), use_gemm_ex_(gemm_ex) {}

template <typename DataType>
BaseLayer<DataType>::BaseLayer(int c, int h, int w, BaseLayer* ip)
    : input_(ip),
      C(c),
      H(h),
      W(w),
      nhwc_(ip ? ip->nhwc_ : false),
      use_gemm_ex_(false) {}

#ifdef USE_CUDNN
template <typename DataType>
void ConvLayer<DataType>::init() {
  // Allocate memory for weights (filter tensor) and biases.
  const size_t weight_size =
      sizeof(DataType) * c_input_ * C * filter_size_ * filter_size_;
  ReportCUDAErrors(cudaMalloc(&weights, weight_size));

  const size_t bias_size = sizeof(DataType) * C;
  ReportCUDAErrors(cudaMalloc(&biases, bias_size));

  const bool fp16 = std::is_same<half, DataType>::value;
  const cudnnDataType_t dataType =
      std::is_same<half, DataType>::value ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;

  const cudnnTensorFormat_t layout =
      nhwc_ ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;

  // Create cudnn objects for various tensors, algorithms, etc.
  cudnnCreateFilterDescriptor(&filter_desc_);
  cudnnCreateConvolutionDescriptor(&conv_desc_);
  cudnnCreateTensorDescriptor(&out_tensor_desc_);
  cudnnCreateTensorDescriptor(&in_tensor_desc_);
  cudnnCreateTensorDescriptor(&bias_desc_);
  cudnnCreateActivationDescriptor(&activation_);

  cudnnSetFilter4dDescriptor(filter_desc_, dataType, layout, GetC(), c_input_,
                             filter_size_, filter_size_);

  ReportCUDNNErrors(
      cudnnSetTensor4dDescriptor(bias_desc_, layout, dataType, 1, C, 1, 1));

  const int padding = filter_size_ / 2;
  const bool crossCorr = 1;

  ReportCUDNNErrors(cudnnSetConvolution2dDescriptor(
      conv_desc_, padding, padding, 1, 1, 1, 1,
      crossCorr ? CUDNN_CROSS_CORRELATION : CUDNN_CONVOLUTION, dataType));

  if (fp16 && nhwc_)
    ReportCUDNNErrors(
        cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH));

  // TODO: dynamic selection of algorithm!
  if ((C > 32) && (!nhwc_) && (filter_size_ > 1)) {
    conv_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
  } else {
    conv_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  }

  if (act_ == ACTIVATION_RELU) {
    cudnnSetActivationDescriptor(activation_, CUDNN_ACTIVATION_RELU,
                                 CUDNN_NOT_PROPAGATE_NAN, 0.0);
  }
#if CUDNN_MAJOR != 7 || CUDNN_MINOR != 0
  else {
    cudnnSetActivationDescriptor(activation_, CUDNN_ACTIVATION_IDENTITY,
                                 CUDNN_NOT_PROPAGATE_NAN, 0.0);
  }
#endif
}

template <typename DataType>
ConvLayer<DataType>::ConvLayer(BaseLayer<DataType>* ip, int C, int H, int W,
                               int filter, int Cin,
                               ActivationFunction activation, bool bias)
    : BaseLayer<DataType>(C, H, W, ip),
      c_input_(Cin),
      filter_size_(filter),
      act_(activation),
      use_bias_(bias) {
  init();
}

template <typename DataType>
ConvLayer<DataType>::ConvLayer(bool nhwc, int C, int H, int W, int filter,
                               int Cin, ActivationFunction activation,
                               bool bias)
    : BaseLayer<DataType>(C, H, W, nullptr, nhwc),
      c_input_(Cin),
      filter_size_(filter),
      act_(activation),
      use_bias_(bias) {
  init();
}

template <>
void ConvLayer<half>::LoadWeights(float* pfilter, float* pBias, void* scratch) {
  const size_t weight_size =
      sizeof(float) * c_input_ * C * filter_size_ * filter_size_;
  const size_t bias_size = sizeof(float) * C;
  // Also need to convert from fp32 NCHW to fp16 NHWC
  // first copy from CPU memory to scratch space in GPU memory
  // and then do the type / layout conversion using a kernel.
  assert(scratch);
  ReportCUDAErrors(
      cudaMemcpy(scratch, pfilter, weight_size, cudaMemcpyHostToDevice));

  if (nhwc_) {
    convertNCHWtoNHWC((half*)weights, (float*)scratch, C, c_input_, C, c_input_,
                      filter_size_, filter_size_, 0);
  } else {
    copyTypeConverted((half*)weights, (float*)scratch,
                      C * c_input_ * filter_size_ * filter_size_, 0);
  }

  if (pBias) {
    ReportCUDAErrors(
        cudaMemcpy(scratch, pBias, bias_size, cudaMemcpyHostToDevice));

    copyTypeConverted((half*)biases, (float*)scratch, C, 0);
  }
}

template <>
void ConvLayer<float>::LoadWeights(float* pfilter, float* pBias,
                                   void* /*scratch*/) {
  const size_t weight_size =
      sizeof(float) * c_input_ * C * filter_size_ * filter_size_;
  const size_t bias_size = sizeof(float) * C;
  ReportCUDAErrors(
      cudaMemcpy(weights, pfilter, weight_size, cudaMemcpyHostToDevice));

  if (pBias) {
    ReportCUDAErrors(
        cudaMemcpy(biases, pBias, bias_size, cudaMemcpyHostToDevice));
  } else {
    ReportCUDAErrors(cudaMemset(biases, 0, bias_size));
  }
}

template <typename DataType>
void ConvLayer<DataType>::Eval(int N, DataType* output, const DataType* input,
                               const DataType* input2, void* scratch,
                               size_t scratch_size, cudnnHandle_t cudnn,
                               cublasHandle_t /*cublas*/, cudaStream_t stream,
                               DataType***) {
  const cudnnDataType_t dataType =
      std::is_same<half, DataType>::value ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;

  const cudnnTensorFormat_t layout =
      nhwc_ ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;

  ReportCUDNNErrors(cudnnSetTensor4dDescriptor(out_tensor_desc_, layout,
                                               dataType, N, C, H, W));

  ReportCUDNNErrors(cudnnSetTensor4dDescriptor(in_tensor_desc_, layout,
                                               dataType, N, c_input_, H, W));

  float alpha = 1.0f, beta = 0.0f;

  if (!(act_ != ACTIVATION_NONE || use_bias_ || input2)) {
    ReportCUDNNErrors(cudnnConvolutionForward(
        cudnn, &alpha, in_tensor_desc_, input, filter_desc_, weights,
        conv_desc_, conv_algo_, scratch, scratch_size, &beta, out_tensor_desc_,
        output));
  }
#if CUDNN_MAJOR != 7 || CUDNN_MINOR != 0
  else if (input2 && (act_ == ACTIVATION_RELU || act_ == ACTIVATION_NONE) &&
           use_bias_) {
    // fused bias + sum + relu!
    ReportCUDNNErrors(cudnnConvolutionBiasActivationForward(
        cudnn, &alpha, in_tensor_desc_, input, filter_desc_, weights,
        conv_desc_, conv_algo_, scratch, scratch_size, &alpha, out_tensor_desc_,
        input2, bias_desc_, biases, activation_, out_tensor_desc_, output));
  } else {
    // For some reason cudnn doesn't support just Convolution + Bias with nchw
    // (winograd algorithm) it works fine when RELU is also needed which is
    // somewhat strange.
    if ((act_ == ACTIVATION_RELU || (act_ == ACTIVATION_NONE && nhwc_)) &&
        !input2 && use_bias_) {
      ReportCUDNNErrors(cudnnConvolutionBiasActivationForward(
          cudnn, &alpha, in_tensor_desc_, input, filter_desc_, weights,
          conv_desc_, conv_algo_, scratch, scratch_size, &beta,
          out_tensor_desc_, output, bias_desc_, biases, activation_,
          out_tensor_desc_, output));
    } else {
      // The no special case path...
      ReportCUDNNErrors(cudnnConvolutionForward(
          cudnn, &alpha, in_tensor_desc_, input, filter_desc_, weights,
          conv_desc_, conv_algo_, scratch, scratch_size, &beta,
          out_tensor_desc_, output));
      bool act_done = false;
      if (input2 && input2 != output) {
        // Merge act with residual add unless there is bias.
        addVectors(output, output, (DataType*)input2, N * C * H * W,
                   N * C * H * W, N * C * H * W,
                   use_bias_ ? ACTIVATION_NONE : act_, stream);
        act_done = !use_bias_;
      }
      // Merge act with bias.
      if (use_bias_) {
        if (!nhwc_) {
          // add bias
          addBias_NCHW(output, output, biases, N, C, H, W, act_, stream);
        } else {
          addVectors(output, output, biases, N * C * H * W, N * C * H * W, C,
                     act_, stream);
        }
      } else if (!act_done && act_ != ACTIVATION_NONE) {
        addVectors(output, output, (DataType*)nullptr, N * C * H * W,
                   N * C * H * W, 0, act_, stream);
      }
    }
  }
#else
  else {
    ReportCUDNNErrors(cudnnConvolutionForward(
        cudnn, &alpha, in_tensor_desc_, input, filter_desc_, weights,
        conv_desc_, conv_algo_, scratch, scratch_size,
        (input2 == output) ? &alpha : &beta, out_tensor_desc_, output));
    if (input2 && input2 != output) {
      ReportCUDNNErrors(cudnnAddTensor(cudnn, &alpha, out_tensor_desc_, input2,
                                       &alpha, out_tensor_desc_, output));
    }
    if (use_bias_) {
      ReportCUDNNErrors(cudnnAddTensor(cudnn, &alpha, bias_desc_, biases,
                                       &alpha, out_tensor_desc_, output));
    }
    if (act_ == ACTIVATION_RELU) {
      ReportCUDNNErrors(cudnnActivationForward(cudnn, activation_, &alpha,
                                               out_tensor_desc_, output, &beta,
                                               out_tensor_desc_, output));
    }
    if (act_ != ACTIVATION_RELU && act_ != ACTIVATION_NONE) {
      addVectors(output, output, nullptr, N * C * H * W, N * C * H * W, 0, act_,
                 stream);
      // TODO: check this actually compiles?
    }
  }
#endif
}

template <typename DataType>
ConvLayer<DataType>::~ConvLayer() {
  ReportCUDAErrors(cudaFree(weights));
  ReportCUDAErrors(cudaFree(biases));

  cudnnDestroyFilterDescriptor(filter_desc_);
  cudnnDestroyConvolutionDescriptor(conv_desc_);
  cudnnDestroyTensorDescriptor(bias_desc_);
  cudnnDestroyTensorDescriptor(in_tensor_desc_);
  cudnnDestroyTensorDescriptor(out_tensor_desc_);
  cudnnDestroyActivationDescriptor(activation_);
}
#endif

template <typename DataType>
SELayer<DataType>::SELayer(BaseLayer<DataType>* ip, int fc1Outputs,
                           bool addPrevLayerBias, ActivationFunction activation)
    : BaseLayer<DataType>(ip->GetC(), ip->GetH(), ip->GetW(), ip),
      numFc1Out_(fc1Outputs),
      addPrevLayerBias_(addPrevLayerBias),
      act_(activation) {
  ReportCUDAErrors(cudaMalloc(&w1_, C * numFc1Out_ * sizeof(DataType)));
  ReportCUDAErrors(cudaMalloc(&w2_, 2 * C * numFc1Out_ * sizeof(DataType)));

  if (kUseFusedSELayer && nhwc_) {
    ReportCUDAErrors(cudaMalloc(&w1_t_, C * numFc1Out_ * sizeof(DataType)));
    ReportCUDAErrors(cudaMalloc(&w2_t_, 2 * C * numFc1Out_ * sizeof(DataType)));
  }

  ReportCUDAErrors(cudaMalloc(&b1_, numFc1Out_ * sizeof(DataType)));
  ReportCUDAErrors(cudaMalloc(&b2_, 2 * C * sizeof(DataType)));

  ReportCUDAErrors(cudaMalloc(&bPrev_, C * sizeof(DataType)));
}

template <typename DataType>
SELayer<DataType>::~SELayer() {
  ReportCUDAErrors(cudaFree(w1_));
  ReportCUDAErrors(cudaFree(w2_));
  ReportCUDAErrors(cudaFree(b1_));
  ReportCUDAErrors(cudaFree(b2_));
  ReportCUDAErrors(cudaFree(bPrev_));
}

template <>
void SELayer<float>::LoadWeights(float* w1, float* b1, float* w2, float* b2,
                                 float* prevLayerBias, void* /*scratch*/) {
  const size_t num_weights1 = C * numFc1Out_;
  const size_t weight_size1 = sizeof(float) * num_weights1;

  const size_t weight_size2 = 2 * weight_size1;

  // Weight for the first FC layer.
  ReportCUDAErrors(cudaMemcpy(w1_, w1, weight_size1, cudaMemcpyHostToDevice));

  // Weight for the second FC layer.
  ReportCUDAErrors(cudaMemcpy(w2_, w2, weight_size2, cudaMemcpyHostToDevice));

  // Bias for the first FC layer.
  ReportCUDAErrors(
      cudaMemcpy(b1_, b1, numFc1Out_ * sizeof(float), cudaMemcpyHostToDevice));

  // Bias for the second FC layer.
  ReportCUDAErrors(
      cudaMemcpy(b2_, b2, 2 * C * sizeof(float), cudaMemcpyHostToDevice));

  // Bias for previous layer (Convolution).
  if (prevLayerBias) {
    ReportCUDAErrors(cudaMemcpy(bPrev_, prevLayerBias, C * sizeof(float),
                                cudaMemcpyHostToDevice));
  }
}

void cpuTranspose(float* op, float* ip, int rows, int cols) {
  for (int i = 0; i < rows; i++)
    for (int j = 0; j < cols; j++) op[j * rows + i] = ip[i * cols + j];
}

template <>
void SELayer<half>::LoadWeights(float* w1, float* b1, float* w2, float* b2,
                                float* prevLayerBias, void* scratch) {
  const size_t num_weights1 = C * numFc1Out_;
  size_t weight_size1 = sizeof(float) * num_weights1;

  const size_t num_weights2 = 2 * num_weights1;
  size_t weight_size2 = 2 * weight_size1;

  // Transpose the weight matrices for the fused path.
  std::vector<float> temp(weight_size2);

  // Weight for the first FC layer.
  ReportCUDAErrors(
      cudaMemcpy(scratch, w1, weight_size1, cudaMemcpyHostToDevice));
  copyTypeConverted((half*)w1_, (float*)scratch, (int)num_weights1, 0);
  if (kUseFusedSELayer && nhwc_) {
    // transposed copy for fused SE kernel
    cpuTranspose(temp.data(), w1, numFc1Out_, C);
    ReportCUDAErrors(
        cudaMemcpy(scratch, temp.data(), weight_size1, cudaMemcpyHostToDevice));
    copyTypeConverted((half*)w1_t_, (float*)scratch, (int)num_weights1, 0);
  }

  // Weight for the second FC layer.
  ReportCUDAErrors(
      cudaMemcpy(scratch, w2, weight_size2, cudaMemcpyHostToDevice));
  copyTypeConverted((half*)w2_, (float*)scratch, (int)num_weights2, 0);
  if (kUseFusedSELayer && nhwc_) {
    cpuTranspose(temp.data(), w2, 2 * C, numFc1Out_);
    ReportCUDAErrors(
        cudaMemcpy(scratch, temp.data(), weight_size2, cudaMemcpyHostToDevice));
    copyTypeConverted((half*)w2_t_, (float*)scratch, (int)num_weights2, 0);
  }

  // Bias for the first FC layer.
  ReportCUDAErrors(cudaMemcpy(scratch, b1, numFc1Out_ * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((half*)b1_, (float*)scratch, numFc1Out_, 0);

  // Bias for the second FC layer.
  ReportCUDAErrors(
      cudaMemcpy(scratch, b2, 2 * C * sizeof(float), cudaMemcpyHostToDevice));
  copyTypeConverted((half*)b2_, (float*)scratch, 2 * C, 0);

  // Bias for previous layer (Convolution).
  if (prevLayerBias) {
    ReportCUDAErrors(cudaMemcpy(scratch, prevLayerBias, C * sizeof(float),
                                cudaMemcpyHostToDevice));
    copyTypeConverted((half*)bPrev_, (float*)scratch, C, 0);
  }
}

template <>
void SELayer<float>::Eval(int N, float* output, const float* input,
                          const float* /*input2*/, void* scratch,
                          size_t scratch_size, cudnnHandle_t /*cudnn*/,
                          cublasHandle_t cublas, cudaStream_t stream,
                          float***) {
  // Ping-pong between 'op1' and 'op2' (parts of scratch memory).
  float* op1 = (float*)scratch;
  float* op2 = (float*)scratch + scratch_size / sizeof(float) / 2;

  // 1. Global avg pooling (also adds previous layer bias before computing
  // averages).
  globalAvgPool(N, C, op2, input, bPrev_, false, stream);

  // 2. First fully connected layer.
  float alpha = 1.0f, beta = 0.0f;
  ReportCUBLASErrors(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, numFc1Out_,
                                 N, C, &alpha, w1_, C, op2, C, &beta, op1,
                                 numFc1Out_));
  addVectors(op1, b1_, op1, numFc1Out_ * N, numFc1Out_, numFc1Out_ * N, act_,
             stream);

  // 3. Second fully connected layer.
  ReportCUBLASErrors(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 2 * C, N,
                                 numFc1Out_, &alpha, w2_, numFc1Out_, op1,
                                 numFc1Out_, &beta, op2, 2 * C));
  addVectors(op2, b2_, op2, 2 * C * N, 2 * C, 2 * C * N, ACTIVATION_NONE,
             stream);

  // 4. (Optional prev layer bias add), Global scale, residual add, relu and
  // bias.
  globalScale(N, C, output, input, op2, bPrev_, false, act_, stream);
}

template <>
void SELayer<half>::Eval(int N, half* output, const half* input,
                         const half* input2, void* scratch, size_t scratch_size,
                         cudnnHandle_t /*cudnn*/, cublasHandle_t cublas,
                         cudaStream_t stream, half***) {
  bool se_done = false;
  if (kUseFusedSELayer && nhwc_) {
    se_done = Se_Fp16_NHWC(N, C, numFc1Out_, output, input2, input, w1_t_, b1_,
                           w2_t_, b2_, bPrev_, act_, stream);
  }
  if (!se_done) {
    assert(output == input2);
    // Ping-pong between 'op1' and 'op2' (parts of scratch memory).
    half* op1 = (half*)scratch;
    half* op2 = (half*)scratch + scratch_size / sizeof(half) / 2;

    // 1. Global avg pooling (also adds previous layer bias before computing
    // averages).
    globalAvgPool(N, C, op2, input, bPrev_, nhwc_, stream);

    // 2. First fully connected layer.
    __half_raw one_h{0x3C00};
    __half_raw zero_h{0};
    half alpha = one_h;
    half beta = zero_h;
    ReportCUBLASErrors(cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, numFc1Out_,
                                   N, C, &alpha, w1_, C, op2, C, &beta, op1,
                                   numFc1Out_));
    addVectors(op1, b1_, op1, numFc1Out_ * N, numFc1Out_, numFc1Out_ * N, act_,
               stream);

    // 3. Second fully connected layer.
    ReportCUBLASErrors(cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 2 * C, N,
                                   numFc1Out_, &alpha, w2_, numFc1Out_, op1,
                                   numFc1Out_, &beta, op2, 2 * C));
    addVectors(op2, b2_, op2, 2 * C * N, 2 * C, 2 * C * N, ACTIVATION_NONE,
               stream);

    // 4. (Optional prev layer bias add), Global scale, residual add, relu and
    // bias.
    globalScale(N, C, output, input, op2, bPrev_, nhwc_, act_, stream);
  }
}

template <typename DataType>
FCLayer<DataType>::FCLayer(BaseLayer<DataType>* ip, int C, int H, int W,
                           bool bias, ActivationFunction activation)
    : BaseLayer<DataType>(C, H, W, ip), use_bias_(bias), act_(activation) {
  const size_t weight_size =
      sizeof(DataType) * C * H * W * ip->GetC() * ip->GetH() * ip->GetW();
  const size_t bias_size = sizeof(DataType) * C * H * W;
  ReportCUDAErrors(cudaMalloc(&weights_, weight_size));
  if (use_bias_) {
    ReportCUDAErrors(cudaMalloc(&biases_, bias_size));
  } else {
    biases_ = nullptr;
  }
}

template <>
void FCLayer<half>::LoadWeights(float* cpuWeight, float* cpuBias,
                                void* scratch) {
  const size_t num_weights =
      C * H * W * input_->GetC() * input_->GetH() * input_->GetW();
  const size_t weight_size = sizeof(float) * num_weights;
  const size_t num_biases = C * H * W;
  const size_t bias_size = sizeof(float) * num_biases;

  // also need to convert from fp32 to fp16
  assert(scratch);
  ReportCUDAErrors(
      cudaMemcpy(scratch, cpuWeight, weight_size, cudaMemcpyHostToDevice));

  if (nhwc_) {
    convertNCHWtoNHWC((half*)weights_, (float*)scratch, (int)num_biases,
                      input_->GetC(), (int)num_biases, input_->GetC(),
                      input_->GetH(), input_->GetW(), 0);
  } else {
    copyTypeConverted((half*)weights_, (float*)scratch, (int)num_weights, 0);
  }

  if (cpuBias) {
    ReportCUDAErrors(
        cudaMemcpy(scratch, cpuBias, bias_size, cudaMemcpyHostToDevice));
    copyTypeConverted((half*)biases_, (float*)scratch, (int)num_biases, 0);
  }
}

template <>
void FCLayer<float>::LoadWeights(float* cpuWeight, float* cpuBias,
                                 void* /*scratch*/) {
  const size_t num_weights =
      C * H * W * input_->GetC() * input_->GetH() * input_->GetW();
  const size_t weight_size = sizeof(float) * num_weights;
  const size_t num_biases = C * H * W;
  const size_t bias_size = sizeof(float) * num_biases;

  ReportCUDAErrors(
      cudaMemcpy(weights_, cpuWeight, weight_size, cudaMemcpyHostToDevice));
  if (use_bias_) {
    ReportCUDAErrors(
        cudaMemcpy(biases_, cpuBias, bias_size, cudaMemcpyHostToDevice));
  }
}

template <>
void FCLayer<half>::Eval(int N, half* output_tensor, const half* input_tensor,
                         const half* /*input2*/, void* /*scratch*/,
                         size_t /*scratch_size*/, cudnnHandle_t /*cudnn*/,
                         cublasHandle_t cublas, cudaStream_t stream, half***) {
  const int num_outputs = C * H * W;
  const int num_inputs = input_->GetC() * input_->GetH() * input_->GetW();

  // half alpha = float2half_rn(1.0f), beta = float2half_rn(0.0f);
  const __half_raw one_h{0x3C00};
  const __half_raw zero_h{0};
  half alpha = one_h;
  half beta = zero_h;
  ReportCUBLASErrors(cublasHgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                                 N, num_inputs, &alpha, weights_, num_inputs,
                                 input_tensor, num_inputs, &beta, output_tensor,
                                 num_outputs));

  if (use_bias_ || (act_ != ACTIVATION_NONE)) {
    addVectors(output_tensor, biases_, output_tensor, num_outputs * N,
               num_outputs, num_outputs * N, act_, stream);
  }
}

template <>
void FCLayer<float>::Eval(int N, float* output_tensor,
                          const float* input_tensor, const float* /*input2*/,
                          void* /*scratch*/, size_t /*scratch_size*/,
                          cudnnHandle_t /*cudnn*/, cublasHandle_t cublas,
                          cudaStream_t stream, float***) {
  const int num_outputs = C * H * W;
  const int num_inputs = input_->GetC() * input_->GetH() * input_->GetW();

  float alpha = 1.0f, beta = 0.0f;
  ReportCUBLASErrors(cublasSgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                                 N, num_inputs, &alpha, weights_, num_inputs,
                                 input_tensor, num_inputs, &beta, output_tensor,
                                 num_outputs));

  if (use_bias_ || (act_ != ACTIVATION_NONE)) {
    addVectors(output_tensor, biases_, output_tensor, num_outputs * N,
               num_outputs, num_outputs * N, act_, stream);
  }
}

template <typename DataType>
FCLayer<DataType>::~FCLayer() {
  ReportCUDAErrors(cudaFree(weights_));
  ReportCUDAErrors(cudaFree(biases_));
}

template <typename DataType>
PolicyMapLayer<DataType>::PolicyMapLayer(BaseLayer<DataType>* ip, int C, int H,
                                         int W, int usedSize, bool attention)
    : BaseLayer<DataType>(C, H, W, ip),
      used_size_(usedSize),
      attention_map_(attention) {
  size_t weight_size = sizeof(short) * this->input_->GetC() * 64;
  if (attention) weight_size = sizeof(short) * usedSize;
  ReportCUDAErrors(cudaMalloc(&weights_, weight_size));
}

template <typename DataType>
void PolicyMapLayer<DataType>::LoadWeights(const short* cpuWeight,
                                           void* /*scratch*/) {
  size_t weight_size = sizeof(short) * used_size_;

  if (nhwc_ && !attention_map_) {
    // convert CHW to HWC
    int C = used_size_ / 64;
    int Cin = this->input_->GetC();

    // C is the no. of channels actually used (typically 73).
    // Cin the the no. of channels in previous layer (padded up to 80).
    // Weights of this layer is a mapping to select which output index of the
    // policy vector (1858 elements) maps to every element of input
    // tensor (assuming NCHW layout). Note that there are 73x64 valid inputs
    // (80x64 taking padding), and only 1858 outputs so the mapping isn't
    // one to one. Only few of the indices point to valid index in policy
    // vector. Invalid entries are set to -1.

    // In fp16 mode, the tensor layout is NHWC so the weights need to be
    // adjusted to make them work as intended.

    // This is how the original weights looks like (CHW layout):
    /*
               HW (64)
       ----|-------------|
           |             |
           |             |
    C (73) |             |
           |             |
           |             |
       ------------------|   Cin (80)
           |  padding    |
           |-------------|
    */
    // The padding is not part of the weights provided (used_size_ is 73 x 64).
    //
    // The weights converted to HWC looks like this
    /*
                 C (73)
            |-------------|---|
            |             | P |
            |             | a |
    HW (64) |             | d |
            |             |   |
            |             |   |
            |-----------------|
                     Cin (80)
    */
    // In HWC, because the padding is now part of each row
    // we need to increase the size of weights to account
    // for it.
    // The pad elements point to -1 (invalid output index) and the
    // same kernel works for both HWC and CHW layouts after used_size_
    // is updated to include padding (80x64).

    used_size_ = Cin * 64;
    std::vector<short> convertedWeights(used_size_);

    for (int hw = 0; hw < 64; hw++)
      for (int c = 0; c < Cin; c++) {
        if (c < C)
          convertedWeights[hw * Cin + c] = cpuWeight[c * 64 + hw];
        else
          convertedWeights[hw * Cin + c] = -1;
      }
    ReportCUDAErrors(cudaMemcpy(weights_, convertedWeights.data(),
                                used_size_ * sizeof(short),
                                cudaMemcpyHostToDevice));
  } else {
    ReportCUDAErrors(
        cudaMemcpy(weights_, cpuWeight, weight_size, cudaMemcpyHostToDevice));
  }
}

template <typename DataType>
void PolicyMapLayer<DataType>::Eval(int N, DataType* output_tensor,
                                    const DataType* input_tensor,
                                    const DataType* /*input2*/,
                                    void* /*scratch*/, size_t /*scratch_size*/,
                                    cudnnHandle_t /*cudnn*/,
                                    cublasHandle_t /*cublas*/,
                                    cudaStream_t stream, DataType***) {
  int inputSize =
      this->input_->GetC() * this->input_->GetH() * this->input_->GetW();
  if (attention_map_) inputSize = used_size_;
  int outputSize = this->C * this->H * this->W;
  PolicyMap(N, output_tensor, input_tensor, weights_, inputSize, used_size_,
            outputSize, stream);
}

template <typename DataType>
PolicyMapLayer<DataType>::~PolicyMapLayer() {
  ReportCUDAErrors(cudaFree(weights_));
}

template <typename DataType>
FusedWinogradConvSELayer<DataType>::FusedWinogradConvSELayer(
    BaseLayer<DataType>* ip, int C, int H, int W, int Cin,
    ActivationFunction activation, bool bias, bool skip_add, bool se, int se_k,
    bool use_gemm_ex, bool op_nhcw)
    : BaseLayer<DataType>(C, H, W, ip, false, use_gemm_ex),
      c_input_(Cin),
      act_(activation),
      use_bias_(bias),
      skip_add_(skip_add),
      has_se_(se),
      se_k_(se_k),
      op_nhcw_(op_nhcw) {
  if (act_ != ACTIVATION_RELU && act_ != ACTIVATION_MISH &&
      act_ != ACTIVATION_NONE) {
    throw Exception("Unsupported activation for fused winograd conv SE layer.");
  }
  // Allocate memory for weights (filter tensor) and biases.
  const size_t weight_size = sizeof(DataType) * c_input_ * C * 3 * 3;

  if (use_bias_) {
    const size_t bias_size = sizeof(DataType) * C;
    ReportCUDAErrors(cudaMalloc(&biases_, bias_size));
  }

  // 6x6 transformed filter size, for 3x3 convolution
  ReportCUDAErrors(cudaMalloc(&transformed_weights_, weight_size * 4));

  if (has_se_) {
    const size_t num_weights1 = C * se_k_;
    const size_t num_weights2 = num_weights1 * 2;
    const size_t num_biases1 = se_k_;
    const size_t num_biases2 = 2 * C;

    const size_t weight_size1 = sizeof(DataType) * num_weights1;
    const size_t weight_size2 = sizeof(DataType) * num_weights2;
    const size_t biases_size1 = sizeof(DataType) * num_biases1;
    const size_t biases_size2 = sizeof(DataType) * num_biases2;

    ReportCUDAErrors(cudaMalloc(&w1_, weight_size1));
    ReportCUDAErrors(cudaMalloc(&w2_, weight_size2));
    ReportCUDAErrors(cudaMalloc(&b1_, biases_size1));
    ReportCUDAErrors(cudaMalloc(&b2_, biases_size2));
  }
}

template <typename DataType>
void FusedWinogradConvSELayer<DataType>::LoadWeights(float* pfilter,
                                                     float* pBias,
                                                     void* scratch) {
  const size_t weight_size = sizeof(float) * c_input_ * C * 3 * 3;
  const size_t bias_size = sizeof(float) * C;

  // Store untransformed weights in scratch.
  const DataType* weights = (DataType*)scratch + weight_size + bias_size;

  // first copy from CPU memory to scratch space in GPU memory
  // and then do the type conversion using a kernel
  assert(scratch);
  ReportCUDAErrors(
      cudaMemcpy(scratch, pfilter, weight_size, cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)weights, (float*)scratch, C * c_input_ * 3 * 3,
                    0);

  if (pBias) {
    ReportCUDAErrors(
        cudaMemcpy(scratch, pBias, bias_size, cudaMemcpyHostToDevice));
    copyTypeConverted((DataType*)biases_, (float*)scratch, C, 0);
  }

  // run winograd transform kernel for the filter
  FilterTransform(C, c_input_, transformed_weights_, weights, 0);
}

// TODO: Do this on the GPU to improve network load time!
static inline void CpuTranspose(float* op, float* ip, size_t rows,
                                size_t cols) {
  for (size_t i = 0; i < rows; i++)
    for (size_t j = 0; j < cols; j++) op[j * rows + i] = ip[i * cols + j];
}

template <typename DataType>
void FusedWinogradConvSELayer<DataType>::LoadSEWeights(float* w1, float* b1,
                                                       float* w2, float* b2,
                                                       void* scratch) {
  const size_t num_weights1 = C * se_k_;
  const size_t num_weights2 = num_weights1 * 2;
  const size_t num_biases1 = se_k_;
  const size_t num_biases2 = 2 * C;

  // The shader uses transposed weight matrices.
  std::vector<float> temp_transposed(num_weights2);

  CpuTranspose(temp_transposed.data(), w1, se_k_, C);
  ReportCUDAErrors(cudaMemcpy(scratch, temp_transposed.data(),
                              num_weights1 * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)w1_, (float*)scratch, (int)num_weights1, 0);

  CpuTranspose(temp_transposed.data(), w2, 2 * C, se_k_);
  ReportCUDAErrors(cudaMemcpy(scratch, temp_transposed.data(),
                              num_weights2 * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)w2_, (float*)scratch, (int)num_weights2, 0);

  ReportCUDAErrors(cudaMemcpy(scratch, b1, num_biases1 * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)b1_, (float*)scratch, (int)num_biases1, 0);

  ReportCUDAErrors(cudaMemcpy(scratch, b2, num_biases2 * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)b2_, (float*)scratch, (int)num_biases2, 0);
}

template <>
void BaseLayer<half>::cublasRowMajorMatrixMul(const half* A, const half* B,
                                              half* Out, int M, int N, int K,
                                              int batchSize,
                                              cublasHandle_t cublas) {
  // Need to initialize 1.0 and 0.0 as hexadecimal for fp16 because typecasting
  // float to half type doesn't work before CUDA 10.0
  __half_raw one_h{0x3C00};
  __half_raw zero_h{0};
  half halfOne = one_h;
  half halfZero = zero_h;

  // dimensions of matrix A = M x K
  // dimensions of matrix B = K x N
  // dimensions of output   = M x N

  // cublas supports only col major output
  // to multiply row major matrices, use the trick below
  ReportCUBLASErrors(cublasGemmStridedBatchedEx(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &halfOne, B, CUDA_R_16F, N,
      N * K, A, CUDA_R_16F, K, K * M, &halfZero, Out, CUDA_R_16F, N, N * M,
      batchSize, CUDA_R_16F, CUBLAS_GEMM_DEFAULT));
}

template <>
void BaseLayer<float>::cublasRowMajorMatrixMul(const float* A, const float* B,
                                               float* Out, int M, int N, int K,
                                               int batchSize,
                                               cublasHandle_t cublas) {
  float floatOne = 1.0f;
  float floatZero = 0.0f;
  if (use_gemm_ex_)
    ReportCUBLASErrors(cublasGemmStridedBatchedEx(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &floatOne, B, CUDA_R_32F, N,
        N * K, A, CUDA_R_32F, K, K * M, &floatZero, Out, CUDA_R_32F, N, N * M,
        batchSize, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
  else
    // Much slower on RTX 2060.. why? Maybe a cublas bug :-/
    ReportCUBLASErrors(cublasSgemmStridedBatched(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &floatOne, B, N, N * K, A, K,
        K * M, &floatZero, Out, N, N * M, batchSize));
}

template <typename DataType>
void FusedWinogradConvSELayer<DataType>::Eval(
    int N, DataType* output, const DataType* input, const DataType* input2,
    void* scratch, size_t scratch_size, cudnnHandle_t /*cudnn*/,
    cublasHandle_t cublas, cudaStream_t stream, DataType***) {
  // Split the scratch space into two parts - use first part for holding
  // transformed input and second part for transformed output.
  DataType* transformed_input = (DataType*)scratch;
  DataType* transformed_output =
      transformed_input + scratch_size / (2 * sizeof(DataType));

  InputTransform<DataType, false>(N, c_input_, transformed_input, input,
                                  stream);
  BaseLayer<DataType>::cublasRowMajorMatrixMul(
      transformed_input, transformed_weights_, transformed_output, N * 4, C,
      c_input_, 36, cublas);

  if (act_ == ACTIVATION_NONE) {
    if (!has_se_ && use_bias_ && !skip_add_)
      OutputTransform<DataType, false, ACTIVATION_NONE, true, false, false,
                      false>(N, C, 0, output, transformed_output, nullptr,
                             biases_, nullptr, nullptr, nullptr, nullptr,
                             stream);
    else
      throw Exception("unsupported network type!");
  } else if (act_ == ACTIVATION_RELU) {
    if (has_se_ && use_bias_ && skip_add_)
      OutputTransform<DataType, true, ACTIVATION_RELU, true, true, false,
                      false>(N, C, se_k_, output, transformed_output, input2,
                             biases_, w1_, b1_, w2_, b2_, stream);
    else if (!has_se_ && use_bias_ && !skip_add_) {
      if (op_nhcw_)
        OutputTransform<DataType, false, ACTIVATION_RELU, true, false, false,
                        true>(N, C, 0, output, transformed_output, nullptr,
                              biases_, nullptr, nullptr, nullptr, nullptr,
                              stream);
      else
        OutputTransform<DataType, false, ACTIVATION_RELU, true, false, false,
                        false>(N, C, 0, output, transformed_output, nullptr,
                               biases_, nullptr, nullptr, nullptr, nullptr,
                               stream);
    } else if (!has_se_ && use_bias_ && skip_add_)
      OutputTransform<DataType, false, ACTIVATION_RELU, true, true, false,
                      false>(N, C, 0, output, transformed_output, input2,
                             biases_, nullptr, nullptr, nullptr, nullptr,
                             stream);
    else
      throw Exception("unsupported network type!");
  } else if (act_ == ACTIVATION_MISH) {
    if (has_se_ && use_bias_ && skip_add_)
      OutputTransform<DataType, true, ACTIVATION_MISH, true, true, false,
                      false>(N, C, se_k_, output, transformed_output, input2,
                             biases_, w1_, b1_, w2_, b2_, stream);
    else if (!has_se_ && use_bias_ && !skip_add_) {
      if (op_nhcw_)
        OutputTransform<DataType, false, ACTIVATION_MISH, true, false, false,
                        true>(N, C, 0, output, transformed_output, nullptr,
                              biases_, nullptr, nullptr, nullptr, nullptr,
                              stream);
      else
        OutputTransform<DataType, false, ACTIVATION_MISH, true, false, false,
                        false>(N, C, 0, output, transformed_output, nullptr,
                               biases_, nullptr, nullptr, nullptr, nullptr,
                               stream);
    } else if (!has_se_ && use_bias_ && skip_add_)
      OutputTransform<DataType, false, ACTIVATION_MISH, true, true, false,
                      false>(N, C, 0, output, transformed_output, input2,
                             biases_, nullptr, nullptr, nullptr, nullptr,
                             stream);
    else
      throw Exception("unsupported network type!");
  } else
    throw Exception("unsupported network type!");
}

template <typename DataType>
FusedWinogradConvSELayer<DataType>::~FusedWinogradConvSELayer() {
  ReportCUDAErrors(cudaFree(transformed_weights_));
  if (use_bias_) ReportCUDAErrors(cudaFree(biases_));
  if (has_se_) {
    ReportCUDAErrors(cudaFree(w1_));
    ReportCUDAErrors(cudaFree(w2_));
    ReportCUDAErrors(cudaFree(b1_));
    ReportCUDAErrors(cudaFree(b2_));
  }
}

template <typename DataType>
Conv1Layer<DataType>::Conv1Layer(BaseLayer<DataType>* ip, int C, int H, int W,
                                 int Cin, ActivationFunction activation,
                                 bool bias, bool use_gemm_ex)
    : BaseLayer<DataType>(C, H, W, ip, false, use_gemm_ex),
      c_input_(Cin),
      act_(activation),
      use_bias_(bias) {
  // Allocate memory for weights (filter tensor) and biases.
  const size_t weight_size = sizeof(DataType) * c_input_ * C * 1 * 1;
  ReportCUDAErrors(cudaMalloc(&weights_, weight_size));

  if (use_bias_) {
    const size_t bias_size = sizeof(DataType) * C;
    ReportCUDAErrors(cudaMalloc(&biases_, bias_size));
  }
}

template <typename DataType>
void Conv1Layer<DataType>::LoadWeights(float* pfilter, float* pBias,
                                       void* scratch) {
  const size_t weight_size = sizeof(float) * c_input_ * C * 1 * 1;
  const size_t bias_size = sizeof(float) * C;

  // first copy from CPU memory to scratch space in GPU memory
  // and then do the type conversion using a kernel
  assert(scratch);
  ReportCUDAErrors(
      cudaMemcpy(scratch, pfilter, weight_size, cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)weights_, (float*)scratch, C * c_input_ * 1 * 1,
                    0);

  if (pBias) {
    ReportCUDAErrors(
        cudaMemcpy(scratch, pBias, bias_size, cudaMemcpyHostToDevice));
    copyTypeConverted((DataType*)biases_, (float*)scratch, C, 0);
  }
}
template <>
void Conv1Layer<half>::cublasSpecialMatrixMul(const half* A, const half* B,
                                              half* Out, int M, int N, int K,
                                              int batchSize,
                                              cublasHandle_t cublas) {
  // Need to initialize 1.0 and 0.0 as hexadecimal for fp16 because typecasting
  // float to half type doesn't work before CUDA 10.0
  __half_raw one_h{0x3C00};
  __half_raw zero_h{0};
  half halfOne = one_h;
  half halfZero = zero_h;

  // dimensions of matrix A = M x K
  // dimensions of matrix B = K x N
  // dimensions of output   = M x N

  // cublas supports only col major output
  // to multiply row major matrices, use the trick below
  // NOTE strideB set to 0 below!
  ReportCUBLASErrors(cublasGemmStridedBatchedEx(
      cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &halfOne, B, CUDA_R_16F, N,
      N * K, A, CUDA_R_16F, K, 0, &halfZero, Out, CUDA_R_16F, N, N * M,
      batchSize, CUDA_R_16F, CUBLAS_GEMM_DEFAULT));
}

template <>
void Conv1Layer<float>::cublasSpecialMatrixMul(const float* A, const float* B,
                                               float* Out, int M, int N, int K,
                                               int batchSize,
                                               cublasHandle_t cublas) {
  float floatOne = 1.0f;
  float floatZero = 0.0f;

  // NOTE strideB set to 0 below!
  if (use_gemm_ex_)
    ReportCUBLASErrors(cublasGemmStridedBatchedEx(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &floatOne, B, CUDA_R_32F, N,
        N * K, A, CUDA_R_32F, K, 0, &floatZero, Out, CUDA_R_32F, N, N * M,
        batchSize, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
  else
    // Much slower on RTX 2060.. why? Maybe a cublas bug :-/
    ReportCUBLASErrors(cublasSgemmStridedBatched(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &floatOne, B, N, N * K, A, K,
        0, &floatZero, Out, N, N * M, batchSize));
}

template <typename DataType>
void Conv1Layer<DataType>::Eval(int N, DataType* output, const DataType* input,
                                const DataType* /*input2*/, void* /*scratch*/,
                                size_t /*scratch_size*/,
                                cudnnHandle_t /*cudnn*/, cublasHandle_t cublas,
                                cudaStream_t stream, DataType***) {
  cublasSpecialMatrixMul(weights_, input, output, C, H * W, c_input_, N,
                         cublas);

  if (use_bias_)
    addBias_NCHW(output, output, biases_, N, C, H, W, act_, stream);
  else if (act_ != ACTIVATION_NONE)
    addVectors(output, output, (DataType*)nullptr, N * C * H * W, N * C * H * W,
               0, act_, stream);
}

template <typename DataType>
Conv1Layer<DataType>::~Conv1Layer() {
  ReportCUDAErrors(cudaFree(weights_));
  if (use_bias_) ReportCUDAErrors(cudaFree(biases_));
}

template <typename DataType>
ResidualBlock<DataType>::ResidualBlock(BaseLayer<DataType>* ip, int C, bool se,
                                       int se_k, bool use_gemm_ex, bool first,

                                       bool last, ActivationFunction activation,
                                       int shared_mem_size)
    : BaseLayer<DataType>(C, 8, 8, ip, ip->isNHWC(), use_gemm_ex),
      has_se_(se),
      se_k_(se_k),
      c_input_(C),
      first_block_(first),
      last_block_(last),
      shared_mem_size_(shared_mem_size),
      act_(activation) {
  if (act_ != ACTIVATION_RELU && act_ != ACTIVATION_MISH) {
    throw Exception("Unsupported activation for residual block.");
  }
  // Allocate memory for weights (filter tensor) and biases.
  const size_t weight_size = sizeof(DataType) * C * C * 3 * 3;

  const size_t bias_size = sizeof(DataType) * C;
  ReportCUDAErrors(cudaMalloc(&biases0_, bias_size));
  ReportCUDAErrors(cudaMalloc(&biases1_, bias_size));

  // 6x6 transformed filter size, for 3x3 convolution
  ReportCUDAErrors(cudaMalloc(&transformed_weights0_, weight_size * 4));
  ReportCUDAErrors(cudaMalloc(&transformed_weights1_, weight_size * 4));

  if (has_se_) {
    const size_t num_weights1 = C * se_k_;
    const size_t num_weights2 = num_weights1 * 2;
    const size_t num_biases1 = se_k_;
    const size_t num_biases2 = 2 * C;

    const size_t weight_size1 = sizeof(DataType) * num_weights1;
    const size_t weight_size2 = sizeof(DataType) * num_weights2;
    const size_t biases_size1 = sizeof(DataType) * num_biases1;
    const size_t biases_size2 = sizeof(DataType) * num_biases2;

    ReportCUDAErrors(cudaMalloc(&w1_, weight_size1));
    ReportCUDAErrors(cudaMalloc(&w2_, weight_size2));
    ReportCUDAErrors(cudaMalloc(&b1_, biases_size1));
    ReportCUDAErrors(cudaMalloc(&b2_, biases_size2));
  }
}

template <typename DataType>
void ResidualBlock<DataType>::LoadWeights0(float* pfilter, float* pBias,
                                           void* scratch) {
  const size_t weight_size = sizeof(float) * c_input_ * C * 3 * 3;
  const size_t bias_size = sizeof(float) * C;

  // Store untransformed weights in scratch.
  const DataType* weights = (DataType*)scratch + weight_size;

  // first copy from CPU memory to scratch space in GPU memory
  // and then do the type conversion using a kernel
  assert(scratch);
  ReportCUDAErrors(
      cudaMemcpy(scratch, pfilter, weight_size, cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)weights, (float*)scratch, C * c_input_ * 3 * 3,
                    0);

  if (pBias) {
    ReportCUDAErrors(
        cudaMemcpy(scratch, pBias, bias_size, cudaMemcpyHostToDevice));
    copyTypeConverted((DataType*)biases0_, (float*)scratch, C, 0);
  }

  // run winograd transform kernel for the filter
  FilterTransform(C, c_input_, transformed_weights0_, weights, 0);
}

template <typename DataType>
void ResidualBlock<DataType>::LoadWeights1(float* pfilter, float* pBias,
                                           void* scratch) {
  const size_t weight_size = sizeof(float) * C * C * 3 * 3;
  const size_t bias_size = sizeof(float) * C;

  // Store untransformed weights in scratch.
  const DataType* weights = (DataType*)scratch + weight_size;

  // first copy from CPU memory to scratch space in GPU memory
  // and then do the type conversion using a kernel
  assert(scratch);
  ReportCUDAErrors(
      cudaMemcpy(scratch, pfilter, weight_size, cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)weights, (float*)scratch, C * C * 3 * 3, 0);

  if (pBias) {
    ReportCUDAErrors(
        cudaMemcpy(scratch, pBias, bias_size, cudaMemcpyHostToDevice));
    copyTypeConverted((DataType*)biases1_, (float*)scratch, C, 0);
  }

  // run winograd transform kernel for the filter
  FilterTransform(C, C, transformed_weights1_, weights, 0);
}

template <typename DataType>
void ResidualBlock<DataType>::LoadSEWeights(float* w1, float* b1, float* w2,
                                            float* b2, void* scratch) {
  const size_t num_weights1 = C * se_k_;
  const size_t num_weights2 = num_weights1 * 2;
  const size_t num_biases1 = se_k_;
  const size_t num_biases2 = 2 * C;

  // The shader uses transposed weight matrices.
  std::vector<float> temp_transposed(num_weights2);

  CpuTranspose(temp_transposed.data(), w1, se_k_, C);
  ReportCUDAErrors(cudaMemcpy(scratch, temp_transposed.data(),
                              num_weights1 * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)w1_, (float*)scratch, (int)num_weights1, 0);

  CpuTranspose(temp_transposed.data(), w2, 2 * C, se_k_);
  ReportCUDAErrors(cudaMemcpy(scratch, temp_transposed.data(),
                              num_weights2 * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)w2_, (float*)scratch, (int)num_weights2, 0);

  ReportCUDAErrors(cudaMemcpy(scratch, b1, num_biases1 * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)b1_, (float*)scratch, (int)num_biases1, 0);

  ReportCUDAErrors(cudaMemcpy(scratch, b2, num_biases2 * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)b2_, (float*)scratch, (int)num_biases2, 0);
}

template <typename DataType>
void ResidualBlock<DataType>::Eval(int N, DataType* output,
                                   const DataType* input,
                                   const DataType* /*input2*/, void* scratch,
                                   size_t scratch_size, cudnnHandle_t /*cudnn*/,
                                   cublasHandle_t cublas, cudaStream_t stream,
                                   DataType***) {
  // normally:
  // - "output" initially contains the transformed input,
  //    and after this layer, it contains the transformed input for next layer
  // - "input" contains the original/untransformed input
  // special cases:
  //   - for first_block_, input is real input (untransformed)
  //   - for last_block_, output is the final output of this block
  //   (untransformed)

  // Split the scratch space into two parts - use first part for holding
  // transformed input and second part for transformed output.
  DataType* transformed_input;
  DataType* transformed_output;
  if (!scratch) {
    // Caller wants us to sub-allocate all memory we need from "output" tensor.
    transformed_input = output;  // This is true in normal cases too!
    transformed_output = transformed_input + (N * C * 8 * 8 * 36 / 16);
  } else {
    transformed_input = (DataType*)scratch;
    transformed_output =
        transformed_input + scratch_size / (2 * sizeof(DataType));
  }

  if (first_block_) {
    InputTransform<DataType, true>(N, c_input_, transformed_input, input,
                                   stream);
    BaseLayer<DataType>::cublasRowMajorMatrixMul(
        transformed_input, transformed_weights0_, transformed_output, N * 4, C,
        c_input_, 36, cublas);
  } else {
    BaseLayer<DataType>::cublasRowMajorMatrixMul(output, transformed_weights0_,
                                                 transformed_output, N * 4, C,
                                                 c_input_, 36, cublas);
  }

  if (act_ == ACTIVATION_RELU) {
    OutputInputTransform<DataType, false, ACTIVATION_RELU, true, false>(
        N, C, 0, transformed_input, transformed_output, nullptr, biases0_,
        nullptr, nullptr, nullptr, nullptr, stream);
  } else if (act_ == ACTIVATION_MISH) {
    OutputInputTransform<DataType, false, ACTIVATION_MISH, true, false>(
        N, C, 0, transformed_input, transformed_output, nullptr, biases0_,
        nullptr, nullptr, nullptr, nullptr, stream);
  }
  // "transformed_input" tensor now contains transformed input for the next
  // convolution

  BaseLayer<DataType>::cublasRowMajorMatrixMul(
      transformed_input, transformed_weights1_, transformed_output, N * 4, C, C,
      36, cublas);

  const bool fp16 = std::is_same<half, DataType>::value;
  bool allowFusing =
      (C <= kMaxResBlockFusingChannels) ||
      (fp16 && (shared_mem_size_ >= kMaxResBlockFusingSeFp16AmpereSmem) &&
       (C <= kMaxResBlockFusingSeKFp16Ampere));

  if (act_ == ACTIVATION_RELU) {
    if (last_block_) {
      if (has_se_)
        OutputTransform<DataType, true, ACTIVATION_RELU, true, true, true,
                        false>(N, C, se_k_, output, transformed_output, input,
                               biases1_, w1_, b1_, w2_, b2_, stream);
      else
        OutputTransform<DataType, false, ACTIVATION_RELU, true, true, true,
                        false>(N, C, se_k_, output, transformed_output, input,
                               biases1_, w1_, b1_, w2_, b2_, stream);
    } else {
      if (has_se_) {
        if (allowFusing) {
          OutputInputTransform<DataType, true, ACTIVATION_RELU, true, true>(
              N, C, se_k_, output, transformed_output, input, biases1_, w1_,
              b1_, w2_, b2_, stream);
        } else {
          OutputTransform<DataType, true, ACTIVATION_RELU, true, true, true,
                          true>(N, C, se_k_, (DataType*)input,
                                transformed_output, input, biases1_, w1_, b1_,
                                w2_, b2_, stream);
          InputTransform<DataType, true>(N, C, output, (DataType*)input,
                                         stream);
        }
      } else
        OutputInputTransform<DataType, false, ACTIVATION_RELU, true, true>(
            N, C, se_k_, output, transformed_output, input, biases1_, w1_, b1_,
            w2_, b2_, stream);
    }
  } else if (act_ == ACTIVATION_MISH) {
    if (last_block_) {
      if (has_se_)
        OutputTransform<DataType, true, ACTIVATION_MISH, true, true, true,
                        false>(N, C, se_k_, output, transformed_output, input,
                               biases1_, w1_, b1_, w2_, b2_, stream);
      else
        OutputTransform<DataType, false, ACTIVATION_MISH, true, true, true,
                        false>(N, C, se_k_, output, transformed_output, input,
                               biases1_, w1_, b1_, w2_, b2_, stream);
    } else {
      if (has_se_) {
        if (allowFusing) {
          OutputInputTransform<DataType, true, ACTIVATION_MISH, true, true>(
              N, C, se_k_, output, transformed_output, input, biases1_, w1_,
              b1_, w2_, b2_, stream);
        } else {
          OutputTransform<DataType, true, ACTIVATION_MISH, true, true, true,
                          true>(N, C, se_k_, (DataType*)input,
                                transformed_output, input, biases1_, w1_, b1_,
                                w2_, b2_, stream);
          InputTransform<DataType, true>(N, C, output, (DataType*)input,
                                         stream);
        }
      } else
        OutputInputTransform<DataType, false, ACTIVATION_MISH, true, true>(
            N, C, se_k_, output, transformed_output, input, biases1_, w1_, b1_,
            w2_, b2_, stream);
    }
  }
  // "output" tensor now contains transformed input for the next
  // convolution
}

template <typename DataType>
ResidualBlock<DataType>::~ResidualBlock() {
  ReportCUDAErrors(cudaFree(transformed_weights0_));
  ReportCUDAErrors(cudaFree(biases0_));
  ReportCUDAErrors(cudaFree(transformed_weights1_));
  ReportCUDAErrors(cudaFree(biases1_));
  if (has_se_) {
    ReportCUDAErrors(cudaFree(w1_));
    ReportCUDAErrors(cudaFree(w2_));
    ReportCUDAErrors(cudaFree(b1_));
    ReportCUDAErrors(cudaFree(b2_));
  }
}

template <typename DataType>
void allocAndUpload(DataType** gpu_dest, std::vector<float> cpu_src,
                    void* scratch) {
  size_t size = cpu_src.size() * sizeof(DataType);
  if (size == 0) {
    *gpu_dest = nullptr;
    return;
  }
  // Arena-aware allocation: when a WeightArenaScope is active (i.e.,
  // tl_weight_arena is non-null), allocate the weight slot from the
  // arena's pre-allocated chunk(s) instead of calling cudaMalloc.  This
  // shaves ~1-2 seconds from network startup at 512x60.  When no arena
  // is set, falls back to the existing cudaMalloc path (which keeps
  // every existing destructor's cudaFree(weight_buffer) call valid).
  //
  // CRITICAL: layer destructors must NOT cudaFree(*gpu_dest) when this
  // pointer was arena-allocated, because the arena owns the chunks.
  // Phase B.3 will handle this: each layer that calls allocAndUpload
  // tracks whether its construction was arena-backed (via a member
  // flag) and skips per-buffer cudaFree at destruction.  Until that
  // lands, the WeightArenaScope is never entered → tl_weight_arena
  // stays null → the cudaMalloc path is used for safety.
  if (tl_weight_arena != nullptr) {
    *gpu_dest = static_cast<DataType*>(tl_weight_arena->Allocate(size));
  } else {
    ReportCUDAErrors(cudaMalloc(gpu_dest, size));
  }
  ReportCUDAErrors(cudaMemcpy(scratch, &cpu_src[0],
                              cpu_src.size() * sizeof(float),
                              cudaMemcpyHostToDevice));
  copyTypeConverted((DataType*)(*gpu_dest), (float*)scratch,
                    (int)cpu_src.size(), 0);
}

template <typename DataType>
AttentionPolicyHead<DataType>::AttentionPolicyHead(
    BaseLayer<DataType>* ip, const MultiHeadWeights::PolicyHead& weights,
    void* scratch, bool attention_body, ActivationFunction act,
    int max_batch_size, bool use_gemm_ex)
    : BaseLayer<DataType>(64 * 64 + 24 * 8, 1, 1, ip),
      attention_body_(attention_body),
      // Old networks without attention body (e.g. T79) use hardcoded SELU
      // activations.
      act_(attention_body ? act : ACTIVATION_SELU) {
  embedding_op_size_ = weights.ip_pol_b.size();
  wq_op_size_ = weights.ip2_pol_b.size();
  wk_op_size_ = weights.ip3_pol_b.size();

  encoder_heads_ = weights.pol_encoder_head_count;
  policy_d_model_ = wq_op_size_;

  allocAndUpload<DataType>(&ip_pol_w_, weights.ip_pol_w, scratch);
  allocAndUpload<DataType>(&ip_pol_b_, weights.ip_pol_b, scratch);

  allocAndUpload<DataType>(&ip2_pol_w_, weights.ip2_pol_w, scratch);
  allocAndUpload<DataType>(&ip2_pol_b_, weights.ip2_pol_b, scratch);

  allocAndUpload<DataType>(&ip3_pol_w_, weights.ip3_pol_w, scratch);
  allocAndUpload<DataType>(&ip3_pol_b_, weights.ip3_pol_b, scratch);

  // big allocation to hold wq and wk weights one after the other
  {
    size_t elements = weights.ip2_pol_w.size();
    assert(elements == weights.ip3_pol_w.size());

    size_t size = elements * sizeof(DataType) * 2;
    ReportCUDAErrors(cudaMalloc(&wqk_w_, size));
    ReportCUDAErrors(
        cudaMemcpy(wqk_w_, ip2_pol_w_, size / 2, cudaMemcpyDeviceToDevice));
    ReportCUDAErrors(cudaMemcpy(wqk_w_ + elements, ip3_pol_w_, size / 2,
                                cudaMemcpyDeviceToDevice));

    elements = weights.ip2_pol_b.size();
    size = elements * sizeof(DataType) * 2;
    ReportCUDAErrors(cudaMalloc(&wqk_b_, size));
    ReportCUDAErrors(
        cudaMemcpy(wqk_b_, ip2_pol_b_, size / 2, cudaMemcpyDeviceToDevice));
    ReportCUDAErrors(cudaMemcpy(wqk_b_ + elements, ip3_pol_b_, size / 2,
                                cudaMemcpyDeviceToDevice));
  }

  allocAndUpload<DataType>(&ip4_pol_w_, weights.ip4_pol_w, scratch);

  for (const auto& enc : weights.pol_encoder) {
    EncoderBlock<DataType>* pW = new EncoderBlock<DataType>(
        enc, scratch, encoder_heads_, embedding_op_size_,
        1.0f,        // using alpha = 1 for now (TODO: may change?)
        nullptr, 0,  // smolgen weights not implemented in
                     // policy encoder heads yet.
        max_batch_size, ACTIVATION_SWISH, act_,
        1e-6,          // attentionbody nets don't have policy encoders, so
        use_gemm_ex,   // using old epsilon for backward compatibility with T78.
        false);        // fused_mha (remaining new params use defaults)
    encoder_weights_.emplace_back(pW);
  }
}

template <typename DataType>
EncoderBlock<DataType>::EncoderBlock(
    const MultiHeadWeights::EncoderLayer& cpu_weights, void* scratch, int heads,
    int size, float alpha, DataType* smolgen_global_scratch,
    int smolgen_global_size, int max_batch_size, ActivationFunction smolgen_act,
    ActivationFunction ffn_act, float default_eps, bool use_gemm_ex,
    bool fused_mha, bool use_prenorm, bool use_rms_norm, bool use_swiglu,
    bool use_parallel_ffn,
    float attn_logit_cap,
    bool has_exoformer,
    float smolgen_softcap,
    float v_softcap,
    float swiglu_softcap,
    int kv_heads,
    DataType* smol_dict_p, DataType* smol_dict_dec,
    int smol_dict_m, int smol_dict_rank)
    : embedding_op_size_(size),
      encoder_heads_(heads),
      kv_heads_(kv_heads > 0 ? kv_heads : heads),
      alpha_(alpha),
      default_eps_(default_eps),
      has_smolgen_(cpu_weights.mha.has_smolgen),
      has_smol_dict_(smol_dict_p != nullptr),
      smol_dict_m_(smol_dict_m),
      smol_dict_rank_(smol_dict_rank),
      smol_dict_p_(smol_dict_p),
      smol_dict_dec_(smol_dict_dec),
      smolgen_activation_(smolgen_act),
      ffn_activation_(ffn_act),
      max_batch_size_(max_batch_size),
      use_fused_mha_(fused_mha),
      use_gemm_ex_(use_gemm_ex),
      is_prenorm_(use_prenorm),
      use_rms_norm_(use_rms_norm),
      has_swiglu_(use_swiglu || cpu_weights.ffn.gate_proj_w.size() > 0),
      is_parallel_ffn_(use_parallel_ffn),
      attn_logit_cap_(attn_logit_cap),
      smolgen_softcap_(smolgen_softcap),
      v_softcap_(v_softcap),
      swiglu_softcap_(swiglu_softcap) {
  // Capture the active weight arena (if any) so the destructor knows
  // which weight buffers are arena-owned (skip cudaFree) vs direct
  // cudaMalloc'd (must cudaFree).  See `arena_` member declaration
  // in layers.h.  Captured BEFORE any allocAndUpload() call below so
  // every weight upload during this ctor uses the same arena.
  arena_ = tl_weight_arena;

  mha_q_size_ = cpu_weights.mha.q_b.size();
  if (mha_q_size_ == 0) {
    // Layout-1 bank nets: q_w/q_b are absent (Q = Wq2(bank)); the Q width
    // comes from the second projection's bias instead.
    mha_q_size_ = cpu_weights.mha.q2_b.size();
  }
  mha_k_size_ = cpu_weights.mha.k_b.size();
  if (mha_k_size_ == 0) {
    // K-on-bank nets: k_w/k_b absent (K = Wk2(bank)); width from k2_b.
    mha_k_size_ = cpu_weights.mha.k2_b.size();
  }
  // V bias width.  Under GLU-V, v_b is absent (the V projection splits
  // into v_gate + v_up, each with its own bias); fall back to v_gate_b
  // (same width as the standard v_b would have been).  Without this
  // fallback, mha_v_size_ would be 0 and the fused GLU-V weight
  // allocation at line ~1769 would allocate 0 bytes.
  mha_v_size_ = cpu_weights.mha.v_b.size();
  if (mha_v_size_ == 0) {
    mha_v_size_ = cpu_weights.mha.v_gate_b.size();
  }
  if (mha_v_size_ == 0) {
    // Shared-gate-bank nets: v_gate is replaced by a bank adapter, so only
    // v_up carries the projection width.
    mha_v_size_ = cpu_weights.mha.v_up_b.size();
  }
  mha_dense_size_ = cpu_weights.mha.dense_b.size();
  ffn_dense1_size_ = cpu_weights.ffn.dense1_b.size();
  ffn_dense2_size_ = cpu_weights.ffn.dense2_b.size();

  // Upload Q weights.
  allocAndUpload<DataType>(&mha_q_w, cpu_weights.mha.q_w, scratch);
  allocAndUpload<DataType>(&mha_q_b, cpu_weights.mha.q_b, scratch);

  allocAndUpload<DataType>(&mha_k_w, cpu_weights.mha.k_w, scratch);
  allocAndUpload<DataType>(&mha_k_b, cpu_weights.mha.k_b, scratch);

  allocAndUpload<DataType>(&mha_v_w, cpu_weights.mha.v_w, scratch);
  allocAndUpload<DataType>(&mha_v_b, cpu_weights.mha.v_b, scratch);

  // GQA: when kv_heads_ < encoder_heads_, the K and V projections are
  // sized to kv_dim (= kv_heads_ * depth) rather than d_model.  The
  // proto weights are already kv_dim-wide (PyTorch export does this
  // sizing), so mha_k_w / mha_v_w above are already the right size.
  const bool is_gqa = (kv_heads_ < encoder_heads_);

  if (!has_nla_ && !has_glu_attn_ && !is_gqa && mha_v_w) {
    // Merged QKV weights for batched GEMM fast-path (old nets only).
    // Not used when NLA / GLU-V / GQA is active (Q ≠ K ≠ V in width or
    // computation path).
    size_t elements = cpu_weights.mha.q_w.size();
    size_t size = elements * sizeof(DataType) * 3;
    ReportCUDAErrors(cudaMalloc(&mha_qkv_w, size));
    ReportCUDAErrors(
        cudaMemcpy(mha_qkv_w, mha_q_w, size / 3, cudaMemcpyDeviceToDevice));
    ReportCUDAErrors(cudaMemcpy(mha_qkv_w + elements, mha_k_w, size / 3,
                                cudaMemcpyDeviceToDevice));
    ReportCUDAErrors(cudaMemcpy(mha_qkv_w + elements * 2, mha_v_w, size / 3,
                                cudaMemcpyDeviceToDevice));

    elements = cpu_weights.mha.q_b.size();
    size = elements * sizeof(DataType) * 3;
    ReportCUDAErrors(cudaMalloc(&mha_qkv_b, size));
    ReportCUDAErrors(
        cudaMemcpy(mha_qkv_b, mha_q_b, size / 3, cudaMemcpyDeviceToDevice));
    ReportCUDAErrors(cudaMemcpy(mha_qkv_b + elements, mha_k_b, size / 3,
                                cudaMemcpyDeviceToDevice));
    ReportCUDAErrors(cudaMemcpy(mha_qkv_b + elements * 2, mha_v_b, size / 3,
                                cudaMemcpyDeviceToDevice));
  } else {
    // NLA/GLU V active or no V weights — no merged buffers
    mha_qkv_w = nullptr;
    mha_qkv_b = nullptr;
  }

  allocAndUpload<DataType>(&mha_dense_w, cpu_weights.mha.dense_w, scratch);
  allocAndUpload<DataType>(&mha_dense_b, cpu_weights.mha.dense_b, scratch);

  allocAndUpload<DataType>(&ln1_gammas, cpu_weights.ln1_gammas, scratch);
  allocAndUpload<DataType>(&ln1_betas, cpu_weights.ln1_betas, scratch);

  // FFN weights: SwiGLU or standard
  if (has_swiglu_) {
    allocAndUpload<DataType>(&ffn_gate_w_, cpu_weights.ffn.gate_proj_w, scratch);
    allocAndUpload<DataType>(&ffn_gate_b_, cpu_weights.ffn.gate_proj_b, scratch);
    allocAndUpload<DataType>(&ffn_up_w_, cpu_weights.ffn.up_proj_w, scratch);
    allocAndUpload<DataType>(&ffn_up_b_, cpu_weights.ffn.up_proj_b, scratch);
    allocAndUpload<DataType>(&ffn_down_w_, cpu_weights.ffn.down_proj_w, scratch);
    allocAndUpload<DataType>(&ffn_down_b_, cpu_weights.ffn.down_proj_b, scratch);
    ffn_dff_ = (int)(cpu_weights.ffn.gate_proj_w.size() / embedding_op_size_);
    if (ffn_dff_ == 0) {
      // Shared-gate-bank nets: gate_proj is replaced by a bank adapter,
      // so derive dff from up_proj instead.
      ffn_dff_ = (int)(cpu_weights.ffn.up_proj_w.size() / embedding_op_size_);
    }
    ffn_dense1_w = nullptr; ffn_dense1_b = nullptr;
    ffn_dense2_w = nullptr; ffn_dense2_b = nullptr;

    // Concatenate gate and up weights into [gate_w; up_w] of shape (2*dff, emb)
    // so we can do one fused GEMM instead of two. Used only when parallel FFN
    // with ln1_cache_ is active (scratch is free for the 2*dff output).
    if (ffn_gate_w_ && ffn_up_w_) {
      size_t w_elements = cpu_weights.ffn.gate_proj_w.size();
      size_t w_size = w_elements * sizeof(DataType);
      if (cudaMalloc(&ffn_gate_up_w_, w_size * 2) == cudaSuccess) {
        cudaMemcpy(ffn_gate_up_w_, ffn_gate_w_, w_size, cudaMemcpyDeviceToDevice);
        cudaMemcpy(ffn_gate_up_w_ + w_elements, ffn_up_w_, w_size,
                   cudaMemcpyDeviceToDevice);
      } else {
        ffn_gate_up_w_ = nullptr;
        cudaGetLastError();
      }
    } else {
      ffn_gate_up_w_ = nullptr;
    }
    // Biases: keep separate pointers; the fused kernel takes them individually.
    ffn_gate_up_b_ = nullptr;  // unused (bias applied directly from ffn_gate_b_/ffn_up_b_)
  } else {
    ffn_gate_w_ = nullptr; ffn_gate_b_ = nullptr;
    ffn_up_w_ = nullptr; ffn_up_b_ = nullptr;
    ffn_down_w_ = nullptr; ffn_down_b_ = nullptr;
    ffn_dff_ = 0;
    allocAndUpload<DataType>(&ffn_dense1_w, cpu_weights.ffn.dense1_w, scratch);
    allocAndUpload<DataType>(&ffn_dense1_b, cpu_weights.ffn.dense1_b, scratch);
    allocAndUpload<DataType>(&ffn_dense2_w, cpu_weights.ffn.dense2_w, scratch);
    allocAndUpload<DataType>(&ffn_dense2_b, cpu_weights.ffn.dense2_b, scratch);
  }

  // PGB (path-gating bias on FFN hidden, before down_proj/dense2) is shared
  // between SwiGLU and standard-FFN paths. Load it regardless of has_swiglu_.
  if (cpu_weights.ffn.pgb_ffn.size() > 0)
    allocAndUpload<DataType>(&ffn_pgb_, cpu_weights.ffn.pgb_ffn, scratch);

  // Pre-compute the (mha_dense_b + ffn_final_b) combined bias for the
  // Post-Norm + Parallel FFN fused-LN path.  That path does
  //   y = LN(residual + alpha * (buffer1 + mha_dense_b + ffn_out))
  // where ffn_out already had ffn_down_b / ffn_dense2_b added by a
  // separate addBiasBatched.  Bias addition distributes inside the LN
  // accumulator, so pre-summing
  //   combined = mha_dense_b + ffn_final_b
  // and passing it as NormLayer's `bias` is equivalent — but lets us
  // skip the ffn_*_b addBiasBatched on the FFN bottleneck stream.
  // Saves 1 launch per encoder layer on whichever stream the FFN runs on.
  //
  // Built only when:
  //  - both bias buffers exist (the user's config has both),
  //  - sizes match (= embedding_op_size_),
  //  - we're in a Post-Norm Parallel FFN encoder (else the path uses LN2
  //    separately for FFN and the bias is already inside that LN).
  if (!is_prenorm_ && is_parallel_ffn_) {
    const std::vector<float>* ffn_final_b_host = nullptr;
    if (has_swiglu_ && cpu_weights.ffn.down_proj_b.size() ==
                            (size_t)embedding_op_size_) {
      ffn_final_b_host = &cpu_weights.ffn.down_proj_b;
    } else if (!has_swiglu_ && cpu_weights.ffn.dense2_b.size() ==
                                   (size_t)embedding_op_size_) {
      ffn_final_b_host = &cpu_weights.ffn.dense2_b;
    }
    if (ffn_final_b_host != nullptr &&
        cpu_weights.mha.dense_b.size() == (size_t)embedding_op_size_) {
      std::vector<float> combined(embedding_op_size_);
      for (int i = 0; i < embedding_op_size_; ++i) {
        combined[i] = cpu_weights.mha.dense_b[i] + (*ffn_final_b_host)[i];
      }
      allocAndUpload<DataType>(&attn_ffn_combined_bias_, combined, scratch);
    }
  }

  allocAndUpload<DataType>(&ln2_gammas, cpu_weights.ln2_gammas, scratch);
  allocAndUpload<DataType>(&ln2_betas, cpu_weights.ln2_betas, scratch);

  // Smolgen weights.
  if (has_smolgen_) {
    smol_compress_size_ = cpu_weights.mha.smolgen.compress.size() / mha_q_size_;
    smol_global_size_ = smolgen_global_size;

    allocAndUpload<DataType>(&smol_compress, cpu_weights.mha.smolgen.compress,
                             scratch);

    // Smolgen V1: flatten + linear compression.
    smol_dense_1_size_ = cpu_weights.mha.smolgen.dense1_b.size();
    smol_dense_2_size_ = cpu_weights.mha.smolgen.dense2_b.size();

    allocAndUpload<DataType>(&smol_dense1_w, cpu_weights.mha.smolgen.dense1_w,
                             scratch);
    allocAndUpload<DataType>(&smol_dense1_b, cpu_weights.mha.smolgen.dense1_b,
                             scratch);
    allocAndUpload<DataType>(&smol_dense2_w, cpu_weights.mha.smolgen.dense2_w,
                             scratch);
    allocAndUpload<DataType>(&smol_dense2_b, cpu_weights.mha.smolgen.dense2_b,
                             scratch);

    allocAndUpload<DataType>(&smol_ln1_gammas,
                             cpu_weights.mha.smolgen.ln1_gammas, scratch);
    allocAndUpload<DataType>(&smol_ln1_betas,
                             cpu_weights.mha.smolgen.ln1_betas, scratch);
    allocAndUpload<DataType>(&smol_ln2_gammas,
                             cpu_weights.mha.smolgen.ln2_gammas, scratch);
    allocAndUpload<DataType>(&smol_ln2_betas,
                             cpu_weights.mha.smolgen.ln2_betas, scratch);

    // GPU memory already allocated in AttentionBody.
    smol_global = smolgen_global_scratch;
  }

  // NLA (NonLinear Attention): second-layer Q/K/V projection.
  has_nla_ = cpu_weights.mha.q2_w.size() > 0;
  if (has_nla_) {
    // Full NLA needs BOTH k_w (inner projection) and k2_w (outer).
    // NLA-Q-only nets carry neither k2.  K-on-bank nets carry k2_w
    // (bank → kv_dim) with k_w ABSENT — still the q_only Eval branch
    // (K has no private inner projection; the bank plays that role).
    const bool full_nla = cpu_weights.mha.k2_w.size() > 0 &&
                          cpu_weights.mha.k_w.size() > 0;
    nla_q_only_ = !full_nla;
    allocAndUpload<DataType>(&mha_q2_w_, cpu_weights.mha.q2_w, scratch);
    allocAndUpload<DataType>(&mha_q2_b_, cpu_weights.mha.q2_b, scratch);
    // k2 weights are needed by full NLA AND by K-on-bank.
    if (cpu_weights.mha.k2_w.size() > 0) {
      allocAndUpload<DataType>(&mha_k2_w_, cpu_weights.mha.k2_w, scratch);
      allocAndUpload<DataType>(&mha_k2_b_, cpu_weights.mha.k2_b, scratch);
    }
    if (!nla_q_only_) {

      // Fused Q1+K1 weight: [W_q1; W_k1] (2*d_model, emb).
      // Allows a single GEMM to compute both Q1 and K1 intermediates, halving
      // first-layer GEMM launches from 2 to 1 per encoder block.
      {
        const size_t emb = (size_t)embedding_op_size_;
        const size_t dm  = (size_t)mha_q_size_;
        // Weights — concatenate Q1 and K1 columns in the cuBLAS column-major sense.
        // mha_q_w and mha_k_w are both (emb, dm) column-major = dm*emb elements.
        ReportCUDAErrors(cudaMalloc(&mha_q1k1_w_, 2 * dm * emb * sizeof(DataType)));
        ReportCUDAErrors(cudaMemcpy(mha_q1k1_w_,
                                    mha_q_w, dm * emb * sizeof(DataType),
                                    cudaMemcpyDeviceToDevice));
        ReportCUDAErrors(cudaMemcpy(mha_q1k1_w_ + dm * emb,
                                    mha_k_w, dm * emb * sizeof(DataType),
                                    cudaMemcpyDeviceToDevice));
        // Biases — [q1_bias | k1_bias] (2*dm,)
        ReportCUDAErrors(cudaMalloc(&mha_q1k1_b_, 2 * dm * sizeof(DataType)));
        ReportCUDAErrors(cudaMemcpy(mha_q1k1_b_,
                                    mha_q_b, dm * sizeof(DataType),
                                    cudaMemcpyDeviceToDevice));
        ReportCUDAErrors(cudaMemcpy(mha_q1k1_b_ + dm,
                                    mha_k_b, dm * sizeof(DataType),
                                    cudaMemcpyDeviceToDevice));
      }
    }
  }

  // GLU Attention (V): gated value projection.  Under the shared gate
  // bank, v_gate_w is absent (the gate reads the bank via an adapter) but
  // the V path is still GLU-shaped — key off bank_v_diag too.
  has_glu_attn_ = cpu_weights.mha.v_gate_w.size() > 0 ||
                  cpu_weights.mha.bank_v_diag.size() > 0;
  if (has_glu_attn_) {
    allocAndUpload<DataType>(&mha_v_gate_w_, cpu_weights.mha.v_gate_w, scratch);
    allocAndUpload<DataType>(&mha_v_gate_b_, cpu_weights.mha.v_gate_b, scratch);
    allocAndUpload<DataType>(&mha_v_up_w_, cpu_weights.mha.v_up_w, scratch);
    allocAndUpload<DataType>(&mha_v_up_b_, cpu_weights.mha.v_up_b, scratch);
  }

  // When GLU-V is active, V_gate and V_up both read `mha_input` and can be
  // fused into a single 2*d_model-wide GEMM by concatenating their weights
  // into [W_v_gate; W_v_up] (2*d_model, emb).  The resulting (batch, 2*dff)
  // col-major output is exactly the layout SwiGLUFusedGateUp expects, so
  // the subsequent SwiGLU step reuses that kernel verbatim.  Cuts the
  // V-side GEMM count from 2 to 1 per encoder block.
  if (has_glu_attn_ && mha_v_gate_w_ != nullptr) {
    const size_t emb = (size_t)embedding_op_size_;
    // Under GQA, V_gate/V_up project to kv_dim, NOT d_model.  Use the
    // V bias width (= mha_v_size_ = kv_dim when GQA, = d_model otherwise)
    // as the per-projection output width.
    const size_t v_w = (size_t)mha_v_size_;
    const size_t one_w = v_w * emb * sizeof(DataType);
    auto err = cudaMalloc(&mha_vg_vu_w_, 2 * one_w);
    if (err == cudaSuccess) {
      ReportCUDAErrors(cudaMemcpy(
          mha_vg_vu_w_,                mha_v_gate_w_, one_w,
          cudaMemcpyDeviceToDevice));
      ReportCUDAErrors(cudaMemcpy(
          mha_vg_vu_w_ + v_w * emb,    mha_v_up_w_,   one_w,
          cudaMemcpyDeviceToDevice));
    } else {
      mha_vg_vu_w_ = nullptr;
      cudaGetLastError();  // clear error; fall back to 2 separate GEMMs
    }
  }

  // ── Fused QKV weight: [Wq1 | Wk | Wv_gate | Wv_up] ──
  // When NLA-Q-only + GQA + GLU-V are all active, the projection block
  // is currently three GEMMs (Wq@x, Wk@x, [Wv_gate;Wv_up]@x), each of
  // which re-reads the same `mha_input`.  Concatenating their weights
  // into one buffer lets us run a SINGLE wider GEMM (M = d_model +
  // 3*kv_dim, e.g. 1280 at 512x60 kv=8) on the main attention stream.
  // Saves: 2 GEMM launches per layer, two redundant reads of mha_input
  // (~16 MB/layer), and the larger M dim improves tensor-core occupancy
  // slightly compared to the M=512 Q1 GEMM alone.
  //
  // Output layout (col-major, ldc = mha_qkv_fused_M_):
  //   rows 0..d_model-1                              : Q1
  //   rows d_model..d_model+kv_dim-1                 : K
  //   rows d_model+kv_dim..d_model+2*kv_dim-1        : gate
  //   rows d_model+2*kv_dim..d_model+3*kv_dim-1      : up
  // Downstream (Q1 silu+bias, Q2 GEMM, SwiGLUFusedGateUp,
  // expandKVWeighted) all support arbitrary column strides through
  // the new optional parameters, so they read directly from the fused
  // buffer without an extra split pass.
  //
  // Built only when:
  //  - has_nla_ AND nla_q_only_ (Q goes Wq1→silu→Wq2; K is plain linear)
  //  - is_gqa (kv_heads_ < encoder_heads_) — K/V at kv_dim
  //  - has_glu_attn_ AND mha_vg_vu_w_ != nullptr (V is GLU with the
  //    fused gate+up weight already built above)
  // Additional gate: the fused-QKV intermediates fit in the 6*qkv_size
  // scratch reservation only when kv_heads ≤ heads/2 (so M_fused = d_model
  // + 3*kv_dim ≤ 2.5*d_model and M_fused + kv_dim ≤ 3*d_model).  For
  // larger kv_heads ratios, fall back to the 3-GEMM path which uses
  // the pre-existing 5*qkv layout.
  const bool fused_qkv_scratch_fits =
      (kv_heads_ * 2 <= encoder_heads_);
  // Shared-gate-bank variant: V has only an up-projection (the gate reads
  // the bank), so the fused weight is [Wq1 | Wk | Wv_up] with
  // M = d_model + 2*kv_dim.  Detected from cpu_weights directly because
  // the bank fields are loaded after this builder runs.
  const bool bank_fused_v = cpu_weights.gate_bank_w.size() > 0 &&
                            cpu_weights.mha.bank_v_diag.size() > 0 &&
                            mha_v_up_w_ != nullptr;
  // GLU-Q/K via bank: q2_w is absent (so has_nla_ is false) but Q and K
  // still read x through q_w/k_w (now the content/up projections) — the
  // same [Wq | Wk | Wv_up] fused GEMM applies; the bank gates each slot
  // afterwards.  This is the structural advantage of the GLU forms over
  // Layout-1/K-on-bank, whose h-based reads cannot join this GEMM.
  const bool bank_glu_qk_fused =
      bank_fused_v && !has_nla_ &&
      cpu_weights.mha.bank_q_diag.size() > 0 &&
      cpu_weights.mha.bank_k_diag.size() > 0;
  // Debug escape: LC0_NO_FUSED_QKV=1 forces the separate-GEMM Q/K/V path
  // so fused-vs-separate output parity can be checked on the same net.
  const bool disable_fused_qkv = getenv("LC0_NO_FUSED_QKV") != nullptr;
  if (!disable_fused_qkv &&
      ((has_nla_ && nla_q_only_) || bank_glu_qk_fused) && is_gqa &&
      ((has_glu_attn_ && mha_vg_vu_w_ != nullptr) || bank_fused_v) &&
      mha_q_w != nullptr && mha_k_w != nullptr && fused_qkv_scratch_fits) {
    const size_t emb = (size_t)embedding_op_size_;
    const size_t d_model = (size_t)mha_q_size_;
    const size_t kv_dim  = (size_t)(kv_heads_ * (mha_q_size_ / encoder_heads_));
    const size_t v_cols  = bank_fused_v ? kv_dim : 2 * kv_dim;
    const size_t M_fused = d_model + kv_dim + v_cols;
    const size_t total_bytes = M_fused * emb * sizeof(DataType);
    auto err = cudaMalloc(&mha_qkv_fused_w_, total_bytes);
    if (err == cudaSuccess) {
      // Layout in cuBLAS terms: stored col-major with leading dim emb.
      // For each output column j ∈ [0, M_fused), we have `emb` rows.
      // Concat along the j (output) axis:
      //   j ∈ [0, d_model)               → Wq (col-major, ld=emb)
      //   j ∈ [d_model, d_model+kv_dim)  → Wk
      //   j ∈ [d_model+kv_dim, M_fused)  → mha_vg_vu_w_ ([gate;up],
      //                                    2*kv_dim) or Wv_up (bank, kv_dim)
      const size_t bytes_q  = d_model * emb * sizeof(DataType);
      const size_t bytes_k  = kv_dim  * emb * sizeof(DataType);
      const size_t bytes_v  = v_cols * emb * sizeof(DataType);
      ReportCUDAErrors(cudaMemcpy(
          mha_qkv_fused_w_,                                  mha_q_w,
          bytes_q, cudaMemcpyDeviceToDevice));
      ReportCUDAErrors(cudaMemcpy(
          mha_qkv_fused_w_ + d_model * emb,                  mha_k_w,
          bytes_k, cudaMemcpyDeviceToDevice));
      ReportCUDAErrors(cudaMemcpy(
          mha_qkv_fused_w_ + (d_model + kv_dim) * emb,
          bank_fused_v ? mha_v_up_w_ : mha_vg_vu_w_,
          bytes_v, cudaMemcpyDeviceToDevice));
      mha_qkv_fused_M_ = (int)M_fused;
    } else {
      mha_qkv_fused_w_ = nullptr;
      mha_qkv_fused_M_ = 0;
      cudaGetLastError();  // clear; fall back to 3 separate GEMMs
    }
  }

  // ── GQA expand weights ──
  // When GQA is active, K and V are projected at kv_dim and expanded to
  // d_model via `expandKVWeighted` before attention.  For plain GQA
  // (no weighted_gqa in the proto) we synthesize block-selection
  // weights here: a (heads, kv_heads) matrix with w[h, h*kv_heads/heads]
  // = 1 and 0 elsewhere — i.e. each Q head picks exactly its group's
  // KV head, reproducing plain GQA's repeat-expand exactly.  When the
  // proto carries weighted-GQA matrices, those override the synthesis
  // (see below; weighted-GQA proto fields land in a follow-up).
  if (is_gqa) {
    const int H = encoder_heads_;
    const int KH = kv_heads_;
    const size_t expected = (size_t)H * KH;
    // Weighted GQA: when the proto carries (heads, kv_heads) blend
    // matrices, upload them verbatim — each Q head receives a learnable
    // mix of all KV heads, not just one (arXiv:2407.10855).  Otherwise
    // synthesize block-selection weights (w[h, h*KH/H] = 1) which
    // reproduces plain GQA's repeat-expand semantics exactly.
    const bool has_weighted = cpu_weights.mha.gqa_w_k.size() == expected &&
                              cpu_weights.mha.gqa_w_v.size() == expected;
    if (has_weighted) {
      allocAndUpload<DataType>(&gqa_w_k_, cpu_weights.mha.gqa_w_k, scratch);
      allocAndUpload<DataType>(&gqa_w_v_, cpu_weights.mha.gqa_w_v, scratch);
      has_weighted_gqa_ = true;
    } else {
      std::vector<float> w_block(H * KH, 0.0f);
      for (int h = 0; h < H; ++h) {
        w_block[h * KH + (h * KH / H)] = 1.0f;
      }
      allocAndUpload<DataType>(&gqa_w_k_, w_block, scratch);
      allocAndUpload<DataType>(&gqa_w_v_, w_block, scratch);
    }
  }

  // PGB (Post-Gating Bias on V).
  if (cpu_weights.mha.pgb_v.size() > 0) {
    allocAndUpload<DataType>(&pgb_v_, cpu_weights.mha.pgb_v, scratch);
  }

  // VGA-E (Element-wise gate on attention output).  Under the shared gate
  // bank the full gate projection is absent — key off bank_vga_diag too.
  has_vga_elem_ = cpu_weights.mha.vga_elem_gate_w.size() > 0 ||
                  cpu_weights.mha.bank_vga_diag.size() > 0;
  if (has_vga_elem_) {
    allocAndUpload<DataType>(&vga_elem_gate_w_, cpu_weights.mha.vga_elem_gate_w, scratch);
    allocAndUpload<DataType>(&vga_elem_gate_b_, cpu_weights.mha.vga_elem_gate_b, scratch);
    // Pre-allocate buffer for precomputed gate (pre-norm needs LN1 output as input,
    // but buffer1 gets overwritten by attention. So we precompute the gate early.)
    if (is_prenorm_) {
      size_t gate_size = (size_t)max_batch_size * 64 * mha_q_size_ * sizeof(DataType);
      auto err = cudaMalloc(&vga_gate_buf_, gate_size);
      if (err != cudaSuccess) {
        vga_gate_buf_ = nullptr;  // OOM fallback: use old code path
        cudaGetLastError();  // clear error
      }
    }
  }

  // ── Shared gate bank (A-Layout-2) ──
  // One per-layer nonlinear basis h = silu(W_bank x + b) feeds the GLU-V /
  // VGA-E / SwiGLU-FFN gates through per-site (diag + rank-r) adapters.
  // Presence-detected from gate_bank_w; the replaced projections are
  // absent in bank nets.  Training only wires this for the post-norm
  // parallel-FFN path, so the export cannot produce other combinations.
  has_gate_bank_ = cpu_weights.gate_bank_w.size() > 0;
  if (has_gate_bank_) {
    assert(!is_prenorm_ && is_parallel_ffn_ &&
           "shared gate bank requires post-norm parallel FFN");
    gate_bank_size_ = (int)cpu_weights.gate_bank_b.size();
    allocAndUpload<DataType>(&gate_bank_w_, cpu_weights.gate_bank_w, scratch);
    allocAndUpload<DataType>(&gate_bank_b_, cpu_weights.gate_bank_b, scratch);
    has_bank_v_ = cpu_weights.mha.bank_v_diag.size() > 0;
    has_bank_vga_ = cpu_weights.mha.bank_vga_diag.size() > 0;
    has_bank_ffn_ = cpu_weights.ffn.bank_gate_diag.size() > 0;
    // Layout-1: Q consumes the bank directly (Q = Wq2(h)) — detected from
    // an absent q_w with q2_w present.  No adapter fields needed; Wq2 is
    // a full matrix and absorbs any remix.
    bank_include_q_ = has_nla_ && nla_q_only_ &&
                      cpu_weights.mha.q_w.size() == 0;
    // K-on-bank: K = Wk2(h) — k2_w present with k_w absent (full NLA
    // would carry both).  Cost-neutral vs plain K; K gains the bank's
    // silu nonlinearity.
    bank_include_k_ = has_nla_ && nla_q_only_ &&
                      cpu_weights.mha.k2_w.size() > 0 &&
                      cpu_weights.mha.k_w.size() == 0;
    // GLU-Q/K via bank: the bank gates a private linear view of x
    // (q_w/k_w are the content/up projections; q2_w/k2_w absent).
    bank_glu_q_ = cpu_weights.mha.bank_q_diag.size() > 0;
    bank_glu_k_ = cpu_weights.mha.bank_k_diag.size() > 0;
    if (has_bank_v_) {
      allocAndUpload<DataType>(&bank_v_diag_, cpu_weights.mha.bank_v_diag,
                               scratch);
      allocAndUpload<DataType>(&bank_v_bias_, cpu_weights.mha.bank_v_b,
                               scratch);
      allocAndUpload<DataType>(&bank_v_lr_b_w_, cpu_weights.mha.bank_v_lr_b,
                               scratch);
      bank_rank_v_ =
          (int)(cpu_weights.mha.bank_v_lr_a.size() / gate_bank_size_);
    }
    if (has_bank_vga_) {
      allocAndUpload<DataType>(&bank_vga_diag_, cpu_weights.mha.bank_vga_diag,
                               scratch);
      allocAndUpload<DataType>(&bank_vga_bias_, cpu_weights.mha.bank_vga_b,
                               scratch);
      allocAndUpload<DataType>(&bank_vga_lr_b_w_,
                               cpu_weights.mha.bank_vga_lr_b, scratch);
      bank_rank_vga_ =
          (int)(cpu_weights.mha.bank_vga_lr_a.size() / gate_bank_size_);
    }
    if (has_bank_ffn_) {
      allocAndUpload<DataType>(&bank_ffn_diag_, cpu_weights.ffn.bank_gate_diag,
                               scratch);
      allocAndUpload<DataType>(&bank_ffn_bias_, cpu_weights.ffn.bank_gate_b,
                               scratch);
      allocAndUpload<DataType>(&bank_ffn_lr_b_w_,
                               cpu_weights.ffn.bank_gate_lr_b, scratch);
      bank_rank_ffn_ =
          (int)(cpu_weights.ffn.bank_gate_lr_a.size() / gate_bank_size_);
    }
    if (bank_glu_q_) {
      bank_q_width_ = (int)cpu_weights.mha.bank_q_diag.size();
      allocAndUpload<DataType>(&bank_q_diag_, cpu_weights.mha.bank_q_diag,
                               scratch);
      allocAndUpload<DataType>(&bank_q_bias_, cpu_weights.mha.bank_q_b,
                               scratch);
      allocAndUpload<DataType>(&bank_q_lr_b_w_, cpu_weights.mha.bank_q_lr_b,
                               scratch);
      bank_rank_q_ =
          (int)(cpu_weights.mha.bank_q_lr_a.size() / gate_bank_size_);
    }
    if (bank_glu_k_) {
      bank_k_width_ = (int)cpu_weights.mha.bank_k_diag.size();
      allocAndUpload<DataType>(&bank_k_diag_, cpu_weights.mha.bank_k_diag,
                               scratch);
      allocAndUpload<DataType>(&bank_k_bias_, cpu_weights.mha.bank_k_b,
                               scratch);
      allocAndUpload<DataType>(&bank_k_lr_b_w_, cpu_weights.mha.bank_k_lr_b,
                               scratch);
      bank_rank_k_ =
          (int)(cpu_weights.mha.bank_k_lr_a.size() / gate_bank_size_);
    }
    // Concatenate the sites' lr_a rows ([v; vga; ffn; q; k] order) into
    // one (Σrank, bank) weight so the mixers' first stage is a single
    // GEMM.  q/k append AFTER the original three so existing offsets and
    // the AttentionBody lrb-carve order stay stable.
    std::vector<float> lr_a_cat;
    lr_a_cat.reserve(cpu_weights.mha.bank_v_lr_a.size() +
                     cpu_weights.mha.bank_vga_lr_a.size() +
                     cpu_weights.ffn.bank_gate_lr_a.size() +
                     cpu_weights.mha.bank_q_lr_a.size() +
                     cpu_weights.mha.bank_k_lr_a.size());
    lr_a_cat.insert(lr_a_cat.end(), cpu_weights.mha.bank_v_lr_a.begin(),
                    cpu_weights.mha.bank_v_lr_a.end());
    lr_a_cat.insert(lr_a_cat.end(), cpu_weights.mha.bank_vga_lr_a.begin(),
                    cpu_weights.mha.bank_vga_lr_a.end());
    lr_a_cat.insert(lr_a_cat.end(), cpu_weights.ffn.bank_gate_lr_a.begin(),
                    cpu_weights.ffn.bank_gate_lr_a.end());
    lr_a_cat.insert(lr_a_cat.end(), cpu_weights.mha.bank_q_lr_a.begin(),
                    cpu_weights.mha.bank_q_lr_a.end());
    lr_a_cat.insert(lr_a_cat.end(), cpu_weights.mha.bank_k_lr_a.begin(),
                    cpu_weights.mha.bank_k_lr_a.end());
    allocAndUpload<DataType>(&bank_lr_a_w_, lr_a_cat, scratch);
  }

  // ── Load-time structural validation (ALL nets, not just detected bank
  // nets) ──  An incomplete pb — e.g. a gate-bank model exported through
  // an old torchprocess.py (which never writes bank_* at all) or through
  // a stale net_pb2 (whose unknown fields the exporter silently drops) —
  // carries NO bank fields, so has_gate_bank_ is false and the net looks
  // like a regular net with null gate weights.  Without these checks that
  // dies later as CUBLAS_STATUS_INVALID_VALUE (A=nil) mid-Eval.
  if (has_swiglu_ && ffn_gate_w_ == nullptr && !has_bank_ffn_) {
    throw Exception(
        "SwiGLU FFN has neither gate_proj_w nor bank_gate_* in the proto. "
        "Incomplete export: sync torchprocess.py on the trainer, "
        "regenerate net_pb2.py from the current net.proto, restart "
        "training, and re-export the net.");
  }
  if (cpu_weights.mha.v_up_w.size() > 0 &&
      cpu_weights.mha.v_gate_w.size() == 0 && !has_bank_v_) {
    throw Exception(
        "GLU-V has an up-projection (v_up_w) but neither v_gate_w nor "
        "bank_v_* in the proto — incomplete export (sync torchprocess.py "
        "+ regenerate net_pb2.py, then re-export).");
  }
  if (has_vga_elem_ && vga_elem_gate_w_ == nullptr && !has_bank_vga_) {
    throw Exception(
        "VGA-E has neither vga_elem_gate_w nor bank_vga_* in the proto — "
        "incomplete export (sync torchprocess.py + regenerate net_pb2.py, "
        "then re-export).");
  }
  if (has_nla_ && cpu_weights.mha.q_w.size() == 0 && !has_gate_bank_) {
    throw Exception(
        "NLA net has q2_w but no q_w and no gate bank — incomplete export "
        "(a Layout-1 net needs gate_bank_w; sync torchprocess.py + "
        "regenerate net_pb2.py, then re-export).");
  }
  if (cpu_weights.mha.k2_w.size() > 0 && cpu_weights.mha.k_w.size() == 0 &&
      !has_gate_bank_) {
    throw Exception(
        "net has k2_w but no k_w and no gate bank — incomplete export "
        "(a K-on-bank net needs gate_bank_w; sync torchprocess.py + "
        "regenerate net_pb2.py, then re-export).");
  }
  if ((cpu_weights.mha.bank_q_diag.size() > 0 ||
       cpu_weights.mha.bank_k_diag.size() > 0) &&
      !has_gate_bank_) {
    throw Exception(
        "net has bank_q_*/bank_k_* adapters but no gate_bank_w — "
        "incomplete export (sync torchprocess.py + regenerate net_pb2.py, "
        "then re-export).");
  }
  if (bank_glu_q_) {
    if (cpu_weights.mha.q2_w.size() > 0) {
      throw Exception(
          "net has both bank_q_diag (GLU-Q) and q2_w (NLA/Layout-1) — "
          "ambiguous Q form; the export should never produce this.");
    }
    if (cpu_weights.mha.q_w.size() == 0) {
      throw Exception(
          "GLU-Q net has bank_q_diag but no q_w (the content/up "
          "projection) — incomplete export.");
    }
    if (bank_q_width_ != mha_q_size_) {
      throw Exception("bank_q_diag size " + std::to_string(bank_q_width_) +
                      " != d_model " + std::to_string(mha_q_size_));
    }
  }
  if (bank_glu_k_) {
    if (cpu_weights.mha.k2_w.size() > 0) {
      throw Exception(
          "net has both bank_k_diag (GLU-K) and k2_w (full NLA / "
          "K-on-bank) — ambiguous K form; the export should never "
          "produce this.");
    }
    if (cpu_weights.mha.k_w.size() == 0) {
      throw Exception(
          "GLU-K net has bank_k_diag but no k_w (the content/up "
          "projection) — incomplete export.");
    }
    const int expect_kv =
        (kv_heads_ < encoder_heads_)
            ? kv_heads_ * (mha_q_size_ / encoder_heads_)
            : mha_q_size_;
    if (bank_k_width_ != expect_kv) {
      throw Exception("bank_k_diag size " + std::to_string(bank_k_width_) +
                      " != kv_dim " + std::to_string(expect_kv));
    }
  }

  // Persistent LN1 cache for Pre-Norm paths. Computes LN1 once per Eval.
  // SmolGen, Q/K/V, and (for parallel FFN) the FFN all read from this buffer.
  // Avoids the 1-2 LN1 recomputes the old code needed when SmolGen/dense
  // clobbered buffer1. Size: max_batch * 64 * emb_size * sizeof(DataType).
  if (is_prenorm_) {
    size_t ln1_cache_size = (size_t)max_batch_size * 64 * embedding_op_size_ * sizeof(DataType);
    auto err = cudaMalloc(&ln1_cache_, ln1_cache_size);
    if (err != cudaSuccess) {
      ln1_cache_ = nullptr;  // OOM fallback: old code path recomputes
      cudaGetLastError();
    }
  }

  // ExoFormer: per-layer lambda (stored on host to avoid stream capture issues).
  if (cpu_weights.exo_lambda.size() >= 2) {
    exo_lambda_host_[0] = cpu_weights.exo_lambda[0];
    exo_lambda_host_[1] = cpu_weights.exo_lambda[1];
  }

}

template <typename DataType>
static void cublasXgemm(cublasHandle_t handle, cublasOperation_t transa,
                        cublasOperation_t transb, int m, int n, int k,
                        float alpha, const DataType* A, int lda,
                        const DataType* B, int ldb, float beta, DataType* C,
                        int ldc) {
  const bool fp16 = std::is_same<half, DataType>::value;
  if (fp16) {
    // Default fast path: cublasHgemm with fp16 accumulator — maximum tensor-
    // core throughput. Adequate for most networks (20–40 layer pre-norm
    // transformers) where per-layer precision loss doesn't compound into
    // residual-stream overflow.
    //
    // Opt-in slower path: LC0_CUBLAS_FP32_ACCUM=1 switches to cublasGemmEx
    // with CUBLAS_COMPUTE_32F_FAST_16F. Same fp16 I/O, but fp32 accumulator
    // during the K-way reduction. Adds ~10–20% latency. Needed only for
    // very deep (60+ layer) nets where fp16 accumulation causes the
    // residual stream to overflow fp16 range. Gate it via env var so
    // shallower nets don't pay the precision surcharge.
    // cublasHgemm has kernel-dispatch failures for small n or m (e.g.,
    // n=1 projections in value/policy heads) on some driver/GPU combos,
    // producing CUBLAS_STATUS_EXECUTION_FAILED even though inputs are
    // perfectly valid.  cublasGemmEx with fp16 I/O is more robust for
    // these edge-case shapes.  Auto-fallback: try Hgemm first (fastest
    // for well-behaved shapes on tensor cores), fall back to GemmEx on
    // failure.  LC0_CUBLAS_FP32_ACCUM=1 forces GemmEx unconditionally.
    static const bool kFp32Accum =
        std::getenv("LC0_CUBLAS_FP32_ACCUM") != nullptr;
    static const bool kDumpHgemm =
        std::getenv("LC0_DUMP_HGEMM") != nullptr;
    // NOTE (2026-06-11 debugging war story): an OOB crash on a Linux rig
    // was initially pinned on the legacy Hgemm dispatch for small-m
    // shapes and "fixed" by routing them through GemmEx fp32-accum.
    // The real culprit was getMaxAttentionBodySize deriving d_model from
    // the absent q_b of a Layout-1 bank net (undersized scratch → the
    // encoder carved Q/K/V slabs past the allocation; both Hgemm and
    // GemmEx kernels faithfully wrote through the bad pointer).  The
    // small-m detour was reverted; if a GEMM ever reports
    // EXECUTION_FAILED/INTERNAL_ERROR at valid-looking pointers, suspect
    // a buffer-sizing mismatch FIRST and use LC0_DUMP_HGEMM=1 plus the
    // [bufmap] dump in AttentionBody::Eval to check pointer containment.
    auto call_gemm_ex = [&]() {
      return cublasGemmEx(
          handle, transa, transb, m, n, k, &alpha,
          (const void*)A, CUDA_R_16F, lda,
          (const void*)B, CUDA_R_16F, ldb, &beta,
          (void*)C,       CUDA_R_16F, ldc,
          CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    };
    if (kFp32Accum) {
      ReportCUBLASErrors(call_gemm_ex());
    } else {
#ifdef LC0_HAS_CUBLAS_LT
      // Per-shape tuned cuBLAS-Lt path.  On first call for a given shape,
      // runs heuristic search to pick the best algo; caches the result for
      // all subsequent calls of the same shape.  Falls back to cublasHgemm
      // on any failure (unsupported shape, runtime error, or runtime
      // disable via LC0_DISABLE_CUBLAS_LT_TUNED=1).
      if (TunedGemm16Half(handle, transa, transb, m, n, k, alpha,
                          (const __half*)A, lda, (const __half*)B, ldb,
                          beta, (__half*)C, ldc)) {
        return;
      }
#endif  // LC0_HAS_CUBLAS_LT
      const __half h_alpha = (__half)alpha;
      const __half h_beta  = (__half)beta;
      auto err = cublasHgemm(
          handle, transa, transb, m, n, k, &h_alpha,
          (const __half*)A, lda, (const __half*)B, ldb, &h_beta,
          (__half*)C, ldc);
      if (kDumpHgemm || err != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr,
                "cublasHgemm: transa=%d transb=%d m=%d n=%d k=%d "
                "lda=%d ldb=%d ldc=%d  A=%p B=%p C=%p  rc=%d\n",
                (int)transa, (int)transb, m, n, k, lda, ldb, ldc,
                (const void*)A, (const void*)B, (void*)C, (int)err);
        fflush(stderr);
      }
      if (err == CUBLAS_STATUS_EXECUTION_FAILED) {
        // Retry via GemmEx — handles edge-case shapes Hgemm fails on.
        fprintf(stderr,
                "  (Hgemm failed; retrying via cublasGemmEx)\n");
        fflush(stderr);
        err = call_gemm_ex();
      }
      ReportCUBLASErrors(err);
    }
  } else {
    ReportCUBLASErrors(cublasSgemm(handle, transa, transb, m, n, k, &alpha,
                                   (const float*)A, lda, (const float*)B, ldb,
                                   &beta, (float*)C, ldc));
  }
}

template <typename DataType>
static void cublasXGemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, float alpha, const void* A, int lda,
    long long int strideA, const void* B, int ldb, long long int strideB,
    float beta, void* C, int ldc, long long int strideC, int batchCount,
    bool use_gemm_ex) {
  const bool fp16 = std::is_same<half, DataType>::value;
  if (fp16) {
    unsigned short alpha_h = FP32toFP16(alpha);
    unsigned short beta_h = FP32toFP16(beta);
    ReportCUBLASErrors(cublasGemmStridedBatchedEx(
        handle, transa, transb, m, n, k, &alpha_h, A, CUDA_R_16F, lda, strideA,
        B, CUDA_R_16F, ldb, strideB, &beta_h, C, CUDA_R_16F, ldc, strideC,
        batchCount, CUDA_R_16F, CUBLAS_GEMM_DEFAULT));
  } else {
    if (use_gemm_ex) {
      ReportCUBLASErrors(cublasGemmStridedBatchedEx(
          handle, transa, transb, m, n, k, &alpha, A, CUDA_R_32F, lda, strideA,
          B, CUDA_R_32F, ldb, strideB, &beta, C, CUDA_R_32F, ldc, strideC,
          batchCount, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));
    } else {
      ReportCUBLASErrors(cublasSgemmStridedBatched(
          handle, transa, transb, m, n, k, &alpha, (const float*)A, lda,
          strideA, (const float*)B, ldb, strideB, &beta, (float*)C, ldc,
          strideC, batchCount));
    }
  }
}

template <typename DataType>
static void cublasXGemmBatched(cublasHandle_t handle, cublasOperation_t transa,
                               cublasOperation_t transb, int m, int n, int k,
                               float alpha, DataType** A, int lda, DataType** B,
                               int ldb, float beta, DataType** C, int ldc,
                               int batchCount) {
  const bool fp16 = std::is_same<half, DataType>::value;
  if (fp16) {
    unsigned short alpha_h = FP32toFP16(alpha);
    unsigned short beta_h = FP32toFP16(beta);
    ReportCUBLASErrors(cublasHgemmBatched(
        handle, transa, transb, m, n, k, (const half*)&alpha_h, (half**)A, lda,
        (half**)B, ldb, (const half*)&beta_h, (half**)C, ldc, batchCount));
  } else {
    ReportCUBLASErrors(cublasSgemmBatched(
        handle, transa, transb, m, n, k, &alpha, (float**)A, lda, (float**)B,
        ldb, &beta, (float**)C, ldc, batchCount));
  }
}




// Forward decl — the actual definition lives further down in the NaN-scan
// block. Needed here because debugDumpEnabled() reads it.
extern std::atomic<bool> lc0_nan_scan_armed;

// Encoder-layer counter. Reset to 0 at the start of each AttentionBody::Eval
// forward pass; EncoderBlock::Eval fetch_adds to get its layer id. Thus
// _lid == 0 consistently means "layer 0 of the current forward pass",
// independent of how many warmup or prior forwards fired before.
std::atomic<int> g_enc_layer_counter{0};

static bool debugDumpEnabled() {
  static int enabled = -1;
  if (enabled == -1) enabled = (std::getenv("LC0_DEBUG") != nullptr) ? 1 : 0;
  if (enabled != 1) return false;
  // When LC0_DUMP_TENSORS is also set, bypass-graph mode causes every
  // ComputeBlocking (warmup + real) to run forwardEval, which fires
  // debugDump every time. Gate on the real-input-seen atomic so only the
  // actual startpos/FEN call produces debug lines — without this every
  // dump the user sees could be a CUDA-graph-capture warmup with
  // zero-filled InputPlanes, not the real forward.
  static int gate_by_armed = -1;
  if (gate_by_armed == -1) {
    gate_by_armed = (std::getenv("LC0_DUMP_TENSORS") != nullptr ||
                     std::getenv("LC0_NAN_SCAN") != nullptr) ? 1 : 0;
  }
  if (gate_by_armed == 1) {
    return lc0_nan_scan_armed.load(std::memory_order_relaxed);
  }
  return true;
}

template <typename T>
static void debugDumpT(const char* label, const T* gpu_ptr, int count,
                       cudaStream_t stream) {
  if (!debugDumpEnabled()) return;
  std::vector<T> host(count);
  cudaStreamSynchronize(stream);
  cudaMemcpy(host.data(), gpu_ptr, count * sizeof(T), cudaMemcpyDeviceToHost);
  fprintf(stderr, "[DEBUG] %s:", label);
  for (int i = 0; i < count && i < 8; i++) {
    float v;
    if constexpr (std::is_same_v<T, half>) v = __half2float(host[i]);
    else v = (float)host[i];
    fprintf(stderr, " %.4f", v);
  }
  fprintf(stderr, "\n"); fflush(stderr);
}
#define debugDump(label, ptr, count, stream) debugDumpT(label, ptr, count, stream)

// ── Tensor dump for PyTorch-vs-lc0 divergence diagnosis ─────────────────
// Gated on LC0_DUMP_TENSORS=1. At each checkpoint, prints a block:
//   [TENSOR] <label> count=<n> nan=<c> inf=<c> min=<f> max=<f> mean=<f>
//     head[0..N-1]: v v v ...
//     stride[k*n/5 for k in 0..4]: v v v v v
//
// The `head` samples the first N=16 values (configurable via
// LC0_DUMP_TENSORS_N env var, max 1024).  The `stride` samples 5 equally-
// spaced anchors across the full tensor so you can check convergence past
// the head (useful when the head is all-zero and only later indices
// diverge).  Use by running the same position through a PyTorch forward
// hook that prints the same stats+samples — first checkpoint whose
// head/stride doesn't match PyTorch is where the CUDA backend diverges.
//
// Extra gate: `lc0_nan_scan_armed` (kept the name for compat with
// network_cuda.cc) is flipped true by AddInput on the first REAL (non-
// warmup) input.  Suppresses CUDA-graph-capture warmup noise.
std::atomic<bool> lc0_nan_scan_armed{false};

static bool tensorDumpEnabled() {
  static int enabled = -1;
  if (enabled == -1) {
    enabled = (std::getenv("LC0_DUMP_TENSORS") != nullptr ||
               std::getenv("LC0_NAN_SCAN") != nullptr)
              ? 1 : 0;
  }
  if (enabled != 1) return false;
  return lc0_nan_scan_armed.load(std::memory_order_relaxed);
}

static int tensorDumpHeadN() {
  static int n = -1;
  if (n == -1) {
    const char* s = std::getenv("LC0_DUMP_TENSORS_N");
    n = s ? std::atoi(s) : 16;
    if (n < 1) n = 1;
    if (n > 1024) n = 1024;
  }
  return n;
}

template <typename T>
static void debugTensorDumpT(const char* label, const T* gpu_ptr, int count,
                             cudaStream_t stream) {
  if (!tensorDumpEnabled()) return;
  if (count <= 0 || gpu_ptr == nullptr) return;
  std::vector<T> host(count);
  cudaStreamSynchronize(stream);
  cudaMemcpy(host.data(), gpu_ptr, count * sizeof(T), cudaMemcpyDeviceToHost);

  // Readable multi-line format matching lc0's [DEBUG] style:
  //   [TENSOR] label: count=<N>
  //     [0..15]:  v v v v v v v v v v v v v v v v
  //     [16..31]: v v v v v v v v v v v v v v v v
  //     ...
  // 16 values per row, 4-decimal precision, column-aligned. PyTorch side
  // emits the exact same format so row-by-row diff is trivial.
  fprintf(stderr, "[TENSOR] %s: count=%d\n", label, count);
  const int per_row = 16;
  for (int start = 0; start < count; start += per_row) {
    const int end = std::min(start + per_row, count);
    fprintf(stderr, "  [%d..%d]:", start, end - 1);
    for (int i = start; i < end; ++i) {
      float v;
      if constexpr (std::is_same_v<T, half>) v = __half2float(host[i]);
      else v = (float)host[i];
      fprintf(stderr, " %+8.4f", v);
    }
    fprintf(stderr, "\n");
  }
  fflush(stderr);
}
#define debugScanNaN(label, ptr, count, stream) \
    debugTensorDumpT(label, ptr, count, stream)

template <typename DataType>
static void NormLayer(bool use_rms, int N, int C, DataType* output,
                      const DataType* input, const DataType* bias,
                      const DataType* skip, const DataType* gammas,
                      const DataType* betas, float ep, float alpha,
                      ActivationFunction act, cudaStream_t stream,
                      const DataType* input2 = nullptr) {
  if (use_rms) {
    // PyTorch's custom RMSNorm class (torchprocess.py) hardcodes eps=1e-3
    // inside forward — the class doesn't accept an eps kwarg, so the YAML's
    // `layernorm_eps` (which the CUDA `ep` parameter reflects) only governs
    // nn.LayerNorm, NOT the RMSNorm path. Override per-call to keep the
    // RMSNorm math matched between PyTorch training and CUDA inference.
    // 1e-3 (vs the more common 1e-5/1e-6 in modern LLMs) caps the forward
    // factor 1/sqrt(σ²+ε) at ~31.6 and shrinks the backward 1/(σ²+ε)
    // Jacobian — bf16 stability margin for post-norm + DeepNorm at depth.
    // If this value ever drifts from the PyTorch class default, inference
    // and training output magnitudes diverge (~5-6% compounded over 120
    // RMSNorm calls in a 60-layer net), which biases WDL softmax
    // (magnitude-sensitive over 3 logits) and shows up in search Elo.
    constexpr float kRmsNormEps = 1e-3f;
    RMSNorm<DataType>(N, C, output, input, bias, skip, gammas, kRmsNormEps,
                      alpha, act, stream, input2);
  } else {
    LayerNorm<DataType>(N, C, output, input, bias, skip, gammas, betas, ep,
                        alpha, act, stream, input2);
  }
}

// input/output tensor is in_out_tensor, others are used as scratch.
template <typename DataType>
void EncoderBlock<DataType>::Eval(int N, DataType* in_out_tensor,
                                  DataType* scratch, DataType* buffer1,
                                  DataType* buffer2, cublasHandle_t cublas,
                                  cudaStream_t stream,
                                  DataType*** offset_pointers,
                                  DataType* prev_attn_logits,
                                  const DataType* exo_q_anc,
                                  const DataType* exo_k_anc,
                                  const DataType* exo_v_anc,
                                  const DataType* smolgen_anchor,
                                  cudaStream_t ffn_stream,
                                  cublasHandle_t ffn_cublas,
                                  cudaEvent_t ln1_done_event,
                                  cudaEvent_t ffn_done_event,
                                  DataType* ffn_buf_wide,
                                  DataType* ffn_buf_out,
                                  cudaStream_t vga_stream,
                                  cublasHandle_t vga_cublas,
                                  cudaEvent_t vga_done_event,
                                  cudaStream_t smolgen_stream,
                                  cublasHandle_t smolgen_cublas,
                                  cudaEvent_t smol_done_event,
                                  DataType* smol_gen_out,
                                  DataType* smol_interm,
                                  DataType* smol_interm2,
                                  DataType* bank_h_buf,
                                  DataType* bank_lr_buf,
                                  cudaEvent_t bank_done_event) const {
  int _lid = g_enc_layer_counter.fetch_add(1, std::memory_order_relaxed);

  const int d_model = mha_q_size_;
  const int depth = d_model / encoder_heads_;

  // Pre-Norm: LN1 before MHA. Must run BEFORE SmolGen because in PyTorch
  // SmolGen is inside MHA which receives the LN1 output, not the raw residual.
  // (Using raw residual causes fp16 overflow in deep nets — bug #9.)
  //
  // When ln1_cache_ is allocated, we write LN1 output there (persistent) so
  // SmolGen and Q/K/V can both read from it without recomputing. buffer1 is
  // then free for SmolGen to use as scratch. This saves 1-2 LN1 computes per
  // encoder layer that the old code did as recomputes.
  // FFN input source (for ms_ffn dispatch on ffn_stream):
  //   - Pre-norm: ln1_cache_ (written by main stream's LN1 earlier in Eval)
  //   - Post-norm: in_out_tensor (stable from the previous layer's final LN)
  // Pre-norm without ln1_cache_ (OOM fallback) can't use ms_ffn because the
  // source (buffer1) would race with SmolGen's main-stream work.
  DataType* ffn_src = nullptr;
  if (is_prenorm_ && ln1_cache_) {
    ffn_src = ln1_cache_;
  } else if (!is_prenorm_) {
    ffn_src = in_out_tensor;
  }

  // Multi-stream FFN: active when parallel FFN is configured, the FFN source
  // pointer is valid, a suitable event/buffer context was provided by the
  // caller, and the SwiGLU fused gate+up weight exists (for SwiGLU nets).
  // Under the shared gate bank, SwiGLU's gate comes from the bank adapter
  // and ffn_gate_up_w_ is intentionally null — the bank context (h buffer +
  // done event) takes its place as the SwiGLU-readiness condition.
  const bool ffn_swiglu_ready =
      ffn_gate_up_w_ != nullptr ||
      (has_gate_bank_ && has_bank_ffn_ && bank_h_buf != nullptr &&
       bank_done_event != nullptr);
  const bool ms_ffn =
      is_parallel_ffn_ && ffn_src != nullptr &&
      ffn_stream && ffn_cublas && ln1_done_event && ffn_done_event &&
      ffn_buf_wide && ffn_buf_out &&
      (has_swiglu_ ? ffn_swiglu_ready : ffn_dense1_w != nullptr);

  // Bank nets cannot take the single-stream parallel-FFN fallback (it has
  // no path to gate from the bank, and would issue a GEMM with the null
  // gate_proj weight).  Fail with words instead of a nil-pointer cuBLAS
  // error.  Reachable causes: LC0_FORCE_SINGLE_STREAM set, or the caller
  // didn't provide the FFN/bank stream context.
  if (has_gate_bank_ && is_parallel_ffn_ && !ms_ffn) {
    throw Exception(
        "shared-gate-bank net requires the multi-stream FFN path, but "
        "ms_ffn is disabled (LC0_FORCE_SINGLE_STREAM set, or FFN/bank "
        "stream context missing).");
  }

  DataType* mha_input = in_out_tensor;

  // Third-stream SMOLGEN: run smolgen's full pipeline on smolgen_stream in
  // parallel with Q/K/V on main stream.  Gated both at construction and via
  // caller-side null checks here.  When active, the main-stream smolgen
  // block further down is skipped, and softmax reads bias from smol_gen_out.
  //
  // KNOWN LIMITATION: unlike ms_ffn, this path does NOT have a separate
  // CaptureSmolgenGraph helper yet.  During main-stream CUDA graph capture
  // the cudaStreamWaitEvent(smolgen_stream, ...) below causes smolgen_stream
  // to join the main capture as a fork.  On some drivers that creates
  // cuBLAS workspace issues for fork-of-fork streams.  If you hit those,
  // set LC0_DISABLE_MS_SMOLGEN=1 to fall back to the main-stream smolgen
  // path, or run with --graph-capture=false.
  const bool ms_smol =
      has_smolgen_ && smolgen_stream != nullptr && smolgen_cublas != nullptr &&
      smol_done_event != nullptr && smol_gen_out != nullptr &&
      smol_interm != nullptr && smol_interm2 != nullptr &&
      ln1_done_event != nullptr;

  // For POST-norm, secondary streams read in_out_tensor which was written by
  // the previous layer's final LN on main stream.  Record ln1_done_event NOW
  // (before any main-stream work in this Eval) so secondary streams' waits
  // capture the previous layer's completion.  For pre-norm, the recording
  // happens further down, after LN1 writes ln1_cache_.
  if (!is_prenorm_ && (ms_ffn || ms_smol)) {
    ReportCUDAErrors(cudaEventRecord(ln1_done_event, stream));
  }

  // ── Shared gate bank (A-Layout-2) ──
  // Compute h = silu(W_bank x + b) and the rank-r mixer outputs FIRST on
  // the main stream, so the FFN stream (which needs h + lrb_ffn for its
  // gate) can be released as early as possible.  All three consumers
  // (GLU-V, VGA-E, FFN gate) read the same post-norm residual x that the
  // sublayers read.  Post-norm only — asserted at construction.
  const bool use_bank = has_gate_bank_ && !is_prenorm_ &&
                        bank_h_buf != nullptr && bank_done_event != nullptr;
  const int bank_total_rank = bank_rank_v_ + bank_rank_vga_ + bank_rank_ffn_ +
                              bank_rank_q_ + bank_rank_k_;
  DataType* bank_lrb_v = nullptr;
  DataType* bank_lrb_vga = nullptr;
  DataType* bank_lrb_ffn = nullptr;
  DataType* bank_lrb_q = nullptr;
  DataType* bank_lrb_k = nullptr;
  if (use_bank) {
    const int bank_batch = N * 64;
    cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, gate_bank_size_,
                          bank_batch, embedding_op_size_, 1.0f,
                          (const DataType*)gate_bank_w_, embedding_op_size_,
                          in_out_tensor, embedding_op_size_, 0.0f,
                          bank_h_buf, gate_bank_size_);
    addBiasBatched<DataType>(bank_h_buf, bank_h_buf, gate_bank_b_, 1,
                             bank_batch, gate_bank_size_, ACTIVATION_SWISH,
                             stream);
    // Rank-r mixers: stage 1 is one GEMM over the concatenated
    // [v; vga; ffn; q; k] lr_a rows; stage 2 is per-site lr_b GEMMs whose
    // outputs are carved out of bank_lr_buf AFTER the lr_a region, at
    // fixed max-token strides in [v | vga | ffn | q | k] order (must
    // match the AttentionBody allocation).  Folding stage 2 into the
    // consuming kernels was measured slower (see common_kernels.cu).
    if (bank_total_rank > 0 && bank_lr_buf != nullptr) {
      const size_t bank_max_tokens = (size_t)max_batch_size_ * 64;
      cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, bank_total_rank,
                            bank_batch, gate_bank_size_, 1.0f,
                            (const DataType*)bank_lr_a_w_, gate_bank_size_,
                            bank_h_buf, gate_bank_size_, 0.0f,
                            bank_lr_buf, bank_total_rank);
      DataType* seg = bank_lr_buf + bank_max_tokens * bank_total_rank;
      int r_off = 0;
      if (bank_rank_v_ > 0) {
        bank_lrb_v = seg;
        cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, mha_v_size_,
                              bank_batch, bank_rank_v_, 1.0f,
                              (const DataType*)bank_v_lr_b_w_, bank_rank_v_,
                              bank_lr_buf + r_off, bank_total_rank, 0.0f,
                              bank_lrb_v, mha_v_size_);
        seg += bank_max_tokens * mha_v_size_;
        r_off += bank_rank_v_;
      }
      if (bank_rank_vga_ > 0) {
        bank_lrb_vga = seg;
        cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, mha_q_size_,
                              bank_batch, bank_rank_vga_, 1.0f,
                              (const DataType*)bank_vga_lr_b_w_,
                              bank_rank_vga_, bank_lr_buf + r_off,
                              bank_total_rank, 0.0f, bank_lrb_vga,
                              mha_q_size_);
        seg += bank_max_tokens * mha_q_size_;
        r_off += bank_rank_vga_;
      }
      if (bank_rank_ffn_ > 0) {
        bank_lrb_ffn = seg;
        cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, ffn_dff_,
                              bank_batch, bank_rank_ffn_, 1.0f,
                              (const DataType*)bank_ffn_lr_b_w_,
                              bank_rank_ffn_, bank_lr_buf + r_off,
                              bank_total_rank, 0.0f, bank_lrb_ffn, ffn_dff_);
        seg += bank_max_tokens * ffn_dff_;
        r_off += bank_rank_ffn_;
      }
      if (bank_rank_q_ > 0) {
        bank_lrb_q = seg;
        cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, bank_q_width_,
                              bank_batch, bank_rank_q_, 1.0f,
                              (const DataType*)bank_q_lr_b_w_, bank_rank_q_,
                              bank_lr_buf + r_off, bank_total_rank, 0.0f,
                              bank_lrb_q, bank_q_width_);
        seg += bank_max_tokens * bank_q_width_;
        r_off += bank_rank_q_;
      }
      if (bank_rank_k_ > 0) {
        bank_lrb_k = seg;
        cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, bank_k_width_,
                              bank_batch, bank_rank_k_, 1.0f,
                              (const DataType*)bank_k_lr_b_w_, bank_rank_k_,
                              bank_lr_buf + r_off, bank_total_rank, 0.0f,
                              bank_lrb_k, bank_k_width_);
        r_off += bank_rank_k_;
      }
    }
    ReportCUDAErrors(cudaEventRecord(bank_done_event, stream));
  }

  if (is_prenorm_) {
    DataType* ln1_out = ln1_cache_ ? ln1_cache_ : buffer1;
    NormLayer<DataType>(use_rms_norm_, N * 64, embedding_op_size_, ln1_out,
                        in_out_tensor, (DataType*)nullptr, (DataType*)nullptr,
                        ln1_gammas, ln1_betas, default_eps_, 1.0,
                        ACTIVATION_NONE, stream);
    mha_input = ln1_out;
    // Pre-norm: after LN1 writes ln1_cache_, record the "layer input ready"
    // barrier so ffn_stream / smolgen_stream can safely read ln1_cache_.
    // (The FFN dispatch block itself has been hoisted out of the pre-norm
    // branch and lives further below, because it must also fire for
    // post-norm + parallel_ffn reading from in_out_tensor.)
    if (ms_ffn || ms_smol) {
      ReportCUDAErrors(cudaEventRecord(ln1_done_event, stream));
    }
    // Pre-norm FFN dispatch: ffn_src = ln1_cache_, read on ffn_stream after
    // main records ln1_done_event above.  Post-norm has its own dispatch
    // block outside the prenorm branch (since post-norm skips this entire
    // if (is_prenorm_) { ... } region).
    if (ms_ffn) {
      ReportCUDAErrors(cudaStreamWaitEvent(ffn_stream, ln1_done_event, 0));
      const cudaStream_t   ffn_s = ffn_stream;
      const cublasHandle_t ffn_h = ffn_cublas;
      const int batch = N * 64;
      const int num_inputs = embedding_op_size_;
      if (has_swiglu_) {
        // ── SwiGLU FFN ──
        // Prefer fused path: single concatenated [gate; up] GEMM +
        // SwiGLUFusedGateUp. Falls back to two-GEMM path if the concatenated
        // weight failed to allocate during construction.
        const int dff = ffn_dff_;
        // ffn_buf_wide has capacity for 2*dff*batch.
        //   Fused path: the single GEMM writes 2*dff rows into ffn_buf_wide,
        //               then SwiGLUFusedGateUp packs dff rows back into the
        //               first half.
        //   Split path: first half = gate, second half = up.
        DataType* hidden_out = ffn_buf_wide;  // dff-wide SwiGLU output region.
        if (ffn_gate_up_w_ != nullptr) {
          // Single GEMM: [gate_w; up_w] @ ffn_src → ffn_buf_wide (2*dff wide).
          // ffn_src is ln1_cache_ for pre-norm, in_out_tensor for post-norm.
          cublasXgemm(ffn_h, CUBLAS_OP_T, CUBLAS_OP_N, 2 * dff, batch,
                      num_inputs, 1.0f, (const DataType*)ffn_gate_up_w_,
                      num_inputs, ffn_src, num_inputs, 0.0f,
                      ffn_buf_wide, 2 * dff);
          // Fused kernel: reads interleaved gate/up, adds biases, silu*up.
          // swiglu_softcap_ > 0 caps silu*up product element-wise.
          SwiGLUFusedGateUp<DataType>(batch, dff, hidden_out, ffn_buf_wide,
                                       ffn_gate_b_, ffn_up_b_, ffn_s,
                                       swiglu_softcap_, ffn_pgb_);
        } else {
          // Fallback: two separate GEMMs.
          DataType* gate_out = ffn_buf_wide;
          DataType* up_out   = ffn_buf_wide + (size_t)dff * batch;
          cublasXgemm(ffn_h, CUBLAS_OP_T, CUBLAS_OP_N, dff, batch, num_inputs,
                      1.0f, (const DataType*)ffn_gate_w_, num_inputs,
                      ffn_src, num_inputs, 0.0f, gate_out, dff);
          cublasXgemm(ffn_h, CUBLAS_OP_T, CUBLAS_OP_N, dff, batch, num_inputs,
                      1.0f, (const DataType*)ffn_up_w_, num_inputs,
                      ffn_src, num_inputs, 0.0f, up_out, dff);
          SwiGLUElementwiseWithBias<DataType>(
              batch * dff, dff, hidden_out, gate_out, up_out,
              ffn_gate_b_, ffn_up_b_, ffn_s, swiglu_softcap_, ffn_pgb_);
        }
        // pgb_ffn is fused into the SwiGLU kernel above (passes ffn_pgb_).
        // No separate addBiasBatched needed — saves 1 launch on the FFN
        // stream (the bottleneck) per encoder layer.
        // down_proj: dff → emb
        cublasXgemm(ffn_h, CUBLAS_OP_T, CUBLAS_OP_N, num_inputs, batch, dff,
                    1.0f, (const DataType*)ffn_down_w_, dff, hidden_out, dff,
                    0.0f, ffn_buf_out, num_inputs);
        if (ffn_down_b_) {
          addBiasBatched(ffn_buf_out, ffn_buf_out, ffn_down_b_, 1, batch,
                         num_inputs, ACTIVATION_NONE, ffn_s);
        }
      } else {
        // ── Standard FFN (dense1 → activation → dense2) ──
        const int dff = ffn_dense1_size_;
        // 1) dense1: emb → dff
        cublasXgemm(ffn_h, CUBLAS_OP_T, CUBLAS_OP_N, dff, batch, num_inputs,
                    1.0f, (const DataType*)ffn_dense1_w, num_inputs, ffn_src,
                    num_inputs, 0.0f, ffn_buf_wide, dff);
        addBiasBatched(ffn_buf_wide, ffn_buf_wide, ffn_dense1_b, 1, batch,
                       dff, ffn_activation_, ffn_s);
        // PGB on FFN hidden (post-activation, before down-projection).
        if (ffn_pgb_) {
          addBiasBatched(ffn_buf_wide, ffn_buf_wide, ffn_pgb_, 1, batch, dff,
                         ACTIVATION_NONE, ffn_s);
        }
        // 2) dense2: dff → emb
        cublasXgemm(ffn_h, CUBLAS_OP_T, CUBLAS_OP_N, num_inputs, batch, dff,
                    1.0f, (const DataType*)ffn_dense2_w, dff, ffn_buf_wide, dff,
                    0.0f, ffn_buf_out, num_inputs);
        addBiasBatched(ffn_buf_out, ffn_buf_out, ffn_dense2_b, 1, batch,
                       num_inputs, ACTIVATION_NONE, ffn_s);
      }
      // Signal FFN completion so compute_stream_ can do the 3-way add.
      ReportCUDAErrors(cudaEventRecord(ffn_done_event, ffn_s));
    }
    {
      char _l[64];
      snprintf(_l, 64, "[layer %d] after_ln1", _lid);
      debugDump(_l, ln1_out, 8, stream);
      if (_lid == 0) {
        debugScanNaN("enc00_after_ln1_sq0", ln1_out,
                     embedding_op_size_, stream);
      }
    }
    // Precompute VGA-E gate raw logits from LN1 output.
    // Bias+sigmoid are intentionally deferred to the FusedVGAE call at the
    // application site, saving one kernel launch per encoder layer.
    // NOTE: running this on a third stream was tested and reverted — it removes
    // a GEMM from the main stream critical path and improves median (+2%) but
    // doubles CV (1.7%→3.5%) with no mean gain, because the main/FFN streams are
    // already perfectly balanced. The VGA-E GEMM provides natural timing padding.
    if (has_vga_elem_ && vga_gate_buf_) {
      const int batch = N * 64;
      cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, mha_q_size_, batch,
                            embedding_op_size_, 1.0f,
                            (const DataType*)vga_elem_gate_w_,
                            embedding_op_size_, ln1_out, embedding_op_size_,
                            0.0f, vga_gate_buf_, mha_q_size_);
      // NOTE: no sigmoid here — FusedVGAE below fuses bias+sigmoid+multiply.
    }
  }

  // Post-norm FFN dispatch on ffn_stream.  Reads from in_out_tensor (which
  // is stable throughout the layer — the previous layer's final LN wrote it
  // and no main-stream code in this layer touches it until the combined LN
  // at end of layer).  ln1_done_event was recorded at the top of Eval for
  // post-norm to capture the previous layer's completion.
  // Note: is_prenorm_=true branch above already dispatched on ffn_stream
  // when applicable, so guard this block on !is_prenorm_.
  if (ms_ffn && !is_prenorm_) {
    ReportCUDAErrors(cudaStreamWaitEvent(ffn_stream, ln1_done_event, 0));
    const cudaStream_t   ffn_s = ffn_stream;
    const cublasHandle_t ffn_h = ffn_cublas;
    const int batch = N * 64;
    const int num_inputs = embedding_op_size_;
    if (has_swiglu_) {
      const int dff = ffn_dff_;
      DataType* hidden_out = ffn_buf_wide;
      if (use_bank && has_bank_ffn_) {
        // Bank-adapter SwiGLU: up-only GEMM (the gate comes from the
        // shared bank).  The up GEMM only needs ffn_src, so it runs
        // before waiting on the bank; BankGatedMul then needs h/lrb_ffn,
        // produced on the main stream behind bank_done_event.
        DataType* up_out = ffn_buf_wide + (size_t)dff * batch;
        cublasXgemm(ffn_h, CUBLAS_OP_T, CUBLAS_OP_N, dff, batch, num_inputs,
                    1.0f, (const DataType*)ffn_up_w_, num_inputs,
                    ffn_src, num_inputs, 0.0f, up_out, dff);
        ReportCUDAErrors(cudaStreamWaitEvent(ffn_s, bank_done_event, 0));
        BankGatedMul<DataType>(batch, dff, gate_bank_size_, hidden_out,
                               bank_h_buf, bank_lrb_ffn, up_out,
                               bank_ffn_diag_, bank_ffn_bias_, ffn_up_b_,
                               ffn_pgb_, swiglu_softcap_, /*up_stride=*/0,
                               ffn_s);
      } else if (ffn_gate_up_w_ != nullptr) {
        cublasXgemm(ffn_h, CUBLAS_OP_T, CUBLAS_OP_N, 2 * dff, batch,
                    num_inputs, 1.0f, (const DataType*)ffn_gate_up_w_,
                    num_inputs, ffn_src, num_inputs, 0.0f,
                    ffn_buf_wide, 2 * dff);
        SwiGLUFusedGateUp<DataType>(batch, dff, hidden_out, ffn_buf_wide,
                                     ffn_gate_b_, ffn_up_b_, ffn_s,
                                     swiglu_softcap_, ffn_pgb_);
      } else {
        DataType* gate_out = ffn_buf_wide;
        DataType* up_out   = ffn_buf_wide + (size_t)dff * batch;
        cublasXgemm(ffn_h, CUBLAS_OP_T, CUBLAS_OP_N, dff, batch, num_inputs,
                    1.0f, (const DataType*)ffn_gate_w_, num_inputs,
                    ffn_src, num_inputs, 0.0f, gate_out, dff);
        cublasXgemm(ffn_h, CUBLAS_OP_T, CUBLAS_OP_N, dff, batch, num_inputs,
                    1.0f, (const DataType*)ffn_up_w_, num_inputs,
                    ffn_src, num_inputs, 0.0f, up_out, dff);
        SwiGLUElementwiseWithBias<DataType>(
            batch * dff, dff, hidden_out, gate_out, up_out,
            ffn_gate_b_, ffn_up_b_, ffn_s, swiglu_softcap_, ffn_pgb_);
      }
      // pgb_ffn fused into the SwiGLU kernel above; no separate add needed.
      cublasXgemm(ffn_h, CUBLAS_OP_T, CUBLAS_OP_N, num_inputs, batch, dff,
                  1.0f, (const DataType*)ffn_down_w_, dff, hidden_out, dff,
                  0.0f, ffn_buf_out, num_inputs);
      // Post-Norm + parallel FFN fuses ffn_down_b_ into the combined LN
      // bias (built at ctor as attn_ffn_combined_bias_).  When the combined
      // bias is present, skip the add here — the NormLayer call at the end
      // of this layer will absorb it.  Saves 1 launch on this (bottleneck)
      // stream per encoder layer.
      if (ffn_down_b_ && attn_ffn_combined_bias_ == nullptr) {
        addBiasBatched(ffn_buf_out, ffn_buf_out, ffn_down_b_, 1, batch,
                       num_inputs, ACTIVATION_NONE, ffn_s);
      }
    } else {
      const int dff = ffn_dense1_size_;
      cublasXgemm(ffn_h, CUBLAS_OP_T, CUBLAS_OP_N, dff, batch, num_inputs,
                  1.0f, (const DataType*)ffn_dense1_w, num_inputs, ffn_src,
                  num_inputs, 0.0f, ffn_buf_wide, dff);
      addBiasBatched(ffn_buf_wide, ffn_buf_wide, ffn_dense1_b, 1, batch,
                     dff, ffn_activation_, ffn_s);
      if (ffn_pgb_) {
        addBiasBatched(ffn_buf_wide, ffn_buf_wide, ffn_pgb_, 1, batch, dff,
                       ACTIVATION_NONE, ffn_s);
      }
      cublasXgemm(ffn_h, CUBLAS_OP_T, CUBLAS_OP_N, num_inputs, batch, dff,
                  1.0f, (const DataType*)ffn_dense2_w, dff, ffn_buf_wide, dff,
                  0.0f, ffn_buf_out, num_inputs);
      // Standard FFN: same combined-bias fusion as SwiGLU above.
      if (attn_ffn_combined_bias_ == nullptr) {
        addBiasBatched(ffn_buf_out, ffn_buf_out, ffn_dense2_b, 1, batch,
                       num_inputs, ACTIVATION_NONE, ffn_s);
      }
    }
    ReportCUDAErrors(cudaEventRecord(ffn_done_event, ffn_s));
  }

  // Third-stream smolgen dispatch: when ms_smol is active, run the entire
  // smolgen pipeline on smolgen_stream in parallel with the upcoming Q/K/V
  // projections on the main stream.  Final output goes to the dedicated
  // smol_gen_out buffer (one per encoder layer, so layer i+1 can start
  // without waiting for main to consume layer i's output).
  //
  // smolgen_stream uses smol_interm / smol_interm2 as its private scratch
  // instead of the main scratch/buffer1, avoiding any aliasing with Q/K/V.
  // Main stream will wait on smol_done_event right before softmax consumes
  // the bias.
  if (ms_smol) {
    ReportCUDAErrors(cudaStreamWaitEvent(smolgen_stream, ln1_done_event, 0));
    const cudaStream_t   sg_s = smolgen_stream;
    const cublasHandle_t sg_h = smolgen_cublas;
    const int H = encoder_heads_;

    // Step 1: Compress.
    {
      const int num_inputs = d_model;
      const int num_outputs = smol_compress_size_;
      const int batch = N * 64;
      cublasXgemm<DataType>(
          sg_h, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch, num_inputs,
          1.0f, (const DataType*)smol_compress, num_inputs, mha_input,
          num_inputs, 0.0f, smol_interm, num_outputs);
    }
    // Step 2: Dense1 + LN.
    {
      const int num_inputs = 64 * smol_compress_size_;
      const int num_outputs = smol_dense_1_size_;
      const int batch = N;
      cublasXgemm<DataType>(sg_h, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                            batch, num_inputs, 1.0f,
                            (const DataType*)smol_dense1_w, num_inputs,
                            smol_interm, num_inputs, 0.0f, smol_interm2,
                            num_outputs);
      LayerNorm<DataType>(batch, num_outputs, smol_interm, smol_interm2,
                          smol_dense1_b, (DataType*)nullptr,
                          smol_ln1_gammas, smol_ln1_betas, 1e-3, 1.0,
                          smolgen_activation_, sg_s);
    }
    // Step 3: Dense2 + LN.
    {
      const int num_inputs = smol_dense_1_size_;
      const int num_outputs = smol_dense_2_size_;
      const int batch = N;
      cublasXgemm<DataType>(sg_h, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                            batch, num_inputs, 1.0f,
                            (const DataType*)smol_dense2_w, num_inputs,
                            smol_interm, num_inputs, 0.0f, smol_interm2,
                            num_outputs);
      // Exo-as-smolgen-bias is fused as the LN's out_add: when non-null,
      // smolgen_anchor is added per-element to the LN output in the same
      // kernel, replacing the separate addVectors that used to follow.
      LayerNorm<DataType>(batch, num_outputs, smol_interm, smol_interm2,
                          smol_dense2_b, (DataType*)nullptr,
                          smol_ln2_gammas, smol_ln2_betas, 1e-3, 1.0,
                          smolgen_activation_, sg_s, /*input2=*/nullptr,
                          /*out_add=*/smolgen_anchor);
    }
    // Step 5: Final weight-gen GEMM -> smol_gen_out.
    {
      const int num_inputs = smol_dense_2_size_ / H; /* gen_sz */
      const int num_outputs = smol_global_size_; /* 64 * 64 */
      const int batch = N * H;
      if (has_smol_dict_) {
        // Smolgen dictionary: one shared decoder emits [alpha | U | V]
        // per head; bias = mixture over the shared atom bank P plus a
        // rank-r dynamic residual U·Vᵀ.  smol_interm2 is free here
        // (step 3's LN wrote smol_interm) and holds the decoder output.
        const int M = smol_dict_m_;
        const int r = smol_dict_rank_;
        const int LD = M + 2 * 64 * r;
        DataType* dec_out = smol_interm2;
        cublasXgemm<DataType>(sg_h, CUBLAS_OP_T, CUBLAS_OP_N, LD, batch,
                              num_inputs, 1.0f,
                              (const DataType*)smol_dict_dec_, num_inputs,
                              smol_interm, num_inputs, 0.0f, dec_out, LD);
        // Compose: out[o, b] = Σ_m P[m, o]·alpha[m, b].  P row-major
        // (M, 4096) = col-major (4096, M) → OP_N, lda=4096; alpha is
        // rows [0, M) of each decoder column (k=M, ldb=LD skips U/V).
        cublasXgemm<DataType>(sg_h, CUBLAS_OP_N, CUBLAS_OP_N, num_outputs,
                              batch, M, 1.0f,
                              (const DataType*)smol_dict_p_, num_outputs,
                              dec_out, LD, 0.0f, smol_gen_out, num_outputs);
        // U·Vᵀ residual accumulated into the composed maps (beta=1).
        // U/V blocks are row-major (64, r) within a decoder column =
        // col-major (r, 64) with ld=r.  D = op_T(V)·U (m=64 over j,
        // n=64 over i) lands D[j,i] = Σ_k V[j,k]U[i,k] at flat i*64+j —
        // exactly C[i,j] in the row-major map layout softmax consumes.
        if (r > 0) {
          cublasXGemmStridedBatched<DataType>(
              sg_h, CUBLAS_OP_T, CUBLAS_OP_N, 64, 64, r, 1.0f,
              dec_out + M + 64 * r, r, LD,
              dec_out + M, r, LD,
              1.0f, smol_gen_out, 64, num_outputs, batch, use_gemm_ex_);
        }
      } else {
        cublasXgemm<DataType>(sg_h, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                              batch, num_inputs, 1.0f,
                              (const DataType*)smol_global, num_inputs,
                              smol_interm, num_inputs, 0.0f, smol_gen_out,
                              num_outputs);
      }
    }
    // Signal main stream that smolgen bias in smol_gen_out is ready.
    ReportCUDAErrors(cudaEventRecord(smol_done_event, sg_s));
  }

  // Calculate smolgen weights using LN1 output (mha_input) for pre-norm.
  // Gated on !ms_smol — when smolgen runs on smolgen_stream above, we must
  // NOT also run it here (would race on scratch/buffer1 with Q/K/V).
  if (has_smolgen_ && !ms_smol) {
    const int H = encoder_heads_;

    // Step 1: Compress (shared between V1 and V2).
    // input: (N, 64, d_model), output: (N*64, hidden_channels) in scratch.
    {
      const int num_inputs = d_model;
      const int num_outputs = smol_compress_size_;
      const int batch = N * 64;
      cublasXgemm<DataType>(
          cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch, num_inputs,
          1.0f, (const DataType*)smol_compress, num_inputs, mha_input,
          num_inputs, 0.0f, scratch, num_outputs);
    }

    // Smolgen V1: flatten + linear compression.
    {
      // Hidden 1 dense: (N, 64*hc) → (N, hidden_sz)
      const int num_inputs = 64 * smol_compress_size_;
      const int num_outputs = smol_dense_1_size_;
      const int batch = N;
      cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                            batch, num_inputs, 1.0f,
                            (const DataType*)smol_dense1_w, num_inputs,
                            scratch, num_inputs, 0.0f, buffer1, num_outputs);
      LayerNorm<DataType>(batch, num_outputs, scratch, buffer1, smol_dense1_b,
                          (DataType*)nullptr, smol_ln1_gammas, smol_ln1_betas,
                          1e-3, 1.0, smolgen_activation_, stream);
    }
    {
      // Hidden 2 dense (gen_from): (N, hidden_sz) → (N, H*gen_sz)
      const int num_inputs = smol_dense_1_size_;
      const int num_outputs = smol_dense_2_size_;
      const int batch = N;
      cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                            batch, num_inputs, 1.0f,
                            (const DataType*)smol_dense2_w, num_inputs,
                            scratch, num_inputs, 0.0f, buffer1, num_outputs);
      // Exo-as-smolgen-bias is fused as the LN's out_add: when non-null,
      // smolgen_anchor is added per-element to the LN output in the same
      // kernel, replacing the separate addVectors that used to follow.
      // Shape: both `scratch` and `smolgen_anchor` are (N, H*gen_sz) flat.
      LayerNorm<DataType>(batch, num_outputs, scratch, buffer1, smol_dense2_b,
                          (DataType*)nullptr, smol_ln2_gammas, smol_ln2_betas,
                          1e-3, 1.0, smolgen_activation_, stream,
                          /*input2=*/nullptr, /*out_add=*/smolgen_anchor);
    }

    {
      // Final smolgen weight generation.
      // scratch: (gen_sz, N*H) col-major → buffer2: (64*64, N*H) col-major.
      const int num_inputs =
          smol_dense_2_size_ / H; /* gen_sz */
      const int num_outputs = smol_global_size_; /* 64 * 64 */
      const int batch = N * H;
      if (has_smol_dict_) {
        // Smolgen dictionary — see the ms_smol copy above for the math.
        // buffer1 is free here (dense2's GEMM output was consumed by the
        // LN that wrote `scratch`) and holds the decoder output.
        const int M = smol_dict_m_;
        const int r = smol_dict_rank_;
        const int LD = M + 2 * 64 * r;
        DataType* dec_out = buffer1;
        cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, LD, batch,
                              num_inputs, 1.0f,
                              (const DataType*)smol_dict_dec_, num_inputs,
                              scratch, num_inputs, 0.0f, dec_out, LD);
        cublasXgemm<DataType>(cublas, CUBLAS_OP_N, CUBLAS_OP_N, num_outputs,
                              batch, M, 1.0f,
                              (const DataType*)smol_dict_p_, num_outputs,
                              dec_out, LD, 0.0f, buffer2, num_outputs);
        if (r > 0) {
          cublasXGemmStridedBatched<DataType>(
              cublas, CUBLAS_OP_T, CUBLAS_OP_N, 64, 64, r, 1.0f,
              dec_out + M + 64 * r, r, LD,
              dec_out + M, r, LD,
              1.0f, buffer2, 64, num_outputs, batch, use_gemm_ex_);
        }
      } else {
        cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                              batch, num_inputs, 1.0f,
                              (const DataType*)smol_global, num_inputs, scratch,
                              num_inputs, 0.0f, buffer2, num_outputs);
      }
    }
  }

  // SmolGen overwrites buffer1 during its computation. If ln1_cache_ is
  // available, LN1 is preserved there and we don't need to recompute.
  // Otherwise (OOM fallback), recompute LN1 into buffer1 for Q/K/V.
  // When ms_smol is active, the main-stream smolgen block above is skipped,
  // so buffer1 is not clobbered — no recompute needed.
  if (is_prenorm_ && has_smolgen_ && !ln1_cache_ && !ms_smol) {
    NormLayer<DataType>(use_rms_norm_, N * 64, embedding_op_size_, buffer1,
                        in_out_tensor, (DataType*)nullptr, (DataType*)nullptr,
                        ln1_gammas, ln1_betas, default_eps_, 1.0,
                        ACTIVATION_NONE, stream);
    mha_input = buffer1;
  }
  DataType* mha_q;
  DataType* mha_k;
  DataType* mha_v;

  // ── ExoFormer fusion gate (Eval-scope so it's visible to both the
  //    projection block and the AddExoAnchors block below) ──
  // When is_gqa AND default lambdas [0,1] AND all three anchors are
  // present, fold K/V exo into expandKVWeighted and Q exo into Q's bias
  // add (via addBiasAndAdd).  Replaces a separate AddExoAnchors launch
  // with 0 extra launches — the bias adds were happening anyway, and
  // expandKVWeighted was happening anyway; just slightly more
  // arithmetic per output.  Default-lambda check matches the condition
  // in the AddExoAnchors block below: if either lambda is non-default
  // (WeightedAdd path), we fall through to the legacy path.
  const bool exo_active_top =
      (exo_q_anc != nullptr) && (exo_k_anc != nullptr) &&
      (exo_v_anc != nullptr);
  const bool exo_has_lambda_top =
      (exo_lambda_host_[0] != 0.0f && exo_lambda_host_[1] != 0.0f) &&
      (exo_lambda_host_[0] != 0.0f || exo_lambda_host_[1] != 1.0f);
  const bool is_gqa_top = (kv_heads_ < encoder_heads_);
  const bool exo_fused_kv =
      is_gqa_top && exo_active_top && !exo_has_lambda_top;

  {
    const int num_inputs = embedding_op_size_;
    const int batch = N * 64;
    const int max_batch = max_batch_size_ * 64;

    mha_q = scratch;

    // ── GQA locals ──
    // When kv_heads_ < encoder_heads_, K and V are projected at kv_dim
    // (not d_model) and physically expanded to d_model via
    // expandKVWeighted before attention.  Downstream code (exo, V-norm,
    // QKᵀ, etc.) sees d_model-wide K/V, unchanged from non-GQA.
    const int depth = d_model / encoder_heads_;
    const bool is_gqa = is_gqa_top;
    const int kv_dim = is_gqa ? (kv_heads_ * depth) : d_model;

    if (has_nla_) {
      mha_k = mha_q + d_model * max_batch;
      mha_v = mha_k + d_model * max_batch;
      // nla_qk_temp holds [Q1 | K1] interleaved (full NLA), or just Q1
      // (Q-only). Sized 2*d_model wide either way — see
      // getMaxAttentionBodySize (now 6*qkv_size when NLA is active to
      // also hold the fused-QKV output + v_kv when that path engages).
      DataType* nla_qk_temp = mha_v + d_model * max_batch;

      // ── Fused QKV path ──
      // When mha_qkv_fused_w_ is built (NLA-Q-only + GQA + GLU-V all
      // active, see ctor), do ONE wide GEMM that produces Q1 + K + gate
      // + up concatenated in the output, then dispatch strided
      // downstream kernels.  Saves 2 GEMM launches and 2 redundant
      // reads of mha_input per layer.  Output layout (col-major,
      // ldc = mha_qkv_fused_M_):
      //   rows 0..d_model-1                              : Q1
      //   rows d_model..d_model+kv_dim-1                 : K
      //   rows d_model+kv_dim..d_model+2*kv_dim-1        : gate
      //   rows d_model+2*kv_dim..d_model+3*kv_dim-1      : up
      if (nla_q_only_ && mha_qkv_fused_w_ != nullptr) {
        const int M_fused = mha_qkv_fused_M_;       // d_model + 3*kv_dim
        const int kv_dim_f = kv_heads_ * (d_model / encoder_heads_);
        // Scratch placement: qkv_fused_out replaces nla_qk_temp's role.
        // v_kv lives just past it.  Total scratch beyond mha_q/k/v:
        //   M_fused + kv_dim = 1280 + 256 = 1536 = 3*d_model at user shape.
        // getMaxAttentionBodySize reserves 6*qkv_size for this branch.
        DataType* qkv_fused_out = nla_qk_temp;       // = scratch + 3*d_model*max_batch
        DataType* v_kv_fused    = qkv_fused_out + (size_t)M_fused * max_batch;

        // 1) Single fused GEMM: [Wq1 | Wk | Wvgate | Wvup]^T @ mha_input.
        cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, M_fused,
                              batch, num_inputs, 1.0f,
                              (const DataType*)mha_qkv_fused_w_, num_inputs,
                              mha_input, num_inputs, 0.0f,
                              qkv_fused_out, M_fused);

        // 2) Q1 silu+bias (strided, in-place at offset 0 of qkv_fused_out).
        addBiasSiluStrided<DataType>(qkv_fused_out, qkv_fused_out, mha_q_b,
                                      batch, d_model, M_fused, stream);

        // 3) Q2 = Wq2 @ Q1 + bq2 — Q1 lives in qkv_fused_out with ldb=M_fused.
        cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, d_model,
                              batch, d_model, 1.0f,
                              (const DataType*)mha_q2_w_, d_model,
                              qkv_fused_out, M_fused, 0.0f,
                              mha_q, d_model);

        // 4) Q2 bias + optional Q exo anchor fusion.
        if (exo_fused_kv) {
          addBiasAndAdd<DataType>(mha_q, mha_q, mha_q2_b_, exo_q_anc,
                                  batch * d_model, d_model, stream);
        } else {
          addBiasBatched<DataType>(mha_q, mha_q, mha_q2_b_, 1, batch, d_model,
                                   ACTIVATION_NONE, stream);
        }

        // 5) Gate the V up-projection → v_kv_fused.
        //    (No pgb here — V's PGB is folded into expandKVWeighted below.)
        if (use_bank && has_bank_v_) {
          // Bank layout [Wq1|Wk|Wv_up]: up lives at offset d_model+kv_dim
          // with row stride M_fused; the gate comes from the shared bank.
          BankGatedMul<DataType>(
              batch, kv_dim_f, gate_bank_size_, v_kv_fused, bank_h_buf,
              bank_lrb_v, qkv_fused_out + (d_model + kv_dim_f), bank_v_diag_,
              bank_v_bias_, mha_v_up_b_, /*pgb=*/(const DataType*)nullptr,
              /*softcap=*/0.0f, /*up_stride=*/M_fused, stream);
        } else {
          // GLU layout [Wq1|Wk|Wvgate|Wvup]: gate at offset d_model+kv_dim,
          // up at d_model+2*kv_dim; pass the base pointer at d_model+kv_dim
          // and column_stride=M_fused.  Within each column, gate=[0..kv_dim),
          // up=[kv_dim..2*kv_dim).
          SwiGLUFusedGateUp<DataType>(
              batch, kv_dim_f, v_kv_fused,
              qkv_fused_out + (d_model + kv_dim_f),
              mha_v_gate_b_, mha_v_up_b_, stream,
              /*swiglu_softcap=*/0.0f,  // V-side SwiGLU never used softcap
              /*pgb_bias=*/nullptr,
              /*column_stride=*/M_fused);
        }

        // 5b) GLU-K via bank: gate the K slot IN PLACE at its strided
        //     location (out_stride == up_stride == M_fused, same-index
        //     read-then-write).  The K bias moves INSIDE the gate product
        //     — K = silu(adapter(h)) ⊙ (Wk·x + b_k) — so expandKVWeighted
        //     below must NOT add it again (b_k_fused = nullptr).
        if (bank_glu_k_) {
          assert(use_bank && "GLU-K requires the bank context");
          BankGatedMul<DataType>(
              batch, kv_dim_f, gate_bank_size_, qkv_fused_out + d_model,
              bank_h_buf, bank_lrb_k, qkv_fused_out + d_model, bank_k_diag_,
              bank_k_bias_, mha_k_b, /*pgb=*/(const DataType*)nullptr,
              /*softcap=*/0.0f, /*up_stride=*/M_fused, stream,
              /*out_stride=*/M_fused);
        }

        // 6) expandKVWeighted reads K from qkv_fused_out at offset d_model
        //    with stride M_fused; V from v_kv_fused at stride kv_dim_f.
        //    K bias and V PGB/softcap still folded in via b_k/b_v/v_softcap
        //    (K bias already consumed by the GLU-K gate when active).
        const DataType* b_k_fused = bank_glu_k_ ? nullptr : mha_k_b;
        const DataType* b_v_fused = pgb_v_;  // GLU-V branch: PGB only
        const float     v_softcap_fused = v_softcap_;
        const DataType* exo_k_f = exo_fused_kv ? exo_k_anc : nullptr;
        const DataType* exo_v_f = exo_fused_kv ? exo_v_anc : nullptr;
        expandKVWeighted<DataType>(
            mha_k, mha_v,
            qkv_fused_out + d_model, v_kv_fused,
            gqa_w_k_, gqa_w_v_, N, encoder_heads_,
            kv_heads_, depth, d_model, kv_dim_f,
            stream,
            b_k_fused, b_v_fused, v_softcap_fused,
            exo_k_f, exo_v_f,
            /*k_in_stride=*/M_fused,
            /*v_in_stride=*/0  /* 0 = use kv_dim default for v_kv_fused */);
      } else {

      // ── GQA transient buffers ──
      // When GQA is active, K and V are first projected at kv_dim (smaller
      // than d_model) and then expanded into mha_k/mha_v via
      // expandKVWeighted.  We carve transient kv-sized buffers out of
      // nla_qk_temp (2*d_model wide × max_batch).  Layout:
      //   [0 .. kv_dim*max_batch)              : k_kv
      //   [kv_dim*max_batch .. 3*kv_dim*max_batch) : gate_up_temp (2*kv_dim)
      //   [3*kv_dim*max_batch .. 4*kv_dim*max_batch) : v_kv
      // Total: 4*kv_dim*max_batch ≤ 2*d_model*max_batch when
      // kv_heads ≤ heads/2 (true for the user's kv=8, heads=16 config).
      DataType* k_kv = is_gqa ? nla_qk_temp : nullptr;
      DataType* gu_kv = is_gqa ? (nla_qk_temp + kv_dim * max_batch) : nullptr;
      DataType* v_kv = is_gqa ? (nla_qk_temp + 3 * kv_dim * max_batch) : nullptr;

      if (nla_q_only_) {
        // ── NLA Q-only ──
        // Q goes through the nonlinear two-layer path; K is a plain linear
        // projection (no wk2, no silu on K) — exactly the PyTorch
        // nla_q_only=true behaviour. Two separate GEMMs are clearer here
        // than reusing the fused [Q1|K1] buffer with a strided silu.
        //
        // NOTE: under GQA the q1 scratch is also nla_qk_temp, but we only
        // need it long enough to materialize Q (above the K/V GQA writes),
        // and Q is the first thing computed, so there's no clash.
        if (bank_include_q_) {
          // Layout-1: the shared bank IS Q's inner layer — Q = Wq2(h).
          // h (bank_h_buf) was computed at the top of Eval, so the wq
          // GEMM and its silu launch vanish entirely.
          assert(use_bank && "Layout-1 requires the bank context");
          cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, d_model,
                                batch, gate_bank_size_, 1.0f,
                                (const DataType*)mha_q2_w_, gate_bank_size_,
                                bank_h_buf, gate_bank_size_, 0.0f,
                                mha_q, d_model);
        } else {
          DataType* q1 = nla_qk_temp;  // reuse scratch (d_model wide used)
          // Q1 = Wq @ x ; silu(Q1 + bq)
          cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, d_model,
                                batch, num_inputs, 1.0f,
                                (const DataType*)mha_q_w, num_inputs,
                                mha_input, num_inputs, 0.0f, q1, d_model);
          addBiasBatched<DataType>(q1, q1, mha_q_b, 1, batch, d_model,
                                   ACTIVATION_SWISH, stream);
          // Q  = Wq2 @ Q1 + bq2 (+ exo_q_anc fused when exo_fused_kv)
          cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, d_model,
                                batch, d_model, 1.0f,
                                (const DataType*)mha_q2_w_, d_model,
                                q1, d_model, 0.0f, mha_q, d_model);
        }
        if (exo_fused_kv) {
          // Fold the Q exo anchor add into the Q2 bias add — one kernel
          // instead of two (addBiasBatched + the Q part of AddExoAnchors).
          // K/V exo go into expandKVWeighted below; AddExoAnchors is then
          // skipped entirely for this layer.
          addBiasAndAdd<DataType>(mha_q, mha_q, mha_q2_b_, exo_q_anc,
                                  batch * d_model, d_model, stream);
        } else {
          addBiasBatched<DataType>(mha_q, mha_q, mha_q2_b_, 1, batch, d_model,
                                   ACTIVATION_NONE, stream);
        }
        // K = Wk @ x + bk  (plain linear, no silu, no second projection),
        // or K = Wk2 @ h + bk2 under K-on-bank (nonlinear via the bank).
        // Under GQA, K projects to kv_dim into k_kv; otherwise directly to
        // mha_k at d_model.  expandKVWeighted will fan k_kv → mha_k below
        // and absorbs the bias add, so under GQA we skip the addBiasBatched
        // launch (one fewer kernel per encoder layer × 60 layers).
        DataType* k_dst = is_gqa ? k_kv : mha_k;
        const int k_w_out = is_gqa ? kv_dim : d_model;
        if (bank_include_k_) {
          assert(use_bank && "K-on-bank requires the bank context");
          cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, k_w_out,
                                batch, gate_bank_size_, 1.0f,
                                (const DataType*)mha_k2_w_, gate_bank_size_,
                                bank_h_buf, gate_bank_size_, 0.0f, k_dst,
                                k_w_out);
        } else {
          cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, k_w_out,
                                batch, num_inputs, 1.0f,
                                (const DataType*)mha_k_w, num_inputs,
                                mha_input, num_inputs, 0.0f, k_dst, k_w_out);
          // GLU-K via bank: gate the plain projection in place; the K
          // bias moves INSIDE the gate product, so it must not be added
          // again below / in expandKVWeighted.
          if (bank_glu_k_) {
            assert(use_bank && "GLU-K requires the bank context");
            BankGatedMul<DataType>(batch, k_w_out, gate_bank_size_, k_dst,
                                   bank_h_buf, bank_lrb_k, k_dst,
                                   bank_k_diag_, bank_k_bias_, mha_k_b,
                                   /*pgb=*/(const DataType*)nullptr,
                                   /*softcap=*/0.0f, /*up_stride=*/0, stream);
          }
        }
        const DataType* k_bias =
            bank_glu_k_ ? nullptr
                        : (bank_include_k_ ? mha_k2_b_ : mha_k_b);
        if (!is_gqa && k_bias != nullptr) {
          addBiasBatched<DataType>(k_dst, k_dst, k_bias, 1, batch, k_w_out,
                                   ACTIVATION_NONE, stream);
        }
        // else: bias deferred — passed as `b_k` to expandKVWeighted below
        // (already consumed inside the gate when GLU-K is active).
      } else {
        // ── Full NLA: nonlinear on BOTH Q and K (existing fused path) ──
        // Single GEMM: [W_q1; W_k1]^T @ x → [Q1 | K1] in one launch.
        // W concat = (emb, 2*d_model) col-major; output = (2*d_model, batch).
        // GQA + full NLA is unsupported (would need separate kv_dim-wide K
        // path through Wk2). The export disallows that combination too.
        assert(!is_gqa && "Full NLA + GQA not supported; use nla_q_only.");
        cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 2 * d_model,
                              batch, num_inputs, 1.0f,
                              (const DataType*)mha_q1k1_w_, num_inputs,
                              mha_input, num_inputs, 0.0f,
                              nla_qk_temp, 2 * d_model);
        // Fused [q1_bias | k1_bias] + SiLU across the 2*d_model wide output.
        addBiasBatched<DataType>(nla_qk_temp, nla_qk_temp, mha_q1k1_b_, 1,
                                 batch, 2 * d_model, ACTIVATION_SWISH, stream);

        // Q2: read Q1 rows (first d_model of each col) with ldb=2*d_model.
        cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, d_model, batch,
                              d_model, 1.0f, (const DataType*)mha_q2_w_,
                              d_model, nla_qk_temp, 2 * d_model, 0.0f,
                              mha_q, d_model);
        addBiasBatched<DataType>(mha_q, mha_q, mha_q2_b_, 1, batch, d_model,
                                 ACTIVATION_NONE, stream);

        // K2: read K1 rows (rows d_model..2*d_model-1) with ldb=2*d_model.
        cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, d_model, batch,
                              d_model, 1.0f, (const DataType*)mha_k2_w_,
                              d_model, nla_qk_temp + d_model, 2 * d_model, 0.0f,
                              mha_k, d_model);
        addBiasBatched<DataType>(mha_k, mha_k, mha_k2_b_, 1, batch, d_model,
                                 ACTIVATION_NONE, stream);
      }

      // Under GQA, V projection writes to kv_dim-wide v_kv (not mha_v).
      // expandKVWeighted below fans v_kv → mha_v at d_model.  The fused
      // [gate;up] GEMM and SwiGLUFusedGateUp must use kv_dim as the
      // per-projection width.
      DataType* v_dst = is_gqa ? v_kv : mha_v;
      const int v_w_out = is_gqa ? kv_dim : d_model;
      if (has_glu_attn_) {
        // GLU-V: V = SiLU(W_gate^T @ x + b_gate) * (W_up^T @ x + b_up).
        // Fused path: one 2*v_w_out-wide GEMM with [W_v_gate; W_v_up] concat
        // weight writes directly into the layout SwiGLUFusedGateUp expects
        // (col-major (batch, 2*v_w_out) with gate rows first, up rows next).
        // Under GQA the fused weight was built at width = mha_v_size_ (= kv_dim).
        // Fallback (two GEMMs) only fires if mha_vg_vu_w_ alloc failed.
        if (use_bank && has_bank_v_) {
          // Bank-adapter GLU-V: single up-only GEMM into v_dst, then the
          // fused kernel gates it IN PLACE from the shared bank (out == up
          // aliasing is same-index read-then-write, safe).  PGB + v_softcap
          // intentionally NOT applied here — they follow the standard path
          // below (non-GQA) or fold into expandKVWeighted (GQA), exactly
          // as for the non-bank GLU-V.
          cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, v_w_out,
                                batch, num_inputs, 1.0f,
                                (const DataType*)mha_v_up_w_, num_inputs,
                                mha_input, num_inputs, 0.0f, v_dst, v_w_out);
          BankGatedMul<DataType>(batch, v_w_out, gate_bank_size_, v_dst,
                                 bank_h_buf, bank_lrb_v, v_dst,
                                 bank_v_diag_, bank_v_bias_, mha_v_up_b_,
                                 /*pgb=*/(const DataType*)nullptr,
                                 /*softcap=*/0.0f, /*up_stride=*/0, stream);
        } else if (mha_vg_vu_w_ != nullptr) {
          // Under GQA, the fused [gate;up] output lives in gu_kv (carved
          // earlier from nla_qk_temp).  In the non-GQA case we reuse
          // nla_qk_temp itself — same layout, just full width.
          DataType* gu_dst = is_gqa ? gu_kv : nla_qk_temp;
          cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 2 * v_w_out,
                                batch, num_inputs, 1.0f,
                                (const DataType*)mha_vg_vu_w_, num_inputs,
                                mha_input, num_inputs, 0.0f,
                                gu_dst, 2 * v_w_out);
          SwiGLUFusedGateUp<DataType>(batch, v_w_out, v_dst, gu_dst,
                                       mha_v_gate_b_, mha_v_up_b_, stream);
        } else {
          // Two-GEMM fallback.  Under GQA we need a glu_temp the right
          // (kv_dim) width; carving it from nla_qk_temp like the fused
          // path is fine since v_dst = v_kv lives elsewhere in there.
          DataType* glu_temp =
              is_gqa ? gu_kv : (mha_v + d_model * max_batch);
          cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, v_w_out, batch,
                                num_inputs, 1.0f, (const DataType*)mha_v_gate_w_,
                                num_inputs, mha_input, num_inputs, 0.0f,
                                glu_temp, v_w_out);
          cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, v_w_out, batch,
                                num_inputs, 1.0f, (const DataType*)mha_v_up_w_,
                                num_inputs, mha_input, num_inputs, 0.0f,
                                v_dst, v_w_out);
          SwiGLUElementwiseWithBias<DataType>(N * 64 * v_w_out, v_w_out, v_dst,
                                               glu_temp, v_dst, mha_v_gate_b_,
                                               mha_v_up_b_, stream);
        }
        // PGB add fused with v_softcap when both are active. Three modes:
        //   PGB on,  cap on : addBiasAndSoftCap (one fused kernel)
        //   PGB on,  cap off: addBiasBatched (legacy)
        //   PGB off, cap on : addBiasAndSoftCap with bias=nullptr (cap-only)
        //   PGB off, cap off: no-op
        // All operate at v_w_out width (= kv_dim under GQA, = d_model otherwise).
        //
        // Under GQA the PGB add + softcap are deferred into expandKVWeighted
        // below — math is identical and saves a kernel launch per layer.
        if (!is_gqa) {
          if (pgb_v_ && v_softcap_ > 0.0f) {
            addBiasAndSoftCap<DataType>(v_dst, v_dst, pgb_v_,
                                         N * 64 * v_w_out, v_w_out,
                                         v_softcap_, stream);
          } else if (pgb_v_) {
            addBiasBatched<DataType>(v_dst, v_dst, pgb_v_, 1, batch,
                                     v_w_out, ACTIVATION_NONE, stream);
          } else if (v_softcap_ > 0.0f) {
            addBiasAndSoftCap<DataType>(v_dst, v_dst, /*bias=*/nullptr,
                                         N * 64 * v_w_out, v_w_out,
                                         v_softcap_, stream);
          }
        }
      } else if (mha_v_w) {
        // Plain linear V (no GLU-V).  Under GQA the proto weight is sized
        // to kv_dim already; just project to v_w_out.  Under GQA the bias
        // (mha_v_b) is deferred into expandKVWeighted as `b_v`.
        cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, v_w_out, batch,
                              num_inputs, 1.0f, (const DataType*)mha_v_w,
                              num_inputs, mha_input, num_inputs, 0.0f,
                              v_dst, v_w_out);
        if (!is_gqa) {
          addBiasBatched<DataType>(v_dst, v_dst, mha_v_b, 1, batch,
                                   v_w_out, ACTIVATION_NONE, stream);
        }
      }

      // ── GQA expand ──
      // Now k_kv and v_kv hold (kv_heads × depth) outputs per token.
      // expandKVWeighted physically blends them into mha_k/mha_v at
      // (heads × depth) using the (heads, kv_heads) blend matrices
      // gqa_w_k_/gqa_w_v_.  For plain GQA those are the synthesized
      // block-selection weights (one 1.0 per row); for weighted-GQA they
      // come from the proto.  Downstream code (exo, V-norm, QKᵀ) consumes
      // d_model-wide K/V identically to the non-GQA path.
      //
      // Fusion (this branch only):
      //   b_k = mha_k_b        — kv_dim K bias (was a separate addBiasBatched)
      //   b_v = pgb_v_ or mha_v_b
      //     - GLU-V path: SwiGLU consumed v_gate_b/v_up_b internally; the
      //       remaining bias is PGB (pgb_v_) — may be null if PGB disabled.
      //     - Plain V path: the V projection bias (mha_v_b).
      //   v_softcap = v_softcap_ — applied per-(kv_heads × depth) value
      //     before the blend sum.  GLU-V path only; plain-V path passes 0.
      // The kernel absorbs both and applies softcap-then-blend (math
      // identical to softcap-then-blend done as separate kernels).
      if (is_gqa) {
        const DataType* b_k_fused =
            bank_glu_k_ ? nullptr
                        : (bank_include_k_ ? mha_k2_b_ : mha_k_b);
        const DataType* b_v_fused = has_glu_attn_ ? pgb_v_ : mha_v_b;
        const float v_softcap_fused = has_glu_attn_ ? v_softcap_ : 0.0f;
        const DataType* exo_k_fused = exo_fused_kv ? exo_k_anc : nullptr;
        const DataType* exo_v_fused = exo_fused_kv ? exo_v_anc : nullptr;
        expandKVWeighted<DataType>(mha_k, mha_v, k_kv, v_kv,
                                   gqa_w_k_, gqa_w_v_, N, encoder_heads_,
                                   kv_heads_, depth, d_model, kv_dim,
                                   stream, b_k_fused, b_v_fused,
                                   v_softcap_fused,
                                   exo_k_fused, exo_v_fused);
      }
      }  // end of !fused-QKV branch
    } else if (mha_qkv_w) {
      const int num_outputs = d_model;
      mha_k = mha_q + num_outputs * max_batch;
      mha_v = mha_k + num_outputs * max_batch;
      if (has_glu_attn_) {
        cublasXGemmStridedBatched<DataType>(
            cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch, num_inputs,
            1.0f, mha_qkv_w, num_inputs, num_inputs * num_outputs, mha_input,
            num_inputs, 0, 0.0f, mha_q, num_outputs, num_outputs * max_batch,
            2, use_gemm_ex_);
        addBiasBatched<DataType>(mha_q, mha_q, mha_qkv_b, 2, batch,
                                 num_outputs, max_batch, ACTIVATION_NONE, stream);
        DataType* glu_temp = mha_v + num_outputs * max_batch;
        cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                              batch, num_inputs, 1.0f,
                              (const DataType*)mha_v_gate_w_, num_inputs,
                              mha_input, num_inputs, 0.0f, glu_temp,
                              num_outputs);
        cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                              batch, num_inputs, 1.0f,
                              (const DataType*)mha_v_up_w_, num_inputs,
                              mha_input, num_inputs, 0.0f, mha_v,
                              num_outputs);
        // Fused bias+silu+mult
        SwiGLUElementwiseWithBias<DataType>(N * 64 * num_outputs, num_outputs,
                                             mha_v, glu_temp, mha_v,
                                             mha_v_gate_b_, mha_v_up_b_,
                                             stream);
        // PGB + v_softcap fusion (see comment in NLA branch above).
        if (pgb_v_ && v_softcap_ > 0.0f) {
          addBiasAndSoftCap<DataType>(mha_v, mha_v, pgb_v_,
                                       N * 64 * num_outputs, num_outputs,
                                       v_softcap_, stream);
        } else if (pgb_v_) {
          addBiasBatched<DataType>(mha_v, mha_v, pgb_v_, 1, batch,
                                   num_outputs, ACTIVATION_NONE, stream);
        } else if (v_softcap_ > 0.0f) {
          addBiasAndSoftCap<DataType>(mha_v, mha_v, /*bias=*/nullptr,
                                       N * 64 * num_outputs, num_outputs,
                                       v_softcap_, stream);
        }
      } else {
        cublasXGemmStridedBatched<DataType>(
            cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch, num_inputs,
            1.0f, mha_qkv_w, num_inputs, num_inputs * num_outputs, mha_input,
            num_inputs, 0, 0.0f, mha_q, num_outputs, num_outputs * max_batch,
            3, use_gemm_ex_);
        addBiasBatched<DataType>(mha_q, mha_q, mha_qkv_b, 3, batch,
                                 num_outputs, max_batch, ACTIVATION_NONE, stream);
      }
    } else {
      // Fallback: no NLA, no merged QKV (e.g. GLU-V on its own, plain
      // GQA, or weighted-GQA without NLA).  Separate projections; under
      // GQA, K and V are projected at kv_dim into transient buffers
      // (k_kv, v_kv, gu_kv) carved out of the scratch that follows
      // mha_v at d_model.  The buffer plan mirrors the NLA branch:
      //   k_kv   = mha_v + d_model*max_batch                + 0
      //   gu_kv  = mha_v + d_model*max_batch + kv_dim*max_batch
      //   v_kv   = mha_v + d_model*max_batch + 3*kv_dim*max_batch
      // Total scratch consumed: 4*kv_dim*max_batch — fits within the
      // (≥4*qkv_size = 4*d_model*max_batch) reservation in
      // getMaxAttentionBodySize whenever kv_heads ≤ heads.
      mha_k = mha_q + d_model * max_batch;
      mha_v = mha_k + d_model * max_batch;
      DataType* kv_scratch = mha_v + d_model * max_batch;
      DataType* k_kv2 = is_gqa ? kv_scratch : nullptr;
      DataType* gu_kv2 = is_gqa ? (kv_scratch + kv_dim * max_batch) : nullptr;
      DataType* v_kv2 = is_gqa ? (kv_scratch + 3 * kv_dim * max_batch) : nullptr;

      // ── Fused [Wq | Wk | Wv_up] GEMM for GLU-Q/K bank nets ──
      // Q/K/V all read x (the bank only GATES), so the NLA path's single
      // input GEMM applies unchanged here; each slot is then gated from
      // the bank.  This is the structural payoff of the GLU forms over
      // Layout-1/K-on-bank, whose h-based reads cannot join this GEMM.
      // Scratch: qkv_fused_out (M_fused ≤ 2*d_model) + v_kv_fused
      // (kv_dim) reuse the GQA transient region — 2.5*d_model ≤ the
      // 4*d_model this region holds under the 7*qkv reservation.
      if (mha_qkv_fused_w_ != nullptr && bank_glu_q_ && bank_glu_k_) {
        assert(use_bank && "GLU-Q/K requires the bank context");
        assert(is_gqa && "fused [Wq|Wk|Wv_up] is built under GQA only");
        const int M_fused = mha_qkv_fused_M_;  // d_model + 2*kv_dim
        DataType* qkv_fused_out = kv_scratch;
        DataType* v_kv_fused = qkv_fused_out + (size_t)M_fused * max_batch;

        // 1) Single fused GEMM: [Wq | Wk | Wv_up]^T @ mha_input.
        cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, M_fused,
                              batch, num_inputs, 1.0f,
                              (const DataType*)mha_qkv_fused_w_, num_inputs,
                              mha_input, num_inputs, 0.0f,
                              qkv_fused_out, M_fused);

        // 2) Q = silu(adapter_q(h)) ⊙ (up_q + b_q) → mha_q (contiguous).
        //    up_q lives at offset 0 of the fused output, row stride
        //    M_fused; the Q bias moves inside the gate product.
        BankGatedMul<DataType>(batch, d_model, gate_bank_size_, mha_q,
                               bank_h_buf, bank_lrb_q, qkv_fused_out,
                               bank_q_diag_, bank_q_bias_, mha_q_b,
                               /*pgb=*/(const DataType*)nullptr,
                               /*softcap=*/0.0f, /*up_stride=*/M_fused,
                               stream);
        if (exo_fused_kv) {
          // Anchor adds AFTER the gate (matches torch: q += exo_q after
          // the projection completes).  Bias already inside the gate.
          addVectors<DataType>(mha_q, mha_q,
                               const_cast<DataType*>(exo_q_anc),
                               batch * d_model, batch * d_model,
                               batch * d_model, ACTIVATION_NONE, stream);
        }

        // 3) K gated IN PLACE at its strided slot (offset d_model); the
        //    K bias is consumed here, so expandKVWeighted gets b_k=null.
        BankGatedMul<DataType>(batch, kv_dim, gate_bank_size_,
                               qkv_fused_out + d_model, bank_h_buf,
                               bank_lrb_k, qkv_fused_out + d_model,
                               bank_k_diag_, bank_k_bias_, mha_k_b,
                               /*pgb=*/(const DataType*)nullptr,
                               /*softcap=*/0.0f, /*up_stride=*/M_fused,
                               stream, /*out_stride=*/M_fused);

        // 4) V: gate the up slot (offset d_model+kv_dim) from the bank
        //    → v_kv_fused (contiguous).  PGB + v_softcap deferred into
        //    expandKVWeighted, exactly as the non-fused GLU-V path.
        BankGatedMul<DataType>(batch, kv_dim, gate_bank_size_, v_kv_fused,
                               bank_h_buf, bank_lrb_v,
                               qkv_fused_out + (d_model + kv_dim),
                               bank_v_diag_, bank_v_bias_, mha_v_up_b_,
                               /*pgb=*/(const DataType*)nullptr,
                               /*softcap=*/0.0f, /*up_stride=*/M_fused,
                               stream);

        // 5) Expand: K strided from the fused buffer, V contiguous.
        expandKVWeighted<DataType>(
            mha_k, mha_v, qkv_fused_out + d_model, v_kv_fused,
            gqa_w_k_, gqa_w_v_, N, encoder_heads_, kv_heads_, depth,
            d_model, kv_dim, stream,
            /*b_k=*/nullptr, /*b_v=*/pgb_v_, v_softcap_,
            exo_fused_kv ? exo_k_anc : nullptr,
            exo_fused_kv ? exo_v_anc : nullptr,
            /*k_in_stride=*/M_fused, /*v_in_stride=*/0);
      } else {

      // Q projection (always d_model — Q has the full head count).
      // When exo_fused_kv, fold Q exo into Q's bias add (one kernel
      // instead of bias + Q part of AddExoAnchors).
      cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, d_model, batch,
                            num_inputs, 1.0f, (const DataType*)mha_q_w,
                            num_inputs, mha_input, num_inputs, 0.0f,
                            mha_q, d_model);
      if (bank_glu_q_) {
        // GLU-Q via bank: gate the plain projection in place; the Q bias
        // moves INSIDE the gate product (silu(adapter(h)) ⊙ (Wq·x + b)).
        assert(use_bank && "GLU-Q requires the bank context");
        BankGatedMul<DataType>(batch, d_model, gate_bank_size_, mha_q,
                               bank_h_buf, bank_lrb_q, mha_q, bank_q_diag_,
                               bank_q_bias_, mha_q_b,
                               /*pgb=*/(const DataType*)nullptr,
                               /*softcap=*/0.0f, /*up_stride=*/0, stream);
        if (exo_fused_kv) {
          addVectors<DataType>(mha_q, mha_q,
                               const_cast<DataType*>(exo_q_anc),
                               batch * d_model, batch * d_model,
                               batch * d_model, ACTIVATION_NONE, stream);
        }
      } else if (exo_fused_kv) {
        addBiasAndAdd<DataType>(mha_q, mha_q, mha_q_b, exo_q_anc,
                                batch * d_model, d_model, stream);
      } else {
        addBiasBatched<DataType>(mha_q, mha_q, mha_q_b, 1, batch, d_model,
                                 ACTIVATION_NONE, stream);
      }

      // K projection: kv_dim under GQA into k_kv2, else d_model into mha_k.
      // Under GQA the bias is deferred into expandKVWeighted as `b_k`
      // (unless GLU-K consumed it inside the gate product).
      {
        DataType* k_dst = is_gqa ? k_kv2 : mha_k;
        const int k_w_out = is_gqa ? kv_dim : d_model;
        cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, k_w_out, batch,
                              num_inputs, 1.0f, (const DataType*)mha_k_w,
                              num_inputs, mha_input, num_inputs, 0.0f,
                              k_dst, k_w_out);
        if (bank_glu_k_) {
          // GLU-K via bank: gate in place; K bias consumed here.
          assert(use_bank && "GLU-K requires the bank context");
          BankGatedMul<DataType>(batch, k_w_out, gate_bank_size_, k_dst,
                                 bank_h_buf, bank_lrb_k, k_dst,
                                 bank_k_diag_, bank_k_bias_, mha_k_b,
                                 /*pgb=*/(const DataType*)nullptr,
                                 /*softcap=*/0.0f, /*up_stride=*/0, stream);
        } else if (!is_gqa) {
          addBiasBatched<DataType>(k_dst, k_dst, mha_k_b, 1, batch, k_w_out,
                                   ACTIVATION_NONE, stream);
        }
      }

      // V projection: kv_dim under GQA into v_kv2, else d_model into mha_v.
      // Under GQA the PGB + softcap are deferred into expandKVWeighted.
      DataType* v_dst = is_gqa ? v_kv2 : mha_v;
      const int v_w_out = is_gqa ? kv_dim : d_model;
      if (has_glu_attn_) {
        if (use_bank && has_bank_v_) {
          // Bank-adapter GLU-V (non-NLA route, e.g. GLU-Q nets): single
          // up-only GEMM, then gate in place from the shared bank.  PGB
          // + v_softcap follow the standard handling below, exactly as
          // the non-bank GLU-V.
          cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, v_w_out,
                                batch, num_inputs, 1.0f,
                                (const DataType*)mha_v_up_w_, num_inputs,
                                mha_input, num_inputs, 0.0f, v_dst, v_w_out);
          BankGatedMul<DataType>(batch, v_w_out, gate_bank_size_, v_dst,
                                 bank_h_buf, bank_lrb_v, v_dst,
                                 bank_v_diag_, bank_v_bias_, mha_v_up_b_,
                                 /*pgb=*/(const DataType*)nullptr,
                                 /*softcap=*/0.0f, /*up_stride=*/0, stream);
        } else {
        DataType* glu_temp = is_gqa ? gu_kv2 : (mha_v + d_model * max_batch);
        cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, v_w_out, batch,
                              num_inputs, 1.0f, (const DataType*)mha_v_gate_w_,
                              num_inputs, mha_input, num_inputs, 0.0f,
                              glu_temp, v_w_out);
        cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, v_w_out, batch,
                              num_inputs, 1.0f, (const DataType*)mha_v_up_w_,
                              num_inputs, mha_input, num_inputs, 0.0f,
                              v_dst, v_w_out);
        // Fused bias+silu+mult
        SwiGLUElementwiseWithBias<DataType>(N * 64 * v_w_out, v_w_out, v_dst,
                                             glu_temp, v_dst, mha_v_gate_b_,
                                             mha_v_up_b_, stream);
        }
        // PGB + v_softcap fusion (see comment in NLA branch above).
        // Under GQA: deferred into expandKVWeighted.
        if (!is_gqa) {
          if (pgb_v_ && v_softcap_ > 0.0f) {
            addBiasAndSoftCap<DataType>(v_dst, v_dst, pgb_v_,
                                         N * 64 * v_w_out, v_w_out,
                                         v_softcap_, stream);
          } else if (pgb_v_) {
            addBiasBatched<DataType>(v_dst, v_dst, pgb_v_, 1, batch,
                                     v_w_out, ACTIVATION_NONE, stream);
          } else if (v_softcap_ > 0.0f) {
            addBiasAndSoftCap<DataType>(v_dst, v_dst, /*bias=*/nullptr,
                                         N * 64 * v_w_out, v_w_out,
                                         v_softcap_, stream);
          }
        }
      } else if (mha_v_w) {
        cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, v_w_out, batch,
                              num_inputs, 1.0f, (const DataType*)mha_v_w,
                              num_inputs, mha_input, num_inputs, 0.0f,
                              v_dst, v_w_out);
        if (!is_gqa) {
          addBiasBatched<DataType>(v_dst, v_dst, mha_v_b, 1, batch, v_w_out,
                                   ACTIVATION_NONE, stream);
        }
      }

      // ── GQA expand (non-NLA branch) ──
      // Mirror the NLA path: fan k_kv2/v_kv2 (kv_dim wide) → mha_k/mha_v
      // (d_model wide) via the (heads, kv_heads) blend matrices.  Bias +
      // PGB + softcap + K/V exo fused (see expandKVWeighted in kernels.h).
      // Caller fuses Q exo into Q's bias add separately when exo_fused_kv;
      // in the non-NLA fallback that's the addBiasBatched on mha_q above.
      // For now, the non-NLA branch keeps Q exo in AddExoAnchors and only
      // fuses K/V here.  AddExoAnchors gates Q-only mode via the
      // (exo_fused_kv) check below.
      if (is_gqa) {
        const DataType* b_k_fused =
            bank_glu_k_ ? nullptr
                        : (bank_include_k_ ? mha_k2_b_ : mha_k_b);
        const DataType* b_v_fused = has_glu_attn_ ? pgb_v_ : mha_v_b;
        const float v_softcap_fused = has_glu_attn_ ? v_softcap_ : 0.0f;
        const DataType* exo_k_fused = exo_fused_kv ? exo_k_anc : nullptr;
        const DataType* exo_v_fused = exo_fused_kv ? exo_v_anc : nullptr;
        expandKVWeighted<DataType>(mha_k, mha_v, k_kv2, v_kv2,
                                   gqa_w_k_, gqa_w_v_, N, encoder_heads_,
                                   kv_heads_, depth, d_model, kv_dim,
                                   stream, b_k_fused, b_v_fused,
                                   v_softcap_fused,
                                   exo_k_fused, exo_v_fused);
      }
      }  // end of !fused-GLU-QK branch
    }
  }

  {
    char _l[64];
    snprintf(_l, 64, "[layer %d] Q", _lid);
    debugDump(_l, mha_q, 8, stream);
    snprintf(_l, 64, "[layer %d] K", _lid);
    debugDump(_l, mha_k, 8, stream);
    snprintf(_l, 64, "[layer %d] V", _lid);
    debugDump(_l, mha_v, 8, stream);
    if (_lid == 0) {
      debugScanNaN("enc00_Q_sq0", mha_q, d_model, stream);
      debugScanNaN("enc00_K_sq0", mha_k, d_model, stream);
      debugScanNaN("enc00_V_sq0", mha_v, d_model, stream);
    }
  }


  // ExoFormer: add pre-computed anchor projections to Q/K/V.
  // When exo_fused_kv was true, K/V exo was folded into expandKVWeighted
  // and Q exo was folded into the Q bias add — nothing more to do.
  if (exo_q_anc && exo_k_anc && exo_v_anc && !exo_fused_kv) {
    const int q_total = N * 64 * d_model;
    const int kv_total = N * 64 * d_model;
    const bool has_lambda = (exo_lambda_host_[0] != 0.0f && exo_lambda_host_[1] != 0.0f)
                          && (exo_lambda_host_[0] != 0.0f || exo_lambda_host_[1] != 1.0f);
    if (!has_lambda) {
      // Fast path (default lambdas [0,1]): fuse all three adds into
      // one kernel launch, saving 2 launch overheads × N layers.
      AddExoAnchors<DataType>(q_total, mha_q, exo_q_anc,
                              mha_k, exo_k_anc,
                              mha_v, exo_v_anc, stream);
    } else {
      WeightedAdd<DataType>(q_total, mha_q, exo_lambda_host_[0], exo_q_anc,
                            exo_lambda_host_[1], mha_q, stream);
      WeightedAdd<DataType>(kv_total, mha_k, exo_lambda_host_[0], exo_k_anc,
                            exo_lambda_host_[1], mha_k, stream);
      WeightedAdd<DataType>(kv_total, mha_v, exo_lambda_host_[0], exo_v_anc,
                            exo_lambda_host_[1], mha_v, stream);
    }
  }

  {
    char _l[64];
    snprintf(_l, 64, "[layer %d] Q_post_exo", _lid);
    debugDump(_l, mha_q, 8, stream);
    if (_lid == 0) {
      debugScanNaN("enc00_Q_post_exo_sq0", mha_q, d_model, stream);
    }
  }

  // Apply split_heads() to q, k and v
  // which basically transposes (batch_size, 64, num_heads, depth)
  // to (batch_size, num_heads, 64, depth)
  // Do we really need to transpose here?
  // (Maybe not, we can play with strides of the gemm and do independent gemms
  // for each encoder head)

  // Apply scaled dot product attention:
  /*
      matmul_qk = tf.matmul(q, k, transpose_b=True)
      dk = tf.cast(tf.shape(k)[-1], self.model_dtype)
      scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
      attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
      output = tf.matmul(attention_weights, v)
  */

  // shape(k)[-1] = depth
  float factor = 1.0f / sqrt((float)depth);

  // Resolve the smolgen-bias source: when ms_smol dispatched smolgen on
  // smolgen_stream, read from the dedicated smol_gen_out buffer; otherwise
  // fall back to the legacy in-place buffer2 output.  Non-const because
  // downstream consumers (fusedMHA, FusedAttention64, Softmax) have mixed
  // const / non-const / void* signatures.
  DataType* smol_bias =
      ms_smol ? smol_gen_out : (has_smolgen_ ? buffer2 : nullptr);

  // Main stream must wait for smolgen_stream to finish writing smol_gen_out
  // before any attention kernel can consume it as a bias.  The event was
  // recorded at the end of the ms_smol block above.
  if (ms_smol) {
    ReportCUDAErrors(cudaStreamWaitEvent(stream, smol_done_event, 0));
  }

  // One-shot diagnostic: report which attention path engaged on layer 0 of
  // the first forward pass.  LC0_MHA_PATH_LOG=1 enables it; stderr output.
  // Remove this block once verified — it's purely diagnostic.
  {
    static const bool kMhaPathLog =
        std::getenv("LC0_MHA_PATH_LOG") != nullptr;
    static std::atomic<bool> s_mha_path_logged{false};
    bool was_logged = s_mha_path_logged.exchange(true);
    if (kMhaPathLog && !was_logged && _lid == 0) {
#ifdef USE_CUTLASS
      bool cutlass_compiled = true;
#else
      bool cutlass_compiled = false;
#endif
      const bool cutlass_gate_pass =
          cutlass_compiled && use_fused_mha_ && attn_logit_cap_ == 0.0f;
      fprintf(stderr,
          "[LC0_MHA_PATH_LOG] USE_CUTLASS=%d use_fused_mha_=%d "
          "attn_logit_cap_=%.3f -> %s\n",
          (int)cutlass_compiled, (int)use_fused_mha_, attn_logit_cap_,
          cutlass_gate_pass ? "CUTLASS fusedMHA" : "cuBLAS softmax fallback");
      fflush(stderr);
    }
  }

#ifdef USE_CUTLASS
  if (use_fused_mha_ && attn_logit_cap_ == 0.0f &&
      smolgen_softcap_ == 0.0f) {
    // CUTLASS fused MHA hardcodes 1/sqrt(d) scaling internally and has no
    // softcap support.  Routed here only when:
    //   - attention softcap is disabled (cap == 0), AND
    //   - smolgen-bias softcap is disabled (cap == 0). Smolgen cap requires
    //     pre-add tanh on input2 which CUTLASS can't express; non-zero falls
    //     through to the Softmax-kernel path below which fuses both caps.
    fusedMHA(buffer2, mha_q, mha_k, mha_v, smol_bias, N,
             encoder_heads_, depth, stream);
  } else
#endif
  // ── Attention: Q×K^T + bias → softmax → ×V → output ──
  {
    // Use fused kernel when possible: no RPE, no residual attention.
    // These features require the full 64×64 logits in global memory.
    // Fused attention disabled: current kernel is slower than cuBLAS batched GEMMs
    // due to low occupancy (64 threads/block, heavy register pressure).
    // Needs tiled parallelization to compete. Keep code for future optimization.
    const bool can_fuse = false;
    (void)can_fuse;  // suppress unused warning

    if (can_fuse && use_fused_mha_) {
      // Fused path: entire attention in one kernel via shared memory.
      // SmolGen bias comes from smol_gen_out (ms_smol) or buffer2 (legacy).
      FusedAttention64<DataType>(
          N, encoder_heads_, depth, d_model,
          buffer1, mha_q, mha_k, mha_v,
          smol_bias,
          stream);
      // Swap: attention output now in buffer1, move to buffer2 for downstream.
      std::swap(buffer1, buffer2);
    } else {
      // Non-fused path: separate GEMM + softmax + GEMM (for residual attn).
      if (*offset_pointers == nullptr) {
#ifndef NDEBUG
        cudaStreamCaptureStatus capture;
        ReportCUDAErrors(cudaStreamIsCapturing(stream, &capture));
        assert(capture !=
                   cudaStreamCaptureStatus::cudaStreamCaptureStatusActive &&
               "Stream capture is active, cannot allocate memory for offset "
               "pointers");
#endif
        ReportCUDAErrors(
            cudaMalloc((void**)offset_pointers,
                       encoder_heads_ * max_batch_size_ * 5 * sizeof(DataType*)));
        genOffsetPointers((DataType**)*offset_pointers, encoder_heads_,
                          max_batch_size_, depth, d_model, mha_k, mha_q,
                          buffer1, mha_v, buffer2, stream);
      }

      // matmul_qk = Q @ K^T -> buffer1
      cublasXGemmBatched<DataType>(
          cublas, CUBLAS_OP_T, CUBLAS_OP_N, 64, 64, depth, factor,
          *offset_pointers,
          d_model,
          *offset_pointers + encoder_heads_ * max_batch_size_,
          d_model,
          0.0f,
          *offset_pointers + encoder_heads_ * max_batch_size_ * 2,
          64,
          N * encoder_heads_);

      Softmax(encoder_heads_ * N * 64, 64, buffer1, buffer1,
              smol_bias, stream, attn_logit_cap_, smolgen_softcap_);

      // attn_weights @ V -> buffer2
      cublasXGemmBatched<DataType>(
          cublas, CUBLAS_OP_N, CUBLAS_OP_N, depth, 64, 64, 1.0f,
          *offset_pointers + encoder_heads_ * max_batch_size_ * 3,
          d_model,
          *offset_pointers + encoder_heads_ * max_batch_size_ * 2,
          64,
          0.0f,
          *offset_pointers + encoder_heads_ * max_batch_size_ * 4,
          d_model,
          N * encoder_heads_);
    }
  }

  {
    char _l[64];
    snprintf(_l, 64, "[layer %d] attn_out", _lid);
    debugDump(_l, buffer2, 8, stream);
    if (_lid == 0) {
      debugScanNaN("enc00_attn_out_sq0", buffer2, d_model, stream);
    }
  }

  // VGA-E: element-wise gate on attention output.
  // buffer2 (attention output) *= sigmoid(Wg * input + bg)
  if (has_vga_elem_) {
    if (use_bank && has_bank_vga_) {
      // Bank-adapter VGA-E: gate pre-activation comes from the shared
      // bank (diag ⊙ h + b + folded rank-r mixer) — no gate GEMM at all.
      FusedVGAEBank<DataType>(N * 64 * d_model, buffer2, bank_h_buf,
                              gate_bank_size_, bank_lrb_vga,
                              bank_vga_diag_, bank_vga_bias_, d_model,
                              stream);
    } else if (vga_gate_buf_) {
      // Pre-norm: vga_gate_buf_ holds raw GEMM logits (bias+sigmoid deferred).
      // FusedVGAE applies bias+sigmoid+multiply in a single kernel, replacing
      // the separate addVectors(sigmoid) + ElementwiseMultiply pair.
      FusedVGAE<DataType>(N * 64 * d_model, buffer2, vga_gate_buf_,
                          vga_elem_gate_b_, mha_q_size_, stream);
    } else {
      // Post-norm: compute gate logits from raw input (in_out_tensor) into
      // scratch, then FusedVGAE applies bias+sigmoid+multiply in a single
      // kernel — same fused pattern the pre-norm path uses, replacing the
      // separate addVectors(sigmoid) + ElementwiseMultiply pair.
      const int batch = N * 64;
      cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, d_model, batch,
                            embedding_op_size_, 1.0f,
                            (const DataType*)vga_elem_gate_w_,
                            embedding_op_size_, in_out_tensor, embedding_op_size_,
                            0.0f, scratch, d_model);
      FusedVGAE<DataType>(N * 64 * d_model, buffer2, scratch,
                          vga_elem_gate_b_, d_model, stream);
    }
  }

  {
    char _l[64];
    snprintf(_l, 64, "[layer %d] after_vgae", _lid);
    debugDump(_l, buffer2, 8, stream);
    if (_lid == 0) {
      debugScanNaN("enc00_after_vgae_sq0", buffer2, d_model, stream);
    }
  }

  // dense projection: buffer2 -> buffer1
  {
    const int num_inputs = d_model;
    const int num_outputs = embedding_op_size_;
    const int batch = N * 64;
    cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                num_inputs, 1.0f, (const DataType*)mha_dense_w, num_inputs,
                buffer2, num_inputs, 0.0f, buffer1, num_outputs);
  }

  if (is_prenorm_) {
    // ── Pre-Norm path ──
    // Add dense projection bias to buffer1 (attention sublayer output)
    addBiasBatched<DataType>(buffer1, buffer1, mha_dense_b, 1, N * 64,
                             embedding_op_size_, ACTIVATION_NONE, stream);
    {
      char _l[64];
      snprintf(_l, 64, "[layer %d] dense_out", _lid);
      debugDump(_l, buffer1, 8, stream);
      if (_lid == 0) {
        debugScanNaN("enc00_dense_out_sq0", buffer1,
                     embedding_op_size_, stream);
      }
    }

    if (is_parallel_ffn_) {
      // ── Parallel FFN (PaLM-style) ──
      // buffer1 = attention output (with bias).
      // FFN reads from LN1 output (same input as attention), not LN2.
      // No LN2, no intermediate residual add.
      // If ln1_cache_ is available, FFN will read from it directly (no recompute).
      // Otherwise recompute LN1 into scratch.
      if (!ln1_cache_) {
        NormLayer<DataType>(use_rms_norm_, N * 64, embedding_op_size_, scratch,
                            in_out_tensor, (DataType*)nullptr, (DataType*)nullptr,
                            ln1_gammas, ln1_betas, default_eps_, 1.0,
                            ACTIVATION_NONE, stream);
      }
    } else {
      // ── Sequential FFN (standard) ──
      // Residual add + LN2 before FFN
      addVectors(in_out_tensor, in_out_tensor, buffer1,
                 N * 64 * embedding_op_size_, N * 64 * embedding_op_size_,
                 N * 64 * embedding_op_size_, ACTIVATION_NONE, stream);
      {
        char _l[64];
        snprintf(_l, 64, "[layer %d] mha_res", _lid);
        debugDump(_l, in_out_tensor, 8, stream);
        if (_lid == 0) {
          debugScanNaN("enc00_mha_res_sq0", in_out_tensor,
                       embedding_op_size_, stream);
        }
      }
      NormLayer<DataType>(use_rms_norm_, N * 64, embedding_op_size_, scratch,
                          in_out_tensor, (DataType*)nullptr, (DataType*)nullptr,
                          ln2_gammas, ln2_betas, default_eps_, 1.0,
                          ACTIVATION_NONE, stream);
      {
        char _l[64];
        snprintf(_l, 64, "[layer %d] ln2_out", _lid);
        debugDump(_l, scratch, 8, stream);
        if (_lid == 0) {
          debugScanNaN("enc00_ln2_out_sq0", scratch,
                       embedding_op_size_, stream);
        }
      }
    }
    // Determine FFN input source:
    // - Sequential: scratch (holds LN2 output)
    // - Parallel + ln1_cache_: read from ln1_cache_ directly (no recompute)
    // - Parallel + no cache (OOM fallback): scratch (holds recomputed LN1)
    const DataType* ffn_input =
        (is_parallel_ffn_ && ln1_cache_) ? ln1_cache_ : scratch;

    // For parallel FFN with SwiGLU: add attn_out (buffer1) to residual NOW
    // to free buffer1 for the SwiGLU gate_proj computation.
    // Then FFN output just needs a single residual add at the end.
    // Skip this when multi-stream FFN is active — we'll do a fused 3-way add
    // at the end instead.
    if (is_parallel_ffn_ && !ms_ffn) {
      addVectors(in_out_tensor, in_out_tensor, buffer1,
                 N * 64 * embedding_op_size_, N * 64 * embedding_op_size_,
                 N * 64 * embedding_op_size_, ACTIVATION_NONE, stream);
    }

    // Multi-stream: FFN already running (or its External placeholder captured).
    // Wait for FFN completion before the 3-way add.
    if (ms_ffn) {
      // Wait for FFN stream to finish before the 3-way residual add.
      ReportCUDAErrors(cudaStreamWaitEvent(stream, ffn_done_event, 0));
      // Fused 3-way add: in_out = in_out + buffer1 (attn) + ffn_buf_out (ffn)
      const int total = N * 64 * embedding_op_size_;
      Add3<DataType>(total, in_out_tensor, in_out_tensor, buffer1,
                     ffn_buf_out, stream);
      return;  // Done with this layer in multi-stream mode
    }

    if (has_swiglu_) {
      const int num_inputs = embedding_op_size_;
      const int dff = ffn_dff_;
      const int batch = N * 64;
      // Prefer fused path: single concatenated [gate; up] GEMM + fused
      // SwiGLUFusedGateUp kernel. Falls back to two-GEMM path if the
      // concatenated weight failed to allocate during construction.
      const bool can_fuse_gate_up = (ffn_gate_up_w_ != nullptr);
      if (can_fuse_gate_up) {
        // Single GEMM: [gate_w; up_w] @ ffn_input → scratch (size 2*dff*batch)
        cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, 2 * dff, batch,
                    num_inputs, 1.0f, (const DataType*)ffn_gate_up_w_,
                    num_inputs, ffn_input, num_inputs, 0.0f, scratch, 2 * dff);
        // Fused kernel: reads interleaved gate/up, adds biases, silu*up → buffer1
        // swiglu_softcap_ > 0 caps silu*up product element-wise.
        SwiGLUFusedGateUp<DataType>(batch, dff, buffer1, scratch,
                                     ffn_gate_b_, ffn_up_b_, stream,
                                     swiglu_softcap_, ffn_pgb_);
      } else {
        // Fallback: 2 separate GEMMs
        cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, dff, batch, num_inputs,
                    1.0f, (const DataType*)ffn_gate_w_, num_inputs, ffn_input,
                    num_inputs, 0.0f, buffer1, dff);
        cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, dff, batch, num_inputs,
                    1.0f, (const DataType*)ffn_up_w_, num_inputs, ffn_input,
                    num_inputs, 0.0f, buffer2, dff);
        // Fused bias+silu+mult: saves 2 addBiasBatched kernel launches.
        // pgb_ffn folded in via ffn_pgb_ (post-mult, post-softcap).
        SwiGLUElementwiseWithBias<DataType>(batch * dff, dff, buffer1, buffer1,
                                             buffer2, ffn_gate_b_, ffn_up_b_,
                                             stream, swiglu_softcap_,
                                             ffn_pgb_);
      }
      {
        char _l[64];
        snprintf(_l, 64, "[layer %d] ffn_swiglu_out", _lid);
        debugDump(_l, buffer1, 8, stream);
      }
      // pgb_ffn fused into the SwiGLU kernel above; no separate add needed.
      {
        char _l[64];
        snprintf(_l, 64, "[layer %d] ffn_swiglu", _lid);
        debugDump(_l, buffer1, 8, stream);
      }
      cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_inputs, batch, dff,
                  1.0f, (const DataType*)ffn_down_w_, dff, buffer1, dff,
                  0.0f, buffer2, num_inputs);
      if (ffn_down_b_)
        addBiasBatched(buffer2, buffer2, ffn_down_b_, 1, batch,
                       num_inputs, ACTIVATION_NONE, stream);
      {
        char _l[64];
        snprintf(_l, 64, "[layer %d] ffn_out", _lid);
        debugDump(_l, buffer2, 8, stream);
        if (_lid == 0) {
          debugScanNaN("enc00_ffn_out_sq0", buffer2,
                       embedding_op_size_, stream);
        }
      }
      // FFN output is in buffer2. Add to residual.
      // (For parallel FFN, attn_out was already added to in_out above.)
      addVectors(in_out_tensor, in_out_tensor, buffer2,
                 N * 64 * embedding_op_size_, N * 64 * embedding_op_size_,
                 N * 64 * embedding_op_size_, ACTIVATION_NONE, stream);
    } else {
      // Standard (non-SwiGLU) FFN. Reads from ffn_input (LN1 cache, scratch, etc).
      // For parallel FFN, attn_out was already added to in_out, buffer1 is free.
      const int batch = N * 64;
      cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, ffn_dense1_size_, batch,
                  embedding_op_size_, 1.0f, (const DataType*)ffn_dense1_w,
                  embedding_op_size_, ffn_input, embedding_op_size_, 0.0f,
                  buffer1, ffn_dense1_size_);
      addBiasBatched(buffer1, buffer1, ffn_dense1_b, 1, batch,
                     ffn_dense1_size_, ffn_activation_, stream);
      // PGB on FFN hidden (post-activation, before dense2).
      if (ffn_pgb_) {
        addBiasBatched(buffer1, buffer1, ffn_pgb_, 1, batch,
                       ffn_dense1_size_, ACTIVATION_NONE, stream);
      }
      cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, embedding_op_size_, batch,
                  ffn_dense1_size_, 1.0f, (const DataType*)ffn_dense2_w,
                  ffn_dense1_size_, buffer1, ffn_dense1_size_, 0.0f,
                  buffer2, embedding_op_size_);
      addBiasBatched(buffer2, buffer2, ffn_dense2_b, 1, batch,
                     embedding_op_size_, ACTIVATION_NONE, stream);
      // FFN residual add (attn already in in_out for parallel, or sequential)
      addVectors(in_out_tensor, in_out_tensor, buffer2,
                 N * 64 * embedding_op_size_, N * 64 * embedding_op_size_,
                 N * 64 * embedding_op_size_, ACTIVATION_NONE, stream);
    }
  } else if (is_parallel_ffn_) {
    // ── Post-Norm + Parallel FFN (PaLM-style sublayers under DeepNorm) ──
    //
    // PyTorch equivalent:
    //   ffn_output = self.ffn(inputs)          # FFN reads RAW residual
    //   delta = (attn_output + ffn_output) * alpha
    //   out2  = ln1(inputs + delta)            # single LN per block
    //
    // Differences from sequential post-norm:
    //   - FFN reads from raw in_out_tensor, NOT from post_ln1 output
    //   - No LN2; single LN merges both sublayer deltas with residual
    //   - buffer2 is free after MHA's dense projection; we use it for FFN
    //
    // At entry:
    //   buffer1 = attention output (BEFORE mha_dense_b)
    //   buffer2 = free (dense proj input; already consumed)
    //   scratch = free (previous MHA temps no longer needed)
    //   in_out_tensor = raw residual (unchanged since layer entry)
    //
    // We add mha_dense_b and ffn_down_b along the way so the final LN
    // kernel applies the combined attn+ffn delta scaled by alpha on top
    // of the residual in a single fused pass.
    const int batch = N * 64;
    const int emb = embedding_op_size_;
    // FFN output source for the combined LN below.  When ms_ffn is active,
    // the FFN ran concurrently on ffn_stream and wrote into ffn_buf_out
    // (with ffn_down_b / ffn_dense2_b baked in by the ffn_stream path).
    // Otherwise we compute FFN here on the main stream into buffer2.
    const DataType* ffn_out_ptr = ms_ffn ? ffn_buf_out : buffer2;
    if (!ms_ffn) {
      if (has_swiglu_) {
        const int dff = ffn_dff_;
        // gate_proj(in_out_tensor) -> scratch
        cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, dff, batch, emb, 1.0f,
                    (const DataType*)ffn_gate_w_, emb, in_out_tensor, emb,
                    0.0f, scratch, dff);
        // up_proj(in_out_tensor) -> buffer2
        cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, dff, batch, emb, 1.0f,
                    (const DataType*)ffn_up_w_, emb, in_out_tensor, emb, 0.0f,
                    buffer2, dff);
        // Fused bias+silu+mult+pgb -> scratch (in-place).  pgb_ffn folded
        // into the SwiGLU kernel via ffn_pgb_ (post-multiply, post-softcap).
        SwiGLUElementwiseWithBias<DataType>(batch * dff, dff, scratch, scratch,
                                             buffer2, ffn_gate_b_, ffn_up_b_,
                                             stream, swiglu_softcap_,
                                             ffn_pgb_);
        // down_proj(scratch) -> buffer2 (FFN output without bias)
        cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, emb, batch, dff, 1.0f,
                    (const DataType*)ffn_down_w_, dff, scratch, dff, 0.0f,
                    buffer2, emb);
        // Post-Norm + parallel FFN: ffn_down_b_ is folded into the combined
        // bias attn_ffn_combined_bias_ consumed by NormLayer below.  Skip
        // the separate add when that fusion is active.
        if (attn_ffn_combined_bias_ == nullptr) {
          addBiasBatched(buffer2, buffer2, ffn_down_b_, 1, batch, emb,
                         ACTIVATION_NONE, stream);
        }
      } else {
        // Standard FFN (non-SwiGLU).
        const int dff = ffn_dense1_size_;
        cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, dff, batch, emb, 1.0f,
                    (const DataType*)ffn_dense1_w, emb, in_out_tensor, emb,
                    0.0f, scratch, dff);
        addBiasBatched(scratch, scratch, ffn_dense1_b, 1, batch, dff,
                       ffn_activation_, stream);
        // PGB on FFN hidden (post-activation, before dense2).
        if (ffn_pgb_) {
          addBiasBatched(scratch, scratch, ffn_pgb_, 1, batch, dff,
                         ACTIVATION_NONE, stream);
        }
        cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, emb, batch, dff, 1.0f,
                    (const DataType*)ffn_dense2_w, dff, scratch, dff, 0.0f,
                    buffer2, emb);
        // Standard FFN: same combined-bias fusion as SwiGLU above.
        if (attn_ffn_combined_bias_ == nullptr) {
          addBiasBatched(buffer2, buffer2, ffn_dense2_b, 1, batch, emb,
                         ACTIVATION_NONE, stream);
        }
      }
    } else {
      // ms_ffn active: main stream must wait for ffn_stream to write
      // ffn_buf_out before the combined LN below reads it as input2.
      ReportCUDAErrors(cudaStreamWaitEvent(stream, ffn_done_event, 0));
    }
    // Fully fused: LN(alpha * (buffer1 + ffn_out_ptr + mha_dense_b) + in_out).
    // buffer1 holds attn-before-bias, ffn_out_ptr holds ffn-with-bias (either
    // main-stream buffer2 or ffn_stream's ffn_buf_out).  Passing it as input2
    // saves an extra addVectors kernel.
    //
    // When attn_ffn_combined_bias_ is set (Post-Norm parallel FFN with both
    // mha_dense_b and ffn_*_final_b available), ffn_out_ptr was written
    // WITHOUT its bias.  Pass the combined bias so the LN absorbs both —
    // saves one addBiasBatched launch per layer on the FFN bottleneck stream.
    const DataType* fused_ln_bias =
        attn_ffn_combined_bias_ ? attn_ffn_combined_bias_ : mha_dense_b;
    NormLayer<DataType>(use_rms_norm_, N * 64, emb, in_out_tensor, buffer1,
                        fused_ln_bias, in_out_tensor, ln1_gammas, ln1_betas,
                        default_eps_, alpha_, ACTIVATION_NONE, stream,
                        ffn_out_ptr);
    {
      char _l[64];
      snprintf(_l, 64, "[layer %d] post_ln_parallel", _lid);
      debugDump(_l, in_out_tensor, 8, stream);
    }
  } else {
    // ── Post-Norm path (sequential FFN) ──
    // LN1: LN((sublayer + bias) * alpha + residual) -> scratch
    // PyTorch: out1 = ln1(inputs + attn_output * alpha)
    NormLayer<DataType>(use_rms_norm_, N * 64, embedding_op_size_, scratch,
                        buffer1, mha_dense_b, in_out_tensor, ln1_gammas,
                        ln1_betas, default_eps_, alpha_, ACTIVATION_NONE,
                        stream);
    {
      char _l[64];
      snprintf(_l, 64, "[layer %d] post_ln1", _lid);
      debugDump(_l, scratch, 8, stream);
    }

    if (has_swiglu_) {
      const int num_inputs = embedding_op_size_;
      const int dff = ffn_dff_;
      const int batch = N * 64;
      // gate_proj(scratch) -> buffer1
      cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, dff, batch, num_inputs,
                  1.0f, (const DataType*)ffn_gate_w_, num_inputs, scratch,
                  num_inputs, 0.0f, buffer1, dff);
      // up_proj(scratch) -> buffer2 (free at this point)
      cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, dff, batch, num_inputs,
                  1.0f, (const DataType*)ffn_up_w_, num_inputs, scratch,
                  num_inputs, 0.0f, buffer2, dff);
      // Fused bias+silu+mult+pgb -> buffer1 (pgb_ffn folded in).
      SwiGLUElementwiseWithBias<DataType>(batch * dff, dff, buffer1, buffer1,
                                           buffer2, ffn_gate_b_, ffn_up_b_,
                                           stream, swiglu_softcap_,
                                           ffn_pgb_);
      // down_proj(buffer1) -> buffer2
      cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_inputs, batch, dff,
                  1.0f, (const DataType*)ffn_down_w_, dff, buffer1, dff,
                  0.0f, buffer2, num_inputs);
      // LN2: LN((ffn_out + bias) * alpha + mha_res) -> in_out_tensor
      NormLayer<DataType>(use_rms_norm_, N * 64, embedding_op_size_,
                          in_out_tensor, buffer2, ffn_down_b_, scratch,
                          ln2_gammas, ln2_betas, default_eps_, alpha_,
                          ACTIVATION_NONE, stream);
      {
        char _l[64];
        snprintf(_l, 64, "[layer %d] post_ln2", _lid);
        debugDump(_l, in_out_tensor, 8, stream);
      }
    } else {
      // Standard FFN
      // FFN dense 1, scratch -> in_out_tensor
      {
        const int num_inputs = embedding_op_size_;
        const int num_outputs = ffn_dense1_size_;
        const int batch = N * 64;
        cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                    num_inputs, 1.0f, (const DataType*)ffn_dense1_w, num_inputs,
                    scratch, num_inputs, 0.0f, in_out_tensor, num_outputs);
        addBiasBatched(in_out_tensor, in_out_tensor, ffn_dense1_b, 1, batch,
                       num_outputs, ffn_activation_, stream);
        // PGB on FFN hidden (post-activation, before dense2).
        if (ffn_pgb_) {
          addBiasBatched(in_out_tensor, in_out_tensor, ffn_pgb_, 1, batch,
                         num_outputs, ACTIVATION_NONE, stream);
        }
      }

      // FFN dense 2, in_out_tensor -> buffer1
      {
        const int num_inputs = ffn_dense1_size_;
        const int num_outputs = embedding_op_size_;
        const int batch = N * 64;
        cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                    num_inputs, 1.0f, (const DataType*)ffn_dense2_w, num_inputs,
                    in_out_tensor, num_inputs, 0.0f, buffer1, num_outputs);
      }

      // LN2: LN((ffn_out + bias) * alpha + mha_res) -> in_out_tensor
      NormLayer<DataType>(use_rms_norm_, N * 64, embedding_op_size_,
                          in_out_tensor, buffer1, ffn_dense2_b, scratch,
                          ln2_gammas, ln2_betas, default_eps_, alpha_,
                          ACTIVATION_NONE, stream);
      {
        char _l[64];
        snprintf(_l, 64, "[layer %d] post_ln2", _lid);
        debugDump(_l, in_out_tensor, 8, stream);
      }
    }
  }
}

template <typename DataType>
void AttentionPolicyHead<DataType>::Eval(
    int N, DataType* output, const DataType* input, const DataType* input2,
    void* scratch, size_t scratch_size, cudnnHandle_t /*cudnn*/,
    cublasHandle_t cublas, cudaStream_t stream, DataType*** offset_pointers) {
  DataType* input2_tensor = (DataType*)input2;
  DataType* buffer1 = output + scratch_size / (2 * sizeof(DataType));
  DataType* buffer2 = input2_tensor + scratch_size / (2 * sizeof(DataType));

  int inputC = this->input_->GetC();
  bool input_nhwc = attention_body_ || this->input_->isNHWC();
  if (!input_nhwc)
    convertNCHWtoNHWC((DataType*)scratch, input, N, inputC, N, inputC, 8, 8,
                      stream);

  // 1. Policy embedding (fully connected layer)
  // Input data in NHWC layout N*(64)*C, output is N*(64)*embedding_op_size_
  DataType* pol_embedding = input2_tensor;
  {
    const int num_outputs = embedding_op_size_;
    const int num_inputs = inputC;
    const int batch = N * 64;
    cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                          num_inputs, 1.0f, (const DataType*)ip_pol_w_,
                          num_inputs,
                          input_nhwc ? input : (DataType*)scratch,
                          num_inputs, 0.0f, pol_embedding, num_outputs);
    addBiasBatched(pol_embedding, pol_embedding, ip_pol_b_, 1, batch,
                   num_outputs, act_, stream);
  }
  debugDump("pol_embedding", pol_embedding, 8, stream);

  // 2. Encoder layers
  for (const auto pEnc : encoder_weights_) {
    pEnc->Eval(N, input2_tensor, (DataType*)scratch, buffer1, buffer2, cublas,
               stream, offset_pointers);
  }  // End of encoder blocks

  DataType* wq;
  DataType* wk;
  {
    const int num_inputs = embedding_op_size_;
    const int num_outputs = policy_d_model_;
    const int batch = N * 64;
    wq = (DataType*)scratch;
    wk = wq + num_outputs * batch;

    cublasXGemmStridedBatched<DataType>(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch, num_inputs, 1.0f,
        wqk_w_, num_inputs, num_inputs * num_outputs, input2_tensor, num_inputs,
        0, 0.0f, wq, num_outputs, num_outputs * batch, 2, use_gemm_ex_);

    addBiasBatched<DataType>(wq, wq, wqk_b_, 2, batch, num_outputs,
                             ACTIVATION_NONE, stream);
  }

  debugDump("pol_Q", wq, 8, stream);
  debugDump("pol_K", wk, 8, stream);

  // dk = tf.math.sqrt(tf.cast(tf.shape(keys)[-1], self.model_dtype))
  // policy matmul_qk = tf.matmul(queries, keys, transpose_b=True)
  // policy_attn_logits = matmul_qk / dk
  {
    // shape(keys)[-1] = policy_d_model_
    float factor = 1.0f / sqrt((float)policy_d_model_);

    // A/B, and M/N are swapped for row-major to col-major transform
    // leave 8*24 after each batch to interleave promotion_logits (computed
    // later below)
    cublasXGemmStridedBatched<DataType>(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N, 64 /*M*/, 64 /*N*/,
        policy_d_model_ /*K*/,
        factor,  // to handle "/ tf.math.sqrt(dk)"
        wk /*A*/, policy_d_model_ /*LDA*/, 64 * policy_d_model_, /*strideA*/
        wq /*B*/, policy_d_model_ /*LDB*/, 64 * policy_d_model_, /*strideB*/
        0.0f, output /*C*/,  // output (policy_attn_logits)
        64 /*LDC*/, 64 * 64 + 8 * 24 /*strideC*/, N, use_gemm_ex_);
  }

  // Compute promotion_logits in a single kernel (and put the result just after
  // policy_attn_logits interleaved to get concat for free)
  DataType* promotion_logits = output + 64 * 64;

  ComputePromotionLogits<DataType>(N, policy_d_model_, promotion_logits, wk,
                                   ip4_pol_w_, output, stream);
}

template <typename DataType>
AttentionPolicyHead<DataType>::~AttentionPolicyHead() {
  ReportCUDAErrors(cudaFree(ip_pol_w_));
  ReportCUDAErrors(cudaFree(ip_pol_b_));
  ReportCUDAErrors(cudaFree(ip2_pol_w_));
  ReportCUDAErrors(cudaFree(ip2_pol_b_));
  ReportCUDAErrors(cudaFree(ip3_pol_w_));
  ReportCUDAErrors(cudaFree(ip3_pol_b_));
  ReportCUDAErrors(cudaFree(ip4_pol_w_));
  ReportCUDAErrors(cudaFree(wqk_w_));
  ReportCUDAErrors(cudaFree(wqk_b_));
  for (const auto pEnc : encoder_weights_) delete pEnc;
}

template <typename DataType>
EncoderBlock<DataType>::~EncoderBlock() {
  // All weight buffers below are allocated via allocAndUpload(),
  // which routes through the arena when one is active.  FreeWeight()
  // skips cudaFree on arena-owned pointers (the arena frees its
  // chunks once when ~WeightArena runs).  For non-arena
  // construction, FreeWeight() degenerates to cudaFree() — identical
  // to the pre-Phase-B.3 behavior.
  //
  // Runtime buffers (ln1_cache_, vga_gate_buf_, mha_qkv_fused_w_,
  // attn_ffn_combined_bias_) are allocated via direct cudaMalloc,
  // NOT via the arena.  FreeWeight() correctly cudaFree's them
  // because arena->Owns() returns false for these (the pointers
  // are outside the arena's chunks).
  FreeWeight(mha_q_w, arena_);
  FreeWeight(mha_q_b, arena_);
  FreeWeight(mha_k_w, arena_);
  FreeWeight(mha_k_b, arena_);
  FreeWeight(mha_v_w, arena_);
  FreeWeight(mha_v_b, arena_);
  FreeWeight(mha_qkv_w, arena_);
  FreeWeight(mha_qkv_b, arena_);
  FreeWeight(mha_dense_w, arena_);
  FreeWeight(mha_dense_b, arena_);
  FreeWeight(ln1_gammas, arena_);
  FreeWeight(ln1_betas, arena_);
  FreeWeight(ffn_dense1_w, arena_);
  FreeWeight(ffn_dense1_b, arena_);
  FreeWeight(ffn_dense2_w, arena_);
  FreeWeight(ffn_dense2_b, arena_);
  FreeWeight(ln2_gammas, arena_);
  FreeWeight(ln2_betas, arena_);
  if (has_smolgen_) {
    FreeWeight(smol_compress, arena_);
    FreeWeight(smol_dense1_w, arena_);
    FreeWeight(smol_dense1_b, arena_);
    FreeWeight(smol_dense2_w, arena_);
    FreeWeight(smol_dense2_b, arena_);
    FreeWeight(smol_ln1_gammas, arena_);
    FreeWeight(smol_ln1_betas, arena_);
    FreeWeight(smol_ln2_gammas, arena_);
    FreeWeight(smol_ln2_betas, arena_);
  }
  FreeWeight(ffn_gate_up_w_, arena_);
  FreeWeight(ffn_gate_up_b_, arena_);
  FreeWeight(ffn_gate_w_, arena_);
  FreeWeight(ffn_gate_b_, arena_);
  FreeWeight(ffn_up_w_, arena_);
  FreeWeight(ffn_up_b_, arena_);
  FreeWeight(ffn_down_w_, arena_);
  FreeWeight(ffn_pgb_, arena_);
  FreeWeight(ffn_down_b_, arena_);
  FreeWeight(mha_q2_w_, arena_);
  FreeWeight(mha_q2_b_, arena_);
  FreeWeight(mha_k2_w_, arena_);
  FreeWeight(mha_k2_b_, arena_);
  FreeWeight(mha_q1k1_w_, arena_);
  FreeWeight(mha_q1k1_b_, arena_);
  FreeWeight(mha_vg_vu_w_, arena_);
  FreeWeight(mha_v_gate_w_, arena_);
  FreeWeight(mha_v_gate_b_, arena_);
  FreeWeight(mha_v_up_w_, arena_);
  FreeWeight(mha_v_up_b_, arena_);
  FreeWeight(pgb_v_, arena_);
  FreeWeight(vga_elem_gate_w_, arena_);
  FreeWeight(vga_elem_gate_b_, arena_);
  FreeWeight(vga_gate_buf_, arena_);          // runtime buffer (direct cudaMalloc)
  FreeWeight(ln1_cache_, arena_);             // runtime buffer (direct cudaMalloc)
  // GQA blend matrices (synthesized for plain GQA, or read from proto for
  // weighted-GQA).  Null when kv_heads == encoder_heads.
  FreeWeight(gqa_w_k_, arena_);
  FreeWeight(gqa_w_v_, arena_);
  FreeWeight(mha_kv_w_, arena_);
  FreeWeight(mha_kv_b_, arena_);
  FreeWeight(attn_ffn_combined_bias_, arena_);  // runtime buffer (direct cudaMalloc)
  FreeWeight(mha_qkv_fused_w_, arena_);          // built from arena slots, see ctor
  // Shared gate bank (A-Layout-2) weights.
  FreeWeight(gate_bank_w_, arena_);
  FreeWeight(gate_bank_b_, arena_);
  FreeWeight(bank_lr_a_w_, arena_);
  FreeWeight(bank_v_diag_, arena_);
  FreeWeight(bank_v_bias_, arena_);
  FreeWeight(bank_v_lr_b_w_, arena_);
  FreeWeight(bank_vga_diag_, arena_);
  FreeWeight(bank_vga_bias_, arena_);
  FreeWeight(bank_vga_lr_b_w_, arena_);
  FreeWeight(bank_ffn_diag_, arena_);
  FreeWeight(bank_ffn_bias_, arena_);
  FreeWeight(bank_ffn_lr_b_w_, arena_);
  FreeWeight(bank_q_diag_, arena_);
  FreeWeight(bank_q_bias_, arena_);
  FreeWeight(bank_q_lr_b_w_, arena_);
  FreeWeight(bank_k_diag_, arena_);
  FreeWeight(bank_k_bias_, arena_);
  FreeWeight(bank_k_lr_b_w_, arena_);
  // exo_lambda_ is host-side (exo_lambda_host_), no GPU free needed.
}

template <typename DataType>
EmbeddingLayer<DataType>::EmbeddingLayer(BaseLayer<DataType>* ip,
                                         const std::vector<float>& weights,
                                         const std::vector<float>& biases,
                                         void* scratch, ActivationFunction act)
    : BaseLayer<DataType>(biases.size(), 8, 8, ip), act_(act) {
  allocAndUpload<DataType>(&weights_, weights, scratch);
  allocAndUpload<DataType>(&biases_, biases, scratch);
}

template <typename DataType>
EmbeddingLayer<DataType>::~EmbeddingLayer() {
  ReportCUDAErrors(cudaFree(weights_));
  ReportCUDAErrors(cudaFree(biases_));
}

template <typename DataType>
void EmbeddingLayer<DataType>::Eval(
    int N, DataType* output, const DataType* input, const DataType* /*input2*/,
    void* /*scratch*/, size_t /*scratch_size*/, cudnnHandle_t /*cudnn*/,
    cublasHandle_t cublas, cudaStream_t stream, DataType***) {
  const int num_outputs = this->GetC();
  const int num_inputs = this->input_->GetC();
  const int batch = N * 64;
  cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                        num_inputs, 1.0f, weights_, num_inputs, input,
                        num_inputs, 0.0f, output, num_outputs);
  addBiasBatched(output, output, biases_, 1, batch, num_outputs, act_, stream);
}

template <typename DataType>
void EncoderBlock<DataType>::EvalFFNOnly(int N, cudaStream_t ffn_s,
                                          cublasHandle_t ffn_h,
                                          DataType* ffn_buf_wide,
                                          DataType* ffn_buf_out) const {
  const int batch = N * 64;
  const int num_inputs = embedding_op_size_;
  if (has_swiglu_) {
    // Prefer fused path: single [gate; up] GEMM + SwiGLUFusedGateUp. Falls
    // back to two-GEMM path if the concatenated weight failed to allocate.
    const int dff = ffn_dff_;
    DataType* hidden_out = ffn_buf_wide;  // dff-wide SwiGLU output region.
    if (ffn_gate_up_w_ != nullptr) {
      cublasXgemm(ffn_h, CUBLAS_OP_T, CUBLAS_OP_N, 2 * dff, batch, num_inputs,
                  1.0f, (const DataType*)ffn_gate_up_w_, num_inputs,
                  ln1_cache_, num_inputs, 0.0f, ffn_buf_wide, 2 * dff);
      SwiGLUFusedGateUp<DataType>(batch, dff, hidden_out, ffn_buf_wide,
                                   ffn_gate_b_, ffn_up_b_, ffn_s,
                                   swiglu_softcap_, ffn_pgb_);
    } else {
      DataType* gate_out = ffn_buf_wide;
      DataType* up_out   = ffn_buf_wide + (size_t)dff * batch;
      cublasXgemm(ffn_h, CUBLAS_OP_T, CUBLAS_OP_N, dff, batch, num_inputs,
                  1.0f, (const DataType*)ffn_gate_w_, num_inputs, ln1_cache_,
                  num_inputs, 0.0f, gate_out, dff);
      cublasXgemm(ffn_h, CUBLAS_OP_T, CUBLAS_OP_N, dff, batch, num_inputs,
                  1.0f, (const DataType*)ffn_up_w_, num_inputs, ln1_cache_,
                  num_inputs, 0.0f, up_out, dff);
      SwiGLUElementwiseWithBias<DataType>(
          batch * dff, dff, hidden_out, gate_out, up_out,
          ffn_gate_b_, ffn_up_b_, ffn_s, swiglu_softcap_, ffn_pgb_);
    }
    // pgb_ffn fused into the SwiGLU kernel above; no separate add needed.
    cublasXgemm(ffn_h, CUBLAS_OP_T, CUBLAS_OP_N, num_inputs, batch, dff,
                1.0f, (const DataType*)ffn_down_w_, dff, hidden_out, dff,
                0.0f, ffn_buf_out, num_inputs);
    if (ffn_down_b_) {
      addBiasBatched(ffn_buf_out, ffn_buf_out, ffn_down_b_, 1, batch,
                     num_inputs, ACTIVATION_NONE, ffn_s);
    }
  } else {
    const int dff = ffn_dense1_size_;
    cublasXgemm(ffn_h, CUBLAS_OP_T, CUBLAS_OP_N, dff, batch, num_inputs,
                1.0f, (const DataType*)ffn_dense1_w, num_inputs, ln1_cache_,
                num_inputs, 0.0f, ffn_buf_wide, dff);
    addBiasBatched(ffn_buf_wide, ffn_buf_wide, ffn_dense1_b, 1, batch,
                   dff, ffn_activation_, ffn_s);
    // PGB on FFN hidden (post-activation, before dense2).
    if (ffn_pgb_) {
      addBiasBatched(ffn_buf_wide, ffn_buf_wide, ffn_pgb_, 1, batch, dff,
                     ACTIVATION_NONE, ffn_s);
    }
    cublasXgemm(ffn_h, CUBLAS_OP_T, CUBLAS_OP_N, num_inputs, batch, dff,
                1.0f, (const DataType*)ffn_dense2_w, dff, ffn_buf_wide, dff,
                0.0f, ffn_buf_out, num_inputs);
    addBiasBatched(ffn_buf_out, ffn_buf_out, ffn_dense2_b, 1, batch,
                   num_inputs, ACTIVATION_NONE, ffn_s);
  }
}

template <typename DataType>
AttentionBody<DataType>::AttentionBody(const MultiHeadWeights& weights,
                                       void* scratch, Activations activations,
                                       int num_res_blocks, int input_c,
                                       int max_batch_size,
                                       bool is_pe_dense_embedding,
                                       bool use_gemm_ex, bool fused_mha)
    : BaseLayer<DataType>(weights.ip_emb_b.size(), 8, 8, nullptr, false,
                          use_gemm_ex),
      embedding_op_size_(weights.ip_emb_b.size()),
      encoder_head_count_(weights.encoder_head_count),
      activations_(activations),
      num_resi_blocks_(num_res_blocks),
      input_c_(input_c),
      has_gating_(weights.ip_mult_gate.size() > 0 &&
                  weights.ip_add_gate.size() > 0),
      has_smolgen_(weights.has_smolgen),
      is_pe_dense_embedding_(is_pe_dense_embedding),
      use_fused_mha_(fused_mha),
      is_prenorm_(weights.is_prenorm),
      use_rms_norm_(weights.use_rms_norm),
      has_swiglu_(weights.use_swiglu_ffn ||
                  (!weights.encoder.empty() &&
                   weights.encoder[0].ffn.gate_proj_w.size() > 0)),
      is_parallel_ffn_(weights.use_parallel_ffn),
      has_material_info_(weights.use_material_info),
      has_attack_maps_(weights.use_attack_maps),
      attn_logit_cap_(weights.attn_logit_cap),
      smolgen_softcap_(weights.smolgen_softcap),
      v_softcap_(weights.v_softcap),
      swiglu_softcap_(weights.swiglu_softcap),
      value_branch_at_layer_(weights.value_branch_at_layer) {
  // Capture the active weight arena (if any) so the destructor can
  // skip cudaFree on arena-owned buffers.  See WeightArena::Owns().
  // Captured before any allocAndUpload() call so every weight upload
  // during this ctor (including those in child EncoderBlock ctors
  // launched below) uses the same arena.
  arena_ = tl_weight_arena;

  allocAndUpload<DataType>(&ip_emb_w_, weights.ip_emb_w, scratch);
  allocAndUpload<DataType>(&ip_emb_b_, weights.ip_emb_b, scratch);

  // Rich embedding replaces dense preproc but shares LN/FFN.
  const bool has_rich_emb_early = weights.rich_emb_sq_w1.size() > 0;
  if (has_rich_emb_early) is_pe_dense_embedding_ = false;

  if (is_pe_dense_embedding_) {
    // Dense preproc path (standard embedding)
    allocAndUpload<DataType>(&ip_emb_pre_w_, weights.ip_emb_preproc_w, scratch);
    allocAndUpload<DataType>(&ip_emb_pre_b_, weights.ip_emb_preproc_b, scratch);
  } else if (!has_rich_emb_early) {
    // Positional encoding table (legacy: no dense preproc, no rich emb)
    size_t size = 64 * kNumPosEncodingChannels * sizeof(float);
    ReportCUDAErrors(cudaMalloc(&pos_encoding_, size));
    ReportCUDAErrors(
        cudaMemcpy(scratch, kPosEncoding, size, cudaMemcpyHostToDevice));
    copyTypeConverted(pos_encoding_, (float*)scratch, size, 0);
  }

  // Embedding LN, FFN, sizes — needed for dense preproc and rich.
  if (is_pe_dense_embedding_ || has_rich_emb_early) {
    allocAndUpload<DataType>(&ip_emb_ln_g_, weights.ip_emb_ln_gammas, scratch);
    allocAndUpload<DataType>(&ip_emb_ln_b_, weights.ip_emb_ln_betas, scratch);

    has_emb_ffn_swiglu_ = weights.ip_emb_ffn.gate_proj_w.size() > 0;
    if (has_emb_ffn_swiglu_) {
      allocAndUpload<DataType>(&ip_emb_ffn_gate_w_, weights.ip_emb_ffn.gate_proj_w, scratch);
      allocAndUpload<DataType>(&ip_emb_ffn_gate_b_, weights.ip_emb_ffn.gate_proj_b, scratch);
      allocAndUpload<DataType>(&ip_emb_ffn_up_w_, weights.ip_emb_ffn.up_proj_w, scratch);
      allocAndUpload<DataType>(&ip_emb_ffn_up_b_, weights.ip_emb_ffn.up_proj_b, scratch);
      allocAndUpload<DataType>(&ip_emb_ffn_down_w_, weights.ip_emb_ffn.down_proj_w, scratch);
      allocAndUpload<DataType>(&ip_emb_ffn_down_b_, weights.ip_emb_ffn.down_proj_b, scratch);
      ip_emb_ffn_d1_w_ = nullptr; ip_emb_ffn_d1_b_ = nullptr;
      ip_emb_ffn_d2_w_ = nullptr; ip_emb_ffn_d2_b_ = nullptr;
    } else {
      allocAndUpload<DataType>(&ip_emb_ffn_d1_w_, weights.ip_emb_ffn.dense1_w, scratch);
      allocAndUpload<DataType>(&ip_emb_ffn_d1_b_, weights.ip_emb_ffn.dense1_b, scratch);
      allocAndUpload<DataType>(&ip_emb_ffn_d2_w_, weights.ip_emb_ffn.dense2_w, scratch);
      allocAndUpload<DataType>(&ip_emb_ffn_d2_b_, weights.ip_emb_ffn.dense2_b, scratch);
    }
    allocAndUpload<DataType>(&ip_emb_ffn_ln_g_, weights.ip_emb_ffn_ln_gammas, scratch);
    allocAndUpload<DataType>(&ip_emb_ffn_ln_b_, weights.ip_emb_ffn_ln_betas, scratch);

    if (has_rich_emb_early) {
      embedding_dense_size_ = weights.rich_emb_sq_b2.size();
    } else {
      embedding_dense_size_ = weights.ip_emb_preproc_b.size() / 64;
    }
    if (has_emb_ffn_swiglu_) {
      embedding_ffn_size_ = embedding_op_size_;
      embedding_ffn_dff_ = (int)(weights.ip_emb_ffn.gate_proj_w.size() / embedding_op_size_);
    } else {
      embedding_ffn_size_ = weights.ip_emb_ffn.dense2_b.size();
      embedding_ffn_dff_ = weights.ip_emb_ffn.dense1_b.size();
    }
  }

  if (has_gating_) {
    allocAndUpload<DataType>(&ip_mult_gate_, weights.ip_mult_gate, scratch);
    allocAndUpload<DataType>(&ip_add_gate_, weights.ip_add_gate, scratch);
  }

  if (has_smolgen_) {
    allocAndUpload<DataType>(&smolgen_global_, weights.smolgen_w, scratch);
    smolgen_global_size_ = 64 * 64;
    // Smolgen dictionary (motif bank): shared atom bank + fused decoder.
    if (weights.smolgen_dict_p.size() > 0) {
      if (weights.smolgen_w.size() > 0) {
        throw Exception(
            "net has both smolgen_w and smolgen_dict_p — ambiguous "
            "smolgen decoder; the export should never produce this.");
      }
      if (weights.smolgen_dict_dec_w.size() == 0) {
        throw Exception(
            "smolgen_dict_p present but smolgen_dict_dec_w missing — "
            "incomplete export (sync torchprocess.py + regenerate "
            "net_pb2.py, then re-export).");
      }
      if (weights.smolgen_dict_p.size() % (64 * 64) != 0) {
        throw Exception("smolgen_dict_p size not a multiple of 4096");
      }
      smol_dict_m_ = (int)(weights.smolgen_dict_p.size() / (64 * 64));
      // Decoder rows = M + 2*64*r; columns = gen_sz (from layer 0's
      // dense2 width / head count).
      const auto& sg0 = weights.encoder[0].mha.smolgen;
      const int gen_sz = (int)(sg0.dense2_b.size() / encoder_head_count_);
      if (gen_sz <= 0 ||
          weights.smolgen_dict_dec_w.size() % gen_sz != 0) {
        throw Exception("smolgen_dict_dec_w size not divisible by gen_sz");
      }
      const int dec_rows = (int)(weights.smolgen_dict_dec_w.size() / gen_sz);
      // dec_rows == M is valid: rank-0 (mixture-only, no UVᵀ residual).
      if ((dec_rows - smol_dict_m_) % (2 * 64) != 0 ||
          dec_rows < smol_dict_m_) {
        throw Exception(
            "smolgen_dict_dec_w rows (" + std::to_string(dec_rows) +
            ") don't decompose as M + 2*64*r with M=" +
            std::to_string(smol_dict_m_));
      }
      smol_dict_rank_ = (dec_rows - smol_dict_m_) / (2 * 64);
      allocAndUpload<DataType>(&smol_dict_p_, weights.smolgen_dict_p,
                               scratch);
      allocAndUpload<DataType>(&smol_dict_dec_, weights.smolgen_dict_dec_w,
                               scratch);
    }
  }

  // Encoder final norm (Pre-Norm only)
  // Encoder final norm: historically wired only for pre-norm (where it's
  // required because pre-norm skips per-layer output LN).  Extended to
  // post-norm too — Python training can now enable `use_final_norm: true`
  // for post-norm nets as a cheap output-boundary clamp.  Presence of the
  // weight in the pb.gz is the single source of truth; the is_prenorm_
  // gate has been dropped.  For legacy post-norm nets that don't have
  // this weight populated, the size check below still skips allocation.
  if (weights.encoder_final_norm_gammas.size() > 0) {
    allocAndUpload<DataType>(&enc_final_norm_g_, weights.encoder_final_norm_gammas, scratch);
    allocAndUpload<DataType>(&enc_final_norm_b_, weights.encoder_final_norm_betas, scratch);
  }

  max_batch_size_ = max_batch_size;

  int num_encoders = weights.encoder.size();
  float alpha = is_prenorm_ ? 1.0f : (float)pow(2.0 * num_encoders, -0.25);
  // ExoFormer indicator for the temp-fold gate in EncoderBlock: body-level
  // weights are the single source of truth (per-layer exo_lambda is not
  // populated by the Python exporter).
  const bool body_has_exoformer = weights.exo_q_anc_w.size() > 0;
  for (const auto& enc : weights.encoder) {
    EncoderBlock<DataType>* pW = new EncoderBlock<DataType>(
        enc, scratch, encoder_head_count_, embedding_op_size_, alpha,
        smolgen_global_, smolgen_global_size_, max_batch_size,
        activations_.smolgen_activation, activations_.ffn_activation,
        1e-3, use_gemm_ex, use_fused_mha_,
        is_prenorm_, use_rms_norm_, has_swiglu_,
        is_parallel_ffn_, attn_logit_cap_,
        body_has_exoformer, smolgen_softcap_, v_softcap_, swiglu_softcap_,
        weights.kv_headcount,
        smol_dict_p_, smol_dict_dec_, smol_dict_m_, smol_dict_rank_);
    encoder_weights_.emplace_back(pW);
  }

  // Multi-exit value branch buffer. Allocated only when the proto's
  // value_branch_at_layer is in [0, num_encoders - 1). Captures the encoder
  // output at that layer's index during Eval, then has final_norm applied
  // (mirroring the main flow's final pass) so consumers see a fully-prepped
  // flow. Sized for max_batch × all-tokens × emb so it can hold the encoder
  // state including registers (stripping happens at consume time).
  if (value_branch_at_layer_ >= 0 &&
      value_branch_at_layer_ < (int)encoder_weights_.size() - 1) {
    const size_t branch_bytes = (size_t)max_batch_size * 64 *
                                 embedding_op_size_ * sizeof(DataType);
    auto err = cudaMalloc(&value_branch_buf_, branch_bytes);
    if (err != cudaSuccess) {
      value_branch_buf_ = nullptr;
      value_branch_at_layer_ = -1;  // disable on OOM, fall back to final flow
      fprintf(stderr,
              "[value_branch] OOM allocating buffer (%zu MB); disabled.\n",
              branch_bytes / (1024 * 1024));
    } else {
      ReportCUDAErrors(cudaMemset(value_branch_buf_, 0, branch_bytes));
      fprintf(stderr,
              "[value_branch] enabled at layer %d (encoder has %d layers); "
              "value heads will read flow at this layer.\n",
              value_branch_at_layer_, (int)encoder_weights_.size());
    }
  } else if (value_branch_at_layer_ >= 0) {
    // Set but out of range — disable cleanly with a warning.
    fprintf(stderr,
            "[value_branch] value_branch_at_layer=%d out of range "
            "[0, %d); disabled.\n",
            value_branch_at_layer_, (int)encoder_weights_.size() - 1);
    value_branch_at_layer_ = -1;
  }

  // Rich embedding: widened per-square MLP (piece + per-square extras).
  has_rich_emb_ = weights.rich_emb_sq_w1.size() > 0;
  if (has_rich_emb_) {
    // Override embedding_dense_size_ from rich emb output (sq_b2 size)
    embedding_dense_size_ = weights.rich_emb_sq_b2.size();
    allocAndUpload<DataType>(&rich_emb_sq_w1_, weights.rich_emb_sq_w1, scratch);
    allocAndUpload<DataType>(&rich_emb_sq_b1_, weights.rich_emb_sq_b1, scratch);
    allocAndUpload<DataType>(&rich_emb_sq_w2_, weights.rich_emb_sq_w2, scratch);
    allocAndUpload<DataType>(&rich_emb_sq_b2_, weights.rich_emb_sq_b2, scratch);
    allocAndUpload<DataType>(&rich_emb_global_w_, weights.rich_emb_global_w, scratch);
    allocAndUpload<DataType>(&rich_emb_global_b_, weights.rich_emb_global_b, scratch);
    rich_emb_global_sz_ = weights.rich_emb_global_b.size();

    // Infer per-square MLP dimensions from weight shapes:
    //   b1 is (hidden,)                → rich_hidden_sz_
    //   w1 is (hidden, per_sq_in)      → per_sq_in = w1.size / hidden
    rich_hidden_sz_ = weights.rich_emb_sq_b1.size();
    rich_per_sq_input_sz_ = weights.rich_emb_sq_w1.size() / rich_hidden_sz_;

    // Enriched-feature enablement is driven strictly by the NetworkFormat
    // flags (has_material_info_, has_attack_maps_) set in
    // the AttentionBody ctor initializer list. No weight-shape auto-detect:
    // if the pb.gz flags don't match the rich MLP weight shape, that's a
    // config error to fix on the training side.

    // Allocate per-square MLP input buffer (computed each forward).
    size_t buf_elems = (size_t)max_batch_size * 64 * rich_per_sq_input_sz_;
    ReportCUDAErrors(cudaMalloc(&rich_per_sq_buf_,
                                buf_elems * sizeof(DataType)));

  }

  // Main-embedding preprocess path is chosen strictly from NetworkFormat
  // flags: if has_material_info_/has_attack_maps_ are set,
  // the preprocess concatenates those extras before the main embedding
  // GEMM. No weight-shape auto-detection. A shape mismatch against the
  // concatenated layout indicates a config error on the training side.

  // ExoFormer: load anchor projection weights and allocate buffers.
  has_exoformer_ = weights.exo_q_anc_w.size() > 0;
  if (has_exoformer_) {
    allocAndUpload<DataType>(&exo_q_anc_w_, weights.exo_q_anc_w, scratch);
    allocAndUpload<DataType>(&exo_k_anc_w_, weights.exo_k_anc_w, scratch);
    allocAndUpload<DataType>(&exo_v_anc_w_, weights.exo_v_anc_w, scratch);

    // Single combined allocation for Q/K/V anchor buffers, laid out contiguously.
    // exo_q/k/v_anc_buf_ are pointer aliases into exo_all_anc_buf_.
    size_t anc_elems = (size_t)max_batch_size * 64 * embedding_op_size_;
    size_t anc_size  = anc_elems * sizeof(DataType);
    ReportCUDAErrors(cudaMalloc(&exo_all_anc_buf_, 3 * anc_size));
    exo_q_anc_buf_ = exo_all_anc_buf_;
    exo_k_anc_buf_ = exo_all_anc_buf_ + anc_elems;
    exo_v_anc_buf_ = exo_all_anc_buf_ + 2 * anc_elems;

    // Ones buffer for RMSNorm (scale=False)
    int depth = embedding_op_size_ / encoder_head_count_;
    std::vector<float> ones(depth, 1.0f);
    allocAndUpload<DataType>(&exo_norm_ones_, ones, scratch);
  }

  // Exo-as-smolgen-bias: load projection weight and allocate anchor buffer.
  // Detect presence by the weight vector being non-empty.  The training-
  // side 1/sqrt(N_layers) forward scaling is baked into the exported weight,
  // so at runtime we apply the projection as-is with no extra scaling.
  //
  // ENV-VAR KILL SWITCH: LC0_DISABLE_EXO_SMOL_BIAS=1 forces has_exo_smol_anchor_
  // to false EVEN IF the net has the weight populated, so NO new code runs
  // (no buffers allocated, no kernels launched).  Lets you isolate whether
  // any exo-bias code is involved in a crash.
  static const bool kExoSmolBiasCtorDisabled =
      std::getenv("LC0_DISABLE_EXO_SMOL_BIAS") != nullptr;
  has_exo_smol_anchor_ = weights.exo_smol_anchor_w.size() > 0
                         && !kExoSmolBiasCtorDisabled;
  if (has_exo_smol_anchor_) {
    allocAndUpload<DataType>(&exo_smol_anchor_w_,
                             weights.exo_smol_anchor_w, scratch);
    // Output buffer: (max_batch, H*gen_sz).  Shares shape with each
    // layer's smolgen gen_from output (post-LN), so we can add it
    // directly at the injection point.
    //
    // The H*gen_sz size comes from the first encoder's smol_dense_2_size_.
    // If no encoders have been built yet (weird failure mode) or smolgen
    // is not enabled, we still allocate using weight shape inferred size.
    size_t out_dim = 0;
    if (!encoder_weights_.empty() &&
        encoder_weights_[0]->smol_dense_2_size_ > 0) {
      out_dim = encoder_weights_[0]->smol_dense_2_size_;
    } else {
      // Fallback: infer from weight shape.  weight is (out, in) flat =
      // H*gen_sz * emb_size.  We know emb_size = embedding_op_size_.
      out_dim = weights.exo_smol_anchor_w.size() / embedding_op_size_;
    }
    size_t anchor_elems = (size_t)max_batch_size * out_dim;
    ReportCUDAErrors(cudaMalloc(&exo_smol_anchor_buf_,
                                anchor_elems * sizeof(DataType)));
    // Pool buffer: (max_batch, emb_size) — holds mean-over-tokens result.
    size_t pool_elems = (size_t)max_batch_size * embedding_op_size_;
    ReportCUDAErrors(cudaMalloc(&exo_smol_pool_buf_,
                                pool_elems * sizeof(DataType)));
    // Length-64 ones vector used for mean-pooling via strided-batched GEMM
    // (pooled[n] = (1/64) * flow[n] @ ones_64 gives the per-batch mean over
    // the 64 square tokens).  Allocated once at init.
    std::vector<float> ones_64(64, 1.0f);
    allocAndUpload<DataType>(&exo_smol_ones_64_, ones_64, scratch);
  }

  // Material info buffer: (max_batch, 64, 18) computed once per forward.
  if (has_material_info_) {
    size_t mat_size = (size_t)max_batch_size * 64 * 18 * sizeof(DataType);
    auto err = cudaMalloc(&material_features_buf_, mat_size);
    if (err != cudaSuccess) {
      material_features_buf_ = nullptr;
      has_material_info_ = false;  // disable on OOM
      cudaGetLastError();
    }
  }

  // Attack maps buffer: (max_batch, 64, 6) computed once per forward.
  if (has_attack_maps_) {
    size_t atk_size = (size_t)max_batch_size * 64 * 6 * sizeof(DataType);
    auto err = cudaMalloc(&attack_map_buf_, atk_size);
    if (err != cudaSuccess) {
      attack_map_buf_ = nullptr;
      has_attack_maps_ = false;  // disable on OOM
      cudaGetLastError();
    }
  }

  // Multi-stream FFN: runs FFN on ffn_stream_ concurrently with attention.
  // ffn_stream_ joins the main graph capture as a fork-of-fork (via per-layer
  // ln1_done_events_). This is the configuration that achieves 23811 mean /
  // 23905 median / 1.7% CV nps.
  multi_stream_ffn_ = is_parallel_ffn_;
  if (multi_stream_ffn_) {
    // Derive dff from first encoder (all layers have same dff)
    int dff = 0;
    if (!encoder_weights_.empty()) {
      dff = encoder_weights_[0]->ffn_dff_;  // SwiGLU path
      if (dff == 0) dff = encoder_weights_[0]->ffn_dense1_size_;  // standard FFN
    }
    if (dff > 0) {
      ReportCUDAErrors(cudaStreamCreateWithFlags(&ffn_stream_, cudaStreamNonBlocking));
      ReportCUBLASErrors(cublasCreate(&ffn_cublas_));
      ReportCUBLASErrors(cublasSetStream(ffn_cublas_, ffn_stream_));
      num_encoder_layers_ = (int)encoder_weights_.size();
      for (int i = 0; i < num_encoder_layers_; i++) {
        ReportCUDAErrors(cudaEventCreateWithFlags(&ln1_done_events_[i],
                                                  cudaEventDisableTiming));
        ReportCUDAErrors(cudaEventCreateWithFlags(&ffn_done_events_[i],
                                                  cudaEventDisableTiming));
      }
      // Pre-allocate FFN graph exec vector (one slot per batch size).
      ffn_graph_execs_.assign(max_batch_size_, nullptr);
      size_t wide = (size_t)max_batch_size * 64 * 2 * dff * sizeof(DataType);
      size_t out = (size_t)max_batch_size * 64 * embedding_op_size_ * sizeof(DataType);
      ReportCUDAErrors(cudaMalloc(&ffn_buf_wide_, wide));
      ReportCUDAErrors(cudaMalloc(&ffn_buf_out_, out));

      // ── Shared gate bank (A-Layout-2) buffers ──
      // Dedicated allocations: h is written on the main stream and read by
      // both streams within a layer, so it cannot live in reusable scratch.
      // Segment widths/order in bank_lrb_buf_ ([v | vga | ffn], rank>0
      // sites only, fixed max-token strides) must match the carve in
      // EncoderBlock::Eval exactly.
      {
        const auto* enc0 = encoder_weights_[0];
        has_gate_bank_ = enc0->has_gate_bank_;
        if (has_gate_bank_) {
          const size_t max_tokens = (size_t)max_batch_size * 64;
          ReportCUDAErrors(cudaMalloc(
              &bank_h_buf_,
              max_tokens * enc0->gate_bank_size_ * sizeof(DataType)));
          const int total_rank = enc0->bank_rank_v_ + enc0->bank_rank_vga_ +
                                 enc0->bank_rank_ffn_ + enc0->bank_rank_q_ +
                                 enc0->bank_rank_k_;
          if (total_rank > 0) {
            // Layout: [lr_a output (total_rank) | lrb_v | lrb_vga |
            // lrb_ffn | lrb_q | lrb_k], each at max-token stride — must
            // match the carve in EncoderBlock::Eval.
            size_t width = total_rank;
            if (enc0->bank_rank_v_ > 0) width += enc0->mha_v_size_;
            if (enc0->bank_rank_vga_ > 0) width += enc0->mha_q_size_;
            if (enc0->bank_rank_ffn_ > 0) width += enc0->ffn_dff_;
            if (enc0->bank_rank_q_ > 0) width += enc0->bank_q_width_;
            if (enc0->bank_rank_k_ > 0) width += enc0->bank_k_width_;
            ReportCUDAErrors(cudaMalloc(
                &bank_lr_buf_, max_tokens * width * sizeof(DataType)));
          }
          for (int i = 0; i < num_encoder_layers_; i++) {
            ReportCUDAErrors(cudaEventCreateWithFlags(
                &bank_done_events_[i], cudaEventDisableTiming));
          }
        }
      }

      // Third-stream VGA-E was tested and reverted: median improved +2% but
      // CV doubled (1.7%→3.5%), mean flat. The main/FFN streams are already
      // perfectly balanced; removing VGA-E from the main stream critical path
      // tips the balance and causes occasional stalls. Left here as dead code
      // — re-enable if FFN stream is sped up and creates timing headroom.

      // Third-stream SMOLGEN overlap.  Gated on has_smolgen_ (obvious) and
      // on an opt-out env var so regressions can be isolated at runtime.
      // For post-norm nets with no current ffn_stream overlap opportunity
      // (ln1_cache_ is null), this is pure headroom: smolgen runs entirely
      // hidden behind the main-stream Q/K/V work.  For pre-norm nets where
      // main/ffn are already balanced, benefit depends on which stream is
      // the critical path.
      static const bool kMsSmolgenDisabled =
          std::getenv("LC0_DISABLE_MS_SMOLGEN") != nullptr;
      const bool any_has_smolgen =
          !encoder_weights_.empty() && encoder_weights_[0]->has_smolgen_;
      if (any_has_smolgen && !kMsSmolgenDisabled) {
        // Stream + cublas handle.
        ReportCUDAErrors(cudaStreamCreateWithFlags(&smolgen_stream_,
                                                    cudaStreamNonBlocking));
        ReportCUBLASErrors(cublasCreate(&smolgen_cublas_));
        ReportCUBLASErrors(cublasSetStream(smolgen_cublas_, smolgen_stream_));
        // Per-layer done events (external so CUDA-graph capture can wait
        // across stream forks without isolation errors, matching the
        // ffn_done_events_ pattern above).
        for (int i = 0; i < num_encoder_layers_; i++) {
          ReportCUDAErrors(cudaEventCreateWithFlags(&smol_done_events_[i],
                                                     cudaEventDisableTiming));
        }
        // Single shared gen output buffer (max_batch, H*64*64).  One is
        // enough — the event chain through smol_done → softmax → ln1_done
        // forces smolgen_stream_ to block before overwriting the buffer
        // while main stream still has pending reads at layer i.  Sized for
        // the worst case (max configured batch).
        const auto* enc0 = encoder_weights_[0];
        const int H = encoder_head_count_;
        const size_t gen_out_bytes =
            (size_t)max_batch_size * (size_t)H * 64 * 64 * sizeof(DataType);
        // Shared intermediate buffers (reused layer-to-layer, safe because
        // smolgen_stream_ is serial across layers).  Sized for the widest
        // intermediate.  Post-compress output: (batch*64, compress_size).
        // Gen-from output: (batch, H*gen_sz).  Take the max.
        const size_t interm_elems =
            std::max((size_t)max_batch_size * 64 * embedding_op_size_,
                     (size_t)max_batch_size * enc0->smol_dense_2_size_);
        const size_t interm_bytes = interm_elems * sizeof(DataType);
        bool alloc_ok = true;
        if (cudaMalloc(&smol_intermediate_, interm_bytes) != cudaSuccess) {
          smol_intermediate_ = nullptr;
          cudaGetLastError();
          alloc_ok = false;
        }
        if (alloc_ok &&
            cudaMalloc(&smol_intermediate2_, interm_bytes) != cudaSuccess) {
          smol_intermediate2_ = nullptr;
          cudaGetLastError();
          alloc_ok = false;
        }
        if (alloc_ok &&
            cudaMalloc(&smol_gen_out_, gen_out_bytes) != cudaSuccess) {
          smol_gen_out_ = nullptr;
          cudaGetLastError();
          alloc_ok = false;
        }
        if (alloc_ok) {
          multi_stream_smolgen_ = true;
        } else {
          // OOM: free what we got and disable.  Old code path (smolgen on
          // main stream, buffer2 output) remains correct.
          if (smol_gen_out_) {
            cudaFree(smol_gen_out_);
            smol_gen_out_ = nullptr;
          }
          if (smol_intermediate_) {
            cudaFree(smol_intermediate_);
            smol_intermediate_ = nullptr;
          }
          if (smol_intermediate2_) {
            cudaFree(smol_intermediate2_);
            smol_intermediate2_ = nullptr;
          }
        }
      }
    } else {
      multi_stream_ffn_ = false;  // disable if no valid dff
    }
  }
}

template <typename DataType>
AttentionBody<DataType>::~AttentionBody() {
  // FreeWeight() handles both arena-owned buffers (no-op; arena frees
  // its chunks) and direct-cudaMalloc buffers (cudaFree).  Owns()
  // distinguishes them at runtime.  Replaces the previous cudaFree
  // calls 1:1 without changing semantics for the non-arena case.
  FreeWeight(ip_emb_w_, arena_);
  FreeWeight(ip_emb_b_, arena_);
  if (is_pe_dense_embedding_) {
    FreeWeight(ip_emb_pre_w_, arena_);
    FreeWeight(ip_emb_pre_b_, arena_);
  }
  if (is_pe_dense_embedding_ || has_rich_emb_) {
    FreeWeight(ip_emb_ln_g_, arena_);
    FreeWeight(ip_emb_ln_b_, arena_);
    FreeWeight(ip_emb_ffn_d1_w_, arena_);
    FreeWeight(ip_emb_ffn_d1_b_, arena_);
    FreeWeight(ip_emb_ffn_d2_w_, arena_);
    FreeWeight(ip_emb_ffn_d2_b_, arena_);
    FreeWeight(ip_emb_ffn_ln_g_, arena_);
    FreeWeight(ip_emb_ffn_ln_b_, arena_);
  } else {
    FreeWeight(pos_encoding_, arena_);
  }
  if (has_gating_) {
    FreeWeight(ip_mult_gate_, arena_);
    FreeWeight(ip_add_gate_, arena_);
  }
  if (has_smolgen_) {
    FreeWeight(smolgen_global_, arena_);
    FreeWeight(smol_dict_p_, arena_);
    FreeWeight(smol_dict_dec_, arena_);
  }
  FreeWeight(enc_final_norm_g_, arena_);
  FreeWeight(enc_final_norm_b_, arena_);
  if (has_emb_ffn_swiglu_) {
    FreeWeight(ip_emb_ffn_gate_w_, arena_);
    FreeWeight(ip_emb_ffn_gate_b_, arena_);
    FreeWeight(ip_emb_ffn_up_w_, arena_);
    FreeWeight(ip_emb_ffn_up_b_, arena_);
    FreeWeight(ip_emb_ffn_down_w_, arena_);
    FreeWeight(ip_emb_ffn_down_b_, arena_);
  }
  for (const auto pEnc : encoder_weights_) delete pEnc;
  FreeWeight(rich_emb_sq_w1_, arena_);
  FreeWeight(rich_emb_sq_b1_, arena_);
  FreeWeight(rich_emb_sq_w2_, arena_);
  FreeWeight(rich_emb_sq_b2_, arena_);
  FreeWeight(rich_emb_global_w_, arena_);
  FreeWeight(rich_emb_global_b_, arena_);
  FreeWeight(rich_per_sq_buf_, arena_);          // runtime buffer
  FreeWeight(exo_q_anc_w_, arena_);
  FreeWeight(exo_k_anc_w_, arena_);
  FreeWeight(exo_v_anc_w_, arena_);
  // exo_q/k/v_anc_buf_ are pointer aliases into exo_all_anc_buf_ — free only the base.
  FreeWeight(exo_all_anc_buf_, arena_);          // runtime buffer
  FreeWeight(exo_norm_ones_, arena_);
  FreeWeight(exo_smol_anchor_w_, arena_);
  FreeWeight(exo_smol_anchor_buf_, arena_);      // runtime buffer
  FreeWeight(exo_smol_pool_buf_, arena_);        // runtime buffer
  FreeWeight(exo_smol_ones_64_, arena_);
  FreeWeight(material_features_buf_, arena_);    // runtime buffer
  FreeWeight(attack_map_buf_, arena_);           // runtime buffer
  FreeWeight(value_branch_buf_, arena_);         // runtime buffer
  FreeWeight(ffn_buf_wide_, arena_);             // runtime buffer
  FreeWeight(ffn_buf_out_, arena_);              // runtime buffer
  FreeWeight(bank_h_buf_, arena_);               // runtime buffer
  FreeWeight(bank_lr_buf_, arena_);              // runtime buffer
  for (int i = 0; i < num_encoder_layers_; i++) {
    if (ln1_done_events_[i]) cudaEventDestroy(ln1_done_events_[i]);
    if (ffn_done_events_[i]) cudaEventDestroy(ffn_done_events_[i]);
    if (bank_done_events_[i]) cudaEventDestroy(bank_done_events_[i]);
  }
  for (auto exec : ffn_graph_execs_) {
    if (exec) cudaGraphExecDestroy(exec);
  }
  if (ffn_cublas_) cublasDestroy(ffn_cublas_);
  if (ffn_stream_) cudaStreamDestroy(ffn_stream_);
  // vga_stream_/vga_cublas_/vga_done_event_ are null (third-stream VGA-E
  // reverted — see constructor comment). Guards left for safety.
  if (vga_done_event_) cudaEventDestroy(vga_done_event_);
  if (vga_cublas_) cublasDestroy(vga_cublas_);
  if (vga_stream_) cudaStreamDestroy(vga_stream_);

  // Multi-stream smolgen cleanup.
  for (int i = 0; i < num_encoder_layers_; i++) {
    if (smol_done_events_[i]) cudaEventDestroy(smol_done_events_[i]);
  }
  if (smol_gen_out_) cudaFree(smol_gen_out_);
  if (smol_intermediate_) cudaFree(smol_intermediate_);
  if (smol_intermediate2_) cudaFree(smol_intermediate2_);
  if (smolgen_cublas_) cublasDestroy(smolgen_cublas_);
  if (smolgen_stream_) cudaStreamDestroy(smolgen_stream_);
}

template <typename DataType>
void AttentionBody<DataType>::Eval(int N, DataType* output,
                                   const DataType* input,
                                   const DataType* input2, void* scratch,
                                   size_t scratch_size, cudnnHandle_t /*cudnn*/,
                                   cublasHandle_t cublas, cudaStream_t stream,
                                   DataType*** offset_pointers) {
  // Debug buffer map (LC0_DUMP_HGEMM=1): one-time print of every base
  // pointer the encoder derives GEMM outputs from, plus their sizes.
  // Cross-reference against the per-GEMM A/B/C dumps to attribute an
  // out-of-range pointer to the buffer (and the max_batch arithmetic)
  // it was carved from.
  {
    static const bool kDumpBufMap =
        std::getenv("LC0_DUMP_HGEMM") != nullptr;
    static std::atomic<bool> dumped{false};
    if (kDumpBufMap && !dumped.exchange(true)) {
      fprintf(stderr,
              "[bufmap] N=%d scratch=%p scratch_size=%zu output=%p "
              "input=%p max_batch=%d emb=%d\n",
              N, scratch, scratch_size, (void*)output, (const void*)input,
              max_batch_size_, embedding_op_size_);
      fprintf(stderr,
              "[bufmap] bank_h=%p bank_lr=%p smol_gen_out0=%p "
              "smol_interm=%p smol_interm2=%p ffn_wide=%p ffn_out=%p\n",
              (void*)bank_h_buf_, (void*)bank_lr_buf_,
              (void*)(num_encoder_layers_ > 0 ? smol_gen_out_ : nullptr),
              (void*)smol_intermediate_, (void*)smol_intermediate2_,
              (void*)ffn_buf_wide_, (void*)ffn_buf_out_);
      fflush(stderr);
    }
  }
  // Reset per-forward encoder layer counter so _lid == 0 always means
  // "layer 0 of this forward pass" (not a cumulative counter).
  g_enc_layer_counter.store(0, std::memory_order_relaxed);

  // Track actual N for dual-graph: forwardEval may pad batchSize up to
  // min_batch_size_ before calling Eval. CaptureFFNGraph reads this to
  // ensure the FFN graph's GEMM dimensions match the main graph.
  last_eval_n_ = N;

  DataType* output_tensor = (DataType*)output;
  DataType* buffer1 = (DataType*)input2;
  DataType* buffer2 = buffer1 + scratch_size / (2 * sizeof(DataType));

  // Diagnostic: per-layer CUDA event timing (LC0_LAYER_TIMING=1). Declared
  // at function scope so both the in-loop recording and post-loop report
  // can reach them. Static + thread_local so the storage survives across
  // forwards without per-call allocation.
  static thread_local std::vector<cudaEvent_t> layer_starts;
  static thread_local std::vector<cudaEvent_t> layer_ends;
  static thread_local std::vector<double> layer_total_ms;
  static std::atomic<int> timing_forward_count{0};
  static const bool kLayerTimingEnabled =
      std::getenv("LC0_LAYER_TIMING") != nullptr;

  // Diagnostic: "ExoFormer warmup" experiment. If LC0_CUBLAS_WARMUP=1 is
  // set, run 3 dummy GEMMs at the start of each forward matching the
  // shapes ExoFormer would have used (d_model × batch*64 × d_model).
  // Theory: ExoFormer's presence warms cuBLAS's algorithm cache for
  // encoder-shaped GEMMs. Without it, encoder GEMMs hit cold cache and
  // get slower algorithms, explaining the observed nps regression when
  // ExoFormer is disabled. If this experiment closes the gap, the fix
  // is to add init-time warmup GEMMs permanently.
  static const bool kCublasWarmup =
      std::getenv("LC0_CUBLAS_WARMUP") != nullptr;
  if (kCublasWarmup && !has_exoformer_) {
    const int d = embedding_op_size_;
    const int batch = N * 64;
    // Use scratch for all three operands to avoid lda mismatch with real
    // weight buffers. Sizes needed (FP16): A = d*d = 128 KB at d=256,
    // B = d*batch, C = d*batch. scratch is sized for worst-case
    // intermediates and has plenty of headroom.
    DataType* A_warm = (DataType*)scratch;
    DataType* B_warm = A_warm + (size_t)d * d;
    DataType* C_warm = B_warm + (size_t)d * batch;
    // Content doesn't matter — we only care about warming the cuBLAS algo
    // cache for this shape. Memory is valid and tight-strided with lda=d.
    for (int i = 0; i < 3; ++i) {
      cublasXgemm<DataType>(
          cublas, CUBLAS_OP_T, CUBLAS_OP_N, d, batch, d, 1.0f,
          A_warm, d, B_warm, d, 0.0f, C_warm, d);
    }
  }

  // Dump raw NCHW input (skip first call — may be warmup with empty data)
  if (debugDumpEnabled()) {
    static int call_count = 0;
    call_count++;
    if (call_count == 2) {  // dump on second call
      // Print first 16 values of raw input to verify it's not all zeros
      debugDump("raw_input_first16", input, 16, stream);
      int total = N * 112 * 8 * 8;
      std::vector<DataType> host(total);
      cudaStreamSynchronize(stream);
      cudaMemcpy(host.data(), input, total * sizeof(DataType), cudaMemcpyDeviceToHost);
      int nz = 0;
      for (int i = 0; i < total; i++) {
        float v;
        if constexpr (std::is_same_v<DataType, half>) v = __half2float(host[i]);
        else v = (float)host[i];
        if (v != 0.0f) nz++;
      }
      fprintf(stderr, "[DEBUG] Input stats: N=%d total=%d nonzero=%d\n", N, total, nz);
      // Write as float32
      std::vector<float> fhost(total);
      for (int i = 0; i < total; i++) {
        if constexpr (std::is_same_v<DataType, half>) fhost[i] = __half2float(host[i]);
        else fhost[i] = (float)host[i];
      }
      FILE* f = fopen("cuda_input_nchw.bin", "wb");
      if (f) {
        fwrite(fhost.data(), sizeof(float), total, f);
        fclose(f);
        fprintf(stderr, "[DEBUG] Wrote cuda_input_nchw.bin (%d floats)\n", total);
      }
    }
  }

  int inputC = input_c_;
  if (num_resi_blocks_ == 0) {
    assert(inputC == kInputPlanes);
    /*
      # if there are no residual blocks (pure transformer), do some input
      processing
    */
    if (has_rich_emb_) {
      // Rich embedding: per-square MLP input is widened with per-square
      // extras. Extras are computed up-front (also used by the preprocess
      // kernel below, so the compute is paid exactly once).
      const bool do_mat = has_material_info_ && material_features_buf_;
      const bool do_atk = has_attack_maps_ && attack_map_buf_;
      if (do_mat) computeMaterialFeatures<DataType>(N, material_features_buf_, input, stream);
      if (do_atk) computeAttackMaps<DataType>(N, attack_map_buf_, input, stream);

      // 1. Build per-square MLP input:
      //    [piece_12, attack_6?, material_masks_2?]
      //    NHWC shape (N, 64, rich_per_sq_input_sz_).
      buildRichPerSquareInput<DataType>(
          rich_per_sq_buf_, input,
          do_atk ? attack_map_buf_        : nullptr, do_atk ? 6  : 0,
          do_mat ? material_features_buf_ : nullptr, do_mat ? 18 : 0,
          N, rich_per_sq_input_sz_, stream);

      // 2. Per-square MLP: Linear(in, hidden) → SiLU → Linear(hidden, emb_dense)
      const int sq_batch = N * 64;
      // First GEMM's k = rich_per_sq_input_sz_ can be odd (e.g. 35 when all
      // three extras are active: 12 + 6 + 2 + 15). fp16 tensor-core kernels
      // on Ada + CUDA 12.x reject non-aligned k with CUBLAS_STATUS_NOT_SUPPORTED.
      // Try CUBLAS_COMPUTE_32F (fp32 accumulation, non-pedantic, tensor cores
      // allowed) — broader kernel coverage than the pedantic SIMT path, which
      // failed for this specific shape. Only needed for fp16.
      if (std::is_same<half, DataType>::value &&
          (rich_per_sq_input_sz_ % 8) != 0) {
        const float alpha = 1.0f, beta = 0.0f;
        ReportCUBLASErrors(cublasGemmEx(
            cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            rich_hidden_sz_, sq_batch, rich_per_sq_input_sz_,
            &alpha,
            (const void*)rich_emb_sq_w1_, CUDA_R_16F, rich_per_sq_input_sz_,
            (const void*)rich_per_sq_buf_, CUDA_R_16F, rich_per_sq_input_sz_,
            &beta,
            (void*)buffer2, CUDA_R_16F, rich_hidden_sz_,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
      } else {
        cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                              rich_hidden_sz_, sq_batch, rich_per_sq_input_sz_,
                              1.0f, (const DataType*)rich_emb_sq_w1_,
                              rich_per_sq_input_sz_,
                              (const DataType*)rich_per_sq_buf_,
                              rich_per_sq_input_sz_, 0.0f,
                              buffer2, rich_hidden_sz_);
      }
      addBiasBatched<DataType>(buffer2, buffer2, rich_emb_sq_b1_, 1, sq_batch,
                               rich_hidden_sz_, ACTIVATION_SWISH, stream);
      cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                            embedding_dense_size_, sq_batch, rich_hidden_sz_,
                            1.0f, (const DataType*)rich_emb_sq_w2_,
                            rich_hidden_sz_,
                            buffer2, rich_hidden_sz_, 0.0f,
                            buffer1, embedding_dense_size_);
      addBiasBatched<DataType>(buffer1, buffer1, rich_emb_sq_b2_, 1, sq_batch,
                               embedding_dense_size_, ACTIVATION_NONE, stream);

      // 3. Global summary: flatten (N, 64*12) → Linear → (N, global_sz)
      convertNCHWtoNHWC((DataType*)scratch, input, N, inputC, N, 12, 8, 8,
                        stream);
      cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                            rich_emb_global_sz_, N, 64 * 12, 1.0f,
                            (const DataType*)rich_emb_global_w_, 64 * 12,
                            (const DataType*)scratch, 64 * 12, 0.0f,
                            buffer2, rich_emb_global_sz_);
      addBiasBatched<DataType>(buffer2, buffer2, rich_emb_global_b_, 1, N,
                               rich_emb_global_sz_, ACTIVATION_NONE, stream);

      // 4. Concat per-square + broadcast global → encoding buffer.
      // Same aliasing-safe offset as the light path: push encoding past the
      // maximum preprocess write region (extras included).
      const int enc_sz = embedding_dense_size_ + rich_emb_global_sz_;
      const int max_extras = (do_mat ? 18 : 0) + (do_atk ? 6 : 0);
      DataType* encoding = (DataType*)scratch +
          N * 64 * (kInputPlanes + max_extras + enc_sz);
      ConcatSquareAndGlobal<DataType>(N, embedding_dense_size_, rich_emb_global_sz_,
                                      encoding, buffer1, buffer2, stream);

      // 5. Extended preprocess: extras are already in their buffers, no need
      //    to recompute. Output layout: [input(112), material(18)?, attack(6)?,
      //    encoding].
      //
      // Preprocess path is driven strictly from NetworkFormat flags:
      // whatever training asserted via the flags is what we concatenate
      // here. The main-embedding weight's col count must match
      // (112 + enc_sz + mat(18)? + atk(6)?) — if it doesn't, that's a
      // config mismatch to fix on the training side, not auto-patch here.
      const bool emb_has_mat = do_mat;
      const bool emb_has_atk = do_atk;

      if (emb_has_mat && emb_has_atk) {
        inputPreprocessForAttentionBodyWithTwoExtras<DataType>(
            (DataType*)scratch, input,
            material_features_buf_, /*extra1_size=*/18,
            attack_map_buf_,        /*extra2_size=*/6,
            encoding, N, kInputPlanes, enc_sz, true, stream);
        inputC += 18 + 6 + enc_sz;
      } else if (emb_has_mat) {
        inputPreprocessForAttentionBodyWithMaterial<DataType>(
            (DataType*)scratch, input, material_features_buf_, encoding, N,
            kInputPlanes, /*material_size=*/18, enc_sz, true, stream);
        inputC += 18 + enc_sz;
      } else if (emb_has_atk) {
        inputPreprocessForAttentionBodyWithMaterial<DataType>(
            (DataType*)scratch, input, attack_map_buf_, encoding, N,
            kInputPlanes, /*material_size=*/6, enc_sz, true, stream);
        inputC += 6 + enc_sz;
      } else {
        // No enriched features are consumed by the main embedding — either
        // the flags are off, or (more commonly) they're on but the net was
        // trained with ip_emb_w narrow enough that feeding extras here
        // would exceed the real weight width. The extras still reach the
        // rich MLP via material_features_buf_/attack_map_buf_ earlier in
        // this Eval; they're just not directly concatenated before
        // self.embedding.
        inputPreprocessForAttentionBody((DataType*)scratch, input, encoding, N,
                                        kInputPlanes, enc_sz, true, stream);
        inputC += enc_sz;
      }

    } else if (is_pe_dense_embedding_) {
      // Heavy embedding: global dense projection
      const int num_outputs = 64 * embedding_dense_size_;
      const int num_inputs = 64 * 12;
      const int batch = N;

      convertNCHWtoNHWC((DataType*)scratch, input, N, inputC, N, 12, 8, 8,
                        stream);
      cublasXgemm<DataType>(
          cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch, num_inputs,
          1.0f, (const DataType*)ip_emb_pre_w_, num_inputs,
          (const DataType*)scratch, num_inputs, 0.0f, buffer1, num_outputs);

      const int size = num_outputs * N;
      addVectors(buffer1, buffer1, ip_emb_pre_b_, size, size, num_outputs,
                 ACTIVATION_NONE, stream);
      debugDump("heavy_emb_preproc", buffer1, 8, stream);
      {
        const bool do_mat = has_material_info_ && material_features_buf_;
        const bool do_atk = has_attack_maps_ && attack_map_buf_;
        if (do_mat) computeMaterialFeatures<DataType>(N, material_features_buf_, input, stream);
        if (do_atk) computeAttackMaps<DataType>(N, attack_map_buf_, input, stream);
        if (do_mat && do_atk) {
          inputPreprocessForAttentionBodyWithTwoExtras<DataType>(
              (DataType*)scratch, input,
              material_features_buf_, /*extra1_size=*/18,
              attack_map_buf_,        /*extra2_size=*/6,
              buffer1, N, kInputPlanes, embedding_dense_size_, true, stream);
          inputC += 18 + 6 + embedding_dense_size_;
        } else if (do_mat) {
          inputPreprocessForAttentionBodyWithMaterial<DataType>(
              (DataType*)scratch, input, material_features_buf_, buffer1, N,
              kInputPlanes, /*material_size=*/18, embedding_dense_size_, true,
              stream);
          inputC += 18 + embedding_dense_size_;
        } else if (do_atk) {
          inputPreprocessForAttentionBodyWithMaterial<DataType>(
              (DataType*)scratch, input, attack_map_buf_, buffer1, N,
              kInputPlanes, /*material_size=*/6, embedding_dense_size_, true,
              stream);
          inputC += 6 + embedding_dense_size_;
        } else {
          inputPreprocessForAttentionBody((DataType*)scratch, input, buffer1, N,
                                          kInputPlanes, embedding_dense_size_,
                                          true, stream);
          inputC += embedding_dense_size_;
        }
      }
      debugDump("after_preprocess", (DataType*)scratch, 8, stream);
    } else {
      /*
      flow = tf.transpose(inputs, perm=[0, 2, 3, 1])
      flow = tf.reshape(flow, [-1, 64, tf.shape(inputs)[1]])
      # add positional encoding for each square to the input
      positional_encoding = tf.broadcast_to(tf.convert_to_tensor(self.POS_ENC,
      dtype=self.model_dtype), [tf.shape(flow)[0], 64,
      tf.shape(self.POS_ENC)[2]]) flow = tf.concat([flow, positional_encoding],
      axis=2)
      */
      inputPreprocessForAttentionBody((DataType*)scratch, input, pos_encoding_,
                                      N, kInputPlanes, kNumPosEncodingChannels,
                                      false, stream);
      inputC += kNumPosEncodingChannels;
    }
  } else {
    // #redirect flow through encoder blocks
    // flow = tf.transpose(flow, perm = [ 0, 2, 3, 1 ])
    // flow = tf.reshape(flow, [ -1, 64, self.RESIDUAL_FILTERS ])
    convertNCHWtoNHWC((DataType*)scratch, input, N, inputC, N, inputC, 8, 8,
                      stream);
  }

  if (is_pe_dense_embedding_ || has_rich_emb_) {
    // 1. square embedding (fully connected layer)
    // Input data in NHWC layout N*(64)*C, output is N*(64)*embedding_op_size_
    DataType* embedding = output_tensor;
    DataType* temp = (DataType*)scratch;
    {
      const int num_outputs = embedding_op_size_;
      const int num_inputs = inputC;
      const int batch = N * 64;
      // Dump the preprocessed input — skip warmup (first call has empty data)
      if (debugDumpEnabled()) {
        static int emb_call = 0;
        emb_call++;
        if (emb_call >= 2) {  // Skip first call (warmup with empty input)
          debugDump("emb_input_sq0_feat0_8", temp, 8, stream);
          debugDump("emb_input_sq0_feat112_8", temp + 112, 8, stream);
          debugDump("emb_input_sq0_aux104_8", temp + 104, 8, stream);
          fprintf(stderr, "[DEBUG] emb_input: inputC=%d batch=%d\n", num_inputs, batch);
          std::vector<float> sq0(num_inputs);
          cudaStreamSynchronize(stream);
          if constexpr (std::is_same_v<DataType, half>) {
            std::vector<half> h(num_inputs);
            cudaMemcpy(h.data(), temp, num_inputs * sizeof(half), cudaMemcpyDeviceToHost);
            for (int i = 0; i < num_inputs; i++) sq0[i] = __half2float(h[i]);
          } else {
            cudaMemcpy(sq0.data(), temp, num_inputs * sizeof(float), cudaMemcpyDeviceToHost);
          }
          int nz = 0;
          for (int i = 0; i < 112; i++) if (sq0[i] != 0.0f) nz++;
          fprintf(stderr, "[DEBUG] sq0 raw 112 nonzero: %d/112\n", nz);
          for (int step = 0; step < 8; step++) {
            int base = step * 13;
            bool all_zero = true;
            for (int j = 0; j < 13; j++) if (sq0[base+j] != 0.0f) all_zero = false;
            if (!all_zero) {
              fprintf(stderr, "[DEBUG] sq0 step%d: ", step);
              for (int j = 0; j < 13; j++) fprintf(stderr, "%.1f ", sq0[base+j]);
              fprintf(stderr, "\n");
            }
          }
          fprintf(stderr, "[DEBUG] sq0 aux[104:112]: ");
          for (int j = 104; j < 112 && j < num_inputs; j++) fprintf(stderr, "%.1f ", sq0[j]);
          fprintf(stderr, "\n");
        }
      }
      // Pre-GEMM input x dump — this is exactly the tensor PyTorch calls
      // `flow` right before `self.embedding(flow)`. Compare sq=0 slice
      // against a forward_pre_hook on self.embedding on the PyTorch side.
      //   full_432 = the entire sq=0 vector
      //   input112 = first 112 entries (raw piece/castling/aux planes)
      //   sq_feat  = next 256 entries (per-square MLP output)
      //   global   = last 64 entries (global projection, broadcast)
      // Split so each region can be diffed independently; bug likely in
      // sq_feat or global (per prior diff).
      debugScanNaN("emb_input_sq0_full_432", temp, num_inputs, stream);
      debugScanNaN("emb_input_sq0_input112", temp, 112, stream);
      debugScanNaN("emb_input_sq0_sq_feat",
                   temp + 112, embedding_dense_size_, stream);
      debugScanNaN("emb_input_sq0_global",
                   temp + 112 + embedding_dense_size_,
                   num_inputs - 112 - embedding_dense_size_, stream);
      // Main embedding GEMM. num_inputs = inputC = 112 + extras + enc_sz.
      // For rich+mat+atk+tac this is 311 (not a multiple of 8), which can
      // trigger CUBLAS_STATUS_NOT_SUPPORTED or silently-wrong results on
      // fp16 tensor-core kernels (Ada + CUDA 12.x). Fall back to
      // cublasGemmEx with CUBLAS_COMPUTE_32F when k is misaligned — same
      // mitigation the rich MLP first GEMM already uses for k=35.
      if (std::is_same<half, DataType>::value && (num_inputs % 8) != 0) {
        const float alpha = 1.0f, beta = 0.0f;
        ReportCUBLASErrors(cublasGemmEx(
            cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch, num_inputs,
            &alpha, (const void*)ip_emb_w_, CUDA_R_16F, num_inputs,
            (const void*)temp, CUDA_R_16F, num_inputs, &beta,
            (void*)embedding, CUDA_R_16F, num_outputs, CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT));
      } else {
        cublasXgemm<DataType>(
            cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch, num_inputs,
            1.0f, (const DataType*)ip_emb_w_, num_inputs, temp, num_inputs,
            0.0f, embedding, num_outputs);
      }
      // embedding layer norm with fused in bias add of previous gemm.
      // NOTE: Always LayerNorm here, even when encoder uses RMSNorm.
      // The embedding_ln is nn.LayerNorm in PyTorch regardless of use_rms_norm.
      debugDump("emb_gemm_out", embedding, 8, stream);
      debugDump("emb_bias", ip_emb_b_, 8, stream);
      debugDump("emb_ln_g", ip_emb_ln_g_, 8, stream);
      debugDump("emb_ln_b", ip_emb_ln_b_, 8, stream);
      if (debugDumpEnabled()) {
        fprintf(stderr, "[DEBUG] emb_act=%d N=%d C=%d\n", (int)activations_.default_activation, N*64, embedding_op_size_);
      }
      LayerNorm<DataType>(N * 64, embedding_op_size_, temp, embedding,
                          ip_emb_b_, (DataType*)nullptr, ip_emb_ln_g_,
                          ip_emb_ln_b_, 1e-3, 1.0,
                          activations_.default_activation, stream);
      debugDump("emb_after_ln", temp, 8, stream);
    }

    // Input gating
    if (has_gating_) {
      applyInputGating<DataType>(temp, temp, ip_mult_gate_, ip_add_gate_, N, 64,
                                 embedding_op_size_, stream);
    }

    if (is_prenorm_) {
      // Pre-norm: LN before FFN, simple residual add after.
      // temp holds post-gating embedding. embedding = output_tensor.
      // PyTorch: ffn_out = emb_ffn(emb_ffn_ln(flow)); flow = flow + ffn_out

      // LN(temp) -> buffer1
      NormLayer<DataType>(use_rms_norm_, N * 64, embedding_op_size_, buffer1,
                          temp, (DataType*)nullptr, (DataType*)nullptr,
                          ip_emb_ffn_ln_g_, ip_emb_ffn_ln_b_, 1e-3, 1.0,
                          ACTIVATION_NONE, stream);

      if (has_emb_ffn_swiglu_) {
        const int num_inputs = embedding_ffn_size_;
        const int dff = embedding_ffn_dff_;
        const int batch = N * 64;
        // Dump LN2 output (= input to SwiGLU) at sq=0.
        debugScanNaN("emb_ffn_ln_out_sq0", buffer1, num_inputs, stream);
        // gate_proj: buffer1 -> embedding (reuse as temp storage)
        cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, dff, batch, num_inputs,
                    1.0f, (const DataType*)ip_emb_ffn_gate_w_, num_inputs,
                    buffer1, num_inputs, 0.0f, embedding, dff);
        debugScanNaN("emb_ffn_gate_prebias_sq0", embedding, dff, stream);
        // up_proj: buffer1 -> buffer2
        cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, dff, batch, num_inputs,
                    1.0f, (const DataType*)ip_emb_ffn_up_w_, num_inputs,
                    buffer1, num_inputs, 0.0f, buffer2, dff);
        debugScanNaN("emb_ffn_up_prebias_sq0", buffer2, dff, stream);
        // Fused bias+silu+mult -> embedding
        // swiglu_softcap_ shared between encoder FFN and embedding FFN.
        SwiGLUElementwiseWithBias<DataType>(batch * dff, dff, embedding,
                                             embedding, buffer2,
                                             ip_emb_ffn_gate_b_,
                                             ip_emb_ffn_up_b_, stream,
                                             swiglu_softcap_);
        debugScanNaN("emb_ffn_swiglu_hidden_sq0", embedding, dff, stream);
        // down_proj: embedding -> buffer2
        cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_inputs, batch, dff,
                    1.0f, (const DataType*)ip_emb_ffn_down_w_, dff, embedding,
                    dff, 0.0f, buffer2, num_inputs);
        if (ip_emb_ffn_down_b_)
          addBiasBatched(buffer2, buffer2, ip_emb_ffn_down_b_, 1, batch,
                         num_inputs, ACTIVATION_NONE, stream);
        debugScanNaN("emb_ffn_downproj_out_sq0", buffer2, num_inputs, stream);
      } else {
        const int batch = N * 64;
        // dense1: buffer1 -> embedding
        cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, embedding_ffn_dff_,
                    batch, embedding_ffn_size_, 1.0f,
                    (const DataType*)ip_emb_ffn_d1_w_, embedding_ffn_size_,
                    buffer1, embedding_ffn_size_, 0.0f, embedding,
                    embedding_ffn_dff_);
        addBiasBatched(embedding, embedding, ip_emb_ffn_d1_b_, 1, batch,
                       embedding_ffn_dff_, activations_.ffn_activation, stream);
        // dense2: embedding -> buffer2
        cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, embedding_ffn_size_,
                    batch, embedding_ffn_dff_, 1.0f,
                    (const DataType*)ip_emb_ffn_d2_w_, embedding_ffn_dff_,
                    embedding, embedding_ffn_dff_, 0.0f, buffer2,
                    embedding_ffn_size_);
        addBiasBatched(buffer2, buffer2, ip_emb_ffn_d2_b_, 1, batch,
                       embedding_ffn_size_, ACTIVATION_NONE, stream);
      }

      // Simple residual: embedding = temp + buffer2
      addVectors(embedding, temp, buffer2,
                 N * 64 * embedding_op_size_, N * 64 * embedding_op_size_,
                 N * 64 * embedding_op_size_, ACTIVATION_NONE, stream);
      debugDump("emb_ffn_out", embedding, 8, stream);

    } else if (has_emb_ffn_swiglu_) {
      // Post-norm SwiGLU (shouldn't happen in practice, but handle it)
      const int num_inputs = embedding_ffn_size_;
      const int dff = embedding_ffn_dff_;
      const int batch = N * 64;
      cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, dff, batch, num_inputs,
                  1.0f, (const DataType*)ip_emb_ffn_gate_w_, num_inputs, temp,
                  num_inputs, 0.0f, buffer1, dff);
      cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, dff, batch, num_inputs,
                  1.0f, (const DataType*)ip_emb_ffn_up_w_, num_inputs, temp,
                  num_inputs, 0.0f, buffer2, dff);
      SwiGLUElementwiseWithBias<DataType>(batch * dff, dff, buffer1, buffer1,
                                           buffer2, ip_emb_ffn_gate_b_,
                                           ip_emb_ffn_up_b_, stream,
                                           swiglu_softcap_);
      cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_inputs, batch, dff,
                  1.0f, (const DataType*)ip_emb_ffn_down_w_, dff, buffer1, dff,
                  0.0f, buffer2, num_inputs);
      float alpha = (float)pow(2. * encoder_weights_.size(), -0.25);
      NormLayer<DataType>(use_rms_norm_, N * 64, embedding_ffn_size_, embedding,
                          buffer2, ip_emb_ffn_down_b_, temp, ip_emb_ffn_ln_g_,
                          ip_emb_ffn_ln_b_, 1e-3, alpha, ACTIVATION_NONE, stream);
      debugDump("emb_ffn_out", embedding, 8, stream);
    } else {
      // Post-norm standard FFN (original upstream path)
      {
        const int num_inputs = embedding_ffn_size_;
        const int num_outputs = embedding_ffn_dff_;
        const int batch = N * 64;
        cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                    num_inputs, 1.0f, (const DataType*)ip_emb_ffn_d1_w_,
                    num_inputs, temp, num_inputs, 0.0f, buffer1, num_outputs);
        addBiasBatched(buffer1, buffer1, ip_emb_ffn_d1_b_, 1, batch, num_outputs,
                       activations_.ffn_activation, stream);
      }
      {
        const int num_inputs = embedding_ffn_dff_;
        const int num_outputs = embedding_ffn_size_;
        const int batch = N * 64;
        cublasXgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                    num_inputs, 1.0f, (const DataType*)ip_emb_ffn_d2_w_,
                    num_inputs, buffer1, num_inputs, 0.0f, buffer2, num_outputs);
        float alpha = (float)pow(2. * encoder_weights_.size(), -0.25);
        NormLayer<DataType>(use_rms_norm_, N * 64, embedding_ffn_size_,
                            embedding, buffer2, ip_emb_ffn_d2_b_, temp,
                            ip_emb_ffn_ln_g_, ip_emb_ffn_ln_b_, 1e-3, alpha,
                            ACTIVATION_NONE, stream);
      }
      debugDump("emb_ffn_out", embedding, 8, stream);
    }

  } else {
    // 1. square embedding (fully connected layer)
    // Input data in NHWC layout N*(64)*C, output is N*(64)*embedding_op_size_
    DataType* embedding = output_tensor;
    {
      const int num_outputs = embedding_op_size_;
      const int num_inputs = inputC;
      const int batch = N * 64;
      cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                            batch, num_inputs, 1.0f, (const DataType*)ip_emb_w_,
                            num_inputs, (DataType*)scratch, num_inputs, 0.0f,
                            embedding, num_outputs);
      addBiasBatched(embedding, embedding, ip_emb_b_, 1, batch, num_outputs,
                     activations_.default_activation, stream);
    }
    // Input gating
    if (has_gating_) {
      applyInputGating<DataType>(embedding, embedding, ip_mult_gate_,
                                 ip_add_gate_, N, 64, embedding_op_size_,
                                 stream);
    }
  }

  // 3b. ExoFormer: compute anchor Q/K/V from embedding output.
  // output_tensor contains the embedded representation at this point.
  DataType* exo_q = nullptr;
  DataType* exo_k = nullptr;
  DataType* exo_v = nullptr;
  if (has_exoformer_) {
    const int d_model = embedding_op_size_;
    const int batch = N * 64;
    exo_q = exo_q_anc_buf_;
    exo_k = exo_k_anc_buf_;
    exo_v = exo_v_anc_buf_;
    // Q_anc = Wq_anc^T @ embedding
    cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, d_model, batch,
                          d_model, 1.0f, (const DataType*)exo_q_anc_w_,
                          d_model, output_tensor, d_model, 0.0f,
                          exo_q, d_model);
    // K_anc = Wk_anc^T @ embedding
    cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, d_model, batch,
                          d_model, 1.0f, (const DataType*)exo_k_anc_w_,
                          d_model, output_tensor, d_model, 0.0f,
                          exo_k, d_model);
    // V_anc = Wv_anc^T @ embedding
    cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, d_model, batch,
                          d_model, 1.0f, (const DataType*)exo_v_anc_w_,
                          d_model, output_tensor, d_model, 0.0f,
                          exo_v, d_model);
    // RMSNorm anchors per-head (scale=False): x / sqrt(mean(x^2) + eps)
    // Layout: (N*64, d_model) where d_model = heads * depth.
    // Treat as (N*64*heads, depth) for per-head normalization.
    const int depth = d_model / encoder_head_count_;
    NormLayer<DataType>(true, batch * encoder_head_count_, depth,
                        exo_q, exo_q, (DataType*)nullptr, (DataType*)nullptr,
                        exo_norm_ones_, (DataType*)nullptr,
                        1e-5f, 1.0f, ACTIVATION_NONE, stream);
    NormLayer<DataType>(true, batch * encoder_head_count_, depth,
                        exo_k, exo_k, (DataType*)nullptr, (DataType*)nullptr,
                        exo_norm_ones_, (DataType*)nullptr,
                        1e-5f, 1.0f, ACTIVATION_NONE, stream);
    NormLayer<DataType>(true, batch * encoder_head_count_, depth,
                        exo_v, exo_v, (DataType*)nullptr, (DataType*)nullptr,
                        exo_norm_ones_, (DataType*)nullptr,
                        1e-5f, 1.0f, ACTIVATION_NONE, stream);
  }

  // 3c. Exo-as-smolgen-bias: compute pooled-flow anchor once per forward.
  // output_tensor is (N, 64, embedding_op_size_) row-major at this point.
  // Steps:
  //   (1) pooled[n, e] = (1/64) * sum_t output_tensor[n, t, e]
  //       Implemented as N per-batch GEMVs: pooled[n] = (1/64) * flow[n] @ ones_64
  //       where flow[n] is (emb, 64) col-major (== (64, emb) row-major).
  //   (2) anchor[n, :] = pooled[n, :] @ anchor_w^T   (plain projection;
  //       the 1/sqrt(N_layers) training-side scaling is baked into anchor_w).
  // The anchor is then added to every encoder's smolgen gen_from (post-LN,
  // post-activation) before the shared weight_gen decoder.
  DataType* smolgen_anchor_ptr = nullptr;
  // Defensive: env-var kill-switch lets you disable exo-bias at runtime
  // without rebuilding (LC0_DISABLE_EXO_SMOL_BIAS=1).  Also gate on
  // presence of allocated buffers + smolgen_dense_2_size > 0 so a
  // mis-configured weight (e.g., loaded for a net without smolgen) can't
  // crash the encoder loop.
  static const bool kExoSmolBiasDisabled =
      std::getenv("LC0_DISABLE_EXO_SMOL_BIAS") != nullptr;
  if (has_exo_smol_anchor_ && !encoder_weights_.empty()
      && exo_smol_anchor_w_ != nullptr
      && exo_smol_anchor_buf_ != nullptr
      && exo_smol_pool_buf_ != nullptr
      && exo_smol_ones_64_ != nullptr
      && encoder_weights_[0]->smol_dense_2_size_ > 0
      && !kExoSmolBiasDisabled) {
    const int emb = embedding_op_size_;
    const int h_gen_sz = encoder_weights_[0]->smol_dense_2_size_;
    // Step 1: mean-pool each position's 64 tokens into an emb-dim vector.
    //   For each batch b: pooled[b] (emb,) = (1/64) * flow[b] (emb, 64) @ ones(64).
    //   Uses strided-batched GEMM: one batched call instead of N small GEMVs.
    //   - A: flow, stride = 64 * emb per batch (col-major (emb, 64) per batch)
    //   - B: ones_64, stride = 0 (shared across batches)
    //   - C: pooled, stride = emb per batch (col-major (emb, 1) per batch)
    const float mean_alpha = 1.0f / 64.0f;
    cublasXGemmStridedBatched<DataType>(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        emb, 1, 64,
        mean_alpha,
        (const void*)output_tensor, emb, (long long)64 * emb,
        (const void*)exo_smol_ones_64_, 64, 0LL,
        0.0f,
        (void*)exo_smol_pool_buf_, emb, (long long)emb,
        N, use_gemm_ex_);
    // Step 2: batched projection pool @ W^T -> (N, H*gen_sz).
    //   W is stored row-major (H*gen_sz, emb) == col-major (emb, H*gen_sz).
    //   CUBLAS_OP_T on W gives effective (H*gen_sz, emb) for the matmul.
    cublasXgemm<DataType>(
        cublas, CUBLAS_OP_T, CUBLAS_OP_N,
        h_gen_sz, N, emb,
        1.0f,
        (const DataType*)exo_smol_anchor_w_, emb,
        exo_smol_pool_buf_, emb,
        0.0f,
        exo_smol_anchor_buf_, h_gen_sz);
    smolgen_anchor_ptr = exo_smol_anchor_buf_;
  }

  // 4. Encoder blocks
  for (size_t enc_idx = 0; enc_idx < encoder_weights_.size(); enc_idx++) {
    const auto pEnc = encoder_weights_[enc_idx];
    // Diagnostic: LC0_FORCE_SINGLE_STREAM=1 forces sequential FFN on the
    // main stream, disabling the multi-stream FFN optimization.
    static const bool kForceSingleStream =
        std::getenv("LC0_FORCE_SINGLE_STREAM") != nullptr;
    const bool ms = multi_stream_ffn_ && !kForceSingleStream;

    // Skip event recording if the stream is being captured for a CUDA
    // graph — timing-enabled cudaEventRecord isn't capturable and would
    // abort the capture. Run with graph_capture=false when diagnosing.
    cudaStreamCaptureStatus cap_status = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &cap_status);
    const bool in_capture = (cap_status == cudaStreamCaptureStatusActive);
    if (kLayerTimingEnabled && !in_capture) {
      if (layer_starts.empty()) {
        layer_starts.resize(encoder_weights_.size());
        layer_ends.resize(encoder_weights_.size());
        layer_total_ms.assign(encoder_weights_.size(), 0.0);
        for (size_t i = 0; i < encoder_weights_.size(); ++i) {
          cudaEventCreate(&layer_starts[i]);
          cudaEventCreate(&layer_ends[i]);
        }
      }
      cudaEventRecord(layer_starts[enc_idx], stream);
    }

    // Third-stream smolgen is gated both at construction (see constructor)
    // and at runtime via env var LC0_DISABLE_MS_SMOLGEN (same envvar is
    // checked at construction, this is a redundant guard for safety).
    static const bool kMsSmolgenDisabled =
        std::getenv("LC0_DISABLE_MS_SMOLGEN") != nullptr;
    const bool ms_smol =
        multi_stream_smolgen_ && !kMsSmolgenDisabled &&
        smolgen_stream_ != nullptr && smol_gen_out_ != nullptr;
    // ln1_done_events_[enc_idx] is the "layer input is ready for secondary
    // streams to read" barrier — shared between ms_ffn and ms_smol.  Pass it
    // whenever either secondary stream is active.
    const bool any_secondary = ms || ms_smol;
    pEnc->Eval(N, output_tensor, (DataType*)scratch, buffer1, buffer2, cublas,
               stream, offset_pointers,
               /*prev_attn_logits=*/nullptr,
               exo_q, exo_k, exo_v,
               smolgen_anchor_ptr,
               ms ? ffn_stream_ : nullptr,
               ms ? ffn_cublas_ : nullptr,
               any_secondary ? ln1_done_events_[enc_idx] : nullptr,
               ms ? ffn_done_events_[enc_idx] : nullptr,
               ms ? ffn_buf_wide_ : nullptr,
               ms ? ffn_buf_out_ : nullptr,
               vga_stream_, vga_cublas_, vga_done_event_,
               ms_smol ? smolgen_stream_ : nullptr,
               ms_smol ? smolgen_cublas_ : nullptr,
               ms_smol ? smol_done_events_[enc_idx] : nullptr,
               ms_smol ? smol_gen_out_ : nullptr,
               ms_smol ? smol_intermediate_ : nullptr,
               ms_smol ? smol_intermediate2_ : nullptr,
               // Shared gate bank context.  NOT gated on `ms`: the V/VGA
               // adapters run on the main stream and need h regardless;
               // EncoderBlock::Eval itself rejects bank nets if the FFN
               // multi-stream path is unavailable (single-stream fallback
               // cannot gate from the bank).
               has_gate_bank_ ? bank_h_buf_ : nullptr,
               has_gate_bank_ ? bank_lr_buf_ : nullptr,
               has_gate_bank_ ? bank_done_events_[enc_idx] : nullptr);

    if (kLayerTimingEnabled && !in_capture) {
      cudaEventRecord(layer_ends[enc_idx], stream);
    }

    // Multi-exit value branch: capture encoder output at the configured
    // branch layer. The flow at this point is the post-encoder-block
    // residual (which has had its block-final LN applied in post-norm /
    // is the running residual sum in pre-norm). final_norm is applied
    // below after the loop ends — for a parity with the main path, we
    // must apply it to the captured branch flow too. Done after the loop
    // (single LN pass on value_branch_buf_) to keep the capture itself a
    // cheap memcpy on the main stream.
    if (value_branch_at_layer_ >= 0 &&
        value_branch_buf_ != nullptr &&
        (int)enc_idx == value_branch_at_layer_) {
      const size_t bytes =
          (size_t)N * 64 * embedding_op_size_ * sizeof(DataType);
      ReportCUDAErrors(cudaMemcpyAsync(value_branch_buf_, output_tensor,
                                        bytes, cudaMemcpyDeviceToDevice,
                                        stream));
    }
  }

  // Per-forward layer timing report. Skipped during graph capture; only
  // accurate in eager mode (pair with graph_capture=false).
  cudaStreamCaptureStatus cap2 = cudaStreamCaptureStatusNone;
  cudaStreamIsCapturing(stream, &cap2);
  if (kLayerTimingEnabled && !layer_starts.empty() &&
      cap2 != cudaStreamCaptureStatusActive) {
    cudaStreamSynchronize(stream);
    for (size_t i = 0; i < encoder_weights_.size(); ++i) {
      float ms_i = 0.0f;
      cudaEventElapsedTime(&ms_i, layer_starts[i], layer_ends[i]);
      layer_total_ms[i] += ms_i;
    }
    const int count = timing_forward_count.fetch_add(1) + 1;
    if ((count % 100) == 0) {
      fprintf(stderr, "[LC0_LAYER_TIMING] mean us over %d fwds: ", count);
      for (size_t i = 0; i < layer_total_ms.size(); ++i) {
        fprintf(stderr, "%.1f ", (layer_total_ms[i] / count) * 1000.0);
      }
      fprintf(stderr, "\n");
    }
  }

  // 5. Encoder final norm (applied whenever the weight is populated).
  // Previously gated on is_prenorm_; now post-norm nets can opt in too
  // via the Python `use_final_norm` config flag (see Python-side
  // LeelaModel for docs).  Clamps end-of-encoder-stack activations
  // before the heads read them — addresses the layer-N max≈9 spike
  // observed in activation diagnostics, independent of depth.
  if (enc_final_norm_g_) {
    NormLayer<DataType>(use_rms_norm_, N * 64, embedding_op_size_,
                        (DataType*)scratch, output_tensor,
                        (DataType*)nullptr, (DataType*)nullptr,
                        enc_final_norm_g_, enc_final_norm_b_,
                        1e-3f, 1.0f,
                        ACTIVATION_NONE, stream);
    ReportCUDAErrors(cudaMemcpyAsync(
        output_tensor, scratch,
        N * 64 * embedding_op_size_ * sizeof(DataType),
        cudaMemcpyDeviceToDevice, stream));

    // Multi-exit: apply final_norm to value branch buffer too so the
    // branched flow has the same prep as the main flow before value
    // heads consume it. Mirrors PyTorch:
    //   value_branch_flow = final_norm(aux_flows_local[branch_layer])
    if (value_branch_at_layer_ >= 0 && value_branch_buf_ != nullptr) {
      NormLayer<DataType>(use_rms_norm_, N * 64, embedding_op_size_,
                          (DataType*)scratch, value_branch_buf_,
                          (DataType*)nullptr, (DataType*)nullptr,
                          enc_final_norm_g_, enc_final_norm_b_,
                          1e-3f, 1.0f,
                          ACTIVATION_NONE, stream);
      ReportCUDAErrors(cudaMemcpyAsync(
          value_branch_buf_, scratch,
          N * 64 * embedding_op_size_ * sizeof(DataType),
          cudaMemcpyDeviceToDevice, stream));
    }
  }
}

template <typename DataType>
void AttentionBody<DataType>::CaptureFFNGraph(int N) {
  if (!multi_stream_ffn_ || num_encoder_layers_ == 0) return;
  if (N < 1 || N > max_batch_size_) return;
  // The GEMM dimensions must match the actual N used in the preceding
  // forwardEval call (which may have padded logical N up to min_batch_size_).
  // last_eval_n_ is set by Eval() on every forwardEval call.
  const int gemm_n = (last_eval_n_ > 0) ? last_eval_n_ : N;
  // Destroy previous graph for this batch size if it exists.
  if (ffn_graph_execs_[N - 1]) {
    ReportCUDAErrors(cudaGraphExecDestroy(ffn_graph_execs_[N - 1]));
    ffn_graph_execs_[N - 1] = nullptr;
  }
  // Capture FFN graph: all encoder layers' FFN ops on ffn_stream_.
  // External wait/record flags link it to the main graph via cross-graph events.
  // Use Relaxed mode (not ThreadLocal): cuBLAS dispatches work from internal
  // threads for workspace management and algorithm selection; ThreadLocal would
  // exclude these operations from the capture and cause CUBLAS_STATUS_INTERNAL_ERROR
  // when ffn_stream_ is the capture root. Relaxed mode captures all GPU work
  // associated with ffn_stream_ regardless of which CPU thread submits it.
  cudaGraph_t graph;
  ReportCUDAErrors(cudaStreamBeginCapture(ffn_stream_,
                                           cudaStreamCaptureModeRelaxed));
  for (int i = 0; i < num_encoder_layers_; i++) {
#if LAYERS_EXTERNAL_EVENTS
    ReportCUDAErrors(cudaStreamWaitEvent(ffn_stream_, ln1_done_events_[i],
                                          cudaEventWaitExternal));
#else
    ReportCUDAErrors(cudaStreamWaitEvent(ffn_stream_, ln1_done_events_[i], 0));
#endif
    encoder_weights_[i]->EvalFFNOnly(gemm_n, ffn_stream_, ffn_cublas_,
                                      ffn_buf_wide_, ffn_buf_out_);
#if LAYERS_EXTERNAL_EVENTS
    ReportCUDAErrors(cudaEventRecordWithFlags(ffn_done_events_[i], ffn_stream_,
                                              cudaEventRecordExternal));
#else
    ReportCUDAErrors(cudaEventRecord(ffn_done_events_[i], ffn_stream_));
#endif
  }
  ReportCUDAErrors(cudaStreamEndCapture(ffn_stream_, &graph));
  ReportCUDAErrors(cudaGraphInstantiate(&ffn_graph_execs_[N - 1], graph,
                                         nullptr, nullptr, 0));
  ReportCUDAErrors(cudaGraphDestroy(graph));
}

template <typename DataType>
void AttentionBody<DataType>::LaunchFFNGraph(int N) {
  if (!multi_stream_ffn_ || N < 1 || ffn_graph_execs_.empty() ||
      !ffn_graph_execs_[N - 1])
    return;
  ReportCUDAErrors(cudaGraphLaunch(ffn_graph_execs_[N - 1], ffn_stream_));
}

template <typename DataType>
void AttentionBody<DataType>::JoinFFNStream(cudaEvent_t ev) {
  // Make ffn_stream_ a sibling first-level fork of the CUDA graph capture root.
  // ffn_stream_ waits on upload_done_event_ (recorded on upload_stream_, the
  // capture root), same as compute_stream_. This makes both streams first-level
  // forks, so cuBLAS on ffn_stream_ works inside the captured graph.
  // During normal (non-capture) execution this is a harmless ordering hint:
  // FFN cannot start until data upload is done (already implied by ln1_done_events_
  // transitively, but the explicit wait ensures correct graph topology).
  if (multi_stream_ffn_) {
    ReportCUDAErrors(cudaStreamWaitEvent(ffn_stream_, ev, 0));
  }
}

template <typename DataType>
ValueHead<DataType>::ValueHead(BaseLayer<DataType>* ip,
                               const MultiHeadWeights::ValueHead& weights,
                               void* scratch, bool attention_body, bool wdl,
                               ActivationFunction act, int /*max_batch_size*/,
                               bool use_gemm_ex)
    : BaseLayer<DataType>(weights.ip_val_b.size(), 8, 8, ip),
      embedding_size_(attention_body ? weights.ip_val_b.size()
                                     : weights.value.biases.size()),
      value_hidden_size_(weights.ip1_val_b.size()),
      wdl_(wdl),
      attention_body_(attention_body),
      act_(act) {
  if (attention_body_) {
    allocAndUpload<DataType>(&ip_val_w_, weights.ip_val_w, scratch);
    allocAndUpload<DataType>(&ip_val_b_, weights.ip_val_b, scratch);
  } else {
    conv_ = std::make_unique<Conv1Layer<DataType>>(
        ip, weights.value.biases.size(), 8, 8, ip->GetC(), act, true,
        use_gemm_ex);
    conv_->LoadWeights((float*)&weights.value.weights[0],
                       (float*)&weights.value.biases[0], scratch);
  }

  allocAndUpload<DataType>(&ip1_val_w_, weights.ip1_val_w, scratch);
  allocAndUpload<DataType>(&ip1_val_b_, weights.ip1_val_b, scratch);

  allocAndUpload<DataType>(&ip2_val_w_, weights.ip2_val_w, scratch);
  allocAndUpload<DataType>(&ip2_val_b_, weights.ip2_val_b, scratch);

  // SimPool attentive pooling
  has_simpool_ = weights.simpool_query.size() > 0;
  if (has_simpool_) {
    allocAndUpload<DataType>(&simpool_query_, weights.simpool_query, scratch);
    allocAndUpload<DataType>(&simpool_key_w_, weights.simpool_key_w, scratch);
    allocAndUpload<DataType>(&simpool_key_b_, weights.simpool_key_b, scratch);
  }
}

template <typename DataType>
ValueHead<DataType>::~ValueHead() {
  if (attention_body_) {
    ReportCUDAErrors(cudaFree(ip_val_w_));
    ReportCUDAErrors(cudaFree(ip_val_b_));
  }
  ReportCUDAErrors(cudaFree(ip1_val_w_));
  ReportCUDAErrors(cudaFree(ip1_val_b_));
  ReportCUDAErrors(cudaFree(ip2_val_w_));
  ReportCUDAErrors(cudaFree(ip2_val_b_));
  if (has_simpool_) {
    ReportCUDAErrors(cudaFree(simpool_query_));
    ReportCUDAErrors(cudaFree(simpool_key_w_));
    ReportCUDAErrors(cudaFree(simpool_key_b_));
  }
}

template <typename DataType>
void ValueHead<DataType>::Eval(int N, DataType* output, const DataType* input,
                               const DataType* input2, void* scratch,
                               size_t scratch_size, cudnnHandle_t /*cudnn*/,
                               cublasHandle_t cublas, cudaStream_t stream,
                               DataType***) {
  // Multi-exit override: when AttentionBody has a value branch buffer,
  // SetInputOverride was called to point this head at it. Read from that
  // intermediate-layer flow instead of the network-chained `input` (which
  // would be the FINAL encoder output). Override remains active for the
  // lifetime of the head; nullptr means "use input as normal."
  if (input_override_ != nullptr) {
    input = input_override_;
  }
  DataType* buffer = (DataType*)input2;
  {
    const int num_inputs = this->input_->GetC();
    const int num_outputs = embedding_size_;
    const int batch = N * 64;
    if (attention_body_) {
      cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs,
                            batch, num_inputs, 1.0f, (const DataType*)ip_val_w_,
                            num_inputs, input, num_inputs, 0.0f, buffer,
                            num_outputs);
      addBiasBatched<DataType>(buffer, buffer, ip_val_b_, 1, batch, num_outputs,
                               act_, stream);
    } else {
      conv_->Eval(N, buffer, input, nullptr, scratch, scratch_size, nullptr,
                  cublas, stream);
    }
  }

  if (has_simpool_) {
    // SimPool: attentive pooling instead of flatten.
    // buffer is (N*64, emb) after embedding + activation.
    const int emb = embedding_size_;
    const int batch64 = N * 64;
    DataType* keys = (DataType*)scratch;  // (N*64, emb)
    DataType* attn = keys + batch64 * emb;  // (N, 64) — attention weights
    DataType* pooled = attn + N * 64;  // (N, emb)

    // 1. Key projection: keys = Wk @ buffer + bk
    cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, emb, batch64, emb,
                          1.0f, simpool_key_w_, emb, buffer, emb, 0.0f,
                          keys, emb);
    if (simpool_key_b_) {
      addBiasBatched<DataType>(keys, keys, simpool_key_b_, 1, batch64, emb,
                               ACTIVATION_NONE, stream);
    }

    // 2. Attention: attn[n][s] = query @ keys[n][s]^T / sqrt(emb)
    // query is (1, emb) shared across batch. keys is (N*64, emb).
    // Single GEMM: query(1, emb) × keys^T(emb, N*64) → attn(1, N*64)
    // then reshape attn as (N, 64) for per-batch softmax.
    float scale = 1.0f / sqrtf((float)emb);
    cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                          batch64, 1, emb,
                          scale, keys, emb, simpool_query_, emb,
                          0.0f, attn, batch64);

    // 3. Softmax over 64 positions per batch item
    Softmax<DataType>(N, 64, attn, attn, (const DataType*)nullptr, stream);
    debugDump("val_simpool_attn", attn, 8, stream);

    // 4. Weighted sum: pooled[n] = attn[n] @ buffer[n] → (emb,)
    // attn (N, 1, 64) × buffer (N, 64, emb) → pooled (N, emb)
    cublasXGemmStridedBatched<DataType>(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N, emb, 1, 64,
        1.0f, buffer, emb, 64 * emb,  // V: (emb, 64) per batch
        attn, 64, 64,                   // attn: (64, 1) per batch
        0.0f, pooled, emb, emb,         // out: (emb, 1) per batch
        N, false);

    // 5. Dense1: ip1_val_w(hidden, emb) × pooled(N, emb)
    DataType* layer_out = pooled + N * emb;
    cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, value_hidden_size_,
                          N, emb, 1.0f, ip1_val_w_, emb, pooled, emb, 0.0f,
                          layer_out, value_hidden_size_);
    addBiasBatched<DataType>(layer_out, layer_out, ip1_val_b_, 1, N,
                             value_hidden_size_, act_, stream);
    // Copy to scratch start for dense2 to read from
    ReportCUDAErrors(cudaMemcpyAsync(
        scratch, layer_out, N * value_hidden_size_ * sizeof(DataType),
        cudaMemcpyDeviceToDevice, stream));
  } else {
    // Standard flatten path
    const int num_inputs = embedding_size_ * 64;
    const int num_outputs = value_hidden_size_;
    const int batch = N;
    DataType* layer_out = (DataType*)scratch;
    cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                          num_inputs, 1.0f, (const DataType*)ip1_val_w_,
                          num_inputs, buffer, num_inputs, 0.0f, layer_out,
                          num_outputs);
    addBiasBatched<DataType>(layer_out, layer_out, ip1_val_b_, 1, batch,
                             num_outputs, act_, stream);
  }

  {
    // Value dense 2 (WDL: m=3, non-WDL: m=1).
    const int num_inputs = value_hidden_size_;
    const int num_outputs = wdl_ ? 3 : 1;
    const int batch = N;
    DataType* layer_out = (DataType*)output;
    // Ada Lovelace + CUDA 12.x: during CUDA graph capture cublasHgemm with
    // CUBLAS_TENSOR_OP_MATH only considers graph-capture-compatible kernels,
    // which are tensor-core-aligned and require m%8==0.  m=3 WDL has no such
    // kernel → CUBLAS_STATUS_INTERNAL_ERROR.
    //
    // cublasGemmEx lets us specify computeType independently of the handle's
    // math mode. CUBLAS_COMPUTE_32F_PEDANTIC never uses tensor cores, selects
    // a SIMT kernel for any m, and is graph-capture-compatible on all arches.
    // The fix applies only to FP16 small-m (WDL m=3); everything else goes
    // through the normal cublasXgemm path.
    if (std::is_same<half, DataType>::value && num_outputs < 8) {
      const float alpha = 1.0f, beta = 0.0f;
      ReportCUBLASErrors(cublasGemmEx(
          cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch, num_inputs,
          &alpha,
          (const void*)ip2_val_w_, CUDA_R_16F, num_inputs,
          (const void*)scratch,    CUDA_R_16F, num_inputs,
          &beta,
          (void*)layer_out,        CUDA_R_16F, num_outputs,
          CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
    } else {
      cublasXgemm<DataType>(cublas, CUBLAS_OP_T, CUBLAS_OP_N, num_outputs, batch,
                            num_inputs, 1.0f, (const DataType*)ip2_val_w_,
                            num_inputs, (const DataType*)scratch, num_inputs,
                            0.0f, layer_out, num_outputs);
    }
    addVectors<DataType>(layer_out, layer_out, (DataType*)ip2_val_b_,
                         num_outputs * batch, num_outputs * batch, num_outputs,
                         wdl_ ? ACTIVATION_NONE : act_, stream);
  }
}

// Template instantiation.
#ifdef USE_CUDNN
template class ConvLayer<half>;
template class ConvLayer<float>;
#endif

template class FCLayer<half>;
template class FCLayer<float>;

template class SELayer<half>;
template class SELayer<float>;

template class PolicyMapLayer<half>;
template class PolicyMapLayer<float>;

template class FusedWinogradConvSELayer<half>;
template class FusedWinogradConvSELayer<float>;

template class Conv1Layer<half>;
template class Conv1Layer<float>;

template class ResidualBlock<half>;
template class ResidualBlock<float>;

template class AttentionPolicyHead<half>;
template class AttentionPolicyHead<float>;

template class EncoderBlock<half>;
template class EncoderBlock<float>;

template class AttentionBody<half>;
template class AttentionBody<float>;

template class EmbeddingLayer<half>;
template class EmbeddingLayer<float>;

template class ValueHead<half>;
template class ValueHead<float>;

// Misc error handling stuff.
#ifdef USE_CUDNN
void CudnnError(cudnnStatus_t status, const char* file, const int& line) {
  if (status != CUDNN_STATUS_SUCCESS) {
    char message[128];
    sprintf(message, "CUDNN error: %s (%s:%d) ", cudnnGetErrorString(status),
            file, line);
    CERR << message;
    throw Exception(message);
  }
}
#endif

const char* CublasGetErrorString(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }
  return "unknown cublas error";
}

void CublasError(cublasStatus_t status, const char* file, const int& line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    char message[128];
    sprintf(message, "CUBLAS error: %s (%s:%d) ", CublasGetErrorString(status),
            file, line);
    CERR << message;
    throw Exception(message);
  }
}

void CudaError(cudaError_t status, const char* file, const int& line) {
  if (status != cudaSuccess) {
    char message[128];
    sprintf(message, "CUDA error: %s (%s:%d) ", cudaGetErrorString(status),
            file, line);
    CERR << message;
    throw Exception(message);
  }
}

}  // namespace cudnn_backend
}  // namespace lczero
