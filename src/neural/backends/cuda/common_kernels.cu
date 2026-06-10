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

#include <algorithm>
#include <cassert>

#include "cuda_common.h"
#include "neural/tables/activation_function.h"
#include "neural/tables/attention_policy_map.h"
#include "utils/exception.h"
#include "winograd_helper.inc"

// ── Fused Attention kernels (outside namespace for MSVC CUDA compat) ─────
// Each kernel handles one (batch, head) pair. 64 threads = 64 query rows.
// K, V loaded to shared memory; Q·K^T + bias → softmax → ×V in registers.

__global__ void fusedAttention64_kernel_half(
    half* __restrict__ output, const half* __restrict__ Q,
    const half* __restrict__ K, const half* __restrict__ V,
    const half* __restrict__ smolgen_bias, float scale, int depth,
    int d_model, int num_heads) {
  const int bh = blockIdx.x;
  const int row = threadIdx.x;
  if (row >= 64) return;
  const int batch_idx = bh / num_heads;
  const int head_idx = bh % num_heads;
  const half* q_ptr = Q + (size_t)batch_idx * 64 * d_model + head_idx * depth;
  const half* k_ptr = K + (size_t)batch_idx * 64 * d_model + head_idx * depth;
  const half* v_ptr = V + (size_t)batch_idx * 64 * d_model + head_idx * depth;
  half* out_ptr = output + (size_t)batch_idx * 64 * d_model + head_idx * depth;
  __shared__ float s_k[64][65];  // max depth 64, +1 to avoid bank conflicts
  __shared__ float s_v[64][65];
  for (int d = 0; d < depth; d++) {
    s_k[row][d] = (float)k_ptr[row * d_model + d];
    s_v[row][d] = (float)v_ptr[row * d_model + d];
  }
  __syncthreads();
  float q_reg[64];
  for (int d = 0; d < depth; d++)
    q_reg[d] = (float)q_ptr[row * d_model + d];
  float logits[64];
  float max_val = -1e9f;
  const half* bias_ptr = smolgen_bias
      ? smolgen_bias + (size_t)bh * 64 * 64 + row * 64 : nullptr;
  for (int col = 0; col < 64; col++) {
    float dot = 0.0f;
    for (int d = 0; d < depth; d++)
      dot += q_reg[d] * s_k[col][d];
    dot *= scale;
    if (bias_ptr) dot += (float)bias_ptr[col];
    logits[col] = dot;
    max_val = fmaxf(max_val, dot);
  }
  float sum_exp = 0.0f;
  for (int col = 0; col < 64; col++) {
    logits[col] = expf(logits[col] - max_val);
    sum_exp += logits[col];
  }
  float inv_sum = 1.0f / sum_exp;
  for (int col = 0; col < 64; col++)
    logits[col] *= inv_sum;
  for (int d = 0; d < depth; d++) {
    float acc = 0.0f;
    for (int col = 0; col < 64; col++)
      acc += logits[col] * s_v[col][d];
    out_ptr[row * d_model + d] = (half)acc;
  }
}

__global__ void fusedAttention64_kernel_float(
    float* __restrict__ output, const float* __restrict__ Q,
    const float* __restrict__ K, const float* __restrict__ V,
    const float* __restrict__ smolgen_bias, float scale, int depth,
    int d_model, int num_heads) {
  const int bh = blockIdx.x;
  const int row = threadIdx.x;
  if (row >= 64) return;
  const int batch_idx = bh / num_heads;
  const int head_idx = bh % num_heads;
  const float* q_ptr = Q + (size_t)batch_idx * 64 * d_model + head_idx * depth;
  const float* k_ptr = K + (size_t)batch_idx * 64 * d_model + head_idx * depth;
  const float* v_ptr = V + (size_t)batch_idx * 64 * d_model + head_idx * depth;
  float* out_ptr = output + (size_t)batch_idx * 64 * d_model + head_idx * depth;
  __shared__ float s_k[64][65];
  __shared__ float s_v[64][65];
  for (int d = 0; d < depth; d++) {
    s_k[row][d] = k_ptr[row * d_model + d];
    s_v[row][d] = v_ptr[row * d_model + d];
  }
  __syncthreads();
  float q_reg[64];
  for (int d = 0; d < depth; d++)
    q_reg[d] = q_ptr[row * d_model + d];
  float logits[64];
  float max_val = -1e9f;
  const float* bias_ptr = smolgen_bias
      ? smolgen_bias + (size_t)bh * 64 * 64 + row * 64 : nullptr;
  for (int col = 0; col < 64; col++) {
    float dot = 0.0f;
    for (int d = 0; d < depth; d++)
      dot += q_reg[d] * s_k[col][d];
    dot *= scale;
    if (bias_ptr) dot += bias_ptr[col];
    logits[col] = dot;
    max_val = fmaxf(max_val, dot);
  }
  float sum_exp = 0.0f;
  for (int col = 0; col < 64; col++) {
    logits[col] = expf(logits[col] - max_val);
    sum_exp += logits[col];
  }
  float inv_sum = 1.0f / sum_exp;
  for (int col = 0; col < 64; col++)
    logits[col] *= inv_sum;
  for (int d = 0; d < depth; d++) {
    float acc = 0.0f;
    for (int col = 0; col < 64; col++)
      acc += logits[col] * s_v[col][d];
    out_ptr[row * d_model + d] = acc;
  }
}

namespace lczero {
namespace cudnn_backend {
namespace {
constexpr int kInputPlanes = 112;
}  // namespace

/////////////////////////////////////////////////////////////////////////////
//          Simple CUDA kernels used by certain layers                     //
/////////////////////////////////////////////////////////////////////////////

template <typename T>
__global__ void addVectors_kernel(T* c, T* a, T* b, int size, int asize,
                                  int bsize, ActivationFunction activation) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    float aVal = 0;
    float bVal = 0;
    if (a) aVal = (float)(a[i % asize]);
    if (b) bVal = (float)(b[i % bsize]);

    float cVal = aVal + bVal;

    cVal = activate(cVal, activation);

    c[i] = (T)cVal;
  }
}

// Adds two vectors (possibly of different sizes), also do optional relu
// activation.
template <typename T>
void addVectors(T* c, T* a, T* b, int size, int asize, int bsize,
                ActivationFunction activation, cudaStream_t stream) {
  const int kBlockSize = 256;
  int blocks = DivUp(size, kBlockSize);

  addVectors_kernel<<<blocks, kBlockSize, 0, stream>>>(c, a, b, size, asize,
                                                       bsize, activation);
  ReportCUDAErrors(cudaGetLastError());
}

template <typename T>
__global__ void addVectorsHNC_NHC_kernel(T* a, T* b, int N, int H, int C) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < N * H * C) {
    int orig_i = i;
    int c = i % C;
    i /= C;
    int n = i % N;
    i /= N;
    int h = i;
    float aVal = (float)a[orig_i];
    float bVal = (float)b[n * H * C + h * C + c];

    float cVal = aVal + bVal;

    a[orig_i] = (T)cVal;
  }
}

template <typename T>
void addVectorsHNC_NHC(T* a, T* b, int N, int H, int C, cudaStream_t stream) {
  const int kBlockSize = 256;
  int blocks = DivUp(N * H * C, kBlockSize);
  addVectorsHNC_NHC_kernel<<<blocks, kBlockSize, 0, stream>>>(a, b, N, H, C);

  ReportCUDAErrors(cudaGetLastError());
}

template <typename T, ActivationFunction act>
__global__ void addBiasBatched_kernel(T* output, const T* input, const T* bias,
                                      int N, int C) {
  int batch = blockIdx.y;
  int n = blockIdx.x * blockDim.y + threadIdx.y;
  if (n >= N) return;
  int c = threadIdx.x * 4;

  int biasIndex = batch * C + c;
  int tensorIndex = batch * N * C + n * C + c;

  float val[4];
  float b[4];

  // Load from memory
  const bool fp16 = std::is_same<half, T>::value;
  if (fp16) {
    half inp[4];
    copyAs<uint2>(&inp[0], &input[tensorIndex]);
#pragma unroll
    for (int i = 0; i < 4; i++) val[i] = (float)inp[i];

    copyAs<uint2>(&inp[0], &bias[biasIndex]);
#pragma unroll
    for (int i = 0; i < 4; i++) b[i] = (float)inp[i];
  } else {
    copyAs<uint4>(&val[0], &input[tensorIndex]);
    copyAs<uint4>(&b[0], &bias[biasIndex]);
  }

  // Perform bias add and activation
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float x = val[i] + b[i];
    x = activate(x, act);
    val[i] = x;
  }

  // write to memory
  if (fp16) {
    half op[4];
#pragma unroll
    for (int i = 0; i < 4; i++) op[i] = (half)val[i];
    copyAs<uint2>(&output[tensorIndex], &op[0]);
  } else {
    copyAs<uint4>(&output[tensorIndex], &val[0]);
  }
}

// Input/output tensors are Batch * N * C
// bias tensor is N * C (i.e, different bias for each Batch dimension)
template <typename T>
void addBiasBatched(T* output, const T* input, const T* bias, int Batch, int N,
                    int C, ActivationFunction activation, cudaStream_t stream) {
  // process 4 elements per thread to achieve close to peak memory bandwidth
  if (C % 4 != 0) throw Exception("unsupported filter size");
  if (C > 4096) throw Exception("unsupported filter size");

  dim3 blockDim, gridDim;
  blockDim.x = C / 4;
  blockDim.y = std::min(std::max(512 / blockDim.x, 1u), (unsigned int)N);
  blockDim.z = 1;
  gridDim.x = DivUp(N, blockDim.y);
  gridDim.y = Batch;
  gridDim.z = 1;

  switch (activation) {
    case ACTIVATION_NONE:
      addBiasBatched_kernel<T, ACTIVATION_NONE>
          <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C);
      break;
    case ACTIVATION_SELU:
      addBiasBatched_kernel<T, ACTIVATION_SELU>
          <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C);
      break;
    case ACTIVATION_MISH:
      addBiasBatched_kernel<T, ACTIVATION_MISH>
          <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C);
      break;
    case ACTIVATION_RELU:
      addBiasBatched_kernel<T, ACTIVATION_RELU>
          <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C);
      break;
    case ACTIVATION_SWISH:
      addBiasBatched_kernel<T, ACTIVATION_SWISH>
          <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C);
      break;
    case ACTIVATION_RELU_2:  // square relu
      addBiasBatched_kernel<T, ACTIVATION_RELU_2>
          <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C);
      break;
    default:
      throw Exception(
          "unsupported activation in addBiasBatched. Add in switch-case here");
  }

  ReportCUDAErrors(cudaGetLastError());
}

template <typename T, ActivationFunction act>
__global__ void addBiasBatched_kernel(T* output, const T* input, const T* bias,
                                      int N, int C, int Nstride) {
  int batch = blockIdx.y;
  int n = blockIdx.x * blockDim.y + threadIdx.y;
  if (n >= N) return;
  int c = threadIdx.x * 4;

  int biasIndex = batch * C + c;
  int tensorIndex = batch * Nstride * C + n * C + c;

  float val[4];
  float b[4];

  // Load from memory
  const bool fp16 = std::is_same<half, T>::value;
  if (fp16) {
    half inp[4];
    copyAs<uint2>(&inp[0], &input[tensorIndex]);
#pragma unroll
    for (int i = 0; i < 4; i++) val[i] = (float)inp[i];

    copyAs<uint2>(&inp[0], &bias[biasIndex]);
#pragma unroll
    for (int i = 0; i < 4; i++) b[i] = (float)inp[i];
  } else {
    copyAs<uint4>(&val[0], &input[tensorIndex]);
    copyAs<uint4>(&b[0], &bias[biasIndex]);
  }

  // Perform bias add and activation
#pragma unroll
  for (int i = 0; i < 4; i++) {
    float x = val[i] + b[i];
    x = activate(x, act);
    val[i] = x;
  }

  // write to memory
  if (fp16) {
    half op[4];
#pragma unroll
    for (int i = 0; i < 4; i++) op[i] = (half)val[i];
    copyAs<uint2>(&output[tensorIndex], &op[0]);
  } else {
    copyAs<uint4>(&output[tensorIndex], &val[0]);
  }
}

// Input/output tensors are Batch * N * C
// bias tensor is N * C (i.e, different bias for each Batch dimension)
template <typename T>
void addBiasBatched(T* output, const T* input, const T* bias, int Batch, int N,
                    int C, int Nstride, ActivationFunction activation,
                    cudaStream_t stream) {
  // process 4 elements per thread to achieve close to peak memory bandwidth
  if (C % 4 != 0) throw Exception("unsupported filter size");
  if (C > 4096) throw Exception("unsupported filter size");

  dim3 blockDim, gridDim;
  blockDim.x = C / 4;
  blockDim.y = std::min(std::max(512 / blockDim.x, 1u), (unsigned int)N);
  blockDim.z = 1;
  gridDim.x = DivUp(N, blockDim.y);
  gridDim.y = Batch;
  gridDim.z = 1;

  switch (activation) {
    case ACTIVATION_NONE:
      addBiasBatched_kernel<T, ACTIVATION_NONE>
          <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C,
                                             Nstride);
      break;
    case ACTIVATION_SELU:
      addBiasBatched_kernel<T, ACTIVATION_SELU>
          <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C,
                                             Nstride);
      break;
    case ACTIVATION_MISH:
      addBiasBatched_kernel<T, ACTIVATION_MISH>
          <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C,
                                             Nstride);
      break;
    case ACTIVATION_RELU:
      addBiasBatched_kernel<T, ACTIVATION_RELU>
          <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C,
                                             Nstride);
      break;
    case ACTIVATION_SWISH:
      addBiasBatched_kernel<T, ACTIVATION_SWISH>
          <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C,
                                             Nstride);
      break;
    case ACTIVATION_RELU_2:  // square relu
      addBiasBatched_kernel<T, ACTIVATION_RELU_2>
          <<<gridDim, blockDim, 0, stream>>>(output, input, bias, N, C,
                                             Nstride);
      break;
    default:
      throw Exception(
          "unsupported activation in addBiasBatched. Add in switch-case here");
  }

  ReportCUDAErrors(cudaGetLastError());
}

// Fused bias-add + tanh soft-cap.  Used for the V-projection finalization
// in GLU-V attention paths: PyTorch order is `v = silu(gate)*up; v +=
// pgb_v; v = tanh(v / cap) * cap`.  When `bias` is non-null, behaves as
// addBiasBatched with Batch=1 (bias is a length-C vector broadcast across
// `total / C` rows). When `bias` is null, performs in-place soft-cap only.
// `softcap > 0` is the gate; if 0 (and bias non-null) this degenerates to
// a plain bias add — call addBiasBatched in that case for simplicity.
template <typename T>
__global__ void addBiasAndSoftCap_kernel(T* output, const T* input,
                                          const T* bias, int total, int C,
                                          float softcap) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  float v = (float)input[idx];
  if (bias != nullptr) {
    v += (float)bias[idx % C];
  }
  if (softcap > 0.0f) {
    v = tanhf(v / softcap) * softcap;
  }
  output[idx] = (T)v;
}

template <typename T>
void addBiasAndSoftCap(T* output, const T* input, const T* bias, int total,
                       int C, float softcap, cudaStream_t stream) {
  const int kBlockSize = 256;
  const int blocks = DivUp(total, kBlockSize);
  addBiasAndSoftCap_kernel<T>
      <<<blocks, kBlockSize, 0, stream>>>(output, input, bias, total, C,
                                          softcap);
  ReportCUDAErrors(cudaGetLastError());
}

template <typename T>
__global__ void addBias_NCHW_kernel(T* c, T* a, T* b, int N, int C, int H,
                                    int W, ActivationFunction activation) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  int size = N * C * H * W;
  if (i < size) {
    float aVal = (float)a[i];

    // All this math can be optimized, but the kernel is memory bound anyway.
    int biasIndex = (i / (H * W)) % C;
    float bVal = (float)b[biasIndex];

    float cVal = aVal + bVal;

    cVal = activate(cVal, activation);

    c[i] = (T)cVal;
  }
}

// Add bias to convolution's output.
template <typename T>
void addBias_NCHW(T* c, T* a, T* b, int N, int C, int H, int W,
                  ActivationFunction activation, cudaStream_t stream) {
  int size = N * C * H * W;
  const int kBlockSize = 256;
  int blocks = DivUp(size, kBlockSize);

  addBias_NCHW_kernel<<<blocks, kBlockSize, 0, stream>>>(c, a, b, N, C, H, W,
                                                         activation);
  ReportCUDAErrors(cudaGetLastError());
}

template <typename dT, typename sT>
__device__ dT readNCHW(const sT* input_tensor, int n, int c, int h, int w,
                       int Nin, int Cin, int H, int W) {
  if (n >= Nin || c >= Cin) return 0;

  int index;
  index = n;
  index *= Cin;
  index += c;
  index *= H;
  index += h;
  index *= W;
  index += w;

  return (dT)(input_tensor[index]);
}

template <typename dT, typename sT>
__global__ void NCHWtoNHWC_kernel(dT* output_tensor, const sT* input_tensor,
                                  int Nin, int Cin, int Nout, int Cout, int H,
                                  int W) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= Nout * Cout * H * W) return;

  int index = tid;

  int c = (index % Cout);
  index /= Cout;
  int w = index % W;
  index /= W;
  int h = index % H;
  index /= H;
  int n = index;

  output_tensor[tid] =
      readNCHW<dT, sT>(input_tensor, n, c, h, w, Nin, Cin, H, W);
}

template <typename DstType, typename SrcType>
void convertNCHWtoNHWC(DstType* output_tensor, const SrcType* input_tensor,
                       int Nin, int Cin, int Nout, int Cout, int H, int W,
                       cudaStream_t stream) {
  size_t numElements = Nout * Cout * H * W;
  const int blockSize = 256;
  int blocks = DivUp(numElements, blockSize);
  NCHWtoNHWC_kernel<<<blocks, blockSize, 0, stream>>>(
      output_tensor, input_tensor, Nin, Cin, Nout, Cout, H, W);
}

template <typename DstType, typename SrcType>
__global__ void copyTypeConverted_kernel(DstType* op, SrcType* ip, int N) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= N) return;

  DstType el = (DstType)ip[tid];
  op[tid] = el;
}

template <typename DstType, typename SrcType>
void copyTypeConverted(DstType* op, SrcType* ip, int N, cudaStream_t stream) {
  const int kBlockSize = 256;
  int blocks = DivUp(N, kBlockSize);
  copyTypeConverted_kernel<<<blocks, kBlockSize, 0, stream>>>(op, ip, N);
}

template <typename T>
__global__ void batchNorm_kernel(T* output, const T* input, const T* skipInput,
                                 int N, int C, int H, int W, const float* means,
                                 const float* varMultipliers,
                                 ActivationFunction activation) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;

  int wIndex = 0;
  if (sizeof(T) == sizeof(float))
    wIndex = (index / (H * W)) % C;  // NCHW for fp32.
  else
    wIndex = index % C;  // NHWC for fp16.

  float el = input[index];
  float mean = means[wIndex];
  float varMulti = varMultipliers[wIndex];

  el -= mean;
  el *= varMulti;

  if (skipInput) el += (float)skipInput[index];

  el = activate(el, activation);

  output[index] = (T)el;
}

// Every thread processes single element.
template <typename T>
void batchNorm(T* output, const T* input, const T* skipInput, int N, int C,
               int H, int W, float* means, float* var_multipliers,
               ActivationFunction activation, cudaStream_t stream) {
  const int total_elements = N * C * H * W;
  const int kBlockSize = 256;
  int blocks = DivUp(total_elements, kBlockSize);

  batchNorm_kernel<<<blocks, kBlockSize, 0, stream>>>(
      output, input, skipInput, N, C, H, W, means, var_multipliers, activation);

  ReportCUDAErrors(cudaGetLastError());
}

template <typename T>
__global__ void expandPlanes_kernel_NHWC(T* output, const uint64_t* masks,
                                         const T* values, int n) {
  const int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index >= n * 8 * 8) return;

  const int planeIndex = index % kInputPlanes;
  const int boardIndex = index / (kInputPlanes * 8 * 8);
  const int sqIndex = (index / kInputPlanes) & 0x3F;

  uint64_t mask = masks[boardIndex * kInputPlanes + planeIndex];

  T op = 0;
  bool set = !!(mask & (1ull << sqIndex));
  if (set) {
    op = values[boardIndex * kInputPlanes + planeIndex];
  }
  output[index] = op;
}

template <typename T>
void expandPlanes_NHWC(T* output, const uint64_t* masks, const T* values, int n,
                       cudaStream_t stream) {
  int threads = n * 8 * 8;  // Each thread writes a single element.
  const int kBlockSize = 256;
  int blocks = DivUp(threads, kBlockSize);
  expandPlanes_kernel_NHWC<<<blocks, kBlockSize, 0, stream>>>(output, masks,
                                                              values, n);
  ReportCUDAErrors(cudaGetLastError());
}

template <typename T>
__global__ void expandPlanes_kernel_NCHW(T* output, const uint64_t* masks,
                                         const T* values, unsigned n) {
  unsigned index = threadIdx.x + blockDim.x * blockIdx.x;

  index *= 2;
  unsigned planeIndex = index >> 6;

  if (planeIndex >= n) return;

  uint64_t mask = masks[planeIndex];

  int sqIndex = index & 0x3F;
  T op[2] = {0, 0};

  bool set = !!(mask & (1ull << sqIndex));
  if (set) {
    op[0] = values[planeIndex];
  }
  sqIndex++;
  set = !!(mask & (1ull << sqIndex));
  if (set) {
    op[1] = values[planeIndex];
  }
  output[index + 0] = op[0];
  output[index + 1] = op[1];
}

template <typename T>
void expandPlanes_NCHW(T* output, const uint64_t* masks, const T* values,
                            int n, cudaStream_t stream) {
  unsigned threads = n * 8 * 8 / 2;  // each thread writes two elements.
  const int blockSize = 256;
  unsigned blocks = DivUp(threads, blockSize);
  expandPlanes_kernel_NCHW<<<blocks, blockSize, 0, stream>>>(output, masks,
                                                             values, n);
  ReportCUDAErrors(cudaGetLastError());
}

template <typename T>
__global__ void globalScale_kernel(T* output, const T* input,
                                   const T* scaleBias, const T* prevLayerBias,
                                   int inputSize, int C,
                                   ActivationFunction activation) {
  const int kPlaneSize = 64;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid > inputSize) return;

  int nc = tid / kPlaneSize;
  int n = nc / C;
  int c = nc % C;

  float val1 = input[tid];   // Output of residual block to be scaled.
  float val2 = output[tid];  // Skip connection to be added directly.

  if (prevLayerBias) {
    val1 += (float)(prevLayerBias[c]);
  }

  int startIdx = n * 2 * C;  // Scale and bias interleaved.

  float s = scaleBias[startIdx + c];
  s = 1.0f / (1.0f + exp(-s));  // Sigmoid on scale.

  float b = scaleBias[startIdx + c + C];

  float op = val1 * s + val2 + b;
  op = activate(op, activation);
  output[tid] = (T)op;
}

__global__ void globalScale_kernel_fp16_nhwc(half* output, const half* input,
                                             const half* scaleBias,
                                             const half* prevLayerBias,
                                             int inputSize, int C, int HWC,
                                             ActivationFunction activation) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid > inputSize) return;

  int c = tid % C;
  int n = tid / (HWC);

  float val1 = (float)input[tid];   // Output of residual block to be scaled.
  float val2 = (float)output[tid];  // Skip connection to be added directly.
  if (prevLayerBias) {
    val1 += (float)prevLayerBias[c];
  }

  int startIdx = n * 2 * C;  // Scale and bias interleaved.

  float s = scaleBias[startIdx + c];
  s = 1.0f / (1.0f + exp(-s));  // Sigmoid on scale.

  float b = scaleBias[startIdx + c + C];

  float op = val1 * s + val2 + b;
  op = activate(op, activation);

  output[tid] = (half)op;
}

// N blocks.
// C threads per block.
// 'HWC' input data processed by thread block.
// Each thread writes a single output.
__global__ void globalAvgPool_kernel_NHWC_fp16(half* output, const half* input,
                                               const half* prevLayerBias,
                                               int inputSize, int outputSize) {
  const int elementsPerThread = 64;  // 8x8 board.

  int blockStart = blockIdx.x * blockDim.x;

  float S = 0;

#pragma unroll
  for (int i = 0; i < elementsPerThread; i++) {
    int localIndex = i * blockDim.x + threadIdx.x;
    int inputIndex = blockStart * elementsPerThread + localIndex;
    if (inputIndex < inputSize) S += (float)(input[inputIndex]);
  }

  float avg = S / elementsPerThread;

  // Add bias from previous layer.
  if (prevLayerBias) avg += (float)(prevLayerBias[threadIdx.x]);

  int opIndex = blockStart + threadIdx.x;
  if (opIndex < outputSize) output[opIndex] = (half)avg;
}

// Each thread reads 2 inputs (8x8/32), and each warp writes a single output.
template <typename T>
__global__ void globalAvgPool_kernel(T* output, const T* input,
                                     const T* prevLayerBias, int inputSize,
                                     int outputSize, int C) {
  const int elementsPerWarp = 64;
  const int elementsPerThread = 2;

  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int laneId = threadIdx.x & 0x1F;
  int laneStartIndex = (tid - laneId) * elementsPerThread;

  // Compute per-thread sum for elementsPerThread elements.
  float S = 0;

#pragma unroll
  for (int i = 0; i < elementsPerWarp; i += 32) {
    int index = laneStartIndex + laneId + i;
    if (index < inputSize) S += (float)(input[index]);
  }

// Compute warp wide sum (for entire plane - elementsPerWarp elements).
#pragma unroll
  for (int offset = 1; offset < 32; offset *= 2) {
    S += __shfl_down_sync(0xFFFFFFFF, S, offset);
  }

  float avg = S / elementsPerWarp;
  int opIndex = tid >> 5;

  // First thread in warp has the sum, write it in output.
  if (laneId == 0) {
    if (opIndex < outputSize) {
      if (prevLayerBias) avg += (float)prevLayerBias[opIndex % C];
      output[opIndex] = (T)avg;
    }
  }
}

template <typename T>
void globalAvgPool(int N, int C, T* output, const T* input,
                   const T* prevLayerBias, bool nhwc, cudaStream_t stream) {
  const int kPlaneSize = 64;
  if (nhwc) {
    assert((std::is_same<half, T>::value));
    // For NHWC fp16, simply launch N blocks, each with C threads.
    globalAvgPool_kernel_NHWC_fp16<<<N, C, 0, stream>>>(
        (half*)output, (half*)input, (half*)prevLayerBias, N * C * kPlaneSize,
        N * C);
  } else {
    // For NCHW layout (used with fp32),
    // each warp processes a full plane (64 elements), and writes a single
    // average N*C warps are launched.

    const int kTotalWarps = N * C;
    const int kWarpsPerBlock = 8;
    const int kBlockSize = kWarpsPerBlock * 32;

    int blocks = DivUp(kTotalWarps, kWarpsPerBlock);
    globalAvgPool_kernel<<<blocks, kBlockSize, 0, stream>>>(
        output, input, prevLayerBias, N * C * kPlaneSize, N * C, C);
  }
  ReportCUDAErrors(cudaGetLastError());
}

template <typename T>
void globalScale(int N, int C, T* output, const T* input, const T* scaleBias,
                 const T* prevLayerBias, bool nhwc,
                 ActivationFunction activation, cudaStream_t stream) {
  // Each thread writes one output.
  const int kBlockSize = 256;
  const int kBlocks = DivUp(N * 8 * 8 * C, kBlockSize);

  if (nhwc) {
    assert((std::is_same<half, T>::value));
    globalScale_kernel_fp16_nhwc<<<kBlocks, kBlockSize, 0, stream>>>(
        (half*)output, (half*)input, (half*)scaleBias, (half*)prevLayerBias,
        N * C * 8 * 8, C, 8 * 8 * C, activation);
  } else {
    globalScale_kernel<<<kBlocks, kBlockSize, 0, stream>>>(
        output, input, scaleBias, prevLayerBias, N * C * 8 * 8, C, activation);
  }
  ReportCUDAErrors(cudaGetLastError());
}

template <typename T>
__global__ void policyMap_kernel(T* output, const T* input,
                                 const short* indices, int N, int inputSize,
                                 int usedSize, int outputSize) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int n = tid / usedSize;
  int i = tid % usedSize;

  if (n >= N) return;

  int j = indices[i];

  if (j >= 0) {
    output[n * outputSize + j] = input[n * inputSize + i];
  }
}

template <typename T>
void PolicyMap(int N, T* output, const T* input, const short* indices,
               int inputSize, int usedSize, int outputSize,
               cudaStream_t stream) {
  // Each thread processes one input element
  // Only some of the threads (with valid mapping) write output
  const int kBlockSize = 256;
  const int kBlocks = DivUp(N * usedSize, kBlockSize);

  policyMap_kernel<T><<<kBlocks, kBlockSize, 0, stream>>>(
      (T*)output, (T*)input, (short*)indices, N, inputSize, usedSize,
      outputSize);
  ReportCUDAErrors(cudaGetLastError());
}

template <typename T = float, bool use_se, ActivationFunction activation,
          bool use_bias, bool use_skip>
void OutputInputTransform(int N, int C, int se_K, T* output, const T* input,
                          const T* skip, const T* bias, const T* w1,
                          const T* b1, const T* w2, const T* b2,
                          cudaStream_t stream) {
  // Each thread processes entire chess board
  if (use_se == false) {
    dim3 grid_dim(DivUp(C, kOpInpTransformBlockSize), N, 1);
    OutputTransform_relu_InputTransform_kernel<float, activation, use_bias,
                                               use_skip>
        <<<grid_dim, kOpInpTransformBlockSize, 0, stream>>>(N, C, output, input,
                                                            (float*)skip, bias);
  } else if (C > kMaxResBlockFusingChannels) {
    throw Exception(
        "res block fusing opt not supported for the given data type and no "
        "of filters\n");
  } else {
    OutputTransform_SE_relu_InputTransform_kernel<float, activation, use_bias,
                                                  use_skip>
        <<<N, C, 0, stream>>>(N, C, se_K, output, input, (float*)skip, bias, w1,
                              b1, w2, b2);
  }

  ReportCUDAErrors(cudaGetLastError());
}

__device__ __forceinline__ float clamp(float val, float low, float high) {
  if (__builtin_expect(isnan(val), 0)) return val;
  return fminf(fmaxf(val, low), high);
}

namespace {
constexpr float kTwiceHalfMax = 131008.0f;  // Twice the max finite fp16 value.
}  // namespace

// softmax along C dimension which is assumed to be 64
// each thread processes two elements. Each warp computes a sum (over 64
// elements)
template <typename T>
__global__ void softmax_opt_64_kernel(T* output, const T* input,
                                      const T* input2, int N, float softcap,
                                      float smolgen_cap) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  if (index >= N) return;

  float x[4];
  float ex[2];

  // Load from memory
  const bool fp16 = std::is_same<half, T>::value;
  if (fp16) {
    half inp[2];
    copyAs<int>(&inp[0], &input[index * 2]);
    x[0] = (float)inp[0];
    x[1] = (float)inp[1];
    if (input2 != nullptr) {
      copyAs<int>(&inp[0], &input2[index * 2]);
      x[2] = (float)inp[0];
      x[3] = (float)inp[1];
    }
  } else {
    copyAs<uint2>(&x[0], &input[index * 2]);
    if (input2 != nullptr) {
      copyAs<uint2>(&x[2], &input2[index * 2]);
    }
  }

  if (input2 != nullptr) {
    // Smolgen output soft-cap: tanh-bound the smolgen bias element-wise
    // BEFORE adding it to the QK^T logits. Mirrors PyTorch SmolGen's
    // tanh(out/cap)*cap on the (B,H,64*64) tensor (line ~1685 of
    // torchprocess.py). Fused into the softmax kernel — no extra kernel
    // launch, no extra memory traffic. smolgen_cap <= 0 disables.
    if (smolgen_cap > 0.0f) {
      const float inv_sg = 1.0f / smolgen_cap;
      x[2] = tanhf(x[2] * inv_sg) * smolgen_cap;
      x[3] = tanhf(x[3] * inv_sg) * smolgen_cap;
    }
    x[0] += x[2];
    x[1] += x[3];
  }
  if (fp16) {
    // Guard against Inf from fp16 overflow.
    x[0] = clamp(x[0], -kTwiceHalfMax, kTwiceHalfMax);
    x[1] = clamp(x[1], -kTwiceHalfMax, kTwiceHalfMax);
  }
  // Attention logit soft-cap (Gemma 2 style): applied AFTER all additive
  // biases (including smolgen in input2) and BEFORE softmax. softcap<=0
  // disables it; the caller decides per-call.
  if (softcap > 0.0f) {
    const float inv = 1.0f / softcap;
    x[0] = tanhf(x[0] * inv) * softcap;
    x[1] = tanhf(x[1] * inv) * softcap;
  }
  float threadMax = max(x[0], x[1]);
  float maxval = warpMax(threadMax);
  maxval = __shfl_sync(0xFFFFFFFF, maxval, 0);

  ex[0] = exp(x[0] - maxval);
  ex[1] = exp(x[1] - maxval);

  float threadSum = ex[0] + ex[1];
  float Sum = warpReduce(threadSum);
  Sum = __shfl_sync(0xFFFFFFFF, Sum, 0);

  ex[0] = ex[0] / Sum;
  ex[1] = ex[1] / Sum;

  // Store to memory
  if (fp16) {
    half op[2];
    op[0] = (half)ex[0];
    op[1] = (half)ex[1];
    copyAs<int>(&output[index * 2], &op[0]);
  } else {
    copyAs<uint2>(&output[index * 2], &ex[0]);
  }
}

// N * C Tensors
// performs softmax along the C dimension
// Each thread processes one element
// Sums are computed in shared memory
// C threads per block, N blocks
template <typename T>
__global__ void softmax_kernel(T* output, const T* input, const T* input2,
                               float softcap, float smolgen_cap) {
  int n = blockIdx.x;
  int c = threadIdx.x;
  int C = blockDim.x;
  int index = n * C + c;

  // softmax = tf.exp(logits) / tf.reduce_sum(tf.exp(logits), axis)

  float x = (float)input[index];
  if (input2 != nullptr) {
    float bias = (float)input2[index];
    // Smolgen soft-cap: tanh-bound bias element-wise before the add.
    // Fused (no extra launch). smolgen_cap <= 0 disables.
    if (smolgen_cap > 0.0f) {
      bias = tanhf(bias / smolgen_cap) * smolgen_cap;
    }
    x += bias;
  }
  if (std::is_same<half, T>::value) {
    // Guard against Inf from fp16 overflow.
    x = clamp(x, -kTwiceHalfMax, kTwiceHalfMax);
  }
  // Attention logit soft-cap: see softmax_opt_64_kernel for rationale.
  if (softcap > 0.0f) {
    x = tanhf(x / softcap) * softcap;
  }

  __shared__ float sum, maxval;
  if (c == 0) {
    sum = 0;
    maxval = x;
  }

  __syncthreads();

  // Get max across warp first, and then update across C dimension
  float warpmax = warpMax(x);
  if ((c & 0x1F) == 0) atomicMaxFloat(&maxval, warpmax);

  __syncthreads();

  float ex = exp(x - maxval);

  // compute warp wide sums first
  float val = warpReduce(ex);

  // update shared memory sum across C dimension
  if ((c & 0x1F) == 0) atomicAdd(&sum, val);

  __syncthreads();

  float op = ex / sum;

  output[index] = (T)op;
}

template <typename T>
void Softmax(int N, int C, T* output, const T* input, const T* input2,
             cudaStream_t stream, float softcap, float smolgen_cap) {
  if (C == 64) {
    int size = N * 32;  // Total no of threads needed
    const int kBlockSize = 256;
    int blocks = DivUp(size, kBlockSize);
    softmax_opt_64_kernel<T>
        <<<blocks, kBlockSize, 0, stream>>>(output, input, input2, size,
                                            softcap, smolgen_cap);
  } else {
    softmax_kernel<T><<<N, C, 0, stream>>>(output, input, input2, softcap,
                                            smolgen_cap);
  }

  ReportCUDAErrors(cudaGetLastError());
}

__device__ __forceinline__ float shared_sum_for_layer_norm(float x) {
  // compute warp-wide sum
  float s = warpReduce(x);

  // warp-wide sums
  // Max product of the two dimension for the below array is 16 (512/32), but
  // we make each dimension 16 for simplicity. if shared memory capacity is the
  // bottleneck (it's not), we can convert these to single dim array and
  // dynamically index
  __shared__ float sum[16][16];

  // compute sum across C dimension using the warp wide partial sums
  if (threadIdx.x == 0) sum[threadIdx.z][threadIdx.y] = s;
  __syncthreads();

  if (threadIdx.x == 0 && threadIdx.y == 0) {
    float cSum = 0;
    for (int j = 0; j < blockDim.y; j++) cSum += sum[threadIdx.z][j];
    sum[threadIdx.z][0] = cSum;
  }
  __syncthreads();

  // s now contains the sum across C dimension
  return sum[threadIdx.z][0];
}

// Each thread processes 4 elements
// 1. Perform Bias add, and skip add
// 2. Perform layer norm (normalize across C dimension)
//
// Optional `input2` is an additional pre-activation summand: when non-null,
// the effective input becomes (input + input2 + bias) before activation.
// This lets callers fuse "buffer1 += buffer2" into the LN, saving a kernel
// launch at call sites that combine two sublayer outputs (e.g., parallel
// FFN post-norm: attn_out + ffn_out → fused LN).
//
// Optional `out_add` is an additional post-output per-element summand:
// when non-null, the final written value becomes (output + out_add).
// Used to fuse an external additive bias (e.g., smolgen's exo-as-smolgen
// anchor, added after LN2 normalization but before the weight-gen decoder).
// Separate from `betas` because out_add is per-element (N×C) while betas
// is per-channel (C,) — we can't pre-combine them.
template <typename T>
__global__ void layer_norm_kernel(int N, int C, T* output, const T* input,
                                  const T* bias, const T* skip, const T* gammas,
                                  const T* betas, float ep, float alpha,
                                  ActivationFunction act, const T* input2,
                                  const T* out_add) {
  int n = blockIdx.x * blockDim.z + threadIdx.z;
  if (n >= N) return;
  int c = (threadIdx.y * 32 + threadIdx.x) * 16;
  bool oobThread = c >= C;

  int biasIndex = c;
  int tensorIndex = n * C + c;

  float val[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  float oth[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  const bool fp16 = std::is_same<half, T>::value;
  if (!oobThread) {
    // Load from memory (16 elements a time)
    if (fp16) {
      half inp[8];
      copyAs<uint4>(&inp[0], &input[tensorIndex]);
      for (int i = 0; i < 8; i++) val[i] = (float)inp[i];
      copyAs<uint4>(&inp[0], &input[tensorIndex + 8]);
      for (int i = 0; i < 8; i++) val[i + 8] = (float)inp[i];
      if (bias) {
        copyAs<uint4>(&inp[0], &bias[biasIndex]);
        for (int i = 0; i < 8; i++) oth[i] = (float)inp[i];
        copyAs<uint4>(&inp[0], &bias[biasIndex + 8]);
        for (int i = 0; i < 8; i++) oth[i + 8] = (float)inp[i];
        for (int i = 0; i < 16; i++) val[i] += oth[i];
      }
    } else {
      copyAs<uint4>(&val[0], &input[tensorIndex]);
      copyAs<uint4>(&val[4], &input[tensorIndex + 4]);
      copyAs<uint4>(&val[8], &input[tensorIndex + 8]);
      copyAs<uint4>(&val[12], &input[tensorIndex + 12]);
      if (bias) {
        copyAs<uint4>(&oth[0], &bias[biasIndex]);
        copyAs<uint4>(&oth[4], &bias[biasIndex + 4]);
        copyAs<uint4>(&oth[8], &bias[biasIndex + 8]);
        copyAs<uint4>(&oth[12], &bias[biasIndex + 12]);
        for (int i = 0; i < 16; i++) val[i] += oth[i];
      }
    }
    // Optional second input: fuses "input += input2" into the LN.
    if (input2 != nullptr) {
      if (fp16) {
        half inp[8];
        copyAs<uint4>(&inp[0], &input2[tensorIndex]);
        for (int i = 0; i < 8; i++) oth[i] = (float)inp[i];
        copyAs<uint4>(&inp[0], &input2[tensorIndex + 8]);
        for (int i = 0; i < 8; i++) oth[i + 8] = (float)inp[i];
      } else {
        copyAs<uint4>(&oth[0], &input2[tensorIndex]);
        copyAs<uint4>(&oth[4], &input2[tensorIndex + 4]);
        copyAs<uint4>(&oth[8], &input2[tensorIndex + 8]);
        copyAs<uint4>(&oth[12], &input2[tensorIndex + 12]);
      }
      for (int i = 0; i < 16; i++) val[i] += oth[i];
    }
  }

  if (!oobThread) {
    if (skip != nullptr) {
      // Load from memory (16 elements a time)
      if (fp16) {
        half inp[8];
        copyAs<uint4>(&inp[0], &skip[tensorIndex]);
        for (int i = 0; i < 8; i++) oth[i] = (float)inp[i];
        copyAs<uint4>(&inp[0], &skip[tensorIndex + 8]);
        for (int i = 0; i < 8; i++) oth[i + 8] = (float)inp[i];
      } else {
        copyAs<uint4>(&oth[0], &skip[tensorIndex]);
        copyAs<uint4>(&oth[4], &skip[tensorIndex + 4]);
        copyAs<uint4>(&oth[8], &skip[tensorIndex + 8]);
        copyAs<uint4>(&oth[12], &skip[tensorIndex + 12]);
      }
    }
  }

  // 1. Compute mean
  float s = 0;
  if (!oobThread)
    if (skip != nullptr) {
      for (int i = 0; i < 16; i++) {
        val[i] = activate(val[i], act) * alpha + oth[i];
        s += val[i];
      }
    } else {
      for (int i = 0; i < 16; i++) {
        val[i] = activate(val[i], act) * alpha;
        s += val[i];
      }
    }

  s = shared_sum_for_layer_norm(s);
  float mean = s / C;

  // 2. Compute varience
  s = 0;
  if (!oobThread)
    for (int i = 0; i < 16; i++) {
      float d = val[i] - mean;
      float d_sq = d * d;
      s += d_sq;
    }
  s = shared_sum_for_layer_norm(s);
  float var = s / C;

  if (!oobThread) {
    // Load from memory (16 elements a time)
    if (fp16) {
      half inp[8];
      copyAs<uint4>(&inp[0], &gammas[biasIndex]);
      for (int i = 0; i < 8; i++) oth[i] = (float)inp[i];
      copyAs<uint4>(&inp[0], &gammas[biasIndex + 8]);
      for (int i = 0; i < 8; i++) oth[i + 8] = (float)inp[i];
    } else {
      copyAs<uint4>(&oth[0], &gammas[biasIndex]);
      copyAs<uint4>(&oth[4], &gammas[biasIndex + 4]);
      copyAs<uint4>(&oth[8], &gammas[biasIndex + 8]);
      copyAs<uint4>(&oth[12], &gammas[biasIndex + 12]);
    }
  }

  // 3. Normalize
  for (int i = 0; i < 16; i++) {
    float d = val[i] - mean;
    float norm = d / sqrt(var + ep);
    float op = norm * oth[i];
    val[i] = op;
  }

  if (!oobThread) {
    // Load from memory (16 elements a time)
    if (fp16) {
      half inp[8];
      copyAs<uint4>(&inp[0], &betas[biasIndex]);
      for (int i = 0; i < 8; i++) oth[i] = (float)inp[i];
      copyAs<uint4>(&inp[0], &betas[biasIndex + 8]);
      for (int i = 0; i < 8; i++) oth[i + 8] = (float)inp[i];
    } else {
      copyAs<uint4>(&oth[0], &betas[biasIndex]);
      copyAs<uint4>(&oth[4], &betas[biasIndex + 4]);
      copyAs<uint4>(&oth[8], &betas[biasIndex + 8]);
      copyAs<uint4>(&oth[12], &betas[biasIndex + 12]);
    }
  }

  for (int i = 0; i < 16; i++) {
    val[i] += oth[i];
  }

  // Optional post-output per-element add (fuses e.g. smolgen's exo anchor
  // into the LN kernel, saving an addVectors launch per layer).
  if (!oobThread && out_add != nullptr) {
    if (fp16) {
      half inp[8];
      copyAs<uint4>(&inp[0], &out_add[tensorIndex]);
      for (int i = 0; i < 8; i++) oth[i] = (float)inp[i];
      copyAs<uint4>(&inp[0], &out_add[tensorIndex + 8]);
      for (int i = 0; i < 8; i++) oth[i + 8] = (float)inp[i];
    } else {
      copyAs<uint4>(&oth[0], &out_add[tensorIndex]);
      copyAs<uint4>(&oth[4], &out_add[tensorIndex + 4]);
      copyAs<uint4>(&oth[8], &out_add[tensorIndex + 8]);
      copyAs<uint4>(&oth[12], &out_add[tensorIndex + 12]);
    }
    for (int i = 0; i < 16; i++) val[i] += oth[i];
  }

  if (!oobThread) {
    // Write to memory
    if (fp16) {
      half op[8];
      for (int i = 0; i < 8; i++) op[i] = (half)val[i];
      copyAs<uint4>(&output[tensorIndex], &op[0]);
      for (int i = 0; i < 8; i++) op[i] = (half)val[i + 8];
      copyAs<uint4>(&output[tensorIndex + 8], &op[0]);
    } else {
      copyAs<uint4>(&output[tensorIndex], &val[0]);
      copyAs<uint4>(&output[tensorIndex + 4], &val[4]);
      copyAs<uint4>(&output[tensorIndex + 8], &val[8]);
      copyAs<uint4>(&output[tensorIndex + 12], &val[12]);
    }
  }
}

// add (optional) skip connection to input, and then perform Layer normalization
// normalization is done across C dimension (i.e, sums and std deviations taken
// over elements in C dim)
template <typename T>
void LayerNorm(int N, int C, T* output, const T* input, const T* bias,
               const T* skip, const T* gammas, const T* betas, float ep,
               float alpha, ActivationFunction act, cudaStream_t stream,
               const T* input2, const T* out_add) {
  // process 4 elements per thread to achieve close to peak memory bandwidth
  if (C % 16 != 0) throw Exception("unsupported filter size");
  if (C > 16384) throw Exception("unsupported filter size");

  dim3 blockDim, gridDim;
  blockDim.x = 32;
  blockDim.y = DivUp(C / 16, 32);
  blockDim.z =
      std::min(std::max(512 / (blockDim.x * blockDim.y), 1u), (unsigned int)N);
  gridDim.x = DivUp(N, blockDim.z);
  gridDim.y = 1;
  gridDim.z = 1;

  layer_norm_kernel<T><<<gridDim, blockDim, 0, stream>>>(
      N, C, output, input, bias, skip, gammas, betas, ep, alpha, act, input2,
      out_add);

  ReportCUDAErrors(cudaGetLastError());
}

// ── RMSNorm ──────────────────────────────────────────────────────────────
// RMSNorm: x * gamma * rsqrt(mean(x^2) + eps)
// No mean-centering, no beta.  Supports optional bias-add + skip (same
// semantics as LayerNorm above) so the caller doesn't need extra kernels.
template <typename T>
__global__ void rms_norm_kernel(int N, int C, T* output, const T* input,
                                const T* bias, const T* skip, const T* gammas,
                                float ep, float alpha,
                                ActivationFunction act, const T* input2) {
  int n = blockIdx.x * blockDim.z + threadIdx.z;
  if (n >= N) return;
  int c = (threadIdx.y * 32 + threadIdx.x) * 16;
  bool oobThread = c >= C;

  int biasIndex = c;
  int tensorIndex = n * C + c;

  float val[16] = {0};
  float oth[16] = {0};

  const bool fp16 = std::is_same<half, T>::value;

  // Load input and optional bias
  if (!oobThread) {
    if (fp16) {
      half inp[8];
      copyAs<uint4>(&inp[0], &input[tensorIndex]);
      for (int i = 0; i < 8; i++) val[i] = (float)inp[i];
      copyAs<uint4>(&inp[0], &input[tensorIndex + 8]);
      for (int i = 0; i < 8; i++) val[i + 8] = (float)inp[i];
      if (bias) {
        copyAs<uint4>(&inp[0], &bias[biasIndex]);
        for (int i = 0; i < 8; i++) oth[i] = (float)inp[i];
        copyAs<uint4>(&inp[0], &bias[biasIndex + 8]);
        for (int i = 0; i < 8; i++) oth[i + 8] = (float)inp[i];
        for (int i = 0; i < 16; i++) val[i] += oth[i];
      }
    } else {
      copyAs<uint4>(&val[0], &input[tensorIndex]);
      copyAs<uint4>(&val[4], &input[tensorIndex + 4]);
      copyAs<uint4>(&val[8], &input[tensorIndex + 8]);
      copyAs<uint4>(&val[12], &input[tensorIndex + 12]);
      if (bias) {
        copyAs<uint4>(&oth[0], &bias[biasIndex]);
        copyAs<uint4>(&oth[4], &bias[biasIndex + 4]);
        copyAs<uint4>(&oth[8], &bias[biasIndex + 8]);
        copyAs<uint4>(&oth[12], &bias[biasIndex + 12]);
        for (int i = 0; i < 16; i++) val[i] += oth[i];
      }
    }
    // Optional second input: fuses "input += input2" into RMSNorm.
    if (input2 != nullptr) {
      if (fp16) {
        half inp[8];
        copyAs<uint4>(&inp[0], &input2[tensorIndex]);
        for (int i = 0; i < 8; i++) oth[i] = (float)inp[i];
        copyAs<uint4>(&inp[0], &input2[tensorIndex + 8]);
        for (int i = 0; i < 8; i++) oth[i + 8] = (float)inp[i];
      } else {
        copyAs<uint4>(&oth[0], &input2[tensorIndex]);
        copyAs<uint4>(&oth[4], &input2[tensorIndex + 4]);
        copyAs<uint4>(&oth[8], &input2[tensorIndex + 8]);
        copyAs<uint4>(&oth[12], &input2[tensorIndex + 12]);
      }
      for (int i = 0; i < 16; i++) val[i] += oth[i];
    }
  }

  // Optional skip connection
  if (!oobThread && skip != nullptr) {
    if (fp16) {
      half inp[8];
      copyAs<uint4>(&inp[0], &skip[tensorIndex]);
      for (int i = 0; i < 8; i++) oth[i] = (float)inp[i];
      copyAs<uint4>(&inp[0], &skip[tensorIndex + 8]);
      for (int i = 0; i < 8; i++) oth[i + 8] = (float)inp[i];
    } else {
      copyAs<uint4>(&oth[0], &skip[tensorIndex]);
      copyAs<uint4>(&oth[4], &skip[tensorIndex + 4]);
      copyAs<uint4>(&oth[8], &skip[tensorIndex + 8]);
      copyAs<uint4>(&oth[12], &skip[tensorIndex + 12]);
    }
  }

  // Apply activation + alpha + skip
  float s = 0;
  if (!oobThread) {
    if (skip != nullptr) {
      for (int i = 0; i < 16; i++) {
        val[i] = activate(val[i], act) * alpha + oth[i];
        s += val[i] * val[i];
      }
    } else {
      for (int i = 0; i < 16; i++) {
        val[i] = activate(val[i], act) * alpha;
        s += val[i] * val[i];
      }
    }
  }

  // Compute mean of squares across C
  s = shared_sum_for_layer_norm(s);
  float rms = sqrtf(s / C + ep);

  // Load gammas
  if (!oobThread) {
    if (fp16) {
      half inp[8];
      copyAs<uint4>(&inp[0], &gammas[biasIndex]);
      for (int i = 0; i < 8; i++) oth[i] = (float)inp[i];
      copyAs<uint4>(&inp[0], &gammas[biasIndex + 8]);
      for (int i = 0; i < 8; i++) oth[i + 8] = (float)inp[i];
    } else {
      copyAs<uint4>(&oth[0], &gammas[biasIndex]);
      copyAs<uint4>(&oth[4], &gammas[biasIndex + 4]);
      copyAs<uint4>(&oth[8], &gammas[biasIndex + 8]);
      copyAs<uint4>(&oth[12], &gammas[biasIndex + 12]);
    }
  }

  // Normalize: val * abs(gamma) / rms
  for (int i = 0; i < 16; i++) {
    val[i] = val[i] / rms * fabsf(oth[i]);
  }

  // Write output
  if (!oobThread) {
    if (fp16) {
      half op[8];
      for (int i = 0; i < 8; i++) op[i] = (half)val[i];
      copyAs<uint4>(&output[tensorIndex], &op[0]);
      for (int i = 0; i < 8; i++) op[i] = (half)val[i + 8];
      copyAs<uint4>(&output[tensorIndex + 8], &op[0]);
    } else {
      copyAs<uint4>(&output[tensorIndex], &val[0]);
      copyAs<uint4>(&output[tensorIndex + 4], &val[4]);
      copyAs<uint4>(&output[tensorIndex + 8], &val[8]);
      copyAs<uint4>(&output[tensorIndex + 12], &val[12]);
    }
  }
}

template <typename T>
void RMSNorm(int N, int C, T* output, const T* input, const T* bias,
             const T* skip, const T* gammas, float ep, float alpha,
             ActivationFunction act, cudaStream_t stream, const T* input2) {
  if (C % 16 != 0) throw Exception("RMSNorm: unsupported filter size");
  if (C > 16384) throw Exception("RMSNorm: unsupported filter size");

  dim3 blockDim, gridDim;
  blockDim.x = 32;
  blockDim.y = DivUp(C / 16, 32);
  blockDim.z =
      std::min(std::max(512 / (blockDim.x * blockDim.y), 1u), (unsigned int)N);
  gridDim.x = DivUp(N, blockDim.z);
  gridDim.y = 1;
  gridDim.z = 1;

  rms_norm_kernel<T><<<gridDim, blockDim, 0, stream>>>(
      N, C, output, input, bias, skip, gammas, ep, alpha, act, input2);

  ReportCUDAErrors(cudaGetLastError());
}

// ── SwiGLU fused elementwise (with optional bias fusion) ────────────────
// output = silu(gate + gate_bias) * (up + up_bias)  where silu(x) = x * sigmoid(x)
// gate/up tensors are Batch * N * C (same as addBiasBatched layout).
// When gate_bias/up_bias are non-null, biases are applied inside the kernel,
// saving 2 kernel launches per SwiGLU block. `dff` is the per-token feature
// dimension used for bias indexing (ignored when biases are null).
// Template-specialized variant: lift null-pointer / zero-cap checks to
// compile time so the runtime predicates are dead-code-eliminated.
// Dispatcher (in SwiGLUElementwiseWithBias below) picks the right
// specialization based on which optional args are present.
template <typename T, bool kHasGate, bool kHasUp, bool kHasSoftcap,
          bool kHasPgb>
__global__ void swiglu_kernel_spec(T* output, const T* gate, const T* up,
                                    const T* gate_bias, const T* up_bias,
                                    const T* pgb_bias,
                                    int total, int dff,
                                    float swiglu_softcap) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;

  float g = (float)gate[i];
  float u = (float)up[i];
  if constexpr (kHasGate) g += (float)gate_bias[i % dff];
  if constexpr (kHasUp)   u += (float)up_bias[i % dff];

  float sig = 1.0f / (1.0f + expf(-g));
  float out = g * sig * u;
  if constexpr (kHasSoftcap) {
    out = swiglu_softcap * tanhf(out / swiglu_softcap);
  }
  if constexpr (kHasPgb) out += (float)pgb_bias[i % dff];
  output[i] = (T)out;
}

// Generic fallback: all flags are runtime — used for uncommon
// configurations that the dispatcher doesn't have a specialization for.
template <typename T>
__global__ void swiglu_kernel(T* output, const T* gate, const T* up,
                              const T* gate_bias, const T* up_bias,
                              const T* pgb_bias,
                              int total, int dff, float swiglu_softcap) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;

  float g = (float)gate[i];
  float u = (float)up[i];
  if (gate_bias) g += (float)gate_bias[i % dff];
  if (up_bias)   u += (float)up_bias[i % dff];

  // silu(g) * u = g * sigmoid(g) * u
  float sig = 1.0f / (1.0f + expf(-g));
  float out = g * sig * u;
  // SwiGLU soft-cap: tanh-bound the silu*up product. 0 disables.
  // Must match training (PyTorch torchprocess.py applies same tanh at the
  // FFN's silu*up output position).
  if (swiglu_softcap > 0.0f) {
    out = swiglu_softcap * tanhf(out / swiglu_softcap);
  }
  // Optional PGB add (post-multiply, post-softcap): out += pgb[i % dff].
  // Fuses the separate addBiasBatched(ffn_pgb_) launch into this kernel.
  // PyTorch SwiGLUFFN order is: `hidden = silu(g)*u + pgb`, matching here.
  if (pgb_bias) out += (float)pgb_bias[i % dff];
  output[i] = (T)out;
}

template <typename T>
void SwiGLUElementwise(int total, T* output, const T* gate, const T* up,
                       const T* gate_bias, const T* up_bias,
                       cudaStream_t stream, float swiglu_softcap) {
  const int kBlockSize = 256;
  int blocks = DivUp(total, kBlockSize);
  // When biases are null, dff is unused. Otherwise pass it via a separate API
  // call (SwiGLUElementwiseWithBias) that includes dff explicitly.
  swiglu_kernel<T><<<blocks, kBlockSize, 0, stream>>>(
      output, gate, up, gate_bias, up_bias, /*pgb_bias=*/nullptr, total,
      /*dff=*/1, swiglu_softcap);
  ReportCUDAErrors(cudaGetLastError());
}

// Fused SwiGLU with biases applied inside the kernel. Saves 2 kernel launches.
// pgb_bias (optional): broadcasts (dff,) PGB add over the silu*up output.
//
// Dispatcher: picks a template-specialized kernel for the hot paths
// (all 4 set = FFN with softcap+pgb; gate+up only = V-side attention),
// falls through to the generic runtime-branched kernel for uncommon
// configurations.  Saves ~20-30% on the per-thread instruction count
// for the bandwidth-bound SwiGLU kernel.
template <typename T>
void SwiGLUElementwiseWithBias(int total, int dff, T* output, const T* gate,
                                const T* up, const T* gate_bias,
                                const T* up_bias, cudaStream_t stream,
                                float swiglu_softcap, const T* pgb_bias) {
  const int kBlockSize = 256;
  int blocks = DivUp(total, kBlockSize);
  const bool hg = gate_bias != nullptr;
  const bool hu = up_bias   != nullptr;
  const bool hs = swiglu_softcap > 0.0f;
  const bool hp = pgb_bias  != nullptr;
  if (hg && hu && hs && hp) {
    swiglu_kernel_spec<T, true, true, true, true>
        <<<blocks, kBlockSize, 0, stream>>>(
            output, gate, up, gate_bias, up_bias, pgb_bias, total, dff,
            swiglu_softcap);
  } else if (hg && hu && hs && !hp) {
    swiglu_kernel_spec<T, true, true, true, false>
        <<<blocks, kBlockSize, 0, stream>>>(
            output, gate, up, gate_bias, up_bias, pgb_bias, total, dff,
            swiglu_softcap);
  } else if (hg && hu && !hs && !hp) {
    // V-side SwiGLU (no FFN softcap, no FFN pgb).
    swiglu_kernel_spec<T, true, true, false, false>
        <<<blocks, kBlockSize, 0, stream>>>(
            output, gate, up, gate_bias, up_bias, pgb_bias, total, dff,
            swiglu_softcap);
  } else {
    swiglu_kernel<T><<<blocks, kBlockSize, 0, stream>>>(
        output, gate, up, gate_bias, up_bias, pgb_bias, total, dff,
        swiglu_softcap);
  }
  ReportCUDAErrors(cudaGetLastError());
}

// Explicit instantiations
template void SwiGLUElementwiseWithBias<float>(int, int, float*, const float*,
                                                const float*, const float*,
                                                const float*, cudaStream_t,
                                                float, const float*);
template void SwiGLUElementwiseWithBias<half>(int, int, half*, const half*,
                                               const half*, const half*,
                                               const half*, cudaStream_t,
                                               float, const half*);

// ── SwiGLU fused with bias + interleaved gate_up layout ──────────────────
// Input `gate_up` has shape (batch, 2*dff) in column-major: per batch element
// the first dff values are gate, next dff are up. This is the direct output
// of a single batched GEMM producing [gate; up].
// Output: (batch, dff) = silu(gate + gate_bias) * (up + up_bias) + pgb_bias
// (last term optional, fused to save the pgb_ffn add launch on the FFN
// bottleneck stream).
//
// column_stride parameter (default 2*dff): for an isolated [gate;up]
// buffer it's 2*dff; for the fused-QKV path where gate_up is embedded
// in a wider (qkv_total_M, batch) buffer with the gate/up block at some
// offset, column_stride is the wider stride (e.g. 1280) and the caller
// passes a pointer already advanced to the start of the gate block.
// Template-specialized variant for the FFN bottleneck path.  Same
// fusion as swiglu_fused_kernel but with compile-time control over the
// optional ops — dead-code elimination removes the runtime branches.
template <typename T, bool kHasGate, bool kHasUp, bool kHasSoftcap,
          bool kHasPgb>
__global__ void swiglu_fused_kernel_spec(T* output, const T* gate_up,
                                          const T* gate_bias,
                                          const T* up_bias,
                                          const T* pgb_bias,
                                          int batch, int dff,
                                          int column_stride,
                                          float swiglu_softcap) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * dff;
  if (idx >= total) return;

  int b = idx / dff;
  int d = idx % dff;

  float g = (float)gate_up[b * column_stride + d];
  float u = (float)gate_up[b * column_stride + dff + d];
  if constexpr (kHasGate) g += (float)gate_bias[d];
  if constexpr (kHasUp)   u += (float)up_bias[d];

  float sig = 1.0f / (1.0f + expf(-g));
  float out = g * sig * u;
  if constexpr (kHasSoftcap) {
    out = swiglu_softcap * tanhf(out / swiglu_softcap);
  }
  if constexpr (kHasPgb) out += (float)pgb_bias[d];
  output[idx] = (T)out;
}

// Generic runtime-branched fallback for uncommon configurations.
template <typename T>
__global__ void swiglu_fused_kernel(T* output, const T* gate_up,
                                    const T* gate_bias, const T* up_bias,
                                    const T* pgb_bias,
                                    int batch, int dff, int column_stride,
                                    float swiglu_softcap) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * dff;
  if (idx >= total) return;

  int b = idx / dff;
  int d = idx % dff;

  // Column-major with per-column stride = column_stride.  Within the
  // column, gate occupies rows [0..dff), up rows [dff..2*dff).
  float g = (float)gate_up[b * column_stride + d];
  float u = (float)gate_up[b * column_stride + dff + d];
  if (gate_bias) g += (float)gate_bias[d];
  if (up_bias)   u += (float)up_bias[d];

  float sig = 1.0f / (1.0f + expf(-g));
  float out = g * sig * u;
  // SwiGLU soft-cap: tanh-bound the silu*up product. 0 disables.
  // Must match training (PyTorch torchprocess.py applies same tanh).
  if (swiglu_softcap > 0.0f) {
    out = swiglu_softcap * tanhf(out / swiglu_softcap);
  }
  // Optional PGB add (post-multiply, post-softcap).  Same bias position
  // as the separate addBiasBatched the FFN path used to call after this
  // kernel — folding it here saves one launch on the bottleneck stream.
  if (pgb_bias) out += (float)pgb_bias[d];
  output[idx] = (T)out;
}

// Dispatcher: hot-path specializations for the FFN bottleneck and the
// V-side attention call (which omits softcap+pgb).  Other configurations
// fall through to the generic runtime-branched kernel.
template <typename T>
void SwiGLUFusedGateUp(int batch, int dff, T* output, const T* gate_up,
                      const T* gate_bias, const T* up_bias,
                      cudaStream_t stream, float swiglu_softcap,
                      const T* pgb_bias, int column_stride) {
  const int kBlockSize = 256;
  int total = batch * dff;
  int blocks = DivUp(total, kBlockSize);
  // Default column_stride = 2*dff for the standalone [gate;up] buffer.
  const int cs = (column_stride > 0) ? column_stride : (2 * dff);
  const bool hg = gate_bias != nullptr;
  const bool hu = up_bias   != nullptr;
  const bool hs = swiglu_softcap > 0.0f;
  const bool hp = pgb_bias  != nullptr;
  if (hg && hu && hs && hp) {
    // FFN bottleneck path: gate+up+softcap+pgb (user's config).
    swiglu_fused_kernel_spec<T, true, true, true, true>
        <<<blocks, kBlockSize, 0, stream>>>(
            output, gate_up, gate_bias, up_bias, pgb_bias, batch, dff, cs,
            swiglu_softcap);
  } else if (hg && hu && hs && !hp) {
    // FFN without pgb.
    swiglu_fused_kernel_spec<T, true, true, true, false>
        <<<blocks, kBlockSize, 0, stream>>>(
            output, gate_up, gate_bias, up_bias, pgb_bias, batch, dff, cs,
            swiglu_softcap);
  } else if (hg && hu && !hs && !hp) {
    // V-side attention SwiGLU (no softcap, no pgb).
    swiglu_fused_kernel_spec<T, true, true, false, false>
        <<<blocks, kBlockSize, 0, stream>>>(
            output, gate_up, gate_bias, up_bias, pgb_bias, batch, dff, cs,
            swiglu_softcap);
  } else {
    swiglu_fused_kernel<T><<<blocks, kBlockSize, 0, stream>>>(
        output, gate_up, gate_bias, up_bias, pgb_bias, batch, dff, cs,
        swiglu_softcap);
  }
  ReportCUDAErrors(cudaGetLastError());
}

// Explicit instantiation
template void SwiGLUFusedGateUp<float>(int, int, float*, const float*,
                                        const float*, const float*,
                                        cudaStream_t, float, const float*, int);
template void SwiGLUFusedGateUp<half>(int, int, half*, const half*,
                                       const half*, const half*,
                                       cudaStream_t, float, const half*, int);

// ── Concat per-square + broadcast global ─────────────────────────────────
// For light embedding: merge per-square MLP output with global summary.
// output[n][sq][c] = sq_features[n*64+sq][c]       if c < sq_size
//                  = global_features[n][c-sq_size]  if c >= sq_size
template <typename T>
__global__ void concat_sq_global_kernel(T* output, const T* sq_features,
                                         const T* global_features,
                                         int sq_size, int global_size,
                                         int total_c) {
  int n = blockIdx.x;
  int sq = blockIdx.y;
  int c = threadIdx.x;
  if (c >= total_c) return;
  int out_idx = n * 64 * total_c + sq * total_c + c;
  if (c < sq_size) {
    output[out_idx] = sq_features[(n * 64 + sq) * sq_size + c];
  } else {
    output[out_idx] = global_features[n * global_size + (c - sq_size)];
  }
}

// ── Fused Attention wrapper (calls kernels defined outside namespace) ────
template <typename T>
void FusedAttention64(int N, int num_heads, int depth, int d_model,
                      T* output, const T* Q, const T* K, const T* V,
                      const T* smolgen_bias, cudaStream_t stream) {
  assert(depth <= 64 && "FusedAttention64: depth exceeds max (64)");
  const int num_blocks = N * num_heads;
  const float scale = 1.0f / sqrtf((float)depth);
  if constexpr (std::is_same<T, half>::value) {
    fusedAttention64_kernel_half<<<num_blocks, 64, 0, stream>>>(
        output, Q, K, V, smolgen_bias, scale, depth, d_model, num_heads);
  } else {
    fusedAttention64_kernel_float<<<num_blocks, 64, 0, stream>>>(
        output, Q, K, V, smolgen_bias, scale, depth, d_model, num_heads);
  }
  ReportCUDAErrors(cudaGetLastError());
}

template <typename T>
void ConcatSquareAndGlobal(int N, int sq_size, int global_size,
                           T* output, const T* sq_features,
                           const T* global_features, cudaStream_t stream) {
  int total_c = sq_size + global_size;
  dim3 grid(N, 64);
  concat_sq_global_kernel<T><<<grid, total_c, 0, stream>>>(
      output, sq_features, global_features, sq_size, global_size, total_c);
  ReportCUDAErrors(cudaGetLastError());
}

// ── Weighted blend ───────────────────────────────────────────────────────
// output[i] = alpha * a[i] + beta * b[i]
template <typename T>
__global__ void weighted_add_kernel(T* output, float alpha, const T* a,
                                     float beta, const T* b, int total) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  output[i] = (T)(alpha * (float)a[i] + beta * (float)b[i]);
}

template <typename T>
void WeightedAdd(int total, T* output, float alpha, const T* a,
                 float beta, const T* b, cudaStream_t stream) {
  const int kBlockSize = 256;
  int blocks = DivUp(total, kBlockSize);
  weighted_add_kernel<T><<<blocks, kBlockSize, 0, stream>>>(
      output, alpha, a, beta, b, total);
  ReportCUDAErrors(cudaGetLastError());
}

// ── Element-wise multiply ────────────────────────────────────────────────
// output[i] = a[i] * b[i].  Used for VGA-E where sigmoid is pre-applied.
template <typename T>
__global__ void elementwise_mul_kernel(T* output, const T* a, const T* b,
                                       int total) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  output[i] = (T)((float)a[i] * (float)b[i]);
}

template <typename T>
void ElementwiseMultiply(int total, T* output, const T* a, const T* b,
                         cudaStream_t stream) {
  const int kBlockSize = 256;
  int blocks = DivUp(total, kBlockSize);
  elementwise_mul_kernel<T><<<blocks, kBlockSize, 0, stream>>>(
      output, a, b, total);
  ReportCUDAErrors(cudaGetLastError());
}

// ── Fused VGA-E: sigmoid(gate + bias) * output, in one pass ─────────────
// Replaces: addVectors(sigmoid) + ElementwiseMultiply with a single kernel.
// output[i] *= sigmoid(gate[i] + bias[i % bias_size])
__global__ void fusedVGAE_kernel_half(
    half* output, const half* gate, const half* bias,
    int total, int bias_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  float g = (float)gate[i] + (float)bias[i % bias_size];
  float s = 1.0f / (1.0f + expf(-g));
  output[i] = (half)((float)output[i] * s);
}

__global__ void fusedVGAE_kernel_float(
    float* output, const float* gate, const float* bias,
    int total, int bias_size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  float g = gate[i] + bias[i % bias_size];
  float s = 1.0f / (1.0f + expf(-g));
  output[i] = output[i] * s;
}

template <typename T>
void FusedVGAE(int total, T* output, const T* gate, const T* bias,
               int bias_size, cudaStream_t stream) {
  const int kBlockSize = 256;
  int blocks = DivUp(total, kBlockSize);
  if constexpr (std::is_same<T, half>::value) {
    fusedVGAE_kernel_half<<<blocks, kBlockSize, 0, stream>>>(
        output, gate, bias, total, bias_size);
  } else {
    fusedVGAE_kernel_float<<<blocks, kBlockSize, 0, stream>>>(
        output, gate, bias, total, bias_size);
  }
  ReportCUDAErrors(cudaGetLastError());
}

// ── Shared gate bank adapters (A-Layout-2) ──────────────────────────────
// The encoder layer's bank h = silu(W_bank x + b) is column-major
// (bank_dim, tokens), ld = bank_dim — same layout every projection GEMM in
// this backend produces.  A site adapter forms its gate pre-activation as
//   pre[c] = diag[c] * h[token, c] + gate_b[c] (+ lrb[token, c])
// where lrb is the optional rank-r mixer output (lr_b @ lr_a @ h), already
// materialized at (out_dim, tokens) by two small GEMMs.
//
// bank_gated_mul: V/FFN consumption — out = silu(pre) * (up + up_b)
// [+ pgb] [softcap].  out == up aliasing is safe (same-index read/write).
// NOTE on the rank-r mixer (lr_b) stage: folding it into these kernels as
// a per-element rank-length dot was tried and measured SLOWER than the
// GEMM form — 4.5x slower with channel-major lr_b (uncoalesced), still
// 1.4x slower with rank-major (coalesced) lr_b, because the fold turns a
// tensor-core GEMM into ~0.5 GB/site/layer of L2-bandwidth-bound scalar
// loads.  The lrb tensors are therefore materialized by small cuBLAS
// GEMMs (see EncoderBlock::Eval) and consumed here via a plain indexed
// read.
template <typename T>
__global__ void bank_gated_mul_kernel(int total, int out_dim, int bank_dim,
                                      T* output, const T* h, const T* lrb,
                                      const T* up, const T* diag,
                                      const T* gate_b, const T* up_b,
                                      const T* pgb, float softcap,
                                      int up_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int c = i % out_dim;
  const size_t row = (size_t)(i / out_dim);
  float pre = (float)diag[c] * (float)h[row * bank_dim + c] + (float)gate_b[c];
  if (lrb) pre += (float)lrb[i];
  const float gate = pre / (1.0f + expf(-pre));  // silu
  float u = (float)up[row * (size_t)up_stride + c];
  if (up_b) u += (float)up_b[c];
  float val = gate * u;
  if (pgb) val += (float)pgb[c];
  if (softcap > 0.0f) val = softcap * tanhf(val / softcap);
  output[i] = (T)val;
}

template <typename T>
void BankGatedMul(int batch, int out_dim, int bank_dim, T* output,
                  const T* h, const T* lrb, const T* up, const T* diag,
                  const T* gate_b, const T* up_b, const T* pgb,
                  float softcap, int up_stride, cudaStream_t stream) {
  const int total = batch * out_dim;
  const int kBlockSize = 256;
  int blocks = DivUp(total, kBlockSize);
  if (up_stride <= 0) up_stride = out_dim;
  bank_gated_mul_kernel<T><<<blocks, kBlockSize, 0, stream>>>(
      total, out_dim, bank_dim, output, h, lrb, up, diag, gate_b, up_b, pgb,
      softcap, up_stride);
  ReportCUDAErrors(cudaGetLastError());
}

// fused_vgae_bank: VGA-E consumption — output *= sigmoid(pre).
template <typename T>
__global__ void fused_vgae_bank_kernel(int total, int dim, int bank_dim,
                                       T* output, const T* h, const T* lrb,
                                       const T* diag, const T* bias) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int c = i % dim;
  const size_t row = (size_t)(i / dim);
  float pre = (float)diag[c] * (float)h[row * bank_dim + c] + (float)bias[c];
  if (lrb) pre += (float)lrb[i];
  const float s = 1.0f / (1.0f + expf(-pre));
  output[i] = (T)((float)output[i] * s);
}

template <typename T>
void FusedVGAEBank(int total, T* output, const T* h, int bank_dim,
                   const T* lrb, const T* diag, const T* bias, int dim,
                   cudaStream_t stream) {
  const int kBlockSize = 256;
  int blocks = DivUp(total, kBlockSize);
  fused_vgae_bank_kernel<T><<<blocks, kBlockSize, 0, stream>>>(
      total, dim, bank_dim, output, h, lrb, diag, bias);
  ReportCUDAErrors(cudaGetLastError());
}

// ── Fused Residual Add + LayerNorm ──────────────────────────────────────
// Replaces: addVectors(residual) + NormLayer with a single kernel.
// output[i] = LN(residual[i] + delta[i])
// One pass: compute sum, then mean/variance, then normalize.
// Each block handles one row of the (tokens, emb) matrix.
// Fused Residual Add + LayerNorm: ln_output = LN(residual + delta),
// and optionally writes residual_output = residual + delta.
// residual_output can be the same pointer as residual (in-place update).
// If residual_output is nullptr, only ln_output is written.
__global__ void fusedResAddLN_kernel_half(
    half* ln_output, half* residual_output,
    const half* residual, const half* delta,
    const half* gamma, const half* beta, int emb, float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  extern __shared__ float smem[];
  float* row_data = smem;

  for (int d = tid; d < emb; d += blockDim.x) {
    float val = (float)residual[row * emb + d] + (float)delta[row * emb + d];
    row_data[d] = val;
    if (residual_output) residual_output[row * emb + d] = (half)val;
  }
  __syncthreads();

  float sum = 0.0f;
  for (int d = tid; d < emb; d += blockDim.x)
    sum += row_data[d];
  for (int offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  __shared__ float s_mean, s_var;
  if (tid == 0) s_mean = sum / emb;
  __syncthreads();

  float var_sum = 0.0f;
  for (int d = tid; d < emb; d += blockDim.x) {
    float diff = row_data[d] - s_mean;
    var_sum += diff * diff;
  }
  for (int offset = 16; offset > 0; offset >>= 1)
    var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
  if (tid == 0) s_var = var_sum / emb;
  __syncthreads();

  float inv_std = rsqrtf(s_var + eps);
  for (int d = tid; d < emb; d += blockDim.x) {
    float normed = (row_data[d] - s_mean) * inv_std;
    float g = gamma ? (float)gamma[d] : 1.0f;
    float b = beta ? (float)beta[d] : 0.0f;
    ln_output[row * emb + d] = (half)(normed * g + b);
  }
}

__global__ void fusedResAddLN_kernel_float(
    float* ln_output, float* residual_output,
    const float* residual, const float* delta,
    const float* gamma, const float* beta, int emb, float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  extern __shared__ float smem[];
  float* row_data = smem;

  for (int d = tid; d < emb; d += blockDim.x) {
    float val = residual[row * emb + d] + delta[row * emb + d];
    row_data[d] = val;
    if (residual_output) residual_output[row * emb + d] = val;
  }
  __syncthreads();

  float sum = 0.0f;
  for (int d = tid; d < emb; d += blockDim.x)
    sum += row_data[d];
  for (int offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(0xffffffff, sum, offset);
  __shared__ float s_mean, s_var;
  if (tid == 0) s_mean = sum / emb;
  __syncthreads();

  float var_sum = 0.0f;
  for (int d = tid; d < emb; d += blockDim.x) {
    float diff = row_data[d] - s_mean;
    var_sum += diff * diff;
  }
  for (int offset = 16; offset > 0; offset >>= 1)
    var_sum += __shfl_down_sync(0xffffffff, var_sum, offset);
  if (tid == 0) s_var = var_sum / emb;
  __syncthreads();

  float inv_std = rsqrtf(s_var + eps);
  for (int d = tid; d < emb; d += blockDim.x) {
    float normed = (row_data[d] - s_mean) * inv_std;
    float g = gamma ? gamma[d] : 1.0f;
    float b = beta ? beta[d] : 0.0f;
    ln_output[row * emb + d] = normed * g + b;
  }
}

template <typename T>
void FusedResidualAddLN(int tokens, int emb, T* ln_output,
                        T* residual_output, const T* residual,
                        const T* delta, const T* gamma, const T* beta,
                        float eps, cudaStream_t stream) {
  int threads = min(256, emb);
  int smem_bytes = emb * sizeof(float);
  if constexpr (std::is_same<T, half>::value) {
    fusedResAddLN_kernel_half<<<tokens, threads, smem_bytes, stream>>>(
        ln_output, residual_output, residual, delta, gamma, beta, emb, eps);
  } else {
    fusedResAddLN_kernel_float<<<tokens, threads, smem_bytes, stream>>>(
        ln_output, residual_output, residual, delta, gamma, beta, emb, eps);
  }
  ReportCUDAErrors(cudaGetLastError());
}

// ── Fused 3-way vector add (for parallel FFN: out = a + b + c) ───────────
// Saves one kernel launch + one pass over the residual stream compared to
// two sequential addVectors calls.
template <typename T>
__global__ void add3_kernel(T* output, const T* a, const T* b, const T* c,
                            int total) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  output[i] = (T)((float)a[i] + (float)b[i] + (float)c[i]);
}

template <typename T>
void Add3(int total, T* output, const T* a, const T* b, const T* c,
          cudaStream_t stream) {
  const int kBlockSize = 256;
  int blocks = DivUp(total, kBlockSize);
  add3_kernel<T><<<blocks, kBlockSize, 0, stream>>>(output, a, b, c, total);
  ReportCUDAErrors(cudaGetLastError());
}

template void Add3<float>(int, float*, const float*, const float*,
                           const float*, cudaStream_t);
template void Add3<half>(int, half*, const half*, const half*, const half*,
                          cudaStream_t);

// ── GPU-side optimistic policy blend ─────────────────────────────────────
// Linear interpolation of vanilla and optimistic policy LOGITS in place
// (writes back into the optimistic buffer).  Math:
//
//   blended_logit[i] = (1 - α) · vanilla_logit[i] + α · optimistic_logit[i]
//
// After the host-side softmax (wrapper.cc SoftmaxPolicy), this is
// mathematically equivalent to the per-edge geometric blend
// P_main^(1-α) · P_opt^α that the CPU blend computes per node:
//
//   softmax((1-α)·L_v + α·L_o) = softmax(L_v)^(1-α) · softmax(L_o)^α / Z
//
// Constants drop out of softmax, so the linear blend in logit space
// recovers the geometric blend in probability space exactly.  Saves the
// per-node CPU blend cost in search.cc (~30 pow + sum + renorm per
// node) by doing one cheap linear blend once per inference on GPU.
//
// Triggered by setting backend option `gpu_blend_alpha` to a value in
// (0, 1).  Search then needs to be configured with --optimistic-policy-
// weight=1.0 and --optimistic-policy-weight-internal=1.0 so it uses
// the fast path (SetP from p_optimistic directly without further
// blending) — the GPU has already done the blend.  At α_root = α_
// internal the GPU blend is mathematically complete; at split alphas
// don't enable the GPU blend (fall back to the per-node CPU path).
template <typename T>
__global__ void blendPolicyLogits_kernel(T* output, const T* vanilla,
                                          const T* optimistic, float alpha,
                                          int total) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const float v = (float)vanilla[i];
  const float o = (float)optimistic[i];
  output[i] = (T)((1.0f - alpha) * v + alpha * o);
}

template <typename T>
void BlendPolicyLogits(int total, T* output, const T* vanilla,
                       const T* optimistic, float alpha, cudaStream_t stream) {
  // `output` may alias either `vanilla` or `optimistic` — each thread
  // does the two reads before the single write, so in-place is safe.
  const int kBlockSize = 256;
  int blocks = DivUp(total, kBlockSize);
  blendPolicyLogits_kernel<T><<<blocks, kBlockSize, 0, stream>>>(
      output, vanilla, optimistic, alpha, total);
  ReportCUDAErrors(cudaGetLastError());
}

template void BlendPolicyLogits<float>(int, float*, const float*,
                                        const float*, float, cudaStream_t);
template void BlendPolicyLogits<half>(int, half*, const half*, const half*,
                                       float, cudaStream_t);

// Dual-output variant: writes two pre-blended buffers in a single read
// pass over vanilla and optimistic.  Used to support split-alpha (root
// vs internal) on GPU — the search-time fast path then picks which
// pre-blended buffer to use per depth without any per-node blend math.
//
// out_v[i] = (1 - alpha_root)     · vanilla[i] + alpha_root     · optimistic[i]
// out_o[i] = (1 - alpha_internal) · vanilla[i] + alpha_internal · optimistic[i]
//
// Each thread reads vanilla[i] and optimistic[i] ONCE, computes both
// blends, writes both outputs.  Safe with full aliasing — `out_v` may
// alias `vanilla`, `out_o` may alias `optimistic`, both writes happen
// AFTER both reads in the same thread.
template <typename T>
__global__ void blendPolicyLogitsDual_kernel(
    T* out_v, T* out_o, const T* vanilla, const T* optimistic,
    float alpha_root, float alpha_internal, int total) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const float v = (float)vanilla[i];
  const float o = (float)optimistic[i];
  out_v[i] = (T)((1.0f - alpha_root) * v + alpha_root * o);
  out_o[i] = (T)((1.0f - alpha_internal) * v + alpha_internal * o);
}

template <typename T>
void BlendPolicyLogitsDual(int total, T* out_v, T* out_o, const T* vanilla,
                            const T* optimistic, float alpha_root,
                            float alpha_internal, cudaStream_t stream) {
  const int kBlockSize = 256;
  int blocks = DivUp(total, kBlockSize);
  blendPolicyLogitsDual_kernel<T><<<blocks, kBlockSize, 0, stream>>>(
      out_v, out_o, vanilla, optimistic, alpha_root, alpha_internal, total);
  ReportCUDAErrors(cudaGetLastError());
}

template void BlendPolicyLogitsDual<float>(int, float*, float*, const float*,
                                            const float*, float, float,
                                            cudaStream_t);
template void BlendPolicyLogitsDual<half>(int, half*, half*, const half*,
                                           const half*, float, float,
                                           cudaStream_t);

// ── Fused ExoFormer anchor add: Q += Qa, K += Ka, V += Va ────────────────
// Collapses 3 separate addVectors kernel launches into 1 for the common
// default-lambda ExoFormer path. Saves ~2 launches per encoder layer.
template <typename T>
__global__ void addExoAnchorsKernel(T* __restrict__ q, const T* __restrict__ aq,
                                     T* __restrict__ k, const T* __restrict__ ak,
                                     T* __restrict__ v, const T* __restrict__ av,
                                     int total) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  q[i] = (T)((float)q[i] + (float)aq[i]);
  k[i] = (T)((float)k[i] + (float)ak[i]);
  v[i] = (T)((float)v[i] + (float)av[i]);
}

template <typename T>
void AddExoAnchors(int total, T* q, const T* aq, T* k, const T* ak,
                   T* v, const T* av, cudaStream_t stream) {
  const int kBlockSize = 256;
  int blocks = DivUp(total, kBlockSize);
  addExoAnchorsKernel<T><<<blocks, kBlockSize, 0, stream>>>(
      q, aq, k, ak, v, av, total);
  ReportCUDAErrors(cudaGetLastError());
}

template void AddExoAnchors<half>(int, half*, const half*, half*, const half*,
                                   half*, const half*, cudaStream_t);
template void AddExoAnchors<float>(int, float*, const float*, float*, const float*,
                                    float*, const float*, cudaStream_t);

// Compute promotion logits in a single kernel
// keys matrix is of N * 64 * C (but we use only last 8 from the 'rows'
// dimension, so N * 8 * C)
// ppo matrix is 4 * C (weights for dense layer / matrix multiplication)
// policy_attn_logits matrix is N * 64 * 64, but we use only 8x8 part of it
// from each batch dimension (so, N * 8 * 8)
// output matrix (promotion logits) is of N * 8 * 24 size
template <typename T>
__global__ void promotion_logits_kernel(int C, T* output, const T* keys,
                                        const T* ppo,
                                        const T* policy_attn_logits) {
  constexpr int output_stride = 64 * 64 + 8 * 24;
  int n = blockIdx.x;   // [0..N)
  int y = threadIdx.y;  // [0..8)
  int x = threadIdx.x;  // [0..24)     // Can split into 8 * 3

  int threadInGroup = threadIdx.y * 24 + threadIdx.x;

  // phase 1 : compute promotion_offsets by multiplying keys and ppo matrices
  const T* keys_start =
      keys + n * 64 * C + C * 56;  // we are interested only in last 8 out of 64
                                   // 'rows' of keys matrix
  __shared__ float promotion_offsets[4][8];

  // only 32 threads out of 192 in the group are active in this phase, and each
  // thread computes one element of the promotion_offsets matrix
  // TODO: opt idea1, can use more threads to reduce the length of the loop for
  // the matrix multiply (do parallel reduction of partial sums later)
  //       opt idea2, the below loop for matrix mul has very poor memory access
  //       pattern, can do the loop over 32, and do parallel reductions
  if (threadInGroup < 32) {
    int x = threadInGroup % 4;
    int y = threadInGroup / 4;

    float S = 0;
    for (int i = 0; i < C;
         i++) {  // TODO: modify to loop over 32 instead of C (doing parallel
                 // reductions for the 32 sums)
      float a = (float)keys_start[y * C + i];
      float b =
          (float)ppo[x * C + i];  // weight matrix is transposed (col major)
      S += a * b;
    }

    // write the product (promotion_offsets) in shared memory
    promotion_offsets[x][y] = S;
  }

  __syncthreads();

  // phase 2: add the last "row" to the other 3
  // #knight offset is added to the other three
  // promotion_offsets = promotion_offsets[:, :3, :] + promotion_offsets[:, 3:4,
  // :]
  // Only 24 threads in the group are active in this phase
  if (threadInGroup < 32) {
    int x = threadInGroup % 4;
    int y = threadInGroup / 4;
    if (x < 3) {
      promotion_offsets[x][y] += promotion_offsets[3][y];
    }
  }

  __syncthreads();

  // phase 3: add 8x8 chunk of policy_attn_logits matrix to promotion offsets
  //          the output is 3x8x8 (written as 8 * 24)
  // All threads are active in this phase and they compute one element each
  int w = x / 3;
  int c = x % 3;

  // n_promo_logits = matmul_qk[:, -16:-8, -8:]  # default traversals from rank
  // 7 to rank 8
  float n_promo_logit =
      (float)policy_attn_logits[n * output_stride + (48 + y) * 64 + (56 + w)];
  float promo_offset = promotion_offsets[c][w];

  float op = n_promo_logit + promo_offset;

  output[n * output_stride + threadInGroup] = (T)op;
}

template <typename T>
void ComputePromotionLogits(int N, int C, T* output, const T* keys,
                            const T* ppo, const T* policy_attn_logits,
                            cudaStream_t stream) {
  // N blocks
  // 8 * 24 threads
  // Each thread computes a single output element
  dim3 blockDim(24, 8, 1);
  promotion_logits_kernel<T>
      <<<N, blockDim, 0, stream>>>(C, output, keys, ppo, policy_attn_logits);
}

template <typename T>
__global__ void preprocess_for_attention_body_kernel(
    T* output, const T* input, const T* encoding, int input_size,
    int encoding_size, bool is_pe_dense_embedding) {
  int n = blockIdx.x;
  int hw = blockIdx.y;
  int c = threadIdx.x;

  T op;
  if (c >= input_size) {
    // concatenate from position encoding array
    if (is_pe_dense_embedding) {
      op = (T)(encoding[n * 64 * encoding_size + hw * encoding_size +
                        (c - input_size)]);
    } else {
      op = (T)(encoding[64 * hw + (c - input_size)]);
    }
  } else {
    op = input[n * input_size * 64 + c * 64 + hw];  // nchw
  }

  int outputC = input_size + encoding_size;

  // convert to nhwc
  output[n * 64 * outputC + hw * outputC + c] = op;
}

template <typename T>
void inputPreprocessForAttentionBody(T* output, const T* input,
                                     const T* encoding, int N, int input_size,
                                     int encoding_size,
                                     bool is_pe_dense_embedding,
                                     cudaStream_t stream) {
  // N * 64 blocks
  // (kInputPlanes + kNumPosEncodingChannels) threads
  // Each thread computes a single output element
  dim3 gridSize = dim3(N, 64);
  int blockSize = input_size + encoding_size;
  preprocess_for_attention_body_kernel<T><<<gridSize, blockSize, 0, stream>>>(
      output, input, encoding, input_size, encoding_size,
      is_pe_dense_embedding);
}

// ── Material features: 18 derived chess features from 112 input planes ──
// Mirrors PyTorch _derive_chess_features:
//   [0..4]  : own piece counts (pawn, knight, bishop, rook, queen) normalized
//   [5..9]  : opp piece counts
//   [10..14]: material diff (own - opp) / max_counts
//   [15]    : own piece mask (1 if own piece on this square)
//   [16]    : opp piece mask (1 if opp piece on this square)
//   [17]    : total material diff ((sum own*values) - (sum opp*values)) / 39
// Input: (N, 112, 8, 8) NCHW. Output: (N, 64, 18) NHWC.
template <typename T>
__global__ void material_features_kernel(T* output, const T* input, int N) {
  // Each block handles one batch element; threads compute features for
  // different squares in parallel.
  int n = blockIdx.x;
  int sq = threadIdx.x;          // 0..63
  if (sq >= 64) return;

  // Piece counts are per-batch globals; use shared memory + block reduction.
  __shared__ float own_cnt[5];   // pawn, knight, bishop, rook, queen
  __shared__ float opp_cnt[5];

  // Step 1: each thread reads its own square's value for each piece plane and
  // atomicAdds to the shared counter. Only first thread initializes memory.
  if (sq == 0) {
    for (int i = 0; i < 5; i++) { own_cnt[i] = 0.0f; opp_cnt[i] = 0.0f; }
  }
  __syncthreads();

  // Each square contributes 1 to count if that piece is on this square.
  // Plane layout: channels 0..5 = own (pawn..king), 6..11 = opp (pawn..king).
  // We skip king (channel 5 / 11) since it's always 1.
  // Input is NCHW: input[n][c][h][w] at offset n*112*64 + c*64 + (h*8+w)
  const int base = n * 112 * 64;
  for (int p = 0; p < 5; p++) {
    float vo = (float)input[base + p * 64 + sq];
    float ve = (float)input[base + (6 + p) * 64 + sq];
    if (vo > 0.5f) atomicAdd(&own_cnt[p], 1.0f);
    if (ve > 0.5f) atomicAdd(&opp_cnt[p], 1.0f);
  }
  __syncthreads();

  // Step 2: compute the 18 output features for this square.
  const float max_counts[5] = {8.0f, 2.0f, 2.0f, 2.0f, 1.0f};
  const float piece_values[5] = {1.0f, 3.0f, 3.0f, 5.0f, 9.0f};

  // own_mask = any own piece on this square (channels 0..5)
  float own_mask = 0.0f, opp_mask = 0.0f;
  for (int p = 0; p < 6; p++) {
    if ((float)input[base + p * 64 + sq] > 0.5f) own_mask = 1.0f;
    if ((float)input[base + (6 + p) * 64 + sq] > 0.5f) opp_mask = 1.0f;
  }

  // Material totals (normalized by max 39 = 1+3+3+5+9+3+3+5+9 excluding king+pawn x8 = let's trust training)
  // PyTorch uses 39 as the denominator; we match exactly.
  float own_mat = 0.0f, opp_mat = 0.0f;
  for (int p = 0; p < 5; p++) {
    own_mat += own_cnt[p] * piece_values[p];
    opp_mat += opp_cnt[p] * piece_values[p];
  }
  float mat_total_diff = (own_mat - opp_mat) / 39.0f;

  // Write 18 features for (n, sq). NHWC output: offset = n*64*18 + sq*18 + f
  T* out = output + n * 64 * 18 + sq * 18;
  for (int p = 0; p < 5; p++) {
    out[p] = (T)(own_cnt[p] / max_counts[p]);                    // 0..4 own counts
    out[5 + p] = (T)(opp_cnt[p] / max_counts[p]);                // 5..9 opp counts
    out[10 + p] = (T)((own_cnt[p] - opp_cnt[p]) / max_counts[p]); // 10..14 diffs
  }
  out[15] = (T)own_mask;
  out[16] = (T)opp_mask;
  out[17] = (T)mat_total_diff;
}

template <typename T>
void computeMaterialFeatures(int N, T* output, const T* input,
                             cudaStream_t stream) {
  dim3 grid(N);
  dim3 block(64);  // one thread per square
  material_features_kernel<T><<<grid, block, 0, stream>>>(output, input, N);
  ReportCUDAErrors(cudaGetLastError());
}

template void computeMaterialFeatures<float>(int, float*, const float*,
                                              cudaStream_t);
template void computeMaterialFeatures<half>(int, half*, const half*,
                                             cudaStream_t);

// Extended preprocess: also concatenates material features after input planes.
// Output layout: [input(input_size), material(material_size), encoding(encoding_size)]
template <typename T>
__global__ void preprocess_with_material_kernel(
    T* output, const T* input, const T* material, const T* encoding,
    int input_size, int material_size, int encoding_size,
    bool is_pe_dense_embedding) {
  int n = blockIdx.x;
  int hw = blockIdx.y;
  int c = threadIdx.x;
  int total = input_size + material_size + encoding_size;
  if (c >= total) return;

  T op;
  if (c < input_size) {
    // NCHW input
    op = input[n * input_size * 64 + c * 64 + hw];
  } else if (c < input_size + material_size) {
    // NHWC material
    int mc = c - input_size;
    op = material[n * 64 * material_size + hw * material_size + mc];
  } else {
    // encoding (NHWC or broadcasted)
    int ec = c - input_size - material_size;
    if (is_pe_dense_embedding) {
      op = encoding[n * 64 * encoding_size + hw * encoding_size + ec];
    } else {
      op = encoding[64 * hw + ec];
    }
  }

  output[n * 64 * total + hw * total + c] = op;
}

template <typename T>
void inputPreprocessForAttentionBodyWithMaterial(
    T* output, const T* input, const T* material, const T* encoding,
    int N, int input_size, int material_size, int encoding_size,
    bool is_pe_dense_embedding, cudaStream_t stream) {
  dim3 gridSize(N, 64);
  int blockSize = input_size + material_size + encoding_size;
  preprocess_with_material_kernel<T><<<gridSize, blockSize, 0, stream>>>(
      output, input, material, encoding, input_size, material_size,
      encoding_size, is_pe_dense_embedding);
  ReportCUDAErrors(cudaGetLastError());
}

template void inputPreprocessForAttentionBodyWithMaterial<float>(
    float*, const float*, const float*, const float*, int, int, int, int,
    bool, cudaStream_t);
template void inputPreprocessForAttentionBodyWithMaterial<half>(
    half*, const half*, const half*, const half*, int, int, int, int, bool,
    cudaStream_t);

// ── Attack maps: 6 tactical features from piece planes with ray blocking ──
// Mirrors PyTorch _compute_attack_maps:
//   [0] own_attacked:     is this square attacked by any own piece (binary)
//   [1] opp_attacked:     is this square attacked by any opp piece (binary)
//   [2] own_attack_count: how many own pieces attack this square (/ 6.0)
//   [3] opp_attack_count: how many opp pieces attack this square (/ 6.0)
//   [4] own_defended:     own piece on this square AND attacked by own piece
//   [5] opp_hanging:      opp piece here AND attacked by own AND not defended
// Input: (N, 112, 8, 8) NCHW. Output: (N, 64, 6) NHWC.
template <typename T>
__global__ void attack_map_kernel(T* output, const T* input, int N) {
  int n = blockIdx.x;
  int sq = threadIdx.x;  // target square 0..63
  if (sq >= 64) return;

  const int base = n * 112 * 64;

  // Load piece planes 0-11 into shared memory (one value per plane per square)
  __shared__ float piece[12][64];
  for (int p = 0; p < 12; p++) {
    piece[p][sq] = (float)input[base + p * 64 + sq];
  }
  __syncthreads();

  // Build occupancy: 1 if any piece on this square
  __shared__ float occ[64];
  float o = 0.0f;
  for (int p = 0; p < 12; p++) o += piece[p][sq];
  occ[sq] = (o > 0.5f) ? 1.0f : 0.0f;
  __syncthreads();

  const int r = sq / 8, c = sq % 8;

  // ── Own piece attacks on square sq ──
  bool own_attacked = false;
  float own_cnt = 0.0f;

  // Own pawn: pawns at (r-1, c±1) attack this square (own pawns advance +row)
  {
    float pawn = 0.0f;
    if (r - 1 >= 0) {
      if (c - 1 >= 0 && piece[0][(r-1)*8+(c-1)] > 0.5f) pawn = 1.0f;
      if (c + 1 < 8  && piece[0][(r-1)*8+(c+1)] > 0.5f) pawn = 1.0f;
    }
    if (pawn > 0.0f) { own_attacked = true; own_cnt += pawn; }
  }

  // Own knight: check all 8 L-shaped jumps
  {
    const int kdr[8] = {-2,-2,-1,-1, 1, 1, 2, 2};
    const int kdc[8] = {-1, 1,-2, 2,-2, 2,-1, 1};
    float kn = 0.0f;
    for (int k = 0; k < 8; k++) {
      int kr = r + kdr[k], kc = c + kdc[k];
      if (kr >= 0 && kr < 8 && kc >= 0 && kc < 8 &&
          piece[1][kr*8+kc] > 0.5f) kn = 1.0f;
    }
    if (kn > 0.0f) { own_attacked = true; own_cnt += kn; }
  }

  // Own diagonal sliders (bishop ch2, queen ch4) along 4 diagonal rays
  {
    const int dr[4] = {-1,-1, 1, 1};
    const int dc[4] = {-1, 1,-1, 1};
    for (int d = 0; d < 4; d++) {
      int tr = r, tc = c;
      for (int s = 0; s < 7; s++) {
        tr += dr[d]; tc += dc[d];
        if (tr < 0 || tr >= 8 || tc < 0 || tc >= 8) break;
        int tsq = tr*8+tc;
        if (piece[2][tsq] > 0.5f || piece[4][tsq] > 0.5f) {
          own_attacked = true; own_cnt += 1.0f;
        }
        if (occ[tsq] > 0.5f) break;
      }
    }
  }

  // Own orthogonal sliders (rook ch3, queen ch4) along 4 orthogonal rays
  {
    const int dr[4] = {-1, 1, 0, 0};
    const int dc[4] = { 0, 0,-1, 1};
    for (int d = 0; d < 4; d++) {
      int tr = r, tc = c;
      for (int s = 0; s < 7; s++) {
        tr += dr[d]; tc += dc[d];
        if (tr < 0 || tr >= 8 || tc < 0 || tc >= 8) break;
        int tsq = tr*8+tc;
        if (piece[3][tsq] > 0.5f || piece[4][tsq] > 0.5f) {
          own_attacked = true; own_cnt += 1.0f;
        }
        if (occ[tsq] > 0.5f) break;
      }
    }
  }

  // Own king (ch5): 8 adjacent squares
  {
    float kg = 0.0f;
    for (int dr = -1; dr <= 1; dr++) for (int dc = -1; dc <= 1; dc++) {
      if (dr == 0 && dc == 0) continue;
      int kr = r+dr, kc = c+dc;
      if (kr >= 0 && kr < 8 && kc >= 0 && kc < 8 &&
          piece[5][kr*8+kc] > 0.5f) kg = 1.0f;
    }
    if (kg > 0.0f) { own_attacked = true; own_cnt += kg; }
  }

  // ── Opponent piece attacks on square sq ──
  bool opp_attacked = false;
  float opp_cnt = 0.0f;

  // Opp pawn: at (r+1, c±1) (opp pawns advance -row toward rank 1)
  {
    float pawn = 0.0f;
    if (r + 1 < 8) {
      if (c - 1 >= 0 && piece[6][(r+1)*8+(c-1)] > 0.5f) pawn = 1.0f;
      if (c + 1 < 8  && piece[6][(r+1)*8+(c+1)] > 0.5f) pawn = 1.0f;
    }
    if (pawn > 0.0f) { opp_attacked = true; opp_cnt += pawn; }
  }

  // Opp knight (ch7)
  {
    const int kdr[8] = {-2,-2,-1,-1, 1, 1, 2, 2};
    const int kdc[8] = {-1, 1,-2, 2,-2, 2,-1, 1};
    float kn = 0.0f;
    for (int k = 0; k < 8; k++) {
      int kr = r + kdr[k], kc = c + kdc[k];
      if (kr >= 0 && kr < 8 && kc >= 0 && kc < 8 &&
          piece[7][kr*8+kc] > 0.5f) kn = 1.0f;
    }
    if (kn > 0.0f) { opp_attacked = true; opp_cnt += kn; }
  }

  // Opp diagonal sliders (bishop ch8, queen ch10)
  {
    const int dr[4] = {-1,-1, 1, 1};
    const int dc[4] = {-1, 1,-1, 1};
    for (int d = 0; d < 4; d++) {
      int tr = r, tc = c;
      for (int s = 0; s < 7; s++) {
        tr += dr[d]; tc += dc[d];
        if (tr < 0 || tr >= 8 || tc < 0 || tc >= 8) break;
        int tsq = tr*8+tc;
        if (piece[8][tsq] > 0.5f || piece[10][tsq] > 0.5f) {
          opp_attacked = true; opp_cnt += 1.0f;
        }
        if (occ[tsq] > 0.5f) break;
      }
    }
  }

  // Opp orthogonal sliders (rook ch9, queen ch10)
  {
    const int dr[4] = {-1, 1, 0, 0};
    const int dc[4] = { 0, 0,-1, 1};
    for (int d = 0; d < 4; d++) {
      int tr = r, tc = c;
      for (int s = 0; s < 7; s++) {
        tr += dr[d]; tc += dc[d];
        if (tr < 0 || tr >= 8 || tc < 0 || tc >= 8) break;
        int tsq = tr*8+tc;
        if (piece[9][tsq] > 0.5f || piece[10][tsq] > 0.5f) {
          opp_attacked = true; opp_cnt += 1.0f;
        }
        if (occ[tsq] > 0.5f) break;
      }
    }
  }

  // Opp king (ch11)
  {
    float kg = 0.0f;
    for (int dr = -1; dr <= 1; dr++) for (int dc = -1; dc <= 1; dc++) {
      if (dr == 0 && dc == 0) continue;
      int kr = r+dr, kc = c+dc;
      if (kr >= 0 && kr < 8 && kc >= 0 && kc < 8 &&
          piece[11][kr*8+kc] > 0.5f) kg = 1.0f;
    }
    if (kg > 0.0f) { opp_attacked = true; opp_cnt += kg; }
  }

  // ── Compute own/opp occupancy of this square ──
  float own_mask = 0.0f, opp_mask = 0.0f;
  for (int p = 0; p < 6; p++) {
    if (piece[p  ][sq] > 0.5f) own_mask = 1.0f;
    if (piece[p+6][sq] > 0.5f) opp_mask = 1.0f;
  }

  float own_ab = own_attacked ? 1.0f : 0.0f;
  float opp_ab = opp_attacked ? 1.0f : 0.0f;

  // Write (N, 64, 6) NHWC
  T* out = output + n * 64 * 6 + sq * 6;
  out[0] = (T)own_ab;
  out[1] = (T)opp_ab;
  out[2] = (T)(own_cnt / 6.0f);
  out[3] = (T)(opp_cnt / 6.0f);
  out[4] = (T)(own_mask * own_ab);
  out[5] = (T)(opp_mask * own_ab * (1.0f - opp_ab));
}

template <typename T>
void computeAttackMaps(int N, T* output, const T* input, cudaStream_t stream) {
  attack_map_kernel<T><<<N, 64, 0, stream>>>(output, input, N);
  ReportCUDAErrors(cudaGetLastError());
}

template void computeAttackMaps<float>(int, float*, const float*, cudaStream_t);
template void computeAttackMaps<half>(int, half*, const half*, cudaStream_t);

// Extended preprocess for both material (18) and attack maps (6) active.
// Output layout: [input(input_size), extra1(extra1_size), extra2(extra2_size),
//                 encoding(encoding_size)] all in NHWC.
template <typename T>
__global__ void preprocess_with_two_extras_kernel(
    T* output, const T* input,
    const T* extra1, int extra1_size,
    const T* extra2, int extra2_size,
    const T* encoding, int input_size, int encoding_size,
    bool is_pe_dense_embedding) {
  int n = blockIdx.x;
  int hw = blockIdx.y;
  int c = threadIdx.x;
  int total = input_size + extra1_size + extra2_size + encoding_size;
  if (c >= total) return;

  T op;
  if (c < input_size) {
    op = input[n * input_size * 64 + c * 64 + hw];
  } else if (c < input_size + extra1_size) {
    int ec = c - input_size;
    op = extra1[n * 64 * extra1_size + hw * extra1_size + ec];
  } else if (c < input_size + extra1_size + extra2_size) {
    int ec = c - input_size - extra1_size;
    op = extra2[n * 64 * extra2_size + hw * extra2_size + ec];
  } else {
    int ec = c - input_size - extra1_size - extra2_size;
    if (is_pe_dense_embedding) {
      op = encoding[n * 64 * encoding_size + hw * encoding_size + ec];
    } else {
      op = encoding[64 * hw + ec];
    }
  }
  output[n * 64 * total + hw * total + c] = op;
}

template <typename T>
void inputPreprocessForAttentionBodyWithTwoExtras(
    T* output, const T* input,
    const T* extra1, int extra1_size,
    const T* extra2, int extra2_size,
    const T* encoding, int N, int input_size, int encoding_size,
    bool is_pe_dense_embedding, cudaStream_t stream) {
  dim3 gridSize(N, 64);
  int blockSize = input_size + extra1_size + extra2_size + encoding_size;
  preprocess_with_two_extras_kernel<T><<<gridSize, blockSize, 0, stream>>>(
      output, input, extra1, extra1_size, extra2, extra2_size,
      encoding, input_size, encoding_size, is_pe_dense_embedding);
  ReportCUDAErrors(cudaGetLastError());
}

template void inputPreprocessForAttentionBodyWithTwoExtras<float>(
    float*, const float*, const float*, int, const float*, int,
    const float*, int, int, int, bool, cudaStream_t);
template void inputPreprocessForAttentionBodyWithTwoExtras<half>(
    half*, const half*, const half*, int, const half*, int,
    const half*, int, int, int, bool, cudaStream_t);

// ── Rich embedding per-square MLP input builder ──
// Output layout (NHWC): [piece_12, attack_6?, material_mask_2?]
// `material_size` is the full material_info feature count (18 if enabled); we
// read indices 15 and 16 (own_mask, opp_mask) — the only per-square slots.
template <typename T>
__global__ void build_rich_per_sq_input_kernel(
    T* output, const T* input_nchw,
    const T* attack_nhwc, int attack_size,
    const T* material_nhwc, int material_size,
    int total_sz) {
  int n = blockIdx.x;
  int hw = blockIdx.y;
  int c = threadIdx.x;
  if (c >= total_sz) return;

  T op;
  if (c < 12) {
    // Piece plane: NCHW → scalar
    op = input_nchw[n * 112 * 64 + c * 64 + hw];
  } else if (c < 12 + attack_size) {
    int ec = c - 12;
    op = attack_nhwc[n * 64 * attack_size + hw * attack_size + ec];
  } else {
    // Material mask slice: indices 15 (own_mask) and 16 (opp_mask).
    int ec = c - 12 - attack_size;               // 0 or 1
    int mat_idx = 15 + ec;
    op = material_nhwc[n * 64 * material_size + hw * material_size + mat_idx];
  }
  output[n * 64 * total_sz + hw * total_sz + c] = op;
}

template <typename T>
void buildRichPerSquareInput(
    T* output, const T* input_nchw,
    const T* attack_nhwc, int attack_size,
    const T* material_nhwc, int material_size,
    int N, int total_sz, cudaStream_t stream) {
  dim3 grid(N, 64);
  build_rich_per_sq_input_kernel<T><<<grid, total_sz, 0, stream>>>(
      output, input_nchw, attack_nhwc, attack_size,
      material_nhwc, material_size, total_sz);
  ReportCUDAErrors(cudaGetLastError());
}

template void buildRichPerSquareInput<float>(
    float*, const float*, const float*, int, const float*, int,
    int, int, cudaStream_t);
template void buildRichPerSquareInput<half>(
    half*, const half*, const half*, int, const half*, int,
    int, int, cudaStream_t);

template <typename T>
__global__ void input_gating_kernel(T* output, const T* input, const T* mult,
                                    const T* add, int HW, int C) {
  int n_offset = blockIdx.z * HW * C;
  int idx = threadIdx.y * C + blockIdx.x * blockDim.x +
            threadIdx.x;  // index in input
  int idxT = (blockIdx.x * blockDim.x + threadIdx.x) * HW +
             threadIdx.y;  // index in transposed weights arrays mult and add.

  if (idx < HW * C) {
    // Combine multiply gating, add gating and weights transpose.
    float op =
        (float)input[n_offset + idx] * (float)mult[idxT] + (float)add[idxT];
    output[n_offset + idx] = (T)op;
  }
}

template <typename T>
void applyInputGating(T* output, const T* input, const T* mult, const T* add,
                      int N, int HW, int C, cudaStream_t stream) {
  // Multiple blocks to fit into each input area / volume
  // Block x position indicates horizontal section of area
  // Block y position indicates batch
  // Each thread computes a single output element
  dim3 blockSize, gridSize;
  blockSize.x = DivUp(1024, HW);
  blockSize.y = HW;
  blockSize.z = 1;
  gridSize.x = DivUp(C, blockSize.x);
  gridSize.y = 1;
  gridSize.z = N;
  input_gating_kernel<T>
      <<<gridSize, blockSize, 0, stream>>>(output, input, mult, add, HW, C);

  ReportCUDAErrors(cudaGetLastError());
}

template <typename T, int kWorkPerThread>
__global__ void genOffsetPointers_kernel(T** offsets, int heads, int block_size,
                                         int depth, int d_model, T* k, T* q,
                                         T* b1, T* v, T* b2) {
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) * kWorkPerThread;
  if (i >= block_size) return;
  const int h = i % heads;
  const int n = i / heads;
  int w;
  T* res[kWorkPerThread];
  for (w = 0; w < kWorkPerThread; w++) {
    res[w] = k + h * depth + 64 * d_model * n + w * depth;
    offsets[i + w] = res[w];
  }

  for (w = 0; w < kWorkPerThread; w++) {
    res[w] = q + h * depth + 64 * d_model * n + w * depth;
    offsets[i + w + block_size] = res[w];
  }

  for (w = 0; w < kWorkPerThread; w++) {
    res[w] = b1 + i * 64 * 64 + w * 64 * 64;
    offsets[i + w + 2 * block_size] = res[w];
  }

  for (w = 0; w < kWorkPerThread; w++) {
    res[w] = v + h * depth + 64 * d_model * n + w * depth;
    offsets[i + w + 3 * block_size] = res[w];
  }

  for (w = 0; w < kWorkPerThread; w++) {
    res[w] = b2 + h * depth + 64 * d_model * n + w * depth;
    offsets[i + w + 4 * block_size] = res[w];
  }
}

template <typename T>
void genOffsetPointers(T** offsets, int heads, int max_batch, int depth,
                       int d_model, T* k, T* q, T* b1, T* v, T* b2,
                       cudaStream_t stream) {
  const int block_size = heads * max_batch;
  // Process two elements per thread to use 128 bit store instructions.
  constexpr int kWorkPerThread = 2;
  constexpr int kWorkGroupSize = 128;
  if (block_size % kWorkPerThread != 0) {
    // Handle odd block sizes.
    int grid = DivUp(block_size, kWorkGroupSize);
    genOffsetPointers_kernel<T, 1><<<grid, kWorkGroupSize, 0, stream>>>(
        offsets, heads, block_size, depth, d_model, k, q, b1, v, b2);
  } else {
    // Handle even block size
    int grid = DivUp(block_size, kWorkGroupSize * kWorkPerThread);
    genOffsetPointers_kernel<T, kWorkPerThread>
        <<<grid, kWorkGroupSize, 0, stream>>>(offsets, heads, block_size, depth,
                                              d_model, k, q, b1, v, b2);
  }
}

// GQA variant of genOffsetPointers.  Each Q head h maps to a shared KV
// head h_kv = h * kv_heads / heads (= h / kv_groups when num_heads %
// kv_heads == 0).  K and V use stride kv_dim (smaller); Q and the
// outputs use the full d_model stride.  No physical KV expand — the
// offset pointers alias the kv_heads-wide K/V slice across the heads
// in a group.
template <typename T, int kWorkPerThread>
__global__ void genOffsetPointers_GQA_kernel(
    T** offsets, int heads, int kv_heads, int block_size, int depth,
    int d_model, int kv_dim, T* k, T* q, T* b1, T* v, T* b2) {
  const int i = (blockIdx.x * blockDim.x + threadIdx.x) * kWorkPerThread;
  if (i >= block_size) return;
  const int h = i % heads;
  const int n = i / heads;
  const int h_kv = h * kv_heads / heads;
  int w;
  T* res[kWorkPerThread];
  for (w = 0; w < kWorkPerThread; w++) {
    res[w] = k + h_kv * depth + 64 * kv_dim * n + w * depth;
    offsets[i + w] = res[w];
  }
  for (w = 0; w < kWorkPerThread; w++) {
    res[w] = q + h * depth + 64 * d_model * n + w * depth;
    offsets[i + w + block_size] = res[w];
  }
  for (w = 0; w < kWorkPerThread; w++) {
    res[w] = b1 + i * 64 * 64 + w * 64 * 64;
    offsets[i + w + 2 * block_size] = res[w];
  }
  for (w = 0; w < kWorkPerThread; w++) {
    res[w] = v + h_kv * depth + 64 * kv_dim * n + w * depth;
    offsets[i + w + 3 * block_size] = res[w];
  }
  for (w = 0; w < kWorkPerThread; w++) {
    res[w] = b2 + h * depth + 64 * d_model * n + w * depth;
    offsets[i + w + 4 * block_size] = res[w];
  }
}

template <typename T>
void genOffsetPointers_GQA(T** offsets, int heads, int kv_heads, int max_batch,
                           int depth, int d_model, int kv_dim, T* k, T* q,
                           T* b1, T* v, T* b2, cudaStream_t stream) {
  const int block_size = heads * max_batch;
  constexpr int kWorkPerThread = 2;
  constexpr int kWorkGroupSize = 128;
  if (block_size % kWorkPerThread != 0) {
    int grid = DivUp(block_size, kWorkGroupSize);
    genOffsetPointers_GQA_kernel<T, 1><<<grid, kWorkGroupSize, 0, stream>>>(
        offsets, heads, kv_heads, block_size, depth, d_model, kv_dim,
        k, q, b1, v, b2);
  } else {
    int grid = DivUp(block_size, kWorkGroupSize * kWorkPerThread);
    genOffsetPointers_GQA_kernel<T, kWorkPerThread>
        <<<grid, kWorkGroupSize, 0, stream>>>(
            offsets, heads, kv_heads, block_size, depth, d_model, kv_dim,
            k, q, b1, v, b2);
  }
}

// Physical weighted-GQA expand: each Q head's K/V is a learnable blend
// over the kv_heads dimension (vs plain GQA which copies via offset
// pointers).  Math (distributing scalars/biases across the blend sum):
//   K_out[h,s,d] = Σ_k W_k[h,k] * (K_in[k,s,d] + b_k[k,d])
//   V_out[h,s,d] = Σ_k W_v[h,k] * softcap(V_in[k,s,d] + b_v[k,d])
// where softcap(x) = v_softcap * tanh(x / v_softcap) when v_softcap > 0,
// identity otherwise.  Optional fusion:
//   b_k=nullptr     → skip K bias add
//   b_v=nullptr     → skip V bias (PGB) add
//   v_softcap<=0    → skip V softcap
// Bias is broadcast across (N, 64) — indexed by (kh * depth + d), not the
// full in_off, so a single (kv_dim,) buffer applies to every position.
// Input K/V are (N, kv_heads, 64, depth) col-major in the heads dim with
// stride kv_dim; output is (N, heads, 64, depth) with stride d_model.
//
// Smem layout: 2*kv_heads floats holding the (h-th) row of w_k and w_v.
// kv_heads ≤ 64 in practice (training-side limit); we statically size to
// 128 to leave headroom.  Each block uses ≤ 1KB smem — negligible.
// k_in_stride / v_in_stride parameters: by default each per-position
// column of K (or V) is kv_dim halfs (the standalone GQA case).  When
// K is embedded in the fused QKV output buffer the columns are wider
// (e.g. d_model + 3*kv_dim), and the caller passes a pointer already
// advanced to the K row offset within each column; k_in_stride is then
// the WIDER column stride.  V comes from a separate SwiGLU output and
// remains kv_dim-strided.
template <typename T>
__global__ void expandKVWeighted_kernel(
    T* k_out, T* v_out, const T* k_in, const T* v_in,
    const T* b_k, const T* b_v,
    const T* w_k, const T* w_v,
    const T* exo_k_anc, const T* exo_v_anc,
    int N, int heads, int kv_heads,
    int depth, int d_model, int kv_dim,
    int k_in_stride, int v_in_stride,
    float v_softcap) {
  // grid: (N * heads, 64), thread: depth.
  const int d = threadIdx.x;
  const int s = blockIdx.y;     // 0..63
  const int nh = blockIdx.x;
  const int n = nh / heads;
  const int h = nh % heads;

  // Cache this block's blend row in shared memory.  Threads d in
  // [0, kv_heads) cooperate — kv_heads ≤ depth always (training enforces
  // kv_heads | heads, depth = d_model/heads, kv_dim = kv_heads*depth).
  __shared__ float w_k_row[128];
  __shared__ float w_v_row[128];
  if (d < kv_heads) {
    w_k_row[d] = (float)w_k[h * kv_heads + d];
    w_v_row[d] = (float)w_v[h * kv_heads + d];
  }
  __syncthreads();

  if (d >= depth) return;
  const int out_off = h * depth + s * d_model + 64 * d_model * n + d;
  float ak = 0.f, av = 0.f;
  const float inv_cap = (v_softcap > 0.f) ? (1.0f / v_softcap) : 0.f;
  #pragma unroll 4
  for (int kh = 0; kh < kv_heads; kh++) {
    const int k_in_off = kh * depth + s * k_in_stride
                       + 64 * k_in_stride * n + d;
    const int v_in_off = kh * depth + s * v_in_stride
                       + 64 * v_in_stride * n + d;
    const int b_off    = kh * depth + d;

    // K: optional pre-blend bias add.
    float k_val = (float)k_in[k_in_off];
    if (b_k) k_val += (float)b_k[b_off];
    ak += w_k_row[kh] * k_val;

    // V: optional pre-blend PGB add, then optional softcap.
    float v_val = (float)v_in[v_in_off];
    if (b_v) v_val += (float)b_v[b_off];
    if (v_softcap > 0.f) v_val = v_softcap * tanhf(v_val * inv_cap);
    av += w_v_row[kh] * v_val;
  }
  // Optional ExoFormer anchor add — per-element on the d_model-wide
  // output.  When non-null this replaces the K/V portion of
  // AddExoAnchors (Q anchor still added separately).
  if (exo_k_anc) ak += (float)exo_k_anc[out_off];
  if (exo_v_anc) av += (float)exo_v_anc[out_off];
  k_out[out_off] = (T)ak;
  v_out[out_off] = (T)av;
}

template <typename T>
void expandKVWeighted(T* k_out, T* v_out, const T* k_in, const T* v_in,
                     const T* w_k, const T* w_v, int N, int heads,
                     int kv_heads, int depth, int d_model, int kv_dim,
                     cudaStream_t stream,
                     const T* b_k, const T* b_v, float v_softcap,
                     const T* exo_k_anc, const T* exo_v_anc,
                     int k_in_stride, int v_in_stride) {
  dim3 grid(N * heads, 64);
  dim3 block(depth);
  // Defaults: per-position column stride = kv_dim (standalone packing).
  const int ks = (k_in_stride > 0) ? k_in_stride : kv_dim;
  const int vs = (v_in_stride > 0) ? v_in_stride : kv_dim;
  expandKVWeighted_kernel<T><<<grid, block, 0, stream>>>(
      k_out, v_out, k_in, v_in, b_k, b_v, w_k, w_v,
      exo_k_anc, exo_v_anc,
      N, heads, kv_heads, depth, d_model, kv_dim, ks, vs, v_softcap);
}

// addBiasAndAdd: out[i] = in[i] + broadcast_bias[i % width] + add[i].
// Used to fuse the Q2 bias add with the Q exo anchor add: the bias is
// per-feature (d_model wide) broadcast across (N, 64); the add is the
// per-element exo anchor at the same total size as in/out.  Saves one
// kernel launch per encoder layer when GQA + ExoFormer are both active
// (the K/V exo is fused into expandKVWeighted, so this kernel
// completes the elimination of AddExoAnchors on that path).
template <typename T>
__global__ void addBiasAndAdd_kernel(T* out, const T* in, const T* bias,
                                      const T* add, int total, int width) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  float v = (float)in[i] + (float)bias[i % width] + (float)add[i];
  out[i] = (T)v;
}

template <typename T>
void addBiasAndAdd(T* out, const T* in, const T* bias, const T* add,
                   int total, int width, cudaStream_t stream) {
  const int kBlock = 256;
  int blocks = DivUp(total, kBlock);
  addBiasAndAdd_kernel<T><<<blocks, kBlock, 0, stream>>>(
      out, in, bias, add, total, width);
  ReportCUDAErrors(cudaGetLastError());
}

template void addBiasAndAdd<float>(float*, const float*, const float*,
                                    const float*, int, int, cudaStream_t);
template void addBiasAndAdd<half>(half*, const half*, const half*,
                                   const half*, int, int, cudaStream_t);

// addBiasSiluStrided: per-element silu(x + bias[c]) where x lives in a
// strided buffer.  Used for Q1 in the fused QKV path — Q1 occupies the
// first d_model rows of each column of the fused (1280, batch) output;
// stride is the wider column dim (= 1280), inner is d_model.
//
// Layout: for batch element b ∈ [0, batch) and channel c ∈ [0, inner):
//   offset = b * stride + c
// Bias is indexed by c only (broadcast across batch).
template <typename T>
__global__ void addBiasSiluStrided_kernel(T* out, const T* in, const T* bias,
                                           int batch, int inner, int stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int total = batch * inner;
  if (i >= total) return;
  const int b = i / inner;
  const int c = i % inner;
  const int off = b * stride + c;
  float v = (float)in[off] + (float)bias[c];
  // SiLU: v = v * sigmoid(v)
  v = v * (1.0f / (1.0f + __expf(-v)));
  out[off] = (T)v;
}

template <typename T>
void addBiasSiluStrided(T* out, const T* in, const T* bias,
                        int batch, int inner, int stride,
                        cudaStream_t stream) {
  const int kBlock = 256;
  int blocks = DivUp(batch * inner, kBlock);
  addBiasSiluStrided_kernel<T><<<blocks, kBlock, 0, stream>>>(
      out, in, bias, batch, inner, stride);
  ReportCUDAErrors(cudaGetLastError());
}

template void addBiasSiluStrided<float>(float*, const float*, const float*,
                                         int, int, int, cudaStream_t);
template void addBiasSiluStrided<half>(half*, const half*, const half*,
                                        int, int, int, cudaStream_t);

// Template instantiation.
template void copyTypeConverted<half, float>(half* op, float* ip, int N,
                                             cudaStream_t stream);
template void copyTypeConverted<float, half>(float* op, half* ip, int N,
                                             cudaStream_t stream);
template void copyTypeConverted<float, float>(float* op, float* ip, int N,
                                              cudaStream_t stream);
template void copyTypeConverted<half, half>(half* op, half* ip, int N,
                                            cudaStream_t stream);

template void batchNorm<float>(float* output, const float* input,
                               const float* skipInput, int N, int C, int H,
                               int W, float* means, float* var_multipliers,
                               ActivationFunction activation,
                               cudaStream_t stream);
template void batchNorm<half>(half* output, const half* input,
                              const half* skipInput, int N, int C, int H, int W,
                              float* means, float* var_multipliers,
                              ActivationFunction activation,
                              cudaStream_t stream);

template void addVectors<float>(float* c, float* a, float* b, int size,
                                int asize, int bsize, ActivationFunction act,
                                cudaStream_t stream);
template void addVectors<half>(half* c, half* a, half* b, int size, int asize,
                               int bsize, ActivationFunction act,
                               cudaStream_t stream);

template void addVectorsHNC_NHC<float>(float* a, float* b, int N, int H, int C,
                                       cudaStream_t stream);
template void addVectorsHNC_NHC<half>(half* a, half* b, int N, int H, int C,
                                      cudaStream_t stream);

template void addBiasBatched<float>(float* output, const float* input,
                                    const float* bias, int Batch, int N, int C,
                                    ActivationFunction activation,
                                    cudaStream_t stream);
template void addBiasBatched<half>(half* output, const half* input,
                                   const half* bias, int Batch, int N, int C,
                                   ActivationFunction activation,
                                   cudaStream_t stream);

template void addBiasAndSoftCap<float>(float* output, const float* input,
                                       const float* bias, int total, int C,
                                       float softcap, cudaStream_t stream);
template void addBiasAndSoftCap<half>(half* output, const half* input,
                                      const half* bias, int total, int C,
                                      float softcap, cudaStream_t stream);

template void addBiasBatched<float>(float* output, const float* input,
                                    const float* bias, int Batch, int N, int C,
                                    int Nstride, ActivationFunction activation,
                                    cudaStream_t stream);
template void addBiasBatched<half>(half* output, const half* input,
                                   const half* bias, int Batch, int N, int C,
                                   int Nstride, ActivationFunction activation,
                                   cudaStream_t stream);

template void addBias_NCHW<float>(float* c, float* a, float* b, int N, int C,
                                  int H, int W, ActivationFunction activation,
                                  cudaStream_t stream);

template void addBias_NCHW<half>(half* c, half* a, half* b, int N, int C, int H,
                                 int W, ActivationFunction activation,
                                 cudaStream_t stream);

template void globalAvgPool<float>(int N, int C, float* output,
                                   const float* input,
                                   const float* prevLayerBias, bool nhwc,
                                   cudaStream_t stream);
template void globalAvgPool<half>(int N, int C, half* output, const half* input,
                                  const half* prevLayerBias, bool nhwc,
                                  cudaStream_t stream);

template void expandPlanes_NHWC<float>(float* output, const uint64_t* masks,
                                       const float* values, int n,
                                       cudaStream_t stream);
template void expandPlanes_NHWC<half>(half* output, const uint64_t* masks,
                                      const half* values, int n,
                                      cudaStream_t stream);

template void expandPlanes_NCHW<float>(float* output, const uint64_t* masks,
                                       const float* values, int n,
                                       cudaStream_t stream);
template void expandPlanes_NCHW<half>(half* output, const uint64_t* masks,
                                      const half* values, int n,
                                      cudaStream_t stream);

template void globalScale<float>(int N, int C, float* output,
                                 const float* input, const float* scaleBias,
                                 const float* prevLayerBias, bool nhwc,
                                 ActivationFunction activation,
                                 cudaStream_t stream);
template void globalScale<half>(int N, int C, half* output, const half* input,
                                const half* scaleBias,
                                const half* prevLayerBias, bool nhwc,
                                ActivationFunction activation,
                                cudaStream_t stream);

template void PolicyMap<float>(int N, float* output, const float* input,
                               const short* indices, int inputSize,
                               int usedSize, int outputSize,
                               cudaStream_t stream);

template void PolicyMap<half>(int N, half* output, const half* input,
                              const short* indices, int inputSize, int usedSize,
                              int outputSize, cudaStream_t stream);

template void FilterTransform<float>(int N, int C, float* transformedFilter,
                                     const float* filter, cudaStream_t stream);

template void InputTransform<float, true>(int N, int C,
                                          float* transformed_input,
                                          const float* input,
                                          cudaStream_t stream);

template void InputTransform<float, false>(int N, int C,
                                           float* transformed_input,
                                           const float* input,
                                           cudaStream_t stream);

template void OutputTransform<float, true, ACTIVATION_RELU, true, true, false,
                              false>(int N, int C, int se_K, float* output,
                                     const float* input, const float* skip,
                                     const float* bias, const float* w1,
                                     const float* b1, const float* w2,
                                     const float* b2, cudaStream_t stream);

template void
OutputTransform<float, false, ACTIVATION_RELU, true, true, false, false>(

    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2, cudaStream_t stream);

template void OutputTransform<float, true, ACTIVATION_RELU, true, true, true,
                              false>(int N, int C, int se_K, float* output,
                                     const float* input, const float* skip,
                                     const float* bias, const float* w1,
                                     const float* b1, const float* w2,
                                     const float* b2, cudaStream_t stream);

template void OutputTransform<float, false, ACTIVATION_RELU, true, true, true,
                              false>(int N, int C, int se_K, float* output,
                                     const float* input, const float* skip,
                                     const float* bias, const float* w1,
                                     const float* b1, const float* w2,
                                     const float* b2, cudaStream_t stream);

template void OutputTransform<float, false, ACTIVATION_RELU, true, false, false,
                              false>(int N, int C, int se_K, float* output,
                                     const float* input, const float* skip,
                                     const float* bias, const float* w1,
                                     const float* b1, const float* w2,
                                     const float* b2, cudaStream_t stream);

template void OutputTransform<float, false, ACTIVATION_RELU, true, false, false,
                              true>(int N, int C, int se_K, float* output,
                                    const float* input, const float* skip,
                                    const float* bias, const float* w1,
                                    const float* b1, const float* w2,
                                    const float* b2, cudaStream_t stream);

template void OutputTransform<float, true, ACTIVATION_RELU, true, true, true,
                              true>(int N, int C, int se_K, float* output,
                                    const float* input, const float* skip,
                                    const float* bias, const float* w1,
                                    const float* b1, const float* w2,
                                    const float* b2, cudaStream_t stream);

template void OutputTransform<float, true, ACTIVATION_MISH, true, true, false,
                              false>(int N, int C, int se_K, float* output,
                                     const float* input, const float* skip,
                                     const float* bias, const float* w1,
                                     const float* b1, const float* w2,
                                     const float* b2, cudaStream_t stream);

template void OutputTransform<float, false, ACTIVATION_MISH, true, true, false,
                              false>(int N, int C, int se_K, float* output,
                                     const float* input, const float* skip,
                                     const float* bias, const float* w1,
                                     const float* b1, const float* w2,
                                     const float* b2, cudaStream_t stream);

template void OutputTransform<float, true, ACTIVATION_MISH, true, true, true,
                              false>(int N, int C, int se_K, float* output,
                                     const float* input, const float* skip,
                                     const float* bias, const float* w1,
                                     const float* b1, const float* w2,
                                     const float* b2, cudaStream_t stream);

template void OutputTransform<float, false, ACTIVATION_MISH, true, true, true,
                              false>(int N, int C, int se_K, float* output,
                                     const float* input, const float* skip,
                                     const float* bias, const float* w1,
                                     const float* b1, const float* w2,
                                     const float* b2, cudaStream_t stream);

template void OutputTransform<float, false, ACTIVATION_MISH, true, false, false,
                              false>(int N, int C, int se_K, float* output,
                                     const float* input, const float* skip,
                                     const float* bias, const float* w1,
                                     const float* b1, const float* w2,
                                     const float* b2, cudaStream_t stream);

template void OutputTransform<float, false, ACTIVATION_MISH, true, false, false,
                              true>(int N, int C, int se_K, float* output,
                                    const float* input, const float* skip,
                                    const float* bias, const float* w1,
                                    const float* b1, const float* w2,
                                    const float* b2, cudaStream_t stream);

template void OutputTransform<float, true, ACTIVATION_MISH, true, true, true,
                              true>(int N, int C, int se_K, float* output,
                                    const float* input, const float* skip,
                                    const float* bias, const float* w1,
                                    const float* b1, const float* w2,
                                    const float* b2, cudaStream_t stream);

template void OutputTransform<float, false, ACTIVATION_NONE, true, false, false,
                              false>(int N, int C, int se_K, float* output,
                                     const float* input, const float* skip,
                                     const float* bias, const float* w1,
                                     const float* b1, const float* w2,
                                     const float* b2, cudaStream_t stream);

template void OutputInputTransform<float, true, ACTIVATION_RELU, true, true>(
    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2, cudaStream_t stream);

template void OutputInputTransform<float, false, ACTIVATION_RELU, true, true>(
    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2, cudaStream_t stream);

template void OutputInputTransform<float, false, ACTIVATION_RELU, true, false>(
    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2, cudaStream_t stream);

template void OutputInputTransform<float, true, ACTIVATION_MISH, true, true>(
    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2, cudaStream_t stream);

template void OutputInputTransform<float, false, ACTIVATION_MISH, true, true>(
    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2, cudaStream_t stream);

template void OutputInputTransform<float, false, ACTIVATION_MISH, true, false>(
    int N, int C, int se_K, float* output, const float* input,
    const float* skip, const float* bias, const float* w1, const float* b1,
    const float* w2, const float* b2, cudaStream_t stream);

template void Softmax<half>(int N, int C, half* output, const half* input,
                            const half* input2, cudaStream_t stream,
                            float softcap, float smolgen_cap);
template void Softmax<float>(int N, int C, float* output, const float* input,
                             const float* input2, cudaStream_t stream,
                             float softcap, float smolgen_cap);

template void LayerNorm<half>(int N, int C, half* output, const half* input,
                              const half* bias, const half* skip,
                              const half* gammas, const half* betas, float ep,
                              float alpha, ActivationFunction act,
                              cudaStream_t stream, const half* input2,
                              const half* out_add);
template void LayerNorm<float>(int N, int C, float* output, const float* input,
                               const float* bias, const float* skip,
                               const float* gammas, const float* betas,
                               float ep, float alpha, ActivationFunction act,
                               cudaStream_t stream, const float* input2,
                               const float* out_add);

template void ComputePromotionLogits<half>(int N, int C, half* output,
                                           const half* keys, const half* ppo,
                                           const half* policy_attn_logits,
                                           cudaStream_t stream);
template void ComputePromotionLogits<float>(int N, int C, float* output,
                                            const float* keys, const float* ppo,
                                            const float* policy_attn_logits,
                                            cudaStream_t stream);

template void convertNCHWtoNHWC<half, float>(half* output_tensor,
                                             const float* input_tensor, int Nin,
                                             int Cin, int Nout, int Cout, int H,
                                             int W, cudaStream_t stream);
template void convertNCHWtoNHWC<float, float>(float* output_tensor,
                                              const float* input_tensor,
                                              int Nin, int Cin, int Nout,
                                              int Cout, int H, int W,
                                              cudaStream_t stream);
template void convertNCHWtoNHWC<half, half>(half* output_tensor,
                                            const half* input_tensor, int Nin,
                                            int Cin, int Nout, int Cout, int H,
                                            int W, cudaStream_t stream);

template void inputPreprocessForAttentionBody<half>(
    half* output, const half* input, const half* encoding, int N,
    int input_size, int encoding_size, bool is_pe_dense_embedding,
    cudaStream_t stream);

template void inputPreprocessForAttentionBody<float>(
    float* output, const float* input, const float* encoding, int N,
    int input_size, int encoding_size, bool is_pe_dense_embedding,
    cudaStream_t stream);

template void applyInputGating<half>(half* output, const half* input,
                                     const half* mult, const half* add, int N,
                                     int C, int output_size,
                                     cudaStream_t stream);

template void applyInputGating<float>(float* output, const float* input,
                                      const float* mult, const float* add,
                                      int N, int C, int output_size,
                                      cudaStream_t stream);

template void genOffsetPointers<float>(float** offsets, int heads,
                                       int max_batch, int depth, int d_model,
                                       float* k, float* q, float* b1, float* v,
                                       float* b2, cudaStream_t stream);
template void genOffsetPointers<half>(half** offsets, int heads, int max_batch,
                                      int depth, int d_model, half* k, half* q,
                                      half* b1, half* v, half* b2,
                                      cudaStream_t stream);

template void genOffsetPointers_GQA<float>(
    float** offsets, int heads, int kv_heads, int max_batch, int depth,
    int d_model, int kv_dim, float* k, float* q, float* b1, float* v,
    float* b2, cudaStream_t stream);
template void genOffsetPointers_GQA<half>(
    half** offsets, int heads, int kv_heads, int max_batch, int depth,
    int d_model, int kv_dim, half* k, half* q, half* b1, half* v,
    half* b2, cudaStream_t stream);

template void expandKVWeighted<float>(
    float* k_out, float* v_out, const float* k_in, const float* v_in,
    const float* w_k, const float* w_v, int N, int heads, int kv_heads,
    int depth, int d_model, int kv_dim, cudaStream_t stream,
    const float* b_k, const float* b_v, float v_softcap,
    const float* exo_k_anc, const float* exo_v_anc,
    int k_in_stride, int v_in_stride);
template void expandKVWeighted<half>(
    half* k_out, half* v_out, const half* k_in, const half* v_in,
    const half* w_k, const half* w_v, int N, int heads, int kv_heads,
    int depth, int d_model, int kv_dim, cudaStream_t stream,
    const half* b_k, const half* b_v, float v_softcap,
    const half* exo_k_anc, const half* exo_v_anc,
    int k_in_stride, int v_in_stride);

template void RMSNorm<half>(int N, int C, half* output, const half* input,
                            const half* bias, const half* skip,
                            const half* gammas, float ep, float alpha,
                            ActivationFunction act, cudaStream_t stream,
                            const half* input2);
template void RMSNorm<float>(int N, int C, float* output, const float* input,
                             const float* bias, const float* skip,
                             const float* gammas, float ep, float alpha,
                             ActivationFunction act, cudaStream_t stream,
                             const float* input2);

template void SwiGLUElementwise<half>(int total, half* output, const half* gate,
                                      const half* up, const half* gate_bias,
                                      const half* up_bias,
                                      cudaStream_t stream,
                                      float swiglu_softcap);
template void SwiGLUElementwise<float>(int total, float* output,
                                       const float* gate, const float* up,
                                       const float* gate_bias,
                                       const float* up_bias,
                                       cudaStream_t stream,
                                       float swiglu_softcap);

template void FusedVGAE<half>(int total, half* output, const half* gate,
                              const half* bias, int bias_size,
                              cudaStream_t stream);
template void FusedVGAE<float>(int total, float* output, const float* gate,
                               const float* bias, int bias_size,
                               cudaStream_t stream);

template void BankGatedMul<half>(int batch, int out_dim, int bank_dim,
                                 half* output, const half* h, const half* lrb,
                                 const half* up, const half* diag,
                                 const half* gate_b, const half* up_b,
                                 const half* pgb, float softcap,
                                 int up_stride, cudaStream_t stream);
template void BankGatedMul<float>(int batch, int out_dim, int bank_dim,
                                  float* output, const float* h,
                                  const float* lrb, const float* up,
                                  const float* diag, const float* gate_b,
                                  const float* up_b, const float* pgb,
                                  float softcap, int up_stride,
                                  cudaStream_t stream);

template void FusedVGAEBank<half>(int total, half* output, const half* h,
                                  int bank_dim, const half* lrb,
                                  const half* diag, const half* bias, int dim,
                                  cudaStream_t stream);
template void FusedVGAEBank<float>(int total, float* output, const float* h,
                                   int bank_dim, const float* lrb,
                                   const float* diag, const float* bias,
                                   int dim, cudaStream_t stream);

template void FusedResidualAddLN<half>(int tokens, int emb, half* ln_output,
                                       half* residual_output,
                                       const half* residual, const half* delta,
                                       const half* gamma, const half* beta,
                                       float eps, cudaStream_t stream);
template void FusedResidualAddLN<float>(int tokens, int emb, float* ln_output,
                                        float* residual_output,
                                        const float* residual, const float* delta,
                                        const float* gamma, const float* beta,
                                        float eps, cudaStream_t stream);

template void ElementwiseMultiply<half>(int total, half* output,
                                        const half* a, const half* b,
                                        cudaStream_t stream);
template void ElementwiseMultiply<float>(int total, float* output,
                                         const float* a, const float* b,
                                         cudaStream_t stream);

template void ConcatSquareAndGlobal<half>(int N, int sq_size, int global_size,
                                          half* output, const half* sq_features,
                                          const half* global_features,
                                          cudaStream_t stream);
template void ConcatSquareAndGlobal<float>(int N, int sq_size, int global_size,
                                           float* output, const float* sq_features,
                                           const float* global_features,
                                           cudaStream_t stream);

template void WeightedAdd<half>(int total, half* output, float alpha,
                                const half* a, float beta, const half* b,
                                cudaStream_t stream);
template void WeightedAdd<float>(int total, float* output, float alpha,
                                  const float* a, float beta, const float* b,
                                  cudaStream_t stream);

template void FusedAttention64<half>(int N, int num_heads, int depth,
                                     int d_model,
                                     half* output, const half* Q, const half* K,
                                     const half* V, const half* smolgen_bias,
                                     cudaStream_t stream);
template void FusedAttention64<float>(int N, int num_heads, int depth,
                                      int d_model,
                                      float* output, const float* Q,
                                      const float* K, const float* V,
                                      const float* smolgen_bias,
                                      cudaStream_t stream);

}  // namespace cudnn_backend
}  // namespace lczero
