/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2022 The LCZero Authors

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
#include <atomic>
#include <cassert>
#include <cstdlib>
#include <list>
#include <memory>
#include <mutex>
#include <sstream>
#include <type_traits>
#include <vector>

#include "cuda_common.h"
#include "inputs_outputs.h"
#include "kernels.h"
#include "layers.h"
#include "neural/factory.h"
#include "neural/network_legacy.h"
#include "neural/tables/attention_policy_map.h"
#include "weight_arena.h"
#include "neural/tables/policy_map.h"
#include "utils/exception.h"
#include "utils/fp16_utils.h"
#include "utils/trace.h"

#if CUDART_VERSION >= 11010
#define CUDA_GRAPH_SUPPORTS_EXTERNAL_EVENTS 1
#else
#define CUDA_GRAPH_SUPPORTS_EXTERNAL_EVENTS 0
#undef cudaEventWaitExternal
#undef cudaEventRecordExternal
#endif

namespace lczero {
using namespace cudnn_backend;

// Defined in layers.cc. Flipped true by AddInput when a real (non-warmup)
// input arrives; gates LC0_NAN_SCAN diagnostics so CUDA-graph warmup
// doesn't pollute the output. Declared here so AddInput can set it.
namespace cudnn_backend {
extern std::atomic<bool> lc0_nan_scan_armed;
}

template <typename DataType>
class CudaNetwork;

static size_t getMaxAttentionHeadSize(
    const MultiHeadWeights::PolicyHead& weights, int N) {
  const size_t embedding_op_size = weights.ip_pol_b.size();
  const size_t policy_d_model = weights.ip2_pol_b.size();
  assert(policy_d_model == weights.ip3_pol_b.size());

  size_t encoder_d_model = 0;
  size_t encoder_dff = 0;

  if (weights.pol_encoder.size() > 0) {
    encoder_d_model = weights.pol_encoder[0].mha.q_b.size();
    encoder_dff = weights.pol_encoder[0].ffn.dense1_b.size();

    assert(encoder_d_model == weights.pol_encoder[0].mha.k_b.size());
    // v_b is absent under GLU-V (the V projection splits into v_gate +
    // v_up, each with its own bias).  Only assert width when v_b is
    // actually present.
    if (weights.pol_encoder[0].mha.v_b.size() > 0) {
      assert(encoder_d_model == weights.pol_encoder[0].mha.v_b.size());
    } else {
      assert(weights.pol_encoder[0].mha.v_gate_w.size() > 0 &&
             "v_b empty but no GLU-V weights either — net is malformed");
    }
    assert(embedding_op_size == weights.pol_encoder[0].ffn.dense2_b.size());
  }

  const size_t encoder_heads = weights.pol_encoder_head_count;

  size_t size =
      N * 64 *
      std::max(std::max(embedding_op_size, encoder_dff), policy_d_model);

  // size of matmul_qk matrix = encoder_heads_ * Batch * 64 * 64
  const size_t matmul_qk_size = encoder_heads * N * 64 * 64;
  const size_t output_size = N * (64 * 64 + 8 * 24);
  size = std::max(size, std::max(matmul_qk_size, output_size));

  size_t qkv_size = N * 64 * encoder_d_model;
  // We store qkv in single allocation, and other intermediate tensors are
  // sometimes stored by splitting an allocation into two halves.
  // NLA/GLU Attention need a 4th temp buffer; VGA-E also writes to scratch.
  // Use 5× to be safe (Q + K + V + temp + VGA-E gate).
  size = std::max(2 * size, 5 * qkv_size);
  return size;
}

static size_t getMaxAttentionBodySize(const MultiHeadWeights& weights, int N) {
  const size_t embedding_op_size = weights.ip_emb_b.size();

  size_t encoder_d_model = 0;
  size_t encoder_dff = 0;

  if (weights.encoder.size() > 0) {
    encoder_d_model = weights.encoder[0].mha.q_b.size();
    if (encoder_d_model == 0) {
      // Layout-1 / bank nets: q_w/q_b are ABSENT (Q = Wq2(h)); the Q width
      // comes from the second projection's bias instead — the exact same
      // fallback EncoderBlock's ctor uses for mha_q_size_.  Without this,
      // encoder_d_model = 0 collapses the qkv reservation below and the
      // encoder carves its Q/K/V/temp slabs PAST the end of scratch:
      // out-of-bounds writes that surface (or not) depending on the
      // machine's allocator layout.  Found the hard way on a Linux rig
      // (sanitizer-verified OOB at scratch + 3 slabs in a 2-slab buffer).
      encoder_d_model = weights.encoder[0].mha.q2_b.size();
    }

    // SwiGLU nets use gate_proj instead of dense1 — pick whichever is present.
    // Derive dff from weight matrix size (bias may be absent if bias=False).
    encoder_dff = weights.encoder[0].ffn.dense1_b.size();
    if (encoder_dff == 0) {
      encoder_dff = weights.encoder[0].ffn.gate_proj_b.size();
    }
    if (encoder_dff == 0 && weights.encoder[0].ffn.gate_proj_w.size() > 0) {
      encoder_dff = weights.encoder[0].ffn.gate_proj_w.size() /
                    embedding_op_size;
    }

    // K-on-bank nets carry k2_b instead of k_b; bank-form Q carries q2_b.
    assert(weights.encoder[0].mha.k_b.size() > 0 ||
           weights.encoder[0].mha.k2_b.size() > 0);
    assert(encoder_d_model > 0);
    // v_b is absent under GLU-V (split into v_gate_b + v_up_b).  Only
    // require v_b OR v_gate_w to be present, not both.  Under the shared
    // gate bank the gate weight is absent too — v_up_w marks GLU-V then.
    assert(weights.encoder[0].mha.v_b.size() > 0 ||
           weights.encoder[0].mha.v_gate_w.size() > 0 ||
           weights.encoder[0].mha.v_up_w.size() > 0);
  }

  const size_t encoder_heads = weights.encoder_head_count;

  size_t size =
      N * 64 *
      std::max(std::max(embedding_op_size, encoder_dff), encoder_d_model);

  // size of matmul_qk matrix = encoder_heads_ * Batch * 64 * 64
  const size_t matmul_qk_size = encoder_heads * N * 64 * 64;
  const size_t output_size = N * (64 * 64 + 8 * 24);
  size = std::max(size, std::max(matmul_qk_size, output_size));

  size_t qkv_size = N * 64 * encoder_d_model;
  // NLA fused Q1+K1 path needs a 2*d_model temp (Q+K interleaved) beyond
  // the Q/K/V slots: total = Q(1) + K(1) + V(1) + nla_qk_temp(2) = 5 * qkv.
  // The old separate path needed only 4 * qkv (single d_model temp).
  // GLU-V (no NLA) needs Q/K/V + glu_temp = 4 * qkv.
  // GQA in the non-NLA branch additionally carves transient k_kv2/v_kv2/
  // gu_kv2 (4 * kv_dim ≤ 4 * d_model) past Q/K/V, requiring 7 * qkv worst
  // case; with NLA active those transients live inside the 2*d_model
  // nla_qk_temp region so the 5*qkv reservation already covers GQA.
  //
  // Fused-QKV path (mha_qkv_fused_w_, built when NLA-Q-only + GQA + GLU-V
  // are all active) replaces 3 GEMMs with one, but the wider output
  // (d_model + 3*kv_dim = up to 2.5*d_model) plus v_kv (= 0.5*d_model)
  // = 3*d_model of intermediate beyond Q/K/V — bumps to 6*qkv when has_nla.
  // GLU-Q/K bank nets (q2_w absent → !has_nla) also build the fused
  // weight ([Wq|Wk|Wv_up], ≤ 2*d_model) and stage it plus v_kv_fused
  // (0.5*d_model) in the GQA transient region — 5.5*qkv total, inside
  // the 7*qkv non-NLA + GQA reservation below.
  bool has_nla = weights.encoder.size() > 0 &&
                 weights.encoder[0].mha.q2_w.size() > 0;
  bool has_glu_v = weights.encoder.size() > 0 &&
                   weights.encoder[0].mha.v_gate_w.size() > 0;
  bool has_gqa = weights.kv_headcount > 0 &&
                 weights.kv_headcount < (int)weights.encoder_head_count;
  size_t total_qkv;
  if (has_nla) {
    total_qkv = 6 * qkv_size;  // 5*qkv (NLA) + 1*qkv headroom for fused-QKV
  } else if (has_gqa) {
    total_qkv = 7 * qkv_size;  // Q/K/V + 4*kv_dim transients (kv_dim ≤ d_model)
  } else if (has_glu_v) {
    total_qkv = 4 * qkv_size;
  } else {
    total_qkv = 3 * qkv_size;
  }
  // We store qkv in single allocation, and other intermediate tensors are
  // sometimes stored by splitting an allocation into two halves.
  size = std::max(2 * size, total_qkv);
  return size;
}

template <typename DataType>
class CudaNetworkComputation : public NetworkComputation {
 public:
  CudaNetworkComputation(CudaNetwork<DataType>* network, bool wdl,
                         bool moves_left);
  ~CudaNetworkComputation();

  void AddInput(InputPlanes&& input) override {
    const auto iter_mask =
        &inputs_outputs_->input_masks_mem_[batch_size_ * kInputPlanes];
    const auto iter_val =
        &inputs_outputs_->input_val_mem_[batch_size_ * kInputPlanes];

    // Diagnostic: LC0_DUMP_INPUT_PLANES=1 prints each of the 112 input
    // planes for the first AddInput call. One line per plane:
    //   P<idx> nbits=<count> value=<val> mask=<hex>
    // Use to diff against pytorch_infer_batch.py's fen_to_planes_112 dump
    // on the same FEN — divergence means the lc0 encoder and the
    // PyTorch-side input builder disagree, explaining WDL drift on
    // non-trivial positions.
    static const bool kDumpInput =
        std::getenv("LC0_DUMP_INPUT_PLANES") != nullptr;
    // Detect real input for both the plane dump and to arm the NaN scanner.
    bool is_real = false;
    for (int i = 0; i < 12; ++i) {
      if (input[i].mask != 0) { is_real = true; break; }
    }
    if (is_real) {
      // layers.cc's LC0_NAN_SCAN gate reads this; suppresses warmup noise.
      cudnn_backend::lc0_nan_scan_armed.store(
          true, std::memory_order_relaxed);
    }
    if (kDumpInput) {
      // Fire on every REAL AddInput so multiple positions can be compared
      // in one process; each dump is tagged with a counter.
      if (is_real) {
        static std::atomic<int> dump_counter{0};
        const int n = dump_counter.fetch_add(1, std::memory_order_relaxed);
        fprintf(stderr, "[LC0_INPUT_PLANES] real AddInput #%d, 112 planes:\n",
                n);
        for (int i = 0; i < kInputPlanes; ++i) {
          const auto& plane = input[i];
          const uint64_t mask = plane.mask;
          int nbits = 0;
          for (int b = 0; b < 64; ++b) if (mask & (1ULL << b)) ++nbits;
          fprintf(stderr,
                  "  P%03d nbits=%2d value=%8.4f mask=0x%016llx\n",
                  i, nbits, (double)plane.value,
                  (unsigned long long)mask);
        }
        fflush(stderr);
      }
    }

    assert(input.size() == kInputPlanes);
    for (int i = 0; i < kInputPlanes; i++) {
      const auto& plane = input[i];
      iter_mask[i] = plane.mask;
      ToType(iter_val[i], plane.value);
    }

    batch_size_++;
  }

  void ComputeBlocking() override;

  void CaptureGraph(std::unique_lock<std::mutex>&& lock = {});

  int GetBatchSize() const override { return batch_size_; }

  float GetQVal(int sample) const override {
    if (wdl_) {
      const float* wdl =
          sizeof(inputs_outputs_->op_value_mem_[0]) == sizeof(float)
              ? (float*)inputs_outputs_->op_value_mem_
              : inputs_outputs_->wdl_cpu_softmax_.get();
      return wdl[2 * sample];
    }
    return FromType(inputs_outputs_->op_value_mem_[sample]);
  }

  float GetDVal(int sample) const override {
    if (wdl_) {
      const float* wdl =
          sizeof(inputs_outputs_->op_value_mem_[0]) == sizeof(float)
              ? (float*)inputs_outputs_->op_value_mem_
              : inputs_outputs_->wdl_cpu_softmax_.get();
      return wdl[2 * sample + 1];
    }
    return 0.0f;
  }

  float GetPVal(int sample, int move_id) const override {
    return FromType(
        inputs_outputs_->op_policy_mem_[sample * kNumOutputPolicy + move_id]);
  }

  // Optimistic policy head: only valid when the backend was constructed
  // with the optimistic head enabled.  op_policy_opt_mem_ is nullptr
  // otherwise; HasOptimisticPolicy() (below) tells the search layer to
  // skip blending in that case.
  float GetPValOptimistic(int sample, int move_id) const override {
    if (inputs_outputs_->op_policy_opt_mem_ == nullptr) {
      return GetPVal(sample, move_id);
    }
    return FromType(inputs_outputs_->op_policy_opt_mem_[
        sample * kNumOutputPolicy + move_id]);
  }

  bool HasOptimisticPolicy() const override {
    return inputs_outputs_->op_policy_opt_mem_ != nullptr;
  }

  float GetMVal(int sample) const override {
    if (moves_left_) {
      return FromType(inputs_outputs_->op_moves_left_mem_[sample]);
    }
    return 0.0f;
  }

 private:
  // Memory holding inputs, outputs.
  std::unique_ptr<InputsOutputs<DataType>> inputs_outputs_;
  int batch_size_;
  bool wdl_;
  bool moves_left_;

  CudaNetwork<DataType>* network_;
};

template <typename DataType>
class CudaNetwork : public Network {
 public:
  CudaNetwork(const WeightsFile& file, const OptionsDict& options)
      : capabilities_{file.format().network_format().input(),
                      file.format().network_format().output(),
                      file.format().network_format().moves_left()} {
    MultiHeadWeights weights(file.weights());

    // Set format flags from NetworkFormat (not available to BaseWeights ctor).
    {
      const auto& nf_flags = file.format().network_format();
      weights.is_prenorm = nf_flags.encoder_norm_style() ==
          pblczero::NetworkFormat::ENCODER_NORM_PRENORM;
      weights.use_rms_norm = nf_flags.use_rms_norm();
      weights.use_swiglu_ffn = nf_flags.use_swiglu_ffn();
      weights.use_parallel_ffn = nf_flags.use_parallel_ffn();
      weights.use_material_info = nf_flags.use_material_info();
      weights.use_attack_maps = nf_flags.use_attack_maps();
      // Soft-cap values: non-zero value = enabled. The legacy
      // `use_attn_logit_softcap` boolean (proto field 19) is honored ONLY
      // for backward compat — if it's set true on an old net but the value
      // field is absent, fall back to Gemma 2 default 50.0. New nets export
      // only the value; the value > 0 is the gate.
      weights.attn_logit_cap = nf_flags.attn_logit_cap();
      if (weights.attn_logit_cap == 0.0f &&
          nf_flags.use_attn_logit_softcap()) {
        weights.attn_logit_cap = 50.0f;  // legacy default
      }
      weights.smolgen_softcap = nf_flags.smolgen_softcap();
      weights.v_softcap = nf_flags.v_softcap();
      weights.swiglu_softcap = nf_flags.swiglu_softcap();
      // Multi-exit value branch layer. Proto field absent → default -1
      // (disabled). Value 0 is technically valid (branch at layer 0)
      // but not meaningful; values >= encoder count would be no-ops and
      // are handled by AttentionBody.
      weights.value_branch_at_layer = nf_flags.has_value_branch_at_layer()
                                          ? nf_flags.value_branch_at_layer()
                                          : -1;

      // Enriched-feature enablement is driven strictly by the NetworkFormat
      // flags above. A pb.gz without those flags set is treated as having no
      // enriched features, regardless of the rich MLP weight shape. If the
      // training code forgot to set the flags, regenerate the pb.gz — we no
      // longer auto-detect / override here.
    }

    gpu_id_ = options.GetOrDefault<int>("gpu", 0);
    enable_graph_capture_ = options.GetOrDefault<bool>("graph_capture", true);
    // GPU-side optimistic policy blend.  Two backend options:
    //   gpu_blend_alpha       — internal-node alpha (also default for root)
    //   gpu_blend_alpha_root  — optional override for root nodes
    //
    // Uniform mode (gpu_blend_alpha > 0, gpu_blend_alpha_root unset or
    // == gpu_blend_alpha):
    //   - GPU blends both buffers to the same alpha → only the vanilla
    //     buffer is needed on host (Speedup A: hides optimistic head
    //     from CPU pipeline, single softmax in wrapper).
    //   - User sets --optimistic-policy-weight=0 and
    //     --optimistic-policy-weight-internal=0.
    //
    // Split mode (gpu_blend_alpha > 0 AND gpu_blend_alpha_root > 0
    // AND they differ):
    //   - Dual-output kernel produces vanilla buffer at α_root and
    //     optimistic buffer at α_internal in one read pass.
    //   - Both buffers memcpy'd to host so search can pick which to
    //     use per depth.  Wrapper softmaxes both (~5% CPU overhead vs
    //     uniform mode, but still no per-node host blend math).
    //   - User sets --optimistic-policy-weight=0 (root uses
    //     pre-blended vanilla buffer directly via SetP from p) and
    //     --optimistic-policy-weight-internal=1.0 (internal fast
    //     path: SetP from p_optimistic).
    //
    // Both default 0 (disabled).  Only honored when the net has the
    // optimistic head built (has_optimistic_policy_).
    gpu_blend_alpha_ = options.GetOrDefault<float>("gpu_blend_alpha", 0.0f);
    gpu_blend_alpha_root_ =
        options.GetOrDefault<float>("gpu_blend_alpha_root", 0.0f);
    // Note: parallel FFN (multi-stream FFN) is compatible with graph capture
    // via the dual-graph approach: the main graph uses External event flags
    // to signal/wait on the FFN stream without causing isolation errors.
    // A separate FFN graph is captured and launched alongside the main graph.

    const auto nf = file.format().network_format();
    using NF = pblczero::NetworkFormat;
    conv_policy_ = nf.policy() == NF::POLICY_CONVOLUTION;
    attn_policy_ = nf.policy() == NF::POLICY_ATTENTION;
    attn_body_ = nf.network() == NF::NETWORK_ATTENTIONBODY_WITH_HEADFORMAT ||
                 nf.network() == NF::NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT;

    max_batch_size_ = options.GetOrDefault<int>("max_batch", 1024);
    // min_batch_size_ is chosen as 4 as it is common that for sizes less than
    // 4 that there is no performance gain, but there is variance in the
    // outputs, which means that there is extra non-determinism in some
    // scenarios, including using the multiplexing backend.
    min_batch_size_ =
        options.GetOrDefault<int>("min_batch", std::min(4, max_batch_size_));
    if (max_batch_size_ < min_batch_size_)
      throw Exception("Max batch must not be less than min_batch setting.");

    showInfo();

#ifdef USE_CUTLASS
    CERR << "Compiled with CUTLASS enabled";
#endif

    int total_gpus;
    ReportCUDAErrors(cudaGetDeviceCount(&total_gpus));

    if (gpu_id_ >= total_gpus)
      throw Exception("Invalid GPU Id: " + std::to_string(gpu_id_));

    cudaDeviceProp deviceProp = {};
    cudaGetDeviceProperties(&deviceProp, gpu_id_);
    showDeviceInfo(deviceProp, gpu_id_);

    l2_cache_size_ = deviceProp.l2CacheSize;
    sm_count_ = deviceProp.multiProcessorCount;

    allow_cache_opt_ = options.GetOrDefault<bool>("cache_opt", false);

    // Select GPU to run on (for *the current* thread).
    ReportCUDAErrors(cudaSetDevice(gpu_id_));

    multi_stream_ = options.GetOrDefault<bool>("multi_stream", false);

    // layout used by cuda backend is nchw.
    has_tensor_cores_ = false;
    constexpr bool fp16 = std::is_same<half, DataType>::value;

    if (fp16) {
      // Check if the GPU support FP16.

      if ((deviceProp.major == 6 && deviceProp.minor != 1) ||
          (deviceProp.major == 5 && deviceProp.minor == 3)) {
        // FP16 without tensor cores supported on GP100 (SM 6.0) and Jetson
        // (SM 5.3 and 6.2). SM 6.1 GPUs also have FP16, but slower than FP32.
        ;
      } else if (deviceProp.major >= 7) {
        // Some GPUs (GTX 16xx) are SM 7.5 but don't have tensor cores
        // enabling TENSOR_OP_MATH for them works but is very very slow
        // (likely because the system emulates it).
        if (!strstr(deviceProp.name, "GTX 16")) {
          has_tensor_cores_ = true;
        }
      } else {
        throw Exception("Your GPU doesn't support FP16");
      }
    }

    if (!multi_stream_) {
      ReportCUDAErrors(
          cudaStreamCreateWithFlags(&compute_stream_, cudaStreamNonBlocking));
      ReportCUDAErrors(
          cudaStreamCreateWithFlags(&upload_stream_, cudaStreamNonBlocking));
      ReportCUDAErrors(
          cudaStreamCreateWithFlags(&download_stream_, cudaStreamNonBlocking));
      ReportCUDAErrors(cudaEventCreateWithFlags(&compute_ordering_event_,
                                                cudaEventDisableTiming));
      ReportCUDAErrors(cudaEventCreateWithFlags(&attn_body_done_event_,
                                                cudaEventDisableTiming));
      ReportCUBLASErrors(cublasCreate(&cublas_));
      ReportCUBLASErrors(cublasSetStream(cublas_, compute_stream_));
      if (has_tensor_cores_)
        ReportCUBLASErrors(cublasSetMathMode(
            cublas_,
            CUBLAS_TENSOR_OP_MATH));  // Deprecated on CUDA 11.0 and later
      else if (fp16)
        ReportCUBLASErrors(cublasSetMathMode(
            cublas_,
            CUBLAS_PEDANTIC_MATH));  // Explicitly set PEDANTIC_MATH mode to
                                     // avoid cublas bug of making use of tensor
                                     // core math on TU11x GPUs that don't
                                     // support it.
      // Dedicated value-head stream.  It receives exactly 1 cross-stream wait
      // per inference (attn_body_done_event_), so the Ada Lovelace + CUDA 12.9
      // fork-join taint that accumulates on compute_stream_ (N events from
      // ffn_stream_ + 1 from upload_stream_) and on ffn_stream_ (N events from
      // compute_stream_) never builds up here.  PEDANTIC_MATH unconditionally
      // prevents tensor-core selection so m=3 WDL GEMMs always find a kernel.
      ReportCUDAErrors(
          cudaStreamCreateWithFlags(&value_stream_, cudaStreamNonBlocking));
      ReportCUBLASErrors(cublasCreate(&value_cublas_));
      ReportCUBLASErrors(cublasSetStream(value_cublas_, value_stream_));
      if (fp16)
        ReportCUBLASErrors(
            cublasSetMathMode(value_cublas_, CUBLAS_PEDANTIC_MATH));
    }

    const int kNumInputPlanes = kInputPlanes;
    const int kNumFilters = (int)weights.input.biases.size();
    numBlocks_ = (int)weights.residual.size();
    numFilters_ = kNumFilters;

    num_encoder_blocks_ = (int)weights.encoder.size();
    if (attn_body_) {
      assert(weights.ip_emb_b.size() > 0);
    }

    // Warn if the memory required for storing transformed weights is
    // going to exceed 40% of total video memory, force custom_winograd off
    // if it's going to exceed 50% of memory.
    size_t residual_single_layer_weight_size =
        3 * 3 * kNumFilters * kNumFilters * sizeof(DataType);
    size_t residual_weight_size =
        residual_single_layer_weight_size * numBlocks_ * 2;
    size_t transformed_residual_weight_size = residual_weight_size * 4;

    if (transformed_residual_weight_size > 0.4 * deviceProp.totalGlobalMem) {
      CERR << "WARNING: Low GPU video memory. You may run into OOM errors. Try "
              "using a smaller network.";
    }

    // Disable res block fusing for fp32 for now (not worth it)
    // TODO: make it work for filters not a multiple of 32.
    // Note that when used with SE, the optimization
    // works only when filter count is <= 384 (pre-Ampere), or less than 512
    // (Ampere)
    // It turns dynamically off based on filter count (see
    // ResidualBlock<DataType>::Eval)
    if (kNumFilters % 32 == 0 && std::is_same<half, DataType>::value) {
      use_res_block_winograd_fuse_opt_ = true;
    } else {
      use_res_block_winograd_fuse_opt_ = false;
    }
    // Override if set in backend-opts.
    if (options.Exists<bool>("res_block_fusing")) {
      use_res_block_winograd_fuse_opt_ = options.Get<bool>("res_block_fusing");
    }

    bool use_fused_mha = false;
    if (deviceProp.major >= 8 && fp16) {
      use_fused_mha = options.GetOrDefault<bool>("fused_mha", true);
    }

    const bool use_gemm_ex = deviceProp.major >= 5;

    // 0. Check for SE.
    has_se_ = false;
    if (numBlocks_ && weights.residual[0].has_se) {
      has_se_ = true;
    }

    // Have some minumum as we also use this for transforming weights.
    size_t max_weight_size = 128 * 1024 * 1024;

    // parts from scratch allocation are suballocated to hold various weights
    // and biases when transforming winograd weights (one layer at a time), 128
    // MB is way more than that what we need but make sure it's at least 3x of
    // single layer's weight size to be safe.
    if (max_weight_size < 3 * residual_single_layer_weight_size)
      max_weight_size = 3 * residual_single_layer_weight_size;

    scratch_size_ = max_weight_size;

    // Need additional space for transformed input/outputs which are 36/16
    // times size (4x4 block transformed into 6x6).
    if (numBlocks_ > 0) {
      const size_t transformed_tensor_size =
          (size_t)(max_batch_size_ * kNumFilters * 64 * (36.0 / 16.0) *
                   sizeof(DataType));
      scratch_size_ = std::max(scratch_size_, 2 * transformed_tensor_size);
    }

    std::string policy_head =
        options.GetOrDefault<std::string>("policy_head", "vanilla");
    // Check that selected policy head exists.
    if (!weights.policy_heads.contains(policy_head)) {
      throw Exception("The policy head you specified '" + policy_head +
                      "' does not exist in this net.");
    }
    std::string value_head =
        options.GetOrDefault<std::string>("value_head", "winner");
    // Check that selected value head exists.
    if (!weights.value_heads.contains(value_head)) {
      throw Exception("The value head you specified '" + value_head +
                      "' does not exist in this net.");
    }

    // Attention policy head or body may need more memory
    const size_t attentionPolicySize =
        getMaxAttentionHeadSize(weights.policy_heads.at(policy_head),
                                max_batch_size_) *
        sizeof(DataType);

    const size_t attentionBodySize =
        getMaxAttentionBodySize(weights, max_batch_size_) * sizeof(DataType);
    scratch_size_ = std::max(scratch_size_,
                             std::max(attentionPolicySize, attentionBodySize));

    ReportCUDAErrors(cudaMalloc(&scratch_mem_, scratch_size_));

    const auto default_act = file.format().network_format().default_activation();
    ActivationFunction act;
    if (default_act == pblczero::NetworkFormat::DEFAULT_ACTIVATION_SILU)
      act = ACTIVATION_SWISH;
    else if (default_act == pblczero::NetworkFormat::DEFAULT_ACTIVATION_MISH)
      act = ACTIVATION_MISH;
    else
      act = ACTIVATION_RELU;

    // 2. Build the network, and copy the weights to GPU memory.

    // Input conv only used if there are residual blocks in the network
    if (numBlocks_ > 0) {
      // Input.
      {
        auto inputConv = std::make_unique<FusedWinogradConvSELayer<DataType>>(
            nullptr, kNumFilters, 8, 8, kNumInputPlanes, act, true, false,
            false, 0, use_gemm_ex, use_res_block_winograd_fuse_opt_);
        inputConv->LoadWeights(&weights.input.weights[0],
                               &weights.input.biases[0], scratch_mem_);
        network_.emplace_back(std::move(inputConv));
      }

      // Residual block.
      for (int block = 0; block < numBlocks_; block++) {
        bool has_se = weights.residual[block].has_se;
        int se_k = (int)weights.residual[block].se.b1.size();

        if (use_res_block_winograd_fuse_opt_) {
          auto layer = std::make_unique<ResidualBlock<DataType>>(
              getLastLayer(), kNumFilters, has_se, se_k, use_gemm_ex,
              block == 0, block == (numBlocks_ - 1), act,
              deviceProp.sharedMemPerBlockOptin);
          layer->LoadWeights0(&weights.residual[block].conv1.weights[0],
                              &weights.residual[block].conv1.biases[0],
                              scratch_mem_);
          layer->LoadWeights1(&weights.residual[block].conv2.weights[0],
                              &weights.residual[block].conv2.biases[0],
                              scratch_mem_);
          if (has_se)
            layer->LoadSEWeights(&weights.residual[block].se.w1[0],
                                 &weights.residual[block].se.b1[0],
                                 &weights.residual[block].se.w2[0],
                                 &weights.residual[block].se.b2[0],
                                 scratch_mem_);
          network_.emplace_back(std::move(layer));
        } else {
          auto conv1 = std::make_unique<FusedWinogradConvSELayer<DataType>>(
              getLastLayer(), kNumFilters, 8, 8, kNumFilters, act, true, false,
              false, 0, use_gemm_ex);
          conv1->LoadWeights(&weights.residual[block].conv1.weights[0],
                             &weights.residual[block].conv1.biases[0],
                             scratch_mem_);
          network_.emplace_back(std::move(conv1));

          auto conv2 = std::make_unique<FusedWinogradConvSELayer<DataType>>(
              getLastLayer(), kNumFilters, 8, 8, kNumFilters, act, true, true,
              has_se, se_k, use_gemm_ex);
          conv2->LoadWeights(&weights.residual[block].conv2.weights[0],
                             &weights.residual[block].conv2.biases[0],
                             scratch_mem_);
          if (has_se)
            conv2->LoadSEWeights(&weights.residual[block].se.w1[0],
                                 &weights.residual[block].se.b1[0],
                                 &weights.residual[block].se.w2[0],
                                 &weights.residual[block].se.b2[0],
                                 scratch_mem_);
          network_.emplace_back(std::move(conv2));
        }
      }
      resi_last_ = getLastLayer();
    }

    if (attn_body_) {
      Activations activations;
      const auto smolgen_activation =
          file.format().network_format().smolgen_activation();
      activations.smolgen_activation =
          smolgen_activation == pblczero::NetworkFormat::ACTIVATION_DEFAULT
              ? act
              : static_cast<ActivationFunction>(smolgen_activation);
      const auto ffn_activation =
          file.format().network_format().ffn_activation();
      activations.ffn_activation =
          ffn_activation == pblczero::NetworkFormat::ACTIVATION_DEFAULT
              ? act
              : static_cast<ActivationFunction>(ffn_activation);
      activations.default_activation = act;

      // Weight arena: when enabled (env var LC0_USE_WEIGHT_ARENA=1
      // or backend option `weight_arena=true`), allocate weight
      // buffers from a single big-chunk arena instead of per-buffer
      // cudaMalloc.  Saves ~1-3 seconds at network load on large nets
      // (60+ encoder blocks).  Default off until validated in
      // production runs — set the env var to opt in.
      //
      // Scope: arena is active only during AttentionBody construction
      // (which includes all child EncoderBlock ctors).  Other layers
      // (input conv if residual, policy head, value head, MLH) are
      // built outside this scope and continue to use cudaMalloc.
      // Backend option `weight_arena=true` takes precedence over the
      // env var if both are present.  Either path activates the arena;
      // unset/false on both gives the legacy per-buffer cudaMalloc
      // path.  Default false — opt-in until measured Elo / nps impact
      // (none expected; pure load-time optimization).
      const bool arena_option =
          options.GetOrDefault<bool>("weight_arena", false);
      const char* use_arena_env = std::getenv("LC0_USE_WEIGHT_ARENA");
      const bool use_arena =
          arena_option || (use_arena_env != nullptr &&
                            std::string(use_arena_env) == "1");
      if (use_arena) {
        weight_arena_ = std::make_unique<WeightArena>();
        // No up-front Reserve(): with 64 MB chunks the arena self-
        // sizes via bump-allocator growth.  A typical 512×60 fp16 net
        // (~540 MB) ends at ~9 chunks = 576 MB total, each chunk
        // costing one cudaMalloc (~1-2 ms).  Total growth overhead =
        // ~9-18 ms — negligible vs the rest of net load, and the
        // chunks pack tight to actual usage (~6% fragmentation in the
        // last chunk; no hand-enumeration of the weight schema to
        // predict size up front).  See the design note in
        // weight_arena.h for chunk-size tuning rationale.
        CERR << "[weight_arena] activated; grows in 64 MB chunks";
      }

      // RAII scope: when use_arena is true, tl_weight_arena is set
      // for the duration of AttentionBody construction.  After this
      // block exits, the scope destructor restores the previous TLS
      // value (null), so subsequent head construction uses cudaMalloc.
      {
        WeightArenaScope arena_scope(use_arena ? weight_arena_.get()
                                               : nullptr);
        auto attention_body = std::make_unique<AttentionBody<DataType>>(
            weights, scratch_mem_, activations, numBlocks_,
            numBlocks_ > 0 ? kNumFilters : kInputPlanes, max_batch_size_,
            static_cast<InputEmbedding>(
                file.format().network_format().input_embedding()) ==
                InputEmbedding::INPUT_EMBEDDING_PE_DENSE,
            use_gemm_ex, use_fused_mha);
        attention_body_layer_ = attention_body.get();
        network_.emplace_back(std::move(attention_body));
      }

      if (use_arena && weight_arena_) {
        CERR << "[weight_arena] AttentionBody built; "
             << weight_arena_->BytesUsed() / (1024.0 * 1024.0)
             << " MB used across "
             << weight_arena_->ChunkCount() << " chunk(s) of "
             << weight_arena_->TotalAllocated() / (1024.0 * 1024.0)
             << " MB total";
      }

      encoder_last_ = getLastLayer();
    }

    // Policy head.
    {
      MultiHeadWeights::PolicyHead& head = weights.policy_heads.at(policy_head);
      // Cache encoder_last_ explicitly — after we push the vanilla
      // policy head onto network_, getLastLayer() advances past the
      // encoder and the optimistic head needs to read from the SAME
      // trunk output (encoder_last_), not from the vanilla policy
      // chain's intermediate buffers.
      BaseLayer<DataType>* policy_input =
          attn_body_ ? encoder_last_ : getLastLayer();
      if (attn_policy_) {
        auto AttentionPolicy = std::make_unique<AttentionPolicyHead<DataType>>(
            policy_input, head, scratch_mem_, attn_body_, act,
            max_batch_size_, use_gemm_ex);
        network_.emplace_back(std::move(AttentionPolicy));

        auto policymap = std::make_unique<PolicyMapLayer<DataType>>(
            getLastLayer(), kNumOutputPolicy, 1, 1, 64 * 64 + 8 * 24, true);
        policymap->LoadWeights(kAttnPolicyMap, scratch_mem_);
        network_.emplace_back(std::move(policymap));

        // Build optional optimistic policy head reading from the SAME
        // policy_input (encoder_last_).  The KataGo-style optimistic
        // head is trained to upweight policy mass on positions where
        // the played move's short-term value beat the network's
        // prediction (z-scored by predicted err head).  At inference
        // we blend its output with the vanilla policy at root edges.
        //
        // Auto-enabled when the proto carries a second head named
        // "optimistic" — no separate config option needed.  Search
        // gates blending via --optimistic-policy-weight (default 0,
        // so even a net with the head present is a no-op cost wise
        // beyond the extra forward pass through the small policy
        // head layers).
        if (weights.policy_heads.count("optimistic")) {
          auto& opt_head = weights.policy_heads.at("optimistic");
          auto AttentionPolicyOpt =
              std::make_unique<AttentionPolicyHead<DataType>>(
                  policy_input, opt_head, scratch_mem_, attn_body_, act,
                  max_batch_size_, use_gemm_ex);
          network_.emplace_back(std::move(AttentionPolicyOpt));

          auto policymap_opt = std::make_unique<PolicyMapLayer<DataType>>(
              getLastLayer(), kNumOutputPolicy, 1, 1, 64 * 64 + 8 * 24, true);
          policymap_opt->LoadWeights(kAttnPolicyMap, scratch_mem_);
          network_.emplace_back(std::move(policymap_opt));
          has_optimistic_policy_ = true;
        }
      } else {
        if (conv_policy_) {
          assert(!attn_body_);  // not supported with attention body
          auto conv1 = std::make_unique<FusedWinogradConvSELayer<DataType>>(
              resi_last_, kNumFilters, 8, 8, kNumFilters, act, true, false,
              false, 0, use_gemm_ex);
          conv1->LoadWeights(&head.policy1.weights[0], &head.policy1.biases[0],
                             scratch_mem_);
          network_.emplace_back(std::move(conv1));

          auto pol_channels = head.policy.biases.size();

          // No relu
          auto conv2 = std::make_unique<FusedWinogradConvSELayer<DataType>>(
              getLastLayer(), pol_channels, 8, 8, kNumFilters, ACTIVATION_NONE,
              true, false, false, 0, use_gemm_ex);
          conv2->LoadWeights(&head.policy.weights[0], &head.policy.biases[0],
                             scratch_mem_);
          network_.emplace_back(std::move(conv2));

          auto policymap = std::make_unique<PolicyMapLayer<DataType>>(
              getLastLayer(), kNumOutputPolicy, 1, 1, 73 * 8 * 8, false);
          policymap->LoadWeights(kConvPolicyMap, scratch_mem_);

          network_.emplace_back(std::move(policymap));
        } else {
          assert(!attn_body_);  // not supported with attention body
          auto convPol = std::make_unique<Conv1Layer<DataType>>(
              resi_last_, head.policy.biases.size(), 8, 8, kNumFilters, act,
              true, use_gemm_ex);
          convPol->LoadWeights(&head.policy.weights[0], &head.policy.biases[0],
                               scratch_mem_);
          network_.emplace_back(std::move(convPol));

          auto FCPol = std::make_unique<FCLayer<DataType>>(
              getLastLayer(), head.ip_pol_b.size(), 1, 1, true,
              ACTIVATION_NONE);
          FCPol->LoadWeights(&head.ip_pol_w[0], &head.ip_pol_b[0],
                             scratch_mem_);
          network_.emplace_back(std::move(FCPol));
        }
      }
    }

    // Value heads.
    {
      const MultiHeadWeights::ValueHead& head =
          weights.value_heads.at(value_head);
      wdl_ = file.format().network_format().value() ==
             pblczero::NetworkFormat::VALUE_WDL;
      BaseLayer<DataType>* lastlayer = attn_body_ ? encoder_last_ : resi_last_;
      auto value_main = std::make_unique<ValueHead<DataType>>(
          lastlayer, head, scratch_mem_, attn_body_, wdl_, act, max_batch_size_,
          use_gemm_ex);
      // Multi-exit value branch: if AttentionBody has a branch buffer
      // (value_branch_at_layer was set in the proto), point the value
      // head at it so it reads the intermediate-layer flow instead of
      // the final encoder output. Policy heads continue reading the
      // final flow as normal.
      if (attention_body_layer_ != nullptr &&
          attention_body_layer_->HasValueBranch()) {
        value_main->SetInputOverride(
            attention_body_layer_->GetValueBranchOutput());
      }
      network_.emplace_back(std::move(value_main));
    }

    // Moves left head
    moves_left_ = (file.format().network_format().moves_left() ==
                   pblczero::NetworkFormat::MOVES_LEFT_V1) &&
                  options.GetOrDefault<bool>("mlh", true);
    if (moves_left_) {
      if (attn_body_) {
        auto embedded_mov = std::make_unique<EmbeddingLayer<DataType>>(
            encoder_last_, weights.ip_mov_w, weights.ip_mov_b, scratch_mem_,
            act);
        network_.emplace_back(std::move(embedded_mov));
      } else {
        auto convMov = std::make_unique<Conv1Layer<DataType>>(
            resi_last_, weights.moves_left.biases.size(), 8, 8, kNumFilters,
            act, true, use_gemm_ex);
        convMov->LoadWeights(&weights.moves_left.weights[0],
                             &weights.moves_left.biases[0], scratch_mem_);
        network_.emplace_back(std::move(convMov));
      }
      auto FCMov1 = std::make_unique<FCLayer<DataType>>(
          getLastLayer(), weights.ip1_mov_b.size(), 1, 1, true, act);
      FCMov1->LoadWeights(&weights.ip1_mov_w[0], &weights.ip1_mov_b[0],
                          scratch_mem_);
      network_.emplace_back(std::move(FCMov1));

      auto FCMov2 = std::make_unique<FCLayer<DataType>>(getLastLayer(), 1, 1, 1,
                                                        true, ACTIVATION_RELU);
      FCMov2->LoadWeights(&weights.ip2_mov_w[0], &weights.ip2_mov_b[0],
                          scratch_mem_);
      network_.emplace_back(std::move(FCMov2));
    }

    // 3. Allocate GPU memory for running the network:
    //    - three buffers of max size are enough (one to hold input, second to
    //      hold output and third to hold skip connection's input).

    // size of input to the network
    size_t maxSize = max_batch_size_ * kNumInputPlanes * 64 * sizeof(DataType);

    // take max size of all layers
    for (auto& layer : network_) {
      maxSize = std::max(maxSize, layer->GetOutputSize(max_batch_size_));
    }

    if ((attn_policy_ || use_res_block_winograd_fuse_opt_ || attn_body_) &&
        (scratch_size_ > maxSize)) {
      maxSize = scratch_size_;
    }

    if (!multi_stream_) {
      for (auto& mem : tensor_mem_) {
        ReportCUDAErrors(cudaMalloc(&mem, maxSize));
        ReportCUDAErrors(cudaMemset(mem, 0, maxSize));
      }
    }

    tensor_mem_size_ = multi_stream_ ? maxSize : 0;

    // pre-allocate cuda graphs for search threads
    auto allocateCudaGraphs = [&] {
      ReportCUDAErrors(cudaSetDevice(gpu_id_));
      CudaNetworkComputation<DataType> comp(this, wdl_, moves_left_);
      comp.AddInput(InputPlanes{(size_t)kNumInputPlanes});
      // Make sure cublas is initialized in this thread.
      comp.ComputeBlocking();
      for (int i = 0; i < GetMiniBatchSize(); i++) {
        comp.AddInput(InputPlanes{(size_t)kNumInputPlanes});
        auto lock = LockEval();
        comp.CaptureGraph(std::move(lock));
      }
    };
    std::thread t2(allocateCudaGraphs);
    allocateCudaGraphs();
    t2.join();
  }

  std::unique_lock<std::mutex> LockEval() {
    if (multi_stream_) {
      return {};
    } else {
      return std::unique_lock<std::mutex>{lock_};
    }
  }

  bool GetGraphCaptureEnabled() const { return enable_graph_capture_; }

  // Captures the FFN graph for the given batch size (no-op if no parallel FFN).
  void CaptureFFNGraphForBatch(int N) {
    if (attention_body_layer_) attention_body_layer_->CaptureFFNGraph(N);
  }

  CudaGraphCapture<DataType> BeginCapture(InputsOutputs<DataType>& io) {
    if (!multi_stream_) {
#if CUDA_GRAPH_SUPPORTS_EXTERNAL_EVENTS
      return {io, upload_stream_, download_stream_};
#else
      return {io, compute_stream_, download_stream_};
#endif
    } else {
      return {io, io.upload_stream_, io.download_stream_};
    }
  }

  void UploadInputs(InputsOutputs<DataType>* io, int batchSize) {
    // Multu-stream can capture uploads without external events.
    if (multi_stream_) return;
    ReportCUDAErrors(
        cudaMemcpyAsync(io->input_masks_mem_gpu_, io->input_masks_mem_,
                        batchSize * kInputPlanes * sizeof(uint64_t),
                        cudaMemcpyHostToDevice, upload_stream_));
    ReportCUDAErrors(cudaMemcpyAsync(
        io->input_val_mem_gpu_, io->input_val_mem_,
        batchSize * kInputPlanes * sizeof(io->input_val_mem_[0]),
        cudaMemcpyHostToDevice, upload_stream_));
    ReportCUDAErrors(cudaEventRecord(io->upload_done_event_, upload_stream_));
    ReportCUDAErrors(
        cudaStreamWaitEvent(compute_stream_, io->upload_done_event_, 0));
  }

  void GraphLaunch(InputsOutputs<DataType>* io, int batchSize) {
#if CUDA_GRAPH_SUPPORTS_EXTERNAL_EVENTS
    io->cuda_graphs_[batchSize - 1].Launch(io->exec_stream_);
#else
    if (!multi_stream_) {
      UploadInputs(io, batchSize);

      io->cuda_graphs_[batchSize - 1].Launch(compute_stream_);
      ReportCUDAErrors(
          cudaEventRecord(io->download_done_event_, compute_stream_));
    } else {
      io->cuda_graphs_[batchSize - 1].Launch(io->exec_stream_);
      ReportCUDAErrors(
          cudaEventRecord(io->download_done_event_, io->exec_stream_));
    }
#endif
  }

  void forwardEval(InputsOutputs<DataType>* io, int batchSize,
                   [[maybe_unused]] bool capture = false) {
    // It is safe to evaluate larger than the batchSize
    // as all buffers are designed to handle max_batch_size
    // and the extra invalid results are never read.
    if (batchSize < min_batch_size_) batchSize = min_batch_size_;

#ifdef DEBUG_RAW_NPS
    auto t_start = std::chrono::high_resolution_clock::now();
#endif

    // Expand packed planes to full planes.
    uint64_t* ipDataMasks = io->input_masks_mem_gpu_;
    auto* ipDataValues = io->input_val_mem_gpu_;

    DataType* tensor_mem[3];
    void* scratch_mem;
    DataType*** offset_pointers;
    DataType*** head_offset_pointers;
    cudaStream_t compute_stream, upload_stream, download_stream;
    cublasHandle_t cublas;
    if (multi_stream_) {
      // We use tensor and scratch memory from InputOutputs (so that multiple
      // requests can run in parallel)
      for (int i = 0; i < 3; i++) tensor_mem[i] = (DataType*)io->tensor_mem_[i];
      scratch_mem = io->scratch_mem_;
      offset_pointers = (DataType***)&io->offset_pointers_;
      head_offset_pointers = (DataType***)&io->head_offset_pointers_;
      compute_stream = io->compute_stream_;
      upload_stream = io->upload_stream_;
      download_stream = io->download_stream_;
      cublas = io->cublas_;
    } else {
      for (int i = 0; i < 3; i++) tensor_mem[i] = tensor_mem_[i];
      scratch_mem = scratch_mem_;
      offset_pointers = (DataType***)&offset_pointers_;
      head_offset_pointers = (DataType***)&head_offset_pointers_;
      compute_stream = compute_stream_;
      upload_stream = upload_stream_;
      download_stream = download_stream_;
      cublas = cublas_;
    }

    if (multi_stream_ || CUDA_GRAPH_SUPPORTS_EXTERNAL_EVENTS) {
      ReportCUDAErrors(
          cudaMemcpyAsync(io->input_masks_mem_gpu_, io->input_masks_mem_,
                          batchSize * kInputPlanes * sizeof(uint64_t),
                          cudaMemcpyHostToDevice, upload_stream));
      ReportCUDAErrors(cudaMemcpyAsync(
          io->input_val_mem_gpu_, io->input_val_mem_,
          batchSize * kInputPlanes * sizeof(io->input_val_mem_[0]),
          cudaMemcpyHostToDevice, upload_stream));
      ReportCUDAErrors(cudaEventRecord(io->upload_done_event_, upload_stream));
      ReportCUDAErrors(
          cudaStreamWaitEvent(compute_stream, io->upload_done_event_, 0));
    }

    if (!multi_stream_) {
#if CUDA_GRAPH_SUPPORTS_EXTERNAL_EVENTS
      ReportCUDAErrors(
          cudaStreamWaitEvent(compute_stream, compute_ordering_event_,
                              capture ? cudaEventWaitExternal : 0));
#endif
    }

    expandPlanes_NCHW(tensor_mem[0], ipDataMasks, ipDataValues,
                      batchSize * kInputPlanes, compute_stream);

    auto* opPol = io->op_policy_mem_gpu_;
    auto* opVal = io->op_value_mem_gpu_;
    auto* opMov = io->op_moves_left_mem_gpu_;

    // Figure out if the memory requirment for running the res block would fit
    // in the L2 cache.
    bool enableCacheOpt = false;
    DataType* skip_connection =
        use_res_block_winograd_fuse_opt_ ? tensor_mem[1] : tensor_mem[2];

#if CUDART_VERSION >= 11000
    const int pre_transform_tensor_size =
        batchSize * numFilters_ * 8 * 8 * sizeof(DataType);
    const int transformed_tensor_size = pre_transform_tensor_size * 36 / 16;
    const int res_block_mem =
        transformed_tensor_size * 2 + pre_transform_tensor_size;

    cudaStreamAttrValue stream_attribute = {};
    stream_attribute.accessPolicyWindow.base_ptr = tensor_mem[2];
    stream_attribute.accessPolicyWindow.num_bytes = res_block_mem;
    stream_attribute.accessPolicyWindow.hitRatio = 1.0f;
    stream_attribute.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    stream_attribute.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;

    if (allow_cache_opt_ && use_res_block_winograd_fuse_opt_ &&
        (static_cast<size_t>(res_block_mem) <= scratch_size_) &&
        (res_block_mem <= l2_cache_size_)) {
      // we can use a single alloc to hold all the required tensors, and enable
      // persistent L2 caching on it
      ReportCUDAErrors(cudaStreamSetAttribute(
          compute_stream, cudaStreamAttributeAccessPolicyWindow,
          &stream_attribute));

      enableCacheOpt = true;
      skip_connection =
          tensor_mem[2] + 2 * transformed_tensor_size / sizeof(DataType);
    }
#endif

    int l = 0;

    DataType* flow = tensor_mem[0];
    DataType* spare1 = tensor_mem[1];
    DataType* spare2 = tensor_mem[2];

    if (numBlocks_ > 0) {
      // Input.
      network_[l++]->Eval(batchSize, skip_connection, tensor_mem[0], nullptr,
                          scratch_mem, scratch_size_, nullptr, cublas,
                          compute_stream);  // input conv

      // Residual block.
      for (int block = 0; block < numBlocks_; block++) {
        if (use_res_block_winograd_fuse_opt_) {
          network_[l++]->Eval(batchSize, tensor_mem[2], skip_connection,
                              nullptr, enableCacheOpt ? nullptr : scratch_mem,
                              scratch_size_, nullptr, cublas,
                              compute_stream);  // block
        } else {
          network_[l++]->Eval(batchSize, tensor_mem[0], tensor_mem[2], nullptr,
                              scratch_mem, scratch_size_, nullptr, cublas,
                              compute_stream);  // conv1

          network_[l++]->Eval(batchSize, tensor_mem[2], tensor_mem[0],
                              tensor_mem[2], scratch_mem, scratch_size_,
                              nullptr, cublas, compute_stream);  // conv2
        }
      }

      flow = tensor_mem[2];
      spare1 = tensor_mem[0];
      spare2 = tensor_mem[1];
    }

    if (attn_body_) {
      network_[l++]->Eval(
          batchSize, tensor_mem[1],
          (numBlocks_ > 0) ? tensor_mem[2] : tensor_mem[0],
          (numBlocks_ > 0) ? tensor_mem[0] : tensor_mem[2], scratch_mem,
          scratch_size_, nullptr, cublas, compute_stream,
          offset_pointers);  // Entire attention body of the network

      flow = tensor_mem[1];
      spare1 = tensor_mem[0];
      spare2 = tensor_mem[2];
    }

#if CUDART_VERSION >= 11000
    if (enableCacheOpt) {
      // reset the cache settings
      stream_attribute.accessPolicyWindow.num_bytes = 0;
      cudaStreamSetAttribute(compute_stream,
                             cudaStreamAttributeAccessPolicyWindow,
                             &stream_attribute);
      cudaCtxResetPersistingL2Cache();
    }
#endif

    // Policy head.
    if (attn_policy_) {
      network_[l++]->Eval(
          batchSize, spare1, flow, spare2, scratch_mem, scratch_size_, nullptr,
          cublas, compute_stream,
          head_offset_pointers);  // Entire Attention policy head except for the
                                  // policy map
      network_[l++]->Eval(
          batchSize, (DataType*)opPol, spare1, nullptr, scratch_mem,
          scratch_size_, nullptr, cublas,
          compute_stream);  // policy map layer  // POLICY output

    } else if (conv_policy_) {
      network_[l++]->Eval(batchSize, spare1, flow, nullptr, scratch_mem,
                          scratch_size_, nullptr, cublas,
                          compute_stream);  // policy conv1

      network_[l++]->Eval(batchSize, spare2, spare1, nullptr, scratch_mem,
                          scratch_size_, nullptr, cublas,
                          compute_stream);  // policy conv2

      network_[l++]->Eval(
          batchSize, (DataType*)opPol, spare2, nullptr, scratch_mem,
          scratch_size_, nullptr, cublas,
          compute_stream);  // policy map layer  // POLICY output
    } else {
      network_[l++]->Eval(batchSize, spare1, flow, nullptr, scratch_mem,
                          scratch_size_, nullptr, cublas,
                          compute_stream);  // pol conv

      network_[l++]->Eval(batchSize, (DataType*)opPol, spare1, nullptr,
                          scratch_mem, scratch_size_, nullptr, cublas,
                          compute_stream);  // pol FC  // POLICY
    }
    // When a GPU-side blend is active, the blend kernel below OVERWRITES
    // op_policy_mem_gpu_ (the vanilla slot) on compute_stream *after* the
    // vanilla policy map.  If we copied the vanilla slot to host here —
    // gated only on policy_done_event_ (recorded before the optimistic
    // chain + blend) — the copy would (a) race the blend kernel's write
    // across streams, and (b) capture PRE-blend vanilla logits, so the
    // blend's effect on the vanilla slot would never reach the host.
    // Defer the vanilla copy to after the blend in that case (see the
    // matching block below).  When no blend overwrites the vanilla slot,
    // copy eagerly here so the D2H transfer overlaps the optimistic chain.
    const bool gpu_blend_active =
        has_optimistic_policy_ && gpu_blend_alpha_ > 0.0f;
    if (!gpu_blend_active) {
      ReportCUDAErrors(cudaEventRecord(io->policy_done_event_, compute_stream));
      ReportCUDAErrors(
          cudaStreamWaitEvent(download_stream, io->policy_done_event_, 0));

      // Copy policy output from device memory to host memory.
      ReportCUDAErrors(cudaMemcpyAsync(
          io->op_policy_mem_, io->op_policy_mem_gpu_,
          sizeof(io->op_policy_mem_[0]) * kNumOutputPolicy * batchSize,
          cudaMemcpyDeviceToHost, download_stream));
    }

    // Optimistic policy head — runs on compute_stream right after the
    // vanilla policy chain, reads the SAME encoder output (`flow`)
    // that the vanilla AttentionPolicyHead read, writes to a separate
    // device buffer.  Each AttentionPolicyHead chain is self-contained
    // (the first GEMM writes input2/spare2 with beta=0 before reading
    // anything from it), so running vanilla then optimistic on the
    // same scratch buffers is safe.
    //
    // Only attention-policy is supported by this branch — the legacy
    // conv-policy / FC-policy code paths are pre-attention-body and
    // do not produce nets with the optimistic head.
    if (has_optimistic_policy_) {
      network_[l++]->Eval(
          batchSize, spare1, flow, spare2, scratch_mem, scratch_size_,
          nullptr, cublas, compute_stream,
          head_offset_pointers);  // optimistic AttentionPolicyHead
      network_[l++]->Eval(
          batchSize, (DataType*)io->op_policy_opt_mem_gpu_, spare1,
          nullptr, scratch_mem, scratch_size_, nullptr, cublas,
          compute_stream);  // optimistic policy map  // OPT POLICY output
      // GPU-side optimistic policy blend.  Two paths depending on
      // gpu_blend_alpha_:
      //
      //   gpu_blend_alpha_ == 0  → current default behaviour
      //     - Do nothing here.  Vanilla logits stay in op_policy_mem_gpu_;
      //       optimistic logits stay in op_policy_opt_mem_gpu_.  Both
      //       get memcpy'd to host.  Wrapper softmaxes both.  Search
      //       does the host-side blend per node.
      //
      //   gpu_blend_alpha_ > 0  → GPU-only blend (Speedup A)
      //     - Blend kernel writes (1-α)·vanilla + α·optimistic INTO the
      //       vanilla buffer op_policy_mem_gpu_.  The optimistic buffer
      //       is no longer needed on host so we skip its memcpy entirely.
      //     - InputsOutputs allocated with optimistic_policy_host=false,
      //       so op_policy_opt_mem_ == nullptr.  HasOptimisticPolicy()
      //       returns false.  Wrapper softmaxes only the vanilla slot
      //       (now blended).  Memcache stores only p.  Search uses the
      //       vanilla SetP path.  Eliminates the second host softmax,
      //       the p_optimistic propagation cost, and the optimistic
      //       memcpy bandwidth.
      //     - User must run with --optimistic-policy-weight=0 and
      //       --optimistic-policy-weight-internal=0 — the blend has
      //       already been done on GPU; setting search alphas > 0 would
      //       try to re-blend zero-initialised p_optimistic spans and
      //       produce catastrophic priors.
      //
      // Compute_stream ordering: the blend kernel reads both vanilla
      // (written earlier on the same stream) and optimistic (just
      // written above) and writes vanilla — all three operations are
      // sequenced by compute_stream.  Safe to read+write in place.
      // Dispatch blend kernel.  Three cases:
      //   (a) gpu_blend_alpha_ == 0: no GPU blend, host does it per-node
      //   (b) gpu_blend_alpha_ > 0, gpu_blend_alpha_root_ == 0 (or
      //       equal to gpu_blend_alpha_): uniform mode.  Blend writes
      //       into vanilla buffer; optimistic host buffer not needed.
      //   (c) gpu_blend_alpha_ > 0 AND gpu_blend_alpha_root_ > 0 AND
      //       they differ: split mode.  Dual-output kernel writes
      //       vanilla=blended-at-root and optimistic=blended-at-internal
      //       in one read pass.
      const bool gpu_blend_split =
          gpu_blend_alpha_ > 0.0f && gpu_blend_alpha_root_ > 0.0f &&
          gpu_blend_alpha_ != gpu_blend_alpha_root_;
      if (gpu_blend_split) {
        BlendPolicyLogitsDual<DataType>(
            kNumOutputPolicy * batchSize,
            (DataType*)io->op_policy_mem_gpu_,        // out_v = vanilla
            (DataType*)io->op_policy_opt_mem_gpu_,    // out_o = optimistic
            (const DataType*)io->op_policy_mem_gpu_,  // input vanilla
            (const DataType*)io->op_policy_opt_mem_gpu_,  // input optimistic
            gpu_blend_alpha_root_, gpu_blend_alpha_, compute_stream);
      } else if (gpu_blend_alpha_ > 0.0f) {
        BlendPolicyLogits<DataType>(
            kNumOutputPolicy * batchSize,
            (DataType*)io->op_policy_mem_gpu_,        // output = vanilla
            (const DataType*)io->op_policy_mem_gpu_,  // input vanilla
            (const DataType*)io->op_policy_opt_mem_gpu_, gpu_blend_alpha_,
            compute_stream);
      }
      // Deferred vanilla copy (see the gpu_blend_active comment above the
      // optimistic chain).  Now that the blend kernel has written its
      // result into op_policy_mem_gpu_ on compute_stream, record the
      // policy_done_event_ AFTER the blend and copy the (blended) vanilla
      // slot to host.  This is the copy that was skipped earlier so it
      // wouldn't race the blend / capture pre-blend vanilla.  Only the
      // gpu_blend_active path reaches here with the early copy skipped.
      if (gpu_blend_active) {
        ReportCUDAErrors(
            cudaEventRecord(io->policy_done_event_, compute_stream));
        ReportCUDAErrors(
            cudaStreamWaitEvent(download_stream, io->policy_done_event_, 0));
        ReportCUDAErrors(cudaMemcpyAsync(
            io->op_policy_mem_, io->op_policy_mem_gpu_,
            sizeof(io->op_policy_mem_[0]) * kNumOutputPolicy * batchSize,
            cudaMemcpyDeviceToHost, download_stream));
      }
      if (io->op_policy_opt_mem_ != nullptr) {
        // Host buffer was allocated → CPU pipeline expects the
        // optimistic logits.  Record the event and copy them.
        ReportCUDAErrors(
            cudaEventRecord(io->policy_opt_done_event_, compute_stream));
        ReportCUDAErrors(cudaStreamWaitEvent(
            download_stream, io->policy_opt_done_event_, 0));
        ReportCUDAErrors(cudaMemcpyAsync(
            io->op_policy_opt_mem_, io->op_policy_opt_mem_gpu_,
            sizeof(io->op_policy_opt_mem_[0]) * kNumOutputPolicy * batchSize,
            cudaMemcpyDeviceToHost, download_stream));
      }

      // Debug: dump first N policy values from BOTH heads on first
      // inference, when LC0_DEBUG_OPT_HEAD=1 is set in env.  Prints
      // raw GPU-side values so we can see whether vanilla and
      // optimistic chains diverge, and whether the optimistic chain's
      // output is garbage (NaN, 0, extreme).  Compare against running
      // with policy_head=optimistic backend-opt — that path uses the
      // SAME weights but as the primary head; if op_policy_opt_mem_
      // here differs significantly from op_policy_mem_ in that run,
      // the secondary chain has a bug.
      static std::atomic<int> opt_dbg_remaining{
          [](){
            const char* e = std::getenv("LC0_DEBUG_OPT_HEAD");
            return e ? std::atoi(e) : 0;
          }()};
      if (opt_dbg_remaining.load(std::memory_order_relaxed) > 0) {
        const int slot =
            opt_dbg_remaining.fetch_sub(1, std::memory_order_relaxed);
        if (slot > 0) {
          // Sync compute_stream so we can safely read from GPU.
          ReportCUDAErrors(cudaStreamSynchronize(compute_stream));
          ReportCUDAErrors(cudaStreamSynchronize(download_stream));
          std::vector<DataType> vanilla_dev(kNumOutputPolicy);
          std::vector<DataType> optimistic_dev(kNumOutputPolicy);
          ReportCUDAErrors(cudaMemcpy(
              vanilla_dev.data(), io->op_policy_mem_gpu_,
              sizeof(DataType) * kNumOutputPolicy,
              cudaMemcpyDeviceToHost));
          ReportCUDAErrors(cudaMemcpy(
              optimistic_dev.data(), io->op_policy_opt_mem_gpu_,
              sizeof(DataType) * kNumOutputPolicy,
              cudaMemcpyDeviceToHost));
          // Compute stats over the full 1858-slot buffer and sample
          // the slots at starting-position legal moves (so we can
          // diff directly against the verbose-search printout).
          auto stats = [](const std::vector<DataType>& v) {
            float mn = +std::numeric_limits<float>::infinity();
            float mx = -std::numeric_limits<float>::infinity();
            double sum = 0.0;
            int nan_count = 0, inf_count = 0;
            for (auto x : v) {
              float f = FromType(x);
              if (std::isnan(f)) { ++nan_count; continue; }
              if (std::isinf(f)) { ++inf_count; continue; }
              mn = std::min(mn, f);
              mx = std::max(mx, f);
              sum += f;
            }
            int valid = static_cast<int>(v.size()) - nan_count -
                         inf_count;
            return std::make_tuple(mn, mx, valid > 0 ? sum / valid : 0.0,
                                    nan_count, inf_count);
          };
          auto [vmn, vmx, vmean, vnan, vinf] = stats(vanilla_dev);
          auto [omn, omx, omean, onan, oinf] = stats(optimistic_dev);
          // Starting-position legal move indices (from the verbose
          // search output's "(NNN )" tags).
          const std::array<int, 20> start_moves = {
              34, 36, 159, 161, 204, 207, 230, 234, 259, 264,
              288, 293, 317, 322, 346, 351, 374, 378, 400, 403};
          std::ostringstream oss;
          oss << "[DBG OPT HEAD] slot=" << slot
              << " batchSize=" << batchSize
              << " fp" << (std::is_same<DataType, half>::value ? 16 : 32);
          oss << "\n  vanilla stats: min=" << vmn << " max=" << vmx
              << " mean=" << vmean << " nan=" << vnan
              << " inf=" << vinf;
          oss << "\n  optimstc stats: min=" << omn << " max=" << omx
              << " mean=" << omean << " nan=" << onan
              << " inf=" << oinf;
          oss << "\n  vanilla @ start-pos legal-move idx:";
          for (int idx : start_moves) {
            oss << " " << FromType(vanilla_dev[idx]);
          }
          oss << "\n  optimstc @ start-pos legal-move idx:";
          for (int idx : start_moves) {
            oss << " " << FromType(optimistic_dev[idx]);
          }
          CERR << oss.str();
        }
      }
    }

    // ── PARALLEL-FFN TAINT FIX ──────────────────────────────────────────
    // Route the value head to value_stream_/value_cublas_ when parallel FFN is
    // active.  value_stream_ receives exactly 1 cross-stream wait per inference
    // (attn_body_done_event_), keeping it below the Ada Lovelace + CUDA 12.9
    // fork-join taint threshold.  compute_stream accumulates N+1 taints (N
    // ffn_done + 1 upload_done); ffn_stream_ also accumulates N taints (N
    // ln1_done).  Both are enough to trigger tensor-core-only kernel selection
    // for the m=3 WDL GEMM → INTERNAL_ERROR.  value_stream_ avoids this.
    // Gate on IsParallelFFN() (is_parallel_ffn_), not HasMultiStreamFFN()
    // (multi_stream_ffn_).  Even when the FFN runs sequentially (no separate
    // ffn_stream_), the volume of large-m tensor-core GEMMs in the parallel-FFN
    // encoder path primes cuBLAS into tensor-core-only kernel selection.  That
    // causes INTERNAL_ERROR for the subsequent m=3 WDL dense-2 GEMM because no
    // tensor-core-aligned kernel exists for m=3.  value_cublas_ (PEDANTIC_MATH,
    // fresh handle, never used for any prior GEMM) always finds a non-TC kernel.
    const bool val_on_ffn =
        attn_body_ && !multi_stream_ && attention_body_layer_ &&
        attention_body_layer_->IsParallelFFN();
    cudaStream_t val_stream = val_on_ffn ? value_stream_ : compute_stream;
    cublasHandle_t val_cublas = val_on_ffn ? value_cublas_ : cublas;
    if (val_on_ffn) {
      // Signal value_stream_ that the attention-body output is ready.
      ReportCUDAErrors(cudaEventRecord(attn_body_done_event_, compute_stream));
      ReportCUDAErrors(
          cudaStreamWaitEvent(val_stream, attn_body_done_event_, 0));
    }

    // value head
    network_[l++]->Eval(batchSize, (DataType*)opVal, flow, spare2, scratch_mem,
                        scratch_size_, nullptr, val_cublas,
                        val_stream);  // value head

    // Record value_done_event on val_stream so both compute_stream
    // (tensor_mem safety) and download_stream (memcpy ordering) can wait.
    ReportCUDAErrors(cudaEventRecord(io->value_done_event_, val_stream));
    if (val_on_ffn) {
      // Sync compute_stream with value_stream_ to guard tensor_mem[1] before
      // the next inference.  This adds 1 more cross-stream taint to compute_stream
      // but the value head has already completed; subsequent compute_stream work
      // uses only large-m GEMMs that are unaffected by the taint.
      ReportCUDAErrors(
          cudaStreamWaitEvent(compute_stream, io->value_done_event_, 0));
    }
    if (!moves_left_ && !multi_stream_) {
#if CUDA_GRAPH_SUPPORTS_EXTERNAL_EVENTS
      ReportCUDAErrors(
          cudaEventRecordWithFlags(compute_ordering_event_, compute_stream,
                                   capture ? cudaEventRecordExternal : 0));
#endif
    }
    ReportCUDAErrors(
        cudaStreamWaitEvent(download_stream, io->value_done_event_, 0));
    ReportCUDAErrors(cudaMemcpyAsync(
        io->op_value_mem_, io->op_value_mem_gpu_,
        sizeof(io->op_value_mem_[0]) * (wdl_ ? 3 : 1) * batchSize,
        cudaMemcpyDeviceToHost, download_stream));

    if (wdl_) {
#if CUDA_GRAPH_SUPPORTS_EXTERNAL_EVENTS
      ReportCUDAErrors(cudaEventRecordWithFlags(
          io->wdl_download_done_event_, download_stream,
          capture ? cudaEventRecordExternal : 0));
#endif
    }

    if (moves_left_) {
      // Moves left head
      network_[l++]->Eval(batchSize, spare1, flow, nullptr, scratch_mem,
                          scratch_size_, nullptr, cublas,
                          compute_stream);  // moves conv or embedding

      network_[l++]->Eval(batchSize, spare2, spare1, nullptr, scratch_mem,
                          scratch_size_, nullptr, cublas,
                          compute_stream);  // moves FC1

      // Moves left FC2
      network_[l++]->Eval(batchSize, (DataType*)opMov, spare2, nullptr,
                          scratch_mem, scratch_size_, nullptr, cublas,
                          compute_stream);
      if (!multi_stream_) {
#if CUDA_GRAPH_SUPPORTS_EXTERNAL_EVENTS
        ReportCUDAErrors(
            cudaEventRecordWithFlags(compute_ordering_event_, compute_stream,
                                     capture ? cudaEventRecordExternal : 0));
#endif
      }
      ReportCUDAErrors(
          cudaEventRecord(io->moves_left_done_event_, compute_stream));
      ReportCUDAErrors(
          cudaStreamWaitEvent(download_stream, io->moves_left_done_event_, 0));
      ReportCUDAErrors(
          cudaMemcpyAsync(io->op_moves_left_mem_, io->op_moves_left_mem_gpu_,
                          sizeof(io->op_moves_left_mem_[0]) * batchSize,
                          cudaMemcpyDeviceToHost, download_stream));
    }
#if CUDA_GRAPH_SUPPORTS_EXTERNAL_EVENTS
    ReportCUDAErrors(
        cudaEventRecordWithFlags(io->download_done_event_, download_stream,
                                 capture ? cudaEventRecordExternal : 0));
#else
    if (!capture) {
      ReportCUDAErrors(
          cudaEventRecord(io->download_done_event_, download_stream));
    }
#endif
  }

  void finishEval(InputsOutputs<DataType>* io, int batchSize) {
#if !CUDA_GRAPH_SUPPORTS_EXTERNAL_EVENTS
    ReportCUDAErrors(cudaEventSynchronize(io->download_done_event_));
#endif
    if (wdl_) {
#if CUDA_GRAPH_SUPPORTS_EXTERNAL_EVENTS
      ReportCUDAErrors(cudaEventSynchronize(io->wdl_download_done_event_));
#endif
      // Value softmax done cpu side.
      for (int i = 0; i < batchSize; i++) {
        float* wdl = sizeof(io->op_value_mem_[0]) == sizeof(float)
                         ? (float*)io->op_value_mem_
                         : io->wdl_cpu_softmax_.get();
        float w = FromType(io->op_value_mem_[3 * i + 0]);
        float d = FromType(io->op_value_mem_[3 * i + 1]);
        float l = FromType(io->op_value_mem_[3 * i + 2]);
        float m = std::max({w, d, l});
        w = std::exp(w - m);
        d = std::exp(d - m);
        l = std::exp(l - m);
        float sum = w + d + l;
        w /= sum;
        l /= sum;
        d /= sum;
        wdl[2 * i + 0] = w - l;
        wdl[2 * i + 1] = d;
      }
    }
#if CUDA_GRAPH_SUPPORTS_EXTERNAL_EVENTS
    ReportCUDAErrors(cudaEventSynchronize(io->download_done_event_));
#endif
  }

  ~CudaNetwork() {
    if (scratch_mem_) ReportCUDAErrors(cudaFree(scratch_mem_));
    if (!multi_stream_) {
      for (auto mem : tensor_mem_) {
        if (mem) ReportCUDAErrors(cudaFree(mem));
      }
      if (offset_pointers_) ReportCUDAErrors(cudaFree(offset_pointers_));
      if (head_offset_pointers_)
        ReportCUDAErrors(cudaFree(head_offset_pointers_));
      ReportCUBLASErrors(cublasDestroy(cublas_));
      ReportCUDAErrors(cudaStreamDestroy(compute_stream_));
      ReportCUDAErrors(cudaStreamDestroy(upload_stream_));
      ReportCUDAErrors(cudaStreamDestroy(download_stream_));
      ReportCUDAErrors(cudaEventDestroy(compute_ordering_event_));
      if (attn_body_done_event_)
        ReportCUDAErrors(cudaEventDestroy(attn_body_done_event_));
      if (value_cublas_) ReportCUBLASErrors(cublasDestroy(value_cublas_));
      if (value_stream_) ReportCUDAErrors(cudaStreamDestroy(value_stream_));
    }
  }

  const NetworkCapabilities& GetCapabilities() const override {
    return capabilities_;
  }

  int GetMiniBatchSize() const override {
    // Simple heuristic that seems to work for a wide range of GPUs.
    return 2 * sm_count_;
  }

  int GetPreferredBatchStep() const override {
    int preferred_split = 7;
    while (sm_count_ % preferred_split != 0) preferred_split++;
    return preferred_split;
  }

  int GetThreads() const override { return 1 + multi_stream_; }

  std::unique_ptr<NetworkComputation> NewComputation() override {
    // Set correct gpu id for this computation (as it might have been called
    // from a different thread).
    int device = -1;
    ReportCUDAErrors(cudaGetDevice(&device));
    if (device != gpu_id_) {
      ReportCUDAErrors(cudaSetDevice(gpu_id_));
    }
    return std::make_unique<CudaNetworkComputation<DataType>>(this, wdl_,
                                                              moves_left_);
  }

  std::unique_ptr<InputsOutputs<DataType>> GetInputsOutputs() {
    std::lock_guard<std::mutex> lock(inputs_outputs_lock_);
    if (free_inputs_outputs_.empty()) {
      // We need the host-side optimistic buffer when EITHER:
      //   (a) GPU blend is OFF — search reads p_optimistic on host
      //       and does the per-node blend math itself.
      //   (b) GPU blend is in SPLIT mode — gpu_blend_alpha_root_ > 0
      //       in addition to gpu_blend_alpha_, so two pre-blended
      //       buffers are produced and both need to reach the host.
      // In the uniform GPU blend case (gpu_blend_alpha_ > 0 alone),
      // the blended priors are written into the vanilla buffer and
      // the optimistic host buffer isn't needed → leaving it null
      // makes HasOptimisticPolicy() return false, eliminating the
      // second host softmax, memcache p_optimistic propagation, and
      // search's blend branch.
      const bool split_mode =
          gpu_blend_alpha_ > 0.0f && gpu_blend_alpha_root_ > 0.0f &&
          gpu_blend_alpha_ != gpu_blend_alpha_root_;
      const bool need_opt_host =
          has_optimistic_policy_ &&
          (gpu_blend_alpha_ == 0.0f || split_mode);
      return std::make_unique<InputsOutputs<DataType>>(
          max_batch_size_, wdl_, moves_left_, tensor_mem_size_, scratch_size_,
          !has_tensor_cores_ && std::is_same<half, DataType>::value,
          has_optimistic_policy_, need_opt_host);
    } else {
      std::unique_ptr<InputsOutputs<DataType>> resource =
          std::move(free_inputs_outputs_.front());
      free_inputs_outputs_.pop_front();
      return resource;
    }
  }

  void ReleaseInputsOutputs(std::unique_ptr<InputsOutputs<DataType>> resource) {
    std::lock_guard<std::mutex> lock(inputs_outputs_lock_);
    free_inputs_outputs_.push_back(std::move(resource));
  }

  // Apparently nvcc doesn't see constructor invocations through make_unique.
  // This function invokes constructor just to please complier and silence
  // warning. Is never called (but compiler thinks that it could).
  void UglyFunctionToSilenceNvccWarning() {
    InputsOutputs<DataType> io(0, false, false, 0, 0, false, false);
  }

 private:
  NetworkCapabilities capabilities_;
  int gpu_id_;
  int l2_cache_size_;
  int sm_count_;
  int max_batch_size_;
  int min_batch_size_;
  bool enable_graph_capture_;
  // GPU-side optimistic policy blend coefficients.  When gpu_blend_
  // alpha_ > 0 and the net has the optimistic head built, runs the
  // blend kernel after both policy maps to linearly blend logits.
  // When gpu_blend_alpha_root_ is also > 0 and differs, runs the
  // dual-output kernel for split-alpha (root vs internal).  Both
  // default 0 (disabled).  See ctor for the option reads.
  float gpu_blend_alpha_ = 0.0f;
  float gpu_blend_alpha_root_ = 0.0f;
  bool wdl_;
  bool moves_left_;
  bool use_res_block_winograd_fuse_opt_;  // fuse operations inside the residual
                                          // tower
  bool multi_stream_;                     // run multiple parallel network evals
  bool allow_cache_opt_;  // try to fit residual block activations in L2 cache

  // Currently only one NN Eval can happen a time (we can fix this if needed
  // by allocating more memory).
  mutable std::mutex lock_;

  int numBlocks_;
  int numFilters_;
  bool has_se_;
  bool conv_policy_;
  bool attn_policy_;
  bool attn_body_;
  int num_encoder_blocks_;

  // Optimistic policy head wiring.  When the proto carries a second
  // policy head named "optimistic" (i.e. policy_optimistic_st in the
  // PyTorch model), we build a parallel AttentionPolicyHead reading
  // from encoder_last_ and writing to op_policy_opt_mem_gpu_.  Search
  // blends it with the vanilla policy at root only when
  // --optimistic-policy-weight > 0.
  //
  bool has_optimistic_policy_ = false;

  // WeightArena owns the big-chunk cudaMallocs behind every weight
  // buffer in the attention body + encoder blocks (when activated).
  // CRITICAL: declared BEFORE `network_` so its destructor runs
  // AFTER all layers' destructors — layers' FreeWeight() helper
  // calls arena_->Owns() and requires the arena to still exist.
  // Reverse declaration order would cause use-after-free at shutdown.
  // Phase B.3 activates this; null when neither the LC0_USE_WEIGHT_ARENA
  // env var nor the `weight_arena=true` backend option is set (legacy
  // per-buffer cudaMalloc/cudaFree path).
  std::unique_ptr<WeightArena> weight_arena_;

  std::vector<std::unique_ptr<BaseLayer<DataType>>> network_;
  BaseLayer<DataType>* getLastLayer() { return network_.back().get(); }

  BaseLayer<DataType>* resi_last_;
  BaseLayer<DataType>* encoder_last_;
  AttentionBody<DataType>* attention_body_layer_ = nullptr;

  size_t tensor_mem_size_;
  size_t scratch_size_;

  // this copy is used only for initialization when multi-stream is enabled
  void* scratch_mem_;
  // this is only used when multi-stream is disabled
  void** offset_pointers_ = nullptr;
  void** head_offset_pointers_ = nullptr;

  bool has_tensor_cores_;

  // not used when multi-steam is enabled
  cudaStream_t compute_stream_ = nullptr;
  cudaStream_t upload_stream_ = nullptr;
  cudaStream_t download_stream_ = nullptr;
  cudaEvent_t compute_ordering_event_ = nullptr;
  // Signals attention-body completion to value_stream_.  Only valid when
  // !multi_stream_ and HasMultiStreamFFN().
  cudaEvent_t attn_body_done_event_ = nullptr;
  // Dedicated stream + cuBLAS for the value head (PEDANTIC_MATH, no tensor
  // cores). Receives exactly 1 cross-stream wait per inference (attn_body_done)
  // → always below the Ada Lovelace fork-join taint threshold that breaks m=3
  // WDL GEMMs.  Only valid when !multi_stream_ and HasMultiStreamFFN().
  cudaStream_t value_stream_ = nullptr;
  cublasHandle_t value_cublas_ = nullptr;
  cublasHandle_t cublas_;
  DataType* tensor_mem_[3];

  mutable std::mutex inputs_outputs_lock_;
  std::list<std::unique_ptr<InputsOutputs<DataType>>> free_inputs_outputs_;

  void showInfo() const {
    int version;
    int ret = cudaRuntimeGetVersion(&version);
    switch (ret) {
      case cudaErrorInitializationError:
        throw Exception("CUDA driver and/or runtime could not be initialized");
      case cudaErrorInsufficientDriver:
        throw Exception("No CUDA driver, or one older than the CUDA library");
      case cudaErrorNoDevice:
        throw Exception("No CUDA-capable devices detected");
    }
    int major = version / 1000;
    int minor = (version - major * 1000) / 10;
    int pl = version - major * 1000 - minor * 10;
    CERR << "CUDA Runtime version: " << major << "." << minor << "." << pl;
    if (version != CUDART_VERSION) {
      major = CUDART_VERSION / 1000;
      minor = (CUDART_VERSION - major * 1000) / 10;
      pl = CUDART_VERSION - major * 1000 - minor * 10;
      // After cuda 11, newer version with same major is OK.
      if (major < 11 || (major != version / 1000) || version < CUDART_VERSION) {
        CERR << "WARNING: CUDA Runtime version mismatch, was compiled with "
                "version "
             << major << "." << minor << "." << pl;
      }
    }
    cudaDriverGetVersion(&version);
    major = version / 1000;
    minor = (version - major * 1000) / 10;
    pl = version - major * 1000 - minor * 10;
    CERR << "Latest version of CUDA supported by the driver: " << major << "."
         << minor << "." << pl;
    if (version < CUDART_VERSION) {
      CERR << "WARNING: code was compiled with unsupported CUDA version.";
    }
  }

  void showDeviceInfo(const cudaDeviceProp& deviceProp,
                      [[maybe_unused]] int deviceId) const {
    CERR << "GPU: " << deviceProp.name;
    CERR << "GPU memory: " << deviceProp.totalGlobalMem / std::pow(2.0f, 30)
         << " Gb";
    // Get clock rate
    float clockRateMHz;
#if CUDART_VERSION >= 13000
    int clockRatekHz;
    cudaError_t err =
        cudaDeviceGetAttribute(&clockRatekHz, cudaDevAttrClockRate, deviceId);
    if (err != cudaSuccess) {
      CERR << "Error getting clock rate: " << cudaGetErrorString(err);
      clockRateMHz = 0.0f;  // Fallback value
    } else {
      clockRateMHz = clockRatekHz / 1e3f;
    }
#else
    clockRateMHz = deviceProp.clockRate / 1e3f;
#endif
    CERR << "GPU clock frequency: " << clockRateMHz << " MHz";
    CERR << "GPU compute capability: " << deviceProp.major << "."
         << deviceProp.minor;
    CERR << "L2 cache capacity: " << deviceProp.l2CacheSize;
    if (std::is_same<float, DataType>::value && deviceProp.major >= 7) {
      CERR << "WARNING: you will probably get better performance from the "
              "cuda-fp16 backend.";
    }
  }
};

template <typename DataType>
CudaNetworkComputation<DataType>::CudaNetworkComputation(
    CudaNetwork<DataType>* network, bool wdl, bool moves_left)
    : wdl_(wdl), moves_left_(moves_left), network_(network) {
  batch_size_ = 0;
  inputs_outputs_ = network_->GetInputsOutputs();
}

template <typename DataType>
CudaNetworkComputation<DataType>::~CudaNetworkComputation() {
  network_->ReleaseInputsOutputs(std::move(inputs_outputs_));
}

template <typename DataType>
void CudaNetworkComputation<DataType>::CaptureGraph(
    std::unique_lock<std::mutex>&& lock) {
  if (!network_->GetGraphCaptureEnabled()) return;
  if (!CudaGraphCapture<DataType>::EnsureEnoughFreeMemory()) {
    static std::once_flag flag;
    std::call_once(flag, []() {
      CERR << "WARNING: Not enough GPU memory to capture CUDA graphs.";
    });
    return;
  }
  // Diagnostic: track how often we re-capture for each batch size.  If a
  // captured graph is being reused correctly, this should fire exactly once
  // per unique batch size (times the number of distinct InputsOutputs in
  // the pool).  Enable with LC0_GRAPH_CAPTURE_LOG=1.
  static const bool kGraphCaptureLog =
      std::getenv("LC0_GRAPH_CAPTURE_LOG") != nullptr;
  if (kGraphCaptureLog) {
    static std::atomic<int> s_capture_count{0};
    int n = ++s_capture_count;
    fprintf(stderr,
            "[LC0_GRAPH_CAPTURE_LOG] capture #%d for batch=%d io=%p slot_was=%s\n",
            n, GetBatchSize(), (void*)inputs_outputs_.get(),
            inputs_outputs_->cuda_graphs_[GetBatchSize() - 1] ? "populated"
                                                              : "empty");
    fflush(stderr);
  }
  auto capture = network_->BeginCapture(*inputs_outputs_);
  network_->forwardEval(inputs_outputs_.get(), GetBatchSize(), true);
  capture.EndCapture();
  if (lock.owns_lock()) lock.unlock();
  inputs_outputs_->cuda_graphs_[GetBatchSize() - 1] = capture;
}

template <typename DataType>
void CudaNetworkComputation<DataType>::ComputeBlocking() {
  LCTRACE_FUNCTION_SCOPE;
  if (GetBatchSize() == 0) return;
  // Diagnostic bypass: LC0_DUMP_TENSORS=1 forces the non-graph path so the
  // host-side fprintf() statements inside AttentionBody::Eval / ValueHead::
  // Eval actually execute.  CUDA-graph replay only re-runs the recorded
  // kernel launches, not their enclosing host code, so the tensor dumps
  // would otherwise be invisible on real (post-warmup) inference.
  //
  // Also bypasses for LC0_NAN_SCAN (deprecated alias) and LC0_DEBUG
  // (debugDump family) so those dumps fire during actual eval too.
  static const bool kBypassGraph =
      std::getenv("LC0_DUMP_TENSORS") != nullptr ||
      std::getenv("LC0_NAN_SCAN") != nullptr ||
      std::getenv("LC0_DEBUG") != nullptr;
  if (!kBypassGraph && inputs_outputs_->cuda_graphs_[GetBatchSize() - 1]) {
    std::unique_lock<std::mutex> lock = network_->LockEval();
    network_->GraphLaunch(inputs_outputs_.get(), GetBatchSize());
  } else {
    std::unique_lock<std::mutex> lock = network_->LockEval();
#if !CUDA_GRAPH_SUPPORTS_EXTERNAL_EVENTS
    network_->UploadInputs(inputs_outputs_.get(), GetBatchSize());
#endif
    network_->forwardEval(inputs_outputs_.get(), GetBatchSize());
    if (!kBypassGraph) {
      // Skip graph capture during diagnostic runs so the next eval also
      // takes the non-graph path (otherwise one-shot behavior: first real
      // eval dumps, all subsequent evals silently replay the graph).
      CaptureGraph(std::move(lock));
    }
  }
  network_->finishEval(inputs_outputs_.get(), GetBatchSize());
}

template <typename DataType>
std::unique_ptr<Network> MakeCudaNetwork(const std::optional<WeightsFile>& w,
                                         const OptionsDict& options) {
  if (!w) {
    throw Exception(
        "The cuda" +
        std::string(std::is_same<half, DataType>::value ? "-fp16" : "") +
        " backend requires a network file.");
  }
  const WeightsFile& weights = *w;
  auto nf = weights.format().network_format();
  using NF = pblczero::NetworkFormat;
  switch (nf.network()) {
    case NF::NETWORK_CLASSICAL_WITH_HEADFORMAT:
    case NF::NETWORK_SE_WITH_HEADFORMAT:
    case NF::NETWORK_ATTENTIONBODY_WITH_HEADFORMAT:
    case NF::NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT:
      break;
    default:
      throw Exception("Network format " +
                      NF::NetworkStructure_Name(nf.network()) +
                      " is not supported by the CUDA backend.");
  }
  switch (nf.policy()) {
    case NF::POLICY_CLASSICAL:
    case NF::POLICY_CONVOLUTION:
    case NF::POLICY_ATTENTION:
      break;
    default:
      throw Exception("Policy format " + NF::PolicyFormat_Name(nf.policy()) +
                      " is not supported by the CUDA backend.");
  }
  switch (nf.value()) {
    case NF::VALUE_CLASSICAL:
    case NF::VALUE_WDL:
      break;
    default:
      throw Exception("Value format " + NF::ValueFormat_Name(nf.value()) +
                      " is not supported by the CUDA backend.");
  }
  switch (nf.moves_left()) {
    case NF::MOVES_LEFT_NONE:
    case NF::MOVES_LEFT_V1:
      break;
    default:
      throw Exception("Moves left head format " +
                      NF::MovesLeftFormat_Name(nf.moves_left()) +
                      " is not supported by the CUDA backend.");
  }
  switch (nf.default_activation()) {
    case NF::DEFAULT_ACTIVATION_RELU:
    case NF::DEFAULT_ACTIVATION_MISH:
    case NF::DEFAULT_ACTIVATION_SILU:
      break;
    default:
      throw Exception("Default activation " +
                      NF::DefaultActivation_Name(nf.default_activation()) +
                      " is not supported by the CUDA backend.");
  }
  switch (nf.input_embedding()) {
    case NF::INPUT_EMBEDDING_NONE:
    case NF::INPUT_EMBEDDING_PE_MAP:
    case NF::INPUT_EMBEDDING_PE_DENSE:
      break;
    default:
      throw Exception("Input embedding " +
                      NF::InputEmbeddingFormat_Name(nf.input_embedding()) +
                      " is not supported by the CUDA backend.");
  }
  return std::make_unique<CudaNetwork<DataType>>(weights, options);
}

std::unique_ptr<Network> MakeCudaNetworkAuto(
    const std::optional<WeightsFile>& weights, const OptionsDict& options) {
  int gpu_id = options.GetOrDefault<int>("gpu", 0);
  cudaDeviceProp deviceProp = {};
  // No error checking here, this will be repeated later.
  cudaGetDeviceProperties(&deviceProp, gpu_id);

  // Check if the GPU supports FP16.
  if (deviceProp.major >= 7 ||
      (deviceProp.major == 6 && deviceProp.minor != 1) ||
      (deviceProp.major == 5 && deviceProp.minor == 3)) {
    CERR << "Switching to [cuda-fp16]...";
    return MakeCudaNetwork<half>(weights, options);
  }
  CERR << "Switching to [cuda]...";
  return MakeCudaNetwork<float>(weights, options);
}

REGISTER_NETWORK("cuda-auto", MakeCudaNetworkAuto, 104)
REGISTER_NETWORK("cuda", MakeCudaNetwork<float>, 103)
REGISTER_NETWORK("cuda-fp16", MakeCudaNetwork<half>, 102)

}  // namespace lczero
