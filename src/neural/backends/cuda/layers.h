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
#pragma once

#include <cublas_v2.h>
#include <cstdint>

#include <cstddef>
#include <memory>
#include <vector>

#include "cuda_common.h"
#include "neural/network_legacy.h"
#include "neural/tables/activation_function.h"
#include "weight_arena.h"

#ifdef USE_CUDNN
#include <cudnn.h>
#else
typedef void* cudnnHandle_t;
#endif

namespace lczero {
namespace cudnn_backend {

// The Layer objects only hold memory for weights, biases, etc
// memory for input and output tensors is provided by caller of Eval.

template <typename DataType>
class BaseLayer {
 public:
  int GetC() const { return C; }
  int GetH() const { return H; }
  int GetW() const { return W; }
  bool isNHWC() const { return nhwc_; }

  BaseLayer(int c, int h, int w, BaseLayer* ip);
  BaseLayer(int c, int h, int w, BaseLayer* ip, bool nhwc);
  BaseLayer(int c, int h, int w, BaseLayer* ip, bool nhwc, bool use_gemm_ex);
  virtual ~BaseLayer() = default;
  size_t GetOutputSize(int N) const { return sizeof(DataType) * N * C * H * W; }

  // Input2 is optional (skip connection).
  virtual void Eval(int N, DataType* output, const DataType* input,
                    const DataType* input2, void* scratch, size_t scratch_size,
                    cudnnHandle_t cudnn, cublasHandle_t cublas,
                    cudaStream_t stream, DataType*** = nullptr) = 0;

 protected:
  BaseLayer* input_;

  int C;  // Output tensor dimensions.
  int H;
  int W;

  bool nhwc_;  // tensor layout
  const bool use_gemm_ex_;

  void cublasRowMajorMatrixMul(const DataType* A, const DataType* B,
                               DataType* Out, int M, int N, int K,
                               int batchSize, cublasHandle_t cublas);
};

#ifdef USE_CUDNN
template <typename DataType>
class ConvLayer : public BaseLayer<DataType> {
  using BaseLayer<DataType>::C;
  using BaseLayer<DataType>::H;
  using BaseLayer<DataType>::W;
  using BaseLayer<DataType>::GetC;
  using BaseLayer<DataType>::GetH;
  using BaseLayer<DataType>::GetW;
  using BaseLayer<DataType>::nhwc_;

 public:
  ConvLayer(BaseLayer<DataType>* ip, int C, int H, int W, int size, int Cin,
            ActivationFunction activation = ACTIVATION_NONE, bool bias = false);

  ConvLayer(bool nhwc, int C, int H, int W, int size, int Cin,
            ActivationFunction activation = ACTIVATION_NONE, bool bias = false);

  ~ConvLayer();
  void LoadWeights(float* pfilter, float* pBias, void* scratch);
  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas, cudaStream_t stream,
            DataType*** = nullptr) override;

 private:
  const int c_input_;
  const int filter_size_;
  const ActivationFunction act_;
  const bool use_bias_;

  DataType* biases = nullptr;
  DataType* weights = nullptr;

  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnConvolutionFwdAlgo_t conv_algo_;

  cudnnTensorDescriptor_t bias_desc_;
  cudnnTensorDescriptor_t in_tensor_desc_;
  cudnnTensorDescriptor_t out_tensor_desc_;
  cudnnActivationDescriptor_t activation_;

  void init();
};

#endif

template <typename DataType>
class FCLayer : public BaseLayer<DataType> {
  using BaseLayer<DataType>::nhwc_;

 public:
  FCLayer(BaseLayer<DataType>* ip, int C, int H, int W, bool bias,
          ActivationFunction activation);
  ~FCLayer();

  void LoadWeights(float* cpuWeight, float* cpuBias, void* scratch);
  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas, cudaStream_t stream,
            DataType*** = nullptr) override;

 private:
  const bool use_bias_;
  const ActivationFunction act_;
  DataType* weights_ = nullptr;
  DataType* biases_ = nullptr;
};

template <typename DataType>
class PolicyMapLayer : public BaseLayer<DataType> {
  using BaseLayer<DataType>::nhwc_;

 public:
  PolicyMapLayer(BaseLayer<DataType>* ip, int C, int H, int W, int usedSize,
                 bool attention);
  ~PolicyMapLayer();

  void LoadWeights(const short* cpuWeight, void* scratch);
  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas, cudaStream_t stream,
            DataType*** = nullptr) override;

 private:
  int used_size_;  // Size of the input without padding (typically 73x64).
                   // This is over-written to contain size with padding
                   // (typically 80x64) after CHW->HWC conversion for fp16.
  const bool attention_map_;
  short* weights_ = nullptr;
};

// Fused SE layer:
// (optional bias add +) global avg -> FC1 -> FC2 -> global scale -> add skip
// connection -> RELU.
template <typename DataType>
class SELayer : public BaseLayer<DataType> {
  using BaseLayer<DataType>::C;
  using BaseLayer<DataType>::nhwc_;

 public:
  SELayer(BaseLayer<DataType>* ip, int numFc1Out, bool addPrevLayerBias,
          ActivationFunction activation);
  ~SELayer();

  void LoadWeights(float* w1, float* b1, float* w2, float* b2,
                   float* prevLayerBias, void* scratch);

  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas, cudaStream_t stream,
            DataType*** = nullptr) override;

 private:
  DataType* w1_ = nullptr;
  DataType* w1_t_ = nullptr;  // transposed copy used by fused SE kernel
  DataType* b1_ = nullptr;
  DataType* w2_ = nullptr;
  DataType* w2_t_ = nullptr;
  DataType* b2_ = nullptr;
  DataType* bPrev_ = nullptr;
  int numFc1Out_;
  bool addPrevLayerBias_;
  const ActivationFunction act_;
};

// Multi-pass Winograd Conv fused with (optional) SE
template <typename DataType>
class FusedWinogradConvSELayer : public BaseLayer<DataType> {
  using BaseLayer<DataType>::C;
  using BaseLayer<DataType>::H;
  using BaseLayer<DataType>::W;
  using BaseLayer<DataType>::GetC;
  using BaseLayer<DataType>::GetH;
  using BaseLayer<DataType>::GetW;
  using BaseLayer<DataType>::nhwc_;

 public:
  FusedWinogradConvSELayer(BaseLayer<DataType>* ip, int C, int H, int W,
                           int Cin, ActivationFunction activation, bool bias,
                           bool skipAdd, bool se, int se_k, bool use_gemm_ex,
                           bool op_nhcw = false);

  ~FusedWinogradConvSELayer();
  void LoadWeights(float* pfilter, float* pBias, void* scratch);
  void LoadSEWeights(float* w1, float* b1, float* w2, float* b2, void* scratch);
  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas, cudaStream_t stream,
            DataType*** = nullptr) override;

 private:
  const int c_input_;
  const ActivationFunction act_;
  const bool use_bias_;
  const bool skip_add_;
  const bool has_se_;
  const int se_k_;
  const bool op_nhcw_;

  DataType* biases_ = nullptr;
  DataType* transformed_weights_ = nullptr;  // After winograd transform.

  // Weights and Biases for (optional) SE.
  DataType* w1_;
  DataType* w2_;
  DataType* b1_;
  DataType* b2_;
};

template <typename DataType>
class Conv1Layer : public BaseLayer<DataType> {
  using BaseLayer<DataType>::C;
  using BaseLayer<DataType>::H;
  using BaseLayer<DataType>::W;
  using BaseLayer<DataType>::GetC;
  using BaseLayer<DataType>::GetH;
  using BaseLayer<DataType>::GetW;
  using BaseLayer<DataType>::nhwc_;

 public:
  Conv1Layer(BaseLayer<DataType>* ip, int C, int H, int W, int Cin,
             ActivationFunction activation, bool bias, bool use_gemm_ex);

  ~Conv1Layer();
  void LoadWeights(float* pfilter, float* pBias, void* scratch);
  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas, cudaStream_t stream,
            DataType*** = nullptr) override;

 private:
  const int c_input_;
  const ActivationFunction act_;
  const bool use_bias_;

  DataType* biases_ = nullptr;
  DataType* weights_ = nullptr;

  // uses stride of 0 to read a vector as a matrix
  void cublasSpecialMatrixMul(const DataType* A, const DataType* B,
                              DataType* Out, int M, int N, int K, int batchSize,
                              cublasHandle_t cublas);
};

// Multi-pass Winograd Conv fused with (optional) SE
template <typename DataType>
class ResidualBlock : public BaseLayer<DataType> {
  using BaseLayer<DataType>::C;
  using BaseLayer<DataType>::H;
  using BaseLayer<DataType>::W;
  using BaseLayer<DataType>::GetC;
  using BaseLayer<DataType>::GetH;
  using BaseLayer<DataType>::GetW;

 public:
  ResidualBlock(BaseLayer<DataType>* ip, int C, bool se, int se_k,
                bool use_gemm_ex, bool first, bool last,
                ActivationFunction activation, int shared_mem_size);

  ~ResidualBlock();
  void LoadWeights0(float* pfilter, float* pBias, void* scratch);
  void LoadWeights1(float* pfilter, float* pBias, void* scratch);
  void LoadSEWeights(float* w1, float* b1, float* w2, float* b2, void* scratch);

  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas, cudaStream_t stream,
            DataType*** = nullptr) override;

 private:
  const bool has_se_;
  const int se_k_;
  const int c_input_;
  const bool first_block_;
  const bool last_block_;
  const int shared_mem_size_;
  const ActivationFunction act_;

  DataType* biases0_ = nullptr;
  DataType* biases1_ = nullptr;
  DataType* transformed_weights0_ = nullptr;  // After winograd transform.
  DataType* transformed_weights1_ = nullptr;  // After winograd transform.

  // Weights and Biases for (optional) SE.
  DataType* w1_;
  DataType* w2_;
  DataType* b1_;
  DataType* b2_;
};

template <typename DataType>
class EncoderBlock {
 public:
  EncoderBlock(const MultiHeadWeights::EncoderLayer& cpu_weights, void* scratch,
               int heads, int size, float alpha,
               DataType* smolgen_global_scratch, int smolgen_global_size,
               int max_batch_size, ActivationFunction smolgen_act,
               ActivationFunction ffn_act, float default_eps, bool use_gemm_ex,
               bool fused_mha,
               bool use_prenorm = false, bool use_rms_norm = false,
               bool use_swiglu = false,
               bool use_parallel_ffn = false,
               float attn_logit_cap = 0.0f,
               bool has_exoformer = false,
               float smolgen_softcap = 0.0f,
               float v_softcap = 0.0f,
               float swiglu_softcap = 0.0f,
               // GQA: number of KV heads (0 → full MHA).  When
               // kv_heads < heads, K/V projections are sized to
               // kv_dim = kv_heads*depth and reused across Q heads via
               // genOffsetPointers_GQA (no physical expand).
               int kv_heads = 0);
  ~EncoderBlock();

  void Eval(int N, DataType* inpop, DataType* scratch0, DataType* scratch1,
            DataType* scratch2, cublasHandle_t cublas, cudaStream_t stream,
            DataType*** offset_pointers,
            DataType* prev_attn_logits = nullptr,
            const DataType* exo_q_anc = nullptr,
            const DataType* exo_k_anc = nullptr,
            const DataType* exo_v_anc = nullptr,
            // Exo-as-smolgen-bias anchor, shape (N, H*gen_sz).  When
            // non-null, it is added to the layer's smolgen gen_from
            // output (post-LN, post-activation) before the shared
            // weight_gen decoder.  Computed once at the AttentionBody
            // level and shared across all encoder layers.
            const DataType* smolgen_anchor = nullptr,
            // Multi-stream FFN context (all null = disabled).
            cudaStream_t ffn_stream = nullptr,
            cublasHandle_t ffn_cublas = nullptr,
            cudaEvent_t ln1_done_event = nullptr,
            cudaEvent_t ffn_done_event = nullptr,
            DataType* ffn_buf_wide = nullptr,
            DataType* ffn_buf_out = nullptr,
            // Third-stream VGA-E context (all null = disabled).
            // VGA-E precompute GEMM runs concurrently with Q/K/V on main stream.
            cudaStream_t vga_stream = nullptr,
            cublasHandle_t vga_cublas = nullptr,
            cudaEvent_t vga_done_event = nullptr,
            // Third-stream Smolgen context (all null = disabled).
            // Runs the full smolgen pipeline concurrently with Q/K/V on main
            // stream; main waits for smol_done_event before softmax reads
            // smol_gen_out as the attention bias.
            cudaStream_t smolgen_stream = nullptr,
            cublasHandle_t smolgen_cublas = nullptr,
            cudaEvent_t smol_done_event = nullptr,
            DataType* smol_gen_out = nullptr,
            DataType* smol_interm = nullptr,
            DataType* smol_interm2 = nullptr,
            // Shared gate bank context (all null = no bank).
            //   bank_h_buf:   (max_batch*64, bank) — h, written on main.
            //   bank_lr_buf:  (max_batch*64, Σrank) — lr_a stage output;
            //                 the lr_b stage is folded into the consuming
            //                 kernels.
            //   bank_done_event: recorded on main once h + lr_a are ready;
            //                 the FFN stream waits on it before its
            //                 BankGatedMul.
            DataType* bank_h_buf = nullptr,
            DataType* bank_lr_buf = nullptr,
            cudaEvent_t bank_done_event = nullptr) const;

  // Weight arena (or nullptr).  Captured at construction time from
  // tl_weight_arena.  When non-null, all GPU weight buffers below
  // (mha_*, ffn_*, smol_*, etc.) are slices owned by the arena's
  // chunks — destructor MUST skip cudaFree on them (the arena's
  // own destructor will free its chunks).  WeightArena::Owns(ptr)
  // is the runtime check used in ~EncoderBlock.
  WeightArena* arena_ = nullptr;

  // all GPU side pointers
  DataType *mha_q_w, *mha_q_b;
  DataType *mha_k_w, *mha_k_b;
  DataType *mha_v_w, *mha_v_b;
  DataType *mha_qkv_w, *mha_qkv_b;
  DataType *mha_dense_w, *mha_dense_b;

  DataType *ln1_gammas, *ln1_betas;

  DataType *ffn_dense1_w, *ffn_dense1_b;
  DataType *ffn_dense2_w, *ffn_dense2_b;

  DataType *ln2_gammas, *ln2_betas;

  DataType *smol_compress;
  DataType *smol_dense1_w, *smol_dense1_b;
  DataType *smol_dense2_w, *smol_dense2_b;
  DataType *smol_ln1_gammas, *smol_ln1_betas;
  DataType *smol_ln2_gammas, *smol_ln2_betas;
  DataType *smol_global;

  int mha_q_size_;
  int mha_k_size_;
  int mha_v_size_;
  int mha_dense_size_;

  int ffn_dense1_size_;
  int ffn_dense2_size_;

  int embedding_op_size_;
  int encoder_heads_;
  // GQA: number of KV heads.  Equals encoder_heads_ for full MHA;
  // < encoder_heads_ means K/V are kv_dim-wide (= kv_heads_*depth) and
  // shared across Q-head groups via offset pointers.
  int kv_heads_;
  // Packed K+V weight buffer for GQA's projection (K then V, each
  // kv_dim wide).  When GQA is off, K/V come from mha_qkv_w_ or the
  // separate mha_k_w/mha_v_w pointers.
  DataType* mha_kv_w_ = nullptr;
  DataType* mha_kv_b_ = nullptr;
  // Weighted GQA (arXiv:2407.10855): learnable (heads, kv_heads) blend
  // matrices that replace the offset-pointer copy with a per-head linear
  // combination of the kv_heads slices.  Auto-detected at load time
  // from the presence of gqa_w_k/gqa_w_v weights in the proto.  Output
  // K/V become full d_model-wide; the standard genOffsetPointers path
  // is then used (not _GQA).
  bool has_weighted_gqa_ = false;
  DataType* gqa_w_k_ = nullptr;
  DataType* gqa_w_v_ = nullptr;

  float alpha_;  // scale to apply to skip connection add
  float default_eps_;  // value of epsilon where it wasn't specified in training

  const bool has_smolgen_;
  const ActivationFunction smolgen_activation_;
  const ActivationFunction ffn_activation_;

  // Output sizes for smolgen layers.
  int smol_compress_size_;
  int smol_dense_1_size_;
  int smol_dense_2_size_;
  int smol_global_size_;

  const int max_batch_size_;
  const bool use_fused_mha_;
  const bool use_gemm_ex_;

  // New feature flags
  bool is_prenorm_ = false;
  bool use_rms_norm_ = false;
  bool has_swiglu_ = false;
  bool is_parallel_ffn_ = false;
  // Attention logit soft-cap: tanh(logits/cap)*cap pre-softmax.
  // 0.0 disables; non-zero = cap value (Gemma 2 default is 50).
  float attn_logit_cap_ = 0.0f;
  // Smolgen output soft-cap: tanh(bias/cap)*cap on smolgen-generated
  // attention bias BEFORE adding to QK^T. Fused into softmax kernel.
  float smolgen_softcap_ = 0.0f;
  // V-projection soft-cap: tanh(V/cap)*cap on V after GLU-V (silu(gate)*up)
  // and PGB. Fused into the PGB add (or SwiGLU kernel when no PGB).
  float v_softcap_ = 0.0f;
  // SwiGLU FFN output soft-cap: tanh(out/cap)*cap on silu(gate)*up
  // before the down-projection. Fused into the SwiGLU kernel (no extra
  // launches). Matches PyTorch training-side tanh on the same value.
  float swiglu_softcap_ = 0.0f;

  // SwiGLU FFN weights
  DataType* ffn_gate_w_ = nullptr;
  DataType* ffn_gate_b_ = nullptr;
  DataType* ffn_up_w_ = nullptr;
  DataType* ffn_up_b_ = nullptr;
  DataType* ffn_down_w_ = nullptr;
  DataType* ffn_down_b_ = nullptr;
  DataType* ffn_pgb_ = nullptr;  // PGB on FFN hidden (before down_proj/dense2)
  // Fused gate+up weight/bias: [gate_w; up_w] of shape (2*dff, emb).
  // Saves one GEMM launch and one read of the input per SwiGLU FFN call.
  DataType* ffn_gate_up_w_ = nullptr;
  DataType* ffn_gate_up_b_ = nullptr;
  int ffn_dff_ = 0;

  // Pre-summed (mha_dense_b + ffn_final_b) for the Post-Norm parallel
  // FFN fused-LN path.  When non-null, NormLayer takes this as its
  // `bias` instead of mha_dense_b, and the separate
  // addBiasBatched(ffn_down_b_ / ffn_dense2_b) calls before NormLayer
  // are skipped.  Saves 1 launch per encoder layer on the FFN-bottleneck
  // stream.  Null when not in Post-Norm + parallel FFN mode, or when
  // either contributing bias is absent/wrong size.
  DataType* attn_ffn_combined_bias_ = nullptr;

  // Fused QKV projection weight: [Wq1 | Wk | Wv_gate | Wv_up] of shape
  // (emb, d_model + 3*kv_dim) col-major.  Replaces 3 separate GEMMs
  // (Wq @ x, Wk @ x, [Wv_gate;Wv_up] @ x) with a single wider GEMM in
  // the NLA-Q-only + GQA + GLU-V path.  Built only when all those
  // features are active and the constituent weights are present and
  // properly sized; null otherwise.
  DataType* mha_qkv_fused_w_ = nullptr;
  // Width of the fused-QKV output column: d_model + 3*kv_dim.  Stored
  // here so the Eval branch can compute the cublas ldb and the strided
  // kernel arguments without recomputing it from kv_heads_/d_model.
  int mha_qkv_fused_M_ = 0;

  // NLA (NonLinear Attention): second-layer Q/K projection
  bool has_nla_ = false;
  // NLA Q-only variant: K stays a plain linear projection (no wk2, no
  // silu on K). Auto-detected at load time from absent k2_w in the proto
  // (the PyTorch export omits wk2 when training with nla_q_only).
  bool nla_q_only_ = false;
  DataType* mha_q2_w_ = nullptr;
  DataType* mha_q2_b_ = nullptr;
  DataType* mha_k2_w_ = nullptr;
  DataType* mha_k2_b_ = nullptr;
  // Fused Q1+K1 weight: [W_q1; W_k1] of shape (2*d_model, emb).
  // Built at constructor time from mha_q_w and mha_k_w to enable a single
  // GEMM that computes both Q1 and K1 intermediate outputs simultaneously.
  DataType* mha_q1k1_w_ = nullptr;
  DataType* mha_q1k1_b_ = nullptr;  // [q1_bias; k1_bias] (2*d_model)
  // Fused V-gate + V-up weight: [W_v_gate; W_v_up] of shape (2*d_model, emb).
  // Built when has_glu_attn_ — lets a single 2*d_model-wide GEMM produce
  // both intermediates, processed by the existing SwiGLUFusedGateUp kernel
  // (which expects exactly this [gate; up] col-major layout).  Cuts the
  // V-side GEMM count from 2 to 1 per encoder block.  Nullptr if alloc
  // failed — runtime falls back to the 2-GEMM path.
  DataType* mha_vg_vu_w_ = nullptr;

  // GLU Attention (V): gated value projection
  // V = SiLU(Wv_gate(x)) * Wv_up(x)
  bool has_glu_attn_ = false;
  DataType* mha_v_gate_w_ = nullptr;
  DataType* mha_v_gate_b_ = nullptr;
  DataType* mha_v_up_w_ = nullptr;
  DataType* mha_v_up_b_ = nullptr;
  // PGB (Post-Gating Bias): bias after GLU on V
  DataType* pgb_v_ = nullptr;

  // VGA-E (Element-wise): per-dimension gate on attention output
  bool has_vga_elem_ = false;
  DataType* vga_elem_gate_w_ = nullptr;
  DataType* vga_elem_gate_b_ = nullptr;
  DataType* vga_gate_buf_ = nullptr;  // precomputed gate for pre-norm path

  // ── Shared gate bank (A-Layout-2) ──
  // One per-layer nonlinear basis h = silu(W_bank x + b) consumed by the
  // GLU-V / VGA-E / SwiGLU-FFN gates through diag + rank-r adapters.
  // Presence-detected from gate_bank_w in the proto; per-site presence
  // from each bank_*_diag.  The replaced projections (v_gate /
  // vga_elem_gate / gate_proj) are ABSENT in bank nets — has_glu_attn_ /
  // has_vga_elem_ are therefore also keyed off the bank fields, and the
  // weight-concat fast paths (mha_vg_vu_w_, ffn_gate_up_w_,
  // mha_qkv_fused_w_) stay null under the bank.  Post-norm parallel-FFN
  // only (matches the training-side constraint).
  bool has_gate_bank_ = false;
  bool has_bank_v_ = false;
  bool has_bank_vga_ = false;
  bool has_bank_ffn_ = false;
  // Layout-1: Q = Wq2(h) — the bank replaces NLA-Q's inner projection.
  // Detected from absent q_w with q2_w present (export writes no q_w).
  bool bank_include_q_ = false;
  int gate_bank_size_ = 0;
  int bank_rank_v_ = 0;
  int bank_rank_vga_ = 0;
  int bank_rank_ffn_ = 0;
  DataType* gate_bank_w_ = nullptr;
  DataType* gate_bank_b_ = nullptr;
  DataType* bank_lr_a_w_ = nullptr;  // concat [v; vga; ffn] rows, (Σr, bank)
  DataType* bank_v_diag_ = nullptr;
  DataType* bank_v_bias_ = nullptr;
  DataType* bank_v_lr_b_w_ = nullptr;
  DataType* bank_vga_diag_ = nullptr;
  DataType* bank_vga_bias_ = nullptr;
  DataType* bank_vga_lr_b_w_ = nullptr;
  DataType* bank_ffn_diag_ = nullptr;
  DataType* bank_ffn_bias_ = nullptr;
  DataType* bank_ffn_lr_b_w_ = nullptr;

  // Persistent LN1 output buffer — computes LN1 once per Eval, reused by
  // SmolGen, Q/K/V, and (for parallel FFN) the FFN. Avoids the 2-3 LN1
  // recomputes that the old code did because SmolGen/dense clobber buffer1.
  DataType* ln1_cache_ = nullptr;

  // ExoFormer: per-layer anchor blending lambda (2 scalars, stored on host)
  float exo_lambda_host_[2] = {0.0f, 1.0f};

  void EvalFFNOnly(int N, cudaStream_t ffn_s, cublasHandle_t ffn_h,
                   DataType* ffn_buf_wide, DataType* ffn_buf_out) const;
};

// The Attention policy head implementation
// Responsible for loading weights into GPU memory, and evaluating the entire
// policy head
template <typename DataType>
class AttentionPolicyHead : public BaseLayer<DataType> {
  using BaseLayer<DataType>::C;
  using BaseLayer<DataType>::H;
  using BaseLayer<DataType>::W;
  using BaseLayer<DataType>::GetC;
  using BaseLayer<DataType>::GetH;
  using BaseLayer<DataType>::GetW;
  using BaseLayer<DataType>::use_gemm_ex_;

 public:
  AttentionPolicyHead(BaseLayer<DataType>* ip,
                      const MultiHeadWeights::PolicyHead& weights,
                      void* scratch, bool attention_body,
                      ActivationFunction act, int max_batch_size,
                      bool use_gemm_ex);
  ~AttentionPolicyHead();
  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas, cudaStream_t stream,
            DataType*** = nullptr) override;

 private:
  // GPU allocations to hold various weights used by the attention policy head
  DataType *ip_pol_w_, *ip_pol_b_;    // "embedding" in policy attention
  DataType *ip2_pol_w_, *ip2_pol_b_;  // "wq" in policy attention
  DataType *ip3_pol_w_, *ip3_pol_b_;  // "wk" in policy attention
  DataType *ip4_pol_w_;               // "ppo" in policy attention

  DataType *wqk_w_, *wqk_b_;  // allocation containing both "wq" and "wq"

  int embedding_op_size_;
  int wq_op_size_;
  int wk_op_size_;

  int encoder_heads_;
  int policy_d_model_;
  bool attention_body_;
  ActivationFunction act_;

  std::vector<EncoderBlock<DataType>*> encoder_weights_;
};

template <typename DataType>
class EmbeddingLayer : public BaseLayer<DataType> {
  using BaseLayer<DataType>::C;
  using BaseLayer<DataType>::H;
  using BaseLayer<DataType>::W;

 public:
  EmbeddingLayer(BaseLayer<DataType>* ip, const std::vector<float>& weights,
                 const std::vector<float>& biases, void* scratch,
                 ActivationFunction activation);
  ~EmbeddingLayer();

  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas, cudaStream_t stream,
            DataType*** = nullptr) override;

 private:
  DataType *weights_, *biases_;
  ActivationFunction act_;
};

// The Attention body implementation
// Responsible for loading weights into GPU memory, and evaluating the entire
// attention network part of the body including the stack of encoder layers
template <typename DataType>
class AttentionBody : public BaseLayer<DataType> {
  using BaseLayer<DataType>::C;
  using BaseLayer<DataType>::H;
  using BaseLayer<DataType>::W;
  using BaseLayer<DataType>::GetC;
  using BaseLayer<DataType>::GetH;
  using BaseLayer<DataType>::GetW;
  // Bring use_gemm_ex_ into the template scope so member functions can
  // reference it unqualified (needed by cublasXGemmStridedBatched in the
  // exo-as-smolgen-bias pooling path at Eval time).
  using BaseLayer<DataType>::use_gemm_ex_;

 public:
  AttentionBody(const MultiHeadWeights& weights, void* scratch,
                Activations activations, int num_res_blocks, int input_c,
                int max_batch_size, bool is_pe_dense_embedding,
                bool use_gemm_ex, bool fused_mha);
  ~AttentionBody();
  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas, cudaStream_t stream,
            DataType*** = nullptr) override;

  // Returns true when the parallel FFN path is active (ffn_stream_ running
  // FFN concurrently with attention on compute_stream).
  bool HasMultiStreamFFN() const { return multi_stream_ffn_; }

  // Returns true when the model uses parallel (PaLM-style) FFN regardless of
  // whether the multi-stream FFN infrastructure is active.  Used by CudaNetwork
  // to determine whether the cuBLAS value-head bypass is needed: even when the
  // FFN runs sequentially on compute_stream (multi_stream_ffn_=false), the
  // volume of large-m tensor-core GEMMs primes cuBLAS into tensor-core-only
  // kernel selection, which fails for the m=3 WDL GEMM.
  bool IsParallelFFN() const { return is_parallel_ffn_; }

  // Accessors for the parallel FFN stream and cuBLAS handle.
  cudaStream_t GetFFNStream() const { return ffn_stream_; }
  cublasHandle_t GetFFNCublas() const { return ffn_cublas_; }

  // Dual-graph support: capture and launch the FFN graph separately from
  // the main CUDA graph. Called by CudaNetwork after main graph capture.
  void CaptureFFNGraph(int N);
  void LaunchFFNGraph(int N);

  // Sibling-fork: make ffn_stream_ a first-level fork of the CUDA graph
  // capture root by waiting on the same upload-done event as compute_stream_.
  // Called in forwardEval right after compute_stream_ waits on upload_done_event_.
  // Makes cuBLAS work inside the captured graph (cuBLAS requires first-level forks).
  void JoinFFNStream(cudaEvent_t ev);

 private:
  // Weight arena (or nullptr) captured at construction.  When non-null,
  // every weight buffer below is a slice owned by the arena's chunks
  // and must NOT be cudaFree'd individually in ~AttentionBody.  The
  // FreeWeight() helper in layers.cc routes through WeightArena::Owns()
  // to do the right thing.  Note: child EncoderBlocks have their own
  // arena_ captured from tl_weight_arena at THEIR construction time
  // (which is during this ctor under the same scope) — they manage
  // their own buffers' arena-awareness independently.
  WeightArena* arena_ = nullptr;

  // GPU allocations to hold various weights used by the attention net body.
  DataType *ip_emb_pre_w_, *ip_emb_pre_b_;  // input position preprocessing weights.
  DataType *ip_emb_w_, *ip_emb_b_;          // "embedding" layer in net body
  DataType *ip_emb_ln_g_, *ip_emb_ln_b_;  // input embedding layernorm gamma and beta
  DataType *ip_mult_gate_, *ip_add_gate_;   // input gating
  DataType *ip_emb_ffn_d1_w_, *ip_emb_ffn_d1_b_;  // input embedding FFN dense1 weights
  DataType *ip_emb_ffn_d2_w_, *ip_emb_ffn_d2_b_;  // input embedding FFN dense2 weights
  DataType *ip_emb_ffn_ln_g_, *ip_emb_ffn_ln_b_;  // input embedding FFN layernorm gamma and beta

  // Embedding FFN SwiGLU weights
  DataType *ip_emb_ffn_gate_w_ = nullptr, *ip_emb_ffn_gate_b_ = nullptr;
  DataType *ip_emb_ffn_up_w_ = nullptr, *ip_emb_ffn_up_b_ = nullptr;
  DataType *ip_emb_ffn_down_w_ = nullptr, *ip_emb_ffn_down_b_ = nullptr;
  // Fused gate+up: [gate_w; up_w] of shape (2*dff, emb).
  DataType *ip_emb_ffn_gate_up_w_ = nullptr, *ip_emb_ffn_gate_up_b_ = nullptr;
  bool has_emb_ffn_swiglu_ = false;
  DataType *smolgen_global_;  // global smolgen weights for all encoder layers
  DataType *pos_encoding_ = nullptr;
  int embedding_dense_size_;
  int embedding_op_size_;
  int embedding_ffn_size_;
  int embedding_ffn_dff_;
  int encoder_head_count_;
  std::vector<EncoderBlock<DataType>*> encoder_weights_;
  Activations activations_;
  int num_resi_blocks_;
  int input_c_;
  int smolgen_global_size_;
  const bool has_gating_;
  const bool has_smolgen_;
  bool is_pe_dense_embedding_;  // flag for dense position encoding
  const bool use_fused_mha_;

  // Light embedding: per-square MLP + global summary

  // Rich embedding: widened per-square MLP (piece + per-square extras).
  bool has_rich_emb_ = false;
  int rich_per_sq_input_sz_ = 0;  // 12 + 6*has_atk + 2*has_mat
  int rich_hidden_sz_ = 0;
  int rich_emb_global_sz_ = 0;

  DataType *rich_emb_sq_w1_ = nullptr, *rich_emb_sq_b1_ = nullptr;
  DataType *rich_emb_sq_w2_ = nullptr, *rich_emb_sq_b2_ = nullptr;
  DataType *rich_emb_global_w_ = nullptr, *rich_emb_global_b_ = nullptr;
  DataType *rich_per_sq_buf_ = nullptr;  // (max_batch, 64, rich_per_sq_input_sz_)

  // ExoFormer: anchor projections (computed once from embedding output)
  bool has_exoformer_ = false;
  DataType *exo_q_anc_w_ = nullptr;
  DataType *exo_k_anc_w_ = nullptr;
  DataType *exo_v_anc_w_ = nullptr;
  // Single combined allocation backing exo_q/k/v_anc_buf_ contiguously.
  // exo_q = base, exo_k = base + max_batch*d, exo_v = base + 2*max_batch*d.
  DataType *exo_all_anc_buf_ = nullptr;
  // Aliases into exo_all_anc_buf_ for each anchor output
  DataType *exo_q_anc_buf_ = nullptr;
  DataType *exo_k_anc_buf_ = nullptr;
  DataType *exo_v_anc_buf_ = nullptr;
  DataType *exo_norm_ones_ = nullptr;  // ones buffer for RMSNorm (scale=False)

  // Exo-as-smolgen-bias: single projection from pooled initial flow to a
  // (H * gen_sz) anchor code.  Computed once per forward in AttentionBody::Eval
  // and added to every encoder layer's smolgen gen_from before the shared
  // weight_gen decoder.  The training-side 1/sqrt(N_layers) forward scaling
  // is baked into the exported weight — CUDA applies the projection as-is.
  bool has_exo_smol_anchor_ = false;
  DataType *exo_smol_anchor_w_ = nullptr;    // (H*gen_sz, emb_size)
  DataType *exo_smol_anchor_buf_ = nullptr;  // (max_batch, H*gen_sz)
  DataType *exo_smol_pool_buf_ = nullptr;    // (max_batch, emb_size) reused pool
  DataType *exo_smol_ones_64_ = nullptr;     // length-64 ones vector for pooling

  // New format flags
  bool is_prenorm_ = false;
  bool use_rms_norm_ = false;
  bool has_swiglu_ = false;
  bool is_parallel_ffn_ = false;
  // Attention logit soft-cap (Gemma 2): 0 disables, >0 = cap value.
  // Threaded into every EncoderBlock at construction.
  float attn_logit_cap_ = 0.0f;
  // Smolgen output soft-cap (per-head per-position attention bias) and
  // V-projection soft-cap. 0 disables, >0 = cap value. Same plumbing
  // pattern as attn_logit_cap_: read from proto in AttentionBody ctor,
  // forwarded to every EncoderBlock at construction.
  float smolgen_softcap_ = 0.0f;
  float v_softcap_ = 0.0f;
  // SwiGLU FFN output soft-cap (applies to encoder FFN and embedding FFN).
  // Forwarded to every EncoderBlock at construction.
  float swiglu_softcap_ = 0.0f;

 public:
  // Multi-exit: value heads read encoder flow at this layer instead of
  // the final encoder output. -1 disables. When set:
  //   - During Eval, after the encoder block at this index runs, the
  //     output_tensor's contents are copied to value_branch_buf_, then
  //     final_norm is applied to it (mirroring the main flow's final_norm
  //     pass) so consumers see a fully-prepped flow.
  //   - GetValueBranchOutput() returns value_branch_buf_ for downstream
  //     ValueHead's input override (set up at network construction).
  int value_branch_at_layer_ = -1;
  DataType* GetValueBranchOutput() const { return value_branch_buf_; }
  bool HasValueBranch() const {
    return value_branch_at_layer_ >= 0 && value_branch_buf_ != nullptr;
  }

 private:
  DataType* value_branch_buf_ = nullptr;  // (max_batch, 64+R, embedding)

  // Material info: 18 derived chess features (piece counts, diffs, masks)
  // computed from input planes and concatenated to embedding input.
  bool has_material_info_ = false;
  DataType* material_features_buf_ = nullptr;  // (max_batch, 64, 18)

  // Attack maps: 6 tactical features per square (own/opp attacked, attack
  // counts, own_defended, opp_hanging) with ray blocking.
  bool has_attack_maps_ = false;
  DataType* attack_map_buf_ = nullptr;  // (max_batch, 64, 6)

  // Multi-stream FFN overlap. Auto-enabled when model uses parallel FFN +
  // SwiGLU. Runs FFN on a second CUDA stream concurrent with attention for
  // ~10-15% throughput improvement at batch 128.
  bool multi_stream_ffn_ = false;
  cudaStream_t ffn_stream_ = nullptr;
  cublasHandle_t ffn_cublas_ = nullptr;
  // Per-layer events for dual-graph synchronization (supports up to kMaxEncoderLayers layers).
  // External flags allow cross-graph event signaling without isolation errors.
  static constexpr int kMaxEncoderLayers = 64;
  cudaEvent_t ln1_done_events_[kMaxEncoderLayers] = {};
  cudaEvent_t ffn_done_events_[kMaxEncoderLayers] = {};
  int num_encoder_layers_ = 0;
  // Dedicated FFN temp buffers (shared across all encoder layers).
  DataType* ffn_buf_wide_ = nullptr;   // (batch, 2*dff) for fused gate+up GEMM
  DataType* ffn_buf_out_ = nullptr;    // (batch, emb)   for FFN final output

  // Shared gate bank (A-Layout-2) buffers — dedicated allocations because
  // h outlives both streams' reads within a layer (main: V/VGA adapters;
  // FFN stream: FFN gate adapter).  Sized from encoder[0]'s bank dims.
  // bank_done_events_ are per-layer for the same cross-graph reasons as
  // ln1_done_events_/ffn_done_events_.
  bool has_gate_bank_ = false;
  DataType* bank_h_buf_ = nullptr;
  DataType* bank_lr_buf_ = nullptr;
  cudaEvent_t bank_done_events_[kMaxEncoderLayers] = {};

  // Third-stream VGA-E overlap. Runs VGA-E precompute GEMM concurrently with
  // Q/K/V projections on the main stream. Enabled when multi_stream_ffn_ is
  // active and the model has VGA-E (shares ln1_done_events_[i] as the start gate).
  cudaStream_t vga_stream_ = nullptr;
  cublasHandle_t vga_cublas_ = nullptr;
  cudaEvent_t vga_done_event_ = nullptr;

  // Third-stream Smolgen overlap. Runs the full smolgen pipeline (compress →
  // dense1+LN → dense2+LN → exo-bias add → weight-gen GEMM) on a dedicated
  // stream, concurrent with Q/K/V projections on the main stream.  Gated
  // on has_smolgen_ && multi_stream_ffn_ at construction.
  //   - Main stream records ln1_done_events_[i] before layer i's Eval starts
  //     modifying scratch (reused as smolgen's start gate).
  //   - smolgen_stream_ runs smolgen into smol_gen_out_ (dedicated, not buffer2).
  //   - Main stream waits smol_done_events_[i] before softmax consumes the bias.
  //   - Softmax / fusedMHA read smolgen bias from smol_gen_out_ instead of
  //     buffer2 when multi_stream_smolgen_ is active.
  bool multi_stream_smolgen_ = false;
  cudaStream_t smolgen_stream_ = nullptr;
  cublasHandle_t smolgen_cublas_ = nullptr;
  cudaEvent_t smol_done_events_[kMaxEncoderLayers] = {};
  // Single shared gen output buffer — one is enough because the event chain
  // (smol_done_events_[i] → main softmax[i] → ln1_done_events_[i+1] →
  // smolgen_stream wait[i+1]) forces smolgen_stream to block before
  // overwriting the buffer that main stream is still reading at layer i.
  DataType* smol_gen_out_ = nullptr;
  // Shared intermediate scratch for smolgen (compress output + LN outputs
  // + dense hidden activations).  Reused every layer — layer i's smolgen
  // finishes using this before layer i+1's starts (serial on smolgen_stream_).
  DataType* smol_intermediate_ = nullptr;
  DataType* smol_intermediate2_ = nullptr;

  // FFN graph: one exec per batch size, captured separately from main graph.
  // Indexed by batchSize - 1. Null until captured.
  std::vector<cudaGraphExec_t> ffn_graph_execs_;

  // Encoder final norm (Pre-Norm only)
  DataType* enc_final_norm_g_ = nullptr;
  DataType* enc_final_norm_b_ = nullptr;

  int max_batch_size_ = 0;
  // Actual N used in the most recent Eval call (may be > logical batch size
  // due to min_batch_size_ padding in forwardEval). Used by CaptureFFNGraph
  // to ensure the FFN graph's GEMM dimensions match the main graph.
  int last_eval_n_ = 0;
};

// The value head implementation
// Responsible for loading weights into GPU memory, and evaluating the value
// head and value error head
template <typename DataType>
class ValueHead : public BaseLayer<DataType> {
  using BaseLayer<DataType>::C;
  using BaseLayer<DataType>::H;
  using BaseLayer<DataType>::W;
  using BaseLayer<DataType>::GetC;
  using BaseLayer<DataType>::GetH;
  using BaseLayer<DataType>::GetW;

 public:
  ValueHead(BaseLayer<DataType>* ip, const MultiHeadWeights::ValueHead& weights,
            void* scratch, bool attention_body, bool wdl,
            ActivationFunction act, int max_batch_size, bool use_gemm_ex);
  ~ValueHead();
  void Eval(int N, DataType* output, const DataType* input,
            const DataType* input2, void* scratch, size_t scratch_size,
            cudnnHandle_t cudnn, cublasHandle_t cublas, cudaStream_t stream,
            DataType*** = nullptr) override;

  // Multi-exit override: when set (via SetInputOverride), Eval reads from
  // this pointer instead of its `input` argument. Used for multi-exit
  // architectures where value heads consume an intermediate encoder layer's
  // flow rather than the final encoder output. Set once at network
  // construction (after AttentionBody is built and its branch buffer is
  // allocated). nullptr disables override (default behavior).
  void SetInputOverride(const DataType* override_ptr) {
    input_override_ = override_ptr;
  }

 private:
  const DataType* input_override_ = nullptr;
  // "convolution" in value head (legacy)
  std::unique_ptr<Conv1Layer<DataType>> conv_;

  // GPU allocations to hold various weights used by the attention policy head
  DataType *ip_val_w_, *ip_val_b_;          // "embedding" in value head
  DataType *ip1_val_w_, *ip1_val_b_;        // "FC1" in value head
  DataType *ip2_val_w_, *ip2_val_b_;        // "FC2" in value head
  DataType *ip_val_err_w_, *ip_val_err_b_;  // value error "FC" weights

  int embedding_size_;
  int value_hidden_size_;
  bool wdl_;
  bool attention_body_;
  ActivationFunction act_;

  // SimPool attentive pooling
  bool has_simpool_ = false;
  DataType* simpool_query_ = nullptr;  // (1, val_emb_size)
  DataType* simpool_key_w_ = nullptr;  // (val_emb_size, val_emb_size)
  DataType* simpool_key_b_ = nullptr;  // (val_emb_size,)
};

}  // namespace cudnn_backend
}  // namespace lczero
