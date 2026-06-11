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

#include "cuda_common.h"
#include "neural/tables/activation_function.h"

#include <cstdint>

namespace lczero {
namespace cudnn_backend {

// Adds two vectors (possibly of different sizes), also do optional
// activation (relu, tanh or sigmoid).
template <typename T>
void addVectors(T* c, T* a, T* b, int size, int asize, int bsize,
                ActivationFunction activation, cudaStream_t stream);

// Adds two vectors of equal size overwriting the first with the sum.
// This specialisation performs a transposition of the first 2 indexes
// of the second while performing the addition.
template <typename T>
void addVectorsHNC_NHC(T* a, T* b, int N, int H, int C, cudaStream_t stream);

// Optimized kernel to add bias to innermost dimension
// and perform optional activation (to be used with GEMMs/fully connected)
template <typename T>
void addBiasBatched(T* output, const T* input, const T* bias, int Batch, int N,
                    int C, ActivationFunction activation, cudaStream_t stream);

// Optimized kernel to add bias to innermost dimension
// and perform optional activation (to be used with GEMMs/fully connected)
template <typename T>
void addBiasBatched(T* output, const T* input, const T* bias, int Batch, int N,
                    int C, int Nstride, ActivationFunction activation,
                    cudaStream_t stream);

// Fused bias-add + tanh soft-cap. Used for V-projection finalization in
// GLU-V attention paths (PyTorch order: silu(gate)*up → +pgb_v → softcap).
//   total : total number of elements in input/output (e.g. N*64*d_model)
//   C     : bias-broadcast period (i.e. d_model). Bias is length-C broadcast
//           across `total / C` rows. Pass nullptr for bias to skip the add
//           (cap-only mode, used when PGB is off but v_softcap > 0).
//   softcap : tanh cap value; <=0 means no cap (kernel still copies, so
//             prefer addBiasBatched in that case to avoid the write).
template <typename T>
void addBiasAndSoftCap(T* output, const T* input, const T* bias, int total,
                       int C, float softcap, cudaStream_t stream);

// Add bias to convolution's output.
template <typename T>
void addBias_NCHW(T* c, T* a, T* b, int N, int C, int H, int W,
                  ActivationFunction activation, cudaStream_t stream);

// Conversion from NCHW to NHWC, can also change datatype depending on template
// params, also pad/un-pad elements from Batch or Channel dimensions
template <typename DstType, typename SrcType>
void convertNCHWtoNHWC(DstType* output_tensor, const SrcType* input_tensor,
                       int Nin, int Cin, int Nout, int Cout, int H, int W,
                       cudaStream_t stream);

// Plain data-type conversion (no layout conversion).
template <typename DstType, typename SrcType>
void copyTypeConverted(DstType* op, SrcType* ip, int N, cudaStream_t stream);

// Perform batch normilization.
template <typename T>
void batchNorm(T* output, const T* input, const T* skipInput, int N, int C,
               int H, int W, float* means, float* var_multipliers,
               ActivationFunction activation, cudaStream_t stream);

// Unpack planes (input to network).
template <typename T>
void expandPlanes_NHWC(T* output, const uint64_t* masks, const T* values, int n,
                       cudaStream_t stream);

template <typename T>
void expandPlanes_NCHW(T* output, const uint64_t* masks, const T* values, int n,
                       cudaStream_t stream);

// Perform global avg pool.
template <typename T>
void globalAvgPool(int N, int C, T* output, const T* input,
                   const T* prevLayerBias, bool nhwc, cudaStream_t steam);

// Perform global scale.
template <typename T>
void globalScale(int N, int C, T* output, const T* input, const T* scaleBias,
                 const T* prevLayerBias, bool nhwc,
                 ActivationFunction activation, cudaStream_t steam);

// Perform Squeeze-and-Excitation (SE) in a single fused kernel.
// Returns false if the fused kernel can't handle the sizes.
bool Se_Fp16_NHWC(int N, int C, int numFc1Out, half* output, const half* skip,
                  const half* input, const half* w1, const half* b1,
                  const half* w2, const half* b2, const half* bPrev,
                  ActivationFunction activation, cudaStream_t stream);

template <typename T>
void PolicyMap(int N, T* output, const T* input, const short* indices,
               int inputSize, int usedSize, int outputSize,
               cudaStream_t stream);

// Custom winograd helper functions
template <typename T>
void FilterTransform(int N, int C, T* transformedFilter, const T* filter,
                     cudaStream_t stream);

template <typename T, bool nhcw>
void InputTransform(int N, int C, T* transformedInput, const T* input,
                    cudaStream_t stream);

template <typename T, bool use_se, ActivationFunction activation, bool use_bias,
          bool use_skip, bool skipInput_nhcw, bool output_nhcw>
void OutputTransform(int N, int C, int se_K, T* output, const T* input,
                     const T* skip, const T* bias, const T* w1, const T* b1,
                     const T* w2, const T* b2, cudaStream_t stream);

template <typename T, bool use_se, ActivationFunction activation, bool use_bias,
          bool use_skip>
void OutputInputTransform(int N, int C, int se_K, T* output, const T* input,
                          const T* skip, const T* bias, const T* w1,
                          const T* b1, const T* w2, const T* b2,
                          cudaStream_t stream);

// uv/uv_ld/uv_rank: optional smolgen-dictionary rank-r residual, fused
// into the C==64 path — bias += U_b[i,:]·V_b[j,:] per logit, added to the
// raw map BEFORE smolgen_cap (torch caps alpha·P + UVᵀ as a whole).  uv
// points at the U block of the dict decoder output (column-major, one
// column per (batch, head) map, ld = uv_ld; V block at +64*uv_rank).
template <typename T>
void Softmax(int N, int C, T* output, const T* input, const T* input2,
             cudaStream_t stream, float softcap = 0.0f,
             float smolgen_cap = 0.0f, const T* uv = nullptr, int uv_ld = 0,
             int uv_rank = 0);

template <typename T>
void LayerNorm(int N, int C, T* output, const T* input, const T* bias,
               const T* skip, const T* gammas, const T* betas, float ep,
               float alpha, ActivationFunction act, cudaStream_t stream,
               const T* input2 = nullptr, const T* out_add = nullptr);

template <typename T>
void ComputePromotionLogits(int N, int C, T* output, const T* keys,
                            const T* ppo, const T* policy_attn_logits,
                            cudaStream_t stream);

template <typename T>
void inputPreprocessForAttentionBody(T* output, const T* input,
                                     const T* encoding, int N, int input_size,
                                     int encoding_size,
                                     bool is_pe_dense_embedding,
                                     cudaStream_t stream);

// Compute 18 material features from (N, 112, 8, 8) NCHW input planes.
// Output: (N, 64, 18) NHWC. See common_kernels.cu for feature definitions.
template <typename T>
void computeMaterialFeatures(int N, T* output, const T* input,
                             cudaStream_t stream);

// Extended preprocess: [input(NCHW), material(NHWC), encoding] → NHWC output.
template <typename T>
void inputPreprocessForAttentionBodyWithMaterial(
    T* output, const T* input, const T* material, const T* encoding,
    int N, int input_size, int material_size, int encoding_size,
    bool is_pe_dense_embedding, cudaStream_t stream);

// Build per-square input for rich embedding MLP: concatenates the 12 piece
// planes (from NCHW input), the 6 attack-map features (NHWC, optional), and
// the 2 per-square material_info slots (indices 15/16 of 18-dim NHWC buf,
// optional) into an NHWC (N, 64, total_sz) output buffer. `attack` / `material`
// may be null if their respective features are disabled (sizes 0).
template <typename T>
void buildRichPerSquareInput(
    T* output, const T* input_nchw,
    const T* attack_nhwc, int attack_size,
    const T* material_nhwc, int material_size,
    int N, int total_sz, cudaStream_t stream);

// Compute 6 attack-map features from (N, 112, 8, 8) NCHW input planes.
// Output: (N, 64, 6) NHWC. Features per square (see common_kernels.cu):
//   [0] own_attacked  [1] opp_attacked  [2] own_attack_count/6
//   [3] opp_attack_count/6  [4] own_defended  [5] opp_hanging
template <typename T>
void computeAttackMaps(int N, T* output, const T* input, cudaStream_t stream);

// Extended preprocess: [input(NCHW), extra1(NHWC), extra2(NHWC), encoding] → NHWC.
// Used when both material_info and attack_maps are active (extra1=material 18,
// extra2=attack_maps 6).
template <typename T>
void inputPreprocessForAttentionBodyWithTwoExtras(
    T* output, const T* input,
    const T* extra1, int extra1_size,
    const T* extra2, int extra2_size,
    const T* encoding, int N, int input_size, int encoding_size,
    bool is_pe_dense_embedding, cudaStream_t stream);

template <typename T>
void applyInputGating(T* output, const T* input, const T* mult, const T* add,
                      int N, int HW, int C, cudaStream_t stream);

template <typename T>
void genOffsetPointers(T** offsets, int heads, int max_batch, int depth,
                       int d_model, T* k, T* q, T* b1, T* v, T* b2,
                       cudaStream_t stream);

// GQA variant of genOffsetPointers — each Q head h aliases the shared KV
// head h_kv = h * kv_heads / heads.  K/V use stride kv_dim, Q/outputs use
// d_model.  No physical KV expand (plain copy via offset aliasing).
template <typename T>
void genOffsetPointers_GQA(T** offsets, int heads, int kv_heads, int max_batch,
                           int depth, int d_model, int kv_dim, T* k, T* q,
                           T* b1, T* v, T* b2, cudaStream_t stream);

// Weighted-GQA physical expand: K_out[n,h,s,d] = Σ_k W_k[h,k]*K_in[n,k,s,d]
// (same for V).  Used when plain GQA's offset-aliasing isn't enough because
// each Q head needs a learnable blend across the kv_heads.  Input K/V are
// kv_dim-strided; output K/V are d_model-strided.
//
// Optional fused pre-blend ops (pointer == nullptr / softcap == 0 to skip):
//   b_k        : (kv_dim,) bias added to k_in before the blend sum.
//   b_v        : (kv_dim,) bias (PGB) added to v_in before softcap+blend.
//   v_softcap  : applies v_cap * tanh(v_val / v_cap) per kv-slice.
// Math identity:  Σ_k W[h,k] * (X[k] + b[k])  ==  Σ_k W[h,k]*X[k] + Σ_k W[h,k]*b[k]
// so fusing bias inside the loop yields the same K_out/V_out as a
// separate addBias kernel followed by the unfused expand.
// k_in_stride / v_in_stride (optional, default 0 → use kv_dim):
//   The per-position column stride of K/V in their input buffers.  Pass
//   non-zero when K is embedded in a wider container (e.g. the fused
//   QKV output buffer where K lives at offset d_model with stride
//   d_model + 3*kv_dim).  The caller passes a K-pointer already offset
//   to K's first row; this stride is then the WIDER container stride.
//   V usually remains kv_dim-strided (its own SwiGLU output buffer).
template <typename T>
void expandKVWeighted(T* k_out, T* v_out, const T* k_in, const T* v_in,
                      const T* w_k, const T* w_v, int N, int heads,
                      int kv_heads, int depth, int d_model, int kv_dim,
                      cudaStream_t stream,
                      const T* b_k = nullptr, const T* b_v = nullptr,
                      float v_softcap = 0.0f,
                      const T* exo_k_anc = nullptr,
                      const T* exo_v_anc = nullptr,
                      int k_in_stride = 0, int v_in_stride = 0);

// addBiasAndAdd: out[i] = in[i] + broadcast_bias[i % width] + add[i].
// Used to fuse Q2's bias add with the Q exo anchor add — the K/V exo
// goes into expandKVWeighted, so this kernel completes AddExoAnchors
// elimination on the GQA + ExoFormer path.
template <typename T>
void addBiasAndAdd(T* out, const T* in, const T* bias, const T* add,
                   int total, int width, cudaStream_t stream);

// addBiasSiluStrided: out[b*stride + c] = silu(in[b*stride + c] + bias[c])
// for b in [0, batch), c in [0, inner).  Used for Q1's silu+bias step
// when Q1 lives in the fused QKV output buffer at offset 0 with the
// wider column stride (= d_model + 3*kv_dim).
template <typename T>
void addBiasSiluStrided(T* out, const T* in, const T* bias,
                        int batch, int inner, int stride,
                        cudaStream_t stream);

// RMSNorm: x * |gamma| / rms(x).  Same calling convention as LayerNorm
// but omits mean-centering and beta.  bias/skip may be null.
template <typename T>
void RMSNorm(int N, int C, T* output, const T* input, const T* bias,
             const T* skip, const T* gammas, float ep, float alpha,
             ActivationFunction act, cudaStream_t stream,
             const T* input2 = nullptr);

// Fused SiLU(gate) * up elementwise (SwiGLU FFN).
// swiglu_softcap > 0 applies tanh(out/cap)*cap to the silu*up product before
// returning. 0 disables (no cap). Used by FFN paths; V projection paths
// continue to use v_softcap via a separate fusion in their own kernels.
template <typename T>
void SwiGLUElementwise(int total, T* output, const T* gate, const T* up,
                       const T* gate_bias, const T* up_bias,
                       cudaStream_t stream,
                       float swiglu_softcap = 0.0f);

// SwiGLU with biases fused into the elementwise kernel. Saves 2 kernel
// launches per SwiGLU FFN call versus separate addBiasBatched + SwiGLU.
// `dff` is the per-token feature dim for bias indexing.
// swiglu_softcap > 0 applies tanh-bound to the output; 0 disables.
// pgb_bias (optional): broadcasts (dff,) PGB add over the silu*up output
//   (post-softcap, pre-write).  Folds the FFN's separate
//   addBiasBatched(ffn_pgb_) launch into this kernel.
template <typename T>
void SwiGLUElementwiseWithBias(int total, int dff, T* output, const T* gate,
                                const T* up, const T* gate_bias,
                                const T* up_bias, cudaStream_t stream,
                                float swiglu_softcap = 0.0f,
                                const T* pgb_bias = nullptr);

// Fused SwiGLU for single-GEMM output [gate; up] of shape (batch, 2*dff).
// Reads interleaved gate/up per batch element, adds biases, applies silu * up.
// Output shape: (batch, dff).
// swiglu_softcap > 0 applies tanh-bound to the output; 0 disables.
// pgb_bias (optional): see SwiGLUElementwiseWithBias above.
// column_stride (optional, default = 2*dff): the per-batch stride in
// the gate_up buffer.  Pass non-zero to read gate/up out of a wider
// container (e.g. the fused QKV output buffer where gate/up live at
// some offset with stride d_model+3*kv_dim).  Within each column, gate
// occupies rows [0..dff), up rows [dff..2*dff).  Pass 0 to use the
// natural 2*dff packing.
template <typename T>
void SwiGLUFusedGateUp(int batch, int dff, T* output, const T* gate_up,
                       const T* gate_bias, const T* up_bias,
                       cudaStream_t stream,
                       float swiglu_softcap = 0.0f,
                       const T* pgb_bias = nullptr,
                       int column_stride = 0);

// Concatenate per-square features with broadcast global features.
// output: (N, 64, sq_size + global_size) from sq: (N*64, sq_size) and
// global: (N, global_size) broadcast to all 64 squares.
template <typename T>
void ConcatSquareAndGlobal(int N, int sq_size, int global_size,
                           T* output, const T* sq_features,
                           const T* global_features, cudaStream_t stream);

// Weighted blend: output[i] = alpha * a[i] + beta * b[i].
// Used for ExoFormer anchor blending.
template <typename T>
void WeightedAdd(int total, T* output, float alpha, const T* a,
                 float beta, const T* b, cudaStream_t stream);

// Fused ExoFormer anchor add: Q += Qa, K += Ka, V += Va in one kernel launch.
// All three tensors must have the same element count.
// Saves 2 kernel launch overheads per encoder layer vs 3 separate addVectors.
template <typename T>
void AddExoAnchors(int total, T* q, const T* aq, T* k, const T* ak,
                   T* v, const T* av, cudaStream_t stream);

#ifdef USE_CUTLASS
// CUTLASS GEMM + Bias + Activation: C = act(W^T @ X + bias).
// Fuses the projection GEMM with bias addition and optional SiLU activation.
// Half precision only (tensor cores). Falls back to cuBLAS when not available.
void cutlassGemmBias(void* output, const void* weight, const void* input,
                     const void* bias, int M, int N, int K,
                     ActivationFunction activation, cudaStream_t stream);
#endif

// Fused attention for 64-token sequences: Q×K^T + bias → softmax → ×V → output.
// Entire attention matrix lives in shared memory — never touches global.
// One thread block per (batch, head). SmolGen bias fused if non-null.
template <typename T>
void FusedAttention64(int N, int num_heads, int depth, int d_model,
                      T* output, const T* Q, const T* K, const T* V,
                      const T* smolgen_bias, cudaStream_t stream);

// Fused VGA-E: output[i] *= sigmoid(gate[i] + bias[i % bias_size]).
// Replaces addVectors(sigmoid) + ElementwiseMultiply in one pass.
template <typename T>
void FusedVGAE(int total, T* output, const T* gate, const T* bias,
               int bias_size, cudaStream_t stream);

// ── Shared gate bank adapters (A-Layout-2) ──
// h is the encoder layer's shared nonlinear basis silu(W_bank x + b),
// stored column-major (bank_dim, tokens).  Site gate pre-activation:
//   pre = diag[c] * h[token, c] + gate_b[c] (+ lrb[token, c])
// lrb (optional, (out_dim, tokens)) is the rank-r mixer output,
// materialized by small cuBLAS GEMMs (folding it into these kernels was
// measured slower — see note in common_kernels.cu).  Null => rank-0.
//
// BankGatedMul (GLU-V / GLU-Q / GLU-K / SwiGLU-FFN sites):
//   output = silu(pre) * (up + up_b) [+ pgb] [tanh-softcap]
// up is read at row stride `up_stride` (0 => out_dim, contiguous) so it
// can live inside a wider fused-GEMM output; output is written at row
// stride `out_stride` (0 => out_dim) so a strided fused-GEMM slot can be
// gated IN PLACE (out == up with out_stride == up_stride is same-index
// read-then-write, safe — likewise the contiguous out == up case).
template <typename T>
void BankGatedMul(int batch, int out_dim, int bank_dim, T* output,
                  const T* h, const T* lrb, const T* up, const T* diag,
                  const T* gate_b, const T* up_b, const T* pgb,
                  float softcap, int up_stride, cudaStream_t stream,
                  int out_stride = 0);

// FusedVGAEBank (VGA-E site): output[i] *= sigmoid(pre).
template <typename T>
void FusedVGAEBank(int total, T* output, const T* h, int bank_dim,
                   const T* lrb, const T* diag, const T* bias, int dim,
                   cudaStream_t stream);

// Fused Residual Add + LayerNorm: ln_output = LN(residual + delta).
// Also writes residual_output = residual + delta if non-null.
// Replaces addVectors + NormLayer in one pass.
template <typename T>
void FusedResidualAddLN(int tokens, int emb, T* ln_output,
                        T* residual_output, const T* residual,
                        const T* delta, const T* gamma, const T* beta,
                        float eps, cudaStream_t stream);

// Fused 3-way vector add: output = a + b + c. For parallel FFN where we
// need in_out += attn_out + ffn_out. Saves one kernel launch + one read/write
// pass over the biggest tensor (in_out) versus two addVectors calls.
template <typename T>
void Add3(int total, T* output, const T* a, const T* b, const T* c,
          cudaStream_t stream);

// GPU-side optimistic policy blend.  Linearly interpolates vanilla and
// optimistic policy LOGITS:
//   output[i] = (1 - alpha) * vanilla[i] + alpha * optimistic[i]
// `output` may alias either input (safe for in-place).  After host
// softmax this is mathematically equivalent to the per-edge geometric
// blend P_main^(1-α) · P_opt^α the CPU search code computes per node —
// but runs once per inference on GPU instead of N times per inference
// on CPU.  See common_kernels.cu for the derivation.
template <typename T>
void BlendPolicyLogits(int total, T* output, const T* vanilla,
                       const T* optimistic, float alpha, cudaStream_t stream);

// Dual-output variant: writes vanilla buffer with blend at alpha_root
// AND optimistic buffer with blend at alpha_internal in one read pass.
// Used for split-alpha mode where root and internal nodes want different
// blends — both pre-computed on GPU so search uses fast-path SetP at
// every node, picking the pre-blended buffer that matches its depth.
template <typename T>
void BlendPolicyLogitsDual(int total, T* out_v, T* out_o, const T* vanilla,
                            const T* optimistic, float alpha_root,
                            float alpha_internal, cudaStream_t stream);

// Element-wise multiply: output[i] = a[i] * b[i].
// Used for VGA-E gating where sigmoid is already applied to one operand.
template <typename T>
void ElementwiseMultiply(int total, T* output, const T* a, const T* b,
                         cudaStream_t stream);

void fusedMHA(void* output, void* mha_q, void* mha_k, void* mha_v, void* skip,
              int batch_size, int num_heads, int depth, cudaStream_t stream);

}  // namespace cudnn_backend
}  // namespace lczero
