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
 */

#pragma once

#include <unordered_map>
#include <vector>

#include "proto/net.pb.h"

namespace lczero {

struct BaseWeights {
  explicit BaseWeights(const pblczero::Weights& weights);

  using Vec = std::vector<float>;
  struct ConvBlock {
    explicit ConvBlock(const pblczero::Weights::ConvBlock& block);

    Vec weights;
    Vec biases;
    Vec bn_gammas;
    Vec bn_betas;
    Vec bn_means;
    Vec bn_stddivs;
  };

  struct SEunit {
    explicit SEunit(const pblczero::Weights::SEunit& se);
    Vec w1;
    Vec b1;
    Vec w2;
    Vec b2;
  };

  struct Residual {
    explicit Residual(const pblczero::Weights::Residual& residual);
    ConvBlock conv1;
    ConvBlock conv2;
    SEunit se;
    bool has_se;
  };

  struct Smolgen {
    explicit Smolgen(const pblczero::Weights::Smolgen& smolgen);
    Vec compress;
    Vec dense1_w;
    Vec dense1_b;
    Vec ln1_gammas;
    Vec ln1_betas;
    Vec dense2_w;
    Vec dense2_b;
    Vec ln2_gammas;
    Vec ln2_betas;
  };

  struct MHA {
    explicit MHA(const pblczero::Weights::MHA& mha);
    Vec q_w;
    Vec q_b;
    Vec k_w;
    Vec k_b;
    Vec v_w;
    Vec v_b;
    Vec dense_w;
    Vec dense_b;
    Smolgen smolgen;
    bool has_smolgen;

    // Ray-conditioned RPE (per-layer bias tables, num_heads * 225 each)
    Vec rpe_base;
    Vec rpe_clear;
    Vec rpe_blocked;

    // NLA (NonLinear Attention) second-layer Q/K projection
    Vec q2_w;
    Vec q2_b;
    Vec k2_w;
    Vec k2_b;

    // GLU Attention (V): gated value projection
    Vec v_gate_w;
    Vec v_gate_b;
    Vec v_up_w;
    Vec v_up_b;

    // PGB (Post-Gating Bias on V)
    Vec pgb_v;

    // VGA-E (Element-wise gate on attention output)
    Vec vga_elem_gate_w;
    Vec vga_elem_gate_b;

    // Weighted GQA blend matrices (heads, kv_heads), row-major.
    // Empty when use_weighted_gqa is false; under plain GQA the CUDA
    // backend synthesizes block-selection weights at load time.
    Vec gqa_w_k;
    Vec gqa_w_v;

    // Shared gate bank adapters (A-Layout-2).  Non-empty diag means the
    // site reads the encoder layer's shared bank instead of owning a
    // full projection (v_gate_w / vga_elem_gate_w are then empty):
    //   pre = diag * h[..:out] + lr_b(lr_a(h)) + b
    // lr_a = (rank, bank) row-major, lr_b = (out, rank) row-major.
    // Empty lr_a/lr_b => rank 0 (diagonal-only).
    Vec bank_v_diag;
    Vec bank_v_b;
    Vec bank_v_lr_a;
    Vec bank_v_lr_b;
    Vec bank_vga_diag;
    Vec bank_vga_b;
    Vec bank_vga_lr_a;
    Vec bank_vga_lr_b;
    // GLU-Q/K via bank: gate adapters on Q (d_model) and K (kv_dim);
    // q_w/k_w are then the content (up) projections and q2_w/k2_w are
    // absent.
    Vec bank_q_diag;
    Vec bank_q_b;
    Vec bank_q_lr_a;
    Vec bank_q_lr_b;
    Vec bank_k_diag;
    Vec bank_k_b;
    Vec bank_k_lr_a;
    Vec bank_k_lr_b;
  };

  struct FFN {
    explicit FFN(const pblczero::Weights::FFN& mha);
    Vec dense1_w;
    Vec dense1_b;
    Vec dense2_w;
    Vec dense2_b;

    // SwiGLU FFN weights
    Vec gate_proj_w;
    Vec gate_proj_b;
    Vec up_proj_w;
    Vec up_proj_b;
    Vec down_proj_w;
    Vec down_proj_b;
    Vec pgb_ffn;  // Post-Gating Bias on SwiGLU hidden

    // Shared gate bank adapter for the SwiGLU gate (A-Layout-2);
    // replaces gate_proj_w/gate_proj_b when non-empty.
    Vec bank_gate_diag;
    Vec bank_gate_b;
    Vec bank_gate_lr_a;
    Vec bank_gate_lr_b;
  };

  struct EncoderLayer {
    explicit EncoderLayer(const pblczero::Weights::EncoderLayer& encoder);
    MHA mha;
    Vec ln1_gammas;
    Vec ln1_betas;
    FFN ffn;
    Vec ln2_gammas;
    Vec ln2_betas;

    // ExoFormer: per-layer anchor blending coefficients (2 scalars)
    Vec exo_lambda;

    // Shared gate bank (A-Layout-2): h = silu(gate_bank_w x + gate_bank_b),
    // consumed by the GLU-V / VGA-E / FFN gate adapters (see MHA / FFN
    // bank_* fields).  Non-empty gate_bank_w enables the feature.
    Vec gate_bank_w;
    Vec gate_bank_b;
  };

  // Input convnet.
  ConvBlock input;

  // Embedding preprocess layer.
  Vec ip_emb_preproc_w;
  Vec ip_emb_preproc_b;

  // Embedding layer
  Vec ip_emb_w;
  Vec ip_emb_b;

  // Embedding layernorm
  // @todo can this be folded into weights?
  Vec ip_emb_ln_gammas;
  Vec ip_emb_ln_betas;

  // Input gating
  Vec ip_mult_gate;
  Vec ip_add_gate;

  // Embedding feedforward network
  FFN ip_emb_ffn;
  Vec ip_emb_ffn_ln_gammas;
  Vec ip_emb_ffn_ln_betas;

  // Encoder stack.
  std::vector<EncoderLayer> encoder;
  int encoder_head_count;
  // Grouped-Query Attention: number of KV heads (< encoder_head_count
  // means GQA active).  0 = no GQA / full MHA.
  int kv_headcount = 0;

  // Encoder final norm (Pre-Norm only)
  Vec encoder_final_norm_gammas;
  Vec encoder_final_norm_betas;

  // ExoFormer: anchor projections (model-level, computed once from embedding)
  Vec exo_q_anc_w;
  Vec exo_k_anc_w;
  Vec exo_v_anc_w;

  // Exo-as-smolgen-bias: single projection from pooled initial flow to
  // (H * gen_sz) anchor code.  Shape: (H*gen_sz, emb_size) stored flat in
  // PyTorch nn.Linear(emb_size, H*gen_sz) convention (out, in).  Training
  // applies a 1/sqrt(N_layers) forward-pass scaling; this scaling is baked
  // into the exported weight so runtime uses it as a plain projection.
  Vec exo_smol_anchor_w;

  // Rich embedding: per-square MLP input is widened with
  // per-square extras (piece bits + attack_maps + material_info masks).
  Vec rich_emb_sq_w1;
  Vec rich_emb_sq_b1;
  Vec rich_emb_sq_w2;
  Vec rich_emb_sq_b2;
  Vec rich_emb_global_w;
  Vec rich_emb_global_b;

  // Format flags
  bool is_prenorm = false;
  bool use_rms_norm = false;
  bool use_swiglu_ffn = false;
  bool use_parallel_ffn = false;
  bool use_material_info = false;
  bool use_attack_maps = false;
  // Soft-cap values. 0.0 disables; non-zero value enables. No separate bool
  // flag — the value itself is the gate. Old nets that exported the legacy
  // `use_attn_logit_softcap` boolean are handled at proto-read time.
  float attn_logit_cap = 0.0f;
  float smolgen_softcap = 0.0f;
  float v_softcap = 0.0f;
  // SwiGLU FFN soft-cap: tanh(out/cap)*cap on silu(gate)*up before
  // down-projection. Bounds FFN multiplicative output, preventing
  // mid-late stack gradient explosions. 0.0 disables.
  float swiglu_softcap = 0.0f;

  // Multi-exit value branch layer. -1 = disabled (default); >=0 = value
  // heads read encoder flow at this layer instead of the final encoder
  // output. Policy heads still read final flow.
  int value_branch_at_layer = -1;

  // Residual tower.
  std::vector<Residual> residual;

  // Moves left head
  ConvBlock moves_left;
  Vec ip_mov_w;
  Vec ip_mov_b;
  Vec ip1_mov_w;
  Vec ip1_mov_b;
  Vec ip2_mov_w;
  Vec ip2_mov_b;

  // Smolgen global weights
  Vec smolgen_w;
  bool has_smolgen;
};

struct LegacyWeights : public BaseWeights {
  explicit LegacyWeights(const pblczero::Weights& weights);

  // Policy head
  // Extra convolution for AZ-style policy head
  ConvBlock policy1;
  ConvBlock policy;
  Vec ip_pol_w;
  Vec ip_pol_b;
  // Extra params for attention policy head
  Vec ip2_pol_w;
  Vec ip2_pol_b;
  Vec ip3_pol_w;
  Vec ip3_pol_b;
  Vec ip4_pol_w;
  int pol_encoder_head_count;
  std::vector<EncoderLayer> pol_encoder;

  // Value head
  ConvBlock value;
  Vec ip_val_w;
  Vec ip_val_b;
  Vec ip1_val_w;
  Vec ip1_val_b;
  Vec ip2_val_w;
  Vec ip2_val_b;
};

struct MultiHeadWeights : public BaseWeights {
  explicit MultiHeadWeights(const pblczero::Weights& weights);

  struct PolicyHead {
    explicit PolicyHead(const pblczero::Weights::PolicyHead& policyhead, Vec& w,
                        Vec& b);
    // Policy head
   private:
    // Storage in case _ip_pol_w/b are not shared among heads.
    Vec _ip_pol_w;
    Vec _ip_pol_b;

   public:
    // Reference to possibly shared value (to avoid unnecessary copies).
    Vec& ip_pol_w;
    Vec& ip_pol_b;
    // Extra convolution for AZ-style policy head
    ConvBlock policy1;
    ConvBlock policy;
    // Extra params for attention policy head
    Vec ip2_pol_w;
    Vec ip2_pol_b;
    Vec ip3_pol_w;
    Vec ip3_pol_b;
    Vec ip4_pol_w;
    int pol_encoder_head_count;
    std::vector<EncoderLayer> pol_encoder;
  };

  struct ValueHead {
    explicit ValueHead(const pblczero::Weights::ValueHead& valuehead);
    // Value head
    ConvBlock value;
    Vec ip_val_w;
    Vec ip_val_b;
    Vec ip1_val_w;
    Vec ip1_val_b;
    Vec ip2_val_w;
    Vec ip2_val_b;
    Vec ip_val_err_w;
    Vec ip_val_err_b;

    // SimPool attentive pooling
    Vec simpool_query;
    Vec simpool_key_w;
    Vec simpool_key_b;
  };

 private:
  Vec ip_pol_w;
  Vec ip_pol_b;

 public:
  // Policy and value multiheads
  std::unordered_map<std::string, ValueHead> value_heads;
  std::unordered_map<std::string, PolicyHead> policy_heads;
};

enum InputEmbedding {
  INPUT_EMBEDDING_NONE = 0,
  INPUT_EMBEDDING_PE_MAP = 1,
  INPUT_EMBEDDING_PE_DENSE = 2,
};

}  // namespace lczero
