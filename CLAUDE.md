# Lc0 Exostack Inference Optimization — Session State

> **READ FIRST AT SESSION START**: `C:\Users\maste\Downloads\TRAINING_DIARY.md`
> Contains diagnostic results, user preferences, failed/rejected experiments, and architectural decisions that persist across context compactions. CLAUDE.md (this file) covers CUDA/inference; TRAINING_DIARY.md covers training-side state. If a user asks about value head jaggedness, softcaps, smolgen, stability decisions, or "did we already test X" — check the diary before re-suggesting.

## Recent additions (May 2026 — GQA + NLA-Q-only + Weighted GQA)

User's current production net moved from 256×20 (38.9M) to **512×60 (269.8M params)** with these new features now wired end-to-end (proto → PyTorch export → CUDA inference):

- **`nla_q_only: true`** — NLA applies only to Q (`Q = Wq2(silu(Wq(x)))`); K is plain linear. Auto-detected from absent `k2_w` in proto.
- **`kv_heads: 8`** under 16 Q heads — Grouped-Query Attention. K and V are projected at `kv_dim = kv_heads × depth` (= 256) and physically expanded to `d_model` (= 512) via `expandKVWeighted` before attention. Downstream code (Exo, V-norm, QKᵀ) sees d_model-wide K/V unchanged.
- **`use_weighted_gqa: true`** (arXiv:2407.10855) — proto fields `gqa_w_k=43` / `gqa_w_v=44` carry learnable `(heads, kv_heads)` blend matrices. Each Q head sees a learnable mix of all KV heads, not just one. When absent under GQA, inference synthesizes block-selection weights (`w[h, h*KH/H]=1.0`) which reproduces plain GQA exactly — so old nets and `use_weighted_gqa=False` continue to work.
- **`attack_maps`** — confirmed supported (was incorrectly listed as unsupported in earlier CLAUDE.md). Computed on the fly from input planes via `attack_map_kernel` in `common_kernels.cu`; 6 features/square × 64 squares × N. No proto weights.

**Hard constraint:** GQA + full NLA is asserted off — the fused `mha_q1k1_w_` GEMM assumes d_model-wide K. Use `nla_q_only: true` to combine. The assert lives in the NLA branch of `EncoderBlock::Eval`.

**Buffer math (GQA + NLA-Q-only path):** Transient `k_kv`, `gu_kv`, `v_kv` buffers are carved out of `nla_qk_temp` (2×d_model wide × max_batch). Total `4*kv_dim*max_batch` must fit, i.e. `kv_heads ≤ heads/2`. User's `kv=8 / heads=16` lands exactly on the bound at both 256×20 (kv_dim=128, d_model=256) and 512×60 (kv_dim=256, d_model=512). For `kv_heads > heads/2`, bump `getMaxAttentionBodySize` reservation. For the non-NLA branch (plain GQA or GQA+GLU-V without NLA), reservation is `7*qkv` (covers Q/K/V + 4×kv_dim transients).

**Commits (May 2026):**
- `b14f5b0` NLA-Q-only + GQA foundation kernels (`expandKVWeighted_kernel`, `genOffsetPointers_GQA_kernel` dead-code)
- `84e3208` full GQA wiring — kv_dim K/V projections + expandKVWeighted in both NLA and non-NLA branches
- `d89565b` proto fields 43/44 + `network_legacy` reader + ctor uses proto blend when sized correctly
- `5d1a003` `build.sh` passes `-Db_lto_threads=0` to parallelize GCC LTRANS (was serial, 50 jobs)
- `cb1c035` CLAUDE.md feature-status fixes (attack_maps moved to Supported)

## Shared gate bank (June 2026) — program state

User's 640×60 net (t6-640x60.yaml: emb=640, heads=20, kv=10, NLA-Q-only + GLU-V + VGA-E + PGB + SwiGLU parallel-FFN post-norm). One per-layer nonlinear basis `h = silu(W_bank·x + b)` feeds gate sites through `BankAdapter` (`pre = diag⊙h[:w] + b (+ lr_b(lr_a(h)))`; rank 0 = diagonal-only). Presence-detected in CUDA — **no format flags**:

| arm | yaml | detection | form |
|---|---|---|---|
| base bank | `use_shared_gate_bank: true` | `gate_bank_w` + per-site `bank_*_diag` | GLU-V/VGA-E/FFN gates read h |
| rank-0 | `gate_bank_rank: 0` | absent `*_lr_a` | diagonal-only adapters (speed arm) |
| Layout-1 | `gate_bank_include_q: true` | `q2_w` present, `q_w` ABSENT | Q = Wq2(h) |
| K-on-bank | `gate_bank_include_k: true` | `k2_w` present, `k_w` ABSENT | K = Wk2(h) |
| GLU-Q | `gate_bank_glu_q: true` | `bank_q_diag` (+ `q_w`, no `q2_w`) | Q = silu(adapter(h))⊙(Wq·x+b) |
| GLU-K | `gate_bank_glu_k: true` | `bank_k_diag` (+ `k_w`, no `k2_w`) | K = silu(adapter(h))⊙(Wk·x+b) |
| gated-Q | `gate_bank_gated_q: true` (needs include_q) | `bank_q_diag` + `q2_w` | Q = silu(adapter(h))⊙(Wq2·h+b) — 3rd-order addressing |
| gated-K | `gate_bank_gated_k: true` (needs include_k) | `bank_k_diag` + `k2_w` | K likewise; identity init (diag=0, b=1.27846 → gate≡1): step 0 == ungated arm, usable as fine-tune extension; verified BIT-IDENTICAL vs ungated same-seed pb; cost −5.9% at h10 (4153 vs 4414) |

All Q/K arms require `use_nla: true` + `nla_q_only: true` in yaml. GLU-Q drops q2_w → net is structurally non-NLA in CUDA and routes via the fallback Q/K/V branch (which has a bank GLU-V arm). GLU-K alone routes via the NLA-Q-only branch. glu_q+include_k is forbidden (not routable). Proto: EncoderLayer `gate_bank_w/b=14/15`; MHA `bank_v_*=45-48`, `bank_vga_*=49-52`, `bank_q_*=53-56`, `bank_k_*=57-60`; FFN `bank_gate_*=18-21`. Bias semantics: GLU-form K/Q bias lives INSIDE the gate product — expandKVWeighted gets `b_k=nullptr` under GLU-K.

**Bench (4090, 640×60 seed-42 random, batch 128, 25 batches, SAME build/session, mean nps):**

| arm | capture-on | capture-off |
|---|---|---|
| nobank60 | 3198 | 3043 |
| bank60_r0 | 3357 | 3233 |
| bank60_r0_L1 (Q) | 3658 | 3484 |
| bank60_r0_QK (Q+K) | **3657** | **3552** |
| bank60_r0_GLUQK | 3539 | 3374 |

- **Cross-session bench numbers are NOT comparable** (QK measured 3206 capture-on last session, 3657 this session, same pb). Always re-baseline all arms in-session before concluding anything.
- The earlier "K-on-bank −6% under graph capture" finding did NOT reproduce on this build — QK ties L1 capture-on now.
- gluQK pays ~3-5% vs QK/L1: its Q/K gate kernels re-read h (~47 MB/layer) and that HBM traffic competes with the bottleneck FFN stream. In exchange Q/K keep private x-subspaces (the A-screen/`bank_q_overlap.py` diagnostic showed trained wq AVOIDS the bank basis — lift ~1.0 vs gates 1.8-2.2), and the form is 3rd-order. It's the capacity-hedge arm; QK is the max-speed arm.
- Fused [Wq|Wk|Wv_up] input GEMM extended to gluQK nets: tied with separate GEMMs at 60L (launch savings absorbed into FFN stall slack — same as 512×60). `LC0_NO_FUSED_QKV=1` forces the separate path (debug/parity).
- r64 (trained t6-640x60-bank-swa-7826) measured −6.4% vs nobank earlier; rank-0 is the production direction. lr_b folding into consuming kernels measured 1.4-4.5× slower than GEMM form — do not retry (note in common_kernels.cu).
- Verification pattern for new arms: 2L torch fwd/bwd + export-structure check → 2L fused-vs-separate CUDA output parity on same pb → 60L load + finite eval + legal bestmove → bench. Trained-checkpoint parity happens when an arm trains.
**Profile of the QK arm (2026-06-11, nsys capture-off — no node-trace inflation, batch 128).** Component map, % of GPU kernel time: 640-col GEMMs ×5/layer (bank, Q2, dense, FFN up, FFN down — dff=640) 28.6%; attention matmuls 17.1% + softmax 4.8%; **expandKVWeighted 10.7%** (weighted-GQA dense 10-way blend, ~4.8 ms/forward ON THE MAIN STREAM — the kernel re-reads every kv slice per output head); norms 9.5% (1 DeepNorm RMS/layer @ 43.8 µs = 6.1%; the 2 LN/layer are smolgen-internal = 3.4%); K2+V-up 320-col GEMMs 8.2% (poor tile efficiency at M=320); bank gate kernels 6.8% + VGA 3.3%; smolgen gen GEMM 4.9% + dense1 1.1%. Heads/embedding < 1%. GPU-busy sum ≈ 44.8 ms vs ~36 ms wall → main/FFN stream overlap working.

**Ablation prices (same build/session, capture-on means, QK base 3702):**
| change | nps | params |
|---|---|---|
| encoder_heads 20→10 (d_k 64, kv_heads 5; kv_dim stays 320) | **4339 (+17.2%)** | 241M (−41M) |
| smolgen off | 4054 (+9.5%) | 165.8M |
| smolgen_hidden_sz/gen_sz 256→128 (all heads kept) | 3780 (+2.1%) | **205.1M (−77M)** |

Notes: heads-10 win = softmax+smolgen-bias bandwidth halves + d_k 64 attention tiles much more efficient than d_k 32; its capacity cost is halved smolgen per-head bias maps. Smolgen ≈ 10-12% of wall and 117M params (41% of net). Weighted GQA's 10.7% is mostly a KERNEL deficiency (dense Σ with 10× read amplification) — fix in backend before reconsidering the feature. Backend queue from this profile: (1) expandKVWeighted rewrite (shared-mem staging / per-token mini-GEMM, est +6-8% nps, no capacity change); (2) graph_capture=true in selfplay (+3-7% measured, conf flag); (3) [Q2|K2] fused h-GEMM ~+2%; (4) int8 W8A8 QAT remains the big lever (25-40%).

**Smolgen dictionary probe (2026-06-11, `C:\Users\maste\Downloads\smolgen_rank_probe.py` — weight-space Stage 1).** Every emittable smolgen map lives in col(G) (G = shared smolgen_w, 4096×256). Probe measures: G spectrum, atom matrix-rank, per-site (layer×head, via dense2 blocks) atom concentration, shared-top-64-atom energy share. Results (share of total site energy in the shared top-64 atoms / per-site atoms for 90% energy):

| net | steps | top-64 share | site r90 |
|---|---|---|---|
| random baseline | — | 33.6% | 216 |
| t6-640x60-bank | 7.8k | 84.7% | 112 |
| t9-512x60 (smolgenbias) | 12.5k | 83.4% | 122 |
| t9-512x60-exostack | 272.5k | 89.1% | 73 |
| BT4 (official) | 6.1M | **94.2%** | 68 |

**Verdict: dictionary hypothesis licensed — concentration strengthens monotonically with maturity.** Converged smolgen ≈ shared ~64-atom dictionary + per-site mixtures. Atoms are NOT low matrix-rank (rank-8 holds only ~51-62% energy) → store dictionary atoms DENSE (M=64 → 512KB fp16, L2-resident), don't factor them. Mature-net per-site min share 53% → keep a small per-site dynamic residual (UVᵀ rank ~8) for outlier sites. Replacement design: `B_h = softcap(Σ_m α_h,m(g)·P_m + U_h(g)V_h(g)ᵀ)` — est +7-9% nps, −90M params vs current smolgen (10-12% wall, 117M params = 41% of 640×60 net). Init P from measured atoms (`smolgen_dict_t6bank.npz`, `smolgen_dict_t9_272k.npz` in Downloads). Stage 2 (realized-map stats: α-only variance share, residual rank sizing) needs activations — checkpoint hooks or an lc0 latent-dump patch; refines M and r but does not gate the arm.

**Smolgen dictionary — IMPLEMENTED end-to-end (2026-06-11).** `use_smolgen_dict: true` + `smolgen_dict_atoms` (M) + `smolgen_dict_rank` (r) + `smolgen_gen_sz: 64` (slim code — where the savings come from) + optional `smolgen_dict_init: <probe npz>`. Torch: SmolGen dict branch (`B = Σ_m α_m(z)·P_m + U(z)V(z)ᵀ`), shared `smol_dict_P` (M×4096, init from probe atoms) + fused bias-less decoder (gen_sz → M+2·64·r, rows [α|U|V], V zero-init). Proto: top-level `smolgen_dict_p=66`, `smolgen_dict_dec_w=67`; smolgen_w absent under dict. CUDA: AttentionBody uploads + passes to EncoderBlock (like smol_global); both smolgen step-5 sites (ms_smol + main) replaced by decoder GEMM → compose GEMM (4096×NH×M, P consumed OP_N lda=4096) → strided-batched UVᵀ (OP_T/OP_N trick lands C[i,j] at i·64+j) with beta=1; softmax/softcap unchanged. Verified: ms_smol-vs-main path outputs identical (LC0_DISABLE_MS_SMOLGEN=1); finite eval, legal bestmove.

**Measured (vs QK base same-session rerun 3649 capture-on / 3421 capture-off mean):**
| variant | capture-on | capture-off | params |
|---|---|---|---|
| M=64 r=8 | 3584 (−1.8%) | 3552 | 219.6M |
| **M=64 r=0** | **3902 (+6.9%)** | **3771** | 219.4M |
| M=128 r=0 | 3853 (+5.6%) | 3559 (noisy; median 3714) | 219.7M |

**LESSON: the r>0 UVᵀ strided-batched GEMM with beta=1 into the 21MB map buffer re-reads + re-writes the whole buffer (~40µs/layer) and cancels the entire gen-GEMM saving.** Rank-0 (mixture-only) delivers the predicted win. To rescue r>0: fuse the UVᵀ add into softmax_opt_64 (read U/V from dec_out, +r fma/logit, add BEFORE the smolgen softcap; dec_out in smol_interm2 provably survives until the same layer's softmax — layer i+1's smolgen waits on ln1_done(i+1)) — NOT implemented yet. Params: smolgen 117M → ~36M (dict arms ≈219.6M total net, −63M vs QK arm). Training arm recipe: `use_smolgen_dict: true, smolgen_dict_atoms: 64, smolgen_dict_rank: 0, smolgen_gen_sz: 64, smolgen_dict_init: smolgen_dict_t9_272k.npz` (npz in Downloads; 64 atoms saved — M>64 needs a re-run of the probe with `--atoms M`).

**Bank-net scratch-sizing bug (2026-06-11, fixed in b5d6654f) — READ BEFORE DEBUGGING "IMPOSSIBLE" CUBLAS ERRORS.** `getMaxAttentionBodySize` derived `encoder_d_model` from `mha.q_b`, which is ABSENT in Layout-1/QK bank pbs → d_model=0 → scratch reservation collapsed to the 2-slab fallback while EncoderBlock carves 6 slabs → Q/K/V/temp pointers up to ~340MB past the end of scratch. **Symptom depended on the machine's allocator layout**: unmapped pages on the Linux 8×4090 rig (CUBLAS INTERNAL_ERROR/EXECUTION_FAILED at innocent victim GEMMs, "moved" with max_batch and capture settings), silently-mapped neighbor allocations on the Windows dev box (everything "worked", sanitizer clean — ALL local bank-arm benches before this date ran in that state; timings valid, outputs luck-protected). Debugging ladder that found it: blocking+capture-off run → standalone same-shape GEMM (passed → args suspect, not library) → compute-sanitizer ON THE FAILING BOX (named the OOB kernel) → fix detour (small-m GemmEx, reverted) → sanitizer showed SAME OOB offset from a DIFFERENT kernel family = bad pointer argument, not bad kernel → `LC0_DUMP_HGEMM=1` now also prints a one-time `[bufmap]` (AttentionBody::Eval) — subtracting GEMM C-pointers from buffer bases gave `C = scratch + 3 slabs` in a 2-slab buffer. Lessons: (1) any new presence-detected weight form must be checked against EVERY place that derives sizes from field presence (grep `q_b.size()`-style derivations); (2) CUDA_LAUNCH_BLOCKING is INVALID under graph capture (sync-during-capture poisons it); (3) a cuBLAS error at valid-looking pointers on one machine but not another = suspect buffer sizing first. Rig result after fix: h10 QK net 4368 nps capture-on (old prod net 3295 = **+33%**), all max_batch/capture combos green.

- All arms are training-gated next: user trains r0 / L1 / QK / gluQK and picks on Elo-vs-nps. Sync torchprocess.py to the trainer + regenerate net_pb2.py there (`python -m grpc_tools.protoc --proto_path=<worktree>/proto --python_out=Downloads/proto net.proto`) BEFORE enabling new flags — stale net_pb2 silently drops fields (load-time validation in layers.cc now catches this).

## fp8 inference: tried, abandoned (May 2026)

**Verdict: fp8 inference on Ada (4090) is not viable for lc0 quality at acceptable throughput. Do not re-attempt without one of the prerequisites below.**

### What was tried (all reverted)

1. **cuBLAS-Lt fp8 with CUBLAS_COMPUTE_16F (fp16 accumulator)** — heuristic_fall=1045, no algos available. cuBLAS-Lt only exposes fp32 accumulator for fp8 on Ada.
2. **cuBLAS-Lt fp8 with CUBLAS_COMPUTE_32F (fp32 accumulator)** — supported, ~310 TFLOPS. Was working but we never benchmarked nps before switching to CUTLASS.
3. **Custom CUTLASS device::Gemm with fp16 accumulator** — overflowed real activations (k=768 × max_fp8²=448² = 154M >> fp16 65504). Cascading NaN crashed downstream.
4. **CUTLASS with fp32 accumulator + fp16 output** — epilogue saturated the unscaled accumulator to fp16 Inf BEFORE per-tensor scale could apply. 100% non-finite output.
5. **CUTLASS with fp32 accumulator + fp32 output + post-scale-then-cast-to-fp16** — finite output, ~310 TFLOPS. **Produced bad chess at multi-node search.** 1-node looked fine but MCTS amplified per-tensor quantization error across 60 layers.

### Why per-tensor fp8 doesn't work for lc0 inference

- E4M3 fp8 has 256 levels. Per-tensor scale to max_abs gives ~max/128 resolution per element = ~12% relative error at typical weight magnitudes (~0.1).
- Compounded across 60 encoder layers + final heads, residual stream noise pushes policy/value off enough that MCTS makes bad moves.
- Smolgen-compress (m=64) and value/policy heads are especially sensitive — their outputs feed softmaxes that amplify any error exponentially.
- Per-tensor scaling treats outliers uniformly: one large weight forces every other weight in the tensor to use lower precision.

### What the CUTLASS hardware/software actually supports on Ada

- `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` — fp32 accumulator. Works. cuBLAS-Lt CUBLAS_COMPUTE_32F dispatches this.
- `mma.sync.aligned.m16n8k16.row.col.f16.e4m3.e4m3.f16` — fp16 accumulator. Hardware instruction exists; cuBLAS-Lt does NOT emit algos using it. Reaches it via custom CUTLASS only. But accumulator overflows real activations regardless (see #3 above).
- CUTLASS device::Gemm 2.x is the only sm_89 fp8 path (3.x CollectiveBuilder is Hopper+).
- CUTLASS fp8 on sm_89 requires LayoutC=RowMajor (using ColumnMajor silently dispatches a broken kernel that ran cleanly but produced 100% NaN).

### Prerequisites before retrying fp8

Any ONE of these would change the calculus:

1. **Blackwell hardware (sm_100+) with MXFP8** — native block scaling (32-element blocks have per-block scales). Outliers don't dominate the whole-tensor scale. fp16-accumulator becomes safe because per-product magnitudes are bounded by block scales.
2. **Per-channel scaling implementation** — one scale per row of A and per column of B. Multi-day implementation: stride-aware quantize kernels, per-channel scale buffers, post-scale that applies row/column scales. Sometimes salvages quality on per-tensor-unsafe nets.
3. **Quantization-aware training (QAT)** — train with simulated fp8 in forward pass. Multi-week project; weights learn to be quantization-friendly.

### Throughput on 4090 (theoretical, measured where known)

| Mode | TFLOPS | vs cublasHgemm |
|---|---|---|
| fp16/fp16-accum (`cublasHgemm`, default) | ~330 | 1.00× |
| fp8/fp32-accum (cuBLAS-Lt or CUTLASS) | ~310 | ~0.94× |
| fp8/fp16-accum (custom kernel, gau-nernst) | ~512 | ~1.55× (but overflows on real data) |

So even if fp8 worked numerically, the only path that beats fp16 (fp16-accum) is the one that overflows. fp8 on Ada is a **net loss** for lc0 today.

### Files that existed during the attempt (now removed)

- `src/neural/backends/cuda/fp8_helpers.{cu,h}` — quantize kernels, descriptor + weight cache, Fp8HalfIoGemm dispatch
- `src/neural/backends/cuda/fp8_cutlass_gemm.{cu,h}` — CUTLASS GEMM template instantiation
- meson.build had custom_targets for both

If you re-attempt fp8 with Blackwell or per-channel scaling, the git history (commits `3da14f2` through `308b878`) has the scaffolding to start from.

## Project context

User is training custom Leela Chess Zero (lc0) transformer nets with a feature-rich "exostack" architecture and optimizing the CUDA inference backend for those features.

**Key directories:**
- `C:\Users\maste\OneDrive\Desktop\lc0-git\` — lc0 C++ main checkout (CUDA backend). Branch: `qk-norm-and-post-norm-bookend`.
- `C:\Users\maste\OneDrive\Desktop\lc0-git\.claude\worktrees\romantic-aryabhata-3f9b04\` — **the git worktree this assistant edits from**. Branch: `claude/romantic-aryabhata-3f9b04`. Pushes to `origin/qk-norm-and-post-norm-bookend`.
- `C:\Users\maste\Downloads\` — PyTorch training (`torchprocess.py`). `model_to_net.py` is NO LONGER on disk (was deleted).
- `C:\Users\maste\Downloads\configs\` — YAML training configs
- `C:\Users\maste\Downloads\proto\` — generated Python protobuf (`net_pb2.py`)

**Worktree gotcha (READ FIRST IF USER ASKS "WHY ISN'T MY EDIT VISIBLE"):** Edits the assistant makes via `Edit`/`Write` go into the *worktree's* file copy, not the main checkout. Workflow: edit in worktree → commit → push to remote → user must `git pull --ff-only` in main checkout. Main and worktree share `.git` but have separate working trees, separate branches, separate file copies. If user says "I don't see your edit", they're almost certainly looking at the stale main checkout.

## The exostack config (parallel FFN net)

**Current production (May 2026): 512×60 net, 245.7M params (dff dropped 768→512).**
- Heads: 16, d_k: 32, **dff: 512** (was 768; user took the FFN-bandwidth-bound speed win for accepted strength cost). KV heads: 8 (GQA, 2:1 ratio).
- Norm: Post-Norm + DeepNorm (α=0.302, β=0.214) + Parallel FFN
- Features: SwiGLU + SwiGLU softcap (70) + NLA-Q-only (K linear) + RMSNorm + SmolGen + Soft Policy + Optimistic ST + Value ST + Value Q + GQA(kv=8) + Weighted GQA + VGA-Elementwise (G1, gate=input) + GLU-V (silu) + PGB + Exo-SmolGen-Bias + Rich Embedding (per_sq_in=20, hidden=64, global_sz=32) + SimPool Value + Aux Exits @ [7,15,25,35,45,50,55,58] (training-only; inference skips) + Attn/V/SmolGen softcap=30 + Enriched Input (attack_maps + material_info) + No Input Gate.
- Activation: silu. Optimizer: nadam. AMP: bf16.

**dff transition (768 → 512, May 2026):** 33% FFN compute + bandwidth reduction. ~24M params saved (~9% of total). Expected ~5-10% nps gain at 512×60 batch 128 since FFN was the bottleneck. SwiGLU "recipe minimum" is dff = (2/3)·2·emb = 683 — 512 is below that, so there's some capacity loss. User accepted the tradeoff after exhausting kernel-level FFN optimizations.

**Previous production (256×20, 38.9M):** `256x20-b4-parallel-t9.yaml`. Pre-Norm + NLA + GLU-V + PGB + ExoFormer + VGA-E + SwiGLU + SimPool + learnable_temp + aux_exits + light_embedding + categorical_value_buckets=128 + parallel FFN. Benchmark history below is for this config. MUDD intentionally OFF (too expensive). Most of the 23,811 nps optimization journey is on this config — assume the same observations don't necessarily transfer to 512×60 (multi-stream balance shifts as GEMMs saturate the GPU at the larger size).

**Batch sizing — DO NOT recommend batch=1.** User's MCTS runs at medium-to-large batch (≥32, typically benchmarked at 128). Batch=1 is too slow to be relevant in practice. When analyzing where speedups come from, always frame in terms of batch≥32; don't fall back on "but at batch 1 the savings are bigger" — the user considers that a non-argument because they never run there.

**Strength-loss-for-speed decisions** (assistant should not re-frame these as quality wins):
- **NLA-Q-only**: deliberate strength loss to save the Wk2 GEMM. Not a quality choice.
- **GQA (kv_heads=8)**: deliberate strength loss to halve K/V projection cost.
- **Weighted GQA**: paper claims partial strength recovery vs plain GQA, but user treats it as part of the same strength-cost-for-speed bucket. Don't sell it as a quality win.
- The user has explicitly traded strength for speed on these; the assistant's job is to maximize the speed yield, not to re-litigate the tradeoff.

**EMPIRICAL: FFN stream is the bottleneck at 512×60 batch 128.**

Tested 2026-05-22: full NLA QK + no GQA vs NLA-Q-only + GQA(kv=8) ran at the same speed. That's ~8.6 GFLOPs/layer of main stream compute eliminated with zero measurable wall-time effect. Conclusion: main wall time was already at least 8.6 GFLOPs of "slack" below FFN wall time. **Main-stream optimizations don't pay until FFN gets faster first.**

This invalidates several theoretical models the assistant has used in this codebase (including my own earlier in this session) that put main stream as the bottleneck. The user's measurement is authoritative.

Concrete implications:
- The recent expandKVWeighted fusion work (commits 9a90898, d3d84f0) saves 3 launches/layer on main stream → likely absorbed into stall. Marginal nps gain, may improve CV.
- **CUTLASS fusedMHA gate is moot for nps gain** until FFN is unblocked, even if you could find a way to engage it past the softcap gates.
- **cublasLt epilogue on Q1/Q2** — also moot for nps until FFN is unblocked.
- **Where to attack**: FFN GEMMs (gate+up, down), FFN bias chain (silu+mult+softcap+pgb+down_bias), FFN bandwidth (hidden buffer reads/writes).
- **FFN wins that DID land**: pgb_ffn into SwiGLU kernel (`074b83e`) and ffn_down_b into combined LN bias (`681c7b7`). Both eliminated real GPU work on the FFN bottleneck stream and gave a small but measurable nps bump.
- **FFN wins that DID NOT land — see "Dead-end attempts" below.**
- **Main-stream optimizations landed for completeness but no nps gain (per the empirical FFN-bottleneck reality):**
  - `9a90898` K bias + V PGB + V softcap fused into expandKVWeighted
  - `d3d84f0` ExoFormer K/V anchor fused into expandKVWeighted
  - `77daa1f` Fused QKV GEMM (3 GEMMs → 1, M=1280 at user's shape).  Math verified correct, same nps as before — savings absorbed into FFN-stream stall slack.
  - All of these become real nps wins automatically if FFN ever gets faster (e.g., a future raw-PTX SwiGLU+down kernel that beats cuBLAS).  No further engineering needed at that point.

## Dead-end attempts — DO NOT REPEAT WITHOUT NEW HARDWARE/APPROACH

### Custom WMMA SwiGLU+down fused kernel (2026-05-22) — ABANDONED

Spent a session writing a from-scratch CUDA kernel that fuses SwiGLU + down GEMM, eliminating the dff-wide hidden buffer round-trip (~25 MB/layer, ~2.5 ms/inference theoretical savings).

**Attempts tried, all reverted (code removed at the end):**
1. Basic WMMA kernel (sync loads, ThreadblockShape 128×64): correct math, **tied with cuBLAS+SwiGLU**. The bandwidth savings were exactly absorbed by the slower-than-cuBLAS matmul portion.
2. B-tile coalescing fix (consecutive threads now hit consecutive `gate_up` addresses): still tied. The coalescing was a real bug fix but not the main bottleneck.
3. cp.async 2-stage pipelining with double-buffered shared tiles: **way slower**. Shared memory bloat collapsed occupancy from ~6 blocks/SM to 2.
4. Same with `out_smem` staging removed: still slow. cp.async issue overhead at 4 warps / 128 threads / 8 KB chunks was a poor compute-to-load ratio on Ada.
5. Drop cp.async, expand to ThreadblockShape 128×128 with direct col-major epilogue store: **still slow**.

**Why none worked:** `nvcuda::wmma` wrapper has ~20-30% overhead vs cuBLAS's hand-tuned `mma.sync.aligned.m16n8k16` PTX + `ldmatrix.sync` path on Ada. Any kernel using WMMA can't exceed cuBLAS at the matmul portion on this shape (M=512, K=768, N=8192) regardless of how clever the surrounding fusion is.

**Prerequisites before retrying SwiGLU+down kernel fusion:**
- Drop nvcuda::wmma entirely, write the kernel with raw `mma.sync.aligned.m16n8k16` PTX + `ldmatrix.sync` for fragment loads. Multi-session engineering, risky (PTX inline asm has many ways to be wrong).
- OR write a CUTLASS device::Gemm with a custom B-source iterator that does SwiGLU on load. Uses CUTLASS's tuned templates (which DO use the right PTX internally). Multi-session CUTLASS template programming.
- OR wait for Blackwell (sm_100+) where TMA + cp.async.bulk might shift the load/compute ratio enough that WMMA-based fusion catches up.

**Files that existed during the attempt (now removed):**
- `src/neural/backends/cuda/swiglu_down_fused.{h,cu}` — WMMA kernel, dispatch wrapper, all the iterations above
- meson.build had a `custom_target` for it
- layers.cc had `TryFusedSwiGLUDown<DataType>` helper + dispatch in the post-norm parallel FFN path

Commits during this attempt: `38c78f9` (scaffold), `6f84140` (coalescing fix), `96740b5` (cp.async with staging), `b6f0dd6` (cp.async no staging), `97144a8` (128×128 sync). All reverted in cleanup commit.

### Other paths considered and not pursued

- **SwiMGLU**: parameter-efficient FFN (1 W matrix + N learnable binary masks). Inference compute is ~4× SwiGLU's at nm=4 — it's a param-count optimization, NOT a speed optimization. Would slow down the bottleneck.
- **L2 persistence on `down_w`**: 47 MB of weights fit in 4090's 48 MB L2 carveout. Software-only stream attribute. Estimated 1-2% nps. Not attempted; quick win for next session.
- **dff reduction**: user explicitly refuses any strength trade on FFN dimensions.
- **Disable multi-stream FFN**: experimental, not measured.

### CUTLASS for FFN GEMMs (2026-05-22) — ABANDONED

Tested via `LC0_FFN_CUTLASS=1` env var that swapped cuBLAS GEMM with a CUTLASS `device::Gemm` at the same 128×128×32 tile / m16n8k16 instruction shape used by `cutlassGemmBias` (existing scaffolding from earlier work).

**Result: CUTLASS was SLOWER than cuBLAS** at the user's FFN shapes (M=1536/512, K=512/768, N=8192).  cuBLAS heuristic picks a better per-shape algo than a single fixed CUTLASS template.

**Implications:**
- `dual_gemm` would inherit this matmul slowdown — the bandwidth fusion (~25 MB/layer of gate_up round-trip, much of it L2-cached anyway) can't compensate.
- ANY CUTLASS-for-FFN path would need profile-guided tile selection per-shape (CUTLASS profiler tool, multi-session) to beat cuBLAS.  And even then, the upside is small because cuBLAS is already near-optimal here.
- **CUTLASS-for-FFN is dead on this hardware/shape combo.**  Do not retry without one of:
  - Different shapes (e.g. if user changes emb or dff substantially)
  - CUTLASS profiler-tuned per-shape templates
  - Blackwell hardware (sm_100+) with new mma instructions

Files that existed during the attempt (now removed):
- `cutlassGemmFp16NoBias` function in `cutlass_kernels.cu`
- `TryCutlassFFNGemm<DataType>` dispatcher in `layers.cc`
- Header declaration in `kernels.h`

Commits during this attempt: `c3905e7` (added), reverted in cleanup commit.

## Benchmark history (batch 128, MCTS-relevant)

| State | nps | Notes |
|---|---|---|
| smolgen-only baseline (256x20, dff=512) | ~36400 | Reference. Most features off. |
| Parallel exostack, initial | 17833 | Before optimizations |
| + persistent LN1 cache | 18326 | +2.8% |
| + bias fusion in SwiGLU kernels | ~18326 | noise |
| + fused gate+up GEMM | 18909 | +3.2% |
| + multi-stream FFN overlap | 19601 | +3.7% |
| Parallel net without SwiGLU | 20399 | Confirmed SwiGLU free in multi-stream mode |
| + NLA Q1+K1 fused GEMM + ExoFormer fused add + VGA-E deferred sigmoid (no SwiGLU) | 22763 mean / 23868 median | vs 20399 prior: +11.6% mean / +17% median; CV=6.4%, max=24086, min=19288 |
| + same opts, WITH SwiGLU | 21001 mean / 22594 median | vs 19601 prior: +7.1% mean / +15.3% median; CV=9.2%, max=22812, min=16847 |
| **Round 2 attempts (both reverted)** | | |
| + GLU-V fused GEMM (no SwiGLU) | 22313 mean / 24703 median | CV=12.9% — net negative vs R1 |
| + ExoFormer 3→1 strided-batched GEMM (no SwiGLU) | 22568 mean / 24841 median | CV=12.1% — net negative vs R1 |
| **After reverts (current stable)** | | |
| GLU-V reverted + contiguous exo_all_anc_buf_ kept (no SwiGLU) | **23811 mean / 23905 median** | **CV=1.7%**, max=24050, min=21321 — **+4.6% mean vs R1** |
| same, WITH SwiGLU | **22688 mean / 22788 median** | **CV=1.86%**, max=22918, min=18518 — **+8.0% mean vs R1** |

**Current stable state: 23811/23905 (no SWI), 22688/22788 (SWI). Best results so far.**
SwiGLU overhead: 4.7% (was 7.7% in R1). Total gain over initial: ~34% mean nps.

Total optimization win over initial: ~28% nps (without SwiGLU config). Gap to smolgen-only (36400) still ~35% — architectural cost of NLA/GLU-V/VGA-E/ExoFormer GEMMs.

**Why NLA fusion beat expectations (~12% not ~2%):** Fusing Q1+K1 into a single `(512×8192×256)` GEMM gives cuBLAS a much better-shaped problem than 2× `(256×8192×256)`. Tensor core occupancy and weight tile reuse both improve substantially with the larger M dimension. The fused weight (256KB fp16) also fits better in L2. Kernel launch reduction from ExoFormer and VGA-E changes contributed the rest.

**SwiGLU overhead has grown (3.9% → 7.7% mean):** The NLA attention speedup shifted the multi-stream balance. Gate+up fused GEMM `(768×8192×256)` now routinely outlasts the faster attention path, causing the main stream to stall at `ffn_done_event`. The high CV (9.2%) and low minimum (16,847) confirm occasional hard stalls. SwiGLU is **no longer "free"** after the NLA optimization.

**Why GLU-V and ExoFormer fusions were reverted (Round 2 lesson):** Both fusions improved the median (~+4%) but doubled CV (6.4% → 12-13%) and worsened the min/mean. Root cause: the parallel FFN setup has a main stream (attention) and secondary stream (FFN) synchronized at `ffn_done_event`. NLA fusion already made the main stream fast enough that it frequently blocks waiting for the secondary stream. Each further speedup of the main stream (GLU-V, ExoFormer) widens that imbalance, increasing stall frequency and variance. The **multi-stream balance is the ceiling** — gains that only speed up the main stream without speeding up the FFN stream will be absorbed by longer stalls.

**Contiguous exo_all_anc_buf_ is the hidden win:** Keeping `exo_all_anc_buf_` (combined Q/K/V anchor buffer) while reverting GLU-V fusion yielded +4.6% mean and CV collapse from 6.4% → 1.7%. Mechanism: `AddExoAnchors` is called 20× per forward pass on the main stream, reading from all 3 anchor buffers. Contiguous layout improves TLB/L2 cache consistency across those 20 accesses, tightening main stream timing jitter enough that the main/FFN streams are nearly perfectly balanced (CV 1.7% ≈ pure measurement noise). This is the best state so far.

**ExoFormer strided-batched specific note:** The 3 anchor GEMMs are computed once per forward pass (not per layer), so kernel-launch savings are ~2 per inference — negligible. The `strideB=0` broadcast pattern is also uncommon in cuBLAS and may trigger slower code paths.

## CUDA changes I made

### `proto/net.proto`
Added (across sessions):
- `use_parallel_ffn = 15`, `use_material_info = 16` (network_format flags)
- `kv_headcount = 48` (top-level `Weights`, GQA: 0 or =heads means full MHA)
- MHA fields: `gqa_w_k = 43`, `gqa_w_v = 44` (Weighted GQA blend matrices, row-major (heads, kv_heads))

### `src/neural/network_legacy.{h,cc}`
- `bool use_parallel_ffn`, `bool use_material_info`, `bool use_attack_maps` flags
- `int kv_headcount = 0` on `BaseWeights` (top-level)
- `Vec gqa_w_k`, `Vec gqa_w_v` on `BaseWeights::MHA`; reader pulls them from proto

### `src/neural/backends/cuda/network_cuda.cc` + `network_cudnn.cc`
- Read all the new network_format flags
- `getMaxAttentionBodySize` reservation: `5*qkv` for NLA (covers GQA transients via `nla_qk_temp`), `7*qkv` for non-NLA + GQA, `4*qkv` for non-NLA + GLU-V, `3*qkv` baseline

### `src/neural/backends/cuda/layers.h`
**EncoderBlock additions (across sessions):**
- `bool is_parallel_ffn_`
- `DataType* ln1_cache_` — persistent LN1 output, avoids 2 recomputes per layer
- `DataType* ffn_gate_up_w_` — concatenated gate+up weight for fused FFN GEMM
- `bool nla_q_only_` — auto-detected from absent `k2_w`
- `int kv_heads_` — defaults to `encoder_heads_` (no GQA) when ctor arg is 0
- `bool has_weighted_gqa_`; `DataType* gqa_w_k_`, `DataType* gqa_w_v_` — blend matrices (synthesized for plain GQA, from proto for weighted)
- `DataType* mha_kv_w_`, `DataType* mha_kv_b_` — reserved fields (currently unused; reserved for a future fused-KV variant)
- Extra `Eval()` params: `ffn_stream`, `ffn_cublas`, `ln1_done_event`, `ffn_done_event`, `ffn_buf_wide`, `ffn_buf_out`

**AttentionBody additions:**
- `bool is_parallel_ffn_`, `bool has_material_info_`, `bool has_attack_maps_`
- `multi_stream_ffn_` — auto-enabled when `is_parallel_ffn_` (was gated on env var, now always on)
- `cudaStream_t ffn_stream_`, `cublasHandle_t ffn_cublas_`, events, buffers
- `DataType* material_features_buf_` for 18 material features
- `DataType* attack_map_buf_` for 6 attack features/square × 64 squares (allocated to `N*64*6*sizeof(T)`)

### `src/neural/backends/cuda/layers.cc`
Main changes in `EncoderBlock::Eval` (across sessions):
1. **Persistent LN1 cache**: `ln1_cache_` survives SmolGen and Q/K/V. Falls back to recompute on OOM.
2. **Fused gate+up GEMM** (parallel FFN + SwiGLU only): one GEMM producing `(2*dff, batch)`, then `SwiGLUFusedGateUp` kernel applies biases + silu + mult.
3. **Multi-stream FFN path**: When `is_parallel_ffn_`, FFN runs on secondary stream concurrent with attention. Final `Add3` kernel does `in_out = in_out + attn_out + ffn_out` in one pass. Works for both SwiGLU and standard FFN.
4. **NLA-Q-only branch**: when `nla_q_only_`, Q goes through `Wq → silu → Wq2`; K is plain `Wk·x + b_k` (no silu, no second projection). Auto-engaged via `cpu_weights.mha.k2_w.size() == 0`.
5. **GQA (physical-expand)**: under `is_gqa = kv_heads_ < encoder_heads_`, K and V project at `kv_dim` into transient buffers `k_kv` / `v_kv` (carved out of `nla_qk_temp` for NLA path, after Q/K/V for non-NLA path). `expandKVWeighted<T>` then physically fans them into `mha_k` / `mha_v` at `d_model` using `(heads, kv_heads)` blend matrices. GLU-V's fused `[gate;up]` GEMM and `SwiGLUFusedGateUp` both run at `kv_dim` width under GQA. PGB / v_softcap also kv_dim width. Downstream code (ExoFormer, V-norm, QKᵀ) sees d_model-wide K/V unchanged — that's the entire point of physical expand vs offset-pointer alternatives.
6. **`assert(!is_gqa)` in full-NLA branch**: combining full NLA with GQA is not supported (fused `mha_q1k1_w_` assumes d_model-wide K).
7. **GQA blend synthesis at ctor**: when `is_gqa`, if proto carries non-empty `gqa_w_k`/`gqa_w_v` of size `H*KH`, upload verbatim and set `has_weighted_gqa_=true`. Otherwise synthesize block-selection weights (`w[h, h*KH/H]=1.0`) and upload those. Either way, downstream code calls `expandKVWeighted` identically.

`AttentionBody::Eval`:
- If `has_material_info_`: calls `computeMaterialFeatures` (18 features/batch, 64 threads × N blocks).
- If `has_attack_maps_`: calls `computeAttackMaps` (6 features × 64 squares × N) — `attack_map_kernel` reads input planes directly, no proto weights.
- Concat layout: `[input | material | attack | encoding]` NHWC via `preprocess_with_two_extras_kernel` when both extras are active; single-extra variants exist too.

### `src/neural/backends/cuda/common_kernels.cu` + `kernels.h`
New kernels (across sessions):
- `material_features_kernel` / `computeMaterialFeatures` — 18 piece-count / material-diff / mask features
- `attack_map_kernel` / `computeAttackMaps` — 6 attack features/square (own/opp attacked bits, attack counts/6, masked combos), one block per batch × 64 threads/square
- `preprocess_with_material_kernel` — 3-way concat for embedding input
- `preprocess_with_two_extras_kernel` — 4-way concat `[input | material | attack | encoding]`
- `swiglu_fused_kernel` / `SwiGLUFusedGateUp` — reads interleaved `[gate;up]` from fused GEMM output
- `swiglu_kernel` (existing) extended to support biases (`SwiGLUElementwiseWithBias`)
- `add3_kernel` / `Add3`, `add3_store_kernel` / `Add3WithStore` — 3-way residual add
- `expandKVWeighted_kernel` / `expandKVWeighted<T>` — GQA physical expand. Grid: `(N*heads, 64)`, block: `depth`. Each thread computes `out[n,h,s,d] = Σ_k W[h,k] * in[n,k,s,d]`. Both float and half template instantiations.
- `genOffsetPointers_GQA_kernel` — DEAD CODE under the physical-expand approach (was the alternative offset-aliasing path). Kept compiled but unused.

### Python export (`C:\Users\maste\Downloads\torchprocess.py`)
- `_init_net_format()`: sets `use_parallel_ffn`, `use_material_info`, `use_attack_maps` flags. Sets `kv_headcount = kv_heads` when `kv_heads > 0`.
- `save_weights_to_pb` encoder loop writes `gqa_w_k` / `gqa_w_v` to `enc.mha` for each layer (only when the PyTorch MHA module has them — i.e. `use_weighted_gqa=True` and GQA active).
- `model_to_net.py` (standalone `.pt → .pb.gz` export) no longer on disk. If recreated, copy the GQA write calls from `torchprocess.py`.

### Build (`build.sh`)
- Passes `-Db_lto_threads=0` so GCC emits `-flto=auto` and parallelizes LTRANS. Without this, GCC runs all 50 LTO partitions serially at link time (multiple minutes of dead air on release builds).

## Important user learnings (from session)

1. **Export pipeline**: training uses `torchprocess.py`'s `save_weights_to_pb`. `model_to_net.py` is no longer on disk; if recreated it needs the same flag-setting code and per-encoder MHA writes (including `gqa_w_k`/`gqa_w_v`).
2. **Training process must be RESTARTED** after regenerating `net_pb2.py` to pick up new proto fields.
3. **Multi-stream FFN hides SwiGLU cost**: FFN runs concurrent with attention, so at 256×20 removing SwiGLU only saves ~3-4% nps. SwiGLU is "effectively free" as long as attention path > FFN path. At 512×60 this likely diminishes (GEMMs saturate the GPU).
4. **Architectural costs at batch 128** (256×20):
   - NLA adds ~1.1ms (2 extra Q/K GEMMs) — biggest target. **Now mitigated by NLA-Q-only** (saves the Wk2 GEMM and silu).
   - GLU-V adds ~0.5ms (extra V GEMM)
   - VGA-E adds ~0.5ms (precompute GEMM)
   - SwiGLU adds ~0.5ms (free in multi-stream) — probably keep
   - GQA at kv=heads/2 halves K/V projection cost on net but adds `expandKVWeighted` kernel (~tiny — one launch per layer); net win on memory bandwidth.
5. **Gap to smolgen-only (36k nps)** is ~3.3ms, mostly explained by these extra GEMMs. Cannot be optimized away without retraining.

## User's current experiments

- Production: **512×60 (269.8M params)** with NLA-Q-only + GQA(kv=8) + Weighted GQA + GLU-V + VGA-E + Parallel FFN + Exo-SmolGen-Bias + SimPool + all four softcaps + material_info + attack_maps. Build pending verification at this scale.
- Material_info, attack_maps CUDA implementations are **complete but untested against PyTorch** — pending verification.
- Weighted GQA blend matrices initialise to plain-GQA block-selection in PyTorch, so the trained net starts identical to plain GQA and learns to blend.

## Features user rejected earlier (during pattern analysis)

DyT (not as good as LayerNorm), MoE FFN (underperformed), Q-Gate (redundant with VGA-E for Pre-Norm), Nexus, LIMe, self-distillation, entmax (diverged), LayerScale (gradient explosion). Recently tested: diff_attn (neutral), dwconv (probably neutral).

## The "break linearity" pattern

User's analysis revealed: every architectural WIN was an order increase (first→second or second→third). Implications:
- NLA, GLU-V, VGA-E, SwiGLU, SmolGen, GRU residual: all wins, all raise interaction order
- SmolGen biggest win (0→3rd order via global compression + pairwise biases)
- "Adding more information" (dwconv, tactical features, attack maps) without order increase: neutral

## Feature implementation status in lc0 CUDA

Supported:
- GLU-V, PGB, ExoFormer, VGA-E, SmolGen V1/V2/V3, NLA, **NLA-Q-only** (K linear; auto-detected from absent k2_w), MUDD, SimPool value, SwiGLU, Pre-Norm, Post-Norm (DeepNorm), ResFormer value residual, QK-norm, RMSNorm, categorical value head, parallel FFN, light embedding, heavy embedding, **material_info**, **attack_maps** (computed on the fly from input planes, no weights — `attack_map_kernel` in common_kernels.cu), **GQA** (kv_headcount<encoder_head_count; physical-expand via expandKVWeighted), **Weighted GQA** (arXiv:2407.10855; proto fields gqa_w_k=43/gqa_w_v=44 carry (heads,kv_heads) blend matrices; plain GQA falls back to synthesized block-selection weights).

Not supported in CUDA (training-only / experimental):
- DyT, MoE FFN, Q-Gate, Nexus, LIMe, entmax, LayerScale, TPA, MoH, dwconv, GRU residual, pair_bias, smolgen_qmod, aux exits (inference skips them — training only), tactical features.

GQA + full NLA is asserted off — use `nla_q_only: true` when combining (the fused mha_q1k1_w_ assumes d_model-wide K and would need a separate kv_dim path through Wk2).

## Known gotchas

- **Worktree vs main checkout (READ FIRST)**: assistant edits go into the worktree at `.claude\worktrees\romantic-aryabhata-3f9b04\`, not the main `lc0-git\`. User must `git pull --ff-only origin qk-norm-and-post-norm-bookend` in the main checkout to see the edits. The remote branch is always up-to-date after pushes.
- **Proto regeneration required after editing `net.proto`**: run `protoc --proto_path=C:/Users/maste/OneDrive/Desktop/lc0-git/proto --python_out=C:/Users/maste/Downloads/proto C:/Users/maste/OneDrive/Desktop/lc0-git/proto/net.proto`. Training process must be restarted after.
- **Empty LN2 weights when parallel FFN enabled**: PyTorch model skips creating `ln2` when `use_parallel_ffn=True`. Export writes empty LN2 gammas/betas to proto. CUDA handles this: null pointers never dereferenced in parallel path.
- **build.cmd** was modified to disable ONNX build (`-Donnx=false`) and prompt before deleting build dir. ONNX_PATH removed.
- **build.sh** passes `-Db_lto_threads=0` to parallelize GCC LTO. Override with `-Db_lto_threads=N` in caller args if needed.
- **GCC LTO warning** ("using serial compilation of N LTRANS jobs") = b_lto_threads not set — fix is `-Db_lto_threads=0`.
- **Multi-stream FFN works at 256x20** (~10-15% gain), probably diminishes at 512x60 where GEMMs saturate the GPU on their own.
- **GQA buffer carve-out bound**: NLA-Q-only + GQA + GLU-V requires `kv_heads ≤ heads/2` (so `4*kv_dim ≤ 2*d_model = nla_qk_temp width`). User's `kv=8, heads=16` hits this bound exactly. For larger `kv_heads`, bump `getMaxAttentionBodySize` reservation in `network_cuda.cc`.
- **Full NLA + GQA is asserted off** — use `nla_q_only: true` to combine.

## Open issues / TODOs

### FRC/DFRC advisor + opponent reliability (MEDIUM priority)

Status: workaround in place, root-cause fix pending.

**Symptom:** Advisor-mode selfplay on DFRC starting positions fails after a few moves with `Invalid move (no piece to move): <move>` parse errors in stderr.  lc0 and SF agree on the first N moves, then SF returns a move lc0 can't parse against its current board.  All subsequent SF moves are unparseable for the rest of the game — game continues without advisor (best-effort, no adjudication).

**Likely root cause:** KQkq castling field ambiguity in FRC positions like `RKRQBBNN/...` where multiple rooks per side exist.  lc0 and SF use different inference rules to identify which rook is "the kingside rook," and after the first castling move their views of the position diverge.  Plus a secondary suspect: castling-move format mismatch (king-takes-rook vs standard notation).

**Workaround:** `generate_advisor.py` and `generate_sf.py` use UHO_XXL standard chess only.  `generate.py` (pure selfplay, no external engine) is unaffected and uses dfrc.pgn + `--chess960=true`.

**Fix plan:**
1. Restore `FenWithShredderCastling()` call in `external_engine.cc` (function still exists, just not invoked).  Rewrites KQkq → file-letter form before sending FEN to SF.  Removes castling-rook ambiguity.  Earlier removed because SF plays weaker in 960 mode for direct play; that doesn't matter for the advisor path (SF doesn't play, just suggests).
2. Add a test harness: pick N random DFRC start positions, send each through 10 moves of advisor query, assert no parse failures.
3. Flip generators back to dfrc.pgn + `--chess960=true` once verified.

**Files:** detailed TODO comment at top of `src/selfplay/external_engine.cc`, just above the `FenWithShredderCastling` definition.

## Low-hanging fruit (DONE — Round 1, stable)

1. **NLA Q1+K1 fused GEMM** — builds `[W_q1; W_k1]` at load time; single GEMM with `ldb=2*d_model` strides Q2/K2 reads. Scratch bumped to `5×qkv_size` in `getMaxAttentionBodySize`.
2. **Fused ExoFormer anchor adds** — new `AddExoAnchors` kernel does Q+=Qa, K+=Ka, V+=Va in one launch (saves 2 launches/layer × 20 layers). GQA + lambda paths still use separate calls.
3. **Fused VGA-E deferred sigmoid** — removed `addVectors(sigmoid)` from precompute step; `FusedVGAE` at apply site now owns bias+sigmoid+multiply in one kernel (saves 1 launch/layer × 20 layers).
4. **Contiguous ExoFormer anchor buffer** — `exo_all_anc_buf_` is a single allocation; `exo_q/k/v_anc_buf_` are pointer aliases. Reduces allocations; Q/K/V outputs are contiguous in memory (structural cleanup, no perf change).

## Attempted optimizations that were reverted (Rounds 2 & 3)

- **GLU-V fused GEMM** — median +4% but CV doubled (6.4% → 12.9%), mean −2%. Cause: further speeds up the main attention stream, increasing stall time at `ffn_done_event`. Net negative.
- **ExoFormer 3→1 strided-batched GEMM** — similar CV spike. Only saves 2 kernel launches per inference (ExoFormer runs once per forward pass, not per layer); not worth the `strideB=0` overhead.
- **VGA-E precompute on third CUDA stream** — median +2%, max +2%, CV doubled (1.7% → 3.5%), mean flat. Infrastructure kept in layers.h/layers.cc as dead code (vga_stream_, vga_cublas_, vga_done_event_ in AttentionBody; vga_stream/vga_cublas/vga_done_event params in EncoderBlock::Eval) for future re-enablement.

**Key insight — "Perfect balance" ceiling:** At 23,811 nps / CV=1.7%, the main attention stream and FFN secondary stream run at almost exactly the same wall-clock time per layer. ANY optimization that removes work from the main stream (GLU-V, VGA-E, ExoFormer) breaks this balance — the main stream finishes sooner, stalls at `ffn_done_event`, and variance spikes. The VGA-E GEMM is not dead weight; it serves as natural timing padding keeping the streams synchronized. The only path through this ceiling is to speed up the FFN secondary stream first (to create slack), then speed up the main stream by an equivalent amount.

**Theoretical next steps (if FFN stream gets faster):**
- **cublasLt epilogue fusion** on FFN GEMMs: fuses bias+activation into the GEMM kernel (`CUBLASLT_EPILOGUE_BIAS` + activation), saving ~2 kernel launches per FFN layer × 20 layers. Estimated ~3-4% FFN speedup. Would then unlock VGA-E third stream for a clean ~2% mean gain. Implementation is significantly more verbose than cuBLAS.
- **INT8/FP8 quantization**: reduces memory bandwidth for both GEMMs; FFN benefits more (smaller, more bandwidth-limited). Requires quantization-aware training infrastructure.
- **GPU upgrade**: A100/H100 memory bandwidth favors FFN more than attention GEMMs; balance would shift and reverted optimizations (GLU-V, VGA-E stream) may become beneficial.

**Total optimization journey:** 17,833 → 23,811 nps (no SwiGLU) = **+33.5% over initial**. The `exo_all_anc_buf_` contiguous allocation was the surprise final piece — it tightened timing jitter across AddExoAnchors' 20× reads and accidentally landed the system in near-perfect multi-stream balance.

## Files modified across recent sessions (all committed and pushed to `origin/qk-norm-and-post-norm-bookend`)

CUDA / proto:
- `proto/net.proto` — `kv_headcount=48`, `gqa_w_k=43`, `gqa_w_v=44`, `use_parallel_ffn`, `use_material_info`, `use_attack_maps`
- `src/neural/network_legacy.{h,cc}` — `kv_headcount`, `Vec gqa_w_k`, `Vec gqa_w_v` on MHA struct + reader
- `src/neural/backends/cuda/layers.h` — `nla_q_only_`, `kv_heads_`, `gqa_w_k_`, `gqa_w_v_`, `has_weighted_gqa_`, `attack_map_buf_`, `mha_kv_w_`/`mha_kv_b_` reserved
- `src/neural/backends/cuda/layers.cc` — NLA-Q-only branch, GQA physical-expand path (both NLA and non-NLA branches), `expandKVWeighted` calls, GQA blend synthesis at ctor, GQA destructor cleanup, AttentionBody passes `kv_headcount`
- `src/neural/backends/cuda/common_kernels.cu` + `kernels.h` — `expandKVWeighted_kernel`, `attack_map_kernel`, `material_features_kernel`, `preprocess_with_two_extras_kernel`, `SwiGLUFusedGateUp`, `Add3`, `AddExoAnchors`, `FusedVGAE`
- `src/neural/backends/cuda/network_cuda.cc` — `getMaxAttentionBodySize` scratch reservation aware of GQA + GLU-V
- `src/neural/backends/cuda/network_cudnn.cc` — same flag wiring as network_cuda.cc
- `build.cmd` — ONNX disabled, build-dir prompt
- `build.sh` — `-Db_lto_threads=0` for parallel LTO

PyTorch side (in `C:\Users\maste\Downloads\`):
- `torchprocess.py` — `_init_net_format()` sets all the new flags; `save_weights_to_pb` writes `gqa_w_k`/`gqa_w_v` per layer
- configs in `Downloads/configs/` — `kv_heads: 8`, `nla_q_only: true`, `use_weighted_gqa: true`, etc.
- `model_to_net.py` is NO LONGER on disk

Key remote branch: `qk-norm-and-post-norm-bookend` on `https://github.com/masterkni6/lc0`. Worktree branch: `claude/romantic-aryabhata-3f9b04`. Both share the worktree commits.
