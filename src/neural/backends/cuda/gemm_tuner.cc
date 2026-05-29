/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2026 The LCZero Authors

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

#include "gemm_tuner.h"

#ifdef LC0_HAS_CUBLAS_LT

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

namespace lczero {
namespace cudnn_backend {

namespace {

struct GemmKey {
  int transa;   // cublasOperation_t fits in int
  int transb;
  int m, n, k;
  int lda, ldb, ldc;

  bool operator==(const GemmKey& o) const {
    return transa == o.transa && transb == o.transb && m == o.m && n == o.n &&
           k == o.k && lda == o.lda && ldb == o.ldb && ldc == o.ldc;
  }
};

struct GemmKeyHash {
  size_t operator()(const GemmKey& k) const noexcept {
    // Simple FNV-like mix. 8 ints → 32 bytes. Collisions are fine; the
    // cache is keyed by full equality too.
    uint64_t h = 1469598103934665603ull;
    auto mix = [&](uint64_t v) {
      h ^= v;
      h *= 1099511628211ull;
    };
    mix((uint32_t)k.transa);
    mix((uint32_t)k.transb);
    mix((uint32_t)k.m);
    mix((uint32_t)k.n);
    mix((uint32_t)k.k);
    mix((uint32_t)k.lda);
    mix((uint32_t)k.ldb);
    mix((uint32_t)k.ldc);
    return (size_t)h;
  }
};

struct GemmCacheEntry {
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t a_layout = nullptr;
  cublasLtMatrixLayout_t b_layout = nullptr;
  cublasLtMatrixLayout_t c_layout = nullptr;
  cublasLtMatmulHeuristicResult_t heuristic{};
  bool valid = false;
  // `negative` marks this shape as permanently unsupported so we stop
  // retrying — the caller always falls back on false.  Populated on the
  // first call that fails heuristic lookup.
  bool negative = false;
};

class GemmTuner {
 public:
  static GemmTuner& Get() {
    static GemmTuner instance;
    return instance;
  }

  bool Matmul(cublasHandle_t cublas, cublasOperation_t transa,
              cublasOperation_t transb, int m, int n, int k, float alpha,
              const __half* A, int lda, const __half* B, int ldb,
              float beta, __half* C, int ldc) {
    if (disabled_) return false;
    if (!EnsureInit()) return false;

    cudaStream_t stream = nullptr;
    if (cublasGetStream(cublas, &stream) != CUBLAS_STATUS_SUCCESS) {
      return false;
    }

    GemmKey key{(int)transa, (int)transb, m, n, k, lda, ldb, ldc};
    GemmCacheEntry* entry = nullptr;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto it = cache_.find(key);
      if (it == cache_.end()) {
        GemmCacheEntry fresh;
        if (!BuildEntry(key, &fresh)) {
          fresh.negative = true;
        }
        auto insert = cache_.emplace(key, fresh);
        entry = &insert.first->second;
      } else {
        entry = &it->second;
      }
    }

    if (entry->negative || !entry->valid) return false;

    // Scale type was chosen by BuildEntry; pick the matching alpha/beta
    // representation here.  For CUBLAS_COMPUTE_16F (fp16 accum, matches
    // cublasHgemm) we need __half scales.  For 32F_FAST_16F (fp32 accum)
    // we pass float scales.
    cublasStatus_t st;
    if (use_fp16_compute_) {
      const __half h_alpha = (__half)alpha;
      const __half h_beta  = (__half)beta;
      st = cublasLtMatmul(
          lt_handle_, entry->desc,
          &h_alpha, A, entry->a_layout, B, entry->b_layout,
          &h_beta,  C, entry->c_layout, C, entry->c_layout,
          &entry->heuristic.algo, workspace_, workspace_bytes_, stream);
    } else {
      float alpha_f = alpha;
      float beta_f = beta;
      st = cublasLtMatmul(
          lt_handle_, entry->desc,
          &alpha_f, A, entry->a_layout, B, entry->b_layout,
          &beta_f,  C, entry->c_layout, C, entry->c_layout,
          &entry->heuristic.algo, workspace_, workspace_bytes_, stream);
    }
    if (st != CUBLAS_STATUS_SUCCESS) {
      // One-shot shouldn't disable globally; just report and fall back.
      if (LogErrors()) {
        fprintf(stderr,
                "[gemm_tuner] cublasLtMatmul failed rc=%d m=%d n=%d k=%d — "
                "falling back to Hgemm for this shape.\n",
                (int)st, m, n, k);
        fflush(stderr);
      }
      // Mark this entry as negative so subsequent calls skip straight to
      // the fallback without the Lt overhead.
      std::lock_guard<std::mutex> lock(mutex_);
      entry->negative = true;
      return false;
    }
    return true;
  }

 private:
  GemmTuner() {
    // Default-OFF: cuBLAS-Lt's top-1 heuristic picks an algo nearly identical
    // to what cublasHgemm dispatches internally for common fp16 shapes, but
    // cublasLtMatmul adds per-call setup overhead that cancels any gain.
    // Empirically neutral (±10 nps) at batch=128 on sm_89.  Code kept and
    // wired in for future experiments (top-N timing instead of top-1
    // heuristic would likely beat cublasHgemm; left as follow-up work).
    // Set LC0_ENABLE_CUBLAS_LT_TUNED=1 to opt in.
    disabled_ = std::getenv("LC0_ENABLE_CUBLAS_LT_TUNED") == nullptr;
  }
  ~GemmTuner() {
    // Let OS reclaim — destructor order during program teardown is fragile
    // with CUDA context already gone.  Not worth explicit cleanup.
  }
  GemmTuner(const GemmTuner&) = delete;
  GemmTuner& operator=(const GemmTuner&) = delete;

  bool LogErrors() {
    static const bool kLog =
        std::getenv("LC0_CUBLAS_LT_TUNED_LOG") != nullptr;
    return kLog;
  }

  bool EnsureInit() {
    if (init_done_) return init_ok_;
    std::lock_guard<std::mutex> lock(mutex_);
    if (init_done_) return init_ok_;

    if (cublasLtCreate(&lt_handle_) != CUBLAS_STATUS_SUCCESS) {
      init_done_ = true;
      return false;
    }
    // 16 MB workspace: plenty for any fp16 algo we'll see at inference
    // shapes (typical algo needs <1 MB; split-K reductions need a few MB).
    workspace_bytes_ = 16 * 1024 * 1024;
    if (cudaMalloc(&workspace_, workspace_bytes_) != cudaSuccess) {
      cublasLtDestroy(lt_handle_);
      lt_handle_ = nullptr;
      init_done_ = true;
      return false;
    }
    init_done_ = true;
    init_ok_ = true;
    return true;
  }

  // Populate a fresh entry: build descriptors and run heuristic search.
  // Returns false if anything in that chain fails (entry will be marked
  // negative by the caller and never retried).
  bool BuildEntry(const GemmKey& key, GemmCacheEntry* out) {
    // Compute type: CUBLAS_COMPUTE_16F gives fp16 accum (matches cublasHgemm
    // behavior and maximizes TensorCore throughput on sm_89).
    // LC0_CUBLAS_FP32_ACCUM=1 upgrades to 32F accum for deep nets that
    // overflow fp16 during reduction — mirrors the legacy fast path's
    // env-var switch exactly.  Scale type must match compute type per
    // cuBLAS-Lt docs: 16F compute → 16F scale, 32F_FAST_16F → 32F scale.
    static const bool kFp32Accum =
        std::getenv("LC0_CUBLAS_FP32_ACCUM") != nullptr;
    cublasComputeType_t compute_type =
        kFp32Accum ? CUBLAS_COMPUTE_32F_FAST_16F : CUBLAS_COMPUTE_16F;
    cudaDataType_t scale_type = kFp32Accum ? CUDA_R_32F : CUDA_R_16F;
    use_fp16_compute_ = !kFp32Accum;

    cublasLtMatmulDesc_t desc = nullptr;
    if (cublasLtMatmulDescCreate(&desc, compute_type, scale_type) !=
        CUBLAS_STATUS_SUCCESS) {
      return false;
    }
    int32_t tra = (int32_t)key.transa;
    int32_t trb = (int32_t)key.transb;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &tra,
                                    sizeof(tra));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &trb,
                                    sizeof(trb));

    // Matrix layouts — cuBLAS is column-major.  For A (m,k) with
    // CUBLAS_OP_T, the stored shape is (k,m) and the logical op yields
    // an (m,k) matrix.  Layout rows/cols use the STORED (pre-op) shape.
    const int a_rows = (key.transa == CUBLAS_OP_N) ? key.m : key.k;
    const int a_cols = (key.transa == CUBLAS_OP_N) ? key.k : key.m;
    const int b_rows = (key.transb == CUBLAS_OP_N) ? key.k : key.n;
    const int b_cols = (key.transb == CUBLAS_OP_N) ? key.n : key.k;

    cublasLtMatrixLayout_t a_layout = nullptr;
    cublasLtMatrixLayout_t b_layout = nullptr;
    cublasLtMatrixLayout_t c_layout = nullptr;
    if (cublasLtMatrixLayoutCreate(&a_layout, CUDA_R_16F, a_rows, a_cols,
                                    key.lda) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&b_layout, CUDA_R_16F, b_rows, b_cols,
                                    key.ldb) != CUBLAS_STATUS_SUCCESS ||
        cublasLtMatrixLayoutCreate(&c_layout, CUDA_R_16F, key.m, key.n,
                                    key.ldc) != CUBLAS_STATUS_SUCCESS) {
      if (a_layout) cublasLtMatrixLayoutDestroy(a_layout);
      if (b_layout) cublasLtMatrixLayoutDestroy(b_layout);
      if (c_layout) cublasLtMatrixLayoutDestroy(c_layout);
      cublasLtMatmulDescDestroy(desc);
      return false;
    }

    // Heuristic preference: constrain to what our workspace can hold, and
    // prefer algorithms that support our compute configuration.
    cublasLtMatmulPreference_t pref = nullptr;
    if (cublasLtMatmulPreferenceCreate(&pref) != CUBLAS_STATUS_SUCCESS) {
      cublasLtMatrixLayoutDestroy(a_layout);
      cublasLtMatrixLayoutDestroy(b_layout);
      cublasLtMatrixLayoutDestroy(c_layout);
      cublasLtMatmulDescDestroy(desc);
      return false;
    }
    cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes_,
        sizeof(workspace_bytes_));

    cublasLtMatmulHeuristicResult_t heuristic{};
    int returned = 0;
    auto st = cublasLtMatmulAlgoGetHeuristic(
        lt_handle_, desc, a_layout, b_layout, c_layout, c_layout, pref, 1,
        &heuristic, &returned);
    cublasLtMatmulPreferenceDestroy(pref);

    if (st != CUBLAS_STATUS_SUCCESS || returned == 0) {
      cublasLtMatrixLayoutDestroy(a_layout);
      cublasLtMatrixLayoutDestroy(b_layout);
      cublasLtMatrixLayoutDestroy(c_layout);
      cublasLtMatmulDescDestroy(desc);
      if (LogErrors()) {
        fprintf(stderr,
                "[gemm_tuner] no heuristic for transa=%d transb=%d m=%d n=%d "
                "k=%d lda=%d ldb=%d ldc=%d (rc=%d, returned=%d)\n",
                (int)key.transa, (int)key.transb, key.m, key.n, key.k,
                key.lda, key.ldb, key.ldc, (int)st, returned);
        fflush(stderr);
      }
      return false;
    }

    out->desc = desc;
    out->a_layout = a_layout;
    out->b_layout = b_layout;
    out->c_layout = c_layout;
    out->heuristic = heuristic;
    out->valid = true;
    if (LogErrors()) {
      fprintf(stderr,
              "[gemm_tuner] cached algo for transa=%d transb=%d m=%d n=%d "
              "k=%d lda=%d ldb=%d ldc=%d (workspace=%zu)\n",
              (int)key.transa, (int)key.transb, key.m, key.n, key.k, key.lda,
              key.ldb, key.ldc, (size_t)heuristic.workspaceSize);
      fflush(stderr);
    }
    return true;
  }

  bool disabled_ = false;
  bool init_done_ = false;
  bool init_ok_ = false;
  bool use_fp16_compute_ = true;  // set by BuildEntry based on env var
  cublasLtHandle_t lt_handle_ = nullptr;
  void* workspace_ = nullptr;
  size_t workspace_bytes_ = 0;
  std::mutex mutex_;
  std::unordered_map<GemmKey, GemmCacheEntry, GemmKeyHash> cache_;
};

}  // namespace

bool TunedGemm16Half(cublasHandle_t cublas, cublasOperation_t transa,
                     cublasOperation_t transb, int m, int n, int k,
                     float alpha, const __half* A, int lda,
                     const __half* B, int ldb, float beta, __half* C,
                     int ldc) {
  return GemmTuner::Get().Matmul(cublas, transa, transb, m, n, k, alpha, A,
                                   lda, B, ldb, beta, C, ldc);
}

}  // namespace cudnn_backend
}  // namespace lczero

#endif  // LC0_HAS_CUBLAS_LT
