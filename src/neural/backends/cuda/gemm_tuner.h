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

// Per-shape tuned cuBLAS-Lt matmul for fp16 I/O.  On first call for a
// given (transa, transb, m, n, k, lda, ldb, ldc) shape, we run cuBLAS-Lt's
// heuristic search to pick the best algorithm for that exact shape and
// cache the descriptor set + algo.  Subsequent calls reuse the cached
// artifacts and skip the heuristic lookup entirely.
//
// Why this helps: cuBLAS's `cublasHgemm` and `cublasGemmEx` dispatch
// through a generic heuristic that picks one algo based on shape
// approximations.  On a workload like transformer inference — where the
// SAME shape is called 60× per forward × thousands of forwards — a
// per-shape tuned algo choice is typically 5-15% faster than the default.
// The tuning cost (one heuristic search per unique shape) is paid once
// at first forward, then free forever.
//
// Usage (from layers.cc):
//
//   if (TunedGemm16Half(cublas, transa, transb, m, n, k, alpha,
//                       A, lda, B, ldb, beta, C, ldc)) {
//     // success — skip legacy cublasXgemm
//   } else {
//     // fall back to cublasHgemm / cublasGemmEx path
//   }
//
// Guarded by LC0_HAS_CUBLAS_LT (set by meson when cublasLt is found).
// DEFAULT OFF: the heuristic top-1 algo nearly matches cublasHgemm's
// internal choice for common fp16 shapes (neutral at batch=128 on sm_89).
// Set LC0_ENABLE_CUBLAS_LT_TUNED=1 to opt in.  A future top-N timing
// variant (pick actually-fastest algo by benchmarking candidates) would
// likely beat cublasHgemm, but that's follow-up work.

#pragma once

#include <cublas_v2.h>
#include <cuda_fp16.h>

#ifdef LC0_HAS_CUBLAS_LT

namespace lczero {
namespace cudnn_backend {

// Tries a cuBLAS-Lt tuned fp16 matmul.  Returns true on success, false if
// the caller should fall back to the legacy cublasHgemm path (runtime
// disable, first-call heuristic failure, shape out of cache, etc.).
//
// The `cublas` handle provides the stream to execute on; the cuBLAS-Lt
// handle is a global singleton internal to the tuner.
bool TunedGemm16Half(cublasHandle_t cublas, cublasOperation_t transa,
                     cublasOperation_t transb, int m, int n, int k,
                     float alpha, const __half* A, int lda,
                     const __half* B, int ldb, float beta, __half* C,
                     int ldc);

}  // namespace cudnn_backend
}  // namespace lczero

#endif  // LC0_HAS_CUBLAS_LT
