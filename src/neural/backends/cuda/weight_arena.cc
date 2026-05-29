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

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/

#include "neural/backends/cuda/weight_arena.h"

#include <algorithm>
#include <cstdint>

#include "neural/backends/cuda/cuda_common.h"

namespace lczero {
namespace cudnn_backend {

// Definition of the thread-local arena pointer declared in
// weight_arena.h.  `extern thread_local` in the header + concrete
// thread_local definition here keeps a single TLS slot across
// translation units (vs each TU getting its own slot if defined
// inline in the header).
thread_local WeightArena* tl_weight_arena = nullptr;

namespace {
// Round `n` up to the nearest multiple of `align`.  `align` must be a
// power of two.  Used to enforce kAlignment-byte alignment of returned
// pointers + chunk sizes.
inline size_t AlignUp(size_t n, size_t align) {
  return (n + align - 1) & ~(align - 1);
}
}  // namespace

WeightArena::~WeightArena() {
  // Free all chunks back to the CUDA runtime.  Done as part of network
  // destruction — replaces the dozens of per-buffer cudaFree() calls
  // that used to live in EncoderBlock / AttentionBody destructors.
  // Failures here are logged but not propagated; this is destruction.
  for (auto& chunk : chunks_) {
    if (chunk.base != nullptr) {
      cudaFree(chunk.base);
    }
  }
  chunks_.clear();
}

void WeightArena::AddChunk(size_t min_bytes) {
  // Always allocate at least kDefaultChunkSize so we amortize the
  // cudaMalloc cost over many subsequent Allocate() calls.  If a
  // single request exceeds the default, satisfy it exactly (rounded
  // up to alignment).
  const size_t chunk_size = std::max(
      kDefaultChunkSize, AlignUp(min_bytes, kAlignment));
  void* base = nullptr;
  ReportCUDAErrors(cudaMalloc(&base, chunk_size));
  chunks_.push_back({base, chunk_size, 0});
}

void* WeightArena::Allocate(size_t bytes) {
  if (bytes == 0) return nullptr;

  const size_t aligned_bytes = AlignUp(bytes, kAlignment);

  // Try to fit in the current chunk's remaining capacity.  If not,
  // grow.  This is the only path that creates a new cudaMalloc during
  // network construction.
  if (chunks_.empty() ||
      chunks_.back().used + aligned_bytes > chunks_.back().capacity) {
    AddChunk(aligned_bytes);
  }

  Chunk& current = chunks_.back();
  void* ptr = static_cast<uint8_t*>(current.base) + current.used;
  current.used += aligned_bytes;
  bytes_used_ += bytes;  // accounting in "user bytes," not aligned
  return ptr;
}

void WeightArena::Reserve(size_t bytes) {
  if (bytes == 0) return;
  // Single up-front chunk sized to the caller's hint.  Subsequent
  // Allocate() calls fill into this chunk; only allocations beyond
  // `bytes` create growth chunks.  Use this when the caller has a
  // reasonable estimate of total weight bytes — typically computed
  // via a walk over cpu_weights before the AttentionBody ctor runs.
  AddChunk(bytes);
}

size_t WeightArena::TotalAllocated() const {
  size_t total = 0;
  for (const auto& chunk : chunks_) {
    total += chunk.capacity;
  }
  return total;
}

bool WeightArena::Owns(const void* ptr) const {
  // O(chunks) per call.  In practice chunks_.size() is 1-3, so a tight
  // loop is faster than a sorted-by-base lookup or interval tree.
  // Used by layer destructors to decide whether to skip cudaFree.
  if (ptr == nullptr) return false;
  const auto* p = static_cast<const uint8_t*>(ptr);
  for (const auto& chunk : chunks_) {
    const auto* base = static_cast<const uint8_t*>(chunk.base);
    if (p >= base && p < base + chunk.capacity) return true;
  }
  return false;
}

}  // namespace cudnn_backend
}  // namespace lczero
