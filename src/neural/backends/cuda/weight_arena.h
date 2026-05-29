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

#pragma once

#include <cstddef>
#include <vector>

namespace lczero {
namespace cudnn_backend {

// Bump-allocator over one (or a few) large cudaMalloc()s, used for
// network weight uploads at construction time.
//
// Problem this solves: lc0's AttentionBody / EncoderBlock ctors do
// ~25 cudaMalloc()s per encoder block × 60 blocks ≈ 1500 calls at
// network load.  Each cudaMalloc() costs ~0.5-2ms on Linux+CUDA due
// to the runtime allocator + device page-table updates → 1-3 seconds
// of pure allocator overhead before the first inference can start.
//
// WeightArena replaces those per-buffer cudaMalloc()s with a single
// (or a few) large allocation(s) backing all weight slots.  Each
// EncoderBlock buffer becomes a non-owning offset into the arena's
// big slab.  Destruction is a single cudaFree per chunk instead of
// per-buffer — cleaner shutdown too.
//
// Growth strategy: arena starts empty.  First Allocate() that exceeds
// the current chunk's remaining capacity creates a new chunk via
// cudaMalloc.  Default chunk size = 256 MB; if a single allocation
// is larger, we allocate exactly that size.  In practice, lc0's
// weights at 512x60 are ~490 MB → typically 2-3 chunks total.
//
// All allocations are 256-byte aligned (CUDA's preferred alignment
// for global-memory transactions on modern architectures).
//
// Thread-safety: Allocate() is NOT thread-safe.  Used during
// single-threaded network construction.  Once construction completes,
// the arena is read-only (the buffers it owns are accessed by many
// GPU streams but the arena itself isn't touched).
class WeightArena {
 public:
  // Default chunk size when growing the arena.  Sized to balance:
  //   - small enough that chunk-level fragmentation (last chunk
  //     usually half-empty) doesn't waste much VRAM
  //   - large enough that growth-cudaMalloc count stays low and the
  //     CUDA allocator's per-chunk bookkeeping is amortized
  //
  // 64 MB sweet spot for typical lc0 nets:
  //   - 540 MB net (512x60 fp16) → 9 chunks × 64 MB = 576 MB
  //     allocated, expected ~32 MB last-chunk waste (~6%), ~13 ms of
  //     cudaMalloc overhead at load.
  //   - 1 GB net → 16 chunks, ~32 MB waste (3.2%), ~24 ms load.
  //   - Smaller nets (<128 MB) still pack into 1-2 chunks fine.
  //
  // History: 256 MB original (too wasteful at our net sizes — ~25%
  // fragmentation), 128 MB intermediate (~12%), 64 MB current (~6%).
  // Drop further to 32 MB only if running multiple lc0 instances per
  // GPU or moving to a memory-tight platform.
  static constexpr size_t kDefaultChunkSize = 64 * 1024 * 1024;

  // Alignment requirement for all returned pointers.  256 bytes matches
  // CUDA's coalesced-access preference for fp16/fp32 global memory.
  static constexpr size_t kAlignment = 256;

  WeightArena() = default;
  ~WeightArena();

  // Non-copyable, non-movable (owns CUDA resources).
  WeightArena(const WeightArena&) = delete;
  WeightArena& operator=(const WeightArena&) = delete;
  WeightArena(WeightArena&&) = delete;
  WeightArena& operator=(WeightArena&&) = delete;

  // Allocate `bytes` of device memory from the arena.  Returns a 256-
  // byte-aligned device pointer.  Calling with bytes=0 returns nullptr.
  // Grows the arena (new cudaMalloc chunk) if the current chunk can't
  // satisfy the request.
  void* Allocate(size_t bytes);

  // Pre-reserve a chunk of `bytes` to avoid a growth-cudaMalloc during
  // construction.  Optional optimization — if the caller has a rough
  // estimate of total weight bytes, calling Reserve() once up-front
  // turns the typical "2-3 cudaMallocs during construction" pattern
  // into "1 cudaMalloc during construction".
  void Reserve(size_t bytes);

  // Total bytes allocated across all chunks.
  size_t TotalAllocated() const;

  // Total bytes actually handed out via Allocate() (excluding alignment
  // padding between slots).  Useful for telemetry.
  size_t BytesUsed() const { return bytes_used_; }

  // Number of cudaMalloc calls made.  Should typically be 1-3.
  size_t ChunkCount() const { return chunks_.size(); }

  // Returns true if `ptr` is a slice inside any chunk this arena owns.
  // Used by layer destructors to decide whether to skip cudaFree (the
  // arena will free the underlying chunk in its own destructor) vs
  // call cudaFree (the pointer came from a direct cudaMalloc, not the
  // arena).  Cost is O(chunks) per call; typical chunk count is 1-3.
  bool Owns(const void* ptr) const;

 private:
  struct Chunk {
    void* base;          // device pointer from cudaMalloc
    size_t capacity;     // bytes allocated
    size_t used;         // bytes handed out (bump position)
  };

  std::vector<Chunk> chunks_;
  // Total bytes returned by Allocate() across all calls.  Diverges
  // slightly from sum(chunks_[i].used) when alignment padding within
  // a chunk consumes capacity but isn't "used" in the accounting sense.
  size_t bytes_used_ = 0;

  // Add a new chunk of at least `min_bytes`, large enough to satisfy a
  // pending Allocate() call.  Chunk size = max(kDefaultChunkSize,
  // round_up(min_bytes, kAlignment)).
  void AddChunk(size_t min_bytes);
};

// Thread-local pointer set by `WeightArenaScope` (below) during the
// network-construction phase.  When non-null, `allocAndUpload` in
// layers.cc allocates from this arena instead of calling cudaMalloc.
// Layer constructors should also capture this pointer into a member if
// they want to be arena-aware (i.e., skip cudaFree in their destructor
// for buffers that came from the arena).  See WeightArena::Owns().
extern thread_local WeightArena* tl_weight_arena;

// RAII guard for setting the thread-local arena pointer.  Construct at
// the top of a layer-building phase, destruct at the end.  Restores
// the previous value of `tl_weight_arena` on scope exit (supports
// nested scopes correctly, though we don't currently nest).
class WeightArenaScope {
 public:
  explicit WeightArenaScope(WeightArena* arena) : prev_(tl_weight_arena) {
    tl_weight_arena = arena;
  }
  ~WeightArenaScope() { tl_weight_arena = prev_; }
  WeightArenaScope(const WeightArenaScope&) = delete;
  WeightArenaScope& operator=(const WeightArenaScope&) = delete;

 private:
  WeightArena* prev_;
};

}  // namespace cudnn_backend
}  // namespace lczero
