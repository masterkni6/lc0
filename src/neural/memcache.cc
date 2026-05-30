/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2025 The LCZero Authors

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

#include "neural/memcache.h"

#include "neural/shared_params.h"
#include "utils/atomic_vector.h"
#include "utils/cache.h"
#include "utils/smallarray.h"

namespace lczero {
namespace {

// TODO For now it uses the hash of the current position, ignoring repetitions
// and history. We'll likely need to have configurable hash function that we'll
// also reuse as a tree hash key.
uint64_t ComputeEvalPositionHash(const EvalPosition& pos) {
  return pos.pos.back().Hash();
}

struct CachedValue {
  float q;
  float d;
  float m;
  uint8_t num_moves;
  bool has_opt = false;
  // Combined storage for both policy distributions.  Layout:
  //   [0..num_moves)            — vanilla policy (p)
  //   [num_moves..2*num_moves)  — optimistic policy (when has_opt)
  // Optimization vs the previous two-pointer layout: only ONE heap
  // allocation per cache write instead of two.  Under blend mode in
  // production (selfplay parallelism × many workers), the allocator
  // hot-path was a measurable source of contention — halving the
  // alloc count cuts that contention proportionally.  No change to
  // bytes-per-entry; the optimistic half just lives contiguously
  // after the vanilla half in the same `new float[]`.
  std::unique_ptr<float[]> storage;

  // Accessors mirror the old field semantics:
  //   p()            — non-null when num_moves > 0
  //   p_optimistic() — non-null only when has_opt is true
  float* p() const { return storage.get(); }
  float* p_optimistic() const {
    return has_opt ? storage.get() + num_moves : nullptr;
  }
};

void CachedValueToEvalResult(const CachedValue& cv, const EvalResultPtr& ptr) {
  if (ptr.d) *ptr.d = cv.d;
  if (ptr.q) *ptr.q = cv.q;
  if (ptr.m) *ptr.m = cv.m;
  std::copy(cv.p(), cv.p() + ptr.p.size(), ptr.p.begin());
  // Copy optimistic policy too, when both sides are present.  If the
  // cached value has it but the caller didn't allocate space, skip
  // (no-op).  If the caller allocated but the cached value is missing
  // it, this is a cache-version mismatch — copy what we have for q/d/m
  // and let search's blend fall back via its !empty() guard.
  if (cv.p_optimistic() && !ptr.p_optimistic.empty()) {
    std::copy(cv.p_optimistic(),
              cv.p_optimistic() + ptr.p_optimistic.size(),
              ptr.p_optimistic.begin());
  }
}

class MemCache : public CachingBackend {
 public:
  MemCache(std::unique_ptr<Backend> wrapped, const OptionsDict& options)
      : wrapped_backend_(std::move(wrapped)),
        cache_(options.Get<int>(SharedBackendParams::kNNCacheSizeId)),
        max_batch_size_(wrapped_backend_->GetAttributes().maximum_batch_size) {}

  BackendAttributes GetAttributes() const override {
    return wrapped_backend_->GetAttributes();
  }
  std::unique_ptr<BackendComputation> CreateComputation() override;
  std::optional<EvalResult> GetCachedEvaluation(const EvalPosition&) override;

  void ClearCache() override { cache_.Clear(); }

  UpdateConfigurationResult UpdateConfiguration(
      const OptionsDict& options) override {
    auto ret = wrapped_backend_->UpdateConfiguration(options);
    if (ret == Backend::UPDATE_OK) {
      // Check if we need to clear the cache.
      if (!wrapped_backend_->IsSameConfiguration(options)) {
        cache_.Clear();
      }
    }
    return ret;
  }

  bool IsSameConfiguration(const OptionsDict& options) const override {
    return wrapped_backend_->IsSameConfiguration(options);
  }

  void SetCacheSize(size_t size) override { cache_.SetCapacity(size); }

 private:
  std::unique_ptr<Backend> wrapped_backend_;
  HashKeyedCache<CachedValue> cache_;
  const size_t max_batch_size_;
  friend class MemCacheComputation;
};

class MemCacheComputation : public BackendComputation {
 public:
  MemCacheComputation(std::unique_ptr<BackendComputation> wrapped_computation,
                      MemCache* memcache)
      : wrapped_computation_(std::move(wrapped_computation)),
        memcache_(memcache),
        entries_(memcache->max_batch_size_) {}

 private:
  size_t UsedBatchSize() const override {
    return wrapped_computation_->UsedBatchSize();
  }
  virtual AddInputResult AddInput(const EvalPosition& pos,
                                  EvalResultPtr result) override {
    assert(pos.legal_moves.size() == result.p.size() || result.p.empty());
    const uint64_t hash = ComputeEvalPositionHash(pos);
    {
      HashKeyedCacheLock<CachedValue> lock(&memcache_->cache_, hash);
      // Sometimes search queries NN without passing the legal moves. It is
      // still cached in this case, but in subsequent queries we only return it
      // if legal moves are not passed again. Otherwise check the size to guard
      // against hash collisions.
      //
      // Additional check: when the caller asked for p_optimistic (non-empty
      // span in the EvalResultPtr) but the cached value doesn't have it
      // (entry inserted before the blend was enabled), treat as a cache
      // miss and recompute.  Otherwise the search's p_optimistic span would
      // stay at zeros and the blend would silently fall through to vanilla.
      const bool caller_wants_opt = !result.p_optimistic.empty();
      const bool cached_has_opt = lock.holds_value() && lock->p_optimistic();
      if (lock.holds_value() &&
          (pos.legal_moves.empty() ||
           (lock->p() && lock->num_moves == pos.legal_moves.size())) &&
          (!caller_wants_opt || cached_has_opt)) {
        CachedValueToEvalResult(**lock, result);
        return AddInputResult::FETCHED_IMMEDIATELY;
      }
    }
    size_t entry_idx = entries_.emplace_back(
        Entry{hash, std::make_unique<CachedValue>(), result});
    auto& value = entries_[entry_idx].value;
    const size_t N = pos.legal_moves.size();
    // Allocate optimistic policy storage only when the caller asked
    // for it (signalled by passing a non-empty p_optimistic span).
    // Keeps cache size unchanged for runs that don't use the blend.
    const bool want_opt = !result.p_optimistic.empty() && N > 0;
    // Single combined allocation: 1×N for vanilla, +N more if optimistic.
    // Halves heap allocator calls vs the old two-pointer layout.
    if (N > 0) {
      value->storage.reset(new float[want_opt ? 2 * N : N]);
    }
    value->num_moves = static_cast<uint8_t>(N);
    value->has_opt = want_opt;
    return wrapped_computation_->AddInput(
        pos, EvalResultPtr{
                 &value->q, &value->d, &value->m,
                 value->p() ? std::span<float>{value->p(), N}
                            : std::span<float>{},
                 want_opt ? std::span<float>{value->p_optimistic(), N}
                          : std::span<float>{}});
  }

  virtual void ComputeBlocking() override {
    if (wrapped_computation_->UsedBatchSize() == 0) return;
    wrapped_computation_->ComputeBlocking();
    for (auto& entry : entries_) {
      CachedValueToEvalResult(*entry.value, entry.result_ptr);
      memcache_->cache_.Insert(entry.key, std::move(entry.value));
    }
  }

  struct Entry {
    uint64_t key;
    std::unique_ptr<CachedValue> value;
    EvalResultPtr result_ptr;
  };

  std::unique_ptr<BackendComputation> wrapped_computation_;
  MemCache* memcache_;
  AtomicVector<Entry> entries_;
};

std::unique_ptr<BackendComputation> MemCache::CreateComputation() {
  return std::make_unique<MemCacheComputation>(
      wrapped_backend_->CreateComputation(), this);
}
std::optional<EvalResult> MemCache::GetCachedEvaluation(
    const EvalPosition& pos) {
  const uint64_t hash = ComputeEvalPositionHash(pos);
  HashKeyedCacheLock<CachedValue> lock(&cache_, hash);
  if (!lock.holds_value() ||
      (!pos.legal_moves.empty() &&
       !(lock->p() && lock->num_moves == pos.legal_moves.size()))) {
    return std::nullopt;
  }
  EvalResult result;
  result.d = lock->d;
  result.q = lock->q;
  result.m = lock->m;
  if (lock->p()) {
    result.p.reserve(pos.legal_moves.size());
    std::copy(lock->p(), lock->p() + pos.legal_moves.size(),
              std::back_inserter(result.p));
  }
  // Copy optimistic policy through when cached.  Caller decides whether
  // to use it (by setting --optimistic-policy-weight* > 0 in search).
  if (lock->p_optimistic()) {
    result.p_optimistic.reserve(pos.legal_moves.size());
    std::copy(lock->p_optimistic(),
              lock->p_optimistic() + pos.legal_moves.size(),
              std::back_inserter(result.p_optimistic));
  }
  return result;
}

}  // namespace

std::unique_ptr<CachingBackend> CreateMemCache(std::unique_ptr<Backend> wrapped,
                                               const OptionsDict& options) {
  return std::make_unique<MemCache>(std::move(wrapped), options);
}

}  // namespace lczero
