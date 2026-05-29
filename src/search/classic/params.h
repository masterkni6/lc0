/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2025 The LCZero Authors

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

#include "neural/encoder.h"
#include "utils/optionsdict.h"
#include "utils/optionsparser.h"

namespace lczero {
namespace classic {

enum class ContemptMode { PLAY, WHITE, BLACK, NONE };

class BaseSearchParams {
 public:
  BaseSearchParams(const OptionsDict& options);
  BaseSearchParams(const BaseSearchParams&) = delete;

  // Use struct for WDLRescaleParams calculation to make them const.
  struct WDLRescaleParams {
    WDLRescaleParams(float r, float d) {
      ratio = r;
      diff = d;
    }
    float ratio;
    float diff;
  };

  // Populates UciOptions with search parameters.
  static void Populate(OptionsParser* options);

  // Parameter getters.
  int GetMiniBatchSize() const { return kMiniBatchSize; }
  float GetCpuct(bool at_root) const { return at_root ? kCpuctAtRoot : kCpuct; }
  float GetCpuctBase(bool at_root) const {
    return at_root ? kCpuctBaseAtRoot : kCpuctBase;
  }
  float GetCpuctFactor(bool at_root) const {
    return at_root ? kCpuctFactorAtRoot : kCpuctFactor;
  }
  bool GetTwoFoldDraws() const { return kTwoFoldDraws; }
  float GetTemperature() const { return options_.Get<float>(kTemperatureId); }
  float GetTemperatureVisitOffset() const {
    return options_.Get<float>(kTemperatureVisitOffsetId);
  }
  int GetTempDecayMoves() const { return options_.Get<int>(kTempDecayMovesId); }
  int GetTempDecayDelayMoves() const {
    return options_.Get<int>(kTempDecayDelayMovesId);
  }
  int GetTemperatureCutoffMove() const {
    return options_.Get<int>(kTemperatureCutoffMoveId);
  }
  float GetTemperatureEndgame() const {
    return options_.Get<float>(kTemperatureEndgameId);
  }
  float GetTemperatureWinpctCutoff() const {
    return options_.Get<float>(kTemperatureWinpctCutoffId);
  }
  float GetNoiseEpsilon() const { return kNoiseEpsilon; }
  float GetNoiseAlpha() const { return kNoiseAlpha; }
  bool GetVerboseStats() const { return options_.Get<bool>(kVerboseStatsId); }
  bool GetLogLiveStats() const { return options_.Get<bool>(kLogLiveStatsId); }
  bool GetFpuAbsolute(bool at_root) const {
    return at_root ? kFpuAbsoluteAtRoot : kFpuAbsolute;
  }
  float GetFpuValue(bool at_root) const {
    return at_root ? kFpuValueAtRoot : kFpuValue;
  }
  int GetCacheHistoryLength() const { return kCacheHistoryLength; }
  float GetPolicySoftmaxTemp() const { return kPolicySoftmaxTemp; }
  int GetMaxCollisionEvents() const { return kMaxCollisionEvents; }
  int GetMaxCollisionVisits() const { return kMaxCollisionVisits; }
  bool GetOutOfOrderEval() const { return kOutOfOrderEval; }
  bool GetStickyEndgames() const { return kStickyEndgames; }
  bool GetSyzygyFastPlay() const { return kSyzygyFastPlay; }
  int GetMultiPv() const { return options_.Get<int>(kMultiPvId); }
  bool GetPerPvCounters() const { return options_.Get<bool>(kPerPvCountersId); }
  std::string GetScoreType() const {
    return options_.Get<std::string>(kScoreTypeId);
  }
  FillEmptyHistory GetHistoryFill() const { return kHistoryFill; }
  float GetMovesLeftMaxEffect() const { return kMovesLeftMaxEffect; }
  float GetMovesLeftThreshold() const { return kMovesLeftThreshold; }
  float GetMovesLeftSlope() const { return kMovesLeftSlope; }
  float GetMovesLeftConstantFactor() const { return kMovesLeftConstantFactor; }
  float GetMovesLeftScaledFactor() const { return kMovesLeftScaledFactor; }
  float GetMovesLeftQuadraticFactor() const {
    return kMovesLeftQuadraticFactor;
  }
  int GetMaxConcurrentSearchers() const { return kMaxConcurrentSearchers; }
  float GetDrawScore() const { return kDrawScore; }
  ContemptMode GetContemptMode() const {
    std::string mode = options_.Get<std::string>(kContemptModeId);
    if (mode == "play") return ContemptMode::PLAY;
    if (mode == "white_side_analysis") return ContemptMode::WHITE;
    if (mode == "black_side_analysis") return ContemptMode::BLACK;
    assert(mode == "disable");
    return ContemptMode::NONE;
  }
  float GetWDLRescaleRatio() const { return kWDLRescaleParams.ratio; }
  float GetWDLRescaleDiff() const { return kWDLRescaleParams.diff; }
  float GetWDLMaxS() const { return kWDLMaxS; }
  float GetWDLEvalObjectivity() const { return kWDLEvalObjectivity; }
  float GetMaxOutOfOrderEvalsFactor() const {
    return kMaxOutOfOrderEvalsFactor;
  }
  float GetNpsLimit() const { return kNpsLimit; }

  int GetTaskWorkersPerSearchWorker() const {
    return kTaskWorkersPerSearchWorker;
  }
  int GetMinimumWorkSizeForProcessing() const {
    return kMinimumWorkSizeForProcessing;
  }
  int GetMinimumWorkSizeForPicking() const {
    return kMinimumWorkSizeForPicking;
  }
  int GetMinimumRemainingWorkSizeForPicking() const {
    return kMinimumRemainingWorkSizeForPicking;
  }
  int GetMinimumWorkPerTaskForProcessing() const {
    return kMinimumWorkPerTaskForProcessing;
  }
  int GetIdlingMinimumWork() const { return kIdlingMinimumWork; }
  int GetThreadIdlingThreshold() const { return kThreadIdlingThreshold; }
  int GetMaxCollisionVisitsScalingStart() const {
    return kMaxCollisionVisitsScalingStart;
  }
  int GetMaxCollisionVisitsScalingEnd() const {
    return kMaxCollisionVisitsScalingEnd;
  }
  float GetMaxCollisionVisitsScalingPower() const {
    return kMaxCollisionVisitsScalingPower;
  }
  bool GetSearchSpinBackoff() const { return kSearchSpinBackoff; }

  float GetGarbageCollectionDelay() const {
    return kGarbageCollectionDelay;
  }

  // Search parameter IDs.
  static const OptionId kMiniBatchSizeId;
  static const OptionId kCpuctId;
  static const OptionId kCpuctAtRootId;
  static const OptionId kCpuctBaseId;
  static const OptionId kCpuctBaseAtRootId;
  static const OptionId kCpuctFactorId;
  static const OptionId kCpuctFactorAtRootId;
  static const OptionId kRootHasOwnCpuctParamsId;
  static const OptionId kTwoFoldDrawsId;
  static const OptionId kTemperatureId;
  static const OptionId kTempDecayMovesId;
  static const OptionId kTempDecayDelayMovesId;
  static const OptionId kTemperatureCutoffMoveId;
  static const OptionId kTemperatureEndgameId;
  static const OptionId kTemperatureWinpctCutoffId;
  static const OptionId kTemperatureVisitOffsetId;
  static const OptionId kNoiseEpsilonId;
  static const OptionId kNoiseAlphaId;
  static const OptionId kVerboseStatsId;
  static const OptionId kLogLiveStatsId;
  static const OptionId kFpuStrategyId;
  static const OptionId kFpuValueId;
  static const OptionId kFpuStrategyAtRootId;
  static const OptionId kFpuValueAtRootId;
  static const OptionId kCacheHistoryLengthId;
  static const OptionId kMaxCollisionEventsId;
  static const OptionId kMaxCollisionVisitsId;
  static const OptionId kOutOfOrderEvalId;
  static const OptionId kStickyEndgamesId;
  static const OptionId kSyzygyFastPlayId;
  static const OptionId kMultiPvId;
  static const OptionId kPerPvCountersId;
  static const OptionId kScoreTypeId;
  static const OptionId kMovesLeftMaxEffectId;
  static const OptionId kMovesLeftThresholdId;
  static const OptionId kMovesLeftConstantFactorId;
  static const OptionId kMovesLeftScaledFactorId;
  static const OptionId kMovesLeftQuadraticFactorId;
  static const OptionId kMovesLeftSlopeId;
  static const OptionId kMaxConcurrentSearchersId;
  static const OptionId kDrawScoreId;
  static const OptionId kContemptModeId;
  static const OptionId kContemptId;
  static const OptionId kContemptMaxValueId;
  static const OptionId kWDLCalibrationEloId;
  static const OptionId kWDLContemptAttenuationId;
  static const OptionId kWDLMaxSId;
  static const OptionId kWDLEvalObjectivityId;
  static const OptionId kWDLDrawRateTargetId;
  static const OptionId kWDLDrawRateReferenceId;
  static const OptionId kWDLBookExitBiasId;
  static const OptionId kMaxOutOfOrderEvalsFactorId;
  static const OptionId kNpsLimitId;
  static const OptionId kTaskWorkersPerSearchWorkerId;
  static const OptionId kMinimumWorkSizeForProcessingId;
  static const OptionId kMinimumWorkSizeForPickingId;
  static const OptionId kMinimumRemainingWorkSizeForPickingId;
  static const OptionId kMinimumWorkPerTaskForProcessingId;
  static const OptionId kIdlingMinimumWorkId;
  static const OptionId kThreadIdlingThresholdId;
  static const OptionId kMaxCollisionVisitsScalingStartId;
  static const OptionId kMaxCollisionVisitsScalingEndId;
  static const OptionId kMaxCollisionVisitsScalingPowerId;
  static const OptionId kUCIOpponentId;
  static const OptionId kUCIRatingAdvId;
  static const OptionId kSearchSpinBackoffId;
  static const OptionId kGarbageCollectionDelayId;
  // KataGo-style forced exploration: each root edge gets at least
  // sqrt(P * N_total * factor) visits before PUCT is allowed to skip
  // it.  factor=0 disables.  Default is 0 (off) for backward compat;
  // selfplay tournament code sets factor=2.0 (KataGo's value).  See
  // GetForcedExplorationFactor() accessor + PUCT root override in
  // search.cc.  Paired with policy-target pruning at training-data
  // write time (see Search::GetTrainingTargetVisits) so the trained
  // policy doesn't get biased toward forcibly-explored moves.
  static const OptionId kForcedExplorationFactorId;

  float GetForcedExplorationFactor() const {
    return kForcedExplorationFactor;
  }

  // KataGo-style Policy Target Pruning at training-data write time —
  // clamps each non-best root edge's training-target visit count to the
  // PUCT-equilibrium count (target = min(n_raw, N_eq)).  When false
  // (default), PTP only runs implicitly when forced-exploration-factor>0
  // or an advisor is active (because their distortion needs the clamp).
  // When true, PTP runs even without those — useful if you want the
  // sharpened policy target without paying the per-game search-quality
  // cost of forced exploration.  See Search::GetTrainingTargetVisits.
  static const OptionId kUsePolicyTargetPruningId;

  bool GetUsePolicyTargetPruning() const {
    return kUsePolicyTargetPruning;
  }

  // ── Gumbel-MuZero policy improvement (ICLR 2022, Danihelka et al.) ──
  // Alternative recipe to AZ-style PUCT + visit-count training targets.
  // When on at root: replaces PUCT with sequential halving over Gumbel-
  // perturbed policy logits, then constructs the training target as the
  // softmax of `g + p + σ(q)` rather than the visit-count distribution.
  // Provable policy improvement at small visit budgets (designed for
  // ≤32 visits/move; we run at 250).  At root only — non-root nodes
  // still use plain PUCT.
  //
  // Parameters control:
  //   - whether the Gumbel-MuZero path runs at all (default off)
  //   - m: action subset size for sequential halving (default 16; chess
  //     typically has 25-40 legal root moves so 16 covers the meaningful
  //     candidates)
  //   - c_visit / c_scale: σ(q) transform constants from the paper.
  //     σ(q) = (c_visit + max_N_so_far) × c_scale × q where max_N_so_far
  //     is the most-visited child's N.  Paper recommends c_visit=50,
  //     c_scale=1.0.  Re-tunable for chess.
  //
  // Composes with the advisor mechanism: the advisor's recommended
  // move is force-included in the SH initial set (m-1 Gumbel-picked +
  // 1 advisor) and gets the same per-round visit treatment.  Composes
  // with neither forced exploration nor PTP — they're mutually
  // exclusive alternatives.  When gumbel-muzero is on, force-
  // exploration-factor and use-policy-target-pruning are ignored.
  static const OptionId kUseGumbelMuZeroId;
  static const OptionId kGumbelMuZeroMId;
  static const OptionId kGumbelMuZeroCVisitId;
  static const OptionId kGumbelMuZeroCScaleId;

  bool GetUseGumbelMuZero() const { return kUseGumbelMuZero; }
  int GetGumbelMuZeroM() const { return kGumbelMuZeroM; }
  float GetGumbelMuZeroCVisit() const { return kGumbelMuZeroCVisit; }
  float GetGumbelMuZeroCScale() const { return kGumbelMuZeroCScale; }

  // ── Gumbel-improved policy target (Phase 1.6 subset of Gumbel-MuZero) ──
  // Writes softmax(prior_logits + σ(q)) as the policy training target,
  // replacing visit-count distributions or PTP-clamped variants.  Search
  // remains plain PUCT (no Gumbel SH); only the chunk-write step changes.
  // Compose with --forced-exploration-factor (gives better Q coverage)
  // and advisor (broadens visited candidates) — both still run at search
  // time but no longer trigger auto-PTP (PTP is bypassed entirely under
  // this target).  Mutually exclusive with --use-policy-target-pruning
  // (both define the policy target; only one can apply).  See
  // Search::GetGumbelImprovedPolicyTarget() for the formula.
  //
  // sqrt-scaled confidence weighting on the σ(q) contribution per edge:
  //   conf = sqrt(N_edge) / sqrt(max_N_at_root)
  //   contribution = conf × visit_scale × rescaled_q
  // Edges with N=0 contribute zero σ (target = softmax(prior_logits)
  // for the unvisited slot).  Edges with few visits get damped σ.
  // Well-visited edges get full σ.  Eliminates the v-mix-overweights-
  // unvisited-moves failure mode under PUCT-driven search.
  static const OptionId kUseGumbelImprovedTargetId;
  bool GetUseGumbelImprovedTarget() const {
    return kUseGumbelImprovedTarget;
  }

  // ── Grill et al. (ICML 2020) improved-policy training target ──
  // Writes π̄(a) = λ_N · prior(a) / (α − q(a)) as the policy training
  // target, replacing visit counts or other alternatives.  λ_N = c /
  // √(total_visits) is the regularization strength — strong at low N
  // (target stays near prior), weak at high N (target shifts toward
  // Q-greedy).  α is solved via dichotomic search so π̄ sums to 1.
  // See Search::GetGrillImprovedPolicyTarget() for the formula + Grill
  // Appendix B.3 Proposition 4 for the α-search bounds.
  //
  // Why this over Gumbel improved target: Gumbel's σ(q) scales UP with
  // max_N (sharper at high N).  Grill's λ_N scales DOWN with total_N
  // (sharper only when search is well-converged).  At 250-visit
  // budgets, Gumbel over-sharpened tail policy; Grill's inverse-sqrt
  // regularization keeps the target close to prior when uncertain.
  //
  // Mutually exclusive with --use-gumbel-improved-target AND with
  // --use-policy-target-pruning (all three define the policy target;
  // only one can apply).  Caught at SearchParams ctor.
  static const OptionId kUseGrillImprovedTargetId;
  static const OptionId kGrillCId;
  bool GetUseGrillImprovedTarget() const {
    return kUseGrillImprovedTarget;
  }
  float GetGrillC() const { return kGrillC; }

  // ── Optimistic policy blend (KataGo v1.13-style) ──
  // Mixes the trained optimistic-st policy head into the root prior:
  //   P_blended(a) = (1 - alpha) * P_main(a) + alpha * P_opt(a)
  // applied at root only, BEFORE Dirichlet noise.
  //
  // Requires: net has policy_optimistic_st head AND the backend exposes
  // it as EvalResult::p_optimistic.  If either is missing the blend
  // silently degrades to no-op (search.cc falls back to main prior).
  //
  // Recommended range 0.05–0.30.  Higher = more tactical exploration
  // bias; KataGo reports +40–90 Elo with values in this range.
  // Default 0.0 = feature off, no behavior change.
  static const OptionId kOptimisticPolicyWeightId;
  float GetOptimisticPolicyWeight() const {
    return kOptimisticPolicyWeight;
  }
  // Internal-node alpha for the optimistic-policy blend.  KataGo's
  // recommended setting is asymmetric: rootPolicyOptimism = 0.2,
  // policyOptimism (internal) = 1.0.  Internal nodes have much less
  // visit budget per edge, so the prior dominates exploration there;
  // a high internal alpha biases tactical discovery deep in the tree
  // without disrupting root-level decisions.
  static const OptionId kOptimisticPolicyWeightInternalId;
  float GetOptimisticPolicyWeightInternal() const {
    return kOptimisticPolicyWeightInternal;
  }

 protected:
  const OptionsDict& options_;

 private:
  // Cached parameter values. Values have to be cached if either:
  // 1. Parameter is accessed often and has to be cached for performance
  // reasons.
  // 2. Parameter has to stay the same during the search.
  // TODO(crem) Some of those parameters can be converted to be dynamic after
  //            trivial search optimizations.
  const float kCpuct;
  const float kCpuctAtRoot;
  const float kCpuctBase;
  const float kCpuctBaseAtRoot;
  const float kCpuctFactor;
  const float kCpuctFactorAtRoot;
  const float kForcedExplorationFactor;
  const bool kUsePolicyTargetPruning;
  const bool kUseGumbelMuZero;
  const int kGumbelMuZeroM;
  const float kGumbelMuZeroCVisit;
  const float kGumbelMuZeroCScale;
  const bool kUseGumbelImprovedTarget;
  const bool kUseGrillImprovedTarget;
  const float kGrillC;
  const float kOptimisticPolicyWeight;
  const float kOptimisticPolicyWeightInternal;
  const bool kTwoFoldDraws;
  const float kNoiseEpsilon;
  const float kNoiseAlpha;
  const bool kFpuAbsolute;
  const float kFpuValue;
  const bool kFpuAbsoluteAtRoot;
  const float kFpuValueAtRoot;
  const int kCacheHistoryLength;
  const float kPolicySoftmaxTemp;
  const int kMaxCollisionEvents;
  const int kMaxCollisionVisits;
  const bool kOutOfOrderEval;
  const bool kStickyEndgames;
  const bool kSyzygyFastPlay;
  const FillEmptyHistory kHistoryFill;
  const int kMiniBatchSize;
  const float kMovesLeftMaxEffect;
  const float kMovesLeftThreshold;
  const float kMovesLeftSlope;
  const float kMovesLeftConstantFactor;
  const float kMovesLeftScaledFactor;
  const float kMovesLeftQuadraticFactor;
  const int kMaxConcurrentSearchers;
  const float kDrawScore;
  const float kContempt;
  const WDLRescaleParams kWDLRescaleParams;
  const float kWDLMaxS;
  const float kWDLEvalObjectivity;
  const float kMaxOutOfOrderEvalsFactor;
  const float kNpsLimit;
  const int kTaskWorkersPerSearchWorker;
  const int kMinimumWorkSizeForProcessing;
  const int kMinimumWorkSizeForPicking;
  const int kMinimumRemainingWorkSizeForPicking;
  const int kMinimumWorkPerTaskForProcessing;
  const int kIdlingMinimumWork;
  const int kThreadIdlingThreshold;
  const int kMaxCollisionVisitsScalingStart;
  const int kMaxCollisionVisitsScalingEnd;
  const float kMaxCollisionVisitsScalingPower;
  const bool kSearchSpinBackoff;
  const float kGarbageCollectionDelay;
};

class SearchParams : public BaseSearchParams {
 public:
  SearchParams(const OptionsDict& options);
  SearchParams(const SearchParams&) = delete;

  // Populates UciOptions with search parameters.
  static void Populate(OptionsParser* options);

  // Parameter getters.
  int GetMaxPrefetchBatch() const {
    return options_.Get<int>(kMaxPrefetchBatchId);
  }
  int GetSolidTreeThreshold() const { return kSolidTreeThreshold; }

  // Search parameter IDs.
  static const OptionId kMaxPrefetchBatchId;
  static const OptionId kSolidTreeThresholdId;

 private:
  const int kSolidTreeThreshold;
};
}  // namespace classic
}  // namespace lczero
