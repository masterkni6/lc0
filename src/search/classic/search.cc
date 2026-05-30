/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2023 The LCZero Authors

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

#include "search/classic/search.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <sstream>
#include <thread>

#include "neural/encoder.h"
#include "search/classic/node.h"
#include "utils/fastmath.h"
#include "utils/random.h"
#include "utils/spinhelper.h"
#include "utils/trace.h"

namespace lczero {
namespace classic {

namespace {
// Maximum delay between outputting "uci info" when nothing interesting happens.
const int kUciInfoMinimumFrequencyMs = 5000;

MoveList MakeRootMoveFilter(const MoveList& searchmoves,
                            SyzygyTablebase* syzygy_tb,
                            const PositionHistory& history, bool fast_play,
                            std::atomic<int>* tb_hits, bool* dtz_success) {
  assert(tb_hits);
  assert(dtz_success);
  // Search moves overrides tablebase.
  if (!searchmoves.empty()) return searchmoves;
  const auto& board = history.Last().GetBoard();
  MoveList root_moves;
  if (!syzygy_tb || !board.castlings().no_legal_castle() ||
      (board.ours() | board.theirs()).count() > syzygy_tb->max_cardinality()) {
    return root_moves;
  }
  if (syzygy_tb->root_probe(
          history.Last(), fast_play || history.DidRepeatSinceLastZeroingMove(),
          false, &root_moves)) {
    *dtz_success = true;
    tb_hits->fetch_add(1, std::memory_order_acq_rel);
  } else if (syzygy_tb->root_probe_wdl(history.Last(), &root_moves)) {
    tb_hits->fetch_add(1, std::memory_order_acq_rel);
  }
  return root_moves;
}

class MEvaluator {
 public:
  MEvaluator()
      : enabled_{false},
        m_slope_{0.0f},
        m_cap_{0.0f},
        a_constant_{0.0f},
        a_linear_{0.0f},
        a_square_{0.0f},
        q_threshold_{0.0f},
        parent_m_{0.0f} {}

  MEvaluator(const SearchParams& params, const Node* parent = nullptr)
      : enabled_{true},
        m_slope_{params.GetMovesLeftSlope()},
        m_cap_{params.GetMovesLeftMaxEffect()},
        a_constant_{params.GetMovesLeftConstantFactor()},
        a_linear_{params.GetMovesLeftScaledFactor()},
        a_square_{params.GetMovesLeftQuadraticFactor()},
        q_threshold_{params.GetMovesLeftThreshold()},
        parent_m_{parent ? parent->GetM() : 0.0f},
        parent_within_threshold_{parent ? WithinThreshold(parent, q_threshold_)
                                        : false} {}

  void SetParent(const Node* parent) {
    assert(parent);
    if (enabled_) {
      parent_m_ = parent->GetM();
      parent_within_threshold_ = WithinThreshold(parent, q_threshold_);
    }
  }

  // Calculates the utility for favoring shorter wins and longer losses.
  float GetMUtility(Node* child, float q) const {
    if (!enabled_ || !parent_within_threshold_) return 0.0f;
    const float child_m = child->GetM();
    float m = std::clamp(m_slope_ * (child_m - parent_m_), -m_cap_, m_cap_);
    m *= FastSign(-q);
    if (q_threshold_ > 0.0f && q_threshold_ < 1.0f) {
      // This allows a smooth M effect with higher q thresholds, which is
      // necessary for using MLH together with contempt.
      q = std::max(0.0f, (std::abs(q) - q_threshold_)) / (1.0f - q_threshold_);
    }
    m *= a_constant_ + a_linear_ * std::abs(q) + a_square_ * q * q;
    return m;
  }

  float GetMUtility(const EdgeAndNode& child, float q) const {
    if (!enabled_ || !parent_within_threshold_) return 0.0f;
    if (child.GetN() == 0) return GetDefaultMUtility();
    return GetMUtility(child.node(), q);
  }

  // The M utility to use for unvisited nodes.
  float GetDefaultMUtility() const { return 0.0f; }

 private:
  static bool WithinThreshold(const Node* parent, float q_threshold) {
    return std::abs(parent->GetQ(0.0f)) > q_threshold;
  }

  const bool enabled_;
  const float m_slope_;
  const float m_cap_;
  const float a_constant_;
  const float a_linear_;
  const float a_square_;
  const float q_threshold_;
  float parent_m_ = 0.0f;
  bool parent_within_threshold_ = false;
};

}  // namespace

Search::Search(const NodeTree& tree, Backend* backend,
               std::unique_ptr<UciResponder> uci_responder,
               const MoveList& searchmoves,
               std::chrono::steady_clock::time_point start_time,
               std::unique_ptr<SearchStopper> stopper, bool infinite,
               bool ponder, const OptionsDict& options,
               SyzygyTablebase* syzygy_tb)
    : ok_to_respond_bestmove_(!infinite && !ponder),
      stopper_(std::move(stopper)),
      root_node_(tree.GetCurrentHead()),
      syzygy_tb_(syzygy_tb),
      played_history_(tree.GetPositionHistory()),
      backend_(backend),
      backend_attributes_(backend->GetAttributes()),
      params_(options),
      searchmoves_(searchmoves),
      start_time_(start_time),
      initial_visits_(root_node_->GetN()),
      root_move_filter_(MakeRootMoveFilter(
          searchmoves_, syzygy_tb_, played_history_,
          params_.GetSyzygyFastPlay(), &tb_hits_, &root_is_in_dtz_)),
      uci_responder_(std::move(uci_responder)) {
  if (params_.GetMaxConcurrentSearchers() != 0) {
    pending_searchers_.store(params_.GetMaxConcurrentSearchers(),
                             std::memory_order_release);
  }
  contempt_mode_ = params_.GetContemptMode();
  // Make sure the contempt mode is never "play" beyond this point.
  if (contempt_mode_ == ContemptMode::PLAY) {
    if (infinite) {
      // For infinite search disable contempt, only "white"/"black" make sense.
      contempt_mode_ = ContemptMode::NONE;
      // Issue a warning only if contempt mode would have an effect.
      if (params_.GetWDLRescaleDiff() != 0.0f) {
        std::vector<ThinkingInfo> info(1);
        info.back().comment =
            "WARNING: Contempt mode set to 'disable' as 'play' not supported "
            "for infinite search.";
        uci_responder_->OutputThinkingInfo(&info);
      }
    } else {
      // Otherwise set it to the root move's side, unless pondering.
      contempt_mode_ = played_history_.IsBlackToMove() != ponder
                           ? ContemptMode::BLACK
                           : ContemptMode::WHITE;
    }
  }
}

namespace {
void ApplyDirichletNoise(Node* node, float eps, double alpha) {
  float total = 0;
  std::vector<float> noise;

  for (int i = 0; i < node->GetNumEdges(); ++i) {
    float eta = Random::Get().GetGamma(alpha, 1.0);
    noise.emplace_back(eta);
    total += eta;
  }

  if (total < std::numeric_limits<float>::min()) return;

  int noise_idx = 0;
  for (const auto& child : node->Edges()) {
    auto* edge = child.edge();
    edge->SetP(edge->GetP() * (1 - eps) + eps * noise[noise_idx++] / total);
  }
}
}  // namespace

namespace {
// WDL conversion formula based on random walk model.
inline double WDLRescale(float& v, float& d, float wdl_rescale_ratio,
                         float wdl_rescale_diff, float sign, bool invert,
                         float max_reasonable_s) {
  if (invert) {
    wdl_rescale_diff = -wdl_rescale_diff;
    wdl_rescale_ratio = 1.0f / wdl_rescale_ratio;
  }
  auto w = (1 + v - d) / 2;
  auto l = (1 - v - d) / 2;
  // Safeguard against numerical issues; skip WDL transformation if WDL is too
  // extreme.
  const float eps = 0.0001f;
  if (w > eps && d > eps && l > eps && w < (1.0f - eps) && d < (1.0f - eps) &&
      l < (1.0f - eps)) {
    auto a = FastLog(1 / l - 1);
    auto b = FastLog(1 / w - 1);
    auto s = 2 / (a + b);
    // Safeguard against unrealistically broad WDL distributions coming from
    // the NN. Originally hardcoded, made into a parameter for piece odds.
    if (!invert) s = std::min(max_reasonable_s, s);
    auto mu = (a - b) / (a + b);
    auto s_new = s * wdl_rescale_ratio;
    if (invert) {
      std::swap(s, s_new);
      s = std::min(max_reasonable_s, s);
    }
    auto mu_new = mu + sign * s * s * wdl_rescale_diff;
    auto w_new = FastLogistic((-1.0f + mu_new) / s_new);
    auto l_new = FastLogistic((-1.0f - mu_new) / s_new);
    v = w_new - l_new;
    d = std::max(0.0f, 1.0f - w_new - l_new);
    return mu_new;
  }
  return 0;
}
}  // namespace

void Search::SendUciInfo() REQUIRES(nodes_mutex_) REQUIRES(counters_mutex_) {
  const auto max_pv = params_.GetMultiPv();
  const auto edges = GetBestChildrenNoTemperature(root_node_, max_pv, 0);
  const auto score_type = params_.GetScoreType();
  const auto per_pv_counters = params_.GetPerPvCounters();
  const auto draw_score = GetDrawScore(false);

  std::vector<ThinkingInfo> uci_infos;

  // Info common for all multipv variants.
  ThinkingInfo common_info;
  common_info.depth = cum_depth_ / (total_playouts_ ? total_playouts_ : 1);
  common_info.seldepth = max_depth_;
  common_info.time = GetTimeSinceStart();
  if (!per_pv_counters) {
    common_info.nodes = total_playouts_ + initial_visits_;
  }
  if (nps_start_time_) {
    const auto time_since_first_batch_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - *nps_start_time_)
            .count();
    if (time_since_first_batch_ms > 0) {
      common_info.nps = total_playouts_ * 1000 / time_since_first_batch_ms;
      common_info.eps = network_evaluations_ * 1000 / time_since_first_batch_ms;
    }
  }
  common_info.tb_hits = tb_hits_.load(std::memory_order_acquire);

  int multipv = 0;
  const auto default_q = -root_node_->GetQ(-draw_score);
  const auto default_wl = -root_node_->GetWL();
  const auto default_d = root_node_->GetD();
  for (const auto& edge : edges) {
    ++multipv;
    uci_infos.emplace_back(common_info);
    auto& uci_info = uci_infos.back();
    auto wl = edge.GetWL(default_wl);
    auto d = edge.GetD(default_d);
    float mu_uci = 0.0f;
    if (score_type == "WDL_mu" || (params_.GetWDLRescaleDiff() != 0.0f &&
                                   contempt_mode_ != ContemptMode::NONE)) {
      auto sign = ((contempt_mode_ == ContemptMode::BLACK) ==
                   played_history_.IsBlackToMove())
                      ? 1.0f
                      : -1.0f;
      mu_uci = WDLRescale(
          wl, d, params_.GetWDLRescaleRatio(),
          contempt_mode_ == ContemptMode::NONE
              ? 0
              : params_.GetWDLRescaleDiff() * params_.GetWDLEvalObjectivity(),
          sign, true, params_.GetWDLMaxS());
    }
    const auto q = edge.GetQ(default_q, draw_score);
    if (edge.IsTerminal() && wl != 0.0f) {
      uci_info.mate = std::copysign(
          std::round(edge.GetM(0.0f)) / 2 + (edge.IsTbTerminal() ? 101 : 1),
          wl);
    } else if (score_type == "centipawn_with_drawscore") {
      uci_info.score = 90 * tan(1.5637541897 * q);
    } else if (score_type == "centipawn") {
      uci_info.score = 90 * tan(1.5637541897 * wl);
    } else if (score_type == "centipawn_2019") {
      uci_info.score = 295 * wl / (1 - 0.976953126 * std::pow(wl, 14));
    } else if (score_type == "centipawn_2018") {
      uci_info.score = 290.680623072 * tan(1.548090806 * wl);
    } else if (score_type == "win_percentage") {
      uci_info.score = wl * 5000 + 5000;
    } else if (score_type == "Q") {
      uci_info.score = q * 10000;
    } else if (score_type == "W-L") {
      uci_info.score = wl * 10000;
    } else if (score_type == "WDL_mu") {
      // Reports the WDL mu value whenever it is reasonable, and defaults to
      // centipawn otherwise.
      const float centipawn_fallback_threshold = 0.996f;
      float centipawn_score = 45 * tan(1.56728071628 * wl);
      uci_info.score =
          backend_attributes_.has_wdl && mu_uci != 0.0f &&
                  std::abs(wl) + d < centipawn_fallback_threshold &&
                  (std::abs(mu_uci) < 1.0f ||
                   std::abs(centipawn_score) < std::abs(100 * mu_uci))
              ? 100 * mu_uci
              : centipawn_score;
    }

    auto wdl_w =
        std::max(0, static_cast<int>(std::round(500.0 * (1.0 + wl - d))));
    auto wdl_l =
        std::max(0, static_cast<int>(std::round(500.0 * (1.0 - wl - d))));
    // Using 1000-w-l so that W+D+L add up to 1000.0.
    auto wdl_d = 1000 - wdl_w - wdl_l;
    if (wdl_d < 0) {
      wdl_w = std::min(1000, std::max(0, wdl_w + wdl_d / 2));
      wdl_l = 1000 - wdl_w;
      wdl_d = 0;
    }
    uci_info.wdl = ThinkingInfo::WDL{wdl_w, wdl_d, wdl_l};
    if (backend_attributes_.has_mlh) {
      uci_info.moves_left = static_cast<int>(
          (1.0f + edge.GetM(1.0f + root_node_->GetM())) / 2.0f);
    }
    if (max_pv > 1) uci_info.multipv = multipv;
    if (per_pv_counters) uci_info.nodes = edge.GetN();
    bool flip = played_history_.IsBlackToMove();
    int depth = 0;
    for (auto iter = edge; iter;
         iter = GetBestChildNoTemperature(iter.node(), depth), flip = !flip) {
      uci_info.pv.push_back(iter.GetMove(flip));
      if (!iter.node()) break;  // Last edge was dangling, cannot continue.
      depth += 1;
    }
  }

  if (!uci_infos.empty()) last_outputted_uci_info_ = uci_infos.front();
  if (current_best_edge_ && !edges.empty()) {
    last_outputted_info_edge_ = current_best_edge_.edge();
  }

  uci_responder_->OutputThinkingInfo(&uci_infos);
}

// Decides whether anything important changed in stats and new info should be
// shown to a user.
void Search::MaybeOutputInfo() {
  SharedMutex::Lock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  if (!bestmove_is_sent_ && current_best_edge_ &&
      (current_best_edge_.edge() != last_outputted_info_edge_ ||
       last_outputted_uci_info_.depth !=
           static_cast<int>(cum_depth_ /
                            (total_playouts_ ? total_playouts_ : 1)) ||
       last_outputted_uci_info_.seldepth != max_depth_ ||
       last_outputted_uci_info_.time + kUciInfoMinimumFrequencyMs <
           GetTimeSinceStart())) {
    SendUciInfo();
    if (params_.GetLogLiveStats()) {
      SendMovesStats();
    }
    if (stop_.load(std::memory_order_acquire) && !ok_to_respond_bestmove_) {
      std::vector<ThinkingInfo> info(1);
      info.back().comment =
          "WARNING: Search has reached limit and does not make any progress.";
      uci_responder_->OutputThinkingInfo(&info);
    }
  }
}

int64_t Search::GetTimeSinceStart() const {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - start_time_)
      .count();
}

int64_t Search::GetTimeSinceFirstBatch() const REQUIRES(counters_mutex_) {
  if (!nps_start_time_) return 0;
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now() - *nps_start_time_)
      .count();
}

// Root is depth 0, i.e. even depth.
float Search::GetDrawScore(bool is_odd_depth) const {
  return (is_odd_depth == played_history_.IsBlackToMove()
              ? params_.GetDrawScore()
              : -params_.GetDrawScore());
}

namespace {
inline float GetFpu(const SearchParams& params, const Node* node, bool is_root_node,
                    float draw_score) {
  const auto value = params.GetFpuValue(is_root_node);
  return params.GetFpuAbsolute(is_root_node)
             ? value
             : -node->GetQ(-draw_score) -
                   value * std::sqrt(node->GetVisitedPolicy());
}

// Faster version for if visited_policy is readily available already.
inline float GetFpu(const SearchParams& params, const Node* node, bool is_root_node,
                    float draw_score, float visited_pol) {
  const auto value = params.GetFpuValue(is_root_node);
  return params.GetFpuAbsolute(is_root_node)
             ? value
             : -node->GetQ(-draw_score) - value * std::sqrt(visited_pol);
}

inline float ComputeCpuct(const SearchParams& params, uint32_t N,
                          bool is_root_node) {
  const float init = params.GetCpuct(is_root_node);
  const float k = params.GetCpuctFactor(is_root_node);
  const float base = params.GetCpuctBase(is_root_node);
  return init + (k ? k * FastLog((N + base) / base) : 0.0f);
}

// KataGo-style forced exploration quota (Wu 2019, §5.1).
// Returns the minimum visit count this edge should accumulate before
// PUCT may stop visiting it.
inline float GetForcedExploration(float policy, float total_visits_with_vl,
                                  float factor) {
  return std::sqrt(policy * total_visits_with_vl * factor);
}

// Dispatch forced visits to a single root edge if it's under quota,
// per PR 2415's adaptation of KataGo's recipe to lc0's batched +
// multi-threaded MCTS.  Differences from the paper:
//   - Uses `nstarted` (= N + n_in_flight) rather than `N`.  Accounts
//     for visits in flight via virtual loss across worker threads.
//   - Uses parent's GetChildrenVisits + GetNInFlight (parent's total
//     "started" work) in the formula, not just GetChildrenVisits.
//   - Caller must guard against IsTerminal() and against
//     `cur_iters[idx].GetN() == 0` (delay forcing until the edge has
//     received its first natural PUCT visit) before calling this.
//
// Composable with an external advisor: when this edge happens to be
// the advisor's recommended move, the minimum is max'd with
// advisor_min_visits.  When advisor isn't applicable or disabled,
// pass advisor_min_visits = 0.
//
// Mutates `cur_limit`, `nstarted`, and the child node's n_in_flight.
// Returns the number of visits dispatched (which the caller adds to
// visits_to_perform[idx]).
template <typename Iter>
int AddForcedExploration(const SearchParams& params, Node* node,
                         Iter& iter, int& cur_limit, float policy,
                         int& nstarted, Move advisor_move,
                         int advisor_min_visits) {
  const float factor = params.GetForcedExplorationFactor();
  int minimum_visits = 0;
  if (factor > 0.0f) {
    minimum_visits = static_cast<int>(GetForcedExploration(
        policy,
        static_cast<float>(node->GetChildrenVisits() + node->GetNInFlight()),
        factor));
  }
  // Advisor floor: if this edge matches the advisor's recommendation,
  // ensure at least advisor_min_visits.  Disabled when 0.
  if (advisor_min_visits > 0 && iter.GetMove() == advisor_move) {
    minimum_visits = std::max(minimum_visits, advisor_min_visits);
  }
  if (nstarted >= minimum_visits) return 0;

  Node* child_node = iter.GetOrSpawnNode(node);
  const int new_visits = std::min(minimum_visits - nstarted, cur_limit);
  cur_limit -= new_visits;
  nstarted += new_visits;
  child_node->IncrementNInFlight(new_visits);
  return new_visits;
}
}  // namespace

// Ignore the last tuple element when sorting in GetVerboseStats
static bool operator<(const EdgeAndNode&, const EdgeAndNode&) { return false; }

std::vector<float> Search::GetTrainingTargetVisits() const {
  // Policy Target Pruning (KataGo Wu 2019 §5.1; lc0 adaptation per PR 2415).
  //
  // Cap: PUCT(c*) using FULL utility = Q + M + U  (not just Q + U).  M
  // is the moves-left contribution from the MLH head; PR 2415 includes
  // it because lc0's full PUCT score is Q + M + U.  Without M the cap
  // would be inconsistent with what PUCT actually used to select edges.
  //
  // Algorithm:
  //   1. c* = argmax_N over root edges (most-visited, NOT highest-utility).
  //   2. best_utility = Q(c*) + M(c*) + U(c*) at c*'s actual N.
  //   3. For each non-c* child c:
  //        n_forced(c) = sqrt(P(c) * N_total * factor)
  //                    + advisor floor if c is advisor's move
  //        If Q(c) + M(c) + U(c at current N) >= best_utility, keep all
  //        visits (PUCT would have visited c at least this much anyway).
  //        Otherwise: N_eq = U_coeff * P(c) / (best_utility - (Q+M)(c)) - 1
  //        target(c) = max(N_eq, n_raw(c) - n_forced(c))
  //        target(c) = clamp(target(c), 0, n_raw(c)).
  //   4. c* keeps all its visits.
  //
  // Why c* by N: anchoring to "the move PUCT consensus has selected
  // most often" is more robust than highest-Q (a low-policy edge with
  // few visits but transiently high Q would dethrone the real best).
  std::vector<float> result;
  if (root_node_ == nullptr) return result;
  result.reserve(root_node_->GetNumEdges());

  // Fast path: no pruning machinery enabled → just return raw N.
  // Pruning runs when ANY of the following is true:
  //   - forced exploration is active (factor > 0): pruning removes the
  //     forced-visit distortion from the training target
  //   - advisor is active (advisor_min_visits > 0): pruning removes the
  //     advisor-forced-visit distortion (same mechanism)
  //   - UsePolicyTargetPruning is explicitly enabled: pruning runs even
  //     without those, to clamp PUCT early-exploration noise.  This is
  //     KataGo's actual implementation behavior — their PTP clamp is
  //     unconditional, not gated on forced visits being on.
  const float factor = params_.GetForcedExplorationFactor();
  const bool ptp_explicit = params_.GetUsePolicyTargetPruning();
  const bool any_prune =
      (factor > 0.0f) || (advisor_min_visits_ > 0) || ptp_explicit;
  if (!any_prune) {
    for (const auto& edge : root_node_->Edges()) {
      result.push_back(static_cast<float>(edge.GetN()));
    }
    return result;
  }

  // Pre-compute the Q/M/U scaffolding for the root.
  const bool is_root = true;
  const bool is_odd_depth = !is_root;  // root depth is even
  const float draw_score = GetDrawScore(is_odd_depth);
  const float cpuct = ComputeCpuct(params_, root_node_->GetN(), is_root);
  const float n_total_for_uct =
      std::sqrt(std::max(root_node_->GetChildrenVisits(), 1u));
  const float U_coeff = cpuct * n_total_for_uct;
  // FPU uses the 4-arg form which queries GetVisitedPolicy internally —
  // same as what the PUCT inner loop uses, so c's Q values match what
  // PUCT saw during search.
  const float fpu = GetFpu(params_, root_node_, is_root, draw_score);
  const auto m_evaluator = backend_attributes_.has_mlh
                               ? MEvaluator(params_, root_node_)
                               : MEvaluator();

  // Step 1: find c* (child with most playouts).
  int c_star_idx = -1;
  uint32_t c_star_n = 0;
  int idx = 0;
  for (const auto& edge : root_node_->Edges()) {
    const uint32_t n = edge.GetN();
    if (n > c_star_n) {
      c_star_n = n;
      c_star_idx = idx;
    }
    ++idx;
  }
  // No visits at all → return raw counts.
  if (c_star_idx < 0) {
    for (const auto& edge : root_node_->Edges()) {
      result.push_back(static_cast<float>(edge.GetN()));
    }
    return result;
  }

  // Step 2: compute best_utility = Q(c*) + M(c*) + U(c*) at c*'s N.
  float best_utility = 0.0f;
  Move advisor_move = advisor_move_;
  idx = 0;
  for (const auto& edge : root_node_->Edges()) {
    if (idx == c_star_idx) {
      const float q_star = edge.GetQ(fpu, draw_score);
      const float m_star = m_evaluator.GetMUtility(edge, q_star);
      const float u_star =
          U_coeff * edge.GetP() / (1.0f + static_cast<float>(c_star_n));
      best_utility = q_star + m_star + u_star;
      break;
    }
    ++idx;
  }

  // Step 3 + 4: per-edge pruning.
  idx = 0;
  for (const auto& edge : root_node_->Edges()) {
    const uint32_t n_raw_u = edge.GetN();
    const float n_raw = static_cast<float>(n_raw_u);

    if (idx == c_star_idx) {
      result.push_back(n_raw);  // c* keeps all visits
      ++idx;
      continue;
    }
    if (n_raw_u == 0) {
      result.push_back(0.0f);
      ++idx;
      continue;
    }

    const float p = edge.GetP();
    const float q = edge.GetQ(fpu, draw_score);
    const float m = m_evaluator.GetMUtility(edge, q);
    const float u_now =
        U_coeff * p / (1.0f + n_raw);
    const float qm = q + m;  // utility for non-best

    // Early-out: c's full PUCT (Q + M + U) already >= best_utility →
    // PUCT would have naturally visited c at least this many times,
    // so n_raw <= N_eq and the clamp below wouldn't reduce anything.
    // Skip the math.
    if (qm + u_now >= best_utility) {
      result.push_back(n_raw);
      ++idx;
      continue;
    }

    // N_eq: where PUCT(c at N_eq) = best_utility.
    //   qm + U_coeff * p / (1 + N_eq) = best_utility
    //   N_eq = U_coeff * p / (best_utility - qm) - 1
    const float n_eq = U_coeff * p / (best_utility - qm) - 1.0f;

    // KataGo-style clamp: if actual visits exceeded PUCT-equilibrium,
    // clamp down to equilibrium.  This is the actual KataGo
    // implementation (getReducedPlaySelectionWeight: "if childWeight >
    // childWeightWeRetrospectivelyWanted return childWeightWeRetro...").
    // The paper's "subtract n_forced" framing is equivalent to this
    // when forced visits inflated n_raw above equilibrium, but the
    // implementation is the more general clamp.
    // Operates whenever any_prune is true — so this fires for any of
    // the three triggers (forced-exploration, advisor, or explicit
    // UsePolicyTargetPruning).  When n_raw <= N_eq the early-out above
    // already short-circuited; in the fallthrough we know n_raw > N_eq
    // and the clamp meaningfully reduces visits.
    float target = std::max(0.0f, std::min(n_eq, n_raw));
    result.push_back(target);
    ++idx;
  }
  return result;
}

// Gumbel-MuZero improved-policy training target.  Constructed at chunk-
// write time from final search state.  Formula (from mctx's
// qtransform_completed_by_mix_value + the gumbel_muzero_policy training
// target builder):
//
//   target[i] = softmax(prior_logit[i] + conf[i] × σ(q)[i])
//
// where:
//   prior_logit[i] = log(P(i))  — using lc0's post-softmax priors directly
//   σ(q)[i]        = (maxvisit_init + max_N) × value_scale × rescaled_q[i]
//   rescaled_q[i]  = (q[i] - q_min) / (q_max - q_min + ε)  ∈ [0, 1]
//   q[i]           = empirical Q for visited edges
//                  = v-mix imputed value for unvisited edges
//   conf[i]        = sqrt(N[i]) / sqrt(max_N)  — sqrt-scaled confidence
//                    that damps σ contribution for low-N (and ZEROs out
//                    for N=0).  This is OUR deviation from mctx, needed
//                    because PUCT-driven search doesn't visit all
//                    candidates like Gumbel-SH would, so v-mix imputation
//                    for unvisited moves is selection-biased.
//
// v-mix formula (mctx _compute_mixed_value):
//   weighted_q  = sum(prior[i] × q[i] × visited[i]) / sum(prior[i] × visited[i])
//   mixed_value = (raw_value + sum_visit_counts × weighted_q)
//                 / (sum_visit_counts + 1)
//
// raw_value = root_node_->GetWL() — value-head prediction at root.
//
// Composes with forced exploration / advisor (more visited candidates =
// better v-mix and more reliable σ(q)).  Mutually exclusive with PTP —
// caller in selfplay/game.cc dispatches to either this or
// GetTrainingTargetVisits, never both.
std::vector<float> Search::GetGumbelImprovedPolicyTarget() const {
  std::vector<float> result;
  if (root_node_ == nullptr) return result;
  const int num_edges = root_node_->GetNumEdges();
  result.reserve(num_edges);

  // Pull params for the σ transform.
  const float maxvisit_init = params_.GetGumbelMuZeroCVisit();
  const float value_scale = params_.GetGumbelMuZeroCScale();
  constexpr float kEps = 1e-8f;

  // Root is even-depth.
  const bool is_odd_depth = false;
  const float draw_score = GetDrawScore(is_odd_depth);
  const float fpu = GetFpu(params_, root_node_, /*is_root=*/true, draw_score);

  // First pass: collect prior logits, raw Q (with fpu for unvisited),
  // visit counts; track sums and extremes.
  std::vector<float> logits;
  std::vector<float> qvals;
  std::vector<uint32_t> visits;
  logits.reserve(num_edges);
  qvals.reserve(num_edges);
  visits.reserve(num_edges);

  uint32_t total_visits = 0;
  uint32_t max_visits = 0;
  float weighted_q_num = 0.0f;
  float weighted_q_den = 0.0f;

  for (const auto& edge : root_node_->Edges()) {
    const float p = std::max(edge.GetP(), kEps);
    const float lp = std::log(p);
    const uint32_t n = edge.GetN();
    const float q_visited = edge.GetQ(fpu, draw_score);  // = node Q if N>0
    logits.push_back(lp);
    visits.push_back(n);
    qvals.push_back(q_visited);  // placeholder; overwritten for unvisited
    total_visits += n;
    if (n > max_visits) max_visits = n;
    if (n > 0) {
      weighted_q_num += p * q_visited;
      weighted_q_den += p;
    }
  }

  // No visits at all → return uniform-by-prior softmax.  Numerically
  // stable softmax of logits alone.
  if (total_visits == 0 || num_edges == 0) {
    float lmax = -std::numeric_limits<float>::infinity();
    for (float l : logits) lmax = std::max(lmax, l);
    float sum = 0.0f;
    for (float l : logits) sum += std::exp(l - lmax);
    if (sum <= 0.0f) sum = 1.0f;
    for (float l : logits) result.push_back(std::exp(l - lmax) / sum);
    return result;
  }

  // v-mix imputed value for unvisited edges.  raw_value = network's
  // value-head prediction at root.  GetWL() is the running mean which
  // converges to that prediction; close enough for our purposes here.
  //
  // Sign-convention note: root_node_->GetWL() is stored in root's
  // PARENT's perspective (see backprop trace at search.cc:2844 + 2948),
  // i.e. opposite sign vs edge.GetQ() at root.  Negate so raw_value
  // lives in the same root-to-move convention as qvals[] (which were
  // populated from edge.GetQ() above).  Without this flip the v-mix
  // averages two values with opposite sign conventions, which silently
  // biases the imputed Q for unvisited edges by ~2·raw_value.
  const float raw_value = -root_node_->GetWL();
  const float weighted_q =
      weighted_q_den > kEps ? (weighted_q_num / weighted_q_den) : raw_value;
  const float mixed_value =
      (raw_value + static_cast<float>(total_visits) * weighted_q) /
      (static_cast<float>(total_visits) + 1.0f);

  // Replace unvisited Q with mixed_value; track min/max for rescale.
  float qmin = std::numeric_limits<float>::infinity();
  float qmax = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < num_edges; ++i) {
    if (visits[i] == 0) qvals[i] = mixed_value;
    if (qvals[i] < qmin) qmin = qvals[i];
    if (qvals[i] > qmax) qmax = qvals[i];
  }

  // Rescale Q to [0, 1].  Guard against qmax == qmin (all equal).
  const float qrange = std::max(qmax - qmin, kEps);
  for (int i = 0; i < num_edges; ++i) {
    qvals[i] = (qvals[i] - qmin) / qrange;
  }

  // σ transform scale factor (constant per call).
  const float visit_scale =
      (maxvisit_init + static_cast<float>(max_visits)) * value_scale;

  // sqrt-scaled confidence per edge.  N=0 → conf=0 → no σ contribution.
  // Use max_visits as the denominator; if max_visits==0 (unreachable
  // here since total_visits>0) treat as 1 to avoid divide-by-zero.
  const float max_visits_sqrt =
      std::sqrt(static_cast<float>(std::max(max_visits, 1u)));

  // Combine: improved_logits[i] = logit + conf × σ(q).
  std::vector<float> improved_logits(num_edges);
  float lmax = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < num_edges; ++i) {
    const float conf =
        std::sqrt(static_cast<float>(visits[i])) / max_visits_sqrt;
    improved_logits[i] = logits[i] + conf * visit_scale * qvals[i];
    if (improved_logits[i] > lmax) lmax = improved_logits[i];
  }

  // Numerically stable softmax → output probability distribution.
  float sum = 0.0f;
  for (int i = 0; i < num_edges; ++i) {
    improved_logits[i] = std::exp(improved_logits[i] - lmax);
    sum += improved_logits[i];
  }
  if (sum <= 0.0f) sum = 1.0f;
  for (int i = 0; i < num_edges; ++i) {
    result.push_back(improved_logits[i] / sum);
  }
  return result;
}

std::vector<float> Search::GetGrillImprovedPolicyTarget() const {
  // Implements the "Learn" variant of Grill et al. (ICML 2020), §4.2:
  // π̄(a) = λ_N · prior(a) / (α − q(a))
  // where α is solved via dichotomic search so π̄ sums to 1.
  //
  // UCI options (read once via params_.GetGrillC()):
  //   --use-grill-improved-target  (bool) — gate at game.cc dispatch
  //   --grill-c                    (float, default 1.0) — c constant
  std::vector<float> result;
  if (root_node_ == nullptr) return result;
  result.reserve(root_node_->GetNumEdges());

  const float kGrillC = params_.GetGrillC();

  // Read root-edge state.  Q-imputation for unvisited edges: use the
  // root's network value, NEGATED.  Simpler than v-mix; the paper doesn't
  // specify so we pick the straightforward option and refine later if
  // results suggest it.
  //
  // Sign-convention note (verified against backprop in
  // DoBackupUpdateSingleNode + line 2844 NN-q flip):
  //   - node.wl_ is stored in the PARENT-of-node's perspective.
  //   - edge.GetQ(fpu, draw_score) reads CHILD node's wl_, so it is in
  //     the SELECTING node's (= root's) to-move perspective — exactly
  //     what PUCT consumes at line 2765.
  //   - root_node_->GetWL() is in root's PARENT's perspective, i.e.
  //     the OPPOSITE sign of edge.GetQ() at root.  We therefore negate
  //     it before mixing with edge.GetQ() values, so the unvisited
  //     imputation lives in the same convention as the visited Q's.
  const bool is_root = true;
  const bool is_odd_depth = !is_root;
  const float draw_score = GetDrawScore(is_odd_depth);
  const float fpu = GetFpu(params_, root_node_, is_root, draw_score);
  const float raw_value =
      -static_cast<float>(root_node_->GetWL());  // flip to root-to-move view

  std::vector<float> priors;
  std::vector<float> qs;
  uint32_t total_visits = 0;
  for (const auto& edge : root_node_->Edges()) {
    priors.push_back(edge.GetP());
    if (edge.GetN() > 0) {
      qs.push_back(edge.GetQ(fpu, draw_score));
    } else {
      // Unvisited: impute Q with the root's network value.  More
      // sophisticated approaches (v-mix from visited siblings)
      // could go here.
      qs.push_back(raw_value);
    }
    total_visits += edge.GetN();
  }

  if (total_visits == 0) {
    // No search happened.  Return prior as-is (no information to
    // shape it Q-wise).
    for (float p : priors) result.push_back(p);
    return result;
  }

  // λ_N = c · √N / (|𝒜| + N) per Grill et al. (ICML 2020), Eq. 4.
  // |𝒜| is the number of legal root actions (priors.size()).  This shape
  // peaks around N ≈ |𝒜| and decays as N grows, so π̄ relaxes from the
  // prior toward the Q-greedy distribution at high visit counts.
  const float lambda_N =
      kGrillC * std::sqrt(static_cast<float>(total_visits)) /
      (static_cast<float>(priors.size()) +
       static_cast<float>(total_visits));

  // Degenerate case: λ_N = 0 makes π̄ = 0 / (α − q) = 0 for all actions,
  // which cannot sum to 1.  This happens when grill-c is set to 0 (the
  // formula collapses to "no Q signal at all").  Defensive guard rather
  // than silently producing an invalid distribution OR hanging in the
  // safety-expansion loop below (which doubles a 0 gap and never
  // terminates).  Fall back to the prior.
  if (lambda_N <= 0.0f) {
    for (float p : priors) result.push_back(p);
    return result;
  }

  // Solve for α via dichotomic search.  Implementation notes vs the
  // straightforward "bisect on α directly in float" approach:
  //
  //   1. Variable transformation: β = α − q_max.  This eliminates
  //      catastrophic floating-point cancellation when prior(argmax_q)
  //      is small.  The original formula has α very close to q_max in
  //      that regime; subtracting two ~O(0.6) numbers to get a O(1e-30)
  //      difference destroys all precision in fp32.  Working in β
  //      directly keeps the magnitude where we need precision.
  //
  //   2. Bound on β: from Grill Appendix B.3, β ∈ (0, λ_N].
  //      f(β) := Σ_a λ_N · prior(a) / (β + Δq(a)) where Δq = q_max − q.
  //      f is strictly decreasing on (0, ∞), f → ∞ as β → 0,
  //      f(λ_N) ≤ Σ_a prior(a) = 1.  So β* exists uniquely in (0, λ_N].
  //
  //   3. Double precision: the bisection and the final π̄ computation
  //      use double.  At β ≈ 1e-30, fp32 has no precision left;
  //      fp64 gives us ~15 digits which is enough for any realistic
  //      prior distribution.
  //
  //   4. Geometric-mean midpoint: arithmetic-mean bisection wastes
  //      iterations when β* is many orders of magnitude smaller than
  //      β_high.  Using √(β_low · β_high) instead halves the LOG range
  //      per iteration — converges in 50 iterations regardless of
  //      β-scale (from λ_N down to 1e-300).
  //
  //   5. Initial β_low: any value where f(β_low) ≥ 1.  We use
  //      std::numeric_limits<double>::min() (~2.2e-308) as a safe
  //      lower bound that f(β_low) is definitely ≥ 1 at — the term
  //      for the argmax_q action alone contributes
  //      λ_N · prior(a*) / 2.2e-308 which is astronomically large.

  double q_max = -std::numeric_limits<double>::infinity();
  for (float q : qs) q_max = std::max(q_max, static_cast<double>(q));

  // Precompute Δq(a) = q_max − q(a) in double precision (the difference
  // CAN suffer cancellation here, but q_max and q are both O(1) in
  // magnitude so the result is at worst ~ulp(1) ≈ 1e-16 error — fine).
  std::vector<double> dq;
  dq.reserve(qs.size());
  for (float q : qs) dq.push_back(q_max - static_cast<double>(q));
  const double lambda_d = static_cast<double>(lambda_N);

  // f(β) — the policy sum at α = q_max + β.  Each term is well-
  // conditioned because both β and Δq(a) are ≥ 0 and we ADD them
  // (no subtraction).
  auto f_at = [&](double beta) -> double {
    double sum = 0;
    for (size_t i = 0; i < priors.size(); ++i) {
      sum += lambda_d * static_cast<double>(priors[i]) / (beta + dq[i]);
    }
    return sum;
  };

  // Bisection in β-space using geometric mean.  The geometric mean
  // halves the log of the search interval each iteration, which is
  // exactly what we want when β* could be anywhere from λ_N down to
  // ~prior(a*) · λ_N (potentially 1e-30 or smaller).
  double beta_low  = std::numeric_limits<double>::min();
  double beta_high = lambda_d;

  // Sanity: f(β_high) should be ≤ 1.  By construction it is (proven
  // above) unless priors don't sum to ~1.  Skip the upper-bound
  // expansion loop entirely — not needed in the β-space formulation.

  for (int iter = 0; iter < 60; ++iter) {
    // Geometric mean = exp((log_low + log_high) / 2) but √(a·b) is
    // numerically equivalent and avoids the log/exp round-trip.  Use
    // std::sqrt to keep it simple; std::sqrt on double is exact to
    // within 0.5 ulp.
    const double beta_mid = std::sqrt(beta_low * beta_high);
    if (beta_mid == beta_low || beta_mid == beta_high) {
      // Converged to representation limit.  Stop.
      break;
    }
    const double s = f_at(beta_mid);
    if (std::abs(s - 1.0) < 1e-9) {
      beta_low = beta_high = beta_mid;
      break;
    }
    if (s > 1.0) {
      beta_low = beta_mid;
    } else {
      beta_high = beta_mid;
    }
  }
  // Use the geometric mean of the final interval as our β estimate.
  const double beta_final = std::sqrt(beta_low * beta_high);

  // Compute π̄(a) using the same well-conditioned formula.  Sum in
  // double then renormalize so the fp32 result sums to exactly 1.0
  // within float precision.
  double sum = 0;
  std::vector<double> pi_bars(priors.size());
  for (size_t i = 0; i < priors.size(); ++i) {
    pi_bars[i] = lambda_d * static_cast<double>(priors[i]) /
                 (beta_final + dq[i]);
    sum += pi_bars[i];
  }
  // sum should be very close to 1.  If it's degenerate (zero, NaN, Inf)
  // we have a bug or pathological input — fall back to prior rather
  // than emitting a corrupted distribution.  This branch should never
  // fire under the β-space formulation; keeping it as a tripwire.
  if (!(sum > 0.0) || !std::isfinite(sum)) {
    result.assign(priors.begin(), priors.end());
    return result;
  }
  result.reserve(pi_bars.size());
  for (double pb : pi_bars) {
    result.push_back(static_cast<float>(pb / sum));
  }
  return result;
}

std::vector<std::string> Search::GetVerboseStats(const Node* node) const {
  assert(node == root_node_ || node->GetParent() == root_node_);
  const bool is_root = (node == root_node_);
  const bool is_odd_depth = !is_root;
  const bool is_black_to_move = (played_history_.IsBlackToMove() == is_root);
  const float draw_score = GetDrawScore(is_odd_depth);
  const float fpu = GetFpu(params_, node, is_root, draw_score);
  const float cpuct = ComputeCpuct(params_, node->GetN(), is_root);
  const float U_coeff =
      cpuct * std::sqrt(std::max(node->GetChildrenVisits(), 1u));
  std::vector<std::tuple<uint32_t, float, EdgeAndNode>> edges;
  edges.reserve(node->GetNumEdges());
  for (const auto& edge : node->Edges()) {
    edges.emplace_back(edge.GetN(),
                       edge.GetQ(fpu, draw_score) + edge.GetU(U_coeff),
                       edge);
  }
  std::sort(edges.begin(), edges.end());

  auto print = [](auto* oss, auto pre, auto v, auto post, auto w, int p = 0) {
    *oss << pre << std::setw(w) << std::setprecision(p) << v << post;
  };
  auto print_head = [&](auto* oss, auto label, int i, auto n, auto f, auto p) {
    *oss << std::fixed;
    print(oss, "", label, " ", 5);
    print(oss, "(", i, ") ", 4);
    *oss << std::right;
    print(oss, "N: ", n, " ", 7);
    print(oss, "(+", f, ") ", 2);
    print(oss, "(P: ", p * 100, "%) ", 5, p >= 0.99995f ? 1 : 2);
  };
  auto print_stats = [&](auto* oss, const auto* n) {
    const auto sign = n == node ? -1 : 1;
    if (n) {
      auto wl = sign * n->GetWL();
      auto d = n->GetD();
      auto is_perspective = ((contempt_mode_ == ContemptMode::BLACK) ==
                             played_history_.IsBlackToMove())
                                ? 1.0f
                                : -1.0f;
      WDLRescale(
          wl, d, params_.GetWDLRescaleRatio(),
          contempt_mode_ == ContemptMode::NONE
              ? 0
              : params_.GetWDLRescaleDiff() * params_.GetWDLEvalObjectivity(),
          is_perspective, true, params_.GetWDLMaxS());
      print(oss, "(WL: ", wl, ") ", 8, 5);
      print(oss, "(D: ", d, ") ", 5, 3);
      print(oss, "(M: ", n->GetM(), ") ", 4, 1);
      print(oss, "(Q: ", wl + draw_score * d, ") ", 8, 5);
    } else {
      *oss << "(WL:  -.-----) (D: -.---) (M:  -.-) ";
      print(oss, "(Q: ", fpu, ") ", 8, 5);
    }
  };
  auto print_tail = [&](auto* oss, const auto* n) {
    const auto sign = n == node ? -1 : 1;
    std::optional<float> v;
    if (n && n->IsTerminal()) {
      v = n->GetQ(sign * draw_score);
    } else if (n) {
      auto history = GetPositionHistoryAtNode(n);
      std::optional<EvalResult> nneval = backend_->GetCachedEvaluation(
          EvalPosition{history.GetPositions(), {}});
      if (nneval) v = -nneval->q;
    }
    if (v) {
      print(oss, "(V: ", sign * *v, ") ", 7, 4);
    } else {
      *oss << "(V:  -.----) ";
    }

    if (n) {
      auto [lo, up] = n->GetBounds();
      if (sign == -1) {
        lo = -lo;
        up = -up;
        std::swap(lo, up);
      }
      *oss << (lo == up                                                ? "(T) "
               : lo == GameResult::DRAW && up == GameResult::WHITE_WON ? "(W) "
               : lo == GameResult::BLACK_WON && up == GameResult::DRAW ? "(L) "
                                                                       : "");
    }
  };

  std::vector<std::string> infos;
  const auto m_evaluator =
      backend_attributes_.has_mlh ? MEvaluator(params_, node) : MEvaluator();
  for (const auto& edge_tuple : edges) {
    const auto& edge = std::get<2>(edge_tuple);
    float Q = edge.GetQ(fpu, draw_score);
    float M = m_evaluator.GetMUtility(edge, Q);
    std::ostringstream oss;
    oss << std::left;
    // TODO: should this be displaying transformed index?
    print_head(&oss, edge.GetMove(is_black_to_move).ToString(true),
               MoveToNNIndex(edge.GetMove(), 0), edge.GetN(),
               edge.GetNInFlight(), edge.GetP());
    print_stats(&oss, edge.node());
    print(&oss, "(U: ", edge.GetU(U_coeff), ") ", 6, 5);
    print(&oss, "(S: ", Q + edge.GetU(U_coeff) + M, ") ", 8, 5);
    print_tail(&oss, edge.node());
    infos.emplace_back(oss.str());
  }

  // Include stats about the node in similar format to its children above.
  std::ostringstream oss;
  print_head(&oss, "node ", node->GetNumEdges(), node->GetN(),
             node->GetNInFlight(), node->GetVisitedPolicy());
  print_stats(&oss, node);
  print_tail(&oss, node);
  infos.emplace_back(oss.str());
  return infos;
}

void Search::SendMovesStats() const REQUIRES(counters_mutex_) {
  auto move_stats = GetVerboseStats(root_node_);

  if (params_.GetVerboseStats()) {
    std::vector<ThinkingInfo> infos;
    std::transform(move_stats.begin(), move_stats.end(),
                   std::back_inserter(infos), [](const std::string& line) {
                     ThinkingInfo info;
                     info.comment = line;
                     return info;
                   });
    uci_responder_->OutputThinkingInfo(&infos);
  } else {
    LOGFILE << "=== Move stats:";
    for (const auto& line : move_stats) LOGFILE << line;
  }
  for (auto& edge : root_node_->Edges()) {
    if (!(edge.GetMove(played_history_.IsBlackToMove()) == final_bestmove_)) {
      continue;
    }
    if (edge.HasNode()) {
      LOGFILE << "--- Opponent moves after: " << final_bestmove_.ToString(true);
      for (const auto& line : GetVerboseStats(edge.node())) {
        LOGFILE << line;
      }
    }
  }
}

PositionHistory Search::GetPositionHistoryAtNode(const Node* node) const {
  PositionHistory history(played_history_);
  std::vector<Move> rmoves;
  for (const Node* n = node; n != root_node_; n = n->GetParent()) {
    rmoves.push_back(n->GetOwnEdge()->GetMove());
  }
  for (auto it = rmoves.rbegin(); it != rmoves.rend(); it++) {
    history.Append(*it);
  }
  return history;
}

void Search::MaybeTriggerStop(const IterationStats& stats,
                              StoppersHints* hints) {
  hints->Reset();
  if (params_.GetNpsLimit() > 0) {
    hints->UpdateEstimatedNps(params_.GetNpsLimit());
  }
  SharedMutex::Lock nodes_lock(nodes_mutex_);
  Mutex::Lock lock(counters_mutex_);
  // Already responded bestmove, nothing to do here.
  if (bestmove_is_sent_) return;
  // Don't stop when the root node is not yet expanded.
  if (stats.total_nodes == 0) return;

  if (!stop_.load(std::memory_order_acquire)) {
    if (stopper_->ShouldStop(stats, hints)) FireStopInternal();
  }

  // If we are the first to see that stop is needed.
  if (stop_.load(std::memory_order_acquire) && ok_to_respond_bestmove_ &&
      !bestmove_is_sent_) {
    SendUciInfo();
    EnsureBestMoveKnown();
    SendMovesStats();
    BestMoveInfo info(final_bestmove_, final_pondermove_);
    uci_responder_->OutputBestMove(&info);
    stopper_->OnSearchDone(stats);
    bestmove_is_sent_ = true;
    current_best_edge_ = EdgeAndNode();
  }
}

// Return the evaluation of the actual best child, regardless of temperature
// settings. This differs from GetBestMove, which does obey any temperature
// settings. So, somethimes, they may return results of different moves.
Eval Search::GetBestEval(Move* move, bool* is_terminal) const {
  SharedMutex::SharedLock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  float parent_wl = -root_node_->GetWL();
  float parent_d = root_node_->GetD();
  float parent_m = root_node_->GetM();
  if (!root_node_->HasChildren()) return {parent_wl, parent_d, parent_m};
  EdgeAndNode best_edge = GetBestChildNoTemperature(root_node_, 0);
  if (move) *move = best_edge.GetMove(played_history_.IsBlackToMove());
  if (is_terminal) *is_terminal = best_edge.IsTerminal();
  return {best_edge.GetWL(parent_wl), best_edge.GetD(parent_d),
          best_edge.GetM(parent_m - 1) + 1};
}

std::pair<Move, Move> Search::GetBestMove() {
  SharedMutex::Lock lock(nodes_mutex_);
  Mutex::Lock counters_lock(counters_mutex_);
  EnsureBestMoveKnown();
  return {final_bestmove_, final_pondermove_};
}

std::int64_t Search::GetTotalPlayouts() const {
  SharedMutex::SharedLock lock(nodes_mutex_);
  return total_playouts_;
}

void Search::ResetBestMove() {
  SharedMutex::Lock nodes_lock(nodes_mutex_);
  Mutex::Lock lock(counters_mutex_);
  bool old_sent = bestmove_is_sent_;
  bestmove_is_sent_ = false;
  EnsureBestMoveKnown();
  bestmove_is_sent_ = old_sent;
}

// Computes the best move, maybe with temperature (according to the settings).
void Search::EnsureBestMoveKnown() REQUIRES(nodes_mutex_)
    REQUIRES(counters_mutex_) {
  if (bestmove_is_sent_) return;
  if (root_node_->GetN() == 0) return;
  if (!root_node_->HasChildren()) return;

  float temperature = params_.GetTemperature();
  const int cutoff_move = params_.GetTemperatureCutoffMove();
  const int decay_delay_moves = params_.GetTempDecayDelayMoves();
  const int decay_moves = params_.GetTempDecayMoves();
  const int moves = played_history_.Last().GetGamePly() / 2;

  if (cutoff_move && (moves + 1) >= cutoff_move) {
    temperature = params_.GetTemperatureEndgame();
  } else if (temperature && decay_moves) {
    if (moves >= decay_delay_moves + decay_moves) {
      temperature = 0.0;
    } else if (moves >= decay_delay_moves) {
      temperature *=
          static_cast<float>(decay_delay_moves + decay_moves - moves) /
          decay_moves;
    }
    // don't allow temperature to decay below endgame temperature
    if (temperature < params_.GetTemperatureEndgame()) {
      temperature = params_.GetTemperatureEndgame();
    }
  }

  auto bestmove_edge = temperature
                           ? GetBestRootChildWithTemperature(temperature)
                           : GetBestChildNoTemperature(root_node_, 0);
  final_bestmove_ = bestmove_edge.GetMove(played_history_.IsBlackToMove());

  if (bestmove_edge.GetN() > 0 && bestmove_edge.node()->HasChildren()) {
    final_pondermove_ = GetBestChildNoTemperature(bestmove_edge.node(), 1)
                            .GetMove(!played_history_.IsBlackToMove());
  }
}

// Returns @count children with most visits.
std::vector<EdgeAndNode> Search::GetBestChildrenNoTemperature(Node* parent,
                                                              int count,
                                                              int depth) const {
  // Even if Edges is populated at this point, its a race condition to access
  // the node, so exit quickly.
  if (parent->GetN() == 0) return {};
  const bool is_odd_depth = (depth % 2) == 1;
  const float draw_score = GetDrawScore(is_odd_depth);
  // Best child is selected using the following criteria:
  // * Prefer shorter terminal wins / avoid shorter terminal losses.
  // * Largest number of playouts.
  // * If two nodes have equal number:
  //   * If that number is 0, the one with larger prior wins.
  //   * If that number is larger than 0, the one with larger eval wins.
  std::vector<EdgeAndNode> edges;
  for (auto& edge : parent->Edges()) {
    if (parent == root_node_ && !root_move_filter_.empty() &&
        std::find(root_move_filter_.begin(), root_move_filter_.end(),
                  edge.GetMove()) == root_move_filter_.end()) {
      continue;
    }
    edges.push_back(edge);
  }
  const auto middle = (static_cast<int>(edges.size()) > count)
                          ? edges.begin() + count
                          : edges.end();
  std::partial_sort(
      edges.begin(), middle, edges.end(),
      [draw_score](const auto& a, const auto& b) {
        // The function returns "true" when a is preferred to b.

        // Lists edge types from less desirable to more desirable.
        enum EdgeRank {
          kTerminalLoss,
          kTablebaseLoss,
          kNonTerminal,  // Non terminal or terminal draw.
          kTablebaseWin,
          kTerminalWin,
        };

        auto GetEdgeRank = [](const EdgeAndNode& edge) {
          // This default isn't used as wl only checked for case edge is
          // terminal.
          const auto wl = edge.GetWL(0.0f);
          // Not safe to access IsTerminal if GetN is 0.
          if (edge.GetN() == 0 || !edge.IsTerminal() || !wl) {
            return kNonTerminal;
          }
          if (edge.IsTbTerminal()) {
            return wl < 0.0 ? kTablebaseLoss : kTablebaseWin;
          }
          return wl < 0.0 ? kTerminalLoss : kTerminalWin;
        };

        // If moves have different outcomes, prefer better outcome.
        const auto a_rank = GetEdgeRank(a);
        const auto b_rank = GetEdgeRank(b);
        if (a_rank != b_rank) return a_rank > b_rank;

        // If both are terminal draws, try to make it shorter.
        // Not safe to access IsTerminal if GetN is 0.
        if (a_rank == kNonTerminal && a.GetN() != 0 && b.GetN() != 0 &&
            a.IsTerminal() && b.IsTerminal()) {
          if (a.IsTbTerminal() != b.IsTbTerminal()) {
            // Prefer non-tablebase draws.
            return a.IsTbTerminal() < b.IsTbTerminal();
          }
          // Prefer shorter draws.
          return a.GetM(0.0f) < b.GetM(0.0f);
        }

        // Neither is terminal, use standard rule.
        if (a_rank == kNonTerminal) {
          // Prefer largest playouts then eval then prior.
          if (a.GetN() != b.GetN()) return a.GetN() > b.GetN();
          // Default doesn't matter here so long as they are the same as either
          // both are N==0 (thus we're comparing equal defaults) or N!=0 and
          // default isn't used.
          if (a.GetQ(0.0f, draw_score) != b.GetQ(0.0f, draw_score)) {
            return a.GetQ(0.0f, draw_score) > b.GetQ(0.0f, draw_score);
          }
          return a.GetP() > b.GetP();
        }

        // Both variants are winning, prefer shortest win.
        if (a_rank > kNonTerminal) {
          return a.GetM(0.0f) < b.GetM(0.0f);
        }

        // Both variants are losing, prefer longest losses.
        return a.GetM(0.0f) > b.GetM(0.0f);
      });

  if (count < static_cast<int>(edges.size())) {
    edges.resize(count);
  }
  return edges;
}

// Returns a child with most visits.
EdgeAndNode Search::GetBestChildNoTemperature(Node* parent, int depth) const {
  auto res = GetBestChildrenNoTemperature(parent, 1, depth);
  return res.empty() ? EdgeAndNode() : res.front();
}

// Returns a child of a root chosen according to weighted-by-temperature visit
// count.
EdgeAndNode Search::GetBestRootChildWithTemperature(float temperature) const {
  // Root is at even depth.
  const float draw_score = GetDrawScore(/* is_odd_depth= */ false);

  // KataGo recipe: when forced exploration is active, the played-move
  // temperature distribution should use PRUNED visit counts (forced
  // visits subtracted from non-best edges), not raw N.  Without this,
  // forced visits inflate temperature sampling weight on low-policy
  // edges — selfplay games then over-sample those edges and look
  // "random" / decisive-but-shallow.
  //
  // KataGo applies this clamp in BOTH places: at training-target time
  // (we do it in GetTrainingTargetVisits) AND at play-selection time
  // (here).  Equivalent to KataGo's getReducedPlaySelectionWeight,
  // which returns min(actual_N, N_eq-from-PUCT-equilibrium).
  //
  // When forced exploration is disabled, GetTrainingTargetVisits hits
  // its fast-path and returns raw N, so this is a clean no-op.
  const bool any_force = params_.GetForcedExplorationFactor() > 0.0f ||
                         advisor_min_visits_ > 0;
  std::vector<float> pruned_visits;
  if (any_force) pruned_visits = GetTrainingTargetVisits();
  auto effective_n = [&](int idx, const EdgeAndNode& edge) -> float {
    return any_force ? pruned_visits[idx] : static_cast<float>(edge.GetN());
  };

  std::vector<float> cumulative_sums;
  float sum = 0.0;
  float max_n = 0.0;
  const float offset = params_.GetTemperatureVisitOffset();
  float max_eval = -1.0f;
  const float fpu =
      GetFpu(params_, root_node_, /* is_root= */ true, draw_score);

  int idx = 0;
  for (auto& edge : root_node_->Edges()) {
    if (!root_move_filter_.empty() &&
        std::find(root_move_filter_.begin(), root_move_filter_.end(),
                  edge.GetMove()) == root_move_filter_.end()) {
      ++idx;
      continue;
    }
    const float n_eff = effective_n(idx, edge) + offset;
    if (n_eff > max_n) {
      max_n = n_eff;
      max_eval = edge.GetQ(fpu, draw_score);
    }
    ++idx;
  }

  // TODO(crem) Simplify this code when samplers.h is merged.
  const float min_eval =
      max_eval - params_.GetTemperatureWinpctCutoff() / 50.0f;
  idx = 0;
  for (auto& edge : root_node_->Edges()) {
    if (!root_move_filter_.empty() &&
        std::find(root_move_filter_.begin(), root_move_filter_.end(),
                  edge.GetMove()) == root_move_filter_.end()) {
      ++idx;
      continue;
    }
    if (edge.GetQ(fpu, draw_score) < min_eval) {
      ++idx;
      continue;
    }
    sum += std::pow(
        std::max(0.0f, (max_n <= 0.0f
                            ? edge.GetP()
                            : ((effective_n(idx, edge) + offset) / max_n))),
        1 / temperature);
    cumulative_sums.push_back(sum);
    ++idx;
  }
  assert(sum);

  const float toss = Random::Get().GetFloat(cumulative_sums.back());
  int select =
      std::lower_bound(cumulative_sums.begin(), cumulative_sums.end(), toss) -
      cumulative_sums.begin();

  idx = 0;
  for (auto& edge : root_node_->Edges()) {
    if (!root_move_filter_.empty() &&
        std::find(root_move_filter_.begin(), root_move_filter_.end(),
                  edge.GetMove()) == root_move_filter_.end()) {
      ++idx;
      continue;
    }
    if (edge.GetQ(fpu, draw_score) < min_eval) {
      ++idx;
      continue;
    }
    if (select-- == 0) return edge;
    ++idx;
  }
  assert(false);
  return {};
}

void Search::StartThreads(size_t how_many) {
  Mutex::Lock lock(threads_mutex_);
  if (how_many == 0 && threads_.size() == 0) {
    how_many = backend_attributes_.suggested_num_search_threads +
               !backend_attributes_.runs_on_cpu;
  }
  thread_count_.store(how_many, std::memory_order_release);
  // First thread is a watchdog thread.
  if (threads_.size() == 0) {
    threads_.emplace_back([this]() { WatchdogThread(); });
  }
  // Start working threads.
  for (size_t i = 0; i < how_many; i++) {
    threads_.emplace_back([this]() {
      SearchWorker worker(this, params_);
      worker.RunBlocking();
    });
  }
  LOGFILE << "Search started. "
          << std::chrono::duration_cast<std::chrono::milliseconds>(
                 std::chrono::steady_clock::now() - start_time_)
                 .count()
          << "ms already passed.";
}

void Search::RunBlocking(size_t threads) {
  StartThreads(threads);
  Wait();
}

bool Search::IsSearchActive() const {
  return !stop_.load(std::memory_order_acquire);
}

void Search::PopulateCommonIterationStats(IterationStats* stats) {
  stats->time_since_movestart = GetTimeSinceStart();

  SharedMutex::SharedLock nodes_lock(nodes_mutex_);
  {
    Mutex::Lock counters_lock(counters_mutex_);
    stats->time_since_first_batch = GetTimeSinceFirstBatch();
    if (!nps_start_time_ && total_playouts_ > 0) {
      nps_start_time_ = std::chrono::steady_clock::now();
    }
  }
  stats->total_nodes = total_playouts_ + initial_visits_;
  stats->nodes_since_movestart = total_playouts_;
  stats->batches_since_movestart = total_batches_;
  stats->average_depth = cum_depth_ / (total_playouts_ ? total_playouts_ : 1);
  stats->edge_n.clear();
  stats->win_found = false;
  stats->may_resign = true;
  stats->num_losing_edges = 0;
  stats->time_usage_hint_ = IterationStats::TimeUsageHint::kNormal;
  stats->mate_depth = std::numeric_limits<int>::max();

  // If root node hasn't finished first visit, none of this code is safe.
  if (root_node_->GetN() > 0) {
    const auto draw_score = GetDrawScore(true);
    const float fpu =
        GetFpu(params_, root_node_, /* is_root_node */ true, draw_score);
    float max_q_plus_m = -1000;
    uint64_t max_n = 0;
    bool max_n_has_max_q_plus_m = true;
    const auto m_evaluator = backend_attributes_.has_mlh
                                 ? MEvaluator(params_, root_node_)
                                 : MEvaluator();
    for (const auto& edge : root_node_->Edges()) {
      const auto n = edge.GetN();
      const auto q = edge.GetQ(fpu, draw_score);
      const auto m = m_evaluator.GetMUtility(edge, q);
      const auto q_plus_m = q + m;
      stats->edge_n.push_back(n);
      if (n > 0 && edge.IsTerminal() && edge.GetWL(0.0f) > 0.0f) {
        stats->win_found = true;
      }
      if (n > 0 && edge.IsTerminal() && edge.GetWL(0.0f) < 0.0f) {
        stats->num_losing_edges += 1;
      }
      if (n > 0 && edge.IsTerminal() && edge.GetWL(0.0f) == 1.0f &&
          !edge.IsTbTerminal()) {
        stats->mate_depth =
            std::min(stats->mate_depth,
                     static_cast<int>(std::round(edge.GetM(0.0f))) / 2 + 1);
      }

      // If game is resignable, no need for moving quicker. This allows
      // proving mate when losing anyway for better score output.
      // Hardcoded resign threshold, because there is no available parameter.
      if (n > 0 && q > -0.98f) {
        stats->may_resign = false;
      }
      if (max_n < n) {
        max_n = n;
        max_n_has_max_q_plus_m = false;
      }
      if (max_q_plus_m <= q_plus_m) {
        max_n_has_max_q_plus_m = (max_n == n);
        max_q_plus_m = q_plus_m;
      }
    }
    if (!max_n_has_max_q_plus_m) {
      stats->time_usage_hint_ = IterationStats::TimeUsageHint::kNeedMoreTime;
    }
  }
}

void Search::WatchdogThread() {
  LOGFILE << "Start a watchdog thread.";
  StoppersHints hints;
  IterationStats stats;
  while (true) {
    PopulateCommonIterationStats(&stats);
    MaybeTriggerStop(stats, &hints);
    MaybeOutputInfo();

    constexpr auto kMaxWaitTimeMs = 100;
    constexpr auto kMinWaitTimeMs = 1;

    Mutex::Lock lock(counters_mutex_);
    // Only exit when bestmove is responded. It may happen that search threads
    // already all exited, and we need at least one thread that can do that.
    if (bestmove_is_sent_) break;

    auto remaining_time = hints.GetEstimatedRemainingTimeMs();
    if (remaining_time > kMaxWaitTimeMs) remaining_time = kMaxWaitTimeMs;
    if (remaining_time < kMinWaitTimeMs) remaining_time = kMinWaitTimeMs;
    // There is no real need to have max wait time, and sometimes it's fine
    // to wait without timeout at all (e.g. in `go nodes` mode), but we
    // still limit wait time for exotic cases like when pc goes to sleep
    // mode during thinking.
    // Minimum wait time is there to prevent busy wait and other threads
    // starvation.
    watchdog_cv_.wait_for(
        lock.get_raw(), std::chrono::milliseconds(remaining_time),
        [this]() { return stop_.load(std::memory_order_acquire); });
  }
  LOGFILE << "End a watchdog thread.";
}

void Search::FireStopInternal() {
  stop_.store(true, std::memory_order_release);
  watchdog_cv_.notify_all();
}

void Search::Stop() {
  Mutex::Lock lock(counters_mutex_);
  ok_to_respond_bestmove_ = true;
  FireStopInternal();
  LOGFILE << "Stopping search due to `stop` uci command.";
}

void Search::Abort() {
  Mutex::Lock lock(counters_mutex_);
  if (!stop_.load(std::memory_order_acquire) ||
      (!bestmove_is_sent_ && !ok_to_respond_bestmove_)) {
    bestmove_is_sent_ = true;
    FireStopInternal();
  }
  LOGFILE << "Aborting search, if it is still active.";
}

void Search::Wait() {
  Mutex::Lock lock(threads_mutex_);
  while (!threads_.empty()) {
    threads_.back().join();
    threads_.pop_back();
  }
}

void Search::CancelSharedCollisions() REQUIRES(nodes_mutex_) {
  for (auto& entry : shared_collisions_) {
    Node* node = entry.first;
    for (node = node->GetParent(); node != root_node_->GetParent();
         node = node->GetParent()) {
      node->CancelScoreUpdate(entry.second);
    }
  }
  shared_collisions_.clear();
}

Search::~Search() {
  Abort();
  Wait();
  {
    SharedMutex::Lock lock(nodes_mutex_);
    CancelSharedCollisions();
  }
  LOGFILE << "Search destroyed.";
}

//////////////////////////////////////////////////////////////////////////////
// SearchWorker
//////////////////////////////////////////////////////////////////////////////

void SearchWorker::RunTasks(int tid) {
  while (true) {
    PickTask* task = nullptr;
    int id = 0;
    {
      int spins = 0;
      while (true) {
        int nta = tasks_taken_.load(std::memory_order_acquire);
        int tc = task_count_.load(std::memory_order_acquire);
        if (nta < tc) {
          int val = 0;
          if (task_taking_started_.compare_exchange_weak(
                  val, 1, std::memory_order_acq_rel,
                  std::memory_order_relaxed)) {
            nta = tasks_taken_.load(std::memory_order_acquire);
            tc = task_count_.load(std::memory_order_acquire);
            // We got the spin lock, double check we're still in the clear.
            if (nta < tc) {
              id = tasks_taken_.fetch_add(1, std::memory_order_acq_rel);
              task = picking_tasks_.data() + id;
              task_taking_started_.store(0, std::memory_order_release);
              break;
            }
            task_taking_started_.store(0, std::memory_order_release);
          }
          SpinloopPause();
          spins = 0;
          continue;
        } else if (tc != -1) {
          spins++;
          if (spins >= 512) {
            std::this_thread::yield();
            spins = 0;
          } else {
            SpinloopPause();
          }
          continue;
        }
        spins = 0;
        // Looks like sleep time.
        Mutex::Lock lock(picking_tasks_mutex_);
        // Refresh them now we have the lock.
        nta = tasks_taken_.load(std::memory_order_acquire);
        tc = task_count_.load(std::memory_order_acquire);
        if (tc != -1) continue;
        if (nta >= tc && exiting_) return;
        task_added_.wait(lock.get_raw());
        // And refresh again now we're awake.
        nta = tasks_taken_.load(std::memory_order_acquire);
        tc = task_count_.load(std::memory_order_acquire);
        if (nta >= tc && exiting_) return;
      }
    }
    if (task != nullptr) {
      switch (task->task_type) {
        case PickTask::kGathering: {
          PickNodesToExtendTask(task->start, task->base_depth,
                                task->collision_limit, task->moves_to_base,
                                &(task->results), &(task_workspaces_[tid]));
          break;
        }
        case PickTask::kProcessing: {
          ProcessPickedTask(task->start_idx, task->end_idx,
                            &(task_workspaces_[tid]));
          break;
        }
      }
      picking_tasks_.data()[id].complete = true;
      completed_tasks_.fetch_add(1, std::memory_order_acq_rel);
    }
  }
}

void SearchWorker::ExecuteOneIteration() {
  // 1. Initialize internal structures.
  InitializeIteration();

  if (params_.GetMaxConcurrentSearchers() != 0) {
    std::unique_ptr<SpinHelper> spin_helper;
    if (params_.GetSearchSpinBackoff()) {
      spin_helper = std::make_unique<ExponentialBackoffSpinHelper>();
    } else {
      // This is a hard spin lock to reduce latency but at the expense of busy
      // wait cpu usage. If search worker count is large, this is probably a
      // bad idea.
      spin_helper = std::make_unique<SpinHelper>();
    }

    while (true) {
      // If search is stop, we've not gathered or done anything and we don't
      // want to, so we can safely skip all below. But make sure we have done
      // at least one iteration.
      if (search_->stop_.load(std::memory_order_acquire) &&
          search_->GetTotalPlayouts() + search_->initial_visits_ > 0) {
        return;
      }

      int available =
          search_->pending_searchers_.load(std::memory_order_acquire);
      if (available == 0) {
        spin_helper->Wait();
        continue;
      }

      if (search_->pending_searchers_.compare_exchange_weak(
              available, available - 1, std::memory_order_acq_rel)) {
        break;
      } else {
        spin_helper->Backoff();
      }
    }
  }

  // 2. Gather minibatch.
  GatherMinibatch();
  task_count_.store(-1, std::memory_order_release);
  search_->backend_waiting_counter_.fetch_add(1, std::memory_order_relaxed);

  // 2b. Collect collisions.
  CollectCollisions();

  // 3. Prefetch into cache.
  MaybePrefetchIntoCache();

  if (params_.GetMaxConcurrentSearchers() != 0) {
    search_->pending_searchers_.fetch_add(1, std::memory_order_acq_rel);
  }

  // 4. Run NN computation.
  RunNNComputation();
  search_->backend_waiting_counter_.fetch_add(-1, std::memory_order_relaxed);

  // 5. Retrieve NN computations (and terminal values) into nodes.
  FetchMinibatchResults();

  // 6. Propagate the new nodes' information to all their parents in the tree.
  DoBackupUpdate();

  // 7. Update the Search's status and progress information.
  UpdateCounters();

  // If required, waste time to limit nps.
  if (params_.GetNpsLimit() > 0) {
    while (search_->IsSearchActive()) {
      int64_t time_since_first_batch_ms = 0;
      {
        Mutex::Lock lock(search_->counters_mutex_);
        time_since_first_batch_ms = search_->GetTimeSinceFirstBatch();
      }
      if (time_since_first_batch_ms <= 0) {
        time_since_first_batch_ms = search_->GetTimeSinceStart();
      }
      auto nps = search_->GetTotalPlayouts() * 1e3f / time_since_first_batch_ms;
      if (nps > params_.GetNpsLimit()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      } else {
        break;
      }
    }
  }
}

// 1. Initialize internal structures.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void SearchWorker::InitializeIteration() {
  LCTRACE_FUNCTION_SCOPE;
  // Free the old computation before allocating a new one. This works better
  // when backend caches buffer allocations between computations.
  computation_.reset();
  computation_ = search_->backend_->CreateComputation();
  minibatch_.clear();
  minibatch_.reserve(2 * target_minibatch_size_);
}

// 2. Gather minibatch.
// ~~~~~~~~~~~~~~~~~~~~
namespace {
int Mix(int high, int low, float ratio) {
  return static_cast<int>(std::round(static_cast<float>(low) +
                                     static_cast<float>(high - low) * ratio));
}

int CalculateCollisionsLeft(int64_t nodes, const SearchParams& params) {
  // End checked first
  if (nodes >= params.GetMaxCollisionVisitsScalingEnd()) {
    return params.GetMaxCollisionVisits();
  }
  if (nodes <= params.GetMaxCollisionVisitsScalingStart()) {
    return 1;
  }
  return Mix(params.GetMaxCollisionVisits(), 1,
             std::pow((static_cast<float>(nodes) -
                       params.GetMaxCollisionVisitsScalingStart()) /
                          (params.GetMaxCollisionVisitsScalingEnd() -
                           params.GetMaxCollisionVisitsScalingStart()),
                      params.GetMaxCollisionVisitsScalingPower()));
}
}  // namespace

void SearchWorker::GatherMinibatch() {
  LCTRACE_FUNCTION_SCOPE;
  // Total number of nodes to process.
  int minibatch_size = 0;
  int cur_n = 0;
  {
    SharedMutex::Lock lock(search_->nodes_mutex_);
    cur_n = search_->root_node_->GetN();
  }
  // TODO: GetEstimatedRemainingPlayouts has already had smart pruning factor
  // applied, which doesn't clearly make sense to include here...
  int64_t remaining_n =
      latest_time_manager_hints_.GetEstimatedRemainingPlayouts();
  int collisions_left = CalculateCollisionsLeft(
      std::min(static_cast<int64_t>(cur_n), remaining_n), params_);

  // Number of nodes processed out of order.
  number_out_of_order_ = 0;

  int thread_count = search_->thread_count_.load(std::memory_order_acquire);

  // Gather nodes to process in the current batch.
  // If we had too many nodes out of order, also interrupt the iteration so
  // that search can exit.
  while (minibatch_size < target_minibatch_size_ &&
         number_out_of_order_ < max_out_of_order_) {
    // If there's something to process without touching slow neural net, do it.
    if (minibatch_size > 0 && computation_->UsedBatchSize() == 0) return;

    // If there is backend work to be done, and the backend is idle - exit
    // immediately.
    // Only do this fancy work if there are multiple threads as otherwise we
    // early exit from every batch since there is never another search thread to
    // be keeping the backend busy. Which would mean that threads=1 has a
    // massive nps drop.
    if (thread_count > 1 && minibatch_size > 0 &&
        static_cast<int>(computation_->UsedBatchSize()) >
            params_.GetIdlingMinimumWork() &&
        thread_count - search_->backend_waiting_counter_.load(
                           std::memory_order_relaxed) >
            params_.GetThreadIdlingThreshold()) {
      return;
    }

    int new_start = static_cast<int>(minibatch_.size());

    PickNodesToExtend(
        std::min({collisions_left, target_minibatch_size_ - minibatch_size,
                  max_out_of_order_ - number_out_of_order_}));

    // Count the non-collisions.
    int non_collisions = 0;
    for (int i = new_start; i < static_cast<int>(minibatch_.size()); i++) {
      auto& picked_node = minibatch_[i];
      if (picked_node.IsCollision()) {
        continue;
      }
      ++non_collisions;
      ++minibatch_size;
    }

    bool needs_wait = false;
    int ppt_start = new_start;
    if (task_workers_ > 0 &&
        non_collisions >= params_.GetMinimumWorkSizeForProcessing()) {
      const int num_tasks = std::clamp(
          non_collisions / params_.GetMinimumWorkPerTaskForProcessing(), 2,
          task_workers_ + 1);
      // Round down, left overs can go to main thread so it waits less.
      int per_worker = non_collisions / num_tasks;
      needs_wait = true;
      ResetTasks();
      int found = 0;
      for (int i = new_start; i < static_cast<int>(minibatch_.size()); i++) {
        auto& picked_node = minibatch_[i];
        if (picked_node.IsCollision()) {
          continue;
        }
        ++found;
        if (found == per_worker) {
          picking_tasks_.emplace_back(ppt_start, i + 1);
          task_count_.fetch_add(1, std::memory_order_acq_rel);
          ppt_start = i + 1;
          found = 0;
          if (picking_tasks_.size() == static_cast<size_t>(num_tasks - 1)) {
            break;
          }
        }
      }
    }
    ProcessPickedTask(ppt_start, static_cast<int>(minibatch_.size()),
                      &main_workspace_);
    if (needs_wait) {
      WaitForTasks();
    }
    bool some_ooo = false;
    for (int i = static_cast<int>(minibatch_.size()) - 1; i >= new_start; i--) {
      if (minibatch_[i].ooo_completed) {
        some_ooo = true;
        break;
      }
    }
    if (some_ooo) {
      LCTRACE_FUNCTION_SCOPE;
      SharedMutex::Lock lock(search_->nodes_mutex_);
      for (int i = static_cast<int>(minibatch_.size()) - 1; i >= new_start;
           i--) {
        // If there was any OOO, revert 'all' new collisions - it isn't possible
        // to identify exactly which ones are afterwards and only prune those.
        // This may remove too many items, but hopefully most of the time they
        // will just be added back in the same in the next gather.
        if (minibatch_[i].IsCollision()) {
          Node* node = minibatch_[i].node;
          for (node = node->GetParent();
               node != search_->root_node_->GetParent();
               node = node->GetParent()) {
            node->CancelScoreUpdate(minibatch_[i].multivisit);
          }
          minibatch_.erase(minibatch_.begin() + i);
        } else if (minibatch_[i].ooo_completed) {
          DoBackupUpdateSingleNode(minibatch_[i]);
          minibatch_.erase(minibatch_.begin() + i);
          --minibatch_size;
          ++number_out_of_order_;
        }
      }
    }

    LCTRACE_FUNCTION_SCOPE;
    // Check for stop at the end so we have at least one node.
    for (size_t i = new_start; i < minibatch_.size(); i++) {
      auto& picked_node = minibatch_[i];

      if (picked_node.IsCollision()) {
        // Check to see if we can upsize the collision to exit sooner.
        if (picked_node.maxvisit > 0 &&
            collisions_left > picked_node.multivisit) {
          SharedMutex::Lock lock(search_->nodes_mutex_);
          int extra = std::min(picked_node.maxvisit, collisions_left) -
                      picked_node.multivisit;
          picked_node.multivisit += extra;
          Node* node = picked_node.node;
          for (node = node->GetParent();
               node != search_->root_node_->GetParent();
               node = node->GetParent()) {
            node->IncrementNInFlight(extra);
          }
        }
        if ((collisions_left -= picked_node.multivisit) <= 0) return;
        if (search_->stop_.load(std::memory_order_acquire)) return;
      }
    }
  }
}

void SearchWorker::ProcessPickedTask(int start_idx, int end_idx,
                                     TaskWorkspace* workspace) {
  LCTRACE_FUNCTION_SCOPE;
  auto& history = workspace->history;
  history = search_->played_history_;

  for (int i = start_idx; i < end_idx; i++) {
    auto& picked_node = minibatch_[i];
    if (picked_node.IsCollision()) continue;
    auto* node = picked_node.node;

    // If node is already known as terminal (win/loss/draw according to rules
    // of the game), it means that we already visited this node before.
    if (picked_node.IsExtendable()) {
      // Node was never visited, extend it.
      ExtendNode(node, picked_node.depth, picked_node.moves_to_visit, &history);
      if (!node->IsTerminal()) {
        picked_node.nn_queried = true;
        MoveList legal_moves;
        legal_moves.reserve(node->GetNumEdges());
        std::transform(node->Edges().begin(), node->Edges().end(),
                       std::back_inserter(legal_moves),
                       [](const auto& edge) { return edge.GetMove(); });
        picked_node.eval->p.resize(legal_moves.size());
        // Pre-allocate the optimistic policy span only when the
        // operator opted in via EITHER --optimistic-policy-weight > 0
        // (root blend) OR --optimistic-policy-weight-internal > 0
        // (internal-node blend).  The wrapper/backend skips
        // populating the span when it's empty, so this is the gate
        // that turns the extra optimistic-head output on/off without
        // recompiling.
        //
        // Both knobs need to be checked here because the per-node
        // blend code (see ~line 2912) selects which alpha to use
        // based on node depth — root nodes use the root weight,
        // internal nodes use the internal weight.  If we only
        // allocated when the root weight was set, internal-only
        // configs (e.g. root=0, internal=1.0 to match KataGo) would
        // silently no-op despite p_optimistic appearing populated.
        if (params_.GetOptimisticPolicyWeight() > 0.0f ||
            params_.GetOptimisticPolicyWeightInternal() > 0.0f) {
          picked_node.eval->p_optimistic.resize(legal_moves.size());
        }
        picked_node.is_cache_hit = computation_->AddInput(
                                       EvalPosition{
                                           .pos = history.GetPositions(),
                                           .legal_moves = legal_moves,
                                       },
                                       picked_node.eval->AsPtr()) ==
                                   BackendComputation::FETCHED_IMMEDIATELY;
      }
    }
    if (params_.GetOutOfOrderEval() && picked_node.CanEvalOutOfOrder()) {
      // Perform out of order eval for the last entry in minibatch_.
      FetchSingleNodeResult(&picked_node);
      picked_node.ooo_completed = true;
    }
  }
}

#define MAX_TASKS 100

void SearchWorker::ResetTasks() {
  task_count_.store(0, std::memory_order_release);
  tasks_taken_.store(0, std::memory_order_release);
  completed_tasks_.store(0, std::memory_order_release);
  picking_tasks_.clear();
  // Reserve because resizing breaks pointers held by the task threads.
  picking_tasks_.reserve(MAX_TASKS);
}

int SearchWorker::WaitForTasks() {
  // Spin lock, other tasks should be done soon.
  while (true) {
    int completed = completed_tasks_.load(std::memory_order_acquire);
    int todo = task_count_.load(std::memory_order_acquire);
    if (todo == completed) return completed;
    SpinloopPause();
  }
}

void SearchWorker::PickNodesToExtend(int collision_limit) {
  ResetTasks();
  if (task_workers_ > 0 && !search_->backend_attributes_.runs_on_cpu) {
    // While nothing is ready yet - wake the task runners so they are ready to
    // receive quickly.
    Mutex::Lock lock(picking_tasks_mutex_);
    task_added_.notify_all();
  }
  std::vector<Move> empty_movelist;
  // This lock must be held until after the task_completed_ wait succeeds below.
  // Since the tasks perform work which assumes they have the lock, even though
  // actually this thread does.
  SharedMutex::Lock lock(search_->nodes_mutex_);
  PickNodesToExtendTask(search_->root_node_, 0, collision_limit, empty_movelist,
                        &minibatch_, &main_workspace_);

  WaitForTasks();
  for (int i = 0; i < static_cast<int>(picking_tasks_.size()); i++) {
    for (int j = 0; j < static_cast<int>(picking_tasks_[i].results.size());
         j++) {
      minibatch_.emplace_back(std::move(picking_tasks_[i].results[j]));
    }
  }
}

void SearchWorker::EnsureNodeTwoFoldCorrectForDepth(Node* child_node,
                                                    int depth) {
  // Check whether first repetition was before root. If yes, remove
  // terminal status of node and revert all visits in the tree.
  // Length of repetition was stored in m_. This code will only do
  // something when tree is reused and twofold visits need to be
  // reverted.
  if (child_node->IsTwoFoldTerminal() && depth < child_node->GetM()) {
    // Take a mutex - any SearchWorker specific mutex... since this is
    // not safe to do concurrently between multiple tasks.
    Mutex::Lock lock(picking_tasks_mutex_);
    int depth_counter = 0;
    // Cache node's values as we reset them in the process. We could
    // manually set wl and d, but if we want to reuse this for reverting
    // other terminal nodes this is the way to go.
    const auto wl = child_node->GetWL();
    const auto d = child_node->GetD();
    const auto m = child_node->GetM();
    const auto terminal_visits = child_node->GetN();
    for (Node* node_to_revert = child_node; node_to_revert != nullptr;
         node_to_revert = node_to_revert->GetParent()) {
      // Revert all visits on twofold draw when making it non terminal.
      node_to_revert->RevertTerminalVisits(wl, d, m + (float)depth_counter,
                                           terminal_visits);
      depth_counter++;
      // Even if original tree still exists, we don't want to revert
      // more than until new root.
      if (depth_counter > depth) break;
      // If wl != 0, we would have to switch signs at each depth.
    }
    // Mark the prior twofold draw as non terminal to extend it again.
    child_node->MakeNotTerminal();
    // When reverting the visits, we also need to revert the initial
    // visits, as we reused fewer nodes than anticipated.
    search_->initial_visits_ -= terminal_visits;
    // Max depth doesn't change when reverting the visits, and
    // cum_depth_ only counts the average depth of new nodes, not reused
    // ones.
  }
}

void SearchWorker::PickNodesToExtendTask(
    Node* node, int base_depth, int collision_limit,
    const std::vector<Move>& moves_to_base,
    std::vector<NodeToProcess>* receiver,
    TaskWorkspace* workspace) NO_THREAD_SAFETY_ANALYSIS {
  LCTRACE_FUNCTION_SCOPE;
  // TODO: Bring back pre-cached nodes created outside locks in a way that works
  // with tasks.
  // TODO: pre-reserve visits_to_perform for expected depth and likely maximum
  // width. Maybe even do so outside of lock scope.
  auto& vtp_buffer = workspace->vtp_buffer;
  auto& visits_to_perform = workspace->visits_to_perform;
  visits_to_perform.clear();
  auto& vtp_last_filled = workspace->vtp_last_filled;
  vtp_last_filled.clear();
  auto& current_path = workspace->current_path;
  current_path.clear();
  auto& moves_to_path = workspace->moves_to_path;
  moves_to_path = moves_to_base;
  // Sometimes receiver is reused, othertimes not, so only jump start if small.
  if (receiver->capacity() < 30) {
    receiver->reserve(receiver->size() + 30);
  }

  // These 2 are 'filled pre-emptively'.
  std::array<float, 256> current_pol;
  std::array<float, 256> current_util;

  // These 3 are 'filled on demand'.
  std::array<float, 256> current_score;
  std::array<int, 256> current_nstarted;
  auto& cur_iters = workspace->cur_iters;

  Node::Iterator best_edge;
  Node::Iterator second_best_edge;
  // Fetch the current best root node visits for possible smart pruning.
  const int64_t best_node_n = search_->current_best_edge_.GetN();

  int passed_off = 0;
  int completed_visits = 0;

  bool is_root_node = node == search_->root_node_;
  const float even_draw_score = search_->GetDrawScore(false);
  const float odd_draw_score = search_->GetDrawScore(true);
  const auto& root_move_filter = search_->root_move_filter_;
  auto m_evaluator = moves_left_support_ ? MEvaluator(params_) : MEvaluator();

  int max_limit = std::numeric_limits<int>::max();

  current_path.push_back(-1);
  while (current_path.size() > 0) {
    // First prepare visits_to_perform.
    if (current_path.back() == -1) {
      // Need to do n visits, where n is either collision_limit, or comes from
      // visits_to_perform for the current path.
      int cur_limit = collision_limit;
      if (current_path.size() > 1) {
        cur_limit =
            (*visits_to_perform.back())[current_path[current_path.size() - 2]];
      }
      // First check if node is terminal or not-expanded.  If either than create
      // a collision of appropriate size and pop current_path.
      if (node->GetN() == 0 || node->IsTerminal()) {
        if (is_root_node) {
          // Root node is special - since its not reached from anywhere else, so
          // it needs its own logic. Still need to create the collision to
          // ensure the outer gather loop gives up.
          if (node->TryStartScoreUpdate()) {
            cur_limit -= 1;
            minibatch_.push_back(NodeToProcess::Visit(
                node, static_cast<uint16_t>(current_path.size() + base_depth)));
            completed_visits++;
          }
        }
        // Visits are created elsewhere, just need the collisions here.
        if (cur_limit > 0) {
          int max_count = 0;
          if (cur_limit == collision_limit && base_depth == 0 &&
              max_limit > cur_limit) {
            max_count = max_limit;
          }
          receiver->push_back(NodeToProcess::Collision(
              node, static_cast<uint16_t>(current_path.size() + base_depth),
              cur_limit, max_count));
          completed_visits += cur_limit;
        }
        node = node->GetParent();
        current_path.pop_back();
        continue;
      }
      if (is_root_node) {
        // Root node is again special - needs its n in flight updated separately
        // as its not handled on the path to it, since there isn't one.
        node->IncrementNInFlight(cur_limit);
      }

      // Create visits_to_perform new back entry for this level.
      if (vtp_buffer.size() > 0) {
        visits_to_perform.push_back(std::move(vtp_buffer.back()));
        vtp_buffer.pop_back();
      } else {
        visits_to_perform.push_back(std::make_unique<std::array<int, 256>>());
      }
      vtp_last_filled.push_back(-1);

      // Cache all constant UCT parameters.
      // When we're near the leaves we can copy less of the policy, since there
      // is no way iteration will ever reach it.
      // TODO: This is a very conservative formula. It assumes every visit we're
      // aiming to add is going to trigger a new child, and that any visits
      // we've already had have also done so and then a couple extra since we go
      // to 2 unvisited to get second best in worst case.
      // Unclear we can do better without having already walked the children.
      // Which we are putting off until after policy is copied so we can create
      // visited policy without having to cache it in the node (allowing the
      // node to stay at 64 bytes).
      int max_needed = node->GetNumEdges();
      if (!is_root_node || root_move_filter.empty()) {
        max_needed = std::min(max_needed, node->GetNStarted() + cur_limit + 2);
      }
      node->CopyPolicy(max_needed, current_pol.data());
      for (int i = 0; i < max_needed; i++) {
        current_util[i] = std::numeric_limits<float>::lowest();
      }
      // Root depth is 1 here, while for GetDrawScore() it's 0-based, that's why
      // the weirdness.
      const float draw_score = ((current_path.size() + base_depth) % 2 == 0)
                                   ? odd_draw_score
                                   : even_draw_score;
      m_evaluator.SetParent(node);
      float visited_pol = 0.0f;
      for (Node* child : node->VisitedNodes()) {
        int index = child->Index();
        visited_pol += current_pol[index];
        float q = child->GetQ(draw_score);
        current_util[index] = q + m_evaluator.GetMUtility(child, q);
      }
      const float fpu =
          GetFpu(params_, node, is_root_node, draw_score, visited_pol);
      for (int i = 0; i < max_needed; i++) {
        if (current_util[i] == std::numeric_limits<float>::lowest()) {
          current_util[i] = fpu + m_evaluator.GetDefaultMUtility();
        }
      }

      const float cpuct = ComputeCpuct(params_, node->GetN(), is_root_node);
      const float puct_mult =
          cpuct * std::sqrt(std::max(node->GetChildrenVisits(), 1u));
      int cache_filled_idx = -1;

      // ─── Forced exploration pre-phase (root only) ──────────────────────
      // Per KataGo's "Forced Playouts" recipe (Wu 2019 §5.1), adapted for
      // lc0's batched multi-threaded MCTS per PR 2415.  Runs BEFORE the
      // natural PUCT loop and dispatches forced visits directly via
      // AddForcedExploration().  Each forced edge gets its
      // n_forced(c) = sqrt(P(c) * (children_visits + parent_n_in_flight) * factor)
      // visits accounted for here; the natural PUCT loop then runs with
      // whatever cur_limit remains.
      //
      // Eligibility rules (KataGo paper + PR 2415):
      //   - Only at root_node_.
      //   - Stop iterating at the first edge with GetN() == 0; forcing
      //     waits until natural PUCT has given the edge its first visit.
      //   - Skip terminal edges (no exploration value).
      //
      // Composes with the advisor mechanism: AddForcedExploration takes
      // the advisor's move + min_visits and floors the per-edge minimum
      // at max(policy_n_forced, advisor_min_visits) for the matching edge.
      if (is_root_node) {
        const float factor = params_.GetForcedExplorationFactor();
        const int advisor_min = search_->advisor_min_visits_;
        if (factor > 0.0f || advisor_min > 0) {
          while (cache_filled_idx + 1 < max_needed && cur_limit > 0) {
            const int idx = cache_filled_idx + 1;
            if (idx == 0) {
              cur_iters[idx] = node->Edges();
            } else {
              cur_iters[idx] = cur_iters[idx - 1];
              ++cur_iters[idx];
            }
            current_nstarted[idx] = cur_iters[idx].GetNStarted();
            // Initialize visits_to_perform[idx] = 0; the natural PUCT
            // loop will += to it if it later picks this edge.
            (*visits_to_perform.back())[idx] = 0;
            if (vtp_last_filled.back() < idx) vtp_last_filled.back() = idx;

            const bool is_unvisited = (cur_iters[idx].GetN() == 0);
            const bool is_terminal = cur_iters[idx].IsTerminal();
            // Honor --searchmoves: if a root_move_filter is set, only
            // edges in that filter are allowed to receive (any) visits,
            // forced included.  Without this guard, forced exploration
            // would dispatch visits to moves the user explicitly
            // excluded — corrupting analysis under UCI searchmoves.
            const bool in_filter =
                root_move_filter.empty() ||
                std::find(root_move_filter.begin(), root_move_filter.end(),
                          cur_iters[idx].GetMove()) != root_move_filter.end();
            if (!is_unvisited && !is_terminal && in_filter) {
              const int new_forced = AddForcedExploration(
                  params_, node, cur_iters[idx], cur_limit,
                  current_pol[idx], current_nstarted[idx],
                  search_->advisor_move_, advisor_min);
              if (new_forced > 0) {
                (*visits_to_perform.back())[idx] += new_forced;
              }
            }
            // CRITICAL: The natural PUCT loop below checks
            // `if (idx > cache_filled_idx)` to decide whether to compute
            // `current_score[idx]`.  If we advance cache_filled_idx
            // without populating current_score[idx], the inner loop
            // reads uninitialized memory and picks edges at random,
            // which silently destroys search quality at root.  Compute
            // it here for every idx we touch, post-AddForcedExploration
            // (so nstarted reflects the just-allocated forced visits).
            current_score[idx] =
                current_pol[idx] * puct_mult /
                    (1 + current_nstarted[idx]) +
                current_util[idx];
            ++cache_filled_idx;

            if (is_unvisited) {
              // KataGo paper: don't force on edges that haven't
              // received their first natural PUCT visit yet.  This idx
              // is cached so the natural PUCT loop sees it; stop here.
              break;
            }
          }
        }
      }

      while (cur_limit > 0) {
        // Perform UCT for current node.
        float best = std::numeric_limits<float>::lowest();
        int best_idx = -1;
        float best_without_u = std::numeric_limits<float>::lowest();
        float second_best = std::numeric_limits<float>::lowest();
        bool can_exit = false;
        best_edge.Reset();
        for (int idx = 0; idx < max_needed; ++idx) {
          if (idx > cache_filled_idx) {
            if (idx == 0) {
              cur_iters[idx] = node->Edges();
            } else {
              cur_iters[idx] = cur_iters[idx - 1];
              ++cur_iters[idx];
            }
            current_nstarted[idx] = cur_iters[idx].GetNStarted();
          }
          int nstarted = current_nstarted[idx];
          const float util = current_util[idx];
          if (idx > cache_filled_idx) {
            current_score[idx] =
                current_pol[idx] * puct_mult / (1 + nstarted) + util;
            cache_filled_idx++;
          }
          if (is_root_node) {
            // If there's no chance to catch up to the current best node with
            // remaining playouts, don't consider it.
            // best_move_node_ could have changed since best_node_n was
            // retrieved. To ensure we have at least one node to expand, always
            // include current best node.
            if (cur_iters[idx] != search_->current_best_edge_ &&
                latest_time_manager_hints_.GetEstimatedRemainingPlayouts() <
                    best_node_n - cur_iters[idx].GetN()) {
              continue;
            }
            // If root move filter exists, make sure move is in the list.
            if (!root_move_filter.empty() &&
                std::find(root_move_filter.begin(), root_move_filter.end(),
                          cur_iters[idx].GetMove()) == root_move_filter.end()) {
              continue;
            }
          }

          float score = current_score[idx];
          // Note: forced-exploration overrides were previously applied
          // here as `score = max_float` overrides.  That approach was
          // wrong for lc0's batched + multi-threaded MCTS — it ignored
          // virtual loss (used N instead of nstarted), didn't include
          // parent_n_in_flight in the quota formula, and didn't guard
          // against unvisited / terminal edges.  Forced exploration is
          // now handled in the pre-phase above (AddForcedExploration);
          // by the time we reach this PUCT loop, the forced visits have
          // already been allocated into visits_to_perform[idx].
          if (score > best) {
            second_best = best;
            second_best_edge = best_edge;
            best = score;
            best_idx = idx;
            best_without_u = util;
            best_edge = cur_iters[idx];
          } else if (score > second_best) {
            second_best = score;
            second_best_edge = cur_iters[idx];
          }
          if (can_exit) break;
          if (nstarted == 0) {
            // One more loop will get 2 unvisited nodes, which is sufficient to
            // ensure second best is correct. This relies upon the fact that
            // edges are sorted in policy decreasing order.
            can_exit = true;
          }
        }
        int new_visits = 0;
        if (second_best_edge) {
          int estimated_visits_to_change_best = std::numeric_limits<int>::max();
          if (best_without_u < second_best) {
            const auto n1 = current_nstarted[best_idx] + 1;
            estimated_visits_to_change_best = static_cast<int>(
                std::max(1.0f, std::min(current_pol[best_idx] * puct_mult /
                                                (second_best - best_without_u) -
                                            n1 + 1,
                                        1e9f)));
          }
          second_best_edge.Reset();
          max_limit = std::min(max_limit, estimated_visits_to_change_best);
          new_visits = std::min(cur_limit, estimated_visits_to_change_best);
        } else {
          // No second best - only one edge, so everything goes in here.
          new_visits = cur_limit;
        }
        if (best_idx >= vtp_last_filled.back()) {
          auto* vtp_array = visits_to_perform.back().get()->data();
          std::fill(vtp_array + (vtp_last_filled.back() + 1),
                    vtp_array + best_idx + 1, 0);
        }
        (*visits_to_perform.back())[best_idx] += new_visits;
        cur_limit -= new_visits;
        Node* child_node = best_edge.GetOrSpawnNode(/* parent */ node);

        // Probably best place to check for two-fold draws consistently.
        // Depth starts with 1 at root, so real depth is depth - 1.
        EnsureNodeTwoFoldCorrectForDepth(
            child_node, current_path.size() + base_depth + 1 - 1);

        bool decremented = false;
        if (child_node->TryStartScoreUpdate()) {
          current_nstarted[best_idx]++;
          new_visits -= 1;
          decremented = true;
          if (child_node->GetN() > 0 && !child_node->IsTerminal()) {
            child_node->IncrementNInFlight(new_visits);
            current_nstarted[best_idx] += new_visits;
          }
          current_score[best_idx] = current_pol[best_idx] * puct_mult /
                                        (1 + current_nstarted[best_idx]) +
                                    current_util[best_idx];
        }
        if ((decremented &&
             (child_node->GetN() == 0 || child_node->IsTerminal()))) {
          // Reduce 1 for the visits_to_perform to ensure the collision created
          // doesn't include this visit.
          (*visits_to_perform.back())[best_idx] -= 1;
          receiver->push_back(NodeToProcess::Visit(
              child_node,
              static_cast<uint16_t>(current_path.size() + 1 + base_depth)));
          completed_visits++;
          receiver->back().moves_to_visit.reserve(moves_to_path.size() + 1);
          receiver->back().moves_to_visit = moves_to_path;
          receiver->back().moves_to_visit.push_back(best_edge.GetMove());
        }
        if (best_idx > vtp_last_filled.back() &&
            (*visits_to_perform.back())[best_idx] > 0) {
          vtp_last_filled.back() = best_idx;
        }
      }
      is_root_node = false;
      // Actively do any splits now rather than waiting for potentially long
      // tree walk to get there.
      for (int i = 0; i <= vtp_last_filled.back(); i++) {
        int child_limit = (*visits_to_perform.back())[i];
        if (task_workers_ > 0 &&
            child_limit > params_.GetMinimumWorkSizeForPicking() &&
            child_limit <
                ((collision_limit - passed_off - completed_visits) * 2 / 3) &&
            child_limit + passed_off + completed_visits <
                collision_limit -
                    params_.GetMinimumRemainingWorkSizeForPicking()) {
          Node* child_node = cur_iters[i].GetOrSpawnNode(/* parent */ node);
          // Don't split if not expanded or terminal.
          if (child_node->GetN() == 0 || child_node->IsTerminal()) continue;

          bool passed = false;
          {
            // Multiple writers, so need mutex here.
            Mutex::Lock lock(picking_tasks_mutex_);
            // Ensure not to exceed size of reservation.
            if (picking_tasks_.size() < MAX_TASKS) {
              moves_to_path.push_back(cur_iters[i].GetMove());
              picking_tasks_.emplace_back(
                  child_node, current_path.size() - 1 + base_depth + 1,
                  moves_to_path, child_limit);
              moves_to_path.pop_back();
              task_count_.fetch_add(1, std::memory_order_acq_rel);
              task_added_.notify_all();
              passed = true;
              passed_off += child_limit;
            }
          }
          if (passed) {
            (*visits_to_perform.back())[i] = 0;
          }
        }
      }
      // Fall through to select the first child.
    }
    int min_idx = current_path.back();
    bool found_child = false;
    if (vtp_last_filled.back() > min_idx) {
      int idx = -1;
      for (auto& child : node->Edges()) {
        idx++;
        if (idx > min_idx && (*visits_to_perform.back())[idx] > 0) {
          if (moves_to_path.size() != current_path.size() + base_depth) {
            moves_to_path.push_back(child.GetMove());
          } else {
            moves_to_path.back() = child.GetMove();
          }
          current_path.back() = idx;
          current_path.push_back(-1);
          node = child.GetOrSpawnNode(/* parent */ node);
          found_child = true;
          break;
        }
        if (idx >= vtp_last_filled.back()) break;
      }
    }
    if (!found_child) {
      node = node->GetParent();
      if (!moves_to_path.empty()) moves_to_path.pop_back();
      current_path.pop_back();
      vtp_buffer.push_back(std::move(visits_to_perform.back()));
      visits_to_perform.pop_back();
      vtp_last_filled.pop_back();
    }
  }
}

void SearchWorker::ExtendNode(Node* node, int depth,
                              const std::vector<Move>& moves_to_node,
                              PositionHistory* history) {
  // Initialize position sequence with pre-move position.
  history->Trim(search_->played_history_.GetLength());
  for (size_t i = 0; i < moves_to_node.size(); i++) {
    history->Append(moves_to_node[i]);
  }

  // We don't need the mutex because other threads will see that N=0 and
  // N-in-flight=1 and will not touch this node.
  const auto& board = history->Last().GetBoard();
  auto legal_moves = board.GenerateLegalMoves();

  // Check whether it's a draw/lose by position. Importantly, we must check
  // these before doing the by-rule checks below.
  if (legal_moves.empty()) {
    // Could be a checkmate or a stalemate
    if (board.IsUnderCheck()) {
      node->MakeTerminal(GameResult::WHITE_WON);
    } else {
      node->MakeTerminal(GameResult::DRAW);
    }
    return;
  }

  // We can shortcircuit these draws-by-rule only if they aren't root;
  // if they are root, then thinking about them is the point.
  if (node != search_->root_node_) {
    if (!board.HasMatingMaterial()) {
      node->MakeTerminal(GameResult::DRAW);
      return;
    }

    if (history->Last().GetRule50Ply() >= 100) {
      node->MakeTerminal(GameResult::DRAW);
      return;
    }

    const auto repetitions = history->Last().GetRepetitions();
    // Mark two-fold repetitions as draws according to settings.
    // Depth starts with 1 at root, so number of plies in PV is depth - 1.
    if (repetitions >= 2) {
      node->MakeTerminal(GameResult::DRAW);
      return;
    } else if (repetitions == 1 && depth - 1 >= 4 &&
               params_.GetTwoFoldDraws() &&
               depth - 1 >= history->Last().GetPliesSincePrevRepetition()) {
      const auto cycle_length = history->Last().GetPliesSincePrevRepetition();
      // use plies since first repetition as moves left; exact if forced draw.
      node->MakeTerminal(GameResult::DRAW, (float)cycle_length,
                         Node::Terminal::TwoFold);
      return;
    }

    // Neither by-position or by-rule termination, but maybe it's a TB position.
    if (search_->syzygy_tb_ && !search_->root_is_in_dtz_ &&
        board.castlings().no_legal_castle() &&
        history->Last().GetRule50Ply() == 0 &&
        (board.ours() | board.theirs()).count() <=
            search_->syzygy_tb_->max_cardinality()) {
      ProbeState state;
      const WDLScore wdl =
          search_->syzygy_tb_->probe_wdl(history->Last(), &state);
      // Only fail state means the WDL is wrong, probe_wdl may produce correct
      // result with a stat other than OK.
      if (state != FAIL) {
        // TB nodes don't have NN evaluation, assign M from parent node.
        float m = 0.0f;
        // Need a lock to access parent, in case MakeSolid is in progress.
        {
          SharedMutex::SharedLock lock(search_->nodes_mutex_);
          auto parent = node->GetParent();
          if (parent) {
            m = std::max(0.0f, parent->GetM() - 1.0f);
          }
        }
        // If the colors seem backwards, check the checkmate check above.
        if (wdl == WDL_WIN) {
          node->MakeTerminal(GameResult::BLACK_WON, m,
                             Node::Terminal::Tablebase);
        } else if (wdl == WDL_LOSS) {
          node->MakeTerminal(GameResult::WHITE_WON, m,
                             Node::Terminal::Tablebase);
        } else {  // Cursed wins and blessed losses count as draws.
          node->MakeTerminal(GameResult::DRAW, m, Node::Terminal::Tablebase);
        }
        search_->tb_hits_.fetch_add(1, std::memory_order_acq_rel);
        return;
      }
    }
  }

  // Add legal moves as edges of this node.
  node->CreateEdges(legal_moves);
}

// 2b. Copy collisions into shared collisions.
void SearchWorker::CollectCollisions() {
  LCTRACE_FUNCTION_SCOPE;
  SharedMutex::Lock lock(search_->nodes_mutex_);

  for (const NodeToProcess& node_to_process : minibatch_) {
    if (node_to_process.IsCollision()) {
      search_->shared_collisions_.emplace_back(node_to_process.node,
                                               node_to_process.multivisit);
    }
  }
}

// 3. Prefetch into cache.
// ~~~~~~~~~~~~~~~~~~~~~~~
void SearchWorker::MaybePrefetchIntoCache() {
  LCTRACE_FUNCTION_SCOPE;
  // TODO(mooskagh) Remove prefetch into cache if node collisions work well.
  // If there are requests to NN, but the batch is not full, try to prefetch
  // nodes which are likely useful in future.
  if (search_->stop_.load(std::memory_order_acquire)) return;
  if (computation_->UsedBatchSize() > 0 &&
      static_cast<int>(computation_->UsedBatchSize()) <
          params_.GetMaxPrefetchBatch()) {
    history_.Trim(search_->played_history_.GetLength());
    SharedMutex::SharedLock lock(search_->nodes_mutex_);
    PrefetchIntoCache(
        search_->root_node_,
        params_.GetMaxPrefetchBatch() - computation_->UsedBatchSize(), false);
  }
}

// Prefetches up to @budget nodes into cache. Returns number of nodes
// prefetched.
int SearchWorker::PrefetchIntoCache(Node* node, int budget, bool is_odd_depth) {
  const float draw_score = search_->GetDrawScore(is_odd_depth);
  if (budget <= 0) return 0;

  // We are in a leaf, which is not yet being processed.
  if (!node || node->GetNStarted() == 0) {
    if (search_->backend_->GetCachedEvaluation(
            EvalPosition{history_.GetPositions(), {}})) {
      // Make it return 0 to make it not use the slot, so that the function
      // tries hard to find something to cache even among unpopular moves.
      // In practice that slows things down a lot though, as it's not always
      // easy to find what to cache.
      return 1;
    }
    auto moves = history_.Last().GetBoard().GenerateLegalMoves();
    computation_->AddInput(EvalPosition{history_.GetPositions(), moves},
                           EvalResultPtr{});
    return 1;
  }

  assert(node);
  // n = 0 and n_in_flight_ > 0, that means the node is being extended.
  if (node->GetN() == 0) return 0;
  // The node is terminal; don't prefetch it.
  if (node->IsTerminal()) return 0;

  // Populate all subnodes and their scores.
  typedef std::pair<float, EdgeAndNode> ScoredEdge;
  // NOTE: This function recurses (see PrefetchIntoCache call below) and
  // continues iterating `scores` after the recursion returns.  A shared
  // per-worker buffer would be clobbered by the recursive call's
  // clear+refill, so this stays as a local allocation.  If this ever
  // becomes a hot bottleneck, the fix is a stack-of-buffers indexed by
  // recursion depth, not a single member buffer.
  std::vector<ScoredEdge> scores;
  const float cpuct =
      ComputeCpuct(params_, node->GetN(), node == search_->root_node_);
  const float puct_mult =
      cpuct * std::sqrt(std::max(node->GetChildrenVisits(), 1u));
  const float fpu =
      GetFpu(params_, node, node == search_->root_node_, draw_score);
  for (auto& edge : node->Edges()) {
    if (edge.GetP() == 0.0f) continue;
    // Flip the sign of a score to be able to easily sort.
    // TODO: should this use logit_q if set??
    scores.emplace_back(-edge.GetU(puct_mult) - edge.GetQ(fpu, draw_score),
                        edge);
  }

  size_t first_unsorted_index = 0;
  int total_budget_spent = 0;
  int budget_to_spend = budget;  // Initialize for the case where there's only
                                 // one child.
  for (size_t i = 0; i < scores.size(); ++i) {
    if (search_->stop_.load(std::memory_order_acquire)) break;
    if (budget <= 0) break;

    // Sort next chunk of a vector. 3 at a time. Most of the time it's fine.
    if (first_unsorted_index != scores.size() &&
        i + 2 >= first_unsorted_index) {
      const int new_unsorted_index =
          std::min(scores.size(), budget < 2 ? first_unsorted_index + 2
                                             : first_unsorted_index + 3);
      std::partial_sort(scores.begin() + first_unsorted_index,
                        scores.begin() + new_unsorted_index, scores.end(),
                        [](const ScoredEdge& a, const ScoredEdge& b) {
                          return a.first < b.first;
                        });
      first_unsorted_index = new_unsorted_index;
    }

    auto edge = scores[i].second;
    // Last node gets the same budget as prev-to-last node.
    if (i != scores.size() - 1) {
      // Sign of the score was flipped for sorting, so flip it back.
      const float next_score = -scores[i + 1].first;
      // TODO: As above - should this use logit_q if set?
      const float q = edge.GetQ(-fpu, draw_score);
      if (next_score > q) {
        budget_to_spend =
            std::min(budget, int(edge.GetP() * puct_mult / (next_score - q) -
                                 edge.GetNStarted()) +
                                 1);
      } else {
        budget_to_spend = budget;
      }
    }
    history_.Append(edge.GetMove());
    const int budget_spent =
        PrefetchIntoCache(edge.node(), budget_to_spend, !is_odd_depth);
    history_.Pop();
    budget -= budget_spent;
    total_budget_spent += budget_spent;
  }
  return total_budget_spent;
}

// 4. Run NN computation.
// ~~~~~~~~~~~~~~~~~~~~~~
void SearchWorker::RunNNComputation() {
  if (computation_->UsedBatchSize() > 0) computation_->ComputeBlocking();
}

// 5. Retrieve NN computations (and terminal values) into nodes.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void SearchWorker::FetchMinibatchResults() {
  LCTRACE_FUNCTION_SCOPE;
  // Populate NN/cached results, or terminal results, into nodes.
  for (auto& node_to_process : minibatch_) {
    FetchSingleNodeResult(&node_to_process);
  }
}

void SearchWorker::FetchSingleNodeResult(NodeToProcess* node_to_process) {
  if (node_to_process->IsCollision()) return;
  Node* node = node_to_process->node;
  if (!node_to_process->nn_queried) {
    // Terminal nodes don't involve the neural NetworkComputation, nor do
    // they require any further processing after value retrieval.
    node_to_process->eval->q = node->GetWL();
    node_to_process->eval->d = node->GetD();
    node_to_process->eval->m = node->GetM();
    return;
  }
  node_to_process->eval->q = -node_to_process->eval->q;
  // For NN results, we need to populate policy as well as value.
  // First the value...
  if (params_.GetWDLRescaleRatio() != 1.0f ||
      (params_.GetWDLRescaleDiff() != 0.0f &&
       search_->contempt_mode_ != ContemptMode::NONE)) {
    // Check whether root moves are from the set perspective.
    bool root_stm = (search_->contempt_mode_ == ContemptMode::BLACK) ==
                    search_->played_history_.Last().IsBlackToMove();
    auto sign = (root_stm ^ (node_to_process->depth & 1)) ? 1.0f : -1.0f;
    WDLRescale(node_to_process->eval->q, node_to_process->eval->d,
               params_.GetWDLRescaleRatio(),
               search_->contempt_mode_ == ContemptMode::NONE
                   ? 0
                   : params_.GetWDLRescaleDiff(),
               sign, false, params_.GetWDLMaxS());
  }
  // Blend in the optimistic-st policy head.  Geometric interpolation:
  //   P_blended(a) ∝ P_main(a)^(1-alpha) * P_opt(a)^alpha
  // then renormalized so Σ P_blended = 1.  Equivalent to a weighted
  // mean in LOG space:
  //   log P_blended(a) = (1-alpha) log P_main(a) + alpha log P_opt(a)
  //                      - log(Z)
  // where Z is the normalization constant.  Matches the intent
  // described in KataGo's searchparams.h ("interpolate geometrically
  // between raw policy and optimistic policy" — though their CPU
  // path passes policyOptimism as an NN input rather than blending
  // in CPU code; here we implement the equivalent at search time
  // because the network exposes two separate head outputs).
  //
  // Geometric vs linear behavioral difference:
  //   - alpha = 0 → P_blended = P_main (no change)            [identical]
  //   - alpha = 1 → P_blended = P_opt   (full replace)        [identical]
  //   - alpha ∈ (0, 1) → depends on whether the two heads agree:
  //     * If both heads peak on the same move: geometric is SHARPER
  //       than linear (peaks reinforce, valleys reinforce).
  //     * If they disagree: geometric is SMOOTHER (any factor that's
  //       near zero kills that edge — mass migrates to consensus
  //       moves where both heads have meaningful probability).
  //     Per-edge the geometric value is always ≤ the arithmetic mean
  //     (AM-GM inequality), so renormalization is what determines
  //     the final shape relative to linear.
  //
  // Applied BEFORE Dirichlet noise (which only fires at root anyway).
  //
  // Two alphas, by depth — matches KataGo's two-knob design
  // (rootPolicyOptimism vs policyOptimism in their search params):
  //   * Root nodes use --optimistic-policy-weight (typically 0.05–0.20,
  //     low because the root visit budget is large; the prior shapes
  //     a lot of search and we don't want to over-bias).
  //   * Non-root nodes use --optimistic-policy-weight-internal
  //     (typically 0.5–1.0; internal visit budget per edge is sparse,
  //     so the prior dominates exploration and biasing it harder
  //     toward tactical moves drives deeper tactical discovery).
  //
  // Both default 0.0 (off).  Three guards on the blend per node:
  //   1. alpha > 0 at this depth class (option enabled),
  //   2. p_optimistic is non-empty (backend exposes a second head),
  //   3. sizes match (defensive — should always hold when present).
  // If any guard fails we fall through to the single-head path.
  const bool is_root = (node == search_->root_node_);
  const float kOptAlpha =
      is_root ? params_.GetOptimisticPolicyWeight()
              : params_.GetOptimisticPolicyWeightInternal();
  const bool blend_optimistic =
      kOptAlpha > 0.0f &&
      !node_to_process->eval->p_optimistic.empty() &&
      node_to_process->eval->p_optimistic.size() ==
          node_to_process->eval->p.size();
  if (blend_optimistic && kOptAlpha == 1.0f) {
    // Fast path at alpha = 1.0 → pure optimistic.  Skip the pow/sum/
    // renorm chain (it mathematically reduces to p_opt) so that this
    // path is byte-identical to standalone policy_head=optimistic.
    // Floating-point precision in the pow() pipeline could otherwise
    // produce values that differ in the last few bits and over many
    // search steps compound into measurably different PUCT decisions.
    size_t p_idx = 0;
    for (auto& edge : node->Edges()) {
      edge.edge()->SetP(node_to_process->eval->p_optimistic[p_idx]);
      ++p_idx;
    }
  } else if (blend_optimistic) {
    // Two-pass: compute per-edge p_main^(1-α) * p_opt^α, accumulate
    // sum, then write renormalized values.  Small floor (1e-30) on
    // each prior prevents pow(0, x) underflow issues on the
    // pathological case of a near-zero prior; in practice both
    // heads are softmax outputs so values are bounded away from
    // exact zero for legal moves and the floor is never triggered.
    const float one_minus_a = 1.0f - kOptAlpha;
    constexpr float kFloor = 1e-30f;
    const size_t n_edges = node_to_process->eval->p.size();
    // Reuse a thread-local scratch buffer to avoid per-call heap
    // allocations.  Must be thread_local (NOT a SearchWorker member)
    // because FetchSingleNodeResult is called from both the main
    // SearchWorker thread (via FetchMinibatchResults) AND from task
    // worker threads (via ProcessPickedTask → out-of-order eval at
    // search.cc:2098).  A shared member buffer would race under
    // --task-workers > 0.  thread_local gives each calling thread its
    // own buffer, persisting across calls for the thread's lifetime —
    // capacity retained, so push_back hits the in-place fast path
    // after the first call.  At 4000 fetches/sec under blend mode,
    // this saves the per-call heap alloc + dealloc-on-scope-exit
    // pair that previously contended on the process-wide allocator
    // lock at parallelism=16.
    thread_local std::vector<float> blended;
    blended.clear();
    blended.reserve(n_edges);
    double sum = 0.0;
    for (size_t p_idx = 0; p_idx < n_edges; ++p_idx) {
      const float p_main = node_to_process->eval->p[p_idx];
      const float p_opt = node_to_process->eval->p_optimistic[p_idx];
      const float b =
          std::pow(std::max(p_main, kFloor), one_minus_a) *
          std::pow(std::max(p_opt, kFloor), kOptAlpha);
      blended.push_back(b);
      sum += b;
    }
    // Renormalize so the blended distribution sums to 1.  Falling
    // back to the main prior on a degenerate sum=0 case (impossible
    // in practice given the kFloor above, but defensive).
    if (sum > 0.0) {
      const float renorm = static_cast<float>(1.0 / sum);
      size_t p_idx = 0;
      for (auto& edge : node->Edges()) {
        edge.edge()->SetP(blended[p_idx] * renorm);
        ++p_idx;
      }
    } else {
      size_t p_idx = 0;
      for (auto& edge : node->Edges()) {
        edge.edge()->SetP(node_to_process->eval->p[p_idx]);
        ++p_idx;
      }
    }
  } else {
    // Blend disabled (or backend doesn't expose the second head):
    // use the main policy directly.
    for (size_t p_idx = 0; auto& edge : node->Edges()) {
      edge.edge()->SetP(node_to_process->eval->p[p_idx]);
      ++p_idx;
    }
  }
  // Add Dirichlet noise if enabled and at root.  Noise is applied AFTER
  // the optimistic blend so it perturbs the blended prior (matches the
  // "blend first, then noise" framing — the blended prior is what the
  // network believes, then we add exploration noise on top).
  if (params_.GetNoiseEpsilon() && node == search_->root_node_) {
    ApplyDirichletNoise(node, params_.GetNoiseEpsilon(),
                        params_.GetNoiseAlpha());
  }
  node->SortEdges();
}

// 6. Propagate the new nodes' information to all their parents in the tree.
// ~~~~~~~~~~~~~~
void SearchWorker::DoBackupUpdate() {
  LCTRACE_FUNCTION_SCOPE;
  // Nodes mutex for doing node updates.
  SharedMutex::Lock lock(search_->nodes_mutex_);

  bool work_done = number_out_of_order_ > 0;
  for (const NodeToProcess& node_to_process : minibatch_) {
    DoBackupUpdateSingleNode(node_to_process);
    if (!node_to_process.IsCollision()) {
      work_done = true;
    }
  }
  if (!work_done) return;
  search_->CancelSharedCollisions();
  search_->total_batches_ += 1;
}

void SearchWorker::DoBackupUpdateSingleNode(
    const NodeToProcess& node_to_process) REQUIRES(search_->nodes_mutex_) {
  Node* node = node_to_process.node;
  if (node_to_process.IsCollision()) {
    // Collisions are handled via shared_collisions instead.
    return;
  }

  // For the first visit to a terminal, maybe update parent bounds too.
  auto update_parent_bounds =
      params_.GetStickyEndgames() && node->IsTerminal() && !node->GetN();

  // Backup V value up to a root. After 1 visit, V = Q.
  float v = node_to_process.eval->q;
  float d = node_to_process.eval->d;
  float m = node_to_process.eval->m;
  int n_to_fix = 0;
  float v_delta = 0.0f;
  float d_delta = 0.0f;
  float m_delta = 0.0f;
  uint32_t solid_threshold =
      static_cast<uint32_t>(params_.GetSolidTreeThreshold());
  for (Node *n = node, *p; n != search_->root_node_->GetParent(); n = p) {
    p = n->GetParent();

    // Current node might have become terminal from some other descendant, so
    // backup the rest of the way with more accurate values.
    if (n->IsTerminal()) {
      v = n->GetWL();
      d = n->GetD();
      m = n->GetM();
    }
    n->FinalizeScoreUpdate(v, d, m, node_to_process.multivisit);
    if (n_to_fix > 0 && !n->IsTerminal()) {
      n->AdjustForTerminal(v_delta, d_delta, m_delta, n_to_fix);
    }
    if (n->GetN() >= solid_threshold) {
      if (n->MakeSolid() && n == search_->root_node_) {
        // If we make the root solid, the current_best_edge_ becomes invalid and
        // we should repopulate it.
        search_->current_best_edge_ =
            search_->GetBestChildNoTemperature(search_->root_node_, 0);
      }
    }

    // Nothing left to do without ancestors to update.
    if (!p) break;

    bool old_update_parent_bounds = update_parent_bounds;
    // If parent already is terminal further adjustment is not required.
    if (p->IsTerminal()) n_to_fix = 0;
    // Try setting parent bounds except the root or those already terminal.
    update_parent_bounds =
        update_parent_bounds && p != search_->root_node_ && !p->IsTerminal() &&
        MaybeSetBounds(p, m, &n_to_fix, &v_delta, &d_delta, &m_delta);

    // Q will be flipped for opponent.
    v = -v;
    v_delta = -v_delta;
    m++;

    // Update the stats.
    // Best move.
    // If update_parent_bounds was set, we just adjusted bounds on the
    // previous loop or there was no previous loop, so if n is a terminal, it
    // just became that way and could be a candidate for changing the current
    // best edge. Otherwise a visit can only change best edge if its to an edge
    // that isn't already the best and the new n is equal or greater to the old
    // n.
    if (p == search_->root_node_ &&
        ((old_update_parent_bounds && n->IsTerminal()) ||
         (n != search_->current_best_edge_.node() &&
          search_->current_best_edge_.GetN() <= n->GetN()))) {
      search_->current_best_edge_ =
          search_->GetBestChildNoTemperature(search_->root_node_, 0);
    }
  }
  search_->total_playouts_ += node_to_process.multivisit;
  if (node_to_process.nn_queried && !node_to_process.is_cache_hit) {
    search_->network_evaluations_++;
  }
  search_->cum_depth_ += node_to_process.depth * node_to_process.multivisit;
  search_->max_depth_ = std::max(search_->max_depth_, node_to_process.depth);
}

bool SearchWorker::MaybeSetBounds(Node* p, float m, int* n_to_fix,
                                  float* v_delta, float* d_delta,
                                  float* m_delta) const {
  auto losing_m = 0.0f;
  auto prefer_tb = false;

  // Determine the maximum (lower, upper) bounds across all children.
  // (-1,-1) Loss (initial and lowest bounds)
  // (-1, 0) Can't Win
  // (-1, 1) Regular node
  // ( 0, 0) Draw
  // ( 0, 1) Can't Lose
  // ( 1, 1) Win (highest bounds)
  auto lower = GameResult::BLACK_WON;
  auto upper = GameResult::BLACK_WON;
  for (const auto& edge : p->Edges()) {
    const auto [edge_lower, edge_upper] = edge.GetBounds();
    lower = std::max(edge_lower, lower);
    upper = std::max(edge_upper, upper);

    // Checkmate is the best, so short-circuit.
    const auto is_tb = edge.IsTbTerminal();
    if (edge_lower == GameResult::WHITE_WON && !is_tb) {
      prefer_tb = false;
      break;
    } else if (edge_upper == GameResult::BLACK_WON) {
      // Track the longest loss.
      losing_m = std::max(losing_m, edge.GetM(0.0f));
    }
    prefer_tb = prefer_tb || is_tb;
  }

  // The parent's bounds are flipped from the children (-max(U), -max(L))
  // aggregated as if it was a single child (forced move) of the same bound.
  //       Loss (-1,-1) -> ( 1, 1) Win
  //  Can't Win (-1, 0) -> ( 0, 1) Can't Lose
  //    Regular (-1, 1) -> (-1, 1) Regular
  //       Draw ( 0, 0) -> ( 0, 0) Draw
  // Can't Lose ( 0, 1) -> (-1, 0) Can't Win
  //        Win ( 1, 1) -> (-1,-1) Loss

  // Nothing left to do for ancestors if the parent would be a regular node.
  if (lower == GameResult::BLACK_WON && upper == GameResult::WHITE_WON) {
    return false;
  } else if (lower == upper) {
    // Search can stop at the parent if the bounds can't change anymore, so make
    // it terminal preferring shorter wins and longer losses.
    *n_to_fix = p->GetN();
    assert(*n_to_fix > 0);
    float cur_v = p->GetWL();
    float cur_d = p->GetD();
    float cur_m = p->GetM();
    p->MakeTerminal(
        -upper,
        (upper == GameResult::BLACK_WON ? std::max(losing_m, m) : m) + 1.0f,
        prefer_tb ? Node::Terminal::Tablebase : Node::Terminal::EndOfGame);
    // Negate v_delta because we're calculating for the parent, but immediately
    // afterwards we'll negate v_delta in case it has come from the child.
    *v_delta = -(p->GetWL() - cur_v);
    *d_delta = p->GetD() - cur_d;
    *m_delta = p->GetM() - cur_m;
  } else {
    p->SetBounds(-upper, -lower);
  }

  // Bounds were set, so indicate we should check the parent too.
  return true;
}

// 7. Update the Search's status and progress information.
//~~~~~~~~~~~~~~~~~~~~
void SearchWorker::UpdateCounters() {
  LCTRACE_FUNCTION_SCOPE;
  search_->PopulateCommonIterationStats(&iteration_stats_);
  search_->MaybeTriggerStop(iteration_stats_, &latest_time_manager_hints_);
  search_->MaybeOutputInfo();

  // If this thread had no work, not even out of order, then sleep for some
  // milliseconds. Collisions don't count as work, so have to enumerate to find
  // out if there was anything done.
  bool work_done = number_out_of_order_ > 0;
  if (!work_done) {
    for (NodeToProcess& node_to_process : minibatch_) {
      if (!node_to_process.IsCollision()) {
        work_done = true;
        break;
      }
    }
  }
  if (!work_done) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}

}  // namespace classic
}  // namespace lczero
