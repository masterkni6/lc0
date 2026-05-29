/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2021 The LCZero Authors

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

#include "trainingdata/trainingdata.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <sstream>

namespace lczero {

namespace {
std::tuple<float, float> DriftCorrect(float q, float d) {
  // Training data doesn't have a high number of nodes, so there shouldn't be
  // too much drift. Highest known value not caused by backend bug was 1.5e-7.
  const float allowed_eps = 0.000001f;
  if (q > 1.0f) {
    if (q > 1.0f + allowed_eps) {
      CERR << "Unexpectedly large drift in q " << q;
    }
    q = 1.0f;
  }
  if (q < -1.0f) {
    if (q < -1.0f - allowed_eps) {
      CERR << "Unexpectedly large drift in q " << q;
    }
    q = -1.0f;
  }
  if (d > 1.0f) {
    if (d > 1.0f + allowed_eps) {
      CERR << "Unexpectedly large drift in d " << d;
    }
    d = 1.0f;
  }
  if (d < 0.0f) {
    if (d < 0.0f - allowed_eps) {
      CERR << "Unexpectedly large drift in d " << d;
    }
    d = 0.0f;
  }
  float w = (1.0f - d + q) / 2.0f;
  float l = w - q;
  // Assume q drift is rarer than d drift and apply all correction to d.
  if (w < 0.0f || l < 0.0f) {
    float drift = 2.0f * std::min(w, l);
    if (drift < -allowed_eps) {
      CERR << "Unexpectedly large drift correction for d based on q. " << drift;
    }
    d += drift;
    // Since q is in range -1 to 1 - this correction should never push d outside
    // of range, but precision could be lost in calculations so just in case.
    if (d < 0.0f) {
      d = 0.0f;
    }
  }
  return {q, d};
}
}  // namespace

void V7TrainingDataArray::Write(TrainingDataWriter* writer, GameResult result,
                                bool adjudicated) const {
  if (training_data_.empty()) return;
  // Base estimate off of best_m.  If needed external processing can use a
  // different approach.
  float m_estimate = training_data_.back().best_m + training_data_.size() - 1;
  for (auto chunk : training_data_) {
    bool black_to_move = chunk.side_to_move_or_enpassant;
    if (IsCanonicalFormat(static_cast<pblczero::NetworkFormat::InputFormat>(
            chunk.input_format))) {
      black_to_move = (chunk.invariance_info & (1u << 7)) != 0;
    }
    if (result == GameResult::WHITE_WON) {
      chunk.result_q = black_to_move ? -1 : 1;
      chunk.result_d = 0;
    } else if (result == GameResult::BLACK_WON) {
      chunk.result_q = black_to_move ? 1 : -1;
      chunk.result_d = 0;
    } else {
      chunk.result_q = 0;
      chunk.result_d = 1;
    }
    if (adjudicated) {
      chunk.invariance_info |= 1u << 5;  // Game adjudicated.
    }
    if (adjudicated && result == GameResult::UNDECIDED) {
      chunk.invariance_info |= 1u << 4;  // Max game length exceeded.
    }
    chunk.plies_left = m_estimate;
    m_estimate -= 1.0f;
    writer->WriteChunk(chunk);
  }
}

void V7TrainingDataArray::Add(
    const classic::Node* node, const PositionHistory& history,
    classic::Eval best_eval, classic::Eval played_eval, bool best_is_proven,
    Move best_move, Move played_move, std::span<Move> legal_moves,
    const std::optional<EvalResult>& nneval, float policy_softmax_temp,
    const std::vector<float>* processed_visits) {
  V7TrainingData result;
  const auto& position = history.Last();

  // Set version.  V7 is the canonical record format; selfplay populates
  // the V7-only fields with zeros and the rescorer fills them in later
  // via backward EMA (q_st / d_st) and lookahead (opp/next_played_idx).
  result.version = 7;
  result.input_format = input_format_;
  // V7 extras: zeroed at selfplay time, filled by the rescorer.
  result.d_st = 0.0f;
  result.opp_played_idx = 0;
  result.next_played_idx = 0;
  for (float& r : result.reserved) r = 0.0f;

  // Populate planes.
  int transform;
  InputPlanes planes = EncodePositionForNN(
      input_format_, history, 8, fill_empty_history_[position.IsBlackToMove()],
      &transform);
  int plane_idx = 0;
  for (auto& plane : result.planes) {
    plane = ReverseBitsInBytes(planes[plane_idx++].mask);
  }

  // Populate probabilities.
  //
  // `processed_visits` (if non-null) overrides raw edge.GetN() with the
  // KataGo-style policy-target-pruned counts from Search::
  // GetTrainingTargetVisits().  When non-null AND its size matches the
  // edge count, recompute total_n as the sum of pruned counts.
  // Otherwise (null, empty, or size mismatch) fall back to raw edge
  // visits.
  const bool use_pruned = (processed_visits != nullptr) &&
                          (processed_visits->size() == node->GetNumEdges());
  float total_n_f;
  if (use_pruned) {
    total_n_f = std::accumulate(processed_visits->begin(),
                                processed_visits->end(), 0.0f);
  } else {
    total_n_f = static_cast<float>(node->GetChildrenVisits());
  }
  // Prevent garbage/invalid training data from being uploaded to server.
  // It's possible to have N=0 when there is only one legal move in position
  // (due to smart pruning).
  if (total_n_f == 0.0f && node->GetNumEdges() != 1) {
    throw Exception("Search generated invalid data!");
  }
  // Set illegal moves to have -1 probability.
  std::fill(std::begin(result.probabilities), std::end(result.probabilities),
            -1);
  // Set moves probabilities according to their relative amount of visits.
  // Compute Kullback-Leibler divergence in nats (between policy and visits).
  float kld_sum = 0;
  float total = 0.0;
  size_t edge_idx = 0;
  for (const auto& child : node->Edges()) {
    const Move move = child.GetMove();
    // Per-edge N: use pruned count if provided AND sized correctly,
    // else raw.  Pruned[i] is aligned with the iteration order of
    // node->Edges() (search.cc produces them via the same iteration).
    const float n_for_target = use_pruned
                                   ? (*processed_visits)[edge_idx]
                                   : static_cast<float>(child.GetN());
    float fracv = total_n_f > 0.0f ? n_for_target / total_n_f : 1.0f;
    if (nneval) {
      size_t move_idx =
          std::find(legal_moves.begin(), legal_moves.end(), move) -
          legal_moves.begin();
      // Undo any softmax temperature in the cached data.
      float P = std::pow(nneval->p[move_idx], policy_softmax_temp);
      if (fracv > 0) {
        kld_sum += fracv * std::log(fracv / P);
      }
      total += P;
    }
    result.probabilities[MoveToNNIndex(move, transform)] = fracv;
    ++edge_idx;
  }
  if (nneval) {
    // Add small epsilon for backward compatibility with earlier value of 0.
    auto epsilon = std::numeric_limits<float>::min();
    kld_sum = std::max(kld_sum + std::log(total), 0.0f) + epsilon;
  }
  result.policy_kld = kld_sum;

  // Debug: dump target-shape comparison for first N positions when
  // LC0_DEBUG_TARGET_SHIFT=<N> is set in the environment.  Shows raw
  // visit distribution vs the actual written target side-by-side, plus
  // pol_kld computed both ways.  Useful for verifying improved-policy
  // targets (Gumbel/Grill/PTP) reshape distributions sensibly and for
  // seeing how much pol_kld would shift if we wrote raw-N instead.
  static std::atomic<int> dbg_remaining{[]() {
    const char* env = std::getenv("LC0_DEBUG_TARGET_SHIFT");
    return env ? std::atoi(env) : 0;
  }()};
  if (nneval && dbg_remaining.load(std::memory_order_relaxed) > 0) {
    const int slot = dbg_remaining.fetch_sub(1, std::memory_order_relaxed);
    if (slot > 0) {
      // Recompute raw-N target + its pol_kld for side-by-side display.
      float raw_total = 0.0f;
      for (const auto& c : node->Edges()) {
        raw_total += static_cast<float>(c.GetN());
      }
      struct Row {
        std::string mv;
        float P;
        float rawN;
        float raw_pi;
        float tgt_pi;
      };
      std::vector<Row> rows;
      rows.reserve(node->GetNumEdges());
      float raw_kld_acc = 0.0f, tgt_kld_acc = 0.0f, P_sum_dbg = 0.0f;
      size_t i = 0;
      for (const auto& c : node->Edges()) {
        const Move dbg_move = c.GetMove();
        size_t mi = std::find(legal_moves.begin(), legal_moves.end(),
                              dbg_move) -
                    legal_moves.begin();
        const float P = std::pow(nneval->p[mi], policy_softmax_temp);
        const float rawN = static_cast<float>(c.GetN());
        const float raw_pi = raw_total > 0.0f ? rawN / raw_total : 0.0f;
        const float tgt_pi =
            total_n_f > 0.0f
                ? (use_pruned ? (*processed_visits)[i] : rawN) / total_n_f
                : 0.0f;
        if (raw_pi > 0.0f) raw_kld_acc += raw_pi * std::log(raw_pi / P);
        if (tgt_pi > 0.0f) tgt_kld_acc += tgt_pi * std::log(tgt_pi / P);
        P_sum_dbg += P;
        rows.push_back({dbg_move.ToString(true), P, rawN, raw_pi, tgt_pi});
        ++i;
      }
      const float raw_kld_dbg =
          std::max(raw_kld_acc + std::log(P_sum_dbg), 0.0f);
      const float tgt_kld_dbg =
          std::max(tgt_kld_acc + std::log(P_sum_dbg), 0.0f);
      std::sort(rows.begin(), rows.end(),
                [](const Row& a, const Row& b) { return a.rawN > b.rawN; });
      std::ostringstream oss;
      oss.setf(std::ios::fixed);
      oss.precision(4);
      oss << "[DBG TGT] slot=" << slot
          << " raw_total_N=" << raw_total
          << " tgt_total=" << total_n_f
          << " use_pruned=" << (use_pruned ? "yes" : "no")
          << " pol_kld raw=" << raw_kld_dbg
          << " tgt=" << tgt_kld_dbg
          << " ratio=" << (raw_kld_dbg > 0 ? tgt_kld_dbg / raw_kld_dbg : 0.0f);
      const int top = std::min<int>(8, static_cast<int>(rows.size()));
      for (int j = 0; j < top; ++j) {
        oss << "\n  " << rows[j].mv
            << "  P=" << rows[j].P
            << "  N=" << rows[j].rawN
            << "  raw_pi=" << rows[j].raw_pi
            << "  tgt_pi=" << rows[j].tgt_pi
            << "  d=" << (rows[j].tgt_pi - rows[j].raw_pi);
      }
      CERR << oss.str();
    }
  }

  const auto& castlings = position.GetBoard().castlings();
  // Populate castlings.
  // For non-frc trained nets, just send 1 like we used to.
  uint8_t our_queen_side = 1;
  uint8_t our_king_side = 1;
  uint8_t their_queen_side = 1;
  uint8_t their_king_side = 1;
  // If frc trained, send the bit mask representing rook position.
  if (Is960CastlingFormat(input_format_)) {
    our_queen_side <<= castlings.our_queenside_rook.idx;
    our_king_side <<= castlings.our_kingside_rook.idx;
    their_queen_side <<= castlings.their_queenside_rook.idx;
    their_king_side <<= castlings.their_kingside_rook.idx;
  }

  result.castling_us_ooo = castlings.we_can_000() ? our_queen_side : 0;
  result.castling_us_oo = castlings.we_can_00() ? our_king_side : 0;
  result.castling_them_ooo = castlings.they_can_000() ? their_queen_side : 0;
  result.castling_them_oo = castlings.they_can_00() ? their_king_side : 0;

  // Other params.
  if (IsCanonicalFormat(input_format_)) {
    result.side_to_move_or_enpassant =
        position.GetBoard().en_passant().as_int() >> 56;
    if ((transform & FlipTransform) != 0) {
      result.side_to_move_or_enpassant =
          ReverseBitsInBytes(result.side_to_move_or_enpassant);
    }
    // Send transform in deprecated move count so rescorer can reverse it to
    // calculate the actual move list from the input data.
    result.invariance_info =
        transform | (position.IsBlackToMove() ? (1u << 7) : 0u);
  } else {
    result.side_to_move_or_enpassant = position.IsBlackToMove() ? 1 : 0;
    result.invariance_info = 0;
  }
  if (best_is_proven) {
    result.invariance_info |= 1u << 3;  // Best node is proven best;
  }
  result.dummy = 0;
  result.rule50_count = position.GetRule50Ply();

  // Game result is undecided.
  result.result_q = 0;
  result.result_d = 1;

  classic::Eval orig_eval;
  if (nneval) {
    orig_eval.wl = nneval->q;
    orig_eval.d = nneval->d;
    orig_eval.ml = nneval->m;
  } else {
    orig_eval.wl = std::numeric_limits<float>::quiet_NaN();
    orig_eval.d = std::numeric_limits<float>::quiet_NaN();
    orig_eval.ml = std::numeric_limits<float>::quiet_NaN();
  }

  // Aggregate evaluation WL.
  result.root_q = -node->GetWL();
  result.best_q = best_eval.wl;
  result.played_q = played_eval.wl;
  result.orig_q = orig_eval.wl;

  // Draw probability of WDL head.
  result.root_d = node->GetD();
  result.best_d = best_eval.d;
  result.played_d = played_eval.d;
  result.orig_d = orig_eval.d;

  std::tie(result.best_q, result.best_d) =
      DriftCorrect(result.best_q, result.best_d);
  std::tie(result.root_q, result.root_d) =
      DriftCorrect(result.root_q, result.root_d);
  std::tie(result.played_q, result.played_d) =
      DriftCorrect(result.played_q, result.played_d);

  result.root_m = node->GetM();
  result.best_m = best_eval.ml;
  result.played_m = played_eval.ml;
  result.orig_m = orig_eval.ml;

  result.visits = node->GetN();
  if (position.IsBlackToMove()) {
    best_move.Flip();
    played_move.Flip();
  }
  result.best_idx = MoveToNNIndex(best_move, transform);
  result.played_idx = MoveToNNIndex(played_move, transform);
  result.q_st = 0.0f;

  // Unknown here - will be filled in once the full data has been collected.
  result.plies_left = 0;
  training_data_.push_back(result);
}

void V7TrainingDataArray::AddPlaceholder(const PositionHistory& history,
                                         Move played_move) {
  // Build a "filler" chunk for a position whose move was selected by an
  // external engine.  This mirrors Add() but skips all MCTS-derived
  // fields (visits, root_q/d/m, KLD) and marks the chunk as a placeholder
  // so the rescorer drops it from its output.  We still need correct
  // plane data and a `played_idx` so the rescorer's plane-diff move
  // decoder finds the right transition between consecutive chunks.
  //
  // V7 extras stay zero-initialised (already zeroed by the `{}` above).
  // The rescorer's backward EMA pass overwrites q_st / d_st and the
  // lookahead pass overwrites opp_played_idx / next_played_idx — but
  // placeholders get dropped on rescorer output anyway, so these
  // fields are never seen by the trainer.
  V7TrainingData result = {};  // zero-init; covers most fields.
  const auto& position = history.Last();

  result.version = 7;
  result.input_format = input_format_;

  // Planes — same encoder as Add().  This is what the rescorer's
  // DecodeMoveFromInput diffs against the next chunk's planes.
  int transform;
  InputPlanes planes = EncodePositionForNN(
      input_format_, history, 8, fill_empty_history_[position.IsBlackToMove()],
      &transform);
  int plane_idx = 0;
  for (auto& plane : result.planes) {
    plane = ReverseBitsInBytes(planes[plane_idx++].mask);
  }

  // Castling, side-to-move, and transform/invariance bookkeeping — needed
  // so the rescorer's input-format checks and Validate() pass.
  const auto& castlings = position.GetBoard().castlings();
  uint8_t our_queen_side = 1, our_king_side = 1;
  uint8_t their_queen_side = 1, their_king_side = 1;
  if (Is960CastlingFormat(input_format_)) {
    our_queen_side <<= castlings.our_queenside_rook.idx;
    our_king_side <<= castlings.our_kingside_rook.idx;
    their_queen_side <<= castlings.their_queenside_rook.idx;
    their_king_side <<= castlings.their_kingside_rook.idx;
  }
  result.castling_us_ooo = castlings.we_can_000() ? our_queen_side : 0;
  result.castling_us_oo = castlings.we_can_00() ? our_king_side : 0;
  result.castling_them_ooo = castlings.they_can_000() ? their_queen_side : 0;
  result.castling_them_oo = castlings.they_can_00() ? their_king_side : 0;
  if (IsCanonicalFormat(input_format_)) {
    result.side_to_move_or_enpassant =
        position.GetBoard().en_passant().as_int() >> 56;
    if ((transform & FlipTransform) != 0) {
      result.side_to_move_or_enpassant =
          ReverseBitsInBytes(result.side_to_move_or_enpassant);
    }
    result.invariance_info =
        transform | (position.IsBlackToMove() ? (1u << 7) : 0u);
  } else {
    result.side_to_move_or_enpassant = position.IsBlackToMove() ? 1 : 0;
    result.invariance_info = 0;
  }
  // Placeholder bit (1u << 6). Rescorer drops these on output; the
  // PyTorch trainer never sees them.
  result.invariance_info |= 1u << 6;
  result.rule50_count = position.GetRule50Ply();

  // Probabilities: one-hot on the played move so the rescorer's prob-sum
  // validator (must sum to ~1.0) is satisfied without us inventing a
  // synthetic policy distribution.  All other slots marked illegal (-1)
  // — placeholder chunks should never be used as a policy target anyway.
  std::fill(std::begin(result.probabilities), std::end(result.probabilities),
            -1.0f);
  if (position.IsBlackToMove()) played_move.Flip();
  const int idx = MoveToNNIndex(played_move, transform);
  result.probabilities[idx] = 1.0f;
  result.played_idx = idx;
  result.best_idx = idx;

  // visits=0 tells the rescorer this is non-MCTS data; combined with the
  // placeholder bit, all per-move sanity checks (legality, played_idx
  // match) are skipped for this chunk.
  result.visits = 0;
  result.policy_kld = 0.0f;
  result.q_st = 0.0f;

  // Q/D/M fields: 0 is a valid sentinel (in-range for DriftCorrect).
  // result_q/d get overwritten with the game outcome in Write() anyway.
  // orig_q/d/m use NaN to mark "no nneval available" (Add() does this too).
  result.root_q = 0.0f;
  result.root_d = 1.0f;  // pure-draw prior, neutral default
  result.root_m = 0.0f;
  result.best_q = 0.0f;
  result.best_d = 1.0f;
  result.best_m = 0.0f;
  result.played_q = 0.0f;
  result.played_d = 1.0f;
  result.played_m = 0.0f;
  result.orig_q = std::numeric_limits<float>::quiet_NaN();
  result.orig_d = std::numeric_limits<float>::quiet_NaN();
  result.orig_m = std::numeric_limits<float>::quiet_NaN();
  result.result_q = 0.0f;
  result.result_d = 1.0f;
  result.plies_left = 0.0f;  // filled in by Write()

  training_data_.push_back(result);
}

}  // namespace lczero
