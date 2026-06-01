/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2021 The LCZero Authors

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

#include "chess/pgn.h"
#include "chess/position.h"
#include "chess/uciloop.h"
#include "neural/backend.h"
#include "search/classic/search.h"
#include "search/classic/stoppers/stoppers.h"
#include "search/dag_classic/node.h"
#include "search/dag_classic/search.h"
#include "selfplay/external_engine.h"
#include "trainingdata/trainingdata.h"
#include "utils/optionsparser.h"

namespace lczero {

struct SelfPlayLimits {
  std::int64_t visits = -1;
  std::int64_t playouts = -1;
  std::int64_t movetime = -1;

  std::unique_ptr<classic::ChainedSearchStopper> MakeSearchStopper() const;
};

struct PlayerOptions {
  using OpeningCallback = std::function<void(const Opening&)>;
  // Backend to use by the player.
  Backend* backend;
  // Callback when player moves.
  CallbackUciResponder::BestMoveCallback best_move_callback;
  // Callback when player outputs info.
  CallbackUciResponder::ThinkingCallback info_callback;
  // Callback when player discards a selected move due to low visits.
  OpeningCallback discarded_callback;
  // User options dictionary.
  const OptionsDict* uci_options;
  // Limits to use for every move.
  SelfPlayLimits search_limits;

  // External UCI opponent.  When `external_engine_path` is non-empty,
  // this side's moves are obtained from the named subprocess instead of
  // from lc0's MCTS.  Training data is NOT written for these moves (so a
  // game with an opponent on one side yields ~half as many training
  // positions as a pure-selfplay game).  The game result is still
  // recorded onto the lc0-side positions at game end via the existing
  // WriteTrainingData() path.
  std::string external_engine_path;
  std::vector<std::string> external_engine_args;
  std::vector<std::pair<std::string, std::string>> external_engine_uci_options;
  // Argument string appended after "go " — e.g. "movetime 100", "depth 12".
  std::string external_engine_go_command = "movetime 100";

  // External advisor engine (independent of opponent above).  When set,
  // lc0 consults this engine for a recommended move at each lc0-side
  // turn, then forces lc0's MCTS to spend `advisor_min_visits` visits
  // on that move at the root.  The trained policy still comes from
  // lc0's own MCTS visit distribution — the advisor only biases which
  // positions get explored, not what value/policy targets get written.
  //
  // Skipped automatically when this side is being played by an external
  // opponent (no MCTS happens, no advisor to inject).
  //
  // Cost is one extra UCI subprocess call per lc0 move; with
  // `advisor_engine_go_command = "movetime 100"` that's ~100ms per
  // move per game.  Tune lower for higher selfplay throughput.
  std::string advisor_engine_path;
  std::vector<std::string> advisor_engine_args;
  std::vector<std::pair<std::string, std::string>> advisor_engine_uci_options;
  std::string advisor_engine_go_command = "movetime 100";
  // Number of root-MCTS visits to force on the advisor's move.  Out of
  // a typical `--visits=800` budget, 20-50 forced visits gives the
  // advisor's choice meaningful exploration weight without dominating
  // the search.  0 disables advisor mode for this side.
  int advisor_min_visits = 0;
  // Probability of PLAYING the advisor's move (vs the temperature-sampled move)
  // at a turn where the advisor was consulted.  0 = never force-play (advisor
  // only forces search visits; conservative default).  >0 injects the advisor's
  // move so the net learns its OUTCOME via the value head; the policy target is
  // unaffected (stays PTP-clean).  Gated on temp-cutoff-move (no endgame inject).
  float advisor_force_play_prob = 0.0f;
  // Always force-play (and play out — resign is suppressed) a move the advisor
  // reports as a forced MATE for the side to move, regardless of
  // advisor_force_play_prob or the temperature cutoff.  A reported mate is
  // guaranteed correct, so this injects deep mating lines the low-visit search
  // can't see.  Default on.
  bool advisor_force_play_mates = true;
  // Disagreement gate (first cut): when true, probabilistic force-play only
  // fires if the advisor's move differs from the net's own best move — inject
  // only where the advisor actually disagrees, not where the net already agrees.
  // Mates bypass this gate.
  bool advisor_force_play_on_disagree = false;
};

// Plays a single game vs itself.
class SelfPlayGame {
 public:
  // Player options may point to the same network/cache/etc.
  // If shared_tree is true, search tree is reused between players.
  // (useful for training games). Otherwise the tree is separate for black
  // and white (useful i.e. when they use different networks).
  SelfPlayGame(PlayerOptions white, PlayerOptions black, bool shared_tree,
               const Opening& opening);

  // Populate command line options that it uses.
  static void PopulateUciParams(OptionsParser* options);

  // True if `opts` selects the dag-preview search (vs classic).  Exposed so
  // the tournament can refuse dag in code paths that don't honor it (e.g. the
  // batched value/policy-mode games run by MultiSelfPlayGames).
  static bool IsDagRequested(const OptionsDict& opts);

  // Starts the game and blocks until the game is finished.
  void Play(int white_threads, int black_threads, bool training,
            SyzygyTablebase* syzygy_tb, bool enable_resign = true);
  // Aborts the game currently played, doesn't matter if it's synchronous or
  // not.
  void Abort();

  // Number of ply used from the given opening.
  int GetStartPly() const { return start_ply_; }

  // Writes training data to a file.
  void WriteTrainingData(TrainingDataWriter* writer) const;

  GameResult GetGameResult() const { return game_result_; }
  std::vector<Move> GetMoves() const;
  // Gets the eval which required the biggest swing up to get the final outcome.
  // Eval is the expected outcome in the range 0<->1.
  float GetWorstEvalForWinnerOrDraw() const;
  int move_count_ = 0;
  uint64_t nodes_total_ = 0;

 private:
  // Per-side search-algorithm support.  Each side (white = index 0, black =
  // index 1) independently chooses "classic" or "dag-preview" via its
  // player's --search-algorithm option, so a single game can pit dag on one
  // side against classic on the other (the point of testing dag).  When
  // EITHER side uses dag the game runs through PlayPerSide(); when both sides
  // are classic the original Play() path runs untouched.
  //
  // Stage 1 scope: move generation only.  PlayPerSide throws if --training is
  // set or an external opponent/advisor is configured for a dag side (dag
  // training-data extraction is Stage 2).
  void PlayPerSide(int white_threads, int black_threads, bool training,
                   SyzygyTablebase* syzygy_tb, bool enable_resign);

  // options_[0] is for white player, [1] for black.
  PlayerOptions options_[2];
  // Node tree for player1 and player2. If the tree is shared between players,
  // tree_[0] == tree_[1].  For a side that uses dag-preview, the classic
  // tree_[s] is null and dag_tree_[s] is used instead.
  std::shared_ptr<classic::NodeTree> tree_[2];
  std::string orig_fen_;
  int start_ply_;

  // ── per-side dag-preview state (parallel to the classic tree_/search_) ──
  // side_uses_dag_[s] selects the engine for side s.  separate_trees_ is true
  // when the two sides keep distinct trees (always so when the sides differ
  // in engine; also when shared_tree was not requested).
  bool side_uses_dag_[2] = {false, false};
  bool separate_trees_ = true;
  // dag node trees, mirroring tree_[2] (only built for dag sides).
  std::shared_ptr<dag_classic::NodeTree> dag_tree_[2];
  // Per-tree transposition table (holds weak refs to the tree's low nodes, so
  // its lifetime is tied to the tree).  Only used for dag sides.
  dag_classic::TranspositionTable dag_tt_[2];
  // dag search in progress (Abort() stops whichever of search_/dag_search_
  // is active).  Declared after dag_tree_/dag_tt_ so it destructs first.
  std::unique_ptr<dag_classic::Search> dag_search_;

  // Search that is currently in progress. Stored in members so that Abort()
  // can stop it.
  std::unique_ptr<classic::Search> search_;
  bool abort_ = false;
  GameResult game_result_ = GameResult::UNDECIDED;
  bool adjudicated_ = false;
  // Set once the advisor reports a forced mate and we begin force-playing it;
  // suppresses resign for the rest of the game (both sides) so the mate plays
  // out to checkmate and the whole mating line is captured.
  bool playing_out_mate_ = false;
  // Track minimum eval for each player so that GetWorstEvalForWinnerOrDraw()
  // can be calculated after end of game.
  float min_eval_[2] = {1.0f, 1.0f};
  // Track the maximum eval for white win, draw, black win for comparison to
  // actual outcome.
  float max_eval_[3] = {0.0f, 0.0f, 0.0f};
  const bool chess960_;
  std::mutex mutex_;

  // Training data to send.
  V7TrainingDataArray training_data_;

  std::unique_ptr<SyzygyTablebase> syzygy_tb_;

  // External UCI engines per side, lazily instantiated when first needed.
  // Empty index = lc0 plays that side (MCTS).
  std::unique_ptr<ExternalEngine> external_engines_[2];

  // Advisor engines per side, lazily instantiated when first needed.
  // Only used when lc0 is playing this side (i.e. external_engines_[idx]
  // is null) AND advisor_engine_path is set in PlayerOptions.  The
  // advisor's move biases lc0's MCTS root visit distribution but does
  // not directly select the played move; lc0's own MCTS makes the
  // final choice.
  std::unique_ptr<ExternalEngine> advisor_engines_[2];
};

}  // namespace lczero
