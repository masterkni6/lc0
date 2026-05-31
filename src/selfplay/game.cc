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

#include "selfplay/game.h"

#include <algorithm>

#include "chess/position.h"
#include "search/classic/stoppers/common.h"
#include "search/classic/stoppers/factory.h"
#include "utils/random.h"

namespace lczero {

namespace {
const OptionId kReuseTreeId{"reuse-tree", "ReuseTree",
                            "Reuse the search tree between moves."};
const OptionId kResignPercentageId{
    "resign-percentage", "ResignPercentage",
    "Resign when win percentage drops below specified value."};
const OptionId kResignWDLStyleId{
    "resign-wdlstyle", "ResignWDLStyle",
    "If set, resign percentage applies to any output state being above "
    "100% minus the percentage instead of winrate being below."};
const OptionId kResignEarliestMoveId{"resign-earliest-move",
                                     "ResignEarliestMove",
                                     "Earliest move that resign is allowed."};
const OptionId kMinimumAllowedVistsId{
    "minimum-allowed-visits", "MinimumAllowedVisits",
    "Unless the selected move is the best move, temperature based selection "
    "will be retried until visits of selected move is greater than or equal to "
    "this threshold."};
const OptionId kUciChess960{
    "chess960", "UCI_Chess960",
    "Castling moves are encoded as \"king takes rook\"."};
const OptionId kSyzygyTablebaseId{
    "syzygy-paths", "SyzygyPath",
    "List of Syzygy tablebase directories, list entries separated by system "
    "separator (\";\" for Windows, \":\" for Linux).",
    's'};
const OptionId kOpeningStopProbId{
    "opening-stop-prob", "OpeningStopProb",
    "From each opening move, start a self-play game with probability max(p, "
    "1/n), where p is the value given and n the opening moves remaining."};
const OptionId kSearchAlgorithmId{
    "search-algorithm", "SearchAlgorithm",
    "Which search to use for selfplay games: \"classic\" (default) or "
    "\"dag-preview\" (transposition-aware DAG search). Stage 1: dag-preview "
    "supports move generation only; training data with dag is not yet "
    "implemented and will throw if --training is set."};
}  // namespace

void SelfPlayGame::PopulateUciParams(OptionsParser* options) {
  options->Add<BoolOption>(kReuseTreeId) = false;
  options->Add<BoolOption>(kResignWDLStyleId) = false;
  options->Add<FloatOption>(kResignPercentageId, 0.0f, 100.0f) = 0.0f;
  options->Add<IntOption>(kResignEarliestMoveId, 0, 1000) = 0;
  options->Add<IntOption>(kMinimumAllowedVistsId, 0, 1000000) = 0;
  options->Add<BoolOption>(kUciChess960) = false;
  PopulateTimeManagementOptions(classic::RunType::kSelfplay, options);
  options->Add<StringOption>(kSyzygyTablebaseId);
  options->Add<FloatOption>(kOpeningStopProbId, 0.0f, 1.0f) = 0.0f;
  std::vector<std::string> search_algorithms = {"classic", "dag-preview"};
  options->Add<ChoiceOption>(kSearchAlgorithmId, search_algorithms) = "classic";
}

SelfPlayGame::SelfPlayGame(PlayerOptions white, PlayerOptions black,
                           bool shared_tree, const Opening& opening)
    : options_{white, black},
      chess960_{white.uci_options->Get<bool>(kUciChess960) ||
                black.uci_options->Get<bool>(kUciChess960)},
      training_data_(classic::SearchParams(*white.uci_options).GetHistoryFill(),
                     classic::SearchParams(*black.uci_options).GetHistoryFill(),
                     pblczero::NetworkFormat::INPUT_CLASSICAL_112_PLANE) {
  orig_fen_ = opening.start_fen;
  side_uses_dag_[0] =
      white.uci_options->Get<std::string>(kSearchAlgorithmId) == "dag-preview";
  side_uses_dag_[1] =
      black.uci_options->Get<std::string>(kSearchAlgorithmId) == "dag-preview";
  // The two sides keep separate trees unless a shared tree was requested AND
  // both sides use the same engine (mixed dag/classic can't share a tree —
  // different node types).
  separate_trees_ = !shared_tree || (side_uses_dag_[0] != side_uses_dag_[1]);

  auto white_prob = white.uci_options->Get<float>(kOpeningStopProbId);
  auto black_prob = black.uci_options->Get<float>(kOpeningStopProbId);
  if (white_prob != black_prob && white_prob != 0 && black_prob != 0) {
    throw Exception("Stop probabilities must be both equal or zero!");
  }

  if (side_uses_dag_[0] || side_uses_dag_[1]) {
    // At least one side uses dag-preview.  Build each side's tree in its own
    // engine's node type; both trees track the same game (kept in sync by
    // applying every played move to both).  dag_classic::NodeTree exposes the
    // same ResetToPosition(fen, moves)/IsBlackToMove()/MakeMove() API as
    // classic::NodeTree.
    auto build_side = [&](int s) {
      if (side_uses_dag_[s]) {
        dag_tree_[s] = std::make_shared<dag_classic::NodeTree>();
        dag_tree_[s]->ResetToPosition(orig_fen_, {});
      } else {
        tree_[s] = std::make_shared<classic::NodeTree>();
        tree_[s]->ResetToPosition(orig_fen_, {});
      }
    };
    build_side(0);
    if (separate_trees_) {
      build_side(1);
    } else {
      // Same engine + shared tree requested: side 1 aliases side 0.
      if (side_uses_dag_[0]) {
        dag_tree_[1] = dag_tree_[0];
      } else {
        tree_[1] = tree_[0];
      }
    }
    auto side0_black = [&]() {
      return side_uses_dag_[0] ? dag_tree_[0]->IsBlackToMove()
                               : tree_[0]->IsBlackToMove();
    };
    auto make_move_all = [&](Move internal) {
      if (side_uses_dag_[0]) {
        dag_tree_[0]->MakeMove(internal);
      } else {
        tree_[0]->MakeMove(internal);
      }
      if (separate_trees_) {
        if (side_uses_dag_[1]) {
          dag_tree_[1]->MakeMove(internal);
        } else {
          tree_[1]->MakeMove(internal);
        }
      }
    };
    int ply = 0;
    for (Move m : opening.moves) {
      auto exit_prob_now = side0_black() ? black_prob : white_prob;
      auto exit_prob_next = side0_black() ? white_prob : black_prob;
      int positions = opening.moves.size() - ply + 1;
      if (exit_prob_now > 0.0f &&
          Random::Get().GetFloat(1.0f) <
              std::max(exit_prob_now,
                       exit_prob_now / (exit_prob_now * ((positions + 1) / 2) +
                                        exit_prob_next * (positions / 2)))) {
        break;
      }
      if (side0_black()) m.Flip();
      make_move_all(m);
      ply++;
    }
    start_ply_ = ply;
    return;
  }

  tree_[0] = std::make_shared<classic::NodeTree>();
  tree_[0]->ResetToPosition(orig_fen_, {});

  if (shared_tree) {
    tree_[1] = tree_[0];
  } else {
    tree_[1] = std::make_shared<classic::NodeTree>();
    tree_[1]->ResetToPosition(orig_fen_, {});
  }
  int ply = 0;

  for (Move m : opening.moves) {
    // For early exit from the opening, we support two cases: a) where both
    // sides have the same exit probability and b) where one side's exit
    // probability is zero. In the following formula, `positions` is the number
    // of possible exit points remaining, used for adjusting the exit
    // probability (to avoid favoring the last position).
    auto exit_prob_now = tree_[0]->IsBlackToMove() ? black_prob : white_prob;
    auto exit_prob_next = tree_[0]->IsBlackToMove() ? white_prob : black_prob;
    int positions = opening.moves.size() - ply + 1;
    if (exit_prob_now > 0.0f &&
        Random::Get().GetFloat(1.0f) <
            std::max(exit_prob_now,
                     exit_prob_now / (exit_prob_now * ((positions + 1) / 2) +
                                      exit_prob_next * (positions / 2)))) {
      break;
    }
    if (tree_[0]->IsBlackToMove()) m.Flip();
    tree_[0]->MakeMove(m);
    if (tree_[0] != tree_[1]) tree_[1]->MakeMove(m);
    ply++;
  }
  start_ply_ = ply;
}

void SelfPlayGame::Play(int white_threads, int black_threads, bool training,
                        SyzygyTablebase* syzygy_tb, bool enable_resign) {
  if (side_uses_dag_[0] || side_uses_dag_[1]) {
    PlayPerSide(white_threads, black_threads, training, syzygy_tb,
                enable_resign);
    return;
  }
  bool blacks_move = tree_[0]->IsBlackToMove();

  // Take syzygy tablebases from player1 options.
  std::string tb_paths =
      options_[0].uci_options->Get<std::string>(kSyzygyTablebaseId);
  if (!tb_paths.empty()) {  // && tb_paths != tb_paths_) {
    syzygy_tb_ = std::make_unique<SyzygyTablebase>();
    CERR << "Loading Syzygy tablebases from " << tb_paths;
    if (!syzygy_tb_->init(tb_paths)) {
      CERR << "Failed to load Syzygy tablebases!";
      syzygy_tb_ = nullptr;
    }
  }
  // Do moves while not end of the game. (And while not abort_)
  while (!abort_) {
    game_result_ = tree_[0]->GetPositionHistory().ComputeGameResult();

    // If endgame, stop.
    if (game_result_ != GameResult::UNDECIDED) break;
    if (tree_[0]->GetPositionHistory().Last().GetGamePly() >= 450) {
      adjudicated_ = true;
      break;
    }
    const int idx = blacks_move ? 1 : 0;

    // ─── External UCI opponent path ───
    // If this side is played by an external engine, get a move from it,
    // apply to the tree, and skip MCTS + training data entirely.  Game
    // result (recorded at game end) still gets written back to all the
    // lc0-side training positions via WriteTrainingData().
    if (!options_[idx].external_engine_path.empty()) {
      // FRC mode is driven by lc0's --chess960=true flag (stored in
      // chess960_).  SF doesn't auto-detect FRC from the FEN —
      // engines are told explicitly via the UCI_Chess960 option, so
      // we mirror that convention: the user passes --chess960=true
      // for FRC/DFRC games, and we propagate UCI_Chess960=true to
      // the external engine + emit FRC castling encoding.
      //
      // For standard chess (no --chess960 flag), we don't set
      // UCI_Chess960 on SF (so it plays at full strength), and we
      // emit castling in standard "e1g1" form (which SF accepts in
      // any mode).
      //
      // If chess960_ doesn't reflect --chess960=true at runtime
      // (option-context plumbing issue), DFRC games will desync —
      // SF will reject FRC castling moves.  We'll see this in the
      // adjudication diagnostics and can debug from there.
      const bool is_frc_position = chess960_;

      if (!external_engines_[idx]) {
        try {
          external_engines_[idx] = std::make_unique<ExternalEngine>(
              options_[idx].external_engine_path,
              options_[idx].external_engine_args,
              options_[idx].external_engine_uci_options,
              options_[idx].external_engine_go_command,
              /*chess960=*/is_frc_position);
        } catch (const Exception& e) {
          CERR << "External engine spawn failed on side "
               << (blacks_move ? "B" : "W") << ": " << e.what()
               << " — adjudicating as loss for opponent.";
          game_result_ =
              blacks_move ? GameResult::WHITE_WON : GameResult::BLACK_WON;
          adjudicated_ = true;
          break;
        }
      }

      // Build "position fen <X> moves <m1> <m2> ..." for Stockfish.  Use
      // the original starting FEN + accumulated move list rather than
      // the current FEN.  This lets the engine see the full game history
      // for repetition / 50-move-rule purposes, which matters when the
      // engine is configured with contempt or 3-fold-aware evaluation.
      //
      // Emit moves using FRC detection from the starting FEN (NOT the
      // chess960_ flag, which doesn't always reflect --chess960=true
      // due to lc0's option-context plumbing).  For FRC starting
      // positions we send king-takes-rook castling ("b1a1") so the
      // 960-mode engine recognizes it; for standard chess we send
      // standard "e1g1" so a non-960-mode engine recognizes it.
      // Encoding and the engine's UCI_Chess960 setting must agree —
      // both are now driven by is_frc_position above.
      const std::vector<Move> played_moves = GetMoves();
      std::vector<std::string> moves_uci;
      moves_uci.reserve(played_moves.size());
      for (const Move& m : played_moves) {
        moves_uci.push_back(m.ToString(/*is_chess960=*/is_frc_position));
      }

      std::string uci_move;
      try {
        uci_move = external_engines_[idx]->GetMove(orig_fen_, moves_uci);
      } catch (const Exception& e) {
        // Engine crashed / timed out.  Tear it down (so the next game's
        // Spawn() gets a fresh process) and adjudicate the current game
        // as a loss for the external-engine side so we don't write
        // garbage training data with no result.
        external_engines_[idx].reset();
        CERR << "External engine error on side " << (blacks_move ? "B" : "W")
             << ": " << e.what() << " — adjudicating as loss for opponent.";
        game_result_ = blacks_move ? GameResult::WHITE_WON : GameResult::BLACK_WON;
        adjudicated_ = true;
        break;
      }

      // Parse the UCI move into lc0's internal representation.  We parse
      // against the current side-to-move's board (which is internally
      // mirrored for black-to-move positions); ChessBoard::ParseMove
      // handles the un-mirroring of UCI coordinates automatically.
      Move move;
      try {
        move = tree_[idx]
                   ->GetPositionHistory()
                   .Last()
                   .GetBoard()
                   .ParseMove(uci_move);
      } catch (const Exception& e) {
        // Dump everything we know about the desync. We want:
        //   - the move SF returned
        //   - lc0's current view of the position (FEN)
        //   - lc0's view of the original starting FEN
        //   - the full move list we sent to SF
        // Together these let us reproduce SF's view by hand and
        // identify which move caused the divergence.
        std::string current_fen = PositionToFen(
            tree_[idx]->GetPositionHistory().Last());
        std::string moves_str;
        for (const auto& m : moves_uci) {
          moves_str += " ";
          moves_str += m;
        }
        CERR << "External engine returned unparseable move '" << uci_move
             << "': " << e.what() << " — adjudicating as loss for opponent.";
        CERR << "  side-to-move: " << (blacks_move ? "black" : "white");
        CERR << "  lc0-view FEN: " << current_fen;
        CERR << "  orig FEN sent to engine: " << orig_fen_;
        CERR << "  moves sent to engine:" << moves_str;
        external_engines_[idx].reset();
        game_result_ = blacks_move ? GameResult::WHITE_WON : GameResult::BLACK_WON;
        adjudicated_ = true;
        break;
      }
      // Intentionally do NOT increment move_count_ for external engine
      // moves. The tournament's `npm` stat is total lc0 MCTS nodes /
      // total moves — incrementing here would halve npm because external
      // moves contribute 0 nodes. Keeping move_count_ as a count of
      // lc0-side moves only makes npm = average visits per lc0 move,
      // which is what users actually want to see vs --visits=N.

      // Write a placeholder training chunk for this opponent move BEFORE
      // applying it.  The rescorer recovers the per-move list by diffing
      // consecutive chunks' position planes; without a placeholder for
      // each opponent move, consecutive lc0-side chunks would be 2 ply
      // apart and DecodeMoveFromInput would fail.  The chunk is marked
      // with `invariance_info & 64`; the rescorer drops it on output, so
      // PyTorch training only ever sees lc0-side positions.
      //
      // AddPlaceholder (like Add) expects the move in real-world UCI
      // coords — it does its own per-side flip internally.  Our `move`
      // is in storage form (from ParseMove), so convert by flipping for
      // a black-to-move position.
      if (training) {
        Move real_move = move;
        if (tree_[idx]->IsBlackToMove()) real_move.Flip();
        training_data_.AddPlaceholder(tree_[idx]->GetPositionHistory(),
                                       real_move);
      }

      // Apply move to both trees.
      //
      // Subtle: the MCTS path (below) does `if (IsBlackToMove) move.Flip()`
      // before MakeMove because Search::GetBestMove() returns the move in
      // real-world UCI coords (it calls Edge::GetMove(is_black) which
      // flips for black).  MakeMove walks `current_head_->Edges()` and
      // compares against Edge::move_ which is stored in white-perspective
      // form — hence the flip is needed to convert real-world → storage.
      //
      // ChessBoard::ParseMove already returns moves in storage form (it
      // un-flips the ranks internally before constructing Move::White),
      // so we MUST NOT flip again here.  Doing so would feed MakeMove a
      // move it can't find in the edge list, silently mis-applying to
      // an arbitrary CreateSingleChildNode fallback.
      tree_[0]->MakeMove(move);
      if (tree_[0] != tree_[1]) tree_[1]->MakeMove(move);
      blacks_move = !blacks_move;
      continue;
    }

    // Initialize search.
    if (!options_[idx].uci_options->Get<bool>(kReuseTreeId)) {
      tree_[idx]->TrimTreeAtHead();
    }

    // ─── Optional: consult advisor engine (e.g. Stockfish) ───
    // If an advisor engine is configured for this side AND lc0 is the
    // one playing this side (we're not in the external-opponent branch
    // above), query it for its preferred move.  We'll then tell MCTS to
    // force some visits onto that move at the root.  Failure is non-
    // fatal — if the advisor crashes or returns garbage, we tear it
    // down and proceed with normal MCTS (no adjudication; advisor is
    // best-effort, never authoritative).
    Move advisor_move;
    bool has_advisor_move = false;
    if (!options_[idx].advisor_engine_path.empty() &&
        options_[idx].advisor_min_visits > 0) {
      const bool is_frc_position = chess960_;
      if (!advisor_engines_[idx]) {
        try {
          advisor_engines_[idx] = std::make_unique<ExternalEngine>(
              options_[idx].advisor_engine_path,
              options_[idx].advisor_engine_args,
              options_[idx].advisor_engine_uci_options,
              options_[idx].advisor_engine_go_command,
              /*chess960=*/is_frc_position);
        } catch (const Exception& e) {
          CERR << "Advisor engine spawn failed on side "
               << (blacks_move ? "B" : "W") << ": " << e.what()
               << " — continuing without advisor for the rest of this game.";
          // Leave advisor_engines_[idx] null; we'll skip the query path
          // on every subsequent move of this game without retrying.
        }
      }
      if (advisor_engines_[idx]) {
        try {
          const std::vector<Move> played_moves = GetMoves();
          std::vector<std::string> moves_uci;
          moves_uci.reserve(played_moves.size());
          for (const Move& m : played_moves) {
            moves_uci.push_back(m.ToString(is_frc_position));
          }
          std::string uci_move =
              advisor_engines_[idx]->GetMove(orig_fen_, moves_uci);
          advisor_move = tree_[idx]
                             ->GetPositionHistory()
                             .Last()
                             .GetBoard()
                             .ParseMove(uci_move);
          has_advisor_move = true;
        } catch (const Exception& e) {
          // Engine crashed / timed out / parse failure on the move.
          // Tear it down so the next move's branch above tries to
          // respawn — and proceed with this move's MCTS *without*
          // advisor injection.  No adjudication: advisor failure isn't
          // a game-correctness issue.
          CERR << "Advisor engine query failed on side "
               << (blacks_move ? "B" : "W") << ": " << e.what()
               << " — skipping advisor for this move.";
          advisor_engines_[idx].reset();
        }
      }
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (abort_) break;
      auto stoppers = options_[idx].search_limits.MakeSearchStopper();
      classic::PopulateIntrinsicStoppers(stoppers.get(),
                                         *options_[idx].uci_options);

      std::unique_ptr<UciResponder> responder =
          std::make_unique<CallbackUciResponder>(
              options_[idx].best_move_callback, options_[idx].info_callback);

      search_ = std::make_unique<classic::Search>(
          *tree_[idx], options_[idx].backend, std::move(responder),
          /* searchmoves */ MoveList(), std::chrono::steady_clock::now(),
          std::move(stoppers), /* infinite */ false, /* ponder */ false,
          *options_[idx].uci_options, syzygy_tb);

      // Inject advisor move into search BEFORE starting workers.  The
      // override is a single Move + int set in Search; SearchWorker
      // reads them in its PUCT loop.  Must be set before RunBlocking
      // so all worker threads see the same value.
      if (has_advisor_move) {
        search_->SetAdvisorMove(advisor_move,
                                options_[idx].advisor_min_visits);
      }
    }

    // Do search.
    search_->RunBlocking(blacks_move ? black_threads : white_threads);
    move_count_++;
    nodes_total_ += search_->GetTotalPlayouts();
    if (abort_) break;

    Move best_move;
    bool best_is_terminal;
    const auto best_eval = search_->GetBestEval(&best_move, &best_is_terminal);
    float eval = best_eval.wl;
    eval = (eval + 1) / 2;
    if (eval < min_eval_[idx]) min_eval_[idx] = eval;
    const int move_number = tree_[0]->GetPositionHistory().GetLength() / 2 + 1;
    auto best_w = (best_eval.wl + 1.0f - best_eval.d) / 2.0f;
    auto best_d = best_eval.d;
    auto best_l = best_w - best_eval.wl;
    max_eval_[0] = std::max(max_eval_[0], blacks_move ? best_l : best_w);
    max_eval_[1] = std::max(max_eval_[1], best_d);
    max_eval_[2] = std::max(max_eval_[2], blacks_move ? best_w : best_l);
    if (enable_resign && move_number >= options_[idx].uci_options->Get<int>(
                                            kResignEarliestMoveId)) {
      const float resignpct =
          options_[idx].uci_options->Get<float>(kResignPercentageId) / 100;
      if (options_[idx].uci_options->Get<bool>(kResignWDLStyleId)) {
        auto threshold = 1.0f - resignpct;
        if (best_w > threshold) {
          game_result_ =
              blacks_move ? GameResult::BLACK_WON : GameResult::WHITE_WON;
          adjudicated_ = true;
          break;
        }
        if (best_l > threshold) {
          game_result_ =
              blacks_move ? GameResult::WHITE_WON : GameResult::BLACK_WON;
          adjudicated_ = true;
          break;
        }
        if (best_d > threshold) {
          game_result_ = GameResult::DRAW;
          adjudicated_ = true;
          break;
        }
      } else {
        if (eval < resignpct) {  // always false when resignpct == 0
          game_result_ =
              blacks_move ? GameResult::WHITE_WON : GameResult::BLACK_WON;
          adjudicated_ = true;
          break;
        }
      }
    }

    auto node = tree_[idx]->GetCurrentHead();
    classic::Eval played_eval = best_eval;
    Move move;
    while (true) {
      move = search_->GetBestMove().first;
      uint32_t max_n = 0;
      uint32_t cur_n = 0;

      for (auto& edge : node->Edges()) {
        if (edge.GetN() > max_n) {
          max_n = edge.GetN();
        }
        if (edge.GetMove(tree_[idx]->IsBlackToMove()) == move) {
          cur_n = edge.GetN();
          played_eval.wl = edge.GetWL(-node->GetWL());
          played_eval.d = edge.GetD(node->GetD());
          played_eval.ml = edge.GetM(node->GetM() - 1) + 1;
        }
      }
      // If 'best move' is less than allowed visits and not max visits,
      // discard it and try again.
      if (cur_n == max_n ||
          static_cast<int>(cur_n) >=
              options_[idx].uci_options->Get<int>(kMinimumAllowedVistsId)) {
        break;
      }
      PositionHistory history_copy = tree_[idx]->GetPositionHistory();
      Move move_for_history = move;
      if (tree_[idx]->IsBlackToMove()) move_for_history.Flip();
      history_copy.Append(move_for_history);
      // Ensure not to discard games that are already decided.
      if (history_copy.ComputeGameResult() == GameResult::UNDECIDED) {
        auto move_list_to_discard = GetMoves();
        move_list_to_discard.push_back(move);
        options_[idx].discarded_callback({orig_fen_, move_list_to_discard});
      }
      search_->ResetBestMove();
    }

    if (training) {
      bool best_is_proof = best_is_terminal;  // But check for better moves.
      if (best_is_proof && best_eval.wl < 1) {
        auto best =
            (best_eval.wl == 0) ? GameResult::DRAW : GameResult::BLACK_WON;
        auto upper = best;
        for (const auto& edge : node->Edges()) {
          upper = std::max(edge.GetBounds().second, upper);
        }
        if (best < upper) {
          best_is_proof = false;
        }
      }
      // Append training data. The GameResult is later overwritten.
      std::vector<Move> legal_moves = tree_[idx]
                                          ->GetPositionHistory()
                                          .Last()
                                          .GetBoard()
                                          .GenerateLegalMoves();
      std::optional<EvalResult> nneval =
          options_[idx].backend->GetCachedEvaluation(EvalPosition{
              tree_[idx]->GetPositionHistory().GetPositions(), legal_moves});
      // Policy training target dispatch.  Three mutually-exclusive paths
      // (all enforced as conflicting at SearchParams ctor):
      //   1. UseGrillImprovedTarget=true  → Grill et al. (ICML 2020)
      //      regularized target π̄(a) = λ_N · prior / (α − q).
      //   2. UseGumbelImprovedTarget=true → softmax(prior + σ(q)) target
      //      (Gumbel-MuZero improved-policy, Phase 1.6).
      //   3. Default → GetTrainingTargetVisits (raw-N or PTP-clamped
      //      visit-count distribution, depending on
      //      forced-exploration-factor / advisor / UsePolicyTargetPruning).
      // All three keep PUCT search unchanged; only the chunk-write step
      // differs.  Mutual exclusivity is checked at param-construction
      // time so reaching this branch with multiple set would be a bug.
      const auto& sp = search_->GetParams();
      const std::vector<float> processed_visits =
          sp.GetUseGrillImprovedTarget()
              ? search_->GetGrillImprovedPolicyTarget()
              : sp.GetUseGumbelImprovedTarget()
                    ? search_->GetGumbelImprovedPolicyTarget()
                    : search_->GetTrainingTargetVisits();
      training_data_.Add(tree_[idx]->GetCurrentHead(),
                         tree_[idx]->GetPositionHistory(), best_eval,
                         played_eval, best_is_proof, best_move, move,
                         legal_moves, nneval,
                         search_->GetParams().GetPolicySoftmaxTemp(),
                         &processed_visits);
    }
    // Must reset the search before mutating the tree.
    search_.reset();

    // Add best move to the tree.
    if (tree_[0]->IsBlackToMove()) move.Flip();
    tree_[0]->MakeMove(move);
    if (tree_[0] != tree_[1]) tree_[1]->MakeMove(move);
    blacks_move = !blacks_move;
  }
}

void SelfPlayGame::PlayPerSide(int white_threads, int black_threads,
                              bool training, SyzygyTablebase* syzygy_tb,
                              bool enable_resign) {
  // ── Scope guards ──
  // External opponent/advisor engines aren't wired for the dag tree yet, so
  // refuse rather than silently produce wrong/no data.
  if (!options_[0].external_engine_path.empty() ||
      !options_[1].external_engine_path.empty() ||
      !options_[0].advisor_engine_path.empty() ||
      !options_[1].advisor_engine_path.empty()) {
    throw Exception(
        "dag-preview selfplay does not support external opponent/advisor "
        "engines yet. Use --search-algorithm=classic for those.");
  }
  // PlayPerSide writes the raw root visit-count distribution as the policy
  // target for BOTH branches (see capture_training below) — it does not port
  // classic's improved-policy target reshaping (PTP / forced-exploration /
  // Grill / Gumbel) to this path.  The dag search has no such reshaping at
  // all, and replicating it for the classic branch here would silently
  // diverge from the real Play() path.  So refuse those on EITHER side when a
  // dag side is present, rather than write a mistargeted policy.  (Advisor
  // forcing is already refused above; both-classic games never reach here and
  // keep full target support via Play().)
  if (training) {
    for (int s = 0; s < 2; ++s) {
      const classic::SearchParams sp(*options_[s].uci_options);
      if (sp.GetForcedExplorationFactor() > 0.0f ||
          sp.GetUsePolicyTargetPruning() || sp.GetUseGrillImprovedTarget() ||
          sp.GetUseGumbelImprovedTarget()) {
        throw Exception(
            "dag-preview training writes raw visit-count policy targets; "
            "improved-policy targets (--forced-exploration-factor, "
            "--policy-target-pruning, Grill, Gumbel) are not supported when a "
            "dag side is present (they would mistarget the policy). Disable "
            "them, or run both sides with --search-algorithm=classic.");
      }
    }
  }

  // Helpers operating across the two (possibly different-typed) trees, which
  // both track the same game.  Side 0 is the reference for position queries.
  auto side0_black = [&]() {
    return side_uses_dag_[0] ? dag_tree_[0]->IsBlackToMove()
                             : tree_[0]->IsBlackToMove();
  };
  auto ref_history = [&]() -> const PositionHistory& {
    return side_uses_dag_[0] ? dag_tree_[0]->GetPositionHistory()
                             : tree_[0]->GetPositionHistory();
  };
  auto make_move_all = [&](Move internal) {
    if (side_uses_dag_[0]) {
      dag_tree_[0]->MakeMove(internal);
    } else {
      tree_[0]->MakeMove(internal);
    }
    if (separate_trees_) {
      if (side_uses_dag_[1]) {
        dag_tree_[1]->MakeMove(internal);
      } else {
        tree_[1]->MakeMove(internal);
      }
    }
  };

  bool blacks_move = side0_black();

  // Syzygy tablebases from player1 options (mirrors Play()).
  std::string tb_paths =
      options_[0].uci_options->Get<std::string>(kSyzygyTablebaseId);
  if (!tb_paths.empty()) {
    syzygy_tb_ = std::make_unique<SyzygyTablebase>();
    CERR << "Loading Syzygy tablebases from " << tb_paths;
    if (!syzygy_tb_->init(tb_paths)) {
      CERR << "Failed to load Syzygy tablebases!";
      syzygy_tb_ = nullptr;
    }
  }

  // Shared training-data capture, generic over the tree/search type so the
  // dag and classic branches call it identically; the templated
  // V7TrainingDataArray::Add resolves on the concrete node type.  Only called
  // when `training`.  The policy target is the raw per-root-edge visit count
  // (classic's GetTrainingTargetVisits fast path); passing it as
  // `processed_visits` makes Add normalize by THEIR sum, which is exact and
  // self-consistent even for the dag tree (root moves never transpose to each
  // other, so each root edge's N is clean per-move visits).
  auto capture_training = [&](int tr_idx, auto& tree_ref, auto& search_ref,
                              Move best_move, bool best_is_terminal,
                              classic::Eval best_eval,
                              classic::Eval played_eval, Move played_move) {
    auto* head = tree_ref.GetCurrentHead();
    // Proof flag: best is proven only if no sibling has a strictly better
    // proven upper bound (mirrors the classic Play() training block).
    bool best_is_proof = best_is_terminal;
    if (best_is_proof && best_eval.wl < 1) {
      auto best =
          (best_eval.wl == 0) ? GameResult::DRAW : GameResult::BLACK_WON;
      auto upper = best;
      for (const auto& edge : head->Edges()) {
        upper = std::max(edge.GetBounds().second, upper);
      }
      if (best < upper) best_is_proof = false;
    }
    std::vector<Move> legal_moves =
        tree_ref.GetPositionHistory().Last().GetBoard().GenerateLegalMoves();
    std::optional<EvalResult> nneval =
        options_[tr_idx].backend->GetCachedEvaluation(EvalPosition{
            tree_ref.GetPositionHistory().GetPositions(), legal_moves});
    std::vector<float> processed_visits;
    processed_visits.reserve(head->GetNumEdges());
    for (const auto& edge : head->Edges()) {
      processed_visits.push_back(static_cast<float>(edge.GetN()));
    }
    training_data_.Add(head, tree_ref.GetPositionHistory(), best_eval,
                       played_eval, best_is_proof, best_move, played_move,
                       legal_moves, nneval,
                       search_ref.GetParams().GetPolicySoftmaxTemp(),
                       &processed_visits);
  };

  while (!abort_) {
    game_result_ = ref_history().ComputeGameResult();
    if (game_result_ != GameResult::UNDECIDED) break;
    if (ref_history().Last().GetGamePly() >= 450) {
      adjudicated_ = true;
      break;
    }
    const int idx = blacks_move ? 1 : 0;
    const int threads = blacks_move ? black_threads : white_threads;

    // Trim the to-move side's tree unless reuse-tree is set (mirrors the
    // classic Play() path so dag and classic sides reuse/trim identically).
    if (!options_[idx].uci_options->Get<bool>(kReuseTreeId)) {
      if (side_uses_dag_[idx]) {
        dag_tree_[idx]->TrimTreeAtHead();
      } else {
        tree_[idx]->TrimTreeAtHead();
      }
    }

    // The played move (board frame) plus the data needed for the resign
    // decision and (when `training`) the training chunk, set by whichever
    // engine this side uses.  best_eval/played_eval are classic::Eval {wl,d,ml}
    // even on a dag side — dag_classic::Eval has the same fields, copied over.
    Move move;
    Move best_move;
    bool best_is_terminal = false;
    classic::Eval best_eval{};    // root eval, no temperature.
    classic::Eval played_eval{};  // == best_eval until the played edge found.

    if (side_uses_dag_[idx]) {
      // ── dag-preview side ──
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (abort_) break;
        auto stoppers = options_[idx].search_limits.MakeSearchStopper();
        classic::PopulateIntrinsicStoppers(stoppers.get(),
                                           *options_[idx].uci_options);
        std::unique_ptr<UciResponder> responder =
            std::make_unique<CallbackUciResponder>(
                options_[idx].best_move_callback, options_[idx].info_callback);
        dag_classic::TranspositionTable* tt =
            &dag_tt_[separate_trees_ ? idx : 0];
        dag_search_ = std::make_unique<dag_classic::Search>(
            *dag_tree_[idx], options_[idx].backend, std::move(responder),
            /* searchmoves */ MoveList(), std::chrono::steady_clock::now(),
            std::move(stoppers), /* infinite */ false, /* ponder */ false,
            *options_[idx].uci_options, tt, syzygy_tb);
      }
      dag_search_->RunBlocking(threads);
      move_count_++;
      nodes_total_ += dag_search_->GetTotalPlayouts();
      if (abort_) break;
      const auto raw_eval =
          dag_search_->GetBestEval(&best_move, &best_is_terminal);
      best_eval.wl = raw_eval.wl;
      best_eval.d = raw_eval.d;
      best_eval.ml = raw_eval.ml;
      played_eval = best_eval;
      auto node = dag_tree_[idx]->GetCurrentHead();
      while (true) {
        move = dag_search_->GetBestMove().first;
        uint32_t max_n = 0;
        uint32_t cur_n = 0;
        for (auto& edge : node->Edges()) {
          if (edge.GetN() > max_n) max_n = edge.GetN();
          if (edge.GetMove(dag_tree_[idx]->IsBlackToMove()) == move) {
            cur_n = edge.GetN();
            played_eval.wl = edge.GetWL(-node->GetWL());
            played_eval.d = edge.GetD(node->GetD());
            played_eval.ml = edge.GetM(node->GetM() - 1) + 1;
          }
        }
        if (cur_n == max_n ||
            static_cast<int>(cur_n) >=
                options_[idx].uci_options->Get<int>(kMinimumAllowedVistsId)) {
          break;
        }
        PositionHistory hc = dag_tree_[idx]->GetPositionHistory();
        Move mh = move;
        if (dag_tree_[idx]->IsBlackToMove()) mh.Flip();
        hc.Append(mh);
        if (hc.ComputeGameResult() == GameResult::UNDECIDED) {
          auto discard = GetMoves();
          discard.push_back(move);
          options_[idx].discarded_callback({orig_fen_, discard});
        }
        dag_search_->ResetBestMove();
      }
      if (training) {
        capture_training(idx, *dag_tree_[idx], *dag_search_, best_move,
                         best_is_terminal, best_eval, played_eval, move);
      }
      dag_search_.reset();
    } else {
      // ── classic side ──
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (abort_) break;
        auto stoppers = options_[idx].search_limits.MakeSearchStopper();
        classic::PopulateIntrinsicStoppers(stoppers.get(),
                                           *options_[idx].uci_options);
        std::unique_ptr<UciResponder> responder =
            std::make_unique<CallbackUciResponder>(
                options_[idx].best_move_callback, options_[idx].info_callback);
        search_ = std::make_unique<classic::Search>(
            *tree_[idx], options_[idx].backend, std::move(responder),
            /* searchmoves */ MoveList(), std::chrono::steady_clock::now(),
            std::move(stoppers), /* infinite */ false, /* ponder */ false,
            *options_[idx].uci_options, syzygy_tb);
      }
      search_->RunBlocking(threads);
      move_count_++;
      nodes_total_ += search_->GetTotalPlayouts();
      if (abort_) break;
      const auto raw_eval = search_->GetBestEval(&best_move, &best_is_terminal);
      best_eval.wl = raw_eval.wl;
      best_eval.d = raw_eval.d;
      best_eval.ml = raw_eval.ml;
      played_eval = best_eval;
      auto node = tree_[idx]->GetCurrentHead();
      while (true) {
        move = search_->GetBestMove().first;
        uint32_t max_n = 0;
        uint32_t cur_n = 0;
        for (auto& edge : node->Edges()) {
          if (edge.GetN() > max_n) max_n = edge.GetN();
          if (edge.GetMove(tree_[idx]->IsBlackToMove()) == move) {
            cur_n = edge.GetN();
            played_eval.wl = edge.GetWL(-node->GetWL());
            played_eval.d = edge.GetD(node->GetD());
            played_eval.ml = edge.GetM(node->GetM() - 1) + 1;
          }
        }
        if (cur_n == max_n ||
            static_cast<int>(cur_n) >=
                options_[idx].uci_options->Get<int>(kMinimumAllowedVistsId)) {
          break;
        }
        PositionHistory hc = tree_[idx]->GetPositionHistory();
        Move mh = move;
        if (tree_[idx]->IsBlackToMove()) mh.Flip();
        hc.Append(mh);
        if (hc.ComputeGameResult() == GameResult::UNDECIDED) {
          auto discard = GetMoves();
          discard.push_back(move);
          options_[idx].discarded_callback({orig_fen_, discard});
        }
        search_->ResetBestMove();
      }
      if (training) {
        capture_training(idx, *tree_[idx], *search_, best_move,
                         best_is_terminal, best_eval, played_eval, move);
      }
      search_.reset();
    }

    // ── shared post-search: eval tracking + resign + advance ──
    const float wl = best_eval.wl;
    const float d = best_eval.d;
    float eval = (wl + 1) / 2;
    if (eval < min_eval_[idx]) min_eval_[idx] = eval;
    const int move_number = ref_history().GetLength() / 2 + 1;
    auto best_w = (wl + 1.0f - d) / 2.0f;
    auto best_d = d;
    auto best_l = best_w - wl;
    max_eval_[0] = std::max(max_eval_[0], blacks_move ? best_l : best_w);
    max_eval_[1] = std::max(max_eval_[1], best_d);
    max_eval_[2] = std::max(max_eval_[2], blacks_move ? best_w : best_l);
    if (enable_resign && move_number >= options_[idx].uci_options->Get<int>(
                                            kResignEarliestMoveId)) {
      const float resignpct =
          options_[idx].uci_options->Get<float>(kResignPercentageId) / 100;
      if (options_[idx].uci_options->Get<bool>(kResignWDLStyleId)) {
        auto threshold = 1.0f - resignpct;
        if (best_w > threshold) {
          game_result_ =
              blacks_move ? GameResult::BLACK_WON : GameResult::WHITE_WON;
          adjudicated_ = true;
          break;
        }
        if (best_l > threshold) {
          game_result_ =
              blacks_move ? GameResult::WHITE_WON : GameResult::BLACK_WON;
          adjudicated_ = true;
          break;
        }
        if (best_d > threshold) {
          game_result_ = GameResult::DRAW;
          adjudicated_ = true;
          break;
        }
      } else {
        if (eval < resignpct) {
          game_result_ =
              blacks_move ? GameResult::WHITE_WON : GameResult::BLACK_WON;
          adjudicated_ = true;
          break;
        }
      }
    }

    // Advance every tree with the played move (board frame → internal).
    Move internal = move;
    if (side0_black()) internal.Flip();
    make_move_all(internal);
    blacks_move = !blacks_move;
  }
}

std::vector<Move> SelfPlayGame::GetMoves() const {
  if (side_uses_dag_[0]) {
    // A DAG node has no unique parent, so we can't walk head→begin like the
    // classic path.  dag_classic::NodeTree stores the played moves directly
    // (internal/flipped frame, oldest-first); replay them forward to recover
    // board-frame moves using the same un-flip rule as the classic path.
    std::vector<Move> result;
    Position pos = dag_tree_[0]->GetPositionHistory().Starting();
    for (Move move : dag_tree_[0]->GetMoves()) {
      pos = Position(pos, move);
      if (!pos.IsBlackToMove()) move.Flip();
      result.push_back(move);
    }
    return result;
  }
  std::vector<Move> moves;
  for (classic::Node* node = tree_[0]->GetCurrentHead();
       node != tree_[0]->GetGameBeginNode(); node = node->GetParent()) {
    moves.push_back(node->GetParent()->GetEdgeToNode(node)->GetMove());
  }
  std::vector<Move> result;
  Position pos = tree_[0]->GetPositionHistory().Starting();
  while (!moves.empty()) {
    Move move = moves.back();
    moves.pop_back();
    pos = Position(pos, move);
    // Position already flipped, therefore flip the move if white to move.
    if (!pos.IsBlackToMove()) move.Flip();
    result.push_back(move);
  }
  return result;
}

float SelfPlayGame::GetWorstEvalForWinnerOrDraw() const {
  // TODO: This assumes both players have the same resign style.
  // Supporting otherwise involves mixing the meaning of worst.
  if (options_[0].uci_options->Get<bool>(kResignWDLStyleId)) {
    if (game_result_ == GameResult::WHITE_WON) {
      return std::max(max_eval_[1], max_eval_[2]);
    } else if (game_result_ == GameResult::BLACK_WON) {
      return std::max(max_eval_[1], max_eval_[0]);
    } else {
      return std::max(max_eval_[2], max_eval_[0]);
    }
  }
  if (game_result_ == GameResult::WHITE_WON) return min_eval_[0];
  if (game_result_ == GameResult::BLACK_WON) return min_eval_[1];
  return std::min(min_eval_[0], min_eval_[1]);
}

void SelfPlayGame::Abort() {
  std::lock_guard<std::mutex> lock(mutex_);
  abort_ = true;
  if (search_) search_->Abort();
  if (dag_search_) dag_search_->Abort();
}

void SelfPlayGame::WriteTrainingData(TrainingDataWriter* writer) const {
  training_data_.Write(writer, game_result_, adjudicated_);
}

std::unique_ptr<classic::ChainedSearchStopper>
SelfPlayLimits::MakeSearchStopper() const {
  auto result = std::make_unique<classic::ChainedSearchStopper>();

  // always set VisitsStopper to avoid exceeding the limit 4000000000, the
  // default value when visits = 0
  result->AddStopper(std::make_unique<classic::VisitsStopper>(visits, false));
  if (playouts >= 0) {
    result->AddStopper(
        std::make_unique<classic::PlayoutsStopper>(playouts, false));
  }
  if (movetime >= 0) {
    result->AddStopper(std::make_unique<classic::TimeLimitStopper>(movetime));
  }
  return result;
}

}  // namespace lczero
