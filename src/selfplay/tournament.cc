/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018 The LCZero Authors

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

#include "selfplay/tournament.h"

#include <fstream>
#include <sstream>

#include "chess/pgn.h"
#include "neural/memcache.h"
#include "neural/shared_params.h"
#include "search/classic/search.h"
#include "search/classic/stoppers/factory.h"
#include "selfplay/game.h"
#include "selfplay/multigame.h"
#include "utils/optionsparser.h"
#include "utils/random.h"

namespace lczero {
namespace {
const OptionId kShareTreesId{"share-trees", "ShareTrees",
                             "When on, game tree is shared for two players; "
                             "when off, each side has a separate tree."};
const OptionId kTotalGamesId{
    "games", "Games",
    "Number of games to play. -1 to play forever, -2 to play equal to book "
    "length, or double book length if mirrored."};
const OptionId kParallelGamesId{"parallelism", "Parallelism",
                                "Number of games to play in parallel."};
const OptionId kThreadsId{
    "threads", "Threads",
    "Number of (CPU) worker threads to use for every game,", 't'};
const OptionId kPlayoutsId{"playouts", "Playouts",
                           "Number of playouts per move to search."};
const OptionId kVisitsId{"visits", "Visits",
                         "Number of visits per move to search."};
const OptionId kTimeMsId{"movetime", "MoveTime",
                         "Time per move, in milliseconds."};
const OptionId kTrainingId{
    "training", "Training",
    "Enables writing training data. The training data is stored into a "
    "temporary subdirectory that the engine creates."};
const OptionId kVerboseThinkingId{"verbose-thinking", "VerboseThinking",
                                  "Show verbose thinking messages."};
const OptionId kPolicyModeSizeId{"policy-mode-size", "PolicyModeSize",
                                 "Number of games per thread in policy only "
                                 "mode. Set to 0 to not use policy only mode."};
const OptionId kValueModeSizeId{"value-mode-size", "ValueModeSize",
                                "Number of games per thread in value only "
                                "mode. Set to 0 to not use value only mode."};
const OptionId kTournamentResultsFileId{
    "tournament-results-file", "TournamentResultsFile",
    "Name of file to append the tournament results in fake pgn format."};
const OptionId kMoveThinkingId{"move-thinking", "MoveThinking",
                               "Show all the per-move thinking."};
const OptionId kResignPlaythroughId{
    "resign-playthrough", "ResignPlaythrough",
    "The percentage of games which ignore resign."};
const OptionId kDiscardedStartChanceId{
    "discarded-start-chance", "DiscardedStartChance",
    "The percentage chance each game will attempt to start from a position "
    "discarded due to not getting enough visits."};
const OptionId kOpeningsFileId{
    "openings-pgn", "OpeningsPgnFile",
    "A path name to a pgn file containing openings to use."};
const OptionId kOpeningsMirroredId{
    "mirror-openings", "MirrorOpenings",
    "If true, each opening will be played in pairs. "
    "Not really compatible with openings mode random."};
const OptionId kOpeningsModeId{"openings-mode", "OpeningsMode",
                               "A choice of sequential, shuffled, or random."};
const OptionId kSyzygyTablebaseId{
    "syzygy-paths", "SyzygyPath",
    "List of Syzygy tablebase directories, list entries separated by system "
    "separator (\";\" for Windows, \":\" for Linux).",
    's'};

// ─── External UCI opponent flags ───
const OptionId kOpponentPathId{
    "opponent-path", "OpponentPath",
    "Path to an external UCI engine (e.g. /usr/games/stockfish). When set, "
    "the engine plays the side(s) selected by --opponent-side; training "
    "data is only written for lc0's moves."};
const OptionId kOpponentArgsId{
    "opponent-args", "OpponentArgs",
    "Whitespace-separated command-line arguments passed to the opponent "
    "binary on spawn."};
const OptionId kOpponentUciOptionsId{
    "opponent-options", "OpponentOptions",
    "Semicolon-separated UCI setoption pairs sent to the opponent at "
    "startup, e.g. 'Threads=1;Hash=64;Skill Level=15;UCI_LimitStrength=true;"
    "UCI_Elo=2400'. Order is preserved."};
const OptionId kOpponentGoCommandId{
    "opponent-go", "OpponentGo",
    "Argument string appended after 'go' on every move, e.g. 'movetime 50', "
    "'depth 12', 'nodes 100000'.",
};

// ─── External UCI advisor flags ───
// The advisor is queried at every lc0-side move; its recommended move
// receives forced MCTS visits at root before PUCT selection resumes
// normal behavior.  Trained policy still comes from lc0's MCTS visit
// distribution — advisor only biases which positions get explored.
const OptionId kAdvisorPathId{
    "advisor-path", "AdvisorPath",
    "Path to an external UCI engine to use as an MCTS advisor (e.g. "
    "Stockfish).  When set, lc0 consults this engine at every lc0-side "
    "move and forces --advisor-min-visits visits on the advisor's move "
    "at root.  Empty disables advisor mode."};
const OptionId kAdvisorArgsId{
    "advisor-args", "AdvisorArgs",
    "Whitespace-separated command-line arguments passed to the advisor "
    "binary on spawn."};
const OptionId kAdvisorUciOptionsId{
    "advisor-options", "AdvisorOptions",
    "Semicolon-separated UCI setoption pairs sent to the advisor at "
    "startup, e.g. 'Threads=2;Hash=256'. Order is preserved."};
const OptionId kAdvisorGoCommandId{
    "advisor-go", "AdvisorGo",
    "Argument string appended after 'go' on every advisor query, e.g. "
    "'movetime 100', 'nodes 100000'.  Lower = faster selfplay throughput; "
    "higher = stronger advisor recommendations."};
const OptionId kAdvisorForcePlayId{
    "advisor-force-play-prob", "AdvisorForcePlayProb",
    "Probability of PLAYING the advisor's move (instead of the temperature-"
    "sampled move) at an lc0-side turn where the advisor was consulted.  0 = "
    "never force-play (the advisor only forces search visits — the conservative "
    "default).  >0 injects the advisor's move into the game so the net learns "
    "its outcome via the value head; the policy target is unaffected (stays "
    "PTP-clean).  Gated on temp-cutoff-move (not applied in the greedy endgame)."};
const OptionId kAdvisorForcePlayMatesId{
    "advisor-force-play-mates", "AdvisorForcePlayMates",
    "When on, always force-play (and play out, suppressing resign) a move the "
    "advisor reports as a forced mate for the side to move — regardless of "
    "advisor-force-play-prob and the temperature cutoff.  A reported mate is "
    "guaranteed correct, so this injects deep mating lines the low-visit search "
    "can't see.  Default on; only takes effect when the advisor is configured."};
const OptionId kAdvisorForcePlayDisagreeId{
    "advisor-force-play-on-disagree", "AdvisorForcePlayOnDisagree",
    "When on, probabilistic advisor force-play only fires if the advisor's move "
    "differs from the net's own best move (inject only on genuine disagreement). "
    "Mates ignore this gate.  Default off (force-play purely on probability)."};
const OptionId kAdvisorMinVisitsId{
    "advisor-min-visits", "AdvisorMinVisits",
    "Number of root MCTS visits to force on the advisor's move at every "
    "lc0-side turn.  0 disables advisor injection (the advisor is "
    "consulted but its move is ignored, which makes no sense — set this "
    "to a positive value to actually use advisor mode).  Typical: 20-50 "
    "out of --visits=800.  Must be < --visits."};
const OptionId kOpponentSideId{
    "opponent-side", "OpponentSide",
    "Which colour(s) the opponent plays: white, black, alternate, or none. "
    "'alternate' switches sides every game so lc0 sees both colours. "
    "'none' disables the opponent (pure selfplay)."};

}  // namespace

void SelfPlayTournament::PopulateOptions(OptionsParser* options) {
  options->AddContext("player1");
  options->AddContext("player2");
  options->AddContext("white");
  options->AddContext("black");
  for (const auto context : {"player1", "player2"}) {
    auto* dict = options->GetMutableOptions(context);
    dict->AddSubdict("white")->AddAliasDict(&options->GetOptionsDict("white"));
    dict->AddSubdict("black")->AddAliasDict(&options->GetOptionsDict("black"));
  }

  SharedBackendParams::Populate(options);
  options->Add<IntOption>(kThreadsId, 1, 8) = 1;
  classic::SearchParams::Populate(options);

  options->Add<BoolOption>(kShareTreesId) = true;
  options->Add<IntOption>(kTotalGamesId, -2, 999999) = -1;
  options->Add<IntOption>(kParallelGamesId, 1, 256) = 8;
  options->Add<IntOption>(kPlayoutsId, -1, 999999999) = -1;
  options->Add<IntOption>(kVisitsId, -1, 999999999) = -1;
  options->Add<IntOption>(kTimeMsId, -1, 999999999) = -1;
  options->Add<BoolOption>(kTrainingId) = false;
  options->Add<BoolOption>(kVerboseThinkingId) = false;
  options->Add<IntOption>(kPolicyModeSizeId, 0, 1024) = 0;
  options->Add<IntOption>(kValueModeSizeId, 0, 64) = 0;
  options->Add<StringOption>(kTournamentResultsFileId) = "";
  options->Add<BoolOption>(kMoveThinkingId) = false;
  options->Add<FloatOption>(kResignPlaythroughId, 0.0f, 100.0f) = 0.0f;
  options->Add<FloatOption>(kDiscardedStartChanceId, 0.0f, 100.0f) = 0.0f;
  options->Add<StringOption>(kOpeningsFileId) = "";
  options->Add<BoolOption>(kOpeningsMirroredId) = false;
  std::vector<std::string> openings_modes = {"sequential", "shuffled",
                                             "random"};
  options->Add<ChoiceOption>(kOpeningsModeId, openings_modes) = "sequential";

  options->Add<StringOption>(kSyzygyTablebaseId);

  // External opponent. Default empty -> pure lc0 selfplay.
  options->Add<StringOption>(kOpponentPathId) = "";
  options->Add<StringOption>(kOpponentArgsId) = "";
  options->Add<StringOption>(kOpponentUciOptionsId) = "";
  options->Add<StringOption>(kOpponentGoCommandId) = "movetime 100";
  options->Add<StringOption>(kAdvisorPathId) = "";
  options->Add<StringOption>(kAdvisorArgsId) = "";
  options->Add<StringOption>(kAdvisorUciOptionsId) = "";
  options->Add<StringOption>(kAdvisorGoCommandId) = "movetime 100";
  options->Add<IntOption>(kAdvisorMinVisitsId, 0, 100000) = 0;
  options->Add<FloatOption>(kAdvisorForcePlayId, 0.0f, 1.0f) = 0.0f;
  options->Add<BoolOption>(kAdvisorForcePlayMatesId) = true;
  options->Add<BoolOption>(kAdvisorForcePlayDisagreeId) = false;
  std::vector<std::string> opponent_sides = {"none", "white", "black",
                                              "alternate"};
  options->Add<ChoiceOption>(kOpponentSideId, opponent_sides) = "none";

  SelfPlayGame::PopulateUciParams(options);

  auto defaults = options->GetMutableDefaultsOptions();
  defaults->Set<int>(classic::SearchParams::kMiniBatchSizeId, 32);
  defaults->Set<float>(classic::SearchParams::kCpuctId, 1.2f);
  defaults->Set<float>(classic::SearchParams::kCpuctFactorId, 0.0f);
  defaults->Set<float>(SharedBackendParams::kPolicySoftmaxTemp, 1.0f);
  defaults->Set<int>(classic::SearchParams::kMaxCollisionVisitsId, 1);
  defaults->Set<int>(classic::SearchParams::kMaxCollisionEventsId, 1);
  defaults->Set<int>(classic::SearchParams::kCacheHistoryLengthId, 7);
  defaults->Set<bool>(classic::SearchParams::kOutOfOrderEvalId, false);
  defaults->Set<float>(classic::SearchParams::kTemperatureId, 1.0f);
  defaults->Set<float>(classic::SearchParams::kNoiseEpsilonId, 0.25f);
  defaults->Set<float>(classic::SearchParams::kFpuValueId, 0.0f);
  defaults->Set<std::string>(SharedBackendParams::kHistoryFill, "no");
  defaults->Set<std::string>(SharedBackendParams::kBackendId, "multiplexing");
  defaults->Set<bool>(classic::SearchParams::kStickyEndgamesId, false);
  defaults->Set<bool>(classic::SearchParams::kTwoFoldDrawsId, false);
  defaults->Set<int>(classic::SearchParams::kTaskWorkersPerSearchWorkerId, 0);
}

SelfPlayTournament::SelfPlayTournament(const OptionsDict& options,
                                       UciResponder* uci_responder,
                                       GameInfo::Callback game_info,
                                       TournamentInfo::Callback tournament_info)
    : player_options_{{options.GetSubdict("player1").GetSubdict("white"),
                       options.GetSubdict("player1").GetSubdict("black")},
                      {options.GetSubdict("player2").GetSubdict("white"),
                       options.GetSubdict("player2").GetSubdict("black")}},
      uci_responder_(uci_responder),
      game_callback_(game_info),
      tournament_callback_(tournament_info),
      kTotalGames(options.Get<int>(kTotalGamesId)),
      kShareTree(options.Get<bool>(kShareTreesId)),
      kParallelism(options.Get<int>(kParallelGamesId)),
      kTraining(options.Get<bool>(kTrainingId)),
      kResignPlaythrough(options.Get<float>(kResignPlaythroughId)),
      kPolicyGamesSize(options.Get<int>(kPolicyModeSizeId)),
      kValueGamesSize(options.Get<int>(kValueModeSizeId)),
      kTournamentResultsFile(
          options.Get<std::string>(kTournamentResultsFileId)),
      kDiscardedStartChance(options.Get<float>(kDiscardedStartChanceId)) {
  multi_games_size_ = std::max(kPolicyGamesSize, kValueGamesSize);
  std::string book = options.Get<std::string>(kOpeningsFileId);
  if (!book.empty()) {
    PgnReader book_reader;
    book_reader.AddPgnFile(book);
    openings_ = book_reader.ReleaseGames();
    if (options.Get<std::string>(kOpeningsModeId) == "shuffled") {
      Random::Get().Shuffle(openings_.begin(), openings_.end());
    }
  }
  if (kPolicyGamesSize > 0 && kValueGamesSize > 0) {
    throw Exception("Can't do both policy and value games at the same time.");
  }
  if (multi_games_size_ > 0 && openings_.size() == 0) {
    throw Exception(
        "Policy/Value games are deterministic, needs opening book to be "
        "useful.");
  }
  if (multi_games_size_ > 0 &&
      (kTotalGames == -1 ||
       (kTotalGames > 0 &&
        static_cast<size_t>(kTotalGames) > openings_.size() * 2))) {
    throw Exception(
        "Policy/Value games are deterministic, you do not want to go through "
        "the "
        "opening book more than once.");
  }
  // The batched value/policy-mode games run through MultiSelfPlayGames, which
  // is hard-wired to the classic search and ignores --search-algorithm.  Refuse
  // dag-preview here rather than silently run classic and invalidate the run.
  if (multi_games_size_ > 0) {
    for (int pl = 0; pl < 2; ++pl) {
      for (int color = 0; color < 2; ++color) {
        if (SelfPlayGame::IsDagRequested(player_options_[pl][color])) {
          throw Exception(
              "--search-algorithm=dag-preview is not supported with "
              "--value-mode-size / --policy-mode-size (those run the classic "
              "batched value/policy path). Use normal selfplay games (drop "
              "value/policy-mode-size) to exercise dag-preview.");
        }
      }
    }
  }
  // dag-preview training supports raw-N policy targets AND policy-target-pruning
  // (dag_classic::Search::GetTrainingTargetVisits applies the KataGo equilibrium
  // clamp, which is what prunes an advisor's forced root visits).  It does NOT
  // implement the forced-exploration *factor* in the search itself, nor the
  // Grill/Gumbel improved-policy targets, so refuse those on either player when
  // a dag side is present rather than silently mistarget the policy.  (Advisor
  // forcing and --policy-target-pruning are fine.)  Checked here (startup, main
  // thread) so a misconfig errors cleanly instead of throwing from a worker
  // thread mid-run.
  if (kTraining) {
    bool any_dag = false;
    for (int pl = 0; pl < 2; ++pl) {
      for (int color = 0; color < 2; ++color) {
        if (SelfPlayGame::IsDagRequested(player_options_[pl][color])) {
          any_dag = true;
        }
      }
    }
    if (any_dag) {
      for (int pl = 0; pl < 2; ++pl) {
        for (int color = 0; color < 2; ++color) {
          const classic::SearchParams sp(player_options_[pl][color]);
          if (sp.GetForcedExplorationFactor() > 0.0f ||
              sp.GetUseGrillImprovedTarget() ||
              sp.GetUseGumbelImprovedTarget()) {
            throw Exception(
                "dag-preview training does not support --forced-exploration-"
                "factor (the dag search does not apply it) or the Grill/Gumbel "
                "improved-policy targets. Disable them, or run both sides with "
                "--search-algorithm=classic. (--policy-target-pruning and the "
                "advisor ARE supported.)");
          }
        }
      }
    }
  }
  // If playing just one game, the player1 is white, otherwise randomize.
  if (kTotalGames != 1) {
    first_game_black_ = Random::Get().GetBool();
  }

  static constexpr const char* kPlayerNames[2] = {"player1", "player2"};
  static constexpr const char* kPlayerColors[2] = {"white", "black"};

  // Initializing networks.
  std::vector<std::shared_ptr<Backend>> backend_list;
  for (int name_idx : {0, 1}) {
    for (int color_idx : {0, 1}) {
      const auto& name = kPlayerNames[name_idx];
      const auto& color = kPlayerColors[color_idx];
      const auto& opts = options.GetSubdict(name).GetSubdict(color);
      for (const auto& backend : backend_list) {
        if (backend->IsSameConfiguration(opts)) {
          backends_[name_idx][color_idx] = backend;
          break;
        }
      }
      if (!backends_[name_idx][color_idx]) {
        backends_[name_idx][color_idx] =
            CreateMemCache(BackendManager::Get()->CreateFromParams(opts),
                           options.GetSubdict(name));
        backend_list.emplace_back(backends_[name_idx][color_idx]);
      }
    }
  }

  // SearchLimits.
  for (int name_idx : {0, 1}) {
    for (int color_idx : {0, 1}) {
      auto& limits = search_limits_[name_idx][color_idx];
      const auto& dict = options.GetSubdict(kPlayerNames[name_idx])
                             .GetSubdict(kPlayerColors[color_idx]);
      limits.playouts = dict.Get<int>(kPlayoutsId);
      limits.visits = dict.Get<int>(kVisitsId);
      limits.movetime = dict.Get<int>(kTimeMsId);

      if (multi_games_size_ == 0 && limits.playouts == -1 &&
          limits.visits == -1 && limits.movetime == -1) {
        throw Exception(
            "Please define --visits, --playouts or --movetime, otherwise it's "
            "not clear when to stop search.");
      } else if (multi_games_size_ > 0 &&
                 (limits.playouts != -1 || limits.visits != -1 ||
                  limits.movetime != -1)) {
        throw Exception(
            "Policy mode does not need --visits, --playouts or --movetime as "
            "it has fixed 1 node behaviour.");
      }
    }
  }

  // Take syzygy tablebases from options.
  std::string tb_paths = options.Get<std::string>(kSyzygyTablebaseId);
  if (!tb_paths.empty()) {
    syzygy_tb_ = std::make_unique<SyzygyTablebase>();
    CERR << "Loading Syzygy tablebases from " << tb_paths;
    if (!syzygy_tb_->init(tb_paths)) {
      CERR << "Failed to load Syzygy tablebases!";
      syzygy_tb_ = nullptr;
    }
  }
}

void SelfPlayTournament::PlayOneGame(int game_number) {
  bool player1_black;  // Whether player1 will player as black in this game.
  Opening opening;
  {
    Mutex::Lock lock(mutex_);
    player1_black = ((game_number % 2) == 1) != first_game_black_;
    if (!openings_.empty()) {
      if (player_options_[0][0].Get<bool>(kOpeningsMirroredId)) {
        opening = openings_[(game_number / 2) % openings_.size()];
      } else if (player_options_[0][0].Get<std::string>(kOpeningsModeId) ==
                 "random") {
        opening = openings_[Random::Get().GetInt(0, openings_.size() - 1)];
      } else {
        opening = openings_[game_number % openings_.size()];
      }
    }
    if (discard_pile_.size() > 0 &&
        Random::Get().GetFloat(100.0f) < kDiscardedStartChance) {
      const size_t idx = Random::Get().GetInt(0, discard_pile_.size() - 1);
      if (idx != discard_pile_.size() - 1) {
        std::swap(discard_pile_[idx], discard_pile_.back());
      }
      opening = discard_pile_.back();
      discard_pile_.pop_back();
    }
  }
  const int color_idx[2] = {player1_black ? 1 : 0, player1_black ? 0 : 1};

  PlayerOptions options[2];

  std::vector<ThinkingInfo> last_thinking_info;
  for (int pl_idx : {0, 1}) {
    const int color = color_idx[pl_idx];
    const bool verbose_thinking =
        player_options_[pl_idx][color].Get<bool>(kVerboseThinkingId);
    const bool move_thinking =
        player_options_[pl_idx][color].Get<bool>(kMoveThinkingId);
    // Populate per-player options.
    PlayerOptions& opt = options[color_idx[pl_idx]];
    opt.backend = backends_[pl_idx][color].get();
    opt.uci_options = &player_options_[pl_idx][color];
    opt.search_limits = search_limits_[pl_idx][color];

    // "bestmove" callback.
    opt.best_move_callback = [this, game_number, pl_idx, player1_black,
                              verbose_thinking, move_thinking,
                              &last_thinking_info](const BestMoveInfo& info) {
      if (!move_thinking) {
        last_thinking_info.clear();
        return;
      }
      // In non-verbose mode, output the last "info" message.
      if (!verbose_thinking && !last_thinking_info.empty()) {
        uci_responder_->OutputThinkingInfo(&last_thinking_info);
        last_thinking_info.clear();
      }
      BestMoveInfo rich_info = info;
      rich_info.player = pl_idx + 1;
      rich_info.is_black = player1_black ? pl_idx == 0 : pl_idx != 0;
      rich_info.game_id = game_number;
      uci_responder_->OutputBestMove(&rich_info);
    };

    opt.info_callback =
        [this, game_number, pl_idx, player1_black, verbose_thinking,
         &last_thinking_info](const std::vector<ThinkingInfo>& infos) {
          std::vector<ThinkingInfo> rich_info = infos;
          for (auto& info : rich_info) {
            info.player = pl_idx + 1;
            info.is_black = player1_black ? pl_idx == 0 : pl_idx != 0;
            info.game_id = game_number;
          }
          if (verbose_thinking) {
            uci_responder_->OutputThinkingInfo(&rich_info);
          } else {
            // In non-verbose mode, remember the last "info" messages.
            last_thinking_info = std::move(rich_info);
          }
        };
    opt.discarded_callback = [this](const Opening& moves) {
      // Only track discards if discard start chance is non-zero.
      if (kDiscardedStartChance == 0.0f) return;
      Mutex::Lock lock(mutex_);
      discard_pile_.push_back(moves);
      // 10k seems it should be enough to keep a good mix and avoid running out
      // of ram.
      if (discard_pile_.size() > 10000) {
        // Swap a random element to end and pop it to avoid growing.
        const size_t idx = Random::Get().GetInt(0, discard_pile_.size() - 1);
        if (idx != discard_pile_.size() - 1) {
          std::swap(discard_pile_[idx], discard_pile_.back());
        }
        discard_pile_.pop_back();
      }
    };

    // ─── External UCI opponent (Stockfish etc.) plumbing ───
    // We read all opponent settings from player1/white's option dict —
    // there's only one external opponent in a tournament. The opponent
    // replaces one or both lc0 sides per --opponent-side.
    const std::string opponent_path =
        player_options_[0][0].Get<std::string>(kOpponentPathId);
    if (!opponent_path.empty()) {
      const std::string opponent_side =
          player_options_[0][0].Get<std::string>(kOpponentSideId);
      // Which actual color (0=white, 1=black) does the opponent play in
      // this game?  -1 means neither (pure lc0 selfplay).
      int opponent_color = -1;
      if (opponent_side == "white") {
        opponent_color = 0;
      } else if (opponent_side == "black") {
        opponent_color = 1;
      } else if (opponent_side == "alternate") {
        // SF should sit in the player2 slot consistently, with colors
        // flipping per game via lc0's standard player1_black flag.
        // The naive `(game_number % 2 == 0) ? 1 : 0` assumed P1 starts
        // white, but tournament.cc:247 randomizes first_game_black_
        // at startup — so on roughly half the processes, the assumption
        // is INVERTED and SF ends up landing in the player1 slot, with
        // lc0 in player2.  P1 stats then track SF instead of lc0,
        // which is why the per-process tournamentstatus output was
        // bimodal (some processes showing "P1 dominates" = lc0 wins,
        // others showing "P2 dominates" = also lc0 wins but lc0 is
        // labeled P2 there).
        //
        // Correct: SF's color is the OPPOSITE of player1's color, so
        // SF always lands in the player2 slot regardless of the
        // first_game_black_ RNG.
        opponent_color = player1_black ? 0 : 1;
      }
      // The opponent is the "other" player slot — if pl_idx is player2,
      // and player2 is playing color `color`, and that color matches
      // opponent_color, then THIS side gets the external engine.  This
      // is independent of player1/player2 distinction; we just check
      // "does this slot's color == opponent_color?".
      if (color == opponent_color) {
        opt.external_engine_path = opponent_path;

        // args: whitespace-split the string.  Quoted args not supported
        // yet — Stockfish doesn't need them.  If you ever do, switch to
        // a real shell-style splitter.
        const std::string args_str =
            player_options_[0][0].Get<std::string>(kOpponentArgsId);
        std::istringstream args_iss(args_str);
        std::string tok;
        while (args_iss >> tok) opt.external_engine_args.push_back(tok);

        // UCI options: semicolon-separated Name=Value pairs.
        const std::string uci_str =
            player_options_[0][0].Get<std::string>(kOpponentUciOptionsId);
        std::istringstream uci_iss(uci_str);
        std::string entry;
        while (std::getline(uci_iss, entry, ';')) {
          // Trim leading/trailing spaces.
          size_t b = 0;
          while (b < entry.size() && entry[b] == ' ') ++b;
          size_t e = entry.size();
          while (e > b && entry[e - 1] == ' ') --e;
          entry = entry.substr(b, e - b);
          if (entry.empty()) continue;
          auto eq = entry.find('=');
          if (eq == std::string::npos) {
            throw Exception(
                "Bad opponent UCI option (missing '='): '" + entry + "'");
          }
          opt.external_engine_uci_options.emplace_back(
              entry.substr(0, eq), entry.substr(eq + 1));
        }

        opt.external_engine_go_command =
            player_options_[0][0].Get<std::string>(kOpponentGoCommandId);
      }
    }
  }

  // ─── Advisor engine plumbing ───
  // Advisor configuration is read PER PLAYER from each player's options
  // subdict.  OptionsDict's fallback semantics mean:
  //   --advisor-path=foo               → both player1 and player2 see it
  //   --player1.advisor-path=foo       → only player1
  //   --player2.advisor-path=foo       → only player2
  //   --player1.advisor-path=foo --player2.advisor-path=bar  → different
  //
  // The game loop in game.cc skips advisor injection automatically on
  // sides where an external opponent is playing (no MCTS happens there),
  // so configuring an advisor on a side that ends up being SF is a no-op.
  for (int pl_idx : {0, 1}) {
    const int color = color_idx[pl_idx];
    const auto& dict = player_options_[pl_idx][color];

    const std::string advisor_path = dict.Get<std::string>(kAdvisorPathId);
    const int advisor_min_visits = dict.Get<int>(kAdvisorMinVisitsId);
    if (advisor_path.empty() || advisor_min_visits <= 0) continue;

    PlayerOptions& opt = options[color];
    opt.advisor_engine_path = advisor_path;
    opt.advisor_min_visits = advisor_min_visits;
    opt.advisor_force_play_prob = dict.Get<float>(kAdvisorForcePlayId);
    opt.advisor_force_play_mates = dict.Get<bool>(kAdvisorForcePlayMatesId);
    opt.advisor_force_play_on_disagree =
        dict.Get<bool>(kAdvisorForcePlayDisagreeId);
    opt.advisor_engine_go_command =
        dict.Get<std::string>(kAdvisorGoCommandId);

    // args: whitespace-split
    {
      std::istringstream iss(dict.Get<std::string>(kAdvisorArgsId));
      std::string tok;
      while (iss >> tok) opt.advisor_engine_args.push_back(tok);
    }
    // UCI options: semicolon-separated Name=Value pairs
    {
      std::istringstream iss(dict.Get<std::string>(kAdvisorUciOptionsId));
      std::string entry;
      while (std::getline(iss, entry, ';')) {
        size_t b = 0;
        while (b < entry.size() && entry[b] == ' ') ++b;
        size_t e = entry.size();
        while (e > b && entry[e - 1] == ' ') --e;
        entry = entry.substr(b, e - b);
        if (entry.empty()) continue;
        auto eq = entry.find('=');
        if (eq == std::string::npos) {
          throw Exception("Bad advisor UCI option (missing '='): '" +
                          entry + "'");
        }
        opt.advisor_engine_uci_options.emplace_back(
            entry.substr(0, eq), entry.substr(eq + 1));
      }
    }
  }

  // Iterator to store the game in. Have to keep it so that later we can
  // delete it. Need to expose it in games_ member variable only because
  // of possible Abort() that should stop them all.
  std::list<std::unique_ptr<SelfPlayGame>>::iterator game_iter;
  SyzygyTablebase* syzygy_tb;
  {
    Mutex::Lock lock(mutex_);
    games_.emplace_front(std::make_unique<SelfPlayGame>(options[0], options[1],
                                                        kShareTree, opening));
    game_iter = games_.begin();
    syzygy_tb = syzygy_tb_.get();
  }
  auto& game = **game_iter;

  // If kResignPlaythrough == 0, then this comparison is unconditionally true
  const bool enable_resign =
      Random::Get().GetFloat(100.0f) >= kResignPlaythrough;

  // PLAY GAME!
  auto player1_threads = player_options_[0][color_idx[0]].Get<int>(kThreadsId);
  auto player2_threads = player_options_[1][color_idx[1]].Get<int>(kThreadsId);
  game.Play(player1_threads, player2_threads, kTraining, syzygy_tb,
            enable_resign);

  // If game was aborted, it's still undecided.
  if (game.GetGameResult() != GameResult::UNDECIDED) {
    // Game callback.
    GameInfo game_info;
    game_info.game_result = game.GetGameResult();
    game_info.is_black = player1_black;
    game_info.game_id = game_number;
    game_info.initial_fen = opening.start_fen;
    game_info.moves = game.GetMoves();
    game_info.play_start_ply = game.GetStartPly();
    game_info.sf_forced_mate = game.DidForceAdvisorMate();
    if (!enable_resign) {
      game_info.min_false_positive_threshold =
          game.GetWorstEvalForWinnerOrDraw();
    }
    if (kTraining &&
        game_info.play_start_ply < static_cast<int>(game_info.moves.size())) {
      TrainingDataWriter writer(game_number);
      game.WriteTrainingData(&writer);
      writer.Finalize();
      game_info.training_filename = writer.GetFileName();
    }
    game_callback_(game_info);

    // Update tournament stats.
    {
      Mutex::Lock lock(mutex_);
      int result = game.GetGameResult() == GameResult::DRAW        ? 1
                   : game.GetGameResult() == GameResult::WHITE_WON ? 0
                                                                   : 2;
      if (player1_black) result = 2 - result;
      ++tournament_info_.results[result][player1_black ? 1 : 0];
      tournament_info_.move_count_ += game.move_count_;
      tournament_info_.nodes_total_ += game.nodes_total_;
      tournament_callback_(tournament_info_);
    }
  }

  {
    Mutex::Lock lock(mutex_);
    games_.erase(game_iter);
  }
}

void SelfPlayTournament::PlayMultiGames(int game_id, size_t game_count) {
  bool use_value = kValueGamesSize > 0;
  std::vector<Opening> openings;
  openings.reserve(game_count / 2);
  size_t opening_basis = game_id / 2;
  {
    Mutex::Lock lock(mutex_);
    for (size_t i = 0; i < game_count / 2; i++) {
      openings.push_back(openings_[(opening_basis + i) % openings_.size()]);
    }
  }

  PlayerOptions options[2];
  options[0].backend = backends_[0][0].get();
  options[1].backend = backends_[1][1].get();

  std::list<std::unique_ptr<MultiSelfPlayGames>>::iterator game1_iter;
  auto aborted = false;
  {
    Mutex::Lock lock(mutex_);
    multigames_.emplace_front(std::make_unique<MultiSelfPlayGames>(
        options[0], options[1], openings, syzygy_tb_.get(), use_value));
    game1_iter = multigames_.begin();
    aborted = abort_;
  }
  auto& game1 = **game1_iter;

  // PLAY GAMEs!
  if (!aborted) game1.Play();

  options[0].backend = backends_[0][1].get();
  options[1].backend = backends_[1][0].get();

  std::list<std::unique_ptr<MultiSelfPlayGames>>::iterator game2_iter;
  {
    Mutex::Lock lock(mutex_);
    multigames_.emplace_front(std::make_unique<MultiSelfPlayGames>(
        options[1], options[0], openings, syzygy_tb_.get(), use_value));
    game2_iter = multigames_.begin();
    aborted = abort_;
  }
  auto& game2 = **game2_iter;
  // PLAY reverse GAMEs!
  if (!aborted) game2.Play();

  for (size_t i = 0; i < openings.size(); i++) {
    auto game1_res = game1.GetGameResult(i);
    if (game1_res != GameResult::UNDECIDED) {
      // Game callback.
      GameInfo game_info;
      game_info.game_result = game1_res;
      game_info.is_black = false;
      game_info.game_id = game_id + 2 * i;
      game_info.moves = game1.GetMoves(i);
      game_info.initial_fen = openings[i].start_fen;
      game_info.play_start_ply = openings[i].moves.size();
      game_callback_(game_info);

      // Update tournament stats.
      {
        Mutex::Lock lock(mutex_);
        int result = game1_res == GameResult::DRAW        ? 1
                     : game1_res == GameResult::WHITE_WON ? 0
                                                          : 2;
        ++tournament_info_.results[result][0];
        tournament_callback_(tournament_info_);
      }
    }
    auto game2_res = game2.GetGameResult(i);
    if (game2_res != GameResult::UNDECIDED) {
      // Game callback.
      GameInfo game_info;
      game_info.game_result = game2_res;
      game_info.is_black = true;
      game_info.game_id = game_id + 2 * i + 1;
      game_info.moves = game2.GetMoves(i);
      game_info.initial_fen = openings[i].start_fen;
      game_info.play_start_ply = openings[i].moves.size();
      game_callback_(game_info);

      // Update tournament stats.
      {
        Mutex::Lock lock(mutex_);
        int result = game2_res == GameResult::DRAW        ? 1
                     : game2_res == GameResult::WHITE_WON ? 2
                                                          : 0;
        ++tournament_info_.results[result][1];
        tournament_callback_(tournament_info_);
      }
    }
  }
  {
    Mutex::Lock lock(mutex_);
    multigames_.erase(game2_iter);
    multigames_.erase(game1_iter);
  }
}

void SelfPlayTournament::Worker() {
  // Play games while game limit is not reached (or while not aborted).
  while (true) {
    int game_id;
    int count = 0;
    {
      Mutex::Lock lock(mutex_);
      if (abort_) break;
      if (multi_games_size_) {
        if (!player_options_[0][0].Get<bool>(kOpeningsMirroredId)) {
          throw Exception(
              "Policy/Value multi games mode only supports mirrored openings.");
        }
        if (kTotalGames > 0 && kTotalGames % 2 == 1) {
          throw Exception(
              "Policy/Value multi games can't support mirrored with an odd "
              "number "
              "of games.");
        }
        int to_take = 2 * multi_games_size_;
        int max_take = 2 * multi_games_size_;
        if (kTotalGames != -1) {
          int cap = kTotalGames == -2 ? openings_.size() * 2 : kTotalGames;
          to_take = std::min(max_take, cap - games_count_);
        }
        if (to_take <= 0) {
          break;
        }
        game_id = games_count_;
        count = to_take;
        games_count_ += to_take;
      } else {
        bool mirrored = player_options_[0][0].Get<bool>(kOpeningsMirroredId);
        if ((kTotalGames >= 0 && games_count_ >= kTotalGames) ||
            (kTotalGames == -2 && !openings_.empty() &&
             games_count_ >=
                 static_cast<int>(openings_.size()) * (mirrored ? 2 : 1)))
          break;
        game_id = games_count_++;
      }
    }
    if (multi_games_size_) {
      PlayMultiGames(game_id, count);
    } else {
      PlayOneGame(game_id);
    }
  }
}

void SelfPlayTournament::StartAsync() {
  Mutex::Lock lock(threads_mutex_);
  while (threads_.size() < kParallelism) {
    threads_.emplace_back([&]() { Worker(); });
  }
}

void SelfPlayTournament::RunBlocking() {
  if (kParallelism == 1) {
    // No need for multiple threads if there is one worker.
    Worker();
    Mutex::Lock lock(mutex_);
    if (!abort_) {
      SaveResults();
      tournament_info_.finished = true;
      tournament_callback_(tournament_info_);
    }
  } else {
    StartAsync();
    Wait();
  }
}

void SelfPlayTournament::Wait() {
  {
    Mutex::Lock lock(threads_mutex_);
    while (!threads_.empty()) {
      threads_.back().join();
      threads_.pop_back();
    }
  }
  {
    Mutex::Lock lock(mutex_);
    if (!abort_) {
      SaveResults();
      tournament_info_.finished = true;
      tournament_callback_(tournament_info_);
    }
  }
}

void SelfPlayTournament::Abort() {
  Mutex::Lock lock(mutex_);
  abort_ = true;
  for (auto& game : games_)
    if (game) game->Abort();
  for (auto& game : multigames_)
    if (game) game->Abort();
}

void SelfPlayTournament::Stop() {
  Mutex::Lock lock(mutex_);
  abort_ = true;
}

SelfPlayTournament::~SelfPlayTournament() {
  Abort();
  Wait();
}

void SelfPlayTournament::SaveResults() {
  if (kTournamentResultsFile.empty()) return;
  std::ofstream output(kTournamentResultsFile, std::ios_base::app);
  auto p1name =
      player_options_[0][0].Get<std::string>(SharedBackendParams::kWeightsId);
  auto p2name =
      player_options_[1][0].Get<std::string>(SharedBackendParams::kWeightsId);

  output << std::endl;
  output << "[White \"" << p1name << "\"]" << std::endl;
  output << "[Black \"" << p2name << "\"]" << std::endl;
  output << "[Results \"" << tournament_info_.results[0][0] << " "
         << tournament_info_.results[2][0] << " "
         << tournament_info_.results[1][0] << "\"]" << std::endl;
  output << std::endl;
  output << "[White \"" << p2name << "\"]" << std::endl;
  output << "[Black \"" << p1name << "\"]" << std::endl;
  output << "[Results \"" << tournament_info_.results[2][1] << " "
         << tournament_info_.results[0][1] << " "
         << tournament_info_.results[1][1] << "\"]" << std::endl;
}

}  // namespace lczero
