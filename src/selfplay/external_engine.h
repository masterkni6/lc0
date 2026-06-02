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

#include <chrono>
#include <string>
#include <utility>
#include <vector>

namespace lczero {

// The external engine's evaluation of the position it was queried about, from
// the side-to-move's perspective.  Optionally filled by GetMove() from the last
// `info ... score ...` line before `bestmove`.
struct AdvisorScore {
  bool valid = false;    // false if no `score` line was parsed
  bool is_mate = false;  // true if the engine reported `score mate N`
  int mate_in = 0;       // N from `score mate N` (>0 = side-to-move mates)
  int score_cp = 0;      // X from `score cp X` (valid iff !is_mate)
  // SF's win/draw/loss, per-mille (sum ~1000), side-to-move POV, parsed from a
  // `wdl W D L` token when UCI_ShowWDL is enabled.  has_wdl is false if absent.
  // A calibrated value target — no cp->value conversion needed.
  bool has_wdl = false;
  int wdl_w = 0;
  int wdl_d = 0;
  int wdl_l = 0;
};

// Drives an external UCI engine subprocess (e.g. Stockfish) used as an
// opponent during selfplay. One instance per side per game; reused across
// moves within a game (re-instantiated per game so engine state is reset).
//
// POSIX only for now. On Windows, ExternalEngine throws on construction.
class ExternalEngine {
 public:
  // path: executable to spawn (e.g. "/usr/games/stockfish").
  // args: command-line args passed verbatim.
  // uci_options: ordered list of (name, value) sent as `setoption` lines
  //              between `uci` and `isready`.  Ordered (not a map) because
  //              some engines are sensitive to setoption order — e.g.
  //              Threads must precede Hash on some Stockfish builds.
  // go_command: the argument string appended after "go " on every move,
  //             e.g. "movetime 100", "depth 12", "nodes 50000".
  // chess960:   if true, the engine is configured for FRC/DFRC: we set
  //             UCI_Chess960=true on the engine (unless the user already
  //             did) and rewrite FEN castling rights to Shredder-FEN
  //             file letters before each `position` command.  If false,
  //             standard chess is assumed and the engine runs in default
  //             mode — important because some engines (Stockfish) play
  //             measurably differently in 960 mode even for standard
  //             positions, and forcing 960 mode degrades their strength.
  ExternalEngine(
      const std::string& path,
      const std::vector<std::string>& args,
      const std::vector<std::pair<std::string, std::string>>& uci_options,
      const std::string& go_command,
      bool chess960 = false);

  ~ExternalEngine();

  ExternalEngine(const ExternalEngine&) = delete;
  ExternalEngine& operator=(const ExternalEngine&) = delete;

  // Send `position fen <fen>` (optionally with `moves <m1> <m2> ...` if
  // moves_uci is non-empty), then `go <go_command>`, then read until the
  // engine emits `bestmove <uci>`.  Returns the UCI move string.  Throws
  // on subprocess failure or parse failure.
  // If `score` is non-null it is filled with the engine's final reported
  // evaluation (mate/cp, side-to-move POV), parsed from the last
  // `info ... score ...` line before `bestmove`.  Used by the advisor to detect
  // forced mates (and to support eval-disagreement gating).
  std::string GetMove(const std::string& fen,
                      const std::vector<std::string>& moves_uci,
                      AdvisorScore* score = nullptr);

 private:
  // Spawn the subprocess and set up stdin/stdout pipes.  Throws on
  // fork/exec failure.
  void Spawn(const std::string& path, const std::vector<std::string>& args);

  // Write a single line ("<msg>\n") to engine stdin.  Throws if the
  // engine's stdin is closed.
  void WriteLine(const std::string& msg);

  // Read a single line (up to and including '\n') from engine stdout.
  // The trailing '\n' is stripped.  Times out after `timeout` and throws
  // if the engine hasn't emitted a full line in that window — this is
  // how we detect a hung Stockfish without blocking selfplay forever.
  std::string ReadLine(std::chrono::milliseconds timeout);

  // Read lines until one starts with `prefix` (whitespace-trimmed).
  // Returns that line.  Bounded by total elapsed `timeout`.  When
  // `echo_dropped` is true, any non-matching line is logged to stderr
  // with an "[opponent]" prefix — used during the handshake so the
  // user can confirm NNUE loaded.  During gameplay we keep it false
  // because SF re-emits the NNUE info string on every `go` command,
  // which would flood the log.
  // If `last_score_line` is non-null, the most recent dropped line containing
  // " score " is copied there (the engine's deepest eval before `bestmove`).
  std::string ReadUntilPrefix(const std::string& prefix,
                              std::chrono::milliseconds timeout,
                              bool echo_dropped = false,
                              std::string* last_score_line = nullptr);

#ifdef _WIN32
  // Placeholder fields so the class still compiles on Windows.  The
  // constructor throws there, so these are never actually used.
  int unused_ = 0;
#else
  int stdin_fd_ = -1;   // engine's stdin (we write here)
  int stdout_fd_ = -1;  // engine's stdout (we read here)
  int pid_ = -1;
  std::string read_buf_;  // line buffer for ReadLine
#endif

  std::string go_command_;
  bool chess960_ = false;
};

}  // namespace lczero
