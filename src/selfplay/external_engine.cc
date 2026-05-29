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

#include "selfplay/external_engine.h"

#include <algorithm>
#include <sstream>

#include "utils/exception.h"

#ifndef _WIN32
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <string.h>
#include <sys/select.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/prctl.h>
#endif
#endif

namespace lczero {

namespace {

// Strip trailing CR/LF and leading whitespace.  UCI engines on Windows
// occasionally emit "bestmove e2e4\r\n" so we strip both.
std::string Trim(const std::string& s) {
  size_t b = 0;
  while (b < s.size() && (s[b] == ' ' || s[b] == '\t')) ++b;
  size_t e = s.size();
  while (e > b && (s[e - 1] == '\n' || s[e - 1] == '\r' || s[e - 1] == ' '))
    --e;
  return s.substr(b, e - b);
}

// True if `s` starts with `prefix` after trimming.
bool StartsWith(const std::string& s, const std::string& prefix) {
  return s.size() >= prefix.size() &&
         s.compare(0, prefix.size(), prefix) == 0;
}

// ─── TODO: FRC/DFRC advisor + opponent reliability ──────────────────────
// As of 2026-05-25, advisor-mode selfplay in FRC/DFRC fails after a few
// moves with "Invalid move (no piece to move): <some_move>" parse errors.
// The pattern: lc0 sends SF a FEN + move list, SF returns a move, lc0
// can't parse it against its current board.  The divergence happens
// silently — lc0 and SF agree on the first N moves and disagree from
// some point on, after which every SF move is unparseable for lc0.
//
// Root cause (likely, not yet confirmed):
//   1. KQkq castling field is ambiguous in FRC when multiple rooks per
//      side exist (e.g. RKRQBBNN — there are two rooks on each side of
//      the white king).  lc0 walks outward from the king to find the
//      nearest rook on each side; SF uses a different inference rule.
//      First castling move silently moves a different rook in each
//      engine's view → positions diverge.
//   2. Castling move format mismatch: FRC castling can be UCI
//      "king-takes-rook" (e1h1) or standard (e1g1).  lc0's
//      Move::ToString(is_chess960=true) emits king-takes-rook; SF with
//      UCI_Chess960=true expects king-takes-rook.  Should match, but
//      some edge cases (e.g. king and rook already adjacent) may differ.
//
// Workaround (now being retired): generate_advisor.py and generate_sf.py
// were previously pinned to UHO_XXL standard chess (no --chess960) to
// avoid the divergence bug.  Pure-selfplay generate.py is unaffected.
//
// Step A — Shredder-FEN rewrite restored in GetMove() (see below).
//   FenWithShredderCastling() converts ambiguous KQkq castling into
//   explicit file-letter form (HAha) so lc0 and SF agree on which rook
//   is the castling rook on each side.  Was previously disabled under
//   the theory that "SF plays measurably weaker in 960 mode" — but
//   that A/B was confounded by other factors and the bug recurred on
//   DFRC, proving KQkq was NOT actually unambiguous in production.
//   The rewrite is on for chess960 mode now.  If opponent-mode play
//   strength regresses noticeably we'll add an is_advisor gate
//   instead of disabling it again.
//
// Step B (TODO) — comprehensive harness: send SF a sequence of FRC
//   moves from N random DFRC start positions, run each for 10 moves
//   with both engines, assert no parse failures.  Catches future
//   regressions if the rewrite gets disabled again.
//
// Step C (TODO) — flip generate_advisor.py back to dfrc.pgn +
//   --chess960=true and remove the workaround pin to UHO_XXL.
//
// Priority: now medium-high (Step A landed; B+C pending verification).

// Rewrite a FEN so its castling field uses Shredder-FEN file letters
// (e.g. "HAha") instead of the ambiguous standard notation "KQkq".
//
// Why this exists:
//   In FRC/DFRC positions, the standard "KQkq" notation only tells you
//   "castling is available", not WHICH rook is the castling rook when
//   there are multiple rooks on the same side of the king.  Engines
//   then have to *infer* the castling rook positions, and they don't
//   all use the same inference rule:
//     - lc0: uses the rook nearest the king on each side.
//     - Stockfish: scans outward from king but with quirks around
//       positions like RKRQBBNN where there's no h1 rook at all.
//   Result: lc0 and SF disagree on castling rights → SF eventually
//   returns a castling move from a position where lc0 has no king,
//   we get the "no piece to move" parse error.
//
// Fix: emit explicit rook files so both engines agree.  Scan the back
// ranks for the actual rook positions on each side of the king and
// substitute file letters.  Standard chess starting positions are
// handled transparently — `KQkq` becomes `HAha` (capital letters for
// white, lowercase for black; "HA" means h-file kingside rook and
// a-file queenside rook).
std::string FenWithShredderCastling(const std::string& fen) {
  // FEN format: <pieces> <stm> <castling> <ep> <r50> <ply>
  std::vector<std::string> parts;
  std::istringstream iss(fen);
  std::string tok;
  while (iss >> tok) parts.push_back(tok);
  if (parts.size() < 3) return fen;  // malformed — pass through
  const std::string& pieces = parts[0];
  const std::string& castling = parts[2];
  if (castling == "-" || castling.empty()) return fen;

  // If the castling field already uses file letters, leave it alone —
  // we only convert "KQkq"-style.  Quick check: any KQkq char present?
  bool needs_conversion = false;
  for (char c : castling) {
    if (c == 'K' || c == 'Q' || c == 'k' || c == 'q') {
      needs_conversion = true;
      break;
    }
  }
  if (!needs_conversion) return fen;

  // Find white and black king files by walking the back-rank piece
  // strings.  rank8 = before first '/'; rank1 = after last '/'.
  auto first_slash = pieces.find('/');
  auto last_slash = pieces.rfind('/');
  if (first_slash == std::string::npos || last_slash == std::string::npos) {
    return fen;
  }
  std::string rank8 = pieces.substr(0, first_slash);
  std::string rank1 = pieces.substr(last_slash + 1);

  // Expand a FEN rank ("p1p1k2r") to a file→piece array indexed 0..7.
  auto expand = [](const std::string& r) -> std::string {
    std::string out(8, '.');
    int idx = 0;
    for (char c : r) {
      if (idx >= 8) break;
      if (c >= '1' && c <= '8') {
        idx += (c - '0');
      } else {
        out[idx++] = c;
      }
    }
    return out;
  };

  std::string r1 = expand(rank1);  // white pieces (uppercase)
  std::string r8 = expand(rank8);  // black pieces (lowercase)

  // Locate king file on each rank.
  int white_king = -1, black_king = -1;
  for (int f = 0; f < 8; ++f) {
    if (r1[f] == 'K') white_king = f;
    if (r8[f] == 'k') black_king = f;
  }
  if (white_king < 0 && black_king < 0) return fen;  // no kings? bail.

  // For each castling character, find the matching rook file.
  // Strategy mirrors what lc0's own parser does: scan from the king
  // outward (kingside = right, queenside = left) for the first rook on
  // the back rank.  This matches the convention used by every standard
  // FRC opening tool I checked.
  auto find_rook_file = [](const std::string& rank, char rook_glyph,
                            int king_file, bool kingside) -> int {
    if (king_file < 0) return -1;
    if (kingside) {
      for (int f = 7; f > king_file; --f) {
        if (rank[f] == rook_glyph) return f;
      }
    } else {
      for (int f = 0; f < king_file; ++f) {
        if (rank[f] == rook_glyph) return f;
      }
    }
    return -1;
  };

  std::string new_castling;
  for (char c : castling) {
    int file = -1;
    switch (c) {
      case 'K':
        file = find_rook_file(r1, 'R', white_king, /*kingside=*/true);
        if (file >= 0) new_castling.push_back('A' + file);  // uppercase
        break;
      case 'Q':
        file = find_rook_file(r1, 'R', white_king, /*kingside=*/false);
        if (file >= 0) new_castling.push_back('A' + file);
        break;
      case 'k':
        file = find_rook_file(r8, 'r', black_king, /*kingside=*/true);
        if (file >= 0) new_castling.push_back('a' + file);  // lowercase
        break;
      case 'q':
        file = find_rook_file(r8, 'r', black_king, /*kingside=*/false);
        if (file >= 0) new_castling.push_back('a' + file);
        break;
      default:
        // Already a file letter — pass through unchanged.
        new_castling.push_back(c);
        break;
    }
  }
  if (new_castling.empty()) new_castling = "-";

  // Reassemble FEN.
  parts[2] = new_castling;
  std::ostringstream out;
  for (size_t i = 0; i < parts.size(); ++i) {
    if (i) out << ' ';
    out << parts[i];
  }
  return out.str();
}

}  // namespace

ExternalEngine::ExternalEngine(
    const std::string& path,
    const std::vector<std::string>& args,
    const std::vector<std::pair<std::string, std::string>>& uci_options,
    const std::string& go_command,
    bool chess960)
    : go_command_(go_command), chess960_(chess960) {
#ifdef _WIN32
  (void)path;
  (void)args;
  (void)uci_options;
  throw Exception(
      "External UCI engine support is not implemented on Windows yet.");
#else
  Spawn(path, args);

  // UCI handshake.  Per spec: send `uci`, wait for `uciok`, send any
  // `setoption` lines, then `isready` / `readyok`, then `ucinewgame`.
  // Generous budget: with N parallel selfplay games spawning N engines
  // simultaneously, NNUE-network loading + thread allocation can take
  // several seconds under disk and CPU contention.  60s is well above
  // any reasonable real startup time; faster than that risks spurious
  // adjudications during the first move of every game.
  const auto handshake_timeout = std::chrono::seconds(60);
  WriteLine("uci");
  // Don't echo handshake output — at parallelism=N over many games the
  // banner+option-list re-spam is unhelpful. Set echo_dropped=true here
  // temporarily if you need to debug NNUE-load issues.
  ReadUntilPrefix("uciok", handshake_timeout);

  // Auto-set UCI_Chess960=true on the engine when we're playing an FRC
  // position. SF cannot otherwise interpret king-takes-rook castling
  // moves like "c8d8" (king at c8, rook at d8) — without 960 mode it
  // sees "king takes own rook" and rejects the move, causing position
  // desync. Re-enabled after the Shredder-FEN rewrite was removed; the
  // earlier "SF plays weaker in 960 mode" issue was apparently caused
  // by the FEN castling-field rewrite, not the option itself. cute-chess
  // also sets this and gets the expected SF strength.
  if (chess960_) {
    bool user_set_chess960 = false;
    for (const auto& [name, value] : uci_options) {
      if (name.size() == 12 &&
          (name == "UCI_Chess960" || name == "UCI_chess960")) {
        user_set_chess960 = true;
        break;
      }
    }
    if (!user_set_chess960) {
      WriteLine("setoption name UCI_Chess960 value true");
    }
  }

  for (const auto& [name, value] : uci_options) {
    WriteLine("setoption name " + name + " value " + value);
  }

  WriteLine("isready");
  ReadUntilPrefix("readyok", handshake_timeout);

  WriteLine("ucinewgame");
#endif
}

ExternalEngine::~ExternalEngine() {
#ifndef _WIN32
  // Best-effort graceful shutdown: send "quit", then close pipes and
  // reap.  If the engine doesn't exit within 500ms after pipe close,
  // SIGKILL it — we never want a stuck Stockfish leaking into the next
  // game's process slot.
  if (stdin_fd_ >= 0) {
    const char quit[] = "quit\n";
    // Use write() ignoring errors — pipe may already be closed.
    ssize_t n = write(stdin_fd_, quit, sizeof(quit) - 1);
    (void)n;
    close(stdin_fd_);
    stdin_fd_ = -1;
  }
  if (stdout_fd_ >= 0) {
    close(stdout_fd_);
    stdout_fd_ = -1;
  }
  if (pid_ > 0) {
    // Poll for ~500ms, then SIGKILL.
    for (int i = 0; i < 50; ++i) {
      int status;
      pid_t r = waitpid(pid_, &status, WNOHANG);
      if (r == pid_ || r < 0) {
        pid_ = -1;
        break;
      }
      usleep(10000);  // 10ms
    }
    if (pid_ > 0) {
      kill(pid_, SIGKILL);
      waitpid(pid_, nullptr, 0);
      pid_ = -1;
    }
  }
#endif
}

#ifndef _WIN32

void ExternalEngine::Spawn(const std::string& path,
                           const std::vector<std::string>& args) {
  // Create two pipes: parent->child stdin, child->parent stdout.
  int in_pipe[2];   // parent writes in_pipe[1], child reads in_pipe[0]
  int out_pipe[2];  // child writes out_pipe[1], parent reads out_pipe[0]
  if (pipe(in_pipe) != 0 || pipe(out_pipe) != 0) {
    throw Exception(std::string("ExternalEngine: pipe() failed: ") +
                    strerror(errno));
  }

  pid_t pid = fork();
  if (pid < 0) {
    throw Exception(std::string("ExternalEngine: fork() failed: ") +
                    strerror(errno));
  }

  if (pid == 0) {
    // ─── Child ───
    // Auto-die when the parent (lc0) exits, no matter how — Ctrl-C,
    // SIGKILL, segfault, anything.  Without this, killing lc0 with
    // Ctrl-C leaves orphan Stockfish processes that keep eating CPU
    // until the next reboot or manual pkill.  Worse: on the next lc0
    // run, the orphans saturate the cores and the new SF instances
    // can't respond to UCI handshake within 60s, causing 100% spawn
    // failures.  PR_SET_PDEATHSIG fixes this at the kernel level.
    //
    // Linux-only; on other Unix the user has to manually pkill on Ctrl-C.
#ifdef __linux__
    prctl(PR_SET_PDEATHSIG, SIGKILL);
    // Race: if the parent died between fork and prctl, we'd never get
    // signaled.  Check immediately after.
    if (getppid() == 1) _exit(0);
#endif

    // Wire stdin/stdout to the parent pipes.
    dup2(in_pipe[0], STDIN_FILENO);
    dup2(out_pipe[1], STDOUT_FILENO);
    // We intentionally do NOT redirect stderr; engine errors go to our
    // stderr so the user sees them mixed with lc0's logs, which is what
    // you want when debugging a misconfigured opponent.
    close(in_pipe[0]);
    close(in_pipe[1]);
    close(out_pipe[0]);
    close(out_pipe[1]);

    // execvp expects char* const argv[].  Build it.
    std::vector<std::string> argv_storage;
    argv_storage.reserve(args.size() + 1);
    argv_storage.push_back(path);
    for (const auto& a : args) argv_storage.push_back(a);
    std::vector<char*> argv;
    argv.reserve(argv_storage.size() + 1);
    for (auto& s : argv_storage) argv.push_back(s.data());
    argv.push_back(nullptr);

    execvp(path.c_str(), argv.data());
    // If we got here, exec failed.  Write to stderr and exit; the parent
    // will see EOF on out_pipe and throw.
    fprintf(stderr, "ExternalEngine: execvp(%s) failed: %s\n", path.c_str(),
            strerror(errno));
    _exit(127);
  }

  // ─── Parent ───
  close(in_pipe[0]);
  close(out_pipe[1]);
  stdin_fd_ = in_pipe[1];
  stdout_fd_ = out_pipe[0];
  pid_ = pid;

  // Set stdout non-blocking so ReadLine can use select() + read() and
  // implement a real timeout without permanently blocking on a dead engine.
  int flags = fcntl(stdout_fd_, F_GETFL, 0);
  if (flags == -1) flags = 0;
  fcntl(stdout_fd_, F_SETFL, flags | O_NONBLOCK);
}

void ExternalEngine::WriteLine(const std::string& msg) {
  std::string out = msg + "\n";
  size_t written = 0;
  while (written < out.size()) {
    ssize_t n = write(stdin_fd_, out.data() + written, out.size() - written);
    if (n < 0) {
      if (errno == EINTR) continue;
      throw Exception("ExternalEngine: write to engine failed: " +
                      std::string(strerror(errno)));
    }
    written += static_cast<size_t>(n);
  }
}

std::string ExternalEngine::ReadLine(std::chrono::milliseconds timeout) {
  // First check if our line buffer already contains a complete line.
  auto extract_line = [&]() -> std::string {
    auto nl = read_buf_.find('\n');
    if (nl == std::string::npos) return std::string();
    std::string line = read_buf_.substr(0, nl);
    read_buf_.erase(0, nl + 1);
    // Strip trailing CR (Windows line endings from some engines).
    if (!line.empty() && line.back() == '\r') line.pop_back();
    return line;
  };

  // Drain any complete buffered line immediately — common case after the
  // engine emits multiple `info` lines and a `bestmove` in quick succession.
  if (read_buf_.find('\n') != std::string::npos) {
    return extract_line();
  }

  auto deadline = std::chrono::steady_clock::now() + timeout;
  char chunk[4096];
  while (true) {
    auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      throw Exception("ExternalEngine: timed out waiting for output");
    }
    auto remaining =
        std::chrono::duration_cast<std::chrono::microseconds>(deadline - now);

    fd_set rfds;
    FD_ZERO(&rfds);
    FD_SET(stdout_fd_, &rfds);
    struct timeval tv;
    tv.tv_sec = remaining.count() / 1000000;
    tv.tv_usec = remaining.count() % 1000000;
    int sret = select(stdout_fd_ + 1, &rfds, nullptr, nullptr, &tv);
    if (sret < 0) {
      if (errno == EINTR) continue;
      throw Exception("ExternalEngine: select() failed: " +
                      std::string(strerror(errno)));
    }
    if (sret == 0) {
      throw Exception("ExternalEngine: timed out waiting for output");
    }
    ssize_t n = read(stdout_fd_, chunk, sizeof(chunk));
    if (n < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) continue;
      throw Exception("ExternalEngine: read failed: " +
                      std::string(strerror(errno)));
    }
    if (n == 0) {
      // EOF — engine exited.  Drain whatever's in the buffer; if there's
      // no terminating newline, treat it as a complete line so the caller
      // can see partial output before erroring on the next read.
      if (!read_buf_.empty()) {
        std::string line = read_buf_;
        read_buf_.clear();
        return line;
      }
      throw Exception("ExternalEngine: engine closed stdout (crashed?)");
    }
    read_buf_.append(chunk, chunk + n);
    if (read_buf_.find('\n') != std::string::npos) {
      return extract_line();
    }
  }
}

std::string ExternalEngine::ReadUntilPrefix(
    const std::string& prefix, std::chrono::milliseconds timeout,
    bool echo_dropped) {
  auto deadline = std::chrono::steady_clock::now() + timeout;
  while (true) {
    auto now = std::chrono::steady_clock::now();
    if (now >= deadline) {
      throw Exception("ExternalEngine: timed out waiting for '" + prefix + "'");
    }
    auto remaining =
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
    std::string line = Trim(ReadLine(remaining));
    if (StartsWith(line, prefix)) return line;
    // Only echo during handshake (echo_dropped=true).  During gameplay
    // SF emits the NNUE/Network-replica info strings on every `go`
    // command, plus per-iteration search progress — would flood the
    // log.  The handshake echo gives one-shot visibility into whether
    // NNUE actually loaded; that's enough.
    if (echo_dropped) {
      fprintf(stderr, "[opponent] %s\n", line.c_str());
    }
  }
}

std::string ExternalEngine::GetMove(const std::string& fen,
                                    const std::vector<std::string>& moves_uci) {
  // FEN rewriting for FRC/DFRC.
  //
  // History: this rewrite was originally added to disambiguate KQkq
  // castling notation when multiple rooks per side exist (DFRC
  // positions like RKRQBBNN).  It was REMOVED later because A/B
  // testing against cute-chess seemed to show SF playing weaker in
  // 960 mode whenever the rewrite was on — under the theory that
  // KQkq was unambiguous for "typical" DFRC start positions (only 2
  // rooks per side).
  //
  // That theory turned out to be incomplete: in production with
  // DFRC dfrc.pgn + advisor mode we DO observe the divergence bug
  // — SF returns moves lc0 can't parse a few moves into the game.
  // Conclusion: KQkq is ambiguous often enough on real DFRC start
  // positions to break advisor mode.  Re-enabling the rewrite
  // unconditionally for chess960 mode; if it actually does cost
  // SF strength in opponent mode, we'll measure it and add an
  // is_advisor flag to gate the rewrite to advisor-only later.
  //
  // For standard chess (!chess960_) the rewrite is a no-op semantic
  // pass-through but we skip it anyway to keep the diff minimal and
  // avoid changing the hot path for the common case.
  std::ostringstream pos_cmd;
  pos_cmd << "position fen "
          << (chess960_ ? FenWithShredderCastling(fen) : fen);
  if (!moves_uci.empty()) {
    pos_cmd << " moves";
    for (const auto& m : moves_uci) pos_cmd << " " << m;
  }
  WriteLine(pos_cmd.str());
  WriteLine("go " + go_command_);

  // Search budget.  For "movetime N" mode we can derive a tight bound
  // (N + 10s slack).  For "nodes N" or "depth N" modes we can't predict
  // wall-clock time without knowing the engine speed and CPU load, so
  // use a very generous 300s default.  At 4 parallel games × 4 SF
  // threads on a 16-core machine the first move can occasionally stretch
  // past 60s on complex positions due to thread contention; this
  // headroom prevents spurious adjudications.
  std::chrono::milliseconds search_timeout = std::chrono::seconds(300);
  {
    // Look for "movetime N" prefix and adjust.
    auto pos = go_command_.find("movetime");
    if (pos != std::string::npos) {
      std::istringstream iss(go_command_.substr(pos + 8));
      long ms = 0;
      if (iss >> ms && ms > 0) {
        search_timeout = std::chrono::milliseconds(ms + 10000);
      }
    }
  }

  std::string line = ReadUntilPrefix("bestmove", search_timeout);
  // line is like "bestmove e2e4" or "bestmove e7e8q ponder ...".
  std::istringstream iss(line);
  std::string token, mv;
  iss >> token >> mv;  // "bestmove", then the move
  if (mv.empty() || mv == "(none)" || mv == "0000") {
    throw Exception("ExternalEngine: no move from engine ('" + line + "')");
  }
  return mv;
}

#else  // _WIN32

void ExternalEngine::Spawn(const std::string&,
                           const std::vector<std::string>&) {}
void ExternalEngine::WriteLine(const std::string&) {}
std::string ExternalEngine::ReadLine(std::chrono::milliseconds) { return ""; }
std::string ExternalEngine::ReadUntilPrefix(const std::string&,
                                            std::chrono::milliseconds) {
  return "";
}
std::string ExternalEngine::GetMove(const std::string&,
                                    const std::vector<std::string>&) {
  return "";
}

#endif

}  // namespace lczero
