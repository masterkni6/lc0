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

#pragma once

#include <cstdint>

#include "utils/cppattributes.h"

namespace lczero {

// V7 is the canonical training-data record format.  It extends V6 with
// short-term D running average (d_st), the opponent's played move from
// the next position (opp_played_idx), the next-next position's played
// move (next_played_idx), and 8 reserved float slots for future use.
//
// Earlier formats (V3 .. V6) are read transparently by TrainingDataReader
// (see reader.cc) which performs an in-place upgrade: V6 chunks on disk
// (8356 bytes, header version=6) are zero-padded to V7 size on read and
// the version byte is bumped to 7.  Selfplay writes V7 natively; the
// rescorer also produces V7 output.  Anything on disk older than V7 is
// silently upgraded during the read pass.
//
// Size is exactly 8396 bytes.  The Python chunkparser asserts the same
// constant; if you change this struct the trainer's V7_STRUCT_STRING
// must change in lockstep.
#pragma pack(push, 1)

struct V7TrainingData {
  uint32_t version;
  uint32_t input_format;
  float probabilities[1858];
  uint64_t planes[104];
  uint8_t castling_us_ooo;
  uint8_t castling_us_oo;
  uint8_t castling_them_ooo;
  uint8_t castling_them_oo;
  // For input type 3 contains enpassant column as a mask.
  uint8_t side_to_move_or_enpassant;
  uint8_t rule50_count;
  // Bitfield with the following allocation:
  //  bit 7: side to move (input type 3)
  //  bit 6: position marked for deletion by the rescorer (set by
  //         AddPlaceholder for external-opponent moves)
  //  bit 5: game adjudicated (v6+)
  //  bit 4: max game length exceeded (v6+)
  //  bit 3: best_q is for proven best move (v6+)
  //  bit 2: transpose transform (input type 3)
  //  bit 1: mirror transform (input type 3)
  //  bit 0: flip transform (input type 3)
  // In versions prior to v5 this spot contained an unused move count field.
  uint8_t invariance_info;
  // In versions prior to v6 this spot contained the result as an int8_t.
  uint8_t dummy;
  float root_q;
  float best_q;
  float root_d;
  float best_d;
  float root_m;      // In plies.
  float best_m;      // In plies.
  float plies_left;  // This is the training target for MLH.
  float result_q;
  float result_d;
  float played_q;
  float played_d;
  float played_m;
  // The following may be NaN if not found in cache.
  float orig_q;  // For value repair.
  float orig_d;
  float orig_m;
  uint32_t visits;
  // Indices in the probabilities array.
  uint16_t played_idx;
  uint16_t best_idx;
  // Kullback-Leibler divergence between visits and policy (denominator)
  float policy_kld;
  // Q standard deviation (short-term EMA).  Was uint32_t reserved in
  // original v6 spec; lc0 repurposed it for v6.5 / v7.
  float q_st;
  // ─── V7 extras (added 2023) ───
  // D standard deviation, computed alongside q_st via backward EMA.
  float d_st;
  // played_idx of the next position (i.e. the opponent's reply move).
  // 65535 sentinel when past end-of-game.
  uint16_t opp_played_idx;
  // played_idx of the position two plies ahead (our next own move).
  // 65535 sentinel when past end-of-game.
  uint16_t next_played_idx;
  // Reserved for future use.  Always zero in current writers.
  float reserved[8];
} PACKED_STRUCT;
static_assert(sizeof(V7TrainingData) == 8396, "Wrong struct size");

#pragma pack(pop)

}  // namespace lczero
