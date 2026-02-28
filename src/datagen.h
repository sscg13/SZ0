#include "position.h"

#pragma once

struct TrainingPosition {
    float move_probabilities[maxmoves];
    float root_q;
    U16 move_indices[maxmoves];
    U8 board_tokens[64];
    U8 num_moves;
    U8 halfmove_clock;
    I8 outcome;
};