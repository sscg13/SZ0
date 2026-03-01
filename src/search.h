#include "node.h"
#include "position.h"

#pragma once

void search_position(NNEvaluator &nn, TreeArena &arena,
                     const Position &current_pos,
                     const std::vector<uint64_t> &game_hashes, int timelimit,
                     U64 nodelimit, int threadcount, bool print_info);
