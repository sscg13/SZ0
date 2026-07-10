#include "node.h"
#include "position.h"

#pragma once

// particle_eta > 0 enables particle MCTS (arXiv:2605.08982) with proposal
// temperature eta in place of PUCT + virtual visits; ignored while
// contempt_nscl > 0.
void search_position(NNEvaluator &nn, TreeArena &arena,
                     const Position &current_pos,
                     const std::vector<uint64_t> &game_hashes, int timelimit,
                     U64 nodelimit, int threadcount, bool print_info,
                     int contempt_nscl = 0, float particle_eta = 0.0f);
