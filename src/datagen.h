#include "inference.h"
#include "node.h"
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

struct DatagenGame {
  Position root_pos;
  TreeArena arena;
  std::vector<U64> game_hashes;

  // MCTS Step State
  Position leaf_pos;
  std::vector<U32> search_path;
  int rollouts_completed = 0;

  // Datagen History
  std::vector<TrainingPosition> game_history;
  int ply_count = 0;
  bool is_active = true;

  DatagenGame(size_t arena_size) : arena(arena_size) { root_pos.initialize(); }

  void reset(std::mt19937 &rng) {
    std::uniform_int_distribution<int> dist(0, 359);
    int seed1 = dist(rng);
    int seed2 = dist(rng);

    std::string fen = get129600FEN(seed1, seed2);
    root_pos.parseFEN(fen);

    // Clear all state for the new game
    arena.clear();
    game_hashes.clear();
    game_history.clear();
    ply_count = 0;
    rollouts_completed = 0;
    is_active = true;
  }
};

void generate_batched_selfplay_games(NNEvaluator &nn,
                                     const std::string &output_prefix,
                                     U64 nodecount, int total_games_to_play);