#include "position.h"

#include <atomic>
#include <memory>
#include <random>
#include <vector>
#pragma once

struct alignas(8) Node {
  U32 first_child_idx;
  std::atomic<int> visits;
  std::atomic<float> value_sum;
  float prior;
  std::atomic_flag is_expanding = ATOMIC_FLAG_INIT;

  Move move;
  U8 num_children;
  std::atomic<U8> virtual_visits;

  Node()
      : first_child_idx(0), visits(0), value_sum(0.0f), prior(0.0f),
        move(Move()), num_children(0), virtual_visits(0) {}
};

struct TreeArena {
  std::unique_ptr<Node[]> nodes;
  std::atomic<size_t> active_nodes;
  size_t max_nodes;

  TreeArena(size_t initial_capacity = 3145728) {
    nodes = std::make_unique<Node[]>(initial_capacity);
    active_nodes.store(0, std::memory_order_relaxed);
    max_nodes = initial_capacity;
  }

  void resize(int megabytes);
  void clear();
};

float dummy_evaluate(const Position &pos);
U32 select_best_puct(const TreeArena &arena, U32 node_idx);
int mcts_rollout(TreeArena &arena, const Position &root_pos,
                 const std::vector<U64> &game_hashes);