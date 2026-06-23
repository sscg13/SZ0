#include "inference.h"
#include "position.h"

#include <atomic>
#include <future>
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
  // Frozen visit count for search-contempt Thompson Sampling (arXiv:2504.07757).
  // Sits in the padding byte between is_expanding and move — Node stays 24 bytes.
  std::atomic<U8> frozen_visits;
  Move move;
  U8 num_children;
  std::atomic<U8> virtual_visits;

  Node()
      : first_child_idx(-1), visits(0), value_sum(0.0f), prior(0.0f),
        frozen_visits(0), move(Move()), num_children(0), virtual_visits(0) {}
};
static_assert(sizeof(Node) == 24, "Node size must remain 24 bytes");

struct TreeArena {
  std::unique_ptr<Node[]> nodes;
  std::atomic<size_t> active_nodes;
  size_t max_nodes;

  TreeArena(size_t initial_capacity) {
    nodes = std::make_unique<Node[]>(initial_capacity);
    active_nodes.store(1, std::memory_order_relaxed);
    max_nodes = initial_capacity;
  }

  void resize(int megabytes);
  void clear();
};

// State preserved between mcts_select and mcts_expand_and_backprop.
struct PendingRollout {
  std::vector<U32> search_path;
  std::vector<U64> rollout_hashes;
  Position leaf_pos;
  Move moves[maxmoves];
  int movecount;
  int depth;
};

bool is_repetition(const Position &pos, const std::vector<U64> &game_hashes,
                   const std::vector<U64> &rollout_hashes);
U32 select_best_puct(const TreeArena &arena, U32 node_idx);

// CPU path: monolithic rollout with a blocking NN call.
int mcts_rollout(NNEvaluator &nn, TreeArena &arena, const Position &root_pos,
                 const std::vector<U64> &game_hashes, int nscl = 0);

// GPU path — three-phase split so workers stay busy while the GPU runs.
//
// Walk the tree to a leaf and decide what to do:
//   Returns 0  — collision: another thread is already expanding this leaf.
//                Discard and retry; the path has been cleaned up.
//   Returns 1  — terminal: value is written to out_value.
//                Call mcts_backprop(arena, pending.search_path, out_value).
//   Returns 2  — needs NN: pending and out_future are valid.
//                Call mcts_expand_and_backprop once out_future.get() returns.
int mcts_select(BatchEvaluator &batch_eval, TreeArena &arena,
                const Position &root_pos, const std::vector<U64> &game_hashes,
                PendingRollout &pending, std::future<NNOutput> &out_future,
                float &out_value, int nscl = 0);

// Expand the leaf's children from the NN result and backpropagate.
void mcts_expand_and_backprop(TreeArena &arena, PendingRollout &pending,
                              const NNOutput &raw_nn, int nscl = 0);

// Propagate a value (terminal or cached) back along search_path.
void mcts_backprop(TreeArena &arena, const std::vector<U32> &path, float value,
                   int nscl = 0);
