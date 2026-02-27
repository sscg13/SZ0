#include "node.h"
#include "position.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

void TreeArena::clear() {
  nodes[0].visits.store(0, std::memory_order_relaxed);
  nodes[0].value_sum.store(0.0f, std::memory_order_relaxed);
  nodes[0].is_expanding.clear(std::memory_order_release);
  nodes[0].first_child_idx = -1;
  nodes[0].num_children = 0;
  active_nodes.store(1, std::memory_order_relaxed);
}

void TreeArena::resize(int megabytes) {
  size_t total_bytes = static_cast<size_t>(megabytes) * 1024ULL * 1024ULL;
  size_t new_max_nodes = total_bytes / sizeof(Node);
  nodes = std::make_unique<Node[]>(new_max_nodes);
  max_nodes = new_max_nodes;
}

float dummy_evaluate(const Position &pos) {
  thread_local std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  return dist(rng);
}

float materialist(const Position &pos) {
  int material = pos.material();
  return std::tanh(static_cast<float>(material) / 400.0f);
}

U32 select_best_puct(const TreeArena &arena, U32 node_idx) {
  const Node &parent = arena.nodes[node_idx];
  const float c_puct = 2.0f;
  float sqrt_parent_visits = std::sqrt(static_cast<float>(
      std::max(1, parent.visits.load(std::memory_order_relaxed))));

  U32 best_idx = parent.first_child_idx;
  float best_score = -std::numeric_limits<float>::infinity();

  for (U16 i = 0; i < parent.num_children; ++i) {
    U32 child_idx = parent.first_child_idx + i;
    const Node &child = arena.nodes[child_idx];

    float q_value = (child.visits > 0)
                        ? (-child.value_sum / static_cast<float>(child.visits))
                        : 0.0f;
    float u_value =
        c_puct * child.prior * (sqrt_parent_visits / (1.0f + child.visits));
    float score = q_value + u_value;

    if (score > best_score) {
      best_score = score;
      best_idx = child_idx;
    }
  }

  return best_idx;
}

bool is_repetition(const Position &pos, const std::vector<U64> &game_hashes,
                   const std::vector<U64> &rollout_hashes) {
  int halfmoves = pos.halfmovecount;
  if (halfmoves < 4) {
    return false;
  }
  uint64_t target_hash = pos.zobristhash;
  int lookback = 4;
  int appearance_count = 1;

  int hist_idx = (int)rollout_hashes.size() - lookback - 1;
  while (lookback <= halfmoves && hist_idx >= 0) {
    if (rollout_hashes[hist_idx] == target_hash) {
      return true;
    }
    lookback += 2;
    hist_idx -= 2;
  }

  hist_idx =
      (int)game_hashes.size() + (int)rollout_hashes.size() - lookback - 1;
  while (lookback <= halfmoves && hist_idx >= 0) {
    if (game_hashes[hist_idx] == target_hash) {
      appearance_count++;
      if (appearance_count >= 3) {
        return true;
      }
    }
    lookback += 2;
    hist_idx -= 2;
  }

  return false;
}

int mcts_rollout(TreeArena &arena, const Position &root_pos,
                 const std::vector<U64> &game_hashes) {
  Position current_pos = root_pos;
  U32 current_idx = 0;
  std::vector<U32> search_path;
  std::vector<uint64_t> rollout_hashes;
  rollout_hashes.reserve(256);
  search_path.reserve(256);
  search_path.push_back(current_idx);
  rollout_hashes.push_back(current_pos.zobristhash);
  int depth = 0;

  // SELECTION
  while (arena.nodes[current_idx].num_children > 0) {
    U32 best_child_idx = select_best_puct(arena, current_idx);
    current_pos.makemove(arena.nodes[best_child_idx].move);
    current_idx = best_child_idx;
    search_path.push_back(current_idx);
    rollout_hashes.push_back(current_pos.zobristhash);
    depth++;
  }

  // EXPANSION
  float value;
  if (arena.nodes[current_idx].visits.load(std::memory_order_relaxed) > 0) {
    float v_sum =
        arena.nodes[current_idx].value_sum.load(std::memory_order_relaxed);
    int vis = arena.nodes[current_idx].visits.load(std::memory_order_relaxed);
    value = v_sum / static_cast<float>(vis);
  } else {
    if (!arena.nodes[current_idx].is_expanding.test_and_set(
            std::memory_order_acquire)) {
      if (current_pos.twokings()) {
        value = 0.0f;
      } else if (current_pos.bareking(!current_pos.stm)) {
        value = 1.0f;
      } else if (current_pos.halfmovecount >= 140) {
        value = 0.0f;
      } else if (is_repetition(current_pos, game_hashes, rollout_hashes)) {
        value = 0.0f;
      } else {
        Move moves[maxmoves];
        int movecount = current_pos.generatemoves(moves);
        if (movecount == 0) {
          value = -1.0f;
        } else {
          U32 child_start = arena.active_nodes.fetch_add(
              movecount, std::memory_order_relaxed);

          if (child_start + movecount < arena.max_nodes) {
            arena.nodes[current_idx].first_child_idx = child_start;
            arena.nodes[current_idx].num_children = movecount;

            float uniform_prior = 1.0f / movecount;
            for (int i = 0; i < movecount; ++i) {
              arena.nodes[child_start + i].move = moves[i];
              arena.nodes[child_start + i].prior = uniform_prior;
              arena.active_nodes++;
            }
          }

          // Get the dummy evaluation (soon to be neural network)
          value = materialist(current_pos);
        }
      }
    } else {
      value = materialist(current_pos);
    }
  }

  // BACKPROPAGATION
  for (int i = search_path.size() - 1; i >= 0; --i) {
    U32 idx = search_path[i];
    arena.nodes[idx].visits.fetch_add(1, std::memory_order_relaxed);
    arena.nodes[idx].value_sum.fetch_add(value, std::memory_order_relaxed);
    value = -value;
  }
  return depth;
}