#include "node.h"
#include "position.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

U32 TreeArena::allocate_children(U16 count) {
  U32 start_idx = static_cast<U32>(nodes.size());
  nodes.resize(nodes.size() + count);
  return start_idx;
}

void TreeArena::clear() {
  nodes.clear();
  nodes.emplace_back();
  active_nodes = 0;
}

void TreeArena::resize(int megabytes) {
  size_t total_bytes = static_cast<size_t>(megabytes) * 1024ULL * 1024ULL;
  size_t new_max_nodes = total_bytes / sizeof(Node);
  nodes.resize(new_max_nodes);
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
  float sqrt_parent_visits =
      std::sqrt(static_cast<float>(std::max(1, parent.visits)));

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
  if (arena.nodes[current_idx].visits > 0) {
    // If this leaf already has multiple visits, it must be terminal
    value = arena.nodes[current_idx].value_sum /
            static_cast<float>(arena.nodes[current_idx].visits);
  } else {
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
        uint32_t child_start = arena.allocate_children(movecount);
        arena.nodes[current_idx].first_child_idx = child_start;
        arena.nodes[current_idx].num_children = movecount;

        float uniform_prior = 1.0f / movecount;
        for (int i = 0; i < movecount; ++i) {
          arena.nodes[child_start + i].move = moves[i];
          arena.nodes[child_start + i].prior = uniform_prior;
          arena.active_nodes++;
        }

        // Get the dummy evaluation (soon to be neural network)
        value = materialist(current_pos);
      }
    }
  }

  // BACKPROPAGATION
  for (int i = search_path.size() - 1; i >= 0; --i) {
    U32 idx = search_path[i];
    arena.nodes[idx].visits += 1;
    arena.nodes[idx].value_sum += value;
    value = -value;
  }
  return depth;
}