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
  nodes[0].frozen_visits.store(0, std::memory_order_relaxed);
  active_nodes.store(1, std::memory_order_relaxed);
}

void TreeArena::resize(int megabytes) {
  size_t total_bytes = static_cast<size_t>(megabytes) * 1024ULL * 1024ULL;
  size_t new_max_nodes = total_bytes / sizeof(Node);
  nodes = std::make_unique<Node[]>(new_max_nodes);
  max_nodes = new_max_nodes;
}

U32 select_best_puct(const TreeArena &arena, U32 node_idx) {
  const Node &parent = arena.nodes[node_idx];
  const float c_puct = 2.0f;

  int parent_visits = parent.visits.load(std::memory_order_relaxed);
  float sqrt_parent_visits =
      std::sqrt(static_cast<float>(std::max(1, parent_visits)));

  float parent_q = (parent_visits > 0)
                       ? (parent.value_sum / static_cast<float>(parent_visits))
                       : 0.0f;

  U32 best_idx = parent.first_child_idx;
  float best_score = -std::numeric_limits<float>::infinity();

  for (U8 i = 0; i < parent.num_children; ++i) {
    U32 child_idx = parent.first_child_idx + i;
    const Node &child = arena.nodes[child_idx];
    int real_visits = child.visits.load(std::memory_order_relaxed);
    int virtual_visits = child.virtual_visits.load(std::memory_order_relaxed);
    int effective_visits = real_visits + virtual_visits;
    float q_value = (real_visits > 0)
                        ? (-child.value_sum / static_cast<float>(real_visits))
                        : parent_q;
    float u_value =
        c_puct * child.prior * (sqrt_parent_visits / (1.0f + effective_visits));
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

// Sample a child proportionally to frozen_visits (Thompson Sampling).
// Uses a thread-local RNG so it is safe to call from multiple worker threads.
static U32 sample_frozen_child(const TreeArena &arena, U32 node_idx) {
  thread_local std::mt19937 rng(std::random_device{}());
  const Node &node = arena.nodes[node_idx];
  int total = 0;
  for (U8 c = 0; c < node.num_children; ++c)
    total += arena.nodes[node.first_child_idx + c].frozen_visits.load(
        std::memory_order_relaxed);
  if (total == 0) {
    std::uniform_int_distribution<int> dist(0, node.num_children - 1);
    return node.first_child_idx + dist(rng);
  }
  std::uniform_int_distribution<int> dist(0, total - 1);
  int r = dist(rng);
  int cumulative = 0;
  for (U8 c = 0; c < node.num_children; ++c) {
    cumulative += arena.nodes[node.first_child_idx + c].frozen_visits.load(
        std::memory_order_relaxed);
    if (r < cumulative)
      return node.first_child_idx + c;
  }
  return node.first_child_idx + node.num_children - 1;
}

// Snapshot each child's current visit count into frozen_visits.
static void freeze_children(TreeArena &arena, U32 node_idx) {
  const Node &node = arena.nodes[node_idx];
  for (U8 c = 0; c < node.num_children; ++c) {
    Node &child = arena.nodes[node.first_child_idx + c];
    int cv = child.visits.load(std::memory_order_relaxed);
    child.frozen_visits.store(static_cast<U8>(cv), std::memory_order_relaxed);
  }
}

int mcts_rollout(NNEvaluator &nn, TreeArena &arena, const Position &root_pos,
                 const std::vector<U64> &game_hashes, int nscl) {
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
    U32 best_child_idx;
    if (nscl > 0 && depth % 2 == 1 &&
        arena.nodes[current_idx].visits.load(std::memory_order_relaxed) >
            nscl) {
      best_child_idx = sample_frozen_child(arena, current_idx);
    } else {
      best_child_idx = select_best_puct(arena, current_idx);
    }
    current_pos.makemove(arena.nodes[best_child_idx].move);
    current_idx = best_child_idx;
    search_path.push_back(current_idx);
    rollout_hashes.push_back(current_pos.zobristhash);
    arena.nodes[current_idx].virtual_visits.fetch_add(
        1, std::memory_order_relaxed);
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
          NNOutput raw_nn = nn.infer(current_pos);
          MCTSEval processed =
              parse_nn_output(raw_nn, moves, movecount, current_pos.stm);
          value = processed.qscore;

          U32 child_start = arena.active_nodes.fetch_add(
              movecount, std::memory_order_relaxed);

          if (child_start + movecount < arena.max_nodes) {
            for (int i = 0; i < movecount; i++) {
              size_t child_idx = child_start + i;
              arena.nodes[child_idx].move = moves[i];
              arena.nodes[child_idx].prior = processed.priors[i];

              arena.nodes[child_idx].visits.store(0, std::memory_order_relaxed);
              arena.nodes[child_idx].value_sum.store(0.0f,
                                                     std::memory_order_relaxed);
              arena.nodes[child_idx].first_child_idx = -1;
              arena.nodes[child_idx].num_children = 0;
              arena.nodes[child_idx].is_expanding.clear(
                  std::memory_order_relaxed);
              arena.nodes[child_idx].virtual_visits.store(
                  0, std::memory_order_relaxed);
              arena.nodes[child_idx].frozen_visits.store(
                  0, std::memory_order_relaxed);
            }

            std::atomic_thread_fence(std::memory_order_release);
            arena.nodes[current_idx].first_child_idx = child_start;
            arena.nodes[current_idx].num_children = movecount;
          }
        }
      }
    } else {
      for (int i = search_path.size() - 1; i >= 0; --i) {
        U32 idx = search_path[i];
        if (i > 0) {
          arena.nodes[idx].virtual_visits.fetch_sub(1,
                                                    std::memory_order_relaxed);
        }
      }
      return 0;
    }
  }

  // BACKPROPAGATION
  for (int i = static_cast<int>(search_path.size()) - 1; i >= 0; --i) {
    U32 idx = search_path[i];
    if (i > 0) {
      arena.nodes[idx].virtual_visits.fetch_sub(1, std::memory_order_relaxed);
    }
    int child_visits =
        arena.nodes[idx].visits.fetch_add(1, std::memory_order_relaxed);
    arena.nodes[idx].value_sum.fetch_add(value, std::memory_order_relaxed);
    if (nscl > 0 && i % 2 == 1 && child_visits == nscl &&
        arena.nodes[idx].num_children > 0) {
      freeze_children(arena, idx);
    }
    value = -value;
  }
  return depth;
}

void mcts_backprop(TreeArena &arena, const std::vector<U32> &path, float value,
                   int nscl) {
  for (int i = static_cast<int>(path.size()) - 1; i >= 0; --i) {
    U32 idx = path[i];
    if (i > 0)
      arena.nodes[idx].virtual_visits.fetch_sub(1, std::memory_order_relaxed);
    int child_visits =
        arena.nodes[idx].visits.fetch_add(1, std::memory_order_relaxed);
    arena.nodes[idx].value_sum.fetch_add(value, std::memory_order_relaxed);
    if (nscl > 0 && i % 2 == 1 && child_visits == nscl &&
        arena.nodes[idx].num_children > 0) {
      freeze_children(arena, idx);
    }
    value = -value;
  }
}

int mcts_select(BatchEvaluator &batch_eval, TreeArena &arena,
                const Position &root_pos, const std::vector<U64> &game_hashes,
                PendingRollout &pending, std::future<NNOutput> &out_future,
                float &out_value, int nscl) {
  Position current_pos = root_pos;
  U32 current_idx = 0;
  pending.search_path.clear();
  pending.rollout_hashes.clear();
  pending.search_path.reserve(64);
  pending.rollout_hashes.reserve(64);
  pending.search_path.push_back(0);
  pending.rollout_hashes.push_back(current_pos.zobristhash);
  pending.depth = 0;

  // SELECTION
  while (arena.nodes[current_idx].num_children > 0) {
    U32 best_child;
    if (nscl > 0 && pending.depth % 2 == 1 &&
        arena.nodes[current_idx].visits.load(std::memory_order_relaxed) >
            nscl) {
      best_child = sample_frozen_child(arena, current_idx);
    } else {
      best_child = select_best_puct(arena, current_idx);
    }
    current_pos.makemove(arena.nodes[best_child].move);
    current_idx = best_child;
    pending.search_path.push_back(current_idx);
    pending.rollout_hashes.push_back(current_pos.zobristhash);
    arena.nodes[current_idx].virtual_visits.fetch_add(
        1, std::memory_order_relaxed);
    pending.depth++;
  }

  Node &leaf = arena.nodes[current_idx];

  // Already evaluated (e.g. a previously-seen terminal hit by a race).
  if (leaf.visits.load(std::memory_order_relaxed) > 0) {
    float v_sum = leaf.value_sum.load(std::memory_order_relaxed);
    int vis = leaf.visits.load(std::memory_order_relaxed);
    out_value = v_sum / static_cast<float>(vis);
    return 1;
  }

  // Try to claim this leaf for expansion.
  if (!leaf.is_expanding.test_and_set(std::memory_order_acquire)) {
    bool stm = current_pos.stm;

    if (current_pos.twokings()) {
      out_value = 0.0f;
      return 1;
    }
    if (current_pos.bareking(!stm)) {
      out_value = 1.0f;
      return 1;
    }
    if (current_pos.halfmovecount >= 140) {
      out_value = 0.0f;
      return 1;
    }
    if (is_repetition(current_pos, game_hashes, pending.rollout_hashes)) {
      out_value = 0.0f;
      return 1;
    }

    pending.movecount = current_pos.generatemoves(pending.moves);
    if (pending.movecount == 0) {
      out_value = -1.0f;
      return 1;
    }

    // Non-terminal: submit to the batch evaluator and return immediately.
    pending.leaf_pos = current_pos;
    out_future = batch_eval.submit(current_pos);
    return 2;
  }

  // Collision: another thread is expanding this leaf. Clean up virtual visits.
  for (int i = static_cast<int>(pending.search_path.size()) - 1; i > 0; --i)
    arena.nodes[pending.search_path[i]].virtual_visits.fetch_sub(
        1, std::memory_order_relaxed);
  return 0;
}

void mcts_expand_and_backprop(TreeArena &arena, PendingRollout &pending,
                              const NNOutput &raw_nn, int nscl) {
  MCTSEval processed = parse_nn_output(raw_nn, pending.moves, pending.movecount,
                                       pending.leaf_pos.stm);
  float value = processed.qscore;
  U32 leaf_idx = pending.search_path.back();

  U32 child_start = arena.active_nodes.fetch_add(pending.movecount,
                                                 std::memory_order_relaxed);

  if (child_start + pending.movecount < arena.max_nodes) {
    for (int i = 0; i < pending.movecount; ++i) {
      size_t child_idx = child_start + i;
      arena.nodes[child_idx].move = pending.moves[i];
      arena.nodes[child_idx].prior = processed.priors[i];
      arena.nodes[child_idx].visits.store(0, std::memory_order_relaxed);
      arena.nodes[child_idx].value_sum.store(0.0f, std::memory_order_relaxed);
      arena.nodes[child_idx].first_child_idx = -1;
      arena.nodes[child_idx].num_children = 0;
      arena.nodes[child_idx].is_expanding.clear(std::memory_order_relaxed);
      arena.nodes[child_idx].virtual_visits.store(0, std::memory_order_relaxed);
      arena.nodes[child_idx].frozen_visits.store(0, std::memory_order_relaxed);
    }
    std::atomic_thread_fence(std::memory_order_release);
    arena.nodes[leaf_idx].first_child_idx = child_start;
    arena.nodes[leaf_idx].num_children = pending.movecount;
  }

  mcts_backprop(arena, pending.search_path, value, nscl);
}
