#include "position.h"

#include <cstdint>
#include <random>
#include <vector>

struct alignas(8) Node {
  U32 first_child_idx;
  I32 visits;
  float value_sum;
  float prior;

  Move move;
  U16 num_children;

  Node()
      : move(Move()), num_children(0), first_child_idx(0), visits(0),
        value_sum(0.0f), prior(0.0f) {}
};

struct TreeArena {
  std::vector<Node> nodes;
  size_t active_nodes;
  size_t max_nodes;

  TreeArena(size_t initial_capacity = 3145728) {
    nodes.reserve(initial_capacity);
    nodes.emplace_back();
    max_nodes = initial_capacity;
  }

  U32 allocate_children(U16 count);
  void resize(int megabytes);
  void clear();
};

float dummy_evaluate(const Position &pos);
U32 select_best_puct(const TreeArena &arena, U32 node_idx);
int mcts_rollout(TreeArena &arena, const Position &root_pos,
                 const std::vector<U64> &game_hashes);