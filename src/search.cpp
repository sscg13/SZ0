#include "search.h"
#include "node.h"
#include "position.h"

#include <chrono>
#include <sstream>
#include <thread>

std::atomic<bool> stop_search(false);
std::atomic<U64> total_rollouts(0);
std::atomic<int> seldepth(0);
std::atomic<U64> depthsum(0);

template <typename T> void atomic_fetch_max(std::atomic<T> &obj, T val) {
  T prev = obj.load(std::memory_order_relaxed);
  while (prev < val && !obj.compare_exchange_weak(prev, val)) {
  }
}

void printinfostring(const TreeArena &arena, int timetaken, int avgdepth,
                     int seldepth) {
  U32 current_idx = 0;
  std::stringstream pv;
  int nodecount = arena.nodes[0].visits;
  int score = 1000 * (arena.nodes[0].value_sum / nodecount);
  while (arena.nodes[current_idx].num_children > 0) {
    U32 best_child = 0;
    I32 max_visits = -1;

    for (U16 i = 0; i < arena.nodes[current_idx].num_children; ++i) {
      U32 child_idx = arena.nodes[current_idx].first_child_idx + i;
      if (arena.nodes[child_idx].visits > max_visits) {
        max_visits = arena.nodes[child_idx].visits;
        best_child = child_idx;
      }
    }

    if (max_visits == 0)
      break;
    pv << algebraic(arena.nodes[best_child].move) << " ";

    current_idx = best_child;
  }
  std::cout << "info depth " << avgdepth << " seldepth " << seldepth << " time "
            << timetaken << " score cp " << score << " nodes " << nodecount
            << " pv " << pv.str() << "\n";
}

void mcts_worker(TreeArena &arena, Position root_pos,
                 std::vector<uint64_t> game_history) {
  while (!stop_search.load(std::memory_order_relaxed)) {
    int depth = mcts_rollout(arena, root_pos, game_history);
    total_rollouts.fetch_add(1, std::memory_order_relaxed);
    if (arena.active_nodes.load(std::memory_order_relaxed) >=
        arena.max_nodes - 256) {
      stop_search.store(true, std::memory_order_relaxed);
    }
    atomic_fetch_max(seldepth, depth);
    depthsum.fetch_add(depth, std::memory_order_relaxed);
  }
}

Move get_best_move(TreeArena &arena) {
  const Node &root_node = arena.nodes[0];
  uint32_t best_child_idx = 0;
  int32_t max_visits = -1;

  for (uint16_t i = 0; i < root_node.num_children; ++i) {
    uint32_t child_idx = root_node.first_child_idx + i;
    if (arena.nodes[child_idx].visits > max_visits) {
      max_visits = arena.nodes[child_idx].visits;
      best_child_idx = child_idx;
    }
  }

  return arena.nodes[best_child_idx].move;
}

void search_position(TreeArena &arena, const Position &current_pos,
                     const std::vector<uint64_t> &game_hashes, int timelimit,
                     U64 nodelimit, int threadcount, bool print_info) {
  stop_search.store(false, std::memory_order_relaxed);
  total_rollouts.store(0, std::memory_order_relaxed);
  seldepth.store(0, std::memory_order_relaxed);
  depthsum.store(0, std::memory_order_relaxed);
  arena.clear();

  auto start = std::chrono::steady_clock::now();
  auto last_info = start;

  std::vector<std::thread> threads;
  for (int i = 0; i < threadcount; ++i) {
    threads.emplace_back(mcts_worker, std::ref(arena), current_pos,
                         game_hashes);
  }
  while (!stop_search.load(std::memory_order_relaxed)) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    auto now = std::chrono::steady_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - start)
            .count();
    if (elapsed >= timelimit && timelimit > 0) {
      stop_search.store(true, std::memory_order_relaxed);
      break;
    }

    U64 current_nodes = total_rollouts.load(std::memory_order_relaxed);
    if (current_nodes >= nodelimit && nodelimit > 0) {
      stop_search.store(true, std::memory_order_relaxed);
      break;
    }

    auto time_since_info =
        std::chrono::duration_cast<std::chrono::milliseconds>(now - last_info)
            .count();
    if (time_since_info >= 400 && print_info) {

      printinfostring(arena, elapsed,
                      depthsum.load(std::memory_order_relaxed) /
                          total_rollouts.load(std::memory_order_relaxed),
                      seldepth.load(std::memory_order_relaxed));
      last_info = now;
    }
  }

  for (auto &t : threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  auto now = std::chrono::steady_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(now - start)
          .count();
  if (print_info) {
  printinfostring(arena, elapsed,
                  depthsum.load(std::memory_order_relaxed) /
                      total_rollouts.load(std::memory_order_relaxed),
                  seldepth.load(std::memory_order_relaxed));
  }
  Move best = get_best_move(arena);
  if (best == Move()) {
    Move moves[maxmoves];
    Position copy_pos = current_pos;
    copy_pos.generatemoves(moves);
    best = moves[0];
  }
  if (print_info) {
    std::cout << "bestmove " << algebraic(best) << "\n";
  }
}