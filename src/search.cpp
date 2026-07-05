#include "search.h"
#include "node.h"
#include "position.h"

#include <chrono>
#include <future>
#include <sstream>
#include <thread>

// The batched (in-flight rollout) pipeline is required for CUDA but can also
// be built for CPU inference (make BATCH=yes) to emulate GPU-level batching
// in local tests.
#if defined(USE_CUDA) || defined(USE_BATCHED_SEARCH)
#define BATCHED_SEARCH
#endif

std::atomic<bool> stop_search(false);
std::atomic<U64> total_rollouts(0);
// Rollout tickets claimed before each rollout starts. Enforcing the node
// limit here instead of in the 1ms monitor loop makes "go nodes N" exact:
// exactly N rollouts are started (collisions return their ticket), so the
// limit does not drift with search speed.
std::atomic<U64> started_rollouts(0);
std::atomic<int> seldepth(0);
std::atomic<U64> depthsum(0);
// Rollouts discarded because another thread was already expanding the same
// leaf. Diagnostic for comparing virtual-visit vs particle selection.
std::atomic<U64> leaf_collisions(0);

template <typename T> void atomic_fetch_max(std::atomic<T> &obj, T val) {
  T prev = obj.load(std::memory_order_relaxed);
  while (prev < val && !obj.compare_exchange_weak(prev, val)) {
  }
}

int scoretocp(float q) {
  float clamped_q = std::clamp(q, -0.9999f, 0.9999f);
  float cp = 182 * std::atanh(clamped_q);

  return static_cast<int>(std::round(cp));
}

void printinfostring(const TreeArena &arena, int timetaken, int avgdepth,
                     int seldepth) {
  U32 current_idx = 0;
  std::stringstream pv;
  float nodecount = arena.nodes[0].visits;
  int score = scoretocp(arena.nodes[0].value_sum / nodecount);
  while (arena.nodes[current_idx].num_children > 0) {
    U32 best_child = 0;
    float max_visits = -1.0f;

    for (U16 i = 0; i < arena.nodes[current_idx].num_children; ++i) {
      U32 child_idx = arena.nodes[current_idx].first_child_idx + i;
      if (arena.nodes[child_idx].visits > max_visits) {
        max_visits = arena.nodes[child_idx].visits;
        best_child = child_idx;
      }
    }

    if (max_visits == 0.0f)
      break;
    pv << algebraic(arena.nodes[best_child].move) << " ";

    current_idx = best_child;
  }
  std::cout << "info depth " << avgdepth << " seldepth " << seldepth << " time "
            << timetaken << " score cp " << score << " nodes "
            << static_cast<U64>(nodecount) << " pv " << pv.str() << "\n";
}

#ifndef BATCHED_SEARCH
void mcts_worker(NNEvaluator &nn, TreeArena &arena, Position root_pos,
                 std::vector<uint64_t> game_history, int nscl, float eta,
                 U64 nodelimit) {
  bool collided = false;
  while (!stop_search.load(std::memory_order_relaxed)) {
    if (nodelimit > 0 &&
        started_rollouts.fetch_add(1, std::memory_order_relaxed) >=
            nodelimit) {
      stop_search.store(true, std::memory_order_relaxed);
      break;
    }
    // Deterministic selection retries the identical path after a collision
    // (its virtual visits were rolled back), so break the tie with one
    // sampled (eta = 1) rollout before returning to deterministic mode.
    float sel_eta = (eta < 0.0f && collided) ? 1.0f : eta;
    int depth = mcts_rollout(nn, arena, root_pos, game_history, nscl, sel_eta);
    collided = (depth == 0);
    if (depth == 0) {
      if (nodelimit > 0)
        started_rollouts.fetch_sub(1, std::memory_order_relaxed);
      leaf_collisions.fetch_add(1, std::memory_order_relaxed);
      std::this_thread::yield();
      continue;
    }
    total_rollouts.fetch_add(1, std::memory_order_relaxed);
    if (arena.active_nodes.load(std::memory_order_relaxed) >=
        arena.max_nodes - 256) {
      stop_search.store(true, std::memory_order_relaxed);
    }
    atomic_fetch_max(seldepth, depth);
    depthsum.fetch_add(depth, std::memory_order_relaxed);
  }
}
#else
void mcts_worker(BatchEvaluator &batch_eval, TreeArena &arena,
                 Position root_pos, std::vector<uint64_t> game_history,
                 int max_in_flight, int nscl, float eta, U64 nodelimit) {
  struct InFlight {
    PendingRollout pending;
    std::future<NNOutput> future;
  };

  std::vector<InFlight> in_flight;
  in_flight.reserve(max_in_flight);
  bool collided = false;

  auto finish_rollout = [&](PendingRollout &pending) {
    int depth = pending.depth;
    total_rollouts.fetch_add(1, std::memory_order_relaxed);
    if (arena.active_nodes.load(std::memory_order_relaxed) >=
        arena.max_nodes - 256) {
      stop_search.store(true, std::memory_order_relaxed);
    }
    atomic_fetch_max(seldepth, depth);
    depthsum.fetch_add(depth, std::memory_order_relaxed);
  };

  while (!stop_search.load(std::memory_order_relaxed)) {
    // Complete any in-flight rollouts whose futures are ready.
    for (auto it = in_flight.begin(); it != in_flight.end();) {
      if (it->future.wait_for(std::chrono::seconds(0)) ==
          std::future_status::ready) {
        NNOutput raw_nn = it->future.get();
        mcts_expand_and_backprop(arena, it->pending, raw_nn, nscl);
        finish_rollout(it->pending);
        it = in_flight.erase(it);
      } else {
        ++it;
      }
    }

    // Start a new rollout if below the in-flight limit and a rollout ticket
    // is still available under the node limit.
    bool can_start = static_cast<int>(in_flight.size()) < max_in_flight;
    if (can_start && nodelimit > 0) {
      if (started_rollouts.fetch_add(1, std::memory_order_relaxed) >=
          nodelimit) {
        started_rollouts.fetch_sub(1, std::memory_order_relaxed);
        can_start = false;
        if (in_flight.empty()) {
          // Budget exhausted and nothing left to finish.
          stop_search.store(true, std::memory_order_relaxed);
          break;
        }
      }
    }

    if (can_start) {
      PendingRollout pending;
      std::future<NNOutput> future;
      float value;

      // After a collision, deterministic selection would retry the same
      // path (its virtual visits were rolled back); break the tie with one
      // sampled (eta = 1) selection before returning to deterministic mode.
      float sel_eta = (eta < 0.0f && collided) ? 1.0f : eta;
      int outcome = mcts_select(batch_eval, arena, root_pos, game_history,
                                pending, future, value, nscl, sel_eta);
      collided = (outcome == 0);

      if (outcome == 2) {
        in_flight.push_back({std::move(pending), std::move(future)});
      } else if (outcome == 1) {
        mcts_backprop(arena, pending.search_path, value, nscl, pending.weight);
        finish_rollout(pending);
      } else {
        // outcome == 0 (collision): discard and retry next iteration.
        if (nodelimit > 0)
          started_rollouts.fetch_sub(1, std::memory_order_relaxed);
        leaf_collisions.fetch_add(1, std::memory_order_relaxed);
      }
    } else {
      // All slots are full (or the node budget is spent) and no future is
      // ready yet; yield briefly to avoid spinning while the GPU processes
      // the batch.
      std::this_thread::yield();
    }
  }

  // Drain any in-flight rollouts so virtual visits are removed cleanly.
  // These count toward the node budget: their tickets were claimed, so the
  // completed-rollout total still matches the requested limit exactly.
  for (auto &item : in_flight) {
    NNOutput raw_nn = item.future.get();
    mcts_expand_and_backprop(arena, item.pending, raw_nn, nscl);
    finish_rollout(item.pending);
  }
}
#endif

Move get_best_move(TreeArena &arena) {
  const Node &root_node = arena.nodes[0];
  uint32_t best_child_idx = 0;
  float max_visits = -1.0f;

  for (uint16_t i = 0; i < root_node.num_children; ++i) {
    uint32_t child_idx = root_node.first_child_idx + i;
    if (arena.nodes[child_idx].visits > max_visits) {
      max_visits = arena.nodes[child_idx].visits;
      best_child_idx = child_idx;
    }
  }

  return arena.nodes[best_child_idx].move;
}

void search_position(NNEvaluator &nn, TreeArena &arena,
                     const Position &current_pos,
                     const std::vector<uint64_t> &game_hashes, int timelimit,
                     U64 nodelimit, int threadcount, bool print_info,
                     int contempt_nscl, float particle_eta) {
  stop_search.store(false, std::memory_order_relaxed);
  total_rollouts.store(0, std::memory_order_relaxed);
  started_rollouts.store(0, std::memory_order_relaxed);
  seldepth.store(0, std::memory_order_relaxed);
  depthsum.store(0, std::memory_order_relaxed);
  leaf_collisions.store(0, std::memory_order_relaxed);
  arena.clear();

  // The interaction between particle selection and search-contempt Thompson
  // Sampling is unexplored; contempt takes precedence for now.
  if (contempt_nscl > 0) {
    particle_eta = 0.0f;
  }

#ifdef BATCHED_SEARCH
  // Each worker keeps ceil(searchbatchsize / threadcount) rollouts in flight
  // so that across all workers the batch fills before the inference thread
  // fires, maximising GPU utilisation.
  int max_in_flight = (searchbatchsize + threadcount - 1) / threadcount;
  BatchEvaluator batch_eval(nn, searchbatchsize);
  std::thread infer_thread([&batch_eval] { batch_eval.run_inference_loop(); });
#endif

  auto start = std::chrono::steady_clock::now();
  auto last_info = start;

  std::vector<std::thread> threads;
  for (int i = 0; i < threadcount; ++i) {
#ifdef BATCHED_SEARCH
    threads.emplace_back(mcts_worker, std::ref(batch_eval), std::ref(arena),
                         current_pos, game_hashes, max_in_flight,
                         contempt_nscl, particle_eta, nodelimit);
#else
    threads.emplace_back(mcts_worker, std::ref(nn), std::ref(arena),
                         current_pos, game_hashes, contempt_nscl,
                         particle_eta, nodelimit);
#endif
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
    if (time_since_info >= 400 && print_info && current_nodes > 0) {

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

#ifdef BATCHED_SEARCH
  // All workers are done — no more submit() calls can arrive.
  batch_eval.stop();
  infer_thread.join();
#endif

  auto now = std::chrono::steady_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(now - start)
          .count();
  if (print_info) {
    printinfostring(arena, elapsed,
                    depthsum.load(std::memory_order_relaxed) /
                        total_rollouts.load(std::memory_order_relaxed),
                    seldepth.load(std::memory_order_relaxed));
    std::cout << "info string rollouts "
              << total_rollouts.load(std::memory_order_relaxed)
              << " collisions "
              << leaf_collisions.load(std::memory_order_relaxed) << "\n";
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
