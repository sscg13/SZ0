#include "datagen.h"
#include "node.h"
#include "search.h"

#include <atomic>
#include <barrier>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>

int save_game_to_binary(const std::vector<TrainingPosition> &game_history,
                        std::ofstream &out) {
  if (!out) {
    std::cerr << "Error: Could not open pipe for writing.\n";
    return 0;
  }

  for (const auto &pos : game_history) {
    // metadata and value data
    out.write(reinterpret_cast<const char *>(&pos.root_q), sizeof(pos.root_q));
    out.write(reinterpret_cast<const char *>(&pos.num_moves),
              sizeof(pos.num_moves));
    out.write(reinterpret_cast<const char *>(&pos.halfmove_clock),
              sizeof(pos.halfmove_clock));
    out.write(reinterpret_cast<const char *>(&pos.outcome),
              sizeof(pos.outcome));

    // board
    out.write(reinterpret_cast<const char *>(pos.board_tokens),
              sizeof(pos.board_tokens));

    // policy data
    if (pos.num_moves > 0) {
      out.write(reinterpret_cast<const char *>(pos.move_indices),
                pos.num_moves * sizeof(uint16_t));
      out.write(reinterpret_cast<const char *>(pos.move_probabilities),
                pos.num_moves * sizeof(float));
    }
  }

  return (int)game_history.size();
}

void get_canonical_tokens(const Position &pos, uint8_t tokens[64]) {
  for (int sq = 0; sq < 64; ++sq) {
    int perspective_sq = pos.stm ? (sq ^ 56) : sq;

    Piece p = pos.pieces[sq];
    if (p.empty()) {
      tokens[perspective_sq] = 0;
    } else {
      bool is_friendly = (p.color() == pos.stm);
      tokens[perspective_sq] = p.type() + (is_friendly ? 0 : 6) - 1;
    }
  }
}

bool is_repetition(const Position &pos, const std::vector<U64> &game_hashes) {
  int halfmoves = pos.halfmovecount;
  if (halfmoves < 4) {
    return false;
  }
  uint64_t target_hash = pos.zobristhash;
  int lookback = 4;
  int appearance_count = 1;

  int hist_idx = (int)game_hashes.size() - lookback;
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

void backprop(DatagenGame &game, float value) {
  for (int i = static_cast<int>(game.search_path.size()) - 1; i >= 0; --i) {
    U32 idx = game.search_path[i];
    Node &node = game.arena.nodes[idx];
    float child_visits =
        node.visits.fetch_add(1.0f, std::memory_order_relaxed);
    node.value_sum.fetch_add(value, std::memory_order_relaxed);

    // When an odd-depth (opponent) node first reaches N_scl visits, freeze each
    // child's current visit count for subsequent Thompson Sampling selection.
    // Datagen backups always add exactly 1, so counts stay integral.
    if (i % 2 == 1 && static_cast<int>(child_visits) == contempt_nscl &&
        node.num_children > 0) {
      for (U8 c = 0; c < node.num_children; ++c) {
        Node &child = game.arena.nodes[node.first_child_idx + c];
        int cv = static_cast<int>(child.visits.load(std::memory_order_relaxed));
        child.frozen_visits.store(static_cast<U8>(cv), std::memory_order_relaxed);
      }
    }

    value = -value;
  }
}

// Sample a child node proportionally to frozen_visits (Thompson Sampling).
static U32 sample_frozen_child(const TreeArena &arena, U32 node_idx,
                                std::mt19937 &rng) {
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

bool select(DatagenGame &game, std::mt19937 &rng) {
  game.search_path.clear();
  game.leaf_pos = game.root_pos;
  U32 current_idx = 0;
  game.search_path.push_back(current_idx);

  std::vector<uint64_t> rollout_hashes;
  rollout_hashes.push_back(game.leaf_pos.zobristhash);

  // SELECTION
  int depth = 0;
  while (game.arena.nodes[current_idx].num_children > 0) {
    U32 best_child_idx;
    // Odd-depth nodes are opponent moves: use Thompson Sampling once frozen.
    if (depth % 2 == 1 &&
        game.arena.nodes[current_idx].visits.load(std::memory_order_relaxed) >
            contempt_nscl) {
      best_child_idx = sample_frozen_child(game.arena, current_idx, rng);
    } else {
      best_child_idx = select_best_puct(game.arena, current_idx);
    }
    game.leaf_pos.makemove(game.arena.nodes[best_child_idx].move);
    current_idx = best_child_idx;

    game.search_path.push_back(current_idx);
    rollout_hashes.push_back(game.leaf_pos.zobristhash);
    depth++;
  }

  // TERMINAL CHECKS
  float value = 0.0f;
  bool is_terminal = false;

  if (game.leaf_pos.twokings()) {
    value = 0.0f;
    is_terminal = true;
  } else if (game.leaf_pos.bareking(!game.leaf_pos.stm)) {
    value = 1.0f;
    is_terminal = true;
  } else if (game.leaf_pos.halfmovecount >= 140) {
    value = 0.0f;
    is_terminal = true;
  } else if (is_repetition(game.leaf_pos, game.game_hashes, rollout_hashes)) {
    value = 0.0f;
    is_terminal = true;
  } else {
    Move moves[maxmoves];
    int movecount = game.leaf_pos.generatemoves(moves);
    if (movecount == 0) {
      value = -1.0f;
      is_terminal = true;
    }
  }

  // If terminal, backprop immediately and bypass the GPU
  if (is_terminal) {
    backprop(game, value);
    game.rollouts_completed++;
    return false;
  }

  // Wait for GPU batch
  return true;
}

void expand(DatagenGame &game, const float *policy, const float *value_logits,
            U64 &arena_overflows) {
  U32 current_idx = game.search_path.back();

  Move moves[maxmoves];
  int movecount = game.leaf_pos.generatemoves(moves);

  MCTSEval processed =
      parse_nn_output(policy, value_logits, moves, movecount, game.leaf_pos.stm);
  float value = processed.qscore;

  U32 child_start = game.arena.active_nodes.load(std::memory_order_relaxed);

  if (child_start + movecount >= game.arena.max_nodes) {
    // Silent truncation: the rollout still backprops, but the leaf keeps zero
    // children and will be re-selected forever. Counted so a too-small
    // datagenarenasize shows up as a number instead of as quiet quality loss.
    ++arena_overflows;
  }
  if (child_start + movecount < game.arena.max_nodes) {
    game.arena.active_nodes.fetch_add(movecount, std::memory_order_relaxed);

    for (int i = 0; i < movecount; i++) {
      size_t child_idx = child_start + i;
      game.arena.nodes[child_idx].move = moves[i];
      game.arena.nodes[child_idx].prior = processed.priors[i];
      game.arena.nodes[child_idx].visits = 0.0f;
      game.arena.nodes[child_idx].value_sum = 0.0f;
      game.arena.nodes[child_idx].first_child_idx = -1;
      game.arena.nodes[child_idx].num_children = 0;
      game.arena.nodes[child_idx].frozen_visits.store(0, std::memory_order_relaxed);
    }

    game.arena.nodes[current_idx].first_child_idx = child_start;
    game.arena.nodes[current_idx].num_children = movecount;
  }
  backprop(game, value);
  game.rollouts_completed++;
}

void generate_batched_selfplay_games(NNEvaluator &nn,
                                     const std::string &output_prefix,
                                     U64 nodecount, int total_positions) {
  std::string filename = output_prefix + ".data";
  std::ofstream out(filename, std::ios::binary | std::ios::app);
  if (!out.is_open()) {
    std::cerr << "Failed to open output file!" << std::endl;
    return;
  }

  auto begin = std::chrono::high_resolution_clock::now();
  auto last_print = begin;
  std::atomic<U64> total_nodes_evaluated{0};

  // --- Two independent game pools, alternating ---
  //
  // Inference is ~85% of a batch cycle and used to run with seven of eight
  // workers parked at a barrier. It cannot be overlapped within one pool:
  // select() for the next rollout depends on expand() of the current one. So
  // there are two pools, and while the GPU evaluates pool p the workers do the
  // tree work for pool 1-p. The GPU becomes the only thing on the critical
  // path.
  //
  // Everything a batch touches is per-pool: the games, the packed input
  // arrays, the index maps, and the evaluator slot the results land in. Nothing
  // is shared between pools except the output file and the counters.
  struct Pool {
    std::vector<std::unique_ptr<DatagenGame>> games;
    std::vector<U8> needs_nn;
    std::vector<int32_t> pieces;
    std::vector<int32_t> halfmoves;
    // Dense batch row -> game index, and its inverse. Both are rebuilt every
    // select phase; the round trip is asserted before any result is consumed.
    std::vector<int> batch_to_game;
    std::vector<int> game_to_batch;
    std::atomic<int> batch_size{0};
    int submitted_size = 0; // batch_size at submit, read by the consumer

    Pool()
        : needs_nn(datagenbatchsize, 0), pieces(datagenbatchsize * 64),
          halfmoves(datagenbatchsize), batch_to_game(datagenbatchsize, -1),
          game_to_batch(datagenbatchsize, -1) {
      games.reserve(datagenbatchsize);
      for (int i = 0; i < datagenbatchsize; ++i)
        games.push_back(std::make_unique<DatagenGame>(datagenarenasize));
    }
  };
  Pool pools[kBatchSlots];

  // --- Multi-threading Setup ---
  std::mutex file_mutex;
  std::atomic<int> games_completed{0};
  std::atomic<int> positions_written{0};
  std::atomic<bool> keep_running{true};

  // Peak arena occupancy across every game and every root move, and the number
  // of times a leaf could not be expanded because the arena was full.
  std::atomic<size_t> peak_arena_nodes{0};
  std::atomic<U64> total_arena_overflows{0};

  // Handshake between the workers and the dedicated inference thread, one per
  // pool. `submitted` counts batches handed over, `completed` counts batches
  // whose results are in the evaluator slot. A worker consumes pool p's results
  // only once completed > the number it has already consumed, so a missed or
  // duplicated wakeup cannot be mistaken for fresh data.
  struct PoolSync {
    std::mutex m;
    std::condition_variable cv;
    U64 submitted = 0;
    U64 completed = 0;
    bool failed = false;
  };
  PoolSync sync[kBatchSlots];
  std::atomic<bool> inference_failed{false};
  std::string inference_error;

  std::barrier sync_point(datagenthreads);

  // SZ0_TIME_IO=1 also turns on this loop-level breakdown, measured on thread 0
  // only. The `[io]` timer inside NNEvaluator covers just the inference phase,
  // which is the one phase independent of nodes/move — everything that scales
  // with tree depth lives in select/expand and is invisible there.
  //
  // Now that inference is pipelined, `gpu_wait` is the headline number: it is
  // time the workers spent blocked on results, i.e. the GPU being the
  // bottleneck. That is the healthy state. If it goes to zero the tree work has
  // become the bottleneck instead, and adding a third pool would do nothing.
  // `*_wait` is barrier time, i.e. load imbalance between workers.
  struct DatagenTimer {
    bool on = false;
    long long report_every = 1000, iters = 0;
    double select = 0, select_wait = 0, gpu_wait = 0, expand = 0,
           expand_wait = 0;
    // Occupied rows per batch. Below datagenbatchsize because terminal leaves
    // backprop inside select() without an NN eval, and a game that has
    // finished its rollouts spends the iteration picking a move. The captured
    // graph runs all datagenbatchsize rows regardless, so the shortfall is
    // wasted GPU work — and it is also why reported nps is lower than
    // datagenbatchsize/cycle would imply.
    double batch_fill = 0;
    DatagenTimer() {
      const char *v = std::getenv("SZ0_TIME_IO");
      on = (v && v[0] == '1');
      if (const char *e = std::getenv("SZ0_TIME_IO_EVERY")) {
        long long p = atoll(e);
        if (p > 0)
          report_every = p;
      }
    }
    void tick() {
      if (++iters < report_every)
        return;
      double n = static_cast<double>(iters);
      double total = select + select_wait + gpu_wait + expand + expand_wait;
      // Report the nps this cycle time implies, so it can be checked against
      // the engine's own TRUE NPS without doing the arithmetic by hand.
      double implied_nps = batch_fill / n * 1e6 / (total / n);
      fprintf(stderr,
              "[dg] n=%lld | select %.0f (+%.0f wait)  expand %.0f (+%.0f "
              "wait)  gpu_wait %.0f  | cycle %.0f us, fill %.0f/%d, "
              "gpu-bound %.1f%%, implied %.0fK nps\n",
              iters, select / n, select_wait / n, expand / n, expand_wait / n,
              gpu_wait / n, total / n, batch_fill / n, datagenbatchsize,
              100.0 * gpu_wait / total, implied_nps / 1000.0);
      iters = 0;
      select = select_wait = gpu_wait = expand = expand_wait = batch_fill = 0;
    }
  } dg_timer;
  using dgclk = std::chrono::steady_clock;
  auto dg_lap = [](dgclk::time_point &t) {
    auto now = dgclk::now();
    double d = std::chrono::duration<double, std::micro>(now - t).count();
    t = now;
    return d;
  };

  auto worker = [&](int thread_idx) {
    // Spread the remainder one game per thread instead of dumping all of it on
    // the last one. At 284/8 the old split gave thread 7 thirty-nine games
    // against everyone else's thirty-five, and since every phase ends at a
    // barrier, all eight threads paid for the slowest.
    int chunk_size = datagenbatchsize / datagenthreads;
    int remainder = datagenbatchsize % datagenthreads;
    int start_idx = thread_idx * chunk_size + std::min(thread_idx, remainder);
    int end_idx = start_idx + chunk_size + (thread_idx < remainder ? 1 : 0);

    std::mt19937 rng(std::random_device{}() + thread_idx);
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);

    // Accumulated per thread and merged once at the end, so the hot loop never
    // touches a shared atomic.
    size_t local_peak_arena = 0;
    U64 local_overflows = 0;

    for (int p = 0; p < kBatchSlots; ++p) {
      for (int i = start_idx; i < end_idx; ++i) {
        pools[p].games[i]->reset(rng);
      }
    }

    const bool dg_timing = dg_timer.on && thread_idx == 0;
    dgclk::time_point dg_t{};

    // How many of each pool's completed batches this worker has already
    // consumed. Compared against sync[p].completed rather than trusting a
    // wakeup, so a spurious or coalesced notify cannot be read as fresh data.
    U64 consumed[kBatchSlots] = {0, 0};

    bool running = true;
    for (U64 iter = 0; running; ++iter) {
      // Pools strictly alternate. Pool p was last submitted at iter-kBatchSlots,
      // so for the first kBatchSlots iterations there is nothing to consume —
      // that is the pipeline filling.
      const int p = static_cast<int>(iter % kBatchSlots);
      Pool &pool = pools[p];

      if (dg_timing)
        dg_t = dgclk::now();

      // ==========================================
      // Wait for this pool's results (GPU is the bottleneck here)
      // ==========================================
      const float *batch_policy = nullptr;
      const float *batch_value = nullptr;
      if (iter >= kBatchSlots) {
        std::unique_lock<std::mutex> lk(sync[p].m);
        sync[p].cv.wait(lk, [&] {
          return sync[p].completed > consumed[p] || sync[p].failed;
        });
        if (sync[p].failed) {
          lk.unlock();
          // Leave via the same path as a normal stop so every worker still
          // arrives at the barriers below the expected number of times.
          keep_running.store(false, std::memory_order_relaxed);
        } else {
          consumed[p] = sync[p].completed;
          lk.unlock();
          batch_policy = nn.batch_policy(p);
          batch_value = nn.batch_value(p);
        }
      }

      if (dg_timing)
        dg_timer.gpu_wait += dg_lap(dg_t);

      // ==========================================
      // Expand & Move Check (Parallelized)
      // ==========================================
      for (int i = start_idx; i < end_idx; ++i) {
        DatagenGame &g = *pool.games[i];

        if (batch_policy && pool.needs_nn[i] && g.rollouts_completed < nodecount) {
          int b = pool.game_to_batch[i];
          // Round-trip the index map before touching any result. A mis-scatter
          // here would put another game's policy into this game's tree and
          // silently poison the training data rather than crash, so it is worth
          // two array reads to make it impossible.
          if (b < 0 || b >= pool.submitted_size || pool.batch_to_game[b] != i) {
            fprintf(stderr,
                    "FATAL: batch index map inconsistent (pool %d, game %d, "
                    "row %d, size %d)\n",
                    p, i, b, pool.submitted_size);
            std::abort();
          }
          expand(g, batch_policy + static_cast<size_t>(b) * 4096,
                 batch_value + static_cast<size_t>(b) * 3, local_overflows);
        }

        if (g.rollouts_completed >= nodecount) {
          Node &root = g.arena.nodes[0];
          int game_result = 0;
          bool is_terminal = false;

          if (root.num_children == 0) {
            game_result = g.root_pos.stm ? 1 : -1;
            is_terminal = true;
          } else {
            float total_child_visits = 0.0f;
            for (int c = 0; c < root.num_children; ++c) {
              total_child_visits +=
                  g.arena.nodes[root.first_child_idx + c].visits.load(
                      std::memory_order_relaxed);
            }

            TrainingPosition train_pos;
            train_pos.halfmove_clock = g.root_pos.halfmovecount;
            get_canonical_tokens(g.root_pos, train_pos.board_tokens);
            train_pos.num_moves = root.num_children;
            train_pos.root_q =
                root.value_sum.load(std::memory_order_relaxed) /
                std::max(1.0f, root.visits.load(std::memory_order_relaxed));

            float temperature;
            if (g.ply_count < 60) {
              temperature = 1.0f - 0.7f * g.ply_count / 60.0f;
            } else {
              temperature = 0.3f;
            }

            int best_child_idx = 0;
            float max_visits = -1.0f;

            for (int c = 0; c < root.num_children; ++c) {
              float v = g.arena.nodes[root.first_child_idx + c].visits.load(
                  std::memory_order_relaxed);
              if (v > max_visits) {
                max_visits = v;
                best_child_idx = c;
              }
            }

            // Default to greedy best move; overridden below for T > 0
            Move selected_move =
                g.arena.nodes[root.first_child_idx + best_child_idx].move;

            for (int c = 0; c < root.num_children; ++c) {
              Node &child = g.arena.nodes[root.first_child_idx + c];
              Move m = child.move;

              int from_sq = g.root_pos.stm ? (m.from() ^ 56) : m.from();
              int to_sq = g.root_pos.stm ? (m.to() ^ 56) : m.to();
              train_pos.move_indices[c] = (from_sq * 64) + to_sq;

              float visit_prob =
                  child.visits.load(std::memory_order_relaxed) /
                  total_child_visits;
              train_pos.move_probabilities[c] = visit_prob;
            }

            if (temperature > 0.0f) {
              // Normalize by max visits before exponentiation to prevent
              // overflow: (v/v_max)^(1/T) is equivalent since v_max^(1/T)
              // is a common factor that cancels in normalization.
              float inv_temp = 1.0f / temperature;
              float sum_tempered = 0.0f;
              for (int c = 0; c < root.num_children; ++c) {
                float v = g.arena.nodes[root.first_child_idx + c].visits.load(
                    std::memory_order_relaxed);
                sum_tempered += std::pow(v / max_visits, inv_temp);
              }
              float random_choice = prob_dist(rng) * sum_tempered;
              float cumulative = 0.0f;
              for (int c = 0; c < root.num_children; ++c) {
                float v = g.arena.nodes[root.first_child_idx + c].visits.load(
                    std::memory_order_relaxed);
                cumulative += std::pow(v / max_visits, inv_temp);
                if (random_choice <= cumulative) {
                  selected_move = g.arena.nodes[root.first_child_idx + c].move;
                  break;
                }
              }
            }

            g.game_history.push_back(train_pos);
            g.game_hashes.push_back(g.root_pos.zobristhash);
            g.root_pos.makemove(selected_move);
            g.ply_count++;

            if (g.root_pos.twokings()) {
              game_result = 0;
              is_terminal = true;
            } else if (g.root_pos.bareking(!g.root_pos.stm)) {
              game_result = g.root_pos.stm ? -1 : 1;
              is_terminal = true;
            } else if (g.root_pos.halfmovecount >= 140) {
              game_result = 0;
              is_terminal = true;
            } else if (is_repetition(g.root_pos, g.game_hashes)) {
              game_result = 0;
              is_terminal = true;
            }
          }

          if (is_terminal) {
            for (int ply = 0; ply < g.ply_count; ++ply) {
              if (game_result == 0) {
                g.game_history[ply].outcome = 0;
              } else {
                bool is_white_turn = (ply % 2 == 0);
                g.game_history[ply].outcome = (game_result == 1)
                                                  ? (is_white_turn ? 1 : -1)
                                                  : (is_white_turn ? -1 : 1);
              }
            }

            {
              std::lock_guard<std::mutex> lock(file_mutex);
              positions_written += save_game_to_binary(g.game_history, out);
              int current_games = ++games_completed;

              std::cout << "Positions: " << positions_written.load() << " / "
                        << total_positions << "  Games: " << current_games
                        << "\r" << std::flush;

              g.reset(rng);
            }
          }

          if (g.root_pos.halfmovecount == 0 && !is_terminal) {
            g.game_hashes.clear();
          }
          // Sampled here because the arena is cleared once per root move, so
          // this is its high-water mark for one search of `nodecount` rollouts.
          local_peak_arena =
              std::max(local_peak_arena,
                       g.arena.active_nodes.load(std::memory_order_relaxed));
          g.arena.clear();
          g.rollouts_completed = 0;
        }
      }

      if (dg_timing)
        dg_timer.expand += dg_lap(dg_t);

      sync_point.arrive_and_wait();
      if (dg_timing)
        dg_timer.expand_wait += dg_lap(dg_t);

      // ==========================================
      // MCTS Select (Parallelized) — builds pool p's next batch
      // ==========================================
      for (int i = start_idx; i < end_idx; ++i) {
        pool.needs_nn[i] = 0;
        DatagenGame &g = *pool.games[i];

        // Retry rather than yielding the slot. select() finishes a rollout by
        // itself when the leaf is terminal (bare king, 70-move, repetition,
        // stalemate) — it backprops and returns false, having already counted
        // the rollout. Leaving the batch row empty in that case wastes a row of
        // a fixed-shape captured batch, which the GPU runs whether or not it is
        // occupied. Terminal leaves get commoner as games reach endgames, which
        // is why fill (and so nps) used to decay over a run.
        //
        // Bounded: the only false return increments rollouts_completed.
        while (g.rollouts_completed < nodecount) {
          if (select(g, rng)) {
            pool.needs_nn[i] = 1;

            // Lock-free atomic reservation for this board's slot in the batch
            int b_idx = pool.batch_size.fetch_add(1, std::memory_order_relaxed);
            pool.batch_to_game[b_idx] = i;
            pool.game_to_batch[i] = b_idx;

            // Thread computes perspective flips directly into the shared GPU
            // buffer
            for (int sq = 0; sq < 64; ++sq) {
              int p_sq = g.leaf_pos.stm ? (sq ^ 56) : sq;
              pool.pieces[(b_idx * 64) + p_sq] =
                  perspectivepiece(g.leaf_pos.pieces[sq], g.leaf_pos.stm) +
                  13 * p_sq;
            }
            pool.halfmoves[b_idx] = clamp_halfmove(g.leaf_pos.halfmovecount);
            break; // slot filled; this game is done for the iteration
          }
        }
      }

      if (dg_timing)
        dg_timer.select += dg_lap(dg_t);

      sync_point.arrive_and_wait();
      if (dg_timing)
        dg_timer.select_wait += dg_lap(dg_t); // (submit + final barrier added
                                              // below, so cycle stays exact)

      // ==========================================
      // Hand the batch to the inference thread (thread 0 only)
      // ==========================================
      if (thread_idx == 0) {
        int batch_size = pool.batch_size.load(std::memory_order_relaxed);
        // Written before `submitted` is incremented under the mutex, so the
        // release/acquire pair publishes it to the inference thread.
        pool.submitted_size = batch_size;
        total_nodes_evaluated.fetch_add(batch_size, std::memory_order_relaxed);
        if (dg_timing)
          dg_timer.batch_fill += batch_size;
        // Reset now, not before the next select: this pool is not touched again
        // until kBatchSlots iterations later, with several barriers between.
        pool.batch_size.store(0, std::memory_order_relaxed);

        {
          std::lock_guard<std::mutex> lk(sync[p].m);
          ++sync[p].submitted;
        }
        sync[p].cv.notify_all();

        if (positions_written.load(std::memory_order_relaxed) >=
            total_positions) {
          keep_running.store(false, std::memory_order_relaxed);
        }
        auto now = std::chrono::high_resolution_clock::now();
        auto since_print = std::chrono::duration_cast<std::chrono::milliseconds>(
                               now - last_print)
                               .count();
        if (since_print >= 1000) {
          auto elapsed_ms =
              std::chrono::duration_cast<std::chrono::milliseconds>(now - begin)
                  .count();
          if (elapsed_ms > 0) {
            std::cout << "TRUE NPS: "
                      << (1000 * total_nodes_evaluated.load(
                                     std::memory_order_relaxed) /
                          elapsed_ms)
                      << "\r" << std::flush;
          }
          last_print = now;
        }
      }

      // Single exit point, read identically by every worker after a barrier, so
      // all eight leave the loop on the same iteration and the barrier count
      // stays consistent. No arrive_and_drop needed anywhere.
      sync_point.arrive_and_wait();
      running = keep_running.load(std::memory_order_relaxed);
      if (dg_timing) {
        dg_timer.select_wait += dg_lap(dg_t);
        dg_timer.tick();
      }
    }

    // Every exit above breaks out of the loop, so one merge here covers all of
    // them. No fetch_max before C++26, hence the CAS.
    total_arena_overflows.fetch_add(local_overflows, std::memory_order_relaxed);
    size_t seen = peak_arena_nodes.load(std::memory_order_relaxed);
    while (local_peak_arena > seen &&
           !peak_arena_nodes.compare_exchange_weak(seen, local_peak_arena,
                                                   std::memory_order_relaxed)) {
    }
  };

  // The one thread that touches the GPU. Pools alternate in the same order the
  // workers submit them, so a batch is always waiting by the time the previous
  // one finishes and the device never idles.
  std::atomic<bool> inference_stop{false};
  auto inference_loop = [&]() {
    U64 processed[kBatchSlots] = {0, 0};
    for (U64 iter = 0;; ++iter) {
      const int p = static_cast<int>(iter % kBatchSlots);
      {
        std::unique_lock<std::mutex> lk(sync[p].m);
        sync[p].cv.wait(lk, [&] {
          return sync[p].submitted > processed[p] ||
                 inference_stop.load(std::memory_order_relaxed);
        });
        if (sync[p].submitted <= processed[p])
          return; // stopping, nothing left for this pool
      }
      ++processed[p];

      try {
        // An empty batch happens roughly once per nodecount iterations, when
        // every game reaches its rollout budget together. Skip the Run but
        // still publish completion, or the workers would wait forever.
        if (pools[p].submitted_size > 0)
          nn.infer_packed(pools[p].pieces, pools[p].halfmoves, p);
      } catch (const std::exception &e) {
        // Fail both pools: a worker may be blocked on either one.
        inference_error = e.what();
        inference_failed.store(true, std::memory_order_relaxed);
        keep_running.store(false, std::memory_order_relaxed);
        for (int q = 0; q < kBatchSlots; ++q) {
          {
            std::lock_guard<std::mutex> lk(sync[q].m);
            sync[q].failed = true;
          }
          sync[q].cv.notify_all();
        }
        return;
      }

      {
        std::lock_guard<std::mutex> lk(sync[p].m);
        ++sync[p].completed;
      }
      sync[p].cv.notify_all();
    }
  };

  // --- Fire up the threads ---
  std::thread inference_thread(inference_loop);
  std::vector<std::thread> threads;
  for (int i = 0; i < datagenthreads; ++i) {
    threads.emplace_back(worker, i);
  }

  // --- Wait for the entire datagen run to finish ---
  for (auto &t : threads) {
    t.join();
  }
  // Only now can the inference thread be released: until every worker has
  // stopped, one of them may still be waiting on a batch it submitted.
  inference_stop.store(true, std::memory_order_relaxed);
  for (int p = 0; p < kBatchSlots; ++p) {
    {
      std::lock_guard<std::mutex> lk(sync[p].m);
    }
    sync[p].cv.notify_all();
  }
  inference_thread.join();

  if (inference_failed.load(std::memory_order_relaxed)) {
    std::cerr << "\nERROR: inference thread failed: " << inference_error
              << "\n";
  }

  size_t peak = peak_arena_nodes.load(std::memory_order_relaxed);
  U64 overflows = total_arena_overflows.load(std::memory_order_relaxed);
  std::cout << "\nPeak arena occupancy: " << peak << " / " << datagenarenasize
            << " nodes (" << (100.0 * peak / datagenarenasize) << "%, "
            << (peak * sizeof(Node) / 1024) << " KB of "
            << (datagenarenasize * sizeof(Node) / 1024) << " KB per game)\n";
  if (overflows > 0) {
    std::cout << "WARNING: " << overflows
              << " leaf expansions were dropped because the arena was full — "
                 "raise datagenarenasize in src/consts.h\n";
  }
}
