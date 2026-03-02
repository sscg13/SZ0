#include "datagen.h"
#include "node.h"
#include "search.h"

#include <barrier>
#include <fstream>
#include <thread>

void save_game_to_binary(const std::vector<TrainingPosition> &game_history,
                         std::ofstream &out) {
  if (!out) {
    std::cerr << "Error: Could not open pipe for writing.\n";
    return;
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
  for (int i = game.search_path.size() - 1; i >= 0; --i) {
    U32 idx = game.search_path[i];
    game.arena.nodes[idx].visits.fetch_add(1, std::memory_order_relaxed);
    game.arena.nodes[idx].value_sum.fetch_add(value, std::memory_order_relaxed);
    value = -value;
  }
}

bool select(DatagenGame &game) {
  game.search_path.clear();
  game.leaf_pos = game.root_pos;
  U32 current_idx = 0;
  game.search_path.push_back(current_idx);

  std::vector<uint64_t> rollout_hashes;
  rollout_hashes.push_back(game.leaf_pos.zobristhash);

  // SELECTION
  while (game.arena.nodes[current_idx].num_children > 0) {
    U32 best_child_idx = select_best_puct(game.arena, current_idx);
    game.leaf_pos.makemove(game.arena.nodes[best_child_idx].move);
    current_idx = best_child_idx;

    game.search_path.push_back(current_idx);
    rollout_hashes.push_back(game.leaf_pos.zobristhash);
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

void expand(DatagenGame &game, const NNOutput &raw_nn) {
  U32 current_idx = game.search_path.back();

  Move moves[maxmoves];
  int movecount = game.leaf_pos.generatemoves(moves);

  MCTSEval processed =
      parse_nn_output(raw_nn, moves, movecount, game.leaf_pos.stm);
  float value = processed.qscore;

  U32 child_start = game.arena.active_nodes;

  if (child_start + movecount < game.arena.max_nodes) {
    game.arena.active_nodes.fetch_add(movecount, std::memory_order_relaxed);

    for (int i = 0; i < movecount; i++) {
      size_t child_idx = child_start + i;
      game.arena.nodes[child_idx].move = moves[i];
      game.arena.nodes[child_idx].prior = processed.priors[i];
      game.arena.nodes[child_idx].visits = 0;
      game.arena.nodes[child_idx].value_sum = 0.0f;
      game.arena.nodes[child_idx].first_child_idx = -1;
      game.arena.nodes[child_idx].num_children = 0;
    }

    game.arena.nodes[current_idx].first_child_idx = child_start;
    game.arena.nodes[current_idx].num_children = movecount;
  }
  backprop(game, value);
  game.rollouts_completed++;
}

void generate_batched_selfplay_games(NNEvaluator &nn,
                                     const std::string &output_prefix,
                                     U64 nodecount, int total_games_to_play) {
  std::string filename = output_prefix + ".data";
  std::ofstream out(filename, std::ios::binary | std::ios::app);
  if (!out.is_open()) {
    std::cerr << "Failed to open output file!" << std::endl;
    return;
  }

  auto begin = std::chrono::high_resolution_clock::now();
  auto last_print = begin;
  int total_nodes_evaluated = 0;
  bool flush_onnx_output = false;

  std::vector<std::unique_ptr<DatagenGame>> games;
  games.reserve(datagenbatchsize);
  for (int i = 0; i < datagenbatchsize; ++i) {
    games.push_back(std::make_unique<DatagenGame>(datagenarenasize));
  }

  // --- Multi-threading Setup ---
  std::mutex file_mutex;
  std::atomic<int> games_completed{0};
  std::atomic<int> positions_written{0};
  std::atomic<bool> keep_running{true};

  std::vector<U8> shared_needs_nn(datagenbatchsize, 0);
  std::vector<NNOutput> shared_nn_results(datagenbatchsize);

  // Pre-allocated flat arrays for the GPU batch
  std::vector<int32_t> batched_pieces(datagenbatchsize * 64);
  std::vector<int32_t> batched_halfmoves(datagenbatchsize);

  // Maps the dense GPU batch index (0 to current_batch_size) back to the global
  // game index (0 to datagenbatchsize)
  std::vector<int> batch_to_game_idx(datagenbatchsize);
  std::atomic<int> current_batch_size{0};

  std::barrier sync_point(datagenthreads);

  auto worker = [&](int thread_idx) {
    int chunk_size = datagenbatchsize / datagenthreads;
    int start_idx = thread_idx * chunk_size;
    int end_idx = (thread_idx == datagenthreads - 1) ? datagenbatchsize
                                                     : start_idx + chunk_size;

    std::mt19937 rng(std::random_device{}() + thread_idx);
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);

    for (int i = start_idx; i < end_idx; ++i) {
        games[i]->reset(rng);
    }

    while (true) {

      // ==========================================
      // MCTS Select (Parallelized)
      // ==========================================
      for (int i = start_idx; i < end_idx; ++i) {
        shared_needs_nn[i] = 0;
        DatagenGame &g = *games[i];
        if (!g.is_active)
          continue;

        if (g.rollouts_completed < nodecount) {
          bool needs_nn = select(g);
          if (needs_nn) {
            shared_needs_nn[i] = 1;

            // Lock-free atomic reservation for this board's slot in the batch
            int b_idx =
                current_batch_size.fetch_add(1, std::memory_order_relaxed);
            batch_to_game_idx[b_idx] = i;

            // Thread computes perspective flips directly into the shared GPU
            // buffer
            for (int sq = 0; sq < 64; ++sq) {
              int p_sq = g.leaf_pos.stm ? (sq ^ 56) : sq;
              batched_pieces[(b_idx * 64) + p_sq] =
                  perspectivepiece(g.leaf_pos.pieces[sq], g.leaf_pos.stm);
            }
            batched_halfmoves[b_idx] = g.leaf_pos.halfmovecount;
          }
        }
      }

      if (!keep_running.load(std::memory_order_relaxed)) {
        sync_point.arrive_and_drop();
        break;
      }
      sync_point.arrive_and_wait();

      // ==========================================
      // GPU Inference (Thread 0 Only)
      // ==========================================
      if (thread_idx == 0) {
        int batch_size = current_batch_size.load(std::memory_order_relaxed);
        total_nodes_evaluated += batch_size;

        if (batch_size > 0) {
          // Hand the pre-packed flat arrays directly to ONNX
          nn.infer_packed(batched_pieces, batched_halfmoves, shared_nn_results,
                          batch_to_game_idx);
        }

        // Reset the atomic batch size for the next MCTS loop
        current_batch_size.store(0, std::memory_order_relaxed);

        if (games_completed.load(std::memory_order_relaxed) >=
            total_games_to_play) {
          keep_running.store(false, std::memory_order_relaxed);
        }
        auto now = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                            now - last_print)
                            .count();
        int nps =
            1000 * total_nodes_evaluated /
            std::chrono::duration_cast<std::chrono::milliseconds>(now - begin)
                .count();
        if (duration >= 1000) {
          std::cout << "TRUE NPS: " << nps << "\r" << std::flush;
          last_print = now;
        }
      }

      if (!keep_running.load(std::memory_order_relaxed)) {
        sync_point.arrive_and_drop();
        break;
      }
      sync_point.arrive_and_wait();

      // ==========================================
      // Expand & Move Check (Parallelized)
      // ==========================================
      for (int i = start_idx; i < end_idx; ++i) {
        DatagenGame &g = *games[i];
        if (!g.is_active)
          continue;

        if (shared_needs_nn[i] && g.rollouts_completed < nodecount) {
          expand(g, shared_nn_results[i]);
        }

        if (g.rollouts_completed >= nodecount) {
          Node &root = g.arena.nodes[0];
          int game_result = 0;
          bool is_terminal = false;

          if (root.num_children == 0) {
            game_result = g.root_pos.stm ? 1 : -1;
            is_terminal = true;
          } else {
            int total_child_visits = 0;
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
                root.value_sum /
                std::max(1, root.visits.load(std::memory_order_relaxed));

            float random_choice = prob_dist(rng);
            float cumulative_prob = 0.0f;
            Move selected_move = g.arena.nodes[root.first_child_idx].move;

            for (int c = 0; c < root.num_children; ++c) {
              Node &child = g.arena.nodes[root.first_child_idx + c];
              Move m = child.move;

              int from_sq = g.root_pos.stm ? (m.from() ^ 56) : m.from();
              int to_sq = g.root_pos.stm ? (m.to() ^ 56) : m.to();

              train_pos.move_indices[c] = (from_sq * 64) + to_sq;
              float visit_prob = child.visits / total_child_visits;
              train_pos.move_probabilities[c] = visit_prob;

              cumulative_prob += visit_prob;
              if (random_choice <= cumulative_prob && random_choice != -1.0f) {
                selected_move = m;
                random_choice = -1.0f;
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
            for (int p = 0; p < g.ply_count; ++p) {
              if (game_result == 0) {
                g.game_history[p].outcome = 0;
              } else {
                bool is_white_turn = (p % 2 == 0);
                g.game_history[p].outcome = (game_result == 1)
                                                ? (is_white_turn ? 1 : -1)
                                                : (is_white_turn ? -1 : 1);
              }
            }

            {
              std::lock_guard<std::mutex> lock(file_mutex);
              save_game_to_binary(g.game_history, out);
              int current_games = ++games_completed;
              positions_written += g.game_history.size();

              std::cout << "                   Games completed: "
                        << current_games << " / " << total_games_to_play
                        << " (Positions written: " << positions_written.load()
                        << ")\r" << std::flush;

              // Reset game logic (something not quite right here?)
              if (current_games + datagenbatchsize <= total_games_to_play) {
                g.reset(rng);
              } else {
                g.is_active = false;
              }
            }
          }

          if (g.root_pos.halfmovecount == 0 && !is_terminal) {
            g.game_hashes.clear();
          }
          g.arena.clear();
          g.rollouts_completed = 0;
        }
      }

      if (!keep_running.load(std::memory_order_relaxed)) {
        sync_point.arrive_and_drop();
        break;
      }
      sync_point.arrive_and_wait();
    }
  };

  // --- Fire up the threads ---
  std::vector<std::thread> threads;
  for (int i = 0; i < datagenthreads; ++i) {
    threads.emplace_back(worker, i);
  }

  // --- Wait for the entire datagen run to finish ---
  for (auto &t : threads) {
    t.join();
  }
}
