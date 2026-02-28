#include "datagen.h"
#include "node.h"
#include "search.h"

#include <fstream>

void save_game_to_binary(const std::vector<TrainingPosition>& game_history, const std::string& filename) {
    std::ofstream out(filename, std::ios::app | std::ios::binary);
    
    if (!out) {
        std::cerr << "Error: Could not open " << filename << " for writing.\n";
        return;
    }

    for (const auto& pos : game_history) {
        // metadata and value data
        out.write(reinterpret_cast<const char*>(&pos.root_q), sizeof(pos.root_q));
        out.write(reinterpret_cast<const char*>(&pos.num_moves), sizeof(pos.num_moves));
        out.write(reinterpret_cast<const char*>(&pos.halfmove_clock), sizeof(pos.halfmove_clock));
        out.write(reinterpret_cast<const char*>(&pos.outcome), sizeof(pos.outcome));
        
        // board
        out.write(reinterpret_cast<const char*>(pos.board_tokens), sizeof(pos.board_tokens));
        
        // policy data
        if (pos.num_moves > 0) {
            out.write(reinterpret_cast<const char*>(pos.move_indices), pos.num_moves * sizeof(uint16_t));
            out.write(reinterpret_cast<const char*>(pos.move_probabilities), pos.num_moves * sizeof(float));
        }
    }
    
    out.close();
}

void get_canonical_tokens(const Position& pos, uint8_t tokens[64]) {
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

int generate_mcts_selfplay_game(const std::string& output_filename, U64 nodecount) {
    Position pos;
    pos.initialize();
    
    // Instantiate your search tree memory pool
    TreeArena arena(datagenarenasize); 
    std::vector<TrainingPosition> game_history;
    std::vector<U64> game_hashes;
    
    std::mt19937 rng(std::random_device{}());

    int ply_count = 0;
    int game_result = 0; // 0: Draw, 1: White Wins, -1: Black Wins

    while (true) {
        if (pos.halfmovecount == 0) {
            game_hashes.clear();
        }
        arena.clear();
        search_position(arena, pos, game_hashes, 0, nodecount, 1, false);
        Node& root = arena.nodes[0];
        if (root.num_children == 0) {
            game_result = pos.stm ? 1 : -1; 
            break;
        }
        // Calculate total visits across all children to form the probability distribution
        float total_child_visits = 0.0f;
        for (int i = 0; i < root.num_children; ++i) {
            total_child_visits += arena.nodes[root.first_child_idx + i].visits.load();
        }
        if (total_child_visits == 0.0f) {
            total_child_visits = 1.0f;
        }
        // Write training data
        TrainingPosition train_pos;
        train_pos.halfmove_clock = pos.halfmovecount;
        get_canonical_tokens(pos, train_pos.board_tokens);
        train_pos.num_moves = root.num_children;
        train_pos.root_q = root.value_sum.load() / std::max(1, root.visits.load()); 
        
        // Select actual move by sampling from visit counts (Temperature = 1.0)
        std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
        float random_choice = prob_dist(rng);
        float cumulative_prob = 0.0f;
        Move selected_move = arena.nodes[root.first_child_idx].move;

        for (int i = 0; i < root.num_children; ++i) {
            Node& child = arena.nodes[root.first_child_idx + i];
            Move m = child.move;
            // Write policy data with perspective
            int from_sq = pos.stm ? (m.from() ^ 56) : m.from();
            int to_sq = pos.stm ? (m.to() ^ 56) : m.to();
            
            train_pos.move_indices[i] = (from_sq * 64) + to_sq; 
            
            float visit_prob = child.visits.load() / total_child_visits;
            train_pos.move_probabilities[i] = visit_prob;

            cumulative_prob += visit_prob;
            if (random_choice <= cumulative_prob && random_choice != -1.0f) {
                selected_move = m;
                random_choice = -1.0f;
            }
        }

        game_history.push_back(train_pos);
        game_hashes.push_back(pos.zobristhash);

        pos.makemove(selected_move);
        ply_count++;
        
        if (pos.twokings()) {
            game_result = 0;
            break;
        } 
        if (pos.bareking(!pos.stm)) {
            game_result = pos.stm ? -1 : 1;
            break;
        } 
        if (pos.halfmovecount >= 140) {
            game_result = 0;
            break;
        } 
        if (is_repetition(pos, game_hashes)) {
            game_result = 0;
            break;
        }
    }

    for (int i = 0; i < ply_count; ++i) {
        if (game_result == 0) {
            game_history[i].outcome = 0;
        } else {
            bool is_white_turn = (i % 2 == 0); 
            if (game_result == 1) { 
                game_history[i].outcome = is_white_turn ? 1 : -1;
            } else { 
                game_history[i].outcome = is_white_turn ? -1 : 1;
            }
        }
    }

    save_game_to_binary(game_history, output_filename);
    return ply_count;
}

void datagen_worker(int thread_id, int node_limit, int game_count, const std::string& base_filename) {
    std::string output_filename = base_filename + std::to_string(thread_id) + ".data";
    std::cout << "[Thread " << thread_id << "] Started playing. Writing to: " << output_filename << "\n";
    
    int games_played = 0;
    int positions_saved = 0;
    
    // Infinite loop: keep playing games until you Ctrl+C the program
    while (games_played < game_count) {
        positions_saved += generate_mcts_selfplay_game(output_filename, node_limit);
        games_played++;
        // Print an update every 10 games so you know it hasn't frozen
        if (games_played % 10 == 0) {
            std::cout << "[Thread " << thread_id << "] Completed " << games_played << " games and "
                                                                   << positions_saved << " positions\n";
        }
    }
}