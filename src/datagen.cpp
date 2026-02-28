#include "datagen.h"

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
        out.write(reinterpret_cast<const char*>(&pos.num_legal_moves), sizeof(pos.num_legal_moves));
        out.write(reinterpret_cast<const char*>(&pos.halfmove_clock), sizeof(pos.halfmove_clock));
        out.write(reinterpret_cast<const char*>(&pos.outcome), sizeof(pos.outcome));
        
        // board
        out.write(reinterpret_cast<const char*>(pos.board_tokens), sizeof(pos.board_tokens));
        
        // policy data
        if (pos.num_legal_moves > 0) {
            out.write(reinterpret_cast<const char*>(pos.move_indices), pos.num_legal_moves * sizeof(uint16_t));
            out.write(reinterpret_cast<const char*>(pos.move_probabilities), pos.num_legal_moves * sizeof(float));
        }
    }
    
    out.close();
}