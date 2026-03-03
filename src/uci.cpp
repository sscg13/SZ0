#include "datagen.h"
#include "inference.h"
#include "node.h"
#include "position.h"
#include "search.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// clang-format off
std::string uciinfostring = 
"id name Shatranj Zer0\n"
"id author sscg13\n"
"option name UCI_Variant type combo default shatranj var shatranj\n"
"option name Threads type spin default 1 min 1 max 16\n"
"option name Hash type spin default 72 min 1 max 32768\n"
"uciok\n";
// clang-format on

void uci() {
  std::string ucicommand;
  Position current_pos;
  current_pos.initialize();
  TreeArena arena(defaultarenasize);
  std::vector<U64> game_hashes;
  NNEvaluator nn(NNFILE);
  int threadcount = 1;

  while (std::getline(std::cin, ucicommand)) {
    std::stringstream tokens(ucicommand);
    std::string token;
    tokens >> token;
    if (token == "quit") {
      break;
    }
    if (token == "uci") {
      std::cout << uciinfostring;
    }
    if (token == "isready") {
      std::cout << "readyok\n";
    }
    if (token == "ucinewgame") {
      arena.clear();
      game_hashes.clear();
    }
    if (token == "position") {
      game_hashes.clear();
      tokens >> token;
      if (token == "startpos") {
        current_pos.initialize();
      } else if (token == "fen") {
        std::string fen;
        for (int i = 0; i < 6 && tokens >> token; i++) {
          fen += token + " ";
        }
        current_pos.parseFEN(fen);
      }
      while (tokens >> token) {
        if (token != "moves") {
          Move moves[maxmoves];
          int len = current_pos.generatemoves(moves);
          int played = -1;
          for (int j = 0; j < len; j++) {
            if (algebraic(moves[j]) == token) {
              played = j;
            }
          }
          if (played >= 0) {
            game_hashes.push_back(current_pos.zobristhash);
            current_pos.makemove(moves[played]);
            if (current_pos.halfmovecount == 0) {
              game_hashes.clear();
            }
          }
        }
      }
    }
    if (token == "go") {
      int wtime = 0;
      int btime = 0;
      int winc = 0;
      int binc = 0;
      int movetime = 0;
      U64 nodecount = 0;
      while (tokens >> token) {
        if (token == "wtime") {
          tokens >> token;
          wtime = std::stoi(token);
        }
        if (token == "winc") {
          tokens >> token;
          winc = std::stoi(token);
        }
        if (token == "btime") {
          tokens >> token;
          btime = std::stoi(token);
        }
        if (token == "binc") {
          tokens >> token;
          binc = std::stoi(token);
        }
        if (token == "movetime") {
          tokens >> token;
          movetime = std::stoi(token);
        }
        if (token == "nodes") {
          tokens >> token;
          nodecount = std::stoi(token);
        }
      }
      if (movetime == 0) {
        int ourtime = current_pos.stm ? btime : wtime;
        int ourinc = current_pos.stm ? binc : winc;
        if (ourtime > 0) {
          movetime = (ourtime + 7 * ourinc) / 10;
        }
      }
      arena.clear();
      search_position(nn, arena, current_pos, game_hashes, movetime, nodecount,
                      threadcount, true);
    }
    if (token == "setoption") {
      tokens >> token;
      tokens >> token;
      if (token == "Hash") {
        tokens >> token;
        tokens >> token;
        arena.resize(std::stoi(token));
      }
      if (token == "Threads") {
        tokens >> token;
        tokens >> token;
        threadcount = std::stoi(token);
      }
    }
    if (token == "eval") {
      Move moves[maxmoves];
      int movecount = current_pos.generatemoves(moves);
      NNOutput raw_nn = nn.infer(current_pos);
      MCTSEval processed =
          parse_nn_output(raw_nn, moves, movecount, current_pos.stm);

      std::cout << "Value (STM): " << processed.qscore << "\n";
      float max_v_logit =
          std::max({raw_nn.value[0], raw_nn.value[1], raw_nn.value[2]});
      float exp_w = std::exp(raw_nn.value[0] - max_v_logit);
      float exp_d = std::exp(raw_nn.value[1] - max_v_logit);
      float exp_l = std::exp(raw_nn.value[2] - max_v_logit);
      float sum_v = exp_w + exp_d + exp_l;

      float prob_win = exp_w / sum_v;
      float prob_draw = exp_d / sum_v;
      float prob_loss = exp_l / sum_v;
      std::cout << "W: " << prob_win << " D: " << prob_draw
                << " L: " << prob_loss << "\n";
      struct MovePrior {
        Move m;
        float p;
      };
      std::vector<MovePrior> ranked_moves;
      for (int i = 0; i < movecount; i++) {
        ranked_moves.push_back({moves[i], processed.priors[i]});
      }

      std::sort(
          ranked_moves.begin(), ranked_moves.end(),
          [](const MovePrior &a, const MovePrior &b) { return a.p > b.p; });

      std::cout << "Policy:\n";
      for (int i = 0; i < movecount; i++) {
        std::cout << "  " << i + 1 << ". " << algebraic(ranked_moves[i].m)
                  << " | Prior: " << ranked_moves[i].p * 100.0f << "%\n";
      }
    }
  }
}

int main(int argc, char *argv[]) {
  initializeleaperattacks();
  initializemasks();
  initializerankattacks();
  initializezobrist();
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc > 1 && std::string(argv[1]) == "datagen") {
    if (argc < 5) {
      std::cerr << "Proper usage: ./(exe) datagen <game_count> "
                   "<nodes> <output_file>\n";
      return 0;
    }
    int num_games = atoi(argv[2]);
    int node_limit = atoi(argv[3]);
    std::string outputfile(argv[4]);

    std::cout << "Starting Data Generation Engine...\n";
    std::cout << "Nodes/Move: " << node_limit << "\n";
    std::cout << "Data Output: " << outputfile << ".data\n";

    NNEvaluator nn(NNFILE);
    generate_batched_selfplay_games(nn, outputfile, node_limit, num_games);

    return 0;
  } else {
    uci();
  }
  return 0;
}