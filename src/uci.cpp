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
  NNEvaluator nn("sz0_epoch9.onnx");
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
  }
}

int main(int argc, char *argv[]) {
  initializeleaperattacks();
  initializemasks();
  initializerankattacks();
  initializezobrist();
  setvbuf(stdout, NULL, _IONBF, 0);
  if (argc > 1 && std::string(argv[1]) == "datagen") {
    if (argc != 6) {
      std::cerr << "Proper usage: ./(exe) datagen <game_count> <concurrency> "
                   "<nodes> <output_file>\n";
      return 0;
    }
    int num_games = atoi(argv[2]);
    int concurrency = atoi(argv[3]);
    int node_limit = atoi(argv[4]);
    std::string outputfile(argv[5]);

    std::cout << "Starting Data Generation Engine...\n";
    std::cout << "Concurrency: " << concurrency << "\n";
    std::cout << "Nodes/Move: " << node_limit << "\n";
    std::cout << "Data Output: " << outputfile << ".data\n";

    NNEvaluator nn("sz0_epoch9.onnx");
    generate_batched_selfplay_games(nn, outputfile, node_limit, concurrency, num_games);

    return 0;
  } else {
    uci();
  }
  return 0;
}