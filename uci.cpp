#include "node.h"
#include "position.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// clang-format off
std::string uciinfostring = 
"id name Shatranj Zer0\n"
"id author sscg13\n"
"option name UCI_Variant type combo default shatranj var shatranj\n"
"option name Hash type spin default 72 min 1 max 32768\n"
"uciok\n";
// clang-format on

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

void uci() {
  std::string ucicommand;
  Position current_pos;
  current_pos.initialize();
  TreeArena arena(3145728);
  std::vector<U64> game_hashes;

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
      int nodecount = 0;
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
          movetime = (ourtime + 9 * ourinc) / 10;
        }
      }
      arena.clear();
      std::atomic<bool> stop_search(false);
      auto start = std::chrono::steady_clock::now();
      U64 rollouts = 0;
      int seldepth = 0;
      U64 depthsum = 0;
      int nextprint = 400;
      while (!stop_search) {
        int depth = mcts_rollout(arena, current_pos, game_hashes);
        rollouts++;
        seldepth = std::max(seldepth, depth);
        depthsum += depth;
        if ((rollouts & 1023) == 0) {
          auto now = std::chrono::steady_clock::now();
          auto elapsed =
              std::chrono::duration_cast<std::chrono::milliseconds>(now - start)
                  .count();
          if (elapsed >= movetime && movetime > 0) {
            stop_search = true;
          } else if (elapsed >= nextprint) {
            printinfostring(arena, elapsed, depthsum / rollouts, seldepth);
            nextprint += 400;
          }
        }
        if (rollouts >= nodecount && nodecount > 0) {
          stop_search = true;
        }
        if (arena.active_nodes >= arena.max_nodes - 256) {
          std::cout << "info string reached memory limit" << std::endl;
          stop_search = true;
        }
      }
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
      auto now = std::chrono::steady_clock::now();
      auto elapsed =
          std::chrono::duration_cast<std::chrono::milliseconds>(now - start)
              .count();
      printinfostring(arena, elapsed, depthsum / rollouts, seldepth);
      if (arena.nodes[best_child_idx].move == Move()) {
        Move moves[maxmoves];
        current_pos.generatemoves(moves);
        std::cout << "bestmove " << algebraic(moves[0]) << "\n";
      }
      std::cout << "bestmove " << algebraic(arena.nodes[best_child_idx].move)
                << "\n";
    }
    if (token == "setoption") {
      tokens >> token;
      tokens >> token;
      if (token == "Hash") {
        tokens >> token;
        tokens >> token;
        arena.resize(std::stoi(token));
      }
    }
  }
}

int main() {
  initializeleaperattacks();
  initializemasks();
  initializerankattacks();
  initializezobrist();
  setvbuf(stdout, NULL, _IONBF, 0);
  uci();
  return 0;
}