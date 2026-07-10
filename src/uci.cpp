#include "datagen.h"
#include "inference.h"
#include "nnfile.h"
#include "node.h"
#include "position.h"
#include "search.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

std::unique_ptr<NNEvaluator> nn = nullptr;

void uci() {
  std::string ucicommand;
  Position current_pos;
  current_pos.initialize();
  TreeArena arena(defaultarenasize);
  std::vector<U64> game_hashes;
  std::string default_weights = get_latest_onnx_file();
  reload_network(default_weights);
  int threadcount = 1;
  int search_contempt_nscl = 0;
  bool particle_search = true;
  bool particle_greedy = true;
  int particle_eta_x100 = 150;

  while (std::getline(std::cin, ucicommand)) {
    std::stringstream tokens(ucicommand);
    std::string token;
    tokens >> token;
    if (token == "quit") {
      break;
    }
    if (token == "uci") {
      std::cout
          << "id name Shatranj Zer0\n"
          << "id author sscg13\n"
          << "option name UCI_Variant type combo default shatranj var "
             "shatranj\n"
          << "option name Threads type spin default 1 min 1 max 16\n"
          << "option name Hash type spin default 72 min 1 max 32768\n"
          << "option name WeightsFile type string default <autodiscover>\n"
          << "option name SearchContemptNodeLimit type spin default 0 min 0 "
             "max 255\n"
          << "option name ParticleSearch type check default true\n"
          << "option name ParticleEta type spin default 150 min 100 max "
             "400\n"
          << "option name ParticleGreedy type check default true\n"
          << "option name CPuct type spin default 200 min 25 max 800\n"
          << "uciok\n";
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
          movetime = (ourtime + 35 * ourinc) / 40;
        }
      }
      if (!nn) {
        std::cout << "info string ERROR: no network loaded, cannot search "
                     "(set WeightsFile or put a .onnx in the working "
                     "directory)\n";
        continue;
      }
      arena.clear();
      // Greedy mode (eta = -1 sentinel) takes the argmax of the improved
      // policy instead of sampling: same selection formula, no stochasticity
      // or importance weights.
      float particle_eta = 0.0f;
      if (particle_search) {
        particle_eta =
            particle_greedy ? -1.0f : particle_eta_x100 / 100.0f;
      }
      search_position(*nn, arena, current_pos, game_hashes, movetime, nodecount,
                      threadcount, true, search_contempt_nscl, particle_eta);
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
      if (token == "WeightsFile") {
        tokens >> token;
        tokens >> token;
        if (token != "<autodiscover>") {
          reload_network(token);
        }
      }
      if (token == "SearchContemptNodeLimit") {
        tokens >> token;
        tokens >> token;
        search_contempt_nscl = std::clamp(std::stoi(token), 0, 255);
      }
      if (token == "ParticleSearch") {
        tokens >> token;
        tokens >> token;
        particle_search = (token == "true");
      }
      if (token == "ParticleEta") {
        tokens >> token;
        tokens >> token;
        particle_eta_x100 = std::clamp(std::stoi(token), 100, 400);
      }
      if (token == "ParticleGreedy") {
        tokens >> token;
        tokens >> token;
        particle_greedy = (token == "true");
      }
      if (token == "CPuct") {
        tokens >> token;
        tokens >> token;
        cpuct_value = std::clamp(std::stoi(token), 25, 800) / 100.0f;
      }
    }
    if (token == "eval") {
      if (!nn) {
        std::cout << "info string ERROR: no network loaded, cannot eval\n";
        continue;
      }
      Move moves[maxmoves];
      int movecount = current_pos.generatemoves(moves);
      NNOutput raw_nn;
      try {
        raw_nn = nn->infer(current_pos);
      } catch (const Ort::Exception &e) {
        // Batched builds pin the session's batch dim to searchbatchsize, so
        // the batch-1 eval path is rejected by ORT instead of crashing us.
        std::cout << "info string ERROR: eval inference failed: " << e.what()
                  << "\n";
        continue;
      }
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
      std::cerr << "Proper usage: ./(exe) datagen <position_count> "
                   "<nodes> <output_file>\n";
      return 0;
    }
    std::string default_weights = get_latest_onnx_file();
    if (default_weights == "<empty>") {
      std::cerr << "Error: No onnx file found in directory\n";
      return 0;
    }

    int num_positions = atoi(argv[2]);
    int node_limit = atoi(argv[3]);
    std::string outputfile(argv[4]);

    std::cout << "Starting Data Generation Engine...\n";
    std::cout << "Using weights file: " << default_weights << "\n";
    std::cout << "Nodes/Move: " << node_limit << "\n";
    std::cout << "Position target: " << num_positions << "\n";
    std::cout << "Data Output: " << outputfile << ".data\n";

    try {
      NNEvaluator nn(default_weights.c_str(), datagenbatchsize);
      generate_batched_selfplay_games(nn, outputfile, node_limit,
                                      num_positions);
    } catch (const Ort::Exception &e) {
      std::cerr << "Error loading network " << default_weights << " at batch "
                << datagenbatchsize << ": " << e.what() << "\n";
      return 1;
    }

    return 0;
  } else {
    uci();
  }
  return 0;
}