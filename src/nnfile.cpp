#include "inference.h"

#include <filesystem>
#include <iostream>
#include <string>

extern std::unique_ptr<NNEvaluator> nn;

std::string get_latest_onnx_file(const std::string &directory) {
  std::string latest_file = "<empty>";
  std::filesystem::file_time_type latest_time =
      std::filesystem::file_time_type::min();

  if (!std::filesystem::exists(directory)) {
    return latest_file;
  }

  for (const auto &entry : std::filesystem::directory_iterator(directory)) {
    if (entry.is_regular_file() && entry.path().extension() == ".onnx") {
      auto time = std::filesystem::last_write_time(entry);
      if (time > latest_time) {
        latest_time = time;
        latest_file = entry.path().string();
        if (latest_file.rfind("./", 0) == 0 ||
            latest_file.rfind(".\\", 0) == 0) {
          latest_file = latest_file.substr(2);
        }
      }
    }
  }
  return latest_file;
}

void reload_network(const std::string &path) {
  if (path == "<empty>") {
    std::cout << "info string WARNING: no .onnx file found in working "
                 "directory, no network loaded\n";
    return;
  }

  std::cout << "info string Loading network: " << path << "...\n";
  nn.reset();

  try {
    // Pin a symbolic batch dim to the size this build actually feeds the
    // session: the search batch in batched builds, single positions
    // otherwise. Fixed-batch exports are unaffected.
#if defined(USE_CUDA) || defined(USE_BATCHED_SEARCH)
    nn = std::make_unique<NNEvaluator>(path.c_str(), searchbatchsize);
#else
    nn = std::make_unique<NNEvaluator>(path.c_str(), 1);
#endif
    std::cout << "info string Network loaded successfully.\n";
  } catch (const Ort::Exception &e) {
    // stdout, not stderr: GUIs/cutechess only log the UCI stream, so a
    // stderr-only error leaves no trace of why the engine has no network.
    std::cout << "info string Error loading network: " << e.what() << "\n";
  }
}