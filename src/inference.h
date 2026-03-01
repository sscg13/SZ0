#include "position.h"

#include <onnxruntime_cxx_api.h>
#include <vector>

#pragma once

// ONNX raw outputs
struct NNOutput {
  float value[3];
  float policy[4096];
};

// Processed outputs
struct MCTSEval {
  float qscore;
  std::vector<float> priors;
};

MCTSEval parse_nn_output(const NNOutput &raw_nn, const Move *moves,
                         int movecount, bool stm);

class NNEvaluator {
private:
  Ort::Env env;
  Ort::Session session{nullptr};
  Ort::MemoryInfo memory_info;

  // According to ONNX
  std::vector<const char *> input_names = {"in_0", "in_1"};
  std::vector<const char *> output_names = {"div_out_6", "out_37"};

public:
  NNEvaluator(const char *model_path)
      : env(ORT_LOGGING_LEVEL_WARNING, "ShatranjZer0"),
        memory_info(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);

    // TODO: CUDA
    /*
    try {
        OrtCUDAProviderOptionsV2* cuda_options = nullptr;
        Ort::GetApi().CreateCUDAProviderOptions(&cuda_options);

        // Optional: Set specific GPU options (Device 0, use Arena, etc.)
        std::vector<const char*> keys = {"device_id", "arena_extend_strategy"};
        std::vector<const char*> values = {"0", "kSameAsRequested"};
        Ort::GetApi().UpdateCUDAProviderOptions(cuda_options, keys.data(), values.data(), keys.size());

        session_options.AppendExecutionProvider_CUDA_V2(cuda_options);
        
        // Always release the options pointer after appending
        Ort::GetApi().ReleaseCUDAProviderOptions(cuda_options);
    } catch (const Ort::Exception& e) {
        fprintf(stderr, "Warning: CUDA not available, falling back to CPU. Error: %s\n", e.what());
    }*/

#ifdef _WIN32
    // Windows requires wide strings (wchar_t) for paths
    std::string str_path(model_path);
    std::wstring wide_path(str_path.begin(), str_path.end());
    session = Ort::Session(env, wide_path.c_str(), session_options);
#else
    // Linux uses standard char arrays
    session = Ort::Session(env, model_path, session_options);
#endif
  }

  NNOutput infer(const Position &pos);
  std::vector<NNOutput> infer_batch(const std::vector<Position> &positions);
};