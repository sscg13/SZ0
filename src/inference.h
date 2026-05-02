#include "position.h"

#include <onnxruntime_cxx_api.h>
#include <atomic>
#include <condition_variable>
#include <future>
#include <mutex>
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
  Ort::Env env;
  Ort::Session session{nullptr};
  Ort::MemoryInfo memory_info;

  // According to ONNX
  std::vector<const char *> input_names = {"in_0", "in_1"};
  std::vector<const char *> output_names = {"policy", "value"};

public:
  NNEvaluator(const char *model_path)
      : env(ORT_LOGGING_LEVEL_WARNING, "ShatranjZer0"),
        memory_info(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    // session_options.EnableProfiling("shatranj_profile.json");

#ifdef USE_CUDA
    try {
      OrtCUDAProviderOptionsV2 *cuda_options = nullptr;
      Ort::GetApi().CreateCUDAProviderOptions(&cuda_options);

      std::vector<const char *> keys = {"device_id", "arena_extend_strategy"};
      std::vector<const char *> values = {"0", "kSameAsRequested"};
      Ort::GetApi().UpdateCUDAProviderOptions(cuda_options, keys.data(),
                                              values.data(), keys.size());

      // Use the C-API directly to bypass the C++ reference bug.
      // The C++ session_options object implicitly converts to
      // OrtSessionOptions*
      OrtStatus *status =
          Ort::GetApi().SessionOptionsAppendExecutionProvider_CUDA_V2(
              session_options, cuda_options);

      if (status != nullptr) {
        fprintf(stderr, "Failed to append CUDA provider: %s\n",
                Ort::GetApi().GetErrorMessage(status));
        Ort::GetApi().ReleaseStatus(status);
      } else {
        printf("CUDA Execution Provider attached successfully.\n");
      }

      Ort::GetApi().ReleaseCUDAProviderOptions(cuda_options);
    } catch (const Ort::Exception &e) {
      fprintf(stderr, "Warning: CUDA exception caught. Error: %s\n", e.what());
    }
#endif

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
  void infer_packed(const std::vector<int32_t> &flat_pieces,
                    const std::vector<int32_t> &flat_halfmoves,
                    std::vector<NNOutput> &shared_results,
                    const std::vector<int> &batch_to_game_idx);
  std::vector<NNOutput> infer_dynamic_batch(
      const std::vector<int32_t> &flat_pieces,
      const std::vector<int32_t> &flat_halfmoves,
      int batch_size);
};

// Collects leaf positions from MCTS workers, batches them for GPU inference,
// and delivers results back via promise/future. A single dedicated thread
// runs run_inference_loop(); workers call evaluate() which blocks until the
// batch containing their request is processed.
class BatchEvaluator {
public:
  BatchEvaluator(NNEvaluator &nn, int max_batch_size)
      : nn_(nn), max_batch_(max_batch_size), stopped_(false) {}

  // Called by MCTS worker threads. Encodes the position, enqueues a request,
  // and blocks until the inference thread fulfills it.
  NNOutput evaluate(const Position &pos);

  // Called on a dedicated thread. Drains the queue in batches until stop()
  // is called and no pending requests remain.
  void run_inference_loop();

  // Signal the inference thread to exit after draining remaining requests.
  // Must be called after all worker threads have finished.
  void stop();

private:
  struct Request {
    int32_t board_data[64];
    int32_t halfmove;
    std::promise<NNOutput> promise;

    Request() = default;
    Request(Request &&) = default;
    Request &operator=(Request &&) = default;
  };

  void process_batch(std::vector<Request> batch);

  NNEvaluator &nn_;
  int max_batch_;
  bool stopped_;
  std::mutex mtx_;
  std::condition_variable cv_;
  std::vector<Request> pending_;
};