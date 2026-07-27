#include "position.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <memory>
#include <mutex>
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
  Ort::Env env;
  Ort::Session session{nullptr};
  Ort::MemoryInfo memory_info;

#ifdef USE_CUDA
  // CUDA-graph capture path. ORT records the kernel sequence on the first Run
  // and replays it as a single launch afterwards, which removes the ~1.8 us
  // per-node dispatch cost that dominates small batches (~0.59 ms of a 1.07 ms
  // batch-32 inference). Capture requires a fixed shape and device-resident
  // IO at stable addresses, so every Run must use exactly graph_batch_ —
  // smaller requests are zero-padded up to it.
  bool cuda_graph_ = false;
  int graph_batch_ = 0;
  std::unique_ptr<Ort::MemoryInfo> cuda_info_;
  std::unique_ptr<Ort::Allocator> cuda_alloc_;
  std::unique_ptr<Ort::IoBinding> binding_;
  std::vector<Ort::Value> bound_; // owns the device tensors
  // Concurrent Run is safe on a plain session but not when every call shares
  // one set of device buffers. Uncontended in practice (datagen infers on
  // thread 0 only, search on one inference thread) — this just stops the UCI
  // `eval` command from corrupting an in-flight search batch.
  std::mutex graph_mutex_;
  void *d_board_ = nullptr;
  void *d_half_ = nullptr;
  void *d_policy_ = nullptr;
  void *d_value_ = nullptr;

  // Returns false if anything about capture setup fails; caller then keeps
  // using the ordinary Run path.
  bool setup_cuda_graph(int batch);
  std::vector<NNOutput> infer_bound(const std::vector<int32_t> &flat_pieces,
                                    const std::vector<int32_t> &flat_halfmoves,
                                    int real_count);
#endif

  // According to ONNX
  std::vector<const char *> input_names = {"in_0", "in_1"};
  std::vector<const char *> output_names = {"policy", "value"};

public:
  // fixed_batch > 0 pins a symbolic batch dimension named "batch" (if the
  // model has one) to that size before the session is created, so ORT
  // optimizes the graph as fully static — fusions, memory planning, and CUDA
  // graph capture behave as with a fixed-batch export. Models with
  // hard-coded batch dims are unaffected (the override matches nothing).
  NNEvaluator(const char *model_path, int fixed_batch = 0)
      : env(ORT_LOGGING_LEVEL_WARNING, "ShatranjZer0"),
        memory_info(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetInterOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    if (fixed_batch > 0) {
      // Older ORT C++ headers lack the SessionOptions wrapper for this, so
      // call the C API directly (same pattern as the CUDA options below).
      Ort::ThrowOnError(Ort::GetApi().AddFreeDimensionOverrideByName(
          session_options, "batch", fixed_batch));
    }
    // session_options.EnableProfiling("shatranj_profile.json");

    // Set SZ0_DUMP_OPTIMIZED=<path> to write the graph ORT actually executes,
    // after provider partitioning and fusion. This is the only faithful way to
    // see it: the pip onnxruntime is a different build (and usually CPU-only)
    // than the CUDA-enabled library linked here, and fusion differs by both
    // provider and version. Inspect the result with src/nn/inspect_graph.py
    // --raw, which only reads the file.
    if (const char *dump_path = std::getenv("SZ0_DUMP_OPTIMIZED")) {
#ifdef _WIN32
      std::string dump_str(dump_path);
      std::wstring dump_wide(dump_str.begin(), dump_str.end());
      session_options.SetOptimizedModelFilePath(dump_wide.c_str());
#else
      session_options.SetOptimizedModelFilePath(dump_path);
#endif
      printf("Dumping ORT-optimized graph to %s\n", dump_path);
    }

#ifdef USE_CUDA
    try {
      OrtCUDAProviderOptionsV2 *cuda_options = nullptr;
      Ort::GetApi().CreateCUDAProviderOptions(&cuda_options);

      std::vector<const char *> keys = {"device_id", "arena_extend_strategy"};
      std::vector<const char *> values = {"0", "kSameAsRequested"};
      // SZ0_CUDA_GRAPH=1 opts into graph capture. Off by default: capture
      // pins the batch shape for the session's lifetime, so it is a behaviour
      // change, not just a speedup.
      const char *want_graph = std::getenv("SZ0_CUDA_GRAPH");
      if (want_graph && want_graph[0] == '1' && fixed_batch > 0) {
        keys.push_back("enable_cuda_graph");
        values.push_back("1");
        cuda_graph_ = true;
        graph_batch_ = fixed_batch;
      }
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

#ifdef USE_CUDA
    // Must happen before the warmup Run: that Run is what ORT captures.
    if (cuda_graph_ && !setup_cuda_graph(graph_batch_)) {
      fprintf(stderr, "CUDA graph setup failed; using the standard Run path\n");
      cuda_graph_ = false;
    }
#endif

    // Warm the session with one dummy batch so provider autotuning (cuDNN /
    // cuBLAS algorithm selection) and arena allocation happen at load time,
    // not on the clock during the first timed search.
    int warm_batch = (fixed_batch > 0) ? fixed_batch : 1;
    std::vector<int32_t> warm_pieces(static_cast<size_t>(warm_batch) * 64, 0);
    std::vector<int32_t> warm_halfmoves(warm_batch, 0);
    infer_dynamic_batch(warm_pieces, warm_halfmoves, warm_batch);
  }

  NNOutput infer(const Position &pos);
  void infer_packed(const std::vector<int32_t> &flat_pieces,
                    const std::vector<int32_t> &flat_halfmoves,
                    std::vector<NNOutput> &shared_results,
                    const std::vector<int> &batch_to_game_idx);
  std::vector<NNOutput>
  infer_dynamic_batch(const std::vector<int32_t> &flat_pieces,
                      const std::vector<int32_t> &flat_halfmoves,
                      int batch_size);
};

// Collects leaf positions from MCTS workers, batches them for GPU inference,
// and delivers results back via promise/future. A single dedicated thread
// runs run_inference_loop(); workers call submit() to enqueue a position and
// receive a future they can poll without blocking.
class BatchEvaluator {
public:
  // max_batch_size must match the ONNX model's exported batch dimension.
  explicit BatchEvaluator(NNEvaluator &nn, int max_batch_size)
      : nn_(nn), max_batch_(max_batch_size), stopped_(false) {}

  // Non-blocking: encode the position, enqueue a request, and return a future.
  // The future is fulfilled by the inference thread once its batch is
  // processed.
  std::future<NNOutput> submit(const Position &pos);

  // Blocking convenience wrapper (used by datagen / single-threaded callers).
  NNOutput evaluate(const Position &pos) { return submit(pos).get(); }

  // Called on a dedicated thread. Fires batches when max_batch_ requests are
  // queued, or after a short timeout for partial batches near end-of-search.
  void run_inference_loop();

  // Signal the inference thread to drain remaining requests and exit.
  // Must be called after all worker threads have finished submitting.
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
