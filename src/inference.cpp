#include "inference.h"

#include <algorithm>
#include <chrono>
#include <cmath>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

MCTSEval parse_nn_output(const NNOutput &raw_nn, const Move *moves,
                         int movecount, bool stm) {
  MCTSEval result;
  result.priors.resize(movecount);

  // Numerically stable softmax
  float max_v_logit =
      std::max({raw_nn.value[0], raw_nn.value[1], raw_nn.value[2]});
  float exp_w = std::exp(raw_nn.value[0] - max_v_logit);
  float exp_d = std::exp(raw_nn.value[1] - max_v_logit);
  float exp_l = std::exp(raw_nn.value[2] - max_v_logit);
  float sum_v = exp_w + exp_d + exp_l;

  float prob_win = exp_w / sum_v;
  float prob_loss = exp_l / sum_v;

  result.qscore = prob_win - prob_loss;
  if (movecount == 0) {
    return result;
  }

  std::vector<float> move_logits(movecount);
  float max_logit = -1e9f;

  // Filter legal moves and apply perspective
  for (int i = 0; i < movecount; i++) {
    int from_sq = moves[i].from() ^ (stm ? 56 : 0);
    int to_sq = moves[i].to() ^ (stm ? 56 : 0);

    int index = from_sq * 64 + to_sq;
    float logit = raw_nn.policy[index];

    move_logits[i] = logit;
    if (logit > max_logit) {
      max_logit = logit;
    }
  }

  // Numerically stable softmax
  float sum_p = 0.0f;
  for (int i = 0; i < movecount; i++) {
    result.priors[i] = std::exp(move_logits[i] - max_logit);
    sum_p += result.priors[i];
  }

  for (int i = 0; i < movecount; i++) {
    result.priors[i] /= sum_p;
  }

  return result;
}

NNOutput NNEvaluator::infer(const Position &pos) {
  int32_t board_data[64];
  int32_t halfmove_data[1] = {clamp_halfmove(pos.halfmovecount)};

  for (int i = 0; i < 64; ++i) {
    int perspective_square = pos.stm ? (i ^ 56) : i;
    board_data[perspective_square] =
        perspectivepiece(pos.pieces[i], pos.stm) + 13 * perspective_square;
  }

#ifdef USE_CUDA
  // Capture pinned the shape, so a batch-1 Run is no longer possible: pad up
  // to graph_batch_ and keep the first result. Only the `eval` command and
  // sequential (non-batched) search land here, so the waste is irrelevant.
  if (cuda_graph_) {
    std::vector<int32_t> pieces(static_cast<size_t>(graph_batch_) * 64, 0);
    std::vector<int32_t> halfmoves(graph_batch_, 0);
    std::copy(board_data, board_data + 64, pieces.begin());
    halfmoves[0] = halfmove_data[0];
    return infer_bound(pieces, halfmoves, 1)[0];
  }
#endif

  static const int64_t board_shape[] = {1, 64};
  static const int64_t halfmove_shape[] = {1};

  Ort::Value input_tensors[2] = {
      Ort::Value::CreateTensor<int32_t>(memory_info, board_data, 64,
                                        board_shape, 2),
      Ort::Value::CreateTensor<int32_t>(memory_info, halfmove_data, 1,
                                        halfmove_shape, 1)};

  auto output_tensors =
      session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors,
                  2, output_names.data(), output_names.size());

  NNOutput result;
  const float *policy_ptr = output_tensors[0].GetTensorData<float>();
  const float *value_ptr = output_tensors[1].GetTensorData<float>();

  std::copy(policy_ptr, policy_ptr + 4096, result.policy);
  std::copy(value_ptr, value_ptr + 3, result.value);

  return result;
}

#ifdef USE_CUDA
bool NNEvaluator::setup_cuda_graph(int batch) {
  try {
    cuda_info_ = std::make_unique<Ort::MemoryInfo>(
        "Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    cuda_alloc_ = std::make_unique<Ort::Allocator>(session, *cuda_info_);

    const size_t n = static_cast<size_t>(batch);
    d_board_ = cuda_alloc_->Alloc(n * 64 * sizeof(int32_t));
    d_half_ = cuda_alloc_->Alloc(n * sizeof(int32_t));
    d_policy_ = cuda_alloc_->Alloc(n * 4096 * sizeof(float));
    d_value_ = cuda_alloc_->Alloc(n * 3 * sizeof(float));
    if (!d_board_ || !d_half_ || !d_policy_ || !d_value_)
      return false;

    // Policy is [batch, 64, 64]; the engine treats it as 4096 flat per position.
    std::array<int64_t, 2> board_shape{batch, 64};
    std::array<int64_t, 1> half_shape{batch};
    std::array<int64_t, 3> policy_shape{batch, 64, 64};
    std::array<int64_t, 2> value_shape{batch, 3};

    bound_.clear();
    bound_.push_back(Ort::Value::CreateTensor<int32_t>(
        *cuda_info_, static_cast<int32_t *>(d_board_), n * 64,
        board_shape.data(), board_shape.size()));
    bound_.push_back(Ort::Value::CreateTensor<int32_t>(
        *cuda_info_, static_cast<int32_t *>(d_half_), n, half_shape.data(),
        half_shape.size()));
    bound_.push_back(Ort::Value::CreateTensor<float>(
        *cuda_info_, static_cast<float *>(d_policy_), n * 4096,
        policy_shape.data(), policy_shape.size()));
    bound_.push_back(Ort::Value::CreateTensor<float>(
        *cuda_info_, static_cast<float *>(d_value_), n * 3, value_shape.data(),
        value_shape.size()));

    binding_ = std::make_unique<Ort::IoBinding>(session);
    binding_->BindInput(input_names[0], bound_[0]);
    binding_->BindInput(input_names[1], bound_[1]);
    binding_->BindOutput(output_names[0], bound_[2]);
    binding_->BindOutput(output_names[1], bound_[3]);

    printf("CUDA graph capture enabled at batch %d\n", batch);
    return true;
  } catch (const Ort::Exception &e) {
    fprintf(stderr, "CUDA graph setup error: %s\n", e.what());
    return false;
  }
}

// SZ0_TIME_IO=1 breaks a captured-graph inference into phases. Only `run` is
// GPU work; every other phase is host-side cost that a persistent pinned
// staging buffer would remove or shrink, so the split says directly whether
// that change is worth making. Prints a running mean to stderr every
// SZ0_TIME_IO_EVERY calls (default 1000) and then resets, so a long datagen run
// shows drift rather than one lifetime average.
namespace {
using clk = std::chrono::steady_clock;

// Elapsed since *t, and advance *t to now. One clock read per phase boundary.
inline double us_since(clk::time_point *t) {
  auto now = clk::now();
  double d = std::chrono::duration<double, std::micro>(now - *t).count();
  *t = now;
  return d;
}

struct IoTimer {
  bool on = false;
  long long report_every = 1000;
  long long calls = 0;
  // Phases, in execution order.
  double h2d = 0, run = 0, alloc_results = 0, alloc_stage = 0, d2h = 0,
         copy_out = 0;
  // Charged by infer_packed: the extra scatter the graph path pays that the
  // ordinary Run path does not (it copies straight out of the ORT tensor).
  double scatter = 0;

  IoTimer() {
    const char *v = std::getenv("SZ0_TIME_IO");
    on = (v && v[0] == '1');
    if (const char *e = std::getenv("SZ0_TIME_IO_EVERY")) {
      long long parsed = atoll(e);
      if (parsed > 0)
        report_every = parsed;
    }
  }

  void tick(int batch) {
    if (++calls < report_every)
      return;
    double n = static_cast<double>(calls);
    double host = h2d + alloc_results + alloc_stage + d2h + copy_out + scatter;
    double total = host + run;
    fprintf(stderr,
            "[io] batch %d n=%lld | h2d %.0f  run %.0f  results %.0f  "
            "stage %.0f  d2h %.0f  copy %.0f  scatter %.0f  "
            "| total %.0f us, host-side %.1f%%\n",
            batch, calls, h2d / n, run / n, alloc_results / n, alloc_stage / n,
            d2h / n, copy_out / n, scatter / n, total / n,
            100.0 * host / total);
    calls = 0;
    h2d = run = alloc_results = alloc_stage = d2h = copy_out = scatter = 0;
  }
};

IoTimer io_timer;

// Set by infer_packed so infer_bound defers reporting until the scatter that
// follows it has been charged. Datagen calls infer_packed from one thread only
// and search never calls it at all, so this needs no synchronisation.
bool io_in_packed = false;
} // namespace

// Runs the captured graph. Inputs are padded to graph_batch_ by the caller;
// only the first real_count results are returned.
std::vector<NNOutput>
NNEvaluator::infer_bound(const std::vector<int32_t> &flat_pieces,
                         const std::vector<int32_t> &flat_halfmoves,
                         int real_count) {
  std::lock_guard<std::mutex> lock(graph_mutex_);
  const size_t n = static_cast<size_t>(graph_batch_);
  const bool timing = io_timer.on;
  clk::time_point t{};
  if (timing)
    t = clk::now();

  cudaMemcpy(d_board_, flat_pieces.data(), n * 64 * sizeof(int32_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_half_, flat_halfmoves.data(), n * sizeof(int32_t),
             cudaMemcpyHostToDevice);
  if (timing)
    io_timer.h2d += us_since(&t);

  session.Run(Ort::RunOptions{nullptr}, *binding_);
  binding_->SynchronizeOutputs();
  if (timing)
    io_timer.run += us_since(&t);

  // Value-initialized, so this zero-fills real_count * 16 KB before anything
  // is written to it.
  std::vector<NNOutput> results(real_count);
  if (timing)
    io_timer.alloc_results += us_since(&t);

  const size_t policy_floats = static_cast<size_t>(real_count) * 4096;
  const size_t value_floats = static_cast<size_t>(real_count) * 3;
  std::vector<float> policy(policy_floats), value(value_floats);
  if (timing)
    io_timer.alloc_stage += us_since(&t);

  cudaMemcpy(policy.data(), d_policy_, policy_floats * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(value.data(), d_value_, value_floats * sizeof(float),
             cudaMemcpyDeviceToHost);
  if (timing)
    io_timer.d2h += us_since(&t);

  for (int b = 0; b < real_count; ++b) {
    std::copy(policy.begin() + b * 4096, policy.begin() + (b + 1) * 4096,
              results[b].policy);
    std::copy(value.begin() + b * 3, value.begin() + (b + 1) * 3,
              results[b].value);
  }
  if (timing) {
    io_timer.copy_out += us_since(&t);
    // infer_packed adds its scatter before reporting; search has none, so
    // report here only when nobody else will.
    if (!io_in_packed)
      io_timer.tick(graph_batch_);
  }
  return results;
}
#endif

std::vector<NNOutput>
NNEvaluator::infer_dynamic_batch(const std::vector<int32_t> &flat_pieces,
                                 const std::vector<int32_t> &flat_halfmoves,
                                 int batch_size) {
#ifdef USE_CUDA
  if (cuda_graph_ && batch_size == graph_batch_)
    return infer_bound(flat_pieces, flat_halfmoves, batch_size);
#endif
  std::array<int64_t, 2> board_shape{batch_size, 64};
  std::array<int64_t, 1> halfmove_shape{batch_size};

  Ort::Value board_tensor = Ort::Value::CreateTensor<int32_t>(
      memory_info, const_cast<int32_t *>(flat_pieces.data()),
      flat_pieces.size(), board_shape.data(), board_shape.size());

  Ort::Value halfmove_tensor = Ort::Value::CreateTensor<int32_t>(
      memory_info, const_cast<int32_t *>(flat_halfmoves.data()),
      flat_halfmoves.size(), halfmove_shape.data(), halfmove_shape.size());

  Ort::Value input_tensors[2] = {std::move(board_tensor),
                                 std::move(halfmove_tensor)};

  auto output_tensors =
      session.Run(Ort::RunOptions{nullptr}, input_names.data(), input_tensors,
                  2, output_names.data(), output_names.size());

  std::vector<NNOutput> results(batch_size);
  const float *policy_ptr = output_tensors[0].GetTensorData<float>();
  const float *value_ptr = output_tensors[1].GetTensorData<float>();

  for (int b = 0; b < batch_size; ++b) {
    std::copy(policy_ptr + b * 4096, policy_ptr + (b + 1) * 4096,
              results[b].policy);
    std::copy(value_ptr + b * 3, value_ptr + (b + 1) * 3, results[b].value);
  }

  return results;
}

void NNEvaluator::infer_packed(const std::vector<int32_t> &flat_pieces,
                               const std::vector<int32_t> &flat_halfmoves,
                               std::vector<NNOutput> &shared_results,
                               const std::vector<int> &batch_to_game_idx) {
#ifdef USE_CUDA
  if (cuda_graph_ && graph_batch_ == datagenbatchsize) {
    // NOTE: this path copies the policy block one more time than the ordinary
    // Run path below, which scatters straight out of the ORT output tensor.
    // At batch 284 that is an extra ~4.65 MB per call. SZ0_TIME_IO=1 prices it
    // as the `scatter` column.
    io_in_packed = io_timer.on;
    auto results = infer_bound(flat_pieces, flat_halfmoves, datagenbatchsize);
    clk::time_point t{};
    if (io_timer.on)
      t = clk::now();
    for (int b = 0; b < datagenbatchsize; ++b)
      shared_results[batch_to_game_idx[b]] = results[b];
    if (io_timer.on) {
      io_timer.scatter += us_since(&t);
      io_in_packed = false;
      io_timer.tick(datagenbatchsize);
    }
    return;
  }
#endif

  std::array<int64_t, 2> board_shape{datagenbatchsize, 64};
  std::array<int64_t, 1> halfmove_shape{datagenbatchsize};

  Ort::Value board_tensor = Ort::Value::CreateTensor<int32_t>(
      memory_info, const_cast<int32_t *>(flat_pieces.data()),
      flat_pieces.size(), board_shape.data(), board_shape.size());

  Ort::Value halfmove_tensor = Ort::Value::CreateTensor<int32_t>(
      memory_info, const_cast<int32_t *>(flat_halfmoves.data()),
      flat_halfmoves.size(), halfmove_shape.data(), halfmove_shape.size());

  std::vector<Ort::Value> input_tensors;
  input_tensors.push_back(std::move(board_tensor));
  input_tensors.push_back(std::move(halfmove_tensor));

  auto output_tensors = session.Run(
      Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(),
      input_tensors.size(), output_names.data(), output_names.size());

  // Slice Outputs
  std::vector<NNOutput> results(datagenbatchsize);
  float *policy_ptr = output_tensors[0].GetTensorMutableData<float>();
  float *value_ptr = output_tensors[1].GetTensorMutableData<float>();

  for (int b = 0; b < datagenbatchsize; ++b) {
    int target_game = batch_to_game_idx[b];
    std::copy(policy_ptr + (b * 4096), policy_ptr + ((b + 1) * 4096),
              shared_results[target_game].policy);
    std::copy(value_ptr + (b * 3), value_ptr + ((b + 1) * 3),
              shared_results[target_game].value);
  }
}

std::future<NNOutput> BatchEvaluator::submit(const Position &pos) {
  int32_t board_data[64];
  int32_t halfmove = clamp_halfmove(pos.halfmovecount);

  for (int i = 0; i < 64; ++i) {
    int ps = pos.stm ? (i ^ 56) : i;
    board_data[ps] = perspectivepiece(pos.pieces[i], pos.stm) + 13 * ps;
  }

  std::promise<NNOutput> promise;
  std::future<NNOutput> future = promise.get_future();

  {
    std::lock_guard<std::mutex> lk(mtx_);
    pending_.emplace_back();
    Request &req = pending_.back();
    std::copy(board_data, board_data + 64, req.board_data);
    req.halfmove = halfmove;
    req.promise = std::move(promise);
    cv_.notify_one();
  }

  return future;
}

void BatchEvaluator::process_batch(std::vector<Request> batch) {
  int real_count = static_cast<int>(batch.size());

  // Always allocate max_batch_ slots so the tensor shape matches the model's
  // hardcoded batch dimension. Unused slots are zero-padded.
  std::vector<int32_t> flat_pieces(max_batch_ * 64, 0);
  std::vector<int32_t> flat_halfmoves(max_batch_, 0);

  for (int i = 0; i < real_count; ++i) {
    std::copy(batch[i].board_data, batch[i].board_data + 64,
              flat_pieces.data() + i * 64);
    flat_halfmoves[i] = batch[i].halfmove;
  }

  auto results =
      nn_.infer_dynamic_batch(flat_pieces, flat_halfmoves, max_batch_);

  for (int i = 0; i < real_count; ++i) {
    batch[i].promise.set_value(std::move(results[i]));
  }
}

void BatchEvaluator::run_inference_loop() {
  using namespace std::chrono_literals;
  while (true) {
    std::vector<Request> batch;
    {
      std::unique_lock<std::mutex> lk(mtx_);
      // Fire when the full batch is ready; fall back after 500 µs so partial
      // batches aren't stranded when workers hit terminals or search ends.
      cv_.wait_for(lk, 500us, [&] {
        return static_cast<int>(pending_.size()) >= max_batch_ || stopped_;
      });
      if (pending_.empty() && stopped_)
        break;
      // Cap at max_batch_ so we never exceed the model's tensor dimension.
      int take = std::min(static_cast<int>(pending_.size()), max_batch_);
      batch.reserve(take);
      for (int i = 0; i < take; ++i)
        batch.push_back(std::move(pending_[i]));
      pending_.erase(pending_.begin(), pending_.begin() + take);
    }

    if (!batch.empty())
      process_batch(std::move(batch));
  }
}

void BatchEvaluator::stop() {
  {
    std::lock_guard<std::mutex> lk(mtx_);
    stopped_ = true;
  }
  cv_.notify_all();
}