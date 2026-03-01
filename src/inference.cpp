#include "inference.h"

#include <algorithm>
#include <cmath>

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

std::vector<NNOutput> NNEvaluator::infer_batch(const std::vector<Position> &positions) {
  int64_t batch_size = positions.size();
  
  // Return early if the queue/batch is empty
  if (batch_size == 0) {
    return {};
  }

  std::vector<I32> batched_pieces(batch_size * 64);
  std::vector<I32> batched_halfmoves(batch_size);

  for (int64_t b = 0; b < batch_size; ++b) {
    const Position &pos = positions[b];
    
    for (int i = 0; i < 64; ++i) {
      int perspective_square = pos.stm ? (i ^ 56) : i;
      // Use flat indexing: (batch_index * 64) + square_index
      batched_pieces[(b * 64) + perspective_square] = perspectivepiece(pos.pieces[i], pos.stm);
    }
    batched_halfmoves[b] = pos.halfmovecount;
  }

  // 3. Define the new batched shapes
  std::array<int64_t, 2> board_shape{batch_size, 64};
  std::array<int64_t, 1> halfmove_shape{batch_size}; // See note below if ONNX complains

  // 4. Create tensors pointing to our flat vectors
  Ort::Value board_tensor = Ort::Value::CreateTensor<int32_t>(
      memory_info, batched_pieces.data(), batched_pieces.size(),
      board_shape.data(), board_shape.size());

  Ort::Value halfmove_tensor = Ort::Value::CreateTensor<int32_t>(
      memory_info, batched_halfmoves.data(), batched_halfmoves.size(),
      halfmove_shape.data(), halfmove_shape.size());

  std::vector<Ort::Value> input_tensors;
  input_tensors.push_back(std::move(board_tensor));
  input_tensors.push_back(std::move(halfmove_tensor));

  // 5. Run inference on the whole batch at once!
  auto output_tensors = session.Run(
      Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(),
      input_tensors.size(), output_names.data(), output_names.size());

  // 6. Extract and slice the outputs
  std::vector<NNOutput> results(batch_size);
  float *policy_ptr = output_tensors[0].GetTensorMutableData<float>();
  float *value_ptr = output_tensors[1].GetTensorMutableData<float>();

  for (int64_t b = 0; b < batch_size; ++b) {
    // Copy 4096 policy floats and 3 value floats for this specific board
    std::copy(policy_ptr + (b * 4096), policy_ptr + ((b + 1) * 4096), results[b].policy);
    std::copy(value_ptr + (b * 3), value_ptr + ((b + 1) * 3), results[b].value);
  }

  return results;
}

NNOutput NNEvaluator::infer(const Position &pos) {
  std::vector<Position> dummy_batch;
  dummy_batch.push_back(pos);
  
  std::vector<NNOutput> batched_results = infer_batch(dummy_batch);
  
  return batched_results[0]; // Return the first (and only) result
}