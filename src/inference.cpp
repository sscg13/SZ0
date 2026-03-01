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

NNOutput NNEvaluator::infer(const Position &pos) {
  std::array<I32, 64> perspective_pieces;
  std::array<I32, 1> halfmove;

  for (int i = 0; i < 64; ++i) {
    perspective_pieces[i] = perspectivepiece(pos.pieces[i], pos.stm);
  }
  halfmove[0] = pos.halfmovecount;

  std::array<int64_t, 2> board_shape{1, 64};
  std::array<int64_t, 1> halfmove_shape{1};

  Ort::Value board_tensor = Ort::Value::CreateTensor<int32_t>(
      memory_info, perspective_pieces.data(), perspective_pieces.size(),
      board_shape.data(), board_shape.size());

  Ort::Value halfmove_tensor = Ort::Value::CreateTensor<int32_t>(
      memory_info, halfmove.data(), halfmove.size(), halfmove_shape.data(),
      halfmove_shape.size());

  std::vector<Ort::Value> input_tensors;
  input_tensors.push_back(std::move(board_tensor));
  input_tensors.push_back(std::move(halfmove_tensor));

  auto output_tensors = session.Run(
      Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(),
      input_tensors.size(), output_names.data(), output_names.size());

  NNOutput result;

  // index 0 = div_out_6 (Policy, 64x64), index 1 = out_37 (Value, 3)
  float *policy_ptr = output_tensors[0].GetTensorMutableData<float>();
  float *value_ptr = output_tensors[1].GetTensorMutableData<float>();

  std::copy(policy_ptr, policy_ptr + 4096, result.policy);
  std::copy(value_ptr, value_ptr + 3, result.value);

  return result;
}