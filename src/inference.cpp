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
    // 1. Use stack-allocated arrays instead of std::vector to avoid heap allocation
    int32_t board_data[64];
    int32_t halfmove_data[1] = { static_cast<int32_t>(pos.halfmovecount) };

    // 2. Fill the board data directly
    for (int i = 0; i < 64; ++i) {
        int perspective_square = pos.stm ? (i ^ 56) : i;
        board_data[perspective_square] = perspectivepiece(pos.pieces[i], pos.stm);
    }

    // 3. Define fixed shapes (static const to avoid re-initialization)
    static const int64_t board_shape[] = {1, 64};
    static const int64_t halfmove_shape[] = {1};

    // 4. Create tensors pointing to stack memory
    // Note: We use the version of CreateTensor that doesn't own the memory
    Ort::Value input_tensors[2] = {
        Ort::Value::CreateTensor<int32_t>(
            memory_info, board_data, 64, board_shape, 2),
        Ort::Value::CreateTensor<int32_t>(
            memory_info, halfmove_data, 1, halfmove_shape, 1)
    };

    // 5. Run inference
    // Assuming input_names and output_names are std::vector<const char*> members
    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, 
        input_names.data(), input_tensors, 2, 
        output_names.data(), output_names.size()
    );

    // 6. Extract results directly into the return struct
    NNOutput result;
    const float* policy_ptr = output_tensors[0].GetTensorData<float>();
    const float* value_ptr = output_tensors[1].GetTensorData<float>();

    std::copy(policy_ptr, policy_ptr + 4096, result.policy);
    std::copy(value_ptr, value_ptr + 3, result.value);

    return result;
}

void NNEvaluator::infer_packed(const std::vector<int32_t>& flat_pieces, 
                                const std::vector<int32_t>& flat_halfmoves,
                                std::vector<NNOutput>& shared_results,
                                const std::vector<int>& batch_to_game_idx) {
  std::array<int64_t, 2> board_shape{datagenbatchsize, 64};
  std::array<int64_t, 1> halfmove_shape{datagenbatchsize};

  // Wrap the exact memory the threads just packed (zero-copy)
  Ort::Value board_tensor = Ort::Value::CreateTensor<int32_t>(
      memory_info, const_cast<int32_t*>(flat_pieces.data()), flat_pieces.size(),
      board_shape.data(), board_shape.size());

  Ort::Value halfmove_tensor = Ort::Value::CreateTensor<int32_t>(
      memory_info, const_cast<int32_t*>(flat_halfmoves.data()), flat_halfmoves.size(),
      halfmove_shape.data(), halfmove_shape.size());

  std::vector<Ort::Value> input_tensors;
  input_tensors.push_back(std::move(board_tensor));
  input_tensors.push_back(std::move(halfmove_tensor));

  // Run DMA Inference
  auto output_tensors = session.Run(
      Ort::RunOptions{nullptr}, input_names.data(), input_tensors.data(),
      input_tensors.size(), output_names.data(), output_names.size());

  // Slice Outputs
  std::vector<NNOutput> results(datagenbatchsize);
  float *policy_ptr = output_tensors[0].GetTensorMutableData<float>();
  float *value_ptr = output_tensors[1].GetTensorMutableData<float>();

  for (int b = 0; b < datagenbatchsize; ++b) {
        int target_game = batch_to_game_idx[b];
        std::copy(policy_ptr + (b * 4096), policy_ptr + ((b + 1) * 4096), shared_results[target_game].policy);
        std::copy(value_ptr + (b * 3), value_ptr + ((b + 1) * 3), shared_results[target_game].value);
    }
}