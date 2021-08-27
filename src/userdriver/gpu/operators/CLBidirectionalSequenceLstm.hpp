#pragma once

#include "userdriver/gpu/common/CLParameter.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLActivation.hpp"

namespace enn {
namespace ud {
namespace gpu {

struct BidirectionalSequenceLstmParameters : public Parameters {
    float cell_clip = 0.0f;
    float proj_clip = 0.0f;
    bool merge_outputs = false;
    bool time_major = true;
    bool androidNN = false;
    bool force_alloc_state_at_init = false;
    bool weights_as_input = false;
    std::shared_ptr<ActivationInfo> activation_info = std::make_shared<ActivationInfo>();
};

class CLBidirectionalSequenceLstm {
public:
    CLBidirectionalSequenceLstm(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);
    Status initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                      const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                      const std::shared_ptr<Parameters> parameters);
    Status execute();
    Status release();

private:
    // 1. Runtime context
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;

    // 2. Operator resource
    uint32_t max_time_;
    uint32_t n_batch_;
    uint32_t n_input_;
    uint32_t n_fw_output_;
    uint32_t n_bw_output_;
    uint32_t n_fw_cell_;
    uint32_t n_bw_cell_;
    uint32_t fw_output_batch_leading_dim_;
    uint32_t bw_output_batch_leading_dim_;
    uint32_t bw_output_offset_;
    uint32_t aux_input_size_;
    NDims fw_scratch_dims_;
    NDims bw_scratch_dims_;
    ActivationInfo activation_info_;
    std::shared_ptr<BidirectionalSequenceLstmParameters> parameters_;

    std::shared_ptr<CLTensor> fw_in_ = nullptr;
    std::shared_ptr<CLTensor> fw_out_ = nullptr;
    std::shared_ptr<CLTensor> bw_out_ = nullptr;
    std::shared_ptr<CLTensor> fw_input_to_input_weights_ = nullptr;
    std::shared_ptr<CLTensor> fw_input_to_forget_weights_ = nullptr;
    std::shared_ptr<CLTensor> fw_input_to_cell_weights_ = nullptr;
    std::shared_ptr<CLTensor> fw_input_to_output_weights_ = nullptr;
    std::shared_ptr<CLTensor> fw_recurrent_to_input_weights_ = nullptr;
    std::shared_ptr<CLTensor> fw_recurrent_to_forget_weights_ = nullptr;
    std::shared_ptr<CLTensor> fw_recurrent_to_cell_weights_ = nullptr;
    std::shared_ptr<CLTensor> fw_recurrent_to_output_weights_ = nullptr;
    std::shared_ptr<CLTensor> fw_cell_to_input_weights_ = nullptr;
    std::shared_ptr<CLTensor> fw_cell_to_forget_weights_ = nullptr;
    std::shared_ptr<CLTensor> fw_cell_to_output_weights_ = nullptr;
    std::shared_ptr<CLTensor> fw_input_gate_bias_ = nullptr;
    std::shared_ptr<CLTensor> fw_forget_gate_bias_ = nullptr;
    std::shared_ptr<CLTensor> fw_cell_gate_bias_ = nullptr;
    std::shared_ptr<CLTensor> fw_output_gate_bias_ = nullptr;
    std::shared_ptr<CLTensor> fw_projection_weights_ = nullptr;
    std::shared_ptr<CLTensor> fw_projection_bias_ = nullptr;
    std::shared_ptr<CLTensor> fw_input_layer_norm_weights_ = nullptr;
    std::shared_ptr<CLTensor> fw_forget_layer_norm_weights_ = nullptr;
    std::shared_ptr<CLTensor> fw_cell_layer_norm_weights_ = nullptr;
    std::shared_ptr<CLTensor> fw_output_layer_norm_weights_ = nullptr;
    std::shared_ptr<CLTensor> bw_input_to_input_weights_ = nullptr;
    std::shared_ptr<CLTensor> bw_input_to_forget_weights_ = nullptr;
    std::shared_ptr<CLTensor> bw_input_to_cell_weights_ = nullptr;
    std::shared_ptr<CLTensor> bw_input_to_output_weights_ = nullptr;
    std::shared_ptr<CLTensor> bw_recurrent_to_input_weights_ = nullptr;
    std::shared_ptr<CLTensor> bw_recurrent_to_forget_weights_ = nullptr;
    std::shared_ptr<CLTensor> bw_recurrent_to_cell_weights_ = nullptr;
    std::shared_ptr<CLTensor> bw_recurrent_to_output_weights_ = nullptr;
    std::shared_ptr<CLTensor> bw_cell_to_input_weights_ = nullptr;
    std::shared_ptr<CLTensor> bw_cell_to_forget_weights_ = nullptr;
    std::shared_ptr<CLTensor> bw_cell_to_output_weights_ = nullptr;
    std::shared_ptr<CLTensor> bw_input_gate_bias_ = nullptr;
    std::shared_ptr<CLTensor> bw_forget_gate_bias_ = nullptr;
    std::shared_ptr<CLTensor> bw_cell_gate_bias_ = nullptr;
    std::shared_ptr<CLTensor> bw_output_gate_bias_ = nullptr;
    std::shared_ptr<CLTensor> bw_projection_weights_ = nullptr;
    std::shared_ptr<CLTensor> bw_projection_bias_ = nullptr;
    std::shared_ptr<CLTensor> bw_input_layer_norm_weights_ = nullptr;
    std::shared_ptr<CLTensor> bw_forget_layer_norm_weights_ = nullptr;
    std::shared_ptr<CLTensor> bw_cell_layer_norm_weights_ = nullptr;
    std::shared_ptr<CLTensor> bw_output_layer_norm_weights_ = nullptr;
    std::shared_ptr<CLTensor> fw_input_activation_state_ = nullptr;
    std::shared_ptr<CLTensor> fw_input_cell_state_ = nullptr;
    std::shared_ptr<CLTensor> bw_input_activation_state_ = nullptr;
    std::shared_ptr<CLTensor> bw_input_cell_state_ = nullptr;
    std::shared_ptr<CLTensor> fw_activation_state_ = nullptr;
    std::shared_ptr<CLTensor> fw_cell_state_ = nullptr;
    std::shared_ptr<CLTensor> bw_activation_state_ = nullptr;
    std::shared_ptr<CLTensor> bw_cell_state_ = nullptr;
    std::shared_ptr<CLTensor> aux_in_ = nullptr;
    std::shared_ptr<CLTensor> fw_aux_input_to_input_weights_ = nullptr;
    std::shared_ptr<CLTensor> fw_aux_input_to_forget_weights_ = nullptr;
    std::shared_ptr<CLTensor> fw_aux_input_to_cell_weights_ = nullptr;
    std::shared_ptr<CLTensor> fw_aux_input_to_output_weights_ = nullptr;
    std::shared_ptr<CLTensor> bw_aux_input_to_input_weights_ = nullptr;
    std::shared_ptr<CLTensor> bw_aux_input_to_forget_weights_ = nullptr;
    std::shared_ptr<CLTensor> bw_aux_input_to_cell_weights_ = nullptr;
    std::shared_ptr<CLTensor> bw_aux_input_to_output_weights_ = nullptr;
    std::shared_ptr<CLTensor> fw_input_gate_scratch_buffer_ = nullptr;
    std::shared_ptr<CLTensor> fw_forget_gate_scratch_buffer_ = nullptr;
    std::shared_ptr<CLTensor> fw_cell_gate_scratch_buffer_ = nullptr;
    std::shared_ptr<CLTensor> fw_output_gate_scratch_buffer_ = nullptr;
    std::shared_ptr<CLTensor> bw_input_gate_scratch_buffer_ = nullptr;
    std::shared_ptr<CLTensor> bw_forget_gate_scratch_buffer_ = nullptr;
    std::shared_ptr<CLTensor> bw_cell_gate_scratch_buffer_ = nullptr;
    std::shared_ptr<CLTensor> bw_output_gate_scratch_buffer_ = nullptr;

    // for optimization
    std::shared_ptr<CLTensor> input_output_buffer_ = nullptr;  // {sequence_length, n_batch, n_input + n_output, 1}
    std::shared_ptr<CLTensor> fw_input_recurrent_to_4gate_weights_ = nullptr;  // {4 * n_cell, n_input + n_output, 1, 1}
    std::shared_ptr<CLTensor> fw_input_recurrent_to_4gate_weights_reordered_ = nullptr;
    std::shared_ptr<CLTensor> bw_input_recurrent_to_4gate_weights_ = nullptr;  // {4 * n_cell, n_input + n_output, 1, 1}
    std::shared_ptr<CLTensor> bw_input_recurrent_to_4gate_weights_reordered_ = nullptr;
    std::shared_ptr<CLTensor> fw_4gate_bias_ = nullptr;  // {4 * n_cell, 1, 1, 1}
    std::shared_ptr<CLTensor> bw_4gate_bias_ = nullptr;  // {4 * n_cell, 1, 1, 1}

    // 3. execute functions
    Status reorder_4gates_matrix();
    Status merge_weights_bias();
    Status optional_inputs();
    Status set_output_shape();
};  // class CLBidirectionalSequenceLstm

}  // namespace gpu
}  // namespace ud
}  // namespace enn
