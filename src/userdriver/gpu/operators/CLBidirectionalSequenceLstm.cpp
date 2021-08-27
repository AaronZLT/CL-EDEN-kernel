#include "common/enn_utils.h"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/cl_rnnlstm_utils/LstmEval.hpp"
#include "userdriver/gpu/operators/CLActivation.hpp"
#include "userdriver/gpu/operators/CLBidirectionalSequenceLstm.hpp"

namespace enn {
namespace ud {
namespace gpu {

namespace {
const uint32_t INPUT_INDEX = 0;
const uint32_t FW_INPUT_TO_INPUT_WEIGHT = 1;
const uint32_t FW_INPUT_TO_FORGET_WEIGHT = 2;
const uint32_t FW_INPUT_TO_CELL_WEIGHT = 3;
const uint32_t FW_INPUT_TO_OUTPUT_WEIGHT = 4;
const uint32_t FW_RECURRENT_TO_INPUT_WEIGHT = 5;
const uint32_t FW_RECURRENT_TO_FORGET_WEIGHT = 6;
const uint32_t FW_RECURRENT_TO_CELL_WEIGHT = 7;
const uint32_t FW_RECURRENT_TO_OUTPUT_WEIGHT = 8;
const uint32_t FW_CELL_TO_INPUT_WEIGHT = 9;
const uint32_t FW_CELL_TO_FORGET_WEIGHT = 10;
const uint32_t FW_CELL_TO_OUTPUT_WEIGHT = 11;
const uint32_t FW_INPUT_GATE_BIAS = 12;
const uint32_t FW_FORGET_GATE_BIAS = 13;
const uint32_t FW_CELL_GATE_BIAS = 14;
const uint32_t FW_OUTPUT_GATE_BIAS = 15;
const uint32_t FW_PROJECTION_WEIGHT = 16;
const uint32_t FW_PROJECTION_BIAS = 17;
const uint32_t BW_INPUT_TO_INPUT_WEIGHT = 18;
const uint32_t BW_INPUT_TO_FORGET_WEIGHT = 19;
const uint32_t BW_INPUT_TO_CELL_WEIGHT = 20;
const uint32_t BW_INPUT_TO_OUTPUT_WEIGHT = 21;
const uint32_t BW_RECURRENT_TO_INPUT_WEIGHT = 22;
const uint32_t BW_RECURRENT_TO_FORGET_WEIGHT = 23;
const uint32_t BW_RECURRENT_TO_CELL_WEIGHT = 24;
const uint32_t BW_RECURRENT_TO_OUTPUT_WEIGHT = 25;
const uint32_t BW_CELL_TO_INPUT_WEIGHT = 26;
const uint32_t BW_CELL_TO_FORGET_WEIGHT = 27;
const uint32_t BW_CELL_TO_OUTPUT_WEIGHT = 28;
const uint32_t BW_INPUT_GATE_BIAS = 29;
const uint32_t BW_FORGET_GATE_BIAS = 30;
const uint32_t BW_CELL_GATE_BIAS = 31;
const uint32_t BW_OUTPUT_GATE_BIAS = 32;
const uint32_t BW_PROJECTION_WEIGHT = 33;
const uint32_t BW_PROJECTION_BIAS = 34;
const uint32_t FW_INPUT_ACTIVATION_STATE = 35;
const uint32_t FW_INPUT_CELL_STATE = 36;
const uint32_t BW_INPUT_ACTIVATION_STATE = 37;
const uint32_t BW_INPUT_CELL_STATE = 38;
const uint32_t AUX_INPUT = 39;
const uint32_t FW_AUX_INPUT_TO_INPUT_WEIGHT = 40;
const uint32_t FW_AUX_INPUT_TO_FORGET_WEIGHT = 41;
const uint32_t FW_AUX_INPUT_TO_CELL_WEIGHT = 42;
const uint32_t FW_AUX_INPUT_TO_OUTPUT_WEIGHT = 43;
const uint32_t BW_AUX_INPUT_TO_INPUT_WEIGHT = 44;
const uint32_t BW_AUX_INPUT_TO_FORGET_WEIGHT = 45;
const uint32_t BW_AUX_INPUT_TO_CELL_WEIGHT = 46;
const uint32_t BW_AUX_INPUT_TO_OUTPUT_WEIGHT = 47;
const uint32_t FW_INPUT_LAYER_NORM_WEIGHT = 48;
const uint32_t FW_FORGET_LAYER_NORM_WEIGHT = 49;
const uint32_t FW_CELL_LAYER_NORM_WEIGHT = 50;
const uint32_t FW_OUTPUT_LAYER_NORM_WEIGHT = 51;
const uint32_t BW_INPUT_LAYER_NORM_WEIGHT = 52;
const uint32_t BW_FORGET_LAYER_NORM_WEIGHT = 53;
const uint32_t BW_CELL_LAYER_NORM_WEIGHT = 54;
const uint32_t BW_OUTPUT_LAYER_NORM_WEIGHT = 55;

const uint32_t FW_OUTPUT = 0;
const uint32_t BW_OUTPUT = 1;
}  // namespace

CLBidirectionalSequenceLstm::CLBidirectionalSequenceLstm(const std::shared_ptr<CLRuntime> runtime,
                                                         const PrecisionType &precision) :
    runtime_(runtime),
    precision_(precision), max_time_(0), n_batch_(0), n_input_(0), n_fw_output_(0), n_bw_output_(0), n_fw_cell_(0),
    n_bw_cell_(0), fw_output_batch_leading_dim_(0), bw_output_batch_leading_dim_(0), bw_output_offset_(0),
    aux_input_size_(0), fw_scratch_dims_({0, 0, 0, 0}), bw_scratch_dims_({0, 0, 0, 0}) {
    ENN_DBG_PRINT("CLBidirectionalSequenceLstm is created");
}

Status CLBidirectionalSequenceLstm::initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                                               const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                                               const std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLBidirectionalSequenceLstm::initialize() is called");

    fw_in_ = std::static_pointer_cast<CLTensor>(input_tensors.at(INPUT_INDEX));
    fw_out_ = std::static_pointer_cast<CLTensor>(output_tensors.at(FW_OUTPUT));
    bw_out_ = (output_tensors.size() == 1) ? fw_out_ : std::static_pointer_cast<CLTensor>(output_tensors.at(BW_OUTPUT));

    fw_input_to_input_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(FW_INPUT_TO_INPUT_WEIGHT));
    fw_input_to_forget_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(FW_INPUT_TO_FORGET_WEIGHT));
    fw_input_to_cell_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(FW_INPUT_TO_CELL_WEIGHT));
    fw_input_to_output_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(FW_INPUT_TO_OUTPUT_WEIGHT));

    fw_recurrent_to_input_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(FW_RECURRENT_TO_INPUT_WEIGHT));
    fw_recurrent_to_forget_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(FW_RECURRENT_TO_FORGET_WEIGHT));
    fw_recurrent_to_cell_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(FW_RECURRENT_TO_CELL_WEIGHT));
    fw_recurrent_to_output_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(FW_RECURRENT_TO_OUTPUT_WEIGHT));

    fw_cell_to_input_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(FW_CELL_TO_INPUT_WEIGHT));
    fw_cell_to_forget_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(FW_CELL_TO_FORGET_WEIGHT));
    fw_cell_to_output_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(FW_CELL_TO_OUTPUT_WEIGHT));

    fw_input_gate_bias_ = std::static_pointer_cast<CLTensor>(input_tensors.at(FW_INPUT_GATE_BIAS));
    fw_forget_gate_bias_ = std::static_pointer_cast<CLTensor>(input_tensors.at(FW_FORGET_GATE_BIAS));
    fw_cell_gate_bias_ = std::static_pointer_cast<CLTensor>(input_tensors.at(FW_CELL_GATE_BIAS));
    fw_output_gate_bias_ = std::static_pointer_cast<CLTensor>(input_tensors.at(FW_OUTPUT_GATE_BIAS));

    fw_projection_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(FW_PROJECTION_WEIGHT));
    fw_projection_bias_ = std::static_pointer_cast<CLTensor>(input_tensors.at(FW_PROJECTION_BIAS));

    bw_input_to_input_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(BW_INPUT_TO_INPUT_WEIGHT));
    bw_input_to_forget_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(BW_INPUT_TO_FORGET_WEIGHT));
    bw_input_to_cell_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(BW_INPUT_TO_CELL_WEIGHT));
    bw_input_to_output_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(BW_INPUT_TO_OUTPUT_WEIGHT));

    bw_recurrent_to_input_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(BW_RECURRENT_TO_INPUT_WEIGHT));
    bw_recurrent_to_forget_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(BW_RECURRENT_TO_FORGET_WEIGHT));
    bw_recurrent_to_cell_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(BW_RECURRENT_TO_CELL_WEIGHT));
    bw_recurrent_to_output_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(BW_RECURRENT_TO_OUTPUT_WEIGHT));

    bw_cell_to_input_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(BW_CELL_TO_INPUT_WEIGHT));
    bw_cell_to_forget_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(BW_CELL_TO_FORGET_WEIGHT));
    bw_cell_to_output_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(BW_CELL_TO_OUTPUT_WEIGHT));

    bw_input_gate_bias_ = std::static_pointer_cast<CLTensor>(input_tensors.at(BW_INPUT_GATE_BIAS));
    bw_forget_gate_bias_ = std::static_pointer_cast<CLTensor>(input_tensors.at(BW_FORGET_GATE_BIAS));
    bw_cell_gate_bias_ = std::static_pointer_cast<CLTensor>(input_tensors.at(BW_CELL_GATE_BIAS));
    bw_output_gate_bias_ = std::static_pointer_cast<CLTensor>(input_tensors.at(BW_OUTPUT_GATE_BIAS));
    bw_projection_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(BW_PROJECTION_WEIGHT));
    bw_projection_bias_ = std::static_pointer_cast<CLTensor>(input_tensors.at(BW_PROJECTION_BIAS));

    fw_input_activation_state_ = std::static_pointer_cast<CLTensor>(input_tensors.at(FW_INPUT_ACTIVATION_STATE));
    fw_input_cell_state_ = std::static_pointer_cast<CLTensor>(input_tensors.at(FW_INPUT_CELL_STATE));
    bw_input_activation_state_ = std::static_pointer_cast<CLTensor>(input_tensors.at(BW_INPUT_ACTIVATION_STATE));
    bw_input_cell_state_ = std::static_pointer_cast<CLTensor>(input_tensors.at(BW_INPUT_CELL_STATE));

    aux_in_ = std::static_pointer_cast<CLTensor>(input_tensors.at(FW_AUX_INPUT_TO_INPUT_WEIGHT));
    fw_aux_input_to_input_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(FW_AUX_INPUT_TO_INPUT_WEIGHT));
    fw_aux_input_to_forget_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(FW_AUX_INPUT_TO_FORGET_WEIGHT));
    fw_aux_input_to_cell_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(FW_AUX_INPUT_TO_CELL_WEIGHT));
    fw_aux_input_to_output_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(FW_AUX_INPUT_TO_OUTPUT_WEIGHT));

    bw_aux_input_to_input_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(BW_AUX_INPUT_TO_INPUT_WEIGHT));
    bw_aux_input_to_forget_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(BW_AUX_INPUT_TO_FORGET_WEIGHT));
    bw_aux_input_to_cell_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(BW_AUX_INPUT_TO_CELL_WEIGHT));
    bw_aux_input_to_output_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(BW_AUX_INPUT_TO_OUTPUT_WEIGHT));

    fw_input_layer_norm_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(FW_INPUT_LAYER_NORM_WEIGHT));
    fw_forget_layer_norm_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(FW_FORGET_LAYER_NORM_WEIGHT));
    fw_cell_layer_norm_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(FW_CELL_LAYER_NORM_WEIGHT));
    fw_output_layer_norm_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(FW_OUTPUT_LAYER_NORM_WEIGHT));
    bw_input_layer_norm_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(BW_INPUT_LAYER_NORM_WEIGHT));
    bw_forget_layer_norm_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(BW_FORGET_LAYER_NORM_WEIGHT));
    bw_cell_layer_norm_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(BW_CELL_LAYER_NORM_WEIGHT));
    bw_output_layer_norm_weights_ = std::static_pointer_cast<CLTensor>(input_tensors.at(BW_OUTPUT_LAYER_NORM_WEIGHT));

    parameters_ = std::static_pointer_cast<BidirectionalSequenceLstmParameters>(parameters);
    ENN_DBG_PRINT("DeconvolutionParameters: cell_clip %f; proj_clip %f; merge_outputs %d; time_major %d;"
                  "androidNN %d; force_alloc_state_at_init %d; weights_as_input %d;"
                  "activation_info_: isEnabled() %d,activation() %d;\n",
                  parameters_->cell_clip,
                  parameters_->proj_clip,
                  parameters_->merge_outputs,
                  parameters_->time_major,
                  parameters_->androidNN,
                  parameters_->force_alloc_state_at_init,
                  parameters_->weights_as_input,
                  parameters_->activation_info->isEnabled(),
                  parameters_->activation_info->activation());
    activation_info_ = *parameters_->activation_info.get();

    // [AI Benchmark v4] For CRNN's precision, we force BiLstm to allocate states at init stage so that it has an
    // identical bahavior to tflite's.
    if (fw_input_activation_state_->getDataPtr() == nullptr && parameters_->force_alloc_state_at_init) {
        fw_input_activation_state_ = std::make_shared<CLTensor>(
            runtime_, precision_, fw_input_activation_state_->getDataType(), fw_input_activation_state_->getDim());
    }
    if (fw_input_activation_state_->getDataPtr()) {
        runtime_->zeroBuf(fw_input_activation_state_->getNumOfBytes(), fw_input_activation_state_->getDataPtr());
    }

    if (fw_input_cell_state_->getDataPtr() == nullptr && parameters_->force_alloc_state_at_init) {
        fw_input_cell_state_ = std::make_shared<CLTensor>(
            runtime_, precision_, fw_input_cell_state_->getDataType(), fw_input_cell_state_->getDim());
    }
    if (fw_input_cell_state_->getDataPtr()) {
        runtime_->zeroBuf(fw_input_cell_state_->getNumOfBytes(), fw_input_cell_state_->getDataPtr());
    }

    if (bw_input_activation_state_->getDataPtr() == nullptr && parameters_->force_alloc_state_at_init) {
        bw_input_activation_state_ = std::make_shared<CLTensor>(
            runtime_, precision_, bw_input_activation_state_->getDataType(), bw_input_activation_state_->getDim());
    }
    if (bw_input_activation_state_->getDataPtr()) {
        runtime_->zeroBuf(bw_input_activation_state_->getNumOfBytes(), bw_input_activation_state_->getDataPtr());
    }

    if (bw_input_cell_state_->getDataPtr() == nullptr && parameters_->force_alloc_state_at_init) {
        bw_input_cell_state_ = std::make_shared<CLTensor>(
            runtime_, precision_, bw_input_cell_state_->getDataType(), bw_input_cell_state_->getDim());
    }
    if (bw_input_cell_state_->getDataPtr()) {
        runtime_->zeroBuf(bw_input_cell_state_->getNumOfBytes(), bw_input_cell_state_->getDataPtr());
    }

    if (parameters_->time_major) {
        max_time_ = fw_in_->getDims(0);
        n_batch_ = fw_in_->getDims(1);
    } else {
        max_time_ = fw_in_->getDims(1);
        n_batch_ = fw_in_->getDims(0);
    }

    n_input_ = fw_in_->getDims(2);
    n_fw_cell_ = fw_input_to_output_weights_->getDims(0);
    n_bw_cell_ = bw_input_to_output_weights_->getDims(0);
    n_fw_output_ = fw_recurrent_to_output_weights_->getDims(1);
    n_bw_output_ = bw_recurrent_to_output_weights_->getDims(1);

    if (parameters_->merge_outputs) {
        fw_output_batch_leading_dim_ = n_fw_output_ + n_bw_output_;
        bw_output_offset_ = fw_recurrent_to_output_weights_->getDims(1);
        bw_output_batch_leading_dim_ = n_fw_output_ + n_bw_output_;
    } else {
        fw_output_batch_leading_dim_ = n_fw_output_;
        bw_output_offset_ = 0;
        bw_output_batch_leading_dim_ = n_bw_output_;
    }
    if (parameters_->androidNN) {
        Status state = set_output_shape();
        CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "set_output_shape failed.");
    }
    // for optimization
    if (fw_input_to_input_weights_ != nullptr && bw_input_to_input_weights_ != nullptr && n_fw_cell_ == n_bw_cell_ &&
        n_fw_output_ == n_bw_output_) {
        Dim4 input_output_dims = {1, n_batch_, n_input_ + n_fw_output_, 1};
        input_output_buffer_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, input_output_dims);

        Dim4 input_recurrent_to_4gate_weights_dims = {4 * n_fw_cell_, n_input_ + n_fw_output_, 1, 1};
        Dim4 reordered_input_recurrent_to_4gate_weights_dims = {
            IntegralDivideRoundUp(input_recurrent_to_4gate_weights_dims.c, 16),
            input_recurrent_to_4gate_weights_dims.n * 16,
            input_recurrent_to_4gate_weights_dims.h,
            input_recurrent_to_4gate_weights_dims.w};

        fw_input_recurrent_to_4gate_weights_ =
            std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, input_recurrent_to_4gate_weights_dims);
        fw_input_recurrent_to_4gate_weights_reordered_ = std::make_shared<CLTensor>(
            runtime_, precision_, DataType::FLOAT, reordered_input_recurrent_to_4gate_weights_dims);
        bw_input_recurrent_to_4gate_weights_ =
            std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, input_recurrent_to_4gate_weights_dims);
        bw_input_recurrent_to_4gate_weights_reordered_ = std::make_shared<CLTensor>(
            runtime_, precision_, DataType::FLOAT, reordered_input_recurrent_to_4gate_weights_dims);

        Dim4 gate4_bias_dims = {4 * n_fw_cell_, 1, 1, 1};
        fw_4gate_bias_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, gate4_bias_dims);
        bw_4gate_bias_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, gate4_bias_dims);

        // merge weights and bias
        if (!parameters_->weights_as_input) {
            merge_weights_bias();
        }
    }

    if (output_tensors.size() == 6) {
        fw_activation_state_ = std::static_pointer_cast<CLTensor>(output_tensors.at(2));
        fw_cell_state_ = std::static_pointer_cast<CLTensor>(output_tensors.at(3));
        bw_activation_state_ = std::static_pointer_cast<CLTensor>(output_tensors.at(4));
        bw_cell_state_ = std::static_pointer_cast<CLTensor>(output_tensors.at(5));
        runtime_->zeroBuf(fw_activation_state_->getNumOfBytes(), fw_activation_state_->getDataPtr());
        runtime_->zeroBuf(fw_cell_state_->getNumOfBytes(), fw_cell_state_->getDataPtr());
        runtime_->zeroBuf(bw_activation_state_->getNumOfBytes(), bw_activation_state_->getDataPtr());
        runtime_->zeroBuf(bw_cell_state_->getNumOfBytes(), bw_cell_state_->getDataPtr());
    } else {
        fw_activation_state_ = fw_input_activation_state_;
        fw_cell_state_ = fw_input_cell_state_;
        bw_activation_state_ = bw_input_activation_state_;
        bw_cell_state_ = bw_input_cell_state_;
    }

    return Status::SUCCESS;
}

Status CLBidirectionalSequenceLstm::reorder_4gates_matrix() {
    // reorder 4gates matrix from H * W to W / 16 * H * 16
    std::shared_ptr<struct _cl_kernel> align_weights_kernel;
    Status state = runtime_->setKernel(&align_weights_kernel, "reorder_4gates_matrix", precision_);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernel failure\n");

    // fw align
    Dim4 fw_4gates_dim = fw_input_recurrent_to_4gate_weights_->getDim();
    size_t global_fw[2] = {fw_4gates_dim.n, IntegralDivideRoundUp(fw_4gates_dim.c, 16)};
    state = runtime_->setKernelArg(align_weights_kernel.get(),
                                   fw_input_recurrent_to_4gate_weights_->getDataPtr(),
                                   fw_input_recurrent_to_4gate_weights_reordered_->getDataPtr(),
                                   fw_4gates_dim.n,
                                   fw_4gates_dim.c);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernelArg failure\n");
    state = runtime_->enqueueKernel(align_weights_kernel.get(), (cl_uint)2, global_fw, NULL);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "execute kernel failure\n");

    // bw align
    Dim4 bw_4gates_dim = bw_input_recurrent_to_4gate_weights_->getDim();
    size_t global_bw[2] = {bw_4gates_dim.n, IntegralDivideRoundUp(bw_4gates_dim.c, 16)};
    state = runtime_->setKernelArg(align_weights_kernel.get(),
                                   bw_input_recurrent_to_4gate_weights_->getDataPtr(),
                                   bw_input_recurrent_to_4gate_weights_reordered_->getDataPtr(),
                                   bw_4gates_dim.n,
                                   bw_4gates_dim.c);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernelArg failure\n");
    state = runtime_->enqueueKernel(align_weights_kernel.get(), (cl_uint)2, global_bw, NULL);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "execute kernel failure\n");
    return Status::SUCCESS;
}

Status CLBidirectionalSequenceLstm::merge_weights_bias() {
    Status state = Status::FAILURE;
    std::shared_ptr<struct _cl_kernel> kernel_;
    state = runtime_->setKernel(&kernel_, "merge_weights", precision_);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernel failure\n");

    size_t global[1] = {static_cast<size_t>(n_fw_cell_)};

    // merge fw weights
    auto fw_input_to_input_weights_tensor = fw_input_to_input_weights_->getDataPtr();
    auto fw_input_to_forget_weights_tensor = fw_input_to_forget_weights_->getDataPtr();
    auto fw_input_to_cell_weights_tensor = fw_input_to_cell_weights_->getDataPtr();
    auto fw_input_to_output_weights_tensor = fw_input_to_output_weights_->getDataPtr();
    auto fw_recurrent_to_input_weights_tensor = fw_recurrent_to_input_weights_->getDataPtr();
    auto fw_recurrent_to_forget_weights_tensor = fw_recurrent_to_forget_weights_->getDataPtr();
    auto fw_recurrent_to_cell_weights_tensor = fw_recurrent_to_cell_weights_->getDataPtr();
    auto fw_recurrent_to_output_weights_tensor = fw_recurrent_to_output_weights_->getDataPtr();

    state = runtime_->setKernelArg(kernel_.get(),
                                   fw_input_to_input_weights_tensor,
                                   fw_input_to_forget_weights_tensor,
                                   fw_input_to_cell_weights_tensor,
                                   fw_input_to_output_weights_tensor,
                                   fw_recurrent_to_input_weights_tensor,
                                   fw_recurrent_to_forget_weights_tensor,
                                   fw_recurrent_to_cell_weights_tensor,
                                   fw_recurrent_to_output_weights_tensor,
                                   fw_input_recurrent_to_4gate_weights_->getDataPtr(),
                                   n_input_,
                                   n_fw_output_,
                                   n_fw_cell_);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernelArg failure\n");

    state = runtime_->enqueueKernel(kernel_.get(), (cl_uint)1, global, NULL);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "execute kernel failure\n");

    // merge bw weights
    auto bw_input_to_input_weights_tensor = bw_input_to_input_weights_->getDataPtr();
    auto bw_input_to_forget_weights_tensor = bw_input_to_forget_weights_->getDataPtr();
    auto bw_input_to_cell_weights_tensor = bw_input_to_cell_weights_->getDataPtr();
    auto bw_input_to_output_weights_tensor = bw_input_to_output_weights_->getDataPtr();
    auto bw_recurrent_to_input_weights_tensor = bw_recurrent_to_input_weights_->getDataPtr();
    auto bw_recurrent_to_forget_weights_tensor = bw_recurrent_to_forget_weights_->getDataPtr();
    auto bw_recurrent_to_cell_weights_tensor = bw_recurrent_to_cell_weights_->getDataPtr();
    auto bw_recurrent_to_output_weights_tensor = bw_recurrent_to_output_weights_->getDataPtr();

    state = runtime_->setKernelArg(kernel_.get(),
                                   bw_input_to_input_weights_tensor,
                                   bw_input_to_forget_weights_tensor,
                                   bw_input_to_cell_weights_tensor,
                                   bw_input_to_output_weights_tensor,
                                   bw_recurrent_to_input_weights_tensor,
                                   bw_recurrent_to_forget_weights_tensor,
                                   bw_recurrent_to_cell_weights_tensor,
                                   bw_recurrent_to_output_weights_tensor,
                                   bw_input_recurrent_to_4gate_weights_->getDataPtr(),
                                   n_input_,
                                   n_fw_output_,
                                   n_fw_cell_);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernelArg failure\n");

    state = runtime_->enqueueKernel(kernel_.get(), (cl_uint)1, global, NULL);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "execute kernel failure\n");

    state = reorder_4gates_matrix();
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "reorder 4gates_matrix fail\n");

    // merge bias
    size_t type_size = runtime_->getRuntimeTypeBytes(precision_);
    size_t src_offset = 0;
    size_t dst_offset = 0;
    size_t copy_bytes = (size_t)n_fw_cell_ * type_size;
    runtime_->copyBuffer(
        fw_4gate_bias_->getDataPtr(), fw_input_gate_bias_->getDataPtr(), dst_offset, src_offset, copy_bytes);
    dst_offset += n_fw_cell_ * type_size;
    runtime_->copyBuffer(
        fw_4gate_bias_->getDataPtr(), fw_forget_gate_bias_->getDataPtr(), dst_offset, src_offset, copy_bytes);
    dst_offset += n_fw_cell_ * type_size;
    runtime_->copyBuffer(fw_4gate_bias_->getDataPtr(), fw_cell_gate_bias_->getDataPtr(), dst_offset, src_offset, copy_bytes);
    dst_offset += n_fw_cell_ * type_size;
    runtime_->copyBuffer(
        fw_4gate_bias_->getDataPtr(), fw_output_gate_bias_->getDataPtr(), dst_offset, src_offset, copy_bytes);

    dst_offset = 0;
    runtime_->copyBuffer(
        bw_4gate_bias_->getDataPtr(), bw_input_gate_bias_->getDataPtr(), dst_offset, src_offset, copy_bytes);
    dst_offset += n_fw_cell_ * type_size;
    runtime_->copyBuffer(
        bw_4gate_bias_->getDataPtr(), bw_forget_gate_bias_->getDataPtr(), dst_offset, src_offset, copy_bytes);
    dst_offset += n_fw_cell_ * type_size;
    runtime_->copyBuffer(bw_4gate_bias_->getDataPtr(), bw_cell_gate_bias_->getDataPtr(), dst_offset, src_offset, copy_bytes);
    dst_offset += n_fw_cell_ * type_size;
    runtime_->copyBuffer(
        bw_4gate_bias_->getDataPtr(), bw_output_gate_bias_->getDataPtr(), dst_offset, src_offset, copy_bytes);

    return state;
}

Status CLBidirectionalSequenceLstm::execute() {
    ENN_DBG_PRINT("CLBidirectionalSequenceLstm::execute is called");

    // for optional inputs
    optional_inputs();

    bool use_layer_norm = fw_input_layer_norm_weights_ != nullptr;
    bool fw_use_cifg = fw_input_to_input_weights_ == nullptr;
    // bool output_state = output.size() == 6;
    fw_scratch_dims_ = {n_batch_, n_fw_cell_};
    if (fw_use_cifg) {
        fw_input_gate_scratch_buffer_ = nullptr;
        fw_forget_gate_scratch_buffer_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, fw_scratch_dims_);
        fw_cell_gate_scratch_buffer_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, fw_scratch_dims_);
        fw_output_gate_scratch_buffer_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, fw_scratch_dims_);
    } else {
        fw_input_gate_scratch_buffer_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, fw_scratch_dims_);
        fw_forget_gate_scratch_buffer_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, fw_scratch_dims_);
        fw_cell_gate_scratch_buffer_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, fw_scratch_dims_);
        fw_output_gate_scratch_buffer_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, fw_scratch_dims_);
    }

    bool bw_use_cifg = bw_input_to_input_weights_ == nullptr;
    bw_scratch_dims_ = {n_batch_, n_bw_cell_};
    if (bw_use_cifg) {
        bw_input_gate_scratch_buffer_ = nullptr;
        bw_forget_gate_scratch_buffer_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, bw_scratch_dims_);
        bw_cell_gate_scratch_buffer_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, bw_scratch_dims_);
        bw_output_gate_scratch_buffer_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, bw_scratch_dims_);
    } else {
        bw_input_gate_scratch_buffer_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, bw_scratch_dims_);
        bw_forget_gate_scratch_buffer_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, bw_scratch_dims_);
        bw_cell_gate_scratch_buffer_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, bw_scratch_dims_);
        bw_output_gate_scratch_buffer_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, bw_scratch_dims_);
    }

    if (nullptr != aux_in_) {
        aux_input_size_ = aux_in_->getDims(2);
    }

    if (use_layer_norm == true) {
        fw_aux_input_to_input_weights_ = fw_input_layer_norm_weights_;
        fw_aux_input_to_forget_weights_ = fw_forget_layer_norm_weights_;
        fw_aux_input_to_cell_weights_ = fw_cell_layer_norm_weights_;
        fw_aux_input_to_output_weights_ = fw_output_layer_norm_weights_;
        bw_aux_input_to_input_weights_ = bw_input_layer_norm_weights_;
        bw_aux_input_to_forget_weights_ = bw_forget_layer_norm_weights_;
        bw_aux_input_to_cell_weights_ = bw_cell_layer_norm_weights_;
        bw_aux_input_to_output_weights_ = bw_output_layer_norm_weights_;
        aux_input_size_ = 0;
        aux_in_ = nullptr;
    }

    std::shared_ptr<CLTensor> bw_in = fw_in_;
    const bool is_parallel_linking = (aux_in_ != nullptr && fw_aux_input_to_forget_weights_ == nullptr);
    if (is_parallel_linking) {
        bw_in = aux_in_;
        aux_in_ = nullptr;
    }
    const bool both_use_cifg = (fw_input_to_input_weights_ == nullptr && bw_input_to_input_weights_ == nullptr);
    const bool both_use_peephole = (fw_cell_to_output_weights_ != nullptr && bw_cell_to_output_weights_ != nullptr);
    if (both_use_cifg == 0 && both_use_peephole == 0 && aux_in_ == nullptr && fw_projection_weights_ == nullptr &&
        fw_projection_bias_ == nullptr && bw_projection_weights_ == nullptr && bw_projection_bias_ == nullptr &&
        parameters_->cell_clip <= 0.000000000001f && use_layer_norm == false &&
        activation_info_.activation() == ActivationInfo::ActivationType::TANH && n_batch_ == 1 &&
        fw_output_batch_leading_dim_ != n_fw_output_ && bw_output_batch_leading_dim_ != n_bw_output_) {
        // merge weights and bias

        if (parameters_->weights_as_input) {
            merge_weights_bias();
        }

        evalFloatOpt(runtime_,
                     precision_,
                     fw_in_,
                     input_output_buffer_,
                     fw_input_recurrent_to_4gate_weights_reordered_,
                     fw_4gate_bias_,
                     max_time_,
                     n_batch_,
                     n_input_,
                     n_fw_cell_,
                     n_fw_output_,
                     fw_output_batch_leading_dim_,
                     parameters_->proj_clip,
                     true,
                     0,
                     fw_input_gate_scratch_buffer_,
                     fw_forget_gate_scratch_buffer_,
                     fw_cell_gate_scratch_buffer_,
                     fw_output_gate_scratch_buffer_,
                     fw_activation_state_,
                     fw_cell_state_,
                     fw_out_);
        if (parameters_->time_major) {
            max_time_ = bw_in->getDims(0);
            n_batch_ = bw_in->getDims(1);
        } else {
            max_time_ = bw_in->getDims(1);
            n_batch_ = bw_in->getDims(0);
        }
        n_input_ = bw_in->getDims(2);
        evalFloatOpt(runtime_,
                     precision_,
                     bw_in,
                     input_output_buffer_,
                     bw_input_recurrent_to_4gate_weights_reordered_,
                     bw_4gate_bias_,
                     max_time_,
                     n_batch_,
                     n_input_,
                     n_bw_cell_,
                     n_bw_output_,
                     bw_output_batch_leading_dim_,
                     parameters_->proj_clip,
                     false,
                     bw_output_offset_,
                     bw_input_gate_scratch_buffer_,
                     bw_forget_gate_scratch_buffer_,
                     bw_cell_gate_scratch_buffer_,
                     bw_output_gate_scratch_buffer_,
                     bw_activation_state_,
                     bw_cell_state_,
                     bw_out_);

    } else {
        evalFloat(runtime_,
                  precision_,
                  fw_in_,
                  fw_input_to_input_weights_,
                  fw_input_to_forget_weights_,
                  fw_input_to_cell_weights_,
                  fw_input_to_output_weights_,
                  fw_recurrent_to_input_weights_,
                  fw_recurrent_to_forget_weights_,
                  fw_recurrent_to_cell_weights_,
                  fw_recurrent_to_output_weights_,
                  fw_cell_to_input_weights_,
                  fw_cell_to_forget_weights_,
                  fw_cell_to_output_weights_,
                  aux_in_,
                  fw_aux_input_to_input_weights_,
                  fw_aux_input_to_forget_weights_,
                  fw_aux_input_to_cell_weights_,
                  fw_aux_input_to_output_weights_,
                  fw_input_gate_bias_,
                  fw_forget_gate_bias_,
                  fw_cell_gate_bias_,
                  fw_output_gate_bias_,
                  fw_projection_weights_,
                  fw_projection_bias_,
                  max_time_,
                  n_batch_,
                  n_input_,
                  aux_input_size_,
                  n_fw_cell_,
                  n_fw_output_,
                  fw_output_batch_leading_dim_,
                  activation_info_,
                  parameters_->cell_clip,
                  parameters_->proj_clip,
                  true,
                  0,
                  fw_input_gate_scratch_buffer_,
                  fw_forget_gate_scratch_buffer_,
                  fw_cell_gate_scratch_buffer_,
                  fw_output_gate_scratch_buffer_,
                  fw_activation_state_,
                  fw_cell_state_,
                  fw_out_,
                  use_layer_norm);
        if (parameters_->time_major) {
            max_time_ = bw_in->getDims(0);
            n_batch_ = bw_in->getDims(1);
        } else {
            max_time_ = bw_in->getDims(1);
            n_batch_ = bw_in->getDims(0);
        }
        n_input_ = bw_in->getDims(2);

        evalFloat(runtime_,
                  precision_,
                  bw_in,
                  bw_input_to_input_weights_,
                  bw_input_to_forget_weights_,
                  bw_input_to_cell_weights_,
                  bw_input_to_output_weights_,
                  bw_recurrent_to_input_weights_,
                  bw_recurrent_to_forget_weights_,
                  bw_recurrent_to_cell_weights_,
                  bw_recurrent_to_output_weights_,
                  bw_cell_to_input_weights_,
                  bw_cell_to_forget_weights_,
                  bw_cell_to_output_weights_,
                  aux_in_,
                  bw_aux_input_to_input_weights_,
                  bw_aux_input_to_forget_weights_,
                  bw_aux_input_to_cell_weights_,
                  bw_aux_input_to_output_weights_,
                  bw_input_gate_bias_,
                  bw_forget_gate_bias_,
                  bw_cell_gate_bias_,
                  bw_output_gate_bias_,
                  bw_projection_weights_,
                  bw_projection_bias_,
                  max_time_,
                  n_batch_,
                  n_input_,
                  aux_input_size_,
                  n_bw_cell_,
                  n_bw_output_,
                  bw_output_batch_leading_dim_,
                  activation_info_,
                  parameters_->cell_clip,
                  parameters_->proj_clip,
                  false,
                  bw_output_offset_,
                  bw_input_gate_scratch_buffer_,
                  bw_forget_gate_scratch_buffer_,
                  bw_cell_gate_scratch_buffer_,
                  bw_output_gate_scratch_buffer_,
                  bw_activation_state_,
                  bw_cell_state_,
                  bw_out_,
                  use_layer_norm);
    }

    return Status::SUCCESS;
}

Status CLBidirectionalSequenceLstm::optional_inputs() {
    fw_input_to_input_weights_ =
        (fw_input_to_input_weights_ == nullptr || fw_input_to_input_weights_->getTotalSizeFromDims() == 0)
            ? nullptr
            : fw_input_to_input_weights_;
    fw_recurrent_to_input_weights_ =
        (fw_recurrent_to_input_weights_ == nullptr || fw_recurrent_to_input_weights_->getTotalSizeFromDims() == 0)
            ? nullptr
            : fw_recurrent_to_input_weights_;
    fw_cell_to_input_weights_ =
        (fw_cell_to_input_weights_ == nullptr || fw_cell_to_input_weights_->getTotalSizeFromDims() == 0)
            ? nullptr
            : fw_cell_to_input_weights_;
    fw_cell_to_forget_weights_ =
        (fw_cell_to_forget_weights_ == nullptr || fw_cell_to_forget_weights_->getTotalSizeFromDims() == 0)
            ? nullptr
            : fw_cell_to_forget_weights_;
    fw_cell_to_output_weights_ =
        (fw_cell_to_output_weights_ == nullptr || fw_cell_to_output_weights_->getTotalSizeFromDims() == 0)
            ? nullptr
            : fw_cell_to_output_weights_;

    fw_input_gate_bias_ =
        (fw_input_gate_bias_ == nullptr || fw_input_gate_bias_->getTotalSizeFromDims() == 0) ? nullptr : fw_input_gate_bias_;
    fw_projection_weights_ = (fw_projection_weights_ == nullptr || fw_projection_weights_->getTotalSizeFromDims() == 0)
                                 ? nullptr
                                 : fw_projection_weights_;
    fw_projection_bias_ =
        (fw_projection_bias_ == nullptr || fw_projection_bias_->getTotalSizeFromDims() == 0) ? nullptr : fw_projection_bias_;

    bw_input_to_input_weights_ =
        (bw_input_to_input_weights_ == nullptr || bw_input_to_input_weights_->getTotalSizeFromDims() == 0)
            ? nullptr
            : bw_input_to_input_weights_;
    bw_recurrent_to_input_weights_ =
        (bw_recurrent_to_input_weights_ == nullptr || bw_recurrent_to_input_weights_->getTotalSizeFromDims() == 0)
            ? nullptr
            : bw_recurrent_to_input_weights_;

    bw_cell_to_input_weights_ =
        (bw_cell_to_input_weights_ == nullptr || bw_cell_to_input_weights_->getTotalSizeFromDims() == 0)
            ? nullptr
            : bw_cell_to_input_weights_;
    bw_cell_to_forget_weights_ =
        (bw_cell_to_forget_weights_ == nullptr || bw_cell_to_forget_weights_->getTotalSizeFromDims() == 0)
            ? nullptr
            : bw_cell_to_forget_weights_;
    bw_cell_to_output_weights_ =
        (bw_cell_to_output_weights_ == nullptr || bw_cell_to_output_weights_->getTotalSizeFromDims() == 0)
            ? nullptr
            : bw_cell_to_output_weights_;

    bw_input_gate_bias_ =
        (bw_input_gate_bias_ == nullptr || bw_input_gate_bias_->getTotalSizeFromDims() == 0) ? nullptr : bw_input_gate_bias_;
    bw_projection_weights_ = (bw_projection_weights_ == nullptr || bw_projection_weights_->getTotalSizeFromDims() == 0)
                                 ? nullptr
                                 : bw_projection_weights_;
    bw_projection_bias_ =
        (bw_projection_bias_ == nullptr || bw_projection_bias_->getTotalSizeFromDims() == 0) ? nullptr : bw_projection_bias_;

    fw_aux_input_to_input_weights_ =
        (fw_aux_input_to_input_weights_ == nullptr || fw_aux_input_to_input_weights_->getTotalSizeFromDims() == 0)
            ? nullptr
            : fw_aux_input_to_input_weights_;
    fw_aux_input_to_forget_weights_ =
        (fw_aux_input_to_forget_weights_ == nullptr || fw_aux_input_to_forget_weights_->getTotalSizeFromDims() == 0)
            ? nullptr
            : fw_aux_input_to_forget_weights_;
    fw_aux_input_to_cell_weights_ =
        (fw_aux_input_to_cell_weights_ == nullptr || fw_aux_input_to_cell_weights_->getTotalSizeFromDims() == 0)
            ? nullptr
            : fw_aux_input_to_cell_weights_;
    fw_aux_input_to_output_weights_ =
        (fw_aux_input_to_output_weights_ == nullptr || fw_aux_input_to_output_weights_->getTotalSizeFromDims() == 0)
            ? nullptr
            : fw_aux_input_to_output_weights_;

    bw_aux_input_to_input_weights_ =
        (bw_aux_input_to_input_weights_ == nullptr || bw_aux_input_to_input_weights_->getTotalSizeFromDims() == 0)
            ? nullptr
            : bw_aux_input_to_input_weights_;
    bw_aux_input_to_forget_weights_ =
        (bw_aux_input_to_forget_weights_ == nullptr || bw_aux_input_to_forget_weights_->getTotalSizeFromDims() == 0)
            ? nullptr
            : bw_aux_input_to_forget_weights_;
    bw_aux_input_to_cell_weights_ =
        (bw_aux_input_to_cell_weights_ == nullptr || bw_aux_input_to_cell_weights_->getTotalSizeFromDims() == 0)
            ? nullptr
            : bw_aux_input_to_cell_weights_;
    bw_aux_input_to_output_weights_ =
        (bw_aux_input_to_output_weights_ == nullptr || bw_aux_input_to_output_weights_->getTotalSizeFromDims() == 0)
            ? nullptr
            : bw_aux_input_to_output_weights_;

    fw_input_layer_norm_weights_ =
        (fw_input_layer_norm_weights_ == nullptr || fw_input_layer_norm_weights_->getTotalSizeFromDims() == 0)
            ? nullptr
            : fw_input_layer_norm_weights_;
    return Status::SUCCESS;
}

Status CLBidirectionalSequenceLstm::set_output_shape() {
    ENN_DBG_PRINT("CLBidirectionalSequenceLstm::%s(+)", __func__);
    Status state = Status::SUCCESS;
    NDims fw_output_ndim, bw_output_ndim;
    if (parameters_->merge_outputs) {
        fw_output_ndim = {fw_in_->getDims(0), fw_in_->getDims(1), n_fw_output_ + n_bw_output_};
        if (fw_out_->getDims() != fw_output_ndim) {
            state = fw_out_->reconfigureDimsAndBuffer(fw_output_ndim);
        }
    } else {
        fw_output_ndim = {fw_in_->getDims(0), fw_in_->getDims(1), n_fw_output_};
        bw_output_ndim = {fw_in_->getDims(0), fw_in_->getDims(1), n_bw_output_};
        if (fw_out_->getDims() != fw_output_ndim) {
            state = fw_out_->reconfigureDimsAndBuffer(fw_output_ndim);
        }
        if (bw_out_->getDims() != bw_output_ndim) {
            state = bw_out_->reconfigureDimsAndBuffer(bw_output_ndim);
        }
    }
    ENN_DBG_PRINT("CLBidirectionalSequenceLstm::%s(-)", __func__);
    return state;
}

Status CLBidirectionalSequenceLstm::release() {
    ENN_DBG_PRINT("CLBidirectionalSequenceLstm::release is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
