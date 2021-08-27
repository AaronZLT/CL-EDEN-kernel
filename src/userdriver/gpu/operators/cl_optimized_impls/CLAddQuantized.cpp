#include "CLAddQuantized.hpp"

namespace enn {
namespace ud {
namespace gpu {
namespace {
constexpr int INPUT_INDEX_0 = 0;
constexpr int INPUT_INDEX_1 = 1;
}  // namespace

CLAddQuantized::CLAddQuantized(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    ENN_DBG_PRINT("CLAddQuantized is created");
    reshaped_input_tensor_0_ = nullptr;
    reshaped_input_tensor_1_ = nullptr;
    output_tensor_ = nullptr;
    activation_info_ = ActivationInfo();
    kernel_ = nullptr;
}

Status CLAddQuantized::initialize(const std::vector<std::shared_ptr<ITensor>> inputs,
                                  std::shared_ptr<ITensor> output,
                                  const ActivationInfo &activation_info) {
    ENN_DBG_PRINT("CLAddQuantized::initialize() is called");

    reshaped_input_tensor_0_ = std::static_pointer_cast<CLTensor>(inputs.at(INPUT_INDEX_0));
    reshaped_input_tensor_1_ = std::static_pointer_cast<CLTensor>(inputs.at(INPUT_INDEX_1));
    output_tensor_ = std::static_pointer_cast<CLTensor>(output);

    // Quantized Add op only supports two inputs and no coefficient.
    CHECK_EXPR_RETURN_FAILURE(2 == inputs.size(), "CLAddQuantized only support 2 inputs");

    if (reshaped_input_tensor_0_->getDims() != output_tensor_->getDims() ||
        reshaped_input_tensor_1_->getDims() != output_tensor_->getDims()) {
        ENN_ERR_PRINT("CLAddQuantized NOT support broadcast for inputs, output.\n");
        const auto dim_in_0 = reshaped_input_tensor_0_->getDim();
        const auto dim_in_1 = reshaped_input_tensor_1_->getDim();
        const auto dim_out = output_tensor_->getDim();
        ENN_ERR_PRINT(
            "CLAddQuantized input[0]->dim = %u, %u, %u, %u; input[1]->dim = %u, %u, %u, %u; output->dim = %u, %u, %u, %u\n",
            dim_in_0.n,
            dim_in_0.c,
            dim_in_0.h,
            dim_in_0.w,
            dim_in_1.n,
            dim_in_1.c,
            dim_in_1.h,
            dim_in_1.w,
            dim_out.n,
            dim_out.c,
            dim_out.h,
            dim_out.w);
        return Status::FAILURE;
    }

    this->activation_info_ = activation_info;

    Status status = Status::SUCCESS;
    if (precision_ == PrecisionType::INT8) {
        status = runtime_->setKernel(&kernel_, "SIGNEDeltwiseAddQuantized", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "create SIGNEDeltwiseAddQuantized failure\n");
    } else {
        status = runtime_->setKernel(&kernel_, "eltwiseAddQuantized_opt", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "create eltwiseAddQuantized_opt failure\n");
    }

    return Status::SUCCESS;
}

Status CLAddQuantized::execute() {
    ENN_DBG_PRINT("CLAddQuantized::execute() is called");
    eltAdd_descriptor_.input0_offset_ = -reshaped_input_tensor_0_->getZeroPoint();
    eltAdd_descriptor_.input1_offset_ = -reshaped_input_tensor_1_->getZeroPoint();
    eltAdd_descriptor_.output_offset_ = output_tensor_->getZeroPoint();
    eltAdd_descriptor_.left_shift_ = 20;

    const double twice_max_input_scale =
        2 * std::max(reshaped_input_tensor_0_->getScale(), reshaped_input_tensor_1_->getScale());
    const double real_input0_multiplier = reshaped_input_tensor_0_->getScale() / twice_max_input_scale;
    const double real_input1_multiplier = reshaped_input_tensor_1_->getScale() / twice_max_input_scale;
    const double real_output_multiplier =
        twice_max_input_scale / ((1 << eltAdd_descriptor_.left_shift_) * output_tensor_->getScale());
    QuantizeMultiplierSmallerThanOneExp(
        real_input0_multiplier, &eltAdd_descriptor_.input0_multiplier_, &eltAdd_descriptor_.input0_shift_);
    QuantizeMultiplierSmallerThanOneExp(
        real_input1_multiplier, &eltAdd_descriptor_.input1_multiplier_, &eltAdd_descriptor_.input1_shift_);
    QuantizeMultiplierSmallerThanOneExp(
        real_output_multiplier, &eltAdd_descriptor_.output_multiplier_, &eltAdd_descriptor_.output_shift_);

    if (activation_info_.isEnabled()) {
        if (precision_ == PrecisionType::INT8) {
            CalculateActivationRangeInt8(activation_info_.activation(),
                                         output_tensor_->getScale(),
                                         output_tensor_->getZeroPoint(),
                                         &eltAdd_descriptor_.act_min_,
                                         &eltAdd_descriptor_.act_max_);
        } else {
            CalculateActivationRangeUint8(activation_info_.activation(),
                                          output_tensor_->getScale(),
                                          output_tensor_->getZeroPoint(),
                                          &eltAdd_descriptor_.act_min_,
                                          &eltAdd_descriptor_.act_max_);
        }
    } else {
        if (precision_ == PrecisionType::INT8) {
            eltAdd_descriptor_.act_min_ = std::numeric_limits<int8_t>::min();
            eltAdd_descriptor_.act_max_ = std::numeric_limits<int8_t>::max();
        } else {
            eltAdd_descriptor_.act_min_ = std::numeric_limits<uint8_t>::min();
            eltAdd_descriptor_.act_max_ = std::numeric_limits<uint8_t>::max();
        }
    }

    Status status = Status::SUCCESS;
    const auto output_dim = output_tensor_->getDim();
    if (precision_ == PrecisionType::INT8) {
        status = runtime_->setKernelArg(kernel_.get(),
                                        reshaped_input_tensor_0_->getDataPtr(),
                                        reshaped_input_tensor_1_->getDataPtr(),
                                        output_tensor_->getDataPtr(),
                                        output_dim.h * output_dim.w,
                                        reshaped_input_tensor_0_->getZeroPoint(),
                                        reshaped_input_tensor_0_->getScale(),
                                        reshaped_input_tensor_1_->getZeroPoint(),
                                        reshaped_input_tensor_1_->getScale(),
                                        output_tensor_->getZeroPoint(),
                                        output_tensor_->getScale(),
                                        eltAdd_descriptor_.act_min_,
                                        eltAdd_descriptor_.act_max_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");

        size_t local[2] = {1, 32};
        size_t global[2] = {output_dim.n * output_dim.c, 0};
        const int gsize1 = ceil(output_dim.h * output_dim.w / 8.0);
        global[1] = alignTo(gsize1, local[1]);

        status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)2, global, local);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "execute kernel failure\n");
    } else {
        status = runtime_->setKernelArg(kernel_.get(),
                                        reshaped_input_tensor_0_->getDataPtr(),
                                        reshaped_input_tensor_1_->getDataPtr(),
                                        output_tensor_->getDataPtr(),
                                        eltAdd_descriptor_.left_shift_,
                                        eltAdd_descriptor_.input0_offset_,
                                        eltAdd_descriptor_.input1_offset_,
                                        eltAdd_descriptor_.output_offset_,
                                        eltAdd_descriptor_.input0_shift_,
                                        eltAdd_descriptor_.input1_shift_,
                                        eltAdd_descriptor_.output_shift_,
                                        eltAdd_descriptor_.input0_multiplier_,
                                        eltAdd_descriptor_.input1_multiplier_,
                                        eltAdd_descriptor_.output_multiplier_,
                                        eltAdd_descriptor_.act_min_,
                                        eltAdd_descriptor_.act_max_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");

        size_t global[2] = {output_dim.n * output_dim.c, output_dim.h * output_dim.w};
        size_t local[2] = {1, static_cast<size_t>(findMaxFactor(global[1], 128))};
        status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)2, global, local);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "execute kernel failure\n");
    }

    return Status::SUCCESS;
}

Status CLAddQuantized::release() {
    ENN_DBG_PRINT("CLAddQuantized::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
