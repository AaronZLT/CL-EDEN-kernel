#include "userdriver/gpu/operators/cl_quantized_utils/KernelUtil.hpp"
#include "userdriver/gpu/operators/cl_quantized_utils/QuantizationUtil.hpp"
#include "CLMul.hpp"

namespace enn {
namespace ud {
namespace gpu {
namespace {
constexpr int INPUT_INDEX_0 = 0;
constexpr int INPUT_INDEX_1 = 1;
constexpr int OUTPUT_INDEX = 0;
}  // namespace

CLMul::CLMul(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    ENN_DBG_PRINT("CLMul is created\n");
    input_tensors_.clear();
    output_tensor_ = nullptr;
    parameters_ = nullptr;
    cl_activation_ = nullptr;
    kernel_ = nullptr;
    kernel_one_input_ = nullptr;
    input_broadcast_0_ = nullptr;
    input_broadcast_1_ = nullptr;
    input_broadcast_2_ = nullptr;
    map_tensor_0_ = nullptr;
    map_tensor_1_ = nullptr;
}

Status CLMul::initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                         const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                         const std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLMul::initialize() is called\n");

    for (auto input_tensor : input_tensors) {
        input_tensors_.push_back(std::static_pointer_cast<CLTensor>(input_tensor));
    }
    CHECK_EXPR_RETURN_FAILURE(
        input_tensors_.size() >= 2, "CLMul at least has two input tensors, here only %u\n", input_tensors_.size());
    output_tensor_ = std::static_pointer_cast<CLTensor>(output_tensors.at(OUTPUT_INDEX));
    parameters_ = std::static_pointer_cast<MulParameters>(parameters);
    DEBUG_PRINT("MulParameters, androidNN: %d, isNCHW: %d\n", parameters_->androidNN, parameters_->isNCHW);

    Status status = Status::SUCCESS;
    if (parameters_->androidNN) {
        NDims dims_in_0 = input_tensors_[INPUT_INDEX_0]->getDims();
        NDims dims_in_1 = input_tensors_[INPUT_INDEX_1]->getDims();
        if (parameters_->isNCHW) {
            // optimize for AI Benchmark model's MUL with 2 inputs: a 4-D Tensor and a 1-D Tensor
            if (input_tensors_[INPUT_INDEX_0]->getNumOfDims() == 1)
                reorder123DimsTo4DimsForBroadcast(input_tensors_[INPUT_INDEX_0]->getDims(), dims_in_0);
            if (input_tensors_[INPUT_INDEX_1]->getNumOfDims() == 1)
                reorder123DimsTo4DimsForBroadcast(input_tensors_[INPUT_INDEX_1]->getDims(), dims_in_1);
        }
        NDims broadcasted_dims;
        bool ret = getBroadcastedDims(dims_in_0, dims_in_1, broadcasted_dims);
        CHECK_EXPR_RETURN_FAILURE(true == ret, "Invalid input dims, which are not broadcastable.");
        if (output_tensor_->getDims() != broadcasted_dims) {
            output_tensor_->reconfigureDimsAndBuffer(broadcasted_dims);
        }
    }

    input_broadcast_0_ = input_tensors_[INPUT_INDEX_0];
    if (input_tensors_[INPUT_INDEX_0]->getDims() != output_tensor_->getDims()) {
        CHECK_EXPR_RETURN_FAILURE(isBroadcastable(input_tensors_[INPUT_INDEX_0]->getDims(), output_tensor_->getDims()),
                                  "Invalid input dims, which are not broadcastable.");
        input_broadcast_0_ = std::make_shared<CLTensor>(runtime_,
                                                        precision_,
                                                        input_tensors_[INPUT_INDEX_0]->getDataType(),
                                                        output_tensor_->getDims(),
                                                        input_tensors_[INPUT_INDEX_0]->getDataOrder(),
                                                        input_tensors_[INPUT_INDEX_0]->getScale(),
                                                        input_tensors_[INPUT_INDEX_0]->getZeroPoint());
    }

    input_broadcast_1_ = input_tensors_[INPUT_INDEX_1];
    if (input_tensors_[INPUT_INDEX_1]->getDims() != output_tensor_->getDims()) {
        CHECK_EXPR_RETURN_FAILURE(isBroadcastable(input_tensors_[INPUT_INDEX_1]->getDims(), output_tensor_->getDims()),
                                  "Invalid input dims, which are not broadcastable.");
        input_broadcast_1_ = std::make_shared<CLTensor>(runtime_,
                                                        precision_,
                                                        input_tensors_[INPUT_INDEX_1]->getDataType(),
                                                        output_tensor_->getDims(),
                                                        input_tensors_[INPUT_INDEX_1]->getDataOrder(),
                                                        input_tensors_[INPUT_INDEX_1]->getScale(),
                                                        input_tensors_[INPUT_INDEX_1]->getZeroPoint());
    }

    if (precision_ == PrecisionType::INT8) {
        status = runtime_->setKernel(&kernel_, "SIGNEDeltwise_mul_zero_one", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel SIGNEDeltwise_mul_zero_one failure\n");
    } else if (precision_ == PrecisionType::UINT8) {
        status = runtime_->setKernel(&kernel_, "eltwise_mul_zero_one", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel eltwise_mul_zero_one failure\n");
    } else if (output_tensor_->getDataType() == DataType::INT32) {
        status = runtime_->setKernel(&kernel_, "eltwise_mul_int", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel eltwise_mul_int failure\n");
    } else {
        status = runtime_->setKernel(&kernel_, "eltwise_mul_zero_one", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel eltwise_mul_zero_one failure\n");

        if (input_tensors_.size() > 2) {
            status = runtime_->setKernel(&kernel_one_input_, "eltwise_mul_two_more", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel eltwise_mul_two_more failure\n");
        }
    }

    if (parameters_->activation_info.isEnabled()) {
        auto activation_parameters = std::make_shared<ActivationParameters>();
        activation_parameters->activation_info = parameters_->activation_info;

        cl_activation_ = std::make_shared<CLActivation>(runtime_, precision_);
        status = cl_activation_->initialize({output_tensor_}, {output_tensor_}, activation_parameters);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "activation initialize failure\n");
    }

    return Status::SUCCESS;
}

Status CLMul::execute() {
    ENN_DBG_PRINT("CLMul::execute() is called\n");

    for (auto input_tensor : input_tensors_) {
        if (input_tensor->getTotalSizeFromDims() == 0) {
            ENN_DBG_PRINT("CLMul execute return for zero_sized input\n");
            return Status::SUCCESS;
        }
    }

    auto input_tensor_0 = input_tensors_[INPUT_INDEX_0];
    auto input_tensor_1 = input_tensors_[INPUT_INDEX_1];

    Status status = Status::SUCCESS;
    if (input_tensor_0 != input_broadcast_0_) {
        status = input_tensor_0->broadCastTo(input_broadcast_0_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "input_tensor_0->broadCastTo execute failure\n");
    }
    if (input_tensor_1 != input_broadcast_1_) {
        status = input_tensor_1->broadCastTo(input_broadcast_1_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "input_tensor_1->broadCastTo execute failure\n");
    }

    if (precision_ == PrecisionType::UINT8 || precision_ == PrecisionType::INT8) {
        status = mulQuant();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "mulQuant execute failure\n");
    } else {
        status = mulFloat();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "mulFloat execute failure\n");
    }

    if (parameters_->activation_info.isEnabled()) {
        status = cl_activation_->execute();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "activation execute failure\n");
    }

    return Status::SUCCESS;
}

Status CLMul::mulQuant() {
    ENN_DBG_PRINT("CLMul::mulQuant() is called\n");
    auto output_dim = output_tensor_->getDim();

    int32_t activation_min = 0;
    int32_t activation_max = 0;
    if (precision_ == PrecisionType::INT8) {
        CalculateActivationRangeInt8(parameters_->activation_info.activation(),
                                     output_tensor_->getScale(),
                                     output_tensor_->getZeroPoint(),
                                     &activation_min,
                                     &activation_max);

    } else {
        CalculateActivationRangeUint8(parameters_->activation_info.activation(),
                                      output_tensor_->getScale(),
                                      output_tensor_->getZeroPoint(),
                                      &activation_min,
                                      &activation_max);
    }

    const double real_multiplier =
        input_tensors_[INPUT_INDEX_0]->getScale() * input_tensors_[INPUT_INDEX_1]->getScale() / output_tensor_->getScale();
    int32_t output_multiplier = 0;
    int32_t output_shift = 0;
    QuantizeMultiplierSmallerThanOneExp(real_multiplier, &output_multiplier, &output_shift);

    Status status = runtime_->setKernelArg(kernel_.get(),
                                           input_broadcast_0_->getDataPtr(),
                                           input_broadcast_1_->getDataPtr(),
                                           output_tensor_->getDataPtr(),
                                           -input_broadcast_0_->getZeroPoint(),
                                           -input_broadcast_1_->getZeroPoint(),
                                           output_tensor_->getZeroPoint(),
                                           output_multiplier,
                                           output_shift,
                                           activation_min,
                                           activation_max);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel failure\n");

    size_t global[3] = {0, 0, 0};
    size_t local[3] = {1, 1, 1};
    global[0] = output_dim.n;
    global[1] = output_dim.c * output_dim.h * output_dim.w;
    global[2] = 1;

    status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)3, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "execute kernel failure\n");

    return Status::SUCCESS;
}

Status CLMul::mulFloat() {
    ENN_DBG_PRINT("CLMul::mulFloat() is called\n");
    auto output_dim = output_tensor_->getDim();

    Status status = Status::SUCCESS;
    status = runtime_->setKernelArg(kernel_.get(),
                                    input_broadcast_0_->getDataPtr(),
                                    input_broadcast_1_->getDataPtr(),
                                    output_tensor_->getDataPtr(),
                                    output_dim.h * output_dim.w);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");

    size_t global[2] = {0, 0};
    size_t local[2] = {1, 64};
    global[0] = output_dim.n * output_dim.c;
    global[1] = alignTo(ceil(output_dim.h * output_dim.w / 8.0), local[1]);

    status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)2, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "enqueueKernel failure\n");

    if (input_tensors_.size() > 2) {
        for (size_t i = 2; i < input_tensors_.size(); ++i) {
            auto input_tensor = input_tensors_[i];
            if (input_tensor->getDims() != output_tensor_->getDims()) {
                CHECK_EXPR_RETURN_FAILURE(isBroadcastable(input_tensor->getDims(), output_tensor_->getDims()),
                                          "Invalid input dims, which are not broadcastable.");
                status = input_tensor->broadCastTo(input_broadcast_0_);
                CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "input_tensors[%u]->broadCastTo execute failure\n", i);
            } else {
                input_broadcast_0_ = input_tensor;
            }

            status = runtime_->setKernelArg(kernel_one_input_.get(),
                                            input_broadcast_0_->getDataPtr(),
                                            output_tensor_->getDataPtr());
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");

            const size_t local[2] = {1, 1};
            const size_t global[2] = {
                output_dim.n * output_dim.c,
                alignTo(ceil(output_dim.h * output_dim.w), local[1])};

            status = runtime_->enqueueKernel(kernel_one_input_.get(), (cl_uint)2, global, local);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "enqueueKernel failure\n");
        }
    }

    return Status::SUCCESS;
}

Status CLMul::release() {
    ENN_DBG_PRINT("CLMul::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
