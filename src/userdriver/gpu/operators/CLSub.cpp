#include "userdriver/gpu/operators/cl_quantized_utils/KernelUtil.hpp"
#include "CLSub.hpp"

namespace enn {
namespace ud {
namespace gpu {
namespace {
constexpr int INPUT_INDEX_0 = 0;
constexpr int INPUT_INDEX_1 = 1;
constexpr int OUTPUT_INDEX = 0;
}  // namespace

CLSub::CLSub(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    ENN_DBG_PRINT("CLSub is created\n");
    input_tensor_0_ = nullptr;
    input_tensor_1_ = nullptr;
    output_tensor_ = nullptr;
    parameters_ = nullptr;
    cl_activation_ = nullptr;
    kernel_ = nullptr;
    input_broadcast_0_ = nullptr;
    input_broadcast_1_ = nullptr;
}

Status CLSub::initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                         const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                         const std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLSub::initialize() is called\n");

    input_tensor_0_ = std::static_pointer_cast<CLTensor>(input_tensors.at(INPUT_INDEX_0));
    input_tensor_1_ = std::static_pointer_cast<CLTensor>(input_tensors.at(INPUT_INDEX_1));
    output_tensor_ = std::static_pointer_cast<CLTensor>(output_tensors.at(OUTPUT_INDEX));
    parameters_ = std::static_pointer_cast<SubParameters>(parameters);

    Status status = Status::SUCCESS;
    if (parameters_->androidNN) {
        // for dynamic output
        NDims input_dims_0 = input_tensor_0_->getDims();
        NDims input_dims_1 = input_tensor_1_->getDims();
        NDims broadcast_dims;
        const bool ret = getBroadcastedDims(input_dims_0, input_dims_1, broadcast_dims);
        CHECK_EXPR_RETURN_FAILURE(true == ret, "Invalid input dims, which are not broadcastable.");
        if (output_tensor_->getDims() != broadcast_dims) {
            output_tensor_->reconfigureDimsAndBuffer(broadcast_dims);
        }
    }

    input_broadcast_0_ = input_tensor_0_;
    if (input_tensor_0_->getDims() != output_tensor_->getDims()) {
        CHECK_EXPR_RETURN_FAILURE(isBroadcastable(input_tensor_0_->getDims(), output_tensor_->getDims()),
                                  "Invalid input dims, which are not broadcastable.");
        input_broadcast_0_ = std::make_shared<CLTensor>(runtime_,
                                                        precision_,
                                                        input_tensor_0_->getDataType(),
                                                        output_tensor_->getDims(),
                                                        input_tensor_0_->getDataOrder(),
                                                        input_tensor_0_->getScale(),
                                                        input_tensor_0_->getZeroPoint());
    }

    input_broadcast_1_ = input_tensor_1_;
    if (input_tensor_1_->getDims() != output_tensor_->getDims()) {
        CHECK_EXPR_RETURN_FAILURE(isBroadcastable(input_tensor_1_->getDims(), output_tensor_->getDims()),
                                  "Invalid input dims, which are not broadcastable.");
        input_broadcast_1_ = std::make_shared<CLTensor>(runtime_,
                                                        precision_,
                                                        input_tensor_1_->getDataType(),
                                                        output_tensor_->getDims(),
                                                        input_tensor_1_->getDataOrder(),
                                                        input_tensor_1_->getScale(),
                                                        input_tensor_1_->getZeroPoint());
    }

    if (precision_ == PrecisionType::INT8) {
        status = runtime_->setKernel(&kernel_, "SIGNEDsub", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel SIGNEDsub failure\n");
    } else if (output_tensor_->getDataType() == DataType::INT32) {
        status = runtime_->setKernel(&kernel_, "INT32sub", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel INT32sub failure\n");
    } else {
        status = runtime_->setKernel(&kernel_, "sub", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel sub failure\n");
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

Status CLSub::execute() {
    ENN_DBG_PRINT("CLSub::execute() is called\n");

    if (input_tensor_0_->getTotalSizeFromDims() == 0 || input_tensor_1_->getTotalSizeFromDims() == 0) {
        ENN_DBG_PRINT("CLSub execute return for zero_sized input. input_0 size = %d, input_1 size = %d\n",
                      static_cast<int>(input_tensor_0_->getTotalSizeFromDims()),
                      static_cast<int>(input_tensor_1_->getTotalSizeFromDims()));
        return Status::SUCCESS;
    }

    Status status = Status::SUCCESS;
    if (input_tensor_0_ != input_broadcast_0_) {
        status = input_tensor_0_->broadCastTo(input_broadcast_0_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "input_tensor_0_->broadCastTo execute failure\n");
    }
    if (input_tensor_1_ != input_broadcast_1_) {
        status = input_tensor_1_->broadCastTo(input_broadcast_1_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "input_tensor_1_->broadCastTo execute failure\n");
    }

    if (precision_ == PrecisionType::UINT8 || precision_ == PrecisionType::INT8) {
        status = subQuant();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "subQuant execute failure\n");
    } else {
        status = subFloat();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "subFloat execute failure\n");
    }

    if (parameters_->activation_info.isEnabled()) {
        status = cl_activation_->execute();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "activation execute failure\n");
    }

    return Status::SUCCESS;
}

Status CLSub::subQuant() {
    ENN_DBG_PRINT("CLSub::subQuant() is called\n");
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

    Status status = runtime_->setKernelArg(kernel_.get(),
                                           input_broadcast_0_->getDataPtr(),
                                           input_broadcast_1_->getDataPtr(),
                                           output_tensor_->getDataPtr(),
                                           output_dim.h * output_dim.w,
                                           input_broadcast_0_->getZeroPoint(),
                                           input_broadcast_0_->getScale(),
                                           input_broadcast_1_->getZeroPoint(),
                                           input_broadcast_1_->getScale(),
                                           output_tensor_->getZeroPoint(),
                                           output_tensor_->getScale(),
                                           activation_min,
                                           activation_max);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel failure\n");

    size_t global[2] = {output_dim.n * output_dim.c, 0};
    size_t local[2] = {1, 32};
    global[1] = alignTo(ceil(output_dim.h * output_dim.w / 8.0), local[1]);

    status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)2, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "execute kernel failure\n");

    return Status::SUCCESS;
}

Status CLSub::subFloat() {
    ENN_DBG_PRINT("CLSub::subFloat() is called\n");
    auto output_dim = output_tensor_->getDim();

    Status status = Status::SUCCESS;
    status = runtime_->setKernelArg(kernel_.get(),
                                    input_broadcast_0_->getDataPtr(),
                                    input_broadcast_1_->getDataPtr(),
                                    output_tensor_->getDataPtr());
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");

    size_t global[3] = {0, 0, 0};
    size_t local[3] = {1, 1, 1};
    global[0] = output_dim.n;
    global[1] = output_dim.c;
    global[2] = output_dim.h * output_dim.w;

    status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)3, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "enqueueKernel failure\n");

    return Status::SUCCESS;
}

Status CLSub::release() {
    ENN_DBG_PRINT("CLSub::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
