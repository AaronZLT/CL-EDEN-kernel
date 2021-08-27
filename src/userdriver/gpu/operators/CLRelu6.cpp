#include "CLRelu6.hpp"

namespace enn {
namespace ud {
namespace gpu {
namespace {
constexpr int INPUT_INDEX = 0;
constexpr int OUTPUT_INDEX = 0;
}  // namespace

CLRelu6::CLRelu6(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    ENN_DBG_PRINT("CLRelu6 is created\n");
    input_tensor_ = nullptr;
    output_tensor_ = nullptr;
    kernel_ = nullptr;
}

Status CLRelu6::initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                           const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                           const std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLRelu6::initialize() is called\n");

    input_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(INPUT_INDEX));
    output_tensor_ = std::static_pointer_cast<CLTensor>(output_tensors.at(OUTPUT_INDEX));
    CHECK_EXPR_RETURN_FAILURE(nullptr == parameters, "CLRelu6 doesn't have parameters\n");

    Status status = Status::SUCCESS;
    if (precision_ == PrecisionType::INT8) {
        status = runtime_->setKernel(&kernel_, "SIGNEDrelu6", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel SIGNEDrelu6 failure\n");
    } else {
        status = runtime_->setKernel(&kernel_, "relu6", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel relu6 failure\n");
    }

    return Status::SUCCESS;
}

Status CLRelu6::execute() {
    ENN_DBG_PRINT("CLRelu6::execute() is called\n");

    if (input_tensor_->getTotalSizeFromDims() == 0) {
        ENN_DBG_PRINT("CLRelu6 execute return for zero_sized input\n");
        return Status::SUCCESS;
    }

    if (precision_ == PrecisionType::UINT8 || precision_ == PrecisionType::INT8) {
        Status status = relu6Quant();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "relu6Quant execute failure\n");
    } else {
        Status status = relu6Float();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "relu6Float execute failure\n");
    }

    return Status::SUCCESS;
}

Status CLRelu6::relu6Quant() {
    ENN_DBG_PRINT("CLRelu6::relu6Quant() is called\n");
    auto input_data = input_tensor_->getDataPtr();
    auto output_data = output_tensor_->getDataPtr();
    auto input_dim = input_tensor_->getDim();
    auto output_dim = output_tensor_->getDim();
    const uint32_t total_num = input_dim.n * input_dim.c * input_dim.h * input_dim.w;

    int32_t qmin = 0;
    if (precision_ == PrecisionType::INT8) {
        qmin = std::numeric_limits<int8_t>::min();
    } else {
        qmin = std::numeric_limits<uint8_t>::min();
    }

    const int32_t offset = output_tensor_->getZeroPoint();
    const double scale = output_tensor_->getScale();
    auto quantize = [scale, offset](float f) { return offset + static_cast<int32_t>(round(f / scale)); };

    const int32_t min = std::max(qmin, quantize(0.0f));
    const int32_t max = quantize(6.0f);

    Status status = runtime_->setKernelArg(kernel_.get(), input_data, output_data, min, max);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel failure\n");

    size_t global[1] = {total_num};

    status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)1, global, NULL);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "execute kernel failure\n");

    return Status::SUCCESS;
}

Status CLRelu6::relu6Float() {
    ENN_DBG_PRINT("CLRelu6::relu6Float() is called\n");
    auto input_data = input_tensor_->getDataPtr();
    auto output_data = output_tensor_->getDataPtr();
    auto input_dim = input_tensor_->getDim();
    auto output_dim = output_tensor_->getDim();

    Status status = runtime_->setKernelArg(kernel_.get(), input_data, output_data, output_dim.c, output_dim.h, output_dim.w);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");

    size_t global[3] = {0, 0, 0};
    size_t local[3] = {1, 1, 32};
    global[0] = input_dim.n;
    global[1] = alignTo(output_dim.c, local[1]);
    global[2] = alignTo(ceil(output_dim.h * output_dim.w / 8.0), local[2]);

    status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)3, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "enqueueKernel failure\n");

    return Status::SUCCESS;
}

Status CLRelu6::release() {
    ENN_DBG_PRINT("CLRelu6::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
