#include "CLRelu.hpp"

namespace enn {
namespace ud {
namespace gpu {
namespace {
constexpr int INPUT_INDEX = 0;
constexpr int OUTPUT_INDEX = 0;
}  // namespace

CLRelu::CLRelu(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    ENN_DBG_PRINT("CLRelu is created\n");
    input_tensor_ = nullptr;
    output_tensor_ = nullptr;
    parameters_ = nullptr;
    kernel_ = nullptr;
}

Status CLRelu::initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                          const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                          const std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLRelu::initialize() is called\n");

    input_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(INPUT_INDEX));
    output_tensor_ = std::static_pointer_cast<CLTensor>(output_tensors.at(OUTPUT_INDEX));
    parameters_ =
        parameters == nullptr ? std::make_shared<ReluParameters>() : std::static_pointer_cast<ReluParameters>(parameters);

    ENN_DBG_PRINT("negative_slope: %f\n", parameters_->negative_slope);

    Status status = Status::SUCCESS;
    if (precision_ == PrecisionType::INT8) {
        status = runtime_->setKernel(&kernel_, "SIGNEDrelu", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel SIGNEDrelu failure\n");
    } else {
        status = runtime_->setKernel(&kernel_, "relu", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel relu failure\n");
    }

    return Status::SUCCESS;
}

Status CLRelu::execute() {
    ENN_DBG_PRINT("CLRelu::execute() is called\n");

    if (input_tensor_->getTotalSizeFromDims() == 0) {
        ENN_DBG_PRINT("CLRelu execute return for zero_sized input\n");
        return Status::SUCCESS;
    }

    if (precision_ == PrecisionType::UINT8 || precision_ == PrecisionType::INT8) {
        Status status = reluQuant();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "reluQuant execute failure\n");
    } else {
        Status status = reluFloat();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "reluFloat execute failure\n");
    }

    return Status::SUCCESS;
}

Status CLRelu::reluQuant() {
    ENN_DBG_PRINT("CLRelu::reluQuant() is called\n");
    auto input_data = input_tensor_->getDataPtr();
    auto output_data = output_tensor_->getDataPtr();
    auto input_dim = input_tensor_->getDim();
    auto output_dim = output_tensor_->getDim();
    const uint32_t total_num = input_dim.n * input_dim.c * input_dim.h * input_dim.w;

    int32_t qmin = 0;
    int32_t qmax = 0;
    if (precision_ == PrecisionType::INT8) {
        qmin = std::numeric_limits<int8_t>::min();
        qmax = std::numeric_limits<int8_t>::max();
    } else {
        qmin = std::numeric_limits<uint8_t>::min();
        qmax = std::numeric_limits<uint8_t>::max();
    }

    const int32_t offset = output_tensor_->getZeroPoint();
    const double scale = output_tensor_->getScale();
    auto quantize = [scale, offset](float f) { return offset + static_cast<int32_t>(round(f / scale)); };

    const int32_t min = std::max(qmin, quantize(0.0));
    const int32_t max = qmax;

    size_t global[1] = {total_num};

    Status status = runtime_->setKernelArg(kernel_.get(), input_data, output_data, min, max);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel failure\n");

    status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)1, global, NULL);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "execute kernel failure\n");

    return Status::SUCCESS;
}

Status CLRelu::reluFloat() {
    ENN_DBG_PRINT("CLRelu::reluFloat() is called\n");
    auto input_data = input_tensor_->getDataPtr();
    auto output_data = output_tensor_->getDataPtr();
    auto input_dim = input_tensor_->getDim();
    auto output_dim = output_tensor_->getDim();

    Status status = runtime_->setKernelArg(
        kernel_.get(), input_data, output_data, parameters_->negative_slope, output_dim.c, output_dim.h, output_dim.w);
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

Status CLRelu::release() {
    ENN_DBG_PRINT("CLRelu::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
