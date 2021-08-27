#include "CLTanh.hpp"

namespace enn {
namespace ud {
namespace gpu {
namespace {
constexpr int INPUT_INDEX = 0;
constexpr int OUTPUT_INDEX = 0;
}  // namespace

CLTanh::CLTanh(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    ENN_DBG_PRINT("CLTanh is created\n");
    input_tensor_ = nullptr;
    output_tensor_ = nullptr;
    kernel_ = nullptr;
}

Status CLTanh::initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                          const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                          const std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLTanh::initialize() is called\n");

    input_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(INPUT_INDEX));
    output_tensor_ = std::static_pointer_cast<CLTensor>(output_tensors.at(OUTPUT_INDEX));
    CHECK_EXPR_RETURN_FAILURE(nullptr == parameters, "CLTanh doesn't have parameters\n");

    Status status = Status::SUCCESS;
    if (precision_ == PrecisionType::INT8) {
        status = runtime_->setKernel(&kernel_, "SIGNEDtanh", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel SIGNEDtanh failure\n");
    } else {
        status = runtime_->setKernel(&kernel_, "tanh", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel tanh failure\n");
    }

    return Status::SUCCESS;
}

Status CLTanh::execute() {
    ENN_DBG_PRINT("CLTanh::execute() is called\n");

    if (input_tensor_->getTotalSizeFromDims() == 0) {
        ENN_DBG_PRINT("CLTanh execute return for zero_sized input\n");
        return Status::SUCCESS;
    }

    if (precision_ == PrecisionType::UINT8 || precision_ == PrecisionType::INT8) {
        Status status = tanhQuant();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "tanhQuant execute failure\n");
    } else {
        Status status = tanhFloat();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "tanhFloat execute failure\n");
    }

    return Status::SUCCESS;
}

Status CLTanh::tanhQuant() {
    ENN_DBG_PRINT("CLTanh::tanhQuant() is called\n");
    auto input_data = input_tensor_->getDataPtr();
    auto output_data = output_tensor_->getDataPtr();
    auto input_dim = input_tensor_->getDim();
    auto output_dim = output_tensor_->getDim();
    const uint32_t total_num = input_dim.n * input_dim.c * input_dim.h * input_dim.w;

    const int32_t qmin =
        precision_ == PrecisionType::INT8 ? std::numeric_limits<int8_t>::min() : std::numeric_limits<uint8_t>::min();
    const int32_t qmax =
        precision_ == PrecisionType::INT8 ? std::numeric_limits<int8_t>::max() : std::numeric_limits<uint8_t>::max();
    const float in_scale = input_tensor_->getScale();
    const int32_t in_zero = input_tensor_->getZeroPoint();
    const float out_scale = output_tensor_->getScale();
    const int32_t out_zero = output_tensor_->getZeroPoint();

    Status status = runtime_->setKernelArg(kernel_.get(),
                                           input_data,
                                           output_data,
                                           input_dim.n,
                                           input_dim.c,
                                           input_dim.h,
                                           input_dim.w,
                                           qmin,
                                           qmax,
                                           in_scale,
                                           in_zero,
                                           out_scale,
                                           out_zero);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel failure\n");

    size_t global[3] = {0, 0, 0};
    size_t local[3] = {1, 1, 32};
    global[0] = input_dim.n;
    global[1] = alignTo(input_dim.c, local[1]);
    global[2] = alignTo(ceil(input_dim.h * input_dim.w), local[2]);

    status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)3, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "execute kernel failure\n");

    return Status::SUCCESS;
}

Status CLTanh::tanhFloat() {
    ENN_DBG_PRINT("CLTanh::tanhFloat() is called\n");
    auto input_data = input_tensor_->getDataPtr();
    auto output_data = output_tensor_->getDataPtr();
    auto input_dim = input_tensor_->getDim();
    auto output_dim = output_tensor_->getDim();

    Status status = runtime_->setKernelArg(kernel_.get(), input_data, output_data, input_dim.c, input_dim.h, input_dim.w);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");

    size_t global[3] = {0, 0, 0};
    size_t local[3] = {1, 1, 32};
    global[0] = input_dim.n;
    global[1] = alignTo(input_dim.c, local[1]);
    global[2] = alignTo(ceil(input_dim.h * input_dim.w / 8.0), local[2]);

    status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)3, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "enqueueKernel failure\n");

    return Status::SUCCESS;
}

Status CLTanh::release() {
    ENN_DBG_PRINT("CLTanh::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
