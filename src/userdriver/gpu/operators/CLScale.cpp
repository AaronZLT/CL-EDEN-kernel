#include "CLScale.hpp"

namespace enn {
namespace ud {
namespace gpu {
namespace {
constexpr int INPUT_INDEX = 0;
constexpr int SCALE_INDEX = 1;
constexpr int BIAS_INDEX = 2;
constexpr int OUTPUT_INDEX = 0;
}  // namespace

CLScale::CLScale(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    ENN_DBG_PRINT("CLScale is created\n");
    input_tensor_ = nullptr;
    scale_tensor_ = nullptr;
    bias_tensor_ = nullptr;
    output_tensor_ = nullptr;
    kernel_ = nullptr;
}

Status CLScale::initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                           const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                           const std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLScale::initialize() is called\n");
    input_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(INPUT_INDEX));
    scale_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(SCALE_INDEX));
    if (input_tensors.size() == 3) {
        bias_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(BIAS_INDEX));
    } else {
        bias_term_ = 0;
    }
    output_tensor_ = std::static_pointer_cast<CLTensor>(output_tensors.at(OUTPUT_INDEX));

    Status status = runtime_->setKernel(&kernel_, "scale", precision_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel scale failure\n");

    return Status::SUCCESS;
}

Status CLScale::execute() {
    ENN_DBG_PRINT("CLScale::execute() is called\n");
    if (input_tensor_->getTotalSizeFromDims() == 0) {
        ENN_DBG_PRINT("CLScale execute return for zero_sized input\n");
        return Status::SUCCESS;
    }

    auto output_dim = output_tensor_->getDim();
    unsigned int output_batch = output_dim.n;
    unsigned int output_channel = output_dim.c;
    unsigned int output_height = output_dim.h;
    unsigned int output_width = output_dim.w;

    Status status = Status::SUCCESS;
    status = runtime_->setKernelArg(kernel_.get(),
                                    input_tensor_->getDataPtr(),
                                    scale_tensor_->getDataPtr(),
                                    bias_tensor_->getDataPtr(),
                                    output_tensor_->getDataPtr(),
                                    output_channel,
                                    output_height,
                                    output_width,
                                    bias_term_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");

    size_t global[3] = {0, 0, 0};
    size_t local[3] = {1, 1, 32};
    global[0] = output_batch;
    global[1] = output_channel;
    int gsize2 = ceil(output_height * output_width / 8.0);
    global[2] = alignTo(gsize2, 32);

    status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)3, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "execute kernel failure\n");

    return Status::SUCCESS;
}

Status CLScale::release() {
    ENN_DBG_PRINT("CLScale::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
