#include "CLSlice.hpp"

namespace enn {
namespace ud {
namespace gpu {
namespace {
constexpr int INPUT_INDEX = 0;
}  // namespace

CLSlice::CLSlice(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    ENN_DBG_PRINT("CLSlice is created\n");
    input_tensor_ = nullptr;
    parameters_ = nullptr;
    kernel_NC_ = nullptr;
    kernel_HW_ = nullptr;
}

Status CLSlice::initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                           const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                           const std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLSlice::initialize() is called\n");
    input_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(INPUT_INDEX));
    for (auto out_tensor : output_tensors) {
        output_tensors_.emplace_back(std::static_pointer_cast<CLTensor>(out_tensor));
    }
    parameters_ = std::static_pointer_cast<SliceParameters>(parameters);
    CHECK_AND_RETURN_ERR(nullptr == parameters_, Status::FAILURE, "CLSlice must have parameters\n");

    Status status = Status::SUCCESS;
    if (parameters_->axis == 0 || parameters_->axis == 1) {
        status = runtime_->setKernel(&kernel_NC_, "slice_NC", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel slice_NC failure\n");
    } else {
        status = runtime_->setKernel(&kernel_HW_, "slice_HW", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel slice_HW failure\n");
    }

    return Status::SUCCESS;
}

Status CLSlice::execute() {
    ENN_DBG_PRINT("CLSlice::execute() is called\n");
    if (input_tensor_->getTotalSizeFromDims() == 0) {
        ENN_DBG_PRINT("CLSlice execute return for zero_sized input\n");
        return Status::SUCCESS;
    }

    auto output_size = output_tensors_.size();
    int offset_implicit = 0;
    if (parameters_->slice_point.size() == 0) {
        if (parameters_->axis == 0) {
            offset_implicit = input_tensor_->getDim().n / output_size;
        } else if (parameters_->axis == 1) {
            offset_implicit = input_tensor_->getDim().c / output_size;
        } else if (parameters_->axis == 2) {
            offset_implicit = input_tensor_->getDim().h / output_size;
        } else {
            offset_implicit = input_tensor_->getDim().w / output_size;
        }
    }

    Status status = Status::SUCCESS;
    if (parameters_->axis == 0 || parameters_->axis == 1) {  // TODO(wuke): use vload to optimize
        int offset = 0;
        for (uint32_t i = 0; i < output_size; ++i) {
            auto out = output_tensors_[i];
            size_t global[3] = {out->getDim().n, out->getDim().c, out->getDim().h * out->getDim().w};
            size_t local[3] = {1, 1, (size_t)(findMaxFactor(global[2], 128))};
            status = runtime_->setKernelArg(kernel_NC_.get(),
                                            input_tensor_->getDataPtr(),
                                            out->getDataPtr(),
                                            parameters_->axis,
                                            offset,
                                            input_tensor_->getDim().n,
                                            input_tensor_->getDim().c,
                                            input_tensor_->getDim().h,
                                            input_tensor_->getDim().w,
                                            out->getDim().n,
                                            out->getDim().c,
                                            out->getDim().h,
                                            out->getDim().w);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel failure\n");
            status = runtime_->enqueueKernel(kernel_NC_.get(), (cl_uint)3, global, local);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "execute kernel failure\n");
            if (parameters_->slice_point.size() != 0 && i < parameters_->slice_point.size()) {
                offset = parameters_->slice_point[i];
            } else {
                offset += offset_implicit;
            }
        }
    } else {
        int offset = 0;
        for (uint32_t i = 0; i < output_size; ++i) {
            auto out = output_tensors_[i];
            size_t global[3] = {out->getDim().n * out->getDim().c, out->getDim().h, out->getDim().w};
            size_t local[3] = {1, 1, (size_t)(findMaxFactor(global[2], 128))};
            Status state = runtime_->setKernelArg(kernel_HW_.get(),
                                                  input_tensor_->getDataPtr(),
                                                  out->getDataPtr(),
                                                  parameters_->axis,
                                                  offset,
                                                  input_tensor_->getDim().n,
                                                  input_tensor_->getDim().c,
                                                  input_tensor_->getDim().h,
                                                  input_tensor_->getDim().w,
                                                  out->getDim().n,
                                                  out->getDim().c,
                                                  out->getDim().h,
                                                  out->getDim().w);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
            state = runtime_->enqueueKernel(kernel_HW_.get(), (cl_uint)3, global, local);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
            if (parameters_->slice_point.size() != 0 && i < parameters_->slice_point.size()) {
                offset = parameters_->slice_point[i];
            } else {
                offset += offset_implicit;
            }
        }
    }

    return Status::SUCCESS;
}

Status CLSlice::release() {
    ENN_DBG_PRINT("CLSlice::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
