#include "CLTFSlice.hpp"

namespace enn {
namespace ud {
namespace gpu {
namespace {
constexpr int INPUT_INDEX = 0;
constexpr int BEGIN_INDEX = 1;
constexpr int SIZE_INDEX = 2;
constexpr int OUTPUT_INDEX = 0;
constexpr int kMaxNumOfDim = 4;
}  // namespace

CLTFSlice::CLTFSlice(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    ENN_DBG_PRINT("CLTFSlice is created\n");
    input_tensor_ = nullptr;
    begin_tensor_ = nullptr;
    size_tensor_ = nullptr;
    output_tensor_ = nullptr;
    parameters_ = nullptr;
    kernel_ = nullptr;
}

Status CLTFSlice::initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                             const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                             const std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLTFSlice::initialize() is called\n");

    input_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(INPUT_INDEX));
    begin_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(BEGIN_INDEX));
    size_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(SIZE_INDEX));
    output_tensor_ = std::static_pointer_cast<CLTensor>(output_tensors.at(OUTPUT_INDEX));
    parameters_ = std::static_pointer_cast<TFSliceParameters>(parameters);
    CHECK_EXPR_RETURN_FAILURE(nullptr != parameters_, "CLTFSlice must have parameters\n");
    ENN_DBG_PRINT("TFSliceParameters Info: androidNN = %d\n", parameters_->androidNN);

    parameters_->begin.resize(kMaxNumOfDim);
    parameters_->size.resize(kMaxNumOfDim);
    if (begin_tensor_->is_const()) {
        begin_tensor_->readData(parameters_->begin.data());
    }
    if (size_tensor_->is_const()) {
        size_tensor_->readData(parameters_->size.data());
    }

    Status status = runtime_->setKernel(&kernel_, "tf_slice", precision_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel tf_slice failure\n");

    return Status::SUCCESS;
}

Status CLTFSlice::execute() {
    ENN_DBG_PRINT("CLTFSlice::execute() is called\n");
    if (input_tensor_->getTotalSizeFromDims() == 0) {
        ENN_DBG_PRINT("CLTFSlice execute return for zero_sized input\n");
        return Status::SUCCESS;
    }

    if (!begin_tensor_->is_const()) {
        begin_tensor_->readData(parameters_->begin.data());
    }
    if (!size_tensor_->is_const()) {
        size_tensor_->readData(parameters_->size.data());
    }

    for (int i = begin_tensor_->getTotalSizeFromDims(); i < kMaxNumOfDim; ++i) {
        parameters_->begin[i] = 0;
    }
    for (int i = size_tensor_->getTotalSizeFromDims(); i < kMaxNumOfDim; ++i) {
        parameters_->size[i] = 1;
    }

    auto input_dim = input_tensor_->getDim();
    // If `size[i]` is -1, all remaining elements in dimension i are included in the slice.
    // In other words, this is equivalent to setting: `size[i] = input.dim_size(i) - begin[i]`
    for (int k = 0; k < kMaxNumOfDim; k++) {
        if (parameters_->size[k] == -1) {
            switch (k) {
            case 0: parameters_->size[0] = input_dim.n - parameters_->begin[0]; break;
            case 1: parameters_->size[1] = input_dim.c - parameters_->begin[1]; break;
            case 2: parameters_->size[2] = input_dim.h - parameters_->begin[2]; break;
            case 3: parameters_->size[3] = input_dim.w - parameters_->begin[3]; break;
            }
        }
    }

    // for dynamic output
    NDims ndim_out = std::vector<uint32_t>(input_tensor_->getNumOfDims());
    for (uint32_t i = 0; i < input_tensor_->getNumOfDims(); ++i) {
        CHECK_EXPR_RETURN_FAILURE(0 <= parameters_->begin[i] && parameters_->begin[i] < getDim(input_dim, i),
                                  "The beginning index (%d) is out of range\n",
                                  parameters_->begin[i]);
        CHECK_EXPR_RETURN_FAILURE(0 < parameters_->size[i] &&
                                      (parameters_->begin[i] + parameters_->size[i]) <= getDim(input_dim, i),
                                  "The size (%d) of the slice is out of range\n",
                                  parameters_->size[i]);
        ndim_out[i] = parameters_->size[i];
    }
    if (!isDimsSame(output_tensor_->getDims(), ndim_out)) {
        output_tensor_->reconfigureDimsAndBuffer(ndim_out);
    }

    Status status = Status::SUCCESS;

    status = runtime_->setKernelArg(kernel_.get(),
                                    input_tensor_->getDataPtr(),
                                    output_tensor_->getDataPtr(),
                                    input_dim.w,
                                    parameters_->begin[0],
                                    parameters_->begin[1],
                                    parameters_->begin[2],
                                    parameters_->begin[3],
                                    parameters_->size[0],
                                    parameters_->size[1],
                                    parameters_->size[2],
                                    parameters_->size[3]);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");

    size_t global[3] = {input_dim.n, input_dim.c, input_dim.h * input_dim.w};

    status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)3, global, NULL);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "execute kernel failure\n");

    return Status::SUCCESS;
}

Status CLTFSlice::release() {
    ENN_DBG_PRINT("CLTFSlice::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
