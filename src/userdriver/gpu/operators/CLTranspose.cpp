#include "CLTranspose.hpp"

namespace enn {
namespace ud {
namespace gpu {
namespace {
constexpr int INPUT_INDEX = 0;
constexpr int PERM_INDEX = 1;
constexpr int OUTPUT_INDEX = 0;
}  // namespace

CLTranspose::CLTranspose(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    ENN_DBG_PRINT("CLTranspose is created\n");
    input_tensor_ = nullptr;
    perm_tensor_ = nullptr;
    output_tensor_ = nullptr;
    parameters_ = nullptr;
    kernel_ = nullptr;
}

Status CLTranspose::initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                               const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                               const std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLTranspose::initialize() is called\n");

    input_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(INPUT_INDEX));
    perm_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(PERM_INDEX));
    output_tensor_ = std::static_pointer_cast<CLTensor>(output_tensors.at(OUTPUT_INDEX));
    parameters_ = std::static_pointer_cast<TransposeParameters>(parameters);
    CHECK_AND_RETURN_ERR(nullptr == parameters_, Status::FAILURE, "CLTranspose must have parameters\n");

    parameters_->perm.resize(perm_tensor_->getTotalSizeFromDims());
    perm_tensor_->readData(parameters_->perm.data());

    if (parameters_->perm.empty()) {
        for (int32_t idx = 0; idx < input_tensor_->getNumOfDims(); ++idx) {
            parameters_->perm.push_back(input_tensor_->getNumOfDims() - 1 - idx);
        }
    }

    if (parameters_->androidNN) {
        NDims output_shape;
        for (auto p : parameters_->perm) {
            output_shape.push_back(input_tensor_->getDims(p));
        }
        if (output_tensor_->getDims() != output_shape) {
            output_tensor_->reconfigureDimsAndBuffer(output_shape);
        }
    }

    Status status = runtime_->setKernel(&kernel_, "transpose", precision_);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "setKernel transpose failure\n");

    return Status::SUCCESS;
}

Status CLTranspose::execute() {
    ENN_DBG_PRINT("CLTranspose::execute() is called\n");

    if (input_tensor_->getTotalSizeFromDims() == 0) {
        ENN_DBG_PRINT("CLTranspose execute return for zero_sized input\n");
        return Status::SUCCESS;
    }

    auto input_dim = input_tensor_->getDim();
    auto output_dim = output_tensor_->getDim();

    const int32_t kOutputDimensionNum = 4;
    const int32_t size = parameters_->perm.size();
    int32_t complete_perm[kOutputDimensionNum];
    uint32_t out_sizes[4];
    for (int32_t output_k = 0; output_k < size; ++output_k) {
        complete_perm[output_k] = parameters_->perm[output_k];
    }
    for (int32_t k = size; k < kOutputDimensionNum; ++k) {
        complete_perm[k] = k;
    }
    for (int32_t k = 0; k < 4; k++) {
        out_sizes[k] = getDim(input_dim, complete_perm[k]);
    }

    Status status = runtime_->setKernelArg(kernel_.get(),
                                           input_tensor_->getDataPtr(),
                                           output_tensor_->getDataPtr(),
                                           complete_perm[0],
                                           complete_perm[1],
                                           complete_perm[2],
                                           complete_perm[3],
                                           input_dim.n,
                                           input_dim.c,
                                           input_dim.h,
                                           input_dim.w,
                                           out_sizes[1],
                                           out_sizes[2],
                                           out_sizes[3]);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "setKernelArg failure\n");

    size_t local = 48;
    size_t global = alignTo(ceil(input_dim.n * input_dim.c * input_dim.h * input_dim.w / 8.0), local);

    status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)1, &global, &local);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "execute kernel failure\n");

    return Status::SUCCESS;
}

Status CLTranspose::release() {
    ENN_DBG_PRINT("CLTranspose::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
