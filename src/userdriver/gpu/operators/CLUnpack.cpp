#include "CLUnpack.hpp"

namespace enn {
namespace ud {
namespace gpu {
namespace {
constexpr int INPUT_INDEX = 0;
constexpr int OUTPUT_INDEX_0 = 0;
}  // namespace

CLUnpack::CLUnpack(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    ENN_DBG_PRINT("CLUnpack is created\n");
    input_tensor_ = nullptr;
    output_tensors_.clear();
    kernel_ = nullptr;
}

Status CLUnpack::initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                            const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                            const std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLUnpack::initialize() is called\n");

    input_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(INPUT_INDEX));
    for (auto output_tensor : output_tensors) {
        output_tensors_.push_back(std::static_pointer_cast<CLTensor>(output_tensor));
    }
    CHECK_AND_RETURN_ERR(output_tensors_.size() == 0, Status::FAILURE, "CLUnpack must have at least 1 output\n");
    parameters_ = std::static_pointer_cast<UnpackParameters>(parameters);
    CHECK_AND_RETURN_ERR(nullptr == parameters_, Status::FAILURE, "CLUnpack must have parameters\n");
    CHECK_AND_RETURN_ERR(parameters_->num != output_tensors_.size(),
                         Status::FAILURE,
                         "CLUnpack parameters' num must equal to output count\n");
    ENN_DBG_PRINT("UnpackParameters Info: num = %d, axis = %d\n", parameters_->num, parameters_->axis);

    if (parameters_->axis < 0)
        parameters_->axis += input_tensor_->getDims().size();

    Status status = runtime_->setKernel(&kernel_, "unpack", precision_);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "CLUnpack initialize is failed\n");

    return Status::SUCCESS;
}

Status CLUnpack::execute() {
    ENN_DBG_PRINT("CLUnpack::execute() is called\n");

    auto input_dim = input_tensor_->getDim();
    auto output_dim = output_tensors_[OUTPUT_INDEX_0]->getDim();
    const int dimensions = input_tensor_->getNumOfDims();

    std::vector<int> input_shapes;
    std::vector<int> output_shapes;
    input_shapes.clear();
    output_shapes.clear();
    for (int i = 0; i < dimensions; ++i) {
        input_shapes.push_back(getDim(input_dim, i));
        output_shapes.push_back(getDim(output_dim, i));
    }

    int outer_size = 1;
    for (int i = 0; i < parameters_->axis; ++i) {
        outer_size *= input_shapes[i];
    }

    int copy_size = 1;
    for (int i = parameters_->axis + 1; i < dimensions; ++i) {
        copy_size *= input_shapes[i];
    }

    Status status = Status::SUCCESS;

    size_t global = outer_size;
    size_t local = 1;
    for (int i = 0; i < parameters_->num; ++i) {
        status = runtime_->setKernelArg(kernel_.get(),
                                        input_tensor_->getDataPtr(),
                                        output_tensors_[i]->getDataPtr(),
                                        copy_size,
                                        parameters_->num,
                                        i * copy_size);
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "setKernelArg failure\n");

        status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)1, &global, &local);
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "execute kernel failure\n");
    }

    return Status::SUCCESS;
}

Status CLUnpack::release() {
    ENN_DBG_PRINT("CLUnpack::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
