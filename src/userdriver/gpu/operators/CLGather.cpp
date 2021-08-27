#include "CLGather.hpp"

namespace enn {
namespace ud {
namespace gpu {
namespace {
const uint32_t INPUT_INDEX = 0;
const uint32_t INDICES_INDEX = 1;
const uint32_t OUTPUT_INDEX = 0;
}  // namespace

CLGather::CLGather(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    ENN_DBG_PRINT("CLGather is created");
    input_tensor_ = nullptr;
    output_tensor_ = nullptr;
    indices_tensor_ = nullptr;
    parameters_ = nullptr;
    kernel_ = nullptr;
}

Status CLGather::initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                            const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                            const std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLGather::initialize() is called");
    input_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors[INPUT_INDEX]);
    output_tensor_ = std::static_pointer_cast<CLTensor>(output_tensors[OUTPUT_INDEX]);
    indices_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors[INDICES_INDEX]);
    parameters_ = std::static_pointer_cast<GatherParameters>(parameters);
    ENN_DBG_PRINT("GatherParameters: axis: %d androidNN: %d\n",  parameters_->axis, parameters_->androidNN);

    // compute output dims.
    if (parameters_->androidNN) {
        std::vector<uint32_t> output_shape;
        output_shape.insert(
            output_shape.end(), input_tensor_->getDims().begin(), input_tensor_->getDims().begin() + parameters_->axis);
        output_shape.insert(output_shape.end(), indices_tensor_->getDims().begin(), indices_tensor_->getDims().end());
        output_shape.insert(
            output_shape.end(), input_tensor_->getDims().begin() + parameters_->axis + 1, input_tensor_->getDims().end());
        if (!isDimsSame(output_tensor_->getDims(), output_shape)) {
            output_tensor_->reconfigureDimsAndBuffer(output_shape);
        }
    }

    if (parameters_->axis < 0)
        parameters_->axis += input_tensor_->getNumOfDims();

    if (precision_ == PrecisionType::INT8) {
        return runtime_->setKernel(&kernel_, "SIGNEDgather", precision_);
    } else if (precision_ != PrecisionType::UINT8 && input_tensor_->getDataType() == DataType::INT32) {
        Status status = runtime_->setKernel(&kernel_, "INT32gather", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel INT32gather failure\n");
    } else {
        Status status = runtime_->setKernel(&kernel_, "gather", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel gather failure\n");
    }

    return Status::SUCCESS;
}

Status CLGather::execute() {
    ENN_DBG_PRINT("CLGather::execute() is called");
    auto input_data = input_tensor_->getDataPtr();
    auto output_data = output_tensor_->getDataPtr();
    auto indices_data = indices_tensor_->getDataPtr();

    const int axis_size = input_tensor_->getDims().at(parameters_->axis);
    const int indices_count = indices_tensor_->getTotalSizeFromDims();

    NDims input_ndims = input_tensor_->getDims();
    int outer_size = 1;
    for (int i = 0; i < parameters_->axis; ++i) {
        outer_size *= input_ndims[i];
    }

    int inner_size = 1;
    for (int i = parameters_->axis + 1; i < input_tensor_->getNumOfDims(); ++i) {
        inner_size *= input_ndims[i];
    }

    Status status =
        runtime_->setKernelArg(kernel_.get(), input_data, indices_data, output_data, inner_size, indices_count, axis_size);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");

    size_t global[2] = {static_cast<size_t>(outer_size), static_cast<size_t>(indices_count)};
    size_t local[2] = {1, static_cast<size_t>(findMaxFactor(global[1], 128))};

    status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)2, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "execute kernel failure\n");
    return Status::SUCCESS;
}

Status CLGather::release() {
    DEBUG_PRINT("CLGather::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
