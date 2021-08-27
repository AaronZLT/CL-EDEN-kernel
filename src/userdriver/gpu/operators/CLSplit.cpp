#include "userdriver/gpu/operators/CLSplit.hpp"

namespace enn {
namespace ud {
namespace gpu {

namespace {
constexpr int INPUT_INDEX = 0;
}  // namespace

CLSplit::CLSplit(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision), kernel_(nullptr) {
    ENN_DBG_PRINT("CLSplit is created");
    kernel_ = nullptr;
    input_tensor_ = nullptr;
    parameters_ = std::make_shared<SplitParameters>();
    output_tensors_.clear();
}

Status CLSplit::initialize(std::vector<std::shared_ptr<ITensor>> input_tensors,
                           std::vector<std::shared_ptr<ITensor>> output_tensors,
                           std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLSplit::initialize() is called");

    input_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(INPUT_INDEX));
    for (int i = 0; i != output_tensors.size(); ++i) {
        output_tensors_.push_back(std::static_pointer_cast<CLTensor>(output_tensors.at(i)));
    }

    parameters_ = std::static_pointer_cast<SplitParameters>(parameters);
    ENN_DBG_PRINT("SplitParameters: axis %d; num_outputs %d; androidNN %d\n",
                  parameters_->axis,
                  parameters_->num_outputs,
                  parameters_->androidNN);

    parameters_->axis = parameters_->axis < 0 ? parameters_->axis + input_tensor_->getDims().size() : parameters_->axis;

    if (parameters_->androidNN) {
        const uint32_t axis_size = input_tensor_->getDims(parameters_->axis);
        const uint32_t slice_size = axis_size / parameters_->num_outputs;
        NDims output_shape = input_tensor_->getDims();
        output_shape.at(parameters_->axis) = slice_size;
        for (auto iter = output_tensors.begin(); iter != output_tensors.end(); ++iter) {
            std::shared_ptr<ITensor> out_tensor = *iter;
            if (!isDimsSame(out_tensor->getDims(), output_shape)) {
                out_tensor->reconfigureDimsAndBuffer(output_shape);
            }
        }
    }

    Status state;
    const Dim4 &input_dim = input_tensor_->getDim();
    if (precision_ == PrecisionType::INT8) {
        state = runtime_->setKernel(&kernel_, "SIGNEDsplit", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel split failure\n");
    } else if (input_dim.n == 1 && parameters_->axis == 1 && parameters_->num_outputs == 2) {
        state = runtime_->setKernel(&kernel_, "split_2_2d_slices", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel split_2_2d_slices failure\n");
    } else {
        if (input_tensor_->getDataType() == DataType::INT32) {
            state = runtime_->setKernel(&kernel_, "split_int32", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel split failure\n");
        } else {
            state = runtime_->setKernel(&kernel_, "split", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel split failure\n");
        }
    }
    return Status::SUCCESS;
}

Status CLSplit::execute() {
    ENN_DBG_PRINT("CLSplit::execute() is called");
    auto input_tensor = std::static_pointer_cast<CLTensor>(input_tensor_);
    auto input_data = input_tensor->getDataPtr();
    std::vector<cl_mem> output_data;
    for (int i = 0; i != output_tensors_.size(); ++i) {
        output_data.push_back(output_tensors_[i]->getDataPtr());
    }

    auto input_dim = input_tensor->getDim();
    auto output_dim = output_tensors_[0]->getDim();

    size_t outerSize = input_tensor->getTotalSizeFromDims();
    std::vector<uint32_t> output_shapes = output_tensors_[0]->getDims();

    int baseInnerSize = 1;
    for (int i = parameters_->axis + 1; i < input_tensor->getNumOfDims(); ++i) {
        baseInnerSize *= input_tensor->getDims(i);
    }

    if (precision_ == PrecisionType::INT8) {
        size_t global = outerSize;
        size_t local = 1;

        const int copySize = output_shapes[parameters_->axis] * baseInnerSize;
        for (int i = 0; i != parameters_->num_outputs; ++i) {
            Status state = runtime_->setKernelArg(
                kernel_.get(), input_data, output_data[i], copySize, copySize * parameters_->num_outputs, i * copySize);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");

            state = runtime_->enqueueKernel(kernel_.get(), (cl_uint)1, &global, &local);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
        }
    } else if (input_dim.n == 1 && parameters_->axis == 1 && parameters_->num_outputs == 2) {
        size_t global[2] = {0, 0};
        size_t local[2] = {1, 24};

        global[0] = input_dim.c;
        global[1] = alignTo(ceil(input_dim.h * input_dim.w / 8.0), local[1]);

        int split_size = input_dim.c / 2;
        int copy_size = input_dim.h * input_dim.w;
        Status state =
            runtime_->setKernelArg(kernel_.get(), input_data, output_data[0], output_data[1], split_size, copy_size);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");

        state = runtime_->enqueueKernel(kernel_.get(), (cl_uint)2, global, local);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
    } else {
        size_t global = outerSize;
        size_t local = 1;

        const int copySize = output_shapes[parameters_->axis] * baseInnerSize;
        if (input_tensor->getDataType() == DataType::INT32) {
            for (int i = 0; i != parameters_->num_outputs; ++i) {
                Status state = runtime_->setKernelArg(
                    kernel_.get(), input_data, output_data[i], copySize, copySize * parameters_->num_outputs, i * copySize);
                CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");

                state = runtime_->enqueueKernel(kernel_.get(), (cl_uint)1, &global, &local);
                CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
            }
        } else {
            for (int i = 0; i != parameters_->num_outputs; ++i) {
                Status state = runtime_->setKernelArg(
                    kernel_.get(), input_data, output_data[i], copySize, copySize * parameters_->num_outputs, i * copySize);
                CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");

                state = runtime_->enqueueKernel(kernel_.get(), (cl_uint)1, &global, &local);
                CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
            }
        }
    }
    return Status::SUCCESS;
}

Status CLSplit::release() {
    ENN_DBG_PRINT("CLSplit::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
