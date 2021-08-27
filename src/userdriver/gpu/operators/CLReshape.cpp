#include "userdriver/gpu/operators/cl_quantized_utils/KernelUtil.hpp"
#include "CLReshape.hpp"

namespace enn {
namespace ud {
namespace gpu {
namespace {
constexpr int INPUT_INDEX = 0;
constexpr int SHAPE_INDEX = 1;
constexpr int OUTPUT_INDEX = 0;
}  // namespace

CLReshape::CLReshape(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    ENN_DBG_PRINT("CLReshape is created\n");
    input_tensor_ = nullptr;
    shape_tensor_ = nullptr;
    output_tensor_ = nullptr;
    parameters_ = nullptr;
    kernel_ = nullptr;
}

Status CLReshape::initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                             const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                             const std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLReshape::initialize() is called\n");

    input_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(INPUT_INDEX));
    shape_tensor_ = input_tensors.size() >= 2 ? std::static_pointer_cast<CLTensor>(input_tensors.at(SHAPE_INDEX)) : nullptr;
    output_tensor_ = std::static_pointer_cast<CLTensor>(output_tensors.at(OUTPUT_INDEX));
    parameters_ = std::static_pointer_cast<ReshapeParameters>(parameters);
    CHECK_AND_RETURN_ERR(nullptr == parameters_, Status::FAILURE, "CLReshape must have parameters\n");

    Status status = Status::SUCCESS;

    if (parameters_->androidNN) {
        if (parameters_->new_shape.empty()) {
            if (shape_tensor_ && shape_tensor_->is_const()) {
                const int32_t target_dim_size = shape_tensor_->getTotalSizeFromDims();
                if (shape_tensor_->getDataType() == DataType::INT32) {
                    parameters_->new_shape.resize(target_dim_size);
                    status = shape_tensor_->readData(parameters_->new_shape.data());
                    CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "compute_output_shape readData failure\n");
                } else {
                    ENN_ERR_PRINT("compute_output_shape shape_tensor_ don't support DataType: %d!",
                                  shape_tensor_->getDataType());
                    return Status::FAILURE;
                }
            }
        }

        if (!parameters_->new_shape.empty()) {
            status = compute_output_shape();
            CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "CLReshape compute_output_shape failure\n");
        }
    }

    if (parameters_->compute_type == ComputeType::Caffe || parameters_->androidNN || !parameters_->isNCHW) {
        if (precision_ == PrecisionType::INT8) {
            status = runtime_->setKernel(&kernel_, "SIGNEDreshape", precision_);
            CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "setKernel SIGNEDreshape failure\n");
        }
        if (precision_ != PrecisionType::UINT8 && input_tensor_->getDataType() == DataType::INT32) {
            status = runtime_->setKernel(&kernel_, "INT32reshape", precision_);
            CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "setKernel INT32reshape failure\n");
        } else {
            status = runtime_->setKernel(&kernel_, "reshape", precision_);
            CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "setKernel reshape failure\n");
        }
    } else {
        if (precision_ == PrecisionType::INT8) {
            status = runtime_->setKernel(&kernel_, "SIGNEDreshape_tflite", precision_);
            CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "setKernel SIGNEDreshape_tflite failure\n");
        }
        status = runtime_->setKernel(&kernel_, "reshape_tflite", precision_);
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "setKernel reshape_tflite failure\n");
    }

    return Status::SUCCESS;
}

Status CLReshape::execute() {
    ENN_DBG_PRINT("CLReshape::execute() is called\n");
    if (parameters_->androidNN) {
        if (parameters_->new_shape.empty()) {
            if (shape_tensor_ && !shape_tensor_->is_const()) {
                ENN_DBG_PRINT("CLReshape execute with weights_as_input\n");
                const int32_t target_dim_size = shape_tensor_->getTotalSizeFromDims();
                if (shape_tensor_->getDataType() == DataType::INT32) {
                    parameters_->new_shape.resize(target_dim_size);
                    Status status = shape_tensor_->readData(parameters_->new_shape.data());
                    CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "compute_output_shape readData failure\n");
                } else {
                    ENN_ERR_PRINT("compute_output_shape shape_tensor_ don't support DataType: %d!",
                                  shape_tensor_->getDataType());
                    return Status::FAILURE;
                }

                Status status = compute_output_shape();
                CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "CLReshape compute_output_shape failure\n");
            }
        }
    }

    if (precision_ == PrecisionType::UINT8 || precision_ == PrecisionType::INT8) {
        Status status = reshapeQuant();
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "reshapeQuant execute failure\n");
    } else {
        Status status = reshapeFloat();
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "reshapeFloat execute failure\n");
    }

    return Status::SUCCESS;
}

Status CLReshape::reshapeQuant() {
    ENN_DBG_PRINT("CLReshape::reshapeQuant() is called");
    const auto input_dim = input_tensor_->getDim();
    const auto output_dim = output_tensor_->getDim();

    Status status = Status::SUCCESS;
    if (parameters_->compute_type == ComputeType::Caffe || parameters_->androidNN) {
        status = runtime_->setKernelArg(
            kernel_.get(), input_tensor_->getDataPtr(), output_tensor_->getDataPtr(), input_dim.c, input_dim.h, input_dim.w);
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "setKernelArg failure\n");

        const int gsize2 = ceil(input_dim.h * input_dim.w / 8.0);
        size_t local[3] = {1, 1, 32};
        size_t global[3] = {0, 0, 0};
        global[0] = input_dim.n;
        global[1] = alignTo(input_dim.c, local[1]);
        global[2] = alignTo(gsize2, local[2]);

        status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)3, global, local);
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "enqueueKernel failure\n");
    } else {
        status = runtime_->setKernelArg(kernel_.get(),
                                        input_tensor_->getDataPtr(),
                                        output_tensor_->getDataPtr(),
                                        input_dim.n,
                                        input_dim.c,
                                        input_dim.h,
                                        input_dim.w,
                                        output_dim.n,
                                        output_dim.c,
                                        output_dim.h,
                                        output_dim.w);
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "setKernelArg failure\n");

        size_t local[3] = {1, 1, 32};
        size_t global[3] = {input_dim.n, input_dim.c, input_dim.h * input_dim.w};
        local[0] = 1;
        local[1] = findMaxFactor(input_dim.c, 128 / local[2]);
        local[2] = findMaxFactor(input_dim.h * input_dim.w, 128);

        status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)3, global, local);
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "enqueueKernel failure\n");
    }

    return Status::SUCCESS;
}

Status CLReshape::reshapeFloat() {
    ENN_DBG_PRINT("CLReshape::reshapeFloat() is called");
    const auto input_dim = input_tensor_->getDim();
    const auto output_dim = output_tensor_->getDim();

    Status status = Status::SUCCESS;
    if (parameters_->compute_type == ComputeType::Caffe || parameters_->androidNN || !parameters_->isNCHW) {
        status = runtime_->setKernelArg(
            kernel_.get(), input_tensor_->getDataPtr(), output_tensor_->getDataPtr(), input_dim.c, input_dim.h, input_dim.w);
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "setKernelArg failure\n");

        const int gsize2 = ceil(input_dim.h * input_dim.w / 8.0);
        size_t local[3] = {1, 1, 32};
        size_t global[3] = {0, 0, 0};
        global[0] = input_dim.n;
        global[1] = alignTo(input_dim.c, local[1]);
        global[2] = alignTo(gsize2, local[2]);

        status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)3, global, local);
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "enqueueKernel failure\n");
    } else {
        status = runtime_->setKernelArg(kernel_.get(),
                                        input_tensor_->getDataPtr(),
                                        output_tensor_->getDataPtr(),
                                        input_dim.n,
                                        input_dim.c,
                                        input_dim.h,
                                        input_dim.w,
                                        output_dim.n,
                                        output_dim.c,
                                        output_dim.h,
                                        output_dim.w);
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "setKernelArg failure\n");

        size_t local[3] = {1, 1, 24};
        size_t global[3] = {input_dim.n, input_dim.c, 0};
        global[2] = alignTo(input_dim.h * input_dim.w, local[2]);

        status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)3, global, local);
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "enqueueKernel failure\n");
    }

    return Status::SUCCESS;
}

Status CLReshape::compute_output_shape() {
    ENN_DBG_PRINT("CLReshape::compute_output_shape() is called");
    Status status = Status::SUCCESS;

    const int32_t target_dim_size = parameters_->new_shape.size();

    NDims out_dim(target_dim_size);
    int32_t num_output_element = 1;
    int32_t strech_dim = -1;
    for (int32_t i = 0; i < target_dim_size; i++) {
        int32_t value = parameters_->new_shape[i];
        if (value == -1) {
            strech_dim = i;
        } else {
            num_output_element *= value;
            out_dim[i] = value;
        }
    }
    if (strech_dim != -1) {
        int32_t strech_value = input_tensor_->getTotalSizeFromDims() / num_output_element;
        out_dim[strech_dim] = strech_value;
        num_output_element *= strech_value;
    }

    CHECK_AND_RETURN_ERR(input_tensor_->getTotalSizeFromDims() != num_output_element,
                         Status::FAILURE,
                         "input_tensor_->getTotalSizeFromDims() != num_output_element Error!\n");

    if (!isDimsSame(out_dim, output_tensor_->getDims())) {
        output_tensor_->reconfigureDimsAndBuffer(out_dim);
    }
    output_tensor_->setZeroPoint(input_tensor_->getZeroPoint());
    output_tensor_->setScale(input_tensor_->getScale());

    return Status::SUCCESS;
}

Status CLReshape::release() {
    ENN_DBG_PRINT("CLReshape::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
