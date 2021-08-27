#include "CLSoftmax.hpp"

namespace enn {
namespace ud {
namespace gpu {
namespace {
constexpr int INPUT_INDEX = 0;
constexpr int OUTPUT_INDEX = 0;
}  // namespace

CLSoftmax::CLSoftmax(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    ENN_DBG_PRINT("CLSoftmax is created\n");
    input_tensor_ = nullptr;
    output_tensor_ = nullptr;
    parameters_ = nullptr;
    kernel_ = nullptr;
    dequantization_ = nullptr;
    quantization_ = nullptr;
    channel_max_tensor_ = nullptr;
    channel_sum_tensor_ = nullptr;
    map_input_tensor_ = nullptr;
    inner_number_ = 0;
    channels_ = 0;
    out_number_ = 0;
}

Status CLSoftmax::initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                             const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                             const std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLSoftmax::initialize() is called\n");

    input_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(INPUT_INDEX));
    output_tensor_ = std::static_pointer_cast<CLTensor>(output_tensors.at(OUTPUT_INDEX));
    parameters_ = std::static_pointer_cast<SoftmaxParameters>(parameters);
    CHECK_EXPR_RETURN_FAILURE(nullptr != parameters_, "CLSoftmax must have parameters\n");

    if (parameters_->androidNN && !isDimsSame(input_tensor_->getDims(), output_tensor_->getDims())) {
        output_tensor_->reconfigureDimsAndBuffer(input_tensor_->getDims());
    }

    const auto input_dim = input_tensor_->getDim();
    const auto output_dim = output_tensor_->getDim();

    const int tmp_axis = parameters_->axis < 0 ? parameters_->axis + 4 : parameters_->axis;

    switch (tmp_axis) {
    case 0: {
        inner_number_ = input_dim.c * input_dim.h * input_dim.w;
        channels_ = input_dim.n;
        out_number_ = 1;
        break;
    }
    case 1: {
        inner_number_ = input_dim.h * input_dim.w;
        channels_ = input_dim.c;
        out_number_ = input_dim.n;
        break;
    }
    case 2: {
        inner_number_ = input_dim.w;
        channels_ = input_dim.h;
        out_number_ = input_dim.n * input_dim.c;
        break;
    }
    default: {
        inner_number_ = 1;
        channels_ = input_dim.w;
        out_number_ = input_dim.n * input_dim.c * input_dim.h;
        break;
    }
    }

    channel_max_ptr_ = std::make_unique<float[]>(input_tensor_->getTotalSizeFromDims());
    map_input_tensor_ = std::make_shared<CLTensor>(runtime_, PrecisionType::FP32, DataType::FLOAT, input_dim);

    channel_max_tensor_ = std::make_shared<CLTensor>(runtime_, PrecisionType::FP32, DataType::FLOAT, input_dim);
    channel_sum_tensor_ = std::make_shared<CLTensor>(runtime_, PrecisionType::FP32, DataType::FLOAT, input_dim);

    Status status = Status::SUCCESS;
    if (precision_ == PrecisionType::INT8 || precision_ == PrecisionType::UINT8) {
        auto map_input_tensor = std::make_shared<CLTensor>(runtime_, PrecisionType::FP32, DataType::FLOAT, input_dim);
        auto map_output_tensor = std::make_shared<CLTensor>(runtime_, PrecisionType::FP32, DataType::FLOAT, output_dim);

        dequantization_ = std::make_shared<CLDeQuantization>(runtime_, PrecisionType::FP32);
        quantization_ = std::make_shared<CLQuantization>(runtime_, PrecisionType::FP32);

        auto dequantization_parameters = std::make_shared<DeQuantizationParameters>();
        status = dequantization_->initialize({input_tensor_}, {map_input_tensor}, dequantization_parameters);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "dequantization initialize failure\n");

        status = quantization_->initialize({map_output_tensor}, {output_tensor_}, nullptr);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "quantization initialize failure\n");

#ifdef BENCHMARK
        // for supporting AITuTu quant Inception-V3, which needs a float output in a quant model
        input_tensor_ = map_input_tensor;
#else
        input_tensor_ = map_input_tensor;
        output_tensor_ = map_output_tensor;
#endif

        status = runtime_->setKernel(&kernel_, "softmax_axis2", PrecisionType::FP32);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel softmax_axis2 failure\n");
    } else {
        if (parameters_->axis == 2) {
            status = runtime_->setKernel(&kernel_, "softmax_axis2", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel failure\n");
        } else if (parameters_->axis != 2 && channels_ % 8 == 0 && inner_number_ == 1 && !parameters_->adjustAcc) {
            status = runtime_->setKernel(&kernel_, "softmax", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel failure\n");
        }
    }

    return Status::SUCCESS;
}

Status CLSoftmax::execute() {
    ENN_DBG_PRINT("CLSoftmax::execute() is called\n");

    if (input_tensor_->getTotalSizeFromDims() == 0) {
        ENN_DBG_PRINT("CLSoftmax execute return for zero_sized input\n");
        return Status::SUCCESS;
    }

    if (precision_ == PrecisionType::UINT8 || precision_ == PrecisionType::INT8) {
        Status status = softmaxQuant();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "softmaxQuant execute failure\n");
    } else {
        Status status = softmaxFloat();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "softmaxFloat execute failure\n");
    }

    return Status::SUCCESS;
}

Status CLSoftmax::softmaxQuant() {
    ENN_DBG_PRINT("CLSoftmax::softmaxQuant() is called\n");
    Status status = Status::SUCCESS;

    status = dequantization_->execute();
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "dequantization execute failure\n");

    status = this->softmaxAxis2();
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "call softmaxAxis2 execute failure\n");

#ifndef BENCHMARK
    status = quantization_->execute();
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "quantization execute failure\n");
#endif

    return Status::SUCCESS;
}

Status CLSoftmax::softmaxFloat() {
    ENN_DBG_PRINT("CLSoftmax::softmaxFloat() is called\n");

    if (parameters_->axis == 2) {
        Status status = this->softmaxAxis2();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "call softmaxAxis2 execute failure\n");

        return Status::SUCCESS;
    }

    const uint32_t input_size = input_tensor_->getTotalSizeFromDims();
    const uint32_t dim = input_size / out_number_;

    Status status;
    if (channels_ % 8 == 0 && inner_number_ == 1 && !parameters_->adjustAcc) {
        status = runtime_->setKernelArg(
            kernel_.get(), input_tensor_->getDataPtr(), output_tensor_->getDataPtr(), channels_, dim, parameters_->beta);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");

        size_t global[1] = {out_number_};
        size_t local[1] = {1};

        status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)1, global, NULL);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "execute kernel failure\n");
    } else {
        const uint32_t input_size = input_tensor_->getTotalSizeFromDims();
        const uint32_t output_size = output_tensor_->getTotalSizeFromDims();
        auto channel_max = channel_max_ptr_.get();
        float channel_sum = 0.0;

        cl_float *tmp_input = nullptr;
        cl_float *tmp_output = nullptr;
        if (precision_ == PrecisionType::FP32) {
            tmp_input = static_cast<cl_float *>(clEnqueueMapBuffer(runtime_->getQueue(),
                                                                   input_tensor_->getDataPtr(),
                                                                   CL_TRUE,
                                                                   CL_MAP_READ | CL_MAP_WRITE,
                                                                   0,
                                                                   sizeof(cl_float) * input_size,
                                                                   0,
                                                                   NULL,
                                                                   NULL,
                                                                   NULL));
            tmp_output = static_cast<cl_float *>(clEnqueueMapBuffer(runtime_->getQueue(),
                                                                    output_tensor_->getDataPtr(),
                                                                    CL_TRUE,
                                                                    CL_MAP_READ | CL_MAP_WRITE,
                                                                    0,
                                                                    sizeof(cl_float) * output_size,
                                                                    0,
                                                                    NULL,
                                                                    NULL,
                                                                    NULL));
        } else {
            CHECK_EXPR_RETURN_FAILURE(input_size == output_size, "input_size must equal to output_size\n");
            status = runtime_->copyHalf2Float(map_input_tensor_->getDataPtr(), input_tensor_->getDataPtr(), input_size);
            tmp_input = static_cast<cl_float *>(clEnqueueMapBuffer(runtime_->getQueue(),
                                                                   map_input_tensor_->getDataPtr(),
                                                                   CL_TRUE,
                                                                   CL_MAP_READ | CL_MAP_WRITE,
                                                                   0,
                                                                   sizeof(cl_float) * input_size,
                                                                   0,
                                                                   NULL,
                                                                   NULL,
                                                                   NULL));
            tmp_output = tmp_input;
        }

        for (int j = 0; j < out_number_; j++) {
            int offset = j * dim;
            for (int i = 0; i < inner_number_; i++) {
                channel_max[i] = tmp_input[i + offset];
                for (int c = 1; c < channels_; c++) {
                    if (tmp_input[offset + i + inner_number_ * c] > channel_max[i]) {
                        channel_max[i] = tmp_input[offset + i + inner_number_ * c];
                    }
                }
            }
            for (int i = 0; i < inner_number_; i++) {
                channel_sum = 0.0;
                for (int c = 0; c < channels_; c++) {
                    int data_offset = offset + i + inner_number_ * c;
                    tmp_output[data_offset] = exp((tmp_input[data_offset] - channel_max[i]) * parameters_->beta);
                    channel_sum += tmp_output[data_offset];
                }

                for (int c = 0; c < channels_; c++) {
                    int data_offset = offset + i + inner_number_ * c;
                    tmp_output[data_offset] /= channel_sum;
                }
            }
        }

        if (precision_ == PrecisionType::FP32) {
            clEnqueueUnmapMemObject(runtime_->getQueue(), input_tensor_->getDataPtr(), tmp_input, 0, NULL, NULL);
            clEnqueueUnmapMemObject(runtime_->getQueue(), output_tensor_->getDataPtr(), tmp_output, 0, NULL, NULL);
        } else {
            clEnqueueUnmapMemObject(runtime_->getQueue(), map_input_tensor_->getDataPtr(), tmp_output, 0, NULL, NULL);
            status = runtime_->copyFloat2Half(output_tensor_->getDataPtr(), map_input_tensor_->getDataPtr(), output_size);
        }
    }

    return Status::SUCCESS;
}

Status CLSoftmax::softmaxAxis2() {
    ENN_DBG_PRINT("CLSoftmax::softmaxAxis2() is called\n");
    Status status = runtime_->setKernelArg(kernel_.get(),
                                           input_tensor_->getDataPtr(),
                                           output_tensor_->getDataPtr(),
                                           channel_max_tensor_->getDataPtr(),
                                           channel_sum_tensor_->getDataPtr(),
                                           channels_,
                                           inner_number_,
                                           out_number_,
                                           parameters_->beta);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");
    size_t global[1] = {out_number_};

    status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)1, global, NULL);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "executeKernel failure\n");
    return Status::SUCCESS;
}

Status CLSoftmax::release() {
    ENN_DBG_PRINT("CLSoftmax::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
