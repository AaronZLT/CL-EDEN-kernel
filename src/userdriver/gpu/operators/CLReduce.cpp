#include "userdriver/gpu/operators/CLReduce.hpp"

namespace enn {
namespace ud {
namespace gpu {

namespace {
const uint32_t INPUT_INDEX = 0;
const uint32_t AXIS_INDEX = 1;
const uint32_t OUTPUT_INDEX = 0;
}  // namespace

CLReduce::CLReduce(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    ENN_DBG_PRINT("CLReduce is created");

    input_tensor_ = nullptr;
    axis_tensor_ = nullptr;
    output_tensor_ = nullptr;
    map_tensor_ = nullptr;
    reduce_kernel_ = nullptr;
    output_kernel_ = nullptr;
    quantization_ = nullptr;
    dequantization_ = nullptr;
    parameters_ = std::make_shared<ReduceParameters>();
    axis_.clear();
}

Status CLReduce::initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                            const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                            const std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLReduce::initialize() is called");

    input_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(INPUT_INDEX));
    axis_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(AXIS_INDEX));
    output_tensor_ = std::static_pointer_cast<CLTensor>(output_tensors.at(OUTPUT_INDEX));

    parameters_ = std::static_pointer_cast<ReduceParameters>(parameters);
    ENN_DBG_PRINT(
        "ReduceParameters: keep_dims %d; parameters_->reducer: %d\n", parameters_->keep_dims, parameters_->reducer);

    std::shared_ptr<int32_t> tmp_axis = make_shared_array<int32_t>(axis_tensor_->getTotalSizeFromDims());
    axis_tensor_->readData(tmp_axis.get());

    for (int i = 0; i < axis_tensor_->getTotalSizeFromDims(); i++) {
        axis_.push_back(tmp_axis.get()[i]);
    }

    if (axis_.empty()) {
        for (uint32_t ax = 0; ax < input_tensor_->getNumOfDims(); ++ax) {
            axis_.push_back(ax);
        }
    } else {
        for (auto it = axis_.begin(); it != axis_.end(); ++it) {
            *it = *it < 0 ? *it + input_tensor_->getNumOfDims() : *it;
        }
        sort(axis_.begin(), axis_.end());
        axis_.erase(unique(axis_.begin(), axis_.end()), axis_.end());
    }

    NDims output_dim;
    NDims input_dim = input_tensor_->getDims();
    if (parameters_->keep_dims) {
        for (uint32_t i = 0; i < input_dim.size(); i++) {
            if (std::find(axis_.begin(), axis_.end(), i) == axis_.end())
                output_dim.push_back(input_dim[i]);
            else
                output_dim.push_back(1);
        }
    } else {
        for (uint32_t i = 0; i < input_dim.size(); i++) {
            if (std::find(axis_.begin(), axis_.end(), i) == axis_.end())
                output_dim.push_back(input_dim[i]);
        }
    }
    if (output_dim.empty()) {
        output_dim = {1};
    }
    if (!isDimsSame(output_dim, output_tensor_->getDims())) {
        output_tensor_->reconfigureDimsAndBuffer(output_dim);
    }
    Status state;

    for (int it = axis_.size(); it > 0; it--) {
        int32_t ax = axis_.at(it - 1);
        if (input_dim.at(ax) > 1) {
            if (precision_ == PrecisionType::INT8) {
                state = runtime_->setKernel(&reduce_kernel_, "SIGNED_" + kernel_str(parameters_->reducer), precision_);
                CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel SIGNED_reduce failure\n");
            } else {
                state = runtime_->setKernel(&reduce_kernel_, kernel_str(parameters_->reducer), precision_);
                CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel kernel_Reduce_ failure\n");
            }
            break;
        }
    }
    if (precision_ == PrecisionType::INT8) {
        state = runtime_->setKernel(&output_kernel_, "SIGNED_" + kernel_str(parameters_->reducer) + "_output", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel SIGNED_reduce_output failure\n");
    } else {
        state = runtime_->setKernel(&output_kernel_, kernel_str(parameters_->reducer) + "_output", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel kernel_Reduce_ failure\n");
    }

    if (precision_ == PrecisionType::UINT8 || precision_ == PrecisionType::INT8) {
        auto input_tensor = std::static_pointer_cast<CLTensor>(input_tensor_);
        auto output_tensor = std::static_pointer_cast<CLTensor>(output_tensor_);
        const float &input_scale = input_tensor->getScale();
        const float &output_scale = output_tensor->getScale();
        const int32_t &input_zero_point = input_tensor->getZeroPoint();
        const int32_t &output_zero_point = output_tensor->getZeroPoint();
        if (input_scale != output_scale || input_zero_point != output_zero_point) {
            map_tensor_ = std::make_shared<CLTensor>(runtime_, PrecisionType::FP32, DataType::FLOAT, input_tensor->getDim());

            quantization_ = std::make_shared<CLQuantization>(runtime_, PrecisionType::FP32);
            state = quantization_->initialize({map_tensor_}, {input_tensor}, nullptr);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "quantization initialize failure\n");

            auto dequantization_parameters = std::make_shared<DeQuantizationParameters>();
            dequantization_parameters->per_channel_quant = false;
            dequantization_parameters->channel_dim = 0;
            dequantization_parameters->scales.clear();
            dequantization_ = std::make_shared<CLDeQuantization>(runtime_, PrecisionType::FP32);
            state = dequantization_->initialize({input_tensor}, {map_tensor_}, dequantization_parameters);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "dequantization_input_ initialize failure\n");
        }
    }

    return Status::SUCCESS;
}

Status CLReduce::execute() {
    ENN_DBG_PRINT("CLReduce::execute() is called");
    auto in_tensor = std::static_pointer_cast<CLTensor>(input_tensor_);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output_tensor_);

    Status state;
    auto input_tensor =
        std::make_shared<CLTensor>(runtime_, precision_, input_tensor_->getDataType(), input_tensor_->getDim());
    size_t copy_bytes = in_tensor->getNumOfBytes();
    size_t offset = 0;

    state = runtime_->copyBuffer(input_tensor->getDataPtr(), in_tensor->getDataPtr(), offset, offset, copy_bytes);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "copyBuffer failure\n");

    if (precision_ == PrecisionType::UINT8 || precision_ == PrecisionType::INT8) {
        input_tensor->setScale(std::static_pointer_cast<CLTensor>(input_tensor_)->getScale());
        input_tensor->setZeroPoint(std::static_pointer_cast<CLTensor>(input_tensor_)->getZeroPoint());
        if (input_tensor->getScale() != output_tensor->getScale() ||
            input_tensor->getZeroPoint() != output_tensor->getZeroPoint()) {
            state = dequantization_->execute();
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "dequantization execute failure\n");

            input_tensor->setScale(output_tensor->getScale());
            input_tensor->setZeroPoint(output_tensor->getZeroPoint());
            state = quantization_->execute();
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "quantization execute failure\n");
        }
    }

    return eval(input_tensor, output_tensor);
}

Status CLReduce::eval(std::shared_ptr<CLTensor> input, std::shared_ptr<CLTensor> output) {
    auto in_dim = input->getDim();
    uint32_t dm_dim[4] = {in_dim.n, in_dim.c, in_dim.h, in_dim.w};  // dynamic dim
    // do reduction by axis in input mem
    for (int it = axis_.size(); it > 0; it--) {
        int32_t ax = axis_[it - 1];
        uint32_t k = 1;
        for (uint32_t i = ax + 1; i < 4; i++) {
            k *= getDim(in_dim, i);
        }
        while (dm_dim[ax] > 1) {
            uint32_t offset = dm_dim[ax] - (dm_dim[ax] >> 1);
            if (offset * 2 > dm_dim[ax]) {
                dm_dim[ax] = offset - 1;
            } else {
                dm_dim[ax] = offset;
            }
            uint32_t offset_k = offset * k;
            size_t global = input->getTotalSizeFromDims();

            Status state = runtime_->setKernelArg(reduce_kernel_.get(),
                                                  input->getDataPtr(),
                                                  offset_k,
                                                  in_dim.n,
                                                  in_dim.c,
                                                  in_dim.h,
                                                  in_dim.w,
                                                  dm_dim[0],
                                                  dm_dim[1],
                                                  dm_dim[2],
                                                  dm_dim[3]);
            dm_dim[ax] = offset;
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
            state = runtime_->enqueueKernel(reduce_kernel_.get(), (cl_uint)1, &global, NULL);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
        }
    }

    // write result to output mem
    size_t global = input->getTotalSizeFromDims();
    Status state = runtime_->setKernelArg(output_kernel_.get(),
                                          input->getDataPtr(),
                                          output->getDataPtr(),
                                          in_dim.n,
                                          in_dim.c,
                                          in_dim.h,
                                          in_dim.w,
                                          dm_dim[0],
                                          dm_dim[1],
                                          dm_dim[2],
                                          dm_dim[3]);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
    state = runtime_->enqueueKernel(output_kernel_.get(), (cl_uint)1, &global, NULL);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
    return Status::SUCCESS;
}

Status CLReduce::release() {
    ENN_DBG_PRINT("CLReduce::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
