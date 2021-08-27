#include "userdriver/gpu/operators/cl_optimized_impls/CLConvolutionPerChannelQuantized.hpp"

namespace enn {
namespace ud {
namespace gpu {

CLConvolutionPerChannelQuantized::CLConvolutionPerChannelQuantized(const std::shared_ptr<CLRuntime> runtime,
                                                                   const PrecisionType &precision) :
    runtime_(runtime),
    precision_(precision) {
    ENN_DBG_PRINT("CLConvolutionPerChannelQuantized is created");
    input_dim_ = {0, 0, 0, 0};
    output_dim_ = {0, 0, 0, 0};
    stride_ = {0, 0};
    kernel_ = {0, 0};
    padding_ = {0, 0, 0, 0};
    group_ = 0;
    output_multiplier_ = nullptr;
    output_shift_ = nullptr;
    androidNN_ = false;
}

Status CLConvolutionPerChannelQuantized::initialize(const Dim4 &input_dim,
                                                    const Dim4 &output_dim,
                                                    const Pad4 &padding,
                                                    const Dim2 &stride,
                                                    const uint32_t &group_size,
                                                    const std::shared_ptr<ITensor> weight,
                                                    const std::shared_ptr<ITensor> bias,
                                                    const ActivationInfo &activate_info,
                                                    const std::vector<float> &scales,
                                                    const bool &androidNN) {
    ENN_DBG_PRINT("CLConvolutionPerChannelQuantized::initialize() is called");
    androidNN_ = androidNN;
    auto weight_tensor = std::static_pointer_cast<CLTensor>(weight);
    Dim4 weight_size = weight_tensor->getDim();

    if (androidNN_) {
        weight_size = convertDimToNCHW(weight_size);
        weight_nchw_ = std::make_shared<CLTensor>(runtime_,
                                                  precision_,
                                                  weight_tensor->getDataType(),
                                                  weight_size,
                                                  weight_tensor->getDataOrder(),
                                                  weight_tensor->getScale(),
                                                  weight_tensor->getZeroPoint());
    }

    kernel_ = {weight_size.h, weight_size.w};
    filter_ = weight_tensor;
    auto bias_tensor = std::static_pointer_cast<CLTensor>(bias);
    bias_ = bias_tensor;

    input_dim_ = input_dim;
    output_dim_ = output_dim;
    stride_ = stride;
    padding_ = padding;
    group_ = group_size;
    scales_ = scales;
    activation_info_ = activate_info;
    output_multiplier_data_.reset(new int[output_dim_.c], std::default_delete<int[]>());
    output_shift_data_.reset(new int[output_dim_.c], std::default_delete<int[]>());
    Dim4 dim_quant = {output_dim.c, 1, 1, 1};
    output_multiplier_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::INT32, dim_quant);
    output_shift_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::INT32, dim_quant);

    if (precision_ == PrecisionType::INT8) {
        return runtime_->setKernel(&per_channel_quantized_kernel_, "SIGNEDconv_per_channel_quantized", precision_);
    } else {
        return runtime_->setKernel(&per_channel_quantized_kernel_, "conv_per_channel_quantized", precision_);
    }
}

Status CLConvolutionPerChannelQuantized::execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) {
    ENN_DBG_PRINT("CLConvolutionPerChannelQuantized is execute");
    auto input_tensor = std::static_pointer_cast<CLTensor>(input);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output);
    auto input_data = input_tensor->getDataPtr();
    auto bias_data = bias_->getDataPtr();
    auto output_data = output_tensor->getDataPtr();
    if (androidNN_) {
        Status state = filter_->convertToNCHW(weight_nchw_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "convertToNCHW failure\n");
    }
    auto weight_data = androidNN_ ? weight_nchw_->getDataPtr() : filter_->getDataPtr();

    for (int i = 0; i < output_dim_.c; ++i) {
        double real_multiplier = 0.0;
        GetQuantizedConvolutionMultipler(input_tensor->getScale(),
                                         scales_[i],
                                         input_tensor->getScale() * scales_[i],
                                         output_tensor->getScale(),
                                         &real_multiplier);
        QuantizeMultiplierSmallerThanOneExp(
            real_multiplier, &output_multiplier_data_.get()[i], &output_shift_data_.get()[i]);
    }

    output_multiplier_->writeData(output_multiplier_data_.get());
    output_shift_->writeData(output_shift_data_.get());

    int32_t act_min;
    int32_t act_max;
    if (activation_info_.isEnabled() == false) {
        if (precision_ == PrecisionType::INT8) {
            act_min = std::numeric_limits<int8_t>::min();
            act_max = std::numeric_limits<int8_t>::max();
        } else {
            act_min = std::numeric_limits<uint8_t>::min();
            act_max = std::numeric_limits<uint8_t>::max();
        }

    } else {
        if (precision_ == PrecisionType::INT8) {
            CalculateActivationRangeInt8(
                activation_info_.activation(), output_tensor->getScale(), output_tensor->getZeroPoint(), &act_min, &act_max);
        } else {
            CalculateActivationRangeUint8(
                activation_info_.activation(), output_tensor->getScale(), output_tensor->getZeroPoint(), &act_min, &act_max);
        }
    }
    size_t global[3];
    global[0] = output_dim_.n;
    global[1] = output_dim_.c / group_;
    global[2] = output_dim_.h * output_dim_.w;

    Status state;
    for (int i = 0; i < group_; i++) {
        state = runtime_->setKernelArg(per_channel_quantized_kernel_.get(),
                                       input_data,
                                       weight_data,
                                       bias_data,
                                       output_multiplier_->getDataPtr(),
                                       output_shift_->getDataPtr(),
                                       output_data,
                                       group_,
                                       i,
                                       kernel_.h,
                                       kernel_.w,
                                       stride_.h,
                                       stride_.w,
                                       padding_.l,
                                       padding_.t,
                                       output_dim_.c,
                                       output_dim_.h,
                                       output_dim_.w,
                                       input_dim_.c,
                                       input_dim_.h,
                                       input_dim_.w,
                                       -input_tensor->getZeroPoint(),
                                       0,
                                       output_tensor->getZeroPoint(),
                                       act_min,
                                       act_max);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "set kernelArg failure\n");
        state = runtime_->enqueueKernel(per_channel_quantized_kernel_.get(), 3, global, NULL);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
    }

    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
