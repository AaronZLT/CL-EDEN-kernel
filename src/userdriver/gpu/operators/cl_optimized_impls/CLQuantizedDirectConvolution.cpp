#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLQuantizedDirectConvolution.hpp"

namespace enn {
namespace ud {
namespace gpu {

CLQuantizedDirectConvolution::CLQuantizedDirectConvolution(const std::shared_ptr<CLRuntime> runtime,
                                                           const PrecisionType &precision) {
    ENN_DBG_PRINT("CLQuantizedDirectConvolution::CLQuantizedDirectConvolution is called");
    runtime_ = runtime;
    precision_ = precision;
    direct_ = nullptr;
    computed_top_channel_numbers_ = DIRECT_TOP_CHANNEL_4;
    input_dim_ = {0, 0, 0, 0};
    weight_dim_ = {0, 0, 0, 0};
    output_dim_ = {0, 0, 0, 0};
    padding_ = {0, 0, 0, 0};
    stride_ = {0, 0};
    dilation_ = {0, 0};
    convert_input_dim_ = {0, 0, 0, 0};
    input_offset_ = 0;
    output_multiplier_ = 0;
    output_shift_ = 0;
    weights_as_input_ = false;
    bias_as_input_ = false;
    androidNN_ = false;
}

Status CLQuantizedDirectConvolution::initialize(const std::shared_ptr<ITensor> input,
                                                const Dim4 &output_dim,
                                                const Dim4 &weight_dim,
                                                const Pad4 &padding,
                                                const Dim2 &stride,
                                                const Dim2 &dilation,
                                                const std::shared_ptr<ITensor> weight,
                                                const std::shared_ptr<ITensor> bias,
                                                const ActivationInfo &activate_info,
                                                const bool &weights_as_input,
                                                const bool &bias_as_input,
                                                const bool &androidNN) {
    Status state = Status::FAILURE;
    weight_dim_ = weight->getDim();
    activation_info_ = activate_info;
    padding_ = padding;
    stride_ = stride;
    dilation_ = dilation;
    bias_ = std::static_pointer_cast<CLTensor>(bias);
    weight_ = std::static_pointer_cast<CLTensor>(weight);
    weights_as_input_ = weights_as_input;
    bias_as_input_ = bias_as_input;
    androidNN_ = androidNN;
    input_dim_ = input->getDim();
    output_dim_ = output_dim;

    if (androidNN_) {
        weight_dim_ = convertDimToNCHW(weight_dim_);
        weight_nchw_ = std::make_shared<CLTensor>(runtime_,
                                                  precision_,
                                                  weight_->getDataType(),
                                                  weight_dim_,
                                                  weight_->getDataOrder(),
                                                  weight_->getScale(),
                                                  weight_->getZeroPoint());
        if (!weights_as_input && !bias_as_input) {
            state = weight_->convertToNCHW(weight_nchw_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "convertToNCHW failure\n");
        }
    }
    Dim2 kernel_size = {weight_dim_.h, weight_dim_.w};

    computed_top_channel_numbers_ = DIRECT_TOP_CHANNEL_8;

    state = runtime_->setKernel(&align_weight_kernel_, "align_weight_direct", precision_);
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel align_weight_direct failed\n");
    if (weight_dim_.h == 3 && weight_dim_.w == 3) {
        state = runtime_->setKernel(&direct_, "direct3x3_8x4", precision_);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel direct3x3_8x4 failed!\n");
    } else if (weight_dim_.h == 5 && weight_dim_.w == 5) {
        state = runtime_->setKernel(&direct_, "direct5x5_8x4", precision_);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel direct3x3_8x4 failed!\n");
    } else if (weight_dim_.h == 7 && weight_dim_.w == 7) {
        state = runtime_->setKernel(&direct_, "direct7x7_8x4", precision_);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel direct3x3_8x4 failed!\n");
    } else if (weight_dim_.h == 9 && weight_dim_.w == 9) {
        state = runtime_->setKernel(&direct_, "direct9x9_8x4", precision_);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel direct9x9_8x4 failed!\n");
    } else {
        ERROR_PRINT("Non supported kernel size %dx%d in direct convolution\n", weight_dim_.h, weight_dim_.w);
        return Status::FAILURE;
    }

    // convert weight
    if (!weights_as_input && !bias_as_input) {
        weightConvert();
        moveWeightOffset2Bias(input->getZeroPoint(), weight_->getZeroPoint());
    }

    // pad input
    uint32_t aligned_c = ceil(static_cast<double>(input_dim_.c) / 4.0) * 4;
    convert_input_dim_ = {
        input_dim_.n, aligned_c, input_dim_.h + padding_.t + padding_.b, input_dim_.w + padding_.l + padding_.r};
    convert_input_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::UINT8, convert_input_dim_);

    auto convert_input_buffer = make_shared_array<uint8_t>(convert_input_->getNumOfBytes());
    unsigned char input_zero_point = (unsigned char)input->getZeroPoint();
    memset(convert_input_buffer.get(), input_zero_point, convert_input_->getNumOfBytes() * sizeof(uint8_t));

    convert_input_->writeData(convert_input_buffer.get());

    pad_convert_executor_ = std::make_shared<CLPadConvert>(runtime_, precision_);
    pad_convert_executor_->initialize(
        input_dim_, padding_, kernel_size, stride_, 1, output_dim, false, CLPadConvert::PadConvertType::QuantizedDirect);

    return Status::SUCCESS;
}

Status CLQuantizedDirectConvolution::weightConvert() {
    Status state = Status::FAILURE;
    int aligned_weight_w = weight_dim_.w * computed_top_channel_numbers_ * 4;
    int aligned_weight_h = weight_dim_.h;
    int aligned_weight_c = ceil(static_cast<double>(weight_dim_.c) / 4.0);
    int aligned_weight_n = ceil(static_cast<double>(weight_dim_.n) / computed_top_channel_numbers_);
    Dim4 alinged_weight_dim = {static_cast<uint32_t>(aligned_weight_n),
                               static_cast<uint32_t>(aligned_weight_c),
                               static_cast<uint32_t>(aligned_weight_h),
                               static_cast<uint32_t>(aligned_weight_w)};
    aligned_weight_ = std::make_shared<CLTensor>(runtime_, precision_, weight_->getDataType(), alinged_weight_dim);
    auto weight_data = androidNN_ ? weight_nchw_->getDataPtr() : weight_->getDataPtr();
    state = runtime_->setKernelArg(align_weight_kernel_.get(),
                                   weight_data,
                                   aligned_weight_->getDataPtr(),
                                   weight_dim_.n,
                                   weight_dim_.c,
                                   weight_dim_.h,
                                   weight_dim_.w,
                                   aligned_weight_n,
                                   aligned_weight_c,
                                   aligned_weight_h,
                                   aligned_weight_w,
                                   computed_top_channel_numbers_,
                                   weight_->getZeroPoint());
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernelArg align_weight_direct failed\n");

    size_t local[3] = {1, 1, 1};
    size_t global[3] = {static_cast<size_t>(aligned_weight_w),
                        static_cast<size_t>(aligned_weight_h),
                        static_cast<size_t>(aligned_weight_c * aligned_weight_n)};
    state = runtime_->enqueueKernel(align_weight_kernel_.get(), (cl_uint)3, global, local);
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "enqueue align_weight_direct failed\n");

    return state;
}
Status CLQuantizedDirectConvolution::moveWeightOffset2Bias(int inputZeroPoint, int filterZeroPoint) {
    Status status = Status::FAILURE;

    status = runtime_->setKernel(&update_bias_kernel_, "update_direct_conv_bias", precision_);
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == status, "setKernel update_direct_conv_bias failed\n");

    int kernel_size = weight_dim_.h * weight_dim_.w;
    auto weight_data = androidNN_ ? weight_nchw_->getDataPtr() : weight_->getDataPtr();
    status = runtime_->setKernelArg(update_bias_kernel_.get(),
                                    weight_data,
                                    bias_->getDataPtr(),
                                    inputZeroPoint,
                                    filterZeroPoint,
                                    input_dim_.c,
                                    kernel_size);
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == status, "setKernelArg update_bias_kernel_ failed\n");
    size_t local[1] = {1};
    size_t global[1] = {output_dim_.c};
    status = runtime_->enqueueKernel(update_bias_kernel_.get(), (cl_uint)1, global, local);
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == status, "enqueue update_bias_kernel_ failed\n");
    return status;
}

Status CLQuantizedDirectConvolution::execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) {
    ENN_DBG_PRINT("CLQuantizedDirectConvolution::execute is called\n");
    Status status = Status::FAILURE;
    if (weights_as_input_ || bias_as_input_) {
        if (androidNN_) {
            status = weight_->convertToNCHW(weight_nchw_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "convertToNCHW failure\n");
        }
        weightConvert();
        moveWeightOffset2Bias(input->getZeroPoint(), weight_->getZeroPoint());
    }

    auto input_tensor = std::static_pointer_cast<CLTensor>(input);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output);
    Dim4 output_dim = output_tensor->getDim();

    // calculate quantization parameter
    double real_multiplier = 0.0;
    GetQuantizedConvolutionMultipler(
        input_tensor->getScale(), weight_->getScale(), bias_->getScale(), output_tensor->getScale(), &real_multiplier);

    QuantizeMultiplierSmallerThanOneExp(real_multiplier, &output_multiplier_, &output_shift_);
    output_shift_ *= -1;

    int32_t act_min;
    int32_t act_max;
    if (activation_info_.isEnabled() == false) {
        act_min = std::numeric_limits<uint8_t>::min();
        act_max = std::numeric_limits<uint8_t>::max();
    } else {
        CalculateActivationRangeUint8(
            activation_info_.activation(), output_tensor->getScale(), output_tensor->getZeroPoint(), &act_min, &act_max);
    }

    // pad input
    pad_convert_executor_->quantizedDirectPadRun(
        input_tensor, convert_input_, padding_, (unsigned char)input_tensor->getZeroPoint());

    size_t local[3] = {16, 1, 1};
    size_t global[3] = {0, 0, 0};
    global[0] = alignTo(ceil(static_cast<double>(output_dim.w) / 4), local[0]);
    global[1] = alignTo(output_dim.h, local[1]);
    global[2] = alignTo(ceil(static_cast<double>(output_dim.c) / computed_top_channel_numbers_) * output_dim.n, local[2]);

    status = runtime_->setKernelArg(direct_.get(),
                                    convert_input_->getDataPtr(),
                                    aligned_weight_->getDataPtr(),
                                    bias_->getDataPtr(),
                                    output_tensor->getDataPtr(),
                                    padding_.t,
                                    padding_.l,
                                    output_dim.n,
                                    output_dim.c,
                                    output_dim.h,
                                    output_dim.w,
                                    convert_input_dim_.n,
                                    convert_input_dim_.c,
                                    convert_input_dim_.h,
                                    convert_input_dim_.w,
                                    input_tensor->getZeroPoint(),
                                    weight_->getZeroPoint(),
                                    output_tensor->getZeroPoint(),
                                    output_multiplier_,
                                    -output_shift_,
                                    act_min,
                                    act_max);
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == status, "setKernelArg direct_conv kernel failed\n");

    status = runtime_->enqueueKernel(direct_.get(), (cl_uint)3, global, local);
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == status, "enqueue direct_conv kernel failed\n");

    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
