#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLDepthwiseConvolutionPerChannelQuantized.hpp"

namespace enn {
namespace ud {
namespace gpu {

CLDepthwiseConvolutionPerChannelQuantized::CLDepthwiseConvolutionPerChannelQuantized(
    const std::shared_ptr<CLRuntime> runtime,
    const PrecisionType &precision) :
    runtime_(runtime),
    precision_(precision) {
    ENN_DBG_PRINT("CLDepthwiseConvolutionPerChannelQuantized is created");
    input_dim_ = {0, 0, 0, 0};
    output_dim_ = {0, 0, 0, 0};
    stride_ = {0, 0};
    pad_ = {0, 0, 0, 0};
    kernel_ = {0, 0};
    depth_multiplier_ = 0;
    weights_as_input_ = false;
    androidNN_ = false;
}

Status CLDepthwiseConvolutionPerChannelQuantized::initialize(const Dim4 &input_dim,
                                                             const Dim4 &output_dim,
                                                             const std::shared_ptr<ITensor> filter,
                                                             const std::shared_ptr<ITensor> bias,
                                                             const Dim2 &stride,
                                                             const Pad4 &pad,
                                                             const uint32_t &depth_multiplier,
                                                             const ActivationInfo &activate_info,
                                                             const std::vector<float> &scales,
                                                             const bool &weight_as_input,
                                                             const bool &androidNN) {
    ENN_DBG_PRINT("CLDepthwiseConvolutionPerChannelQuantized::initialize() is called");
    stride_ = stride;
    pad_ = pad;
    depth_multiplier_ = depth_multiplier;
    activation_info_ = activate_info;
    weights_as_input_ = weight_as_input;
    androidNN_ = androidNN;

    Status state;
    filter_ = std::static_pointer_cast<CLTensor>(filter);
    Dim4 filter_dim = filter->getDim();
    if (androidNN_) {
        filter_dim = convertDimToNCHW(filter_dim);
        weight_nchw_ = std::make_shared<CLTensor>(runtime_,
                                                  precision_,
                                                  filter->getDataType(),
                                                  filter_dim,
                                                  filter->getDataOrder(),
                                                  filter->getScale(),
                                                  filter->getZeroPoint());

        if (!weights_as_input_) {
            state = filter_->convertToNCHW(weight_nchw_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "convertToNCHW failure\n");
        }
    }

    bias_ = std::static_pointer_cast<CLTensor>(bias);
    scales_ = scales;

    output_multiplier_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::INT32, bias->getDim());
    output_shift_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::INT32, bias->getDim());

    output_multiplier_data_.reset(new int32_t[output_dim.c], std::default_delete<int32_t[]>());
    output_shift_data_.reset(new int32_t[output_dim.c], std::default_delete<int32_t[]>());

    kernel_.h = filter->getDim().h;
    kernel_.w = filter->getDim().w;

    if (pad_.t != 0 || pad_.b != 0 || pad_.l != 0 || pad_.r != 0) {
        pad_buffer_h_ = input_dim.h + pad_.b + pad_.t;
        pad_buffer_w_ = input_dim.w + pad_.l + pad_.r;
        pad_size_ = pad_buffer_h_ * pad_buffer_w_;
        Dim4 dim_pad = {input_dim.n, input_dim.c, pad_buffer_h_, pad_buffer_w_};
        pad_buffer_ = std::make_shared<CLTensor>(runtime_, precision_, filter_->getDataType(), dim_pad);

        if (precision_ == PrecisionType::INT8) {
            state = runtime_->setKernel(&kernel_pad_, "SIGNEDpad", precision_);
        } else {
            state = runtime_->setKernel(&kernel_pad_, "pad", precision_);
        }
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "pad setKernel failure\n");
    }

    if (precision_ == PrecisionType::INT8) {
        state = runtime_->setKernel(&kernel_depthwise_conv_per_channel_, "SIGNEDdepthwise_conv_per_channel", precision_);
    } else {
        state = runtime_->setKernel(&kernel_depthwise_conv_per_channel_, "depthwise_conv_per_channel", precision_);
    }
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv_per_channel setKernel failure\n");

    return Status::SUCCESS;
}

Status CLDepthwiseConvolutionPerChannelQuantized::execute(const std::shared_ptr<ITensor> input,
                                                          std::shared_ptr<ITensor> output) {
    ENN_DBG_PRINT("CLDepthwiseConvolutionPerChannelQuantized::execute() is called");
    auto input_tensor = std::static_pointer_cast<CLTensor>(input);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output);
    auto input_data = input_tensor->getDataPtr();
    auto output_data = output_tensor->getDataPtr();

    Status state;
    if (androidNN_ && weights_as_input_) {
        state = filter_->convertToNCHW(weight_nchw_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "convertToNCHW failure\n");
    }

    auto filter_data = androidNN_ ? weight_nchw_->getDataPtr() : filter_->getDataPtr();
    auto bias_data = bias_->getDataPtr();

    auto input_dim = input_tensor->getDim();
    auto filter_dim = androidNN_ ? weight_nchw_->getDim() : filter_->getDim();
    auto output_dim = output_tensor->getDim();

    uint32_t num_batches = input_dim.n;
    uint32_t input_height = input_dim.h;
    uint32_t input_width = input_dim.w;
    uint32_t input_depth = input_dim.c;
    uint32_t filter_height = filter_dim.h;
    uint32_t filter_width = filter_dim.w;
    uint32_t output_height = output_dim.h;
    uint32_t output_width = output_dim.w;
    uint32_t output_depth = output_dim.c;

    auto input_zero_point =
        precision_ == PrecisionType::INT8 ? input_tensor->getZeroPoint() + 128 : input_tensor->getZeroPoint();
    auto filter_zero_point = precision_ == PrecisionType::INT8 ? filter_->getZeroPoint() + 128 : filter_->getZeroPoint();
    auto output_zero_point =
        precision_ == PrecisionType::INT8 ? output_tensor->getZeroPoint() + 128 : output_tensor->getZeroPoint();

    int32_t input_offset = -input_zero_point;
    int32_t weight_offset = -filter_zero_point;
    int32_t output_offset = output_zero_point;

    for (int i = 0; i < output_depth; i++) {
        double real_multiplier = 0.0;
        float filter_scale_channel = scales_[i];
        float bias_scale_channel = scales_[i] * input_tensor->getScale();
        GetQuantizedConvolutionMultipler(
            input_tensor->getScale(), filter_scale_channel, bias_scale_channel, output_tensor->getScale(), &real_multiplier);

        QuantizeMultiplierSmallerThanOneExp(
            real_multiplier, &output_multiplier_data_.get()[i], &output_shift_data_.get()[i]);
    }
    output_multiplier_->writeData(output_multiplier_data_.get());
    output_shift_->writeData(output_shift_data_.get());

    int32_t act_min = 0, act_max = 0;
    if (activation_info_.isEnabled() == false) {
        act_min = std::numeric_limits<uint8_t>::min();
        act_max = std::numeric_limits<uint8_t>::max();
    } else {
        CalculateActivationRangeUint8(
            activation_info_.activation(), output_tensor->getScale(), output_zero_point, &act_min, &act_max);
    }

    size_t global[3] = {0, 0, 0};
    size_t local[3] = {0, 0, 0};
    if (pad_.t == 0 && pad_.b == 0 && pad_.l == 0 && pad_.r == 0) {
        local[0] = 1;
        local[1] = 1;
        local[2] = 32;
        int align_out = alignTo(output_dim.h * output_dim.w, local[2]);
        global[0] = num_batches;
        global[1] = input_depth;
        global[2] = align_out;

        state = runtime_->setKernelArg(kernel_depthwise_conv_per_channel_.get(),
                                       input_data,
                                       filter_data,
                                       bias_data,
                                       output_data,
                                       input_height,
                                       input_width,
                                       filter_height,
                                       filter_width,
                                       stride_.h,
                                       stride_.w,
                                       output_height,
                                       output_width,
                                       depth_multiplier_,
                                       input_offset,
                                       weight_offset,
                                       output_offset,
                                       output_multiplier_->getDataPtr(),
                                       output_shift_->getDataPtr(),
                                       act_min,
                                       act_max);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv_per_channel setKernel failure\n");
    } else {
        if (precision_ == PrecisionType::INT8) {
            char zero = input_zero_point;
            state = runtime_->setKernelArg(kernel_pad_.get(),
                                           input_data,
                                           pad_buffer_->getDataPtr(),
                                           zero,
                                           pad_.t,
                                           pad_.r,
                                           pad_.b,
                                           pad_.l,
                                           input_dim.w,
                                           input_dim.h);
        } else {
            unsigned char zero = input_zero_point;
            state = runtime_->setKernelArg(kernel_pad_.get(),
                                           input_data,
                                           pad_buffer_->getDataPtr(),
                                           zero,
                                           pad_.t,
                                           pad_.r,
                                           pad_.b,
                                           pad_.l,
                                           input_dim.w,
                                           input_dim.h);
        }
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "pad setKernelArg failure\n");

        size_t global_pad[3] = {input_dim.n, input_dim.c, 0};
        size_t local_pad[3] = {1, 1, 32};
        global_pad[2] = alignTo(pad_size_, local_pad[2]);
        state = runtime_->enqueueKernel(kernel_pad_.get(), (cl_uint)3, global_pad, local_pad);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "pad enqueueKernel failure\n");

        local[0] = 1;
        local[1] = 1;
        local[2] = 32;
        int align_out = alignTo(output_dim.h * output_dim.w, local[2]);
        global[0] = num_batches;
        global[1] = input_depth;
        global[2] = align_out;

        state = runtime_->setKernelArg(kernel_depthwise_conv_per_channel_.get(),
                                       pad_buffer_->getDataPtr(),
                                       filter_data,
                                       bias_data,
                                       output_data,
                                       pad_buffer_h_,
                                       pad_buffer_w_,
                                       filter_height,
                                       filter_width,
                                       stride_.h,
                                       stride_.w,
                                       output_height,
                                       output_width,
                                       depth_multiplier_,
                                       input_offset,
                                       weight_offset,
                                       output_offset,
                                       output_multiplier_->getDataPtr(),
                                       output_shift_->getDataPtr(),
                                       act_min,
                                       act_max);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv_per_channel setKernel failure\n");
    }
    state = runtime_->enqueueKernel(kernel_depthwise_conv_per_channel_.get(), (cl_uint)3, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv_per_channel enqueueKernel failure\n");

    return Status::SUCCESS;
}

Status CLDepthwiseConvolutionPerChannelQuantized::release() {
    ENN_DBG_PRINT("CLDepthwiseConvolutionPerChannelQuantized::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
