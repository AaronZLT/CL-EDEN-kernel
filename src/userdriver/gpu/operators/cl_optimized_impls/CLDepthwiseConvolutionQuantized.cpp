#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLDepthwiseConvolutionQuantized.hpp"

namespace enn {
namespace ud {
namespace gpu {

CLDepthwiseConvolutionQuantized::CLDepthwiseConvolutionQuantized(const std::shared_ptr<CLRuntime> runtime,
                                                                 const PrecisionType &precision) :
    runtime_(runtime),
    precision_(precision) {
    ENN_DBG_PRINT("CLDepthwiseConvolutionQuantized is created");
    stride_ = {0, 0};
    pad_ = {0, 0, 0, 0};
    kernel_ = {0, 0};
    dialation_kernel_ = {0, 0};
    depth_multiplier_ = 0;
    dilation_ = {1, 1};
    weights_as_input_ = false;
    androidNN_ = false;
    is_dilation_ = false;
}

Status CLDepthwiseConvolutionQuantized::dilation_weight(const std::shared_ptr<CLTensor> weight_tensor) {
    is_dilation_ = true;
    uint32_t batch_weight = weight_tensor->getDim().n;
    uint32_t channel_weight = weight_tensor->getDim().c;
    uint32_t extent_height = dilation_.h * (kernel_.h - 1) + 1;
    uint32_t extent_width = dilation_.w * (kernel_.w - 1) + 1;
    uint32_t weight_dilation_size = batch_weight * channel_weight * extent_height * extent_width;

    Dim4 weight_dilation_dim = {weight_dilation_size, 1, 1, 1};
    auto weight_dilation = std::make_shared<CLTensor>(runtime_,
                                                      precision_,
                                                      weight_tensor->getDataType(),
                                                      weight_dilation_dim,
                                                      weight_tensor->getDataOrder(),
                                                      weight_tensor->getScale(),
                                                      weight_tensor->getZeroPoint());

    Status state;
    size_t global_init = weight_dilation_size;
    std::shared_ptr<struct _cl_kernel> kernel_dilation_init;
    if (precision_ == PrecisionType::INT8) {
        state = runtime_->setKernel(&kernel_dilation_init, "SIGNEDdilation_init", precision_);
    } else {
        state = runtime_->setKernel(&kernel_dilation_init, "dilation_init", precision_);
    }
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
    state = runtime_->setKernelArg(kernel_dilation_init.get(), weight_dilation->getDataPtr(), weight_tensor->getZeroPoint());
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");
    state = runtime_->enqueueKernel(kernel_dilation_init.get(), (cl_uint)1, &global_init, NULL);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "enqueueKernel failure\n");

    size_t global[3];
    global[0] = batch_weight * channel_weight;
    global[1] = kernel_.h;
    global[2] = kernel_.w;
    std::shared_ptr<struct _cl_kernel> kernel_dilation;
    if (precision_ == PrecisionType::INT8) {
        state = runtime_->setKernel(&kernel_dilation, "SIGNEDdilation", precision_);
    } else {
        state = runtime_->setKernel(&kernel_dilation, "dilation", precision_);
    }
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
    state = runtime_->setKernelArg(kernel_dilation.get(),
                                   weight_tensor->getDataPtr(),
                                   weight_dilation->getDataPtr(),
                                   extent_height,
                                   extent_width,
                                   dilation_.h,
                                   dilation_.w);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");
    state = runtime_->enqueueKernel(kernel_dilation.get(), (cl_uint)3, global, NULL);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "enqueueKernel failure\n");

    dilation_filter_ = weight_dilation;

    return Status::SUCCESS;
}

Status CLDepthwiseConvolutionQuantized::initialize(const Dim4 &input_dim,
                                                   const Dim4 &output_dim,
                                                   const std::shared_ptr<ITensor> filter,
                                                   const std::shared_ptr<ITensor> bias,
                                                   const Dim2 &stride,
                                                   const Pad4 &pad,
                                                   const uint32_t &depth_multiplier,
                                                   const Dim2 &dilation,
                                                   const ActivationInfo &activate_info,
                                                   const bool &weight_as_input,
                                                   const bool &androidNN) {
    ENN_DBG_PRINT("CLDepthwiseConvolutionQuantized::initialize() is called");
    stride_ = stride;
    pad_ = pad;
    depth_multiplier_ = depth_multiplier;
    dilation_.h = dilation.h;
    dilation_.w = dilation.w;
    weights_as_input_ = weight_as_input;
    androidNN_ = androidNN;
    activation_info_ = activate_info;

    bias_ = std::static_pointer_cast<CLTensor>(bias);
    filter_ = std::static_pointer_cast<CLTensor>(filter);

    Status state;
    Dim4 filter_dim = filter->getDim();
    if (androidNN_) {
        filter_dim = convertDimToNCHW(filter_dim);
        filter_nchw_ = std::make_shared<CLTensor>(runtime_,
                                                  precision_,
                                                  filter->getDataType(),
                                                  filter_dim,
                                                  filter->getDataOrder(),
                                                  filter->getScale(),
                                                  filter->getZeroPoint());
        if (weights_as_input_ == false) {
            state = filter_->convertToNCHW(filter_nchw_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "convertToNCHW failure\n");
        }
    }

    kernel_.h = filter_dim.h;
    kernel_.w = filter_dim.w;

    dialation_kernel_.h = dilation.h * (kernel_.h - 1) + 1;
    dialation_kernel_.w = dilation.w * (kernel_.w - 1) + 1;

    if (pad_.t != 0 || pad_.b != 0 || pad_.l != 0 || pad_.r != 0) {
        padbuffer_h_ = input_dim.h + pad_.b + pad_.t;
        padbuffer_w_ = input_dim.w + pad_.l + pad_.r;
        size_pad_ = padbuffer_h_ * padbuffer_w_;
        Dim4 dim_pad = {input_dim.n, input_dim.c, padbuffer_h_, padbuffer_w_};
        padbuffer_ = std::make_shared<CLTensor>(runtime_, precision_, filter_->getDataType(), dim_pad);

        if (precision_ == PrecisionType::INT8) {
            state = runtime_->setKernel(&kernel_pad_, "SIGNEDpad", precision_);
        } else {
            state = runtime_->setKernel(&kernel_pad_, "pad", precision_);
        }
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "pad setKernel failure\n");
    }

    if (!weights_as_input_ && (dilation_.h > 1 || dilation_.w > 1) && !(kernel_.h == 1 && kernel_.w == 1)) {
        auto weight_tensor = std::static_pointer_cast<CLTensor>(androidNN_ ? filter_nchw_ : filter_);
        dilation_weight(weight_tensor);
    }

    if (precision_ == PrecisionType::INT8) {
        state = runtime_->setKernel(&depthwisekernel_, "SIGNEDdepthwise_conv", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "SIGNEDdepthwise_conv setKernel failure\n");

        if (input_dim.c * depth_multiplier_ < output_dim.c) {
            state = runtime_->setKernel(&kernel_unequal_, "SIGNEDdepthwise_conv_unequal", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "SIGNEDdepthwise_conv_unequal setKernel failure\n");
        }
    } else {
        if (depth_multiplier_ == 1 && dilation_.h == 1 && dilation_.w == 1 && kernel_.h == 3 && kernel_.w == 3 &&
            stride_.h == 1 && stride_.w == 1) {
            state = runtime_->setKernel(&depthwisekernel_, "depthwise_conv_3x3s1_pad_merge", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv_3x3s1_pad_merge setKernel failure\n");
        } else if (depth_multiplier_ == 1 && dilation_.h == 1 && dilation_.w == 1 && kernel_.h == 3 && kernel_.w == 3 &&
                   stride_.h == 2 && stride_.w == 2) {
            state = runtime_->setKernel(&depthwisekernel_, "depthwise_conv_3x3s2_pad_merge", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv_3x3s2_pad_merge setKernel failure\n");
        } else if (depth_multiplier_ == 1 && dilation_.h == 2 && dilation_.w == 2 && kernel_.h == 3 && kernel_.w == 3 &&
                   stride_.h == 1 && stride_.w == 1) {
            state = runtime_->setKernel(&depthwisekernel_, "depthwise_conv_3x3s1d2_pad_merge", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv_3x3s1d2_pad_merge setKernel failure\n");
        } else if (kernel_.h == 3 && kernel_.w == 3 && stride_.h == 1 && stride_.w == 1 && dilation_.h == 1 &&
                   dilation_.w == 1) {
            state = runtime_->setKernel(&depthwisekernel_, "depthwise_conv_3x3s1_4P", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv_3x3s1_4P setKernel failure\n");
        } else if (kernel_.h == 3 && kernel_.w == 3 && stride_.h == 2 && stride_.w == 2 && dilation_.h == 1 &&
                   dilation_.w == 1) {
            state = runtime_->setKernel(&depthwisekernel_, "depthwise_conv_3x3s2", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv_3x3s2 setKernel failure\n");
        } else {
            state = runtime_->setKernel(&depthwisekernel_, "depthwise_conv", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv setKernel failure\n");
        }

        if (input_dim.c * depth_multiplier_ < output_dim.c) {
            state = runtime_->setKernel(&kernel_unequal_, "depthwise_conv_unequal", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv_unequal setKernel failure\n");
        }
    }

    return Status::SUCCESS;
}

Status CLDepthwiseConvolutionQuantized::execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) {
    ENN_DBG_PRINT("CLDepthwiseConvolutionQuantized::execute() is called");
    Status state;
    if (androidNN_ && weights_as_input_) {
        state = filter_->convertToNCHW(filter_nchw_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "convertToNCHW failure\n");
    }

    auto input_tensor = std::static_pointer_cast<CLTensor>(input);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output);
    auto input_data = input_tensor->getDataPtr();
    auto output_data = output_tensor->getDataPtr();
    auto filter_data = androidNN_ ? filter_nchw_->getDataPtr() : filter_->getDataPtr();
    auto bias_data = bias_->getDataPtr();

    auto input_zero_point =
        precision_ == PrecisionType::INT8 ? input_tensor->getZeroPoint() + 128 : input_tensor->getZeroPoint();
    auto filter_zero_point = precision_ == PrecisionType::INT8 ? filter_->getZeroPoint() + 128 : filter_->getZeroPoint();
    auto output_zero_point =
        precision_ == PrecisionType::INT8 ? output_tensor->getZeroPoint() + 128 : output_tensor->getZeroPoint();

    if ((weights_as_input_ == true) && (dilation_.h > 1 || dilation_.w > 1) && !(kernel_.h == 1 && kernel_.w == 1)) {
        dilation_weight(androidNN_ ? filter_nchw_ : filter_);
    }

    double real_multiplier = 0.0;
    GetQuantizedConvolutionMultipler(
        input_tensor->getScale(), filter_->getScale(), bias_->getScale(), output_tensor->getScale(), &real_multiplier);
    int32_t output_multiplier;
    int output_shift;
    QuantizeMultiplierSmallerThanOneExp(real_multiplier, &output_multiplier, &output_shift);
    output_shift *= -1;

    int32_t act_min;
    int32_t act_max;

    if (activation_info_.isEnabled() == false) {
        act_min = std::numeric_limits<uint8_t>::min();
        act_max = std::numeric_limits<uint8_t>::max();
    } else {
        CalculateActivationRangeUint8(
            activation_info_.activation(), output_tensor->getScale(), output_zero_point, &act_min, &act_max);
    }

    int32_t input_offset = -input_zero_point;
    int32_t weight_offset = -filter_zero_point;
    int32_t output_offset = output_zero_point;

    auto input_dim = input_tensor->getDim();
    auto output_dim = output_tensor->getDim();
    if (pad_.t == 0 && pad_.b == 0 && pad_.l == 0 && pad_.r == 0) {
        if (is_dilation_) {
            state = excute_kernels_depends(input_data,
                                           dilation_filter_->getDataPtr(),
                                           bias_data,
                                           input_dim,
                                           output_dim,
                                           input_dim.h,
                                           input_dim.w,
                                           dialation_kernel_.h,
                                           dialation_kernel_.w,
                                           &output_data,
                                           input_offset,
                                           weight_offset,
                                           output_offset,
                                           output_multiplier,
                                           output_shift,
                                           act_min,
                                           act_max);
        } else {
            state = excute_kernels_depends(input_data,
                                           filter_data,
                                           bias_data,
                                           input_dim,
                                           output_dim,
                                           input_dim.h,
                                           input_dim.w,
                                           kernel_.h,
                                           kernel_.w,
                                           &output_data,
                                           input_offset,
                                           weight_offset,
                                           output_offset,
                                           output_multiplier,
                                           output_shift,
                                           act_min,
                                           act_max);
        }
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "pad setKernelArg failure\n");
    } else {
        if (precision_ == PrecisionType::INT8) {
            char zero = input_tensor->getZeroPoint();
            state = runtime_->setKernelArg(kernel_pad_.get(),
                                           input_data,
                                           padbuffer_->getDataPtr(),
                                           zero,
                                           pad_.t,
                                           pad_.r,
                                           pad_.b,
                                           pad_.l,
                                           input_dim.w,
                                           input_dim.h);
        } else {
            unsigned char zero = input_tensor->getZeroPoint();
            state = runtime_->setKernelArg(kernel_pad_.get(),
                                           input_data,
                                           padbuffer_->getDataPtr(),
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
        global_pad[2] = alignTo(size_pad_, local_pad[2]);

        if (is_dilation_) {
            if (depth_multiplier_ == 1 && dilation_.h == 2 && dilation_.w == 2 && kernel_.h == 3 && kernel_.w == 3 &&
                stride_.h == 1 && stride_.w == 1) {
                state = excute_kernels_depends(input_data,
                                               filter_data,
                                               bias_data,
                                               input_dim,
                                               output_dim,
                                               input_dim.h,
                                               input_dim.w,
                                               kernel_.h,
                                               kernel_.w,
                                               &output_data,
                                               input_offset,
                                               weight_offset,
                                               output_offset,
                                               output_multiplier,
                                               output_shift,
                                               act_min,
                                               act_max);
            } else {
                state = runtime_->enqueueKernel(kernel_pad_.get(), (cl_uint)3, global_pad, local_pad);
                CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "pad enqueueKernel failure\n");

                state = excute_kernels_depends(padbuffer_->getDataPtr(),
                                               dilation_filter_->getDataPtr(),
                                               bias_data,
                                               input_dim,
                                               output_dim,
                                               padbuffer_h_,
                                               padbuffer_w_,
                                               dialation_kernel_.h,
                                               dialation_kernel_.w,
                                               &output_data,
                                               input_offset,
                                               weight_offset,
                                               output_offset,
                                               output_multiplier,
                                               output_shift,
                                               act_min,
                                               act_max);
            }
        } else if ((depth_multiplier_ == 1 && dilation_.h == 1 && dilation_.w == 1 && kernel_.h == 3 && kernel_.w == 3) &&
                   ((stride_.h == 1 && stride_.w == 1) || (stride_.h == 2 && stride_.w == 2))) {
            state = excute_kernels_depends(input_data,
                                           filter_data,
                                           bias_data,
                                           input_dim,
                                           output_dim,
                                           input_dim.h,
                                           input_dim.w,
                                           kernel_.h,
                                           kernel_.w,
                                           &output_data,
                                           input_offset,
                                           weight_offset,
                                           output_offset,
                                           output_multiplier,
                                           output_shift,
                                           act_min,
                                           act_max);
        } else {
            state = runtime_->enqueueKernel(kernel_pad_.get(), (cl_uint)3, global_pad, local_pad);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "pad enqueueKernel failure\n");

            state = excute_kernels_depends(padbuffer_->getDataPtr(),
                                           filter_data,
                                           bias_data,
                                           input_dim,
                                           output_dim,
                                           padbuffer_h_,
                                           padbuffer_w_,
                                           kernel_.h,
                                           kernel_.w,
                                           &output_data,
                                           input_offset,
                                           weight_offset,
                                           output_offset,
                                           output_multiplier,
                                           output_shift,
                                           act_min,
                                           act_max);
        }

        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "excute_kernels_depends failure\n");
    }

    return Status::SUCCESS;
}

Status CLDepthwiseConvolutionQuantized::release() {
    ENN_DBG_PRINT("CLDepthwiseConvolutionQuantized::release() is called");
    return Status::SUCCESS;
}

Status CLDepthwiseConvolutionQuantized::excute_kernels_depends(const cl_mem &input_data,
                                                               const cl_mem &filter_data,
                                                               const cl_mem &bias_data,
                                                               const Dim4 &input_dim,
                                                               const Dim4 &output_dim,
                                                               const uint32_t &input_h,
                                                               const uint32_t &input_w,
                                                               const uint32_t &filter_h,
                                                               const uint32_t &filter_w,
                                                               cl_mem *output_data,
                                                               const int32_t &input_offset,
                                                               const int32_t &weight_offset,
                                                               const int32_t &output_offset,
                                                               const int32_t &output_multiplier,
                                                               const int32_t &output_shift,
                                                               const int32_t &act_min,
                                                               const int32_t &act_max) {
    size_t global[3] = {0, 0, 0};
    size_t local[3] = {0, 0, 0};
    Status state;

    if (precision_ == PrecisionType::INT8) {
        local[0] = 1;
        local[1] = 1;
        local[2] = 32;
        int align_out = alignTo(output_dim.h * output_dim.w, local[2]);
        global[0] = output_dim.n;
        global[1] = input_dim.c;
        global[2] = align_out;

        state = runtime_->setKernelArg(depthwisekernel_.get(),
                                       input_data,
                                       filter_data,
                                       bias_data,
                                       *output_data,
                                       input_h,
                                       input_w,
                                       filter_h,
                                       filter_w,
                                       stride_.h,
                                       stride_.w,
                                       output_dim.h,
                                       output_dim.w,
                                       depth_multiplier_,
                                       input_offset,
                                       weight_offset,
                                       output_offset,
                                       output_multiplier,
                                       output_shift,
                                       act_min,
                                       act_max);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "SIGNEDdepthwise_conv setKernel failure\n");
    } else {
        if ((depth_multiplier_ == 1 && dilation_.h == 1 && dilation_.w == 1 && kernel_.h == 3 && kernel_.w == 3) &&
            ((stride_.h == 1 && stride_.w == 1))) {
            local[0] = 2;
            local[1] = 1;
            local[2] = 8;
            global[0] = alignTo(output_dim.n * input_dim.c, local[0]);
            global[1] = alignTo(ceil(output_dim.h / 4.0), local[1]);
            global[2] = alignTo(ceil(output_dim.w / 4.0), local[2]);

            state = runtime_->setKernelArg(depthwisekernel_.get(),
                                           input_data,
                                           filter_data,
                                           bias_data,
                                           *output_data,
                                           input_h,
                                           input_w,
                                           output_dim.h,
                                           output_dim.w,
                                           output_dim.c,
                                           depth_multiplier_,
                                           pad_.t,
                                           pad_.l,
                                           input_offset,
                                           weight_offset,
                                           output_offset,
                                           output_multiplier,
                                           output_shift,
                                           act_min,
                                           act_max);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv_3x3s1_pad_merge setKernelArg failure\n");
        } else if (depth_multiplier_ == 1 && dilation_.h == 1 && dilation_.w == 1 && kernel_.h == 3 && kernel_.w == 3 &&
                   stride_.h == 2 && stride_.w == 2) {
            local[0] = 1;
            local[1] = 1;
            local[2] = 16;
            int align_height = alignTo(output_dim.h, local[1]);
            int align_width = alignTo(output_dim.w, local[2]);
            global[0] = output_dim.n * input_dim.c;
            global[1] = align_height;
            global[2] = align_width;

            state = runtime_->setKernelArg(depthwisekernel_.get(),
                                           input_data,
                                           filter_data,
                                           bias_data,
                                           *output_data,
                                           input_h,
                                           input_w,
                                           output_dim.h,
                                           output_dim.w,
                                           output_dim.c,
                                           depth_multiplier_,
                                           pad_.t,
                                           pad_.l,
                                           input_offset,
                                           weight_offset,
                                           output_offset,
                                           output_multiplier,
                                           output_shift,
                                           act_min,
                                           act_max);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv_3x3s1_pad_merge setKernelArg failure\n");
        } else if (depth_multiplier_ == 1 && dilation_.h == 2 && dilation_.w == 2 && kernel_.h == 3 && kernel_.w == 3 &&
                   stride_.h == 1 && stride_.w == 1) {
            local[0] = 1;
            local[1] = 2;
            local[2] = 8;
            int align_height = alignTo(output_dim.h, local[1]);
            int align_width = alignTo(ceil(output_dim.w / 4.0), local[2]);
            global[0] = output_dim.n * input_dim.c;
            global[1] = align_height;
            global[2] = align_width;

            state = runtime_->setKernelArg(depthwisekernel_.get(),
                                           input_data,
                                           filter_data,
                                           bias_data,
                                           *output_data,
                                           input_h,
                                           input_w,
                                           output_dim.h,
                                           output_dim.w,
                                           output_dim.c,
                                           depth_multiplier_,
                                           pad_.t,
                                           pad_.l,
                                           input_offset,
                                           weight_offset,
                                           output_offset,
                                           output_multiplier,
                                           output_shift,
                                           act_min,
                                           act_max);

            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv_3x3s1_pad_merge setKernelArg failure\n");
        } else if (filter_h == 3 && filter_w == 3 && stride_.h == 1 && stride_.w == 1 && dilation_.h == 1 &&
                   dilation_.w == 1) {
            local[0] = 1;
            local[1] = 1;
            local[2] = 16;
            int align_height = alignTo(output_dim.h, 2 * local[1]);
            int align_width = alignTo(output_dim.w, 2 * local[2]);
            global[0] = output_dim.n * input_dim.c;
            global[1] = align_height / 2;
            global[2] = align_width / 2;

            state = runtime_->setKernelArg(depthwisekernel_.get(),
                                           input_data,
                                           filter_data,
                                           bias_data,
                                           *output_data,
                                           input_h,
                                           input_w,
                                           output_dim.h,
                                           output_dim.w,
                                           output_dim.c,
                                           depth_multiplier_,
                                           input_offset,
                                           weight_offset,
                                           output_offset,
                                           output_multiplier,
                                           output_shift,
                                           act_min,
                                           act_max);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv_3x3s1_4P setKernelArg failure\n");
        } else if (filter_h == 3 && filter_w == 3 && stride_.h == 2 && stride_.w == 2 && dilation_.h == 1 &&
                   dilation_.w == 1) {
            local[0] = 1;
            local[1] = 1;
            local[2] = 32;
            int align_out = alignTo(output_dim.h * output_dim.w, local[2]);
            global[0] = output_dim.n;
            global[1] = input_dim.c;
            global[2] = align_out;

            state = runtime_->setKernelArg(depthwisekernel_.get(),
                                           input_data,
                                           filter_data,
                                           bias_data,
                                           *output_data,
                                           input_h,
                                           input_w,
                                           output_dim.h,
                                           output_dim.w,
                                           output_dim.c,
                                           depth_multiplier_,
                                           input_offset,
                                           weight_offset,
                                           output_offset,
                                           output_multiplier,
                                           output_shift,
                                           act_min,
                                           act_max);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv_3x3s2 setKernel failure\n");
        } else {
            local[0] = 1;
            local[1] = 1;
            local[2] = 32;
            int align_out = alignTo(output_dim.h * output_dim.w, local[2]);
            global[0] = output_dim.n;
            global[1] = input_dim.c;
            global[2] = align_out;

            state = runtime_->setKernelArg(depthwisekernel_.get(),
                                           input_data,
                                           filter_data,
                                           bias_data,
                                           *output_data,
                                           input_h,
                                           input_w,
                                           filter_h,
                                           filter_w,
                                           stride_.h,
                                           stride_.w,
                                           output_dim.h,
                                           output_dim.w,
                                           depth_multiplier_,
                                           input_offset,
                                           weight_offset,
                                           output_offset,
                                           output_multiplier,
                                           output_shift,
                                           act_min,
                                           act_max);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv setKernel failure\n");
        }
    }

    state = runtime_->enqueueKernel(depthwisekernel_.get(), (cl_uint)3, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "enqueueKernel failure\n");

    if (input_dim.c * depth_multiplier_ < output_dim.c) {
        global[0] = output_dim.n;
        global[1] = output_dim.c - (input_dim.c * depth_multiplier_);
        global[2] = output_dim.h * output_dim.w;
        local[2] = findMaxFactor(global[2], 128);
        local[1] = findMaxFactor(global[1], 128 / local[2]);
        local[0] = 1;
        state = runtime_->setKernelArg(kernel_unequal_.get(),
                                       bias_data,
                                       *output_data,
                                       input_dim.c * depth_multiplier_,
                                       output_offset,
                                       output_multiplier,
                                       output_shift,
                                       act_min,
                                       act_max);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv_unequal setKernel failure\n");
        state = runtime_->enqueueKernel(kernel_unequal_.get(), (cl_uint)3, global, local);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "enqueueKernel failure\n");
    }

    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
