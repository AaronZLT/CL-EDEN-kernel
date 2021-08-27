#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLDirectConvolution.hpp"

namespace enn {
namespace ud {
namespace gpu {

CLDirectConvolution::CLDirectConvolution(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) {
    ENN_DBG_PRINT("CLDirectConvolution is created");
    runtime_ = runtime;
    precision_ = precision;
    direct_ = nullptr;
    computed_top_channel_numbers_ = DIRECT_TOP_CHANNEL_8;
    computed_top_width_numbers_ = DIRECT_TOP_WIDTH_8;
    computed_top_height_numbers_ = 1;
    input_dim_ = {0, 0, 0, 0};
    weight_dim_ = {0, 0, 0, 0};
    padding_ = {0, 0, 0, 0};
    stride_ = {0, 0};
    dilation_ = {0, 0};
    weights_as_input_ = false;
    androidNN_ = false;
}

Status CLDirectConvolution::initialize(const Dim4 &input_dim,
                                       const Dim4 &output_dim,
                                       const Dim4 &weight_dim,
                                       const Pad4 &padding,
                                       const Dim2 &stride,
                                       const Dim2 &dilation,
                                       const std::shared_ptr<ITensor> weight,
                                       const std::shared_ptr<ITensor> bias,
                                       const ActivationInfo &activate_info,
                                       const bool &weights_as_input,
                                       const bool &androidNN,
                                       const bool &isNCHW) {
    ENN_DBG_PRINT("CLDirectConvolution::initialize() is called");
    weight_dim_ = weight_dim;
    activation_info_ = activate_info;
    padding_ = padding;
    stride_ = stride;
    dilation_ = dilation;
    bias_ = std::static_pointer_cast<CLTensor>(bias);
    weight_ = std::static_pointer_cast<CLTensor>(weight);
    weights_as_input_ = weights_as_input;
    androidNN_ = androidNN;
    isNCHW_ = isNCHW;

    auto weight_tensor = std::static_pointer_cast<CLTensor>(weight);
    weight_ = weight_tensor;

    Status state = Status::FAILURE;
    if (androidNN_ || !isNCHW_) {
        Dim4 weight_dim_nchw = {weight_dim.n, weight_dim.w, weight_dim.c, weight_dim.h};
        weight_nchw_ = std::make_shared<CLTensor>(runtime_,
                                                  precision_,
                                                  weight_tensor->getDataType(),
                                                  weight_dim_nchw,
                                                  weight_tensor->getDataOrder(),
                                                  weight_tensor->getScale(),
                                                  weight_tensor->getZeroPoint());
        weight_dim_ = weight_dim_nchw;
        if (!weights_as_input_) {
            state = weight_tensor->convertToNCHW(weight_nchw_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "convertToNCHW failure\n");
        }
    }

    if (!weights_as_input && (dilation.h > 1 || dilation.w > 1)) {
        dilationWeight((androidNN_ || !isNCHW_) ? weight_nchw_ : weight_);
    }

    weight_dim_.h = dilation.h * (weight_dim_.h - 1) + 1;
    weight_dim_.w = dilation.w * (weight_dim_.w - 1) + 1;

    computed_top_width_numbers_ = DIRECT_TOP_WIDTH_8;
    computed_top_height_numbers_ = 1;
    is_4x8_7x7_or_9x9_ = false;

    if (precision_ == PrecisionType::FP16 &&
        ((weight_dim_.h == 7 && weight_dim_.w == 7) || (weight_dim_.h == 9 && weight_dim_.w == 9))) {
        is_4x8_7x7_or_9x9_ = true;
    }

    std::string direct_kernel_name;
    if (activation_info_.isEnabled() && activate_info.activation() == ActivationInfo::ActivationType::RELU) {
        direct_kernel_name = "RELU";
    } else if (activation_info_.isEnabled() && activate_info.activation() == ActivationInfo::ActivationType::RELU6) {
        direct_kernel_name = "RELU6";
    }

    if (precision_ == PrecisionType::FP32 || weight_dim_.n < 4 || (weight_dim_.h > 3 && weight_dim_.w == weight_dim_.h)) {
        // FP32 computes 4 channels by default
        computed_top_channel_numbers_ = DIRECT_TOP_CHANNEL_4;
        // default impl
        if ((weight_dim_.h == 5 && weight_dim_.w == 5) || (weight_dim_.h == 7 && weight_dim_.w == 7) ||
            (weight_dim_.h == 9 && weight_dim_.w == 9)) {
            direct_kernel_name +=
                "direct" + std::to_string(weight_dim_.h) + "x" + std::to_string(weight_dim_.w) + "_" + "4x8";
        } else if (weight_dim_.h == 3 && weight_dim_.w == 3) {
            if (output_dim.w % DIRECT_TOP_WIDTH_8 == 0 && output_dim.w / DIRECT_TOP_WIDTH_8 >= 16) {
                computed_top_width_numbers_ = DIRECT_TOP_WIDTH_8;
                direct_kernel_name +=
                    "direct" + std::to_string(weight_dim_.h) + "x" + std::to_string(weight_dim_.w) + "_" + "4x8";
            } else {
                computed_top_width_numbers_ = DIRECT_TOP_WIDTH_4;
                direct_kernel_name +=
                    "direct" + std::to_string(weight_dim_.h) + "x" + std::to_string(weight_dim_.w) + "_" + "4x4";
            }
        } else if (weight_dim_.h == 1 && weight_dim_.w == 1) {
            computed_top_channel_numbers_ = DIRECT_TOP_CHANNEL_8;
            computed_top_width_numbers_ = DIRECT_TOP_WIDTH_4;
            direct_kernel_name +=
                "direct" + std::to_string(weight_dim_.h) + "x" + std::to_string(weight_dim_.w) + "_" + "8x4";
        } else {
            ERROR_PRINT("Non supported kernel size %dx%d in direct convolution\n", weight_dim_.h, weight_dim_.w);
            return Status::FAILURE;
        }
    } else {
        // FP16 computes 8 channels by default
        computed_top_channel_numbers_ = DIRECT_TOP_CHANNEL_8;
        if ((weight_dim_.h == 5 && weight_dim_.w == 5) || (weight_dim_.h == 7 && weight_dim_.w == 7) ||
            (weight_dim_.h == 9 && weight_dim_.w == 9) || (weight_dim_.h == 3 && weight_dim_.w == 3)) {
            direct_kernel_name +=
                "direct" + std::to_string(weight_dim_.h) + "x" + std::to_string(weight_dim_.w) + "_" + "8x8";
        } else {
            ERROR_PRINT("Non supported kernel size %dx%d in direct convolution\n", weight_dim_.h, weight_dim_.w);
            return Status::FAILURE;
        }
    }
    state = runtime_->setKernel(&direct_, direct_kernel_name.c_str(), precision_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel %s failed!\n", direct_kernel_name.c_str());

    if (!weights_as_input) {
        weightConvert();
    }

    std::string direct_merge_kernel_name;
    if (is_4x8_7x7_or_9x9_) {
        if (activation_info_.isEnabled() && activate_info.activation() == ActivationInfo::ActivationType::RELU) {
            direct_merge_kernel_name = "RELU";
        } else if (activation_info_.isEnabled() && activate_info.activation() == ActivationInfo::ActivationType::RELU6) {
            direct_merge_kernel_name = "RELU6";
        }
        splite_num_ = 2;
        Dim4 alinged_output_dim = {static_cast<uint32_t>(output_dim.n),
                                   static_cast<uint32_t>(output_dim.c),
                                   static_cast<uint32_t>(output_dim.h),
                                   static_cast<uint32_t>(output_dim.w * splite_num_)};
        aligned_output_tm_ = std::make_shared<CLTensor>(runtime_, precision_, weight_->getDataType(), alinged_output_dim);

        direct_merge_kernel_name +=
            "direct" + std::to_string(weight_dim_.h) + "x" + std::to_string(weight_dim_.w) + "_" + "4x8_merge";
        state = runtime_->setKernel(&direct_merge_, direct_merge_kernel_name.c_str(), precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel %s failed!\n", direct_merge_kernel_name.c_str());
    }
    return Status::SUCCESS;
}

Status CLDirectConvolution::weightConvert() {
    Status state = Status::FAILURE;
    int aligned_wight_w = weight_dim_.w * computed_top_channel_numbers_;
    int aligned_weight_h = weight_dim_.h;
    int aligned_weight_c = weight_dim_.c;
    int aligned_weight_n = ceil(static_cast<double>(weight_dim_.n) / computed_top_channel_numbers_);
    Dim4 alinged_weight_dim = {static_cast<uint32_t>(aligned_weight_n),
                               static_cast<uint32_t>(aligned_weight_c),
                               static_cast<uint32_t>(aligned_weight_h),
                               static_cast<uint32_t>(aligned_wight_w)};
    aligned_weight_ = std::make_shared<CLTensor>(runtime_, precision_, weight_->getDataType(), alinged_weight_dim);

    std::shared_ptr<struct _cl_kernel> align_weight_kernel;
    state = runtime_->setKernel(&align_weight_kernel, "align_weight_direct", precision_);
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel align_weight_direct failed\n");

    auto weight_data = (dilation_.h > 1 || dilation_.w > 1)
                           ? dilation_filter_->getDataPtr()
                           : (androidNN_ || !isNCHW_) ? weight_nchw_->getDataPtr() : weight_->getDataPtr();

    state = runtime_->setKernelArg(align_weight_kernel.get(),
                                   weight_data,
                                   aligned_weight_->getDataPtr(),
                                   weight_dim_.n,
                                   weight_dim_.c,
                                   weight_dim_.h,
                                   weight_dim_.w,
                                   aligned_weight_n,
                                   aligned_weight_c,
                                   aligned_weight_h,
                                   aligned_wight_w,
                                   computed_top_channel_numbers_);
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernelArg align_weight_direct failed\n");

    size_t local[3] = {1, 1, 1};
    size_t global[3] = {static_cast<size_t>(aligned_wight_w),
                        static_cast<size_t>(aligned_weight_h),
                        static_cast<size_t>(aligned_weight_c * aligned_weight_n)};
    state = runtime_->enqueueKernel(align_weight_kernel.get(), (cl_uint)3, global, local);
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "enqueue align_weight_direct failed\n");
    return Status::SUCCESS;
}

Status CLDirectConvolution::execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) {
    ENN_DBG_PRINT("CLDirectConvolution::execute is called\n");
    Status status = Status::SUCCESS;
    if (weights_as_input_) {
        if (androidNN_ || !isNCHW_) {
            status = weight_->convertToNCHW(weight_nchw_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "convertToNCHW failure\n");
        }
        if ((dilation_.h > 1 || dilation_.w > 1)) {
            dilationWeight((androidNN_ || !isNCHW_) ? weight_nchw_ : weight_);
        }
        weightConvert();
    }

    auto input_ = std::static_pointer_cast<CLTensor>(input);
    auto output_ = std::static_pointer_cast<CLTensor>(output);

    Dim4 output_dim = output_->getDim();
    Dim4 input_dim = input_->getDim();

    if (weight_dim_.h == 1 && weight_dim_.w == 1) {
        size_t local[2] = {16, 1};
        size_t global[2] = {0, 0};

        global[0] = alignTo(ceil(static_cast<double>(output_dim.h * output_dim.w) / computed_top_width_numbers_), local[0]);
        global[1] =
            alignTo(ceil(static_cast<double>(output_dim.c) / computed_top_channel_numbers_) * output_dim.n, local[1]);

        status = runtime_->setKernelArg(direct_.get(),
                                        input_->getDataPtr(),
                                        aligned_weight_->getDataPtr(),
                                        bias_->getDataPtr(),
                                        output_->getDataPtr(),
                                        padding_.t,
                                        padding_.r,
                                        padding_.b,
                                        padding_.l,
                                        output_dim.n,
                                        output_dim.c,
                                        output_dim.h,
                                        output_dim.w,
                                        input_dim.n,
                                        input_dim.c,
                                        input_dim.h,
                                        input_dim.w);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == status, "setKernelArg direct5x5_ failed\n");

        status = runtime_->enqueueKernel(direct_.get(), (cl_uint)2, global, local);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == status, "enqueue direct5x5_ failed\n");
    } else {
        size_t local[3] = {16, 1, 1};
        size_t global[3] = {0, 0, 0};

        if (output_dim.w / computed_top_width_numbers_ < 16) {
            local[0] = 4;
            local[1] = 4;
        }
        global[0] = alignTo(ceil(static_cast<double>(output_dim.w) / computed_top_width_numbers_), local[0]);
        global[1] = alignTo(ceil(static_cast<double>(output_dim.h) / computed_top_height_numbers_), local[1]);
        global[2] =
            alignTo(ceil(static_cast<double>(output_dim.c) / computed_top_channel_numbers_) * output_dim.n, local[2]);

        if (is_4x8_7x7_or_9x9_) {
            global[1] = global[1] * splite_num_;
            local[1] = 4 * splite_num_;
            status = runtime_->setKernelArg(direct_.get(),
                                            input_->getDataPtr(),
                                            aligned_weight_->getDataPtr(),
                                            bias_->getDataPtr(),
                                            aligned_output_tm_->getDataPtr(),
                                            padding_.t,
                                            padding_.r,
                                            padding_.b,
                                            padding_.l,
                                            output_dim.n,
                                            output_dim.c,
                                            output_dim.h,
                                            output_dim.w,
                                            input_dim.n,
                                            input_dim.c,
                                            input_dim.h,
                                            input_dim.w,
                                            splite_num_);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == status, "setKernelArg direct_splite_ failed\n");
            status = runtime_->setKernelArg(direct_merge_.get(),
                                            bias_->getDataPtr(),
                                            aligned_output_tm_->getDataPtr(),
                                            output_->getDataPtr(),
                                            padding_.t,
                                            padding_.r,
                                            padding_.b,
                                            padding_.l,
                                            output_dim.n,
                                            output_dim.c,
                                            output_dim.h,
                                            output_dim.w,
                                            input_dim.n,
                                            input_dim.c,
                                            input_dim.h,
                                            input_dim.w,
                                            splite_num_);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == status, "setKernelArg direct_merge failed\n");
        } else {
            status = runtime_->setKernelArg(direct_.get(),
                                            input_->getDataPtr(),
                                            aligned_weight_->getDataPtr(),
                                            bias_->getDataPtr(),
                                            output_->getDataPtr(),
                                            padding_.t,
                                            padding_.r,
                                            padding_.b,
                                            padding_.l,
                                            output_dim.n,
                                            output_dim.c,
                                            output_dim.h,
                                            output_dim.w,
                                            input_dim.n,
                                            input_dim.c,
                                            input_dim.h,
                                            input_dim.w);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == status, "setKernelArg direct_ failed\n");
        }

        CHECK_EXPR_NO_RETURN(Status::SUCCESS == status, "setKernelArg direct5x5_ failed\n");
        status = runtime_->enqueueKernel(direct_.get(), (cl_uint)3, global, local);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == status, "enqueue direct5x5_ failed\n");

        if (is_4x8_7x7_or_9x9_) {
            global[0] = alignTo(ceil(static_cast<double>(output_dim.w) / computed_top_width_numbers_), local[0]);
            global[1] = alignTo(ceil(static_cast<double>(output_dim.h) / computed_top_height_numbers_), local[1]);
            global[2] =
                alignTo(ceil(static_cast<double>(output_dim.c) / computed_top_channel_numbers_) * output_dim.n, local[2]);
            status = runtime_->enqueueKernel(direct_merge_.get(), (cl_uint)3, global, local);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == status, "enqueue direct_splite_merge failed\n");
        }
    }
    return Status::SUCCESS;
}

Status CLDirectConvolution::dilationWeight(const std::shared_ptr<CLTensor> weight_tensor) {
    uint32_t batch_weight = weight_tensor->getDim().n;
    uint32_t channel_weight = weight_tensor->getDim().c;
    uint32_t kernel_height = weight_tensor->getDim().h;
    uint32_t kernel_width = weight_tensor->getDim().w;

    uint32_t extent_height = dilation_.h * (kernel_height - 1) + 1;
    uint32_t extent_width = dilation_.w * (kernel_width - 1) + 1;
    Dim4 dim_extent = {batch_weight, channel_weight, extent_height, extent_width};
    dilation_filter_ = std::make_shared<CLTensor>(runtime_,
                                                  precision_,
                                                  weight_tensor->getDataType(),
                                                  dim_extent,
                                                  weight_tensor->getDataOrder(),
                                                  weight_tensor->getScale(),
                                                  weight_tensor->getZeroPoint());

    size_t global[3];
    global[0] = batch_weight * channel_weight;
    global[1] = kernel_height;
    global[2] = kernel_width;
    std::shared_ptr<struct _cl_kernel> kernel_dilation;
    Status state = runtime_->setKernel(&kernel_dilation, "dilation", precision_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
    state = runtime_->setKernelArg(kernel_dilation.get(),
                                   weight_tensor->getDataPtr(),
                                   dilation_filter_->getDataPtr(),
                                   extent_height,
                                   extent_width,
                                   dilation_.h,
                                   dilation_.w);

    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");
    state = runtime_->enqueueKernel(kernel_dilation.get(), (cl_uint)3, global, NULL);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "enqueueKernel failure\n");

    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
