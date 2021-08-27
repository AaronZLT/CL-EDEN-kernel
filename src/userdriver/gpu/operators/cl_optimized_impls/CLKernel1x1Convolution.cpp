#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLKernel1x1Convolution.hpp"

namespace enn {
namespace ud {
namespace gpu {

CLKernel1x1Convolution::CLKernel1x1Convolution(const std::shared_ptr<CLRuntime> runtime,
                                               const PrecisionType &precision,
                                               const Dim4 &input_dim,
                                               const Dim4 &output_dim) :
    runtime_(runtime),
    precision_(precision), input_dim_(input_dim), output_dim_(output_dim) {
    weights_as_input_ = false;
    androidNN_ = false;
    src_total_count_ = 0;
    src_unit_count_ = 0;
    dst_align_count_ = 0;
    weight_data_ = nullptr;
    converted_filter_data_ = nullptr;
    isNCHW_ = true;
}

Status CLKernel1x1Convolution::initialize(const Pad4 &padding,
                                          const Dim2 &stride,
                                          const uint32_t &group_size,
                                          const uint32_t &axis,
                                          const Dim2 &dilation,
                                          const std::shared_ptr<ITensor> weight,
                                          const std::shared_ptr<ITensor> bias,
                                          const ActivationInfo &activate_info,
                                          const bool &weights_as_input,
                                          const bool &androidNN,
                                          const bool &isNCHW) {
    auto weight_tensor = std::static_pointer_cast<CLTensor>(weight);
    auto bias_tensor = std::static_pointer_cast<CLTensor>(bias);
    conv_descriptor_.filter_ = weight_tensor;
    conv_descriptor_.bias_ = bias_tensor;

    weights_as_input_ = weights_as_input;
    androidNN_ = androidNN;
    isNCHW_ = isNCHW;

    Dim4 weight_size = weight_tensor->getDim();
    if (androidNN_ || !isNCHW_) {
        weight_size = convertDimToNCHW(weight_size);
        weight_nchw_ = std::make_shared<CLTensor>(runtime_,
                                                  precision_,
                                                  weight_tensor->getDataType(),
                                                  weight_size,
                                                  weight_tensor->getDataOrder(),
                                                  weight_tensor->getScale(),
                                                  weight_tensor->getZeroPoint());
        if (!weights_as_input_) {
            Status state = weight_tensor->convertToNCHW(weight_nchw_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "convertToNCHW failure\n");
        }
    }

    conv_descriptor_.num_output_ = weight_size.n;
    conv_descriptor_.kernel_channel_ = weight_size.c;
    conv_descriptor_.pad_right_ = padding.r;
    conv_descriptor_.pad_left_ = padding.l;
    conv_descriptor_.pad_top_ = padding.t;
    conv_descriptor_.pad_bottom_ = padding.b;
    conv_descriptor_.kernel_height_ = weight_size.h;
    conv_descriptor_.kernel_width_ = weight_size.w;
    conv_descriptor_.stride_height_ = stride.h;
    conv_descriptor_.stride_width_ = stride.w;
    conv_descriptor_.group_ = group_size;
    conv_descriptor_.axis_ = axis;
    conv_descriptor_.dilation_ = dilation;
    activation_info_ = activate_info;

    if (padding.r == 0 && padding.l == 0 && padding.t == 0 && padding.b == 0) {
        is_need_pad_ = false;
    } else {
        is_need_pad_ = true;

        Dim4 pad_output_dim = {input_dim_.n,
                               input_dim_.c,
                               input_dim_.h + conv_descriptor_.pad_top_ + conv_descriptor_.pad_bottom_,
                               input_dim_.w + conv_descriptor_.pad_left_ + conv_descriptor_.pad_right_};
        pad_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, pad_output_dim);
    }

    // align weight and conv11
    uint32_t align_weight_width =
        alignTo(conv_descriptor_.kernel_height_ * conv_descriptor_.kernel_width_ * input_dim_.c, 16);
    uint32_t align_weight_height = alignTo(output_dim_.c, 16);
    uint32_t weight_count = align_weight_height * align_weight_width;

    src_total_count_ = conv_descriptor_.num_output_ * conv_descriptor_.kernel_width_ * conv_descriptor_.kernel_height_ *
                       input_dim_.c / conv_descriptor_.group_;
    src_unit_count_ = src_total_count_ / output_dim_.c;
    dst_align_count_ = alignTo(src_unit_count_, 16);

    Dim4 converted_filter_dim = {weight_count, 1, 1, 1};
    converted_filter_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, converted_filter_dim);

    Status state = runtime_->setKernel(&copybuffer_kernel_, "copybuffer", precision_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");

    weight_data_ = (androidNN_ || !isNCHW_) ? weight_nchw_->getDataPtr() : conv_descriptor_.filter_->getDataPtr();
    converted_filter_data_ = converted_filter_->getDataPtr();

    if (weights_as_input_ == false) {
        state = memAlign(weight_data_, converted_filter_data_, src_total_count_, src_unit_count_, dst_align_count_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "mem align failure\n");
    }

    if (is_need_pad_) {
        state = runtime_->setKernel(&pad_kernel_, "pad", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
    }

    if (activation_info_.isEnabled()) {
        if (stride.h == 2 && stride.w == 2) {
            if (activation_info_.activation() == ActivationInfo::ActivationType::RELU) {
                state = runtime_->setKernel(&conv11_kernel_, "RELUconv11_stride2", precision_);
                CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
            } else if (activation_info_.activation() == ActivationInfo::ActivationType::RELU6 &&
                       precision_ == PrecisionType::FP16) {
                state = runtime_->setKernel(&conv11_kernel_, "RELU6conv11_stride2", precision_);
                CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
            } else {
                state = runtime_->setKernel(&conv11_kernel_, "conv11_stride2", precision_);
                CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
            }
        } else {
            if (activation_info_.activation() == ActivationInfo::ActivationType::RELU) {
                state = runtime_->setKernel(&conv11_kernel_, "RELUconv11", precision_);
                CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
            } else if (activation_info_.activation() == ActivationInfo::ActivationType::RELU6 &&
                       precision_ == PrecisionType::FP16) {
                state = runtime_->setKernel(&conv11_kernel_, "RELU6conv11", precision_);
                CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
            } else {
                state = runtime_->setKernel(&conv11_kernel_, "conv11", precision_);
                CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
            }
        }
    } else {
        if (stride.h == 2 && stride.w == 2) {
            state = runtime_->setKernel(&conv11_kernel_, "conv11_stride2", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
        } else {
            state = runtime_->setKernel(&conv11_kernel_, "conv11", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
        }
    }

    return state;
}

Status CLKernel1x1Convolution::execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) {
    ENN_DBG_PRINT("CLKernel1x1Convolution is execute");

    if (weights_as_input_ == true) {
        Status state = Status::FAILURE;
        if (androidNN_ || !isNCHW_) {
            state = conv_descriptor_.filter_->convertToNCHW(weight_nchw_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "convertToNCHW failure\n");
        }
        // if weight as input, weight_data_ is nullptr before execution, so getDataPtr should be
        // invoked again
        weight_data_ = (androidNN_ || !isNCHW_) ? weight_nchw_->getDataPtr() : conv_descriptor_.filter_->getDataPtr();
        converted_filter_data_ = converted_filter_->getDataPtr();
        state = memAlign(conv_descriptor_.filter_->getDataPtr(),
                         converted_filter_data_,
                         src_total_count_,
                         src_unit_count_,
                         dst_align_count_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "mem align failure\n");
    }

    auto input_tensor = std::static_pointer_cast<CLTensor>(input);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output);
    return convKernel1x1GPU(input_tensor, output_tensor);
}

Status CLKernel1x1Convolution::convKernel1x1GPU(const std::shared_ptr<CLTensor> input, std::shared_ptr<CLTensor> output) {
    Status state;
    auto input_data = input->getDataPtr();
    auto output_data = output->getDataPtr();

    int input_width = input_dim_.w;
    int input_height = input_dim_.h;

    if (is_need_pad_) {
        auto pad_data = pad_->getDataPtr();
        size_t global[3] = {0, 0, 0};
        size_t local[3] = {1, 1, 32};
        global[0] = input_dim_.n;
        global[1] = input_dim_.c;
        global[2] = alignTo(pad_->getDim().h * pad_->getDim().w, local[2]);
        Status state = runtime_->setKernelArg(pad_kernel_.get(),
                                              input_data,
                                              pad_data,
                                              conv_descriptor_.pad_top_,
                                              conv_descriptor_.pad_left_,
                                              input_width,
                                              input_height);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");

        state = runtime_->enqueueKernel(pad_kernel_.get(), (cl_uint)3, global, local);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
    }

    size_t globalConv[3] = {0, 0, 0}, localConv[3] = {0, 0, 0};
    globalConv[0] = input_dim_.n;
    globalConv[1] = ceil(output_dim_.c / 2.0);
    int gsize2 = (precision_ == PrecisionType::FP32) ? ceil(output_dim_.h * output_dim_.w / 4.0)
                                                     : ceil(output_dim_.h * output_dim_.w / 8.0);
    int lsize2 = 16;

    if (conv_descriptor_.stride_height_ == 2 && conv_descriptor_.stride_width_ == 2) {
        globalConv[2] = alignTo(output_dim_.h * ceil(output_dim_.w / 4.0), lsize2);
    } else {
        globalConv[2] = alignTo(gsize2, lsize2);
    }

    localConv[0] = 1;
    localConv[1] = (globalConv[1] % 2 == 0) ? 2 : 1;
    localConv[2] = lsize2;
    cl_mem bias_data = conv_descriptor_.bias_->getDataPtr();
    int aligned_input_channel = alignTo(input_dim_.c, 16);
    if (is_need_pad_) {
        auto pad_data = pad_->getDataPtr();
        if (conv_descriptor_.stride_height_ == 2 && conv_descriptor_.stride_width_ == 2) {
            state = runtime_->setKernelArg(conv11_kernel_.get(),
                                           pad_data,
                                           converted_filter_->getDataPtr(),
                                           bias_data,
                                           output_data,
                                           input_dim_.c,
                                           output_dim_.c,
                                           input_dim_.w,
                                           input_dim_.h,
                                           output_dim_.w,
                                           output_dim_.h,
                                           aligned_input_channel);
        } else {
            state = runtime_->setKernelArg(conv11_kernel_.get(),
                                           pad_data,
                                           converted_filter_->getDataPtr(),
                                           bias_data,
                                           output_data,
                                           input_dim_.c,
                                           output_dim_.c,
                                           output_dim_.w,
                                           output_dim_.h,
                                           aligned_input_channel);
        }

        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");

    } else {
        if (conv_descriptor_.stride_height_ == 2 && conv_descriptor_.stride_width_ == 2) {
            state = runtime_->setKernelArg(conv11_kernel_.get(),
                                           input_data,
                                           converted_filter_->getDataPtr(),
                                           bias_data,
                                           output_data,
                                           input_dim_.c,
                                           output_dim_.c,
                                           input_dim_.w,
                                           input_dim_.h,
                                           output_dim_.w,
                                           output_dim_.h,
                                           aligned_input_channel);
        } else {
            state = runtime_->setKernelArg(conv11_kernel_.get(),
                                           input_data,
                                           converted_filter_->getDataPtr(),
                                           bias_data,
                                           output_data,
                                           input_dim_.c,
                                           output_dim_.c,
                                           output_dim_.w,
                                           output_dim_.h,
                                           aligned_input_channel);
        }

        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");
    }

    state = runtime_->enqueueKernel(conv11_kernel_.get(), (cl_uint)3, globalConv, localConv);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "enqueue kernel failure\n");

    return Status::SUCCESS;
}

Status CLKernel1x1Convolution::memAlign(const cl_mem &src,
                                        cl_mem &dst,
                                        uint32_t src_total_count,
                                        uint32_t src_unit_count,
                                        uint32_t dst_align_count) {
    uint32_t lines = src_total_count / src_unit_count;

    size_t global = static_cast<size_t>(lines);
    Status state = runtime_->setKernelArg(copybuffer_kernel_.get(), src, dst, src_unit_count, dst_align_count);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");

    state = runtime_->enqueueKernel(copybuffer_kernel_.get(), (cl_uint)1, &global, NULL);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
