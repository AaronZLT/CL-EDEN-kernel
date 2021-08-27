/*
 * Copyright (C) 2020 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

/**
 * @file    CLPowerVRConvolution.cpp
 * @brief
 * @details
 * @version
 */

#include "userdriver/gpu/operators/cl_optimized_impls/CLPowerVRConvolution.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/common/operator_interfaces/common/Debug.hpp"

namespace enn {
namespace ud {
namespace gpu {

CLPowerVRConvolution::CLPowerVRConvolution(const std::shared_ptr<CLRuntime> runtime,
                                           const PrecisionType &precision,
                                           const Dim4 &input_dim,
                                           const Dim4 &output_dim) :
    runtime_(runtime),
    precision_(precision), input_dim_(input_dim), output_dim_(output_dim) {
    ENN_DBG_PRINT("CLPowerVRConvolution is created");
    convert_out_dim_ = {0, 0, 0, 0};
    stride_ = {0, 0};
    kernel_ = {0, 0};
    mergeadd_ = false;
    kernel_max_work_group_size_ = 0;
    weights_as_input_ = false;
    block_size_x_ = 0;
    block_size_y_ = 0;
    block_size_z_ = 0;
    src_x_ = 0;
    src_y_ = 0;
    src_z_ = 0;
    src_w_ = 0;
    dst_x_ = 0;
    dst_y_ = 0;
    dst_z_ = 0;
    dst_w_ = 0;
}

Status CLPowerVRConvolution::initialize(const Pad4 &padding,
                                        const Dim2 &stride,
                                        const uint32_t &group_size,
                                        const uint32_t &axis,
                                        const Dim2 &dilation,
                                        const std::shared_ptr<ITensor> weight,
                                        const std::shared_ptr<ITensor> bias,
                                        const ActivationInfo &activate_info,
                                        bool weights_as_input,
                                        bool mergeadd) {
    ENN_DBG_PRINT("CLPowerVRConvolution::initialize() is called");

    Status state;
    auto weight_tensor = std::static_pointer_cast<CLTensor>(weight);
    auto bias_tensor = std::static_pointer_cast<CLTensor>(bias);
    const Dim4 &weight_size = weight_tensor->getDim();

    mergeadd_ = mergeadd;
    stride_ = stride;
    kernel_ = {weight_size.h, weight_size.w};
    conv_descriptor_.num_output_ = weight_size.n;
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
    conv_descriptor_.bias_ = bias_tensor;
    activation_info_ = activate_info;
    weights_as_input_ = weights_as_input;

    //  The Dim order of weight is NCHW, but the data order of weight is NHWC.
    //  The data order of weight ned to convert to NCHW
    auto filter_tensor = std::static_pointer_cast<CLTensor>(weight);
    filter_nchw_ = std::make_shared<CLTensor>(runtime_,
                                              precision_,
                                              weight->getDataType(),
                                              weight->getDim(),
                                              weight->getDataOrder(),
                                              weight->getScale(),
                                              weight->getZeroPoint());

    DataType mdata_type = weight->getDataType();
    if (weight->getDataType() == DataType::FLOAT && precision_ == PrecisionType::FP16) {
        mdata_type = DataType::HALF;
    } else if (weight->getDataType() == DataType::HALF && precision_ == PrecisionType::FP32) {
        mdata_type = DataType::FLOAT;
    }
    runtime_->NHWC2NCHW(filter_tensor->getDataPtr(),
                        filter_nchw_->getDataPtr(),
                        filter_nchw_->getDim(),
                        mdata_type,
                        PrecisionChangeMode::OTHER);
    conv_descriptor_.filter_ = filter_nchw_;

    generateKernelName(activate_info);
    state = runtime_->setKernel(&powervr_kernel_, _kernel_name_, precision_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");

    state =
        runtime_->GetKernelMaxWorkGroupSize(powervr_kernel_.get(), runtime_->getDeviceID(), &kernel_max_work_group_size_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "GetKernelMaxWorkGroupSize failure\n");

    state = alignWeight();
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "alignWeight failure\n");

    work_group_size_[0] = 4;
    work_group_size_[1] = 4;
    work_group_size_[2] = 1;
    GetGridSize(grid_size_);
    GetBestWorkGroupConv(grid_size_, work_group_size_);
    GetGridSize(grid_size_);

    src_x_ = input_dim_.w * input_dim_.n;
    src_y_ = input_dim_.h;
    src_z_ = IntegralDivideRoundUp(input_dim_.c, 4);
    src_w_ = input_dim_.n;
    dst_x_ = output_dim_.w * output_dim_.n;
    dst_y_ = output_dim_.h;
    dst_z_ = IntegralDivideRoundUp(output_dim_.c, 4);
    dst_w_ = output_dim_.n;
    return Status::SUCCESS;
}

Status CLPowerVRConvolution::execute(const std::vector<std::shared_ptr<ITensor>> input, std::shared_ptr<ITensor> output) {
    ENN_DBG_PRINT("CLPowerVRConvolution is execute");
    auto input_tensor = std::static_pointer_cast<CLTensor>(input[0]);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output);

    cl_mem input_add_data = nullptr;
    if (mergeadd_) {
        input_add_data = std::static_pointer_cast<CLTensor>(input[1])->getDataPtr();
    }

    // execute->local&global
    size_t local[3] = {static_cast<size_t>(work_group_size_[0]),
                       static_cast<size_t>(work_group_size_[1]),
                       static_cast<size_t>(work_group_size_[2])};
    size_t global[3];
    global[0] = static_cast<size_t>(AlignByN(grid_size_[0], work_group_size_[0]));
    global[1] = static_cast<size_t>(AlignByN(grid_size_[1], work_group_size_[1]));
    global[2] = static_cast<size_t>(AlignByN(grid_size_[2], work_group_size_[2]));

    // execute->enqueue
    if (is1x1_) {
        if (mergeadd_) {
            Status state = runtime_->setKernelArg(powervr_kernel_.get(),
                                                  input_tensor->getDataPtr(),
                                                  input_add_data,
                                                  converted_filter_->getDataPtr(),
                                                  conv_descriptor_.bias_->getDataPtr(),
                                                  output_tensor->getDataPtr(),
                                                  src_x_,
                                                  src_y_,
                                                  src_z_,
                                                  src_w_,
                                                  dst_x_,
                                                  dst_y_,
                                                  dst_z_,
                                                  dst_w_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "set kernel failure\n");
        } else {
            Status state = runtime_->setKernelArg(powervr_kernel_.get(),
                                                  input_tensor->getDataPtr(),
                                                  converted_filter_->getDataPtr(),
                                                  conv_descriptor_.bias_->getDataPtr(),
                                                  output_tensor->getDataPtr(),
                                                  src_x_,
                                                  src_y_,
                                                  src_z_,
                                                  src_w_,
                                                  dst_x_,
                                                  dst_y_,
                                                  dst_z_,
                                                  dst_w_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "set kernel failure\n");
        }
    } else {
        if (mergeadd_) {
            Status state = runtime_->setKernelArg(powervr_kernel_.get(),
                                                  input_tensor->getDataPtr(),
                                                  input_add_data,
                                                  converted_filter_->getDataPtr(),
                                                  conv_descriptor_.bias_->getDataPtr(),
                                                  output_tensor->getDataPtr(),
                                                  src_x_,
                                                  src_y_,
                                                  src_z_,
                                                  src_w_,
                                                  dst_x_,
                                                  dst_y_,
                                                  dst_z_,
                                                  dst_w_,
                                                  kernel_.w,
                                                  kernel_.h,
                                                  conv_descriptor_.dilation_.w * src_w_,
                                                  conv_descriptor_.dilation_.h,
                                                  stride_.w,
                                                  stride_.h,
                                                  (-conv_descriptor_.pad_left_) * src_w_,
                                                  (-conv_descriptor_.pad_top_));
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "set kernel failure\n");
        } else {
            Status state = runtime_->setKernelArg(powervr_kernel_.get(),
                                                  input_tensor->getDataPtr(),
                                                  converted_filter_->getDataPtr(),
                                                  conv_descriptor_.bias_->getDataPtr(),
                                                  output_tensor->getDataPtr(),
                                                  src_x_,
                                                  src_y_,
                                                  src_z_,
                                                  src_w_,
                                                  dst_x_,
                                                  dst_y_,
                                                  dst_z_,
                                                  dst_w_,
                                                  kernel_.w,
                                                  kernel_.h,
                                                  conv_descriptor_.dilation_.w * src_w_,
                                                  conv_descriptor_.dilation_.h,
                                                  stride_.w,
                                                  stride_.h,
                                                  (-conv_descriptor_.pad_left_) * src_w_,
                                                  (-conv_descriptor_.pad_top_));
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "set kernel failure\n");
        }
    }
    Status status = runtime_->enqueueKernel(powervr_kernel_.get(), 3, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "execute kernel failure\n");
    return Status::SUCCESS;
}

Status CLPowerVRConvolution::alignWeight() {
    ENN_DBG_PRINT("CLPowerVRConvolution is in alignWeight");
    uint32_t div = 4;
    auto weight_dim = conv_descriptor_.filter_->getDim();
    const int dst_slices = IntegralDivideRoundUp(weight_dim.n, div);
    const int src_slices = IntegralDivideRoundUp(weight_dim.c, div);
    const int dst_groups = IntegralDivideRoundUp(dst_slices, block_size_z_);
    const int dst_depth_aligned = AlignByN(dst_slices, block_size_z_);

    const int kernel_x = weight_dim.w;
    const int kernel_y = weight_dim.h;
    convert_out_dim_.n = dst_depth_aligned * 4;
    convert_out_dim_.c = src_slices * 4;
    convert_out_dim_.h = kernel_y;
    convert_out_dim_.w = kernel_x;
    converted_filter_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, convert_out_dim_);

    auto dst = converted_filter_->getDataPtr();
    auto src = conv_descriptor_.filter_->getDataPtr();

    size_t local_align_weight[3] = {8, 4, 4};
    size_t global_align_weight[3] = {
        static_cast<size_t>(dst_groups), static_cast<size_t>(src_slices), static_cast<size_t>(kernel_y * kernel_x)};
    global_align_weight[0] = alignTo(global_align_weight[0], local_align_weight[0]);
    global_align_weight[1] = alignTo(global_align_weight[1], local_align_weight[1]);
    global_align_weight[2] = alignTo(global_align_weight[2], local_align_weight[2]);

    Status state = runtime_->setKernel(&align_weight_kernel_, "alignWeight_powervr", precision_);
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel failure\n");
    state = runtime_->setKernelArg(align_weight_kernel_.get(),
                                   src,
                                   dst,
                                   weight_dim.n,
                                   weight_dim.c,
                                   kernel_y,
                                   kernel_x,
                                   dst_groups,
                                   src_slices,
                                   block_size_z_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "set kernel failure\n");
    state = runtime_->enqueueKernel(align_weight_kernel_.get(), 3, global_align_weight, local_align_weight);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
