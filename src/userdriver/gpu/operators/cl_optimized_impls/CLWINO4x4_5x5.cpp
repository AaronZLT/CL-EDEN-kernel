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
 * @file    CLWINO4x4_5x5.cpp
 * @brief
 * @details
 * @version
 */
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLWINO4x4_5x5.hpp"
#include "userdriver/common/operator_interfaces/common/Debug.hpp"

namespace enn {
namespace ud {
namespace gpu {

CLWINO4x4_5x5::CLWINO4x4_5x5(const std::shared_ptr<CLRuntime> runtime,
                             const PrecisionType &precision,
                             const Dim4 &input_dim,
                             const Dim4 &output_dim) :
    runtime_(runtime),
    precision_(precision), input_dim_(input_dim), output_dim_(output_dim) {
    ENN_DBG_PRINT("CLWINO4x4_5x5 is created");
    top_channel_need_aligned_ = false;
    branch_number_ = 0;
    speed_up_ = 0;
    bifrost_speedup_ = false;
    coalescing_feature_height_ = 0;
    compute_output_number_ = 0;
    splite_number_ = 0;
    wgrad_stride_ = 0;
    wgrad_tile_size_ = 0;
    wgrad_tile_height_ = 0;
    wgrad_tile_width_ = 0;
    wgrad_tile_total_ = 0;
    aligned_tiles_ = 0;
    aligned_inch_ = 0;
    aligned_outch_ = 0;
    aligned_pixel_ = 0;
    weights_as_input_ = false;
    androidNN_ = false;
    tile_size_ = 0;
    aligned_width_ = 0;
    weight_dim_ = {0, 0, 0, 0};
}

Status CLWINO4x4_5x5::initialize(const Pad4 &padding,
                                 const Dim2 &stride,
                                 const uint32_t &group_size,
                                 const uint32_t &axis,
                                 const Dim2 &dilation,
                                 const std::shared_ptr<ITensor> weight,
                                 const std::shared_ptr<ITensor> bias,
                                 const ActivationInfo &activate_info,
                                 const bool &weights_as_input,
                                 const bool &androidNN) {
    ENN_DBG_PRINT("CLWINO4x4_5x5::initialize() is called");
    runtime_->resetIntraBuffer();
    weight_tensor_ = std::static_pointer_cast<CLTensor>(weight);
    bias_tensor_ = std::static_pointer_cast<CLTensor>(bias);
    weight_dim_ = weight_tensor_->getDim();
    conv_descriptor_.num_output_ = weight_dim_.n;
    conv_descriptor_.pad_right_ = padding.r;
    conv_descriptor_.pad_left_ = padding.l;
    conv_descriptor_.pad_top_ = padding.t;
    conv_descriptor_.pad_bottom_ = padding.b;
    conv_descriptor_.kernel_height_ = weight_dim_.h;
    conv_descriptor_.kernel_width_ = weight_dim_.w;
    conv_descriptor_.stride_height_ = stride.h;
    conv_descriptor_.stride_width_ = stride.w;
    conv_descriptor_.group_ = group_size;
    conv_descriptor_.axis_ = axis;
    conv_descriptor_.dilation_ = dilation;
    conv_descriptor_.filter_ = weight_tensor_;
    conv_descriptor_.bias_ = bias_tensor_;

    activation_info_ = activate_info;
    weights_as_input_ = weights_as_input;
    androidNN_ = androidNN;

    Status state = runtime_->setKernel(&weight_tm_kernel_, "wino5x5_weight_tm_nchw_opt_4x16", precision_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel wino5x5_weight_tm_nchw_opt_4x16 failure!\n");

    state = runtime_->setKernel(&input_tm_kernel_, "wino5x5_input_tm_nchw_opt_4x16", precision_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel win5x5_input_tm_nchw_opt_4x16 failed!\n");

    state = runtime_->setKernel(&dot_multiply_kernel_, "wino5x5_dot_multiply_nchw_opt_4x16", precision_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel wino5x5_dot_multiply_nchw_opt_4x16 failed!\n");
    if (!activation_info_.isEnabled()) {
        state = runtime_->setKernel(&output_tm_kernel_, "wino5x5_output_tm_nchw_opt_4x16", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel wino5x5_output_tm_nchw_opt_4x16 failed!\n");
    } else {
        if (activate_info.activation() == ActivationInfo::ActivationType::RELU) {
            state = runtime_->setKernel(&output_tm_kernel_, "RELUwino5x5_output_tm_nchw_opt_4x16", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel RELUwino5x5_output_tm_nchw_opt_4x16 failed!\n");
        } else {
            ERROR_PRINT("Non supported kernel size %dx%d in wino4x4_5x5 convolution\n", weight_dim_.h, weight_dim_.w);
            return Status::FAILURE;
        }
    }
    // tile parameter
    aligned_inch_ = 2;
    aligned_outch_ = 16;

    tile_size_ = 64;
    aligned_width_ = 2;

    aligned_pixel_ = 4;

    wgrad_tile_height_ = (output_dim_.h + 3) / 4;
    wgrad_tile_width_ = (output_dim_.w + 3) / 4;
    wgrad_tile_width_ = alignTo(wgrad_tile_width_, aligned_width_);
    wgrad_tile_total_ = wgrad_tile_width_ * wgrad_tile_height_;

    if (0 == wgrad_tile_total_ % 32) {
        coalescing_feature_height_ = 32;
    } else if (0 == wgrad_tile_total_ % 24) {
        coalescing_feature_height_ = 24;
    } else {
        coalescing_feature_height_ = 64;
    }

    aligned_tiles_ = alignTo(wgrad_tile_total_, coalescing_feature_height_);
    Dim4 input_tm_dim_ = {
        input_dim_.n, (uint32_t)alignTo(input_dim_.c, aligned_inch_), aligned_tiles_, (uint32_t)tile_size_};
    input_tm_buffer_ = std::make_shared<CLTensor>(
        runtime_, precision_, DataType::HALF, input_tm_dim_, DataOrder::OTHER, 1.0, 0, BufferType::DEDICATED); // INTRA_SHARED?

    uint32_t aligned_outch = alignTo(output_dim_.c, aligned_outch_);
    Dim4 dot_dim_ = {output_dim_.n, aligned_outch, aligned_tiles_, (uint32_t)tile_size_};
    dot_buffer_ = std::make_shared<CLTensor>(
        runtime_, precision_, DataType::HALF, dot_dim_, DataOrder::OTHER, 1.0, 0, BufferType::DEDICATED); // INTRA_SHARED?

    if (weights_as_input == false) {
        state = convertWeight();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel convertWeight() failed!\n");
    }
    return Status::SUCCESS;
}

Status CLWINO4x4_5x5::execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) {
    ENN_DBG_PRINT("CLWINO4x4_5x5::execute() is called");
    if (weights_as_input_) {
        Status state = convertWeight();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel convertWeight() failed!\n");
    }
    auto input_tensor = std::static_pointer_cast<CLTensor>(input);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output);
    // wino
    Dim4 input_dim = input_tensor->getDim();
    Dim4 output_dim = output_tensor->getDim();
    int input_c_aligned = alignTo(input_dim.c, aligned_inch_);

    size_t global[3];
    size_t local[3];
    // 1. input_tm
    global[0] = aligned_tiles_ / 2;
    global[1] = 4;
    global[2] = input_c_aligned * input_dim.n;

    local[0] = coalescing_feature_height_ / 2;
    local[1] = 4;
    local[2] = aligned_inch_;

    global[0] = alignTo(global[0], local[0]);
    global[1] = alignTo(global[1], local[1]);
    global[2] = alignTo(global[2], local[2]);

    int src_n = input_dim.n;
    int src_c = input_dim.c;
    int src_h = input_dim.h;
    int src_w = input_dim.w;
    int dst_n = input_dim.n;
    int dst_c = input_c_aligned;
    int dst_h = tile_size_;
    int dst_w = aligned_tiles_;

    Status status = runtime_->setKernelArg(input_tm_kernel_.get(),
                                           input_tensor->getDataPtr(),
                                           input_tm_buffer_->getDataPtr(),
                                           src_n,
                                           src_c,
                                           src_h,
                                           src_w,
                                           dst_n,
                                           dst_c,
                                           dst_h,
                                           dst_w,
                                           conv_descriptor_.pad_left_,
                                           conv_descriptor_.pad_top_,
                                           wgrad_tile_total_,
                                           coalescing_feature_height_,
                                           wgrad_tile_width_,
                                           aligned_inch_,
                                           aligned_pixel_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg wino5x5_input_tm_nchw_opt_4x16 failed\n");

    status = runtime_->enqueueKernel(input_tm_kernel_.get(), (cl_uint)3, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "enqueue wino5x5_input_tm_nchw_opt_4x16 failed\n");

    // 2. dot
    global[0] = aligned_tiles_ / 4;
    global[1] = 64;
    global[2] = alignTo(output_dim.c, aligned_outch_) / (aligned_outch_)*output_dim.n;

    local[0] = coalescing_feature_height_ / 4;
    local[1] = aligned_pixel_;
    local[2] = 4;

    global[0] = alignTo(global[0], local[0]);
    global[1] = alignTo(global[1], local[1]);
    global[2] = alignTo(global[2], local[2]);

    src_n = input_dim.n;
    src_c = input_dim.c;
    src_h = 64;
    src_w = aligned_tiles_;
    dst_n = input_dim.n;
    dst_c = alignTo(output_dim.c, aligned_outch_) / aligned_outch_;
    dst_h = 64;
    dst_w = src_w;

    status = runtime_->setKernelArg(dot_multiply_kernel_.get(),
                                    input_tm_buffer_->getDataPtr(),
                                    weight_tm_buffer_->getDataPtr(),
                                    dot_buffer_->getDataPtr(),
                                    dst_n,
                                    dst_c,
                                    dst_h,
                                    dst_w,
                                    input_dim.c,
                                    output_dim.c,
                                    coalescing_feature_height_,
                                    aligned_outch_,
                                    aligned_inch_,
                                    aligned_pixel_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg wino5x5_dot_multiply_nchw_opt_4x16 failed\n");

    status = runtime_->enqueueKernel(dot_multiply_kernel_.get(), (cl_uint)3, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "enqueue wino5x5_dot_multiply_nchw_opt_4x16 failed\n");

    // 3. output_tm
    global[0] = aligned_tiles_ / 2;
    global[1] = 4;
    global[2] = (alignTo(output_dim.c, aligned_outch_)) * output_dim.n;

    local[0] = coalescing_feature_height_ / 2;
    local[1] = 4;
    local[2] = 1;

    src_n = output_dim.n;
    src_c = alignTo(output_dim.c, aligned_outch_);
    src_h = 64;
    src_w = aligned_tiles_;

    status = runtime_->setKernelArg(output_tm_kernel_.get(),
                                    dot_buffer_->getDataPtr(),
                                    bias_tensor_->getDataPtr(),
                                    output_tensor->getDataPtr(),
                                    src_n,
                                    src_c,
                                    src_h,
                                    src_w,
                                    output_dim.n,
                                    output_dim.c,
                                    output_dim.h,
                                    output_dim.w,
                                    coalescing_feature_height_,
                                    wgrad_tile_width_,
                                    aligned_outch_,
                                    aligned_inch_,
                                    aligned_pixel_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg wino5x5_output_tm_nchw_opt_4x16 failed\n");

    status = runtime_->enqueueKernel(output_tm_kernel_.get(), (cl_uint)3, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "enqueue wino5x5_output_tm_nchw_opt_4x16 failed\n");

    return Status::SUCCESS;
}

Status CLWINO4x4_5x5::convertWeight() {
    ENN_DBG_PRINT("CLWINO4x4_5x5 convertWeight");
    if (androidNN_) {
        Dim4 weight_dim_nchw = {weight_dim_.n, weight_dim_.w, weight_dim_.c, weight_dim_.h};
        weight_nchw_ = std::make_shared<CLTensor>(runtime_,
                                                  precision_,
                                                  weight_tensor_->getDataType(),
                                                  weight_dim_nchw,
                                                  weight_tensor_->getDataOrder(),
                                                  weight_tensor_->getScale(),
                                                  weight_tensor_->getZeroPoint());
        weight_dim_ = weight_dim_nchw;
        Status state = weight_tensor_->convertToNCHW(weight_nchw_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "convertToNCHW failure\n");
    }
    size_t local[2] = {16, 1};
    size_t global[2] = {(size_t)alignTo(weight_dim_.c, aligned_inch_), (size_t)alignTo(weight_dim_.n, aligned_outch_ * 2)};

    global[0] = alignTo(global[0], local[0]);
    global[1] = alignTo(global[1], local[1]);

    int weight_tm_size = tile_size_ * alignTo(weight_dim_.c, aligned_inch_) * alignTo(weight_dim_.n, aligned_outch_ * 2);
    Dim4 weight_tm_buffer_dim = {(uint32_t)weight_tm_size, 1, 1, 1};
    weight_tm_buffer_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::HALF, weight_tm_buffer_dim);

    Status state = runtime_->setKernelArg(weight_tm_kernel_.get(),
                                          androidNN_ ? weight_nchw_->getDataPtr() : weight_tensor_->getDataPtr(),
                                          weight_tm_buffer_->getDataPtr(),
                                          weight_dim_.n,
                                          weight_dim_.c,
                                          aligned_outch_,
                                          aligned_inch_,
                                          aligned_pixel_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");

    state = runtime_->enqueueKernel(weight_tm_kernel_.get(), (cl_uint)2, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "enqueue kernel failure\n");

    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
