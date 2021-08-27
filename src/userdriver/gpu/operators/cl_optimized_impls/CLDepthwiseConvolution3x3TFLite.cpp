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
 * @file    CLDepthwiseConvolution3x3TFLite.cpp
 * @brief
 * @details
 * @version
 */

#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLDepthwiseConvolution3x3TFLite.hpp"

namespace enn {
namespace ud {
namespace gpu {

CLDepthwiseConvolution3x3TFLite::CLDepthwiseConvolution3x3TFLite(const std::shared_ptr<CLRuntime> runtime,
                                                                 const PrecisionType &precision) :
    runtime_(runtime),
    precision_(precision) {
    ENN_DBG_PRINT("CLDepthwiseConvolution3x3TFLite is created");
    input_dim_ = {0, 0, 0, 0};
    output_dim_ = {0, 0, 0, 0};
    stride_ = {0, 0};
    pad_ = {0, 0, 0, 0};
    dst_size_x_ = 0;
    dst_size_y_ = 0;
    dst_size_z_ = 0;

    grid_[0] = 0;
    grid_[1] = 0;
    grid_[2] = 0;
    best_work_group_[0] = 0;
    best_work_group_[1] = 0;
    best_work_group_[2] = 0;
}

Status CLDepthwiseConvolution3x3TFLite::rearrange_weights_and_biases_data(const std::shared_ptr<CLTensor> &weights,
                                                                          const std::shared_ptr<CLTensor> &biases,
                                                                          std::shared_ptr<CLTensor> &weights_biases) {
    Status state = Status::FAILURE;
    const int src_depth = IntegralDivideRoundUp(weights->getDim().c, 4);

    std::shared_ptr<struct _cl_kernel> kernel_depthwise_3x3_rearrange_w;
    state = runtime_->setKernel(&kernel_depthwise_3x3_rearrange_w, "dw_tflite_3x3_rearrange_w", precision_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "dw_tflite_3x3_rearrange_w setKernel failure\n");

    size_t global = src_depth;
    state = runtime_->setKernelArg(kernel_depthwise_3x3_rearrange_w.get(),
                                   weights->getDataPtr(),
                                   biases->getDataPtr(),
                                   weights_biases->getDataPtr(),
                                   weights->getDim().c,
                                   src_depth);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "kernel_depthwise_3x3_rearrange_w setKernelArg failure\n");

    state = runtime_->enqueueKernel(kernel_depthwise_3x3_rearrange_w.get(), (cl_uint)1, &global, NULL);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "kernel_depthwise_3x3_rearrange_w enqueueKernel failure\n");

    return state;
}

Status CLDepthwiseConvolution3x3TFLite::upload_weights_and_biases() {
    Status state = Status::FAILURE;

    int src_depth = IntegralDivideRoundUp(filter_->getDim().c, 4);
    int texture_width = 10;  // 3x3 kernel + 1 bias
    int texture_height = src_depth;
    uint32_t elements_count = texture_width * texture_height * 4;
    int float4_size = 8;

    Dim4 weight_bias_dim = {elements_count, 1, 1, 1};
    weight_bias_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, weight_bias_dim);

    state = rearrange_weights_and_biases_data(
        std::static_pointer_cast<CLTensor>(filter_), std::static_pointer_cast<CLTensor>(bias_), weight_bias_);

    return state;
}

Status CLDepthwiseConvolution3x3TFLite::initialize(const Dim4 &input_dim,
                                                   const Dim4 &output_dim,
                                                   const std::shared_ptr<ITensor> filter,
                                                   const std::shared_ptr<ITensor> bias,
                                                   const Dim2 &stride,
                                                   const Pad4 &pad,
                                                   const ActivationInfo &activate_info) {
    ENN_DBG_PRINT("CLDepthwiseConvolution3x3TFLite is initialized");
    Status state = Status::FAILURE;

    input_dim_ = input_dim;
    output_dim_ = output_dim;
    filter_ = filter;
    bias_ = bias;
    stride_ = stride;
    pad_ = pad;
    activation_info_ = activate_info;

    int dst_slice = IntegralDivideRoundUp(output_dim_.c, 4);
    dst_size_x_ = output_dim_.w;
    dst_size_y_ = output_dim_.h;
    dst_size_z_ = dst_slice;

    grid_[0] = (int)IntegralDivideRoundUp(output_dim_.w, 2);
    grid_[1] = (int)IntegralDivideRoundUp(output_dim_.h, 2);
    grid_[2] = dst_slice;

    //  The Dim order of filter is NCHW, but the data order of filter is NHWC.
    //  The data order of filter need to convert to NCHW
    auto filter_tensor = std::static_pointer_cast<CLTensor>(filter);
    std::shared_ptr<CLTensor> filter_nchw_;
    filter_nchw_ = std::make_shared<CLTensor>(runtime_,
                                              precision_,
                                              filter->getDataType(),
                                              filter->getDim(),
                                              filter->getDataOrder(),
                                              filter->getScale(),
                                              filter->getZeroPoint());

    DataType mdata_type = filter->getDataType();
    if (filter->getDataType() == DataType::FLOAT && precision_ == PrecisionType::FP16) {
        mdata_type = DataType::HALF;
    } else if (filter->getDataType() == DataType::HALF && precision_ == PrecisionType::FP32) {
        mdata_type = DataType::FLOAT;
    }
    runtime_->NHWC2NCHW(filter_tensor->getDataPtr(),
                        filter_nchw_->getDataPtr(),
                        filter_nchw_->getDim(),
                        mdata_type,
                        PrecisionChangeMode::OTHER);

    filter_ = std::static_pointer_cast<ITensor>(filter_nchw_);

    GetBestWorkGroup(grid_, best_work_group_, (int)runtime_->getMaxWorkGroupSize()[2]);

    // convert weight nchw -> nc/4hw4;
    upload_weights_and_biases();

    // create kernel
    if (activation_info_.activation() == ActivationInfo::ActivationType::RELU) {
        state = runtime_->setKernel(&kernel_depthwise_tflite_3x3_, "RELUdw_tflite_3x3", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "RELUdw_tflite_3x3 setKernel failure\n");
    } else if (activation_info_.activation() == ActivationInfo::ActivationType::RELU6) {
        state = runtime_->setKernel(&kernel_depthwise_tflite_3x3_, "RELU6dw_tflite_3x3", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "RELU6dw_tflite_3x3 setKernel failure\n");
    } else {
        state = runtime_->setKernel(&kernel_depthwise_tflite_3x3_, "dw_tflite_3x3", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "dw_tflite_3x3 setKernel failure\n");
    }
    return state;
}

Status CLDepthwiseConvolution3x3TFLite::execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) {
    ENN_DBG_PRINT("CLDepthwiseDeconvolution::execute() is called");
    Status state = Status::FAILURE;

    auto input_tensor = std::static_pointer_cast<CLTensor>(input);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output);
    auto input_data = input_tensor->getDataPtr();
    auto output_data = output_tensor->getDataPtr();
    auto filter_bias_data = weight_bias_->getDataPtr();

    size_t local[3] = {static_cast<size_t>(best_work_group_[0]),
                       static_cast<size_t>(best_work_group_[1]),
                       static_cast<size_t>(best_work_group_[2])};
    size_t global[3] = {0, 0, 0};
    global[0] = static_cast<size_t>(AlignByN(IntegralDivideRoundUp(output_dim_.w, 2), local[0]));
    global[1] = static_cast<size_t>(AlignByN(IntegralDivideRoundUp(output_dim_.h, 2), local[1]));
    global[2] = static_cast<size_t>(AlignByN(IntegralDivideRoundUp(output_dim_.c, 4), local[2]));

    state = runtime_->setKernelArg(kernel_depthwise_tflite_3x3_.get(),
                                   input_data,
                                   filter_bias_data,
                                   output_data,
                                   dst_size_x_,
                                   dst_size_y_,
                                   dst_size_z_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "kernel_depthwise_tflite_3x3_ setKernelArg failure\n");

    state = runtime_->enqueueKernel(kernel_depthwise_tflite_3x3_.get(), (cl_uint)3, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "kernel_depthwise_tflite_3x3_ enqueueKernel failure\n");
    return state;
}

Status CLDepthwiseConvolution3x3TFLite::release() { return Status::SUCCESS; }

}  // namespace gpu
}  // namespace ud
}  // namespace enn
