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
 * @file    CLDepthwiseConvolutionTFLite.cpp
 * @brief
 * @details
 * @version
 */

#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLDepthwiseConvolutionTFLite.hpp"

namespace enn {
namespace ud {
namespace gpu {

CLDepthwiseConvolutionTFLite::CLDepthwiseConvolutionTFLite(const std::shared_ptr<CLRuntime> runtime,
                                                           const PrecisionType &precision) :
    runtime_(runtime),
    precision_(precision) {
    ENN_DBG_PRINT("CLDepthwiseConvolution3x3TFLite is created");
    input_dim_ = {0, 0, 0, 0};
    output_dim_ = {0, 0, 0, 0};
    stride_ = {0, 0};
    pad_ = {0, 0, 0, 0};
    dilation_ = {0, 0};
    kernel_size_x_ = 0;
    kernel_size_y_ = 0;
    stride_size_x_ = 0;
    stride_size_y_ = 0;
    pad_size_x_ = 0;
    pad_size_y_ = 0;
    dilation_size_x_ = 0;
    dilation_size_y_ = 0;
    src_size_x_ = 0;
    src_size_y_ = 0;
    src_size_z_ = 0;
    dst_size_x_ = 0;
    dst_size_y_ = 0;
    dst_size_z_ = 0;
}

Status CLDepthwiseConvolutionTFLite::RearrangeWeightsAndBiasesData(const std::shared_ptr<CLTensor> &weights,
                                                                   std::shared_ptr<CLTensor> &weights_converted) {
    Status state = Status::FAILURE;
    const int dst_channels = weights->getDim().c * weights->getDim().n;
    const int dst_depth = IntegralDivideRoundUp(dst_channels, 4);
    const int kernel_x = weights->getDim().w;
    const int kernel_y = weights->getDim().h;

    std::shared_ptr<struct _cl_kernel> kernel_depthwise_rearrange_w;
    state = runtime_->setKernel(&kernel_depthwise_rearrange_w, "dw_tflite_rearrange_w", precision_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "dw_tflite_rearrange_w setKernel failure\n");

    size_t global = dst_depth;
    state = runtime_->setKernelArg(kernel_depthwise_rearrange_w.get(),
                                   weights->getDataPtr(),
                                   weights_converted->getDataPtr(),
                                   kernel_y,
                                   kernel_x,
                                   dst_channels,
                                   dst_depth);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "kernel_depthwise_rearrange_w setKernelArg failure\n");

    state = runtime_->enqueueKernel(kernel_depthwise_rearrange_w.get(), (cl_uint)1, &global, NULL);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "kernel_depthwise_rearrange_w enqueueKernel failure\n");
    if (weights->getDim().c == 32) {
        weights_converted->dumpTensorData("eden_w.txt");
    }

    return state;
}

Status CLDepthwiseConvolutionTFLite::UploadWeightsAndBiases() {
    Status state = Status::FAILURE;

    uint32_t dst_channels = filter_->getDim().c * filter_->getDim().n;
    uint32_t dst_depth = IntegralDivideRoundUp(dst_channels, 4);
    uint32_t kernel_x = filter_->getDim().w;
    uint32_t kernel_y = filter_->getDim().h;

    uint32_t elements_count = kernel_x * kernel_y * dst_depth * 4;
    uint32_t float4_size = 8;

    Dim4 weight_converted_dim = {elements_count, 1, 1, 1};
    weight_converted_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, weight_converted_dim);
    state = RearrangeWeightsAndBiasesData(std::static_pointer_cast<CLTensor>(filter_), weight_converted_);

    return state;
}

Status CLDepthwiseConvolutionTFLite::initialize(const Dim4 &input_dim,
                                                const Dim4 &output_dim,
                                                const std::shared_ptr<ITensor> filter,
                                                const std::shared_ptr<ITensor> bias,
                                                const Dim2 &stride,
                                                const Pad4 &pad,
                                                const Dim2 &dilation,
                                                const ActivationInfo &activate_info) {
    ENN_DBG_PRINT("CLDepthwiseConvolution3x3TFLite is initialized");
    Status state = Status::FAILURE;

    input_dim_ = input_dim;
    output_dim_ = output_dim;
    filter_ = filter;
    bias_ = bias;
    stride_ = stride;
    pad_ = pad;
    dilation_ = dilation;
    activation_info_ = activate_info;

    kernel_size_x_ = filter_->getDim().w;
    kernel_size_y_ = filter_->getDim().h;
    stride_size_x_ = stride_.w;
    stride_size_y_ = stride_.h;
    pad_size_x_ = -pad_.l;
    pad_size_y_ = -pad_.t;
    dilation_size_x_ = dilation_.w;
    dilation_size_y_ = dilation_.h;
    src_size_x_ = input_dim_.w * input_dim_.n;
    src_size_y_ = input_dim_.h;
    src_size_z_ = IntegralDivideRoundUp(input_dim_.c, 4);
    dst_size_x_ = output_dim_.w * output_dim_.n;
    dst_size_y_ = output_dim_.h;
    dst_size_z_ = IntegralDivideRoundUp(output_dim_.c, 4);

    grid_[0] = (int)(output_dim_.w * output_dim_.n);
    grid_[1] = (int)output_dim_.h;
    grid_[2] = dst_size_z_;
    get_best_workgroup(grid_, best_work_group_, (int)runtime_->getMaxWorkGroupSize()[2]);

    //  The Dim order of filter is NCHW, but the data order of filter is NHWC.
    //  The data order of filter ned to convert to NCHW
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

    // convert weight nchw -> nc/4hw4;
    UploadWeightsAndBiases();

    // create kernel
    if (activation_info_.activation() == ActivationInfo::ActivationType::RELU) {
        state = runtime_->setKernel(&kernel_depthwise_tflite_, "RELUdw_tflite", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "RELUdw_tflite setKernel failure\n");
    } else if (activation_info_.activation() == ActivationInfo::ActivationType::RELU6) {
        state = runtime_->setKernel(&kernel_depthwise_tflite_, "RELU6dw_tflite", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "RELU6dw_tflite setKernel failure\n");
    } else {
        state = runtime_->setKernel(&kernel_depthwise_tflite_, "dw_tflite", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "dw_tflite setKernel failure\n");
    }

    return state;
}

Status CLDepthwiseConvolutionTFLite::execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) {
    ENN_DBG_PRINT("CLDepthwiseConvolutionTFLite::execute() is called");
    Status state = Status::FAILURE;

    auto input_tensor = std::static_pointer_cast<CLTensor>(input);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output);
    auto bias_tensor = std::static_pointer_cast<CLTensor>(bias_);
    auto input_data = input_tensor->getDataPtr();
    auto output_data = output_tensor->getDataPtr();
    auto filter_data = weight_converted_->getDataPtr();
    auto bias_data = bias_tensor->getDataPtr();

    size_t local[3] = {static_cast<size_t>(best_work_group_[0]),
                       static_cast<size_t>(best_work_group_[1]),
                       static_cast<size_t>(best_work_group_[2])};
    size_t global[3] = {0, 0, 0};

    global[0] = static_cast<size_t>(AlignByN(output_dim_.w * output_dim_.n, local[0]));
    global[1] = static_cast<size_t>(AlignByN(output_dim_.h, local[1]));
    global[2] = static_cast<size_t>(AlignByN(IntegralDivideRoundUp(output_dim_.c, 4), local[2]));

    state = runtime_->setKernelArg(kernel_depthwise_tflite_.get(),
                                   input_data,
                                   filter_data,
                                   bias_data,
                                   output_data,
                                   kernel_size_x_,
                                   kernel_size_y_,
                                   stride_size_x_,
                                   stride_size_y_,
                                   pad_size_x_,
                                   pad_size_y_,
                                   dilation_size_x_,
                                   dilation_size_y_,
                                   src_size_x_,
                                   src_size_y_,
                                   src_size_z_,
                                   dst_size_x_,
                                   dst_size_y_,
                                   dst_size_z_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "kernel_depthwise_tflite_ setKernelArg failure\n");

    state = runtime_->enqueueKernel(kernel_depthwise_tflite_.get(), (cl_uint)3, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "kernel_depthwise_tflite_ enqueueKernel failure\n");

    return state;
}

Status CLDepthwiseConvolutionTFLite::release() { return Status::SUCCESS; }

}  // namespace gpu
}  // namespace ud
}  // namespace enn
