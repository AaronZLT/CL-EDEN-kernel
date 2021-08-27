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
 * @file    CLFCTFLiteTexture2D.cpp
 * @brief
 * @details
 * @version
 */

#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLFCTFLiteTexture2D.hpp"

namespace enn {
namespace ud {
namespace gpu {

CLFCTFLiteTexture2D::CLFCTFLiteTexture2D(const std::shared_ptr<CLRuntime> runtime,
                                         const PrecisionType &precision,
                                         const Dim4 &input_dim,
                                         const Dim4 &output_dim) :
    runtime_(runtime),
    precision_(precision), input_dim_(input_dim), output_dim_(output_dim) {
    ENN_DBG_PRINT("CLFCTFLiteTexture2D is created");
    convert_out_dim_ = {0, 0, 0, 0};
    weight_dim_ = {1, 1, 1, 1};
    weights_as_input_ = false;
}

Status CLFCTFLiteTexture2D::initialize(const std::shared_ptr<ITensor> weight,
                                       const std::shared_ptr<ITensor> bias,
                                       const ActivationInfo &activate_info,
                                       bool weights_as_input) {
    ENN_DBG_PRINT("CLFCTFLiteTexture2D::initialize() is called");
    Status state;
    filter_ = std::static_pointer_cast<CLTensor>(weight);
    bias_ = std::static_pointer_cast<CLTensor>(bias);
    weight_dim_ = filter_->getDim();
    alignWeight();
    if (activate_info.isEnabled()) {
        switch (activate_info.activation()) {
        case ActivationInfo::ActivationType::RELU:
            state = runtime_->setKernel(&kernel_, "RELUtflite_texture2d_fc", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
            break;
        case ActivationInfo::ActivationType::RELU6:
            state = runtime_->setKernel(&kernel_, "RELU6tflite_texture2d_fc", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
            break;
        default:
            state = runtime_->setKernel(&kernel_, "tflite_texture2d_fc", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
            break;
        }
    } else {
        state = runtime_->setKernel(&kernel_, "tflite_texture2d_fc", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
    }
    return Status::SUCCESS;
}

Status CLFCTFLiteTexture2D::execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) {
    //    return Status::SUCCESS;
    ENN_DBG_PRINT("CLFCTFLiteTexture2D is execute");
    auto input_tensor = std::static_pointer_cast<CLTensor>(input);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output);
    // execute->prepare
    auto input_dim = input_tensor->getDim();
    auto output_dim = output_tensor->getDim();

    int src_slices = IntegralDivideRoundUp(input_dim.c, 4);
    int dst_slices = IntegralDivideRoundUp(output_dim.c, 4);

    size_t local[3] = {32, 4, 1};
    size_t global[3] = {1, 1, 1};
    global[0] = AlignByN(static_cast<size_t>(dst_slices), local[0]);
    global[1] = AlignByN(global[1], local[1]);
    global[2] = AlignByN(global[2], local[2]);

    Status state = runtime_->setKernelArg(kernel_.get(),
                                          input_tensor->getDataPtr(),
                                          converted_filter_->getDataPtr(),
                                          bias_->getDataPtr(),
                                          output_tensor->getDataPtr(),
                                          src_slices,
                                          dst_slices);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "set kernel failure\n");
    Status status = runtime_->enqueueKernel(kernel_.get(), 3, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "execute kernel failure\n");
    return Status::SUCCESS;
}

Status CLFCTFLiteTexture2D::alignWeight() {
    ENN_DBG_PRINT("CLFCTFLiteTexture2D is in alignWeight");
    uint32_t div = 4;
    const int dst_slices = IntegralDivideRoundUp(weight_dim_.n, div);
    const int src_slices = IntegralDivideRoundUp(weight_dim_.c, div);

    const int kernel_x = weight_dim_.w;
    const int kernel_y = weight_dim_.h;
    convert_out_dim_.n = dst_slices * div;
    convert_out_dim_.c = src_slices * div;
    convert_out_dim_.h = kernel_y;
    convert_out_dim_.w = kernel_x;
    converted_filter_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, convert_out_dim_);

    auto dst = converted_filter_->getDataPtr();
    auto src = filter_->getDataPtr();

    size_t local_align_weight[2] = {32, 4};
    size_t global_align_weight[2] = {static_cast<size_t>(src_slices), static_cast<size_t>(dst_slices)};
    global_align_weight[0] = alignTo(global_align_weight[0], local_align_weight[0]);
    global_align_weight[1] = alignTo(global_align_weight[1], local_align_weight[1]);

    Status state = runtime_->setKernel(&align_weight_kernel_, "alignWeight_tflitefc", precision_);
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel failure\n");
    state = runtime_->setKernelArg(
        align_weight_kernel_.get(), src, dst, weight_dim_.n, weight_dim_.c, kernel_y, kernel_x, dst_slices, src_slices);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "set kernel failure\n");
    state = runtime_->enqueueKernel(align_weight_kernel_.get(), 2, global_align_weight, local_align_weight);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
    return Status::SUCCESS;
}

Status CLFCTFLiteTexture2D::release() { return Status ::SUCCESS; }

}  // namespace gpu
}  // namespace ud
}  // namespace enn
