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
 * @file    CLDirectFullyConnected.cpp
 * @brief
 * @details
 * @version
 */

#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLDirectFullyConnected.hpp"

namespace enn {
namespace ud {
namespace gpu {

CLDirectFullyConnected::CLDirectFullyConnected(const std::shared_ptr<CLRuntime> runtime,
                                               const PrecisionType &precision,
                                               const std::shared_ptr<ITensor> input,
                                               const std::shared_ptr<ITensor> weight,
                                               const std::shared_ptr<ITensor> bias,
                                               const std::shared_ptr<ITensor> output,
                                               bool weights_as_input) {
    ENN_DBG_PRINT("CLDirectFullyConnected is called");
    Status state = Status::FAILURE;
    runtime_ = runtime;
    weight_ = std::static_pointer_cast<CLTensor>(weight);
    bias_ = std::static_pointer_cast<CLTensor>(bias);
    precision_ = precision;
    weights_as_input_ = weights_as_input;

    // set kernel
    state = runtime_->setKernel(&kernel_, "fc_direct_opt", precision_);
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "fc_direct_opt setKernel failure\n");
    state = runtime_->setKernel(&kernel_weight_cvt_, "fc_weight_cvt", precision_);
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "fc_weight_cvt_ setKernel failure\n");
    ENN_DBG_PRINT("CLDirectFullyConnected is created");

    // weight convert
    Dim4 weight_cvt_dim = {
        (uint32_t)(alignTo(weight->getDim().c, 16) / 16), weight->getDim().n * 16, weight->getDim().h, weight->getDim().w};
    weight_cvt_buffer_ = std::make_shared<CLTensor>(runtime_, precision_, weight->getDataType(), weight_cvt_dim);
    if (!weights_as_input) {
        weight_convert();
    }
}

Status CLDirectFullyConnected::execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) {
    auto input_tensor = std::static_pointer_cast<CLTensor>(input);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output);

    if (weights_as_input_) {
        weight_convert();
    }

    return fullyConnectedFloat(input_tensor, output_tensor);
}

Status CLDirectFullyConnected::fullyConnectedFloat(const std::shared_ptr<CLTensor> input_tensor,
                                                   std::shared_ptr<CLTensor> output_tensor) {
    ENN_DBG_PRINT("CLDirectFullyConnected::execute() is called");

    Status state = Status::FAILURE;

    uint32_t input_batch = input_tensor->getDim().n;
    uint32_t input_channel = input_tensor->getDim().c;
    uint32_t input_height = input_tensor->getDim().h;
    uint32_t input_width = input_tensor->getDim().w;
    uint32_t output_channel = output_tensor->getDim().c;

    int matrix_width = input_channel * input_height * input_width;
    int matrix_height = output_tensor->getDim().c * output_tensor->getDim().h * output_tensor->getDim().w;

    size_t local[1] = {32};
    size_t global[1] = {(size_t)alignTo(matrix_height, local[0])};

    state = runtime_->setKernelArg(kernel_.get(),
                                   input_tensor->getDataPtr(),
                                   weight_cvt_buffer_->getDataPtr(),
                                   bias_->getDataPtr(),
                                   output_tensor->getDataPtr(),
                                   matrix_height,
                                   matrix_width);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "fc_direct_opt setKernelArg failure\n");
    state = runtime_->enqueueKernel(kernel_.get(), (cl_uint)1, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute fc_direct_opt kernel failure\n");

    return Status::SUCCESS;
}

Status CLDirectFullyConnected::release() { return Status::SUCCESS; }

Status CLDirectFullyConnected::weight_convert() {
    Status state = Status::FAILURE;

    int matrix_height = weight_->getDim().n;
    int matrix_width = weight_->getDim().c;

    size_t global[2] = {(size_t)weight_cvt_buffer_->getDim().n, (size_t)(weight_cvt_buffer_->getDim().c)};
    state = runtime_->setKernelArg(
        kernel_weight_cvt_.get(), weight_->getDataPtr(), weight_cvt_buffer_->getDataPtr(), matrix_height, matrix_width);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "kernel_weight_cvt_ setKernelArg failure\n");

    state = runtime_->enqueueKernel(kernel_weight_cvt_.get(), (cl_uint)2, global, NULL);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel_weight_cvt_ kernel failure\n");

    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
