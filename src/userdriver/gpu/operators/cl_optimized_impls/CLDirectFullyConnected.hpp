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
 * @file    CLDirectFullyConnected.hpp
 * @brief
 * @details
 * @version
 */

#pragma once

#include "userdriver/common/operator_interfaces/common/ActivationInfo.hpp"
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/ITensor.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLDeQuantization.hpp"
#include "userdriver/gpu/operators/CLQuantization.hpp"

namespace enn {
namespace ud {
namespace gpu {

class CLDirectFullyConnected {
  public:
    CLDirectFullyConnected(const std::shared_ptr<CLRuntime> runtime,
                           const PrecisionType &precision,
                           const std::shared_ptr<ITensor> input,
                           const std::shared_ptr<ITensor> weight,
                           const std::shared_ptr<ITensor> bias,
                           const std::shared_ptr<ITensor> output,
                           bool weights_as_input = false);

    Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output);

    Status release();

    Status weight_convert();

  private:
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;
    bool weights_as_input_;
    std::shared_ptr<struct _cl_kernel> kernel_;
    std::shared_ptr<struct _cl_kernel> kernel_weight_cvt_;
    std::shared_ptr<CLTensor> weight_;
    std::shared_ptr<CLTensor> bias_;
    std::shared_ptr<CLTensor> weight_cvt_buffer_;

    Status fullyConnectedQuant(const std::shared_ptr<CLTensor> input, std::shared_ptr<CLTensor> output);
    Status fullyConnectedFloat(const std::shared_ptr<CLTensor> input, std::shared_ptr<CLTensor> output);
};  // class CLDirectFullyConnected

}  // namespace gpu
}  // namespace ud
}  // namespace enn
