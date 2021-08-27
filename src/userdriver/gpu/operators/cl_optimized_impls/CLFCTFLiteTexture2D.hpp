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
 * @file    CLFCTFLiteTexture2D.hpp
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

namespace enn {
namespace ud {
namespace gpu {

class CLFCTFLiteTexture2D {
  public:
    CLFCTFLiteTexture2D(const std::shared_ptr<CLRuntime> runtime,
                        const PrecisionType &precision,
                        const Dim4 &input_dim,
                        const Dim4 &output_dim);

    Status initialize(const std::shared_ptr<ITensor> weight,
                      const std::shared_ptr<ITensor> bias,
                      const ActivationInfo &activate_info,
                      bool weights_as_input = false);

    Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output);

    Status release();

  private:
    Status alignWeight();

    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;

    Dim4 input_dim_;
    Dim4 output_dim_;
    Dim4 convert_out_dim_;
    Dim4 weight_dim_;
    ActivationInfo activation_info_;

    bool weights_as_input_;

    std::shared_ptr<CLTensor> bias_;
    std::shared_ptr<CLTensor> filter_;
    std::shared_ptr<CLTensor> converted_filter_;
    std::shared_ptr<struct _cl_kernel> align_weight_kernel_;
    std::shared_ptr<struct _cl_kernel> kernel_;
};  // class CLFCTFLiteTexture2D

}  // namespace gpu
}  // namespace ud
}  // namespace enn
