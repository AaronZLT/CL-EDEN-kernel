#pragma once

#include "userdriver/common/operator_interfaces/common/ActivationInfo.hpp"
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/common/operator_interfaces/common/Error.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/cl_quantized_utils/KernelUtil.hpp"
#include "userdriver/gpu/operators/cl_quantized_utils/QuantizationUtil.hpp"

namespace enn {
namespace ud {
namespace gpu {

class CLAddQuantized {
  public:
    CLAddQuantized(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);
    Status initialize(const std::vector<std::shared_ptr<ITensor>> inputs,
                      std::shared_ptr<ITensor> output,
                      const ActivationInfo &activation_info);
    Status execute();
    Status release();

  private:
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;

    std::shared_ptr<CLTensor> reshaped_input_tensor_0_;
    std::shared_ptr<CLTensor> reshaped_input_tensor_1_;
    std::shared_ptr<CLTensor> output_tensor_;
    ActivationInfo activation_info_;

    std::shared_ptr<struct _cl_kernel> kernel_;

  private:
    struct EltwiseAddParams {
        int left_shift_ = 0;
        int input0_offset_ = 0;
        int input1_offset_ = 0;
        int output_offset_ = 0;
        int input0_shift_ = 0;
        int input1_shift_ = 0;
        int output_shift_ = 0;
        int input0_multiplier_ = 0;
        int input1_multiplier_ = 0;
        int output_multiplier_ = 0;
        int act_min_ = 0;
        int act_max_ = 0;
    } eltAdd_descriptor_;
};

}  // namespace gpu
}  // namespace ud
}  // namespace enn
