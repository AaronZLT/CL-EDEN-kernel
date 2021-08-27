#pragma once
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/common/operator_interfaces/common/Error.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/common/CLParameter.hpp"

namespace enn {
namespace ud {
namespace gpu {

struct NormalizationrParameters : public Parameters {
    bool use_FP32_input_for_fp16 = false;
};

class CLNormalization {
  public:
    CLNormalization(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);
    Status initialize(std::vector<std::shared_ptr<ITensor>> inputs,
                      std::vector<std::shared_ptr<ITensor>> outputs,
                      std::shared_ptr<Parameters> parameters);
    Status execute();
    Status release();

  private:
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;
    std::shared_ptr<struct _cl_kernel> kernel_uint8_ = nullptr;
    std::shared_ptr<struct _cl_kernel> kernel_float_ = nullptr;
    std::shared_ptr<struct _cl_kernel> kernel_float_for_fp16_ = nullptr;

    std::shared_ptr<CLTensor> input_;
    std::shared_ptr<CLTensor> output_;
    std::shared_ptr<CLTensor> mean_;
    std::shared_ptr<CLTensor> scale_;
    std::shared_ptr<NormalizationrParameters> parameters_;
};  // class CLNormalization

}  // namespace gpu
}  // namespace ud
}  // namespace enn
