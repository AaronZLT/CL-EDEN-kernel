#pragma once
#include "userdriver/common/operator_interfaces/common/ActivationInfo.hpp"
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/common/operator_interfaces/common/Error.hpp"
#include "userdriver/gpu/common/CLParameter.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLRelu.hpp"
#include "userdriver/gpu/operators/CLRelu1.hpp"
#include "userdriver/gpu/operators/CLRelu6.hpp"
#include "userdriver/gpu/operators/CLSigmoid.hpp"
#include "userdriver/gpu/operators/CLTanh.hpp"

namespace enn {
namespace ud {
namespace gpu {

struct ActivationParameters : public Parameters {
    ActivationInfo activation_info = ActivationInfo();
    std::shared_ptr<ReluParameters> relu_parameters;
};

class CLActivation {
  public:
    CLActivation(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);
    Status initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                      const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                      const std::shared_ptr<Parameters> parameters);
    Status execute();
    Status release();

  private:
    // 1. Runtime Context
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;

    // 2. Operator Resource
    std::shared_ptr<ActivationParameters> parameters_;

    // 2. activation types
    std::shared_ptr<CLRelu> relu_;
    std::shared_ptr<CLRelu1> relu1_;
    std::shared_ptr<CLRelu6> relu6_;
    std::shared_ptr<CLSigmoid> sigmoid_;
    std::shared_ptr<CLTanh> tanh_;
};

}  // namespace gpu
}  // namespace ud
}  // namespace enn
