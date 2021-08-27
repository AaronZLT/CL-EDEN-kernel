#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/common/operator_interfaces/common/Error.hpp"
#include "userdriver/gpu/common/CLParameter.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLActivation.hpp"

namespace enn {
namespace ud {
namespace gpu {

struct MulParameters : public Parameters {
    ActivationInfo activation_info = ActivationInfo();
    std::vector<float> coeff = {};
    bool androidNN = false;
    bool isNCHW = false;
};

class CLMul {
  public:
    CLMul(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);
    Status initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                      const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                      const std::shared_ptr<Parameters> parameters);
    Status execute();
    Status release();

  private:
    // 1. Runtime context
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;

    // 2. Operator resource
    std::vector<std::shared_ptr<CLTensor>> input_tensors_;
    std::shared_ptr<CLTensor> output_tensor_;
    std::shared_ptr<MulParameters> parameters_;
    std::shared_ptr<CLActivation> cl_activation_;

    // 3. Operator kernels
    std::shared_ptr<struct _cl_kernel> kernel_;
    std::shared_ptr<struct _cl_kernel> kernel_one_input_;

    // 4. for broadcast
    std::shared_ptr<CLTensor> input_broadcast_0_;
    std::shared_ptr<CLTensor> input_broadcast_1_;
    std::shared_ptr<CLTensor> input_broadcast_2_;
    std::shared_ptr<CLTensor> map_tensor_0_;
    std::shared_ptr<CLTensor> map_tensor_1_;

    // 5. execute functions
    Status mulFloat();
    Status mulQuant();
};

}  // namespace gpu
}  // namespace ud
}  // namespace enn
