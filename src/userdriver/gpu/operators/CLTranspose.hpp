#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/common/operator_interfaces/common/Error.hpp"
#include "userdriver/gpu/common/CLParameter.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"

namespace enn {
namespace ud {
namespace gpu {

struct TransposeParameters : public Parameters {
    std::vector<int32_t> perm;
    bool androidNN = false;
};

class CLTranspose {
  public:
    CLTranspose(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);
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
    std::shared_ptr<CLTensor> input_tensor_;
    std::shared_ptr<CLTensor> perm_tensor_;
    std::shared_ptr<CLTensor> output_tensor_;
    std::shared_ptr<TransposeParameters> parameters_;

    // 3. Operator kernels
    std::shared_ptr<struct _cl_kernel> kernel_;
};

}  // namespace gpu
}  // namespace ud
}  // namespace enn
