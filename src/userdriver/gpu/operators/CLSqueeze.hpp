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

struct SqueezeParameters : public Parameters {
    std::vector<int32_t> squeeze_dims = {};
    bool androidNN = false;
    bool isNCHW = false;
};

class CLSqueeze {
public:
    CLSqueeze(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);
    Status initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                      const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                      const std::shared_ptr<Parameters> parameters);
    Status execute();
    Status release();

private:
    Status reconfigure_output();

private:
    // 1. Runtime context
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;

    // 2. Operator resource
    std::shared_ptr<CLTensor> input_tensor_;
    std::shared_ptr<CLTensor> squeeze_dims_;
    std::shared_ptr<CLTensor> output_tensor_;
    std::shared_ptr<SqueezeParameters> parameters_;

    // 3. Operator kernels
    std::shared_ptr<struct _cl_kernel> kernel_;

    // 4. Operator variables
    bool squeeze_dims_as_input_;
    std::shared_ptr<int32_t> squeeze_dims_arr_;
};

}  // namespace gpu
}  // namespace ud
}  // namespace enn
