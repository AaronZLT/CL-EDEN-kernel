#pragma once

#include <set>
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/common/operator_interfaces/common/Error.hpp"
#include "userdriver/gpu/common/CLParameter.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"

namespace enn {
namespace ud {
namespace gpu {
struct SliceParameters : public Parameters {
    int32_t axis = 0;
    std::vector<int32_t> slice_point;
};

class CLSlice {
public:
    CLSlice(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);
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
    std::vector<std::shared_ptr<CLTensor>> output_tensors_;
    std::shared_ptr<SliceParameters> parameters_;

    // 3. Operator kernels
    std::shared_ptr<struct _cl_kernel> kernel_NC_;
    std::shared_ptr<struct _cl_kernel> kernel_HW_;

    // 4. Operator variables
    int32_t axis_ = 0;
    std::vector<int32_t> slice_point_;
};

}  // namespace gpu
}  // namespace ud
}  // namespace enn
