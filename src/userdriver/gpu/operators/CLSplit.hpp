#pragma once
#include "userdriver/gpu/common/CLParameter.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"

namespace enn {
namespace ud {
namespace gpu {

struct SplitParameters : public Parameters {
    int32_t axis = 0;
    int32_t num_outputs = 1;
    bool androidNN = false;
};

class CLSplit {
public:
    CLSplit(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);
    Status initialize(std::vector<std::shared_ptr<ITensor>> input_tensors,
                      std::vector<std::shared_ptr<ITensor>> output_tensors,
                      std::shared_ptr<Parameters> parameters);
    Status execute();
    Status release();

private:
    // 1. Runtime context
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;

    // 2. Operator resource
    std::shared_ptr<ITensor> input_tensor_;
    std::vector<std::shared_ptr<CLTensor>> output_tensors_;
    std::shared_ptr<SplitParameters> parameters_;

    // 3. Operator kernels
    std::shared_ptr<struct _cl_kernel> kernel_;
};

}  // namespace gpu
}  // namespace ud
}  // namespace enn
