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

class CLMaxpool {
public:
    CLMaxpool(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);
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
    std::shared_ptr<CLTensor> output_tensor_;
    std::shared_ptr<CLTensor> input_nchw_tensor_;
    std::shared_ptr<CLTensor> output_nchw_tensor_;
    std::shared_ptr<Pool2DParameters> parameters_;
    std::shared_ptr<CLActivation> cl_activation_;

    // 3. Operator kernels
    std::shared_ptr<struct _cl_kernel> kernel_;

    // 4. execute functions
    Status maxpool_float(const std::shared_ptr<CLTensor> input, std::shared_ptr<CLTensor> output);
    Status maxpool_quant(const std::shared_ptr<CLTensor> input, std::shared_ptr<CLTensor> output);
};

}  // namespace gpu
}  // namespace ud
}  // namespace enn
