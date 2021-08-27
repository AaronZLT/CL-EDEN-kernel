#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/common/operator_interfaces/common/Error.hpp"
#include "userdriver/gpu/common/CLParameter.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLDeQuantization.hpp"
#include "userdriver/gpu/operators/CLQuantization.hpp"

namespace enn {
namespace ud {
namespace gpu {

struct SoftmaxParameters : public Parameters {
    int32_t axis = 0;
    float beta = 0.0f;
    bool androidNN = false;
    bool adjustAcc = false;
};

class CLSoftmax {
public:
    CLSoftmax(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);
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
    std::shared_ptr<SoftmaxParameters> parameters_;

    // 3. Operator kernels
    std::shared_ptr<struct _cl_kernel> kernel_;

    // 4. other members
    std::shared_ptr<CLDeQuantization> dequantization_;
    std::shared_ptr<CLQuantization> quantization_;
    std::shared_ptr<CLTensor> channel_max_tensor_;
    std::shared_ptr<CLTensor> channel_sum_tensor_;
    std::shared_ptr<CLTensor> map_input_tensor_;
    std::unique_ptr<float[]> channel_max_ptr_;
    uint32_t inner_number_;
    uint32_t channels_;
    uint32_t out_number_;

    // 5. execute functions
    Status softmaxFloat();
    Status softmaxQuant();
    Status softmaxAxis2();
};

}  // namespace gpu
}  // namespace ud
}  // namespace enn
