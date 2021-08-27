#pragma once

#include "userdriver/common/operator_interfaces/common/ActivationInfo.hpp"
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/ITensor.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLDeQuantization.hpp"
#include "userdriver/gpu/operators/CLQuantization.hpp"

namespace enn {
namespace ud {
namespace gpu {

class CLBaseFullyConnected {
  public:
    CLBaseFullyConnected(const std::shared_ptr<CLRuntime> runtime,
                         const PrecisionType &precision,
                         const std::shared_ptr<ITensor> input,
                         const std::shared_ptr<ITensor> weight,
                         const std::shared_ptr<ITensor> bias,
                         std::shared_ptr<ITensor> output,
                         const bool &weights_as_input = false);

    Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output);

    Status release();

  private:
    uint32_t computeSplitNum(const uint32_t &input_count);
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;
    std::shared_ptr<struct _cl_kernel> kernel_split_;
    std::shared_ptr<struct _cl_kernel> kernel_merge_;
    std::shared_ptr<CLTensor> weight_;
    std::shared_ptr<CLTensor> bias_;
    std::shared_ptr<CLTensor> split_buffer_;

    uint32_t split_number_;

    std::shared_ptr<CLQuantization> quantization_;
    std::shared_ptr<CLDeQuantization> dequantization_input_;
    std::shared_ptr<CLDeQuantization> dequantization_weight_;
    std::shared_ptr<CLDeQuantization> dequantization_bias_;
    std::shared_ptr<CLTensor> map_input_tensor_;
    std::shared_ptr<CLTensor> map_weight_tensor_;
    std::shared_ptr<CLTensor> map_bias_tensor_;
    std::shared_ptr<CLTensor> map_output_tensor_;

    Status fullyConnectedQuant(const std::shared_ptr<CLTensor> input,
                               const std::shared_ptr<CLTensor> weight,
                               const std::shared_ptr<CLTensor> bias,
                               std::shared_ptr<CLTensor> output);
    Status fullyConnectedFloat(const std::shared_ptr<CLTensor> input,
                               const std::shared_ptr<CLTensor> weight,
                               const std::shared_ptr<CLTensor> bias,
                               std::shared_ptr<CLTensor> output);
};  // class CLBaseFullyConnected

}  // namespace gpu
}  // namespace ud
}  // namespace enn
