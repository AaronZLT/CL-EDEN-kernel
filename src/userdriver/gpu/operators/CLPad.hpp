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

struct PadParameters : public Parameters {
    std::vector<int32_t> padding;
    float pad_value = 0.0f;
    int quant_pad_value = 0;
    bool androidNN = false;
    bool isNCHW = true;
};

class CLPad {
  public:
    CLPad(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);
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
    std::shared_ptr<CLTensor> padding_tensor_;
    std::shared_ptr<CLTensor> pad_value_tensor_;
    std::shared_ptr<CLTensor> output_tensor_;
    std::shared_ptr<PadParameters> parameters_;

    // 3. Operator kernels
    std::shared_ptr<struct _cl_kernel> kernel_;

    // 4. Used for get padding and pad_value
    std::shared_ptr<int32_t> tmp_padding_;
    std::shared_ptr<half_float::half> tmp_buffer_;

    // 5. execute functions
    Status pad_float();
    Status pad_quant();
};

}  // namespace gpu
}  // namespace ud
}  // namespace enn
