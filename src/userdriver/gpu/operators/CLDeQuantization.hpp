#pragma once
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/gpu/common/CLParameter.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"

namespace enn {
namespace ud {
namespace gpu {

struct DeQuantizationParameters : public Parameters {
    bool per_channel_quant = false;
    uint32_t channel_dim = 0;
    std::vector<float> scales;
};

class CLDeQuantization {
public:
    CLDeQuantization(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);
    Status initialize(std::vector<std::shared_ptr<ITensor>> inputs,
                      std::vector<std::shared_ptr<ITensor>> outputs,
                      std::shared_ptr<Parameters> parameters);
    Status execute();
    Status release();

private:
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;

    bool is_inited_;
    std::shared_ptr<struct _cl_kernel> kernel_;
    std::shared_ptr<CLTensor> input_;
    std::shared_ptr<CLTensor> output_;
    std::shared_ptr<DeQuantizationParameters> parameters_;
};  // class CLDeQuantization

}  // namespace gpu
}  // namespace ud
}  // namespace enn
