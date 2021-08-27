#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/gpu/common/CLParameter.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"

namespace enn {
namespace ud {
namespace gpu {

struct Depth2SpaceParameters : public Parameters {
    uint32_t block_size = 0;
    bool androidNN = false;
    bool isNCHW = true;
};

class CLDepth2Space {
public:
    CLDepth2Space(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);
    Status initialize(std::vector<std::shared_ptr<ITensor>> input_tensors,
                      std::vector<std::shared_ptr<ITensor>> output_tensors,
                      std::shared_ptr<Parameters> parameters);
    Status execute();
    Status release();

private:
    Status eval(const std::shared_ptr<CLTensor> input_tensor, std::shared_ptr<CLTensor> output_tensor);

    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;
    std::shared_ptr<struct _cl_kernel> kernel_;
    std::shared_ptr<struct _cl_kernel> kernel_opt_;

    std::shared_ptr<CLTensor> input_;
    std::shared_ptr<CLTensor> output_;
    std::shared_ptr<CLTensor> input_nchw_;
    std::shared_ptr<CLTensor> output_nchw_;
    std::shared_ptr<Depth2SpaceParameters> parameters_;

};  // class CLDepth2Space

}  // namespace gpu
}  // namespace ud
}  // namespace enn
