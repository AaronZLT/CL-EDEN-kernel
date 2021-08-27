#pragma once

#include "userdriver/common/operator_interfaces/common/ActivationInfo.hpp"
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"

namespace enn {
namespace ud {
namespace gpu {

class CLDepthwiseDeconvolution {
public:
    explicit CLDepthwiseDeconvolution(const std::shared_ptr<CLRuntime> runtime,
                                      const PrecisionType &precision,
                                      const Dim4 &input_dim,
                                      const Dim4 &output_dim,
                                      const std::shared_ptr<ITensor> filter,
                                      const std::shared_ptr<ITensor> bias,
                                      const Pad4 &padding,
                                      const Dim2 &stride,
                                      const uint32_t &group,
                                      const ActivationInfo &activate_info);
    Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output);
    Status release();

private:
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;
    std::shared_ptr<struct _cl_kernel> kernel_gemm_;
    std::shared_ptr<struct _cl_kernel> kernel_convert_;

    Pad4 pad_;
    Dim2 stride_;
    std::shared_ptr<ITensor> filter_;
    std::shared_ptr<ITensor> bias_;
    uint32_t group_;
    ActivationInfo activation_info_;

    std::shared_ptr<CLTensor> output_convert_buffer_;
};  // class CLDepthwiseDeconvolution

}  // namespace gpu
}  // namespace ud
}  // namespace enn
