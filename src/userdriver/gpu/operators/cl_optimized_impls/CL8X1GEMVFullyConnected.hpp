#pragma once

#include "userdriver/common/operator_interfaces/common/ActivationInfo.hpp"
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/ITensor.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"

namespace enn {
namespace ud {
namespace gpu {

class CL8X1GEMVFullyConnected {
  public:
    CL8X1GEMVFullyConnected(const std::shared_ptr<CLRuntime> runtime,
                            const PrecisionType &precision,
                            const Dim4 &input_dim,
                            const Dim4 &output_dim,
                            const std::shared_ptr<ITensor> weight,
                            const std::shared_ptr<ITensor> bias,
                            bool weights_as_input = false);
    Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output);

    Status release();

  private:
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;
    std::shared_ptr<struct _cl_kernel> kernel_gemv_;
    std::shared_ptr<struct _cl_kernel> kernel_interleave_;
    std::shared_ptr<CLTensor> weight_;
    std::shared_ptr<CLTensor> bias_;
    std::shared_ptr<CLTensor> interleave_buffer_;
    std::shared_ptr<CLTensor> weight_buffer_;
    Dim4 interleave_dim_;

    Status align_weight(int output_dim_c);
    bool weights_input;
};  // class CL8X1GEMVFullyConnected

}  // namespace gpu
}  // namespace ud
}  // namespace enn
