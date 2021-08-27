#pragma once

// #include "operators/cl_optimized_impls/CLMemAlign.hpp"

#include "userdriver/common/operator_interfaces/common/ActivationInfo.hpp"
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/ITensor.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"

namespace enn {
namespace ud {
namespace gpu {

class CL8x1FullyConnected {
  public:
    CL8x1FullyConnected(const std::shared_ptr<CLRuntime> runtime,
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
    std::shared_ptr<struct _cl_kernel> kernel_split_;
    std::shared_ptr<struct _cl_kernel> kernel_merge_;
    std::shared_ptr<struct _cl_kernel> kernel_interleave_;
    int split_number_;
    int split_size_;

    std::shared_ptr<CLTensor> weight_;
    std::shared_ptr<CLTensor> bias_;
    std::shared_ptr<CLTensor> interleave_input_;
    std::shared_ptr<CLTensor> split_output_;
    std::shared_ptr<CLTensor> weight_buffer_;
    ActivationInfo activation_info_;

    Status align_weight(int input_dim_c);
    bool weights_input;
};  // class CL8x1FullyConnected

}  // namespace gpu
}  // namespace ud
}  // namespace enn
