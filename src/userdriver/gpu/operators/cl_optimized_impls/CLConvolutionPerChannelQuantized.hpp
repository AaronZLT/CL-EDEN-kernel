#pragma once

#include "userdriver/common/operator_interfaces/common/ActivationInfo.hpp"
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/ITensor.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/cl_quantized_utils/KernelUtil.hpp"
#include "userdriver/gpu/operators/cl_quantized_utils/QuantizationUtil.hpp"


namespace enn {
namespace ud {
namespace gpu {

class CLConvolutionPerChannelQuantized {
  public:
    CLConvolutionPerChannelQuantized(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);

    Status initialize(const Dim4 &input_dim,
                      const Dim4 &output_dim,
                      const Pad4 &padding,
                      const Dim2 &stride,
                      const uint32_t &group_size,
                      const std::shared_ptr<ITensor> weight,
                      const std::shared_ptr<ITensor> bias,
                      const ActivationInfo &activate_info,
                      const std::vector<float> &scales,
                      const bool &androidNN = false);

    Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output);

  private:
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;

    Dim4 input_dim_;
    Dim4 output_dim_;
    Dim2 stride_;
    Dim2 kernel_;
    Pad4 padding_;
    ActivationInfo activation_info_;
    uint32_t group_;
    std::shared_ptr<CLTensor> filter_;
    std::shared_ptr<CLTensor> bias_;
    std::shared_ptr<CLTensor> output_multiplier_;
    std::shared_ptr<CLTensor> output_shift_;

    std::shared_ptr<int> output_multiplier_data_;
    std::shared_ptr<int> output_shift_data_;
    std::vector<float> scales_;
    bool androidNN_;
    std::shared_ptr<CLTensor> weight_nchw_;
    std::shared_ptr<struct _cl_kernel> per_channel_quantized_kernel_;

};  // class CLConvolutionPerChannelQuantized

}  // namespace gpu
}  // namespace ud
}  // namespace enn
