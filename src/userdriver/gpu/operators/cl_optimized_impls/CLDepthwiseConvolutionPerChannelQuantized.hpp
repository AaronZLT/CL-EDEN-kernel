#pragma once

#include "userdriver/common/operator_interfaces/common/ActivationInfo.hpp"
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/cl_quantized_utils/KernelUtil.hpp"
#include "userdriver/gpu/operators/cl_quantized_utils/QuantizationUtil.hpp"

namespace enn {
namespace ud {
namespace gpu {

class CLDepthwiseConvolutionPerChannelQuantized {
public:
    CLDepthwiseConvolutionPerChannelQuantized(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);
    Status initialize(const Dim4 &input_dim,
                      const Dim4 &output_dim,
                      const std::shared_ptr<ITensor> filter,
                      const std::shared_ptr<ITensor> bias,
                      const Dim2 &stride,
                      const Pad4 &pad,
                      const uint32_t &depth_multiplier,
                      const ActivationInfo &activate_info,
                      const std::vector<float> &scales,
                      const bool &weight_as_input = false,
                      const bool &androidNN = false);
    Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output);
    Status release();

private:
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;

    Dim4 input_dim_;
    Dim4 output_dim_;
    Dim2 stride_;
    Pad4 pad_;
    Dim2 kernel_;
    uint32_t depth_multiplier_;
    ActivationInfo activation_info_;

    std::shared_ptr<CLTensor> filter_;
    std::shared_ptr<CLTensor> bias_;
    std::vector<float> scales_;

    std::shared_ptr<CLTensor> output_multiplier_;
    std::shared_ptr<CLTensor> output_shift_;

    std::shared_ptr<int32_t> output_multiplier_data_;
    std::shared_ptr<int32_t> output_shift_data_;

    std::shared_ptr<struct _cl_kernel> kernel_depthwise_conv_per_channel_;
    std::shared_ptr<struct _cl_kernel> kernel_pad_;

    uint32_t pad_buffer_h_ = 0;
    uint32_t pad_buffer_w_ = 0;
    uint32_t pad_size_ = 0;
    std::shared_ptr<CLTensor> pad_buffer_;

    bool weights_as_input_;
    bool androidNN_;
    std::shared_ptr<CLTensor> weight_nchw_;
};  // class CLDepthwiseConvolutionPerChannelQuantized

}  // namespace gpu
}  // namespace ud
}  // namespace enn
