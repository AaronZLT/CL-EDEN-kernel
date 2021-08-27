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

class CLDepthwiseConvolutionQuantized {
public:
    CLDepthwiseConvolutionQuantized(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);
    Status initialize(const Dim4 &input_dim,
                      const Dim4 &output_dim,
                      const std::shared_ptr<ITensor> filter,
                      const std::shared_ptr<ITensor> bias,
                      const Dim2 &stride,
                      const Pad4 &pad,
                      const uint32_t &depth_multiplier,
                      const Dim2 &dilation,
                      const ActivationInfo &activate_info,
                      const bool &weight_as_input,
                      const bool &androidNN = false);
    Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output);
    Status release();

private:
    Status dilation_weight(const std::shared_ptr<CLTensor> weight);
    Status excute_kernels_depends(const cl_mem &input_data,
                                  const cl_mem &filter_data,
                                  const cl_mem &bias_data,
                                  const Dim4 &input_dim,
                                  const Dim4 &output_dim,
                                  const uint32_t &input_h,
                                  const uint32_t &input_w,
                                  const uint32_t &filter_h,
                                  const uint32_t &filter_w,
                                  cl_mem *output_data,
                                  const int32_t &input_offset,
                                  const int32_t &weight_offset,
                                  const int32_t &output_offset,
                                  const int32_t &output_multiplier,
                                  const int32_t &output_shift,
                                  const int32_t &act_min,
                                  const int32_t &act_max);

    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;

    std::shared_ptr<struct _cl_kernel> depthwisekernel_;
    std::shared_ptr<struct _cl_kernel> kernel_pad_;
    std::shared_ptr<struct _cl_kernel> kernel_unequal_;
    std::shared_ptr<CLTensor> filter_;
    std::shared_ptr<CLTensor> dilation_filter_;
    std::shared_ptr<CLTensor> bias_;
    Dim2 stride_;
    Pad4 pad_;
    Dim2 kernel_;
    Dim2 dialation_kernel_;
    uint32_t depth_multiplier_;
    Dim2 dilation_;
    bool weights_as_input_;
    bool androidNN_;
    std::shared_ptr<CLTensor> filter_nchw_;
    bool is_dilation_;

    std::shared_ptr<CLTensor> out_int;

    ActivationInfo activation_info_;
    std::shared_ptr<CLTensor> padbuffer_;
    uint32_t padbuffer_h_ = 0;
    uint32_t padbuffer_w_ = 0;
    uint32_t size_pad_ = 0;
};  // class CLDepthwiseConvolutionQuantized

}  // namespace gpu
}  // namespace ud
}  // namespace enn
