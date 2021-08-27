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

class CLDeconvolutionPerChannelQuantized {
public:
    CLDeconvolutionPerChannelQuantized(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);
    Status initialize(const Dim4 &input_dim,
                      const Dim4 &output_dim,
                      const int &input_zeropoint,
                      const std::shared_ptr<ITensor> filter,
                      const std::shared_ptr<ITensor> bias,
                      const std::vector<float> &scales,
                      const Pad4 &padding,
                      const Dim2 &stride,
                      const uint32_t &group,
                      const ActivationInfo &activate_info,
                      const bool &weights_as_input = false,
                      const bool &androidNN = false);
    Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output);
    Status release();

private:
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;
    std::shared_ptr<struct _cl_kernel> kernel_align_;
    std::shared_ptr<struct _cl_kernel> kernel_trans_;
    std::shared_ptr<struct _cl_kernel> kernel_trans_signed_;
    std::shared_ptr<struct _cl_kernel> kernel_gemm_;
    std::shared_ptr<struct _cl_kernel> kernel_convert_pre_channel_;

    Pad4 pad_;
    Dim2 stride_;
    Dim2 kernel_;
    std::shared_ptr<CLTensor> filter_ = nullptr;
    std::shared_ptr<CLTensor> bias_ = nullptr;
    uint32_t group_ = 0;
    ActivationInfo activation_info_;
    bool weights_as_input_ = false;
    bool androidNN_ = false;
    std::shared_ptr<CLTensor> weight_nchw_;

    std::vector<float> scales_;
    std::shared_ptr<CLTensor> filter_buffer_;
    std::shared_ptr<CLTensor> input_trans_buffer_;
    std::shared_ptr<CLTensor> output_convert_buffer_;

    std::shared_ptr<CLTensor> output_multiplier_;
    std::shared_ptr<CLTensor> output_shift_;

    std::shared_ptr<int32_t> output_multiplier_data_;
    std::shared_ptr<int32_t> output_shift_data_;
};  // class CLDeconvolutionPerChannelQuantized

}  // namespace gpu
}  // namespace ud
}  // namespace enn
