#pragma once

#include "userdriver/common/operator_interfaces/common/ActivationInfo.hpp"
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/gpu/common/CLParameter.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLActivation.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLDepthwiseConvolution3x3TFLite.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLDepthwiseConvolutionPerChannelQuantized.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLDepthwiseConvolutionQuantized.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLDepthwiseConvolutionTFLite.hpp"

namespace enn {
namespace ud {
namespace gpu {

struct DepthwiseConvolutionParameters : public Parameters {
    Pad4 padding = {0, 0, 0, 0};
    Dim2 dilation = {1, 1};
    Dim2 stride = {1, 1};
    uint32_t depth_multiplier = 1;
    bool per_channel_quant = false;
    bool androidNN = false;
    bool isNCHW = true;
    StorageType storage_type = StorageType::BUFFER;
    std::shared_ptr<ActivationInfo> activation_info = std::make_shared<ActivationInfo>();
    std::vector<float> scales;
};

class CLDepthwiseConvolution {
public:
    CLDepthwiseConvolution(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);
    Status initialize(std::vector<std::shared_ptr<ITensor>> inputs,
                      std::vector<std::shared_ptr<ITensor>> outputs,
                      std::shared_ptr<Parameters> parameters);
    Status execute();
    Status release();

private:
    bool Is_depthwise_conv3x3_supported();

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
                                  cl_mem *output_data);

    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;

    std::shared_ptr<struct _cl_kernel> kernel_depthwise_;
    std::shared_ptr<struct _cl_kernel> kernel_pad_;
    std::shared_ptr<struct _cl_kernel> kernel_unequal_;

    std::shared_ptr<CLTensor> input_;
    std::shared_ptr<CLTensor> output_;
    std::shared_ptr<CLTensor> weight_;
    std::shared_ptr<CLTensor> dilation_filter_;
    std::shared_ptr<CLTensor> bias_;
    std::shared_ptr<CLTensor> input_nchw_;
    std::shared_ptr<CLTensor> output_nchw_;
    std::shared_ptr<CLTensor> filter_nchw_;
    std::shared_ptr<CLTensor> empty_bias_buffer_;
    std::shared_ptr<CLTensor> pad_buffer_;

    std::shared_ptr<CLDepthwiseConvolutionQuantized> quantized_dw_ = nullptr;
    std::shared_ptr<CLDepthwiseConvolutionPerChannelQuantized> per_channel_quantized_dw_ = nullptr;
    std::shared_ptr<CLDepthwiseConvolution3x3TFLite> dw_tflite_3x3_ = nullptr;
    std::shared_ptr<CLDepthwiseConvolutionTFLite> dw_tflite_ = nullptr;
    std::shared_ptr<DepthwiseConvolutionParameters> parameters_;

    Dim2 stride_;
    Pad4 pad_;
    Dim2 kernel_;
    Dim2 dialation_kernel_;
    Dim2 dilation_;
    ActivationInfo activation_info_;
    std::shared_ptr<CLActivation> cl_activation_;
    bool empty_bias_ = false;
    uint32_t pad_buffer_h_ = 0;
    uint32_t pad_buffer_w_ = 0;
    uint32_t size_pad_ = 0;
    bool weights_as_input_;

    Status execute_nchw(const std::shared_ptr<CLTensor> input, std::shared_ptr<CLTensor> output);
};  // class CLDepthwiseConvolution

}  // namespace gpu
}  // namespace ud
}  // namespace enn
