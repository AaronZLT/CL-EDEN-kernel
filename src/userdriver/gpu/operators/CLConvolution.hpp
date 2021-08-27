#pragma once

#include "userdriver/gpu/common/CLParameter.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/common/operator_interfaces/common/ActivationInfo.hpp"
#include "userdriver/common/operator_interfaces/common/Common.hpp"

#include "userdriver/gpu/operators/cl_optimized_impls/CLConvolutionPerChannelQuantized.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLDilationConvolution.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLDirectConvolution.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLGEMMConvolution.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLGEMMConvolutionQuantized.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLGEMM1xXConvolution.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLGEMM1xXConvolutionQuantized.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLKernel1x1Convolution.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLPowerVRConvolution.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLQuantizedDirectConvolution.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLWINOConvolution.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLWINO2x2_7x7.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLWINO4x4_5x5.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLWINO6x6_3x3.hpp"
#include "userdriver/gpu/operators/CLActivation.hpp"
#include "userdriver/gpu/operators/CLDeQuantization.hpp"
#include "userdriver/gpu/operators/CLQuantization.hpp"

namespace enn {
namespace ud {
namespace gpu {

struct ConvolutionParameters : public Parameters {
    Pad4 padding = {0, 0, 0, 0};
    Dim2 dilation = {1, 1};
    Dim2 stride = {1, 1};
    uint32_t group_size = 1;
    uint32_t axis = 0;
    bool per_channel_quant = false;
    bool androidNN = false;
    bool isNCHW = true;
    StorageType storage_type = StorageType::BUFFER;
    bool openAibWino = false;
    std::shared_ptr<ActivationInfo> activation_info = std::make_shared<ActivationInfo>();
    std::vector<float> scales;
};

class CLConvolution {
public:
    enum class ConvolutionKernelType {
        GEMM,
        GEMM1xX,
        WINO,
        Kernel1x1,
        DIRECT,
        PowerVR,
        WINO6x6_3x3,
        WINO4x4_5x5,
        WINO2x2_7x7,
        DilationConv
    };

    CLConvolution(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);
    Status initialize(std::vector<std::shared_ptr<ITensor>> inputs,
                      std::vector<std::shared_ptr<ITensor>> outputs,
                      std::shared_ptr<Parameters> parameters);
    Status execute();
    Status release();

private:
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;

    std::shared_ptr<CLTensor> input_;
    std::shared_ptr<CLTensor> output_;
    std::shared_ptr<CLTensor> weight_;
    std::shared_ptr<CLTensor> bias_;
    std::shared_ptr<CLTensor> bias_copied_;
    std::shared_ptr<CLTensor> tmp_weight_tensor_;
    std::shared_ptr<CLTensor> input_nchw_;
    std::shared_ptr<CLTensor> output_nchw_;

    Pad4 padding_;
    Dim2 stride_;
    Dim2 kernel_;
    uint32_t group_size_;
    uint32_t axis_;
    Dim2 dilation_;
    ActivationInfo activation_info_;
    std::shared_ptr<CLActivation> cl_activation_;
    bool weights_as_input_;
    bool bias_as_input_;
    bool per_channel_quant_;
    bool androidNN_;
    bool isNCHW_;
    StorageType storage_type_;
    bool openAibWino_;
    std::vector<float> scales_;
    std::shared_ptr<ConvolutionParameters> parameters_;

    ConvolutionKernelType conv_kernel_type_;
    std::shared_ptr<CLGEMMConvolutionQuantized> quantized_gemm_convolution_;
    std::shared_ptr<CLGEMM1xXConvolutionQuantized> quantized_gemm1xX_convolution_;
    std::shared_ptr<CLConvolutionPerChannelQuantized> quantized_per_channel_convolution_;
    std::shared_ptr<CLGEMMConvolution> gemm_convolution_;
    std::shared_ptr<CLWINOConvolution> wino_convolution_;
    std::shared_ptr<CLKernel1x1Convolution> kernel1x1_convolution_;
    std::shared_ptr<CLGEMM1xXConvolution> gemm1xx_convolution_;
    std::shared_ptr<CLDirectConvolution> direct_convolution_;
    std::shared_ptr<CLQuantizedDirectConvolution> quantized_direct_convolution_;
    std::shared_ptr<CLPowerVRConvolution> powervr_convolution_;
    std::shared_ptr<CLWINO6x6_3x3> wino_6x6_3x3_;
    std::shared_ptr<CLWINO4x4_5x5> wino_4x4_5x5_;
    std::shared_ptr<CLWINO2x2_7x7> wino_2x2_7x7_;
    std::shared_ptr<CLDilationConvolution> dilation_convolution_;

    bool isFitKernel1x1(const Dim4 &input_dim, const Dim4 &output_dim);
    bool isFitDirect(const Dim4 &input_dim, const Dim4 &output_dim, const Dim2 &kernel_dim);
    bool isFitDilationOpt(const Dim4 &input_dim,
                          const Dim4 &output_dim,
                          const Dim2 &kernel_dim,
                          const Dim2 &dilation_dim,
                          const Pad4 &pad_dim);
    Status executeNCHW(const std::shared_ptr<CLTensor> input, std::shared_ptr<CLTensor> output);
};  // class CLConvolution

}  // namespace gpu
}  // namespace ud
}  // namespace enn
