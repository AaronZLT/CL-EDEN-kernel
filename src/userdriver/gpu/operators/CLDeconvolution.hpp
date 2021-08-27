#pragma once

#include "userdriver/common/operator_interfaces/common/ActivationInfo.hpp"
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/gpu/common/CLParameter.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLActivation.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLDeconvolution1x8.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLDeconvolutionGeneral.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLDeconvolutionMakalu.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLDeconvolutionPerChannelQuantized.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLDeconvolutionQuantized.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLDepthwiseDeconvolution.hpp"

namespace enn {
namespace ud {
namespace gpu {

struct DeconvolutionParameters : public Parameters {
    Pad4 padding = {0, 0, 0, 0};
    Dim2 stride = {1, 1};
    uint32_t group_size = 1;
    bool weights_as_input = false;
    bool per_channel_quant = false;
    bool androidNN = false;
    bool isNCHW = true;
    bool openAibWino = false;
    std::shared_ptr<ActivationInfo> activation_info = std::make_shared<ActivationInfo>();
    std::vector<float> scales;
};

class CLDeconvolution {
public:
    CLDeconvolution(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);
    Status initialize(std::vector<std::shared_ptr<ITensor>> inputs_tensors,
                      std::vector<std::shared_ptr<ITensor>> outputs_tensors,
                      std::shared_ptr<Parameters> parameters);
    Status execute();
    Status release();

private:
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;

    uint32_t row_weight_;
    uint32_t row_input_;
    ActivationInfo activation_info_;

    std::shared_ptr<CLTensor> input_nchw_;
    std::shared_ptr<CLTensor> output_nchw_;

    std::shared_ptr<CLTensor> input_tensor_;
    std::shared_ptr<CLTensor> output_tensor_;
    std::shared_ptr<CLTensor> weight_tensor_;
    std::shared_ptr<CLTensor> bias_tensor_;

    std::shared_ptr<CLDeconvolutionGeneral> deconvgeneral_;
    std::shared_ptr<CLDeconvolution1x8> deconv1x8_;
    std::shared_ptr<CLDeconvolutionMakalu> deconvmakalu_;
    std::shared_ptr<CLDepthwiseDeconvolution> depthdeconvmakalu_;
    std::shared_ptr<CLDeconvolutionQuantized> quantized_deconv_;
    std::shared_ptr<CLDeconvolutionPerChannelQuantized> per_channel_quantized_deconv_;
    std::shared_ptr<DeconvolutionParameters> parameters_;

    std::shared_ptr<CLActivation> cl_activation_;

    Status execute_nchw(const std::shared_ptr<CLTensor> input, std::shared_ptr<CLTensor> output);
};  // class CLDeconvolution

}  // namespace gpu
}  // namespace ud
}  // namespace enn
