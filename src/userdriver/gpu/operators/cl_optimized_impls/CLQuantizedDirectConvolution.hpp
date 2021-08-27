#pragma once

#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLPadConvert.hpp"
#include "userdriver/gpu/operators/cl_quantized_utils/KernelUtil.hpp"
#include "userdriver/gpu/operators/cl_quantized_utils/QuantizationUtil.hpp"
#include "userdriver/common/operator_interfaces/common/ActivationInfo.hpp"
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/ITensor.hpp"

namespace enn {
namespace ud {
namespace gpu {
#define DIRECT_TOP_CHANNEL_4 4
#define DIRECT_TOP_CHANNEL_8 8

class CLQuantizedDirectConvolution {
  public:
    CLQuantizedDirectConvolution(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);

    Status initialize(const std::shared_ptr<ITensor> input,
                      const Dim4 &output_dim,
                      const Dim4 &weight_dim,
                      const Pad4 &padding,
                      const Dim2 &stride,
                      const Dim2 &dilation,
                      const std::shared_ptr<ITensor> weight,
                      const std::shared_ptr<ITensor> bias,
                      const ActivationInfo &activate_info,
                      const bool &weights_as_input = false,
                      const bool &bias_as_input = false,
                      const bool &androidNN = false);

    Status weightConvert();
    Status moveWeightOffset2Bias(int inputZeroPoint, int filterZeroPoint);
    Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output);

  private:
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;
    std::shared_ptr<struct _cl_kernel> direct_;
    std::shared_ptr<struct _cl_kernel> align_weight_kernel_;
    std::shared_ptr<struct _cl_kernel> update_bias_kernel_;

    std::shared_ptr<CLPadConvert> pad_convert_executor_;

    Dim4 input_dim_;
    Dim4 weight_dim_;
    Dim4 output_dim_;
    Pad4 padding_;
    Dim2 stride_;
    Dim2 dilation_;
    Dim4 convert_input_dim_;

    // quantized info
    int32_t input_offset_;
    int32_t output_multiplier_;
    int output_shift_;

    bool weights_as_input_;
    bool androidNN_;
    std::shared_ptr<CLTensor> weight_nchw_;
    bool bias_as_input_;
    int computed_top_channel_numbers_;

    std::shared_ptr<CLTensor> bias_;
    std::shared_ptr<CLTensor> weight_;
    std::shared_ptr<CLTensor> aligned_weight_;
    std::shared_ptr<CLTensor> convert_input_;

    ActivationInfo activation_info_;
};

}  // namespace gpu
}  // namespace ud
}  // namespace enn
