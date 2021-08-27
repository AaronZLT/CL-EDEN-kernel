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

class CLGEMMConvolutionQuantized {
  public:
    CLGEMMConvolutionQuantized(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);

    Status initialize(const Dim4 &input_dim,
                      const Dim4 &output_dim,
                      const Pad4 &padding,
                      const Dim2 &stride,
                      const uint32_t &group_size,
                      const std::shared_ptr<ITensor> weight,
                      const std::shared_ptr<ITensor> bias,
                      const ActivationInfo &activate_info,
                      const bool &weights_as_input = false,
                      const bool &androidNN = false);

    Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output);

  private:
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;

    Dim4 input_dim_;
    Dim4 output_dim_;
    Dim4 convert_out_dim_;
    Dim2 stride_;
    Dim2 kernel_;
    Pad4 padding_;
    std::shared_ptr<CLPadConvert> pad_convert_executor_;
    std::shared_ptr<CLTensor> convert_output_;
    ActivationInfo activation_info_;
    uint32_t group_;
    std::shared_ptr<CLTensor> filter_;
    std::shared_ptr<CLTensor> bias_;
    std::shared_ptr<CLTensor> converted_filter_;
    bool weights_as_input_;
    bool androidNN_;
    std::shared_ptr<CLTensor> weight_nchw_;

    int32_t input_offset_;
    int32_t output_multiplier_;
    int output_shift_;

    uint32_t src_unit_count_;
    uint32_t dst_align_count_;
    uint32_t group_top_channel_;
    uint32_t aligned_group_top_channel_;
    int filter_offset_;
    bool signed_quant_;

    std::shared_ptr<struct _cl_kernel> align_quantized_weight_kernel_;
    std::shared_ptr<struct _cl_kernel> quantized_gemm_kernel_;

    Status quantizedGEMMRun(const std::shared_ptr<CLTensor> input, std::shared_ptr<CLTensor> output);

    Status quantizedConv2DGPU(const std::shared_ptr<CLTensor> input, std::shared_ptr<CLTensor> output);

    Status alignQuantizedWeightGEMM(const cl_mem &src,
                                    cl_mem &dst,
                                    const uint32_t &srcWidth,
                                    const uint32_t &srcHeight,
                                    const uint32_t &dstWidth,
                                    const uint32_t &groupTopChannel,
                                    const uint32_t &alignedGroupTopChannel,
                                    int filterOffset);
};  // class CLGEMMConvolutionQuantized

}  // namespace gpu
}  // namespace ud
}  // namespace enn
