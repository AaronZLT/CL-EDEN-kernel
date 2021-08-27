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

#define GEMM1XX_TILE_SIZE_2 2
#define GEMM1XX_TILE_SIZE_4 4
#define GEMM1XX_TILE_SIZE_8 8
#define GEMM1XX_NO_COALESCING 1
#define GEMM1XX_COALESCING_SIZE 24
#ifdef GEMM1XX_VECTOR_SIZE
#undef  GEMM1XX_VECTOR_SIZE
#endif
#define GEMM1XX_VECTOR_SIZE 16
#define GEMM1XX_COALESCING_SIZE_4_LINES_12_THREADS 48
#define GEMM1XX_COALESCING_SIZE_4_LINES_16_THREADS 64
#define GEMM1XX_COALESCING_SIZE_4_LINES_8_THREADS 32

class CLGEMM1xXConvolutionQuantized {
  public:
    CLGEMM1xXConvolutionQuantized(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);

    Status initialize(const std::shared_ptr<ITensor> input,
                      const Dim4 &output_dim,
                      const Pad4 &padding,
                      const Dim2 &stride,
                      const uint32_t &group_size,
                      const uint32_t &axis,
                      const Dim2 &dilation,
                      const std::shared_ptr<ITensor> weight,
                      const std::shared_ptr<ITensor> bias,
                      const ActivationInfo &activate_info,
                      const bool &weights_as_input = false,
                      const bool &bias_as_input = false,
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
    std::shared_ptr<CLTensor> weight_buffer_makalu_;
    std::shared_ptr<CLTensor> dilation_filter_;
    int32_t input_offset_;
    int32_t output_multiplier_;
    int output_shift_;
    int kBlock_;
    int computed_top_number_;
    int coalescing_feature_height_;
    Dim2 dilation_;

    bool weights_as_input_;
    bool androidNN_;
    std::shared_ptr<CLTensor> weight_nchw_;
    bool bias_as_input_;
    uint32_t unaligned_weight_height_;
    uint32_t unaligned_weight_width_;
    uint32_t align_weight_height_makalu_;
    uint32_t align_weight_width_makalu_;
    bool signed_quant_;

    std::shared_ptr<struct _cl_kernel> align_weight1xXmakalu_kernel_;
    std::shared_ptr<struct _cl_kernel> quantized_gemm1xX_kernel_;
    std::shared_ptr<struct _cl_kernel> weight_offset_kernel_;

    Status gemm1xXConv2DGPU(const std::shared_ptr<CLTensor> input, std::shared_ptr<CLTensor> output);
    Status alignWeight1xXMakalu(const cl_mem &src,
                                cl_mem &dst,
                                int src_width,
                                int src_height,
                                int dst_width,
                                int dst_height,
                                int computed_top);
    Status dilationWeight(const std::shared_ptr<CLTensor> weight);

    Status moveWeightOffset2Bias(const std::shared_ptr<CLTensor> weight_tensor,
                                 const std::shared_ptr<CLTensor> input_tensor,
                                 std::shared_ptr<CLTensor> &bias_tensor);
};  // class CLGEMM1xXConvolutionQuantized

}  // namespace gpu
}  // namespace ud
}  // namespace enn
