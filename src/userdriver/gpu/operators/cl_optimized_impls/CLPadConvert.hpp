#pragma once

#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/common/operator_interfaces/common/ActivationInfo.hpp"
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/ITensor.hpp"

namespace enn {
namespace ud {
namespace gpu {

class CLPadConvert {
  public:
    enum class PadConvertType {
        NOSET,
        GEMM,
        QuantizedGEMM1xXMakalu,
        QuantizedDirect,
        GEMM4x4Makalu,
        GEMM1xX,
        GEMM1xXMakalu,
        GEMM4x4MakaluNHWC1X1
    };
    CLPadConvert(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);

    Status initialize(const Dim4 &input_dim,
                      const Pad4 &pad,
                      const Dim2 &kernel,
                      const Dim2 &stride,
                      const uint32_t &group,
                      const Dim4 &top_dim = {1, 1, 1, 1},
                      const bool need_pad_buffer = true,
                      const PadConvertType &pad_convert_type = PadConvertType::NOSET,
                      const bool use_makalu_2_2 = false);

    Status execute(const std::shared_ptr<CLTensor> input, std::shared_ptr<CLTensor> output);

    Status padRun(const std::shared_ptr<CLTensor> &input, std::shared_ptr<CLTensor> &output, const Pad4 &pad);

    Status img2ColRun(const std::shared_ptr<CLTensor> &input,
                      std::shared_ptr<CLTensor> &output,
                      const std::shared_ptr<CLTensor> &padded_input,
                      const Pad4 &pad,
                      const Dim2 &kernel,
                      const Dim2 &stride,
                      const unsigned int group);
    Status img2ColWithPad1xXRun(const std::shared_ptr<CLTensor> &input,
                                std::shared_ptr<CLTensor> &output,
                                const Pad4 &pad,
                                const Dim2 &kernel,
                                const Dim2 &stride,
                                const unsigned int collapse_height);
    Status img2ColWithPad4x4MakaluRun(const std::shared_ptr<CLTensor> &input,
                                      std::shared_ptr<CLTensor> &output,
                                      const Pad4 &pad,
                                      const Dim2 &kernel,
                                      const Dim2 &stride,
                                      const unsigned int collapse_height,
                                      int top_hw_split_size,
                                      int pos_id);
    Status quantizedPadRun(const std::shared_ptr<CLTensor> &input,
                           std::shared_ptr<CLTensor> &output,
                           const Pad4 &pad,
                           int byte_zero);
    Status quantizedDirectPadRun(const std::shared_ptr<CLTensor> &input,
                                 std::shared_ptr<CLTensor> &output,
                                 const Pad4 &pad,
                                 unsigned char byte_zero);
    Status quantizedImg2ColRun(const std::shared_ptr<CLTensor> &input,
                               std::shared_ptr<CLTensor> &output,
                               const std::shared_ptr<CLTensor> &padded_input,
                               const Pad4 &pad,
                               const Dim2 &kernel,
                               const Dim2 &stride,
                               const unsigned int group,
                               int top_height,
                               int top_width);
    Status quantizedIm2colAlign1xXMakalu(const std::shared_ptr<CLTensor> &input,
                                         std::shared_ptr<CLTensor> &output,
                                         const Pad4 &padding,
                                         int kernel_height,
                                         int kernel_width,
                                         int stride_height,
                                         int stride_width,
                                         int collapse_height,
                                         int byte_zero,
                                         int top_height,
                                         int top_width);

    ~CLPadConvert() = default;

  private:
    PadConvertType pad_convert_type_;
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;
    bool need_pad_flag_;
    Pad4 padding_;
    Dim2 stride_;
    Dim2 kernel_;
    uint32_t group_;
    std::shared_ptr<CLTensor> pad_;
    Dim4 top_dim_;
    std::shared_ptr<struct _cl_kernel> pad_kernel_;
    std::shared_ptr<struct _cl_kernel> convert_blocked_kernel_;
    std::shared_ptr<struct _cl_kernel> convert_optimized_kernel_;
    std::shared_ptr<struct _cl_kernel> gemm1xx_align_convert1x1_kernel_;
    std::shared_ptr<struct _cl_kernel> gemm1xx_align_convert_withpad_Kernel_;
    std::shared_ptr<struct _cl_kernel> gemm1xx_align_convert1x1_makalu_kernel_;
    std::shared_ptr<struct _cl_kernel> gemm1xx_align_convert_withpad_makalu_Kernel_;
    std::shared_ptr<struct _cl_kernel> gemm1xx_align_convert_withpad_makalu_Kernel_5x5_;
    std::shared_ptr<struct _cl_kernel> gemm1xx_align_convert1x1_makalu_2_2_kernel_;
    std::shared_ptr<struct _cl_kernel> gemm1xx_align_convert_withpad_makalu_2_2_Kernel_;
    std::shared_ptr<struct _cl_kernel> quantized_gemm1xX_align_convert_withpad_kernel_;
    std::shared_ptr<struct _cl_kernel> pad_opt_kernel_;
    std::shared_ptr<struct _cl_kernel> pad_direct_aligned_kernel_;
};  // class CLPadConvert

}  // namespace gpu
}  // namespace ud
}  // namespace enn
