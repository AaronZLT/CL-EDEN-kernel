#pragma once

#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLPadConvert.hpp"
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
#define GEMM1XX_COALESCING_INPUT_4_THREAD_8 32
#define GEMM1XX_COALESCING_INPUT_4_THREAD_12 48
#define GEMM1XX_COALESCING_INPUT_2_THREAD_12 24
#ifdef GEMM1XX_VECTOR_SIZE
#undef  GEMM1XX_VECTOR_SIZE
#endif
#define GEMM1XX_VECTOR_SIZE 8
#define GEMM1XX_HW_SPLIT_SIZE 9216
// 3072 = 48 * 64 * 3
class CLGEMM1xXConvolution {
  public:
    CLGEMM1xXConvolution(const std::shared_ptr<CLRuntime> runtime,
                         const PrecisionType &precision,
                         const Dim4 &input_dim,
                         const Dim4 &output_dim);

    Status initialize(const Pad4 &padding,
                      const Dim2 &stride,
                      const uint32_t &group_size,
                      const uint32_t &axis,
                      const Dim2 &dilation,
                      const std::shared_ptr<ITensor> weight,
                      const std::shared_ptr<ITensor> bias,
                      const ActivationInfo &activate_info,
                      const bool &weights_as_input = false,
                      const bool &androidNN = false,
                      const bool &isNCHW = true);

    Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output);

  private:
    typedef struct {
        uint32_t num_output_;
        uint32_t pad_right_;  // currently, only padright = padleft and padtop = padbottom is supported
        uint32_t pad_left_;
        uint32_t pad_top_;
        uint32_t pad_bottom_;
        uint32_t kernel_height_;
        uint32_t kernel_width_;
        uint32_t stride_height_;
        uint32_t stride_width_;
        uint32_t group_;
        uint32_t axis_;
        Dim2 dilation_;
        bool force_nd_im2col_;  // currently, force_nd_im2col_ and activation_ is not supported.
        std::shared_ptr<CLTensor> filter_;
        std::shared_ptr<CLTensor> bias_;
    } ConvDescriptor;

    ConvDescriptor conv_descriptor_;
    enum class GEMM1xXKernelType {
        GEMMInit,
        GEMM8x4,
        GEMM4x4,
        GEMM1x1,
        GEMM2x4,
        GEMM2x8,
        GEMM2x4Scalar,
        GEMM2x8Scalar,
        GEMMValhall4x4
    };
    enum class GPUPlatform {
        Unknow,
        Biforst,
        Makalu,
        Valhall,
    };
    GEMM1xXKernelType gemm_kernel_type_;
    GPUPlatform gpu_platform_;
    // generate kernel name
    std::string _kernel_name_;
    inline void generateKernelName(ActivationInfo activation) {
        if (activation.isEnabled()) {
            switch (activation.activation()) {
            case ActivationInfo::ActivationType::RELU: _kernel_name_ = "RELU" + _kernel_name_; break;
            case ActivationInfo::ActivationType::RELU6:
                if (precision_ == PrecisionType::FP16) {
                    _kernel_name_ = "RELU6" + _kernel_name_;
                }
                break;
            default: break;
            }
        }
    }

    bool weights_as_input_;
    bool androidNN_;
    bool isNCHW_;
    std::shared_ptr<CLTensor> weight_nchw_;

    int unaligned_weight_height_;
    int unaligned_weight_width_;

    int align_weight_height_;
    int align_weight_width_;

    int align_input_height_;
    int align_input_width_;

    Status alignWeight();
    Status dilationWeight(const std::shared_ptr<CLTensor> weight);

    Status gemm1xXRun(const std::shared_ptr<CLTensor> input, const cl_mem weightBuffer, std::shared_ptr<CLTensor> output);

    Status gemm4x4Run(const std::shared_ptr<CLTensor> input, const cl_mem weightBuffer, std::shared_ptr<CLTensor> output);
    Status alignWeight1xXGEMM(const cl_mem &src, cl_mem &dst, int srcWidth, int srcHeight, int dstWidth, int computedTop);

    Status alignWeight1xXMakalu(const cl_mem &src, cl_mem &dst, int srcWidth, int srcHeight, int dstWidth, int dstHeight);

    Status
    gemmValhall4x4Run(const std::shared_ptr<CLTensor> input, const cl_mem weightBuffer, std::shared_ptr<CLTensor> output);

    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;

    cl_mem weight_buffer_;
    std::shared_ptr<CLTensor> converted_filter_;

    std::shared_ptr<struct _cl_kernel> align_weight_kernel_;
    std::shared_ptr<struct _cl_kernel> gemm1xx_kernel_;
    int computed_top_number_;
    bool is_makalu_branch_;

    std::shared_ptr<CLTensor> dilation_filter_;

    int top_HW_split_count_;
    int top_HW_split_size_;
    Dim4 input_dim_;
    Dim4 output_dim_;
    Dim4 convert_out_dim_;
    Dim2 kernel_;
    Dim2 stride_;
    Pad4 pad_temp_;
    std::shared_ptr<CLPadConvert> pad_convert_executor_;
    std::shared_ptr<CLTensor> convert_output_;

    ActivationInfo activation_info_;
};  // class CLGEMM1xXConvolution

}  // namespace gpu
}  // namespace ud
}  // namespace enn
