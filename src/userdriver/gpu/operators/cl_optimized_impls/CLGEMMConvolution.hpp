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

class CLGEMMConvolution {
  public:
    CLGEMMConvolution(const std::shared_ptr<CLRuntime> runtime,
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
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;

    Dim4 input_dim_;
    Dim4 output_dim_;
    Dim4 convert_out_dim_;
    Dim2 stride_;
    Dim2 kernel_;
    std::shared_ptr<CLPadConvert> pad_convert_executor_;
    std::shared_ptr<CLTensor> convert_output_;
    ActivationInfo activation_info_;

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
        std::shared_ptr<CLTensor> filter_;
        std::shared_ptr<CLTensor> bias_;
    } ConvDescriptor;

    ConvDescriptor conv_descriptor_;
    bool weights_as_input_;
    bool androidNN_;
    bool isNCHW_;
    std::shared_ptr<CLTensor> weight_nchw_;

    std::shared_ptr<CLTensor> converted_filter_;
    std::shared_ptr<struct _cl_kernel> align_weight_kernel_;
    std::shared_ptr<struct _cl_kernel> gemm_kernel_;

    Status GEMMRun(const std::shared_ptr<CLTensor> input, std::shared_ptr<CLTensor> output);

    Status conv2DGPU(const std::shared_ptr<CLTensor> input, std::shared_ptr<CLTensor> output);

    Status alignWeightGEMM(const cl_mem &src,
                           cl_mem &dst,
                           const uint32_t &srcWidth,
                           const uint32_t &srcHeight,
                           const uint32_t &dstWidth,
                           const uint32_t &groupTopChannel,
                           const uint32_t &alignedGroupTopChannel);
    Status alignWeight();
};  // class CLGEMMConvolution

}  // namespace gpu
}  // namespace ud
}  // namespace enn
