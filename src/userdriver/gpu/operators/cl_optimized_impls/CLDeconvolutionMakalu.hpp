#pragma once

#include "userdriver/common/operator_interfaces/common/ActivationInfo.hpp"
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define GEMM1XX_COALESCING_INPUT_4_THREAD_12 48
#define GEMM1XX_SHARING_WEIGHT_8 8
#define GEMM1XX_ALIGN_SIZE_4 4

#define MAX_LOCAL_SIZE 128

class CLDeconvolutionMakalu {
public:
    explicit CLDeconvolutionMakalu(const std::shared_ptr<CLRuntime> runtime,
                                   const PrecisionType &precision,
                                   const Dim4 &input_dim,
                                   const Dim4 &output_dim,
                                   const std::shared_ptr<ITensor> filter,
                                   const std::shared_ptr<ITensor> bias,
                                   const Pad4 &padding,
                                   const Dim2 &stride,
                                   const uint32_t &group,
                                   const ActivationInfo &activate_info,
                                   const bool &weights_as_input = false,
                                   const bool &androidNN = false);
    bool isTwoTimesDeconv();
    Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output);
    Status release();

private:
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;
    std::shared_ptr<struct _cl_kernel> kernel_input_coalesced_trans_;
    std::shared_ptr<struct _cl_kernel> kernel_weight_shared_trans_;
    std::shared_ptr<struct _cl_kernel> kernel_trans_;
    std::shared_ptr<struct _cl_kernel> kernel_gemm_;
    std::shared_ptr<struct _cl_kernel> kernel_convert_;
    Pad4 pad_;
    Dim2 stride_;
    Dim2 kernel_;
    std::shared_ptr<CLTensor> filter_;
    std::shared_ptr<CLTensor> bias_;
    uint32_t group_;
    ActivationInfo activation_info_;
    bool weights_as_input_ = false;
    bool androidNN_ = false;
    std::shared_ptr<CLTensor> weight_nchw_;

    std::shared_ptr<CLTensor> filter_buffer_;
    std::shared_ptr<CLTensor> weight_trans_buffer_;
    std::shared_ptr<CLTensor> input_trans_buffer_;
    std::shared_ptr<CLTensor> output_convert_buffer_;

    uint32_t transposed_filter_width_ = 0;
    uint32_t transposed_filter_height_ = 0;
    uint32_t transformed_filter_width_ = 0;
    uint32_t transformed_filter_height_ = 0;
};  // class CLDeconvolutionMakalu

}  // namespace gpu
}  // namespace ud
}  // namespace enn
