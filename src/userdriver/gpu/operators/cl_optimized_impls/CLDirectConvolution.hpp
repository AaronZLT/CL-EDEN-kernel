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
#define DIRECT_TOP_CHANNEL_4 4
#define DIRECT_TOP_CHANNEL_8 8
#define DIRECT_TOP_WIDTH_4 4
#define DIRECT_TOP_WIDTH_8 8

class CLDirectConvolution {
  public:
    CLDirectConvolution(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);

    Status initialize(const Dim4 &input_dim,
                      const Dim4 &output_dim,
                      const Dim4 &weight_dim,
                      const Pad4 &padding,
                      const Dim2 &stride,
                      const Dim2 &dilation,
                      const std::shared_ptr<ITensor> weight,
                      const std::shared_ptr<ITensor> bias,
                      const ActivationInfo &activate_info,
                      const bool &weights_as_input = false,
                      const bool &androidNN = false,
                      const bool &isNCHW = true);

    Status weightConvert();
    Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output);

  private:
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;
    std::shared_ptr<struct _cl_kernel> direct_;
    std::shared_ptr<struct _cl_kernel> direct_merge_;

    Dim4 input_dim_;
    Dim4 weight_dim_;
    Pad4 padding_;
    Dim2 stride_;
    Dim2 dilation_;

    bool weights_as_input_;
    bool androidNN_;
    bool isNCHW_;
    std::shared_ptr<CLTensor> weight_nchw_;
    int computed_top_channel_numbers_;
    int computed_top_height_numbers_;
    int computed_top_width_numbers_;

    std::shared_ptr<CLTensor> bias_;
    std::shared_ptr<CLTensor> weight_;
    std::shared_ptr<CLTensor> aligned_weight_;
    std::shared_ptr<CLTensor> aligned_output_tm_;

    int splite_num_;
    bool is_4x8_7x7_or_9x9_;  // target direct7x7_4x8 and direct9x9_4x8

    ActivationInfo activation_info_;

    std::shared_ptr<CLTensor> dilation_filter_;
    Status dilationWeight(const std::shared_ptr<CLTensor> weight_tensor);
};

}  // namespace gpu
}  // namespace ud
}  // namespace enn
