#pragma once

#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/common/operator_interfaces/common/ActivationInfo.hpp"
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/ITensor.hpp"

namespace enn {
namespace ud {
namespace gpu {
#define DILATION_TOP_CHANNEL_4 4
#define DILATION_TOP_WIDTH_8 8

class CLDilationConvolution {
  public:
    CLDilationConvolution(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);

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
    std::shared_ptr<struct _cl_kernel> dilation_conv_kernel_;

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

    ActivationInfo activation_info_;
};

}  // namespace gpu
}  // namespace ud
}  // namespace enn
