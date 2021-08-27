#pragma once

#include "userdriver/common/operator_interfaces/common/ActivationInfo.hpp"
#include "userdriver/common/operator_interfaces/common/Includes.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"

namespace enn {
namespace ud {
namespace gpu {

// Calculates the multiplication factor for a quantized convolution (or
// quantized depthwise convolution) involving the given tensors. Returns an
// error if the scales of the tensors are not compatible.
Status GetQuantizedConvolutionMultipler(double input_scale,
                                        double filter_scale,
                                        double bias_scale,
                                        double output_scale,
                                        double *multiplier);

// Calculates the useful quantized range of an activation layer given its
// activation type.
void CalculateActivationRangeUint8(ActivationInfo::ActivationType activation_type,
                                   float scale,
                                   int32_t zero_point,
                                   int32_t *act_min,
                                   int32_t *act_max);

void CalculateActivationRangeInt8(ActivationInfo::ActivationType activation_type,
                                  float scale,
                                  int32_t zero_point,
                                  int32_t *act_min,
                                  int32_t *act_max);

}  // namespace gpu
}  // namespace ud
}  // namespace enn
