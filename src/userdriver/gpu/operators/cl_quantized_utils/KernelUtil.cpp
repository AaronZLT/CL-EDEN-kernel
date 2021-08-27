#include "userdriver/gpu/operators/cl_quantized_utils/KernelUtil.hpp"

namespace enn {
namespace ud {
namespace gpu {

Status GetQuantizedConvolutionMultipler(double input_scale,
                                        double filter_scale,
                                        double bias_scale,
                                        double output_scale,
                                        double *multiplier) {
    const double input_product_scale = input_scale * filter_scale;

    if (!(std::abs(input_product_scale - bias_scale) <= 1e-6 * std::min(input_product_scale, bias_scale))) {
        return Status::FAILURE;
    }

    if (!(input_product_scale >= 0)) {
        return Status::FAILURE;
    }

    *multiplier = input_product_scale / output_scale;

    return Status::SUCCESS;
}

void CalculateActivationRangeQuantizedImpl(ActivationInfo::ActivationType activation_type,
                                           int32_t qmin,
                                           int32_t qmax,
                                           float scale,
                                           int32_t zero_point,
                                           int32_t *act_min,
                                           int32_t *act_max) {
    auto quantize = [scale, zero_point](float f) { return zero_point + static_cast<int32_t>(round(f / scale)); };

    if (activation_type == ActivationInfo::ActivationType::RELU) {
        *act_min = std::max(qmin, quantize(0.0));
        *act_max = qmax;
    } else if (activation_type == ActivationInfo::ActivationType::RELU6) {
        *act_min = std::max(qmin, quantize(0.0));
        *act_max = std::min(qmax, quantize(6.0));
    } else if (activation_type == ActivationInfo::ActivationType::RELU1) {
        *act_min = std::max(qmin, quantize(-1.0));
        *act_max = std::min(qmax, quantize(1.0));
    } else {
        *act_min = qmin;
        *act_max = qmax;
    }
}

void CalculateActivationRangeUint8(ActivationInfo::ActivationType activation_type,
                                   float scale,
                                   int32_t zero_point,
                                   int32_t *act_min,
                                   int32_t *act_max) {
    const int32_t qmin = std::numeric_limits<uint8_t>::min();
    const int32_t qmax = std::numeric_limits<uint8_t>::max();

    CalculateActivationRangeQuantizedImpl(activation_type, qmin, qmax, scale, zero_point, act_min, act_max);
}

void CalculateActivationRangeInt8(ActivationInfo::ActivationType activation_type,
                                  float scale,
                                  int32_t zero_point,
                                  int32_t *act_min,
                                  int32_t *act_max) {
    const int32_t qmin = std::numeric_limits<int8_t>::min();
    const int32_t qmax = std::numeric_limits<int8_t>::max();

    CalculateActivationRangeQuantizedImpl(activation_type, qmin, qmax, scale, zero_point, act_min, act_max);
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
