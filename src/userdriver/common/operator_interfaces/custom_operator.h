#ifndef USERDRIVER_COMMON_OPERATOR_INTERFACES_CUSTOM_OPTIONS_H_
#define USERDRIVER_COMMON_OPERATOR_INTERFACES_CUSTOM_OPTIONS_H_

#include <cstdint>
#include "userdriver/common/operator_interfaces/common/ActivationInfo.hpp"

namespace enn {
namespace ud {

enum CustomOperator {
    BASE = 1000,
    CustomOperator_Normalization,
    CustomOperator_AsymmQuantization,
    CustomOperator_AsymmDequantization,
    CustomOperator_ConvertCFU,
    CustomOperator_InverseCFU,
    CustomOperator_Concat,
    CustomOperator_Quantization,
    CustomOperator_Dequantization,
    CustomOperator_NormalizationQuantization,
    CustomOperator_Pad,
    CustomOperator_Softmax,
    CustomOperator_SigDet,
    SIZE
};

inline const char* const* EnumNamesCustomOperator() {
    static const char* const names[] = {
        "Normalization",
        "AsymmQuantization",
        "AsymmDequantization",
        "ConvertCFU",
        "InverseCFU",
        "Concat",
        "Quantization",
        "Dequantization",
        "NormalizationQuantization",
        "Pad",
        "Softmax",
        "SigDet",
        nullptr
    };
    return names;
};

class TC_SoftmaxOptions {
public:
    float beta_;
    int32_t axis_;

    float beta() {
        return beta_;
    };

    int32_t axis() {
        return axis_;
    }
};

class TC_InverseCFUOptions {
public:
    int32_t cols_in_cell_;
    int32_t lines_in_cell_;
    int32_t interleaved_slices_;

    int32_t cols_in_cell() {
        return cols_in_cell_;
    }
    int32_t lines_in_cell() {
        return lines_in_cell_;
    }
    int32_t interleaved_slices() {
        return interleaved_slices_;
    }
};

class TC_DequantizeOptions {
public:
    uint32_t type_;
    std::vector<int32_t> fractional_length_;
    float scale_out_;
    int32_t zero_point_output_;

    uint32_t type() {
        return type_;
    }
    const std::vector<int32_t>& fractional_length() {
        return fractional_length_;
    }
    float scale_out() {
        return scale_out_;
    }
    int32_t zero_point_output() {
        return zero_point_output_;
    }
};

struct TC_ReluOptions {
    float negative_slope;
};

struct TC_ConcatDOptions {
    int32_t axis;
    ActivationInfo activation_info;
};

struct TC_SubOptions {
    ActivationInfo activation_info = ActivationInfo();
    std::vector<float> coeff = {};
    bool pot_scale_int16 = false;
};

struct TC_MulOptions {
    ActivationInfo activation_info = ActivationInfo();
    std::vector<float> coeff = {};
};

}  // namespace ud
}  // namespace enn

#endif  // USERDRIVER_COMMON_OPERATOR_INTERFACES_CUSTOM_OPTIONS_H_
