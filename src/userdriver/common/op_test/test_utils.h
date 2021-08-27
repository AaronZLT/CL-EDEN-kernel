#pragma once

#ifndef SRC_USERDRIVER_COMMON_OP_TEST_H_
#define SRC_USERDRIVER_COMMON_OP_TEST_H_

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <utility>
#include <vector>
#include <limits>

#include "userdriver/common/operator_interfaces/common/Common.hpp"
const float ERROR_THRESHOLD = 1e-5;

/**
 * Nice to have, ToDo(empire.jung, TBD): This file is cloned from EDNE's. Cleaning and optimizing codes later.
 *
 * http://10.166.101.43:81/#/c/634757/14/src/userdriver/common/op_test/test_utils.h (comment by hjkim)
 */

template<typename T1, typename T2>
inline void Compare(const T1 *input1, const T2 *input2, size_t size, float error_threshold_) {
    for (size_t i = 0; i < size; i++) {
        EXPECT_NEAR(input1[i], input2[i], error_threshold_) << i;
    }
}

inline float RelativeError(float reference, float actual) {
    return std::abs(reference - actual) / std::max(FLT_MIN, std::abs(reference));
}

inline float Median(std::vector<float> &array) {
    std::nth_element(array.begin(), array.begin() + array.size() / 2, array.end());
    return array[array.size() / 2];
}

template <typename T>
inline bool GenerateRandom(T *ptr, const uint32_t &size, const T &min, const T &max) {
    std::random_device rd;
    std::mt19937 gen(rd());

    if ((typeid(T) == typeid(float)) || (typeid(T) == typeid(_Float16_t))) {
        std::uniform_real_distribution<> dis(min, max);
        for (uint32_t idx = 0; idx < size; idx++)
            *(ptr + idx) = dis(gen);
    } else {
        std::uniform_int_distribution<> dis(min, max);
        for (uint32_t idx = 0; idx < size; idx++)
            *(ptr + idx) = dis(gen);
    }
    return true;
}

inline uint32_t GetDimSize(const Dim2& dims) {
    return dims.h * dims.w;
}

inline uint32_t GetDimSize(const Dim4& dims) {
    return dims.n * dims.c * dims.h * dims.w;
}

inline uint32_t GetDimSize(const NDims &dims) {
    return dims.size() ? std::accumulate(dims.begin(), dims.end(), 1u, std::multiplies<uint32_t>()) : 0;
}

inline std::int32_t SaturatingRoundingDoublingHighMul(std::int32_t a, std::int32_t b) {
    bool overflow = a == b && a == std::numeric_limits<std::int32_t>::min();
    std::int64_t a_64(a);
    std::int64_t b_64(b);
    std::int64_t ab_64 = a_64 * b_64;
    std::int32_t nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));
    auto ab_x2_high32 = static_cast<std::int32_t>((ab_64 + nudge) / (1ll << 31));
    return overflow ? std::numeric_limits<std::int32_t>::max() : ab_x2_high32;
}

inline std::int32_t RoundingDivideByPOT(std::int32_t x, int exponent) {
    assert(exponent >= 0);
    assert(exponent <= 31);
    const auto mask = static_cast<std::int32_t>((1ll << exponent) - 1);
    const std::int32_t remainder = x & mask;
    const std::int32_t threshold = (mask >> 1) + (((x < 0) ? ~0 : 0) & 1);
    return (x >> exponent) + (((remainder > threshold) ? ~0 : 0) & 1);
}

inline std::int32_t MultiplyByQuantizedMultiplier(int32_t x, int32_t quantized_multiplier, int shift) {
    int left_shift = shift > 0 ? shift : 0;
    int right_shift = shift > 0 ? 0 : -shift;
    return RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(x * (1 << left_shift), quantized_multiplier), right_shift);
}

template <typename T>
std::pair<float, int32_t> QuantizationParams(float f_min, float f_max) {
    // These are required by many quantized operations.
    T q_min = std::numeric_limits<T>::min();
    T q_max = std::numeric_limits<T>::max();
    float range = q_max - q_min;
    float scale = (f_max - f_min) / range;
    int32_t zero_point = std::min(q_max, std::max(q_min, static_cast<T>(round(q_min - f_min / scale))));
    return {scale, zero_point};
}

template <typename T>
inline std::vector<T> Quantize(const std::vector<float> &data, float scale, int32_t zero_point) {
    std::vector<T> q;
    for (float f : data) {
        q.push_back(static_cast<T>(
            std::max<float>(std::numeric_limits<T>::min(),
                            std::min<float>(std::numeric_limits<T>::max(), round(zero_point + (f / scale))))));
    }
    return q;
}

template <typename T>
void PermuteData(const T *input, const Dim4 &input_dims, const DataOrder &typeorder_input, T *output, Dim4 &output_dim) {
    int permuted_axes[4] = {0, 2, 3, 1};  // DataOrder::NCHW

    if (typeorder_input == DataOrder::NHWC) {
        permuted_axes[0] = 0;
        permuted_axes[1] = 3;
        permuted_axes[2] = 1;
        permuted_axes[3] = 2;
    }

    int out_sizes[4];
    for (int k = 0; k < 4; k++) {
        out_sizes[k] = getDim(input_dims, permuted_axes[k]);
    }
    int o[4];
    int i[4];
    for (o[0] = 0; o[0] < out_sizes[0]; o[0]++) {
        i[permuted_axes[0]] = o[0];
        for (o[1] = 0; o[1] < out_sizes[1]; o[1]++) {
            i[permuted_axes[1]] = o[1];
            for (o[2] = 0; o[2] < out_sizes[2]; o[2]++) {
                i[permuted_axes[2]] = o[2];
                for (o[3] = 0; o[3] < out_sizes[3]; o[3]++) {
                    i[permuted_axes[3]] = o[3];
                    int out_id = o[0] * out_sizes[1] * out_sizes[2] * out_sizes[3] + o[1] * out_sizes[2] * out_sizes[3]
                                 + o[2] * out_sizes[3] + o[3];
                    int in_id  = i[0] * input_dims.c * input_dims.h * input_dims.w + i[1] * input_dims.h * input_dims.w
                                 + i[2] * input_dims.w + i[3];
                    output[out_id] = input[in_id];
                }
            }
        }
    }
    output_dim.n = out_sizes[0];
    output_dim.c = out_sizes[1];
    output_dim.h = out_sizes[2];
    output_dim.w = out_sizes[3];
}

template <typename T>
inline std::vector<float> Dequantize(const std::vector<T> &data, float scale, int32_t zero_point) {
    std::vector<float> f;
    for (T q : data) {
        f.push_back(scale * (q - zero_point));
    }
    return f;
}

template <typename T>
int CountLeadingZeros(T integer_input) {
    static_assert(std::is_unsigned<T>::value, "Only unsigned integer types handled.");
    if (integer_input == 0) {
        return std::numeric_limits<T>::digits;
    }
    const T one_in_leading_positive = static_cast<T>(1) << (std::numeric_limits<T>::digits - 1);
    int leading_zeros = 0;
    while (integer_input < one_in_leading_positive) {
        integer_input <<= 1;
        ++leading_zeros;
    }
    return leading_zeros;
}

template <typename T>
void NHWC2NCHW(const T *nhwc, T *nchw, int batch, int channel, int height, int width) {
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channel; c++) {
            for (int h = 0; h < height; h++) {
                for (int w = 0; w < width; w++) {
                    int nchw_idx = b * channel * height * width + c * height * width + h * width + w;
                    int nhwc_idx = b * channel * height * width + h * width * channel + w * channel + c;
                    nchw[nchw_idx] = nhwc[nhwc_idx];
                }
            }
        }
    }
}

template <typename T>
PrecisionType getPrecisionType(const T *in) {
    UNUSED(in);

    PrecisionType precision = PrecisionType::FP32;

    if (typeid(T) == typeid(float)) {
        precision = PrecisionType::FP32;
    } else if (typeid(T) == typeid(int32_t)) {
        precision = PrecisionType::INT32;
    } else if (typeid(T) == typeid(int16_t)) {
        precision = PrecisionType::INT16;
    } else if (typeid(T) == typeid(int8_t)) {
        precision = PrecisionType::INT8;
    } else if (typeid(T) == typeid(uint8_t)) {
        precision = PrecisionType::UINT8;
    } else if (typeid(T) == typeid(_Float16_t)) {
        precision = PrecisionType::FP16;
    } else {
        ERROR_PRINT("illegal Precision type.\n");
    }

    return precision;
}

#endif  // SRC_USERDRIVER_COMMON_OP_TEST_H_
