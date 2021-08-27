#include <assert.h>
#include "userdriver/gpu/operators/cl_quantized_utils/QuantizationUtil.hpp"

namespace enn {
namespace ud {
namespace gpu {

void QuantizeMultiplier(double double_multiplier, int32_t *quantized_multiplier, int *shift) {
    if (double_multiplier == 0.) {
        *quantized_multiplier = 0;
        *shift = 0;
        return;
    }
    const double q = std::frexp(double_multiplier, shift);
    auto q_fixed = static_cast<int64_t>(round(q * (1ll << 31)));
    if (q_fixed == (1ll << 31)) {
        q_fixed /= 2;
        ++*shift;
    }
    *quantized_multiplier = static_cast<int32_t>(q_fixed);
}

void QuantizeMultiplierGreaterThanOne(double double_multiplier,
                                      int32_t *quantized_multiplier,
                                      int *left_shift) {
    QuantizeMultiplier(double_multiplier, quantized_multiplier, left_shift);
}

void QuantizeMultiplierSmallerThanOneExp(double double_multiplier,
                                         int32_t *quantized_multiplier,
                                         int *left_shift) {
    int shift;
    QuantizeMultiplier(double_multiplier, quantized_multiplier, &shift);
    *left_shift = shift;
}

void PreprocessSoftmaxScaling(double beta,
                              double input_scale,
                              int input_integer_bits,
                              int32_t *quantized_multiplier,
                              int *left_shift) {
    const double input_beta_real_multiplier =
        std::min(beta * input_scale * (1 << (31 - input_integer_bits)), (1ll << 31) - 1.0);
    QuantizeMultiplierGreaterThanOne(input_beta_real_multiplier, quantized_multiplier, left_shift);
}

void PreprocessLogSoftmaxScalingExp(double beta,
                                    double input_scale,
                                    int input_integer_bits,
                                    int32_t *quantized_multiplier,
                                    int *left_shift,
                                    int32_t *reverse_scaling_divisor,
                                    int *reverse_scaling_left_shift) {
    PreprocessSoftmaxScaling(
        beta, input_scale, input_integer_bits, quantized_multiplier, left_shift);
    const double real_reverse_scaling_divisor =
        (1 << (31 - *left_shift)) / static_cast<double>(*quantized_multiplier);
    QuantizeMultiplierSmallerThanOneExp(
        real_reverse_scaling_divisor, reverse_scaling_divisor, reverse_scaling_left_shift);
}

int CalculateInputRadius(int input_integer_bits, int input_left_shift) {
    const double max_input_rescaled = 1.0 * ((1 << input_integer_bits) - 1) *
                                      (1ll << (31 - input_integer_bits)) /
                                      (1ll << input_left_shift);
    return static_cast<int>(std::floor(max_input_rescaled));
}

void NudgeQuantizationRange(const float min,
                            const float max,
                            const int quant_min,
                            const int quant_max,
                            float *nudged_min,
                            float *nudged_max,
                            float *scale) {
    const float quant_min_float = static_cast<float>(quant_min);
    const float quant_max_float = static_cast<float>(quant_max);
    *scale = (max - min) / (quant_max_float - quant_min_float);
    const float zero_point_from_min = quant_min_float - min / *scale;
    uint16_t nudged_zero_point;
    if (zero_point_from_min < quant_min_float) {
        nudged_zero_point = static_cast<uint16_t>(quant_min);
    } else if (zero_point_from_min > quant_max_float) {
        nudged_zero_point = static_cast<uint16_t>(quant_max);
    } else {
        nudged_zero_point = static_cast<uint16_t>(round(zero_point_from_min));
    }
    *nudged_min = (quant_min_float - nudged_zero_point) * (*scale);
    *nudged_max = (quant_max_float - nudged_zero_point) * (*scale);
}

bool CheckedLog2(const float x, int *log2_result) {
    const float x_log2 = std::log(x) * (1.0f / std::log(2.0f));
    const float x_log2_rounded = round(x_log2);
    const float x_log2_fracpart = x_log2 - x_log2_rounded;
    *log2_result = static_cast<int>(x_log2_rounded);
    return std::abs(x_log2_fracpart) < 1e-3;
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

int32_t MultiplyByQuantizedMultiplierSmallerThanOneExp(int32_t x,
                                                       int32_t quantized_multiplier,
                                                       int left_shift) {
    return RoundingDivideByPOT(SaturatingRoundingDoublingHighMul(x, quantized_multiplier),
                               -left_shift);
}

inline int32_t MultiplyByQuantizedMultiplierGreaterThanOne(int32_t x,
                                                           int32_t quantized_multiplier,
                                                           int left_shift) {
    return SaturatingRoundingDoublingHighMul(x * (1 << left_shift), quantized_multiplier);
}

template <typename T> int CountLeadingZeros(T integer_input) {
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

void GetInvSqrtQuantizedMultiplierExp(int32_t input,
                                      int reverse_shift,
                                      int32_t *output_inv_sqrt,
                                      int *output_shift) {
    *output_shift = 11;
    while (input >= (1 << 29)) {
        input /= 4;
        ++*output_shift;
    }
    const unsigned max_left_shift_bits = CountLeadingZeros(static_cast<uint32_t>(input)) - 1;
    const unsigned max_left_shift_bit_pairs = max_left_shift_bits / 2;
    const unsigned left_shift_bit_pairs = max_left_shift_bit_pairs - 1;
    *output_shift -= left_shift_bit_pairs;
    input <<= 2 * left_shift_bit_pairs;

    using gemmlowp::FixedPoint;
    using gemmlowp::Rescale;
    using gemmlowp::SaturatingRoundingMultiplyByPOT;
    // Using 3 integer bits gives us enough room for the internal arithmetic in
    // this Newton-Raphson iteration.
    using F3 = FixedPoint<int32_t, 3>;
    using F0 = FixedPoint<int32_t, 0>;
    const F3 fixedpoint_input = F3::FromRaw(input >> 1);
    const F3 fixedpoint_half_input = SaturatingRoundingMultiplyByPOT<-1>(fixedpoint_input);
    const F3 fixedpoint_half_three =
        GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F3, (1 << 28) + (1 << 27), 1.5);
    // Newton-Raphson iteration
    // Naive unoptimized starting guess: x = 1
    F3 x = F3::One();
    // Naive unoptimized number of iterations: 5
    for (int i = 0; i < 5; i++) {
        const F3 x3 = Rescale<3>(x * x * x);
        x = Rescale<3>(fixedpoint_half_three * x - fixedpoint_half_input * x3);
    }
    const F0 fixedpoint_half_sqrt_2 =
        GEMMLOWP_CHECKED_FIXEDPOINT_CONSTANT(F0, 1518500250, std::sqrt(2.) / 2.);
    x = x * fixedpoint_half_sqrt_2;
    *output_inv_sqrt = x.raw();
    if (*output_shift < 0) {
        *output_inv_sqrt <<= -*output_shift;
        *output_shift = 0;
    }
    // Convert right shift (right is positive) to left shift.
    *output_shift *= reverse_shift;
}

void MatrixScalarMultiplyAccumulate(const int8_t* matrix, int32_t scalar, int32_t n_row,
                                    int32_t n_col, int32_t* output) {
    for (int i = 0; i < n_row; ++i) {
        int32_t row_sum = 0;
        for (int j = 0; j < n_col; ++j) {
            row_sum += *matrix++;
        }
        output[i] += row_sum * scalar;
    }
}

bool PrecomputeZeroPointTimesWeightWithBias(int32_t zero_point, const int8_t* weight_tensor,
                                            const Dim4& weight_shape, const int32_t* bias_tensor,
                                            std::unique_ptr<int32_t[]>* output) {
    if (weight_tensor == nullptr) {
        return true;
    }

    // NN_RET_CHECK_EQ(weight_shape.dimensions.size(), 2u);
    const int row = weight_shape.n;  // weight_shape.dimensions[0]
    const int col = weight_shape.c;  // weight_shape.dimensions[1]
    *output = std::make_unique<int32_t[]>(row);
    if (bias_tensor == nullptr) {
        memset(output->get(), 0, row * sizeof(int32_t));
    } else {
        memcpy(output->get(), bias_tensor, row * sizeof(int32_t));
    }
    if (zero_point != 0) {
        MatrixScalarMultiplyAccumulate(weight_tensor, zero_point, row, col, output->get());
    }
    return true;
}

void VectorBatchVectorCwiseProductAccumulate(const int16_t* vector, int v_size,
                                             const int16_t* batch_vector, int n_batch,
                                             int32_t multiplier, int shift, int16_t* result) {
    for (int b = 0; b < n_batch; b++) {
        for (int v = 0; v < v_size; v++) {
            int32_t prod = vector[v] * *batch_vector++;
            prod = MultiplyByQuantizedMultiplier(prod, multiplier, shift);
            int32_t output = prod + *result;
            output = std::max(std::min(32767, output), -32768);
            *result++ = output;
        }
    }
}

void ApplyLayerNorm(const int16_t* input, const int16_t* layer_norm_weights, const int32_t* bias,
                    int32_t layer_norm_scale_a, int32_t layer_norm_scale_b, int32_t variance_limit,
                    int n_batch, int n_input, int16_t* output) {
    static const int kOverflowGuard = 1 << 20;
    for (int i = 0; i < n_batch; ++i) {
        int64_t sum = 0;
        int64_t sum_sq = 0;
        for (int j = 0; j < n_input; ++j) {
            const int32_t index = i * n_input + j;
            int32_t val = static_cast<int32_t>(input[index]);
            sum += val;
            sum_sq += val * val;
        }
        int32_t mean = static_cast<int32_t>(static_cast<int64_t>(sum) * 1024 / n_input);
        // TODO(jianlijianli): Avoids overflow but only works for POT n_input.
        int32_t temp = kOverflowGuard / n_input;
        int64_t variance = sum_sq * temp - static_cast<int64_t>(mean) * static_cast<int64_t>(mean);
        int32_t variance2 = static_cast<int32_t>(variance / kOverflowGuard);
        if (variance2 < 1) {
            variance2 = variance_limit;
        }
        int32_t stddev_inverse_a;
        int stddev_inverse_b;
        GetInvSqrtQuantizedMultiplierExp(variance2, /*reverse_shift*/ -1, &stddev_inverse_a,
                                         &stddev_inverse_b);

        for (int j = 0; j < n_input; ++j) {
            const int32_t index = i * n_input + j;
            int32_t val = static_cast<int32_t>(input[index]);
            int32_t shifted = 1024 * val - mean;
            int32_t rescaled =
                    MultiplyByQuantizedMultiplier(shifted, stddev_inverse_a, stddev_inverse_b);
            // TODO(jianlijianli): Saturate this.
            int64_t val3 = (int64_t)rescaled * layer_norm_weights[j] + bias[j];
            int32_t val4 = static_cast<int32_t>((val3 > 0 ? val3 + 512 : val3 - 512) / 1024);
            int32_t val5 = MultiplyByQuantizedMultiplier(val4, layer_norm_scale_a,
                                                         layer_norm_scale_b + 12);
            val5 = std::min(std::max(INT16_MIN, val5), INT16_MAX);
            output[index] = static_cast<int16_t>(val5);
        }
    }
}

void ApplySigmoid(const int16_t* input, int32_t n_batch, int32_t n_input, int16_t* output) {
    for (int batch = 0; batch < n_batch; ++batch) {
        for (int c = 0; c < n_input; c++) {
            using F3 = gemmlowp::FixedPoint<std::int16_t, 3>;
            using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;
            const int index = batch * n_input + c;
            F3 sigmoid_input = F3::FromRaw(input[index]);
            F0 sigmoid_output = gemmlowp::logistic(sigmoid_input);
            output[index] = sigmoid_output.raw();
        }
    }
}

void CwiseMul(const int16_t* input_1, const int16_t* input_2, int n_batch, int n_input, int shift,
              int16_t* output) {
    for (int batch = 0; batch < n_batch; ++batch) {
        for (int i = 0; i < n_input; ++i) {
            const int index = batch * n_input + i;
            const int16_t a = input_1[index];
            const int16_t b = input_2[index];
            const int32_t value = static_cast<int32_t>(a) * static_cast<int32_t>(b);
            output[index] = static_cast<int16_t>(gemmlowp::RoundingDivideByPOT(value, shift));
        }
    }
}

void CwiseMul(const int16_t* input_1, const int16_t* input_2, int32_t multiplier, int32_t shift,
              int32_t n_batch, int32_t n_input, int32_t output_zp, int8_t* output) {
    for (int batch = 0; batch < n_batch; ++batch) {
        for (int i = 0; i < n_input; ++i) {
            const int index = batch * n_input + i;
            const int16_t a = input_1[index];
            const int16_t b = input_2[index];
            int32_t value = static_cast<int32_t>(a) * static_cast<int32_t>(b);
            value = MultiplyByQuantizedMultiplier(value, multiplier, shift);
            value -= output_zp;
            value = std::min(std::max(-128, value), 127);

            output[index] = static_cast<int8_t>(value);
        }
    }
}

void CwiseAdd(const int16_t* input_1, const int16_t* input_2, int n_batch, int n_input,
              int16_t* output) {
    for (int batch = 0; batch < n_batch; ++batch) {
        for (int i = 0; i < n_input; ++i) {
            const int index = batch * n_input + i;
            int32_t sum = input_1[index] + input_2[index];
            const int32_t sum_clamped = std::min(INT16_MAX, std::max(INT16_MIN, sum));
            output[index] = static_cast<int16_t>(sum_clamped);
        }
    }
}

void CwiseClipping(int16_t* input, const int16_t clipping_value, int32_t n_batch, int32_t n_input) {
    for (int batch = 0; batch < n_batch; ++batch) {
        for (int i = 0; i < n_input; ++i) {
            const int index = batch * n_input + i;
            if (input[index] > clipping_value) {
                input[index] = clipping_value;
            }
            if (input[index] < -clipping_value) {
                input[index] = -clipping_value;
            }
        }
    }
}

void CwiseClipping(int8_t* input, const int8_t clipping_value, int32_t n_batch, int32_t n_input) {
    for (int batch = 0; batch < n_batch; ++batch) {
        for (int i = 0; i < n_input; ++i) {
            const int index = batch * n_input + i;
            if (input[index] > clipping_value) {
                input[index] = clipping_value;
            }
            if (input[index] < -clipping_value) {
                input[index] = -clipping_value;
            }
        }
    }
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
