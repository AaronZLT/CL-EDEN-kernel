#pragma once

#include "userdriver/common/operator_interfaces/common/Includes.hpp"
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/gpu/operators/gemmlowp/fixedpoint/fixedpoint.h"
#include "userdriver/gpu/operators/gemmlowp/public/gemmlowp.h"

namespace enn {
namespace ud {
namespace gpu {

// Decompose a double multiplier into a Q0.31 int32 representation of its
// significand, and shift representation of NEGATIVE its exponent ---
// this is intended as a RIGHT-shift.
//
// Restricted to the case where the multiplier < 1 (and non-negative).
void QuantizeMultiplierSmallerThanOneExp(double double_multiplier,
                                         int32_t *quantized_multiplier,
                                         int *left_shift);

// Decompose a double multiplier into a Q0.31 int32 representation of its
// significand, and shift representation of its exponent.
//
// Restricted to the case where the multiplier > 1.
void QuantizeMultiplierGreaterThanOne(double double_multiplier,
                                      int32_t *quantized_multiplier,
                                      int *left_shift);

// Decompose a double multiplier into a Q0.31 int32 representation of its
// significand, and shift representation of its exponent.
//
// Handles an arbitrary positive multiplier. The 'shift' output-value is
// basically the 'floating-point exponent' of the multiplier:
// Negative for a right-shift (when the multiplier is <1), positive for a
// left-shift (when the multiplier is >1)
void QuantizeMultiplier(double double_multiplier, int32_t *quantized_multiplier, int *shift);

// This first creates a multiplier in a double equivalent of
// Q(input_integer_bits).(31-input_integer_bits) representation, with extra
// precision in the double's fractional bits.  It then splits the result into
// significand and exponent.
void PreprocessSoftmaxScaling(double beta,
                              double input_scale,
                              int input_integer_bits,
                              int32_t *quantized_multiplier,
                              int *left_shift);

// Like PreprocessSoftmaxScaling, but inverse scaling factors also calculated.
void PreprocessLogSoftmaxScalingExp(double beta,
                                    double input_scale,
                                    int input_integer_bits,
                                    int32_t *quantized_multiplier,
                                    int *left_shift,
                                    int32_t *reverse_scaling_divisor,
                                    int *reverse_scaling_left_shift);
// Calculate the largest input that will result in a within-bounds intermediate
// result within MultiplyByQuantizedMultiplierGreaterThanOne.  In other words,
// it must not overflow before we reduce the value by multiplication by the
// input multiplier.  The negative radius is used as the minimum difference in
// Softmax.
int CalculateInputRadius(int input_integer_bits, int input_left_shift);

// Nudges a min/max quantization range to ensure zero is zero.
// Gymnastics with nudged zero point is to ensure that real zero maps to
// an integer, which is required for e.g. zero-padding in convolutional layers.
// Outputs nudged_min, nudged_max, nudged_scale.
void NudgeQuantizationRange(const float min,
                            const float max,
                            const int quant_min,
                            const int quant_max,
                            float *nudged_min,
                            float *nudged_max,
                            float *scale);

// If x is approximately a power of two (with any positive or negative
// exponent), stores that exponent (i.e. log2(x)) in *log2_result, otherwise
// returns false.
bool CheckedLog2(const float x, int *log2_result);

inline int32_t MultiplyByQuantizedMultiplierGreaterThanOne(int32_t x,
                                                           int32_t quantized_multiplier,
                                                           int left_shift);

int32_t MultiplyByQuantizedMultiplierSmallerThanOneExp(int32_t x,
                                                       int32_t quantized_multiplier,
                                                       int left_shift);

template <typename T> int CountLeadingZeros(T integer_input);

void GetInvSqrtQuantizedMultiplierExp(int32_t input,
                                      int reverse_shift,
                                      int32_t *output_inv_sqrt,
                                      int *output_shift);

void MatrixScalarMultiplyAccumulate(const int8_t* matrix, int32_t scalar, int32_t n_row,
                                    int32_t n_col, int32_t* output);

bool PrecomputeZeroPointTimesWeightWithBias(int32_t zero_point, const int8_t* weight_tensor,
                                            const Dim4& weight_shape, const int32_t* bias_tensor,
                                            std::unique_ptr<int32_t[]>* output);

inline int32_t MultiplyByQuantizedMultiplier(int32_t x, int32_t quantized_multiplier, int shift) {
    using gemmlowp::RoundingDivideByPOT;
    using gemmlowp::SaturatingRoundingDoublingHighMul;
    int left_shift = shift > 0 ? shift : 0;
    int right_shift = shift > 0 ? 0 : -shift;
    return RoundingDivideByPOT(
            SaturatingRoundingDoublingHighMul(x * (1 << left_shift), quantized_multiplier),
            right_shift);
}

template <typename T>
void MatrixBatchVectorMultiplyAccumulate(const int8_t* input, const int32_t* bias,
                                         const int8_t* input_to_gate_weights, int32_t multiplier,
                                         int32_t shift, int32_t n_batch, int32_t n_input,
                                         int32_t n_output, int32_t output_zp, T* output) {
    const int16_t output_max = std::numeric_limits<T>::max();
    const int16_t output_min = std::numeric_limits<T>::min();
    for (int batch = 0; batch < n_batch; ++batch) {
        for (int row = 0; row < n_output; ++row) {
            int32_t acc = bias[row];
            for (int col = 0; col < n_input; ++col) {
                int8_t input_val = input[batch * n_input + col];
                int8_t weights_val = input_to_gate_weights[row * n_input + col];
                acc += input_val * weights_val;
            }
            acc = MultiplyByQuantizedMultiplier(acc, multiplier, shift);
            acc += output_zp;
            acc += output[batch * n_output + row];
            if (acc > output_max) {
                acc = output_max;
            }
            if (acc < output_min) {
                acc = output_min;
            }
            output[batch * n_output + row] = static_cast<T>(acc);
        }
    }
}

void VectorBatchVectorCwiseProductAccumulate(const int16_t* vector, int v_size,
                                             const int16_t* batch_vector, int n_batch,
                                             int32_t multiplier, int shift, int16_t* result);

void ApplyLayerNorm(const int16_t* input, const int16_t* layer_norm_weights, const int32_t* bias,
                    int32_t layer_norm_scale_a, int32_t layer_norm_scale_b, int32_t variance_limit,
                    int n_batch, int n_input, int16_t* output);

void ApplySigmoid(const int16_t* input, int32_t n_batch, int32_t n_input, int16_t* output);

template <int IntegerBits>
void ApplyTanh(const int16_t* input, int32_t n_batch, int32_t n_input, int16_t* output) {
    using FX = gemmlowp::FixedPoint<std::int16_t, IntegerBits>;
    using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;
    for (int batch = 0; batch < n_batch; ++batch) {
        for (int i = 0; i < n_input; ++i) {
            const int index = batch * n_input + i;
            FX tanh_input = FX::FromRaw(input[index]);
            F0 tanh_output = gemmlowp::tanh(tanh_input);
            output[index] = tanh_output.raw();
        }
    }
}

inline void ApplyTanh(int32_t integer_bits, const int16_t* input, int32_t n_batch, int32_t n_input,
                      int16_t* output) {
    assert(integer_bits <= 6);
#define DISPATCH_TANH(i)                               \
    case i:                                            \
        ApplyTanh<i>(input, n_batch, n_input, output); \
        break;
    switch (integer_bits) {
        DISPATCH_TANH(0);
        DISPATCH_TANH(1);
        DISPATCH_TANH(2);
        DISPATCH_TANH(3);
        DISPATCH_TANH(4);
        DISPATCH_TANH(5);
        DISPATCH_TANH(6);
        default:
            return;
    }
#undef DISPATCH_TANH
}

inline void Sub1Vector(const int16_t* vector, int v_size, int16_t* result) {
    static const int16_t kOne = 32767;
    for (int v = 0; v < v_size; v++) {
        *result++ = kOne - *vector++;
    }
}

void CwiseMul(const int16_t* input_1, const int16_t* input_2, int n_batch, int n_input, int shift,
              int16_t* output);
void CwiseMul(const int16_t* input_1, const int16_t* input_2, int32_t multiplier, int32_t shift,
              int32_t n_batch, int32_t n_input, int32_t output_zp, int8_t* output);

void CwiseAdd(const int16_t* input_1, const int16_t* input_2, int n_batch, int n_input,
              int16_t* output);

void CwiseClipping(int16_t* input, const int16_t clipping_value, int32_t n_batch, int32_t n_input);

void CwiseClipping(int8_t* input, const int8_t clipping_value, int32_t n_batch, int32_t n_input);

}  // namespace gpu
}  // namespace ud
}  // namespace enn
