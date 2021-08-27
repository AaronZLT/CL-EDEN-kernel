#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define ELTWISE_MUL_ZERO_ONE(inputA, inputB, output, outputHW) \
        int g0 = get_global_id(0); \
        int g1 = get_global_id(1) * 8; \
        if (g1 < outputHW) { \
            int index = g0 * outputHW + g1; \
            DATA_T8 in_a = vload8(0, inputA + index); \
            DATA_T8 in_b = vload8(0, inputB + index); \
            vstore8(in_a * in_b, 0, output + index); \
        }

#define ELTWISE_MUL_INT(inputA, inputB, output, outputHW) \
        int g0 = get_global_id(0); \
        int g1 = get_global_id(1) * 8; \
        if (g1 < outputHW) { \
            int index = g0 * outputHW + g1; \
            int8 in_a = vload8(0, inputA + index); \
            int8 in_b = vload8(0, inputB + index); \
            vstore8(in_a * in_b, 0, output + index); \
        }

#define ELTWISE_MUL_TWO_MORE(input, output) \
        int index = get_global_id(0) * get_global_size(1) + get_global_id(1); \
        output[index] = input[index] * output[index];

#define ELTWISE_MUL_VECTOR_CONSTANT(inputA, inputB, output, outputHW) \
        int g0 = get_global_id(0); \
        int g1 = get_global_id(1) * 8; \
        if (g1 < outputHW) { \
            int index = g0 * outputHW + g1; \
            DATA_T8 in_a = vload8(0, inputA + index); \
            vstore8(in_a * inputB[g0], 0, output + index); \
        }

/********  FP16 KERNELS ********/
#define DATA_T half
#define DATA_T2 half2
#define DATA_T3 half3
#define DATA_T4 half4
#define DATA_T8 half8
#define DATA_T16 half16

ADD_SINGLE_KERNEL(eltwise_mul_zero_one_FP16, (__global DATA_T *inputA,
                                            __global DATA_T *inputB,
                                            __global DATA_T *output,
                                            int outputHW) {
    ELTWISE_MUL_ZERO_ONE(inputA, inputB, output, outputHW)
})

ADD_SINGLE_KERNEL(eltwise_mul_int_FP16, (__global int *inputA,
                                       __global int *inputB,
                                       __global int *output,
                                       int outputHW) {
    ELTWISE_MUL_INT(inputA, inputB, output, outputHW)
})

ADD_SINGLE_KERNEL(eltwise_mul_two_more_FP16, (__global DATA_T *input,
                                            __global DATA_T *output) {
    ELTWISE_MUL_TWO_MORE(input, output)
})

ADD_SINGLE_KERNEL(eltwise_mul_vector_constant_FP16, (__global DATA_T *inputA,
                                                   __global DATA_T *inputB,
                                                   __global DATA_T *output,
                                                   int outputHW) {
    ELTWISE_MUL_VECTOR_CONSTANT(inputA, inputB, output, outputHW)
})

#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // half


/********  FP32 KERNELS ********/
#define DATA_T float
#define DATA_T2 float2
#define DATA_T3 float3
#define DATA_T4 float4
#define DATA_T8 float8
#define DATA_T16 float16

ADD_SINGLE_KERNEL(eltwise_mul_zero_one_FP32, (__global DATA_T *inputA,
                                            __global DATA_T *inputB,
                                            __global DATA_T *output,
                                            int outputHW) {
    ELTWISE_MUL_ZERO_ONE(inputA, inputB, output, outputHW)
})

ADD_SINGLE_KERNEL(eltwise_mul_int_FP32, (__global int *inputA,
                                       __global int *inputB,
                                       __global int *output,
                                       int outputHW) {
    ELTWISE_MUL_INT(inputA, inputB, output, outputHW)
})

ADD_SINGLE_KERNEL(eltwise_mul_two_more_FP32, (__global DATA_T *input,
                                            __global DATA_T *output) {
    ELTWISE_MUL_TWO_MORE(input, output)
})

ADD_SINGLE_KERNEL(eltwise_mul_vector_constant_FP32, (__global DATA_T *inputA,
                                                   __global DATA_T *inputB,
                                                   __global DATA_T *output,
                                                   int outputHW) {
    ELTWISE_MUL_VECTOR_CONSTANT(inputA, inputB, output, outputHW)
})

#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // float

#define Q_ELTWISE_MUL_ZERO_ONE(inputA, inputB, output, input0_offset, input1_offset, output_offset, output_multiplier, \
                             output_shift, min, max) \
        int index = get_global_id(0) * get_global_size(1) + get_global_id(1); \
        const int input1_val = input0_offset + inputA[index]; \
        const int input2_val = input1_offset + inputB[index]; \
        int left_shift = output_shift > 0 ? output_shift : 0; \
        int right_shift = output_shift > 0 ? 0 : -output_shift; \
        const int unclamped_result = output_offset + rounding_DivideByPOT( \
            saturating_RoundingDoublingHighMul(input1_val * input2_val * (1 << left_shift), \
                                               output_multiplier), \
            right_shift); \
        int clamped_output = min > unclamped_result ? min : unclamped_result; \
        clamped_output = max < clamped_output ? max : clamped_output; \
        output[index] = (DATA_T)(clamped_output);

/********  QUANTIZED KERNELS ********/
#define DATA_T uchar
#define DATA_T2 uchar2
#define DATA_T3 uchar3
#define DATA_T4 uchar4
#define DATA_T8 uchar8
#define DATA_T16 uchar16

ADD_KERNEL_HEADER(eltwise_mul_zero_one_INT8,
                  {DEFINE_FUNC_SATURATING_ROUNDING_DOUBLING_HIGH_MUL, DEFINE_FUNC_ROUNDING_DIVIDE_BY_POT})
ADD_SINGLE_KERNEL(eltwise_mul_zero_one_INT8, (__global DATA_T *inputA,
                                            __global DATA_T *inputB,
                                            __global DATA_T *output,
                                            int input0_offset,
                                            int input1_offset,
                                            int output_offset,
                                            int output_multiplier,
                                            int output_shift,
                                            int min,
                                            int max) {
    Q_ELTWISE_MUL_ZERO_ONE(inputA, inputB, output, input0_offset, input1_offset, output_offset, output_multiplier, \
                          output_shift, min, max)
})

#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // uchar


#define DATA_T char
#define DATA_T2 char2
#define DATA_T3 char3
#define DATA_T4 char4
#define DATA_T8 char8
#define DATA_T16 char16

ADD_KERNEL_HEADER(SIGNEDeltwise_mul_zero_one_INT8,
                  {DEFINE_FUNC_SATURATING_ROUNDING_DOUBLING_HIGH_MUL, DEFINE_FUNC_ROUNDING_DIVIDE_BY_POT})
ADD_SINGLE_KERNEL(SIGNEDeltwise_mul_zero_one_INT8, (__global DATA_T *inputA,
                                            __global DATA_T *inputB,
                                            __global DATA_T *output,
                                            int input0_offset,
                                            int input1_offset,
                                            int output_offset,
                                            int output_multiplier,
                                            int output_shift,
                                            int min,
                                            int max) {
    Q_ELTWISE_MUL_ZERO_ONE(inputA, inputB, output, input0_offset, input1_offset, output_offset, output_multiplier, \
                         output_shift, min, max)
})

#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // char

}  // namespace gpu
}  // namespace ud
}  // namespace enn
