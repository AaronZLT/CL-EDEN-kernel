#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {
#define ELTWISEADDQUANTIZED(inputA,                                                                                         \
                            inputB,                                                                                         \
                            output,                                                                                         \
                            topHW,                                                                                          \
                            input1_offset,                                                                                  \
                            input1_scale,                                                                                   \
                            input2_offset,                                                                                  \
                            input2_scale,                                                                                   \
                            output_offset,                                                                                  \
                            output_scale,                                                                                   \
                            act_min,                                                                                        \
                            act_max)                                                                                        \
    int globalID0 = get_global_id(0);                                                                                       \
    int globalID1 = get_global_id(1) * 8;                                                                                   \
    int index = globalID0 * topHW + globalID1;                                                                              \
    if (globalID1 < topHW) {                                                                                                \
        int8 input1_val = convert_int8(vload8(0, inputA + index));                                                          \
        int8 input2_val = convert_int8(vload8(0, inputB + index));                                                          \
        input1_val = input1_val - (int8)(input1_offset);                                                                    \
        input2_val = input2_val - (int8)(input2_offset);                                                                    \
        const float8 in1f32 = convert_float8(input1_val) * (float8)(input1_scale);                                          \
        const float8 in2f32 = convert_float8(input2_val) * (float8)(input2_scale);                                          \
        float8 qresf32 = (in1f32 + in2f32) / ((float8)output_scale) + ((float8)output_offset);                              \
        qresf32 = min(max(qresf32, act_min), act_max);                                                                      \
        DATA_T8 res = CONVERT_TO_DATA_T8_SAT(convert_int8_rte(qresf32));                                                    \
        if (globalID1 + 8 <= topHW) {                                                                                       \
            vstore8(res, 0, output + index);                                                                                \
        } else {                                                                                                            \
            int num = topHW - globalID1;                                                                                    \
            if (num == 1) {                                                                                                 \
                output[index] = res.s0;                                                                                     \
            } else if (num == 2) {                                                                                          \
                vstore2(res.s01, 0, output + index);                                                                        \
            } else if (num == 3) {                                                                                          \
                vstore3(res.s012, 0, output + index);                                                                       \
            } else if (num == 4) {                                                                                          \
                vstore4(res.s0123, 0, output + index);                                                                      \
            } else if (num == 5) {                                                                                          \
                vstore4(res.s0123, 0, output + index);                                                                      \
                output[index + 4] = res.s4;                                                                                 \
            } else if (num == 6) {                                                                                          \
                vstore4(res.s0123, 0, output + index);                                                                      \
                vstore2(res.s45, 0, output + index + 4);                                                                    \
            } else if (num == 7) {                                                                                          \
                vstore4(res.s0123, 0, output + index);                                                                      \
                vstore3(res.s456, 0, output + index + 4);                                                                   \
            }                                                                                                               \
        }                                                                                                                   \
    }

/********  QUANTIZED KERNELS ********/
#define DATA_T uchar
#define DATA_T2 uchar2
#define DATA_T3 uchar3
#define DATA_T4 uchar4
#define DATA_T8 uchar8
#define DATA_T16 uchar16
#define CONVERT_TO_DATA_T8_SAT(x) convert_uchar8_sat(x)
ADD_SINGLE_KERNEL(eltwiseAddQuantized_INT8,
                  (__global DATA_T * inputA,
                   __global DATA_T *inputB,
                   __global DATA_T *output,
                   int topHW,
                   int input1_offset,
                   float input1_scale,
                   int input2_offset,
                   float input2_scale,
                   int output_offset,
                   float output_scale,
                   int act_min,
                   int act_max){ELTWISEADDQUANTIZED(inputA,
                                                    inputB,
                                                    output,
                                                    topHW,
                                                    input1_offset,
                                                    input1_scale,
                                                    input2_offset,
                                                    input2_scale,
                                                    output_offset,
                                                    output_scale,
                                                    act_min,
                                                    act_max)})
#undef CONVERT_TO_DATA_T8_SAT
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
#define CONVERT_TO_DATA_T8_SAT(x) convert_char8_sat(x)
ADD_SINGLE_KERNEL(SIGNEDeltwiseAddQuantized_INT8,
                  (__global DATA_T * inputA,
                   __global DATA_T *inputB,
                   __global DATA_T *output,
                   int topHW,
                   int input1_offset,
                   float input1_scale,
                   int input2_offset,
                   float input2_scale,
                   int output_offset,
                   float output_scale,
                   int act_min,
                   int act_max){ELTWISEADDQUANTIZED(inputA,
                                                    inputB,
                                                    output,
                                                    topHW,
                                                    input1_offset,
                                                    input1_scale,
                                                    input2_offset,
                                                    input2_scale,
                                                    output_offset,
                                                    output_scale,
                                                    act_min,
                                                    act_max)})
#undef CONVERT_TO_DATA_T8_SAT
#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // char
ADD_SINGLE_KERNEL(eltwiseAddQuantized_opt_INT8,
                  (__global unsigned char *inputA,
                   __global unsigned char *inputB,
                   __global unsigned char *output,
                   int left_shift,
                   int input1_offset,
                   int input2_offset,
                   int output_offset,
                   int input1_shift,
                   int input2_shift,
                   int output_shift,
                   int input1_multiplier,
                   int input2_multiplier,
                   int output_multiplier,
                   int act_min,
                   int act_max) {
                      int globalID0 = get_global_id(0);
                      int globalID1 = get_global_id(1);
                      int index = globalID0 * get_global_size(1) + globalID1;

                      int input1_val = input1_offset + inputA[index];
                      int input2_val = input2_offset + inputB[index];
                      int shifted_input1_val = input1_val * (1 << left_shift);
                      int shifted_input2_val = input2_val * (1 << left_shift);

                      // compute the input1
                      bool overflow1 = (shifted_input1_val == input1_multiplier) && (shifted_input1_val == -2147483648);
                      long shifted_input1_val_64 = shifted_input1_val;
                      long input1_multiplier_64 = input1_multiplier;
                      long input1_64 = shifted_input1_val_64 * input1_multiplier_64;
                      int nudge1 = (input1_64 >= 0) ? (1 << 30) : (1 - (1 << 30));
                      int input1_x2_high32 = (input1_64 + nudge1) / (1ll << 31);
                      int x1 = overflow1 ? 2147483647 : input1_x2_high32;

                      int scaled_input1_val;
                      int mask1;
                      int remainder1;
                      int threshold1;

                      if (-input1_shift >= 0 && -input1_shift <= 31) {
                          mask1 = (int)(((long)1 << -input1_shift) - 1);
                          remainder1 = x1 & mask1;
                          threshold1 = (mask1 >> 1) + (((x1 < 0) ? ~0 : 0) & 1);
                          scaled_input1_val = (x1 >> -input1_shift) + (((remainder1 > threshold1) ? ~0 : 0) & 1);
                      }

                      // compute the input2
                      bool overflow2 = (shifted_input2_val == input2_multiplier) && (shifted_input2_val == -2147483648);
                      long int shifted_input2_val_64 = shifted_input2_val;
                      long int input2_multiplier_64 = input2_multiplier;
                      long int input2_64 = shifted_input2_val_64 * input2_multiplier_64;
                      int nudge2 = (input2_64 >= 0) ? (1 << 30) : (1 - (1 << 30));
                      int input2_x2_high32 = (input2_64 + nudge2) / (1ll << 31);
                      int x2 = overflow2 ? 2147483647 : input2_x2_high32;
                      int scaled_input2_val;

                      if (-input2_shift >= 0 && -input2_shift <= 31) {
                          int mask2 = (int)(((long)1 << -input2_shift) - 1);
                          int remainder2 = x2 & mask2;
                          int threshold2 = (mask2 >> 1) + (((x2 < 0) ? ~0 : 0) & 1);
                          scaled_input2_val = (x2 >> -input2_shift) + (((remainder2 > threshold2) ? ~0 : 0) & 1);
                      }

                      // compute the output
                      int raw_sum = scaled_input1_val + scaled_input2_val;
                      bool overflow_output = (raw_sum == output_multiplier) && (raw_sum == -2147483648);
                      long int shifted_output_val_64 = raw_sum;
                      long int output_multiplier_64 = output_multiplier;
                      long int output_64 = shifted_output_val_64 * output_multiplier_64;
                      int nudge_output = (output_64 >= 0) ? (1 << 30) : (1 - (1 << 30));
                      int output_x2_high32 = (output_64 + nudge_output) / (1ll << 31);
                      int x_output = overflow_output ? 2147483647 : output_x2_high32;
                      int raw_output;

                      if (-output_shift >= 0 && -output_shift <= 31) {
                          int mask_output = (int)(((long)1 << -output_shift) - 1);
                          int remainder_output = x_output & mask_output;
                          int threshold_output = (mask_output >> 1) + (((x_output < 0) ? ~0 : 0) & 1);
                          raw_output = (x_output >> -output_shift) + (((remainder_output > threshold_output) ? ~0 : 0) & 1) +
                                       output_offset;
                      }

                      int temp_max = (act_min >= raw_output) ? act_min : raw_output;
                      output[index] = (act_max <= temp_max) ? act_max : temp_max;
                  }

)

}  // namespace gpu
}  // namespace ud
}  // namespace enn
