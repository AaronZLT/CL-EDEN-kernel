#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define SUB(input1, input2, output) \
        int globalID0 = get_global_id(0); \
        int globalID1 = get_global_id(1); \
        int globalID2 = get_global_id(2); \
        int index = globalID0 * get_global_size(1) * get_global_size(2) + \
                    globalID1 * get_global_size(2) + globalID2; \
        output[index] = input1[index] - input2[index];

#define DATA_T float
ADD_SINGLE_KERNEL(sub_FP32, (__global const DATA_T *input1,
                           __global const DATA_T *input2,
                           __global DATA_T *output) {
    SUB(input1, input2, output)
})
#undef DATA_T

#define DATA_T int
ADD_SINGLE_KERNEL(INT32sub_FP32, (__global const DATA_T *input1,
                           __global const DATA_T *input2,
                           __global DATA_T *output) {
    SUB(input1, input2, output)
})
ADD_SINGLE_KERNEL(INT32sub_FP16, (__global const DATA_T *input1,
                           __global const DATA_T *input2,
                           __global DATA_T *output) {
    SUB(input1, input2, output)
})
#undef DATA_T

#define DATA_T half
ADD_SINGLE_KERNEL(sub_FP16, (__global const DATA_T *input1,
                           __global const DATA_T *input2,
                           __global DATA_T *output) {
    SUB(input1, input2, output)
})
#undef DATA_T

#define Q_SUB(inputA, inputB, output, topHW, input1_offset, input1_scale, input2_offset, input2_scale, \
                output_offset, output_scale, act_min, act_max) \
        int globalID0 = get_global_id(0); \
        int globalID1 = get_global_id(1) * 8; \
        int index = globalID0 * topHW + globalID1; \
        if (globalID1 < topHW) { \
            int8 input1_val = convert_int8(vload8(0, inputA + index)); \
            int8 input2_val = convert_int8(vload8(0, inputB + index)); \
            input1_val = input1_val - (int8)(input1_offset); \
            input2_val = input2_val - (int8)(input2_offset); \
            const float8 in1f32 = convert_float8(input1_val) * (float8)(input1_scale); \
            const float8 in2f32 = convert_float8(input2_val) * (float8)(input2_scale); \
            float8 qresf32 = \
                (in1f32 - in2f32) / ((float8)output_scale) + ((float8)output_offset); \
            qresf32 = min(max(qresf32, act_min), act_max); \
            DATA_T8 res = CONVERT_TO_DATA_T8_SAT(convert_int8_rte(qresf32)); \
            if (globalID1 + 8 <= topHW) { \
                vstore8(res, 0, output + index); \
            } else { \
                int num = topHW - globalID1; \
                if (num == 1) { \
                    output[index] = res.s0; \
                } else if (num == 2) { \
                    vstore2(res.s01, 0, output + index); \
                } else if (num == 3) { \
                    vstore3(res.s012, 0, output + index); \
                } else if (num == 4) { \
                    vstore4(res.s0123, 0, output + index); \
                } else if (num == 5) { \
                    vstore4(res.s0123, 0, output + index); \
                    output[index + 4] = res.s4; \
                } else if (num == 6) { \
                    vstore4(res.s0123, 0, output + index); \
                    vstore2(res.s45, 0, output + index + 4); \
                } else if (num == 7) { \
                    vstore4(res.s0123, 0, output + index); \
                    vstore3(res.s456, 0, output + index + 4); \
                } \
            } \
        }


/********  QUANTIZED KERNELS ********/
#define DATA_T uchar
#define DATA_T2 uchar2
#define DATA_T3 uchar3
#define DATA_T4 uchar4
#define DATA_T8 uchar8
#define DATA_T16 uchar16
#define CONVERT_TO_DATA_T8_SAT(x) convert_uchar8_sat(x)
ADD_SINGLE_KERNEL(sub_INT8, (__global DATA_T *inputA,
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
                          int act_max) {
    Q_SUB(inputA, inputB, output, topHW, input1_offset, input1_scale, input2_offset, input2_scale, output_offset, \
            output_scale, act_min, act_max)
})
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
ADD_SINGLE_KERNEL(SIGNEDsub_INT8, (__global DATA_T *inputA,
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
                          int act_max) {
    Q_SUB(inputA, inputB, output, topHW, input1_offset, input1_scale, input2_offset, input2_scale, \
            output_offset, output_scale, act_min, act_max)
})
#undef CONVERT_TO_DATA_T8_SAT
#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // char

}  // namespace gpu
}  // namespace ud
}  // namespace enn
