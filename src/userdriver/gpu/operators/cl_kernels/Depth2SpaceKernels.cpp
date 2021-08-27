#include "userdriver/gpu/common/CLKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define DEPTH_TO_SPACE(input, output, block_size, output_batch, input_channel, input_height, input_width) \
    int globalID0 = get_global_id(0); \
    int globalID1 = get_global_id(1); \
    int globalID2 = get_global_id(2); \
    for (int b = 0; b < output_batch; b++) { \
        int output_index = b * get_global_size(0) * get_global_size(1) * get_global_size(2) + \
                            globalID0 * get_global_size(1) * get_global_size(2) + \
                            globalID1 * get_global_size(2) + globalID2; \
        int input_channel_idx = globalID0 + \
                                (globalID1 % block_size * block_size + globalID2 % block_size) * \
                                get_global_size(0); \
        int input_height_idx = globalID1 / block_size; \
        int input_width_idx = globalID2 / block_size; \
        int input_index = b * input_channel * input_height * input_width + \
                            input_channel_idx * input_height * input_width + \
                            input_height_idx * input_width + input_width_idx; \
        output[output_index] = input[input_index]; \
    }

#define DEPTH_TO_SPACE_OPT_VLOAD(input, output, block_size, output_batch, output_channel, output_height, \
                                 output_width, input_width) \
    int globalID0 = get_global_id(0) * 8; \
    int globalID1 = get_global_id(1); \
    int globalID2 = get_global_id(2); \
    if (globalID0 >= input_width) { \
        return; \
    } \
    if (globalID0 + 8 <= input_width) {  \
        for (int b = 0; b < output_batch; ++b) { \
            int input_index = \
                ((b * get_global_size(2) + globalID2) * get_global_size(1) + globalID1) * input_width + globalID0; \
            int output_channel_idx = globalID2 % output_channel; \
            int output_height_idx = globalID2 / (output_channel * block_size) + globalID1 * block_size; \
            int output_width_idx = globalID2 / output_channel % block_size + globalID0 * block_size; \
            int output_index = \
                ((b * output_channel + output_channel_idx) * output_height + output_height_idx) * output_width + \
                output_width_idx; \
            __global DATA_T *ptr = output + output_index; \
            DATA_T8 v0 = vload8(0, input + input_index); \
            *ptr = v0.s0; \
            ptr += 2; \
            *ptr = v0.s1; \
            ptr += 2; \
            *ptr = v0.s2; \
            ptr += 2; \
            *ptr = v0.s3; \
            ptr += 2; \
            *ptr = v0.s4; \
            ptr += 2; \
            *ptr = v0.s5; \
            ptr += 2; \
            *ptr = v0.s6; \
            ptr += 2; \
            *ptr = v0.s7; \
        } \
    } else { \
        int num = input_width - globalID0; \
        for (int b = 0; b < output_batch; ++b) { \
            int input_index = \
                ((b * get_global_size(2) + globalID2) * get_global_size(1) + globalID1) * input_width + globalID0; \
            int output_channel_idx = globalID2 % output_channel; \
            int output_height_idx = globalID2 / (output_channel * block_size) + globalID1 * block_size; \
            int output_width_idx = globalID2 / output_channel % block_size + globalID0 * block_size; \
            int output_index = \
                ((b * output_channel + output_channel_idx) * output_height + output_height_idx) * output_width + \
                output_width_idx; \
            __global DATA_T *ptr = output + output_index; \
            if (num >= 4) { \
                DATA_T4 v0 = vload4(0, input + input_index); \
                *ptr = v0.s0; \
                ptr += 2; \
                *ptr = v0.s1; \
                ptr += 2; \
                *ptr = v0.s2; \
                ptr += 2; \
                *ptr = v0.s3; \
                num -= 4; \
                input_index += 4; \
                output_index += 8; \
            } \
            int i = num % 4; \
            ptr = output + output_index; \
            if (i >= 2) { \
                DATA_T2 v0 = vload2(0, input + input_index); \
                *ptr = v0.s0; \
                ptr += 2; \
                *ptr = v0.s1; \
                num -= 2; \
                input_index += 2; \
                output_index += 4; \
            } \
            i = num % 2; \
            ptr = output + output_index; \
            if (1 == i) { \
                *ptr = input[input_index]; \
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

ADD_SINGLE_KERNEL(depth_to_space_INT8, (__global DATA_T *input,
                                      __global DATA_T *output,
                                      unsigned int block_size,
                                      unsigned int output_batch,
                                      unsigned int input_channel,
                                      unsigned int input_height,
                                      unsigned int input_width) {
    DEPTH_TO_SPACE(input, output, block_size, output_batch, input_channel, input_height, input_width)
})

ADD_SINGLE_KERNEL(depth_to_space_opt_vload_INT8, (__global DATA_T *input,
                                                __global DATA_T *output,
                                                unsigned int block_size,
                                                unsigned int output_batch,
                                                unsigned int output_channel,
                                                unsigned int output_height,
                                                unsigned int output_width,
                                                unsigned int input_width) {
    DEPTH_TO_SPACE_OPT_VLOAD(input, output, block_size, output_batch, output_channel, output_height, \
                             output_width, input_width)
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

ADD_SINGLE_KERNEL(SIGNEDdepth_to_space_INT8, (__global DATA_T *input,
                                      __global DATA_T *output,
                                      unsigned int block_size,
                                      unsigned int output_batch,
                                      unsigned int input_channel,
                                      unsigned int input_height,
                                      unsigned int input_width) {
    DEPTH_TO_SPACE(input, output, block_size, output_batch, input_channel, input_height, input_width)
})

#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // char

/********  FP16 KERNELS ********/
#define DATA_T half
#define DATA_T2 half2
#define DATA_T3 half3
#define DATA_T4 half4
#define DATA_T8 half8
#define DATA_T16 half16

ADD_SINGLE_KERNEL(depth_to_space_FP16, (__global DATA_T *input,
                                      __global DATA_T *output,
                                      unsigned int block_size,
                                      unsigned int output_batch,
                                      unsigned int input_channel,
                                      unsigned int input_height,
                                      unsigned int input_width) {
    DEPTH_TO_SPACE(input, output, block_size, output_batch, input_channel, input_height, input_width)
})

ADD_SINGLE_KERNEL(depth_to_space_opt_vload_FP16, (__global DATA_T *input,
                                                __global DATA_T *output,
                                                unsigned int block_size,
                                                unsigned int output_batch,
                                                unsigned int output_channel,
                                                unsigned int output_height,
                                                unsigned int output_width,
                                                unsigned int input_width) {
    DEPTH_TO_SPACE_OPT_VLOAD(input, output, block_size, output_batch, output_channel, output_height, \
                             output_width, input_width)
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

ADD_SINGLE_KERNEL(depth_to_space_FP32, (__global DATA_T *input,
                                      __global DATA_T *output,
                                      unsigned int block_size,
                                      unsigned int output_batch,
                                      unsigned int input_channel,
                                      unsigned int input_height,
                                      unsigned int input_width) {
    DEPTH_TO_SPACE(input, output, block_size, output_batch, input_channel, input_height, input_width)
})

ADD_SINGLE_KERNEL(depth_to_space_opt_vload_FP32, (__global DATA_T *input,
                                                __global DATA_T *output,
                                                unsigned int block_size,
                                                unsigned int output_batch,
                                                unsigned int output_channel,
                                                unsigned int output_height,
                                                unsigned int output_width,
                                                unsigned int input_width) {
    DEPTH_TO_SPACE_OPT_VLOAD(input, output, block_size, output_batch, output_channel, output_height, \
                             output_width, input_width)
})

#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // float

}  // namespace gpu
}  // namespace ud
}  // namespace enn
