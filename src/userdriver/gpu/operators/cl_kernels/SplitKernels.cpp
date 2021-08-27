#include "userdriver/gpu/common/CLKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define SPLIT(input, output, copy_size, base_offset, src_offset) \
    int src_idx = base_offset * get_global_id(0) + src_offset; \
    int dst_idx = get_global_id(0) * copy_size; \
    for (int i = 0; i != copy_size; ++i) { \
        output[dst_idx + i] = input[src_idx + i]; \
    } \
    return;

#define SPLIT_2_2D_SLICES(input, output_0, output_1, split_size, copy_size) \
    int global_id_0 = get_global_id(0); \
    int global_id_1 = get_global_id(1) * 8; \
    int input_offset = global_id_0 * copy_size + global_id_1; \
    int output_offset = (global_id_0 % split_size) * copy_size + global_id_1; \
    if (global_id_0 < split_size) { \
        if (global_id_1 + 7 < copy_size) { \
            vstore8(vload8(0, input + input_offset), 0, output_0 + output_offset); \
        } else if (global_id_1 + 6 < copy_size) { \
            vstore4(vload4(0, input + input_offset), 0, output_0 + output_offset); \
            vstore3(vload3(0, input + input_offset + 4), 0, output_0 + output_offset + 4); \
        } else if (global_id_1 + 5 < copy_size) { \
            vstore4(vload4(0, input + input_offset), 0, output_0 + output_offset); \
            vstore2(vload2(0, input + input_offset + 4), 0, output_0 + output_offset + 4); \
        } else if (global_id_1 + 4 < copy_size) { \
            vstore4(vload4(0, input + input_offset), 0, output_0 + output_offset); \
            output_0[output_offset + 4] = input[input_offset + 4]; \
        } else if (global_id_1 + 3 < copy_size) { \
            vstore4(vload4(0, input + input_offset), 0, output_0 + output_offset); \
        } else if (global_id_1 + 2 < copy_size) { \
            vstore3(vload3(0, input + input_offset), 0, output_0 + output_offset); \
        } else if (global_id_1 + 1 < copy_size) { \
            vstore2(vload2(0, input + input_offset), 0, output_0 + output_offset); \
        } else if (global_id_1 < copy_size) { \
            output_0[output_offset] = input[input_offset]; \
        } \
    } else { \
        if (global_id_1 + 7 < copy_size) { \
            vstore8(vload8(0, input + input_offset), 0, output_1 + output_offset); \
        } else if (global_id_1 + 6 < copy_size) { \
            vstore4(vload4(0, input + input_offset), 0, output_1 + output_offset); \
            vstore3(vload3(0, input + input_offset + 4), 0, output_1 + output_offset + 4); \
        } else if (global_id_1 + 5 < copy_size) { \
            vstore4(vload4(0, input + input_offset), 0, output_1 + output_offset); \
            vstore2(vload2(0, input + input_offset + 4), 0, output_1 + output_offset + 4); \
        } else if (global_id_1 + 4 < copy_size) { \
            vstore4(vload4(0, input + input_offset), 0, output_1 + output_offset); \
            output_1[output_offset + 4] = input[input_offset + 4]; \
        } else if (global_id_1 + 3 < copy_size) { \
            vstore4(vload4(0, input + input_offset), 0, output_1 + output_offset); \
        } else if (global_id_1 + 2 < copy_size) { \
            vstore3(vload3(0, input + input_offset), 0, output_1 + output_offset); \
        } else if (global_id_1 + 1 < copy_size) { \
            vstore2(vload2(0, input + input_offset), 0, output_1 + output_offset); \
        } else if (global_id_1 < copy_size) { \
            output_1[output_offset] = input[input_offset]; \
        } \
    }

ADD_SINGLE_KERNEL(split_FP16, (__global const half *input,
                             __global half *output,
                             int copy_size,
                             int base_offset,
                             int src_offset) {
        SPLIT(input, output, copy_size, base_offset, src_offset)
})

ADD_SINGLE_KERNEL(split_int32_FP16, (__global const int *input,
                                   __global int *output,
                                   int copy_size,
                                   int base_offset,
                                   int src_offset) {
        SPLIT(input, output, copy_size, base_offset, src_offset)
})

ADD_SINGLE_KERNEL(split_2_2d_slices_FP16, (__global const half *input,
                                         __global half *output_0,
                                         __global half *output_1,
                                         int split_size,
                                         int copy_size) {
        SPLIT_2_2D_SLICES(input, output_0, output_1, split_size, copy_size)
})

ADD_SINGLE_KERNEL(split_FP32, (__global const float *input,
                             __global float *output,
                             int copy_size,
                             int base_offset,
                             int src_offset) {
        SPLIT(input, output, copy_size, base_offset, src_offset)
})

ADD_SINGLE_KERNEL(split_int32_FP32, (__global const int *input,
                                   __global int *output,
                                   int copy_size,
                                   int base_offset,
                                   int src_offset) {
        SPLIT(input, output, copy_size, base_offset, src_offset)
})

ADD_SINGLE_KERNEL(split_2_2d_slices_FP32, (__global const float *input,
                                         __global float *output_0,
                                         __global float *output_1,
                                         int split_size,
                                         int copy_size) {
        SPLIT_2_2D_SLICES(input, output_0, output_1, split_size, copy_size)
})

#define DATA_T uchar
ADD_SINGLE_KERNEL(split_INT8, (__global const DATA_T *input,
                             __global DATA_T *output,
                             int copy_size,
                             int base_offset,
                             int src_offset) {
    SPLIT(input, output, copy_size, base_offset, src_offset)
})
#undef DATA_T

#define DATA_T char
ADD_SINGLE_KERNEL(SIGNEDsplit_INT8, (__global const DATA_T *input,
                             __global DATA_T *output,
                             int copy_size,
                             int base_offset,
                             int src_offset) {
    SPLIT(input, output, copy_size, base_offset, src_offset)
})
#undef DATA_T

ADD_SINGLE_KERNEL(split_int32_INT8, (__global const int *input,
                                     __global int *output,
                                     int copy_size,
                                     int base_offset,
                                     int src_offset) {
        SPLIT(input, output, copy_size, base_offset, src_offset)
})

ADD_SINGLE_KERNEL(split_2_2d_slices_INT8, (__global const unsigned char *input,
                                           __global unsigned char *output_0,
                                           __global unsigned char *output_1,
                                           int split_size,
                                           int copy_size) {
        SPLIT_2_2D_SLICES(input, output_0, output_1, split_size, copy_size)
})

}  // namespace gpu
}  // namespace ud
}  // namespace enn
