#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {


#define CONCAT_AXIS0(input, output, offset) \
        int input_index = get_global_id(0) * get_global_size(1) * get_global_size(2) + \
                          get_global_id(1) * get_global_size(2) + get_global_id(2); \
        int output_index = get_global_id(0) * get_global_size(1) * get_global_size(2) + \
                           get_global_id(1) * get_global_size(2) + get_global_id(2) + offset; \
        output[output_index] = input[input_index];

#define CONCAT_AXIS1(input, output, offset, picOff, bottomSize) \
        const int globalID1 = get_global_id(1) * 8; \
        if (globalID1 < bottomSize) { \
            int inputIndex = get_global_id(0) * bottomSize + globalID1; \
            int outputIndex = globalID1 + offset + get_global_id(0) * picOff; \
            if (globalID1 + 8 <= bottomSize) { \
                vstore8(vload8(0, input + inputIndex), 0, output + outputIndex); \
            } else { \
                const int num = bottomSize - globalID1; \
                if (num == 1) { \
                    output[outputIndex] = input[inputIndex]; \
                } else if (num == 2) { \
                    vstore2(vload2(0, input + inputIndex), 0, output + outputIndex); \
                } else if (num == 3) { \
                    vstore3(vload3(0, input + inputIndex), 0, output + outputIndex); \
                } else if (num == 4) { \
                    vstore4(vload4(0, input + inputIndex), 0, output + outputIndex); \
                } else if (num == 5) { \
                    vstore4(vload4(0, input + inputIndex), 0, output + outputIndex); \
                    output[outputIndex + 4] = input[inputIndex + 4]; \
                } else if (num == 6) { \
                    vstore4(vload4(0, input + inputIndex), 0, output + outputIndex); \
                    vstore2(vload2(0, input + inputIndex + 4), 0, output + outputIndex + 4); \
                } else if (num == 7) { \
                    vstore4(vload4(0, input + inputIndex), 0, output + outputIndex); \
                    vstore3(vload3(0, input + inputIndex + 4), 0, output + outputIndex + 4); \
                } \
            } \
        }

#define CONCAT_AXIS1_SIZE6(input0, input1, input2, input3, input4, input5, output, offset0, offset1, offset2, \
    offset3, offset4, offset5, bottomSize0, bottomSize1, bottomSize2, bottomSize3, bottomSize4, bottomSize5, picOff) \
        if (get_global_id(1) == 0) { \
            const int globalID2 = get_global_id(2) * 8; \
            if (globalID2 < bottomSize0) { \
                int inputIndex = get_global_id(0) * bottomSize0 + globalID2; \
                int outputIndex = globalID2 + offset0 + get_global_id(0) * picOff; \
                if (globalID2 + 8 <= bottomSize0) { \
                    vstore8(vload8(0, input0 + inputIndex), 0, output + outputIndex); \
                } else { \
                    const int num = bottomSize0 - globalID2; \
                    if (num == 1) { \
                        output[outputIndex] = input0[inputIndex]; \
                    } else if (num == 2) { \
                        vstore2(vload2(0, input0 + inputIndex), 0, output + outputIndex); \
                    } else if (num == 3) { \
                        vstore3(vload3(0, input0 + inputIndex), 0, output + outputIndex); \
                    } else if (num == 4) { \
                        vstore4(vload4(0, input0 + inputIndex), 0, output + outputIndex); \
                    } else if (num == 5) { \
                        vstore4(vload4(0, input0 + inputIndex), 0, output + outputIndex); \
                        output[outputIndex + 4] = input0[inputIndex + 4]; \
                    } else if (num == 6) { \
                        vstore4(vload4(0, input0 + inputIndex), 0, output + outputIndex); \
                        vstore2(vload2(0, input0 + inputIndex + 4), 0, output + outputIndex + 4); \
                    } else if (num == 7) { \
                        vstore4(vload4(0, input0 + inputIndex), 0, output + outputIndex); \
                        vstore3(vload3(0, input0 + inputIndex + 4), 0, output + outputIndex + 4); \
                    } \
                } \
            } \
        } else if (get_global_id(1) == 1) { \
            const int globalID2 = get_global_id(2) * 8; \
            if (globalID2 < bottomSize1) { \
                int inputIndex = get_global_id(0) * bottomSize1 + globalID2; \
                int outputIndex = globalID2 + offset1 + get_global_id(0) * picOff; \
                if (globalID2 + 8 <= bottomSize1) { \
                    vstore8(vload8(0, input1 + inputIndex), 0, output + outputIndex); \
                } else { \
                    const int num = bottomSize1 - globalID2; \
                    if (num == 1) { \
                        output[outputIndex] = input1[inputIndex]; \
                    } else if (num == 2) { \
                        vstore2(vload2(0, input1 + inputIndex), 0, output + outputIndex); \
                    } else if (num == 3) { \
                        vstore3(vload3(0, input1 + inputIndex), 0, output + outputIndex); \
                    } else if (num == 4) { \
                        vstore4(vload4(0, input1 + inputIndex), 0, output + outputIndex); \
                    } else if (num == 5) { \
                        vstore4(vload4(0, input1 + inputIndex), 0, output + outputIndex); \
                        output[outputIndex + 4] = input1[inputIndex + 4]; \
                    } else if (num == 6) { \
                        vstore4(vload4(0, input1 + inputIndex), 0, output + outputIndex); \
                        vstore2(vload2(0, input1 + inputIndex + 4), 0, output + outputIndex + 4); \
                    } else if (num == 7) { \
                        vstore4(vload4(0, input1 + inputIndex), 0, output + outputIndex); \
                        vstore3(vload3(0, input1 + inputIndex + 4), 0, output + outputIndex + 4); \
                    } \
                } \
            } \
        } else if (get_global_id(1) == 2) { \
            const int globalID2 = get_global_id(2) * 8; \
            if (globalID2 < bottomSize2) { \
                int inputIndex = get_global_id(0) * bottomSize2 + globalID2; \
                int outputIndex = globalID2 + offset2 + get_global_id(0) * picOff; \
                if (globalID2 + 8 <= bottomSize2) { \
                    vstore8(vload8(0, input2 + inputIndex), 0, output + outputIndex); \
                } else { \
                    const int num = bottomSize2 - globalID2; \
                    if (num == 1) { \
                        output[outputIndex] = input2[inputIndex]; \
                    } else if (num == 2) { \
                        vstore2(vload2(0, input2 + inputIndex), 0, output + outputIndex); \
                    } else if (num == 3) { \
                        vstore3(vload3(0, input2 + inputIndex), 0, output + outputIndex); \
                    } else if (num == 4) { \
                        vstore4(vload4(0, input2 + inputIndex), 0, output + outputIndex); \
                    } else if (num == 5) { \
                        vstore4(vload4(0, input2 + inputIndex), 0, output + outputIndex); \
                        output[outputIndex + 4] = input2[inputIndex + 4]; \
                    } else if (num == 6) { \
                        vstore4(vload4(0, input2 + inputIndex), 0, output + outputIndex); \
                        vstore2(vload2(0, input2 + inputIndex + 4), 0, output + outputIndex + 4); \
                    } else if (num == 7) { \
                        vstore4(vload4(0, input2 + inputIndex), 0, output + outputIndex); \
                        vstore3(vload3(0, input2 + inputIndex + 4), 0, output + outputIndex + 4); \
                    } \
                } \
            } \
        } else if (get_global_id(1) == 3) { \
            const int globalID2 = get_global_id(2) * 8; \
            if (globalID2 < bottomSize3) { \
                int inputIndex = get_global_id(0) * bottomSize3 + globalID2; \
                int outputIndex = globalID2 + offset3 + get_global_id(0) * picOff; \
                if (globalID2 + 8 <= bottomSize3) { \
                    vstore8(vload8(0, input3 + inputIndex), 0, output + outputIndex); \
                } else { \
                    const int num = bottomSize3 - globalID2; \
                    if (num == 1) { \
                        output[outputIndex] = input3[inputIndex]; \
                    } else if (num == 2) { \
                        vstore2(vload2(0, input3 + inputIndex), 0, output + outputIndex); \
                    } else if (num == 3) { \
                        vstore3(vload3(0, input3 + inputIndex), 0, output + outputIndex); \
                    } else if (num == 4) { \
                        vstore4(vload4(0, input3 + inputIndex), 0, output + outputIndex); \
                    } else if (num == 5) { \
                        vstore4(vload4(0, input3 + inputIndex), 0, output + outputIndex); \
                        output[outputIndex + 4] = input3[inputIndex + 4]; \
                    } else if (num == 6) { \
                        vstore4(vload4(0, input3 + inputIndex), 0, output + outputIndex); \
                        vstore2(vload2(0, input3 + inputIndex + 4), 0, output + outputIndex + 4); \
                    } else if (num == 7) { \
                        vstore4(vload4(0, input3 + inputIndex), 0, output + outputIndex); \
                        vstore3(vload3(0, input3 + inputIndex + 4), 0, output + outputIndex + 4); \
                    } \
                } \
            } \
        } else if (get_global_id(1) == 4) { \
            const int globalID2 = get_global_id(2) * 8; \
            if (globalID2 < bottomSize4) { \
                int inputIndex = get_global_id(0) * bottomSize4 + globalID2; \
                int outputIndex = globalID2 + offset4 + get_global_id(0) * picOff; \
                if (globalID2 + 8 <= bottomSize4) { \
                    vstore8(vload8(0, input4 + inputIndex), 0, output + outputIndex); \
                } else { \
                    const int num = bottomSize4 - globalID2; \
                    if (num == 1) { \
                        output[outputIndex] = input4[inputIndex]; \
                    } else if (num == 2) { \
                        vstore2(vload2(0, input4 + inputIndex), 0, output + outputIndex); \
                    } else if (num == 3) { \
                        vstore3(vload3(0, input4 + inputIndex), 0, output + outputIndex); \
                    } else if (num == 4) { \
                        vstore4(vload4(0, input4 + inputIndex), 0, output + outputIndex); \
                    } else if (num == 5) { \
                        vstore4(vload4(0, input4 + inputIndex), 0, output + outputIndex); \
                        output[outputIndex + 4] = input4[inputIndex + 4]; \
                    } else if (num == 6) { \
                        vstore4(vload4(0, input4 + inputIndex), 0, output + outputIndex); \
                        vstore2(vload2(0, input4 + inputIndex + 4), 0, output + outputIndex + 4); \
                    } else if (num == 7) { \
                        vstore4(vload4(0, input4 + inputIndex), 0, output + outputIndex); \
                        vstore3(vload3(0, input4 + inputIndex + 4), 0, output + outputIndex + 4); \
                    } \
                } \
            } \
        } else if (get_global_id(1) == 5) { \
            const int globalID2 = get_global_id(2) * 8; \
            if (globalID2 < bottomSize5) { \
                int inputIndex = get_global_id(0) * bottomSize5 + globalID2; \
                int outputIndex = globalID2 + offset5 + get_global_id(0) * picOff; \
                if (globalID2 + 8 <= bottomSize5) { \
                    vstore8(vload8(0, input5 + inputIndex), 0, output + outputIndex); \
                } else { \
                    const int num = bottomSize5 - globalID2; \
                    if (num == 1) { \
                        output[outputIndex] = input5[inputIndex]; \
                    } else if (num == 2) { \
                        vstore2(vload2(0, input5 + inputIndex), 0, output + outputIndex); \
                    } else if (num == 3) { \
                        vstore3(vload3(0, input5 + inputIndex), 0, output + outputIndex); \
                    } else if (num == 4) { \
                        vstore4(vload4(0, input5 + inputIndex), 0, output + outputIndex); \
                    } else if (num == 5) { \
                        vstore4(vload4(0, input5 + inputIndex), 0, output + outputIndex); \
                        output[outputIndex + 4] = input5[inputIndex + 4]; \
                    } else if (num == 6) { \
                        vstore4(vload4(0, input5 + inputIndex), 0, output + outputIndex); \
                        vstore2(vload2(0, input5 + inputIndex + 4), 0, output + outputIndex + 4); \
                    } else if (num == 7) { \
                        vstore4(vload4(0, input5 + inputIndex), 0, output + outputIndex); \
                        vstore3(vload3(0, input5 + inputIndex + 4), 0, output + outputIndex + 4); \
                    } \
                } \
            } \
        }

#define CONCAT_AXIS2(input, output, bacthOffset, channelOffset, heightOff, bottomWidth) \
        int input_index = get_global_id(0) * get_global_size(1) * get_global_size(2) + \
                          get_global_id(1) * get_global_size(2) + get_global_id(2); \
        int output_index = get_global_id(0) * bacthOffset + get_global_id(1) * channelOffset + \
                           heightOff * bottomWidth + get_global_id(2); \
        output[output_index] = input[input_index];

#define CONCAT_AXIS3(input, output, bacthOffset, channelOffset, heightOffset, widthOffset, bottomWidth) \
        int input_index = get_global_id(0) * get_global_size(1) * get_global_size(2) + \
                          get_global_id(1) * get_global_size(2) + get_global_id(2); \
        int output_index = get_global_id(0) * bacthOffset + get_global_id(1) * channelOffset + \
                           +get_global_id(2) / bottomWidth * heightOffset + \
                           get_global_id(2) % bottomWidth + widthOffset; \
        output[output_index] = input[input_index];

#define CONCAT_AXIS1_TFLITE_FP16(src_data_0, src_data_1, dst_data, src_size_0_depth, src_size_1_depth, dst_size_x, \
                                 dst_size_y, dst_size_z) \
        int X = get_global_id(0); \
        int Y = get_global_id(1); \
        if (X >= dst_size_x || Y >= dst_size_y) \
            return; \
        int Z = 0; \
        for (int i = 0; i < src_size_0_depth; i += 2) { \
            half4 result0 = read_imageh(src_data_0, smp_none, (int2)((X), (Y)*src_size_0_depth + (i))); \
            half4 result1 = read_imageh(src_data_0, smp_none, (int2)((X), (Y)*src_size_0_depth + (i + 1))); \
            int2 dst_adr0 = (int2)((X), (Y)*dst_size_z + (Z)); \
            int2 dst_adr1 = (int2)((X), (Y)*dst_size_z + (Z + 1)); \
            write_imageh(dst_data, (int2)((X), (Y)*dst_size_z + (Z)), result0); \
            write_imageh(dst_data, (int2)((X), (Y)*dst_size_z + (Z + 1)), result1); \
            Z += 2; \
        } \
        for (int i = 0; i < src_size_1_depth; i += 2) { \
            half4 result0 = read_imageh(src_data_1, smp_none, (int2)((X), (Y)*src_size_1_depth + (i))); \
            half4 result1 = read_imageh(src_data_1, smp_none, (int2)((X), (Y)*src_size_1_depth + (i + 1))); \
            int2 dst_adr0 = (int2)((X), (Y)*dst_size_z + (Z)); \
            int2 dst_adr1 = (int2)((X), (Y)*dst_size_z + (Z + 1)); \
            write_imageh(dst_data, (int2)((X), (Y)*dst_size_z + (Z)), result0); \
            write_imageh(dst_data, (int2)((X), (Y)*dst_size_z + (Z + 1)), result1); \
            Z += 2; \
        }

#define DATA_T half

ADD_SINGLE_KERNEL(concat_axis0_FP16, (__global const DATA_T *input,
                                    __global DATA_T *output,
                                    unsigned int offset) {
    CONCAT_AXIS0(input, output, offset)
})

ADD_SINGLE_KERNEL(concat_axis1_FP16, (__global const DATA_T *input,
                                    __global DATA_T *output,
                                    unsigned int offset,
                                    unsigned int picOff,
                                    unsigned int bottomSize) {
    CONCAT_AXIS1(input, output, offset, picOff, bottomSize)
})

ADD_SINGLE_KERNEL(concat_axis1_size6_FP16, (__global const DATA_T *input0,
                                          __global const DATA_T *input1,
                                          __global const DATA_T *input2,
                                          __global const DATA_T *input3,
                                          __global const DATA_T *input4,
                                          __global const DATA_T *input5,
                                          __global DATA_T *output,
                                          unsigned int offset0,
                                          unsigned int offset1,
                                          unsigned int offset2,
                                          unsigned int offset3,
                                          unsigned int offset4,
                                          unsigned int offset5,
                                          unsigned int bottomSize0,
                                          unsigned int bottomSize1,
                                          unsigned int bottomSize2,
                                          unsigned int bottomSize3,
                                          unsigned int bottomSize4,
                                          unsigned int bottomSize5,
                                          unsigned int picOff) {
    CONCAT_AXIS1_SIZE6(input0, input1, input2, input3, input4, input5, output, offset0, offset1, offset2, \
    offset3, offset4, offset5, bottomSize0, bottomSize1, bottomSize2, bottomSize3, bottomSize4, bottomSize5, picOff)
})

ADD_SINGLE_KERNEL(concat_axis2_FP16, (__global const DATA_T *input,
                                      __global DATA_T *output,
                                      unsigned int bacthOffset,
                                      unsigned int channelOffset,
                                      unsigned int heightOff,
                                      unsigned int bottomWidth) {
    CONCAT_AXIS2(input, output, bacthOffset, channelOffset, heightOff, bottomWidth)
})

ADD_SINGLE_KERNEL(concat_axis3_FP16, (__global const DATA_T *input,
                                    __global DATA_T *output,
                                    unsigned int bacthOffset,
                                    unsigned int channelOffset,
                                    unsigned int heightOffset,
                                    unsigned int widthOffset,
                                    unsigned int bottomWidth) {
    CONCAT_AXIS3(input, output, bacthOffset, channelOffset, heightOffset, widthOffset, bottomWidth)
})

ADD_SINGLE_KERNEL(concat_axis1_tflite_FP16, (__read_only image2d_t src_data_0,
                                           __read_only image2d_t src_data_1,
                                           __write_only image2d_t dst_data,
                                           int src_size_0_depth,
                                           int src_size_1_depth,
                                           int dst_size_x,
                                           int dst_size_y,
                                           int dst_size_z) {
    CONCAT_AXIS1_TFLITE_FP16(src_data_0, src_data_1, dst_data, src_size_0_depth, src_size_1_depth, \
                    dst_size_x, dst_size_y, dst_size_z)
})

#undef DATA_T  // half

#define DATA_T float

ADD_SINGLE_KERNEL(concat_axis0_FP32, (__global const DATA_T *input,
                                    __global DATA_T *output,
                                    unsigned int offset) {
    CONCAT_AXIS0(input, output, offset)
})

ADD_SINGLE_KERNEL(concat_axis1_FP32, (__global const DATA_T *input,
                                    __global DATA_T *output,
                                    unsigned int offset,
                                    unsigned int picOff,
                                    unsigned int bottomSize) {
    CONCAT_AXIS1(input, output, offset, picOff, bottomSize)
})

ADD_SINGLE_KERNEL(concat_axis1_size6_FP32, (__global const DATA_T *input0,
                                          __global const DATA_T *input1,
                                          __global const DATA_T *input2,
                                          __global const DATA_T *input3,
                                          __global const DATA_T *input4,
                                          __global const DATA_T *input5,
                                          __global DATA_T *output,
                                          unsigned int offset0,
                                          unsigned int offset1,
                                          unsigned int offset2,
                                          unsigned int offset3,
                                          unsigned int offset4,
                                          unsigned int offset5,
                                          unsigned int bottomSize0,
                                          unsigned int bottomSize1,
                                          unsigned int bottomSize2,
                                          unsigned int bottomSize3,
                                          unsigned int bottomSize4,
                                          unsigned int bottomSize5,
                                          unsigned int picOff) {
    CONCAT_AXIS1_SIZE6(input0, input1, input2, input3, input4, input5, output, offset0, offset1, offset2, \
    offset3, offset4, offset5, bottomSize0, bottomSize1, bottomSize2, bottomSize3, bottomSize4, bottomSize5, picOff)
})

ADD_SINGLE_KERNEL(concat_axis2_FP32, (__global const DATA_T *input,
                                      __global DATA_T *output,
                                      unsigned int bacthOffset,
                                      unsigned int channelOffset,
                                      unsigned int heightOff,
                                      unsigned int bottomWidth) {
    CONCAT_AXIS2(input, output, bacthOffset, channelOffset, heightOff, bottomWidth)
})

ADD_SINGLE_KERNEL(concat_axis3_FP32, (__global const DATA_T *input,
                                    __global DATA_T *output,
                                    unsigned int bacthOffset,
                                    unsigned int channelOffset,
                                    unsigned int heightOffset,
                                    unsigned int widthOffset,
                                    unsigned int bottomWidth) {
    CONCAT_AXIS3(input, output, bacthOffset, channelOffset, heightOffset, widthOffset, bottomWidth)
})

#undef DATA_T  // float

#define Q_CONCAT_AXIS0(input, output, offset, input_scale, input_zeropoint, output_scale, \
                        output_zeropoint, qmin, qmax) \
    int input_index = get_global_id(0) * get_global_size(1) * get_global_size(2) + \
                        get_global_id(1) * get_global_size(2) + get_global_id(2); \
    int output_index = get_global_id(0) * get_global_size(1) * get_global_size(2) + \
                        get_global_id(1) * get_global_size(2) + get_global_id(2) + offset; \
    if (input_scale == output_scale) { \
        output[output_index] = input[input_index]; \
    } else { \
        const float inverse_output_scale = 1.f / output_scale; \
        const float scale = input_scale * inverse_output_scale; \
        const float bias = -input_zeropoint * scale; \
        int value = (int)round((input[input_index] * scale + bias)) + output_zeropoint; \
        value = value > qmin ? value : qmin; \
        value = value < qmax ? value : qmax; \
        output[output_index] = value; \
    }

#define Q_CONCAT_AXIS1(input, output, bacthOffset, channel_size, offset, input_scale, input_zeropoint, output_scale, \
                     output_zeropoint, qmin, qmax) \
    int input_index = get_global_id(0) * get_global_size(1) * get_global_size(2) + \
                        get_global_id(1) * get_global_size(2) + get_global_id(2); \
    int output_index = get_global_id(0) * bacthOffset + get_global_id(1) * channel_size + \
                        get_global_id(2) + offset; \
    if (input_scale == output_scale) { \
        output[output_index] = input[input_index]; \
    } else { \
        const float inverse_output_scale = 1.f / output_scale; \
        const float scale = input_scale * inverse_output_scale; \
        const float bias = -input_zeropoint * scale; \
        int value = (int)round((input[input_index] * scale + bias)) + output_zeropoint; \
        value = value > qmin ? value : qmin; \
        value = value < qmax ? value : qmax; \
        output[output_index] = value; \
    }

#define Q_CONCAT_AXIS2(input, output, bacthOffset, channelOffset, heightOff, bottomWidth, \
                        input_scale, input_zeropoint, output_scale, output_zeropoint, qmin, qmax) \
            int input_index = get_global_id(0) * get_global_size(1) * get_global_size(2) + \
                              get_global_id(1) * get_global_size(2) + get_global_id(2); \
            int output_index = get_global_id(0) * bacthOffset + get_global_id(1) * channelOffset + \
                               heightOff * bottomWidth + get_global_id(2); \
            if (input_scale == output_scale) { \
                output[output_index] = input[input_index]; \
            } else { \
                const float inverse_output_scale = 1.f / output_scale; \
                const float scale = input_scale * inverse_output_scale; \
                const float bias = -input_zeropoint * scale; \
                int value = (int)round((input[input_index] * scale + bias)) + output_zeropoint; \
                value = value > qmin ? value : qmin; \
                value = value < qmax ? value : qmax; \
                output[output_index] = value; \
            }

#define Q_CONCAT_AXIS3(input, output, bacthOffset, channelOffset, WidthOffset, current_width, intput_width, \
                      input_scale, input_zeropoint, output_scale, output_zeropoint, qmin, qmax) \
        int input_index = get_global_id(0) * get_global_size(1) * get_global_size(2) + \
                          get_global_id(1) * get_global_size(2) + get_global_id(2); \
        int output_index = get_global_id(0) * bacthOffset + get_global_id(1) * channelOffset + \
                           (get_global_id(2) / intput_width) * WidthOffset + \
                           get_global_id(2) % intput_width + current_width; \
        if (input_scale == output_scale) { \
            output[output_index] = input[input_index]; \
        } else { \
            const float inverse_output_scale = 1.f / output_scale; \
            const float scale = input_scale * inverse_output_scale; \
            const float bias = -input_zeropoint * scale; \
            int value = (int)round((input[input_index] * scale + bias)) + output_zeropoint; \
            value = value > qmin ? value : qmin; \
            value = value < qmax ? value : qmax; \
            output[output_index] = value; \
        }

/********  QUANTIZED KERNELS ********/
#define DATA_T uchar
#define DATA_T2 uchar2
#define DATA_T3 uchar3
#define DATA_T4 uchar4
#define DATA_T8 uchar8
#define DATA_T16 uchar16

ADD_SINGLE_KERNEL(concat_axis0_INT8, (__global DATA_T *input, __global DATA_T *output,
                                    unsigned int offset, float input_scale,
                                    int input_zeropoint, float output_scale, int output_zeropoint,
                                    int qmin, int qmax) {
    Q_CONCAT_AXIS0(input, output, offset, input_scale, input_zeropoint, output_scale, output_zeropoint, qmin, qmax)
})

ADD_SINGLE_KERNEL(concat_axis1_INT8, (__global DATA_T *input, __global DATA_T *output,
    unsigned int bacthOffset, unsigned int channel_size, unsigned int offset,
    float input_scale, int input_zeropoint, float output_scale, int output_zeropoint,
    int qmin, int qmax) {
    Q_CONCAT_AXIS1(input, output, bacthOffset, channel_size, offset, input_scale, input_zeropoint, output_scale, \
                    output_zeropoint, qmin, qmax)
})

ADD_SINGLE_KERNEL(concat_axis2_INT8, (__global DATA_T *input, __global DATA_T *output,
    unsigned int bacthOffset, unsigned int channelOffset,
    unsigned int heightOff, unsigned int bottomWidth, float input_scale,
    int input_zeropoint, float output_scale, int output_zeropoint, int qmin, int qmax) {
    Q_CONCAT_AXIS2(input, output, bacthOffset, channelOffset, heightOff, bottomWidth, input_scale, input_zeropoint, \
                    output_scale, output_zeropoint, qmin, qmax)
})

ADD_SINGLE_KERNEL(concat_axis3_INT8, (__global DATA_T *input, __global DATA_T *output,
                                    unsigned int bacthOffset,
                                    unsigned int channelOffset,
                                    unsigned int WidthOffset,
                                    unsigned int current_width,
                                    unsigned int intput_width,
                                    float input_scale,
                                    int input_zeropoint,
                                    float output_scale,
                                    int output_zeropoint,
                                    int qmin,
                                    int qmax) {
    Q_CONCAT_AXIS3(input, output, bacthOffset, channelOffset, WidthOffset, current_width, intput_width, input_scale, \
                    input_zeropoint, output_scale, output_zeropoint, qmin, qmax)
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

ADD_SINGLE_KERNEL(SIGNEDconcat_axis0_INT8, (__global DATA_T *input, __global DATA_T *output,
                                    unsigned int offset, float input_scale,
                                    int input_zeropoint, float output_scale, int output_zeropoint,
                                    int qmin, int qmax) {
    Q_CONCAT_AXIS0(input, output, offset, input_scale, input_zeropoint, output_scale, output_zeropoint, qmin, qmax)
})

ADD_SINGLE_KERNEL(SIGNEDconcat_axis1_INT8, (__global DATA_T *input, __global DATA_T *output,
    unsigned int bacthOffset, unsigned int channel_size, unsigned int offset,
    float input_scale, int input_zeropoint, float output_scale, int output_zeropoint,
    int qmin, int qmax) {
    Q_CONCAT_AXIS1(input, output, bacthOffset, channel_size, offset, input_scale, input_zeropoint, output_scale, \
                    output_zeropoint, qmin, qmax)
})

ADD_SINGLE_KERNEL(SIGNEDconcat_axis2_INT8, (__global DATA_T *input, __global DATA_T *output,
    unsigned int bacthOffset, unsigned int channelOffset,
    unsigned int heightOff, unsigned int bottomWidth, float input_scale,
    int input_zeropoint, float output_scale, int output_zeropoint, int qmin, int qmax) {
    Q_CONCAT_AXIS2(input, output, bacthOffset, channelOffset, heightOff, bottomWidth, input_scale, input_zeropoint, \
                    output_scale, output_zeropoint, qmin, qmax)
})

ADD_SINGLE_KERNEL(SIGNEDconcat_axis3_INT8, (__global DATA_T *input, __global DATA_T *output,
                                    unsigned int bacthOffset,
                                    unsigned int channelOffset,
                                    unsigned int WidthOffset,
                                    unsigned int current_width,
                                    unsigned int intput_width,
                                    float input_scale,
                                    int input_zeropoint,
                                    float output_scale,
                                    int output_zeropoint,
                                    int qmin,
                                    int qmax) {
    Q_CONCAT_AXIS3(input, output, bacthOffset, channelOffset, WidthOffset, current_width, intput_width, input_scale, \
                    input_zeropoint, output_scale, output_zeropoint, qmin, qmax)
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
