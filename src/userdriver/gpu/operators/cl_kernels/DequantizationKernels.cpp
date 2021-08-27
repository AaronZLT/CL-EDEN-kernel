#include "userdriver/gpu/common/CLKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define DEQUANTIZATION_UINT8(input, output, scale, zero_point, channel, height, width) \
        int globalID0 = get_global_id(0); \
        int globalID1 = get_global_id(1); \
        int globalID2 = get_global_id(2) * 8; \
        int wh = width * height; \
        if (globalID1 < channel && globalID2 < wh) { \
            int base = globalID0 * channel * wh + globalID1 * wh + globalID2; \
            if (globalID2 + 8 <= wh) { \
                vstore8( \
                    CONVERT_TO_DATA_T8(scale * (convert_float8(vload8(0, input + base)) - zero_point)), \
                    0, \
                    output + base); \
            } else { \
                int num = wh - globalID2; \
                if (num == 1) { \
                    output[base] = scale * (input[base] - zero_point); \
                } else if (num == 2) { \
                    vstore2(CONVERT_TO_DATA_T2(scale * \
                                          (convert_float2(vload2(0, input + base)) - zero_point)), \
                            0, \
                            output + base); \
                } else if (num == 3) { \
                    vstore3(CONVERT_TO_DATA_T3(scale * \
                                          (convert_float3(vload3(0, input + base)) - zero_point)), \
                            0, \
                            output + base); \
                } else if (num == 4) { \
                    vstore4(CONVERT_TO_DATA_T4(scale * \
                                          (convert_float4(vload4(0, input + base)) - zero_point)), \
                            0, \
                            output + base); \
                } else if (num == 5) { \
                    vstore4(CONVERT_TO_DATA_T4(scale * \
                                          (convert_float4(vload4(0, input + base)) - zero_point)), \
                            0, \
                            output + base); \
                    output[base + 4] = scale * (input[base + 4] - zero_point); \
                } else if (num == 6) { \
                    vstore4(CONVERT_TO_DATA_T4(scale * \
                                          (convert_float4(vload4(0, input + base)) - zero_point)), \
                            0, \
                            output + base); \
                    vstore2(CONVERT_TO_DATA_T2( \
                                scale * (convert_float2(vload2(0, input + base + 4)) - zero_point)), \
                            0, \
                            output + base + 4); \
                } else if (num == 7) { \
                    vstore4(CONVERT_TO_DATA_T4(scale * \
                                          (convert_float4(vload4(0, input + base)) - zero_point)), \
                            0, \
                            output + base); \
                    vstore3(CONVERT_TO_DATA_T3( \
                                scale * (convert_float3(vload3(0, input + base + 4)) - zero_point)), \
                            0, \
                            output + base + 4); \
                } \
            } \
        }

// // ToDo(all): support all channel_dim
#define DEQUANTIZATION_PARAM(input, output, scales, channel_dim) \
        int globalID0 = get_global_id(0);  \
        int globalID1 = get_global_id(1);  \
        int globalID2 = get_global_id(2);  \
        if (channel_dim == 0) { \
            output[globalID0 * get_global_size(1) * get_global_size(2) + \
                   globalID1 * get_global_size(2) + globalID2] = \
                input[globalID0 * get_global_size(1) * get_global_size(2) + \
                      globalID1 * get_global_size(2) + globalID2] * \
                scales[globalID0]; \
        } else if (channel_dim == 1) { \
            output[globalID0 * get_global_size(1) * get_global_size(2) + \
                   globalID1 * get_global_size(2) + globalID2] = \
                input[globalID0 * get_global_size(1) * get_global_size(2) + \
                      globalID1 * get_global_size(2) + globalID2] * \
                scales[globalID1]; \
        }

#define DATA_T float
#define CONVERT_TO_DATA_T2(x) x
#define CONVERT_TO_DATA_T3(x) x
#define CONVERT_TO_DATA_T4(x) x
#define CONVERT_TO_DATA_T8(x) x

ADD_SINGLE_KERNEL(dequantization_half_FP32, (__global float *input,
                                             __global float *output,
                                             float scale,
                                             int zero_point,
                                             unsigned int channel,
                                             unsigned int height,
                                             unsigned int width) {
        DEQUANTIZATION_UINT8(input, output, scale, zero_point, channel, height, width)
})

ADD_SINGLE_KERNEL(dequantization_uint8_FP32, (__global uchar *input,
                                              __global float *output,
                                              float scale,
                                              int zero_point,
                                              unsigned int channel,
                                              unsigned int height,
                                              unsigned int width) {
        DEQUANTIZATION_UINT8(input, output, scale, zero_point, channel, height, width)
})

ADD_SINGLE_KERNEL(dequantization_int8_FP32, (__global char *input,
                                             __global float *output,
                                             float scale,
                                             int zero_point,
                                             unsigned int channel,
                                             unsigned int height,
                                             unsigned int width) {
       DEQUANTIZATION_UINT8(input, output, scale, zero_point, channel, height, width)
})

ADD_SINGLE_KERNEL(dequantization_int_FP32, (__global int *input,
                                            __global float *output,
                                            float scale,
                                            int zero_point,
                                            unsigned int channel,
                                            unsigned int height,
                                            unsigned int width) {
       DEQUANTIZATION_UINT8(input, output, scale, zero_point, channel, height, width)
})

ADD_SINGLE_KERNEL(dequantization_short_FP32, (__global short *input,
                                              __global float *output,
                                              float scale,
                                              int zero_point,
                                              unsigned int channel,
                                              unsigned int height,
                                              unsigned int width) {
        DEQUANTIZATION_UINT8(input, output, scale, zero_point, channel, height, width)
})

ADD_SINGLE_KERNEL(dequantization_ushort_FP32, (__global ushort *input,
                                               __global float *output,
                                               float scale,
                                               int zero_point,
                                               unsigned int channel,
                                               unsigned int height,
                                               unsigned int width) {
        DEQUANTIZATION_UINT8(input, output, scale, zero_point, channel, height, width)
})

ADD_SINGLE_KERNEL(dequantization_uint8_param_FP32, (__global uchar *input,
                                                  __global float *output,
                                                  __global float *scales,
                                                  unsigned int channel_dim) {
        DEQUANTIZATION_PARAM(input, output, scales, channel_dim)
})

ADD_SINGLE_KERNEL(dequantization_int8_param_FP32, (__global char *input,
                                                 __global float *output,
                                                 __global float *scales,
                                                 unsigned int channel_dim) {
        DEQUANTIZATION_PARAM(input, output, scales, channel_dim)
})

ADD_SINGLE_KERNEL(dequantization_int_param_FP32, (__global int *input,
                                                __global float *output,
                                                __global float *scales,
                                                unsigned int channel_dim) {
        DEQUANTIZATION_PARAM(input, output, scales, channel_dim)
})

ADD_SINGLE_KERNEL(dequantization_ushort_param_FP32, (__global ushort *input,
                                                   __global float *output,
                                                   __global float *scales,
                                                   unsigned int channel_dim) {
        DEQUANTIZATION_PARAM(input, output, scales, channel_dim)
})

ADD_SINGLE_KERNEL(dequantization_short_param_FP32, (__global short *input,
                                                  __global float *output,
                                                  __global float *scales,
                                                  unsigned int channel_dim) {
        DEQUANTIZATION_PARAM(input, output, scales, channel_dim)
})

#undef DATA_T
#undef CONVERT_TO_DATA_T2
#undef CONVERT_TO_DATA_T3
#undef CONVERT_TO_DATA_T4
#undef CONVERT_TO_DATA_T8

#define DATA_T half
#define CONVERT_TO_DATA_T2(x) convert_half2(x)
#define CONVERT_TO_DATA_T3(x) convert_half3(x)
#define CONVERT_TO_DATA_T4(x) convert_half4(x)
#define CONVERT_TO_DATA_T8(x) convert_half8(x)

ADD_SINGLE_KERNEL(dequantization_half_FP16, (__global half *input,
                                             __global half *output,
                                             float scale,
                                             int zero_point,
                                             unsigned int channel,
                                             unsigned int height,
                                             unsigned int width) {
        DEQUANTIZATION_UINT8(input, output, scale, zero_point, channel, height, width)
})

ADD_SINGLE_KERNEL(dequantization_uint8_FP16, (__global uchar *input,
                                              __global half *output,
                                              float scale,
                                              int zero_point,
                                              unsigned int channel,
                                              unsigned int height,
                                              unsigned int width) {
        DEQUANTIZATION_UINT8(input, output, scale, zero_point, channel, height, width)
})

ADD_SINGLE_KERNEL(dequantization_int8_FP16, (__global char *input,
                                             __global half *output,
                                             float scale,
                                             int zero_point,
                                             unsigned int channel,
                                             unsigned int height,
                                             unsigned int width) {
        DEQUANTIZATION_UINT8(input, output, scale, zero_point, channel, height, width)
})

ADD_SINGLE_KERNEL(dequantization_int_FP16, (__global int *input,
                                            __global half *output,
                                            float scale,
                                            int zero_point,
                                            unsigned int channel,
                                            unsigned int height,
                                            unsigned int width) {
        DEQUANTIZATION_UINT8(input, output, scale, zero_point, channel, height, width)
})

ADD_SINGLE_KERNEL(dequantization_short_FP16, (__global short *input,
                                              __global half *output,
                                              float scale,
                                              int zero_point,
                                              unsigned int channel,
                                              unsigned int height,
                                              unsigned int width) {
        DEQUANTIZATION_UINT8(input, output, scale, zero_point, channel, height, width)
})

ADD_SINGLE_KERNEL(dequantization_ushort_FP16, (__global ushort *input,
                                              __global half *output,
                                              float scale,
                                              int zero_point,
                                              unsigned int channel,
                                              unsigned int height,
                                              unsigned int width) {
        DEQUANTIZATION_UINT8(input, output, scale, zero_point, channel, height, width)
})

ADD_SINGLE_KERNEL(dequantization_uint8_param_FP16, (__global uchar *input,
                                                  __global half *output,
                                                  __global half *scales,
                                                  unsigned int channel_dim) {
        DEQUANTIZATION_PARAM(input, output, scales, channel_dim)
})

ADD_SINGLE_KERNEL(dequantization_int8_param_FP16, (__global char *input,
                                                 __global half *output,
                                                 __global half *scales,
                                                 unsigned int channel_dim) {
        DEQUANTIZATION_PARAM(input, output, scales, channel_dim)
})

ADD_SINGLE_KERNEL(dequantization_ushort_param_FP16, (__global ushort *input,
                                                   __global half *output,
                                                   __global half *scales,
                                                   unsigned int channel_dim) {
        DEQUANTIZATION_PARAM(input, output, scales, channel_dim)
})

ADD_SINGLE_KERNEL(dequantization_short_param_FP16, (__global short *input,
                                                  __global half *output,
                                                  __global half *scales,
                                                  unsigned int channel_dim) {
        DEQUANTIZATION_PARAM(input, output, scales, channel_dim)
})

ADD_SINGLE_KERNEL(dequantization_int_param_FP16, (__global int *input,
                                                __global half *output,
                                                __global half *scales,
                                                unsigned int channel_dim) {
        DEQUANTIZATION_PARAM(input, output, scales, channel_dim)
})
#undef DATA_T
#undef CONVERT_TO_DATA_T2
#undef CONVERT_TO_DATA_T3
#undef CONVERT_TO_DATA_T4
#undef CONVERT_TO_DATA_T8

}  // namespace gpu
}  // namespace ud
}  // namespace enn
