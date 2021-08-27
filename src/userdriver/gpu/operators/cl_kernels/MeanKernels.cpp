#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define SUM_AXIS(input, offset, N, C, H, W, dm_N, dm_C, dm_H, dm_W) \
    unsigned int x[4]; \
    x[0] = C * H * W; \
    x[1] = H * W; \
    x[2] = W; \
    x[3] = 1; \
    unsigned int in_idx[4]; \
    unsigned int gid = get_global_id(0); \
    for (int j = 0; j < 4; j++) { \
        uint tmp = 0; \
        if (x[j] == 1) { \
            tmp = gid; \
            gid = 0; \
        } \
        while (gid >= x[j]) { \
            gid -= x[j]; \
            tmp++; \
        } \
        in_idx[j] = tmp; \
    } \
    if (in_idx[3] < dm_W && in_idx[2] < dm_H && in_idx[1] < dm_C && in_idx[0] < dm_N) { \
        input[get_global_id(0)] += input[get_global_id(0) + offset]; \
    }

#define MEAN(input, output, axis_size, N, C, H, W, dm_N, dm_C, dm_H, dm_W) \
    unsigned int x[4]; \
    x[0] = C * H * W; \
    x[1] = H * W; \
    x[2] = W; \
    x[3] = 1; \
    unsigned int in_idx[4]; \
    unsigned int gid = get_global_id(0); \
    for (int j = 0; j < 4; j++) { \
        uint tmp = 0; \
        if (x[j] == 1) { \
            tmp = gid; \
            gid = 0; \
        } \
        while (gid >= x[j]) { \
            gid -= x[j]; \
            tmp++; \
        } \
        in_idx[j] = tmp; \
    } \
    if (in_idx[3] < dm_W && in_idx[2] < dm_H && in_idx[1] < dm_C && in_idx[0] < dm_N) { \
        unsigned int out_idx = in_idx[3] + in_idx[2] * dm_W + in_idx[1] * dm_H * dm_W + \
                                in_idx[0] * dm_C * dm_H * dm_W; \
        output[out_idx] = input[get_global_id(0)] / axis_size; \
        }

#define REMAIN_FEATURE_SUM(feature_size, vector_size, offset, input, feature_sum) \
    unsigned int remain_size = feature_size - vector_size * 8; \
    if (remain_size == 1) { \
        feature_sum += input[offset + vector_size * 8]; \
    } else if (remain_size == 2) { \
        feature_sum += input[offset + vector_size * 8] + input[offset + vector_size * 8 + 1]; \
    } else if (remain_size == 3) { \
        feature_sum += input[offset + vector_size * 8] + input[offset + vector_size * 8 + 1] + \
                        input[offset + vector_size * 8 + 2]; \
    } else if (remain_size == 4) { \
        feature_sum += input[offset + vector_size * 8] + input[offset + vector_size * 8 + 1] + \
                        input[offset + vector_size * 8 + 2] + \
                        input[offset + vector_size * 8 + 3]; \
    } else if (remain_size == 5) { \
        feature_sum += input[offset + vector_size * 8] + input[offset + vector_size * 8 + 1] + \
                        input[offset + vector_size * 8 + 2] + \
                        input[offset + vector_size * 8 + 3] + \
                        input[offset + vector_size * 8 + 4]; \
    } else if (remain_size == 6) { \
        feature_sum += \
            input[offset + vector_size * 8] + input[offset + vector_size * 8 + 1] + \
            input[offset + vector_size * 8 + 2] + input[offset + vector_size * 8 + 3] + \
            input[offset + vector_size * 8 + 4] + input[offset + vector_size * 8 + 5]; \
    } else if (remain_size == 7) { \
        feature_sum += \
            input[offset + vector_size * 8] + input[offset + vector_size * 8 + 1] + \
            input[offset + vector_size * 8 + 2] + input[offset + vector_size * 8 + 3] + \
            input[offset + vector_size * 8 + 4] + input[offset + vector_size * 8 + 5] + \
            input[offset + vector_size * 8 + 6]; \
    }

#define MEAN_FEATURE_OPT(input, output, feature_size, num_elements_in_axis) \
    unsigned int GID0 = get_global_id(0); \
    DATA_T8 feature_sum8 = 0.0f; \
    unsigned int offset = GID0 * feature_size; \
    unsigned int vector_size = feature_size / 8; \
    for (int i = 0; i < vector_size; i++) { \
        feature_sum8 += vload8(i, input + offset); \
    } \
    DATA_T feature_sum = feature_sum8.s0 + feature_sum8.s1 + feature_sum8.s2 + feature_sum8.s3 + \
                        feature_sum8.s4 + feature_sum8.s5 + feature_sum8.s6 + feature_sum8.s7; \
    REMAIN_FEATURE_SUM(feature_size, vector_size, offset, input, feature_sum) \
    output[GID0] = feature_sum / num_elements_in_axis;


#define Q_MEAN_FEATURE_OPT(input, output, feature_size, num_elements_in_axis) \
    int GID0 = get_global_id(0); \
    int8 feature_sum8 = 0.0f; \
    unsigned int offset = GID0 * feature_size; \
    unsigned int vector_size = feature_size / 8; \
    for (int i = 0; i < vector_size; i++) { \
        feature_sum8 += convert_int8(vload8(i, input + offset)); \
    } \
    int feature_sum = feature_sum8.s0 + feature_sum8.s1 + feature_sum8.s2 + feature_sum8.s3 + feature_sum8.s4 + \
                        feature_sum8.s5 + feature_sum8.s6 + feature_sum8.s7; \
    REMAIN_FEATURE_SUM(feature_size, vector_size, offset, input, feature_sum) \
    output[GID0] = feature_sum / num_elements_in_axis;

#define DATA_T uchar
ADD_SINGLE_KERNEL(sum_axis_INT8, (__global int *input,
                                uint offset,
                                int N,
                                int C,
                                int H,
                                int W,
                                int dm_N,
                                int dm_C,
                                int dm_H,
                                int dm_W) {
    SUM_AXIS(input, offset, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})

ADD_SINGLE_KERNEL(mean_INT8, (__global int *input,
                            __global DATA_T *output,
                            int axis_size,
                            int N,
                            int C,
                            int H,
                            int W,
                            int dm_N,
                            int dm_C,
                            int dm_H,
                            int dm_W) {
    MEAN(input, output, axis_size, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})

ADD_SINGLE_KERNEL(mean_feature_opt_INT8, (__global DATA_T *input,
                                        __global DATA_T *output,
                                        int feature_size,
                                        int num_elements_in_axis) {
    Q_MEAN_FEATURE_OPT(input, output, feature_size, num_elements_in_axis)
})
#undef DATA_T  // uchar

#define DATA_T char
ADD_SINGLE_KERNEL(SIGNEDmean_feature_opt_INT8, (__global DATA_T *input,
                                        __global DATA_T *output,
                                        int feature_size,
                                        int num_elements_in_axis) {
    Q_MEAN_FEATURE_OPT(input, output, feature_size, num_elements_in_axis)
})
#undef DATA_T  // char

/********  FP16 KERNELS ********/
#define DATA_T half
#define DATA_T2 half2
#define DATA_T3 half3
#define DATA_T4 half4
#define DATA_T8 half8
#define DATA_T16 half16

ADD_SINGLE_KERNEL(sum_axis_FP16, (__global DATA_T *input,
                                uint offset,
                                int N,
                                int C,
                                int H,
                                int W,
                                int dm_N,
                                int dm_C,
                                int dm_H,
                                int dm_W) {
    SUM_AXIS(input, offset, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})

ADD_SINGLE_KERNEL(mean_FP16, (__global DATA_T *input,
                            __global DATA_T *output,
                            int axis_size,
                            int N,
                            int C,
                            int H,
                            int W,
                            int dm_N,
                            int dm_C,
                            int dm_H,
                            int dm_W) {
    MEAN(input, output, axis_size, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})

ADD_SINGLE_KERNEL(mean_feature_opt_FP16, (__global DATA_T *input,
                                        __global DATA_T *output,
                                        unsigned int feature_size,
                                        unsigned int num_elements_in_axis) {
    MEAN_FEATURE_OPT(input, output, feature_size, num_elements_in_axis)
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

ADD_SINGLE_KERNEL(sum_axis_FP32, (__global DATA_T *input,
                                uint offset,
                                int N,
                                int C,
                                int H,
                                int W,
                                int dm_N,
                                int dm_C,
                                int dm_H,
                                int dm_W) {
    SUM_AXIS(input, offset, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})

ADD_SINGLE_KERNEL(mean_FP32, (__global DATA_T *input,
                            __global DATA_T *output,
                            int axis_size,
                            int N,
                            int C,
                            int H,
                            int W,
                            int dm_N,
                            int dm_C,
                            int dm_H,
                            int dm_W) {
    MEAN(input, output, axis_size, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})

ADD_SINGLE_KERNEL(mean_feature_opt_FP32, (__global DATA_T *input,
                                        __global DATA_T *output,
                                        unsigned int feature_size,
                                        unsigned int num_elements_in_axis) {
    MEAN_FEATURE_OPT(input, output, feature_size, num_elements_in_axis)
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
