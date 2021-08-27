#include "userdriver/gpu/common/CLKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define SLICE_HW(input, output, axis, offset, IN_N, IN_C, IN_H, IN_W, OUT_N, OUT_C, OUT_H, OUT_W) \
    int GID0 = get_global_id(0); \
    int GID1 = get_global_id(1); \
    int GID2 = get_global_id(2); \
    int batch = GID0 / OUT_C; \
    int channel = GID0 % OUT_C; \
    int out_index = batch * OUT_C * OUT_H * OUT_W + channel * OUT_H * OUT_W + GID1 * OUT_W + GID2; \
    int in_index = 0; \
    if (axis == 2) { \
        in_index = batch * IN_C * IN_H * IN_W + channel * IN_H * IN_W + (GID1 + offset) * IN_W + GID2; \
    } else if (axis == 3) { \
        in_index = batch * IN_C * IN_H * IN_W + channel * IN_H * IN_W + GID1 * IN_W + (GID2 + offset); \
    } \
    output[out_index] = input[in_index];

#define SLICE_NC(input, output, axis, offset, IN_N, IN_C, IN_H, IN_W, OUT_N, OUT_C, OUT_H, OUT_W) \
    int GID0 = get_global_id(0); \
    int GID1 = get_global_id(1); \
    int GID2 = get_global_id(2); \
    int out_index = GID0 * OUT_C * OUT_H * OUT_W + GID1 * OUT_H * OUT_W + GID2; \
    int in_index = 0; \
    if (axis == 0) { \
        in_index = (GID0 + offset) * IN_C * IN_H * IN_W + GID1 * IN_H * IN_W + GID2; \
    } else if (axis == 1) { \
        in_index = GID0 * IN_C * IN_H * IN_W + (GID1 + offset) * IN_H * IN_W + GID2; \
    } \
    output[out_index] = input[in_index];

#define DATA_T half
#define DATA_T2 half2
#define DATA_T3 half3
#define DATA_T4 half4
#define DATA_T8 half8
#define ZERO (half)0.0f
ADD_SINGLE_KERNEL(slice_HW_FP16, (__global const half *input,
                                  __global half *output,
                                  unsigned int axis,
                                  int offset,
                                  int IN_N,
                                  int IN_C,
                                  int IN_H,
                                  int IN_W,
                                  int OUT_N,
                                  int OUT_C,
                                  int OUT_H,
                                  int OUT_W) {
    SLICE_HW(input, output, axis, offset, IN_N, IN_C, IN_H, IN_W, OUT_N, OUT_C, OUT_H, OUT_W)
})

ADD_SINGLE_KERNEL(slice_NC_FP16, (__global const half *input,
                                  __global half *output,
                                  unsigned int axis,
                                  int offset,
                                  int IN_N,
                                  int IN_C,
                                  int IN_H,
                                  int IN_W,
                                  int OUT_N,
                                  int OUT_C,
                                  int OUT_H,
                                  int OUT_W) {
    SLICE_NC(input, output, axis, offset, IN_N, IN_C, IN_H, IN_W, OUT_N, OUT_C, OUT_H, OUT_W)
})

#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T
#undef ZERO

#define DATA_T float
#define DATA_T2 float2
#define DATA_T3 float3
#define DATA_T4 float4
#define DATA_T8 float8
#define ZERO 0.0f
ADD_SINGLE_KERNEL(slice_HW_FP32, (__global const float *input,
                                  __global float *output,
                                  unsigned int axis,
                                  int offset,
                                  int IN_N,
                                  int IN_C,
                                  int IN_H,
                                  int IN_W,
                                  int OUT_N,
                                  int OUT_C,
                                  int OUT_H,
                                  int OUT_W) {
    SLICE_HW(input, output, axis, offset, IN_N, IN_C, IN_H, IN_W, OUT_N, OUT_C, OUT_H, OUT_W)
})

ADD_SINGLE_KERNEL(slice_NC_FP32, (__global const float *input,
                                  __global float *output,
                                  unsigned int axis,
                                  int offset,
                                  int IN_N,
                                  int IN_C,
                                  int IN_H,
                                  int IN_W,
                                  int OUT_N,
                                  int OUT_C,
                                  int OUT_H,
                                  int OUT_W) {
    SLICE_NC(input, output, axis, offset, IN_N, IN_C, IN_H, IN_W, OUT_N, OUT_C, OUT_H, OUT_W)
})
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T
#undef ZERO

}  // namespace gpu
}  // namespace ud
}  // namespace enn
