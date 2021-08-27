#include "userdriver/gpu/common/CLKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define COMPUTE_SUM(x, y) x += y;
#define COMPUTE_MIN(x, y) x = min(x, y);
#define COMPUTE_MAX(x, y) x = max(x, y);
#define COMPUTE_PRODUCT(x, y) x *= y;
#define COMPUTE_BOOL_ALL(x, y) x &= y;
#define COMPUTE_BOOL_ANY(x, y) x |= y;

#define REDUCE(input, offset, N, C, H, W, dm_N, dm_C, dm_H, dm_W) \
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
            COMPUTE_REDUCE(input[get_global_id(0)], input[get_global_id(0) + offset]) \
        }

#define REDUCE_OUTPUT(input, output, N, C, H, W, dm_N, dm_C, dm_H, dm_W) \
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
            output[out_idx] = input[get_global_id(0)]; \
        }

#define DATA_T float
#define COMPUTE_REDUCE(x, y) COMPUTE_SUM(x, y)
ADD_SINGLE_KERNEL(SUM_reduce_FP32, (__global DATA_T *input, uint offset, int N, int C, int H, int W, int dm_N,
                              int dm_C, int dm_H, int dm_W) {
    REDUCE(input, offset, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
ADD_SINGLE_KERNEL(SUM_reduce_output_FP32, (__global DATA_T *input, __global DATA_T *output,
                                     int N, int C, int H, int W,
                                     int dm_N, int dm_C, int dm_H, int dm_W) {
    REDUCE_OUTPUT(input, output, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
#undef COMPUTE_REDUCE

#define COMPUTE_REDUCE(x, y) COMPUTE_MIN(x, y)
ADD_SINGLE_KERNEL(MIN_reduce_FP32, (__global DATA_T *input, uint offset, int N, int C, int H, int W, int dm_N,
                              int dm_C, int dm_H, int dm_W) {
    REDUCE(input, offset, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
ADD_SINGLE_KERNEL(MIN_reduce_output_FP32, (__global DATA_T *input, __global DATA_T *output,
                                     int N, int C, int H, int W,
                                     int dm_N, int dm_C, int dm_H, int dm_W) {
    REDUCE_OUTPUT(input, output, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
#undef COMPUTE_REDUCE

#define COMPUTE_REDUCE(x, y) COMPUTE_MAX(x, y)
ADD_SINGLE_KERNEL(MAX_reduce_FP32, (__global DATA_T *input, uint offset, int N, int C, int H, int W, int dm_N,
                              int dm_C, int dm_H, int dm_W) {
    REDUCE(input, offset, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
ADD_SINGLE_KERNEL(MAX_reduce_output_FP32, (__global DATA_T *input, __global DATA_T *output,
                                     int N, int C, int H, int W,
                                     int dm_N, int dm_C, int dm_H, int dm_W) {
    REDUCE_OUTPUT(input, output, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
#undef COMPUTE_REDUCE

#define COMPUTE_REDUCE(x, y) COMPUTE_PRODUCT(x, y)
ADD_SINGLE_KERNEL(PROD_reduce_FP32, (__global DATA_T *input, uint offset, int N, int C, int H, int W, int dm_N,
                              int dm_C, int dm_H, int dm_W) {
    REDUCE(input, offset, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
ADD_SINGLE_KERNEL(PROD_reduce_output_FP32, (__global DATA_T *input, __global DATA_T *output,
                                     int N, int C, int H, int W,
                                     int dm_N, int dm_C, int dm_H, int dm_W) {
    REDUCE_OUTPUT(input, output, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
#undef COMPUTE_REDUCE
#undef DATA_T  // float

#define DATA_T half
#define COMPUTE_REDUCE(x, y) COMPUTE_SUM(x, y)
ADD_SINGLE_KERNEL(SUM_reduce_FP16, (__global DATA_T *input, uint offset, int N, int C, int H, int W, int dm_N,
                              int dm_C, int dm_H, int dm_W) {
    REDUCE(input, offset, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
ADD_SINGLE_KERNEL(SUM_reduce_output_FP16, (__global DATA_T *input, __global DATA_T *output,
                                     int N, int C, int H, int W,
                                     int dm_N, int dm_C, int dm_H, int dm_W) {
    REDUCE_OUTPUT(input, output, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
#undef COMPUTE_REDUCE

#define COMPUTE_REDUCE(x, y) COMPUTE_MIN(x, y)
ADD_SINGLE_KERNEL(MIN_reduce_FP16, (__global DATA_T *input, uint offset, int N, int C, int H, int W, int dm_N,
                              int dm_C, int dm_H, int dm_W) {
    REDUCE(input, offset, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
ADD_SINGLE_KERNEL(MIN_reduce_output_FP16, (__global DATA_T *input, __global DATA_T *output,
                                     int N, int C, int H, int W,
                                     int dm_N, int dm_C, int dm_H, int dm_W) {
    REDUCE_OUTPUT(input, output, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
#undef COMPUTE_REDUCE

#define COMPUTE_REDUCE(x, y) COMPUTE_MAX(x, y)
ADD_SINGLE_KERNEL(MAX_reduce_FP16, (__global DATA_T *input, uint offset, int N, int C, int H, int W, int dm_N,
                              int dm_C, int dm_H, int dm_W) {
    REDUCE(input, offset, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
ADD_SINGLE_KERNEL(MAX_reduce_output_FP16, (__global DATA_T *input, __global DATA_T *output,
                                     int N, int C, int H, int W,
                                     int dm_N, int dm_C, int dm_H, int dm_W) {
    REDUCE_OUTPUT(input, output, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
#undef COMPUTE_REDUCE

#define COMPUTE_REDUCE(x, y) COMPUTE_PRODUCT(x, y)
ADD_SINGLE_KERNEL(PROD_reduce_FP16, (__global DATA_T *input, uint offset, int N, int C, int H, int W, int dm_N,
                              int dm_C, int dm_H, int dm_W) {
    REDUCE(input, offset, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
ADD_SINGLE_KERNEL(PROD_reduce_output_FP16, (__global DATA_T *input, __global DATA_T *output,
                                     int N, int C, int H, int W,
                                     int dm_N, int dm_C, int dm_H, int dm_W) {
    REDUCE_OUTPUT(input, output, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
#undef COMPUTE_REDUCE
#undef DATA_T  // half

#define DATA_T uchar

#define COMPUTE_REDUCE(x, y) COMPUTE_SUM(x, y)
ADD_SINGLE_KERNEL(SUM_reduce_INT8, (__global DATA_T *input, uint offset, int N, int C, int H, int W, int dm_N,
                              int dm_C, int dm_H, int dm_W) {
    REDUCE(input, offset, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
ADD_SINGLE_KERNEL(SUM_reduce_output_INT8, (__global DATA_T *input, __global DATA_T *output,
                                     int N, int C, int H, int W,
                                     int dm_N, int dm_C, int dm_H, int dm_W) {
    REDUCE_OUTPUT(input, output, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
#undef COMPUTE_REDUCE

#define COMPUTE_REDUCE(x, y) COMPUTE_MIN(x, y)
ADD_SINGLE_KERNEL(MIN_reduce_INT8, (__global DATA_T *input, uint offset, int N, int C, int H, int W, int dm_N,
                              int dm_C, int dm_H, int dm_W) {
    REDUCE(input, offset, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
ADD_SINGLE_KERNEL(MIN_reduce_output_INT8, (__global DATA_T *input, __global DATA_T *output,
                                     int N, int C, int H, int W,
                                     int dm_N, int dm_C, int dm_H, int dm_W) {
    REDUCE_OUTPUT(input, output, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
#undef COMPUTE_REDUCE

#define COMPUTE_REDUCE(x, y) COMPUTE_MAX(x, y)
ADD_SINGLE_KERNEL(MAX_reduce_INT8, (__global DATA_T *input, uint offset, int N, int C, int H, int W, int dm_N,
                              int dm_C, int dm_H, int dm_W) {
    REDUCE(input, offset, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
ADD_SINGLE_KERNEL(MAX_reduce_output_INT8, (__global DATA_T *input, __global DATA_T *output,
                                     int N, int C, int H, int W,
                                     int dm_N, int dm_C, int dm_H, int dm_W) {
    REDUCE_OUTPUT(input, output, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
#undef COMPUTE_REDUCE

#define COMPUTE_REDUCE(x, y) COMPUTE_PRODUCT(x, y)
ADD_SINGLE_KERNEL(PROD_reduce_INT8, (__global DATA_T *input, uint offset, int N, int C, int H, int W, int dm_N,
                              int dm_C, int dm_H, int dm_W) {
    REDUCE(input, offset, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
ADD_SINGLE_KERNEL(PROD_reduce_output_INT8, (__global DATA_T *input, __global DATA_T *output,
                                     int N, int C, int H, int W,
                                     int dm_N, int dm_C, int dm_H, int dm_W) {
    REDUCE_OUTPUT(input, output, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
#undef COMPUTE_REDUCE

#undef DATA_T  // uchar

#define DATA_T char
#define COMPUTE_REDUCE(x, y) COMPUTE_MAX(x, y)
ADD_SINGLE_KERNEL(SIGNED_MAX_reduce_INT8, (__global DATA_T *input, uint offset, int N, int C, int H, int W, int dm_N,
                              int dm_C, int dm_H, int dm_W) {
    REDUCE(input, offset, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
ADD_SINGLE_KERNEL(SIGNED_MAX_reduce_output_INT8, (__global DATA_T *input, __global DATA_T *output,
                                     int N, int C, int H, int W,
                                     int dm_N, int dm_C, int dm_H, int dm_W) {
    REDUCE_OUTPUT(input, output, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
#undef COMPUTE_REDUCE

#define COMPUTE_REDUCE(x, y) COMPUTE_MIN(x, y)
ADD_SINGLE_KERNEL(SIGNED_MIN_reduce_INT8, (__global DATA_T *input, uint offset, int N, int C, int H, int W, int dm_N,
                              int dm_C, int dm_H, int dm_W) {
    REDUCE(input, offset, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
ADD_SINGLE_KERNEL(SIGNED_MIN_reduce_output_INT8, (__global DATA_T *input, __global DATA_T *output,
                                     int N, int C, int H, int W,
                                     int dm_N, int dm_C, int dm_H, int dm_W) {
    REDUCE_OUTPUT(input, output, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
#undef COMPUTE_REDUCE
#undef DATA_T  // char


#define DATA_T bool
#define COMPUTE_REDUCE(x, y) COMPUTE_BOOL_ALL(x, y)
ADD_SINGLE_KERNEL(ALL_reduce_FP32, (__global DATA_T *input, uint offset, int N, int C, int H, int W, int dm_N,
                              int dm_C, int dm_H, int dm_W) {
    REDUCE(input, offset, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
ADD_SINGLE_KERNEL(ALL_reduce_output_FP32, (__global DATA_T *input, __global DATA_T *output,
                                     int N, int C, int H, int W,
                                     int dm_N, int dm_C, int dm_H, int dm_W) {
    REDUCE_OUTPUT(input, output, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})

ADD_SINGLE_KERNEL(ALL_reduce_FP16, (__global DATA_T *input, uint offset, int N, int C, int H, int W, int dm_N,
                              int dm_C, int dm_H, int dm_W) {
    REDUCE(input, offset, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
ADD_SINGLE_KERNEL(ALL_reduce_output_FP16, (__global DATA_T *input, __global DATA_T *output,
                                     int N, int C, int H, int W,
                                     int dm_N, int dm_C, int dm_H, int dm_W) {
    REDUCE_OUTPUT(input, output, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})

ADD_SINGLE_KERNEL(ALL_reduce_INT8, (__global DATA_T *input, uint offset, int N, int C, int H, int W, int dm_N,
                              int dm_C, int dm_H, int dm_W) {
    REDUCE(input, offset, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
ADD_SINGLE_KERNEL(ALL_reduce_output_INT8, (__global DATA_T *input, __global DATA_T *output,
                                     int N, int C, int H, int W,
                                     int dm_N, int dm_C, int dm_H, int dm_W) {
    REDUCE_OUTPUT(input, output, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
#undef COMPUTE_REDUCE

#define COMPUTE_REDUCE(x, y) COMPUTE_BOOL_ANY(x, y)
ADD_SINGLE_KERNEL(ANY_reduce_FP32, (__global DATA_T *input, uint offset, int N, int C, int H, int W, int dm_N,
                              int dm_C, int dm_H, int dm_W) {
    REDUCE(input, offset, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
ADD_SINGLE_KERNEL(ANY_reduce_output_FP32, (__global DATA_T *input, __global DATA_T *output,
                                     int N, int C, int H, int W,
                                     int dm_N, int dm_C, int dm_H, int dm_W) {
    REDUCE_OUTPUT(input, output, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})

ADD_SINGLE_KERNEL(ANY_reduce_FP16, (__global DATA_T *input, uint offset, int N, int C, int H, int W, int dm_N,
                              int dm_C, int dm_H, int dm_W) {
    REDUCE(input, offset, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
ADD_SINGLE_KERNEL(ANY_reduce_output_FP16, (__global DATA_T *input, __global DATA_T *output,
                                     int N, int C, int H, int W,
                                     int dm_N, int dm_C, int dm_H, int dm_W) {
    REDUCE_OUTPUT(input, output, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})

ADD_SINGLE_KERNEL(ANY_reduce_INT8, (__global DATA_T *input, uint offset, int N, int C, int H, int W, int dm_N,
                              int dm_C, int dm_H, int dm_W) {
    REDUCE(input, offset, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
ADD_SINGLE_KERNEL(ANY_reduce_output_INT8, (__global DATA_T *input, __global DATA_T *output,
                                     int N, int C, int H, int W,
                                     int dm_N, int dm_C, int dm_H, int dm_W) {
    REDUCE_OUTPUT(input, output, N, C, H, W, dm_N, dm_C, dm_H, dm_W)
})
#undef COMPUTE_REDUCE
#undef DATA_T  // bool

}  // namespace gpu
}  // namespace ud
}  // namespace enn
