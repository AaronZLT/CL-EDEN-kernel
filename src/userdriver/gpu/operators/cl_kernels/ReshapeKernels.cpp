#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define RESHAPE(input, output, channel, height, width) \
    int globalID0 = get_global_id(0); \
    int globalID1 = get_global_id(1); \
    int globalID2 = get_global_id(2) * 8; \
    int wh = width * height; \
    if (globalID1 < channel && globalID2 < wh) { \
        int base = globalID0 * channel * wh + globalID1 * wh + globalID2; \
        if (globalID2 + 8 <= wh) { \
            vstore8(vload8(0, input + base), 0, output + base); \
        } else { \
            int num = wh - globalID2; \
            if (num == 1) { \
                output[base] = input[base]; \
            } else if (num == 2) { \
                vstore2(vload2(0, input + base), 0, output + base); \
            } else if (num == 3) { \
                vstore3(vload3(0, input + base), 0, output + base); \
            } else if (num == 4) { \
                vstore4(vload4(0, input + base), 0, output + base); \
            } else if (num == 5) { \
                vstore4(vload4(0, input + base), 0, output + base); \
                output[base + 4] = input[base + 4]; \
            } else if (num == 6) { \
                vstore4(vload4(0, input + base), 0, output + base); \
                vstore2(vload2(0, input + base + 4), 0, output + base + 4); \
            } else if (num == 7) { \
                vstore4(vload4(0, input + base), 0, output + base); \
                vstore3(vload3(0, input + base + 4), 0, output + base + 4); \
            } \
        } \
    }

#define RESHAPE_TFLITE(input, output, in_n, in_c, in_h, in_w, out_n, out_c, out_h, out_w) \
        unsigned int n = get_global_id(0); \
        unsigned int c = get_global_id(1); \
        unsigned int hw = get_global_id(2); \
        if (hw < in_h * in_w) { \
            unsigned int out_chw = out_c * out_h * out_w; \
            unsigned int out_hw = out_h * out_w; \
            unsigned int out_wc = out_w * out_c; \
            unsigned int h = hw / in_w; \
            unsigned int w = hw % in_w; \
            unsigned int in_NCHW_index = n * in_c * in_h * in_w + c * in_h * in_w + hw; \
            unsigned int in_NHWC_index = n * in_h * in_w * in_c + h * in_w * in_c + w * in_c + c; \
            unsigned int out_NHWC_c = in_NHWC_index % out_c; \
            unsigned int out_NHWC_w = (in_NHWC_index / out_c) % out_w; \
            unsigned int out_NHWC_h = (in_NHWC_index / out_wc) % out_h; \
            unsigned int out_NHWC_n = (in_NHWC_index / out_chw) % out_n; \
            unsigned int out_NCHW_index = out_NHWC_n * out_chw + out_NHWC_c * out_hw + \
                                          out_NHWC_h * out_w + out_NHWC_w; \
            output[out_NCHW_index] = input[in_NCHW_index]; \
        }

#define DATA_T uchar
ADD_SINGLE_KERNEL(reshape_INT8, (__global const DATA_T *input,
                           __global DATA_T *output,
                           unsigned int channel,
                           unsigned int height,
                           unsigned int width) {
    RESHAPE(input, output, channel, height, width)
})

ADD_SINGLE_KERNEL(reshape_tflite_INT8, (__global const DATA_T *input,
                                  __global DATA_T *output,
                                  unsigned int in_n,
                                  unsigned int in_c,
                                  unsigned int in_h,
                                  unsigned int in_w,
                                  unsigned int out_n,
                                  unsigned int out_c,
                                  unsigned int out_h,
                                  unsigned int out_w) {
    RESHAPE_TFLITE(input, output, in_n, in_c, in_h, in_w, out_n, out_c, out_h, out_w)
})
#undef DATA_T

#define DATA_T char
ADD_SINGLE_KERNEL(SIGNEDreshape_INT8, (__global const DATA_T *input,
                           __global DATA_T *output,
                           unsigned int channel,
                           unsigned int height,
                           unsigned int width) {
    RESHAPE(input, output, channel, height, width)
})

ADD_SINGLE_KERNEL(SIGNEDreshape_tflite_INT8, (__global const DATA_T *input,
                                  __global DATA_T *output,
                                  unsigned int in_n,
                                  unsigned int in_c,
                                  unsigned int in_h,
                                  unsigned int in_w,
                                  unsigned int out_n,
                                  unsigned int out_c,
                                  unsigned int out_h,
                                  unsigned int out_w) {
    RESHAPE_TFLITE(input, output, in_n, in_c, in_h, in_w, out_n, out_c, out_h, out_w)
})
#undef DATA_T

#define DATA_T float
ADD_SINGLE_KERNEL(reshape_FP32, (__global const DATA_T *input,
                           __global DATA_T *output,
                           unsigned int channel,
                           unsigned int height,
                           unsigned int width) {
    RESHAPE(input, output, channel, height, width)
})

ADD_SINGLE_KERNEL(reshape_tflite_FP32, (__global const DATA_T *input,
                                  __global DATA_T *output,
                                  unsigned int in_n,
                                  unsigned int in_c,
                                  unsigned int in_h,
                                  unsigned int in_w,
                                  unsigned int out_n,
                                  unsigned int out_c,
                                  unsigned int out_h,
                                  unsigned int out_w) {
    RESHAPE_TFLITE(input, output, in_n, in_c, in_h, in_w, out_n, out_c, out_h, out_w)
})
#undef DATA_T

#define DATA_T half
ADD_SINGLE_KERNEL(reshape_FP16, (__global const DATA_T *input,
                           __global DATA_T *output,
                           unsigned int channel,
                           unsigned int height,
                           unsigned int width) {
    RESHAPE(input, output, channel, height, width)
})

ADD_SINGLE_KERNEL(reshape_tflite_FP16, (__global const DATA_T *input,
                                  __global DATA_T *output,
                                  unsigned int in_n,
                                  unsigned int in_c,
                                  unsigned int in_h,
                                  unsigned int in_w,
                                  unsigned int out_n,
                                  unsigned int out_c,
                                  unsigned int out_h,
                                  unsigned int out_w) {
    RESHAPE_TFLITE(input, output, in_n, in_c, in_h, in_w, out_n, out_c, out_h, out_w)
})
#undef DATA_T

#define DATA_T int
ADD_SINGLE_KERNEL(INT32reshape_FP32, (__global const DATA_T *input,
                           __global DATA_T *output,
                           unsigned int channel,
                           unsigned int height,
                           unsigned int width) {
    RESHAPE(input, output, channel, height, width)
})

ADD_SINGLE_KERNEL(INT32reshape_FP16, (__global const DATA_T *input,
                           __global DATA_T *output,
                           unsigned int channel,
                           unsigned int height,
                           unsigned int width) {
    RESHAPE(input, output, channel, height, width)
})
#undef DATA_T

}  // namespace gpu
}  // namespace ud
}  // namespace enn
