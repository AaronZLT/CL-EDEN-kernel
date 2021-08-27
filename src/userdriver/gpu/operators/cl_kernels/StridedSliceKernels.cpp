#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define STRIDE_SLICE(input, output, output_width, input_channel, input_height, input_width, start_b, stride_b, \
                    start_d, stride_d, start_h, stride_h, start_w, stride_w) \
        int in_b = start_b + get_global_id(0) * stride_b; \
        int in_c = start_d + get_global_id(1) * stride_d; \
        int in_h = start_h + (get_global_id(2) / output_width) * stride_h; \
        int in_w = start_w + (get_global_id(2) % output_width) * stride_w; \
        int output_index = get_global_id(0) * get_global_size(1) * get_global_size(2) + \
                           get_global_id(1) * get_global_size(2) + get_global_id(2); \
        int input_index = in_b * input_channel * input_height * input_width + \
                          in_c * input_height * input_width + in_h * input_width + in_w; \
        output[output_index] = input[input_index];

#define DATA_T uchar
ADD_SINGLE_KERNEL(stride_slice_INT8, (__global const DATA_T *input,
                                    __global DATA_T *output,
                                    int output_width,
                                    int input_channel,
                                    int input_height,
                                    int input_width,
                                    int start_b,
                                    int stride_b,
                                    int start_d,
                                    int stride_d,
                                    int start_h,
                                    int stride_h,
                                    int start_w,
                                    int stride_w) {
    STRIDE_SLICE(input, output, output_width, input_channel, input_height, input_width, start_b, stride_b, \
                    start_d, stride_d, start_h, stride_h, start_w, stride_w)
})
#undef DATA_T

#define DATA_T char
ADD_SINGLE_KERNEL(SIGNEDstride_slice_INT8, (__global const DATA_T *input,
                                    __global DATA_T *output,
                                    int output_width,
                                    int input_channel,
                                    int input_height,
                                    int input_width,
                                    int start_b,
                                    int stride_b,
                                    int start_d,
                                    int stride_d,
                                    int start_h,
                                    int stride_h,
                                    int start_w,
                                    int stride_w) {
    STRIDE_SLICE(input, output, output_width, input_channel, input_height, input_width, start_b, stride_b, \
                    start_d, stride_d, start_h, stride_h, start_w, stride_w)
})
#undef DATA_T

#define DATA_T half
ADD_SINGLE_KERNEL(stride_slice_FP16, (__global const DATA_T *input,
                                    __global DATA_T *output,
                                    int output_width,
                                    int input_channel,
                                    int input_height,
                                    int input_width,
                                    int start_b,
                                    int stride_b,
                                    int start_d,
                                    int stride_d,
                                    int start_h,
                                    int stride_h,
                                    int start_w,
                                    int stride_w) {
    STRIDE_SLICE(input, output, output_width, input_channel, input_height, input_width, start_b, stride_b, \
                    start_d, stride_d, start_h, stride_h, start_w, stride_w)
})
#undef DATA_T

#define DATA_T float
ADD_SINGLE_KERNEL(stride_slice_FP32, (__global const DATA_T *input,
                                    __global DATA_T *output,
                                    int output_width,
                                    int input_channel,
                                    int input_height,
                                    int input_width,
                                    int start_b,
                                    int stride_b,
                                    int start_d,
                                    int stride_d,
                                    int start_h,
                                    int stride_h,
                                    int start_w,
                                    int stride_w) {
    STRIDE_SLICE(input, output, output_width, input_channel, input_height, input_width, start_b, stride_b, \
                    start_d, stride_d, start_h, stride_h, start_w, stride_w)
})
#undef DATA_T

}  // namespace gpu
}  // namespace ud
}  // namespace enn
