#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define PAD8(input_data, output_data, input_batch, input_channel, input_height, input_width, output_batch, \
             output_channel, output_height, output_width, batch_top, batch_down, high_top, high_down, width_left, \
             width_right, channel_top, channel_down, pad_value) \
    int out_b = get_global_id(0); \
    int out_c = get_global_id(1); \
    if (get_global_id(2) < output_height * output_width) { \
        int out_h = get_global_id(2) / output_width; \
        int out_w = get_global_id(2) % output_width; \
        int out_index = get_global_id(0) * output_channel * output_height * output_width + \
                        get_global_id(1) * output_height * output_width + get_global_id(2); \
        int input_index = \
            (get_global_id(0) - batch_top) * input_channel * input_height * input_width + \
            (get_global_id(1) - channel_top) * input_height * input_width + \
            (out_h - high_top) * input_width + out_w - width_left; \
        if (out_b < batch_top || out_b >= batch_down || out_c < channel_top || \
            out_c >= channel_down || out_h < high_top || out_h >= high_down || out_w < width_left || \
            out_w >= width_right) { \
            output_data[out_index] = (DATA_T)pad_value; \
        } else { \
            output_data[out_index] = input_data[input_index]; \
        } \
    }

#define DATA_T uchar
ADD_SINGLE_KERNEL(Pad8_INT8, (__global unsigned char *input_data,
                                                 __global unsigned char *output_data,
                                                 unsigned int input_batch,
                                                 unsigned int input_channel,
                                                 unsigned int input_height,
                                                 unsigned int input_width,
                                                 unsigned int output_batch,
                                                 unsigned int output_channel,
                                                 unsigned int output_height,
                                                 unsigned int output_width,
                                                 unsigned int batch_top,
                                                 unsigned int batch_down,
                                                 unsigned int high_top,
                                                 unsigned int high_down,
                                                 unsigned int width_left,
                                                 unsigned int width_right,
                                                 unsigned int channel_top,
                                                 unsigned int channel_down,
                                                 int pad_value) {
    PAD8(input_data, output_data, input_batch, input_channel, input_height, input_width, output_batch, \
         output_channel, output_height, output_width, batch_top, batch_down, high_top, high_down, width_left, \
         width_right, channel_top, channel_down, pad_value)
})
#undef DATA_T

#define DATA_T half
ADD_SINGLE_KERNEL(Pad8_FP16, (__global half *input_data,
                                        __global half *output_data,
                                        unsigned int input_batch,
                                        unsigned int input_channel,
                                        unsigned int input_height,
                                        unsigned int input_width,
                                        unsigned int output_batch,
                                        unsigned int output_channel,
                                        unsigned int output_height,
                                        unsigned int output_width,
                                        unsigned int batch_top,
                                        unsigned int batch_down,
                                        unsigned int high_top,
                                        unsigned int high_down,
                                        unsigned int width_left,
                                        unsigned int width_right,
                                        unsigned int channel_top,
                                        unsigned int channel_down,
                                        float pad_value) {
    PAD8(input_data, output_data, input_batch, input_channel, input_height, input_width, output_batch, \
         output_channel, output_height, output_width, batch_top, batch_down, high_top, high_down, width_left, \
         width_right, channel_top, channel_down, pad_value)
})
#undef DATA_T

#define DATA_T float
ADD_SINGLE_KERNEL(Pad8_FP32, (__global float *input_data,
                                        __global float *output_data,
                                        unsigned int input_batch,
                                        unsigned int input_channel,
                                        unsigned int input_height,
                                        unsigned int input_width,
                                        unsigned int output_batch,
                                        unsigned int output_channel,
                                        unsigned int output_height,
                                        unsigned int output_width,
                                        unsigned int batch_top,
                                        unsigned int batch_down,
                                        unsigned int high_top,
                                        unsigned int high_down,
                                        unsigned int width_left,
                                        unsigned int width_right,
                                        unsigned int channel_top,
                                        unsigned int channel_down,
                                        float pad_value) {
    PAD8(input_data, output_data, input_batch, input_channel, input_height, input_width, output_batch, \
         output_channel, output_height, output_width, batch_top, batch_down, high_top, high_down, width_left, \
         width_right, channel_top, channel_down, pad_value)
})
#undef DATA_T

}  // namespace gpu
}  // namespace ud
}  // namespace enn
