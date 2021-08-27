#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

ADD_SINGLE_KERNEL(resize_bilinear_tflite_FP16, (__read_only image2d_t src_data,
                                                          __write_only image2d_t dst_data,
                                                          int src_size_x,
                                                          int src_size_y,
                                                          int src_size_z,
                                                          int dst_size_x,
                                                          int dst_size_y,
                                                          int dst_size_z,
                                                          int border_x,
                                                          int border_y,
                                                          float scale_factor_x,
                                                          float scale_factor_y) {
    int Y = get_global_id(1);
    int Z = get_global_id(2);
    int X = get_global_id(0);
    if (X >= dst_size_x || Y >= dst_size_y || Z >= dst_size_z)
        return;
    float2 f_coords = (float2)(X, Y) * (float2)(scale_factor_x, scale_factor_y);
    float2 f_coords_floor = floor(f_coords);
    int2 coords_floor = (int2)(f_coords_floor.x, f_coords_floor.y);
    int4 st;
    st.xy = max(coords_floor, (int2)(0, 0));
    st.zw = min(coords_floor + (int2)(1, 1), (int2)(border_x, border_y));
    float2 t = f_coords - f_coords_floor;
    float4 src0 = read_imagef(src_data, smp_none, (int2)((st.x), (st.y) * src_size_z + (Z)));
    float4 src1 = read_imagef(src_data, smp_none, (int2)((st.z), (st.y) * src_size_z + (Z)));
    float4 src2 = read_imagef(src_data, smp_none, (int2)((st.x), (st.w) * src_size_z + (Z)));
    float4 src3 = read_imagef(src_data, smp_none, (int2)((st.z), (st.w) * src_size_z + (Z)));
    half4 r0 = convert_half4(mix(mix(src0, src1, t.x), mix(src2, src3, t.x), t.y));
    write_imageh(dst_data, (int2)((X), (Y)*dst_size_z + (Z)), r0);
}
)

#define RESIZE_BILINEAR(input, output, input_height, input_width, output_num, output_channel, output_height, \
                          output_width, height_scale, width_scale, half_pixel_centers) \
    int output_idx = get_global_id(0); \
    if (output_idx < output_num * output_channel * output_height * output_width) { \
        int n = output_idx / output_width / output_height / output_channel; \
        int c = output_idx / output_width / output_height % output_channel; \
        int h = output_idx / output_width % output_height; \
        int w = output_idx % output_width; \
        float input_y = 0.f; \
        if (half_pixel_centers == 1) { \
            input_y = ((float)h + 0.5f) * height_scale - 0.5f; \
        } else { \
            input_y = h * height_scale; \
        } \
        unsigned int y0 = fmax(floor(input_y), 0); \
        int y1 = fmin(ceil(input_y), input_height - 1); \
        float input_x = 0.f; \
        if (half_pixel_centers == 1) { \
            input_x = ((float)w + 0.5f) * width_scale - 0.5f; \
        } else { \
            input_x = w * width_scale; \
        } \
        unsigned int x0 = fmax(floor(input_x), 0); \
        int x1 = fmin(ceil(input_x), input_width - 1); \
        int x0_y0_offset = y0 * input_width + x0; \
        int x0_y1_offset = y1 * input_width + x0; \
        int x1_y0_offset = y0 * input_width + x1; \
        int x1_y1_offset = y1 * input_width + x1; \
        float x0_y0_weight = (1 - (input_y - y0)) * (1 - (input_x - x0)); \
        float x0_y1_weight = (input_y - y0) * (1 - (input_x - x0)); \
        float x1_y0_weight = (1 - (input_y - y0)) * (input_x - x0); \
        float x1_y1_weight = (input_y - y0) * (input_x - x0); \
        int base_idx = \
            n * output_channel * input_height * input_width + c * input_height * input_width; \
        int x0_y0_idx = base_idx + x0_y0_offset; \
        int x0_y1_idx = base_idx + x0_y1_offset; \
        int x1_y0_idx = base_idx + x1_y0_offset; \
        int x1_y1_idx = base_idx + x1_y1_offset; \
        DATA_FP interpolation = \
            (DATA_FP)input[x0_y0_idx] * x0_y0_weight + (DATA_FP)input[x0_y1_idx] * x0_y1_weight + \
            (DATA_FP)input[x1_y0_idx] * x1_y0_weight + (DATA_FP)input[x1_y1_idx] * x1_y1_weight; \
        output[output_idx] = (DATA_T)interpolation; \
    }

#define DATA_T half
#define DATA_FP DATA_T
ADD_SINGLE_KERNEL(resize_bilinear_FP16, (__global const DATA_T *input,
                                                   __global DATA_T *output,
                                                   unsigned int input_height,
                                                   unsigned int input_width,
                                                   unsigned int output_num,
                                                   unsigned int output_channel,
                                                   unsigned int output_height,
                                                   unsigned int output_width,
                                                   float height_scale,
                                                   float width_scale,
                                                   int half_pixel_centers) {
    RESIZE_BILINEAR(input, output, input_height, input_width, output_num, output_channel, output_height, \
                    output_width, height_scale, width_scale, half_pixel_centers)
})
#undef DATA_FP
#undef DATA_T

#define DATA_T float
#define DATA_FP DATA_T
ADD_SINGLE_KERNEL(resize_bilinear_FP32, (__global const DATA_T *input,
                                                   __global DATA_T *output,
                                                   unsigned int input_height,
                                                   unsigned int input_width,
                                                   unsigned int output_num,
                                                   unsigned int output_channel,
                                                   unsigned int output_height,
                                                   unsigned int output_width,
                                                   float height_scale,
                                                   float width_scale,
                                                   int half_pixel_centers) {
    RESIZE_BILINEAR(input, output, input_height, input_width, output_num, output_channel, output_height, \
                    output_width, height_scale, width_scale, half_pixel_centers)
})
#undef DATA_FP
#undef DATA_T

#define DATA_T uchar
#define DATA_FP float
ADD_SINGLE_KERNEL(resize_bilinear_INT8, (__global const DATA_T *input,
                                                __global DATA_T *output,
                                                unsigned int input_height,
                                                unsigned int input_width,
                                                unsigned int output_num,
                                                unsigned int output_channel,
                                                unsigned int output_height,
                                                unsigned int output_width,
                                                float height_scale,
                                                float width_scale,
                                                int half_pixel_centers) {
    RESIZE_BILINEAR(input, output, input_height, input_width, output_num, output_channel, output_height, \
                    output_width, height_scale, width_scale, half_pixel_centers)
})
#undef DATA_FP
#undef DATA_T

#define DATA_T char
#define DATA_FP float
ADD_SINGLE_KERNEL(SIGNEDresize_bilinear_INT8, (__global const DATA_T *input,
                                                __global DATA_T *output,
                                                unsigned int input_height,
                                                unsigned int input_width,
                                                unsigned int output_num,
                                                unsigned int output_channel,
                                                unsigned int output_height,
                                                unsigned int output_width,
                                                float height_scale,
                                                float width_scale,
                                                int half_pixel_centers) {
    RESIZE_BILINEAR(input, output, input_height, input_width, output_num, output_channel, output_height, \
                        output_width, height_scale, width_scale, half_pixel_centers)
})
#undef DATA_FP
#undef DATA_T

#define COMPUTE_INTER2(scaled_x, scaled_y, grid_w_corner, col, width_scale, x0, y0, y1, input_width, input_base, \
                       x0_y0_offset, x0_y1_offset, x0_y0_idx, x0_y1_idx, input, x0y0_x1y0, x0y1_x1y1, \
                       x0y1_x1y1_weight, x0y0_x1y0_weight, inter2) \
    scaled_x = (grid_w_corner + col) * width_scale; \
    x0 = floor(scaled_x); \
    x0_y0_offset = y0 * input_width + x0; \
    x0_y1_offset = y1 * input_width + x0; \
    x0_y0_idx = input_base + x0_y0_offset; \
    x0_y1_idx = input_base + x0_y1_offset; \
    x0y0_x1y0 = convert_float2(vload2(0, input + x0_y0_idx)); \
    x0y1_x1y1 = convert_float2(vload2(0, input + x0_y1_idx)); \
    x0y1_x1y1_weight.s1 = (scaled_y - y0) * (scaled_x - x0); \
    x0y0_x1y0_weight.s1 = (scaled_x - x0) - x0y1_x1y1_weight.s1; \
    x0y1_x1y1_weight.s0 = (scaled_y - y0) - x0y1_x1y1_weight.s1; \
    x0y0_x1y0_weight.s0 = 1 - (scaled_x + scaled_y) + (x0 + y0) + x0y1_x1y1_weight.s1; \
    inter2 = x0y0_x1y0 * x0y0_x1y0_weight + x0y1_x1y1 * x0y1_x1y1_weight;

ADD_SINGLE_KERNEL(resize_bilinear_32_to_512_INT8, (__global const unsigned char *input,
                                                 __global unsigned char *output,
                                                 unsigned int input_height,
                                                 unsigned int input_width,
                                                 unsigned int output_num,
                                                 unsigned int output_channel,
                                                 unsigned int output_height,
                                                 unsigned int output_width,
                                                 float height_scale,
                                                 float width_scale) {
        int output_idx = get_global_id(0);
        int row = get_global_id(1);
        int n = output_idx / (output_width / 16) / (output_height / 16) / output_channel;
        int c = output_idx / (output_width / 16) / (output_height / 16) % output_channel;
        int grid_h = output_idx / (output_width / 16) % (output_height / 16);
        int grid_w = output_idx % (output_width / 16);  // 16*16 grid
        int grid_h_corner = grid_h * 16;
        int grid_w_corner = grid_w * 16;

        int input_base =
            n * output_channel * input_height * input_width + c * input_height * input_width;
        int output_row_base = n * output_channel * output_height * output_width +
                              c * output_height * output_width + grid_h_corner * output_width +
                              grid_w_corner;
        float scaled_x = 0.0f;
        unsigned int x0 = 0;
        int x0_y0_offset = 0;
        int x0_y1_offset = 0;
        int x0_y0_idx = 0;
        int x0_y1_idx = 0;
        uchar16 interpolation = 0;
        float2 x0y0_x1y0 = 0.0f;
        float2 x0y1_x1y1 = 0.0f;
        float2 x0y0_x1y0_weight = 0.0f;
        float2 x0y1_x1y1_weight = 0.0f;
        float2 inter2 = 0.0f;

        float scaled_y = (grid_h_corner + row) * height_scale;
        unsigned int y0 = floor(scaled_y);
        int y1 = min(y0 + 1, input_height - 1);

        int col = 0;
        COMPUTE_INTER2(scaled_x, scaled_y, grid_w_corner, col, width_scale, x0, y0, y1, input_width, input_base, \
                       x0_y0_offset, x0_y1_offset, x0_y0_idx, x0_y1_idx, input, x0y0_x1y0, x0y1_x1y1, \
                       x0y1_x1y1_weight, x0y0_x1y0_weight, inter2)
        interpolation.s0 = (uchar)(inter2.s0 + inter2.s1);

        col = 1;
        COMPUTE_INTER2(scaled_x, scaled_y, grid_w_corner, col, width_scale, x0, y0, y1, input_width, input_base, \
                       x0_y0_offset, x0_y1_offset, x0_y0_idx, x0_y1_idx, input, x0y0_x1y0, x0y1_x1y1, \
                       x0y1_x1y1_weight, x0y0_x1y0_weight, inter2)
        interpolation.s1 = (uchar)(inter2.s0 + inter2.s1);

        col = 2;
        COMPUTE_INTER2(scaled_x, scaled_y, grid_w_corner, col, width_scale, x0, y0, y1, input_width, input_base, \
                       x0_y0_offset, x0_y1_offset, x0_y0_idx, x0_y1_idx, input, x0y0_x1y0, x0y1_x1y1, \
                       x0y1_x1y1_weight, x0y0_x1y0_weight, inter2)
        interpolation.s2 = (uchar)(inter2.s0 + inter2.s1);

        col = 3;
        COMPUTE_INTER2(scaled_x, scaled_y, grid_w_corner, col, width_scale, x0, y0, y1, input_width, input_base, \
                       x0_y0_offset, x0_y1_offset, x0_y0_idx, x0_y1_idx, input, x0y0_x1y0, x0y1_x1y1, \
                       x0y1_x1y1_weight, x0y0_x1y0_weight, inter2)
        interpolation.s3 = (uchar)(inter2.s0 + inter2.s1);

        col = 4;
        COMPUTE_INTER2(scaled_x, scaled_y, grid_w_corner, col, width_scale, x0, y0, y1, input_width, input_base, \
                       x0_y0_offset, x0_y1_offset, x0_y0_idx, x0_y1_idx, input, x0y0_x1y0, x0y1_x1y1, \
                       x0y1_x1y1_weight, x0y0_x1y0_weight, inter2)
        interpolation.s4 = (uchar)(inter2.s0 + inter2.s1);

        col = 5;
        COMPUTE_INTER2(scaled_x, scaled_y, grid_w_corner, col, width_scale, x0, y0, y1, input_width, input_base, \
                       x0_y0_offset, x0_y1_offset, x0_y0_idx, x0_y1_idx, input, x0y0_x1y0, x0y1_x1y1, \
                       x0y1_x1y1_weight, x0y0_x1y0_weight, inter2)
        interpolation.s5 = (uchar)(inter2.s0 + inter2.s1);

        col = 6;
        COMPUTE_INTER2(scaled_x, scaled_y, grid_w_corner, col, width_scale, x0, y0, y1, input_width, input_base, \
                       x0_y0_offset, x0_y1_offset, x0_y0_idx, x0_y1_idx, input, x0y0_x1y0, x0y1_x1y1, \
                       x0y1_x1y1_weight, x0y0_x1y0_weight, inter2)
        interpolation.s6 = (uchar)(inter2.s0 + inter2.s1);

        col = 7;
        COMPUTE_INTER2(scaled_x, scaled_y, grid_w_corner, col, width_scale, x0, y0, y1, input_width, input_base, \
                       x0_y0_offset, x0_y1_offset, x0_y0_idx, x0_y1_idx, input, x0y0_x1y0, x0y1_x1y1, \
                       x0y1_x1y1_weight, x0y0_x1y0_weight, inter2)
        interpolation.s7 = (uchar)(inter2.s0 + inter2.s1);

        col = 8;
        COMPUTE_INTER2(scaled_x, scaled_y, grid_w_corner, col, width_scale, x0, y0, y1, input_width, input_base, \
                       x0_y0_offset, x0_y1_offset, x0_y0_idx, x0_y1_idx, input, x0y0_x1y0, x0y1_x1y1, \
                       x0y1_x1y1_weight, x0y0_x1y0_weight, inter2)
        interpolation.s8 = (uchar)(inter2.s0 + inter2.s1);

        col = 9;
        COMPUTE_INTER2(scaled_x, scaled_y, grid_w_corner, col, width_scale, x0, y0, y1, input_width, input_base, \
                       x0_y0_offset, x0_y1_offset, x0_y0_idx, x0_y1_idx, input, x0y0_x1y0, x0y1_x1y1, \
                       x0y1_x1y1_weight, x0y0_x1y0_weight, inter2)
        interpolation.s9 = (uchar)(inter2.s0 + inter2.s1);

        col = 10;
        COMPUTE_INTER2(scaled_x, scaled_y, grid_w_corner, col, width_scale, x0, y0, y1, input_width, input_base, \
                       x0_y0_offset, x0_y1_offset, x0_y0_idx, x0_y1_idx, input, x0y0_x1y0, x0y1_x1y1, \
                       x0y1_x1y1_weight, x0y0_x1y0_weight, inter2)
        interpolation.sa = (uchar)(inter2.s0 + inter2.s1);

        col = 11;
        COMPUTE_INTER2(scaled_x, scaled_y, grid_w_corner, col, width_scale, x0, y0, y1, input_width, input_base, \
                       x0_y0_offset, x0_y1_offset, x0_y0_idx, x0_y1_idx, input, x0y0_x1y0, x0y1_x1y1, \
                       x0y1_x1y1_weight, x0y0_x1y0_weight, inter2)
        interpolation.sb = (uchar)(inter2.s0 + inter2.s1);

        col = 12;
        COMPUTE_INTER2(scaled_x, scaled_y, grid_w_corner, col, width_scale, x0, y0, y1, input_width, input_base, \
                       x0_y0_offset, x0_y1_offset, x0_y0_idx, x0_y1_idx, input, x0y0_x1y0, x0y1_x1y1, \
                       x0y1_x1y1_weight, x0y0_x1y0_weight, inter2)
        interpolation.sc = (uchar)(inter2.s0 + inter2.s1);

        col = 13;
        COMPUTE_INTER2(scaled_x, scaled_y, grid_w_corner, col, width_scale, x0, y0, y1, input_width, input_base, \
                       x0_y0_offset, x0_y1_offset, x0_y0_idx, x0_y1_idx, input, x0y0_x1y0, x0y1_x1y1, \
                       x0y1_x1y1_weight, x0y0_x1y0_weight, inter2)
        interpolation.sd = (uchar)(inter2.s0 + inter2.s1);

        col = 14;
        COMPUTE_INTER2(scaled_x, scaled_y, grid_w_corner, col, width_scale, x0, y0, y1, input_width, input_base, \
                       x0_y0_offset, x0_y1_offset, x0_y0_idx, x0_y1_idx, input, x0y0_x1y0, x0y1_x1y1, \
                       x0y1_x1y1_weight, x0y0_x1y0_weight, inter2)
        interpolation.se = (uchar)(inter2.s0 + inter2.s1);

        col = 15;
        scaled_x = (grid_w_corner + col) * width_scale;
        x0 = floor(scaled_x);
        int x1 = min(x0 + 1, input_width - 1);
        x0_y0_offset = y0 * input_width + x0;
        x0_y1_offset = y1 * input_width + x0;
        int x1_y0_offset = y0 * input_width + x1;
        int x1_y1_offset = y1 * input_width + x1;
        x0_y0_idx = input_base + x0_y0_offset;
        int x1_y0_idx = input_base + x1_y0_offset;
        x0_y1_idx = input_base + x0_y1_offset;
        int x1_y1_idx = input_base + x1_y1_offset;
        x0y0_x1y0.s0 = (float)input[x0_y0_idx];
        x0y0_x1y0.s1 = (float)input[x1_y0_idx];
        x0y1_x1y1.s0 = (float)input[x0_y1_idx];
        x0y1_x1y1.s1 = (float)input[x1_y1_idx];
        x0y1_x1y1_weight.s1 = (scaled_y - y0) * (scaled_x - x0);
        x0y0_x1y0_weight.s1 = (scaled_x - x0) - x0y1_x1y1_weight.s1;
        x0y1_x1y1_weight.s0 = (scaled_y - y0) - x0y1_x1y1_weight.s1;
        x0y0_x1y0_weight.s0 = 1 - (scaled_x + scaled_y) + (x0 + y0) + x0y1_x1y1_weight.s1;
        inter2 = x0y0_x1y0 * x0y0_x1y0_weight + x0y1_x1y1 * x0y1_x1y1_weight;
        interpolation.sf = (uchar)(inter2.s0 + inter2.s1);
        vstore16(interpolation, 0, output + output_row_base + row * output_width);
})

}  // namespace gpu
}  // namespace ud
}  // namespace enn
