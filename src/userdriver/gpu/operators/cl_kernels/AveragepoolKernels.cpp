#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define Q_AVEPOOLING_TFLITE(input, output, input_height, input_width, output_height, output_width, kernel_height, \
                          kernel_width, stride_height, stride_width, pad_height, pad_width, act_max, act_min) \
    if (get_global_id(2) < output_height * output_width) { \
        int quantized_avepooling = 0; \
        int start_i = get_global_id(2) / output_width * stride_height - pad_height; \
        int start_j = get_global_id(2) % output_width * stride_width - pad_width; \
        int end_i = min(start_i + kernel_height, input_height); \
        int end_j = min(start_j + kernel_width, input_width); \
        start_i = max(start_i, 0); \
        start_j = max(start_j, 0); \
        end_i = min(end_i, (int)input_height); \
        end_j = min(end_j, (int)input_width); \
        int pool_size = (end_i - start_i) * (end_j - start_j); \
        int input_base = input_height * input_width * \
                            (get_global_id(0) * get_global_size(1) + get_global_id(1)); \
        for (int i = start_i; i < end_i; i++) { \
            for (int j = start_j; j < end_j; j++) { \
                quantized_avepooling += (int)input[input_base + input_width * i + j]; \
            } \
        } \
        quantized_avepooling = QUANTIZED_AVE(quantized_avepooling, pool_size); \
        quantized_avepooling = max(quantized_avepooling, act_min); \
        quantized_avepooling = min(quantized_avepooling, act_max); \
        output[(get_global_id(0) * get_global_size(1) + get_global_id(1)) * output_height * \
                    output_width + get_global_id(2)] = (DATA_T)quantized_avepooling; \
    }

/********  QUANTIZED KERNELS ********/
#define DATA_T uchar
#define DATA_T2 uchar2
#define DATA_T3 uchar3
#define DATA_T4 uchar4
#define DATA_T8 uchar8
#define DATA_T16 uchar16
#define QUANTIZED_AVE(quantized_avepooling, pool_size) \
    ((quantized_avepooling + pool_size / 2) / pool_size)

ADD_SINGLE_KERNEL(avepooling_tflite_INT8, (__global DATA_T *input,
                               __global DATA_T *output,
                               unsigned int input_height, unsigned int input_width,
                               unsigned int output_height, unsigned int output_width,
                               unsigned int kernel_height, unsigned int kernel_width,
                               unsigned int stride_height, unsigned int stride_width,
                               unsigned int pad_height, unsigned int pad_width,
                               int act_max, int act_min) {
    Q_AVEPOOLING_TFLITE(input, output, input_height, input_width, output_height, output_width, kernel_height, \
                      kernel_width, stride_height, stride_width, pad_height, pad_width, act_max, act_min)
})

#undef QUANTIZED_AVE
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
#define QUANTIZED_AVE(quantized_avepooling, pool_size) \
    (quantized_avepooling > 0 \
        ? (quantized_avepooling + pool_size / 2) / pool_size \
        : (quantized_avepooling - pool_size / 2) / pool_size)

ADD_SINGLE_KERNEL(SIGNEDavepooling_tflite_INT8, (__global DATA_T *input,
                               __global DATA_T *output,
                               unsigned int input_height, unsigned int input_width,
                               unsigned int output_height, unsigned int output_width,
                               unsigned int kernel_height, unsigned int kernel_width,
                               unsigned int stride_height, unsigned int stride_width,
                               unsigned int pad_height, unsigned int pad_width,
                               int act_max, int act_min) {
    Q_AVEPOOLING_TFLITE(input, output, input_height, input_width, output_height, output_width, kernel_height, \
                      kernel_width, stride_height, stride_width, pad_height, pad_width, act_max, act_min)
})

#undef QUANTIZED_AVE
#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // char

#define AVEPOOLING_CAFFE(input, aveOutput, inputHeight, inputWidth, outputHeight, outputWidth, kernelHeight, \
                         kernelWidth, strideHeight, strideWidth, padHeight, padWidth) \
    if (get_global_id(2) < outputHeight * outputWidth) { \
        DATA_T avepooling = 0.0f; \
        int start_i = get_global_id(2) / outputWidth * strideHeight - padHeight; \
        int start_j = get_global_id(2) % outputWidth * strideWidth - padWidth; \
        int end_i = min(start_i + kernelHeight, inputHeight + padHeight); \
        int end_j = min(start_j + kernelWidth, inputWidth + padWidth); \
        int pool_size = (end_i - start_i) * (end_j - start_j); \
        start_i = max(start_i, 0); \
        start_j = max(start_j, 0); \
        end_i = min(end_i, (int)inputHeight); \
        end_j = min(end_j, (int)inputWidth); \
        int inputBase = inputHeight * inputWidth * \
                        (get_global_id(0) * get_global_size(1) + get_global_id(1)); \
        for (int i = start_i; i < end_i; i++) { \
            for (int j = start_j; j < end_j; j++) { \
                avepooling += input[inputBase + inputWidth * i + j]; \
            } \
        } \
        aveOutput[(get_global_id(0) * get_global_size(1) + get_global_id(1)) * outputHeight * \
                        outputWidth + \
                    get_global_id(2)] = avepooling / pool_size; \
    }

#define AVEPOOLING_TFLITE(input, output, input_height, input_width, output_height, output_width, kernel_height, \
                          kernel_width, stride_height, stride_width, pad_height, pad_width) \
        if (get_global_id(2) < output_height * output_width) { \
            float quantized_avepooling = 0.0f; \
            int start_i = get_global_id(2) / output_width * stride_height - pad_height; \
            int start_j = get_global_id(2) % output_width * stride_width - pad_width; \
            int end_i = min(start_i + kernel_height, input_height); \
            int end_j = min(start_j + kernel_width, input_width); \
            start_i = max(start_i, 0); \
            start_j = max(start_j, 0); \
            end_i = min(end_i, (int)input_height); \
            end_j = min(end_j, (int)input_width); \
            int pool_size = (end_i - start_i) * (end_j - start_j); \
            int input_base = input_height * input_width * \
                             (get_global_id(0) * get_global_size(1) + get_global_id(1)); \
            for (int i = start_i; i < end_i; i++) { \
                for (int j = start_j; j < end_j; j++) { \
                    quantized_avepooling += input[input_base + input_width * i + j]; \
                } \
            } \
            output[(get_global_id(0) * get_global_size(1) + get_global_id(1)) * output_height * \
                       output_width + \
                   get_global_id(2)] = (DATA_T)quantized_avepooling / pool_size; \
        }

#define AVEPOOLING_TFLITE_TEXTURE2D(src_data, dst_data, src_x, src_y, src_z, src_w, dst_x, dst_y, dst_z, dst_w, kw, \
                                    kh, stride_x, stride_y, padding_x, padding_y) \
        int X = get_global_id(0); \
        int Y = get_global_id(1); \
        int Z = get_global_id(2); \
        if (X >= dst_x || Y >= dst_y || Z >= dst_z) \
            return; \
        float4 r = (float4)(0.0f); \
        float window_size = 0.0f; \
        int xs = X * stride_x + padding_x; \
        int ys = Y * stride_y + padding_y; \
        for (int ky = 0; ky < kh; ++ky) { \
            int y_c = ys + ky; \
            bool outside_y = y_c < 0 || y_c >= src_y; \
            for (int kx = 0; kx < kw; ++kx) { \
                int x_c = xs + kx; \
                bool outside = outside_y || x_c < 0 || x_c >= src_x; \
                r += read_imagef(src_data, smp_zero, (int2)((x_c), (y_c)*src_z + (Z))); \
                window_size += !outside ? 1.0f : 0.0f; \
            } \
        } \
        DATA_T4 result = CONVERT_TO_DATA_T4(r / window_size); \
        WRITE_IMAGE_T(dst_data, (int2)((X), (Y)*dst_z + (Z)), result);


/********  FP32 KERNELS ********/
#define DATA_T float
#define DATA_T2 float2
#define DATA_T3 float3
#define DATA_T4 float4
#define DATA_T8 float8
#define DATA_T16 float16
#define CONVERT_TO_DATA_T(x) (DATA_T)x
#define CONVERT_TO_DATA_T4(x) convert_float4(x)
#define READ_IMAGE_T(x, SAMP, COORD) read_imagef(x, SAMP, COORD)
#define WRITE_IMAGE_T(y, COORD, x) write_imagef(y, COORD, x)

ADD_SINGLE_KERNEL(avepooling_caffe_FP32, (__global DATA_T *input,
                                        __global DATA_T *aveOutput,
                                        unsigned int inputHeight,
                                        unsigned int inputWidth,
                                        unsigned int outputHeight,
                                        unsigned int outputWidth,
                                        unsigned int kernelHeight,
                                        unsigned int kernelWidth,
                                        unsigned int strideHeight,
                                        unsigned int strideWidth,
                                        unsigned int padHeight,
                                        unsigned int padWidth) {
    AVEPOOLING_CAFFE(input, aveOutput, inputHeight, inputWidth, outputHeight, outputWidth, kernelHeight, \
        kernelWidth, strideHeight, strideWidth, padHeight, padWidth)
})

ADD_SINGLE_KERNEL(avepooling_tflite_FP32, (__global DATA_T *input,
                                         __global DATA_T *output,
                                         unsigned int input_height,
                                         unsigned int input_width,
                                         unsigned int output_height,
                                         unsigned int output_width,
                                         unsigned int kernel_height,
                                         unsigned int kernel_width,
                                         unsigned int stride_height,
                                         unsigned int stride_width,
                                         unsigned int pad_height,
                                         unsigned int pad_width) {
    AVEPOOLING_TFLITE(input, output, input_height, input_width, output_height, output_width, kernel_height, \
        kernel_width, stride_height, stride_width, pad_height, pad_width)
})

ADD_SINGLE_KERNEL(avepooling_tflite_texture2d_FP32, (__read_only image2d_t src_data,
                                                   __write_only image2d_t dst_data,
                                                   int src_x,
                                                   int src_y,
                                                   int src_z,
                                                   int src_w,
                                                   int dst_x,
                                                   int dst_y,
                                                   int dst_z,
                                                   int dst_w,
                                                   int kw,
                                                   int kh,
                                                   int stride_x,
                                                   int stride_y,
                                                   int padding_x,
                                                   int padding_y) {
    AVEPOOLING_TFLITE_TEXTURE2D(src_data, dst_data, src_x, src_y, src_z, src_w, dst_x, dst_y, dst_z, dst_w, kw, \
        kh, stride_x, stride_y, padding_x, padding_y)
})

ADD_SINGLE_KERNEL(avepooling_caffe_big_kernelsize_FP16, (__global half *input,
                                                       __global half *aveOutput,
                                                       unsigned int inputHeight,
                                                       unsigned int inputWidth,
                                                       unsigned int outputHeight,
                                                       unsigned int outputWidth,
                                                       unsigned int kernelHeight,
                                                       unsigned int kernelWidth,
                                                       unsigned int strideHeight,
                                                       unsigned int strideWidth,
                                                       unsigned int padHeight,
                                                       unsigned int padWidth) {
    AVEPOOLING_CAFFE(input, aveOutput, inputHeight, inputWidth, outputHeight, outputWidth, kernelHeight, \
        kernelWidth, strideHeight, strideWidth, padHeight, padWidth)
})

#undef WRITE_IMAGE_T
#undef READ_IMAGE_T
#undef CONVERT_TO_DATA_T4
#undef CONVERT_TO_DATA_T
#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // float


/********  FP16 KERNELS ********/
#define DATA_T half
#define DATA_T2 half2
#define DATA_T3 half3
#define DATA_T4 half4
#define DATA_T8 half8
#define DATA_T16 half16
#define CONVERT_TO_DATA_T(x) (DATA_T)x
#define CONVERT_TO_DATA_T4(x) convert_half4(x)
#define READ_IMAGE_T(x, SAMP, COORD) read_imageh(x, SAMP, COORD)
#define WRITE_IMAGE_T(y, COORD, x) write_imageh(y, COORD, x)

ADD_SINGLE_KERNEL(avepooling_caffe_FP16, (__global DATA_T *input,
                                        __global DATA_T *aveOutput,
                                        unsigned int inputHeight,
                                        unsigned int inputWidth,
                                        unsigned int outputHeight,
                                        unsigned int outputWidth,
                                        unsigned int kernelHeight,
                                        unsigned int kernelWidth,
                                        unsigned int strideHeight,
                                        unsigned int strideWidth,
                                        unsigned int padHeight,
                                        unsigned int padWidth) {
    AVEPOOLING_CAFFE(input, aveOutput, inputHeight, inputWidth, outputHeight, outputWidth, kernelHeight, \
        kernelWidth, strideHeight, strideWidth, padHeight, padWidth)
})

ADD_SINGLE_KERNEL(avepooling_tflite_FP16, (__global DATA_T *input,
                                         __global DATA_T *output,
                                         unsigned int input_height,
                                         unsigned int input_width,
                                         unsigned int output_height,
                                         unsigned int output_width,
                                         unsigned int kernel_height,
                                         unsigned int kernel_width,
                                         unsigned int stride_height,
                                         unsigned int stride_width,
                                         unsigned int pad_height,
                                         unsigned int pad_width) {
    AVEPOOLING_TFLITE(input, output, input_height, input_width, output_height, output_width, kernel_height, \
        kernel_width, stride_height, stride_width, pad_height, pad_width)
})

ADD_SINGLE_KERNEL(avepooling_tflite_texture2d_FP16, (__read_only image2d_t src_data,
                                                   __write_only image2d_t dst_data,
                                                   int src_x,
                                                   int src_y,
                                                   int src_z,
                                                   int src_w,
                                                   int dst_x,
                                                   int dst_y,
                                                   int dst_z,
                                                   int dst_w,
                                                   int kw,
                                                   int kh,
                                                   int stride_x,
                                                   int stride_y,
                                                   int padding_x,
                                                   int padding_y) {
    AVEPOOLING_TFLITE_TEXTURE2D(src_data, dst_data, src_x, src_y, src_z, src_w, dst_x, dst_y, dst_z, dst_w, kw, kh, \
        stride_x, stride_y, padding_x, padding_y)
})

#undef WRITE_IMAGE_T
#undef READ_IMAGE_T
#undef CONVERT_TO_DATA_T4
#undef CONVERT_TO_DATA_T
#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // half

}  // namespace gpu
}  // namespace ud
}  // namespace enn
