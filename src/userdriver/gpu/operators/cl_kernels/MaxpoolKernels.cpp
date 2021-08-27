#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define MAXPOOLING(input, maxOutput, inputHeight, inputWidth, kernelHeight, kernelWidth, strideHeight, \
                   strideWidth, padHeight, padWidth, width, height) \
    int globalID0 = get_global_id(0); \
    int globalID1 = get_global_id(1); \
    int globalID2 = get_global_id(2); \
    if (globalID2 < height * width) { \
        { \
            int start_i = globalID2 / width * strideHeight - padHeight; \
            int start_j = globalID2 % width * strideWidth - padWidth; \
            int end_i = min(start_i + kernelHeight, inputHeight); \
            int end_j = min(start_j + kernelWidth, inputWidth); \
            start_i = max(start_i, 0); \
            start_j = max(start_j, 0); \
            DATA_T maxpooling = DATA_T_MINIMUM; \
            int inputBase = inputHeight * inputWidth * (globalID0 * get_global_size(1) + globalID1); \
            for (int i = start_i; i < end_i; i++) { \
                for (int j = start_j; j < end_j; j++) { \
                    maxpooling = fmax(input[inputBase + inputWidth * i + j], maxpooling); \
                } \
            } \
            maxOutput[(globalID0 * get_global_size(1) + globalID1) * height * width + globalID2] = \
                maxpooling; \
        } \
    }

#define Q_MAXPOOLING(input, output, input_height, input_width, kernel_height, kernel_width, stride_height, \
                    stride_width, pad_height, pad_width, width, height, act_max, act_min) \
    if (get_global_id(2) < height * width) { \
        { \
            int quantized_maxpooling = DATA_T_MINIMUM; \
            int start_i = get_global_id(2) / width * stride_height - pad_height; \
            int start_j = get_global_id(2) % width * stride_width - pad_width; \
            int end_i = min(start_i + kernel_height, input_height); \
            int end_j = min(start_j + kernel_width, input_width); \
            start_i = max(start_i, 0); \
            start_j = max(start_j, 0); \
            int input_base = input_height * input_width * \
                             (get_global_id(0) * get_global_size(1) + get_global_id(1)); \
            for (int i = start_i; i < end_i; i++) { \
                for (int j = start_j; j < end_j; j++) { \
                    quantized_maxpooling = \
                        max((int)input[input_base + input_width * i + j], quantized_maxpooling); \
                } \
            } \
            quantized_maxpooling = max(quantized_maxpooling, act_min); \
            quantized_maxpooling = min(quantized_maxpooling, act_max); \
            output[(get_global_id(0) * get_global_size(1) + get_global_id(1)) * height * width + \
                   get_global_id(2)] = (DATA_T)quantized_maxpooling; \
        } \
    }

/********  QUANTIZED KERNELS ********/
#define DATA_T uchar
#define DATA_T2 uchar2
#define DATA_T3 uchar3
#define DATA_T4 uchar4
#define DATA_T8 uchar8
#define DATA_T16 uchar16
#define DATA_T_MINIMUM 0

ADD_SINGLE_KERNEL(maxpooling_INT8, (__global DATA_T *input,
                                    __global DATA_T *output,
                                    unsigned int input_height,
                                    unsigned int input_width,
                                    unsigned int kernel_height,
                                    unsigned int kernel_width,
                                    unsigned int stride_height,
                                    unsigned int stride_width,
                                    unsigned int pad_height,
                                    unsigned int pad_width,
                                    unsigned int width,
                                    unsigned int height,
                                    int act_max,
                                    int act_min) {
    Q_MAXPOOLING(input, output, input_height, input_width, kernel_height, kernel_width, stride_height, stride_width, \
               pad_height, pad_width, width, height, act_max, act_min)
})

#undef DATA_T_MINIMUM
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
#define DATA_T_MINIMUM (-127 - 1)
ADD_SINGLE_KERNEL(SIGNEDmaxpooling_INT8, (__global DATA_T *input,
                                          __global DATA_T *output,
                                          unsigned int input_height,
                                          unsigned int input_width,
                                          unsigned int kernel_height,
                                          unsigned int kernel_width,
                                          unsigned int stride_height,
                                          unsigned int stride_width,
                                          unsigned int pad_height,
                                          unsigned int pad_width,
                                          unsigned int width,
                                          unsigned int height,
                                          int act_max,
                                          int act_min) {
    Q_MAXPOOLING(input, output, input_height, input_width, kernel_height, kernel_width, stride_height, stride_width, \
               pad_height, pad_width, width, height, act_max, act_min)
})
#undef DATA_T_MINIMUM
#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // char

#define DATA_T float
#define DATA_T_MINIMUM FLOAT_MIN
ADD_SINGLE_KERNEL(maxpooling_FP32, (__global DATA_T *input,
                                    __global DATA_T *maxOutput,
                                    unsigned int inputHeight,
                                    unsigned int inputWidth,
                                    unsigned int kernelHeight,
                                    unsigned int kernelWidth,
                                    unsigned int strideHeight,
                                    unsigned int strideWidth,
                                    unsigned int padHeight,
                                    unsigned int padWidth,
                                    unsigned int width,
                                    unsigned int height) {
    MAXPOOLING(input, maxOutput, inputHeight, inputWidth, kernelHeight, kernelWidth, strideHeight, \
               strideWidth, padHeight, padWidth, width, height)
})
#undef DATA_T_MINIMUM
#undef DATA_T  // float

#define DATA_T half
#define DATA_T_MINIMUM HALF_MIN
ADD_SINGLE_KERNEL(maxpooling_FP16, (__global DATA_T *input,
                                    __global DATA_T *maxOutput,
                                    unsigned int inputHeight,
                                    unsigned int inputWidth,
                                    unsigned int kernelHeight,
                                    unsigned int kernelWidth,
                                    unsigned int strideHeight,
                                    unsigned int strideWidth,
                                    unsigned int padHeight,
                                    unsigned int padWidth,
                                    unsigned int width,
                                    unsigned int height) {
    MAXPOOLING(input, maxOutput, inputHeight, inputWidth, kernelHeight, kernelWidth, strideHeight, \
               strideWidth, padHeight, padWidth, width, height)
})
#undef DATA_T_MINIMUM
#undef DATA_T  // half

}  // namespace gpu
}  // namespace ud
}  // namespace enn
