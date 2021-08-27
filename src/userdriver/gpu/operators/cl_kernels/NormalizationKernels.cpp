#include "userdriver/gpu/common/CLKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define NORMALIZATION_UINT8(input, mean, output, inputWidth, inputHeight, spaceW, spaceH, channelNum, \
                            outputWidth, outputHeight, scale) \
    int globalID0 = get_global_id(0); \
    int globalID1 = get_global_id(1); \
    int globalID2 = get_global_id(2); \
    if (globalID2 < outputWidth) { \
        int inputIndex = globalID0 * inputHeight * inputWidth + (globalID1 + spaceH) * inputWidth + \
                         spaceW + globalID2; \
        int outputIndex = \
            globalID0 * outputHeight * outputWidth + globalID1 * outputWidth + globalID2; \
        DATA_T f_input = CONVERT_TO_FLOAT(input[inputIndex]); \
        output[outputIndex] =  \
            CONVERT_TO_DATA_T(NORM_WITH_MEAN_STD(f_input, mean, scale, globalID0 % channelNum)); \
    }

#define DATA_T float
#define CONVERT_TO_FLOAT(x) x / 1.0f
#define CONVERT_TO_DATA_T(x) x
#define NORM_WITH_MEAN_STD(x, mean, std, idx) (x - mean[idx]) * std[idx]
ADD_SINGLE_KERNEL(normalization_uint8_FP32, (__global const unsigned char *input,
                                       __global const float *mean,
                                       __global float *output,
                                       unsigned int inputWidth,
                                       unsigned int inputHeight,
                                       unsigned int spaceW,
                                       unsigned int spaceH,
                                       unsigned int channelNum,
                                       unsigned int outputWidth,
                                       unsigned int outputHeight,
                                       __global const float *scale) {
    NORMALIZATION_UINT8(input, mean, output, inputWidth, inputHeight, spaceW, spaceH, channelNum, \
                        outputWidth, outputHeight, scale)
})
#undef NORM_WITH_MEAN_STD
#undef CONVERT_TO_DATA_T
#undef CONVERT_TO_FLOAT
#undef DATA_T

#define DATA_T float
#define CONVERT_TO_FLOAT(x) x
#define CONVERT_TO_DATA_T(x) x
#define NORM_WITH_MEAN_STD(x, mean, std, idx) (x - mean[idx]) * std[idx]

ADD_SINGLE_KERNEL(normalization_float_FP32, (__global const float *input,
                                       __global const float *mean,
                                       __global float *output,
                                       unsigned int inputWidth,
                                       unsigned int inputHeight,
                                       unsigned int spaceW,
                                       unsigned int spaceH,
                                       unsigned int channelNum,
                                       unsigned int outputWidth,
                                       unsigned int outputHeight,
                                       __global const float *scale) {
    NORMALIZATION_UINT8(input, mean, output, inputWidth, inputHeight, spaceW, spaceH, channelNum, \
                        outputWidth, outputHeight, scale)
})

ADD_SINGLE_KERNEL(normalization_float_for_fp16_FP32, (__global const float *input,
                                                __global const float *mean,
                                                __global float *output,
                                                unsigned int inputWidth,
                                                unsigned int inputHeight,
                                                unsigned int spaceW,
                                                unsigned int spaceH,
                                                unsigned int channelNum,
                                                unsigned int outputWidth,
                                                unsigned int outputHeight,
                                                __global const float *scale) {
    NORMALIZATION_UINT8(input, mean, output, inputWidth, inputHeight, spaceW, spaceH, channelNum, \
                        outputWidth, outputHeight, scale)
})
#undef NORM_WITH_MEAN_STD
#undef CONVERT_TO_DATA_T
#undef CONVERT_TO_FLOAT
#undef DATA_T

#define DATA_T half
#define CONVERT_TO_FLOAT(x) x / 1.0f
#define CONVERT_TO_DATA_T(x) x
#define NORM_WITH_MEAN_STD(x, mean, std, idx) (x - mean[idx]) * std[idx]
ADD_SINGLE_KERNEL(normalization_uint8_FP16, (__global const unsigned char *input,
                                       __global const half *mean,
                                       __global half *output,
                                       unsigned int inputWidth,
                                       unsigned int inputHeight,
                                       unsigned int spaceW,
                                       unsigned int spaceH,
                                       unsigned int channelNum,
                                       unsigned int outputWidth,
                                       unsigned int outputHeight,
                                       __global const half *scale) {
    NORMALIZATION_UINT8(input, mean, output, inputWidth, inputHeight, spaceW, spaceH, channelNum, \
                        outputWidth, outputHeight, scale)
})
#undef NORM_WITH_MEAN_STD
#undef CONVERT_TO_DATA_T
#undef CONVERT_TO_FLOAT
#undef DATA_T

#define DATA_T half
#define CONVERT_TO_FLOAT(x) x
#define CONVERT_TO_DATA_T(x) x
#define NORM_WITH_MEAN_STD(x, mean, std, idx) (x - mean[idx]) * std[idx]
ADD_SINGLE_KERNEL(normalization_float_FP16, (__global const half *input,
                                        __global const half *mean,
                                        __global half *output,
                                        unsigned int inputWidth,
                                        unsigned int inputHeight,
                                        unsigned int spaceW,
                                        unsigned int spaceH,
                                        unsigned int channelNum,
                                        unsigned int outputWidth,
                                        unsigned int outputHeight,
                                        __global const half *scale) {
    NORMALIZATION_UINT8(input, mean, output, inputWidth, inputHeight, spaceW, spaceH, channelNum, \
                        outputWidth, outputHeight, scale)
})
#undef NORM_WITH_MEAN_STD
#undef CONVERT_TO_DATA_T
#undef CONVERT_TO_FLOAT
#undef DATA_T

#define DATA_T float
#define CONVERT_TO_FLOAT(x) x
#define CONVERT_TO_DATA_T(x) (half)(x)
#define NORM_WITH_MEAN_STD(x, mean, std, idx) (x - mean[idx]) * std[idx]
ADD_SINGLE_KERNEL(normalization_float_for_fp16_FP16, (__global const float *input,
                                                __global const half *mean,
                                                __global half *output,
                                                unsigned int inputWidth,
                                                unsigned int inputHeight,
                                                unsigned int spaceW,
                                                unsigned int spaceH,
                                                unsigned int channelNum,
                                                unsigned int outputWidth,
                                                unsigned int outputHeight,
                                                __global const half *scale) {
    NORMALIZATION_UINT8(input, mean, output, inputWidth, inputHeight, spaceW, spaceH, channelNum, \
                        outputWidth, outputHeight, scale)
})
#undef NORM_WITH_MEAN_STD
#undef CONVERT_TO_DATA_T
#undef CONVERT_TO_FLOAT
#undef DATA_T

#define DATA_T char
#define CONVERT_TO_FLOAT(x) x / 1.0f
#define CONVERT_TO_DATA_T(x) x
#define NORM_WITH_MEAN_STD(x, ...) x

ADD_SINGLE_KERNEL(normalization_uint8_INT8, (__global const unsigned char *input,
                                                       __global const half *mean,
                                                       __global char *output,
                                                       unsigned int inputWidth,
                                                       unsigned int inputHeight,
                                                       unsigned int spaceW,
                                                       unsigned int spaceH,
                                                       unsigned int channelNum,
                                                       unsigned int outputWidth,
                                                       unsigned int outputHeight,
                                                       __global const half *scale) {
    NORMALIZATION_UINT8(input, mean, output, inputWidth, inputHeight, spaceW, spaceH, channelNum, \
                        outputWidth, outputHeight, scale)
})

ADD_SINGLE_KERNEL(normalization_float_INT8, (__global const half *input,
                                        __global const half *mean,
                                        __global half *output,
                                        unsigned int inputWidth,
                                        unsigned int inputHeight,
                                        unsigned int spaceW,
                                        unsigned int spaceH,
                                        unsigned int channelNum,
                                        unsigned int outputWidth,
                                        unsigned int outputHeight,
                                        __global const half *scale) {})

ADD_SINGLE_KERNEL(normalization_float_for_fp16_INT8, (__global const float *input,
                                                __global const half *mean,
                                                __global half *output,
                                                unsigned int inputWidth,
                                                unsigned int inputHeight,
                                                unsigned int spaceW,
                                                unsigned int spaceH,
                                                unsigned int channelNum,
                                                unsigned int outputWidth,
                                                unsigned int outputHeight,
                                                __global const half *scale){})
#undef NORM_WITH_MEAN_STD
#undef CONVERT_TO_DATA_T
#undef CONVERT_TO_FLOAT
#undef DATA_T

}  // namespace gpu
}  // namespace ud
}  // namespace enn
