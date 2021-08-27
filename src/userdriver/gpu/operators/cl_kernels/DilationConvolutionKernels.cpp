#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-escape-sequence"

#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"
namespace enn{
namespace ud {
namespace gpu {

#define DILATION_CONV_K3D8P8_4X8(input, weight, bias, output, padT, padL, outputNumber, outputChannel, outputHeight, outputWidth, inputNum, inputChannel, inputHeight, inputWidth) \
        int outputN = get_global_id(2) / ((outputChannel + 3) / 4); \
        int outputC = get_global_id(2) % ((outputChannel + 3) / 4) * 4; \
        int outputH = get_global_id(1); \
        int outputW = get_global_id(0) * 8; \
        if (outputW < outputWidth && outputH < outputHeight) { \
            DATA_T4 output0 = (DATA_T4)0.0f; \
            DATA_T4 output1 = (DATA_T4)0.0f; \
            DATA_T4 output2 = (DATA_T4)0.0f; \
            DATA_T4 output3 = (DATA_T4)0.0f; \
            DATA_T4 output4 = (DATA_T4)0.0f; \
            DATA_T4 output5 = (DATA_T4)0.0f; \
            DATA_T4 output6 = (DATA_T4)0.0f; \
            DATA_T4 output7 = (DATA_T4)0.0f; \
            int inputHStart = (outputH - padT < 0) ? (outputH - padT + 8) : (outputH - padT); \
            int inputHEnd = (outputH - padT + 17 > inputHeight) ? (outputH - padT + 17 - 8) : (outputH - padT + 17); \
            int inputW = outputW - padL; \
            DATA_T16 inputMask_0 = (DATA_T16)1.0f; \
            DATA_T8  inputMask_1 = (DATA_T8)1.0f; \
            if (inputW < 0 || inputW >= inputWidth) { \
                inputMask_0.s0 = 0.0f; \
            } \
            if (inputW + 1 < 0 || inputW + 1 >= inputWidth) { \
                inputMask_0.s1 = 0.0f; \
            } \
            if (inputW + 2 < 0 || inputW + 2 >= inputWidth) { \
                inputMask_0.s2 = 0.0f; \
            } \
            if (inputW + 3 < 0 || inputW + 3 >= inputWidth) { \
                inputMask_0.s3 = 0.0f; \
            } \
            if (inputW + 4 < 0 || inputW + 4 >= inputWidth) { \
                inputMask_0.s4 = 0.0f; \
            } \
            if (inputW + 5 < 0 || inputW + 5 >= inputWidth) { \
                inputMask_0.s5 = 0.0f; \
            } \
            if (inputW + 6 < 0 || inputW + 6 >= inputWidth) { \
                inputMask_0.s6 = 0.0f; \
            } \
            if (inputW + 7 < 0 || inputW + 7 >= inputWidth) { \
                inputMask_0.s7 = 0.0f; \
            } \
            if (inputW + 8 < 0 || inputW + 8 >= inputWidth) { \
                inputMask_0.s8 = 0.0f; \
            } \
            if (inputW + 9 < 0 || inputW + 9 >= inputWidth) { \
                inputMask_0.s9 = 0.0f; \
            } \
            if (inputW + 10 < 0 || inputW + 10 >= inputWidth) { \
                inputMask_0.sa = 0.0f; \
            } \
            if (inputW + 11 < 0 || inputW + 11 >= inputWidth) { \
                inputMask_0.sb = 0.0f; \
            } \
            if (inputW + 12 < 0 || inputW + 12 >= inputWidth) { \
                inputMask_0.sc = 0.0f; \
            } \
            if (inputW + 13 < 0 || inputW + 13 >= inputWidth) { \
                inputMask_0.sd = 0.0f; \
            } \
            if (inputW + 14 < 0 || inputW + 14 >= inputWidth) { \
                inputMask_0.se = 0.0f; \
            } \
            if (inputW + 15 < 0 || inputW + 15 >= inputWidth) { \
                inputMask_0.sf = 0.0f; \
            } \
            if (inputW + 16 < 0 || inputW + 16 >= inputWidth) { \
                inputMask_1.s0 = 0.0f; \
            } \
            if (inputW + 17 < 0 || inputW + 17 >= inputWidth) { \
                inputMask_1.s1 = 0.0f; \
            } \
            if (inputW + 18 < 0 || inputW + 18 >= inputWidth) { \
                inputMask_1.s2 = 0.0f; \
            } \
            if (inputW + 19 < 0 || inputW + 19 >= inputWidth) { \
               inputMask_1.s3 = 0.0f; \
            } \
            if (inputW + 20 < 0 || inputW + 20 >= inputWidth) { \
                inputMask_1.s4 = 0.0f; \
            } \
            if (inputW + 21 < 0 || inputW + 21 >= inputWidth) { \
                inputMask_1.s5 = 0.0f; \
            } \
            if (inputW + 22 < 0 || inputW + 22 >= inputWidth) { \
                inputMask_1.s6 = 0.0f; \
            } \
            if (inputW + 23 < 0 || inputW + 23 >= inputWidth) { \
                inputMask_1.s7 = 0.0f; \
            } \
            int inputBase = outputN * inputChannel * inputHeight * inputWidth + inputW; \
            int khOffset = (outputH - padT < 0) ? 1 : 0; \
            int weightBase = outputC * inputChannel * 3 * 3 + khOffset * 4 * 3; \
            DATA_T16 intputVector; \
            DATA_T16 weightVector; \
            for (int inputC = 0; inputC < inputChannel; inputC++) { \
                int inputIndex = inputBase + inputC * inputHeight * inputWidth; \
                int weightIndex = weightBase + inputC * 4 * 3 * 3; \
                for (int kh = inputHStart; kh < inputHEnd; kh += 8) { \
                    intputVector = vload16(0, input + inputIndex + kh * inputWidth); \
                    intputVector = intputVector * inputMask_0; \
                    weightVector = vload16(0, weight + weightIndex); \
                    weightIndex += 12; \
                    output0 = mad(intputVector.s0, weightVector.lo.lo, output0); \
                    output1 = mad(intputVector.s1, weightVector.lo.lo, output1); \
                    output2 = mad(intputVector.s2, weightVector.lo.lo, output2); \
                    output3 = mad(intputVector.s3, weightVector.lo.lo, output3); \
                    output4 = mad(intputVector.s4, weightVector.lo.lo, output4); \
                    output5 = mad(intputVector.s5, weightVector.lo.lo, output5); \
                    output6 = mad(intputVector.s6, weightVector.lo.lo, output6); \
                    output7 = mad(intputVector.s7, weightVector.lo.lo, output7); \
                    output0 = mad(intputVector.s8, weightVector.lo.hi, output0); \
                    output1 = mad(intputVector.s9, weightVector.lo.hi, output1); \
                    output2 = mad(intputVector.sa, weightVector.lo.hi, output2); \
                    output3 = mad(intputVector.sb, weightVector.lo.hi, output3); \
                    output4 = mad(intputVector.sc, weightVector.lo.hi, output4); \
                    output5 = mad(intputVector.sd, weightVector.lo.hi, output5); \
                    output6 = mad(intputVector.se, weightVector.lo.hi, output6); \
                    output7 = mad(intputVector.sf, weightVector.lo.hi, output7); \
                    intputVector.lo = vload8(0, input + inputIndex + kh * inputWidth + 16); \
                    intputVector.lo = intputVector.lo * inputMask_1; \
                    output0 = mad(intputVector.s0, weightVector.hi.lo, output0); \
                    output1 = mad(intputVector.s1, weightVector.hi.lo, output1); \
                    output2 = mad(intputVector.s2, weightVector.hi.lo, output2); \
                    output3 = mad(intputVector.s3, weightVector.hi.lo, output3); \
                    output4 = mad(intputVector.s4, weightVector.hi.lo, output4); \
                    output5 = mad(intputVector.s5, weightVector.hi.lo, output5); \
                    output6 = mad(intputVector.s6, weightVector.hi.lo, output6); \
                    output7 = mad(intputVector.s7, weightVector.hi.lo, output7); \
                } \
            } \
            int outputIndex = outputN * outputChannel * outputHeight * outputWidth + \
                              outputC * outputHeight * outputWidth + outputH * outputWidth + \
                              outputW; \
            DATA_T4 biasVector = vload4(0, bias + outputC); \
            output0 = ACT_VEC_F(DATA_T4, output0 + biasVector); \
            output1 = ACT_VEC_F(DATA_T4, output1 + biasVector); \
            output2 = ACT_VEC_F(DATA_T4, output2 + biasVector); \
            output3 = ACT_VEC_F(DATA_T4, output3 + biasVector); \
            output4 = ACT_VEC_F(DATA_T4, output4 + biasVector); \
            output5 = ACT_VEC_F(DATA_T4, output5 + biasVector); \
            output6 = ACT_VEC_F(DATA_T4, output6 + biasVector); \
            output7 = ACT_VEC_F(DATA_T4, output7 + biasVector); \
            if (outputW + 7 < outputWidth) { \
                if (outputC + 3 < outputChannel) { \
                    vstore8((DATA_T8)(output0.s0, \
                                    output1.s0, \
                                    output2.s0, \
                                    output3.s0, \
                                    output4.s0, \
                                    output5.s0, \
                                    output6.s0, \
                                    output7.s0), \
                            0, \
                            output + outputIndex); \
                    vstore8((DATA_T8)(output0.s1, \
                                    output1.s1, \
                                    output2.s1, \
                                    output3.s1, \
                                    output4.s1, \
                                    output5.s1, \
                                    output6.s1, \
                                    output7.s1), \
                            0, \
                            output + outputIndex + outputHeight * outputWidth); \
                    vstore8((DATA_T8)(output0.s2, \
                                    output1.s2, \
                                    output2.s2, \
                                    output3.s2, \
                                    output4.s2, \
                                    output5.s2, \
                                    output6.s2, \
                                    output7.s2), \
                            0, \
                            output + outputIndex + 2 * outputHeight * outputWidth); \
                    vstore8((DATA_T8)(output0.s3, \
                                    output1.s3, \
                                    output2.s3, \
                                    output3.s3, \
                                    output4.s3, \
                                    output5.s3, \
                                    output6.s3, \
                                    output7.s3), \
                            0, \
                            output + outputIndex + 3 * outputHeight * outputWidth); \
                } else if (outputC + 2 < outputChannel) { \
                    vstore8((DATA_T8)(output0.s0, \
                                    output1.s0, \
                                    output2.s0, \
                                    output3.s0, \
                                    output4.s0, \
                                    output5.s0, \
                                    output6.s0, \
                                    output7.s0), \
                            0, \
                            output + outputIndex); \
                    vstore8((DATA_T8)(output0.s1, \
                                    output1.s1, \
                                    output2.s1, \
                                    output3.s1, \
                                    output4.s1, \
                                    output5.s1, \
                                    output6.s1, \
                                    output7.s1), \
                            0, \
                            output + outputIndex + outputHeight * outputWidth); \
                    vstore8((DATA_T8)(output0.s2, \
                                    output1.s2, \
                                    output2.s2, \
                                    output3.s2, \
                                    output4.s2, \
                                    output5.s2, \
                                    output6.s2, \
                                    output7.s2), \
                            0, \
                            output + outputIndex + 2 * outputHeight * outputWidth); \
                } else if (outputC + 1 < outputChannel) { \
                    vstore8((DATA_T8)(output0.s0, \
                                    output1.s0, \
                                    output2.s0, \
                                    output3.s0, \
                                    output4.s0, \
                                    output5.s0, \
                                    output6.s0, \
                                    output7.s0), \
                            0, \
                            output + outputIndex); \
                    vstore8((DATA_T8)(output0.s1, \
                                    output1.s1, \
                                    output2.s1, \
                                    output3.s1, \
                                    output4.s1, \
                                    output5.s1, \
                                    output6.s1, \
                                    output7.s1), \
                            0, \
                            output + outputIndex + outputHeight * outputWidth); \
                } else if (outputC < outputChannel) { \
                    vstore8((DATA_T8)(output0.s0, \
                                    output1.s0, \
                                    output2.s0, \
                                    output3.s0, \
                                    output4.s0, \
                                    output5.s0, \
                                    output6.s0, \
                                    output7.s0), \
                            0, \
                            output + outputIndex); \
                } \
            } \
        }

/********  FP16 KERNELS ********/
#define DATA_T half
#define DATA_T2 half2
#define DATA_T3 half3
#define DATA_T4 half4
#define DATA_T8 half8
#define DATA_T16 half16

// kernels without activation func
#define ACT_VEC_F(VEC_T, x) x
ADD_SINGLE_KERNEL(dilation_conv_k3d8p8_4x8_FP16, (__global const DATA_T *input,
                                                  __global const DATA_T *weight,
                                                  __global const DATA_T *bias,
                                                  __global DATA_T *output,
                                                  int padT,
                                                  int padL,
                                                  int outputNumber,
                                                  int outputChannel,
                                                  int outputHeight,
                                                  int outputWidth,
                                                  int inputNum,
                                                  int inputChannel,
                                                  int inputHeight,
                                                  int inputWidth) {
     DILATION_CONV_K3D8P8_4X8(input, weight, bias, output, padT, padL, outputNumber, outputChannel, outputHeight,
                              outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

#undef ACT_VEC_F  // NONE

// apply RELU to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELUdilation_conv_k3d8p8_4x8_FP16, (__global const DATA_T *input,
                                                      __global const DATA_T *weight,
                                                      __global const DATA_T *bias,
                                                      __global DATA_T *output,
                                                      int padT,
                                                      int padL,
                                                      int outputNumber,
                                                      int outputChannel,
                                                      int outputHeight,
                                                      int outputWidth,
                                                      int inputNum,
                                                      int inputChannel,
                                                      int inputHeight,
                                                      int inputWidth) {
     DILATION_CONV_K3D8P8_4X8(input, weight, bias, output, padT, padL, outputNumber, outputChannel, outputHeight,
                              outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})
#undef ACT_VEC_F  // RELU

// apply RELU6 to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU6_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELU6dilation_conv_k3d8p8_4x8_FP16, (__global const DATA_T *input,
                                                       __global const DATA_T *weight,
                                                       __global const DATA_T *bias,
                                                       __global DATA_T *output,
                                                       int padT,
                                                       int padL,
                                                       int outputNumber,
                                                       int outputChannel,
                                                       int outputHeight,
                                                       int outputWidth,
                                                       int inputNum,
                                                       int inputChannel,
                                                       int inputHeight,
                                                       int inputWidth) {
     DILATION_CONV_K3D8P8_4X8(input, weight, bias, output, padT, padL, outputNumber, outputChannel, outputHeight,
                              outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})
#undef ACT_VEC_F  // RELU6

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

// kernels without activation func
#define ACT_VEC_F(VEC_T, x) x
ADD_SINGLE_KERNEL(dilation_conv_k3d8p8_4x8_FP32, (__global const DATA_T *input,
                                                  __global const DATA_T *weight,
                                                  __global const DATA_T *bias,
                                                  __global DATA_T *output,
                                                  int padT,
                                                  int padL,
                                                  int outputNumber,
                                                  int outputChannel,
                                                  int outputHeight,
                                                  int outputWidth,
                                                  int inputNum,
                                                  int inputChannel,
                                                  int inputHeight,
                                                  int inputWidth) {
     DILATION_CONV_K3D8P8_4X8(input, weight, bias, output, padT, padL, outputNumber, outputChannel, outputHeight,
                              outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

#undef ACT_VEC_F  // NONE

// apply RELU to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELUdilation_conv_k3d8p8_4x8_FP32, (__global const DATA_T *input,
                                                      __global const DATA_T *weight,
                                                      __global const DATA_T *bias,
                                                      __global DATA_T *output,
                                                      int padT,
                                                      int padL,
                                                      int outputNumber,
                                                      int outputChannel,
                                                      int outputHeight,
                                                      int outputWidth,
                                                      int inputNum,
                                                      int inputChannel,
                                                      int inputHeight,
                                                      int inputWidth) {
     DILATION_CONV_K3D8P8_4X8(input, weight, bias, output, padT, padL, outputNumber, outputChannel, outputHeight,
                              outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})
#undef ACT_VEC_F  // RELU

// apply RELU6 to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU6_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELU6dilation_conv_k3d8p8_4x8_FP32, (__global const DATA_T *input,
                                                       __global const DATA_T *weight,
                                                       __global const DATA_T *bias,
                                                       __global DATA_T *output,
                                                       int padT,
                                                       int padL,
                                                       int outputNumber,
                                                       int outputChannel,
                                                       int outputHeight,
                                                       int outputWidth,
                                                       int inputNum,
                                                       int inputChannel,
                                                       int inputHeight,
                                                       int inputWidth) {
     DILATION_CONV_K3D8P8_4X8(input, weight, bias, output, padT, padL, outputNumber, outputChannel, outputHeight,
                              outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})
#undef ACT_VEC_F  // RELU6
#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // float

}
}
}