#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"
namespace enn{
namespace ud {
namespace gpu {

#define ALIGN_WEIGHT_DIRECT(weight, alinged_weight, inputNumber, inputChannel, inputHeight, inputWidth, outputNumber,  \
                            outputChannel, outputHeight, outputWidth, align_size)                                      \
    int outputW = get_global_id(0);                                                                                    \
    int outputH = get_global_id(1);                                                                                    \
    int outputC = get_global_id(2) % outputChannel;                                                                    \
    int outputN = get_global_id(2) / outputChannel;                                                                    \
    int inputW = outputW / align_size;                                                                                 \
    int inputH = outputH;                                                                                              \
    int inputC = outputC;                                                                                              \
    int inputN = outputN * align_size + outputW % align_size;                                                          \
    if (inputW < inputWidth && inputH < inputHeight && inputC < inputChannel && inputN < inputNumber) {                \
        alinged_weight[outputN * outputChannel * outputHeight * outputWidth + outputC * outputHeight * outputWidth +   \
                       outputH * outputWidth + outputW] =                                                              \
            weight[inputN * inputChannel * inputHeight * inputWidth + inputC * inputHeight * inputWidth +              \
                   inputH * inputWidth + inputW];                                                                      \
    } else {                                                                                                           \
        alinged_weight[outputN * outputChannel * outputHeight * outputWidth + outputC * outputHeight * outputWidth +   \
                       outputH * outputWidth + outputW] = 0.0f;                                                        \
    }

#define SET_DIRECT_INPUT_MASK16(inputW, inputWidth, inputMask) \
    if (inputW < 0 || inputW >= inputWidth) { inputMask.s0 = 0.0f; } \
    if (inputW + 1 < 0 || inputW + 1 >= inputWidth) { inputMask.s1 = 0.0f; } \
    if (inputW + 2 < 0 || inputW + 2 >= inputWidth) { inputMask.s2 = 0.0f; } \
    if (inputW + 3 < 0 || inputW + 3 >= inputWidth) { inputMask.s3 = 0.0f; } \
    if (inputW + 4 < 0 || inputW + 4 >= inputWidth) { inputMask.s4 = 0.0f; } \
    if (inputW + 5 < 0 || inputW + 5 >= inputWidth) { inputMask.s5 = 0.0f; } \
    if (inputW + 6 < 0 || inputW + 6 >= inputWidth) { inputMask.s6 = 0.0f; } \
    if (inputW + 7 < 0 || inputW + 7 >= inputWidth) { inputMask.s7 = 0.0f; } \
    if (inputW + 8 < 0 || inputW + 8 >= inputWidth) { inputMask.s8 = 0.0f; } \
    if (inputW + 9 < 0 || inputW + 9 >= inputWidth) { inputMask.s9 = 0.0f; } \
    if (inputW + 10 < 0 || inputW + 10 >= inputWidth) { inputMask.sa = 0.0f; } \
    if (inputW + 11 < 0 || inputW + 11 >= inputWidth) { inputMask.sb = 0.0f; } \
    if (inputW + 12 < 0 || inputW + 12 >= inputWidth) { inputMask.sc = 0.0f; } \
    if (inputW + 13 < 0 || inputW + 13 >= inputWidth) { inputMask.sd = 0.0f; } \
    if (inputW + 14 < 0 || inputW + 14 >= inputWidth) { inputMask.se = 0.0f; } \
    if (inputW + 15 < 0 || inputW + 15 >= inputWidth) { inputMask.sf = 0.0f; }

#define FEED_DIRECT_OUTPUT_4X8(bias, outputC, outputW, outputChannel, outputHeight, outputWidth, output0, output1, \
                               output2, output3, output4, output5, output6, output7, output, outputIndex) \
    DATA_T4 biasVector = vload4(0, bias + outputC); \
    output0 += biasVector; \
    output1 += biasVector; \
    output2 += biasVector; \
    output3 += biasVector; \
    output4 += biasVector; \
    output5 += biasVector; \
    output6 += biasVector; \
    output7 += biasVector; \
    if (outputW + 7 < outputWidth) { \
        if (outputC + 3 < outputChannel) { \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s0, output1.s0, output2.s0, output3.s0, \
                                                 output4.s0, output5.s0, output6.s0, output7.s0)), \
                    0, output + outputIndex); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s1, output1.s1, output2.s1, output3.s1, \
                                                 output4.s1, output5.s1, output6.s1, output7.s1)), \
                    0, output + outputIndex + outputHeight * outputWidth); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s2, output1.s2, output2.s2, output3.s2, \
                                                 output4.s2, output5.s2, output6.s2, output7.s2)), \
                    0, output + outputIndex + 2 * outputHeight * outputWidth); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s3, output1.s3, output2.s3, output3.s3, \
                                                 output4.s3, output5.s3, output6.s3, output7.s3)), \
                    0, output + outputIndex + 3 * outputHeight * outputWidth); \
        } else if (outputC + 2 < outputChannel) { \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s0, output1.s0, output2.s0, output3.s0, \
                                                 output4.s0, output5.s0, output6.s0, output7.s0)), \
                    0, output + outputIndex); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s1, output1.s1, output2.s1, output3.s1, \
                                                 output4.s1, output5.s1, output6.s1, output7.s1)), \
                    0, output + outputIndex + outputHeight * outputWidth); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s2, output1.s2, output2.s2, output3.s2, \
                                                 output4.s2, output5.s2, output6.s2, output7.s2)), \
                    0, output + outputIndex + 2 * outputHeight * outputWidth); \
        } else if (outputC + 1 < outputChannel) { \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s0, output1.s0, output2.s0, output3.s0, \
                                                 output4.s0, output5.s0, output6.s0, output7.s0)), \
                    0, output + outputIndex); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s1, output1.s1, output2.s1, output3.s1, \
                                                 output4.s1, output5.s1, output6.s1, output7.s1)), \
                    0, output + outputIndex + outputHeight * outputWidth); \
        } else if (outputC < outputChannel) { \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s0, output1.s0, output2.s0, output3.s0, \
                                                 output4.s0, output5.s0, output6.s0, output7.s0)), \
                    0, output + outputIndex); \
        } \
    }

#define FEED_DIRECT_OUTPUT_8X8(bias, outputC, outputW, outputChannel, outputHeight, outputWidth, output0, output1, \
                               output2, output3, output4, output5, output6, output7, output, outputIndex) \
    DATA_T8 biasVector = vload8(0, bias + outputC); \
    output0 += biasVector; \
    output1 += biasVector; \
    output2 += biasVector; \
    output3 += biasVector; \
    output4 += biasVector; \
    output5 += biasVector; \
    output6 += biasVector; \
    output7 += biasVector; \
    if (outputW + 7 < outputWidth) { \
        if (outputC + 7 < outputChannel) { \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s0, output1.s0, output2.s0, output3.s0, \
                                                 output4.s0, output5.s0, output6.s0, output7.s0)), \
                    0, output + outputIndex); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s1, output1.s1, output2.s1, output3.s1, \
                                                 output4.s1, output5.s1, output6.s1, output7.s1)), \
                    0, output + outputIndex + outputHeight * outputWidth); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s2, output1.s2, output2.s2, output3.s2, \
                                                 output4.s2, output5.s2, output6.s2, output7.s2)), \
                    0, output + outputIndex + 2 * outputHeight * outputWidth); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s3, output1.s3, output2.s3, output3.s3, \
                                                 output4.s3, output5.s3, output6.s3, output7.s3)), \
                    0, output + outputIndex + 3 * outputHeight * outputWidth); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s4, output1.s4, output2.s4, output3.s4, \
                                                 output4.s4, output5.s4, output6.s4, output7.s4)), \
                    0, output + outputIndex + 4 * outputHeight * outputWidth); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s5, output1.s5, output2.s5, output3.s5, \
                                                 output4.s5, output5.s5, output6.s5, output7.s5)), \
                    0, output + outputIndex + 5 * outputHeight * outputWidth); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s6, output1.s6, output2.s6, output3.s6, \
                                                 output4.s6, output5.s6, output6.s6, output7.s6)), \
                    0, output + outputIndex + 6 * outputHeight * outputWidth); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s7, output1.s7, output2.s7, output3.s7, \
                                                 output4.s7, output5.s7, output6.s7, output7.s7)), \
                    0, output + outputIndex + 7 * outputHeight * outputWidth); \
        } else if (outputC + 6 < outputChannel) { \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s0, output1.s0, output2.s0, output3.s0, \
                                                 output4.s0, output5.s0, output6.s0, output7.s0)), \
                    0, output + outputIndex); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s1, output1.s1, output2.s1, output3.s1, \
                                                 output4.s1, output5.s1, output6.s1, output7.s1)), \
                    0, output + outputIndex + outputHeight * outputWidth); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s2, output1.s2, output2.s2, output3.s2, \
                                                 output4.s2, output5.s2, output6.s2, output7.s2)), \
                    0, output + outputIndex + 2 * outputHeight * outputWidth); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s3, output1.s3, output2.s3, output3.s3, \
                                                 output4.s3, output5.s3, output6.s3, output7.s3)), \
                    0, output + outputIndex + 3 * outputHeight * outputWidth); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s4, output1.s4, output2.s4, output3.s4, \
                                                 output4.s4, output5.s4, output6.s4, output7.s4)), \
                    0, output + outputIndex + 4 * outputHeight * outputWidth); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s5, output1.s5, output2.s5, output3.s5, \
                                                 output4.s5, output5.s5, output6.s5, output7.s5)), \
                    0, output + outputIndex + 5 * outputHeight * outputWidth); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s6, output1.s6, output2.s6, output3.s6, \
                                                 output4.s6, output5.s6, output6.s6, output7.s6)), \
                    0, output + outputIndex + 6 * outputHeight * outputWidth); \
        } else if (outputC + 5 < outputChannel) { \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s0, output1.s0, output2.s0, output3.s0, \
                                                 output4.s0, output5.s0, output6.s0, output7.s0)), \
                    0, output + outputIndex); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s1, output1.s1, output2.s1, output3.s1, \
                                                 output4.s1, output5.s1, output6.s1, output7.s1)), \
                    0, output + outputIndex + outputHeight * outputWidth); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s2, output1.s2, output2.s2, output3.s2, \
                                                 output4.s2, output5.s2, output6.s2, output7.s2)), \
                    0, output + outputIndex + 2 * outputHeight * outputWidth); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s3, output1.s3, output2.s3, output3.s3, \
                                                 output4.s3, output5.s3, output6.s3, output7.s3)), \
                    0, output + outputIndex + 3 * outputHeight * outputWidth); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s4, output1.s4, output2.s4, output3.s4, \
                                                 output4.s4, output5.s4, output6.s4, output7.s4)), \
                    0, output + outputIndex + 4 * outputHeight * outputWidth); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s5, output1.s5, output2.s5, output3.s5, \
                                                 output4.s5, output5.s5, output6.s5, output7.s5)), \
                    0, output + outputIndex + 5 * outputHeight * outputWidth); \
        } else if (outputC + 4 < outputChannel) { \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s0, output1.s0, output2.s0, output3.s0, \
                                                 output4.s0, output5.s0, output6.s0, output7.s0)), \
                    0, output + outputIndex); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s1, output1.s1, output2.s1, output3.s1, \
                                                 output4.s1, output5.s1, output6.s1, output7.s1)), \
                    0, output + outputIndex + outputHeight * outputWidth); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s2, output1.s2, output2.s2, output3.s2, \
                                                 output4.s2, output5.s2, output6.s2, output7.s2)), \
                    0, output + outputIndex + 2 * outputHeight * outputWidth); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s3, output1.s3, output2.s3, output3.s3, \
                                                 output4.s3, output5.s3, output6.s3, output7.s3)), \
                    0, output + outputIndex + 3 * outputHeight * outputWidth); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s4, output1.s4, output2.s4, output3.s4, \
                                                 output4.s4, output5.s4, output6.s4, output7.s4)), \
                    0, output + outputIndex + 4 * outputHeight * outputWidth); \
        } else if (outputC + 3 < outputChannel) { \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s0, output1.s0, output2.s0, output3.s0, \
                                                 output4.s0, output5.s0, output6.s0, output7.s0)), \
                    0, output + outputIndex); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s1, output1.s1, output2.s1, output3.s1, \
                                                 output4.s1, output5.s1, output6.s1, output7.s1)), \
                    0, output + outputIndex + outputHeight * outputWidth); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s2, output1.s2, output2.s2, output3.s2, \
                                                 output4.s2, output5.s2, output6.s2, output7.s2)), \
                    0, output + outputIndex + 2 * outputHeight * outputWidth); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s3, output1.s3, output2.s3, output3.s3, \
                                                 output4.s3, output5.s3, output6.s3, output7.s3)), \
                    0, output + outputIndex + 3 * outputHeight * outputWidth); \
        } else if (outputC + 2 < outputChannel) { \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s0, output1.s0, output2.s0, output3.s0, \
                                                 output4.s0, output5.s0, output6.s0, output7.s0)), \
                    0, output + outputIndex); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s1, output1.s1, output2.s1, output3.s1, \
                                                 output4.s1, output5.s1, output6.s1, output7.s1)), \
                    0, output + outputIndex + outputHeight * outputWidth); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s2, output1.s2, output2.s2, output3.s2, \
                                                 output4.s2, output5.s2, output6.s2, output7.s2)), \
                    0, output + outputIndex + 2 * outputHeight * outputWidth); \
        } else if (outputC + 1 < outputChannel) { \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s0, output1.s0, output2.s0, output3.s0, \
                                                 output4.s0, output5.s0, output6.s0, output7.s0)), \
                    0, output + outputIndex); \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s1, output1.s1, output2.s1, output3.s1, \
                                    output4.s1, output5.s1, output6.s1, output7.s1)), \
                    0, output + outputIndex + outputHeight * outputWidth); \
        } else if (outputC < outputChannel) { \
            vstore8(ACT_VEC_F(DATA_T8, (DATA_T8)(output0.s0, output1.s0, output2.s0, output3.s0, \
                                    output4.s0, output5.s0, output6.s0, output7.s0)), \
                    0, output + outputIndex); \
        } \
    }

#define DIRECT3X3_4X8_FP16(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, \
                           outputHeight, outputWidth, inputNum, inputChannel, inputHeight, inputWidth) \
    int outputN = get_global_id(2) / ((outputChannel + 3) / 4); \
    int outputC = get_global_id(2) % ((outputChannel + 3) / 4) * 4; \
    int outputH = get_global_id(1); \
    int outputW = get_global_id(0) * 8; \
    if (outputW < outputWidth && outputH < outputHeight) { \
        DATA_T4 output0 = 0.0f; \
        DATA_T4 output1 = 0.0f; \
        DATA_T4 output2 = 0.0f; \
        DATA_T4 output3 = 0.0f; \
        DATA_T4 output4 = 0.0f; \
        DATA_T4 output5 = 0.0f; \
        DATA_T4 output6 = 0.0f; \
        DATA_T4 output7 = 0.0f; \
        int inputHStart = max(outputH - padT, 0); \
        int inputHEnd = min(outputH - padT + 3, inputHeight); \
        int inputW = outputW - padL; \
        DATA_T16 inputMask = 1.0f; \
        SET_DIRECT_INPUT_MASK16(inputW, inputWidth, inputMask) \
        int inputBase = outputN * inputChannel * inputHeight * inputWidth + inputW; \
        int khOffset = (outputH - padT < 0) ? 3 - (inputHEnd - inputHStart) : 0; \
        int weightBase = outputC * inputChannel * 3 * 3 + khOffset * 4 * 3; \
        DATA_T16 intputVector; \
        DATA_T16 weightVector; \
        for (int inputC = 0; inputC < inputChannel; inputC++) { \
            int inputIndex = inputBase + inputC * inputHeight * inputWidth; \
            int weightIndex = weightBase + inputC * 4 * 3 * 3; \
            for (int kh = inputHStart; kh < inputHEnd; kh++) { \
                intputVector = vload16(0, input + inputIndex + kh * inputWidth); \
                intputVector = intputVector * inputMask; \
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
                output0 = mad(intputVector.s1, weightVector.lo.hi, output0); \
                output1 = mad(intputVector.s2, weightVector.lo.hi, output1); \
                output2 = mad(intputVector.s3, weightVector.lo.hi, output2); \
                output3 = mad(intputVector.s4, weightVector.lo.hi, output3); \
                output4 = mad(intputVector.s5, weightVector.lo.hi, output4); \
                output5 = mad(intputVector.s6, weightVector.lo.hi, output5); \
                output6 = mad(intputVector.s7, weightVector.lo.hi, output6); \
                output7 = mad(intputVector.s8, weightVector.lo.hi, output7); \
                output0 = mad(intputVector.s2, weightVector.hi.lo, output0); \
                output1 = mad(intputVector.s3, weightVector.hi.lo, output1); \
                output2 = mad(intputVector.s4, weightVector.hi.lo, output2); \
                output3 = mad(intputVector.s5, weightVector.hi.lo, output3); \
                output4 = mad(intputVector.s6, weightVector.hi.lo, output4); \
                output5 = mad(intputVector.s7, weightVector.hi.lo, output5); \
                output6 = mad(intputVector.s8, weightVector.hi.lo, output6); \
                output7 = mad(intputVector.s9, weightVector.hi.lo, output7); \
            } \
        } \
        int outputIndex = outputN * outputChannel * outputHeight * outputWidth + \
                            outputC * outputHeight * outputWidth + outputH * outputWidth + \
                            outputW; \
        FEED_DIRECT_OUTPUT_4X8(bias, outputC, outputW, outputChannel, outputHeight, outputWidth, output0, output1, \
                                output2, output3, output4, output5, output6, output7, output, outputIndex) \
    }

#define DIRECT3X3_4X8_FP32(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, \
                           outputHeight, outputWidth, inputNum, inputChannel, inputHeight, inputWidth) \
    int outputN = get_global_id(2) / ((outputChannel + 3) / 4); \
    int outputC = get_global_id(2) % ((outputChannel + 3) / 4) * 4; \
    int outputH = get_global_id(1); \
    int outputW = get_global_id(0) * 8; \
    if (outputW < outputWidth && outputH < outputHeight) { \
        DATA_T4 output0 = 0.0f; \
        DATA_T4 output1 = 0.0f; \
        DATA_T4 output2 = 0.0f; \
        DATA_T4 output3 = 0.0f; \
        DATA_T4 output4 = 0.0f; \
        DATA_T4 output5 = 0.0f; \
        DATA_T4 output6 = 0.0f; \
        DATA_T4 output7 = 0.0f; \
        int inputHStart = max(outputH - padT, 0); \
        int inputHEnd = min(outputH - padT + 3, inputHeight); \
        int inputW = outputW - padL; \
        DATA_T16 inputMask = 1.0f; \
        SET_DIRECT_INPUT_MASK16(inputW, inputWidth, inputMask) \
        int inputBase = outputN * inputChannel * inputHeight * inputWidth + inputW; \
        int khOffset = (outputH - padT < 0) ? 3 - (inputHEnd - inputHStart) : 0; \
        int weightBase = outputC * inputChannel * 3 * 3 + khOffset * 4 * 3; \
        DATA_T16 intputVector; \
        DATA_T8 weightVector; \
        for (int inputC = 0; inputC < inputChannel; inputC++) { \
            int inputIndex = inputBase + inputC * inputHeight * inputWidth; \
            int weightIndex = weightBase + inputC * 4 * 3 * 3; \
            for (int kh = inputHStart; kh < inputHEnd; kh++) { \
                intputVector = vload16(0, input + inputIndex + kh * inputWidth); \
                intputVector = intputVector * inputMask; \
                weightVector = vload8(0, weight + weightIndex); \
                weightIndex += 8; \
                output0 = mad(intputVector.s0, weightVector.lo, output0); \
                output1 = mad(intputVector.s1, weightVector.lo, output1); \
                output2 = mad(intputVector.s2, weightVector.lo, output2); \
                output3 = mad(intputVector.s3, weightVector.lo, output3); \
                output4 = mad(intputVector.s4, weightVector.lo, output4); \
                output5 = mad(intputVector.s5, weightVector.lo, output5); \
                output6 = mad(intputVector.s6, weightVector.lo, output6); \
                output7 = mad(intputVector.s7, weightVector.lo, output7); \
                output0 = mad(intputVector.s1, weightVector.hi, output0); \
                output1 = mad(intputVector.s2, weightVector.hi, output1); \
                output2 = mad(intputVector.s3, weightVector.hi, output2); \
                output3 = mad(intputVector.s4, weightVector.hi, output3); \
                output4 = mad(intputVector.s5, weightVector.hi, output4); \
                output5 = mad(intputVector.s6, weightVector.hi, output5); \
                output6 = mad(intputVector.s7, weightVector.hi, output6); \
                output7 = mad(intputVector.s8, weightVector.hi, output7); \
                weightVector = vload8(0, weight + weightIndex); \
                weightIndex += 4; \
                output0 = mad(intputVector.s2, weightVector.lo, output0); \
                output1 = mad(intputVector.s3, weightVector.lo, output1); \
                output2 = mad(intputVector.s4, weightVector.lo, output2); \
                output3 = mad(intputVector.s5, weightVector.lo, output3); \
                output4 = mad(intputVector.s6, weightVector.lo, output4); \
                output5 = mad(intputVector.s7, weightVector.lo, output5); \
                output6 = mad(intputVector.s8, weightVector.lo, output6); \
                output7 = mad(intputVector.s9, weightVector.lo, output7); \
            } \
        } \
        int outputIndex = outputN * outputChannel * outputHeight * outputWidth + \
                            outputC * outputHeight * outputWidth + outputH * outputWidth + \
                            outputW; \
        FEED_DIRECT_OUTPUT_4X8(bias, outputC, outputW, outputChannel, outputHeight, outputWidth, output0, output1, \
                                output2, output3, output4, output5, output6, output7, output, outputIndex) \
    }

#define DIRECT3X3_8X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                      outputWidth, inputNum, inputChannel, inputHeight, inputWidth) \
    int outputN = get_global_id(2) / ((outputChannel + 7) / 8); \
    int outputC = get_global_id(2) % ((outputChannel + 7) / 8) * 8; \
    int outputH = get_global_id(1); \
    int outputW = get_global_id(0) * 8; \
    if (outputW < outputWidth && outputH < outputHeight) { \
        DATA_T8 output0 = 0.0f; \
        DATA_T8 output1 = 0.0f; \
        DATA_T8 output2 = 0.0f; \
        DATA_T8 output3 = 0.0f; \
        DATA_T8 output4 = 0.0f; \
        DATA_T8 output5 = 0.0f; \
        DATA_T8 output6 = 0.0f; \
        DATA_T8 output7 = 0.0f; \
        int inputHStart = max(outputH - padT, 0); \
        int inputHEnd = min(outputH - padT + 3, inputHeight); \
        int inputW = outputW - padL; \
        DATA_T16 inputMask = 1.0f; \
        SET_DIRECT_INPUT_MASK16(inputW, inputWidth, inputMask) \
        int inputBase = outputN * inputChannel * inputHeight * inputWidth + inputW; \
        int khOffset = (outputH - padT < 0) ? 3 - (inputHEnd - inputHStart) : 0; \
        int weightBase = outputC * inputChannel * 3 * 3 + khOffset * 8 * 3; \
        DATA_T16 intputVector; \
        DATA_T16 weightVector; \
        for (int inputC = 0; inputC < inputChannel; inputC++) { \
            int inputIndex = inputBase + inputC * inputHeight * inputWidth; \
            int weightIndex = weightBase + inputC * 8 * 3 * 3; \
            for (int kh = inputHStart; kh < inputHEnd; kh++) { \
                intputVector = vload16(0, input + inputIndex + kh * inputWidth); \
                intputVector = intputVector * inputMask; \
                weightVector = vload16(0, weight + weightIndex); \
                weightIndex += 16; \
                output0 = mad(intputVector.s0, weightVector.lo, output0); \
                output1 = mad(intputVector.s1, weightVector.lo, output1); \
                output2 = mad(intputVector.s2, weightVector.lo, output2); \
                output3 = mad(intputVector.s3, weightVector.lo, output3); \
                output4 = mad(intputVector.s4, weightVector.lo, output4); \
                output5 = mad(intputVector.s5, weightVector.lo, output5); \
                output6 = mad(intputVector.s6, weightVector.lo, output6); \
                output7 = mad(intputVector.s7, weightVector.lo, output7); \
                output0 = mad(intputVector.s1, weightVector.hi, output0); \
                output1 = mad(intputVector.s2, weightVector.hi, output1); \
                output2 = mad(intputVector.s3, weightVector.hi, output2); \
                output3 = mad(intputVector.s4, weightVector.hi, output3); \
                output4 = mad(intputVector.s5, weightVector.hi, output4); \
                output5 = mad(intputVector.s6, weightVector.hi, output5); \
                output6 = mad(intputVector.s7, weightVector.hi, output6); \
                output7 = mad(intputVector.s8, weightVector.hi, output7); \
                weightVector.lo = vload8(0, weight + weightIndex); \
                weightIndex += 8; \
                output0 = mad(intputVector.s2, weightVector.lo, output0); \
                output1 = mad(intputVector.s3, weightVector.lo, output1); \
                output2 = mad(intputVector.s4, weightVector.lo, output2); \
                output3 = mad(intputVector.s5, weightVector.lo, output3); \
                output4 = mad(intputVector.s6, weightVector.lo, output4); \
                output5 = mad(intputVector.s7, weightVector.lo, output5); \
                output6 = mad(intputVector.s8, weightVector.lo, output6); \
                output7 = mad(intputVector.s9, weightVector.lo, output7); \
            } \
        } \
        int outputIndex = outputN * outputChannel * outputHeight * outputWidth + \
                            outputC * outputHeight * outputWidth + outputH * outputWidth + \
                            outputW; \
        FEED_DIRECT_OUTPUT_8X8(bias, outputC, outputW, outputChannel, outputHeight, outputWidth, output0, output1, \
                            output2, output3, output4, output5, output6, output7, output, outputIndex) \
    }

#define DIRECT5X5_4X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                      outputWidth, inputNum, inputChannel, inputHeight, inputWidth) \
    int outputN = get_global_id(2) / ((outputChannel + 3) / 4); \
    int outputC = get_global_id(2) % ((outputChannel + 3) / 4) * 4; \
    int outputH = get_global_id(1); \
    int outputW = get_global_id(0) * 8; \
    if (outputW < outputWidth && outputH < outputHeight) { \
        DATA_T4 output0 = 0.0f; \
        DATA_T4 output1 = 0.0f; \
        DATA_T4 output2 = 0.0f; \
        DATA_T4 output3 = 0.0f; \
        DATA_T4 output4 = 0.0f; \
        DATA_T4 output5 = 0.0f; \
        DATA_T4 output6 = 0.0f; \
        DATA_T4 output7 = 0.0f; \
        int inputHStart = max(outputH - padT, 0); \
        int inputHEnd = min(outputH - padT + 5, inputHeight); \
        int inputW = outputW - padL; \
        DATA_T16 inputMask = 1.0f; \
        SET_DIRECT_INPUT_MASK16(inputW, inputWidth, inputMask) \
        int inputBase = outputN * inputChannel * inputHeight * inputWidth + inputW; \
        int khOffset = (outputH - padT < 0) ? 5 - (inputHEnd - inputHStart) : 0; \
        int weightBase = outputC * inputChannel * 5 * 5 + khOffset * 4 * 5; \
        DATA_T16 intputVector; \
        DATA_T16 weightVector; \
        for (int inputC = 0; inputC < inputChannel; inputC++) { \
            int inputIndex = inputBase + inputC * inputHeight * inputWidth; \
            int weightIndex = weightBase + inputC * 4 * 5 * 5; \
            for (int kh = inputHStart; kh < inputHEnd; kh++) { \
                intputVector = vload16(0, input + inputIndex + kh * inputWidth); \
                intputVector = intputVector * inputMask; \
                weightVector = vload16(0, weight + weightIndex); \
                weightIndex += 16; \
                output0 = mad(intputVector.s0, weightVector.lo.lo, output0); \
                output1 = mad(intputVector.s1, weightVector.lo.lo, output1); \
                output2 = mad(intputVector.s2, weightVector.lo.lo, output2); \
                output3 = mad(intputVector.s3, weightVector.lo.lo, output3); \
                output4 = mad(intputVector.s4, weightVector.lo.lo, output4); \
                output5 = mad(intputVector.s5, weightVector.lo.lo, output5); \
                output6 = mad(intputVector.s6, weightVector.lo.lo, output6); \
                output7 = mad(intputVector.s7, weightVector.lo.lo, output7); \
                output0 = mad(intputVector.s1, weightVector.lo.hi, output0); \
                output1 = mad(intputVector.s2, weightVector.lo.hi, output1); \
                output2 = mad(intputVector.s3, weightVector.lo.hi, output2); \
                output3 = mad(intputVector.s4, weightVector.lo.hi, output3); \
                output4 = mad(intputVector.s5, weightVector.lo.hi, output4); \
                output5 = mad(intputVector.s6, weightVector.lo.hi, output5); \
                output6 = mad(intputVector.s7, weightVector.lo.hi, output6); \
                output7 = mad(intputVector.s8, weightVector.lo.hi, output7); \
                output0 = mad(intputVector.s2, weightVector.hi.lo, output0); \
                output1 = mad(intputVector.s3, weightVector.hi.lo, output1); \
                output2 = mad(intputVector.s4, weightVector.hi.lo, output2); \
                output3 = mad(intputVector.s5, weightVector.hi.lo, output3); \
                output4 = mad(intputVector.s6, weightVector.hi.lo, output4); \
                output5 = mad(intputVector.s7, weightVector.hi.lo, output5); \
                output6 = mad(intputVector.s8, weightVector.hi.lo, output6); \
                output7 = mad(intputVector.s9, weightVector.hi.lo, output7); \
                output0 = mad(intputVector.s3, weightVector.hi.hi, output0); \
                output1 = mad(intputVector.s4, weightVector.hi.hi, output1); \
                output2 = mad(intputVector.s5, weightVector.hi.hi, output2); \
                output3 = mad(intputVector.s6, weightVector.hi.hi, output3); \
                output4 = mad(intputVector.s7, weightVector.hi.hi, output4); \
                output5 = mad(intputVector.s8, weightVector.hi.hi, output5); \
                output6 = mad(intputVector.s9, weightVector.hi.hi, output6); \
                output7 = mad(intputVector.sa, weightVector.hi.hi, output7); \
                weightVector.lo.lo = vload4(0, weight + weightIndex); \
                weightIndex += 4; \
                output0 = mad(intputVector.s4, weightVector.lo.lo, output0); \
                output1 = mad(intputVector.s5, weightVector.lo.lo, output1); \
                output2 = mad(intputVector.s6, weightVector.lo.lo, output2); \
                output3 = mad(intputVector.s7, weightVector.lo.lo, output3); \
                output4 = mad(intputVector.s8, weightVector.lo.lo, output4); \
                output5 = mad(intputVector.s9, weightVector.lo.lo, output5); \
                output6 = mad(intputVector.sa, weightVector.lo.lo, output6); \
                output7 = mad(intputVector.sb, weightVector.lo.lo, output7); \
            } \
        } \
        int outputIndex = outputN * outputChannel * outputHeight * outputWidth + \
                            outputC * outputHeight * outputWidth + outputH * outputWidth + outputW; \
        FEED_DIRECT_OUTPUT_4X8(bias, outputC, outputW, outputChannel, outputHeight, outputWidth, output0, output1, \
                            output2, output3, output4, output5, output6, output7, output, outputIndex) \
    }

#define DIRECT5X5_8X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                      outputWidth, inputNum, inputChannel, inputHeight, inputWidth) \
    int outputN = get_global_id(2) / ((outputChannel + 7) / 8); \
    int outputC = get_global_id(2) % ((outputChannel + 7) / 8) * 8; \
    int outputH = get_global_id(1); \
    int outputW = get_global_id(0) * 8; \
    if (outputW < outputWidth && outputH < outputHeight) { \
        DATA_T8 output0 = 0.0f; \
        DATA_T8 output1 = 0.0f; \
        DATA_T8 output2 = 0.0f; \
        DATA_T8 output3 = 0.0f; \
        DATA_T8 output4 = 0.0f; \
        DATA_T8 output5 = 0.0f; \
        DATA_T8 output6 = 0.0f; \
        DATA_T8 output7 = 0.0f; \
        int inputHStart = max(outputH - padT, 0); \
        int inputHEnd = min(outputH - padT + 5, inputHeight); \
        int inputW = outputW - padL; \
        DATA_T16 inputMask = 1.0f; \
        SET_DIRECT_INPUT_MASK16(inputW, inputWidth, inputMask) \
        int inputBase = outputN * inputChannel * inputHeight * inputWidth + inputW; \
        int khOffset = (outputH - padT < 0) ? 5 - (inputHEnd - inputHStart) : 0; \
        int weightBase = outputC * inputChannel * 5 * 5 + khOffset * 8 * 5; \
        DATA_T16 intputVector; \
        DATA_T16 weightVector; \
        for (int inputC = 0; inputC < inputChannel; inputC++) { \
            int inputIndex = inputBase + inputC * inputHeight * inputWidth; \
            int weightIndex = weightBase + inputC * 8 * 5 * 5; \
            for (int kh = inputHStart; kh < inputHEnd; kh++) { \
                intputVector = vload16(0, input + inputIndex + kh * inputWidth); \
                intputVector = intputVector * inputMask; \
                weightVector = vload16(0, weight + weightIndex); \
                weightIndex += 16; \
                output0 = mad(intputVector.s0, weightVector.lo, output0); \
                output1 = mad(intputVector.s1, weightVector.lo, output1); \
                output2 = mad(intputVector.s2, weightVector.lo, output2); \
                output3 = mad(intputVector.s3, weightVector.lo, output3); \
                output4 = mad(intputVector.s4, weightVector.lo, output4); \
                output5 = mad(intputVector.s5, weightVector.lo, output5); \
                output6 = mad(intputVector.s6, weightVector.lo, output6); \
                output7 = mad(intputVector.s7, weightVector.lo, output7); \
                output0 = mad(intputVector.s1, weightVector.hi, output0); \
                output1 = mad(intputVector.s2, weightVector.hi, output1); \
                output2 = mad(intputVector.s3, weightVector.hi, output2); \
                output3 = mad(intputVector.s4, weightVector.hi, output3); \
                output4 = mad(intputVector.s5, weightVector.hi, output4); \
                output5 = mad(intputVector.s6, weightVector.hi, output5); \
                output6 = mad(intputVector.s7, weightVector.hi, output6); \
                output7 = mad(intputVector.s8, weightVector.hi, output7); \
                weightVector = vload16(0, weight + weightIndex); \
                weightIndex += 16; \
                output0 = mad(intputVector.s2, weightVector.lo, output0); \
                output1 = mad(intputVector.s3, weightVector.lo, output1); \
                output2 = mad(intputVector.s4, weightVector.lo, output2); \
                output3 = mad(intputVector.s5, weightVector.lo, output3); \
                output4 = mad(intputVector.s6, weightVector.lo, output4); \
                output5 = mad(intputVector.s7, weightVector.lo, output5); \
                output6 = mad(intputVector.s8, weightVector.lo, output6); \
                output7 = mad(intputVector.s9, weightVector.lo, output7); \
                output0 = mad(intputVector.s3, weightVector.hi, output0); \
                output1 = mad(intputVector.s4, weightVector.hi, output1); \
                output2 = mad(intputVector.s5, weightVector.hi, output2); \
                output3 = mad(intputVector.s6, weightVector.hi, output3); \
                output4 = mad(intputVector.s7, weightVector.hi, output4); \
                output5 = mad(intputVector.s8, weightVector.hi, output5); \
                output6 = mad(intputVector.s9, weightVector.hi, output6); \
                output7 = mad(intputVector.sa, weightVector.hi, output7); \
                weightVector.lo = vload8(0, weight + weightIndex); \
                weightIndex += 8; \
                output0 = mad(intputVector.s4, weightVector.lo, output0); \
                output1 = mad(intputVector.s5, weightVector.lo, output1); \
                output2 = mad(intputVector.s6, weightVector.lo, output2); \
                output3 = mad(intputVector.s7, weightVector.lo, output3); \
                output4 = mad(intputVector.s8, weightVector.lo, output4); \
                output5 = mad(intputVector.s9, weightVector.lo, output5); \
                output6 = mad(intputVector.sa, weightVector.lo, output6); \
                output7 = mad(intputVector.sb, weightVector.lo, output7); \
            } \
        } \
        int outputIndex = outputN * outputChannel * outputHeight * outputWidth + \
                            outputC * outputHeight * outputWidth + outputH * outputWidth + \
                            outputW; \
        FEED_DIRECT_OUTPUT_8X8(bias, outputC, outputW, outputChannel, outputHeight, outputWidth, output0, output1, \
                            output2, output3, output4, output5, output6, output7, output, outputIndex) \
    }

#define DIRECT7X7_4X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                      outputWidth, inputNum, inputChannel, inputHeight, inputWidth) \
    int outputN = get_global_id(2) / ((outputChannel + 3) / 4); \
    int outputC = get_global_id(2) % ((outputChannel + 3) / 4) * 4; \
    int outputH = get_global_id(1); \
    int outputW = get_global_id(0) * 8; \
    if (outputW < outputWidth && outputH < outputHeight) { \
        DATA_T4 output0 = 0.0f; \
        DATA_T4 output1 = 0.0f; \
        DATA_T4 output2 = 0.0f; \
        DATA_T4 output3 = 0.0f; \
        DATA_T4 output4 = 0.0f; \
        DATA_T4 output5 = 0.0f; \
        DATA_T4 output6 = 0.0f; \
        DATA_T4 output7 = 0.0f; \
        int inputHStart = max(outputH - padT, 0); \
        int inputHEnd = min(outputH - padT + 7, inputHeight); \
        int inputW = outputW - padL; \
        DATA_T16 inputMask = 1.0f; \
        SET_DIRECT_INPUT_MASK16(inputW, inputWidth, inputMask) \
        int inputBase = outputN * inputChannel * inputHeight * inputWidth + inputW; \
        int khOffset = (outputH - padT < 0) ? 7 - (inputHEnd - inputHStart) : 0; \
        int weightBase = outputC * inputChannel * 7 * 7 + khOffset * 4 * 7; \
        DATA_T16 intputVector; \
        DATA_T16 weightVector; \
        for (int inputC = 0; inputC < inputChannel; inputC++) { \
            int inputIndex = inputBase + inputC * inputHeight * inputWidth; \
            int weightIndex = weightBase + inputC * 4 * 7 * 7; \
            for (int kh = inputHStart; kh < inputHEnd; kh++) { \
                intputVector = vload16(0, input + inputIndex + kh * inputWidth); \
                intputVector = intputVector * inputMask; \
                weightVector = vload16(0, weight + weightIndex); \
                weightIndex += 16; \
                output0 = mad(intputVector.s0, weightVector.lo.lo, output0); \
                output1 = mad(intputVector.s1, weightVector.lo.lo, output1); \
                output2 = mad(intputVector.s2, weightVector.lo.lo, output2); \
                output3 = mad(intputVector.s3, weightVector.lo.lo, output3); \
                output4 = mad(intputVector.s4, weightVector.lo.lo, output4); \
                output5 = mad(intputVector.s5, weightVector.lo.lo, output5); \
                output6 = mad(intputVector.s6, weightVector.lo.lo, output6); \
                output7 = mad(intputVector.s7, weightVector.lo.lo, output7); \
                output0 = mad(intputVector.s1, weightVector.lo.hi, output0); \
                output1 = mad(intputVector.s2, weightVector.lo.hi, output1); \
                output2 = mad(intputVector.s3, weightVector.lo.hi, output2); \
                output3 = mad(intputVector.s4, weightVector.lo.hi, output3); \
                output4 = mad(intputVector.s5, weightVector.lo.hi, output4); \
                output5 = mad(intputVector.s6, weightVector.lo.hi, output5); \
                output6 = mad(intputVector.s7, weightVector.lo.hi, output6); \
                output7 = mad(intputVector.s8, weightVector.lo.hi, output7); \
                output0 = mad(intputVector.s2, weightVector.hi.lo, output0); \
                output1 = mad(intputVector.s3, weightVector.hi.lo, output1); \
                output2 = mad(intputVector.s4, weightVector.hi.lo, output2); \
                output3 = mad(intputVector.s5, weightVector.hi.lo, output3); \
                output4 = mad(intputVector.s6, weightVector.hi.lo, output4); \
                output5 = mad(intputVector.s7, weightVector.hi.lo, output5); \
                output6 = mad(intputVector.s8, weightVector.hi.lo, output6); \
                output7 = mad(intputVector.s9, weightVector.hi.lo, output7); \
                output0 = mad(intputVector.s3, weightVector.hi.hi, output0); \
                output1 = mad(intputVector.s4, weightVector.hi.hi, output1); \
                output2 = mad(intputVector.s5, weightVector.hi.hi, output2); \
                output3 = mad(intputVector.s6, weightVector.hi.hi, output3); \
                output4 = mad(intputVector.s7, weightVector.hi.hi, output4); \
                output5 = mad(intputVector.s8, weightVector.hi.hi, output5); \
                output6 = mad(intputVector.s9, weightVector.hi.hi, output6); \
                output7 = mad(intputVector.sa, weightVector.hi.hi, output7); \
                weightVector.lo = vload8(0, weight + weightIndex); \
                weightIndex += 8; \
                output0 = mad(intputVector.s4, weightVector.lo.lo, output0); \
                output1 = mad(intputVector.s5, weightVector.lo.lo, output1); \
                output2 = mad(intputVector.s6, weightVector.lo.lo, output2); \
                output3 = mad(intputVector.s7, weightVector.lo.lo, output3); \
                output4 = mad(intputVector.s8, weightVector.lo.lo, output4); \
                output5 = mad(intputVector.s9, weightVector.lo.lo, output5); \
                output6 = mad(intputVector.sa, weightVector.lo.lo, output6); \
                output7 = mad(intputVector.sb, weightVector.lo.lo, output7); \
                output0 = mad(intputVector.s5, weightVector.lo.hi, output0); \
                output1 = mad(intputVector.s6, weightVector.lo.hi, output1); \
                output2 = mad(intputVector.s7, weightVector.lo.hi, output2); \
                output3 = mad(intputVector.s8, weightVector.lo.hi, output3); \
                output4 = mad(intputVector.s9, weightVector.lo.hi, output4); \
                output5 = mad(intputVector.sa, weightVector.lo.hi, output5); \
                output6 = mad(intputVector.sb, weightVector.lo.hi, output6); \
                output7 = mad(intputVector.sc, weightVector.lo.hi, output7); \
                weightVector.lo.lo = vload4(0, weight + weightIndex); \
                weightIndex += 4; \
                output0 = mad(intputVector.s6, weightVector.lo.lo, output0); \
                output1 = mad(intputVector.s7, weightVector.lo.lo, output1); \
                output2 = mad(intputVector.s8, weightVector.lo.lo, output2); \
                output3 = mad(intputVector.s9, weightVector.lo.lo, output3); \
                output4 = mad(intputVector.sa, weightVector.lo.lo, output4); \
                output5 = mad(intputVector.sb, weightVector.lo.lo, output5); \
                output6 = mad(intputVector.sc, weightVector.lo.lo, output6); \
                output7 = mad(intputVector.sd, weightVector.lo.lo, output7); \
            } \
        } \
        int outputIndex = outputN * outputChannel * outputHeight * outputWidth + \
                            outputC * outputHeight * outputWidth + outputH * outputWidth + outputW; \
        FEED_DIRECT_OUTPUT_4X8(bias, outputC, outputW, outputChannel, outputHeight, outputWidth, output0, output1, \
                                output2, output3, output4, output5, output6, output7, output, outputIndex) \
    }

#define DIRECT7X7_4X8_MERGE(bias, aligned_output, output, padT, padR, padB, padL, outputNumber, outputChannel, \
                            outputHeight, outputWidth, inputNum, inputChannel, inputHeight, inputWidth, splite_num) \
    int outputN = get_global_id(2) / ((outputChannel + 3) / 4); \
    int outputC = get_global_id(2) * 4; \
    int outputH = get_global_id(1); \
    int outputW = get_global_id(0) * 8; \
    if (outputW < outputWidth && outputH < outputHeight) { \
        DATA_T4 output0 = 0.0f; \
        DATA_T4 output1 = 0.0f; \
        DATA_T4 output2 = 0.0f; \
        DATA_T4 output3 = 0.0f; \
        DATA_T4 output4 = 0.0f; \
        DATA_T4 output5 = 0.0f; \
        DATA_T4 output6 = 0.0f; \
        DATA_T4 output7 = 0.0f; \
        int outputIndex = outputN * outputChannel * outputHeight * outputWidth + \
                            outputC * outputHeight * outputWidth + outputH * outputWidth + outputW; \
        int aligned_output_base = outputN * outputChannel * outputHeight * outputWidth / 32 + \
                            outputC / 4 * outputHeight * outputWidth / 8 + outputH * outputWidth / 8 + outputW / 8; \
        aligned_output_base *= 32 * splite_num; \
        if (2 == splite_num) { \
            DATA_T16 r0 = vload16(0, aligned_output + aligned_output_base); \
            DATA_T16 r1 = vload16(1, aligned_output + aligned_output_base); \
            DATA_T16 r2 = vload16(2, aligned_output + aligned_output_base); \
            DATA_T16 r3 = vload16(3, aligned_output + aligned_output_base); \
            r0 += r2; \
            r1 += r3; \
            output0 = r0.lo.lo; \
            output1 = r0.lo.hi; \
            output2 = r0.hi.lo; \
            output3 = r0.hi.hi; \
            output4 = r1.lo.lo; \
            output5 = r1.lo.hi; \
            output6 = r1.hi.lo; \
            output7 = r1.hi.hi; \
        } else if (4 == splite_num) { \
            DATA_T16 r0 = vload16(0, aligned_output + aligned_output_base); \
            DATA_T16 r1 = vload16(1, aligned_output + aligned_output_base); \
            DATA_T16 r2 = vload16(2, aligned_output + aligned_output_base); \
            DATA_T16 r3 = vload16(3, aligned_output + aligned_output_base); \
            r0 += r2; \
            r1 += r3; \
            r2 = vload16(4, aligned_output + aligned_output_base); \
            r3 = vload16(5, aligned_output + aligned_output_base); \
            r0 += r2; \
            r1 += r3; \
            r2 = vload16(6, aligned_output + aligned_output_base); \
            r3 = vload16(7, aligned_output + aligned_output_base); \
            r0 += r2; \
            r1 += r3; \
            output0 = r0.lo.lo; \
            output1 = r0.lo.hi; \
            output2 = r0.hi.lo; \
            output3 = r0.hi.hi; \
            output4 = r1.lo.lo; \
            output5 = r1.lo.hi; \
            output6 = r1.hi.lo; \
            output7 = r1.hi.hi; \
        } else { \
            DATA_T16 r0 = vload16(0, aligned_output + aligned_output_base); \
            DATA_T16 r1 = vload16(1, aligned_output + aligned_output_base); \
            for (int i = 1; i < splite_num; ++i) { \
                    DATA_T16 r2 = vload16(i * 2, aligned_output + aligned_output_base); \
                    DATA_T16 r3 = vload16(i * 2 + 1, aligned_output + aligned_output_base); \
                    r0 += r2; \
                    r1 += r3; \
            } \
            output0 = r0.lo.lo; \
            output1 = r0.lo.hi; \
            output2 = r0.hi.lo; \
            output3 = r0.hi.hi; \
            output4 = r1.lo.lo; \
            output5 = r1.lo.hi; \
            output6 = r1.hi.lo; \
            output7 = r1.hi.hi; \
        } \
        FEED_DIRECT_OUTPUT_4X8(bias, outputC, outputW, outputChannel, outputHeight, outputWidth, output0, output1, \
                               output2, output3, output4, output5, output6, output7, output, outputIndex) \
    }

#define DIRECT7X7_8X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                      outputWidth, inputNum, inputChannel, inputHeight, inputWidth) \
    int outputN = get_global_id(2) / ((outputChannel + 7) / 8); \
    int outputC = get_global_id(2) % ((outputChannel + 7) / 8) * 8; \
    int outputH = get_global_id(1); \
    int outputW = get_global_id(0) * 8; \
    if (outputW < outputWidth && outputH < outputHeight) { \
        DATA_T8 output0 = 0.0f; \
        DATA_T8 output1 = 0.0f; \
        DATA_T8 output2 = 0.0f; \
        DATA_T8 output3 = 0.0f; \
        DATA_T8 output4 = 0.0f; \
        DATA_T8 output5 = 0.0f; \
        DATA_T8 output6 = 0.0f; \
        DATA_T8 output7 = 0.0f; \
        int inputHStart = max(outputH - padT, 0); \
        int inputHEnd = min(outputH - padT + 7, inputHeight); \
        int inputW = outputW - padL; \
        DATA_T16 inputMask = 1.0f; \
        SET_DIRECT_INPUT_MASK16(inputW, inputWidth, inputMask) \
        int inputBase = outputN * inputChannel * inputHeight * inputWidth + inputW; \
        int khOffset = (outputH - padT < 0) ? 7 - (inputHEnd - inputHStart) : 0; \
        int weightBase = outputC * inputChannel * 7 * 7 + khOffset * 8 * 7; \
        DATA_T16 intputVector; \
        DATA_T16 weightVector; \
        for (int inputC = 0; inputC < inputChannel; inputC++) { \
            int inputIndex = inputBase + inputC * inputHeight * inputWidth; \
            int weightIndex = weightBase + inputC * 8 * 7 * 7; \
            for (int kh = inputHStart; kh < inputHEnd; kh++) { \
                intputVector = vload16(0, input + inputIndex + kh * inputWidth); \
                intputVector = intputVector * inputMask; \
                weightVector = vload16(0, weight + weightIndex); \
                weightIndex += 16; \
                output0 = mad(intputVector.s0, weightVector.lo, output0); \
                output1 = mad(intputVector.s1, weightVector.lo, output1); \
                output2 = mad(intputVector.s2, weightVector.lo, output2); \
                output3 = mad(intputVector.s3, weightVector.lo, output3); \
                output4 = mad(intputVector.s4, weightVector.lo, output4); \
                output5 = mad(intputVector.s5, weightVector.lo, output5); \
                output6 = mad(intputVector.s6, weightVector.lo, output6); \
                output7 = mad(intputVector.s7, weightVector.lo, output7); \
                output0 = mad(intputVector.s1, weightVector.hi, output0); \
                output1 = mad(intputVector.s2, weightVector.hi, output1); \
                output2 = mad(intputVector.s3, weightVector.hi, output2); \
                output3 = mad(intputVector.s4, weightVector.hi, output3); \
                output4 = mad(intputVector.s5, weightVector.hi, output4); \
                output5 = mad(intputVector.s6, weightVector.hi, output5); \
                output6 = mad(intputVector.s7, weightVector.hi, output6); \
                output7 = mad(intputVector.s8, weightVector.hi, output7); \
                weightVector = vload16(0, weight + weightIndex); \
                weightIndex += 16; \
                output0 = mad(intputVector.s2, weightVector.lo, output0); \
                output1 = mad(intputVector.s3, weightVector.lo, output1); \
                output2 = mad(intputVector.s4, weightVector.lo, output2); \
                output3 = mad(intputVector.s5, weightVector.lo, output3); \
                output4 = mad(intputVector.s6, weightVector.lo, output4); \
                output5 = mad(intputVector.s7, weightVector.lo, output5); \
                output6 = mad(intputVector.s8, weightVector.lo, output6); \
                output7 = mad(intputVector.s9, weightVector.lo, output7); \
                output0 = mad(intputVector.s3, weightVector.hi, output0); \
                output1 = mad(intputVector.s4, weightVector.hi, output1); \
                output2 = mad(intputVector.s5, weightVector.hi, output2); \
                output3 = mad(intputVector.s6, weightVector.hi, output3); \
                output4 = mad(intputVector.s7, weightVector.hi, output4); \
                output5 = mad(intputVector.s8, weightVector.hi, output5); \
                output6 = mad(intputVector.s9, weightVector.hi, output6); \
                output7 = mad(intputVector.sa, weightVector.hi, output7); \
                weightVector = vload16(0, weight + weightIndex); \
                weightIndex += 16; \
                output0 = mad(intputVector.s4, weightVector.lo, output0); \
                output1 = mad(intputVector.s5, weightVector.lo, output1); \
                output2 = mad(intputVector.s6, weightVector.lo, output2); \
                output3 = mad(intputVector.s7, weightVector.lo, output3); \
                output4 = mad(intputVector.s8, weightVector.lo, output4); \
                output5 = mad(intputVector.s9, weightVector.lo, output5); \
                output6 = mad(intputVector.sa, weightVector.lo, output6); \
                output7 = mad(intputVector.sb, weightVector.lo, output7); \
                output0 = mad(intputVector.s5, weightVector.hi, output0); \
                output1 = mad(intputVector.s6, weightVector.hi, output1); \
                output2 = mad(intputVector.s7, weightVector.hi, output2); \
                output3 = mad(intputVector.s8, weightVector.hi, output3); \
                output4 = mad(intputVector.s9, weightVector.hi, output4); \
                output5 = mad(intputVector.sa, weightVector.hi, output5); \
                output6 = mad(intputVector.sb, weightVector.hi, output6); \
                output7 = mad(intputVector.sc, weightVector.hi, output7); \
                weightVector.lo = vload8(0, weight + weightIndex); \
                weightIndex += 8; \
                output0 = mad(intputVector.s6, weightVector.lo, output0); \
                output1 = mad(intputVector.s7, weightVector.lo, output1); \
                output2 = mad(intputVector.s8, weightVector.lo, output2); \
                output3 = mad(intputVector.s9, weightVector.lo, output3); \
                output4 = mad(intputVector.sa, weightVector.lo, output4); \
                output5 = mad(intputVector.sb, weightVector.lo, output5); \
                output6 = mad(intputVector.sc, weightVector.lo, output6); \
                output7 = mad(intputVector.sd, weightVector.lo, output7); \
            } \
        } \
        int outputIndex = outputN * outputChannel * outputHeight * outputWidth + \
                            outputC * outputHeight * outputWidth + outputH * outputWidth + outputW; \
        FEED_DIRECT_OUTPUT_8X8(bias, outputC, outputW, outputChannel, outputHeight, outputWidth, output0, output1, \
                                output2, output3, output4, output5, output6, output7, output, outputIndex) \
    }

#define DIRECT9X9_4X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                      outputWidth, inputNum, inputChannel, inputHeight, inputWidth) \
    int outputN = get_global_id(2) / ((outputChannel + 3) / 4); \
    int outputC = get_global_id(2) % ((outputChannel + 3) / 4) * 4; \
    int outputH = get_global_id(1); \
    int outputW = get_global_id(0) * 8; \
    if (outputW < outputWidth && outputH < outputHeight) { \
        DATA_T4 output0 = 0.0f; \
        DATA_T4 output1 = 0.0f; \
        DATA_T4 output2 = 0.0f; \
        DATA_T4 output3 = 0.0f; \
        DATA_T4 output4 = 0.0f; \
        DATA_T4 output5 = 0.0f; \
        DATA_T4 output6 = 0.0f; \
        DATA_T4 output7 = 0.0f; \
        int inputHStart = max(outputH - padT, 0); \
        int inputHEnd = min(outputH - padT + 9, inputHeight); \
        int inputW = outputW - padL; \
        DATA_T16 inputMask = 1.0f; \
        SET_DIRECT_INPUT_MASK16(inputW, inputWidth, inputMask) \
        int inputBase = outputN * inputChannel * inputHeight * inputWidth + inputW; \
        int khOffset = (outputH - padT < 0) ? 9 - (inputHEnd - inputHStart) : 0; \
        int weightBase = outputC * inputChannel * 9 * 9 + khOffset * 4 * 9; \
        DATA_T16 intputVector; \
        DATA_T16 weightVector; \
        for (int inputC = 0; inputC < inputChannel; inputC++) { \
            int inputIndex = inputBase + inputC * inputHeight * inputWidth; \
            int weightIndex = weightBase + inputC * 4 * 9 * 9; \
            for (int kh = inputHStart; kh < inputHEnd; kh++) { \
                intputVector = vload16(0, input + inputIndex + kh * inputWidth); \
                intputVector = intputVector * inputMask; \
                weightVector = vload16(0, weight + weightIndex); \
                weightIndex += 16; \
                output0 = mad(intputVector.s0, weightVector.lo.lo, output0); \
                output1 = mad(intputVector.s1, weightVector.lo.lo, output1); \
                output2 = mad(intputVector.s2, weightVector.lo.lo, output2); \
                output3 = mad(intputVector.s3, weightVector.lo.lo, output3); \
                output4 = mad(intputVector.s4, weightVector.lo.lo, output4); \
                output5 = mad(intputVector.s5, weightVector.lo.lo, output5); \
                output6 = mad(intputVector.s6, weightVector.lo.lo, output6); \
                output7 = mad(intputVector.s7, weightVector.lo.lo, output7); \
                output0 = mad(intputVector.s1, weightVector.lo.hi, output0); \
                output1 = mad(intputVector.s2, weightVector.lo.hi, output1); \
                output2 = mad(intputVector.s3, weightVector.lo.hi, output2); \
                output3 = mad(intputVector.s4, weightVector.lo.hi, output3); \
                output4 = mad(intputVector.s5, weightVector.lo.hi, output4); \
                output5 = mad(intputVector.s6, weightVector.lo.hi, output5); \
                output6 = mad(intputVector.s7, weightVector.lo.hi, output6); \
                output7 = mad(intputVector.s8, weightVector.lo.hi, output7); \
                output0 = mad(intputVector.s2, weightVector.hi.lo, output0); \
                output1 = mad(intputVector.s3, weightVector.hi.lo, output1); \
                output2 = mad(intputVector.s4, weightVector.hi.lo, output2); \
                output3 = mad(intputVector.s5, weightVector.hi.lo, output3); \
                output4 = mad(intputVector.s6, weightVector.hi.lo, output4); \
                output5 = mad(intputVector.s7, weightVector.hi.lo, output5); \
                output6 = mad(intputVector.s8, weightVector.hi.lo, output6); \
                output7 = mad(intputVector.s9, weightVector.hi.lo, output7); \
                output0 = mad(intputVector.s3, weightVector.hi.hi, output0); \
                output1 = mad(intputVector.s4, weightVector.hi.hi, output1); \
                output2 = mad(intputVector.s5, weightVector.hi.hi, output2); \
                output3 = mad(intputVector.s6, weightVector.hi.hi, output3); \
                output4 = mad(intputVector.s7, weightVector.hi.hi, output4); \
                output5 = mad(intputVector.s8, weightVector.hi.hi, output5); \
                output6 = mad(intputVector.s9, weightVector.hi.hi, output6); \
                output7 = mad(intputVector.sa, weightVector.hi.hi, output7); \
                weightVector = vload16(0, weight + weightIndex); \
                weightIndex += 16; \
                output0 = mad(intputVector.s4, weightVector.lo.lo, output0); \
                output1 = mad(intputVector.s5, weightVector.lo.lo, output1); \
                output2 = mad(intputVector.s6, weightVector.lo.lo, output2); \
                output3 = mad(intputVector.s7, weightVector.lo.lo, output3); \
                output4 = mad(intputVector.s8, weightVector.lo.lo, output4); \
                output5 = mad(intputVector.s9, weightVector.lo.lo, output5); \
                output6 = mad(intputVector.sa, weightVector.lo.lo, output6); \
                output7 = mad(intputVector.sb, weightVector.lo.lo, output7); \
                output0 = mad(intputVector.s5, weightVector.lo.hi, output0); \
                output1 = mad(intputVector.s6, weightVector.lo.hi, output1); \
                output2 = mad(intputVector.s7, weightVector.lo.hi, output2); \
                output3 = mad(intputVector.s8, weightVector.lo.hi, output3); \
                output4 = mad(intputVector.s9, weightVector.lo.hi, output4); \
                output5 = mad(intputVector.sa, weightVector.lo.hi, output5); \
                output6 = mad(intputVector.sb, weightVector.lo.hi, output6); \
                output7 = mad(intputVector.sc, weightVector.lo.hi, output7); \
                output0 = mad(intputVector.s6, weightVector.hi.lo, output0); \
                output1 = mad(intputVector.s7, weightVector.hi.lo, output1); \
                output2 = mad(intputVector.s8, weightVector.hi.lo, output2); \
                output3 = mad(intputVector.s9, weightVector.hi.lo, output3); \
                output4 = mad(intputVector.sa, weightVector.hi.lo, output4); \
                output5 = mad(intputVector.sb, weightVector.hi.lo, output5); \
                output6 = mad(intputVector.sc, weightVector.hi.lo, output6); \
                output7 = mad(intputVector.sd, weightVector.hi.lo, output7); \
                output0 = mad(intputVector.s7, weightVector.hi.hi, output0); \
                output1 = mad(intputVector.s8, weightVector.hi.hi, output1); \
                output2 = mad(intputVector.s9, weightVector.hi.hi, output2); \
                output3 = mad(intputVector.sa, weightVector.hi.hi, output3); \
                output4 = mad(intputVector.sb, weightVector.hi.hi, output4); \
                output5 = mad(intputVector.sc, weightVector.hi.hi, output5); \
                output6 = mad(intputVector.sd, weightVector.hi.hi, output6); \
                output7 = mad(intputVector.se, weightVector.hi.hi, output7); \
                weightVector.lo.lo = vload4(0, weight + weightIndex); \
                weightIndex += 4; \
                output0 = mad(intputVector.s8, weightVector.lo.lo, output0); \
                output1 = mad(intputVector.s9, weightVector.lo.lo, output1); \
                output2 = mad(intputVector.sa, weightVector.lo.lo, output2); \
                output3 = mad(intputVector.sb, weightVector.lo.lo, output3); \
                output4 = mad(intputVector.sc, weightVector.lo.lo, output4); \
                output5 = mad(intputVector.sd, weightVector.lo.lo, output5); \
                output6 = mad(intputVector.se, weightVector.lo.lo, output6); \
                output7 = mad(intputVector.sf, weightVector.lo.lo, output7); \
            } \
        } \
        int outputIndex = outputN * outputChannel * outputHeight * outputWidth + \
                            outputC * outputHeight * outputWidth + outputH * outputWidth + \
                            outputW; \
        FEED_DIRECT_OUTPUT_4X8(bias, outputC, outputW, outputChannel, outputHeight, outputWidth, output0, output1, \
                                output2, output3, output4, output5, output6, output7, output, outputIndex) \
    }

#define DIRECT9X9_4X8_MERGE(bias, aligned_output, output, padT, padR, padB, padL, outputNumber, outputChannel, \
                            outputHeight, outputWidth, inputNum, inputChannel, inputHeight, inputWidth, splite_num) \
    int outputN = get_global_id(2) / ((outputChannel + 3) / 4); \
    int outputC = get_global_id(2) * 4; \
    int outputH = get_global_id(1); \
    int outputW = get_global_id(0) * 8; \
    if (outputW < outputWidth && outputH < outputHeight) { \
        DATA_T4 output0 = 0.0f; \
        DATA_T4 output1 = 0.0f; \
        DATA_T4 output2 = 0.0f; \
        DATA_T4 output3 = 0.0f; \
        DATA_T4 output4 = 0.0f; \
        DATA_T4 output5 = 0.0f; \
        DATA_T4 output6 = 0.0f; \
        DATA_T4 output7 = 0.0f; \
        int outputIndex = outputN * outputChannel * outputHeight * outputWidth + \
                          outputC * outputHeight * outputWidth + outputH * outputWidth + outputW; \
        int aligned_output_base = outputN * outputChannel * outputHeight * outputWidth / 32 + \
                          outputC / 4 * outputHeight * outputWidth / 8 + outputH * outputWidth / 8 + outputW / 8;; \
        aligned_output_base *= 32 * splite_num; \
        if (2 == splite_num) { \
            DATA_T16 r0 = vload16(0, aligned_output + aligned_output_base); \
            DATA_T16 r1 = vload16(1, aligned_output + aligned_output_base); \
            DATA_T16 r2 = vload16(2, aligned_output + aligned_output_base); \
            DATA_T16 r3 = vload16(3, aligned_output + aligned_output_base); \
            r0 += r2; \
            r1 += r3; \
            output0 = r0.lo.lo; \
            output1 = r0.lo.hi; \
            output2 = r0.hi.lo; \
            output3 = r0.hi.hi; \
            output4 = r1.lo.lo; \
            output5 = r1.lo.hi; \
            output6 = r1.hi.lo; \
            output7 = r1.hi.hi; \
        } else if (4 == splite_num) { \
            DATA_T16 r0 = vload16(0, aligned_output + aligned_output_base); \
            DATA_T16 r1 = vload16(1, aligned_output + aligned_output_base); \
            DATA_T16 r2 = vload16(2, aligned_output + aligned_output_base); \
            DATA_T16 r3 = vload16(3, aligned_output + aligned_output_base); \
            r0 += r2; \
            r1 += r3; \
            r2 = vload16(4, aligned_output + aligned_output_base); \
            r3 = vload16(5, aligned_output + aligned_output_base); \
            r0 += r2; \
            r1 += r3; \
            r2 = vload16(6, aligned_output + aligned_output_base); \
            r3 = vload16(7, aligned_output + aligned_output_base); \
            r0 += r2; \
            r1 += r3; \
            output0 = r0.lo.lo; \
            output1 = r0.lo.hi; \
            output2 = r0.hi.lo; \
            output3 = r0.hi.hi; \
            output4 = r1.lo.lo; \
            output5 = r1.lo.hi; \
            output6 = r1.hi.lo; \
            output7 = r1.hi.hi; \
        } else { \
            DATA_T16 r0 = vload16(0, aligned_output + aligned_output_base); \
            DATA_T16 r1 = vload16(1, aligned_output + aligned_output_base); \
            for (int i = 1; i < splite_num; ++i) { \
                    DATA_T16 r2 = vload16(i * 2, aligned_output + aligned_output_base); \
                    DATA_T16 r3 = vload16(i * 2 + 1, aligned_output + aligned_output_base); \
                    r0 += r2; \
                    r1 += r3; \
            } \
            output0 = r0.lo.lo; \
            output1 = r0.lo.hi; \
            output2 = r0.hi.lo; \
            output3 = r0.hi.hi; \
            output4 = r1.lo.lo; \
            output5 = r1.lo.hi; \
            output6 = r1.hi.lo; \
            output7 = r1.hi.hi; \
        } \
        FEED_DIRECT_OUTPUT_4X8(bias, outputC, outputW, outputChannel, outputHeight, outputWidth, output0, output1, \
                               output2, output3, output4, output5, output6, output7, output, outputIndex) \
    }

#define DIRECT9X9_8X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                      outputWidth, inputNum, inputChannel, inputHeight, inputWidth) \
    int outputN = get_global_id(2) / ((outputChannel + 7) / 8); \
    int outputC = get_global_id(2) % ((outputChannel + 7) / 8) * 8; \
    int outputH = get_global_id(1); \
    int outputW = get_global_id(0) * 8; \
    if (outputW < outputWidth && outputH < outputHeight) { \
        DATA_T8 output0 = 0.0f; \
        DATA_T8 output1 = 0.0f; \
        DATA_T8 output2 = 0.0f; \
        DATA_T8 output3 = 0.0f; \
        DATA_T8 output4 = 0.0f; \
        DATA_T8 output5 = 0.0f; \
        DATA_T8 output6 = 0.0f; \
        DATA_T8 output7 = 0.0f; \
        int inputHStart = max(outputH - padT, 0); \
        int inputHEnd = min(outputH - padT + 9, inputHeight); \
        int inputW = outputW - padL; \
        DATA_T16 inputMask = 1.0f; \
        SET_DIRECT_INPUT_MASK16(inputW, inputWidth, inputMask) \
        int inputBase = outputN * inputChannel * inputHeight * inputWidth + inputW; \
        int khOffset = (outputH - padT < 0) ? 9 - (inputHEnd - inputHStart) : 0; \
        int weightBase = outputC * inputChannel * 9 * 9 + khOffset * 8 * 9; \
        DATA_T16 intputVector; \
        DATA_T16 weightVector; \
        for (int inputC = 0; inputC < inputChannel; inputC++) { \
            int inputIndex = inputBase + inputC * inputHeight * inputWidth; \
            int weightIndex = weightBase + inputC * 8 * 9 * 9; \
            for (int kh = inputHStart; kh < inputHEnd; kh++) { \
                intputVector = vload16(0, input + inputIndex + kh * inputWidth); \
                intputVector = intputVector * inputMask; \
                weightVector = vload16(0, weight + weightIndex); \
                weightIndex += 16; \
                output0 = mad(intputVector.s0, weightVector.lo, output0); \
                output1 = mad(intputVector.s1, weightVector.lo, output1); \
                output2 = mad(intputVector.s2, weightVector.lo, output2); \
                output3 = mad(intputVector.s3, weightVector.lo, output3); \
                output4 = mad(intputVector.s4, weightVector.lo, output4); \
                output5 = mad(intputVector.s5, weightVector.lo, output5); \
                output6 = mad(intputVector.s6, weightVector.lo, output6); \
                output7 = mad(intputVector.s7, weightVector.lo, output7); \
                output0 = mad(intputVector.s1, weightVector.hi, output0); \
                output1 = mad(intputVector.s2, weightVector.hi, output1); \
                output2 = mad(intputVector.s3, weightVector.hi, output2); \
                output3 = mad(intputVector.s4, weightVector.hi, output3); \
                output4 = mad(intputVector.s5, weightVector.hi, output4); \
                output5 = mad(intputVector.s6, weightVector.hi, output5); \
                output6 = mad(intputVector.s7, weightVector.hi, output6); \
                output7 = mad(intputVector.s8, weightVector.hi, output7); \
                weightVector = vload16(0, weight + weightIndex); \
                weightIndex += 16; \
                output0 = mad(intputVector.s2, weightVector.lo, output0); \
                output1 = mad(intputVector.s3, weightVector.lo, output1); \
                output2 = mad(intputVector.s4, weightVector.lo, output2); \
                output3 = mad(intputVector.s5, weightVector.lo, output3); \
                output4 = mad(intputVector.s6, weightVector.lo, output4); \
                output5 = mad(intputVector.s7, weightVector.lo, output5); \
                output6 = mad(intputVector.s8, weightVector.lo, output6); \
                output7 = mad(intputVector.s9, weightVector.lo, output7); \
                output0 = mad(intputVector.s3, weightVector.hi, output0); \
                output1 = mad(intputVector.s4, weightVector.hi, output1); \
                output2 = mad(intputVector.s5, weightVector.hi, output2); \
                output3 = mad(intputVector.s6, weightVector.hi, output3); \
                output4 = mad(intputVector.s7, weightVector.hi, output4); \
                output5 = mad(intputVector.s8, weightVector.hi, output5); \
                output6 = mad(intputVector.s9, weightVector.hi, output6); \
                output7 = mad(intputVector.sa, weightVector.hi, output7); \
                weightVector = vload16(0, weight + weightIndex); \
                weightIndex += 16; \
                output0 = mad(intputVector.s4, weightVector.lo, output0); \
                output1 = mad(intputVector.s5, weightVector.lo, output1); \
                output2 = mad(intputVector.s6, weightVector.lo, output2); \
                output3 = mad(intputVector.s7, weightVector.lo, output3); \
                output4 = mad(intputVector.s8, weightVector.lo, output4); \
                output5 = mad(intputVector.s9, weightVector.lo, output5); \
                output6 = mad(intputVector.sa, weightVector.lo, output6); \
                output7 = mad(intputVector.sb, weightVector.lo, output7); \
                output0 = mad(intputVector.s5, weightVector.hi, output0); \
                output1 = mad(intputVector.s6, weightVector.hi, output1); \
                output2 = mad(intputVector.s7, weightVector.hi, output2); \
                output3 = mad(intputVector.s8, weightVector.hi, output3); \
                output4 = mad(intputVector.s9, weightVector.hi, output4); \
                output5 = mad(intputVector.sa, weightVector.hi, output5); \
                output6 = mad(intputVector.sb, weightVector.hi, output6); \
                output7 = mad(intputVector.sc, weightVector.hi, output7); \
                weightVector = vload16(0, weight + weightIndex); \
                weightIndex += 16; \
                output0 = mad(intputVector.s6, weightVector.lo, output0); \
                output1 = mad(intputVector.s7, weightVector.lo, output1); \
                output2 = mad(intputVector.s8, weightVector.lo, output2); \
                output3 = mad(intputVector.s9, weightVector.lo, output3); \
                output4 = mad(intputVector.sa, weightVector.lo, output4); \
                output5 = mad(intputVector.sb, weightVector.lo, output5); \
                output6 = mad(intputVector.sc, weightVector.lo, output6); \
                output7 = mad(intputVector.sd, weightVector.lo, output7); \
                output0 = mad(intputVector.s7, weightVector.hi, output0); \
                output1 = mad(intputVector.s8, weightVector.hi, output1); \
                output2 = mad(intputVector.s9, weightVector.hi, output2); \
                output3 = mad(intputVector.sa, weightVector.hi, output3); \
                output4 = mad(intputVector.sb, weightVector.hi, output4); \
                output5 = mad(intputVector.sc, weightVector.hi, output5); \
                output6 = mad(intputVector.sd, weightVector.hi, output6); \
                output7 = mad(intputVector.se, weightVector.hi, output7); \
                weightVector.lo = vload8(0, weight + weightIndex); \
                weightIndex += 8; \
                output0 = mad(intputVector.s8, weightVector.lo, output0); \
                output1 = mad(intputVector.s9, weightVector.lo, output1); \
                output2 = mad(intputVector.sa, weightVector.lo, output2); \
                output3 = mad(intputVector.sb, weightVector.lo, output3); \
                output4 = mad(intputVector.sc, weightVector.lo, output4); \
                output5 = mad(intputVector.sd, weightVector.lo, output5); \
                output6 = mad(intputVector.se, weightVector.lo, output6); \
                output7 = mad(intputVector.sf, weightVector.lo, output7); \
            } \
        } \
        int outputIndex = outputN * outputChannel * outputHeight * outputWidth + \
                            outputC * outputHeight * outputWidth + outputH * outputWidth + \
                            outputW; \
        FEED_DIRECT_OUTPUT_8X8(bias, outputC, outputW, outputChannel, outputHeight, outputWidth, output0, output1, \
                            output2, output3, output4, output5, output6, output7, output, outputIndex) \
    }

#define DIRECT3X3_4X4(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, \
                      outputHeight, outputWidth, inputNum, inputChannel, inputHeight, inputWidth) \
    int outputN = get_global_id(2) / ((outputChannel + 3) / 4); \
    int outputC = get_global_id(2) % ((outputChannel + 3) / 4) * 4; \
    int outputH = get_global_id(1); \
    int outputW = get_global_id(0) * 4; \
    if (outputW < outputWidth && outputH < outputHeight) { \
        DATA_T4 output0 = 0.0f; \
        DATA_T4 output1 = 0.0f; \
        DATA_T4 output2 = 0.0f; \
        DATA_T4 output3 = 0.0f; \
        int inputHStart = max(outputH - padT, 0); \
        int inputHEnd = min(outputH - padT + 3, inputHeight); \
        int inputW = outputW - padL; \
        DATA_T8 inputMask = 1.0f; \
        if (inputW < 0 || inputW >= inputWidth) { \
            inputMask.s0 = 0.0f; \
        } \
        if (inputW + 1 < 0 || inputW + 1 >= inputWidth) { \
            inputMask.s1 = 0.0f; \
        } \
        if (inputW + 2 < 0 || inputW + 2 >= inputWidth) { \
            inputMask.s2 = 0.0f; \
        } \
        if (inputW + 3 < 0 || inputW + 3 >= inputWidth) { \
            inputMask.s3 = 0.0f; \
        } \
        if (inputW + 4 < 0 || inputW + 4 >= inputWidth) { \
            inputMask.s4 = 0.0f; \
        } \
        if (inputW + 5 < 0 || inputW + 5 >= inputWidth) { \
            inputMask.s5 = 0.0f; \
        } \
        if (inputW + 6 < 0 || inputW + 6 >= inputWidth) { \
            inputMask.s6 = 0.0f; \
        } \
        if (inputW + 7 < 0 || inputW + 7 >= inputWidth) { \
            inputMask.s7 = 0.0f; \
        } \
        int inputBase = outputN * inputChannel * inputHeight * inputWidth + inputW; \
        int khOffset = (outputH - padT < 0) ? 3 - (inputHEnd - inputHStart) : 0; \
        int weightBase = outputC * inputChannel * 3 * 3 + khOffset * 4 * 3; \
        DATA_T8 intputVector; \
        DATA_T8 weightVector; \
        for (int inputC = 0; inputC < inputChannel; inputC++) { \
            int inputIndex = inputBase + inputC * inputHeight * inputWidth; \
            int weightIndex = weightBase + inputC * 4 * 3 * 3; \
            for (int kh = inputHStart; kh < inputHEnd; kh++) { \
                intputVector = vload8(0, input + inputIndex + kh * inputWidth); \
                intputVector = intputVector * inputMask; \
                weightVector = vload8(0, weight + weightIndex); \
                weightIndex += 8; \
                output0 = mad(intputVector.s0, weightVector.lo, output0); \
                output1 = mad(intputVector.s1, weightVector.lo, output1); \
                output2 = mad(intputVector.s2, weightVector.lo, output2); \
                output3 = mad(intputVector.s3, weightVector.lo, output3); \
                output0 = mad(intputVector.s1, weightVector.hi, output0); \
                output1 = mad(intputVector.s2, weightVector.hi, output1); \
                output2 = mad(intputVector.s3, weightVector.hi, output2); \
                output3 = mad(intputVector.s4, weightVector.hi, output3); \
                weightVector = vload8(0, weight + weightIndex); \
                weightIndex += 4; \
                output0 = mad(intputVector.s2, weightVector.lo, output0); \
                output1 = mad(intputVector.s3, weightVector.lo, output1); \
                output2 = mad(intputVector.s4, weightVector.lo, output2); \
                output3 = mad(intputVector.s5, weightVector.lo, output3); \
            } \
        } \
        int outputIndex = outputN * outputChannel * outputHeight * outputWidth + \
                            outputC * outputHeight * outputWidth + outputH * outputWidth + \
                            outputW; \
        DATA_T4 biasVector = vload4(0, bias + outputC); \
        output0 += biasVector; \
        output1 += biasVector; \
        output2 += biasVector; \
        output3 += biasVector; \
        if (outputW + 3 < outputWidth) { \
            if (outputC + 3 < outputChannel) { \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s0, output1.s0, output2.s0, output3.s0)), \
                        0, \
                        output + outputIndex); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s1, output1.s1, output2.s1, output3.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s2, output1.s2, output2.s2, output3.s2)), \
                        0, \
                        output + outputIndex + 2 * outputHeight * outputWidth); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s3, output1.s3, output2.s3, output3.s3)), \
                        0, \
                        output + outputIndex + 3 * outputHeight * outputWidth); \
            } else if (outputC + 2 < outputChannel) { \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s0, output1.s0, output2.s0, output3.s0)), \
                        0, \
                        output + outputIndex); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s1, output1.s1, output2.s1, output3.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s2, output1.s2, output2.s2, output3.s2)), \
                        0, \
                        output + outputIndex + 2 * outputHeight * outputWidth); \
            } else if (outputC + 1 < outputChannel) { \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s0, output1.s0, output2.s0, output3.s0)), \
                        0, \
                        output + outputIndex); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s1, output1.s1, output2.s1, output3.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
            } else if (outputC < outputChannel) { \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s0, output1.s0, output2.s0, output3.s0)), \
                        0, \
                        output + outputIndex); \
            } \
        } else if (outputW + 2 < outputWidth) { \
            if (outputC + 3 < outputChannel) { \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s0, output1.s0, output2.s0)), 0, output + outputIndex); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s1, output1.s1, output2.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s2, output1.s2, output2.s2)), \
                        0, \
                        output + outputIndex + 2 * outputHeight * outputWidth); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s3, output1.s3, output2.s3)), \
                        0, \
                        output + outputIndex + 3 * outputHeight * outputWidth); \
            } else if (outputC + 2 < outputChannel) { \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s0, output1.s0, output2.s0)), 0, output + outputIndex); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s1, output1.s1, output2.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s2, output1.s2, output2.s2)), \
                        0, \
                        output + outputIndex + 2 * outputHeight * outputWidth); \
            } else if (outputC + 1 < outputChannel) { \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s0, output1.s0, output2.s0)), 0, output + outputIndex); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s1, output1.s1, output2.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
            } else if (outputC < outputChannel) { \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s0, output1.s0, output2.s0)), 0, output + outputIndex); \
            } \
        } else if (outputW + 1 < outputWidth) { \
            if (outputC + 3 < outputChannel) { \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s0, output1.s0)), 0, output + outputIndex); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s1, output1.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s2, output1.s2)), \
                        0, \
                        output + outputIndex + 2 * outputHeight * outputWidth); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s3, output1.s3)), \
                        0, \
                        output + outputIndex + 3 * outputHeight * outputWidth); \
            } else if (outputC + 2 < outputChannel) { \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s0, output1.s0)), 0, output + outputIndex); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s1, output1.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s2, output1.s2)), \
                        0, \
                        output + outputIndex + 2 * outputHeight * outputWidth); \
            } else if (outputC + 1 < outputChannel) { \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s0, output1.s0)), 0, output + outputIndex); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s1, output1.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
            } else if (outputC < outputChannel) { \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s0, output1.s0)), 0, output + outputIndex); \
            } \
        } else if (outputW < outputWidth) { \
            if (outputC + 3 < outputChannel) { \
                output[outputIndex] = ACT_VEC_F(DATA_T, output0.s0); \
                output[outputIndex + outputHeight * outputWidth] = ACT_VEC_F(DATA_T, output0.s1); \
                output[outputIndex + 2 * outputHeight * outputWidth] = ACT_VEC_F(DATA_T, output0.s2); \
                output[outputIndex + 3 * outputHeight * outputWidth] = ACT_VEC_F(DATA_T, output0.s3); \
            } else if (outputC + 2 < outputChannel) { \
                output[outputIndex] = ACT_VEC_F(DATA_T, output0.s0); \
                output[outputIndex + outputHeight * outputWidth] = ACT_VEC_F(DATA_T, output0.s1); \
                output[outputIndex + 2 * outputHeight * outputWidth] = ACT_VEC_F(DATA_T, output0.s2); \
            } else if (outputC + 1 < outputChannel) { \
                output[outputIndex] = ACT_VEC_F(DATA_T, output0.s0); \
                output[outputIndex + outputHeight * outputWidth] = ACT_VEC_F(DATA_T, output0.s1); \
            } else if (outputC < outputChannel) { \
                output[outputIndex] = ACT_VEC_F(DATA_T, output0.s0); \
            } \
        } \
    }

#define DIRECT3X3_4X2X2_FP16(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel,         \
                             outputHeight, outputWidth, inputNum, inputChannel, inputHeight, inputWidth)               \
    int outputN = get_global_id(2) / ((outputChannel + 3) / 4);                                                        \
    int outputC = get_global_id(2) % ((outputChannel + 3) / 4) * 4;                                                    \
    int outputH = get_global_id(1) * 2;                                                                                \
    int outputW = get_global_id(0) * 2;                                                                                \
    if (outputW < outputWidth && outputH < outputHeight) {                                                             \
        DATA_T4 output00 = 0.0f;                                                                                       \
        DATA_T4 output01 = 0.0f;                                                                                       \
        DATA_T4 output10 = 0.0f;                                                                                       \
        DATA_T4 output11 = 0.0f;                                                                                       \
        int inputHStart = max(outputH - padT, 0);                                                                      \
        int inputHEnd = min(outputH - padT + 3, inputHeight);                                                          \
        int inputH = outputH - padT;                                                                                   \
        int inputW = outputW - padL;                                                                                   \
        DATA_T4 inputMask = 1.0f;                                                                                      \
        if (inputW < 0 || inputW >= inputWidth) {                                                                      \
            inputMask.s0 = 0.0f;                                                                                       \
        }                                                                                                              \
        if (inputW + 1 < 0 || inputW + 1 >= inputWidth) {                                                              \
            inputMask.s1 = 0.0f;                                                                                       \
        }                                                                                                              \
        if (inputW + 2 < 0 || inputW + 2 >= inputWidth) {                                                              \
            inputMask.s2 = 0.0f;                                                                                       \
        }                                                                                                              \
        if (inputW + 3 < 0 || inputW + 3 >= inputWidth) {                                                              \
            inputMask.s3 = 0.0f;                                                                                       \
        }                                                                                                              \
        DATA_T4 inputMask1 = inputMask;                                                                                \
        DATA_T4 inputMask2 = inputMask;                                                                                \
        DATA_T4 inputMask3 = inputMask;                                                                                \
        DATA_T4 inputMask4 = inputMask;                                                                                \
        if (inputH < 0 || inputH >= inputHeight) {                                                                     \
            inputMask1 = 0.0f;                                                                                         \
        }                                                                                                              \
        if (inputH + 1 < 0 || inputH + 1 >= inputHeight) {                                                             \
            inputMask2 = 0.0f;                                                                                         \
        }                                                                                                              \
        if (inputH + 2 < 0 || inputH + 2 >= inputHeight) {                                                             \
            inputMask3 = 0.0f;                                                                                         \
        }                                                                                                              \
        if (inputH + 3 < 0 || inputH + 3 >= inputHeight) {                                                             \
            inputMask4 = 0.0f;                                                                                         \
        }                                                                                                              \
        int inputBase = outputN * inputChannel * inputHeight * inputWidth + inputW;                                    \
        int weightBase = outputC * inputChannel * 3 * 3;                                                               \
        DATA_T4 intputVector0;                                                                                         \
        DATA_T4 intputVector1;                                                                                         \
        DATA_T4 intputVector2;                                                                                         \
        DATA_T4 intputVector3;                                                                                         \
        DATA_T16 weightVector0;                                                                                        \
        DATA_T16 weightVector1;                                                                                        \
        DATA_T4 weightVector2;                                                                                         \
        for (int inputC = 0; inputC < inputChannel; inputC++) {                                                        \
            int inputIndex = inputBase + inputC * inputHeight * inputWidth;                                            \
            int weightIndex = weightBase + inputC * 4 * 3 * 3;                                                         \
            {                                                                                                          \
                intputVector0 = vload4(0, input + inputIndex + inputH * inputWidth);                                   \
                intputVector0 = intputVector0 * inputMask1;                                                            \
                intputVector1 = vload4(0, input + inputIndex + (inputH + 1) * inputWidth);                             \
                intputVector1 = intputVector1 * inputMask2;                                                            \
                intputVector2 = vload4(0, input + inputIndex + (inputH + 2) * inputWidth);                             \
                intputVector2 = intputVector2 * inputMask3;                                                            \
                intputVector3 = vload4(0, input + inputIndex + (inputH + 3) * inputWidth);                             \
                intputVector3 = intputVector3 * inputMask4;                                                            \
                weightVector0 = vload16(0, weight + weightIndex);                                                      \
                weightIndex += 16;                                                                                     \
                weightVector1 = vload16(0, weight + weightIndex);                                                      \
                weightIndex += 16;                                                                                     \
                weightVector2 = vload4(0, weight + weightIndex);                                                       \
                weightIndex += 4;                                                                                      \
                output00 = mad(intputVector0.s0, weightVector0.lo.lo, output00);                                       \
                output01 = mad(intputVector0.s1, weightVector0.lo.lo, output01);                                       \
                output00 = mad(intputVector0.s1, weightVector0.lo.hi, output00);                                       \
                output01 = mad(intputVector0.s2, weightVector0.lo.hi, output01);                                       \
                output00 = mad(intputVector0.s2, weightVector0.hi.lo, output00);                                       \
                output01 = mad(intputVector0.s3, weightVector0.hi.lo, output01);                                       \
                output10 = mad(intputVector1.s0, weightVector0.lo.lo, output10);                                       \
                output11 = mad(intputVector1.s1, weightVector0.lo.lo, output11);                                       \
                output10 = mad(intputVector1.s1, weightVector0.lo.hi, output10);                                       \
                output11 = mad(intputVector1.s2, weightVector0.lo.hi, output11);                                       \
                output10 = mad(intputVector1.s2, weightVector0.hi.lo, output10);                                       \
                output11 = mad(intputVector1.s3, weightVector0.hi.lo, output11);                                       \
                output00 = mad(intputVector1.s0, weightVector0.hi.hi, output00);                                       \
                output01 = mad(intputVector1.s1, weightVector0.hi.hi, output01);                                       \
                output00 = mad(intputVector1.s1, weightVector1.lo.lo, output00);                                       \
                output01 = mad(intputVector1.s2, weightVector1.lo.lo, output01);                                       \
                output00 = mad(intputVector1.s2, weightVector1.lo.hi, output00);                                       \
                output01 = mad(intputVector1.s3, weightVector1.lo.hi, output01);                                       \
                output10 = mad(intputVector2.s0, weightVector0.hi.hi, output10);                                       \
                output11 = mad(intputVector2.s1, weightVector0.hi.hi, output11);                                       \
                output10 = mad(intputVector2.s1, weightVector1.lo.lo, output10);                                       \
                output11 = mad(intputVector2.s2, weightVector1.lo.lo, output11);                                       \
                output10 = mad(intputVector2.s2, weightVector1.lo.hi, output10);                                       \
                output11 = mad(intputVector2.s3, weightVector1.lo.hi, output11);                                       \
                output00 = mad(intputVector2.s0, weightVector1.hi.lo, output00);                                       \
                output01 = mad(intputVector2.s1, weightVector1.hi.lo, output01);                                       \
                output00 = mad(intputVector2.s1, weightVector1.hi.hi, output00);                                       \
                output01 = mad(intputVector2.s2, weightVector1.hi.hi, output01);                                       \
                output00 = mad(intputVector2.s2, weightVector2, output00);                                             \
                output01 = mad(intputVector2.s3, weightVector2, output01);                                             \
                output10 = mad(intputVector3.s0, weightVector1.hi.lo, output10);                                       \
                output11 = mad(intputVector3.s1, weightVector1.hi.lo, output11);                                       \
                output10 = mad(intputVector3.s1, weightVector1.hi.hi, output10);                                       \
                output11 = mad(intputVector3.s2, weightVector1.hi.hi, output11);                                       \
                output10 = mad(intputVector3.s2, weightVector2, output10);                                             \
                output11 = mad(intputVector3.s3, weightVector2, output11);                                             \
            }                                                                                                          \
        }                                                                                                              \
        int outputIndex = outputN * outputChannel * outputHeight * outputWidth +                                       \
                          outputC * outputHeight * outputWidth + outputH * outputWidth + outputW;                      \
        DATA_T4 biasVector = vload4(0, bias + outputC);                                                                \
        output00 += biasVector;                                                                                        \
        output01 += biasVector;                                                                                        \
        output10 += biasVector;                                                                                        \
        output11 += biasVector;                                                                                        \
        if (outputW + 1 < outputWidth) {                                                                               \
            if (outputC + 3 < outputChannel) {                                                                         \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output00.s0, output01.s0)), 0, output + outputIndex);             \
                vstore2(                                                                                               \
                    ACT_VEC_F(DATA_T2, (DATA_T2)(output10.s0, output11.s0)), 0, output + outputIndex + outputWidth);   \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output00.s1, output01.s1)),                                       \
                        0,                                                                                             \
                        output + outputIndex + outputHeight * outputWidth);                                            \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output10.s1, output11.s1)),                                       \
                        0,                                                                                             \
                        output + outputIndex + (outputHeight + 1) * outputWidth);                                      \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output00.s2, output01.s2)),                                       \
                        0,                                                                                             \
                        output + outputIndex + 2 * outputHeight * outputWidth);                                        \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output10.s2, output11.s2)),                                       \
                        0,                                                                                             \
                        output + outputIndex + 2 * outputHeight * outputWidth + outputWidth);                          \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output00.s3, output01.s3)),                                       \
                        0,                                                                                             \
                        output + outputIndex + 3 * outputHeight * outputWidth);                                        \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output10.s3, output11.s3)),                                       \
                        0,                                                                                             \
                        output + outputIndex + 3 * outputHeight * outputWidth + outputWidth);                          \
            } else if (outputC + 2 < outputChannel) {                                                                  \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output00.s0, output01.s0)), 0, output + outputIndex);             \
                vstore2(                                                                                               \
                    ACT_VEC_F(DATA_T2, (DATA_T2)(output10.s0, output11.s0)), 0, output + outputIndex + outputWidth);   \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output00.s1, output01.s1)),                                       \
                        0,                                                                                             \
                        output + outputIndex + outputHeight * outputWidth);                                            \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output10.s1, output11.s1)),                                       \
                        0,                                                                                             \
                        output + outputIndex + (outputHeight + 1) * outputWidth);                                      \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output00.s2, output01.s2)),                                       \
                        0,                                                                                             \
                        output + outputIndex + 2 * outputHeight * outputWidth);                                        \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output10.s2, output11.s2)),                                       \
                        0,                                                                                             \
                        output + outputIndex + 2 * outputHeight * outputWidth + outputWidth);                          \
            } else if (outputC + 1 < outputChannel) {                                                                  \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output00.s0, output01.s0)), 0, output + outputIndex);             \
                vstore2(                                                                                               \
                    ACT_VEC_F(DATA_T2, (DATA_T2)(output10.s0, output11.s0)), 0, output + outputIndex + outputWidth);   \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output00.s1, output01.s1)),                                       \
                        0,                                                                                             \
                        output + outputIndex + outputHeight * outputWidth);                                            \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output10.s1, output11.s1)),                                       \
                        0,                                                                                             \
                        output + outputIndex + (outputHeight + 1) * outputWidth);                                      \
            } else if (outputC < outputChannel) {                                                                      \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output00.s0, output01.s0)), 0, output + outputIndex);             \
                vstore2(                                                                                               \
                    ACT_VEC_F(DATA_T2, (DATA_T2)(output10.s0, output11.s0)), 0, output + outputIndex + outputWidth);   \
            }                                                                                                          \
        }                                                                                                              \
    }

#define DIRECT3X3_4X4X2_FP16(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, \
                             outputHeight, outputWidth, inputNum, inputChannel, inputHeight, inputWidth) \
    int outputN = get_global_id(2) / ((outputChannel + 3) / 4); \
    int outputC = get_global_id(2) % ((outputChannel + 3) / 4) * 4; \
    int outputH = get_global_id(1) * 2; \
    int outputW = get_global_id(0) * 4; \
    if (outputW < outputWidth && outputH < outputHeight) { \
        DATA_T4 output00 = 0.0f; \
        DATA_T4 output01 = 0.0f; \
        DATA_T4 output02 = 0.0f; \
        DATA_T4 output03 = 0.0f; \
        DATA_T4 output10 = 0.0f; \
        DATA_T4 output11 = 0.0f; \
        DATA_T4 output12 = 0.0f; \
        DATA_T4 output13 = 0.0f; \
        int inputH = outputH - padT; \
        int inputW = outputW - padL; \
        DATA_T4 inputMask = 1.0f; \
        DATA_T2 inputMask_ = 1.0f; \
        if (inputW < 0 || inputW >= inputWidth) { \
            inputMask.s0 = 0.0f; \
        } \
        if (inputW + 1 < 0 || inputW + 1 >= inputWidth) { \
            inputMask.s1 = 0.0f; \
        } \
        if (inputW + 2 < 0 || inputW + 2 >= inputWidth) { \
            inputMask.s2 = 0.0f; \
        } \
        if (inputW + 3 < 0 || inputW + 3 >= inputWidth) { \
            inputMask.s3 = 0.0f; \
        } \
        if (inputW + 4 < 0 || inputW + 4 >= inputWidth) { \
            inputMask_.s0 = 0.0f; \
        } \
        if (inputW + 5 < 0 || inputW + 5 >= inputWidth) { \
            inputMask_.s1 = 0.0f; \
        } \
        DATA_T4 inputMask1 = inputMask; \
        DATA_T4 inputMask2 = inputMask; \
        DATA_T4 inputMask3 = inputMask; \
        DATA_T4 inputMask4 = inputMask; \
        DATA_T2 inputMask1_ = inputMask_; \
        DATA_T2 inputMask2_ = inputMask_; \
        DATA_T2 inputMask3_ = inputMask_; \
        DATA_T2 inputMask4_ = inputMask_; \
        if (inputH < 0 || inputH >= inputHeight) { \
            inputMask1 = 0.0f; \
            inputMask1_ = 0.0f; \
        } \
        if (inputH + 1 < 0 || inputH + 1 >= inputHeight) { \
            inputMask2 = 0.0f; \
            inputMask2_ = 0.0f; \
        } \
        if (inputH + 2 < 0 || inputH + 2 >= inputHeight) { \
            inputMask3 = 0.0f; \
            inputMask3_ = 0.0f; \
        } \
        if (inputH + 3 < 0 || inputH + 3 >= inputHeight) { \
            inputMask4 = 0.0f; \
            inputMask4_ = 0.0f; \
        } \
        int inputBase = outputN * inputChannel * inputHeight * inputWidth + inputW; \
        int weightBase = outputC * inputChannel * 3 * 3; \
        DATA_T4 intputVector0; \
        DATA_T4 intputVector1; \
        DATA_T4 intputVector2; \
        DATA_T4 intputVector3; \
        DATA_T2 intputVector0_; \
        DATA_T2 intputVector1_; \
        DATA_T2 intputVector2_; \
        DATA_T2 intputVector3_; \
        DATA_T16 weightVector0; \
        DATA_T16 weightVector1; \
        DATA_T4 weightVector2; \
        for (int inputC = 0; inputC < inputChannel; inputC++) { \
            int inputIndex = inputBase + inputC * inputHeight * inputWidth; \
            int weightIndex = weightBase + inputC * 4 * 3 * 3; \
            { \
                intputVector0 = vload4(0, input + inputIndex + inputH * inputWidth); \
                intputVector0 = intputVector0 * inputMask1; \
                intputVector0_ = vload2(0, input + inputIndex + inputH * inputWidth + 4); \
                intputVector0_ = intputVector0_ * inputMask1_; \
                intputVector1 = vload4(0, input + inputIndex + (inputH + 1) * inputWidth); \
                intputVector1 = intputVector1 * inputMask2; \
                intputVector1_ = vload2(0, input + inputIndex + (inputH + 1) * inputWidth + 4); \
                intputVector1_ = intputVector1_ * inputMask2_; \
                weightVector0 = vload16(0, weight + weightIndex); \
                weightIndex += 16; \
                output00 = mad(intputVector0.s0, weightVector0.lo.lo, output00); \
                output01 = mad(intputVector0.s1, weightVector0.lo.lo, output01); \
                output02 = mad(intputVector0.s2, weightVector0.lo.lo, output02); \
                output03 = mad(intputVector0.s3, weightVector0.lo.lo, output03); \
                output00 = mad(intputVector0.s1, weightVector0.lo.hi, output00); \
                output01 = mad(intputVector0.s2, weightVector0.lo.hi, output01); \
                output02 = mad(intputVector0.s3, weightVector0.lo.hi, output02); \
                output03 = mad(intputVector0_.s0, weightVector0.lo.hi, output03); \
                output00 = mad(intputVector0.s2, weightVector0.hi.lo, output00); \
                output01 = mad(intputVector0.s3, weightVector0.hi.lo, output01); \
                output02 = mad(intputVector0_.s0, weightVector0.hi.lo, output02); \
                output03 = mad(intputVector0_.s1, weightVector0.hi.lo, output03); \
                output10 = mad(intputVector1.s0, weightVector0.lo.lo, output10); \
                output11 = mad(intputVector1.s1, weightVector0.lo.lo, output11); \
                output12 = mad(intputVector1.s2, weightVector0.lo.lo, output12); \
                output13 = mad(intputVector1.s3, weightVector0.lo.lo, output13); \
                output10 = mad(intputVector1.s1, weightVector0.lo.hi, output10); \
                output11 = mad(intputVector1.s2, weightVector0.lo.hi, output11); \
                output12 = mad(intputVector1.s3, weightVector0.lo.hi, output12); \
                output13 = mad(intputVector1_.s0, weightVector0.lo.hi, output13); \
                output10 = mad(intputVector1.s2, weightVector0.hi.lo, output10); \
                output11 = mad(intputVector1.s3, weightVector0.hi.lo, output11); \
                output12 = mad(intputVector1_.s0, weightVector0.hi.lo, output12); \
                output13 = mad(intputVector1_.s1, weightVector0.hi.lo, output13); \
                intputVector2 = vload4(0, input + inputIndex + (inputH + 2) * inputWidth); \
                intputVector2 = intputVector2 * inputMask3; \
                intputVector2_ = vload2(0, input + inputIndex + (inputH + 2) * inputWidth + 4); \
                intputVector2_ = intputVector2_ * inputMask3_; \
                weightVector1 = vload16(0, weight + weightIndex); \
                weightIndex += 16; \
                output00 = mad(intputVector1.s0, weightVector0.hi.hi, output00); \
                output01 = mad(intputVector1.s1, weightVector0.hi.hi, output01); \
                output02 = mad(intputVector1.s2, weightVector0.hi.hi, output02); \
                output03 = mad(intputVector1.s3, weightVector0.hi.hi, output03); \
                output00 = mad(intputVector1.s1, weightVector1.lo.lo, output00); \
                output01 = mad(intputVector1.s2, weightVector1.lo.lo, output01); \
                output02 = mad(intputVector1.s3, weightVector1.lo.lo, output02); \
                output03 = mad(intputVector1_.s0, weightVector1.lo.lo, output03); \
                output00 = mad(intputVector1.s2, weightVector1.lo.hi, output00); \
                output01 = mad(intputVector1.s3, weightVector1.lo.hi, output01); \
                output02 = mad(intputVector1_.s0, weightVector1.lo.hi, output02); \
                output03 = mad(intputVector1_.s1, weightVector1.lo.hi, output03); \
                output10 = mad(intputVector2.s0, weightVector0.hi.hi, output10); \
                output11 = mad(intputVector2.s1, weightVector0.hi.hi, output11); \
                output12 = mad(intputVector2.s2, weightVector0.hi.hi, output12); \
                output13 = mad(intputVector2.s3, weightVector0.hi.hi, output13); \
                output10 = mad(intputVector2.s1, weightVector1.lo.lo, output10); \
                output11 = mad(intputVector2.s2, weightVector1.lo.lo, output11); \
                output12 = mad(intputVector2.s3, weightVector1.lo.lo, output12); \
                output13 = mad(intputVector2_.s0, weightVector1.lo.lo, output13); \
                output10 = mad(intputVector2.s2, weightVector1.lo.hi, output10); \
                output11 = mad(intputVector2.s3, weightVector1.lo.hi, output11); \
                output12 = mad(intputVector2_.s0, weightVector1.lo.hi, output12); \
                output13 = mad(intputVector2_.s1, weightVector1.lo.hi, output13); \
                intputVector3 = vload4(0, input + inputIndex + (inputH + 3) * inputWidth); \
                intputVector3 = intputVector3 * inputMask4; \
                intputVector3_ = vload2(0, input + inputIndex + (inputH + 3) * inputWidth + 4); \
                intputVector3_ = intputVector3_ * inputMask4_; \
                weightVector2 = vload4(0, weight + weightIndex); \
                weightIndex += 4; \
                output00 = mad(intputVector2.s0, weightVector1.hi.lo, output00); \
                output01 = mad(intputVector2.s1, weightVector1.hi.lo, output01); \
                output02 = mad(intputVector2.s2, weightVector1.hi.lo, output02); \
                output03 = mad(intputVector2.s3, weightVector1.hi.lo, output03); \
                output00 = mad(intputVector2.s1, weightVector1.hi.hi, output00); \
                output01 = mad(intputVector2.s2, weightVector1.hi.hi, output01); \
                output02 = mad(intputVector2.s3, weightVector1.hi.hi, output02); \
                output03 = mad(intputVector2_.s0, weightVector1.hi.hi, output03); \
                output00 = mad(intputVector2.s2, weightVector2, output00); \
                output01 = mad(intputVector2.s3, weightVector2, output01); \
                output02 = mad(intputVector2_.s0, weightVector2, output02); \
                output03 = mad(intputVector2_.s1, weightVector2, output03); \
                output10 = mad(intputVector3.s0, weightVector1.hi.lo, output10); \
                output11 = mad(intputVector3.s1, weightVector1.hi.lo, output11); \
                output12 = mad(intputVector3.s2, weightVector1.hi.lo, output12); \
                output13 = mad(intputVector3.s3, weightVector1.hi.lo, output13); \
                output10 = mad(intputVector3.s1, weightVector1.hi.hi, output10); \
                output11 = mad(intputVector3.s2, weightVector1.hi.hi, output11); \
                output12 = mad(intputVector3.s3, weightVector1.hi.hi, output12); \
                output13 = mad(intputVector3_.s0, weightVector1.hi.hi, output13); \
                output10 = mad(intputVector3.s2, weightVector2, output10); \
                output11 = mad(intputVector3.s3, weightVector2, output11); \
                output12 = mad(intputVector3_.s0, weightVector2, output12); \
                output13 = mad(intputVector3_.s1, weightVector2, output13); \
            } \
        } \
        int outputIndex = outputN * outputChannel * outputHeight * outputWidth + \
                            outputC * outputHeight * outputWidth + outputH * outputWidth + outputW; \
        DATA_T4 biasVector = vload4(0, bias + outputC); \
        output00 += biasVector; \
        output01 += biasVector; \
        output02 += biasVector; \
        output03 += biasVector; \
        output10 += biasVector; \
        output11 += biasVector; \
        output12 += biasVector; \
        output13 += biasVector; \
        if (outputW + 3 < outputWidth) { \
            if (outputC + 3 < outputChannel) { \
                    vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output00.s0, output01.s0, output02.s0, output03.s0)), \
                        0, \
                        output + outputIndex); \
                    vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output10.s0, output11.s0, output12.s0, output13.s0)), \
                        0, \
                        output + outputIndex + outputWidth); \
                    vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output00.s1, output01.s1, output02.s1, output03.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
                    vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output10.s1, output11.s1, output12.s1, output13.s1)), \
                        0, \
                        output + outputIndex + (outputHeight + 1) * outputWidth); \
                    vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output00.s2, output01.s2, output02.s2, output03.s2)), \
                        0, \
                        output + outputIndex + 2 * outputHeight * outputWidth); \
                    vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output10.s2, output11.s2, output12.s2, output13.s2)), \
                        0, \
                        output + outputIndex + 2 * outputHeight * outputWidth + outputWidth); \
                    vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output00.s3, output01.s3, output02.s3, output03.s3)), \
                        0, \
                        output + outputIndex + 3 * outputHeight * outputWidth); \
                    vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output10.s3, output11.s3, output12.s3, output13.s3)), \
                        0, \
                        output + outputIndex + 3 * outputHeight * outputWidth + outputWidth); \
            } else if (outputC + 2 < outputChannel) { \
                    vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output00.s0, output01.s0, output02.s0, output03.s0)), \
                        0, \
                        output + outputIndex); \
                    vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output10.s0, output11.s0, output12.s0, output13.s0)), \
                        0, \
                        output + outputIndex + outputWidth); \
                    vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output00.s1, output01.s1, output02.s1, output03.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
                    vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output10.s1, output11.s1, output12.s1, output13.s1)), \
                        0, \
                        output + outputIndex + (outputHeight + 1) * outputWidth); \
                    vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output00.s2, output01.s2, output02.s2, output03.s2)), \
                        0, \
                        output + outputIndex + 2 * outputHeight * outputWidth); \
                    vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output10.s2, output11.s2, output12.s2, output13.s2)), \
                        0, \
                        output + outputIndex + 2 * outputHeight * outputWidth + outputWidth); \
            } else if (outputC + 1 < outputChannel) { \
                    vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output00.s0, output01.s0, output02.s0, output03.s0)), \
                        0, \
                        output + outputIndex); \
                    vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output10.s0, output11.s0, output12.s0, output13.s0)), \
                        0, \
                        output + outputIndex + outputWidth); \
                    vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output00.s1, output01.s1, output02.s1, output03.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
                    vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output10.s1, output11.s1, output12.s1, output13.s1)), \
                        0, \
                        output + outputIndex + (outputHeight + 1) * outputWidth); \
            } else if (outputC < outputChannel) { \
                    vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output00.s0, output01.s0, output02.s0, output03.s0)), \
                        0, \
                        output + outputIndex); \
                    vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output10.s0, output11.s0, output12.s0, output13.s0)), \
                        0, \
                        output + outputIndex + outputWidth); \
            } \
        } \
    }

#define DIRECT1X1_8X4_FP32(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, \
                           outputHeight, outputWidth, inputNum, inputChannel, inputHeight, inputWidth) \
    int outputN = get_global_id(1) / ((outputChannel + 7) / 8); \
    int outputC = get_global_id(1) % ((outputChannel + 7) / 8) * 8; \
    int outputHW = get_global_id(0) * 4; \
    if (outputHW < outputWidth * outputHeight) { \
        DATA_T8 output0 = 0.0f; \
        DATA_T8 output1 = 0.0f; \
        DATA_T8 output2 = 0.0f; \
        DATA_T8 output3 = 0.0f; \
        int inputBase = outputN * inputChannel * inputHeight * inputWidth + outputHW; \
        int weightBase = outputC * inputChannel; \
        DATA_T4 intputVector; \
        DATA_T8 weightVector; \
        for (int inputC = 0; inputC < inputChannel; \
                inputC++, inputBase += inputHeight * inputWidth) { \
            intputVector = vload4(0, input + inputBase); \
            weightVector = vload8(inputC, weight + weightBase); \
            output0 = mad(intputVector.s0, weightVector, output0); \
            output1 = mad(intputVector.s1, weightVector, output1); \
            output2 = mad(intputVector.s2, weightVector, output2); \
            output3 = mad(intputVector.s3, weightVector, output3); \
        } \
        int outputIndex = outputN * outputChannel * outputHeight * outputWidth + \
                            outputC * outputHeight * outputWidth + outputHW; \
        DATA_T8 biasVector = vload8(0, bias + outputC); \
        output0 += biasVector; \
        output1 += biasVector; \
        output2 += biasVector; \
        output3 += biasVector; \
        if (outputHW + 3 < outputHeight * outputWidth) { \
            if (outputC + 7 < outputChannel) { \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s0, output1.s0, output2.s0, output3.s0)), \
                        0, \
                        output + outputIndex); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s1, output1.s1, output2.s1, output3.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s2, output1.s2, output2.s2, output3.s2)), \
                        0, \
                        output + outputIndex + 2 * outputHeight * outputWidth); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s3, output1.s3, output2.s3, output3.s3)), \
                        0, \
                        output + outputIndex + 3 * outputHeight * outputWidth); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s4, output1.s4, output2.s4, output3.s4)), \
                        0, \
                        output + outputIndex + 4 * outputHeight * outputWidth); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s5, output1.s5, output2.s5, output3.s5)), \
                        0, \
                        output + outputIndex + 5 * outputHeight * outputWidth); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s6, output1.s6, output2.s6, output3.s6)), \
                        0, \
                        output + outputIndex + 6 * outputHeight * outputWidth); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s7, output1.s7, output2.s7, output3.s7)), \
                        0, \
                        output + outputIndex + 7 * outputHeight * outputWidth); \
            } else if (outputC + 6 < outputChannel) { \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s0, output1.s0, output2.s0, output3.s0)), \
                        0, \
                        output + outputIndex); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s1, output1.s1, output2.s1, output3.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s2, output1.s2, output2.s2, output3.s2)), \
                        0, \
                        output + outputIndex + 2 * outputHeight * outputWidth); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s3, output1.s3, output2.s3, output3.s3)), \
                        0, \
                        output + outputIndex + 3 * outputHeight * outputWidth); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s4, output1.s4, output2.s4, output3.s4)), \
                        0, \
                        output + outputIndex + 4 * outputHeight * outputWidth); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s5, output1.s5, output2.s5, output3.s5)), \
                        0, \
                        output + outputIndex + 5 * outputHeight * outputWidth); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s6, output1.s6, output2.s6, output3.s6)), \
                        0, \
                        output + outputIndex + 6 * outputHeight * outputWidth); \
            } else if (outputC + 5 < outputChannel) { \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s0, output1.s0, output2.s0, output3.s0)), \
                        0, \
                        output + outputIndex); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s1, output1.s1, output2.s1, output3.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s2, output1.s2, output2.s2, output3.s2)), \
                        0, \
                        output + outputIndex + 2 * outputHeight * outputWidth); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s3, output1.s3, output2.s3, output3.s3)), \
                        0, \
                        output + outputIndex + 3 * outputHeight * outputWidth); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s4, output1.s4, output2.s4, output3.s4)), \
                        0, \
                        output + outputIndex + 4 * outputHeight * outputWidth); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s5, output1.s5, output2.s5, output3.s5)), \
                        0, \
                        output + outputIndex + 5 * outputHeight * outputWidth); \
            } else if (outputC + 4 < outputChannel) { \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s0, output1.s0, output2.s0, output3.s0)), \
                        0, \
                        output + outputIndex); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s1, output1.s1, output2.s1, output3.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s2, output1.s2, output2.s2, output3.s2)), \
                        0, \
                        output + outputIndex + 2 * outputHeight * outputWidth); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s3, output1.s3, output2.s3, output3.s3)), \
                        0, \
                        output + outputIndex + 3 * outputHeight * outputWidth); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s4, output1.s4, output2.s4, output3.s4)), \
                        0, \
                        output + outputIndex + 4 * outputHeight * outputWidth); \
            } else if (outputC + 3 < outputChannel) { \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s0, output1.s0, output2.s0, output3.s0)), \
                        0, \
                        output + outputIndex); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s1, output1.s1, output2.s1, output3.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s2, output1.s2, output2.s2, output3.s2)), \
                        0, \
                        output + outputIndex + 2 * outputHeight * outputWidth); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s3, output1.s3, output2.s3, output3.s3)), \
                        0, \
                        output + outputIndex + 3 * outputHeight * outputWidth); \
            } else if (outputC + 2 < outputChannel) { \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s0, output1.s0, output2.s0, output3.s0)), \
                        0, \
                        output + outputIndex); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s1, output1.s1, output2.s1, output3.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s2, output1.s2, output2.s2, output3.s2)), \
                        0, \
                        output + outputIndex + 2 * outputHeight * outputWidth); \
            } else if (outputC + 1 < outputChannel) { \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s0, output1.s0, output2.s0, output3.s0)), \
                        0, \
                        output + outputIndex); \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s1, output1.s1, output2.s1, output3.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
            } else if (outputC < outputChannel) { \
                vstore4(ACT_VEC_F(DATA_T4, (DATA_T4)(output0.s0, output1.s0, output2.s0, output3.s0)), \
                        0, \
                        output + outputIndex); \
            } \
        } else if (outputHW + 2 < outputHeight * outputWidth) { \
            if (outputC + 7 < outputChannel) { \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s0, output1.s0, output2.s0)), \
                        0, \
                        output + outputIndex); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s1, output1.s1, output2.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s2, output1.s2, output2.s2)), \
                        0, \
                        output + outputIndex + 2 * outputHeight * outputWidth); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s3, output1.s3, output2.s3)), \
                        0, \
                        output + outputIndex + 3 * outputHeight * outputWidth); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s4, output1.s4, output2.s4)), \
                        0, \
                        output + outputIndex + 4 * outputHeight * outputWidth); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s5, output1.s5, output2.s5)), \
                        0, \
                        output + outputIndex + 5 * outputHeight * outputWidth); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s6, output1.s6, output2.s6)), \
                        0, \
                        output + outputIndex + 6 * outputHeight * outputWidth); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s7, output1.s7, output2.s7)), \
                        0, \
                        output + outputIndex + 7 * outputHeight * outputWidth); \
            } else if (outputC + 6 < outputChannel) { \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s0, output1.s0, output2.s0)), \
                        0, \
                        output + outputIndex); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s1, output1.s1, output2.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s2, output1.s2, output2.s2)), \
                        0, \
                        output + outputIndex + 2 * outputHeight * outputWidth); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s3, output1.s3, output2.s3)), \
                        0, \
                        output + outputIndex + 3 * outputHeight * outputWidth); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s4, output1.s4, output2.s4)), \
                        0, \
                        output + outputIndex + 4 * outputHeight * outputWidth); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s5, output1.s5, output2.s5)), \
                        0, \
                        output + outputIndex + 5 * outputHeight * outputWidth); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s6, output1.s6, output2.s6)), \
                        0, \
                        output + outputIndex + 6 * outputHeight * outputWidth); \
            } else if (outputC + 5 < outputChannel) { \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s0, output1.s0, output2.s0)), \
                        0, \
                        output + outputIndex); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s1, output1.s1, output2.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s2, output1.s2, output2.s2)), \
                        0, \
                        output + outputIndex + 2 * outputHeight * outputWidth); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s3, output1.s3, output2.s3)), \
                        0, \
                        output + outputIndex + 3 * outputHeight * outputWidth); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s4, output1.s4, output2.s4)), \
                        0, \
                        output + outputIndex + 4 * outputHeight * outputWidth); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s5, output1.s5, output2.s5)), \
                        0, \
                        output + outputIndex + 5 * outputHeight * outputWidth); \
            } else if (outputC + 4 < outputChannel) { \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s0, output1.s0, output2.s0)), \
                        0, \
                        output + outputIndex); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s1, output1.s1, output2.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s2, output1.s2, output2.s2)), \
                        0, \
                        output + outputIndex + 2 * outputHeight * outputWidth); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s3, output1.s3, output2.s3)), \
                        0, \
                        output + outputIndex + 3 * outputHeight * outputWidth); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s4, output1.s4, output2.s4)), \
                        0, \
                        output + outputIndex + 4 * outputHeight * outputWidth); \
            } else if (outputC + 3 < outputChannel) { \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s0, output1.s0, output2.s0)), \
                        0, \
                        output + outputIndex); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s1, output1.s1, output2.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s2, output1.s2, output2.s2)), \
                        0, \
                        output + outputIndex + 2 * outputHeight * outputWidth); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s3, output1.s3, output2.s3)), \
                        0, \
                        output + outputIndex + 3 * outputHeight * outputWidth); \
            } else if (outputC + 2 < outputChannel) { \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s0, output1.s0, output2.s0)), \
                        0, \
                        output + outputIndex); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s1, output1.s1, output2.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s2, output1.s2, output2.s2)), \
                        0, \
                        output + outputIndex + 2 * outputHeight * outputWidth); \
            } else if (outputC + 1 < outputChannel) { \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s0, output1.s0, output2.s0)), \
                        0, \
                        output + outputIndex); \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s1, output1.s1, output2.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
            } else if (outputC < outputChannel) { \
                vstore3(ACT_VEC_F(DATA_T3, (DATA_T3)(output0.s0, output1.s0, output2.s0)), \
                        0, \
                        output + outputIndex); \
            } \
        } else if (outputHW + 1 < outputHeight * outputWidth) { \
            if (outputC + 7 < outputChannel) { \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s0, output1.s0)), \
                        0, \
                        output + outputIndex); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s1, output1.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s2, output1.s2)), \
                        0, \
                        output + outputIndex + 2 * outputHeight * outputWidth); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s3, output1.s3)), \
                        0, \
                        output + outputIndex + 3 * outputHeight * outputWidth); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s4, output1.s4)), \
                        0, \
                        output + outputIndex + 4 * outputHeight * outputWidth); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s5, output1.s5)), \
                        0, \
                        output + outputIndex + 5 * outputHeight * outputWidth); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s6, output1.s6)), \
                        0, \
                        output + outputIndex + 6 * outputHeight * outputWidth); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s7, output1.s7)), \
                        0, \
                        output + outputIndex + 7 * outputHeight * outputWidth); \
            } else if (outputC + 6 < outputChannel) { \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s0, output1.s0)), \
                        0, \
                        output + outputIndex); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s1, output1.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s2, output1.s2)), \
                        0, \
                        output + outputIndex + 2 * outputHeight * outputWidth); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s3, output1.s3)), \
                        0, \
                        output + outputIndex + 3 * outputHeight * outputWidth); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s4, output1.s4)), \
                        0, \
                        output + outputIndex + 4 * outputHeight * outputWidth); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s5, output1.s5)), \
                        0, \
                        output + outputIndex + 5 * outputHeight * outputWidth); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s6, output1.s6)), \
                        0, \
                        output + outputIndex + 6 * outputHeight * outputWidth); \
            } else if (outputC + 5 < outputChannel) { \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s0, output1.s0)), \
                        0, \
                        output + outputIndex); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s1, output1.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s2, output1.s2)), \
                        0, \
                        output + outputIndex + 2 * outputHeight * outputWidth); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s3, output1.s3)), \
                        0, \
                        output + outputIndex + 3 * outputHeight * outputWidth); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s4, output1.s4)), \
                        0, \
                        output + outputIndex + 4 * outputHeight * outputWidth); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s5, output1.s5)), \
                        0, \
                        output + outputIndex + 5 * outputHeight * outputWidth); \
            } else if (outputC + 4 < outputChannel) { \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s0, output1.s0)), \
                        0, \
                        output + outputIndex); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s1, output1.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s2, output1.s2)), \
                        0, \
                        output + outputIndex + 2 * outputHeight * outputWidth); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s3, output1.s3)), \
                        0, \
                        output + outputIndex + 3 * outputHeight * outputWidth); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s4, output1.s4)), \
                        0, \
                        output + outputIndex + 4 * outputHeight * outputWidth); \
            } else if (outputC + 3 < outputChannel) { \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s0, output1.s0)), \
                        0, \
                        output + outputIndex); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s1, output1.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s2, output1.s2)), \
                        0, \
                        output + outputIndex + 2 * outputHeight * outputWidth); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s3, output1.s3)), \
                        0, \
                        output + outputIndex + 3 * outputHeight * outputWidth); \
            } else if (outputC + 2 < outputChannel) { \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s0, output1.s0)), \
                        0, \
                        output + outputIndex); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s1, output1.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s2, output1.s2)), \
                        0, \
                        output + outputIndex + 2 * outputHeight * outputWidth); \
            } else if (outputC + 1 < outputChannel) { \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s0, output1.s0)), \
                        0, \
                        output + outputIndex); \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s1, output1.s1)), \
                        0, \
                        output + outputIndex + outputHeight * outputWidth); \
            } else if (outputC < outputChannel) { \
                vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(output0.s0, output1.s0)), \
                        0, \
                        output + outputIndex); \
            } \
        } else if (outputHW < outputHeight * outputWidth) { \
            if (outputC + 7 < outputChannel) { \
                output[outputIndex] = ACT_VEC_F(DATA_T, output0.s0); \
                output[outputIndex + outputHeight * outputWidth] = \
                    ACT_VEC_F(DATA_T, output0.s1); \
                output[outputIndex + 2 * outputHeight * outputWidth] = \
                    ACT_VEC_F(DATA_T, output0.s2); \
                output[outputIndex + 3 * outputHeight * outputWidth] = \
                    ACT_VEC_F(DATA_T, output0.s3); \
                output[outputIndex + 4 * outputHeight * outputWidth] = \
                    ACT_VEC_F(DATA_T, output0.s4); \
                output[outputIndex + 5 * outputHeight * outputWidth] = \
                    ACT_VEC_F(DATA_T, output0.s5); \
                output[outputIndex + 6 * outputHeight * outputWidth] = \
                    ACT_VEC_F(DATA_T, output0.s6); \
                output[outputIndex + 7 * outputHeight * outputWidth] = \
                    ACT_VEC_F(DATA_T, output0.s7); \
            } else if (outputC + 6 < outputChannel) { \
                output[outputIndex] = ACT_VEC_F(DATA_T, output0.s0); \
                output[outputIndex + outputHeight * outputWidth] = \
                    ACT_VEC_F(DATA_T, output0.s1); \
                output[outputIndex + 2 * outputHeight * outputWidth] = \
                    ACT_VEC_F(DATA_T, output0.s2); \
                output[outputIndex + 3 * outputHeight * outputWidth] = \
                    ACT_VEC_F(DATA_T, output0.s3); \
                output[outputIndex + 4 * outputHeight * outputWidth] = \
                    ACT_VEC_F(DATA_T, output0.s4); \
                output[outputIndex + 5 * outputHeight * outputWidth] = \
                    ACT_VEC_F(DATA_T, output0.s5); \
                output[outputIndex + 6 * outputHeight * outputWidth] = \
                    ACT_VEC_F(DATA_T, output0.s6); \
            } else if (outputC + 5 < outputChannel) { \
                output[outputIndex] = ACT_VEC_F(DATA_T, output0.s0); \
                output[outputIndex + outputHeight * outputWidth] = \
                    ACT_VEC_F(DATA_T, output0.s1); \
                output[outputIndex + 2 * outputHeight * outputWidth] = \
                    ACT_VEC_F(DATA_T, output0.s2); \
                output[outputIndex + 3 * outputHeight * outputWidth] = \
                    ACT_VEC_F(DATA_T, output0.s3); \
                output[outputIndex + 4 * outputHeight * outputWidth] = \
                    ACT_VEC_F(DATA_T, output0.s4); \
                output[outputIndex + 5 * outputHeight * outputWidth] = \
                    ACT_VEC_F(DATA_T, output0.s5); \
            } else if (outputC + 4 < outputChannel) { \
                output[outputIndex] = ACT_VEC_F(DATA_T, output0.s0); \
                output[outputIndex + outputHeight * outputWidth] = \
                    ACT_VEC_F(DATA_T, output0.s1); \
                output[outputIndex + 2 * outputHeight * outputWidth] = \
                    ACT_VEC_F(DATA_T, output0.s2); \
                output[outputIndex + 3 * outputHeight * outputWidth] = \
                    ACT_VEC_F(DATA_T, output0.s3); \
                output[outputIndex + 4 * outputHeight * outputWidth] = \
                    ACT_VEC_F(DATA_T, output0.s4); \
            } else if (outputC + 3 < outputChannel) { \
                output[outputIndex] = ACT_VEC_F(DATA_T, output0.s0); \
                output[outputIndex + outputHeight * outputWidth] = \
                    ACT_VEC_F(DATA_T, output0.s1); \
                output[outputIndex + 2 * outputHeight * outputWidth] = \
                    ACT_VEC_F(DATA_T, output0.s2); \
                output[outputIndex + 3 * outputHeight * outputWidth] = \
                    ACT_VEC_F(DATA_T, output0.s3); \
            } else if (outputC + 2 < outputChannel) { \
                output[outputIndex] = ACT_VEC_F(DATA_T, output0.s0); \
                output[outputIndex + outputHeight * outputWidth] = \
                    ACT_VEC_F(DATA_T, output0.s1); \
                output[outputIndex + 2 * outputHeight * outputWidth] = \
                    ACT_VEC_F(DATA_T, output0.s2); \
            } else if (outputC + 1 < outputChannel) { \
                output[outputIndex] = ACT_VEC_F(DATA_T, output0.s0); \
                output[outputIndex + outputHeight * outputWidth] = \
                    ACT_VEC_F(DATA_T, output0.s1); \
            } else if (outputC < outputChannel) { \
                output[outputIndex] = ACT_VEC_F(DATA_T, output0.s0); \
            } \
        } \
    }
#define DIRECT7X7_4X8_FP16(input, weight, bias, aligned_output, padT, padR, padB, padL, outputNumber, \
        outputChannel, outputHeight, outputWidth, inputNum, inputChannel, inputHeight, inputWidth, splite_num) \
        int outputN = get_global_id(2) / ((outputChannel + 3) / 4); \
        int outputC = get_global_id(2) % ((outputChannel + 3) / 4) * 4; \
        int outputH = get_global_id(1) / splite_num; \
        int outputW = get_global_id(0) * 8; \
        int parall_id = get_local_id(1) % splite_num; \
        if (outputW < outputWidth && outputH < outputHeight) { \
            DATA_T4 output0 = 0.0f; \
            DATA_T4 output1 = 0.0f; \
            DATA_T4 output2 = 0.0f; \
            DATA_T4 output3 = 0.0f; \
            DATA_T4 output4 = 0.0f; \
            DATA_T4 output5 = 0.0f; \
            DATA_T4 output6 = 0.0f; \
            DATA_T4 output7 = 0.0f; \
            int inputHStart = max(outputH - padT, 0); \
            int inputHEnd = min(outputH - padT + 7, inputHeight); \
            int inputW = outputW - padL; \
            DATA_T16 inputMask = 1.0f; \
            if (inputW < 0 || inputW >= inputWidth) { \
                inputMask.s0 = 0.0f; \
            } \
            if (inputW + 1 < 0 || inputW + 1 >= inputWidth) { \
                inputMask.s1 = 0.0f; \
            } \
            if (inputW + 2 < 0 || inputW + 2 >= inputWidth) { \
                inputMask.s2 = 0.0f; \
            } \
            if (inputW + 3 < 0 || inputW + 3 >= inputWidth) { \
                inputMask.s3 = 0.0f; \
            } \
            if (inputW + 4 < 0 || inputW + 4 >= inputWidth) { \
                inputMask.s4 = 0.0f; \
            } \
            if (inputW + 5 < 0 || inputW + 5 >= inputWidth) { \
                inputMask.s5 = 0.0f; \
            } \
            if (inputW + 6 < 0 || inputW + 6 >= inputWidth) { \
                inputMask.s6 = 0.0f; \
            } \
            if (inputW + 7 < 0 || inputW + 7 >= inputWidth) { \
                inputMask.s7 = 0.0f; \
            } \
            if (inputW + 8 < 0 || inputW + 8 >= inputWidth) { \
                inputMask.s8 = 0.0f; \
            } \
            if (inputW + 9 < 0 || inputW + 9 >= inputWidth) { \
                inputMask.s9 = 0.0f; \
            } \
            if (inputW + 10 < 0 || inputW + 10 >= inputWidth) { \
                inputMask.sa = 0.0f; \
            } \
            if (inputW + 11 < 0 || inputW + 11 >= inputWidth) { \
                inputMask.sb = 0.0f; \
            } \
            if (inputW + 12 < 0 || inputW + 12 >= inputWidth) { \
                inputMask.sc = 0.0f; \
            } \
            if (inputW + 13 < 0 || inputW + 13 >= inputWidth) { \
                inputMask.sd = 0.0f; \
            } \
            if (inputW + 14 < 0 || inputW + 14 >= inputWidth) { \
                inputMask.se = 0.0f; \
            } \
            if (inputW + 15 < 0 || inputW + 15 >= inputWidth) { \
                inputMask.sf = 0.0f; \
            } \
            int inputBase = outputN * inputChannel * inputHeight * inputWidth + inputW; \
            int khOffset = (outputH - padT < 0) ? 7 - (inputHEnd - inputHStart) : 0; \
            int weightBase = outputC * inputChannel * 7 * 7 + khOffset * 4 * 7; \
            DATA_T16 intputVector; \
            DATA_T16 weightVector; \
            for (int inputC = 0; inputC < inputChannel; inputC += splite_num) { \
                int inputIndex = inputBase + (inputC + parall_id) * inputHeight * inputWidth; \
                int weightIndex = weightBase + (inputC + parall_id) * 4 * 7 * 7; \
                for (int kh = inputHStart; kh < inputHEnd; kh++) { \
                    intputVector = vload16(0, input + inputIndex + kh * inputWidth); \
                    intputVector = intputVector * inputMask; \
                    weightVector = vload16(0, weight + weightIndex); \
                    weightIndex += 16; \
                    output0 = mad(intputVector.s0, weightVector.lo.lo, output0); \
                    output1 = mad(intputVector.s1, weightVector.lo.lo, output1); \
                    output2 = mad(intputVector.s2, weightVector.lo.lo, output2); \
                    output3 = mad(intputVector.s3, weightVector.lo.lo, output3); \
                    output4 = mad(intputVector.s4, weightVector.lo.lo, output4); \
                    output5 = mad(intputVector.s5, weightVector.lo.lo, output5); \
                    output6 = mad(intputVector.s6, weightVector.lo.lo, output6); \
                    output7 = mad(intputVector.s7, weightVector.lo.lo, output7); \
                    output0 = mad(intputVector.s1, weightVector.lo.hi, output0); \
                    output1 = mad(intputVector.s2, weightVector.lo.hi, output1); \
                    output2 = mad(intputVector.s3, weightVector.lo.hi, output2); \
                    output3 = mad(intputVector.s4, weightVector.lo.hi, output3); \
                    output4 = mad(intputVector.s5, weightVector.lo.hi, output4); \
                    output5 = mad(intputVector.s6, weightVector.lo.hi, output5); \
                    output6 = mad(intputVector.s7, weightVector.lo.hi, output6); \
                    output7 = mad(intputVector.s8, weightVector.lo.hi, output7); \
                    output0 = mad(intputVector.s2, weightVector.hi.lo, output0); \
                    output1 = mad(intputVector.s3, weightVector.hi.lo, output1); \
                    output2 = mad(intputVector.s4, weightVector.hi.lo, output2); \
                    output3 = mad(intputVector.s5, weightVector.hi.lo, output3); \
                    output4 = mad(intputVector.s6, weightVector.hi.lo, output4); \
                    output5 = mad(intputVector.s7, weightVector.hi.lo, output5); \
                    output6 = mad(intputVector.s8, weightVector.hi.lo, output6); \
                    output7 = mad(intputVector.s9, weightVector.hi.lo, output7); \
                    output0 = mad(intputVector.s3, weightVector.hi.hi, output0); \
                    output1 = mad(intputVector.s4, weightVector.hi.hi, output1); \
                    output2 = mad(intputVector.s5, weightVector.hi.hi, output2); \
                    output3 = mad(intputVector.s6, weightVector.hi.hi, output3); \
                    output4 = mad(intputVector.s7, weightVector.hi.hi, output4); \
                    output5 = mad(intputVector.s8, weightVector.hi.hi, output5); \
                    output6 = mad(intputVector.s9, weightVector.hi.hi, output6); \
                    output7 = mad(intputVector.sa, weightVector.hi.hi, output7); \
                    weightVector.lo = vload8(0, weight + weightIndex); \
                    weightIndex += 8; \
                    output0 = mad(intputVector.s4, weightVector.lo.lo, output0); \
                    output1 = mad(intputVector.s5, weightVector.lo.lo, output1); \
                    output2 = mad(intputVector.s6, weightVector.lo.lo, output2); \
                    output3 = mad(intputVector.s7, weightVector.lo.lo, output3); \
                    output4 = mad(intputVector.s8, weightVector.lo.lo, output4); \
                    output5 = mad(intputVector.s9, weightVector.lo.lo, output5); \
                    output6 = mad(intputVector.sa, weightVector.lo.lo, output6); \
                    output7 = mad(intputVector.sb, weightVector.lo.lo, output7); \
                    output0 = mad(intputVector.s5, weightVector.lo.hi, output0); \
                    output1 = mad(intputVector.s6, weightVector.lo.hi, output1); \
                    output2 = mad(intputVector.s7, weightVector.lo.hi, output2); \
                    output3 = mad(intputVector.s8, weightVector.lo.hi, output3); \
                    output4 = mad(intputVector.s9, weightVector.lo.hi, output4); \
                    output5 = mad(intputVector.sa, weightVector.lo.hi, output5); \
                    output6 = mad(intputVector.sb, weightVector.lo.hi, output6); \
                    output7 = mad(intputVector.sc, weightVector.lo.hi, output7); \
                    weightVector.lo.lo = vload4(0, weight + weightIndex); \
                    weightIndex += 4; \
                    output0 = mad(intputVector.s6, weightVector.lo.lo, output0); \
                    output1 = mad(intputVector.s7, weightVector.lo.lo, output1); \
                    output2 = mad(intputVector.s8, weightVector.lo.lo, output2); \
                    output3 = mad(intputVector.s9, weightVector.lo.lo, output3); \
                    output4 = mad(intputVector.sa, weightVector.lo.lo, output4); \
                    output5 = mad(intputVector.sb, weightVector.lo.lo, output5); \
                    output6 = mad(intputVector.sc, weightVector.lo.lo, output6); \
                    output7 = mad(intputVector.sd, weightVector.lo.lo, output7); \
                } \
            } \
            int outputIndex = outputN * outputChannel * outputHeight * outputWidth / 32 + \
                              outputC / 4 * outputHeight * outputWidth / 8 + outputH * outputWidth / 8 + outputW / 8; \
            int tmp_offset = parall_id * 32; \
            int aligned_output_base = outputIndex * 32 * splite_num; \
            aligned_output_base += tmp_offset; \
            { \
                vstore16((DATA_T16)(output0, output1, output2, output3), 0, aligned_output + aligned_output_base); \
                vstore16((DATA_T16)(output4, output5, output6, output7), 1, aligned_output + aligned_output_base); \
            } \
        }


#define DIRECT9X9_4X8_FP16(input, weight, bias, aligned_output, padT, padR, padB, padL, outputNumber, \
        outputChannel, outputHeight, outputWidth, inputNum, inputChannel, inputHeight, inputWidth, splite_num) \
        int outputN = get_global_id(2) / ((outputChannel + 3) / 4); \
        int outputC = get_global_id(2) % ((outputChannel + 3) / 4) * 4; \
        int outputH = get_global_id(1) / splite_num; \
        int outputW = get_global_id(0) * 8; \
        int parall_id = get_global_id(1) % splite_num; \
        if (outputW < outputWidth && outputH < outputHeight) { \
            DATA_T4 output0 = 0.0f; \
            DATA_T4 output1 = 0.0f; \
            DATA_T4 output2 = 0.0f; \
            DATA_T4 output3 = 0.0f; \
            DATA_T4 output4 = 0.0f; \
            DATA_T4 output5 = 0.0f; \
            DATA_T4 output6 = 0.0f; \
            DATA_T4 output7 = 0.0f; \
            int inputHStart = max(outputH - padT, 0); \
            int inputHEnd = min(outputH - padT + 9, inputHeight); \
            int inputW = outputW - padL; \
            DATA_T16 inputMask = 1.0f; \
            if (inputW < 0 || inputW >= inputWidth) { \
                inputMask.s0 = 0.0f; \
            } \
            if (inputW + 1 < 0 || inputW + 1 >= inputWidth) { \
                inputMask.s1 = 0.0f; \
            } \
            if (inputW + 2 < 0 || inputW + 2 >= inputWidth) { \
                inputMask.s2 = 0.0f; \
            } \
            if (inputW + 3 < 0 || inputW + 3 >= inputWidth) { \
                inputMask.s3 = 0.0f; \
            } \
            if (inputW + 4 < 0 || inputW + 4 >= inputWidth) { \
                inputMask.s4 = 0.0f; \
            } \
            if (inputW + 5 < 0 || inputW + 5 >= inputWidth) { \
                inputMask.s5 = 0.0f; \
            } \
            if (inputW + 6 < 0 || inputW + 6 >= inputWidth) { \
                inputMask.s6 = 0.0f; \
            } \
            if (inputW + 7 < 0 || inputW + 7 >= inputWidth) { \
                inputMask.s7 = 0.0f; \
            } \
            if (inputW + 8 < 0 || inputW + 8 >= inputWidth) { \
                inputMask.s8 = 0.0f; \
            } \
            if (inputW + 9 < 0 || inputW + 9 >= inputWidth) { \
                inputMask.s9 = 0.0f; \
            } \
            if (inputW + 10 < 0 || inputW + 10 >= inputWidth) { \
                inputMask.sa = 0.0f; \
            } \
            if (inputW + 11 < 0 || inputW + 11 >= inputWidth) { \
                inputMask.sb = 0.0f; \
            } \
            if (inputW + 12 < 0 || inputW + 12 >= inputWidth) { \
                inputMask.sc = 0.0f; \
            } \
            if (inputW + 13 < 0 || inputW + 13 >= inputWidth) { \
                inputMask.sd = 0.0f; \
            } \
            if (inputW + 14 < 0 || inputW + 14 >= inputWidth) { \
                inputMask.se = 0.0f; \
            } \
            if (inputW + 15 < 0 || inputW + 15 >= inputWidth) { \
                inputMask.sf = 0.0f; \
            } \
            int inputBase = outputN * inputChannel * inputHeight * inputWidth + inputW; \
            int khOffset = (outputH - padT < 0) ? 9 - (inputHEnd - inputHStart) : 0; \
            int weightBase = outputC * inputChannel * 9 * 9 + khOffset * 4 * 9; \
            DATA_T16 intputVector; \
            DATA_T16 weightVector; \
            for (int inputC = 0; inputC < inputChannel; inputC += splite_num) { \
                int inputIndex = inputBase + (inputC + parall_id)* inputHeight * inputWidth; \
                int weightIndex = weightBase + (inputC + parall_id) * 4 * 9 * 9; \
                for (int kh = inputHStart; kh < inputHEnd; kh++) { \
                    intputVector = vload16(0, input + inputIndex + kh * inputWidth); \
                    intputVector = intputVector * inputMask; \
                    weightVector = vload16(0, weight + weightIndex); \
                    weightIndex += 16; \
                    output0 = mad(intputVector.s0, weightVector.lo.lo, output0); \
                    output1 = mad(intputVector.s1, weightVector.lo.lo, output1); \
                    output2 = mad(intputVector.s2, weightVector.lo.lo, output2); \
                    output3 = mad(intputVector.s3, weightVector.lo.lo, output3); \
                    output4 = mad(intputVector.s4, weightVector.lo.lo, output4); \
                    output5 = mad(intputVector.s5, weightVector.lo.lo, output5); \
                    output6 = mad(intputVector.s6, weightVector.lo.lo, output6); \
                    output7 = mad(intputVector.s7, weightVector.lo.lo, output7); \
                    output0 = mad(intputVector.s1, weightVector.lo.hi, output0); \
                    output1 = mad(intputVector.s2, weightVector.lo.hi, output1); \
                    output2 = mad(intputVector.s3, weightVector.lo.hi, output2); \
                    output3 = mad(intputVector.s4, weightVector.lo.hi, output3); \
                    output4 = mad(intputVector.s5, weightVector.lo.hi, output4); \
                    output5 = mad(intputVector.s6, weightVector.lo.hi, output5); \
                    output6 = mad(intputVector.s7, weightVector.lo.hi, output6); \
                    output7 = mad(intputVector.s8, weightVector.lo.hi, output7); \
                    output0 = mad(intputVector.s2, weightVector.hi.lo, output0); \
                    output1 = mad(intputVector.s3, weightVector.hi.lo, output1); \
                    output2 = mad(intputVector.s4, weightVector.hi.lo, output2); \
                    output3 = mad(intputVector.s5, weightVector.hi.lo, output3); \
                    output4 = mad(intputVector.s6, weightVector.hi.lo, output4); \
                    output5 = mad(intputVector.s7, weightVector.hi.lo, output5); \
                    output6 = mad(intputVector.s8, weightVector.hi.lo, output6); \
                    output7 = mad(intputVector.s9, weightVector.hi.lo, output7); \
                    output0 = mad(intputVector.s3, weightVector.hi.hi, output0); \
                    output1 = mad(intputVector.s4, weightVector.hi.hi, output1); \
                    output2 = mad(intputVector.s5, weightVector.hi.hi, output2); \
                    output3 = mad(intputVector.s6, weightVector.hi.hi, output3); \
                    output4 = mad(intputVector.s7, weightVector.hi.hi, output4); \
                    output5 = mad(intputVector.s8, weightVector.hi.hi, output5); \
                    output6 = mad(intputVector.s9, weightVector.hi.hi, output6); \
                    output7 = mad(intputVector.sa, weightVector.hi.hi, output7); \
                    weightVector = vload16(0, weight + weightIndex); \
                    weightIndex += 16; \
                    output0 = mad(intputVector.s4, weightVector.lo.lo, output0); \
                    output1 = mad(intputVector.s5, weightVector.lo.lo, output1); \
                    output2 = mad(intputVector.s6, weightVector.lo.lo, output2); \
                    output3 = mad(intputVector.s7, weightVector.lo.lo, output3); \
                    output4 = mad(intputVector.s8, weightVector.lo.lo, output4); \
                    output5 = mad(intputVector.s9, weightVector.lo.lo, output5); \
                    output6 = mad(intputVector.sa, weightVector.lo.lo, output6); \
                    output7 = mad(intputVector.sb, weightVector.lo.lo, output7); \
                    output0 = mad(intputVector.s5, weightVector.lo.hi, output0); \
                    output1 = mad(intputVector.s6, weightVector.lo.hi, output1); \
                    output2 = mad(intputVector.s7, weightVector.lo.hi, output2); \
                    output3 = mad(intputVector.s8, weightVector.lo.hi, output3); \
                    output4 = mad(intputVector.s9, weightVector.lo.hi, output4); \
                    output5 = mad(intputVector.sa, weightVector.lo.hi, output5); \
                    output6 = mad(intputVector.sb, weightVector.lo.hi, output6); \
                    output7 = mad(intputVector.sc, weightVector.lo.hi, output7); \
                    output0 = mad(intputVector.s6, weightVector.hi.lo, output0); \
                    output1 = mad(intputVector.s7, weightVector.hi.lo, output1); \
                    output2 = mad(intputVector.s8, weightVector.hi.lo, output2); \
                    output3 = mad(intputVector.s9, weightVector.hi.lo, output3); \
                    output4 = mad(intputVector.sa, weightVector.hi.lo, output4); \
                    output5 = mad(intputVector.sb, weightVector.hi.lo, output5); \
                    output6 = mad(intputVector.sc, weightVector.hi.lo, output6); \
                    output7 = mad(intputVector.sd, weightVector.hi.lo, output7); \
                    output0 = mad(intputVector.s7, weightVector.hi.hi, output0); \
                    output1 = mad(intputVector.s8, weightVector.hi.hi, output1); \
                    output2 = mad(intputVector.s9, weightVector.hi.hi, output2); \
                    output3 = mad(intputVector.sa, weightVector.hi.hi, output3); \
                    output4 = mad(intputVector.sb, weightVector.hi.hi, output4); \
                    output5 = mad(intputVector.sc, weightVector.hi.hi, output5); \
                    output6 = mad(intputVector.sd, weightVector.hi.hi, output6); \
                    output7 = mad(intputVector.se, weightVector.hi.hi, output7); \
                    weightVector.lo.lo = vload4(0, weight + weightIndex); \
                    weightIndex += 4; \
                    output0 = mad(intputVector.s8, weightVector.lo.lo, output0); \
                    output1 = mad(intputVector.s9, weightVector.lo.lo, output1); \
                    output2 = mad(intputVector.sa, weightVector.lo.lo, output2); \
                    output3 = mad(intputVector.sb, weightVector.lo.lo, output3); \
                    output4 = mad(intputVector.sc, weightVector.lo.lo, output4); \
                    output5 = mad(intputVector.sd, weightVector.lo.lo, output5); \
                    output6 = mad(intputVector.se, weightVector.lo.lo, output6); \
                    output7 = mad(intputVector.sf, weightVector.lo.lo, output7); \
                } \
            } \
            int outputIndex = outputN * outputChannel * outputHeight * outputWidth / 32 + \
                              outputC / 4 * outputHeight * outputWidth / 8 + outputH * outputWidth / 8 + \
                              outputW / 8; \
            int tmp_offset = parall_id * 32; \
            int aligned_output_base = outputIndex * 32 * splite_num; \
            aligned_output_base += tmp_offset; \
            { \
                vstore16((DATA_T16)(output0, output1, output2, output3), 0, aligned_output + aligned_output_base); \
                vstore16((DATA_T16)(output4, output5, output6, output7), 1, aligned_output + aligned_output_base); \
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
ADD_SINGLE_KERNEL(direct3x3_4x8_FP16, (__global const DATA_T *input,
                                       __global const DATA_T *weight,
                                       __global const DATA_T *bias,
                                       __global DATA_T *output,
                                       int padT,
                                       int padR,
                                       int padB,
                                       int padL,
                                       int outputNumber,
                                       int outputChannel,
                                       int outputHeight,
                                       int outputWidth,
                                       int inputNum,
                                       int inputChannel,
                                       int inputHeight,
                                       int inputWidth) {
    DIRECT3X3_4X8_FP16(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                       outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(direct3x3_8x8_FP16, (__global const DATA_T *input,
                                       __global const DATA_T *weight,
                                       __global const DATA_T *bias,
                                       __global DATA_T *output,
                                       int padT,
                                       int padR,
                                       int padB,
                                       int padL,
                                       int outputNumber,
                                       int outputChannel,
                                       int outputHeight,
                                       int outputWidth,
                                       int inputNum,
                                       int inputChannel,
                                       int inputHeight,
                                       int inputWidth) {
    DIRECT3X3_8X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(direct5x5_4x8_FP16, (__global const DATA_T *input,
                                       __global const DATA_T *weight,
                                       __global const DATA_T *bias,
                                       __global DATA_T *output,
                                       int padT,
                                       int padR,
                                       int padB,
                                       int padL,
                                       int outputNumber,
                                       int outputChannel,
                                       int outputHeight,
                                       int outputWidth,
                                       int inputNum,
                                       int inputChannel,
                                       int inputHeight,
                                       int inputWidth) {
    DIRECT5X5_4X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(direct5x5_8x8_FP16, (__global const DATA_T *input,
                                       __global const DATA_T *weight,
                                       __global const DATA_T *bias,
                                       __global DATA_T *output,
                                       int padT,
                                       int padR,
                                       int padB,
                                       int padL,
                                       int outputNumber,
                                       int outputChannel,
                                       int outputHeight,
                                       int outputWidth,
                                       int inputNum,
                                       int inputChannel,
                                       int inputHeight,
                                       int inputWidth) {
    DIRECT5X5_8X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})
ADD_SINGLE_KERNEL(direct7x7_4x8_FP16, (__global const half *input,
                                     __global const half *weight,
                                     __global const half *bias,
                                     __global half *aligned_output,
                                     int padT,
                                     int padR,
                                     int padB,
                                     int padL,
                                     int outputNumber,
                                     int outputChannel,
                                     int outputHeight,
                                     int outputWidth,
                                     int inputNum,
                                     int inputChannel,
                                     int inputHeight,
                                     int inputWidth,
                                     int splite_num) {
        DIRECT7X7_4X8_FP16(input, weight, bias, aligned_output, padT, padR, padB, padL, outputNumber, outputChannel, \
                  outputHeight, outputWidth, inputNum, inputChannel, inputHeight,\
                  inputWidth, splite_num)
})

ADD_SINGLE_KERNEL(direct7x7_4x8_merge_FP16, (__global const DATA_T *bias,
                                             __global DATA_T *aligned_output,
                                             __global DATA_T *output,
                                             int padT,
                                             int padR,
                                             int padB,
                                             int padL,
                                             int outputNumber,
                                             int outputChannel,
                                             int outputHeight,
                                             int outputWidth,
                                             int inputNum,
                                             int inputChannel,
                                             int inputHeight,
                                             int inputWidth,
                                             int splite_num) {
    DIRECT7X7_4X8_MERGE(bias, aligned_output, output, padT, padR, padB, padL, outputNumber, outputChannel, \
                        outputHeight, outputWidth, inputNum, inputChannel, inputHeight, inputWidth, splite_num)
})

ADD_SINGLE_KERNEL(direct7x7_8x8_FP16, (__global const DATA_T *input,
                                       __global const DATA_T *weight,
                                       __global const DATA_T *bias,
                                       __global DATA_T *output,
                                       int padT,
                                       int padR,
                                       int padB,
                                       int padL,
                                       int outputNumber,
                                       int outputChannel,
                                       int outputHeight,
                                       int outputWidth,
                                       int inputNum,
                                       int inputChannel,
                                       int inputHeight,
                                       int inputWidth) {
    DIRECT7X7_8X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(direct9x9_4x8_FP16, (__global const half *input,
                                     __global const half *weight,
                                     __global const half *bias,
                                     __global half *aligned_output,
                                     int padT,
                                     int padR,
                                     int padB,
                                     int padL,
                                     int outputNumber,
                                     int outputChannel,
                                     int outputHeight,
                                     int outputWidth,
                                     int inputNum,
                                     int inputChannel,
                                     int inputHeight,
                                     int inputWidth,
                                     int splite_num) {
    DIRECT9X9_4X8_FP16(input, weight, bias, aligned_output, padT, padR, padB, padL, outputNumber, outputChannel, \
     outputHeight, outputWidth, inputNum, inputChannel, inputHeight, inputWidth, splite_num)
})

ADD_SINGLE_KERNEL(direct9x9_4x8_merge_FP16, (__global const DATA_T *bias,
                                             __global DATA_T *aligned_output,
                                             __global DATA_T *output,
                                             int padT,
                                             int padR,
                                             int padB,
                                             int padL,
                                             int outputNumber,
                                             int outputChannel,
                                             int outputHeight,
                                             int outputWidth,
                                             int inputNum,
                                             int inputChannel,
                                             int inputHeight,
                                             int inputWidth,
                                             int splite_num) {
    DIRECT9X9_4X8_MERGE(bias, aligned_output, output, padT, padR, padB, padL, outputNumber, outputChannel, \
                        outputHeight, outputWidth, inputNum, inputChannel, inputHeight, inputWidth, splite_num)
})

ADD_SINGLE_KERNEL(direct9x9_8x8_FP16, (__global const DATA_T *input,
                                       __global const DATA_T *weight,
                                       __global const DATA_T *bias,
                                       __global DATA_T *output,
                                       int padT,
                                       int padR,
                                       int padB,
                                       int padL,
                                       int outputNumber,
                                       int outputChannel,
                                       int outputHeight,
                                       int outputWidth,
                                       int inputNum,
                                       int inputChannel,
                                       int inputHeight,
                                       int inputWidth) {
    DIRECT9X9_8X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(direct3x3_4x4_FP16, (__global const DATA_T *input,
                                       __global const DATA_T *weight,
                                       __global const DATA_T *bias,
                                       __global DATA_T *output,
                                       int padT,
                                       int padR,
                                       int padB,
                                       int padL,
                                       int outputNumber,
                                       int outputChannel,
                                       int outputHeight,
                                       int outputWidth,
                                       int inputNum,
                                       int inputChannel,
                                       int inputHeight,
                                       int inputWidth) {
    DIRECT3X3_4X4(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(direct3x3_4x2x2_FP16, (__global const DATA_T *input,
                                         __global const DATA_T *weight,
                                         __global const DATA_T *bias,
                                         __global DATA_T *output,
                                         int padT,
                                         int padR,
                                         int padB,
                                         int padL,
                                         int outputNumber,
                                         int outputChannel,
                                         int outputHeight,
                                         int outputWidth,
                                         int inputNum,
                                         int inputChannel,
                                         int inputHeight,
                                         int inputWidth) {
    DIRECT3X3_4X2X2_FP16(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, \
                             outputHeight, outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(direct3x3_4x4x2_FP16, (__global const DATA_T *input,
                                         __global const DATA_T *weight,
                                         __global const DATA_T *bias,
                                         __global DATA_T *output,
                                         int padT,
                                         int padR,
                                         int padB,
                                         int padL,
                                         int outputNumber,
                                         int outputChannel,
                                         int outputHeight,
                                         int outputWidth,
                                         int inputNum,
                                         int inputChannel,
                                         int inputHeight,
                                         int inputWidth) {
    DIRECT3X3_4X4X2_FP16(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, \
                        outputHeight, outputWidth, inputNum, inputChannel, inputHeight, inputWidth) \
})
#undef ACT_VEC_F  // NONE

// apply RELU to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELUdirect3x3_4x8_FP16, (__global const DATA_T *input,
                                           __global const DATA_T *weight,
                                           __global const DATA_T *bias,
                                           __global DATA_T *output,
                                           int padT,
                                           int padR,
                                           int padB,
                                           int padL,
                                           int outputNumber,
                                           int outputChannel,
                                           int outputHeight,
                                           int outputWidth,
                                           int inputNum,
                                           int inputChannel,
                                           int inputHeight,
                                           int inputWidth) {
    DIRECT3X3_4X8_FP16(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                       outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELUdirect3x3_8x8_FP16, (__global const DATA_T *input,
                                           __global const DATA_T *weight,
                                           __global const DATA_T *bias,
                                           __global DATA_T *output,
                                           int padT,
                                           int padR,
                                           int padB,
                                           int padL,
                                           int outputNumber,
                                           int outputChannel,
                                           int outputHeight,
                                           int outputWidth,
                                           int inputNum,
                                           int inputChannel,
                                           int inputHeight,
                                           int inputWidth) {
    DIRECT3X3_8X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELUdirect5x5_4x8_FP16, (__global const DATA_T *input,
                                           __global const DATA_T *weight,
                                           __global const DATA_T *bias,
                                           __global DATA_T *output,
                                           int padT,
                                           int padR,
                                           int padB,
                                           int padL,
                                           int outputNumber,
                                           int outputChannel,
                                           int outputHeight,
                                           int outputWidth,
                                           int inputNum,
                                           int inputChannel,
                                           int inputHeight,
                                           int inputWidth) {
    DIRECT5X5_4X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELUdirect5x5_8x8_FP16, (__global const DATA_T *input,
                                           __global const DATA_T *weight,
                                           __global const DATA_T *bias,
                                           __global DATA_T *output,
                                           int padT,
                                           int padR,
                                           int padB,
                                           int padL,
                                           int outputNumber,
                                           int outputChannel,
                                           int outputHeight,
                                           int outputWidth,
                                           int inputNum,
                                           int inputChannel,
                                           int inputHeight,
                                           int inputWidth) {
    DIRECT5X5_8X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELUdirect7x7_4x8_merge_FP16, (__global const DATA_T *bias,
                                                 __global DATA_T *aligned_output,
                                                 __global DATA_T *output,
                                                 int padT,
                                                 int padR,
                                                 int padB,
                                                 int padL,
                                                 int outputNumber,
                                                 int outputChannel,
                                                 int outputHeight,
                                                 int outputWidth,
                                                 int inputNum,
                                                 int inputChannel,
                                                 int inputHeight,
                                                 int inputWidth,
                                                 int splite_num) {
    DIRECT7X7_4X8_MERGE(bias, aligned_output, output, padT, padR, padB, padL, outputNumber, outputChannel, \
                        outputHeight, outputWidth, inputNum, inputChannel, inputHeight, inputWidth, splite_num)
})

ADD_SINGLE_KERNEL(RELUdirect7x7_8x8_FP16, (__global const DATA_T *input,
                                           __global const DATA_T *weight,
                                           __global const DATA_T *bias,
                                           __global DATA_T *output,
                                           int padT,
                                           int padR,
                                           int padB,
                                           int padL,
                                           int outputNumber,
                                           int outputChannel,
                                           int outputHeight,
                                           int outputWidth,
                                           int inputNum,
                                           int inputChannel,
                                           int inputHeight,
                                           int inputWidth) {
    DIRECT7X7_8X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELUdirect9x9_4x8_merge_FP16, (__global const DATA_T *bias,
                                                 __global DATA_T *aligned_output,
                                                 __global DATA_T *output,
                                                 int padT,
                                                 int padR,
                                                 int padB,
                                                 int padL,
                                                 int outputNumber,
                                                 int outputChannel,
                                                 int outputHeight,
                                                 int outputWidth,
                                                 int inputNum,
                                                 int inputChannel,
                                                 int inputHeight,
                                                 int inputWidth,
                                                 int splite_num) {
    DIRECT9X9_4X8_MERGE(bias, aligned_output, output, padT, padR, padB, padL, outputNumber, outputChannel, \
                        outputHeight, outputWidth, inputNum, inputChannel, inputHeight, inputWidth, splite_num)
})

ADD_SINGLE_KERNEL(RELUdirect9x9_8x8_FP16, (__global const DATA_T *input,
                                           __global const DATA_T *weight,
                                           __global const DATA_T *bias,
                                           __global DATA_T *output,
                                           int padT,
                                           int padR,
                                           int padB,
                                           int padL,
                                           int outputNumber,
                                           int outputChannel,
                                           int outputHeight,
                                           int outputWidth,
                                           int inputNum,
                                           int inputChannel,
                                           int inputHeight,
                                           int inputWidth) {
    DIRECT9X9_8X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELUdirect3x3_4x4_FP16, (__global const DATA_T *input,
                                       __global const DATA_T *weight,
                                       __global const DATA_T *bias,
                                       __global DATA_T *output,
                                       int padT,
                                       int padR,
                                       int padB,
                                       int padL,
                                       int outputNumber,
                                       int outputChannel,
                                       int outputHeight,
                                       int outputWidth,
                                       int inputNum,
                                       int inputChannel,
                                       int inputHeight,
                                       int inputWidth) {
    DIRECT3X3_4X4(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELUdirect3x3_4x2x2_FP16, (__global const DATA_T *input,
                                         __global const DATA_T *weight,
                                         __global const DATA_T *bias,
                                         __global DATA_T *output,
                                         int padT,
                                         int padR,
                                         int padB,
                                         int padL,
                                         int outputNumber,
                                         int outputChannel,
                                         int outputHeight,
                                         int outputWidth,
                                         int inputNum,
                                         int inputChannel,
                                         int inputHeight,
                                         int inputWidth) {
    DIRECT3X3_4X2X2_FP16(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, \
                         outputHeight, outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELUdirect3x3_4x4x2_FP16, (__global const DATA_T *input,
                                         __global const DATA_T *weight,
                                         __global const DATA_T *bias,
                                         __global DATA_T *output,
                                         int padT,
                                         int padR,
                                         int padB,
                                         int padL,
                                         int outputNumber,
                                         int outputChannel,
                                         int outputHeight,
                                         int outputWidth,
                                         int inputNum,
                                         int inputChannel,
                                         int inputHeight,
                                         int inputWidth) {
    DIRECT3X3_4X4X2_FP16(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, \
                         outputHeight, outputWidth, inputNum, inputChannel, inputHeight, inputWidth) \
})

ADD_SINGLE_KERNEL(RELUdirect7x7_4x8_FP16, (__global const half *input,
                                     __global const half *weight,
                                     __global const half *bias,
                                     __global half *aligned_output,
                                     int padT,
                                     int padR,
                                     int padB,
                                     int padL,
                                     int outputNumber,
                                     int outputChannel,
                                     int outputHeight,
                                     int outputWidth,
                                     int inputNum,
                                     int inputChannel,
                                     int inputHeight,
                                     int inputWidth,
                                     int splite_num) {
    DIRECT7X7_4X8_FP16(input, weight, bias, aligned_output, padT, padR, padB, padL, outputNumber, outputChannel, \
                  outputHeight, outputWidth, inputNum, inputChannel, inputHeight,\
                  inputWidth, splite_num)
})

ADD_SINGLE_KERNEL(RELUdirect9x9_4x8_FP16, (__global const half *input,
                                     __global const half *weight,
                                     __global const half *bias,
                                     __global half *aligned_output,
                                     int padT,
                                     int padR,
                                     int padB,
                                     int padL,
                                     int outputNumber,
                                     int outputChannel,
                                     int outputHeight,
                                     int outputWidth,
                                     int inputNum,
                                     int inputChannel,
                                     int inputHeight,
                                     int inputWidth,
                                     int splite_num) {
    DIRECT9X9_4X8_FP16(input, weight, bias, aligned_output, padT, padR, padB, padL, outputNumber, outputChannel, \
     outputHeight, outputWidth, inputNum, inputChannel, inputHeight, inputWidth, splite_num)
})
#undef ACT_VEC_F  // RELU

// apply RELU6 to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU6_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELU6direct3x3_4x8_FP16, (__global const DATA_T *input,
                                            __global const DATA_T *weight,
                                            __global const DATA_T *bias,
                                            __global DATA_T *output,
                                            int padT,
                                            int padR,
                                            int padB,
                                            int padL,
                                            int outputNumber,
                                            int outputChannel,
                                            int outputHeight,
                                            int outputWidth,
                                            int inputNum,
                                            int inputChannel,
                                            int inputHeight,
                                            int inputWidth) {
    DIRECT3X3_4X8_FP16(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                       outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELU6direct3x3_8x8_FP16, (__global const DATA_T *input,
                                            __global const DATA_T *weight,
                                            __global const DATA_T *bias,
                                            __global DATA_T *output,
                                            int padT,
                                            int padR,
                                            int padB,
                                            int padL,
                                            int outputNumber,
                                            int outputChannel,
                                            int outputHeight,
                                            int outputWidth,
                                            int inputNum,
                                            int inputChannel,
                                            int inputHeight,
                                            int inputWidth) {
    DIRECT3X3_8X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELU6direct5x5_4x8_FP16, (__global const DATA_T *input,
                                            __global const DATA_T *weight,
                                            __global const DATA_T *bias,
                                            __global DATA_T *output,
                                            int padT,
                                            int padR,
                                            int padB,
                                            int padL,
                                            int outputNumber,
                                            int outputChannel,
                                            int outputHeight,
                                            int outputWidth,
                                            int inputNum,
                                            int inputChannel,
                                            int inputHeight,
                                            int inputWidth) {
    DIRECT5X5_4X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELU6direct5x5_8x8_FP16, (__global const DATA_T *input,
                                            __global const DATA_T *weight,
                                            __global const DATA_T *bias,
                                            __global DATA_T *output,
                                            int padT,
                                            int padR,
                                            int padB,
                                            int padL,
                                            int outputNumber,
                                            int outputChannel,
                                            int outputHeight,
                                            int outputWidth,
                                            int inputNum,
                                            int inputChannel,
                                            int inputHeight,
                                            int inputWidth) {
    DIRECT5X5_8X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELU6direct7x7_4x8_merge_FP16, (__global const DATA_T *bias,
                                                  __global DATA_T *aligned_output,
                                                  __global DATA_T *output,
                                                  int padT,
                                                  int padR,
                                                  int padB,
                                                  int padL,
                                                  int outputNumber,
                                                  int outputChannel,
                                                  int outputHeight,
                                                  int outputWidth,
                                                  int inputNum,
                                                  int inputChannel,
                                                  int inputHeight,
                                                  int inputWidth,
                                                  int splite_num) {
    DIRECT7X7_4X8_MERGE(bias, aligned_output, output, padT, padR, padB, padL, outputNumber, outputChannel, \
                        outputHeight, outputWidth, inputNum, inputChannel, inputHeight, inputWidth, splite_num)
})

ADD_SINGLE_KERNEL(RELU6direct7x7_8x8_FP16, (__global const DATA_T *input,
                                            __global const DATA_T *weight,
                                            __global const DATA_T *bias,
                                            __global DATA_T *output,
                                            int padT,
                                            int padR,
                                            int padB,
                                            int padL,
                                            int outputNumber,
                                            int outputChannel,
                                            int outputHeight,
                                            int outputWidth,
                                            int inputNum,
                                            int inputChannel,
                                            int inputHeight,
                                            int inputWidth) {
    DIRECT7X7_8X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELU6direct9x9_4x8_merge_FP16, (__global const DATA_T *bias,
                                                  __global DATA_T *aligned_output,
                                                  __global DATA_T *output,
                                                  int padT,
                                                  int padR,
                                                  int padB,
                                                  int padL,
                                                  int outputNumber,
                                                  int outputChannel,
                                                  int outputHeight,
                                                  int outputWidth,
                                                  int inputNum,
                                                  int inputChannel,
                                                  int inputHeight,
                                                  int inputWidth,
                                                  int splite_num) {
    DIRECT9X9_4X8_MERGE(bias, aligned_output, output, padT, padR, padB, padL, outputNumber, outputChannel, \
                        outputHeight, outputWidth, inputNum, inputChannel, inputHeight, inputWidth, splite_num)
})

ADD_SINGLE_KERNEL(RELU6direct9x9_8x8_FP16, (__global const DATA_T *input,
                                            __global const DATA_T *weight,
                                            __global const DATA_T *bias,
                                            __global DATA_T *output,
                                            int padT,
                                            int padR,
                                            int padB,
                                            int padL,
                                            int outputNumber,
                                            int outputChannel,
                                            int outputHeight,
                                            int outputWidth,
                                            int inputNum,
                                            int inputChannel,
                                            int inputHeight,
                                            int inputWidth) {
    DIRECT9X9_8X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELU6direct3x3_4x4_FP16, (__global const DATA_T *input,
                                       __global const DATA_T *weight,
                                       __global const DATA_T *bias,
                                       __global DATA_T *output,
                                       int padT,
                                       int padR,
                                       int padB,
                                       int padL,
                                       int outputNumber,
                                       int outputChannel,
                                       int outputHeight,
                                       int outputWidth,
                                       int inputNum,
                                       int inputChannel,
                                       int inputHeight,
                                       int inputWidth) {
    DIRECT3X3_4X4(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELU6direct3x3_4x2x2_FP16, (__global const DATA_T *input,
                                         __global const DATA_T *weight,
                                         __global const DATA_T *bias,
                                         __global DATA_T *output,
                                         int padT,
                                         int padR,
                                         int padB,
                                         int padL,
                                         int outputNumber,
                                         int outputChannel,
                                         int outputHeight,
                                         int outputWidth,
                                         int inputNum,
                                         int inputChannel,
                                         int inputHeight,
                                         int inputWidth) {
    DIRECT3X3_4X2X2_FP16(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, \
                         outputHeight, outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELU6direct3x3_4x4x2_FP16, (__global const DATA_T *input,
                                         __global const DATA_T *weight,
                                         __global const DATA_T *bias,
                                         __global DATA_T *output,
                                         int padT,
                                         int padR,
                                         int padB,
                                         int padL,
                                         int outputNumber,
                                         int outputChannel,
                                         int outputHeight,
                                         int outputWidth,
                                         int inputNum,
                                         int inputChannel,
                                         int inputHeight,
                                         int inputWidth) {
    DIRECT3X3_4X4X2_FP16(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, \
                         outputHeight, outputWidth, inputNum, inputChannel, inputHeight, inputWidth) \
})

ADD_SINGLE_KERNEL(RELU6direct7x7_4x8_FP16, (__global const half *input,
                                     __global const half *weight,
                                     __global const half *bias,
                                     __global half *aligned_output,
                                     int padT,
                                     int padR,
                                     int padB,
                                     int padL,
                                     int outputNumber,
                                     int outputChannel,
                                     int outputHeight,
                                     int outputWidth,
                                     int inputNum,
                                     int inputChannel,
                                     int inputHeight,
                                     int inputWidth,
                                     int splite_num) {
    DIRECT7X7_4X8_FP16(input, weight, bias, aligned_output, padT, padR, padB, padL, outputNumber, outputChannel, \
                  outputHeight, outputWidth, inputNum, inputChannel, inputHeight,\
                  inputWidth, splite_num)
})

ADD_SINGLE_KERNEL(RELU6direct9x9_4x8_FP16, (__global const half *input,
                                     __global const half *weight,
                                     __global const half *bias,
                                     __global half *aligned_output,
                                     int padT,
                                     int padR,
                                     int padB,
                                     int padL,
                                     int outputNumber,
                                     int outputChannel,
                                     int outputHeight,
                                     int outputWidth,
                                     int inputNum,
                                     int inputChannel,
                                     int inputHeight,
                                     int inputWidth,
                                     int splite_num) {
    DIRECT9X9_4X8_FP16(input, weight, bias, aligned_output, padT, padR, padB, padL, outputNumber, outputChannel, \
     outputHeight, outputWidth, inputNum, inputChannel, inputHeight, inputWidth, splite_num)
})
#undef ACT_VEC_F  // RELU6

ADD_SINGLE_KERNEL(align_weight_direct_FP16, (__global const DATA_T *weight,
                                             __global DATA_T *alinged_weight,
                                             int inputNumber,
                                             int inputChannel,
                                             int inputHeight,
                                             int inputWidth,
                                             int outputNumber,
                                             int outputChannel,
                                             int outputHeight,
                                             int outputWidth,
                                             int align_size) {
    ALIGN_WEIGHT_DIRECT(weight, alinged_weight, inputNumber, inputChannel, inputHeight, inputWidth, outputNumber, \
                        outputChannel, outputHeight, outputWidth, align_size)
})
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
ADD_SINGLE_KERNEL(direct3x3_4x8_FP32, (__global const DATA_T *input,
                                       __global const DATA_T *weight,
                                       __global const DATA_T *bias,
                                       __global DATA_T *output,
                                       int padT,
                                       int padR,
                                       int padB,
                                       int padL,
                                       int outputNumber,
                                       int outputChannel,
                                       int outputHeight,
                                       int outputWidth,
                                       int inputNum,
                                       int inputChannel,
                                       int inputHeight,
                                       int inputWidth) {
    DIRECT3X3_4X8_FP32(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                       outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(direct3x3_8x8_FP32, (__global const DATA_T *input,
                                       __global const DATA_T *weight,
                                       __global const DATA_T *bias,
                                       __global DATA_T *output,
                                       int padT,
                                       int padR,
                                       int padB,
                                       int padL,
                                       int outputNumber,
                                       int outputChannel,
                                       int outputHeight,
                                       int outputWidth,
                                       int inputNum,
                                       int inputChannel,
                                       int inputHeight,
                                       int inputWidth) {
    DIRECT3X3_8X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(direct5x5_4x8_FP32, (__global const DATA_T *input,
                                       __global const DATA_T *weight,
                                       __global const DATA_T *bias,
                                       __global DATA_T *output,
                                       int padT,
                                       int padR,
                                       int padB,
                                       int padL,
                                       int outputNumber,
                                       int outputChannel,
                                       int outputHeight,
                                       int outputWidth,
                                       int inputNum,
                                       int inputChannel,
                                       int inputHeight,
                                       int inputWidth) {
    DIRECT5X5_4X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(direct5x5_8x8_FP32, (__global const DATA_T *input,
                                       __global const DATA_T *weight,
                                       __global const DATA_T *bias,
                                       __global DATA_T *output,
                                       int padT,
                                       int padR,
                                       int padB,
                                       int padL,
                                       int outputNumber,
                                       int outputChannel,
                                       int outputHeight,
                                       int outputWidth,
                                       int inputNum,
                                       int inputChannel,
                                       int inputHeight,
                                       int inputWidth) {
    DIRECT5X5_8X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(direct7x7_4x8_FP32, (__global const DATA_T *input,
                                       __global const DATA_T *weight,
                                       __global const DATA_T *bias,
                                       __global DATA_T *output,
                                       int padT,
                                       int padR,
                                       int padB,
                                       int padL,
                                       int outputNumber,
                                       int outputChannel,
                                       int outputHeight,
                                       int outputWidth,
                                       int inputNum,
                                       int inputChannel,
                                       int inputHeight,
                                       int inputWidth) {
    DIRECT7X7_4X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(direct7x7_8x8_FP32, (__global const DATA_T *input,
                                       __global const DATA_T *weight,
                                       __global const DATA_T *bias,
                                       __global DATA_T *output,
                                       int padT,
                                       int padR,
                                       int padB,
                                       int padL,
                                       int outputNumber,
                                       int outputChannel,
                                       int outputHeight,
                                       int outputWidth,
                                       int inputNum,
                                       int inputChannel,
                                       int inputHeight,
                                       int inputWidth) {
    DIRECT7X7_8X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(direct9x9_4x8_FP32, (__global const DATA_T *input,
                                       __global const DATA_T *weight,
                                       __global const DATA_T *bias,
                                       __global DATA_T *output,
                                       int padT,
                                       int padR,
                                       int padB,
                                       int padL,
                                       int outputNumber,
                                       int outputChannel,
                                       int outputHeight,
                                       int outputWidth,
                                       int inputNum,
                                       int inputChannel,
                                       int inputHeight,
                                       int inputWidth) {
    DIRECT9X9_4X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(direct9x9_8x8_FP32, (__global const DATA_T *input,
                                       __global const DATA_T *weight,
                                       __global const DATA_T *bias,
                                       __global DATA_T *output,
                                       int padT,
                                       int padR,
                                       int padB,
                                       int padL,
                                       int outputNumber,
                                       int outputChannel,
                                       int outputHeight,
                                       int outputWidth,
                                       int inputNum,
                                       int inputChannel,
                                       int inputHeight,
                                       int inputWidth) {
    DIRECT9X9_8X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(direct3x3_4x4_FP32, (__global const DATA_T *input,
                                       __global const DATA_T *weight,
                                       __global const DATA_T *bias,
                                       __global DATA_T *output,
                                       int padT,
                                       int padR,
                                       int padB,
                                       int padL,
                                       int outputNumber,
                                       int outputChannel,
                                       int outputHeight,
                                       int outputWidth,
                                       int inputNum,
                                       int inputChannel,
                                       int inputHeight,
                                       int inputWidth) {
    DIRECT3X3_4X4(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(direct1x1_8x4_FP32, (__global const DATA_T *input,
                                       __global const DATA_T *weight,
                                       __global const DATA_T *bias,
                                       __global DATA_T *output,
                                       int padT,
                                       int padR,
                                       int padB,
                                       int padL,
                                       int outputNumber,
                                       int outputChannel,
                                       int outputHeight,
                                       int outputWidth,
                                       int inputNum,
                                       int inputChannel,
                                       int inputHeight,
                                       int inputWidth) {
    DIRECT1X1_8X4_FP32(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                       outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})
#undef ACT_VEC_F  // NONE

// apply RELU to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELUdirect3x3_4x8_FP32, (__global const DATA_T *input,
                                           __global const DATA_T *weight,
                                           __global const DATA_T *bias,
                                           __global DATA_T *output,
                                           int padT,
                                           int padR,
                                           int padB,
                                           int padL,
                                           int outputNumber,
                                           int outputChannel,
                                           int outputHeight,
                                           int outputWidth,
                                           int inputNum,
                                           int inputChannel,
                                           int inputHeight,
                                           int inputWidth) {
    DIRECT3X3_4X8_FP32(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                       outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELUdirect3x3_8x8_FP32, (__global const DATA_T *input,
                                           __global const DATA_T *weight,
                                           __global const DATA_T *bias,
                                           __global DATA_T *output,
                                           int padT,
                                           int padR,
                                           int padB,
                                           int padL,
                                           int outputNumber,
                                           int outputChannel,
                                           int outputHeight,
                                           int outputWidth,
                                           int inputNum,
                                           int inputChannel,
                                           int inputHeight,
                                           int inputWidth) {
    DIRECT3X3_8X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELUdirect5x5_4x8_FP32, (__global const DATA_T *input,
                                           __global const DATA_T *weight,
                                           __global const DATA_T *bias,
                                           __global DATA_T *output,
                                           int padT,
                                           int padR,
                                           int padB,
                                           int padL,
                                           int outputNumber,
                                           int outputChannel,
                                           int outputHeight,
                                           int outputWidth,
                                           int inputNum,
                                           int inputChannel,
                                           int inputHeight,
                                           int inputWidth) {
    DIRECT5X5_4X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELUdirect5x5_8x8_FP32, (__global const DATA_T *input,
                                           __global const DATA_T *weight,
                                           __global const DATA_T *bias,
                                           __global DATA_T *output,
                                           int padT,
                                           int padR,
                                           int padB,
                                           int padL,
                                           int outputNumber,
                                           int outputChannel,
                                           int outputHeight,
                                           int outputWidth,
                                           int inputNum,
                                           int inputChannel,
                                           int inputHeight,
                                           int inputWidth) {
    DIRECT5X5_8X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELUdirect7x7_4x8_FP32, (__global const DATA_T *input,
                                           __global const DATA_T *weight,
                                           __global const DATA_T *bias,
                                           __global DATA_T *output,
                                           int padT,
                                           int padR,
                                           int padB,
                                           int padL,
                                           int outputNumber,
                                           int outputChannel,
                                           int outputHeight,
                                           int outputWidth,
                                           int inputNum,
                                           int inputChannel,
                                           int inputHeight,
                                           int inputWidth) {
    DIRECT7X7_4X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELUdirect7x7_8x8_FP32, (__global const DATA_T *input,
                                           __global const DATA_T *weight,
                                           __global const DATA_T *bias,
                                           __global DATA_T *output,
                                           int padT,
                                           int padR,
                                           int padB,
                                           int padL,
                                           int outputNumber,
                                           int outputChannel,
                                           int outputHeight,
                                           int outputWidth,
                                           int inputNum,
                                           int inputChannel,
                                           int inputHeight,
                                           int inputWidth) {
    DIRECT7X7_8X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELUdirect9x9_4x8_FP32, (__global const DATA_T *input,
                                           __global const DATA_T *weight,
                                           __global const DATA_T *bias,
                                           __global DATA_T *output,
                                           int padT,
                                           int padR,
                                           int padB,
                                           int padL,
                                           int outputNumber,
                                           int outputChannel,
                                           int outputHeight,
                                           int outputWidth,
                                           int inputNum,
                                           int inputChannel,
                                           int inputHeight,
                                           int inputWidth) {
    DIRECT9X9_4X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELUdirect9x9_8x8_FP32, (__global const DATA_T *input,
                                           __global const DATA_T *weight,
                                           __global const DATA_T *bias,
                                           __global DATA_T *output,
                                           int padT,
                                           int padR,
                                           int padB,
                                           int padL,
                                           int outputNumber,
                                           int outputChannel,
                                           int outputHeight,
                                           int outputWidth,
                                           int inputNum,
                                           int inputChannel,
                                           int inputHeight,
                                           int inputWidth) {
    DIRECT9X9_8X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELUdirect3x3_4x4_FP32, (__global const DATA_T *input,
                                           __global const DATA_T *weight,
                                           __global const DATA_T *bias,
                                           __global DATA_T *output,
                                           int padT,
                                           int padR,
                                           int padB,
                                           int padL,
                                           int outputNumber,
                                           int outputChannel,
                                           int outputHeight,
                                           int outputWidth,
                                           int inputNum,
                                           int inputChannel,
                                           int inputHeight,
                                           int inputWidth) {
    DIRECT3X3_4X4(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELUdirect1x1_8x4_FP32, (__global const DATA_T *input,
                                           __global const DATA_T *weight,
                                           __global const DATA_T *bias,
                                           __global DATA_T *output,
                                           int padT,
                                           int padR,
                                           int padB,
                                           int padL,
                                           int outputNumber,
                                           int outputChannel,
                                           int outputHeight,
                                           int outputWidth,
                                           int inputNum,
                                           int inputChannel,
                                           int inputHeight,
                                           int inputWidth) {
    DIRECT1X1_8X4_FP32(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                       outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})
#undef ACT_VEC_F  // RELU

// apply RELU6 to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU6_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELU6direct3x3_4x8_FP32, (__global const DATA_T *input,
                                            __global const DATA_T *weight,
                                            __global const DATA_T *bias,
                                            __global DATA_T *output,
                                            int padT,
                                            int padR,
                                            int padB,
                                            int padL,
                                            int outputNumber,
                                            int outputChannel,
                                            int outputHeight,
                                            int outputWidth,
                                            int inputNum,
                                            int inputChannel,
                                            int inputHeight,
                                            int inputWidth) {
    DIRECT3X3_4X8_FP32(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                       outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELU6direct3x3_8x8_FP32, (__global const DATA_T *input,
                                            __global const DATA_T *weight,
                                            __global const DATA_T *bias,
                                            __global DATA_T *output,
                                            int padT,
                                            int padR,
                                            int padB,
                                            int padL,
                                            int outputNumber,
                                            int outputChannel,
                                            int outputHeight,
                                            int outputWidth,
                                            int inputNum,
                                            int inputChannel,
                                            int inputHeight,
                                            int inputWidth) {
    DIRECT3X3_8X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELU6direct5x5_4x8_FP32, (__global const DATA_T *input,
                                     __global const DATA_T *weight,
                                     __global const DATA_T *bias,
                                     __global DATA_T *output,
                                     int padT,
                                     int padR,
                                     int padB,
                                     int padL,
                                     int outputNumber,
                                     int outputChannel,
                                     int outputHeight,
                                     int outputWidth,
                                     int inputNum,
                                     int inputChannel,
                                     int inputHeight,
                                     int inputWidth) {
    DIRECT5X5_4X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELU6direct5x5_8x8_FP32, (__global const DATA_T *input,
                                     __global const DATA_T *weight,
                                     __global const DATA_T *bias,
                                     __global DATA_T *output,
                                     int padT,
                                     int padR,
                                     int padB,
                                     int padL,
                                     int outputNumber,
                                     int outputChannel,
                                     int outputHeight,
                                     int outputWidth,
                                     int inputNum,
                                     int inputChannel,
                                     int inputHeight,
                                     int inputWidth) {
    DIRECT5X5_8X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELU6direct7x7_4x8_FP32, (__global const DATA_T *input,
                                     __global const DATA_T *weight,
                                     __global const DATA_T *bias,
                                     __global DATA_T *output,
                                     int padT,
                                     int padR,
                                     int padB,
                                     int padL,
                                     int outputNumber,
                                     int outputChannel,
                                     int outputHeight,
                                     int outputWidth,
                                     int inputNum,
                                     int inputChannel,
                                     int inputHeight,
                                     int inputWidth) {
    DIRECT7X7_4X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELU6direct7x7_8x8_FP32, (__global const DATA_T *input,
                                     __global const DATA_T *weight,
                                     __global const DATA_T *bias,
                                     __global DATA_T *output,
                                     int padT,
                                     int padR,
                                     int padB,
                                     int padL,
                                     int outputNumber,
                                     int outputChannel,
                                     int outputHeight,
                                     int outputWidth,
                                     int inputNum,
                                     int inputChannel,
                                     int inputHeight,
                                     int inputWidth) {
    DIRECT7X7_8X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELU6direct9x9_4x8_FP32, (__global const DATA_T *input,
                                     __global const DATA_T *weight,
                                     __global const DATA_T *bias,
                                     __global DATA_T *output,
                                     int padT,
                                     int padR,
                                     int padB,
                                     int padL,
                                     int outputNumber,
                                     int outputChannel,
                                     int outputHeight,
                                     int outputWidth,
                                     int inputNum,
                                     int inputChannel,
                                     int inputHeight,
                                     int inputWidth) {
    DIRECT9X9_4X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELU6direct9x9_8x8_FP32, (__global const DATA_T *input,
                                     __global const DATA_T *weight,
                                     __global const DATA_T *bias,
                                     __global DATA_T *output,
                                     int padT,
                                     int padR,
                                     int padB,
                                     int padL,
                                     int outputNumber,
                                     int outputChannel,
                                     int outputHeight,
                                     int outputWidth,
                                     int inputNum,
                                     int inputChannel,
                                     int inputHeight,
                                     int inputWidth) {
    DIRECT9X9_8X8(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELU6direct3x3_4x4_FP32, (__global const DATA_T *input,
                                            __global const DATA_T *weight,
                                            __global const DATA_T *bias,
                                            __global DATA_T *output,
                                            int padT,
                                            int padR,
                                            int padB,
                                            int padL,
                                            int outputNumber,
                                            int outputChannel,
                                            int outputHeight,
                                            int outputWidth,
                                            int inputNum,
                                            int inputChannel,
                                            int inputHeight,
                                            int inputWidth) {
    DIRECT3X3_4X4(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                  outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})

ADD_SINGLE_KERNEL(RELU6direct1x1_8x4_FP32, (__global const DATA_T *input,
                                            __global const DATA_T *weight,
                                            __global const DATA_T *bias,
                                            __global DATA_T *output,
                                            int padT,
                                            int padR,
                                            int padB,
                                            int padL,
                                            int outputNumber,
                                            int outputChannel,
                                            int outputHeight,
                                            int outputWidth,
                                            int inputNum,
                                            int inputChannel,
                                            int inputHeight,
                                            int inputWidth) {
    DIRECT1X1_8X4_FP32(input, weight, bias, output, padT, padR, padB, padL, outputNumber, outputChannel, outputHeight, \
                       outputWidth, inputNum, inputChannel, inputHeight, inputWidth)
})
#undef ACT_VEC_F  // RELU6

ADD_SINGLE_KERNEL(align_weight_direct_FP32, (__global const DATA_T *weight,
                                             __global DATA_T *alinged_weight,
                                             int inputNumber,
                                             int inputChannel,
                                             int inputHeight,
                                             int inputWidth,
                                             int outputNumber,
                                             int outputChannel,
                                             int outputHeight,
                                             int outputWidth,
                                             int align_size) {
    ALIGN_WEIGHT_DIRECT(weight, alinged_weight, inputNumber, inputChannel, inputHeight, inputWidth, outputNumber, \
                        outputChannel, outputHeight, outputWidth, align_size)
})
#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // float

#define Q_FEED_DIRECT_OUTPUT_8X4(bias, outputC, outputW, outputChannel, outputHeight, outputWidth, output_shift, \
                                 output_multiplier, outputOffset, activation_min, activation_max, output0, output1, \
                                 output2, output3, output, outputIndex) \
    int8 biasVector = vload8(0, bias + outputC); \
    output0 += biasVector; \
    output1 += biasVector; \
    output2 += biasVector; \
    output3 += biasVector; \
    int left_shift = output_shift > 0 ? output_shift + 2 : 2; \
    int right_shift = output_shift > 0 ? 0 : -output_shift; \
    int mask = (((long)1 << right_shift) - 1); \
    int threshold_mask = mask >> 1; \
    int8 threshold8; \
    output0 = mul_hi(output0 << left_shift, output_multiplier); \
    output0 = rhadd(output0, 0); \
    threshold8 = threshold_mask + select((int8)(0), (int8)(1), output0 < 0); \
    output0 = (output0 >> right_shift) + select((int8)(0), (int8)(1), (output0 & mask) > threshold8); \
    output0 += outputOffset; \
    reQuantizedAct(output0.s0, activation_min, activation_max); \
    reQuantizedAct(output0.s1, activation_min, activation_max); \
    reQuantizedAct(output0.s2, activation_min, activation_max); \
    reQuantizedAct(output0.s3, activation_min, activation_max); \
    reQuantizedAct(output0.s4, activation_min, activation_max); \
    reQuantizedAct(output0.s5, activation_min, activation_max); \
    reQuantizedAct(output0.s6, activation_min, activation_max); \
    reQuantizedAct(output0.s7, activation_min, activation_max); \
    output1 = mul_hi(output1 << left_shift, output_multiplier); \
    output1 = rhadd(output1, 0); \
    threshold8 = threshold_mask + select((int8)(0), (int8)(1), output1 < 0); \
    output1 = (output1 >> right_shift) + select((int8)(0), (int8)(1), (output1 & mask) > threshold8); \
    output1 += outputOffset; \
    reQuantizedAct(output1.s0, activation_min, activation_max); \
    reQuantizedAct(output1.s1, activation_min, activation_max); \
    reQuantizedAct(output1.s2, activation_min, activation_max); \
    reQuantizedAct(output1.s3, activation_min, activation_max); \
    reQuantizedAct(output1.s4, activation_min, activation_max); \
    reQuantizedAct(output1.s5, activation_min, activation_max); \
    reQuantizedAct(output1.s6, activation_min, activation_max); \
    reQuantizedAct(output1.s7, activation_min, activation_max); \
    output2 = mul_hi(output2 << left_shift, output_multiplier); \
    output2 = rhadd(output2, 0); \
    threshold8 = threshold_mask + select((int8)(0), (int8)(1), output2 < 0); \
    output2 = (output2 >> right_shift) + select((int8)(0), (int8)(1), (output2 & mask) > threshold8); \
    output2 += outputOffset; \
    reQuantizedAct(output2.s0, activation_min, activation_max); \
    reQuantizedAct(output2.s1, activation_min, activation_max); \
    reQuantizedAct(output2.s2, activation_min, activation_max); \
    reQuantizedAct(output2.s3, activation_min, activation_max); \
    reQuantizedAct(output2.s4, activation_min, activation_max); \
    reQuantizedAct(output2.s5, activation_min, activation_max); \
    reQuantizedAct(output2.s6, activation_min, activation_max); \
    reQuantizedAct(output2.s7, activation_min, activation_max); \
    output3 = mul_hi(output3 << left_shift, output_multiplier); \
    output3 = rhadd(output3, 0); \
    threshold8 = threshold_mask + select((int8)(0), (int8)(1), output3 < 0); \
    output3 = (output3 >> right_shift) + select((int8)(0), (int8)(1), (output3 & mask) > threshold8); \
    output3 += outputOffset; \
    reQuantizedAct(output3.s0, activation_min, activation_max); \
    reQuantizedAct(output3.s1, activation_min, activation_max); \
    reQuantizedAct(output3.s2, activation_min, activation_max); \
    reQuantizedAct(output3.s3, activation_min, activation_max); \
    reQuantizedAct(output3.s4, activation_min, activation_max); \
    reQuantizedAct(output3.s5, activation_min, activation_max); \
    reQuantizedAct(output3.s6, activation_min, activation_max); \
    reQuantizedAct(output3.s7, activation_min, activation_max); \
    if (outputW + 3 < outputWidth) { \
        if (outputC + 7 < outputChannel) { \
            vstore4((DATA_T4)(output0.s0, output1.s0, output2.s0, output3.s0), 0, output + outputIndex); \
            vstore4((DATA_T4)(output0.s1, output1.s1, output2.s1, output3.s1), \
                    0, \
                    output + outputIndex + outputHeight * outputWidth); \
            vstore4((DATA_T4)(output0.s2, output1.s2, output2.s2, output3.s2), \
                    0, \
                    output + outputIndex + 2 * outputHeight * outputWidth); \
            vstore4((DATA_T4)(output0.s3, output1.s3, output2.s3, output3.s3), \
                    0, \
                    output + outputIndex + 3 * outputHeight * outputWidth); \
            vstore4((DATA_T4)(output0.s4, output1.s4, output2.s4, output3.s4), \
                    0, \
                    output + outputIndex + 4 * outputHeight * outputWidth); \
            vstore4((DATA_T4)(output0.s5, output1.s5, output2.s5, output3.s5), \
                    0, \
                    output + outputIndex + 5 * outputHeight * outputWidth); \
            vstore4((DATA_T4)(output0.s6, output1.s6, output2.s6, output3.s6), \
                    0, \
                    output + outputIndex + 6 * outputHeight * outputWidth); \
            vstore4((DATA_T4)(output0.s7, output1.s7, output2.s7, output3.s7), \
                    0, \
                    output + outputIndex + 7 * outputHeight * outputWidth); \
        } else if (outputC + 6 < outputChannel) { \
            vstore4((DATA_T4)(output0.s0, output1.s0, output2.s0, output3.s0), 0, output + outputIndex); \
            vstore4((DATA_T4)(output0.s1, output1.s1, output2.s1, output3.s1), \
                    0, \
                    output + outputIndex + outputHeight * outputWidth); \
            vstore4((DATA_T4)(output0.s2, output1.s2, output2.s2, output3.s2), \
                    0, \
                    output + outputIndex + 2 * outputHeight * outputWidth); \
            vstore4((DATA_T4)(output0.s3, output1.s3, output2.s3, output3.s3), \
                    0, \
                    output + outputIndex + 3 * outputHeight * outputWidth); \
            vstore4((DATA_T4)(output0.s4, output1.s4, output2.s4, output3.s4), \
                    0, \
                    output + outputIndex + 4 * outputHeight * outputWidth); \
            vstore4((DATA_T4)(output0.s5, output1.s5, output2.s5, output3.s5), \
                    0, \
                    output + outputIndex + 5 * outputHeight * outputWidth); \
            vstore4((DATA_T4)(output0.s6, output1.s6, output2.s6, output3.s6), \
                    0, \
                    output + outputIndex + 6 * outputHeight * outputWidth); \
        } else if (outputC + 5 < outputChannel) { \
            vstore4((DATA_T4)(output0.s0, output1.s0, output2.s0, output3.s0), 0, output + outputIndex); \
            vstore4((DATA_T4)(output0.s1, output1.s1, output2.s1, output3.s1), \
                    0, \
                    output + outputIndex + outputHeight * outputWidth); \
            vstore4((DATA_T4)(output0.s2, output1.s2, output2.s2, output3.s2), \
                    0, \
                    output + outputIndex + 2 * outputHeight * outputWidth); \
            vstore4((DATA_T4)(output0.s3, output1.s3, output2.s3, output3.s3), \
                    0, \
                    output + outputIndex + 3 * outputHeight * outputWidth); \
            vstore4((DATA_T4)(output0.s4, output1.s4, output2.s4, output3.s4), \
                    0, \
                    output + outputIndex + 4 * outputHeight * outputWidth); \
            vstore4((DATA_T4)(output0.s5, output1.s5, output2.s5, output3.s5), \
                    0, \
                    output + outputIndex + 5 * outputHeight * outputWidth); \
        } else if (outputC + 4 < outputChannel) { \
            vstore4((DATA_T4)(output0.s0, output1.s0, output2.s0, output3.s0), 0, output + outputIndex); \
            vstore4((DATA_T4)(output0.s1, output1.s1, output2.s1, output3.s1), \
                    0, \
                    output + outputIndex + outputHeight * outputWidth); \
            vstore4((DATA_T4)(output0.s2, output1.s2, output2.s2, output3.s2), \
                    0, \
                    output + outputIndex + 2 * outputHeight * outputWidth); \
            vstore4((DATA_T4)(output0.s3, output1.s3, output2.s3, output3.s3), \
                    0, \
                    output + outputIndex + 3 * outputHeight * outputWidth); \
            vstore4((DATA_T4)(output0.s4, output1.s4, output2.s4, output3.s4), \
                    0, \
                    output + outputIndex + 4 * outputHeight * outputWidth); \
        } else if (outputC + 3 < outputChannel) { \
            vstore4((DATA_T4)(output0.s0, output1.s0, output2.s0, output3.s0), 0, output + outputIndex); \
            vstore4((DATA_T4)(output0.s1, output1.s1, output2.s1, output3.s1), \
                    0, \
                    output + outputIndex + outputHeight * outputWidth); \
            vstore4((DATA_T4)(output0.s2, output1.s2, output2.s2, output3.s2), \
                    0, \
                    output + outputIndex + 2 * outputHeight * outputWidth); \
            vstore4((DATA_T4)(output0.s3, output1.s3, output2.s3, output3.s3), \
                    0, \
                    output + outputIndex + 3 * outputHeight * outputWidth); \
        } else if (outputC + 2 < outputChannel) { \
            vstore4((DATA_T4)(output0.s0, output1.s0, output2.s0, output3.s0), 0, output + outputIndex); \
            vstore4((DATA_T4)(output0.s1, output1.s1, output2.s1, output3.s1), \
                    0, \
                    output + outputIndex + outputHeight * outputWidth); \
            vstore4((DATA_T4)(output0.s2, output1.s2, output2.s2, output3.s2), \
                    0, \
                    output + outputIndex + 2 * outputHeight * outputWidth); \
        } else if (outputC + 1 < outputChannel) { \
            vstore4((DATA_T4)(output0.s0, output1.s0, output2.s0, output3.s0), 0, output + outputIndex); \
            vstore4((DATA_T4)(output0.s1, output1.s1, output2.s1, output3.s1), \
                    0, \
                    output + outputIndex + outputHeight * outputWidth); \
        } else if (outputC < outputChannel) { \
            vstore4((DATA_T4)(output0.s0, output1.s0, output2.s0, output3.s0), 0, output + outputIndex); \
        } \
    }

#define Q_ARM_DOT_I4W16O8(I4, W16_0, W16_1, O8) \
    ARM_DOT(I4, W16_0.s0123, O8.s0); \
    ARM_DOT(I4, W16_0.s4567, O8.s1); \
    ARM_DOT(I4, W16_0.s89ab, O8.s2); \
    ARM_DOT(I4, W16_0.scdef, O8.s3); \
    ARM_DOT(I4, W16_1.s0123, O8.s4); \
    ARM_DOT(I4, W16_1.s4567, O8.s5); \
    ARM_DOT(I4, W16_1.s89ab, O8.s6); \
    ARM_DOT(I4, W16_1.scdef, O8.s7);

#define Q_ARM_DOT_SLIDEWINDOW_0TO9(weight, weightIndex, intputVector0, intputVector1, \
                    weightVector0, weightVector1, output0, output1, output2, output3) \
    weightVector0 = vload16(0, weight + weightIndex); \
    weightVector1 = vload16(1, weight + weightIndex); \
    Q_ARM_DOT_I4W16O8(intputVector0.lo.lo, weightVector0, weightVector1, output0) \
    Q_ARM_DOT_I4W16O8(intputVector0.lo.hi, weightVector0, weightVector1, output1) \
    Q_ARM_DOT_I4W16O8(intputVector0.hi.lo, weightVector0, weightVector1, output2) \
    Q_ARM_DOT_I4W16O8(intputVector0.hi.hi, weightVector0, weightVector1, output3) \
    weightVector0 = vload16(2, weight + weightIndex); \
    weightVector1 = vload16(3, weight + weightIndex); \
    Q_ARM_DOT_I4W16O8(intputVector0.lo.hi, weightVector0, weightVector1, output0) \
    Q_ARM_DOT_I4W16O8(intputVector0.hi.lo, weightVector0, weightVector1, output1) \
    Q_ARM_DOT_I4W16O8(intputVector0.hi.hi, weightVector0, weightVector1, output2) \
    Q_ARM_DOT_I4W16O8(intputVector1.lo.lo, weightVector0, weightVector1, output3) \
    weightVector0 = vload16(4, weight + weightIndex); \
    weightVector1 = vload16(5, weight + weightIndex); \
    Q_ARM_DOT_I4W16O8(intputVector0.hi.lo, weightVector0, weightVector1, output0) \
    Q_ARM_DOT_I4W16O8(intputVector0.hi.hi, weightVector0, weightVector1, output1) \
    Q_ARM_DOT_I4W16O8(intputVector1.lo.lo, weightVector0, weightVector1, output2) \
    Q_ARM_DOT_I4W16O8(intputVector1.lo.hi, weightVector0, weightVector1, output3) \
    weightVector0 = vload16(6, weight + weightIndex); \
    weightVector1 = vload16(7, weight + weightIndex); \
    Q_ARM_DOT_I4W16O8(intputVector0.hi.hi, weightVector0, weightVector1, output0) \
    Q_ARM_DOT_I4W16O8(intputVector1.lo.lo, weightVector0, weightVector1, output1) \
    Q_ARM_DOT_I4W16O8(intputVector1.lo.hi, weightVector0, weightVector1, output2) \
    Q_ARM_DOT_I4W16O8(intputVector1.hi.lo, weightVector0, weightVector1, output3) \
    weightVector0 = vload16(8, weight + weightIndex); \
    weightVector1 = vload16(9, weight + weightIndex); \
    Q_ARM_DOT_I4W16O8(intputVector1.lo.lo, weightVector0, weightVector1, output0) \
    Q_ARM_DOT_I4W16O8(intputVector1.lo.hi, weightVector0, weightVector1, output1) \
    Q_ARM_DOT_I4W16O8(intputVector1.hi.lo, weightVector0, weightVector1, output2) \
    Q_ARM_DOT_I4W16O8(intputVector1.hi.hi, weightVector0, weightVector1, output3)

/********  QUANTIZED KERNELS ********/
// UINT8 kernels
#define DATA_T uchar
#define DATA_T2 uchar2
#define DATA_T3 uchar3
#define DATA_T4 uchar4
#define DATA_T8 uchar8
#define DATA_T16 uchar16
#define CONVERT_TO_DATA_T(x) (DATA_T)x
#define CONVERT_TO_DATA_T2(x) convert_uchar2(x)
#define CONVERT_TO_DATA_T3(x) convert_uchar3(x)
#define CONVERT_TO_DATA_T4(x) convert_uchar4(x)

ADD_SINGLE_KERNEL(align_weight_direct_INT8, (__global const unsigned char *weight,
                                           __global unsigned char *alinged_weight,
                                           int inputNumber,
                                           int inputChannel,
                                           int inputHeight,
                                           int inputWidth,
                                           int outputNumber,
                                           int outputChannel,
                                           int outputHeight,
                                           int outputWidth,
                                           int align_size,
                                           int weightZeroPoint) {
        int outputW = get_global_id(0);
        int outputH = get_global_id(1);
        int outputC = get_global_id(2) % outputChannel;
        int outputN = get_global_id(2) / outputChannel;

        int inputW = outputW / (align_size * 4);
        int inputH = outputH;
        int inputC = outputC * 4 + (outputW % (align_size * 4) % 4);
        int inputN = outputN * align_size + (outputW % (align_size * 4)) / 4;

        if (inputW < inputWidth && inputH < inputHeight && inputC < inputChannel &&
            inputN < inputNumber) {
            alinged_weight[outputN * outputChannel * outputHeight * outputWidth +
                           outputC * outputHeight * outputWidth + outputH * outputWidth + outputW] =
                weight[inputN * inputChannel * inputHeight * inputWidth +
                       inputC * inputHeight * inputWidth + inputH * inputWidth + inputW];
        } else {
            alinged_weight[outputN * outputChannel * outputHeight * outputWidth +
                           outputC * outputHeight * outputWidth + outputH * outputWidth + outputW] =
                (unsigned char)weightZeroPoint;
        }
})

ADD_KERNEL_HEADER(direct3x3_8x4_INT8, {DEFINE_REQUANTIZED_ACT})

ADD_SINGLE_KERNEL(direct3x3_8x4_INT8, (__global const unsigned char *input,
                                     __global const unsigned char *weight,
                                     __global const int *bias,
                                     __global unsigned char *output,
                                     int padT,
                                     int padL,
                                     int outputNumber,
                                     int outputChannel,
                                     int outputHeight,
                                     int outputWidth,
                                     int inputNum,
                                     int inputChannel,
                                     int inputHeight,
                                     int inputWidth,
                                     int inputZeroPoint,
                                     int filterZeroPoint,
                                     int outputOffset,
                                     int output_multiplier,
                                     int output_shift,
                                     int activation_min,
                                     int activation_max) {
        int outputN = get_global_id(2) / ((outputChannel + 7) / 8);
        int outputC = get_global_id(2) % ((outputChannel + 7) / 8) * 8;
        int outputH = get_global_id(1);
        int outputW = get_global_id(0) * 4;

        if (outputW < outputWidth && outputH < outputHeight) {
            int8 output0 = (int8)0;
            int8 output1 = (int8)0;
            int8 output2 = (int8)0;
            int8 output3 = (int8)0;

            int in_0 = 0;
            int in_1 = 0;
            int in_2 = 0;
            int in_3 = 0;
            int in_4 = 0;
            int in_5 = 0;

            int inputBase = outputN * inputChannel * inputHeight * inputWidth + outputW * 4;
            int weightBase = outputC * inputChannel * 3 * 3;

            uchar16 intputVector0;
            uchar8 intputVector1;
            uchar16 weightVector_0;
            uchar16 weightVector_1;
            for (int inputC = 0; inputC < inputChannel / 4; inputC++) {
                int inputIndex = inputBase + inputC * 4 * inputHeight * inputWidth;
                int weightIndex = weightBase + inputC * 8 * 4 * 3 * 3;
                for (int kh = outputH; kh < outputH + 3; kh++) {
                    intputVector0 = vload16(0, input + inputIndex + kh * inputWidth * 4);
                    intputVector1 = vload8(2, input + inputIndex + kh * inputWidth * 4);

                    ARM_DOT(intputVector0.lo.lo, (uchar4)(filterZeroPoint), in_0);
                    ARM_DOT(intputVector0.lo.hi, (uchar4)(filterZeroPoint), in_1);
                    ARM_DOT(intputVector0.hi.lo, (uchar4)(filterZeroPoint), in_2);
                    ARM_DOT(intputVector0.hi.hi, (uchar4)(filterZeroPoint), in_3);
                    ARM_DOT(intputVector1.lo, (uchar4)(filterZeroPoint), in_4);
                    ARM_DOT(intputVector1.hi, (uchar4)(filterZeroPoint), in_5);

                    weightVector_0 = vload16(0, weight + weightIndex);
                    weightVector_1 = vload16(1, weight + weightIndex);
                    Q_ARM_DOT_I4W16O8(intputVector0.lo.lo, weightVector_0, weightVector_1, output0)
                    Q_ARM_DOT_I4W16O8(intputVector0.lo.hi, weightVector_0, weightVector_1, output1)
                    Q_ARM_DOT_I4W16O8(intputVector0.hi.lo, weightVector_0, weightVector_1, output2)
                    Q_ARM_DOT_I4W16O8(intputVector0.hi.hi, weightVector_0, weightVector_1, output3)

                    weightVector_0 = vload16(2, weight + weightIndex);
                    weightVector_1 = vload16(3, weight + weightIndex);
                    Q_ARM_DOT_I4W16O8(intputVector0.lo.hi, weightVector_0, weightVector_1, output0)
                    Q_ARM_DOT_I4W16O8(intputVector0.hi.lo, weightVector_0, weightVector_1, output1)
                    Q_ARM_DOT_I4W16O8(intputVector0.hi.hi, weightVector_0, weightVector_1, output2)
                    Q_ARM_DOT_I4W16O8(intputVector1.lo, weightVector_0, weightVector_1, output3)

                    weightVector_0 = vload16(4, weight + weightIndex);
                    weightVector_1 = vload16(5, weight + weightIndex);
                    Q_ARM_DOT_I4W16O8(intputVector0.hi.lo, weightVector_0, weightVector_1, output0)
                    Q_ARM_DOT_I4W16O8(intputVector0.hi.hi, weightVector_0, weightVector_1, output1)
                    Q_ARM_DOT_I4W16O8(intputVector1.lo, weightVector_0, weightVector_1, output2)
                    Q_ARM_DOT_I4W16O8(intputVector1.hi, weightVector_0, weightVector_1, output3)

                    weightIndex += 96;
                }
            }
            output0 = output0 - (in_0 + in_1 + in_2);
            output1 = output1 - (in_1 + in_2 + in_3);
            output2 = output2 - (in_2 + in_3 + in_4);
            output3 = output3 - (in_3 + in_4 + in_5);

            int outputIndex = outputN * outputChannel * outputHeight * outputWidth +
                              outputC * outputHeight * outputWidth + outputH * outputWidth +
                              outputW;
            int8 biasVector = vload8(0, bias + outputC);
            output0 += biasVector;
            output1 += biasVector;
            output2 += biasVector;
            output3 += biasVector;

            int left_shift = output_shift > 0 ? output_shift + 2 : 2;
            int right_shift = output_shift > 0 ? 0 : -output_shift;

            int mask = (((long)1 << right_shift) - 1);
            int threshold_mask = mask >> 1;
            int8 threshold8;

            output0 = mul_hi(output0 << left_shift, output_multiplier);
            output0 = rhadd(output0, 0);

            threshold8 = threshold_mask + select((int8)(0), (int8)(1), output0 < 0);
            output0 = (output0 >> right_shift) +
                      select((int8)(0), (int8)(1), (output0 & mask) > threshold8);
            output0 += outputOffset;

            reQuantizedAct(output0.s0, activation_min, activation_max);
            reQuantizedAct(output0.s1, activation_min, activation_max);
            reQuantizedAct(output0.s2, activation_min, activation_max);
            reQuantizedAct(output0.s3, activation_min, activation_max);
            reQuantizedAct(output0.s4, activation_min, activation_max);
            reQuantizedAct(output0.s5, activation_min, activation_max);
            reQuantizedAct(output0.s6, activation_min, activation_max);
            reQuantizedAct(output0.s7, activation_min, activation_max);

            output1 = mul_hi(output1 << left_shift, output_multiplier);
            output1 = rhadd(output1, 0);

            threshold8 = threshold_mask + select((int8)(0), (int8)(1), output1 < 0);
            output1 = (output1 >> right_shift) +
                      select((int8)(0), (int8)(1), (output1 & mask) > threshold8);
            output1 += outputOffset;

            reQuantizedAct(output1.s0, activation_min, activation_max);
            reQuantizedAct(output1.s1, activation_min, activation_max);
            reQuantizedAct(output1.s2, activation_min, activation_max);
            reQuantizedAct(output1.s3, activation_min, activation_max);
            reQuantizedAct(output1.s4, activation_min, activation_max);
            reQuantizedAct(output1.s5, activation_min, activation_max);
            reQuantizedAct(output1.s6, activation_min, activation_max);
            reQuantizedAct(output1.s7, activation_min, activation_max);

            output2 = mul_hi(output2 << left_shift, output_multiplier);
            output2 = rhadd(output2, 0);

            threshold8 = threshold_mask + select((int8)(0), (int8)(1), output2 < 0);
            output2 = (output2 >> right_shift) +
                      select((int8)(0), (int8)(1), (output2 & mask) > threshold8);
            output2 += outputOffset;

            reQuantizedAct(output2.s0, activation_min, activation_max);
            reQuantizedAct(output2.s1, activation_min, activation_max);
            reQuantizedAct(output2.s2, activation_min, activation_max);
            reQuantizedAct(output2.s3, activation_min, activation_max);
            reQuantizedAct(output2.s4, activation_min, activation_max);
            reQuantizedAct(output2.s5, activation_min, activation_max);
            reQuantizedAct(output2.s6, activation_min, activation_max);
            reQuantizedAct(output2.s7, activation_min, activation_max);

            output3 = mul_hi(output3 << left_shift, output_multiplier);
            output3 = rhadd(output3, 0);

            threshold8 = threshold_mask + select((int8)(0), (int8)(1), output3 < 0);
            output3 = (output3 >> right_shift) +
                      select((int8)(0), (int8)(1), (output3 & mask) > threshold8);
            output3 += outputOffset;

            reQuantizedAct(output3.s0, activation_min, activation_max);
            reQuantizedAct(output3.s1, activation_min, activation_max);
            reQuantizedAct(output3.s2, activation_min, activation_max);
            reQuantizedAct(output3.s3, activation_min, activation_max);
            reQuantizedAct(output3.s4, activation_min, activation_max);
            reQuantizedAct(output3.s5, activation_min, activation_max);
            reQuantizedAct(output3.s6, activation_min, activation_max);
            reQuantizedAct(output3.s7, activation_min, activation_max);

            if (outputW + 3 < outputWidth) {
                if (outputC + 7 < outputChannel) {
                    vstore4((uchar4)(output0.s0, output1.s0, output2.s0, output3.s0),
                            0,
                            output + outputIndex);
                    vstore4((uchar4)(output0.s1, output1.s1, output2.s1, output3.s1),
                            0,
                            output + outputIndex + outputHeight * outputWidth);
                    vstore4((uchar4)(output0.s2, output1.s2, output2.s2, output3.s2),
                            0,
                            output + outputIndex + 2 * outputHeight * outputWidth);
                    vstore4((uchar4)(output0.s3, output1.s3, output2.s3, output3.s3),
                            0,
                            output + outputIndex + 3 * outputHeight * outputWidth);
                    vstore4((uchar4)(output0.s4, output1.s4, output2.s4, output3.s4),
                            0,
                            output + outputIndex + 4 * outputHeight * outputWidth);
                    vstore4((uchar4)(output0.s5, output1.s5, output2.s5, output3.s5),
                            0,
                            output + outputIndex + 5 * outputHeight * outputWidth);
                    vstore4((uchar4)(output0.s6, output1.s6, output2.s6, output3.s6),
                            0,
                            output + outputIndex + 6 * outputHeight * outputWidth);
                    vstore4((uchar4)(output0.s7, output1.s7, output2.s7, output3.s7),
                            0,
                            output + outputIndex + 7 * outputHeight * outputWidth);
                } else if (outputC + 6 < outputChannel) {
                    vstore4((uchar4)(output0.s0, output1.s0, output2.s0, output3.s0),
                            0,
                            output + outputIndex);
                    vstore4((uchar4)(output0.s1, output1.s1, output2.s1, output3.s1),
                            0,
                            output + outputIndex + outputHeight * outputWidth);
                    vstore4((uchar4)(output0.s2, output1.s2, output2.s2, output3.s2),
                            0,
                            output + outputIndex + 2 * outputHeight * outputWidth);
                    vstore4((uchar4)(output0.s3, output1.s3, output2.s3, output3.s3),
                            0,
                            output + outputIndex + 3 * outputHeight * outputWidth);
                    vstore4((uchar4)(output0.s4, output1.s4, output2.s4, output3.s4),
                            0,
                            output + outputIndex + 4 * outputHeight * outputWidth);
                    vstore4((uchar4)(output0.s5, output1.s5, output2.s5, output3.s5),
                            0,
                            output + outputIndex + 5 * outputHeight * outputWidth);
                    vstore4((uchar4)(output0.s6, output1.s6, output2.s6, output3.s6),
                            0,
                            output + outputIndex + 6 * outputHeight * outputWidth);
                } else if (outputC + 5 < outputChannel) {
                    vstore4((uchar4)(output0.s0, output1.s0, output2.s0, output3.s0),
                            0,
                            output + outputIndex);
                    vstore4((uchar4)(output0.s1, output1.s1, output2.s1, output3.s1),
                            0,
                            output + outputIndex + outputHeight * outputWidth);
                    vstore4((uchar4)(output0.s2, output1.s2, output2.s2, output3.s2),
                            0,
                            output + outputIndex + 2 * outputHeight * outputWidth);
                    vstore4((uchar4)(output0.s3, output1.s3, output2.s3, output3.s3),
                            0,
                            output + outputIndex + 3 * outputHeight * outputWidth);
                    vstore4((uchar4)(output0.s4, output1.s4, output2.s4, output3.s4),
                            0,
                            output + outputIndex + 4 * outputHeight * outputWidth);
                    vstore4((uchar4)(output0.s5, output1.s5, output2.s5, output3.s5),
                            0,
                            output + outputIndex + 5 * outputHeight * outputWidth);
                } else if (outputC + 4 < outputChannel) {
                    vstore4((uchar4)(output0.s0, output1.s0, output2.s0, output3.s0),
                            0,
                            output + outputIndex);
                    vstore4((uchar4)(output0.s1, output1.s1, output2.s1, output3.s1),
                            0,
                            output + outputIndex + outputHeight * outputWidth);
                    vstore4((uchar4)(output0.s2, output1.s2, output2.s2, output3.s2),
                            0,
                            output + outputIndex + 2 * outputHeight * outputWidth);
                    vstore4((uchar4)(output0.s3, output1.s3, output2.s3, output3.s3),
                            0,
                            output + outputIndex + 3 * outputHeight * outputWidth);
                    vstore4((uchar4)(output0.s4, output1.s4, output2.s4, output3.s4),
                            0,
                            output + outputIndex + 4 * outputHeight * outputWidth);
                } else if (outputC + 3 < outputChannel) {
                    vstore4((uchar4)(output0.s0, output1.s0, output2.s0, output3.s0),
                            0,
                            output + outputIndex);
                    vstore4((uchar4)(output0.s1, output1.s1, output2.s1, output3.s1),
                            0,
                            output + outputIndex + outputHeight * outputWidth);
                    vstore4((uchar4)(output0.s2, output1.s2, output2.s2, output3.s2),
                            0,
                            output + outputIndex + 2 * outputHeight * outputWidth);
                    vstore4((uchar4)(output0.s3, output1.s3, output2.s3, output3.s3),
                            0,
                            output + outputIndex + 3 * outputHeight * outputWidth);
                } else if (outputC + 2 < outputChannel) {
                    vstore4((uchar4)(output0.s0, output1.s0, output2.s0, output3.s0),
                            0,
                            output + outputIndex);
                    vstore4((uchar4)(output0.s1, output1.s1, output2.s1, output3.s1),
                            0,
                            output + outputIndex + outputHeight * outputWidth);
                    vstore4((uchar4)(output0.s2, output1.s2, output2.s2, output3.s2),
                            0,
                            output + outputIndex + 2 * outputHeight * outputWidth);
                } else if (outputC + 1 < outputChannel) {
                    vstore4((uchar4)(output0.s0, output1.s0, output2.s0, output3.s0),
                            0,
                            output + outputIndex);
                    vstore4((uchar4)(output0.s1, output1.s1, output2.s1, output3.s1),
                            0,
                            output + outputIndex + outputHeight * outputWidth);
                } else if (outputC < outputChannel) {
                    vstore4((uchar4)(output0.s0, output1.s0, output2.s0, output3.s0),
                            0,
                            output + outputIndex);
                }
            } else if (outputW + 2 < outputWidth) {
                if (outputC + 7 < outputChannel) {
                    vstore3((uchar3)(output0.s0, output1.s0, output2.s0), 0, output + outputIndex);
                    vstore3((uchar3)(output0.s1, output1.s1, output2.s1),
                            0,
                            output + outputIndex + outputHeight * outputWidth);
                    vstore3((uchar3)(output0.s2, output1.s2, output2.s2),
                            0,
                            output + outputIndex + 2 * outputHeight * outputWidth);
                    vstore3((uchar3)(output0.s3, output1.s3, output2.s3),
                            0,
                            output + outputIndex + 3 * outputHeight * outputWidth);
                    vstore3((uchar3)(output0.s4, output1.s4, output2.s4),
                            0,
                            output + outputIndex + 4 * outputHeight * outputWidth);
                    vstore3((uchar3)(output0.s5, output1.s5, output2.s5),
                            0,
                            output + outputIndex + 5 * outputHeight * outputWidth);
                    vstore3((uchar3)(output0.s6, output1.s6, output2.s6),
                            0,
                            output + outputIndex + 6 * outputHeight * outputWidth);
                    vstore3((uchar3)(output0.s7, output1.s7, output2.s7),
                            0,
                            output + outputIndex + 7 * outputHeight * outputWidth);
                } else if (outputC + 6 < outputChannel) {
                    vstore3((uchar3)(output0.s0, output1.s0, output2.s0), 0, output + outputIndex);
                    vstore3((uchar3)(output0.s1, output1.s1, output2.s1),
                            0,
                            output + outputIndex + outputHeight * outputWidth);
                    vstore3((uchar3)(output0.s2, output1.s2, output2.s2),
                            0,
                            output + outputIndex + 2 * outputHeight * outputWidth);
                    vstore3((uchar3)(output0.s3, output1.s3, output2.s3),
                            0,
                            output + outputIndex + 3 * outputHeight * outputWidth);
                    vstore3((uchar3)(output0.s4, output1.s4, output2.s4),
                            0,
                            output + outputIndex + 4 * outputHeight * outputWidth);
                    vstore3((uchar3)(output0.s5, output1.s5, output2.s5),
                            0,
                            output + outputIndex + 5 * outputHeight * outputWidth);
                    vstore3((uchar3)(output0.s6, output1.s6, output2.s6),
                            0,
                            output + outputIndex + 6 * outputHeight * outputWidth);
                } else if (outputC + 5 < outputChannel) {
                    vstore3((uchar3)(output0.s0, output1.s0, output2.s0), 0, output + outputIndex);
                    vstore3((uchar3)(output0.s1, output1.s1, output2.s1),
                            0,
                            output + outputIndex + outputHeight * outputWidth);
                    vstore3((uchar3)(output0.s2, output1.s2, output2.s2),
                            0,
                            output + outputIndex + 2 * outputHeight * outputWidth);
                    vstore3((uchar3)(output0.s3, output1.s3, output2.s3),
                            0,
                            output + outputIndex + 3 * outputHeight * outputWidth);
                    vstore3((uchar3)(output0.s4, output1.s4, output2.s4),
                            0,
                            output + outputIndex + 4 * outputHeight * outputWidth);
                    vstore3((uchar3)(output0.s5, output1.s5, output2.s5),
                            0,
                            output + outputIndex + 5 * outputHeight * outputWidth);
                } else if (outputC + 4 < outputChannel) {
                    vstore3((uchar3)(output0.s0, output1.s0, output2.s0), 0, output + outputIndex);
                    vstore3((uchar3)(output0.s1, output1.s1, output2.s1),
                            0,
                            output + outputIndex + outputHeight * outputWidth);
                    vstore3((uchar3)(output0.s2, output1.s2, output2.s2),
                            0,
                            output + outputIndex + 2 * outputHeight * outputWidth);
                    vstore3((uchar3)(output0.s3, output1.s3, output2.s3),
                            0,
                            output + outputIndex + 3 * outputHeight * outputWidth);
                    vstore3((uchar3)(output0.s4, output1.s4, output2.s4),
                            0,
                            output + outputIndex + 4 * outputHeight * outputWidth);
                } else if (outputC + 3 < outputChannel) {
                    vstore3((uchar3)(output0.s0, output1.s0, output2.s0), 0, output + outputIndex);
                    vstore3((uchar3)(output0.s1, output1.s1, output2.s1),
                            0,
                            output + outputIndex + outputHeight * outputWidth);
                    vstore3((uchar3)(output0.s2, output1.s2, output2.s2),
                            0,
                            output + outputIndex + 2 * outputHeight * outputWidth);
                    vstore3((uchar3)(output0.s3, output1.s3, output2.s3),
                            0,
                            output + outputIndex + 3 * outputHeight * outputWidth);
                } else if (outputC + 2 < outputChannel) {
                    vstore3((uchar3)(output0.s0, output1.s0, output2.s0), 0, output + outputIndex);
                    vstore3((uchar3)(output0.s1, output1.s1, output2.s1),
                            0,
                            output + outputIndex + outputHeight * outputWidth);
                    vstore3((uchar3)(output0.s2, output1.s2, output2.s2),
                            0,
                            output + outputIndex + 2 * outputHeight * outputWidth);
                } else if (outputC + 1 < outputChannel) {
                    vstore3((uchar3)(output0.s0, output1.s0, output2.s0), 0, output + outputIndex);
                    vstore3((uchar3)(output0.s1, output1.s1, output2.s1),
                            0,
                            output + outputIndex + outputHeight * outputWidth);
                } else if (outputC < outputChannel) {
                    vstore3((uchar3)(output0.s0, output1.s0, output2.s0), 0, output + outputIndex);
                }
            } else if (outputW + 1 < outputWidth) {
                if (outputC + 7 < outputChannel) {
                    vstore2((uchar2)(output0.s0, output1.s0), 0, output + outputIndex);
                    vstore2((uchar2)(output0.s1, output1.s1),
                            0,
                            output + outputIndex + outputHeight * outputWidth);
                    vstore2((uchar2)(output0.s2, output1.s2),
                            0,
                            output + outputIndex + 2 * outputHeight * outputWidth);
                    vstore2((uchar2)(output0.s3, output1.s3),
                            0,
                            output + outputIndex + 3 * outputHeight * outputWidth);
                    vstore2((uchar2)(output0.s4, output1.s4),
                            0,
                            output + outputIndex + 4 * outputHeight * outputWidth);
                    vstore2((uchar2)(output0.s5, output1.s5),
                            0,
                            output + outputIndex + 5 * outputHeight * outputWidth);
                    vstore2((uchar2)(output0.s6, output1.s6),
                            0,
                            output + outputIndex + 6 * outputHeight * outputWidth);
                    vstore2((uchar2)(output0.s7, output1.s7),
                            0,
                            output + outputIndex + 7 * outputHeight * outputWidth);
                } else if (outputC + 6 < outputChannel) {
                    vstore2((uchar2)(output0.s0, output1.s0), 0, output + outputIndex);
                    vstore2((uchar2)(output0.s1, output1.s1),
                            0,
                            output + outputIndex + outputHeight * outputWidth);
                    vstore2((uchar2)(output0.s2, output1.s2),
                            0,
                            output + outputIndex + 2 * outputHeight * outputWidth);
                    vstore2((uchar2)(output0.s3, output1.s3),
                            0,
                            output + outputIndex + 3 * outputHeight * outputWidth);
                    vstore2((uchar2)(output0.s4, output1.s4),
                            0,
                            output + outputIndex + 4 * outputHeight * outputWidth);
                    vstore2((uchar2)(output0.s5, output1.s5),
                            0,
                            output + outputIndex + 5 * outputHeight * outputWidth);
                    vstore2((uchar2)(output0.s6, output1.s6),
                            0,
                            output + outputIndex + 6 * outputHeight * outputWidth);
                } else if (outputC + 5 < outputChannel) {
                    vstore2((uchar2)(output0.s0, output1.s0), 0, output + outputIndex);
                    vstore2((uchar2)(output0.s1, output1.s1),
                            0,
                            output + outputIndex + outputHeight * outputWidth);
                    vstore2((uchar2)(output0.s2, output1.s2),
                            0,
                            output + outputIndex + 2 * outputHeight * outputWidth);
                    vstore2((uchar2)(output0.s3, output1.s3),
                            0,
                            output + outputIndex + 3 * outputHeight * outputWidth);
                    vstore2((uchar2)(output0.s4, output1.s4),
                            0,
                            output + outputIndex + 4 * outputHeight * outputWidth);
                    vstore2((uchar2)(output0.s5, output1.s5),
                            0,
                            output + outputIndex + 5 * outputHeight * outputWidth);
                } else if (outputC + 4 < outputChannel) {
                    vstore2((uchar2)(output0.s0, output1.s0), 0, output + outputIndex);
                    vstore2((uchar2)(output0.s1, output1.s1),
                            0,
                            output + outputIndex + outputHeight * outputWidth);
                    vstore2((uchar2)(output0.s2, output1.s2),
                            0,
                            output + outputIndex + 2 * outputHeight * outputWidth);
                    vstore2((uchar2)(output0.s3, output1.s3),
                            0,
                            output + outputIndex + 3 * outputHeight * outputWidth);
                    vstore2((uchar2)(output0.s4, output1.s4),
                            0,
                            output + outputIndex + 4 * outputHeight * outputWidth);
                } else if (outputC + 3 < outputChannel) {
                    vstore2((uchar2)(output0.s0, output1.s0), 0, output + outputIndex);
                    vstore2((uchar2)(output0.s1, output1.s1),
                            0,
                            output + outputIndex + outputHeight * outputWidth);
                    vstore2((uchar2)(output0.s2, output1.s2),
                            0,
                            output + outputIndex + 2 * outputHeight * outputWidth);
                    vstore2((uchar2)(output0.s3, output1.s3),
                            0,
                            output + outputIndex + 3 * outputHeight * outputWidth);
                } else if (outputC + 2 < outputChannel) {
                    vstore2((uchar2)(output0.s0, output1.s0), 0, output + outputIndex);
                    vstore2((uchar2)(output0.s1, output1.s1),
                            0,
                            output + outputIndex + outputHeight * outputWidth);
                    vstore2((uchar2)(output0.s2, output1.s2),
                            0,
                            output + outputIndex + 2 * outputHeight * outputWidth);
                } else if (outputC + 1 < outputChannel) {
                    vstore2((uchar2)(output0.s0, output1.s0), 0, output + outputIndex);
                    vstore2((uchar2)(output0.s1, output1.s1),
                            0,
                            output + outputIndex + outputHeight * outputWidth);
                } else if (outputC < outputChannel) {
                    vstore2((uchar2)(output0.s0, output1.s0), 0, output + outputIndex);
                }
            } else if (outputW < outputWidth) {
                if (outputC + 7 < outputChannel) {
                    output[outputIndex] = (uchar)(output0.s0);
                    output[outputIndex + outputHeight * outputWidth] = (uchar)(output0.s1);
                    output[outputIndex + 2 * outputHeight * outputWidth] = (uchar)(output0.s2);
                    output[outputIndex + 3 * outputHeight * outputWidth] = (uchar)(output0.s3);
                    output[outputIndex + 4 * outputHeight * outputWidth] = (uchar)(output0.s4);
                    output[outputIndex + 5 * outputHeight * outputWidth] = (uchar)(output0.s5);
                    output[outputIndex + 6 * outputHeight * outputWidth] = (uchar)(output0.s6);
                    output[outputIndex + 7 * outputHeight * outputWidth] = (uchar)(output0.s7);
                } else if (outputC + 6 < outputChannel) {
                    output[outputIndex] = (uchar)(output0.s0);
                    output[outputIndex + outputHeight * outputWidth] = (uchar)(output0.s1);
                    output[outputIndex + 2 * outputHeight * outputWidth] = (uchar)(output0.s2);
                    output[outputIndex + 3 * outputHeight * outputWidth] = (uchar)(output0.s3);
                    output[outputIndex + 4 * outputHeight * outputWidth] = (uchar)(output0.s4);
                    output[outputIndex + 5 * outputHeight * outputWidth] = (uchar)(output0.s5);
                    output[outputIndex + 6 * outputHeight * outputWidth] = (uchar)(output0.s6);
                } else if (outputC + 5 < outputChannel) {
                    output[outputIndex] = (uchar)(output0.s0);
                    output[outputIndex + outputHeight * outputWidth] = (uchar)(output0.s1);
                    output[outputIndex + 2 * outputHeight * outputWidth] = (uchar)(output0.s2);
                    output[outputIndex + 3 * outputHeight * outputWidth] = (uchar)(output0.s3);
                    output[outputIndex + 4 * outputHeight * outputWidth] = (uchar)(output0.s4);
                    output[outputIndex + 5 * outputHeight * outputWidth] = (uchar)(output0.s5);
                } else if (outputC + 4 < outputChannel) {
                    output[outputIndex] = (uchar)(output0.s0);
                    output[outputIndex + outputHeight * outputWidth] = (uchar)(output0.s1);
                    output[outputIndex + 2 * outputHeight * outputWidth] = (uchar)(output0.s2);
                    output[outputIndex + 3 * outputHeight * outputWidth] = (uchar)(output0.s3);
                    output[outputIndex + 4 * outputHeight * outputWidth] = (uchar)(output0.s4);
                } else if (outputC + 3 < outputChannel) {
                    output[outputIndex] = (uchar)(output0.s0);
                    output[outputIndex + outputHeight * outputWidth] = (uchar)(output0.s1);
                    output[outputIndex + 2 * outputHeight * outputWidth] = (uchar)(output0.s2);
                    output[outputIndex + 3 * outputHeight * outputWidth] = (uchar)(output0.s3);
                } else if (outputC + 2 < outputChannel) {
                    output[outputIndex] = (uchar)(output0.s0);
                    output[outputIndex + outputHeight * outputWidth] = (uchar)(output0.s1);
                    output[outputIndex + 2 * outputHeight * outputWidth] = (uchar)(output0.s2);
                } else if (outputC + 1 < outputChannel) {
                    output[outputIndex] = (uchar)(output0.s0);
                    output[outputIndex + outputHeight * outputWidth] = (uchar)(output0.s1);
                } else if (outputC < outputChannel) {
                    output[outputIndex] = (uchar)(output0.s0);
                }
            }
        }
})

ADD_SINGLE_KERNEL(update_direct_conv_bias_INT8, (__global const unsigned char *weight,
                                               __global int *bias,
                                               int inputZeroPoint,
                                               int filterZeroPoint,
                                               int inputChannel,
                                               int kernelSize) {
        int g0 = get_global_id(0);
        int sum = 0;
        for (int i = 0; i < inputChannel * kernelSize; i++) {
            sum += weight[g0 * inputChannel * kernelSize + i] * inputZeroPoint;
        }
        sum -= inputChannel * kernelSize * inputZeroPoint * filterZeroPoint;
        bias[g0] -= sum;
})

ADD_KERNEL_HEADER(direct5x5_8x4_INT8, {DEFINE_REQUANTIZED_ACT})

ADD_SINGLE_KERNEL(direct5x5_8x4_INT8, (__global const unsigned char *input,
                                     __global const unsigned char *weight,
                                     __global const int *bias,
                                     __global unsigned char *output,
                                     int padT,
                                     int padL,
                                     int outputNumber,
                                     int outputChannel,
                                     int outputHeight,
                                     int outputWidth,
                                     int inputNum,
                                     int inputChannel,
                                     int inputHeight,
                                     int inputWidth,
                                     int inputZeroPoint,
                                     int filterZeroPoint,
                                     int outputOffset,
                                     int output_multiplier,
                                     int output_shift,
                                     int activation_min,
                                     int activation_max) {
        int outputN = get_global_id(2) / ((outputChannel + 7) / 8);
        int outputC = get_global_id(2) % ((outputChannel + 7) / 8) * 8;
        int outputH = get_global_id(1);
        int outputW = get_global_id(0) * 4;

        if (outputW < outputWidth && outputH < outputHeight) {
            int8 output0 = (int8)0;
            int8 output1 = (int8)0;
            int8 output2 = (int8)0;
            int8 output3 = (int8)0;

            int in_0 = 0;
            int in_1 = 0;
            int in_2 = 0;
            int in_3 = 0;

            int inputBase = outputN * inputChannel * inputHeight * inputWidth + outputW * 4;
            int weightBase = outputC * inputChannel * 5 * 5;

            uchar16 intputVector0;
            uchar16 intputVector1;

            uchar16 weightVector0;
            uchar16 weightVector1;
            for (int inputC = 0; inputC < inputChannel / 4; inputC++) {
                int inputIndex = inputBase + inputC * 4 * inputHeight * inputWidth;
                int weightIndex = weightBase + inputC * 5 * 5 * 8 * 4;
                for (int kh = outputH; kh < outputH + 5; kh++) {
                    intputVector0 = vload16(0, input + inputIndex + kh * inputWidth * 4);
                    intputVector1 = vload16(1, input + inputIndex + kh * inputWidth * 4);

                    ARM_DOT(intputVector0.lo.lo, (uchar4)(filterZeroPoint), in_0);
                    ARM_DOT(intputVector0.lo.hi, (uchar4)(filterZeroPoint), in_0);
                    ARM_DOT(intputVector0.hi.lo, (uchar4)(filterZeroPoint), in_0);
                    ARM_DOT(intputVector0.hi.hi, (uchar4)(filterZeroPoint), in_0);
                    ARM_DOT(intputVector1.lo.lo, (uchar4)(filterZeroPoint), in_0);

                    ARM_DOT(intputVector0.lo.hi, (uchar4)(filterZeroPoint), in_1);
                    ARM_DOT(intputVector0.hi.lo, (uchar4)(filterZeroPoint), in_1);
                    ARM_DOT(intputVector0.hi.hi, (uchar4)(filterZeroPoint), in_1);
                    ARM_DOT(intputVector1.lo.lo, (uchar4)(filterZeroPoint), in_1);
                    ARM_DOT(intputVector1.lo.hi, (uchar4)(filterZeroPoint), in_1);

                    ARM_DOT(intputVector0.hi.lo, (uchar4)(filterZeroPoint), in_2);
                    ARM_DOT(intputVector0.hi.hi, (uchar4)(filterZeroPoint), in_2);
                    ARM_DOT(intputVector1.lo.lo, (uchar4)(filterZeroPoint), in_2);
                    ARM_DOT(intputVector1.lo.hi, (uchar4)(filterZeroPoint), in_2);
                    ARM_DOT(intputVector1.hi.lo, (uchar4)(filterZeroPoint), in_2);

                    ARM_DOT(intputVector0.hi.hi, (uchar4)(filterZeroPoint), in_3);
                    ARM_DOT(intputVector1.lo.lo, (uchar4)(filterZeroPoint), in_3);
                    ARM_DOT(intputVector1.lo.hi, (uchar4)(filterZeroPoint), in_3);
                    ARM_DOT(intputVector1.hi.lo, (uchar4)(filterZeroPoint), in_3);
                    ARM_DOT(intputVector1.hi.hi, (uchar4)(filterZeroPoint), in_3);

                    Q_ARM_DOT_SLIDEWINDOW_0TO9(weight, weightIndex, intputVector0, intputVector1, \
                    weightVector0, weightVector1, output0, output1, output2, output3)

                    weightIndex += 160;
                }
            }
            output0 = output0 - in_0;
            output1 = output1 - in_1;
            output2 = output2 - in_2;
            output3 = output3 - in_3;

            int outputIndex = outputN * outputChannel * outputHeight * outputWidth +
                              outputC * outputHeight * outputWidth + outputH * outputWidth +
                              outputW;
            Q_FEED_DIRECT_OUTPUT_8X4(bias, outputC, outputW, outputChannel, outputHeight, outputWidth, output_shift, \
                                     output_multiplier, outputOffset, activation_min, activation_max, output0, \
                                     output1, output2, output3, output, outputIndex)
        }
})

ADD_KERNEL_HEADER(direct7x7_8x4_INT8, {DEFINE_REQUANTIZED_ACT})

ADD_SINGLE_KERNEL(direct7x7_8x4_INT8, (__global const unsigned char *input,
                                     __global const unsigned char *weight,
                                     __global const int *bias,
                                     __global unsigned char *output,
                                     int padT,
                                     int padL,
                                     int outputNumber,
                                     int outputChannel,
                                     int outputHeight,
                                     int outputWidth,
                                     int inputNum,
                                     int inputChannel,
                                     int inputHeight,
                                     int inputWidth,
                                     int inputZeroPoint,
                                     int filterZeroPoint,
                                     int outputOffset,
                                     int output_multiplier,
                                     int output_shift,
                                     int activation_min,
                                     int activation_max) {
        int outputN = get_global_id(2) / ((outputChannel + 7) / 8);
        int outputC = get_global_id(2) % ((outputChannel + 7) / 8) * 8;
        int outputH = get_global_id(1);
        int outputW = get_global_id(0) * 4;

        if (outputW < outputWidth && outputH < outputHeight) {
            int8 output0 = (int8)0;
            int8 output1 = (int8)0;
            int8 output2 = (int8)0;
            int8 output3 = (int8)0;

            int in_0 = 0;
            int in_1 = 0;
            int in_2 = 0;
            int in_3 = 0;

            int inputBase = outputN * inputChannel * inputHeight * inputWidth + outputW * 4;
            int weightBase = outputC * inputChannel * 7 * 7;

            uchar16 intputVector0;
            uchar16 intputVector1;
            uchar16 intputVector2;

            uchar16 weightVector0;
            uchar16 weightVector1;
            for (int inputC = 0; inputC < inputChannel / 4; inputC++) {
                int inputIndex = inputBase + inputC * 4 * inputHeight * inputWidth;
                int weightIndex = weightBase + inputC * 7 * 7 * 8 * 4;
                for (int kh = outputH; kh < outputH + 7; kh++) {
                    intputVector0 = vload16(0, input + inputIndex + kh * inputWidth * 4);
                    intputVector1 = vload16(1, input + inputIndex + kh * inputWidth * 4);
                    intputVector2.lo = vload8(4, input + inputIndex + kh * inputWidth * 4);

                    ARM_DOT(intputVector0.lo.lo, (uchar4)(filterZeroPoint), in_0);
                    ARM_DOT(intputVector0.lo.hi, (uchar4)(filterZeroPoint), in_0);
                    ARM_DOT(intputVector0.hi.lo, (uchar4)(filterZeroPoint), in_0);
                    ARM_DOT(intputVector0.hi.hi, (uchar4)(filterZeroPoint), in_0);
                    ARM_DOT(intputVector1.lo.lo, (uchar4)(filterZeroPoint), in_0);
                    ARM_DOT(intputVector1.lo.hi, (uchar4)(filterZeroPoint), in_0);
                    ARM_DOT(intputVector1.hi.lo, (uchar4)(filterZeroPoint), in_0);

                    ARM_DOT(intputVector0.lo.hi, (uchar4)(filterZeroPoint), in_1);
                    ARM_DOT(intputVector0.hi.lo, (uchar4)(filterZeroPoint), in_1);
                    ARM_DOT(intputVector0.hi.hi, (uchar4)(filterZeroPoint), in_1);
                    ARM_DOT(intputVector1.lo.lo, (uchar4)(filterZeroPoint), in_1);
                    ARM_DOT(intputVector1.lo.hi, (uchar4)(filterZeroPoint), in_1);
                    ARM_DOT(intputVector1.hi.lo, (uchar4)(filterZeroPoint), in_1);
                    ARM_DOT(intputVector1.hi.hi, (uchar4)(filterZeroPoint), in_1);

                    ARM_DOT(intputVector0.hi.lo, (uchar4)(filterZeroPoint), in_2);
                    ARM_DOT(intputVector0.hi.hi, (uchar4)(filterZeroPoint), in_2);
                    ARM_DOT(intputVector1.lo.lo, (uchar4)(filterZeroPoint), in_2);
                    ARM_DOT(intputVector1.lo.hi, (uchar4)(filterZeroPoint), in_2);
                    ARM_DOT(intputVector1.hi.lo, (uchar4)(filterZeroPoint), in_2);
                    ARM_DOT(intputVector1.hi.hi, (uchar4)(filterZeroPoint), in_2);
                    ARM_DOT(intputVector2.lo.lo, (uchar4)(filterZeroPoint), in_2);

                    ARM_DOT(intputVector0.hi.hi, (uchar4)(filterZeroPoint), in_3);
                    ARM_DOT(intputVector1.lo.lo, (uchar4)(filterZeroPoint), in_3);
                    ARM_DOT(intputVector1.lo.hi, (uchar4)(filterZeroPoint), in_3);
                    ARM_DOT(intputVector1.hi.lo, (uchar4)(filterZeroPoint), in_3);
                    ARM_DOT(intputVector1.hi.hi, (uchar4)(filterZeroPoint), in_3);
                    ARM_DOT(intputVector2.lo.lo, (uchar4)(filterZeroPoint), in_3);
                    ARM_DOT(intputVector2.lo.hi, (uchar4)(filterZeroPoint), in_3);

                    Q_ARM_DOT_SLIDEWINDOW_0TO9(weight, weightIndex, intputVector0, intputVector1, \
                                  weightVector0, weightVector1, output0, output1, output2, output3)

                    weightVector0 = vload16(10, weight + weightIndex);
                    weightVector1 = vload16(11, weight + weightIndex);
                    Q_ARM_DOT_I4W16O8(intputVector1.lo.hi, weightVector0, weightVector1, output0)
                    Q_ARM_DOT_I4W16O8(intputVector1.hi.lo, weightVector0, weightVector1, output1)
                    Q_ARM_DOT_I4W16O8(intputVector1.hi.hi, weightVector0, weightVector1, output2)
                    Q_ARM_DOT_I4W16O8(intputVector2.lo.lo, weightVector0, weightVector1, output3)

                    weightVector0 = vload16(12, weight + weightIndex);
                    weightVector1 = vload16(13, weight + weightIndex);
                    Q_ARM_DOT_I4W16O8(intputVector1.hi.lo, weightVector0, weightVector1, output0)
                    Q_ARM_DOT_I4W16O8(intputVector1.hi.hi, weightVector0, weightVector1, output1)
                    Q_ARM_DOT_I4W16O8(intputVector2.lo.lo, weightVector0, weightVector1, output2)
                    Q_ARM_DOT_I4W16O8(intputVector2.lo.hi, weightVector0, weightVector1, output3)

                    weightIndex += 224;
                }
            }
            output0 = output0 - in_0;
            output1 = output1 - in_1;
            output2 = output2 - in_2;
            output3 = output3 - in_3;

            int outputIndex = outputN * outputChannel * outputHeight * outputWidth +
                              outputC * outputHeight * outputWidth + outputH * outputWidth + outputW;
            Q_FEED_DIRECT_OUTPUT_8X4(bias, outputC, outputW, outputChannel, outputHeight, outputWidth, output_shift, \
                                     output_multiplier, outputOffset, activation_min, activation_max, output0, \
                                     output1, output2, output3, output, outputIndex)
        }
})

ADD_KERNEL_HEADER(direct9x9_8x4_INT8, {DEFINE_REQUANTIZED_ACT})

ADD_SINGLE_KERNEL(direct9x9_8x4_INT8, (__global const unsigned char *input,
                                     __global const unsigned char *weight,
                                     __global const int *bias,
                                     __global unsigned char *output,
                                     int padT,
                                     int padL,
                                     int outputNumber,
                                     int outputChannel,
                                     int outputHeight,
                                     int outputWidth,
                                     int inputNum,
                                     int inputChannel,
                                     int inputHeight,
                                     int inputWidth,
                                     int inputZeroPoint,
                                     int filterZeroPoint,
                                     int outputOffset,
                                     int output_multiplier,
                                     int output_shift,
                                     int activation_min,
                                     int activation_max) {
        int outputN = get_global_id(2) / ((outputChannel + 7) / 8);
        int outputC = get_global_id(2) % ((outputChannel + 7) / 8) * 8;
        int outputH = get_global_id(1);
        int outputW = get_global_id(0) * 4;

        if (outputW < outputWidth && outputH < outputHeight) {
            int8 output0 = (int8)0;
            int8 output1 = (int8)0;
            int8 output2 = (int8)0;
            int8 output3 = (int8)0;

            int in_0 = 0;
            int in_1 = 0;
            int in_2 = 0;
            int in_3 = 0;

            int inputBase = outputN * inputChannel * inputHeight * inputWidth + outputW * 4;
            int weightBase = outputC * inputChannel * 9 * 9;

            uchar16 intputVector0;
            uchar16 intputVector1;
            uchar16 intputVector2;
            uchar16 weightVector0;
            uchar16 weightVector1;
            for (int inputC = 0; inputC < inputChannel / 4; inputC++) {
                int inputIndex = inputBase + inputC * 4 * inputHeight * inputWidth;
                int weightIndex = weightBase + inputC * 9 * 9 * 8 * 4;
                for (int kh = outputH; kh < outputH + 9; kh++) {
                    intputVector0 = vload16(0, input + inputIndex + kh * inputWidth * 4);
                    intputVector1 = vload16(1, input + inputIndex + kh * inputWidth * 4);
                    intputVector2 = vload16(2, input + inputIndex + kh * inputWidth * 4);

                    ARM_DOT(intputVector0.lo.lo, (uchar4)(filterZeroPoint), in_0);
                    ARM_DOT(intputVector0.lo.hi, (uchar4)(filterZeroPoint), in_0);
                    ARM_DOT(intputVector0.hi.lo, (uchar4)(filterZeroPoint), in_0);
                    ARM_DOT(intputVector0.hi.hi, (uchar4)(filterZeroPoint), in_0);
                    ARM_DOT(intputVector1.lo.lo, (uchar4)(filterZeroPoint), in_0);
                    ARM_DOT(intputVector1.lo.hi, (uchar4)(filterZeroPoint), in_0);
                    ARM_DOT(intputVector1.hi.lo, (uchar4)(filterZeroPoint), in_0);
                    ARM_DOT(intputVector1.hi.hi, (uchar4)(filterZeroPoint), in_0);
                    ARM_DOT(intputVector2.lo.lo, (uchar4)(filterZeroPoint), in_0);

                    ARM_DOT(intputVector0.lo.hi, (uchar4)(filterZeroPoint), in_1);
                    ARM_DOT(intputVector0.hi.lo, (uchar4)(filterZeroPoint), in_1);
                    ARM_DOT(intputVector0.hi.hi, (uchar4)(filterZeroPoint), in_1);
                    ARM_DOT(intputVector1.lo.lo, (uchar4)(filterZeroPoint), in_1);
                    ARM_DOT(intputVector1.lo.hi, (uchar4)(filterZeroPoint), in_1);
                    ARM_DOT(intputVector1.hi.lo, (uchar4)(filterZeroPoint), in_1);
                    ARM_DOT(intputVector1.hi.hi, (uchar4)(filterZeroPoint), in_1);
                    ARM_DOT(intputVector2.lo.lo, (uchar4)(filterZeroPoint), in_1);
                    ARM_DOT(intputVector2.lo.hi, (uchar4)(filterZeroPoint), in_1);

                    ARM_DOT(intputVector0.hi.lo, (uchar4)(filterZeroPoint), in_2);
                    ARM_DOT(intputVector0.hi.hi, (uchar4)(filterZeroPoint), in_2);
                    ARM_DOT(intputVector1.lo.lo, (uchar4)(filterZeroPoint), in_2);
                    ARM_DOT(intputVector1.lo.hi, (uchar4)(filterZeroPoint), in_2);
                    ARM_DOT(intputVector1.hi.lo, (uchar4)(filterZeroPoint), in_2);
                    ARM_DOT(intputVector1.hi.hi, (uchar4)(filterZeroPoint), in_2);
                    ARM_DOT(intputVector2.lo.lo, (uchar4)(filterZeroPoint), in_2);
                    ARM_DOT(intputVector2.lo.hi, (uchar4)(filterZeroPoint), in_2);
                    ARM_DOT(intputVector2.hi.lo, (uchar4)(filterZeroPoint), in_2);

                    ARM_DOT(intputVector0.hi.hi, (uchar4)(filterZeroPoint), in_3);
                    ARM_DOT(intputVector1.lo.lo, (uchar4)(filterZeroPoint), in_3);
                    ARM_DOT(intputVector1.lo.hi, (uchar4)(filterZeroPoint), in_3);
                    ARM_DOT(intputVector1.hi.lo, (uchar4)(filterZeroPoint), in_3);
                    ARM_DOT(intputVector1.hi.hi, (uchar4)(filterZeroPoint), in_3);
                    ARM_DOT(intputVector2.lo.lo, (uchar4)(filterZeroPoint), in_3);
                    ARM_DOT(intputVector2.lo.hi, (uchar4)(filterZeroPoint), in_3);
                    ARM_DOT(intputVector2.hi.lo, (uchar4)(filterZeroPoint), in_3);
                    ARM_DOT(intputVector2.hi.hi, (uchar4)(filterZeroPoint), in_3);

                    Q_ARM_DOT_SLIDEWINDOW_0TO9(weight, weightIndex, intputVector0, intputVector1, \
                    weightVector0, weightVector1, output0, output1, output2, output3)

                    weightVector0 = vload16(10, weight + weightIndex);
                    weightVector1 = vload16(11, weight + weightIndex);
                    Q_ARM_DOT_I4W16O8(intputVector1.lo.hi, weightVector0, weightVector1, output0)
                    Q_ARM_DOT_I4W16O8(intputVector1.hi.lo, weightVector0, weightVector1, output1)
                    Q_ARM_DOT_I4W16O8(intputVector1.hi.hi, weightVector0, weightVector1, output2)
                    Q_ARM_DOT_I4W16O8(intputVector2.lo.lo, weightVector0, weightVector1, output3)

                    weightVector0 = vload16(12, weight + weightIndex);
                    weightVector1 = vload16(13, weight + weightIndex);
                    Q_ARM_DOT_I4W16O8(intputVector1.hi.lo, weightVector0, weightVector1, output0)
                    Q_ARM_DOT_I4W16O8(intputVector1.hi.hi, weightVector0, weightVector1, output1)
                    Q_ARM_DOT_I4W16O8(intputVector2.lo.lo, weightVector0, weightVector1, output2)
                    Q_ARM_DOT_I4W16O8(intputVector2.lo.hi, weightVector0, weightVector1, output3)

                    weightVector0 = vload16(14, weight + weightIndex);
                    weightVector1 = vload16(15, weight + weightIndex);
                    Q_ARM_DOT_I4W16O8(intputVector1.hi.hi, weightVector0, weightVector1, output0)
                    Q_ARM_DOT_I4W16O8(intputVector2.lo.lo, weightVector0, weightVector1, output1)
                    Q_ARM_DOT_I4W16O8(intputVector2.lo.hi, weightVector0, weightVector1, output2)
                    Q_ARM_DOT_I4W16O8(intputVector2.hi.lo, weightVector0, weightVector1, output3)

                    weightVector0 = vload16(16, weight + weightIndex);
                    weightVector1 = vload16(17, weight + weightIndex);
                    Q_ARM_DOT_I4W16O8(intputVector2.lo.lo, weightVector0, weightVector1, output0)
                    Q_ARM_DOT_I4W16O8(intputVector2.lo.hi, weightVector0, weightVector1, output1)
                    Q_ARM_DOT_I4W16O8(intputVector2.hi.lo, weightVector0, weightVector1, output2)
                    Q_ARM_DOT_I4W16O8(intputVector2.hi.hi, weightVector0, weightVector1, output3)

                    weightIndex += 288;
                }
            }

            output0 = output0 - in_0;
            output1 = output1 - in_1;
            output2 = output2 - in_2;
            output3 = output3 - in_3;

            int outputIndex = outputN * outputChannel * outputHeight * outputWidth +
                              outputC * outputHeight * outputWidth + outputH * outputWidth +
                              outputW;
            Q_FEED_DIRECT_OUTPUT_8X4(bias, outputC, outputW, outputChannel, outputHeight, outputWidth, output_shift, \
                                     output_multiplier, outputOffset, activation_min, activation_max, output0, \
                                     output1, output2, output3, output, outputIndex)
        }
})

#undef CONVERT_TO_DATA_T4
#undef CONVERT_TO_DATA_T3
#undef CONVERT_TO_DATA_T2
#undef CONVERT_TO_DATA_T
#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // uint8
}  // namespace gpu
}  // namespace ud
}  // namespace enn
