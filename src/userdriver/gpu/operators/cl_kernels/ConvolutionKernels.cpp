#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"
namespace enn{
namespace ud {
namespace gpu {

#define ALIGNWEIGHT_GEMM(src, dst, srcWidth, dstWidth, groupTopChannel, alignedGroupTopChannel) \
    uint globalID0 = get_global_id(0); \
    uint globalID1 = get_global_id(1) * 8; \
    uint srcIndex = globalID0 * srcWidth + globalID1; \
    uint dstIndex = globalID0 / groupTopChannel * dstWidth * alignedGroupTopChannel + \
                    globalID0 % groupTopChannel / 4 * 4 * dstWidth + globalID1 * 4 + \
                    globalID0 % groupTopChannel % 4 * 8; \
    if (globalID1 + 7 < srcWidth) { \
        vstore8(vload8(0, src + srcIndex), 0, dst + dstIndex); \
    } else if (globalID1 + 6 < srcWidth) { \
        vstore4(vload4(0, src + srcIndex), 0, dst + dstIndex); \
        vstore3(vload3(0, src + srcIndex + 4), 0, dst + dstIndex + 4); \
    } else if (globalID1 + 5 < srcWidth) { \
        vstore4(vload4(0, src + srcIndex), 0, dst + dstIndex); \
        vstore2(vload2(0, src + srcIndex + 4), 0, dst + dstIndex + 4); \
    } else if (globalID1 + 4 < srcWidth) { \
        vstore4(vload4(0, src + srcIndex), 0, dst + dstIndex); \
        dst[dstIndex + 4] = src[srcIndex + 4]; \
    } else if (globalID1 + 3 < srcWidth) { \
        vstore4(vload4(0, src + srcIndex), 0, dst + dstIndex); \
    } else if (globalID1 + 2 < srcWidth) { \
        vstore3(vload3(0, src + srcIndex), 0, dst + dstIndex); \
    } else if (globalID1 + 1 < srcWidth) { \
        vstore2(vload2(0, src + srcIndex), 0, dst + dstIndex); \
    } else if (globalID1 < srcWidth) { \
        dst[dstIndex] = src[srcIndex]; \
    }

#define GEMMBLOCKED(input, weight, bias, output, alignWidth, outputHeight, outputWidth, groupNum, group) \
    int globalID0 = get_global_id(0); \
    int globalID1 = get_global_id(1) * 4; \
    int globalID2 = get_global_id(2); \
    if (globalID2 < outputWidth) { \
        if (globalID1 + 3 < outputHeight / groupNum) { \
            DATA_T8 out = (DATA_T8)0.0f; \
            DATA_T8 out2 = (DATA_T8)0.0f; \
            DATA_T8 out3 = (DATA_T8)0.0f; \
            DATA_T8 out4 = (DATA_T8)0.0f; \
            DATA_T4 bias4 = vload4(0, bias + group * outputHeight / groupNum + globalID1); \
            { \
                int inputIndex = globalID0 * alignWidth * get_global_size(2) * groupNum + \
                                    alignWidth * get_global_size(2) * group + \
                                    globalID2 * alignWidth; \
                int weightIndex = \
                    group * get_global_size(1) * 4 * alignWidth + globalID1 * alignWidth; \
                for (int i = 0; i<alignWidth>> 3; i++) { \
                    DATA_T8 input8; \
                    DATA_T8 weight8; \
                    input8 = vload8(0, input + inputIndex); \
                    inputIndex += 8; \
                    weight8 = vload8(0, weight + weightIndex); \
                    out += input8 * weight8; \
                    weight8 = vload8(0, weight + weightIndex + 8); \
                    out2 += input8 * weight8; \
                    weight8 = vload8(0, weight + weightIndex + 16); \
                    out3 += input8 * weight8; \
                    weight8 = vload8(0, weight + weightIndex + 24); \
                    out4 += input8 * weight8; \
                    weightIndex += 32; \
                } \
            } \
            { \
                int outputIndex = globalID0 * outputHeight * outputWidth + \
                                    group * outputHeight * outputWidth / groupNum + \
                                    globalID1 * outputWidth + globalID2; \
                output[outputIndex] = \
                    ACT_VEC_F(DATA_T, out.s0 + out.s1 + out.s2 + out.s3 + out.s4 + out.s5 + out.s6 + \
                                out.s7 + bias4.s0); \
                output[outputIndex + outputWidth] = \
                    ACT_VEC_F(DATA_T, out2.s0 + out2.s1 + out2.s2 + out2.s3 + out2.s4 + out2.s5 + out2.s6 + \
                                out2.s7 + bias4.s1); \
                output[outputIndex + 2 * outputWidth] = \
                    ACT_VEC_F(DATA_T, out3.s0 + out3.s1 + out3.s2 + out3.s3 + out3.s4 + out3.s5 + out3.s6 + \
                                out3.s7 + bias4.s2); \
                output[outputIndex + 3 * outputWidth] = \
                    ACT_VEC_F(DATA_T, out4.s0 + out4.s1 + out4.s2 + out4.s3 + out4.s4 + out4.s5 + out4.s6 + \
                                out4.s7 + bias4.s3); \
            } \
        } else if (globalID1 + 2 < outputHeight / groupNum) { \
            DATA_T8 out = (DATA_T8)0.0f; \
            DATA_T8 out2 = (DATA_T8)0.0f; \
            DATA_T8 out3 = (DATA_T8)0.0f; \
            DATA_T3 bias3 = vload3(0, bias + group * outputHeight / groupNum + globalID1); \
            { \
                int inputIndex = globalID0 * alignWidth * get_global_size(2) * groupNum + \
                                    alignWidth * get_global_size(2) * group + \
                                    globalID2 * alignWidth; \
                int weightIndex = \
                    group * get_global_size(1) * 4 * alignWidth + globalID1 * alignWidth; \
                for (int i = 0; i<alignWidth>> 3; i++) { \
                    DATA_T8 input8; \
                    DATA_T8 weight8; \
                    input8 = vload8(0, input + inputIndex); \
                    inputIndex += 8; \
                    weight8 = vload8(0, weight + weightIndex); \
                    out += input8 * weight8; \
                    weight8 = vload8(0, weight + weightIndex + 8); \
                    out2 += input8 * weight8; \
                    weight8 = vload8(0, weight + weightIndex + 16); \
                    out3 += input8 * weight8; \
                    weightIndex += 32; \
                } \
            } \
            { \
                int outputIndex = globalID0 * outputHeight * outputWidth + \
                                    group * outputHeight * outputWidth / groupNum + \
                                    globalID1 * outputWidth + globalID2; \
                output[outputIndex] = \
                    ACT_VEC_F(DATA_T, out.s0 + out.s1 + out.s2 + out.s3 + out.s4 + out.s5 + out.s6 + out.s7 + \
                                bias3.s0); \
                output[outputIndex + outputWidth] = \
                    ACT_VEC_F(DATA_T, out2.s0 + out2.s1 + out2.s2 + out2.s3 + out2.s4 + out2.s5 + out2.s6 + \
                                out2.s7 + bias3.s1); \
                output[outputIndex + 2 * outputWidth] = \
                    ACT_VEC_F(DATA_T, out3.s0 + out3.s1 + out3.s2 + out3.s3 + out3.s4 + out3.s5 + out3.s6 + \
                                out3.s7 + bias3.s2); \
            } \
        } else if (globalID1 + 1 < outputHeight / groupNum) { \
            DATA_T8 out = (DATA_T8)0.0f; \
            DATA_T8 out2 = (DATA_T8)0.0f; \
            DATA_T2 bias2 = vload2(0, bias + group * outputHeight / groupNum + globalID1); \
            { \
                int inputIndex = globalID0 * alignWidth * get_global_size(2) * groupNum + \
                                    alignWidth * get_global_size(2) * group + \
                                    globalID2 * alignWidth; \
                int weightIndex = \
                    group * get_global_size(1) * 4 * alignWidth + globalID1 * alignWidth; \
                for (int i = 0; i<alignWidth>> 3; i++) { \
                    DATA_T8 input8; \
                    DATA_T8 weight8; \
                    input8 = vload8(0, input + inputIndex); \
                    inputIndex += 8; \
                    weight8 = vload8(0, weight + weightIndex); \
                    out += input8 * weight8; \
                    weight8 = vload8(0, weight + weightIndex + 8); \
                    out2 += input8 * weight8; \
                    weightIndex += 32; \
                } \
            } \
            { \
                int outputIndex = globalID0 * outputHeight * outputWidth + \
                                    group * outputHeight * outputWidth / groupNum + \
                                    globalID1 * outputWidth + globalID2; \
                output[outputIndex] = \
                    ACT_VEC_F(DATA_T, out.s0 + out.s1 + out.s2 + out.s3 + out.s4 + out.s5 + out.s6 + out.s7 + \
                                bias2.s0); \
                output[outputIndex + outputWidth] = \
                    ACT_VEC_F(DATA_T, out2.s0 + out2.s1 + out2.s2 + out2.s3 + out2.s4 + out2.s5 + out2.s6 + \
                                out2.s7 + bias2.s1); \
            } \
        } else if (globalID1 < outputHeight / groupNum) { \
            DATA_T8 out = (DATA_T8)0.0f; \
            DATA_T bias1 = bias[group * outputHeight / groupNum + globalID1]; \
            { \
                int inputIndex = globalID0 * alignWidth * get_global_size(2) * groupNum + \
                                    alignWidth * get_global_size(2) * group + \
                                    globalID2 * alignWidth; \
                int weightIndex = \
                    group * get_global_size(1) * 4 * alignWidth + globalID1 * alignWidth; \
                for (int i = 0; i<alignWidth>> 3; i++) { \
                    DATA_T8 input8; \
                    DATA_T8 weight8; \
                    input8 = vload8(0, input + inputIndex); \
                    inputIndex += 8; \
                    weight8 = vload8(0, weight + weightIndex); \
                    out += input8 * weight8; \
                    weightIndex += 32; \
                } \
            } \
            { \
                int outputIndex = globalID0 * outputHeight * outputWidth + \
                                    group * outputHeight * outputWidth / groupNum + \
                                    globalID1 * outputWidth + globalID2; \
                output[outputIndex] = \
                    ACT_VEC_F(DATA_T, out.s0 + out.s1 + out.s2 + out.s3 + out.s4 + out.s5 + out.s6 + out.s7 + \
                                bias1); \
            } \
        } \
    }

#define CONV11_FP16(input, weight, bias, output, inputChannel, outputChannel, width, height, alignedInputChannel) \
    int globalID0 = get_global_id(0); \
    int globalID1 = get_global_id(1) * 2; \
    int globalID2 = get_global_id(2) * 8; \
    int wh = width * height; \
    if (globalID1 + 2 <= outputChannel) { \
        if (globalID2 < wh) { \
            int inputBase = globalID0 * inputChannel * wh + globalID2; \
            int outChannelIndex_1 = globalID1; \
            int outChannelIndex_2 = outChannelIndex_1 + 1; \
            int outputIndex_1 = \
                globalID0 * outputChannel * wh + outChannelIndex_1 * wh + globalID2; \
            int outputIndex_2 = outputIndex_1 + wh; \
            int weightBase_1 = outChannelIndex_1 * alignedInputChannel; \
            int weightBase_2 = weightBase_1 + alignedInputChannel; \
            int inputIndex = 0; \
            DATA_T8 input8 = (DATA_T8)(0.0f); \
            int weightIndex_1 = 0; \
            int weightIndex_2 = 0; \
            DATA_T8 weight8_1 = (DATA_T8)(0.0f); \
            DATA_T8 weight8_2 = (DATA_T8)(0.0f); \
            DATA_T8 res8_1 = (DATA_T8)(0.0f); \
            DATA_T8 res8_2 = (DATA_T8)(0.0f); \
            for (int i = 0; i < inputChannel / 8; i++) { \
                int split = i * 8; \
                weightIndex_1 = weightBase_1 + split; \
                weightIndex_2 = weightBase_2 + split; \
                weight8_1 = vload8(0, weight + weightIndex_1); \
                weight8_2 = vload8(0, weight + weightIndex_2); \
                inputIndex = inputBase + split * wh; \
                input8 = vload8(0, input + inputIndex); \
                res8_1 = res8_1 + input8 * weight8_1.s0; \
                res8_2 = res8_2 + input8 * weight8_2.s0; \
                inputIndex = inputIndex + wh; \
                input8 = vload8(0, input + inputIndex); \
                res8_1 = res8_1 + input8 * weight8_1.s1; \
                res8_2 = res8_2 + input8 * weight8_2.s1; \
                inputIndex = inputIndex + wh; \
                input8 = vload8(0, input + inputIndex); \
                res8_1 = res8_1 + input8 * weight8_1.s2; \
                res8_2 = res8_2 + input8 * weight8_2.s2; \
                inputIndex = inputIndex + wh; \
                input8 = vload8(0, input + inputIndex); \
                res8_1 = res8_1 + input8 * weight8_1.s3; \
                res8_2 = res8_2 + input8 * weight8_2.s3; \
                inputIndex = inputIndex + wh; \
                input8 = vload8(0, input + inputIndex); \
                res8_1 = res8_1 + input8 * weight8_1.s4; \
                res8_2 = res8_2 + input8 * weight8_2.s4; \
                inputIndex = inputIndex + wh; \
                input8 = vload8(0, input + inputIndex); \
                res8_1 = res8_1 + input8 * weight8_1.s5; \
                res8_2 = res8_2 + input8 * weight8_2.s5; \
                inputIndex = inputIndex + wh; \
                input8 = vload8(0, input + inputIndex); \
                res8_1 = res8_1 + input8 * weight8_1.s6; \
                res8_2 = res8_2 + input8 * weight8_2.s6; \
                inputIndex = inputIndex + wh; \
                input8 = vload8(0, input + inputIndex); \
                res8_1 = res8_1 + input8 * weight8_1.s7; \
                res8_2 = res8_2 + input8 * weight8_2.s7; \
            } \
            for (int i = inputChannel / 8 * 8; i < inputChannel; i++) { \
                weightIndex_1 = weightBase_1 + i; \
                weightIndex_2 = weightBase_2 + i; \
                DATA_T w_1 = weight[weightIndex_1]; \
                DATA_T w_2 = weight[weightIndex_2]; \
                inputIndex = inputBase + i * wh; \
                input8 = vload8(0, input + inputIndex); \
                res8_1 = res8_1 + input8 * w_1; \
                res8_2 = res8_2 + input8 * w_2; \
            } \
            res8_1 = res8_1 + bias[outChannelIndex_1]; \
            res8_2 = res8_2 + bias[outChannelIndex_2]; \
            if (globalID2 + 8 <= wh) { \
                vstore8(ACT_VEC_F(DATA_T8, res8_1), 0, output + outputIndex_1); \
                vstore8(ACT_VEC_F(DATA_T8, res8_2), 0, output + outputIndex_2); \
            } else { \
                int num = wh - globalID2; \
                if (num == 1) { \
                    output[outputIndex_1] = ACT_VEC_F(DATA_T, res8_1.s0); \
                    output[outputIndex_2] = ACT_VEC_F(DATA_T, res8_2.s0); \
                } else if (num == 2) { \
                    vstore2(ACT_VEC_F(DATA_T2, res8_1.s01), 0, output + outputIndex_1); \
                    vstore2(ACT_VEC_F(DATA_T2, res8_2.s01), 0, output + outputIndex_2); \
                } else if (num == 3) { \
                    vstore3(ACT_VEC_F(DATA_T3, res8_1.s012), 0, output + outputIndex_1); \
                    vstore3(ACT_VEC_F(DATA_T3, res8_2.s012), 0, output + outputIndex_2); \
                } else if (num == 4) { \
                    vstore4(ACT_VEC_F(DATA_T4, res8_1.s0123), 0, output + outputIndex_1); \
                    vstore4(ACT_VEC_F(DATA_T4, res8_2.s0123), 0, output + outputIndex_2); \
                } else if (num == 5) { \
                    vstore4(ACT_VEC_F(DATA_T4, res8_1.s0123), 0, output + outputIndex_1); \
                    output[outputIndex_1 + 4] = ACT_VEC_F(DATA_T, res8_1.s4); \
                    vstore4(ACT_VEC_F(DATA_T4, res8_2.s0123), 0, output + outputIndex_2); \
                    output[outputIndex_2 + 4] = ACT_VEC_F(DATA_T, res8_2.s4); \
                } else if (num == 6) { \
                    vstore4(ACT_VEC_F(DATA_T4, res8_1.s0123), 0, output + outputIndex_1); \
                    vstore2(ACT_VEC_F(DATA_T2, res8_1.s45), 0, output + outputIndex_1 + 4); \
                    vstore4(ACT_VEC_F(DATA_T4, res8_2.s0123), 0, output + outputIndex_2); \
                    vstore2(ACT_VEC_F(DATA_T2, res8_2.s45), 0, output + outputIndex_2 + 4); \
                } else if (num == 7) { \
                    vstore4(ACT_VEC_F(DATA_T4, res8_1.s0123), 0, output + outputIndex_1); \
                    vstore3(ACT_VEC_F(DATA_T3, res8_1.s456), 0, output + outputIndex_1 + 4); \
                    vstore4(ACT_VEC_F(DATA_T4, res8_2.s0123), 0, output + outputIndex_2); \
                    vstore3(ACT_VEC_F(DATA_T3, res8_2.s456), 0, output + outputIndex_2 + 4); \
                } \
            } \
        } \
    } else if (globalID1 < outputChannel) { \
        if (globalID2 < wh) { \
            int inputBase = globalID0 * inputChannel * wh + globalID2; \
            int outChannelIndex_1 = globalID1; \
            int outputIndex_1 = \
                globalID0 * outputChannel * wh + outChannelIndex_1 * wh + globalID2; \
            int weightBase_1 = outChannelIndex_1 * alignedInputChannel; \
            int inputIndex = 0; \
            DATA_T8 input8 = (DATA_T8)(0.0f); \
            int weightIndex_1 = 0; \
            DATA_T8 weight8_1 = (DATA_T8)(0.0f); \
            DATA_T8 res8_1 = (DATA_T8)(0.0f); \
            for (int i = 0; i < inputChannel / 8; i++) { \
                int split = i * 8; \
                weightIndex_1 = weightBase_1 + split; \
                weight8_1 = vload8(0, weight + weightIndex_1); \
                inputIndex = inputBase + split * wh; \
                input8 = vload8(0, input + inputIndex); \
                res8_1 = res8_1 + input8 * weight8_1.s0; \
                inputIndex = inputIndex + wh; \
                input8 = vload8(0, input + inputIndex); \
                res8_1 = res8_1 + input8 * weight8_1.s1; \
                inputIndex = inputIndex + wh; \
                input8 = vload8(0, input + inputIndex); \
                res8_1 = res8_1 + input8 * weight8_1.s2; \
                inputIndex = inputIndex + wh; \
                input8 = vload8(0, input + inputIndex); \
                res8_1 = res8_1 + input8 * weight8_1.s3; \
                inputIndex = inputIndex + wh; \
                input8 = vload8(0, input + inputIndex); \
                res8_1 = res8_1 + input8 * weight8_1.s4; \
                inputIndex = inputIndex + wh; \
                input8 = vload8(0, input + inputIndex); \
                res8_1 = res8_1 + input8 * weight8_1.s5; \
                inputIndex = inputIndex + wh; \
                input8 = vload8(0, input + inputIndex); \
                res8_1 = res8_1 + input8 * weight8_1.s6; \
                inputIndex = inputIndex + wh; \
                input8 = vload8(0, input + inputIndex); \
                res8_1 = res8_1 + input8 * weight8_1.s7; \
            } \
            for (int i = inputChannel / 8 * 8; i < inputChannel; i++) { \
                weightIndex_1 = weightBase_1 + i; \
                DATA_T w_1 = weight[weightIndex_1]; \
                inputIndex = inputBase + i * wh; \
                input8 = vload8(0, input + inputIndex); \
                res8_1 = res8_1 + input8 * w_1; \
            } \
            res8_1 = res8_1 + bias[outChannelIndex_1]; \
            if (globalID2 + 8 <= wh) { \
                vstore8(ACT_VEC_F(DATA_T8, res8_1), 0, output + outputIndex_1); \
            } else { \
                int num = wh - globalID2; \
                if (num == 1) { \
                    output[outputIndex_1] = ACT_VEC_F(DATA_T, res8_1.s0); \
                } else if (num == 2) { \
                    vstore2(ACT_VEC_F(DATA_T2, res8_1.s01), 0, output + outputIndex_1); \
                } else if (num == 3) { \
                    vstore3(ACT_VEC_F(DATA_T3, res8_1.s012), 0, output + outputIndex_1); \
                } else if (num == 4) { \
                    vstore4(ACT_VEC_F(DATA_T4, res8_1.s0123), 0, output + outputIndex_1); \
                } else if (num == 5) { \
                    vstore4(ACT_VEC_F(DATA_T4, res8_1.s0123), 0, output + outputIndex_1); \
                    output[outputIndex_1 + 4] = ACT_VEC_F(DATA_T, res8_1.s4); \
                } else if (num == 6) { \
                    vstore4(ACT_VEC_F(DATA_T4, res8_1.s0123), 0, output + outputIndex_1); \
                    vstore2(ACT_VEC_F(DATA_T2, res8_1.s45), 0, output + outputIndex_1 + 4); \
                } else if (num == 7) { \
                    vstore4(ACT_VEC_F(DATA_T4, res8_1.s0123), 0, output + outputIndex_1); \
                    vstore3(ACT_VEC_F(DATA_T3, res8_1.s456), 0, output + outputIndex_1 + 4); \
                } \
            } \
        } \
    }

#define CONV11_FP32(input, weight, bias, output, inputChannel, outputChannel, width, height, alignedInputChannel) \
    int globalID0 = get_global_id(0); \
    int globalID1 = get_global_id(1) * 2; \
    int globalID2 = get_global_id(2) * 4; \
    int wh = width * height; \
    if (globalID1 + 2 <= outputChannel) { \
        if (globalID2 < wh) { \
            int inputBase = globalID0 * inputChannel * wh + globalID2; \
            int outChannelIndex_1 = globalID1; \
            int outChannelIndex_2 = outChannelIndex_1 + 1; \
            int outputIndex_1 = \
                globalID0 * outputChannel * wh + outChannelIndex_1 * wh + globalID2; \
            int outputIndex_2 = outputIndex_1 + wh; \
            int weightBase_1 = outChannelIndex_1 * alignedInputChannel; \
            int weightBase_2 = weightBase_1 + alignedInputChannel; \
            int inputIndex = 0; \
            DATA_T4 input4 = (DATA_T4)(0.0f); \
            int weightIndex_1 = 0; \
            int weightIndex_2 = 0; \
            DATA_T4 weight4_1 = (DATA_T4)(0.0f); \
            DATA_T4 weight4_2 = (DATA_T4)(0.0f); \
            DATA_T4 res4_1 = (DATA_T4)(0.0f); \
            DATA_T4 res4_2 = (DATA_T4)(0.0f); \
            for (int i = 0; i < inputChannel / 4; i++) { \
                int split = i * 4; \
                weightIndex_1 = weightBase_1 + split; \
                weightIndex_2 = weightBase_2 + split; \
                weight4_1 = vload4(0, weight + weightIndex_1); \
                weight4_2 = vload4(0, weight + weightIndex_2); \
                inputIndex = inputBase + split * wh; \
                input4 = vload4(0, input + inputIndex); \
                res4_1 = res4_1 + input4 * weight4_1.s0; \
                res4_2 = res4_2 + input4 * weight4_2.s0; \
                inputIndex = inputIndex + wh; \
                input4 = vload4(0, input + inputIndex); \
                res4_1 = res4_1 + input4 * weight4_1.s1; \
                res4_2 = res4_2 + input4 * weight4_2.s1; \
                inputIndex = inputIndex + wh; \
                input4 = vload4(0, input + inputIndex); \
                res4_1 = res4_1 + input4 * weight4_1.s2; \
                res4_2 = res4_2 + input4 * weight4_2.s2; \
                inputIndex = inputIndex + wh; \
                input4 = vload4(0, input + inputIndex); \
                res4_1 = res4_1 + input4 * weight4_1.s3; \
                res4_2 = res4_2 + input4 * weight4_2.s3; \
            } \
            for (int i = inputChannel / 4 * 4; i < inputChannel; i++) { \
                weightIndex_1 = weightBase_1 + i; \
                weightIndex_2 = weightBase_2 + i; \
                DATA_T w_1 = weight[weightIndex_1]; \
                DATA_T w_2 = weight[weightIndex_2]; \
                inputIndex = inputBase + i * wh; \
                input4 = vload4(0, input + inputIndex); \
                res4_1 = res4_1 + input4 * w_1; \
                res4_2 = res4_2 + input4 * w_2; \
            } \
            res4_1 = res4_1 + bias[outChannelIndex_1]; \
            res4_2 = res4_2 + bias[outChannelIndex_2]; \
            if (globalID2 + 4 <= wh) { \
                vstore4(ACT_VEC_F(DATA_T4, res4_1), 0, output + outputIndex_1); \
                vstore4(ACT_VEC_F(DATA_T4, res4_2), 0, output + outputIndex_2); \
            } else { \
                int num = wh - globalID2; \
                if (num == 1) { \
                    output[outputIndex_1] = ACT_VEC_F(DATA_T, res4_1.s0); \
                    output[outputIndex_2] = ACT_VEC_F(DATA_T, res4_2.s0); \
                } else if (num == 2) { \
                    vstore2(ACT_VEC_F(DATA_T2, res4_1.s01), 0, output + outputIndex_1); \
                    vstore2(ACT_VEC_F(DATA_T2, res4_2.s01), 0, output + outputIndex_2); \
                } else if (num == 3) { \
                    vstore3(ACT_VEC_F(DATA_T3, res4_1.s012), 0, output + outputIndex_1); \
                    vstore3(ACT_VEC_F(DATA_T3, res4_2.s012), 0, output + outputIndex_2); \
                } \
            } \
        } \
    } else if (globalID1 < outputChannel) { \
        if (globalID2 < wh) { \
            int inputBase = globalID0 * inputChannel * wh + globalID2; \
            int outChannelIndex_1 = globalID1; \
            int outputIndex_1 = \
                globalID0 * outputChannel * wh + outChannelIndex_1 * wh + globalID2; \
            int weightBase_1 = outChannelIndex_1 * alignedInputChannel; \
            int inputIndex = 0; \
            DATA_T4 input4 = (DATA_T4)(0.0f); \
            int weightIndex_1 = 0; \
            DATA_T4 weight4_1 = (DATA_T4)(0.0f); \
            DATA_T4 res4_1 = (DATA_T4)(0.0f); \
            for (int i = 0; i < inputChannel / 4; i++) { \
                int split = i * 4; \
                weightIndex_1 = weightBase_1 + split; \
                weight4_1 = vload4(0, weight + weightIndex_1); \
                inputIndex = inputBase + split * wh; \
                input4 = vload4(0, input + inputIndex); \
                res4_1 = res4_1 + input4 * weight4_1.s0; \
                inputIndex = inputIndex + wh; \
                input4 = vload4(0, input + inputIndex); \
                res4_1 = res4_1 + input4 * weight4_1.s1; \
                inputIndex = inputIndex + wh; \
                input4 = vload4(0, input + inputIndex); \
                res4_1 = res4_1 + input4 * weight4_1.s2; \
                inputIndex = inputIndex + wh; \
                input4 = vload4(0, input + inputIndex); \
                res4_1 = res4_1 + input4 * weight4_1.s3; \
            } \
            for (int i = inputChannel / 4 * 4; i < inputChannel; i++) { \
                weightIndex_1 = weightBase_1 + i; \
                DATA_T w_1 = weight[weightIndex_1]; \
                inputIndex = inputBase + i * wh; \
                input4 = vload4(0, input + inputIndex); \
                res4_1 = res4_1 + input4 * w_1; \
            } \
            res4_1 = res4_1 + bias[outChannelIndex_1]; \
            if (globalID2 + 4 <= wh) { \
                vstore4(ACT_VEC_F(DATA_T4, res4_1), 0, output + outputIndex_1); \
            } else { \
                int num = wh - globalID2; \
                if (num == 1) { \
                    output[outputIndex_1] = ACT_VEC_F(DATA_T, res4_1.s0); \
                } else if (num == 2) { \
                    vstore2(ACT_VEC_F(DATA_T2, res4_1.s01), 0, output + outputIndex_1); \
                } else if (num == 3) { \
                    vstore3(ACT_VEC_F(DATA_T3, res4_1.s012), 0, output + outputIndex_1); \
                } \
            } \
        } \
    }

#define CONV11_STRIDE2(input, weight, bias, output, inputChannel, outputChannel, inputWidth, inputHeight, \
                       outputWidth, outputHeight, alignedInputChannel) \
    int globalID0 = get_global_id(0); \
    int globalID1 = get_global_id(1) * 2; \
    int topHIndex = get_global_id(2) / ((outputWidth + 3) / 4); \
    int topWIndex = (get_global_id(2) % ((outputWidth + 3) / 4)) * 4; \
    int outputWh = outputWidth * outputHeight; \
    int inputWh = inputWidth * inputHeight; \
    if (globalID1 + 2 <= outputChannel) { \
        if (topWIndex < outputWidth && topHIndex < outputHeight) { \
            int inputBase = \
                globalID0 * inputChannel * inputWh + topHIndex * 2 * inputWidth + topWIndex * 2; \
            int outChannelIndex_1 = globalID1; \
            int outChannelIndex_2 = outChannelIndex_1 + 1; \
            int outputIndex_1 = globalID0 * outputChannel * outputWh + \
                                outChannelIndex_1 * outputWh + topHIndex * outputWidth + \
                                topWIndex; \
            int outputIndex_2 = outputIndex_1 + outputWh; \
            int weightBase_1 = outChannelIndex_1 * alignedInputChannel; \
            int weightBase_2 = weightBase_1 + alignedInputChannel; \
            int inputIndex = 0; \
            DATA_T8 input8 = (DATA_T8)(0.0f); \
            int weightIndex_1 = 0; \
            int weightIndex_2 = 0; \
            DATA_T8 weight8_1 = (DATA_T8)(0.0f); \
            DATA_T8 weight8_2 = (DATA_T8)(0.0f); \
            DATA_T4 res4_1 = (DATA_T4)(0.0f); \
            DATA_T4 res4_2 = (DATA_T4)(0.0f); \
            for (int i = 0; i < inputChannel / 8; i++) { \
                int split = i * 8; \
                weightIndex_1 = weightBase_1 + split; \
                weightIndex_2 = weightBase_2 + split; \
                weight8_1 = vload8(0, weight + weightIndex_1); \
                weight8_2 = vload8(0, weight + weightIndex_2); \
                inputIndex = inputBase + split * inputWh; \
                input8 = vload8(0, input + inputIndex); \
                res4_1 = res4_1 + input8.s0246 * weight8_1.s0; \
                res4_2 = res4_2 + input8.s0246 * weight8_2.s0; \
                inputIndex = inputIndex + inputWh; \
                input8 = vload8(0, input + inputIndex); \
                res4_1 = res4_1 + input8.s0246 * weight8_1.s1; \
                res4_2 = res4_2 + input8.s0246 * weight8_2.s1; \
                inputIndex = inputIndex + inputWh; \
                input8 = vload8(0, input + inputIndex); \
                res4_1 = res4_1 + input8.s0246 * weight8_1.s2; \
                res4_2 = res4_2 + input8.s0246 * weight8_2.s2; \
                inputIndex = inputIndex + inputWh; \
                input8 = vload8(0, input + inputIndex); \
                res4_1 = res4_1 + input8.s0246 * weight8_1.s3; \
                res4_2 = res4_2 + input8.s0246 * weight8_2.s3; \
                inputIndex = inputIndex + inputWh; \
                input8 = vload8(0, input + inputIndex); \
                res4_1 = res4_1 + input8.s0246 * weight8_1.s4; \
                res4_2 = res4_2 + input8.s0246 * weight8_2.s4; \
                inputIndex = inputIndex + inputWh; \
                input8 = vload8(0, input + inputIndex); \
                res4_1 = res4_1 + input8.s0246 * weight8_1.s5; \
                res4_2 = res4_2 + input8.s0246 * weight8_2.s5; \
                inputIndex = inputIndex + inputWh; \
                input8 = vload8(0, input + inputIndex); \
                res4_1 = res4_1 + input8.s0246 * weight8_1.s6; \
                res4_2 = res4_2 + input8.s0246 * weight8_2.s6; \
                inputIndex = inputIndex + inputWh; \
                input8 = vload8(0, input + inputIndex); \
                res4_1 = res4_1 + input8.s0246 * weight8_1.s7; \
                res4_2 = res4_2 + input8.s0246 * weight8_2.s7; \
            } \
            for (int i = inputChannel / 8 * 8; i < inputChannel; i++) { \
                weightIndex_1 = weightBase_1 + i; \
                weightIndex_2 = weightBase_2 + i; \
                DATA_T w_1 = weight[weightIndex_1]; \
                DATA_T w_2 = weight[weightIndex_2]; \
                inputIndex = inputBase + i * inputWh; \
                input8 = vload8(0, input + inputIndex); \
                res4_1 = res4_1 + input8.s0246 * w_1; \
                res4_2 = res4_2 + input8.s0246 * w_2; \
            } \
            res4_1 = res4_1 + bias[outChannelIndex_1]; \
            res4_2 = res4_2 + bias[outChannelIndex_2]; \
            if (topWIndex + 4 <= outputWidth) { \
                vstore4(ACT_VEC_F(DATA_T4, res4_1), 0, output + outputIndex_1); \
                vstore4(ACT_VEC_F(DATA_T4, res4_2), 0, output + outputIndex_2); \
            } else { \
                int num = outputWidth - topWIndex; \
                if (num == 1) { \
                    output[outputIndex_1] = ACT_VEC_F(DATA_T, res4_1.s0); \
                    output[outputIndex_2] = ACT_VEC_F(DATA_T, res4_2.s0); \
                } else if (num == 2) { \
                    vstore2(ACT_VEC_F(DATA_T2, res4_1.s01), 0, output + outputIndex_1); \
                    vstore2(ACT_VEC_F(DATA_T2, res4_2.s01), 0, output + outputIndex_2); \
                } else if (num == 3) { \
                    vstore3(ACT_VEC_F(DATA_T3, res4_1.s012), 0, output + outputIndex_1); \
                    vstore3(ACT_VEC_F(DATA_T3, res4_2.s012), 0, output + outputIndex_2); \
                } \
            } \
        } \
    } else if (globalID1 < outputChannel) { \
        if (topWIndex < outputWidth && topHIndex < outputHeight) { \
            int inputBase = \
                globalID0 * inputChannel * inputWh + topHIndex * 2 * inputWidth + topWIndex * 2; \
            int outChannelIndex_1 = globalID1; \
            int outputIndex_1 = globalID0 * outputChannel * outputWh + \
                                outChannelIndex_1 * outputWh + topHIndex * outputWidth + \
                                topWIndex; \
            int weightBase_1 = outChannelIndex_1 * alignedInputChannel; \
            int inputIndex = 0; \
            DATA_T8 input8 = (DATA_T8)(0.0f); \
            int weightIndex_1 = 0; \
            DATA_T8 weight8_1 = (DATA_T8)(0.0f); \
            DATA_T4 res4_1 = (DATA_T4)(0.0f); \
            for (int i = 0; i < inputChannel / 8; i++) { \
                int split = i * 8; \
                weightIndex_1 = weightBase_1 + split; \
                weight8_1 = vload8(0, weight + weightIndex_1); \
                inputIndex = inputBase + split * inputWh; \
                input8 = vload8(0, input + inputIndex); \
                res4_1 = res4_1 + input8.s0246 * weight8_1.s0; \
                inputIndex = inputIndex + inputWh; \
                input8 = vload8(0, input + inputIndex); \
                res4_1 = res4_1 + input8.s0246 * weight8_1.s1; \
                inputIndex = inputIndex + inputWh; \
                input8 = vload8(0, input + inputIndex); \
                res4_1 = res4_1 + input8.s0246 * weight8_1.s2; \
                inputIndex = inputIndex + inputWh; \
                input8 = vload8(0, input + inputIndex); \
                res4_1 = res4_1 + input8.s0246 * weight8_1.s3; \
                inputIndex = inputIndex + inputWh; \
                input8 = vload8(0, input + inputIndex); \
                res4_1 = res4_1 + input8.s0246 * weight8_1.s4; \
                inputIndex = inputIndex + inputWh; \
                input8 = vload8(0, input + inputIndex); \
                res4_1 = res4_1 + input8.s0246 * weight8_1.s5; \
                inputIndex = inputIndex + inputWh; \
                input8 = vload8(0, input + inputIndex); \
                res4_1 = res4_1 + input8.s0246 * weight8_1.s6; \
                inputIndex = inputIndex + inputWh; \
                input8 = vload8(0, input + inputIndex); \
                res4_1 = res4_1 + input8.s0246 * weight8_1.s7; \
            } \
            for (int i = inputChannel / 8 * 8; i < inputChannel; i++) { \
                weightIndex_1 = weightBase_1 + i; \
                DATA_T w_1 = weight[weightIndex_1]; \
                inputIndex = inputBase + i * inputWh; \
                input8 = vload8(0, input + inputIndex); \
                res4_1 = res4_1 + input8.s0246 * w_1; \
            } \
            res4_1 = res4_1 + bias[outChannelIndex_1]; \
            if (topWIndex + 4 <= outputWidth) { \
                vstore4(ACT_VEC_F(DATA_T4, res4_1), 0, output + outputIndex_1); \
            } else { \
                int num = outputWidth - topWIndex; \
                if (num == 1) { \
                    output[outputIndex_1] = ACT_VEC_F(DATA_T, res4_1.s0); \
                } else if (num == 2) { \
                    vstore2(ACT_VEC_F(DATA_T2, res4_1.s01), 0, output + outputIndex_1); \
                } else if (num == 3) { \
                    vstore3(ACT_VEC_F(DATA_T3, res4_1.s012), 0, output + outputIndex_1); \
                } \
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
ADD_SINGLE_KERNEL(gemmBlocked_FP16, (__global const DATA_T *input,
                                     __global const DATA_T *weight,
                                     __global const DATA_T *bias,
                                     __global DATA_T *output,
                                     unsigned int alignWidth,
                                     unsigned int outputHeight,
                                     unsigned int outputWidth,
                                     unsigned int groupNum,
                                     unsigned int group) {
    GEMMBLOCKED(input, weight, bias, output, alignWidth, outputHeight, outputWidth, groupNum, group)
})

ADD_SINGLE_KERNEL(conv11_FP16, (__global const DATA_T *input,
                                __global const DATA_T *weight,
                                __global const DATA_T *bias,
                                __global DATA_T *output,
                                unsigned int inputChannel,
                                unsigned int outputChannel,
                                unsigned int width,
                                unsigned int height,
                                unsigned int alignedInputChannel) {
    CONV11_FP16(input, weight, bias, output, inputChannel, outputChannel, width, height, alignedInputChannel)
})

ADD_SINGLE_KERNEL(conv11_stride2_FP16, (__global const DATA_T *input,
                                        __global const DATA_T *weight,
                                        __global const DATA_T *bias,
                                        __global DATA_T *output,
                                        unsigned int inputChannel,
                                        unsigned int outputChannel,
                                        unsigned int inputWidth,
                                        unsigned int inputHeight,
                                        unsigned int outputWidth,
                                        unsigned int outputHeight,
                                        unsigned int alignedInputChannel) {
    CONV11_STRIDE2(input, weight, bias, output, inputChannel, outputChannel, inputWidth, inputHeight, \
                   outputWidth, outputHeight, alignedInputChannel)
})
#undef ACT_VEC_F  // NONE

// apply RELU to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELUgemmBlocked_FP16, (__global const DATA_T *input,
                                         __global const DATA_T *weight,
                                         __global const DATA_T *bias,
                                         __global DATA_T *output,
                                         unsigned int alignWidth,
                                         unsigned int outputHeight,
                                         unsigned int outputWidth,
                                         unsigned int groupNum,
                                         unsigned int group) {
    GEMMBLOCKED(input, weight, bias, output, alignWidth, outputHeight, outputWidth, groupNum, group)
})

ADD_SINGLE_KERNEL(RELUconv11_FP16, (__global const DATA_T *input,
                                    __global const DATA_T *weight,
                                    __global const DATA_T *bias,
                                    __global DATA_T *output,
                                    unsigned int inputChannel,
                                    unsigned int outputChannel,
                                    unsigned int width,
                                    unsigned int height,
                                    unsigned int alignedInputChannel) {
    CONV11_FP16(input, weight, bias, output, inputChannel, outputChannel, width, height, alignedInputChannel)
})

ADD_SINGLE_KERNEL(RELUconv11_stride2_FP16, (__global const DATA_T *input,
                                            __global const DATA_T *weight,
                                            __global const DATA_T *bias,
                                            __global DATA_T *output,
                                            unsigned int inputChannel,
                                            unsigned int outputChannel,
                                            unsigned int inputWidth,
                                            unsigned int inputHeight,
                                            unsigned int outputWidth,
                                            unsigned int outputHeight,
                                            unsigned int alignedInputChannel) {
    CONV11_STRIDE2(input, weight, bias, output, inputChannel, outputChannel, inputWidth, inputHeight, \
                   outputWidth, outputHeight, alignedInputChannel)
})
#undef ACT_VEC_F  // RELU

// apply RELU6 to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU6_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELU6gemmBlocked_FP16, (__global const DATA_T *input,
                                          __global const DATA_T *weight,
                                          __global const DATA_T *bias,
                                          __global DATA_T *output,
                                          unsigned int alignWidth,
                                          unsigned int outputHeight,
                                          unsigned int outputWidth,
                                          unsigned int groupNum,
                                          unsigned int group) {
    GEMMBLOCKED(input, weight, bias, output, alignWidth, outputHeight, outputWidth, groupNum, group)
})

ADD_SINGLE_KERNEL(RELU6conv11_FP16, (__global const DATA_T *input,
                                     __global const DATA_T *weight,
                                     __global const DATA_T *bias,
                                     __global DATA_T *output,
                                     unsigned int inputChannel,
                                     unsigned int outputChannel,
                                     unsigned int width,
                                     unsigned int height,
                                     unsigned int alignedInputChannel) {
    CONV11_FP16(input, weight, bias, output, inputChannel, outputChannel, width, height, alignedInputChannel)
})

ADD_SINGLE_KERNEL(RELU6conv11_stride2_FP16, (__global const DATA_T *input,
                                             __global const DATA_T *weight,
                                             __global const DATA_T *bias,
                                             __global DATA_T *output,
                                             unsigned int inputChannel,
                                             unsigned int outputChannel,
                                             unsigned int inputWidth,
                                             unsigned int inputHeight,
                                             unsigned int outputWidth,
                                             unsigned int outputHeight,
                                             unsigned int alignedInputChannel) {
    CONV11_STRIDE2(input, weight, bias, output, inputChannel, outputChannel, inputWidth, inputHeight, \
                   outputWidth, outputHeight, alignedInputChannel)
})
#undef ACT_VEC_F  // RELU6

ADD_SINGLE_KERNEL(alignWeight_gemm_FP16, (__global const DATA_T *src,
                                          __global DATA_T *dst,
                                          uint srcWidth,
                                          uint dstWidth,
                                          uint groupTopChannel,
                                          uint alignedGroupTopChannel) {
    ALIGNWEIGHT_GEMM(src, dst, srcWidth, dstWidth, groupTopChannel, alignedGroupTopChannel)
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
ADD_SINGLE_KERNEL(gemmBlocked_FP32, (__global const DATA_T *input,
                                     __global const DATA_T *weight,
                                     __global const DATA_T *bias,
                                     __global DATA_T *output,
                                     unsigned int alignWidth,
                                     unsigned int outputHeight,
                                     unsigned int outputWidth,
                                     unsigned int groupNum,
                                     unsigned int group) {
    GEMMBLOCKED(input, weight, bias, output, alignWidth, outputHeight, outputWidth, groupNum, group)
})

ADD_SINGLE_KERNEL(conv11_FP32, (__global const DATA_T *input,
                                __global const DATA_T *weight,
                                __global const DATA_T *bias,
                                __global DATA_T *output,
                                unsigned int inputChannel,
                                unsigned int outputChannel,
                                unsigned int width,
                                unsigned int height,
                                unsigned int alignedInputChannel) {
    CONV11_FP32(input, weight, bias, output, inputChannel, outputChannel, width, height, alignedInputChannel)
})

ADD_SINGLE_KERNEL(conv11_stride2_FP32, (__global const DATA_T *input,
                                        __global const DATA_T *weight,
                                        __global const DATA_T *bias,
                                        __global DATA_T *output,
                                        unsigned int inputChannel,
                                        unsigned int outputChannel,
                                        unsigned int inputWidth,
                                        unsigned int inputHeight,
                                        unsigned int outputWidth,
                                        unsigned int outputHeight,
                                        unsigned int alignedInputChannel) {
    CONV11_STRIDE2(input, weight, bias, output, inputChannel, outputChannel, inputWidth, inputHeight, \
                   outputWidth, outputHeight, alignedInputChannel)
})
#undef ACT_VEC_F  // NONE

// apply RELU to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELUgemmBlocked_FP32, (__global const DATA_T *input,
                                         __global const DATA_T *weight,
                                         __global const DATA_T *bias,
                                         __global DATA_T *output,
                                         unsigned int alignWidth,
                                         unsigned int outputHeight,
                                         unsigned int outputWidth,
                                         unsigned int groupNum,
                                         unsigned int group) {
    GEMMBLOCKED(input, weight, bias, output, alignWidth, outputHeight, outputWidth, groupNum, group)
})

ADD_SINGLE_KERNEL(RELUconv11_FP32, (__global const DATA_T *input,
                                    __global const DATA_T *weight,
                                    __global const DATA_T *bias,
                                    __global DATA_T *output,
                                    unsigned int inputChannel,
                                    unsigned int outputChannel,
                                    unsigned int width,
                                    unsigned int height,
                                    unsigned int alignedInputChannel) {
    CONV11_FP32(input, weight, bias, output, inputChannel, outputChannel, width, height, alignedInputChannel)
})

ADD_SINGLE_KERNEL(RELUconv11_stride2_FP32, (__global const DATA_T *input,
                                            __global const DATA_T *weight,
                                            __global const DATA_T *bias,
                                            __global DATA_T *output,
                                            unsigned int inputChannel,
                                            unsigned int outputChannel,
                                            unsigned int inputWidth,
                                            unsigned int inputHeight,
                                            unsigned int outputWidth,
                                            unsigned int outputHeight,
                                            unsigned int alignedInputChannel) {
    CONV11_STRIDE2(input, weight, bias, output, inputChannel, outputChannel, inputWidth, inputHeight, \
                   outputWidth, outputHeight, alignedInputChannel)
})
#undef ACT_VEC_F  // RELU

// apply RELU6 to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU6_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELU6gemmBlocked_FP32, (__global const DATA_T *input,
                                          __global const DATA_T *weight,
                                          __global const DATA_T *bias,
                                          __global DATA_T *output,
                                          unsigned int alignWidth,
                                          unsigned int outputHeight,
                                          unsigned int outputWidth,
                                          unsigned int groupNum,
                                          unsigned int group) {
    GEMMBLOCKED(input, weight, bias, output, alignWidth, outputHeight, outputWidth, groupNum, group)
})

ADD_SINGLE_KERNEL(RELU6conv11_FP32, (__global const DATA_T *input,
                                     __global const DATA_T *weight,
                                     __global const DATA_T *bias,
                                     __global DATA_T *output,
                                     unsigned int inputChannel,
                                     unsigned int outputChannel,
                                     unsigned int width,
                                     unsigned int height,
                                     unsigned int alignedInputChannel) {
    CONV11_FP32(input, weight, bias, output, inputChannel, outputChannel, width, height, alignedInputChannel)
})

ADD_SINGLE_KERNEL(RELU6conv11_stride2_FP32, (__global const DATA_T *input,
                                             __global const DATA_T *weight,
                                             __global const DATA_T *bias,
                                             __global DATA_T *output,
                                             unsigned int inputChannel,
                                             unsigned int outputChannel,
                                             unsigned int inputWidth,
                                             unsigned int inputHeight,
                                             unsigned int outputWidth,
                                             unsigned int outputHeight,
                                             unsigned int alignedInputChannel) {
    CONV11_STRIDE2(input, weight, bias, output, inputChannel, outputChannel, inputWidth, inputHeight, \
                   outputWidth, outputHeight, alignedInputChannel)
})
#undef ACT_VEC_F  // RELU6

ADD_SINGLE_KERNEL(alignWeight_gemm_FP32, (__global const DATA_T *src,
                                          __global DATA_T *dst,
                                          uint srcWidth,
                                          uint dstWidth,
                                          uint groupTopChannel,
                                          uint alignedGroupTopChannel) {
    ALIGNWEIGHT_GEMM(src, dst, srcWidth, dstWidth, groupTopChannel, alignedGroupTopChannel)
})
#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // float

#define QUANTIZEDGEMMBLOCKED(input, weight, bias, output, alignWidth, outputHeight, outputWidth, groupNum, \
                             group, inputOffset, outputOffset, output_multiplier, output_shift, activation_min, \
                             activation_max) \
    output_shift *= -1; \
    int left_shift = output_shift > 0 ? output_shift : 0; \
    int right_shift = output_shift > 0 ? 0 : -output_shift; \
    int globalID0 = get_global_id(0); \
    int globalID1 = get_global_id(1) * 4; \
    int globalID2 = get_global_id(2); \
    if (globalID2 < outputWidth) { \
        int o_temp; \
        bool overflow; \
        long ab_64; \
        int nudge; \
        int mask; \
        int remainder; \
        int threshold; \
        if (globalID1 + 3 < outputHeight / groupNum) { \
            int out = (int)0; \
            int out2 = (int)0; \
            int out3 = (int)0; \
            int out4 = (int)0; \
            int4 bias4 = vload4(0, bias + group * outputHeight / groupNum + globalID1); \
            { \
                int inputIndex = globalID0 * alignWidth * get_global_size(2) * groupNum + \
                                    alignWidth * get_global_size(2) * group + \
                                    globalID2 * alignWidth; \
                int weightIndex = \
                    group * get_global_size(1) * 4 * alignWidth + globalID1 * alignWidth; \
                for (int i = 0; i<alignWidth>> 4; i++) { \
                    DATA_T16 input16; \
                    short16 weight16; \
                    input16 = vload16(0, input + inputIndex); \
                    inputIndex += 16; \
                    weight16 = vload16(0, weight + weightIndex); \
                    out += (input16.s0 + inputOffset) * weight16.s0 + \
                            (input16.s1 + inputOffset) * weight16.s1 + \
                            (input16.s2 + inputOffset) * weight16.s2 + \
                            (input16.s3 + inputOffset) * weight16.s3 + \
                            (input16.s4 + inputOffset) * weight16.s4 + \
                            (input16.s5 + inputOffset) * weight16.s5 + \
                            (input16.s6 + inputOffset) * weight16.s6 + \
                            (input16.s7 + inputOffset) * weight16.s7; \
                    out2 += (input16.s0 + inputOffset) * weight16.s8 + \
                            (input16.s1 + inputOffset) * weight16.s9 + \
                            (input16.s2 + inputOffset) * weight16.sa + \
                            (input16.s3 + inputOffset) * weight16.sb + \
                            (input16.s4 + inputOffset) * weight16.sc + \
                            (input16.s5 + inputOffset) * weight16.sd + \
                            (input16.s6 + inputOffset) * weight16.se + \
                            (input16.s7 + inputOffset) * weight16.sf; \
                    weight16 = vload16(1, weight + weightIndex); \
                    out3 += (input16.s0 + inputOffset) * weight16.s0 + \
                            (input16.s1 + inputOffset) * weight16.s1 + \
                            (input16.s2 + inputOffset) * weight16.s2 + \
                            (input16.s3 + inputOffset) * weight16.s3 + \
                            (input16.s4 + inputOffset) * weight16.s4 + \
                            (input16.s5 + inputOffset) * weight16.s5 + \
                            (input16.s6 + inputOffset) * weight16.s6 + \
                            (input16.s7 + inputOffset) * weight16.s7; \
                    out4 += (input16.s0 + inputOffset) * weight16.s8 + \
                            (input16.s1 + inputOffset) * weight16.s9 + \
                            (input16.s2 + inputOffset) * weight16.sa + \
                            (input16.s3 + inputOffset) * weight16.sb + \
                            (input16.s4 + inputOffset) * weight16.sc + \
                            (input16.s5 + inputOffset) * weight16.sd + \
                            (input16.s6 + inputOffset) * weight16.se + \
                            (input16.s7 + inputOffset) * weight16.sf; \
                    weight16 = vload16(2, weight + weightIndex); \
                    out += (input16.s8 + inputOffset) * weight16.s0 + \
                            (input16.s9 + inputOffset) * weight16.s1 + \
                            (input16.sa + inputOffset) * weight16.s2 + \
                            (input16.sb + inputOffset) * weight16.s3 + \
                            (input16.sc + inputOffset) * weight16.s4 + \
                            (input16.sd + inputOffset) * weight16.s5 + \
                            (input16.se + inputOffset) * weight16.s6 + \
                            (input16.sf + inputOffset) * weight16.s7; \
                    out2 += (input16.s8 + inputOffset) * weight16.s8 + \
                            (input16.s9 + inputOffset) * weight16.s9 + \
                            (input16.sa + inputOffset) * weight16.sa + \
                            (input16.sb + inputOffset) * weight16.sb + \
                            (input16.sc + inputOffset) * weight16.sc + \
                            (input16.sd + inputOffset) * weight16.sd + \
                            (input16.se + inputOffset) * weight16.se + \
                            (input16.sf + inputOffset) * weight16.sf; \
                    weight16 = vload16(3, weight + weightIndex); \
                    out3 += (input16.s8 + inputOffset) * weight16.s0 + \
                            (input16.s9 + inputOffset) * weight16.s1 + \
                            (input16.sa + inputOffset) * weight16.s2 + \
                            (input16.sb + inputOffset) * weight16.s3 + \
                            (input16.sc + inputOffset) * weight16.s4 + \
                            (input16.sd + inputOffset) * weight16.s5 + \
                            (input16.se + inputOffset) * weight16.s6 + \
                            (input16.sf + inputOffset) * weight16.s7; \
                    out4 += (input16.s8 + inputOffset) * weight16.s8 + \
                            (input16.s9 + inputOffset) * weight16.s9 + \
                            (input16.sa + inputOffset) * weight16.sa + \
                            (input16.sb + inputOffset) * weight16.sb + \
                            (input16.sc + inputOffset) * weight16.sc + \
                            (input16.sd + inputOffset) * weight16.sd + \
                            (input16.se + inputOffset) * weight16.se + \
                            (input16.sf + inputOffset) * weight16.sf; \
                    weightIndex += 64; \
                } \
            } \
            { \
                int outputIndex = globalID0 * outputHeight * outputWidth + \
                                    group * outputHeight * outputWidth / groupNum + \
                                    globalID1 * outputWidth + globalID2; \
                o_temp = out + bias4.s0; \
                saturatingRoundingDoublingHighMul(o_temp * (1 << left_shift), \
                                                    output_multiplier, \
                                                    o_temp, \
                                                    overflow, \
                                                    ab_64, \
                                                    nudge); \
                roundingDivideByPOT(o_temp, right_shift, o_temp, mask, remainder, threshold); \
                o_temp += outputOffset; \
                o_temp = max(o_temp, activation_min); \
                o_temp = min(o_temp, activation_max); \
                output[outputIndex] = (DATA_T)o_temp; \
                o_temp = out2 + bias4.s1; \
                saturatingRoundingDoublingHighMul(o_temp * (1 << left_shift), \
                                                    output_multiplier, \
                                                    o_temp, \
                                                    overflow, \
                                                    ab_64, \
                                                    nudge); \
                roundingDivideByPOT(o_temp, right_shift, o_temp, mask, remainder, threshold); \
                o_temp += outputOffset; \
                o_temp = max(o_temp, activation_min); \
                o_temp = min(o_temp, activation_max); \
                output[outputIndex + outputWidth] = (DATA_T)o_temp; \
                o_temp = out3 + bias4.s2; \
                saturatingRoundingDoublingHighMul(o_temp * (1 << left_shift), \
                                                    output_multiplier, \
                                                    o_temp, \
                                                    overflow, \
                                                    ab_64, \
                                                    nudge); \
                roundingDivideByPOT(o_temp, right_shift, o_temp, mask, remainder, threshold); \
                o_temp += outputOffset; \
                o_temp = max(o_temp, activation_min); \
                o_temp = min(o_temp, activation_max); \
                output[outputIndex + 2 * outputWidth] = (DATA_T)o_temp; \
                o_temp = out4 + bias4.s3; \
                saturatingRoundingDoublingHighMul(o_temp * (1 << left_shift), \
                                                    output_multiplier, \
                                                    o_temp, \
                                                    overflow, \
                                                    ab_64, \
                                                    nudge); \
                roundingDivideByPOT(o_temp, right_shift, o_temp, mask, remainder, threshold); \
                o_temp += outputOffset; \
                o_temp = max(o_temp, activation_min); \
                o_temp = min(o_temp, activation_max); \
                output[outputIndex + 3 * outputWidth] = (DATA_T)o_temp; \
            } \
        } else if (globalID1 + 2 < outputHeight / groupNum) { \
            int8 out = (int8)0; \
            int8 out2 = (int8)0; \
            int8 out3 = (int8)0; \
            int3 bias3 = vload3(0, bias + group * outputHeight / groupNum + globalID1); \
            { \
                int inputIndex = globalID0 * alignWidth * get_global_size(2) * groupNum + \
                                    alignWidth * get_global_size(2) * group + \
                                    globalID2 * alignWidth; \
                int weightIndex = \
                    group * get_global_size(1) * 4 * alignWidth + globalID1 * alignWidth; \
                for (int i = 0; i<alignWidth>> 3; i++) { \
                    DATA_T8 input8; \
                    short8 weight8; \
                    input8 = vload8(0, input + inputIndex); \
                    inputIndex += 8; \
                    weight8 = vload8(0, weight + weightIndex); \
                    out += (convert_int8(input8) + inputOffset) * convert_int8(weight8); \
                    weight8 = vload8(0, weight + weightIndex + 8); \
                    out2 += (convert_int8(input8) + inputOffset) * convert_int8(weight8); \
                    weight8 = vload8(0, weight + weightIndex + 16); \
                    out3 += (convert_int8(input8) + inputOffset) * convert_int8(weight8); \
                    weightIndex += 32; \
                } \
            } \
            { \
                int outputIndex = globalID0 * outputHeight * outputWidth + \
                                    group * outputHeight * outputWidth / groupNum + \
                                    globalID1 * outputWidth + globalID2; \
                o_temp = out.s0 + out.s1 + out.s2 + out.s3 + out.s4 + out.s5 + out.s6 + out.s7 + \
                            bias3.s0; \
                saturatingRoundingDoublingHighMul(o_temp * (1 << left_shift), \
                                                    output_multiplier, \
                                                    o_temp, \
                                                    overflow, \
                                                    ab_64, \
                                                    nudge); \
                roundingDivideByPOT(o_temp, right_shift, o_temp, mask, remainder, threshold); \
                o_temp += outputOffset; \
                o_temp = max(o_temp, activation_min); \
                o_temp = min(o_temp, activation_max); \
                output[outputIndex] = (DATA_T)o_temp; \
                o_temp = out2.s0 + out2.s1 + out2.s2 + out2.s3 + out2.s4 + out2.s5 + \
                                out2.s6 + out2.s7 + bias3.s1; \
                saturatingRoundingDoublingHighMul(o_temp * (1 << left_shift), \
                                                    output_multiplier, \
                                                    o_temp, \
                                                    overflow, \
                                                    ab_64, \
                                                    nudge); \
                roundingDivideByPOT(o_temp, right_shift, o_temp, mask, remainder, threshold); \
                o_temp += outputOffset; \
                o_temp = max(o_temp, activation_min); \
                o_temp = min(o_temp, activation_max); \
                output[outputIndex + outputWidth] = (DATA_T)o_temp; \
                o_temp = out3.s0 + out3.s1 + out3.s2 + out3.s3 + out3.s4 + out3.s5 + \
                                out3.s6 + out3.s7 + bias3.s2; \
                saturatingRoundingDoublingHighMul(o_temp * (1 << left_shift), \
                                                    output_multiplier, \
                                                    o_temp, \
                                                    overflow, \
                                                    ab_64, \
                                                    nudge); \
                roundingDivideByPOT(o_temp, right_shift, o_temp, mask, remainder, threshold); \
                o_temp += outputOffset; \
                o_temp = max(o_temp, activation_min); \
                o_temp = min(o_temp, activation_max); \
                output[outputIndex + 2 * outputWidth] = (DATA_T)o_temp; \
            } \
        } else if (globalID1 + 1 < outputHeight / groupNum) { \
            int16 out = (int16)0; \
            int2 bias2 = vload2(0, bias + group * outputHeight / groupNum + globalID1); \
            { \
                int inputIndex = globalID0 * alignWidth * get_global_size(2) * groupNum + \
                                    alignWidth * get_global_size(2) * group + \
                                    globalID2 * alignWidth; \
                int weightIndex = \
                    group * get_global_size(1) * 4 * alignWidth + globalID1 * alignWidth; \
                for (int i = 0; i<alignWidth>> 3; i++) { \
                    DATA_T8 input8; \
                    DATA_T16 input16; \
                    short16 weight16; \
                    input8 = vload8(0, input + inputIndex); \
                    inputIndex += 8; \
                    input16.lo = input8; \
                    input16.hi = input8; \
                    weight16 = vload16(0, weight + weightIndex); \
                    out += (convert_int16(input16) + inputOffset) * convert_int16(weight16); \
                    weightIndex += 32; \
                } \
            } \
            { \
                int outputIndex = globalID0 * outputHeight * outputWidth + \
                                    group * outputHeight * outputWidth / groupNum + \
                                    globalID1 * outputWidth + globalID2; \
                o_temp = out.s0 + out.s1 + out.s2 + out.s3 + out.s4 + out.s5 + out.s6 + out.s7 + \
                            bias2.s0; \
                saturatingRoundingDoublingHighMul(o_temp * (1 << left_shift), \
                                                    output_multiplier, \
                                                    o_temp, \
                                                    overflow, \
                                                    ab_64, \
                                                    nudge); \
                roundingDivideByPOT(o_temp, right_shift, o_temp, mask, remainder, threshold); \
                o_temp += outputOffset; \
                o_temp = max(o_temp, activation_min); \
                o_temp = min(o_temp, activation_max); \
                output[outputIndex] = (DATA_T)o_temp; \
                o_temp = out.s8 + out.s9 + out.sa + out.sb + out.sc + out.sd + out.se + \
                                out.sf + bias2.s1; \
                saturatingRoundingDoublingHighMul(o_temp * (1 << left_shift), \
                                                    output_multiplier, \
                                                    o_temp, \
                                                    overflow, \
                                                    ab_64, \
                                                    nudge); \
                roundingDivideByPOT(o_temp, right_shift, o_temp, mask, remainder, threshold); \
                o_temp += outputOffset; \
                o_temp = max(o_temp, activation_min); \
                o_temp = min(o_temp, activation_max); \
                output[outputIndex + outputWidth] = (DATA_T)o_temp; \
            } \
        } else if (globalID1 < outputHeight / groupNum) { \
            int8 out = (int8)0; \
            int bias1 = bias[group * outputHeight / groupNum + globalID1]; \
            { \
                int inputIndex = globalID0 * alignWidth * get_global_size(2) * groupNum + \
                                    alignWidth * get_global_size(2) * group + \
                                    globalID2 * alignWidth; \
                int weightIndex = \
                    group * get_global_size(1) * 4 * alignWidth + globalID1 * alignWidth; \
                for (int i = 0; i<alignWidth>> 3; i++) { \
                    DATA_T8 input8; \
                    short8 weight8; \
                    input8 = vload8(0, input + inputIndex); \
                    inputIndex += 8; \
                    weight8 = vload8(0, weight + weightIndex); \
                    out += (convert_int8(input8) + inputOffset) * convert_int8(weight8); \
                    weightIndex += 32; \
                } \
            } \
            { \
                int outputIndex = globalID0 * outputHeight * outputWidth + \
                                    group * outputHeight * outputWidth / groupNum + \
                                    globalID1 * outputWidth + globalID2; \
                o_temp = out.s0 + out.s1 + out.s2 + out.s3 + out.s4 + out.s5 + out.s6 + out.s7 + \
                            bias1; \
                saturatingRoundingDoublingHighMul(o_temp * (1 << left_shift), \
                                                    output_multiplier, \
                                                    o_temp, \
                                                    overflow, \
                                                    ab_64, \
                                                    nudge); \
                roundingDivideByPOT(o_temp, right_shift, o_temp, mask, remainder, threshold); \
                o_temp += outputOffset; \
                o_temp = max(o_temp, activation_min); \
                o_temp = min(o_temp, activation_max); \
                output[outputIndex] = (DATA_T)o_temp; \
            } \
        } \
    }

#define ALIGNQUANTIZEDWEIGHT_GEMM(src, dst, srcWidth, dstWidth, groupTopChannel, alignedGroupTopChannel, filterOffset) \
    uint globalID0 = get_global_id(0); \
    uint globalID1 = get_global_id(1) * 8; \
    uint srcIndex = globalID0 * srcWidth + globalID1; \
    uint dstIndex = \
        globalID0 / groupTopChannel * dstWidth * alignedGroupTopChannel + \
        globalID0 % groupTopChannel / 4 * 4 * dstWidth + globalID1 * 4 + \
        globalID0 % groupTopChannel % 4 * 8; \
    short offset = (short)filterOffset; \
    if (globalID1 + 7 < srcWidth) { \
        vstore8( \
            convert_short8(vload8(0, src + srcIndex)) + offset, 0, dst + dstIndex); \
    } else if (globalID1 + 6 < srcWidth) { \
        vstore4( \
            convert_short4(vload4(0, src + srcIndex)) + offset, 0, dst + dstIndex); \
        vstore3(convert_short3(vload3(0, src + srcIndex + 4)) + offset, \
                0, \
                dst + dstIndex + 4); \
    } else if (globalID1 + 5 < srcWidth) { \
        vstore4( \
            convert_short4(vload4(0, src + srcIndex)) + offset, 0, dst + dstIndex); \
        vstore2(convert_short2(vload2(0, src + srcIndex + 4)) + offset, \
                0, \
                dst + dstIndex + 4); \
    } else if (globalID1 + 4 < srcWidth) { \
        vstore4( \
            convert_short4(vload4(0, src + srcIndex)) + offset, 0, dst + dstIndex); \
        dst[dstIndex + 4] = (short)src[srcIndex + 4] + offset; \
    } else if (globalID1 + 3 < srcWidth) { \
        vstore4( \
            convert_short4(vload4(0, src + srcIndex)) + offset, 0, dst + dstIndex); \
    } else if (globalID1 + 2 < srcWidth) { \
        vstore3( \
            convert_short3(vload3(0, src + srcIndex)) + offset, 0, dst + dstIndex); \
    } else if (globalID1 + 1 < srcWidth) { \
        vstore2( \
            convert_short2(vload2(0, src + srcIndex)) + offset, 0, dst + dstIndex); \
    } else if (globalID1 < srcWidth) { \
        dst[dstIndex] = (short)src[srcIndex] + offset; \
    }

#define CONV_PER_CHANNEL_QUANTIZED(input, weight, bias, output_multiplier, output_shift, output, group, out_g, \
                                   kernel_height, kernel_width, stride_height, stride_width, pad_left, pad_top, \
                                   out_channel, out_height, out_width, in_channel, in_height, in_width, input_offset, \
                                   weight_offset, output_offset, act_min, act_max) \
    uint out_b = get_global_id(0); \
    uint out_c = get_global_id(1); \
    uint hw_out = get_global_id(2); \
    int out_h = hw_out / out_width; \
    int out_w = hw_out % out_width; \
    int acc = 0; \
    for (int in_c = 0; in_c < in_channel / group; in_c++) { \
        for (int k_h = 0; k_h < kernel_height; k_h++) { \
            for (int k_w = 0; k_w < kernel_width; k_w++) { \
                int in_h = out_h * stride_height + k_h - pad_top; \
                int in_w = out_w * stride_width + k_w - pad_left; \
                if (in_h >= 0 && in_h < in_height && in_w >= 0 && in_w < in_width) { \
                    int in_index = \
                        out_b * in_channel * in_height * in_width + \
                        out_g * (in_channel / group) * in_height * in_width + \
                        in_c * in_height * in_width + in_h * in_width + in_w; \
                    int weight_index = out_g * (in_channel / group) * \
                                            (out_channel / group) * kernel_height * \
                                            kernel_width + \
                                        out_c * (in_channel / group) * \
                                            kernel_height * kernel_width + \
                                        in_c * kernel_height * kernel_width + \
                                        k_h * kernel_width + k_w; \
                    acc += (input[in_index] + input_offset) * \
                            (weight[weight_index] + weight_offset); \
                } \
            } \
        } \
    } \
    acc += bias[out_g * (out_channel / group) + out_c]; \
    bool overflow; \
    long ab_64; \
    int nudge; \
    int mask; \
    int remainder; \
    int threshold; \
    int shift = output_shift[out_g * (out_channel / group) + out_c]; \
    int left_shift = shift > 0 ? shift : 0; \
    int right_shift = shift > 0 ? 0 : -shift; \
    acc = acc * (1 << left_shift); \
    reQuantized(acc, \
                output_multiplier[out_g * (out_channel / group) + out_c], \
                right_shift, \
                output_offset, \
                act_min, \
                act_max, \
                overflow, \
                ab_64, \
                nudge, \
                mask, \
                remainder, \
                threshold); \
    output[out_b * out_channel * out_height * out_width + \
    out_g * (out_channel / group) * out_height * out_width + \
    out_c * out_height * out_width + hw_out] = (DATA_T)acc;

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

ADD_KERNEL_HEADER(quantizedGemmBlocked_INT8,
                  {DEFINE_SATURATING_ROUNDING_DOUBLING_HIGH_MUL, DEFINE_ROUNDING_DIVIDE_BY_POT})
ADD_SINGLE_KERNEL(quantizedGemmBlocked_INT8, (__global const DATA_T *input,
                                              __global short *weight,
                                              __global const int *bias,
                                              __global DATA_T *output,
                                              unsigned int alignWidth,
                                              unsigned int outputHeight,
                                              unsigned int outputWidth,
                                              unsigned int groupNum,
                                              unsigned int group,
                                              int inputOffset,
                                              int outputOffset,
                                              int output_multiplier,
                                              int output_shift,
                                              int activation_min,
                                              int activation_max) {
    QUANTIZEDGEMMBLOCKED(input, weight, bias, output, alignWidth, outputHeight, outputWidth, groupNum, group, \
                         inputOffset, outputOffset, output_multiplier, output_shift, activation_min, activation_max)
})

ADD_SINGLE_KERNEL(alignQuantizedWeight_gemm_INT8, (__global const DATA_T *src,
                                                   __global short *dst,
                                                   uint srcWidth,
                                                   uint dstWidth,
                                                   uint groupTopChannel,
                                                   uint alignedGroupTopChannel,
                                                   int filterOffset) {
    ALIGNQUANTIZEDWEIGHT_GEMM(src, dst, srcWidth, dstWidth, groupTopChannel, alignedGroupTopChannel, filterOffset)
})

ADD_KERNEL_HEADER(conv_per_channel_quantized_INT8,
                  {DEFINE_SATURATING_ROUNDING_DOUBLING_HIGH_MUL, DEFINE_ROUNDING_DIVIDE_BY_POT, DEFINE_REQUANTIZED})
ADD_SINGLE_KERNEL(conv_per_channel_quantized_INT8, (__global const DATA_T *input,
                                                    __global const char *weight,
                                                    __global const int *bias,
                                                    __global const int *output_multiplier,
                                                    __global const int *output_shift,
                                                    __global DATA_T *output,
                                                    int group,
                                                    int out_g,
                                                    int kernel_height,
                                                    int kernel_width,
                                                    int stride_height,
                                                    int stride_width,
                                                    int pad_left,
                                                    int pad_top,
                                                    int out_channel,
                                                    int out_height,
                                                    int out_width,
                                                    int in_channel,
                                                    int in_height,
                                                    int in_width,
                                                    int input_offset,
                                                    int weight_offset,
                                                    int output_offset,
                                                    int act_min,
                                                    int act_max) {
    CONV_PER_CHANNEL_QUANTIZED(input, weight, bias, output_multiplier, output_shift, output, group, out_g, \
                               kernel_height, kernel_width, stride_height, stride_width, pad_left, pad_top, \
                               out_channel, out_height, out_width, in_channel, in_height, in_width, input_offset, \
                               weight_offset, output_offset, act_min, act_max)
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

// SIGNED INT8 kernels
#define DATA_T char
#define DATA_T2 char2
#define DATA_T3 char3
#define DATA_T4 char4
#define DATA_T8 char8
#define DATA_T16 char16
#define CONVERT_TO_DATA_T(x) (DATA_T)x
#define CONVERT_TO_DATA_T2(x) convert_char2(x)
#define CONVERT_TO_DATA_T3(x) convert_char3(x)
#define CONVERT_TO_DATA_T4(x) convert_char4(x)

ADD_KERNEL_HEADER(SIGNEDquantizedGemmBlocked_INT8,
                  {DEFINE_SATURATING_ROUNDING_DOUBLING_HIGH_MUL, DEFINE_ROUNDING_DIVIDE_BY_POT})
ADD_SINGLE_KERNEL(SIGNEDquantizedGemmBlocked_INT8, (__global const DATA_T *input,
                                                    __global short *weight,
                                                    __global const int *bias,
                                                    __global DATA_T *output,
                                                    unsigned int alignWidth,
                                                    unsigned int outputHeight,
                                                    unsigned int outputWidth,
                                                    unsigned int groupNum,
                                                    unsigned int group,
                                                    int inputOffset,
                                                    int outputOffset,
                                                    int output_multiplier,
                                                    int output_shift,
                                                    int activation_min,
                                                    int activation_max) {
    QUANTIZEDGEMMBLOCKED(input, weight, bias, output, alignWidth, outputHeight, outputWidth, groupNum, group, \
                         inputOffset, outputOffset, output_multiplier, output_shift, activation_min, activation_max)
})

ADD_SINGLE_KERNEL(SIGNEDalignQuantizedWeight_gemm_INT8, (__global const DATA_T *src,
                                                         __global short *dst,
                                                         uint srcWidth,
                                                         uint dstWidth,
                                                         uint groupTopChannel,
                                                         uint alignedGroupTopChannel,
                                                         int filterOffset) {
    ALIGNQUANTIZEDWEIGHT_GEMM(src, dst, srcWidth, dstWidth, groupTopChannel, alignedGroupTopChannel, filterOffset)
})

ADD_KERNEL_HEADER(SIGNEDconv_per_channel_quantized_INT8,
                  {DEFINE_SATURATING_ROUNDING_DOUBLING_HIGH_MUL, DEFINE_ROUNDING_DIVIDE_BY_POT, DEFINE_REQUANTIZED})
ADD_SINGLE_KERNEL(SIGNEDconv_per_channel_quantized_INT8, (__global const DATA_T *input,
                                                          __global const char *weight,
                                                          __global const int *bias,
                                                          __global const int *output_multiplier,
                                                          __global const int *output_shift,
                                                          __global DATA_T *output,
                                                          int group,
                                                          int out_g,
                                                          int kernel_height,
                                                          int kernel_width,
                                                          int stride_height,
                                                          int stride_width,
                                                          int pad_left,
                                                          int pad_top,
                                                          int out_channel,
                                                          int out_height,
                                                          int out_width,
                                                          int in_channel,
                                                          int in_height,
                                                          int in_width,
                                                          int input_offset,
                                                          int weight_offset,
                                                          int output_offset,
                                                          int act_min,
                                                          int act_max) {
    CONV_PER_CHANNEL_QUANTIZED(input, weight, bias, output_multiplier, output_shift, output, group, out_g, \
                               kernel_height, kernel_width, stride_height, stride_width, pad_left, pad_top, \
                               out_channel, out_height, out_width, in_channel, in_height, in_width, input_offset, \
                               weight_offset, output_offset, act_min, act_max)
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
#undef DATA_T  // int8

}  // namespace gpu
}  // namespace ud
}  // namespace enn
