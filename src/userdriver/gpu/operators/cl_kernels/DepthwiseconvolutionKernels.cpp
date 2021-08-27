#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {


#define DEPTHWISE_CONV_UNEQUAL(bias, output, start) \
        int g0 = get_global_id(0); \
        int g1 = get_global_id(1); \
        int g2 = get_global_id(2); \
        int outputIndex = (g0 * get_global_size(1) + g1 + start) * get_global_size(2) + g2; \
        output[outputIndex] = ACT_VEC_F(DATA_T, bias[g1 + start]);

#define DEPTHWISE_CONV(input, weight, bias, output, inputHeight, inputWidth, kernelHeight, kernelWidth, strideHeight, \
                       strideWidth, topHeight, topWidth, depth_multiplier) \
        int g0 = get_global_id(0); \
        int g1 = get_global_id(1); \
        int g2 = get_global_id(2); \
        if (g2 < topHeight * topWidth) { \
            int inputX = (g2 % topWidth) * strideWidth; \
            int inputY = (g2 / topWidth) * strideHeight; \
            for (int dep = 0; dep < depth_multiplier; dep++) { \
                int out_c = dep + g1 * depth_multiplier; \
                int inputIndex = (g0 * get_global_size(1) + g1) * inputHeight * inputWidth; \
                int outputIndex = \
                    (g0 * get_global_size(1) * depth_multiplier + out_c) * topHeight * topWidth; \
                DATA_T temp_out = bias[out_c]; \
                for (int i = 0; i < kernelHeight; i++) { \
                    for (int j = 0; j < kernelWidth; j++) { \
                        temp_out += \
                            input[inputIndex + (inputY + i) * inputWidth + j + inputX] * \
                            weight[out_c * kernelHeight * kernelWidth + i * kernelWidth + j]; \
                    } \
                } \
                output[outputIndex + g2] = ACT_VEC_F(DATA_T, temp_out); \
            } \
        }

#define DEPTHWISE_CONV_3X3S1_4P(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                topChannel, depth_multiplier) \
        int g0 = get_global_id(0); \
        int g1 = get_global_id(1) * 2; \
        int g2 = get_global_id(2) * 2; \
        for (int dep = 0; dep < depth_multiplier; dep++) { \
            int out_g0 = dep + g0 * depth_multiplier; \
            int channel = out_g0 % topChannel; \
            int inputIndex = g0 * inputHeight * inputWidth + g1 * inputWidth + g2; \
            int outputIndex = out_g0 * topHeight * topWidth + g1 * topWidth + g2; \
            DATA_T bias_tmp = bias[channel]; \
            if ((g2 + 1) < topWidth && (g1 + 1) < topHeight) {  \
                DATA_T4 out00; \
                DATA_T4 out01; \
                DATA_T4 k012 = vload4(0, weight + channel * 9); \
                DATA_T4 k345 = vload4(0, weight + channel * 9 + 3); \
                DATA_T4 k678 = vload4(0, weight + channel * 9 + 6); \
                DATA_T4 input00 = vload4(0, input + inputIndex); \
                DATA_T4 input10 = vload4(0, input + inputIndex + inputWidth); \
                DATA_T4 input20 = vload4(0, input + inputIndex + 2 * inputWidth); \
                DATA_T4 input30 = vload4(0, input + inputIndex + 3 * inputWidth); \
                DATA_T4 input01 = (DATA_T4)(input00.s1, input00.s2, input00.s3, 0); \
                DATA_T4 input11 = (DATA_T4)(input10.s1, input10.s2, input10.s3, 0); \
                DATA_T4 input21 = (DATA_T4)(input20.s1, input20.s2, input20.s3, 0); \
                DATA_T4 input31 = (DATA_T4)(input30.s1, input30.s2, input30.s3, 0); \
                out00 = k012 * input00; \
                out01 = k012 * input01; \
                out00 += k345 * input10; \
                out01 += k345 * input11; \
                out00 += k678 * input20; \
                out01 += k678 * input21; \
                output[outputIndex] = \
                    ACT_VEC_F(DATA_T, out00.s0 + out00.s1 + out00.s2 + bias_tmp); \
                output[outputIndex + 1] = \
                    ACT_VEC_F(DATA_T, out01.s0 + out01.s1 + out01.s2 + bias_tmp); \
                out00 = k012 * input10; \
                out01 = k012 * input11; \
                out00 += k345 * input20; \
                out01 += k345 * input21; \
                out00 += k678 * input30; \
                out01 += k678 * input31; \
                output[outputIndex + topWidth] = \
                    ACT_VEC_F(DATA_T, out00.s0 + out00.s1 + out00.s2 + bias_tmp); \
                output[outputIndex + topWidth + 1] = \
                    ACT_VEC_F(DATA_T, out01.s0 + out01.s1 + out01.s2 + bias_tmp); \
            } else if (g2 < topWidth && (g1 + 1) < topHeight) {  \
                DATA_T4 out00; \
                DATA_T4 k012 = vload4(0, weight + channel * 9); \
                DATA_T4 k345 = vload4(0, weight + channel * 9 + 3); \
                DATA_T4 k678 = vload4(0, weight + channel * 9 + 6); \
                DATA_T4 input00 = vload4(0, input + inputIndex); \
                DATA_T4 input10 = vload4(0, input + inputIndex + inputWidth); \
                DATA_T4 input20 = vload4(0, input + inputIndex + 2 * inputWidth); \
                DATA_T4 input30 = vload4(0, input + inputIndex + 3 * inputWidth); \
                out00 = k012 * input00; \
                out00 += k345 * input10; \
                out00 += k678 * input20; \
                output[outputIndex] = \
                    ACT_VEC_F(DATA_T, out00.s0 + out00.s1 + out00.s2 + bias_tmp); \
                out00 = k012 * input10; \
                out00 += k345 * input20; \
                out00 += k678 * input30; \
                output[outputIndex + topWidth] = \
                    ACT_VEC_F(DATA_T, out00.s0 + out00.s1 + out00.s2 + bias_tmp); \
            } else if ((g2 + 1) < topWidth && g1 < topHeight) {  \
                DATA_T4 out00; \
                DATA_T4 out01; \
                DATA_T4 k012 = vload4(0, weight + channel * 9); \
                DATA_T4 k345 = vload4(0, weight + channel * 9 + 3); \
                DATA_T4 k678 = vload4(0, weight + channel * 9 + 6); \
                DATA_T4 input00 = vload4(0, input + inputIndex); \
                DATA_T4 input10 = vload4(0, input + inputIndex + inputWidth); \
                DATA_T4 input20 = vload4(0, input + inputIndex + 2 * inputWidth); \
                DATA_T4 input01 = (DATA_T4)(input00.s1, input00.s2, input00.s3, 0); \
                DATA_T4 input11 = (DATA_T4)(input10.s1, input10.s2, input10.s3, 0); \
                DATA_T4 input21 = (DATA_T4)(input20.s1, input20.s2, input20.s3, 0); \
                out00 = k012 * input00; \
                out01 = k012 * input01; \
                out00 += k345 * input10; \
                out01 += k345 * input11; \
                out00 += k678 * input20; \
                out01 += k678 * input21; \
                output[outputIndex] = \
                    ACT_VEC_F(DATA_T, out00.s0 + out00.s1 + out00.s2 + bias_tmp); \
                output[outputIndex + 1] = \
                    ACT_VEC_F(DATA_T, out01.s0 + out01.s1 + out01.s2 + bias_tmp); \
            } else if (g2 < topWidth && g1 < topHeight) {  \
                DATA_T4 out00; \
                DATA_T4 k012 = vload4(0, weight + channel * 9); \
                DATA_T4 k345 = vload4(0, weight + channel * 9 + 3); \
                DATA_T4 k678 = vload4(0, weight + channel * 9 + 6); \
                DATA_T4 input00 = vload4(0, input + inputIndex); \
                DATA_T4 input10 = vload4(0, input + inputIndex + inputWidth); \
                DATA_T4 input20 = vload4(0, input + inputIndex + 2 * inputWidth); \
                out00 = k012 * input00; \
                out00 += k345 * input10; \
                out00 += k678 * input20; \
                output[outputIndex] = \
                    ACT_VEC_F(DATA_T, out00.s0 + out00.s1 + out00.s2 + bias_tmp); \
            } \
        }

#define DEPTHWISE_CONV_3X3D2S1_PAD_MERGE(input, weight, bias, output, inputHeight, inputWidth, topHeight, \
                                         topWidth, topChannel, padH, padW) \
        int g0 = get_global_id(0); \
        int g1 = get_global_id(1); \
        int g2 = get_global_id(2) * 4; \
        int inputRowIndex = g1 - padH; \
        int inputColIndex = g2 - padW; \
        int channel = g0 % topChannel; \
        int inputIndex = g0 * inputHeight * inputWidth + inputRowIndex * inputWidth + inputColIndex; \
        int outputIndex = g0 * topHeight * topWidth + g1 * topWidth + g2; \
        if (g2 < topWidth) {  \
            DATA_T16 k = vload16(0, weight + channel * 9); \
            DATA_T8 input00 = 0.0f; \
            DATA_T8 input20 = 0.0f; \
            DATA_T8 input40 = 0.0f; \
            if (inputRowIndex >= 0 && inputRowIndex < inputHeight) { \
                input00 = vload8(0, input + inputIndex); \
            } \
            if (inputRowIndex + 2 >= 0 && inputRowIndex + 2 < inputHeight) { \
                input20 = vload8(0, input + inputIndex + 2 * inputWidth); \
            } \
            if (inputRowIndex + 4 >= 0 && inputRowIndex + 4 < inputHeight) { \
                input40 = vload8(0, input + inputIndex + 4 * inputWidth); \
            } \
            if (inputColIndex < 0 || inputColIndex >= inputWidth) { \
                input00.s0 = 0.0f; \
                input20.s0 = 0.0f; \
                input40.s0 = 0.0f; \
            } \
            if (inputColIndex + 1 < 0 || inputColIndex + 1 >= inputWidth) { \
                input00.s1 = 0.0f; \
                input20.s1 = 0.0f; \
                input40.s1 = 0.0f; \
            } \
            if (inputColIndex + 2 < 0 || inputColIndex + 2 >= inputWidth) { \
                input00.s2 = 0.0f; \
                input20.s2 = 0.0f; \
                input40.s2 = 0.0f; \
            } \
            if (inputColIndex + 3 < 0 || inputColIndex + 3 >= inputWidth) { \
                input00.s3 = 0.0f; \
                input20.s3 = 0.0f; \
                input40.s3 = 0.0f; \
            } \
            if (inputColIndex + 4 < 0 || inputColIndex + 4 >= inputWidth) { \
                input00.s4 = 0.0f; \
                input20.s4 = 0.0f; \
                input40.s4 = 0.0f; \
            } \
            if (inputColIndex + 5 < 0 || inputColIndex + 5 >= inputWidth) { \
                input00.s5 = 0.0f; \
                input20.s5 = 0.0f; \
                input40.s5 = 0.0f; \
            } \
            if (inputColIndex + 6 < 0 || inputColIndex + 6 >= inputWidth) { \
                input00.s6 = 0.0f; \
                input20.s6 = 0.0f; \
                input40.s6 = 0.0f; \
            } \
            if (inputColIndex + 7 < 0 || inputColIndex + 7 >= inputWidth) { \
                input00.s7 = 0.0f; \
                input20.s7 = 0.0f; \
                input40.s7 = 0.0f; \
            } \
            DATA_T4 out4 = (DATA_T4)(bias[channel]); \
            DATA_T4 out00 = input00.lo * k.s0 + (DATA_T4)(input00.lo.hi, input00.hi.lo) * k.s1 + \
                          input00.hi * k.s2; \
            DATA_T4 out10 = input20.lo * k.s3 + (DATA_T4)(input20.lo.hi, input20.hi.lo) * k.s4 + \
                          input20.hi * k.s5; \
            DATA_T4 out20 = input40.lo * k.s6 + (DATA_T4)(input40.lo.hi, input40.hi.lo) * k.s7 + \
                          input40.hi * k.s8; \
            out4 += out00 + out10 + out20; \
            if (g2 + 3 < topWidth) { \
                vstore4(ACT_VEC_F(DATA_T4, out4), 0, output + outputIndex); \
            } else if (g2 + 2 < topWidth) { \
                vstore3(ACT_VEC_F(DATA_T3, out4.s012), 0, output + outputIndex); \
            } else if (g2 + 1 < topWidth) { \
                vstore2(ACT_VEC_F(DATA_T2, out4.s01), 0, output + outputIndex); \
            } else { \
                output[outputIndex] = ACT_VEC_F(DATA_T, out4.s0); \
            } \
        }

#define DEPTHWISE_CONV_3X3D4S1_PAD_MERGE(input, weight, bias, output, inputHeight, inputWidth, topHeight, \
                                         topWidth, topChannel, padH, padW) \
        int g0 = get_global_id(0); \
        int g1 = get_global_id(1); \
        int g2 = get_global_id(2) * 8; \
        int inputRowIndex = g1 - padH; \
        int inputColIndex = g2 - padW; \
        int channel = g0 % topChannel; \
        int inputIndex = g0 * inputHeight * inputWidth + inputRowIndex * inputWidth + inputColIndex; \
        int outputIndex = g0 * topHeight * topWidth + g1 * topWidth + g2; \
        if (g2 < topWidth) {  \
            DATA_T16 k = vload16(0, weight + channel * 9); \
            DATA_T16 input00 = 0.0f; \
            DATA_T16 input40 = 0.0f; \
            DATA_T16 input80 = 0.0f; \
            if (inputRowIndex >= 0 && inputRowIndex < inputHeight) { \
                input00 = vload16(0, input + inputIndex); \
            } \
            if (inputRowIndex + 4 >= 0 && inputRowIndex + 4 < inputHeight) { \
                input40 = vload16(0, input + inputIndex + 4 * inputWidth); \
            } \
            if (inputRowIndex + 8 >= 0 && inputRowIndex + 8 < inputHeight) { \
                input80 = vload16(0, input + inputIndex + 8 * inputWidth); \
            } \
            if (inputColIndex < 0 || inputColIndex >= inputWidth) { \
                input00.s0 = 0.0f; \
                input40.s0 = 0.0f; \
                input80.s0 = 0.0f; \
            } \
            if (inputColIndex + 1 < 0 || inputColIndex + 1 >= inputWidth) { \
                input00.s1 = 0.0f; \
                input40.s1 = 0.0f; \
                input80.s1 = 0.0f; \
            } \
            if (inputColIndex + 2 < 0 || inputColIndex + 2 >= inputWidth) { \
                input00.s2 = 0.0f; \
                input40.s2 = 0.0f; \
                input80.s2 = 0.0f; \
            } \
            if (inputColIndex + 3 < 0 || inputColIndex + 3 >= inputWidth) { \
                input00.s3 = 0.0f; \
                input40.s3 = 0.0f; \
                input80.s3 = 0.0f; \
            } \
            if (inputColIndex + 4 < 0 || inputColIndex + 4 >= inputWidth) { \
                input00.s4 = 0.0f; \
                input40.s4 = 0.0f; \
                input80.s4 = 0.0f; \
            } \
            if (inputColIndex + 5 < 0 || inputColIndex + 5 >= inputWidth) { \
                input00.s5 = 0.0f; \
                input40.s5 = 0.0f; \
                input80.s5 = 0.0f; \
            } \
            if (inputColIndex + 6 < 0 || inputColIndex + 6 >= inputWidth) { \
                input00.s6 = 0.0f; \
                input40.s6 = 0.0f; \
                input80.s6 = 0.0f; \
            } \
            if (inputColIndex + 7 < 0 || inputColIndex + 7 >= inputWidth) { \
                input00.s7 = 0.0f; \
                input40.s7 = 0.0f; \
                input80.s7 = 0.0f; \
            } \
            if (inputColIndex + 8 < 0 || inputColIndex + 8 >= inputWidth) { \
                input00.s8 = 0.0f; \
                input40.s8 = 0.0f; \
                input80.s8 = 0.0f; \
            } \
            if (inputColIndex + 9 < 0 || inputColIndex + 9 >= inputWidth) { \
                input00.s9 = 0.0f; \
                input40.s9 = 0.0f; \
                input80.s9 = 0.0f; \
            } \
            if (inputColIndex + 10 < 0 || inputColIndex + 10 >= inputWidth) { \
                input00.sa = 0.0f; \
                input40.sa = 0.0f; \
                input80.sa = 0.0f; \
            } \
            if (inputColIndex + 11 < 0 || inputColIndex + 11 >= inputWidth) { \
                input00.sb = 0.0f; \
                input40.sb = 0.0f; \
                input80.sb = 0.0f; \
            } \
            if (inputColIndex + 12 < 0 || inputColIndex + 12 >= inputWidth) { \
                input00.sc = 0.0f; \
                input40.sc = 0.0f; \
                input80.sc = 0.0f; \
            } \
            if (inputColIndex + 13 < 0 || inputColIndex + 13 >= inputWidth) { \
                input00.sd = 0.0f; \
                input40.sd = 0.0f; \
                input80.sd = 0.0f; \
            } \
            if (inputColIndex + 14 < 0 || inputColIndex + 14 >= inputWidth) { \
                input00.se = 0.0f; \
                input40.se = 0.0f; \
                input80.se = 0.0f; \
            } \
            if (inputColIndex + 15 < 0 || inputColIndex + 15 >= inputWidth) { \
                input00.sf = 0.0f; \
                input40.sf = 0.0f; \
                input80.sf = 0.0f; \
            } \
            DATA_T8 out8 = (DATA_T8)(bias[channel]); \
            DATA_T8 out00 = input00.lo * k.s0 + (DATA_T8)(input00.lo.hi, input00.hi.lo) * k.s1 + \
                          input00.hi * k.s2; \
            DATA_T8 out10 = input40.lo * k.s3 + (DATA_T8)(input40.lo.hi, input40.hi.lo) * k.s4 + \
                          input40.hi * k.s5; \
            DATA_T8 out20 = input80.lo * k.s6 + (DATA_T8)(input80.lo.hi, input80.hi.lo) * k.s7 + \
                          input80.hi * k.s8; \
            out8 += out00 + out10 + out20; \
            if (g2 + 7 < topWidth) { \
                vstore8(ACT_VEC_F(DATA_T8, out8), 0, output + outputIndex); \
            } else if (g2 + 6 < topWidth) { \
                vstore4(ACT_VEC_F(DATA_T4, out8.s0123), 0, output + outputIndex); \
                vstore3(ACT_VEC_F(DATA_T3, out8.s456), 0, output + outputIndex + 4); \
            } else if (g2 + 5 < topWidth) { \
                vstore4(ACT_VEC_F(DATA_T4, out8.s0123), 0, output + outputIndex); \
                vstore2(ACT_VEC_F(DATA_T2, out8.s45), 0, output + outputIndex + 4); \
            } else if (g2 + 4 < topWidth) { \
                vstore4(ACT_VEC_F(DATA_T4, out8.s0123), 0, output + outputIndex); \
                output[outputIndex + 4] = ACT_VEC_F(DATA_T, out8.s4); \
            } else if (g2 + 3 < topWidth) { \
                vstore4(ACT_VEC_F(DATA_T4, out8.s0123), 0, output + outputIndex); \
            } else if (g2 + 2 < topWidth) { \
                vstore3(ACT_VEC_F(DATA_T3, out8.s012), 0, output + outputIndex); \
            } else if (g2 + 1 < topWidth) { \
                vstore2(ACT_VEC_F(DATA_T2, out8.s01), 0, output + outputIndex); \
            } else { \
                output[outputIndex] = ACT_VEC_F(DATA_T, out8.s0); \
            } \
        }

#define DEPTHWISE_CONV_3X3S1_PAD_MERGE_FP16(input, weight, bias, output, inputHeight, inputWidth, topHeight, \
                                            topWidth, topChannel, padH, padW) \
        int g0 = get_global_id(0); \
        int g1 = get_global_id(1); \
        int g2 = get_global_id(2) * 2; \
        int inputRowIndex = g1 - padH; \
        int inputColIndex = g2 - padW; \
        int channel = g0 % topChannel; \
        int inputIndex = g0 * inputHeight * inputWidth + inputRowIndex * inputWidth + inputColIndex; \
        int outputIndex = g0 * topHeight * topWidth + g1 * topWidth + g2; \
        if (g2 < topWidth) {  \
            DATA_T16 k = vload16(0, weight + channel * 9); \
            DATA_T4 input00 = 0.0f; \
            DATA_T4 input10 = 0.0f; \
            DATA_T4 input20 = 0.0f; \
            if (inputRowIndex >= 0 && inputRowIndex < inputHeight) { \
                input00 = vload4(0, input + inputIndex); \
            } \
            if (inputRowIndex + 1 >= 0 && inputRowIndex + 1 < inputHeight) { \
                input10 = vload4(0, input + inputIndex + inputWidth); \
            } \
            if (inputRowIndex + 2 >= 0 && inputRowIndex + 2 < inputHeight) { \
                input20 = vload4(0, input + inputIndex + 2 * inputWidth); \
            } \
            if (inputColIndex < 0 || inputColIndex >= inputWidth) { \
                input00.s0 = 0.0f; \
                input10.s0 = 0.0f; \
                input20.s0 = 0.0f; \
            } \
            if (inputColIndex + 1 < 0 || inputColIndex + 1 >= inputWidth) { \
                input00.s1 = 0.0f; \
                input10.s1 = 0.0f; \
                input20.s1 = 0.0f; \
            } \
            if (inputColIndex + 2 < 0 || inputColIndex + 2 >= inputWidth) { \
                input00.s2 = 0.0f; \
                input10.s2 = 0.0f; \
                input20.s2 = 0.0f; \
            } \
            if (inputColIndex + 3 < 0 || inputColIndex + 3 >= inputWidth) { \
                input00.s3 = 0.0f; \
                input10.s3 = 0.0f; \
                input20.s3 = 0.0f; \
            } \
            DATA_T2 out2 = (DATA_T2)(bias[channel]); \
            DATA_T4 out00 = k.s0123 * input00; \
            out00 += k.s3456 * input10; \
            out00 += k.s6789 * input20; \
            out2.s0 += out00.s0 + out00.s1 + out00.s2; \
            out00 = k.s0123 * (DATA_T4)(input00.s123, 0.0f); \
            out00 += k.s3456 * (DATA_T4)(input10.s123, 0.0f); \
            out00 += k.s6789 * (DATA_T4)(input20.s123, 0.0f); \
            out2.s1 += out00.s0 + out00.s1 + out00.s2; \
            if (g2 + 1 < topWidth) { \
                    vstore2(ACT_VEC_F(DATA_T2, out2), 0, output + outputIndex); \
            } else { \
                output[outputIndex] = ACT_VEC_F(DATA_T, out2.s0); \
            } \
        }

#define DEPTHWISE_CONV_3X3S1_PAD_MERGE_FP32(input, weight, bias, output, inputHeight, inputWidth, topHeight, \
                                            topWidth, topChannel, padH, padW) \
        int g0 = get_global_id(0); \
        int g1 = get_global_id(1); \
        int g2 = get_global_id(2) * 2; \
        int inputRowIndex = g1 - padH; \
        int inputColIndex = g2 - padW; \
        int channel = g0 % topChannel; \
        int inputIndex = g0 * inputHeight * inputWidth + inputRowIndex * inputWidth + inputColIndex; \
        int outputIndex = g0 * topHeight * topWidth + g1 * topWidth + g2; \
        if (g2 < topWidth) {  \
            DATA_T16 k = vload16(0, weight + channel * 9); \
            DATA_T4 input00 = 0.0f; \
            DATA_T4 input10 = 0.0f; \
            DATA_T4 input20 = 0.0f; \
            if (inputRowIndex >= 0 && inputRowIndex < inputHeight) { \
                input00 = vload4(0, input + inputIndex); \
            } \
            if (inputRowIndex + 1 >= 0 && inputRowIndex + 1 < inputHeight) { \
                input10 = vload4(0, input + inputIndex + inputWidth); \
            } \
            if (inputRowIndex + 2 >= 0 && inputRowIndex + 2 < inputHeight) { \
                input20 = vload4(0, input + inputIndex + 2 * inputWidth); \
            } \
            if (inputColIndex < 0 || inputColIndex >= inputWidth) { \
                input00.s0 = 0.0f; \
                input10.s0 = 0.0f; \
                input20.s0 = 0.0f; \
            } \
            if (inputColIndex + 1 < 0 || inputColIndex + 1 >= inputWidth) { \
                input00.s1 = 0.0f; \
                input10.s1 = 0.0f; \
                input20.s1 = 0.0f; \
            } \
            if (inputColIndex + 2 < 0 || inputColIndex + 2 >= inputWidth) { \
                input00.s2 = 0.0f; \
                input10.s2 = 0.0f; \
                input20.s2 = 0.0f; \
            } \
            if (inputColIndex + 3 < 0 || inputColIndex + 3 >= inputWidth) { \
                input00.s3 = 0.0f; \
                input10.s3 = 0.0f; \
                input20.s3 = 0.0f; \
            } \
            DATA_T2 out2 = (DATA_T2)(bias[channel]); \
            out2 += k.s0 * input00.s01; \
            out2 += k.s1 * input00.s12; \
            out2 += k.s2 * input00.s23; \
            out2 += k.s3 * input10.s01; \
            out2 += k.s4 * input10.s12; \
            out2 += k.s5 * input10.s23; \
            out2 += k.s6 * input20.s01; \
            out2 += k.s7 * input20.s12; \
            out2 += k.s8 * input20.s23; \
            if (g2 + 1 < topWidth) { \
                vstore2(ACT_VEC_F(DATA_T2, out2), 0, output + outputIndex); \
            } else { \
                output[outputIndex] = ACT_VEC_F(DATA_T, out2.s0); \
            } \
        }

#define DEPTHWISE_CONV_3X3S2_PAD_MERGE(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                       topChannel, padH, padW) \
        int g0 = get_global_id(0); \
        int g1 = get_global_id(1); \
        int g2 = get_global_id(2) * 3; \
        int inputRowIndex = g1 * 2 - padH; \
        int inputColIndex = g2 * 2 - padW; \
        int channel = g0 % topChannel; \
        int inputIndex = g0 * inputHeight * inputWidth + inputRowIndex * inputWidth + inputColIndex; \
        int outputIndex = g0 * topHeight * topWidth + g1 * topWidth + g2; \
        if (g2 < topWidth) {  \
            DATA_T16 k = vload16(0, weight + channel * 9); \
            DATA_T8 input00 = 0.0f; \
            DATA_T8 input10 = 0.0f; \
            DATA_T8 input20 = 0.0f; \
            if (inputRowIndex >= 0 && inputRowIndex < inputHeight) { \
                input00 = vload8(0, input + inputIndex); \
            } \
            if (inputRowIndex + 1 >= 0 && inputRowIndex + 1 < inputHeight) { \
                input10 = vload8(0, input + inputIndex + inputWidth); \
            } \
            if (inputRowIndex + 2 >= 0 && inputRowIndex + 2 < inputHeight) { \
                input20 = vload8(0, input + inputIndex + 2 * inputWidth); \
            } \
            if (inputColIndex < 0 || inputColIndex >= inputWidth) { \
                input00.s0 = 0.0f; \
                input10.s0 = 0.0f; \
                input20.s0 = 0.0f; \
            } \
            if (inputColIndex + 1 < 0 || inputColIndex + 1 >= inputWidth) { \
                input00.s1 = 0.0f; \
                input10.s1 = 0.0f; \
                input20.s1 = 0.0f; \
            } \
            if (inputColIndex + 2 < 0 || inputColIndex + 2 >= inputWidth) { \
                input00.s2 = 0.0f; \
                input10.s2 = 0.0f; \
                input20.s2 = 0.0f; \
            } \
            if (inputColIndex + 3 < 0 || inputColIndex + 3 >= inputWidth) { \
                input00.s3 = 0.0f; \
                input10.s3 = 0.0f; \
                input20.s3 = 0.0f; \
            } \
            if (inputColIndex + 4 < 0 || inputColIndex + 4 >= inputWidth) { \
                input00.s4 = 0.0f; \
                input10.s4 = 0.0f; \
                input20.s4 = 0.0f; \
            } \
            if (inputColIndex + 5 < 0 || inputColIndex + 5 >= inputWidth) { \
                input00.s5 = 0.0f; \
                input10.s5 = 0.0f; \
                input20.s5 = 0.0f; \
            } \
            if (inputColIndex + 6 < 0 || inputColIndex + 6 >= inputWidth) { \
                input00.s6 = 0.0f; \
                input10.s6 = 0.0f; \
                input20.s6 = 0.0f; \
            } \
            DATA_T3 out3 = (DATA_T3)(bias[channel]); \
            DATA_T3 out00; \
            out00 = k.s012 * input00.s012; \
            out00 += k.s345 * input10.s012; \
            out00 += k.s678 * input20.s012; \
            out3.s0 += out00.s0 + out00.s1 + out00.s2; \
            out00 = k.s012 * input00.s234; \
            out00 += k.s345 * input10.s234; \
            out00 += k.s678 * input20.s234; \
            out3.s1 += out00.s0 + out00.s1 + out00.s2; \
            out00 = k.s012 * input00.s456; \
            out00 += k.s345 * input10.s456; \
            out00 += k.s678 * input20.s456; \
            out3.s2 += out00.s0 + out00.s1 + out00.s2; \
            if (g2 + 2 < topWidth) { \
                vstore3(ACT_VEC_F(DATA_T3, out3), 0, output + outputIndex); \
            } else if (g2 + 1 < topWidth) { \
                vstore2(ACT_VEC_F(DATA_T2, out3.s01), 0, output + outputIndex); \
            } else { \
                output[outputIndex] = ACT_VEC_F(DATA_T, out3.s0); \
            } \
        }

#define DEPTHWISE_CONV_3X3S2_FP16(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                  topChannel, depth_multiplier) \
        int g0 = get_global_id(0); \
        int g1 = get_global_id(1); \
        int g2 = get_global_id(2); \
        if (g2 < topHeight * topWidth) { \
            DATA_T4 input0; \
            DATA_T4 input1; \
            DATA_T4 input2; \
            DATA_T4 w0; \
            DATA_T4 w1; \
            DATA_T4 w2; \
            DATA_T4 out0; \
            int inputX = (g2 % topWidth) * 2; \
            int inputY = (g2 / topWidth) * 2; \
            for (int dep = 0; dep < depth_multiplier; dep++) { \
                int out_c = dep + g1 * depth_multiplier; \
                int inputIndex = (g0 * get_global_size(1) + g1) * inputHeight * inputWidth + \
                                 (inputY)*inputWidth + inputX; \
                int outputIndex = \
                    (g0 * get_global_size(1) * depth_multiplier + out_c) * topHeight * topWidth + \
                    g2; \
                input0 = vload4(0, input + inputIndex); \
                input1 = vload4(0, input + inputIndex + inputWidth); \
                input2 = vload4(0, input + inputIndex + 2 * inputWidth); \
                w0 = vload4(0, weight + out_c * 9); \
                w1 = vload4(0, weight + out_c * 9 + 3); \
                w2 = vload4(0, weight + out_c * 9 + 6); \
                out0 = input0 * w0; \
                out0 += input1 * w1; \
                out0 += input2 * w2; \
                output[outputIndex] = \
                    ACT_VEC_F(DATA_T, out0.s0 + out0.s1 + out0.s2 + bias[out_c]); \
            } \
        }

#define DEPTHWISE_CONV_3X3S2_FP32(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                  topChannel, depth_multiplier) \
    int g0 = get_global_id(0); \
    int g1 = get_global_id(1); \
    int g2 = get_global_id(2); \
    DATA_T4 input0; \
    DATA_T4 input1; \
    DATA_T4 input2; \
    DATA_T4 w0; \
    DATA_T4 w1; \
    DATA_T4 w2; \
    DATA_T4 out0; \
    DATA_T4 out1; \
    DATA_T4 out2; \
    if (g2 < topHeight * topWidth) { \
        int inputX = (g2 % topWidth) * 2; \
        int inputY = (g2 / topWidth) * 2; \
        for (int dep = 0; dep < depth_multiplier; dep++) { \
            int out_c = dep + g1 * depth_multiplier; \
            int inputIndex = (g0 * get_global_size(1) + g1) * inputHeight * inputWidth + \
                                (inputY)*inputWidth + inputX; \
            int outputIndex = \
                (g0 * get_global_size(1) * depth_multiplier + out_c) * topHeight * topWidth + \
                g2; \
            output[outputIndex] = bias[out_c]; \
            input0 = vload4(0, input + inputIndex); \
            input1 = vload4(0, input + inputIndex + inputWidth); \
            input2 = vload4(0, input + inputIndex + 2 * inputWidth); \
            w0 = vload4(0, weight + out_c * 9); \
            w1 = vload4(0, weight + out_c * 9 + 3); \
            w2 = vload4(0, weight + out_c * 9 + 6); \
            out0 = input0 * w0; \
            out1 = input1 * w1; \
            out2 = input2 * w2; \
            output[outputIndex] += out0.s0 + out0.s1 + out0.s2 + out1.s0 + out1.s1 + out1.s2 + \
                                    out2.s0 + out2.s1 + out2.s2; \
        } \
    }

#define DW_TFLITE_3X3(src_data, filters, dst_data, dst_size_x, dst_size_y, dst_size_z) \
        int X = get_global_id(0) * 2; \
        int Y = get_global_id(1) * 2; \
        int Z = get_global_id(2); \
        DATA_T4 r0 = (DATA_T4)(0.0f); \
        DATA_T4 r1 = (DATA_T4)(0.0f); \
        DATA_T4 r2 = (DATA_T4)(0.0f); \
        DATA_T4 r3 = (DATA_T4)(0.0f); \
        if (X >= dst_size_x || Y >= dst_size_y || Z >= dst_size_z) \
            return; \
        __global DATA_T4 *f = filters + Z * 10; \
        DATA_T4 s0; \
        DATA_T4 s1; \
        DATA_T4 s2; \
        DATA_T4 s3; \
        { \
            s0 = read_imageh(src_data, smp_zero, (int2)((X - 1), (Y - 1) * dst_size_z + (Z))); \
            s1 = read_imageh(src_data, smp_zero, (int2)((X), (Y - 1) * dst_size_z + (Z))); \
            s2 = read_imageh(src_data, smp_zero, (int2)((X + 1), (Y - 1) * dst_size_z + (Z))); \
            s3 = read_imageh(src_data, smp_zero, (int2)((X + 2), (Y - 1) * dst_size_z + (Z))); \
            r0 = fma(f[0], s0, r0); \
            r0 = fma(f[1], s1, r0); \
            r1 = fma(f[0], s1, r1); \
            r0 = fma(f[2], s2, r0); \
            r1 = fma(f[1], s2, r1); \
            r1 = fma(f[2], s3, r1); \
        } \
        { \
            s0 = read_imageh(src_data, smp_zero, (int2)((X - 1), (Y)*dst_size_z + (Z))); \
            s1 = read_imageh(src_data, smp_zero, (int2)((X), (Y)*dst_size_z + (Z))); \
            s2 = read_imageh(src_data, smp_zero, (int2)((X + 1), (Y)*dst_size_z + (Z))); \
            s3 = read_imageh(src_data, smp_zero, (int2)((X + 2), (Y)*dst_size_z + (Z))); \
             r0 = fma(f[3], s0, r0); \
             r2 = fma(f[0], s0, r2); \
             r0 = fma(f[4], s1, r0); \
             r1 = fma(f[3], s1, r1); \
             r2 = fma(f[1], s1, r2); \
             r3 = fma(f[0], s1, r3); \
             r0 = fma(f[5], s2, r0); \
             r1 = fma(f[4], s2, r1); \
             r2 = fma(f[2], s2, r2); \
             r3 = fma(f[1], s2, r3); \
             r1 = fma(f[5], s3, r1); \
             r3 = fma(f[2], s3, r3); \
        } \
        { \
            s0 = read_imageh(src_data, smp_zero, (int2)((X - 1), (Y + 1) * dst_size_z + (Z))); \
            s1 = read_imageh(src_data, smp_zero, (int2)((X), (Y + 1) * dst_size_z + (Z))); \
            s2 = read_imageh(src_data, smp_zero, (int2)((X + 1), (Y + 1) * dst_size_z + (Z))); \
            s3 = read_imageh(src_data, smp_zero, (int2)((X + 2), (Y + 1) * dst_size_z + (Z))); \
              r0 = fma(f[6], s0, r0); \
              r2 = fma(f[3], s0, r2); \
              r0 = fma(f[7], s1, r0); \
              r1 = fma(f[6], s1, r1); \
              r2 = fma(f[4], s1, r2); \
              r3 = fma(f[3], s1, r3); \
              r0 = fma(f[8], s2, r0); \
              r1 = fma(f[7], s2, r1); \
              r2 = fma(f[5], s2, r2); \
              r3 = fma(f[4], s2, r3); \
              r1 = fma(f[8], s3, r1); \
              r3 = fma(f[5], s3, r3); \
        } \
        { \
            s0 = read_imageh(src_data, smp_zero, (int2)((X - 1), (Y + 2) * dst_size_z + (Z))); \
            s1 = read_imageh(src_data, smp_zero, (int2)((X), (Y + 2) * dst_size_z + (Z))); \
            s2 = read_imageh(src_data, smp_zero, (int2)((X + 1), (Y + 2) * dst_size_z + (Z))); \
            s3 = read_imageh(src_data, smp_zero, (int2)((X + 2), (Y + 2) * dst_size_z + (Z))); \
            r2 = fma(f[6], s0, r2); \
            r2 = fma(f[7], s1, r2); \
            r3 = fma(f[6], s1, r3); \
            r2 = fma(f[8], s2, r2); \
            r3 = fma(f[7], s2, r3); \
            r3 = fma(f[8], s3, r3); \
        } \
        r0 += CONVERT_TO_DATA_T4(f[9]); \
        r1 += CONVERT_TO_DATA_T4(f[9]); \
        r2 += CONVERT_TO_DATA_T4(f[9]); \
        r3 += CONVERT_TO_DATA_T4(f[9]); \
        if (X + 0 < dst_size_x && Y + 0 < dst_size_y) { \
            DATA_T4 result = CONVERT_TO_DATA_T4(r0); \
            int2 address = (int2)((X + 0), (Y + 0) * dst_size_z + (Z)); \
            result = ACT_VEC_F(DATA_T4, result); \
            write_imageh(dst_data, (int2)((X + 0), (Y + 0) * dst_size_z + (Z)), result); \
        } \
        if (X + 1 < dst_size_x && Y + 0 < dst_size_y) { \
            DATA_T4 result = CONVERT_TO_DATA_T4(r1); \
            result = ACT_VEC_F(DATA_T4, result); \
            write_imageh(dst_data, (int2)((X + 1), (Y + 0) * dst_size_z + (Z)), result); \
        } \
        if (X + 0 < dst_size_x && Y + 1 < dst_size_y) { \
            DATA_T4 result = CONVERT_TO_DATA_T4(r2); \
            result = ACT_VEC_F(DATA_T4, result); \
            write_imageh(dst_data, (int2)((X + 0), (Y + 1) * dst_size_z + (Z)), result); \
        } \
        if (X + 1 < dst_size_x && Y + 1 < dst_size_y) { \
            DATA_T4 result = CONVERT_TO_DATA_T4(r3); \
            result = ACT_VEC_F(DATA_T4, result); \
            write_imageh(dst_data, (int2)((X + 1), (Y + 1) * dst_size_z + (Z)), result); \
        }

#define DW_TFLITE_3X3_REARRANGE_W(weights, biases, dst, input_channel, src_depth) \
        DATA_T4 b = (DATA_T4)0.0f; \
        int s = get_global_id(0); \
        for (int y = 0; y < 3; ++y) { \
            for (int x = 0; x < 3; ++x) { \
                DATA_T4 w = (DATA_T4)0.0f; \
                const int s_ch = (s * 4 * 9 + y * 3 + x); \
                if (s * 4 < input_channel) \
                    w.s0 = weights[s_ch]; \
                if (s * 4 + 1 < input_channel) \
                    w.s1 = weights[s_ch + 9]; \
                if (s * 4 + 2 < input_channel) \
                    w.s2 = weights[s_ch + 18]; \
                if (s * 4 + 3 < input_channel) \
                    w.s3 = weights[s_ch + 27]; \
                vstore4(w, 0, dst + s * (3 * 3 * 4 + 4) + (y * 3 + x) * 4); \
            } \
        } \
        const int dst_ch = (s * 4); \
        b.s0 = dst_ch >= input_channel ? 0.0f : biases[dst_ch]; \
        b.s1 = (dst_ch + 1) >= input_channel ? 0.0f : biases[dst_ch + 1]; \
        b.s2 = (dst_ch + 2) >= input_channel ? 0.0f : biases[dst_ch + 2]; \
        b.s3 = (dst_ch + 3) >= input_channel ? 0.0f : biases[dst_ch + 3]; \
        vstore4(b, 0, dst + s * (3 * 3 * 4 + 4) + 36);

#define DW_TFLITE(src_data, filters, biases, dst_data, kernel_size_x, kernel_size_y, stride_x, stride_y, \
                  padding_x, padding_y, dilation_x, dilation_y, src_size_x, src_size_y, src_size_z, dst_size_x, \
                  dst_size_y, dst_size_z) \
        int X = get_global_id(0); \
        int Y = get_global_id(1); \
        int Z = get_global_id(2); \
        if (X >= dst_size_x || Y >= dst_size_y || Z >= dst_size_z) \
            return; \
        DATA_T4 r = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f); \
        int x_offseted = X * stride_x + padding_x; \
        int y_offseted = Y * stride_y + padding_y; \
        int fx_c = Z * kernel_size_x * kernel_size_y; \
        for (int ky = 0; ky < kernel_size_y; ++ky) { \
            int y_c = y_offseted + ky * dilation_y; \
            for (int kx = 0; kx < kernel_size_x; ++kx) { \
                int x_c = x_offseted + kx * dilation_x; \
                DATA_T4 src_final = read_imageh(src_data, smp_zero, (int2)((x_c), (y_c)*src_size_z + (Z))); \
                DATA_T4 f = filters[fx_c]; \
                fx_c++; \
                r = fma(src_final, f, r); \
            } \
        } \
        DATA_T4 bias_val = biases[Z]; \
        DATA_T4 res0 = CONVERT_TO_DATA_T4(r) + bias_val; \
        res0 = ACT_VEC_F(DATA_T4, res0); \
        write_imageh(dst_data, (int2)((X), (Y)*dst_size_z + (Z)), res0);

#define DW_TFLITE_REARRANGE_W(weights, dst, kernel_y, kernel_x, dst_channels, dst_depth) \
        int d = get_global_id(0); \
        for (int y = 0; y < kernel_y; ++y) { \
            for (int x = 0; x < kernel_x; ++x) { \
                DATA_T4 w = (DATA_T4)0.0f; \
                const int d_ch = (d * 4 * kernel_y * kernel_x + y * kernel_x + x); \
                if (d * 4 < dst_channels) \
                    w.s0 = weights[d_ch]; \
                if (d * 4 + 1 < dst_channels) \
                    w.s1 = weights[d_ch + kernel_y * kernel_x]; \
                if (d * 4 + 2 < dst_channels) \
                    w.s2 = weights[d_ch + 2 * kernel_y * kernel_x]; \
                if (d * 4 + 3 < dst_channels) \
                    w.s3 = weights[d_ch + 3 * kernel_y * kernel_x]; \
                vstore4(w, 0, dst + d * kernel_y * kernel_x * 4 + (y * kernel_x + x) * 4); \
            } \
        }

#define DW_CONV_COMMON_PARAMS __global const DATA_T *input, __global const DATA_T *weight, \
    __global const DATA_T *bias, __global DATA_T *output, unsigned int inputHeight, unsigned int inputWidth
#define DW_CONV_4P_PARAMS DW_CONV_COMMON_PARAMS, unsigned int topHeight, unsigned int topWidth, unsigned int topChannel
/********  FP16 KERNELS ********/
#define DATA_T half
#define DATA_T2 half2
#define DATA_T3 half3
#define DATA_T4 half4
#define DATA_T8 half8
#define DATA_T16 half16
#define CONVERT_TO_DATA_T4(x) convert_half4(x)

ADD_SINGLE_KERNEL(dw_tflite_3x3_rearrange_w_FP16, (__global DATA_T *weights,
                                                 __global DATA_T *biases,
                                                 __global DATA_T *dst,
                                                 int input_channel,
                                                 int src_depth) {
    DW_TFLITE_3X3_REARRANGE_W(weights, biases, dst, input_channel, src_depth)
})

ADD_SINGLE_KERNEL(dw_tflite_rearrange_w_FP16, (__global DATA_T *weights,
                                             __global DATA_T *dst,
                                             int kernel_y,
                                             int kernel_x,
                                             int dst_channels,
                                             int dst_depth) {
    DW_TFLITE_REARRANGE_W(weights, dst, kernel_y, kernel_x, dst_channels, dst_depth)
})


// kernels without activation func
#define ACT_VEC_F(VEC_T, x) x
ADD_SINGLE_KERNEL(depthwise_conv_unequal_FP16, (__global const DATA_T *bias,
                                              __global DATA_T *output,
                                              unsigned int start) {
    DEPTHWISE_CONV_UNEQUAL(bias, output, start)
})

ADD_SINGLE_KERNEL(depthwise_conv_FP16, (DW_CONV_COMMON_PARAMS, unsigned int kernelHeight, unsigned int kernelWidth, \
                                        unsigned int strideHeight, unsigned int strideWidth, unsigned int topHeight, \
                                        unsigned int topWidth, unsigned int depth_multiplier) {
    DEPTHWISE_CONV(input, weight, bias, output, inputHeight, inputWidth, kernelHeight, kernelWidth, strideHeight, \
                   strideWidth, topHeight, topWidth, depth_multiplier)
})

ADD_SINGLE_KERNEL(depthwise_conv_3x3s1_4P_FP16, (DW_CONV_4P_PARAMS, unsigned int depth_multiplier) {
    DEPTHWISE_CONV_3X3S1_4P(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, topChannel, \
                            depth_multiplier)
})

ADD_SINGLE_KERNEL(depthwise_conv_3x3d2s1_pad_merge_FP16, (DW_CONV_4P_PARAMS, unsigned int padH, unsigned int padW) {
    DEPTHWISE_CONV_3X3D2S1_PAD_MERGE(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                     topChannel, padH, padW)
})

ADD_SINGLE_KERNEL(depthwise_conv_3x3d4s1_pad_merge_FP16, (DW_CONV_4P_PARAMS, unsigned int padH, unsigned int padW) {
    DEPTHWISE_CONV_3X3D4S1_PAD_MERGE(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                     topChannel, padH, padW)
})

ADD_SINGLE_KERNEL(depthwise_conv_3x3s1_pad_merge_FP16, (DW_CONV_4P_PARAMS, unsigned int padH, unsigned int padW) {
    DEPTHWISE_CONV_3X3S1_PAD_MERGE_FP16(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                        topChannel, padH, padW)
})

ADD_SINGLE_KERNEL(depthwise_conv_3x3s2_pad_merge_FP16, (DW_CONV_4P_PARAMS, unsigned int padH, unsigned int padW) {
    DEPTHWISE_CONV_3X3S2_PAD_MERGE(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                   topChannel, padH, padW)
})

ADD_SINGLE_KERNEL(depthwise_conv_3x3s2_FP16, (DW_CONV_4P_PARAMS, unsigned int depth_multiplier) {
    DEPTHWISE_CONV_3X3S2_FP16(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                              topChannel, depth_multiplier)
})

ADD_SINGLE_KERNEL(dw_tflite_3x3_FP16, (__read_only image2d_t src_data,
                                     __global DATA_T4 *filters,
                                     __write_only image2d_t dst_data,
                                     int dst_size_x,
                                     int dst_size_y,
                                     int dst_size_z) {
    DW_TFLITE_3X3(src_data, filters, dst_data, dst_size_x, dst_size_y, dst_size_z)
})

ADD_SINGLE_KERNEL(dw_tflite_FP16, (__read_only image2d_t src_data,
                                 __global DATA_T4 *filters,
                                 __global DATA_T4 *biases,
                                 __write_only image2d_t dst_data,
                                 int kernel_size_x,
                                 int kernel_size_y,
                                 int stride_x,
                                 int stride_y,
                                 int padding_x,
                                 int padding_y,
                                 int dilation_x,
                                 int dilation_y,
                                 int src_size_x,
                                 int src_size_y,
                                 int src_size_z,
                                 int dst_size_x,
                                 int dst_size_y,
                                 int dst_size_z) {
    DW_TFLITE(src_data, filters, biases, dst_data, kernel_size_x, kernel_size_y, stride_x, stride_y, padding_x, \
              padding_y, dilation_x, dilation_y, src_size_x, src_size_y, src_size_z, dst_size_x, dst_size_y, dst_size_z)
})

#undef ACT_VEC_F  // NONE

// apply RELU to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELUdepthwise_conv_unequal_FP16, (__global const DATA_T *bias,
                                              __global DATA_T *output,
                                              unsigned int start) {
    DEPTHWISE_CONV_UNEQUAL(bias, output, start)
})

ADD_SINGLE_KERNEL(RELUdepthwise_conv_FP16, (DW_CONV_COMMON_PARAMS, unsigned int kernelHeight, \
                                            unsigned int kernelWidth, unsigned int strideHeight, \
                                            unsigned int strideWidth, unsigned int topHeight, unsigned int topWidth, \
                                            unsigned int depth_multiplier) {
    DEPTHWISE_CONV(input, weight, bias, output, inputHeight, inputWidth, kernelHeight, kernelWidth, strideHeight, \
                   strideWidth, topHeight, topWidth, depth_multiplier)
})

ADD_SINGLE_KERNEL(RELUdepthwise_conv_3x3s1_4P_FP16, (DW_CONV_4P_PARAMS, unsigned int depth_multiplier) {
    DEPTHWISE_CONV_3X3S1_4P(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, topChannel, \
                            depth_multiplier)
})

ADD_SINGLE_KERNEL(RELUdepthwise_conv_3x3d2s1_pad_merge_FP16, (DW_CONV_4P_PARAMS, unsigned int padH, unsigned int padW) {
    DEPTHWISE_CONV_3X3D2S1_PAD_MERGE(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                     topChannel, padH, padW)
})

ADD_SINGLE_KERNEL(RELUdepthwise_conv_3x3d4s1_pad_merge_FP16, (DW_CONV_4P_PARAMS, unsigned int padH, unsigned int padW) {
    DEPTHWISE_CONV_3X3D4S1_PAD_MERGE(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                     topChannel, padH, padW)
})

ADD_SINGLE_KERNEL(RELUdepthwise_conv_3x3s1_pad_merge_FP16, (DW_CONV_4P_PARAMS, unsigned int padH, unsigned int padW) {
    DEPTHWISE_CONV_3X3S1_PAD_MERGE_FP16(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                        topChannel, padH, padW)
})

ADD_SINGLE_KERNEL(RELUdepthwise_conv_3x3s2_pad_merge_FP16, (DW_CONV_4P_PARAMS, unsigned int padH, unsigned int padW) {
    DEPTHWISE_CONV_3X3S2_PAD_MERGE(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                   topChannel, padH, padW)
})

ADD_SINGLE_KERNEL(RELUdepthwise_conv_3x3s2_FP16, (DW_CONV_4P_PARAMS, unsigned int depth_multiplier) {
    DEPTHWISE_CONV_3X3S2_FP16(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                topChannel, depth_multiplier)
})

ADD_SINGLE_KERNEL(RELUdw_tflite_3x3_FP16, (__read_only image2d_t src_data,
                                     __global DATA_T4 *filters,
                                     __write_only image2d_t dst_data,
                                     int dst_size_x,
                                     int dst_size_y,
                                     int dst_size_z) {
    DW_TFLITE_3X3(src_data, filters, dst_data, dst_size_x, dst_size_y, dst_size_z)
})

ADD_SINGLE_KERNEL(RELUdw_tflite_FP16, (__read_only image2d_t src_data,
                                 __global DATA_T4 *filters,
                                 __global DATA_T4 *biases,
                                 __write_only image2d_t dst_data,
                                 int kernel_size_x,
                                 int kernel_size_y,
                                 int stride_x,
                                 int stride_y,
                                 int padding_x,
                                 int padding_y,
                                 int dilation_x,
                                 int dilation_y,
                                 int src_size_x,
                                 int src_size_y,
                                 int src_size_z,
                                 int dst_size_x,
                                 int dst_size_y,
                                 int dst_size_z) {
    DW_TFLITE(src_data, filters, biases, dst_data, kernel_size_x, kernel_size_y, stride_x, stride_y, padding_x, \
              padding_y, dilation_x, dilation_y, src_size_x, src_size_y, src_size_z, dst_size_x, dst_size_y, dst_size_z)
})

#undef ACT_VEC_F  // RELU

// apply RELU6 to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU6_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELU6depthwise_conv_unequal_FP16, (__global const DATA_T *bias,
                                              __global DATA_T *output,
                                              unsigned int start) {
    DEPTHWISE_CONV_UNEQUAL(bias, output, start)
})

ADD_SINGLE_KERNEL(RELU6depthwise_conv_FP16, (DW_CONV_COMMON_PARAMS, unsigned int kernelHeight, \
                                             unsigned int kernelWidth, unsigned int strideHeight, \
                                             unsigned int strideWidth, unsigned int topHeight, \
                                             unsigned int topWidth, unsigned int depth_multiplier) {
    DEPTHWISE_CONV(input, weight, bias, output, inputHeight, inputWidth, kernelHeight, kernelWidth, strideHeight, \
                    strideWidth, topHeight, topWidth, depth_multiplier)
})

ADD_SINGLE_KERNEL(RELU6depthwise_conv_3x3s1_4P_FP16, (DW_CONV_4P_PARAMS, unsigned int depth_multiplier) {
    DEPTHWISE_CONV_3X3S1_4P(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, topChannel, \
                            depth_multiplier)
})

ADD_SINGLE_KERNEL(RELU6depthwise_conv_3x3d2s1_pad_merge_FP16, (DW_CONV_4P_PARAMS, \
                                                               unsigned int padH, unsigned int padW) {
    DEPTHWISE_CONV_3X3D2S1_PAD_MERGE(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                      topChannel, padH, padW)
})

ADD_SINGLE_KERNEL(RELU6depthwise_conv_3x3d4s1_pad_merge_FP16, (DW_CONV_4P_PARAMS, \
                                                               unsigned int padH, unsigned int padW) {
    DEPTHWISE_CONV_3X3D4S1_PAD_MERGE(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                topChannel, padH, padW)
})

ADD_SINGLE_KERNEL(RELU6depthwise_conv_3x3s1_pad_merge_FP16, (DW_CONV_4P_PARAMS, unsigned int padH, unsigned int padW) {
    DEPTHWISE_CONV_3X3S1_PAD_MERGE_FP16(input, weight, bias, output, inputHeight, inputWidth, topHeight, \
                                           topWidth, topChannel, padH, padW)
})

ADD_SINGLE_KERNEL(RELU6depthwise_conv_3x3s2_pad_merge_FP16, (DW_CONV_4P_PARAMS, unsigned int padH, unsigned int padW) {
    DEPTHWISE_CONV_3X3S2_PAD_MERGE(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                    topChannel, padH, padW)
})

ADD_SINGLE_KERNEL(RELU6depthwise_conv_3x3s2_FP16, (DW_CONV_4P_PARAMS, unsigned int depth_multiplier) {
    DEPTHWISE_CONV_3X3S2_FP16(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                               topChannel, depth_multiplier)
})

ADD_SINGLE_KERNEL(RELU6dw_tflite_3x3_FP16, (__read_only image2d_t src_data,
                                     __global DATA_T4 *filters,
                                     __write_only image2d_t dst_data,
                                     int dst_size_x,
                                     int dst_size_y,
                                     int dst_size_z) {
    DW_TFLITE_3X3(src_data, filters, dst_data, dst_size_x, dst_size_y, dst_size_z)
})

ADD_SINGLE_KERNEL(RELU6dw_tflite_FP16, (__read_only image2d_t src_data,
                                 __global DATA_T4 *filters,
                                 __global DATA_T4 *biases,
                                 __write_only image2d_t dst_data,
                                 int kernel_size_x,
                                 int kernel_size_y,
                                 int stride_x,
                                 int stride_y,
                                 int padding_x,
                                 int padding_y,
                                 int dilation_x,
                                 int dilation_y,
                                 int src_size_x,
                                 int src_size_y,
                                 int src_size_z,
                                 int dst_size_x,
                                 int dst_size_y,
                                 int dst_size_z) {
    DW_TFLITE(src_data, filters, biases, dst_data, kernel_size_x, kernel_size_y, stride_x, stride_y, padding_x, \
        padding_y, dilation_x, dilation_y, src_size_x, src_size_y, src_size_z, dst_size_x, dst_size_y, dst_size_z)
})

#undef ACT_VEC_F  // RELU6


#undef AS_DATA_T16
#undef CONVERT_TO_DATA_T4
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
#define CONVERT_TO_DATA_T4(x) convert_float4(x)

// kernels without activation func
#define ACT_VEC_F(VEC_T, x) x
ADD_SINGLE_KERNEL(depthwise_conv_unequal_FP32, (__global const DATA_T *bias,
                                              __global DATA_T *output,
                                              unsigned int start) {
    DEPTHWISE_CONV_UNEQUAL(bias, output, start)
})

ADD_SINGLE_KERNEL(depthwise_conv_FP32, (DW_CONV_COMMON_PARAMS, unsigned int kernelHeight, unsigned int kernelWidth, \
                                        unsigned int strideHeight, unsigned int strideWidth, unsigned int topHeight, \
                                        unsigned int topWidth, unsigned int depth_multiplier) {
    DEPTHWISE_CONV(input, weight, bias, output, inputHeight, inputWidth, kernelHeight, kernelWidth, strideHeight, \
                        strideWidth, topHeight, topWidth, depth_multiplier)
})

ADD_SINGLE_KERNEL(depthwise_conv_3x3s1_4P_FP32, (DW_CONV_4P_PARAMS, unsigned int depth_multiplier) {
    DEPTHWISE_CONV_3X3S1_4P(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, topChannel, \
                            depth_multiplier)
})

ADD_SINGLE_KERNEL(depthwise_conv_3x3d2s1_pad_merge_FP32, (DW_CONV_4P_PARAMS, unsigned int padH, unsigned int padW) {
    DEPTHWISE_CONV_3X3D2S1_PAD_MERGE(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                     topChannel, padH, padW)
})

ADD_SINGLE_KERNEL(depthwise_conv_3x3d4s1_pad_merge_FP32, (DW_CONV_4P_PARAMS, unsigned int padH, unsigned int padW) {
    DEPTHWISE_CONV_3X3D4S1_PAD_MERGE(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                     topChannel, padH, padW)
})

ADD_SINGLE_KERNEL(depthwise_conv_3x3s1_pad_merge_FP32, (DW_CONV_4P_PARAMS, unsigned int padH, unsigned int padW) {
    DEPTHWISE_CONV_3X3S1_PAD_MERGE_FP32(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                        topChannel, padH, padW)
})

ADD_SINGLE_KERNEL(depthwise_conv_3x3s2_pad_merge_FP32, (DW_CONV_4P_PARAMS, unsigned int padH, unsigned int padW) {
    DEPTHWISE_CONV_3X3S2_PAD_MERGE(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                   topChannel, padH, padW)
})

ADD_SINGLE_KERNEL(depthwise_conv_3x3s2_FP32, (DW_CONV_4P_PARAMS, unsigned int depth_multiplier) {
    DEPTHWISE_CONV_3X3S2_FP32(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, topChannel, \
                              depth_multiplier)
})

#undef ACT_VEC_F  // NONE

// apply RELU to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELUdepthwise_conv_unequal_FP32, (__global const DATA_T *bias,
                                              __global DATA_T *output,
                                              unsigned int start) {
    DEPTHWISE_CONV_UNEQUAL(bias, output, start)
})

ADD_SINGLE_KERNEL(RELUdepthwise_conv_FP32, (DW_CONV_COMMON_PARAMS, unsigned int kernelHeight, \
                                            unsigned int kernelWidth, unsigned int strideHeight, \
                                            unsigned int strideWidth, unsigned int topHeight, \
                                            unsigned int topWidth, unsigned int depth_multiplier) {
    DEPTHWISE_CONV(input, weight, bias, output, inputHeight, inputWidth, kernelHeight, kernelWidth, strideHeight, \
                   strideWidth, topHeight, topWidth, depth_multiplier)
})

ADD_SINGLE_KERNEL(RELUdepthwise_conv_3x3s1_4P_FP32, (DW_CONV_4P_PARAMS, unsigned int depth_multiplier) {
    DEPTHWISE_CONV_3X3S1_4P(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, topChannel, \
                            depth_multiplier)
})

ADD_SINGLE_KERNEL(RELUdepthwise_conv_3x3d2s1_pad_merge_FP32, (DW_CONV_4P_PARAMS, unsigned int padH, unsigned int padW) {
    DEPTHWISE_CONV_3X3D2S1_PAD_MERGE(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                     topChannel, padH, padW)
})

ADD_SINGLE_KERNEL(RELUdepthwise_conv_3x3d4s1_pad_merge_FP32, (DW_CONV_4P_PARAMS, unsigned int padH, unsigned int padW) {
    DEPTHWISE_CONV_3X3D4S1_PAD_MERGE(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                     topChannel, padH, padW)
})

ADD_SINGLE_KERNEL(RELUdepthwise_conv_3x3s1_pad_merge_FP32, (DW_CONV_4P_PARAMS, unsigned int padH, unsigned int padW) {
    DEPTHWISE_CONV_3X3S1_PAD_MERGE_FP32(input, weight, bias, output, inputHeight, inputWidth, topHeight, \
                                        topWidth, topChannel, padH, padW)
})

ADD_SINGLE_KERNEL(RELUdepthwise_conv_3x3s2_pad_merge_FP32, (DW_CONV_4P_PARAMS, unsigned int padH, unsigned int padW) {
    DEPTHWISE_CONV_3X3S2_PAD_MERGE(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                   topChannel, padH, padW)
})

#undef ACT_VEC_F  // RELU

// apply RELU6 to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU6_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELU6depthwise_conv_unequal_FP32, (__global const DATA_T *bias,
                                              __global DATA_T *output,
                                              unsigned int start) {
    DEPTHWISE_CONV_UNEQUAL(bias, output, start)
})

ADD_SINGLE_KERNEL(RELU6depthwise_conv_FP32, (DW_CONV_COMMON_PARAMS, unsigned int kernelHeight, \
                                             unsigned int kernelWidth, unsigned int strideHeight, \
                                             unsigned int strideWidth, unsigned int topHeight, \
                                             unsigned int topWidth, unsigned int depth_multiplier) {
    DEPTHWISE_CONV(input, weight, bias, output, inputHeight, inputWidth, kernelHeight, kernelWidth, strideHeight, \
                   strideWidth, topHeight, topWidth, depth_multiplier)
})

ADD_SINGLE_KERNEL(RELU6depthwise_conv_3x3s1_4P_FP32, (DW_CONV_4P_PARAMS, unsigned int depth_multiplier) {
    DEPTHWISE_CONV_3X3S1_4P(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, topChannel, \
                            depth_multiplier)
})

ADD_SINGLE_KERNEL(RELU6depthwise_conv_3x3d2s1_pad_merge_FP32, (DW_CONV_4P_PARAMS, \
                                                               unsigned int padH, unsigned int padW) {
    DEPTHWISE_CONV_3X3D2S1_PAD_MERGE(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                     topChannel, padH, padW)
})

ADD_SINGLE_KERNEL(RELU6depthwise_conv_3x3d4s1_pad_merge_FP32, (DW_CONV_4P_PARAMS, \
                                                               unsigned int padH, unsigned int padW) {
    DEPTHWISE_CONV_3X3D4S1_PAD_MERGE(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                     topChannel, padH, padW)
})

ADD_SINGLE_KERNEL(RELU6depthwise_conv_3x3s1_pad_merge_FP32, (DW_CONV_4P_PARAMS, unsigned int padH, unsigned int padW) {
    DEPTHWISE_CONV_3X3S1_PAD_MERGE_FP32(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                        topChannel, padH, padW)
})

ADD_SINGLE_KERNEL(RELU6depthwise_conv_3x3s2_pad_merge_FP32, (DW_CONV_4P_PARAMS, unsigned int padH, unsigned int padW) {
    DEPTHWISE_CONV_3X3S2_PAD_MERGE(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                   topChannel, padH, padW)
})


#undef ACT_VEC_F  // RELU6

#undef AS_DATA_T16
#undef CONVERT_TO_DATA_T4
#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // float

#define Q_DEPTHWISE_CONV_UNEQUAL(bias, output, start, output_offset, output_multiplier, output_shift, \
                                 act_min, act_max) \
        int g0 = get_global_id(0); \
        int g1 = get_global_id(1); \
        int g2 = get_global_id(2); \
        bool overflow; \
        long ab_64; \
        int nudge; \
        int mask; \
        int remainder; \
        int threshold; \
        int shift = -output_shift; \
        int left_shift = shift > 0 ? shift : 0; \
        int right_shift = shift > 0 ? 0 : -shift; \
        int acc = bias[g1 + start]; \
        acc = acc * (1 << left_shift); \
        reQuantized(acc, \
                    output_multiplier, \
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
        int outputIndex = (g0 * get_global_size(1) + g1 + start) * get_global_size(2) + g2; \
        output[outputIndex] = (DATA_T)UNSIGNED_TO_SIGNED(acc);

#define Q_DEPTHWISE_CONV(input, weight, bias, output, inputHeight, inputWidth, kernelHeight, kernelWidth, \
                       strideHeight, strideWidth, topHeight, topWidth, depth_multiplier, input_offset, \
                       weight_offset, output_offset, output_multiplier, output_shift, act_min, act_max) \
        int g0 = get_global_id(0); \
        int g1 = get_global_id(1); \
        int g2 = get_global_id(2); \
        bool overflow; \
        long ab_64; \
        int nudge; \
        int mask; \
        int remainder; \
        int threshold; \
        int shift = -output_shift; \
        int left_shift = shift > 0 ? shift : 0; \
        int right_shift = shift > 0 ? 0 : -shift; \
        if (g2 < topHeight * topWidth) { \
            int inputX = (g2 % topWidth) * strideWidth; \
            int inputY = (g2 / topWidth) * strideHeight; \
            for (int dep = 0; dep < depth_multiplier; dep++) { \
                int out_c = dep + g1 * depth_multiplier; \
                int inputIndex = (g0 * get_global_size(1) + g1) * inputHeight * inputWidth; \
                int outputIndex = \
                    (g0 * get_global_size(1) * depth_multiplier + out_c) * topHeight * topWidth; \
                int acc = bias[out_c]; \
                for (int i = 0; i < kernelHeight; i++) { \
                    for (int j = 0; j < kernelWidth; j++) { \
                        int input_value = \
                            SIGNED_TO_UNSIGNED(input[inputIndex + (inputY + i) * inputWidth + j + inputX]); \
                        int weight_value = \
                            SIGNED_TO_UNSIGNED(weight[out_c * kernelHeight * kernelWidth + i * kernelWidth + j]); \
                        acc += (input_value + input_offset) * (weight_value + weight_offset); \
                    } \
                } \
                acc = acc * (1 << left_shift); \
                reQuantized(acc, \
                            output_multiplier, \
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
                output[outputIndex + g2] = (DATA_T)UNSIGNED_TO_SIGNED(acc); \
            } \
        }

#define Q_DEPTHWISE_CONV_PER_CHANNEL(input, weight, bias, output, inputHeight, inputWidth, kernelHeight, kernelWidth, \
                                   strideHeight, strideWidth, topHeight,  topWidth,  depth_multiplier, input_offset, \
                                   weight_offset, output_offset, output_multiplier, output_shift, act_min, act_max) \
        int g0 = get_global_id(0); \
        int g1 = get_global_id(1); \
        int g2 = get_global_id(2); \
        bool overflow; \
        long ab_64; \
        int nudge; \
        int mask; \
        int remainder; \
        int threshold; \
        if (g2 < topHeight * topWidth) { \
            int inputX = (g2 % topWidth) * strideWidth; \
            int inputY = (g2 / topWidth) * strideHeight; \
            for (int dep = 0; dep < depth_multiplier; dep++) { \
                int shift = output_shift[g1 * depth_multiplier + dep]; \
                int left_shift = shift > 0 ? shift : 0; \
                int right_shift = shift > 0 ? 0 : -shift; \
                int out_c = dep + g1 * depth_multiplier; \
                int inputIndex = (g0 * get_global_size(1) + g1) * inputHeight * inputWidth; \
                int outputIndex = \
                    (g0 * get_global_size(1) * depth_multiplier + out_c) * topHeight * topWidth; \
                int acc = bias[out_c]; \
                for (int i = 0; i < kernelHeight; i++) { \
                    for (int j = 0; j < kernelWidth; j++) { \
                        int input_value = \
                            SIGNED_TO_UNSIGNED(input[inputIndex + (inputY + i) * inputWidth + j + inputX]); \
                        int weight_value = \
                            SIGNED_TO_UNSIGNED(weight[out_c * kernelHeight * kernelWidth + i * kernelWidth + j]); \
                        acc += (input_value + input_offset) * (weight_value + weight_offset); \
                    } \
                } \
                acc = acc * (1 << left_shift); \
                reQuantized(acc, \
                            output_multiplier[g1 * depth_multiplier + dep], \
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
                output[outputIndex + g2] = (DATA_T)UNSIGNED_TO_SIGNED(acc); \
            } \
        }

#define Q_DEPTHWISE_CONV_3X3S1_4P(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                topChannel, depth_multiplier, input_offset, weight_offset, output_offset, \
                                output_multiplier, output_shift, act_min, act_max) \
        int g0 = get_global_id(0); \
        int g1 = get_global_id(1) * 2; \
        int g2 = get_global_id(2) * 2; \
        bool overflow; \
        long ab_64; \
        int nudge; \
        int mask; \
        int remainder; \
        int threshold; \
        int shift = -output_shift; \
        int left_shift = shift > 0 ? shift : 0; \
        int right_shift = shift > 0 ? 0 : -shift; \
        int acc; \
        for (int dep = 0; dep < depth_multiplier; dep++) { \
            int out_g0 = dep + g0 * depth_multiplier; \
            int channel = out_g0 % topChannel; \
            int inputIndex = g0 * inputHeight * inputWidth + g1 * inputWidth + g2; \
            int outputIndex = out_g0 * topHeight * topWidth + g1 * topWidth + g2; \
            int bias_tmp = bias[channel]; \
            if ((g2 + 1) < topWidth && (g1 + 1) < topHeight) {  \
                int4 out00; \
                int4 out01; \
                DATA_T4 k012 = vload4(0, weight + channel * 9); \
                DATA_T4 k345 = vload4(0, weight + channel * 9 + 3); \
                DATA_T4 k678 = vload4(0, weight + channel * 9 + 6); \
                DATA_T4 input00 = vload4(0, input + inputIndex); \
                DATA_T4 input10 = vload4(0, input + inputIndex + inputWidth); \
                DATA_T4 input20 = vload4(0, input + inputIndex + 2 * inputWidth); \
                DATA_T4 input30 = vload4(0, input + inputIndex + 3 * inputWidth); \
                DATA_T4 input01 = (DATA_T4)(input00.s1, input00.s2, input00.s3, 0); \
                DATA_T4 input11 = (DATA_T4)(input10.s1, input10.s2, input10.s3, 0); \
                DATA_T4 input21 = (DATA_T4)(input20.s1, input20.s2, input20.s3, 0); \
                DATA_T4 input31 = (DATA_T4)(input30.s1, input30.s2, input30.s3, 0); \
                out00 = \
                    (convert_int4(k012) + weight_offset) * (convert_int4(input00) + input_offset); \
                out01 = \
                    (convert_int4(k012) + weight_offset) * (convert_int4(input01) + input_offset); \
                out00 += \
                    (convert_int4(k345) + weight_offset) * (convert_int4(input10) + input_offset); \
                out01 += \
                    (convert_int4(k345) + weight_offset) * (convert_int4(input11) + input_offset); \
                out00 += \
                    (convert_int4(k678) + weight_offset) * (convert_int4(input20) + input_offset); \
                out01 += \
                    (convert_int4(k678) + weight_offset) * (convert_int4(input21) + input_offset); \
                acc = out00.s0 + out00.s1 + out00.s2 + bias_tmp; \
                acc = acc * (1 << left_shift); \
                reQuantized(acc, \
                            output_multiplier, \
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
                output[outputIndex] = (DATA_T)acc; \
                acc = out01.s0 + out01.s1 + out01.s2 + bias_tmp; \
                acc = acc * (1 << left_shift); \
                reQuantized(acc, \
                            output_multiplier, \
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
                output[outputIndex + 1] = (DATA_T)acc; \
                out00 = \
                    (convert_int4(k012) + weight_offset) * (convert_int4(input10) + input_offset); \
                out01 = \
                    (convert_int4(k012) + weight_offset) * (convert_int4(input11) + input_offset); \
                out00 += \
                    (convert_int4(k345) + weight_offset) * (convert_int4(input20) + input_offset); \
                out01 += \
                    (convert_int4(k345) + weight_offset) * (convert_int4(input21) + input_offset); \
                out00 += \
                    (convert_int4(k678) + weight_offset) * (convert_int4(input30) + input_offset); \
                out01 += \
                    (convert_int4(k678) + weight_offset) * (convert_int4(input31) + input_offset); \
                acc = out00.s0 + out00.s1 + out00.s2 + bias_tmp; \
                acc = acc * (1 << left_shift); \
                reQuantized(acc, \
                            output_multiplier, \
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
                output[outputIndex + topWidth] = (DATA_T)acc; \
                acc = out01.s0 + out01.s1 + out01.s2 + bias_tmp; \
                acc = acc * (1 << left_shift); \
                reQuantized(acc, \
                            output_multiplier, \
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
                output[outputIndex + topWidth + 1] = (DATA_T)acc; \
            } else if (g2 < topWidth && (g1 + 1) < topHeight) {  \
                int4 out00; \
                DATA_T4 k012 = vload4(0, weight + channel * 9); \
                DATA_T4 k345 = vload4(0, weight + channel * 9 + 3); \
                DATA_T4 k678 = vload4(0, weight + channel * 9 + 6); \
                DATA_T4 input00 = vload4(0, input + inputIndex); \
                DATA_T4 input10 = vload4(0, input + inputIndex + inputWidth); \
                DATA_T4 input20 = vload4(0, input + inputIndex + 2 * inputWidth); \
                DATA_T4 input30 = vload4(0, input + inputIndex + 3 * inputWidth); \
                out00 = \
                    (convert_int4(k012) + weight_offset) * (convert_int4(input00) + input_offset); \
                out00 += \
                    (convert_int4(k345) + weight_offset) * (convert_int4(input10) + input_offset); \
                out00 += \
                    (convert_int4(k678) + weight_offset) * (convert_int4(input20) + input_offset); \
                acc = out00.s0 + out00.s1 + out00.s2 + bias_tmp; \
                acc = acc * (1 << left_shift); \
                reQuantized(acc, \
                            output_multiplier, \
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
                output[outputIndex] = (DATA_T)acc; \
                out00 = \
                    (convert_int4(k012) + weight_offset) * (convert_int4(input10) + input_offset); \
                out00 += \
                    (convert_int4(k345) + weight_offset) * (convert_int4(input20) + input_offset); \
                out00 += \
                    (convert_int4(k678) + weight_offset) * (convert_int4(input30) + input_offset); \
                acc = out00.s0 + out00.s1 + out00.s2 + bias_tmp; \
                acc = acc * (1 << left_shift); \
                reQuantized(acc, \
                            output_multiplier, \
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
                output[outputIndex + topWidth] = (DATA_T)acc; \
            } else if ((g2 + 1) < topWidth && g1 < topHeight) {  \
                int4 out00; \
                int4 out01; \
                DATA_T4 k012 = vload4(0, weight + channel * 9); \
                DATA_T4 k345 = vload4(0, weight + channel * 9 + 3); \
                DATA_T4 k678 = vload4(0, weight + channel * 9 + 6); \
                DATA_T4 input00 = vload4(0, input + inputIndex); \
                DATA_T4 input10 = vload4(0, input + inputIndex + inputWidth); \
                DATA_T4 input20 = vload4(0, input + inputIndex + 2 * inputWidth); \
                DATA_T4 input01 = (DATA_T4)(input00.s1, input00.s2, input00.s3, 0); \
                DATA_T4 input11 = (DATA_T4)(input10.s1, input10.s2, input10.s3, 0); \
                DATA_T4 input21 = (DATA_T4)(input20.s1, input20.s2, input20.s3, 0); \
                out00 = \
                    (convert_int4(k012) + weight_offset) * (convert_int4(input00) + input_offset); \
                out01 = \
                    (convert_int4(k012) + weight_offset) * (convert_int4(input01) + input_offset); \
                out00 += \
                    (convert_int4(k345) + weight_offset) * (convert_int4(input10) + input_offset); \
                out01 += \
                    (convert_int4(k345) + weight_offset) * (convert_int4(input11) + input_offset); \
                out00 += \
                    (convert_int4(k678) + weight_offset) * (convert_int4(input20) + input_offset); \
                out01 += \
                    (convert_int4(k678) + weight_offset) * (convert_int4(input21) + input_offset); \
                acc = out00.s0 + out00.s1 + out00.s2 + bias_tmp; \
                acc = acc * (1 << left_shift); \
                reQuantized(acc, \
                            output_multiplier, \
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
                output[outputIndex] = (DATA_T)acc; \
                acc = out01.s0 + out01.s1 + out01.s2 + bias_tmp; \
                acc = acc * (1 << left_shift); \
                reQuantized(acc, \
                            output_multiplier, \
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
                output[outputIndex + 1] = (DATA_T)acc; \
            } else if (g2 < topWidth && g1 < topHeight) {  \
                int4 out00; \
                DATA_T4 k012 = vload4(0, weight + channel * 9); \
                DATA_T4 k345 = vload4(0, weight + channel * 9 + 3); \
                DATA_T4 k678 = vload4(0, weight + channel * 9 + 6); \
                DATA_T4 input00 = vload4(0, input + inputIndex); \
                DATA_T4 input10 = vload4(0, input + inputIndex + inputWidth); \
                DATA_T4 input20 = vload4(0, input + inputIndex + 2 * inputWidth); \
                out00 = \
                    (convert_int4(k012) + weight_offset) * (convert_int4(input00) + input_offset); \
                out00 += \
                    (convert_int4(k345) + weight_offset) * (convert_int4(input10) + input_offset); \
                out00 += \
                    (convert_int4(k678) + weight_offset) * (convert_int4(input20) + input_offset); \
                acc = out00.s0 + out00.s1 + out00.s2 + bias_tmp; \
                acc = acc * (1 << left_shift); \
                reQuantized(acc, \
                            output_multiplier, \
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
                output[outputIndex] = (DATA_T)acc; \
            } \
        }

#define Q_DEPTHWISE_CONV_3X3S1_PAD_MERGE(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                            topChannel, depth_multiplier, padH, padW, input_offset, weight_offset, \
                                            output_offset, output_multiplier, output_shift, act_min, act_max) \
        int g0 = get_global_id(0); \
        int g1 = get_global_id(1) * 4; \
        int g2 = get_global_id(2) * 4; \
        int shift = -output_shift; \
        int left_shift = shift > 0 ? shift + 2 : 2; \
        int right_shift = shift > 0 ? 0 : -shift; \
        int mask = (((long)1 << right_shift) - 1); \
        int threshold_mask = mask >> 1; \
        int4 threshold4; \
        int inputRowIndex = g1 - padH; \
        int inputColIndex = g2 - padW; \
        int channel = g0 % topChannel; \
        int inputIndex = g0 * inputHeight * inputWidth + inputRowIndex * inputWidth + inputColIndex; \
        int outputIndex = g0 * topHeight * topWidth + g1 * topWidth + g2; \
        if (g2 < topWidth && g1 < topHeight) { \
            DATA_T16 k = vload16(0, weight + channel * 9); \
            DATA_T8 input00 = -input_offset; \
            DATA_T8 input10 = -input_offset; \
            DATA_T8 input20 = -input_offset; \
            DATA_T8 input30 = -input_offset; \
            DATA_T8 input40 = -input_offset; \
            DATA_T8 input50 = -input_offset; \
            if (inputRowIndex >= 0 && inputRowIndex < inputHeight) { \
                input00 = vload8(0, input + inputIndex); \
            } \
            if (inputRowIndex + 1 >= 0 && inputRowIndex + 1 < inputHeight) { \
                input10 = vload8(0, input + inputIndex + inputWidth); \
            } \
            if (inputRowIndex + 2 >= 0 && inputRowIndex + 2 < inputHeight) { \
                input20 = vload8(0, input + inputIndex + 2 * inputWidth); \
            } \
            if (inputRowIndex + 3 >= 0 && inputRowIndex + 3 < inputHeight) { \
                input30 = vload8(0, input + inputIndex + 3 * inputWidth); \
            } \
            if (inputRowIndex + 4 >= 0 && inputRowIndex + 4 < inputHeight) { \
                input40 = vload8(0, input + inputIndex + 4 * inputWidth); \
            } \
            if (inputRowIndex + 5 >= 0 && inputRowIndex + 5 < inputHeight) { \
                input50 = vload8(0, input + inputIndex + 5 * inputWidth); \
            } \
            if (inputColIndex < 0 || inputColIndex >= inputWidth) { \
                input00.s0 = -input_offset; \
                input10.s0 = -input_offset; \
                input20.s0 = -input_offset; \
                input30.s0 = -input_offset; \
                input40.s0 = -input_offset; \
                input50.s0 = -input_offset; \
            } \
            if (inputColIndex + 1 < 0 || inputColIndex + 1 >= inputWidth) { \
                input00.s1 = -input_offset; \
                input10.s1 = -input_offset; \
                input20.s1 = -input_offset; \
                input30.s1 = -input_offset; \
                input40.s1 = -input_offset; \
                input50.s1 = -input_offset; \
            } \
            if (inputColIndex + 2 < 0 || inputColIndex + 2 >= inputWidth) { \
                input00.s2 = -input_offset; \
                input10.s2 = -input_offset; \
                input20.s2 = -input_offset; \
                input30.s2 = -input_offset; \
                input40.s2 = -input_offset; \
                input50.s2 = -input_offset; \
            } \
            if (inputColIndex + 3 < 0 || inputColIndex + 3 >= inputWidth) { \
                input00.s3 = -input_offset; \
                input10.s3 = -input_offset; \
                input20.s3 = -input_offset; \
                input30.s3 = -input_offset; \
                input40.s3 = -input_offset; \
                input50.s3 = -input_offset; \
            } \
            if (inputColIndex + 4 < 0 || inputColIndex + 4 >= inputWidth) { \
                input00.s4 = -input_offset; \
                input10.s4 = -input_offset; \
                input20.s4 = -input_offset; \
                input30.s4 = -input_offset; \
                input40.s4 = -input_offset; \
                input50.s4 = -input_offset; \
            } \
            if (inputColIndex + 5 < 0 || inputColIndex + 5 >= inputWidth) { \
                input00.s5 = -input_offset; \
                input10.s5 = -input_offset; \
                input20.s5 = -input_offset; \
                input30.s5 = -input_offset; \
                input40.s5 = -input_offset; \
                input50.s5 = -input_offset; \
            } \
            int4 out0 = (int4)bias[channel]; \
            int4 out1 = (int4)bias[channel]; \
            int4 out2 = (int4)bias[channel]; \
            int4 out3 = (int4)bias[channel]; \
            ARM_DOT((DATA_T4)(k.s012, 0), input00.s0123, out0.s0); \
            ARM_DOT((DATA_T4)(k.s345, 0), input10.s0123, out0.s0); \
            ARM_DOT((DATA_T4)(k.s678, 0), input20.s0123, out0.s0); \
            ARM_DOT((DATA_T4)(k.s012, 0), (DATA_T4)(input00.s123, 0), out0.s1); \
            ARM_DOT((DATA_T4)(k.s345, 0), (DATA_T4)(input10.s123, 0), out0.s1); \
            ARM_DOT((DATA_T4)(k.s678, 0), (DATA_T4)(input20.s123, 0), out0.s1); \
            ARM_DOT((DATA_T4)(k.s012, 0), (DATA_T4)(input00.s234, 0), out0.s2); \
            ARM_DOT((DATA_T4)(k.s345, 0), (DATA_T4)(input10.s234, 0), out0.s2); \
            ARM_DOT((DATA_T4)(k.s678, 0), (DATA_T4)(input20.s234, 0), out0.s2); \
            ARM_DOT((DATA_T4)(k.s012, 0), (DATA_T4)(input00.s345, 0), out0.s3); \
            ARM_DOT((DATA_T4)(k.s345, 0), (DATA_T4)(input10.s345, 0), out0.s3); \
            ARM_DOT((DATA_T4)(k.s678, 0), (DATA_T4)(input20.s345, 0), out0.s3); \
            ARM_DOT((DATA_T4)(k.s012, 0), input10.s0123, out1.s0); \
            ARM_DOT((DATA_T4)(k.s345, 0), input20.s0123, out1.s0); \
            ARM_DOT((DATA_T4)(k.s678, 0), input30.s0123, out1.s0); \
            ARM_DOT((DATA_T4)(k.s012, 0), (DATA_T4)(input10.s123, 0), out1.s1); \
            ARM_DOT((DATA_T4)(k.s345, 0), (DATA_T4)(input20.s123, 0), out1.s1); \
            ARM_DOT((DATA_T4)(k.s678, 0), (DATA_T4)(input30.s123, 0), out1.s1); \
            ARM_DOT((DATA_T4)(k.s012, 0), (DATA_T4)(input10.s234, 0), out1.s2); \
            ARM_DOT((DATA_T4)(k.s345, 0), (DATA_T4)(input20.s234, 0), out1.s2); \
            ARM_DOT((DATA_T4)(k.s678, 0), (DATA_T4)(input30.s234, 0), out1.s2); \
            ARM_DOT((DATA_T4)(k.s012, 0), (DATA_T4)(input10.s345, 0), out1.s3); \
            ARM_DOT((DATA_T4)(k.s345, 0), (DATA_T4)(input20.s345, 0), out1.s3); \
            ARM_DOT((DATA_T4)(k.s678, 0), (DATA_T4)(input30.s345, 0), out1.s3); \
            ARM_DOT((DATA_T4)(k.s012, 0), input20.s0123, out2.s0); \
            ARM_DOT((DATA_T4)(k.s345, 0), input30.s0123, out2.s0); \
            ARM_DOT((DATA_T4)(k.s678, 0), input40.s0123, out2.s0); \
            ARM_DOT((DATA_T4)(k.s012, 0), (DATA_T4)(input20.s123, 0), out2.s1); \
            ARM_DOT((DATA_T4)(k.s345, 0), (DATA_T4)(input30.s123, 0), out2.s1); \
            ARM_DOT((DATA_T4)(k.s678, 0), (DATA_T4)(input40.s123, 0), out2.s1); \
            ARM_DOT((DATA_T4)(k.s012, 0), (DATA_T4)(input20.s234, 0), out2.s2); \
            ARM_DOT((DATA_T4)(k.s345, 0), (DATA_T4)(input30.s234, 0), out2.s2); \
            ARM_DOT((DATA_T4)(k.s678, 0), (DATA_T4)(input40.s234, 0), out2.s2); \
            ARM_DOT((DATA_T4)(k.s012, 0), (DATA_T4)(input20.s345, 0), out2.s3); \
            ARM_DOT((DATA_T4)(k.s345, 0), (DATA_T4)(input30.s345, 0), out2.s3); \
            ARM_DOT((DATA_T4)(k.s678, 0), (DATA_T4)(input40.s345, 0), out2.s3); \
            ARM_DOT((DATA_T4)(k.s012, 0), input30.s0123, out3.s0); \
            ARM_DOT((DATA_T4)(k.s345, 0), input40.s0123, out3.s0); \
            ARM_DOT((DATA_T4)(k.s678, 0), input50.s0123, out3.s0); \
            ARM_DOT((DATA_T4)(k.s012, 0), (DATA_T4)(input30.s123, 0), out3.s1); \
            ARM_DOT((DATA_T4)(k.s345, 0), (DATA_T4)(input40.s123, 0), out3.s1); \
            ARM_DOT((DATA_T4)(k.s678, 0), (DATA_T4)(input50.s123, 0), out3.s1); \
            ARM_DOT((DATA_T4)(k.s012, 0), (DATA_T4)(input30.s234, 0), out3.s2); \
            ARM_DOT((DATA_T4)(k.s345, 0), (DATA_T4)(input40.s234, 0), out3.s2); \
            ARM_DOT((DATA_T4)(k.s678, 0), (DATA_T4)(input50.s234, 0), out3.s2); \
            ARM_DOT((DATA_T4)(k.s012, 0), (DATA_T4)(input30.s345, 0), out3.s3); \
            ARM_DOT((DATA_T4)(k.s345, 0), (DATA_T4)(input40.s345, 0), out3.s3); \
            ARM_DOT((DATA_T4)(k.s678, 0), (DATA_T4)(input50.s345, 0), out3.s3); \
            if (weight_offset != 0) { \
                int4 inoff = (int4)0; \
                ARM_DOT((DATA_T4)(input00.s012, 0), (DATA_T4)(-weight_offset), inoff.s0); \
                ARM_DOT((DATA_T4)(input10.s012, 0), (DATA_T4)(-weight_offset), inoff.s0); \
                ARM_DOT((DATA_T4)(input20.s012, 0), (DATA_T4)(-weight_offset), inoff.s0); \
                ARM_DOT((DATA_T4)(input00.s123, 0), (DATA_T4)(-weight_offset), inoff.s1); \
                ARM_DOT((DATA_T4)(input10.s123, 0), (DATA_T4)(-weight_offset), inoff.s1); \
                ARM_DOT((DATA_T4)(input20.s123, 0), (DATA_T4)(-weight_offset), inoff.s1); \
                ARM_DOT((DATA_T4)(input00.s234, 0), (DATA_T4)(-weight_offset), inoff.s2); \
                ARM_DOT((DATA_T4)(input10.s234, 0), (DATA_T4)(-weight_offset), inoff.s2); \
                ARM_DOT((DATA_T4)(input20.s234, 0), (DATA_T4)(-weight_offset), inoff.s2); \
                ARM_DOT((DATA_T4)(input00.s345, 0), (DATA_T4)(-weight_offset), inoff.s3); \
                ARM_DOT((DATA_T4)(input10.s345, 0), (DATA_T4)(-weight_offset), inoff.s3); \
                ARM_DOT((DATA_T4)(input20.s345, 0), (DATA_T4)(-weight_offset), inoff.s3); \
                out0 -= inoff; \
                inoff = (int4)0; \
                ARM_DOT((DATA_T4)(input10.s012, 0), (DATA_T4)(-weight_offset), inoff.s0); \
                ARM_DOT((DATA_T4)(input20.s012, 0), (DATA_T4)(-weight_offset), inoff.s0); \
                ARM_DOT((DATA_T4)(input30.s012, 0), (DATA_T4)(-weight_offset), inoff.s0); \
                ARM_DOT((DATA_T4)(input10.s123, 0), (DATA_T4)(-weight_offset), inoff.s1); \
                ARM_DOT((DATA_T4)(input20.s123, 0), (DATA_T4)(-weight_offset), inoff.s1); \
                ARM_DOT((DATA_T4)(input30.s123, 0), (DATA_T4)(-weight_offset), inoff.s1); \
                ARM_DOT((DATA_T4)(input10.s234, 0), (DATA_T4)(-weight_offset), inoff.s2); \
                ARM_DOT((DATA_T4)(input20.s234, 0), (DATA_T4)(-weight_offset), inoff.s2); \
                ARM_DOT((DATA_T4)(input30.s234, 0), (DATA_T4)(-weight_offset), inoff.s2); \
                ARM_DOT((DATA_T4)(input10.s345, 0), (DATA_T4)(-weight_offset), inoff.s3); \
                ARM_DOT((DATA_T4)(input20.s345, 0), (DATA_T4)(-weight_offset), inoff.s3); \
                ARM_DOT((DATA_T4)(input30.s345, 0), (DATA_T4)(-weight_offset), inoff.s3); \
                out1 -= inoff; \
                inoff = (int4)0; \
                ARM_DOT((DATA_T4)(input20.s012, 0), (DATA_T4)(-weight_offset), inoff.s0); \
                ARM_DOT((DATA_T4)(input30.s012, 0), (DATA_T4)(-weight_offset), inoff.s0); \
                ARM_DOT((DATA_T4)(input40.s012, 0), (DATA_T4)(-weight_offset), inoff.s0); \
                ARM_DOT((DATA_T4)(input20.s123, 0), (DATA_T4)(-weight_offset), inoff.s1); \
                ARM_DOT((DATA_T4)(input30.s123, 0), (DATA_T4)(-weight_offset), inoff.s1); \
                ARM_DOT((DATA_T4)(input40.s123, 0), (DATA_T4)(-weight_offset), inoff.s1); \
                ARM_DOT((DATA_T4)(input20.s234, 0), (DATA_T4)(-weight_offset), inoff.s2); \
                ARM_DOT((DATA_T4)(input30.s234, 0), (DATA_T4)(-weight_offset), inoff.s2); \
                ARM_DOT((DATA_T4)(input40.s234, 0), (DATA_T4)(-weight_offset), inoff.s2); \
                ARM_DOT((DATA_T4)(input20.s345, 0), (DATA_T4)(-weight_offset), inoff.s3); \
                ARM_DOT((DATA_T4)(input30.s345, 0), (DATA_T4)(-weight_offset), inoff.s3); \
                ARM_DOT((DATA_T4)(input40.s345, 0), (DATA_T4)(-weight_offset), inoff.s3); \
                out2 -= inoff; \
                inoff = (int4)0; \
                ARM_DOT((DATA_T4)(input30.s012, 0), (DATA_T4)(-weight_offset), inoff.s0); \
                ARM_DOT((DATA_T4)(input40.s012, 0), (DATA_T4)(-weight_offset), inoff.s0); \
                ARM_DOT((DATA_T4)(input50.s012, 0), (DATA_T4)(-weight_offset), inoff.s0); \
                ARM_DOT((DATA_T4)(input30.s123, 0), (DATA_T4)(-weight_offset), inoff.s1); \
                ARM_DOT((DATA_T4)(input40.s123, 0), (DATA_T4)(-weight_offset), inoff.s1); \
                ARM_DOT((DATA_T4)(input50.s123, 0), (DATA_T4)(-weight_offset), inoff.s1); \
                ARM_DOT((DATA_T4)(input30.s234, 0), (DATA_T4)(-weight_offset), inoff.s2); \
                ARM_DOT((DATA_T4)(input40.s234, 0), (DATA_T4)(-weight_offset), inoff.s2); \
                ARM_DOT((DATA_T4)(input50.s234, 0), (DATA_T4)(-weight_offset), inoff.s2); \
                ARM_DOT((DATA_T4)(input30.s345, 0), (DATA_T4)(-weight_offset), inoff.s3); \
                ARM_DOT((DATA_T4)(input40.s345, 0), (DATA_T4)(-weight_offset), inoff.s3); \
                ARM_DOT((DATA_T4)(input50.s345, 0), (DATA_T4)(-weight_offset), inoff.s3); \
                out3 -= inoff; \
            } \
            if (input_offset != 0) { \
                int koff = 0; \
                ARM_DOT((DATA_T4)(k.s012, 0), (DATA_T4)(-input_offset), koff); \
                ARM_DOT((DATA_T4)(k.s345, 0), (DATA_T4)(-input_offset), koff); \
                ARM_DOT((DATA_T4)(k.s678, 0), (DATA_T4)(-input_offset), koff); \
                out0 -= (int4)koff; \
                out1 -= (int4)koff; \
                out2 -= (int4)koff; \
                out3 -= (int4)koff; \
            } \
            if (input_offset != 0 && weight_offset != 0) { \
                int4 off = (int4)(9 * input_offset * weight_offset); \
                out0 += off; \
                out1 += off; \
                out2 += off; \
                out3 += off; \
            } \
            out0 = mul_hi(out0 << left_shift, output_multiplier); \
            out0 = rhadd(out0, 0); \
            threshold4 = threshold_mask + select((int4)(0), (int4)(1), out0 < 0); \
            out0 = (out0 >> right_shift) + select((int4)(0), (int4)(1), (out0 & mask) > threshold4); \
            out0 += output_offset; \
            out0 = max(out0, act_min); \
            out0 = min(out0, act_max); \
            out1 = mul_hi(out1 << left_shift, output_multiplier); \
            out1 = rhadd(out1, 0); \
            threshold4 = threshold_mask + select((int4)(0), (int4)(1), out1 < 0); \
            out1 = (out1 >> right_shift) + select((int4)(0), (int4)(1), (out1 & mask) > threshold4); \
            out1 += output_offset; \
            out1 = max(out1, act_min); \
            out1 = min(out1, act_max); \
            out2 = mul_hi(out2 << left_shift, output_multiplier); \
            out2 = rhadd(out2, 0); \
            threshold4 = threshold_mask + select((int4)(0), (int4)(1), out2 < 0); \
            out2 = (out2 >> right_shift) + select((int4)(0), (int4)(1), (out2 & mask) > threshold4); \
            out2 += output_offset; \
            out2 = max(out2, act_min); \
            out2 = min(out2, act_max); \
            out3 = mul_hi(out3 << left_shift, output_multiplier); \
            out3 = rhadd(out3, 0); \
            threshold4 = threshold_mask + select((int4)(0), (int4)(1), out3 < 0); \
            out3 = (out3 >> right_shift) + select((int4)(0), (int4)(1), (out3 & mask) > threshold4); \
            out3 += output_offset; \
            out3 = max(out3, act_min); \
            out3 = min(out3, act_max); \
            if (g1 + 3 < topHeight) { \
                if (g2 + 3 < topWidth) { \
                    vstore4((DATA_T4)(out0.s0, out0.s1, out0.s2, out0.s3), 0, output + outputIndex); \
                    vstore4((DATA_T4)(out1.s0, out1.s1, out1.s2, out1.s3), \
                            0, \
                            output + outputIndex + topWidth); \
                    vstore4((DATA_T4)(out2.s0, out2.s1, out2.s2, out2.s3), \
                            0, \
                            output + outputIndex + 2 * topWidth); \
                    vstore4((DATA_T4)(out3.s0, out3.s1, out3.s2, out3.s3), \
                            0, \
                            output + outputIndex + 3 * topWidth); \
                } else if (g2 + 2 < topWidth) { \
                    vstore3((DATA_T3)(out0.s0, out0.s1, out0.s2), 0, output + outputIndex); \
                    vstore3( \
                        (DATA_T3)(out1.s0, out1.s1, out1.s2), 0, output + outputIndex + topWidth); \
                    vstore3((DATA_T3)(out2.s0, out2.s1, out2.s2), \
                            0, \
                            output + outputIndex + 2 * topWidth); \
                    vstore3((DATA_T3)(out3.s0, out3.s1, out3.s2), \
                            0, \
                            output + outputIndex + 3 * topWidth); \
                } else if (g2 + 1 < topWidth) { \
                    vstore2((DATA_T2)(out0.s0, out0.s1), 0, output + outputIndex); \
                    vstore2((DATA_T2)(out1.s0, out1.s1), 0, output + outputIndex + topWidth); \
                    vstore2((DATA_T2)(out2.s0, out2.s1), 0, output + outputIndex + 2 * topWidth); \
                    vstore2((DATA_T2)(out3.s0, out3.s1), 0, output + outputIndex + 3 * topWidth); \
                } else { \
                    output[outputIndex] = (DATA_T)(out0.s0); \
                    output[outputIndex + topWidth] = (DATA_T)(out1.s0); \
                    output[outputIndex + 2 * topWidth] = (DATA_T)(out2.s0); \
                    output[outputIndex + 3 * topWidth] = (DATA_T)(out3.s0); \
                } \
            } else if (g1 + 2 < topHeight) { \
                if (g2 + 3 < topWidth) { \
                    vstore4((DATA_T4)(out0.s0, out0.s1, out0.s2, out0.s3), 0, output + outputIndex); \
                    vstore4((DATA_T4)(out1.s0, out1.s1, out1.s2, out1.s3), \
                            0, \
                            output + outputIndex + topWidth); \
                    vstore4((DATA_T4)(out2.s0, out2.s1, out2.s2, out2.s3), \
                            0, \
                            output + outputIndex + 2 * topWidth); \
                } else if (g2 + 2 < topWidth) { \
                    vstore3((DATA_T3)(out0.s0, out0.s1, out0.s2), 0, output + outputIndex); \
                    vstore3( \
                        (DATA_T3)(out1.s0, out1.s1, out1.s2), 0, output + outputIndex + topWidth); \
                    vstore3((DATA_T3)(out2.s0, out2.s1, out2.s2), \
                            0, \
                            output + outputIndex + 2 * topWidth); \
                } else if (g2 + 1 < topWidth) { \
                    vstore2((DATA_T2)(out0.s0, out0.s1), 0, output + outputIndex); \
                    vstore2((DATA_T2)(out1.s0, out1.s1), 0, output + outputIndex + topWidth); \
                    vstore2((DATA_T2)(out2.s0, out2.s1), 0, output + outputIndex + 2 * topWidth); \
                } else { \
                    output[outputIndex] = (DATA_T)(out0.s0); \
                    output[outputIndex + topWidth] = (DATA_T)(out1.s0); \
                    output[outputIndex + 2 * topWidth] = (DATA_T)(out2.s0); \
                } \
            } else if (g1 + 1 < topHeight) { \
                if (g2 + 3 < topWidth) { \
                    vstore4((DATA_T4)(out0.s0, out0.s1, out0.s2, out0.s3), 0, output + outputIndex); \
                    vstore4((DATA_T4)(out1.s0, out1.s1, out1.s2, out1.s3), \
                            0, \
                            output + outputIndex + topWidth); \
                } else if (g2 + 2 < topWidth) { \
                    vstore3((DATA_T3)(out0.s0, out0.s1, out0.s2), 0, output + outputIndex); \
                    vstore3( \
                        (DATA_T3)(out1.s0, out1.s1, out1.s2), 0, output + outputIndex + topWidth); \
                } else if (g2 + 1 < topWidth) { \
                    vstore2((DATA_T2)(out0.s0, out0.s1), 0, output + outputIndex); \
                    vstore2((DATA_T2)(out1.s0, out1.s1), 0, output + outputIndex + topWidth); \
                } else { \
                    output[outputIndex] = (DATA_T)(out0.s0); \
                    output[outputIndex + topWidth] = (DATA_T)(out1.s0); \
                } \
            } else { \
                if (g2 + 3 < topWidth) { \
                    vstore4((DATA_T4)(out0.s0, out0.s1, out0.s2, out0.s3), 0, output + outputIndex); \
                } else if (g2 + 2 < topWidth) { \
                    vstore3((DATA_T3)(out0.s0, out0.s1, out0.s2), 0, output + outputIndex); \
                } else if (g2 + 1 < topWidth) { \
                    vstore2((DATA_T2)(out0.s0, out0.s1), 0, output + outputIndex); \
                } else { \
                    output[outputIndex] = (DATA_T)(out0.s0); \
                } \
            } \
        }

#define Q_DEPTHWISE_CONV_3X3S2_PAD_MERGE(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                       topChannel, depth_multiplier, padH, padW, input_offset, weight_offset, \
                                       output_offset, output_multiplier, output_shift, act_min, act_max) \
        int g0 = get_global_id(0); \
        int g1 = get_global_id(1); \
        int g2 = get_global_id(2) * 3; \
        int shift = -output_shift; \
        int left_shift = shift > 0 ? shift + 2 : 2; \
        int right_shift = shift > 0 ? 0 : -shift; \
        int mask = (((long)1 << right_shift) - 1); \
        int threshold_mask = mask >> 1; \
        int inputRowIndex = g1 * 2 - padH; \
        int inputColIndex = g2 * 2 - padW; \
        int channel = g0 % topChannel; \
        int inputIndex = g0 * inputHeight * inputWidth + inputRowIndex * inputWidth + inputColIndex; \
        int outputIndex = g0 * topHeight * topWidth + g1 * topWidth + g2; \
        if (g2 < topWidth) { \
            DATA_T16 k = vload16(0, weight + channel * 9); \
            DATA_T8 input00 = -input_offset; \
            DATA_T8 input10 = -input_offset; \
            DATA_T8 input20 = -input_offset; \
            if (inputRowIndex >= 0 && inputRowIndex < inputHeight) { \
                input00 = vload8(0, input + inputIndex); \
            } \
            if (inputRowIndex + 1 >= 0 && inputRowIndex + 1 < inputHeight) { \
                input10 = vload8(0, input + inputIndex + inputWidth); \
            } \
            if (inputRowIndex + 2 >= 0 && inputRowIndex + 2 < inputHeight) { \
                input20 = vload8(0, input + inputIndex + 2 * inputWidth); \
            } \
            if (inputColIndex < 0 || inputColIndex >= inputWidth) { \
                input00.s0 = -input_offset; \
                input10.s0 = -input_offset; \
                input20.s0 = -input_offset; \
            } \
            if (inputColIndex + 1 < 0 || inputColIndex + 1 >= inputWidth) { \
                input00.s1 = -input_offset; \
                input10.s1 = -input_offset; \
                input20.s1 = -input_offset; \
            } \
            if (inputColIndex + 2 < 0 || inputColIndex + 2 >= inputWidth) { \
                input00.s2 = -input_offset; \
                input10.s2 = -input_offset; \
                input20.s2 = -input_offset; \
            } \
            if (inputColIndex + 3 < 0 || inputColIndex + 3 >= inputWidth) { \
                input00.s3 = -input_offset; \
                input10.s3 = -input_offset; \
                input20.s3 = -input_offset; \
            } \
            if (inputColIndex + 4 < 0 || inputColIndex + 4 >= inputWidth) { \
                input00.s4 = -input_offset; \
                input10.s4 = -input_offset; \
                input20.s4 = -input_offset; \
            } \
            if (inputColIndex + 5 < 0 || inputColIndex + 5 >= inputWidth) { \
                input00.s5 = -input_offset; \
                input10.s5 = -input_offset; \
                input20.s5 = -input_offset; \
            } \
            if (inputColIndex + 6 < 0 || inputColIndex + 6 >= inputWidth) { \
                input00.s6 = -input_offset; \
                input10.s6 = -input_offset; \
                input20.s6 = -input_offset; \
            } \
            if (inputColIndex + 7 < 0 || inputColIndex + 7 >= inputWidth) { \
                input00.s7 = -input_offset; \
                input10.s7 = -input_offset; \
                input20.s7 = -input_offset; \
            } \
            int3 out3 = (int3)bias[channel]; \
            ARM_DOT((DATA_T4)(k.s012, 0), input00.s0123, out3.s0); \
            ARM_DOT((DATA_T4)(k.s345, 0), input10.s0123, out3.s0); \
            ARM_DOT((DATA_T4)(k.s678, 0), input20.s0123, out3.s0); \
            ARM_DOT((DATA_T4)(k.s012, 0), input00.s2345, out3.s1); \
            ARM_DOT((DATA_T4)(k.s345, 0), input10.s2345, out3.s1); \
            ARM_DOT((DATA_T4)(k.s678, 0), input20.s2345, out3.s1); \
            ARM_DOT((DATA_T4)(k.s012, 0), input00.s4567, out3.s2); \
            ARM_DOT((DATA_T4)(k.s345, 0), input10.s4567, out3.s2); \
            ARM_DOT((DATA_T4)(k.s678, 0), input20.s4567, out3.s2); \
            if (weight_offset != 0) { \
                int3 inoff = 0; \
                ARM_DOT((DATA_T4)(input00.s012, 0), (DATA_T4)(-weight_offset), inoff.s0); \
                ARM_DOT((DATA_T4)(input10.s012, 0), (DATA_T4)(-weight_offset), inoff.s0); \
                ARM_DOT((DATA_T4)(input20.s012, 0), (DATA_T4)(-weight_offset), inoff.s0); \
                ARM_DOT((DATA_T4)(input00.s234, 0), (DATA_T4)(-weight_offset), inoff.s1); \
                ARM_DOT((DATA_T4)(input10.s234, 0), (DATA_T4)(-weight_offset), inoff.s1); \
                ARM_DOT((DATA_T4)(input20.s234, 0), (DATA_T4)(-weight_offset), inoff.s1); \
                ARM_DOT((DATA_T4)(input00.s456, 0), (DATA_T4)(-weight_offset), inoff.s2); \
                ARM_DOT((DATA_T4)(input10.s456, 0), (DATA_T4)(-weight_offset), inoff.s2); \
                ARM_DOT((DATA_T4)(input20.s456, 0), (DATA_T4)(-weight_offset), inoff.s2); \
                out3 -= inoff; \
            } \
            if (input_offset != 0) { \
                int koff = 0; \
                ARM_DOT((DATA_T4)(k.s012, 0), (DATA_T4)(-input_offset), koff); \
                ARM_DOT((DATA_T4)(k.s345, 0), (DATA_T4)(-input_offset), koff); \
                ARM_DOT((DATA_T4)(k.s678, 0), (DATA_T4)(-input_offset), koff); \
                out3 -= (int3)koff; \
            } \
            if (input_offset != 0 && weight_offset != 0) { \
                out3 += (9 * input_offset * weight_offset); \
            } \
            int3 threshold3; \
            out3 = mul_hi(out3 << left_shift, output_multiplier); \
            out3 = rhadd(out3, 0); \
            threshold3 = threshold_mask + select((int3)(0), (int3)(1), out3 < 0); \
            out3 = (out3 >> right_shift) + select((int3)(0), (int3)(1), (out3 & mask) > threshold3); \
            out3 += output_offset; \
            out3 = max(out3, act_min); \
            out3 = min(out3, act_max); \
            if (g2 + 2 < topWidth) { \
                vstore2((DATA_T2)(out3.s0, out3.s1), 0, output + outputIndex); \
                output[outputIndex + 2] = (DATA_T)out3.s2; \
            } else if (g2 + 1 < topWidth) { \
                vstore2((DATA_T2)(out3.s0, out3.s1), 0, output + outputIndex); \
            } else { \
                output[outputIndex] = (DATA_T)out3.s0; \
            } \
        }

#define Q_DEPTHWISE_CONV_3X3S1D2_PAD_MERGE(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                         topChannel, depth_multiplier, padH, padW, input_offset, weight_offset, \
                                         output_offset, output_multiplier, output_shift, act_min, act_max) \
        int g0 = get_global_id(0); \
        int g1 = get_global_id(1); \
        int g2 = get_global_id(2) * 4; \
        int inputRowIndex = g1 - padH; \
        int inputColIndex = g2 - padW; \
        int shift = -output_shift; \
        int left_shift = shift > 0 ? shift + 2 : 2; \
        int right_shift = shift > 0 ? 0 : -shift; \
        int mask = (((long)1 << right_shift) - 1); \
        int threshold_mask = mask >> 1; \
        int channel = g0 % topChannel; \
        int inputIndex = g0 * inputHeight * inputWidth + inputRowIndex * inputWidth + inputColIndex; \
        int outputIndex = g0 * topHeight * topWidth + g1 * topWidth + g2; \
        if (g2 < topWidth && g1 < topHeight) { \
            DATA_T16 k = vload16(0, weight + channel * 9); \
            DATA_T8 input00 = -input_offset; \
            DATA_T8 input20 = -input_offset; \
            DATA_T8 input40 = -input_offset; \
            if (inputRowIndex >= 0 && inputRowIndex < inputHeight) { \
                input00 = vload8(0, input + inputIndex); \
            } \
            if (inputRowIndex + 2 >= 0 && inputRowIndex + 2 < inputHeight) { \
                input20 = vload8(0, input + inputIndex + 2 * inputWidth); \
            } \
            if (inputRowIndex + 4 >= 0 && inputRowIndex + 4 < inputHeight) { \
                input40 = vload8(0, input + inputIndex + 4 * inputWidth); \
            } \
            if (inputColIndex < 0 || inputColIndex >= inputWidth) { \
                input00.s0 = -input_offset; \
                input20.s0 = -input_offset; \
                input40.s0 = -input_offset; \
            } \
            if (inputColIndex + 1 < 0 || inputColIndex + 1 >= inputWidth) { \
                input00.s1 = -input_offset; \
                input20.s1 = -input_offset; \
                input40.s1 = -input_offset; \
            } \
            if (inputColIndex + 2 < 0 || inputColIndex + 2 >= inputWidth) { \
                input00.s2 = -input_offset; \
                input20.s2 = -input_offset; \
                input40.s2 = -input_offset; \
            } \
            if (inputColIndex + 3 < 0 || inputColIndex + 3 >= inputWidth) { \
                input00.s3 = -input_offset; \
                input20.s3 = -input_offset; \
                input40.s3 = -input_offset; \
            } \
            if (inputColIndex + 4 < 0 || inputColIndex + 4 >= inputWidth) { \
                input00.s4 = -input_offset; \
                input20.s4 = -input_offset; \
                input40.s4 = -input_offset; \
            } \
            if (inputColIndex + 5 < 0 || inputColIndex + 5 >= inputWidth) { \
                input00.s5 = -input_offset; \
                input20.s5 = -input_offset; \
                input40.s5 = -input_offset; \
            } \
            if (inputColIndex + 6 < 0 || inputColIndex + 6 >= inputWidth) { \
                input00.s6 = -input_offset; \
                input20.s6 = -input_offset; \
                input40.s6 = -input_offset; \
            } \
            if (inputColIndex + 7 < 0 || inputColIndex + 7 >= inputWidth) { \
                input00.s7 = -input_offset; \
                input20.s7 = -input_offset; \
                input40.s7 = -input_offset; \
            } \
            int4 out4 = (int4)bias[channel]; \
            ARM_DOT((DATA_T4)(k.s012, 0), (DATA_T4)(input00.s024, 0), out4.s0); \
            ARM_DOT((DATA_T4)(k.s345, 0), (DATA_T4)(input20.s024, 0), out4.s0); \
            ARM_DOT((DATA_T4)(k.s678, 0), (DATA_T4)(input40.s024, 0), out4.s0); \
            ARM_DOT((DATA_T4)(k.s012, 0), (DATA_T4)(input00.s135, 0), out4.s1); \
            ARM_DOT((DATA_T4)(k.s345, 0), (DATA_T4)(input20.s135, 0), out4.s1); \
            ARM_DOT((DATA_T4)(k.s678, 0), (DATA_T4)(input40.s135, 0), out4.s1); \
            ARM_DOT((DATA_T4)(k.s012, 0), (DATA_T4)(input00.s246, 0), out4.s2); \
            ARM_DOT((DATA_T4)(k.s345, 0), (DATA_T4)(input20.s246, 0), out4.s2); \
            ARM_DOT((DATA_T4)(k.s678, 0), (DATA_T4)(input40.s246, 0), out4.s2); \
            ARM_DOT((DATA_T4)(k.s012, 0), (DATA_T4)(input00.s357, 0), out4.s3); \
            ARM_DOT((DATA_T4)(k.s345, 0), (DATA_T4)(input20.s357, 0), out4.s3); \
            ARM_DOT((DATA_T4)(k.s678, 0), (DATA_T4)(input40.s357, 0), out4.s3); \
            if (weight_offset != 0) { \
                int4 inoff = 0; \
                ARM_DOT((DATA_T4)(input00.s024, 0), (DATA_T4)(-weight_offset), inoff.s0); \
                ARM_DOT((DATA_T4)(input20.s024, 0), (DATA_T4)(-weight_offset), inoff.s0); \
                ARM_DOT((DATA_T4)(input40.s024, 0), (DATA_T4)(-weight_offset), inoff.s0); \
                ARM_DOT((DATA_T4)(input00.s135, 0), (DATA_T4)(-weight_offset), inoff.s1); \
                ARM_DOT((DATA_T4)(input20.s135, 0), (DATA_T4)(-weight_offset), inoff.s1); \
                ARM_DOT((DATA_T4)(input40.s135, 0), (DATA_T4)(-weight_offset), inoff.s1); \
                ARM_DOT((DATA_T4)(input00.s246, 0), (DATA_T4)(-weight_offset), inoff.s2); \
                ARM_DOT((DATA_T4)(input20.s246, 0), (DATA_T4)(-weight_offset), inoff.s2); \
                ARM_DOT((DATA_T4)(input40.s246, 0), (DATA_T4)(-weight_offset), inoff.s2); \
                ARM_DOT((DATA_T4)(input00.s357, 0), (DATA_T4)(-weight_offset), inoff.s3); \
                ARM_DOT((DATA_T4)(input20.s357, 0), (DATA_T4)(-weight_offset), inoff.s3); \
                ARM_DOT((DATA_T4)(input40.s357, 0), (DATA_T4)(-weight_offset), inoff.s3); \
                out4 -= inoff; \
            } \
            if (input_offset != 0) { \
                int koff = 0; \
                ARM_DOT((DATA_T4)(k.s012, 0), (DATA_T4)(-input_offset), koff); \
                ARM_DOT((DATA_T4)(k.s345, 0), (DATA_T4)(-input_offset), koff); \
                ARM_DOT((DATA_T4)(k.s678, 0), (DATA_T4)(-input_offset), koff); \
                out4 -= (int4)koff; \
            } \
            if (input_offset != 0 && weight_offset != 0) { \
                out4 += (9 * input_offset * weight_offset); \
            } \
            int4 threshold4; \
            out4 = mul_hi(out4 << left_shift, output_multiplier); \
            out4 = rhadd(out4, 0); \
            threshold4 = threshold_mask + select((int4)(0), (int4)(1), out4 < 0); \
            out4 = (out4 >> right_shift) + select((int4)(0), (int4)(1), (out4 & mask) > threshold4); \
            out4 += output_offset; \
            out4 = max(out4, act_min); \
            out4 = min(out4, act_max); \
            if (g2 + 3 < topWidth) { \
                vstore4((DATA_T4)(out4.s0, out4.s1, out4.s2, out4.s3), 0, output + outputIndex); \
            } else if (g2 + 2 < topWidth) { \
                vstore2((DATA_T2)(out4.s0, out4.s1), 0, output + outputIndex); \
                output[outputIndex + 2] = (DATA_T)out4.s2; \
            } else if (g2 + 1 < topWidth) { \
                vstore2((DATA_T2)(out4.s0, out4.s1), 0, output + outputIndex); \
            } else { \
                output[outputIndex] = (DATA_T)out4.s0; \
            } \
        }

#define Q_DEPTHWISE_CONV_3X3S2(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, topChannel, \
                             depth_multiplier, input_offset, weight_offset, output_offset, output_multiplier, \
                             output_shift, act_min, act_max) \
        int g0 = get_global_id(0); \
        int g1 = get_global_id(1); \
        int g2 = get_global_id(2); \
        bool overflow; \
        long ab_64; \
        int nudge; \
        int mask; \
        int remainder; \
        int threshold; \
        int shift = -output_shift; \
        int left_shift = shift > 0 ? shift : 0; \
        int right_shift = shift > 0 ? 0 : -shift; \
        int acc; \
        DATA_T4 input0; \
        DATA_T4 input1; \
        DATA_T4 input2; \
        DATA_T4 w0; \
        DATA_T4 w1; \
        DATA_T4 w2; \
        int4 out0; \
        int4 out1; \
        int4 out2; \
        if (g2 < topHeight * topWidth) { \
            int inputX = (g2 % topWidth) * 2; \
            int inputY = (g2 / topWidth) * 2; \
            for (int dep = 0; dep < depth_multiplier; dep++) { \
                int out_c = dep + g1 * depth_multiplier; \
                int inputIndex = (g0 * get_global_size(1) + g1) * inputHeight * inputWidth + \
                                 (inputY)*inputWidth + inputX; \
                int outputIndex = \
                    (g0 * get_global_size(1) * depth_multiplier + out_c) * topHeight * topWidth + \
                    g2; \
                acc = bias[out_c]; \
                input0 = vload4(0, input + inputIndex); \
                input1 = vload4(0, input + inputIndex + inputWidth); \
                input2 = vload4(0, input + inputIndex + 2 * inputWidth); \
                w0 = vload4(0, weight + out_c * 9); \
                w1 = vload4(0, weight + out_c * 9 + 3); \
                w2 = vload4(0, weight + out_c * 9 + 6); \
                out0 = (convert_int4(input0) + input_offset) * (convert_int4(w0) + weight_offset); \
                out1 = (convert_int4(input1) + input_offset) * (convert_int4(w1) + weight_offset); \
                out2 = (convert_int4(input2) + input_offset) * (convert_int4(w2) + weight_offset); \
                acc += out0.s0 + out0.s1 + out0.s2 + out1.s0 + out1.s1 + out1.s2 + out2.s0 + \
                       out2.s1 + out2.s2; \
                acc = acc * (1 << left_shift); \
                reQuantized(acc, \
                            output_multiplier, \
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
                output[outputIndex] = (DATA_T)acc; \
            } \
        }

#define Q_DW_CONV_COMMON_PARAMS __global const DATA_T *input, __global const DATA_T *weight, __global const int *bias, \
                                __global DATA_T *output, unsigned int inputHeight, unsigned int inputWidth
#define Q_DW_CONV_4P_PARAMS Q_DW_CONV_COMMON_PARAMS, unsigned int topHeight, unsigned int topWidth, \
                            unsigned int topChannel, unsigned int depth_multiplier
#define Q_DW_CONV_OFFSET_SHIFT_ACT int input_offset, int weight_offset, int output_offset, int output_multiplier, \
                                   int output_shift, int act_min, int act_max
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
#define SIGNED_TO_UNSIGNED(x) x
#define UNSIGNED_TO_SIGNED(x) x

ADD_KERNEL_HEADER(depthwise_conv_unequal_INT8,
                  {DEFINE_SATURATING_ROUNDING_DOUBLING_HIGH_MUL, DEFINE_ROUNDING_DIVIDE_BY_POT, DEFINE_REQUANTIZED})
ADD_SINGLE_KERNEL(depthwise_conv_unequal_INT8, (__global const int *bias,
                                              __global DATA_T *output,
                                              unsigned int start,
                                              int output_offset,
                                              int output_multiplier,
                                              int output_shift,
                                              int act_min,
                                              int act_max) {
    Q_DEPTHWISE_CONV_UNEQUAL(bias, output, start, output_offset, output_multiplier, output_shift, act_min, act_max)
})

ADD_KERNEL_HEADER(depthwise_conv_INT8,
                  {DEFINE_SATURATING_ROUNDING_DOUBLING_HIGH_MUL, DEFINE_ROUNDING_DIVIDE_BY_POT, DEFINE_REQUANTIZED})
ADD_SINGLE_KERNEL(depthwise_conv_INT8, (Q_DW_CONV_COMMON_PARAMS, unsigned int kernelHeight, unsigned int kernelWidth, \
                                        unsigned int strideHeight, unsigned int strideWidth, unsigned int topHeight, \
                                        unsigned int topWidth, unsigned int depth_multiplier, \
                                        Q_DW_CONV_OFFSET_SHIFT_ACT) {
    Q_DEPTHWISE_CONV(input, weight, bias, output, inputHeight, inputWidth, kernelHeight, kernelWidth, strideHeight, \
                   strideWidth, topHeight, topWidth, depth_multiplier, input_offset, weight_offset, output_offset, \
                   output_multiplier, output_shift, act_min, act_max)
})

ADD_KERNEL_HEADER(depthwise_conv_per_channel_INT8,
                  {DEFINE_SATURATING_ROUNDING_DOUBLING_HIGH_MUL, DEFINE_ROUNDING_DIVIDE_BY_POT, DEFINE_REQUANTIZED})
ADD_SINGLE_KERNEL(depthwise_conv_per_channel_INT8, (__global const DATA_T *input, __global const char *weight, \
                                                    __global const int *bias, __global DATA_T *output, \
                                                    unsigned int inputHeight, unsigned int inputWidth, \
                                                    unsigned int kernelHeight, unsigned int kernelWidth, \
                                                    unsigned int strideHeight, unsigned int strideWidth, \
                                                    unsigned int topHeight, unsigned int topWidth, \
                                                    unsigned int depth_multiplier, int input_offset, \
                                                    int weight_offset, int output_offset, \
                                                    __global int *output_multiplier, __global int *output_shift, \
                                                    int act_min, int act_max) {
    Q_DEPTHWISE_CONV_PER_CHANNEL(input, weight, bias, output, inputHeight, inputWidth, kernelHeight, kernelWidth, \
                               strideHeight, strideWidth, topHeight,  topWidth,  depth_multiplier, input_offset, \
                               weight_offset, output_offset, output_multiplier, output_shift, act_min, act_max)
})

ADD_KERNEL_HEADER(depthwise_conv_3x3s1_4P_INT8,
                  {DEFINE_SATURATING_ROUNDING_DOUBLING_HIGH_MUL, DEFINE_ROUNDING_DIVIDE_BY_POT, DEFINE_REQUANTIZED})
ADD_SINGLE_KERNEL(depthwise_conv_3x3s1_4P_INT8, (Q_DW_CONV_4P_PARAMS, Q_DW_CONV_OFFSET_SHIFT_ACT) {
    Q_DEPTHWISE_CONV_3X3S1_4P(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, topChannel, \
                            depth_multiplier, input_offset, weight_offset, output_offset, output_multiplier, \
                            output_shift, act_min, act_max)
})

ADD_SINGLE_KERNEL(depthwise_conv_3x3s1_pad_merge_INT8, (Q_DW_CONV_4P_PARAMS, unsigned int padH, unsigned int padW, \
                                                        Q_DW_CONV_OFFSET_SHIFT_ACT) {
    Q_DEPTHWISE_CONV_3X3S1_PAD_MERGE(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                        topChannel, depth_multiplier, padH, padW, input_offset, weight_offset, \
                                        output_offset, output_multiplier, output_shift, act_min, act_max)
})

ADD_SINGLE_KERNEL(depthwise_conv_3x3s2_pad_merge_INT8, (Q_DW_CONV_4P_PARAMS, unsigned int padH, unsigned int padW, \
                                                        Q_DW_CONV_OFFSET_SHIFT_ACT) {
    Q_DEPTHWISE_CONV_3X3S2_PAD_MERGE(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                   topChannel, depth_multiplier, padH, padW, input_offset, weight_offset, \
                                   output_offset, output_multiplier, output_shift, act_min, act_max)
})

ADD_SINGLE_KERNEL(depthwise_conv_3x3s1d2_pad_merge_INT8, (Q_DW_CONV_4P_PARAMS, unsigned int padH, unsigned int padW, \
                                                          Q_DW_CONV_OFFSET_SHIFT_ACT) {
    Q_DEPTHWISE_CONV_3X3S1D2_PAD_MERGE(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, \
                                     topChannel, depth_multiplier, padH, padW, input_offset, weight_offset, \
                                     output_offset, output_multiplier, output_shift, act_min, act_max)
})

ADD_KERNEL_HEADER(depthwise_conv_3x3s2_INT8,
                  {DEFINE_SATURATING_ROUNDING_DOUBLING_HIGH_MUL, DEFINE_ROUNDING_DIVIDE_BY_POT, DEFINE_REQUANTIZED})
ADD_SINGLE_KERNEL(depthwise_conv_3x3s2_INT8, (Q_DW_CONV_4P_PARAMS, Q_DW_CONV_OFFSET_SHIFT_ACT) {
    Q_DEPTHWISE_CONV_3X3S2(input, weight, bias, output, inputHeight, inputWidth, topHeight, topWidth, topChannel, \
                         depth_multiplier, input_offset, weight_offset, output_offset, output_multiplier, \
                         output_shift, act_min, act_max)
})

#undef UNSIGNED_TO_SIGNED
#undef SIGNED_TO_UNSIGNED
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
#define SIGNED_TO_UNSIGNED(x) (x + 128)
#define UNSIGNED_TO_SIGNED(x) (x - 128)

ADD_KERNEL_HEADER(SIGNEDdepthwise_conv_unequal_INT8,
                  {DEFINE_SATURATING_ROUNDING_DOUBLING_HIGH_MUL, DEFINE_ROUNDING_DIVIDE_BY_POT, DEFINE_REQUANTIZED})
ADD_SINGLE_KERNEL(SIGNEDdepthwise_conv_unequal_INT8, (__global const int *bias,
                                              __global DATA_T *output,
                                              unsigned int start,
                                              int output_offset,
                                              int output_multiplier,
                                              int output_shift,
                                              int act_min,
                                              int act_max) {
    Q_DEPTHWISE_CONV_UNEQUAL(bias, output, start, output_offset, output_multiplier, output_shift, act_min, act_max)
})

ADD_KERNEL_HEADER(SIGNEDdepthwise_conv_INT8,
                  {DEFINE_SATURATING_ROUNDING_DOUBLING_HIGH_MUL, DEFINE_ROUNDING_DIVIDE_BY_POT, DEFINE_REQUANTIZED})
ADD_SINGLE_KERNEL(SIGNEDdepthwise_conv_INT8, (Q_DW_CONV_COMMON_PARAMS, unsigned int kernelHeight, \
                                              unsigned int kernelWidth, unsigned int strideHeight, \
                                              unsigned int strideWidth, unsigned int topHeight, \
                                              unsigned int topWidth, unsigned int depth_multiplier, \
                                              Q_DW_CONV_OFFSET_SHIFT_ACT) {
    Q_DEPTHWISE_CONV(input, weight, bias, output, inputHeight, inputWidth, kernelHeight, kernelWidth, strideHeight, \
                   strideWidth, topHeight, topWidth, depth_multiplier, input_offset, weight_offset, output_offset, \
                   output_multiplier, output_shift, act_min, act_max)
})

ADD_KERNEL_HEADER(SIGNEDdepthwise_conv_per_channel_INT8,
                  {DEFINE_SATURATING_ROUNDING_DOUBLING_HIGH_MUL, DEFINE_ROUNDING_DIVIDE_BY_POT, DEFINE_REQUANTIZED})
ADD_SINGLE_KERNEL(SIGNEDdepthwise_conv_per_channel_INT8, (Q_DW_CONV_COMMON_PARAMS, unsigned int kernelHeight, \
                                                          unsigned int kernelWidth, unsigned int strideHeight, \
                                                          unsigned int strideWidth, unsigned int topHeight, \
                                                          unsigned int topWidth, unsigned int depth_multiplier, \
                                                          int input_offset, int weight_offset, int output_offset, \
                                                          __global int *output_multiplier, __global int *output_shift, \
                                                          int act_min, int act_max) {
    Q_DEPTHWISE_CONV_PER_CHANNEL(input, weight, bias, output, inputHeight, inputWidth, kernelHeight, kernelWidth, \
                               strideHeight, strideWidth, topHeight,  topWidth,  depth_multiplier, input_offset, \
                               weight_offset, output_offset, output_multiplier, output_shift, act_min, act_max)
})

#undef UNSIGNED_TO_SIGNED
#undef SIGNED_TO_UNSIGNED
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
