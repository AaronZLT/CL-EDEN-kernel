#include "userdriver/gpu/common/CLKernels.hpp"

namespace enn{
namespace ud {
namespace gpu {

#define Q_PAD(input, temp_out, byte_zero, padTop, padRight, padBottom, padLeft, inPutWidth, inPutHeight) \
        int outputWidth = inPutWidth + padRight + padLeft; \
        int outputHeight = inPutHeight + padBottom + padTop; \
        if (get_global_id(2) < outputWidth * outputHeight) { \
            int rowIndex = get_global_id(2) / outputWidth; \
            int colIndex = get_global_id(2) % outputWidth; \
            int channelIndex = get_global_id(0) * get_global_size(1) + get_global_id(1); \
            int outputIndex = channelIndex * outputWidth * outputHeight + get_global_id(2); \
            if (rowIndex < padTop || rowIndex >= inPutHeight + padTop || colIndex < padLeft || \
                colIndex >= inPutWidth + padLeft) { \
                temp_out[outputIndex] = byte_zero; \
            } else { \
                int inputIndex = channelIndex * inPutHeight * inPutWidth + \
                                 (rowIndex - padTop) * inPutWidth + colIndex - padLeft; \
                temp_out[outputIndex] = input[inputIndex]; \
            } \
        }

#define Q_PAD_OPT(input, pad_out, byte_zero, padT, padL, inputHeight, inputWidth, outputHeight, outputWidth) \
        int outputNC = get_global_id(0); \
        int outputH = get_global_id(1); \
        int outputW = get_global_id(2) * 16; \
        if (outputH < outputHeight && outputW < outputWidth) { \
            int inputId = outputNC * inputHeight * inputWidth + (outputH - padT) * inputWidth + \
                          (outputW - padL); \
            int outputId = outputNC * outputHeight * outputWidth + outputH * outputWidth + outputW; \
            DATA_T16 in = (DATA_T16)byte_zero; \
            if ((outputH - padT) >= 0 && (outputH - padT) < inputHeight) { \
                in = vload16(0, input + inputId); \
            } \
            if ((outputW - padL) < 0 || (outputW - padL) >= inputWidth) \
                in.s0 = (DATA_T)byte_zero; \
            if ((outputW - padL + 1) < 0 || (outputW - padL + 1) >= inputWidth) \
                in.s1 = (DATA_T)byte_zero; \
            if ((outputW - padL + 2) < 0 || (outputW - padL + 2) >= inputWidth) \
                in.s2 = (DATA_T)byte_zero; \
            if ((outputW - padL + 3) < 0 || (outputW - padL + 3) >= inputWidth) \
                in.s3 = (DATA_T)byte_zero; \
            if ((outputW - padL + 4) < 0 || (outputW - padL + 4) >= inputWidth) \
                in.s4 = (DATA_T)byte_zero; \
            if ((outputW - padL + 5) < 0 || (outputW - padL + 5) >= inputWidth) \
                in.s5 = (DATA_T)byte_zero; \
            if ((outputW - padL + 6) < 0 || (outputW - padL + 6) >= inputWidth) \
                in.s6 = (DATA_T)byte_zero; \
            if ((outputW - padL + 7) < 0 || (outputW - padL + 7) >= inputWidth) \
                in.s7 = (DATA_T)byte_zero; \
            if ((outputW - padL + 8) < 0 || (outputW - padL + 8) >= inputWidth) \
                in.s8 = (DATA_T)byte_zero; \
            if ((outputW - padL + 9) < 0 || (outputW - padL + 9) >= inputWidth) \
                in.s9 = (DATA_T)byte_zero; \
            if ((outputW - padL + 10) < 0 || (outputW - padL + 10) >= inputWidth) \
                in.sa = (DATA_T)byte_zero; \
            if ((outputW - padL + 11) < 0 || (outputW - padL + 11) >= inputWidth) \
                in.sb = (DATA_T)byte_zero; \
            if ((outputW - padL + 12) < 0 || (outputW - padL + 12) >= inputWidth) \
                in.sc = (DATA_T)byte_zero; \
            if ((outputW - padL + 13) < 0 || (outputW - padL + 13) >= inputWidth) \
                in.sd = (DATA_T)byte_zero; \
            if ((outputW - padL + 14) < 0 || (outputW - padL + 14) >= inputWidth) \
                in.se = (DATA_T)byte_zero; \
            if ((outputW - padL + 15) < 0 || (outputW - padL + 15) >= inputWidth) \
                in.sf = (DATA_T)byte_zero; \
            if (outputW + 15 < outputWidth) { \
                vstore16(in, 0, pad_out + outputId); \
            } else if (outputW + 14 < outputWidth) { \
                vstore8(in.s01234567, 0, pad_out + outputId); \
                vstore4(in.s89ab, 0, pad_out + outputId + 8); \
                vstore3(in.scde, 0, pad_out + outputId + 12); \
            } else if (outputW + 13 < outputWidth) { \
                vstore8(in.s01234567, 0, pad_out + outputId); \
                vstore4(in.s89ab, 0, pad_out + outputId + 8); \
                vstore2(in.scd, 0, pad_out + outputId + 12); \
            } else if (outputW + 12 < outputWidth) { \
                vstore8(in.s01234567, 0, pad_out + outputId); \
                vstore4(in.s89ab, 0, pad_out + outputId + 8); \
                pad_out[outputId + 12] = in.sc; \
            } else if (outputW + 11 < outputWidth) { \
                vstore8(in.s01234567, 0, pad_out + outputId); \
                vstore4(in.s89ab, 0, pad_out + outputId + 8); \
            } else if (outputW + 10 < outputWidth) { \
                vstore8(in.s01234567, 0, pad_out + outputId); \
                vstore3(in.s89a, 0, pad_out + outputId + 8); \
            } else if (outputW + 9 < outputWidth) { \
                vstore8(in.s01234567, 0, pad_out + outputId); \
                vstore2(in.s89, 0, pad_out + outputId + 8); \
            } else if (outputW + 8 < outputWidth) { \
                vstore8(in.s01234567, 0, pad_out + outputId); \
                pad_out[outputId + 8] = in.s8; \
            } else if (outputW + 7 < outputWidth) { \
                vstore8(in.s01234567, 0, pad_out + outputId); \
            } else if (outputW + 6 < outputWidth) { \
                vstore4(in.s0123, 0, pad_out + outputId); \
                vstore3(in.s456, 0, pad_out + outputId + 4); \
            } else if (outputW + 5 < outputWidth) { \
                vstore4(in.s0123, 0, pad_out + outputId); \
                vstore2(in.s45, 0, pad_out + outputId + 4); \
            } else if (outputW + 4 < outputWidth) { \
                vstore4(in.s0123, 0, pad_out + outputId); \
                pad_out[outputId + 4] = in.s4; \
            } else if (outputW + 3 < outputWidth) { \
                vstore4(in.s0123, 0, pad_out + outputId); \
            } else if (outputW + 2 < outputWidth) { \
                vstore3(in.s012, 0, pad_out + outputId); \
            } else if (outputW + 1 < outputWidth) { \
                vstore2(in.s01, 0, pad_out + outputId); \
            } else if (outputW < outputWidth) { \
                pad_out[outputId] = in.s0; \
            } \
        }

#define CONVERT_OPTIMIZED(input, output, kernelWidth, kernelHeight, strideWidth, strideHeight, inputHeight, \
                            inputWidth, inputChannel, alignHeight, alignWidth, outputWidth, outputHeight) \
        int globalID0 = get_global_id(0); \
        int globalID1 = get_global_id(1); \
        int globalID2 = get_global_id(2); \
        if (globalID2 < inputChannel * kernelHeight) { \
            int outputIndex = globalID0 * alignHeight * alignWidth + globalID1 * alignWidth + \
                              globalID2 * kernelWidth; \
            int inputIndex = globalID0 * inputWidth * inputHeight * inputChannel + \
                             globalID2 / kernelHeight * inputWidth * inputHeight + \
                             globalID1 / outputWidth * strideHeight * inputWidth + \
                             globalID1 % outputWidth * strideWidth + \
                             globalID2 % kernelHeight * inputWidth; \
            if (kernelWidth == 5) { \
                vstore4(vload4(0, input + inputIndex), 0, output + outputIndex); \
                output[outputIndex + 4] = input[inputIndex + 4]; \
            } else if (kernelWidth == 3) { \
                vstore3(vload3(0, input + inputIndex), 0, output + outputIndex); \
            } else if (kernelWidth == 1) { \
                output[outputIndex] = input[inputIndex]; \
            } else if (kernelWidth == 7) { \
                vstore4(vload4(0, input + inputIndex), 0, output + outputIndex); \
                vstore3(vload3(0, input + inputIndex + 4), 0, output + outputIndex + 4); \
            } else { \
                for (int i = 0; i < kernelWidth; i++) { \
                    output[outputIndex++] = input[inputIndex++]; \
                } \
            } \
        }

#define CONVERTBLOCKED2(input, output, kernel_sizeW, kernel_sizeH, strideW, strideH, inputHeight, inputWidth, \
                        inputnum, alignHeight, alignWidth, group, outputWidth, groupNum, outputHeight) \
        int globalID0 = get_global_id(0);  \
        int globalID1 = get_global_id(1);  \
        int globalID2 = get_global_id(2);  \
        if (globalID1 < outputWidth * outputHeight && \
            globalID2 < inputnum * kernel_sizeH * kernel_sizeW) { \
            int blk_size = kernel_sizeW * kernel_sizeH; \
            int i = globalID1 * alignWidth + globalID2 + alignHeight * alignWidth * group + \
                    globalID0 * alignHeight * alignWidth * groupNum; \
            int pic_off = globalID2 / blk_size; \
            int index = globalID2 % blk_size; \
            int row_off = index / kernel_sizeW; \
            int col_off = index % kernel_sizeW; \
            int row = globalID1 / outputWidth; \
            int col = globalID1 % outputWidth; \
            output[i] = input[(row_off + row * strideH) * inputWidth + col_off + col * strideW + \
                              pic_off * inputWidth * inputHeight + \
                              inputWidth * inputHeight * inputnum * group + \
                              globalID0 * inputWidth * inputHeight * inputnum * groupNum]; \
        }

#define Q_CONVERT_INPUT_4_THREAD_8_WITHPAD(input, output, kernel_sizeW, kernel_sizeH, padT, padL, strideW, strideH, \
                                            inputHeight, inputWidth, inputnum, alignHeight, alignWidth, outputWidth, \
                                            outputHeight, collapseHeight, byte_zero) \
        int outputRowIndex = get_global_id(0); \
        int inputChannelIndex = get_global_id(1); \
        int batchIndex = get_global_id(2); \
        if (outputRowIndex < outputWidth * outputHeight) { \
            int outputColBase = inputChannelIndex * kernel_sizeW * kernel_sizeH; \
            int outputBase = batchIndex * alignHeight * alignWidth + \
                             (outputRowIndex / collapseHeight) * collapseHeight * alignWidth + \
                             outputRowIndex % collapseHeight / 4 * 16 + \
                             outputRowIndex % collapseHeight % 4 * 4; \
            int inputBase = (batchIndex * inputnum + inputChannelIndex) * inputHeight * inputWidth; \
            int interval = 4 * collapseHeight; \
            int inputRow = outputRowIndex / outputWidth * strideH - padT; \
            int inputCol = outputRowIndex % outputWidth * strideW - padL; \
            for (int row = 0; row < kernel_sizeH; row++) { \
                for (int col = 0; col < kernel_sizeW; col++) { \
                    int outputCol = outputColBase + row * kernel_sizeW + col; \
                    int outputIndex = outputBase + (outputCol >> 2) * interval + outputCol % 4; \
                    int inputRowIter = inputRow + row; \
                    int inputColIter = inputCol + col; \
                    if (inputRowIter >= 0 && inputColIter >= 0 && inputRowIter < inputHeight && \
                        inputColIter < inputWidth) { \
                        int inputIndex = inputBase + inputRowIter * inputWidth + inputColIter; \
                        output[outputIndex] = input[inputIndex]; \
                    } else { \
                        output[outputIndex] = byte_zero; \
                    } \
                } \
            } \
        }

#define Q_CONVERT_INPUT_4_THREAD_8_WITHPAD_VALHALL(input, output, kernel_sizeW, kernel_sizeH, padT, padL, strideW, \
                                                    strideH, inputHeight, inputWidth, inputnum, alignHeight, \
                                                    alignWidth, outputWidth, outputHeight, collapseHeight, byte_zero) \
        if (get_global_id(0) < outputWidth * outputHeight) { \
            int outputColBase = get_global_id(1) * kernel_sizeW * kernel_sizeH; \
            int outputBase = get_global_id(2) * alignHeight * alignWidth + \
                             (get_global_id(0) / collapseHeight) * collapseHeight * alignWidth + \
                             get_global_id(0) % collapseHeight / 4 * 16 + \
                             get_global_id(0) % collapseHeight % 4 * 4; \
            int inputBase = \
                (get_global_id(2) * inputnum + get_global_id(1)) * inputHeight * inputWidth; \
            int inputRow = get_global_id(0) / outputWidth * strideH - padT; \
            int inputCol = get_global_id(0) % outputWidth * strideW - padL; \
            for (int row = 0; row < kernel_sizeH; row++) { \
                for (int col = 0; col < kernel_sizeW; col++) { \
                    int outputCol = outputColBase + row * kernel_sizeW + col; \
                    int outputIndex = \
                        outputBase + (outputCol >> 2) * 4 * collapseHeight + outputCol % 4; \
                    int inputRowIter = inputRow + row; \
                    int inputColIter = inputCol + col; \
                    if (inputRowIter >= 0 && inputColIter >= 0 && inputRowIter < inputHeight && \
                        inputColIter < inputWidth) { \
                        int inputIndex = inputBase + inputRowIter * inputWidth + inputColIter; \
                        output[outputIndex] = input[inputIndex]; \
                    } else { \
                        output[outputIndex] = byte_zero; \
                    } \
                } \
            } \
        }

/********  QUANTIZED KERNELS ********/
#define DATA_T uchar
#define DATA_T2 uchar2
#define DATA_T3 uchar3
#define DATA_T4 uchar4
#define DATA_T8 uchar8
#define DATA_T16 uchar16

ADD_SINGLE_KERNEL(pad_INT8, (__global const DATA_T *input,
                           __global DATA_T *temp_out,
                           DATA_T byte_zero,
                           unsigned int padTop,
                           unsigned int padRight,
                           unsigned int padBottom,
                           unsigned int padLeft,
                           unsigned int inPutWidth,
                           unsigned int inPutHeight) {
    Q_PAD(input, temp_out, byte_zero, padTop, padRight, padBottom, padLeft, inPutWidth, inPutHeight)
})

ADD_SINGLE_KERNEL(pad_opt_INT8, (__global const DATA_T *input,
                               __global DATA_T *pad_out,
                               int byte_zero,
                               int padT,
                               int padL,
                               int inputHeight,
                               int inputWidth,
                               int outputHeight,
                               int outputWidth) {
    Q_PAD_OPT(input, pad_out, byte_zero, padT, padL, inputHeight, inputWidth, outputHeight, outputWidth)
})

ADD_SINGLE_KERNEL(convert_optimized_INT8, (__global const DATA_T *input,
                                         __global DATA_T *output,
                                         unsigned int kernelWidth,
                                         unsigned int kernelHeight,
                                         unsigned int strideWidth,
                                         unsigned int strideHeight,
                                         unsigned int inputHeight,
                                         unsigned int inputWidth,
                                         unsigned int inputChannel,
                                         unsigned int alignHeight,
                                         unsigned int alignWidth,
                                         unsigned int outputWidth,
                                         unsigned int outputHeight) {
    CONVERT_OPTIMIZED(input, output, kernelWidth, kernelHeight, strideWidth, strideHeight, inputHeight, \
                        inputWidth, inputChannel, alignHeight, alignWidth, outputWidth, outputHeight)
})

ADD_SINGLE_KERNEL(convertBlocked2_INT8, (__global const DATA_T *input,
                                       __global DATA_T *output,
                                       unsigned int kernel_sizeW,
                                       unsigned int kernel_sizeH,
                                       unsigned int strideW,
                                       unsigned int strideH,
                                       unsigned int inputHeight,
                                       unsigned int inputWidth,
                                       unsigned int inputnum,
                                       unsigned int alignHeight,
                                       unsigned int alignWidth,
                                       unsigned int group,
                                       unsigned int outputWidth,
                                       unsigned int groupNum,
                                       unsigned int outputHeight) {
    CONVERTBLOCKED2(input, output, kernel_sizeW, kernel_sizeH, strideW, strideH, inputHeight, inputWidth, inputnum, \
                        alignHeight, alignWidth, group, outputWidth, groupNum, outputHeight)
})

ADD_SINGLE_KERNEL(convert_input_4_thread_8_withpad_INT8, (__global const DATA_T *input,
                                                        __global DATA_T *output,
                                                        unsigned int kernel_sizeW,
                                                        unsigned int kernel_sizeH,
                                                        int padT,
                                                        int padL,
                                                        unsigned int strideW,
                                                        unsigned int strideH,
                                                        unsigned int inputHeight,
                                                        unsigned int inputWidth,
                                                        unsigned int inputnum,
                                                        unsigned int alignHeight,
                                                        unsigned int alignWidth,
                                                        unsigned int outputWidth,
                                                        unsigned int outputHeight,
                                                        unsigned int collapseHeight,
                                                        DATA_T byte_zero) {
    Q_CONVERT_INPUT_4_THREAD_8_WITHPAD(input, output, kernel_sizeW, kernel_sizeH, padT, padL, strideW, strideH, \
            inputHeight, inputWidth, inputnum, alignHeight, alignWidth, outputWidth, outputHeight, \
            collapseHeight, byte_zero)
})

ADD_SINGLE_KERNEL(convert_input_4_thread_8_withpad_valhall_INT8, (__global const DATA_T *input,
                                                                __global DATA_T *output,
                                                                unsigned int kernel_sizeW,
                                                                unsigned int kernel_sizeH,
                                                                int padT,
                                                                int padL,
                                                                unsigned int strideW,
                                                                unsigned int strideH,
                                                                unsigned int inputHeight,
                                                                unsigned int inputWidth,
                                                                unsigned int inputnum,
                                                                unsigned int alignHeight,
                                                                unsigned int alignWidth,
                                                                unsigned int outputWidth,
                                                                unsigned int outputHeight,
                                                                unsigned int collapseHeight,
                                                                DATA_T byte_zero) {
    Q_CONVERT_INPUT_4_THREAD_8_WITHPAD_VALHALL(input, output, kernel_sizeW, kernel_sizeH, padT, padL, strideW, \
            strideH, inputHeight, inputWidth, inputnum, alignHeight, alignWidth, outputWidth, outputHeight, \
            collapseHeight, byte_zero)
})


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

ADD_SINGLE_KERNEL(SIGNEDpad_INT8, (__global const DATA_T *input,
                           __global DATA_T *temp_out,
                           DATA_T byte_zero,
                           unsigned int padTop,
                           unsigned int padRight,
                           unsigned int padBottom,
                           unsigned int padLeft,
                           unsigned int inPutWidth,
                           unsigned int inPutHeight) {
    Q_PAD(input, temp_out, byte_zero, padTop, padRight, padBottom, padLeft, inPutWidth, inPutHeight)
})

ADD_SINGLE_KERNEL(SIGNEDpad_opt_INT8, (__global const DATA_T *input,
                               __global DATA_T *pad_out,
                               int byte_zero,
                               int padT,
                               int padL,
                               int inputHeight,
                               int inputWidth,
                               int outputHeight,
                               int outputWidth) {
    Q_PAD_OPT(input, pad_out, byte_zero, padT, padL, inputHeight, inputWidth, outputHeight, outputWidth)
})

ADD_SINGLE_KERNEL(SIGNEDconvert_optimized_INT8, (__global const DATA_T *input,
                                         __global DATA_T *output,
                                         unsigned int kernelWidth,
                                         unsigned int kernelHeight,
                                         unsigned int strideWidth,
                                         unsigned int strideHeight,
                                         unsigned int inputHeight,
                                         unsigned int inputWidth,
                                         unsigned int inputChannel,
                                         unsigned int alignHeight,
                                         unsigned int alignWidth,
                                         unsigned int outputWidth,
                                         unsigned int outputHeight) {
    CONVERT_OPTIMIZED(input, output, kernelWidth, kernelHeight, strideWidth, strideHeight, inputHeight, \
                        inputWidth, inputChannel, alignHeight, alignWidth, outputWidth, outputHeight)
})

ADD_SINGLE_KERNEL(SIGNEDconvertBlocked2_INT8, (__global const DATA_T *input,
                                       __global DATA_T *output,
                                       unsigned int kernel_sizeW,
                                       unsigned int kernel_sizeH,
                                       unsigned int strideW,
                                       unsigned int strideH,
                                       unsigned int inputHeight,
                                       unsigned int inputWidth,
                                       unsigned int inputnum,
                                       unsigned int alignHeight,
                                       unsigned int alignWidth,
                                       unsigned int group,
                                       unsigned int outputWidth,
                                       unsigned int groupNum,
                                       unsigned int outputHeight) {
    CONVERTBLOCKED2(input, output, kernel_sizeW, kernel_sizeH, strideW, strideH, inputHeight, inputWidth, \
                        inputnum, alignHeight, alignWidth, group, outputWidth, groupNum, outputHeight)
})

ADD_SINGLE_KERNEL(SIGNEDconvert_input_4_thread_8_withpad_INT8, (__global const DATA_T *input,
                                                        __global DATA_T *output,
                                                        unsigned int kernel_sizeW,
                                                        unsigned int kernel_sizeH,
                                                        int padT,
                                                        int padL,
                                                        unsigned int strideW,
                                                        unsigned int strideH,
                                                        unsigned int inputHeight,
                                                        unsigned int inputWidth,
                                                        unsigned int inputnum,
                                                        unsigned int alignHeight,
                                                        unsigned int alignWidth,
                                                        unsigned int outputWidth,
                                                        unsigned int outputHeight,
                                                        unsigned int collapseHeight,
                                                        DATA_T byte_zero) {
    Q_CONVERT_INPUT_4_THREAD_8_WITHPAD(input, output, kernel_sizeW, kernel_sizeH, padT, padL, strideW, \
                                        strideH, inputHeight, inputWidth, inputnum, alignHeight, alignWidth, \
                                        outputWidth, outputHeight, collapseHeight, byte_zero)
})

ADD_SINGLE_KERNEL(SIGNEDconvert_input_4_thread_8_withpad_valhall_INT8, (__global const DATA_T *input,
                                                                __global DATA_T *output,
                                                                unsigned int kernel_sizeW,
                                                                unsigned int kernel_sizeH,
                                                                int padT,
                                                                int padL,
                                                                unsigned int strideW,
                                                                unsigned int strideH,
                                                                unsigned int inputHeight,
                                                                unsigned int inputWidth,
                                                                unsigned int inputnum,
                                                                unsigned int alignHeight,
                                                                unsigned int alignWidth,
                                                                unsigned int outputWidth,
                                                                unsigned int outputHeight,
                                                                unsigned int collapseHeight,
                                                                DATA_T byte_zero) {
    Q_CONVERT_INPUT_4_THREAD_8_WITHPAD_VALHALL(input, output, kernel_sizeW, kernel_sizeH, padT, padL, strideW, \
                                strideH, inputHeight, inputWidth, inputnum, alignHeight, alignWidth, outputWidth, \
                                outputHeight, collapseHeight, byte_zero)
})

#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // char

ADD_SINGLE_KERNEL(pad_copy_align_c4_INT8, (__global const unsigned char *input,
                                         __global unsigned char *pad_out,
                                         unsigned char byte_zero,
                                         int padT,
                                         int padL,
                                         int inputChannel,
                                         int inputHeight,
                                         int inputWidth,
                                         int outputChannel,
                                         int outputHeight,
                                         int outputWidth) {
        int inputW = get_global_id(0) * 4;
        int inputH = get_global_id(1);
        int inputC = get_global_id(2) % ((inputChannel + 3) / 4) * 4;
        int inputN = get_global_id(2) / ((inputChannel + 3) / 4);

        if (inputW < inputWidth) {
            int outputW = (inputW + padL) * 4;
            int outputH = inputH + padT;
            int outputC = inputC / 4;
            int inputBase = inputN * inputChannel * inputHeight * inputWidth +
                            inputC * inputHeight * inputWidth + inputH * inputWidth + inputW;

            uchar4 input_0 = byte_zero;
            uchar4 input_1 = byte_zero;
            uchar4 input_2 = byte_zero;
            uchar4 input_3 = byte_zero;

            input_0 = vload4(0, input + inputBase);
            if (inputC + 1 < inputChannel) {
                input_1 = vload4(0, input + inputBase + inputHeight * inputWidth);
            }
            if (inputC + 2 < inputChannel) {
                input_2 = vload4(0, input + inputBase + 2 * inputHeight * inputWidth);
            }
            if (inputC + 3 < inputChannel) {
                input_3 = vload4(0, input + inputBase + 3 * inputHeight * inputWidth);
            }

            int outputBase = inputN * outputChannel * outputHeight * outputWidth +
                             outputC * outputHeight * outputWidth + outputH * outputWidth + outputW;

            uchar16 outputVector = (uchar16)(input_0.s0,
                                             input_1.s0,
                                             input_2.s0,
                                             input_3.s0,
                                             input_0.s1,
                                             input_1.s1,
                                             input_2.s1,
                                             input_3.s1,
                                             input_0.s2,
                                             input_1.s2,
                                             input_2.s2,
                                             input_3.s2,
                                             input_0.s3,
                                             input_1.s3,
                                             input_2.s3,
                                             input_3.s3);
            if (inputW + 3 < inputWidth) {
                vstore16(outputVector, 0, pad_out + outputBase);
            } else if (inputW + 2 < inputWidth) {
                vstore8(outputVector.lo, 0, pad_out + outputBase);
                vstore4(outputVector.hi.lo, 0, pad_out + outputBase + 8);
            } else if (inputW + 1 < inputWidth) {
                vstore8(outputVector.lo, 0, pad_out + outputBase);
            } else {
                vstore4(outputVector.lo.lo, 0, pad_out + outputBase);
            }
        }
})

ADD_SINGLE_KERNEL(convert_input_4_thread_8_withpad_9x9_INT8, (__global const unsigned char *input,
                                                            __global unsigned char *output,
                                                            unsigned int inputHeight,
                                                            unsigned int inputWidth,
                                                            unsigned int inputnum,
                                                            unsigned int alignHeight,
                                                            unsigned int alignWidth,
                                                            unsigned int collapseHeight,
                                                            unsigned char byte_zero) {
        int HWIndex = get_global_id(0) * 8;
        int CinKKIndex = get_global_id(1);
        int batchIndex = get_global_id(2);
        if (HWIndex < inputHeight * inputWidth && CinKKIndex < inputnum * 81) {
            int inputChannelIndex = CinKKIndex / 81;
            int KIndex = CinKKIndex % 81;
            int Row = HWIndex / inputWidth + KIndex / 9 - 4;
            int Col = HWIndex % inputWidth + KIndex % 9 - 4;
            int interval = 4 * collapseHeight;
            int inputBase = (batchIndex * inputnum + inputChannelIndex) * inputHeight *
                            inputWidth;  // 384 // -pad
            int outRow = HWIndex / collapseHeight;
            int outCol =
                HWIndex % collapseHeight / 4 * 16 + CinKKIndex % 4 + CinKKIndex / 4 * interval;  //
            int outputBase = batchIndex * alignHeight * alignWidth +
                             outRow * alignWidth * collapseHeight + outCol;
            uchar8 input_0 = (uchar8)byte_zero;
            if (Row >= 0 && Row < inputHeight) {
                input_0 = vload8(0, input + inputBase + Row * inputWidth + Col);
            }
            if (Col < 0) {
                input_0.s0 = byte_zero;
            }
            if (Col + 1 < 0) {
                input_0.s1 = byte_zero;
            }
            if (Col + 2 < 0) {
                input_0.s2 = byte_zero;
            }
            if (Col + 3 < 0) {
                input_0.s3 = byte_zero;
            }
            if (Col + 4 >= inputWidth) {
                input_0.s4 = byte_zero;
            }
            if (Col + 5 >= inputWidth) {
                input_0.s5 = byte_zero;
            }
            if (Col + 6 >= inputWidth) {
                input_0.s6 = byte_zero;
            }
            if (Col + 7 >= inputWidth) {
                input_0.s7 = byte_zero;
            }
            //        vstore4(input_0, 0, output + outputBase);
            output[outputBase] = input_0.s0;
            output[outputBase + 4] = input_0.s1;
            output[outputBase + 8] = input_0.s2;
            output[outputBase + 12] = input_0.s3;
            output[outputBase + 16] = input_0.s4;
            output[outputBase + 16 + 4] = input_0.s5;
            output[outputBase + 16 + 8] = input_0.s6;
            output[outputBase + 16 + 12] = input_0.s7;
        }
})

ADD_SINGLE_KERNEL(convert_input_4_thread_8_withpad_5x5_INT8, (__global const unsigned char *input,
                                                            __global unsigned char *output,
                                                            unsigned int inputHeight,
                                                            unsigned int inputWidth,
                                                            unsigned int inputnum,
                                                            unsigned int alignHeight,
                                                            unsigned int alignWidth,
                                                            unsigned int collapseHeight,
                                                            unsigned char byte_zero) {
        int HWIndex = get_global_id(0) * 8;
        int CinKKIndex = get_global_id(1);
        int batchIndex = get_global_id(2);
        if (HWIndex < inputHeight * inputWidth && CinKKIndex < inputnum * 25) {
            int inputChannelIndex = CinKKIndex / 25;
            int KIndex = CinKKIndex % 25;
            int Row = HWIndex / inputWidth + KIndex / 5 - 2;
            int Col = HWIndex % inputWidth + KIndex % 5 - 2;
            int interval = 4 * collapseHeight;
            int inputBase = (batchIndex * inputnum + inputChannelIndex) * inputHeight *
                            inputWidth;  // 384 // -pad
            int outRow = HWIndex / collapseHeight;
            int outCol =
                HWIndex % collapseHeight / 4 * 16 + CinKKIndex % 4 + CinKKIndex / 4 * interval;  //
            int outputBase = batchIndex * alignHeight * alignWidth +
                             outRow * alignWidth * collapseHeight + outCol;
            uchar8 input_0 = (uchar8)byte_zero;
            if (Row >= 0 && Row < inputHeight) {
                input_0 = vload8(0, input + inputBase + Row * inputWidth + Col);
            }
            if (Col < 0) {
                input_0.s0 = byte_zero;
            }
            if (Col + 1 < 0) {
                input_0.s1 = byte_zero;
            }
            if (Col + 6 >= inputWidth) {
                input_0.s6 = byte_zero;
            }
            if (Col + 7 >= inputWidth) {
                input_0.s7 = byte_zero;
            }
            //        vstore4(input_0, 0, output + outputBase);
            output[outputBase] = input_0.s0;
            output[outputBase + 4] = input_0.s1;
            output[outputBase + 8] = input_0.s2;
            output[outputBase + 12] = input_0.s3;
            output[outputBase + 16] = input_0.s4;
            output[outputBase + 16 + 4] = input_0.s5;
            output[outputBase + 16 + 8] = input_0.s6;
            output[outputBase + 16 + 12] = input_0.s7;
        }
})

ADD_SINGLE_KERNEL(convert_input_4_thread_8_withpad_3x3_opt_INT8, (__global const unsigned char *input,
                                                                __global unsigned char *output,
                                                                unsigned int inputHeight,
                                                                unsigned int inputWidth,
                                                                unsigned int inputnum,
                                                                unsigned int alignHeight,
                                                                unsigned int alignWidth,
                                                                unsigned int collapseHeight,
                                                                unsigned char byte_zero) {
        int HWIndex = get_global_id(0) * 8;
        int CinKKIndex = get_global_id(1);
        int batchIndex = get_global_id(2);
        if (HWIndex < inputHeight * inputWidth && CinKKIndex < inputnum * 9) {
            int inputChannelIndex = CinKKIndex / 9;
            int KIndex = CinKKIndex % 9;
            int Row = HWIndex / inputWidth + KIndex / 3 - 1;
            int Col = HWIndex % inputWidth + KIndex % 3 - 1;
            int interval = 4 * collapseHeight;
            int inputBase = (batchIndex * inputnum + inputChannelIndex) * inputHeight *
                            inputWidth;  // 384 // -pad
            int outRow = HWIndex / collapseHeight;
            int outCol =
                HWIndex % collapseHeight / 4 * 16 + CinKKIndex % 4 + CinKKIndex / 4 * interval;  //
            int outputBase = batchIndex * alignHeight * alignWidth +
                             outRow * alignWidth * collapseHeight + outCol;
            uchar8 input_0 = (uchar8)byte_zero;
            if (Row >= 0 && Row < inputHeight) {
                input_0 = vload8(0, input + inputBase + Row * inputWidth + Col);
            }
            if (Col < 0) {
                input_0.s0 = byte_zero;
            }
            if (Col + 7 >= inputWidth) {
                input_0.s7 = byte_zero;
            }
            //        vstore4(input_0, 0, output + outputBase);
            output[outputBase] = input_0.s0;
            output[outputBase + 4] = input_0.s1;
            output[outputBase + 8] = input_0.s2;
            output[outputBase + 12] = input_0.s3;
            output[outputBase + 16] = input_0.s4;
            output[outputBase + 16 + 4] = input_0.s5;
            output[outputBase + 16 + 8] = input_0.s6;
            output[outputBase + 16 + 12] = input_0.s7;
        }
})

ADD_SINGLE_KERNEL(convert_input_4_thread_8_withpad_3x3_INT8, (__global const unsigned char *input,
                                                            __global unsigned char *output,
                                                            int padT,
                                                            int padL,
                                                            unsigned int strideW,
                                                            unsigned int strideH,
                                                            unsigned int inputHeight,
                                                            unsigned int inputWidth,
                                                            unsigned int inputnum,
                                                            unsigned int alignHeight,
                                                            unsigned int alignWidth,
                                                            unsigned int outputWidth,
                                                            unsigned int outputHeight,
                                                            unsigned int collapseHeight,
                                                            unsigned char byte_zero) {
        int outputRowIndex = get_global_id(0);
        int inputChannelIndex = get_global_id(1);
        int batchIndex = get_global_id(2);

        if (outputRowIndex < outputWidth * outputHeight) {
            int outputColBase = inputChannelIndex * 9;
            int outputBase = batchIndex * alignHeight * alignWidth +
                             (outputRowIndex / collapseHeight) * collapseHeight * alignWidth +
                             outputRowIndex % collapseHeight / 4 * 16 +
                             outputRowIndex % collapseHeight % 4 * 4;

            int inputBase = (batchIndex * inputnum + inputChannelIndex) * inputHeight * inputWidth;
            int interval = 4 * collapseHeight;

            int inputRow = outputRowIndex / outputWidth * strideH - padT;
            int inputCol = outputRowIndex % outputWidth * strideW - padL;

            uchar3 row_1 = byte_zero;
            uchar3 row_2;
            uchar3 row_3 = byte_zero;

            if (inputRow >= 0) {
                row_1 = vload3(0, input + inputBase + inputRow * inputWidth + inputCol);
            }
            row_2 = vload3(0, input + inputBase + (inputRow + 1) * inputWidth + inputCol);
            if (inputRow + 2 < inputHeight) {
                row_3 = vload3(0, input + inputBase + (inputRow + 2) * inputWidth + inputCol);
            }

            if (inputCol < 0) {
                row_1.s0 = byte_zero;
                row_2.s0 = byte_zero;
                row_3.s0 = byte_zero;
            }

            if (inputCol + 2 >= inputWidth) {
                row_1.s2 = byte_zero;
                row_2.s2 = byte_zero;
                row_3.s2 = byte_zero;
            }

            int outputCol = outputColBase + 0 * 3 + 0;
            int outputIndex = outputBase + (outputCol >> 2) * interval + outputCol % 4;
            output[outputIndex] = row_1.s0;

            outputCol = outputColBase + 0 * 3 + 1;
            outputIndex = outputBase + (outputCol >> 2) * interval + outputCol % 4;
            output[outputIndex] = row_1.s1;

            outputCol = outputColBase + 0 * 3 + 2;
            outputIndex = outputBase + (outputCol >> 2) * interval + outputCol % 4;
            output[outputIndex] = row_1.s2;

            outputCol = outputColBase + 1 * 3 + 0;
            outputIndex = outputBase + (outputCol >> 2) * interval + outputCol % 4;
            output[outputIndex] = row_2.s0;

            outputCol = outputColBase + 1 * 3 + 1;
            outputIndex = outputBase + (outputCol >> 2) * interval + outputCol % 4;
            output[outputIndex] = row_2.s1;

            outputCol = outputColBase + 1 * 3 + 2;
            outputIndex = outputBase + (outputCol >> 2) * interval + outputCol % 4;
            output[outputIndex] = row_2.s2;

            outputCol = outputColBase + 2 * 3 + 0;
            outputIndex = outputBase + (outputCol >> 2) * interval + outputCol % 4;
            output[outputIndex] = row_3.s0;

            outputCol = outputColBase + 2 * 3 + 1;
            outputIndex = outputBase + (outputCol >> 2) * interval + outputCol % 4;
            output[outputIndex] = row_3.s1;

            outputCol = outputColBase + 2 * 3 + 2;
            outputIndex = outputBase + (outputCol >> 2) * interval + outputCol % 4;
            output[outputIndex] = row_3.s2;
        }
})

ADD_SINGLE_KERNEL(convert_input_4_thread_8_withpad_3x3_valhall_INT8, (
        __global const unsigned char *input,
        __global unsigned char *output,
        int padT,
        int padL,
        unsigned int strideW,
        unsigned int strideH,
        unsigned int inputHeight,
        unsigned int inputWidth,
        unsigned int inputnum,
        unsigned int alignHeight,
        unsigned int alignWidth,
        unsigned int outputWidth,
        unsigned int outputHeight,
        unsigned int collapseHeight,
        unsigned char byte_zero) {
        //    int outputRowIndex = get_global_id(0);
        //    int inputChannelIndex = get_global_id(1);
        //    int batchIndex = get_global_id(2);

        if (get_global_id(0) < outputWidth * outputHeight) {
            //        int outputColBase = get_global_id(1) * kernel_sizeW * kernel_sizeH;
            int outputBase = get_global_id(2) * alignHeight * alignWidth +
                             (get_global_id(0) / collapseHeight) * collapseHeight * alignWidth +
                             get_global_id(0) % collapseHeight / 4 * 16 +
                             get_global_id(0) % collapseHeight % 4 * 4;

            int inputBase =
                (get_global_id(2) * inputnum + get_global_id(1)) * inputHeight * inputWidth;
            //        int interval = 4 * collapseHeight;

            int inputRow = get_global_id(0) / outputWidth * strideH - padT;
            int inputCol = get_global_id(0) % outputWidth * strideW - padL;

            uchar3 row_1 = byte_zero;
            uchar3 row_2;
            uchar3 row_3 = byte_zero;

            if (inputRow >= 0) {
                row_1 = vload3(0, input + inputBase + inputRow * inputWidth + inputCol);
            }
            row_2 = vload3(0, input + inputBase + (inputRow + 1) * inputWidth + inputCol);
            if (inputRow + 2 < inputHeight) {
                row_3 = vload3(0, input + inputBase + (inputRow + 2) * inputWidth + inputCol);
            }

            if (inputCol < 0) {
                row_1.s0 = byte_zero;
                row_2.s0 = byte_zero;
                row_3.s0 = byte_zero;
            }

            if (inputCol + 2 >= inputWidth) {
                row_1.s2 = byte_zero;
                row_2.s2 = byte_zero;
                row_3.s2 = byte_zero;
            }

            int outputCol = get_global_id(1) * 3 * 3;
            int outputIndex = outputBase + (outputCol >> 2) * 4 * collapseHeight + outputCol % 4;
            output[outputIndex] = row_1.s0;

            outputCol = outputCol + 1;
            outputIndex = outputBase + (outputCol >> 2) * 4 * collapseHeight + outputCol % 4;
            output[outputIndex] = row_1.s1;

            outputCol = outputCol + 1;
            outputIndex = outputBase + (outputCol >> 2) * 4 * collapseHeight + outputCol % 4;
            output[outputIndex] = row_1.s2;

            outputCol = outputCol + 1;
            outputIndex = outputBase + (outputCol >> 2) * 4 * collapseHeight + outputCol % 4;
            output[outputIndex] = row_2.s0;

            outputCol = outputCol + 1;
            outputIndex = outputBase + (outputCol >> 2) * 4 * collapseHeight + outputCol % 4;
            output[outputIndex] = row_2.s1;

            outputCol = outputCol + 1;
            outputIndex = outputBase + (outputCol >> 2) * 4 * collapseHeight + outputCol % 4;
            output[outputIndex] = row_2.s2;

            outputCol = outputCol + 1;
            outputIndex = outputBase + (outputCol >> 2) * 4 * collapseHeight + outputCol % 4;
            output[outputIndex] = row_3.s0;

            outputCol = outputCol + 1;
            outputIndex = outputBase + (outputCol >> 2) * 4 * collapseHeight + outputCol % 4;
            output[outputIndex] = row_3.s1;

            outputCol = outputCol + 1;
            outputIndex = outputBase + (outputCol >> 2) * 4 * collapseHeight + outputCol % 4;
            output[outputIndex] = row_3.s2;
        }
})

ADD_SINGLE_KERNEL(convert_input_4_thread_8_withpad_1x1_INT8, (__global const unsigned char *input,
                                                            __global unsigned char *output,
                                                            unsigned int strideW,
                                                            unsigned int strideH,
                                                            unsigned int inputHeight,
                                                            unsigned int inputWidth,
                                                            unsigned int inputnum,
                                                            unsigned int alignHeight,
                                                            unsigned int alignWidth,
                                                            unsigned int outputWidth,
                                                            unsigned int outputHeight,
                                                            unsigned int collapseHeight,
                                                            unsigned char byte_zero) {
        int g0 = get_global_id(0);
        int g1 = get_global_id(1);
        int g2 = get_global_id(2) * 8;

        if (g2 < outputHeight * outputWidth) {
            uchar8 in = vload8(0, input + (g0 * inputnum + g1) * inputHeight * inputWidth + g2);
            int output_base = g0 * alignHeight * alignWidth + g1 / 4 * 4 * collapseHeight + g1 % 4 +
                              g2 / collapseHeight * collapseHeight * alignWidth +
                              g2 % collapseHeight / 4 * 16;

            output[output_base] = in.s0;
            output[output_base + 4] = in.s1;
            output[output_base + 8] = in.s2;
            output[output_base + 12] = in.s3;
            output[output_base + 16] = in.s4;
            output[output_base + 20] = in.s5;
            output[output_base + 24] = in.s6;
            output[output_base + 28] = in.s7;
        }
    }
)

#define PAD(input, temp_out, padWidth, padHeight, inPutWidth, inPutHeight) \
        int outputWidth = inPutWidth + 2 * padWidth; \
        int outputHeight = inPutHeight + 2 * padHeight; \
        if (get_global_id(2) < outputWidth * outputHeight) { \
            int rowIndex = get_global_id(2) / outputWidth; \
            int colIndex = get_global_id(2) % outputWidth; \
            int channelIndex = get_global_id(0) * get_global_size(1) + get_global_id(1); \
            int outputIndex = channelIndex * outputWidth * outputHeight + get_global_id(2); \
            if (rowIndex < padHeight || rowIndex >= inPutHeight + padHeight || \
                colIndex < padWidth || colIndex >= inPutWidth + padWidth) { \
                temp_out[outputIndex] = (DATA_T)0.0f; \
            } else { \
                int inputIndex = channelIndex * inPutHeight * inPutWidth + \
                                 (rowIndex - padHeight) * inPutWidth + colIndex - padWidth; \
                temp_out[outputIndex] = input[inputIndex]; \
            } \
        }

#define PAD4(input, temp_out, padLeft, padRight, padBottom, padTop, inPutWidth, inPutHeight) \
        int outputWidth = inPutWidth + padLeft + padRight; \
        int outputHeight = inPutHeight + padBottom + padTop; \
        if (get_global_id(2) < outputWidth * outputHeight) { \
            int rowIndex = get_global_id(2) / outputWidth; \
            int colIndex = get_global_id(2) % outputWidth; \
            int channelIndex = get_global_id(0) * get_global_size(1) + get_global_id(1); \
            int outputIndex = channelIndex * outputWidth * outputHeight + get_global_id(2); \
            if (rowIndex < padTop || rowIndex >= inPutHeight + padTop || colIndex < padLeft || \
                colIndex >= inPutWidth + padLeft) { \
                temp_out[outputIndex] = (DATA_T)0.0f; \
            } else { \
                int inputIndex = channelIndex * inPutHeight * inPutWidth + \
                                 (rowIndex - padTop) * inPutWidth + colIndex - padLeft; \
                temp_out[outputIndex] = input[inputIndex]; \
            } \
        }

#define CONVERT(input, output, kernel_sizeW, kernel_sizeH, strideW, strideH, inputHeight, inputWidth, inputnum, \
                alignHeight, alignWidth, outputWidth, outputHeight, collapseHeight) \
        int globalID0 = get_global_id(0); \
        int globalID1 = get_global_id(1); \
        int globalID2 = get_global_id(2); \
        if (globalID2 < inputnum * kernel_sizeH * kernel_sizeW) { \
            int blk_size = kernel_sizeW * kernel_sizeH; \
            int i = (globalID1 / collapseHeight) * collapseHeight * alignWidth + \
                    (globalID2 / 8) * 8 * collapseHeight + (globalID1 % collapseHeight) * 8 + \
                    globalID2 % 8 + globalID0 * alignHeight * alignWidth; \
            int pic_off = globalID2 / blk_size; \
            int index = globalID2 % blk_size; \
            int row_off = index / kernel_sizeW; \
            int col_off = index % kernel_sizeW; \
            int row = globalID1 / outputWidth; \
            int col = globalID1 % outputWidth; \
            output[i] = input[(row_off + row * strideH) * inputWidth + col_off + col * strideW + \
                              pic_off * inputWidth * inputHeight + \
                              globalID0 * inputWidth * inputHeight * inputnum]; \
        }

#define CONVERT_2POINTS_WITHPAD(input, output, kernel_sizeW, kernel_sizeH, padLeft, padRight, padBottom, \
                padTop, strideW, strideH, inputHeight, inputWidth, inputnum, alignHeight, alignWidth, outputWidth, \
                outputHeight, collapseHeight) \
        int batchIndex = get_global_id(0); \
        int inputChannelIndex = get_global_id(1); \
        int outputRowIndex = get_global_id(2); \
        if (outputRowIndex < outputWidth * outputHeight) { \
            int outputColBase = inputChannelIndex * kernel_sizeW * kernel_sizeH; \
            int outputBase = batchIndex * alignHeight * alignWidth + \
                             (outputRowIndex / collapseHeight) * collapseHeight * alignWidth + \
                             outputRowIndex % 2 * collapseHeight / 2 * 8 + \
                             outputRowIndex % collapseHeight / 2 * 8; \
            int interval = 8 * collapseHeight; \
            if (padBottom == 0 && padTop == 0 && kernel_sizeH == 1) { \
                int inputRow = outputRowIndex / outputWidth * strideH; \
                int inputCol = outputRowIndex % outputWidth * strideW - padLeft; \
                for (int col = 0; col < kernel_sizeW; col++) { \
                    int outputCol = outputColBase + col; \
                    int outputIndex = outputBase + (outputCol >> 3) * interval + outputCol % 8; \
                    int inputColIter = inputCol + col; \
                    if (inputColIter >= 0 && inputColIter < inputWidth) { \
                        int inputIndex = \
                            (batchIndex * inputnum + inputChannelIndex) * inputHeight * inputWidth + \
                            inputRow * inputWidth + inputColIter; \
                        output[outputIndex] = input[inputIndex]; \
                    } else { \
                        output[outputIndex] = 0.0f; \
                    } \
                } \
            } else if (padLeft == 0 && padRight == 0 && kernel_sizeW == 1) { \
                int inputRow = outputRowIndex / outputWidth * strideH - padBottom; \
                int inputCol = outputRowIndex % outputWidth * strideW; \
                for (int row = 0; row < kernel_sizeH; row++) { \
                    int outputCol = outputColBase + row * kernel_sizeW; \
                    int outputIndex = outputBase + (outputCol >> 3) * interval + outputCol % 8; \
                    int inputRowIter = inputRow + row; \
                    if (inputRowIter >= 0 && inputRowIter < inputHeight) { \
                        int inputIndex = \
                            (batchIndex * inputnum + inputChannelIndex) * inputHeight * inputWidth + \
                            inputRowIter * inputWidth + inputCol; \
                        output[outputIndex] = input[inputIndex]; \
                    } else { \
                        output[outputIndex] = 0.0f; \
                    } \
                } \
            } else if (padBottom == 0 && padTop == 0 && padLeft == 0 && padRight == 0) { \
                int inputRow = outputRowIndex / outputWidth * strideH; \
                int inputCol = outputRowIndex % outputWidth * strideW; \
                for (int row = 0; row < kernel_sizeH; row++) { \
                    for (int col = 0; col < kernel_sizeW; col++) { \
                        int outputCol = outputColBase + row * kernel_sizeW + col; \
                        int outputIndex = outputBase + (outputCol >> 3) * interval + outputCol % 8; \
                        int inputRowIter = inputRow + row; \
                        int inputColIter = inputCol + col; \
                        int inputIndex = \
                            (batchIndex * inputnum + inputChannelIndex) * inputHeight * inputWidth + \
                            inputRowIter * inputWidth + inputColIter; \
                        output[outputIndex] = input[inputIndex]; \
                    } \
                } \
            } else { \
                int inputRow = outputRowIndex / outputWidth * strideH - padBottom; \
                int inputCol = outputRowIndex % outputWidth * strideW - padLeft; \
                for (int row = 0; row < kernel_sizeH; row++) { \
                    for (int col = 0; col < kernel_sizeW; col++) { \
                        int outputCol = outputColBase + row * kernel_sizeW + col; \
                        int outputIndex = outputBase + (outputCol >> 3) * interval + outputCol % 8; \
                        int inputRowIter = inputRow + row; \
                        int inputColIter = inputCol + col; \
                        if (inputRowIter >= 0 && inputColIter >= 0 && inputRowIter < inputHeight && \
                            inputColIter < inputWidth) { \
                            int inputIndex = (batchIndex * inputnum + inputChannelIndex) * \
                                                 inputHeight * inputWidth + \
                                             inputRow * inputWidth + inputCol + row * inputWidth + \
                                             col; \
                            output[outputIndex] = input[inputIndex]; \
                        } else { \
                            output[outputIndex] = 0.0f; \
                        } \
                    } \
                } \
            } \
        }

#define CONVERT_2POINTS_1X1(input, output, inputHeight, inputWidth, inputnum, alignHeight, alignWidth, collapseHeight) \
        int batchIndex = get_global_id(0); \
        int inputChannelIndex = get_global_id(1); \
        int outputRowIndex = get_global_id(2); \
        if (outputRowIndex < inputHeight * inputWidth) { \
            int inputBase = (batchIndex * inputnum + inputChannelIndex) * inputHeight * inputWidth; \
            int outputBase = batchIndex * alignHeight * alignWidth + \
                             (outputRowIndex / collapseHeight) * collapseHeight * alignWidth + \
                             ((outputRowIndex % 2 * collapseHeight) >> 1) * 8 + \
                             ((outputRowIndex % collapseHeight) >> 1) * 8; \
            int outputIndex = \
                outputBase + (inputChannelIndex >> 3) * collapseHeight * 8 + inputChannelIndex % 8; \
            output[outputIndex] = input[inputBase + (outputRowIndex / inputWidth) * inputWidth + \
                                        (outputRowIndex % inputWidth)]; \
        }

#define CONVERT_INPUT_4_THREAD_8_1X1(input, output, inputHeight, inputWidth, inputnum, alignHeight, alignWidth, \
                                     collapseHeight) \
        int topHWIndex = get_global_id(0) * 8; \
        int inputChannelIndex = get_global_id(1); \
        int batchIndex = get_global_id(2); \
        if (topHWIndex < inputHeight * inputWidth && inputChannelIndex < inputnum) { \
            int inputIndex = \
                (batchIndex * inputnum + inputChannelIndex) * inputHeight * inputWidth + \
                +topHWIndex; \
            DATA_T8 out8 = vload8(0, input + inputIndex); \
            int outputBase = batchIndex * alignHeight * alignWidth + \
                             topHWIndex / collapseHeight * collapseHeight * alignWidth + \
                             inputChannelIndex / 4 * collapseHeight * 4 + \
                             topHWIndex % collapseHeight / 8 * 16 * 2 + inputChannelIndex % 4 * 4; \
            vstore4(out8.lo, 0, output + outputBase); \
            vstore4(out8.hi, 0, output + outputBase + 16); \
        }

#define CONVERT_INPUT_4_THREAD_8_1X1_NHWC(input, output, inputHeight, inputWidth, inputnum, alignHeight, \
                                          alignWidth, collapseHeight) \
        int inputChannelIndex = get_global_id(0) * 8;  \
        int topHWIndex = get_global_id(1) * 4;  \
        int batchIndex = get_global_id(2); \
        if (topHWIndex < inputHeight * inputWidth && inputChannelIndex < inputnum) { \
            int inputIndex = \
                batchIndex * inputHeight * inputWidth * inputnum + \
                + topHWIndex * inputnum + inputChannelIndex; \
            DATA_T8 out8_0 = vload8(0, input + inputIndex); \
            DATA_T8 out8_1 = vload8(0, input + inputIndex + inputnum); \
            DATA_T8 out8_2 = vload8(0, input + inputIndex + 2 * inputnum); \
            DATA_T8 out8_3 = vload8(0, input + inputIndex + 3 * inputnum); \
            DATA_T16 out16_0 = (DATA_T16)(out8_0.s0, out8_1.s0, out8_2.s0, out8_3.s0, \
                                      out8_0.s1, out8_1.s1, out8_2.s1, out8_3.s1, \
                                      out8_0.s2, out8_1.s2, out8_2.s2, out8_3.s2, \
                                      out8_0.s3, out8_1.s3, out8_2.s3, out8_3.s3); \
            DATA_T16 out16_1 = (DATA_T16)(out8_0.s4, out8_1.s4, out8_2.s4, out8_3.s4, \
                                      out8_0.s5, out8_1.s5, out8_2.s5, out8_3.s5, \
                                      out8_0.s6, out8_1.s6, out8_2.s6, out8_3.s6, \
                                      out8_0.s7, out8_1.s7, out8_2.s7, out8_3.s7); \
            int outputBase = batchIndex * alignHeight * alignWidth + \
                             topHWIndex / collapseHeight * collapseHeight * alignWidth + \
                             inputChannelIndex / 4 * collapseHeight * 4 + \
                             topHWIndex % collapseHeight / 4 * 16; \
            vstore16(out16_0, 0, output + outputBase); \
            vstore16(out16_1, 0, output + outputBase + collapseHeight * 4); \
        }

#define CONVERT_INPUT_4_THREAD_8_WITHPAD(input, output, kernel_sizeW, kernel_sizeH, padLeft, padRight, \
                padBottom, padTop, strideW, strideH, inputHeight, inputWidth, inputnum, alignHeight, alignWidth, \
                outputWidth, outputHeight, collapseHeight, HWOffSet) \
        int outputRowIndex = get_global_id(0); \
        int inputChannelIndex = get_global_id(1); \
        int batchIndex = get_global_id(2); \
        if (outputRowIndex + HWOffSet < outputWidth * outputHeight) { \
            outputRowIndex = outputRowIndex + HWOffSet; \
            int outputColBase = inputChannelIndex * kernel_sizeW * kernel_sizeH; \
            int outputBase = \
                batchIndex * alignHeight * alignWidth + \
                ((outputRowIndex - HWOffSet) / collapseHeight) * collapseHeight * alignWidth + \
                outputRowIndex % collapseHeight / 4 * 16 + outputRowIndex % collapseHeight % 4; \
            int inputBase = (batchIndex * inputnum + inputChannelIndex) * inputHeight * inputWidth; \
            int interval = 4 * collapseHeight; \
            if (padBottom == 0 && padTop == 0 && kernel_sizeH == 1) { \
                int inputRow = outputRowIndex / outputWidth * strideH; \
                int inputCol = outputRowIndex % outputWidth * strideW - padLeft; \
                inputBase += inputRow * inputWidth; \
                for (int col = 0; col < kernel_sizeW; col++) { \
                    int outputCol = outputColBase + col; \
                    int outputIndex = outputBase + (outputCol >> 2) * interval + outputCol % 4 * 4; \
                    int inputColIter = inputCol + col; \
                    if (inputColIter >= 0 && inputColIter < inputWidth) { \
                        int inputIndex = inputBase + inputColIter; \
                        output[outputIndex] = input[inputIndex]; \
                    } else { \
                        output[outputIndex] = 0.0f; \
                    } \
                } \
            } else if (padLeft == 0 && padRight == 0 && kernel_sizeW == 1) { \
                int inputRow = outputRowIndex / outputWidth * strideH - padTop; \
                int inputCol = outputRowIndex % outputWidth * strideW; \
                inputBase += inputCol; \
                for (int row = 0; row < kernel_sizeH; row++) { \
                    int outputCol = outputColBase + row; \
                    int outputIndex = outputBase + (outputCol >> 2) * interval + outputCol % 4 * 4; \
                    int inputRowIter = inputRow + row; \
                    if (inputRowIter >= 0 && inputRowIter < inputHeight) { \
                        int inputIndex = inputBase + inputRowIter * inputWidth; \
                        output[outputIndex] = input[inputIndex]; \
                    } else { \
                        output[outputIndex] = 0.0f; \
                    } \
                } \
            } else if (padBottom == 0 && padTop == 0 && padLeft == 0 && padRight == 0) { \
                int inputRow = outputRowIndex / outputWidth * strideH; \
                int inputCol = outputRowIndex % outputWidth * strideW; \
                for (int row = 0; row < kernel_sizeH; row++) { \
                    for (int col = 0; col < kernel_sizeW; col++) { \
                        int outputCol = outputColBase + row * kernel_sizeW + col; \
                        int outputIndex = \
                            outputBase + (outputCol >> 2) * interval + outputCol % 4 * 4; \
                        int inputRowIter = inputRow + row; \
                        int inputColIter = inputCol + col; \
                        int inputIndex = inputBase + inputRowIter * inputWidth + inputColIter; \
                        output[outputIndex] = input[inputIndex]; \
                    } \
                } \
            } else { \
                int inputRow = outputRowIndex / outputWidth * strideH - padTop; \
                int inputCol = outputRowIndex % outputWidth * strideW - padLeft; \
                for (int row = 0; row < kernel_sizeH; row++) { \
                    for (int col = 0; col < kernel_sizeW; col++) { \
                        int outputCol = outputColBase + row * kernel_sizeW + col; \
                        int outputIndex = \
                            outputBase + (outputCol >> 2) * interval + outputCol % 4 * 4; \
                        int inputRowIter = inputRow + row; \
                        int inputColIter = inputCol + col; \
                        if (inputRowIter >= 0 && inputColIter >= 0 && inputRowIter < inputHeight && \
                            inputColIter < inputWidth) { \
                            int inputIndex = inputBase + inputRowIter * inputWidth + inputColIter; \
                            output[outputIndex] = input[inputIndex]; \
                        } else { \
                            output[outputIndex] = 0.0f; \
                        } \
                    } \
                } \
            } \
        }

#define CONVERT_INPUT_4_THREAD_8_WITHPAD_5X5(input, output, inputHeight, inputWidth, inputnum, alignHeight, \
                                             alignWidth, collapseHeight, HWOffSet) \
        int HWIndex = get_global_id(0) * 16; \
        int CinKKIndex = get_global_id(1); \
        int batchIndex = get_global_id(2); \
        if (HWIndex + HWOffSet < inputHeight * inputWidth && CinKKIndex < inputnum * 25) { \
            int inputChannelIndex = CinKKIndex / 25; \
            int KIndex = CinKKIndex % 25; \
            int Row = (HWIndex + HWOffSet) / inputWidth + KIndex / 5 - 2; \
            int Col = (HWIndex + HWOffSet) % inputWidth + KIndex % 5 - 2; \
            int interval = 4 * collapseHeight; \
            int inputBase = (batchIndex * inputnum + inputChannelIndex) * inputHeight * \
                            inputWidth; \
            int outRow = HWIndex / collapseHeight; \
            int outCol = HWIndex % collapseHeight / 4 * 16 + CinKKIndex % 4 * 4 + \
                         CinKKIndex / 4 * interval;  \
            int outputBase = batchIndex * alignHeight * alignWidth + \
                             outRow * alignWidth * collapseHeight + outCol; \
            DATA_T16 input_0 = (DATA_T16)0.0f; \
            if (Row >= 0 && Row < inputHeight) { \
                input_0 = vload16(0, input + inputBase + Row * inputWidth + Col); \
            } \
            if (Col < 0) { \
                input_0.s0 = 0.0f; \
            } \
            if (Col + 1 < 0) { \
                input_0.s1 = 0.0f; \
            } \
            if (Col + 14 >= inputWidth) { \
                input_0.se = 0.0f; \
            } \
            if (Col + 15 >= inputWidth) { \
                input_0.sf = 0.0f; \
            } \
            vstore4(input_0.s0123, 0, output + outputBase); \
            vstore4(input_0.s4567, 0, output + outputBase + 16); \
            vstore4(input_0.s89ab, 0, output + outputBase + 32); \
            vstore4(input_0.scdef, 0, output + outputBase + 48); \
        }

#define CONVERT_INPUT_2_THREAD_8_1X1(input, output, inputHeight, inputWidth, inputnum, alignHeight, alignWidth, \
                                     collapseHeight) \
        int topHWIndex = get_global_id(0) * 4; \
        int inputChannelIndex = get_global_id(1); \
        int batchIndex = get_global_id(2); \
        if (topHWIndex < inputHeight * inputWidth && inputChannelIndex < inputnum) { \
            int inputIndex = \
                (batchIndex * inputnum + inputChannelIndex) * inputHeight * inputWidth + topHWIndex; \
            DATA_T4 out8 = vload4(0, input + inputIndex); \
            int outputBase = batchIndex * alignHeight * alignWidth + \
                             topHWIndex / 24 * 24 * alignWidth + inputChannelIndex / 8 * 192 + \
                             topHWIndex % 24 / 4 * 32 + inputChannelIndex % 8 * 2; \
            vstore2(out8.lo, 0, output + outputBase); \
            vstore2(out8.hi, 0, output + outputBase + 16); \
        }

#define CONVERT_INPUT_2_THREAD_8_WITHPAD(input, output, kernel_sizeW, kernel_sizeH, padLeft, padRight, padBottom, \
                            padTop, strideW, strideH, inputHeight, inputWidth, inputnum, alignHeight, alignWidth, \
                            outputWidth, outputHeight, collapseHeight) \
        int outputRowIndex = get_global_id(0); \
        int inputChannelIndex = get_global_id(1); \
        int batchIndex = get_global_id(2); \
        if (outputRowIndex < outputWidth * outputHeight) { \
            int outputColBase = inputChannelIndex * kernel_sizeW * kernel_sizeH; \
            int outputBase = batchIndex * alignHeight * alignWidth + \
                             (outputRowIndex / 24) * 24 * alignWidth + \
                             outputRowIndex % 24 / 2 * 16 + outputRowIndex % 24 % 2; \
            int inputBase = (batchIndex * inputnum + inputChannelIndex) * inputHeight * inputWidth; \
            int interval = 192; \
            if (padBottom == 0 && padTop == 0 && kernel_sizeH == 1) { \
                int inputRow = outputRowIndex / outputWidth * strideH; \
                int inputCol = outputRowIndex % outputWidth * strideW - padLeft; \
                inputBase += inputRow * inputWidth; \
                for (int col = 0; col < kernel_sizeW; col++) { \
                    int outputCol = outputColBase + col; \
                    int outputIndex = outputBase + (outputCol >> 3) * interval + outputCol % 8 * 2; \
                    int inputColIter = inputCol + col; \
                    if (inputColIter >= 0 && inputColIter < inputWidth) { \
                        int inputIndex = inputBase + inputColIter; \
                        output[outputIndex] = input[inputIndex]; \
                    } else { \
                        output[outputIndex] = 0.0f; \
                    } \
                } \
            } else if (padLeft == 0 && padRight == 0 && kernel_sizeW == 1) { \
                int inputRow = outputRowIndex / outputWidth * strideH - padTop; \
                int inputCol = outputRowIndex % outputWidth * strideW; \
                inputBase += inputCol; \
                for (int row = 0; row < kernel_sizeH; row++) { \
                    int outputCol = outputColBase + row; \
                    int outputIndex = outputBase + (outputCol >> 3) * interval + outputCol % 8 * 2; \
                    int inputRowIter = inputRow + row; \
                    if (inputRowIter >= 0 && inputRowIter < inputHeight) { \
                        int inputIndex = inputBase + inputRowIter * inputWidth; \
                        output[outputIndex] = input[inputIndex]; \
                    } else { \
                        output[outputIndex] = 0.0f; \
                    } \
                } \
            } else if (padBottom == 0 && padTop == 0 && padLeft == 0 && padRight == 0) { \
                int inputRow = outputRowIndex / outputWidth * strideH; \
                int inputCol = outputRowIndex % outputWidth * strideW; \
                for (int row = 0; row < kernel_sizeH; row++) { \
                    for (int col = 0; col < kernel_sizeW; col++) { \
                        int outputCol = outputColBase + row * kernel_sizeW + col; \
                        int outputIndex = \
                            outputBase + (outputCol >> 3) * interval + outputCol % 8 * 2; \
                        int inputRowIter = inputRow + row; \
                        int inputColIter = inputCol + col; \
                        int inputIndex = inputBase + inputRowIter * inputWidth + inputColIter; \
                        output[outputIndex] = input[inputIndex]; \
                    } \
                } \
            } else { \
                int inputRow = outputRowIndex / outputWidth * strideH - padTop; \
                int inputCol = outputRowIndex % outputWidth * strideW - padLeft; \
                for (int row = 0; row < kernel_sizeH; row++) { \
                    for (int col = 0; col < kernel_sizeW; col++) { \
                        int outputCol = outputColBase + row * kernel_sizeW + col; \
                        int outputIndex = \
                            outputBase + (outputCol >> 3) * interval + outputCol % 8 * 2; \
                        int inputRowIter = inputRow + row; \
                        int inputColIter = inputCol + col; \
                        if (inputRowIter >= 0 && inputColIter >= 0 && inputRowIter < inputHeight && \
                            inputColIter < inputWidth) { \
                            int inputIndex = inputBase + inputRowIter * inputWidth + inputColIter; \
                            output[outputIndex] = input[inputIndex]; \
                        } else { \
                            output[outputIndex] = 0.0f; \
                        } \
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

ADD_SINGLE_KERNEL(pad_FP16, (__global const DATA_T *input,
                           __global DATA_T *temp_out,
                           unsigned int padWidth,
                           unsigned int padHeight,
                           unsigned int inPutWidth,
                           unsigned int inPutHeight) {
    PAD(input, temp_out, padWidth, padHeight, inPutWidth, inPutHeight)
})

ADD_SINGLE_KERNEL(pad4_FP16, (__global const DATA_T *input,
                            __global DATA_T *temp_out,
                            unsigned int padLeft,
                            unsigned int padRight,
                            unsigned int padBottom,
                            unsigned int padTop,
                            unsigned int inPutWidth,
                            unsigned int inPutHeight) {
    PAD4(input, temp_out, padLeft, padRight, padBottom, padTop, inPutWidth, inPutHeight)
})

ADD_SINGLE_KERNEL(convert_optimized_FP16, (__global const DATA_T *input,
                                         __global DATA_T *output,
                                         unsigned int kernelWidth,
                                         unsigned int kernelHeight,
                                         unsigned int strideWidth,
                                         unsigned int strideHeight,
                                         unsigned int inputHeight,
                                         unsigned int inputWidth,
                                         unsigned int inputChannel,
                                         unsigned int alignHeight,
                                         unsigned int alignWidth,
                                         unsigned int outputWidth,
                                         unsigned int outputHeight) {
    CONVERT_OPTIMIZED(input, output, kernelWidth, kernelHeight, strideWidth, strideHeight, inputHeight, inputWidth, \
                        inputChannel, alignHeight, alignWidth, outputWidth, outputHeight)
})

ADD_SINGLE_KERNEL(convert_FP16, (__global const DATA_T *input,
                               __global DATA_T *output,
                               unsigned int kernel_sizeW,
                               unsigned int kernel_sizeH,
                               unsigned int strideW,
                               unsigned int strideH,
                               unsigned int inputHeight,
                               unsigned int inputWidth,
                               unsigned int inputnum,
                               unsigned int alignHeight,
                               unsigned int alignWidth,
                               unsigned int outputWidth,
                               unsigned int outputHeight,
                               unsigned int collapseHeight) {
    CONVERT(input, output, kernel_sizeW, kernel_sizeH, strideW, strideH, inputHeight, inputWidth, inputnum, \
            alignHeight, alignWidth, outputWidth, outputHeight, collapseHeight)
})

ADD_SINGLE_KERNEL(convertBlocked2_FP16, (__global const DATA_T *input,
                                       __global DATA_T *output,
                                       unsigned int kernel_sizeW,
                                       unsigned int kernel_sizeH,
                                       unsigned int strideW,
                                       unsigned int strideH,
                                       unsigned int inputHeight,
                                       unsigned int inputWidth,
                                       unsigned int inputnum,
                                       unsigned int alignHeight,
                                       unsigned int alignWidth,
                                       unsigned int group,
                                       unsigned int outputWidth,
                                       unsigned int groupNum,
                                       unsigned int outputHeight) {
    CONVERTBLOCKED2(input, output, kernel_sizeW, kernel_sizeH, strideW, strideH, inputHeight, inputWidth, \
                    inputnum, alignHeight, alignWidth, group, outputWidth, groupNum, outputHeight)
})

ADD_SINGLE_KERNEL(convert_2points_withpad_FP16, (__global const DATA_T *input,
                                               __global DATA_T *output,
                                               unsigned int kernel_sizeW,
                                               unsigned int kernel_sizeH,
                                               int padLeft,
                                               int padRight,
                                               int padBottom,
                                               int padTop,
                                               unsigned int strideW,
                                               unsigned int strideH,
                                               unsigned int inputHeight,
                                               unsigned int inputWidth,
                                               unsigned int inputnum,
                                               unsigned int alignHeight,
                                               unsigned int alignWidth,
                                               unsigned int outputWidth,
                                               unsigned int outputHeight,
                                               unsigned int collapseHeight) {
    CONVERT_2POINTS_WITHPAD(input, output, kernel_sizeW, kernel_sizeH, padLeft, padRight, padBottom, padTop, \
                            strideW, strideH, inputHeight, inputWidth, inputnum, alignHeight, alignWidth, \
                            outputWidth, outputHeight, collapseHeight)
})

ADD_SINGLE_KERNEL(convert_2points_1x1_FP16, (__global const DATA_T *input,
                                           __global DATA_T *output,
                                           unsigned int inputHeight,
                                           unsigned int inputWidth,
                                           unsigned int inputnum,
                                           unsigned int alignHeight,
                                           unsigned int alignWidth,
                                           unsigned int collapseHeight) {
    CONVERT_2POINTS_1X1(input, output, inputHeight, inputWidth, inputnum, alignHeight, alignWidth, collapseHeight)
})

ADD_SINGLE_KERNEL(convert_input_4_thread_8_1x1_FP16, (__global const DATA_T *input,
                                                    __global DATA_T *output,
                                                    unsigned int inputHeight,
                                                    unsigned int inputWidth,
                                                    unsigned int inputnum,
                                                    unsigned int alignHeight,
                                                    unsigned int alignWidth,
                                                    unsigned int collapseHeight) {
    CONVERT_INPUT_4_THREAD_8_1X1(input, output, inputHeight, inputWidth, inputnum, alignHeight, alignWidth, \
                                collapseHeight)
})

ADD_SINGLE_KERNEL(convert_input_4_thread_8_1x1_nhwc_FP16, (__global const DATA_T *input,
                                                    __global DATA_T *output,
                                                    unsigned int inputHeight,
                                                    unsigned int inputWidth,
                                                    unsigned int inputnum,
                                                    unsigned int alignHeight,
                                                    unsigned int alignWidth,
                                                    unsigned int collapseHeight) {
    CONVERT_INPUT_4_THREAD_8_1X1_NHWC(input, output, inputHeight, inputWidth, inputnum, alignHeight, alignWidth, \
                                        collapseHeight)
})

ADD_SINGLE_KERNEL(convert_input_4_thread_8_withpad_FP16, (__global const DATA_T *input,
                                                        __global DATA_T *output,
                                                        unsigned int kernel_sizeW,
                                                        unsigned int kernel_sizeH,
                                                        int padLeft,
                                                        int padRight,
                                                        int padBottom,
                                                        int padTop,
                                                        unsigned int strideW,
                                                        unsigned int strideH,
                                                        unsigned int inputHeight,
                                                        unsigned int inputWidth,
                                                        unsigned int inputnum,
                                                        unsigned int alignHeight,
                                                        unsigned int alignWidth,
                                                        unsigned int outputWidth,
                                                        unsigned int outputHeight,
                                                        unsigned int collapseHeight,
                                                        unsigned int HWOffSet) {
    CONVERT_INPUT_4_THREAD_8_WITHPAD(input, output, kernel_sizeW, kernel_sizeH, padLeft, padRight, padBottom, \
                padTop, strideW, strideH, inputHeight, inputWidth, inputnum, alignHeight, alignWidth, outputWidth, \
                outputHeight, collapseHeight, HWOffSet)
})

ADD_SINGLE_KERNEL(convert_input_4_thread_8_withpad_5x5_FP16, (__global const DATA_T *input,
                                                            __global DATA_T *output,
                                                            unsigned int inputHeight,
                                                            unsigned int inputWidth,
                                                            unsigned int inputnum,
                                                            unsigned int alignHeight,
                                                            unsigned int alignWidth,
                                                            unsigned int collapseHeight,
                                                            unsigned int HWOffSet) {
    CONVERT_INPUT_4_THREAD_8_WITHPAD_5X5(input, output, inputHeight, inputWidth, inputnum, alignHeight, \
                                         alignWidth, collapseHeight, HWOffSet)
})

ADD_SINGLE_KERNEL(convert_input_2_thread_8_1x1_FP16, (__global const DATA_T *input,
                                                    __global DATA_T *output,
                                                    unsigned int inputHeight,
                                                    unsigned int inputWidth,
                                                    unsigned int inputnum,
                                                    unsigned int alignHeight,
                                                    unsigned int alignWidth,
                                                    unsigned int collapseHeight) {
    CONVERT_INPUT_2_THREAD_8_1X1(input, output, inputHeight, inputWidth, inputnum, alignHeight, alignWidth, \
                                 collapseHeight)
})

ADD_SINGLE_KERNEL(convert_input_2_thread_8_withpad_FP16, (__global const DATA_T *input,
                                                        __global DATA_T *output,
                                                        unsigned int kernel_sizeW,
                                                        unsigned int kernel_sizeH,
                                                        int padLeft,
                                                        int padRight,
                                                        int padBottom,
                                                        int padTop,
                                                        unsigned int strideW,
                                                        unsigned int strideH,
                                                        unsigned int inputHeight,
                                                        unsigned int inputWidth,
                                                        unsigned int inputnum,
                                                        unsigned int alignHeight,
                                                        unsigned int alignWidth,
                                                        unsigned int outputWidth,
                                                        unsigned int outputHeight,
                                                        unsigned int collapseHeight) {
    CONVERT_INPUT_2_THREAD_8_WITHPAD(input, output, kernel_sizeW, kernel_sizeH, padLeft, padRight, padBottom, padTop, \
                            strideW, strideH, inputHeight, inputWidth, inputnum, alignHeight, alignWidth, outputWidth, \
                            outputHeight, collapseHeight)
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

ADD_SINGLE_KERNEL(pad_FP32, (__global const DATA_T *input,
                           __global DATA_T *temp_out,
                           unsigned int padWidth,
                           unsigned int padHeight,
                           unsigned int inPutWidth,
                           unsigned int inPutHeight) {
    PAD(input, temp_out, padWidth, padHeight, inPutWidth, inPutHeight)
})

ADD_SINGLE_KERNEL(pad4_FP32, (__global const DATA_T *input,
                            __global DATA_T *temp_out,
                            unsigned int padLeft,
                            unsigned int padRight,
                            unsigned int padBottom,
                            unsigned int padTop,
                            unsigned int inPutWidth,
                            unsigned int inPutHeight) {
    PAD4(input, temp_out, padLeft, padRight, padBottom, padTop, inPutWidth, inPutHeight)
})

ADD_SINGLE_KERNEL(convert_optimized_FP32, (__global const DATA_T *input,
                                         __global DATA_T *output,
                                         unsigned int kernelWidth,
                                         unsigned int kernelHeight,
                                         unsigned int strideWidth,
                                         unsigned int strideHeight,
                                         unsigned int inputHeight,
                                         unsigned int inputWidth,
                                         unsigned int inputChannel,
                                         unsigned int alignHeight,
                                         unsigned int alignWidth,
                                         unsigned int outputWidth,
                                         unsigned int outputHeight) {
    CONVERT_OPTIMIZED(input, output, kernelWidth, kernelHeight, strideWidth, strideHeight, inputHeight, inputWidth, \
                        inputChannel, alignHeight, alignWidth, outputWidth, outputHeight)
})

ADD_SINGLE_KERNEL(convert_FP32, (__global const DATA_T *input,
                               __global DATA_T *output,
                               unsigned int kernel_sizeW,
                               unsigned int kernel_sizeH,
                               unsigned int strideW,
                               unsigned int strideH,
                               unsigned int inputHeight,
                               unsigned int inputWidth,
                               unsigned int inputnum,
                               unsigned int alignHeight,
                               unsigned int alignWidth,
                               unsigned int outputWidth,
                               unsigned int outputHeight,
                               unsigned int collapseHeight) {
    CONVERT(input, output, kernel_sizeW, kernel_sizeH, strideW, strideH, inputHeight, inputWidth, inputnum, \
            alignHeight, alignWidth, outputWidth, outputHeight, collapseHeight)
})

ADD_SINGLE_KERNEL(convertBlocked2_FP32, (__global const DATA_T *input,
                                       __global DATA_T *output,
                                       unsigned int kernel_sizeW,
                                       unsigned int kernel_sizeH,
                                       unsigned int strideW,
                                       unsigned int strideH,
                                       unsigned int inputHeight,
                                       unsigned int inputWidth,
                                       unsigned int inputnum,
                                       unsigned int alignHeight,
                                       unsigned int alignWidth,
                                       unsigned int group,
                                       unsigned int outputWidth,
                                       unsigned int groupNum,
                                       unsigned int outputHeight) {
    CONVERTBLOCKED2(input, output, kernel_sizeW, kernel_sizeH, strideW, strideH, inputHeight, inputWidth, \
                    inputnum, alignHeight, alignWidth, group, outputWidth, groupNum, outputHeight)
})

ADD_SINGLE_KERNEL(convert_2points_withpad_FP32, (__global const DATA_T *input,
                                               __global DATA_T *output,
                                               unsigned int kernel_sizeW,
                                               unsigned int kernel_sizeH,
                                               int padLeft,
                                               int padRight,
                                               int padBottom,
                                               int padTop,
                                               unsigned int strideW,
                                               unsigned int strideH,
                                               unsigned int inputHeight,
                                               unsigned int inputWidth,
                                               unsigned int inputnum,
                                               unsigned int alignHeight,
                                               unsigned int alignWidth,
                                               unsigned int outputWidth,
                                               unsigned int outputHeight,
                                               unsigned int collapseHeight) {
    CONVERT_2POINTS_WITHPAD(input, output, kernel_sizeW, kernel_sizeH, padLeft, padRight, padBottom, padTop, \
                            strideW, strideH, inputHeight, inputWidth, inputnum, alignHeight, alignWidth, \
                            outputWidth, outputHeight, collapseHeight)
})

ADD_SINGLE_KERNEL(convert_2points_1x1_FP32, (__global const DATA_T *input,
                                           __global DATA_T *output,
                                           unsigned int inputHeight,
                                           unsigned int inputWidth,
                                           unsigned int inputnum,
                                           unsigned int alignHeight,
                                           unsigned int alignWidth,
                                           unsigned int collapseHeight) {
    CONVERT_2POINTS_1X1(input, output, inputHeight, inputWidth, inputnum, alignHeight, alignWidth, collapseHeight)
})

ADD_SINGLE_KERNEL(convert_input_4_thread_8_1x1_FP32, (__global const DATA_T *input,
                                                    __global DATA_T *output,
                                                    unsigned int inputHeight,
                                                    unsigned int inputWidth,
                                                    unsigned int inputnum,
                                                    unsigned int alignHeight,
                                                    unsigned int alignWidth,
                                                    unsigned int collapseHeight) {
    CONVERT_INPUT_4_THREAD_8_1X1(input, output, inputHeight, inputWidth, inputnum, alignHeight, alignWidth, \
                                collapseHeight)
})

ADD_SINGLE_KERNEL(convert_input_4_thread_8_1x1_nhwc_FP32, (__global const DATA_T *input,
                                                    __global DATA_T *output,
                                                    unsigned int inputHeight,
                                                    unsigned int inputWidth,
                                                    unsigned int inputnum,
                                                    unsigned int alignHeight,
                                                    unsigned int alignWidth,
                                                    unsigned int collapseHeight) {
    CONVERT_INPUT_4_THREAD_8_1X1_NHWC(input, output, inputHeight, inputWidth, inputnum, alignHeight, alignWidth, \
                                        collapseHeight)
})

ADD_SINGLE_KERNEL(convert_input_4_thread_8_withpad_FP32, (__global const DATA_T *input,
                                                        __global DATA_T *output,
                                                        unsigned int kernel_sizeW,
                                                        unsigned int kernel_sizeH,
                                                        int padLeft,
                                                        int padRight,
                                                        int padBottom,
                                                        int padTop,
                                                        unsigned int strideW,
                                                        unsigned int strideH,
                                                        unsigned int inputHeight,
                                                        unsigned int inputWidth,
                                                        unsigned int inputnum,
                                                        unsigned int alignHeight,
                                                        unsigned int alignWidth,
                                                        unsigned int outputWidth,
                                                        unsigned int outputHeight,
                                                        unsigned int collapseHeight,
                                                        unsigned int HWOffSet) {
    CONVERT_INPUT_4_THREAD_8_WITHPAD(input, output, kernel_sizeW, kernel_sizeH, padLeft, padRight, padBottom, \
                padTop, strideW, strideH, inputHeight, inputWidth, inputnum, alignHeight, alignWidth, outputWidth, \
                outputHeight, collapseHeight, HWOffSet)
})

ADD_SINGLE_KERNEL(convert_input_4_thread_8_withpad_5x5_FP32, (__global const DATA_T *input,
                                                            __global DATA_T *output,
                                                            unsigned int inputHeight,
                                                            unsigned int inputWidth,
                                                            unsigned int inputnum,
                                                            unsigned int alignHeight,
                                                            unsigned int alignWidth,
                                                            unsigned int collapseHeight,
                                                            unsigned int HWOffSet) {
    CONVERT_INPUT_4_THREAD_8_WITHPAD_5X5(input, output, inputHeight, inputWidth, inputnum, alignHeight, \
                                         alignWidth, collapseHeight, HWOffSet)
})

ADD_SINGLE_KERNEL(convert_input_2_thread_8_1x1_FP32, (__global const DATA_T *input,
                                                    __global DATA_T *output,
                                                    unsigned int inputHeight,
                                                    unsigned int inputWidth,
                                                    unsigned int inputnum,
                                                    unsigned int alignHeight,
                                                    unsigned int alignWidth,
                                                    unsigned int collapseHeight) {
    CONVERT_INPUT_2_THREAD_8_1X1(input, output, inputHeight, inputWidth, inputnum, alignHeight, alignWidth, \
                                 collapseHeight)
})

ADD_SINGLE_KERNEL(convert_input_2_thread_8_withpad_FP32, (__global const DATA_T *input,
                                                        __global DATA_T *output,
                                                        unsigned int kernel_sizeW,
                                                        unsigned int kernel_sizeH,
                                                        int padLeft,
                                                        int padRight,
                                                        int padBottom,
                                                        int padTop,
                                                        unsigned int strideW,
                                                        unsigned int strideH,
                                                        unsigned int inputHeight,
                                                        unsigned int inputWidth,
                                                        unsigned int inputnum,
                                                        unsigned int alignHeight,
                                                        unsigned int alignWidth,
                                                        unsigned int outputWidth,
                                                        unsigned int outputHeight,
                                                        unsigned int collapseHeight) {
    CONVERT_INPUT_2_THREAD_8_WITHPAD(input, output, kernel_sizeW, kernel_sizeH, padLeft, padRight, padBottom, padTop, \
                            strideW, strideH, inputHeight, inputWidth, inputnum, alignHeight, alignWidth, outputWidth, \
                            outputHeight, collapseHeight)
})

#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // float

}  // namespace gpu
}  // namespace ud
}  // namespace enn
