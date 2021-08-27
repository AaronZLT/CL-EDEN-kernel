#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"
namespace enn{
namespace ud {
namespace gpu {

#define FAST3X3_1_2X8(                                                                                                 \
    input, output, padHeight, padWidth, inputHeight, inputWidth, chan, height, width, coalescingRowSize)               \
    int picOff = get_global_id(0);                                                                                     \
    int channel = get_global_id(1);                                                                                    \
    int hwIndex = get_global_id(2);                                                                                    \
    if (hwIndex < height * width) {                                                                                    \
        int gID1 = hwIndex / width;                                                                                    \
        int gID2 = hwIndex % width;                                                                                    \
        int start_h_indata = gID1 * 2 - padHeight;                                                                     \
        int start_w_indata = gID2 * 2 - padWidth;                                                                      \
        int row = gID1 * width + gID2;                                                                                 \
        int alignedHW = (height * width + coalescingRowSize - 1) / coalescingRowSize * coalescingRowSize;              \
        int out_index = picOff * chan * alignedHW * 16 + row / coalescingRowSize * coalescingRowSize * chan * 16 +     \
                        row % coalescingRowSize * 4 + channel * coalescingRowSize * 4;                                 \
        int inputBase = picOff * chan * inputHeight * inputWidth + channel * inputHeight * inputWidth;                 \
        DATA_T16 res;                                                                                                  \
        if (start_h_indata >= 0 && start_h_indata + 4 <= inputHeight && start_w_indata >= 0 &&                         \
            start_w_indata + 4 <= inputWidth) {                                                                        \
            int rowBase = inputBase + inputWidth * start_h_indata + start_w_indata;                                    \
            DATA_T4 row0 = vload4(0, input + rowBase);                                                                 \
            rowBase = rowBase + inputWidth;                                                                            \
            DATA_T4 row1 = vload4(0, input + rowBase);                                                                 \
            rowBase = rowBase + inputWidth;                                                                            \
            DATA_T4 row2 = vload4(0, input + rowBase);                                                                 \
            rowBase = rowBase + inputWidth;                                                                            \
            DATA_T4 row3 = vload4(0, input + rowBase);                                                                 \
            res.s0 = row0.s0 - row0.s2 - row2.s0 + row2.s2;                                                            \
            res.s1 = row0.s1 + row0.s2 - row2.s1 - row2.s2;                                                            \
            res.s2 = row0.s2 - row0.s1 + row2.s1 - row2.s2;                                                            \
            res.s3 = row0.s1 - row0.s3 - row2.s1 + row2.s3;                                                            \
            res.s4 = row1.s0 - row1.s2 + row2.s0 - row2.s2;                                                            \
            res.s5 = row1.s1 + row1.s2 + row2.s1 + row2.s2;                                                            \
            res.s6 = row1.s2 - row1.s1 - row2.s1 + row2.s2;                                                            \
            res.s7 = row1.s1 - row1.s3 + row2.s1 - row2.s3;                                                            \
            res.s8 = row1.s2 - row1.s0 + row2.s0 - row2.s2;                                                            \
            res.s9 = row2.s1 - row1.s1 - row1.s2 + row2.s2;                                                            \
            res.sa = row1.s1 - row1.s2 - row2.s1 + row2.s2;                                                            \
            res.sb = row1.s3 - row1.s1 + row2.s1 - row2.s3;                                                            \
            res.sc = row1.s0 - row1.s2 - row3.s0 + row3.s2;                                                            \
            res.sd = row1.s1 + row1.s2 - row3.s1 - row3.s2;                                                            \
            res.se = row1.s2 - row1.s1 + row3.s1 - row3.s2;                                                            \
            res.sf = row1.s1 - row1.s3 - row3.s1 + row3.s3;                                                            \
        } else {                                                                                                       \
            DATA_T A[16] = {(0.0f)};                                                                                   \
            char index = 0;                                                                                            \
            for (int i = start_h_indata; i < start_h_indata + 4; i++)                                                  \
                for (int j = start_w_indata; j < start_w_indata + 4; j++) {                                            \
                    if (i >= 0 && i < inputHeight && j >= 0 && j < inputWidth) {                                       \
                        A[index] = input[inputBase + inputWidth * i + j];                                              \
                    }                                                                                                  \
                    index++;                                                                                           \
                }                                                                                                      \
            res.s0 = A[0] - A[2] - A[8] + A[10];                                                                       \
            res.s1 = A[1] + A[2] - A[9] - A[10];                                                                       \
            res.s2 = A[2] - A[1] + A[9] - A[10];                                                                       \
            res.s3 = A[1] - A[3] - A[9] + A[11];                                                                       \
            res.s4 = A[4] - A[6] + A[8] - A[10];                                                                       \
            res.s5 = A[5] + A[6] + A[9] + A[10];                                                                       \
            res.s6 = A[6] - A[5] - A[9] + A[10];                                                                       \
            res.s7 = A[5] - A[7] + A[9] - A[11];                                                                       \
            res.s8 = A[6] - A[4] + A[8] - A[10];                                                                       \
            res.s9 = A[9] - A[5] - A[6] + A[10];                                                                       \
            res.sa = A[5] - A[6] - A[9] + A[10];                                                                       \
            res.sb = A[7] - A[5] + A[9] - A[11];                                                                       \
            res.sc = A[4] - A[6] - A[12] + A[14];                                                                      \
            res.sd = A[5] + A[6] - A[13] - A[14];                                                                      \
            res.se = A[6] - A[5] + A[13] - A[14];                                                                      \
            res.sf = A[5] - A[7] - A[13] + A[15];                                                                      \
        }                                                                                                              \
        vstore4(res.lo.lo, 0, output + out_index);                                                                     \
        vstore4(res.lo.hi, 0, output + out_index + 4 * coalescingRowSize * chan);                                      \
        vstore4(res.hi.lo, 0, output + out_index + 8 * coalescingRowSize * chan);                                      \
        vstore4(res.hi.hi, 0, output + out_index + 12 * coalescingRowSize * chan);                                     \
    }

#define FAST3X3_2_OPTIMIZED_2X8_SPLITE(input,                                                                          \
                                       weight,                                                                         \
                                       bias,                                                                           \
                                       output,                                                                         \
                                       inputDataNum_per_out2x2,                                                        \
                                       ouputHeight,                                                                    \
                                       ouputWidth,                                                                     \
                                       channel,                                                                        \
                                       wgradTileHeight,                                                                \
                                       wgradTileWidth,                                                                 \
                                       coalescingRowSize)                                                              \
    if (get_global_id(2) * 2 < wgradTileHeight * wgradTileWidth) {                                                     \
        DATA_T16 out16_0 = 0.0f;                                                                                       \
        DATA_T16 out16_1 = 0.0f;                                                                                       \
        DATA_T16 out16_2 = 0.0f;                                                                                       \
        DATA_T16 out16_3 = 0.0f;                                                                                       \
        {                                                                                                              \
            int2 srcIndex =                                                                                            \
                (int2)((get_global_id(2) * 2 / coalescingRowSize) * coalescingRowSize * inputDataNum_per_out2x2 +      \
                           get_global_id(2) * 2 % coalescingRowSize / 2 * 8 +                                          \
                           (get_global_id(0) * 8 / channel) * inputDataNum_per_out2x2 *                                \
                               ((wgradTileHeight * wgradTileWidth + coalescingRowSize - 1) / coalescingRowSize *       \
                                coalescingRowSize) +                                                                   \
                           get_global_id(1) * coalescingRowSize * 4 * (inputDataNum_per_out2x2 / 16),                  \
                       inputDataNum_per_out2x2 * (get_global_id(0) * 8 % channel) +                                    \
                           get_global_id(1) * 32 * (inputDataNum_per_out2x2 / 16));                                    \
            int coalescingSize = coalescingRowSize * 4;                                                                \
            for (int i = 0; i<inputDataNum_per_out2x2>> 4; i++) {                                                     \
                DATA_T8 input8;                                                                                        \
                DATA_T8 input8_1;                                                                                      \
                DATA_T8 weight8;                                                                                       \
                input8 = vload8(0, input + srcIndex.s0);                                                               \
                weight8 = vload8(0, weight + srcIndex.s1);                                                             \
                out16_0.lo += input8.s0 * weight8;                                                                     \
                out16_0.hi += input8.s4 * weight8;                                                                     \
                weight8 = vload8(0, weight + srcIndex.s1 + 8);                                                         \
                out16_1.lo += input8.s1 * weight8;                                                                     \
                out16_1.hi += input8.s5 * weight8;                                                                     \
                weight8 = vload8(0, weight + srcIndex.s1 + 16);                                                        \
                out16_2.lo += input8.s2 * weight8;                                                                     \
                out16_2.hi += input8.s6 * weight8;                                                                     \
                weight8 = vload8(0, weight + srcIndex.s1 + 24);                                                        \
                out16_3.lo += input8.s3 * weight8;                                                                     \
                out16_3.hi += input8.s7 * weight8;                                                                     \
                srcIndex += (int2)(coalescingSize, 32);                                                                \
            }                                                                                                          \
            if (get_global_id(1) == 0) {                                                                               \
                out16_0 = out16_0 + out16_1 + out16_2;                                                                 \
                out16_1 = out16_1 - out16_2 - out16_3;                                                                 \
                out16_2 = 0.0f;                                                                                        \
                out16_3 = 0.0f;                                                                                        \
            } else if (get_global_id(1) == 1) {                                                                        \
                out16_0 = out16_0 + out16_1 + out16_2;                                                                 \
                out16_1 = out16_1 - out16_2 - out16_3;                                                                 \
                out16_2 = out16_0;                                                                                     \
                out16_3 = out16_1;                                                                                     \
            } else if (get_global_id(1) == 2) {                                                                        \
                out16_0 = out16_0 + out16_1 + out16_2;                                                                 \
                out16_1 = out16_1 - out16_2 - out16_3;                                                                 \
                out16_2 = -out16_0;                                                                                    \
                out16_3 = -out16_1;                                                                                    \
            } else {                                                                                                   \
                out16_0 = -out16_0 - out16_1 - out16_2;                                                                \
                out16_1 = -out16_1 + out16_2 + out16_3;                                                                \
                out16_2 = out16_0;                                                                                     \
                out16_3 = out16_1;                                                                                     \
                out16_0 = 0.0f;                                                                                        \
                out16_1 = 0.0f;                                                                                        \
            }                                                                                                          \
        }                                                                                                              \
        {                                                                                                              \
            int dstIndex =                                                                                             \
                (get_global_id(0) * 8 / channel) * (channel / 8) * ((wgradTileHeight * wgradTileWidth + 1) / 2) * 2 *  \
                    8 * 4 * 4 +                                                                                        \
                (get_global_id(0) * 8 % channel) * ((wgradTileHeight * wgradTileWidth + 1) / 2) * 2 * 4 * 4 +          \
                get_global_id(2) * 2 * 8 * 4 * 4 + get_global_id(1) * 4 * 8;                                           \
            vstore8(out16_0.lo, 0, output + dstIndex);                                                                 \
            vstore8(out16_1.lo, 0, output + dstIndex + 8);                                                             \
            vstore8(out16_2.lo, 0, output + dstIndex + 16);                                                            \
            vstore8(out16_3.lo, 0, output + dstIndex + 24);                                                            \
            vstore8(out16_0.hi, 0, output + dstIndex + 128);                                                           \
            vstore8(out16_1.hi, 0, output + dstIndex + 136);                                                           \
            vstore8(out16_2.hi, 0, output + dstIndex + 144);                                                           \
            vstore8(out16_3.hi, 0, output + dstIndex + 152);                                                           \
        }                                                                                                              \
    }

#define FAST3X3_2_OPTIMIZED_2X8_MERGE(input,                                                                           \
                                      weight,                                                                          \
                                      bias,                                                                            \
                                      output,                                                                          \
                                      inputDataNum_per_out2x2,                                                         \
                                      ouputHeight,                                                                     \
                                      ouputWidth,                                                                      \
                                      channel,                                                                         \
                                      wgradTileHeight,                                                                 \
                                      wgradTileWidth)                                                                  \
    if (get_global_id(2) * 2 + 1 < wgradTileHeight * wgradTileWidth) {                                                 \
        DATA_T8 bia = vload8(0, bias + get_global_id(0) * 8 % channel);                                                \
        DATA_T16 out_0 = (DATA_T16)(bia, bia);                                                                         \
        {                                                                                                              \
            int srcIndex =                                                                                             \
                (get_global_id(0) * 8 / channel) * (channel / 8) * ((wgradTileHeight * wgradTileWidth + 1) / 2) * 2 *  \
                    8 * 4 * 4 +                                                                                        \
                (get_global_id(0) * 8 % channel) * ((wgradTileHeight * wgradTileWidth + 1) / 2) * 2 * 4 * 4 +          \
                get_global_id(2) * 2 * 8 * 4 * 4 + get_global_id(1) * 8;                                               \
            DATA_T8 front8;                                                                                            \
            DATA_T8 end8;                                                                                              \
            front8 = vload8(0, input + srcIndex);                                                                      \
            end8 = vload8(0, input + srcIndex + 32);                                                                   \
            out_0.lo += front8 + end8;                                                                                 \
            front8 = vload8(0, input + srcIndex + 64);                                                                 \
            end8 = vload8(0, input + srcIndex + +96);                                                                  \
            out_0.lo += front8 + end8;                                                                                 \
            front8 = vload8(0, input + srcIndex + 128);                                                                \
            end8 = vload8(0, input + srcIndex + 160);                                                                  \
            out_0.hi += front8 + end8;                                                                                 \
            front8 = vload8(0, input + srcIndex + 192);                                                                \
            end8 = vload8(0, input + srcIndex + 224);                                                                  \
            out_0.hi += front8 + end8;                                                                                 \
        }                                                                                                              \
        {                                                                                                              \
            int firstWgradH = get_global_id(2) * 2 / wgradTileWidth;                                                   \
            int firstWgradW = get_global_id(2) * 2 % wgradTileWidth;                                                   \
            int dstIndex = ((get_global_id(0) * 8 / channel) * channel + get_global_id(0) * 8 % channel) *             \
                               ouputHeight * ouputWidth +                                                              \
                           (firstWgradH)*2 * ouputWidth + (firstWgradW)*2 + get_global_id(1) / 2 * ouputWidth +        \
                           get_global_id(1) % 2;                                                                       \
            if ((firstWgradW)*2 + 1 < ouputWidth && (firstWgradH)*2 + 1 < ouputHeight) {                               \
                output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s0);                                                        \
                output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s1);                             \
                output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s2);                         \
                output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s3);                         \
                output[dstIndex + 4 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s4);                         \
                output[dstIndex + 5 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s5);                         \
                output[dstIndex + 6 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s6);                         \
                output[dstIndex + 7 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s7);                         \
            } else if ((firstWgradH)*2 + 1 < ouputHeight) {                                                            \
                if (get_global_id(1) == 0 || get_global_id(1) == 2) {                                                  \
                    output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s0);                                                    \
                    output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s1);                         \
                    output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s2);                     \
                    output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s3);                     \
                    output[dstIndex + 4 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s4);                     \
                    output[dstIndex + 5 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s5);                     \
                    output[dstIndex + 6 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s6);                     \
                    output[dstIndex + 7 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s7);                     \
                }                                                                                                      \
            } else if ((firstWgradW)*2 + 1 < ouputWidth) {                                                             \
                if (get_global_id(1) == 0 || get_global_id(1) == 1) {                                                  \
                    output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s0);                                                    \
                    output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s1);                         \
                    output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s2);                     \
                    output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s3);                     \
                    output[dstIndex + 4 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s4);                     \
                    output[dstIndex + 5 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s5);                     \
                    output[dstIndex + 6 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s6);                     \
                    output[dstIndex + 7 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s7);                     \
                }                                                                                                      \
            } else {                                                                                                   \
                if (get_global_id(1) == 0) {                                                                           \
                    output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s0);                                                    \
                    output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s1);                         \
                    output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s2);                     \
                    output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s3);                     \
                    output[dstIndex + 4 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s4);                     \
                    output[dstIndex + 5 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s5);                     \
                    output[dstIndex + 6 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s6);                     \
                    output[dstIndex + 7 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s7);                     \
                }                                                                                                      \
            }                                                                                                          \
            int secondWgradH = (get_global_id(2) * 2 + 1) / wgradTileWidth;                                            \
            int secondWgradW = (get_global_id(2) * 2 + 1) % wgradTileWidth;                                            \
            dstIndex = ((get_global_id(0) * 8 / channel) * channel + (get_global_id(0) * 8 % channel)) * ouputHeight * \
                           ouputWidth +                                                                                \
                       (secondWgradH)*2 * ouputWidth + (secondWgradW)*2 + get_global_id(1) / 2 * ouputWidth +          \
                       get_global_id(1) % 2;                                                                           \
            if ((secondWgradW)*2 + 1 < ouputWidth && (secondWgradH)*2 + 1 < ouputHeight) {                             \
                output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s8);                                                        \
                output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s9);                             \
                output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.sa);                         \
                output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.sb);                         \
                output[dstIndex + 4 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.sc);                         \
                output[dstIndex + 5 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.sd);                         \
                output[dstIndex + 6 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.se);                         \
                output[dstIndex + 7 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.sf);                         \
            } else if ((secondWgradH)*2 + 1 < ouputHeight) {                                                           \
                if (get_global_id(1) == 0 || get_global_id(1) == 2) {                                                  \
                    output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s8);                                                    \
                    output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s9);                         \
                    output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.sa);                     \
                    output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.sb);                     \
                    output[dstIndex + 4 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.sc);                     \
                    output[dstIndex + 5 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.sd);                     \
                    output[dstIndex + 6 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.se);                     \
                    output[dstIndex + 7 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.sf);                     \
                }                                                                                                      \
            } else if ((secondWgradW)*2 + 1 < ouputWidth) {                                                            \
                if (get_global_id(1) == 0 || get_global_id(1) == 1) {                                                  \
                    output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s8);                                                    \
                    output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s9);                         \
                    output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.sa);                     \
                    output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.sb);                     \
                    output[dstIndex + 4 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.sc);                     \
                    output[dstIndex + 5 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.sd);                     \
                    output[dstIndex + 6 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.se);                     \
                    output[dstIndex + 7 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.sf);                     \
                }                                                                                                      \
            } else {                                                                                                   \
                if (get_global_id(1) == 0) {                                                                           \
                    output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s8);                                                    \
                    output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s9);                         \
                    output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.sa);                     \
                    output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.sb);                     \
                    output[dstIndex + 4 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.sc);                     \
                    output[dstIndex + 5 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.sd);                     \
                    output[dstIndex + 6 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.se);                     \
                    output[dstIndex + 7 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.sf);                     \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    } else if (get_global_id(2) * 2 < wgradTileHeight * wgradTileWidth) {                                              \
        DATA_T8 bia = vload8(0, bias + get_global_id(0) * 8 % channel);                                                \
        DATA_T16 out_0 = (DATA_T16)(bia, bia);                                                                         \
        {                                                                                                              \
            int srcIndex =                                                                                             \
                (get_global_id(0) * 8 / channel) * (channel / 8) * ((wgradTileHeight * wgradTileWidth + 1) / 2) * 2 *  \
                    8 * 4 * 4 +                                                                                        \
                (get_global_id(0) * 8 % channel) * ((wgradTileHeight * wgradTileWidth + 1) / 2) * 2 * 4 * 4 +          \
                get_global_id(2) * 2 * 8 * 4 * 4 + get_global_id(1) * 8;                                               \
            DATA_T8 front8;                                                                                            \
            DATA_T8 end8;                                                                                              \
            front8 = vload8(0, input + srcIndex);                                                                      \
            end8 = vload8(0, input + srcIndex + 32);                                                                   \
            out_0.lo += front8 + end8;                                                                                 \
            front8 = vload8(0, input + srcIndex + 64);                                                                 \
            end8 = vload8(0, input + srcIndex + +96);                                                                  \
            out_0.lo += front8 + end8;                                                                                 \
            front8 = vload8(0, input + srcIndex + 128);                                                                \
            end8 = vload8(0, input + srcIndex + 160);                                                                  \
            out_0.hi += front8 + end8;                                                                                 \
            front8 = vload8(0, input + srcIndex + 192);                                                                \
            end8 = vload8(0, input + srcIndex + 224);                                                                  \
            out_0.hi += front8 + end8;                                                                                 \
        }                                                                                                              \
        {                                                                                                              \
            int firstWgradH = get_global_id(2) * 2 / wgradTileWidth;                                                   \
            int firstWgradW = get_global_id(2) * 2 % wgradTileWidth;                                                   \
            int dstIndex = ((get_global_id(0) * 8 / channel) * channel + get_global_id(0) * 8 % channel) *             \
                               ouputHeight * ouputWidth +                                                              \
                           (firstWgradH)*2 * ouputWidth + (firstWgradW)*2 + get_global_id(1) / 2 * ouputWidth +        \
                           get_global_id(1) % 2;                                                                       \
            if ((firstWgradW)*2 + 1 < ouputWidth && (firstWgradH)*2 + 1 < ouputHeight) {                               \
                output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s0);                                                        \
                output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s1);                             \
                output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s2);                         \
                output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s3);                         \
                output[dstIndex + 4 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s4);                         \
                output[dstIndex + 5 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s5);                         \
                output[dstIndex + 6 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s6);                         \
                output[dstIndex + 7 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s7);                         \
            } else if ((firstWgradH)*2 + 1 < ouputHeight) {                                                            \
                if (get_global_id(1) == 0 || get_global_id(1) == 2) {                                                  \
                    output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s0);                                                    \
                    output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s1);                         \
                    output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s2);                     \
                    output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s3);                     \
                    output[dstIndex + 4 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s4);                     \
                    output[dstIndex + 5 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s5);                     \
                    output[dstIndex + 6 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s6);                     \
                    output[dstIndex + 7 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s7);                     \
                }                                                                                                      \
            } else if ((firstWgradW)*2 + 1 < ouputWidth) {                                                             \
                if (get_global_id(1) == 0 || get_global_id(1) == 1) {                                                  \
                    output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s0);                                                    \
                    output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s1);                         \
                    output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s2);                     \
                    output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s3);                     \
                    output[dstIndex + 4 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s4);                     \
                    output[dstIndex + 5 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s5);                     \
                    output[dstIndex + 6 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s6);                     \
                    output[dstIndex + 7 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s7);                     \
                }                                                                                                      \
            } else {                                                                                                   \
                if (get_global_id(1) == 0) {                                                                           \
                    output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s0);                                                    \
                    output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s1);                         \
                    output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s2);                     \
                    output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s3);                     \
                    output[dstIndex + 4 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s4);                     \
                    output[dstIndex + 5 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s5);                     \
                    output[dstIndex + 6 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s6);                     \
                    output[dstIndex + 7 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s7);                     \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }

#define FAST3X3_1_2X4(                                                                                                 \
    input, output, padHeight, padWidth, inputHeight, inputWidth, chan, height, width, coalescingRowSize)               \
    int picOff = get_global_id(0);                                                                                     \
    int channel = get_global_id(1);                                                                                    \
    int hwIndex = get_global_id(2);                                                                                    \
    if (hwIndex < height * width) {                                                                                    \
        int gID1 = hwIndex / width;                                                                                    \
        int gID2 = hwIndex % width;                                                                                    \
        int start_h_indata = gID1 * 2 - padHeight;                                                                     \
        int start_w_indata = gID2 * 2 - padWidth;                                                                      \
        int row = gID1 * width + gID2;                                                                                 \
        int alignedHW = (height * width + coalescingRowSize - 1) / coalescingRowSize * coalescingRowSize;              \
        int out_index = picOff * chan * alignedHW * 16 + row / coalescingRowSize * coalescingRowSize * chan * 16 +     \
                        row % 2 * coalescingRowSize / 2 * 8 + row % coalescingRowSize / 2 * 8 +                        \
                        channel * coalescingRowSize * 8;                                                               \
        int inputBase = picOff * chan * inputHeight * inputWidth + channel * inputHeight * inputWidth;                 \
        DATA_T16 res;                                                                                                  \
        if (start_h_indata >= 0 && start_h_indata + 4 <= inputHeight && start_w_indata >= 0 &&                         \
            start_w_indata + 4 <= inputWidth) {                                                                        \
            int rowBase = inputBase + inputWidth * start_h_indata + start_w_indata;                                    \
            DATA_T4 row0 = vload4(0, input + rowBase);                                                                 \
            rowBase = rowBase + inputWidth;                                                                            \
            DATA_T4 row1 = vload4(0, input + rowBase);                                                                 \
            rowBase = rowBase + inputWidth;                                                                            \
            DATA_T4 row2 = vload4(0, input + rowBase);                                                                 \
            rowBase = rowBase + inputWidth;                                                                            \
            DATA_T4 row3 = vload4(0, input + rowBase);                                                                 \
            res.s0 = row0.s0 - row0.s2 - row2.s0 + row2.s2;                                                            \
            res.s1 = row0.s1 + row0.s2 - row2.s1 - row2.s2;                                                            \
            res.s2 = row0.s2 - row0.s1 + row2.s1 - row2.s2;                                                            \
            res.s3 = row0.s1 - row0.s3 - row2.s1 + row2.s3;                                                            \
            res.s4 = row1.s0 - row1.s2 + row2.s0 - row2.s2;                                                            \
            res.s5 = row1.s1 + row1.s2 + row2.s1 + row2.s2;                                                            \
            res.s6 = row1.s2 - row1.s1 - row2.s1 + row2.s2;                                                            \
            res.s7 = row1.s1 - row1.s3 + row2.s1 - row2.s3;                                                            \
            res.s8 = row1.s2 - row1.s0 + row2.s0 - row2.s2;                                                            \
            res.s9 = row2.s1 - row1.s1 - row1.s2 + row2.s2;                                                            \
            res.sa = row1.s1 - row1.s2 - row2.s1 + row2.s2;                                                            \
            res.sb = row1.s3 - row1.s1 + row2.s1 - row2.s3;                                                            \
            res.sc = row1.s0 - row1.s2 - row3.s0 + row3.s2;                                                            \
            res.sd = row1.s1 + row1.s2 - row3.s1 - row3.s2;                                                            \
            res.se = row1.s2 - row1.s1 + row3.s1 - row3.s2;                                                            \
            res.sf = row1.s1 - row1.s3 - row3.s1 + row3.s3;                                                            \
        } else {                                                                                                       \
            DATA_T A[16] = {(0.0f)};                                                                                   \
            char index = 0;                                                                                            \
            for (int i = start_h_indata; i < start_h_indata + 4; i++)                                                  \
                for (int j = start_w_indata; j < start_w_indata + 4; j++) {                                            \
                    if (i >= 0 && i < inputHeight && j >= 0 && j < inputWidth) {                                       \
                        A[index] = input[inputBase + inputWidth * i + j];                                              \
                    }                                                                                                  \
                    index++;                                                                                           \
                }                                                                                                      \
            res.s0 = A[0] - A[2] - A[8] + A[10];                                                                       \
            res.s1 = A[1] + A[2] - A[9] - A[10];                                                                       \
            res.s2 = A[2] - A[1] + A[9] - A[10];                                                                       \
            res.s3 = A[1] - A[3] - A[9] + A[11];                                                                       \
            res.s4 = A[4] - A[6] + A[8] - A[10];                                                                       \
            res.s5 = A[5] + A[6] + A[9] + A[10];                                                                       \
            res.s6 = A[6] - A[5] - A[9] + A[10];                                                                       \
            res.s7 = A[5] - A[7] + A[9] - A[11];                                                                       \
            res.s8 = A[6] - A[4] + A[8] - A[10];                                                                       \
            res.s9 = A[9] - A[5] - A[6] + A[10];                                                                       \
            res.sa = A[5] - A[6] - A[9] + A[10];                                                                       \
            res.sb = A[7] - A[5] + A[9] - A[11];                                                                       \
            res.sc = A[4] - A[6] - A[12] + A[14];                                                                      \
            res.sd = A[5] + A[6] - A[13] - A[14];                                                                      \
            res.se = A[6] - A[5] + A[13] - A[14];                                                                      \
            res.sf = A[5] - A[7] - A[13] + A[15];                                                                      \
        }                                                                                                              \
        vstore8(res.lo, 0, output + out_index);                                                                        \
        vstore8(res.hi, 0, output + out_index + 8 * coalescingRowSize * chan);                                         \
    }

#define FAST3X3_2_OPTIMIZED_2X4_SPLITE(input,                                                                          \
                                       weight,                                                                         \
                                       bias,                                                                           \
                                       output,                                                                         \
                                       inputDataNum_per_out2x2,                                                        \
                                       ouputHeight,                                                                    \
                                       ouputWidth,                                                                     \
                                       channel,                                                                        \
                                       wgradTileHeight,                                                                \
                                       wgradTileWidth,                                                                 \
                                       coalescingRowSize)                                                              \
    if (get_global_id(2) * 2 < wgradTileHeight * wgradTileWidth) {                                                     \
        DATA_T8 out_0;                                                                                                 \
        DATA_T8 out_1;                                                                                                 \
        DATA_T8 out_2;                                                                                                 \
        DATA_T8 out_3;                                                                                                 \
        {                                                                                                              \
            DATA_T8 out8_0 = 0.0f;                                                                                     \
            DATA_T8 out8_1 = 0.0f;                                                                                     \
            DATA_T8 out8_2 = 0.0f;                                                                                     \
            DATA_T8 out8_3 = 0.0f;                                                                                     \
            DATA_T8 out8_4 = 0.0f;                                                                                     \
            DATA_T8 out8_5 = 0.0f;                                                                                     \
            DATA_T8 out8_6 = 0.0f;                                                                                     \
            DATA_T8 out8_7 = 0.0f;                                                                                     \
            int2 srcIndex =                                                                                            \
                (int2)((get_global_id(2) * 2 / coalescingRowSize) * coalescingRowSize * inputDataNum_per_out2x2 +      \
                           get_global_id(2) * 2 % coalescingRowSize / 2 * 8 +                                          \
                           (get_global_id(0) * 4 / channel) * inputDataNum_per_out2x2 *                                \
                               ((wgradTileHeight * wgradTileWidth + coalescingRowSize - 1) / coalescingRowSize *       \
                                coalescingRowSize) +                                                                   \
                           get_global_id(1) * coalescingRowSize * 8 * (inputDataNum_per_out2x2 / 16),                  \
                       inputDataNum_per_out2x2 * (get_global_id(0) * 4 % channel) +                                    \
                           get_global_id(1) * 32 * (inputDataNum_per_out2x2 / 16));                                    \
            int coalescingSize = coalescingRowSize * 8;                                                                \
            for (int i = 0; i<inputDataNum_per_out2x2>> 4; i++) {                                                     \
                DATA_T8 input8;                                                                                        \
                DATA_T8 input8_1;                                                                                      \
                DATA_T8 weight8;                                                                                       \
                input8 = vload8(0, input + srcIndex.s0);                                                               \
                input8_1 = vload8(0, input + srcIndex.s0 + 96);                                                        \
                weight8 = vload8(0, weight + srcIndex.s1);                                                             \
                out8_0.lo += input8.s0 * weight8.lo;                                                                   \
                out8_0.hi += input8_1.s0 * weight8.lo;                                                                 \
                out8_1.lo += input8.s1 * weight8.hi;                                                                   \
                out8_1.hi += input8_1.s1 * weight8.hi;                                                                 \
                weight8 = vload8(0, weight + srcIndex.s1 + 8);                                                         \
                out8_2.lo += input8.s2 * weight8.lo;                                                                   \
                out8_2.hi += input8_1.s2 * weight8.lo;                                                                 \
                out8_3.lo += input8.s3 * weight8.hi;                                                                   \
                out8_3.hi += input8_1.s3 * weight8.hi;                                                                 \
                weight8 = vload8(0, weight + srcIndex.s1 + 16);                                                        \
                out8_4.lo += input8.s4 * weight8.lo;                                                                   \
                out8_4.hi += input8_1.s4 * weight8.lo;                                                                 \
                out8_5.lo += input8.s5 * weight8.hi;                                                                   \
                out8_5.hi += input8_1.s5 * weight8.hi;                                                                 \
                weight8 = vload8(0, weight + srcIndex.s1 + 24);                                                        \
                out8_6.lo += input8.s6 * weight8.lo;                                                                   \
                out8_6.hi += input8_1.s6 * weight8.lo;                                                                 \
                out8_7.lo += input8.s7 * weight8.hi;                                                                   \
                out8_7.hi += input8_1.s7 * weight8.hi;                                                                 \
                srcIndex += (int2)(coalescingSize, 32);                                                                \
            }                                                                                                          \
            if (get_global_id(1) == 0) {                                                                               \
                out_0 = out8_0 + out8_1 + out8_2 + out8_4 + out8_5 + out8_6;                                           \
                out_1 = out8_1 - out8_2 - out8_3 + out8_5 - out8_6 - out8_7;                                           \
                out_2 = out8_4 + out8_5 + out8_6;                                                                      \
                out_3 = out8_5 - out8_6 - out8_7;                                                                      \
            } else {                                                                                                   \
                out_0 = out8_0 + out8_1 + out8_2;                                                                      \
                out_1 = out8_1 - out8_2 - out8_3;                                                                      \
                out_2 = -out8_0 - out8_1 - out8_2 - out8_4 - out8_5 - out8_6;                                          \
                out_3 = -out8_1 + out8_2 + out8_3 - out8_5 + out8_6 + out8_7;                                          \
            }                                                                                                          \
        }                                                                                                              \
        {                                                                                                              \
            int dstIndex =                                                                                             \
                (get_global_id(0) * 4 / channel) * (channel / 4) * ((wgradTileHeight * wgradTileWidth + 1) / 2) * 2 *  \
                    4 * 2 * 4 +                                                                                        \
                (get_global_id(0) * 4 % channel) * ((wgradTileHeight * wgradTileWidth + 1) / 2) * 2 * 2 * 4 +          \
                get_global_id(2) * 2 * 4 * 2 * 4 + get_global_id(1) * 2 * 4;                                           \
            vstore8(out_0, 0, output + dstIndex);                                                                      \
            vstore8(out_1, 0, output + dstIndex + 16);                                                                 \
            vstore8(out_2, 0, output + dstIndex + 32);                                                                 \
            vstore8(out_3, 0, output + dstIndex + 48);                                                                 \
        }                                                                                                              \
    }

#define FAST3X3_2_OPTIMIZED_2X4_MERGE(input,                                                                           \
                                      weight,                                                                          \
                                      bias,                                                                            \
                                      output,                                                                          \
                                      inputDataNum_per_out2x2,                                                         \
                                      ouputHeight,                                                                     \
                                      ouputWidth,                                                                      \
                                      channel,                                                                         \
                                      wgradTileHeight,                                                                 \
                                      wgradTileWidth)                                                                  \
    if (get_global_id(2) * 2 + 1 < wgradTileHeight * wgradTileWidth) {                                                 \
        DATA_T4 bia = vload4(0, bias + get_global_id(1) * 4);                                                          \
        DATA_T8 out_0 = (DATA_T8)(bia, bia);                                                                           \
        DATA_T8 out_1 = (DATA_T8)(bia, bia);                                                                           \
        DATA_T8 out_2 = (DATA_T8)(bia, bia);                                                                           \
        DATA_T8 out_3 = (DATA_T8)(bia, bia);                                                                           \
        {                                                                                                              \
            int srcIndex =                                                                                             \
                get_global_id(0) * (channel / 4) * ((wgradTileHeight * wgradTileWidth + 1) / 2) * 2 * 4 * 2 * 4 +      \
                get_global_id(1) * ((wgradTileHeight * wgradTileWidth + 1) / 2) * 2 * 4 * 2 * 4 +                      \
                get_global_id(2) * 2 * 4 * 2 * 4;                                                                      \
            DATA_T8 front8;                                                                                            \
            DATA_T8 end8;                                                                                              \
            front8 = vload8(0, input + srcIndex);                                                                      \
            end8 = vload8(0, input + srcIndex + 8);                                                                    \
            out_0 += front8 + end8;                                                                                    \
            front8 = vload8(0, input + srcIndex + 16);                                                                 \
            end8 = vload8(0, input + srcIndex + 24);                                                                   \
            out_1 += front8 + end8;                                                                                    \
            front8 = vload8(0, input + srcIndex + 32);                                                                 \
            end8 = vload8(0, input + srcIndex + 40);                                                                   \
            out_2 += front8 + end8;                                                                                    \
            front8 = vload8(0, input + srcIndex + 48);                                                                 \
            end8 = vload8(0, input + srcIndex + 56);                                                                   \
            out_3 += front8 + end8;                                                                                    \
        }                                                                                                              \
        {                                                                                                              \
            int firstWgradH = get_global_id(2) * 2 / wgradTileWidth;                                                   \
            int firstWgradW = get_global_id(2) * 2 % wgradTileWidth;                                                   \
            int dstIndex = (get_global_id(0) * channel + get_global_id(1) * 4) * ouputHeight * ouputWidth +            \
                           (firstWgradH)*2 * ouputWidth + (firstWgradW)*2;                                             \
            if ((firstWgradW)*2 + 1 < ouputWidth && (firstWgradH)*2 + 1 < ouputHeight) {                               \
                output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s0);                                                        \
                output[dstIndex + 1] = ACT_VEC_F(DATA_T, out_1.s0);                                                    \
                output[dstIndex + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s0);                                           \
                output[dstIndex + ouputWidth + 1] = ACT_VEC_F(DATA_T, out_3.s0);                                       \
                output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s1);                             \
                output[dstIndex + ouputHeight * ouputWidth + 1] = ACT_VEC_F(DATA_T, out_1.s1);                         \
                output[dstIndex + ouputHeight * ouputWidth + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s1);                \
                output[dstIndex + ouputHeight * ouputWidth + ouputWidth + 1] = ACT_VEC_F(DATA_T, out_3.s1);            \
                output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s2);                         \
                output[dstIndex + 2 * ouputHeight * ouputWidth + 1] = ACT_VEC_F(DATA_T, out_1.s2);                     \
                output[dstIndex + 2 * ouputHeight * ouputWidth + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s2);            \
                output[dstIndex + 2 * ouputHeight * ouputWidth + ouputWidth + 1] = ACT_VEC_F(DATA_T, out_3.s2);        \
                output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s3);                         \
                output[dstIndex + 3 * ouputHeight * ouputWidth + 1] = ACT_VEC_F(DATA_T, out_1.s3);                     \
                output[dstIndex + 3 * ouputHeight * ouputWidth + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s3);            \
                output[dstIndex + 3 * ouputHeight * ouputWidth + ouputWidth + 1] = ACT_VEC_F(DATA_T, out_3.s3);        \
            } else if ((firstWgradH)*2 + 1 < ouputHeight) {                                                            \
                output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s0);                                                        \
                output[dstIndex + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s0);                                           \
                output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s1);                             \
                output[dstIndex + ouputHeight * ouputWidth + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s1);                \
                output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s2);                         \
                output[dstIndex + 2 * ouputHeight * ouputWidth + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s2);            \
                output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s3);                         \
                output[dstIndex + 3 * ouputHeight * ouputWidth + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s3);            \
            } else if ((firstWgradW)*2 + 1 < ouputWidth) {                                                             \
                output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s0);                                                        \
                output[dstIndex + 1] = ACT_VEC_F(DATA_T, out_1.s0);                                                    \
                output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s1);                             \
                output[dstIndex + ouputHeight * ouputWidth + 1] = ACT_VEC_F(DATA_T, out_1.s1);                         \
                output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s2);                         \
                output[dstIndex + 2 * ouputHeight * ouputWidth + 1] = ACT_VEC_F(DATA_T, out_1.s2);                     \
                output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s3);                         \
                output[dstIndex + 3 * ouputHeight * ouputWidth + 1] = ACT_VEC_F(DATA_T, out_1.s3);                     \
            } else {                                                                                                   \
                output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s0);                                                        \
                output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s1);                             \
                output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s2);                         \
                output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s3);                         \
            }                                                                                                          \
            int secondWgradH = (get_global_id(2) * 2 + 1) / wgradTileWidth;                                            \
            int secondWgradW = (get_global_id(2) * 2 + 1) % wgradTileWidth;                                            \
            dstIndex = (get_global_id(0) * channel + get_global_id(1) * 4) * ouputHeight * ouputWidth +                \
                       (secondWgradH)*2 * ouputWidth + (secondWgradW)*2;                                               \
            if ((secondWgradW)*2 + 1 < ouputWidth && (secondWgradH)*2 + 1 < ouputHeight) {                             \
                output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s4);                                                        \
                output[dstIndex + 1] = ACT_VEC_F(DATA_T, out_1.s4);                                                    \
                output[dstIndex + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s4);                                           \
                output[dstIndex + ouputWidth + 1] = ACT_VEC_F(DATA_T, out_3.s4);                                       \
                output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s5);                             \
                output[dstIndex + ouputHeight * ouputWidth + 1] = ACT_VEC_F(DATA_T, out_1.s5);                         \
                output[dstIndex + ouputHeight * ouputWidth + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s5);                \
                output[dstIndex + ouputHeight * ouputWidth + ouputWidth + 1] = ACT_VEC_F(DATA_T, out_3.s5);            \
                output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s6);                         \
                output[dstIndex + 2 * ouputHeight * ouputWidth + 1] = ACT_VEC_F(DATA_T, out_1.s6);                     \
                output[dstIndex + 2 * ouputHeight * ouputWidth + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s6);            \
                output[dstIndex + 2 * ouputHeight * ouputWidth + ouputWidth + 1] = ACT_VEC_F(DATA_T, out_3.s6);        \
                output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s7);                         \
                output[dstIndex + 3 * ouputHeight * ouputWidth + 1] = ACT_VEC_F(DATA_T, out_1.s7);                     \
                output[dstIndex + 3 * ouputHeight * ouputWidth + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s7);            \
                output[dstIndex + 3 * ouputHeight * ouputWidth + ouputWidth + 1] = ACT_VEC_F(DATA_T, out_3.s7);        \
            } else if ((secondWgradH)*2 + 1 < ouputHeight) {                                                           \
                output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s4);                                                        \
                output[dstIndex + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s4);                                           \
                output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s5);                             \
                output[dstIndex + ouputHeight * ouputWidth + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s5);                \
                output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s6);                         \
                output[dstIndex + 2 * ouputHeight * ouputWidth + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s6);            \
                output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s7);                         \
                output[dstIndex + 3 * ouputHeight * ouputWidth + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s7);            \
            } else if ((secondWgradW)*2 + 1 < ouputWidth) {                                                            \
                output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s4);                                                        \
                output[dstIndex + 1] = ACT_VEC_F(DATA_T, out_1.s4);                                                    \
                output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s5);                             \
                output[dstIndex + ouputHeight * ouputWidth + 1] = ACT_VEC_F(DATA_T, out_1.s5);                         \
                output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s6);                         \
                output[dstIndex + 2 * ouputHeight * ouputWidth + 1] = ACT_VEC_F(DATA_T, out_1.s6);                     \
                output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s7);                         \
                output[dstIndex + 3 * ouputHeight * ouputWidth + 1] = ACT_VEC_F(DATA_T, out_1.s7);                     \
            } else {                                                                                                   \
                output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s4);                                                        \
                output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s5);                             \
                output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s6);                         \
                output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s7);                         \
            }                                                                                                          \
        }                                                                                                              \
    } else if (get_global_id(2) * 2 < wgradTileHeight * wgradTileWidth) {                                              \
        DATA_T4 bia = vload4(0, bias + get_global_id(1) * 4);                                                          \
        DATA_T8 out_0 = (DATA_T8)(bia, bia);                                                                           \
        DATA_T8 out_1 = (DATA_T8)(bia, bia);                                                                           \
        DATA_T8 out_2 = (DATA_T8)(bia, bia);                                                                           \
        DATA_T8 out_3 = (DATA_T8)(bia, bia);                                                                           \
        {                                                                                                              \
            int srcIndex =                                                                                             \
                get_global_id(0) * (channel / 4) * ((wgradTileHeight * wgradTileWidth + 1) / 2) * 2 * 4 * 2 * 4 +      \
                get_global_id(1) * ((wgradTileHeight * wgradTileWidth + 1) / 2) * 2 * 4 * 2 * 4 +                      \
                get_global_id(2) * 2 * 4 * 2 * 4;                                                                      \
            DATA_T8 front8;                                                                                            \
            DATA_T8 end8;                                                                                              \
            front8 = vload8(0, input + srcIndex);                                                                      \
            end8 = vload8(0, input + srcIndex + 8);                                                                    \
            out_0 += front8 + end8;                                                                                    \
            front8 = vload8(0, input + srcIndex + 16);                                                                 \
            end8 = vload8(0, input + srcIndex + 24);                                                                   \
            out_1 += front8 + end8;                                                                                    \
            front8 = vload8(0, input + srcIndex + 32);                                                                 \
            end8 = vload8(0, input + srcIndex + 40);                                                                   \
            out_2 += front8 + end8;                                                                                    \
            front8 = vload8(0, input + srcIndex + 48);                                                                 \
            end8 = vload8(0, input + srcIndex + 56);                                                                   \
            out_3 += front8 + end8;                                                                                    \
        }                                                                                                              \
        {                                                                                                              \
            int firstWgradH = get_global_id(2) * 2 / wgradTileWidth;                                                   \
            int firstWgradW = get_global_id(2) * 2 % wgradTileWidth;                                                   \
            int dstIndex = (get_global_id(0) * channel + get_global_id(1) * 4) * ouputHeight * ouputWidth +            \
                           (firstWgradH)*2 * ouputWidth + (firstWgradW)*2;                                             \
            if ((firstWgradW)*2 + 1 < ouputWidth && (firstWgradH)*2 + 1 < ouputHeight) {                               \
                output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s0);                                                        \
                output[dstIndex + 1] = ACT_VEC_F(DATA_T, out_1.s0);                                                    \
                output[dstIndex + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s0);                                           \
                output[dstIndex + ouputWidth + 1] = ACT_VEC_F(DATA_T, out_3.s0);                                       \
                output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s1);                             \
                output[dstIndex + ouputHeight * ouputWidth + 1] = ACT_VEC_F(DATA_T, out_1.s1);                         \
                output[dstIndex + ouputHeight * ouputWidth + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s1);                \
                output[dstIndex + ouputHeight * ouputWidth + ouputWidth + 1] = ACT_VEC_F(DATA_T, out_3.s1);            \
                output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s2);                         \
                output[dstIndex + 2 * ouputHeight * ouputWidth + 1] = ACT_VEC_F(DATA_T, out_1.s2);                     \
                output[dstIndex + 2 * ouputHeight * ouputWidth + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s2);            \
                output[dstIndex + 2 * ouputHeight * ouputWidth + ouputWidth + 1] = ACT_VEC_F(DATA_T, out_3.s2);        \
                output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s3);                         \
                output[dstIndex + 3 * ouputHeight * ouputWidth + 1] = ACT_VEC_F(DATA_T, out_1.s3);                     \
                output[dstIndex + 3 * ouputHeight * ouputWidth + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s3);            \
                output[dstIndex + 3 * ouputHeight * ouputWidth + ouputWidth + 1] = ACT_VEC_F(DATA_T, out_3.s3);        \
            } else if ((firstWgradH)*2 + 1 < ouputHeight) {                                                            \
                output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s0);                                                        \
                output[dstIndex + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s0);                                           \
                output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s1);                             \
                output[dstIndex + ouputHeight * ouputWidth + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s1);                \
                output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s2);                         \
                output[dstIndex + 2 * ouputHeight * ouputWidth + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s2);            \
                output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s3);                         \
                output[dstIndex + 3 * ouputHeight * ouputWidth + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s3);            \
            } else if ((firstWgradW)*2 + 1 < ouputWidth) {                                                             \
                output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s0);                                                        \
                output[dstIndex + 1] = ACT_VEC_F(DATA_T, out_1.s0);                                                    \
                output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s1);                             \
                output[dstIndex + ouputHeight * ouputWidth + 1] = ACT_VEC_F(DATA_T, out_1.s1);                         \
                output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s2);                         \
                output[dstIndex + 2 * ouputHeight * ouputWidth + 1] = ACT_VEC_F(DATA_T, out_1.s2);                     \
                output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s3);                         \
                output[dstIndex + 3 * ouputHeight * ouputWidth + 1] = ACT_VEC_F(DATA_T, out_1.s3);                     \
            } else {                                                                                                   \
                output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s0);                                                        \
                output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s1);                             \
                output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s2);                         \
                output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s3);                         \
            }                                                                                                          \
        }                                                                                                              \
    }

#define FAST3X3_1_1X8_FP16(                                                                                            \
    input, output, padHeight, padWidth, inputHeight, inputWidth, chan, height, width, coalescingRowSize)               \
    int picOff = get_global_id(0);                                                                                     \
    int channel = get_global_id(1);                                                                                    \
    int hwIndex = get_global_id(2);                                                                                    \
    if (hwIndex < height * width) {                                                                                    \
        int gID1 = hwIndex / width;                                                                                    \
        int gID2 = hwIndex % width;                                                                                    \
        int start_h_indata = gID1 * 2 - padHeight;                                                                     \
        int start_w_indata = gID2 * 2 - padWidth;                                                                      \
        int row = gID1 * width + gID2;                                                                                 \
        int alignedHW = (height * width + coalescingRowSize - 1) / coalescingRowSize * coalescingRowSize;              \
        int out_index = picOff * chan * alignedHW * 16 + row / coalescingRowSize * coalescingRowSize * chan * 16 +     \
                        row % coalescingRowSize * 8 + channel * coalescingRowSize * 8;                                 \
        int inputBase = picOff * chan * inputHeight * inputWidth + channel * inputHeight * inputWidth;                 \
        DATA_T16 res;                                                                                                  \
        if (start_h_indata >= 0 && start_h_indata + 4 <= inputHeight && start_w_indata >= 0 &&                         \
            start_w_indata + 4 <= inputWidth) {                                                                        \
            int rowBase = inputBase + inputWidth * start_h_indata + start_w_indata;                                    \
            DATA_T4 row0 = vload4(0, input + rowBase);                                                                 \
            rowBase = rowBase + inputWidth;                                                                            \
            DATA_T4 row1 = vload4(0, input + rowBase);                                                                 \
            rowBase = rowBase + inputWidth;                                                                            \
            DATA_T4 row2 = vload4(0, input + rowBase);                                                                 \
            rowBase = rowBase + inputWidth;                                                                            \
            DATA_T4 row3 = vload4(0, input + rowBase);                                                                 \
            res.s0 = row0.s0 - row0.s2 - row2.s0 + row2.s2;                                                            \
            res.s1 = row0.s1 + row0.s2 - row2.s1 - row2.s2;                                                            \
            res.s2 = row0.s2 - row0.s1 + row2.s1 - row2.s2;                                                            \
            res.s3 = row0.s1 - row0.s3 - row2.s1 + row2.s3;                                                            \
            res.s4 = row1.s0 - row1.s2 + row2.s0 - row2.s2;                                                            \
            res.s5 = row1.s1 + row1.s2 + row2.s1 + row2.s2;                                                            \
            res.s6 = row1.s2 - row1.s1 - row2.s1 + row2.s2;                                                            \
            res.s7 = row1.s1 - row1.s3 + row2.s1 - row2.s3;                                                            \
            res.s8 = row1.s2 - row1.s0 + row2.s0 - row2.s2;                                                            \
            res.s9 = row2.s1 - row1.s1 - row1.s2 + row2.s2;                                                            \
            res.sa = row1.s1 - row1.s2 - row2.s1 + row2.s2;                                                            \
            res.sb = row1.s3 - row1.s1 + row2.s1 - row2.s3;                                                            \
            res.sc = row1.s0 - row1.s2 - row3.s0 + row3.s2;                                                            \
            res.sd = row1.s1 + row1.s2 - row3.s1 - row3.s2;                                                            \
            res.se = row1.s2 - row1.s1 + row3.s1 - row3.s2;                                                            \
            res.sf = row1.s1 - row1.s3 - row3.s1 + row3.s3;                                                            \
        } else {                                                                                                       \
            DATA_T A[16] = {(0.0f)};                                                                                   \
            char index = 0;                                                                                            \
            for (int i = start_h_indata; i < start_h_indata + 4; i++)                                                  \
                for (int j = start_w_indata; j < start_w_indata + 4; j++) {                                            \
                    if (i >= 0 && i < inputHeight && j >= 0 && j < inputWidth) {                                       \
                        A[index] = input[inputBase + inputWidth * i + j];                                              \
                    }                                                                                                  \
                    index++;                                                                                           \
                }                                                                                                      \
            res.s0 = A[0] - A[2] - A[8] + A[10];                                                                       \
            res.s1 = A[1] + A[2] - A[9] - A[10];                                                                       \
            res.s2 = A[2] - A[1] + A[9] - A[10];                                                                       \
            res.s3 = A[1] - A[3] - A[9] + A[11];                                                                       \
            res.s4 = A[4] - A[6] + A[8] - A[10];                                                                       \
            res.s5 = A[5] + A[6] + A[9] + A[10];                                                                       \
            res.s6 = A[6] - A[5] - A[9] + A[10];                                                                       \
            res.s7 = A[5] - A[7] + A[9] - A[11];                                                                       \
            res.s8 = A[6] - A[4] + A[8] - A[10];                                                                       \
            res.s9 = A[9] - A[5] - A[6] + A[10];                                                                       \
            res.sa = A[5] - A[6] - A[9] + A[10];                                                                       \
            res.sb = A[7] - A[5] + A[9] - A[11];                                                                       \
            res.sc = A[4] - A[6] - A[12] + A[14];                                                                      \
            res.sd = A[5] + A[6] - A[13] - A[14];                                                                      \
            res.se = A[6] - A[5] + A[13] - A[14];                                                                      \
            res.sf = A[5] - A[7] - A[13] + A[15];                                                                      \
        }                                                                                                              \
        vstore8(res.lo, 0, output + out_index);                                                                        \
        vstore8(res.hi, 0, output + out_index + 8 * coalescingRowSize * chan);                                         \
    }

#define FAST3X3_1_1X8_FP32(input, output, padHeight, padWidth, inputHeight, inputWidth, chan, height, width)           \
    int gID0 = get_global_id(0);                                                                                       \
    int gID1 = get_global_id(1);                                                                                       \
    int gID2 = get_global_id(2);                                                                                       \
    if (gID1 < height && gID2 < width) {                                                                               \
        int picOff = gID0 / chan;                                                                                      \
        int channel = gID0 % chan;                                                                                     \
        int start_h_indata = gID1 * 2 - padHeight;                                                                     \
        int start_w_indata = gID2 * 2 - padWidth;                                                                      \
        int out_index =                                                                                                \
            picOff * chan * height * width * 16 + chan * 16 * width * gID1 + chan * 16 * gID2 + channel * 4;           \
        int inputBase = picOff * chan * inputHeight * inputWidth + channel * inputHeight * inputWidth;                 \
        DATA_T16 res;                                                                                                  \
        if (start_h_indata >= 0 && start_h_indata + 4 <= inputHeight && start_w_indata >= 0 &&                         \
            start_w_indata + 4 <= inputWidth) {                                                                        \
            int rowBase = inputBase + inputWidth * start_h_indata + start_w_indata;                                    \
            DATA_T4 row0 = vload4(0, input + rowBase);                                                                 \
            rowBase = rowBase + inputWidth;                                                                            \
            DATA_T4 row1 = vload4(0, input + rowBase);                                                                 \
            rowBase = rowBase + inputWidth;                                                                            \
            DATA_T4 row2 = vload4(0, input + rowBase);                                                                 \
            rowBase = rowBase + inputWidth;                                                                            \
            DATA_T4 row3 = vload4(0, input + rowBase);                                                                 \
            res.s0 = row0.s0 - row0.s2 - row2.s0 + row2.s2;                                                            \
            res.s1 = row0.s1 + row0.s2 - row2.s1 - row2.s2;                                                            \
            res.s2 = row0.s2 - row0.s1 + row2.s1 - row2.s2;                                                            \
            res.s3 = row0.s1 - row0.s3 - row2.s1 + row2.s3;                                                            \
            res.s4 = row1.s0 - row1.s2 + row2.s0 - row2.s2;                                                            \
            res.s5 = row1.s1 + row1.s2 + row2.s1 + row2.s2;                                                            \
            res.s6 = row1.s2 - row1.s1 - row2.s1 + row2.s2;                                                            \
            res.s7 = row1.s1 - row1.s3 + row2.s1 - row2.s3;                                                            \
            res.s8 = row1.s2 - row1.s0 + row2.s0 - row2.s2;                                                            \
            res.s9 = row2.s1 - row1.s1 - row1.s2 + row2.s2;                                                            \
            res.sa = row1.s1 - row1.s2 - row2.s1 + row2.s2;                                                            \
            res.sb = row1.s3 - row1.s1 + row2.s1 - row2.s3;                                                            \
            res.sc = row1.s0 - row1.s2 - row3.s0 + row3.s2;                                                            \
            res.sd = row1.s1 + row1.s2 - row3.s1 - row3.s2;                                                            \
            res.se = row1.s2 - row1.s1 + row3.s1 - row3.s2;                                                            \
            res.sf = row1.s1 - row1.s3 - row3.s1 + row3.s3;                                                            \
        } else {                                                                                                       \
            DATA_T A[16] = {(0.0f)};                                                                                   \
            char index = 0;                                                                                            \
            for (int i = start_h_indata; i < start_h_indata + 4; i++)                                                  \
                for (int j = start_w_indata; j < start_w_indata + 4; j++) {                                            \
                    if (i >= 0 && i < inputHeight && j >= 0 && j < inputWidth) {                                       \
                        A[index] = input[inputBase + inputWidth * i + j];                                              \
                    }                                                                                                  \
                    index++;                                                                                           \
                }                                                                                                      \
            res.s0 = A[0] - A[2] - A[8] + A[10];                                                                       \
            res.s1 = A[1] + A[2] - A[9] - A[10];                                                                       \
            res.s2 = A[2] - A[1] + A[9] - A[10];                                                                       \
            res.s3 = A[1] - A[3] - A[9] + A[11];                                                                       \
            res.s4 = A[4] - A[6] + A[8] - A[10];                                                                       \
            res.s5 = A[5] + A[6] + A[9] + A[10];                                                                       \
            res.s6 = A[6] - A[5] - A[9] + A[10];                                                                       \
            res.s7 = A[5] - A[7] + A[9] - A[11];                                                                       \
            res.s8 = A[6] - A[4] + A[8] - A[10];                                                                       \
            res.s9 = A[9] - A[5] - A[6] + A[10];                                                                       \
            res.sa = A[5] - A[6] - A[9] + A[10];                                                                       \
            res.sb = A[7] - A[5] + A[9] - A[11];                                                                       \
            res.sc = A[4] - A[6] - A[12] + A[14];                                                                      \
            res.sd = A[5] + A[6] - A[13] - A[14];                                                                      \
            res.se = A[6] - A[5] + A[13] - A[14];                                                                      \
            res.sf = A[5] - A[7] - A[13] + A[15];                                                                      \
        }                                                                                                              \
        vstore4(res.lo.lo, 0, output + out_index);                                                                     \
        vstore4(res.lo.hi, 0, output + out_index + 4 * chan);                                                          \
        vstore4(res.hi.lo, 0, output + out_index + 8 * chan);                                                          \
        vstore4(res.hi.hi, 0, output + out_index + 12 * chan);                                                         \
    }

#define FAST3X3_1(                                                                                                     \
    input, output, padHeight, padWidth, inputHeight, inputWidth, groupIndex, chan, groupNum, height, width)            \
    int gID0 = get_global_id(0);                                                                                       \
    int gID1 = get_global_id(1);                                                                                       \
    int gID2 = get_global_id(2);                                                                                       \
    if (gID1 < height && gID2 < width) {                                                                               \
        int picOff = gID0 / chan;                                                                                      \
        int channel = gID0 % chan;                                                                                     \
        int start_h_indata = gID1 * 2 - padHeight;                                                                     \
        int start_w_indata = gID2 * 2 - padWidth;                                                                      \
        int out_index = picOff * chan * height * width * 16 * groupNum + chan * height * width * 16 * groupIndex +     \
                        chan * 16 * width * gID1 + chan * 16 * gID2 + channel * 16;                                    \
        int inputBase = picOff * chan * groupNum * inputHeight * inputWidth +                                          \
                        (chan * groupIndex + channel) * inputHeight * inputWidth;                                      \
        if (start_h_indata >= 0 && start_h_indata + 4 <= inputHeight && start_w_indata >= 0 &&                         \
            start_w_indata + 4 <= inputWidth) {                                                                        \
            int rowBase = inputBase + inputWidth * start_h_indata + start_w_indata;                                    \
            DATA_T4 row0 = vload4(0, input + rowBase);                                                                 \
            rowBase = rowBase + inputWidth;                                                                            \
            DATA_T4 row1 = vload4(0, input + rowBase);                                                                 \
            rowBase = rowBase + inputWidth;                                                                            \
            DATA_T4 row2 = vload4(0, input + rowBase);                                                                 \
            rowBase = rowBase + inputWidth;                                                                            \
            DATA_T4 row3 = vload4(0, input + rowBase);                                                                 \
            DATA_T16 res;                                                                                              \
            res.s0 = row0.s0 - row0.s2 - row2.s0 + row2.s2;                                                            \
            res.s1 = row0.s1 + row0.s2 - row2.s1 - row2.s2;                                                            \
            res.s2 = row0.s2 - row0.s1 + row2.s1 - row2.s2;                                                            \
            res.s3 = row0.s1 - row0.s3 - row2.s1 + row2.s3;                                                            \
            res.s4 = row1.s0 - row1.s2 + row2.s0 - row2.s2;                                                            \
            res.s5 = row1.s1 + row1.s2 + row2.s1 + row2.s2;                                                            \
            res.s6 = row1.s2 - row1.s1 - row2.s1 + row2.s2;                                                            \
            res.s7 = row1.s1 - row1.s3 + row2.s1 - row2.s3;                                                            \
            res.s8 = row1.s2 - row1.s0 + row2.s0 - row2.s2;                                                            \
            res.s9 = row2.s1 - row1.s1 - row1.s2 + row2.s2;                                                            \
            res.sa = row1.s1 - row1.s2 - row2.s1 + row2.s2;                                                            \
            res.sb = row1.s3 - row1.s1 + row2.s1 - row2.s3;                                                            \
            res.sc = row1.s0 - row1.s2 - row3.s0 + row3.s2;                                                            \
            res.sd = row1.s1 + row1.s2 - row3.s1 - row3.s2;                                                            \
            res.se = row1.s2 - row1.s1 + row3.s1 - row3.s2;                                                            \
            res.sf = row1.s1 - row1.s3 - row3.s1 + row3.s3;                                                            \
            vstore16(res, 0, output + out_index);                                                                      \
        } else {                                                                                                       \
            DATA_T A[16] = {(0.0f)};                                                                                   \
            char index = 0;                                                                                            \
            DATA_T16 res;                                                                                              \
            for (int i = start_h_indata; i < start_h_indata + 4; i++)                                                  \
                for (int j = start_w_indata; j < start_w_indata + 4; j++) {                                            \
                    if (i >= 0 && i < inputHeight && j >= 0 && j < inputWidth) {                                       \
                        A[index] = input[inputBase + inputWidth * i + j];                                              \
                    }                                                                                                  \
                    index++;                                                                                           \
                }                                                                                                      \
            res.s0 = A[0] - A[2] - A[8] + A[10];                                                                       \
            res.s1 = A[1] + A[2] - A[9] - A[10];                                                                       \
            res.s2 = A[2] - A[1] + A[9] - A[10];                                                                       \
            res.s3 = A[1] - A[3] - A[9] + A[11];                                                                       \
            res.s4 = A[4] - A[6] + A[8] - A[10];                                                                       \
            res.s5 = A[5] + A[6] + A[9] + A[10];                                                                       \
            res.s6 = A[6] - A[5] - A[9] + A[10];                                                                       \
            res.s7 = A[5] - A[7] + A[9] - A[11];                                                                       \
            res.s8 = A[6] - A[4] + A[8] - A[10];                                                                       \
            res.s9 = A[9] - A[5] - A[6] + A[10];                                                                       \
            res.sa = A[5] - A[6] - A[9] + A[10];                                                                       \
            res.sb = A[7] - A[5] + A[9] - A[11];                                                                       \
            res.sc = A[4] - A[6] - A[12] + A[14];                                                                      \
            res.sd = A[5] + A[6] - A[13] - A[14];                                                                      \
            res.se = A[6] - A[5] + A[13] - A[14];                                                                      \
            res.sf = A[5] - A[7] - A[13] + A[15];                                                                      \
            vstore16(res, 0, output + out_index);                                                                      \
        }                                                                                                              \
    }

#define FAST3X3_2(                                                                                                     \
    input, weight, bias, output, inputDataNum_per_out2x2, ouputHeight, ouputWidth, groupIndex, chan, groupNum)         \
    int channel = get_global_id(0) % chan;                                                                             \
    int picOff = get_global_id(0) / chan;                                                                              \
    int input_index = inputDataNum_per_out2x2 * get_global_size(1) * get_global_size(2) * groupIndex +                 \
                      inputDataNum_per_out2x2 * (get_global_size(2) * get_global_id(1) + get_global_id(2)) +           \
                      picOff * inputDataNum_per_out2x2 * get_global_size(1) * get_global_size(2) * groupNum;           \
    int weight_index = inputDataNum_per_out2x2 * chan * groupIndex + inputDataNum_per_out2x2 * channel;                \
    DATA_T16 out16 = (DATA_T16)(0.0f);                                                                                 \
    DATA_T16 input16 = (DATA_T16)(0.0f);                                                                               \
    DATA_T16 weight16 = (DATA_T16)(0.0f);                                                                              \
    for (int i = 0; i < (inputDataNum_per_out2x2 >> 4); i++) {                                                         \
        input16 = vload16(0, input + input_index + (i << 4));                                                          \
        weight16 = vload16(0, weight + weight_index + (i << 4));                                                       \
        out16 += input16 * weight16;                                                                                   \
    }                                                                                                                  \
    int group_out_start =                                                                                              \
        ouputHeight * ouputWidth * chan * groupIndex + picOff * ouputHeight * ouputWidth * chan * groupNum;            \
    output[group_out_start + channel * ouputHeight * ouputWidth + get_global_id(1) * 2 * ouputWidth +                  \
           get_global_id(2) * 2] = ACT_VEC_F(DATA_T,                                                                   \
                                             out16.s0 + out16.s1 + out16.s2 + out16.s4 + out16.s5 + out16.s6 +         \
                                                 out16.s8 + out16.s9 + out16.sA + bias[channel + chan * groupIndex]);  \
    if (get_global_id(2) * 2 + 1 < ouputWidth)                                                                         \
        output[group_out_start + channel * ouputHeight * ouputWidth + get_global_id(1) * 2 * ouputWidth +              \
               get_global_id(2) * 2 + 1] =                                                                             \
            ACT_VEC_F(DATA_T,                                                                                          \
                      out16.s1 - out16.s2 - out16.s3 + out16.s5 - out16.s6 - out16.s7 + out16.s9 - out16.sA -          \
                          out16.sB + bias[channel + chan * groupIndex]);                                               \
    if (get_global_id(1) * 2 + 1 < ouputHeight)                                                                        \
        output[group_out_start + channel * ouputHeight * ouputWidth + (get_global_id(1) * 2 + 1) * ouputWidth +        \
               get_global_id(2) * 2] =                                                                                 \
            ACT_VEC_F(DATA_T,                                                                                          \
                      out16.s4 + out16.s5 + out16.s6 - out16.s8 - out16.s9 - out16.sA - out16.sC - out16.sD -          \
                          out16.sE + bias[channel + chan * groupIndex]);                                               \
    if (get_global_id(2) * 2 + 1 < ouputWidth && get_global_id(1) * 2 + 1 < ouputHeight)                               \
        output[group_out_start + channel * ouputHeight * ouputWidth + (get_global_id(1) * 2 + 1) * ouputWidth +        \
               get_global_id(2) * 2 + 1] =                                                                             \
            ACT_VEC_F(DATA_T,                                                                                          \
                      out16.s5 - out16.s6 - out16.s7 - out16.s9 + out16.sA + out16.sB - out16.sD + out16.sE +          \
                          out16.sF + bias[channel + chan * groupIndex]);

#define FAST3X3_2_OPTIMIZED(input,                                                                                     \
                            weight,                                                                                    \
                            bias,                                                                                      \
                            output,                                                                                    \
                            inputDataNum_per_out2x2,                                                                   \
                            ouputHeight,                                                                               \
                            ouputWidth,                                                                                \
                            channel,                                                                                   \
                            wgradTileHeight,                                                                           \
                            wgradTileWidth,                                                                            \
                            groupNum,                                                                                  \
                            groupIndex)                                                                                \
    if (get_global_id(1) < wgradTileHeight && get_global_id(2) < wgradTileWidth) {                                     \
        int channelIndex = (get_global_id(0) * 2) % channel;                                                           \
        int inputIndex =                                                                                               \
            inputDataNum_per_out2x2 * wgradTileHeight * wgradTileWidth * groupIndex +                                  \
            inputDataNum_per_out2x2 * (wgradTileWidth * get_global_id(1) + get_global_id(2)) +                         \
            (get_global_id(0) * 2) / channel * inputDataNum_per_out2x2 * wgradTileHeight * wgradTileWidth * groupNum;  \
        int weightIndex = inputDataNum_per_out2x2 * channel * groupIndex + inputDataNum_per_out2x2 * channelIndex;     \
        DATA_T16 out16 = (DATA_T16)(0.0f);                                                                             \
        DATA_T16 out16_2 = (DATA_T16)(0.0f);                                                                           \
        for (int i = 0; i<inputDataNum_per_out2x2>> 4; i++) {                                                         \
            DATA_T8 input8;                                                                                            \
            DATA_T8 weight8;                                                                                           \
            input8 = vload8(0, input + inputIndex);                                                                    \
            weight8 = vload8(0, weight + weightIndex);                                                                 \
            out16.lo += input8 * weight8;                                                                              \
            weight8 = vload8(0, weight + weightIndex + 16);                                                            \
            out16_2.lo += input8 * weight8;                                                                            \
            input8 = vload8(0, input + inputIndex + 8);                                                                \
            weight8 = vload8(0, weight + weightIndex + 8);                                                             \
            out16.hi += input8 * weight8;                                                                              \
            weight8 = vload8(0, weight + weightIndex + 24);                                                            \
            out16_2.hi += input8 * weight8;                                                                            \
            inputIndex = inputIndex + 16;                                                                              \
            weightIndex = weightIndex + 32;                                                                            \
        }                                                                                                              \
        DATA_T2 bia = vload2(0, bias + channelIndex + channel * groupIndex);                                           \
        channelIndex = channelIndex + (get_global_id(0) * 2) / channel * channel * groupNum + channel * groupIndex;    \
        if (get_global_id(2) * 2 + 1 < ouputWidth && get_global_id(1) * 2 + 1 < ouputHeight) {                         \
            output[channelIndex * ouputHeight * ouputWidth + get_global_id(1) * 2 * ouputWidth +                       \
                   get_global_id(2) * 2] = ACT_VEC_F(DATA_T,                                                           \
                                                     out16.s0 + out16.s1 + out16.s2 + out16.s4 + out16.s5 + out16.s6 + \
                                                         out16.s8 + out16.s9 + out16.sA + bia.s0);                     \
            output[channelIndex * ouputHeight * ouputWidth + get_global_id(1) * 2 * ouputWidth +                       \
                   get_global_id(2) * 2 + 1] = ACT_VEC_F(DATA_T,                                                       \
                                                         out16.s1 - out16.s2 - out16.s3 + out16.s5 - out16.s6 -        \
                                                             out16.s7 + out16.s9 - out16.sA - out16.sB + bia.s0);      \
            output[(channelIndex + 1) * ouputHeight * ouputWidth + get_global_id(1) * 2 * ouputWidth +                 \
                   get_global_id(2) * 2] = ACT_VEC_F(DATA_T,                                                           \
                                                     out16_2.s0 + out16_2.s1 + out16_2.s2 + out16_2.s4 + out16_2.s5 +  \
                                                         out16_2.s6 + out16_2.s8 + out16_2.s9 + out16_2.sA + bia.s1);  \
            output[(channelIndex + 1) * ouputHeight * ouputWidth + get_global_id(1) * 2 * ouputWidth +                 \
                   get_global_id(2) * 2 + 1] =                                                                         \
                ACT_VEC_F(DATA_T,                                                                                      \
                          out16_2.s1 - out16_2.s2 - out16_2.s3 + out16_2.s5 - out16_2.s6 - out16_2.s7 + out16_2.s9 -   \
                              out16_2.sA - out16_2.sB + bia.s1);                                                       \
            output[channelIndex * ouputHeight * ouputWidth + (get_global_id(1) * 2 + 1) * ouputWidth +                 \
                   get_global_id(2) * 2] = ACT_VEC_F(DATA_T,                                                           \
                                                     out16.s4 + out16.s5 + out16.s6 - out16.s8 - out16.s9 - out16.sA - \
                                                         out16.sC - out16.sD - out16.sE + bia.s0);                     \
            output[channelIndex * ouputHeight * ouputWidth + (get_global_id(1) * 2 + 1) * ouputWidth +                 \
                   get_global_id(2) * 2 + 1] = ACT_VEC_F(DATA_T,                                                       \
                                                         out16.s5 - out16.s6 - out16.s7 - out16.s9 + out16.sA +        \
                                                             out16.sB - out16.sD + out16.sE + out16.sF + bia.s0);      \
            output[(channelIndex + 1) * ouputHeight * ouputWidth + (get_global_id(1) * 2 + 1) * ouputWidth +           \
                   get_global_id(2) * 2] = ACT_VEC_F(DATA_T,                                                           \
                                                     out16_2.s4 + out16_2.s5 + out16_2.s6 - out16_2.s8 - out16_2.s9 -  \
                                                         out16_2.sA - out16_2.sC - out16_2.sD - out16_2.sE + bia.s1);  \
            output[(channelIndex + 1) * ouputHeight * ouputWidth + (get_global_id(1) * 2 + 1) * ouputWidth +           \
                   get_global_id(2) * 2 + 1] =                                                                         \
                ACT_VEC_F(DATA_T,                                                                                      \
                          out16_2.s5 - out16_2.s6 - out16_2.s7 - out16_2.s9 + out16_2.sA + out16_2.sB - out16_2.sD +   \
                              out16_2.sE + out16_2.sF + bia.s1);                                                       \
        } else if (get_global_id(1) * 2 + 1 < ouputHeight) {                                                           \
            output[channelIndex * ouputHeight * ouputWidth + get_global_id(1) * 2 * ouputWidth +                       \
                   get_global_id(2) * 2] = ACT_VEC_F(DATA_T,                                                           \
                                                     out16.s0 + out16.s1 + out16.s2 + out16.s4 + out16.s5 + out16.s6 + \
                                                         out16.s8 + out16.s9 + out16.sA + bia.s0);                     \
            output[(channelIndex + 1) * ouputHeight * ouputWidth + get_global_id(1) * 2 * ouputWidth +                 \
                   get_global_id(2) * 2] = ACT_VEC_F(DATA_T,                                                           \
                                                     out16_2.s0 + out16_2.s1 + out16_2.s2 + out16_2.s4 + out16_2.s5 +  \
                                                         out16_2.s6 + out16_2.s8 + out16_2.s9 + out16_2.sA + bia.s1);  \
            output[channelIndex * ouputHeight * ouputWidth + (get_global_id(1) * 2 + 1) * ouputWidth +                 \
                   get_global_id(2) * 2] = ACT_VEC_F(DATA_T,                                                           \
                                                     out16.s4 + out16.s5 + out16.s6 - out16.s8 - out16.s9 - out16.sA - \
                                                         out16.sC - out16.sD - out16.sE + bia.s0);                     \
            output[(channelIndex + 1) * ouputHeight * ouputWidth + (get_global_id(1) * 2 + 1) * ouputWidth +           \
                   get_global_id(2) * 2] = ACT_VEC_F(DATA_T,                                                           \
                                                     out16_2.s4 + out16_2.s5 + out16_2.s6 - out16_2.s8 - out16_2.s9 -  \
                                                         out16_2.sA - out16_2.sC - out16_2.sD - out16_2.sE + bia.s1);  \
        } else if (get_global_id(2) * 2 + 1 < ouputWidth) {                                                            \
            output[channelIndex * ouputHeight * ouputWidth + get_global_id(1) * 2 * ouputWidth +                       \
                   get_global_id(2) * 2] = ACT_VEC_F(DATA_T,                                                           \
                                                     out16.s0 + out16.s1 + out16.s2 + out16.s4 + out16.s5 + out16.s6 + \
                                                         out16.s8 + out16.s9 + out16.sA + bia.s0);                     \
            output[channelIndex * ouputHeight * ouputWidth + get_global_id(1) * 2 * ouputWidth +                       \
                   get_global_id(2) * 2 + 1] = ACT_VEC_F(DATA_T,                                                       \
                                                         out16.s1 - out16.s2 - out16.s3 + out16.s5 - out16.s6 -        \
                                                             out16.s7 + out16.s9 - out16.sA - out16.sB + bia.s0);      \
            output[(channelIndex + 1) * ouputHeight * ouputWidth + get_global_id(1) * 2 * ouputWidth +                 \
                   get_global_id(2) * 2] = ACT_VEC_F(DATA_T,                                                           \
                                                     out16_2.s0 + out16_2.s1 + out16_2.s2 + out16_2.s4 + out16_2.s5 +  \
                                                         out16_2.s6 + out16_2.s8 + out16_2.s9 + out16_2.sA + bia.s1);  \
            output[(channelIndex + 1) * ouputHeight * ouputWidth + get_global_id(1) * 2 * ouputWidth +                 \
                   get_global_id(2) * 2 + 1] =                                                                         \
                ACT_VEC_F(DATA_T,                                                                                      \
                          out16_2.s1 - out16_2.s2 - out16_2.s3 + out16_2.s5 - out16_2.s6 - out16_2.s7 + out16_2.s9 -   \
                              out16_2.sA - out16_2.sB + bia.s1);                                                       \
        } else {                                                                                                       \
            output[channelIndex * ouputHeight * ouputWidth + get_global_id(1) * 2 * ouputWidth +                       \
                   get_global_id(2) * 2] = ACT_VEC_F(DATA_T,                                                           \
                                                     out16.s0 + out16.s1 + out16.s2 + out16.s4 + out16.s5 + out16.s6 + \
                                                         out16.s8 + out16.s9 + out16.sA + bia.s0);                     \
            output[(channelIndex + 1) * ouputHeight * ouputWidth + get_global_id(1) * 2 * ouputWidth +                 \
                   get_global_id(2) * 2] = ACT_VEC_F(DATA_T,                                                           \
                                                     out16_2.s0 + out16_2.s1 + out16_2.s2 + out16_2.s4 + out16_2.s5 +  \
                                                         out16_2.s6 + out16_2.s8 + out16_2.s9 + out16_2.sA + bia.s1);  \
        }                                                                                                              \
    }

#define FAST3X3_2_OPTIMIZED_1X8_SPLITE_FP16(input,                                                                     \
                                            weight,                                                                    \
                                            bias,                                                                      \
                                            output,                                                                    \
                                            inputDataNum_per_out2x2,                                                   \
                                            ouputHeight,                                                               \
                                            ouputWidth,                                                                \
                                            channel,                                                                   \
                                            wgradTileHeight,                                                           \
                                            wgradTileWidth,                                                            \
                                            coalescingRowSize)                                                         \
    if (get_global_id(2) < wgradTileHeight * wgradTileWidth) {                                                         \
        DATA_T8 out_0;                                                                                                 \
        DATA_T8 out_1;                                                                                                 \
        DATA_T8 out_2;                                                                                                 \
        DATA_T8 out_3;                                                                                                 \
        {                                                                                                              \
            DATA_T8 out8_0 = 0.0f;                                                                                     \
            DATA_T8 out8_1 = 0.0f;                                                                                     \
            DATA_T8 out8_2 = 0.0f;                                                                                     \
            DATA_T8 out8_3 = 0.0f;                                                                                     \
            DATA_T8 out8_4 = 0.0f;                                                                                     \
            DATA_T8 out8_5 = 0.0f;                                                                                     \
            DATA_T8 out8_6 = 0.0f;                                                                                     \
            DATA_T8 out8_7 = 0.0f;                                                                                     \
            int2 srcIndex =                                                                                            \
                (int2)(get_global_id(2) / coalescingRowSize * coalescingRowSize * inputDataNum_per_out2x2 +            \
                           get_global_id(2) % coalescingRowSize * 8 +                                                  \
                           get_global_id(0) * 8 / channel * inputDataNum_per_out2x2 *                                  \
                               ((wgradTileHeight * wgradTileWidth + coalescingRowSize - 1) / coalescingRowSize *       \
                                coalescingRowSize) +                                                                   \
                           get_global_id(1) * coalescingRowSize * 8 * inputDataNum_per_out2x2 / 16,                    \
                       inputDataNum_per_out2x2 * ((get_global_id(0) * 8) % channel) +                                  \
                           get_global_id(1) * 64 * inputDataNum_per_out2x2 / 16);                                      \
            int coalescingSize = coalescingRowSize * 8;                                                                \
            for (int i = 0; i<inputDataNum_per_out2x2>> 4; i++) {                                                     \
                DATA_T8 input8;                                                                                        \
                DATA_T8 weight8;                                                                                       \
                input8 = vload8(0, input + srcIndex.s0);                                                               \
                weight8 = vload8(0, weight + srcIndex.s1);                                                             \
                out8_0 += input8.s0 * weight8;                                                                         \
                weight8 = vload8(0, weight + srcIndex.s1 + 8);                                                         \
                out8_1 += input8.s1 * weight8;                                                                         \
                weight8 = vload8(0, weight + srcIndex.s1 + 16);                                                        \
                out8_2 += input8.s2 * weight8;                                                                         \
                weight8 = vload8(0, weight + srcIndex.s1 + 24);                                                        \
                out8_3 += input8.s3 * weight8;                                                                         \
                weight8 = vload8(0, weight + srcIndex.s1 + 32);                                                        \
                out8_4 += input8.s4 * weight8;                                                                         \
                weight8 = vload8(0, weight + srcIndex.s1 + 40);                                                        \
                out8_5 += input8.s5 * weight8;                                                                         \
                weight8 = vload8(0, weight + srcIndex.s1 + 48);                                                        \
                out8_6 += input8.s6 * weight8;                                                                         \
                weight8 = vload8(0, weight + srcIndex.s1 + 56);                                                        \
                out8_7 += input8.s7 * weight8;                                                                         \
                srcIndex += (int2)(coalescingSize, 64);                                                                \
            }                                                                                                          \
            if (get_global_id(1) == 0) {                                                                               \
                out_0 = out8_0 + out8_1 + out8_2 + out8_4 + out8_5 + out8_6;                                           \
                out_1 = out8_1 - out8_2 - out8_3 + out8_5 - out8_6 - out8_7;                                           \
                out_2 = out8_4 + out8_5 + out8_6;                                                                      \
                out_3 = out8_5 - out8_6 - out8_7;                                                                      \
            } else {                                                                                                   \
                out_0 = out8_0 + out8_1 + out8_2;                                                                      \
                out_1 = out8_1 - out8_2 - out8_3;                                                                      \
                out_2 = -out8_0 - out8_1 - out8_2 - out8_4 - out8_5 - out8_6;                                          \
                out_3 = -out8_1 + out8_2 + out8_3 - out8_5 + out8_6 + out8_7;                                          \
            }                                                                                                          \
        }                                                                                                              \
        {                                                                                                              \
            int dstIndex = get_global_id(0) * wgradTileHeight * wgradTileWidth * 8 * 2 * 4 +                           \
                           get_global_id(2) * 8 * 2 * 4 + get_global_id(1) * 8;                                        \
            vstore8(out_0, 0, output + dstIndex);                                                                      \
            vstore8(out_1, 0, output + dstIndex + 16);                                                                 \
            vstore8(out_2, 0, output + dstIndex + 32);                                                                 \
            vstore8(out_3, 0, output + dstIndex + 48);                                                                 \
        }                                                                                                              \
    }

#define FAST3X3_2_OPTIMIZED_1X8_MERGE_FP16(input,                                                                      \
                                           weight,                                                                     \
                                           bias,                                                                       \
                                           output,                                                                     \
                                           inputDataNum_per_out2x2,                                                    \
                                           ouputHeight,                                                                \
                                           ouputWidth,                                                                 \
                                           channel,                                                                    \
                                           wgradTileHeight,                                                            \
                                           wgradTileWidth)                                                             \
    if (get_global_id(2) < wgradTileHeight * wgradTileWidth) {                                                         \
        DATA_T8 bia = vload8(0, bias + get_global_id(1) * 8);                                                          \
        DATA_T8 out_0 = bia;                                                                                           \
        DATA_T8 out_1 = bia;                                                                                           \
        DATA_T8 out_2 = bia;                                                                                           \
        DATA_T8 out_3 = bia;                                                                                           \
        {                                                                                                              \
            int srcIndex = get_global_id(0) * channel / 8 * wgradTileHeight * wgradTileWidth * 8 * 2 * 4 +             \
                           get_global_id(1) * wgradTileHeight * wgradTileWidth * 8 * 2 * 4 +                           \
                           get_global_id(2) * 8 * 2 * 4;                                                               \
            DATA_T8 front8;                                                                                            \
            DATA_T8 end8;                                                                                              \
            front8 = vload8(0, input + srcIndex);                                                                      \
            end8 = vload8(0, input + srcIndex + 8);                                                                    \
            out_0 += front8 + end8;                                                                                    \
            front8 = vload8(0, input + srcIndex + 16);                                                                 \
            end8 = vload8(0, input + srcIndex + 24);                                                                   \
            out_1 += front8 + end8;                                                                                    \
            front8 = vload8(0, input + srcIndex + 32);                                                                 \
            end8 = vload8(0, input + srcIndex + 40);                                                                   \
            out_2 += front8 + end8;                                                                                    \
            front8 = vload8(0, input + srcIndex + 48);                                                                 \
            end8 = vload8(0, input + srcIndex + 56);                                                                   \
            out_3 += front8 + end8;                                                                                    \
        }                                                                                                              \
        {                                                                                                              \
            int dstIndex = (get_global_id(0) * channel + get_global_id(1) * 8) * ouputHeight * ouputWidth +            \
                           (get_global_id(2) / wgradTileWidth) * 2 * ouputWidth +                                      \
                           (get_global_id(2) % wgradTileWidth) * 2;                                                    \
            if ((get_global_id(2) % wgradTileWidth) * 2 + 1 < ouputWidth &&                                            \
                (get_global_id(2) / wgradTileWidth) * 2 + 1 < ouputHeight) {                                           \
                output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s0);                                                        \
                output[dstIndex + 1] = ACT_VEC_F(DATA_T, out_1.s0);                                                    \
                output[dstIndex + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s0);                                           \
                output[dstIndex + ouputWidth + 1] = ACT_VEC_F(DATA_T, out_3.s0);                                       \
                output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s1);                             \
                output[dstIndex + ouputHeight * ouputWidth + 1] = ACT_VEC_F(DATA_T, out_1.s1);                         \
                output[dstIndex + ouputHeight * ouputWidth + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s1);                \
                output[dstIndex + ouputHeight * ouputWidth + ouputWidth + 1] = ACT_VEC_F(DATA_T, out_3.s1);            \
                output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s2);                         \
                output[dstIndex + 2 * ouputHeight * ouputWidth + 1] = ACT_VEC_F(DATA_T, out_1.s2);                     \
                output[dstIndex + 2 * ouputHeight * ouputWidth + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s2);            \
                output[dstIndex + 2 * ouputHeight * ouputWidth + ouputWidth + 1] = ACT_VEC_F(DATA_T, out_3.s2);        \
                output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s3);                         \
                output[dstIndex + 3 * ouputHeight * ouputWidth + 1] = ACT_VEC_F(DATA_T, out_1.s3);                     \
                output[dstIndex + 3 * ouputHeight * ouputWidth + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s3);            \
                output[dstIndex + 3 * ouputHeight * ouputWidth + ouputWidth + 1] = ACT_VEC_F(DATA_T, out_3.s3);        \
                output[dstIndex + 4 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s4);                         \
                output[dstIndex + 4 * ouputHeight * ouputWidth + 1] = ACT_VEC_F(DATA_T, out_1.s4);                     \
                output[dstIndex + 4 * ouputHeight * ouputWidth + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s4);            \
                output[dstIndex + 4 * ouputHeight * ouputWidth + ouputWidth + 1] = ACT_VEC_F(DATA_T, out_3.s4);        \
                output[dstIndex + 5 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s5);                         \
                output[dstIndex + 5 * ouputHeight * ouputWidth + 1] = ACT_VEC_F(DATA_T, out_1.s5);                     \
                output[dstIndex + 5 * ouputHeight * ouputWidth + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s5);            \
                output[dstIndex + 5 * ouputHeight * ouputWidth + ouputWidth + 1] = ACT_VEC_F(DATA_T, out_3.s5);        \
                output[dstIndex + 6 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s6);                         \
                output[dstIndex + 6 * ouputHeight * ouputWidth + 1] = ACT_VEC_F(DATA_T, out_1.s6);                     \
                output[dstIndex + 6 * ouputHeight * ouputWidth + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s6);            \
                output[dstIndex + 6 * ouputHeight * ouputWidth + ouputWidth + 1] = ACT_VEC_F(DATA_T, out_3.s6);        \
                output[dstIndex + 7 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s7);                         \
                output[dstIndex + 7 * ouputHeight * ouputWidth + 1] = ACT_VEC_F(DATA_T, out_1.s7);                     \
                output[dstIndex + 7 * ouputHeight * ouputWidth + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s7);            \
                output[dstIndex + 7 * ouputHeight * ouputWidth + ouputWidth + 1] = ACT_VEC_F(DATA_T, out_3.s7);        \
            } else if ((get_global_id(2) / wgradTileWidth) * 2 + 1 < ouputHeight) {                                    \
                output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s0);                                                        \
                output[dstIndex + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s0);                                           \
                output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s1);                             \
                output[dstIndex + ouputHeight * ouputWidth + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s1);                \
                output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s2);                         \
                output[dstIndex + 2 * ouputHeight * ouputWidth + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s2);            \
                output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s3);                         \
                output[dstIndex + 3 * ouputHeight * ouputWidth + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s3);            \
                output[dstIndex + 4 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s4);                         \
                output[dstIndex + 4 * ouputHeight * ouputWidth + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s4);            \
                output[dstIndex + 5 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s5);                         \
                output[dstIndex + 5 * ouputHeight * ouputWidth + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s5);            \
                output[dstIndex + 6 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s6);                         \
                output[dstIndex + 6 * ouputHeight * ouputWidth + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s6);            \
                output[dstIndex + 7 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s7);                         \
                output[dstIndex + 7 * ouputHeight * ouputWidth + ouputWidth] = ACT_VEC_F(DATA_T, out_2.s7);            \
            } else if ((get_global_id(2) % wgradTileWidth) * 2 + 1 < ouputWidth) {                                     \
                output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s0);                                                        \
                output[dstIndex + 1] = ACT_VEC_F(DATA_T, out_1.s0);                                                    \
                output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s1);                             \
                output[dstIndex + ouputHeight * ouputWidth + 1] = ACT_VEC_F(DATA_T, out_1.s1);                         \
                output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s2);                         \
                output[dstIndex + 2 * ouputHeight * ouputWidth + 1] = ACT_VEC_F(DATA_T, out_1.s2);                     \
                output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s3);                         \
                output[dstIndex + 3 * ouputHeight * ouputWidth + 1] = ACT_VEC_F(DATA_T, out_1.s3);                     \
                output[dstIndex + 4 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s4);                         \
                output[dstIndex + 4 * ouputHeight * ouputWidth + 1] = ACT_VEC_F(DATA_T, out_1.s4);                     \
                output[dstIndex + 5 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s5);                         \
                output[dstIndex + 5 * ouputHeight * ouputWidth + 1] = ACT_VEC_F(DATA_T, out_1.s5);                     \
                output[dstIndex + 6 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s6);                         \
                output[dstIndex + 6 * ouputHeight * ouputWidth + 1] = ACT_VEC_F(DATA_T, out_1.s6);                     \
                output[dstIndex + 7 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s7);                         \
                output[dstIndex + 7 * ouputHeight * ouputWidth + 1] = ACT_VEC_F(DATA_T, out_1.s7);                     \
            } else {                                                                                                   \
                output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s0);                                                        \
                output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s1);                             \
                output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s2);                         \
                output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s3);                         \
                output[dstIndex + 4 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s4);                         \
                output[dstIndex + 5 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s5);                         \
                output[dstIndex + 6 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s6);                         \
                output[dstIndex + 7 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s7);                         \
            }                                                                                                          \
        }                                                                                                              \
    }

#define FAST3X3_2_OPTIMIZED_1X8_SPLITE_FP32(input,                                                                     \
                                            weight,                                                                    \
                                            bias,                                                                      \
                                            output,                                                                    \
                                            inputDataNum_per_out2x2,                                                   \
                                            ouputHeight,                                                               \
                                            ouputWidth,                                                                \
                                            channel,                                                                   \
                                            wgradTileHeight,                                                           \
                                            wgradTileWidth)                                                            \
    if (get_global_id(2) < wgradTileHeight * wgradTileWidth) {                                                         \
        DATA_T8 out8_0 = 0.0f;                                                                                         \
        DATA_T8 out8_1 = 0.0f;                                                                                         \
        DATA_T8 out8_2 = 0.0f;                                                                                         \
        DATA_T8 out8_3 = 0.0f;                                                                                         \
        {                                                                                                              \
            int2 srcIndex = (int2)(inputDataNum_per_out2x2 * get_global_id(2) +                                        \
                                       get_global_id(0) * 8 / channel * inputDataNum_per_out2x2 * wgradTileHeight *    \
                                           wgradTileWidth +                                                            \
                                       get_global_id(1) * 4 * inputDataNum_per_out2x2 / 16,                            \
                                   inputDataNum_per_out2x2 * ((get_global_id(0) * 8) % channel) +                      \
                                       get_global_id(1) * 32 * inputDataNum_per_out2x2 / 16);                          \
            for (int i = 0; i<inputDataNum_per_out2x2>> 4; i++) {                                                     \
                DATA_T4 input4;                                                                                        \
                DATA_T8 weight8;                                                                                       \
                input4 = vload4(0, input + srcIndex.s0);                                                               \
                weight8 = vload8(0, weight + srcIndex.s1);                                                             \
                out8_0 += input4.s0 * weight8;                                                                         \
                weight8 = vload8(0, weight + srcIndex.s1 + 8);                                                         \
                out8_1 += input4.s1 * weight8;                                                                         \
                weight8 = vload8(0, weight + srcIndex.s1 + 16);                                                        \
                out8_2 += input4.s2 * weight8;                                                                         \
                weight8 = vload8(0, weight + srcIndex.s1 + 24);                                                        \
                out8_3 += input4.s3 * weight8;                                                                         \
                srcIndex += (int2)(4, 32);                                                                             \
            }                                                                                                          \
            if (get_global_id(1) == 0) {                                                                               \
                out8_0 = out8_0 + out8_1 + out8_2;                                                                     \
                out8_1 = out8_1 - out8_2 - out8_3;                                                                     \
                out8_2 = 0.0f;                                                                                         \
                out8_3 = 0.0f;                                                                                         \
            } else if (get_global_id(1) == 1) {                                                                        \
                out8_0 = out8_0 + out8_1 + out8_2;                                                                     \
                out8_1 = out8_1 - out8_2 - out8_3;                                                                     \
                out8_2 = out8_0;                                                                                       \
                out8_3 = out8_1;                                                                                       \
            } else if (get_global_id(1) == 2) {                                                                        \
                out8_0 = out8_0 + out8_1 + out8_2;                                                                     \
                out8_1 = out8_1 - out8_2 - out8_3;                                                                     \
                out8_2 = -out8_0;                                                                                      \
                out8_3 = -out8_1;                                                                                      \
            } else {                                                                                                   \
                out8_0 = -out8_0 - out8_1 - out8_2;                                                                    \
                out8_1 = -out8_1 + out8_2 + out8_3;                                                                    \
                out8_2 = out8_0;                                                                                       \
                out8_3 = out8_1;                                                                                       \
                out8_0 = 0.0f;                                                                                         \
                out8_1 = 0.0f;                                                                                         \
            }                                                                                                          \
        }                                                                                                              \
        {                                                                                                              \
            int dstIndex = get_global_id(0) * wgradTileHeight * wgradTileWidth * 8 * 4 * 4 +                           \
                           get_global_id(2) * 8 * 4 * 4 + get_global_id(1) * 8;                                        \
            vstore8(out8_0, 0, output + dstIndex);                                                                     \
            vstore8(out8_1, 0, output + dstIndex + 32);                                                                \
            vstore8(out8_2, 0, output + dstIndex + 64);                                                                \
            vstore8(out8_3, 0, output + dstIndex + 96);                                                                \
        }                                                                                                              \
    }

#define FAST3X3_2_OPTIMIZED_1X8_MERGE_FP32(input,                                                                      \
                                           weight,                                                                     \
                                           bias,                                                                       \
                                           output,                                                                     \
                                           inputDataNum_per_out2x2,                                                    \
                                           ouputHeight,                                                                \
                                           ouputWidth,                                                                 \
                                           channel,                                                                    \
                                           wgradTileHeight,                                                            \
                                           wgradTileWidth)                                                             \
    if (get_global_id(2) < wgradTileHeight * wgradTileWidth) {                                                         \
        DATA_T8 bia = vload8(0, bias + (get_global_id(0) * 8) % channel);                                              \
        DATA_T8 out_0 = bia;                                                                                           \
        {                                                                                                              \
            int srcIndex =                                                                                             \
                (get_global_id(0) * 8) / channel * channel / 8 * wgradTileHeight * wgradTileWidth * 8 * 4 * 4 +        \
                (get_global_id(0) * 8) % channel * wgradTileHeight * wgradTileWidth * 4 * 4 +                          \
                get_global_id(2) * 8 * 4 * 4 + get_global_id(1) * 2 * 4 * 4;                                           \
            DATA_T8 front8;                                                                                            \
            DATA_T8 end8;                                                                                              \
            front8 = vload8(0, input + srcIndex);                                                                      \
            end8 = vload8(0, input + srcIndex + 8);                                                                    \
            out_0 += front8 + end8;                                                                                    \
            front8 = vload8(0, input + srcIndex + 16);                                                                 \
            end8 = vload8(0, input + srcIndex + 24);                                                                   \
            out_0 += front8 + end8;                                                                                    \
        }                                                                                                              \
        {                                                                                                              \
            int dstIndex = ((get_global_id(0) * 8) / channel * channel + (get_global_id(0) * 8) % channel) *           \
                               ouputHeight * ouputWidth +                                                              \
                           (get_global_id(2) / wgradTileWidth) * 2 * ouputWidth +                                      \
                           (get_global_id(2) % wgradTileWidth) * 2 + get_global_id(1) / 2 * ouputWidth +               \
                           get_global_id(1) % 2;                                                                       \
            if ((get_global_id(2) % wgradTileWidth) * 2 + 1 < ouputWidth &&                                            \
                (get_global_id(2) / wgradTileWidth) * 2 + 1 < ouputHeight) {                                           \
                output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s0);                                                        \
                output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s1);                             \
                output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s2);                         \
                output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s3);                         \
                output[dstIndex + 4 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s4);                         \
                output[dstIndex + 5 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s5);                         \
                output[dstIndex + 6 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s6);                         \
                output[dstIndex + 7 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s7);                         \
            } else if ((get_global_id(2) / wgradTileWidth) * 2 + 1 < ouputHeight) {                                    \
                if (get_global_id(1) == 0 || get_global_id(1) == 2) {                                                  \
                    output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s0);                                                    \
                    output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s1);                         \
                    output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s2);                     \
                    output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s3);                     \
                    output[dstIndex + 4 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s4);                     \
                    output[dstIndex + 5 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s5);                     \
                    output[dstIndex + 6 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s6);                     \
                    output[dstIndex + 7 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s7);                     \
                }                                                                                                      \
            } else if ((get_global_id(2) % wgradTileWidth) * 2 + 1 < ouputWidth) {                                     \
                if (get_global_id(1) == 0 || get_global_id(1) == 1) {                                                  \
                    output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s0);                                                    \
                    output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s1);                         \
                    output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s2);                     \
                    output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s3);                     \
                    output[dstIndex + 4 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s4);                     \
                    output[dstIndex + 5 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s5);                     \
                    output[dstIndex + 6 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s6);                     \
                    output[dstIndex + 7 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s7);                     \
                }                                                                                                      \
            } else {                                                                                                   \
                if (get_global_id(1) == 0) {                                                                           \
                    output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s0);                                                    \
                    output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s1);                         \
                    output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s2);                     \
                    output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s3);                     \
                    output[dstIndex + 4 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s4);                     \
                    output[dstIndex + 5 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s5);                     \
                    output[dstIndex + 6 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s6);                     \
                    output[dstIndex + 7 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s7);                     \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }

#define FAST3X3_1_4X4(                                                                                                 \
    input, output, padHeight, padWidth, inputHeight, inputWidth, chan, height, width, coalescingRowSize)               \
    int picOff = get_global_id(2);                                                                                     \
    int channel = get_global_id(1);                                                                                    \
    int hwIndex = get_global_id(0);                                                                                    \
    if (hwIndex < height * width && channel < chan) {                                                                  \
        int gID1 = hwIndex / width;                                                                                    \
        int gID2 = hwIndex % width;                                                                                    \
        int start_h_indata = gID1 * 2 - padHeight;                                                                     \
        int start_w_indata = gID2 * 2 - padWidth;                                                                      \
        int row = gID1 * width + gID2;                                                                                 \
        int alignedHW = (height * width + coalescingRowSize - 1) / coalescingRowSize * coalescingRowSize;              \
        int out_index = picOff * chan * alignedHW * 16 + row / coalescingRowSize * coalescingRowSize * chan * 16 +     \
                        row % coalescingRowSize * 4 + channel * coalescingRowSize * 4 * 4;                             \
        int inputBase = picOff * chan * inputHeight * inputWidth + channel * inputHeight * inputWidth;                 \
        DATA_T16 res;                                                                                                  \
        DATA_T4 row0 = (DATA_T4)(0.0f);                                                                                \
        DATA_T4 row1 = (DATA_T4)(0.0f);                                                                                \
        DATA_T4 row2 = (DATA_T4)(0.0f);                                                                                \
        DATA_T4 row3 = (DATA_T4)(0.0f);                                                                                \
        int rowBase = inputBase + start_w_indata;                                                                      \
        if (start_h_indata >= 0 && start_h_indata < inputHeight) {                                                     \
            row0 = vload4(0, input + rowBase + inputWidth * start_h_indata);                                           \
        }                                                                                                              \
        if (start_h_indata + 1 >= 0 && start_h_indata + 1 < inputHeight) {                                             \
            row1 = vload4(0, input + rowBase + inputWidth * (start_h_indata + 1));                                     \
        }                                                                                                              \
        if (start_h_indata + 2 >= 0 && start_h_indata + 2 < inputHeight) {                                             \
            row2 = vload4(0, input + rowBase + inputWidth * (start_h_indata + 2));                                     \
        }                                                                                                              \
        if (start_h_indata + 3 >= 0 && start_h_indata + 3 < inputHeight) {                                             \
            row3 = vload4(0, input + rowBase + inputWidth * (start_h_indata + 3));                                     \
        }                                                                                                              \
        if (start_w_indata < 0 || start_w_indata >= inputWidth) {                                                      \
            row0.s0 = (DATA_T)(0.0f);                                                                                  \
            row1.s0 = (DATA_T)(0.0f);                                                                                  \
            row2.s0 = (DATA_T)(0.0f);                                                                                  \
            row3.s0 = (DATA_T)(0.0f);                                                                                  \
        }                                                                                                              \
        if (start_w_indata + 1 < 0 || start_w_indata + 1 >= inputWidth) {                                              \
            row0.s1 = (DATA_T)(0.0f);                                                                                  \
            row1.s1 = (DATA_T)(0.0f);                                                                                  \
            row2.s1 = (DATA_T)(0.0f);                                                                                  \
            row3.s1 = (DATA_T)(0.0f);                                                                                  \
        }                                                                                                              \
        if (start_w_indata + 2 < 0 || start_w_indata + 2 >= inputWidth) {                                              \
            row0.s2 = (DATA_T)(0.0f);                                                                                  \
            row1.s2 = (DATA_T)(0.0f);                                                                                  \
            row2.s2 = (DATA_T)(0.0f);                                                                                  \
            row3.s2 = (DATA_T)(0.0f);                                                                                  \
        }                                                                                                              \
        if (start_w_indata + 3 < 0 || start_w_indata + 3 >= inputWidth) {                                              \
            row0.s3 = (DATA_T)(0.0f);                                                                                  \
            row1.s3 = (DATA_T)(0.0f);                                                                                  \
            row2.s3 = (DATA_T)(0.0f);                                                                                  \
            row3.s3 = (DATA_T)(0.0f);                                                                                  \
        }                                                                                                              \
        res.s0 = row0.s0 - row0.s2 - row2.s0 + row2.s2;                                                                \
        res.s1 = row0.s1 + row0.s2 - row2.s1 - row2.s2;                                                                \
        res.s2 = row0.s2 - row0.s1 + row2.s1 - row2.s2;                                                                \
        res.s3 = row0.s1 - row0.s3 - row2.s1 + row2.s3;                                                                \
        res.s4 = row1.s0 - row1.s2 + row2.s0 - row2.s2;                                                                \
        res.s5 = row1.s1 + row1.s2 + row2.s1 + row2.s2;                                                                \
        res.s6 = row1.s2 - row1.s1 - row2.s1 + row2.s2;                                                                \
        res.s7 = row1.s1 - row1.s3 + row2.s1 - row2.s3;                                                                \
        res.s8 = row1.s2 - row1.s0 + row2.s0 - row2.s2;                                                                \
        res.s9 = row2.s1 - row1.s1 - row1.s2 + row2.s2;                                                                \
        res.sa = row1.s1 - row1.s2 - row2.s1 + row2.s2;                                                                \
        res.sb = row1.s3 - row1.s1 + row2.s1 - row2.s3;                                                                \
        res.sc = row1.s0 - row1.s2 - row3.s0 + row3.s2;                                                                \
        res.sd = row1.s1 + row1.s2 - row3.s1 - row3.s2;                                                                \
        res.se = row1.s2 - row1.s1 + row3.s1 - row3.s2;                                                                \
        res.sf = row1.s1 - row1.s3 - row3.s1 + row3.s3;                                                                \
        vstore4(res.lo.lo, 0, output + out_index);                                                                     \
        vstore4(res.lo.hi, 0, output + out_index + 4 * coalescingRowSize);                                             \
        vstore4(res.hi.lo, 0, output + out_index + 8 * coalescingRowSize);                                             \
        vstore4(res.hi.hi, 0, output + out_index + 12 * coalescingRowSize);                                            \
    }

#define FAST3X3_2_OPTIMIZED_4X4_SPLITE(input,                                                                          \
                                       weight,                                                                         \
                                       bias,                                                                           \
                                       output,                                                                         \
                                       inputDataNum_per_out2x2,                                                        \
                                       ouputHeight,                                                                    \
                                       ouputWidth,                                                                     \
                                       channel,                                                                        \
                                       wgradTileHeight,                                                                \
                                       wgradTileWidth,                                                                 \
                                       coalescingRowSize)                                                              \
    int GID0 = get_global_id(0) * 4;                                                                                   \
    int GID1 = get_global_id(1);                                                                                       \
    int GID2 = get_global_id(2) * 4;                                                                                   \
    if (GID2 < wgradTileHeight * wgradTileWidth) {                                                                     \
        DATA_T16 out16_0 = 0.0f;                                                                                       \
        DATA_T16 out16_1 = 0.0f;                                                                                       \
        DATA_T16 out16_2 = 0.0f;                                                                                       \
        DATA_T16 out16_3 = 0.0f;                                                                                       \
        {                                                                                                              \
            int2 srcIndex = (int2)((GID2 / coalescingRowSize) * coalescingRowSize * inputDataNum_per_out2x2 +          \
                                       GID2 % coalescingRowSize / 4 * 16 +                                             \
                                       (GID0 / channel) * inputDataNum_per_out2x2 *                                    \
                                           ((wgradTileHeight * wgradTileWidth + coalescingRowSize - 1) /               \
                                            coalescingRowSize * coalescingRowSize) +                                   \
                                       GID1 * coalescingRowSize * 4,                                                   \
                                   (GID0 % channel) / 8 * 8 * inputDataNum_per_out2x2 +                                \
                                       (GID0 % channel) % 8 / 4 * 16 + GID1 * 32 * (inputDataNum_per_out2x2 / 16));    \
            int coalescingSize = coalescingRowSize * 4 * 4;                                                            \
            for (int i = 0; i<inputDataNum_per_out2x2>> 4; i++) {                                                     \
                DATA_T16 input16;                                                                                      \
                DATA_T16 weight16;                                                                                     \
                input16 = vload16(0, input + srcIndex.s0);                                                             \
                weight16 = vload16(0, weight + srcIndex.s1);                                                           \
                out16_0.s0123 = mad(input16.s0, weight16.s0123, out16_0.s0123);                                        \
                out16_0.s4567 = mad(input16.s4, weight16.s0123, out16_0.s4567);                                        \
                out16_0.s89AB = mad(input16.s8, weight16.s0123, out16_0.s89AB);                                        \
                out16_0.sCDEF = mad(input16.sC, weight16.s0123, out16_0.sCDEF);                                        \
                out16_1.s0123 = mad(input16.s1, weight16.s4567, out16_1.s0123);                                        \
                out16_1.s4567 = mad(input16.s5, weight16.s4567, out16_1.s4567);                                        \
                out16_1.s89AB = mad(input16.s9, weight16.s4567, out16_1.s89AB);                                        \
                out16_1.sCDEF = mad(input16.sD, weight16.s4567, out16_1.sCDEF);                                        \
                out16_2.s0123 = mad(input16.s2, weight16.s89AB, out16_2.s0123);                                        \
                out16_2.s4567 = mad(input16.s6, weight16.s89AB, out16_2.s4567);                                        \
                out16_2.s89AB = mad(input16.sA, weight16.s89AB, out16_2.s89AB);                                        \
                out16_2.sCDEF = mad(input16.sE, weight16.s89AB, out16_2.sCDEF);                                        \
                out16_3.s0123 = mad(input16.s3, weight16.sCDEF, out16_3.s0123);                                        \
                out16_3.s4567 = mad(input16.s7, weight16.sCDEF, out16_3.s4567);                                        \
                out16_3.s89AB = mad(input16.sB, weight16.sCDEF, out16_3.s89AB);                                        \
                out16_3.sCDEF = mad(input16.sF, weight16.sCDEF, out16_3.sCDEF);                                        \
                srcIndex += (int2)(coalescingSize, 32);                                                                \
            }                                                                                                          \
            if (GID1 == 0) {                                                                                           \
                out16_0 = out16_0 + out16_1 + out16_2;                                                                 \
                out16_1 = out16_1 - out16_2 - out16_3;                                                                 \
            } else if (GID1 == 1) {                                                                                    \
                out16_0 = out16_0 + out16_1 + out16_2;                                                                 \
                out16_1 = out16_1 - out16_2 - out16_3;                                                                 \
            } else if (GID1 == 2) {                                                                                    \
                out16_0 = out16_0 + out16_1 + out16_2;                                                                 \
                out16_1 = out16_1 - out16_2 - out16_3;                                                                 \
            } else {                                                                                                   \
                out16_0 = -out16_0 - out16_1 - out16_2;                                                                \
                out16_1 = -out16_1 + out16_2 + out16_3;                                                                \
            }                                                                                                          \
        }                                                                                                              \
        {                                                                                                              \
            int dstIndex = (GID0 / channel) * (channel / 4) * ((wgradTileHeight * wgradTileWidth + 3) / 4) * 128 +     \
                           (GID0 % channel) * ((wgradTileHeight * wgradTileWidth + 3) / 4) * 32 + GID2 * 32 +          \
                           GID1 * 32;                                                                                  \
            vstore16(out16_0, 0, output + dstIndex);                                                                   \
            vstore16(out16_1, 1, output + dstIndex);                                                                   \
        }                                                                                                              \
    }

#define FAST3X3_1_MAKALU_1X2(                                                                                          \
    input, output, padHeight, padWidth, inputHeight, inputWidth, chan, height, width, coalescingRowSize)               \
    int picOff = get_global_id(0);                                                                                     \
    int channel = get_global_id(1);                                                                                    \
    int hwIndex = get_global_id(2);                                                                                    \
    if (hwIndex < height * width) {                                                                                    \
        int gID1 = hwIndex / width;                                                                                    \
        int gID2 = hwIndex % width;                                                                                    \
        int start_h_indata = gID1 * 2 - padHeight;                                                                     \
        int start_w_indata = gID2 * 2 - padWidth;                                                                      \
        int row = gID1 * width + gID2;                                                                                 \
        int alignedHW = (height * width + coalescingRowSize - 1) / coalescingRowSize * coalescingRowSize;              \
        int out_index = picOff * chan * alignedHW * 16 + row / coalescingRowSize * coalescingRowSize * chan * 16 +     \
                        row % coalescingRowSize * 16 + channel * coalescingRowSize * 16;                               \
        int inputBase = picOff * chan * inputHeight * inputWidth + channel * inputHeight * inputWidth;                 \
        DATA_T16 res;                                                                                                  \
        DATA_T4 row0 = (DATA_T4)(0.0f);                                                                                \
        DATA_T4 row1 = (DATA_T4)(0.0f);                                                                                \
        DATA_T4 row2 = (DATA_T4)(0.0f);                                                                                \
        DATA_T4 row3 = (DATA_T4)(0.0f);                                                                                \
        int rowBase = inputBase + start_w_indata;                                                                      \
        if (start_h_indata >= 0 && start_h_indata < inputHeight) {                                                     \
            row0 = vload4(0, input + rowBase + inputWidth * start_h_indata);                                           \
        }                                                                                                              \
        if (start_h_indata + 1 >= 0 && start_h_indata + 1 < inputHeight) {                                             \
            row1 = vload4(0, input + rowBase + inputWidth * (start_h_indata + 1));                                     \
        }                                                                                                              \
        if (start_h_indata + 2 >= 0 && start_h_indata + 2 < inputHeight) {                                             \
            row2 = vload4(0, input + rowBase + inputWidth * (start_h_indata + 2));                                     \
        }                                                                                                              \
        if (start_h_indata + 3 >= 0 && start_h_indata + 3 < inputHeight) {                                             \
            row3 = vload4(0, input + rowBase + inputWidth * (start_h_indata + 3));                                     \
        }                                                                                                              \
        DATA_T A[16];                                                                                                  \
        A[0] = row0.s0;                                                                                                \
        A[1] = row0.s1;                                                                                                \
        A[2] = row0.s2;                                                                                                \
        A[3] = row0.s3;                                                                                                \
        A[4] = row1.s0;                                                                                                \
        A[5] = row1.s1;                                                                                                \
        A[6] = row1.s2;                                                                                                \
        A[7] = row1.s3;                                                                                                \
        A[8] = row2.s0;                                                                                                \
        A[9] = row2.s1;                                                                                                \
        A[10] = row2.s2;                                                                                               \
        A[11] = row2.s3;                                                                                               \
        A[12] = row3.s0;                                                                                               \
        A[13] = row3.s1;                                                                                               \
        A[14] = row3.s2;                                                                                               \
        A[15] = row3.s3;                                                                                               \
        for (int i = 0; i < 4; i++) {                                                                                  \
            for (int j = start_w_indata; j < start_w_indata + 4; j++) {                                                \
                if (j < 0 || j >= inputWidth) {                                                                        \
                    A[4 * i + j - start_w_indata] = (DATA_T)(0.0f);                                                    \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
        res.s0 = A[0] - A[2] - A[8] + A[10];                                                                           \
        res.s1 = A[1] + A[2] - A[9] - A[10];                                                                           \
        res.s2 = A[2] - A[1] + A[9] - A[10];                                                                           \
        res.s3 = A[1] - A[3] - A[9] + A[11];                                                                           \
        res.s4 = A[4] - A[6] + A[8] - A[10];                                                                           \
        res.s5 = A[5] + A[6] + A[9] + A[10];                                                                           \
        res.s6 = A[6] - A[5] - A[9] + A[10];                                                                           \
        res.s7 = A[5] - A[7] + A[9] - A[11];                                                                           \
        res.s8 = A[6] - A[4] + A[8] - A[10];                                                                           \
        res.s9 = A[9] - A[5] - A[6] + A[10];                                                                           \
        res.sa = A[5] - A[6] - A[9] + A[10];                                                                           \
        res.sb = A[7] - A[5] + A[9] - A[11];                                                                           \
        res.sc = A[4] - A[6] - A[12] + A[14];                                                                          \
        res.sd = A[5] + A[6] - A[13] - A[14];                                                                          \
        res.se = A[6] - A[5] + A[13] - A[14];                                                                          \
        res.sf = A[5] - A[7] - A[13] + A[15];                                                                          \
        vstore16(res, 0, output + out_index);                                                                          \
    }

#define FAST3X3_2_MAKALU_1X2(input,                                                                                    \
                             weight,                                                                                   \
                             bias,                                                                                     \
                             output,                                                                                   \
                             inputDataNum_per_out2x2,                                                                  \
                             ouputHeight,                                                                              \
                             ouputWidth,                                                                               \
                             channel,                                                                                  \
                             wgradTileHeight,                                                                          \
                             wgradTileWidth,                                                                           \
                             coalescingRowSize)                                                                        \
    int GID0 = get_global_id(0);                                                                                       \
    int GID1 = get_global_id(1) * 2;                                                                                   \
    int GID2 = get_global_id(2);                                                                                       \
    if (GID2 < wgradTileHeight * wgradTileWidth) {                                                                     \
        DATA_T16 out16_0 = 0.0f;                                                                                       \
        DATA_T16 out16_1 = 0.0f;                                                                                       \
        int2 srcIndex = (int2)(GID2 / coalescingRowSize * coalescingRowSize * inputDataNum_per_out2x2 +                \
                                   GID2 % coalescingRowSize * 16 +                                                     \
                                   GID0 * inputDataNum_per_out2x2 *                                                    \
                                       ((wgradTileHeight * wgradTileWidth + coalescingRowSize - 1) /                   \
                                        coalescingRowSize * coalescingRowSize),                                        \
                               GID1 / 4 * 4 * inputDataNum_per_out2x2 + GID1 % 4 / 2 * 16);                            \
        int coalescingSize = coalescingRowSize * 16;                                                                   \
        for (int i = 0; i<inputDataNum_per_out2x2>> 4; i++) {                                                         \
            DATA_T16 input16;                                                                                          \
            DATA_T16 weight16_0;                                                                                       \
            DATA_T16 weight16_1;                                                                                       \
            input16 = vload16(0, input + srcIndex.s0);                                                                 \
            weight16_0 = vload16(0, weight + srcIndex.s1);                                                             \
            weight16_1 = vload16(2, weight + srcIndex.s1);                                                             \
            out16_0 = mad(input16, weight16_0, out16_0);                                                               \
            out16_1 = mad(input16, weight16_1, out16_1);                                                               \
            srcIndex += (int2)(coalescingSize, 64);                                                                    \
        }                                                                                                              \
        int channelIndex = GID0 * channel + GID1;                                                                      \
        int topRowIndex = (GID2 / wgradTileWidth) * 2;                                                                 \
        int topColIndex = (GID2 % wgradTileWidth) * 2;                                                                 \
        DATA_T2 bia = vload2(0, bias + GID1);                                                                          \
        if (topColIndex + 1 < ouputWidth && topRowIndex + 1 < ouputHeight) {                                           \
            output[channelIndex * ouputHeight * ouputWidth + topRowIndex * ouputWidth + topColIndex] =                 \
                ACT_VEC_F(DATA_T,                                                                                      \
                          out16_0.s0 + out16_0.s1 + out16_0.s2 + out16_0.s4 + out16_0.s5 + out16_0.s6 + out16_0.s8 +   \
                              out16_0.s9 + out16_0.sA + bia.s0);                                                       \
            output[channelIndex * ouputHeight * ouputWidth + topRowIndex * ouputWidth + topColIndex + 1] =             \
                ACT_VEC_F(DATA_T,                                                                                      \
                          out16_0.s1 - out16_0.s2 - out16_0.s3 + out16_0.s5 - out16_0.s6 - out16_0.s7 + out16_0.s9 -   \
                              out16_0.sA - out16_0.sB + bia.s0);                                                       \
            output[(channelIndex + 1) * ouputHeight * ouputWidth + topRowIndex * ouputWidth + topColIndex] =           \
                ACT_VEC_F(DATA_T,                                                                                      \
                          out16_1.s0 + out16_1.s1 + out16_1.s2 + out16_1.s4 + out16_1.s5 + out16_1.s6 + out16_1.s8 +   \
                              out16_1.s9 + out16_1.sA + bia.s1);                                                       \
            output[(channelIndex + 1) * ouputHeight * ouputWidth + topRowIndex * ouputWidth + topColIndex + 1] =       \
                ACT_VEC_F(DATA_T,                                                                                      \
                          out16_1.s1 - out16_1.s2 - out16_1.s3 + out16_1.s5 - out16_1.s6 - out16_1.s7 + out16_1.s9 -   \
                              out16_1.sA - out16_1.sB + bia.s1);                                                       \
            output[channelIndex * ouputHeight * ouputWidth + (topRowIndex + 1) * ouputWidth + topColIndex] =           \
                ACT_VEC_F(DATA_T,                                                                                      \
                          out16_0.s4 + out16_0.s5 + out16_0.s6 - out16_0.s8 - out16_0.s9 - out16_0.sA - out16_0.sC -   \
                              out16_0.sD - out16_0.sE + bia.s0);                                                       \
            output[channelIndex * ouputHeight * ouputWidth + (topRowIndex + 1) * ouputWidth + topColIndex + 1] =       \
                ACT_VEC_F(DATA_T,                                                                                      \
                          out16_0.s5 - out16_0.s6 - out16_0.s7 - out16_0.s9 + out16_0.sA + out16_0.sB - out16_0.sD +   \
                              out16_0.sE + out16_0.sF + bia.s0);                                                       \
            output[(channelIndex + 1) * ouputHeight * ouputWidth + (topRowIndex + 1) * ouputWidth + topColIndex] =     \
                ACT_VEC_F(DATA_T,                                                                                      \
                          out16_1.s4 + out16_1.s5 + out16_1.s6 - out16_1.s8 - out16_1.s9 - out16_1.sA - out16_1.sC -   \
                              out16_1.sD - out16_1.sE + bia.s1);                                                       \
            output[(channelIndex + 1) * ouputHeight * ouputWidth + (topRowIndex + 1) * ouputWidth + topColIndex + 1] = \
                ACT_VEC_F(DATA_T,                                                                                      \
                          out16_1.s5 - out16_1.s6 - out16_1.s7 - out16_1.s9 + out16_1.sA + out16_1.sB - out16_1.sD +   \
                              out16_1.sE + out16_1.sF + bia.s1);                                                       \
        } else if (topRowIndex + 1 < ouputHeight) {                                                                    \
            output[channelIndex * ouputHeight * ouputWidth + topRowIndex * ouputWidth + topColIndex] =                 \
                ACT_VEC_F(DATA_T,                                                                                      \
                          out16_0.s0 + out16_0.s1 + out16_0.s2 + out16_0.s4 + out16_0.s5 + out16_0.s6 + out16_0.s8 +   \
                              out16_0.s9 + out16_0.sA + bia.s0);                                                       \
            output[(channelIndex + 1) * ouputHeight * ouputWidth + topRowIndex * ouputWidth + topColIndex] =           \
                ACT_VEC_F(DATA_T,                                                                                      \
                          out16_1.s0 + out16_1.s1 + out16_1.s2 + out16_1.s4 + out16_1.s5 + out16_1.s6 + out16_1.s8 +   \
                              out16_1.s9 + out16_1.sA + bia.s1);                                                       \
            output[channelIndex * ouputHeight * ouputWidth + (topRowIndex + 1) * ouputWidth + topColIndex] =           \
                ACT_VEC_F(DATA_T,                                                                                      \
                          out16_0.s4 + out16_0.s5 + out16_0.s6 - out16_0.s8 - out16_0.s9 - out16_0.sA - out16_0.sC -   \
                              out16_0.sD - out16_0.sE + bia.s0);                                                       \
            output[(channelIndex + 1) * ouputHeight * ouputWidth + (topRowIndex + 1) * ouputWidth + topColIndex] =     \
                ACT_VEC_F(DATA_T,                                                                                      \
                          out16_1.s4 + out16_1.s5 + out16_1.s6 - out16_1.s8 - out16_1.s9 - out16_1.sA - out16_1.sC -   \
                              out16_1.sD - out16_1.sE + bia.s1);                                                       \
        } else if (topColIndex + 1 < ouputWidth) {                                                                     \
            output[channelIndex * ouputHeight * ouputWidth + topRowIndex * ouputWidth + topColIndex] =                 \
                ACT_VEC_F(DATA_T,                                                                                      \
                          out16_0.s0 + out16_0.s1 + out16_0.s2 + out16_0.s4 + out16_0.s5 + out16_0.s6 + out16_0.s8 +   \
                              out16_0.s9 + out16_0.sA + bia.s0);                                                       \
            output[channelIndex * ouputHeight * ouputWidth + topRowIndex * ouputWidth + topColIndex + 1] =             \
                ACT_VEC_F(DATA_T,                                                                                      \
                          out16_0.s1 - out16_0.s2 - out16_0.s3 + out16_0.s5 - out16_0.s6 - out16_0.s7 + out16_0.s9 -   \
                              out16_0.sA - out16_0.sB + bia.s0);                                                       \
            output[(channelIndex + 1) * ouputHeight * ouputWidth + topRowIndex * ouputWidth + topColIndex] =           \
                ACT_VEC_F(DATA_T,                                                                                      \
                          out16_1.s0 + out16_1.s1 + out16_1.s2 + out16_1.s4 + out16_1.s5 + out16_1.s6 + out16_1.s8 +   \
                              out16_1.s9 + out16_1.sA + bia.s1);                                                       \
            output[(channelIndex + 1) * ouputHeight * ouputWidth + topRowIndex * ouputWidth + topColIndex + 1] =       \
                ACT_VEC_F(DATA_T,                                                                                      \
                          out16_1.s1 - out16_1.s2 - out16_1.s3 + out16_1.s5 - out16_1.s6 - out16_1.s7 + out16_1.s9 -   \
                              out16_1.sA - out16_1.sB + bia.s1);                                                       \
        } else {                                                                                                       \
            output[channelIndex * ouputHeight * ouputWidth + topRowIndex * ouputWidth + topColIndex] =                 \
                ACT_VEC_F(DATA_T,                                                                                      \
                          out16_0.s0 + out16_0.s1 + out16_0.s2 + out16_0.s4 + out16_0.s5 + out16_0.s6 + out16_0.s8 +   \
                              out16_0.s9 + out16_0.sA + bia.s0);                                                       \
            output[(channelIndex + 1) * ouputHeight * ouputWidth + topRowIndex * ouputWidth + topColIndex] =           \
                ACT_VEC_F(DATA_T,                                                                                      \
                          out16_1.s0 + out16_1.s1 + out16_1.s2 + out16_1.s4 + out16_1.s5 + out16_1.s6 + out16_1.s8 +   \
                              out16_1.s9 + out16_1.sA + bia.s1);                                                       \
        }                                                                                                              \
    }

#define FAST3X3_2_OPTIMIZED_4X4_MERGE(input,                                                                           \
                                      weight,                                                                          \
                                      bias,                                                                            \
                                      output,                                                                          \
                                      inputDataNum_per_out2x2,                                                         \
                                      ouputHeight,                                                                     \
                                      ouputWidth,                                                                      \
                                      channel,                                                                         \
                                      wgradTileHeight,                                                                 \
                                      wgradTileWidth)                                                                  \
    int dstBatch = get_global_id(0);                                                                                   \
    int dstChannel = get_global_id(1) * 4;                                                                             \
    int dstRow = get_global_id(2) / ouputWidth;                                                                        \
    int dstCol = get_global_id(2) % ouputWidth;                                                                        \
    if (dstRow < ouputHeight && dstCol < ouputWidth) {                                                                 \
        DATA_T4 out_0;                                                                                                 \
        DATA_T4 out_1;                                                                                                 \
        DATA_T4 out_2;                                                                                                 \
        DATA_T4 out_3;                                                                                                 \
        int wgradRow = dstRow / 2;                                                                                     \
        int wgradCol = dstCol / 2;                                                                                     \
        int wgradLinearIndex = wgradRow * wgradTileWidth + wgradCol;                                                   \
        DATA_T4 bia = vload4(0, bias + dstChannel);                                                                    \
        int srcIndex = dstBatch * (channel / 4) * ((wgradTileHeight * wgradTileWidth + 3) / 4) * 4 * 4 * 4 * 2 +       \
                       dstChannel * ((wgradTileHeight * wgradTileWidth + 3) / 4) * 4 * 4 * 2 +                         \
                       wgradLinearIndex / 4 * 4 * 4 * 4 * 2 + dstCol % 2 * 4 * 4 + dstRow % 2 * 4 * 4 * 2 +            \
                       wgradLinearIndex % 4 * 4;                                                                       \
        out_0 = vload4(0, input + srcIndex);                                                                           \
        out_1 = vload4(8, input + srcIndex);                                                                           \
        out_2 = vload4(16, input + srcIndex);                                                                          \
        if (dstRow % 2 == 1) {                                                                                         \
            out_1 = -out_1;                                                                                            \
        }                                                                                                              \
        out_0 = out_0 + out_1 + out_2 + bia;                                                                           \
        int dstIndex = (dstBatch * channel + dstChannel) * ouputHeight * ouputWidth + dstRow * ouputWidth + dstCol;    \
        output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s0);                                                                \
        output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s1);                                     \
        output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s2);                                 \
        output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s3);                                 \
    }

#define FAST3X3_2_OPTIMIZED_4X4_MERGE_UNALIGNED(input,                                                                 \
                                                weight,                                                                \
                                                bias,                                                                  \
                                                output,                                                                \
                                                inputDataNum_per_out2x2,                                               \
                                                ouputHeight,                                                           \
                                                ouputWidth,                                                            \
                                                channel,                                                               \
                                                wgradTileHeight,                                                       \
                                                wgradTileWidth)                                                        \
    int dstBatch = get_global_id(0);                                                                                   \
    int dstChannel = get_global_id(1) * 4;                                                                             \
    int dstRow = get_global_id(2) / ouputWidth;                                                                        \
    int dstCol = get_global_id(2) % ouputWidth;                                                                        \
    if (dstRow < ouputHeight && dstCol < ouputWidth) {                                                                 \
        DATA_T4 out_0;                                                                                                 \
        DATA_T4 out_1;                                                                                                 \
        DATA_T4 out_2;                                                                                                 \
        DATA_T4 out_3;                                                                                                 \
        int wgradRow = dstRow / 2;                                                                                     \
        int wgradCol = dstCol / 2;                                                                                     \
        int wgradLinearIndex = wgradRow * wgradTileWidth + wgradCol;                                                   \
        DATA_T4 bia = vload4(0, bias + dstChannel);                                                                    \
        int srcIndex = dstBatch * ((channel + 3) / 4) * ((wgradTileHeight * wgradTileWidth + 3) / 4) * 4 * 4 * 4 * 2 + \
                       dstChannel * ((wgradTileHeight * wgradTileWidth + 3) / 4) * 4 * 4 * 2 +                         \
                       wgradLinearIndex / 4 * 4 * 4 * 4 * 2 + dstCol % 2 * 4 * 4 + dstRow % 2 * 4 * 4 * 2 +            \
                       wgradLinearIndex % 4 * 4;                                                                       \
        out_0 = vload4(0, input + srcIndex);                                                                           \
        out_1 = vload4(8, input + srcIndex);                                                                           \
        out_2 = vload4(16, input + srcIndex);                                                                          \
        if (dstRow % 2 == 1) {                                                                                         \
            out_1 = -out_1;                                                                                            \
        }                                                                                                              \
        out_0 = out_0 + out_1 + out_2 + bia;                                                                           \
        int dstIndex = (dstBatch * channel + dstChannel) * ouputHeight * ouputWidth + dstRow * ouputWidth + dstCol;    \
        if (dstChannel + 3 < channel) {                                                                                \
            output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s0);                                                            \
            output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s1);                                 \
            output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s2);                             \
            output[dstIndex + 3 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s3);                             \
        } else if (dstChannel + 2 < channel) {                                                                         \
            output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s0);                                                            \
            output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s1);                                 \
            output[dstIndex + 2 * ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s2);                             \
        } else if (dstChannel + 1 < channel) {                                                                         \
            output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s0);                                                            \
            output[dstIndex + ouputHeight * ouputWidth] = ACT_VEC_F(DATA_T, out_0.s1);                                 \
        } else if (dstChannel < channel) {                                                                             \
            output[dstIndex] = ACT_VEC_F(DATA_T, out_0.s0);                                                            \
        }                                                                                                              \
    }

#define WINO_CONVERT_WEIGHT_MAKALU_OPT(                                                                                \
    weight, converted_weight, input_channel, output_channel, cpt_output_number, splite_number)                         \
    int idx = get_global_id(0);                                                                                        \
    float16 a;                                                                                                         \
    float converted_weight_slice[16];                                                                                  \
    if (idx < output_channel * input_channel) {                                                                        \
        int tc = idx / input_channel;                                                                                  \
        int bc = idx % input_channel;                                                                                  \
        int inputBase = tc * input_channel * 3 * 3 + bc * 3 * 3;                                                       \
        a.lo = CONVERT_TO_FLOAT8(vload8(0, weight + inputBase));                                                       \
        a.s8 = (float)weight[inputBase + 8];                                                                           \
        converted_weight_slice[0] = a.s0;                                                                              \
        converted_weight_slice[1] = 0.5 * (a.s0 + a.s1 + a.s2);                                                        \
        converted_weight_slice[2] = 0.5 * (a.s0 - a.s1 + a.s2);                                                        \
        converted_weight_slice[3] = a.s2;                                                                              \
        converted_weight_slice[4] = 0.5 * (a.s0 + a.s3 + a.s6);                                                        \
        converted_weight_slice[5] = 0.25 * (a.s0 + a.s1 + a.s2 + a.s3 + a.s4 + a.s5 + a.s6 + a.s7 + a.s8);             \
        converted_weight_slice[6] = 0.25 * (a.s0 - a.s1 + a.s2 + a.s3 - a.s4 + a.s5 + a.s6 - a.s7 + a.s8);             \
        converted_weight_slice[7] = 0.5 * (a.s2 + a.s5 + a.s8);                                                        \
        converted_weight_slice[8] = 0.5 * (a.s0 - a.s3 + a.s6);                                                        \
        converted_weight_slice[9] = 0.25 * (a.s0 + a.s1 + a.s2 - a.s3 - a.s4 - a.s5 + a.s6 + a.s7 + a.s8);             \
        converted_weight_slice[10] = 0.25 * (a.s0 - a.s1 + a.s2 - a.s3 + a.s4 - a.s5 + a.s6 - a.s7 + a.s8);            \
        converted_weight_slice[11] = 0.5 * (a.s2 - a.s5 + a.s8);                                                       \
        converted_weight_slice[12] = a.s6;                                                                             \
        converted_weight_slice[13] = 0.5 * (a.s6 + a.s7 + a.s8);                                                       \
        converted_weight_slice[14] = 0.5 * (a.s6 - a.s7 + a.s8);                                                       \
        converted_weight_slice[15] = a.s8;                                                                             \
        int blockSize = 4 * 4 / splite_number;                                                                         \
        converted_weight[tc / 4 / 2 * input_channel * cpt_output_number * 4 * 4 * 2 +                                  \
                         bc * cpt_output_number * blockSize * 2 +                                                      \
                         0 / blockSize * cpt_output_number * blockSize * input_channel * 2 +                           \
                         0 % blockSize * cpt_output_number + (tc / 4) % 2 * cpt_output_number * blockSize + tc % 4] =  \
            converted_weight_slice[0];                                                                                 \
        converted_weight[tc / 4 / 2 * input_channel * cpt_output_number * 4 * 4 * 2 +                                  \
                         bc * cpt_output_number * blockSize * 2 +                                                      \
                         1 / blockSize * cpt_output_number * blockSize * input_channel * 2 +                           \
                         1 % blockSize * cpt_output_number + (tc / 4) % 2 * cpt_output_number * blockSize + tc % 4] =  \
            converted_weight_slice[1];                                                                                 \
        converted_weight[tc / 4 / 2 * input_channel * cpt_output_number * 4 * 4 * 2 +                                  \
                         bc * cpt_output_number * blockSize * 2 +                                                      \
                         2 / blockSize * cpt_output_number * blockSize * input_channel * 2 +                           \
                         2 % blockSize * cpt_output_number + (tc / 4) % 2 * cpt_output_number * blockSize + tc % 4] =  \
            converted_weight_slice[2];                                                                                 \
        converted_weight[tc / 4 / 2 * input_channel * cpt_output_number * 4 * 4 * 2 +                                  \
                         bc * cpt_output_number * blockSize * 2 +                                                      \
                         3 / blockSize * cpt_output_number * blockSize * input_channel * 2 +                           \
                         3 % blockSize * cpt_output_number + (tc / 4) % 2 * cpt_output_number * blockSize + tc % 4] =  \
            converted_weight_slice[3];                                                                                 \
        converted_weight[tc / 4 / 2 * input_channel * cpt_output_number * 4 * 4 * 2 +                                  \
                         bc * cpt_output_number * blockSize * 2 +                                                      \
                         4 / blockSize * cpt_output_number * blockSize * input_channel * 2 +                           \
                         4 % blockSize * cpt_output_number + (tc / 4) % 2 * cpt_output_number * blockSize + tc % 4] =  \
            converted_weight_slice[4];                                                                                 \
        converted_weight[tc / 4 / 2 * input_channel * cpt_output_number * 4 * 4 * 2 +                                  \
                         bc * cpt_output_number * blockSize * 2 +                                                      \
                         5 / blockSize * cpt_output_number * blockSize * input_channel * 2 +                           \
                         5 % blockSize * cpt_output_number + (tc / 4) % 2 * cpt_output_number * blockSize + tc % 4] =  \
            converted_weight_slice[5];                                                                                 \
        converted_weight[tc / 4 / 2 * input_channel * cpt_output_number * 4 * 4 * 2 +                                  \
                         bc * cpt_output_number * blockSize * 2 +                                                      \
                         6 / blockSize * cpt_output_number * blockSize * input_channel * 2 +                           \
                         6 % blockSize * cpt_output_number + (tc / 4) % 2 * cpt_output_number * blockSize + tc % 4] =  \
            converted_weight_slice[6];                                                                                 \
        converted_weight[tc / 4 / 2 * input_channel * cpt_output_number * 4 * 4 * 2 +                                  \
                         bc * cpt_output_number * blockSize * 2 +                                                      \
                         7 / blockSize * cpt_output_number * blockSize * input_channel * 2 +                           \
                         7 % blockSize * cpt_output_number + (tc / 4) % 2 * cpt_output_number * blockSize + tc % 4] =  \
            converted_weight_slice[7];                                                                                 \
        converted_weight[tc / 4 / 2 * input_channel * cpt_output_number * 4 * 4 * 2 +                                  \
                         bc * cpt_output_number * blockSize * 2 +                                                      \
                         8 / blockSize * cpt_output_number * blockSize * input_channel * 2 +                           \
                         8 % blockSize * cpt_output_number + (tc / 4) % 2 * cpt_output_number * blockSize + tc % 4] =  \
            converted_weight_slice[8];                                                                                 \
        converted_weight[tc / 4 / 2 * input_channel * cpt_output_number * 4 * 4 * 2 +                                  \
                         bc * cpt_output_number * blockSize * 2 +                                                      \
                         9 / blockSize * cpt_output_number * blockSize * input_channel * 2 +                           \
                         9 % blockSize * cpt_output_number + (tc / 4) % 2 * cpt_output_number * blockSize + tc % 4] =  \
            converted_weight_slice[9];                                                                                 \
        converted_weight[tc / 4 / 2 * input_channel * cpt_output_number * 4 * 4 * 2 +                                  \
                         bc * cpt_output_number * blockSize * 2 +                                                      \
                         10 / blockSize * cpt_output_number * blockSize * input_channel * 2 +                          \
                         10 % blockSize * cpt_output_number + (tc / 4) % 2 * cpt_output_number * blockSize + tc % 4] = \
            converted_weight_slice[10];                                                                                \
        converted_weight[tc / 4 / 2 * input_channel * cpt_output_number * 4 * 4 * 2 +                                  \
                         bc * cpt_output_number * blockSize * 2 +                                                      \
                         11 / blockSize * cpt_output_number * blockSize * input_channel * 2 +                          \
                         11 % blockSize * cpt_output_number + (tc / 4) % 2 * cpt_output_number * blockSize + tc % 4] = \
            converted_weight_slice[11];                                                                                \
        converted_weight[tc / 4 / 2 * input_channel * cpt_output_number * 4 * 4 * 2 +                                  \
                         bc * cpt_output_number * blockSize * 2 +                                                      \
                         12 / blockSize * cpt_output_number * blockSize * input_channel * 2 +                          \
                         12 % blockSize * cpt_output_number + (tc / 4) % 2 * cpt_output_number * blockSize + tc % 4] = \
            converted_weight_slice[12];                                                                                \
        converted_weight[tc / 4 / 2 * input_channel * cpt_output_number * 4 * 4 * 2 +                                  \
                         bc * cpt_output_number * blockSize * 2 +                                                      \
                         13 / blockSize * cpt_output_number * blockSize * input_channel * 2 +                          \
                         13 % blockSize * cpt_output_number + (tc / 4) % 2 * cpt_output_number * blockSize + tc % 4] = \
            converted_weight_slice[13];                                                                                \
        converted_weight[tc / 4 / 2 * input_channel * cpt_output_number * 4 * 4 * 2 +                                  \
                         bc * cpt_output_number * blockSize * 2 +                                                      \
                         14 / blockSize * cpt_output_number * blockSize * input_channel * 2 +                          \
                         14 % blockSize * cpt_output_number + (tc / 4) % 2 * cpt_output_number * blockSize + tc % 4] = \
            converted_weight_slice[14];                                                                                \
        converted_weight[tc / 4 / 2 * input_channel * cpt_output_number * 4 * 4 * 2 +                                  \
                         bc * cpt_output_number * blockSize * 2 +                                                      \
                         15 / blockSize * cpt_output_number * blockSize * input_channel * 2 +                          \
                         15 % blockSize * cpt_output_number + (tc / 4) % 2 * cpt_output_number * blockSize + tc % 4] = \
            converted_weight_slice[15];                                                                                \
    }

#define WINO_CONVERT_WEIGHT(                                                                                           \
    weight, convert_weight, weightG, output_channel, cpt_output_number, speed_up, splite_number)                       \
    float G[12];                                                                                                       \
    G[0] = 1;                                                                                                          \
    G[1] = 0;                                                                                                          \
    G[2] = 0;                                                                                                          \
    G[3] = 0.5;                                                                                                        \
    G[4] = 0.5;                                                                                                        \
    G[5] = 0.5;                                                                                                        \
    G[6] = 0.5;                                                                                                        \
    G[7] = -0.5;                                                                                                       \
    G[8] = 0.5;                                                                                                        \
    G[9] = 0;                                                                                                          \
    G[10] = 0;                                                                                                         \
    G[11] = 1;                                                                                                         \
    float GT[12];                                                                                                      \
    GT[0] = 1;                                                                                                         \
    GT[1] = 0.5;                                                                                                       \
    GT[2] = 0.5;                                                                                                       \
    GT[3] = 0;                                                                                                         \
    GT[4] = 0;                                                                                                         \
    GT[5] = 0.5;                                                                                                       \
    GT[6] = -0.5;                                                                                                      \
    GT[7] = 0;                                                                                                         \
    GT[8] = 0;                                                                                                         \
    GT[9] = 0.5;                                                                                                       \
    GT[10] = 0.5;                                                                                                      \
    GT[11] = 1;                                                                                                        \
    int idx0 = get_global_id(0);                                                                                       \
    int idx1 = get_global_id(1);                                                                                       \
    int input_channel = get_global_size(1);                                                                            \
    int channelIndex[8];                                                                                               \
    float temp[8];                                                                                                     \
    for (int i = 0; i < cpt_output_number; i++) {                                                                      \
        channelIndex[i] = (idx0 * cpt_output_number + i) * input_channel + idx1;                                       \
    }                                                                                                                  \
    for (int b = 0; b < 4; b++) {                                                                                      \
        for (int c = 0; c < 3; c++) {                                                                                  \
            for (int i = 0; i < cpt_output_number; i++) {                                                              \
                temp[i] = 0;                                                                                           \
            }                                                                                                          \
            for (int d = 0; d < 3; d++) {                                                                              \
                for (int i = 0; i < cpt_output_number; i++) {                                                          \
                    temp[i] += G[b * 3 + d] * (float)(weight[channelIndex[i] * 3 * 3 + 3 * d + c]);                    \
                }                                                                                                      \
            }                                                                                                          \
            for (int i = 0; i < cpt_output_number; i++) {                                                              \
                weightG[channelIndex[i] * 4 * 3 + b * 3 + c] = temp[i];                                                \
            }                                                                                                          \
        }                                                                                                              \
    }                                                                                                                  \
    for (int b = 0; b < 4; b++) {                                                                                      \
        for (int e = 0; e < 4; e++) {                                                                                  \
            for (int i = 0; i < cpt_output_number; i++) {                                                              \
                temp[i] = 0;                                                                                           \
            }                                                                                                          \
            for (int d = 0; d < 3; d++) {                                                                              \
                for (int i = 0; i < cpt_output_number; i++) {                                                          \
                    temp[i] += weightG[channelIndex[i] * 4 * 3 + b * 3 + d] * GT[4 * d + e];                           \
                }                                                                                                      \
            }                                                                                                          \
            for (int i = 0; i < cpt_output_number; i++) {                                                              \
                if (speed_up == 1) {                                                                                   \
                    int blockSize = 4 * 4 / splite_number;                                                             \
                    convert_weight[idx0 * input_channel * cpt_output_number * 4 * 4 +                                  \
                                   idx1 * cpt_output_number * blockSize +                                              \
                                   (b * 4 + e) / blockSize * cpt_output_number * blockSize * input_channel +           \
                                   (b * 4 + e) % blockSize * cpt_output_number + i] = (DATA_T)(temp[i]);               \
                } else if (speed_up == 2) {                                                                            \
                    int blockSize = 4 * 4 / splite_number;                                                             \
                    convert_weight[idx0 / 2 * input_channel * cpt_output_number * 4 * 4 * 2 +                          \
                                   idx1 * cpt_output_number * blockSize * 2 +                                          \
                                   (b * 4 + e) / blockSize * cpt_output_number * blockSize * input_channel * 2 +       \
                                   (b * 4 + e) % blockSize * cpt_output_number +                                       \
                                   idx0 % 2 * cpt_output_number * blockSize + i] = (DATA_T)(temp[i]);                  \
                } else if (speed_up == 3) {                                                                            \
                    convert_weight[idx0 / 2 * input_channel * cpt_output_number * 4 * 4 * 2 +                          \
                                   idx1 * cpt_output_number * 4 * 4 * 2 + idx0 % 2 * 4 * 4 + b * 4 + e +               \
                                   i * 4 * 4 * 2] = (DATA_T)(temp[i]);                                                 \
                } else {                                                                                               \
                    convert_weight[idx0 * input_channel * cpt_output_number * 4 * 4 +                                  \
                                   idx1 * cpt_output_number * 4 * 4 + b * 4 + e + i * 16] = (DATA_T)(temp[i]);         \
                }                                                                                                      \
            }                                                                                                          \
        }                                                                                                              \
    }

#define COPY_BUFFER(src, dst, src_offset, dst_offset)                                                                  \
    int g0 = get_global_id(0);                                                                                         \
    dst[g0 + dst_offset] = src[g0 + src_offset];

/********  FP16 KERNELS ********/
#define DATA_T half
#define DATA_T2 half2
#define DATA_T3 half3
#define DATA_T4 half4
#define DATA_T8 half8
#define DATA_T16 half16
#define CONVERT_TO_FLOAT8(x) convert_float8(x)
ADD_SINGLE_KERNEL(Fast3x3_1_2x8_FP16,
                  (__global const DATA_T *input,
                   __global DATA_T *output,
                   unsigned int padHeight,
                   unsigned int padWidth,
                   unsigned int inputHeight,
                   unsigned int inputWidth,
                   unsigned int chan,
                   unsigned int height,
                   unsigned int width,
                   unsigned int coalescingRowSize){FAST3X3_1_2X8(input,
                                                                 output,
                                                                 padHeight,
                                                                 padWidth,
                                                                 inputHeight,
                                                                 inputWidth,
                                                                 chan,
                                                                 height,
                                                                 width,
                                                                 coalescingRowSize)})

ADD_SINGLE_KERNEL(Fast3x3_2_optimized_2x8_splite_FP16,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth,
                   unsigned int coalescingRowSize){FAST3X3_2_OPTIMIZED_2X8_SPLITE(input,
                                                                                  weight,
                                                                                  bias,
                                                                                  output,
                                                                                  inputDataNum_per_out2x2,
                                                                                  ouputHeight,
                                                                                  ouputWidth,
                                                                                  channel,
                                                                                  wgradTileHeight,
                                                                                  wgradTileWidth,
                                                                                  coalescingRowSize)})

ADD_SINGLE_KERNEL(Fast3x3_1_2x4_FP16,
                  (__global const DATA_T *input,
                   __global DATA_T *output,
                   unsigned int padHeight,
                   unsigned int padWidth,
                   unsigned int inputHeight,
                   unsigned int inputWidth,
                   unsigned int chan,
                   unsigned int height,
                   unsigned int width,
                   unsigned int coalescingRowSize){FAST3X3_1_2X4(input,
                                                                 output,
                                                                 padHeight,
                                                                 padWidth,
                                                                 inputHeight,
                                                                 inputWidth,
                                                                 chan,
                                                                 height,
                                                                 width,
                                                                 coalescingRowSize)})

ADD_SINGLE_KERNEL(Fast3x3_2_optimized_2x4_splite_FP16,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth,
                   unsigned int coalescingRowSize){FAST3X3_2_OPTIMIZED_2X4_SPLITE(input,
                                                                                  weight,
                                                                                  bias,
                                                                                  output,
                                                                                  inputDataNum_per_out2x2,
                                                                                  ouputHeight,
                                                                                  ouputWidth,
                                                                                  channel,
                                                                                  wgradTileHeight,
                                                                                  wgradTileWidth,
                                                                                  coalescingRowSize)})

ADD_SINGLE_KERNEL(Fast3x3_1_1x8_FP16,
                  (__global const DATA_T *input,
                   __global DATA_T *output,
                   unsigned int padHeight,
                   unsigned int padWidth,
                   unsigned int inputHeight,
                   unsigned int inputWidth,
                   unsigned int chan,
                   unsigned int height,
                   unsigned int width,
                   unsigned int coalescingRowSize){FAST3X3_1_1X8_FP16(input,
                                                                      output,
                                                                      padHeight,
                                                                      padWidth,
                                                                      inputHeight,
                                                                      inputWidth,
                                                                      chan,
                                                                      height,
                                                                      width,
                                                                      coalescingRowSize)})

ADD_SINGLE_KERNEL(Fast3x3_1_FP16,
                  (__global const DATA_T *input,
                   __global DATA_T *output,
                   unsigned int padHeight,
                   unsigned int padWidth,
                   unsigned int inputHeight,
                   unsigned int inputWidth,
                   unsigned int groupIndex,
                   unsigned int chan,
                   unsigned int groupNum,
                   unsigned int height,
                   unsigned int width){FAST3X3_1(input,
                                                 output,
                                                 padHeight,
                                                 padWidth,
                                                 inputHeight,
                                                 inputWidth,
                                                 groupIndex,
                                                 chan,
                                                 groupNum,
                                                 height,
                                                 width)})

ADD_SINGLE_KERNEL(Fast3x3_2_optimized_1x8_splite_FP16,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth,
                   unsigned int coalescingRowSize){FAST3X3_2_OPTIMIZED_1X8_SPLITE_FP16(input,
                                                                                       weight,
                                                                                       bias,
                                                                                       output,
                                                                                       inputDataNum_per_out2x2,
                                                                                       ouputHeight,
                                                                                       ouputWidth,
                                                                                       channel,
                                                                                       wgradTileHeight,
                                                                                       wgradTileWidth,
                                                                                       coalescingRowSize)})

ADD_SINGLE_KERNEL(Fast3x3_1_4x4_FP16,
                  (__global const DATA_T *input,
                   __global DATA_T *output,
                   unsigned int padHeight,
                   unsigned int padWidth,
                   unsigned int inputHeight,
                   unsigned int inputWidth,
                   unsigned int chan,
                   unsigned int height,
                   unsigned int width,
                   unsigned int coalescingRowSize){FAST3X3_1_4X4(input,
                                                                 output,
                                                                 padHeight,
                                                                 padWidth,
                                                                 inputHeight,
                                                                 inputWidth,
                                                                 chan,
                                                                 height,
                                                                 width,
                                                                 coalescingRowSize)})

ADD_SINGLE_KERNEL(Fast3x3_2_optimized_4x4_splite_FP16,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth,
                   unsigned int coalescingRowSize){FAST3X3_2_OPTIMIZED_4X4_SPLITE(input,
                                                                                  weight,
                                                                                  bias,
                                                                                  output,
                                                                                  inputDataNum_per_out2x2,
                                                                                  ouputHeight,
                                                                                  ouputWidth,
                                                                                  channel,
                                                                                  wgradTileHeight,
                                                                                  wgradTileWidth,
                                                                                  coalescingRowSize)})

ADD_SINGLE_KERNEL(Fast3x3_1_makalu_1x2_FP16,
                  (__global const DATA_T *input,
                   __global DATA_T *output,
                   unsigned int padHeight,
                   unsigned int padWidth,
                   unsigned int inputHeight,
                   unsigned int inputWidth,
                   unsigned int chan,
                   unsigned int height,
                   unsigned int width,
                   unsigned int coalescingRowSize){FAST3X3_1_MAKALU_1X2(input,
                                                                        output,
                                                                        padHeight,
                                                                        padWidth,
                                                                        inputHeight,
                                                                        inputWidth,
                                                                        chan,
                                                                        height,
                                                                        width,
                                                                        coalescingRowSize)})

ADD_SINGLE_KERNEL(wino_convert_weight_makalu_opt_FP16,
                  (__global const DATA_T *weight,
                   __global DATA_T *converted_weight,
                   const unsigned int input_channel,
                   const unsigned int output_channel,
                   const unsigned int cpt_output_number,
                   const unsigned int splite_number){WINO_CONVERT_WEIGHT_MAKALU_OPT(weight,
                                                                                    converted_weight,
                                                                                    input_channel,
                                                                                    output_channel,
                                                                                    cpt_output_number,
                                                                                    splite_number)})

ADD_SINGLE_KERNEL(wino_convert_weight_FP16,
                  (__global const DATA_T *weight,
                   __global DATA_T *convert_weight,
                   __global float *weightG,
                   const unsigned int output_channel,
                   const unsigned int cpt_output_number,
                   const int speed_up,
                   const unsigned int splite_number){WINO_CONVERT_WEIGHT(weight,
                                                                         convert_weight,
                                                                         weightG,
                                                                         output_channel,
                                                                         cpt_output_number,
                                                                         speed_up,
                                                                         splite_number)})

ADD_SINGLE_KERNEL(copy_buffer_FP16,
                  (__global const DATA_T *src, __global DATA_T *dst, unsigned int src_offset, unsigned int dst_offset){
                      COPY_BUFFER(src, dst, src_offset, dst_offset)})

// kernels without activation func
#define ACT_VEC_F(VEC_T, x) x
ADD_SINGLE_KERNEL(Fast3x3_2_optimized_2x8_merge_FP16,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth){FAST3X3_2_OPTIMIZED_2X8_MERGE(input,
                                                                              weight,
                                                                              bias,
                                                                              output,
                                                                              inputDataNum_per_out2x2,
                                                                              ouputHeight,
                                                                              ouputWidth,
                                                                              channel,
                                                                              wgradTileHeight,
                                                                              wgradTileWidth)})

ADD_SINGLE_KERNEL(Fast3x3_2_optimized_2x4_merge_FP16,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth){FAST3X3_2_OPTIMIZED_2X4_MERGE(input,
                                                                              weight,
                                                                              bias,
                                                                              output,
                                                                              inputDataNum_per_out2x2,
                                                                              ouputHeight,
                                                                              ouputWidth,
                                                                              channel,
                                                                              wgradTileHeight,
                                                                              wgradTileWidth)})

ADD_SINGLE_KERNEL(Fast3x3_2_FP16,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int groupIndex,
                   unsigned int chan,
                   unsigned int groupNum){FAST3X3_2(input,
                                                    weight,
                                                    bias,
                                                    output,
                                                    inputDataNum_per_out2x2,
                                                    ouputHeight,
                                                    ouputWidth,
                                                    groupIndex,
                                                    chan,
                                                    groupNum)})

ADD_SINGLE_KERNEL(Fast3x3_2_optimized_FP16,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth,
                   unsigned int groupNum,
                   unsigned int groupIndex){FAST3X3_2_OPTIMIZED(input,
                                                                weight,
                                                                bias,
                                                                output,
                                                                inputDataNum_per_out2x2,
                                                                ouputHeight,
                                                                ouputWidth,
                                                                channel,
                                                                wgradTileHeight,
                                                                wgradTileWidth,
                                                                groupNum,
                                                                groupIndex)})

ADD_SINGLE_KERNEL(Fast3x3_2_optimized_1x8_merge_FP16,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth){FAST3X3_2_OPTIMIZED_1X8_MERGE_FP16(input,
                                                                                   weight,
                                                                                   bias,
                                                                                   output,
                                                                                   inputDataNum_per_out2x2,
                                                                                   ouputHeight,
                                                                                   ouputWidth,
                                                                                   channel,
                                                                                   wgradTileHeight,
                                                                                   wgradTileWidth)})

ADD_SINGLE_KERNEL(Fast3x3_2_makalu_1x2_FP16,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth,
                   unsigned int coalescingRowSize){FAST3X3_2_MAKALU_1X2(input,
                                                                        weight,
                                                                        bias,
                                                                        output,
                                                                        inputDataNum_per_out2x2,
                                                                        ouputHeight,
                                                                        ouputWidth,
                                                                        channel,
                                                                        wgradTileHeight,
                                                                        wgradTileWidth,
                                                                        coalescingRowSize)})

ADD_SINGLE_KERNEL(Fast3x3_2_optimized_4x4_merge_FP16,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth){FAST3X3_2_OPTIMIZED_4X4_MERGE(input,
                                                                              weight,
                                                                              bias,
                                                                              output,
                                                                              inputDataNum_per_out2x2,
                                                                              ouputHeight,
                                                                              ouputWidth,
                                                                              channel,
                                                                              wgradTileHeight,
                                                                              wgradTileWidth)})

ADD_SINGLE_KERNEL(Fast3x3_2_optimized_4x4_merge_unaligned_FP16,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth){FAST3X3_2_OPTIMIZED_4X4_MERGE_UNALIGNED(input,
                                                                                        weight,
                                                                                        bias,
                                                                                        output,
                                                                                        inputDataNum_per_out2x2,
                                                                                        ouputHeight,
                                                                                        ouputWidth,
                                                                                        channel,
                                                                                        wgradTileHeight,
                                                                                        wgradTileWidth)})

#undef ACT_VEC_F  // NONE

// apply RELU to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELUFast3x3_2_optimized_2x8_merge_FP16,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth){FAST3X3_2_OPTIMIZED_2X8_MERGE(input,
                                                                              weight,
                                                                              bias,
                                                                              output,
                                                                              inputDataNum_per_out2x2,
                                                                              ouputHeight,
                                                                              ouputWidth,
                                                                              channel,
                                                                              wgradTileHeight,
                                                                              wgradTileWidth)})

ADD_SINGLE_KERNEL(RELUFast3x3_2_optimized_2x4_merge_FP16,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth){FAST3X3_2_OPTIMIZED_2X4_MERGE(input,
                                                                              weight,
                                                                              bias,
                                                                              output,
                                                                              inputDataNum_per_out2x2,
                                                                              ouputHeight,
                                                                              ouputWidth,
                                                                              channel,
                                                                              wgradTileHeight,
                                                                              wgradTileWidth)})

ADD_SINGLE_KERNEL(RELUFast3x3_2_FP16,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int groupIndex,
                   unsigned int chan,
                   unsigned int groupNum){FAST3X3_2(input,
                                                    weight,
                                                    bias,
                                                    output,
                                                    inputDataNum_per_out2x2,
                                                    ouputHeight,
                                                    ouputWidth,
                                                    groupIndex,
                                                    chan,
                                                    groupNum)})

ADD_SINGLE_KERNEL(RELUFast3x3_2_optimized_FP16,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth,
                   unsigned int groupNum,
                   unsigned int groupIndex){FAST3X3_2_OPTIMIZED(input,
                                                                weight,
                                                                bias,
                                                                output,
                                                                inputDataNum_per_out2x2,
                                                                ouputHeight,
                                                                ouputWidth,
                                                                channel,
                                                                wgradTileHeight,
                                                                wgradTileWidth,
                                                                groupNum,
                                                                groupIndex)})

ADD_SINGLE_KERNEL(RELUFast3x3_2_optimized_1x8_merge_FP16,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth){FAST3X3_2_OPTIMIZED_1X8_MERGE_FP16(input,
                                                                                   weight,
                                                                                   bias,
                                                                                   output,
                                                                                   inputDataNum_per_out2x2,
                                                                                   ouputHeight,
                                                                                   ouputWidth,
                                                                                   channel,
                                                                                   wgradTileHeight,
                                                                                   wgradTileWidth)})

ADD_SINGLE_KERNEL(RELUFast3x3_2_makalu_1x2_FP16,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth,
                   unsigned int coalescingRowSize){FAST3X3_2_MAKALU_1X2(input,
                                                                        weight,
                                                                        bias,
                                                                        output,
                                                                        inputDataNum_per_out2x2,
                                                                        ouputHeight,
                                                                        ouputWidth,
                                                                        channel,
                                                                        wgradTileHeight,
                                                                        wgradTileWidth,
                                                                        coalescingRowSize)})

ADD_SINGLE_KERNEL(RELUFast3x3_2_optimized_4x4_merge_FP16,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth){FAST3X3_2_OPTIMIZED_4X4_MERGE(input,
                                                                              weight,
                                                                              bias,
                                                                              output,
                                                                              inputDataNum_per_out2x2,
                                                                              ouputHeight,
                                                                              ouputWidth,
                                                                              channel,
                                                                              wgradTileHeight,
                                                                              wgradTileWidth)})

ADD_SINGLE_KERNEL(RELUFast3x3_2_optimized_4x4_merge_unaligned_FP16,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth){FAST3X3_2_OPTIMIZED_4X4_MERGE_UNALIGNED(input,
                                                                                        weight,
                                                                                        bias,
                                                                                        output,
                                                                                        inputDataNum_per_out2x2,
                                                                                        ouputHeight,
                                                                                        ouputWidth,
                                                                                        channel,
                                                                                        wgradTileHeight,
                                                                                        wgradTileWidth)})

#undef ACT_VEC_F  // RELU

// apply RELU6 to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU6_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELU6Fast3x3_2_optimized_2x8_merge_FP16,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth){FAST3X3_2_OPTIMIZED_2X8_MERGE(input,
                                                                              weight,
                                                                              bias,
                                                                              output,
                                                                              inputDataNum_per_out2x2,
                                                                              ouputHeight,
                                                                              ouputWidth,
                                                                              channel,
                                                                              wgradTileHeight,
                                                                              wgradTileWidth)})

ADD_SINGLE_KERNEL(RELU6Fast3x3_2_optimized_2x4_merge_FP16,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth){FAST3X3_2_OPTIMIZED_2X4_MERGE(input,
                                                                              weight,
                                                                              bias,
                                                                              output,
                                                                              inputDataNum_per_out2x2,
                                                                              ouputHeight,
                                                                              ouputWidth,
                                                                              channel,
                                                                              wgradTileHeight,
                                                                              wgradTileWidth)})

ADD_SINGLE_KERNEL(RELU6Fast3x3_2_FP16,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int groupIndex,
                   unsigned int chan,
                   unsigned int groupNum){FAST3X3_2(input,
                                                    weight,
                                                    bias,
                                                    output,
                                                    inputDataNum_per_out2x2,
                                                    ouputHeight,
                                                    ouputWidth,
                                                    groupIndex,
                                                    chan,
                                                    groupNum)})

ADD_SINGLE_KERNEL(RELU6Fast3x3_2_optimized_FP16,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth,
                   unsigned int groupNum,
                   unsigned int groupIndex){FAST3X3_2_OPTIMIZED(input,
                                                                weight,
                                                                bias,
                                                                output,
                                                                inputDataNum_per_out2x2,
                                                                ouputHeight,
                                                                ouputWidth,
                                                                channel,
                                                                wgradTileHeight,
                                                                wgradTileWidth,
                                                                groupNum,
                                                                groupIndex)})

ADD_SINGLE_KERNEL(RELU6Fast3x3_2_optimized_1x8_merge_FP16,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth){FAST3X3_2_OPTIMIZED_1X8_MERGE_FP16(input,
                                                                                   weight,
                                                                                   bias,
                                                                                   output,
                                                                                   inputDataNum_per_out2x2,
                                                                                   ouputHeight,
                                                                                   ouputWidth,
                                                                                   channel,
                                                                                   wgradTileHeight,
                                                                                   wgradTileWidth)})

ADD_SINGLE_KERNEL(RELU6Fast3x3_2_makalu_1x2_FP16,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth,
                   unsigned int coalescingRowSize){FAST3X3_2_MAKALU_1X2(input,
                                                                        weight,
                                                                        bias,
                                                                        output,
                                                                        inputDataNum_per_out2x2,
                                                                        ouputHeight,
                                                                        ouputWidth,
                                                                        channel,
                                                                        wgradTileHeight,
                                                                        wgradTileWidth,
                                                                        coalescingRowSize)})

ADD_SINGLE_KERNEL(RELU6Fast3x3_2_optimized_4x4_merge_FP16,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth){FAST3X3_2_OPTIMIZED_4X4_MERGE(input,
                                                                              weight,
                                                                              bias,
                                                                              output,
                                                                              inputDataNum_per_out2x2,
                                                                              ouputHeight,
                                                                              ouputWidth,
                                                                              channel,
                                                                              wgradTileHeight,
                                                                              wgradTileWidth)})

ADD_SINGLE_KERNEL(RELU6Fast3x3_2_optimized_4x4_merge_unaligned_FP16,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth){FAST3X3_2_OPTIMIZED_4X4_MERGE_UNALIGNED(input,
                                                                                        weight,
                                                                                        bias,
                                                                                        output,
                                                                                        inputDataNum_per_out2x2,
                                                                                        ouputHeight,
                                                                                        ouputWidth,
                                                                                        channel,
                                                                                        wgradTileHeight,
                                                                                        wgradTileWidth)})

#undef ACT_VEC_F  // RELU6

#undef CONVERT_TO_FLOAT8
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
#define CONVERT_TO_FLOAT8(x) x
ADD_SINGLE_KERNEL(Fast3x3_1_2x8_FP32,
                  (__global const DATA_T *input,
                   __global DATA_T *output,
                   unsigned int padHeight,
                   unsigned int padWidth,
                   unsigned int inputHeight,
                   unsigned int inputWidth,
                   unsigned int chan,
                   unsigned int height,
                   unsigned int width,
                   unsigned int coalescingRowSize){FAST3X3_1_2X8(input,
                                                                 output,
                                                                 padHeight,
                                                                 padWidth,
                                                                 inputHeight,
                                                                 inputWidth,
                                                                 chan,
                                                                 height,
                                                                 width,
                                                                 coalescingRowSize)})

ADD_SINGLE_KERNEL(Fast3x3_2_optimized_2x8_splite_FP32,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth,
                   unsigned int coalescingRowSize){FAST3X3_2_OPTIMIZED_2X8_SPLITE(input,
                                                                                  weight,
                                                                                  bias,
                                                                                  output,
                                                                                  inputDataNum_per_out2x2,
                                                                                  ouputHeight,
                                                                                  ouputWidth,
                                                                                  channel,
                                                                                  wgradTileHeight,
                                                                                  wgradTileWidth,
                                                                                  coalescingRowSize)})

ADD_SINGLE_KERNEL(Fast3x3_1_2x4_FP32,
                  (__global const DATA_T *input,
                   __global DATA_T *output,
                   unsigned int padHeight,
                   unsigned int padWidth,
                   unsigned int inputHeight,
                   unsigned int inputWidth,
                   unsigned int chan,
                   unsigned int height,
                   unsigned int width,
                   unsigned int coalescingRowSize){FAST3X3_1_2X4(input,
                                                                 output,
                                                                 padHeight,
                                                                 padWidth,
                                                                 inputHeight,
                                                                 inputWidth,
                                                                 chan,
                                                                 height,
                                                                 width,
                                                                 coalescingRowSize)})

ADD_SINGLE_KERNEL(Fast3x3_2_optimized_2x4_splite_FP32,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth,
                   unsigned int coalescingRowSize){FAST3X3_2_OPTIMIZED_2X4_SPLITE(input,
                                                                                  weight,
                                                                                  bias,
                                                                                  output,
                                                                                  inputDataNum_per_out2x2,
                                                                                  ouputHeight,
                                                                                  ouputWidth,
                                                                                  channel,
                                                                                  wgradTileHeight,
                                                                                  wgradTileWidth,
                                                                                  coalescingRowSize)})

ADD_SINGLE_KERNEL(
    Fast3x3_1_1x8_FP32,
    (__global const DATA_T *input,
     __global DATA_T *output,
     unsigned int padHeight,
     unsigned int padWidth,
     unsigned int inputHeight,
     unsigned int inputWidth,
     unsigned int chan,
     unsigned int height,
     unsigned int width){
        FAST3X3_1_1X8_FP32(input, output, padHeight, padWidth, inputHeight, inputWidth, chan, height, width)})

ADD_SINGLE_KERNEL(Fast3x3_1_FP32,
                  (__global const DATA_T *input,
                   __global DATA_T *output,
                   unsigned int padHeight,
                   unsigned int padWidth,
                   unsigned int inputHeight,
                   unsigned int inputWidth,
                   unsigned int groupIndex,
                   unsigned int chan,
                   unsigned int groupNum,
                   unsigned int height,
                   unsigned int width){FAST3X3_1(input,
                                                 output,
                                                 padHeight,
                                                 padWidth,
                                                 inputHeight,
                                                 inputWidth,
                                                 groupIndex,
                                                 chan,
                                                 groupNum,
                                                 height,
                                                 width)})

ADD_SINGLE_KERNEL(Fast3x3_2_optimized_1x8_splite_FP32,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth){FAST3X3_2_OPTIMIZED_1X8_SPLITE_FP32(input,
                                                                                    weight,
                                                                                    bias,
                                                                                    output,
                                                                                    inputDataNum_per_out2x2,
                                                                                    ouputHeight,
                                                                                    ouputWidth,
                                                                                    channel,
                                                                                    wgradTileHeight,
                                                                                    wgradTileWidth)})

ADD_SINGLE_KERNEL(Fast3x3_1_4x4_FP32,
                  (__global const DATA_T *input,
                   __global DATA_T *output,
                   unsigned int padHeight,
                   unsigned int padWidth,
                   unsigned int inputHeight,
                   unsigned int inputWidth,
                   unsigned int chan,
                   unsigned int height,
                   unsigned int width,
                   unsigned int coalescingRowSize){FAST3X3_1_4X4(input,
                                                                 output,
                                                                 padHeight,
                                                                 padWidth,
                                                                 inputHeight,
                                                                 inputWidth,
                                                                 chan,
                                                                 height,
                                                                 width,
                                                                 coalescingRowSize)})

ADD_SINGLE_KERNEL(Fast3x3_2_optimized_4x4_splite_FP32,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth,
                   unsigned int coalescingRowSize){FAST3X3_2_OPTIMIZED_4X4_SPLITE(input,
                                                                                  weight,
                                                                                  bias,
                                                                                  output,
                                                                                  inputDataNum_per_out2x2,
                                                                                  ouputHeight,
                                                                                  ouputWidth,
                                                                                  channel,
                                                                                  wgradTileHeight,
                                                                                  wgradTileWidth,
                                                                                  coalescingRowSize)})

ADD_SINGLE_KERNEL(Fast3x3_1_makalu_1x2_FP32,
                  (__global const DATA_T *input,
                   __global DATA_T *output,
                   unsigned int padHeight,
                   unsigned int padWidth,
                   unsigned int inputHeight,
                   unsigned int inputWidth,
                   unsigned int chan,
                   unsigned int height,
                   unsigned int width,
                   unsigned int coalescingRowSize){FAST3X3_1_MAKALU_1X2(input,
                                                                        output,
                                                                        padHeight,
                                                                        padWidth,
                                                                        inputHeight,
                                                                        inputWidth,
                                                                        chan,
                                                                        height,
                                                                        width,
                                                                        coalescingRowSize)})

ADD_SINGLE_KERNEL(wino_convert_weight_makalu_opt_FP32,
                  (__global const DATA_T *weight,
                   __global DATA_T *converted_weight,
                   const unsigned int input_channel,
                   const unsigned int output_channel,
                   const unsigned int cpt_output_number,
                   const unsigned int splite_number){WINO_CONVERT_WEIGHT_MAKALU_OPT(weight,
                                                                                    converted_weight,
                                                                                    input_channel,
                                                                                    output_channel,
                                                                                    cpt_output_number,
                                                                                    splite_number)})

ADD_SINGLE_KERNEL(wino_convert_weight_FP32,
                  (__global const DATA_T *weight,
                   __global DATA_T *convert_weight,
                   __global float *weightG,
                   const unsigned int output_channel,
                   const unsigned int cpt_output_number,
                   const int speed_up,
                   const unsigned int splite_number){WINO_CONVERT_WEIGHT(weight,
                                                                         convert_weight,
                                                                         weightG,
                                                                         output_channel,
                                                                         cpt_output_number,
                                                                         speed_up,
                                                                         splite_number)})

ADD_SINGLE_KERNEL(copy_buffer_FP32,
                  (__global const DATA_T *src, __global DATA_T *dst, unsigned int src_offset, unsigned int dst_offset){
                      COPY_BUFFER(src, dst, src_offset, dst_offset)})

// kernels without activation func
#define ACT_VEC_F(VEC_T, x) x
ADD_SINGLE_KERNEL(Fast3x3_2_optimized_2x8_merge_FP32,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth){FAST3X3_2_OPTIMIZED_2X8_MERGE(input,
                                                                              weight,
                                                                              bias,
                                                                              output,
                                                                              inputDataNum_per_out2x2,
                                                                              ouputHeight,
                                                                              ouputWidth,
                                                                              channel,
                                                                              wgradTileHeight,
                                                                              wgradTileWidth)})

ADD_SINGLE_KERNEL(Fast3x3_2_optimized_2x4_merge_FP32,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth){FAST3X3_2_OPTIMIZED_2X4_MERGE(input,
                                                                              weight,
                                                                              bias,
                                                                              output,
                                                                              inputDataNum_per_out2x2,
                                                                              ouputHeight,
                                                                              ouputWidth,
                                                                              channel,
                                                                              wgradTileHeight,
                                                                              wgradTileWidth)})

ADD_SINGLE_KERNEL(Fast3x3_2_FP32,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int groupIndex,
                   unsigned int chan,
                   unsigned int groupNum){FAST3X3_2(input,
                                                    weight,
                                                    bias,
                                                    output,
                                                    inputDataNum_per_out2x2,
                                                    ouputHeight,
                                                    ouputWidth,
                                                    groupIndex,
                                                    chan,
                                                    groupNum)})

ADD_SINGLE_KERNEL(Fast3x3_2_optimized_FP32,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth,
                   unsigned int groupNum,
                   unsigned int groupIndex){FAST3X3_2_OPTIMIZED(input,
                                                                weight,
                                                                bias,
                                                                output,
                                                                inputDataNum_per_out2x2,
                                                                ouputHeight,
                                                                ouputWidth,
                                                                channel,
                                                                wgradTileHeight,
                                                                wgradTileWidth,
                                                                groupNum,
                                                                groupIndex)})

ADD_SINGLE_KERNEL(Fast3x3_2_optimized_1x8_merge_FP32,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth){FAST3X3_2_OPTIMIZED_1X8_MERGE_FP32(input,
                                                                                   weight,
                                                                                   bias,
                                                                                   output,
                                                                                   inputDataNum_per_out2x2,
                                                                                   ouputHeight,
                                                                                   ouputWidth,
                                                                                   channel,
                                                                                   wgradTileHeight,
                                                                                   wgradTileWidth)})

ADD_SINGLE_KERNEL(Fast3x3_2_makalu_1x2_FP32,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth,
                   unsigned int coalescingRowSize){FAST3X3_2_MAKALU_1X2(input,
                                                                        weight,
                                                                        bias,
                                                                        output,
                                                                        inputDataNum_per_out2x2,
                                                                        ouputHeight,
                                                                        ouputWidth,
                                                                        channel,
                                                                        wgradTileHeight,
                                                                        wgradTileWidth,
                                                                        coalescingRowSize)})

ADD_SINGLE_KERNEL(Fast3x3_2_optimized_4x4_merge_FP32,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth){FAST3X3_2_OPTIMIZED_4X4_MERGE(input,
                                                                              weight,
                                                                              bias,
                                                                              output,
                                                                              inputDataNum_per_out2x2,
                                                                              ouputHeight,
                                                                              ouputWidth,
                                                                              channel,
                                                                              wgradTileHeight,
                                                                              wgradTileWidth)})

ADD_SINGLE_KERNEL(Fast3x3_2_optimized_4x4_merge_unaligned_FP32,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth){FAST3X3_2_OPTIMIZED_4X4_MERGE_UNALIGNED(input,
                                                                                        weight,
                                                                                        bias,
                                                                                        output,
                                                                                        inputDataNum_per_out2x2,
                                                                                        ouputHeight,
                                                                                        ouputWidth,
                                                                                        channel,
                                                                                        wgradTileHeight,
                                                                                        wgradTileWidth)})

#undef ACT_VEC_F  // NONE

// apply RELU to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELUFast3x3_2_optimized_2x8_merge_FP32,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth){FAST3X3_2_OPTIMIZED_2X8_MERGE(input,
                                                                              weight,
                                                                              bias,
                                                                              output,
                                                                              inputDataNum_per_out2x2,
                                                                              ouputHeight,
                                                                              ouputWidth,
                                                                              channel,
                                                                              wgradTileHeight,
                                                                              wgradTileWidth)})

ADD_SINGLE_KERNEL(RELUFast3x3_2_optimized_2x4_merge_FP32,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth){FAST3X3_2_OPTIMIZED_2X4_MERGE(input,
                                                                              weight,
                                                                              bias,
                                                                              output,
                                                                              inputDataNum_per_out2x2,
                                                                              ouputHeight,
                                                                              ouputWidth,
                                                                              channel,
                                                                              wgradTileHeight,
                                                                              wgradTileWidth)})

ADD_SINGLE_KERNEL(RELUFast3x3_2_FP32,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int groupIndex,
                   unsigned int chan,
                   unsigned int groupNum){FAST3X3_2(input,
                                                    weight,
                                                    bias,
                                                    output,
                                                    inputDataNum_per_out2x2,
                                                    ouputHeight,
                                                    ouputWidth,
                                                    groupIndex,
                                                    chan,
                                                    groupNum)})

ADD_SINGLE_KERNEL(RELUFast3x3_2_optimized_FP32,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth,
                   unsigned int groupNum,
                   unsigned int groupIndex){FAST3X3_2_OPTIMIZED(input,
                                                                weight,
                                                                bias,
                                                                output,
                                                                inputDataNum_per_out2x2,
                                                                ouputHeight,
                                                                ouputWidth,
                                                                channel,
                                                                wgradTileHeight,
                                                                wgradTileWidth,
                                                                groupNum,
                                                                groupIndex)})

ADD_SINGLE_KERNEL(RELUFast3x3_2_optimized_1x8_merge_FP32,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth){FAST3X3_2_OPTIMIZED_1X8_MERGE_FP32(input,
                                                                                   weight,
                                                                                   bias,
                                                                                   output,
                                                                                   inputDataNum_per_out2x2,
                                                                                   ouputHeight,
                                                                                   ouputWidth,
                                                                                   channel,
                                                                                   wgradTileHeight,
                                                                                   wgradTileWidth)})

ADD_SINGLE_KERNEL(RELUFast3x3_2_makalu_1x2_FP32,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth,
                   unsigned int coalescingRowSize){FAST3X3_2_MAKALU_1X2(input,
                                                                        weight,
                                                                        bias,
                                                                        output,
                                                                        inputDataNum_per_out2x2,
                                                                        ouputHeight,
                                                                        ouputWidth,
                                                                        channel,
                                                                        wgradTileHeight,
                                                                        wgradTileWidth,
                                                                        coalescingRowSize)})

ADD_SINGLE_KERNEL(RELUFast3x3_2_optimized_4x4_merge_FP32,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth){FAST3X3_2_OPTIMIZED_4X4_MERGE(input,
                                                                              weight,
                                                                              bias,
                                                                              output,
                                                                              inputDataNum_per_out2x2,
                                                                              ouputHeight,
                                                                              ouputWidth,
                                                                              channel,
                                                                              wgradTileHeight,
                                                                              wgradTileWidth)})

ADD_SINGLE_KERNEL(RELUFast3x3_2_optimized_4x4_merge_unaligned_FP32,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth){FAST3X3_2_OPTIMIZED_4X4_MERGE_UNALIGNED(input,
                                                                                        weight,
                                                                                        bias,
                                                                                        output,
                                                                                        inputDataNum_per_out2x2,
                                                                                        ouputHeight,
                                                                                        ouputWidth,
                                                                                        channel,
                                                                                        wgradTileHeight,
                                                                                        wgradTileWidth)})

#undef ACT_VEC_F  // RELU

// apply RELU6 to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU6_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELU6Fast3x3_2_optimized_2x8_merge_FP32,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth){FAST3X3_2_OPTIMIZED_2X8_MERGE(input,
                                                                              weight,
                                                                              bias,
                                                                              output,
                                                                              inputDataNum_per_out2x2,
                                                                              ouputHeight,
                                                                              ouputWidth,
                                                                              channel,
                                                                              wgradTileHeight,
                                                                              wgradTileWidth)})

ADD_SINGLE_KERNEL(RELU6Fast3x3_2_optimized_2x4_merge_FP32,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth){FAST3X3_2_OPTIMIZED_2X4_MERGE(input,
                                                                              weight,
                                                                              bias,
                                                                              output,
                                                                              inputDataNum_per_out2x2,
                                                                              ouputHeight,
                                                                              ouputWidth,
                                                                              channel,
                                                                              wgradTileHeight,
                                                                              wgradTileWidth)})

ADD_SINGLE_KERNEL(RELU6Fast3x3_2_FP32,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int groupIndex,
                   unsigned int chan,
                   unsigned int groupNum){FAST3X3_2(input,
                                                    weight,
                                                    bias,
                                                    output,
                                                    inputDataNum_per_out2x2,
                                                    ouputHeight,
                                                    ouputWidth,
                                                    groupIndex,
                                                    chan,
                                                    groupNum)})

ADD_SINGLE_KERNEL(RELU6Fast3x3_2_optimized_FP32,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth,
                   unsigned int groupNum,
                   unsigned int groupIndex){FAST3X3_2_OPTIMIZED(input,
                                                                weight,
                                                                bias,
                                                                output,
                                                                inputDataNum_per_out2x2,
                                                                ouputHeight,
                                                                ouputWidth,
                                                                channel,
                                                                wgradTileHeight,
                                                                wgradTileWidth,
                                                                groupNum,
                                                                groupIndex)})

ADD_SINGLE_KERNEL(RELU6Fast3x3_2_optimized_1x8_merge_FP32,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth){FAST3X3_2_OPTIMIZED_1X8_MERGE_FP32(input,
                                                                                   weight,
                                                                                   bias,
                                                                                   output,
                                                                                   inputDataNum_per_out2x2,
                                                                                   ouputHeight,
                                                                                   ouputWidth,
                                                                                   channel,
                                                                                   wgradTileHeight,
                                                                                   wgradTileWidth)})

ADD_SINGLE_KERNEL(RELU6Fast3x3_2_makalu_1x2_FP32,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth,
                   unsigned int coalescingRowSize){FAST3X3_2_MAKALU_1X2(input,
                                                                        weight,
                                                                        bias,
                                                                        output,
                                                                        inputDataNum_per_out2x2,
                                                                        ouputHeight,
                                                                        ouputWidth,
                                                                        channel,
                                                                        wgradTileHeight,
                                                                        wgradTileWidth,
                                                                        coalescingRowSize)})

ADD_SINGLE_KERNEL(RELU6Fast3x3_2_optimized_4x4_merge_FP32,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth){FAST3X3_2_OPTIMIZED_4X4_MERGE(input,
                                                                              weight,
                                                                              bias,
                                                                              output,
                                                                              inputDataNum_per_out2x2,
                                                                              ouputHeight,
                                                                              ouputWidth,
                                                                              channel,
                                                                              wgradTileHeight,
                                                                              wgradTileWidth)})

ADD_SINGLE_KERNEL(RELU6Fast3x3_2_optimized_4x4_merge_unaligned_FP32,
                  (__global const DATA_T *input,
                   __global const DATA_T *weight,
                   __global const DATA_T *bias,
                   __global DATA_T *output,
                   unsigned int inputDataNum_per_out2x2,
                   unsigned int ouputHeight,
                   unsigned int ouputWidth,
                   unsigned int channel,
                   unsigned int wgradTileHeight,
                   unsigned int wgradTileWidth){FAST3X3_2_OPTIMIZED_4X4_MERGE_UNALIGNED(input,
                                                                                        weight,
                                                                                        bias,
                                                                                        output,
                                                                                        inputDataNum_per_out2x2,
                                                                                        ouputHeight,
                                                                                        ouputWidth,
                                                                                        channel,
                                                                                        wgradTileHeight,
                                                                                        wgradTileWidth)})

#undef ACT_VEC_F  // RELU6

#undef CONVERT_TO_FLOAT8
#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // float

}  // namespace gpu
}  // namespace ud
}  // namespace enn
