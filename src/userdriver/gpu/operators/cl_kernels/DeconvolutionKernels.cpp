#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define Q_WEIGHTALIGN(matrix, matrixAlign, widthAlign, width) \
        unsigned int outIndex = get_global_id(0) * widthAlign + get_global_id(1); \
        unsigned int inputIndex = get_global_id(0) * width + get_global_id(1); \
        matrixAlign[outIndex] = matrix[inputIndex];

#define MATRIXTRANS(matrix, matrixTrans, widthTransAlign, batchId, groupNum) \
        unsigned int heightTrans = get_global_size(1); \
        unsigned int widthTrans = get_global_size(2); \
        unsigned int outIndex = batchId * groupNum * heightTrans * widthTransAlign + \
                                get_global_id(0) * widthTransAlign * heightTrans + \
                                get_global_id(1) * widthTransAlign + get_global_id(2); \
        unsigned int inputIndex = batchId * groupNum * heightTrans * widthTrans + \
                                  get_global_id(0) * widthTrans * heightTrans + \
                                  get_global_id(2) * heightTrans + get_global_id(1); \
        matrixTrans[outIndex] = matrix[inputIndex];

#define Q_CONVBACKGEMMRXR(A, B, output, rowA, colA, rowB, batchId, groupNum, input_offset, filter_offset) \
        int globalID0 = get_global_id(0);      \
        int globalID1 = get_global_id(1) * 8;  \
        int globalID2 = get_global_id(2);      \
        if (globalID2 < rowB && globalID1 < rowA) { \
            int4 output8_0 = (int4)(0); \
            int4 output8_1 = (int4)(0); \
            int4 output8_2 = (int4)(0); \
            int4 output8_3 = (int4)(0); \
            int4 output8_4 = (int4)(0); \
            int4 output8_5 = (int4)(0); \
            int4 output8_6 = (int4)(0); \
            int4 output8_7 = (int4)(0); \
            int4 input_offset4 = (int4)(input_offset); \
            int4 filter_offset4 = (int4)(filter_offset); \
            WEIGHT_T4 weight8; \
            INPUT_T4 input8; \
            int4 input8_int; \
            unsigned int startA = get_global_id(0) * rowA * colA + \
                                  globalID1 * colA;  \
            unsigned int startB = batchId * rowB * colA * groupNum + \
                                  get_global_id(0) * rowB * colA + \
                                  get_global_id(2) * colA;  \
            unsigned int startC = batchId * rowA * rowB * groupNum + \
                                  get_global_id(0) * rowA * rowB + globalID1 * rowB + \
                                  get_global_id(2); \
            for (int i = 0; i < (colA >> 2); i++) { \
                input8 = vload4(0, B + startB + (i << 2)); \
                input8_int = convert_int4(input8) + input_offset4; \
                weight8 = vload4(0, A + startA + (i << 2)); \
                output8_0 += input8_int * (convert_int4(weight8) + filter_offset4); \
                weight8 = vload4(0, A + startA + (i << 2) + colA); \
                output8_1 += input8_int * (convert_int4(weight8) + filter_offset4); \
                weight8 = vload4(0, A + startA + (i << 2) + colA * 2); \
                output8_2 += input8_int * (convert_int4(weight8) + filter_offset4); \
                weight8 = vload4(0, A + startA + (i << 2) + colA * 3); \
                output8_3 += input8_int * (convert_int4(weight8) + filter_offset4); \
                weight8 = vload4(0, A + startA + (i << 2) + colA * 4); \
                output8_4 += input8_int * (convert_int4(weight8) + filter_offset4); \
                weight8 = vload4(0, A + startA + (i << 2) + colA * 5); \
                output8_5 += input8_int * (convert_int4(weight8) + filter_offset4); \
                weight8 = vload4(0, A + startA + (i << 2) + colA * 6); \
                output8_6 += input8_int * (convert_int4(weight8) + filter_offset4); \
                weight8 = vload4(0, A + startA + (i << 2) + colA * 7); \
                output8_7 += input8_int * (convert_int4(weight8) + filter_offset4); \
            } \
            if (globalID1 + 8 <= rowA) { \
                output[startC] = output8_0.s0 + output8_0.s1 + output8_0.s2 + output8_0.s3; \
                output[startC + rowB] = output8_1.s0 + output8_1.s1 + output8_1.s2 + output8_1.s3; \
                output[startC + rowB * 2] = \
                    output8_2.s0 + output8_2.s1 + output8_2.s2 + output8_2.s3; \
                output[startC + rowB * 3] = \
                    output8_3.s0 + output8_3.s1 + output8_3.s2 + output8_3.s3; \
                output[startC + rowB * 4] = \
                    output8_4.s0 + output8_4.s1 + output8_4.s2 + output8_4.s3; \
                output[startC + rowB * 5] = \
                    output8_5.s0 + output8_5.s1 + output8_5.s2 + output8_5.s3; \
                output[startC + rowB * 6] = \
                    output8_6.s0 + output8_6.s1 + output8_6.s2 + output8_6.s3; \
                output[startC + rowB * 7] = \
                    output8_7.s0 + output8_7.s1 + output8_7.s2 + output8_7.s3; \
            } else { \
                int num = rowA - globalID1; \
                if (num == 1) { \
                    output[startC] = output8_0.s0 + output8_0.s1 + output8_0.s2 + output8_0.s3; \
                } else if (num == 2) { \
                    output[startC] = output8_0.s0 + output8_0.s1 + output8_0.s2 + output8_0.s3; \
                    output[startC + rowB] = \
                        output8_1.s0 + output8_1.s1 + output8_1.s2 + output8_1.s3; \
                } else if (num == 3) { \
                    output[startC] = output8_0.s0 + output8_0.s1 + output8_0.s2 + output8_0.s3; \
                    output[startC + rowB] = \
                        output8_1.s0 + output8_1.s1 + output8_1.s2 + output8_1.s3; \
                    output[startC + rowB * 2] = \
                        output8_2.s0 + output8_2.s1 + output8_2.s2 + output8_2.s3; \
                } else if (num == 4) { \
                    output[startC] = output8_0.s0 + output8_0.s1 + output8_0.s2 + output8_0.s3; \
                    output[startC + rowB] = \
                        output8_1.s0 + output8_1.s1 + output8_1.s2 + output8_1.s3; \
                    output[startC + rowB * 2] = \
                        output8_2.s0 + output8_2.s1 + output8_2.s2 + output8_2.s3; \
                    output[startC + rowB * 3] = \
                        output8_3.s0 + output8_3.s1 + output8_3.s2 + output8_3.s3; \
                } else if (num == 5) { \
                    output[startC] = output8_0.s0 + output8_0.s1 + output8_0.s2 + output8_0.s3; \
                    output[startC + rowB] = \
                        output8_1.s0 + output8_1.s1 + output8_1.s2 + output8_1.s3; \
                    output[startC + rowB * 2] = \
                        output8_2.s0 + output8_2.s1 + output8_2.s2 + output8_2.s3; \
                    output[startC + rowB * 3] = \
                        output8_3.s0 + output8_3.s1 + output8_3.s2 + output8_3.s3; \
                    output[startC + rowB * 4] = \
                        output8_4.s0 + output8_4.s1 + output8_4.s2 + output8_4.s3; \
                } else if (num == 6) { \
                    output[startC] = output8_0.s0 + output8_0.s1 + output8_0.s2 + output8_0.s3; \
                    output[startC + rowB] = \
                        output8_1.s0 + output8_1.s1 + output8_1.s2 + output8_1.s3; \
                    output[startC + rowB * 2] = \
                        output8_2.s0 + output8_2.s1 + output8_2.s2 + output8_2.s3; \
                    output[startC + rowB * 3] = \
                        output8_3.s0 + output8_3.s1 + output8_3.s2 + output8_3.s3; \
                    output[startC + rowB * 4] = \
                        output8_4.s0 + output8_4.s1 + output8_4.s2 + output8_4.s3; \
                    output[startC + rowB * 5] = \
                        output8_5.s0 + output8_5.s1 + output8_5.s2 + output8_5.s3; \
                } else if (num == 7) { \
                    output[startC] = output8_0.s0 + output8_0.s1 + output8_0.s2 + output8_0.s3; \
                    output[startC + rowB] = \
                        output8_1.s0 + output8_1.s1 + output8_1.s2 + output8_1.s3; \
                    output[startC + rowB * 2] = \
                        output8_2.s0 + output8_2.s1 + output8_2.s2 + output8_2.s3; \
                    output[startC + rowB * 3] = \
                        output8_3.s0 + output8_3.s1 + output8_3.s2 + output8_3.s3; \
                    output[startC + rowB * 4] = \
                        output8_4.s0 + output8_4.s1 + output8_4.s2 + output8_4.s3; \
                    output[startC + rowB * 5] = \
                        output8_5.s0 + output8_5.s1 + output8_5.s2 + output8_5.s3; \
                    output[startC + rowB * 6] = \
                        output8_6.s0 + output8_6.s1 + output8_6.s2 + output8_6.s3; \
                } \
            } \
        }

#define Q_CONVERTBOTTOMDIFF2(input, bias, output, height, width, channels, kernelHeight, kernelWidth, padHeight, \
                           padWidth, strideHeight, strideWidth, height_col, width_col, output_offset, \
                           output_multiplier, output_shift, act_min, act_max) \
        int globalID0 = get_global_id(0);  \
        int globalID1 = get_global_id(1);  \
        int globalID2 = get_global_id(2);  \
        bool overflow; \
        long ab_64; \
        int nudge; \
        int mask; \
        int remainder; \
        int threshold; \
        int shift = -output_shift; \
        int left_shift = shift > 0 ? shift : 0; \
        int right_shilt = shift > 0 ? 0 : -shift; \
        if (globalID1 < height && globalID2 < width) { \
            int val = 0; \
            int w_im = get_global_id(2) + padWidth; \
            int h_im = get_global_id(1) + padHeight; \
            int c_im = get_global_id(0); \
            int w_col_start = (w_im < kernelWidth) ? 0 : (w_im - kernelWidth) / strideWidth + 1; \
            int w_col_end = \
                (w_im / strideWidth + 1) < width_col ? (w_im / strideWidth + 1) : width_col; \
            int h_col_start = (h_im < kernelHeight) ? 0 : (h_im - kernelHeight) / strideHeight + 1; \
            int h_col_end = \
                (h_im / strideHeight + 1) < height_col ? (h_im / strideHeight + 1) : height_col; \
            for (int h_col = h_col_start; h_col < h_col_end; ++h_col) { \
                for (int w_col = w_col_start; w_col < w_col_end; ++w_col) { \
                    int c_col = c_im * kernelHeight * kernelWidth + \
                                (h_im - h_col * strideHeight) * kernelWidth + \
                                (w_im - w_col * strideWidth); \
                    val += input[(c_col * height_col + h_col) * width_col + w_col]; \
                } \
            } \
            int biasIndex = get_global_id(0) % channels; \
            val += bias[biasIndex]; \
            val = val * (1 << left_shift); \
            reQuantized(val, \
                        output_multiplier, \
                        right_shilt, \
                        output_offset, \
                        act_min, \
                        act_max, \
                        overflow, \
                        ab_64, \
                        nudge, \
                        mask, \
                        remainder, \
                        threshold); \
            output[get_global_id(0) * height * width + get_global_id(1) * width + \
                get_global_id(2)] = (DATA_T)val; \
        }

#define Q_CONVERTBOTTOMDIFF2_PER_CHANNEL(input, bias, output, height, width, channels, kernelHeight, kernelWidth, \
                                       padHeight, padWidth, strideHeight, strideWidth, height_col, width_col, \
                                       output_offset, output_multiplier, output_shift, act_min, act_max) \
        int globalID0 = get_global_id(0);  \
        int globalID1 = get_global_id(1);  \
        int globalID2 = get_global_id(2);  \
        bool overflow; \
        long ab_64; \
        int nudge; \
        int mask; \
        int remainder; \
        int threshold; \
        if (globalID1 < height && globalID2 < width) { \
            int biasIndex = get_global_id(0) % channels; \
            int shift = output_shift[biasIndex]; \
            int left_shift = shift > 0 ? shift : 0; \
            int right_shilt = shift > 0 ? 0 : -shift; \
            int val = 0; \
            int w_im = get_global_id(2) + padWidth; \
            int h_im = get_global_id(1) + padHeight; \
            int c_im = get_global_id(0); \
            int w_col_start = (w_im < kernelWidth) ? 0 : (w_im - kernelWidth) / strideWidth + 1; \
            int w_col_end = \
                (w_im / strideWidth + 1) < width_col ? (w_im / strideWidth + 1) : width_col; \
            int h_col_start = (h_im < kernelHeight) ? 0 : (h_im - kernelHeight) / strideHeight + 1; \
            int h_col_end = \
                (h_im / strideHeight + 1) < height_col ? (h_im / strideHeight + 1) : height_col; \
            for (int h_col = h_col_start; h_col < h_col_end; ++h_col) { \
                for (int w_col = w_col_start; w_col < w_col_end; ++w_col) { \
                    int c_col = c_im * kernelHeight * kernelWidth + \
                                (h_im - h_col * strideHeight) * kernelWidth + \
                                (w_im - w_col * strideWidth); \
                    val += input[(c_col * height_col + h_col) * width_col + w_col]; \
                } \
            } \
            val += bias[biasIndex]; \
            val = val * (1 << left_shift); \
            reQuantized(val, \
                        output_multiplier[biasIndex], \
                        right_shilt, \
                        output_offset, \
                        act_min, \
                        act_max, \
                        overflow, \
                        ab_64, \
                        nudge, \
                        mask, \
                        remainder, \
                        threshold); \
            output[get_global_id(0) * height * width + get_global_id(1) * width + \
                get_global_id(2)] = (DATA_T)val; \
        }

#define Q_CONV_BACK_GEMMRXR_PARAMS __global const WEIGHT_T *A, __global const INPUT_T *B, __global int *output, \
                                   unsigned int rowA, unsigned int colA, unsigned int rowB, unsigned int batchId, \
                                   unsigned int groupNum, int input_offset, int filter_offset
/********  QUANTIZED KERNELS ********/
// UINT8 kernels
#define DATA_T uchar
#define DATA_T2 uchar2
#define DATA_T3 uchar3
#define DATA_T4 uchar4
#define DATA_T8 uchar8
#define DATA_T16 uchar16
#define INPUT_T DATA_T
#define INPUT_T4 DATA_T4
#define WEIGHT_T DATA_T
#define WEIGHT_T4 DATA_T4

ADD_SINGLE_KERNEL(weightAlign_INT8, (__global const DATA_T *matrix,
                                   __global DATA_T *matrixAlign,
                                   unsigned int widthAlign,
                                   unsigned int width) {
    Q_WEIGHTALIGN(matrix, matrixAlign, widthAlign, width)
})

ADD_SINGLE_KERNEL(matrixTrans_INT8, (__global const DATA_T *matrix,
                                   __global DATA_T *matrixTrans,
                                   unsigned int widthTransAlign,
                                   unsigned int batchId,
                                   unsigned int groupNum) {
    MATRIXTRANS(matrix, matrixTrans, widthTransAlign, batchId, groupNum)
})

ADD_SINGLE_KERNEL(convBackGemmRXR_INT8, (Q_CONV_BACK_GEMMRXR_PARAMS) {
    Q_CONVBACKGEMMRXR(A, B, output, rowA, colA, rowB, batchId, groupNum, input_offset, filter_offset)
})

ADD_KERNEL_HEADER(convertBottomDiff2_INT8,
                  {DEFINE_SATURATING_ROUNDING_DOUBLING_HIGH_MUL, DEFINE_ROUNDING_DIVIDE_BY_POT, DEFINE_REQUANTIZED})
ADD_SINGLE_KERNEL(convertBottomDiff2_INT8, (__global const int *input,
                                          __global const int *bias,
                                          __global DATA_T *output,
                                          unsigned int height,
                                          unsigned int width,
                                          unsigned int channels,
                                          unsigned int kernelHeight,
                                          unsigned int kernelWidth,
                                          unsigned int padHeight,
                                          unsigned int padWidth,
                                          unsigned int strideHeight,
                                          unsigned int strideWidth,
                                          unsigned int height_col,
                                          unsigned int width_col,
                                          int output_offset,
                                          int output_multiplier,
                                          int output_shift,
                                          int act_min,
                                          int act_max) {
    Q_CONVERTBOTTOMDIFF2(input, bias, output, height, width, channels, kernelHeight, kernelWidth, padHeight, padWidth, \
                       strideHeight, strideWidth, height_col, width_col, output_offset, output_multiplier, \
                       output_shift, act_min, act_max)
})

ADD_KERNEL_HEADER(convertBottomDiff2_per_channel_INT8,
                  {DEFINE_SATURATING_ROUNDING_DOUBLING_HIGH_MUL, DEFINE_ROUNDING_DIVIDE_BY_POT, DEFINE_REQUANTIZED})
ADD_SINGLE_KERNEL(convertBottomDiff2_per_channel_INT8, (__global const int *input,
                                                      __global const int *bias,
                                                      __global DATA_T *output,
                                                      unsigned int height,
                                                      unsigned int width,
                                                      unsigned int channels,
                                                      unsigned int kernelHeight,
                                                      unsigned int kernelWidth,
                                                      unsigned int padHeight,
                                                      unsigned int padWidth,
                                                      unsigned int strideHeight,
                                                      unsigned int strideWidth,
                                                      unsigned int height_col,
                                                      unsigned int width_col,
                                                      int output_offset,
                                                      __global int *output_multiplier,
                                                      __global int *output_shift,
                                                      int act_min,
                                                      int act_max) {
    Q_CONVERTBOTTOMDIFF2_PER_CHANNEL(input, bias, output, height, width, channels, kernelHeight, kernelWidth, \
                                   padHeight, padWidth, strideHeight, strideWidth, height_col, width_col, \
                                   output_offset, output_multiplier, output_shift, act_min, act_max)
})

#undef WEIGHT_T4
#undef WEIGHT_T
#undef INPUT_T4
#undef INPUT_T
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
#define INPUT_T DATA_T
#define INPUT_T4 DATA_T4
#define WEIGHT_T DATA_T
#define WEIGHT_T4 DATA_T4


ADD_SINGLE_KERNEL(SIGNEDweightAlign_INT8, (__global const DATA_T *matrix,
                                   __global DATA_T *matrixAlign,
                                   unsigned int widthAlign,
                                   unsigned int width) {
    Q_WEIGHTALIGN(matrix, matrixAlign, widthAlign, width)
})

ADD_SINGLE_KERNEL(SIGNEDmatrixTrans_INT8, (__global const DATA_T *matrix,
                                   __global DATA_T *matrixTrans,
                                   unsigned int widthTransAlign,
                                   unsigned int batchId,
                                   unsigned int groupNum) {
    MATRIXTRANS(matrix, matrixTrans, widthTransAlign, batchId, groupNum)
})

ADD_SINGLE_KERNEL(SIGNEDconvBackGemmRXR_INT8, (Q_CONV_BACK_GEMMRXR_PARAMS) {
    Q_CONVBACKGEMMRXR(A, B, output, rowA, colA, rowB, batchId, groupNum, input_offset, filter_offset)
})

ADD_KERNEL_HEADER(SIGNEDconvertBottomDiff2_INT8,
                  {DEFINE_SATURATING_ROUNDING_DOUBLING_HIGH_MUL, DEFINE_ROUNDING_DIVIDE_BY_POT, DEFINE_REQUANTIZED})
ADD_SINGLE_KERNEL(SIGNEDconvertBottomDiff2_INT8, (__global const int *input,
                                          __global const int *bias,
                                          __global DATA_T *output,
                                          unsigned int height,
                                          unsigned int width,
                                          unsigned int channels,
                                          unsigned int kernelHeight,
                                          unsigned int kernelWidth,
                                          unsigned int padHeight,
                                          unsigned int padWidth,
                                          unsigned int strideHeight,
                                          unsigned int strideWidth,
                                          unsigned int height_col,
                                          unsigned int width_col,
                                          int output_offset,
                                          int output_multiplier,
                                          int output_shift,
                                          int act_min,
                                          int act_max) {
    Q_CONVERTBOTTOMDIFF2(input, bias, output, height, width, channels, kernelHeight, kernelWidth, padHeight, padWidth, \
                       strideHeight, strideWidth, height_col, width_col, output_offset, output_multiplier, \
                       output_shift, act_min, act_max)
})

ADD_KERNEL_HEADER(SIGNEDconvertBottomDiff2_per_channel_INT8,
                  {DEFINE_SATURATING_ROUNDING_DOUBLING_HIGH_MUL, DEFINE_ROUNDING_DIVIDE_BY_POT, DEFINE_REQUANTIZED})
ADD_SINGLE_KERNEL(SIGNEDconvertBottomDiff2_per_channel_INT8, (__global const int *input,
                                                      __global const int *bias,
                                                      __global DATA_T *output,
                                                      unsigned int height,
                                                      unsigned int width,
                                                      unsigned int channels,
                                                      unsigned int kernelHeight,
                                                      unsigned int kernelWidth,
                                                      unsigned int padHeight,
                                                      unsigned int padWidth,
                                                      unsigned int strideHeight,
                                                      unsigned int strideWidth,
                                                      unsigned int height_col,
                                                      unsigned int width_col,
                                                      int output_offset,
                                                      __global int *output_multiplier,
                                                      __global int *output_shift,
                                                      int act_min,
                                                      int act_max) {
    Q_CONVERTBOTTOMDIFF2_PER_CHANNEL(input, bias, output, height, width, channels, kernelHeight, kernelWidth, \
                                   padHeight, padWidth, strideHeight, strideWidth, height_col, width_col, \
                                   output_offset, output_multiplier, output_shift, act_min, act_max)
})

#undef WEIGHT_T4
#undef WEIGHT_T
#undef INPUT_T4
#undef INPUT_T
#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // int8

// PERCHANNEL QUANTIZED KERNELS
#define INPUT_T uchar
#define INPUT_T4 uchar4
#define WEIGHT_T char
#define WEIGHT_T4 char4

ADD_SINGLE_KERNEL(PERCHANNELconvBackGemmRXR_INT8, (Q_CONV_BACK_GEMMRXR_PARAMS) {
    Q_CONVBACKGEMMRXR(A, B, output, rowA, colA, rowB, batchId, groupNum, input_offset, filter_offset)
})

#undef WEIGHT_T4
#undef WEIGHT_T
#undef INPUT_T4
#undef INPUT_T

#define CONVBACKGEMMRXR_FP16(A, B, output, rowA, colA, rowB, batchId, groupNum) \
        int globalID0 = get_global_id(0);      \
        int globalID1 = get_global_id(1) * 8;  \
        int globalID2 = get_global_id(2);      \
        if (globalID2 < rowB && globalID1 < rowA) { \
            DATA_T8 output8_0 = (DATA_T8)(0.0f); \
            DATA_T8 output8_1 = (DATA_T8)(0.0f); \
            DATA_T8 output8_2 = (DATA_T8)(0.0f); \
            DATA_T8 output8_3 = (DATA_T8)(0.0f); \
            DATA_T8 output8_4 = (DATA_T8)(0.0f); \
            DATA_T8 output8_5 = (DATA_T8)(0.0f); \
            DATA_T8 output8_6 = (DATA_T8)(0.0f); \
            DATA_T8 output8_7 = (DATA_T8)(0.0f); \
            DATA_T8 weight8; \
            DATA_T8 input8; \
            DATA_T8 one8 = (DATA_T8)(1.0f); \
            unsigned int startA = get_global_id(0) * rowA * colA + \
                                  globalID1 * colA;  \
            unsigned int startB = batchId * rowB * colA * groupNum + \
                                  get_global_id(0) * rowB * colA + \
                                  get_global_id(2) * colA;  \
            unsigned int startC = batchId * rowA * rowB * groupNum + \
                                  get_global_id(0) * rowA * rowB + globalID1 * rowB + \
                                  get_global_id(2); \
            for (int i = 0; i < (colA >> 3); i++) { \
                input8 = vload8(0, B + startB + (i << 3)); \
                weight8 = vload8(0, A + startA + (i << 3)); \
                output8_0 += input8 * weight8; \
                weight8 = vload8(0, A + startA + (i << 3) + colA); \
                output8_1 += input8 * weight8; \
                weight8 = vload8(0, A + startA + (i << 3) + colA * 2); \
                output8_2 += input8 * weight8; \
                weight8 = vload8(0, A + startA + (i << 3) + colA * 3); \
                output8_3 += input8 * weight8; \
                weight8 = vload8(0, A + startA + (i << 3) + colA * 4); \
                output8_4 += input8 * weight8; \
                weight8 = vload8(0, A + startA + (i << 3) + colA * 5); \
                output8_5 += input8 * weight8; \
                weight8 = vload8(0, A + startA + (i << 3) + colA * 6); \
                output8_6 += input8 * weight8; \
                weight8 = vload8(0, A + startA + (i << 3) + colA * 7); \
                output8_7 += input8 * weight8; \
            } \
            if (globalID1 + 8 <= rowA) { \
                output[startC] = dot8(output8_0, one8); \
                output[startC + rowB] = dot8(output8_1, one8); \
                output[startC + rowB * 2] = dot8(output8_2, one8); \
                output[startC + rowB * 3] = dot8(output8_3, one8); \
                output[startC + rowB * 4] = dot8(output8_4, one8); \
                output[startC + rowB * 5] = dot8(output8_5, one8); \
                output[startC + rowB * 6] = dot8(output8_6, one8); \
                output[startC + rowB * 7] = dot8(output8_7, one8); \
            } else { \
                int num = rowA - globalID1; \
                if (num == 1) { \
                    output[startC] = dot8(output8_0, one8); \
                } else if (num == 2) { \
                    output[startC] = dot8(output8_0, one8); \
                    output[startC + rowB] = dot8(output8_1, one8); \
                } else if (num == 3) { \
                    output[startC] = dot8(output8_0, one8); \
                    output[startC + rowB] = dot8(output8_1, one8); \
                    output[startC + rowB * 2] = dot8(output8_2, one8); \
                } else if (num == 4) { \
                    output[startC] = dot8(output8_0, one8); \
                    output[startC + rowB] = dot8(output8_1, one8); \
                    output[startC + rowB * 2] = dot8(output8_2, one8); \
                    output[startC + rowB * 3] = dot8(output8_3, one8); \
                } else if (num == 5) { \
                    output[startC] = dot8(output8_0, one8); \
                    output[startC + rowB] = dot8(output8_1, one8); \
                    output[startC + rowB * 2] = dot8(output8_2, one8); \
                    output[startC + rowB * 3] = dot8(output8_3, one8); \
                    output[startC + rowB * 4] = dot8(output8_4, one8); \
                } else if (num == 6) { \
                    output[startC] = dot8(output8_0, one8); \
                    output[startC + rowB] = dot8(output8_1, one8); \
                    output[startC + rowB * 2] = dot8(output8_2, one8); \
                    output[startC + rowB * 3] = dot8(output8_3, one8); \
                    output[startC + rowB * 4] = dot8(output8_4, one8); \
                    output[startC + rowB * 5] = dot8(output8_5, one8); \
                } else if (num == 7) { \
                    output[startC] = dot8(output8_0, one8); \
                    output[startC + rowB] = dot8(output8_1, one8); \
                    output[startC + rowB * 2] = dot8(output8_2, one8); \
                    output[startC + rowB * 3] = dot8(output8_3, one8); \
                    output[startC + rowB * 4] = dot8(output8_4, one8); \
                    output[startC + rowB * 5] = dot8(output8_5, one8); \
                    output[startC + rowB * 6] = dot8(output8_6, one8); \
                } \
            } \
        }

#define CONVBACKGEMMRXR_FP32(A, B, output, rowA, colA, rowB, batchId, groupNum) \
        int globalID0 = get_global_id(0); /* group */ \
        int globalID1 = \
            get_global_id(1) * 8;         /* rowA weight K*K*topC/group each thread load 8 line */ \
        int globalID2 = get_global_id(2); /* rowB */ \
        if (globalID2 < rowB && globalID1 < rowA) { \
            DATA_T4 output8_0 = (DATA_T4)(0.0f); \
            DATA_T4 output8_1 = (DATA_T4)(0.0f); \
            DATA_T4 output8_2 = (DATA_T4)(0.0f); \
            DATA_T4 output8_3 = (DATA_T4)(0.0f); \
            DATA_T4 output8_4 = (DATA_T4)(0.0f); \
            DATA_T4 output8_5 = (DATA_T4)(0.0f); \
            DATA_T4 output8_6 = (DATA_T4)(0.0f); \
            DATA_T4 output8_7 = (DATA_T4)(0.0f); \
            DATA_T4 weight8; \
            DATA_T4 input8; \
            unsigned int startA = get_global_id(0) * rowA * colA + \
                                  globalID1 * colA;  \
            unsigned int startB = batchId * rowB * colA * groupNum + \
                                  get_global_id(0) * rowB * colA + \
                                  get_global_id(2) * colA;  \
            unsigned int startC = batchId * rowA * rowB * groupNum + \
                                  get_global_id(0) * rowA * rowB + globalID1 * rowB + \
                                  get_global_id(2); \
            for (int i = 0; i < (colA >> 2); i++) { \
                input8 = vload4(0, B + startB + (i << 2)); \
                weight8 = vload4(0, A + startA + (i << 2)); \
                output8_0 += input8 * weight8; \
                weight8 = vload4(0, A + startA + (i << 2) + colA); \
                output8_1 += input8 * weight8; \
                weight8 = vload4(0, A + startA + (i << 2) + colA * 2); \
                output8_2 += input8 * weight8; \
                weight8 = vload4(0, A + startA + (i << 2) + colA * 3); \
                output8_3 += input8 * weight8; \
                weight8 = vload4(0, A + startA + (i << 2) + colA * 4); \
                output8_4 += input8 * weight8; \
                weight8 = vload4(0, A + startA + (i << 2) + colA * 5); \
                output8_5 += input8 * weight8; \
                weight8 = vload4(0, A + startA + (i << 2) + colA * 6); \
                output8_6 += input8 * weight8; \
                weight8 = vload4(0, A + startA + (i << 2) + colA * 7); \
                output8_7 += input8 * weight8; \
            } \
            if (globalID1 + 8 <= rowA) { \
                output[startC] = output8_0.s0 + output8_0.s1 + output8_0.s2 + output8_0.s3; \
                output[startC + rowB] = output8_1.s0 + output8_1.s1 + output8_1.s2 + output8_1.s3; \
                output[startC + rowB * 2] = \
                    output8_2.s0 + output8_2.s1 + output8_2.s2 + output8_2.s3; \
                output[startC + rowB * 3] = \
                    output8_3.s0 + output8_3.s1 + output8_3.s2 + output8_3.s3; \
                output[startC + rowB * 4] = \
                    output8_4.s0 + output8_4.s1 + output8_4.s2 + output8_4.s3; \
                output[startC + rowB * 5] = \
                    output8_5.s0 + output8_5.s1 + output8_5.s2 + output8_5.s3; \
                output[startC + rowB * 6] = \
                    output8_6.s0 + output8_6.s1 + output8_6.s2 + output8_6.s3; \
                output[startC + rowB * 7] = \
                    output8_7.s0 + output8_7.s1 + output8_7.s2 + output8_7.s3; \
            } else { \
                int num = rowA - globalID1; \
                if (num == 1) { \
                    output[startC] = output8_0.s0 + output8_0.s1 + output8_0.s2 + output8_0.s3; \
                } else if (num == 2) { \
                    output[startC] = output8_0.s0 + output8_0.s1 + output8_0.s2 + output8_0.s3; \
                    output[startC + rowB] = \
                        output8_1.s0 + output8_1.s1 + output8_1.s2 + output8_1.s3; \
                } else if (num == 3) { \
                    output[startC] = output8_0.s0 + output8_0.s1 + output8_0.s2 + output8_0.s3; \
                    output[startC + rowB] = \
                        output8_1.s0 + output8_1.s1 + output8_1.s2 + output8_1.s3; \
                    output[startC + rowB * 2] = \
                        output8_2.s0 + output8_2.s1 + output8_2.s2 + output8_2.s3; \
                } else if (num == 4) { \
                    output[startC] = output8_0.s0 + output8_0.s1 + output8_0.s2 + output8_0.s3; \
                    output[startC + rowB] = \
                        output8_1.s0 + output8_1.s1 + output8_1.s2 + output8_1.s3; \
                    output[startC + rowB * 2] = \
                        output8_2.s0 + output8_2.s1 + output8_2.s2 + output8_2.s3; \
                    output[startC + rowB * 3] = \
                        output8_3.s0 + output8_3.s1 + output8_3.s2 + output8_3.s3; \
                } else if (num == 5) { \
                    output[startC] = output8_0.s0 + output8_0.s1 + output8_0.s2 + output8_0.s3; \
                    output[startC + rowB] = \
                        output8_1.s0 + output8_1.s1 + output8_1.s2 + output8_1.s3; \
                    output[startC + rowB * 2] = \
                        output8_2.s0 + output8_2.s1 + output8_2.s2 + output8_2.s3; \
                    output[startC + rowB * 3] = \
                        output8_3.s0 + output8_3.s1 + output8_3.s2 + output8_3.s3; \
                    output[startC + rowB * 4] = \
                        output8_4.s0 + output8_4.s1 + output8_4.s2 + output8_4.s3; \
                } else if (num == 6) { \
                    output[startC] = output8_0.s0 + output8_0.s1 + output8_0.s2 + output8_0.s3; \
                    output[startC + rowB] = \
                        output8_1.s0 + output8_1.s1 + output8_1.s2 + output8_1.s3; \
                    output[startC + rowB * 2] = \
                        output8_2.s0 + output8_2.s1 + output8_2.s2 + output8_2.s3; \
                    output[startC + rowB * 3] = \
                        output8_3.s0 + output8_3.s1 + output8_3.s2 + output8_3.s3; \
                    output[startC + rowB * 4] = \
                        output8_4.s0 + output8_4.s1 + output8_4.s2 + output8_4.s3; \
                    output[startC + rowB * 5] = \
                        output8_5.s0 + output8_5.s1 + output8_5.s2 + output8_5.s3; \
                } else if (num == 7) { \
                    output[startC] = output8_0.s0 + output8_0.s1 + output8_0.s2 + output8_0.s3; \
                    output[startC + rowB] = \
                        output8_1.s0 + output8_1.s1 + output8_1.s2 + output8_1.s3; \
                    output[startC + rowB * 2] = \
                        output8_2.s0 + output8_2.s1 + output8_2.s2 + output8_2.s3; \
                    output[startC + rowB * 3] = \
                        output8_3.s0 + output8_3.s1 + output8_3.s2 + output8_3.s3; \
                    output[startC + rowB * 4] = \
                        output8_4.s0 + output8_4.s1 + output8_4.s2 + output8_4.s3; \
                    output[startC + rowB * 5] = \
                        output8_5.s0 + output8_5.s1 + output8_5.s2 + output8_5.s3; \
                    output[startC + rowB * 6] = \
                        output8_6.s0 + output8_6.s1 + output8_6.s2 + output8_6.s3; \
                } \
            } \
        }

#define CONVERTBOTTOMDIFF2(input, bias, output, height, width, channels, kernelHeight, kernelWidth, padHeight, \
                           padWidth, strideHeight, strideWidth, height_col, width_col) \
        int globalID0 = get_global_id(0);  \
        int globalID1 = get_global_id(1);  \
        int globalID2 = get_global_id(2);  \
        if (globalID1 < height && globalID2 < width) { \
            DATA_T val = 0.0f; \
            int w_im = get_global_id(2) + padWidth; \
            int h_im = get_global_id(1) + padHeight; \
            int c_im = get_global_id(0); \
            int w_col_start = (w_im < kernelWidth) ? 0 : (w_im - kernelWidth) / strideWidth + 1; \
            int w_col_end = \
                (w_im / strideWidth + 1) < width_col ? (w_im / strideWidth + 1) : width_col; \
            int h_col_start = (h_im < kernelHeight) ? 0 : (h_im - kernelHeight) / strideHeight + 1; \
            int h_col_end = \
                (h_im / strideHeight + 1) < height_col ? (h_im / strideHeight + 1) : height_col; \
            for (int h_col = h_col_start; h_col < h_col_end; ++h_col) { \
                for (int w_col = w_col_start; w_col < w_col_end; ++w_col) { \
                    int c_col = c_im * kernelHeight * kernelWidth + \
                                (h_im - h_col * strideHeight) * kernelWidth + \
                                (w_im - w_col * strideWidth); \
                    val += input[(c_col * height_col + h_col) * width_col + w_col]; \
                } \
            } \
            int biasNum = get_global_id(0) % channels; \
            output[get_global_id(0) * height * width + get_global_id(1) * width + \
                   get_global_id(2)] = val + bias[biasNum]; \
        }

/********  FP16 KERNELS ********/
#define DATA_T half
#define DATA_T2 half2
#define DATA_T3 half3
#define DATA_T4 half4
#define DATA_T8 half8
#define DATA_T16 half16
ADD_SINGLE_KERNEL(matrixTrans_FP16, (__global const DATA_T *matrix,
                                   __global DATA_T *matrixTrans,
                                   unsigned int widthTransAlign,
                                   unsigned int batchId,
                                   unsigned int groupNum) {
    MATRIXTRANS(matrix, matrixTrans, widthTransAlign, batchId, groupNum)
})

ADD_SINGLE_KERNEL(convBackGemmRXR_FP16, (__global const DATA_T *A,
                                       __global const DATA_T *B,
                                       __global DATA_T *output,
                                       unsigned int rowA,
                                       unsigned int colA,
                                       unsigned int rowB,
                                       unsigned int batchId,
                                       unsigned int groupNum) {
    CONVBACKGEMMRXR_FP16(A, B, output, rowA, colA, rowB, batchId, groupNum)
})

ADD_SINGLE_KERNEL(convertBottomDiff2_FP16, (__global const DATA_T *input,
                                          __global const DATA_T *bias,
                                          __global DATA_T *output,
                                          unsigned int height,
                                          unsigned int width,
                                          unsigned int channels,
                                          unsigned int kernelHeight,
                                          unsigned int kernelWidth,
                                          unsigned int padHeight,
                                          unsigned int padWidth,
                                          unsigned int strideHeight,
                                          unsigned int strideWidth,
                                          unsigned int height_col,
                                          unsigned int width_col) {
    CONVERTBOTTOMDIFF2(input, bias, output, height, width, channels, kernelHeight, kernelWidth, padHeight, padWidth, \
                       strideHeight, strideWidth, height_col, width_col)
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

ADD_SINGLE_KERNEL(matrixTrans_FP32, (__global const DATA_T *matrix,
                                   __global DATA_T *matrixTrans,
                                   unsigned int widthTransAlign,
                                   unsigned int batchId,
                                   unsigned int groupNum) {
    MATRIXTRANS(matrix, matrixTrans, widthTransAlign, batchId, groupNum)
})

ADD_SINGLE_KERNEL(convBackGemmRXR_FP32, (__global const DATA_T *A,
                                       __global const DATA_T *B,
                                       __global DATA_T *output,
                                       unsigned int rowA,
                                       unsigned int colA,
                                       unsigned int rowB,
                                       unsigned int batchId,
                                       unsigned int groupNum) {
    CONVBACKGEMMRXR_FP32(A, B, output, rowA, colA, rowB, batchId, groupNum)
})

ADD_SINGLE_KERNEL(convertBottomDiff2_FP32, (__global const DATA_T *input,
                                          __global const DATA_T *bias,
                                          __global DATA_T *output,
                                          unsigned int height,
                                          unsigned int width,
                                          unsigned int channels,
                                          unsigned int kernelHeight,
                                          unsigned int kernelWidth,
                                          unsigned int padHeight,
                                          unsigned int padWidth,
                                          unsigned int strideHeight,
                                          unsigned int strideWidth,
                                          unsigned int height_col,
                                          unsigned int width_col) {
    CONVERTBOTTOMDIFF2(input, bias, output, height, width, channels, kernelHeight, kernelWidth, padHeight, padWidth, \
                       strideHeight, strideWidth, height_col, width_col)
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
