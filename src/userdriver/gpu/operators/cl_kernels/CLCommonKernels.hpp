/*
 * Copyright (C) 2020 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

/**
 * @file    CLCommonKernels.hpp
 * @brief
 * @details
 * @version
 */

#pragma once

#ifdef INFINITY
#undef INFINITY
#endif

#ifdef FLOAT_MIN
#undef FLOAT_MIN
#endif
#define FLOAT_MIN -99999.9999f

#ifdef HALF_MIN
#undef HALF_MIN
#endif
#define HALF_MIN -60000.0f

#ifdef INT_MIN
#undef INT_MIN
#endif
#define INT_MIN -60000

#ifdef CLK_NORMALIZED_COORDS_FALSE
#undef CLK_NORMALIZED_COORDS_FALSE
#endif
#ifdef CLK_ADDRESS_NONE
#undef CLK_ADDRESS_NONE
#endif
#ifdef CLK_FILTER_NEAREST
#undef CLK_FILTER_NEAREST
#endif
#ifdef CLK_ADDRESS_CLAMP
#undef CLK_ADDRESS_CLAMP
#endif

#ifdef smp_none
#undef smp_none
#endif
#define smp_none (CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST)

#ifdef smp_zero
#undef smp_zero
#endif
#define smp_zero (CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST)

#define IDX_IN_3D get_global_id(0) * get_global_size(1) * get_global_size(2) + \
                  get_global_id(1) * get_global_size(2) + get_global_id(2)

// To clear SAM duplicated code violations, we put common kernel defines here
#define FC_INTERLEAVE_8X1(input, output, oriRow, oriCol, dstCol) \
    int globalID0 = get_global_id(0);  \
    int globalID1 = get_global_id(1);  \
    if (globalID1 < oriCol) { \
        if (globalID0 * 8 + 7 < oriRow) { \
            int inputIndex = 8 * globalID0 * oriCol + globalID1; \
            int outIndex = globalID0 * dstCol + globalID1 * 8; \
            DATA_T8 temp8 = (DATA_T8)(0.0f); \
            temp8.s0 = input[inputIndex]; \
            temp8.s1 = input[inputIndex + oriCol]; \
            temp8.s2 = input[inputIndex + oriCol * 2]; \
            temp8.s3 = input[inputIndex + oriCol * 3]; \
            temp8.s4 = input[inputIndex + oriCol * 4]; \
            temp8.s5 = input[inputIndex + oriCol * 5]; \
            temp8.s6 = input[inputIndex + oriCol * 6]; \
            temp8.s7 = input[inputIndex + oriCol * 7]; \
            vstore8(temp8, 0, output + outIndex); \
        } else if (globalID0 * 8 + 6 < oriRow) { \
            int inputIndex = 8 * globalID0 * oriCol + globalID1; \
            int outIndex = globalID0 * dstCol + globalID1 * 8; \
            DATA_T8 temp8 = (DATA_T8)(0.0f); \
            temp8.s0 = input[inputIndex]; \
            temp8.s1 = input[inputIndex + oriCol]; \
            temp8.s2 = input[inputIndex + oriCol * 2]; \
            temp8.s3 = input[inputIndex + oriCol * 3]; \
            temp8.s4 = input[inputIndex + oriCol * 4]; \
            temp8.s5 = input[inputIndex + oriCol * 5]; \
            temp8.s6 = input[inputIndex + oriCol * 6]; \
            output[outIndex] = temp8.s0; \
            output[outIndex + 1] = temp8.s1; \
            output[outIndex + 2] = temp8.s2; \
            output[outIndex + 3] = temp8.s3; \
            output[outIndex + 4] = temp8.s4; \
            output[outIndex + 5] = temp8.s5; \
            output[outIndex + 6] = temp8.s6; \
        } else if (globalID0 * 8 + 5 < oriRow) { \
            int inputIndex = 8 * globalID0 * oriCol + globalID1; \
            int outIndex = globalID0 * dstCol + globalID1 * 8; \
            DATA_T8 temp8 = (DATA_T8)(0.0f); \
            temp8.s0 = input[inputIndex]; \
            temp8.s1 = input[inputIndex + oriCol]; \
            temp8.s2 = input[inputIndex + oriCol * 2]; \
            temp8.s3 = input[inputIndex + oriCol * 3]; \
            temp8.s4 = input[inputIndex + oriCol * 4]; \
            temp8.s5 = input[inputIndex + oriCol * 5]; \
            output[outIndex] = temp8.s0; \
            output[outIndex + 1] = temp8.s1; \
            output[outIndex + 2] = temp8.s2; \
            output[outIndex + 3] = temp8.s3; \
            output[outIndex + 4] = temp8.s4; \
            output[outIndex + 5] = temp8.s5; \
        } else if (globalID0 * 8 + 4 < oriRow) { \
            int inputIndex = 8 * globalID0 * oriCol + globalID1; \
            int outIndex = globalID0 * dstCol + globalID1 * 8; \
            DATA_T8 temp8 = (DATA_T8)(0.0f); \
            temp8.s0 = input[inputIndex]; \
            temp8.s1 = input[inputIndex + oriCol]; \
            temp8.s2 = input[inputIndex + oriCol * 2]; \
            temp8.s3 = input[inputIndex + oriCol * 3]; \
            temp8.s4 = input[inputIndex + oriCol * 4]; \
            output[outIndex] = temp8.s0; \
            output[outIndex + 1] = temp8.s1; \
            output[outIndex + 2] = temp8.s2; \
            output[outIndex + 3] = temp8.s3; \
            output[outIndex + 4] = temp8.s4; \
        } else if (globalID0 * 8 + 3 < oriRow) { \
            int inputIndex = 8 * globalID0 * oriCol + globalID1; \
            int outIndex = globalID0 * dstCol + globalID1 * 8; \
            DATA_T8 temp8 = (DATA_T8)(0.0f); \
            temp8.s0 = input[inputIndex]; \
            temp8.s1 = input[inputIndex + oriCol]; \
            temp8.s2 = input[inputIndex + oriCol * 2]; \
            temp8.s3 = input[inputIndex + oriCol * 3]; \
            output[outIndex] = temp8.s0; \
            output[outIndex + 1] = temp8.s1; \
            output[outIndex + 2] = temp8.s2; \
            output[outIndex + 3] = temp8.s3; \
        } else if (globalID0 * 8 + 2 < oriRow) { \
            int inputIndex = 8 * globalID0 * oriCol + globalID1; \
            int outIndex = globalID0 * dstCol + globalID1 * 8; \
            DATA_T8 temp8 = (DATA_T8)(0.0f); \
            temp8.s0 = input[inputIndex]; \
            temp8.s1 = input[inputIndex + oriCol]; \
            temp8.s2 = input[inputIndex + oriCol * 2]; \
            output[outIndex] = temp8.s0; \
            output[outIndex + 1] = temp8.s1; \
            output[outIndex + 2] = temp8.s2; \
        } else if (globalID0 * 8 + 1 < oriRow) { \
            int inputIndex = 8 * globalID0 * oriCol + globalID1; \
            int outIndex = globalID0 * dstCol + globalID1 * 8; \
            DATA_T8 temp8 = (DATA_T8)(0.0f); \
            temp8.s0 = input[inputIndex]; \
            temp8.s1 = input[inputIndex + oriCol]; \
            output[outIndex] = temp8.s0; \
            output[outIndex + 1] = temp8.s1; \
        } else if (globalID0 * 8 < oriRow) { \
            int inputIndex = 8 * globalID0 * oriCol + globalID1; \
            int outIndex = globalID0 * dstCol + globalID1 * 8; \
            DATA_T8 temp8 = (DATA_T8)(0.0f); \
            temp8.s0 = input[inputIndex]; \
            output[outIndex] = temp8.s0; \
        } \
    }

#define LOGICAL_AND_OR_COMPUTE_INDEX(out_w, n0, c0, h0, w0, n1, c1, h1, w1) \
    int out_n_index = get_global_id(0); \
    int out_c_index = get_global_id(1); \
    int out_h_index = get_global_id(2) / out_w; \
    int out_w_index = get_global_id(2) % out_w; \
    int outIndex = out_n_index * get_global_size(1) * get_global_size(2) + \
                   out_c_index * get_global_size(2) + get_global_id(2); \
    int n0_idx = out_n_index; \
    int c0_idx = out_c_index; \
    int h0_idx = out_h_index; \
    int w0_idx = out_w_index; \
    int n1_idx = out_n_index; \
    int c1_idx = out_c_index; \
    int h1_idx = out_h_index; \
    int w1_idx = out_w_index; \
    if (out_n_index >= n0) { \
        n0_idx = n0 - 1; \
    } \
    if (out_c_index >= c0) { \
        c0_idx = c0 - 1; \
    } \
    if (out_h_index >= h0) { \
        h0_idx = h0 - 1; \
    } \
    if (out_w_index >= w0) { \
        w0_idx = w0 - 1; \
    } \
    if (out_n_index >= n1) { \
        n1_idx = n1 - 1; \
    } \
    if (out_c_index >= c1) { \
        c1_idx = c1 - 1; \
    } \
    if (out_h_index >= h1) { \
        h1_idx = h1 - 1; \
    } \
    if (out_w_index >= w1) { \
        w1_idx = w1 - 1; \
    } \
    int inIndex0 = n0_idx * c0 * h0 * w0 + c0_idx * h0 * w0 + h0_idx * w0 + w0_idx; \
    int inIndex1 = n1_idx * c1 * h1 * w1 + c1_idx * h1 * w1 + h1_idx * w1 + w1_idx;

#define COMPUTE_TRANSPOSE_OUT_INDEX(i_index, perm_n_nchw, perm_c_nchw, perm_h_nchw, perm_w_nchw, inputChannel, \
                                    inputHeight, inputWidth, input_chw, outputChannel, outputHeight, outputWidth, \
                                    o_index) \
    int o_nchw_offset[4]; \
    int input_hw = inputWidth * inputHeight; \
    o_nchw_offset[3] = i_index % inputWidth; \
    o_nchw_offset[2] = (i_index / inputWidth) % inputHeight; \
    o_nchw_offset[1] = (i_index / input_hw) % inputChannel; \
    o_nchw_offset[0] = (i_index / input_chw); \
    o_index.s0 = o_nchw_offset[perm_n_nchw] * outputChannel * outputHeight * outputWidth + \
                    o_nchw_offset[perm_c_nchw] * outputHeight * outputWidth + \
                    o_nchw_offset[perm_h_nchw] * outputWidth + o_nchw_offset[perm_w_nchw]; \
    o_nchw_offset[3] = (i_index + 1) % inputWidth; \
    o_nchw_offset[2] = ((i_index + 1) / inputWidth) % inputHeight; \
    o_nchw_offset[1] = ((i_index + 1) / input_hw) % inputChannel; \
    o_nchw_offset[0] = ((i_index + 1) / input_chw); \
    o_index.s1 = o_nchw_offset[perm_n_nchw] * outputChannel * outputHeight * outputWidth + \
                    o_nchw_offset[perm_c_nchw] * outputHeight * outputWidth + \
                    o_nchw_offset[perm_h_nchw] * outputWidth + o_nchw_offset[perm_w_nchw]; \
    o_nchw_offset[3] = (i_index + 2) % inputWidth; \
    o_nchw_offset[2] = ((i_index + 2) / inputWidth) % inputHeight; \
    o_nchw_offset[1] = ((i_index + 2) / input_hw) % inputChannel; \
    o_nchw_offset[0] = ((i_index + 2) / input_chw); \
    o_index.s2 = o_nchw_offset[perm_n_nchw] * outputChannel * outputHeight * outputWidth + \
                    o_nchw_offset[perm_c_nchw] * outputHeight * outputWidth + \
                    o_nchw_offset[perm_h_nchw] * outputWidth + o_nchw_offset[perm_w_nchw]; \
    o_nchw_offset[3] = (i_index + 3) % inputWidth; \
    o_nchw_offset[2] = ((i_index + 3) / inputWidth) % inputHeight; \
    o_nchw_offset[1] = ((i_index + 3) / input_hw) % inputChannel; \
    o_nchw_offset[0] = ((i_index + 3) / input_chw); \
    o_index.s3 = o_nchw_offset[perm_n_nchw] * outputChannel * outputHeight * outputWidth + \
                    o_nchw_offset[perm_c_nchw] * outputHeight * outputWidth + \
                    o_nchw_offset[perm_h_nchw] * outputWidth + o_nchw_offset[perm_w_nchw]; \
    o_nchw_offset[3] = (i_index + 4) % inputWidth; \
    o_nchw_offset[2] = ((i_index + 4) / inputWidth) % inputHeight; \
    o_nchw_offset[1] = ((i_index + 4) / input_hw) % inputChannel; \
    o_nchw_offset[0] = ((i_index + 4) / input_chw); \
    o_index.s4 = o_nchw_offset[perm_n_nchw] * outputChannel * outputHeight * outputWidth + \
                    o_nchw_offset[perm_c_nchw] * outputHeight * outputWidth + \
                    o_nchw_offset[perm_h_nchw] * outputWidth + o_nchw_offset[perm_w_nchw]; \
    o_nchw_offset[3] = (i_index + 5) % inputWidth; \
    o_nchw_offset[2] = ((i_index + 5) / inputWidth) % inputHeight; \
    o_nchw_offset[1] = ((i_index + 5) / input_hw) % inputChannel; \
    o_nchw_offset[0] = ((i_index + 5) / input_chw); \
    o_index.s5 = o_nchw_offset[perm_n_nchw] * outputChannel * outputHeight * outputWidth + \
                    o_nchw_offset[perm_c_nchw] * outputHeight * outputWidth + \
                    o_nchw_offset[perm_h_nchw] * outputWidth + o_nchw_offset[perm_w_nchw]; \
    o_nchw_offset[3] = (i_index + 6) % inputWidth; \
    o_nchw_offset[2] = ((i_index + 6) / inputWidth) % inputHeight; \
    o_nchw_offset[1] = ((i_index + 6) / input_hw) % inputChannel; \
    o_nchw_offset[0] = ((i_index + 6) / input_chw); \
    o_index.s6 = o_nchw_offset[perm_n_nchw] * outputChannel * outputHeight * outputWidth + \
                    o_nchw_offset[perm_c_nchw] * outputHeight * outputWidth + \
                    o_nchw_offset[perm_h_nchw] * outputWidth + o_nchw_offset[perm_w_nchw]; \
    o_nchw_offset[3] = (i_index + 7) % inputWidth; \
    o_nchw_offset[2] = ((i_index + 7) / inputWidth) % inputHeight; \
    o_nchw_offset[1] = ((i_index + 7) / input_hw) % inputChannel; \
    o_nchw_offset[0] = ((i_index + 7) / input_chw); \
    o_index.s7 = o_nchw_offset[perm_n_nchw] * outputChannel * outputHeight * outputWidth + \
                    o_nchw_offset[perm_c_nchw] * outputHeight * outputWidth + \
                    o_nchw_offset[perm_h_nchw] * outputWidth + o_nchw_offset[perm_w_nchw];

#define FEED_TRANSPOSE_OUTPUT(input_vector, i_index, inputBatch, input_chw, o_index, output) \
    output[o_index.s0] = input_vector.s0; \
    if (i_index + 1 < inputBatch * input_chw) { \
        output[o_index.s1] = input_vector.s1; \
    } \
    if (i_index + 2 < inputBatch * input_chw) { \
        output[o_index.s2] = input_vector.s2; \
    } \
    if (i_index + 3 < inputBatch * input_chw) { \
        output[o_index.s3] = input_vector.s3; \
    } \
    if (i_index + 4 < inputBatch * input_chw) { \
        output[o_index.s4] = input_vector.s4; \
    } \
    if (i_index + 5 < inputBatch * input_chw) { \
        output[o_index.s5] = input_vector.s5; \
    } \
    if (i_index + 6 < inputBatch * input_chw) { \
        output[o_index.s6] = input_vector.s6; \
    } \
    if (i_index + 7 < inputBatch * input_chw) { \
        output[o_index.s7] = input_vector.s7; \
    }


#define TRANSPOSE(input, output, perm_n_nchw, perm_c_nchw, perm_h_nchw, perm_w_nchw, inputBatch, \
                  inputChannel, inputHeight, inputWidth, outputChannel, outputHeight, outputWidth) \
    int i_index = get_global_id(0) * 8; \
    if (i_index < inputBatch * inputChannel * inputHeight * inputWidth) { \
        DATA_T8 input_vector = vload8(0, input + i_index); \
        int8 o_index; \
        int input_chw = inputChannel * inputWidth * inputHeight; \
        COMPUTE_TRANSPOSE_OUT_INDEX(i_index, perm_n_nchw, perm_c_nchw, perm_h_nchw, perm_w_nchw, inputChannel, \
                                    inputHeight, inputWidth, input_chw, outputChannel, outputHeight, outputWidth, \
                                    o_index) \
        FEED_TRANSPOSE_OUTPUT(input_vector, i_index, inputBatch, input_chw, o_index, output) \
    }


#define RELU_X(input, output, channel, height, width) \
    int globalID0 = get_global_id(0); \
    int globalID1 = get_global_id(1); \
    int globalID2 = get_global_id(2) * 8; \
    int wh = width * height; \
    if (globalID1 < channel && globalID2 < wh) { \
        int base = globalID0 * channel * wh + globalID1 * wh + globalID2; \
        if (globalID2 + 8 <= wh) { \
            vstore8(ACT_VEC_F(DATA_T8, vload8(0, input + base)), 0, output + base); \
        } else { \
            int num = wh - globalID2; \
            if (num == 1) { \
                output[base] = ACT_VEC_F(DATA_T, input[base]); \
            } else if (num == 2) { \
                vstore2(ACT_VEC_F(DATA_T2, vload2(0, input + base)), 0, output + base); \
            } else if (num == 3) { \
                vstore3(ACT_VEC_F(DATA_T3, vload3(0, input + base)), 0, output + base); \
            } else if (num == 4) { \
                vstore4(ACT_VEC_F(DATA_T4, vload4(0, input + base)), 0, output + base); \
            } else if (num == 5) { \
                vstore4(ACT_VEC_F(DATA_T4, vload4(0, input + base)), 0, output + base); \
                output[base + 4] = ACT_VEC_F(DATA_T, input[base + 4]); \
            } else if (num == 6) { \
                vstore4(ACT_VEC_F(DATA_T4, vload4(0, input + base)), 0, output + base); \
                vstore2(ACT_VEC_F(DATA_T2, vload2(0, input + base + 4)), 0, output + base + 4); \
            } else if (num == 7) { \
                vstore4(ACT_VEC_F(DATA_T4, vload4(0, input + base)), 0, output + base); \
                vstore3(ACT_VEC_F(DATA_T3, vload3(0, input + base + 4)), 0, output + base + 4); \
            } \
        } \
    }

#define Q_RELU_X(input, output, min, max) \
    int index = get_global_id(0); \
    int res = input[index] < max ? input[index] : max; \
    output[index] = res > min ? res : min;

#define RELU_VEC_F(VEC_T, x) fmax(x, (VEC_T)(0.0f))
#define RELU1_VEC_F(VEC_T, x) fmin(fmax(x, (VEC_T)(-1.0f)), (VEC_T)(1.0f))
#define RELU6_VEC_F(VEC_T, x) fmin(fmax(x, (VEC_T)(0.0f)), (VEC_T)(6.0f))

#define RELU_CLAMP_VEC_F(DATA_T, x) clamp(x, (DATA_T)0.0f, (DATA_T)INFINITY)
#define RELU6_CLAMP_VEC_F(DATA_T, x) clamp(x, (DATA_T)0.0f, (DATA_T)6.0f)

#define MERGEADD_IMAGE2D(x, IMAGE, COORD) x += read_imageh(IMAGE, smp_zero, COORD)
#define MERGEADD_INPUTS __read_only image2d_t src_data, __read_only image2d_t src_data_1
#define SINGLE_SRC_DATA __read_only image2d_t src_data
