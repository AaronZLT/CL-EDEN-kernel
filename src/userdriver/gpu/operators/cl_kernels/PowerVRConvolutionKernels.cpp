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
 * @file    PowerVRConvolutionKernels.cpp
 * @brief
 * @details
 * @version
 */

#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"
namespace enn{
namespace ud {
namespace gpu {

#define ALIGNWEIGHT_POWERVR(src, dst, batch, channel, kh, kw, dst_groups, src_slices, out_group_size)                  \
    int g0 = get_global_id(0);                                                                                         \
    int g1 = get_global_id(1);                                                                                         \
    int g2 = get_global_id(2);                                                                                         \
    if (g0 >= dst_groups || g1 >= src_slices || g2 >= kh * kw)                                                         \
        return;                                                                                                        \
    for (int d_group = 0; d_group < out_group_size; ++d_group) {                                                       \
        for (int j = 0; j < 4; ++j) {                                                                                  \
            DATA_T filter[4];                                                                                          \
            int dst_index = 4 * (g0 * kh * kw * src_slices * out_group_size * 4 +                                      \
                                 g2 * src_slices * out_group_size * 4 + g1 * out_group_size * 4 + d_group * 4 + j);    \
            for (int i = 0; i < 4; ++i) {                                                                              \
                const int s_ch = g1 * 4 + j;                                                                           \
                const int d_ch = (g0 * out_group_size + d_group) * 4 + i;                                              \
                if (s_ch < channel && d_ch < batch) {                                                                  \
                    const int f_index = d_ch * channel * kh * kw + s_ch * kh * kw + g2;                                \
                    filter[i] = src[f_index];                                                                          \
                } else {                                                                                               \
                    filter[i] = 0.0f;                                                                                  \
                }                                                                                                      \
            }                                                                                                          \
            for (int k = 0; k < 4; k++) {                                                                              \
                dst[dst_index + k] = filter[k];                                                                        \
            }                                                                                                          \
        }                                                                                                              \
    }

#define POWERVR_IS1X11_SRCDEPTH1_BLOCK111(                                                                             \
    src_data, src_data_1, filters_buffer, biases, dst_data, src_x, src_y, src_z, src_w, dst_x, dst_y, dst_z, dst_w)    \
    int X = get_global_id(0) * 1;                                                                                      \
    int Y = get_global_id(1) * 1;                                                                                      \
    int Z = get_global_id(2) * 1;                                                                                      \
    if (X >= dst_x || Y >= dst_y || Z >= dst_z) {                                                                      \
        return;                                                                                                        \
    }                                                                                                                  \
    DATA_T4 r000 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    __global DATA_T4 *bias_cache;                                                                                      \
    int weight_offset = 4 * 4 * Z * src_z;                                                                             \
    int s = 0;                                                                                                         \
    do {                                                                                                               \
        DATA_T4 src00;                                                                                                 \
        DATA_T16 weights = vload16(0, filters_buffer + weight_offset);                                                 \
        src00 = read_imageh(src_data, smp_zero, (int2)(((X + 0)), ((Y + 0)) * src_z + (s)));                           \
        s += 1;                                                                                                        \
        r000 = fma(weights.lo.lo, src00.x, r000);                                                                      \
        r000 = fma(weights.lo.hi, src00.y, r000);                                                                      \
        r000 = fma(weights.hi.lo, src00.z, r000);                                                                      \
        r000 = fma(weights.hi.hi, src00.w, r000);                                                                      \
        weight_offset += 16;                                                                                           \
    } while (s < src_z);                                                                                               \
    bias_cache = biases + Z;                                                                                           \
    if (Z + 0 >= dst_z)                                                                                                \
        return;                                                                                                        \
    {                                                                                                                  \
        DATA_T4 bias_val = CONVERT_TO_DATA_T4(bias_cache[0]);                                                          \
        {                                                                                                              \
            DATA_T4 res = CONVERT_TO_DATA_T4(r000) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
    }

#define POWERVR_IS1X11_SRCDEPTH1_BLOCK211(                                                                             \
    src_data, src_data_1, filters_buffer, biases, dst_data, src_x, src_y, src_z, src_w, dst_x, dst_y, dst_z, dst_w)    \
    int X = get_global_id(0) * 2;                                                                                      \
    int Y = get_global_id(1) * 1;                                                                                      \
    int Z = get_global_id(2) * 1;                                                                                      \
    if (X >= dst_x || Y >= dst_y || Z >= dst_z) {                                                                      \
        return;                                                                                                        \
    }                                                                                                                  \
    DATA_T4 r000 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r001 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    __global DATA_T4 *bias_cache;                                                                                      \
    int weight_offset = 4 * 4 * Z * src_z;                                                                             \
    int s = 0;                                                                                                         \
    do {                                                                                                               \
        DATA_T4 src00;                                                                                                 \
        DATA_T4 src01;                                                                                                 \
        DATA_T16 weights_cache0123 = vload16(0, filters_buffer + weight_offset);                                       \
        src00 = read_imageh(src_data, smp_zero, (int2)(((X + 0)), ((Y + 0)) * src_z + (s)));                           \
        src01 = read_imageh(src_data, smp_zero, (int2)(((X + 1)), ((Y + 0)) * src_z + (s)));                           \
        s += 1;                                                                                                        \
        r000 = fma(weights_cache0123.lo.lo, src00.x, r000);                                                            \
        r001 = fma(weights_cache0123.lo.lo, src01.x, r001);                                                            \
        r000 = fma(weights_cache0123.lo.hi, src00.y, r000);                                                            \
        r001 = fma(weights_cache0123.lo.hi, src01.y, r001);                                                            \
        r000 = fma(weights_cache0123.hi.lo, src00.z, r000);                                                            \
        r001 = fma(weights_cache0123.hi.lo, src01.z, r001);                                                            \
        r000 = fma(weights_cache0123.hi.hi, src00.w, r000);                                                            \
        r001 = fma(weights_cache0123.hi.hi, src01.w, r001);                                                            \
        weight_offset += 16;                                                                                           \
    } while (s < src_z);                                                                                               \
    bias_cache = biases + Z;                                                                                           \
    if (Z + 0 >= dst_z)                                                                                                \
        return;                                                                                                        \
    {                                                                                                                  \
        DATA_T4 bias_val = CONVERT_TO_DATA_T4(bias_cache[0]);                                                          \
        {                                                                                                              \
            DATA_T4 res = CONVERT_TO_DATA_T4(r000) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
        if ((X + 1) < dst_x) {                                                                                         \
            DATA_T4 res = CONVERT_TO_DATA_T4(r001) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
    }

#define POWERVR_IS1X11_SRCDEPTH1_BLOCK212(                                                                             \
    src_data, src_data_1, filters_buffer, biases, dst_data, src_x, src_y, src_z, src_w, dst_x, dst_y, dst_z, dst_w)    \
    int X = get_global_id(0) * 2;                                                                                      \
    int Y = get_global_id(1) * 1;                                                                                      \
    int Z = get_global_id(2) * 2;                                                                                      \
    if (X >= dst_x || Y >= dst_y || Z >= dst_z) {                                                                      \
        return;                                                                                                        \
    }                                                                                                                  \
    DATA_T4 r000 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r001 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r100 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r101 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    __global DATA_T4 *bias_cache;                                                                                      \
    int s = 0;                                                                                                         \
    int weight_offset = Z * 4 * src_z * 4;                                                                             \
    do {                                                                                                               \
        DATA_T4 src00;                                                                                                 \
        DATA_T4 src01;                                                                                                 \
        DATA_T16 weights_cache0123 = vload16(0, filters_buffer + weight_offset);                                       \
        src00 = read_imageh(src_data, smp_zero, (int2)(((X + 0)), ((Y + 0)) * src_z + (s)));                           \
        DATA_T16 weights_cache4567 = vload16(0, filters_buffer + weight_offset + 16);                                  \
        src01 = read_imageh(src_data, smp_zero, (int2)(((X + 1)), ((Y + 0)) * src_z + (s)));                           \
        s += 1;                                                                                                        \
        r000 = fma(weights_cache0123.lo.lo, src00.x, r000);                                                            \
        r001 = fma(weights_cache0123.lo.lo, src01.x, r001);                                                            \
        r000 = fma(weights_cache0123.lo.hi, src00.y, r000);                                                            \
        r001 = fma(weights_cache0123.lo.hi, src01.y, r001);                                                            \
        r000 = fma(weights_cache0123.hi.lo, src00.z, r000);                                                            \
        r001 = fma(weights_cache0123.hi.lo, src01.z, r001);                                                            \
        r000 = fma(weights_cache0123.hi.hi, src00.w, r000);                                                            \
        r001 = fma(weights_cache0123.hi.hi, src01.w, r001);                                                            \
        r100 = fma(weights_cache4567.lo.lo, src00.x, r100);                                                            \
        r101 = fma(weights_cache4567.lo.lo, src01.x, r101);                                                            \
        r100 = fma(weights_cache4567.lo.hi, src00.y, r100);                                                            \
        r101 = fma(weights_cache4567.lo.hi, src01.y, r101);                                                            \
        r100 = fma(weights_cache4567.hi.lo, src00.z, r100);                                                            \
        r101 = fma(weights_cache4567.hi.lo, src01.z, r101);                                                            \
        r100 = fma(weights_cache4567.hi.hi, src00.w, r100);                                                            \
        r101 = fma(weights_cache4567.hi.hi, src01.w, r101);                                                            \
        weight_offset += 32;                                                                                           \
    } while (s < src_z);                                                                                               \
    bias_cache = biases + Z;                                                                                           \
    if (Z + 0 >= dst_z)                                                                                                \
        return;                                                                                                        \
    {                                                                                                                  \
        DATA_T4 bias_val = CONVERT_TO_DATA_T4(bias_cache[0]);                                                          \
        {                                                                                                              \
            DATA_T4 res = CONVERT_TO_DATA_T4(r000) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
        if ((X + 1) < dst_x) {                                                                                         \
            DATA_T4 res = CONVERT_TO_DATA_T4(r001) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
    }                                                                                                                  \
    if (Z + 1 >= dst_z)                                                                                                \
        return;                                                                                                        \
    {                                                                                                                  \
        DATA_T4 bias_val = CONVERT_TO_DATA_T4(bias_cache[1]);                                                          \
        {                                                                                                              \
            DATA_T4 res = CONVERT_TO_DATA_T4(r100) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 1)));                                 \
            write_imageh(dst_data, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 1)), res);                               \
        }                                                                                                              \
        if ((X + 1) < dst_x) {                                                                                         \
            DATA_T4 res = CONVERT_TO_DATA_T4(r101) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 1)));                                 \
            write_imageh(dst_data, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 1)), res);                               \
        }                                                                                                              \
    }

#define POWERVR_IS1X11_SRCDEPTH1_BLOCK222(                                                                             \
    src_data, src_data_1, filters_buffer, biases, dst_data, src_x, src_y, src_z, src_w, dst_x, dst_y, dst_z, dst_w)    \
    int X = get_global_id(0) * 2;                                                                                      \
    int Y = get_global_id(1) * 2;                                                                                      \
    int Z = get_global_id(2) * 2;                                                                                      \
    if (X >= dst_x || Y >= dst_y || Z >= dst_z) {                                                                      \
        return;                                                                                                        \
    }                                                                                                                  \
    DATA_T4 r000 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r001 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r010 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r011 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r100 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r101 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r110 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r111 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    __global DATA_T16 *weights_cache;                                                                                  \
    __global DATA_T4 *bias_cache;                                                                                      \
    __global DATA_T16 *filters_loc = filters_buffer + Z * src_z;                                                       \
    int s = 0;                                                                                                         \
    do {                                                                                                               \
        DATA_T4 src00;                                                                                                 \
        DATA_T4 src01;                                                                                                 \
        DATA_T4 src10;                                                                                                 \
        DATA_T4 src11;                                                                                                 \
        weights_cache = filters_loc;                                                                                   \
        src00 = read_imageh(src_data, smp_zero, (int2)(((X + 0)), ((Y + 0)) * src_z + (s)));                           \
        src01 = read_imageh(src_data, smp_zero, (int2)(((X + 1)), ((Y + 0)) * src_z + (s)));                           \
        src10 = read_imageh(src_data, smp_zero, (int2)(((X + 0)), ((Y + 1)) * src_z + (s)));                           \
        src11 = read_imageh(src_data, smp_zero, (int2)(((X + 1)), ((Y + 1)) * src_z + (s)));                           \
        s += 1;                                                                                                        \
        r000 = fma(weights_cache[0].lo.lo, src00.x, r000);                                                             \
        r001 = fma(weights_cache[0].lo.lo, src01.x, r001);                                                             \
        r010 = fma(weights_cache[0].lo.lo, src10.x, r010);                                                             \
        r011 = fma(weights_cache[0].lo.lo, src11.x, r011);                                                             \
        r000 = fma(weights_cache[0].lo.hi, src00.y, r000);                                                             \
        r001 = fma(weights_cache[0].lo.hi, src01.y, r001);                                                             \
        r010 = fma(weights_cache[0].lo.hi, src10.y, r010);                                                             \
        r011 = fma(weights_cache[0].lo.hi, src11.y, r011);                                                             \
        r000 = fma(weights_cache[0].hi.lo, src00.z, r000);                                                             \
        r001 = fma(weights_cache[0].hi.lo, src01.z, r001);                                                             \
        r010 = fma(weights_cache[0].hi.lo, src10.z, r010);                                                             \
        r011 = fma(weights_cache[0].hi.lo, src11.z, r011);                                                             \
        r000 = fma(weights_cache[0].hi.hi, src00.w, r000);                                                             \
        r001 = fma(weights_cache[0].hi.hi, src01.w, r001);                                                             \
        r010 = fma(weights_cache[0].hi.hi, src10.w, r010);                                                             \
        r011 = fma(weights_cache[0].hi.hi, src11.w, r011);                                                             \
        r100 = fma(weights_cache[1].lo.lo, src00.x, r100);                                                             \
        r101 = fma(weights_cache[1].lo.lo, src01.x, r101);                                                             \
        r110 = fma(weights_cache[1].lo.lo, src10.x, r110);                                                             \
        r111 = fma(weights_cache[1].lo.lo, src11.x, r111);                                                             \
        r100 = fma(weights_cache[1].lo.hi, src00.y, r100);                                                             \
        r101 = fma(weights_cache[1].lo.hi, src01.y, r101);                                                             \
        r110 = fma(weights_cache[1].lo.hi, src10.y, r110);                                                             \
        r111 = fma(weights_cache[1].lo.hi, src11.y, r111);                                                             \
        r100 = fma(weights_cache[1].hi.lo, src00.z, r100);                                                             \
        r101 = fma(weights_cache[1].hi.lo, src01.z, r101);                                                             \
        r110 = fma(weights_cache[1].hi.lo, src10.z, r110);                                                             \
        r111 = fma(weights_cache[1].hi.lo, src11.z, r111);                                                             \
        r100 = fma(weights_cache[1].hi.hi, src00.w, r100);                                                             \
        r101 = fma(weights_cache[1].hi.hi, src01.w, r101);                                                             \
        r110 = fma(weights_cache[1].hi.hi, src10.w, r110);                                                             \
        r111 = fma(weights_cache[1].hi.hi, src11.w, r111);                                                             \
        filters_loc += 2;                                                                                              \
    } while (s < src_z);                                                                                               \
    bias_cache = biases + Z;                                                                                           \
    if (Z + 0 >= dst_z)                                                                                                \
        return;                                                                                                        \
    {                                                                                                                  \
        DATA_T4 bias_val = CONVERT_TO_DATA_T4(bias_cache[0]);                                                          \
        {                                                                                                              \
            DATA_T4 res = CONVERT_TO_DATA_T4(r000) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
        if ((X + 1) < dst_x) {                                                                                         \
            DATA_T4 res = CONVERT_TO_DATA_T4(r001) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
        if ((Y + 1) < dst_y) {                                                                                         \
            DATA_T4 res = CONVERT_TO_DATA_T4(r010) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 0)), ((Y + 1)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 0)), ((Y + 1)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
        if ((X + 1) < dst_x && (Y + 1) < dst_y) {                                                                      \
            DATA_T4 res = CONVERT_TO_DATA_T4(r011) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 1)), ((Y + 1)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 1)), ((Y + 1)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
    }                                                                                                                  \
    if (Z + 1 >= dst_z)                                                                                                \
        return;                                                                                                        \
    {                                                                                                                  \
        DATA_T4 bias_val = CONVERT_TO_DATA_T4(bias_cache[1]);                                                          \
        {                                                                                                              \
            DATA_T4 res = CONVERT_TO_DATA_T4(r100) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 1)));                                 \
            write_imageh(dst_data, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 1)), res);                               \
        }                                                                                                              \
        if ((X + 1) < dst_x) {                                                                                         \
            DATA_T4 res = CONVERT_TO_DATA_T4(r101) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 1)));                                 \
            write_imageh(dst_data, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 1)), res);                               \
        }                                                                                                              \
        if ((Y + 1) < dst_y) {                                                                                         \
            DATA_T4 res = CONVERT_TO_DATA_T4(r110) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 0)), ((Y + 1)) * dst_z + (Z + 1)));                                 \
            write_imageh(dst_data, (int2)(((X + 0)), ((Y + 1)) * dst_z + (Z + 1)), res);                               \
        }                                                                                                              \
        if ((X + 1) < dst_x && (Y + 1) < dst_y) {                                                                      \
            DATA_T4 res = CONVERT_TO_DATA_T4(r111) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 1)), ((Y + 1)) * dst_z + (Z + 1)));                                 \
            write_imageh(dst_data, (int2)(((X + 1)), ((Y + 1)) * dst_z + (Z + 1)), res);                               \
        }                                                                                                              \
    }

#define POWERVR_IS1X11_SRCDEPTH2_BLOCK111(                                                                             \
    src_data, src_data_1, filters_buffer, biases, dst_data, src_x, src_y, src_z, src_w, dst_x, dst_y, dst_z, dst_w)    \
    int X = get_global_id(0) * 1;                                                                                      \
    int Y = get_global_id(1) * 1;                                                                                      \
    int Z = get_global_id(2) * 1;                                                                                      \
    if (X >= dst_x || Y >= dst_y || Z >= dst_z) {                                                                      \
        return;                                                                                                        \
    }                                                                                                                  \
    DATA_T4 r000 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    __global DATA_T4 *bias_cache;                                                                                      \
    int weight_offset = 4 * Z * 4 * src_z;                                                                             \
    int s = 0;                                                                                                         \
    do {                                                                                                               \
        DATA_T4 src00;                                                                                                 \
        DATA_T16 weights0123 = vload16(0, filters_buffer + weight_offset);                                             \
        src00 = read_imageh(src_data, smp_zero, (int2)(((X + 0)), ((Y + 0)) * src_z + (s)));                           \
        s += 1;                                                                                                        \
        r000 = fma(weights0123.lo.lo, src00.x, r000);                                                                  \
        r000 = fma(weights0123.lo.hi, src00.y, r000);                                                                  \
        r000 = fma(weights0123.hi.lo, src00.z, r000);                                                                  \
        r000 = fma(weights0123.hi.hi, src00.w, r000);                                                                  \
        DATA_T16 weights4567 = vload16(0, filters_buffer + weight_offset + 16);                                        \
        src00 = read_imageh(src_data, smp_zero, (int2)(((X + 0)), ((Y + 0)) * src_z + (s)));                           \
        r000 = fma(weights4567.lo.lo, src00.x, r000);                                                                  \
        r000 = fma(weights4567.lo.hi, src00.y, r000);                                                                  \
        r000 = fma(weights4567.hi.lo, src00.z, r000);                                                                  \
        r000 = fma(weights4567.hi.hi, src00.w, r000);                                                                  \
        s += 1;                                                                                                        \
        weight_offset += 32;                                                                                           \
    } while (s < src_z);                                                                                               \
    bias_cache = biases + Z;                                                                                           \
    if (Z + 0 >= dst_z)                                                                                                \
        return;                                                                                                        \
    {                                                                                                                  \
        DATA_T4 bias_val = CONVERT_TO_DATA_T4(bias_cache[0]);                                                          \
        {                                                                                                              \
            DATA_T4 res = CONVERT_TO_DATA_T4(r000) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
    }

#define POWERVR_IS1X11_SRCDEPTH2_BLOCK211(                                                                             \
    src_data, src_data_1, filters_buffer, biases, dst_data, src_x, src_y, src_z, src_w, dst_x, dst_y, dst_z, dst_w)    \
    int X = get_global_id(0) * 2;                                                                                      \
    int Y = get_global_id(1) * 1;                                                                                      \
    int Z = get_global_id(2) * 1;                                                                                      \
    if (X >= dst_x || Y >= dst_y || Z >= dst_z) {                                                                      \
        return;                                                                                                        \
    }                                                                                                                  \
    DATA_T4 r000 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r001 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    __global DATA_T4 *bias_cache;                                                                                      \
    int weight_offset = 4 * 4 * Z * src_z;                                                                             \
    int s = 0;                                                                                                         \
    do {                                                                                                               \
        DATA_T4 src00;                                                                                                 \
        DATA_T4 src01;                                                                                                 \
        DATA_T16 weights_cache0123 = vload16(0, filters_buffer + weight_offset);                                       \
        src00 = read_imageh(src_data, smp_zero, (int2)(((X + 0)), ((Y + 0)) * src_z + (s)));                           \
        src01 = read_imageh(src_data, smp_zero, (int2)(((X + 1)), ((Y + 0)) * src_z + (s)));                           \
        s += 1;                                                                                                        \
        r000 = fma(weights_cache0123.lo.lo, src00.x, r000);                                                            \
        r001 = fma(weights_cache0123.lo.lo, src01.x, r001);                                                            \
        r000 = fma(weights_cache0123.lo.hi, src00.y, r000);                                                            \
        r001 = fma(weights_cache0123.lo.hi, src01.y, r001);                                                            \
        r000 = fma(weights_cache0123.hi.lo, src00.z, r000);                                                            \
        r001 = fma(weights_cache0123.hi.lo, src01.z, r001);                                                            \
        r000 = fma(weights_cache0123.hi.hi, src00.w, r000);                                                            \
        r001 = fma(weights_cache0123.hi.hi, src01.w, r001);                                                            \
        DATA_T16 weights_cache4567 = vload16(0, filters_buffer + weight_offset + 16);                                  \
        src00 = read_imageh(src_data, smp_zero, (int2)(((X + 0)), ((Y + 0)) * src_z + (s)));                           \
        src01 = read_imageh(src_data, smp_zero, (int2)(((X + 1)), ((Y + 0)) * src_z + (s)));                           \
        r000 = fma(weights_cache4567.lo.lo, src00.x, r000);                                                            \
        r001 = fma(weights_cache4567.lo.lo, src01.x, r001);                                                            \
        r000 = fma(weights_cache4567.lo.hi, src00.y, r000);                                                            \
        r001 = fma(weights_cache4567.lo.hi, src01.y, r001);                                                            \
        r000 = fma(weights_cache4567.hi.lo, src00.z, r000);                                                            \
        r001 = fma(weights_cache4567.hi.lo, src01.z, r001);                                                            \
        r000 = fma(weights_cache4567.hi.hi, src00.w, r000);                                                            \
        r001 = fma(weights_cache4567.hi.hi, src01.w, r001);                                                            \
        s += 1;                                                                                                        \
        weight_offset += 32;                                                                                           \
    } while (s < src_z);                                                                                               \
    bias_cache = biases + Z;                                                                                           \
    if (Z + 0 >= dst_z)                                                                                                \
        return;                                                                                                        \
    {                                                                                                                  \
        DATA_T4 bias_val = CONVERT_TO_DATA_T4(bias_cache[0]);                                                          \
        {                                                                                                              \
            DATA_T4 res = CONVERT_TO_DATA_T4(r000) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
        if ((X + 1) < dst_x) {                                                                                         \
            DATA_T4 res = CONVERT_TO_DATA_T4(r001) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
    }

#define POWERVR_IS1X11_SRCDEPTH2_BLOCK212(                                                                             \
    src_data, src_data_1, filters_buffer, biases, dst_data, src_x, src_y, src_z, src_w, dst_x, dst_y, dst_z, dst_w)    \
    int X = get_global_id(0) * 2;                                                                                      \
    int Y = get_global_id(1) * 1;                                                                                      \
    int Z = get_global_id(2) * 2;                                                                                      \
    if (X >= dst_x || Y >= dst_y || Z >= dst_z) {                                                                      \
        return;                                                                                                        \
    }                                                                                                                  \
    DATA_T4 r000 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r001 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r100 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r101 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    __global DATA_T4 *bias_cache;                                                                                      \
    int weight_offset = 4 * 4 * Z * src_z;                                                                             \
    int s = 0;                                                                                                         \
    do {                                                                                                               \
        DATA_T4 src00;                                                                                                 \
        DATA_T4 src01;                                                                                                 \
        DATA_T16 weights_cache0123 = vload16(0, filters_buffer + weight_offset);                                       \
        DATA_T16 weights_cache4567 = vload16(0, filters_buffer + weight_offset + 16);                                  \
        src00 = read_imageh(src_data, smp_zero, (int2)(((X + 0)), ((Y + 0)) * src_z + (s)));                           \
        src01 = read_imageh(src_data, smp_zero, (int2)(((X + 1)), ((Y + 0)) * src_z + (s)));                           \
        s += 1;                                                                                                        \
        r000 = fma(weights_cache0123.lo.lo, src00.x, r000);                                                            \
        r001 = fma(weights_cache0123.lo.lo, src01.x, r001);                                                            \
        r000 = fma(weights_cache0123.lo.hi, src00.y, r000);                                                            \
        r001 = fma(weights_cache0123.lo.hi, src01.y, r001);                                                            \
        r000 = fma(weights_cache0123.hi.lo, src00.z, r000);                                                            \
        r001 = fma(weights_cache0123.hi.lo, src01.z, r001);                                                            \
        r000 = fma(weights_cache0123.hi.hi, src00.w, r000);                                                            \
        r001 = fma(weights_cache0123.hi.hi, src01.w, r001);                                                            \
        r100 = fma(weights_cache4567.lo.lo, src00.x, r100);                                                            \
        r101 = fma(weights_cache4567.lo.lo, src01.x, r101);                                                            \
        r100 = fma(weights_cache4567.lo.hi, src00.y, r100);                                                            \
        r101 = fma(weights_cache4567.lo.hi, src01.y, r101);                                                            \
        r100 = fma(weights_cache4567.hi.lo, src00.z, r100);                                                            \
        r101 = fma(weights_cache4567.hi.lo, src01.z, r101);                                                            \
        r100 = fma(weights_cache4567.hi.hi, src00.w, r100);                                                            \
        r101 = fma(weights_cache4567.hi.hi, src01.w, r101);                                                            \
        DATA_T16 weights_cache89ab = vload16(0, filters_buffer + weight_offset + 32);                                  \
        DATA_T16 weights_cachecdef = vload16(0, filters_buffer + weight_offset + 48);                                  \
        src00 = read_imageh(src_data, smp_zero, (int2)(((X + 0)), ((Y + 0)) * src_z + (s)));                           \
        src01 = read_imageh(src_data, smp_zero, (int2)(((X + 1)), ((Y + 0)) * src_z + (s)));                           \
        r000 = fma(weights_cache89ab.lo.lo, src00.x, r000);                                                            \
        r001 = fma(weights_cache89ab.lo.lo, src01.x, r001);                                                            \
        r000 = fma(weights_cache89ab.lo.hi, src00.y, r000);                                                            \
        r001 = fma(weights_cache89ab.lo.hi, src01.y, r001);                                                            \
        r000 = fma(weights_cache89ab.hi.lo, src00.z, r000);                                                            \
        r001 = fma(weights_cache89ab.hi.lo, src01.z, r001);                                                            \
        r000 = fma(weights_cache89ab.hi.hi, src00.w, r000);                                                            \
        r001 = fma(weights_cache89ab.hi.hi, src01.w, r001);                                                            \
        r100 = fma(weights_cachecdef.lo.lo, src00.x, r100);                                                            \
        r101 = fma(weights_cachecdef.lo.lo, src01.x, r101);                                                            \
        r100 = fma(weights_cachecdef.lo.hi, src00.y, r100);                                                            \
        r101 = fma(weights_cachecdef.lo.hi, src01.y, r101);                                                            \
        r100 = fma(weights_cachecdef.hi.lo, src00.z, r100);                                                            \
        r101 = fma(weights_cachecdef.hi.lo, src01.z, r101);                                                            \
        r100 = fma(weights_cachecdef.hi.hi, src00.w, r100);                                                            \
        r101 = fma(weights_cachecdef.hi.hi, src01.w, r101);                                                            \
        s += 1;                                                                                                        \
        weight_offset += 64;                                                                                           \
    } while (s < src_z);                                                                                               \
    bias_cache = biases + Z;                                                                                           \
    if (Z + 0 >= dst_z)                                                                                                \
        return;                                                                                                        \
    {                                                                                                                  \
        DATA_T4 bias_val = CONVERT_TO_DATA_T4(bias_cache[0]);                                                          \
        {                                                                                                              \
            DATA_T4 res = CONVERT_TO_DATA_T4(r000) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
        if ((X + 1) < dst_x) {                                                                                         \
            DATA_T4 res = CONVERT_TO_DATA_T4(r001) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
    }                                                                                                                  \
    if (Z + 1 >= dst_z)                                                                                                \
        return;                                                                                                        \
    {                                                                                                                  \
        DATA_T4 bias_val = CONVERT_TO_DATA_T4(bias_cache[1]);                                                          \
        {                                                                                                              \
            DATA_T4 res = CONVERT_TO_DATA_T4(r100) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 1)));                                 \
            write_imageh(dst_data, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 1)), res);                               \
        }                                                                                                              \
        if ((X + 1) < dst_x) {                                                                                         \
            DATA_T4 res = CONVERT_TO_DATA_T4(r101) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 1)));                                 \
            write_imageh(dst_data, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 1)), res);                               \
        }                                                                                                              \
    }

#define POWERVR_IS1X10_SRCDEPTH1_BLOCK111(src_data,                                                                    \
                                          src_data_1,                                                                  \
                                          filters_buffer,                                                              \
                                          biases,                                                                      \
                                          dst_data,                                                                    \
                                          src_x,                                                                       \
                                          src_y,                                                                       \
                                          src_z,                                                                       \
                                          src_w,                                                                       \
                                          dst_x,                                                                       \
                                          dst_y,                                                                       \
                                          dst_z,                                                                       \
                                          dst_w,                                                                       \
                                          kw,                                                                          \
                                          kh,                                                                          \
                                          dilation_x,                                                                  \
                                          dilation_y,                                                                  \
                                          stride_x,                                                                    \
                                          stride_y,                                                                    \
                                          padding_x,                                                                   \
                                          padding_y)                                                                   \
    int X = get_global_id(0) * 1;                                                                                      \
    int Y = get_global_id(1) * 1;                                                                                      \
    int Z = get_global_id(2) * 1;                                                                                      \
    if (X >= dst_x || Y >= dst_y || Z >= dst_z) {                                                                      \
        return;                                                                                                        \
    }                                                                                                                  \
    DATA_T4 r000 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    int xc0 = (X + 0) * stride_x + padding_x;                                                                          \
    int yc0 = (Y + 0) * stride_y + padding_y;                                                                          \
    __global DATA_T4 *weights_cache;                                                                                   \
    __global DATA_T4 *filters_loc = filters_buffer + Z * 4 * src_z * kw * kh;                                          \
    for (int ky = 0; ky < kh; ++ky) {                                                                                  \
        int yck0 = ky * dilation_y + yc0;                                                                              \
        for (int kx = 0; kx < kw; ++kx) {                                                                              \
            int xck0 = kx * dilation_x + xc0;                                                                          \
            int s = 0;                                                                                                 \
            do {                                                                                                       \
                DATA_T4 src00;                                                                                         \
                weights_cache = filters_loc;                                                                           \
                src00 = read_imageh(src_data, smp_zero, (int2)((xck0), (yck0)*src_z + (s)));                           \
                s += 1;                                                                                                \
                r000 = fma(weights_cache[0], src00.x, r000);                                                           \
                r000 = fma(weights_cache[1], src00.y, r000);                                                           \
                r000 = fma(weights_cache[2], src00.z, r000);                                                           \
                r000 = fma(weights_cache[3], src00.w, r000);                                                           \
                filters_loc += 4;                                                                                      \
            } while (s < src_z);                                                                                       \
        }                                                                                                              \
    }                                                                                                                  \
    weights_cache = biases + Z;                                                                                        \
    if (Z + 0 >= dst_z)                                                                                                \
        return;                                                                                                        \
    {                                                                                                                  \
        DATA_T4 bias_val = CONVERT_TO_DATA_T4(weights_cache[0]);                                                       \
        {                                                                                                              \
            DATA_T4 res = CONVERT_TO_DATA_T4(r000) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
    }

#define POWERVR_IS1X10_SRCDEPTH1_BLOCK211(src_data,                                                                    \
                                          src_data_1,                                                                  \
                                          filters_buffer,                                                              \
                                          biases,                                                                      \
                                          dst_data,                                                                    \
                                          src_x,                                                                       \
                                          src_y,                                                                       \
                                          src_z,                                                                       \
                                          src_w,                                                                       \
                                          dst_x,                                                                       \
                                          dst_y,                                                                       \
                                          dst_z,                                                                       \
                                          dst_w,                                                                       \
                                          kw,                                                                          \
                                          kh,                                                                          \
                                          dilation_x,                                                                  \
                                          dilation_y,                                                                  \
                                          stride_x,                                                                    \
                                          stride_y,                                                                    \
                                          padding_x,                                                                   \
                                          padding_y)                                                                   \
    int X = get_global_id(0) * 2;                                                                                      \
    int Y = get_global_id(1) * 1;                                                                                      \
    int Z = get_global_id(2) * 1;                                                                                      \
    if (X >= dst_x || Y >= dst_y || Z >= dst_z) {                                                                      \
        return;                                                                                                        \
    }                                                                                                                  \
    DATA_T4 r000 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r001 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    int xc0 = (X + 0) * stride_x + padding_x;                                                                          \
    int xc1 = (X + 1) * stride_x + padding_x;                                                                          \
    int yc0 = (Y + 0) * stride_y + padding_y;                                                                          \
    __global DATA_T4 *weights_cache;                                                                                   \
    __global DATA_T4 *filters_loc = filters_buffer + Z * 4 * src_z * kw * kh;                                          \
    for (int ky = 0; ky < kh; ++ky) {                                                                                  \
        int yck0 = ky * dilation_y + yc0;                                                                              \
        for (int kx = 0; kx < kw; ++kx) {                                                                              \
            int xck0 = kx * dilation_x + xc0;                                                                          \
            int xck1 = kx * dilation_x + xc1;                                                                          \
            int s = 0;                                                                                                 \
            do {                                                                                                       \
                DATA_T4 src00;                                                                                         \
                DATA_T4 src01;                                                                                         \
                weights_cache = filters_loc;                                                                           \
                src00 = read_imageh(src_data, smp_zero, (int2)((xck0), (yck0)*src_z + (s)));                           \
                src01 = read_imageh(src_data, smp_zero, (int2)((xck1), (yck0)*src_z + (s)));                           \
                s += 1;                                                                                                \
                r000 = fma(weights_cache[0], src00.x, r000);                                                           \
                r001 = fma(weights_cache[0], src01.x, r001);                                                           \
                r000 = fma(weights_cache[1], src00.y, r000);                                                           \
                r001 = fma(weights_cache[1], src01.y, r001);                                                           \
                r000 = fma(weights_cache[2], src00.z, r000);                                                           \
                r001 = fma(weights_cache[2], src01.z, r001);                                                           \
                r000 = fma(weights_cache[3], src00.w, r000);                                                           \
                r001 = fma(weights_cache[3], src01.w, r001);                                                           \
                filters_loc += 4;                                                                                      \
            } while (s < src_z);                                                                                       \
        }                                                                                                              \
    }                                                                                                                  \
    weights_cache = biases + Z;                                                                                        \
    if (Z + 0 >= dst_z)                                                                                                \
        return;                                                                                                        \
    {                                                                                                                  \
        DATA_T4 bias_val = CONVERT_TO_DATA_T4(weights_cache[0]);                                                       \
        {                                                                                                              \
            DATA_T4 res = CONVERT_TO_DATA_T4(r000) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
        if ((X + 1) < dst_x) {                                                                                         \
            DATA_T4 res = CONVERT_TO_DATA_T4(r001) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
    }

#define POWERVR_IS1X10_SRCDEPTH1_BLOCK212(src_data,                                                                    \
                                          src_data_1,                                                                  \
                                          filters_buffer,                                                              \
                                          biases,                                                                      \
                                          dst_data,                                                                    \
                                          src_x,                                                                       \
                                          src_y,                                                                       \
                                          src_z,                                                                       \
                                          src_w,                                                                       \
                                          dst_x,                                                                       \
                                          dst_y,                                                                       \
                                          dst_z,                                                                       \
                                          dst_w,                                                                       \
                                          kw,                                                                          \
                                          kh,                                                                          \
                                          dilation_x,                                                                  \
                                          dilation_y,                                                                  \
                                          stride_x,                                                                    \
                                          stride_y,                                                                    \
                                          padding_x,                                                                   \
                                          padding_y)                                                                   \
    int X = get_global_id(0) * 2;                                                                                      \
    int Y = get_global_id(1) * 1;                                                                                      \
    int Z = get_global_id(2) * 2;                                                                                      \
    if (X >= dst_x || Y >= dst_y || Z >= dst_z) {                                                                      \
        return;                                                                                                        \
    }                                                                                                                  \
    DATA_T4 r000 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r001 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r100 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r101 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    int xc0 = (X + 0) * stride_x + padding_x;                                                                          \
    int xc1 = (X + 1) * stride_x + padding_x;                                                                          \
    int yc0 = (Y + 0) * stride_y + padding_y;                                                                          \
    __global DATA_T4 *weights_cache;                                                                                   \
    int weight_offset = 4 * Z * 4 * src_z * kh * kw;                                                                   \
    for (int ky = 0; ky < kh; ++ky) {                                                                                  \
        int yck0 = ky * dilation_y + yc0;                                                                              \
        for (int kx = 0; kx < kw; ++kx) {                                                                              \
            int xck0 = kx * dilation_x + xc0;                                                                          \
            int xck1 = kx * dilation_x + xc1;                                                                          \
            int s = 0;                                                                                                 \
            do {                                                                                                       \
                DATA_T4 src00;                                                                                         \
                DATA_T4 src01;                                                                                         \
                DATA_T16 weights_cache0123 = vload16(0, filters_buffer + weight_offset);                               \
                DATA_T16 weights_cache4567 = vload16(0, filters_buffer + weight_offset + 16);                          \
                src00 = read_imageh(src_data, smp_zero, (int2)((xck0), (yck0)*src_z + (s)));                           \
                src01 = read_imageh(src_data, smp_zero, (int2)((xck1), (yck0)*src_z + (s)));                           \
                s += 1;                                                                                                \
                r000 = fma(weights_cache0123.lo.lo, src00.x, r000);                                                    \
                r001 = fma(weights_cache0123.lo.lo, src01.x, r001);                                                    \
                r000 = fma(weights_cache0123.lo.hi, src00.y, r000);                                                    \
                r001 = fma(weights_cache0123.lo.hi, src01.y, r001);                                                    \
                r000 = fma(weights_cache0123.hi.lo, src00.z, r000);                                                    \
                r001 = fma(weights_cache0123.hi.lo, src01.z, r001);                                                    \
                r000 = fma(weights_cache0123.hi.hi, src00.w, r000);                                                    \
                r001 = fma(weights_cache0123.hi.hi, src01.w, r001);                                                    \
                r100 = fma(weights_cache4567.lo.lo, src00.x, r100);                                                    \
                r101 = fma(weights_cache4567.lo.lo, src01.x, r101);                                                    \
                r100 = fma(weights_cache4567.lo.hi, src00.y, r100);                                                    \
                r101 = fma(weights_cache4567.lo.hi, src01.y, r101);                                                    \
                r100 = fma(weights_cache4567.hi.lo, src00.z, r100);                                                    \
                r101 = fma(weights_cache4567.hi.lo, src01.z, r101);                                                    \
                r100 = fma(weights_cache4567.hi.hi, src00.w, r100);                                                    \
                r101 = fma(weights_cache4567.hi.hi, src01.w, r101);                                                    \
                weight_offset += 32;                                                                                   \
            } while (s < src_z);                                                                                       \
        }                                                                                                              \
    }                                                                                                                  \
    weights_cache = biases + Z;                                                                                        \
    if (Z + 0 >= dst_z)                                                                                                \
        return;                                                                                                        \
    {                                                                                                                  \
        DATA_T4 bias_val = CONVERT_TO_DATA_T4(weights_cache[0]);                                                       \
        {                                                                                                              \
            DATA_T4 res = CONVERT_TO_DATA_T4(r000) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
        if ((X + 1) < dst_x) {                                                                                         \
            DATA_T4 res = CONVERT_TO_DATA_T4(r001) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
    }                                                                                                                  \
    if (Z + 1 >= dst_z)                                                                                                \
        return;                                                                                                        \
    {                                                                                                                  \
        DATA_T4 bias_val = CONVERT_TO_DATA_T4(weights_cache[1]);                                                       \
        {                                                                                                              \
            DATA_T4 res = CONVERT_TO_DATA_T4(r100) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 1)));                                 \
            write_imageh(dst_data, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 1)), res);                               \
        }                                                                                                              \
        if ((X + 1) < dst_x) {                                                                                         \
            DATA_T4 res = CONVERT_TO_DATA_T4(r101) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 1)));                                 \
            write_imageh(dst_data, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 1)), res);                               \
        }                                                                                                              \
    }

#define POWERVR_IS1X10_SRCDEPTH2_BLOCK111(src_data,                                                                    \
                                          src_data_1,                                                                  \
                                          filters_buffer,                                                              \
                                          biases,                                                                      \
                                          dst_data,                                                                    \
                                          src_x,                                                                       \
                                          src_y,                                                                       \
                                          src_z,                                                                       \
                                          src_w,                                                                       \
                                          dst_x,                                                                       \
                                          dst_y,                                                                       \
                                          dst_z,                                                                       \
                                          dst_w,                                                                       \
                                          kw,                                                                          \
                                          kh,                                                                          \
                                          dilation_x,                                                                  \
                                          dilation_y,                                                                  \
                                          stride_x,                                                                    \
                                          stride_y,                                                                    \
                                          padding_x,                                                                   \
                                          padding_y)                                                                   \
    int X = get_global_id(0) * 1;                                                                                      \
    int Y = get_global_id(1) * 1;                                                                                      \
    int Z = get_global_id(2) * 1;                                                                                      \
    if (X >= dst_x || Y >= dst_y || Z >= dst_z) {                                                                      \
        return;                                                                                                        \
    }                                                                                                                  \
    DATA_T4 r000 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    int xc0 = (X + 0) * stride_x + padding_x;                                                                          \
    int yc0 = (Y + 0) * stride_y + padding_y;                                                                          \
    __global DATA_T4 *weights_cache;                                                                                   \
    __global DATA_T4 *filters_loc = filters_buffer + Z * 4 * src_z * kh * kw;                                          \
    for (int ky = 0; ky < kh; ++ky) {                                                                                  \
        int yck0 = ky * dilation_y + yc0;                                                                              \
        for (int kx = 0; kx < kw; ++kx) {                                                                              \
            int xck0 = kx * dilation_x + xc0;                                                                          \
            int s = 0;                                                                                                 \
            do {                                                                                                       \
                DATA_T4 src00;                                                                                         \
                weights_cache = filters_loc;                                                                           \
                src00 = read_imageh(src_data, smp_zero, (int2)((xck0), (yck0)*src_z + (s)));                           \
                s += 1;                                                                                                \
                r000 = fma(weights_cache[0], src00.x, r000);                                                           \
                r000 = fma(weights_cache[1], src00.y, r000);                                                           \
                r000 = fma(weights_cache[2], src00.z, r000);                                                           \
                r000 = fma(weights_cache[3], src00.w, r000);                                                           \
                src00 = read_imageh(src_data, smp_zero, (int2)((xck0), (yck0)*src_z + (s)));                           \
                r000 = fma(weights_cache[4], src00.x, r000);                                                           \
                r000 = fma(weights_cache[5], src00.y, r000);                                                           \
                r000 = fma(weights_cache[6], src00.z, r000);                                                           \
                r000 = fma(weights_cache[7], src00.w, r000);                                                           \
                s += 1;                                                                                                \
                filters_loc += 8;                                                                                      \
            } while (s < src_z);                                                                                       \
        }                                                                                                              \
    }                                                                                                                  \
    weights_cache = biases + Z;                                                                                        \
    if (Z + 0 >= dst_z)                                                                                                \
        return;                                                                                                        \
    {                                                                                                                  \
        DATA_T4 bias_val = CONVERT_TO_DATA_T4(weights_cache[0]);                                                       \
        {                                                                                                              \
            DATA_T4 res = CONVERT_TO_DATA_T4(r000) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
    }

#define POWERVR_IS1X10_SRCDEPTH2_BLOCK211(src_data,                                                                    \
                                          src_data_1,                                                                  \
                                          filters_buffer,                                                              \
                                          biases,                                                                      \
                                          dst_data,                                                                    \
                                          src_x,                                                                       \
                                          src_y,                                                                       \
                                          src_z,                                                                       \
                                          src_w,                                                                       \
                                          dst_x,                                                                       \
                                          dst_y,                                                                       \
                                          dst_z,                                                                       \
                                          dst_w,                                                                       \
                                          kw,                                                                          \
                                          kh,                                                                          \
                                          dilation_x,                                                                  \
                                          dilation_y,                                                                  \
                                          stride_x,                                                                    \
                                          stride_y,                                                                    \
                                          padding_x,                                                                   \
                                          padding_y)                                                                   \
    int X = get_global_id(0) * 2;                                                                                      \
    int Y = get_global_id(1) * 1;                                                                                      \
    int Z = get_global_id(2) * 1;                                                                                      \
    if (X >= dst_x || Y >= dst_y || Z >= dst_z) {                                                                      \
        return;                                                                                                        \
    }                                                                                                                  \
    DATA_T4 r000 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r001 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    int xc0 = (X + 0) * stride_x + padding_x;                                                                          \
    int xc1 = (X + 1) * stride_x + padding_x;                                                                          \
    int yc0 = (Y + 0) * stride_y + padding_y;                                                                          \
    __global DATA_T4 *weights_cache;                                                                                   \
    __global DATA_T4 *filters_loc = filters_buffer + Z * 4 * src_z * kh * kw;                                          \
    for (int ky = 0; ky < kh; ++ky) {                                                                                  \
        int yck0 = ky * dilation_y + yc0;                                                                              \
        for (int kx = 0; kx < kw; ++kx) {                                                                              \
            int xck0 = kx * dilation_x + xc0;                                                                          \
            int xck1 = kx * dilation_x + xc1;                                                                          \
            int s = 0;                                                                                                 \
            do {                                                                                                       \
                DATA_T4 src00;                                                                                         \
                DATA_T4 src01;                                                                                         \
                weights_cache = filters_loc;                                                                           \
                src00 = read_imageh(src_data, smp_zero, (int2)((xck0), (yck0)*src_z + (s)));                           \
                src01 = read_imageh(src_data, smp_zero, (int2)((xck1), (yck0)*src_z + (s)));                           \
                s += 1;                                                                                                \
                r000 = fma(weights_cache[0], src00.x, r000);                                                           \
                r001 = fma(weights_cache[0], src01.x, r001);                                                           \
                r000 = fma(weights_cache[1], src00.y, r000);                                                           \
                r001 = fma(weights_cache[1], src01.y, r001);                                                           \
                r000 = fma(weights_cache[2], src00.z, r000);                                                           \
                r001 = fma(weights_cache[2], src01.z, r001);                                                           \
                r000 = fma(weights_cache[3], src00.w, r000);                                                           \
                r001 = fma(weights_cache[3], src01.w, r001);                                                           \
                src00 = read_imageh(src_data, smp_zero, (int2)((xck0), (yck0)*src_z + (s)));                           \
                src01 = read_imageh(src_data, smp_zero, (int2)((xck1), (yck0)*src_z + (s)));                           \
                r000 = fma(weights_cache[4], src00.x, r000);                                                           \
                r001 = fma(weights_cache[4], src01.x, r001);                                                           \
                r000 = fma(weights_cache[5], src00.y, r000);                                                           \
                r001 = fma(weights_cache[5], src01.y, r001);                                                           \
                r000 = fma(weights_cache[6], src00.z, r000);                                                           \
                r001 = fma(weights_cache[6], src01.z, r001);                                                           \
                r000 = fma(weights_cache[7], src00.w, r000);                                                           \
                r001 = fma(weights_cache[7], src01.w, r001);                                                           \
                s += 1;                                                                                                \
                filters_loc += 8;                                                                                      \
            } while (s < src_z);                                                                                       \
        }                                                                                                              \
    }                                                                                                                  \
    weights_cache = biases + Z;                                                                                        \
    if (Z + 0 >= dst_z)                                                                                                \
        return;                                                                                                        \
    {                                                                                                                  \
        DATA_T4 bias_val = CONVERT_TO_DATA_T4(weights_cache[0]);                                                       \
        {                                                                                                              \
            DATA_T4 res = CONVERT_TO_DATA_T4(r000) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
        if ((X + 1) < dst_x) {                                                                                         \
            DATA_T4 res = CONVERT_TO_DATA_T4(r001) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
    }

#define POWERVR_IS1X10_SRCDEPTH2_BLOCK212(src_data,                                                                    \
                                          src_data_1,                                                                  \
                                          filters_buffer,                                                              \
                                          biases,                                                                      \
                                          dst_data,                                                                    \
                                          src_x,                                                                       \
                                          src_y,                                                                       \
                                          src_z,                                                                       \
                                          src_w,                                                                       \
                                          dst_x,                                                                       \
                                          dst_y,                                                                       \
                                          dst_z,                                                                       \
                                          dst_w,                                                                       \
                                          kw,                                                                          \
                                          kh,                                                                          \
                                          dilation_x,                                                                  \
                                          dilation_y,                                                                  \
                                          stride_x,                                                                    \
                                          stride_y,                                                                    \
                                          padding_x,                                                                   \
                                          padding_y)                                                                   \
    int X = get_global_id(0) * 2;                                                                                      \
    int Y = get_global_id(1) * 1;                                                                                      \
    int Z = get_global_id(2) * 2;                                                                                      \
    if (X >= dst_x || Y >= dst_y || Z >= dst_z) {                                                                      \
        return;                                                                                                        \
    }                                                                                                                  \
    DATA_T4 r000 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r001 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r100 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r101 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    int xc0 = (X + 0) * stride_x + padding_x;                                                                          \
    int xc1 = (X + 1) * stride_x + padding_x;                                                                          \
    int yc0 = (Y + 0) * stride_y + padding_y;                                                                          \
    __global DATA_T4 *weights_cache;                                                                                   \
    __global DATA_T4 *filters_loc = filters_buffer + Z * 4 * src_z * kh * kw;                                          \
    for (int ky = 0; ky < kh; ++ky) {                                                                                  \
        int yck0 = ky * dilation_y + yc0;                                                                              \
        for (int kx = 0; kx < kw; ++kx) {                                                                              \
            int xck0 = kx * dilation_x + xc0;                                                                          \
            int xck1 = kx * dilation_x + xc1;                                                                          \
            int s = 0;                                                                                                 \
            do {                                                                                                       \
                DATA_T4 src00;                                                                                         \
                DATA_T4 src01;                                                                                         \
                weights_cache = filters_loc;                                                                           \
                src00 = read_imageh(src_data, smp_zero, (int2)((xck0), (yck0)*src_z + (s)));                           \
                src01 = read_imageh(src_data, smp_zero, (int2)((xck1), (yck0)*src_z + (s)));                           \
                s += 1;                                                                                                \
                r000 = fma(weights_cache[0], src00.x, r000);                                                           \
                r001 = fma(weights_cache[0], src01.x, r001);                                                           \
                r000 = fma(weights_cache[1], src00.y, r000);                                                           \
                r001 = fma(weights_cache[1], src01.y, r001);                                                           \
                r000 = fma(weights_cache[2], src00.z, r000);                                                           \
                r001 = fma(weights_cache[2], src01.z, r001);                                                           \
                r000 = fma(weights_cache[3], src00.w, r000);                                                           \
                r001 = fma(weights_cache[3], src01.w, r001);                                                           \
                r100 = fma(weights_cache[4], src00.x, r100);                                                           \
                r101 = fma(weights_cache[4], src01.x, r101);                                                           \
                r100 = fma(weights_cache[5], src00.y, r100);                                                           \
                r101 = fma(weights_cache[5], src01.y, r101);                                                           \
                r100 = fma(weights_cache[6], src00.z, r100);                                                           \
                r101 = fma(weights_cache[6], src01.z, r101);                                                           \
                r100 = fma(weights_cache[7], src00.w, r100);                                                           \
                r101 = fma(weights_cache[7], src01.w, r101);                                                           \
                src00 = read_imageh(src_data, smp_zero, (int2)((xck0), (yck0)*src_z + (s)));                           \
                src01 = read_imageh(src_data, smp_zero, (int2)((xck1), (yck0)*src_z + (s)));                           \
                r000 = fma(weights_cache[8], src00.x, r000);                                                           \
                r001 = fma(weights_cache[8], src01.x, r001);                                                           \
                r000 = fma(weights_cache[9], src00.y, r000);                                                           \
                r001 = fma(weights_cache[9], src01.y, r001);                                                           \
                r000 = fma(weights_cache[10], src00.z, r000);                                                          \
                r001 = fma(weights_cache[10], src01.z, r001);                                                          \
                r000 = fma(weights_cache[11], src00.w, r000);                                                          \
                r001 = fma(weights_cache[11], src01.w, r001);                                                          \
                r100 = fma(weights_cache[12], src00.x, r100);                                                          \
                r101 = fma(weights_cache[12], src01.x, r101);                                                          \
                r100 = fma(weights_cache[13], src00.y, r100);                                                          \
                r101 = fma(weights_cache[13], src01.y, r101);                                                          \
                r100 = fma(weights_cache[14], src00.z, r100);                                                          \
                r101 = fma(weights_cache[14], src01.z, r101);                                                          \
                r100 = fma(weights_cache[15], src00.w, r100);                                                          \
                r101 = fma(weights_cache[15], src01.w, r101);                                                          \
                s += 1;                                                                                                \
                filters_loc += 16;                                                                                     \
            } while (s < src_z);                                                                                       \
        }                                                                                                              \
    }                                                                                                                  \
    weights_cache = biases + Z;                                                                                        \
    if (Z + 0 >= dst_z)                                                                                                \
        return;                                                                                                        \
    {                                                                                                                  \
        DATA_T4 bias_val = CONVERT_TO_DATA_T4(weights_cache[0]);                                                       \
        {                                                                                                              \
            DATA_T4 res = CONVERT_TO_DATA_T4(r000) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
        if ((X + 1) < dst_x) {                                                                                         \
            DATA_T4 res = CONVERT_TO_DATA_T4(r001) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
    }                                                                                                                  \
    if (Z + 1 >= dst_z)                                                                                                \
        return;                                                                                                        \
    {                                                                                                                  \
        DATA_T4 bias_val = CONVERT_TO_DATA_T4(weights_cache[1]);                                                       \
        {                                                                                                              \
            DATA_T4 res = CONVERT_TO_DATA_T4(r100) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 1)));                                 \
            write_imageh(dst_data, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 1)), res);                               \
        }                                                                                                              \
        if ((X + 1) < dst_x) {                                                                                         \
            DATA_T4 res = CONVERT_TO_DATA_T4(r101) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 1)));                                 \
            write_imageh(dst_data, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 1)), res);                               \
        }                                                                                                              \
    }

#define POWERVR_IS1X11_SRCDEPTH4_BLOCK111(                                                                             \
    src_data, src_data_1, filters_buffer, biases, dst_data, src_x, src_y, src_z, src_w, dst_x, dst_y, dst_z, dst_w)    \
    int X = get_global_id(0) * 1;                                                                                      \
    int Y = get_global_id(1) * 1;                                                                                      \
    int Z = get_global_id(2) * 1;                                                                                      \
    if (X >= dst_x || Y >= dst_y || Z >= dst_z) {                                                                      \
        return;                                                                                                        \
    }                                                                                                                  \
    DATA_T4 r000 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    __global DATA_T4 *bias_cache;                                                                                      \
    int weight_offset = 4 * Z * 4 * src_z;                                                                             \
    int s = 0;                                                                                                         \
    do {                                                                                                               \
        DATA_T4 src00;                                                                                                 \
        DATA_T16 weights0123 = vload16(0, filters_buffer + weight_offset);                                             \
        src00 = read_imageh(src_data, smp_zero, (int2)(((X + 0)), ((Y + 0)) * src_z + (s)));                           \
        r000 = fma(weights0123.lo.lo, src00.x, r000);                                                                  \
        r000 = fma(weights0123.lo.hi, src00.y, r000);                                                                  \
        r000 = fma(weights0123.hi.lo, src00.z, r000);                                                                  \
        r000 = fma(weights0123.hi.hi, src00.w, r000);                                                                  \
        s += 1;                                                                                                        \
        DATA_T16 weights4567 = vload16(0, filters_buffer + weight_offset + 16);                                        \
        src00 = read_imageh(src_data, smp_zero, (int2)(((X + 0)), ((Y + 0)) * src_z + (s)));                           \
        r000 = fma(weights4567.lo.lo, src00.x, r000);                                                                  \
        r000 = fma(weights4567.lo.hi, src00.y, r000);                                                                  \
        r000 = fma(weights4567.hi.lo, src00.z, r000);                                                                  \
        r000 = fma(weights4567.hi.hi, src00.w, r000);                                                                  \
        s += 1;                                                                                                        \
        DATA_T16 weights89ab = vload16(0, filters_buffer + weight_offset + 32);                                        \
        src00 = read_imageh(src_data, smp_zero, (int2)(((X + 0)), ((Y + 0)) * src_z + (s)));                           \
        r000 = fma(weights89ab.lo.lo, src00.x, r000);                                                                  \
        r000 = fma(weights89ab.lo.hi, src00.y, r000);                                                                  \
        r000 = fma(weights89ab.hi.lo, src00.z, r000);                                                                  \
        r000 = fma(weights89ab.hi.hi, src00.w, r000);                                                                  \
        s += 1;                                                                                                        \
        DATA_T16 weightscdef = vload16(0, filters_buffer + weight_offset + 48);                                        \
        src00 = read_imageh(src_data, smp_zero, (int2)(((X + 0)), ((Y + 0)) * src_z + (s)));                           \
        r000 = fma(weightscdef.lo.lo, src00.x, r000);                                                                  \
        r000 = fma(weightscdef.lo.hi, src00.y, r000);                                                                  \
        r000 = fma(weightscdef.hi.lo, src00.z, r000);                                                                  \
        r000 = fma(weightscdef.hi.hi, src00.w, r000);                                                                  \
        s += 1;                                                                                                        \
        weight_offset += 64;                                                                                           \
    } while (s < src_z);                                                                                               \
    bias_cache = biases + Z;                                                                                           \
    if (Z + 0 >= dst_z)                                                                                                \
        return;                                                                                                        \
    {                                                                                                                  \
        DATA_T4 bias_val = CONVERT_TO_DATA_T4(bias_cache[0]);                                                          \
        {                                                                                                              \
            DATA_T4 res = CONVERT_TO_DATA_T4(r000) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
    }

#define POWERVR_IS1X11_SRCDEPTH1_BLOCK221(                                                                             \
    src_data, src_data_1, filters_buffer, biases, dst_data, src_x, src_y, src_z, src_w, dst_x, dst_y, dst_z, dst_w)    \
    int X = get_global_id(0) * 2;                                                                                      \
    int Y = get_global_id(1) * 2;                                                                                      \
    int Z = get_global_id(2) * 1;                                                                                      \
    if (X >= dst_x || Y >= dst_y || Z >= dst_z) {                                                                      \
        return;                                                                                                        \
    }                                                                                                                  \
    DATA_T4 r000 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r001 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r010 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r011 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    __global DATA_T4 *bias_cache;                                                                                      \
    int weight_offset = 4 * Z * 4 * src_z;                                                                             \
    int s = 0;                                                                                                         \
    int x0 = X;                                                                                                        \
    int x1 = X + 1;                                                                                                    \
    int y0 = Y;                                                                                                        \
    int y1 = Y + 1;                                                                                                    \
    do {                                                                                                               \
        DATA_T4 src00;                                                                                                 \
        DATA_T4 src01;                                                                                                 \
        DATA_T4 src10;                                                                                                 \
        DATA_T4 src11;                                                                                                 \
        DATA_T16 weights0123 = vload16(0, filters_buffer + weight_offset);                                             \
        src00 = read_imageh(src_data, smp_zero, (int2)(x0, y0 * src_z + s));                                           \
        src01 = read_imageh(src_data, smp_zero, (int2)(x1, y0 * src_z + s));                                           \
        src10 = read_imageh(src_data, smp_zero, (int2)(x0, y1 * src_z + s));                                           \
        src11 = read_imageh(src_data, smp_zero, (int2)(x1, y1 * src_z + s));                                           \
        s += 1;                                                                                                        \
        r000 += weights0123.lo.lo * src00.x;                                                                           \
        r001 = fma(weights0123.lo.lo, src01.x, r001);                                                                  \
        r010 = fma(weights0123.lo.lo, src10.x, r010);                                                                  \
        r011 = fma(weights0123.lo.lo, src11.x, r011);                                                                  \
        r000 = fma(weights0123.lo.hi, src00.y, r000);                                                                  \
        r001 = fma(weights0123.lo.hi, src01.y, r001);                                                                  \
        r010 = fma(weights0123.lo.hi, src10.y, r010);                                                                  \
        r011 = fma(weights0123.lo.hi, src11.y, r011);                                                                  \
        r000 = fma(weights0123.hi.lo, src00.z, r000);                                                                  \
        r001 = fma(weights0123.hi.lo, src01.z, r001);                                                                  \
        r010 = fma(weights0123.hi.lo, src10.z, r010);                                                                  \
        r011 = fma(weights0123.hi.lo, src11.z, r011);                                                                  \
        r000 = fma(weights0123.hi.hi, src00.w, r000);                                                                  \
        r001 = fma(weights0123.hi.hi, src01.w, r001);                                                                  \
        r010 = fma(weights0123.hi.hi, src10.w, r010);                                                                  \
        r011 = fma(weights0123.hi.hi, src11.w, r011);                                                                  \
        weight_offset += 16;                                                                                           \
    } while (s < src_z);                                                                                               \
    bias_cache = biases + Z;                                                                                           \
    if (Z + 0 >= dst_z)                                                                                                \
        return;                                                                                                        \
    {                                                                                                                  \
        DATA_T4 bias_val = CONVERT_TO_DATA_T4(bias_cache[0]);                                                          \
        {                                                                                                              \
            DATA_T4 res = CONVERT_TO_DATA_T4(r000) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)((X + 0), (Y + 0) * dst_z + (Z + 0)));                                     \
            write_imageh(dst_data, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
        if ((X + 1) < dst_x) {                                                                                         \
            DATA_T4 res = CONVERT_TO_DATA_T4(r001) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)((X + 1), (Y + 0) * dst_z + (Z + 0)));                                     \
            write_imageh(dst_data, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
        if ((Y + 1) < dst_y) {                                                                                         \
            DATA_T4 res = CONVERT_TO_DATA_T4(r010) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)((X + 0), (Y + 1) * dst_z + (Z + 0)));                                     \
            write_imageh(dst_data, (int2)(((X + 0)), ((Y + 1)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
        if ((X + 1) < dst_x && (Y + 1) < dst_y) {                                                                      \
            DATA_T4 res = CONVERT_TO_DATA_T4(r011) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)((X + 1), (Y + 1) * dst_z + (Z + 0)));                                     \
            write_imageh(dst_data, (int2)(((X + 1)), ((Y + 1)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
    }

#define POWERVR_IS1X11_SRCDEPTH2_BLOCK221(                                                                             \
    src_data, src_data_1, filters_buffer, biases, dst_data, src_x, src_y, src_z, src_w, dst_x, dst_y, dst_z, dst_w)    \
    int X = get_global_id(0) * 2;                                                                                      \
    int Y = get_global_id(1) * 2;                                                                                      \
    int Z = get_global_id(2) * 1;                                                                                      \
    if (X >= dst_x || Y >= dst_y || Z >= dst_z) {                                                                      \
        return;                                                                                                        \
    }                                                                                                                  \
    DATA_T4 r000 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r001 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r010 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r011 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    __global DATA_T4 *bias_cache;                                                                                      \
    int weight_offset = 4 * Z * 4 * src_z;                                                                             \
    int s = 0;                                                                                                         \
    int x0 = X;                                                                                                        \
    int x1 = X + 1;                                                                                                    \
    int y0 = Y;                                                                                                        \
    int y1 = Y + 1;                                                                                                    \
    do {                                                                                                               \
        DATA_T4 src00;                                                                                                 \
        DATA_T4 src01;                                                                                                 \
        DATA_T4 src10;                                                                                                 \
        DATA_T4 src11;                                                                                                 \
        DATA_T16 weights0123 = vload16(0, filters_buffer + weight_offset);                                             \
        src00 = read_imageh(src_data, smp_zero, (int2)(x0, y0 * src_z + s));                                           \
        src01 = read_imageh(src_data, smp_zero, (int2)(x1, y0 * src_z + s));                                           \
        src10 = read_imageh(src_data, smp_zero, (int2)(x0, y1 * src_z + s));                                           \
        src11 = read_imageh(src_data, smp_zero, (int2)(x1, y1 * src_z + s));                                           \
        s += 1;                                                                                                        \
        r000 = fma(weights0123.lo.lo, src00.x, r000);                                                                  \
        r001 = fma(weights0123.lo.lo, src01.x, r001);                                                                  \
        r010 = fma(weights0123.lo.lo, src10.x, r010);                                                                  \
        r011 = fma(weights0123.lo.lo, src11.x, r011);                                                                  \
        r000 = fma(weights0123.lo.hi, src00.y, r000);                                                                  \
        r001 = fma(weights0123.lo.hi, src01.y, r001);                                                                  \
        r010 = fma(weights0123.lo.hi, src10.y, r010);                                                                  \
        r011 = fma(weights0123.lo.hi, src11.y, r011);                                                                  \
        r000 = fma(weights0123.hi.lo, src00.z, r000);                                                                  \
        r001 = fma(weights0123.hi.lo, src01.z, r001);                                                                  \
        r010 = fma(weights0123.hi.lo, src10.z, r010);                                                                  \
        r011 = fma(weights0123.hi.lo, src11.z, r011);                                                                  \
        r000 = fma(weights0123.hi.hi, src00.w, r000);                                                                  \
        r001 = fma(weights0123.hi.hi, src01.w, r001);                                                                  \
        r010 = fma(weights0123.hi.hi, src10.w, r010);                                                                  \
        r011 = fma(weights0123.hi.hi, src11.w, r011);                                                                  \
        DATA_T16 weights4567 = vload16(0, filters_buffer + weight_offset + 16);                                        \
        src00 = read_imageh(src_data, smp_zero, (int2)(x0, y0 * src_z + s));                                           \
        src01 = read_imageh(src_data, smp_zero, (int2)(x1, y0 * src_z + s));                                           \
        src10 = read_imageh(src_data, smp_zero, (int2)(x0, y1 * src_z + s));                                           \
        src11 = read_imageh(src_data, smp_zero, (int2)(x1, y1 * src_z + s));                                           \
        r000 = fma(weights4567.lo.lo, src00.x, r000);                                                                  \
        r001 = fma(weights4567.lo.lo, src01.x, r001);                                                                  \
        r010 = fma(weights4567.lo.lo, src10.x, r010);                                                                  \
        r011 = fma(weights4567.lo.lo, src11.x, r011);                                                                  \
        r000 = fma(weights4567.lo.hi, src00.y, r000);                                                                  \
        r001 = fma(weights4567.lo.hi, src01.y, r001);                                                                  \
        r010 = fma(weights4567.lo.hi, src10.y, r010);                                                                  \
        r011 = fma(weights4567.lo.hi, src11.y, r011);                                                                  \
        r000 = fma(weights4567.hi.lo, src00.z, r000);                                                                  \
        r001 = fma(weights4567.hi.lo, src01.z, r001);                                                                  \
        r010 = fma(weights4567.hi.lo, src10.z, r010);                                                                  \
        r011 = fma(weights4567.hi.lo, src11.z, r011);                                                                  \
        r000 = fma(weights4567.hi.hi, src00.w, r000);                                                                  \
        r001 = fma(weights4567.hi.hi, src01.w, r001);                                                                  \
        r010 = fma(weights4567.hi.hi, src10.w, r010);                                                                  \
        r011 = fma(weights4567.hi.hi, src11.w, r011);                                                                  \
        s += 1;                                                                                                        \
        weight_offset += 32;                                                                                           \
    } while (s < src_z);                                                                                               \
    bias_cache = biases + Z;                                                                                           \
    if (Z + 0 >= dst_z)                                                                                                \
        return;                                                                                                        \
    {                                                                                                                  \
        DATA_T4 bias_val = CONVERT_TO_DATA_T4(bias_cache[0]);                                                          \
        {                                                                                                              \
            DATA_T4 res = CONVERT_TO_DATA_T4(r000) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)((X + 0), (Y + 0) * dst_z + (Z + 0)));                                     \
            write_imageh(dst_data, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
        if ((X + 1) < dst_x) {                                                                                         \
            DATA_T4 res = CONVERT_TO_DATA_T4(r001) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)((X + 1), (Y + 0) * dst_z + (Z + 0)));                                     \
            write_imageh(dst_data, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
        if ((Y + 1) < dst_y) {                                                                                         \
            DATA_T4 res = CONVERT_TO_DATA_T4(r010) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)((X + 0), (Y + 1) * dst_z + (Z + 0)));                                     \
            write_imageh(dst_data, (int2)(((X + 0)), ((Y + 1)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
        if ((X + 1) < dst_x && (Y + 1) < dst_y) {                                                                      \
            DATA_T4 res = CONVERT_TO_DATA_T4(r011) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)((X + 1), (Y + 1) * dst_z + (Z + 0)));                                     \
            write_imageh(dst_data, (int2)(((X + 1)), ((Y + 1)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
    }

#define POWERVR_IS1X11_SRCDEPTH1_BLOCK411(                                                                             \
    src_data, src_data_1, filters_buffer, biases, dst_data, src_x, src_y, src_z, src_w, dst_x, dst_y, dst_z, dst_w)    \
    int X = get_global_id(0) * 4;                                                                                      \
    int Y = get_global_id(1) * 1;                                                                                      \
    int Z = get_global_id(2) * 1;                                                                                      \
    if (X >= dst_x || Y >= dst_y || Z >= dst_z) {                                                                      \
        return;                                                                                                        \
    }                                                                                                                  \
    DATA_T4 r000 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r001 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r010 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r011 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    __global DATA_T4 *bias_cache;                                                                                      \
    int weight_offset = 4 * Z * 4 * src_z;                                                                             \
    int s = 0;                                                                                                         \
    do {                                                                                                               \
        DATA_T4 src00;                                                                                                 \
        DATA_T4 src01;                                                                                                 \
        DATA_T4 src10;                                                                                                 \
        DATA_T4 src11;                                                                                                 \
        DATA_T16 weights0123 = vload16(0, filters_buffer + weight_offset);                                             \
        src00 = read_imageh(src_data, smp_zero, (int2)(((X + 0)), ((Y + 0)) * src_z + (s)));                           \
        src01 = read_imageh(src_data, smp_zero, (int2)(((X + 1)), ((Y + 0)) * src_z + (s)));                           \
        src10 = read_imageh(src_data, smp_zero, (int2)(((X + 2)), ((Y + 0)) * src_z + (s)));                           \
        src11 = read_imageh(src_data, smp_zero, (int2)(((X + 3)), ((Y + 0)) * src_z + (s)));                           \
        s += 1;                                                                                                        \
        r000 = fma(weights0123.lo.lo, src00.x, r000);                                                                  \
        r001 = fma(weights0123.lo.lo, src01.x, r001);                                                                  \
        r010 = fma(weights0123.lo.lo, src10.x, r010);                                                                  \
        r011 = fma(weights0123.lo.lo, src11.x, r011);                                                                  \
        r000 = fma(weights0123.lo.hi, src00.y, r000);                                                                  \
        r001 = fma(weights0123.lo.hi, src01.y, r001);                                                                  \
        r010 = fma(weights0123.lo.hi, src10.y, r010);                                                                  \
        r011 = fma(weights0123.lo.hi, src11.y, r011);                                                                  \
        r000 = fma(weights0123.hi.lo, src00.z, r000);                                                                  \
        r001 = fma(weights0123.hi.lo, src01.z, r001);                                                                  \
        r010 = fma(weights0123.hi.lo, src10.z, r010);                                                                  \
        r011 = fma(weights0123.hi.lo, src11.z, r011);                                                                  \
        r000 = fma(weights0123.hi.hi, src00.w, r000);                                                                  \
        r001 = fma(weights0123.hi.hi, src01.w, r001);                                                                  \
        r010 = fma(weights0123.hi.hi, src10.w, r010);                                                                  \
        r011 = fma(weights0123.hi.hi, src11.w, r011);                                                                  \
        weight_offset += 16;                                                                                           \
    } while (s < src_z);                                                                                               \
    bias_cache = biases + Z;                                                                                           \
    if (Z + 0 >= dst_z)                                                                                                \
        return;                                                                                                        \
    {                                                                                                                  \
        DATA_T4 bias_val = CONVERT_TO_DATA_T4(bias_cache[0]);                                                          \
        {                                                                                                              \
            DATA_T4 res = CONVERT_TO_DATA_T4(r000) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
        if ((X + 1) < dst_x) {                                                                                         \
            DATA_T4 res = CONVERT_TO_DATA_T4(r001) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
        if ((X + 2) < dst_x) {                                                                                         \
            DATA_T4 res = CONVERT_TO_DATA_T4(r010) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 2)), ((Y + 0)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 2)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
        if ((X + 3) < dst_x) {                                                                                         \
            DATA_T4 res = CONVERT_TO_DATA_T4(r011) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 3)), ((Y + 0)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 3)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
    }

#define POWERVR_IS1X11_SRCDEPTH2_BLOCK411(                                                                             \
    src_data, src_data_1, filters_buffer, biases, dst_data, src_x, src_y, src_z, src_w, dst_x, dst_y, dst_z, dst_w)    \
    int X = get_global_id(0) * 4;                                                                                      \
    int Y = get_global_id(1) * 1;                                                                                      \
    int Z = get_global_id(2) * 1;                                                                                      \
    if (X >= dst_x || Y >= dst_y || Z >= dst_z) {                                                                      \
        return;                                                                                                        \
    }                                                                                                                  \
    DATA_T4 r000 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r001 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r010 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    DATA_T4 r011 = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f);                                                                  \
    __global DATA_T4 *bias_cache;                                                                                      \
    int weight_offset = 4 * Z * 4 * src_z;                                                                             \
    int s = 0;                                                                                                         \
    do {                                                                                                               \
        DATA_T4 src00;                                                                                                 \
        DATA_T4 src01;                                                                                                 \
        DATA_T4 src10;                                                                                                 \
        DATA_T4 src11;                                                                                                 \
        DATA_T16 weights0123 = vload16(0, filters_buffer + weight_offset);                                             \
        src00 = read_imageh(src_data, smp_zero, (int2)(((X + 0)), ((Y + 0)) * src_z + (s)));                           \
        src01 = read_imageh(src_data, smp_zero, (int2)(((X + 1)), ((Y + 0)) * src_z + (s)));                           \
        src10 = read_imageh(src_data, smp_zero, (int2)(((X + 2)), ((Y + 0)) * src_z + (s)));                           \
        src11 = read_imageh(src_data, smp_zero, (int2)(((X + 3)), ((Y + 0)) * src_z + (s)));                           \
        s += 1;                                                                                                        \
        r000 = fma(weights0123.lo.lo, src00.x, r000);                                                                  \
        r001 = fma(weights0123.lo.lo, src01.x, r001);                                                                  \
        r010 = fma(weights0123.lo.lo, src10.x, r010);                                                                  \
        r011 = fma(weights0123.lo.lo, src11.x, r011);                                                                  \
        r000 = fma(weights0123.lo.hi, src00.y, r000);                                                                  \
        r001 = fma(weights0123.lo.hi, src01.y, r001);                                                                  \
        r010 = fma(weights0123.lo.hi, src10.y, r010);                                                                  \
        r011 = fma(weights0123.lo.hi, src11.y, r011);                                                                  \
        r000 = fma(weights0123.hi.lo, src00.z, r000);                                                                  \
        r001 = fma(weights0123.hi.lo, src01.z, r001);                                                                  \
        r010 = fma(weights0123.hi.lo, src10.z, r010);                                                                  \
        r011 = fma(weights0123.hi.lo, src11.z, r011);                                                                  \
        r000 = fma(weights0123.hi.hi, src00.w, r000);                                                                  \
        r001 = fma(weights0123.hi.hi, src01.w, r001);                                                                  \
        r010 = fma(weights0123.hi.hi, src10.w, r010);                                                                  \
        r011 = fma(weights0123.hi.hi, src11.w, r011);                                                                  \
        DATA_T16 weights4567 = vload16(0, filters_buffer + weight_offset + 16);                                        \
        src00 = read_imageh(src_data, smp_zero, (int2)(((X + 0)), ((Y + 0)) * src_z + (s)));                           \
        src01 = read_imageh(src_data, smp_zero, (int2)(((X + 1)), ((Y + 0)) * src_z + (s)));                           \
        src10 = read_imageh(src_data, smp_zero, (int2)(((X + 2)), ((Y + 0)) * src_z + (s)));                           \
        src11 = read_imageh(src_data, smp_zero, (int2)(((X + 3)), ((Y + 0)) * src_z + (s)));                           \
        s += 1;                                                                                                        \
        r000 = fma(weights4567.lo.lo, src00.x, r000);                                                                  \
        r001 = fma(weights4567.lo.lo, src01.x, r001);                                                                  \
        r010 = fma(weights4567.lo.lo, src10.x, r010);                                                                  \
        r011 = fma(weights4567.lo.lo, src11.x, r011);                                                                  \
        r000 = fma(weights4567.lo.hi, src00.y, r000);                                                                  \
        r001 = fma(weights4567.lo.hi, src01.y, r001);                                                                  \
        r010 = fma(weights4567.lo.hi, src10.y, r010);                                                                  \
        r011 = fma(weights4567.lo.hi, src11.y, r011);                                                                  \
        r000 = fma(weights4567.hi.lo, src00.z, r000);                                                                  \
        r001 = fma(weights4567.hi.lo, src01.z, r001);                                                                  \
        r010 = fma(weights4567.hi.lo, src10.z, r010);                                                                  \
        r011 = fma(weights4567.hi.lo, src11.z, r011);                                                                  \
        r000 = fma(weights4567.hi.hi, src00.w, r000);                                                                  \
        r001 = fma(weights4567.hi.hi, src01.w, r001);                                                                  \
        r010 = fma(weights4567.hi.hi, src10.w, r010);                                                                  \
        r011 = fma(weights4567.hi.hi, src11.w, r011);                                                                  \
        weight_offset += 32;                                                                                           \
    } while (s < src_z);                                                                                               \
    bias_cache = biases + Z;                                                                                           \
    if (Z + 0 >= dst_z)                                                                                                \
        return;                                                                                                        \
    {                                                                                                                  \
        DATA_T4 bias_val = CONVERT_TO_DATA_T4(bias_cache[0]);                                                          \
        {                                                                                                              \
            DATA_T4 res = CONVERT_TO_DATA_T4(r000) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 0)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
        if ((X + 1) < dst_x) {                                                                                         \
            DATA_T4 res = CONVERT_TO_DATA_T4(r001) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 1)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
        if ((X + 2) < dst_x) {                                                                                         \
            DATA_T4 res = CONVERT_TO_DATA_T4(r010) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 2)), ((Y + 0)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 2)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
        if ((X + 3) < dst_x) {                                                                                         \
            DATA_T4 res = CONVERT_TO_DATA_T4(r011) + bias_val;                                                         \
            res = ACT_VEC_F(DATA_T, res);                                                                              \
            MERGEADD(res, src_data_1, (int2)(((X + 3)), ((Y + 0)) * dst_z + (Z + 0)));                                 \
            write_imageh(dst_data, (int2)(((X + 3)), ((Y + 0)) * dst_z + (Z + 0)), res);                               \
        }                                                                                                              \
    }

/********  FP16 KERNELS ********/
#define DATA_T half
#define DATA_T2 half2
#define DATA_T3 half3
#define DATA_T4 half4
#define DATA_T8 half8
#define DATA_T16 half16
#define CONVERT_TO_DATA_T4(x) convert_half4(x)

ADD_SINGLE_KERNEL(alignWeight_powervr_FP16,
                  (__global const DATA_T *src,
                   __global DATA_T *dst,
                   const unsigned int batch,
                   const unsigned int channel,
                   const int kh,
                   const int kw,
                   const int dst_groups,
                   const int src_slices,
                   const int out_group_size){
                      ALIGNWEIGHT_POWERVR(src, dst, batch, channel, kh, kw, dst_groups, src_slices, out_group_size)})

// kernels without activation func
#define ACT_VEC_F(VEC_T, x) x
#define IMAGE2D_INPUTS SINGLE_SRC_DATA
#define MERGEADD(x, ...) {}
ADD_SINGLE_KERNEL(powervr_is1x11_srcdepth1_block111_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK111(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(powervr_is1x11_srcdepth1_block211_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK211(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(powervr_is1x11_srcdepth1_block212_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK212(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(powervr_is1x11_srcdepth1_block222_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T16 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK222(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(powervr_is1x11_srcdepth2_block111_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH2_BLOCK111(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(powervr_is1x11_srcdepth2_block211_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH2_BLOCK211(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(powervr_is1x11_srcdepth2_block212_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH2_BLOCK212(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(powervr_is1x10_srcdepth1_block111_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T4 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH1_BLOCK111(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(powervr_is1x10_srcdepth1_block211_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T4 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH1_BLOCK211(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(powervr_is1x10_srcdepth1_block212_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH1_BLOCK212(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(powervr_is1x10_srcdepth2_block111_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T4 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH2_BLOCK111(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(powervr_is1x10_srcdepth2_block211_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T4 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH2_BLOCK211(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(powervr_is1x10_srcdepth2_block212_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T4 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH2_BLOCK212(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(powervr_is1x11_srcdepth4_block111_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH4_BLOCK111(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(powervr_is1x11_srcdepth1_block221_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK221(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(powervr_is1x11_srcdepth2_block221_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH2_BLOCK221(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(powervr_is1x11_srcdepth1_block411_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK411(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(powervr_is1x11_srcdepth2_block411_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH2_BLOCK411(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

#undef MERGEADD
#undef IMAGE2D_INPUTS

#define IMAGE2D_INPUTS MERGEADD_INPUTS
#define MERGEADD(x, IMAGE, COORD) MERGEADD_IMAGE2D(x, IMAGE, COORD)
ADD_SINGLE_KERNEL(MERGEADDpowervr_is1x11_srcdepth1_block111_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK111(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(MERGEADDpowervr_is1x11_srcdepth1_block211_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK211(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(MERGEADDpowervr_is1x11_srcdepth1_block212_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK212(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(MERGEADDpowervr_is1x11_srcdepth1_block222_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T16 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK222(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(MERGEADDpowervr_is1x11_srcdepth2_block111_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH2_BLOCK111(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(MERGEADDpowervr_is1x11_srcdepth2_block211_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH2_BLOCK211(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(MERGEADDpowervr_is1x11_srcdepth2_block212_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH2_BLOCK212(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(MERGEADDpowervr_is1x10_srcdepth1_block111_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T4 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH1_BLOCK111(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(MERGEADDpowervr_is1x10_srcdepth1_block211_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T4 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH1_BLOCK211(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(MERGEADDpowervr_is1x10_srcdepth1_block212_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH1_BLOCK212(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(MERGEADDpowervr_is1x10_srcdepth2_block111_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T4 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH2_BLOCK111(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(MERGEADDpowervr_is1x10_srcdepth2_block211_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T4 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH2_BLOCK211(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(MERGEADDpowervr_is1x10_srcdepth2_block212_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T4 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH2_BLOCK212(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(MERGEADDpowervr_is1x11_srcdepth4_block111_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH4_BLOCK111(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(MERGEADDpowervr_is1x11_srcdepth1_block221_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK221(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(MERGEADDpowervr_is1x11_srcdepth2_block221_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH2_BLOCK221(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(MERGEADDpowervr_is1x11_srcdepth1_block411_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK411(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(MERGEADDpowervr_is1x11_srcdepth2_block411_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH2_BLOCK411(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})
#undef MERGEADD
#undef IMAGE2D_INPUTS

#undef ACT_VEC_F  // NONE

// apply RELU to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU_CLAMP_VEC_F(VEC_T, x)
#define IMAGE2D_INPUTS SINGLE_SRC_DATA
#define MERGEADD(x, ...)                                                                                               \
    {}
ADD_SINGLE_KERNEL(RELUpowervr_is1x11_srcdepth1_block111_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK111(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELUpowervr_is1x11_srcdepth1_block211_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK211(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELUpowervr_is1x11_srcdepth1_block212_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK212(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELUpowervr_is1x11_srcdepth1_block222_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T16 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK222(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELUpowervr_is1x11_srcdepth2_block111_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH2_BLOCK111(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELUpowervr_is1x11_srcdepth2_block211_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH2_BLOCK211(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELUpowervr_is1x11_srcdepth2_block212_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH2_BLOCK212(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELUpowervr_is1x10_srcdepth1_block111_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T4 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH1_BLOCK111(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(RELUpowervr_is1x10_srcdepth1_block211_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T4 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH1_BLOCK211(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(RELUpowervr_is1x10_srcdepth1_block212_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH1_BLOCK212(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(RELUpowervr_is1x10_srcdepth2_block111_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T4 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH2_BLOCK111(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(RELUpowervr_is1x10_srcdepth2_block211_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T4 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH2_BLOCK211(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(RELUpowervr_is1x10_srcdepth2_block212_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T4 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH2_BLOCK212(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(RELUpowervr_is1x11_srcdepth4_block111_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH4_BLOCK111(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELUpowervr_is1x11_srcdepth1_block221_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK221(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELUpowervr_is1x11_srcdepth2_block221_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH2_BLOCK221(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELUpowervr_is1x11_srcdepth1_block411_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK411(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELUpowervr_is1x11_srcdepth2_block411_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH2_BLOCK411(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

#undef MERGEADD
#undef IMAGE2D_INPUTS

#define IMAGE2D_INPUTS MERGEADD_INPUTS
#define MERGEADD(x, IMAGE, COORD) MERGEADD_IMAGE2D(x, IMAGE, COORD)
ADD_SINGLE_KERNEL(RELUMERGEADDpowervr_is1x11_srcdepth1_block111_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK111(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELUMERGEADDpowervr_is1x11_srcdepth1_block211_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK211(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELUMERGEADDpowervr_is1x11_srcdepth1_block212_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK212(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELUMERGEADDpowervr_is1x11_srcdepth1_block222_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T16 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK222(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELUMERGEADDpowervr_is1x11_srcdepth2_block111_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH2_BLOCK111(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELUMERGEADDpowervr_is1x11_srcdepth2_block211_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH2_BLOCK211(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELUMERGEADDpowervr_is1x11_srcdepth2_block212_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH2_BLOCK212(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELUMERGEADDpowervr_is1x10_srcdepth1_block111_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T4 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH1_BLOCK111(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(RELUMERGEADDpowervr_is1x10_srcdepth1_block211_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T4 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH1_BLOCK211(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(RELUMERGEADDpowervr_is1x10_srcdepth1_block212_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH1_BLOCK212(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(RELUMERGEADDpowervr_is1x10_srcdepth2_block111_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T4 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH2_BLOCK111(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(RELUMERGEADDpowervr_is1x10_srcdepth2_block211_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T4 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH2_BLOCK211(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(RELUMERGEADDpowervr_is1x10_srcdepth2_block212_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T4 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH2_BLOCK212(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(RELUMERGEADDpowervr_is1x11_srcdepth4_block111_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH4_BLOCK111(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELUMERGEADDpowervr_is1x11_srcdepth1_block221_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK221(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELUMERGEADDpowervr_is1x11_srcdepth2_block221_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH2_BLOCK221(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELUMERGEADDpowervr_is1x11_srcdepth1_block411_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK411(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELUMERGEADDpowervr_is1x11_srcdepth2_block411_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH2_BLOCK411(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})
#undef MERGEADD
#undef IMAGE2D_INPUTS

#undef ACT_VEC_F  // RELU

// apply RELU6 to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU6_CLAMP_VEC_F(VEC_T, x)
#define IMAGE2D_INPUTS SINGLE_SRC_DATA
#define MERGEADD(x, ...)                                                                                               \
    {}
ADD_SINGLE_KERNEL(RELU6powervr_is1x11_srcdepth1_block111_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK111(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELU6powervr_is1x11_srcdepth1_block211_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK211(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELU6powervr_is1x11_srcdepth1_block212_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK212(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELU6powervr_is1x11_srcdepth1_block222_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T16 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK222(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELU6powervr_is1x11_srcdepth2_block111_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH2_BLOCK111(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELU6powervr_is1x11_srcdepth2_block211_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH2_BLOCK211(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELU6powervr_is1x11_srcdepth2_block212_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH2_BLOCK212(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELU6powervr_is1x10_srcdepth1_block111_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T4 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH1_BLOCK111(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(RELU6powervr_is1x10_srcdepth1_block211_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T4 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH1_BLOCK211(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(RELU6powervr_is1x10_srcdepth1_block212_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH1_BLOCK212(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(RELU6powervr_is1x10_srcdepth2_block111_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T4 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH2_BLOCK111(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(RELU6powervr_is1x10_srcdepth2_block211_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T4 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH2_BLOCK211(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(RELU6powervr_is1x10_srcdepth2_block212_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T4 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH2_BLOCK212(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(RELU6powervr_is1x11_srcdepth4_block111_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH4_BLOCK111(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELU6powervr_is1x11_srcdepth1_block221_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK221(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELU6powervr_is1x11_srcdepth2_block221_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH2_BLOCK221(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELU6powervr_is1x11_srcdepth1_block411_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK411(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELU6powervr_is1x11_srcdepth2_block411_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH2_BLOCK411(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

#undef MERGEADD
#undef IMAGE2D_INPUTS

#define IMAGE2D_INPUTS MERGEADD_INPUTS
#define MERGEADD(x, IMAGE, COORD) MERGEADD_IMAGE2D(x, IMAGE, COORD)
ADD_SINGLE_KERNEL(RELU6MERGEADDpowervr_is1x11_srcdepth1_block111_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK111(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELU6MERGEADDpowervr_is1x11_srcdepth1_block211_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK211(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELU6MERGEADDpowervr_is1x11_srcdepth1_block212_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK212(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELU6MERGEADDpowervr_is1x11_srcdepth1_block222_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T16 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK222(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELU6MERGEADDpowervr_is1x11_srcdepth2_block111_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH2_BLOCK111(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELU6MERGEADDpowervr_is1x11_srcdepth2_block211_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH2_BLOCK211(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELU6MERGEADDpowervr_is1x11_srcdepth2_block212_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH2_BLOCK212(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELU6MERGEADDpowervr_is1x10_srcdepth1_block111_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T4 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH1_BLOCK111(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(RELU6MERGEADDpowervr_is1x10_srcdepth1_block211_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T4 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH1_BLOCK211(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(RELU6MERGEADDpowervr_is1x10_srcdepth1_block212_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH1_BLOCK212(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(RELU6MERGEADDpowervr_is1x10_srcdepth2_block111_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T4 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH2_BLOCK111(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(RELU6MERGEADDpowervr_is1x10_srcdepth2_block211_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T4 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH2_BLOCK211(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(RELU6MERGEADDpowervr_is1x10_srcdepth2_block212_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T4 *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w,
                   int kw,
                   int kh,
                   int dilation_x,
                   int dilation_y,
                   int stride_x,
                   int stride_y,
                   int padding_x,
                   int padding_y){POWERVR_IS1X10_SRCDEPTH2_BLOCK212(src_data,
                                                                    src_data_1,
                                                                    filters_buffer,
                                                                    biases,
                                                                    dst_data,
                                                                    src_x,
                                                                    src_y,
                                                                    src_z,
                                                                    src_w,
                                                                    dst_x,
                                                                    dst_y,
                                                                    dst_z,
                                                                    dst_w,
                                                                    kw,
                                                                    kh,
                                                                    dilation_x,
                                                                    dilation_y,
                                                                    stride_x,
                                                                    stride_y,
                                                                    padding_x,
                                                                    padding_y)})

ADD_SINGLE_KERNEL(RELU6MERGEADDpowervr_is1x11_srcdepth4_block111_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH4_BLOCK111(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELU6MERGEADDpowervr_is1x11_srcdepth1_block221_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK221(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELU6MERGEADDpowervr_is1x11_srcdepth2_block221_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH2_BLOCK221(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELU6MERGEADDpowervr_is1x11_srcdepth1_block411_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH1_BLOCK411(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})

ADD_SINGLE_KERNEL(RELU6MERGEADDpowervr_is1x11_srcdepth2_block411_FP16,
                  (IMAGE2D_INPUTS,
                   __global DATA_T *filters_buffer,
                   __global DATA_T4 *biases,
                   __write_only image2d_t dst_data,
                   int src_x,
                   int src_y,
                   int src_z,
                   int src_w,
                   int dst_x,
                   int dst_y,
                   int dst_z,
                   int dst_w){POWERVR_IS1X11_SRCDEPTH2_BLOCK411(src_data,
                                                                src_data_1,
                                                                filters_buffer,
                                                                biases,
                                                                dst_data,
                                                                src_x,
                                                                src_y,
                                                                src_z,
                                                                src_w,
                                                                dst_x,
                                                                dst_y,
                                                                dst_z,
                                                                dst_w)})
#undef MERGEADD
#undef IMAGE2D_INPUTS

#undef ACT_VEC_F  // RELU6

#undef CONVERT_TO_DATA_T4
#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // half

}  // namespace gpu
}  // namespace ud
}  // namespace enn
