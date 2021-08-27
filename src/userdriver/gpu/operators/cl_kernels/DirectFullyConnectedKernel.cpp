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
 * @file    DirectFullyConnectedKernel.cpp
 * @brief
 * @details
 * @version
 */


#include "userdriver/gpu/common/CLKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define FC_WEIGHT_CVT(weight, weight_cvt, matrix_height, matrix_width) \
    int g0 = get_global_id(0); \
    int g1 = get_global_id(1); \
    int in_h = g1 / 16; \
    int in_w = g0 * 16 + g1 % 16; \
    int input_index = in_h * matrix_width + in_w; \
    int output_index = g0 * matrix_height * 16 + g1; \
    if (in_h >= matrix_height || in_w >= matrix_width) { \
        weight_cvt[output_index] = (DATA_T)0.0f; \
    } else { \
        weight_cvt[output_index] = weight[input_index]; \
    }

#define FC_DIRECT_OPT(input, weight, bias, output, matrix_height, matrix_width) \
    int g0 = get_global_id(0); \
    if (g0 >= matrix_height) return; \
    DATA_T16 in = (DATA_T16)0.0f; \
    DATA_T16 w = (DATA_T16)0.0f; \
    DATA_T16 out = (DATA_T16)0.0f; \
    int weight_index; \
    int algined_width = (matrix_width + 15) / 16 * 16; \
    for (int i = 0; i < matrix_width / 16; i++) { \
        in = vload16(0, input + i * 16); \
        weight_index = i * 16 * matrix_height + g0 * 16; \
        w = vload16(0, weight + weight_index); \
        out += in*w; \
    } \
    for (int i = matrix_width / 16 * 16; i < matrix_width; i++) { \
        out.s0 += input[i] * weight[i / 16 * 16 * matrix_height + g0 * 16 + i%16]; \
    } \
    out.s01234567 += out.s89abcdef; \
    out.s0123 += out.s4567; \
    out.s0 += out.s1 + out.s2 + out.s3 + bias[g0]; \
    output[g0] = out.s0;

// FP16 kernels
#define DATA_T half
#define DATA_T2 half2
#define DATA_T3 half3
#define DATA_T4 half4
#define DATA_T8 half8
#define DATA_T16 half16

ADD_SINGLE_KERNEL(fc_weight_cvt_FP16, (__global const DATA_T *weight,
                                       __global DATA_T *weight_cvt,
                                       int matrix_height,
                                       int matrix_width) {
    FC_WEIGHT_CVT(weight, weight_cvt, matrix_height, matrix_width)
})

ADD_SINGLE_KERNEL(fc_direct_opt_FP16, (__global const DATA_T *input,
                                       __global const DATA_T *weight,
                                       __global const DATA_T *bias,
                                       __global DATA_T *output,
                                       int matrix_height,
                                       int matrix_width) {
    FC_DIRECT_OPT(input, weight, bias, output, matrix_height, matrix_width)
})

ADD_SINGLE_KERNEL(fc_direct_FP16, (__global const half *input,
                                             __global const half *weight,
                                             __global const half *bias,
                                             __global half *output,
                                             int matrix_height,
                                             int matrix_width) {
    int globalID0 = get_global_id(0);
    int inner = 4;
    if (globalID0 >= matrix_height) {
        return;
    }

    __global const half *p_w = weight + globalID0 * matrix_width;

    __global const half *p_in = input;
    int stride = (matrix_width / 16) * 16;
    int i = 0;
    half result = 0.0f;
    half tmp = 0.0f;
    // 9830 use 16 unroll better
    for (; i + 15 < matrix_width; i += 16) {
        half16 vin = vload16(0, p_in + i);
        half16 vw = vload16(0, p_w + i);

        tmp = 0.0f;
        tmp += dot(vin.s0123, vw.s0123);
        tmp += dot(vin.s4567, vw.s4567);
        tmp += dot(vin.s89ab, vw.s89ab);
        tmp += dot(vin.scdef, vw.scdef);

        result += tmp;
    }
    // 9820 use 8 unroll better
    for (; i + 7 < matrix_width; i += 8) {
        half8 vin = vload8(0, p_in + i);
        half8 vw = vload8(0, p_w + i);
        tmp = 0.0f;
        tmp += dot(vin.s0123, vw.s0123);
        tmp += dot(vin.s4567, vw.s4567);

        result += tmp;
    }
    for (; i + 3 < matrix_width; i += 4) {
        half4 vin = vload4(0, p_in + i);
        half4 vw = vload4(0, p_w + i);

        tmp = 0.0f;
        tmp += dot(vin, vw);

        result += tmp;
    }
    for (; i < matrix_width; ++i) {
        half vin = p_in[i];
        half vw = p_w[i];
        result += vin * vw;
    }
    output[globalID0] = result + bias[globalID0];
})

#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // half

// FP32 kernels
#define DATA_T float
#define DATA_T2 float2
#define DATA_T3 float3
#define DATA_T4 float4
#define DATA_T8 float8
#define DATA_T16 float16

ADD_SINGLE_KERNEL(fc_weight_cvt_FP32, (__global const DATA_T *weight,
                                       __global DATA_T *weight_cvt,
                                       int matrix_height,
                                       int matrix_width) {
    FC_WEIGHT_CVT(weight, weight_cvt, matrix_height, matrix_width)
})

ADD_SINGLE_KERNEL(fc_direct_opt_FP32, (__global const DATA_T *input,
                                       __global const DATA_T *weight,
                                       __global const DATA_T *bias,
                                       __global DATA_T *output,
                                       int matrix_height,
                                       int matrix_width) {
    FC_DIRECT_OPT(input, weight, bias, output, matrix_height, matrix_width)
})

ADD_SINGLE_KERNEL(fc_direct_FP32, (__global const float *input,
                                             __global const float *weight,
                                             __global const float *bias,
                                             __global float *output,
                                             int matrix_height,
                                             int matrix_width) {
    int globalID0 = get_global_id(0);
    int inner = 4;
    if (globalID0 >= matrix_height) {
        return;
    }

    __global const float *p_w = weight + globalID0 * matrix_width;
    // __global const half* p1 = p0 + matrix_width;
    // __global const half* p2 = p1 + matrix_width;
    // __global const half* p3 = p2 + matrix_width;
    __global const float *p_in = input;
    int stride = (matrix_width / 16) * 16;
    int i = 0;
    float result = 0.0f;
    // 9830 use 16 unroll better
    for (; i + 15 < matrix_width; i += 16) {
        float16 vin = vload16(0, p_in + i);
        float16 vw = vload16(0, p_w + i);
        result += dot(vin.s0123, vw.s0123);
        result += dot(vin.s4567, vw.s4567);
        result += dot(vin.s89ab, vw.s89ab);
        result += dot(vin.scdef, vw.scdef);
    }
    // 9820 use 8 unroll better
    for (; i + 7 < matrix_width; i += 8) {
        float8 vin = vload8(0, p_in + i);
        float8 vw = vload8(0, p_w + i);
        result += dot(vin.s0123, vw.s0123);
        result += dot(vin.s4567, vw.s4567);
    }
    for (; i + 3 < matrix_width; i += 4) {
        float4 vin = vload4(0, p_in + i);
        float4 vw = vload4(0, p_w + i);
        result += dot(vin, vw);
    }
    for (; i < matrix_width; ++i) {
        float vin = p_in[i];
        float vw = p_w[i];
        result += vin * vw;
    }
    output[globalID0] = result + bias[globalID0];
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
