/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or
 * distributed, transmitted, transcribed, stored in a retrieval system or
 * translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed to third parties
 * without the express written permission of Samsung Electronics.
 */

#ifndef USERDRIVER_GPU_CL_OPERATORS_CL_KERNELS_HPP_
#define USERDRIVER_GPU_CL_OPERATORS_CL_KERNELS_HPP_

#include <string.h>
#include <map>
#include <string>
#include <vector>

namespace enn {
namespace ud {
namespace gpu {

// ToDo(all): reduce LOC score
#ifdef __SHARP_X
#undef __SHARP_X
#endif
#define __SHARP_X(x) #x
#ifdef _STR
#undef _STR
#endif
#define _STR(s) __SHARP_X(s)

#define __X_HEADER(x) x##_header
#define _X_HEADER(x) __X_HEADER(x)

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

enum PrivateKernelHeader {
    DEFINE_AT7X7 = 0,
    DEFINE_AT5X5 = 1,
    DEFINE_AT5X5_F = 2,
    DEFINE_BT8X8 = 3,
    DEFINE_AT8X8 = 4,
    DEFINE_SATURATING_ROUNDING_DOUBLING_HIGH_MUL = 5,
    DEFINE_QUANTIZED_MULTIPLIER_SMALLER_THAN_ONE = 6,
    DEFINE_ROUNDING_DIVIDE_BY_POT = 7,
    DEFINE_REQUANTIZED = 8,
    DEFINE_REQUANTIZED_ACT = 9,
    DEFINE_FUNC_SATURATING_ROUNDING_DOUBLING_HIGH_MUL = 10,
    DEFINE_FUNC_ROUNDING_DIVIDE_BY_POT = 11,
    DEFINE_INTERSECT_BBOX = 12,
    DEFINE_BBOX_SIZE = 13,
    DEFINE_COMPUTE_INTERSECTION_OVER_UNION = 14,
    DEFINE_COMPUTE_HALF_INTERSECTION_OVER_UNION = 15,
    DEFINE_SMP_NONE = 16,
    NUM_OF_PRIVATE_KERNEL_HEADER
};

constexpr const char *private_kernel_header[] = {
    "constant half At7x7[16] = {\n"
    "1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 1.000000, 0.000000,\n"
    "0.000000, 0.707107, -0.707107, 1.414214, -1.414214, 2.121320, -2.121320, 1.000000,\n"
    "};\n",

    "constant half At5x5[32] = {\n"
    "1.000000,  1.000000,  1.000000,  1.000000,  1.000000, 1.000000,  1.000000, 0.000000,\n"
    "0.000000,  0.707106, -0.707106,  1.414213, -1.414213, 2.121320, -2.121320, 0.000000,\n"
    "0.000000,  0.500000,  0.500000,  2.000000,  2.000000, 4.500000,  4.500000, 0.000000,\n"
    "0.000000,  0.353553, -0.353553,  2.828427, -2.828427, 9.545940, -9.545940, 1.000000,\n"
    "};\n",

    "constant float At5x5_f[32] = {\n"
    "1.000000,  1.000000,  1.000000,  1.000000,  1.000000, 1.000000,  1.000000, 0.000000,\n"
    "0.000000,  0.707106, -0.707106,  1.414213, -1.414213, 2.121320, -2.121320, 0.000000,\n"
    "0.000000,  0.500000,  0.500000,  2.000000,  2.000000, 4.500000,  4.500000, 0.000000,\n"
    "0.000000,  0.353553, -0.353553,  2.828427, -2.828427, 9.545940, -9.545940, 1.000000,\n"
    "};\n",

    "constant half Bt8x8[64] = {\n"
    "1.000000f,  0.000000f, -2.722228f, -0.000000f,  1.555556f, 0.000000f,-0.222222f, 0.000000f,\n"
    "0.000000f,  1.060660f,  1.500000f, -0.766032f, -1.083333f, 0.117851f, 0.166667f, 0.000000f,\n"
    "0.000000f, -1.060660f,  1.500000f,  0.766032f, -1.083333f,-0.117851f, 0.166667f, 0.000000f,\n"
    "0.000000f, -0.212132f, -0.150000f,  0.471404f,  0.333333f,-0.094280f,-0.066666f, 0.000000f,\n"
    "0.000000f,  0.212132f, -0.150000f, -0.471404f,  0.333333f, 0.094280f,-0.066666f, 0.000000f,\n"
    "0.000000f,  0.023570f,  0.011111f, -0.058925f, -0.027777f, 0.023570f, 0.011111f, 0.000000f,\n"
    "0.000000f, -0.023570f,  0.011111f,  0.058925f, -0.027777f,-0.023570f, 0.011111f, 0.000000f,\n"
    "0.000000f, -4.500000f, -0.000000f, 12.250000f,  0.000000f,-7.000000f, 0.000000f, 1.000000f,\n"
    "};\n",

    "constant half At8x8[48] = {\n"
    "1.000000f, 1.000000f,  1.000000f, 1.000000f, 1.000000f,  1.000000f,  1.000000f, 0.000000f,\n"
    "0.000000f, 0.707106f, -0.707106f, 1.414213f,-1.414213f,  2.121320f, -2.121320f, 0.000000f,\n"
    "0.000000f, 0.500000f,  0.500000f, 2.000000f, 2.000000f,  4.500000f,  4.500000f, 0.000000f,\n"
    "0.000000f, 0.353553f, -0.353553f, 2.828427f,-2.828427f,  9.545940f, -9.545940f, 0.000000f,\n"
    "0.000000f, 0.250000f,  0.250000f, 4.000000f, 4.000000f, 20.250000f, 20.250000f, 0.000000f,\n"
    "0.000000f, 0.176776f, -0.176776f, 5.656853f,-5.656853f, 42.956726f,-42.956726f, 1.000000f,\n"
    "};\n",

    "#define saturatingRoundingDoublingHighMul(a, b, out, overflow, ab_64, nudge)\\\n"
    "        {                                                                   \\\n"
    "           overflow = (a == b) && (a == INT_MIN);                           \\\n"
    "           ab_64 = (long)a*(long)b;                                         \\\n"
    "           nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));                \\\n"
    "           out = overflow ? INT_MAX : ((int)((ab_64 + nudge) / ((long)1 << 31))); \\\n"
    "        }\n",

    "void quantizeMultiplierSmallerThanOne(float rm, int *qm, int *rs) {                  \\\n"
    "    if (rm == 0.) {                                                                  \\\n"
    "        *qm = 0;                                                                     \\\n"
    "        *rs = 0;                                                                     \\\n"
    "        return;                                                                      \\\n"
    "    }                                                                                \\\n"
    "    float q = frexp(rm, rs);                                                         \\\n"
    "    *rs *= -1;                                                                       \\\n"
    "    long int q_fixed = (long)round(q * (1 << 31));                                   \\\n"
    "    if (q_fixed == (1 << 31)) {                                                      \\\n"
    "        q_fixed /= 2;                                                                \\\n"
    "        --*rs;                                                                       \\\n"
    "    }                                                                                \\\n"
    "    *qm = (int)q_fixed;                                                              \\\n"
    "}\n",

    "#define roundingDivideByPOT(in, exponent, out, mask, remainder, threshold)  \\\n"
    "        {                                                                   \\\n"
    "            mask = (((long)1 << exponent) - 1);                                 \\\n"
    "            remainder = in & mask;                                          \\\n"
    "            threshold = (mask >> 1) + (((in < 0) ?  ~0 : 0) & 1);           \\\n"
    "            out = (in >> exponent) + (((remainder > threshold) ? ~0 : 0) & 1); \\\n"
    "        }\n",

    "#define reQuantized(x, output_multiplier, output_shift, outputOffset, activation_min, \\\n"
    "                    activation_max, overflow, ab_64, nudge, mask, remainder, threshold)   "
    "         \\\n"
    "        {                                                                                 "
    "         \\\n"
    "            saturatingRoundingDoublingHighMul(x, output_multiplier, x, overflow, ab_64, "
    "nudge);    \\\n"
    "            roundingDivideByPOT(x, output_shift, x, mask, remainder, threshold);          "
    "         \\\n"
    "            x += outputOffset;                                                            "
    "         \\\n"
    "            x = max(x, activation_min);                                                   "
    "         \\\n"
    "            x = min(x, activation_max);                                                   "
    "         \\\n"
    "        }\n",

    "#define reQuantizedAct(x, activation_min, activation_max)                                 "
    "         \\\n"
    "        {                                                                                 "
    "         \\\n"
    "            x = max(x, activation_min);                                                   "
    "         \\\n"
    "            x = min(x, activation_max);                                                   "
    "         \\\n"
    "        }\n",

    "int saturating_RoundingDoublingHighMul(int a, int b) {                                \\\n"
    "    bool overflow = (a == b) && (a == -INT_MIN);                                      \\\n"
    "    long a_64 = a;                                                                    \\\n"
    "    long b_64 = b;                                                                    \\\n"
    "    long ab_64 = a_64 * b_64;                                                         \\\n"
    "    int nudge = ab_64 >= 0 ? (1 << 30) : (1 - (1 << 30));                             \\\n"
    "    int ab_x2_high32 = (int)((ab_64 + nudge) / ((long)1 << 31));                      \\\n"
    "    return overflow ? INT_MAX : ab_x2_high32;                                         \\\n"
    "}\n                                                                                   \\\n",

    "int rounding_DivideByPOT(int x, int exponent) {                                       \\\n"
    "    int mask = (((long)1 << exponent) - 1);                                           \\\n"
    "    const int remainder = x & mask;                                                   \\\n"
    "    const int threshold = (mask >> 1) + (((x < 0) ? ~0 : 0) & 1);                     \\\n"
    "    return (x >> exponent) + (((remainder > threshold) ? ~0 : 0) & 1);                \\\n"
    "}\n",

    "void IntersectBBox(__global float *b1, __global float *b2, float *intersectBbox) {    \\\n"
    "  if (b2[0] > b1[2] || b2[2] < b1[0] || b2[1] > b1[3] || b2[3] < b1[1]) {             \\\n"
    "    intersectBbox[0] = 0.;                                                            \\\n"
    "    intersectBbox[1] = 0.;                                                            \\\n"
    "    intersectBbox[2] = 0.;                                                            \\\n"
    "    intersectBbox[3] = 0.;                                                            \\\n"
    "  } else {                                                                            \\\n"
    "    intersectBbox[0] = fmax(b1[0], b2[0]);                                            \\\n"
    "    intersectBbox[1] = fmax(b1[1], b2[1]);                                            \\\n"
    "    intersectBbox[2] = fmin(b1[2], b2[2]);                                            \\\n"
    "    intersectBbox[3] = fmin(b1[3], b2[3]);                                            \\\n"
    "  }                                                                                   \\\n"
    "}                                                                                     \\\n",

    "float BBoxSize(__global float *bbox) {                                                \\\n"
    "  if (bbox[2] < bbox[0] || bbox[3] < bbox[1]) {                                       \\\n"
    "    return 0.;                                                                        \\\n"
    "  } else {                                                                            \\\n"
    "    float width = bbox[2] - bbox[0];                                                  \\\n"
    "    float height = bbox[3] - bbox[1];                                                 \\\n"
    "    return width * height;                                                            \\\n"
    "  }                                                                                   \\\n"
    "}\n",

    "float ComputeIntersectionOverUnion(__global float *d, unsigned int i, unsigned int j) {\\\n"
    "  float box_i_ymin = d[i * 4 + 0];                                                   \\\n"
    "  float box_i_xmin = d[i * 4 + 1];                                                   \\\n"
    "  float box_i_ymax = d[i * 4 + 2];                                                   \\\n"
    "  float box_i_xmax = d[i * 4 + 3];                                                   \\\n"
    "  float box_j_ymin = d[j * 4 + 0];                                                   \\\n"
    "  float box_j_xmin = d[j * 4 + 1];                                                   \\\n"
    "  float box_j_ymax = d[j * 4 + 2];                                                   \\\n"
    "  float box_j_xmax = d[j * 4 + 3];                                                   \\\n"
    "  const float area_i = (box_i_ymax - box_i_ymin) * (box_i_xmax - box_i_xmin);        \\\n"
    "  const float area_j = (box_j_ymax - box_j_ymin) * (box_j_xmax - box_j_xmin);        \\\n"
    "  if (area_i <= 0 || area_j <= 0) {                                                  \\\n"
    "    return 0.0;                                                                      \\\n"
    "  }                                                                                  \\\n"
    "  const float intersection_ymin = max(box_i_ymin, box_j_ymin);                       \\\n"
    "  const float intersection_xmin = max(box_i_xmin, box_j_xmin);                       \\\n"
    "  const float intersection_ymax = min(box_i_ymax, box_j_ymax);                       \\\n"
    "  const float intersection_xmax = min(box_i_xmax, box_j_xmax);                       \\\n"
    "  const float intersection_area = max(intersection_ymax - intersection_ymin, 0.0) *  \\\n"
    "                                  max(intersection_xmax - intersection_xmin, 0.0);   \\\n"
    "  return intersection_area / (area_i + area_j - intersection_area);                  \\\n"
    "}\n",

    "float ComputeHalfIntersectionOverUnion(__global half *d, unsigned int i, unsigned int j) {\\\n"
    "  float box_i_ymin = d[i * 4 + 0];                                                   \\\n"
    "  float box_i_xmin = d[i * 4 + 1];                                                   \\\n"
    "  float box_i_ymax = d[i * 4 + 2];                                                   \\\n"
    "  float box_i_xmax = d[i * 4 + 3];                                                   \\\n"
    "  float box_j_ymin = d[j * 4 + 0];                                                   \\\n"
    "  float box_j_xmin = d[j * 4 + 1];                                                   \\\n"
    "  float box_j_ymax = d[j * 4 + 2];                                                   \\\n"
    "  float box_j_xmax = d[j * 4 + 3];                                                   \\\n"
    "  const float area_i = (box_i_ymax - box_i_ymin) * (box_i_xmax - box_i_xmin);        \\\n"
    "  const float area_j = (box_j_ymax - box_j_ymin) * (box_j_xmax - box_j_xmin);        \\\n"
    "  if (area_i <= 0 || area_j <= 0) {                                                  \\\n"
    "    return 0.0;                                                                      \\\n"
    "  }                                                                                  \\\n"
    "  const float intersection_ymin = max(box_i_ymin, box_j_ymin);                       \\\n"
    "  const float intersection_xmin = max(box_i_xmin, box_j_xmin);                       \\\n"
    "  const float intersection_ymax = min(box_i_ymax, box_j_ymax);                       \\\n"
    "  const float intersection_xmax = min(box_i_xmax, box_j_xmax);                       \\\n"
    "  const float intersection_area = max(intersection_ymax - intersection_ymin, 0.0) *  \\\n"
    "                                  max(intersection_xmax - intersection_xmin, 0.0);   \\\n"
    "  return intersection_area / (area_i + area_j - intersection_area);                  \\\n"
    "}\n",

    "const sampler_t smp_none = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;\n"};

#define APPEND_PRIVATE_KERNEL_HEADERS(HEADER_STR, PRIVATE_HEADER_VEC) \
    do {                                                              \
        for (PrivateKernelHeader pk : PRIVATE_HEADER_VEC)             \
            if (pk >= 0 && pk < NUM_OF_PRIVATE_KERNEL_HEADER)         \
                HEADER_STR += private_kernel_header[pk];              \
    } while (0)

class Kernels {
   public:
    static std::map<std::string, std::vector<PrivateKernelHeader>> &KernelHeaderMap() {
        static std::map<std::string, std::vector<PrivateKernelHeader>> header_map;
        return header_map;
    }
    static void FillKernelHeaderMap(const std::string &kernel_name,
                                    const std::vector<PrivateKernelHeader> &kernel_headers) {
        KernelHeaderMap().emplace(kernel_name, kernel_headers);
    }

    static size_t &KernelLengthInMap() {
        static size_t kernel_len = 0;
        return kernel_len;
    }
    static std::map<std::string, std::string> &KernelMap() {
        static std::map<std::string, std::string> kernel_map;
        if (!kernel_map.empty()) {
            return kernel_map;
        }
        // kernels for data type conversion
        constexpr const char *datatype_convert_kernel_names[] = {"half2float_FP16",
                                                                 "float2half_FP16",
                                                                 "float2half_FP32",
                                                                 "float2int_FP32",
                                                                 "int2float_FP32",
                                                                 "quant8_to_int32_INT8",
                                                                 "signed_quant8_to_int32_INT8"};
        constexpr const char *A_types[] = {"half", "float", "float", "float", "int", "uchar", "char"};
        constexpr const char *B_types[] = {"float", "half", "half", "int", "float", "int", "int"};
        for (size_t i = 0; i < sizeof(datatype_convert_kernel_names) / sizeof(char *); ++i) {
            std::string kernels;
            std::string kernel_name = datatype_convert_kernel_names[i];

            kernels += "__kernel void " + kernel_name + "(__global const ";
            kernels += A_types[i];
            kernels += " *A, __global ";
            kernels += B_types[i];
            kernels +=
                " *B) { "
                "B[get_global_id(0)] = (";
            kernels += B_types[i];
            kernels += ")A[get_global_id(0)]; }";
            kernel_map.emplace(kernel_name, kernels);
            KernelLengthInMap() += kernels.length();
        }

        constexpr const char *kernel_precisions[] = {"FP16", "FP32", "INT8"};
        constexpr const char *kernel_types[] = {"half", "float", "uchar"};

        // broadcast kernels
        // [1] broadcast
        for (size_t i = 0; i < sizeof(kernel_precisions) / sizeof(char *); ++i) {
            std::string kernels;
            std::string kernel_name = "broadcast_";
            kernel_name += kernel_precisions[i];

            kernels += "__kernel void " + kernel_name + "(__global const ";
            kernels += kernel_types[i];
            kernels += " *indata, __global ";
            kernels += kernel_types[i];
            kernels +=
                " *outdata, "
                "int out_w, int n, int c, int h, int w) { "
                "int out_n_index = get_global_id(0); "
                "int out_c_index = get_global_id(1); "
                "int out_h_index = get_global_id(2) / out_w; "
                "int out_w_index = get_global_id(2) % out_w; "
                "int outIndex = out_n_index * get_global_size(1) * get_global_size(2) +"
                "            out_c_index * get_global_size(2) + get_global_id(2); "
                "if (n == 1) { "
                "    out_n_index = 0; "
                " }"
                "if (c == 1) { "
                "    out_c_index = 0; "
                " }"
                "if (h == 1) { "
                "    out_h_index = 0; "
                " }"
                "if (w == 1) { "
                "    out_w_index = 0; "
                " }"
                "int inIndex ="
                "    out_n_index * c * h * w + out_c_index * h * w + out_h_index * w + out_w_index; "
                "outdata[outIndex] = indata[inIndex]; }";
            kernel_map.emplace(kernel_name, kernels);
            KernelLengthInMap() += kernels.length();
        }
        // [2] ndbroadcast
        for (size_t i = 0; i < sizeof(kernel_precisions) / sizeof(char *); ++i) {
            std::string kernels;
            std::string kernel_name = "ndbroadcast_";
            kernel_name += kernel_precisions[i];

            kernels += "__kernel void " + kernel_name + "(__global const ";
            kernels += kernel_types[i];
            kernels += " *indata, __global ";
            kernels += kernel_types[i];
            kernels +=
                " *outdata, "
                "__global const unsigned int *indims, "
                "__global const unsigned int *outdims, "
                "unsigned int num_of_dims, "
                "unsigned int insize) { "
                "unsigned int inIndex = 0; "
                "unsigned int outInnerSize = get_global_size(0); "
                "unsigned int residualSize = get_global_id(0); "
                "unsigned int inInnerSize = insize; "
                "for (unsigned int idx = 0; idx < num_of_dims; idx++) { "
                "    outInnerSize /= outdims[idx]; "
                "    inInnerSize /= indims[idx]; "
                "    unsigned int curDim = 0; "
                "    if (indims[idx] != 1)"
                "        curDim = residualSize / outInnerSize; "
                "    inIndex += curDim * inInnerSize; "
                "    residualSize %= outInnerSize; "
                " }"
                "outdata[get_global_id(0)] = indata[inIndex]; }";
            kernel_map.emplace(kernel_name, kernels);
            KernelLengthInMap() += kernels.length();
        }

        // copybuffer kernels
        for (size_t i = 0; i < sizeof(kernel_precisions) / sizeof(char *); ++i) {
            std::string kernels;
            std::string kernel_name = "copybuffer_";
            kernel_name += kernel_precisions[i];

            kernels += "__kernel void " + kernel_name + "(__global const ";
            kernels += kernel_types[i];
            kernels += " *input, __global ";
            kernels += kernel_types[i];
            kernels +=
                " *output, "
                "unsigned int src_unit_count, "
                "unsigned int dst_align_count) { "
                "int idx = get_global_id(0); "
                "int src_off = idx * src_unit_count; "
                "int des_off = idx * dst_align_count; "
                "for (int i = 0; i < src_unit_count; i++) { "
                "    output[i + des_off] = input[i + src_off]; "
                " }"
                " }";
            kernel_map.emplace(kernel_name, kernels);
            KernelLengthInMap() += kernels.length();
        }

        constexpr const char *nchw_nhwc_precisions[] = {
            "FP16", "FP32", "int8_INT8", "uint8_INT8", "int16_FP32", "uint16_FP32", "fp32_FP16", "fp16_FP32"};
        constexpr const char *nchw_nhwc_in_types[] = {
            "half", "float", "char", "uchar", "short", "ushort", "float", "half"};
        constexpr const char *nchw_nhwc_out_types[] = {
            "half", "float", "char", "uchar", "short", "ushort", "half", "float"};
        // nchw2nhwc kernels
        for (size_t i = 0; i < sizeof(nchw_nhwc_precisions) / sizeof(char *); ++i) {
            std::string kernels;
            std::string kernel_name = "nchw2nhwc_";
            kernel_name += nchw_nhwc_precisions[i];

            kernels += "__kernel void " + kernel_name + "(__global const ";
            kernels += nchw_nhwc_in_types[i];
            kernels += " *input, __global ";
            kernels += nchw_nhwc_out_types[i];
            kernels +=
                " *output, "
                "int channel, int hw_step) { "
                "int g0 = get_global_id(0); "
                "int g1 = get_global_id(1); "
                "int g2 = get_global_id(2); "
                "if (g2 < hw_step) { "
                "    int input_index = g0 * channel * hw_step + g1 * hw_step + g2; "
                "    int output_index = g0 * channel * hw_step + g2 * channel + g1; "
                "    output[output_index] = ";
            if (strcmp(nchw_nhwc_in_types[i], nchw_nhwc_out_types[i]) != 0) {
                kernels += "(";
                kernels += nchw_nhwc_out_types[i];
                kernels += ")";
            }
            kernels += "input[input_index]; ";
            kernels +=
                " }"
                " }";
            kernel_map.emplace(kernel_name, kernels);
            KernelLengthInMap() += kernels.length();
        }
        // nhwc2nchw kernels
        for (size_t i = 0; i < sizeof(nchw_nhwc_precisions) / sizeof(char *); ++i) {
            std::string kernels;
            std::string kernel_name = "nhwc2nchw_";
            kernel_name += nchw_nhwc_precisions[i];

            kernels += "__kernel void " + kernel_name + "(__global const ";
            kernels += nchw_nhwc_in_types[i];
            kernels += " *input, __global ";
            kernels += nchw_nhwc_out_types[i];
            kernels +=
                " *output, "
                "int channel, int hw_step) { "
                "int g0 = get_global_id(0); "
                "int g1 = get_global_id(1); "
                "int g2 = get_global_id(2); "
                "if (g2 < hw_step) { "
                "    int input_index = g0 * channel * hw_step + g2 * channel + g1; "
                "    int output_index = g0 * channel * hw_step + g1 * hw_step + g2; "
                "    output[output_index] = ";
            if (strcmp(nchw_nhwc_in_types[i], nchw_nhwc_out_types[i]) != 0) {
                kernels += "(";
                kernels += nchw_nhwc_out_types[i];
                kernels += ")";
            }
            kernels += "input[input_index]; ";
            kernels +=
                " }"
                " }";
            kernel_map.emplace(kernel_name, kernels);
            KernelLengthInMap() += kernels.length();
        }

        {
            std::string kernel_name = "zeroBuf_FP16";
            std::string kernels =
                "__kernel void zeroBuf_FP16(__global half * output, unsigned int size) {"
                "int globalID0 = get_global_id(0) * 8;"
                "if (globalID0 < size) {"
                "if (globalID0 + 8 <= size) {"
                "vstore8((half8)(0.0f), 0, output + globalID0);"
                "} else {"
                "int num = size - globalID0;"
                "if (num == 1) {"
                "output[globalID0] = (half)(0.0f);"
                "} else if (num == 2) {"
                "vstore2((half2)(0.0f), 0, output + globalID0);"
                "} else if (num == 3) {"
                "vstore3((half3)(0.0f), 0, output + globalID0);"
                "} else if (num == 4) {"
                "vstore4((half4)(0.0f), 0, output + globalID0);"
                "} else if (num == 5) {"
                "vstore4((half4)(0.0f), 0, output + globalID0);"
                "output[globalID0 + 4] = (half)(0.0f);"
                "} else if (num == 6) {"
                "vstore4((half4)(0.0f), 0, output + globalID0);"
                "vstore2((half2)(0.0f), 0, output + globalID0 + 4);"
                "} else if (num == 7) {"
                "vstore4((half4)(0.0f), 0, output + globalID0);"
                "vstore3((half3)(0.0f), 0, output + globalID0 + 4);"
                "}"
                "}"
                "}"
                "}";
            kernel_map.emplace(kernel_name, kernels);
            KernelLengthInMap() += kernels.length();
        }

        {
            std::string kernel_name = "broadcast_texture2d_FP16";
            FillKernelHeaderMap(kernel_name, {DEFINE_SMP_NONE});
            std::string kernels =
                "__kernel void broadcast_texture2d_FP16("
                "                                   __read_only image2d_t src,"
                "                                   __write_only image2d_t dst,"
                "                                   int batch,"
                "                                   int channel,"
                "                                   int height,"
                "                                   int width,"
                "                                   int depth) {"
                "    int X = get_global_id(0);"
                "    int Y = get_global_id(1);"
                "    int Z = get_global_id(2);"
                "    int X_in;"
                "    int Y_in;"
                "    int Z_in;"
                "    if (X >= width) {"
                "        X_in = width - 1;"
                "    }"
                "    if (Y >= height) {"
                "        Y_in = height - 1;"
                "    }"
                "    if (Z >= depth) {"
                "        Z_in = depth - 1;"
                "    }"
                "    half4 out = read_imageh(src, smp_none, (int2)(X_in, Y_in * depth + Z_in));"
                "    write_imageh(dst, (int2)(X, Y * depth + Z), out);"
                "}";
            kernel_map.emplace(kernel_name, kernels);
            KernelLengthInMap() += kernels.length();
        }

        {
            std::string kernel_name = "half2float_texture2d_FP16";
            FillKernelHeaderMap(kernel_name, {DEFINE_SMP_NONE});
            std::string kernels =
                "__kernel void half2float_texture2d_FP16(__read_only image2d_t A,"
                "                                        __global float *B,"
                "                                        int image_width,"
                "                                        int image_height) {"
                "    int g0 = get_global_id(0);"
                "    int g1 = get_global_id(1);"
                "    if (g0 >= image_width || g1 >= image_height)"
                "        return;"
                "    int index = 4 * (g1 * image_width + g0);"
                "    half4 in = read_imageh(A, smp_none, (int2)(g0, g1));"
                "    vstore4(convert_float4(in), 0, B + index);"
                "}";
            kernel_map.emplace(kernel_name, kernels);
            KernelLengthInMap() += kernels.length();
        }

        {
            std::string kernel_name = "float2half_texture2d_FP16";
            FillKernelHeaderMap(kernel_name, {DEFINE_SMP_NONE});
            std::string kernels =
                "__kernel void float2half_texture2d_FP16("
                "    __read_only image2d_t A, __global half * B, int image_width, int image_height) {"
                "    int g0 = get_global_id(0);"
                "    int g1 = get_global_id(1);"
                "    if (g0 >= image_width || g1 >= image_height)"
                "        return;"
                "    int index = 4 * (g1 * image_width + g0);"
                "    float4 in = read_imagef(A, smp_none, (int2)(g0, g1));"
                "    vstore4(convert_half4(in), 0, B + index);"
                "}";
            kernel_map.emplace(kernel_name, kernels);
            KernelLengthInMap() += kernels.length();
        }

        {
            std::string kernel_name = "nchw2dhwc4_fp322fp16_FP32";
            std::string kernels =
                "__kernel void nchw2dhwc4_fp322fp16_FP32(__global const float *input,"
                "                                   __write_only image2d_t output,"
                "                                   int batch,"
                "                                   int channel,"
                "                                   int height,"
                "                                   int width,"
                "                                   int depth) {"
                "    int g0 = get_global_id(0);"
                "    int g1 = get_global_id(1);"
                "    int g2 = get_global_id(2);"
                "    half4 out;"
                "    int c = g2 * 4;"
                "    if (g0 >= width || g1 >= height || g2 >= depth)"
                "        return;"
                "    int index = g2 * 4 * width * height + g1 * width + g0;"
                "    out.x = (half)input[index];"
                "    out.y = c + 1 < channel ? (half)input[index + height * width] : out.x;"
                "    out.z = c + 2 < channel ? (half)input[index + 2 * height * width] : out.x;"
                "    out.w = c + 3 < channel ? (half)input[index + 3 * height * width] : out.x;"
                "    write_imageh(output, (int2)(g0, g1 * depth + g2), out);"
                "}";
            kernel_map.emplace(kernel_name, kernels);
            KernelLengthInMap() += kernels.length();
        }

        {
            std::string kernel_name = "nchw2dhwc4_fp162fp32_FP16";
            std::string kernels =
                "__kernel void nchw2dhwc4_fp162fp32_FP16(__global const half *input,"
                "                                        __write_only image2d_t output,"
                "                                        int batch,"
                "                                        int channel,"
                "                                        int height,"
                "                                        int width,"
                "                                        int depth) {"
                "    int g0 = get_global_id(0);"
                "    int g1 = get_global_id(1);"
                "    int g2 = get_global_id(2);"
                "    float4 out;"
                "    int c = g2 * 4;"
                "    if (g0 >= width || g1 >= height || g2 >= depth)"
                "    return;"
                "    int index = g2 * 4 * width * height + g1 * width + g0;"
                "    out.x = (float)input[index];"
                "    out.y = c + 1 < channel ? (float)input[index + height * width] : out.x;"
                "    out.z = c + 2 < channel ? (float)input[index + 2 * height * width] : out.x;"
                "    out.w = c + 3 < channel ? (float)input[index + 3 * height * width] : out.x;"
                "    write_imagef(output, (int2)(g0, g1 * depth + g2), out);"
                "}";
            kernel_map.emplace(kernel_name, kernels);
            KernelLengthInMap() += kernels.length();
        }

        {
            std::string kernel_name = "nchw2dhwc4_FP16";
            std::string kernels =
                "__kernel void nchw2dhwc4_FP16(__global const half *input,"
                "                              __write_only image2d_t output,"
                "                              int batch,"
                "                              int channel,"
                "                              int height,"
                "                              int width,"
                "                              int depth) {"
                "    int g0 = get_global_id(0);"
                "    int g1 = get_global_id(1);"
                "    int g2 = get_global_id(2);"
                "    half4 out;"
                "    int c = g2 * 4;"
                "    if (g0 >= width || g1 >= height || g2 >= depth)"
                "        return;"
                "    int index = g2 * 4 * width * height + g1 * width + g0;"
                "    out.x = input[index];"
                "    out.y = c + 1 < channel ? input[index + height * width] : out.x;"
                "    out.z = c + 2 < channel ? input[index + 2 * height * width] : out.x;"
                "    out.w = c + 3 < channel ? input[index + 3 * height * width] : out.x;"
                "    write_imageh(output, (int2)(g0, g1 * depth + g2), out);"
                "}";
            kernel_map.emplace(kernel_name, kernels);
            KernelLengthInMap() += kernels.length();
        }

        {
            std::string kernel_name = "nchw2dhwc4_FP32";
            std::string kernels =
                "__kernel void nchw2dhwc4_FP32(__global const float *input,"
                "                              __write_only image2d_t output,"
                "                              int batch,"
                "                              int channel,"
                "                              int height,"
                "                              int width,"
                "                              int depth) {"
                "    int g0 = get_global_id(0);"
                "    int g1 = get_global_id(1);"
                "    int g2 = get_global_id(2);"
                "    float4 out;"
                "    int c = g2 * 4;"
                "    if (g0 >= width || g1 >= height || g2 >= depth)"
                "        return;"
                "    int index = g2 * 4 * width * height + g1 * width + g0;"
                "    out.x = input[index];"
                "    out.y = c + 1 < channel ? input[index + height * width] : out.x;"
                "    out.z = c + 2 < channel ? input[index + 2 * height * width] : out.x;"
                "    out.w = c + 3 < channel ? input[index + 3 * height * width] : out.x;"
                "    write_imagef(output, (int2)(g0, g1 * depth + g2), out);"
                "}";
            kernel_map.emplace(kernel_name, kernels);
            KernelLengthInMap() += kernels.length();
        }

        {
            std::string kernel_name = "dhwc42nchw_fp162fp32_FP32";
            FillKernelHeaderMap(kernel_name, {DEFINE_SMP_NONE});
            std::string kernels =
                "__kernel void dhwc42nchw_fp162fp32_FP32(__read_only image2d_t input,"
                "                                   __global float *output,"
                "                                   int batch,"
                "                                   int channel,"
                "                                   int height,"
                "                                   int width,"
                "                                   int depth) {"
                "    int g0 = get_global_id(0);"
                "    int g1 = get_global_id(1);"
                "    int g2 = get_global_id(2);"
                "    int c = g2 * 4;"
                "    int index = g2 * 4 * width * height + g1 * width + g0;"
                "    if (g0 >= width || g1 >= height || g2 >= depth)"
                "        return;"
                "    half4 in = read_imageh(input, smp_none, (int2)(g0, g1 * depth + g2));"
                "    output[index] = (float)in.x;"
                "    if (c + 1 < channel)"
                "        output[index + height * width] = (float)in.y;"
                "    if (c + 2 < channel)"
                "        output[index + 2 * height * width] = (float)in.z;"
                "    if (c + 3 < channel)"
                "        output[index + 3 * height * width] = (float)in.w;"
                "}";
            kernel_map.emplace(kernel_name, kernels);
            KernelLengthInMap() += kernels.length();
        }

        {
            std::string kernel_name = "dhwc42nchw_fp322fp16_FP16";
            FillKernelHeaderMap(kernel_name, {DEFINE_SMP_NONE});
            std::string kernels =
                "__kernel void dhwc42nchw_fp322fp16_FP16(__read_only image2d_t input,"
                "                                        __global half *output,"
                "                                        int batch,"
                "                                        int channel,"
                "                                        int height,"
                "                                        int width,"
                "                                        int depth) {"
                "    int g0 = get_global_id(0);"
                "    int g1 = get_global_id(1);"
                "    int g2 = get_global_id(2);"
                "    int c = g2 * 4;"
                "    int index = g2 * 4 * width * height + g1 * width + g0;"
                "    if (g0 >= width || g1 >= height || g2 >= depth)"
                "    return;"
                "    float4 in = read_imagef(input, smp_none, (int2)(g0, g1 * depth + g2));"
                "    output[index] = (half)in.x;"
                "    if (c + 1 < channel)"
                "    output[index + height * width] = (half)in.y;"
                "    if (c + 2 < channel)"
                "    output[index + 2 * height * width] = (half)in.z;"
                "    if (c + 3 < channel)"
                "    output[index + 3 * height * width] = (half)in.w;"
                "}";
            kernel_map.emplace(kernel_name, kernels);
            KernelLengthInMap() += kernels.length();
        }

        {
            std::string kernel_name = "dhwc42nchw_FP32";
            FillKernelHeaderMap(kernel_name, {DEFINE_SMP_NONE});
            std::string kernels =
                "__kernel void dhwc42nchw_FP32(__read_only image2d_t input,"
                "                              __global float *output,"
                "                              int batch,"
                "                              int channel,"
                "                              int height,"
                "                              int width,"
                "                              int depth) {"
                "    int g0 = get_global_id(0);"
                "    int g1 = get_global_id(1);"
                "    int g2 = get_global_id(2);"
                "    int c = g2 * 4;"
                "    int index = g2 * 4 * width * height + g1 * width + g0;"
                "    if (g0 >= width || g1 >= height || g2 >= depth)"
                "        return;"
                "    float4 in = read_imagef(input, smp_none, (int2)(g0, g1 * depth + g2));"
                "    output[index] = in.x;"
                "    if (c + 1 < channel)"
                "        output[index + height * width] = in.y;"
                "    if (c + 2 < channel)"
                "        output[index + 2 * height * width] = in.z;"
                "    if (c + 3 < channel)"
                "        output[index + 3 * height * width] = in.w;"
                "}";
            kernel_map.emplace(kernel_name, kernels);
            KernelLengthInMap() += kernels.length();
        }

        {
            std::string kernel_name = "dhwc42nchw_FP16";
            FillKernelHeaderMap(kernel_name, {DEFINE_SMP_NONE});
            std::string kernels =
                "__kernel void dhwc42nchw_FP16(__read_only image2d_t input,"
                "                              __global half * output,"
                "                              int batch,"
                "                              int channel,"
                "                              int height,"
                "                              int width,"
                "                              int depth) {"
                "    int g0 = get_global_id(0);"
                "    int g1 = get_global_id(1);"
                "    int g2 = get_global_id(2);"
                "    int c = g2 * 4;"
                "    int index = g2 * 4 * width * height + g1 * width + g0;"
                "    if (g0 >= width || g1 >= height || g2 >= depth)"
                "        return;"
                "    half4 in = read_imageh(input, smp_none, (int2)(g0, g1 * depth + g2));"
                "    output[index] = in.x;"
                "    if (c + 1 < channel)"
                "        output[index + height * width] = in.y;"
                "    if (c + 2 < channel)"
                "        output[index + 2 * height * width] = in.z;"
                "    if (c + 3 < channel)"
                "        output[index + 3 * height * width] = in.w;"
                "}";
            kernel_map.emplace(kernel_name, kernels);
            KernelLengthInMap() += kernels.length();
        }

        {
            std::string kernel_name = "nhwc2dhwc4_fp322fp16_FP32";
            std::string kernels =
                "__kernel void nhwc2dhwc4_fp322fp16_FP32(__global const float *input,"
                "                                   __write_only image2d_t output,"
                "                                   int batch,"
                "                                   int channel,"
                "                                   int height,"
                "                                   int width,"
                "                                   int depth) {"
                "    int g0 = get_global_id(0);"
                "    int g1 = get_global_id(1);"
                "    int g2 = get_global_id(2);"
                "    half4 out;"
                "    int c = g2 * 4;"
                "    if (g0 >= width || g1 >= height || g2 >= depth)"
                "        return;"
                "    int index = (g1 * width + g0) * channel + c;"
                "    out.x = (half)input[index];"
                "    out.y = c + 1 < channel ? (half)input[index + 1] : out.x;"
                "    out.z = c + 2 < channel ? (half)input[index + 2] : out.x;"
                "    out.w = c + 3 < channel ? (half)input[index + 3] : out.x;"
                "    write_imageh(output, (int2)(g0, g1 * depth + g2), out);"
                "}";
            kernel_map.emplace(kernel_name, kernels);
            KernelLengthInMap() += kernels.length();
        }

        {
            std::string kernel_name = "nhwc2dhwc4_fp162fp32_FP16";
            std::string kernels =
                "__kernel void nhwc2dhwc4_fp162fp32_FP16(__global const half *input,"
                "                                        __write_only image2d_t output,"
                "                                        int batch,"
                "                                        int channel,"
                "                                        int height,"
                "                                        int width,"
                "                                        int depth) {"
                "                                        int g0 = get_global_id(0);"
                "                                        int g1 = get_global_id(1);"
                "                                        int g2 = get_global_id(2);"
                "    float4 out;"
                "    int c = g2 * 4;"
                "    if (g0 >= width || g1 >= height || g2 >= depth)"
                "    return;"
                "    int index = (g1 * width + g0) * channel + c;"
                "    out.x = (float)input[index];"
                "    out.y = c + 1 < channel ? (float)input[index + 1] : out.x;"
                "    out.z = c + 2 < channel ? (float)input[index + 2] : out.x;"
                "    out.w = c + 3 < channel ? (float)input[index + 3] : out.x;"
                "    write_imagef(output, (int2)(g0, g1 * depth + g2), out);"
                "}";
            kernel_map.emplace(kernel_name, kernels);
            KernelLengthInMap() += kernels.length();
        }

        {
            std::string kernel_name = "nhwc2dhwc4_FP32";
            std::string kernels =
                "__kernel void nhwc2dhwc4_FP32(__global const float *input,"
                "                              __write_only image2d_t output,"
                "                              int batch,"
                "                              int channel,"
                "                              int height,"
                "                              int width,"
                "                              int depth) {"
                "    int g0 = get_global_id(0);"
                "    int g1 = get_global_id(1);"
                "    int g2 = get_global_id(2);"
                "    float4 out;"
                "    int c = g2 * 4;"
                "    if (g0 >= width || g1 >= height || g2 >= depth)"
                "        return;"
                "    int index = (g1 * width + g0) * channel + c;"
                "    out.x = input[index];"
                "    out.y = c + 1 < channel ? input[index + 1] : out.x;"
                "    out.z = c + 2 < channel ? input[index + 2] : out.x;"
                "    out.w = c + 3 < channel ? input[index + 3] : out.x;"
                "    write_imagef(output, (int2)(g0, g1 * depth + g2), out);"
                "}";
            kernel_map.emplace(kernel_name, kernels);
            KernelLengthInMap() += kernels.length();
        }

        {
            std::string kernel_name = "nhwc2dhwc4_FP16";
            std::string kernels =
                "__kernel void nhwc2dhwc4_FP16(__global const half *input,"
                "                              __write_only image2d_t output,"
                "                              int batch,"
                "                              int channel,"
                "                              int height,"
                "                              int width,"
                "                              int depth) {"
                "    int g0 = get_global_id(0);"
                "    int g1 = get_global_id(1);"
                "    int g2 = get_global_id(2);"
                "    half4 out;"
                "    int c = g2 * 4;"
                "    if (g0 >= width || g1 >= height || g2 >= depth)"
                "        return;"
                "    int index = (g1 * width + g0) * channel + c;"
                "    out.x = input[index];"
                "    out.y = c + 1 < channel ? input[index + 1] : out.x;"
                "    out.z = c + 2 < channel ? input[index + 2] : out.x;"
                "    out.w = c + 3 < channel ? input[index + 3] : out.x;"
                "    write_imageh(output, (int2)(g0, g1 * depth + g2), out);"
                "}";
            kernel_map.emplace(kernel_name, kernels);
            KernelLengthInMap() += kernels.length();
        }

        {
            std::string kernel_name = "dhwc42nhwc_fp162fp32_FP32";
            FillKernelHeaderMap(kernel_name, {DEFINE_SMP_NONE});
            std::string kernels =
                "__kernel void dhwc42nhwc_fp162fp32_FP32(__read_only image2d_t input,"
                "                                   __global float *output,"
                "                                   int batch,"
                "                                   int channel,"
                "                                   int height,"
                "                                   int width,"
                "                                   int depth) {"
                "    int g0 = get_global_id(0);"
                "    int g1 = get_global_id(1);"
                "    int g2 = get_global_id(2);"
                "    int c = g2 * 4;"
                "    if (g0 >= width || g1 >= height || g2 >= depth)"
                "        return;"
                "    int index = (g1 * width + g0) * channel + c;"
                "    half4 in = read_imageh(input, smp_none, (int2)(g0, g1 * depth + g2));"
                "    output[index] = (float)in.x;"
                "    if (c + 1 < channel)"
                "        output[index + 1] = (float)in.y;"
                "    if (c + 2 < channel)"
                "        output[index + 2] = (float)in.z;"
                "    if (c + 3 < channel)"
                "        output[index + 3] = (float)in.w;"
                "}";
            kernel_map.emplace(kernel_name, kernels);
            KernelLengthInMap() += kernels.length();
        }

        {
            std::string kernel_name = "dhwc42nhwc_fp322fp16_FP16";
            FillKernelHeaderMap(kernel_name, {DEFINE_SMP_NONE});
            std::string kernels =
                "__kernel void dhwc42nhwc_fp322fp16_FP16(__read_only image2d_t input,"
                "                                        __global half *output,"
                "                                        int batch,"
                "                                        int channel,"
                "                                        int height,"
                "                                        int width,"
                "                                        int depth) {"
                "                                        int g0 = get_global_id(0);"
                "                                        int g1 = get_global_id(1);"
                "                                        int g2 = get_global_id(2);"
                "    int c = g2 * 4;"
                "    if (g0 >= width || g1 >= height || g2 >= depth)"
                "    return;"
                "    int index = (g1 * width + g0) * channel + c;"
                "    float4 in = read_imagef(input, smp_none, (int2)(g0, g1 * depth + g2));"
                "    output[index] = (half)in.x;"
                "    if (c + 1 < channel)"
                "    output[index + 1] = (half)in.y;"
                "    if (c + 2 < channel)"
                "    output[index + 2] = (half)in.z;"
                "    if (c + 3 < channel)"
                "    output[index + 3] = (half)in.w;"
                "}";
            kernel_map.emplace(kernel_name, kernels);
            KernelLengthInMap() += kernels.length();
        }

        {
            std::string kernel_name = "dhwc42nhwc_FP32";
            FillKernelHeaderMap(kernel_name, {DEFINE_SMP_NONE});
            std::string kernels =
                "__kernel void dhwc42nhwc_FP32(__read_only image2d_t input,"
                "                              __global float *output,"
                "                              int batch,"
                "                              int channel,"
                "                              int height,"
                "                              int width,"
                "                              int depth) {"
                "    int g0 = get_global_id(0);"
                "    int g1 = get_global_id(1);"
                "    int g2 = get_global_id(2);"
                "    int c = g2 * 4;"
                "    if (g0 >= width || g1 >= height || g2 >= depth)"
                "        return;"
                "    int index = (g1 * width + g0) * channel + c;"
                "    float4 in = read_imagef(input, smp_none, (int2)(g0, g1 * depth + g2));"
                "    output[index] = in.x;"
                "    if (c + 1 < channel)"
                "        output[index + 1] = in.y;"
                "    if (c + 2 < channel)"
                "        output[index + 2] = in.z;"
                "    if (c + 3 < channel)"
                "        output[index + 3] = in.w;"
                "}";
            kernel_map.emplace(kernel_name, kernels);
            KernelLengthInMap() += kernels.length();
        }

        {
            std::string kernel_name = "dhwc42nhwc_FP16";
            FillKernelHeaderMap(kernel_name, {DEFINE_SMP_NONE});
            std::string kernels =
                "__kernel void dhwc42nhwc_FP16(__read_only image2d_t input,"
                "                              __global half * output,"
                "                              int batch,"
                "                              int channel,"
                "                              int height,"
                "                              int width,"
                "                              int depth) {"
                "    int g0 = get_global_id(0);"
                "    int g1 = get_global_id(1);"
                "    int g2 = get_global_id(2);"
                "    int c = g2 * 4;"
                "    if (g0 >= width || g1 >= height || g2 >= depth)"
                "        return;"
                "    int index = (g1 * width + g0) * channel + c;"
                "    half4 in = read_imageh(input, smp_none, (int2)(g0, g1 * depth + g2));"
                "    output[index] = in.x;"
                "    if (c + 1 < channel)"
                "        output[index + 1] = in.y;"
                "    if (c + 2 < channel)"
                "        output[index + 2] = in.z;"
                "    if (c + 3 < channel)"
                "        output[index + 3] = in.w;"
                "}";
            kernel_map.emplace(kernel_name, kernels);
            KernelLengthInMap() += kernels.length();
        }

        {
            std::string kernel_name = "zeroTexture2D_FP32";
            std::string kernels =
                "__kernel void zeroTexture2D_FP32(__write_only image2d_t buf, int image_width, int image_height) {"
                "    int g0 = get_global_id(0);"
                "    int g1 = get_global_id(1);"
                "    if (g0 >= image_width || g1 >= image_height)"
                "        return;"
                "    float4 zero = (float4)(0.0f, 0.0f, 0.0f, 0.0f);"
                "    write_imagef(buf, (int2)(g0, g1), zero);"
                "}";
            kernel_map.emplace(kernel_name, kernels);
            KernelLengthInMap() += kernels.length();
        }

        {
            std::string kernel_name = "zeroTexture2D_FP16";
            std::string kernels =
                "__kernel void zeroTexture2D_FP16(__write_only image2d_t buf, int image_width, int image_height) {"
                "    int g0 = get_global_id(0);"
                "    int g1 = get_global_id(1);"
                "    if (g0 >= image_width || g1 >= image_height)"
                "        return;"
                "    half4 zero = (half4)(0.0f, 0.0f, 0.0f, 0.0f);"
                "    write_imageh(buf, (int2)(g0, g1), zero);"
                "}";
            kernel_map.emplace(kernel_name, kernels);
            KernelLengthInMap() += kernels.length();
        }

        {
            std::string kernel_name = "copy_texture2d_FP16";
            FillKernelHeaderMap(kernel_name, {DEFINE_SMP_NONE});
            std::string kernels =
                "__kernel void copy_texture2d_FP16(__read_only image2d_t src,"
                "    __write_only image2d_t dst,"
                "    int src_x,"
                "    int src_y,"
                "    int src_z,"
                "    int src_w) {"
                "    int X = get_global_id(0);"
                "    int Y = get_global_id(1);"
                "    int Z = get_global_id(2);"
                "    if (X >= src_x || Y >= src_y || Z >= src_z) {"
                "    return;"
                "    }"
                "    half4 in = read_imageh(src, smp_none, (int2)(X, Y * src_z + Z));"
                "    write_imageh(dst, (int2)(X, Y * src_z + Z), in);"
                "}";
            kernel_map.emplace(kernel_name, kernels);
            KernelLengthInMap() += kernels.length();
        }
        return kernel_map;
    }

    static void FillKernelMap(const std::string &kernel_name, const std::string &kernel) {
        KernelMap().emplace(kernel_name, kernel);
        KernelLengthInMap() += kernel.length();
    }
};

// Registerer that adds a kernel to the global kernel string
class KernelRegisterer {
   public:
    KernelRegisterer(const char *kernel_name, const char *kernel_body) {
        std::string kernel = "__kernel void ";
        kernel += kernel_name;
        kernel += kernel_body;
        Kernels::FillKernelMap(kernel_name, kernel);
    }

    KernelRegisterer(const char *kernel_name, const std::vector<PrivateKernelHeader> &kernel_headers) {
        Kernels::FillKernelHeaderMap(kernel_name, kernel_headers);
    }
};

#define ADD_SINGLE_KERNEL(NAME, BODY) static KernelRegisterer NAME(__SHARP_X(NAME), __SHARP_X(BODY));
#define ADD_SINGLE_KERNEL_STR(NAME, BODY_STR) static KernelRegisterer NAME(__SHARP_X(NAME), BODY_STR);
#define ADD_KERNEL_HEADER(NAME, ...) static KernelRegisterer _X_HEADER(NAME)(__SHARP_X(NAME), __VA_ARGS__);
}  // namespace gpu
}  // namespace ud
}  // namespace enn

#endif  // USERDRIVER_GPU_CL_OPERATORS_CL_KERNELS_HPP_
