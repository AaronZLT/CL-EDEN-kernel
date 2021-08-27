#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"
namespace enn{
namespace ud {
namespace gpu {
#define GEMMBLOCK2X8_FEED_OUTPUT_VSTORE2(topChannel, out8_0, out8_1, output, outputIndex, topHW) \
    if (get_global_id(1) * 8 + 7 < topChannel) { \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s0, out8_1.s0)), 0, output + outputIndex); \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s1, out8_1.s1)), 0, output + outputIndex + topHW); \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s2, out8_1.s2)), 0, output + outputIndex + 2 * topHW); \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s3, out8_1.s3)), 0, output + outputIndex + 3 * topHW); \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s4, out8_1.s4)), 0, output + outputIndex + 4 * topHW); \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s5, out8_1.s5)), 0, output + outputIndex + 5 * topHW); \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s6, out8_1.s6)), 0, output + outputIndex + 6 * topHW); \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s7, out8_1.s7)), 0, output + outputIndex + 7 * topHW); \
    } else if (get_global_id(1) * 8 + 6 < topChannel) { \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s0, out8_1.s0)), 0, output + outputIndex); \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s1, out8_1.s1)), 0, output + outputIndex + topHW); \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s2, out8_1.s2)), 0, output + outputIndex + 2 * topHW); \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s3, out8_1.s3)), 0, output + outputIndex + 3 * topHW); \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s4, out8_1.s4)), 0, output + outputIndex + 4 * topHW); \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s5, out8_1.s5)), 0, output + outputIndex + 5 * topHW); \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s6, out8_1.s6)), 0, output + outputIndex + 6 * topHW); \
    } else if (get_global_id(1) * 8 + 5 < topChannel) { \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s0, out8_1.s0)), 0, output + outputIndex); \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s1, out8_1.s1)), 0, output + outputIndex + topHW); \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s2, out8_1.s2)), 0, output + outputIndex + 2 * topHW); \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s3, out8_1.s3)), 0, output + outputIndex + 3 * topHW); \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s4, out8_1.s4)), 0, output + outputIndex + 4 * topHW); \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s5, out8_1.s5)), 0, output + outputIndex + 5 * topHW); \
    } else if (get_global_id(1) * 8 + 4 < topChannel) { \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s0, out8_1.s0)), 0, output + outputIndex); \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s1, out8_1.s1)), 0, output + outputIndex + topHW); \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s2, out8_1.s2)), 0, output + outputIndex + 2 * topHW); \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s3, out8_1.s3)), 0, output + outputIndex + 3 * topHW); \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s4, out8_1.s4)), 0, output + outputIndex + 4 * topHW); \
    } else if (get_global_id(1) * 8 + 3 < topChannel) { \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s0, out8_1.s0)), 0, output + outputIndex); \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s1, out8_1.s1)), 0, output + outputIndex + topHW); \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s2, out8_1.s2)), 0, output + outputIndex + 2 * topHW); \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s3, out8_1.s3)), 0, output + outputIndex + 3 * topHW); \
    } else if (get_global_id(1) * 8 + 2 < topChannel) { \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s0, out8_1.s0)), 0, output + outputIndex); \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s1, out8_1.s1)), 0, output + outputIndex + topHW); \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s2, out8_1.s2)), 0, output + outputIndex + 2 * topHW); \
    } else if (get_global_id(1) * 8 + 1 < topChannel) { \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s0, out8_1.s0)), 0, output + outputIndex); \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s1, out8_1.s1)), 0, output + outputIndex + topHW); \
    } else if (get_global_id(1) * 8 < topChannel) { \
        vstore2(ACT_VEC_F(DATA_T2, (DATA_T2)(out8_0.s0, out8_1.s0)), 0, output + outputIndex); \
    }

#define GEMMBLOCK2X8_FEED_OUTPUT(topChannel, out8_0, output, outputIndex, topHW) \
    if (get_global_id(1) * 8 + 7 < topChannel) { \
        output[outputIndex] = ACT_VEC_F(DATA_T, out8_0.s0); \
        output[outputIndex + topHW] = ACT_VEC_F(DATA_T, out8_0.s1); \
        output[outputIndex + 2 * topHW] = ACT_VEC_F(DATA_T, out8_0.s2); \
        output[outputIndex + 3 * topHW] = ACT_VEC_F(DATA_T, out8_0.s3); \
        output[outputIndex + 4 * topHW] = ACT_VEC_F(DATA_T, out8_0.s4); \
        output[outputIndex + 5 * topHW] = ACT_VEC_F(DATA_T, out8_0.s5); \
        output[outputIndex + 6 * topHW] = ACT_VEC_F(DATA_T, out8_0.s6); \
        output[outputIndex + 7 * topHW] = ACT_VEC_F(DATA_T, out8_0.s7); \
    } else if (get_global_id(1) * 8 + 6 < topChannel) { \
        output[outputIndex] = ACT_VEC_F(DATA_T, out8_0.s0); \
        output[outputIndex + topHW] = ACT_VEC_F(DATA_T, out8_0.s1); \
        output[outputIndex + 2 * topHW] = ACT_VEC_F(DATA_T, out8_0.s2); \
        output[outputIndex + 3 * topHW] = ACT_VEC_F(DATA_T, out8_0.s3); \
        output[outputIndex + 4 * topHW] = ACT_VEC_F(DATA_T, out8_0.s4); \
        output[outputIndex + 5 * topHW] = ACT_VEC_F(DATA_T, out8_0.s5); \
        output[outputIndex + 6 * topHW] = ACT_VEC_F(DATA_T, out8_0.s6); \
    } else if (get_global_id(1) * 8 + 5 < topChannel) { \
        output[outputIndex] = ACT_VEC_F(DATA_T, out8_0.s0); \
        output[outputIndex + topHW] = ACT_VEC_F(DATA_T, out8_0.s1); \
        output[outputIndex + 2 * topHW] = ACT_VEC_F(DATA_T, out8_0.s2); \
        output[outputIndex + 3 * topHW] = ACT_VEC_F(DATA_T, out8_0.s3); \
        output[outputIndex + 4 * topHW] = ACT_VEC_F(DATA_T, out8_0.s4); \
        output[outputIndex + 5 * topHW] = ACT_VEC_F(DATA_T, out8_0.s5); \
    } else if (get_global_id(1) * 8 + 4 < topChannel) { \
        output[outputIndex] = ACT_VEC_F(DATA_T, out8_0.s0); \
        output[outputIndex + topHW] = ACT_VEC_F(DATA_T, out8_0.s1); \
        output[outputIndex + 2 * topHW] = ACT_VEC_F(DATA_T, out8_0.s2); \
        output[outputIndex + 3 * topHW] = ACT_VEC_F(DATA_T, out8_0.s3); \
        output[outputIndex + 4 * topHW] = ACT_VEC_F(DATA_T, out8_0.s4); \
    } else if (get_global_id(1) * 8 + 3 < topChannel) { \
        output[outputIndex] = ACT_VEC_F(DATA_T, out8_0.s0); \
        output[outputIndex + topHW] = ACT_VEC_F(DATA_T, out8_0.s1); \
        output[outputIndex + 2 * topHW] = ACT_VEC_F(DATA_T, out8_0.s2); \
        output[outputIndex + 3 * topHW] = ACT_VEC_F(DATA_T, out8_0.s3); \
    } else if (get_global_id(1) * 8 + 2 < topChannel) { \
        output[outputIndex] = ACT_VEC_F(DATA_T, out8_0.s0); \
        output[outputIndex + topHW] = ACT_VEC_F(DATA_T, out8_0.s1); \
        output[outputIndex + 2 * topHW] = ACT_VEC_F(DATA_T, out8_0.s2); \
    } else if (get_global_id(1) * 8 + 1 < topChannel) { \
        output[outputIndex] = ACT_VEC_F(DATA_T, out8_0.s0); \
        output[outputIndex + topHW] = ACT_VEC_F(DATA_T, out8_0.s1); \
    } else if (get_global_id(1) * 8 < topChannel) { \
        output[outputIndex] = ACT_VEC_F(DATA_T, out8_0.s0); \
    }

#define GEMMBLOCK2X8(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep) \
    if (get_global_id(2) * 2 + 1 < topHW) { \
        DATA_T8 out8_0 = vload8(0, bias + get_global_id(1) * 8); \
        DATA_T8 out8_1 = out8_0; \
        { \
            uint2 src_addr = (uint2)(get_global_id(0) * bottomStep + \
                                            (get_global_id(2) * 2 / 24) * 24 * alignedCinKK + \
                                            (get_global_id(2) * 2 % 24 / 2) * 8, \
                                        get_global_id(1) * 8 * alignedCinKK);  \
            uint end_alignedCinKK = src_addr.s0 + 24 * alignedCinKK; \
            for (; src_addr.s0 < end_alignedCinKK; src_addr += (uint2)(192, 64)) { \
                DATA_T8 input8; \
                DATA_T8 input8_1; \
                DATA_T8 weight8; \
                DATA_T8 weight8_1; \
                DATA_T8 weight8_2; \
                DATA_T8 weight8_3; \
                DATA_T8 weight8_4; \
                DATA_T8 weight8_5; \
                DATA_T8 weight8_6; \
                DATA_T8 weight8_7; \
                input8 = vload8(0, input + src_addr.s0); \
                input8_1 = vload8(0, input + src_addr.s0 + 96); \
                weight8 = vload8(0, weight + src_addr.s1); \
                out8_0 += input8.s0 * weight8; \
                out8_1 += input8_1.s0 * weight8; \
                weight8_1 = vload8(0, weight + src_addr.s1 + 8); \
                out8_0 += input8.s1 * weight8_1; \
                out8_1 += input8_1.s1 * weight8_1; \
                weight8_2 = vload8(0, weight + src_addr.s1 + 16); \
                out8_0 += input8.s2 * weight8_2; \
                out8_1 += input8_1.s2 * weight8_2; \
                weight8_3 = vload8(0, weight + src_addr.s1 + 24); \
                out8_0 += input8.s3 * weight8_3; \
                out8_1 += input8_1.s3 * weight8_3; \
                weight8_4 = vload8(0, weight + src_addr.s1 + 32); \
                out8_0 += input8.s4 * weight8_4; \
                out8_1 += input8_1.s4 * weight8_4; \
                weight8_5 = vload8(0, weight + src_addr.s1 + 40); \
                out8_0 += input8.s5 * weight8_5; \
                out8_1 += input8_1.s5 * weight8_5; \
                weight8_6 = vload8(0, weight + src_addr.s1 + 48); \
                out8_0 += input8.s6 * weight8_6; \
                out8_1 += input8_1.s6 * weight8_6; \
                weight8_7 = vload8(0, weight + src_addr.s1 + 56); \
                out8_0 += input8.s7 * weight8_7; \
                out8_1 += input8_1.s7 * weight8_7; \
            } \
        } \
        int outputIndex = get_global_id(0) * topChannel * topHW + get_global_id(1) * 8 * topHW + \
                            get_global_id(2) * 2; \
        GEMMBLOCK2X8_FEED_OUTPUT_VSTORE2(topChannel, out8_0, out8_1, output, outputIndex, topHW) \
    } else if (get_global_id(2) * 2 < topHW) { \
        DATA_T8 out8_0 = vload8(0, bias + get_global_id(1) * 8); \
        { \
            uint2 src_addr = (uint2)(get_global_id(0) * bottomStep + \
                                            (get_global_id(2) * 2 / 24) * 24 * alignedCinKK + \
                                            (get_global_id(2) * 2 % 24 / 2) * 8, \
                                        get_global_id(1) * 8 * alignedCinKK);  \
            uint end_alignedCinKK = src_addr.s0 + 24 * alignedCinKK; \
            for (; src_addr.s0 < end_alignedCinKK; src_addr += (uint2)(192, 64)) { \
                DATA_T8 input8; \
                DATA_T8 weight8; \
                DATA_T8 weight8_1; \
                DATA_T8 weight8_2; \
                DATA_T8 weight8_3; \
                DATA_T8 weight8_4; \
                DATA_T8 weight8_5; \
                DATA_T8 weight8_6; \
                DATA_T8 weight8_7; \
                input8 = vload8(0, input + src_addr.s0); \
                weight8 = vload8(0, weight + src_addr.s1); \
                weight8_1 = vload8(0, weight + src_addr.s1 + 8); \
                weight8_2 = vload8(0, weight + src_addr.s1 + 16); \
                weight8_3 = vload8(0, weight + src_addr.s1 + 24); \
                weight8_4 = vload8(0, weight + src_addr.s1 + 32); \
                weight8_5 = vload8(0, weight + src_addr.s1 + 40); \
                weight8_6 = vload8(0, weight + src_addr.s1 + 48); \
                weight8_7 = vload8(0, weight + src_addr.s1 + 56); \
                out8_0 = mad(weight8, input8.s0, out8_0); \
                out8_0 = mad(weight8_1, input8.s1, out8_0); \
                out8_0 = mad(weight8_2, input8.s2, out8_0); \
                out8_0 = mad(weight8_3, input8.s3, out8_0); \
                out8_0 = mad(weight8_4, input8.s4, out8_0); \
                out8_0 = mad(weight8_5, input8.s5, out8_0); \
                out8_0 = mad(weight8_6, input8.s6, out8_0); \
                out8_0 = mad(weight8_7, input8.s7, out8_0); \
            } \
        } \
        int outputIndex = get_global_id(0) * topChannel * topHW + get_global_id(1) * 8 * topHW + \
                            get_global_id(2) * 2; \
        GEMMBLOCK2X8_FEED_OUTPUT(topChannel, out8_0, output, outputIndex, topHW) \
    }

#define GEMMBLOCK2X8_SCALAR_FP16(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep) \
    if (get_global_id(2) * 2 + 1 < topHW) { \
        DATA_T8 out8_0 = vload8(0, bias + get_global_id(1) * 8); \
        DATA_T8 out8_1 = out8_0; \
        { \
            uint2 src_addr = (uint2)(get_global_id(0) * bottomStep + \
                                            (get_global_id(2) * 2 / 24) * 24 * alignedCinKK + \
                                            (get_global_id(2) * 2 % 24 / 2) * 8, \
                                        get_global_id(1) * 8 * alignedCinKK);  \
            uint end_alignedCinKK = src_addr.s0 + 24 * alignedCinKK; \
            for (; src_addr.s0 < end_alignedCinKK; src_addr += (uint2)(192, 64)) { \
                DATA_T8 input8; \
                DATA_T8 input8_1; \
                DATA_T8 weight8; \
                DATA_T8 weight8_1; \
                DATA_T8 weight8_2; \
                DATA_T8 weight8_3; \
                DATA_T8 weight8_4; \
                DATA_T8 weight8_5; \
                DATA_T8 weight8_6; \
                DATA_T8 weight8_7; \
                input8 = vload8(0, input + src_addr.s0); \
                input8_1 = vload8(0, input + src_addr.s0 + 96); \
                weight8 = vload8(0, weight + src_addr.s1); \
                out8_0.s01 += input8.s0 * weight8.s01; \
                out8_0.s23 += input8.s0 * weight8.s23; \
                out8_0.s45 += input8.s0 * weight8.s45; \
                out8_0.s67 += input8.s0 * weight8.s67; \
                out8_1.s01 += input8_1.s0 * weight8.s01; \
                out8_1.s23 += input8_1.s0 * weight8.s23; \
                out8_1.s45 += input8_1.s0 * weight8.s45; \
                out8_1.s67 += input8_1.s0 * weight8.s67; \
                weight8_1 = vload8(0, weight + src_addr.s1 + 8); \
                out8_0.s01 += input8.s1 * weight8_1.s01; \
                out8_0.s23 += input8.s1 * weight8_1.s23; \
                out8_0.s45 += input8.s1 * weight8_1.s45; \
                out8_0.s67 += input8.s1 * weight8_1.s67; \
                out8_1.s01 += input8_1.s1 * weight8_1.s01; \
                out8_1.s23 += input8_1.s1 * weight8_1.s23; \
                out8_1.s45 += input8_1.s1 * weight8_1.s45; \
                out8_1.s67 += input8_1.s1 * weight8_1.s67; \
                weight8_2 = vload8(0, weight + src_addr.s1 + 16); \
                out8_0.s01 += input8.s2 * weight8_2.s01; \
                out8_0.s23 += input8.s2 * weight8_2.s23; \
                out8_0.s45 += input8.s2 * weight8_2.s45; \
                out8_0.s67 += input8.s2 * weight8_2.s67; \
                out8_1.s01 += input8_1.s2 * weight8_2.s01; \
                out8_1.s23 += input8_1.s2 * weight8_2.s23; \
                out8_1.s45 += input8_1.s2 * weight8_2.s45; \
                out8_1.s67 += input8_1.s2 * weight8_2.s67; \
                weight8_3 = vload8(0, weight + src_addr.s1 + 24); \
                out8_0.s01 += input8.s3 * weight8_3.s01; \
                out8_0.s23 += input8.s3 * weight8_3.s23; \
                out8_0.s45 += input8.s3 * weight8_3.s45; \
                out8_0.s67 += input8.s3 * weight8_3.s67; \
                out8_1.s01 += input8_1.s3 * weight8_3.s01; \
                out8_1.s23 += input8_1.s3 * weight8_3.s23; \
                out8_1.s45 += input8_1.s3 * weight8_3.s45; \
                out8_1.s67 += input8_1.s3 * weight8_3.s67; \
                weight8_4 = vload8(0, weight + src_addr.s1 + 32); \
                out8_0.s01 += input8.s4 * weight8_4.s01; \
                out8_0.s23 += input8.s4 * weight8_4.s23; \
                out8_0.s45 += input8.s4 * weight8_4.s45; \
                out8_0.s67 += input8.s4 * weight8_4.s67; \
                out8_1.s01 += input8_1.s4 * weight8_4.s01; \
                out8_1.s23 += input8_1.s4 * weight8_4.s23; \
                out8_1.s45 += input8_1.s4 * weight8_4.s45; \
                out8_1.s67 += input8_1.s4 * weight8_4.s67; \
                weight8_5 = vload8(0, weight + src_addr.s1 + 40); \
                out8_0.s01 += input8.s5 * weight8_5.s01; \
                out8_0.s23 += input8.s5 * weight8_5.s23; \
                out8_0.s45 += input8.s5 * weight8_5.s45; \
                out8_0.s67 += input8.s5 * weight8_5.s67; \
                out8_1.s01 += input8_1.s5 * weight8_5.s01; \
                out8_1.s23 += input8_1.s5 * weight8_5.s23; \
                out8_1.s45 += input8_1.s5 * weight8_5.s45; \
                out8_1.s67 += input8_1.s5 * weight8_5.s67; \
                weight8_6 = vload8(0, weight + src_addr.s1 + 48); \
                out8_0.s01 += input8.s6 * weight8_6.s01; \
                out8_0.s23 += input8.s6 * weight8_6.s23; \
                out8_0.s45 += input8.s6 * weight8_6.s45; \
                out8_0.s67 += input8.s6 * weight8_6.s67; \
                out8_1.s01 += input8_1.s6 * weight8_6.s01; \
                out8_1.s23 += input8_1.s6 * weight8_6.s23; \
                out8_1.s45 += input8_1.s6 * weight8_6.s45; \
                out8_1.s67 += input8_1.s6 * weight8_6.s67; \
                weight8_7 = vload8(0, weight + src_addr.s1 + 56); \
                out8_0.s01 += input8.s7 * weight8_7.s01; \
                out8_0.s23 += input8.s7 * weight8_7.s23; \
                out8_0.s45 += input8.s7 * weight8_7.s45; \
                out8_0.s67 += input8.s7 * weight8_7.s67; \
                out8_1.s01 += input8_1.s7 * weight8_7.s01; \
                out8_1.s23 += input8_1.s7 * weight8_7.s23; \
                out8_1.s45 += input8_1.s7 * weight8_7.s45; \
                out8_1.s67 += input8_1.s7 * weight8_7.s67; \
            } \
        } \
        int outputIndex = get_global_id(0) * topChannel * topHW + get_global_id(1) * 8 * topHW + \
                            get_global_id(2) * 2; \
        GEMMBLOCK2X8_FEED_OUTPUT_VSTORE2(topChannel, out8_0, out8_1, output, outputIndex, topHW) \
    } else if (get_global_id(2) * 2 < topHW) { \
        DATA_T8 out8_0 = vload8(0, bias + get_global_id(1) * 8); \
        { \
            uint2 src_addr = (uint2)(get_global_id(0) * bottomStep + \
                                            (get_global_id(2) * 2 / 24) * 24 * alignedCinKK + \
                                            (get_global_id(2) * 2 % 24 / 2) * 8, \
                                        get_global_id(1) * 8 * alignedCinKK);  \
            uint end_alignedCinKK = src_addr.s0 + 24 * alignedCinKK; \
            for (; src_addr.s0 < end_alignedCinKK; src_addr += (uint2)(192, 64)) { \
                DATA_T8 input8; \
                DATA_T8 weight8; \
                DATA_T8 weight8_1; \
                DATA_T8 weight8_2; \
                DATA_T8 weight8_3; \
                DATA_T8 weight8_4; \
                DATA_T8 weight8_5; \
                DATA_T8 weight8_6; \
                DATA_T8 weight8_7; \
                input8 = vload8(0, input + src_addr.s0); \
                weight8 = vload8(0, weight + src_addr.s1); \
                weight8_1 = vload8(0, weight + src_addr.s1 + 8); \
                weight8_2 = vload8(0, weight + src_addr.s1 + 16); \
                weight8_3 = vload8(0, weight + src_addr.s1 + 24); \
                weight8_4 = vload8(0, weight + src_addr.s1 + 32); \
                weight8_5 = vload8(0, weight + src_addr.s1 + 40); \
                weight8_6 = vload8(0, weight + src_addr.s1 + 48); \
                weight8_7 = vload8(0, weight + src_addr.s1 + 56); \
                out8_0.s01 += input8.s0 * weight8.s01; \
                out8_0.s23 += input8.s0 * weight8.s23; \
                out8_0.s45 += input8.s0 * weight8.s45; \
                out8_0.s67 += input8.s0 * weight8.s67; \
                out8_0.s01 += input8.s1 * weight8_1.s01; \
                out8_0.s23 += input8.s1 * weight8_1.s23; \
                out8_0.s45 += input8.s1 * weight8_1.s45; \
                out8_0.s67 += input8.s1 * weight8_1.s67; \
                out8_0.s01 += input8.s2 * weight8_2.s01; \
                out8_0.s23 += input8.s2 * weight8_2.s23; \
                out8_0.s45 += input8.s2 * weight8_2.s45; \
                out8_0.s67 += input8.s2 * weight8_2.s67; \
                out8_0.s01 += input8.s3 * weight8_3.s01; \
                out8_0.s23 += input8.s3 * weight8_3.s23; \
                out8_0.s45 += input8.s3 * weight8_3.s45; \
                out8_0.s67 += input8.s3 * weight8_3.s67; \
                out8_0.s01 += input8.s4 * weight8_4.s01; \
                out8_0.s23 += input8.s4 * weight8_4.s23; \
                out8_0.s45 += input8.s4 * weight8_4.s45; \
                out8_0.s67 += input8.s4 * weight8_4.s67; \
                out8_0.s01 += input8.s5 * weight8_5.s01; \
                out8_0.s23 += input8.s5 * weight8_5.s23; \
                out8_0.s45 += input8.s5 * weight8_5.s45; \
                out8_0.s67 += input8.s5 * weight8_5.s67; \
                out8_0.s01 += input8.s6 * weight8_6.s01; \
                out8_0.s23 += input8.s6 * weight8_6.s23; \
                out8_0.s45 += input8.s6 * weight8_6.s45; \
                out8_0.s67 += input8.s6 * weight8_6.s67; \
                out8_0.s01 += input8.s7 * weight8_7.s01; \
                out8_0.s23 += input8.s7 * weight8_7.s23; \
                out8_0.s45 += input8.s7 * weight8_7.s45; \
                out8_0.s67 += input8.s7 * weight8_7.s67; \
            } \
        } \
        int outputIndex = get_global_id(0) * topChannel * topHW + get_global_id(1) * 8 * topHW + \
                            get_global_id(2) * 2; \
        GEMMBLOCK2X8_FEED_OUTPUT(topChannel, out8_0, output, outputIndex, topHW) \
    }

#define GEMMBLOCK2X8_SCALAR_FP32(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep) \
    if (get_global_id(2) * 2 + 1 < topHW) { \
        DATA_T8 out8_0 = vload8(0, bias + get_global_id(1) * 8); \
        DATA_T8 out8_1 = out8_0; \
        { \
            uint2 src_addr = (uint2)(get_global_id(0) * bottomStep + \
                                            (get_global_id(2) * 2 / 24) * 24 * alignedCinKK + \
                                            (get_global_id(2) * 2 % 24 / 2) * 8, \
                                        get_global_id(1) * 8 * alignedCinKK);  \
            uint end_alignedCinKK = src_addr.s0 + 24 * alignedCinKK; \
            for (; src_addr.s0 < end_alignedCinKK; src_addr += (uint2)(192, 64)) { \
                DATA_T8 input8; \
                DATA_T8 input8_1; \
                DATA_T8 weight8; \
                input8 = vload8(0, input + src_addr.s0); \
                input8_1 = vload8(0, input + src_addr.s0 + 96); \
                weight8 = vload8(0, weight + src_addr.s1); \
                out8_0.s0 += input8.s0 * weight8.s0; \
                out8_0.s1 += input8.s0 * weight8.s1; \
                out8_0.s2 += input8.s0 * weight8.s2; \
                out8_0.s3 += input8.s0 * weight8.s3; \
                out8_0.s4 += input8.s0 * weight8.s4; \
                out8_0.s5 += input8.s0 * weight8.s5; \
                out8_0.s6 += input8.s0 * weight8.s6; \
                out8_0.s7 += input8.s0 * weight8.s7; \
                out8_1.s0 += input8_1.s0 * weight8.s0; \
                out8_1.s1 += input8_1.s0 * weight8.s1; \
                out8_1.s2 += input8_1.s0 * weight8.s2; \
                out8_1.s3 += input8_1.s0 * weight8.s3; \
                out8_1.s4 += input8_1.s0 * weight8.s4; \
                out8_1.s5 += input8_1.s0 * weight8.s5; \
                out8_1.s6 += input8_1.s0 * weight8.s6; \
                out8_1.s7 += input8_1.s0 * weight8.s7; \
                weight8 = vload8(0, weight + src_addr.s1 + 8); \
                out8_0.s0 += input8.s1 * weight8.s0; \
                out8_0.s1 += input8.s1 * weight8.s1; \
                out8_0.s2 += input8.s1 * weight8.s2; \
                out8_0.s3 += input8.s1 * weight8.s3; \
                out8_0.s4 += input8.s1 * weight8.s4; \
                out8_0.s5 += input8.s1 * weight8.s5; \
                out8_0.s6 += input8.s1 * weight8.s6; \
                out8_0.s7 += input8.s1 * weight8.s7; \
                out8_1.s0 += input8_1.s1 * weight8.s0; \
                out8_1.s1 += input8_1.s1 * weight8.s1; \
                out8_1.s2 += input8_1.s1 * weight8.s2; \
                out8_1.s3 += input8_1.s1 * weight8.s3; \
                out8_1.s4 += input8_1.s1 * weight8.s4; \
                out8_1.s5 += input8_1.s1 * weight8.s5; \
                out8_1.s6 += input8_1.s1 * weight8.s6; \
                out8_1.s7 += input8_1.s1 * weight8.s7; \
                weight8 = vload8(0, weight + src_addr.s1 + 16); \
                out8_0.s0 += input8.s2 * weight8.s0; \
                out8_0.s1 += input8.s2 * weight8.s1; \
                out8_0.s2 += input8.s2 * weight8.s2; \
                out8_0.s3 += input8.s2 * weight8.s3; \
                out8_0.s4 += input8.s2 * weight8.s4; \
                out8_0.s5 += input8.s2 * weight8.s5; \
                out8_0.s6 += input8.s2 * weight8.s6; \
                out8_0.s7 += input8.s2 * weight8.s7; \
                out8_1.s0 += input8_1.s2 * weight8.s0; \
                out8_1.s1 += input8_1.s2 * weight8.s1; \
                out8_1.s2 += input8_1.s2 * weight8.s2; \
                out8_1.s3 += input8_1.s2 * weight8.s3; \
                out8_1.s4 += input8_1.s2 * weight8.s4; \
                out8_1.s5 += input8_1.s2 * weight8.s5; \
                out8_1.s6 += input8_1.s2 * weight8.s6; \
                out8_1.s7 += input8_1.s2 * weight8.s7; \
                weight8 = vload8(0, weight + src_addr.s1 + 24); \
                out8_0.s0 += input8.s3 * weight8.s0; \
                out8_0.s1 += input8.s3 * weight8.s1; \
                out8_0.s2 += input8.s3 * weight8.s2; \
                out8_0.s3 += input8.s3 * weight8.s3; \
                out8_0.s4 += input8.s3 * weight8.s4; \
                out8_0.s5 += input8.s3 * weight8.s5; \
                out8_0.s6 += input8.s3 * weight8.s6; \
                out8_0.s7 += input8.s3 * weight8.s7; \
                out8_1.s0 += input8_1.s3 * weight8.s0; \
                out8_1.s1 += input8_1.s3 * weight8.s1; \
                out8_1.s2 += input8_1.s3 * weight8.s2; \
                out8_1.s3 += input8_1.s3 * weight8.s3; \
                out8_1.s4 += input8_1.s3 * weight8.s4; \
                out8_1.s5 += input8_1.s3 * weight8.s5; \
                out8_1.s6 += input8_1.s3 * weight8.s6; \
                out8_1.s7 += input8_1.s3 * weight8.s7; \
                weight8 = vload8(0, weight + src_addr.s1 + 32); \
                out8_0.s0 += input8.s4 * weight8.s0; \
                out8_0.s1 += input8.s4 * weight8.s1; \
                out8_0.s2 += input8.s4 * weight8.s2; \
                out8_0.s3 += input8.s4 * weight8.s3; \
                out8_0.s4 += input8.s4 * weight8.s4; \
                out8_0.s5 += input8.s4 * weight8.s5; \
                out8_0.s6 += input8.s4 * weight8.s6; \
                out8_0.s7 += input8.s4 * weight8.s7; \
                out8_1.s0 += input8_1.s4 * weight8.s0; \
                out8_1.s1 += input8_1.s4 * weight8.s1; \
                out8_1.s2 += input8_1.s4 * weight8.s2; \
                out8_1.s3 += input8_1.s4 * weight8.s3; \
                out8_1.s4 += input8_1.s4 * weight8.s4; \
                out8_1.s5 += input8_1.s4 * weight8.s5; \
                out8_1.s6 += input8_1.s4 * weight8.s6; \
                out8_1.s7 += input8_1.s4 * weight8.s7; \
                weight8 = vload8(0, weight + src_addr.s1 + 40); \
                out8_0.s0 += input8.s5 * weight8.s0; \
                out8_0.s1 += input8.s5 * weight8.s1; \
                out8_0.s2 += input8.s5 * weight8.s2; \
                out8_0.s3 += input8.s5 * weight8.s3; \
                out8_0.s4 += input8.s5 * weight8.s4; \
                out8_0.s5 += input8.s5 * weight8.s5; \
                out8_0.s6 += input8.s5 * weight8.s6; \
                out8_0.s7 += input8.s5 * weight8.s7; \
                out8_1.s0 += input8_1.s5 * weight8.s0; \
                out8_1.s1 += input8_1.s5 * weight8.s1; \
                out8_1.s2 += input8_1.s5 * weight8.s2; \
                out8_1.s3 += input8_1.s5 * weight8.s3; \
                out8_1.s4 += input8_1.s5 * weight8.s4; \
                out8_1.s5 += input8_1.s5 * weight8.s5; \
                out8_1.s6 += input8_1.s5 * weight8.s6; \
                out8_1.s7 += input8_1.s5 * weight8.s7; \
                weight8 = vload8(0, weight + src_addr.s1 + 48); \
                out8_0.s0 += input8.s6 * weight8.s0; \
                out8_0.s1 += input8.s6 * weight8.s1; \
                out8_0.s2 += input8.s6 * weight8.s2; \
                out8_0.s3 += input8.s6 * weight8.s3; \
                out8_0.s4 += input8.s6 * weight8.s4; \
                out8_0.s5 += input8.s6 * weight8.s5; \
                out8_0.s6 += input8.s6 * weight8.s6; \
                out8_0.s7 += input8.s6 * weight8.s7; \
                out8_1.s0 += input8_1.s6 * weight8.s0; \
                out8_1.s1 += input8_1.s6 * weight8.s1; \
                out8_1.s2 += input8_1.s6 * weight8.s2; \
                out8_1.s3 += input8_1.s6 * weight8.s3; \
                out8_1.s4 += input8_1.s6 * weight8.s4; \
                out8_1.s5 += input8_1.s6 * weight8.s5; \
                out8_1.s6 += input8_1.s6 * weight8.s6; \
                out8_1.s7 += input8_1.s6 * weight8.s7; \
                weight8 = vload8(0, weight + src_addr.s1 + 56); \
                out8_0.s0 += input8.s7 * weight8.s0; \
                out8_0.s1 += input8.s7 * weight8.s1; \
                out8_0.s2 += input8.s7 * weight8.s2; \
                out8_0.s3 += input8.s7 * weight8.s3; \
                out8_0.s4 += input8.s7 * weight8.s4; \
                out8_0.s5 += input8.s7 * weight8.s5; \
                out8_0.s6 += input8.s7 * weight8.s6; \
                out8_0.s7 += input8.s7 * weight8.s7; \
                out8_1.s0 += input8_1.s7 * weight8.s0; \
                out8_1.s1 += input8_1.s7 * weight8.s1; \
                out8_1.s2 += input8_1.s7 * weight8.s2; \
                out8_1.s3 += input8_1.s7 * weight8.s3; \
                out8_1.s4 += input8_1.s7 * weight8.s4; \
                out8_1.s5 += input8_1.s7 * weight8.s5; \
                out8_1.s6 += input8_1.s7 * weight8.s6; \
                out8_1.s7 += input8_1.s7 * weight8.s7; \
            } \
        } \
        int outputIndex = get_global_id(0) * topChannel * topHW + get_global_id(1) * 8 * topHW + \
                            get_global_id(2) * 2; \
        GEMMBLOCK2X8_FEED_OUTPUT_VSTORE2(topChannel, out8_0, out8_1, output, outputIndex, topHW) \
    } else if (get_global_id(2) * 2 < topHW) { \
        DATA_T8 out8_0 = vload8(0, bias + get_global_id(1) * 8); \
        { \
            uint2 src_addr = (uint2)(get_global_id(0) * bottomStep + \
                                            (get_global_id(2) * 2 / 24) * 24 * alignedCinKK + \
                                            (get_global_id(2) * 2 % 24 / 2) * 8, \
                                        get_global_id(1) * 8 * alignedCinKK);  \
            uint end_alignedCinKK = src_addr.s0 + 24 * alignedCinKK; \
            for (; src_addr.s0 < end_alignedCinKK; src_addr += (uint2)(192, 64)) { \
                DATA_T8 input8; \
                DATA_T8 weight8; \
                input8 = vload8(0, input + src_addr.s0); \
                weight8 = vload8(0, weight + src_addr.s1); \
                out8_0.s0 += input8.s0 * weight8.s0; \
                out8_0.s1 += input8.s0 * weight8.s1; \
                out8_0.s2 += input8.s0 * weight8.s2; \
                out8_0.s3 += input8.s0 * weight8.s3; \
                out8_0.s4 += input8.s0 * weight8.s4; \
                out8_0.s5 += input8.s0 * weight8.s5; \
                out8_0.s6 += input8.s0 * weight8.s6; \
                out8_0.s7 += input8.s0 * weight8.s7; \
                weight8 = vload8(0, weight + src_addr.s1 + 8); \
                out8_0.s0 += input8.s1 * weight8.s0; \
                out8_0.s1 += input8.s1 * weight8.s1; \
                out8_0.s2 += input8.s1 * weight8.s2; \
                out8_0.s3 += input8.s1 * weight8.s3; \
                out8_0.s4 += input8.s1 * weight8.s4; \
                out8_0.s5 += input8.s1 * weight8.s5; \
                out8_0.s6 += input8.s1 * weight8.s6; \
                out8_0.s7 += input8.s1 * weight8.s7; \
                weight8 = vload8(0, weight + src_addr.s1 + 16); \
                out8_0.s0 += input8.s2 * weight8.s0; \
                out8_0.s1 += input8.s2 * weight8.s1; \
                out8_0.s2 += input8.s2 * weight8.s2; \
                out8_0.s3 += input8.s2 * weight8.s3; \
                out8_0.s4 += input8.s2 * weight8.s4; \
                out8_0.s5 += input8.s2 * weight8.s5; \
                out8_0.s6 += input8.s2 * weight8.s6; \
                out8_0.s7 += input8.s2 * weight8.s7; \
                weight8 = vload8(0, weight + src_addr.s1 + 24); \
                out8_0.s0 += input8.s3 * weight8.s0; \
                out8_0.s1 += input8.s3 * weight8.s1; \
                out8_0.s2 += input8.s3 * weight8.s2; \
                out8_0.s3 += input8.s3 * weight8.s3; \
                out8_0.s4 += input8.s3 * weight8.s4; \
                out8_0.s5 += input8.s3 * weight8.s5; \
                out8_0.s6 += input8.s3 * weight8.s6; \
                out8_0.s7 += input8.s3 * weight8.s7; \
                weight8 = vload8(0, weight + src_addr.s1 + 32); \
                out8_0.s0 += input8.s4 * weight8.s0; \
                out8_0.s1 += input8.s4 * weight8.s1; \
                out8_0.s2 += input8.s4 * weight8.s2; \
                out8_0.s3 += input8.s4 * weight8.s3; \
                out8_0.s4 += input8.s4 * weight8.s4; \
                out8_0.s5 += input8.s4 * weight8.s5; \
                out8_0.s6 += input8.s4 * weight8.s6; \
                out8_0.s7 += input8.s4 * weight8.s7; \
                weight8 = vload8(0, weight + src_addr.s1 + 40); \
                out8_0.s0 += input8.s5 * weight8.s0; \
                out8_0.s1 += input8.s5 * weight8.s1; \
                out8_0.s2 += input8.s5 * weight8.s2; \
                out8_0.s3 += input8.s5 * weight8.s3; \
                out8_0.s4 += input8.s5 * weight8.s4; \
                out8_0.s5 += input8.s5 * weight8.s5; \
                out8_0.s6 += input8.s5 * weight8.s6; \
                out8_0.s7 += input8.s5 * weight8.s7; \
                weight8 = vload8(0, weight + src_addr.s1 + 48); \
                out8_0.s0 += input8.s6 * weight8.s0; \
                out8_0.s1 += input8.s6 * weight8.s1; \
                out8_0.s2 += input8.s6 * weight8.s2; \
                out8_0.s3 += input8.s6 * weight8.s3; \
                out8_0.s4 += input8.s6 * weight8.s4; \
                out8_0.s5 += input8.s6 * weight8.s5; \
                out8_0.s6 += input8.s6 * weight8.s6; \
                out8_0.s7 += input8.s6 * weight8.s7; \
                weight8 = vload8(0, weight + src_addr.s1 + 56); \
                out8_0.s0 += input8.s7 * weight8.s0; \
                out8_0.s1 += input8.s7 * weight8.s1; \
                out8_0.s2 += input8.s7 * weight8.s2; \
                out8_0.s3 += input8.s7 * weight8.s3; \
                out8_0.s4 += input8.s7 * weight8.s4; \
                out8_0.s5 += input8.s7 * weight8.s5; \
                out8_0.s6 += input8.s7 * weight8.s6; \
                out8_0.s7 += input8.s7 * weight8.s7; \
            } \
        } \
        int outputIndex = get_global_id(0) * topChannel * topHW + get_global_id(1) * 8 * topHW + \
                            get_global_id(2) * 2; \
        GEMMBLOCK2X8_FEED_OUTPUT(topChannel, out8_0, output, outputIndex, topHW) \
    }

#define GEMMBLOCK2X4_FEED_OUTPUT_VSTORE2(topChannel, bias, out8_0, output, outputIndex, topHW) \
    if (get_global_id(1) * 4 + 3 < topChannel) { \
        DATA_T4 bia = vload4(0, bias + get_global_id(1) * 4); \
        out8_0.s0123 += bia; \
        out8_0.s4567 += bia; \
        vstore2(ACT_VEC_F(DATA_T2, out8_0.s04), 0, output + outputIndex); \
        vstore2(ACT_VEC_F(DATA_T2, out8_0.s15), 0, output + outputIndex + topHW); \
        vstore2(ACT_VEC_F(DATA_T2, out8_0.s26), 0, output + outputIndex + 2 * topHW); \
        vstore2(ACT_VEC_F(DATA_T2, out8_0.s37), 0, output + outputIndex + 3 * topHW); \
    } else if (get_global_id(1) * 4 + 2 < topChannel) { \
        DATA_T3 bia = vload3(0, bias + get_global_id(1) * 4); \
        out8_0.s012 += bia; \
        out8_0.s456 += bia; \
        vstore2(ACT_VEC_F(DATA_T2, out8_0.s04), 0, output + outputIndex); \
        vstore2(ACT_VEC_F(DATA_T2, out8_0.s15), 0, output + outputIndex + topHW); \
        vstore2(ACT_VEC_F(DATA_T2, out8_0.s26), 0, output + outputIndex + 2 * topHW); \
    } else if (get_global_id(1) * 4 + 1 < topChannel) { \
        DATA_T2 bia = vload2(0, bias + get_global_id(1) * 4); \
        out8_0.s01 += bia; \
        out8_0.s45 += bia; \
        vstore2(ACT_VEC_F(DATA_T2, out8_0.s04), 0, output + outputIndex); \
        vstore2(ACT_VEC_F(DATA_T2, out8_0.s15), 0, output + outputIndex + topHW); \
    } else if (get_global_id(1) * 4 < topChannel) { \
        DATA_T bia = bias[get_global_id(1) * 4]; \
        out8_0.s0 += bia; \
        out8_0.s4 += bia; \
        vstore2(ACT_VEC_F(DATA_T2, out8_0.s04), 0, output + outputIndex); \
    }

#define GEMMBLOCK2X4_FEED_OUTPUT(topChannel, bias, out8_0, output, outputIndex, topHW) \
    if (get_global_id(1) * 4 + 3 < topChannel) { \
        DATA_T4 bia = vload4(0, bias + get_global_id(1) * 4); \
        out8_0.s0123 += bia; \
        output[outputIndex] = ACT_VEC_F(DATA_T, out8_0.s0); \
        output[outputIndex + topHW] = ACT_VEC_F(DATA_T, out8_0.s1); \
        output[outputIndex + 2 * topHW] = ACT_VEC_F(DATA_T, out8_0.s2); \
        output[outputIndex + 3 * topHW] = ACT_VEC_F(DATA_T, out8_0.s3); \
    } else if (get_global_id(1) * 4 + 2 < topChannel) { \
        DATA_T3 bia = vload3(0, bias + get_global_id(1) * 4); \
        out8_0.s012 += bia; \
        output[outputIndex] = ACT_VEC_F(DATA_T, out8_0.s0); \
        output[outputIndex + topHW] = ACT_VEC_F(DATA_T, out8_0.s1); \
        output[outputIndex + 2 * topHW] = ACT_VEC_F(DATA_T, out8_0.s2); \
    } else if (get_global_id(1) * 4 + 1 < topChannel) { \
        DATA_T2 bia = vload2(0, bias + get_global_id(1) * 4); \
        out8_0.s01 += bia; \
        output[outputIndex] = ACT_VEC_F(DATA_T, out8_0.s0); \
        output[outputIndex + topHW] = ACT_VEC_F(DATA_T, out8_0.s1); \
    } else if (get_global_id(1) * 4 < topChannel) { \
        DATA_T bia = bias[get_global_id(1) * 4]; \
        out8_0.s0 += bia; \
        output[outputIndex] = ACT_VEC_F(DATA_T, out8_0.s0); \
    }

#define GEMMBLOCK2X4(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep) \
    if (get_global_id(2) * 2 + 1 < topHW) { \
        DATA_T out_0 = 0.0f; \
        DATA_T out_1 = 0.0f; \
        DATA_T out_2 = 0.0f; \
        DATA_T out_3 = 0.0f; \
        DATA_T out_4 = 0.0f; \
        DATA_T out_5 = 0.0f; \
        DATA_T out_6 = 0.0f; \
        DATA_T out_7 = 0.0f; \
        uint2 src_addr = (uint2)(get_global_id(0) * bottomStep + \
                                        (get_global_id(2) * 2 / 24) * 24 * alignedCinKK + \
                                        (get_global_id(2) * 2 % 24 / 2) * 8, \
                                    get_global_id(1) * 4 * alignedCinKK);  \
        uint end_alignedCinKK = src_addr.s0 + 24 * alignedCinKK; \
        for (; src_addr.s0 < end_alignedCinKK; src_addr += (uint2)(192, 32)) { \
            DATA_T8 input8_0; \
            DATA_T8 input8_1; \
            DATA_T8 weight8_0; \
            input8_0 = vload8(0, input + src_addr.s0); \
            input8_1 = vload8(0, input + src_addr.s0 + 96); \
            weight8_0 = vload8(0, weight + src_addr.s1); \
            out_0 += dot8(input8_0, weight8_0); \
            out_4 += dot8(input8_1, weight8_0); \
            weight8_0 = vload8(1, weight + src_addr.s1); \
            out_1 += dot8(input8_0, weight8_0); \
            out_5 += dot8(input8_1, weight8_0); \
            weight8_0 = vload8(2, weight + src_addr.s1); \
            out_2 += dot8(input8_0, weight8_0); \
            out_6 += dot8(input8_1, weight8_0); \
            weight8_0 = vload8(3, weight + src_addr.s1); \
            out_3 += dot8(input8_0, weight8_0); \
            out_7 += dot8(input8_1, weight8_0); \
        } \
        DATA_T8 out8_0; \
        out8_0.s0 = out_0; \
        out8_0.s1 = out_1; \
        out8_0.s2 = out_2; \
        out8_0.s3 = out_3; \
        out8_0.s4 = out_4; \
        out8_0.s5 = out_5; \
        out8_0.s6 = out_6; \
        out8_0.s7 = out_7; \
        int outputIndex = get_global_id(0) * topChannel * topHW + get_global_id(1) * 4 * topHW + \
                            get_global_id(2) * 2; \
        GEMMBLOCK2X4_FEED_OUTPUT_VSTORE2(topChannel, bias, out8_0, output, outputIndex, topHW) \
    } else if (get_global_id(2) * 2 < topHW) { \
        DATA_T8 out8_0 = 0.0f; \
        DATA_T8 out8_1 = 0.0f; \
        DATA_T8 out8_2 = 0.0f; \
        DATA_T8 out8_3 = 0.0f; \
        uint2 src_addr = (uint2)(get_global_id(0) * bottomStep + \
                                        (get_global_id(2) * 2 / 24) * 24 * alignedCinKK + \
                                        (get_global_id(2) * 2 % 24 / 2) * 8, \
                                    get_global_id(1) * 4 * alignedCinKK);  \
        uint end_alignedCinKK = src_addr.s0 + 24 * alignedCinKK; \
        for (; src_addr.s0 < end_alignedCinKK; src_addr += (uint2)(192, 32)) { \
            DATA_T8 input8_0; \
            DATA_T8 weight8_0; \
            input8_0 = vload8(0, input + src_addr.s0); \
            weight8_0 = vload8(0, weight + src_addr.s1); \
            out8_0 += input8_0 * weight8_0; \
            weight8_0 = vload8(1, weight + src_addr.s1); \
            out8_1 += input8_0 * weight8_0; \
            weight8_0 = vload8(2, weight + src_addr.s1); \
            out8_2 += input8_0 * weight8_0; \
            weight8_0 = vload8(3, weight + src_addr.s1); \
            out8_3 += input8_0 * weight8_0; \
        } \
        out8_0.s0 = out8_0.s0 + out8_0.s1 + out8_0.s2 + out8_0.s3 + out8_0.s4 + out8_0.s5 + \
                    out8_0.s6 + out8_0.s7; \
        out8_0.s1 = out8_1.s0 + out8_1.s1 + out8_1.s2 + out8_1.s3 + out8_1.s4 + out8_1.s5 + \
                    out8_1.s6 + out8_1.s7; \
        out8_0.s2 = out8_2.s0 + out8_2.s1 + out8_2.s2 + out8_2.s3 + out8_2.s4 + out8_2.s5 + \
                    out8_2.s6 + out8_2.s7; \
        out8_0.s3 = out8_3.s0 + out8_3.s1 + out8_3.s2 + out8_3.s3 + out8_3.s4 + out8_3.s5 + \
                    out8_3.s6 + out8_3.s7; \
        int outputIndex = get_global_id(0) * topChannel * topHW + get_global_id(1) * 4 * topHW + \
                            get_global_id(2) * 2; \
        GEMMBLOCK2X4_FEED_OUTPUT(topChannel, bias, out8_0, output, outputIndex, topHW) \
    }

#define GEMMBLOCK2X4_SCALAR_FP16(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep) \
    if (get_global_id(2) * 2 + 1 < topHW) { \
        DATA_T8 out8_0 = 0.0f; \
        DATA_T2 out2_0 = 0.0f; \
        DATA_T2 out2_1 = 0.0f; \
        DATA_T2 out2_2 = 0.0f; \
        DATA_T2 out2_3 = 0.0f; \
        DATA_T2 out2_4 = 0.0f; \
        DATA_T2 out2_5 = 0.0f; \
        DATA_T2 out2_6 = 0.0f; \
        DATA_T2 out2_7 = 0.0f; \
        uint2 src_addr = (uint2)(get_global_id(0) * bottomStep + \
                                        (get_global_id(2) * 2 / 24) * 24 * alignedCinKK + \
                                        (get_global_id(2) * 2 % 24 / 2) * 8, \
                                    get_global_id(1) * 4 * alignedCinKK);  \
        uint end_alignedCinKK = src_addr.s0 + 24 * alignedCinKK; \
        for (; src_addr.s0 < end_alignedCinKK; src_addr += (uint2)(192, 32)) { \
            DATA_T8 input8_0; \
            DATA_T8 input8_1; \
            DATA_T16 weight16_0; \
            DATA_T16 weight16_1; \
            input8_0 = vload8(0, input + src_addr.s0); \
            input8_1 = vload8(0, input + src_addr.s0 + 96); \
            weight16_0 = vload16(0, weight + src_addr.s1); \
            weight16_1 = vload16(1, weight + src_addr.s1); \
            out2_0 += input8_0.s01 * weight16_0.s01; \
            out2_0 += input8_0.s23 * weight16_0.s23; \
            out2_0 += input8_0.s45 * weight16_0.s45; \
            out2_0 += input8_0.s67 * weight16_0.s67; \
            out2_1 += input8_0.s01 * weight16_0.s89; \
            out2_1 += input8_0.s23 * weight16_0.sab; \
            out2_1 += input8_0.s45 * weight16_0.scd; \
            out2_1 += input8_0.s67 * weight16_0.sef; \
            out2_2 += input8_0.s01 * weight16_1.s01; \
            out2_2 += input8_0.s23 * weight16_1.s23; \
            out2_2 += input8_0.s45 * weight16_1.s45; \
            out2_2 += input8_0.s67 * weight16_1.s67; \
            out2_3 += input8_0.s01 * weight16_1.s89; \
            out2_3 += input8_0.s23 * weight16_1.sab; \
            out2_3 += input8_0.s45 * weight16_1.scd; \
            out2_3 += input8_0.s67 * weight16_1.sef; \
            out2_4 += input8_1.s01 * weight16_0.s01; \
            out2_4 += input8_1.s23 * weight16_0.s23; \
            out2_4 += input8_1.s45 * weight16_0.s45; \
            out2_4 += input8_1.s67 * weight16_0.s67; \
            out2_5 += input8_1.s01 * weight16_0.s89; \
            out2_5 += input8_1.s23 * weight16_0.sab; \
            out2_5 += input8_1.s45 * weight16_0.scd; \
            out2_5 += input8_1.s67 * weight16_0.sef; \
            out2_6 += input8_1.s01 * weight16_1.s01; \
            out2_6 += input8_1.s23 * weight16_1.s23; \
            out2_6 += input8_1.s45 * weight16_1.s45; \
            out2_6 += input8_1.s67 * weight16_1.s67; \
            out2_7 += input8_1.s01 * weight16_1.s89; \
            out2_7 += input8_1.s23 * weight16_1.sab; \
            out2_7 += input8_1.s45 * weight16_1.scd; \
            out2_7 += input8_1.s67 * weight16_1.sef; \
        } \
        out8_0.s0 = out2_0.s0 + out2_0.s1; \
        out8_0.s1 = out2_1.s0 + out2_1.s1; \
        out8_0.s2 = out2_2.s0 + out2_2.s1; \
        out8_0.s3 = out2_3.s0 + out2_3.s1; \
        out8_0.s4 = out2_4.s0 + out2_4.s1; \
        out8_0.s5 = out2_5.s0 + out2_5.s1; \
        out8_0.s6 = out2_6.s0 + out2_6.s1; \
        out8_0.s7 = out2_7.s0 + out2_7.s1; \
        int outputIndex = get_global_id(0) * topChannel * topHW + get_global_id(1) * 4 * topHW + \
                            get_global_id(2) * 2; \
        GEMMBLOCK2X4_FEED_OUTPUT_VSTORE2(topChannel, bias, out8_0, output, outputIndex, topHW) \
    } else if (get_global_id(2) * 2 < topHW) { \
        DATA_T8 out8_0 = 0.0f; \
        DATA_T2 out2_0 = 0.0f; \
        DATA_T2 out2_1 = 0.0f; \
        DATA_T2 out2_2 = 0.0f; \
        DATA_T2 out2_3 = 0.0f; \
        uint2 src_addr = (uint2)(get_global_id(0) * bottomStep + \
                                        (get_global_id(2) * 2 / 24) * 24 * alignedCinKK + \
                                        (get_global_id(2) * 2 % 24 / 2) * 8, \
                                    get_global_id(1) * 4 * alignedCinKK);  \
        uint end_alignedCinKK = src_addr.s0 + 24 * alignedCinKK; \
        for (; src_addr.s0 < end_alignedCinKK; src_addr += (uint2)(192, 32)) { \
            DATA_T8 input8_0; \
            DATA_T16 weight16_0; \
            DATA_T16 weight16_1; \
            input8_0 = vload8(0, input + src_addr.s0); \
            weight16_0 = vload16(0, weight + src_addr.s1); \
            weight16_1 = vload16(1, weight + src_addr.s1); \
            out2_0 += input8_0.s01 * weight16_0.s01; \
            out2_0 += input8_0.s23 * weight16_0.s23; \
            out2_0 += input8_0.s45 * weight16_0.s45; \
            out2_0 += input8_0.s67 * weight16_0.s67; \
            out2_1 += input8_0.s01 * weight16_0.s89; \
            out2_1 += input8_0.s23 * weight16_0.sab; \
            out2_1 += input8_0.s45 * weight16_0.scd; \
            out2_1 += input8_0.s67 * weight16_0.sef; \
            out2_2 += input8_0.s01 * weight16_1.s01; \
            out2_2 += input8_0.s23 * weight16_1.s23; \
            out2_2 += input8_0.s45 * weight16_1.s45; \
            out2_2 += input8_0.s67 * weight16_1.s67; \
            out2_3 += input8_0.s01 * weight16_1.s89; \
            out2_3 += input8_0.s23 * weight16_1.sab; \
            out2_3 += input8_0.s45 * weight16_1.scd; \
            out2_3 += input8_0.s67 * weight16_1.sef; \
        } \
        out8_0.s0 = out2_0.s0 + out2_0.s1; \
        out8_0.s1 = out2_1.s0 + out2_1.s1; \
        out8_0.s2 = out2_2.s0 + out2_2.s1; \
        out8_0.s3 = out2_3.s0 + out2_3.s1; \
        int outputIndex = get_global_id(0) * topChannel * topHW + get_global_id(1) * 4 * topHW + \
                            get_global_id(2) * 2; \
        GEMMBLOCK2X4_FEED_OUTPUT(topChannel, bias, out8_0, output, outputIndex, topHW) \
    }

#define GEMMBLOCK2X4_SCALAR_FP32(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep) \
    if (get_global_id(2) * 2 + 1 < topHW) { \
        DATA_T8 out8_0 = 0.0f; \
        DATA_T out2_0 = 0.0f; \
        DATA_T out2_1 = 0.0f; \
        DATA_T out2_2 = 0.0f; \
        DATA_T out2_3 = 0.0f; \
        DATA_T out2_4 = 0.0f; \
        DATA_T out2_5 = 0.0f; \
        DATA_T out2_6 = 0.0f; \
        DATA_T out2_7 = 0.0f; \
        uint2 src_addr = (uint2)(get_global_id(0) * bottomStep + \
                                        (get_global_id(2) * 2 / 24) * 24 * alignedCinKK + \
                                        (get_global_id(2) * 2 % 24 / 2) * 8, \
                                    get_global_id(1) * 4 * alignedCinKK);  \
        uint end_alignedCinKK = src_addr.s0 + 24 * alignedCinKK; \
        for (; src_addr.s0 < end_alignedCinKK; src_addr += (uint2)(192, 32)) { \
            DATA_T8 input8_0; \
            DATA_T8 input8_1; \
            DATA_T16 weight16_0; \
            DATA_T16 weight16_1; \
            input8_0 = vload8(0, input + src_addr.s0); \
            input8_1 = vload8(0, input + src_addr.s0 + 96); \
            weight16_0 = vload16(0, weight + src_addr.s1); \
            weight16_1 = vload16(1, weight + src_addr.s1); \
            out2_0 += input8_0.s0 * weight16_0.s0; \
            out2_0 += input8_0.s1 * weight16_0.s1; \
            out2_0 += input8_0.s2 * weight16_0.s2; \
            out2_0 += input8_0.s3 * weight16_0.s3; \
            out2_0 += input8_0.s4 * weight16_0.s4; \
            out2_0 += input8_0.s5 * weight16_0.s5; \
            out2_0 += input8_0.s6 * weight16_0.s6; \
            out2_0 += input8_0.s7 * weight16_0.s7; \
            out2_1 += input8_0.s0 * weight16_0.s8; \
            out2_1 += input8_0.s1 * weight16_0.s9; \
            out2_1 += input8_0.s2 * weight16_0.sA; \
            out2_1 += input8_0.s3 * weight16_0.sB; \
            out2_1 += input8_0.s4 * weight16_0.sC; \
            out2_1 += input8_0.s5 * weight16_0.sD; \
            out2_1 += input8_0.s6 * weight16_0.sE; \
            out2_1 += input8_0.s7 * weight16_0.sF; \
            out2_2 += input8_0.s0 * weight16_1.s0; \
            out2_2 += input8_0.s1 * weight16_1.s1; \
            out2_2 += input8_0.s2 * weight16_1.s2; \
            out2_2 += input8_0.s3 * weight16_1.s3; \
            out2_2 += input8_0.s4 * weight16_1.s4; \
            out2_2 += input8_0.s5 * weight16_1.s5; \
            out2_2 += input8_0.s6 * weight16_1.s6; \
            out2_2 += input8_0.s7 * weight16_1.s7; \
            out2_3 += input8_0.s0 * weight16_1.s8; \
            out2_3 += input8_0.s1 * weight16_1.s9; \
            out2_3 += input8_0.s2 * weight16_1.sA; \
            out2_3 += input8_0.s3 * weight16_1.sB; \
            out2_3 += input8_0.s4 * weight16_1.sC; \
            out2_3 += input8_0.s5 * weight16_1.sD; \
            out2_3 += input8_0.s6 * weight16_1.sE; \
            out2_3 += input8_0.s7 * weight16_1.sF; \
            out2_4 += input8_1.s0 * weight16_0.s0; \
            out2_4 += input8_1.s1 * weight16_0.s1; \
            out2_4 += input8_1.s2 * weight16_0.s2; \
            out2_4 += input8_1.s3 * weight16_0.s3; \
            out2_4 += input8_1.s4 * weight16_0.s4; \
            out2_4 += input8_1.s5 * weight16_0.s5; \
            out2_4 += input8_1.s6 * weight16_0.s6; \
            out2_4 += input8_1.s7 * weight16_0.s7; \
            out2_5 += input8_1.s0 * weight16_0.s8; \
            out2_5 += input8_1.s1 * weight16_0.s9; \
            out2_5 += input8_1.s2 * weight16_0.sA; \
            out2_5 += input8_1.s3 * weight16_0.sB; \
            out2_5 += input8_1.s4 * weight16_0.sC; \
            out2_5 += input8_1.s5 * weight16_0.sD; \
            out2_5 += input8_1.s6 * weight16_0.sE; \
            out2_5 += input8_1.s7 * weight16_0.sF; \
            out2_6 += input8_1.s0 * weight16_1.s0; \
            out2_6 += input8_1.s1 * weight16_1.s1; \
            out2_6 += input8_1.s2 * weight16_1.s2; \
            out2_6 += input8_1.s3 * weight16_1.s3; \
            out2_6 += input8_1.s4 * weight16_1.s4; \
            out2_6 += input8_1.s5 * weight16_1.s5; \
            out2_6 += input8_1.s6 * weight16_1.s6; \
            out2_6 += input8_1.s7 * weight16_1.s7; \
            out2_7 += input8_1.s0 * weight16_1.s8; \
            out2_7 += input8_1.s1 * weight16_1.s9; \
            out2_7 += input8_1.s2 * weight16_1.sA; \
            out2_7 += input8_1.s3 * weight16_1.sB; \
            out2_7 += input8_1.s4 * weight16_1.sC; \
            out2_7 += input8_1.s5 * weight16_1.sD; \
            out2_7 += input8_1.s6 * weight16_1.sE; \
            out2_7 += input8_1.s7 * weight16_1.sF; \
        } \
        out8_0.s0 = out2_0; \
        out8_0.s1 = out2_1; \
        out8_0.s2 = out2_2; \
        out8_0.s3 = out2_3; \
        out8_0.s4 = out2_4; \
        out8_0.s5 = out2_5; \
        out8_0.s6 = out2_6; \
        out8_0.s7 = out2_7; \
        int outputIndex = get_global_id(0) * topChannel * topHW + get_global_id(1) * 4 * topHW + \
                            get_global_id(2) * 2; \
        GEMMBLOCK2X4_FEED_OUTPUT_VSTORE2(topChannel, bias, out8_0, output, outputIndex, topHW) \
    } else if (get_global_id(2) * 2 < topHW) { \
        DATA_T8 out8_0 = 0.0f; \
        DATA_T out2_0 = 0.0f; \
        DATA_T out2_1 = 0.0f; \
        DATA_T out2_2 = 0.0f; \
        DATA_T out2_3 = 0.0f; \
        uint2 src_addr = (uint2)(get_global_id(0) * bottomStep + \
                                        (get_global_id(2) * 2 / 24) * 24 * alignedCinKK + \
                                        (get_global_id(2) * 2 % 24 / 2) * 8, \
                                    get_global_id(1) * 4 * alignedCinKK);  \
        uint end_alignedCinKK = src_addr.s0 + 24 * alignedCinKK; \
        for (; src_addr.s0 < end_alignedCinKK; src_addr += (uint2)(192, 32)) { \
            DATA_T8 input8_0; \
            DATA_T16 weight16_0; \
            DATA_T16 weight16_1; \
            input8_0 = vload8(0, input + src_addr.s0); \
            weight16_0 = vload16(0, weight + src_addr.s1); \
            weight16_1 = vload16(1, weight + src_addr.s1); \
            out2_0 += input8_0.s0 * weight16_0.s0; \
            out2_0 += input8_0.s1 * weight16_0.s1; \
            out2_0 += input8_0.s2 * weight16_0.s2; \
            out2_0 += input8_0.s3 * weight16_0.s3; \
            out2_0 += input8_0.s4 * weight16_0.s4; \
            out2_0 += input8_0.s5 * weight16_0.s5; \
            out2_0 += input8_0.s6 * weight16_0.s6; \
            out2_0 += input8_0.s7 * weight16_0.s7; \
            out2_1 += input8_0.s0 * weight16_0.s8; \
            out2_1 += input8_0.s1 * weight16_0.s9; \
            out2_1 += input8_0.s2 * weight16_0.sA; \
            out2_1 += input8_0.s3 * weight16_0.sB; \
            out2_1 += input8_0.s4 * weight16_0.sC; \
            out2_1 += input8_0.s5 * weight16_0.sD; \
            out2_1 += input8_0.s6 * weight16_0.sE; \
            out2_1 += input8_0.s7 * weight16_0.sF; \
            out2_2 += input8_0.s0 * weight16_1.s0; \
            out2_2 += input8_0.s1 * weight16_1.s1; \
            out2_2 += input8_0.s2 * weight16_1.s2; \
            out2_2 += input8_0.s3 * weight16_1.s3; \
            out2_2 += input8_0.s4 * weight16_1.s4; \
            out2_2 += input8_0.s5 * weight16_1.s5; \
            out2_2 += input8_0.s6 * weight16_1.s6; \
            out2_2 += input8_0.s7 * weight16_1.s7; \
            out2_3 += input8_0.s0 * weight16_1.s8; \
            out2_3 += input8_0.s1 * weight16_1.s9; \
            out2_3 += input8_0.s2 * weight16_1.sA; \
            out2_3 += input8_0.s3 * weight16_1.sB; \
            out2_3 += input8_0.s4 * weight16_1.sC; \
            out2_3 += input8_0.s5 * weight16_1.sD; \
            out2_3 += input8_0.s6 * weight16_1.sE; \
            out2_3 += input8_0.s7 * weight16_1.sF; \
        } \
        out8_0.s0 = out2_0; \
        out8_0.s1 = out2_1; \
        out8_0.s2 = out2_2; \
        out8_0.s3 = out2_3; \
        int outputIndex = get_global_id(0) * topChannel * topHW + get_global_id(1) * 4 * topHW + \
                            get_global_id(2) * 2; \
        GEMMBLOCK2X4_FEED_OUTPUT(topChannel, bias, out8_0, output, outputIndex, topHW) \
    }

#define MAD_I16W16O4(I16, W16, O4_0, O4_1, O4_2, O4_3) \
    O4_0 = mad(I16.s0, W16.lo.lo, O4_0); \
    O4_0 = mad(I16.s4, W16.lo.hi, O4_0); \
    O4_0 = mad(I16.s8, W16.hi.lo, O4_0); \
    O4_0 = mad(I16.sC, W16.hi.hi, O4_0); \
    O4_1 = mad(I16.s1, W16.lo.lo, O4_1); \
    O4_1 = mad(I16.s5, W16.lo.hi, O4_1); \
    O4_1 = mad(I16.s9, W16.hi.lo, O4_1); \
    O4_1 = mad(I16.sD, W16.hi.hi, O4_1); \
    O4_2 = mad(I16.s2, W16.lo.lo, O4_2); \
    O4_2 = mad(I16.s6, W16.lo.hi, O4_2); \
    O4_2 = mad(I16.sA, W16.hi.lo, O4_2); \
    O4_2 = mad(I16.sE, W16.hi.hi, O4_2); \
    O4_3 = mad(I16.s3, W16.lo.lo, O4_3); \
    O4_3 = mad(I16.s7, W16.lo.hi, O4_3); \
    O4_3 = mad(I16.sB, W16.hi.lo, O4_3); \
    O4_3 = mad(I16.sF, W16.hi.hi, O4_3); \

#define GEMMVALHALL4X4(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep) \
    int GID2 = get_global_id(2) * 4; \
    int GID1 = get_global_id(1) * 4; \
    if (GID2 < topHW) { \
        DATA_T16 out8_0 = 0.0f; \
        DATA_T4 out_0 = 0.0f; \
        DATA_T4 out_1 = 0.0f; \
        DATA_T4 out_2 = 0.0f; \
        DATA_T4 out_3 = 0.0f; \
        uint2 src_addr = \
            (uint2)(get_global_id(0) * bottomStep + (GID2 / 32) * 32 * alignedCinKK + \
                        (GID2 % 32 / 4) * 16, \
                    GID1 / 8 * 8 * alignedCinKK + GID1 % 8 / 4 * 16);  \
        uint end_alignedCinKK = src_addr.s0 + 32 * alignedCinKK; \
        for (; src_addr.s0 < end_alignedCinKK; src_addr += (uint2)(256, 64)) { \
            DATA_T16 input16_0; \
            DATA_T16 weight16_0; \
            input16_0 = vload16(0, input + src_addr.s0); \
            weight16_0 = vload16(0, weight + src_addr.s1); \
            MAD_I16W16O4(input16_0, weight16_0, out_0, out_1, out_2, out_3) \
            input16_0 = vload16(8, input + src_addr.s0); \
            weight16_0 = vload16(2, weight + src_addr.s1); \
            MAD_I16W16O4(input16_0, weight16_0, out_0, out_1, out_2, out_3) \
        } \
        int outputIndex = get_global_id(0) * topChannel * topHW + GID1 * topHW + GID2; \
        DATA_T4 bia = vload4(0, bias + GID1); \
        out8_0.s048C = out_0 + bia; \
        out8_0.s159D = out_1 + bia; \
        out8_0.s26AE = out_2 + bia; \
        out8_0.s37BF = out_3 + bia; \
        if (GID1 + 3 < topChannel) { \
            if (GID2 + 3 < topHW) { \
                vstore4(ACT_VEC_F(DATA_T4, out8_0.s0123), 0, output + outputIndex); \
                vstore4(ACT_VEC_F(DATA_T4, out8_0.s4567), 0, output + outputIndex + topHW); \
                vstore4(ACT_VEC_F(DATA_T4, out8_0.s89AB), 0, output + outputIndex + 2 * topHW); \
                vstore4(ACT_VEC_F(DATA_T4, out8_0.sCDEF), 0, output + outputIndex + 3 * topHW); \
            } else if (GID2 + 2 < topHW) { \
                vstore3(ACT_VEC_F(DATA_T3, out8_0.s012), 0, output + outputIndex); \
                vstore3(ACT_VEC_F(DATA_T3, out8_0.s456), 0, output + outputIndex + topHW); \
                vstore3(ACT_VEC_F(DATA_T3, out8_0.s89A), 0, output + outputIndex + 2 * topHW); \
                vstore3(ACT_VEC_F(DATA_T3, out8_0.sCDE), 0, output + outputIndex + 3 * topHW); \
            } else if (GID2 + 1 < topHW) { \
                vstore2(ACT_VEC_F(DATA_T2, out8_0.s01), 0, output + outputIndex); \
                vstore2(ACT_VEC_F(DATA_T2, out8_0.s45), 0, output + outputIndex + topHW); \
                vstore2(ACT_VEC_F(DATA_T2, out8_0.s89), 0, output + outputIndex + 2 * topHW); \
                vstore2(ACT_VEC_F(DATA_T2, out8_0.sCD), 0, output + outputIndex + 3 * topHW); \
            } else if (GID2 < topHW) { \
                output[outputIndex] = ACT_VEC_F(DATA_T, out8_0.s0); \
                output[outputIndex + topHW] = ACT_VEC_F(DATA_T, out8_0.s4); \
                output[outputIndex + 2 * topHW] = ACT_VEC_F(DATA_T, out8_0.s8); \
                output[outputIndex + 3 * topHW] = ACT_VEC_F(DATA_T, out8_0.sC); \
            } \
        } else if (GID1 + 2 < topChannel) { \
            if (GID2 + 3 < topHW) { \
                vstore4(ACT_VEC_F(DATA_T4, out8_0.s0123), 0, output + outputIndex); \
                vstore4(ACT_VEC_F(DATA_T4, out8_0.s4567), 0, output + outputIndex + topHW); \
                vstore4(ACT_VEC_F(DATA_T4, out8_0.s89AB), 0, output + outputIndex + 2 * topHW); \
            } else if (GID2 + 2 < topHW) { \
                vstore3(ACT_VEC_F(DATA_T3, out8_0.s012), 0, output + outputIndex); \
                vstore3(ACT_VEC_F(DATA_T3, out8_0.s456), 0, output + outputIndex + topHW); \
                vstore3(ACT_VEC_F(DATA_T3, out8_0.s89A), 0, output + outputIndex + 2 * topHW); \
            } else if (GID2 + 1 < topHW) { \
                vstore2(ACT_VEC_F(DATA_T2, out8_0.s01), 0, output + outputIndex); \
                vstore2(ACT_VEC_F(DATA_T2, out8_0.s45), 0, output + outputIndex + topHW); \
                vstore2(ACT_VEC_F(DATA_T2, out8_0.s89), 0, output + outputIndex + 2 * topHW); \
            } else if (GID2 < topHW) { \
                output[outputIndex] = ACT_VEC_F(DATA_T, out8_0.s0); \
                output[outputIndex + topHW] = ACT_VEC_F(DATA_T, out8_0.s4); \
                output[outputIndex + 2 * topHW] = ACT_VEC_F(DATA_T, out8_0.s8); \
            } \
        } else if (GID1 + 1 < topChannel) { \
            if (GID2 + 3 < topHW) { \
                vstore4(ACT_VEC_F(DATA_T4, out8_0.s0123), 0, output + outputIndex); \
                vstore4(ACT_VEC_F(DATA_T4, out8_0.s4567), 0, output + outputIndex + topHW); \
            } else if (GID2 + 2 < topHW) { \
                vstore3(ACT_VEC_F(DATA_T3, out8_0.s012), 0, output + outputIndex); \
                vstore3(ACT_VEC_F(DATA_T3, out8_0.s456), 0, output + outputIndex + topHW); \
            } else if (GID2 + 1 < topHW) { \
                vstore2(ACT_VEC_F(DATA_T2, out8_0.s01), 0, output + outputIndex); \
                vstore2(ACT_VEC_F(DATA_T2, out8_0.s45), 0, output + outputIndex + topHW); \
            } else if (GID2 < topHW) { \
                output[outputIndex] = ACT_VEC_F(DATA_T, out8_0.s0); \
                output[outputIndex + topHW] = ACT_VEC_F(DATA_T, out8_0.s4); \
            } \
        } else if (GID1 < topChannel) { \
            if (GID2 + 3 < topHW) { \
                vstore4(ACT_VEC_F(DATA_T4, out8_0.s0123), 0, output + outputIndex); \
            } else if (GID2 + 2 < topHW) { \
                vstore3(ACT_VEC_F(DATA_T3, out8_0.s012), 0, output + outputIndex); \
            } else if (GID2 + 1 < topHW) { \
                vstore2(ACT_VEC_F(DATA_T2, out8_0.s01), 0, output + outputIndex); \
            } else if (GID2 < topHW) { \
                output[outputIndex] = ACT_VEC_F(DATA_T, out8_0.s0); \
            } \
        } \
    }

#define GEMMVALHALL4X4NHWC(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep) \
    int GID2 = get_global_id(2) * 4; \
    int GID1 = get_global_id(1) * 4; \
    if (GID2 < topHW) { \
        DATA_T16 out8_0 = 0.0f; \
        DATA_T4 out_0 = 0.0f; \
        DATA_T4 out_1 = 0.0f; \
        DATA_T4 out_2 = 0.0f; \
        DATA_T4 out_3 = 0.0f; \
        uint2 src_addr = \
            (uint2)(get_global_id(0) * bottomStep + (GID2 / 32) * 32 * alignedCinKK + \
                        (GID2 % 32 / 4) * 16, \
                    GID1 / 8 * 8 * alignedCinKK + GID1 % 8 / 4 * 16);  \
        uint end_alignedCinKK = src_addr.s0 + 32 * alignedCinKK; \
        for (; src_addr.s0 < end_alignedCinKK; src_addr += (uint2)(256, 64)) { \
            DATA_T16 input16_0; \
            DATA_T16 weight16_0; \
            input16_0 = vload16(0, input + src_addr.s0); \
            weight16_0 = vload16(0, weight + src_addr.s1); \
            MAD_I16W16O4(input16_0, weight16_0, out_0, out_1, out_2, out_3) \
            input16_0 = vload16(8, input + src_addr.s0); \
            weight16_0 = vload16(2, weight + src_addr.s1); \
            MAD_I16W16O4(input16_0, weight16_0, out_0, out_1, out_2, out_3) \
        } \
        int outputIndex = get_global_id(0) * topChannel * topHW + GID2 * topChannel + GID1; \
        DATA_T4 bia = vload4(0, bias + GID1); \
        out8_0.s048C = out_0 + bia; \
        out8_0.s159D = out_1 + bia; \
        out8_0.s26AE = out_2 + bia; \
        out8_0.s37BF = out_3 + bia; \
        out8_0 = ACT_VEC_F(DATA_T16, out8_0); \
        if (GID2 + 3 < topHW) { \
            if (GID1 + 3 < topChannel) { \
                vstore4(out8_0.s048C, 0, output + outputIndex); \
                vstore4(out8_0.s159D, 0, output + outputIndex + topChannel); \
                vstore4(out8_0.s26AE, 0, output + outputIndex + 2 * topChannel); \
                vstore4(out8_0.s37BF, 0, output + outputIndex + 3 * topChannel); \
            } else if (GID1 + 2 < topChannel) { \
                vstore3(out8_0.s048, 0, output + outputIndex); \
                vstore3(out8_0.s159, 0, output + outputIndex + topChannel); \
                vstore3(out8_0.s26A, 0, output + outputIndex + 2 * topChannel); \
                vstore3(out8_0.s37B, 0, output + outputIndex + 3 * topChannel); \
            } else if (GID1 + 1 < topChannel) { \
                vstore2(out8_0.s04, 0, output + outputIndex); \
                vstore2(out8_0.s15, 0, output + outputIndex + topChannel); \
                vstore2(out8_0.s26, 0, output + outputIndex + 2 * topChannel); \
                vstore2(out8_0.s37, 0, output + outputIndex + 3 * topChannel); \
            } else if (GID1 < topChannel) { \
                output[outputIndex] = out8_0.s0; \
                output[outputIndex + topChannel] = out8_0.s1; \
                output[outputIndex + 2 * topChannel] = out8_0.s2; \
                output[outputIndex + 3 * topChannel] = out8_0.s3; \
            } \
        } else if (GID2 + 2 < topHW) { \
            if (GID1 + 3 < topChannel) { \
                vstore4(out8_0.s048C, 0, output + outputIndex); \
                vstore4(out8_0.s159D, 0, output + outputIndex + topChannel); \
                vstore4(out8_0.s26AE, 0, output + outputIndex + 2 * topChannel); \
            } else if (GID1 + 2 < topChannel) { \
                vstore3(out8_0.s048, 0, output + outputIndex); \
                vstore3(out8_0.s159, 0, output + outputIndex + topChannel); \
                vstore3(out8_0.s26A, 0, output + outputIndex + 2 * topChannel); \
            } else if (GID1 + 1 < topChannel) { \
                vstore2(out8_0.s04, 0, output + outputIndex); \
                vstore2(out8_0.s15, 0, output + outputIndex + topChannel); \
                vstore2(out8_0.s26, 0, output + outputIndex + 2 * topChannel); \
            } else if (GID1 < topChannel) { \
                output[outputIndex] = out8_0.s0; \
                output[outputIndex + topChannel] = out8_0.s1; \
                output[outputIndex + 2 * topChannel] = out8_0.s2; \
            } \
        } else if (GID2 + 1 < topHW) { \
            if (GID1 + 3 < topChannel) { \
                vstore4(out8_0.s048C, 0, output + outputIndex); \
                vstore4(out8_0.s159D, 0, output + outputIndex + topChannel); \
            } else if (GID1 + 2 < topChannel) { \
                vstore3(out8_0.s048, 0, output + outputIndex); \
                vstore3(out8_0.s159, 0, output + outputIndex + topChannel); \
            } else if (GID1 + 1 < topChannel) { \
                vstore2(out8_0.s04, 0, output + outputIndex); \
                vstore2(out8_0.s15, 0, output + outputIndex + topChannel); \
            } else if (GID1 < topChannel) { \
                output[outputIndex] = out8_0.s0; \
                output[outputIndex + topChannel] = out8_0.s1; \
            } \
        } else if (GID2 < topHW) { \
            if (GID1 + 3 < topChannel) { \
                vstore4(out8_0.s048C, 0, output + outputIndex); \
            } else if (GID1 + 2 < topChannel) { \
                vstore3(out8_0.s048, 0, output + outputIndex); \
            } else if (GID1 + 1 < topChannel) { \
                vstore2(out8_0.s04, 0, output + outputIndex); \
            } else if (GID1 < topChannel) { \
                output[outputIndex] = out8_0.s0; \
            } \
        } \
    }

#define GEMMMAKALU(input, weight, bias, output, alignedCinKK, topChannel, topHW, splitTopHW, bottomStep, \
                   inputOffset, outputOffset) \
    int GID2 = get_global_id(2) * 4; \
    int GID1 = get_global_id(1) * 4; \
    if (GID2 < splitTopHW) { \
        DATA_T16 out8_0 = 0.0f; \
        DATA_T4 out_0 = 0.0f; \
        DATA_T4 out_1 = 0.0f; \
        DATA_T4 out_2 = 0.0f; \
        DATA_T4 out_3 = 0.0f; \
        uint2 src_addr = \
            (uint2)(get_global_id(0) * bottomStep + (GID2 / 48) * 48 * alignedCinKK + \
                        (GID2 % 48 / 4) * 16 + inputOffset, \
                    GID1 / 8 * 8 * alignedCinKK + GID1 % 8 / 4 * 16);  \
        uint end_alignedCinKK = src_addr.s0 + 48 * alignedCinKK; \
        for (; src_addr.s0 < end_alignedCinKK; src_addr += (uint2)(192, 32)) { \
            DATA_T16 input16_0; \
            DATA_T16 weight16_0; \
            input16_0 = vload16(0, input + src_addr.s0); \
            weight16_0 = vload16(0, weight + src_addr.s1); \
            MAD_I16W16O4(input16_0, weight16_0, out_0, out_1, out_2, out_3) \
        } \
        int outputIndex = \
            get_global_id(0) * topChannel * topHW + GID1 * topHW + GID2 + outputOffset; \
        if (GID1 + 3 < topChannel) { \
            if (GID2 + 3 < splitTopHW) { \
                DATA_T4 bia = vload4(0, bias + GID1); \
                out8_0.s048C = out_0 + bia; \
                out8_0.s159D = out_1 + bia; \
                out8_0.s26AE = out_2 + bia; \
                out8_0.s37BF = out_3 + bia; \
                vstore4(ACT_VEC_F(DATA_T4, out8_0.s0123), 0, output + outputIndex); \
                vstore4(ACT_VEC_F(DATA_T4, out8_0.s4567), 0, output + outputIndex + topHW); \
                vstore4(ACT_VEC_F(DATA_T4, out8_0.s89AB), 0, output + outputIndex + 2 * topHW); \
                vstore4(ACT_VEC_F(DATA_T4, out8_0.sCDEF), 0, output + outputIndex + 3 * topHW); \
            } else if (GID2 + 2 < splitTopHW) { \
                DATA_T4 bia = vload4(0, bias + GID1); \
                out8_0.s048C = out_0 + bia; \
                out8_0.s159D = out_1 + bia; \
                out8_0.s26AE = out_2 + bia; \
                vstore3(ACT_VEC_F(DATA_T3, out8_0.s012), 0, output + outputIndex); \
                vstore3(ACT_VEC_F(DATA_T3, out8_0.s456), 0, output + outputIndex + topHW); \
                vstore3(ACT_VEC_F(DATA_T3, out8_0.s89A), 0, output + outputIndex + 2 * topHW); \
                vstore3(ACT_VEC_F(DATA_T3, out8_0.sCDE), 0, output + outputIndex + 3 * topHW); \
            } else if (GID2 + 1 < splitTopHW) { \
                DATA_T4 bia = vload4(0, bias + GID1); \
                out8_0.s048C = out_0 + bia; \
                out8_0.s159D = out_1 + bia; \
                vstore2(ACT_VEC_F(DATA_T2, out8_0.s01), 0, output + outputIndex); \
                vstore2(ACT_VEC_F(DATA_T2, out8_0.s45), 0, output + outputIndex + topHW); \
                vstore2(ACT_VEC_F(DATA_T2, out8_0.s89), 0, output + outputIndex + 2 * topHW); \
                vstore2(ACT_VEC_F(DATA_T2, out8_0.sCD), 0, output + outputIndex + 3 * topHW); \
            } else if (GID2 < splitTopHW) { \
                DATA_T4 bia = vload4(0, bias + GID1); \
                out8_0.s048C = out_0 + bia; \
                output[outputIndex] = ACT_VEC_F(DATA_T, out8_0.s0); \
                output[outputIndex + topHW] = ACT_VEC_F(DATA_T, out8_0.s4); \
                output[outputIndex + 2 * topHW] = ACT_VEC_F(DATA_T, out8_0.s8); \
                output[outputIndex + 3 * topHW] = ACT_VEC_F(DATA_T, out8_0.sC); \
            } \
        } else if (GID1 + 2 < topChannel) { \
            if (GID2 + 3 < splitTopHW) { \
                DATA_T4 bia = vload4(0, bias + GID1); \
                out8_0.s048C = out_0 + bia; \
                out8_0.s159D = out_1 + bia; \
                out8_0.s26AE = out_2 + bia; \
                out8_0.s37BF = out_3 + bia; \
                vstore4(ACT_VEC_F(DATA_T4, out8_0.s0123), 0, output + outputIndex); \
                vstore4(ACT_VEC_F(DATA_T4, out8_0.s4567), 0, output + outputIndex + topHW); \
                vstore4(ACT_VEC_F(DATA_T4, out8_0.s89AB), 0, output + outputIndex + 2 * topHW); \
            } else if (GID2 + 2 < splitTopHW) { \
                DATA_T4 bia = vload4(0, bias + GID1); \
                out8_0.s048C = out_0 + bia; \
                out8_0.s159D = out_1 + bia; \
                out8_0.s26AE = out_2 + bia; \
                vstore3(ACT_VEC_F(DATA_T3, out8_0.s012), 0, output + outputIndex); \
                vstore3(ACT_VEC_F(DATA_T3, out8_0.s456), 0, output + outputIndex + topHW); \
                vstore3(ACT_VEC_F(DATA_T3, out8_0.s89A), 0, output + outputIndex + 2 * topHW); \
            } else if (GID2 + 1 < splitTopHW) { \
                DATA_T4 bia = vload4(0, bias + GID1); \
                out8_0.s048C = out_0 + bia; \
                out8_0.s159D = out_1 + bia; \
                vstore2(ACT_VEC_F(DATA_T2, out8_0.s01), 0, output + outputIndex); \
                vstore2(ACT_VEC_F(DATA_T2, out8_0.s45), 0, output + outputIndex + topHW); \
                vstore2(ACT_VEC_F(DATA_T2, out8_0.s89), 0, output + outputIndex + 2 * topHW); \
            } else if (GID2 < splitTopHW) { \
                DATA_T4 bia = vload4(0, bias + GID1); \
                out8_0.s048C = out_0 + bia; \
                output[outputIndex] = ACT_VEC_F(DATA_T, out8_0.s0); \
                output[outputIndex + topHW] = ACT_VEC_F(DATA_T, out8_0.s4); \
                output[outputIndex + 2 * topHW] = ACT_VEC_F(DATA_T, out8_0.s8); \
            } \
        } else if (GID1 + 1 < topChannel) { \
            if (GID2 + 3 < splitTopHW) { \
                DATA_T4 bia = vload4(0, bias + GID1); \
                out8_0.s048C = out_0 + bia; \
                out8_0.s159D = out_1 + bia; \
                out8_0.s26AE = out_2 + bia; \
                out8_0.s37BF = out_3 + bia; \
                vstore4(ACT_VEC_F(DATA_T4, out8_0.s0123), 0, output + outputIndex); \
                vstore4(ACT_VEC_F(DATA_T4, out8_0.s4567), 0, output + outputIndex + topHW); \
            } else if (GID2 + 2 < splitTopHW) { \
                DATA_T4 bia = vload4(0, bias + GID1); \
                out8_0.s048C = out_0 + bia; \
                out8_0.s159D = out_1 + bia; \
                out8_0.s26AE = out_2 + bia; \
                vstore3(ACT_VEC_F(DATA_T3, out8_0.s012), 0, output + outputIndex); \
                vstore3(ACT_VEC_F(DATA_T3, out8_0.s456), 0, output + outputIndex + topHW); \
            } else if (GID2 + 1 < splitTopHW) { \
                DATA_T4 bia = vload4(0, bias + GID1); \
                out8_0.s048C = out_0 + bia; \
                out8_0.s159D = out_1 + bia; \
                vstore2(ACT_VEC_F(DATA_T2, out8_0.s01), 0, output + outputIndex); \
                vstore2(ACT_VEC_F(DATA_T2, out8_0.s45), 0, output + outputIndex + topHW); \
            } else if (GID2 < splitTopHW) { \
                DATA_T4 bia = vload4(0, bias + GID1); \
                out8_0.s048C = out_0 + bia; \
                output[outputIndex] = ACT_VEC_F(DATA_T, out8_0.s0); \
                output[outputIndex + topHW] = ACT_VEC_F(DATA_T, out8_0.s4); \
            } \
        } else if (GID1 < topChannel) { \
            if (GID2 + 3 < splitTopHW) { \
                DATA_T4 bia = vload4(0, bias + GID1); \
                out8_0.s048C = out_0 + bia; \
                out8_0.s159D = out_1 + bia; \
                out8_0.s26AE = out_2 + bia; \
                out8_0.s37BF = out_3 + bia; \
                vstore4(ACT_VEC_F(DATA_T4, out8_0.s0123), 0, output + outputIndex); \
            } else if (GID2 + 2 < splitTopHW) { \
                DATA_T4 bia = vload4(0, bias + GID1); \
                out8_0.s048C = out_0 + bia; \
                out8_0.s159D = out_1 + bia; \
                out8_0.s26AE = out_2 + bia; \
                vstore3(ACT_VEC_F(DATA_T3, out8_0.s012), 0, output + outputIndex); \
            } else if (GID2 + 1 < splitTopHW) { \
                DATA_T4 bia = vload4(0, bias + GID1); \
                out8_0.s048C = out_0 + bia; \
                out8_0.s159D = out_1 + bia; \
                vstore2(ACT_VEC_F(DATA_T2, out8_0.s01), 0, output + outputIndex); \
            } else if (GID2 < splitTopHW) { \
                DATA_T4 bia = vload4(0, bias + GID1); \
                out8_0.s048C = out_0 + bia; \
                output[outputIndex] = ACT_VEC_F(DATA_T, out8_0.s0); \
            } \
        } \
    }

#define ALIGNWEIGHT_1X8(src, dst, srcWidth, dstWidth) \
    uint globalID0 = get_global_id(0); \
    uint globalID1 = get_global_id(1); \
    uint srcIndex = globalID0 * srcWidth + globalID1; \
    uint dstIndex = globalID0 / 8 * 8 * dstWidth + globalID1 * 8 + globalID0 % 8; \
    if (globalID1 < srcWidth) { \
        dst[dstIndex] = src[srcIndex]; \
    }

#define ALIGNWEIGHT_1X4(src, dst, srcWidth, dstWidth) \
    uint globalID0 = get_global_id(0); \
    uint globalID1 = get_global_id(1) * 8; \
    uint srcIndex = globalID0 * srcWidth + globalID1; \
    uint dstIndex = globalID0 / 4 * 4 * dstWidth + globalID1 * 4 + (globalID0 % 4) * 8; \
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

#define ALIGN_WEIGHT_4_ROW_1_COL(src, dst, srcHeight, srcWidth, dstWidth) \
    uint dstRow = get_global_id(0); \
    uint dstCol = get_global_id(1); \
    uint srcRow = dstRow * 8 + dstCol % 32 / 16 * 4 + dstCol % 4; \
    uint srcCol = dstCol / 32 * 4 + dstCol % 16 / 4; \
    uint dstIndex = dstRow * dstWidth + dstCol; \
    if (srcRow < srcHeight && srcCol < srcWidth) { \
        uint srcIndex = srcRow * srcWidth + srcCol; \
        dst[dstIndex] = src[srcIndex]; \
    } else { \
        dst[dstIndex] = 0.0f; \
    }

// FP16 kernels
#define DATA_T half
#define DATA_T2 half2
#define DATA_T3 half3
#define DATA_T4 half4
#define DATA_T8 half8
#define DATA_T16 half16
// kernels without activation func
#define ACT_VEC_F(VEC_T, x) x
ADD_SINGLE_KERNEL(gemmBlock2x8_FP16, (__global const DATA_T *input,
                                      __global const DATA_T *weight,
                                      __global const DATA_T *bias,
                                      __global DATA_T *output,
                                      uint alignedCinKK,
                                      uint topChannel,
                                      uint topHW,
                                      uint bottomStep) {
    GEMMBLOCK2X8(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(gemmBlock2x8_scalar_FP16, (__global const DATA_T *input,
                                             __global const DATA_T *weight,
                                             __global const DATA_T *bias,
                                             __global DATA_T *output,
                                             uint alignedCinKK,
                                             uint topChannel,
                                             uint topHW,
                                             uint bottomStep) {
    GEMMBLOCK2X8_SCALAR_FP16(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(gemmBlock2x4_FP16, (__global const DATA_T *input,
                                      __global const DATA_T *weight,
                                      __global const DATA_T *bias,
                                      __global DATA_T *output,
                                      uint alignedCinKK,
                                      uint topChannel,
                                      uint topHW,
                                      uint bottomStep) {
    GEMMBLOCK2X4(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(gemmBlock2x4_scalar_FP16, (__global const DATA_T *input,
                                             __global const DATA_T *weight,
                                             __global const DATA_T *bias,
                                             __global DATA_T *output,
                                             uint alignedCinKK,
                                             uint topChannel,
                                             uint topHW,
                                             uint bottomStep) {
    GEMMBLOCK2X4_SCALAR_FP16(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(gemmValhall4x4_FP16, (__global const DATA_T *input,
                                        __global const DATA_T *weight,
                                        __global const DATA_T *bias,
                                        __global DATA_T *output,
                                        uint alignedCinKK,
                                        uint topChannel,
                                        uint topHW,
                                        uint bottomStep) {
    GEMMVALHALL4X4(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(gemmValhall4x4NHWC_FP16, (__global const DATA_T *input,
                                            __global const DATA_T *weight,
                                            __global const DATA_T *bias,
                                            __global DATA_T *output,
                                            uint alignedCinKK,
                                            uint topChannel,
                                            uint topHW,
                                            uint bottomStep) {
    GEMMVALHALL4X4NHWC(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(gemmMakalu_FP16, (__global const DATA_T *input,
                                    __global const DATA_T *weight,
                                    __global const DATA_T *bias,
                                    __global DATA_T *output,
                                    uint alignedCinKK,
                                    uint topChannel,
                                    uint topHW,
                                    uint splitTopHW,
                                    uint bottomStep,
                                    uint inputOffset,
                                    uint outputOffset) {
    GEMMMAKALU(input, weight, bias, output, alignedCinKK, topChannel, topHW, splitTopHW, bottomStep, \
               inputOffset, outputOffset)
})
#undef ACT_VEC_F  // NONE

// apply RELU to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELUgemmBlock2x8_FP16, (__global const DATA_T *input,
                                          __global const DATA_T *weight,
                                          __global const DATA_T *bias,
                                          __global DATA_T *output,
                                          uint alignedCinKK,
                                          uint topChannel,
                                          uint topHW,
                                          uint bottomStep) {
    GEMMBLOCK2X8(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(RELUgemmBlock2x8_scalar_FP16, (__global const DATA_T *input,
                                                 __global const DATA_T *weight,
                                                 __global const DATA_T *bias,
                                                 __global DATA_T *output,
                                                 uint alignedCinKK,
                                                 uint topChannel,
                                                 uint topHW,
                                                 uint bottomStep) {
    GEMMBLOCK2X8_SCALAR_FP16(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(RELUgemmBlock2x4_FP16, (__global const DATA_T *input,
                                          __global const DATA_T *weight,
                                          __global const DATA_T *bias,
                                          __global DATA_T *output,
                                          uint alignedCinKK,
                                          uint topChannel,
                                          uint topHW,
                                          uint bottomStep) {
    GEMMBLOCK2X4(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(RELUgemmBlock2x4_scalar_FP16, (__global const DATA_T *input,
                                                 __global const DATA_T *weight,
                                                 __global const DATA_T *bias,
                                                 __global DATA_T *output,
                                                 uint alignedCinKK,
                                                 uint topChannel,
                                                 uint topHW,
                                                 uint bottomStep) {
    GEMMBLOCK2X4_SCALAR_FP16(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(RELUgemmValhall4x4_FP16, (__global const DATA_T *input,
                                            __global const DATA_T *weight,
                                            __global const DATA_T *bias,
                                            __global DATA_T *output,
                                            uint alignedCinKK,
                                            uint topChannel,
                                            uint topHW,
                                            uint bottomStep) {
    GEMMVALHALL4X4(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(RELUgemmValhall4x4NHWC_FP16, (__global const DATA_T *input,
                                                __global const DATA_T *weight,
                                                __global const DATA_T *bias,
                                                __global DATA_T *output,
                                                uint alignedCinKK,
                                                uint topChannel,
                                                uint topHW,
                                                uint bottomStep) {
    GEMMVALHALL4X4NHWC(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(RELUgemmMakalu_FP16, (__global const DATA_T *input,
                                        __global const DATA_T *weight,
                                        __global const DATA_T *bias,
                                        __global DATA_T *output,
                                        uint alignedCinKK,
                                        uint topChannel,
                                        uint topHW,
                                        uint splitTopHW,
                                        uint bottomStep,
                                        uint inputOffset,
                                        uint outputOffset) {
    GEMMMAKALU(input, weight, bias, output, alignedCinKK, topChannel, topHW, splitTopHW, bottomStep, \
               inputOffset, outputOffset)
})
#undef ACT_VEC_F  // RELU

// apply RELU6 to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU6_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELU6gemmBlock2x8_FP16, (__global const DATA_T *input,
                                           __global const DATA_T *weight,
                                           __global const DATA_T *bias,
                                           __global DATA_T *output,
                                           uint alignedCinKK,
                                           uint topChannel,
                                           uint topHW,
                                           uint bottomStep) {
    GEMMBLOCK2X8(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(RELU6gemmBlock2x8_scalar_FP16, (__global const DATA_T *input,
                                                  __global const DATA_T *weight,
                                                  __global const DATA_T *bias,
                                                  __global DATA_T *output,
                                                  uint alignedCinKK,
                                                  uint topChannel,
                                                  uint topHW,
                                                  uint bottomStep) {
    GEMMBLOCK2X8_SCALAR_FP16(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(RELU6gemmBlock2x4_FP16, (__global const DATA_T *input,
                                           __global const DATA_T *weight,
                                           __global const DATA_T *bias,
                                           __global DATA_T *output,
                                           uint alignedCinKK,
                                           uint topChannel,
                                           uint topHW,
                                           uint bottomStep) {
    GEMMBLOCK2X4(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(RELU6gemmBlock2x4_scalar_FP16, (__global const DATA_T *input,
                                                  __global const DATA_T *weight,
                                                  __global const DATA_T *bias,
                                                  __global DATA_T *output,
                                                  uint alignedCinKK,
                                                  uint topChannel,
                                                  uint topHW,
                                                  uint bottomStep) {
    GEMMBLOCK2X4_SCALAR_FP16(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(RELU6gemmValhall4x4_FP16, (__global const DATA_T *input,
                                             __global const DATA_T *weight,
                                             __global const DATA_T *bias,
                                             __global DATA_T *output,
                                             uint alignedCinKK,
                                             uint topChannel,
                                             uint topHW,
                                             uint bottomStep) {
    GEMMVALHALL4X4(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(RELU6gemmValhall4x4NHWC_FP16, (__global const DATA_T *input,
                                                 __global const DATA_T *weight,
                                                 __global const DATA_T *bias,
                                                 __global DATA_T *output,
                                                 uint alignedCinKK,
                                                 uint topChannel,
                                                 uint topHW,
                                                 uint bottomStep) {
    GEMMVALHALL4X4NHWC(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(RELU6gemmMakalu_FP16, (__global const DATA_T *input,
                                         __global const DATA_T *weight,
                                         __global const DATA_T *bias,
                                         __global DATA_T *output,
                                         uint alignedCinKK,
                                         uint topChannel,
                                         uint topHW,
                                         uint splitTopHW,
                                         uint bottomStep,
                                         uint inputOffset,
                                         uint outputOffset) {
    GEMMMAKALU(input, weight, bias, output, alignedCinKK, topChannel, topHW, splitTopHW, bottomStep, \
               inputOffset, outputOffset)
})
#undef ACT_VEC_F  // RELU6

ADD_SINGLE_KERNEL(alignWeight_1x8_FP16, (__global const DATA_T *src,
                                       __global DATA_T *dst,
                                       uint srcWidth,
                                       uint dstWidth) {
        ALIGNWEIGHT_1X8(src, dst, srcWidth, dstWidth)
})

ADD_SINGLE_KERNEL(alignWeight_1x4_FP16, (__global const DATA_T *src,
                                       __global DATA_T *dst,
                                       uint srcWidth,
                                       uint dstWidth) {
        ALIGNWEIGHT_1X4(src, dst, srcWidth, dstWidth)
})

ADD_SINGLE_KERNEL(align_weight_4_row_1_col_FP16, (__global const DATA_T *src,
                                                __global DATA_T *dst,
                                                uint srcHeight,
                                                uint srcWidth,
                                                uint dstWidth) {
        ALIGN_WEIGHT_4_ROW_1_COL(src, dst, srcHeight, srcWidth, dstWidth)
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
// kernels without activation func
#define ACT_VEC_F(VEC_T, x) x
ADD_SINGLE_KERNEL(gemmBlock2x8_FP32, (__global const DATA_T *input,
                                      __global const DATA_T *weight,
                                      __global const DATA_T *bias,
                                      __global DATA_T *output,
                                      uint alignedCinKK,
                                      uint topChannel,
                                      uint topHW,
                                      uint bottomStep) {
    GEMMBLOCK2X8(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(gemmBlock2x8_scalar_FP32, (__global const DATA_T *input,
                                             __global const DATA_T *weight,
                                             __global const DATA_T *bias,
                                             __global DATA_T *output,
                                             uint alignedCinKK,
                                             uint topChannel,
                                             uint topHW,
                                             uint bottomStep) {
    GEMMBLOCK2X8_SCALAR_FP32(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(gemmBlock2x4_FP32, (__global const DATA_T *input,
                                      __global const DATA_T *weight,
                                      __global const DATA_T *bias,
                                      __global DATA_T *output,
                                      uint alignedCinKK,
                                      uint topChannel,
                                      uint topHW,
                                      uint bottomStep) {
    GEMMBLOCK2X4(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(gemmBlock2x4_scalar_FP32, (__global const DATA_T *input,
                                             __global const DATA_T *weight,
                                             __global const DATA_T *bias,
                                             __global DATA_T *output,
                                             uint alignedCinKK,
                                             uint topChannel,
                                             uint topHW,
                                             uint bottomStep) {
    GEMMBLOCK2X4_SCALAR_FP32(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(gemmValhall4x4_FP32, (__global const DATA_T *input,
                                        __global const DATA_T *weight,
                                        __global const DATA_T *bias,
                                        __global DATA_T *output,
                                        uint alignedCinKK,
                                        uint topChannel,
                                        uint topHW,
                                        uint bottomStep) {
    GEMMVALHALL4X4(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(gemmValhall4x4NHWC_FP32, (__global const DATA_T *input,
                                            __global const DATA_T *weight,
                                            __global const DATA_T *bias,
                                            __global DATA_T *output,
                                            uint alignedCinKK,
                                            uint topChannel,
                                            uint topHW,
                                            uint bottomStep) {
    GEMMVALHALL4X4NHWC(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(gemmMakalu_FP32, (__global const DATA_T *input,
                                    __global const DATA_T *weight,
                                    __global const DATA_T *bias,
                                    __global DATA_T *output,
                                    uint alignedCinKK,
                                    uint topChannel,
                                    uint topHW,
                                    uint splitTopHW,
                                    uint bottomStep,
                                    uint inputOffset,
                                    uint outputOffset) {
    GEMMMAKALU(input, weight, bias, output, alignedCinKK, topChannel, topHW, splitTopHW, bottomStep, \
               inputOffset, outputOffset)
})
#undef ACT_VEC_F  // NONE

// apply RELU to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELUgemmBlock2x8_FP32, (__global const DATA_T *input,
                                          __global const DATA_T *weight,
                                          __global const DATA_T *bias,
                                          __global DATA_T *output,
                                          uint alignedCinKK,
                                          uint topChannel,
                                          uint topHW,
                                          uint bottomStep) {
    GEMMBLOCK2X8(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(RELUgemmBlock2x8_scalar_FP32, (__global const DATA_T *input,
                                                 __global const DATA_T *weight,
                                                 __global const DATA_T *bias,
                                                 __global DATA_T *output,
                                                 uint alignedCinKK,
                                                 uint topChannel,
                                                 uint topHW,
                                                 uint bottomStep) {
    GEMMBLOCK2X8_SCALAR_FP32(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(RELUgemmBlock2x4_FP32, (__global const DATA_T *input,
                                          __global const DATA_T *weight,
                                          __global const DATA_T *bias,
                                          __global DATA_T *output,
                                          uint alignedCinKK,
                                          uint topChannel,
                                          uint topHW,
                                          uint bottomStep) {
    GEMMBLOCK2X4(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(RELUgemmBlock2x4_scalar_FP32, (__global const DATA_T *input,
                                                 __global const DATA_T *weight,
                                                 __global const DATA_T *bias,
                                                 __global DATA_T *output,
                                                 uint alignedCinKK,
                                                 uint topChannel,
                                                 uint topHW,
                                                 uint bottomStep) {
    GEMMBLOCK2X4_SCALAR_FP32(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(RELUgemmValhall4x4_FP32, (__global const DATA_T *input,
                                            __global const DATA_T *weight,
                                            __global const DATA_T *bias,
                                            __global DATA_T *output,
                                            uint alignedCinKK,
                                            uint topChannel,
                                            uint topHW,
                                            uint bottomStep) {
    GEMMVALHALL4X4(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(RELUgemmValhall4x4NHWC_FP32, (__global const DATA_T *input,
                                                __global const DATA_T *weight,
                                                __global const DATA_T *bias,
                                                __global DATA_T *output,
                                                uint alignedCinKK,
                                                uint topChannel,
                                                uint topHW,
                                                uint bottomStep) {
    GEMMVALHALL4X4NHWC(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(RELUgemmMakalu_FP32, (__global const DATA_T *input,
                                        __global const DATA_T *weight,
                                        __global const DATA_T *bias,
                                        __global DATA_T *output,
                                        uint alignedCinKK,
                                        uint topChannel,
                                        uint topHW,
                                        uint splitTopHW,
                                        uint bottomStep,
                                        uint inputOffset,
                                        uint outputOffset) {
    GEMMMAKALU(input, weight, bias, output, alignedCinKK, topChannel, topHW, splitTopHW, bottomStep, \
               inputOffset, outputOffset)
})
#undef ACT_VEC_F  // RELU

// apply RELU6 to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU6_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELU6gemmBlock2x8_FP32, (__global const DATA_T *input,
                                           __global const DATA_T *weight,
                                           __global const DATA_T *bias,
                                           __global DATA_T *output,
                                           uint alignedCinKK,
                                           uint topChannel,
                                           uint topHW,
                                           uint bottomStep) {
    GEMMBLOCK2X8(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(RELU6gemmBlock2x8_scalar_FP32, (__global const DATA_T *input,
                                                  __global const DATA_T *weight,
                                                  __global const DATA_T *bias,
                                                  __global DATA_T *output,
                                                  uint alignedCinKK,
                                                  uint topChannel,
                                                  uint topHW,
                                                  uint bottomStep) {
    GEMMBLOCK2X8_SCALAR_FP32(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(RELU6gemmBlock2x4_FP32, (__global const DATA_T *input,
                                           __global const DATA_T *weight,
                                           __global const DATA_T *bias,
                                           __global DATA_T *output,
                                           uint alignedCinKK,
                                           uint topChannel,
                                           uint topHW,
                                           uint bottomStep) {
    GEMMBLOCK2X4(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(RELU6gemmBlock2x4_scalar_FP32, (__global const DATA_T *input,
                                                  __global const DATA_T *weight,
                                                  __global const DATA_T *bias,
                                                  __global DATA_T *output,
                                                  uint alignedCinKK,
                                                  uint topChannel,
                                                  uint topHW,
                                                  uint bottomStep) {
    GEMMBLOCK2X4_SCALAR_FP32(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(RELU6gemmValhall4x4_FP32, (__global const DATA_T *input,
                                             __global const DATA_T *weight,
                                             __global const DATA_T *bias,
                                             __global DATA_T *output,
                                             uint alignedCinKK,
                                             uint topChannel,
                                             uint topHW,
                                             uint bottomStep) {
    GEMMVALHALL4X4(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(RELU6gemmValhall4x4NHWC_FP32, (__global const DATA_T *input,
                                                 __global const DATA_T *weight,
                                                 __global const DATA_T *bias,
                                                 __global DATA_T *output,
                                                 uint alignedCinKK,
                                                 uint topChannel,
                                                 uint topHW,
                                                 uint bottomStep) {
    GEMMVALHALL4X4NHWC(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(RELU6gemmMakalu_FP32, (__global const DATA_T *input,
                                         __global const DATA_T *weight,
                                         __global const DATA_T *bias,
                                         __global DATA_T *output,
                                         uint alignedCinKK,
                                         uint topChannel,
                                         uint topHW,
                                         uint splitTopHW,
                                         uint bottomStep,
                                         uint inputOffset,
                                         uint outputOffset) {
    GEMMMAKALU(input, weight, bias, output, alignedCinKK, topChannel, topHW, splitTopHW, bottomStep, \
               inputOffset, outputOffset)
})
#undef ACT_VEC_F  // RELU6

ADD_SINGLE_KERNEL(alignWeight_1x8_FP32, (__global const DATA_T *src,
                                       __global DATA_T *dst,
                                       uint srcWidth,
                                       uint dstWidth) {
        ALIGNWEIGHT_1X8(src, dst, srcWidth, dstWidth)
})

ADD_SINGLE_KERNEL(alignWeight_1x4_FP32, (__global const DATA_T *src,
                                       __global DATA_T *dst,
                                       uint srcWidth,
                                       uint dstWidth) {
        ALIGNWEIGHT_1X4(src, dst, srcWidth, dstWidth)
})

ADD_SINGLE_KERNEL(align_weight_4_row_1_col_FP32, (__global const DATA_T *src,
                                                __global DATA_T *dst,
                                                uint srcHeight,
                                                uint srcWidth,
                                                uint dstWidth) {
        ALIGN_WEIGHT_4_ROW_1_COL(src, dst, srcHeight, srcWidth, dstWidth)
})

#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // float

/********  QUANTIZED KERNELS ********/
#define Q_WEIGHT_OFFSET(weight, bias, aligned_cin_kk, cin_kk, input_zero_point, filter_zero_point) \
    int g0 = get_global_id(0); \
    int sum = 0; \
    for (int i = 0; i < cin_kk; i++) { \
        sum += weight[g0 * cin_kk + i] * input_zero_point; \
    } \
    for (int i = cin_kk; i < aligned_cin_kk; i++) { \
        sum += filter_zero_point * input_zero_point; \
    } \
    sum -= aligned_cin_kk * filter_zero_point * input_zero_point; \
    bias[g0] -= sum;

#define Q_ALIGN_WEIGHT_4_ROW_1_COL(src, dst, srcHeight, srcWidth, dstWidth) \
    uint dstRow = get_global_id(0); \
    uint dstCol = get_global_id(1); \
    uint srcRow = dstRow * 8 + dstCol % 32 / 16 * 4 + dstCol % 16 / 4; \
    uint srcCol = dstCol / 32 * 4 + dstCol % 16 % 4; \
    uint dstIndex = dstRow * dstWidth + dstCol; \
    if (srcRow < srcHeight && srcCol < srcWidth) { \
        uint srcIndex = srcRow * srcWidth + srcCol; \
        dst[dstIndex] = src[srcIndex]; \
    }

#define Q_REQUANT_VEC(VEC_T, x, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                      outputOffset, activation_min, activation_max) \
    x = mul_hi(x << left_shift, output_multiplier); \
    x = rhadd(x, 0); \
    VEC_T threshold = threshold_mask + select((VEC_T)(0), (VEC_T)(1), x < 0); \
    x = (x >> right_shift) + select((VEC_T)(0), (VEC_T)(1), (x & mask) > threshold); \
    x += outputOffset; \
    x = max(x, activation_min); \
    x = min(x, activation_max);

#define Q_GEMM_BLOCK4X4_FEED_OUTPUT(topChannel, topHW, bias, out16_0, left_shift, output_multiplier, \
                                    threshold_mask, right_shift, mask, outputOffset, activation_min, \
                                    activation_max, output, outputIndex) \
    if (get_global_id(1) * 4 + 3 < topChannel) { \
        if (get_global_id(2) * 4 + 3 < topHW) { \
            int4 bia = vload4(0, bias + get_global_id(1) * 4); \
            out16_0.s0123 += bia; \
            out16_0.s4567 += bia; \
            out16_0.s89AB += bia; \
            out16_0.sCDEF += bia; \
            Q_REQUANT_VEC(int16, out16_0, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                          outputOffset, activation_min, activation_max) \
            vstore4(CONVERT_TO_DATA_T4(out16_0.s048C), 0, output + outputIndex); \
            vstore4(CONVERT_TO_DATA_T4(out16_0.s159D), 0, output + outputIndex + topHW); \
            vstore4(CONVERT_TO_DATA_T4(out16_0.s26AE), 0, output + outputIndex + 2 * topHW); \
            vstore4(CONVERT_TO_DATA_T4(out16_0.s37BF), 0, output + outputIndex + 3 * topHW); \
        } else if (get_global_id(2) * 4 + 2 < topHW) { \
            int4 bia = vload4(0, bias + get_global_id(1) * 4); \
            out16_0.s0123 += bia; \
            out16_0.s4567 += bia; \
            out16_0.s89AB += bia; \
            Q_REQUANT_VEC(int16, out16_0, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                          outputOffset, activation_min, activation_max) \
            vstore3(CONVERT_TO_DATA_T3(out16_0.s048), 0, output + outputIndex); \
            vstore3(CONVERT_TO_DATA_T3(out16_0.s159), 0, output + outputIndex + topHW); \
            vstore3(CONVERT_TO_DATA_T3(out16_0.s26A), 0, output + outputIndex + 2 * topHW); \
            vstore3(CONVERT_TO_DATA_T3(out16_0.s37B), 0, output + outputIndex + 3 * topHW); \
        } else if (get_global_id(2) * 4 + 1 < topHW) { \
            int4 bia = vload4(0, bias + get_global_id(1) * 4); \
            out16_0.s0123 += bia; \
            out16_0.s4567 += bia; \
            int8 out8 = (int8)(out16_0.s0, \
                               out16_0.s1, \
                               out16_0.s2, \
                               out16_0.s3, \
                               out16_0.s4, \
                               out16_0.s5, \
                               out16_0.s6, \
                               out16_0.s7); \
            Q_REQUANT_VEC(int8, out8, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                          outputOffset, activation_min, activation_max) \
            vstore2(CONVERT_TO_DATA_T2(out8.s04), 0, output + outputIndex); \
            vstore2(CONVERT_TO_DATA_T2(out8.s15), 0, output + outputIndex + topHW); \
            vstore2(CONVERT_TO_DATA_T2(out8.s26), 0, output + outputIndex + 2 * topHW); \
            vstore2(CONVERT_TO_DATA_T2(out8.s37), 0, output + outputIndex + 3 * topHW); \
        } else if (get_global_id(2) * 4 < topHW) { \
            int4 bia = vload4(0, bias + get_global_id(1) * 4); \
            out16_0.s0123 += bia; \
            int4 out4 = (int4)(out16_0.s0, out16_0.s1, out16_0.s2, out16_0.s3); \
            Q_REQUANT_VEC(int4, out4, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                          outputOffset, activation_min, activation_max) \
            output[outputIndex] = (DATA_T)out4.s0; \
            output[outputIndex + topHW] = (DATA_T)out4.s1; \
            output[outputIndex + 2 * topHW] = (DATA_T)out4.s2; \
            output[outputIndex + 3 * topHW] = (DATA_T)out4.s3; \
        } \
    } else if (get_global_id(1) * 4 + 2 < topChannel) { \
        if (get_global_id(2) * 4 + 3 < topHW) { \
            int3 bia = vload3(0, bias + get_global_id(1) * 4); \
            out16_0.s012 += bia; \
            out16_0.s456 += bia; \
            out16_0.s89A += bia; \
            out16_0.sCDE += bia; \
            Q_REQUANT_VEC(int16, out16_0, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                          outputOffset, activation_min, activation_max) \
            vstore4(CONVERT_TO_DATA_T4(out16_0.s048C), 0, output + outputIndex); \
            vstore4(CONVERT_TO_DATA_T4(out16_0.s159D), 0, output + outputIndex + topHW); \
            vstore4(CONVERT_TO_DATA_T4(out16_0.s26AE), 0, output + outputIndex + 2 * topHW); \
        } else if (get_global_id(2) * 4 + 2 < topHW) { \
            int3 bia = vload3(0, bias + get_global_id(1) * 4); \
            out16_0.s012 += bia; \
            out16_0.s456 += bia; \
            out16_0.s89A += bia; \
            Q_REQUANT_VEC(int16, out16_0, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                          outputOffset, activation_min, activation_max) \
            vstore3(CONVERT_TO_DATA_T3(out16_0.s048), 0, output + outputIndex); \
            vstore3(CONVERT_TO_DATA_T3(out16_0.s159), 0, output + outputIndex + topHW); \
            vstore3(CONVERT_TO_DATA_T3(out16_0.s26A), 0, output + outputIndex + 2 * topHW); \
        } else if (get_global_id(2) * 4 + 1 < topHW) { \
            int3 bia = vload3(0, bias + get_global_id(1) * 4); \
            out16_0.s012 += bia; \
            out16_0.s456 += bia; \
            Q_REQUANT_VEC(int16, out16_0, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                          outputOffset, activation_min, activation_max) \
            vstore2(CONVERT_TO_DATA_T2(out16_0.s04), 0, output + outputIndex); \
            vstore2(CONVERT_TO_DATA_T2(out16_0.s15), 0, output + outputIndex + topHW); \
            vstore2(CONVERT_TO_DATA_T2(out16_0.s26), 0, output + outputIndex + 2 * topHW); \
        } else if (get_global_id(2) * 4 < topHW) { \
            int3 bia = vload3(0, bias + get_global_id(1) * 4); \
            out16_0.s012 += bia; \
            Q_REQUANT_VEC(int16, out16_0, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                          outputOffset, activation_min, activation_max) \
            output[outputIndex] = (DATA_T)out16_0.s0; \
            output[outputIndex + topHW] = (DATA_T)out16_0.s1; \
            output[outputIndex + 2 * topHW] = (DATA_T)out16_0.s2; \
        } \
    } else if (get_global_id(1) * 4 + 1 < topChannel) { \
        if (get_global_id(2) * 4 + 3 < topHW) { \
            int2 bia = vload2(0, bias + get_global_id(1) * 4); \
            out16_0.s01 += bia; \
            out16_0.s45 += bia; \
            out16_0.s89 += bia; \
            out16_0.sCD += bia; \
            int8 out8 = (int8)(out16_0.s0, \
                               out16_0.s1, \
                               out16_0.s4, \
                               out16_0.s5, \
                               out16_0.s8, \
                               out16_0.s9, \
                               out16_0.sc, \
                               out16_0.sd); \
            Q_REQUANT_VEC(int8, out8, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                          outputOffset, activation_min, activation_max) \
            vstore4(CONVERT_TO_DATA_T4(out8.s0246), 0, output + outputIndex); \
            vstore4(CONVERT_TO_DATA_T4(out8.s1357), 0, output + outputIndex + topHW); \
        } else if (get_global_id(2) * 4 + 2 < topHW) { \
            int2 bia = vload2(0, bias + get_global_id(1) * 4); \
            out16_0.s01 += bia; \
            out16_0.s45 += bia; \
            out16_0.s89 += bia; \
            Q_REQUANT_VEC(int16, out16_0, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                          outputOffset, activation_min, activation_max) \
            vstore3(CONVERT_TO_DATA_T3(out16_0.s048), 0, output + outputIndex); \
            vstore3(CONVERT_TO_DATA_T3(out16_0.s159), 0, output + outputIndex + topHW); \
        } else if (get_global_id(2) * 4 + 1 < topHW) { \
            int2 bia = vload2(0, bias + get_global_id(1) * 4); \
            out16_0.s01 += bia; \
            out16_0.s45 += bia; \
            int4 out4 = (int4)(out16_0.s0, out16_0.s1, out16_0.s4, out16_0.s5); \
            Q_REQUANT_VEC(int4, out4, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                          outputOffset, activation_min, activation_max) \
            vstore2(CONVERT_TO_DATA_T2(out4.s02), 0, output + outputIndex); \
            vstore2(CONVERT_TO_DATA_T2(out4.s13), 0, output + outputIndex + topHW); \
        } else if (get_global_id(2) * 4 < topHW) { \
            int2 bia = vload2(0, bias + get_global_id(1) * 4); \
            out16_0.s01 += bia; \
            int2 out2 = (int2)(out16_0.s0, out16_0.s1); \
            Q_REQUANT_VEC(int2, out2, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                          outputOffset, activation_min, activation_max) \
            output[outputIndex] = (DATA_T)out2.s0; \
            output[outputIndex + topHW] = (DATA_T)out2.s1; \
        } \
    } else if (get_global_id(1) * 4 < topChannel) { \
        if (get_global_id(2) * 4 + 3 < topHW) { \
            int bia = bias[get_global_id(1) * 4]; \
            out16_0.s0 += bia; \
            out16_0.s4 += bia; \
            out16_0.s8 += bia; \
            out16_0.sC += bia; \
            int4 out4 = (int4)(out16_0.s0, out16_0.s4, out16_0.s8, out16_0.sc); \
            Q_REQUANT_VEC(int4, out4, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                          outputOffset, activation_min, activation_max) \
            vstore4(CONVERT_TO_DATA_T4(out4.s0123), 0, output + outputIndex); \
        } else if (get_global_id(2) * 4 + 2 < topHW) { \
            int bia = bias[get_global_id(1) * 4]; \
            out16_0.s0 += bia; \
            out16_0.s4 += bia; \
            out16_0.s8 += bia; \
            int4 out4 = (int4)(out16_0.s0, out16_0.s4, out16_0.s8, 0); \
            Q_REQUANT_VEC(int4, out4, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                          outputOffset, activation_min, activation_max) \
            vstore3(CONVERT_TO_DATA_T3(out4.s012), 0, output + outputIndex); \
        } else if (get_global_id(2) * 4 + 1 < topHW) { \
            int bia = bias[get_global_id(1) * 4]; \
            out16_0.s0 += bia; \
            out16_0.s4 += bia; \
            int2 out2 = (int2)(out16_0.s0, out16_0.s4); \
            Q_REQUANT_VEC(int2, out2, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                          outputOffset, activation_min, activation_max) \
            vstore2(CONVERT_TO_DATA_T2(out2.s01), 0, output + outputIndex); \
        } else if (get_global_id(2) * 4 < topHW) { \
            int bia = bias[get_global_id(1) * 4]; \
            out16_0.s0 += bia; \
            int out = out16_0.s0; \
            Q_REQUANT_VEC(int, out, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                          outputOffset, activation_min, activation_max) \
            output[outputIndex] = (DATA_T)out; \
        } \
    }

#define Q_ARM_DOT_I16W16O16(I16, W16, O16) \
    ARM_DOT(I16.lo.lo, W16.lo.lo, O16.s0); \
    ARM_DOT(I16.lo.lo, W16.lo.hi, O16.s1); \
    ARM_DOT(I16.lo.lo, W16.hi.lo, O16.s2); \
    ARM_DOT(I16.lo.lo, W16.hi.hi, O16.s3); \
    ARM_DOT(I16.lo.hi, W16.lo.lo, O16.s4); \
    ARM_DOT(I16.lo.hi, W16.lo.hi, O16.s5); \
    ARM_DOT(I16.lo.hi, W16.hi.lo, O16.s6); \
    ARM_DOT(I16.lo.hi, W16.hi.hi, O16.s7); \
    ARM_DOT(I16.hi.lo, W16.lo.lo, O16.s8); \
    ARM_DOT(I16.hi.lo, W16.lo.hi, O16.s9); \
    ARM_DOT(I16.hi.lo, W16.hi.lo, O16.sa); \
    ARM_DOT(I16.hi.lo, W16.hi.hi, O16.sb); \
    ARM_DOT(I16.hi.hi, W16.lo.lo, O16.sc); \
    ARM_DOT(I16.hi.hi, W16.lo.hi, O16.sd); \
    ARM_DOT(I16.hi.hi, W16.hi.lo, O16.se); \
    ARM_DOT(I16.hi.hi, W16.hi.hi, O16.sf);

#define Q_GEMM_BLOCK4X4(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep, \
                        coalescing_feature_height_, inputZeroPoint, filterZeroPoint, outputOffset, \
                        output_multiplier, output_shift, activation_min, activation_max) \
    int left_shift = output_shift > 0 ? output_shift + 2 : 2; \
    int right_shift = output_shift > 0 ? 0 : -output_shift; \
    int mask = (((long)1 << right_shift) - 1); \
    int threshold_mask = mask >> 1; \
    if (get_global_id(2) * 4 < topHW) { \
        int16 out16_0 = (int16)0; \
        uint2 src_addr = \
            (uint2)(get_global_id(0) * bottomStep + \
                        (get_global_id(2) * 4 / coalescing_feature_height_) * \
                            coalescing_feature_height_ * alignedCinKK + \
                        (get_global_id(2) * 4 % coalescing_feature_height_ / 4) * 16, \
                    get_global_id(1) * 4 / 8 * 8 * alignedCinKK + \
                        get_global_id(1) * 4 % 8 / 4 * 16);  \
        uint end_alignedCinKK = src_addr.s0 + coalescing_feature_height_ * alignedCinKK; \
        { \
            int in_0 = 0; \
            int in_1 = 0; \
            int in_2 = 0; \
            int in_3 = 0; \
            for (; src_addr.s0 < end_alignedCinKK; \
                    src_addr += (uint2)(coalescing_feature_height_ * 4, 32)) { \
                DATA_T16 input16_0; \
                DATA_T16 weight16_0; \
                input16_0 = vload16(0, input + src_addr.s0); \
                weight16_0 = vload16(0, weight + src_addr.s1); \
                ARM_DOT(input16_0.lo.lo, (DATA_T4)filterZeroPoint, in_0); \
                ARM_DOT(input16_0.lo.hi, (DATA_T4)filterZeroPoint, in_1); \
                ARM_DOT(input16_0.hi.lo, (DATA_T4)filterZeroPoint, in_2); \
                ARM_DOT(input16_0.hi.hi, (DATA_T4)filterZeroPoint, in_3); \
                Q_ARM_DOT_I16W16O16(input16_0, weight16_0, out16_0) \
            } \
            out16_0.lo.lo -= in_0; \
            out16_0.lo.hi -= in_1; \
            out16_0.hi.lo -= in_2; \
            out16_0.hi.hi -= in_3; \
        } \
        int outputIndex = get_global_id(0) * topChannel * topHW + \
                            get_global_id(1) * 4 * topHW + get_global_id(2) * 4; \
        Q_GEMM_BLOCK4X4_FEED_OUTPUT(topChannel, topHW, bias, out16_0, left_shift, output_multiplier, \
                                    threshold_mask, right_shift, mask, outputOffset, activation_min, \
                                    activation_max, output, outputIndex) \
    }

#define Q_GEMM_BLOCK4X4_VALHAL(input, weight, bias, output, alignedCinKK, topChannel, topHW, \
                               bottomStep, coalescing_feature_height_, inputZeroPoint, filterZeroPoint, \
                               outputOffset, output_multiplier, output_shift, activation_min, activation_max) \
    int left_shift = output_shift > 0 ? output_shift + 2 : 2; \
    int right_shift = output_shift > 0 ? 0 : -output_shift; \
    int mask = (((long)1 << right_shift) - 1); \
    int threshold_mask = mask >> 1; \
    int GID2 = get_global_id(2) * 4; \
    int GID1 = get_global_id(1) * 4; \
    if (GID2 < topHW) { \
        int16 out16_0 = (int16)0; \
        uint2 src_addr = \
            (uint2)(get_global_id(0) * bottomStep + (GID2 / 32) * 32 * alignedCinKK + \
                        (GID2 % 32 / 4) * 16, \
                    GID1 / 8 * 8 * alignedCinKK + GID1 % 8 / 4 * 16);  \
        uint end_alignedCinKK = src_addr.s0 + 32 * alignedCinKK; \
        { \
            int in_0 = 0; \
            int in_1 = 0; \
            int in_2 = 0; \
            int in_3 = 0; \
            for (; src_addr.s0 < end_alignedCinKK; src_addr += (uint2)(256, 64)) { \
                DATA_T16 input16_0 = vload16(0, input + src_addr.s0); \
                DATA_T16 weight16_0 = vload16(0, weight + src_addr.s1); \
                ARM_DOT(input16_0.lo.lo, (DATA_T4)filterZeroPoint, in_0); \
                ARM_DOT(input16_0.lo.hi, (DATA_T4)filterZeroPoint, in_1); \
                ARM_DOT(input16_0.hi.lo, (DATA_T4)filterZeroPoint, in_2); \
                ARM_DOT(input16_0.hi.hi, (DATA_T4)filterZeroPoint, in_3); \
                Q_ARM_DOT_I16W16O16(input16_0, weight16_0, out16_0) \
                input16_0 = vload16(8, input + src_addr.s0); \
                weight16_0 = vload16(2, weight + src_addr.s1); \
                ARM_DOT(input16_0.lo.lo, (DATA_T4)filterZeroPoint, in_0); \
                ARM_DOT(input16_0.lo.hi, (DATA_T4)filterZeroPoint, in_1); \
                ARM_DOT(input16_0.hi.lo, (DATA_T4)filterZeroPoint, in_2); \
                ARM_DOT(input16_0.hi.hi, (DATA_T4)filterZeroPoint, in_3); \
                Q_ARM_DOT_I16W16O16(input16_0, weight16_0, out16_0) \
            } \
            out16_0.lo.lo -= in_0; \
            out16_0.lo.hi -= in_1; \
            out16_0.hi.lo -= in_2; \
            out16_0.hi.hi -= in_3; \
        } \
        int outputIndex = get_global_id(0) * topChannel * topHW + GID1 * topHW + GID2; \
        Q_GEMM_BLOCK4X4_FEED_OUTPUT(topChannel, topHW, bias, out16_0, left_shift, output_multiplier, \
                                    threshold_mask, right_shift, mask, outputOffset, activation_min, \
                                    activation_max, output, outputIndex) \
    }

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

ADD_SINGLE_KERNEL(weight_offset_INT8, (__global const DATA_T *weight,
                                           __global int *bias,
                                           uint aligned_cin_kk,
                                           uint cin_kk,
                                           int input_zero_point,
                                           int filter_zero_point) {
        Q_WEIGHT_OFFSET(weight, bias, aligned_cin_kk, cin_kk, input_zero_point, filter_zero_point)
    })

ADD_SINGLE_KERNEL(align_weight_4_row_1_col_INT8, (__global const DATA_T *src,
                                                      __global DATA_T *dst,
                                                      uint srcHeight,
                                                      uint srcWidth,
                                                      uint dstWidth) {
        Q_ALIGN_WEIGHT_4_ROW_1_COL(src, dst, srcHeight, srcWidth, dstWidth)
})

ADD_SINGLE_KERNEL(gemm_block4x4_INT8, (__global const DATA_T *input,
                                           __global const DATA_T *weight,
                                           __global const int *bias,
                                           __global DATA_T *output,
                                           uint alignedCinKK,
                                           uint topChannel,
                                           uint topHW,
                                           uint bottomStep,
                                           uint coalescing_feature_height_,
                                           DATA_T inputZeroPoint,
                                           DATA_T filterZeroPoint,
                                           int outputOffset,
                                           int output_multiplier,
                                           int output_shift,
                                           int activation_min,
                                           int activation_max) {
        Q_GEMM_BLOCK4X4(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep, \
                        coalescing_feature_height_, inputZeroPoint, filterZeroPoint, outputOffset, \
                        output_multiplier, output_shift, activation_min, activation_max)
})

ADD_SINGLE_KERNEL(gemm_block4x4_valhal_INT8, (__global const DATA_T *input,
                                                  __global const DATA_T *weight,
                                                  __global const int *bias,
                                                  __global DATA_T *output,
                                                  uint alignedCinKK,
                                                  uint topChannel,
                                                  uint topHW,
                                                  uint bottomStep,
                                                  uint coalescing_feature_height_,
                                                  DATA_T inputZeroPoint,
                                                  DATA_T filterZeroPoint,
                                                  int outputOffset,
                                                  int output_multiplier,
                                                  int output_shift,
                                                  int activation_min,
                                                  int activation_max) {
        Q_GEMM_BLOCK4X4_VALHAL(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep, \
                               coalescing_feature_height_, inputZeroPoint, filterZeroPoint, outputOffset, \
                               output_multiplier, output_shift, activation_min, activation_max)
})

ADD_SINGLE_KERNEL(gemm_block4x4_valhal_merge_INT8, (__global const unsigned char *input,
                                                  __global const unsigned char *weight,
                                                  __global const int *bias,
                                                  __global unsigned char *concat_out,
                                                  uint alignedCinKK,
                                                  uint topChannel,
                                                  uint topHW,
                                                  uint bottomStep,
                                                  uint coalescing_feature_height_,
                                                  unsigned char inputZeroPoint,
                                                  unsigned char filterZeroPoint,
                                                  int outputOffset,  // conv zero point
                                                  int output_multiplier,
                                                  int output_shift,
                                                  int activation_min,
                                                  int activation_max,
                                                  float conv_out_scale,
                                                  int convout_zeropoint,
                                                  float concat_output_scale,
                                                  int concat_output_zeropoint,
                                                  float concat_scale,
                                                  float concat_bias,
                                                  int concat_out_offset,
                                                  int concat_out_batch,
                                                  int concat_out_channel,
                                                  int concat_out_height,
                                                  int concat_out_width) {
        int left_shift = output_shift > 0 ? output_shift + 2 : 2;
        int right_shift = output_shift > 0 ? 0 : -output_shift;
        int mask = (((long)1 << right_shift) - 1);
        int threshold_mask = mask >> 1;

        int GID2 = get_global_id(2) * 4;
        int GID1 = get_global_id(1) * 4;

        if (GID2 < topHW) {
            int16 out16_0 = (int16)0;

            uint2 src_addr =
                (uint2)(get_global_id(0) * bottomStep + (GID2 / 32) * 32 * alignedCinKK +
                            (GID2 % 32 / 4) * 16,
                        GID1 / 8 * 8 * alignedCinKK + GID1 % 8 / 4 * 16);  // input,weight
            uint end_alignedCinKK = src_addr.s0 + 32 * alignedCinKK;
            {
                int in_0 = 0;
                int in_1 = 0;
                int in_2 = 0;
                int in_3 = 0;
                for (; src_addr.s0 < end_alignedCinKK; src_addr += (uint2)(256, 64)) {
                    uchar16 input16_0;
                    uchar16 weight16_0;

                    input16_0 = vload16(0, input + src_addr.s0);
                    weight16_0 = vload16(0, weight + src_addr.s1);
                    ARM_DOT(input16_0.lo.lo, (uchar4)filterZeroPoint, in_0);
                    ARM_DOT(input16_0.lo.hi, (uchar4)filterZeroPoint, in_1);
                    ARM_DOT(input16_0.hi.lo, (uchar4)filterZeroPoint, in_2);
                    ARM_DOT(input16_0.hi.hi, (uchar4)filterZeroPoint, in_3);

                    Q_ARM_DOT_I16W16O16(input16_0, weight16_0, out16_0)

                    input16_0 = vload16(8, input + src_addr.s0);
                    weight16_0 = vload16(2, weight + src_addr.s1);
                    ARM_DOT(input16_0.lo.lo, (uchar4)filterZeroPoint, in_0);
                    ARM_DOT(input16_0.lo.hi, (uchar4)filterZeroPoint, in_1);
                    ARM_DOT(input16_0.hi.lo, (uchar4)filterZeroPoint, in_2);
                    ARM_DOT(input16_0.hi.hi, (uchar4)filterZeroPoint, in_3);

                    Q_ARM_DOT_I16W16O16(input16_0, weight16_0, out16_0)
                }
                out16_0.lo.lo -= in_0;
                out16_0.lo.hi -= in_1;
                out16_0.hi.lo -= in_2;
                out16_0.hi.hi -= in_3;
            }

            int outputIndex = get_global_id(0) * topChannel * topHW + GID2 * topChannel + GID1;
            int cancatOutIndex = outputIndex + concat_out_offset;
            int Out_HW = concat_out_height * concat_out_width;
            int index = 0;
            if (GID1 + 3 < topChannel) {
                if (GID2 + 3 < topHW) {
                    int4 bia = vload4(0, bias + GID1);
                    out16_0.s0123 += bia;
                    out16_0.s4567 += bia;
                    out16_0.s89AB += bia;
                    out16_0.sCDEF += bia;

                    Q_REQUANT_VEC(int16, out16_0, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                                  outputOffset, activation_min, activation_max)

                    if (conv_out_scale != concat_output_scale) {
                        out16_0 =
                            min(max(convert_int16(round(convert_float16(out16_0) * concat_scale +
                                                        concat_bias) +
                                                  concat_output_zeropoint),
                                    activation_min),
                                activation_max);
                    }

                    vstore4(convert_uchar4(out16_0.s0123), 0, concat_out + cancatOutIndex);
                    vstore4(
                        convert_uchar4(out16_0.s4567), 0, concat_out + cancatOutIndex + topChannel);
                    vstore4(convert_uchar4(out16_0.s89AB),
                            0,
                            concat_out + cancatOutIndex + 2 * topChannel);
                    vstore4(convert_uchar4(out16_0.sCDEF),
                            0,
                            concat_out + cancatOutIndex + 3 * topChannel);

                } else if (get_global_id(2) * 4 + 2 < topHW) {
                    int4 bia = vload4(0, bias + get_global_id(1) * 4);
                    out16_0.s0123 += bia;
                    out16_0.s4567 += bia;
                    out16_0.s89AB += bia;

                    Q_REQUANT_VEC(int16, out16_0, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                                  outputOffset, activation_min, activation_max)

                    if (conv_out_scale != concat_output_scale) {
                        out16_0 =
                            min(max(convert_int16(round(convert_float16(out16_0) * concat_scale +
                                                        concat_bias) +
                                                  concat_output_zeropoint),
                                    activation_min),
                                activation_max);
                    }

                    vstore4(convert_uchar4(out16_0.s0123), 0, concat_out + cancatOutIndex);
                    vstore4(
                        convert_uchar4(out16_0.s4567), 0, concat_out + cancatOutIndex + topChannel);
                    vstore4(convert_uchar4(out16_0.s89AB),
                            0,
                            concat_out + cancatOutIndex + 2 * topChannel);

                } else if (get_global_id(2) * 4 + 1 < topHW) {
                    int4 bia = vload4(0, bias + get_global_id(1) * 4);
                    out16_0.s0123 += bia;
                    out16_0.s4567 += bia;

                    int8 out8 = (int8)(out16_0.s0,
                                       out16_0.s1,
                                       out16_0.s2,
                                       out16_0.s3,
                                       out16_0.s4,
                                       out16_0.s5,
                                       out16_0.s6,
                                       out16_0.s7);
                    Q_REQUANT_VEC(int8, out8, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                                  outputOffset, activation_min, activation_max)

                    if (conv_out_scale != concat_output_scale) {
                        out8 = min(max(convert_int8(round(convert_float8(out8) * concat_scale +
                                                          concat_bias) +
                                                    concat_output_zeropoint),
                                       activation_min),
                                   activation_max);
                    }

                    vstore4(convert_uchar4(out8.s0123), 0, concat_out + cancatOutIndex);
                    vstore4(
                        convert_uchar4(out8.s4567), 0, concat_out + cancatOutIndex + topChannel);

                } else if (get_global_id(2) * 4 < topHW) {
                    int4 bia = vload4(0, bias + get_global_id(1) * 4);
                    out16_0.s0123 += bia;

                    int4 out4 = (int4)(out16_0.s0, out16_0.s1, out16_0.s2, out16_0.s3);

                    Q_REQUANT_VEC(int4, out4, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                                  outputOffset, activation_min, activation_max)

                    if (conv_out_scale != concat_output_scale) {
                        out4 = min(max(convert_int4(round(convert_float4(out4) * concat_scale +
                                                          concat_bias) +
                                                    concat_output_zeropoint),
                                       activation_min),
                                   activation_max);
                    }

                    vstore4(convert_uchar4(out4.s0123), 0, concat_out + cancatOutIndex);
                }
            } else if (get_global_id(1) * 4 + 2 < topChannel) {
                if (get_global_id(2) * 4 + 3 < topHW) {
                    int3 bia = vload3(0, bias + get_global_id(1) * 4);
                    out16_0.s012 += bia;
                    out16_0.s456 += bia;
                    out16_0.s89A += bia;
                    out16_0.sCDE += bia;

                    Q_REQUANT_VEC(int16, out16_0, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                                  outputOffset, activation_min, activation_max)

                    if (conv_out_scale != concat_output_scale) {
                        out16_0 =
                            min(max(convert_int16(round(convert_float16(out16_0) * concat_scale +
                                                        concat_bias) +
                                                  concat_output_zeropoint),
                                    activation_min),
                                activation_max);
                    }

                    vstore3(convert_uchar3(out16_0.s012), 0, concat_out + cancatOutIndex);
                    vstore3(
                        convert_uchar3(out16_0.s456), 0, concat_out + cancatOutIndex + topChannel);
                    vstore3(convert_uchar3(out16_0.s89A),
                            0,
                            concat_out + cancatOutIndex + 2 * topChannel);
                    vstore3(convert_uchar3(out16_0.sCDE),
                            0,
                            concat_out + cancatOutIndex + 3 * topChannel);

                } else if (get_global_id(2) * 4 + 2 < topHW) {
                    int3 bia = vload3(0, bias + get_global_id(1) * 4);
                    out16_0.s012 += bia;
                    out16_0.s456 += bia;
                    out16_0.s89A += bia;

                    Q_REQUANT_VEC(int16, out16_0, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                                  outputOffset, activation_min, activation_max)

                    if (conv_out_scale != concat_output_scale) {
                        out16_0 =
                            min(max(convert_int16(round(convert_float16(out16_0) * concat_scale +
                                                        concat_bias) +
                                                  concat_output_zeropoint),
                                    activation_min),
                                activation_max);
                    }

                    vstore3(convert_uchar3(out16_0.s012), 0, concat_out + cancatOutIndex);
                    vstore3(
                        convert_uchar3(out16_0.s456), 0, concat_out + cancatOutIndex + topChannel);
                    vstore3(convert_uchar3(out16_0.s89A),
                            0,
                            concat_out + cancatOutIndex + 2 * topChannel);

                } else if (get_global_id(2) * 4 + 1 < topHW) {
                    int3 bia = vload3(0, bias + get_global_id(1) * 4);
                    out16_0.s012 += bia;
                    out16_0.s456 += bia;

                    Q_REQUANT_VEC(int16, out16_0, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                                  outputOffset, activation_min, activation_max)

                    if (conv_out_scale != concat_output_scale) {
                        out16_0 =
                            min(max(convert_int16(round(convert_float16(out16_0) * concat_scale +
                                                        concat_bias) +
                                                  concat_output_zeropoint),
                                    activation_min),
                                activation_max);
                    }

                    vstore3(convert_uchar3(out16_0.s012), 0, concat_out + cancatOutIndex);
                    vstore3(
                        convert_uchar3(out16_0.s456), 0, concat_out + cancatOutIndex + topChannel);

                } else if (get_global_id(2) * 4 < topHW) {
                    int3 bia = vload3(0, bias + get_global_id(1) * 4);
                    out16_0.s012 += bia;

                    Q_REQUANT_VEC(int16, out16_0, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                                  outputOffset, activation_min, activation_max)

                    if (conv_out_scale != concat_output_scale) {
                        out16_0 =
                            min(max(convert_int16(round(convert_float16(out16_0) * concat_scale +
                                                        concat_bias) +
                                                  concat_output_zeropoint),
                                    activation_min),
                                activation_max);
                    }

                    vstore3(convert_uchar3(out16_0.s012), 0, concat_out + cancatOutIndex);
                }
            } else if (get_global_id(1) * 4 + 1 < topChannel) {
                if (get_global_id(2) * 4 + 3 < topHW) {
                    int2 bia = vload2(0, bias + get_global_id(1) * 4);
                    out16_0.s01 += bia;
                    out16_0.s45 += bia;
                    out16_0.s89 += bia;
                    out16_0.sCD += bia;

                    int8 out8 = (int8)(out16_0.s0,
                                       out16_0.s1,
                                       out16_0.s4,
                                       out16_0.s5,
                                       out16_0.s8,
                                       out16_0.s9,
                                       out16_0.sc,
                                       out16_0.sd);
                    Q_REQUANT_VEC(int8, out8, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                                  outputOffset, activation_min, activation_max)

                    if (conv_out_scale != concat_output_scale) {
                        out8 = min(max(convert_int8(round(convert_float8(out8) * concat_scale +
                                                          concat_bias) +
                                                    concat_output_zeropoint),
                                       activation_min),
                                   activation_max);
                    }

                    vstore2(convert_uchar2(out8.s01), 0, concat_out + cancatOutIndex);
                    vstore2(convert_uchar2(out8.s23), 0, concat_out + cancatOutIndex + topChannel);
                    vstore2(
                        convert_uchar2(out8.s45), 0, concat_out + cancatOutIndex + 2 * topChannel);
                    vstore2(
                        convert_uchar2(out8.s67), 0, concat_out + cancatOutIndex + 3 * topChannel);

                } else if (get_global_id(2) * 4 + 2 < topHW) {
                    int2 bia = vload2(0, bias + get_global_id(1) * 4);
                    out16_0.s01 += bia;
                    out16_0.s45 += bia;
                    out16_0.s89 += bia;

                    Q_REQUANT_VEC(int16, out16_0, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                                  outputOffset, activation_min, activation_max)

                    if (conv_out_scale != concat_output_scale) {
                        out16_0 =
                            min(max(convert_int16(round(convert_float16(out16_0) * concat_scale +
                                                        concat_bias) +
                                                  concat_output_zeropoint),
                                    activation_min),
                                activation_max);
                    }
                    vstore2(convert_uchar2(out16_0.s01), 0, concat_out + cancatOutIndex);
                    vstore2(
                        convert_uchar2(out16_0.s45), 0, concat_out + cancatOutIndex + topChannel);
                    vstore2(convert_uchar2(out16_0.s89),
                            0,
                            concat_out + cancatOutIndex + 2 * topChannel);

                } else if (get_global_id(2) * 4 + 1 < topHW) {
                    int2 bia = vload2(0, bias + get_global_id(1) * 4);
                    out16_0.s01 += bia;
                    out16_0.s45 += bia;

                    int4 out4 = (int4)(out16_0.s0, out16_0.s1, out16_0.s4, out16_0.s5);
                    Q_REQUANT_VEC(int4, out4, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                                  outputOffset, activation_min, activation_max)

                    if (conv_out_scale != concat_output_scale) {
                        out4 = min(max(convert_int4(round(convert_float4(out4) * concat_scale +
                                                          concat_bias) +
                                                    concat_output_zeropoint),
                                       activation_min),
                                   activation_max);
                    }
                    vstore2(convert_uchar2(out4.s01), 0, concat_out + cancatOutIndex);
                    vstore2(convert_uchar2(out4.s23), 0, concat_out + cancatOutIndex + topChannel);

                } else if (get_global_id(2) * 4 < topHW) {
                    int2 bia = vload2(0, bias + get_global_id(1) * 4);
                    out16_0.s01 += bia;

                    int2 out2 = (int2)(out16_0.s0, out16_0.s1);
                    Q_REQUANT_VEC(int2, out2, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                                  outputOffset, activation_min, activation_max)

                    if (conv_out_scale != concat_output_scale) {
                        out2 = min(max(convert_int2(round(convert_float2(out2) * concat_scale +
                                                          concat_bias) +
                                                    concat_output_zeropoint),
                                       activation_min),
                                   activation_max);
                    }

                    vstore2(convert_uchar2(out2.s01), 0, concat_out + cancatOutIndex);
                }
            } else if (get_global_id(1) * 4 < topChannel) {
                if (get_global_id(2) * 4 + 3 < topHW) {
                    int bia = bias[get_global_id(1) * 4];
                    out16_0.s0 += bia;
                    out16_0.s4 += bia;
                    out16_0.s8 += bia;
                    out16_0.sC += bia;

                    int4 out4 = (int4)(out16_0.s0, out16_0.s4, out16_0.s8, out16_0.sc);
                    Q_REQUANT_VEC(int4, out4, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                                  outputOffset, activation_min, activation_max)

                    if (conv_out_scale != concat_output_scale) {
                        out4 = min(max(convert_int4(round(convert_float4(out4) * concat_scale +
                                                          concat_bias) +
                                                    concat_output_zeropoint),
                                       activation_min),
                                   activation_max);
                    }
                    concat_out[cancatOutIndex] = (uchar)out4.s0;
                    concat_out[cancatOutIndex + topChannel] = (uchar)out4.s1;
                    concat_out[cancatOutIndex + 2 * topChannel] = (uchar)out4.s2;
                    concat_out[cancatOutIndex + 3 * topChannel] = (uchar)out4.s3;

                } else if (get_global_id(2) * 4 + 2 < topHW) {
                    int bia = bias[get_global_id(1) * 4];
                    out16_0.s0 += bia;
                    out16_0.s4 += bia;
                    out16_0.s8 += bia;

                    int4 out4 = (int4)(out16_0.s0, out16_0.s4, out16_0.s8, 0);
                    Q_REQUANT_VEC(int4, out4, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                                  outputOffset, activation_min, activation_max)

                    if (conv_out_scale != concat_output_scale) {
                        out4 = min(max(convert_int4(round(convert_float4(out4) * concat_scale +
                                                          concat_bias) +
                                                    concat_output_zeropoint),
                                       activation_min),
                                   activation_max);
                    }
                    concat_out[cancatOutIndex] = (uchar)out4.s0;
                    concat_out[cancatOutIndex + topChannel] = (uchar)out4.s1;
                    concat_out[cancatOutIndex + 2 * topChannel] = (uchar)out4.s2;

                } else if (get_global_id(2) * 4 + 1 < topHW) {
                    int bia = bias[get_global_id(1) * 4];
                    out16_0.s0 += bia;
                    out16_0.s4 += bia;

                    int2 out2 = (int2)(out16_0.s0, out16_0.s4);
                    Q_REQUANT_VEC(int2, out2, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                                  outputOffset, activation_min, activation_max)

                    if (conv_out_scale != concat_output_scale) {
                        out2 = min(max(convert_int2(round(convert_float2(out2) * concat_scale +
                                                          concat_bias) +
                                                    concat_output_zeropoint),
                                       activation_min),
                                   activation_max);
                    }
                    concat_out[cancatOutIndex] = (uchar)out2.s0;
                    concat_out[cancatOutIndex + topChannel] = (uchar)out2.s1;

                } else if (get_global_id(2) * 4 < topHW) {
                    int bia = bias[get_global_id(1) * 4];
                    out16_0.s0 += bia;

                    int out = out16_0.s0;
                    Q_REQUANT_VEC(int, out, left_shift, right_shift, output_multiplier, threshold_mask, mask, \
                                  outputOffset, activation_min, activation_max)

                    if (conv_out_scale != concat_output_scale) {
                        out = min(max((int)round(out * concat_scale + concat_bias) +
                                          concat_output_zeropoint,
                                      activation_min),
                                  activation_max);
                    }
                    concat_out[cancatOutIndex] = (uchar)out;
                }
            }
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

ADD_SINGLE_KERNEL(SIGNEDweight_offset_INT8, (__global const DATA_T *weight,
                                           __global int *bias,
                                           uint aligned_cin_kk,
                                           uint cin_kk,
                                           int input_zero_point,
                                           int filter_zero_point) {
        Q_WEIGHT_OFFSET(weight, bias, aligned_cin_kk, cin_kk, input_zero_point, filter_zero_point)
})

ADD_SINGLE_KERNEL(SIGNEDalign_weight_4_row_1_col_INT8, (__global const DATA_T *src,
                                                      __global DATA_T *dst,
                                                      uint srcHeight,
                                                      uint srcWidth,
                                                      uint dstWidth) {
        Q_ALIGN_WEIGHT_4_ROW_1_COL(src, dst, srcHeight, srcWidth, dstWidth)
})

ADD_SINGLE_KERNEL(SIGNEDgemm_block4x4_INT8, (__global const DATA_T *input,
                                           __global const DATA_T *weight,
                                           __global const int *bias,
                                           __global DATA_T *output,
                                           uint alignedCinKK,
                                           uint topChannel,
                                           uint topHW,
                                           uint bottomStep,
                                           uint coalescing_feature_height_,
                                           DATA_T inputZeroPoint,
                                           DATA_T filterZeroPoint,
                                           int outputOffset,
                                           int output_multiplier,
                                           int output_shift,
                                           int activation_min,
                                           int activation_max) {
        Q_GEMM_BLOCK4X4(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep, \
                        coalescing_feature_height_, inputZeroPoint, filterZeroPoint, outputOffset, \
                        output_multiplier, output_shift, activation_min, activation_max)
})

ADD_SINGLE_KERNEL(SIGNEDgemm_block4x4_valhal_INT8, (__global const DATA_T *input,
                                                  __global const DATA_T *weight,
                                                  __global const int *bias,
                                                  __global DATA_T *output,
                                                  uint alignedCinKK,
                                                  uint topChannel,
                                                  uint topHW,
                                                  uint bottomStep,
                                                  uint coalescing_feature_height_,
                                                  DATA_T inputZeroPoint,
                                                  DATA_T filterZeroPoint,
                                                  int outputOffset,
                                                  int output_multiplier,
                                                  int output_shift,
                                                  int activation_min,
                                                  int activation_max) {
        Q_GEMM_BLOCK4X4_VALHAL(input, weight, bias, output, alignedCinKK, topChannel, topHW, bottomStep, \
                               coalescing_feature_height_, inputZeroPoint, filterZeroPoint, outputOffset, \
                               output_multiplier, output_shift, activation_min, activation_max)
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
