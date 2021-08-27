#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define PURE_MATRIX_TRANSPOSE(matrix, matrixTrans) \
        unsigned int heightTrans = get_global_size(0); \
        unsigned int widthTrans = get_global_size(1); \
        unsigned int outIndex = get_global_id(0) * widthTrans + get_global_id(1); \
        unsigned int inputIndex = get_global_id(1) * heightTrans + get_global_id(0); \
        matrixTrans[outIndex] = matrix[inputIndex];

#define GEMM_MAKALU_DECONV(input, weight, output, alignedCinKK, topChannel, topHW, bottomStep) \
        int GID2 = get_global_id(2) * 4; \
        int GID1 = get_global_id(1) * 4; \
        if (GID2 < topHW) { \
            DATA_T16 out8_0 = 0.0f; \
            DATA_T4 out_0 = 0.0f; \
            DATA_T4 out_1 = 0.0f; \
            DATA_T4 out_2 = 0.0f; \
            DATA_T4 out_3 = 0.0f; \
            uint2 src_addr = \
                (uint2)(get_global_id(0) * bottomStep + (GID2 / 48) * 48 * alignedCinKK + \
                            (GID2 % 48 / 4) * 16, \
                        GID1 / 8 * 8 * alignedCinKK + GID1 % 8 / 4 * 16);  \
            uint end_alignedCinKK = src_addr.s0 + 48 * alignedCinKK; \
            for (; src_addr.s0 < end_alignedCinKK; src_addr += (uint2)(192, 32)) { \
                DATA_T16 input16_0; \
                DATA_T16 weight16_0; \
                input16_0 = vload16(0, input + src_addr.s0); \
                weight16_0 = vload16(0, weight + src_addr.s1); \
                out_0 = mad(input16_0.s0, weight16_0.lo.lo, out_0); \
                out_0 = mad(input16_0.s4, weight16_0.lo.hi, out_0); \
                out_0 = mad(input16_0.s8, weight16_0.hi.lo, out_0); \
                out_0 = mad(input16_0.sC, weight16_0.hi.hi, out_0); \
                out_1 = mad(input16_0.s1, weight16_0.lo.lo, out_1); \
                out_1 = mad(input16_0.s5, weight16_0.lo.hi, out_1); \
                out_1 = mad(input16_0.s9, weight16_0.hi.lo, out_1); \
                out_1 = mad(input16_0.sD, weight16_0.hi.hi, out_1); \
                out_2 = mad(input16_0.s2, weight16_0.lo.lo, out_2); \
                out_2 = mad(input16_0.s6, weight16_0.lo.hi, out_2); \
                out_2 = mad(input16_0.sA, weight16_0.hi.lo, out_2); \
                out_2 = mad(input16_0.sE, weight16_0.hi.hi, out_2); \
                out_3 = mad(input16_0.s3, weight16_0.lo.lo, out_3); \
                out_3 = mad(input16_0.s7, weight16_0.lo.hi, out_3); \
                out_3 = mad(input16_0.sB, weight16_0.hi.lo, out_3); \
                out_3 = mad(input16_0.sF, weight16_0.hi.hi, out_3); \
            } \
            int outputIndex = get_global_id(0) * topChannel * topHW + GID1 * topHW + GID2; \
            if (GID1 + 3 < topChannel) { \
                if (GID2 + 3 < topHW) { \
                    out8_0.s048C = out_0; \
                    out8_0.s159D = out_1; \
                    out8_0.s26AE = out_2; \
                    out8_0.s37BF = out_3; \
                    vstore4(out8_0.s0123, 0, output + outputIndex); \
                    vstore4(out8_0.s4567, 0, output + outputIndex + topHW); \
                    vstore4(out8_0.s89AB, 0, output + outputIndex + 2 * topHW); \
                    vstore4(out8_0.sCDEF, 0, output + outputIndex + 3 * topHW); \
                } else if (GID2 + 2 < topHW) { \
                    out8_0.s048C = out_0; \
                    out8_0.s159D = out_1; \
                    out8_0.s26AE = out_2; \
                    vstore3(out8_0.s012, 0, output + outputIndex); \
                    vstore3(out8_0.s456, 0, output + outputIndex + topHW); \
                    vstore3(out8_0.s89A, 0, output + outputIndex + 2 * topHW); \
                    vstore3(out8_0.sCDE, 0, output + outputIndex + 3 * topHW); \
                } else if (GID2 + 1 < topHW) { \
                    out8_0.s048C = out_0; \
                    out8_0.s159D = out_1; \
                    vstore2(out8_0.s01, 0, output + outputIndex); \
                    vstore2(out8_0.s45, 0, output + outputIndex + topHW); \
                    vstore2(out8_0.s89, 0, output + outputIndex + 2 * topHW); \
                    vstore2(out8_0.sCD, 0, output + outputIndex + 3 * topHW); \
                } else if (GID2 < topHW) { \
                    out8_0.s048C = out_0; \
                    output[outputIndex] = out8_0.s0; \
                    output[outputIndex + topHW] = out8_0.s4; \
                    output[outputIndex + 2 * topHW] = out8_0.s8; \
                    output[outputIndex + 3 * topHW] = out8_0.sC; \
                } \
            } else if (GID1 + 2 < topChannel) { \
                if (GID2 + 3 < topHW) { \
                    out8_0.s048C = out_0; \
                    out8_0.s159D = out_1; \
                    out8_0.s26AE = out_2; \
                    out8_0.s37BF = out_3; \
                    vstore4(out8_0.s0123, 0, output + outputIndex); \
                    vstore4(out8_0.s4567, 0, output + outputIndex + topHW); \
                    vstore4(out8_0.s89AB, 0, output + outputIndex + 2 * topHW); \
                } else if (GID2 + 2 < topHW) { \
                    out8_0.s048C = out_0; \
                    out8_0.s159D = out_1; \
                    out8_0.s26AE = out_2; \
                    vstore3(out8_0.s012, 0, output + outputIndex); \
                    vstore3(out8_0.s456, 0, output + outputIndex + topHW); \
                    vstore3(out8_0.s89A, 0, output + outputIndex + 2 * topHW); \
                } else if (GID2 + 1 < topHW) { \
                    out8_0.s048C = out_0; \
                    out8_0.s159D = out_1; \
                    vstore2(out8_0.s01, 0, output + outputIndex); \
                    vstore2(out8_0.s45, 0, output + outputIndex + topHW); \
                    vstore2(out8_0.s89, 0, output + outputIndex + 2 * topHW); \
                } else if (GID2 < topHW) { \
                    out8_0.s048C = out_0; \
                    output[outputIndex] = out8_0.s0; \
                    output[outputIndex + topHW] = out8_0.s4; \
                    output[outputIndex + 2 * topHW] = out8_0.s8; \
                } \
            } else if (GID1 + 1 < topChannel) { \
                if (GID2 + 3 < topHW) { \
                    out8_0.s048C = out_0; \
                    out8_0.s159D = out_1; \
                    out8_0.s26AE = out_2; \
                    out8_0.s37BF = out_3; \
                    vstore4(out8_0.s0123, 0, output + outputIndex); \
                    vstore4(out8_0.s4567, 0, output + outputIndex + topHW); \
                } else if (GID2 + 2 < topHW) { \
                    out8_0.s048C = out_0; \
                    out8_0.s159D = out_1; \
                    out8_0.s26AE = out_2; \
                    vstore3(out8_0.s012, 0, output + outputIndex); \
                    vstore3(out8_0.s456, 0, output + outputIndex + topHW); \
                } else if (GID2 + 1 < topHW) { \
                    out8_0.s048C = out_0; \
                    out8_0.s159D = out_1; \
                    vstore2(out8_0.s01, 0, output + outputIndex); \
                    vstore2(out8_0.s45, 0, output + outputIndex + topHW); \
                } else if (GID2 < topHW) { \
                    out8_0.s048C = out_0; \
                    output[outputIndex] = out8_0.s0; \
                    output[outputIndex + topHW] = out8_0.s4; \
                } \
            } else if (GID1 < topChannel) { \
                if (GID2 + 3 < topHW) { \
                    out8_0.s048C = out_0; \
                    out8_0.s159D = out_1; \
                    out8_0.s26AE = out_2; \
                    out8_0.s37BF = out_3; \
                    vstore4(out8_0.s0123, 0, output + outputIndex); \
                } else if (GID2 + 2 < topHW) { \
                    out8_0.s048C = out_0; \
                    out8_0.s159D = out_1; \
                    out8_0.s26AE = out_2; \
                    vstore3(out8_0.s012, 0, output + outputIndex); \
                } else if (GID2 + 1 < topHW) { \
                    out8_0.s048C = out_0; \
                    out8_0.s159D = out_1; \
                    vstore2(out8_0.s01, 0, output + outputIndex); \
                } else if (GID2 < topHW) { \
                    out8_0.s048C = out_0; \
                    output[outputIndex] = out8_0.s0; \
                } \
            } \
        }

#define GEMM_MAKALU_DECONV_OPT(input, weight, output, alignedCinKK, topChannel, topHW, bottomStep, kernelHW) \
        int GID2 = get_global_id(2) * 4; \
        int GID1 = get_global_id(1) * 4; \
        if (GID2 < topHW) { \
            DATA_T4 out_0 = 0.0f; \
            DATA_T4 out_1 = 0.0f; \
            DATA_T4 out_2 = 0.0f; \
            DATA_T4 out_3 = 0.0f; \
            uint2 src_addr = \
                (uint2)(get_global_id(0) * bottomStep + (GID2 / 48) * 48 * alignedCinKK + \
                            (GID2 % 48 / 4) * 16, \
                        GID1 / 8 * 8 * alignedCinKK + GID1 % 8 / 4 * 16);  \
            uint end_alignedCinKK = src_addr.s0 + 48 * alignedCinKK; \
            for (; src_addr.s0 < end_alignedCinKK; src_addr += (uint2)(192, 32)) { \
                DATA_T16 input16_0; \
                DATA_T16 weight16_0; \
                input16_0 = vload16(0, input + src_addr.s0); \
                weight16_0 = vload16(0, weight + src_addr.s1); \
                out_0 = mad(input16_0.s0, weight16_0.lo.lo, out_0); \
                out_0 = mad(input16_0.s4, weight16_0.lo.hi, out_0); \
                out_0 = mad(input16_0.s8, weight16_0.hi.lo, out_0); \
                out_0 = mad(input16_0.sC, weight16_0.hi.hi, out_0); \
                out_1 = mad(input16_0.s1, weight16_0.lo.lo, out_1); \
                out_1 = mad(input16_0.s5, weight16_0.lo.hi, out_1); \
                out_1 = mad(input16_0.s9, weight16_0.hi.lo, out_1); \
                out_1 = mad(input16_0.sD, weight16_0.hi.hi, out_1); \
                out_2 = mad(input16_0.s2, weight16_0.lo.lo, out_2); \
                out_2 = mad(input16_0.s6, weight16_0.lo.hi, out_2); \
                out_2 = mad(input16_0.sA, weight16_0.hi.lo, out_2); \
                out_2 = mad(input16_0.sE, weight16_0.hi.hi, out_2); \
                out_3 = mad(input16_0.s3, weight16_0.lo.lo, out_3); \
                out_3 = mad(input16_0.s7, weight16_0.lo.hi, out_3); \
                out_3 = mad(input16_0.sB, weight16_0.hi.lo, out_3); \
                out_3 = mad(input16_0.sF, weight16_0.hi.hi, out_3); \
            } \
            int outputIndex = get_global_id(0) * topChannel * topHW + \
                              GID1 / kernelHW * kernelHW * topHW + GID1 % kernelHW + \
                              GID2 * kernelHW; \
            if (GID1 + 3 < topChannel) { \
                if (GID2 + 3 < topHW) { \
                    vstore4(out_0, 0, output + outputIndex); \
                    vstore4(out_1, 0, output + outputIndex + kernelHW); \
                    vstore4(out_2, 0, output + outputIndex + 2 * kernelHW); \
                    vstore4(out_3, 0, output + outputIndex + 3 * kernelHW); \
                } else if (GID2 + 2 < topHW) { \
                    vstore4(out_0, 0, output + outputIndex); \
                    vstore4(out_1, 0, output + outputIndex + kernelHW); \
                    vstore4(out_2, 0, output + outputIndex + 2 * kernelHW); \
                } else if (GID2 + 1 < topHW) { \
                    vstore4(out_0, 0, output + outputIndex); \
                    vstore4(out_1, 0, output + outputIndex + kernelHW); \
                } else if (GID2 < topHW) { \
                    vstore4(out_0, 0, output + outputIndex); \
                } \
            } else if (GID1 + 2 < topChannel) { \
                if (GID2 + 3 < topHW) { \
                    vstore3(out_0.s012, 0, output + outputIndex); \
                    vstore3(out_1.s012, 0, output + outputIndex + kernelHW); \
                    vstore3(out_2.s012, 0, output + outputIndex + 2 * kernelHW); \
                    vstore3(out_3.s012, 0, output + outputIndex + 3 * kernelHW); \
                } else if (GID2 + 2 < topHW) { \
                    vstore3(out_0.s012, 0, output + outputIndex); \
                    vstore3(out_1.s012, 0, output + outputIndex + kernelHW); \
                    vstore3(out_2.s012, 0, output + outputIndex + 2 * kernelHW); \
                } else if (GID2 + 1 < topHW) { \
                    vstore3(out_0.s012, 0, output + outputIndex); \
                    vstore3(out_1.s012, 0, output + outputIndex + kernelHW); \
                } else if (GID2 < topHW) { \
                    vstore3(out_0.s012, 0, output + outputIndex); \
                } \
            } else if (GID1 + 1 < topChannel) { \
                if (GID2 + 3 < topHW) { \
                    vstore2(out_0.s01, 0, output + outputIndex); \
                    vstore2(out_1.s01, 0, output + outputIndex + kernelHW); \
                    vstore2(out_2.s01, 0, output + outputIndex + 2 * kernelHW); \
                    vstore2(out_3.s01, 0, output + outputIndex + 3 * kernelHW); \
                } else if (GID2 + 2 < topHW) { \
                    vstore2(out_0.s01, 0, output + outputIndex); \
                    vstore2(out_1.s01, 0, output + outputIndex + kernelHW); \
                    vstore2(out_2.s01, 0, output + outputIndex + 2 * kernelHW); \
                } else if (GID2 + 1 < topHW) { \
                    vstore2(out_0.s01, 0, output + outputIndex); \
                    vstore2(out_1.s01, 0, output + outputIndex + kernelHW); \
                } else if (GID2 < topHW) { \
                    vstore2(out_0.s01, 0, output + outputIndex); \
                } \
            } else if (GID1 < topChannel) { \
                if (GID2 + 3 < topHW) { \
                    output[outputIndex] = out_0.s0; \
                    output[outputIndex + kernelHW] = out_1.s0; \
                    output[outputIndex + 2 * kernelHW] = out_2.s0; \
                    output[outputIndex + 3 * kernelHW] = out_3.s0; \
                } else if (GID2 + 2 < topHW) { \
                    output[outputIndex] = out_0.s0; \
                    output[outputIndex + kernelHW] = out_1.s0; \
                    output[outputIndex + 2 * kernelHW] = out_2.s0; \
                } else if (GID2 + 1 < topHW) { \
                    output[outputIndex] = out_0.s0; \
                    output[outputIndex + kernelHW] = out_1.s0; \
                } else if (GID2 < topHW) { \
                    output[outputIndex] = out_0.s0; \
                } \
            } \
        }

#define GEMM_MAKALU_DECONV_2_TIMES(input, weight, bias, output, alignedCinKK, topChannel, width, topHW, bottomStep) \
        int GID2 = get_global_id(2) * 4; \
        int GID1 = get_global_id(1) * 4; \
        if (GID2 < topHW && GID1 < topChannel) { \
            DATA_T bia = bias[GID1 / 4]; \
            DATA_T16 out8_0 = (DATA_T16)(bia); \
            DATA_T4 out_0 = 0.0f; \
            DATA_T4 out_1 = 0.0f; \
            DATA_T4 out_2 = 0.0f; \
            DATA_T4 out_3 = 0.0f; \
            uint2 src_addr = \
                (uint2)(get_global_id(0) * bottomStep + (GID2 / 48) * 48 * alignedCinKK + \
                            (GID2 % 48 / 4) * 16, \
                        GID1 / 8 * 8 * alignedCinKK + GID1 % 8 / 4 * 16);  \
            uint end_alignedCinKK = src_addr.s0 + 48 * alignedCinKK; \
            for (; src_addr.s0 < end_alignedCinKK; src_addr += (uint2)(192, 32)) { \
                DATA_T16 input16_0; \
                DATA_T16 weight16_0; \
                input16_0 = vload16(0, input + src_addr.s0); \
                weight16_0 = vload16(0, weight + src_addr.s1); \
                out_0 = mad(input16_0.s0, weight16_0.lo.lo, out_0); \
                out_0 = mad(input16_0.s4, weight16_0.lo.hi, out_0); \
                out_0 = mad(input16_0.s8, weight16_0.hi.lo, out_0); \
                out_0 = mad(input16_0.sC, weight16_0.hi.hi, out_0); \
                out_1 = mad(input16_0.s1, weight16_0.lo.lo, out_1); \
                out_1 = mad(input16_0.s5, weight16_0.lo.hi, out_1); \
                out_1 = mad(input16_0.s9, weight16_0.hi.lo, out_1); \
                out_1 = mad(input16_0.sD, weight16_0.hi.hi, out_1); \
                out_2 = mad(input16_0.s2, weight16_0.lo.lo, out_2); \
                out_2 = mad(input16_0.s6, weight16_0.lo.hi, out_2); \
                out_2 = mad(input16_0.sA, weight16_0.hi.lo, out_2); \
                out_2 = mad(input16_0.sE, weight16_0.hi.hi, out_2); \
                out_3 = mad(input16_0.s3, weight16_0.lo.lo, out_3); \
                out_3 = mad(input16_0.s7, weight16_0.lo.hi, out_3); \
                out_3 = mad(input16_0.sB, weight16_0.hi.lo, out_3); \
                out_3 = mad(input16_0.sF, weight16_0.hi.hi, out_3); \
            } \
            int outputIndex = get_global_id(0) * topChannel / 4 * topHW * 4 + GID1 / 4 * topHW * 4 + \
                              GID2 / width * width * 2 * 2 + (GID2 % width) * 2; \
            if (GID2 + 3 < topHW) { \
                out8_0.lo += (DATA_T8)(out_0.s01, out_1.s01, out_2.s01, out_3.s01); \
                out8_0.hi += (DATA_T8)(out_0.s23, out_1.s23, out_2.s23, out_3.s23); \
                out8_0.lo = ACT_VEC_F(DATA_T8, out8_0.lo); \
                out8_0.hi = ACT_VEC_F(DATA_T8, out8_0.hi); \
                vstore8(out8_0.lo, 0, output + outputIndex); \
                vstore8(out8_0.hi, 0, output + outputIndex + 2 * width); \
            } else if (GID2 + 2 < topHW) { \
                out8_0.s0123 += (DATA_T4)(out_0.s01, out_1.s01); \
                out8_0.s89AB += (DATA_T4)(out_0.s23, out_1.s23); \
                out8_0.s45 += (DATA_T2)(out_2.s01); \
                out8_0.sCD += (DATA_T2)(out_2.s23); \
                out8_0.s0123 = ACT_VEC_F(DATA_T4, out8_0.s0123); \
                out8_0.s89AB = ACT_VEC_F(DATA_T4, out8_0.s89AB); \
                out8_0.s45 = ACT_VEC_F(DATA_T2, out8_0.s45); \
                out8_0.sCD = ACT_VEC_F(DATA_T2, out8_0.sCD); \
                vstore4(out8_0.s0123, 0, output + outputIndex); \
                vstore2(out8_0.s45, 0, output + outputIndex + 4); \
                vstore4(out8_0.s89AB, 0, output + outputIndex + 2 * width); \
                vstore2(out8_0.sCD, 0, output + outputIndex + 2 * width + 4); \
            } else if (GID2 + 1 < topHW) { \
                out8_0.s0123 += (DATA_T4)(out_0.s01, out_1.s01); \
                out8_0.s89AB += (DATA_T4)(out_0.s23, out_1.s23); \
                out8_0.s0123 = ACT_VEC_F(DATA_T4, out8_0.s0123); \
                out8_0.s89AB = ACT_VEC_F(DATA_T4, out8_0.s89AB); \
                vstore4(out8_0.s0123, 0, output + outputIndex); \
                vstore4(out8_0.s89AB, 0, output + outputIndex + 2 * width); \
            } else if (GID2 < topHW) { \
                out8_0.s01 += (DATA_T2)(out_0.s01); \
                out8_0.s89 += (DATA_T2)(out_0.s23); \
                out8_0.s01 = ACT_VEC_F(DATA_T2, out8_0.s01); \
                out8_0.s89 = ACT_VEC_F(DATA_T2, out8_0.s89); \
                vstore2(out8_0.s01, 0, output + outputIndex); \
                vstore2(out8_0.s89, 0, output + outputIndex + 2 * width); \
            } \
        }

#define DECONV_8X1_MATRIXTRANS(matrix, matrixTrans, widthTrans, batchId, groupNum) \
        unsigned int heightTrans = get_global_size(1); \
        unsigned int widthTransAlign = get_global_size(2); \
        unsigned int outIndex = batchId * groupNum * heightTrans * widthTransAlign + \
                                get_global_id(0) * widthTransAlign * heightTrans + \
                                get_global_id(1) * widthTransAlign + get_global_id(2); \
        unsigned int inputIndex = batchId * groupNum * heightTrans * widthTrans + \
                                  get_global_id(0) * widthTrans * heightTrans + \
                                  get_global_id(2) * heightTrans + get_global_id(1); \
        if (get_global_id(2) < widthTrans) { \
            matrixTrans[outIndex] = matrix[inputIndex]; \
        } else { \
            matrixTrans[outIndex] = (DATA_T)(0.0f); \
        }

#define DECONVGEMM_1X8_FP16(A, B, output, rowA, colA, rowB, batchId, groupNum) \
        int globalID0 = get_global_id(0); /* group */ \
        int globalID1 = get_global_id(1); /* rowA weight  */ \
        int globalID2 = \
            get_global_id(2) * 8; /* rowB input bottomH*bottomW each thread load 8 line */ \
        if (globalID1 < rowA) { \
            if (globalID2 < rowB) { \
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
                                      get_global_id(0) * rowB * colA + globalID2 * colA;  \
                unsigned int startC = batchId * rowA * rowB * groupNum + \
                                      get_global_id(0) * rowA * rowB + globalID1 * rowB + globalID2; \
                for (int i = 0; i < (colA >> 3); i++) { \
                    weight8 = vload8(0, A + startA + (i << 3)); \
                    input8 = vload8(0, B + startB + (i << 3)); \
                    output8_0 += input8 * weight8; \
                    input8 = vload8(0, B + startB + (i << 3) + colA); \
                    output8_1 += input8 * weight8; \
                    input8 = vload8(0, B + startB + (i << 3) + colA * 2); \
                    output8_2 += input8 * weight8; \
                    input8 = vload8(0, B + startB + (i << 3) + colA * 3); \
                    output8_3 += input8 * weight8; \
                    input8 = vload8(0, B + startB + (i << 3) + colA * 4); \
                    output8_4 += input8 * weight8; \
                    input8 = vload8(0, B + startB + (i << 3) + colA * 5); \
                    output8_5 += input8 * weight8; \
                    input8 = vload8(0, B + startB + (i << 3) + colA * 6); \
                    output8_6 += input8 * weight8; \
                    input8 = vload8(0, B + startB + (i << 3) + colA * 7); \
                    output8_7 += input8 * weight8; \
                } \
                if (globalID2 + 8 <= rowB) { \
                    input8.s0 = dot8(output8_0, one8); \
                    input8.s1 = dot8(output8_1, one8); \
                    input8.s2 = dot8(output8_2, one8); \
                    input8.s3 = dot8(output8_3, one8); \
                    input8.s4 = dot8(output8_4, one8); \
                    input8.s5 = dot8(output8_5, one8); \
                    input8.s6 = dot8(output8_6, one8); \
                    input8.s7 = dot8(output8_7, one8); \
                    vstore8(input8, 0, output + startC); \
                } else { \
                    int num = rowB - globalID2; \
                    if (num == 1) { \
                        output[startC] = dot8(output8_0, one8); \
                    } else if (num == 2) { \
                        output[startC] = dot8(output8_0, one8); \
                        output[startC + 1] = dot8(output8_1, one8); \
                    } else if (num == 3) { \
                        output[startC] = dot8(output8_0, one8); \
                        output[startC + 1] = dot8(output8_1, one8); \
                        output[startC + 2] = dot8(output8_2, one8); \
                    } else if (num == 4) { \
                        output[startC] = dot8(output8_0, one8); \
                        output[startC + 1] = dot8(output8_1, one8); \
                        output[startC + 2] = dot8(output8_2, one8); \
                        output[startC + 3] = dot8(output8_3, one8); \
                    } else if (num == 5) { \
                        output[startC] = dot8(output8_0, one8); \
                        output[startC + 1] = dot8(output8_1, one8); \
                        output[startC + 2] = dot8(output8_2, one8); \
                        output[startC + 3] = dot8(output8_3, one8); \
                        output[startC + 4] = dot8(output8_4, one8); \
                    } else if (num == 6) { \
                        output[startC] = dot8(output8_0, one8); \
                        output[startC + 1] = dot8(output8_1, one8); \
                        output[startC + 2] = dot8(output8_2, one8); \
                        output[startC + 3] = dot8(output8_3, one8); \
                        output[startC + 4] = dot8(output8_4, one8); \
                        output[startC + 5] = dot8(output8_5, one8); \
                    } else if (num == 7) { \
                        output[startC] = dot8(output8_0, one8); \
                        output[startC + 1] = dot8(output8_1, one8); \
                        output[startC + 2] = dot8(output8_2, one8); \
                        output[startC + 3] = dot8(output8_3, one8); \
                        output[startC + 4] = dot8(output8_4, one8); \
                        output[startC + 5] = dot8(output8_5, one8); \
                        output[startC + 6] = dot8(output8_6, one8); \
                    } \
                } \
            } \
        }

#define DECONVGEMM_1X8_FP32(A, B, output, rowA, colA, rowB, batchId, groupNum) \
        int globalID0 = get_global_id(0); /* group */ \
        int globalID1 = get_global_id(1); /* rowA weight  */ \
        int globalID2 = \
            get_global_id(2) * 8; /* rowB input bottomH*bottomW each thread load 8 line */ \
        if (globalID1 < rowA) { \
            if (globalID2 < rowB) { \
                float4 output4_0 = (float4)(0.0f); \
                float4 output4_1 = (float4)(0.0f); \
                float4 output4_2 = (float4)(0.0f); \
                float4 output4_3 = (float4)(0.0f); \
                float4 output4_4 = (float4)(0.0f); \
                float4 output4_5 = (float4)(0.0f); \
                float4 output4_6 = (float4)(0.0f); \
                float4 output4_7 = (float4)(0.0f); \
                float4 weight4; \
                float4 input4; \
                unsigned int startA = get_global_id(0) * rowA * colA + \
                                      globalID1 * colA;  \
                unsigned int startB = batchId * rowB * colA * groupNum + \
                                      get_global_id(0) * rowB * colA + globalID2 * colA;  \
                unsigned int startC = batchId * rowA * rowB * groupNum + \
                                      get_global_id(0) * rowA * rowB + globalID1 * rowB + globalID2; \
                for (int i = 0; i < (colA >> 2); i++) { \
                    weight4 = vload4(0, A + startA + (i << 2)); \
                    input4 = vload4(0, B + startB + (i << 2)); \
                    output4_0 += input4 * weight4; \
                    input4 = vload4(0, B + startB + (i << 2) + colA); \
                    output4_1 += input4 * weight4; \
                    input4 = vload4(0, B + startB + (i << 2) + colA * 2); \
                    output4_2 += input4 * weight4; \
                    input4 = vload4(0, B + startB + (i << 2) + colA * 3); \
                    output4_3 += input4 * weight4; \
                    input4 = vload4(0, B + startB + (i << 2) + colA * 4); \
                    output4_4 += input4 * weight4; \
                    input4 = vload4(0, B + startB + (i << 2) + colA * 5); \
                    output4_5 += input4 * weight4; \
                    input4 = vload4(0, B + startB + (i << 2) + colA * 6); \
                    output4_6 += input4 * weight4; \
                    input4 = vload4(0, B + startB + (i << 2) + colA * 7); \
                    output4_7 += input4 * weight4; \
                } \
                if (globalID2 + 8 <= rowB) { \
                    input4.s0 = output4_0.s0 + output4_0.s1 + output4_0.s2 + output4_0.s3; \
                    input4.s1 = output4_1.s0 + output4_1.s1 + output4_1.s2 + output4_1.s3; \
                    input4.s2 = output4_2.s0 + output4_2.s1 + output4_2.s2 + output4_2.s3; \
                    input4.s3 = output4_3.s0 + output4_3.s1 + output4_3.s2 + output4_3.s3; \
                    vstore4(input4, 0, output + startC); \
                    input4.s0 = output4_4.s0 + output4_4.s1 + output4_4.s2 + output4_4.s3; \
                    input4.s1 = output4_5.s0 + output4_5.s1 + output4_5.s2 + output4_5.s3; \
                    input4.s2 = output4_6.s0 + output4_6.s1 + output4_6.s2 + output4_6.s3; \
                    input4.s3 = output4_7.s0 + output4_7.s1 + output4_7.s2 + output4_7.s3; \
                    vstore4(input4, 0, output + startC + 4); \
                } else { \
                    int num = rowB - globalID2; \
                    if (num == 1) { \
                        output[startC] = output4_0.s0 + output4_0.s1 + output4_0.s2 + output4_0.s3; \
                    } else if (num == 2) { \
                        output[startC] = output4_0.s0 + output4_0.s1 + output4_0.s2 + output4_0.s3; \
                        output[startC + 1] = \
                            output4_1.s0 + output4_1.s1 + output4_1.s2 + output4_1.s3; \
                    } else if (num == 3) { \
                        output[startC] = output4_0.s0 + output4_0.s1 + output4_0.s2 + output4_0.s3; \
                        output[startC + 1] = \
                            output4_1.s0 + output4_1.s1 + output4_1.s2 + output4_1.s3; \
                        output[startC + 2] = \
                            output4_2.s0 + output4_2.s1 + output4_2.s2 + output4_2.s3; \
                    } else if (num == 4) { \
                        output[startC] = output4_0.s0 + output4_0.s1 + output4_0.s2 + output4_0.s3; \
                        output[startC + 1] = \
                            output4_1.s0 + output4_1.s1 + output4_1.s2 + output4_1.s3; \
                        output[startC + 2] = \
                            output4_2.s0 + output4_2.s1 + output4_2.s2 + output4_2.s3; \
                        output[startC + 3] = \
                            output4_3.s0 + output4_3.s1 + output4_3.s2 + output4_3.s3; \
                    } else if (num == 5) { \
                        output[startC] = output4_0.s0 + output4_0.s1 + output4_0.s2 + output4_0.s3; \
                        output[startC + 1] = \
                            output4_1.s0 + output4_1.s1 + output4_1.s2 + output4_1.s3; \
                        output[startC + 2] = \
                            output4_2.s0 + output4_2.s1 + output4_2.s2 + output4_2.s3; \
                        output[startC + 3] = \
                            output4_3.s0 + output4_3.s1 + output4_3.s2 + output4_3.s3; \
                        output[startC + 4] = \
                            output4_4.s0 + output4_4.s1 + output4_4.s2 + output4_4.s3; \
                    } else if (num == 6) { \
                        output[startC] = output4_0.s0 + output4_0.s1 + output4_0.s2 + output4_0.s3; \
                        output[startC + 1] = \
                            output4_1.s0 + output4_1.s1 + output4_1.s2 + output4_1.s3; \
                        output[startC + 2] = \
                            output4_2.s0 + output4_2.s1 + output4_2.s2 + output4_2.s3; \
                        output[startC + 3] = \
                            output4_3.s0 + output4_3.s1 + output4_3.s2 + output4_3.s3; \
                        output[startC + 4] = \
                            output4_4.s0 + output4_4.s1 + output4_4.s2 + output4_4.s3; \
                        output[startC + 5] = \
                            output4_5.s0 + output4_5.s1 + output4_5.s2 + output4_5.s3; \
                    } else if (num == 7) { \
                        output[startC] = output4_0.s0 + output4_0.s1 + output4_0.s2 + output4_0.s3; \
                        output[startC + 1] = \
                            output4_1.s0 + output4_1.s1 + output4_1.s2 + output4_1.s3; \
                        output[startC + 2] = \
                            output4_2.s0 + output4_2.s1 + output4_2.s2 + output4_2.s3; \
                        output[startC + 3] = \
                            output4_3.s0 + output4_3.s1 + output4_3.s2 + output4_3.s3; \
                        output[startC + 4] = \
                            output4_4.s0 + output4_4.s1 + output4_4.s2 + output4_4.s3; \
                        output[startC + 5] = \
                            output4_5.s0 + output4_5.s1 + output4_5.s2 + output4_5.s3; \
                        output[startC + 6] = \
                            output4_6.s0 + output4_6.s1 + output4_6.s2 + output4_6.s3; \
                    } \
                } \
            } \
        }

#define COL2IMG_1X8(input, bias, output, height, width, channels, kernelHeight, kernelWidth, padHeight, padWidth, \
                    strideHeight, strideWidth, height_col, width_col) \
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
                   get_global_id(2)] = ACT_VEC_F(DATA_T, val + bias[biasNum]); \
        }

#define COL2IMG_1X8_OPT(input, bias, output, height, width, channels, kernelHeight, kernelWidth, padHeight, padWidth, \
                        strideHeight, strideWidth, height_col, width_col) \
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
                    int c_col = c_im * height_col * width_col + h_col * width_col + w_col; \
                    val += input[c_col * kernelHeight * kernelWidth + \
                                 (h_im - h_col * strideHeight) * kernelWidth + \
                                 (w_im - w_col * strideWidth)]; \
                } \
            } \
            int biasNum = get_global_id(0) % channels; \
            output[get_global_id(0) * height * width + get_global_id(1) * width + \
                   get_global_id(2)] = ACT_VEC_F(DATA_T, val + bias[biasNum]); \
        }

#define GEMM_DEPTHWISE(input, filter, output, group, input_h, input_w, filter_w, filter_h) \
        int g0 = get_global_id(0); \
        int g1 = get_global_id(1); \
        int g2 = get_global_id(2) * 16; \
        if (g1 < (input_h * input_w) && g2 < (filter_h * filter_w)) { \
            int g = g0 % group; \
            int input_id = g0 * input_h * input_w + g1; \
            int filter_id = g * (filter_h * filter_w) + g2; \
            int output_id = \
                g0 * filter_h * filter_w * input_h * input_w + g1 * filter_h * filter_w + g2; \
            DATA_T16 w_8 = vload16(0, filter + filter_id); \
            DATA_T16 out_8 = input[input_id] * w_8; \
            if (g2 + 15 < filter_h * filter_w) { \
                vstore16(out_8, 0, output + output_id); \
            } else if (g2 + 14 < filter_h * filter_w) { \
                vstore8(out_8.s01234567, 0, output + output_id); \
                vstore4(out_8.s89ab, 0, output + output_id + 8); \
                vstore2(out_8.scd, 0, output + output_id + 12); \
                output[output_id + 14] = out_8.se; \
            } else if (g2 + 13 < filter_h * filter_w) { \
                vstore8(out_8.s01234567, 0, output + output_id); \
                vstore4(out_8.s89ab, 0, output + output_id + 8); \
                vstore2(out_8.scd, 0, output + output_id + 12); \
            } else if (g2 + 12 < filter_h * filter_w) { \
                vstore8(out_8.s01234567, 0, output + output_id); \
                vstore4(out_8.s89ab, 0, output + output_id + 8); \
                output[output_id + 12] = out_8.sc; \
            } else if (g2 + 11 < filter_h * filter_w) { \
                vstore8(out_8.s01234567, 0, output + output_id); \
                vstore4(out_8.s89ab, 0, output + output_id + 8); \
            } else if (g2 + 10 < filter_h * filter_w) { \
                vstore8(out_8.s01234567, 0, output + output_id); \
                vstore2(out_8.s89, 0, output + output_id + 8); \
                output[output_id + 10] = out_8.sa; \
            } else if (g2 + 9 < filter_h * filter_w) { \
                vstore8(out_8.s01234567, 0, output + output_id); \
                vstore2(out_8.s89, 0, output + output_id + 8); \
            } else if (g2 + 8 < filter_h * filter_w) { \
                vstore8(out_8.s01234567, 0, output + output_id); \
                output[output_id + 8] = out_8.s8; \
            } else if (g2 + 7 < filter_h * filter_w) { \
                vstore8(out_8.s01234567, 0, output + output_id); \
            } else if (g2 + 6 < filter_h * filter_w) { \
                vstore4(out_8.s0123, 0, output + output_id); \
                vstore2(out_8.s45, 0, output + output_id + 4); \
                output[output_id + 6] = out_8.s6; \
            } else if (g2 + 5 < filter_h * filter_w) { \
                vstore4(out_8.s0123, 0, output + output_id); \
                vstore2(out_8.s45, 0, output + output_id + 4); \
            } else if (g2 + 4 < filter_h * filter_w) { \
                vstore4(out_8.s0123, 0, output + output_id); \
                output[output_id + 4] = out_8.s4; \
            } else if (g2 + 3 < filter_h * filter_w) { \
                vstore4(out_8.s0123, 0, output + output_id); \
            } else if (g2 + 2 < filter_h * filter_w) { \
                vstore2(out_8.s01, 0, output + output_id); \
                output[output_id + 2] = out_8.s2; \
            } else if (g2 + 1 < filter_h * filter_w) { \
                vstore2(out_8.s01, 0, output + output_id); \
            } else if (g2 < filter_h * filter_w) { \
                output[output_id] = out_8.s0; \
            } \
        }


/********  FP16 KERNELS ********/
#define DATA_T half
#define DATA_T2 half2
#define DATA_T3 half3
#define DATA_T4 half4
#define DATA_T8 half8
#define DATA_T16 half16
ADD_SINGLE_KERNEL(pure_matrix_transpose_FP16, (__global const DATA_T *matrix,
                                             __global DATA_T *matrixTrans) {
    PURE_MATRIX_TRANSPOSE(matrix, matrixTrans)
})

ADD_SINGLE_KERNEL(gemm_makalu_deconv_FP16, (__global const DATA_T *input,
                                          __global const DATA_T *weight,
                                          __global DATA_T *output,
                                          uint alignedCinKK,
                                          uint topChannel,
                                          uint topHW,
                                          uint bottomStep) {
    GEMM_MAKALU_DECONV(input, weight, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(gemm_makalu_deconv_opt_FP16, (__global const DATA_T *input,
                                              __global const DATA_T *weight,
                                              __global DATA_T *output,
                                              uint alignedCinKK,
                                              uint topChannel,
                                              uint topHW,
                                              uint bottomStep,
                                              uint kernelHW) {
    GEMM_MAKALU_DECONV_OPT(input, weight, output, alignedCinKK, topChannel, topHW, bottomStep, kernelHW)
})

ADD_SINGLE_KERNEL(deconv_8x1_matrixTrans_FP16, (__global const DATA_T *matrix,
                                              __global DATA_T *matrixTrans,
                                              unsigned int widthTrans,
                                              unsigned int batchId,
                                              unsigned int groupNum) {
    DECONV_8X1_MATRIXTRANS(matrix, matrixTrans, widthTrans, batchId, groupNum)
})

ADD_SINGLE_KERNEL(deconvGemm_1x8_FP16, (__global const DATA_T *A,
                                      __global const DATA_T *B,
                                      __global DATA_T *output,
                                      unsigned int rowA,
                                      unsigned int colA,
                                      unsigned int rowB,
                                      unsigned int batchId,
                                      unsigned int groupNum) {
    DECONVGEMM_1X8_FP16(A, B, output, rowA, colA, rowB, batchId, groupNum)
})

ADD_SINGLE_KERNEL(gemm_depthwise_FP16, (__global const DATA_T *input,
                                      __global const DATA_T *filter,
                                      __global DATA_T *output,
                                      unsigned int group,
                                      unsigned int input_h,
                                      unsigned int input_w,
                                      unsigned int filter_w,
                                      unsigned int filter_h) {
    GEMM_DEPTHWISE(input, filter, output, group, input_h, input_w, filter_w, filter_h)
})


// kernels without activation func
#define ACT_VEC_F(VEC_T, x) x
ADD_SINGLE_KERNEL(gemm_makalu_deconv_2_times_FP16, (__global const DATA_T *input,
                                                  __global const DATA_T *weight,
                                                  __global const DATA_T *bias,
                                                  __global DATA_T *output,
                                                  uint alignedCinKK,
                                                  uint topChannel,
                                                  uint width,
                                                  uint topHW,
                                                  uint bottomStep) {
    GEMM_MAKALU_DECONV_2_TIMES(input, weight, bias, output, alignedCinKK, topChannel, width, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(col2img_1x8_FP16, (__global const DATA_T *input,
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
    COL2IMG_1X8(input, bias, output, height, width, channels, kernelHeight, kernelWidth, padHeight, padWidth, \
                strideHeight, strideWidth, height_col, width_col)
})

ADD_SINGLE_KERNEL(col2img_1x8_opt_FP16, (__global const DATA_T *input,
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
    COL2IMG_1X8_OPT(input, bias, output, height, width, channels, kernelHeight, kernelWidth, padHeight, padWidth, \
                    strideHeight, strideWidth, height_col, width_col)
})

#undef ACT_VEC_F  // NONE

// apply RELU to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELUgemm_makalu_deconv_2_times_FP16, (__global const DATA_T *input,
                                                  __global const DATA_T *weight,
                                                  __global const DATA_T *bias,
                                                  __global DATA_T *output,
                                                  uint alignedCinKK,
                                                  uint topChannel,
                                                  uint width,
                                                  uint topHW,
                                                  uint bottomStep) {
    GEMM_MAKALU_DECONV_2_TIMES(input, weight, bias, output, alignedCinKK, topChannel, width, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(RELUcol2img_1x8_FP16, (__global const DATA_T *input,
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
    COL2IMG_1X8(input, bias, output, height, width, channels, kernelHeight, kernelWidth, padHeight, padWidth, \
                 strideHeight, strideWidth, height_col, width_col)
})

ADD_SINGLE_KERNEL(RELUcol2img_1x8_opt_FP16, (__global const DATA_T *input,
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
    COL2IMG_1X8_OPT(input, bias, output, height, width, channels, kernelHeight, kernelWidth, padHeight, padWidth, \
                       strideHeight, strideWidth, height_col, width_col)
})

#undef ACT_VEC_F  // RELU

// apply RELU6 to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU6_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELU6gemm_makalu_deconv_2_times_FP16, (__global const DATA_T *input,
                                                  __global const DATA_T *weight,
                                                  __global const DATA_T *bias,
                                                  __global DATA_T *output,
                                                  uint alignedCinKK,
                                                  uint topChannel,
                                                  uint width,
                                                  uint topHW,
                                                  uint bottomStep) {
    GEMM_MAKALU_DECONV_2_TIMES(input, weight, bias, output, alignedCinKK, topChannel, width, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(RELU6col2img_1x8_FP16, (__global const DATA_T *input,
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
    COL2IMG_1X8(input, bias, output, height, width, channels, kernelHeight, kernelWidth, padHeight, padWidth, \
                  strideHeight, strideWidth, height_col, width_col)
})

ADD_SINGLE_KERNEL(RELU6col2img_1x8_opt_FP16, (__global const DATA_T *input,
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
    COL2IMG_1X8_OPT(input, bias, output, height, width, channels, kernelHeight, kernelWidth, padHeight, padWidth, \
                     strideHeight, strideWidth, height_col, width_col)
})

#undef ACT_VEC_F  // RELU6

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
ADD_SINGLE_KERNEL(pure_matrix_transpose_FP32, (__global const DATA_T *matrix,
                                             __global DATA_T *matrixTrans) {
    PURE_MATRIX_TRANSPOSE(matrix, matrixTrans)
})

ADD_SINGLE_KERNEL(gemm_makalu_deconv_FP32, (__global const DATA_T *input,
                                          __global const DATA_T *weight,
                                          __global DATA_T *output,
                                          uint alignedCinKK,
                                          uint topChannel,
                                          uint topHW,
                                          uint bottomStep) {
    GEMM_MAKALU_DECONV(input, weight, output, alignedCinKK, topChannel, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(gemm_makalu_deconv_opt_FP32, (__global const DATA_T *input,
                                              __global const DATA_T *weight,
                                              __global DATA_T *output,
                                              uint alignedCinKK,
                                              uint topChannel,
                                              uint topHW,
                                              uint bottomStep,
                                              uint kernelHW) {
    GEMM_MAKALU_DECONV_OPT(input, weight, output, alignedCinKK, topChannel, topHW, bottomStep, kernelHW)
})

ADD_SINGLE_KERNEL(deconv_8x1_matrixTrans_FP32, (__global const DATA_T *matrix,
                                              __global DATA_T *matrixTrans,
                                              unsigned int widthTrans,
                                              unsigned int batchId,
                                              unsigned int groupNum) {
    DECONV_8X1_MATRIXTRANS(matrix, matrixTrans, widthTrans, batchId, groupNum)
})

ADD_SINGLE_KERNEL(deconvGemm_1x8_FP32, (__global const DATA_T *A,
                                      __global const DATA_T *B,
                                      __global DATA_T *output,
                                      unsigned int rowA,
                                      unsigned int colA,
                                      unsigned int rowB,
                                      unsigned int batchId,
                                      unsigned int groupNum) {
    DECONVGEMM_1X8_FP32(A, B, output, rowA, colA, rowB, batchId, groupNum)
})

ADD_SINGLE_KERNEL(gemm_depthwise_FP32, (__global const DATA_T *input,
                                      __global const DATA_T *filter,
                                      __global DATA_T *output,
                                      unsigned int group,
                                      unsigned int input_h,
                                      unsigned int input_w,
                                      unsigned int filter_w,
                                      unsigned int filter_h) {
    GEMM_DEPTHWISE(input, filter, output, group, input_h, input_w, filter_w, filter_h)
})


// kernels without activation func
#define ACT_VEC_F(VEC_T, x) x
ADD_SINGLE_KERNEL(gemm_makalu_deconv_2_times_FP32, (__global const DATA_T *input,
                                                  __global const DATA_T *weight,
                                                  __global const DATA_T *bias,
                                                  __global DATA_T *output,
                                                  uint alignedCinKK,
                                                  uint topChannel,
                                                  uint width,
                                                  uint topHW,
                                                  uint bottomStep) {
    GEMM_MAKALU_DECONV_2_TIMES(input, weight, bias, output, alignedCinKK, topChannel, width, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(col2img_1x8_FP32, (__global const DATA_T *input,
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
    COL2IMG_1X8(input, bias, output, height, width, channels, kernelHeight, kernelWidth, padHeight, padWidth, \
                strideHeight, strideWidth, height_col, width_col)
})

ADD_SINGLE_KERNEL(col2img_1x8_opt_FP32, (__global const DATA_T *input,
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
    COL2IMG_1X8_OPT(input, bias, output, height, width, channels, kernelHeight, kernelWidth, padHeight, \
                    padWidth, strideHeight, strideWidth, height_col, width_col)
})

#undef ACT_VEC_F  // NONE

// apply RELU to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELUgemm_makalu_deconv_2_times_FP32, (__global const DATA_T *input,
                                                  __global const DATA_T *weight,
                                                  __global const DATA_T *bias,
                                                  __global DATA_T *output,
                                                  uint alignedCinKK,
                                                  uint topChannel,
                                                  uint width,
                                                  uint topHW,
                                                  uint bottomStep) {
    GEMM_MAKALU_DECONV_2_TIMES(input, weight, bias, output, alignedCinKK, topChannel, width, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(RELUcol2img_1x8_FP32, (__global const DATA_T *input,
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
    COL2IMG_1X8(input, bias, output, height, width, channels, kernelHeight, kernelWidth, padHeight, padWidth, \
                strideHeight, strideWidth, height_col, width_col)
})

ADD_SINGLE_KERNEL(RELUcol2img_1x8_opt_FP32, (__global const DATA_T *input,
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
    COL2IMG_1X8_OPT(input, bias, output, height, width, channels, kernelHeight, kernelWidth, padHeight, padWidth, \
                    strideHeight, strideWidth, height_col, width_col)
})

#undef ACT_VEC_F  // RELU

// apply RELU6 to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU6_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELU6gemm_makalu_deconv_2_times_FP32, (__global const DATA_T *input,
                                                  __global const DATA_T *weight,
                                                  __global const DATA_T *bias,
                                                  __global DATA_T *output,
                                                  uint alignedCinKK,
                                                  uint topChannel,
                                                  uint width,
                                                  uint topHW,
                                                  uint bottomStep) {
    GEMM_MAKALU_DECONV_2_TIMES(input, weight, bias, output, alignedCinKK, topChannel, width, topHW, bottomStep)
})

ADD_SINGLE_KERNEL(RELU6col2img_1x8_FP32, (__global const DATA_T *input,
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
    COL2IMG_1X8(input, bias, output, height, width, channels, kernelHeight, kernelWidth, padHeight, padWidth, \
                 strideHeight, strideWidth, height_col, width_col)
})

ADD_SINGLE_KERNEL(RELU6col2img_1x8_opt_FP32, (__global const DATA_T *input,
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
    COL2IMG_1X8_OPT(input, bias, output, height, width, channels, kernelHeight, kernelWidth, padHeight, padWidth, \
                    strideHeight, strideWidth, height_col, width_col)
})

#undef ACT_VEC_F  // RELU6

#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // float

}  // namespace gpu
}  // namespace ud
}  // namespace enn
