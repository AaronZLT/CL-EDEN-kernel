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
 * @file    ConvolutionWINO4x4_5x5Kernels.cpp
 * @brief
 * @details
 * @version
 */

#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"
namespace enn{
namespace ud {
namespace gpu {


#define WINO5X5_WEIGHT_TM_NCHW_OPT_4X16(wt, wt_tm, outch, inch, aligned_outch, aligned_inch, aligned_pixel) \
    int inc = get_global_id(0); \
    int outc = get_global_id(1); \
    int outch_aligned = (outch + aligned_outch - 1) / aligned_outch * aligned_outch; \
    int inch_aligned = (inch + aligned_inch - 1) / aligned_inch * aligned_inch; \
    if (inc >= inch_aligned || outc >= outch_aligned) { \
            return; \
    } \
    int dst_base = outc / aligned_outch * \
                        (aligned_outch * aligned_pixel * aligned_inch) + \
                    outc % aligned_outch; \
    dst_base += \
        inc / aligned_inch * (outch_aligned * aligned_pixel * aligned_inch) + \
        inc % aligned_inch * aligned_outch; \
    int p_dst_base = dst_base; \
    { \
      if (inc >= inch || outc >= outch) { \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              dst_base = p_dst_base + aligned_pixel * outch_aligned * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              dst_base = p_dst_base + 2 * aligned_pixel * outch_aligned * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              dst_base = p_dst_base + 3 * aligned_pixel * outch_aligned * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              dst_base = p_dst_base + 4 * aligned_pixel * outch_aligned * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              dst_base = p_dst_base + 5 * aligned_pixel * outch_aligned * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              dst_base = p_dst_base + 6 * aligned_pixel * outch_aligned * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              dst_base = p_dst_base + 7 * aligned_pixel * outch_aligned * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              dst_base = p_dst_base + 8 * aligned_pixel * outch_aligned * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              dst_base = p_dst_base + 9 * aligned_pixel * outch_aligned * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              dst_base = p_dst_base + 10 * aligned_pixel * outch_aligned * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              dst_base = p_dst_base + 11 * aligned_pixel * outch_aligned * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              dst_base = p_dst_base + 12 * aligned_pixel * outch_aligned * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              dst_base = p_dst_base + 13 * aligned_pixel * outch_aligned * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              dst_base = p_dst_base + 14 * aligned_pixel * outch_aligned * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              dst_base = p_dst_base + 15 * aligned_pixel * outch_aligned * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * aligned_inch; \
              return; \
      } \
      int src_base = (outc * inch + inc) * 25; \
      DATA_T src_in[25]; \
      for (int i = 0; i < 25; ++i) { \
        src_in[i] = wt[src_base + i]; \
      } \
      DATA_T tmp_r0[5]; \
      DATA_T tmp_r1[5]; \
      DATA_T tmp_r2[5]; \
      DATA_T tmp_r3[5]; \
      DATA_T tmp_r4[5]; \
      DATA_T tmp_r5[5]; \
      DATA_T tmp_r6[5]; \
      DATA_T tmp_r7[5]; \
      tmp_r0[0] = src_in[0]; \
      tmp_r0[1] = src_in[1]; \
      tmp_r0[2] = src_in[2]; \
      tmp_r0[3] = src_in[3]; \
      tmp_r0[4] = src_in[4]; \
      tmp_r1[0] = src_in[0] + 0.707107f * src_in[5] + 0.5f * src_in[10] + \
                  0.355553f * src_in[15] + 0.25f * src_in[20]; \
      tmp_r1[1] = src_in[1] + 0.707107f * src_in[6] + 0.5f * src_in[11] + \
                  0.355553f * src_in[16] + 0.25f * src_in[21]; \
      tmp_r1[2] = src_in[2] + 0.707107f * src_in[7] + 0.5f * src_in[12] + \
                  0.355553f * src_in[17] + 0.25f * src_in[22]; \
      tmp_r1[3] = src_in[3] + 0.707107f * src_in[8] + 0.5f * src_in[13] + \
                  0.355553f * src_in[18] + 0.25f * src_in[23]; \
      tmp_r1[4] = src_in[4] + 0.707107f * src_in[9] + 0.5f * src_in[14] + \
                  0.355553f * src_in[19] + 0.25f * src_in[24]; \
      tmp_r2[0] = src_in[0] - 0.707107f * src_in[5] + 0.5f * src_in[10] - \
                  0.355553f * src_in[15] + 0.25f * src_in[20]; \
      tmp_r2[1] = src_in[1] - 0.707107f * src_in[6] + 0.5f * src_in[11] - \
                  0.355553f * src_in[16] + 0.25f * src_in[21]; \
      tmp_r2[2] = src_in[2] - 0.707107f * src_in[7] + 0.5f * src_in[12] - \
                  0.355553f * src_in[17] + 0.25f * src_in[22]; \
      tmp_r2[3] = src_in[3] - 0.707107f * src_in[8] + 0.5f * src_in[13] - \
                  0.355553f * src_in[18] + 0.25f * src_in[23]; \
      tmp_r2[4] = src_in[4] - 0.707107f * src_in[9] + 0.5f * src_in[14] - \
                  0.355553f * src_in[19] + 0.25f * src_in[24]; \
      tmp_r3[0] = src_in[0] + 1.414214f * src_in[5] + 2.0f * src_in[10] + \
                  2.828427f * src_in[15] + 4.0f * src_in[20]; \
      tmp_r3[1] = src_in[1] + 1.414214f * src_in[6] + 2.0f * src_in[11] + \
                  2.828427f * src_in[16] + 4.0f * src_in[21]; \
      tmp_r3[2] = src_in[2] + 1.414214f * src_in[7] + 2.0f * src_in[12] + \
                  2.828427f * src_in[17] + 4.0f * src_in[22]; \
      tmp_r3[3] = src_in[3] + 1.414214f * src_in[8] + 2.0f * src_in[13] + \
                  2.828427f * src_in[18] + 4.0f * src_in[23]; \
      tmp_r3[4] = src_in[4] + 1.414214f * src_in[9] + 2.0f * src_in[14] + \
                  2.828427f * src_in[19] + 4.0f * src_in[24]; \
      tmp_r4[0] = src_in[0] - 1.414214f * src_in[5] + 2.0f * src_in[10] - \
                  2.828427f * src_in[15] + 4.0f * src_in[20]; \
      tmp_r4[1] = src_in[1] - 1.414214f * src_in[6] + 2.0f * src_in[11] - \
                  2.828427f * src_in[16] + 4.0f * src_in[21]; \
      tmp_r4[2] = src_in[2] - 1.414214f * src_in[7] + 2.0f * src_in[12] - \
                  2.828427f * src_in[17] + 4.0f * src_in[22]; \
      tmp_r4[3] = src_in[3] - 1.414214f * src_in[8] + 2.0f * src_in[13] - \
                  2.828427f * src_in[18] + 4.0f * src_in[23]; \
      tmp_r4[4] = src_in[4] - 1.414214f * src_in[9] + 2.0f * src_in[14] - \
                  2.828427f * src_in[19] + 4.0f * src_in[24]; \
      tmp_r5[0] = src_in[0] + 2.121320f * src_in[5] + 4.5f * src_in[10] + \
                  9.545940f * src_in[15] + 20.25f * src_in[20]; \
      tmp_r5[1] = src_in[1] + 2.121320f * src_in[6] + 4.5f * src_in[11] + \
                  9.545940f * src_in[16] + 20.25f * src_in[21]; \
      tmp_r5[2] = src_in[2] + 2.121320f * src_in[7] + 4.5f * src_in[12] + \
                  9.545940f * src_in[17] + 20.25f * src_in[22]; \
      tmp_r5[3] = src_in[3] + 2.121320f * src_in[8] + 4.5f * src_in[13] + \
                  9.545940f * src_in[18] + 20.25f * src_in[23]; \
      tmp_r5[4] = src_in[4] + 2.121320f * src_in[9] + 4.5f * src_in[14] + \
                  9.545940f * src_in[19] + 20.25f * src_in[24]; \
      tmp_r6[0] = src_in[0] - 2.121320f * src_in[5] + 4.5f * src_in[10] - \
                  9.545940f * src_in[15] + 20.25f * src_in[20]; \
      tmp_r6[1] = src_in[1] - 2.121320f * src_in[6] + 4.5f * src_in[11] - \
                  9.545940f * src_in[16] + 20.25f * src_in[21]; \
      tmp_r6[2] = src_in[2] - 2.121320f * src_in[7] + 4.5f * src_in[12] - \
                  9.545940f * src_in[17] + 20.25f * src_in[22]; \
      tmp_r6[3] = src_in[3] - 2.121320f * src_in[8] + 4.5f * src_in[13] - \
                  9.545940f * src_in[18] + 20.25f * src_in[23]; \
      tmp_r6[4] = src_in[4] - 2.121320f * src_in[9] + 4.5f * src_in[14] - \
                  9.545940f * src_in[19] + 20.25f * src_in[24]; \
      tmp_r7[0] = src_in[20]; \
      tmp_r7[1] = src_in[21]; \
      tmp_r7[2] = src_in[22]; \
      tmp_r7[3] = src_in[23]; \
      tmp_r7[4] = src_in[24]; \
      DATA_T dst_r0[8]; \
      DATA_T dst_r1[8]; \
      DATA_T dst_r2[8]; \
      DATA_T dst_r3[8]; \
      DATA_T dst_r4[8]; \
      DATA_T dst_r5[8]; \
      DATA_T dst_r6[8]; \
      DATA_T dst_r7[8]; \
      dst_r0[0] = tmp_r0[0]; \
      dst_r0[1] = tmp_r0[0] + 0.707107f * tmp_r0[1] + 0.5f * tmp_r0[2] + \
                  0.353553f * tmp_r0[3] + 0.25f * tmp_r0[4]; \
      dst_r0[2] = tmp_r0[0] - 0.707107f * tmp_r0[1] + 0.5f * tmp_r0[2] - \
                  0.353553f * tmp_r0[3] + 0.25f * tmp_r0[4]; \
      dst_r0[3] = tmp_r0[0] + 1.414214f * tmp_r0[1] + 2.0f * tmp_r0[2] + \
                  2.828427f * tmp_r0[3] + 4.0f * tmp_r0[4]; \
      dst_r0[4] = tmp_r0[0] - 1.414214f * tmp_r0[1] + 2.0f * tmp_r0[2] - \
                  2.828427f * tmp_r0[3] + 4.0f * tmp_r0[4]; \
      dst_r0[5] = tmp_r0[0] + 2.121320f * tmp_r0[1] + 4.5f * tmp_r0[2] + \
                  9.545940f * tmp_r0[3] + 20.25f * tmp_r0[4]; \
      dst_r0[6] = tmp_r0[0] - 2.121320f * tmp_r0[1] + 4.5f * tmp_r0[2] - \
                  9.545940f * tmp_r0[3] + 20.25f * tmp_r0[4]; \
      dst_r0[7] = tmp_r0[4]; \
      dst_r1[0] = tmp_r1[0]; \
      dst_r1[1] = tmp_r1[0] + 0.707107f * tmp_r1[1] + 0.5f * tmp_r1[2] + \
                  0.353553f * tmp_r1[3] + 0.25f * tmp_r1[4]; \
      dst_r1[2] = tmp_r1[0] - 0.707107f * tmp_r1[1] + 0.5f * tmp_r1[2] - \
                  0.353553f * tmp_r1[3] + 0.25f * tmp_r1[4]; \
      dst_r1[3] = tmp_r1[0] + 1.414214f * tmp_r1[1] + 2.0f * tmp_r1[2] + \
                  2.828427f * tmp_r1[3] + 4.0f * tmp_r1[4]; \
      dst_r1[4] = tmp_r1[0] - 1.414214f * tmp_r1[1] + 2.0f * tmp_r1[2] - \
                  2.828427f * tmp_r1[3] + 4.0f * tmp_r1[4]; \
      dst_r1[5] = tmp_r1[0] + 2.121320f * tmp_r1[1] + 4.5f * tmp_r1[2] + \
                  9.545940f * tmp_r1[3] + 20.25f * tmp_r1[4]; \
      dst_r1[6] = tmp_r1[0] - 2.121320f * tmp_r1[1] + 4.5f * tmp_r1[2] - \
                  9.545940f * tmp_r1[3] + 20.25f * tmp_r1[4]; \
      dst_r1[7] = tmp_r1[4]; \
      dst_r2[0] = tmp_r2[0]; \
      dst_r2[1] = tmp_r2[0] + 0.707107f * tmp_r2[1] + 0.5f * tmp_r2[2] + \
                  0.353553f * tmp_r2[3] + 0.25f * tmp_r2[4]; \
      dst_r2[2] = tmp_r2[0] - 0.707107f * tmp_r2[1] + 0.5f * tmp_r2[2] - \
                  0.353553f * tmp_r2[3] + 0.25f * tmp_r2[4]; \
      dst_r2[3] = tmp_r2[0] + 1.414214f * tmp_r2[1] + 2.0f * tmp_r2[2] + \
                  2.828427f * tmp_r2[3] + 4.0f * tmp_r2[4]; \
      dst_r2[4] = tmp_r2[0] - 1.414214f * tmp_r2[1] + 2.0f * tmp_r2[2] - \
                  2.828427f * tmp_r2[3] + 4.0f * tmp_r2[4]; \
      dst_r2[5] = tmp_r2[0] + 2.121320f * tmp_r2[1] + 4.5f * tmp_r2[2] + \
                  9.545940f * tmp_r2[3] + 20.25f * tmp_r2[4]; \
      dst_r2[6] = tmp_r2[0] - 2.121320f * tmp_r2[1] + 4.5f * tmp_r2[2] - \
                  9.545940f * tmp_r2[3] + 20.25f * tmp_r2[4]; \
      dst_r2[7] = tmp_r2[4]; \
      dst_r3[0] = tmp_r3[0]; \
      dst_r3[1] = tmp_r3[0] + 0.707107f * tmp_r3[1] + 0.5f * tmp_r3[2] + \
                  0.353553f * tmp_r3[3] + 0.25f * tmp_r3[4]; \
      dst_r3[2] = tmp_r3[0] - 0.707107f * tmp_r3[1] + 0.5f * tmp_r3[2] - \
                  0.353553f * tmp_r3[3] + 0.25f * tmp_r3[4]; \
      dst_r3[3] = tmp_r3[0] + 1.414214f * tmp_r3[1] + 2.0f * tmp_r3[2] + \
                  2.828427f * tmp_r3[3] + 4.0f * tmp_r3[4]; \
      dst_r3[4] = tmp_r3[0] - 1.414214f * tmp_r3[1] + 2.0f * tmp_r3[2] - \
                  2.828427f * tmp_r3[3] + 4.0f * tmp_r3[4]; \
      dst_r3[5] = tmp_r3[0] + 2.121320f * tmp_r3[1] + 4.5f * tmp_r3[2] + \
                  9.545940f * tmp_r3[3] + 20.25f * tmp_r3[4]; \
      dst_r3[6] = tmp_r3[0] - 2.121320f * tmp_r3[1] + 4.5f * tmp_r3[2] - \
                  9.545940f * tmp_r3[3] + 20.25f * tmp_r3[4]; \
      dst_r3[7] = tmp_r3[4]; \
      dst_r4[0] = tmp_r4[0]; \
      dst_r4[1] = tmp_r4[0] + 0.707107f * tmp_r4[1] + 0.5f * tmp_r4[2] + \
                  0.353553f * tmp_r4[3] + 0.25f * tmp_r4[4]; \
      dst_r4[2] = tmp_r4[0] - 0.707107f * tmp_r4[1] + 0.5f * tmp_r4[2] - \
                  0.353553f * tmp_r4[3] + 0.25f * tmp_r4[4]; \
      dst_r4[3] = tmp_r4[0] + 1.414214f * tmp_r4[1] + 2.0f * tmp_r4[2] + \
                  2.828427f * tmp_r4[3] + 4.0f * tmp_r4[4]; \
      dst_r4[4] = tmp_r4[0] - 1.414214f * tmp_r4[1] + 2.0f * tmp_r4[2] - \
                  2.828427f * tmp_r4[3] + 4.0f * tmp_r4[4]; \
      dst_r4[5] = tmp_r4[0] + 2.121320f * tmp_r4[1] + 4.5f * tmp_r4[2] + \
                  9.545940f * tmp_r4[3] + 20.25f * tmp_r4[4]; \
      dst_r4[6] = tmp_r4[0] - 2.121320f * tmp_r4[1] + 4.5f * tmp_r4[2] - \
                  9.545940f * tmp_r4[3] + 20.25f * tmp_r4[4]; \
      dst_r4[7] = tmp_r4[4]; \
      dst_r5[0] = tmp_r5[0]; \
      dst_r5[1] = tmp_r5[0] + 0.707107f * tmp_r5[1] + 0.5f * tmp_r5[2] + \
                  0.353553f * tmp_r5[3] + 0.25f * tmp_r5[4]; \
      dst_r5[2] = tmp_r5[0] - 0.707107f * tmp_r5[1] + 0.5f * tmp_r5[2] - \
                  0.353553f * tmp_r5[3] + 0.25f * tmp_r5[4]; \
      dst_r5[3] = tmp_r5[0] + 1.414214f * tmp_r5[1] + 2.0f * tmp_r5[2] + \
                  2.828427f * tmp_r5[3] + 4.0f * tmp_r5[4]; \
      dst_r5[4] = tmp_r5[0] - 1.414214f * tmp_r5[1] + 2.0f * tmp_r5[2] - \
                  2.828427f * tmp_r5[3] + 4.0f * tmp_r5[4]; \
      dst_r5[5] = tmp_r5[0] + 2.121320f * tmp_r5[1] + 4.5f * tmp_r5[2] + \
                  9.545940f * tmp_r5[3] + 20.25f * tmp_r5[4]; \
      dst_r5[6] = tmp_r5[0] - 2.121320f * tmp_r5[1] + 4.5f * tmp_r5[2] - \
                  9.545940f * tmp_r5[3] + 20.25f * tmp_r5[4]; \
      dst_r5[7] = tmp_r5[4]; \
      dst_r6[0] = tmp_r6[0]; \
      dst_r6[1] = tmp_r6[0] + 0.707107f * tmp_r6[1] + 0.5f * tmp_r6[2] + \
                  0.353553f * tmp_r6[3] + 0.25f * tmp_r6[4]; \
      dst_r6[2] = tmp_r6[0] - 0.707107f * tmp_r6[1] + 0.5f * tmp_r6[2] - \
                  0.353553f * tmp_r6[3] + 0.25f * tmp_r6[4]; \
      dst_r6[3] = tmp_r6[0] + 1.414214f * tmp_r6[1] + 2.0f * tmp_r6[2] + \
                  2.828427f * tmp_r6[3] + 4.0f * tmp_r6[4]; \
      dst_r6[4] = tmp_r6[0] - 1.414214f * tmp_r6[1] + 2.0f * tmp_r6[2] - \
                  2.828427f * tmp_r6[3] + 4.0f * tmp_r6[4]; \
      dst_r6[5] = tmp_r6[0] + 2.121320f * tmp_r6[1] + 4.5f * tmp_r6[2] + \
                  9.545940f * tmp_r6[3] + 20.25f * tmp_r6[4]; \
      dst_r6[6] = tmp_r6[0] - 2.121320f * tmp_r6[1] + 4.5f * tmp_r6[2] - \
                  9.545940f * tmp_r6[3] + 20.25f * tmp_r6[4]; \
      dst_r6[7] = tmp_r6[4]; \
      dst_r7[0] = tmp_r7[0]; \
      dst_r7[1] = tmp_r7[0] + 0.707107f * tmp_r7[1] + 0.5f * tmp_r7[2] + \
                  0.353553f * tmp_r7[3] + 0.25f * tmp_r7[4]; \
      dst_r7[2] = tmp_r7[0] - 0.707107f * tmp_r7[1] + 0.5f * tmp_r7[2] - \
                  0.353553f * tmp_r7[3] + 0.25f * tmp_r7[4]; \
      dst_r7[3] = tmp_r7[0] + 1.414214f * tmp_r7[1] + 2.0f * tmp_r7[2] + \
                  2.828427f * tmp_r7[3] + 4.0f * tmp_r7[4]; \
      dst_r7[4] = tmp_r7[0] - 1.414214f * tmp_r7[1] + 2.0f * tmp_r7[2] - \
                  2.828427f * tmp_r7[3] + 4.0f * tmp_r7[4]; \
      dst_r7[5] = tmp_r7[0] + 2.121320f * tmp_r7[1] + 4.5f * tmp_r7[2] + \
                  9.545940f * tmp_r7[3] + 20.25f * tmp_r7[4]; \
      dst_r7[6] = tmp_r7[0] - 2.121320f * tmp_r7[1] + 4.5f * tmp_r7[2] - \
                  9.545940f * tmp_r7[3] + 20.25f * tmp_r7[4]; \
      dst_r7[7] = tmp_r7[4]; \
      wt_tm[dst_base] = dst_r0[0]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r0[1]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r0[2]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r0[3]; dst_base += aligned_outch * aligned_inch; \
      dst_base = p_dst_base + 1 * aligned_pixel * outch_aligned * inch_aligned; \
      wt_tm[dst_base] = dst_r0[4]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r0[5]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r0[6]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r0[7]; dst_base += aligned_outch * aligned_inch; \
      dst_base = p_dst_base + 2 * aligned_pixel * outch_aligned * inch_aligned; \
      wt_tm[dst_base] = dst_r1[0]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r1[1]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r1[2]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r1[3]; dst_base += aligned_outch * aligned_inch; \
      dst_base = p_dst_base + 3 * aligned_pixel * outch_aligned * inch_aligned; \
      wt_tm[dst_base] = dst_r1[4]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r1[5]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r1[6]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r1[7]; dst_base += aligned_outch * aligned_inch; \
      dst_base = p_dst_base + 4 * aligned_pixel * outch_aligned * inch_aligned; \
      wt_tm[dst_base] = dst_r2[0]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r2[1]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r2[2]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r2[3]; dst_base += aligned_outch * aligned_inch; \
      dst_base = p_dst_base + 5 * aligned_pixel * outch_aligned * inch_aligned; \
      wt_tm[dst_base] = dst_r2[4]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r2[5]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r2[6]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r2[7]; dst_base += aligned_outch * aligned_inch; \
      dst_base = p_dst_base + 6 * aligned_pixel * outch_aligned * inch_aligned; \
      wt_tm[dst_base] = dst_r3[0]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r3[1]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r3[2]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r3[3]; dst_base += aligned_outch * aligned_inch; \
      dst_base = p_dst_base + 7 * aligned_pixel * outch_aligned * inch_aligned; \
      wt_tm[dst_base] = dst_r3[4]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r3[5]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r3[6]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r3[7]; dst_base += aligned_outch * aligned_inch; \
      dst_base = p_dst_base + 8 * aligned_pixel * outch_aligned * inch_aligned; \
      wt_tm[dst_base] = dst_r4[0]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r4[1]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r4[2]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r4[3]; dst_base += aligned_outch * aligned_inch; \
      dst_base = p_dst_base + 9 * aligned_pixel * outch_aligned * inch_aligned; \
      wt_tm[dst_base] = dst_r4[4]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r4[5]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r4[6]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r4[7]; dst_base += aligned_outch * aligned_inch; \
      dst_base = p_dst_base + 10 * aligned_pixel * outch_aligned * inch_aligned; \
      wt_tm[dst_base] = dst_r5[0]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r5[1]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r5[2]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r5[3]; dst_base += aligned_outch * aligned_inch; \
      dst_base = p_dst_base + 11 * aligned_pixel * outch_aligned * inch_aligned; \
      wt_tm[dst_base] = dst_r5[4]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r5[5]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r5[6]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r5[7]; dst_base += aligned_outch * aligned_inch; \
      dst_base = p_dst_base + 12 * aligned_pixel * outch_aligned * inch_aligned; \
      wt_tm[dst_base] = dst_r6[0]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r6[1]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r6[2]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r6[3]; dst_base += aligned_outch * aligned_inch; \
      dst_base = p_dst_base + 13 * aligned_pixel * outch_aligned * inch_aligned; \
      wt_tm[dst_base] = dst_r6[4]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r6[5]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r6[6]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r6[7]; dst_base += aligned_outch * aligned_inch; \
      dst_base = p_dst_base + 14 * aligned_pixel * outch_aligned * inch_aligned; \
      wt_tm[dst_base] = dst_r7[0]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r7[1]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r7[2]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r7[3]; dst_base += aligned_outch * aligned_inch; \
      dst_base = p_dst_base + 15 * aligned_pixel * outch_aligned * inch_aligned; \
      wt_tm[dst_base] = dst_r7[4]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r7[5]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r7[6]; dst_base += aligned_outch * aligned_inch; \
      wt_tm[dst_base] = dst_r7[7]; dst_base += aligned_outch * aligned_inch; \
    }

#define WINO5X5_INPUT_TM_NCHW_OPT_4X16(src_data, dst_data, src_n, src_c, src_h, src_w, dst_n, dst_c, dst_h, dst_w, \
                                       padL, padT, tiles_total, coalescing_feature_height_, tiles_x, aligned_inch, \
                                       aligned_pixel) \
    int w_index = get_global_id(0) * 2 % tiles_x; \
    int h_index = get_global_id(0) * 2 / tiles_x; \
    int p_index = get_global_id(1) * 2; \
    int inch_aligned = (src_c + aligned_inch - 1) / aligned_inch * aligned_inch; \
    int c_index = get_global_id(2) % inch_aligned; \
    int n_index = get_global_id(2) / inch_aligned; \
    int aligned_tile = dst_w; \
    if (get_global_id(0) * 2 >= aligned_tile || c_index >= dst_c) { \
            return; \
    } \
    int four_tiles = 2; \
    int dst_base = (n_index * dst_c * dst_h * dst_w); \
    int dst_tile = dst_base + \
                    (get_global_id(0) * 2) / coalescing_feature_height_ * \
                        coalescing_feature_height_ * inch_aligned * \
                        aligned_pixel; \
    dst_tile += c_index / aligned_inch * aligned_inch * \
                    coalescing_feature_height_ * aligned_pixel + \
                c_index % aligned_inch * 2; \
    dst_tile += (get_global_id(0) * 2 % coalescing_feature_height_) / 2 * 2 * \
                aligned_inch; \
    dst_tile += (get_global_id(0) * 2 % coalescing_feature_height_) % 2; \
    dst_tile += p_index * 8 * inch_aligned * aligned_tile; \
    int p_dst_tile = dst_tile; \
    if (c_index >= src_c || get_global_id(0) * 2 >= tiles_total) { \
      DATA_T2 zero2 = (DATA_T2)(0.0f, 0.0f); \
      vstore2(zero2, 0, dst_data + dst_tile); dst_tile += aligned_inch * coalescing_feature_height_; \
      vstore2(zero2, 0, dst_data + dst_tile); dst_tile += aligned_inch * coalescing_feature_height_; \
      vstore2(zero2, 0, dst_data + dst_tile); dst_tile += aligned_inch * coalescing_feature_height_; \
      vstore2(zero2, 0, dst_data + dst_tile); dst_tile += aligned_inch * coalescing_feature_height_; \
      dst_tile = p_dst_tile + aligned_pixel * inch_aligned * aligned_tile; \
      vstore2(zero2, 0, dst_data + dst_tile); dst_tile += aligned_inch * coalescing_feature_height_; \
      vstore2(zero2, 0, dst_data + dst_tile); dst_tile += aligned_inch * coalescing_feature_height_; \
      vstore2(zero2, 0, dst_data + dst_tile); dst_tile += aligned_inch * coalescing_feature_height_; \
      vstore2(zero2, 0, dst_data + dst_tile); dst_tile += aligned_inch * coalescing_feature_height_; \
      dst_tile = p_dst_tile + 2 * aligned_pixel * inch_aligned * aligned_tile; \
      vstore2(zero2, 0, dst_data + dst_tile); dst_tile += aligned_inch * coalescing_feature_height_; \
      vstore2(zero2, 0, dst_data + dst_tile); dst_tile += aligned_inch * coalescing_feature_height_; \
      vstore2(zero2, 0, dst_data + dst_tile); dst_tile += aligned_inch * coalescing_feature_height_; \
      vstore2(zero2, 0, dst_data + dst_tile); dst_tile += aligned_inch * coalescing_feature_height_; \
      dst_tile = p_dst_tile + 3 * aligned_pixel * inch_aligned * aligned_tile; \
      vstore2(zero2, 0, dst_data + dst_tile); dst_tile += aligned_inch * coalescing_feature_height_; \
      vstore2(zero2, 0, dst_data + dst_tile); dst_tile += aligned_inch * coalescing_feature_height_; \
      vstore2(zero2, 0, dst_data + dst_tile); dst_tile += aligned_inch * coalescing_feature_height_; \
      vstore2(zero2, 0, dst_data + dst_tile); dst_tile += aligned_inch * coalescing_feature_height_; \
      return; \
      } \
    int in_w_start = w_index * 4 - padL; \
    int in_h_start = h_index * 4 - padT; \
    int in_base = ((n_index * src_c + c_index) * src_h + in_h_start) * src_w + \
                  in_w_start; \
    DATA_T bt_ar[8]; \
    DATA_T8 t0 = vload8(0, Bt8x8 + p_index * 8); \
    bt_ar[0] = t0.s0; \
    bt_ar[1] = t0.s1; \
    bt_ar[2] = t0.s2; \
    bt_ar[3] = t0.s3; \
    bt_ar[4] = t0.s4; \
    bt_ar[5] = t0.s5; \
    bt_ar[6] = t0.s6; \
    bt_ar[7] = t0.s7; \
    DATA_T bt_br[8]; \
    DATA_T8 t1 = vload8(0, Bt8x8 + (p_index + 1) * 8); \
    bt_br[0] = t1.s0; \
    bt_br[1] = t1.s1; \
    bt_br[2] = t1.s2; \
    bt_br[3] = t1.s3; \
    bt_br[4] = t1.s4; \
    bt_br[5] = t1.s5; \
    bt_br[6] = t1.s6; \
    bt_br[7] = t1.s7; \
    DATA_T2 I0 = (DATA_T2)(0.0f); \
    DATA_T2 I1 = (DATA_T2)(0.0f); \
    DATA_T2 I2 = (DATA_T2)(0.0f); \
    DATA_T2 I3 = (DATA_T2)(0.0f); \
    DATA_T2 I4 = (DATA_T2)(0.0f); \
    DATA_T2 I5 = (DATA_T2)(0.0f); \
    DATA_T2 I6 = (DATA_T2)(0.0f); \
    DATA_T2 I7 = (DATA_T2)(0.0f); \
    DATA_T2 J0 = (DATA_T2)(0.0f); \
    DATA_T2 J1 = (DATA_T2)(0.0f); \
    DATA_T2 J2 = (DATA_T2)(0.0f); \
    DATA_T2 J3 = (DATA_T2)(0.0f); \
    DATA_T2 J4 = (DATA_T2)(0.0f); \
    DATA_T2 J5 = (DATA_T2)(0.0f); \
    DATA_T2 J6 = (DATA_T2)(0.0f); \
    DATA_T2 J7 = (DATA_T2)(0.0f); \
    DATA_T16 row0 = (DATA_T16)(0.0f); \
    DATA_T16 row1 = (DATA_T16)(0.0f); \
    if (in_h_start >= 0 && in_h_start < src_h) { \
        row0.lo = vload8(0, src_data + in_base); \
        row0.hi.lo = vload4(0, src_data + in_base + 8); \
    } \
    if (in_h_start + 1 >= 0 && in_h_start + 1 < src_h) { \
        row1.lo = vload8(0, src_data + in_base + src_w); \
        row1.hi.lo = vload4(0, src_data + in_base + src_w + 8); \
    } \
    short16 inputMask = (short16)(0xffff); \
    if (in_w_start < 0 || in_w_start >= src_w) { \
        inputMask.s0 = 0x0000; \
    } \
    if (in_w_start + 1 < 0 || in_w_start + 1 >= src_w) { \
        inputMask.s1 = 0x0000; \
    } \
    if (in_w_start + 2 < 0 || in_w_start + 2 >= src_w) { \
        inputMask.s2 = 0x0000; \
    } \
    if (in_w_start + 3 < 0 || in_w_start + 3 >= src_w) { \
        inputMask.s3 = 0x0000; \
    } \
    if (in_w_start + 4 < 0 || in_w_start + 4 >= src_w) { \
        inputMask.s4 = 0x0000; \
    } \
    if (in_w_start + 5 < 0 || in_w_start + 5 >= src_w) { \
        inputMask.s5 = 0x0000; \
    } \
    if (in_w_start + 6 < 0 || in_w_start + 6 >= src_w) { \
        inputMask.s6 = 0x0000; \
    } \
    if (in_w_start + 7 < 0 || in_w_start + 7 >= src_w) { \
        inputMask.s7 = 0x0000; \
    } \
    if (in_w_start + 8 < 0 || in_w_start + 8 >= src_w) { \
        inputMask.s8 = 0x0000; \
    } \
    if (in_w_start + 9 < 0 || in_w_start + 9 >= src_w) { \
        inputMask.s9 = 0x0000; \
    } \
    if (in_w_start + 10 < 0 || in_w_start + 10 >= src_w) { \
        inputMask.sA = 0x0000; \
    } \
    if (in_w_start + 11 < 0 || in_w_start + 11 >= src_w) { \
        inputMask.sB = 0x0000; \
    } \
    inputMask.sC = 0x0000; \
    inputMask.sD = 0x0000; \
    inputMask.sE = 0x0000; \
    inputMask.sF = 0x0000; \
    row0 = AS_DATA_T16(as_short16(row0) & inputMask); \
    row1 = AS_DATA_T16(as_short16(row1) & inputMask); \
    DATA_T bt = bt_ar[0]; \
    DATA_T2 src0 = (DATA_T2)(row0.s0, row0.s4); \
    I0 = bt * src0; \
    DATA_T2 src1 = (DATA_T2)(row0.s1, row0.s5); \
    I1 = bt * src1; \
    DATA_T2 src2 = (DATA_T2)(row0.s2, row0.s6); \
    I2 = bt * src2; \
    DATA_T2 src3 = (DATA_T2)(row0.s3, row0.s7); \
    I3 = bt * src3; \
    DATA_T2 src4 = (DATA_T2)(row0.s4, row0.s8); \
    I4 = bt * src4; \
    DATA_T2 src5 = (DATA_T2)(row0.s5, row0.s9); \
    I5 = bt * src5; \
    DATA_T2 src6 = (DATA_T2)(row0.s6, row0.sA); \
    I6 = bt * src6; \
    DATA_T2 src7 = (DATA_T2)(row0.s7, row0.sB); \
    I7 = bt * src7; \
    bt = bt_ar[1]; \
    src0 = (DATA_T2)(row1.s0, row1.s4); \
    I0 += bt * src0; \
    src1 = (DATA_T2)(row1.s1, row1.s5); \
    I1 += bt * src1; \
    src2 = (DATA_T2)(row1.s2, row1.s6); \
    I2 += bt * src2; \
    src3 = (DATA_T2)(row1.s3, row1.s7); \
    I3 += bt * src3; \
    src4 = (DATA_T2)(row1.s4, row1.s8); \
    I4 += bt * src4; \
    src5 = (DATA_T2)(row1.s5, row1.s9); \
    I5 += bt * src5; \
    src6 = (DATA_T2)(row1.s6, row1.sA); \
    I6 += bt * src6; \
    src7 = (DATA_T2)(row1.s7, row1.sB); \
    I7 += bt * src7; \
    bt = bt_br[0]; \
    src0 = (DATA_T2)(row0.s0, row0.s4); \
    J0 = bt * src0; \
    src1 = (DATA_T2)(row0.s1, row0.s5); \
    J1 = bt * src1; \
    src2 = (DATA_T2)(row0.s2, row0.s6); \
    J2 = bt * src2; \
    src3 = (DATA_T2)(row0.s3, row0.s7); \
    J3 = bt * src3; \
    src4 = (DATA_T2)(row0.s4, row0.s8); \
    J4 = bt * src4; \
    src5 = (DATA_T2)(row0.s5, row0.s9); \
    J5 = bt * src5; \
    src6 = (DATA_T2)(row0.s6, row0.sA); \
    J6 = bt * src6; \
    src7 = (DATA_T2)(row0.s7, row0.sB); \
    J7 = bt * src7; \
    bt = bt_br[1]; \
    src0 = (DATA_T2)(row1.s0, row1.s4); \
    J0 += bt * src0; \
    src1 = (DATA_T2)(row1.s1, row1.s5); \
    J1 += bt * src1; \
    src2 = (DATA_T2)(row1.s2, row1.s6); \
    J2 += bt * src2; \
    src3 = (DATA_T2)(row1.s3, row1.s7); \
    J3 += bt * src3; \
    src4 = (DATA_T2)(row1.s4, row1.s8); \
    J4 += bt * src4; \
    src5 = (DATA_T2)(row1.s5, row1.s9); \
    J5 += bt * src5; \
    src6 = (DATA_T2)(row1.s6, row1.sA); \
    J6 += bt * src6; \
    src7 = (DATA_T2)(row1.s7, row1.sB); \
    J7 += bt * src7; \
    row0 = (DATA_T16)(0.0f); \
    row1 = (DATA_T16)(0.0f); \
    if (in_h_start + 2 >= 0 && in_h_start + 2 < src_h) { \
        row0.lo = vload8(0, src_data + in_base + src_w * 2); \
        row0.hi.lo = vload4(0, src_data + in_base + src_w * 2 + 8); \
    } \
    if (in_h_start + 3 >= 0 && in_h_start + 3  < src_h) { \
        row1.lo = vload8(0, src_data + in_base + src_w * 3); \
        row1.hi.lo = vload4(0, src_data + in_base + src_w * 3 + 8); \
    } \
    row0 = AS_DATA_T16(as_short16(row0) & inputMask); \
    row1 = AS_DATA_T16(as_short16(row1) & inputMask); \
    bt = bt_ar[2]; \
    src0 = (DATA_T2)(row0.s0, row0.s4); \
    I0 += bt * src0; \
    src1 = (DATA_T2)(row0.s1, row0.s5); \
    I1 += bt * src1; \
    src2 = (DATA_T2)(row0.s2, row0.s6); \
    I2 += bt * src2; \
    src3 = (DATA_T2)(row0.s3, row0.s7); \
    I3 += bt * src3; \
    src4 = (DATA_T2)(row0.s4, row0.s8); \
    I4 += bt * src4; \
    src5 = (DATA_T2)(row0.s5, row0.s9); \
    I5 += bt * src5; \
    src6 = (DATA_T2)(row0.s6, row0.sA); \
    I6 += bt * src6; \
    src7 = (DATA_T2)(row0.s7, row0.sB); \
    I7 += bt * src7; \
    bt = bt_ar[3]; \
    src0 = (DATA_T2)(row1.s0, row1.s4); \
    I0 += bt * src0; \
    src1 = (DATA_T2)(row1.s1, row1.s5); \
    I1 += bt * src1; \
    src2 = (DATA_T2)(row1.s2, row1.s6); \
    I2 += bt * src2; \
    src3 = (DATA_T2)(row1.s3, row1.s7); \
    I3 += bt * src3; \
    src4 = (DATA_T2)(row1.s4, row1.s8); \
    I4 += bt * src4; \
    src5 = (DATA_T2)(row1.s5, row1.s9); \
    I5 += bt * src5; \
    src6 = (DATA_T2)(row1.s6, row1.sA); \
    I6 += bt * src6; \
    src7 = (DATA_T2)(row1.s7, row1.sB); \
    I7 += bt * src7; \
    bt = bt_br[2]; \
    src0 = (DATA_T2)(row0.s0, row0.s4); \
    J0 += bt * src0; \
    src1 = (DATA_T2)(row0.s1, row0.s5); \
    J1 += bt * src1; \
    src2 = (DATA_T2)(row0.s2, row0.s6); \
    J2 += bt * src2; \
    src3 = (DATA_T2)(row0.s3, row0.s7); \
    J3 += bt * src3; \
    src4 = (DATA_T2)(row0.s4, row0.s8); \
    J4 += bt * src4; \
    src5 = (DATA_T2)(row0.s5, row0.s9); \
    J5 += bt * src5; \
    src6 = (DATA_T2)(row0.s6, row0.sA); \
    J6 += bt * src6; \
    src7 = (DATA_T2)(row0.s7, row0.sB); \
    J7 += bt * src7; \
    bt = bt_br[3]; \
    src0 = (DATA_T2)(row1.s0, row1.s4); \
    J0 += bt * src0; \
    src1 = (DATA_T2)(row1.s1, row1.s5); \
    J1 += bt * src1; \
    src2 = (DATA_T2)(row1.s2, row1.s6); \
    J2 += bt * src2; \
    src3 = (DATA_T2)(row1.s3, row1.s7); \
    J3 += bt * src3; \
    src4 = (DATA_T2)(row1.s4, row1.s8); \
    J4 += bt * src4; \
    src5 = (DATA_T2)(row1.s5, row1.s9); \
    J5 += bt * src5; \
    src6 = (DATA_T2)(row1.s6, row1.sA); \
    J6 += bt * src6; \
    src7 = (DATA_T2)(row1.s7, row1.sB); \
    J7 += bt * src7; \
    row0 = (DATA_T16)(0.0f); \
    row1 = (DATA_T16)(0.0f); \
    if (in_h_start + 4 >= 0 && in_h_start + 4 < src_h) { \
        row0.lo = vload8(0, src_data + in_base + src_w * 4); \
        row0.hi.lo = vload4(0, src_data + in_base + src_w * 4 + 8); \
    } \
    if (in_h_start + 5 >= 0 && in_h_start + 5 < src_h) { \
        row1.lo = vload8(0, src_data + in_base + src_w * 5); \
        row1.hi.lo = vload4(0, src_data + in_base + src_w * 5 + 8); \
    } \
    row0 = AS_DATA_T16(as_short16(row0) & inputMask); \
    row1 = AS_DATA_T16(as_short16(row1) & inputMask); \
    bt = bt_ar[4]; \
    src0 = (DATA_T2)(row0.s0, row0.s4); \
    I0 += bt * src0; \
    src1 = (DATA_T2)(row0.s1, row0.s5); \
    I1 += bt * src1; \
    src2 = (DATA_T2)(row0.s2, row0.s6); \
    I2 += bt * src2; \
    src3 = (DATA_T2)(row0.s3, row0.s7); \
    I3 += bt * src3; \
    src4 = (DATA_T2)(row0.s4, row0.s8); \
    I4 += bt * src4; \
    src5 = (DATA_T2)(row0.s5, row0.s9); \
    I5 += bt * src5; \
    src6 = (DATA_T2)(row0.s6, row0.sA); \
    I6 += bt * src6; \
    src7 = (DATA_T2)(row0.s7, row0.sB); \
    I7 += bt * src7; \
    bt = bt_ar[5]; \
    src0 = (DATA_T2)(row1.s0, row1.s4); \
    I0 += bt * src0; \
    src1 = (DATA_T2)(row1.s1, row1.s5); \
    I1 += bt * src1; \
    src2 = (DATA_T2)(row1.s2, row1.s6); \
    I2 += bt * src2; \
    src3 = (DATA_T2)(row1.s3, row1.s7); \
    I3 += bt * src3; \
    src4 = (DATA_T2)(row1.s4, row1.s8); \
    I4 += bt * src4; \
    src5 = (DATA_T2)(row1.s5, row1.s9); \
    I5 += bt * src5; \
    src6 = (DATA_T2)(row1.s6, row1.sA); \
    I6 += bt * src6; \
    src7 = (DATA_T2)(row1.s7, row1.sB); \
    I7 += bt * src7; \
    bt = bt_br[4]; \
    src0 = (DATA_T2)(row0.s0, row0.s4); \
    J0 += bt * src0; \
    src1 = (DATA_T2)(row0.s1, row0.s5); \
    J1 += bt * src1; \
    src2 = (DATA_T2)(row0.s2, row0.s6); \
    J2 += bt * src2; \
    src3 = (DATA_T2)(row0.s3, row0.s7); \
    J3 += bt * src3; \
    src4 = (DATA_T2)(row0.s4, row0.s8); \
    J4 += bt * src4; \
    src5 = (DATA_T2)(row0.s5, row0.s9); \
    J5 += bt * src5; \
    src6 = (DATA_T2)(row0.s6, row0.sA); \
    J6 += bt * src6; \
    src7 = (DATA_T2)(row0.s7, row0.sB); \
    J7 += bt * src7; \
    bt = bt_br[5]; \
    src0 = (DATA_T2)(row1.s0, row1.s4); \
    J0 += bt * src0; \
    src1 = (DATA_T2)(row1.s1, row1.s5); \
    J1 += bt * src1; \
    src2 = (DATA_T2)(row1.s2, row1.s6); \
    J2 += bt * src2; \
    src3 = (DATA_T2)(row1.s3, row1.s7); \
    J3 += bt * src3; \
    src4 = (DATA_T2)(row1.s4, row1.s8); \
    J4 += bt * src4; \
    src5 = (DATA_T2)(row1.s5, row1.s9); \
    J5 += bt * src5; \
    src6 = (DATA_T2)(row1.s6, row1.sA); \
    J6 += bt * src6; \
    src7 = (DATA_T2)(row1.s7, row1.sB); \
    J7 += bt * src7; \
    row0 = (DATA_T16)(0.0f); \
    row1 = (DATA_T16)(0.0f); \
    if (in_h_start + 6 >= 0 && in_h_start + 6 < src_h) { \
        row0.lo = vload8(0, src_data + in_base + src_w * 6); \
        row0.hi.lo = vload4(0, src_data + in_base + src_w * 6 + 8); \
    } \
    if (in_h_start + 7 >= 0 && in_h_start + 7 < src_h) { \
        row1.lo = vload8(0, src_data + in_base + src_w * 7); \
        row1.hi.lo = vload4(0, src_data + in_base + src_w * 7 + 8); \
    } \
    row0 = AS_DATA_T16(as_short16(row0) & inputMask); \
    row1 = AS_DATA_T16(as_short16(row1) & inputMask); \
    bt = bt_ar[6]; \
    src0 = (DATA_T2)(row0.s0, row0.s4); \
    I0 += bt * src0; \
    src1 = (DATA_T2)(row0.s1, row0.s5); \
    I1 += bt * src1; \
    src2 = (DATA_T2)(row0.s2, row0.s6); \
    I2 += bt * src2; \
    src3 = (DATA_T2)(row0.s3, row0.s7); \
    I3 += bt * src3; \
    src4 = (DATA_T2)(row0.s4, row0.s8); \
    I4 += bt * src4; \
    src5 = (DATA_T2)(row0.s5, row0.s9); \
    I5 += bt * src5; \
    src6 = (DATA_T2)(row0.s6, row0.sA); \
    I6 += bt * src6; \
    src7 = (DATA_T2)(row0.s7, row0.sB); \
    I7 += bt * src7; \
    bt = bt_ar[7]; \
    src0 = (DATA_T2)(row1.s0, row1.s4); \
    I0 += bt * src0; \
    src1 = (DATA_T2)(row1.s1, row1.s5); \
    I1 += bt * src1; \
    src2 = (DATA_T2)(row1.s2, row1.s6); \
    I2 += bt * src2; \
    src3 = (DATA_T2)(row1.s3, row1.s7); \
    I3 += bt * src3; \
    src4 = (DATA_T2)(row1.s4, row1.s8); \
    I4 += bt * src4; \
    src5 = (DATA_T2)(row1.s5, row1.s9); \
    I5 += bt * src5; \
    src6 = (DATA_T2)(row1.s6, row1.sA); \
    I6 += bt * src6; \
    src7 = (DATA_T2)(row1.s7, row1.sB); \
    I7 += bt * src7; \
    bt = bt_br[6]; \
    src0 = (DATA_T2)(row0.s0, row0.s4); \
    J0 += bt * src0; \
    src1 = (DATA_T2)(row0.s1, row0.s5); \
    J1 += bt * src1; \
    src2 = (DATA_T2)(row0.s2, row0.s6); \
    J2 += bt * src2; \
    src3 = (DATA_T2)(row0.s3, row0.s7); \
    J3 += bt * src3; \
    src4 = (DATA_T2)(row0.s4, row0.s8); \
    J4 += bt * src4; \
    src5 = (DATA_T2)(row0.s5, row0.s9); \
    J5 += bt * src5; \
    src6 = (DATA_T2)(row0.s6, row0.sA); \
    J6 += bt * src6; \
    src7 = (DATA_T2)(row0.s7, row0.sB); \
    J7 += bt * src7; \
    bt = bt_br[7]; \
    src0 = (DATA_T2)(row1.s0, row1.s4); \
    J0 += bt * src0; \
    src1 = (DATA_T2)(row1.s1, row1.s5); \
    J1 += bt * src1; \
    src2 = (DATA_T2)(row1.s2, row1.s6); \
    J2 += bt * src2; \
    src3 = (DATA_T2)(row1.s3, row1.s7); \
    J3 += bt * src3; \
    src4 = (DATA_T2)(row1.s4, row1.s8); \
    J4 += bt * src4; \
    src5 = (DATA_T2)(row1.s5, row1.s9); \
    J5 += bt * src5; \
    src6 = (DATA_T2)(row1.s6, row1.sA); \
    J6 += bt * src6; \
    src7 = (DATA_T2)(row1.s7, row1.sB); \
    J7 += bt * src7; \
    float2 I0_f = convert_float2(I0); \
    float2 I1_f = convert_float2(I1); \
    float2 I2_f = convert_float2(I2); \
    float2 I3_f = convert_float2(I3); \
    float2 I4_f = convert_float2(I4); \
    float2 I5_f = convert_float2(I5); \
    float2 I6_f = convert_float2(I6); \
    float2 I7_f = convert_float2(I7); \
    { \
      DATA_T2 r0 = CONVERT_TO_DATA_T2(I0_f + -2.722228f * I2_f + 1.555556f * I4_f + \
                                -0.222222f * I6_f); \
      DATA_T2 r1 = CONVERT_TO_DATA_T2(1.060660f * I1_f + 1.5f * I2_f + \
                                -0.766032f * I3_f + -1.083333f * I4_f + \
                                0.117851f * I5_f + 0.1666667f * I6_f); \
      vstore2((DATA_T2)(r0), 0, dst_data + dst_tile); \
      dst_tile += aligned_inch * coalescing_feature_height_; \
      vstore2((DATA_T2)(r1), 0, dst_data + dst_tile); \
      dst_tile += aligned_inch * coalescing_feature_height_; \
    } \
    { \
      DATA_T2 r0 = CONVERT_TO_DATA_T2(-1.060660f * I1_f + 1.5f * I2_f + \
                                0.766032f * I3_f + -1.083333f * I4_f + \
                                -0.117851f * I5_f + 0.1666667f * I6_f); \
      DATA_T2 r1 = CONVERT_TO_DATA_T2(-0.212132f * I1_f + -0.15f * I2_f + \
                                0.471404f * I3_f + 0.333333f * I4_f + \
                                -0.094280f * I5_f + -0.066666f * I6_f); \
      vstore2((DATA_T2)(r0), 0, dst_data + dst_tile); \
      dst_tile += aligned_inch * coalescing_feature_height_; \
      vstore2((DATA_T2)(r1), 0, dst_data + dst_tile); \
      dst_tile += aligned_inch * coalescing_feature_height_; \
    } \
    dst_tile = p_dst_tile + 1 * aligned_pixel * inch_aligned * aligned_tile; \
    { \
      DATA_T2 r0 = CONVERT_TO_DATA_T2(0.212132f * I1_f + -0.15f * I2_f + \
                                -0.471404f * I3_f + 0.333333f * I4_f + \
                                0.094280f * I5_f + -0.066666f * I6_f); \
      DATA_T2 r1 = CONVERT_TO_DATA_T2(0.023570f * I1_f + 0.011111f * I2_f + \
                                -0.058925f * I3_f + -0.027777f * I4_f + \
                                0.023570f * I5_f + 0.011111f * I6_f); \
      vstore2((DATA_T2)(r0), 0, dst_data + dst_tile); \
      dst_tile += aligned_inch * coalescing_feature_height_; \
      vstore2((DATA_T2)(r1), 0, dst_data + dst_tile); \
      dst_tile += aligned_inch * coalescing_feature_height_; \
    } \
    { \
      DATA_T2 r0 = CONVERT_TO_DATA_T2(-0.023570f * I1_f + 0.011111f * I2_f + \
                                0.058925f * I3_f + -0.027777f * I4_f + \
                                -0.023570f * I5_f + 0.011111f * I6_f); \
      DATA_T2 r1 = CONVERT_TO_DATA_T2(-4.5f * I1_f + 12.25f * I3_f + -7.0f * I5_f + \
                                1.0f * I7_f); \
      vstore2((DATA_T2)(r0), 0, dst_data + dst_tile); \
      dst_tile += aligned_inch * coalescing_feature_height_; \
      vstore2((DATA_T2)(r1), 0, dst_data + dst_tile); \
      dst_tile += aligned_inch * coalescing_feature_height_; \
    } \
      I0_f = convert_float2(J0); \
      I1_f = convert_float2(J1); \
      I2_f = convert_float2(J2); \
      I3_f = convert_float2(J3); \
      I4_f = convert_float2(J4); \
      I5_f = convert_float2(J5); \
      I6_f = convert_float2(J6); \
      I7_f = convert_float2(J7); \
      dst_tile = p_dst_tile + 2 * aligned_pixel * inch_aligned * aligned_tile; \
      { \
        DATA_T2 r0 = CONVERT_TO_DATA_T2(I0_f + -2.722228f * I2_f + 1.555556f * I4_f + \
                                -0.222222f * I6_f); \
        DATA_T2 r1 = CONVERT_TO_DATA_T2(1.060660f * I1_f + 1.5f * I2_f + \
                                -0.766032f * I3_f + -1.083333f * I4_f + \
                                0.117851f * I5_f + 0.1666667f * I6_f); \
        vstore2((DATA_T2)(r0), 0, dst_data + dst_tile); \
        dst_tile += aligned_inch * coalescing_feature_height_; \
        vstore2((DATA_T2)(r1), 0, dst_data + dst_tile); \
        dst_tile += aligned_inch * coalescing_feature_height_; \
      } \
      { \
        DATA_T2 r0 = CONVERT_TO_DATA_T2(-1.060660f * I1_f + 1.5f * I2_f + \
                                0.766032f * I3_f + -1.083333f * I4_f + \
                                -0.117851f * I5_f + 0.1666667f * I6_f); \
        DATA_T2 r1 = CONVERT_TO_DATA_T2(-0.212132f * I1_f + -0.15f * I2_f + \
                                0.471404f * I3_f + 0.333333f * I4_f + \
                                -0.094280f * I5_f + -0.066666f * I6_f); \
        vstore2((DATA_T2)(r0), 0, dst_data + dst_tile); \
        dst_tile += aligned_inch * coalescing_feature_height_; \
        vstore2((DATA_T2)(r1), 0, dst_data + dst_tile); \
        dst_tile += aligned_inch * coalescing_feature_height_; \
      } \
      dst_tile = p_dst_tile + 3 * aligned_pixel * inch_aligned * aligned_tile; \
      { \
        DATA_T2 r0 = CONVERT_TO_DATA_T2(0.212132f * I1_f + -0.15f * I2_f + \
                                -0.471404f * I3_f + 0.333333f * I4_f + \
                                0.094280f * I5_f + -0.066666f * I6_f); \
        DATA_T2 r1 = CONVERT_TO_DATA_T2(0.023570f * I1_f + 0.011111f * I2_f + \
                                -0.058925f * I3_f + -0.027777f * I4_f + \
                                0.023570f * I5_f + 0.011111f * I6_f); \
        vstore2((DATA_T2)(r0), 0, dst_data + dst_tile); \
        dst_tile += aligned_inch * coalescing_feature_height_; \
        vstore2((DATA_T2)(r1), 0, dst_data + dst_tile); \
        dst_tile += aligned_inch * coalescing_feature_height_; \
      } \
      { \
        DATA_T2 r0 = CONVERT_TO_DATA_T2(-0.023570f * I1_f + 0.011111f * I2_f + \
                                0.058925f * I3_f + -0.027777f * I4_f + \
                                -0.023570f * I5_f + 0.011111f * I6_f); \
        DATA_T2 r1 = CONVERT_TO_DATA_T2(-4.5f * I1_f + 12.25f * I3_f + -7.0f * I5_f + \
                                1.0f * I7_f); \
        vstore2((DATA_T2)(r0), 0, dst_data + dst_tile); \
        dst_tile += aligned_inch * coalescing_feature_height_; \
        vstore2((DATA_T2)(r1), 0, dst_data + dst_tile); \
      }

#define WINO5X5_DOT_MULTIPLY_NCHW_OPT_4X16(input_tm, weight_tm, dot_result, dst_n, dst_c, dst_h, dst_w, input_c, \
                                           output_c, coalescing_feature_height_, aligned_outch, aligned_inch, \
                                           aligned_pixel) \
    int tile_index = get_global_id(0) * 4; \
    int p_index = get_global_id(1); \
    int c_index = get_global_id(2) % dst_c; \
    int n_index = get_global_id(2) / dst_c; \
    if (tile_index >= dst_w || c_index >= dst_c || n_index >= dst_n) { \
      return; \
    } \
  DATA_T16 r0 = 0.0f; \
  DATA_T16 r1 = 0.0f; \
  DATA_T16 r2 = 0.0f; \
  DATA_T16 r3 = 0.0f; \
  int align_tile = 2; \
  int align_pixel = aligned_pixel; \
  int input_c_aligned = (input_c + aligned_inch - 1) / aligned_inch * aligned_inch; \
  int output_c_aligned = (output_c + aligned_outch - 1) / aligned_outch * aligned_outch; \
  int input_base = n_index * input_c_aligned * dst_h * dst_w; \
  input_base += tile_index / coalescing_feature_height_ * input_c_aligned * \
                coalescing_feature_height_ * align_pixel; \
  input_base += \
      tile_index % coalescing_feature_height_ / 2 * 2 * aligned_inch; \
  input_base += tile_index % coalescing_feature_height_ % 2; \
  input_base += p_index / align_pixel * align_pixel * dst_w * input_c_aligned; \
  input_base += p_index % align_pixel * aligned_inch * coalescing_feature_height_; \
  int weight_base = c_index * aligned_outch * aligned_inch * align_pixel; \
  weight_base += p_index / align_pixel * align_pixel * output_c_aligned * input_c_aligned; \
  weight_base += p_index % align_pixel * aligned_outch * aligned_inch; \
  int dot_base = n_index * dst_c * aligned_outch * dst_h * dst_w; \
  dot_base += c_index * aligned_outch * 64 * dst_w; \
  dot_base += tile_index % coalescing_feature_height_ / 2 * 2 * align_pixel; \
  dot_base += tile_index % coalescing_feature_height_ % 2; \
  dot_base += tile_index / coalescing_feature_height_ * coalescing_feature_height_ * 64; \
  dot_base += p_index / align_pixel * align_pixel * coalescing_feature_height_; \
  dot_base += p_index % align_pixel * 2; \
  int2 stride = \
      (int2)(coalescing_feature_height_ * aligned_inch * align_pixel, \
              output_c_aligned * aligned_inch * align_pixel); \
  for (int i = 0; i < input_c_aligned; i += aligned_inch) { \
    DATA_T8 vin = vload8(0, input_tm + input_base); \
    DATA_T16 vwl = vload16(0, weight_tm + weight_base); \
    DATA_T16 vwh = vload16(0, weight_tm + weight_base + 16); \
    r0 = mad(vin.s0, vwl, r0); \
    r1 = mad(vin.s1, vwl, r1); \
    r2 = mad(vin.s4, vwl, r2); \
    r3 = mad(vin.s5, vwl, r3); \
    r0 = mad(vin.s2, vwh, r0); \
    r1 = mad(vin.s3, vwh, r1); \
    r2 = mad(vin.s6, vwh, r2); \
    r3 = mad(vin.s7, vwh, r3); \
    input_base += stride.s0; \
    weight_base += stride.s1; \
  } \
  vstore2((DATA_T2)(r0.s0, r1.s0), 0, dot_result + dot_base); \
  vstore2((DATA_T2)(r2.s0, r3.s0), 0, dot_result + dot_base + align_tile * align_pixel); \
  dot_base += 64 * dst_w; \
  vstore2((DATA_T2)(r0.s1, r1.s1), 0, dot_result + dot_base); \
  vstore2((DATA_T2)(r2.s1, r3.s1), 0, dot_result + dot_base + align_tile * align_pixel); \
  dot_base += 64 * dst_w; \
  vstore2((DATA_T2)(r0.s2, r1.s2), 0, dot_result + dot_base); \
  vstore2((DATA_T2)(r2.s2, r3.s2), 0, dot_result + dot_base + align_tile * align_pixel); \
  dot_base += 64 * dst_w; \
  vstore2((DATA_T2)(r0.s3, r1.s3), 0, dot_result + dot_base); \
  vstore2((DATA_T2)(r2.s3, r3.s3), 0, dot_result + dot_base + align_tile * align_pixel); \
  dot_base += 64 * dst_w; \
  vstore2((DATA_T2)(r0.s4, r1.s4), 0, dot_result + dot_base); \
  vstore2((DATA_T2)(r2.s4, r3.s4), 0, dot_result + dot_base + align_tile * align_pixel); \
  dot_base += 64 * dst_w; \
  vstore2((DATA_T2)(r0.s5, r1.s5), 0, dot_result + dot_base); \
  vstore2((DATA_T2)(r2.s5, r3.s5), 0, dot_result + dot_base + align_tile * align_pixel); \
  dot_base += 64 * dst_w; \
  vstore2((DATA_T2)(r0.s6, r1.s6), 0, dot_result + dot_base); \
  vstore2((DATA_T2)(r2.s6, r3.s6), 0, dot_result + dot_base + align_tile * align_pixel); \
  dot_base += 64 * dst_w; \
  vstore2((DATA_T2)(r0.s7, r1.s7), 0, dot_result + dot_base); \
  vstore2((DATA_T2)(r2.s7, r3.s7), 0, dot_result + dot_base + align_tile * align_pixel); \
  dot_base += 64 * dst_w; \
  vstore2((DATA_T2)(r0.s8, r1.s8), 0, dot_result + dot_base); \
  vstore2((DATA_T2)(r2.s8, r3.s8), 0, dot_result + dot_base + align_tile * align_pixel); \
  dot_base += 64 * dst_w; \
  vstore2((DATA_T2)(r0.s9, r1.s9), 0, dot_result + dot_base); \
  vstore2((DATA_T2)(r2.s9, r3.s9), 0, dot_result + dot_base + align_tile * align_pixel); \
  dot_base += 64 * dst_w; \
  vstore2((DATA_T2)(r0.sA, r1.sA), 0, dot_result + dot_base); \
  vstore2((DATA_T2)(r2.sA, r3.sA), 0, dot_result + dot_base + align_tile * align_pixel); \
  dot_base += 64 * dst_w; \
  vstore2((DATA_T2)(r0.sB, r1.sB), 0, dot_result + dot_base); \
  vstore2((DATA_T2)(r2.sB, r3.sB), 0, dot_result + dot_base + align_tile * align_pixel); \
  dot_base += 64 * dst_w; \
  vstore2((DATA_T2)(r0.sC, r1.sC), 0, dot_result + dot_base); \
  vstore2((DATA_T2)(r2.sC, r3.sC), 0, dot_result + dot_base + align_tile * align_pixel); \
  dot_base += 64 * dst_w; \
  vstore2((DATA_T2)(r0.sD, r1.sD), 0, dot_result + dot_base); \
  vstore2((DATA_T2)(r2.sD, r3.sD), 0, dot_result + dot_base + align_tile * align_pixel); \
  dot_base += 64 * dst_w; \
  vstore2((DATA_T2)(r0.sE, r1.sE), 0, dot_result + dot_base); \
  vstore2((DATA_T2)(r2.sE, r3.sE), 0, dot_result + dot_base + align_tile * align_pixel); \
  dot_base += 64 * dst_w; \
  vstore2((DATA_T2)(r0.sF, r1.sF), 0, dot_result + dot_base); \
  vstore2((DATA_T2)(r2.sF, r3.sF), 0, dot_result + dot_base + align_tile * align_pixel);

#define WINO5X5_OUTPUT_TM_NCHW_OPT_4X16(dot_result, bias, output, src_n, src_c, src_h, src_w, output_n, output_c, \
                                        output_h, output_w, coalescing_feature_height_, tile_x, aligned_outch, \
                                        aligned_inch, aligned_pixel) \
    int n_index = get_global_id(2) / src_c; \
    int c_index = get_global_id(2) % src_c; \
    int p_index = get_global_id(1); \
    int h_index = get_global_id(0) * 2 / tile_x; \
    int w_index = get_global_id(0) * 2 % tile_x; \
    int t_index = h_index * tile_x + w_index; \
    int tile_totals = ((output_h + 3) / 4) * tile_x; \
    if (c_index >= output_c || t_index >= tile_totals || \
        h_index * 4 + p_index >= output_h || w_index * 4 >= output_w) { \
      return; \
    } \
    DATA_T at_ar[8]; \
    DATA_T8 t00 = vload8(0, At5x5 + 8 * p_index); \
    at_ar[0] = t00.s0; \
    at_ar[1] = t00.s1; \
    at_ar[2] = t00.s2; \
    at_ar[3] = t00.s3; \
    at_ar[4] = t00.s4; \
    at_ar[5] = t00.s5; \
    at_ar[6] = t00.s6; \
    at_ar[7] = t00.s7; \
    int pixel_align = aligned_pixel; \
    int src_base = n_index * src_c * src_h * src_w; \
    src_base += c_index * src_h * src_w; \
    src_base += t_index % coalescing_feature_height_ / 2 * 2 * pixel_align; \
    src_base += t_index % coalescing_feature_height_ % 2; \
    src_base += t_index / coalescing_feature_height_ * (coalescing_feature_height_ * 64); \
    int dst_base = n_index * output_c * output_h * output_w; \
    dst_base += c_index * output_h * output_w; \
    dst_base += (h_index * 4 + p_index) * output_w + w_index * 4; \
    DATA_T16 ping; \
    ping.lo = vload8(0, dot_result + src_base); \
    src_base += pixel_align * coalescing_feature_height_; \
    ping.hi = vload8(0, dot_result + src_base); \
    src_base += pixel_align * coalescing_feature_height_; \
    DATA_T2 tmp0 = at_ar[0] * ping.s01; \
    DATA_T2 tmp1 = at_ar[0] * ping.s23; \
    DATA_T2 tmp2 = at_ar[0] * ping.s45; \
    DATA_T2 tmp3 = at_ar[0] * ping.s67; \
    DATA_T2 tmp4 = at_ar[0] * ping.s89; \
    DATA_T2 tmp5 = at_ar[0] * ping.sAB; \
    DATA_T2 tmp6 = at_ar[0] * ping.sCD; \
    DATA_T2 tmp7 = at_ar[0] * ping.sEF; \
    ping.lo = vload8(0, dot_result + src_base); \
    src_base += pixel_align * coalescing_feature_height_; \
    ping.hi = vload8(0, dot_result + src_base); \
    src_base += pixel_align * coalescing_feature_height_; \
    tmp0 += at_ar[1] * ping.s01; \
    tmp1 += at_ar[1] * ping.s23; \
    tmp2 += at_ar[1] * ping.s45; \
    tmp3 += at_ar[1] * ping.s67; \
    tmp4 += at_ar[1] * ping.s89; \
    tmp5 += at_ar[1] * ping.sAB; \
    tmp6 += at_ar[1] * ping.sCD; \
    tmp7 += at_ar[1] * ping.sEF; \
    ping.lo = vload8(0, dot_result + src_base); \
    src_base += pixel_align * coalescing_feature_height_; \
    ping.hi = vload8(0, dot_result + src_base); \
    src_base += pixel_align * coalescing_feature_height_; \
    tmp0 += at_ar[2] * ping.s01; \
    tmp1 += at_ar[2] * ping.s23; \
    tmp2 += at_ar[2] * ping.s45; \
    tmp3 += at_ar[2] * ping.s67; \
    tmp4 += at_ar[2] * ping.s89; \
    tmp5 += at_ar[2] * ping.sAB; \
    tmp6 += at_ar[2] * ping.sCD; \
    tmp7 += at_ar[2] * ping.sEF; \
    ping.lo = vload8(0, dot_result + src_base); \
    src_base += pixel_align * coalescing_feature_height_; \
    ping.hi = vload8(0, dot_result + src_base); \
    src_base += pixel_align * coalescing_feature_height_; \
    tmp0 += at_ar[3] * ping.s01; \
    tmp1 += at_ar[3] * ping.s23; \
    tmp2 += at_ar[3] * ping.s45; \
    tmp3 += at_ar[3] * ping.s67; \
    tmp4 += at_ar[3] * ping.s89; \
    tmp5 += at_ar[3] * ping.sAB; \
    tmp6 += at_ar[3] * ping.sCD; \
    tmp7 += at_ar[3] * ping.sEF; \
    ping.lo = vload8(0, dot_result + src_base); \
    src_base += pixel_align * coalescing_feature_height_; \
    ping.hi = vload8(0, dot_result + src_base); \
    src_base += pixel_align * coalescing_feature_height_; \
    tmp0 += at_ar[4] * ping.s01; \
    tmp1 += at_ar[4] * ping.s23; \
    tmp2 += at_ar[4] * ping.s45; \
    tmp3 += at_ar[4] * ping.s67; \
    tmp4 += at_ar[4] * ping.s89; \
    tmp5 += at_ar[4] * ping.sAB; \
    tmp6 += at_ar[4] * ping.sCD; \
    tmp7 += at_ar[4] * ping.sEF; \
    ping.lo = vload8(0, dot_result + src_base); \
    src_base += pixel_align * coalescing_feature_height_; \
    ping.hi = vload8(0, dot_result + src_base); \
    src_base += pixel_align * coalescing_feature_height_; \
    tmp0 += at_ar[5] * ping.s01; \
    tmp1 += at_ar[5] * ping.s23; \
    tmp2 += at_ar[5] * ping.s45; \
    tmp3 += at_ar[5] * ping.s67; \
    tmp4 += at_ar[5] * ping.s89; \
    tmp5 += at_ar[5] * ping.sAB; \
    tmp6 += at_ar[5] * ping.sCD; \
    tmp7 += at_ar[5] * ping.sEF; \
    ping.lo = vload8(0, dot_result + src_base); \
    src_base += pixel_align * coalescing_feature_height_; \
    ping.hi = vload8(0, dot_result + src_base); \
    src_base += pixel_align * coalescing_feature_height_; \
    tmp0 += at_ar[6] * ping.s01; \
    tmp1 += at_ar[6] * ping.s23; \
    tmp2 += at_ar[6] * ping.s45; \
    tmp3 += at_ar[6] * ping.s67; \
    tmp4 += at_ar[6] * ping.s89; \
    tmp5 += at_ar[6] * ping.sAB; \
    tmp6 += at_ar[6] * ping.sCD; \
    tmp7 += at_ar[6] * ping.sEF; \
    ping.lo = vload8(0, dot_result + src_base); \
    src_base += pixel_align * coalescing_feature_height_; \
    ping.hi = vload8(0, dot_result + src_base); \
    tmp0 += at_ar[7] * ping.s01; \
    tmp1 += at_ar[7] * ping.s23; \
    tmp2 += at_ar[7] * ping.s45; \
    tmp3 += at_ar[7] * ping.s67; \
    tmp4 += at_ar[7] * ping.s89; \
    tmp5 += at_ar[7] * ping.sAB; \
    tmp6 += at_ar[7] * ping.sCD; \
    tmp7 += at_ar[7] * ping.sEF; \
    float2 tmp0_f = convert_float2(tmp0); \
    float2 tmp1_f = convert_float2(tmp1); \
    float2 tmp2_f = convert_float2(tmp2); \
    float2 tmp3_f = convert_float2(tmp3); \
    float2 tmp4_f = convert_float2(tmp4); \
    float2 tmp5_f = convert_float2(tmp5); \
    float2 tmp6_f = convert_float2(tmp6); \
    float2 tmp7_f = convert_float2(tmp7); \
    float2 t0 = tmp1_f + tmp2_f; \
    float2 t1 = tmp3_f + tmp4_f; \
    float2 t2 = tmp5_f + tmp6_f; \
    float2 t3 = tmp1_f - tmp2_f; \
    float2 t4 = tmp3_f - tmp4_f; \
    float2 t5 = tmp5_f - tmp6_f; \
    DATA_T2 bias_val = (DATA_T2)(bias[c_index]); \
    if (w_index * 4 + 7 < output_w) { \
      DATA_T2 f0 = CONVERT_TO_DATA_T2(tmp0_f + t0 + t1 + t2) + bias_val; \
      DATA_T2 f1 = CONVERT_TO_DATA_T2(t3 * At5x5_f[9] + t4 * At5x5_f[11] + t5 * At5x5_f[13]) + bias_val; \
      DATA_T2 f2 = CONVERT_TO_DATA_T2(t0 * At5x5_f[17] + t1 * At5x5_f[19] + t2 * At5x5_f[21]) + bias_val; \
      DATA_T2 f3 = CONVERT_TO_DATA_T2(t3 * At5x5_f[25] + t4 * At5x5_f[27] + t5 * At5x5_f[29] + tmp7_f) + bias_val; \
      f0 = ACT_VEC_F(DATA_T2, f0); \
      f1 = ACT_VEC_F(DATA_T2, f1); \
      f2 = ACT_VEC_F(DATA_T2, f2); \
      f3 = ACT_VEC_F(DATA_T2, f3); \
      vstore8((DATA_T8)(f0.s0, f1.s0, f2.s0, f3.s0, f0.s1, f1.s1, f2.s1, f3.s1), 0, output + dst_base); \
    } else { \
      if ((w_index * 4 + 0) < output_w) { \
          DATA_T2 f0 = CONVERT_TO_DATA_T2(tmp0_f + t0 + t1 + t2) + bias_val; \
          f0 = ACT_VEC_F(DATA_T2, f0); \
          output[dst_base] = f0.s0; \
          if (w_index * 4 + 4 < output_w) \
          output[dst_base + 4] = f0.s1; \
      } \
      if ((w_index * 4 + 1) < output_w) { \
      DATA_T2 f0 = CONVERT_TO_DATA_T2(t3 * At5x5_f[9] + t4 * At5x5_f[11] + t5 * At5x5_f[13]) + bias_val; \
      f0 = ACT_VEC_F(DATA_T2, f0); \
      dst_base += 1; \
      output[dst_base] = f0.s0; \
      if (w_index * 4 + 5 < output_w) \
        output[dst_base + 4] = f0.s1; \
    } \
    if ((w_index * 4 + 2) < output_w) { \
      DATA_T2 f0 = CONVERT_TO_DATA_T2(t0 * At5x5_f[17] + t1 * At5x5_f[19] + t2 * At5x5_f[21]) + bias_val; \
          f0 = ACT_VEC_F(DATA_T2, f0); \
      dst_base += 1; \
      output[dst_base] = f0.s0; \
      if (w_index * 4 + 6 < output_w) \
        output[dst_base + 4] = f0.s1; \
    } \
    if ((w_index * 4 + 3) < output_w) { \
      DATA_T2 f0 = CONVERT_TO_DATA_T2(t3 * At5x5_f[25] + t4 * At5x5_f[27] + \
                                t5 * At5x5_f[29] + tmp7_f) + bias_val; \
      f0 = ACT_VEC_F(DATA_T2, f0); \
      dst_base += 1; \
      output[dst_base] = f0.s0; \
      if (w_index * 4 + 7 < output_w) \
        output[dst_base + 4] = f0.s1; \
    } \
    }

/********  FP16 KERNELS ********/
#define DATA_T half
#define DATA_T2 half2
#define DATA_T3 half3
#define DATA_T4 half4
#define DATA_T8 half8
#define DATA_T16 half16
#define CONVERT_TO_DATA_T2(x) convert_half2(x)
#define AS_DATA_T16(x) as_half16(x)

ADD_SINGLE_KERNEL(wino5x5_weight_tm_nchw_opt_4x16_FP16, (__global half *wt,
                                              __global half *wt_tm, int outch,
                                              int inch, int aligned_outch, int aligned_inch, int aligned_pixel) {
    WINO5X5_WEIGHT_TM_NCHW_OPT_4X16(wt, wt_tm, outch, inch, aligned_outch, aligned_inch, aligned_pixel)
})

ADD_KERNEL_HEADER(wino5x5_input_tm_nchw_opt_4x16_FP16, {DEFINE_BT8X8})
ADD_SINGLE_KERNEL(wino5x5_input_tm_nchw_opt_4x16_FP16, (
        __global half *src_data, __global half *dst_data, int src_n, int src_c,
        int src_h, int src_w, int dst_n, int dst_c, int dst_h, int dst_w,
        int padL, int padT, int tiles_total, int coalescing_feature_height_,
        int tiles_x, int aligned_inch, int aligned_pixel) {
    WINO5X5_INPUT_TM_NCHW_OPT_4X16(src_data, dst_data, src_n, src_c, src_h, src_w, dst_n, dst_c, dst_h, dst_w, padL, \
                                   padT, tiles_total, coalescing_feature_height_, tiles_x, aligned_inch, aligned_pixel)
})

ADD_SINGLE_KERNEL(wino5x5_dot_multiply_nchw_opt_4x16_FP16, (
        __global half *input_tm, __global half *weight_tm,
        __global half *dot_result, int dst_n, int dst_c, int dst_h, int dst_w,
        int input_c, int output_c, int coalescing_feature_height_,
        int aligned_outch, int aligned_inch, int aligned_pixel) {
    WINO5X5_DOT_MULTIPLY_NCHW_OPT_4X16(input_tm, weight_tm, dot_result, dst_n, dst_c, dst_h, dst_w, input_c, output_c, \
                                       coalescing_feature_height_, aligned_outch, aligned_inch, aligned_pixel)
})


// kernels without activation func
#define ACT_VEC_F(VEC_T, x) x
ADD_KERNEL_HEADER(wino5x5_output_tm_nchw_opt_4x16_FP16, {DEFINE_AT5X5, DEFINE_AT5X5_F})
ADD_SINGLE_KERNEL(wino5x5_output_tm_nchw_opt_4x16_FP16, (
        __global half *dot_result, __global half *bias, __global half *output,
        int src_n, int src_c, int src_h, int src_w, int output_n, int output_c,
        int output_h, int output_w, int coalescing_feature_height_,
        int tile_x, int aligned_outch, int aligned_inch, int aligned_pixel) {
    WINO5X5_OUTPUT_TM_NCHW_OPT_4X16(dot_result, bias, output, src_n, src_c, src_h, src_w, output_n, output_c, \
                                    output_h, output_w, coalescing_feature_height_, tile_x, aligned_outch, \
                                    aligned_inch, aligned_pixel)
})

#undef ACT_VEC_F  // NONE

// apply RELU to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU_VEC_F(VEC_T, x)
ADD_KERNEL_HEADER(RELUwino5x5_output_tm_nchw_opt_4x16_FP16, {DEFINE_AT5X5, DEFINE_AT5X5_F})
ADD_SINGLE_KERNEL(RELUwino5x5_output_tm_nchw_opt_4x16_FP16, (
        __global half *dot_result, __global half *bias, __global half *output,
        int src_n, int src_c, int src_h, int src_w, int output_n, int output_c,
        int output_h, int output_w, int coalescing_feature_height_,
        int tile_x, int aligned_outch, int aligned_inch, int aligned_pixel) {
    WINO5X5_OUTPUT_TM_NCHW_OPT_4X16(dot_result, bias, output, src_n, src_c, src_h, src_w, output_n, output_c, \
                                    output_h, output_w, coalescing_feature_height_, tile_x, aligned_outch, \
                                    aligned_inch, aligned_pixel)
})

#undef ACT_VEC_F  // RELU

// apply RELU6 to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU6_VEC_F(VEC_T, x)
ADD_KERNEL_HEADER(RELU6wino5x5_output_tm_nchw_opt_4x16_FP16, {DEFINE_AT5X5, DEFINE_AT5X5_F})
ADD_SINGLE_KERNEL(RELU6wino5x5_output_tm_nchw_opt_4x16_FP16, (
        __global half *dot_result, __global half *bias, __global half *output,
        int src_n, int src_c, int src_h, int src_w, int output_n, int output_c,
        int output_h, int output_w, int coalescing_feature_height_,
        int tile_x, int aligned_outch, int aligned_inch, int aligned_pixel) {
    WINO5X5_OUTPUT_TM_NCHW_OPT_4X16(dot_result, bias, output, src_n, src_c, src_h, src_w, output_n, output_c, \
                                    output_h, output_w, coalescing_feature_height_, tile_x, aligned_outch, \
                                    aligned_inch, aligned_pixel)
})

#undef ACT_VEC_F  // RELU6


#undef AS_DATA_T16
#undef CONVERT_TO_DATA_T2
#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // half

}  // namespace gpu
}  // namespace ud
}  // namespace enn
