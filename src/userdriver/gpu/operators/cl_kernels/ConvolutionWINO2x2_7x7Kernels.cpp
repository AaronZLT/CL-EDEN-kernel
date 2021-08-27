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
 * @file    ConvolutionWINO2x2_7x7Kernels.cpp
 * @brief
 * @details
 * @version
 */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-escape-sequence"

#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"
namespace enn{
namespace ud {
namespace gpu {


#define WINO7X7WEIGHT_TM_NCHW_OPT(wt, wt_tm, outch, inch, aligned_outch, aligned_inch) \
    int inc = get_global_id(0); \
    int outc = get_global_id(1); \
    int outch_aligned = (outch + aligned_outch - 1) / aligned_outch * aligned_outch; \
    int inch_aligned = (inch + aligned_inch - 1) / aligned_inch * aligned_inch; \
    if (inc >= inch_aligned || outc >= outch_aligned) { \
            return; \
    } \
    int dst_base = outc / aligned_outch * (aligned_outch * 64 * inch_aligned) + outc % aligned_outch; \
    dst_base += inc * aligned_outch; \
      if (inc >= inch || outc >= outch) { \
          wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; dst_base += aligned_outch * inch_aligned; \
              wt_tm[dst_base] = 0.0f; \
      } \
      int src_base = (outc * inch + inc) * 49; \
      DATA_T src_in[49]; \
      for (int i = 0; i < 49; ++i) { \
        src_in[i] = wt[src_base + i]; \
      } \
      DATA_T tmp_r0[7]; \
      DATA_T tmp_r1[7]; \
      DATA_T tmp_r2[7]; \
      DATA_T tmp_r3[7]; \
      DATA_T tmp_r4[7]; \
      DATA_T tmp_r5[7]; \
      DATA_T tmp_r6[7]; \
      DATA_T tmp_r7[7]; \
      tmp_r0[0] = src_in[0]; \
      tmp_r0[1] = src_in[1]; \
      tmp_r0[2] = src_in[2]; \
      tmp_r0[3] = src_in[3]; \
      tmp_r0[4] = src_in[4]; \
      tmp_r0[5] = src_in[5]; \
      tmp_r0[6] = src_in[6]; \
      tmp_r1[0] = src_in[0] + 0.707107f * src_in[7] + 0.5f * src_in[14] + \
                  0.355553f * src_in[21] + 0.25f * src_in[28] + \
                  0.176776f * src_in[35] + 0.125f * src_in[42]; \
      tmp_r1[1] = src_in[1] + 0.707107f * src_in[8] + 0.5f * src_in[15] + \
                  0.355553f * src_in[22] + 0.25f * src_in[29] + \
                  0.176776f * src_in[36] + 0.125f * src_in[43]; \
      tmp_r1[2] = src_in[2] + 0.707107f * src_in[9] + 0.5f * src_in[16] + \
                  0.355553f * src_in[23] + 0.25f * src_in[30] + \
                  0.176776f * src_in[37] + 0.125f * src_in[44]; \
      tmp_r1[3] = src_in[3] + 0.707107f * src_in[10] + 0.5f * src_in[17] + \
                  0.355553f * src_in[24] + 0.25f * src_in[31] + \
                  0.176776f * src_in[38] + 0.125f * src_in[45]; \
      tmp_r1[4] = src_in[4] + 0.707107f * src_in[11] + 0.5f * src_in[18] + \
                  0.355553f * src_in[25] + 0.25f * src_in[32] + \
                  0.176776f * src_in[39] + 0.125f * src_in[46]; \
      tmp_r1[5] = src_in[5] + 0.707107f * src_in[12] + 0.5f * src_in[19] + \
                  0.355553f * src_in[26] + 0.25f * src_in[33] + \
                  0.176776f * src_in[40] + 0.125f * src_in[47]; \
      tmp_r1[6] = src_in[6] + 0.707107f * src_in[13] + 0.5f * src_in[20] + \
                  0.355553f * src_in[27] + 0.25f * src_in[34] + \
                  0.176776f * src_in[41] + 0.125f * src_in[48]; \
      tmp_r2[0] = src_in[0] - 0.707107f * src_in[7] + 0.5f * src_in[14] - \
                  0.355553f * src_in[21] + 0.25f * src_in[28] - \
                  0.176776f * src_in[35] + 0.125f * src_in[42]; \
      tmp_r2[1] = src_in[1] - 0.707107f * src_in[8] + 0.5f * src_in[15] - \
                  0.355553f * src_in[22] + 0.25f * src_in[29] - \
                  0.176776f * src_in[36] + 0.125f * src_in[43]; \
      tmp_r2[2] = src_in[2] - 0.707107f * src_in[9] + 0.5f * src_in[16] - \
                  0.355553f * src_in[23] + 0.25f * src_in[30] - \
                  0.176776f * src_in[37] + 0.125f * src_in[44]; \
      tmp_r2[3] = src_in[3] - 0.707107f * src_in[10] + 0.5f * src_in[17] - \
                  0.355553f * src_in[24] + 0.25f * src_in[31] - \
                  0.176776f * src_in[38] + 0.125f * src_in[45]; \
      tmp_r2[4] = src_in[4] - 0.707107f * src_in[11] + 0.5f * src_in[18] - \
                  0.355553f * src_in[25] + 0.25f * src_in[32] - \
                  0.176776f * src_in[39] + 0.125f * src_in[46]; \
      tmp_r2[5] = src_in[5] - 0.707107f * src_in[12] + 0.5f * src_in[19] - \
                  0.355553f * src_in[26] + 0.25f * src_in[33] - \
                  0.176776f * src_in[40] + 0.125f * src_in[47]; \
      tmp_r2[6] = src_in[6] - 0.707107f * src_in[13] + 0.5f * src_in[20] - \
                  0.355553f * src_in[27] + 0.25f * src_in[34] - \
                  0.176776f * src_in[41] + 0.125f * src_in[48]; \
      tmp_r3[0] = src_in[0] + 1.414214f * src_in[7] + 2.0f * src_in[14] + \
                  2.828427f * src_in[21] + 4.0f * src_in[28] + \
                  5.656853f * src_in[35] + 8.0f * src_in[42]; \
      tmp_r3[1] = src_in[1] + 1.414214f * src_in[8] + 2.0f * src_in[15] + \
                  2.828427f * src_in[22] + 4.0f * src_in[29] + \
                  5.656853f * src_in[36] + 8.0f * src_in[43]; \
      tmp_r3[2] = src_in[2] + 1.414214f * src_in[9] + 2.0f * src_in[16] + \
                  2.828427f * src_in[23] + 4.0f * src_in[30] + \
                  5.656853f * src_in[37] + 8.0f * src_in[44]; \
      tmp_r3[3] = src_in[3] + 1.414214f * src_in[10] + 2.0f * src_in[17] + \
                  2.828427f * src_in[24] + 4.0f * src_in[31] + \
                  5.656853f * src_in[38] + 8.0f * src_in[45]; \
      tmp_r3[4] = src_in[4] + 1.414214f * src_in[11] + 2.0f * src_in[18] + \
                  2.828427f * src_in[25] + 4.0f * src_in[32] + \
                  5.656853f * src_in[39] + 8.0f * src_in[46]; \
      tmp_r3[5] = src_in[5] + 1.414214f * src_in[12] + 2.0f * src_in[19] + \
                  2.828427f * src_in[26] + 4.0f * src_in[33] + \
                  5.656853f * src_in[40] + 8.0f * src_in[47]; \
      tmp_r3[6] = src_in[6] + 1.414214f * src_in[13] + 2.0f * src_in[20] + \
                  2.828427f * src_in[27] + 4.0f * src_in[34] + \
                  5.656853f * src_in[41] + 8.0f * src_in[48]; \
      tmp_r4[0] = src_in[0] - 1.414214f * src_in[7] + 2.0f * src_in[14] - \
                  2.828427f * src_in[21] + 4.0f * src_in[28] - \
                  5.656853f * src_in[35] + 8.0f * src_in[42]; \
      tmp_r4[1] = src_in[1] - 1.414214f * src_in[8] + 2.0f * src_in[15] - \
                  2.828427f * src_in[22] + 4.0f * src_in[29] - \
                  5.656853f * src_in[36] + 8.0f * src_in[43]; \
      tmp_r4[2] = src_in[2] - 1.414214f * src_in[9] + 2.0f * src_in[16] - \
                  2.828427f * src_in[23] + 4.0f * src_in[30] - \
                  5.656853f * src_in[37] + 8.0f * src_in[44]; \
      tmp_r4[3] = src_in[3] - 1.414214f * src_in[10] + 2.0f * src_in[17] - \
                  2.828427f * src_in[24] + 4.0f * src_in[31] - \
                  5.656853f * src_in[38] + 8.0f * src_in[45]; \
      tmp_r4[4] = src_in[4] - 1.414214f * src_in[11] + 2.0f * src_in[18] - \
                  2.828427f * src_in[25] + 4.0f * src_in[32] - \
                  5.656853f * src_in[39] + 8.0f * src_in[46]; \
      tmp_r4[5] = src_in[5] - 1.414214f * src_in[12] + 2.0f * src_in[19] - \
                  2.828427f * src_in[26] + 4.0f * src_in[33] - \
                  5.656853f * src_in[40] + 8.0f * src_in[47]; \
      tmp_r4[6] = src_in[6] - 1.414214f * src_in[13] + 2.0f * src_in[20] - \
                  2.828427f * src_in[27] + 4.0f * src_in[34] - \
                  5.656853f * src_in[41] + 8.0f * src_in[48]; \
      tmp_r5[0] = src_in[0] + 2.121320f * src_in[7] + 4.5f * src_in[14] + \
                  9.545940f * src_in[21] + 20.25f * src_in[28] + \
                  42.956726f * src_in[35] + 91.124977f * src_in[42]; \
      tmp_r5[1] = src_in[1] + 2.121320f * src_in[8] + 4.5f * src_in[15] + \
                  9.545940f * src_in[22] + 20.25f * src_in[29] + \
                  42.956726f * src_in[36] + 91.124977f * src_in[43]; \
      tmp_r5[2] = src_in[2] + 2.121320f * src_in[9] + 4.5f * src_in[16] + \
                  9.545940f * src_in[23] + 20.25f * src_in[30] + \
                  42.956726f * src_in[37] + 91.124977f * src_in[44]; \
      tmp_r5[3] = src_in[3] + 2.121320f * src_in[10] + 4.5f * src_in[17] + \
                  9.545940f * src_in[24] + 20.25f * src_in[31] + \
                  42.956726f * src_in[38] + 91.124977f * src_in[45]; \
      tmp_r5[4] = src_in[4] + 2.121320f * src_in[11] + 4.5f * src_in[18] + \
                  9.545940f * src_in[25] + 20.25f * src_in[32] + \
                  42.956726f * src_in[39] + 91.124977f * src_in[46]; \
      tmp_r5[5] = src_in[5] + 2.121320f * src_in[12] + 4.5f * src_in[19] + \
                  9.545940f * src_in[26] + 20.25f * src_in[33] + \
                  42.956726f * src_in[40] + 91.124977f * src_in[47]; \
      tmp_r5[6] = src_in[6] + 2.121320f * src_in[13] + 4.5f * src_in[20] + \
                  9.545940f * src_in[27] + 20.25f * src_in[34] + \
                  42.956726f * src_in[41] + 91.124977f * src_in[48]; \
      tmp_r6[0] = src_in[0] - 2.121320f * src_in[7] + 4.5f * src_in[14] - \
                  9.545940f * src_in[21] + 20.25f * src_in[28] - \
                  42.956726f * src_in[35] + 91.124977f * src_in[42]; \
      tmp_r6[1] = src_in[1] - 2.121320f * src_in[8] + 4.5f * src_in[15] - \
                  9.545940f * src_in[22] + 20.25f * src_in[29] - \
                  42.956726f * src_in[36] + 91.124977f * src_in[43]; \
      tmp_r6[2] = src_in[2] - 2.121320f * src_in[9] + 4.5f * src_in[16] - \
                  9.545940f * src_in[23] + 20.25f * src_in[30] - \
                  42.956726f * src_in[37] + 91.124977f * src_in[44]; \
      tmp_r6[3] = src_in[3] - 2.121320f * src_in[10] + 4.5f * src_in[17] - \
                  9.545940f * src_in[24] + 20.25f * src_in[31] - \
                  42.956726f * src_in[38] + 91.124977f * src_in[45]; \
      tmp_r6[4] = src_in[4] - 2.121320f * src_in[11] + 4.5f * src_in[18] - \
                  9.545940f * src_in[25] + 20.25f * src_in[32] - \
                  42.956726f * src_in[39] + 91.124977f * src_in[46]; \
      tmp_r6[5] = src_in[5] - 2.121320f * src_in[12] + 4.5f * src_in[19] - \
                  9.545940f * src_in[26] + 20.25f * src_in[33] - \
                  42.956726f * src_in[40] + 91.124977f * src_in[47]; \
      tmp_r6[6] = src_in[6] - 2.121320f * src_in[13] + 4.5f * src_in[20] - \
                  9.545940f * src_in[27] + 20.25f * src_in[34] - \
                  42.956726f * src_in[41] + 91.124977f * src_in[48]; \
      tmp_r7[0] = src_in[42]; \
      tmp_r7[1] = src_in[43]; \
      tmp_r7[2] = src_in[44]; \
      tmp_r7[3] = src_in[45]; \
      tmp_r7[4] = src_in[46]; \
      tmp_r7[5] = src_in[47]; \
      tmp_r7[6] = src_in[48]; \
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
                  0.355553f * tmp_r0[3] + 0.25f * tmp_r0[4] + \
                  0.176776f * tmp_r0[5] + 0.125f * tmp_r0[6]; \
      dst_r0[2] = tmp_r0[0] - 0.707107f * tmp_r0[1] + 0.5f * tmp_r0[2] - \
                  0.355553f * tmp_r0[3] + 0.25f * tmp_r0[4] - \
                  0.176776f * tmp_r0[5] + 0.125f * tmp_r0[6]; \
      dst_r0[3] = tmp_r0[0] + 1.414214f * tmp_r0[1] + 2.0f * tmp_r0[2] + \
                  2.828427f * tmp_r0[3] + 4.0f * tmp_r0[4] + \
                  5.656853f * tmp_r0[5] + 8.0f * tmp_r0[6]; \
      dst_r0[4] = tmp_r0[0] - 1.414214f * tmp_r0[1] + 2.0f * tmp_r0[2] - \
                  2.828427f * tmp_r0[3] + 4.0f * tmp_r0[4] - \
                  5.656853f * tmp_r0[5] + 8.0f * tmp_r0[6]; \
      dst_r0[5] = tmp_r0[0] + 2.121320f * tmp_r0[1] + 4.5f * tmp_r0[2] + \
                  9.545940f * tmp_r0[3] + 20.25f * tmp_r0[4] + \
                  42.956726f * tmp_r0[5] + 91.124977f * tmp_r0[6]; \
      dst_r0[6] = tmp_r0[0] - 2.121320f * tmp_r0[1] + 4.5f * tmp_r0[2] - \
                  9.545940f * tmp_r0[3] + 20.25f * tmp_r0[4] - \
                  42.956726f * tmp_r0[5] + 91.124977f * tmp_r0[6]; \
      dst_r0[7] = tmp_r0[6]; \
      dst_r1[0] = tmp_r1[0]; \
      dst_r1[1] = tmp_r1[0] + 0.707107f * tmp_r1[1] + 0.5f * tmp_r1[2] + \
                  0.355553f * tmp_r1[3] + 0.25f * tmp_r1[4] + \
                  0.176776f * tmp_r1[5] + 0.125f * tmp_r1[6]; \
      dst_r1[2] = tmp_r1[0] - 0.707107f * tmp_r1[1] + 0.5f * tmp_r1[2] - \
                  0.355553f * tmp_r1[3] + 0.25f * tmp_r1[4] - \
                  0.176776f * tmp_r1[5] + 0.125f * tmp_r1[6]; \
      dst_r1[3] = tmp_r1[0] + 1.414214f * tmp_r1[1] + 2.0f * tmp_r1[2] + \
                  2.828427f * tmp_r1[3] + 4.0f * tmp_r1[4] + \
                  5.656853f * tmp_r1[5] + 8.0f * tmp_r1[6]; \
      dst_r1[4] = tmp_r1[0] - 1.414214f * tmp_r1[1] + 2.0f * tmp_r1[2] - \
                  2.828427f * tmp_r1[3] + 4.0f * tmp_r1[4] - \
                  5.656853f * tmp_r1[5] + 8.0f * tmp_r1[6]; \
      dst_r1[5] = tmp_r1[0] + 2.121320f * tmp_r1[1] + 4.5f * tmp_r1[2] + \
                  9.545940f * tmp_r1[3] + 20.25f * tmp_r1[4] + \
                  42.956726f * tmp_r1[5] + 91.124977f * tmp_r1[6]; \
      dst_r1[6] = tmp_r1[0] - 2.121320f * tmp_r1[1] + 4.5f * tmp_r1[2] - \
                  9.545940f * tmp_r1[3] + 20.25f * tmp_r1[4] - \
                  42.956726f * tmp_r1[5] + 91.124977f * tmp_r1[6]; \
      dst_r1[7] = tmp_r1[6]; \
      dst_r2[0] = tmp_r2[0]; \
      dst_r2[1] = tmp_r2[0] + 0.707107f * tmp_r2[1] + 0.5f * tmp_r2[2] + \
                  0.355553f * tmp_r2[3] + 0.25f * tmp_r2[4] + \
                  0.176776f * tmp_r2[5] + 0.125f * tmp_r2[6]; \
      dst_r2[2] = tmp_r2[0] - 0.707107f * tmp_r2[1] + 0.5f * tmp_r2[2] - \
                  0.355553f * tmp_r2[3] + 0.25f * tmp_r2[4] - \
                  0.176776f * tmp_r2[5] + 0.125f * tmp_r2[6]; \
      dst_r2[3] = tmp_r2[0] + 1.414214f * tmp_r2[1] + 2.0f * tmp_r2[2] + \
                  2.828427f * tmp_r2[3] + 4.0f * tmp_r2[4] + \
                  5.656853f * tmp_r2[5] + 8.0f * tmp_r2[6]; \
      dst_r2[4] = tmp_r2[0] - 1.414214f * tmp_r2[1] + 2.0f * tmp_r2[2] - \
                  2.828427f * tmp_r2[3] + 4.0f * tmp_r2[4] - \
                  5.656853f * tmp_r2[5] + 8.0f * tmp_r2[6]; \
      dst_r2[5] = tmp_r2[0] + 2.121320f * tmp_r2[1] + 4.5f * tmp_r2[2] + \
                  9.545940f * tmp_r2[3] + 20.25f * tmp_r2[4] + \
                  42.956726f * tmp_r2[5] + 91.124977f * tmp_r2[6]; \
      dst_r2[6] = tmp_r2[0] - 2.121320f * tmp_r2[1] + 4.5f * tmp_r2[2] - \
                  9.545940f * tmp_r2[3] + 20.25f * tmp_r2[4] - \
                  42.956726f * tmp_r2[5] + 91.124977f * tmp_r2[6]; \
      dst_r2[7] = tmp_r2[6]; \
      dst_r3[0] = tmp_r3[0]; \
      dst_r3[1] = tmp_r3[0] + 0.707107f * tmp_r3[1] + 0.5f * tmp_r3[2] + \
                  0.355553f * tmp_r3[3] + 0.25f * tmp_r3[4] + \
                  0.176776f * tmp_r3[5] + 0.125f * tmp_r3[6]; \
      dst_r3[2] = tmp_r3[0] - 0.707107f * tmp_r3[1] + 0.5f * tmp_r3[2] - \
                  0.355553f * tmp_r3[3] + 0.25f * tmp_r3[4] - \
                  0.176776f * tmp_r3[5] + 0.125f * tmp_r3[6]; \
      dst_r3[3] = tmp_r3[0] + 1.414214f * tmp_r3[1] + 2.0f * tmp_r3[2] + \
                  2.828427f * tmp_r3[3] + 4.0f * tmp_r3[4] + \
                  5.656853f * tmp_r3[5] + 8.0f * tmp_r3[6]; \
      dst_r3[4] = tmp_r3[0] - 1.414214f * tmp_r3[1] + 2.0f * tmp_r3[2] - \
                  2.828427f * tmp_r3[3] + 4.0f * tmp_r3[4] - \
                  5.656853f * tmp_r3[5] + 8.0f * tmp_r3[6]; \
      dst_r3[5] = tmp_r3[0] + 2.121320f * tmp_r3[1] + 4.5f * tmp_r3[2] + \
                  9.545940f * tmp_r3[3] + 20.25f * tmp_r3[4] + \
                  42.956726f * tmp_r3[5] + 91.124977f * tmp_r3[6]; \
      dst_r3[6] = tmp_r3[0] - 2.121320f * tmp_r3[1] + 4.5f * tmp_r3[2] - \
                  9.545940f * tmp_r3[3] + 20.25f * tmp_r3[4] - \
                  42.956726f * tmp_r3[5] + 91.124977f * tmp_r3[6]; \
      dst_r3[7] = tmp_r3[6]; \
      dst_r4[0] = tmp_r4[0]; \
      dst_r4[1] = tmp_r4[0] + 0.707107f * tmp_r4[1] + 0.5f * tmp_r4[2] + \
                  0.355553f * tmp_r4[3] + 0.25f * tmp_r4[4] + \
                  0.176776f * tmp_r4[5] + 0.125f * tmp_r4[6]; \
      dst_r4[2] = tmp_r4[0] - 0.707107f * tmp_r4[1] + 0.5f * tmp_r4[2] - \
                  0.355553f * tmp_r4[3] + 0.25f * tmp_r4[4] - \
                  0.176776f * tmp_r4[5] + 0.125f * tmp_r4[6]; \
      dst_r4[3] = tmp_r4[0] + 1.414214f * tmp_r4[1] + 2.0f * tmp_r4[2] + \
                  2.828427f * tmp_r4[3] + 4.0f * tmp_r4[4] + \
                  5.656853f * tmp_r4[5] + 8.0f * tmp_r4[6]; \
      dst_r4[4] = tmp_r4[0] - 1.414214f * tmp_r4[1] + 2.0f * tmp_r4[2] - \
                  2.828427f * tmp_r4[3] + 4.0f * tmp_r4[4] - \
                  5.656853f * tmp_r4[5] + 8.0f * tmp_r4[6]; \
      dst_r4[5] = tmp_r4[0] + 2.121320f * tmp_r4[1] + 4.5f * tmp_r4[2] + \
                  9.545940f * tmp_r4[3] + 20.25f * tmp_r4[4] + \
                  42.956726f * tmp_r4[5] + 91.124977f * tmp_r4[6]; \
      dst_r4[6] = tmp_r4[0] - 2.121320f * tmp_r4[1] + 4.5f * tmp_r4[2] - \
                  9.545940f * tmp_r4[3] + 20.25f * tmp_r4[4] - \
                  42.956726f * tmp_r4[5] + 91.124977f * tmp_r4[6]; \
      dst_r4[7] = tmp_r4[6]; \
      dst_r5[0] = tmp_r5[0]; \
      dst_r5[1] = tmp_r5[0] + 0.707107f * tmp_r5[1] + 0.5f * tmp_r5[2] + \
                  0.355553f * tmp_r5[3] + 0.25f * tmp_r5[4] + \
                  0.176776f * tmp_r5[5] + 0.125f * tmp_r5[6]; \
      dst_r5[2] = tmp_r5[0] - 0.707107f * tmp_r5[1] + 0.5f * tmp_r5[2] - \
                  0.355553f * tmp_r5[3] + 0.25f * tmp_r5[4] - \
                  0.176776f * tmp_r5[5] + 0.125f * tmp_r5[6]; \
      dst_r5[3] = tmp_r5[0] + 1.414214f * tmp_r5[1] + 2.0f * tmp_r5[2] + \
                  2.828427f * tmp_r5[3] + 4.0f * tmp_r5[4] + \
                  5.656853f * tmp_r5[5] + 8.0f * tmp_r5[6]; \
      dst_r5[4] = tmp_r5[0] - 1.414214f * tmp_r5[1] + 2.0f * tmp_r5[2] - \
                  2.828427f * tmp_r5[3] + 4.0f * tmp_r5[4] - \
                  5.656853f * tmp_r5[5] + 8.0f * tmp_r5[6]; \
      dst_r5[5] = tmp_r5[0] + 2.121320f * tmp_r5[1] + 4.5f * tmp_r5[2] + \
                  9.545940f * tmp_r5[3] + 20.25f * tmp_r5[4] + \
                  42.956726f * tmp_r5[5] + 91.124977f * tmp_r5[6]; \
      dst_r5[6] = tmp_r5[0] - 2.121320f * tmp_r5[1] + 4.5f * tmp_r5[2] - \
                  9.545940f * tmp_r5[3] + 20.25f * tmp_r5[4] - \
                  42.956726f * tmp_r5[5] + 91.124977f * tmp_r5[6]; \
      dst_r5[7] = tmp_r5[6]; \
      dst_r6[0] = tmp_r6[0]; \
      dst_r6[1] = tmp_r6[0] + 0.707107f * tmp_r6[1] + 0.5f * tmp_r6[2] + \
                  0.355553f * tmp_r6[3] + 0.25f * tmp_r6[4] + \
                  0.176776f * tmp_r6[5] + 0.125f * tmp_r6[6]; \
      dst_r6[2] = tmp_r6[0] - 0.707107f * tmp_r6[1] + 0.5f * tmp_r6[2] - \
                  0.355553f * tmp_r6[3] + 0.25f * tmp_r6[4] - \
                  0.176776f * tmp_r6[5] + 0.125f * tmp_r6[6]; \
      dst_r6[3] = tmp_r6[0] + 1.414214f * tmp_r6[1] + 2.0f * tmp_r6[2] + \
                  2.828427f * tmp_r6[3] + 4.0f * tmp_r6[4] + \
                  5.656853f * tmp_r6[5] + 8.0f * tmp_r6[6]; \
      dst_r6[4] = tmp_r6[0] - 1.414214f * tmp_r6[1] + 2.0f * tmp_r6[2] - \
                  2.828427f * tmp_r6[3] + 4.0f * tmp_r6[4] - \
                  5.656853f * tmp_r6[5] + 8.0f * tmp_r6[6]; \
      dst_r6[5] = tmp_r6[0] + 2.121320f * tmp_r6[1] + 4.5f * tmp_r6[2] + \
                  9.545940f * tmp_r6[3] + 20.25f * tmp_r6[4] + \
                  42.956726f * tmp_r6[5] + 91.124977f * tmp_r6[6]; \
      dst_r6[6] = tmp_r6[0] - 2.121320f * tmp_r6[1] + 4.5f * tmp_r6[2] - \
                  9.545940f * tmp_r6[3] + 20.25f * tmp_r6[4] - \
                  42.956726f * tmp_r6[5] + 91.124977f * tmp_r6[6]; \
      dst_r6[7] = tmp_r6[6]; \
      dst_r7[0] = tmp_r7[0]; \
      dst_r7[1] = tmp_r7[0] + 0.707107f * tmp_r7[1] + 0.5f * tmp_r7[2] + \
                  0.355553f * tmp_r7[3] + 0.25f * tmp_r7[4] + \
                  0.176776f * tmp_r7[5] + 0.125f * tmp_r7[6]; \
      dst_r7[2] = tmp_r7[0] - 0.707107f * tmp_r7[1] + 0.5f * tmp_r7[2] - \
                  0.355553f * tmp_r7[3] + 0.25f * tmp_r7[4] - \
                  0.176776f * tmp_r7[5] + 0.125f * tmp_r7[6]; \
      dst_r7[3] = tmp_r7[0] + 1.414214f * tmp_r7[1] + 2.0f * tmp_r7[2] + \
                  2.828427f * tmp_r7[3] + 4.0f * tmp_r7[4] + \
                  5.656853f * tmp_r7[5] + 8.0f * tmp_r7[6]; \
      dst_r7[4] = tmp_r7[0] - 1.414214f * tmp_r7[1] + 2.0f * tmp_r7[2] - \
                  2.828427f * tmp_r7[3] + 4.0f * tmp_r7[4] - \
                  5.656853f * tmp_r7[5] + 8.0f * tmp_r7[6]; \
      dst_r7[5] = tmp_r7[0] + 2.121320f * tmp_r7[1] + 4.5f * tmp_r7[2] + \
                  9.545940f * tmp_r7[3] + 20.25f * tmp_r7[4] + \
                  42.956726f * tmp_r7[5] + 91.124977f * tmp_r7[6]; \
      dst_r7[6] = tmp_r7[0] - 2.121320f * tmp_r7[1] + 4.5f * tmp_r7[2] - \
                  9.545940f * tmp_r7[3] + 20.25f * tmp_r7[4] - \
                  42.956726f * tmp_r7[5] + 91.124977f * tmp_r7[6]; \
      dst_r7[7] = tmp_r7[6]; \
      wt_tm[dst_base] = dst_r0[0]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r0[1]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r0[2]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r0[3]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r0[4]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r0[5]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r0[6]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r0[7]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r1[0]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r1[1]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r1[2]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r1[3]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r1[4]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r1[5]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r1[6]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r1[7]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r2[0]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r2[1]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r2[2]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r2[3]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r2[4]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r2[5]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r2[6]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r2[7]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r3[0]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r3[1]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r3[2]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r3[3]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r3[4]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r3[5]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r3[6]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r3[7]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r4[0]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r4[1]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r4[2]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r4[3]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r4[4]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r4[5]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r4[6]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r4[7]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r5[0]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r5[1]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r5[2]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r5[3]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r5[4]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r5[5]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r5[6]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r5[7]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r6[0]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r6[1]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r6[2]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r6[3]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r6[4]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r6[5]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r6[6]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r6[7]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r7[0]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r7[1]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r7[2]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r7[3]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r7[4]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r7[5]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r7[6]; dst_base += aligned_outch * inch_aligned; \
      wt_tm[dst_base] = dst_r7[7]; dst_base += aligned_outch * inch_aligned;

#define WINO7X7INPUT_TM_NCHW_OPT(src_data, dst_data, src_n, src_c, src_h, src_w, dst_n, dst_c, dst_h, dst_w, padL, \
                                 padT, tiles_total, aligned_tiles, coalescing_feature_height_, tiles_x, aligned_inch) \
    int w_index = get_global_id(0) % tiles_x * 2; \
    int h_index = get_global_id(0) / tiles_x; \
    int p_index = get_global_id(1) * 2; \
    int inch_aligned = (src_c + aligned_inch - 1) / aligned_inch * aligned_inch; \
    int c_index = get_global_id(2) % inch_aligned; \
    int n_index = get_global_id(2) / inch_aligned; \
    if (get_global_id(0) * 2 >= aligned_tiles || c_index >= dst_c) { \
            return; \
    } \
    int four_tiles = 2; \
    int dst_base = (n_index * dst_c * dst_h * dst_w); \
    int dst_tile = dst_base + \
                    ((get_global_id(0) * four_tiles) / \
                        coalescing_feature_height_ * inch_aligned * 64 + \
                    c_index / aligned_inch * aligned_inch) * \
                        coalescing_feature_height_ + \
                    c_index % aligned_inch * four_tiles; \
    dst_tile += (get_global_id(0) * 2 % coalescing_feature_height_) / 2 * 2 * aligned_inch; \
    dst_tile += (get_global_id(0) * 2 % coalescing_feature_height_) % 2; \
    dst_tile += p_index * 8 * inch_aligned * coalescing_feature_height_; \
    if (c_index >= src_c || get_global_id(0) * 2 >= tiles_total) { \
      DATA_T2 zero2 = (DATA_T2)(0.0f, 0.0f); \
      vstore2(zero2, 0, dst_data + dst_tile); dst_tile += inch_aligned * coalescing_feature_height_; \
      vstore2(zero2, 0, dst_data + dst_tile); dst_tile += inch_aligned * coalescing_feature_height_; \
      vstore2(zero2, 0, dst_data + dst_tile); dst_tile += inch_aligned * coalescing_feature_height_; \
      vstore2(zero2, 0, dst_data + dst_tile); dst_tile += inch_aligned * coalescing_feature_height_; \
      vstore2(zero2, 0, dst_data + dst_tile); dst_tile += inch_aligned * coalescing_feature_height_; \
      vstore2(zero2, 0, dst_data + dst_tile); dst_tile += inch_aligned * coalescing_feature_height_; \
      vstore2(zero2, 0, dst_data + dst_tile); dst_tile += inch_aligned * coalescing_feature_height_; \
      vstore2(zero2, 0, dst_data + dst_tile); dst_tile += inch_aligned * coalescing_feature_height_; \
      vstore2(zero2, 0, dst_data + dst_tile); dst_tile += inch_aligned * coalescing_feature_height_; \
      vstore2(zero2, 0, dst_data + dst_tile); dst_tile += inch_aligned * coalescing_feature_height_; \
      vstore2(zero2, 0, dst_data + dst_tile); dst_tile += inch_aligned * coalescing_feature_height_; \
      vstore2(zero2, 0, dst_data + dst_tile); dst_tile += inch_aligned * coalescing_feature_height_; \
      vstore2(zero2, 0, dst_data + dst_tile); dst_tile += inch_aligned * coalescing_feature_height_; \
      vstore2(zero2, 0, dst_data + dst_tile); dst_tile += inch_aligned * coalescing_feature_height_; \
      vstore2(zero2, 0, dst_data + dst_tile); dst_tile += inch_aligned * coalescing_feature_height_; \
      vstore2(zero2, 0, dst_data + dst_tile); dst_tile += inch_aligned * coalescing_feature_height_; \
    } \
    int in_w_start = w_index * 2 - padL; \
    int in_h_start = h_index * 2 - padT; \
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
        row0.s89 = vload2(0, src_data + in_base + 8); \
    } \
    if (in_h_start + 1 >= 0 && in_h_start + 1 < src_h) { \
        row1.lo = vload8(0, src_data + in_base + src_w); \
        row1.s89 = vload2(0, src_data + in_base + src_w + 8); \
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
    inputMask.sA = 0x0000; \
    inputMask.sB = 0x0000; \
    inputMask.sC = 0x0000; \
    inputMask.sD = 0x0000; \
    inputMask.sE = 0x0000; \
    inputMask.sF = 0x0000; \
    row0 = AS_DATA_T16(as_short16(row0) & inputMask); \
    row1 = AS_DATA_T16(as_short16(row1) & inputMask); \
    DATA_T bt = bt_ar[0]; \
    DATA_T2 src0 = (DATA_T2)(row0.s0, row0.s2); \
    I0 = bt * src0; \
    DATA_T2 src1 = (DATA_T2)(row0.s1, row0.s3); \
    I1 = bt * src1; \
    DATA_T2 src2 = (DATA_T2)(row0.s2, row0.s4); \
    I2 = bt * src2; \
    DATA_T2 src3 = (DATA_T2)(row0.s3, row0.s5); \
    I3 = bt * src3; \
    DATA_T2 src4 = (DATA_T2)(row0.s4, row0.s6); \
    I4 = bt * src4; \
    DATA_T2 src5 = (DATA_T2)(row0.s5, row0.s7); \
    I5 = bt * src5; \
    DATA_T2 src6 = (DATA_T2)(row0.s6, row0.s8); \
    I6 = bt * src6; \
    DATA_T2 src7 = (DATA_T2)(row0.s7, row0.s9); \
    I7 = bt * src7; \
    bt = bt_ar[1]; \
    src0 = (DATA_T2)(row1.s0, row1.s2); \
    I0 += bt * src0; \
    src1 = (DATA_T2)(row1.s1, row1.s3); \
    I1 += bt * src1; \
    src2 = (DATA_T2)(row1.s2, row1.s4); \
    I2 += bt * src2; \
    src3 = (DATA_T2)(row1.s3, row1.s5); \
    I3 += bt * src3; \
    src4 = (DATA_T2)(row1.s4, row1.s6); \
    I4 += bt * src4; \
    src5 = (DATA_T2)(row1.s5, row1.s7); \
    I5 += bt * src5; \
    src6 = (DATA_T2)(row1.s6, row1.s8); \
    I6 += bt * src6; \
    src7 = (DATA_T2)(row1.s7, row1.s9); \
    I7 += bt * src7; \
    bt = bt_br[0]; \
    src0 = (DATA_T2)(row0.s0, row0.s2); \
    J0 = bt * src0; \
    src1 = (DATA_T2)(row0.s1, row0.s3); \
    J1 = bt * src1; \
    src2 = (DATA_T2)(row0.s2, row0.s4); \
    J2 = bt * src2; \
    src3 = (DATA_T2)(row0.s3, row0.s5); \
    J3 = bt * src3; \
    src4 = (DATA_T2)(row0.s4, row0.s6); \
    J4 = bt * src4; \
    src5 = (DATA_T2)(row0.s5, row0.s7); \
    J5 = bt * src5; \
    src6 = (DATA_T2)(row0.s6, row0.s8); \
    J6 = bt * src6; \
    src7 = (DATA_T2)(row0.s7, row0.s9); \
    J7 = bt * src7; \
    bt = bt_br[1]; \
    src0 = (DATA_T2)(row1.s0, row1.s2); \
    J0 += bt * src0; \
    src1 = (DATA_T2)(row1.s1, row1.s3); \
    J1 += bt * src1; \
    src2 = (DATA_T2)(row1.s2, row1.s4); \
    J2 += bt * src2; \
    src3 = (DATA_T2)(row1.s3, row1.s5); \
    J3 += bt * src3; \
    src4 = (DATA_T2)(row1.s4, row1.s6); \
    J4 += bt * src4; \
    src5 = (DATA_T2)(row1.s5, row1.s7); \
    J5 += bt * src5; \
    src6 = (DATA_T2)(row1.s6, row1.s8); \
    J6 += bt * src6; \
    src7 = (DATA_T2)(row1.s7, row1.s9); \
    J7 += bt * src7; \
    row0 = (DATA_T16)(0.0f); \
    row1 = (DATA_T16)(0.0f); \
    if (in_h_start + 2 >= 0 && in_h_start + 2 < src_h) { \
        row0.lo = vload8(0, src_data + in_base + src_w * 2); \
        row0.s89 = vload2(0, src_data + in_base + src_w * 2 + 8); \
    } \
    if (in_h_start + 3 >= 0 && in_h_start + 3  < src_h) { \
        row1.lo = vload8(0, src_data + in_base + src_w * 3); \
        row1.s89 = vload2(0, src_data + in_base + src_w * 3 + 8); \
    } \
    row0 = AS_DATA_T16(as_short16(row0) & inputMask); \
    row1 = AS_DATA_T16(as_short16(row1) & inputMask); \
    bt = bt_ar[2]; \
    src0 = (DATA_T2)(row0.s0, row0.s2); \
    I0 += bt * src0; \
    src1 = (DATA_T2)(row0.s1, row0.s3); \
    I1 += bt * src1; \
    src2 = (DATA_T2)(row0.s2, row0.s4); \
    I2 += bt * src2; \
    src3 = (DATA_T2)(row0.s3, row0.s5); \
    I3 += bt * src3; \
    src4 = (DATA_T2)(row0.s4, row0.s6); \
    I4 += bt * src4; \
    src5 = (DATA_T2)(row0.s5, row0.s7); \
    I5 += bt * src5; \
    src6 = (DATA_T2)(row0.s6, row0.s8); \
    I6 += bt * src6; \
    src7 = (DATA_T2)(row0.s7, row0.s9); \
    I7 += bt * src7; \
    bt = bt_ar[3]; \
    src0 = (DATA_T2)(row1.s0, row1.s2); \
    I0 += bt * src0; \
    src1 = (DATA_T2)(row1.s1, row1.s3); \
    I1 += bt * src1; \
    src2 = (DATA_T2)(row1.s2, row1.s4); \
    I2 += bt * src2; \
    src3 = (DATA_T2)(row1.s3, row1.s5); \
    I3 += bt * src3; \
    src4 = (DATA_T2)(row1.s4, row1.s6); \
    I4 += bt * src4; \
    src5 = (DATA_T2)(row1.s5, row1.s7); \
    I5 += bt * src5; \
    src6 = (DATA_T2)(row1.s6, row1.s8); \
    I6 += bt * src6; \
    src7 = (DATA_T2)(row1.s7, row1.s9); \
    I7 += bt * src7; \
    bt = bt_br[2]; \
    src0 = (DATA_T2)(row0.s0, row0.s2); \
    J0 += bt * src0; \
    src1 = (DATA_T2)(row0.s1, row0.s3); \
    J1 += bt * src1; \
    src2 = (DATA_T2)(row0.s2, row0.s4); \
    J2 += bt * src2; \
    src3 = (DATA_T2)(row0.s3, row0.s5); \
    J3 += bt * src3; \
    src4 = (DATA_T2)(row0.s4, row0.s6); \
    J4 += bt * src4; \
    src5 = (DATA_T2)(row0.s5, row0.s7); \
    J5 += bt * src5; \
    src6 = (DATA_T2)(row0.s6, row0.s8); \
    J6 += bt * src6; \
    src7 = (DATA_T2)(row0.s7, row0.s9); \
    J7 += bt * src7; \
    bt = bt_br[3]; \
    src0 = (DATA_T2)(row1.s0, row1.s2); \
    J0 += bt * src0; \
    src1 = (DATA_T2)(row1.s1, row1.s3); \
    J1 += bt * src1; \
    src2 = (DATA_T2)(row1.s2, row1.s4); \
    J2 += bt * src2; \
    src3 = (DATA_T2)(row1.s3, row1.s5); \
    J3 += bt * src3; \
    src4 = (DATA_T2)(row1.s4, row1.s6); \
    J4 += bt * src4; \
    src5 = (DATA_T2)(row1.s5, row1.s7); \
    J5 += bt * src5; \
    src6 = (DATA_T2)(row1.s6, row1.s8); \
    J6 += bt * src6; \
    src7 = (DATA_T2)(row1.s7, row1.s9); \
    J7 += bt * src7; \
    row0 = (DATA_T16)(0.0f); \
    row1 = (DATA_T16)(0.0f); \
    if (in_h_start + 4 >= 0 && in_h_start + 4 < src_h) { \
        row0.lo = vload8(0, src_data + in_base + src_w * 4); \
        row0.s89 = vload2(0, src_data + in_base + src_w * 4 + 8); \
    } \
    if (in_h_start + 5 >= 0 && in_h_start + 5 < src_h) { \
        row1.lo = vload8(0, src_data + in_base + src_w * 5); \
        row1.s89 = vload2(0, src_data + in_base + src_w * 5 + 8); \
    } \
    row0 = AS_DATA_T16(as_short16(row0) & inputMask); \
    row1 = AS_DATA_T16(as_short16(row1) & inputMask); \
    bt = bt_ar[4]; \
    src0 = (DATA_T2)(row0.s0, row0.s2); \
    I0 += bt * src0; \
    src1 = (DATA_T2)(row0.s1, row0.s3); \
    I1 += bt * src1; \
    src2 = (DATA_T2)(row0.s2, row0.s4); \
    I2 += bt * src2; \
    src3 = (DATA_T2)(row0.s3, row0.s5); \
    I3 += bt * src3; \
    src4 = (DATA_T2)(row0.s4, row0.s6); \
    I4 += bt * src4; \
    src5 = (DATA_T2)(row0.s5, row0.s7); \
    I5 += bt * src5; \
    src6 = (DATA_T2)(row0.s6, row0.s8); \
    I6 += bt * src6; \
    src7 = (DATA_T2)(row0.s7, row0.s9); \
    I7 += bt * src7; \
    bt = bt_ar[5]; \
    src0 = (DATA_T2)(row1.s0, row1.s2); \
    I0 += bt * src0; \
    src1 = (DATA_T2)(row1.s1, row1.s3); \
    I1 += bt * src1; \
    src2 = (DATA_T2)(row1.s2, row1.s4); \
    I2 += bt * src2; \
    src3 = (DATA_T2)(row1.s3, row1.s5); \
    I3 += bt * src3; \
    src4 = (DATA_T2)(row1.s4, row1.s6); \
    I4 += bt * src4; \
    src5 = (DATA_T2)(row1.s5, row1.s7); \
    I5 += bt * src5; \
    src6 = (DATA_T2)(row1.s6, row1.s8); \
    I6 += bt * src6; \
    src7 = (DATA_T2)(row1.s7, row1.s9); \
    I7 += bt * src7; \
    bt = bt_br[4]; \
    src0 = (DATA_T2)(row0.s0, row0.s2); \
    J0 += bt * src0; \
    src1 = (DATA_T2)(row0.s1, row0.s3); \
    J1 += bt * src1; \
    src2 = (DATA_T2)(row0.s2, row0.s4); \
    J2 += bt * src2; \
    src3 = (DATA_T2)(row0.s3, row0.s5); \
    J3 += bt * src3; \
    src4 = (DATA_T2)(row0.s4, row0.s6); \
    J4 += bt * src4; \
    src5 = (DATA_T2)(row0.s5, row0.s7); \
    J5 += bt * src5; \
    src6 = (DATA_T2)(row0.s6, row0.s8); \
    J6 += bt * src6; \
    src7 = (DATA_T2)(row0.s7, row0.s9); \
    J7 += bt * src7; \
    bt = bt_br[5]; \
    src0 = (DATA_T2)(row1.s0, row1.s2); \
    J0 += bt * src0; \
    src1 = (DATA_T2)(row1.s1, row1.s3); \
    J1 += bt * src1; \
    src2 = (DATA_T2)(row1.s2, row1.s4); \
    J2 += bt * src2; \
    src3 = (DATA_T2)(row1.s3, row1.s5); \
    J3 += bt * src3; \
    src4 = (DATA_T2)(row1.s4, row1.s6); \
    J4 += bt * src4; \
    src5 = (DATA_T2)(row1.s5, row1.s7); \
    J5 += bt * src5; \
    src6 = (DATA_T2)(row1.s6, row1.s8); \
    J6 += bt * src6; \
    src7 = (DATA_T2)(row1.s7, row1.s9); \
    J7 += bt * src7; \
    row0 = (DATA_T16)(0.0f); \
    row1 = (DATA_T16)(0.0f); \
    if (in_h_start + 6 >= 0 && in_h_start + 6 < src_h) { \
        row0.lo = vload8(0, src_data + in_base + src_w * 6); \
        row0.s89 = vload2(0, src_data + in_base + src_w * 6 + 8); \
    } \
    if (in_h_start + 7 >= 0 && in_h_start + 7 < src_h) { \
        row1.lo = vload8(0, src_data + in_base + src_w * 7); \
        row1.s89 = vload2(0, src_data + in_base + src_w * 7 + 8); \
    } \
    row0 = AS_DATA_T16(as_short16(row0) & inputMask); \
    row1 = AS_DATA_T16(as_short16(row1) & inputMask); \
    bt = bt_ar[6]; \
    src0 = (DATA_T2)(row0.s0, row0.s2); \
    I0 += bt * src0; \
    src1 = (DATA_T2)(row0.s1, row0.s3); \
    I1 += bt * src1; \
    src2 = (DATA_T2)(row0.s2, row0.s4); \
    I2 += bt * src2; \
    src3 = (DATA_T2)(row0.s3, row0.s5); \
    I3 += bt * src3; \
    src4 = (DATA_T2)(row0.s4, row0.s6); \
    I4 += bt * src4; \
    src5 = (DATA_T2)(row0.s5, row0.s7); \
    I5 += bt * src5; \
    src6 = (DATA_T2)(row0.s6, row0.s8); \
    I6 += bt * src6; \
    src7 = (DATA_T2)(row0.s7, row0.s9); \
    I7 += bt * src7; \
    bt = bt_ar[7]; \
    src0 = (DATA_T2)(row1.s0, row1.s2); \
    I0 += bt * src0; \
    src1 = (DATA_T2)(row1.s1, row1.s3); \
    I1 += bt * src1; \
    src2 = (DATA_T2)(row1.s2, row1.s4); \
    I2 += bt * src2; \
    src3 = (DATA_T2)(row1.s3, row1.s5); \
    I3 += bt * src3; \
    src4 = (DATA_T2)(row1.s4, row1.s6); \
    I4 += bt * src4; \
    src5 = (DATA_T2)(row1.s5, row1.s7); \
    I5 += bt * src5; \
    src6 = (DATA_T2)(row1.s6, row1.s8); \
    I6 += bt * src6; \
    src7 = (DATA_T2)(row1.s7, row1.s9); \
    I7 += bt * src7; \
    bt = bt_br[6]; \
    src0 = (DATA_T2)(row0.s0, row0.s2); \
    J0 += bt * src0; \
    src1 = (DATA_T2)(row0.s1, row0.s3); \
    J1 += bt * src1; \
    src2 = (DATA_T2)(row0.s2, row0.s4); \
    J2 += bt * src2; \
    src3 = (DATA_T2)(row0.s3, row0.s5); \
    J3 += bt * src3; \
    src4 = (DATA_T2)(row0.s4, row0.s6); \
    J4 += bt * src4; \
    src5 = (DATA_T2)(row0.s5, row0.s7); \
    J5 += bt * src5; \
    src6 = (DATA_T2)(row0.s6, row0.s8); \
    J6 += bt * src6; \
    src7 = (DATA_T2)(row0.s7, row0.s9); \
    J7 += bt * src7; \
    bt = bt_br[7]; \
    src0 = (DATA_T2)(row1.s0, row1.s2); \
    J0 += bt * src0; \
    src1 = (DATA_T2)(row1.s1, row1.s3); \
    J1 += bt * src1; \
    src2 = (DATA_T2)(row1.s2, row1.s4); \
    J2 += bt * src2; \
    src3 = (DATA_T2)(row1.s3, row1.s5); \
    J3 += bt * src3; \
    src4 = (DATA_T2)(row1.s4, row1.s6); \
    J4 += bt * src4; \
    src5 = (DATA_T2)(row1.s5, row1.s7); \
    J5 += bt * src5; \
    src6 = (DATA_T2)(row1.s6, row1.s8); \
    J6 += bt * src6; \
    src7 = (DATA_T2)(row1.s7, row1.s9); \
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
      dst_tile += inch_aligned * coalescing_feature_height_; \
      vstore2((DATA_T2)(r1), 0, dst_data + dst_tile); \
      dst_tile += inch_aligned * coalescing_feature_height_; \
    } \
    { \
      DATA_T2 r0 = CONVERT_TO_DATA_T2(-1.060660f * I1_f + 1.5f * I2_f + \
                                0.766032f * I3_f + -1.083333f * I4_f + \
                                -0.117851f * I5_f + 0.1666667f * I6_f); \
      DATA_T2 r1 = CONVERT_TO_DATA_T2(-0.212132f * I1_f + -0.15f * I2_f + \
                                0.471404f * I3_f + 0.333333f * I4_f + \
                                -0.094280f * I5_f + -0.066666f * I6_f); \
      vstore2((DATA_T2)(r0), 0, dst_data + dst_tile); \
      dst_tile += inch_aligned * coalescing_feature_height_; \
      vstore2((DATA_T2)(r1), 0, dst_data + dst_tile); \
      dst_tile += inch_aligned * coalescing_feature_height_; \
    } \
    { \
      DATA_T2 r0 = CONVERT_TO_DATA_T2(0.212132f * I1_f + -0.15f * I2_f + \
                                -0.471404f * I3_f + 0.333333f * I4_f + \
                                0.094280f * I5_f + -0.066666f * I6_f); \
      DATA_T2 r1 = CONVERT_TO_DATA_T2(0.023570f * I1_f + 0.011111f * I2_f + \
                                -0.058925f * I3_f + -0.027777f * I4_f + \
                                0.023570f * I5_f + 0.011111f * I6_f); \
      vstore2((DATA_T2)(r0), 0, dst_data + dst_tile); \
      dst_tile += inch_aligned * coalescing_feature_height_; \
      vstore2((DATA_T2)(r1), 0, dst_data + dst_tile); \
      dst_tile += inch_aligned * coalescing_feature_height_; \
    } \
    { \
      DATA_T2 r0 = CONVERT_TO_DATA_T2(-0.023570f * I1_f + 0.011111f * I2_f + \
                                0.058925f * I3_f + -0.027777f * I4_f + \
                                -0.023570f * I5_f + 0.011111f * I6_f); \
      DATA_T2 r1 = CONVERT_TO_DATA_T2(-4.5f * I1_f + 12.25f * I3_f + -7.0f * I5_f + \
                                1.0f * I7_f); \
      vstore2((DATA_T2)(r0), 0, dst_data + dst_tile); \
      dst_tile += inch_aligned * coalescing_feature_height_; \
      vstore2((DATA_T2)(r1), 0, dst_data + dst_tile); \
      dst_tile += inch_aligned * coalescing_feature_height_; \
    } \
      I0_f = convert_float2(J0); \
      I1_f = convert_float2(J1); \
      I2_f = convert_float2(J2); \
      I3_f = convert_float2(J3); \
      I4_f = convert_float2(J4); \
      I5_f = convert_float2(J5); \
      I6_f = convert_float2(J6); \
      I7_f = convert_float2(J7); \
      { \
        DATA_T2 r0 = CONVERT_TO_DATA_T2(I0_f + -2.722228f * I2_f + 1.555556f * I4_f + \
                                -0.222222f * I6_f); \
        DATA_T2 r1 = CONVERT_TO_DATA_T2(1.060660f * I1_f + 1.5f * I2_f + \
                                -0.766032f * I3_f + -1.083333f * I4_f + \
                                0.117851f * I5_f + 0.1666667f * I6_f); \
        vstore2((DATA_T2)(r0), 0, dst_data + dst_tile); \
        dst_tile += inch_aligned * coalescing_feature_height_; \
        vstore2((DATA_T2)(r1), 0, dst_data + dst_tile); \
        dst_tile += inch_aligned * coalescing_feature_height_; \
      } \
      { \
        DATA_T2 r0 = CONVERT_TO_DATA_T2(-1.060660f * I1_f + 1.5f * I2_f + \
                                0.766032f * I3_f + -1.083333f * I4_f + \
                                -0.117851f * I5_f + 0.1666667f * I6_f); \
        DATA_T2 r1 = CONVERT_TO_DATA_T2(-0.212132f * I1_f + -0.15f * I2_f + \
                                0.471404f * I3_f + 0.333333f * I4_f + \
                                -0.094280f * I5_f + -0.066666f * I6_f); \
        vstore2((DATA_T2)(r0), 0, dst_data + dst_tile); \
        dst_tile += inch_aligned * coalescing_feature_height_; \
        vstore2((DATA_T2)(r1), 0, dst_data + dst_tile); \
        dst_tile += inch_aligned * coalescing_feature_height_; \
      } \
      { \
        DATA_T2 r0 = CONVERT_TO_DATA_T2(0.212132f * I1_f + -0.15f * I2_f + \
                                -0.471404f * I3_f + 0.333333f * I4_f + \
                                0.094280f * I5_f + -0.066666f * I6_f); \
        DATA_T2 r1 = CONVERT_TO_DATA_T2(0.023570f * I1_f + 0.011111f * I2_f + \
                                -0.058925f * I3_f + -0.027777f * I4_f + \
                                0.023570f * I5_f + 0.011111f * I6_f); \
        vstore2((DATA_T2)(r0), 0, dst_data + dst_tile); \
        dst_tile += inch_aligned * coalescing_feature_height_; \
        vstore2((DATA_T2)(r1), 0, dst_data + dst_tile); \
        dst_tile += inch_aligned * coalescing_feature_height_; \
      } \
      { \
        DATA_T2 r0 = CONVERT_TO_DATA_T2(-0.023570f * I1_f + 0.011111f * I2_f + \
                                0.058925f * I3_f + -0.027777f * I4_f + \
                                -0.023570f * I5_f + 0.011111f * I6_f); \
        DATA_T2 r1 = CONVERT_TO_DATA_T2(-4.5f * I1_f + 12.25f * I3_f + -7.0f * I5_f + \
                                1.0f * I7_f); \
        vstore2((DATA_T2)(r0), 0, dst_data + dst_tile); \
        dst_tile += inch_aligned * coalescing_feature_height_; \
        vstore2((DATA_T2)(r1), 0, dst_data + dst_tile); \
      }

#define WINO7X7DOT_MULTIPLY_NCHW_OPT(input_tm, weight_tm, dot_result, dst_n, dst_c, dst_h, dst_w, input_c, \
                                     coalescing_feature_height_, aligned_outch, aligned_inch) \
    int tile_index = get_global_id(0) * 4; \
    int p_index = get_global_id(1); \
    int c_index = get_global_id(2) % dst_c; \
    int n_index = get_global_id(2) / dst_c; \
    if (tile_index >= dst_w || c_index >= dst_c || n_index >= dst_n) { \
      return; \
    } \
    DATA_T16 r0 = 0.0f; \
    DATA_T16 r1 = 0.0f; \
    int input_c_aligned = (input_c + aligned_inch - 1) / aligned_inch * aligned_inch; \
    int input_base = n_index * input_c_aligned * dst_h * dst_w; \
    input_base += tile_index / coalescing_feature_height_ * \
                  (input_c_aligned * coalescing_feature_height_ * 64); \
    input_base += tile_index % coalescing_feature_height_ / 2 * 2 * aligned_inch; \
    input_base += tile_index % coalescing_feature_height_ % 2; \
    input_base += p_index * coalescing_feature_height_ * input_c_aligned; \
    int weight_base = c_index * aligned_outch * 64 * input_c_aligned; \
    weight_base += p_index * aligned_outch * input_c_aligned; \
    int align_pixel = 4; \
    int dot_base = n_index * dst_c * aligned_outch * dst_h * dst_w; \
    dot_base += c_index * aligned_outch * 64 * dst_w; \
    dot_base += tile_index % coalescing_feature_height_ / 2 * 2 * align_pixel; \
    dot_base += tile_index % coalescing_feature_height_ % 2; \
    dot_base += tile_index / coalescing_feature_height_ * aligned_outch * coalescing_feature_height_ * 64; \
    dot_base += p_index / align_pixel * align_pixel * coalescing_feature_height_; \
    dot_base += p_index % align_pixel * 2; \
    int2 stride = (int2)(coalescing_feature_height_ * aligned_inch, aligned_outch * aligned_inch); \
    for (int i = 0; i < input_c_aligned; i += aligned_inch) { \
      DATA_T16 vin = vload16(0, input_tm + input_base); \
      DATA_T16 vwl = vload16(0, weight_tm + weight_base); \
      r0.lo = mad(vin.s0, vwl.lo, r0.lo); \
      r0.hi = mad(vin.s1, vwl.lo, r0.hi); \
      r1.lo = mad(vin.s8, vwl.lo, r1.lo); \
      r1.hi = mad(vin.s9, vwl.lo, r1.hi); \
      DATA_T16 vwh = vload16(0, weight_tm + weight_base + 16); \
      r0.lo = mad(vin.s2, vwl.hi, r0.lo); \
      r0.hi = mad(vin.s3, vwl.hi, r0.hi); \
      r1.lo = mad(vin.sA, vwl.hi, r1.lo); \
      r1.hi = mad(vin.sB, vwl.hi, r1.hi); \
      r0.lo = mad(vin.s4, vwh.lo, r0.lo); \
      r0.hi = mad(vin.s5, vwh.lo, r0.hi); \
      r1.lo = mad(vin.sC, vwh.lo, r1.lo); \
      r1.hi = mad(vin.sD, vwh.lo, r1.hi); \
      r0.lo = mad(vin.s6, vwh.hi, r0.lo); \
      r0.hi = mad(vin.s7, vwh.hi, r0.hi); \
      r1.lo = mad(vin.sE, vwh.hi, r1.lo); \
      r1.hi = mad(vin.sF, vwh.hi, r1.hi); \
      input_base += stride.s0; \
      weight_base += stride.s1; \
    } \
    vstore2(r0.s08, 0, dot_result + dot_base); \
    vstore2(r1.s08, 0, dot_result + dot_base + 8); \
    dot_base += 64 * coalescing_feature_height_; \
    vstore2(r0.s19, 0, dot_result + dot_base); \
    vstore2(r1.s19, 0, dot_result + dot_base + 8); \
    dot_base += 64 * coalescing_feature_height_; \
    vstore2(r0.s2A, 0, dot_result + dot_base); \
    vstore2(r1.s2A, 0, dot_result + dot_base + 8); \
    dot_base += 64 * coalescing_feature_height_; \
    vstore2(r0.s3B, 0, dot_result + dot_base); \
    vstore2(r1.s3B, 0, dot_result + dot_base + 8); \
    dot_base += 64 * coalescing_feature_height_; \
    vstore2(r0.s4C, 0, dot_result + dot_base); \
    vstore2(r1.s4C, 0, dot_result + dot_base + 8); \
    dot_base += 64 * coalescing_feature_height_; \
    vstore2(r0.s5D, 0, dot_result + dot_base); \
    vstore2(r1.s5D, 0, dot_result + dot_base + 8); \
    dot_base += 64 * coalescing_feature_height_; \
    vstore2(r0.s6E, 0, dot_result + dot_base); \
    vstore2(r1.s6E, 0, dot_result + dot_base + 8); \
    dot_base += 64 * coalescing_feature_height_; \
    vstore2(r0.s7F, 0, dot_result + dot_base); \
    vstore2(r1.s7F, 0, dot_result + dot_base + 8);

#define WINO7X7OUTPUT_TM_NCHW_OPT(dot_result, bias, output, src_n, src_c, src_h, src_w, output_n, output_c, output_h, \
                                  output_w, coalescing_feature_height_, tile_x, aligned_outch, aligned_inch) \
    int n_index = get_global_id(2) / src_c; \
    int c_index = get_global_id(2) % src_c; \
    int p_index = get_global_id(1); \
    int h_index = get_global_id(0) * 2 / tile_x; \
    int w_index = get_global_id(0) * 2 % tile_x; \
    int t_index = get_global_id(0) * 2; \
    int tile_totals = ((output_h + 1) / 2) * tile_x; \
    if (c_index >= output_c || t_index >= tile_totals || \
        h_index * 2 + p_index >= output_h || w_index * 2 >= output_w) { \
      return; \
    } \
    int src_base = n_index * src_c * src_h * src_w; \
    src_base += c_index / aligned_outch * src_h * aligned_outch * src_w; \
    src_base += c_index % aligned_outch * 64 * coalescing_feature_height_; \
    src_base += t_index % coalescing_feature_height_ / 2 * 2 * 4; \
    src_base += t_index % coalescing_feature_height_ % 2; \
    src_base += t_index / coalescing_feature_height_ * \
                (8 * coalescing_feature_height_ * 64); \
    DATA_T at_ar[8]; \
    DATA_T8 t00 = vload8(0, At7x7 + 8 * p_index); \
    at_ar[0] = t00.s0; \
    at_ar[1] = t00.s1; \
    at_ar[2] = t00.s2; \
    at_ar[3] = t00.s3; \
    at_ar[4] = t00.s4; \
    at_ar[5] = t00.s5; \
    at_ar[6] = t00.s6; \
    at_ar[7] = t00.s7; \
    int dst_base = n_index * output_c * output_h * output_w; \
    dst_base += c_index * output_h * output_w; \
    dst_base += (h_index * 2 + p_index) * output_w + w_index * 2; \
    DATA_T16 ping; \
    ping.lo = vload8(0, dot_result + src_base); src_base += 4 * coalescing_feature_height_; \
    ping.hi = vload8(0, dot_result + src_base); src_base += 4 * coalescing_feature_height_; \
    DATA_T2 tmp0 = at_ar[0] * ping.s01; \
    DATA_T2 tmp1 = at_ar[0] * ping.s23; \
    DATA_T2 tmp2 = at_ar[0] * ping.s45; \
    DATA_T2 tmp3 = at_ar[0] * ping.s67; \
    DATA_T2 tmp4 = at_ar[0] * ping.s89; \
    DATA_T2 tmp5 = at_ar[0] * ping.sAB; \
    DATA_T2 tmp6 = at_ar[0] * ping.sCD; \
    DATA_T2 tmp7 = at_ar[0] * ping.sEF; \
    ping.lo = vload8(0, dot_result + src_base); src_base += 4 * coalescing_feature_height_; \
    ping.hi = vload8(0, dot_result + src_base); src_base += 4 * coalescing_feature_height_; \
    tmp0 += at_ar[1] * ping.s01; \
    tmp1 += at_ar[1] * ping.s23; \
    tmp2 += at_ar[1] * ping.s45; \
    tmp3 += at_ar[1] * ping.s67; \
    tmp4 += at_ar[1] * ping.s89; \
    tmp5 += at_ar[1] * ping.sAB; \
    tmp6 += at_ar[1] * ping.sCD; \
    tmp7 += at_ar[1] * ping.sEF; \
    ping.lo = vload8(0, dot_result + src_base); src_base += 4 * coalescing_feature_height_; \
    ping.hi = vload8(0, dot_result + src_base); src_base += 4 * coalescing_feature_height_; \
    tmp0 += at_ar[2] * ping.s01; \
    tmp1 += at_ar[2] * ping.s23; \
    tmp2 += at_ar[2] * ping.s45; \
    tmp3 += at_ar[2] * ping.s67; \
    tmp4 += at_ar[2] * ping.s89; \
    tmp5 += at_ar[2] * ping.sAB; \
    tmp6 += at_ar[2] * ping.sCD; \
    tmp7 += at_ar[2] * ping.sEF; \
    ping.lo = vload8(0, dot_result + src_base); src_base += 4 * coalescing_feature_height_; \
    ping.hi = vload8(0, dot_result + src_base); src_base += 4 * coalescing_feature_height_; \
    tmp0 += at_ar[3] * ping.s01; \
    tmp1 += at_ar[3] * ping.s23; \
    tmp2 += at_ar[3] * ping.s45; \
    tmp3 += at_ar[3] * ping.s67; \
    tmp4 += at_ar[3] * ping.s89; \
    tmp5 += at_ar[3] * ping.sAB; \
    tmp6 += at_ar[3] * ping.sCD; \
    tmp7 += at_ar[3] * ping.sEF; \
    ping.lo = vload8(0, dot_result + src_base); src_base += 4 * coalescing_feature_height_; \
    ping.hi = vload8(0, dot_result + src_base); src_base += 4 * coalescing_feature_height_; \
    tmp0 += at_ar[4] * ping.s01; \
    tmp1 += at_ar[4] * ping.s23; \
    tmp2 += at_ar[4] * ping.s45; \
    tmp3 += at_ar[4] * ping.s67; \
    tmp4 += at_ar[4] * ping.s89; \
    tmp5 += at_ar[4] * ping.sAB; \
    tmp6 += at_ar[4] * ping.sCD; \
    tmp7 += at_ar[4] * ping.sEF; \
    ping.lo = vload8(0, dot_result + src_base); src_base += 4 * coalescing_feature_height_; \
    ping.hi = vload8(0, dot_result + src_base); src_base += 4 * coalescing_feature_height_; \
    tmp0 += at_ar[5] * ping.s01; \
    tmp1 += at_ar[5] * ping.s23; \
    tmp2 += at_ar[5] * ping.s45; \
    tmp3 += at_ar[5] * ping.s67; \
    tmp4 += at_ar[5] * ping.s89; \
    tmp5 += at_ar[5] * ping.sAB; \
    tmp6 += at_ar[5] * ping.sCD; \
    tmp7 += at_ar[5] * ping.sEF; \
    ping.lo = vload8(0, dot_result + src_base); src_base += 4 * coalescing_feature_height_; \
    ping.hi = vload8(0, dot_result + src_base); src_base += 4 * coalescing_feature_height_; \
    tmp0 += at_ar[6] * ping.s01; \
    tmp1 += at_ar[6] * ping.s23; \
    tmp2 += at_ar[6] * ping.s45; \
    tmp3 += at_ar[6] * ping.s67; \
    tmp4 += at_ar[6] * ping.s89; \
    tmp5 += at_ar[6] * ping.sAB; \
    tmp6 += at_ar[6] * ping.sCD; \
    tmp7 += at_ar[6] * ping.sEF; \
    ping.lo = vload8(0, dot_result + src_base); src_base += 4 * coalescing_feature_height_; \
    ping.hi = vload8(0, dot_result + src_base); src_base += 4 * coalescing_feature_height_; \
    tmp0 += at_ar[7] * ping.s01; \
    tmp1 += at_ar[7] * ping.s23; \
    tmp2 += at_ar[7] * ping.s45; \
    tmp3 += at_ar[7] * ping.s67; \
    tmp4 += at_ar[7] * ping.s89; \
    tmp5 += at_ar[7] * ping.sAB; \
    tmp6 += at_ar[7] * ping.sCD; \
    tmp7 += at_ar[7] * ping.sEF; \
    int b_dst_base = dst_base; \
    DATA_T2 t0 = tmp1 + tmp2; \
    DATA_T2 t1 = tmp3 + tmp4; \
    DATA_T2 t2 = tmp5 + tmp6; \
    DATA_T2 bias_val = (DATA_T2)(bias[c_index]); \
    if ((w_index * 2 + 0) < output_w) { \
      DATA_T2 f0 = CONVERT_TO_DATA_T2(tmp0 + t0 + t1 + t2) + bias_val; \
      f0 = ACT_VEC_F(DATA_T2, f0); \
      output[dst_base] = f0.s0; \
      if (w_index * 2 + 2 < output_w) \
        output[dst_base + 2] = f0.s1; \
    } \
    DATA_T2 t3 = tmp1 - tmp2; \
    DATA_T2 t4 = tmp3 - tmp4; \
    DATA_T2 t5 = tmp5 - tmp6; \
    if ((w_index * 2 + 1) < output_w) { \
      DATA_T2 f0 = CONVERT_TO_DATA_T2(t3 * At7x7[9] + t4 * At7x7[11] + \
                                t5 * At7x7[13] + tmp7) + bias_val; \
      f0 = ACT_VEC_F(DATA_T2, f0); \
      dst_base += 1; \
      output[dst_base] = f0.s0; \
      if (w_index * 2 + 3 < output_w) \
        output[dst_base + 2] = f0.s1; \
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

ADD_SINGLE_KERNEL(wino7x7weight_tm_nchw_opt_FP16, (
                    __global DATA_T *wt, __global DATA_T *wt_tm,
                    int outch, int inch, int aligned_outch, int aligned_inch) {
    WINO7X7WEIGHT_TM_NCHW_OPT(wt, wt_tm, outch, inch, aligned_outch, aligned_inch)
})

ADD_KERNEL_HEADER(wino7x7input_tm_nchw_opt_FP16, {DEFINE_BT8X8})
ADD_SINGLE_KERNEL(wino7x7input_tm_nchw_opt_FP16, (
        __global DATA_T *src_data, __global DATA_T *dst_data, int src_n, int src_c,
        int src_h, int src_w, int dst_n, int dst_c, int dst_h, int dst_w,
        int padL, int padT, int tiles_total, int aligned_tiles, int coalescing_feature_height_,
        int tiles_x, int aligned_inch) {
    WINO7X7INPUT_TM_NCHW_OPT(src_data, dst_data, src_n, src_c, src_h, src_w, dst_n, dst_c, dst_h, dst_w, padL, padT, \
                             tiles_total, aligned_tiles, coalescing_feature_height_, tiles_x, aligned_inch)
})

ADD_SINGLE_KERNEL(wino7x7dot_multiply_nchw_opt_FP16, (
        __global DATA_T *input_tm, __global DATA_T *weight_tm,
        __global DATA_T *dot_result, int dst_n, int dst_c, int dst_h, int dst_w,
        int input_c, int coalescing_feature_height_,
        int aligned_outch, int aligned_inch) {
    WINO7X7DOT_MULTIPLY_NCHW_OPT(input_tm, weight_tm, dot_result, dst_n, dst_c, dst_h, dst_w, input_c, \
                                 coalescing_feature_height_, aligned_outch, aligned_inch)
})


// kernels without activation func
#define ACT_VEC_F(VEC_T, x) x
ADD_KERNEL_HEADER(wino7x7output_tm_nchw_opt_FP16, {DEFINE_AT7X7})
ADD_SINGLE_KERNEL(wino7x7output_tm_nchw_opt_FP16, (
        __global DATA_T *dot_result, __global DATA_T *bias, __global DATA_T *output,
        int src_n, int src_c, int src_h, int src_w, int output_n, int output_c,
        int output_h, int output_w, int coalescing_feature_height_,
        int tile_x, int aligned_outch, int aligned_inch) {
    WINO7X7OUTPUT_TM_NCHW_OPT(dot_result, bias, output, src_n, src_c, src_h, src_w, output_n, output_c, output_h, \
                              output_w, coalescing_feature_height_, tile_x, aligned_outch, aligned_inch)
})

#undef ACT_VEC_F  // NONE

// apply RELU to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU_VEC_F(VEC_T, x)
ADD_KERNEL_HEADER(RELUwino7x7output_tm_nchw_opt_FP16, {DEFINE_AT7X7})
ADD_SINGLE_KERNEL(RELUwino7x7output_tm_nchw_opt_FP16, (
        __global DATA_T *dot_result, __global DATA_T *bias, __global DATA_T *output,
        int src_n, int src_c, int src_h, int src_w, int output_n, int output_c,
        int output_h, int output_w, int coalescing_feature_height_,
        int tile_x, int aligned_outch, int aligned_inch) {
    WINO7X7OUTPUT_TM_NCHW_OPT(dot_result, bias, output, src_n, src_c, src_h, src_w, output_n, output_c, output_h, \
                              output_w, coalescing_feature_height_, tile_x, aligned_outch, aligned_inch)
})

#undef ACT_VEC_F  // RELU

// apply RELU6 to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU6_VEC_F(VEC_T, x)
ADD_KERNEL_HEADER(RELU6wino7x7output_tm_nchw_opt_FP16, {DEFINE_AT7X7})
ADD_SINGLE_KERNEL(RELU6wino7x7output_tm_nchw_opt_FP16, (
        __global DATA_T *dot_result, __global DATA_T *bias, __global DATA_T *output,
        int src_n, int src_c, int src_h, int src_w, int output_n, int output_c,
        int output_h, int output_w, int coalescing_feature_height_,
        int tile_x, int aligned_outch, int aligned_inch) {
    WINO7X7OUTPUT_TM_NCHW_OPT(dot_result, bias, output, src_n, src_c, src_h, src_w, output_n, output_c, output_h, \
                              output_w, coalescing_feature_height_, tile_x, aligned_outch, aligned_inch)
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
