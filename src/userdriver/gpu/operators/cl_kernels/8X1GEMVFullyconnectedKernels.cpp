#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-escape-sequence"

#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {
#define FC_8X1_GEMV_FEED_OUTPUT(globalID0, globalID1, topChannel, batch, output8_0, output) \
        int outIndex = globalID0 * topChannel + globalID1; \
        if (globalID0 + 7 < batch) { \
            output[outIndex] = output8_0.s0; \
            output[outIndex + topChannel] = output8_0.s1; \
            output[outIndex + topChannel * 2] = output8_0.s2; \
            output[outIndex + topChannel * 3] = output8_0.s3; \
            output[outIndex + topChannel * 4] = output8_0.s4; \
            output[outIndex + topChannel * 5] = output8_0.s5; \
            output[outIndex + topChannel * 6] = output8_0.s6; \
            output[outIndex + topChannel * 7] = output8_0.s7; \
        } else if (globalID0 + 6 < batch) { \
            output[outIndex] = output8_0.s0; \
            output[outIndex + topChannel] = output8_0.s1; \
            output[outIndex + topChannel * 2] = output8_0.s2; \
            output[outIndex + topChannel * 3] = output8_0.s3; \
            output[outIndex + topChannel * 4] = output8_0.s4; \
            output[outIndex + topChannel * 5] = output8_0.s5; \
            output[outIndex + topChannel * 6] = output8_0.s6; \
        } else if (globalID0 + 5 < batch) { \
            output[outIndex] = output8_0.s0; \
            output[outIndex + topChannel] = output8_0.s1; \
            output[outIndex + topChannel * 2] = output8_0.s2; \
            output[outIndex + topChannel * 3] = output8_0.s3; \
            output[outIndex + topChannel * 4] = output8_0.s4; \
            output[outIndex + topChannel * 5] = output8_0.s5; \
        } else if (globalID0 + 4 < batch) { \
            output[outIndex] = output8_0.s0; \
            output[outIndex + topChannel] = output8_0.s1; \
            output[outIndex + topChannel * 2] = output8_0.s2; \
            output[outIndex + topChannel * 3] = output8_0.s3; \
            output[outIndex + topChannel * 4] = output8_0.s4; \
        } else if (globalID0 + 3 < batch) { \
            output[outIndex] = output8_0.s0; \
            output[outIndex + topChannel] = output8_0.s1; \
            output[outIndex + topChannel * 2] = output8_0.s2; \
            output[outIndex + topChannel * 3] = output8_0.s3; \
        } else if (globalID0 + 2 < batch) { \
            output[outIndex] = output8_0.s0; \
            output[outIndex + topChannel] = output8_0.s1; \
            output[outIndex + topChannel * 2] = output8_0.s2; \
        } else if (globalID0 + 1 < batch) { \
            output[outIndex] = output8_0.s0; \
            output[outIndex + topChannel] = output8_0.s1; \
        } else if (globalID0 < batch) { \
            output[outIndex] = output8_0.s0; \
        }

#define DATA_T float
#define DATA_T8 float8

ADD_SINGLE_KERNEL(fc_interleave_8x1_gemv_FP32, (__global const float *input,
                                                          __global float *output,
                                                          unsigned int oriRow,
                                                          unsigned int oriCol,
                                                          unsigned int dstCol) {
    FC_INTERLEAVE_8X1(input, output, oriRow, oriCol, dstCol)
})

ADD_SINGLE_KERNEL(fc_8x1_gemv_FP32, (__global const float *input,
                                 __global const float *weight,
                                 __global const float *bias,
                                 __global float *output,
                                 unsigned int picStep,
                                 unsigned int batch,
                                 unsigned int topChannel) {
    int globalID0 = get_global_id(0) * 8;
    int globalID1 = get_global_id(1);
    if (globalID1 < topChannel && globalID0 < batch) {
        float8 input8;
        float8 weight8;
        float8 output8_0 = (float8)(bias[get_global_id(1)]);
        /* input, weight start index */
        uint2 src_addr = (uint2)(globalID0 * picStep, globalID1 * picStep);
        uint end_alignWidth = src_addr.s1 + picStep;
        /* splitStep should be 8 multiple */
        for (; src_addr.s1 < end_alignWidth; src_addr += (uint2)(64, 8)) {
            weight8 = vload8(0, weight + src_addr.s1);
            input8 = vload8(0, input + src_addr.s0);
            output8_0 += weight8.s0 * input8;
            input8 = vload8(0, input + src_addr.s0 + 8);
            output8_0 += weight8.s1 * input8;
            input8 = vload8(0, input + src_addr.s0 + 8 * 2);
            output8_0 += weight8.s2 * input8;
            input8 = vload8(0, input + src_addr.s0 + 8 * 3);
            output8_0 += weight8.s3 * input8;
            input8 = vload8(0, input + src_addr.s0 + 8 * 4);
            output8_0 += weight8.s4 * input8;
            input8 = vload8(0, input + src_addr.s0 + 8 * 5);
            output8_0 += weight8.s5 * input8;
            input8 = vload8(0, input + src_addr.s0 + 8 * 6);
            output8_0 += weight8.s6 * input8;
            input8 = vload8(0, input + src_addr.s0 + 8 * 7);
            output8_0 += weight8.s7 * input8;
        }
        FC_8X1_GEMV_FEED_OUTPUT(globalID0, globalID1, topChannel, batch, output8_0, output)
    }
})
#undef DATA_T
#undef DATA_T8

#define DATA_T half
#define DATA_T8 half8

ADD_SINGLE_KERNEL(fc_interleave_8x1_gemv_FP16, (__global const half *input,
                                                          __global half *output,
                                                          unsigned int oriRow,
                                                          unsigned int oriCol,
                                                          unsigned int dstCol) {
    FC_INTERLEAVE_8X1(input, output, oriRow, oriCol, dstCol)
})

ADD_SINGLE_KERNEL(fc_8x1_gemv_FP16, (__global const half *input,
                                 __global const half *weight,
                                 __global const half *bias,
                                 __global half *output,
                                 unsigned int picStep,
                                 unsigned int batch,
                                 unsigned int topChannel) {
    int globalID0 = get_global_id(0) * 8;
    int globalID1 = get_global_id(1);
    if (globalID1 < topChannel && globalID0 < batch) {
        half8 input8;
        half8 weight8;
        float8 output8_0 = (float8)(bias[get_global_id(1)]);
        /* input, weight start index */
        uint2 src_addr = (uint2)(globalID0 * picStep, globalID1 * picStep);
        uint end_alignWidth = src_addr.s1 + picStep;
        /* splitStep should be 8 multiple */
        for (; src_addr.s1 < end_alignWidth; src_addr += (uint2)(64, 8)) {
            half8 tem8 = (half8)(0.0f);
            float8 temp8;
            weight8 = vload8(0, weight + src_addr.s1);
            input8 = vload8(0, input + src_addr.s0);
            tem8 += weight8.s0 * input8;
            input8 = vload8(0, input + src_addr.s0 + 8);
            tem8 += weight8.s1 * input8;
            input8 = vload8(0, input + src_addr.s0 + 8 * 2);
            tem8 += weight8.s2 * input8;
            input8 = vload8(0, input + src_addr.s0 + 8 * 3);
            tem8 += weight8.s3 * input8;
            input8 = vload8(0, input + src_addr.s0 + 8 * 4);
            tem8 += weight8.s4 * input8;
            input8 = vload8(0, input + src_addr.s0 + 8 * 5);
            tem8 += weight8.s5 * input8;
            input8 = vload8(0, input + src_addr.s0 + 8 * 6);
            tem8 += weight8.s6 * input8;
            input8 = vload8(0, input + src_addr.s0 + 8 * 7);
            tem8 += weight8.s7 * input8;
            output8_0 += convert_float8(tem8);
        }
        FC_8X1_GEMV_FEED_OUTPUT(globalID0, globalID1, topChannel, batch, output8_0, output)
    }
}

)
#undef DATA_T
#undef DATA_T8

}  // namespace gpu
}  // namespace ud
}  // namespace enn
#pragma clang diagnostic pop
