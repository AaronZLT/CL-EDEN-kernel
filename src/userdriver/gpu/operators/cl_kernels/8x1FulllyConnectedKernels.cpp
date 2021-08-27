#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define FC_SPLITOPT_8X1(input, weight, output, picStep, splitStep, batch, topChannel, splitNumber) \
    int globalID0 = get_global_id(0) * 8; \
    int globalID1 = get_global_id(1);  \
    int globalID2 = get_global_id(2);  \
    if (globalID1 < splitNumber && globalID2 < topChannel) { \
        DATA_T8 input8; \
        DATA_T8 weight8; \
        DATA_T8 output8_0 = (DATA_T8)(0.0f); \
        /* input, weight start index */ \
        uint2 src_addr = (uint2)(globalID0 * picStep + globalID1 * splitStep * 8, \
                                    globalID2 * picStep + globalID1 * splitStep); \
        uint end_alignWidth = src_addr.s1 + splitStep; \
        /* splitStep should be 8 multiple */ \
        for (; src_addr.s1 < end_alignWidth; src_addr += (uint2)(64, 8)) { \
            weight8 = vload8(0, weight + src_addr.s1); \
            input8 = vload8(0, input + src_addr.s0); \
            output8_0 += weight8.s0 * input8; \
            input8 = vload8(0, input + src_addr.s0 + 8); \
            output8_0 += weight8.s1 * input8; \
            input8 = vload8(0, input + src_addr.s0 + 8 * 2); \
            output8_0 += weight8.s2 * input8; \
            input8 = vload8(0, input + src_addr.s0 + 8 * 3); \
            output8_0 += weight8.s3 * input8; \
            input8 = vload8(0, input + src_addr.s0 + 8 * 4); \
            output8_0 += weight8.s4 * input8; \
            input8 = vload8(0, input + src_addr.s0 + 8 * 5); \
            output8_0 += weight8.s5 * input8; \
            input8 = vload8(0, input + src_addr.s0 + 8 * 6); \
            output8_0 += weight8.s6 * input8; \
            input8 = vload8(0, input + src_addr.s0 + 8 * 7); \
            output8_0 += weight8.s7 * input8; \
            /*  loop unrolling */ \
            src_addr += (uint2)(64, 8); \
            weight8 = vload8(0, weight + src_addr.s1); \
            input8 = vload8(0, input + src_addr.s0); \
            output8_0 += weight8.s0 * input8; \
            input8 = vload8(0, input + src_addr.s0 + 8); \
            output8_0 += weight8.s1 * input8; \
            input8 = vload8(0, input + src_addr.s0 + 8 * 2); \
            output8_0 += weight8.s2 * input8; \
            input8 = vload8(0, input + src_addr.s0 + 8 * 3); \
            output8_0 += weight8.s3 * input8; \
            input8 = vload8(0, input + src_addr.s0 + 8 * 4); \
            output8_0 += weight8.s4 * input8; \
            input8 = vload8(0, input + src_addr.s0 + 8 * 5); \
            output8_0 += weight8.s5 * input8; \
            input8 = vload8(0, input + src_addr.s0 + 8 * 6); \
            output8_0 += weight8.s6 * input8; \
            input8 = vload8(0, input + src_addr.s0 + 8 * 7); \
            output8_0 += weight8.s7 * input8; \
        } \
        if (globalID0 + 7 < batch) { \
            int outIndex = \
                globalID0 * splitNumber * topChannel + globalID1 * topChannel + globalID2; \
            output[outIndex] = output8_0.s0; \
            output[outIndex + topChannel * splitNumber] = output8_0.s1; \
            output[outIndex + topChannel * splitNumber * 2] = output8_0.s2; \
            output[outIndex + topChannel * splitNumber * 3] = output8_0.s3; \
            output[outIndex + topChannel * splitNumber * 4] = output8_0.s4; \
            output[outIndex + topChannel * splitNumber * 5] = output8_0.s5; \
            output[outIndex + topChannel * splitNumber * 6] = output8_0.s6; \
            output[outIndex + topChannel * splitNumber * 7] = output8_0.s7; \
        } else if (globalID0 + 6 < batch) { \
            int outIndex = \
                globalID0 * splitNumber * topChannel + globalID1 * topChannel + globalID2; \
            output[outIndex] = output8_0.s0; \
            output[outIndex + topChannel * splitNumber] = output8_0.s1; \
            output[outIndex + topChannel * splitNumber * 2] = output8_0.s2; \
            output[outIndex + topChannel * splitNumber * 3] = output8_0.s3; \
            output[outIndex + topChannel * splitNumber * 4] = output8_0.s4; \
            output[outIndex + topChannel * splitNumber * 5] = output8_0.s5; \
            output[outIndex + topChannel * splitNumber * 6] = output8_0.s6; \
        } else if (globalID0 + 5 < batch) { \
            int outIndex = \
                globalID0 * splitNumber * topChannel + globalID1 * topChannel + globalID2; \
            output[outIndex] = output8_0.s0; \
            output[outIndex + topChannel * splitNumber] = output8_0.s1; \
            output[outIndex + topChannel * splitNumber * 2] = output8_0.s2; \
            output[outIndex + topChannel * splitNumber * 3] = output8_0.s3; \
            output[outIndex + topChannel * splitNumber * 4] = output8_0.s4; \
            output[outIndex + topChannel * splitNumber * 5] = output8_0.s5; \
        } else if (globalID0 + 4 < batch) { \
            int outIndex = \
                globalID0 * splitNumber * topChannel + globalID1 * topChannel + globalID2; \
            output[outIndex] = output8_0.s0; \
            output[outIndex + topChannel * splitNumber] = output8_0.s1; \
            output[outIndex + topChannel * splitNumber * 2] = output8_0.s2; \
            output[outIndex + topChannel * splitNumber * 3] = output8_0.s3; \
            output[outIndex + topChannel * splitNumber * 4] = output8_0.s4; \
        } else if (globalID0 + 3 < batch) { \
            int outIndex = \
                globalID0 * splitNumber * topChannel + globalID1 * topChannel + globalID2; \
            output[outIndex] = output8_0.s0; \
            output[outIndex + topChannel * splitNumber] = output8_0.s1; \
            output[outIndex + topChannel * splitNumber * 2] = output8_0.s2; \
            output[outIndex + topChannel * splitNumber * 3] = output8_0.s3; \
        } else if (globalID0 + 2 < batch) { \
            int outIndex = \
                globalID0 * splitNumber * topChannel + globalID1 * topChannel + globalID2; \
            output[outIndex] = output8_0.s0; \
            output[outIndex + topChannel * splitNumber] = output8_0.s1; \
            output[outIndex + topChannel * splitNumber * 2] = output8_0.s2; \
        } else if (globalID0 + 1 < batch) { \
            int outIndex = \
                globalID0 * splitNumber * topChannel + globalID1 * topChannel + globalID2; \
            output[outIndex] = output8_0.s0; \
            output[outIndex + topChannel * splitNumber] = output8_0.s1; \
        } else if (globalID0 < batch) { \
            int outIndex = \
                globalID0 * splitNumber * topChannel + globalID1 * topChannel + globalID2; \
            output[outIndex] = output8_0.s0; \
        } \
    }

#define FC_MERGEOPT_8X1(input, bias, output, picStep, splitNum, batch, topChannel) \
        int globalID0 = get_global_id(0);  \
        int globalID1 = get_global_id(1);  \
        if (globalID0 < batch && globalID1 < topChannel) { \
            DATA_T temp = 0.0f; \
            for (int i = 0; i < splitNum; i++) { \
                temp += input[globalID0 * picStep + i * topChannel + globalID1]; \
            } \
            output[globalID0 * topChannel + globalID1] = temp + bias[globalID1]; \
        }

#define DATA_T float
#define DATA_T8 float8

ADD_SINGLE_KERNEL(fc_interleave_8x1_FP32, (__global const float *input,
                                         __global float *output,
                                         unsigned int oriRow,
                                         unsigned int oriCol,
                                         unsigned int dstCol) {
        FC_INTERLEAVE_8X1(input, output, oriRow, oriCol, dstCol)
})

ADD_SINGLE_KERNEL(fc_splitopt_8x1_FP32, (__global const float *input,
                                       __global const float *weight,
                                       __global float *output,
                                       unsigned int picStep,
                                       unsigned int splitStep,
                                       unsigned int batch,
                                       unsigned int topChannel,
                                       unsigned int splitNumber) {
        FC_SPLITOPT_8X1(input, weight, output, picStep, splitStep, batch, topChannel, splitNumber)
})

ADD_SINGLE_KERNEL(fc_mergeopt_8x1_FP32, (__global const float *input,
                                       __global const float *bias,
                                       __global float *output,
                                       unsigned int picStep,
                                       unsigned int splitNum,
                                       unsigned int batch,
                                       unsigned int topChannel) {
        FC_MERGEOPT_8X1(input, bias, output, picStep, splitNum, batch, topChannel)
})

#undef DATA_T
#undef DATA_T8

#define DATA_T half
#define DATA_T8 half8

ADD_SINGLE_KERNEL(fc_interleave_8x1_FP16, (__global const half *input,
                                         __global half *output,
                                         unsigned int oriRow,
                                         unsigned int oriCol,
                                         unsigned int dstCol) {
        FC_INTERLEAVE_8X1(input, output, oriRow, oriCol, dstCol)
})

ADD_SINGLE_KERNEL(fc_splitopt_8x1_FP16, (__global const half *input,
                                       __global const half *weight,
                                       __global half *output,
                                       unsigned int picStep,
                                       unsigned int splitStep,
                                       unsigned int batch,
                                       unsigned int topChannel,
                                       unsigned int splitNumber) {
        FC_SPLITOPT_8X1(input, weight, output, picStep, splitStep, batch, topChannel, splitNumber)
})

ADD_SINGLE_KERNEL(fc_mergeopt_8x1_FP16, (__global const half *input,
                                       __global const half *bias,
                                       __global half *output,
                                       unsigned int picStep,
                                       unsigned int splitNum,
                                       unsigned int batch,
                                       unsigned int topChannel) {
        FC_MERGEOPT_8X1(input, bias, output, picStep, splitNum, batch, topChannel)
})

#undef DATA_T
#undef DATA_T8

}  // namespace gpu
}  // namespace ud
}  // namespace enn
