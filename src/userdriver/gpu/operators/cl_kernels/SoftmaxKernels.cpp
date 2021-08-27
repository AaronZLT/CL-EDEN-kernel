#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define SOFTMAX_AXIS2(input, output, channel_max, channel_sum, channels, innerNumber, outNumber, beta) \
    int globalID0 = get_global_id(0); \
    int offset = globalID0 * channels * innerNumber; \
    for (int i = 0; i < innerNumber; i++) { \
        channel_max[i + offset] = input[offset + i]; \
        for (int c = 1; c < channels; c++) { \
            if (input[offset + i + innerNumber * c] > channel_max[i + offset]) { \
                channel_max[i + offset] = input[offset + i + innerNumber * c]; \
            } \
        } \
    } \
    for (int i = 0; i < innerNumber; i++) { \
        channel_sum[i + offset] = 0.0f; \
        for (int c = 0; c < channels; c++) { \
            output[offset + i + innerNumber * c] = \
                exp((input[offset + i + innerNumber * c] - channel_max[i + offset]) * beta); \
            channel_sum[i + offset] += output[offset + i + innerNumber * c]; \
        } \
    } \
    for (int i = 0; i < innerNumber; i++) { \
        for (int c = 0; c < channels; c++) { \
            output[offset + i + innerNumber * c] = \
                output[offset + i + innerNumber * c] / channel_sum[i + offset]; \
        } \
    }

#define SOFTMAX(input, output, channel, bottomStep, beta) \
    int inputOffset = get_global_id(0) * bottomStep; \
    { \
        DATA_T8 maxInput = vload8(0, input + inputOffset); \
        for (int c = 1; c < channel / 8; c++) { \
            maxInput = fmax(maxInput, vload8(c, input + inputOffset)); \
        } \
        maxInput.s01 = fmax(maxInput.s01, maxInput.s23); \
        maxInput.s45 = fmax(maxInput.s45, maxInput.s67); \
        maxInput.s01 = fmax(maxInput.s01, maxInput.s45); \
        maxInput.s0 = fmax(maxInput.s0, maxInput.s1); \
        DATA_T sumInput = 0.0; \
        for (int c = 0; c < channel / 8; c++) { \
            DATA_T8 expRes = vload8(c, input + inputOffset) - maxInput.s0; \
            expRes *= (DATA_T)beta; \
            expRes.s0 = exp(expRes.s0); \
            expRes.s1 = exp(expRes.s1); \
            expRes.s2 = exp(expRes.s2); \
            expRes.s3 = exp(expRes.s3); \
            expRes.s4 = exp(expRes.s4); \
            expRes.s5 = exp(expRes.s5); \
            expRes.s6 = exp(expRes.s6); \
            expRes.s7 = exp(expRes.s7); \
            vstore8(expRes, c, output + inputOffset); \
            sumInput += expRes.s0; \
            sumInput += expRes.s1; \
            sumInput += expRes.s2; \
            sumInput += expRes.s3; \
            sumInput += expRes.s4; \
            sumInput += expRes.s5; \
            sumInput += expRes.s6; \
            sumInput += expRes.s7; \
        } \
        for (int c = 0; c < channel / 8; c++) { \
            vstore8(vload8(c, output + inputOffset) / sumInput, c, output + inputOffset); \
        } \
    }

#define DATA_T float
#define DATA_T8 float8
ADD_SINGLE_KERNEL(softmax_axis2_FP32, (__global float *input,
                                     __global float *output,
                                     __global float *channel_max,
                                     __global float *channel_sum,
                                     unsigned int channels,
                                     unsigned int innerNumber,
                                     unsigned int outNumber,
                                     float beta) {
        SOFTMAX_AXIS2(input, output, channel_max, channel_sum, channels, innerNumber, outNumber, beta)
})

ADD_SINGLE_KERNEL(softmax_FP32, (__global float *input,
                               __global float *output,
                               unsigned int channel,
                               unsigned int bottomStep,
                               float beta) {
        SOFTMAX(input, output, channel, bottomStep, beta)
})

#undef DATA_T8
#undef DATA_T

#define DATA_T half
#define DATA_T8 half8
ADD_SINGLE_KERNEL(softmax_axis2_FP16, (__global half *input,
                                     __global half *output,
                                     __global float *channel_max,
                                     __global float *channel_sum,
                                     unsigned int channels,
                                     unsigned int innerNumber,
                                     unsigned int outNumber,
                                     float beta) {
    SOFTMAX_AXIS2(input, output, channel_max, channel_sum, channels, innerNumber, outNumber, beta)
})

ADD_SINGLE_KERNEL(softmax_FP16, (__global half *input,
                               __global half *output,
                               unsigned int channel,
                               unsigned int bottomStep,
                               float beta) {
    SOFTMAX(input, output, channel, bottomStep, beta)
})

ADD_SINGLE_KERNEL(softmax_texture2d_FP16, (__read_only image2d_t src_data,
                    __write_only image2d_t dst_data,
                    int src_size_x,
                    int src_size_y,
                    int src_size_z,
                    int src_size_w,
                    int slice_size_x,
                    int slice_size_y,
                    float mask0,
                    float mask1,
                    float mask2,
                    float mask3) {
         int offset = 0;
         float sum = 0.0f;
         int s = 0;
         float4 mask = (float4)(mask0, mask1, mask2, mask3);
         int tid = get_local_id(0);
         do {
             int z = offset + tid;
             if (z < slice_size_x) {
                 float4 mask_temp = z == slice_size_x - 1 ? mask : (float4)(1.0f);
                 float4 src = read_imagef(src_data, smp_none, (int2)((0), (0) * src_size_z + (z)));
                 sum += dot(mask_temp, exp(src));
                 offset += 32;
             }
             s++;
         } while (s < slice_size_y);

         __local float4 tmp[8];
         __local float* tmpx1 = (__local float*)tmp;
         tmpx1[tid] = sum;
         barrier(CLK_LOCAL_MEM_FENCE);
         if (tid == 0) {
             sum = dot((float4)(1.0f), tmp[0]);
             sum += dot((float4)(1.0f), tmp[1]);
             sum += dot((float4)(1.0f), tmp[2]);
             sum += dot((float4)(1.0f), tmp[3]);
             sum += dot((float4)(1.0f), tmp[4]);
             sum += dot((float4)(1.0f), tmp[5]);
             sum += dot((float4)(1.0f), tmp[6]);
             sum += dot((float4)(1.0f), tmp[7]);
             tmpx1[0] = 1.0f / sum;
         }

         barrier(CLK_LOCAL_MEM_FENCE);
         sum = tmpx1[0];

         offset = 0;
         s = 0;
         do {
             int z = offset + tid;
             if (z < slice_size_x) {
                float4 src = read_imagef(src_data, smp_none, (int2)((0), (0) * src_size_z + (z)));
                half4 res = convert_half4(exp(src) * sum);
                write_imageh(dst_data, (int2)((0), (0) * src_size_z + (z)), res);
                offset += 32;
             }
            s++;
         } while (s < slice_size_y);
})

ADD_SINGLE_KERNEL(softmax_local_FP16,(__global half *input,
                                   __global half *output,
                                   unsigned int compute_slices,
                                   unsigned int src_channel) {
         int offset = 0;
         float sum = 0.0f;
         int s = 0;

         int tid = get_local_id(0); // 0 - 31
         do {
             int z = offset + tid;
             if (z < src_channel) {
                float t = (float)(input[z]);
                sum += exp(t);
             }

             offset += 32;
             s++;
         } while (s < compute_slices);

         __local float4 tmp[8];
         __local float* tmpx1 = (__local float*)tmp;
         tmpx1[tid] = sum;
         barrier(CLK_LOCAL_MEM_FENCE);
         if (tid == 0) {
             sum = dot((float4)(1.0f), tmp[0]);
             sum += dot((float4)(1.0f), tmp[1]);
             sum += dot((float4)(1.0f), tmp[2]);
             sum += dot((float4)(1.0f), tmp[3]);
             sum += dot((float4)(1.0f), tmp[4]);
             sum += dot((float4)(1.0f), tmp[5]);
             sum += dot((float4)(1.0f), tmp[6]);
             sum += dot((float4)(1.0f), tmp[7]);
             tmpx1[0] = 1.0f / sum;
         }

         barrier(CLK_LOCAL_MEM_FENCE);
         sum = tmpx1[0];

         offset = 0;
         s = 0;
         do {
             int z = offset + tid;
             if (z < src_channel) {
                float out = (float)(input[z]);
                output[z] = (half)(exp(out) * sum);
             }
             offset += 32;
             s++;
         } while (s < compute_slices);
})

ADD_SINGLE_KERNEL(softmax_local_FP32,(__global float *input,
                                      __global float *output,
                                      unsigned int compute_slices,
                                      unsigned int src_channel) {
         int offset = 0;
         float sum = 0.0f;
         int s = 0;

         int tid = get_local_id(0); // 0 - 31
         do {
             int z = offset + tid;
             if (z < src_channel) {
                float t = input[z];
                sum += exp(t);
             }

             offset += 32;
             s++;
         } while (s < compute_slices);

         __local float4 tmp[8];
         __local float* tmpx1 = (__local float*)tmp;
         tmpx1[tid] = sum;
         barrier(CLK_LOCAL_MEM_FENCE);
         if (tid == 0) {
             sum = dot((float4)(1.0f), tmp[0]);
             sum += dot((float4)(1.0f), tmp[1]);
             sum += dot((float4)(1.0f), tmp[2]);
             sum += dot((float4)(1.0f), tmp[3]);
             sum += dot((float4)(1.0f), tmp[4]);
             sum += dot((float4)(1.0f), tmp[5]);
             sum += dot((float4)(1.0f), tmp[6]);
             sum += dot((float4)(1.0f), tmp[7]);
             tmpx1[0] = 1.0f / sum;
         }

         barrier(CLK_LOCAL_MEM_FENCE);
         sum = tmpx1[0];

         offset = 0;
         s = 0;
         do {
             int z = offset + tid;
             if (z < src_channel) {
                float out = input[z];
                output[z] = exp(out) * sum;
             }
             offset += 32;
             s++;
         } while (s < compute_slices);
})

#undef DATA_T8
#undef DATA_T

}  // namespace gpu
}  // namespace ud
}  // namespace enn
