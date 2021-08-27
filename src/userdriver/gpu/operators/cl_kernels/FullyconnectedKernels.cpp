#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define TFLITE_TEXTURE2D_FC(src_data, filters, biases, dst_data, src_slices, dst_slices) \
        int gid = get_global_id(0); \
        bool inside = gid < dst_slices; \
        gid = min(gid, dst_slices - 1); \
        int tid0 = get_local_id(0); \
        int tid1 = get_local_id(1); \
        DATA_T4 s = (DATA_T4)(0.0f, 0.0f, 0.0f, 0.0f); \
        for (int c = tid1; c < src_slices; c += 4) { \
            DATA_T4 v = read_imageh(src_data, smp_none, (int2)((0), (0) * src_slices + (c))); \
            DATA_T16 w = filters[c * dst_slices + gid]; \
            s.x += dot(v, w.s0123); \
            s.y += dot(v, w.s4567); \
            s.z += dot(v, w.s89ab); \
            s.w += dot(v, w.scdef); \
        } \
        __local DATA_T4 temp[32][4]; \
        temp[tid0][tid1] = s; \
        barrier(CLK_LOCAL_MEM_FENCE); \
        if (tid1 == 0 && inside) { \
            s += temp[tid0][1]; \
            s += temp[tid0][2]; \
            s += temp[tid0][3]; \
            DATA_T4 r0 = CONVERT_TO_DATA_T4(s) + biases[gid]; \
            r0 = ACT_VEC_F(DATA_T4, r0); \
            write_imageh(dst_data, (int2)((0), (0) * dst_slices + (gid)), r0); \
        }

#define FC_MERGE(input, bias, picStep, splitStep, output) \
        DATA_T temp = 0.0f; \
        for (int i = 0; i < picStep / splitStep; i++) { \
            temp += input[get_global_id(0) * picStep + i * splitStep + get_global_id(1)]; \
        } \
        output[get_global_id(0) * get_global_size(1) + get_global_id(1)] = \
            temp + bias[get_global_id(1)];

#define FC_SPLIT_COALESCED(input, weight, output, matrix_width, matrix_height, width_split_number) \
        DATA_T8 input8; \
        DATA_T8 weight8; \
        DATA_T8 temp_out = (DATA_T8)(0.0f); \
        unsigned int split_step = 8 * 24 * 32; \
        int input_start = get_global_id(0) * matrix_width + (get_global_id(2) / 24) * split_step + \
                          (get_global_id(2) % 24) * 8; \
        int input_end = get_global_id(0) * matrix_width + matrix_width; \
        int weight_start = get_global_id(1) * matrix_width + (get_global_id(2) / 24) * split_step + \
                           (get_global_id(2) % 24) * 8; \
        for (int i = 0; i < 32; i++) { \
            if (input_start < input_end) { \
                input8 = vload8(0, input + input_start); \
                weight8 = vload8(0, weight + weight_start); \
                temp_out += input8 * weight8; \
                input_start += 192; \
                weight_start += 192; \
            } \
        } \
        output[get_global_id(0) * matrix_height * width_split_number + \
               get_global_id(1) * width_split_number + get_global_id(2)] = \
            temp_out.s0 + temp_out.s1 + temp_out.s2 + temp_out.s3 + temp_out.s4 + temp_out.s5 + \
            temp_out.s6 + temp_out.s7;

#define FC_MERGE_COALESCED(input, bias, matrix_height, width_split_number, output) \
        int batch = get_global_id(0); \
        int c_out = get_global_id(1); \
        if (c_out < matrix_height) { \
            DATA_T8 temp = 0.0f; \
            unsigned int offset = \
                batch * matrix_height * width_split_number + c_out * width_split_number; \
            for (int i = 0; i < width_split_number / 24; i++) { \
                temp += vload8(0, input + offset); \
                temp += vload8(1, input + offset); \
                temp += vload8(2, input + offset); \
                offset += 24; \
            } \
            output[batch * matrix_height + c_out] = temp.s0 + temp.s1 + temp.s2 + temp.s3 + \
                                                    temp.s4 + temp.s5 + temp.s6 + temp.s7 + \
                                                    bias[c_out]; \
        }

/********  FP16 KERNELS ********/
#define DATA_T half
#define DATA_T2 half2
#define DATA_T3 half3
#define DATA_T4 half4
#define DATA_T8 half8
#define DATA_T16 half16
#define CONVERT_TO_DATA_T4(x) convert_half4(x)

// kernels without activation func
#define ACT_VEC_F(VEC_T, x) x
ADD_SINGLE_KERNEL(tflite_texture2d_fc_FP16, (__read_only image2d_t src_data,
                                           __global DATA_T16 *filters,
                                           __global DATA_T4 *biases,
                                           __write_only image2d_t dst_data,
                                           int src_slices,
                                           int dst_slices) {
    TFLITE_TEXTURE2D_FC(src_data, filters, biases, dst_data, src_slices, dst_slices)
})

#undef ACT_VEC_F  // NONE

// apply RELU to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELUtflite_texture2d_fc_FP16, (__read_only image2d_t src_data,
                                           __global DATA_T16 *filters,
                                           __global DATA_T4 *biases,
                                           __write_only image2d_t dst_data,
                                           int src_slices,
                                           int dst_slices) {
    TFLITE_TEXTURE2D_FC(src_data, filters, biases, dst_data, src_slices, dst_slices)
})

#undef ACT_VEC_F  // RELU

// apply RELU6 to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU6_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELU6tflite_texture2d_fc_FP16, (__read_only image2d_t src_data,
                                           __global DATA_T16 *filters,
                                           __global DATA_T4 *biases,
                                           __write_only image2d_t dst_data,
                                           int src_slices,
                                           int dst_slices) {
    TFLITE_TEXTURE2D_FC(src_data, filters, biases, dst_data, src_slices, dst_slices)
})

#undef ACT_VEC_F  // RELU6

ADD_SINGLE_KERNEL(fc_merge_FP16, (__global const DATA_T *input,
                                __global const DATA_T *bias,
                                unsigned int picStep,
                                unsigned int splitStep,
                                __global DATA_T *output) {
    FC_MERGE(input, bias, picStep, splitStep, output)
})

ADD_SINGLE_KERNEL(fc_split_coalesced_FP16, (__global const DATA_T *input,
                                          __global const DATA_T *weight,
                                          __global DATA_T *output,
                                          unsigned int matrix_width,
                                          unsigned int matrix_height,
                                          unsigned int width_split_number) {
    FC_SPLIT_COALESCED(input, weight, output, matrix_width, matrix_height, width_split_number)
})

ADD_SINGLE_KERNEL(fc_merge_coalesced_FP16, (__global const DATA_T *input,
                                          __global const DATA_T *bias,
                                          unsigned int matrix_height,
                                          unsigned int width_split_number,
                                          __global DATA_T *output) {
    FC_MERGE_COALESCED(input, bias, matrix_height, width_split_number, output)
})

ADD_SINGLE_KERNEL(alignWeight_tflitefc_FP16, (__global const half *src,
                                            __global half *dst,
                                            const unsigned int batch,
                                            const unsigned int channel,
                                            const int kh,
                                            const int kw,
                                            const int dst_depth,
                                            const int src_depth) {
        int g0 = get_global_id(0);
        int g1 = get_global_id(1);
        if (g0 >= src_depth || g1 >= dst_depth)
            return;

        int counter = (g0 * dst_depth + g1) * 16;
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                const int dst_ch = g1 * 4 + i;
                const int src_ch = g0 * 4 + j;
                if (dst_ch < batch && src_ch < channel) {
                    const int f_index = dst_ch * kh * kw * channel + src_ch * kh * kw;
                    dst[counter + i * 4 + j] = src[f_index];
                } else {
                    dst[counter + i * 4 + j] = (half)0.0f;
                }
            }
        }
})

ADD_SINGLE_KERNEL(fc_split_FP16, (__global const half *input,
                                __global const half *weight,
                                __global half *output,
                                unsigned int picStep,
                                unsigned int splitStep) {
        half8 input8;
        half8 weight8;
        half8 temp_out = (half8)(0.0f);
        int inputStart = get_global_id(0) * picStep + get_global_id(1) * splitStep;
        int weightStart = get_global_id(2) * picStep + get_global_id(1) * splitStep;

        for (int n = 0; n < (splitStep >> 3); n++) {
            input8 = vload8(0, input + inputStart + n * 8);
            weight8 = vload8(0, weight + weightStart + n * 8);
            temp_out += input8 * weight8;
        }

        output[get_global_id(0) * get_global_size(1) * get_global_size(2) +
               get_global_id(1) * get_global_size(2) + get_global_id(2)] =
            (half)(temp_out.s0 + temp_out.s1 + temp_out.s2 + temp_out.s3 + temp_out.s4 +
                   temp_out.s5 + temp_out.s6 + temp_out.s7);

        if (splitStep % 8 != 0) {
            input8 = vload8(0, input + inputStart + (splitStep >> 3) * 8);
            weight8 = vload8(0, weight + weightStart + (splitStep >> 3) * 8);
            temp_out = input8 * weight8;
            if (splitStep % 8 == 1) {
                output[get_global_id(0) * get_global_size(1) * get_global_size(2) +
                       get_global_id(1) * get_global_size(2) + get_global_id(2)] += temp_out.s0;
            } else if (splitStep % 8 == 2) {
                output[get_global_id(0) * get_global_size(1) * get_global_size(2) +
                       get_global_id(1) * get_global_size(2) + get_global_id(2)] +=
                    temp_out.s0 + temp_out.s1;
            } else if (splitStep % 8 == 3) {
                output[get_global_id(0) * get_global_size(1) * get_global_size(2) +
                       get_global_id(1) * get_global_size(2) + get_global_id(2)] +=
                    temp_out.s0 + temp_out.s1 + temp_out.s2;
            } else if (splitStep % 8 == 4) {
                output[get_global_id(0) * get_global_size(1) * get_global_size(2) +
                       get_global_id(1) * get_global_size(2) + get_global_id(2)] +=
                    temp_out.s0 + temp_out.s1 + temp_out.s2 + temp_out.s3;
            } else if (splitStep % 8 == 5) {
                output[get_global_id(0) * get_global_size(1) * get_global_size(2) +
                       get_global_id(1) * get_global_size(2) + get_global_id(2)] +=
                    temp_out.s0 + temp_out.s1 + temp_out.s2 + temp_out.s3 + temp_out.s4;
            } else if (splitStep % 8 == 6) {
                output[get_global_id(0) * get_global_size(1) * get_global_size(2) +
                       get_global_id(1) * get_global_size(2) + get_global_id(2)] +=
                    temp_out.s0 + temp_out.s1 + temp_out.s2 + temp_out.s3 + temp_out.s4 +
                    temp_out.s5;
            } else if (splitStep % 8 == 7) {
                output[get_global_id(0) * get_global_size(1) * get_global_size(2) +
                       get_global_id(1) * get_global_size(2) + get_global_id(2)] +=
                    temp_out.s0 + temp_out.s1 + temp_out.s2 + temp_out.s3 + temp_out.s4 +
                    temp_out.s5 + temp_out.s6;
            }
        }
})

#undef CONVERT_TO_DATA_T4
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

ADD_SINGLE_KERNEL(fc_merge_FP32, (__global const DATA_T *input,
                                __global const DATA_T *bias,
                                unsigned int picStep,
                                unsigned int splitStep,
                                __global DATA_T *output) {
    FC_MERGE(input, bias, picStep, splitStep, output)
})

ADD_SINGLE_KERNEL(fc_split_coalesced_FP32, (__global const DATA_T *input,
                                          __global const DATA_T *weight,
                                          __global DATA_T *output,
                                          unsigned int matrix_width,
                                          unsigned int matrix_height,
                                          unsigned int width_split_number) {
    FC_SPLIT_COALESCED(input, weight, output, matrix_width, matrix_height, width_split_number)
})

ADD_SINGLE_KERNEL(fc_merge_coalesced_FP32, (__global const DATA_T *input,
                                          __global const DATA_T *bias,
                                          unsigned int matrix_height,
                                          unsigned int width_split_number,
                                          __global DATA_T *output) {
    FC_MERGE_COALESCED(input, bias, matrix_height, width_split_number, output)
})

ADD_SINGLE_KERNEL(fc_split_FP32, (__global const float *input,
                                __global const float *weight,
                                __global float *output,
                                unsigned int picStep,
                                unsigned int splitStep) {
        float out = 0.0f;
        float4 input4;
        float4 weight4;
        int2 inputStart = (int2)(get_global_id(0) * picStep + get_global_id(1) * splitStep,
                                 get_global_id(2) * picStep + get_global_id(1) * splitStep);

        for (int n = 0; n < (splitStep >> 2); n++) {
            input4 = vload4(0, input + inputStart.s0);
            weight4 = vload4(0, weight + inputStart.s1);
            out += dot(input4, weight4);
            inputStart += 4;
        }

        if (splitStep % 4 != 0) {
            float4 temp_out = (float4)(0.0f);
            input4 = vload4(0, input + inputStart.s0);
            weight4 = vload4(0, weight + inputStart.s1);
            temp_out = input4 * weight4;
            if (splitStep % 4 == 1) {
                out += temp_out.x;
            } else if (splitStep % 4 == 2) {
                out += temp_out.x + temp_out.y;
            } else if (splitStep % 4 == 3) {
                out += temp_out.x + temp_out.y + temp_out.z;
            }
        }

        int output_index = get_global_id(0) * get_global_size(1) * get_global_size(2) +
                           get_global_id(1) * get_global_size(2) + get_global_id(2);
        output[output_index] = out;
    }
)

#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // float

}  // namespace gpu
}  // namespace ud
}  // namespace enn
