#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define ELTWISE_ADD_ZERO_ONE(inputA, inputB, output, cofA, cofB, nchw) \
        int offset = get_global_id(0) * 8; \
        if (offset < nchw) { \
            DATA_T cof1 = (DATA_T)cofA; \
            DATA_T cof2 = (DATA_T)cofB; \
            DATA_T8 res = cof1 * vload8(0, inputA + offset) + cof2 * vload8(0, inputB + offset); \
            vstore8(ACT_VEC_F(DATA_T8, res), 0, output + offset); \
        }

#define ELTWISE_ADD_ZERO_ONE_TEXTURE2D(inputA, inputB, output, cofA, cofB, imageW, imageH) \
        int g0 = get_global_id(0); \
        int g1 = get_global_id(1); \
        if (g0 >= imageW || g1 >= imageH) \
            return; \
        DATA_T4 res = READ_IMAGE_T(inputA, smp_none, (int2)(g0, g1)) + READ_IMAGE_T(inputB, smp_none, (int2)(g0, g1)); \
        res = ACT_VEC_F(DATA_T4, res); \
        WRITE_IMAGE_T(output, (int2)(g0, g1), res);

#define ELTWISE_ADD_VECTOR_CONSTANT(inputA, inputB, output, cofA, cofB, hw) \
        int g0 = get_global_id(0); \
        int g1 = get_global_id(1) * 8; \
        int offset = g0 * hw + g1; \
        if (g1 < hw) { \
            DATA_T cof1 = (DATA_T)cofA; \
            DATA_T cof2 = (DATA_T)cofB; \
            DATA_T8 res = cof1 * vload8(0, inputA + offset) + cof2 * inputB[g0]; \
            vstore8(ACT_VEC_F(DATA_T8, res), 0, output + offset); \
        }

/********  FP32 KERNELS ********/
#define DATA_T float
#define DATA_T2 float2
#define DATA_T3 float3
#define DATA_T4 float4
#define DATA_T8 float8
#define DATA_T16 float16
#define READ_IMAGE_T(x, SAMP, COORD) read_imagef(x, SAMP, COORD)
#define WRITE_IMAGE_T(y, COORD, x) write_imagef(y, COORD, x)

// kernels without activation func
#define ACT_VEC_F(VEC_T, x) x
ADD_SINGLE_KERNEL(eltwise_add_zero_one_FP32, (__global DATA_T *inputA,
                                            __global DATA_T *inputB,
                                            __global DATA_T *output,
                                            float cofA,
                                            float cofB,
                                            unsigned int nchw) {
    ELTWISE_ADD_ZERO_ONE(inputA, inputB, output, cofA, cofB, nchw)
})

ADD_SINGLE_KERNEL(eltwise_add_zero_one_texture2d_FP32, (__read_only image2d_t inputA,
                                                      __read_only image2d_t inputB,
                                                      __write_only image2d_t output,
                                                      float cofA,
                                                      float cofB,
                                                      unsigned int imageW,
                                                      unsigned int imageH) {
    ELTWISE_ADD_ZERO_ONE_TEXTURE2D(inputA, inputB, output, cofA, cofB, imageW, imageH)
})

ADD_SINGLE_KERNEL(eltwise_add_vector_constant_FP32, (__global DATA_T *inputA,
                                                   __global DATA_T *inputB,
                                                   __global DATA_T *output,
                                                   float cofA,
                                                   float cofB,
                                                   unsigned int hw) {
    ELTWISE_ADD_VECTOR_CONSTANT(inputA, inputB, output, cofA, cofB, hw)
})

#undef ACT_VEC_F  // NONE

// apply RELU to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELUeltwise_add_zero_one_FP32, (__global DATA_T *inputA,
                                            __global DATA_T *inputB,
                                            __global DATA_T *output,
                                            float cofA,
                                            float cofB,
                                            unsigned int nchw) {
    ELTWISE_ADD_ZERO_ONE(inputA, inputB, output, cofA, cofB, nchw)
})

ADD_SINGLE_KERNEL(RELUeltwise_add_zero_one_texture2d_FP32, (__read_only image2d_t inputA,
                                                      __read_only image2d_t inputB,
                                                      __write_only image2d_t output,
                                                      float cofA,
                                                      float cofB,
                                                      unsigned int imageW,
                                                      unsigned int imageH) {
    ELTWISE_ADD_ZERO_ONE_TEXTURE2D(inputA, inputB, output, cofA, cofB, imageW, imageH)
})

ADD_SINGLE_KERNEL(RELUeltwise_add_vector_constant_FP32, (__global DATA_T *inputA,
                                                   __global DATA_T *inputB,
                                                   __global DATA_T *output,
                                                   float cofA,
                                                   float cofB,
                                                   unsigned int hw) {
    ELTWISE_ADD_VECTOR_CONSTANT(inputA, inputB, output, cofA, cofB, hw)
})

#undef ACT_VEC_F  // RELU

// apply RELU6 to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU6_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELU6eltwise_add_zero_one_FP32, (__global DATA_T *inputA,
                                            __global DATA_T *inputB,
                                            __global DATA_T *output,
                                            float cofA,
                                            float cofB,
                                            unsigned int nchw) {
    ELTWISE_ADD_ZERO_ONE(inputA, inputB, output, cofA, cofB, nchw)
})

ADD_SINGLE_KERNEL(RELU6eltwise_add_zero_one_texture2d_FP32, (__read_only image2d_t inputA,
                                                      __read_only image2d_t inputB,
                                                      __write_only image2d_t output,
                                                      float cofA,
                                                      float cofB,
                                                      unsigned int imageW,
                                                      unsigned int imageH) {
    ELTWISE_ADD_ZERO_ONE_TEXTURE2D(inputA, inputB, output, cofA, cofB, imageW, imageH)
})

ADD_SINGLE_KERNEL(RELU6eltwise_add_vector_constant_FP32, (__global DATA_T *inputA,
                                                   __global DATA_T *inputB,
                                                   __global DATA_T *output,
                                                   float cofA,
                                                   float cofB,
                                                   unsigned int hw) {
    ELTWISE_ADD_VECTOR_CONSTANT(inputA, inputB, output, cofA, cofB, hw)
})

#undef ACT_VEC_F  // RELU6

ADD_SINGLE_KERNEL(eltwise_add_zero_one_int_FP32, (__global int *inputA,
                                                __global int *inputB,
                                                __global int *output,
                                                float cofA,
                                                float cofB,
                                                unsigned int nchw) {
        int offset = get_global_id(0) * 8;
        if (offset < nchw) {
            int8 res = vload8(0, inputA + offset) + vload8(0, inputB + offset);
            vstore8(res, 0, output + offset);
        }
    })

ADD_SINGLE_KERNEL(eltwise_add_two_more_FP32, (__global float *input,
                                            __global float *output,
                                            float coeff) {
        int index = get_global_id(0) * get_global_size(1) + get_global_id(1);
        output[index] += coeff * input[index];
    }
)

#undef WRITE_IMAGE_T
#undef READ_IMAGE_T
#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // float

/********  FP16 KERNELS ********/
#define DATA_T half
#define DATA_T2 half2
#define DATA_T3 half3
#define DATA_T4 half4
#define DATA_T8 half8
#define DATA_T16 half16
#define READ_IMAGE_T(x, SAMP, COORD) read_imageh(x, SAMP, COORD)
#define WRITE_IMAGE_T(y, COORD, x) write_imageh(y, COORD, x)

// kernels without activation func
#define ACT_VEC_F(VEC_T, x) x
ADD_SINGLE_KERNEL(eltwise_add_zero_one_FP16, (__global DATA_T *inputA,
                                            __global DATA_T *inputB,
                                            __global DATA_T *output,
                                            float cofA,
                                            float cofB,
                                            unsigned int nchw) {
    ELTWISE_ADD_ZERO_ONE(inputA, inputB, output, cofA, cofB, nchw)
})

ADD_SINGLE_KERNEL(eltwise_add_zero_one_texture2d_FP16, (__read_only image2d_t inputA,
                                                      __read_only image2d_t inputB,
                                                      __write_only image2d_t output,
                                                      float cofA,
                                                      float cofB,
                                                      unsigned int imageW,
                                                      unsigned int imageH) {
    ELTWISE_ADD_ZERO_ONE_TEXTURE2D(inputA, inputB, output, cofA, cofB, imageW, imageH)
})

ADD_SINGLE_KERNEL(eltwise_add_vector_constant_FP16, (__global DATA_T *inputA,
                                                   __global DATA_T *inputB,
                                                   __global DATA_T *output,
                                                   float cofA,
                                                   float cofB,
                                                   unsigned int hw) {
    ELTWISE_ADD_VECTOR_CONSTANT(inputA, inputB, output, cofA, cofB, hw)
})

#undef ACT_VEC_F  // NONE

// apply RELU to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELUeltwise_add_zero_one_FP16, (__global DATA_T *inputA,
                                            __global DATA_T *inputB,
                                            __global DATA_T *output,
                                            float cofA,
                                            float cofB,
                                            unsigned int nchw) {
    ELTWISE_ADD_ZERO_ONE(inputA, inputB, output, cofA, cofB, nchw)
})

ADD_SINGLE_KERNEL(RELUeltwise_add_zero_one_texture2d_FP16, (__read_only image2d_t inputA,
                                                      __read_only image2d_t inputB,
                                                      __write_only image2d_t output,
                                                      float cofA,
                                                      float cofB,
                                                      unsigned int imageW,
                                                      unsigned int imageH) {
    ELTWISE_ADD_ZERO_ONE_TEXTURE2D(inputA, inputB, output, cofA, cofB, imageW, imageH)
})

ADD_SINGLE_KERNEL(RELUeltwise_add_vector_constant_FP16, (__global DATA_T *inputA,
                                                   __global DATA_T *inputB,
                                                   __global DATA_T *output,
                                                   float cofA,
                                                   float cofB,
                                                   unsigned int hw) {
    ELTWISE_ADD_VECTOR_CONSTANT(inputA, inputB, output, cofA, cofB, hw)
})

#undef ACT_VEC_F  // RELU

// apply RELU6 to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU6_VEC_F(VEC_T, x)
ADD_SINGLE_KERNEL(RELU6eltwise_add_zero_one_FP16, (__global DATA_T *inputA,
                                            __global DATA_T *inputB,
                                            __global DATA_T *output,
                                            float cofA,
                                            float cofB,
                                            unsigned int nchw) {
    ELTWISE_ADD_ZERO_ONE(inputA, inputB, output, cofA, cofB, nchw)
})

ADD_SINGLE_KERNEL(RELU6eltwise_add_zero_one_texture2d_FP16, (__read_only image2d_t inputA,
                                                      __read_only image2d_t inputB,
                                                      __write_only image2d_t output,
                                                      float cofA,
                                                      float cofB,
                                                      unsigned int imageW,
                                                      unsigned int imageH) {
    ELTWISE_ADD_ZERO_ONE_TEXTURE2D(inputA, inputB, output, cofA, cofB, imageW, imageH)
})

ADD_SINGLE_KERNEL(RELU6eltwise_add_vector_constant_FP16, (__global DATA_T *inputA,
                                                   __global DATA_T *inputB,
                                                   __global DATA_T *output,
                                                   float cofA,
                                                   float cofB,
                                                   unsigned int hw) {
    ELTWISE_ADD_VECTOR_CONSTANT(inputA, inputB, output, cofA, cofB, hw)
})

#undef ACT_VEC_F  // RELU6

ADD_SINGLE_KERNEL(eltwise_add_two_more_FP16, (__global half *input,
                                            __global half *output,
                                            float coeff) {
        int index = get_global_id(0) * get_global_size(1) + get_global_id(1);
        output[index] += coeff * input[index];
    }
)

#undef WRITE_IMAGE_T
#undef READ_IMAGE_T
#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // half

}  // namespace gpu
}  // namespace ud
}  // namespace enn
