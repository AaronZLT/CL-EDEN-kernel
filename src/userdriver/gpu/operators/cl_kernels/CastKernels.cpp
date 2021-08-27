#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define TO_SELF(x, y) \
    int idx = IDX_IN_3D; \
    y[idx] = x[idx];

#define TO_FLOAT(x, y) \
    int idx = IDX_IN_3D; \
    y[idx] = (float)x[idx];

#define TO_HALF(x, y) \
    int idx = IDX_IN_3D; \
    y[idx] = (half)x[idx];

#define TO_INT(x, y) \
    int idx = IDX_IN_3D; \
    y[idx] = (int)x[idx];

#define TO_UCHAR(x, y) \
    int idx = IDX_IN_3D; \
    y[idx] = (unsigned char)x[idx];

ADD_SINGLE_KERNEL(cast_float_to_float_FP32, (__global float *input, __global float *output) {
        TO_SELF(input, output)
    })

ADD_SINGLE_KERNEL(cast_float_to_half_FP32, (__global float *input, __global float *output) {
        TO_SELF(input, output)
    })

ADD_SINGLE_KERNEL(cast_float_to_int_FP32, (__global float *input, __global int *output) {
        TO_INT(input, output)
    })

ADD_SINGLE_KERNEL(cast_float_to_int8_FP32, (__global float *input, __global unsigned char *output) {
        TO_UCHAR(input, output)
        if (input[idx] > (float)255.0f)
            output[idx] = (unsigned char)255;
    })

ADD_SINGLE_KERNEL(cast_half_to_float_FP32, (__global float *input, __global float *output) {
        TO_SELF(input, output)
    })

ADD_SINGLE_KERNEL(cast_half_to_half_FP32, (__global float *input, __global float *output) {
        TO_SELF(input, output)
    })

ADD_SINGLE_KERNEL(cast_half_to_int_FP32, (__global float *input, __global int *output) {
        TO_INT(input, output)
    })

ADD_SINGLE_KERNEL(cast_half_to_int8_FP32, (__global float *input, __global unsigned char *output) {
        TO_UCHAR(input, output)
        if (input[idx] > (float)255.0f)
            output[idx] = (unsigned char)255;
    })

ADD_SINGLE_KERNEL(cast_int_to_float_FP32, (__global int *input, __global float *output) {
        TO_FLOAT(input, output)
    })

ADD_SINGLE_KERNEL(cast_int_to_half_FP32, (__global int *input, __global float *output) {
        TO_FLOAT(input, output)
    })

ADD_SINGLE_KERNEL(cast_int_to_int_FP32, (__global int *input, __global int *output) {
        TO_SELF(input, output)
    })

ADD_SINGLE_KERNEL(cast_int_to_int8_FP32, (__global int *input, __global unsigned char *output) {
        TO_UCHAR(input, output)
        if (input[idx] > 255)
            output[idx] = (unsigned char)255;
        if (input[idx] < 0)
            output[idx] = (unsigned char)0;
    })

ADD_SINGLE_KERNEL(cast_int8_to_float_FP32, (__global unsigned char *input, __global float *output) {
        TO_FLOAT(input, output)
    })

ADD_SINGLE_KERNEL(cast_int8_to_half_FP32, (__global unsigned char *input, __global float *output) {
        TO_FLOAT(input, output)
    })

ADD_SINGLE_KERNEL(cast_int8_to_int_FP32, (__global unsigned char *input, __global int *output) {
        TO_INT(input, output)
    })

ADD_SINGLE_KERNEL(cast_int8_to_int8_FP32, (__global unsigned char *input,
                                         __global unsigned char *output) {
        TO_SELF(input, output)
    })

ADD_SINGLE_KERNEL(cast_bool_to_bool_FP32, (__global bool *input, __global bool *output) {
        TO_SELF(input, output)
    })

ADD_SINGLE_KERNEL(cast_uint16_to_uint16_FP32, (__global ushort *input, __global ushort *output) {
        TO_SELF(input, output)
    })

ADD_SINGLE_KERNEL(cast_float_to_float_FP16, (__global half *input, __global half *output) {
        TO_SELF(input, output)
    })

ADD_SINGLE_KERNEL(cast_float_to_half_FP16, (__global half *input, __global half *output) {
        TO_SELF(input, output)
    })

ADD_SINGLE_KERNEL(cast_float_to_int_FP16, (__global half *input, __global int *output) {
        TO_INT(input, output)
    })

ADD_SINGLE_KERNEL(cast_float_to_int8_FP16, (__global half *input, __global unsigned char *output) {
    TO_UCHAR(input, output)
    if (input[idx] > (half)255.0f)
        output[idx] = (unsigned char)255;
})

ADD_SINGLE_KERNEL(cast_half_to_float_FP16, (__global half *input, __global half *output) {
    TO_SELF(input, output)
})

ADD_SINGLE_KERNEL(cast_half_to_half_FP16, (__global half *input, __global half *output) {
    TO_SELF(input, output)
})

ADD_SINGLE_KERNEL(cast_half_to_int_FP16, (__global half *input, __global int *output) {
    TO_INT(input, output)
})

ADD_SINGLE_KERNEL(cast_half_to_int8_FP16, (__global half *input, __global unsigned char *output) {
    TO_UCHAR(input, output)
    if (input[idx] > (half)255.0f)
        output[idx] = (unsigned char)255;
})

ADD_SINGLE_KERNEL(cast_int_to_float_FP16, (__global int *input, __global half *output) {
    TO_HALF(input, output)
})

ADD_SINGLE_KERNEL(cast_int_to_half_FP16, (__global int *input, __global half *output) {
    TO_HALF(input, output)
})

ADD_SINGLE_KERNEL(cast_int8_to_float_FP16, (__global unsigned char *input, __global half *output) {
    TO_HALF(input, output)
})

ADD_SINGLE_KERNEL(cast_int8_to_half_FP16, (__global unsigned char *input, __global half *output) {
    TO_HALF(input, output)
})

}  // namespace gpu
}  // namespace ud
}  // namespace enn
