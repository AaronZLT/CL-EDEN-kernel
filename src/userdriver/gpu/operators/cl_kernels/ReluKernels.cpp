#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define DATA_T uchar
ADD_SINGLE_KERNEL(relu_INT8, (__global DATA_T *input, __global DATA_T *output, int min, int max) {
    Q_RELU_X(input, output, min, max)
})
#undef DATA_T

#define DATA_T char
ADD_SINGLE_KERNEL(SIGNEDrelu_INT8, (__global DATA_T *input, __global DATA_T *output, int min, int max) {
    Q_RELU_X(input, output, min, max)
})
#undef DATA_T

#define RELU_FP(input, output, negative_slope, channel, height, width) \
    int globalID0 = get_global_id(0); \
    int globalID1 = get_global_id(1); \
    int globalID2 = get_global_id(2) * 8; \
    int wh = width * height; \
    if (globalID1 < channel && globalID2 < wh) { \
        int base = globalID0 * channel * wh + globalID1 * wh + globalID2; \
        if (globalID2 + 8 <= wh) { \
            DATA_T8 tmp = vload8(0, input + base); \
            vstore8(select(tmp * (DATA_T)(negative_slope), tmp, isgreater(tmp, (DATA_T8)(0.0f))), \
                    0, output + base); \
        } else { \
            int num = wh - globalID2; \
            if (num >= 4) { \
                DATA_T4 tmp = vload4(0, input + base); \
                vstore4(select(tmp * (DATA_T)(negative_slope), tmp, isgreater(tmp, (DATA_T4)(0.0f))), \
                        0, output + base); \
                num -= 4; \
                base += 4; \
            } \
            if (num == 1) { \
                output[base] = select(input[base] * (DATA_T)(negative_slope), input[base], \
                                      CONVERT_TO_SHORT(isgreater(input[base], (DATA_T)(0.0f)))); \
            } else if (num == 2) { \
                DATA_T2 tmp = vload2(0, input + base); \
                vstore2(select(tmp * (DATA_T)(negative_slope), tmp, isgreater(tmp, (DATA_T2)(0.0f))), \
                        0, output + base); \
            } else if (num == 3) { \
                DATA_T3 tmp = vload3(0, input + base); \
                vstore3(select(tmp * (DATA_T)(negative_slope), tmp, isgreater(tmp, (DATA_T3)(0.0f))), \
                        0, output + base); \
            } \
        } \
    }


/********  FP16 KERNELS ********/
#define DATA_T half
#define DATA_T2 half2
#define DATA_T3 half3
#define DATA_T4 half4
#define DATA_T8 half8
#define DATA_T16 half16
#define CONVERT_TO_SHORT(x) ((short)(x))

ADD_SINGLE_KERNEL(relu_FP16, (__global DATA_T *input,
                              __global DATA_T *output,
                              float negative_slope,
                              unsigned int channel,
                              unsigned int height,
                              unsigned int width) {
    RELU_FP(input, output, negative_slope, channel, height, width)
})

#undef CONVERT_TO_SHORT
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
#define CONVERT_TO_SHORT(x) x

ADD_SINGLE_KERNEL(relu_FP32, (__global DATA_T *input,
                              __global DATA_T *output,
                              float negative_slope,
                              unsigned int channel,
                              unsigned int height,
                              unsigned int width) {
    RELU_FP(input, output, negative_slope, channel, height, width)
})

#undef CONVERT_TO_SHORT
#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // float

}  // namespace gpu
}  // namespace ud
}  // namespace enn
