#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define DATA_T uchar
ADD_SINGLE_KERNEL(relu6_INT8, (__global DATA_T *input, __global DATA_T *output, int min, int max) {
    Q_RELU_X(input, output, min, max)
})
#undef DATA_T

#define DATA_T char
ADD_SINGLE_KERNEL(SIGNEDrelu6_INT8, (__global DATA_T *input, __global DATA_T *output, int min, int max) {
    Q_RELU_X(input, output, min, max)
})
#undef DATA_T

// apply RELU6 to kernels' output
#define ACT_VEC_F(VEC_T, x) RELU6_VEC_F(VEC_T, x)
/********  FP16 KERNELS ********/
#define DATA_T half
#define DATA_T2 half2
#define DATA_T3 half3
#define DATA_T4 half4
#define DATA_T8 half8
#define DATA_T16 half16

ADD_SINGLE_KERNEL(relu6_FP16, (__global DATA_T *input,
                               __global DATA_T *output,
                               unsigned int channel,
                               unsigned int height,
                               unsigned int width) {
    RELU_X(input, output, channel, height, width)
})

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

ADD_SINGLE_KERNEL(relu6_FP32, (__global DATA_T *input,
                               __global DATA_T *output,
                               unsigned int channel,
                               unsigned int height,
                               unsigned int width) {
    RELU_X(input, output, channel, height, width)
})

#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // float

#undef ACT_VEC_F

}  // namespace gpu
}  // namespace ud
}  // namespace enn
