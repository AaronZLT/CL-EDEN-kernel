#include "userdriver/gpu/common/CLKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define SCALE(input, scale_data, bias_data, output, channel, height, width, bias_term) \
    int globalID0 = get_global_id(0); \
    int globalID1 = get_global_id(1); \
    int globalID2 = get_global_id(2) * 8; \
    int wh = width * height; \
    if (globalID1 < channel && globalID2 < wh) { \
        int base = globalID0 * channel * wh + globalID1 * wh + globalID2; \
        DATA_T scale = CLAMP_IF_OVERFLOW(scale_data[globalID1]); \
        DATA_T bias = bias_term == 0 ? ZERO : bias_data[globalID1]; \
        if (globalID2 + 8 <= wh) { \
            vstore8(vload8(0, input + base) * scale + bias, 0, output + base); \
        } else { \
            int num = wh - globalID2; \
            if (num == 1) { \
                output[base] = input[base] * scale + bias; \
            } else if (num == 2) { \
                vstore2(vload2(0, input + base) * scale + bias, 0, output + base); \
            } else if (num == 3) { \
                vstore3(vload3(0, input + base) * scale + bias, 0, output + base); \
            } else if (num == 4) { \
                vstore4(vload4(0, input + base) * scale + bias, 0, output + base); \
            } else if (num == 5) { \
                vstore4(vload4(0, input + base) * scale + bias, 0, output + base); \
                output[base + 4] = input[base + 4] * scale + bias; \
            } else if (num == 6) { \
                vstore4(vload4(0, input + base) * scale + bias, 0, output + base); \
                vstore2(vload2(0, input + base + 4) * scale + bias, 0, output + base + 4); \
            } else if (num == 7) { \
                vstore4(vload4(0, input + base) * scale + bias, 0, output + base); \
                vstore3(vload3(0, input + base + 4) * scale + bias, 0, output + base + 4); \
            } \
        } \
    }

#define DATA_T half
#define DATA_T2 half2
#define DATA_T3 half3
#define DATA_T4 half4
#define DATA_T8 half8
#define ZERO (half)0.0f
#define CLAMP_IF_OVERFLOW(x) ((x) > 65504 ? 65504 : (x))
ADD_SINGLE_KERNEL(scale_FP16, (__global half *input,
                               __global const half *scale_data,
                               __global const half *bias_data,
                               __global half *output,
                               unsigned int channel,
                               unsigned int height,
                               unsigned int width,
                               int bias_term) {
    SCALE(input, scale_data, bias_data, output, channel, height, width, bias_term)
})
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T
#undef ZERO

#define DATA_T float
#define DATA_T2 float2
#define DATA_T3 float3
#define DATA_T4 float4
#define DATA_T8 float8
#define ZERO 0.0f
#define CLAMP_IF_OVERFLOW(x) (x)
ADD_SINGLE_KERNEL(scale_FP32, (__global float *input,
                               __global const float *scale_data,
                               __global const float *bias_data,
                               __global float *output,
                               unsigned int channel,
                               unsigned int height,
                               unsigned int width,
                               int bias_term) {
    SCALE(input, scale_data, bias_data, output, channel, height, width, bias_term)
})
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T
#undef ZERO

}  // namespace gpu
}  // namespace ud
}  // namespace enn
