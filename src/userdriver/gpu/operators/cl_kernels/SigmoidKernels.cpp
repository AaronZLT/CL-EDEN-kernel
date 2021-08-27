#include "userdriver/gpu/common/CLKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define SIGMOID(input, output, channel, height, width) \
    int globalID0 = get_global_id(0); \
    int globalID1 = get_global_id(1); \
    int globalID2 = get_global_id(2) * 8; \
    int wh = width * height; \
    if (globalID1 < channel && globalID2 < wh) { \
        int base = globalID0 * channel * wh + globalID1 * wh + globalID2; \
        if (globalID2 + 8 <= wh) { \
            vstore8( \
                (DATA_T8)(1.0f) / ((DATA_T8)(1.0f) + exp(-vload8(0, input + base))), 0, output + base); \
        } else { \
            int num = wh - globalID2; \
            if (num == 1) { \
                output[base] = (DATA_T)(1.0f) / ((DATA_T)(1.0f) + exp(-input[base])); \
            } else if (num == 2) { \
                vstore2((DATA_T2)(1.0f) / ((DATA_T2)(1.0f) + exp(-vload2(0, input + base))), \
                        0, \
                        output + base); \
            } else if (num == 3) { \
                vstore3((DATA_T3)(1.0f) / ((DATA_T3)(1.0f) + exp(-vload3(0, input + base))), \
                        0, \
                        output + base); \
            } else if (num == 4) { \
                vstore4((DATA_T4)(1.0f) / ((DATA_T4)(1.0f) + exp(-vload4(0, input + base))), \
                        0, \
                        output + base); \
            } else if (num == 5) { \
                vstore4((DATA_T4)(1.0f) / ((DATA_T4)(1.0f) + exp(-vload4(0, input + base))), \
                        0, \
                        output + base); \
                output[base + 4] = (DATA_T)(1.0f) / ((DATA_T)(1.0f) + exp(-input[base + 4])); \
            } else if (num == 6) { \
                vstore4((DATA_T4)(1.0f) / ((DATA_T4)(1.0f) + exp(-vload4(0, input + base))), \
                        0, \
                        output + base); \
                vstore2((DATA_T2)(1.0f) / ((DATA_T2)(1.0f) + exp(-vload2(0, input + base + 4))), \
                        0, \
                        output + base + 4); \
            } else if (num == 7) { \
                vstore4((DATA_T4)(1.0f) / ((DATA_T4)(1.0f) + exp(-vload4(0, input + base))), \
                        0, \
                        output + base); \
                vstore3((DATA_T3)(1.0f) / ((DATA_T3)(1.0f) + exp(-vload3(0, input + base + 4))), \
                        0, \
                        output + base + 4); \
            } \
        } \
    }

#define DATA_T half
#define DATA_T2 half2
#define DATA_T3 half3
#define DATA_T4 half4
#define DATA_T8 half8
ADD_SINGLE_KERNEL(sigmoid_FP16, (__global half *input,
                                           __global half *output,
                                           unsigned int channel,
                                           unsigned int height,
                                           unsigned int width) {
    SIGMOID(input, output, channel, height, width)
})
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T

#define DATA_T float
#define DATA_T2 float2
#define DATA_T3 float3
#define DATA_T4 float4
#define DATA_T8 float8
ADD_SINGLE_KERNEL(sigmoid_FP32, (__global float *input,
                                           __global float *output,
                                           unsigned int channel,
                                           unsigned int height,
                                           unsigned int width) {
    SIGMOID(input, output, channel, height, width)
})
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T

}  // namespace gpu
}  // namespace ud
}  // namespace enn
