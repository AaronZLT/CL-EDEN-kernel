#include "userdriver/gpu/common/CLKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define QUANTIZATION(input, output, inverse_scale, zero_point, channel, height, width) \
    int globalID0 = get_global_id(0); \
    int globalID1 = get_global_id(1); \
    int globalID2 = get_global_id(2) * 8; \
    int wh = width * height; \
    if (globalID1 < channel && globalID2 < wh) { \
        int base = globalID0 * channel * wh + globalID1 * wh + globalID2; \
        if (globalID2 + 8 <= wh) { \
            vstore8(CONVERT_TO_DATA_T8(round(mad(vload8(0, input + base), inverse_scale, zero_point))),\
                    0, output + base); \
        } else { \
            int num = wh - globalID2; \
            if (num >= 4) { \
                vstore4(CONVERT_TO_DATA_T4(round(mad(vload4(0, input + base), inverse_scale, zero_point))),\
                        0, output + base); \
                num -= 4; \
                base += 4; \
            } \
            if (num == 1) { \
                output[base] = CONVERT_TO_DATA_T(round(mad(input[base], inverse_scale, zero_point))); \
            } else if (num == 2) { \
                vstore2(CONVERT_TO_DATA_T2(round(mad(vload2(0, input + base), inverse_scale, zero_point))), \
                        0, output + base); \
            } else if (num == 3) { \
                vstore3(CONVERT_TO_DATA_T3(round(mad(vload3(0, input + base), inverse_scale, zero_point ))), \
                        0, output + base); \
            } \
        } \
    }

#define QUANTIZATION_FP16(input, output, inverse_scale, zero_point, channel, height, width) \
    int globalID0 = get_global_id(0); \
    int globalID1 = get_global_id(1); \
    int globalID2 = get_global_id(2) * 8; \
    int wh = width * height; \
    if (globalID1 < channel && globalID2 < wh) { \
        int base = globalID0 * channel * wh + globalID1 * wh + globalID2; \
        if (globalID2 + 8 <= wh) { \
            vstore8(CONVERT_TO_DATA_T8(round(mad(vload8(0, input + base), (half8)inverse_scale, (float)zero_point))),\
                    0, output + base); \
        } else { \
            int num = wh - globalID2; \
            if (num >= 4) { \
                vstore4(CONVERT_TO_DATA_T4(round(mad(vload4(0, input + base), (half4)inverse_scale, (float)zero_point))),\
                        0, output + base); \
                num -= 4; \
                base += 4; \
            } \
            if (num == 1) { \
                output[base] = CONVERT_TO_DATA_T(round(mad(input[base], (half)inverse_scale, zero_point))); \
            } else if (num == 2) { \
                vstore2(CONVERT_TO_DATA_T2(round(mad(vload2(0, input + base), (half2)inverse_scale, (float)zero_point))), \
                        0, output + base); \
            } else if (num == 3) { \
                vstore3(CONVERT_TO_DATA_T3(round(mad(vload3(0, input + base), (half3)inverse_scale, (float)zero_point ))), \
                        0, output + base); \
            } \
        } \
    }

#ifndef CONVERT_TO_DATA_T(x)
#define CONVERT_TO_DATA_T(x)  convert_uchar_sat(x)
#endif
#ifndef CONVERT_TO_DATA_T2(x)
#define CONVERT_TO_DATA_T2(x) convert_uchar2_sat(x)
#endif
#ifndef CONVERT_TO_DATA_T3(x)
#define CONVERT_TO_DATA_T3(x) convert_uchar3_sat(x)
#endif
#ifndef CONVERT_TO_DATA_T4(x)
#define CONVERT_TO_DATA_T4(x) convert_uchar4_sat(x)
#endif
#ifndef CONVERT_TO_DATA_T8(x)
#define CONVERT_TO_DATA_T8(x) convert_uchar8_sat(x)
#endif

ADD_SINGLE_KERNEL(quantization_FP32, (__global float *input,
                                      __global unsigned char *output,
                                      float inverse_scale,
                                      int zero_point,
                                      unsigned int channel,
                                      unsigned int height,
                                      unsigned int width) {
          QUANTIZATION(input, output, inverse_scale, zero_point, channel, height, width)
})

ADD_SINGLE_KERNEL(quantization_FP16, (__global half *input,
                                      __global unsigned char *output,
                                      float inverse_scale,
                                      int zero_point,
                                      unsigned int channel,
                                      unsigned int height,
                                      unsigned int width) {
         QUANTIZATION_FP16(input, output, inverse_scale, zero_point, channel, height, width)
})

#undef CONVERT_TO_DATA_T
#undef CONVERT_TO_DATA_T2
#undef CONVERT_TO_DATA_T3
#undef CONVERT_TO_DATA_T4
#undef CONVERT_TO_DATA_T8

#define CONVERT_TO_DATA_T(x)  convert_char_sat(x)
#define CONVERT_TO_DATA_T2(x) convert_char2_sat(x)
#define CONVERT_TO_DATA_T3(x) convert_char3_sat(x)
#define CONVERT_TO_DATA_T4(x) convert_char4_sat(x)
#define CONVERT_TO_DATA_T8(x) convert_char8_sat(x)

ADD_SINGLE_KERNEL(quantization_signed_FP32, (__global float *input,
                                             __global char *output,
                                             float inverse_scale,
                                             int zero_point,
                                             unsigned int channel,
                                             unsigned int height,
                                             unsigned int width) {
          QUANTIZATION(input, output, inverse_scale, zero_point, channel, height, width)
})

ADD_SINGLE_KERNEL(quantization_signed_FP16, (__global half *input,
                                             __global char *output,
                                             float inverse_scale,
                                             int zero_point,
                                             unsigned int channel,
                                             unsigned int height,
                                             unsigned int width) {
          QUANTIZATION_FP16(input, output, inverse_scale, zero_point, channel, height, width)
})

#undef CONVERT_TO_DATA_T
#undef CONVERT_TO_DATA_T2
#undef CONVERT_TO_DATA_T3
#undef CONVERT_TO_DATA_T4
#undef CONVERT_TO_DATA_T8

#define CONVERT_TO_DATA_T(x)  convert_ushort_sat(x)
#define CONVERT_TO_DATA_T2(x) convert_ushort2_sat(x)
#define CONVERT_TO_DATA_T3(x) convert_ushort3_sat(x)
#define CONVERT_TO_DATA_T4(x) convert_ushort4_sat(x)
#define CONVERT_TO_DATA_T8(x) convert_ushort8_sat(x)

ADD_SINGLE_KERNEL(quantization_ushort_FP32, (__global float *input,
                                             __global unsigned short *output,
                                             float inverse_scale,
                                             int zero_point,
                                             unsigned int channel,
                                             unsigned int height,
                                             unsigned int width) {
          QUANTIZATION(input, output, inverse_scale, zero_point, channel, height, width)
})

ADD_SINGLE_KERNEL(quantization_ushort_FP16, (__global half *input,
                                                __global unsigned short *output,
                                                float inverse_scale,
                                                int zero_point,
                                                unsigned int channel,
                                                unsigned int height,
                                                unsigned int width) {
       QUANTIZATION_FP16(input, output, inverse_scale, zero_point, channel, height, width)
})

#undef CONVERT_TO_DATA_T
#undef CONVERT_TO_DATA_T2
#undef CONVERT_TO_DATA_T3
#undef CONVERT_TO_DATA_T4
#undef CONVERT_TO_DATA_T8

#define CONVERT_TO_DATA_T(x)  convert_short_sat(x)
#define CONVERT_TO_DATA_T2(x) convert_short2_sat(x)
#define CONVERT_TO_DATA_T3(x) convert_short3_sat(x)
#define CONVERT_TO_DATA_T4(x) convert_short4_sat(x)
#define CONVERT_TO_DATA_T8(x) convert_short8_sat(x)

ADD_SINGLE_KERNEL(quantization_short_FP32, (__global float *input,
                                            __global short *output,
                                            float inverse_scale,
                                            int zero_point,
                                            unsigned int channel,
                                            unsigned int height,
                                            unsigned int width) {

          QUANTIZATION(input, output, inverse_scale, zero_point, channel, height, width)
})

ADD_SINGLE_KERNEL(quantization_short_FP16, (__global half *input,
                                               __global short *output,
                                               float inverse_scale,
                                               int zero_point,
                                               unsigned int channel,
                                               unsigned int height,
                                               unsigned int width) {
         QUANTIZATION_FP16(input, output, inverse_scale, zero_point, channel, height, width)
})

#undef CONVERT_TO_DATA_T
#undef CONVERT_TO_DATA_T2
#undef CONVERT_TO_DATA_T3
#undef CONVERT_TO_DATA_T4
#undef CONVERT_TO_DATA_T8

}  // namespace gpu
}  // namespace ud
}  // namespace enn
