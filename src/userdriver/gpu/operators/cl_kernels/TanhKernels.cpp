#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define TANH(input, output, channel, height, width) \
    int globalID0 = get_global_id(0); \
    int globalID1 = get_global_id(1); \
    int globalID2 = get_global_id(2) * 8; \
    int wh = width * height; \
    if (globalID1 < channel && globalID2 < wh) { \
        int base = globalID0 * channel * wh + globalID1 * wh + globalID2; \
        if (globalID2 + 8 <= wh) { \
            vstore8(tanh(vload8(0, input + base)), 0, output + base); \
        } else { \
            int num = wh - globalID2; \
            if (num == 1) { \
                output[base] = tanh(input[base]); \
            } else if (num == 2) { \
                vstore2(tanh(vload2(0, input + base)), 0, output + base); \
            } else if (num == 3) { \
                vstore3(tanh(vload3(0, input + base)), 0, output + base); \
            } else if (num == 4) { \
                vstore4(tanh(vload4(0, input + base)), 0, output + base); \
            } else if (num == 5) { \
                vstore4(tanh(vload4(0, input + base)), 0, output + base); \
                output[base + 4] = tanh(input[base + 4]); \
            } else if (num == 6) { \
                vstore4(tanh(vload4(0, input + base)), 0, output + base); \
                vstore2(tanh(vload2(0, input + base + 4)), 0, output + base + 4); \
            } else if (num == 7) { \
                vstore4(tanh(vload4(0, input + base)), 0, output + base); \
                vstore3(tanh(vload3(0, input + base + 4)), 0, output + base + 4); \
            } \
        } \
    }

#define Q_TANH(input, output, batch, channel, height, width, qmin, qmax, inscale, inzero, outscale, outzero) \
    int globalID0 = get_global_id(0); \
    int globalID1 = get_global_id(1); \
    int globalID2 = get_global_id(2); \
    int wh = width * height; \
    int base = globalID0 * channel * wh + globalID1 * wh + globalID2; \
    int size = batch * channel * height * width; \
    if (base < size) { \
        output[base] = fmax( \
            qmin, \
            fmin(qmax, round(tanh((input[base] - inzero) * inscale) / outscale) + outzero)); \
    }

#define DATA_T uchar
ADD_SINGLE_KERNEL(tanh_INT8, (__global DATA_T *input,
                            __global DATA_T *output,
                            unsigned int batch,
                            unsigned int channel,
                            unsigned int height,
                            unsigned int width,
                            int qmin,
                            int qmax,
                            float inscale,
                            int inzero,
                            float outscale,
                            int outzero) {
    Q_TANH(input, output, batch, channel, height, width, qmin, qmax, inscale, inzero, outscale, outzero)
})
#undef DATA_T

#define DATA_T char
ADD_SINGLE_KERNEL(SIGNEDtanh_INT8, (__global DATA_T *input,
                            __global DATA_T *output,
                            unsigned int batch,
                            unsigned int channel,
                            unsigned int height,
                            unsigned int width,
                            int qmin,
                            int qmax,
                            float inscale,
                            int inzero,
                            float outscale,
                            int outzero) {
    Q_TANH(input, output, batch, channel, height, width, qmin, qmax, inscale, inzero, outscale, outzero)
})
#undef DATA_T

#define DATA_T half
ADD_SINGLE_KERNEL(tanh_FP16, (__global DATA_T *input,
                            __global DATA_T *output,
                            unsigned int channel,
                            unsigned int height,
                            unsigned int width) {
    TANH(input, output, channel, height, width)
})
#undef DATA_T

#define DATA_T float
ADD_SINGLE_KERNEL(tanh_FP32, (__global DATA_T *input,
                            __global DATA_T *output,
                            unsigned int channel,
                            unsigned int height,
                            unsigned int width) {
    TANH(input, output, channel, height, width)
})
#undef DATA_T

}  // namespace gpu
}  // namespace ud
}  // namespace enn
