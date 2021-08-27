#include "userdriver/gpu/common/CLKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define DILATION_INIT(output, zero_point) \
        output[get_global_id(0)] = zero_point;

#define DILATION(input, output, extent_height, extent_width, dilation_height, dilation_width) \
        int kernel_h = get_global_size(1); \
        int kernel_w = get_global_size(2); \
        int index_dilation = get_global_id(0) * extent_height * extent_width + \
                             get_global_id(1) * dilation_height * extent_width; \
        int index_weight = get_global_id(0) * kernel_h * kernel_w + \
                           get_global_id(1) * kernel_w + get_global_id(2); \
        output[index_dilation + get_global_id(2) * dilation_width] = input[index_weight];

#define DATA_T half
ADD_SINGLE_KERNEL(dilation_FP16, (__global DATA_T *input,
                                __global DATA_T *output,
                                unsigned int extent_height,
                                unsigned int extent_width,
                                unsigned int dilation_height,
                                unsigned int dilation_width) {
    DILATION(input, output, extent_height, extent_width, dilation_height, dilation_width)
})
#undef DATA_T

#define DATA_T float
ADD_SINGLE_KERNEL(dilation_FP32, (__global DATA_T *input,
                                __global DATA_T *output,
                                unsigned int extent_height,
                                unsigned int extent_width,
                                unsigned int dilation_height,
                                unsigned int dilation_width) {
    DILATION(input, output, extent_height, extent_width, dilation_height, dilation_width)
})
#undef DATA_T

/********  QUANTIZED KERNELS ********/
#define DATA_T uchar
#define DATA_T2 uchar2
#define DATA_T3 uchar3
#define DATA_T4 uchar4
#define DATA_T8 uchar8
#define DATA_T16 uchar16

ADD_SINGLE_KERNEL(dilation_init_INT8, (__global DATA_T *output,
                                     unsigned int zero_point) {
    DILATION_INIT(output, zero_point)
})

ADD_SINGLE_KERNEL(dilation_INT8, (__global DATA_T *input,
                                __global DATA_T *output,
                                unsigned int extent_height,
                                unsigned int extent_width,
                                unsigned int dilation_height,
                                unsigned int dilation_width) {
    DILATION(input, output, extent_height, extent_width, dilation_height, dilation_width)
})

#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // uchar


#define DATA_T char
#define DATA_T2 char2
#define DATA_T3 char3
#define DATA_T4 char4
#define DATA_T8 char8
#define DATA_T16 char16

ADD_SINGLE_KERNEL(SIGNEDdilation_init_INT8, (__global DATA_T *output,
                                     unsigned int zero_point) {
    DILATION_INIT(output, zero_point)
})

ADD_SINGLE_KERNEL(SIGNEDdilation_INT8, (__global DATA_T *input,
                                __global DATA_T *output,
                                unsigned int extent_height,
                                unsigned int extent_width,
                                unsigned int dilation_height,
                                unsigned int dilation_width) {
    DILATION(input, output, extent_height, extent_width, dilation_height, dilation_width)
})

#undef DATA_T16
#undef DATA_T8
#undef DATA_T4
#undef DATA_T3
#undef DATA_T2
#undef DATA_T  // char

}  // namespace gpu
}  // namespace ud
}  // namespace enn
