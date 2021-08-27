#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

ADD_SINGLE_KERNEL(squeeze_INT8, (__global const unsigned char *input,
                                           __global unsigned char *output) {
    int globalID0 = get_global_id(0);
    output[globalID0] = input[globalID0];
})

ADD_SINGLE_KERNEL(squeeze_FP32, (__global const float *input, __global float *output) {
    int globalID0 = get_global_id(0);
    output[globalID0] = input[globalID0];
})

ADD_SINGLE_KERNEL(squeeze_FP16, (__global const half *input, __global half *output) {
    int globalID0 = get_global_id(0);
    output[globalID0] = input[globalID0];
})

}  // namespace gpu
}  // namespace ud
}  // namespace enn
