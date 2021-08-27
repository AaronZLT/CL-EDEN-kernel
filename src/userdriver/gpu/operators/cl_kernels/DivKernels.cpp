#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define DIV(input1, input2, output) \
        int globalID0 = get_global_id(0); \
        int globalID1 = get_global_id(1); \
        int globalID2 = get_global_id(2); \
        int index = globalID0 * get_global_size(1) * get_global_size(2) + \
                    globalID1 * get_global_size(2) + globalID2; \
        output[index] = input1[index] / input2[index];

#define DATA_T float
ADD_SINGLE_KERNEL(div_FP32, (__global const DATA_T *input1,
                           __global const DATA_T *input2,
                           __global DATA_T *output) {
    DIV(input1, input2, output)
})
#undef DATA_T

#define DATA_T half
ADD_SINGLE_KERNEL(div_FP16, (__global const DATA_T *input1,
                           __global const DATA_T *input2,
                           __global DATA_T *output) {
    DIV(input1, input2, output)
})
#undef DATA_T

#define DATA_T int
ADD_SINGLE_KERNEL(INT32div_FP32, (__global const DATA_T *input1,
                           __global const DATA_T *input2,
                           __global DATA_T *output) {
    DIV(input1, input2, output)
})

ADD_SINGLE_KERNEL(INT32div_FP16, (__global const DATA_T *input1,
                           __global const DATA_T *input2,
                           __global DATA_T *output) {
    DIV(input1, input2, output)
})
#undef DATA_T

}  // namespace gpu
}  // namespace ud
}  // namespace enn
