#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define GATHER(input, indices, output, inner_size, indices_count, axis_size) \
        int globalID0 = get_global_id(0); \
        int globalID1 = get_global_id(1); \
        int out_offset = (globalID0 * get_global_size(1) + globalID1) * inner_size; \
        int in_offset = (globalID0 * axis_size + indices[globalID1]) * inner_size; \
        for (int i = 0; i != inner_size; ++i) { \
            output[out_offset + i] = input[in_offset + i]; \
        }

#define DATA_T float
ADD_SINGLE_KERNEL(gather_FP32, (__global const DATA_T *input,
                              __global const int *indices,
                              __global DATA_T *output,
                              int inner_size,
                              int indices_count,
                              int axis_size) {
    GATHER(input, indices, output, inner_size, indices_count, axis_size)
})
#undef DATA_T

#define DATA_T half
ADD_SINGLE_KERNEL(gather_FP16, (__global const DATA_T *input,
                              __global const int *indices,
                              __global DATA_T *output,
                              int inner_size,
                              int indices_count,
                              int axis_size) {
    GATHER(input, indices, output, inner_size, indices_count, axis_size)
})
#undef DATA_T

#define DATA_T int
ADD_SINGLE_KERNEL(INT32gather_FP32, (__global const DATA_T *input,
                              __global const int *indices,
                              __global DATA_T *output,
                              int inner_size,
                              int indices_count,
                              int axis_size) {
    GATHER(input, indices, output, inner_size, indices_count, axis_size)
})

ADD_SINGLE_KERNEL(INT32gather_FP16, (__global const DATA_T *input,
                              __global const int *indices,
                              __global DATA_T *output,
                              int inner_size,
                              int indices_count,
                              int axis_size) {
    GATHER(input, indices, output, inner_size, indices_count, axis_size)
})
#undef DATA_T

#define DATA_T uchar
ADD_SINGLE_KERNEL(gather_INT8, (__global const DATA_T *input,
                              __global const int *indices,
                              __global DATA_T *output,
                              int inner_size,
                              int indices_count,
                              int axis_size) {
    GATHER(input, indices, output, inner_size, indices_count, axis_size)
})
#undef DATA_T

#define DATA_T char
ADD_SINGLE_KERNEL(SIGNEDgather_INT8, (__global const DATA_T *input,
                              __global const int *indices,
                              __global DATA_T *output,
                              int inner_size,
                              int indices_count,
                              int axis_size) {
    GATHER(input, indices, output, inner_size, indices_count, axis_size)
})
#undef DATA_T

}  // namespace gpu
}  // namespace ud
}  // namespace enn
