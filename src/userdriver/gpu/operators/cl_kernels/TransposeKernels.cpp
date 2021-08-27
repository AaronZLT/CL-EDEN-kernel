#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define DATA_T uchar
#define DATA_T8 uchar8
ADD_SINGLE_KERNEL(transpose_INT8, (__global const unsigned char *input,
                                             __global unsigned char *output,
                                             unsigned int perm_n_nchw,
                                             unsigned int perm_c_nchw,
                                             unsigned int perm_h_nchw,
                                             unsigned int perm_w_nchw,
                                             unsigned int inputBatch,
                                             unsigned int inputChannel,
                                             unsigned int inputHeight,
                                             unsigned int inputWidth,
                                             unsigned int outputChannel,
                                             unsigned int outputHeight,
                                             unsigned int outputWidth) {
    TRANSPOSE(input, output, perm_n_nchw, perm_c_nchw, perm_h_nchw, perm_w_nchw, inputBatch, inputChannel, \
              inputHeight, inputWidth, outputChannel, outputHeight, outputWidth)
})
#undef DATA_T
#undef DATA_T8

#define DATA_T float
#define DATA_T8 float8
ADD_SINGLE_KERNEL(transpose_FP32, (__global const float *input,
                                             __global float *output,
                                             unsigned int perm_n_nchw,
                                             unsigned int perm_c_nchw,
                                             unsigned int perm_h_nchw,
                                             unsigned int perm_w_nchw,
                                             unsigned int inputBatch,
                                             unsigned int inputChannel,
                                             unsigned int inputHeight,
                                             unsigned int inputWidth,
                                             unsigned int outputChannel,
                                             unsigned int outputHeight,
                                             unsigned int outputWidth) {
    TRANSPOSE(input, output, perm_n_nchw, perm_c_nchw, perm_h_nchw, perm_w_nchw, inputBatch, inputChannel, \
              inputHeight, inputWidth, outputChannel, outputHeight, outputWidth)
})
#undef DATA_T
#undef DATA_T8

#define DATA_T half
#define DATA_T8 half8
ADD_SINGLE_KERNEL(transpose_FP16, (__global const half *input,
                                             __global half *output,
                                             unsigned int perm_n_nchw,
                                             unsigned int perm_c_nchw,
                                             unsigned int perm_h_nchw,
                                             unsigned int perm_w_nchw,
                                             unsigned int inputBatch,
                                             unsigned int inputChannel,
                                             unsigned int inputHeight,
                                             unsigned int inputWidth,
                                             unsigned int outputChannel,
                                             unsigned int outputHeight,
                                             unsigned int outputWidth) {
    TRANSPOSE(input, output, perm_n_nchw, perm_c_nchw, perm_h_nchw, perm_w_nchw, inputBatch, inputChannel, \
              inputHeight, inputWidth, outputChannel, outputHeight, outputWidth)
})
#undef DATA_T
#undef DATA_T8

}  // namespace gpu
}  // namespace ud
}  // namespace enn
