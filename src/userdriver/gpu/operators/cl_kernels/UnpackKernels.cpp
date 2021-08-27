/*
 * Copyright (C) 2018 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define UNPACK(input, output, copy_size, outputs_count, src_offset) \
    int src_idx = get_global_id(0) * outputs_count * copy_size + src_offset; \
    int dst_idx = get_global_id(0) * copy_size; \
    for (int i = 0; i != copy_size; ++i) { \
        output[dst_idx + i] = input[src_idx + i]; \
    }

ADD_SINGLE_KERNEL(unpack_FP16, (__global const half *input,
                              __global half *output,
                              int copy_size,
                              int outputs_count,
                              int src_offset) {
    UNPACK(input, output, copy_size, outputs_count, src_offset)
}
)

ADD_SINGLE_KERNEL(unpack_FP32, (__global const float *input,
                              __global float *output,
                              int copy_size,
                              int outputs_count,
                              int src_offset) {
    UNPACK(input, output, copy_size, outputs_count, src_offset)
}
)

}  // namespace gpu
}  // namespace ud
}  // namespace enn
