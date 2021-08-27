#include "userdriver/gpu/common/CLKernels.hpp"
#include "userdriver/gpu/operators/cl_kernels/CLCommonKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define TF_SLICE(input, output, W, bn, bc, bh, bw, sn, sc, sh, sw) \
    ushort idxn = get_global_id(0); \
    if (idxn < bn || idxn >= bn + sn) \
        return; \
    ushort idxc = get_global_id(1); \
    if (idxc < bc || idxc >= bc + sc) \
        return; \
    half tmp = get_global_id(2) / W; \
    ushort idxh = floor(tmp); \
    if (idxh < bh || idxh >= bh + sh) \
        return; \
    ushort idxw = get_global_id(2) - idxh * W; \
    if (idxw < bw || idxw >= bw + sw) \
        return; \
    uint out_idx = \
        (idxn - bn) * sw * sh * sc + (idxc - bc) * sw * sh + (idxh - bh) * sw + (idxw - bw); \
    uint gid = idxn * get_global_size(1) * get_global_size(2) + idxc * get_global_size(2) + \
                get_global_id(2); \
    output[out_idx] = input[gid];

#define DATA_T half
ADD_SINGLE_KERNEL(tf_slice_FP16, (__global const DATA_T *input,
                                __global DATA_T *output,
                                uint W, uint bn, uint bc, uint bh, uint bw,
                                uint sn, uint sc, uint sh, uint sw) {
    TF_SLICE(input, output, W, bn, bc, bh, bw, sn, sc, sh, sw)
})
#undef DATA_T

#define DATA_T float
ADD_SINGLE_KERNEL(tf_slice_FP32, (__global const DATA_T *input,
                                __global DATA_T *output,
                                uint W, uint bn, uint bc, uint bh, uint bw,
                                uint sn, uint sc, uint sh, uint sw) {
    TF_SLICE(input, output, W, bn, bc, bh, bw, sn, sc, sh, sw)
})
#undef DATA_T

#define DATA_T uchar
ADD_SINGLE_KERNEL(tf_slice_INT8, (__global const DATA_T *input,
                                __global DATA_T *output,
                                uint W, uint bn, uint bc, uint bh, uint bw,
                                uint sn, uint sc, uint sh, uint sw) {
    TF_SLICE(input, output, W, bn, bc, bh, bw, sn, sc, sh, sw)
})
#undef DATA_T

#define DATA_T char
ADD_SINGLE_KERNEL(SIGNEDtf_slice_INT8, (__global const DATA_T *input,
                                __global DATA_T *output,
                                uint W, uint bn, uint bc, uint bh, uint bw,
                                uint sn, uint sc, uint sh, uint sw) {
    TF_SLICE(input, output, W, bn, bc, bh, bw, sn, sc, sh, sw)
})
#undef DATA_T

}  // namespace gpu
}  // namespace ud
}  // namespace enn
