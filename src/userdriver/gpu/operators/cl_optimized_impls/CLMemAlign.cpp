#include "userdriver/gpu/operators/cl_optimized_impls/CLMemAlign.hpp"

namespace enn {
namespace ud {
namespace gpu {

CLMemAlign::CLMemAlign(const std::shared_ptr<CLRuntime> runtime) {
    runtime_ = runtime;
    DEBUG_PRINT("CLMemAlign is created");
}

CLMemAlign::~CLMemAlign() {}

Status CLMemAlign::execute(const cl_mem input,
                           cl_mem output,
                           const PrecisionType &precision,
                           const int &src_total_count,
                           const int &src_unit_count,
                           const int &dst_total_count,
                           const int &dst_align_count,
                           const int &group) {
    int lines = src_total_count / src_unit_count / group;
    int outLines = dst_total_count / dst_align_count / group;
    if (precision == PrecisionType::FP32) {
        int srcOff = 0, desOff = 0;
        for (int j = 0; j < group; j++) {
            for (int i = 0; i < lines; i++) {
                srcOff = i * src_unit_count + j * src_unit_count * lines;
                desOff = i * dst_align_count + j * dst_align_count * outLines;
                int err = clEnqueueCopyBuffer(runtime_->getQueue(),
                                              input,
                                              output,
                                              sizeof(cl_float) * (size_t)srcOff,
                                              sizeof(cl_float) * (size_t)desOff,
                                              sizeof(cl_float) * (size_t)src_unit_count,
                                              0,
                                              NULL,
                                              NULL);
                CHECK_EXPR_RETURN_FAILURE(CL_SUCCESS == err, "clEnqueueCopyBuffer fail: %d", err);
            }
        }
    } else if (precision == PrecisionType::FP16) {
        int srcOff = 0, desOff = 0;
        for (int j = 0; j < group; j++) {
            for (int i = 0; i < lines; i++) {
                srcOff = i * src_unit_count + j * src_unit_count * lines;
                desOff = i * dst_align_count + j * dst_align_count * outLines;
                int err = clEnqueueCopyBuffer(runtime_->getQueue(),
                                              input,
                                              output,
                                              sizeof(cl_half) * (size_t)srcOff,
                                              sizeof(cl_half) * (size_t)desOff,
                                              sizeof(cl_half) * (size_t)src_unit_count,
                                              0,
                                              NULL,
                                              NULL);

                CHECK_EXPR_RETURN_FAILURE(CL_SUCCESS == err, "clEnqueueCopyBuffer fail: %d", err);
            }
        }
    }
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
