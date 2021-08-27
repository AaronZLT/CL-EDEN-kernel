#include "userdriver/gpu/operators/cl_optimized_impls/CLDepthwiseDeconvolution.hpp"

namespace enn {
namespace ud {
namespace gpu {

CLDepthwiseDeconvolution::CLDepthwiseDeconvolution(const std::shared_ptr<CLRuntime> runtime,
                                                   const PrecisionType &precision,
                                                   const Dim4 &input_dim,
                                                   const Dim4 &output_dim,
                                                   const std::shared_ptr<ITensor> filter,
                                                   const std::shared_ptr<ITensor> bias,
                                                   const Pad4 &padding,
                                                   const Dim2 &stride,
                                                   const uint32_t &group,
                                                   const ActivationInfo &activate_info) {
    DEBUG_PRINT("CLDepthwiseDeconvolution is created");
    runtime_ = runtime;
    runtime_->resetIntraBuffer();
    pad_ = padding;
    stride_ = stride;
    group_ = group;
    filter_ = filter;
    bias_ = bias;
    precision_ = precision;
    Dim4 kernel = filter_->getDim();
    activation_info_ = activate_info;

    Status state = runtime_->setKernel(&kernel_gemm_, "gemm_depthwise", precision_);
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "gemm_depthwise setKernel failure\n");

    if (activation_info_.isEnabled()) {
        if (activation_info_.activation() == ActivationInfo::ActivationType::RELU) {
            state = runtime_->setKernel(&kernel_convert_, "RELUcol2img_1x8_opt", precision_);
        } else {
            state = runtime_->setKernel(&kernel_convert_, "col2img_1x8_opt", precision_);
        }
    } else {
        state = runtime_->setKernel(&kernel_convert_, "col2img_1x8_opt", precision_);
    }
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "col2img_1x8_opt setKernel failure\n");

    // convert buffer
    uint32_t output_convert_size = input_dim.n * input_dim.h * input_dim.w * output_dim.c * kernel.h * kernel.w;
    Dim4 output_convert_buffer_dim = {output_convert_size, 1, 1, 1};

    output_convert_buffer_ = std::make_shared<CLTensor>(
        runtime_, precision_, DataType::FLOAT, output_convert_buffer_dim, DataOrder::NCHW, 1.0, 0, BufferType::INTRA_SHARED);
}

Status CLDepthwiseDeconvolution::execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) {
    DEBUG_PRINT("CLDepthwiseDeconvolution::execute() is called");
    auto input_tensor = std::static_pointer_cast<CLTensor>(input);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output);
    auto filter_tensor = std::static_pointer_cast<CLTensor>(filter_);
    auto bias_tensor = std::static_pointer_cast<CLTensor>(bias_);
    auto input_data = input_tensor->getDataPtr();
    auto output_data = output_tensor->getDataPtr();
    auto filter_data = filter_tensor->getDataPtr();
    auto bias_data = bias_tensor->getDataPtr();
    auto input_dim = input_tensor->getDim();
    auto output_dim = output_tensor->getDim();
    uint32_t filter_h = filter_tensor->getDim().h;
    uint32_t filter_w = filter_tensor->getDim().w;
    uint32_t stride_h = stride_.h;
    uint32_t stride_w = stride_.w;
    // FIXME: need to support 4-way padding
    uint32_t pad_h = pad_.t;
    uint32_t pad_w = pad_.l;
    Status state = Status::FAILURE;

    size_t global_gemm[3] = {0, 0, 0};
    size_t local_gemm[3] = {1, 64, 1};
    global_gemm[0] = input_dim.n * group_;
    global_gemm[1] = alignTo(input_dim.h * input_dim.w, local_gemm[1]);
    global_gemm[2] = alignTo(ceil(filter_h * filter_w / 16.0), local_gemm[2]);

    state = runtime_->setKernelArg(kernel_gemm_.get(),
                                   input_data,
                                   filter_data,
                                   output_convert_buffer_->getDataPtr(),
                                   group_,
                                   input_dim.h,
                                   input_dim.w,
                                   filter_h,
                                   filter_w);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setArg failure\n");
    state = runtime_->enqueueKernel(kernel_gemm_.get(), 3, global_gemm, local_gemm);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "kernel_gemm_ execute kernel failure\n");

    // convert output
    size_t global_convert_output[3] = {0, 0, 0};
    size_t local_convert_output[3] = {1, 4, 32};
    global_convert_output[0] = output_dim.n * output_dim.c;
    global_convert_output[1] = alignTo(output_dim.h, 4);
    global_convert_output[2] = alignTo(output_dim.w, 32);
    state = runtime_->setKernelArg(kernel_convert_.get(),
                                   output_convert_buffer_->getDataPtr(),
                                   bias_data,
                                   output_data,
                                   output_dim.h,
                                   output_dim.w,
                                   output_dim.c,
                                   filter_h,
                                   filter_w,
                                   pad_h,
                                   pad_w,
                                   stride_h,
                                   stride_w,
                                   input_dim.h,
                                   input_dim.w);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "kernel_convert_ setKernelArg failure\n");
    state = runtime_->enqueueKernel(kernel_convert_.get(), 3, global_convert_output, local_convert_output);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "kernel_convert_ execute kernel failure\n");

    return Status::SUCCESS;
}

Status CLDepthwiseDeconvolution::release() { return Status::SUCCESS; }

}  // namespace gpu
}  // namespace ud
}  // namespace enn
