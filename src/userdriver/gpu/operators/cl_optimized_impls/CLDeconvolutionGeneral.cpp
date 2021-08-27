#include "userdriver/gpu/operators/cl_optimized_impls/CLDeconvolutionGeneral.hpp"

namespace enn {
namespace ud {
namespace gpu {

constexpr int MAX_GROUP_SIZE = 128;

CLDeconvolutionGeneral::CLDeconvolutionGeneral(const std::shared_ptr<CLRuntime> runtime,
                                               const PrecisionType &precision,
                                               const Dim4 &input_dim,
                                               const Dim4 &output_dim,
                                               const std::shared_ptr<ITensor> filter,
                                               const std::shared_ptr<ITensor> bias,
                                               const Pad4 &padding,
                                               const Dim2 &stride,
                                               const uint32_t &group) {
    DEBUG_PRINT("CLDeconvolutionGeneral is created");
    runtime_ = runtime;
    pad_ = padding;
    stride_ = stride;
    group_ = group;
    filter_ = filter;
    bias_ = bias;
    precision_ = precision;
    Dim4 kernel = filter_->getDim();
    Status state = runtime_->setKernel(&kernel_trans_, "matrixTrans", precision_);
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "matrixTrans setKernel failure\n");
    state = runtime_->setKernel(&kernel_gemm_, "convBackGemmRXR", precision_);
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "convBackGemmRXR setKernel failure\n");
    state = runtime_->setKernel(&kernel_convert_, "convertBottomDiff2", precision_);
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "convertBottomDiff2 setKernel failure\n");

    uint32_t filter_align_width = alignTo(input_dim.c / group_, 8);
    uint32_t filter_align_height = alignTo(kernel.h * kernel.w * output_dim.c, 8);
    uint32_t filter_count = filter_align_width * filter_align_height;

    Dim4 filter_buffer_dim = {filter_count, 1, 1, 1};
    filter_buffer_ =
        std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, filter_buffer_dim);

    uint32_t transinput_align_width = alignTo(input_dim.c / group_, 8) * group_;
    uint32_t size_trans_input =
        input_dim.n * 1 * (input_dim.h * input_dim.w) * transinput_align_width;

    Dim4 input_trans_buffer_dim = {size_trans_input, 1, 1, 1};
    input_trans_buffer_ =
        std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, input_trans_buffer_dim);

    uint32_t output_convert_size =
        input_dim.n * input_dim.h * input_dim.w * output_dim.c * kernel.h * kernel.w;

    Dim4 output_convert_buffer_dim = {output_convert_size, 1, 1, 1};
    output_convert_buffer_ = std::make_shared<CLTensor>(
        runtime_, precision_, DataType::FLOAT, output_convert_buffer_dim);
}

Status CLDeconvolutionGeneral::execute(const std::shared_ptr<ITensor> input,
                                       std::shared_ptr<ITensor> output) {
    DEBUG_PRINT("CLDeconvolutionGeneral::execute() is called");
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

    // filter trans
    uint32_t filter_align_width = alignTo(input_dim.c / group_, 8);
    uint32_t trans_width = input_dim.c / group_;
    uint32_t trans_height = output_dim.c / group_ * filter_h * filter_w;
    uint32_t filter_batch = 0;
    size_t global_weight_trans[3] = {0, 0, 0};
    size_t local_weight_trans[3] = {0, 0, 0};
    global_weight_trans[0] = group_;
    global_weight_trans[1] = trans_height;
    global_weight_trans[2] = trans_width;
    local_weight_trans[2] = findMaxFactor(global_weight_trans[2], MAX_GROUP_SIZE);
    local_weight_trans[1] =
        findMaxFactor(global_weight_trans[1], MAX_GROUP_SIZE / local_weight_trans[2]);
    local_weight_trans[0] = findMaxFactor(
        global_weight_trans[0], MAX_GROUP_SIZE / (local_weight_trans[2] * local_weight_trans[1]));
    Status state = runtime_->setKernelArg(kernel_trans_.get(),
                                          filter_data,
                                          filter_buffer_->getDataPtr(),
                                          filter_align_width,
                                          filter_batch,
                                          group_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "kernel_trans_ setKernelArg failure\n");
    state =
        runtime_->enqueueKernel(kernel_trans_.get(), 3, global_weight_trans, local_weight_trans);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "kernel_trans_ execute kernel failure\n");

    // input trans
    trans_width = input_dim.c / group_;
    trans_height = input_dim.h * input_dim.w;
    uint32_t trans_width_align = alignTo(input_dim.c / group_, 8);
    size_t global_input_trans[3] = {0, 0, 0};
    size_t local_input_trans[3] = {0, 0, 0};
    global_input_trans[0] = group_;
    global_input_trans[1] = trans_height;
    global_input_trans[2] = trans_width;
    local_input_trans[2] = findMaxFactor(global_input_trans[2], MAX_GROUP_SIZE);
    local_input_trans[1] =
        findMaxFactor(global_input_trans[1], MAX_GROUP_SIZE / local_input_trans[2]);
    local_input_trans[0] = findMaxFactor(
        global_input_trans[0], MAX_GROUP_SIZE / (local_input_trans[2] * local_input_trans[1]));
    for (uint32_t i = 0; i < input_dim.n; i++) {
        state = runtime_->setKernelArg(kernel_trans_.get(),
                                       input_data,
                                       input_trans_buffer_->getDataPtr(),
                                       trans_width_align,
                                       i,
                                       group_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "kernel_trans_ setKernelArg failure\n");
        state =
            runtime_->enqueueKernel(kernel_trans_.get(), 3, global_input_trans, local_input_trans);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state,
                                  "kernel_trans_ execute kernel failure\n");
    }

    // matrix multiplication
    uint32_t col = alignTo(input_dim.c / group_, 8);
    uint32_t row_filter = filter_h * filter_w * output_dim.c / group_;
    uint32_t row_input = input_dim.h * input_dim.w;
    size_t global_gemm[3] = {0, 0, 0};
    size_t local_gemm[3] = {1, 4, 32};
    global_gemm[0] = group_;
    global_gemm[1] = alignTo(ceil(row_filter / 8.0), 4);  // Each thread process 8 elements
    global_gemm[2] = alignTo(row_input, 32);
    for (uint32_t i = 0; i < input_dim.n; i++) {
        state = runtime_->setKernelArg(kernel_gemm_.get(),
                                       filter_buffer_->getDataPtr(),
                                       input_trans_buffer_->getDataPtr(),
                                       output_convert_buffer_->getDataPtr(),
                                       row_filter,
                                       col,
                                       row_input,
                                       i,
                                       group_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "kernel_gemm_ setKernelArg failure\n");
        state = runtime_->enqueueKernel(kernel_gemm_.get(), 3, global_gemm, local_gemm);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state,
                                  "kernel_gemm_ execute kernel failure\n");
    }

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
    state = runtime_->enqueueKernel(
        kernel_convert_.get(), 3, global_convert_output, local_convert_output);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "kernel_convert_ execute kernel failure\n");

    return Status::SUCCESS;
}

Status CLDeconvolutionGeneral::release() {
    DEBUG_PRINT("CLDeconvolutionGeneral::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
