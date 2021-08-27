#include "userdriver/gpu/operators/cl_optimized_impls/CLDeconvolutionMakalu.hpp"

namespace enn {
namespace ud {
namespace gpu {

CLDeconvolutionMakalu::CLDeconvolutionMakalu(const std::shared_ptr<CLRuntime> runtime,
                                             const PrecisionType &precision,
                                             const Dim4 &input_dim,
                                             const Dim4 &output_dim,
                                             const std::shared_ptr<ITensor> filter,
                                             const std::shared_ptr<ITensor> bias,
                                             const Pad4 &padding,
                                             const Dim2 &stride,
                                             const uint32_t &group,
                                             const ActivationInfo &activate_info,
                                             const bool &weights_as_input,
                                             const bool &androidNN) {
    DEBUG_PRINT("CLDeconvolutionMakalu is created");
    runtime_ = runtime;
    runtime_->resetIntraBuffer();
    filter_ = std::static_pointer_cast<CLTensor>(filter);
    bias_ = std::static_pointer_cast<CLTensor>(bias);
    pad_ = padding;
    stride_ = stride;
    group_ = group;
    precision_ = precision;
    activation_info_ = activate_info;
    weights_as_input_ = weights_as_input;
    androidNN_ = androidNN;

    Dim4 filter_dim = filter_->getDim();
    if (androidNN_) {
        kernel_.h = filter_dim.c;
        kernel_.w = filter_dim.h;
    } else {
        kernel_.h = filter_dim.h;
        kernel_.w = filter_dim.w;
    }

    Status state = runtime_->setKernel(&kernel_trans_, "pure_matrix_transpose", precision_);
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "pure_matrix_transpose setKernel failure\n");
    state = runtime_->setKernel(&kernel_weight_shared_trans_, "align_weight_4_row_1_col", precision_);
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "align_weight_4_row_1_col setKernel failure\n");
    state = runtime_->setKernel(&kernel_input_coalesced_trans_, "convert_input_4_thread_8_1x1", precision_);
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "convert_input_4_thread_8_1x1 setKernel failure\n");

    if (isTwoTimesDeconv()) {
        if (activation_info_.isEnabled()) {
            if (activation_info_.activation() == ActivationInfo::ActivationType::RELU) {
                state = runtime_->setKernel(&kernel_gemm_, "RELUgemm_makalu_deconv_2_times", precision_);
                CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "RELUgemm_makalu_deconv_2_times setKernel failure\n");
            } else {
                state = runtime_->setKernel(&kernel_gemm_, "gemm_makalu_deconv_2_times", precision_);
                CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "gemm_makalu_deconv_2_times setKernel failure\n");
            }
        } else {
            state = runtime_->setKernel(&kernel_gemm_, "gemm_makalu_deconv_2_times", precision_);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "gemm_makalu_deconv_2_times setKernel failure\n");
        }
    } else if (kernel_.h * kernel_.w % 4 == 0) {
        state = runtime_->setKernel(&kernel_gemm_, "gemm_makalu_deconv_opt", precision_);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "gemm_makalu_deconv_opt setKernel failure\n");

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
    } else {
        state = runtime_->setKernel(&kernel_gemm_, "gemm_makalu_deconv", precision_);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "gemmMakalu_deconv setKernel failure\n");

        if (activation_info_.isEnabled()) {
            if (activation_info_.activation() == ActivationInfo::ActivationType::RELU) {
                state = runtime_->setKernel(&kernel_convert_, "RELUcol2img_1x8", precision_);
            } else {
                state = runtime_->setKernel(&kernel_convert_, "col2img_1x8", precision_);
            }
        } else {
            state = runtime_->setKernel(&kernel_convert_, "col2img_1x8", precision_);
        }
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "col2img_1x8 setKernel failure\n");
    }

    transposed_filter_width_ = input_dim.c;
    transposed_filter_height_ = kernel_.h * kernel_.w * output_dim.c;
    Dim4 filter_buffer_dim = {1, 1, transposed_filter_height_, transposed_filter_width_};
    filter_buffer_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, filter_buffer_dim);

    transformed_filter_width_ = alignTo(input_dim.c, GEMM1XX_ALIGN_SIZE_4);
    transformed_filter_height_ = alignTo(kernel_.h * kernel_.w * output_dim.c, GEMM1XX_SHARING_WEIGHT_8);
    Dim4 transformed_filter_dim = {1, 1, transformed_filter_height_, transformed_filter_width_};
    weight_trans_buffer_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, transformed_filter_dim);

    uint32_t transinput_align_width = alignTo(input_dim.c, GEMM1XX_ALIGN_SIZE_4);
    uint32_t transinput_align_height = alignTo(input_dim.h * input_dim.w, GEMM1XX_COALESCING_INPUT_4_THREAD_12);
    Dim4 input_trans_buffer_dim = {input_dim.n, 1, transinput_align_height, transinput_align_width};

    input_trans_buffer_ = std::make_shared<CLTensor>(runtime_,
                                                     precision_,
                                                     DataType::FLOAT,
                                                     input_trans_buffer_dim,
                                                     DataOrder::NCHW,
                                                     1.0,
                                                     0,
                                                     BufferType::INTRA_SHARED);

    if (!isTwoTimesDeconv()) {
        uint32_t output_convert_size = input_dim.n * input_dim.h * input_dim.w * output_dim.c * kernel_.h * kernel_.w;
        Dim4 output_convert_buffer_dim = {output_convert_size, 1, 1, 1};
        output_convert_buffer_ = std::make_shared<CLTensor>(runtime_,
                                                            precision_,
                                                            DataType::FLOAT,
                                                            output_convert_buffer_dim,
                                                            DataOrder::NCHW,
                                                            1.0,
                                                            0,
                                                            BufferType::INTRA_SHARED);
    }

    if (weights_as_input_ == false) {
        // In EDEN path, deconv's kernel shape is {Cin, Cout, k, k}, and it should be transposed to
        // {Cout, k, k, Cin} for optimization.
        // But in Android NNAPI path, its original shape is {Cout, k, k, Cin}, and the transposition
        // is unnecessary.
        if (!androidNN_) {
            // filter transpose
            auto filter_data = filter_->getDataPtr();
            size_t global_weight_trans[2] = {0, 0};
            size_t local_weight_trans[2] = {0, 0};

            global_weight_trans[0] = transposed_filter_height_;
            global_weight_trans[1] = transposed_filter_width_;
            local_weight_trans[1] = findMaxFactor(global_weight_trans[1], MAX_LOCAL_SIZE);
            local_weight_trans[0] = findMaxFactor(global_weight_trans[0], MAX_LOCAL_SIZE / local_weight_trans[1]);

            // kernel is transposed from {Cin, Cout, K, K} to {Cout, K, K, Cin}
            state = runtime_->setKernelArg(kernel_trans_.get(), filter_data, filter_buffer_->getDataPtr());
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "kernel_trans_ setKernelArg failure\n");
            state = runtime_->enqueueKernel(kernel_trans_.get(), 2, global_weight_trans, local_weight_trans);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "kernel_trans_ execute kernel failure\n");
        }

        auto filter_data = androidNN_ ? filter_->getDataPtr() : filter_buffer_->getDataPtr();
        // filter transform
        size_t globalAlignWeight[2] = {0, 0};
        globalAlignWeight[0] = transformed_filter_height_ / 8;
        globalAlignWeight[1] = transformed_filter_width_ * 8;
        state = runtime_->setKernelArg(kernel_weight_shared_trans_.get(),
                                       filter_data,
                                       weight_trans_buffer_->getDataPtr(),
                                       transposed_filter_height_,
                                       transposed_filter_width_,
                                       transformed_filter_width_ * 8);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "kernel_weight_shared_trans_ setArg kernel failure\n");
        state = runtime_->enqueueKernel(kernel_weight_shared_trans_.get(), 2, globalAlignWeight, NULL);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "kernel_weight_shared_trans_ execute kernel failure\n");
    }
}

bool CLDeconvolutionMakalu::isTwoTimesDeconv() {
    if (pad_.b == 0 && pad_.t == 0 && pad_.r == 0 && pad_.l == 0 && stride_.h == kernel_.h &&
        stride_.w == kernel_.w && stride_.h == 2 && stride_.w == 2) {
        return true;
    } else {
        return false;
    }
}

Status CLDeconvolutionMakalu::execute(const std::shared_ptr<ITensor> input,
                                      std::shared_ptr<ITensor> output) {
    DEBUG_PRINT("CLDeconvolutionMakalu::execute() is called");
    Status state = Status::FAILURE;
    if (weights_as_input_ == true && androidNN_) {
        // filter transform
        size_t globalAlignWeight[2] = {0, 0};
        globalAlignWeight[0] = transformed_filter_height_ / 8;
        globalAlignWeight[1] = transformed_filter_width_ * 8;

        state = runtime_->setKernelArg(kernel_weight_shared_trans_.get(),
                                       filter_->getDataPtr(),
                                       weight_trans_buffer_->getDataPtr(),
                                       transposed_filter_height_,
                                       transposed_filter_width_,
                                       transformed_filter_width_ * 8);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "kernel_weight_shared_trans_ setArg kernel failure\n");
        state = runtime_->enqueueKernel(kernel_weight_shared_trans_.get(), 2, globalAlignWeight, NULL);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "kernel_weight_shared_trans_ execute kernel failure\n");
    }

    auto input_tensor = std::static_pointer_cast<CLTensor>(input);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output);
    auto bias_tensor = std::static_pointer_cast<CLTensor>(bias_);
    auto input_data = input_tensor->getDataPtr();
    auto output_data = output_tensor->getDataPtr();
    auto bias_data = bias_tensor->getDataPtr();

    auto input_dim = input_tensor->getDim();
    auto output_dim = output_tensor->getDim();
    uint32_t filter_h = kernel_.h;
    uint32_t filter_w = kernel_.w;
    uint32_t stride_h = stride_.h;
    uint32_t stride_w = stride_.w;

    // FIXME: need to support 4-way padding
    uint32_t pad_h = pad_.t;
    uint32_t pad_w = pad_.l;

    // input trans
    size_t global_input_trans[3];
    size_t local_input_trans[3] = {24, 4, 1};

    if (ceil(input_dim.h * input_dim.w / 8.0) < 8) {
        local_input_trans[0] = 8;
    }

    global_input_trans[0] = alignTo(ceil(input_dim.h * input_dim.w / 8.0), local_input_trans[0]);
    global_input_trans[1] = alignTo(input_dim.c, local_input_trans[1]);
    global_input_trans[2] = input_dim.n;

    uint32_t transinput_align_height = alignTo(input_dim.h * input_dim.w, GEMM1XX_COALESCING_INPUT_4_THREAD_12);
    uint32_t transinput_align_width = alignTo(input_dim.c, GEMM1XX_ALIGN_SIZE_4);

    state = runtime_->setKernelArg(kernel_input_coalesced_trans_.get(),
                                   input_data,
                                   input_trans_buffer_->getDataPtr(),
                                   input_dim.h,
                                   input_dim.w,
                                   input_dim.c,
                                   transinput_align_height,
                                   transinput_align_width,
                                   GEMM1XX_COALESCING_INPUT_4_THREAD_12);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "set kernel arg failure\n");
    state = runtime_->enqueueKernel(kernel_input_coalesced_trans_.get(), 3, global_input_trans, local_input_trans);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "enqueueKernel failure\n");

    // matrix multiplication
    uint32_t col = alignTo(input_dim.c / group_, 4);
    uint32_t row_filter = filter_h * filter_w * output_dim.c / group_;
    uint32_t row_input = input_dim.h * input_dim.w;
    size_t global_gemm[3] = {0, 0, 0};
    size_t local_gemm[3] = {1, 2, 12};
    global_gemm[0] = input_dim.n;
    global_gemm[1] = alignTo(ceil(static_cast<double>(row_filter) / 4), local_gemm[1]);

    int factor = ceil(static_cast<double>(row_input) / 4);
    if (factor > 384) {
        local_gemm[2] = 128;
    } else if (factor > 96) {
        local_gemm[2] = 96;
    } else if (factor > 48) {
        local_gemm[2] = 48;
    } else if (factor > 24) {
        local_gemm[2] = 24;
    } else {
        local_gemm[2] = 12;
    }
    global_gemm[2] = alignTo(factor, local_gemm[2]);

    if (isTwoTimesDeconv()) {
        state = runtime_->setKernelArg(kernel_gemm_.get(),
                                       input_trans_buffer_->getDataPtr(),
                                       weight_trans_buffer_->getDataPtr(),
                                       bias_data,
                                       output_data,
                                       col,
                                       row_filter,
                                       input_dim.w,
                                       row_input,
                                       input_trans_buffer_->getDim().h *
                                           input_trans_buffer_->getDim().w);
    } else if (filter_h * filter_w % 4 == 0) {
        state = runtime_->setKernelArg(kernel_gemm_.get(),
                                       input_trans_buffer_->getDataPtr(),
                                       weight_trans_buffer_->getDataPtr(),
                                       output_convert_buffer_->getDataPtr(),
                                       col,
                                       row_filter,
                                       row_input,
                                       input_trans_buffer_->getDim().h *
                                           input_trans_buffer_->getDim().w,
                                       filter_h * filter_w);
    } else {
        state = runtime_->setKernelArg(kernel_gemm_.get(),
                                       input_trans_buffer_->getDataPtr(),
                                       weight_trans_buffer_->getDataPtr(),
                                       output_convert_buffer_->getDataPtr(),
                                       col,
                                       row_filter,
                                       row_input,
                                       input_trans_buffer_->getDim().h *
                                           input_trans_buffer_->getDim().w);
    }
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setArg failure\n");
    state = runtime_->enqueueKernel(kernel_gemm_.get(), 3, global_gemm, local_gemm);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "kernel_gemm_ execute kernel failure\n");

    // convert output
    if (!isTwoTimesDeconv()) {
        size_t global_convert_output[3] = {0, 0, 0};
        size_t local_convert_output[3] = {0, 0, 0};
        if (filter_h * filter_w % 4 == 0) {
            local_convert_output[0] = 1;
            local_convert_output[1] = 8;
            local_convert_output[2] = 48;
        } else {
            local_convert_output[0] = 1;
            local_convert_output[1] = 4;
            local_convert_output[2] = 32;
        }
        global_convert_output[0] = output_dim.n * output_dim.c;
        global_convert_output[1] = alignTo(output_dim.h, local_convert_output[1]);
        global_convert_output[2] = alignTo(output_dim.w, local_convert_output[2]);
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
    }

    return Status::SUCCESS;
}

Status CLDeconvolutionMakalu::release() {
    DEBUG_PRINT("CLDeconvolutionMakalu::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
