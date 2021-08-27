#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLPadConvert.hpp"

namespace enn {
namespace ud {
namespace gpu {

CLPadConvert::CLPadConvert(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision), need_pad_flag_(false), pad_(nullptr) {
    ENN_DBG_PRINT("CLPadConvert is created");
    padding_ = {0, 0, 0, 0};
    stride_ = {0, 0};
    kernel_ = {0, 0};
    group_ = 0;
    top_dim_ = {0, 0, 0, 0};
    pad_kernel_ = nullptr;
    convert_blocked_kernel_ = nullptr;
    convert_optimized_kernel_ = nullptr;
    gemm1xx_align_convert1x1_kernel_ = nullptr;
    gemm1xx_align_convert_withpad_Kernel_ = nullptr;
    gemm1xx_align_convert1x1_makalu_kernel_ = nullptr;
    gemm1xx_align_convert_withpad_makalu_Kernel_ = nullptr;
    gemm1xx_align_convert_withpad_makalu_Kernel_5x5_ = nullptr;
    gemm1xx_align_convert1x1_makalu_2_2_kernel_ = nullptr;
    gemm1xx_align_convert_withpad_makalu_2_2_Kernel_ = nullptr;
    quantized_gemm1xX_align_convert_withpad_kernel_ = nullptr;
    pad_opt_kernel_ = nullptr;
    pad_direct_aligned_kernel_ = nullptr;
    pad_convert_type_ = PadConvertType::NOSET;
}

Status CLPadConvert::initialize(const Dim4 &input_dim,
                                const Pad4 &pad,
                                const Dim2 &kernel,
                                const Dim2 &stride,
                                const uint32_t &group,
                                const Dim4 &top_dim,
                                const bool need_pad_buffer,
                                const PadConvertType &pad_convert_type,
                                const bool use_makalu_2_2) {
    ENN_DBG_PRINT("CLPadConvert::initialize() is called");
    padding_ = pad;
    kernel_ = kernel;
    stride_ = stride;
    group_ = group;
    top_dim_ = top_dim;
    pad_convert_type_ = pad_convert_type;

    const int bottom_width = input_dim.w;
    const int kernel_height = kernel.h;
    const int kernel_width = kernel.w;
    const int stride_height = stride.h;
    const int stride_width = stride.w;
    const int pad_bottom = pad.b;
    const int pad_top = pad.t;
    const int pad_left = pad.l;
    const int pad_right = pad.r;

    Status state = Status::FAILURE;
    if (pad_convert_type == PadConvertType::GEMM) {
        Dim4 pad_dim;
        if (precision_ == PrecisionType::UINT8 || precision_ == PrecisionType::INT8) {
            if (padding_.t != 0 || padding_.b != 0 || padding_.l != 0 || padding_.r != 0) {
                need_pad_flag_ = true;
                if (precision_ == PrecisionType::UINT8) {
                    state = runtime_->setKernel(&pad_opt_kernel_, "pad_opt", precision_);
                    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel pad_t1b1l1r1 failure\n");
                } else if (precision_ == PrecisionType::INT8) {
                    state = runtime_->setKernel(&pad_opt_kernel_, "SIGNEDpad_opt", precision_);
                    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel pad_t1b1l1r1 failure\n");
                }

                pad_dim.n = input_dim.n;
                pad_dim.c = input_dim.c;
                pad_dim.h = input_dim.h + padding_.t + padding_.b;
                pad_dim.w = input_dim.w + padding_.l + padding_.r;
                if (need_pad_buffer) {
                    pad_ = std::make_shared<CLTensor>(runtime_,
                                                      precision_,
                                                      precision_ == PrecisionType ::INT8 ? DataType ::INT8 : DataType::UINT8,
                                                      pad_dim);
                }
            }
        } else {
            if (padding_.t != 0 || padding_.b != 0 || padding_.l != 0 || padding_.r != 0) {
                need_pad_flag_ = true;
                if (padding_.t != padding_.b || padding_.l != padding_.r) {
                    state = runtime_->setKernel(&pad_kernel_, "pad4", precision_);
                    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
                } else {
                    state = runtime_->setKernel(&pad_kernel_, "pad", precision_);
                    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
                }
                pad_dim.n = input_dim.n;
                pad_dim.c = input_dim.c;
                pad_dim.h = input_dim.h + padding_.t + padding_.b;
                pad_dim.w = input_dim.w + padding_.l + padding_.r;
                if (need_pad_buffer) {
                    pad_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, pad_dim);
                }
            }
        }
        if (precision_ == PrecisionType::INT8) {
            if (group_ != 1) {
                state = runtime_->setKernel(&convert_blocked_kernel_, "SIGNEDconvertBlocked2", precision_);
                CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel failure\n");
            } else {
                state = runtime_->setKernel(&convert_optimized_kernel_, "SIGNEDconvert_optimized", precision_);
                CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel failure\n");
            }
        } else {
            if (group_ != 1) {
                state = runtime_->setKernel(&convert_blocked_kernel_, "convertBlocked2", precision_);
                CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel failure\n");
            } else {
                state = runtime_->setKernel(&convert_optimized_kernel_, "convert_optimized", precision_);
                CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel failure\n");
            }
        }
    } else if (pad_convert_type == PadConvertType::QuantizedDirect) {
        state = runtime_->setKernel(&pad_direct_aligned_kernel_, "pad_copy_align_c4", precision_);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel pad_t1b1l1r1 failure\n");
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel failure\n");
    } else if (pad_convert_type == PadConvertType::GEMM4x4MakaluNHWC1X1) {
        state =
            runtime_->setKernel(&gemm1xx_align_convert1x1_makalu_kernel_, "convert_input_4_thread_8_1x1_nhwc", precision_);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel failure\n");
    } else if (pad_convert_type == PadConvertType::GEMM4x4Makalu) {
        const int top_width = (bottom_width + pad_left + pad_right - kernel_width + stride_width) / stride_width;
        if (kernel_height == 1 && kernel_width == 1 && stride_height == 1 && stride_width == 1 && pad_bottom == 0 &&
            pad_top == 0 && pad_left == 0 && pad_right == 0) {
            state =
                runtime_->setKernel(&gemm1xx_align_convert1x1_makalu_kernel_, "convert_input_4_thread_8_1x1", precision_);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel failure\n");
        } else if (kernel_height == 5 && kernel_width == 5 && pad_left == 2 && pad_right == 2 && pad_bottom == 2 &&
                   pad_top == 2 && stride_height == 1 && stride_width == 1 && top_width % 16 == 0) {
            state = runtime_->setKernel(
                &gemm1xx_align_convert_withpad_makalu_Kernel_5x5_, "convert_input_4_thread_8_withpad_5x5", precision_);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel failure\n");
        } else {
            state = runtime_->setKernel(
                &gemm1xx_align_convert_withpad_makalu_Kernel_, "convert_input_4_thread_8_withpad", precision_);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel failure\n");
        }
    } else if (pad_convert_type == PadConvertType::GEMM1xX) {
        state = runtime_->setKernel(&gemm1xx_align_convert1x1_kernel_, "convert_2points_1x1", precision_);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel failure\n");
        state = runtime_->setKernel(&gemm1xx_align_convert_withpad_Kernel_, "convert_2points_withpad", precision_);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel failure\n");
    } else if (pad_convert_type == PadConvertType::GEMM1xXMakalu) {
        const int top_width = (bottom_width + pad_left + pad_right - kernel_width + stride_width) / stride_width;
        if (kernel_height == 1 && kernel_width == 1 && stride_height == 1 && stride_width == 1 && pad_bottom == 0 &&
            pad_top == 0 && pad_left == 0 && pad_right == 0) {
            state =
                runtime_->setKernel(&gemm1xx_align_convert1x1_makalu_kernel_, "convert_input_4_thread_8_1x1", precision_);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel failure\n");
            state = runtime_->setKernel(
                &gemm1xx_align_convert1x1_makalu_2_2_kernel_, "convert_input_2_thread_8_1x1", precision_);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel failure\n");
        } else if (kernel_height == 5 && kernel_width == 5 && pad_left == 2 && pad_right == 2 && pad_bottom == 2 &&
                   pad_top == 2 && stride_height == 1 && stride_width == 1 && top_width % 16 == 0) {
            state = runtime_->setKernel(
                &gemm1xx_align_convert_withpad_makalu_Kernel_5x5_, "convert_input_4_thread_8_withpad_5x5", precision_);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel failure\n");
        } else {
            if (use_makalu_2_2) {
                state = runtime_->setKernel(
                    &gemm1xx_align_convert_withpad_makalu_2_2_Kernel_, "convert_input_2_thread_8_withpad", precision_);
                CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel failure\n");
            }
            state = runtime_->setKernel(
                &gemm1xx_align_convert_withpad_makalu_Kernel_, "convert_input_4_thread_8_withpad", precision_);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel failure\n");
        }
    }

    return state;
}

Status CLPadConvert::execute(const std::shared_ptr<CLTensor> input, std::shared_ptr<CLTensor> output) {
    ENN_DBG_PRINT("CLPadConvert::execute() is called");
    if (precision_ == PrecisionType::UINT8 || precision_ == PrecisionType::INT8) {
        if (need_pad_flag_ == true) {
            Status state = quantizedPadRun(input, pad_, padding_, input->getZeroPoint());
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "quantizedPad run failure\n");
            state = quantizedImg2ColRun(input, output, pad_, padding_, kernel_, stride_, group_, top_dim_.h, top_dim_.w);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "quantizedim2col run failure\n");
            return Status::SUCCESS;
        } else {
            return quantizedImg2ColRun(input, output, input, padding_, kernel_, stride_, group_, top_dim_.h, top_dim_.w);
        }
    } else {
        if (need_pad_flag_ == true) {
            Status state = padRun(input, pad_, padding_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "pad run failure\n");
            state = img2ColRun(input, output, pad_, padding_, kernel_, stride_, group_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "pad run failure\n");
            return state;
        } else {
            return img2ColRun(input, output, input, padding_, kernel_, stride_, group_);
        }
    }
}

Status CLPadConvert::padRun(const std::shared_ptr<CLTensor> &input, std::shared_ptr<CLTensor> &output, const Pad4 &pad) {
    auto input_data = input->getDataPtr();
    auto output_data = output->getDataPtr();
    auto input_dim = input->getDim();
    auto output_dim = output->getDim();
    uint32_t input_width = input_dim.w;
    uint32_t input_height = input_dim.h;
    size_t global[3];
    size_t local[3] = {1, 1, 32};
    global[0] = input_dim.n;
    global[1] = input_dim.c;
    global[2] = alignTo(output_dim.h * output_dim.w, local[2]);
    Status state;
    if (padding_.t != padding_.b || padding_.l != padding_.r) {
        state = runtime_->setKernelArg(
            pad_kernel_.get(), input_data, output_data, pad.l, pad.r, pad.b, pad.t, input_width, input_height);
    } else {
        state = runtime_->setKernelArg(pad_kernel_.get(), input_data, output_data, pad.l, pad.t, input_width, input_height);
    }
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");
    state = runtime_->enqueueKernel(pad_kernel_.get(), (cl_uint)3, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
    ENN_DBG_PRINT("CLPadConvert is in padRun");
    return Status::SUCCESS;
}

Status CLPadConvert::quantizedPadRun(const std::shared_ptr<CLTensor> &input,
                                     std::shared_ptr<CLTensor> &output,
                                     const Pad4 &pad,
                                     int byte_zero) {
    ENN_DBG_PRINT("CLPadConvert is in quantizedPadRun");
    Status state = Status::FAILURE;
    auto input_data = input->getDataPtr();
    auto output_data = output->getDataPtr();
    auto input_dim = input->getDim();
    auto output_dim = output->getDim();
    int input_width = input_dim.w;
    int input_height = input_dim.h;
    int output_height = output_dim.h;
    int output_width = output_dim.w;

    size_t global[3];
    size_t local[3] = {1, 4, 64};
    global[0] = input_dim.n * input_dim.c;
    global[1] = alignTo(output_height, local[1]);
    global[2] = alignTo(ceil(output_width / 16.0), local[2]);
    state = runtime_->setKernelArg(pad_opt_kernel_.get(),
                                   input_data,
                                   output_data,
                                   byte_zero,
                                   pad.t,
                                   pad.l,
                                   input_height,
                                   input_width,
                                   output_height,
                                   output_width);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");
    state = runtime_->enqueueKernel(pad_opt_kernel_.get(), (cl_uint)3, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");

    return Status::SUCCESS;
}

Status CLPadConvert::quantizedDirectPadRun(const std::shared_ptr<CLTensor> &input,
                                           std::shared_ptr<CLTensor> &output,
                                           const Pad4 &pad,
                                           unsigned char byte_zero) {
    ENN_DBG_PRINT("CLPadConvert is in quantizedDirectPadRun");
    Status state = Status::FAILURE;
    auto input_data = input->getDataPtr();
    auto output_data = output->getDataPtr();
    auto input_dim = input->getDim();
    auto output_dim = output->getDim();
    int input_width = input_dim.w;
    int input_height = input_dim.h;
    int output_height = output_dim.h;
    int output_width = output_dim.w;
    int aligned_c = ceil(input_dim.c / 4.0);

    size_t global[3];
    size_t local[3] = {16, 1, 1};
    global[0] = alignTo(ceil(input_width / 4.0), local[0]);
    global[1] = alignTo(input_height, local[1]);
    global[2] = input_dim.n * aligned_c;
    state = runtime_->setKernelArg(pad_direct_aligned_kernel_.get(),
                                   input_data,
                                   output_data,
                                   byte_zero,
                                   pad.t,
                                   pad.l,
                                   input_dim.c,
                                   input_height,
                                   input_width,
                                   aligned_c,
                                   output_height,
                                   output_width * 4);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");
    state = runtime_->enqueueKernel(pad_direct_aligned_kernel_.get(), (cl_uint)3, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");

    return Status::SUCCESS;
}

Status CLPadConvert::img2ColRun(const std::shared_ptr<CLTensor> &input,
                                std::shared_ptr<CLTensor> &output,
                                const std::shared_ptr<CLTensor> &padded_input,
                                const Pad4 &pad,
                                const Dim2 &kernel,
                                const Dim2 &stride,
                                const unsigned int group) {
    ENN_DBG_PRINT("CLPadConvert is in img2ColRun");
    auto input_data = input->getDataPtr();
    auto output_data = output->getDataPtr();
    auto padded_data = padded_input->getDataPtr();
    auto input_dim = input->getDim();
    auto output_dim = output->getDim();
    auto padded_dim = padded_input->getDim();
    uint32_t kernel_height = kernel.h;
    uint32_t kernel_width = kernel.w;
    uint32_t stride_height = stride.h;
    uint32_t stride_width = stride.w;
    uint32_t pad_bottom = pad.b;
    uint32_t pad_top = pad.t;
    uint32_t pad_left = pad.l;
    uint32_t pad_right = pad.r;
    uint32_t top_height = (input_dim.h + pad_bottom + pad_top - kernel_height + stride_height) / stride_height;
    uint32_t top_width = (input_dim.w + pad_left + pad_right - kernel_width + stride_width) / stride_width;

    uint32_t add_pad_height = padded_dim.h;
    uint32_t add_pad_width = padded_dim.w;
    uint32_t add_pad_channel = padded_dim.c / group;
    uint32_t group_height = output_dim.h / group;
    uint32_t align_input_width = output_dim.w;
    size_t global_convert[3], local_convert[3] = {1, 1, 16};
    Status state;
    if (group == 1) {
        // the optimization of convert just supports group_size 1
        global_convert[0] = input_dim.n;
        global_convert[1] = alignTo(top_height * top_width, 1);
        // each thread processes elements size of kernel width
        global_convert[2] = alignTo(input_dim.c * kernel_height, 16);
        if (pad_bottom != 0 || pad_top != 0 || pad_left != 0 || pad_right != 0) {
            state = runtime_->setKernelArg(convert_optimized_kernel_.get(),
                                           padded_data,
                                           output_data,
                                           kernel_width,
                                           kernel_height,
                                           stride_width,
                                           stride_height,
                                           add_pad_height,
                                           add_pad_width,
                                           add_pad_channel,
                                           group_height,
                                           align_input_width,
                                           top_width,
                                           top_height);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "set kernel arg failure\n");
        } else {
            state = runtime_->setKernelArg(convert_optimized_kernel_.get(),
                                           input_data,
                                           output_data,
                                           kernel_width,
                                           kernel_height,
                                           stride_width,
                                           stride_height,
                                           add_pad_height,
                                           add_pad_width,
                                           add_pad_channel,
                                           group_height,
                                           align_input_width,
                                           top_width,
                                           top_height);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "set kernel arg failure\n");
        }
        state = runtime_->enqueueKernel(convert_optimized_kernel_.get(), 3, global_convert, local_convert);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "enqueue kernel failure\n");
        return state;
    } else {
        global_convert[0] = input_dim.n;
        global_convert[1] = alignTo(top_height * top_width, 16);
        global_convert[2] = alignTo(input_dim.c * kernel_height * kernel_width / group, 16);
        local_convert[1] = 16;
        if (pad_bottom != 0 || pad_top != 0 || pad_left != 0 || pad_right != 0) {
            state = runtime_->setKernelArg(convert_blocked_kernel_.get(),
                                           padded_data,
                                           output_data,
                                           kernel_width,
                                           kernel_height,
                                           stride_width,
                                           stride_height,
                                           add_pad_height,
                                           add_pad_width,
                                           add_pad_channel,
                                           group_height,
                                           align_input_width,
                                           0,
                                           top_width,
                                           group,
                                           top_height);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "set kernel arg failure\n");
        } else {
            state = runtime_->setKernelArg(convert_blocked_kernel_.get(),
                                           input_data,
                                           output_data,
                                           kernel_width,
                                           kernel_height,
                                           stride_width,
                                           stride_height,
                                           add_pad_height,
                                           add_pad_width,
                                           add_pad_channel,
                                           group_height,
                                           align_input_width,
                                           0,
                                           top_width,
                                           group,
                                           top_height);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "set kernel arg failure\n");
        }
        for (uint32_t i = 0; i < group; i++) {
            clSetKernelArg(convert_blocked_kernel_.get(), 11, sizeof(cl_uint), &(i));
            state = runtime_->enqueueKernel(convert_blocked_kernel_.get(), 3, global_convert, local_convert);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "enqueue kernel failure\n");
        }
        return state;
    }
}

Status CLPadConvert::quantizedImg2ColRun(const std::shared_ptr<CLTensor> &input,
                                         std::shared_ptr<CLTensor> &output,
                                         const std::shared_ptr<CLTensor> &padded_input,
                                         const Pad4 &pad,
                                         const Dim2 &kernel,
                                         const Dim2 &stride,
                                         const unsigned int group,
                                         int top_height,
                                         int top_width) {
    Status state;
    ENN_DBG_PRINT("CLPadConvert is in mg2ColRun");
    auto output_data = output->getDataPtr();
    auto padded_data = padded_input->getDataPtr();
    auto input_dim = input->getDim();
    auto output_dim = output->getDim();
    auto padded_dim = padded_input->getDim();
    uint32_t add_pad_height = padded_dim.h;
    uint32_t add_pad_width = padded_dim.w;
    uint32_t add_pad_channel = padded_dim.c / group;
    uint32_t group_height = output_dim.h / group;
    uint32_t align_input_width = output_dim.w;
    size_t global_convert[3], local_convert[3] = {1, 1, 16};
    if (group == 1) {
        // the optimization of convert just supports group size 1
        global_convert[0] = input_dim.n;
        global_convert[1] = alignTo(top_height * top_width, 1);
        // each thread processes elements size of kernel width
        global_convert[2] = alignTo(input_dim.c * kernel.h, 16);

        state = runtime_->setKernelArg(convert_optimized_kernel_.get(),
                                       padded_data,
                                       output_data,
                                       kernel.w,
                                       kernel.h,
                                       stride.w,
                                       stride.h,
                                       add_pad_height,
                                       add_pad_width,
                                       add_pad_channel,
                                       group_height,
                                       align_input_width,
                                       top_width,
                                       top_height);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "set kernel arg failure\n");
        state = runtime_->enqueueKernel(convert_optimized_kernel_.get(), 3, global_convert, local_convert);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "enqueue kernel failure\n");
        return state;
    } else {
        global_convert[0] = input_dim.n;
        global_convert[1] = alignTo(top_height * top_width, 16);
        global_convert[2] = alignTo(input_dim.c * kernel.h * kernel.w / group, 16);
        local_convert[0] = 1;
        local_convert[1] = 16;
        local_convert[2] = 16;

        state = runtime_->setKernelArg(convert_blocked_kernel_.get(),
                                       padded_data,
                                       output_data,
                                       kernel.w,
                                       kernel.h,
                                       stride.w,
                                       stride.h,
                                       add_pad_height,
                                       add_pad_width,
                                       add_pad_channel,
                                       group_height,
                                       align_input_width,
                                       0,
                                       top_width,
                                       group,
                                       top_height);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "set kernel arg failure\n");

        for (int32_t i = 0; i < group; i++) {
            clSetKernelArg(convert_blocked_kernel_.get(), 11, sizeof(cl_uint), &(i));
            state = runtime_->enqueueKernel(convert_blocked_kernel_.get(), 3, global_convert, local_convert);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "enqueue kernel failure\n");
        }
        return state;
    }
}

Status CLPadConvert::img2ColWithPad1xXRun(const std::shared_ptr<CLTensor> &input,
                                          std::shared_ptr<CLTensor> &output,
                                          const Pad4 &pad,
                                          const Dim2 &kernel,
                                          const Dim2 &stride,
                                          const unsigned int collapse_height) {
    Status state;
    auto input_data = input->getDataPtr();
    auto output_data = output->getDataPtr();
    auto input_dim = input->getDim();
    auto output_dim = output->getDim();
    int bottom_height = input_dim.h;
    int bottom_width = input_dim.w;
    int bottom_channel = input_dim.c;
    int group_height = output_dim.h;
    int align_input_width = output_dim.w;
    int kernel_height = kernel.h;
    int kernel_width = kernel.w;
    int stride_height = stride.h;
    int stride_width = stride.w;
    int pad_bottom = pad.b;
    int pad_top = pad.t;
    int pad_left = pad.l;
    int pad_right = pad.r;
    if (kernel_height == 1 && kernel_width == 1 && stride_height == 1 && stride_width == 1 && pad_bottom == 0 &&
        pad_top == 0 && pad_left == 0 && pad_right == 0) {
        size_t global_convert[3];
        size_t local_convert[3] = {1, 1, 16};
        global_convert[0] = input_dim.n;
        global_convert[1] = alignTo(bottom_channel, 1);
        global_convert[2] = alignTo(bottom_height * bottom_width, 16);
        state = runtime_->setKernelArg(gemm1xx_align_convert1x1_kernel_.get(),
                                       input_data,
                                       output_data,
                                       bottom_height,
                                       bottom_width,
                                       bottom_channel,
                                       group_height,
                                       align_input_width,
                                       collapse_height);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "set kernel arg failure\n");
        state = runtime_->enqueueKernel(gemm1xx_align_convert1x1_kernel_.get(), 3, global_convert, local_convert);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "enqueueKernel failure\n");
        return state;
    } else {
        int top_height = (bottom_height + pad_bottom + pad_top - kernel_height + stride_height) / stride_height;
        int top_width = (bottom_width + pad_left + pad_right - kernel_width + stride_width) / stride_width;
        size_t global_convert[3];
        size_t local_convert[3] = {1, 1, 16};
        global_convert[0] = input_dim.n;
        global_convert[1] = alignTo(bottom_channel, 1);
        global_convert[2] = alignTo(top_height * top_width, 16);
        state = runtime_->setKernelArg(gemm1xx_align_convert_withpad_Kernel_.get(),
                                       input_data,
                                       output_data,
                                       kernel_width,
                                       kernel_height,
                                       pad_left,
                                       pad_right,
                                       pad_bottom,
                                       pad_top,
                                       stride_width,
                                       stride_height,
                                       bottom_height,
                                       bottom_width,
                                       bottom_channel,
                                       group_height,
                                       align_input_width,
                                       top_width,
                                       top_height,
                                       collapse_height);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "set kernel arg failure\n");
        state = runtime_->enqueueKernel(gemm1xx_align_convert_withpad_Kernel_.get(), 3, global_convert, local_convert);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "enqueueKernel failure\n");
        return state;
    }
}

Status CLPadConvert::img2ColWithPad4x4MakaluRun(const std::shared_ptr<CLTensor> &input,
                                                std::shared_ptr<CLTensor> &output,
                                                const Pad4 &pad,
                                                const Dim2 &kernel,
                                                const Dim2 &stride,
                                                const unsigned int collapse_height,
                                                int top_hw_split_size,
                                                int pos_id) {
    Status state;
    auto input_data = input->getDataPtr();
    auto output_data = output->getDataPtr();
    auto input_dim = input->getDim();
    auto output_dim = output->getDim();
    int bottom_height = input_dim.h;
    int bottom_width = input_dim.w;
    int bottom_channel = input_dim.c;
    if (pad_convert_type_ == PadConvertType::GEMM4x4MakaluNHWC1X1) {
        bottom_height = input_dim.c;
        bottom_width = input_dim.h;
        bottom_channel = input_dim.w;
    }

    int group_height;

    group_height = output_dim.h;

    int align_input_width = output_dim.w;
    int kernel_height = kernel.h;
    int kernel_width = kernel.w;
    int stride_height = stride.h;
    int stride_width = stride.w;
    int pad_bottom = pad.b;
    int pad_top = pad.t;
    int pad_left = pad.l;
    int pad_right = pad.r;

    int top_height = (bottom_height + pad_bottom + pad_top - kernel_height + stride_height) / stride_height;
    int top_width = (bottom_width + pad_left + pad_right - kernel_width + stride_width) / stride_width;

    if (pad_convert_type_ == PadConvertType::GEMM4x4MakaluNHWC1X1) {
        size_t global_convert[3];
        size_t local_convert[3] = {1, 1, 24};
        local_convert[0] = 24;
        local_convert[1] = 4;
        local_convert[2] = 1;

        if (ceil(bottom_channel / 8.0) < 8) {
            local_convert[0] = 8;
        }

        global_convert[0] = alignTo(ceil(bottom_channel / 8.0), local_convert[0]);
        global_convert[1] = alignTo(ceil(bottom_height * bottom_width / 4.0), local_convert[1]);
        global_convert[2] = input_dim.n;
        state = runtime_->setKernelArg(gemm1xx_align_convert1x1_makalu_kernel_.get(),
                                       input_data,
                                       output_data,
                                       bottom_height,
                                       bottom_width,
                                       bottom_channel,
                                       group_height,
                                       align_input_width,
                                       collapse_height);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "set kernel arg failure\n");
        state = runtime_->enqueueKernel(gemm1xx_align_convert1x1_makalu_kernel_.get(), 3, global_convert, local_convert);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "enqueueKernel failure\n");

        return state;
    } else if (kernel_height == 1 && kernel_width == 1 && stride_height == 1 && stride_width == 1 && pad_bottom == 0 &&
               pad_top == 0 && pad_left == 0 && pad_right == 0) {
        size_t global_convert[3];
        size_t local_convert[3] = {1, 1, 24};

        local_convert[0] = 24;
        local_convert[1] = 4;
        local_convert[2] = 1;

        if (ceil(bottom_height * bottom_width / 8.0) < 8) {
            local_convert[0] = 8;
        }

        global_convert[0] = alignTo(ceil(bottom_height * bottom_width / 8.0), local_convert[0]);
        global_convert[1] = alignTo(bottom_channel, local_convert[1]);
        global_convert[2] = input_dim.n;
        state = runtime_->setKernelArg(gemm1xx_align_convert1x1_makalu_kernel_.get(),
                                       input_data,
                                       output_data,
                                       bottom_height,
                                       bottom_width,
                                       bottom_channel,
                                       group_height,
                                       align_input_width,
                                       collapse_height);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "set kernel arg failure\n");
        state = runtime_->enqueueKernel(gemm1xx_align_convert1x1_makalu_kernel_.get(), 3, global_convert, local_convert);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "enqueueKernel failure\n");

        return state;
    } else if (kernel_height == 5 && kernel_width == 5 && pad_left == 2 && pad_right == 2 && pad_bottom == 2 &&
               pad_top == 2 && stride_height == 1 && stride_width == 1 && top_width % 16 == 0) {
        size_t global_convert[3];
        size_t local_convert[3] = {1, 1, 24};
        local_convert[0] = 2;
        local_convert[1] = 128;
        local_convert[2] = 1;
        global_convert[0] = alignTo(top_hw_split_size / 16, local_convert[0]);
        global_convert[1] = alignTo(input->getDim().c * 25, local_convert[1]);
        int hwoffset = top_hw_split_size * pos_id;
        global_convert[2] = input->getDim().n;
        state = runtime_->setKernelArg(gemm1xx_align_convert_withpad_makalu_Kernel_5x5_.get(),
                                       input_data,
                                       output_data,
                                       bottom_height,
                                       bottom_width,
                                       bottom_channel,
                                       top_hw_split_size,
                                       align_input_width,
                                       collapse_height,
                                       hwoffset);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "set kernel arg failure\n");
        state = runtime_->enqueueKernel(
            gemm1xx_align_convert_withpad_makalu_Kernel_5x5_.get(), 3, global_convert, local_convert);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "enqueueKernel failure\n");
        return state;
    } else {
        size_t global_convert[3];
        size_t local_convert[3] = {1, 1, 24};

        local_convert[0] = 64;
        local_convert[1] = 1;
        local_convert[2] = 1;

        if (top_height * top_width < 64) {
            local_convert[0] = 24;
        }

        global_convert[0] = alignTo(top_hw_split_size, local_convert[0]);
        global_convert[1] = bottom_channel;
        global_convert[2] = input_dim.n;
        int hwoffset = top_hw_split_size * pos_id;
        state = runtime_->setKernelArg(gemm1xx_align_convert_withpad_makalu_Kernel_.get(),
                                       input_data,
                                       output_data,
                                       kernel_width,
                                       kernel_height,
                                       pad_left,
                                       pad_right,
                                       pad_bottom,
                                       pad_top,
                                       stride_width,
                                       stride_height,
                                       bottom_height,
                                       bottom_width,
                                       bottom_channel,
                                       group_height,
                                       align_input_width,
                                       top_width,
                                       top_height,
                                       collapse_height,
                                       hwoffset);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "set kernel arg failure\n");
        state =
            runtime_->enqueueKernel(gemm1xx_align_convert_withpad_makalu_Kernel_.get(), 3, global_convert, local_convert);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "enqueueKernel failure\n");

        return state;
    }
}

Status CLPadConvert::quantizedIm2colAlign1xXMakalu(const std::shared_ptr<CLTensor> &input,
                                                   std::shared_ptr<CLTensor> &output,
                                                   const Pad4 &padding,
                                                   int kernel_height,
                                                   int kernel_width,
                                                   int stride_height,
                                                   int stride_width,
                                                   int collapse_height,
                                                   int byte_zero,
                                                   int top_height,
                                                   int top_width) {
    bool isSigned = precision_ == PrecisionType ::INT8;
    Status state;
    if (isSigned) {
        if (runtime_->isValhall()) {
            state = runtime_->setKernel(&quantized_gemm1xX_align_convert_withpad_kernel_,
                                        "SIGNEDconvert_input_4_thread_8_withpad_valhall",
                                        precision_);
        } else {
            state = runtime_->setKernel(
                &quantized_gemm1xX_align_convert_withpad_kernel_, "SIGNEDconvert_input_4_thread_8_withpad", precision_);
        }
    } else {
        if (kernel_height == 1 && kernel_width == 1 && stride_height == 1 && stride_width == 1 && padding.b == 0 &&
            padding.t == 0 && padding.l == 0 && padding.r == 0) {
            state = runtime_->setKernel(
                &quantized_gemm1xX_align_convert_withpad_kernel_, "convert_input_4_thread_8_withpad_1x1", precision_);
        } else if (runtime_->isValhall() && kernel_height == 9 && kernel_width == 9 && padding.b == 4 && padding.t == 4 &&
                   padding.l == 4 && padding.r == 4 && stride_height == 1 && stride_width == 1 && top_width % 8 == 0) {
            state = runtime_->setKernel(
                &quantized_gemm1xX_align_convert_withpad_kernel_, "convert_input_4_thread_8_withpad_9x9", precision_);
        } else if (runtime_->isValhall() && kernel_height == 5 && kernel_width == 5 && padding.b == 2 && padding.t == 2 &&
                   padding.l == 2 && padding.r == 2 && stride_height == 1 && stride_width == 1 && top_width % 8 == 0) {
            state = runtime_->setKernel(
                &quantized_gemm1xX_align_convert_withpad_kernel_, "convert_input_4_thread_8_withpad_5x5", precision_);
        } else if (runtime_->isValhall() && kernel_height == 3 && kernel_width == 3 && padding.b == 1 && padding.t == 1 &&
                   padding.l == 1 && padding.r == 1 && stride_height == 1 && stride_width == 1 && top_width % 8 == 0) {
            state = runtime_->setKernel(
                &quantized_gemm1xX_align_convert_withpad_kernel_, "convert_input_4_thread_8_withpad_3x3_opt", precision_);
        } else if (runtime_->isValhall() && kernel_height == 3 && kernel_width == 3 && padding.b < 2 && padding.t < 2 &&
                   padding.l < 2 && padding.r < 2) {
            state = runtime_->setKernel(&quantized_gemm1xX_align_convert_withpad_kernel_,
                                        "convert_input_4_thread_8_withpad_3x3_valhall",
                                        precision_);
        } else if (kernel_height == 3 && kernel_width == 3 && padding.b < 2 && padding.t < 2 && padding.l < 2 &&
                   padding.r < 2) {
            state = runtime_->setKernel(
                &quantized_gemm1xX_align_convert_withpad_kernel_, "convert_input_4_thread_8_withpad_3x3", precision_);
        } else if (runtime_->isValhall()) {
            state = runtime_->setKernel(
                &quantized_gemm1xX_align_convert_withpad_kernel_, "convert_input_4_thread_8_withpad_valhall", precision_);
        } else {
            state = runtime_->setKernel(
                &quantized_gemm1xX_align_convert_withpad_kernel_, "convert_input_4_thread_8_withpad", precision_);
        }
    }
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel failure\n");

    auto bottom_buffer = input->getDataPtr();
    auto convert_buffer = output->getDataPtr();
    int bottom_height = input->getDim().h;
    int bottom_width = input->getDim().w;
    int bottom_channel = input->getDim().c;
    int group_height = output->getDim().h;
    int align_input_width = output->getDim().w;
    size_t global_convert[3], local_convert[3] = {72, 1, 1};
    if (top_height * top_width < 72) {
        local_convert[0] = 24;
    }
    if (runtime_->isValhall()) {
        local_convert[0] = 64;
    }
    global_convert[0] = alignTo(top_height * top_width, local_convert[0]);
    global_convert[1] = input->getDim().c;
    global_convert[2] = input->getDim().n;

    if (isSigned) {
        state = runtime_->setKernelArg(quantized_gemm1xX_align_convert_withpad_kernel_.get(),
                                       bottom_buffer,
                                       convert_buffer,
                                       kernel_width,
                                       kernel_height,
                                       padding.t,
                                       padding.l,
                                       stride_width,
                                       stride_height,
                                       bottom_height,
                                       bottom_width,
                                       bottom_channel,
                                       group_height,
                                       align_input_width,
                                       top_width,
                                       top_height,
                                       collapse_height,
                                       (char)byte_zero);
    } else {
        if (kernel_height == 1 && kernel_width == 1 && stride_height == 1 && stride_width == 1 && padding.b == 0 &&
            padding.t == 0 && padding.l == 0 && padding.r == 0) {
            local_convert[0] = 1;
            local_convert[1] = 1;
            local_convert[2] = 64;
            global_convert[0] = input->getDim().n;
            global_convert[1] = input->getDim().c;
            global_convert[2] = alignTo(ceil(top_height * top_width / 8.0), local_convert[2]);
            state = runtime_->setKernelArg(quantized_gemm1xX_align_convert_withpad_kernel_.get(),
                                           bottom_buffer,
                                           convert_buffer,
                                           stride_width,
                                           stride_height,
                                           bottom_height,
                                           bottom_width,
                                           bottom_channel,
                                           group_height,
                                           align_input_width,
                                           top_width,
                                           top_height,
                                           collapse_height,
                                           (unsigned char)byte_zero);

        } else if (runtime_->isValhall() && kernel_height == 9 && kernel_width == 9 && padding.b == 4 && padding.t == 4 &&
                   padding.l == 4 && padding.r == 4 && stride_height == 1 && stride_width == 1 && top_width % 8 == 0) {
            local_convert[0] = 16;
            local_convert[1] = 16;
            local_convert[2] = 1;
            global_convert[0] = alignTo(top_height * top_width / 8, local_convert[0]);
            global_convert[1] = alignTo(input->getDim().c * 81, local_convert[1]);
            global_convert[2] = input->getDim().n;
            state = runtime_->setKernelArg(quantized_gemm1xX_align_convert_withpad_kernel_.get(),
                                           bottom_buffer,
                                           convert_buffer,
                                           bottom_height,
                                           bottom_width,
                                           bottom_channel,
                                           group_height,
                                           align_input_width,
                                           collapse_height,
                                           (unsigned char)byte_zero);
        } else if (runtime_->isValhall() && kernel_height == 5 && kernel_width == 5 && padding.b == 2 && padding.t == 2 &&
                   padding.l == 2 && padding.r == 2 && stride_height == 1 && stride_width == 1 && top_width % 8 == 0) {
            local_convert[0] = 16;
            local_convert[1] = 16;
            local_convert[2] = 1;
            global_convert[0] = alignTo(top_height * top_width / 8, local_convert[0]);
            global_convert[1] = alignTo(input->getDim().c * 25, local_convert[1]);
            global_convert[2] = input->getDim().n;
            state = runtime_->setKernelArg(quantized_gemm1xX_align_convert_withpad_kernel_.get(),
                                           bottom_buffer,
                                           convert_buffer,
                                           bottom_height,
                                           bottom_width,
                                           bottom_channel,
                                           group_height,
                                           align_input_width,
                                           collapse_height,
                                           (unsigned char)byte_zero);
        } else if (runtime_->isValhall() && kernel_height == 3 && kernel_width == 3 && padding.b == 1 && padding.t == 1 &&
                   padding.l == 1 && padding.r == 1 && stride_height == 1 && stride_width == 1 && top_width % 8 == 0) {
            local_convert[0] = 16;
            local_convert[1] = 16;
            local_convert[2] = 1;
            global_convert[0] = alignTo(top_height * top_width / 8, local_convert[0]);
            global_convert[1] = alignTo(input->getDim().c * 9, local_convert[1]);
            global_convert[2] = input->getDim().n;
            state = runtime_->setKernelArg(quantized_gemm1xX_align_convert_withpad_kernel_.get(),
                                           bottom_buffer,
                                           convert_buffer,
                                           bottom_height,
                                           bottom_width,
                                           bottom_channel,
                                           group_height,
                                           align_input_width,
                                           collapse_height,
                                           (unsigned char)byte_zero);
        } else if (kernel_height == 3 && kernel_width == 3 && padding.b < 2 && padding.t < 2 && padding.l < 2 &&
                   padding.r < 2) {
            state = runtime_->setKernelArg(quantized_gemm1xX_align_convert_withpad_kernel_.get(),
                                           bottom_buffer,
                                           convert_buffer,
                                           padding.t,
                                           padding.l,
                                           stride_width,
                                           stride_height,
                                           bottom_height,
                                           bottom_width,
                                           bottom_channel,
                                           group_height,
                                           align_input_width,
                                           top_width,
                                           top_height,
                                           collapse_height,
                                           (unsigned char)byte_zero);

        } else {
            state = runtime_->setKernelArg(quantized_gemm1xX_align_convert_withpad_kernel_.get(),
                                           bottom_buffer,
                                           convert_buffer,
                                           kernel_width,
                                           kernel_height,
                                           padding.t,
                                           padding.l,
                                           stride_width,
                                           stride_height,
                                           bottom_height,
                                           bottom_width,
                                           bottom_channel,
                                           group_height,
                                           align_input_width,
                                           top_width,
                                           top_height,
                                           collapse_height,
                                           (unsigned char)byte_zero);
        }
    }
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");
    state = runtime_->enqueueKernel(quantized_gemm1xX_align_convert_withpad_kernel_.get(), 3, global_convert, local_convert);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "enqueueKernel failure\n");

    return state;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
