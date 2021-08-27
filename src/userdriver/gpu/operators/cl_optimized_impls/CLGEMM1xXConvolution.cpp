
#include "userdriver/gpu/operators/cl_optimized_impls/CLGEMM1xXConvolution.hpp"
#include "userdriver/common/operator_interfaces/common/Debug.hpp"

namespace enn {
namespace ud {
namespace gpu {

CLGEMM1xXConvolution::CLGEMM1xXConvolution(const std::shared_ptr<CLRuntime> runtime,
                                           const PrecisionType &precision,
                                           const Dim4 &input_dim,
                                           const Dim4 &output_dim) :
    runtime_(runtime),
    precision_(precision), input_dim_(input_dim), output_dim_(output_dim), align_weight_height_(0), align_weight_width_(0),
    align_input_height_(0), align_input_width_(0) {
    ENN_DBG_PRINT("CLGEMM1xXConvolution is created");
    weight_buffer_ = nullptr;
    computed_top_number_ = 0;
    convert_out_dim_ = {0, 0, 0, 0};
    kernel_ = {0, 0};
    stride_ = {0, 0};
    pad_temp_ = {0, 0, 0, 0};
    weights_as_input_ = false;
    androidNN_ = false;
    unaligned_weight_height_ = 0;
    unaligned_weight_width_ = 0;
    if (runtime->isValhall()) {
        gpu_platform_ = GPUPlatform::Valhall;
    } else if (runtime->isMakalu()) {
        gpu_platform_ = GPUPlatform::Makalu;
    } else if (runtime->isBifrost()) {
        gpu_platform_ = GPUPlatform::Biforst;
    } else {
        gpu_platform_ = GPUPlatform::Unknow;
    }

    gemm_kernel_type_ = GEMM1xXKernelType::GEMMInit;

    is_makalu_branch_ = false;
    top_HW_split_count_ = 0;
    top_HW_split_size_ = 0;
}

Status CLGEMM1xXConvolution::initialize(const Pad4 &padding,
                                        const Dim2 &stride,
                                        const uint32_t &group_size,
                                        const uint32_t &axis,
                                        const Dim2 &dilation,
                                        const std::shared_ptr<ITensor> weight,
                                        const std::shared_ptr<ITensor> bias,
                                        const ActivationInfo &activate_info,
                                        const bool &weights_as_input,
                                        const bool &androidNN,
                                        const bool &isNCHW) {
    ENN_DBG_PRINT("CLGEMM1xXConvolution::initialize() is called");
    runtime_->resetIntraBuffer();

    conv_descriptor_.num_output_ = weight->getDim().n;
    conv_descriptor_.pad_right_ = padding.r;
    conv_descriptor_.pad_left_ = padding.l;
    conv_descriptor_.pad_top_ = padding.t;
    conv_descriptor_.pad_bottom_ = padding.b;
    conv_descriptor_.stride_height_ = stride.h;
    conv_descriptor_.stride_width_ = stride.w;
    conv_descriptor_.group_ = group_size;
    conv_descriptor_.axis_ = axis;
    conv_descriptor_.dilation_ = dilation;

    activation_info_ = activate_info;
    weights_as_input_ = weights_as_input;
    androidNN_ = androidNN;
    isNCHW_ = isNCHW;

    auto bias_tensor = std::static_pointer_cast<CLTensor>(bias);
    conv_descriptor_.bias_ = bias_tensor;

    auto weight_tensor = std::static_pointer_cast<CLTensor>(weight);
    conv_descriptor_.filter_ = weight_tensor;

    Dim4 weight_size = weight_tensor->getDim();
    if (androidNN_ || !isNCHW_) {
        weight_size = convertDimToNCHW(weight_size);
        weight_nchw_ = std::make_shared<CLTensor>(runtime_,
                                                  precision_,
                                                  weight_tensor->getDataType(),
                                                  weight_size,
                                                  weight_tensor->getDataOrder(),
                                                  weight_tensor->getScale(),
                                                  weight_tensor->getZeroPoint());
        if (!weights_as_input) {
            Status state = weight_tensor->convertToNCHW(weight_nchw_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "convertToNCHW failure\n");
        }
    }

    if (!weights_as_input && (dilation.h > 1 || dilation.w > 1)) {
        dilationWeight((androidNN_ || !isNCHW_) ? weight_nchw_ : conv_descriptor_.filter_);
    }

    conv_descriptor_.kernel_height_ = dilation.h * (weight_size.h - 1) + 1;
    conv_descriptor_.kernel_width_ = dilation.w * (weight_size.w - 1) + 1;

    kernel_ = {conv_descriptor_.kernel_height_, conv_descriptor_.kernel_width_};
    stride_ = {conv_descriptor_.stride_height_, conv_descriptor_.stride_width_};

    unaligned_weight_height_ = conv_descriptor_.num_output_;
    unaligned_weight_width_ = conv_descriptor_.kernel_height_ * conv_descriptor_.kernel_width_ * input_dim_.c;

    _kernel_name_ = "";
    CLPadConvert::PadConvertType pad_convert_type = CLPadConvert::PadConvertType::NOSET;
    Status state;
    if (gpu_platform_ == GPUPlatform::Valhall) {
        computed_top_number_ = GEMM1XX_TILE_SIZE_4;
        gemm_kernel_type_ = GEMM1xXKernelType::GEMMValhall4x4;
        if (!isNCHW && kernel_.h == 1 && kernel_.w == 1 && stride_.h == 1 && stride_.w == 1 && group_size == 1 &&
            padding.b == 0 && padding.t == 0 && padding.l == 0 && padding.r == 0 && output_dim_.h * output_dim_.w % 4 == 0) {
            pad_convert_type = CLPadConvert::PadConvertType::GEMM4x4MakaluNHWC1X1;
            _kernel_name_ = "gemmValhall4x4NHWC";
        } else {
            pad_convert_type = CLPadConvert::PadConvertType::GEMM4x4Makalu;
            _kernel_name_ = "gemmValhall4x4";
        }
    } else if (gpu_platform_ == GPUPlatform::Makalu) {
        computed_top_number_ = GEMM1XX_TILE_SIZE_4;
        is_makalu_branch_ = true;
        ENN_DBG_PRINT("gemm type is GEMM4x4");
        gemm_kernel_type_ = GEMM1xXKernelType::GEMM4x4;
        _kernel_name_ = "gemmMakalu";
        pad_convert_type = CLPadConvert::PadConvertType::GEMM4x4Makalu;
    } else if (gpu_platform_ == GPUPlatform::Biforst) {
        if (precision_ == PrecisionType::FP16 && (static_cast<double>(output_dim_.h * output_dim_.w * output_dim_.c) /
                                                  GEMM1XX_TILE_SIZE_2 / GEMM1XX_TILE_SIZE_8) /
                                                         12.0 / 20.0 >
                                                     20.0) {
            ENN_DBG_PRINT("gemm type is GEMM2x8Scalar");
            computed_top_number_ = GEMM1XX_TILE_SIZE_8;
            gemm_kernel_type_ = GEMM1xXKernelType::GEMM2x8Scalar;
            _kernel_name_ = "gemmBlock2x8_scalar";
        } else {
            ENN_DBG_PRINT("gemm type is GEMM2x4Scalar");
            computed_top_number_ = GEMM1XX_TILE_SIZE_4;
            gemm_kernel_type_ = GEMM1xXKernelType::GEMM2x4Scalar;
            _kernel_name_ = "gemmBlock2x4_scalar";
        }
    } else if (gpu_platform_ == GPUPlatform::Unknow) {
        if (precision_ == PrecisionType::FP16 && (static_cast<double>(output_dim_.h * output_dim_.w * output_dim_.c) /
                                                  GEMM1XX_TILE_SIZE_2 / GEMM1XX_TILE_SIZE_8) /
                                                         12.0 / 20.0 >
                                                     20.0) {
            ENN_DBG_PRINT("gemm type is GEMM2x8");
            computed_top_number_ = GEMM1XX_TILE_SIZE_8;
            gemm_kernel_type_ = GEMM1xXKernelType::GEMM2x8;
            _kernel_name_ = "gemmBlock2x8";
            pad_convert_type = CLPadConvert::PadConvertType::GEMM1xX;
        } else {
            ENN_DBG_PRINT("gemm type is GEMM2x4");
            computed_top_number_ = GEMM1XX_TILE_SIZE_4;
            gemm_kernel_type_ = GEMM1xXKernelType::GEMM2x4;
            _kernel_name_ = "gemmBlock2x4";
        }
    }
    generateKernelName(activate_info);
    state = runtime_->setKernel(&gemm1xx_kernel_, _kernel_name_, precision_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setArg failure\n");
    if (pad_convert_type != CLPadConvert::PadConvertType::NOSET) {
        pad_convert_executor_ = std::make_shared<CLPadConvert>(runtime_, precision_);
        pad_convert_executor_->initialize(
            input_dim_, padding, kernel_, stride_, group_size, {1, 1, 1, 1}, false, pad_convert_type);
    }

    Dim4 converted_filter_dim;
    switch (gemm_kernel_type_) {
        int topHW;
    case GEMM1xXKernelType::GEMM2x4:
    case GEMM1xXKernelType::GEMM2x4Scalar:
        align_weight_height_ = alignTo(unaligned_weight_height_, GEMM1XX_TILE_SIZE_4);
        align_weight_width_ = alignTo(unaligned_weight_width_, GEMM1XX_VECTOR_SIZE);
        align_input_height_ = alignTo(output_dim_.h * output_dim_.w, GEMM1XX_COALESCING_SIZE);
        align_input_width_ = align_weight_width_;
        break;
    case GEMM1xXKernelType::GEMM2x8:
    case GEMM1xXKernelType::GEMM2x8Scalar:
        align_weight_height_ = alignTo(unaligned_weight_height_, GEMM1XX_TILE_SIZE_8);
        align_weight_width_ = alignTo(unaligned_weight_width_, GEMM1XX_VECTOR_SIZE);
        align_input_height_ = alignTo(output_dim_.h * output_dim_.w, GEMM1XX_COALESCING_SIZE);
        align_input_width_ = align_weight_width_;
        break;
    case GEMM1xXKernelType::GEMM4x4:
        align_weight_height_ = alignTo(unaligned_weight_height_, GEMM1XX_TILE_SIZE_8);
        align_weight_width_ = alignTo(unaligned_weight_width_, GEMM1XX_TILE_SIZE_4);
        align_input_height_ = alignTo(output_dim_.h * output_dim_.w, GEMM1XX_COALESCING_INPUT_4_THREAD_12);
        align_input_width_ = align_weight_width_;
        topHW = output_dim_.h * output_dim_.w;
        top_HW_split_count_ = 1;
        top_HW_split_size_ = topHW;
        if (topHW > GEMM1XX_HW_SPLIT_SIZE && !(kernel_.h == 1 && kernel_.w == 1)) {
            top_HW_split_count_ = ceil(static_cast<double>(topHW) / GEMM1XX_HW_SPLIT_SIZE);
            top_HW_split_size_ = GEMM1XX_HW_SPLIT_SIZE;
            align_input_height_ = GEMM1XX_HW_SPLIT_SIZE;
        }
        break;
    case GEMM1xXKernelType::GEMMValhall4x4:
        align_weight_height_ = alignTo(unaligned_weight_height_, GEMM1XX_TILE_SIZE_8);
        align_weight_width_ = alignTo(unaligned_weight_width_, GEMM1XX_TILE_SIZE_8);
        align_input_height_ = alignTo(output_dim_.h * output_dim_.w, GEMM1XX_COALESCING_INPUT_4_THREAD_8);
        align_input_width_ = align_weight_width_;
        break;
    default: break;
    }

    // do pad and img2col
    pad_temp_ = {
        conv_descriptor_.pad_top_, conv_descriptor_.pad_right_, conv_descriptor_.pad_bottom_, conv_descriptor_.pad_left_};

    convert_out_dim_.n = input_dim_.n;
    convert_out_dim_.c = 1;
    convert_out_dim_.h = align_input_height_;
    convert_out_dim_.w = align_input_width_;
    if (weights_as_input_ == false) {
        convert_output_ = std::make_shared<CLTensor>(
            runtime_, precision_, DataType::FLOAT, convert_out_dim_, DataOrder::NCHW, 1.0, 0, BufferType::INTRA_SHARED);
    } else {
        convert_output_ = std::make_shared<CLTensor>(
            runtime_, precision_, DataType::FLOAT, convert_out_dim_, DataOrder::NCHW, 1.0, 0, BufferType::DEDICATED);
    }

    converted_filter_dim = {static_cast<uint32_t>(align_weight_height_ * align_weight_width_), 1, 1, 1};
    converted_filter_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, converted_filter_dim);
    if (!weights_as_input_) {
        Status state = alignWeight();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "alignWeight failure\n");
    }

    return Status::SUCCESS;
}

Status CLGEMM1xXConvolution::execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) {
    ENN_DBG_PRINT("CLGEMM1xXConvolution::execute() is called");
    Status state = Status::FAILURE;
    if (weights_as_input_) {
        if (!isNCHW_ && kernel_.h == 1 && kernel_.w == 1 && stride_.h == 1 && stride_.w == 1 &&
            conv_descriptor_.group_ == 1 && conv_descriptor_.pad_bottom_ == 0 && conv_descriptor_.pad_top_ == 0 &&
            conv_descriptor_.pad_left_ == 0 && conv_descriptor_.pad_right_ == 0 && conv_descriptor_.dilation_.h == 1 &&
            conv_descriptor_.dilation_.w == 1) {
            Dim4 weight_dim_nhwc = conv_descriptor_.filter_->getDim();
            weight_nchw_ = conv_descriptor_.filter_;
            weight_nchw_->reconfigureDim(convertDimToNCHW(weight_dim_nhwc));
            Status state = alignWeight();
            weight_nchw_->reconfigureDim(weight_dim_nhwc);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "alignWeight failure\n");

        } else {
            if (androidNN_ || !isNCHW_) {
                state = conv_descriptor_.filter_->convertToNCHW(weight_nchw_);
                CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "convertToNCHW failure\n");
            }
            if (conv_descriptor_.dilation_.h > 1 || conv_descriptor_.dilation_.w > 1) {
                dilationWeight((androidNN_ || !isNCHW_) ? weight_nchw_ : conv_descriptor_.filter_);
            }
            Status state = alignWeight();
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "alignWeight failure\n");
        }

        ENN_DBG_PRINT("alignWeight is finished");
    }

    auto input_tensor = std::static_pointer_cast<CLTensor>(input);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output);
    weight_buffer_ = converted_filter_->getDataPtr();
    // choose different gemm
    switch (gemm_kernel_type_) {
    case GEMM1xXKernelType::GEMM4x4: state = gemm4x4Run(input_tensor, weight_buffer_, output_tensor); break;
    case GEMM1xXKernelType::GEMM2x4Scalar:
    case GEMM1xXKernelType::GEMM2x8Scalar:
    case GEMM1xXKernelType::GEMM2x4:
    case GEMM1xXKernelType::GEMM2x8:
        state = pad_convert_executor_->img2ColWithPad1xXRun(
            input_tensor, convert_output_, pad_temp_, kernel_, stride_, GEMM1XX_COALESCING_SIZE);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "gemm1xX convert failure\n");
        state = gemm1xXRun(convert_output_, weight_buffer_, output_tensor);
        break;
    case GEMM1xXKernelType::GEMMValhall4x4: state = gemmValhall4x4Run(input_tensor, weight_buffer_, output_tensor); break;
    default: break;
    }
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "gemm1xX execute\n");

    return state;
}

Status CLGEMM1xXConvolution::alignWeight() {
    cl_mem dst = converted_filter_->getDataPtr();
    cl_mem src = (conv_descriptor_.dilation_.h > 1 || conv_descriptor_.dilation_.w > 1)
                     ? dilation_filter_->getDataPtr()
                     : ((androidNN_ || !isNCHW_) ? weight_nchw_->getDataPtr() : conv_descriptor_.filter_->getDataPtr());
    Status state = Status::FAILURE;
    switch (gemm_kernel_type_) {
    case GEMM1xXKernelType::GEMM2x4:
    case GEMM1xXKernelType::GEMM2x4Scalar:
        state = alignWeight1xXGEMM(
            src, dst, unaligned_weight_width_, unaligned_weight_height_, align_weight_width_, GEMM1XX_TILE_SIZE_4);
        break;
    case GEMM1xXKernelType::GEMM2x8:
    case GEMM1xXKernelType::GEMM2x8Scalar:
        state = alignWeight1xXGEMM(
            src, dst, unaligned_weight_width_, unaligned_weight_height_, align_weight_width_, GEMM1XX_TILE_SIZE_8);
        break;
    case GEMM1xXKernelType::GEMM4x4:
    case GEMM1xXKernelType::GEMMValhall4x4:
        state = alignWeight1xXMakalu(
            src, dst, unaligned_weight_width_, unaligned_weight_height_, align_weight_width_, align_weight_height_);
        break;
    default: break;
    }

    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "align weight failure\n");

    return Status::SUCCESS;
}

Status CLGEMM1xXConvolution::gemm1xXRun(const std::shared_ptr<CLTensor> input,
                                        const cl_mem weightBuffer,
                                        std::shared_ptr<CLTensor> output) {
    ENN_DBG_PRINT("gemm1xXRun is called");
    auto input_tensor = std::static_pointer_cast<CLTensor>(input);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output);
    auto inBuffer = input_tensor->getDataPtr();
    auto outBuffer = output_tensor->getDataPtr();
    auto biasBuffer = conv_descriptor_.bias_->getDataPtr();
    Dim4 in_size = input->getDim();
    Dim4 out_size = output->getDim();
    int alignedCinKK = in_size.w;
    int topHW = out_size.h * out_size.w;
    int topChannel = out_size.c;
    int bottomStep = in_size.h * in_size.w;

    size_t localConv[3] = {1, 1, 12};
    size_t globalConv[3] = {0, 0, 0};
    globalConv[0] = out_size.n;
    globalConv[1] = alignTo(ceil(topChannel / static_cast<double>(computed_top_number_)), localConv[1]);
    globalConv[2] = alignTo(ceil(static_cast<double>(topHW) / GEMM1XX_TILE_SIZE_2), localConv[2]);
    Status state;

    // non-makalu won't split
    state = runtime_->setKernelArg(
        gemm1xx_kernel_.get(), inBuffer, weightBuffer, biasBuffer, outBuffer, alignedCinKK, topChannel, topHW, bottomStep);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setArg failure\n");
    state = runtime_->enqueueKernel(gemm1xx_kernel_.get(), 3, globalConv, localConv);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
    return Status::SUCCESS;
}

Status CLGEMM1xXConvolution::gemmValhall4x4Run(const std::shared_ptr<CLTensor> input,
                                               const cl_mem weightBuffer,
                                               std::shared_ptr<CLTensor> output) {
    auto input_tensor = std::static_pointer_cast<CLTensor>(input);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output);
    auto inBuffer = convert_output_->getDataPtr();
    auto outBuffer = output_tensor->getDataPtr();
    auto biasBuffer = conv_descriptor_.bias_->getDataPtr();
    Dim4 in_size = convert_output_->getDim();
    Dim4 out_size = output->getDim();
    if (!isNCHW_ && kernel_.h == 1 && kernel_.w == 1 && stride_.h == 1 && stride_.w == 1 && conv_descriptor_.group_ == 1 &&
        conv_descriptor_.pad_bottom_ == 0 && conv_descriptor_.pad_top_ == 0 && conv_descriptor_.pad_left_ == 0 &&
        conv_descriptor_.pad_right_ == 0 && output_dim_.h * output_dim_.w % 4 == 0) {
        out_size.n = output->getDim().n;
        out_size.c = output->getDim().w;
        out_size.h = output->getDim().c;
        out_size.w = output->getDim().h;
    }
    int alignedCinKK = in_size.w;
    int topHW = out_size.h * out_size.w;
    int topChannel = out_size.c;
    int bottomStep = in_size.h * in_size.w;
    Status state;
    size_t localConv[3] = {1, (size_t)2, (size_t)8};
    size_t globalConv[3] = {0, 0, 0};
    globalConv[0] = out_size.n;

    if (topChannel < 4) {
        localConv[1] = 1;
    }
    globalConv[1] = alignTo(ceil(topChannel / static_cast<double>(computed_top_number_)), localConv[1]);
    globalConv[2] = alignTo(ceil(static_cast<double>(topHW) / GEMM1XX_TILE_SIZE_4), localConv[2]);

    state = pad_convert_executor_->img2ColWithPad4x4MakaluRun(
        input_tensor, convert_output_, pad_temp_, kernel_, stride_, GEMM1XX_COALESCING_INPUT_4_THREAD_8, topHW, 0);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "gemm1xX convert failure\n");

    state = runtime_->setKernelArg(
        gemm1xx_kernel_.get(), inBuffer, weightBuffer, biasBuffer, outBuffer, alignedCinKK, topChannel, topHW, bottomStep);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setArg failure\n");
    state = runtime_->enqueueKernel(gemm1xx_kernel_.get(), 3, globalConv, localConv);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");

    return Status::SUCCESS;
}

Status CLGEMM1xXConvolution::gemm4x4Run(const std::shared_ptr<CLTensor> input,
                                        const cl_mem weightBuffer,
                                        std::shared_ptr<CLTensor> output) {
    ENN_DBG_PRINT("gemm4x4Run is called");
    auto input_tensor = std::static_pointer_cast<CLTensor>(input);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output);
    auto inBuffer = convert_output_->getDataPtr();
    auto outBuffer = output_tensor->getDataPtr();
    auto biasBuffer = conv_descriptor_.bias_->getDataPtr();
    Dim4 in_size = convert_output_->getDim();
    Dim4 out_size = output->getDim();
    int alignedCinKK = in_size.w;
    int topHW = out_size.h * out_size.w;
    int topChannel = out_size.c;
    int bottomStep = in_size.h * in_size.w;
    Status state;
    size_t localConv[3] = {1, (size_t)2, (size_t)12};
    size_t globalConv[3] = {0, 0, 0};
    globalConv[0] = out_size.n;

    if (topChannel < 4) {
        localConv[1] = 1;
    }
    globalConv[1] = alignTo(ceil(topChannel / static_cast<double>(computed_top_number_)), localConv[1]);
    globalConv[2] = alignTo(ceil(static_cast<double>(topHW) / GEMM1XX_TILE_SIZE_4), localConv[2]);

    // non-split and processing all the data together
    for (int i = 0; i < top_HW_split_count_; i++) {
        // do img2col
        // following two lines of code may be need at optimize period
        //        int ss = convert_output_->getNumOfBytes();
        //        runtime_->zeroBuf(ss,convert_output_->getDataPtr());
        state = pad_convert_executor_->img2ColWithPad4x4MakaluRun(input_tensor,
                                                                  convert_output_,
                                                                  pad_temp_,
                                                                  kernel_,
                                                                  stride_,
                                                                  GEMM1XX_COALESCING_INPUT_4_THREAD_12,
                                                                  top_HW_split_size_,
                                                                  i);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "gemm1xX convert failure\n");
        int inputOffset = 0;  // i * (top_HW_split_size_ / 48) * 48 * alignedCinKK;
        int outputOffset = i * top_HW_split_size_;

        int tmp_top_HW_split_size_ = top_HW_split_size_;
        if ((i + 1) * top_HW_split_size_ > topHW) {
            tmp_top_HW_split_size_ = topHW % top_HW_split_size_;
        }
        int factor = ceil(static_cast<double>(top_HW_split_size_) / 4);
        if (factor > 384) {
            localConv[2] = 128;
        } else if (factor > 96) {
            localConv[2] = 96;
        } else if (factor > 48) {
            localConv[2] = 48;
        } else if (factor > 24) {
            localConv[2] = 24;
        } else {
            localConv[2] = 12;
        }
        if (topChannel <= 64) {
            localConv[2] = 12;
        }
        globalConv[2] = alignTo(factor, localConv[2]);
        state = runtime_->setKernelArg(gemm1xx_kernel_.get(),
                                       inBuffer,
                                       weightBuffer,
                                       biasBuffer,
                                       outBuffer,
                                       alignedCinKK,
                                       topChannel,
                                       topHW,
                                       tmp_top_HW_split_size_,
                                       bottomStep,
                                       inputOffset,
                                       outputOffset);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setArg failure\n");
        state = runtime_->enqueueKernel(gemm1xx_kernel_.get(), 3, globalConv, localConv);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
    }
    return Status::SUCCESS;
}

Status CLGEMM1xXConvolution::alignWeight1xXGEMM(const cl_mem &src,
                                                cl_mem &dst,
                                                int srcWidth,
                                                int srcHeight,
                                                int dstWidth,
                                                int computedTop) {
    auto unalignedSrc = src;
    auto alignedDst = dst;
    if (computedTop == GEMM1XX_TILE_SIZE_8) {
        Status state = runtime_->setKernel(&align_weight_kernel_, "alignWeight_1x8", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setArg kernel failure\n");
    } else {
        Status state = runtime_->setKernel(&align_weight_kernel_, "alignWeight_1x4", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setArg kernel failure\n");
    }
    size_t local_align_weight[2] = {0, 0};
    local_align_weight[0] = 1;
    local_align_weight[1] = 12;
    size_t global_align_weight[2] = {0, 0};
    global_align_weight[0] = srcHeight;
    if (computedTop == GEMM1XX_TILE_SIZE_4) {
        global_align_weight[1] = alignTo(ceil(static_cast<double>(srcWidth) / GEMM1XX_VECTOR_SIZE), local_align_weight[1]);
    } else {
        global_align_weight[1] = alignTo(srcWidth, local_align_weight[1]);
    }
    Status state = runtime_->setKernelArg(align_weight_kernel_.get(), unalignedSrc, alignedDst, srcWidth, dstWidth);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setArg kernel failure\n");
    state = runtime_->enqueueKernel(align_weight_kernel_.get(), 2, global_align_weight, local_align_weight);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
    return Status::SUCCESS;
}

Status CLGEMM1xXConvolution::alignWeight1xXMakalu(const cl_mem &src,
                                                  cl_mem &dst,
                                                  int srcWidth,
                                                  int srcHeight,
                                                  int dstWidth,
                                                  int dstHeight) {
    cl_mem unalignedSrc = src;
    cl_mem alignedDst = dst;

    Status state = runtime_->setKernel(&align_weight_kernel_, "align_weight_4_row_1_col", precision_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setArg kernel failure\n");

    size_t globalAlignWeight[2] = {0, 0};
    globalAlignWeight[0] = dstHeight / 8;
    globalAlignWeight[1] = dstWidth * 8;

    state = runtime_->setKernelArg(align_weight_kernel_.get(), unalignedSrc, alignedDst, srcHeight, srcWidth, dstWidth * 8);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setArg kernel failure\n");

    state = runtime_->enqueueKernel(align_weight_kernel_.get(), 2, globalAlignWeight, NULL);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
    return Status::SUCCESS;
}

Status CLGEMM1xXConvolution::dilationWeight(const std::shared_ptr<CLTensor> weight_tensor) {
    uint32_t batch_weight = weight_tensor->getDim().n;
    uint32_t channel_weight = weight_tensor->getDim().c;
    uint32_t kernel_height = weight_tensor->getDim().h;
    uint32_t kernel_width = weight_tensor->getDim().w;

    uint32_t extent_height = conv_descriptor_.dilation_.h * (kernel_height - 1) + 1;
    uint32_t extent_width = conv_descriptor_.dilation_.w * (kernel_width - 1) + 1;
    Dim4 dim_extent = {batch_weight, channel_weight, extent_height, extent_width};
    auto weight_dilation = std::make_shared<CLTensor>(runtime_,
                                                      precision_,
                                                      weight_tensor->getDataType(),
                                                      dim_extent,
                                                      weight_tensor->getDataOrder(),
                                                      weight_tensor->getScale(),
                                                      weight_tensor->getZeroPoint());

    size_t global[3];
    global[0] = batch_weight * channel_weight;
    global[1] = kernel_height;
    global[2] = kernel_width;
    std::shared_ptr<struct _cl_kernel> kernel_dilation;
    Status state = runtime_->setKernel(&kernel_dilation, "dilation", precision_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
    state = runtime_->setKernelArg(kernel_dilation.get(),
                                   weight_tensor->getDataPtr(),
                                   weight_dilation->getDataPtr(),
                                   extent_height,
                                   extent_width,
                                   conv_descriptor_.dilation_.h,
                                   conv_descriptor_.dilation_.w);

    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");
    state = runtime_->enqueueKernel(kernel_dilation.get(), (cl_uint)3, global, NULL);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "enqueueKernel failure\n");

    dilation_filter_ = weight_dilation;

    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
