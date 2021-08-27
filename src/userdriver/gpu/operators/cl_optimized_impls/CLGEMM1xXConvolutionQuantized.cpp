#include "userdriver/gpu/operators/cl_optimized_impls/CLGEMM1xXConvolutionQuantized.hpp"
#include "userdriver/common/operator_interfaces/common/Debug.hpp"

namespace enn {
namespace ud {
namespace gpu {

CLGEMM1xXConvolutionQuantized::CLGEMM1xXConvolutionQuantized(const std::shared_ptr<CLRuntime> runtime,
                                                             const PrecisionType &precision) :
    runtime_(runtime),
    precision_(precision) {
    ENN_DBG_PRINT("CLGEMM1xXConvolutionQuantized is created");
    input_dim_ = {0, 0, 0, 0};
    output_dim_ = {0, 0, 0, 0};
    convert_out_dim_ = {0, 0, 0, 0};
    stride_ = {0, 0};
    kernel_ = {0, 0};
    padding_ = {0, 0, 0, 0};
    group_ = 0;
    input_offset_ = 0;
    output_multiplier_ = 0;
    output_shift_ = 0;
    kBlock_ = 0;
    computed_top_number_ = 0;
    coalescing_feature_height_ = 0;
    dilation_ = {0, 0};
    signed_quant_ = false;

    weights_as_input_ = false;
    bias_as_input_ = false;
    androidNN_ = false;
    unaligned_weight_height_ = 0;
    unaligned_weight_width_ = 0;
    align_weight_height_makalu_ = 0;
    align_weight_width_makalu_ = 0;
}

Status CLGEMM1xXConvolutionQuantized::initialize(const std::shared_ptr<ITensor> input,
                                                 const Dim4 &output_dim,
                                                 const Pad4 &padding,
                                                 const Dim2 &stride,
                                                 const uint32_t &group_size,
                                                 const uint32_t &axis,
                                                 const Dim2 &dilation,
                                                 const std::shared_ptr<ITensor> weight,
                                                 const std::shared_ptr<ITensor> bias,
                                                 const ActivationInfo &activate_info,
                                                 const bool &weights_as_input,
                                                 const bool &bias_as_input,
                                                 const bool &androidNN) {
    ENN_DBG_PRINT("CLGEMM1xXConvolutionQuantized::initialize() is called");
    Status state = Status::FAILURE;
    input_dim_ = input->getDim();
    output_dim_ = output_dim;
    auto input_tensor = std::static_pointer_cast<CLTensor>(input);
    stride_ = stride;
    padding_ = padding;
    group_ = group_size;
    activation_info_ = activate_info;
    weights_as_input_ = weights_as_input;
    bias_as_input_ = bias_as_input;
    androidNN_ = androidNN;
    dilation_ = dilation;
    signed_quant_ = precision_ == PrecisionType::INT8;

    auto weight_tensor = std::static_pointer_cast<CLTensor>(weight);
    auto bias_tensor = std::static_pointer_cast<CLTensor>(bias);
    filter_ = weight_tensor;
    bias_ = bias_tensor;

    Dim4 weight_size = weight_tensor->getDim();
    if (androidNN_) {
        weight_size = convertDimToNCHW(weight_size);
        weight_nchw_ = std::make_shared<CLTensor>(runtime_,
                                                  precision_,
                                                  weight_tensor->getDataType(),
                                                  weight_size,
                                                  weight_tensor->getDataOrder(),
                                                  weight_tensor->getScale(),
                                                  weight_tensor->getZeroPoint());
        if (weights_as_input_ == false && bias_as_input_ == false) {
            state = weight_tensor->convertToNCHW(weight_nchw_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "convertToNCHW failure\n");
        }
    }

    if (!weights_as_input && !bias_as_input_ && (dilation.h > 1 || dilation.w > 1)) {
        dilationWeight(androidNN_ ? weight_nchw_ : filter_);
    }

    kernel_.h = dilation.h * (weight_size.h - 1) + 1;
    kernel_.w = dilation.w * (weight_size.w - 1) + 1;

    if (signed_quant_) {
        state = runtime_->setKernel(&align_weight1xXmakalu_kernel_, "SIGNEDalign_weight_4_row_1_col", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
        state = runtime_->setKernel(&weight_offset_kernel_, "SIGNEDweight_offset", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel weight_offset_kernel_ failure\n");
    } else {
        state = runtime_->setKernel(&align_weight1xXmakalu_kernel_, "align_weight_4_row_1_col", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
        state = runtime_->setKernel(&weight_offset_kernel_, "weight_offset", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel weight_offset_kernel_ failure\n");
    }

    kBlock_ = GEMM1XX_TILE_SIZE_4;  // 4 elements are loaded to accumulate each time in int8
    if (signed_quant_) {
        if (runtime_->isValhall()) {
            coalescing_feature_height_ = GEMM1XX_COALESCING_SIZE_4_LINES_8_THREADS;
            kBlock_ = 8;
            state = runtime_->setKernel(&quantized_gemm1xX_kernel_, "SIGNEDgemm_block4x4_valhal", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
        } else {
            coalescing_feature_height_ = GEMM1XX_COALESCING_SIZE_4_LINES_12_THREADS;
            state = runtime_->setKernel(&quantized_gemm1xX_kernel_, "SIGNEDgemm_block4x4", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
        }
    } else {
        if (runtime_->isValhall()) {
            coalescing_feature_height_ = GEMM1XX_COALESCING_SIZE_4_LINES_8_THREADS;
            kBlock_ = 8;
            state = runtime_->setKernel(&quantized_gemm1xX_kernel_, "gemm_block4x4_valhal", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
        } else {
            coalescing_feature_height_ = GEMM1XX_COALESCING_SIZE_4_LINES_12_THREADS;
            state = runtime_->setKernel(&quantized_gemm1xX_kernel_, "gemm_block4x4", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
        }
    }
    computed_top_number_ = GEMM1XX_TILE_SIZE_4;  // each thread computes 4 top channels

    unaligned_weight_height_ = weight_size.n;
    unaligned_weight_width_ = kernel_.h * kernel_.w * input_dim_.c;
    align_weight_height_makalu_ = alignTo(unaligned_weight_height_, GEMM1XX_TILE_SIZE_8);
    align_weight_width_makalu_ = alignTo(unaligned_weight_width_, kBlock_);

    uint32_t align_weight_width = alignTo(unaligned_weight_width_, kBlock_);
    uint32_t align_input_height = alignTo(output_dim.h * output_dim.w, coalescing_feature_height_);
    uint32_t align_input_width = align_weight_width;
    pad_convert_executor_ = std::make_shared<CLPadConvert>(runtime_, precision_);

    convert_out_dim_.n = input_dim_.n;
    convert_out_dim_.c = 1;
    convert_out_dim_.h = align_input_height;
    convert_out_dim_.w = align_input_width;
    convert_output_ = std::make_shared<CLTensor>(runtime_, precision_, input->getDataType(), convert_out_dim_);

    // Weight Align
    uint32_t weight_buffer_makalu_size = align_weight_height_makalu_ * align_weight_width_makalu_;
    Dim4 weight_buffer_makalu_dim = {weight_buffer_makalu_size, 1, 1, 1};
    weight_buffer_makalu_ =
        std::make_shared<CLTensor>(runtime_, precision_, weight->getDataType(), weight_buffer_makalu_dim);
    if (signed_quant_) {
        auto makalu_buffer = make_shared_array<int8_t>(weight_buffer_makalu_size);
        memset(makalu_buffer.get(), weight_tensor->getZeroPoint(), weight_buffer_makalu_size * sizeof(int8_t));
        weight_buffer_makalu_->writeData(makalu_buffer.get());
    } else {
        auto makalu_buffer = make_shared_array<uint8_t>(weight_buffer_makalu_size);
        memset(makalu_buffer.get(), weight_tensor->getZeroPoint(), weight_buffer_makalu_size * sizeof(uint8_t));
        weight_buffer_makalu_->writeData(makalu_buffer.get());
    }

    if (weights_as_input_ == false && bias_as_input_ == false) {
        auto weightBufferMakaluData = weight_buffer_makalu_->getDataPtr();
        if (dilation.h > 1 || dilation.w > 1) {
            state = alignWeight1xXMakalu(dilation_filter_->getDataPtr(),
                                         weightBufferMakaluData,
                                         unaligned_weight_width_,
                                         unaligned_weight_height_,
                                         align_weight_width_makalu_,
                                         align_weight_height_makalu_,
                                         GEMM1XX_TILE_SIZE_4);
            state = moveWeightOffset2Bias(dilation_filter_, input_tensor, bias_);
        } else {
            auto filter = androidNN_ ? weight_nchw_ : filter_;
            state = alignWeight1xXMakalu(filter->getDataPtr(),
                                         weightBufferMakaluData,
                                         unaligned_weight_width_,
                                         unaligned_weight_height_,
                                         align_weight_width_makalu_,
                                         align_weight_height_makalu_,
                                         GEMM1XX_TILE_SIZE_4);
            state = moveWeightOffset2Bias(filter, input_tensor, bias_);
        }
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "align weight failure\n");
    }
    return state;
}

Status CLGEMM1xXConvolutionQuantized::execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) {
    ENN_DBG_PRINT("CLGEMM1xXConvolutionQuantized::execute() is called");
    ITensor::placeholder("CLGEMM1xXConvolutionQuantized",ColorType::BLUE);
    CLTensor::checknull(filter_,"filter_");
    CLTensor::checknull(bias_,"bias_tensor");
    auto input_tensor = std::static_pointer_cast<CLTensor>(input);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output);

    if (weights_as_input_ == true || bias_as_input_ == true) {
        auto weightBufferMakaluData = weight_buffer_makalu_->getDataPtr();
        if (androidNN_) {
            Status state = filter_->convertToNCHW(weight_nchw_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "convertToNCHW failure\n");
        }

        if (dilation_.h > 1 || dilation_.w > 1) {
            ITensor::placeholder("dilation_.h > 1 || dilation_.w > 1");
            Status state = dilationWeight(androidNN_ ? weight_nchw_ : filter_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "dilation weight failure\n");
            state = alignWeight1xXMakalu(dilation_filter_->getDataPtr(),
                                         weightBufferMakaluData,
                                         unaligned_weight_width_,
                                         unaligned_weight_height_,
                                         align_weight_width_makalu_,
                                         align_weight_height_makalu_,
                                         GEMM1XX_TILE_SIZE_4);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "align weight failure\n");
            state = moveWeightOffset2Bias(dilation_filter_, input_tensor, bias_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "move weight failure\n");

        } else {
            auto filter = androidNN_ ? weight_nchw_ : filter_;
            Status state = alignWeight1xXMakalu(filter->getDataPtr(),
                                                weightBufferMakaluData,
                                                unaligned_weight_width_,
                                                unaligned_weight_height_,
                                                align_weight_width_makalu_,
                                                align_weight_height_makalu_,
                                                GEMM1XX_TILE_SIZE_4);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "align weight failure\n");
            state = moveWeightOffset2Bias(filter, input_tensor, bias_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "move weight failure\n");
        }
    }

    // calculate quantization parameter
    double real_multiplier = 0.0;
    GetQuantizedConvolutionMultipler(
        input_tensor->getScale(), filter_->getScale(), bias_->getScale(), output_tensor->getScale(), &real_multiplier);

    QuantizeMultiplierSmallerThanOneExp(real_multiplier, &output_multiplier_, &output_shift_);

    output_shift_ *= -1;
    input_offset_ = -input_tensor->getZeroPoint();

    return gemm1xXConv2DGPU(input_tensor, output_tensor);
}

Status CLGEMM1xXConvolutionQuantized::moveWeightOffset2Bias(const std::shared_ptr<CLTensor> weight_tensor,
                                                            const std::shared_ptr<CLTensor> input_tensor,
                                                            std::shared_ptr<CLTensor> &bias_tensor) {
    Status state = Status::FAILURE;
    int input_zero_point = (int)input_tensor->getZeroPoint();
    int filter_zero_point = (int)weight_tensor->getZeroPoint();
    size_t global_size = unaligned_weight_height_;
    state = runtime_->setKernelArg(weight_offset_kernel_.get(),
                                   weight_tensor->getDataPtr(),
                                   bias_tensor->getDataPtr(),
                                   align_weight_width_makalu_,
                                   unaligned_weight_width_,
                                   input_zero_point,
                                   filter_zero_point);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setArg weight_offset_kernel_ failure\n");
    state = runtime_->enqueueKernel(weight_offset_kernel_.get(), 1, &global_size, NULL);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute weight_offset_kernel_ failure\n");

    return state;
}

Status CLGEMM1xXConvolutionQuantized::alignWeight1xXMakalu(const cl_mem &src,
                                                           cl_mem &dst,
                                                           int src_width,
                                                           int src_height,
                                                           int dst_width,
                                                           int dst_height,
                                                           int computed_top) {
    cl_mem unaligned_src = reinterpret_cast<cl_mem>(src);
    cl_mem aligned_dst = reinterpret_cast<cl_mem>(dst);
    size_t global_align_weight[2];
    global_align_weight[0] = dst_height / 8;
    global_align_weight[1] = dst_width * 8;
    Status state = runtime_->setKernelArg(
        align_weight1xXmakalu_kernel_.get(), unaligned_src, aligned_dst, src_height, src_width, dst_width * 8);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setArg kernel failure\n");
    state = runtime_->enqueueKernel(align_weight1xXmakalu_kernel_.get(), 2, global_align_weight, NULL);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
    return Status::SUCCESS;
}

Status CLGEMM1xXConvolutionQuantized::gemm1xXConv2DGPU(const std::shared_ptr<CLTensor> input,
                                                       std::shared_ptr<CLTensor> output) {
    // do pad and img2col
    ITensor::placeholder("do pad and img2col");

    Status state;
    state = pad_convert_executor_->quantizedIm2colAlign1xXMakalu(input,
                                                                 convert_output_,
                                                                 padding_,
                                                                 kernel_.h,
                                                                 kernel_.w,
                                                                 stride_.h,
                                                                 stride_.w,
                                                                 coalescing_feature_height_,
                                                                 input->getZeroPoint(),
                                                                 output_dim_.h,
                                                                 output_dim_.w);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "pad convert failure\n");

    // gemm
    std::shared_ptr<CLTensor> weight_buf_used = weight_buffer_makalu_;
    cl_mem convert_buffer = convert_output_->getDataPtr();
    cl_mem top_buffer = output->getDataPtr();
    cl_mem weight_buffer = weight_buf_used->getDataPtr();
    cl_mem bias_buffer = bias_->getDataPtr();
    int aligned_cinKK = convert_out_dim_.w;
    int top_hw = output_dim_.h * output_dim_.w;
    int top_channel = output_dim_.c;
    int bottom_step = convert_out_dim_.h * convert_out_dim_.w;

    int32_t act_min;
    int32_t act_max;

    if (activation_info_.isEnabled() == false) {
        if (signed_quant_) {
            act_min = std::numeric_limits<int8_t>::min();
            act_max = std::numeric_limits<int8_t>::max();
        } else {
            act_min = std::numeric_limits<uint8_t>::min();
            act_max = std::numeric_limits<uint8_t>::max();
        }

    } else {
        if (signed_quant_) {
            CalculateActivationRangeInt8(
                activation_info_.activation(), output->getScale(), output->getZeroPoint(), &act_min, &act_max);
        } else {
            CalculateActivationRangeUint8(
                activation_info_.activation(), output->getScale(), output->getZeroPoint(), &act_min, &act_max);
        }
    }

    size_t local_conv[3] = {1, 2, 24};
    if (runtime_->isValhall()) {
        local_conv[2] = 8;
    }
    size_t global_conv[3];
    global_conv[0] = output_dim_.n;
    global_conv[1] = alignTo(ceil(static_cast<double>(top_channel) / computed_top_number_), local_conv[1]);
    global_conv[2] = alignTo(ceil(static_cast<double>(top_hw) / GEMM1XX_TILE_SIZE_4), local_conv[2]);

    if (signed_quant_) {
        state = runtime_->setKernelArg(quantized_gemm1xX_kernel_.get(),
                                       convert_buffer,
                                       weight_buffer,
                                       bias_buffer,
                                       top_buffer,
                                       aligned_cinKK,
                                       top_channel,
                                       top_hw,
                                       bottom_step,
                                       coalescing_feature_height_,
                                       (char)input->getZeroPoint(),
                                       (char)filter_->getZeroPoint(),
                                       output->getZeroPoint(),
                                       output_multiplier_,
                                       -output_shift_,
                                       act_min,
                                       act_max);
    } else {
        state = runtime_->setKernelArg(quantized_gemm1xX_kernel_.get(),
                                       convert_buffer,
                                       weight_buffer,
                                       bias_buffer,
                                       top_buffer,
                                       aligned_cinKK,
                                       top_channel,
                                       top_hw,
                                       bottom_step,
                                       coalescing_feature_height_,
                                       (unsigned char)input->getZeroPoint(),
                                       (unsigned char)filter_->getZeroPoint(),
                                       output->getZeroPoint(),
                                       output_multiplier_,
                                       -output_shift_,
                                       act_min,
                                       act_max);
    }

    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setArg kernel failure\n");
    state = runtime_->enqueueKernel(quantized_gemm1xX_kernel_.get(), 3, global_conv, local_conv);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
    ENN_DBG_PRINT("CLGEMM1xXConvolutionQuantized is finished");
    return Status::SUCCESS;
}

Status CLGEMM1xXConvolutionQuantized::dilationWeight(const std::shared_ptr<CLTensor> weight_tensor) {
    CLTensor::checknull(weight_tensor,"weight_tensor");
    uint32_t batch_weight = weight_tensor->getDim().n;
    uint32_t channel_weight = weight_tensor->getDim().c;
    uint32_t kernel_height = weight_tensor->getDim().h;
    uint32_t kernel_width = weight_tensor->getDim().w;

    uint32_t extend_height = dilation_.h * (kernel_height - 1) + 1;
    uint32_t extend_width = dilation_.w * (kernel_width - 1) + 1;
    Dim4 dim_extent = {batch_weight, channel_weight, extend_height, extend_width};
    auto weight_dilation = std::make_shared<CLTensor>(runtime_,
                                                      precision_,
                                                      weight_tensor->getDataType(),
                                                      dim_extent,
                                                      weight_tensor->getDataOrder(),
                                                      weight_tensor->getScale(),
                                                      weight_tensor->getZeroPoint());

    size_t global_init = batch_weight * channel_weight * extend_height * extend_width;
    std::shared_ptr<struct _cl_kernel> kernel_dilation_init;
    Status state = runtime_->setKernel(&kernel_dilation_init, "dilation_init", precision_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
    state = runtime_->setKernelArg(kernel_dilation_init.get(), weight_dilation->getDataPtr(), weight_tensor->getZeroPoint());
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");
    state = runtime_->enqueueKernel(kernel_dilation_init.get(), (cl_uint)1, &global_init, NULL);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "enqueueKernel failure\n");

    size_t global[3];
    global[0] = batch_weight * channel_weight;
    global[1] = kernel_height;
    global[2] = kernel_width;
    std::shared_ptr<struct _cl_kernel> kernel_dilation;
    state = runtime_->setKernel(&kernel_dilation, "dilation", precision_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
    state = runtime_->setKernelArg(kernel_dilation.get(),
                                   weight_tensor->getDataPtr(),
                                   weight_dilation->getDataPtr(),
                                   extend_height,
                                   extend_width,
                                   dilation_.h,
                                   dilation_.w);

    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");
    state = runtime_->enqueueKernel(kernel_dilation.get(), (cl_uint)3, global, NULL);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "enqueueKernel failure\n");

    dilation_filter_ = weight_dilation;

    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
