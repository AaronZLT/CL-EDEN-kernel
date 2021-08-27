#include "userdriver/gpu/operators/cl_optimized_impls/CLGEMMConvolutionQuantized.hpp"
#include "userdriver/common/operator_interfaces/common/Debug.hpp"

namespace enn {
namespace ud {
namespace gpu {

CLGEMMConvolutionQuantized::CLGEMMConvolutionQuantized(const std::shared_ptr<CLRuntime> runtime,
                                                       const PrecisionType &precision) :
    runtime_(runtime),
    precision_(precision) {
    ENN_DBG_PRINT("CLGEMMConvolutionQuantized is created");
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
    weights_as_input_ = false;
    src_unit_count_ = 0;
    dst_align_count_ = 0;
    group_top_channel_ = 0;
    aligned_group_top_channel_ = 0;
    filter_offset_ = 0;
    androidNN_ = false;
    signed_quant_ = false;
}

Status CLGEMMConvolutionQuantized::initialize(const Dim4 &input_dim,
                                              const Dim4 &output_dim,
                                              const Pad4 &padding,
                                              const Dim2 &stride,
                                              const uint32_t &group_size,
                                              const std::shared_ptr<ITensor> weight,
                                              const std::shared_ptr<ITensor> bias,
                                              const ActivationInfo &activate_info,
                                              const bool &weights_as_input,
                                              const bool &androidNN) {
    ENN_DBG_PRINT("CLGEMMConvolutionQuantized::initialize() is called");
    Status state = Status::FAILURE;
    input_dim_ = input_dim;
    output_dim_ = output_dim;
    signed_quant_ = precision_ == PrecisionType::INT8;
    group_ = group_size;
    stride_ = stride;
    padding_ = padding;
    activation_info_ = activate_info;
    weights_as_input_ = weights_as_input;
    androidNN_ = androidNN;

    auto weight_tensor = std::static_pointer_cast<CLTensor>(weight);
    filter_ = weight_tensor;

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
        if (!weights_as_input_) {
            Status state = weight_tensor->convertToNCHW(weight_nchw_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "convertToNCHW failure\n");
        }
    }

    kernel_ = {weight_size.h, weight_size.w};

    auto bias_tensor = std::static_pointer_cast<CLTensor>(bias);
    bias_ = bias_tensor;

    // align weight
    uint32_t align_weight_height = (kernel_.h * kernel_.w * input_dim_.c / group_size + 15) / 16 * 16;
    uint32_t align_weight_width = (output_dim_.c / group_size + 15) / 16 * 16 * group_size;
    uint32_t weightCount = align_weight_height * align_weight_width;

    pad_convert_executor_ = std::make_shared<CLPadConvert>(runtime_, precision_);
    pad_convert_executor_->initialize(
        input_dim_, padding, kernel_, stride_, group_size, output_dim, true, CLPadConvert::PadConvertType::GEMM);
    uint32_t align_input_height = (output_dim_.h * output_dim_.w + 11) / 12 * 12 * group_size;
    uint32_t align_input_width = (kernel_.h * kernel_.w * input_dim_.c / group_size + 15) / 16 * 16;

    convert_out_dim_.n = input_dim_.n;
    convert_out_dim_.c = 1;
    convert_out_dim_.h = align_input_height;
    convert_out_dim_.w = align_input_width;
    if (signed_quant_) {
        convert_output_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::INT8, convert_out_dim_);
    } else {
        convert_output_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::UINT8, convert_out_dim_);
    }

    Dim4 convert_filter_dim = {weightCount, 1, 1, 1};
    converted_filter_ = std::make_shared<CLTensor>(runtime_, PrecisionType::FP16, DataType::FLOAT, convert_filter_dim);
    cl_mem converted_filter_data = converted_filter_->getDataPtr();

    src_unit_count_ = weight_size.c * weight_size.h * weight_size.w;
    dst_align_count_ = alignTo(src_unit_count_, 16);
    group_top_channel_ = weight_size.n / group_size;
    aligned_group_top_channel_ = alignTo(group_top_channel_, 16);
    filter_offset_ = -weight_tensor->getZeroPoint();

    if (signed_quant_) {
        state = runtime_->setKernel(&align_quantized_weight_kernel_, "SIGNEDalignQuantizedWeight_gemm", precision_);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel failure\n");
        state = runtime_->setKernel(&quantized_gemm_kernel_, "SIGNEDquantizedGemmBlocked", precision_);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel failure\n");
    } else {
        state = runtime_->setKernel(&align_quantized_weight_kernel_, "alignQuantizedWeight_gemm", precision_);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel failure\n");
        state = runtime_->setKernel(&quantized_gemm_kernel_, "quantizedGemmBlocked", precision_);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel failure\n");
    }

    if (weights_as_input_ == false) {
        state = alignQuantizedWeightGEMM(weight_tensor->getDataPtr(),
                                         converted_filter_data,
                                         src_unit_count_,
                                         weight_size.n,
                                         dst_align_count_,
                                         group_top_channel_,
                                         aligned_group_top_channel_,
                                         filter_offset_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "align weight failure\n");
    }

    return state;
}

Status CLGEMMConvolutionQuantized::execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) {
    ENN_DBG_PRINT("CLGEMMConvolutionQuantized::execute() is called");
    auto input_tensor = std::static_pointer_cast<CLTensor>(input);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output);

    if (weights_as_input_ == true) {
        Status state = Status ::FAILURE;
        auto converted_filter_data = converted_filter_->getDataPtr();

        if (androidNN_) {
            state = filter_->convertToNCHW(weight_nchw_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "convertToNCHW failure\n");
        }
        auto filter = androidNN_ ? weight_nchw_ : filter_;
        state = alignQuantizedWeightGEMM(filter->getDataPtr(),
                                         converted_filter_data,
                                         src_unit_count_,
                                         filter->getDim().n,
                                         dst_align_count_,
                                         group_top_channel_,
                                         aligned_group_top_channel_,
                                         filter_offset_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "align weight failure\n");
    }

    // calculate quantization parameter
    double real_multiplier = 0.0;
    GetQuantizedConvolutionMultipler(
        input_tensor->getScale(), filter_->getScale(), bias_->getScale(), output_tensor->getScale(), &real_multiplier);
    QuantizeMultiplierSmallerThanOneExp(real_multiplier, &output_multiplier_, &output_shift_);
    output_shift_ *= -1;
    input_offset_ = -input_tensor->getZeroPoint();

    return quantizedConv2DGPU(input_tensor, output_tensor);
}

Status CLGEMMConvolutionQuantized::alignQuantizedWeightGEMM(const cl_mem &src,
                                                            cl_mem &dst,
                                                            const uint32_t &src_width,
                                                            const uint32_t &src_height,
                                                            const uint32_t &dst_width,
                                                            const uint32_t &group_top_channel,
                                                            const uint32_t &aligned_group_top_channel,
                                                            int filter_offset) {
    ENN_DBG_PRINT("CLGEMMConvolutionQuantized is in alignWeightGEMM");
    auto unaligned_src = src;
    auto aligned_dst = dst;
    size_t local_align_weight[2] = {1, 12};
    size_t global_align_weight[2];
    global_align_weight[0] = src_height;
    global_align_weight[1] = alignTo(ceil(src_width / 8.0), local_align_weight[1]);
    Status state = runtime_->setKernelArg(align_quantized_weight_kernel_.get(),
                                          unaligned_src,
                                          aligned_dst,
                                          src_width,
                                          dst_width,
                                          group_top_channel,
                                          aligned_group_top_channel,
                                          filter_offset);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "set kernel failure\n");
    state = runtime_->enqueueKernel(align_quantized_weight_kernel_.get(), 2, global_align_weight, local_align_weight);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
    return Status::SUCCESS;
}

Status CLGEMMConvolutionQuantized::quantizedGEMMRun(const std::shared_ptr<CLTensor> input,
                                                    std::shared_ptr<CLTensor> output) {
    auto input_data = input->getDataPtr();
    auto output_data = output->getDataPtr();
    cl_mem weight_data = converted_filter_->getDataPtr();
    cl_mem bias_data = bias_->getDataPtr();
    auto input_dim = input->getDim();
    auto output_dim = output->getDim();
    int group = group_;
    uint32_t align_input_width = input_dim.w;
    uint32_t align_input_height = input_dim.h;
    uint32_t align_weight_width = alignTo(output_dim.c / group, 16) * group;
    uint32_t temp_width = output_dim.h * output_dim.w;
    uint32_t out_channel = output_dim.c;
    size_t global_conv[3];
    global_conv[0] = output_dim.n;
    global_conv[1] = align_weight_width / group / 4;
    global_conv[2] = align_input_height / group;
    size_t local_conv[3] = {1, 1, 12};
    int output_offset = output->getZeroPoint();

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

    Status state;
    for (int i = 0; i < group; i++) {
        state = runtime_->setKernelArg(quantized_gemm_kernel_.get(),
                                       input_data,
                                       weight_data,
                                       bias_data,
                                       output_data,
                                       align_input_width,
                                       out_channel,
                                       temp_width,
                                       group,
                                       i,
                                       input_offset_,
                                       output_offset,
                                       output_multiplier_,
                                       output_shift_,
                                       act_min,
                                       act_max);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "set kernelArg failure\n");
        state = runtime_->enqueueKernel(quantized_gemm_kernel_.get(), 3, global_conv, local_conv);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
    }
    return Status::SUCCESS;
}

Status CLGEMMConvolutionQuantized::quantizedConv2DGPU(const std::shared_ptr<CLTensor> input,
                                                      std::shared_ptr<CLTensor> output) {
    // do pad and img2col
    Status state;
    state = pad_convert_executor_->execute(input, convert_output_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "pad convert failure\n");

    // gemm
    state = quantizedGEMMRun(convert_output_, output);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "gemm failure\n");
    ENN_DBG_PRINT("CLGEMMConvolutionQuantized is finished");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
