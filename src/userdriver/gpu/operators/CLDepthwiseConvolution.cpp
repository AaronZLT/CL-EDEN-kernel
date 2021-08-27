#include "userdriver/gpu/operators/CLDepthwiseConvolution.hpp"

namespace enn {
namespace ud {
namespace gpu {

namespace {
const uint32_t INPUT_INDEX = 0;
const uint32_t WEIGHT_INDEX = 1;
const uint32_t BIAS_INDEX = 2;
const uint32_t OUTPUT_INDEX = 0;
}  // namespace

CLDepthwiseConvolution::CLDepthwiseConvolution(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    ENN_DBG_PRINT("CLDepthwiseConvolution is created");
    kernel_ = {0, 0};
    pad_ = {0, 0, 0, 0};
    stride_ = {0, 0};
    dialation_kernel_ = {0, 0};
    dilation_ = {1, 1};
    empty_bias_ = false;
    pad_buffer_h_ = 0;
    pad_buffer_w_ = 0;
    size_pad_ = 0;
    weights_as_input_ = false;
    ActivationInfo activation_info_ = ActivationInfo();
}

Status CLDepthwiseConvolution::dilation_weight(const std::shared_ptr<CLTensor> weight_tensor) {
    uint32_t batch_weight = weight_tensor->getDim().n;
    uint32_t channel_weight = weight_tensor->getDim().c;
    uint32_t extent_height = dilation_.h * (kernel_.h - 1) + 1;
    uint32_t extent_width = dilation_.w * (kernel_.w - 1) + 1;
    uint32_t weight_dilation_size = batch_weight * channel_weight * extent_height * extent_width;
    Dim4 dim_extent = {batch_weight, channel_weight, extent_height, extent_width};

    Dim4 weight_dilation_dim = {weight_dilation_size, 1, 1, 1};
    auto weight_dilation = std::make_shared<CLTensor>(runtime_,
                                                      precision_,
                                                      weight_tensor->getDataType(),
                                                      weight_dilation_dim,
                                                      weight_tensor->getDataOrder(),
                                                      weight_tensor->getScale(),
                                                      weight_tensor->getZeroPoint());

    size_t global[3];
    global[0] = batch_weight * channel_weight;
    global[1] = kernel_.h;
    global[2] = kernel_.w;
    std::shared_ptr<struct _cl_kernel> kernel_dilation;
    Status state = runtime_->setKernel(&kernel_dilation, "dilation", precision_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
    state = runtime_->setKernelArg(kernel_dilation.get(),
                                   weight_tensor->getDataPtr(),
                                   weight_dilation->getDataPtr(),
                                   extent_height,
                                   extent_width,
                                   dilation_.h,
                                   dilation_.w);

    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");
    state = runtime_->enqueueKernel(kernel_dilation.get(), (cl_uint)3, global, NULL);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "enqueueKernel failure\n");

    auto dilation_weight_tensor = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, dim_extent);

    size_t offset = 0;
    size_t type_size = runtime_->getRuntimeTypeBytes(precision_);
    size_t src_offset_bytes = type_size * offset;
    size_t dst_offset_bytes = type_size * offset;
    size_t size_bytes = type_size * (size_t)weight_dilation_size;

    runtime_->copyBuffer(
        dilation_weight_tensor->getDataPtr(), weight_dilation->getDataPtr(), dst_offset_bytes, src_offset_bytes, size_bytes);

    dilation_filter_ = weight_dilation;

    return Status::SUCCESS;
}

bool CLDepthwiseConvolution::Is_depthwise_conv3x3_supported() {
    return weight_->getDim().n == 1 && dilation_.w == 1 && dilation_.h == 1 && weight_->getDim().w == 3 &&
           weight_->getDim().h == 3 && stride_.w == 1 && stride_.h == 1 && pad_.t == 1 && pad_.l == 1 && pad_.r == 1 &&
           pad_.b == 1;
}

Status CLDepthwiseConvolution::initialize(std::vector<std::shared_ptr<ITensor>> inputs,
                                          std::vector<std::shared_ptr<ITensor>> outputs,
                                          std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLDepthwiseConvolution::initialize() is called");

    input_ = std::static_pointer_cast<CLTensor>(inputs.at(INPUT_INDEX));
    weight_ = std::static_pointer_cast<CLTensor>(inputs.at(WEIGHT_INDEX));
    bias_ = std::static_pointer_cast<CLTensor>(inputs.at(BIAS_INDEX));
    output_ = std::static_pointer_cast<CLTensor>(outputs.at(OUTPUT_INDEX));

    parameters_ = std::static_pointer_cast<DepthwiseConvolutionParameters>(parameters);
    ENN_DBG_PRINT("DepthwiseConvolutionParameters: padding.l %d padding.r %d padding.t %d padding.b %d; dilation.h %d "
                  "dilation.w %d; stride.h %d  stride.w %d; depth_multiplier %d; per_channel_quant %d; "
                  "androidNN %d; isNCHW %d; storage_type %d; activation_info_: isEnabled() %d,activation() %d;\n",
                  parameters_->padding.l,
                  parameters_->padding.r,
                  parameters_->padding.t,
                  parameters_->padding.b,
                  parameters_->dilation.h,
                  parameters_->dilation.w,
                  parameters_->stride.h,
                  parameters_->stride.w,
                  parameters_->depth_multiplier,
                  parameters_->per_channel_quant,
                  parameters_->androidNN,
                  parameters_->isNCHW,
                  parameters_->storage_type,
                  parameters_->activation_info->isEnabled(),
                  parameters_->activation_info->activation());

    pad_ = parameters_->padding;
    dilation_ = parameters_->dilation;
    stride_ = parameters_->stride;
    weights_as_input_ = !weight_->is_const();
    activation_info_ = *parameters_->activation_info.get();
    kernel_.h = weight_->getDim().h;
    kernel_.w = weight_->getDim().w;

    Dim4 input_dim = input_->getDim();
    Dim4 output_dim = output_->getDim();
    if (parameters_->androidNN && parameters_->storage_type != StorageType::TEXTURE) {
        kernel_.h = weight_->getDim().c;
        kernel_.w = weight_->getDim().h;
        dialation_kernel_.h = dilation_.h * (kernel_.h - 1) + 1;
        dialation_kernel_.w = dilation_.w * (kernel_.w - 1) + 1;

        if (!parameters_->isNCHW) {
            input_dim = convertDimToNCHW(input_dim);
            input_nchw_ = std::make_shared<CLTensor>(runtime_,
                                                     precision_,
                                                     input_->getDataType(),
                                                     input_dim,
                                                     input_->getDataOrder(),
                                                     input_->getScale(),
                                                     input_->getZeroPoint());
        }
        output_dim.n = input_dim.n;
        output_dim.c = weight_->getDim().w;
        output_dim.h = (input_dim.h + pad_.b + pad_.t - dialation_kernel_.h + stride_.h) / stride_.h;
        output_dim.w = (input_dim.w + pad_.l + pad_.r - dialation_kernel_.w + stride_.w) / stride_.w;
        if (!parameters_->isNCHW) {
            output_nchw_ = std::make_shared<CLTensor>(runtime_,
                                                      precision_,
                                                      output_->getDataType(),
                                                      output_dim,
                                                      output_->getDataOrder(),
                                                      output_->getScale(),
                                                      output_->getZeroPoint());
        }
        Dim4 expected_output_dim = output_dim;
        if (!parameters_->isNCHW) {
            expected_output_dim = convertDimToNHWC(output_dim);
        }
        if (!isDimsSame(expected_output_dim, output_->getDim())) {
            output_->reconfigureDimAndBuffer(expected_output_dim);
        }
    } else {
        kernel_.h = weight_->getDim().h;
        kernel_.w = weight_->getDim().w;

        dialation_kernel_.h = dilation_.h * (kernel_.h - 1) + 1;
        dialation_kernel_.w = dilation_.w * (kernel_.w - 1) + 1;
    }

    if (precision_ == PrecisionType::UINT8 || precision_ == PrecisionType::INT8) {
        if (parameters_->per_channel_quant == true) {
            per_channel_quantized_dw_ = std::make_shared<CLDepthwiseConvolutionPerChannelQuantized>(runtime_, precision_);
            return per_channel_quantized_dw_->initialize(input_dim,
                                                         output_dim,
                                                         weight_,
                                                         bias_,
                                                         stride_,
                                                         parameters_->padding,
                                                         parameters_->depth_multiplier,
                                                         activation_info_,
                                                         parameters_->scales,
                                                         weights_as_input_,
                                                         parameters_->androidNN);
        } else {
            quantized_dw_ = std::make_shared<CLDepthwiseConvolutionQuantized>(runtime_, precision_);
            return quantized_dw_->initialize(input_dim,
                                             output_dim,
                                             weight_,
                                             bias_,
                                             stride_,
                                             parameters_->padding,
                                             parameters_->depth_multiplier,
                                             dilation_,
                                             activation_info_,
                                             weights_as_input_,
                                             parameters_->androidNN);
        }
    }

    Status state;
    auto weight_tensor = std::static_pointer_cast<CLTensor>(weight_);
    Dim4 weight_dim = weight_tensor->getDim();
    if (parameters_->androidNN && parameters_->storage_type != StorageType::TEXTURE) {
        weight_dim = convertDimToNCHW(weight_dim);
        filter_nchw_ = std::make_shared<CLTensor>(runtime_,
                                                  precision_,
                                                  weight_tensor->getDataType(),
                                                  weight_dim,
                                                  weight_tensor->getDataOrder(),
                                                  weight_tensor->getScale(),
                                                  weight_tensor->getZeroPoint());
        if (!weights_as_input_) {
            state = weight_tensor->convertToNCHW(filter_nchw_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "convertToNCHW failure\n");
        }
    }

    if (parameters_->storage_type == StorageType::TEXTURE &&
        (precision_ == PrecisionType::FP16 || precision_ == PrecisionType::FP32)) {
        if (precision_ == PrecisionType::FP16 && Is_depthwise_conv3x3_supported() && weights_as_input_ == false &&
            parameters_->per_channel_quant == false && parameters_->depth_multiplier == 1 &&
            (activation_info_.activation() == ActivationInfo::ActivationType::RELU ||
             activation_info_.activation() == ActivationInfo::ActivationType::RELU6 ||
             activation_info_.activation() == ActivationInfo::ActivationType::NONE)) {
            dw_tflite_3x3_ = std::make_shared<CLDepthwiseConvolution3x3TFLite>(runtime_, precision_);
            return dw_tflite_3x3_->initialize(
                input_dim, output_dim, weight_, bias_, stride_, parameters_->padding, activation_info_);
        } else if (precision_ == PrecisionType::FP16 && kernel_.h == 3 && kernel_.w == 3 && output_->getDim().n == 1 &&
                   weights_as_input_ == false && parameters_->per_channel_quant == false &&
                   parameters_->depth_multiplier == 1 &&
                   (activation_info_.activation() == ActivationInfo::ActivationType::RELU ||
                    activation_info_.activation() == ActivationInfo::ActivationType::RELU6 ||
                    activation_info_.activation() == ActivationInfo::ActivationType::NONE)) {
            dw_tflite_ = std::make_shared<CLDepthwiseConvolutionTFLite>(runtime_, precision_);
            return dw_tflite_->initialize(
                input_dim, output_dim, weight_, bias_, stride_, parameters_->padding, dilation_, activation_info_);
        }
    }

    if ((parameters_->depth_multiplier != 1 || dilation_.h != 2 || dilation_.w != 2 || kernel_.h != 3 || kernel_.w != 3 ||
         stride_.h != 1 || stride_.w != 1) &&
        (parameters_->depth_multiplier != 1 || dilation_.h != 4 || dilation_.w != 4 || kernel_.h != 3 || kernel_.w != 3 ||
         stride_.h != 1 || stride_.w != 1) &&
        (parameters_->depth_multiplier != 1 || kernel_.h != 3 || kernel_.w != 3 || stride_.h != 1 || stride_.w != 1) &&
        (parameters_->depth_multiplier != 1 || kernel_.h != 3 || kernel_.w != 3 || stride_.h != 2 || stride_.w != 2) &&
        (pad_.l != 0 || pad_.t != 0 || pad_.r != 0 || pad_.b != 0)) {
        state = runtime_->setKernel(&kernel_pad_, "pad4", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "pad setKernel failure\n");

        pad_buffer_h_ = input_dim.h + pad_.t + pad_.b;
        pad_buffer_w_ = input_dim.w + pad_.l + pad_.r;
        size_pad_ = pad_buffer_h_ * pad_buffer_w_;
        Dim4 dim_pad = {input_dim.n, input_dim.c, pad_buffer_h_, pad_buffer_w_};
        pad_buffer_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, dim_pad);
    }

    uint32_t bias_sz = bias_->getNumOfBytes();
    if (!bias_sz) {
        Dim4 bias_dim = {1, output_dim.c, 1, 1};
        empty_bias_buffer_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, bias_dim);
        empty_bias_ = true;
    }

    if (parameters_->depth_multiplier == 1 && dilation_.h == 2 && dilation_.w == 2 && kernel_.h == 3 && kernel_.w == 3 &&
        stride_.h == 1 && stride_.w == 1) {
        if (activation_info_.activation() == ActivationInfo::ActivationType::RELU) {
            state = runtime_->setKernel(&kernel_depthwise_, "RELUdepthwise_conv_3x3d2s1_pad_merge", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "RELUdepthwise_conv_3x3d2s1_pad_merge setKernel failure\n");
        } else if (activation_info_.activation() == ActivationInfo::ActivationType::RELU6) {
            state = runtime_->setKernel(&kernel_depthwise_, "RELU6depthwise_conv_3x3d2s1_pad_merge", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "RELU6depthwise_conv_3x3d2s1_pad_merge setKernel failure\n");
        } else {
            state = runtime_->setKernel(&kernel_depthwise_, "depthwise_conv_3x3d2s1_pad_merge", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv_3x3d2s1_pad_merge setKernel failure\n");
        }
    } else if (parameters_->depth_multiplier == 1 && dilation_.h == 4 && dilation_.w == 4 && kernel_.h == 3 &&
               kernel_.w == 3 && stride_.h == 1 && stride_.w == 1) {
        if (activation_info_.activation() == ActivationInfo::ActivationType::RELU) {
            state = runtime_->setKernel(&kernel_depthwise_, "RELUdepthwise_conv_3x3d4s1_pad_merge", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "RELUdepthwise_conv_3x3d4s1_pad_merge setKernel failure\n");
        } else if (activation_info_.activation() == ActivationInfo::ActivationType::RELU6) {
            state = runtime_->setKernel(&kernel_depthwise_, "RELU6depthwise_conv_3x3d4s1_pad_merge", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "RELU6depthwise_conv_3x3d4s1_pad_merge setKernel failure\n");
        } else {
            state = runtime_->setKernel(&kernel_depthwise_, "depthwise_conv_3x3d4s1_pad_merge", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv_3x3d4s1_pad_merge setKernel failure\n");
        }
    } else if (parameters_->depth_multiplier == 1 && dilation_.h == 1 && dilation_.w == 1 && kernel_.h == 3 &&
               kernel_.w == 3 && stride_.h == 1 && stride_.w == 1) {
        if (activation_info_.activation() == ActivationInfo::ActivationType::RELU) {
            state = runtime_->setKernel(&kernel_depthwise_, "RELUdepthwise_conv_3x3s1_pad_merge", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "RELUdepthwise_conv_3x3s1_pad_merge setKernel failure\n");
        } else if (activation_info_.activation() == ActivationInfo::ActivationType::RELU6) {
            state = runtime_->setKernel(&kernel_depthwise_, "RELU6depthwise_conv_3x3s1_pad_merge", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "RELU6depthwise_conv_3x3s1_pad_merge setKernel failure\n");
        } else {
            state = runtime_->setKernel(&kernel_depthwise_, "depthwise_conv_3x3s1_pad_merge", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv_3x3s1_pad_merge setKernel failure\n");
        }
    } else if (parameters_->depth_multiplier == 1 && dilation_.h == 1 && dilation_.w == 1 && kernel_.h == 3 &&
               kernel_.w == 3 && stride_.h == 2 && stride_.w == 2) {
        if (activation_info_.activation() == ActivationInfo::ActivationType::RELU) {
            state = runtime_->setKernel(&kernel_depthwise_, "RELUdepthwise_conv_3x3s2_pad_merge", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "RELUdepthwise_conv_3x3s2_pad_merge setKernel failure\n");
        } else if (activation_info_.activation() == ActivationInfo::ActivationType::RELU6) {
            state = runtime_->setKernel(&kernel_depthwise_, "RELU6depthwise_conv_3x3s2_pad_merge", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "RELU6depthwise_conv_3x3s2_pad_merge setKernel failure\n");
        } else {
            state = runtime_->setKernel(&kernel_depthwise_, "depthwise_conv_3x3s2_pad_merge", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv_3x3s2_pad_merge setKernel failure\n");
        }
    } else if (dilation_.h == 1 && dilation_.w == 1 && kernel_.h == 3 && kernel_.w == 3 && stride_.h == 1 &&
               stride_.w == 1) {
        if (activation_info_.activation() == ActivationInfo::ActivationType::RELU) {
            state = runtime_->setKernel(&kernel_depthwise_, "RELUdepthwise_conv_3x3s1_4P", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "RELUdepthwise_conv_3x3s1_4P setKernel failure\n");
        } else if (activation_info_.activation() == ActivationInfo::ActivationType::RELU6) {
            state = runtime_->setKernel(&kernel_depthwise_, "RELU6depthwise_conv_3x3s1_4P", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "RELU6depthwise_conv_3x3s1_4P setKernel failure\n");
        } else {
            state = runtime_->setKernel(&kernel_depthwise_, "depthwise_conv_3x3s1_4P", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv_3x3s1_4P setKernel failure\n");
        }
    } else if (dilation_.h == 1 && dilation_.w == 1 && kernel_.h == 3 && kernel_.w == 3 && stride_.h == 2 &&
               stride_.w == 2) {
        if (activation_info_.activation() == ActivationInfo::ActivationType::RELU) {
            state = runtime_->setKernel(&kernel_depthwise_, "RELUdepthwise_conv_3x3s2", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "RELUdepthwise_conv_3x3s2 setKernel failure\n");
        } else if (activation_info_.activation() == ActivationInfo::ActivationType::RELU6) {
            state = runtime_->setKernel(&kernel_depthwise_, "RELU6depthwise_conv_3x3s2", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "RELU6depthwise_conv_3x3s2 setKernel failure\n");
        } else {
            state = runtime_->setKernel(&kernel_depthwise_, "depthwise_conv_3x3s2", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv_3x3s2 setKernel failure\n");
        }
    } else {
        // dilation weight
        if ((weights_as_input_ == false) && (dilation_.h > 1 || dilation_.w > 1) && !(kernel_.h == 1 && kernel_.w == 1)) {
            auto weight_tensor = std::static_pointer_cast<CLTensor>(parameters_->androidNN ? filter_nchw_ : weight_);
            dilation_weight(weight_tensor);
        }

        if (activation_info_.activation() == ActivationInfo::ActivationType::RELU) {
            state = runtime_->setKernel(&kernel_depthwise_, "RELUdepthwise_conv", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "RELUdepthwise_conv setKernel failure\n");
        } else if (activation_info_.activation() == ActivationInfo::ActivationType::RELU6) {
            state = runtime_->setKernel(&kernel_depthwise_, "RELU6depthwise_conv", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "RELU6depthwise_conv setKernel failure\n");
        } else {
            state = runtime_->setKernel(&kernel_depthwise_, "depthwise_conv", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv setKernel failure\n");
        }
    }

    if (input_dim.c * parameters_->depth_multiplier < output_dim.c) {
        if (activation_info_.activation() == ActivationInfo::ActivationType::RELU) {
            state = runtime_->setKernel(&kernel_unequal_, "RELUdepthwise_conv_unequal", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "RELUdepthwise_conv_unequal setKernel failure\n");
        } else if (activation_info_.activation() == ActivationInfo::ActivationType::RELU6) {
            state = runtime_->setKernel(&kernel_unequal_, "RELU6depthwise_conv_unequal", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "RELU6depthwise_conv_unequal setKernel failure\n");
        } else {
            state = runtime_->setKernel(&kernel_unequal_, "depthwise_conv_unequal", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv_unequal setKernel failure\n");
        }
    }

    if (activation_info_.isEnabled()) {
        if (activation_info_.activation() == ActivationInfo::ActivationType::RELU ||
            activation_info_.activation() == ActivationInfo::ActivationType::RELU6) {
            activation_info_.disable();
        }
    }

    if (precision_ != PrecisionType::UINT8 && precision_ != PrecisionType::INT8) {
        cl_activation_ = std::make_shared<CLActivation>(runtime_, precision_);
        std::shared_ptr<ActivationParameters> act_parameters = std::make_shared<ActivationParameters>();
        act_parameters->activation_info = activation_info_;
        act_parameters->relu_parameters = std::make_shared<ReluParameters>();
        act_parameters->relu_parameters->negative_slope = 0.0f;
        state = cl_activation_->initialize({output_}, {output_}, act_parameters);
    }
    return state;
}

Status CLDepthwiseConvolution::execute() {
    ENN_DBG_PRINT("CLDepthwiseConvolution::execute() is called");
    // for zero_sized input
    if (input_->getTotalSizeFromDims() == 0) {
        return Status::SUCCESS;
    }

    auto input_tensor = std::static_pointer_cast<CLTensor>(input_);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output_);

    Status state = Status::FAILURE;
    if (parameters_->androidNN && !parameters_->isNCHW && parameters_->storage_type != StorageType::TEXTURE) {
        state = input_tensor->convertToNCHW(input_nchw_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "convertToNCHW failure\n");
        state = execute_nchw(input_nchw_, output_nchw_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute_nchw failure\n");
        state = output_nchw_->convertToNHWC(output_tensor);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "convertToNHWC failure\n");
    } else {
        state = execute_nchw(input_tensor, output_tensor);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute_nchw failure\n");
    }
    if (precision_ != PrecisionType::UINT8 && precision_ != PrecisionType::INT8) {
        state = cl_activation_->execute();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "cl_activation_->execute() failure\n");
    }

    return state;
}

Status CLDepthwiseConvolution::execute_nchw(const std::shared_ptr<CLTensor> input, std::shared_ptr<CLTensor> output) {
    if (precision_ == PrecisionType::UINT8 || precision_ == PrecisionType::INT8) {
        if (per_channel_quantized_dw_ != nullptr) {
            return per_channel_quantized_dw_->execute(input, output);
        } else {
            return quantized_dw_->execute(input, output);
        }
    }
    if (parameters_->storage_type == StorageType::TEXTURE &&
        (precision_ == PrecisionType::FP16 || precision_ == PrecisionType::FP32)) {
        if (precision_ == PrecisionType::FP16 && Is_depthwise_conv3x3_supported() && weights_as_input_ == false &&
            parameters_->per_channel_quant == false && parameters_->depth_multiplier == 1 &&
            (activation_info_.activation() == ActivationInfo::ActivationType::RELU ||
             activation_info_.activation() == ActivationInfo::ActivationType::RELU6 ||
             activation_info_.activation() == ActivationInfo::ActivationType::NONE)) {
            return dw_tflite_3x3_->execute(input, output);
        } else if (precision_ == PrecisionType::FP16 && kernel_.h == 3 && kernel_.w == 3 && output->getDim().n == 1 &&
                   weights_as_input_ == false && parameters_->per_channel_quant == false &&
                   parameters_->depth_multiplier == 1 &&
                   (activation_info_.activation() == ActivationInfo::ActivationType::RELU ||
                    activation_info_.activation() == ActivationInfo::ActivationType::RELU6 ||
                    activation_info_.activation() == ActivationInfo::ActivationType::NONE)) {
            return dw_tflite_->execute(input, output);
        }
    }

    Status state;
    if (weights_as_input_ && parameters_->androidNN) {
        auto weight_tensor = std::static_pointer_cast<CLTensor>(weight_);
        state = weight_tensor->convertToNCHW(filter_nchw_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "convertToNCHW failure\n");
    }

    if ((weights_as_input_ == true) && (dilation_.h > 1 || dilation_.w > 1)) {
        auto weight_tensor = std::static_pointer_cast<CLTensor>(parameters_->androidNN ? filter_nchw_ : weight_);
        state = dilation_weight(weight_tensor);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "dilation_weight failure\n");
    }

    auto input_tensor = std::static_pointer_cast<CLTensor>(input);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output);
    auto filter_tensor = std::static_pointer_cast<CLTensor>(parameters_->androidNN ? filter_nchw_ : weight_);
    auto bias_tensor = std::static_pointer_cast<CLTensor>(bias_);
    auto input_data = input_tensor->getDataPtr();
    auto output_data = output_tensor->getDataPtr();
    auto filter_data = filter_tensor->getDataPtr();

    cl_mem bias_data = nullptr;
    if (empty_bias_) {
        bias_data = empty_bias_buffer_->getDataPtr();
    } else {
        bias_data = bias_tensor->getDataPtr();
    }

    auto input_dim = input_tensor->getDim();
    auto output_dim = output_tensor->getDim();
    int pad_t = pad_.t;
    int pad_b = pad_.b;
    int pad_l = pad_.l;
    int pad_r = pad_.r;

    if (parameters_->depth_multiplier == 1 && dilation_.h == 2 && dilation_.w == 2 && kernel_.h == 3 && kernel_.w == 3 &&
        stride_.h == 1 && stride_.w == 1) {
        size_t global[3] = {0, 0, 0};
        size_t local[3] = {0, 0, 0};

        local[0] = 1;
        local[1] = 1;
        local[2] = 8;
        int align_height = alignTo(output_dim.h, local[1]);
        int align_width = alignTo(ceil(output_dim.w / 4.0), local[2]);
        global[0] = output_dim.n * input_dim.c;
        global[1] = align_height;
        global[2] = align_width;

        state = runtime_->setKernelArg(kernel_depthwise_.get(),
                                       input_data,
                                       filter_data,
                                       bias_data,
                                       output_data,
                                       input_dim.h,
                                       input_dim.w,
                                       output_dim.h,
                                       output_dim.w,
                                       output_dim.c,
                                       pad_.t,
                                       pad_.l);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv_3x3d2s1_pad_merge setKernelArg failure\n");

        state = runtime_->enqueueKernel(kernel_depthwise_.get(), (cl_uint)3, global, local);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "enqueueKernel failure\n");
    } else if (parameters_->depth_multiplier == 1 && dilation_.h == 4 && dilation_.w == 4 && kernel_.h == 3 &&
               kernel_.w == 3 && stride_.h == 1 && stride_.w == 1) {
        size_t global[3] = {0, 0, 0};
        size_t local[3] = {0, 0, 0};

        local[0] = 1;
        local[1] = 1;
        local[2] = 4;
        int align_height = alignTo(output_dim.h, local[1]);
        int align_width = alignTo(ceil(output_dim.w / 8.0), local[2]);
        global[0] = output_dim.n * input_dim.c;
        global[1] = align_height;
        global[2] = align_width;

        state = runtime_->setKernelArg(kernel_depthwise_.get(),
                                       input_data,
                                       filter_data,
                                       bias_data,
                                       output_data,
                                       input_dim.h,
                                       input_dim.w,
                                       output_dim.h,
                                       output_dim.w,
                                       output_dim.c,
                                       pad_.t,
                                       pad_.l);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv_3x3d4s1_pad_merge setKernelArg failure\n");

        state = runtime_->enqueueKernel(kernel_depthwise_.get(), (cl_uint)3, global, local);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "enqueueKernel failure\n");
    } else if (parameters_->depth_multiplier == 1 && kernel_.h == 3 && kernel_.w == 3 && stride_.h == 1 && stride_.w == 1) {
        size_t global[3] = {0, 0, 0};
        size_t local[3] = {0, 0, 0};

        local[0] = 1;
        local[1] = 1;
        local[2] = 16;
        int align_height = alignTo(output_dim.h, local[1]);
        int align_width = alignTo(ceil(output_dim.w / 2.0), local[2]);
        global[0] = output_dim.n * input_dim.c;
        global[1] = align_height;
        global[2] = align_width;

        state = runtime_->setKernelArg(kernel_depthwise_.get(),
                                       input_data,
                                       filter_data,
                                       bias_data,
                                       output_data,
                                       input_dim.h,
                                       input_dim.w,
                                       output_dim.h,
                                       output_dim.w,
                                       output_dim.c,
                                       pad_.t,
                                       pad_.l);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv_3x3s1_pad_merge setKernelArg failure\n");

        state = runtime_->enqueueKernel(kernel_depthwise_.get(), (cl_uint)3, global, local);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "enqueueKernel failure\n");
    } else if (parameters_->depth_multiplier == 1 && kernel_.h == 3 && kernel_.w == 3 && stride_.h == 2 && stride_.w == 2) {
        size_t global[3] = {0, 0, 0};
        size_t local[3] = {0, 0, 0};

        local[0] = 1;
        local[1] = 1;
        local[2] = 16;
        int align_height = alignTo(output_dim.h, local[1]);
        int align_width = alignTo(output_dim.w, local[2]);
        global[0] = output_dim.n * input_dim.c;
        global[1] = align_height;
        global[2] = align_width;

        state = runtime_->setKernelArg(kernel_depthwise_.get(),
                                       input_data,
                                       filter_data,
                                       bias_data,
                                       output_data,
                                       input_dim.h,
                                       input_dim.w,
                                       output_dim.h,
                                       output_dim.w,
                                       output_dim.c,
                                       pad_.t,
                                       pad_.l);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv_3x3s2_pad_merge setKernelArg failure\n");

        state = runtime_->enqueueKernel(kernel_depthwise_.get(), (cl_uint)3, global, local);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "enqueueKernel failure\n");
    } else if (pad_t == 0 && pad_b == 0 && pad_l == 0 && pad_r == 0) {
        if ((dilation_.h > 1 || dilation_.w > 1)) {
            state = excute_kernels_depends(input_data,
                                           dilation_filter_->getDataPtr(),
                                           bias_data,
                                           input_dim,
                                           output_dim,
                                           input_dim.h,
                                           input_dim.w,
                                           dialation_kernel_.h,
                                           dialation_kernel_.w,
                                           &output_data);
        } else {
            state = excute_kernels_depends(input_data,
                                           filter_data,
                                           bias_data,
                                           input_dim,
                                           output_dim,
                                           input_dim.h,
                                           input_dim.w,
                                           kernel_.h,
                                           kernel_.w,
                                           &output_data);
        }

        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "pad setKernelArg failure\n");
    } else {
        state = runtime_->setKernelArg(
            kernel_pad_.get(), input_data, pad_buffer_->getDataPtr(), pad_l, pad_r, pad_b, pad_t, input_dim.w, input_dim.h);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "pad setKernelArg failure\n");

        size_t global_pad[3] = {input_dim.n, input_dim.c, 0};
        size_t local_pad[3] = {1, 1, 32};
        global_pad[2] = alignTo(size_pad_, local_pad[2]);

        state = runtime_->enqueueKernel(kernel_pad_.get(), (cl_uint)3, global_pad, local_pad);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "pad enqueueKernel failure\n");

        if ((dilation_.h > 1 || dilation_.w > 1)) {
            state = excute_kernels_depends(pad_buffer_->getDataPtr(),
                                           dilation_filter_->getDataPtr(),
                                           bias_data,
                                           input_dim,
                                           output_dim,
                                           pad_buffer_h_,
                                           pad_buffer_w_,
                                           dialation_kernel_.h,
                                           dialation_kernel_.w,
                                           &output_data);
        } else {
            state = excute_kernels_depends(pad_buffer_->getDataPtr(),
                                           filter_data,
                                           bias_data,
                                           input_dim,
                                           output_dim,
                                           pad_buffer_h_,
                                           pad_buffer_w_,
                                           kernel_.h,
                                           kernel_.w,
                                           &output_data);
        }

        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "excute_kernels_depends failure\n");
    }

    return state;
}

Status CLDepthwiseConvolution::release() {
    ENN_DBG_PRINT("CLDepthwiseConvolution::release() is called");
    if (precision_ == PrecisionType::UINT8) {
        if (per_channel_quantized_dw_ != nullptr) {
            return per_channel_quantized_dw_->release();
        } else {
            return quantized_dw_->release();
        }
    }
    return Status::SUCCESS;
}

Status CLDepthwiseConvolution::excute_kernels_depends(const cl_mem &input_data,
                                                      const cl_mem &filter_data,
                                                      const cl_mem &bias_data,
                                                      const Dim4 &input_dim,
                                                      const Dim4 &output_dim,
                                                      const uint32_t &input_h,
                                                      const uint32_t &input_w,
                                                      const uint32_t &kernel_h,
                                                      const uint32_t &kernel_w,
                                                      cl_mem *output_data) {
    size_t global[3] = {0, 0, 0};
    size_t local[3] = {0, 0, 0};
    Status state;
    if (kernel_h == 3 && kernel_w == 3 && stride_.h == 1 && stride_.w == 1 && dilation_.h == 1 && dilation_.w == 1) {
        local[0] = 1;
        local[1] = 1;
        local[2] = 16;
        int align_height = alignTo(output_dim.h, 2 * local[1]);
        int align_width = alignTo(output_dim.w, 2 * local[2]);
        global[0] = output_dim.n * input_dim.c;
        global[1] = align_height / 2;
        global[2] = align_width / 2;

        state = runtime_->setKernelArg(kernel_depthwise_.get(),
                                       input_data,
                                       filter_data,
                                       bias_data,
                                       *output_data,
                                       input_h,
                                       input_w,
                                       output_dim.h,
                                       output_dim.w,
                                       output_dim.c,
                                       parameters_->depth_multiplier);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv_3x3s1_4P setKernelArg failure\n");
    } else if (kernel_h == 3 && kernel_w == 3 && stride_.h == 2 && stride_.w == 2 && dilation_.h == 1 && dilation_.w == 1) {
        local[0] = 1;
        local[1] = 1;
        local[2] = 32;
        int align_out = alignTo(output_dim.h * output_dim.w, local[2]);
        global[0] = output_dim.n;
        global[1] = input_dim.c;
        global[2] = align_out;

        state = runtime_->setKernelArg(kernel_depthwise_.get(),
                                       input_data,
                                       filter_data,
                                       bias_data,
                                       *output_data,
                                       input_h,
                                       input_w,
                                       output_dim.h,
                                       output_dim.w,
                                       output_dim.c,
                                       parameters_->depth_multiplier);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv_3x3s2 setKernel failure\n");
    } else {
        local[0] = 1;
        local[1] = 1;
        local[2] = 32;
        int align_out = alignTo(output_dim.h * output_dim.w, local[2]);
        global[0] = output_dim.n;
        global[1] = input_dim.c;
        global[2] = align_out;

        state = runtime_->setKernelArg(kernel_depthwise_.get(),
                                       input_data,
                                       filter_data,
                                       bias_data,
                                       *output_data,
                                       input_h,
                                       input_w,
                                       kernel_h,
                                       kernel_w,
                                       stride_.h,
                                       stride_.w,
                                       output_dim.h,
                                       output_dim.w,
                                       parameters_->depth_multiplier);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv setKernel failure\n");
    }
    state = runtime_->enqueueKernel(kernel_depthwise_.get(), (cl_uint)3, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "enqueueKernel failure\n");

    if (input_dim.c * parameters_->depth_multiplier < output_dim.c) {
        global[0] = output_dim.n;
        global[1] = output_dim.c - (input_dim.c * parameters_->depth_multiplier);
        global[2] = output_dim.h * output_dim.w;
        local[2] = findMaxFactor(global[2], 128);
        local[1] = findMaxFactor(global[1], 128 / local[2]);
        local[0] = 1;
        state = runtime_->setKernelArg(
            kernel_unequal_.get(), bias_data, *output_data, input_dim.c * parameters_->depth_multiplier);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "depthwise_conv_unequal setKernel failure\n");
        state = runtime_->enqueueKernel(kernel_unequal_.get(), (cl_uint)3, global, local);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "enqueueKernel failure\n");
    }

    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
