#include "CLConvolution.hpp"
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/common/CLParameter.hpp"

namespace enn {
namespace ud {
namespace gpu {

namespace {
const uint32_t INPUT_INDEX = 0;
const uint32_t WEIGHT_INDEX = 1;
const uint32_t BIAS_INDEX = 2;
const uint32_t OUTPUT_INDEX = 0;
}  // namespace

CLConvolution::CLConvolution(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision), gemm_convolution_(nullptr) {
    ENN_DBG_PRINT("CLConvolution is created");
    padding_ = {0, 0, 0, 0};
    stride_ = {0, 0};
    kernel_ = {0, 0};
    group_size_ = 0;
    axis_ = 0;
    dilation_ = {1, 1};
    conv_kernel_type_ = ConvolutionKernelType::GEMM;
    per_channel_quant_ = false;
    androidNN_ = false;
    isNCHW_ = true;
    openAibWino_ = false;
}

Status CLConvolution::initialize(std::vector<std::shared_ptr<ITensor>> inputs,
                                 std::vector<std::shared_ptr<ITensor>> outputs,
                                 std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLConvolution::initialize() is called");

    input_ = std::static_pointer_cast<CLTensor>(inputs.at(INPUT_INDEX));
    output_ = std::static_pointer_cast<CLTensor>(outputs.at(OUTPUT_INDEX));
    weight_ = std::static_pointer_cast<CLTensor>(inputs.at(WEIGHT_INDEX));
    bias_ = std::static_pointer_cast<CLTensor>(inputs.at(BIAS_INDEX));

    parameters_ = std::static_pointer_cast<ConvolutionParameters>(parameters);
    ENN_DBG_PRINT(
        "ConvolutionParameters: padding.l %d padding.r %d padding.t %d padding.b %d; dilations.h %d "
        "dilations.w %d; stride.h %d stride.w %d; group_size %d; axis %d; per_channel_quant %d;"
        " androidNN %d; isNCHW %d; storage_type %d; openAibWino %d; activation_info_: isEnabled() %d, activation() %d;\n",
        parameters_->padding.l,
        parameters_->padding.r,
        parameters_->padding.t,
        parameters_->padding.b,
        parameters_->dilation.h,
        parameters_->dilation.w,
        parameters_->stride.h,
        parameters_->stride.w,
        parameters_->group_size,
        parameters_->axis,
        parameters_->per_channel_quant,
        parameters_->androidNN,
        parameters_->isNCHW,
        parameters_->storage_type,
        parameters_->openAibWino,
        parameters_->activation_info->isEnabled(),
        parameters_->activation_info->activation());

    padding_ = parameters_->padding;
    stride_ = parameters_->stride;
    group_size_ = parameters_->group_size;
    axis_ = parameters_->axis;
    dilation_ = parameters_->dilation;
    kernel_.h = weight_->getDim().h;
    kernel_.w = weight_->getDim().w;
    activation_info_ = *parameters_->activation_info.get();
    weights_as_input_ = !weight_->is_const();
    bias_as_input_ = !bias_->is_const();
    per_channel_quant_ = parameters_->per_channel_quant;
    androidNN_ = parameters_->androidNN;
    isNCHW_ = parameters_->isNCHW;
    storage_type_ = parameters_->storage_type;
    openAibWino_ = parameters_->openAibWino;
    scales_ = parameters_->scales;

    Dim4 input_dim = input_->getDim();
    Dim4 output_dim = output_->getDim();
    if (storage_type_ != StorageType::TEXTURE && (androidNN_ || !isNCHW_)) {
        if (!isNCHW_) {
            input_dim = convertDimToNCHW(input_dim);
            input_nchw_ = std::make_shared<CLTensor>(runtime_,
                                                     precision_,
                                                     input_->getDataType(),
                                                     input_dim,
                                                     input_->getDataOrder(),
                                                     input_->getScale(),
                                                     input_->getZeroPoint());
        }
        kernel_.h = weight_->getDim().c;
        kernel_.w = weight_->getDim().h;
        int kernel_extent_h = dilation_.h * (kernel_.h - 1) + 1;
        int kernel_extent_w = dilation_.w * (kernel_.w - 1) + 1;
        output_dim.n = input_dim.n;
        output_dim.c = weight_->getDim().n;
        output_dim.h = (input_dim.h + padding_.b + padding_.t - kernel_extent_h + stride_.h) / stride_.h;
        output_dim.w = (input_dim.w + padding_.l + padding_.r - kernel_extent_w + stride_.w) / stride_.w;

        if (!isNCHW_) {
            output_nchw_ = std::make_shared<CLTensor>(runtime_,
                                                      precision_,
                                                      output_->getDataType(),
                                                      output_dim,
                                                      output_->getDataOrder(),
                                                      output_->getScale(),
                                                      output_->getZeroPoint());
        }
        group_size_ = input_dim.c / weight_->getDim().w;
    }

    // for zero_sized input_
    if (weight_->getTotalSizeFromDims() == 0) {
        return Status::SUCCESS;
    }
    Status status = Status::FAILURE;
    if (precision_ != PrecisionType::UINT8 && precision_ != PrecisionType::INT8 &&
        (weight_->getDataType() == DataType::UINT8 || weight_->getDataType() == DataType::INT8)) {  // hybrid op
        auto weight_tensor = std::static_pointer_cast<CLTensor>(weight_);
        const DataType dataType = bias_->getDataType();
        tmp_weight_tensor_ = std::make_shared<CLTensor>(runtime_, precision_, dataType, weight_tensor->getDim());

        std::shared_ptr<CLDeQuantization> dequantization = std::make_shared<CLDeQuantization>(runtime_, precision_);

        auto dequantization_parameters = std::make_shared<DeQuantizationParameters>();
        status = dequantization->initialize({weight_}, {tmp_weight_tensor_}, dequantization_parameters);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "dequantization initialize  failure\n");
        status = dequantization->execute();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "dequantization execute failure\n");
        weight_ = tmp_weight_tensor_;
    }

    if (precision_ == PrecisionType::UINT8 || precision_ == PrecisionType::INT8) {
        std::shared_ptr<CLTensor> bias_tensor = std::static_pointer_cast<CLTensor>(bias_);
        bias_copied_ = std::make_shared<CLTensor>(runtime_,
                                                  precision_,
                                                  bias_tensor->getDataType(),
                                                  bias_tensor->getDim(),
                                                  bias_tensor->getDataOrder(),
                                                  bias_tensor->getScale(),
                                                  bias_tensor->getZeroPoint());
        if (!bias_as_input_) {
            size_t offset = 0;
            size_t bytes = bias_tensor->getNumOfBytes();
            runtime_->copyBuffer(bias_copied_->getDataPtr(), bias_tensor->getDataPtr(), offset, offset, bytes);
        }
        if (per_channel_quant_ == true) {
            quantized_per_channel_convolution_ = std::make_shared<CLConvolutionPerChannelQuantized>(runtime_, precision_);
            status = quantized_per_channel_convolution_->initialize(input_dim,
                                                                    output_dim,
                                                                    padding_,
                                                                    stride_,
                                                                    group_size_,
                                                                    weight_,
                                                                    bias_copied_,
                                                                    activation_info_,
                                                                    scales_,
                                                                    androidNN_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "quantized_per_channel init failure\n");
        } else if (precision_ != PrecisionType ::INT8 &&
                   ((output_dim.w % 8 == 0 && 3 == kernel_.h && 3 == kernel_.w && 1 == group_size_ && 1 == dilation_.h &&
                     1 == dilation_.w && 1 == stride_.h && 1 == stride_.w) ||
                    (output_dim.w % 8 == 0 && 7 == kernel_.h && 7 == kernel_.w && 1 == group_size_ && 1 == dilation_.h &&
                     1 == dilation_.w && 1 == stride_.h && 1 == stride_.w) ||
                    (output_dim.w % 8 == 0 && 9 == kernel_.h && 9 == kernel_.w && 1 == group_size_ && 1 == dilation_.h &&
                     1 == dilation_.w && 1 == stride_.h && 1 == stride_.w))) {
            quantized_direct_convolution_ = std::make_shared<CLQuantizedDirectConvolution>(runtime_, precision_);
            auto input_tensor = (androidNN_ && !isNCHW_) ? input_nchw_ : input_;
            status = quantized_direct_convolution_->initialize(input_tensor,
                                                               output_dim,
                                                               weight_->getDim(),
                                                               padding_,
                                                               stride_,
                                                               dilation_,
                                                               weight_,
                                                               bias_copied_,
                                                               activation_info_,
                                                               weights_as_input_,
                                                               bias_as_input_,
                                                               androidNN_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "quantized_direct init failure\n");
        } else if ((group_size_ == 1 && runtime_->isMakalu()) || dilation_.h > 1 || dilation_.w > 1) {
            quantized_gemm1xX_convolution_ = std::make_shared<CLGEMM1xXConvolutionQuantized>(runtime_, precision_);
            auto input_tensor = (androidNN_ && !isNCHW_) ? input_nchw_ : input_;
            status = quantized_gemm1xX_convolution_->initialize(input_tensor,
                                                                output_dim,
                                                                padding_,
                                                                stride_,
                                                                group_size_,
                                                                axis_,
                                                                dilation_,
                                                                weight_,
                                                                bias_copied_,
                                                                activation_info_,
                                                                weights_as_input_,
                                                                bias_as_input_,
                                                                androidNN_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "quantized_gemmconv init failure\n");
        } else {
            quantized_gemm_convolution_ = std::make_shared<CLGEMMConvolutionQuantized>(runtime_, precision_);
            status = quantized_gemm_convolution_->initialize(input_dim,
                                                             output_dim,
                                                             padding_,
                                                             stride_,
                                                             group_size_,
                                                             weight_,
                                                             bias_copied_,
                                                             activation_info_,
                                                             weights_as_input_,
                                                             androidNN_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "quantized_gemmconv init failure\n");
        }
    } else {
        if (storage_type_ == StorageType::TEXTURE) {
            conv_kernel_type_ = ConvolutionKernelType::PowerVR;
            powervr_convolution_ = std::make_shared<CLPowerVRConvolution>(runtime_, precision_, input_dim, output_dim);
        } else {
            if (dilation_.h > 1 || dilation_.w > 1) {
                Dim2 kernel_extent = {(uint32_t)(dilation_.h * (kernel_.h - 1) + 1),
                                      (uint32_t)(dilation_.w * (kernel_.w - 1) + 1)};
                if (isFitDilationOpt(input_dim, output_dim, kernel_, dilation_, padding_)) {
                    conv_kernel_type_ = ConvolutionKernelType::DilationConv;
                    dilation_convolution_ = std::make_shared<CLDilationConvolution>(runtime_, precision_);
                } else if (isFitDirect(input_dim, output_dim, kernel_extent)) {
                    conv_kernel_type_ = ConvolutionKernelType::DIRECT;
                    direct_convolution_ = std::make_shared<CLDirectConvolution>(runtime_, precision_);
                } else {
                    conv_kernel_type_ = ConvolutionKernelType::GEMM1xX;
                    gemm1xx_convolution_ =
                        std::make_shared<CLGEMM1xXConvolution>(runtime_, precision_, input_dim, output_dim);
                }
            } else if (runtime_->isValhall() && !isNCHW_ && kernel_.h == 1 && kernel_.w == 1 && stride_.h == 1 &&
                       stride_.w == 1 && group_size_ == 1 && padding_.b == 0 && padding_.t == 0 && padding_.l == 0 &&
                       padding_.r == 0 && output_nchw_->getDim().h * output_nchw_->getDim().w % 4 == 0) {
                // special NHWC 1x1 convolution opt for MobileBert
                conv_kernel_type_ = ConvolutionKernelType::GEMM1xX;
                gemm1xx_convolution_ = std::make_shared<CLGEMM1xXConvolution>(runtime_, precision_, input_dim, output_dim);
            } else if (input_dim.w * input_dim.h < 96 * 96 && openAibWino_ && kernel_.h == 7 && kernel_.w == 7 &&
                       stride_.h == 1 && stride_.w == 1) {
                conv_kernel_type_ = ConvolutionKernelType::WINO2x2_7x7;
                wino_2x2_7x7_ = std::make_shared<CLWINO2x2_7x7>(runtime_, precision_, input_dim, output_dim);
            } else if (openAibWino_ && kernel_.h == 5 && kernel_.w == 5 && stride_.h == 1 && stride_.w == 1) {
                conv_kernel_type_ = ConvolutionKernelType::WINO4x4_5x5;
                wino_4x4_5x5_ = std::make_shared<CLWINO4x4_5x5>(runtime_, precision_, input_dim, output_dim);
            } else if (openAibWino_ && kernel_.h == 3 && kernel_.w == 3 && stride_.h == 1 && stride_.w == 1 &&
                       8 <= output_dim.c && 8 <= input_dim.c) {
                conv_kernel_type_ = ConvolutionKernelType::WINO6x6_3x3;
                wino_6x6_3x3_ = std::make_shared<CLWINO6x6_3x3>(runtime_, precision_, input_dim, output_dim);
            } else if (isFitDirect(input_dim, output_dim, kernel_)) {
#if defined(__ANDROID__)
                conv_kernel_type_ = ConvolutionKernelType::DIRECT;
                direct_convolution_ = std::make_shared<CLDirectConvolution>(runtime_, precision_);
#else
                conv_kernel_type_ = ConvolutionKernelType::GEMM;
                gemm_convolution_ = std::make_shared<CLGEMMConvolution>(runtime_, precision_, input_dim, output_dim);
#endif
            } else if (kernel_.h == 3 && kernel_.w == 3 && stride_.h == 1 && stride_.w == 1) {
                conv_kernel_type_ = ConvolutionKernelType::WINO;
                wino_convolution_ = std::make_shared<CLWINOConvolution>(runtime_, precision_, input_dim, output_dim);
            } else if (kernel_.h == 1 && kernel_.w == 1 && group_size_ == 1) {
                if (isFitKernel1x1(input_dim, output_dim)) {
                    conv_kernel_type_ = ConvolutionKernelType ::Kernel1x1;
                    kernel1x1_convolution_ =
                        std::make_shared<CLKernel1x1Convolution>(runtime_, precision_, input_dim, output_dim);
                } else if (input_dim.n == 1 && input_dim.c == 2048 && input_dim.h == 1 && input_dim.w == 1 &&
                           output_dim.n == 1 && output_dim.c == 1001 && output_dim.h == 1 && output_dim.w == 1 &&
                           stride_.h == 1 && stride_.w == 1) {
                    conv_kernel_type_ = ConvolutionKernelType::GEMM;
                    gemm_convolution_ = std::make_shared<CLGEMMConvolution>(runtime_, precision_, input_dim, output_dim);
                } else {
                    conv_kernel_type_ = ConvolutionKernelType::GEMM1xX;
                    gemm1xx_convolution_ =
                        std::make_shared<CLGEMM1xXConvolution>(runtime_, precision_, input_dim, output_dim);
                }
            } else if (group_size_ != 1) {
                conv_kernel_type_ = ConvolutionKernelType::GEMM;
                gemm_convolution_ = std::make_shared<CLGEMMConvolution>(runtime_, precision_, input_dim, output_dim);
            } else {
                conv_kernel_type_ = ConvolutionKernelType::GEMM1xX;
                gemm1xx_convolution_ = std::make_shared<CLGEMM1xXConvolution>(runtime_, precision_, input_dim, output_dim);
            }
        }

        Dim4 weight_dims = weight_->getDim();
        switch (conv_kernel_type_) {
        case ConvolutionKernelType::GEMM:
            status = gemm_convolution_->initialize(padding_,
                                                   stride_,
                                                   group_size_,
                                                   axis_,
                                                   dilation_,
                                                   weight_,
                                                   bias_,
                                                   activation_info_,
                                                   weights_as_input_,
                                                   androidNN_,
                                                   isNCHW_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "GEMM init failure\n");
            break;
        case ConvolutionKernelType::GEMM1xX:
            status = gemm1xx_convolution_->initialize(padding_,
                                                      stride_,
                                                      group_size_,
                                                      axis_,
                                                      dilation_,
                                                      weight_,
                                                      bias_,
                                                      activation_info_,
                                                      weights_as_input_,
                                                      androidNN_,
                                                      isNCHW_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "GEMM1xX init failure\n");
            break;
        case ConvolutionKernelType::WINO:
            status = wino_convolution_->initialize(padding_,
                                                   stride_,
                                                   group_size_,
                                                   axis_,
                                                   dilation_,
                                                   weight_,
                                                   bias_,
                                                   activation_info_,
                                                   weights_as_input_,
                                                   androidNN_,
                                                   isNCHW_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "WINO init failure\n");
            break;
        case ConvolutionKernelType::WINO6x6_3x3:
            status = wino_6x6_3x3_->initialize(padding_,
                                               stride_,
                                               group_size_,
                                               axis_,
                                               dilation_,
                                               weight_,
                                               bias_,
                                               activation_info_,
                                               weights_as_input_,
                                               androidNN_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "WINO init failure\n");
            break;
        case ConvolutionKernelType::WINO4x4_5x5:
            status = wino_4x4_5x5_->initialize(padding_,
                                               stride_,
                                               group_size_,
                                               axis_,
                                               dilation_,
                                               weight_,
                                               bias_,
                                               activation_info_,
                                               weights_as_input_,
                                               androidNN_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "WINO init failure\n");
            break;
        case ConvolutionKernelType::WINO2x2_7x7:
            status = wino_2x2_7x7_->initialize(padding_,
                                               stride_,
                                               group_size_,
                                               axis_,
                                               dilation_,
                                               weight_,
                                               bias_,
                                               activation_info_,
                                               weights_as_input_,
                                               androidNN_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "WINO init failure\n");
            break;
        case ConvolutionKernelType::Kernel1x1:
            status = kernel1x1_convolution_->initialize(padding_,
                                                        stride_,
                                                        group_size_,
                                                        axis_,
                                                        dilation_,
                                                        weight_,
                                                        bias_,
                                                        activation_info_,
                                                        weights_as_input_,
                                                        androidNN_,
                                                        isNCHW_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "Kernel1x1 init failure\n");
            break;
        case ConvolutionKernelType::DIRECT:
            status = direct_convolution_->initialize(input_dim,
                                                     output_dim,
                                                     weight_->getDim(),
                                                     padding_,
                                                     stride_,
                                                     dilation_,
                                                     weight_,
                                                     bias_,
                                                     activation_info_,
                                                     weights_as_input_,
                                                     androidNN_,
                                                     isNCHW_);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == status, "Direct5x5 init failure");
            break;
        case ConvolutionKernelType ::DilationConv:
            status = dilation_convolution_->initialize(input_dim,
                                                       output_dim,
                                                       weight_->getDim(),
                                                       padding_,
                                                       stride_,
                                                       dilation_,
                                                       weight_,
                                                       bias_,
                                                       activation_info_,
                                                       weights_as_input_,
                                                       androidNN_,
                                                       isNCHW_);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == status, "dilation_ conv init failure");
            break;
        case ConvolutionKernelType::PowerVR:
            status = powervr_convolution_->initialize(
                padding_, stride_, group_size_, axis_, dilation_, weight_, bias_, activation_info_, weights_as_input_);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == status, "PowerVR init failure");
            break;
        default: return Status::FAILURE;
        }
        if (activation_info_.isEnabled()) {
            if (activation_info_.activation() == ActivationInfo::ActivationType::RELU ||
                (activation_info_.activation() == ActivationInfo::ActivationType::RELU6 &&
                 precision_ == PrecisionType::FP16)) {
                activation_info_.disable();
            }
        }
    }

    if (storage_type_ != StorageType::TEXTURE && !isNCHW_) {
        output_dim = convertDimToNHWC(output_dim);
    }
    if (storage_type_ != StorageType::TEXTURE && (androidNN_ || !isNCHW_)) {
        if (!isDimsSame(output_dim, output_->getDim())) {
            output_->reconfigureDimAndBuffer(output_dim);
        }
    }

    if (precision_ != PrecisionType::UINT8 && precision_ != PrecisionType::INT8) {
        cl_activation_ = std::make_shared<CLActivation>(runtime_, precision_);
        std::shared_ptr<ActivationParameters> act_parameters = std::make_shared<ActivationParameters>();
        act_parameters->activation_info = activation_info_;
        act_parameters->relu_parameters = std::make_shared<ReluParameters>();
        act_parameters->relu_parameters->negative_slope = 0.0f;
        if (parameters_->androidNN && parameters_->storage_type != StorageType::TEXTURE && !parameters_->isNCHW) {
            status = cl_activation_->initialize({output_nchw_}, {output_nchw_}, act_parameters);
        } else {
            status = cl_activation_->initialize({output_}, {output_}, act_parameters);
        }
    }

    return status;
}

Status CLConvolution::execute() {
    ENN_DBG_PRINT("CLConvolution is executed");
    // for zero_sized input_
    if (input_->getTotalSizeFromDims() == 0) {
        return Status::SUCCESS;
    }

    auto input_tensor = std::static_pointer_cast<CLTensor>(input_);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output_);

    if (storage_type_ != StorageType::TEXTURE && !isNCHW_) {
        if (runtime_->isValhall() && !isNCHW_ && kernel_.h == 1 && kernel_.w == 1 && stride_.h == 1 && stride_.w == 1 &&
            group_size_ == 1 && padding_.b == 0 && padding_.t == 0 && padding_.l == 0 && padding_.r == 0 &&
            output_nchw_->getDim().h * output_nchw_->getDim().w % 4 == 0 && precision_ != PrecisionType::UINT8 &&
            precision_ != PrecisionType::INT8) {
            // special NHWC 1x1 convolution opt for MobileBert
            Status status = executeNCHW(input_tensor, output_tensor);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "executeNCHW failure\n");
        } else {
            Status status = input_tensor->convertToNCHW(input_nchw_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "convertToNCHW failure\n");
            status = executeNCHW(input_nchw_, output_nchw_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "executeNCHW failure\n");
            status = output_nchw_->convertToNHWC(output_tensor);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "convertToNHWC failure\n");
        }
    } else {
        return executeNCHW(input_tensor, output_tensor);
    }

    return Status::SUCCESS;
}

Status CLConvolution::executeNCHW(const std::shared_ptr<CLTensor> input, std::shared_ptr<CLTensor> output) {
    ENN_DBG_PRINT("CLConvolution::executeNCHW() is executed");
    Status status;
    Dim4 output_dim = output->getDim();
    if (precision_ == PrecisionType::UINT8 || precision_ == PrecisionType::INT8) {
        if (bias_as_input_) {
            std::shared_ptr<CLTensor> bias_tensor = std::static_pointer_cast<CLTensor>(bias_);
            size_t offset = 0;
            size_t bytes = bias_tensor->getNumOfBytes();
            runtime_->copyBuffer(bias_copied_->getDataPtr(), bias_tensor->getDataPtr(), offset, offset, bytes);
        }

        if (per_channel_quant_ == true) {
            return quantized_per_channel_convolution_->execute(input, output);
        } else if (precision_ != PrecisionType ::INT8 &&
                   ((output_dim.w % 8 == 0 && 3 == kernel_.h && 3 == kernel_.w && 1 == group_size_ && 1 == dilation_.h &&
                     1 == dilation_.w && 1 == stride_.h && 1 == stride_.w) ||
                    (output_dim.w % 8 == 0 && 7 == kernel_.h && 7 == kernel_.w && 1 == group_size_ && 1 == dilation_.h &&
                     1 == dilation_.w && 1 == stride_.h && 1 == stride_.w) ||
                    (output_dim.w % 8 == 0 && 9 == kernel_.h && 9 == kernel_.w && 1 == group_size_ && 1 == dilation_.h &&
                     1 == dilation_.w && 1 == stride_.h && 1 == stride_.w))) {
            return quantized_direct_convolution_->execute(input, output);
        } else if ((group_size_ == 1 && runtime_->isMakalu()) || dilation_.h > 1 || dilation_.w > 1) {
            return quantized_gemm1xX_convolution_->execute(input, output);
        } else {
            return quantized_gemm_convolution_->execute(input, output);
        }
    } else {
        switch (conv_kernel_type_) {
        case ConvolutionKernelType::GEMM: status = gemm_convolution_->execute(input, output); break;
        case ConvolutionKernelType::GEMM1xX: status = gemm1xx_convolution_->execute(input, output); break;
        case ConvolutionKernelType::WINO: {
            status = wino_convolution_->execute(input, output);
        } break;
        case ConvolutionKernelType::WINO6x6_3x3: {
            status = wino_6x6_3x3_->execute(input, output);
        } break;
        case ConvolutionKernelType::WINO4x4_5x5: {
            status = wino_4x4_5x5_->execute(input, output);
        } break;
        case ConvolutionKernelType::WINO2x2_7x7: {
            status = wino_2x2_7x7_->execute(input, output);
        } break;
        case ConvolutionKernelType::Kernel1x1: status = kernel1x1_convolution_->execute(input, output); break;
        case ConvolutionKernelType::DIRECT: status = direct_convolution_->execute(input, output); break;
        case ConvolutionKernelType ::DilationConv: status = dilation_convolution_->execute(input, output); break;
        case ConvolutionKernelType::PowerVR: {
            std::vector<std::shared_ptr<ITensor>> vec_input;
            vec_input.push_back(input);
            status = powervr_convolution_->execute(vec_input, output);
            break;
        }
        default: return Status::FAILURE;
        }
    }
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "CLConvolution execute failure\n");

    status = cl_activation_->execute();
    return status;
}

Status CLConvolution::release() { return Status::SUCCESS; }

bool CLConvolution::isFitKernel1x1(const Dim4 &input_dim, const Dim4 &output_dim) {
    if ((stride_.h == 1 && stride_.w == 1) || (stride_.h == 2 && stride_.w == 2)) {
        if (runtime_->isMakalu()) {
            if (output_dim.h * output_dim.w * output_dim.c * input_dim.c * kernel_.h * kernel_.w < 15000000 &&
                (input_dim.n > 1 || output_dim.h != output_dim.w)) {
                return true;
            } else {
                return false;
            }
        }
        return (output_dim.h * output_dim.w / 8.0 * output_dim.c / 2.0) / 12.0 / 20.0 > 10.0 &&
               output_dim.h * output_dim.w > 400;
    } else {
        return false;
    }
}

bool CLConvolution::isFitDirect(const Dim4 &input_dim, const Dim4 &output_dim, const Dim2 &kernel_dim) {
    if (output_dim.w % 8 == 0 && 5 == kernel_dim.h && 5 == kernel_dim.w && 1 == group_size_ && 1 == stride_.h &&
        1 == stride_.w) {
        return true;
    } else if (output_dim.w % 8 == 0 && 7 == kernel_dim.h && 7 == kernel_dim.w && 1 == group_size_ && 1 == stride_.h &&
               1 == stride_.w) {
        return true;
    } else if (output_dim.w % 8 == 0 && 9 == kernel_dim.h && 9 == kernel_dim.w && 1 == group_size_ && 1 == stride_.h &&
               1 == stride_.w) {
        return true;
    } else if (precision_ == PrecisionType::FP32 && 3 == kernel_dim.h && 3 == kernel_dim.w && 1 == group_size_ &&
               1 == stride_.h && 1 == stride_.w) {
        return true;
    } else if (openAibWino_ &&
               ((input_dim.h * input_dim.w > 128 * 128) && output_dim.c != input_dim.c &&
                (output_dim.c < 8 || input_dim.c < 8)) &&
               3 == kernel_dim.h && 3 == kernel_dim.w && 1 == group_size_ && 1 == stride_.h && 1 == stride_.w) {
        return true;
    } else if (output_dim.w % 8 == 0 &&
               output_dim.c * output_dim.h * output_dim.w * input_dim.c * kernel_dim.h * kernel_dim.w > 2000000000 &&
               3 == kernel_dim.h && 3 == kernel_dim.w && 1 == group_size_ && 1 == stride_.h && 1 == stride_.w) {
        return true;
    } else if (precision_ == PrecisionType::FP32 && 1 == kernel_dim.h && 1 == kernel_dim.w && 1 == group_size_ &&
               1 == stride_.h && 1 == stride_.w) {
        return true;
    } else {
        return false;
    }
}

bool CLConvolution::isFitDilationOpt(const Dim4 &input_dim,
                                     const Dim4 &output_dim,
                                     const Dim2 &kernel_dim,
                                     const Dim2 &dilation_dim,
                                     const Pad4 &pad_dim) {
    bool ret = false;
    if (dilation_dim.h == 8 && dilation_dim.w == 8 && kernel_dim.h == 3 && kernel_dim.w == 3 && pad_dim.t == 8 &&
        pad_dim.b == 8 && pad_dim.l == 8 && pad_dim.r == 8 && output_dim.w % 8 == 0 && 1 == group_size_ && 1 == stride_.h &&
        1 == stride_.w) {
        // optimization for dilation config
        // (dilation_size=(8,8) && kernel_size=(3, 3) && pad_size=(8,8,8,8) && stride_=(1,1)&& outputW%8==0 && group_size=1)
        ret = true;
    }

    return ret;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
