#include "userdriver/gpu/operators/CLDeconvolution.hpp"

namespace enn {
namespace ud {
namespace gpu {

namespace {
const uint32_t INPUT_INDEX = 0;
const uint32_t WEIGHT_INDEX = 1;
const uint32_t BIAS_INDEX = 2;
const uint32_t OUTPUT_INDEX = 0;
}  // namespace

CLDeconvolution::CLDeconvolution(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    DEBUG_PRINT("CLDeconvolution is created");
    row_weight_ = 0;
    row_input_ = 0;
    activation_info_ = ActivationInfo();
    input_nchw_ = nullptr;
    output_nchw_ = nullptr;
    input_tensor_ = nullptr;
    output_tensor_ = nullptr;
    weight_tensor_ = nullptr;
    bias_tensor_ = nullptr;
    deconvgeneral_ = nullptr;
    deconv1x8_ = nullptr;
    deconvmakalu_ = nullptr;
    depthdeconvmakalu_ = nullptr;
    quantized_deconv_ = nullptr;
    per_channel_quantized_deconv_ = nullptr;
    parameters_ = nullptr;
    cl_activation_ = nullptr;
}

Status CLDeconvolution::initialize(std::vector<std::shared_ptr<ITensor>> input_tensors,
                                   std::vector<std::shared_ptr<ITensor>> outputs_tensors,
                                   std::shared_ptr<Parameters> parameters) {
    DEBUG_PRINT("CLDeconvolution::initialize() is called");

    input_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(INPUT_INDEX));
    weight_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(WEIGHT_INDEX));
    bias_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(BIAS_INDEX));
    output_tensor_ = std::static_pointer_cast<CLTensor>(outputs_tensors.at(OUTPUT_INDEX));

    parameters_ = std::static_pointer_cast<DeconvolutionParameters>(parameters);

    ENN_DBG_PRINT("DeconvolutionParameters: padding.l %d padding.r %d padding.t %d padding.b %d; stride.h %d  stride.w %d; "
                  "group_size %d; weights_as_input %d; per_channel_quant %d; androidNN %d; isNCHW %d; openAibWino %d; "
                  "activation_info_: isEnabled() %d,activation() %d;\n",
                  parameters_->padding.l,
                  parameters_->padding.r,
                  parameters_->padding.t,
                  parameters_->padding.b,
                  parameters_->stride.h,
                  parameters_->stride.w,
                  parameters_->group_size,
                  parameters_->weights_as_input,
                  parameters_->per_channel_quant,
                  parameters_->androidNN,
                  parameters_->isNCHW,
                  parameters_->openAibWino,
                  parameters_->activation_info->isEnabled(),
                  parameters_->activation_info->activation());

    activation_info_ = *parameters_->activation_info.get();

    Dim4 input_dim = input_tensor_->getDim();
    Dim4 output_dim = output_tensor_->getDim();
    Dim4 weight_dim = weight_tensor_->getDim();
    if (parameters_->androidNN) {
        if (!parameters_->isNCHW) {
            input_dim = convertDimToNCHW(input_dim);
            input_nchw_ = std::make_shared<CLTensor>(runtime_,
                                                     precision_,
                                                     input_tensor_->getDataType(),
                                                     input_dim,
                                                     input_tensor_->getDataOrder(),
                                                     input_tensor_->getScale(),
                                                     input_tensor_->getZeroPoint());
        }
        output_dim.n = input_dim.n;
        output_dim.c = weight_dim.n;
        output_dim.h = input_dim.h * parameters_->stride.h + weight_dim.c - parameters_->stride.h - parameters_->padding.t -
                       parameters_->padding.b;
        output_dim.w = input_dim.w * parameters_->stride.w + weight_dim.h - parameters_->stride.w - parameters_->padding.l -
                       parameters_->padding.r;
        if (!parameters_->isNCHW) {
            output_nchw_ = std::make_shared<CLTensor>(runtime_,
                                                      precision_,
                                                      output_tensor_->getDataType(),
                                                      output_dim,
                                                      output_tensor_->getDataOrder(),
                                                      output_tensor_->getScale(),
                                                      output_tensor_->getZeroPoint());
        }

        Dim4 expected_output_dim = output_dim;
        if (!parameters_->isNCHW) {
            expected_output_dim = convertDimToNHWC(output_dim);
        }
        if (!isDimsSame(expected_output_dim, output_tensor_->getDim())) {
            output_tensor_->reconfigureDimAndBuffer(expected_output_dim);
        }
    }

    if (precision_ == PrecisionType::UINT8 || precision_ == PrecisionType::INT8) {
        if (parameters_->per_channel_quant == true) {
            per_channel_quantized_deconv_ = std::make_shared<CLDeconvolutionPerChannelQuantized>(runtime_, precision_);
            return per_channel_quantized_deconv_->initialize(input_dim,
                                                             output_dim,
                                                             input_tensor_->getZeroPoint(),
                                                             weight_tensor_,
                                                             bias_tensor_,
                                                             parameters_->scales,
                                                             parameters_->padding,
                                                             parameters_->stride,
                                                             parameters_->group_size,
                                                             activation_info_,
                                                             parameters_->weights_as_input,
                                                             parameters_->androidNN);
        } else {
            quantized_deconv_ = std::make_shared<CLDeconvolutionQuantized>(runtime_, precision_);
            return quantized_deconv_->initialize(input_dim,
                                                 output_dim,
                                                 input_tensor_->getZeroPoint(),
                                                 weight_tensor_,
                                                 bias_tensor_,
                                                 parameters_->padding,
                                                 parameters_->stride,
                                                 parameters_->group_size,
                                                 activation_info_,
                                                 parameters_->weights_as_input,
                                                 parameters_->androidNN);
        }
    }

    auto weight_tensor = std::static_pointer_cast<CLTensor>(weight_tensor_);
    if (parameters_->androidNN) {
        row_weight_ = weight_dim.c * weight_dim.h * weight_dim.n;
    } else {
        row_weight_ = weight_dim.w * weight_dim.h * weight_dim.n;
    }
    row_input_ = input_dim.h * input_dim.w;

    if (runtime_->isMakalu() && parameters_->group_size == 1) {
        deconvmakalu_ = std::make_shared<CLDeconvolutionMakalu>(runtime_,
                                                                precision_,
                                                                input_dim,
                                                                output_dim,
                                                                weight_tensor_,
                                                                bias_tensor_,
                                                                parameters_->padding,
                                                                parameters_->stride,
                                                                parameters_->group_size,
                                                                activation_info_,
                                                                parameters_->weights_as_input,
                                                                parameters_->androidNN);
        if (activation_info_.isEnabled()) {
            if (activation_info_.activation() == ActivationInfo::ActivationType::RELU) {
                activation_info_.disable();
            }
        }
    } else if (runtime_->isMakalu() && parameters_->group_size == input_dim.c && parameters_->group_size == output_dim.c) {
        depthdeconvmakalu_ = std::make_shared<CLDepthwiseDeconvolution>(runtime_,
                                                                        precision_,
                                                                        input_dim,
                                                                        output_dim,
                                                                        weight_tensor_,
                                                                        bias_tensor_,
                                                                        parameters_->padding,
                                                                        parameters_->stride,
                                                                        parameters_->group_size,
                                                                        activation_info_);
        if (activation_info_.isEnabled()) {
            if (activation_info_.activation() == ActivationInfo::ActivationType::RELU) {
                activation_info_.disable();
            }
        }
    } else {
        if (row_weight_ > row_input_) {
            deconvgeneral_ = std::make_shared<CLDeconvolutionGeneral>(runtime_,
                                                                      precision_,
                                                                      input_dim,
                                                                      output_dim,
                                                                      weight_tensor_,
                                                                      bias_tensor_,
                                                                      parameters_->padding,
                                                                      parameters_->stride,
                                                                      parameters_->group_size);
        } else {
            deconv1x8_ = std::make_shared<CLDeconvolution1x8>(runtime_,
                                                              precision_,
                                                              input_dim,
                                                              output_dim,
                                                              weight_tensor_,
                                                              bias_tensor_,
                                                              parameters_->padding,
                                                              parameters_->stride,
                                                              parameters_->group_size);
        }
    }

    Status status = Status::SUCCESS;
    if (precision_ != PrecisionType::UINT8 && precision_ != PrecisionType::INT8) {
        cl_activation_ = std::make_shared<CLActivation>(runtime_, precision_);
        std::shared_ptr<ActivationParameters> act_parameters = std::make_shared<ActivationParameters>();
        act_parameters->activation_info = activation_info_;
        act_parameters->relu_parameters = std::make_shared<ReluParameters>();
        act_parameters->relu_parameters->negative_slope = 0.0f;
        status = cl_activation_->initialize({output_tensor_}, {output_tensor_}, act_parameters);
    }

    return status;
}

Status CLDeconvolution::execute() {
    DEBUG_PRINT("CLDeconvolution::execute() is called");
    // for zero_sized input
    if (input_tensor_->getTotalSizeFromDims() == 0) {
        return Status::SUCCESS;
    }
    auto input_tensor = std::static_pointer_cast<CLTensor>(input_tensor_);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output_tensor_);
    Status status = Status::SUCCESS;
    if (parameters_->androidNN && !parameters_->isNCHW) {
        status = input_tensor->convertToNCHW(input_nchw_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "convertToNCHW failure\n");
        status = execute_nchw(input_nchw_, output_nchw_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "execute_nchw failure\n");
        status = output_nchw_->convertToNHWC(output_tensor);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "convertToNHWC failure\n");
    } else {
        status = execute_nchw(input_tensor, output_tensor);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "execute_nchw failure\n");
    }
    if (precision_ != PrecisionType::UINT8 && precision_ != PrecisionType::INT8) {
        status = cl_activation_->execute();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "cl_activation_->execute() failure\n");
    }

    return Status::SUCCESS;
}

Status CLDeconvolution::execute_nchw(const std::shared_ptr<CLTensor> input, std::shared_ptr<CLTensor> output) {
    DEBUG_PRINT("CLDeconvolution::execute() is called");
    if (precision_ == PrecisionType::UINT8 || precision_ == PrecisionType::INT8) {
        if (quantized_deconv_ != nullptr) {
            return quantized_deconv_->execute(input, output);
        } else {
            return per_channel_quantized_deconv_->execute(input, output);
        }
    }

    auto input_dim = input->getDim();
    auto output_dim = output->getDim();

    Status status;
    if (runtime_->isMakalu() && parameters_->group_size == 1) {
        status = deconvmakalu_->execute(input, output);
    } else if (runtime_->isMakalu() && parameters_->group_size == input_dim.c && parameters_->group_size == output_dim.c) {
        status = depthdeconvmakalu_->execute(input, output);
    } else {
        if (row_weight_ > row_input_) {
            status = deconvgeneral_->execute(input, output);
        } else {
            status = deconv1x8_->execute(input, output);
        }
    }
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "CLDeconvolution execute failure\n");
    return status;
}

Status CLDeconvolution::release() {
    DEBUG_PRINT("CLDeconvolution::release() is called");
    if (precision_ == PrecisionType::UINT8 || precision_ == PrecisionType::INT8) {
        if (quantized_deconv_ != nullptr) {
            return quantized_deconv_->release();
        } else {
            return per_channel_quantized_deconv_->release();
        }
    }

    if (runtime_->isMakalu() && parameters_->group_size == 1) {
        return deconvmakalu_->release();
    } else {
        if (row_weight_ > row_input_) {
            return deconvgeneral_->release();
        } else {
            return deconv1x8_->release();
        }
    }
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
