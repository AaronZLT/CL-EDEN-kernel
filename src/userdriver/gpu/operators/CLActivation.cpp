#include "userdriver/gpu/operators/CLActivation.hpp"

namespace enn {
namespace ud {
namespace gpu {

CLActivation::CLActivation(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    ENN_DBG_PRINT("CLActivation is created\n");
    parameters_ = nullptr;
    relu_ = nullptr;
    relu1_ = nullptr;
    relu6_ = nullptr;
    sigmoid_ = nullptr;
    tanh_ = nullptr;
}

Status CLActivation::initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                                const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                                const std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLActivation::initialize() is called\n");
    parameters_ = std::static_pointer_cast<ActivationParameters>(parameters);
    CHECK_EXPR_RETURN_FAILURE(nullptr != parameters_, "CLActivation must have parameters\n");

    if (parameters_->activation_info.isEnabled()) {
        switch (parameters_->activation_info.activation()) {
        case ActivationInfo::ActivationType::RELU: {
            relu_ = std::make_shared<CLRelu>(runtime_, precision_);
            Status status = relu_->initialize(input_tensors, output_tensors, parameters_->relu_parameters);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "relu initialize failure\n");
            break;
        }
        case ActivationInfo::ActivationType::RELU1: {
            relu1_ = std::make_shared<CLRelu1>(runtime_, precision_);
            relu1_->initialize(input_tensors, output_tensors, nullptr);
            break;
        }
        case ActivationInfo::ActivationType::RELU6: {
            relu6_ = std::make_shared<CLRelu6>(runtime_, precision_);
            relu6_->initialize(input_tensors, output_tensors, nullptr);
            break;
        }
        case ActivationInfo::ActivationType::SIGMOID: {
            sigmoid_ = std::make_shared<CLSigmoid>(runtime_, precision_);
            sigmoid_->initialize(input_tensors, output_tensors, nullptr);
            break;
        }
        case ActivationInfo::ActivationType::TANH: {
            tanh_ = std::make_shared<CLTanh>(runtime_, precision_);
            tanh_->initialize(input_tensors, output_tensors, nullptr);
            break;
        }
        case ActivationInfo::ActivationType::NONE: {
            UNUSED(input_tensors);
            UNUSED(output_tensors);
            break;
        }
        default: ERROR_PRINT("Non-Support Activation Type"); return Status::FAILURE;
        }
    }

    return Status::SUCCESS;
}

Status CLActivation::execute() {
    ENN_DBG_PRINT("CLActivation::execute() is called\n");
    if (parameters_->activation_info.isEnabled()) {
        switch (parameters_->activation_info.activation()) {
        case ActivationInfo::ActivationType::RELU: {
            Status status = relu_->execute();
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "relu execute failure\n");
            break;
        }
        case ActivationInfo::ActivationType::RELU1: {
            Status status = relu1_->execute();
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "relu1 execute failure\n");
            break;
        }
        case ActivationInfo::ActivationType::RELU6: {
            Status status = relu6_->execute();
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "relu6 execute failure\n");
            break;
        }
        case ActivationInfo::ActivationType::SIGMOID: {
            Status status = sigmoid_->execute();
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "sigmoid execute failure\n");
            break;
        }
        case ActivationInfo::ActivationType::TANH: {
            Status status = tanh_->execute();
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "tanh execute failure\n");
            break;
        }
        case ActivationInfo::ActivationType::NONE: break;
        default: ERROR_PRINT("Non-Support Activation Type"); return Status::FAILURE;
        }
    }
    return Status::SUCCESS;
}

Status CLActivation::release() {
    ENN_DBG_PRINT("CLActivation::release() is called\n");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
