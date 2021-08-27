#include "userdriver/gpu/operators/CLFullyConnected.hpp"

namespace enn {
namespace ud {
namespace gpu {

namespace {
const uint32_t INPUT_INDEX = 0;
const uint32_t WEIGHT_INDEX = 1;
const uint32_t BIAS_INDEX = 2;
const uint32_t OUTPUT_INDEX = 0;
}  // namespace

CLFullyConnected::CLFullyConnected(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    ENN_DBG_PRINT("CLFullyConnected is created");
    storage_type_ = StorageType::BUFFER;
    fc_kernel_type_ = FullyConnectedKernelType::BASE;
    isAndroidNN_ = false;
}

Status CLFullyConnected::initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                                    const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                                    const std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLFullyConnected::initialize() is called");
    input_ = std::static_pointer_cast<CLTensor>(input_tensors.at(INPUT_INDEX));
    output_ = std::static_pointer_cast<CLTensor>(output_tensors.at(OUTPUT_INDEX));
    weight_ = std::static_pointer_cast<CLTensor>(input_tensors.at(WEIGHT_INDEX));
    bias_ = std::static_pointer_cast<CLTensor>(input_tensors.at(BIAS_INDEX));

    parameters_ = std::static_pointer_cast<FullyConnectedParameters>(parameters);
    CHECK_EXPR_RETURN_FAILURE(nullptr != parameters_, "CLMaxpool must have parameters\n");
    storage_type_ = parameters_->storage_type;
    isAndroidNN_ = parameters_->androidNN;
    activation_info_ = *parameters_->activation_info.get();
    weights_as_input_ = !weight_->is_const();

    ENN_DBG_PRINT("FullyConnectedParameters: androidNN %d; weights_as_input %d; storage_type %d;"
                  " activation_info_: isEnabled() %d, activation() %d;\n",
                  isAndroidNN_,
                  weights_as_input_,
                  storage_type_,
                  activation_info_.isEnabled(),
                  activation_info_.activation());

    if (isAndroidNN_ && storage_type_ != StorageType::TEXTURE) {
        fc_nd_input_dims_ = input_->getDims();
        fc_2d_input_dims_ = {input_->getTotalSizeFromDims() / weight_->getDims(1), weight_->getDims(1)};
        if (fc_nd_input_dims_ != fc_2d_input_dims_) {
            ENN_DBG_PRINT("[FC] reconfigure  dims as (nbatch, nunit).");
            input_->reconfigureDims(fc_2d_input_dims_);
        }
        NDims fc_ouput_dims = {input_->getDims(0), weight_->getDims(0)};
        if (output_->getDims() != fc_ouput_dims) {
            output_->reconfigureDimsAndBuffer(fc_ouput_dims);
        }
    }
    Dim4 input_dim = input_->getDim();
    Dim4 output_dim = output_->getDim();

    if ((input_dim.n > FULLY_CONNECTED_OPT_BATCH) && (precision_ != PrecisionType::UINT8) && (precision_ != PrecisionType::INT8)) {
        // test branch
        if (input_dim.c * input_dim.h * input_dim.w <= 256) {
            fc_kernel_type_ = FullyConnectedKernelType::FC8X1GEMV;
            gemv_fullyconnected_ = std::make_shared<CL8X1GEMVFullyConnected>(
                runtime_, precision_, input_dim, output_dim, weight_, bias_, weights_as_input_);
            ENN_DBG_PRINT(" Use FullyConnectedKernelType::FC8X1GEMV.");
        } else {
            fc_kernel_type_ = FullyConnectedKernelType::FC8X1;
            cl8x1_fullyconnected_ = std::make_shared<CL8x1FullyConnected>(
                runtime_, precision_, input_dim, output_dim, weight_, bias_, weights_as_input_);
            ENN_DBG_PRINT(" Use FullyConnectedKernelType::FC8X1.");
        }
    } else {
        if (storage_type_ == StorageType::TEXTURE) {
            fc_kernel_type_ = FullyConnectedKernelType::TFLITE_TEXTURE2D;
            tflite_texture2d_fc_ = std::make_shared<CLFCTFLiteTexture2D>(runtime_, precision_, input_dim, output_dim);
            tflite_texture2d_fc_->initialize(weight_, bias_, activation_info_, weights_as_input_);
        } else if (isAndroidNN_ && precision_ != PrecisionType::UINT8 && precision_ != PrecisionType::INT8 &&
                   1 == input_dim.n) {
            fc_kernel_type_ = FullyConnectedKernelType::DIRECT;
            direct_fullyconnected_ = std::make_shared<CLDirectFullyConnected>(
                runtime_, precision_, input_, weight_, bias_, output_, weights_as_input_);
        } else {
            fc_kernel_type_ = FullyConnectedKernelType::BASE;
            base_fullyconnected_ = std::make_shared<CLBaseFullyConnected>(
                runtime_, precision_, input_, weight_, bias_, output_, weights_as_input_);
        }
    }

    if (fc_2d_input_dims_ != fc_nd_input_dims_ && storage_type_ != StorageType::TEXTURE) {
        ENN_DBG_PRINT("[FC] recover  dims to original one.");
        input_->reconfigureDims(fc_nd_input_dims_);
    }
    cl_activation_ = std::make_shared<CLActivation>(runtime_, precision_);
    std::shared_ptr<ActivationParameters> act_parameters = std::make_shared<ActivationParameters>();
    act_parameters->activation_info = activation_info_;
    act_parameters->relu_parameters = std::make_shared<ReluParameters>();
    act_parameters->relu_parameters->negative_slope = 0.0f;
    cl_activation_->initialize({output_}, {output_}, act_parameters);
    return Status::SUCCESS;
}

Status CLFullyConnected::execute() {
    ENN_DBG_PRINT("CLFullyConnected::execute() is called");
    // for zero_sized
    if (input_->getTotalSizeFromDims() == 0) {
        return Status::SUCCESS;
    }

    if (fc_2d_input_dims_ != fc_nd_input_dims_ && storage_type_ != StorageType::TEXTURE) {
        ENN_DBG_PRINT("[FC] reconfigure  dims as (nbatch, nunit).");
        input_->reconfigureDims(fc_2d_input_dims_);
    }

    Status status = Status::FAILURE;
    switch (fc_kernel_type_) {
    case FullyConnectedKernelType::BASE:
        status = base_fullyconnected_->execute(input_, output_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "base_fullyconnected_ execute failure\n");
        break;
    case FullyConnectedKernelType::FC8X1:
        status = cl8x1_fullyconnected_->execute(input_, output_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "cl8x1_fullyconnected_ execute failure\n");
        break;
    case FullyConnectedKernelType::FC8X1GEMV:
        status = gemv_fullyconnected_->execute(input_, output_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "gemv_fullyconnected_ execute failure\n");
        break;
    case FullyConnectedKernelType::DIRECT:
        status = direct_fullyconnected_->execute(input_, output_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "direct_fullyconnected_ execute failure\n");
        break;
    case FullyConnectedKernelType ::TFLITE_TEXTURE2D:
        status = tflite_texture2d_fc_->execute(input_, output_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "tflite_texture2d_fullyconnected_ execute failure\n");
        break;
    default: ERROR_PRINT("Non-Support Type"); return Status::FAILURE;
    }
    if (fc_2d_input_dims_ != fc_nd_input_dims_ && storage_type_ != StorageType::TEXTURE) {
        ENN_DBG_PRINT("[FC] recover  dims to original one.");
        input_->reconfigureDims(fc_nd_input_dims_);
    }

    if (storage_type_ != StorageType::TEXTURE) {
        status = cl_activation_->execute();
    }
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "cl_activation_->execute failure\n");
    return Status::SUCCESS;
}

Status CLFullyConnected::release() {
    Status status = Status::FAILURE;
    fc_2d_input_dims_.clear();
    fc_nd_input_dims_.clear();
    switch (fc_kernel_type_) {
    case FullyConnectedKernelType ::BASE:
        status = base_fullyconnected_->release();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "base_fullyconnected_ release failure\n");
        break;
    case FullyConnectedKernelType::FC8X1:
        status = cl8x1_fullyconnected_->release();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "cl8x1_fullyconnected_ release failure\n");
        break;
    case FullyConnectedKernelType::FC8X1GEMV:
        status = gemv_fullyconnected_->release();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "gemv_fullyconnected_ release failure\n");
        break;
    case FullyConnectedKernelType::DIRECT:
        status = direct_fullyconnected_->release();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "direct_fullyconnected_ release failure\n");
        break;
    case FullyConnectedKernelType::TFLITE_TEXTURE2D: status = tflite_texture2d_fc_->release(); break;
    default: ERROR_PRINT("Non-Support Type"); return Status::FAILURE;
    }
    return status;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
