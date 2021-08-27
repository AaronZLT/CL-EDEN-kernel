#include "CLAdd.hpp"

namespace enn {
namespace ud {
namespace gpu {
namespace {
constexpr int INPUT_INDEX_0 = 0;
constexpr int INPUT_INDEX_1 = 1;
constexpr int OUTPUT_INDEX = 0;

class CLAddTextureImpl {
  public:
    CLAddTextureImpl(CLAdd *base) : base_(base) {}
    Status initialize() {
        Status status = base_->runtime_->setKernel(&kernel_, "eltwise_add_zero_one_texture2d", base_->precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel eltwise_add_zero_one_texture2d failure\n");
        return Status::SUCCESS;
    }

    Status execute() {
        Status status = Status::SUCCESS;

        auto input_tensor_0 = base_->input_tensors_[INPUT_INDEX_0];
        auto input_tensor_1 = base_->input_tensors_[INPUT_INDEX_1];
        if (input_tensor_0 != base_->input_broadcast_0_) {
            status = input_tensor_0->broadCastTo(base_->input_broadcast_0_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "input_tensor_0->broadCastTo execute failure\n");
        }
        if (input_tensor_1 != base_->input_broadcast_1_) {
            status = input_tensor_1->broadCastTo(base_->input_broadcast_1_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "input_tensor_1->broadCastTo execute failure\n");
        }

        const uint32_t imageH = base_->output_tensor_->getImageH();
        const uint32_t imageW = base_->output_tensor_->getImageW();
        status = base_->runtime_->setKernelArg(kernel_.get(),
                                               base_->input_broadcast_0_->getDataPtr(),
                                               base_->input_broadcast_1_->getDataPtr(),
                                               base_->output_tensor_->getDataPtr(),
                                               base_->parameters_->coeff[0],
                                               base_->parameters_->coeff[1],
                                               imageW,
                                               imageH);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");

        size_t local[2] = {4, 4};
        size_t global[2] = {alignTo(imageW, local[0]), alignTo(imageH, local[1])};

        status = base_->runtime_->enqueueKernel(kernel_.get(), (cl_uint)2, global, local);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "execute kernel failure\n");

        return Status::SUCCESS;
    }

  private:
    CLAdd *base_;
    std::shared_ptr<struct _cl_kernel> kernel_;
};
}  // namespace

CLAdd::CLAdd(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    ENN_DBG_PRINT("CLAdd is created\n");
    input_tensors_.clear();
    output_tensor_ = nullptr;
    parameters_ = nullptr;
    cl_activation_ = nullptr;
    quantized_add_ = nullptr;
    kernel_ = nullptr;
    kernel_one_input_ = nullptr;
    input_broadcast_0_ = nullptr;
    input_broadcast_1_ = nullptr;
    is_vector_add_ = false;
    texture_impl_ = nullptr;
}

Status CLAdd::initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                         const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                         const std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLAdd::initialize() is called\n");

    for (auto input_tensor : input_tensors) {
        input_tensors_.push_back(std::static_pointer_cast<CLTensor>(input_tensor));
    }
    CHECK_EXPR_RETURN_FAILURE(
        input_tensors_.size() >= 2, "CLAdd at least has two input tensors, here only %lu\n", input_tensors_.size());
    output_tensor_ = std::static_pointer_cast<CLTensor>(output_tensors.at(OUTPUT_INDEX));
    parameters_ = std::static_pointer_cast<AddParameters>(parameters);
    ENN_DBG_PRINT("AddParameters, andoridNN: %d, isNCHW: %d, activation: %d\n",
                  parameters_->androidNN,
                  parameters_->isNCHW,
                  parameters_->activation_info.activation());

    if (parameters_->androidNN) {
        if (parameters_->storage_type == StorageType::TEXTURE || parameters_->isNCHW) {
            // optimize for AI Benchmark model's ADD with 2 inputs: a 4-D Tensor and a 1-D Tensor
            NDims dims_in_0 = input_tensors_[INPUT_INDEX_0]->getDims();
            NDims dims_in_1 = input_tensors_[INPUT_INDEX_1]->getDims();
            if (input_tensors_[INPUT_INDEX_0]->getNumOfDims() == 1) {
                reorder123DimsTo4DimsForBroadcast(input_tensors_[INPUT_INDEX_0]->getDims(), dims_in_0);
            }
            if (input_tensors_[INPUT_INDEX_1]->getNumOfDims() == 1) {
                reorder123DimsTo4DimsForBroadcast(input_tensors_[INPUT_INDEX_1]->getDims(), dims_in_1);
            }
            NDims broadcasted_dims;
            bool ret = getBroadcastedDims(dims_in_0, dims_in_1, broadcasted_dims);
            CHECK_EXPR_RETURN_FAILURE(true == ret, "Invalid input dims, which are not broadcastable.");
            if (output_tensor_->getDims() != broadcasted_dims) {
                output_tensor_->reconfigureDimsAndBuffer(broadcasted_dims);
            }
        }
    }

    input_broadcast_0_ = input_tensors_[INPUT_INDEX_0];
    if (input_tensors_[INPUT_INDEX_0]->getDims() != output_tensor_->getDims()) {
        CHECK_EXPR_RETURN_FAILURE(isBroadcastable(input_tensors_[INPUT_INDEX_0]->getDims(), output_tensor_->getDims()),
                                  "Invalid input dims, which are not broadcastable.");
        input_broadcast_0_ = std::make_shared<CLTensor>(runtime_,
                                                        precision_,
                                                        input_tensors_[INPUT_INDEX_0]->getDataType(),
                                                        output_tensor_->getDims(),
                                                        input_tensors_[INPUT_INDEX_0]->getDataOrder(),
                                                        input_tensors_[INPUT_INDEX_0]->getScale(),
                                                        input_tensors_[INPUT_INDEX_0]->getZeroPoint(),
                                                        BufferType::DEDICATED,
                                                        parameters_->storage_type);
    }

    input_broadcast_1_ = input_tensors_[INPUT_INDEX_1];
    if (input_tensors_[INPUT_INDEX_1]->getDims() != output_tensor_->getDims()) {
        CHECK_EXPR_RETURN_FAILURE(isBroadcastable(input_tensors_[INPUT_INDEX_1]->getDims(), output_tensor_->getDims()),
                                  "Invalid input dims, which are not broadcastable.");
        input_broadcast_1_ = std::make_shared<CLTensor>(runtime_,
                                                        precision_,
                                                        input_tensors_[INPUT_INDEX_1]->getDataType(),
                                                        output_tensor_->getDims(),
                                                        input_tensors_[INPUT_INDEX_1]->getDataOrder(),
                                                        input_tensors_[INPUT_INDEX_1]->getScale(),
                                                        input_tensors_[INPUT_INDEX_1]->getZeroPoint(),
                                                        BufferType::DEDICATED,
                                                        parameters_->storage_type);
    }

    const Dim4 output_dim = output_tensor_->getDim();
    const Dim4 input_dim_1 = input_tensors_[INPUT_INDEX_1]->getDim();

    Status status = Status::SUCCESS;
    auto input_tensor_0 = input_tensors_[INPUT_INDEX_0];
    auto input_tensor_1 = input_tensors_[INPUT_INDEX_1];
    if (input_tensor_0 != input_broadcast_0_ && input_tensor_0->is_const()) {
        status = input_tensor_0->broadCastTo(input_broadcast_0_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "input_tensor_0->broadCastTo execute failure\n");
    }
    if (input_tensor_1 != input_broadcast_1_ && input_tensor_1->is_const()) {
        status = input_tensor_1->broadCastTo(input_broadcast_1_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "input_tensor_1->broadCastTo execute failure\n");
    }

    if (precision_ == PrecisionType::UINT8 || precision_ == PrecisionType::INT8) {
        quantized_add_ = std::make_shared<CLAddQuantized>(runtime_, precision_);
        status = quantized_add_->initialize(
            {input_broadcast_0_, input_broadcast_1_}, output_tensor_, parameters_->activation_info);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "CLAddQuantized initialize failure\n");
    } else if (parameters_->storage_type == StorageType::TEXTURE) {
        CHECK_EXPR_RETURN_FAILURE(2 == input_tensors_.size(), "CLAddTextureImpl only support 2 inputs\n");
        texture_impl_ = std::make_shared<CLAddTextureImpl>(this);
        status = texture_impl_->initialize();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "CLAddTextureImpl initialize failure\n");
    } else {
        if (input_broadcast_1_ != input_tensors_[INPUT_INDEX_1] && input_broadcast_0_ == input_tensors_[INPUT_INDEX_0] &&
            output_dim.n == input_dim_1.n && output_dim.c == input_dim_1.c && (output_dim.h * output_dim.w) % 8 == 0 &&
            input_dim_1.h == 1 && input_dim_1.w == 1) {
            is_vector_add_ = true;
            if (parameters_->coeff.size() == 2 &&
                parameters_->activation_info.activation() == ActivationInfo::ActivationType::RELU) {
                status = runtime_->setKernel(&kernel_, "RELUeltwise_add_vector_constant", precision_);
                CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel RELUeltwise_add_vector_constant failure\n");
                parameters_->activation_info.disable();
            } else {
                status = runtime_->setKernel(&kernel_, "eltwise_add_vector_constant", precision_);
                CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel eltwise_add_vector_constant failure\n");
            }
        } else {
            if (output_tensor_->getDataType() == DataType::INT32) {
                status = runtime_->setKernel(&kernel_, "eltwise_add_zero_one_int", PrecisionType::FP32);
                CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel eltwise_add_zero_one_int failure\n");
            } else if (parameters_->coeff.size() == 2 && parameters_->activation_info.isEnabled() &&
                       parameters_->activation_info.activation() == ActivationInfo::ActivationType::RELU) {
                status = runtime_->setKernel(&kernel_, "RELUeltwise_add_zero_one", precision_);
                CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel RELUeltwise_add_zero_one failure\n");
                parameters_->activation_info.disable();
            } else {
                status = runtime_->setKernel(&kernel_, "eltwise_add_zero_one", precision_);
                CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel eltwise_add_zero_one failure\n");
            }
        }

        if (input_tensors_.size() > 2) { // for Caffe model
            status = runtime_->setKernel(&kernel_one_input_, "eltwise_add_two_more", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel eltwise_add_two_more failure\n");
        }

        if (parameters_->activation_info.isEnabled()) {
            auto activation_parameters = std::make_shared<ActivationParameters>();
            activation_parameters->activation_info = parameters_->activation_info;

            cl_activation_ = std::make_shared<CLActivation>(runtime_, precision_);
            status = cl_activation_->initialize({output_tensor_}, {output_tensor_}, activation_parameters);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "activation initialize failure\n");
        }
    }  // parameters_->storage_type != StorageType::TEXTURE

    return Status::SUCCESS;
}

Status CLAdd::execute() {
    ENN_DBG_PRINT("CLAdd::execute() is called\n");
    for (auto input_tensor : input_tensors_) {
        if (input_tensor->getTotalSizeFromDims() == 0) {
            ENN_DBG_PRINT("CLAdd execute return for zero_sized input\n");
            return Status::SUCCESS;
        }
    }

    Status status = Status::SUCCESS;
    auto input_tensor_0 = input_tensors_[INPUT_INDEX_0];
    auto input_tensor_1 = input_tensors_[INPUT_INDEX_1];
    if (input_tensor_0 != input_broadcast_0_ && !input_tensor_0->is_const()) {
        status = input_tensor_0->broadCastTo(input_broadcast_0_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "input_tensor_0->broadCastTo execute failure\n");
    }
    if (input_tensor_1 != input_broadcast_1_ && !input_tensor_1->is_const()) {
        status = input_tensor_1->broadCastTo(input_broadcast_1_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "input_tensor_1->broadCastTo execute failure\n");
    }

    if (precision_ == PrecisionType::UINT8 || precision_ == PrecisionType::INT8) {
        status = quantized_add_->execute();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "CLAddQuantized execute failure\n");
    } else {
        if (parameters_->storage_type == StorageType::TEXTURE) {
            status = texture_impl_->execute();
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "CLAddTextureImpl execute failure\n");
        } else {
            status = addFloat();
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "addFloat execute failure\n");
        }
    }

    return Status::SUCCESS;
}

Status CLAdd::addFloat() {
    ENN_DBG_PRINT("CLAdd::addFloat() is called\n");

    Status status = Status::SUCCESS;
    auto output_dim = output_tensor_->getDim();
    if (is_vector_add_) {
        const uint32_t hw_size = output_dim.h * output_dim.w;
        status = runtime_->setKernelArg(kernel_.get(),
                                        input_broadcast_0_->getDataPtr(),
                                        input_broadcast_1_->getDataPtr(),
                                        output_tensor_->getDataPtr(),
                                        parameters_->coeff[0],
                                        parameters_->coeff[1],
                                        hw_size);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");

        const size_t local[2] = {1, 24};
        const size_t global[2] = {output_dim.n * output_dim.c, alignTo(ceil(hw_size / 8.0), local[1])};
        status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)2, global, local);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "enqueueKernel failure\n");
    } else {
        const uint32_t output_size = output_tensor_->getTotalSizeFromDims();
        status = runtime_->setKernelArg(kernel_.get(),
                                        input_broadcast_0_->getDataPtr(),
                                        input_broadcast_1_->getDataPtr(),
                                        output_tensor_->getDataPtr(),
                                        parameters_->coeff[0],
                                        parameters_->coeff[1],
                                        output_size);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");

        const size_t local[1] = {24};
        const size_t global[1] = {alignTo(ceil(output_size / 8.0), local[0])};
        status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)1, global, local);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "enqueueKernel failure\n");
    }

    if (input_tensors_.size() > 2) {  // for Caffe model
        for (size_t i = 2; i < input_tensors_.size(); ++i) {
            auto input_tensor = input_tensors_[i];
            // TODO(all): Need optimize. If the CTS/VTS check pass, then here can be moved to initialize.
            if (input_tensor->getDims() != output_tensor_->getDims()) {
                CHECK_EXPR_RETURN_FAILURE(isBroadcastable(input_tensor->getDims(), output_tensor_->getDims()),
                                          "Invalid input dims, which are not broadcastable.");
                status = input_tensor->broadCastTo(input_broadcast_0_);
                CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "input_tensors[%zu]->broadCastTo execute failure\n", i);
            } else {
                input_broadcast_0_ = input_tensor;
            }

            status = runtime_->setKernelArg(kernel_one_input_.get(),
                                            input_broadcast_0_->getDataPtr(),
                                            output_tensor_->getDataPtr(),
                                            parameters_->coeff[i]);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");

            const size_t global[2] = {output_dim.n, output_dim.c * output_dim.h * output_dim.w};
            const size_t local[2] = {1, static_cast<size_t>(findMaxFactor(global[1], 128))};

            status = runtime_->enqueueKernel(kernel_one_input_.get(), (cl_uint)2, global, local);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "enqueueKernel failure\n");
        }
    }

    return Status::SUCCESS;
}

Status CLAdd::release() {
    ENN_DBG_PRINT("CLAdd::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
