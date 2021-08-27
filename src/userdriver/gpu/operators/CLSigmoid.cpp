#include "CLSigmoid.hpp"

namespace enn {
namespace ud {
namespace gpu {
namespace {
constexpr int INPUT_INDEX = 0;
constexpr int OUTPUT_INDEX = 0;
}  // namespace

CLSigmoid::CLSigmoid(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    ENN_DBG_PRINT("CLSigmoid is created\n");
    input_tensor_ = nullptr;
    output_tensor_ = nullptr;
    kernel_ = nullptr;
    dequantization_ = nullptr;
    quantization_ = nullptr;
}

Status CLSigmoid::initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                             const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                             const std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLSigmoid::initialize() is called\n");

    input_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(INPUT_INDEX));
    output_tensor_ = std::static_pointer_cast<CLTensor>(output_tensors.at(OUTPUT_INDEX));
    CHECK_EXPR_RETURN_FAILURE(nullptr == parameters, "CLSigmoid doesn't have parameters\n");
    Status status = Status::SUCCESS;
    if (precision_ == PrecisionType::INT8 || precision_ == PrecisionType::UINT8) {
        auto input_dim = input_tensor_->getDim();
        auto output_dim = output_tensor_->getDim();
        auto map_input_tensor = std::make_shared<CLTensor>(runtime_, PrecisionType::FP32, DataType::FLOAT, input_dim);
        auto map_output_tensor = std::make_shared<CLTensor>(runtime_, PrecisionType::FP32, DataType::FLOAT, output_dim);

        dequantization_ = std::make_shared<CLDeQuantization>(runtime_, PrecisionType::FP32);
        quantization_ = std::make_shared<CLQuantization>(runtime_, PrecisionType::FP32);

        auto dequantization_parameters = std::make_shared<DeQuantizationParameters>();
        status = dequantization_->initialize({input_tensor_}, {map_input_tensor}, dequantization_parameters);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "dequantization initialize failure\n");

        status = quantization_->initialize({map_output_tensor}, {output_tensor_}, nullptr);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "quantization initialize failure\n");

        input_tensor_ = map_input_tensor;
        output_tensor_ = map_output_tensor;

        status = runtime_->setKernel(&kernel_, "sigmoid", PrecisionType::FP32);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel sigmoid failure\n");
    } else {
        status = runtime_->setKernel(&kernel_, "sigmoid", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel sigmoid failure\n");
    }

    return Status::SUCCESS;
}

Status CLSigmoid::execute() {
    ENN_DBG_PRINT("CLSigmoid::execute() is called\n");

    if (input_tensor_->getTotalSizeFromDims() == 0) {
        ENN_DBG_PRINT("CLSigmoid execute return for zero_sized input\n");
        return Status::SUCCESS;
    }

    if (precision_ == PrecisionType::UINT8 || precision_ == PrecisionType::INT8) {
        Status status = sigmoidQuant();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "sigmoidQuant execute failure\n");
    } else {
        Status status = sigmoidFloat();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "sigmoidFloat execute failure\n");
    }

    return Status::SUCCESS;
}

Status CLSigmoid::sigmoidQuant() {
    ENN_DBG_PRINT("CLSigmoid::sigmoidQuant() is called\n");
    Status status = Status::SUCCESS;

    status = dequantization_->execute();
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "dequantization execute failure\n");

    status = this->sigmoidFloat();
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "call sigmoidFloat execute failure\n");

    status = quantization_->execute();
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "quantization execute failure\n");

    return Status::SUCCESS;
}

Status CLSigmoid::sigmoidFloat() {
    ENN_DBG_PRINT("CLSigmoid::sigmoidFloat() is called\n");
    auto input_data = input_tensor_->getDataPtr();
    auto output_data = output_tensor_->getDataPtr();
    auto input_dim = input_tensor_->getDim();
    auto output_dim = output_tensor_->getDim();

    Status status = runtime_->setKernelArg(kernel_.get(), input_data, output_data, output_dim.c, output_dim.h, output_dim.w);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel failure\n");

    size_t global[3] = {0, 0, 0};
    size_t local[3] = {1, 1, 32};
    global[0] = input_dim.n;
    global[1] = alignTo(output_dim.c, local[1]);
    global[2] = alignTo(ceil(output_dim.h * output_dim.w / 8.0), local[2]);

    status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)3, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "execute kernel failure\n");

    return Status::SUCCESS;
}

Status CLSigmoid::release() {
    ENN_DBG_PRINT("CLSigmoid::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
