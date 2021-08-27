#include "CLCast.hpp"

namespace enn {
namespace ud {
namespace gpu {
namespace {
constexpr int INPUT_INDEX = 0;
constexpr int OUTPUT_INDEX = 0;

const std::map<int, const std::string> kKernelInfo = {
    {(static_cast<int>(DataType::FLOAT) << 8) | static_cast<int>(DataType::FLOAT), "cast_float_to_float"},
    {(static_cast<int>(DataType::FLOAT) << 8) | static_cast<int>(DataType::HALF), "cast_float_to_half"},
    {(static_cast<int>(DataType::FLOAT) << 8) | static_cast<int>(DataType::INT32), "cast_float_to_int"},
    {(static_cast<int>(DataType::FLOAT) << 8) | static_cast<int>(DataType::UINT8), "cast_float_to_int8"},
    {(static_cast<int>(DataType::HALF) << 8) | static_cast<int>(DataType::FLOAT), "cast_half_to_float"},
    {(static_cast<int>(DataType::HALF) << 8) | static_cast<int>(DataType::HALF), "cast_half_to_half"},
    {(static_cast<int>(DataType::HALF) << 8) | static_cast<int>(DataType::INT32), "cast_half_to_int"},
    {(static_cast<int>(DataType::HALF) << 8) | static_cast<int>(DataType::UINT8), "cast_half_to_int8"},
    {(static_cast<int>(DataType::INT32) << 8) | static_cast<int>(DataType::FLOAT), "cast_int_to_float"},
    {(static_cast<int>(DataType::INT32) << 8) | static_cast<int>(DataType::HALF), "cast_int_to_half"},
    {(static_cast<int>(DataType::INT32) << 8) | static_cast<int>(DataType::INT32), "cast_int_to_int"},
    {(static_cast<int>(DataType::INT32) << 8) | static_cast<int>(DataType::UINT8), "cast_int_to_int8"},
    {(static_cast<int>(DataType::UINT8) << 8) | static_cast<int>(DataType::FLOAT), "cast_int8_to_float"},
    {(static_cast<int>(DataType::UINT8) << 8) | static_cast<int>(DataType::HALF), "cast_int8_to_half"},
    {(static_cast<int>(DataType::UINT8) << 8) | static_cast<int>(DataType::INT32), "cast_int8_to_int"},
    {(static_cast<int>(DataType::UINT8) << 8) | static_cast<int>(DataType::UINT8), "cast_int8_to_int8"},
    {(static_cast<int>(DataType::BOOL) << 8) | static_cast<int>(DataType::BOOL), "cast_bool_to_bool"},
    {(static_cast<int>(DataType::INT16) << 8) | static_cast<int>(DataType::INT16), "cast_uint16_to_uint16"},
};

}  // namespace

CLCast::CLCast(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    ENN_DBG_PRINT("CLCast is created\n");
    input_tensor_ = nullptr;
    output_tensor_ = nullptr;
    parameters_ = nullptr;
    kernel_ = nullptr;
}

Status CLCast::initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                          const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                          const std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLCast::initialize() is called\n");

    input_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(INPUT_INDEX));
    output_tensor_ = std::static_pointer_cast<CLTensor>(output_tensors.at(OUTPUT_INDEX));
    parameters_ = std::static_pointer_cast<CastParameters>(parameters);
    CHECK_AND_RETURN_ERR(nullptr == parameters_, Status::FAILURE, "CLCast must have parameters\n");
    ENN_DBG_PRINT("CastParameters Info: in_data_type = %d, out_data_type = %d, androidNN = %d\n",
                  static_cast<int>(parameters_->in_data_type),
                  static_cast<int>(parameters_->out_data_type),
                  parameters_->androidNN);

    if (parameters_->androidNN && output_tensor_->getDims() != input_tensor_->getDims()) {
        output_tensor_->reconfigureDimsAndBuffer(input_tensor_->getDims());
    }

    if (parameters_->in_data_type != DataType::FLOAT && parameters_->in_data_type != DataType::HALF &&
        parameters_->out_data_type != DataType::FLOAT && parameters_->out_data_type != DataType::HALF) {
        precision_ = PrecisionType::FP32;
    } else if (parameters_->in_data_type == DataType::UINT8 && parameters_->out_data_type == DataType::HALF) {
        precision_ = PrecisionType::FP16;
    } else if (parameters_->in_data_type == DataType::UINT8 && parameters_->out_data_type == DataType::FLOAT) {
        precision_ = PrecisionType::FP32;
    }

    const int key = (static_cast<int>(parameters_->in_data_type) << 8) | static_cast<int>(parameters_->out_data_type);
    const auto iter = kKernelInfo.find(key);
    const std::string kernel_name = iter != kKernelInfo.end() ? iter->second : "";
    CHECK_AND_RETURN_ERR(0 == kernel_name.size(), Status::FAILURE, "CLCast NOT support cast the intput/output data type\n");

    Status status = runtime_->setKernel(&kernel_, kernel_name, precision_);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "setKernel %s failure\n", kernel_name.c_str());

    return Status::SUCCESS;
}

Status CLCast::execute() {
    ENN_DBG_PRINT("CLCast::execute() is called\n");

    auto output_dim = output_tensor_->getDim();

    Status status = runtime_->setKernelArg(kernel_.get(), input_tensor_->getDataPtr(), output_tensor_->getDataPtr());
    CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "setKernelArg failure\n");

    size_t global[3] = {static_cast<size_t>(output_dim.n),
                        static_cast<size_t>(output_dim.c),
                        static_cast<size_t>(output_dim.h * output_dim.w)};
    size_t local[3] = {1, 1, static_cast<size_t>(findMaxFactor(global[2], 128))};

    status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)3, global, local);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "execute kernel failure\n");

    return Status::SUCCESS;
}

Status CLCast::release() {
    ENN_DBG_PRINT("CLCast::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
