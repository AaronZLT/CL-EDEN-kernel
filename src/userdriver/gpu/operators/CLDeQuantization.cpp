#include "userdriver/gpu/operators/CLDeQuantization.hpp"

namespace enn {
namespace ud {
namespace gpu {

namespace {
constexpr int INPUT_INDEX = 0;
constexpr int OUTPUT_INDEX = 0;
}  // namespace

CLDeQuantization::CLDeQuantization(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision), is_inited_(false), kernel_(nullptr) {
    ENN_DBG_PRINT("CLDeQuantization is created");
    is_inited_ = false;
    kernel_ = nullptr;
    input_ = nullptr;
    output_ = nullptr;
    parameters_.reset(new DeQuantizationParameters);
    parameters_->per_channel_quant = false;
    parameters_->channel_dim = 0;
    parameters_->scales.clear();
}

Status CLDeQuantization::initialize(std::vector<std::shared_ptr<ITensor>> inputs,
                                    std::vector<std::shared_ptr<ITensor>> outputs,
                                    std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLDeQuantization::initialize() is called");
    if (is_inited_) {
        return Status::SUCCESS;
    }
    is_inited_ = true;

    input_ = std::static_pointer_cast<CLTensor>(inputs.at(INPUT_INDEX));
    output_ = std::static_pointer_cast<CLTensor>(outputs.at(OUTPUT_INDEX));

    parameters_ = std::static_pointer_cast<DeQuantizationParameters>(parameters);
    CHECK_EXPR_RETURN_FAILURE(nullptr != parameters_, "CLDeQuantization must have parameters\n");
    ENN_DBG_PRINT("DeQuantizationParameters: per_channel_quant %d; channel_dim %d;\n",
                  parameters_->per_channel_quant,
                  parameters_->channel_dim);

    const DataType &input_data_type = input_->getDataType();
    std::string kernel_name = "dequantization_";
    if (input_data_type == DataType::INT32) {
        kernel_name += "int";
    } else if (input_data_type == DataType::INT16) {
        kernel_name += "short";
    } else if (input_data_type == DataType::UINT16) {
        kernel_name += "ushort";
    } else if (input_data_type == DataType::UINT8) {
        kernel_name += "uint8";
    } else if (input_data_type == DataType::INT8) {
        kernel_name += "int8";
    }

    if (parameters_->per_channel_quant == true) {
        kernel_name += "_param";
    }
    Status state = runtime_->setKernel(&kernel_, kernel_name.c_str(), precision_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
    return Status::SUCCESS;
}

Status CLDeQuantization::execute() {
    ENN_DBG_PRINT("CLDeQuantization::execute() is called");
    // for zero_sized input
    if (input_->getTotalSizeFromDims() == 0) {
        return Status::SUCCESS;
    }
    auto input_tensor = std::static_pointer_cast<CLTensor>(input_);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output_);
    auto input_data = input_tensor->getDataPtr();
    auto output_data = output_tensor->getDataPtr();
    Dim4 input_dim = input_tensor->getDim();
    uint32_t input_batch = input_dim.n;
    uint32_t input_channel = input_dim.c;
    uint32_t input_height = input_dim.h;
    uint32_t input_width = input_dim.w;

    if (parameters_->per_channel_quant == false) {
        size_t global[3] = {0, 0, 0};
        size_t local[3] = {1, 1, 32};
        global[0] = input_batch;
        int gsize1 = input_channel;
        global[1] = alignTo(gsize1, local[1]);
        int gsize2 = ceil(input_height * input_width / 8.0);
        global[2] = alignTo(gsize2, local[2]);

        Status state = runtime_->setKernelArg(kernel_.get(),
                                              input_data,
                                              output_data,
                                              input_tensor->getScale(),
                                              input_tensor->getZeroPoint(),
                                              input_channel,
                                              input_height,
                                              input_width);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");
        state = runtime_->enqueueKernel(kernel_.get(), (cl_uint)3, global, local);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
    } else {
        size_t global_param[3] = {input_batch, input_channel, input_height * input_width};
        size_t local_param[3] = {1, 1, 1};

        uint32_t scale_size = parameters_->scales.size();
        Dim4 scale_dims_ = {scale_size, 1, 1, 1};
        auto scale_tensor = std::make_shared<CLTensor>(runtime_, precision_, parameters_->scales.data(), scale_dims_);
        auto scale_data = scale_tensor->getDataPtr();

        Status state = runtime_->setKernelArg(kernel_.get(), input_data, output_data, scale_data, parameters_->channel_dim);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");
        state = runtime_->enqueueKernel(kernel_.get(), (cl_uint)3, global_param, local_param);
    }

    return Status::SUCCESS;
}

Status CLDeQuantization::release() {
    ENN_DBG_PRINT("CLDeQuantization::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
