#include "CLNormalization.hpp"

namespace enn {
namespace ud {
namespace gpu {
namespace {
constexpr int INPUT_INDEX = 0;
constexpr int MEAN_INDEX = 1;
constexpr int SCALE_INDEX = 2;
constexpr int OUTPUT_INDEX = 0;
}  // namespace

CLNormalization::CLNormalization(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    ENN_DBG_PRINT("CLNormalization is created\n");

    kernel_uint8_ = nullptr;
    kernel_float_ = nullptr;
    kernel_float_for_fp16_ = nullptr;
    input_ = nullptr;
    output_ = nullptr;
    mean_ = nullptr;
    scale_ = nullptr;

    parameters_.reset(new NormalizationrParameters);
    parameters_->use_FP32_input_for_fp16 = false;
}

Status CLNormalization::initialize(std::vector<std::shared_ptr<ITensor>> inputs,
                                   std::vector<std::shared_ptr<ITensor>> outputs,
                                   std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLNormalization::initialize() is called\n");
    parameters_.reset(new NormalizationrParameters);

    parameters_ = std::static_pointer_cast<NormalizationrParameters>(parameters);
    input_ = std::static_pointer_cast<CLTensor>(inputs[INPUT_INDEX]);
    mean_ = std::static_pointer_cast<CLTensor>(inputs[MEAN_INDEX]);
    scale_ = std::static_pointer_cast<CLTensor>(inputs[SCALE_INDEX]);
    output_ = std::static_pointer_cast<CLTensor>(outputs[OUTPUT_INDEX]);

    Status state = Status::SUCCESS;
    if (input_->getDataType() == DataType::UINT8) {
        state = runtime_->setKernel(&kernel_uint8_, "normalization_uint8", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "set normalization_uint8 kernel failure\n");
    } else {
        if (parameters_->use_FP32_input_for_fp16) {
            state = runtime_->setKernel(&kernel_float_for_fp16_, "normalization_float_for_fp16", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "set normalization_float kernel failure\n");
        } else {
            state = runtime_->setKernel(&kernel_float_, "normalization_float", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "set normalization_float kernel failure\n");
        }
    }
    return Status::SUCCESS;
}

Status CLNormalization::execute() {
    ENN_DBG_PRINT("CLNormalization::execute() is called");
    Status state;
    auto input_data = input_->getDataPtr();
    auto output_data = output_->getDataPtr();
    auto mean_data = mean_->getDataPtr();
    auto scale_data = scale_->getDataPtr();

    auto input_dim = input_->getDim();
    auto output_dim = output_->getDim();

    uint32_t input_height = input_dim.h;
    uint32_t input_width = input_dim.w;
    uint32_t output_batch = output_dim.n;
    uint32_t output_channel = output_dim.c;
    uint32_t output_height = output_dim.h;
    uint32_t output_width = output_dim.w;
    uint32_t spaceW, spaceH;
    spaceH = (input_height - output_height) / 2;
    spaceW = (input_width - output_width) / 2;

    size_t global[3] = {0, 0, 0};
    size_t local[3] = {1, 1, 64};

    global[0] = output_batch * output_channel;
    global[1] = output_height;
    global[2] = alignTo(output_width, local[2]);
    switch (input_->getDataType()) {
    case DataType::UINT8:
        state = runtime_->setKernelArg(kernel_uint8_.get(),
                                       input_data,
                                       mean_data,
                                       output_data,
                                       input_width,
                                       input_height,
                                       spaceW,
                                       spaceH,
                                       output_channel,
                                       output_width,
                                       output_height,
                                       scale_data);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");

        state = runtime_->enqueueKernel(kernel_uint8_.get(), (cl_uint)3, global, local);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
        break;
    case DataType::FLOAT:
        if (parameters_->use_FP32_input_for_fp16) {
            state = runtime_->setKernelArg(kernel_float_for_fp16_.get(),
                                           input_data,
                                           mean_data,
                                           output_data,
                                           input_width,
                                           input_height,
                                           spaceW,
                                           spaceH,
                                           output_channel,
                                           output_width,
                                           output_height,
                                           scale_data);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");

            state = runtime_->enqueueKernel(kernel_float_for_fp16_.get(), (cl_uint)3, global, local);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
        } else {
            state = runtime_->setKernelArg(kernel_float_.get(),
                                           input_data,
                                           mean_data,
                                           output_data,
                                           input_width,
                                           input_height,
                                           spaceW,
                                           spaceH,
                                           output_channel,
                                           output_width,
                                           output_height,
                                           scale_data);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");

            state = runtime_->enqueueKernel(kernel_float_.get(), (cl_uint)3, global, local);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
        }
        break;
    default: ERROR_PRINT("Not supported input data type"); return Status::FAILURE;
    }

    return Status::SUCCESS;
}

Status CLNormalization::release() { return Status::SUCCESS; }

}  // namespace gpu
}  // namespace ud
}  // namespace enn
