#include "userdriver/gpu/operators/CLQuantization.hpp"

namespace enn {
namespace ud {
namespace gpu {

namespace {
constexpr int INPUT_INDEX = 0;
constexpr int OUTPUT_INDEX = 0;
}  // namespace

CLQuantization::CLQuantization(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision), is_inited_(false), kernel_(nullptr) {
    ENN_DBG_PRINT("CLQuantization is created");
}

Status CLQuantization::initialize(std::vector<std::shared_ptr<ITensor>> inputs,
                                  std::vector<std::shared_ptr<ITensor>> outputs,
                                  std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLQuantization::initialize() is called");
    if (is_inited_)
        return Status::SUCCESS;
    is_inited_ = true;
    Status state = Status::FAILURE;

    input_ = std::static_pointer_cast<CLTensor>(inputs.at(INPUT_INDEX));
    output_ = std::static_pointer_cast<CLTensor>(outputs.at(OUTPUT_INDEX));

    if (precision_ != PrecisionType::FP32 && precision_ != PrecisionType::FP16) {
        ENN_DBG_PRINT("CLQuantization don't support such precision: %d", precision_);
        //  set the precision for NPU_GPU DeepLabv3 model, the precission of the model is UINT8
        //  and the model has a QUANTIZE layer with the float32 input datatype.
        precision_ = PrecisionType::FP32;
        ENN_DBG_PRINT("Change precision to PrecisionType::FP32");
    }
    const DataType &output_data_type = output_->getDataType();

    if (output_data_type == DataType::UINT16) {
        state = runtime_->setKernel(&kernel_, "quantization_ushort", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
    } else if (output_data_type == DataType::INT16) {
        state = runtime_->setKernel(&kernel_, "quantization_short", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
    } else if (output_data_type == DataType::INT8) {
        state = runtime_->setKernel(&kernel_, "quantization_signed", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
    } else if (output_data_type == DataType::UINT8) {
        state = runtime_->setKernel(&kernel_, "quantization", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
    }
    return Status::SUCCESS;
}

Status CLQuantization::execute() {
    ENN_DBG_PRINT("CLQuantization::execute() is called");
    // for zero_sized input
    if (input_->getTotalSizeFromDims() == 0) {
        return Status::SUCCESS;
    }
    auto input_tensor = std::static_pointer_cast<CLTensor>(input_);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output_);

    Dim4 input_dim = input_tensor->getDim();
    uint32_t input_batch = input_dim.n;
    uint32_t input_channel = input_dim.c;
    uint32_t input_height = input_dim.h;
    uint32_t input_width = input_dim.w;

    size_t global[3] = {0, 0, 0};
    size_t local[3] = {1, 1, 32};
    global[0] = input_batch;
    int gsize1 = input_channel;
    global[1] = alignTo(gsize1, local[1]);
    int gsize2 = ceil(input_height * input_width / 8.0);
    global[2] = alignTo(gsize2, local[2]);

    Status state = Status::FAILURE;
    state = runtime_->setKernelArg(kernel_.get(),
                                   input_tensor->getDataPtr(),
                                   output_tensor->getDataPtr(),
                                   1 / output_tensor->getScale(),
                                   output_tensor->getZeroPoint(),
                                   input_channel,
                                   input_height,
                                   input_width);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");
    state = runtime_->enqueueKernel(kernel_.get(), (cl_uint)3, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");

    return state;
}

Status CLQuantization::release() {
    ENN_DBG_PRINT("CLQuantization::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
