#include "CLPad.hpp"

namespace enn {
namespace ud {
namespace gpu {
namespace {
constexpr int INPUT_INDEX = 0;
constexpr int PADDING_INDEX = 1;
constexpr int PAD_VALUE_INDEX = 2;
constexpr int OUTPUT_INDEX = 0;
}  // namespace

CLPad::CLPad(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    ENN_DBG_PRINT("CLPad is created\n");
    input_tensor_ = nullptr;
    padding_tensor_ = nullptr;
    pad_value_tensor_ = nullptr;
    output_tensor_ = nullptr;
    parameters_ = nullptr;
    kernel_ = nullptr;
    tmp_padding_ = nullptr;
    tmp_buffer_ = nullptr;
}

Status CLPad::initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                         const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                         const std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLPad::initialize() is called\n");

    input_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(INPUT_INDEX));
    padding_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(PADDING_INDEX));
    pad_value_tensor_ =
        input_tensors.size() >= 3 ? std::static_pointer_cast<CLTensor>(input_tensors.at(PAD_VALUE_INDEX)) : nullptr;
    output_tensor_ = std::static_pointer_cast<CLTensor>(output_tensors.at(OUTPUT_INDEX));
    parameters_ = std::static_pointer_cast<PadParameters>(parameters);
    CHECK_AND_RETURN_ERR(nullptr == parameters_, Status::FAILURE, "CLPad must have parameters\n");

    if (pad_value_tensor_ != nullptr) {
        if (precision_ == PrecisionType::UINT8) {
            pad_value_tensor_->readData(&(parameters_->quant_pad_value));
        } else if (pad_value_tensor_->getDataType() == DataType::HALF) {
            tmp_buffer_ = std::make_shared<half_float::half>();
            pad_value_tensor_->readData(tmp_buffer_.get());
            parameters_->pad_value = (float)(*(tmp_buffer_.get()));
        } else {
            pad_value_tensor_->readData(&(parameters_->pad_value));
        }
    }

    parameters_->padding = std::vector<int32_t>(8, 0);
    tmp_padding_ = make_shared_array<int32_t>(padding_tensor_->getTotalSizeFromDims());
    padding_tensor_->readData(tmp_padding_.get());

    if (parameters_->androidNN && parameters_->isNCHW) {
        // isNCHW == true implies that its input is a 4-D tensor
        CHECK_AND_RETURN_ERR(padding_tensor_->getTotalSizeFromDims() != 8, Status::FAILURE, "Invalid pad shape.\n");
        parameters_->padding[0] = tmp_padding_.get()[0];
        parameters_->padding[1] = tmp_padding_.get()[1];
        parameters_->padding[2] = tmp_padding_.get()[6];
        parameters_->padding[3] = tmp_padding_.get()[7];
        parameters_->padding[4] = tmp_padding_.get()[2];
        parameters_->padding[5] = tmp_padding_.get()[3];
        parameters_->padding[6] = tmp_padding_.get()[4];
        parameters_->padding[7] = tmp_padding_.get()[5];
    } else {
        memcpy(parameters_->padding.data(), tmp_padding_.get(), sizeof(int32_t) * padding_tensor_->getTotalSizeFromDims());
    }

    NDims input_dim = input_tensor_->getDims();
    NDims output_dim = input_dim;
    for (uint32_t i = 0; i < input_dim.size(); i++) {
        output_dim[i] += parameters_->padding[i * 2];
        output_dim[i] += parameters_->padding[i * 2 + 1];
    }

    if (parameters_->androidNN && !isDimsSame(output_dim, output_tensor_->getDims())) {
        output_tensor_->reconfigureDimsAndBuffer(output_dim);
    }

    Status status = runtime_->setKernel(&kernel_, "Pad8", precision_);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "setKernel Pad8 failure\n");

    return Status::SUCCESS;
}

Status CLPad::execute() {
    ENN_DBG_PRINT("CLPad::execute() is called\n");

    if (precision_ == PrecisionType::UINT8) {
        Status status = pad_quant();
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "pad_quant execute failure\n");
    } else {
        Status status = pad_float();
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "pad_float execute failure\n");
    }

    return Status::SUCCESS;
}

Status CLPad::pad_quant() {
    ENN_DBG_PRINT("CLPad::pad_quant() is called\n");
    auto input_data = input_tensor_->getDataPtr();
    auto output_data = output_tensor_->getDataPtr();
    auto input_dim = input_tensor_->getDim();
    auto output_dim = output_tensor_->getDim();
    parameters_->quant_pad_value =
        parameters_->quant_pad_value == 0 ? output_tensor_->getZeroPoint() : parameters_->quant_pad_value;

    const int32_t padding_n_t = parameters_->padding[0];
    const int32_t padding_c_t = parameters_->padding[2];
    const int32_t padding_t = parameters_->padding[4];
    const int32_t padding_l = parameters_->padding[6];

    const uint32_t batch_top = padding_n_t;
    const uint32_t batch_down = input_dim.n + padding_n_t;
    const uint32_t high_top = padding_t;
    const uint32_t high_down = input_dim.h + padding_t;
    const uint32_t width_left = padding_l;
    const uint32_t width_right = input_dim.w + padding_l;
    const uint32_t channel_top = padding_c_t;
    const uint32_t channel_down = input_dim.c + padding_c_t;

    Status status = runtime_->setKernelArg(kernel_.get(),
                                           input_data,
                                           output_data,
                                           input_dim.n,
                                           input_dim.c,
                                           input_dim.h,
                                           input_dim.w,
                                           output_dim.n,
                                           output_dim.c,
                                           output_dim.h,
                                           output_dim.w,
                                           batch_top,
                                           batch_down,
                                           high_top,
                                           high_down,
                                           width_left,
                                           width_right,
                                           channel_top,
                                           channel_down,
                                           parameters_->quant_pad_value);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "setKernelArg failure\n");

    size_t global[3] = {0, 0, 0};
    size_t local[3] = {1, 1, 1};
    global[0] = output_dim.n;
    global[1] = alignTo(output_dim.c, local[1]);
    global[2] = alignTo(ceil(output_dim.h * output_dim.w), local[2]);

    status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)3, global, local);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "execute kernel failure\n");

    return Status::SUCCESS;
}

Status CLPad::pad_float() {
    ENN_DBG_PRINT("CLPad::pad_float() is called\n");
    auto input_data = input_tensor_->getDataPtr();
    auto output_data = output_tensor_->getDataPtr();
    auto input_dim = input_tensor_->getDim();
    auto output_dim = output_tensor_->getDim();

    const int32_t padding_n_t = parameters_->padding[0];
    const int32_t padding_c_t = parameters_->padding[2];
    const int32_t padding_t = parameters_->padding[4];
    const int32_t padding_l = parameters_->padding[6];

    const uint32_t batch_top = padding_n_t;
    const uint32_t batch_down = input_dim.n + padding_n_t;
    const uint32_t high_top = padding_t;
    const uint32_t high_down = input_dim.h + padding_t;
    const uint32_t width_left = padding_l;
    const uint32_t width_right = input_dim.w + padding_l;
    const uint32_t channel_top = padding_c_t;
    const uint32_t channel_down = input_dim.c + padding_c_t;

    Status status = runtime_->setKernelArg(kernel_.get(),
                                           input_data,
                                           output_data,
                                           input_dim.n,
                                           input_dim.c,
                                           input_dim.h,
                                           input_dim.w,
                                           output_dim.n,
                                           output_dim.c,
                                           output_dim.h,
                                           output_dim.w,
                                           batch_top,
                                           batch_down,
                                           high_top,
                                           high_down,
                                           width_left,
                                           width_right,
                                           channel_top,
                                           channel_down,
                                           parameters_->pad_value);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "setKernelArg failure\n");

    size_t global[3] = {0, 0, 0};
    size_t local[3] = {1, 1, 32};
    global[0] = output_dim.n;
    global[1] = alignTo(output_dim.c, local[1]);
    global[2] = alignTo(ceil(output_dim.h * output_dim.w), local[2]);

    status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)3, global, local);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "enqueueKernel failure\n");

    return Status::SUCCESS;
}

Status CLPad::release() {
    ENN_DBG_PRINT("CLPad::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
