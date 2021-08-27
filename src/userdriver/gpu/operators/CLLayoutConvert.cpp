#include "CLLayoutConvert.hpp"

namespace enn {
namespace ud {
namespace gpu {
namespace {
constexpr int INPUT_INDEX = 0;
constexpr int OUTPUT_INDEX = 0;
}  // namespace

CLLayoutConvert::CLLayoutConvert(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    ENN_DBG_PRINT("CLLayoutConvert is created\n");
    input_tensor_ = nullptr;
    output_tensor_ = nullptr;
    parameters_ = nullptr;
}

Status CLLayoutConvert::initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                                   const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                                   const std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLLayoutConvert::initialize() is called\n");

    input_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(INPUT_INDEX));
    output_tensor_ = std::static_pointer_cast<CLTensor>(output_tensors.at(OUTPUT_INDEX));
    parameters_ = std::static_pointer_cast<LayoutConvertParameters>(parameters);
    CHECK_EXPR_RETURN_FAILURE(nullptr != parameters_, "CLLayoutConvert must have parameters\n");

    return Status::SUCCESS;
}

Status CLLayoutConvert::execute() {
    ENN_DBG_PRINT("CLLayoutConvert::execute() is called\n");

    switch (parameters_->data_order_change_type) {
    case DataOrderChangeType::NHWC2NCHW: {
        Status status = input_tensor_->convertToNCHW(output_tensor_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "CLTensor convertToNCHW failure\n");
        break;
    }
    case DataOrderChangeType::NCHW2NHWC: {
        Status status = input_tensor_->convertToNHWC(output_tensor_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "CLTensor convertToNHWC failure\n");
        break;
    }
    case DataOrderChangeType::OTHER: {
        UNUSED(input_tensor_);
        UNUSED(output_tensor_);
        break;
    }
    default: ERROR_PRINT("Non-Support DataOrderChangeType"); return Status::FAILURE;
    }

    return Status::SUCCESS;
}

Status CLLayoutConvert::release() {
    ENN_DBG_PRINT("CLLayoutConvert::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
