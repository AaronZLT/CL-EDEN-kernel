#include "userdriver/cpu/operators/neon_impl/CommonUtil.hpp"
#include "ArgMax.hpp"

namespace enn {
namespace ud {
namespace cpu {

ArgMax::ArgMax(const PrecisionType& precision) {
    precision_ = precision;
    axis_ = 0;
}

Status ArgMax::initialize(const std::shared_ptr<ITensor> input, const int32_t& axis, std::shared_ptr<ITensor> output,
                          const bool androidNN, const bool isNCHW) {
    CHECK_EXPR_RETURN_FAILURE(isValidAxis(axis, input->getNumOfDims()), "Invalid axis.");
    ENN_UNUSED(input);
    ENN_UNUSED(isNCHW);
    axis_ = axis;
    if (axis_ < 0) {
        axis_ += input->getNumOfDims();
    }
    if (androidNN) {
        NDims expected_out_dims = input->getDims();
        expected_out_dims[axis_] = 1;
        if (output->getDims() != expected_out_dims) {
            output->reconfigureDimsAndBuffer(expected_out_dims);
        }
    }
    return Status::SUCCESS;
}

Status ArgMax::execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) {
    if (precision_ == PrecisionType::UINT8) {
        DEBUG_PRINT("CPU ArgMax UINT8 is not supported\n");
        return Status::FAILURE;
    } else {
        return evalFloat(input, output, axis_, true);
    }
}

Status ArgMax::release() {
    return Status::SUCCESS;
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
