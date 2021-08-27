#pragma once

#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/common/operator_interfaces/common/Error.hpp"
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/operators/INormalization.hpp"
#include "userdriver/cpu/common/NEONTensor.hpp"

namespace enn {
namespace ud {
namespace cpu {

class Normalization : public INormalization {
public:
    explicit Normalization(const PrecisionType& precision);

    Status initialize(const std::shared_ptr<ITensor> input, const std::shared_ptr<ITensor> mean,
                      const std::shared_ptr<ITensor> scale, const uint8_t& bgr_transpose = 0);

    Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output);

    template <typename T>
    Status executeKernel(const std::shared_ptr<NEONTensor<T>> input_tensor,
                         std::shared_ptr<NEONTensor<float>> output_tensor);

    Status release();

private:
    std::shared_ptr<NEONTensor<float>> mean_;
    std::shared_ptr<NEONTensor<float>> scale_;
    uint8_t bgr_transpose_;
    PrecisionType precision_;
};

}  // namespace cpu
}  // namespace ud
}  // namespace enn
