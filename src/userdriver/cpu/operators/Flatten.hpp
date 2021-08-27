#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/operators/IFlatten.hpp"
#include "userdriver/cpu/common/NEONTensor.hpp"

namespace enn {
namespace ud {
namespace cpu {

class Flatten : public IFlatten {
public:
    explicit Flatten(const PrecisionType& precision);

    Status initialize(const int32_t& axis, const int32_t& end_axis);

    Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output, std::vector<uint32_t>* output_dim);

    Status release();

private:
    PrecisionType precision_;
    int32_t axis_, end_axis_;

    template <typename T>
    Status executeKernel(const std::shared_ptr<NEONTensor<T>> input_tensor, std::shared_ptr<NEONTensor<T>> output_tensor,
                         std::vector<uint32_t>* output_dim);
};

}  // namespace cpu
}  // namespace ud
}  // namespace enn
