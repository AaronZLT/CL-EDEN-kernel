#pragma once

#include "userdriver/common/operator_interfaces/common/Error.hpp"
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/operators/IArgMax.hpp"
#include "userdriver/cpu/common/NEONTensor.hpp"

namespace enn {
namespace ud {
namespace cpu {

class ArgMax : public IArgMax {
public:
    explicit ArgMax(const PrecisionType& precision);
    Status initialize(const std::shared_ptr<ITensor> input, const int32_t& axis, std::shared_ptr<ITensor> output,
                      const bool androidNN = false, const bool isNCHW = false);
    Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output);
    Status release();

private:
    int32_t axis_;
    PrecisionType precision_;
};  // class ArgMax

}  // namespace cpu
}  // namespace ud
}  // namespace enn
