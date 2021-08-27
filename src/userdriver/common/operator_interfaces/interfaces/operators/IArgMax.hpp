#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/ITensor.hpp"

namespace enn {
namespace ud {

// Nice to have, ToDo(empire.jung, TBD): Consider variadic argument for all operartor initialize()
class IArgMax {
public:
    virtual Status initialize(const std::shared_ptr<ITensor> input, const int32_t &axis, std::shared_ptr<ITensor> output,
                              const bool androidNN = false, const bool isNCHW = false) = 0;
    virtual Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) = 0;
    virtual Status release() = 0;
    virtual ~IArgMax() = default;
};  // class IArgMax

}  // namespace ud
}  // namespace enn
