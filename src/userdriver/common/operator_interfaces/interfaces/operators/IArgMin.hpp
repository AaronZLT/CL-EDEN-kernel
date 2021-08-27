#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/ITensor.hpp"

namespace enn {
namespace ud {

class IArgMin {
public:
    virtual Status initialize(const std::shared_ptr<ITensor> input, const int32_t &axis, std::shared_ptr<ITensor> output,
                              const bool androidNN = false, const bool isNCHW = false) = 0;
    virtual Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) = 0;
    virtual Status release() = 0;
    virtual ~IArgMin() = default;
};  // class IArgMin

}  // namespace ud
}  // namespace enn
