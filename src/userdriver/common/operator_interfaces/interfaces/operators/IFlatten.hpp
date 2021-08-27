#pragma once

#include <vector>
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/ITensor.hpp"

namespace enn {
namespace ud {

class IFlatten {
public:
    virtual Status initialize(const int32_t &axis, const int32_t &end_axis) = 0;
    virtual Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output,
                           std::vector<uint32_t> *output_dim = nullptr) = 0;
    virtual Status release() = 0;
    virtual ~IFlatten() = default;
};  // class IFlatten

}  // namespace ud
}  // namespace enn
