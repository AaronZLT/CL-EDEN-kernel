#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/ITensor.hpp"

namespace enn {
namespace ud {

class ISoftmax {
public:
    virtual Status initialize(const std::shared_ptr<ITensor> input, const int32_t& width, const int32_t& height,
                              const int32_t& channel, const int32_t& number, const float& beta, const int32_t& axis) = 0;
    virtual Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) = 0;
    virtual Status release() = 0;
    virtual ~ISoftmax() = default;
};  // class ISoftmax

}  // namespace ud
}  // namespace enn