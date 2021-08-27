#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/ITensor.hpp"

namespace enn {
namespace ud {

class IConcat {
public:
    virtual Status initialize(const std::vector<std::shared_ptr<ITensor>> input, const int32_t& inputNum,
                              const int32_t& number, const int32_t& channel, const int32_t& height, const int32_t& width,
                              const int32_t& axis) = 0;
    virtual Status execute(const std::vector<std::shared_ptr<ITensor>> input, std::shared_ptr<ITensor> output) = 0;
    virtual Status release() = 0;
    virtual ~IConcat() = default;
};  // class IConcat

}  // namespace ud
}  // namespace enn
