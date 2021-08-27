#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/ITensor.hpp"

namespace enn {
namespace ud {

class IPad {
public:
    virtual Status initialize(const std::shared_ptr<ITensor> input, const int32_t& width, const int32_t& height,
                              const int32_t& channel, const std::vector<int32_t>& padFront,
                              const std::vector<int32_t>& padEnd, const std::vector<float>& padValue) = 0;
    virtual Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) = 0;
    virtual Status release() = 0;
    virtual ~IPad() = default;
};  // class IPad

}  // namespace ud
}  // namespace enn
