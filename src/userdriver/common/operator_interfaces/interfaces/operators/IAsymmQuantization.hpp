#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/ITensor.hpp"

namespace enn {
namespace ud {

class IAsymmQuantization {
public:
    virtual Status initialize(const std::shared_ptr<ITensor> input, const int32_t& channel, const int32_t& width,
                              const int32_t& height, const float& scale, const int32_t& zeroPoint) = 0;
    virtual Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) = 0;
    virtual Status release() = 0;
    virtual ~IAsymmQuantization() = default;
};  // class IAsymmQuantization

}  // namespace ud
}  // namespace enn
