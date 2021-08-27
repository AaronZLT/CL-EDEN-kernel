#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/ITensor.hpp"

namespace enn {
namespace ud {

class IAsymmDequantization {
public:
    virtual Status initialize(const std::shared_ptr<ITensor> input, const int32_t& numOfData, const float& scale,
                              const int32_t& zeroPoint, const uint32_t& imgSize) = 0;
    virtual Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) = 0;
    virtual Status release() = 0;
    virtual ~IAsymmDequantization() = default;
};  // class IAsymmDequantization

}  // namespace ud
}  // namespace enn
