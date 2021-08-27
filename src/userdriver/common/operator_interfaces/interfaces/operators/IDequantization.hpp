#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/ITensor.hpp"

namespace enn {
namespace ud {

class IDequantization {
public:
    virtual Status initialize(const std::shared_ptr<ITensor> input, const int32_t& numOfData,
                              const std::shared_ptr<ITensor> frac_lens, const uint32_t& imgSize) = 0;
    virtual Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) = 0;
    virtual Status release() = 0;
    virtual ~IDequantization() = default;
};  // class IDequantization

}  // namespace ud
}  // namespace enn
