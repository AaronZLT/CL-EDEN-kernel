#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/ITensor.hpp"

namespace enn {
namespace ud {

class INormalization {
public:
    virtual Status initialize(const std::shared_ptr<ITensor> input, const std::shared_ptr<ITensor> mean,
                              const std::shared_ptr<ITensor> scale, const uint8_t &bgr_transpose = 0) = 0;
    virtual Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) = 0;
    virtual Status release() = 0;
    virtual ~INormalization() = default;
};  // class INormalization

}  // namespace ud
}  // namespace enn
