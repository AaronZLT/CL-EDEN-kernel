#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/ITensor.hpp"

namespace enn {
namespace ud {

class INormalQuantization {
public:
    virtual Status initialize(const std::shared_ptr<ITensor> input, const int32_t& width, const int32_t& height,
                              const int32_t& channel, const std::shared_ptr<ITensor> means,
                              const std::shared_ptr<ITensor> scales, const std::shared_ptr<ITensor> frac_lens) = 0;
    virtual Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) = 0;
    virtual Status release() = 0;
    virtual ~INormalQuantization() = default;
};  // class INormalQuantization

}  // namespace ud
}  // namespace enn
