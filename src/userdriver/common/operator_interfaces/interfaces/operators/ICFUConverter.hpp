#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/ITensor.hpp"

namespace enn {
namespace ud {

class ICFUConverter {
public:
    virtual Status initialize(const std::shared_ptr<ITensor> input, const int32_t& width, const int32_t& height,
                              const int32_t& channel, const int32_t& cols_in_cell, const int32_t& lines_in_cell,
                              const int32_t& interleaved_slices, const int32_t& pad_value = 0) = 0;
    virtual Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) = 0;
    virtual Status release() = 0;
    virtual ~ICFUConverter() = default;
};  // class ICFUConverter

}  // namespace ud
}  // namespace enn
