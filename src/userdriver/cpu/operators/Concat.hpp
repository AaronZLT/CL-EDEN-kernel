#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/operators/IConcat.hpp"
#include "userdriver/cpu/common/NEONTensor.hpp"

namespace enn {
namespace ud {
namespace cpu {

class Concat : public IConcat {
public:
    explicit Concat(const PrecisionType& precision);

    Status initialize(const std::vector<std::shared_ptr<ITensor>> input, const int32_t& inputNum, const int32_t& number,
                      const int32_t& channel, const int32_t& height, const int32_t& width, const int32_t& axis);

    Status execute(const std::vector<std::shared_ptr<ITensor>> input, std::shared_ptr<ITensor> output);

    Status release();

private:
    PrecisionType precision_;

    int32_t inputNum;
    // Nice to have, ToDo(empire.jung, TBD): Consider changing to use dim4 struct.
    int32_t number;
    int32_t channel;
    int32_t height;
    int32_t width;
    int32_t axis;

    template <typename T>
    Status executeKernel(const std::vector<std::shared_ptr<ITensor>> input_tensor,
                         std::shared_ptr<NEONTensor<T>> output_tensor);
};

}  // namespace cpu
}  // namespace ud
}  // namespace enn
