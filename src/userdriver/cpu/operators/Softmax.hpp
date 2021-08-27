#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/operators/ISoftmax.hpp"
#include "userdriver/cpu/common/NEONTensor.hpp"

namespace enn {
namespace ud {
namespace cpu {

class Softmax : public ISoftmax {
public:
    explicit Softmax(const PrecisionType& precision);

    Status initialize(const std::shared_ptr<ITensor> input, const int32_t& width, const int32_t& height,
                      const int32_t& channel, const int32_t& number, const float& beta, const int32_t& axis);

    Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output);

    Status release();

private:
    PrecisionType precision_;

    // Nice to have, ToDo(empire.jung, TBD): Consider changing to use dim4 struct.
    int32_t width;
    int32_t height;
    int32_t channel;
    int32_t number;
    float beta;
    int32_t axis;

    template <typename T>
    Status executeKernel(const std::shared_ptr<NEONTensor<T>> input_tensor, std::shared_ptr<NEONTensor<T>> output_tensor);
};

}  // namespace cpu
}  // namespace ud
}  // namespace enn
