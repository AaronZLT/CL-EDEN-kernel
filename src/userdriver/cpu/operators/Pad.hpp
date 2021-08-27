#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/operators/IPad.hpp"
#include "userdriver/cpu/common/NEONTensor.hpp"

namespace enn {
namespace ud {
namespace cpu {

class Pad : public IPad {
public:
    explicit Pad(const PrecisionType& precision);

    Status initialize(const std::shared_ptr<ITensor> input, const int32_t& width, const int32_t& height,
                      const int32_t& channel, const std::vector<int32_t>& padFront, const std::vector<int32_t>& padEnd,
                      const std::vector<float>& padValue);

    Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output);

    Status release();

private:
    PrecisionType precision_;

    // Nice to have, ToDo(empire.jung, TBD): Consider changing to use dim3 struct.
    int32_t width;
    int32_t height;
    int32_t channel;
    std::vector<int32_t> padFront;
    std::vector<int32_t> padEnd;
    std::vector<float> padValue;

    template <typename T>
    Status executeKernel(const std::shared_ptr<NEONTensor<T>> input_tensor, std::shared_ptr<NEONTensor<T>> output_tensor);
};

}  // namespace cpu
}  // namespace ud
}  // namespace enn
