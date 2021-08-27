#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/operators/IQuantization.hpp"
#include "userdriver/cpu/common/NEONTensor.hpp"

namespace enn {
namespace ud {
namespace cpu {

class Quantization : public IQuantization {
public:
    explicit Quantization(const PrecisionType& precision);

    Status initialize(const std::shared_ptr<ITensor> input, const int32_t& channel, const int32_t& height,
                      const int32_t& width, const std::shared_ptr<ITensor> frac_lens);

    Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output);

    Status release();

private:
    PrecisionType precision_;
    DataType output_data_type;

    int32_t channel;
    int32_t height;
    int32_t width;
    std::shared_ptr<NEONTensor<int32_t>> frac_lens_;

    template <typename T1, typename T2>
    Status executeKernel(const std::shared_ptr<NEONTensor<T1>> input_tensor, std::shared_ptr<NEONTensor<T2>> output_tensor);
};

}  // namespace cpu
}  // namespace ud
}  // namespace enn
