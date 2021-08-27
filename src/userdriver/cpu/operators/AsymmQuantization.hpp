#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/operators/IAsymmQuantization.hpp"
#include "userdriver/cpu/common/NEONTensor.hpp"

namespace enn {
namespace ud {
namespace cpu {

class AsymmQuantization : public IAsymmQuantization {
public:
    explicit AsymmQuantization(const PrecisionType& precision);

    Status initialize(const std::shared_ptr<ITensor> input, const int32_t& channel, const int32_t& width,
                      const int32_t& height, const float& scale, const int32_t& zero_point);

    Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output);

    Status release();

private:
    PrecisionType precision_;
    DataType output_data_type;

    int32_t channel;
    int32_t width;
    int32_t height;
    float scale;
    int32_t zero_point;

    template <typename T1, typename T2>
    Status executeKernel(const std::shared_ptr<NEONTensor<T1>> input_tensor, std::shared_ptr<NEONTensor<T2>> output_tensor);
};

}  // namespace cpu
}  // namespace ud
}  // namespace enn
