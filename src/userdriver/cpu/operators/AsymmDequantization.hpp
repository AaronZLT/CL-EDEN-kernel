#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/operators/IAsymmDequantization.hpp"
#include "userdriver/cpu/common/NEONTensor.hpp"

namespace enn {
namespace ud {
namespace cpu {

class AsymmDequantization : public IAsymmDequantization {
public:
    explicit AsymmDequantization(const PrecisionType& precision);

    Status initialize(const std::shared_ptr<ITensor> input, const int32_t& data_num, const float& scale,
                      const int32_t& zero_point, const uint32_t& img_size);

    Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output);

    Status release();

private:
    PrecisionType precision_;
    DataType input_data_type;

    int32_t data_num;
    float scale;
    int32_t zero_point;
    uint32_t img_size;

    template <typename T1, typename T2>
    Status executeKernel(const std::shared_ptr<NEONTensor<T1>> input_tensor, std::shared_ptr<NEONTensor<T2>> output_tensor);
};

}  // namespace cpu
}  // namespace ud
}  // namespace enn
