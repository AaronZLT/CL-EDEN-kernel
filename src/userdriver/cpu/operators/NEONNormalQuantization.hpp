#pragma once

#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/common/operator_interfaces/common/Error.hpp"
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/operators/INormalQuantization.hpp"
#include "userdriver/cpu/common/NEONTensor.hpp"
#include "userdriver/cpu/common/NEONIncludes.hpp"

namespace enn {
namespace ud {
namespace cpu {

class NEONNormalQuantization : public INormalQuantization {
public:
    explicit NEONNormalQuantization(const PrecisionType& precision);

    Status initialize(const std::shared_ptr<ITensor> input, const int32_t& width, const int32_t& height,
                      const int32_t& channel, const std::shared_ptr<ITensor> means,
                      const std::shared_ptr<ITensor> scales, const std::shared_ptr<ITensor> frac_lens);

    Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output);

    Status release();

private:
    PrecisionType precision_;
    DataType input_data_type;
    DataType output_data_type;

    int32_t width;
    int32_t height;
    int32_t channel;
    std::shared_ptr<NEONTensor<double>> means_;
    std::shared_ptr<NEONTensor<double>> scales_;
    std::shared_ptr<NEONTensor<int32_t>> frac_lens_;

    template <typename T1, typename T2, typename T3>
    Status executeKernel(const std::shared_ptr<NEONTensor<T1>> input_tensor,
                         std::shared_ptr<NEONTensor<T2>> output_tensor, T3 array_type);

#ifdef NEON_OPT
    float32x4x2_t intermediate_get_float32x4x2_t(float* input);
    float32x4x2_t intermediate_get_float32x4x2_t(uint8_t* input);

    void intermediate_vst1(int16_t* output, int16x8_t array);
    void intermediate_vst1(int8_t* output, int8x8_t array);
#endif
};

}  // namespace cpu
}  // namespace ud
}  // namespace enn