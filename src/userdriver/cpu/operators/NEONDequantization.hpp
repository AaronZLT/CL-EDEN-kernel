#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/operators/IDequantization.hpp"
#include "userdriver/cpu/common/NEONTensor.hpp"
#include "userdriver/cpu/common/NEONIncludes.hpp"

namespace enn {
namespace ud {
namespace cpu {

class NEONDequantization : public IDequantization {
public:
    explicit NEONDequantization(const PrecisionType& precision);

    Status initialize(const std::shared_ptr<ITensor> input, const int32_t& data_num,
                      const std::shared_ptr<ITensor> frac_lens, const uint32_t& img_size);

    Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output);

    Status release();

private:
    PrecisionType precision_;
    DataType input_data_type;

    int32_t data_num;
    uint32_t img_size;
    std::shared_ptr<NEONTensor<int32_t>> frac_lens_;

#ifdef NEON_OPT
    bool is_mono_frac_len = true;
    void check_mono_frac_len(int32_t channel, int32_t* frac_lens);
#endif

    template <typename T>
    Status executeKernel(const std::shared_ptr<NEONTensor<T>> input_tensor,
                         std::shared_ptr<NEONTensor<float>> output_tensor);
};

}  // namespace cpu
}  // namespace ud
}  // namespace enn
