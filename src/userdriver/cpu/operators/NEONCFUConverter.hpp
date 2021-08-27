#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/operators/ICFUConverter.hpp"
#include "userdriver/common/op_test/test_capabilities.h"
#include "userdriver/cpu/common/NEONTensor.hpp"
#include "userdriver/cpu/common/NEONIncludes.hpp"

namespace enn {
namespace ud {
namespace cpu {

class NEONCFUConverter : public ICFUConverter {
public:
    explicit NEONCFUConverter(const PrecisionType& precision, const std::string soc_name);

    Status initialize(const std::shared_ptr<ITensor> input, const int32_t& width, const int32_t& height,
                      const int32_t& channel, const int32_t& cols_in_cell, const int32_t& lines_in_cell,
                      const int32_t& interleaved_slices, const int32_t& pad_value = 0);

    Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output);

    Status release();

private:
    PrecisionType precision_;
    DataType data_type;

    std::string soc_name_;

    int32_t width;
    int32_t height;
    int32_t channel;
    int32_t cols_in_cell;
    int32_t lines_in_cell;
    int32_t interleaved_slices;
    uint32_t pad_value;

    template <typename T>
    Status executeKernel(const std::shared_ptr<NEONTensor<T>> input_tensor, std::shared_ptr<NEONTensor<T>> output_tensor);

    /**
     * @brief support exynos 2100/991/9840/9815/9925
     */
    template <typename T>
    Status executeKernel_impl_common(T* src_addr, T* dest_addr, int32_t width, int32_t height, int32_t channel,
                                     int32_t pad_value);

    /**
     * @brief support hr80/exynosauto9 [INT16]
     */
    Status executeKernel_impl_KITT(int16_t* src_addr, int16_t* dest_addr, int32_t width, int32_t height, int32_t channel,
                                   int32_t pad_value);

    /**
     * @brief support hr80/exynosauto9 [INT8]
     */
    Status executeKernel_impl_KITT(int8_t* src_addr, int8_t* dest_addr, int32_t width, int32_t height, int32_t channel,
                                   int32_t pad_value);

    /**
     * @brief Not support hr80/exynosauto9 [FLOAT16]
     */
    Status executeKernel_impl_KITT(_Float16_t* src_addr, _Float16_t* dest_addr, int32_t width, int32_t height,
                                   int32_t channel, int32_t pad_value) {
        ERROR_PRINT("Doesn't support Float16 type.\n");
        ENN_UNUSED(src_addr);
        ENN_UNUSED(dest_addr);
        ENN_UNUSED(width);
        ENN_UNUSED(height);
        ENN_UNUSED(channel);
        ENN_UNUSED(pad_value);
        return Status::INVALID_PARAMS;
    }
};

}  // namespace cpu
}  // namespace ud
}  // namespace enn
