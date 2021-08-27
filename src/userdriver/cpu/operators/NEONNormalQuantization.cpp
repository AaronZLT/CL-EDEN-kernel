#include "NEONNormalQuantization.hpp"

namespace enn {
namespace ud {
namespace cpu {

NEONNormalQuantization::NEONNormalQuantization(const PrecisionType& precision) {
    precision_ = precision;
    input_data_type = DataType::UNKNOWN;
    output_data_type = DataType::UNKNOWN;
    width = 0;
    height = 0;
    channel = 0;
}

Status NEONNormalQuantization::initialize(const std::shared_ptr<ITensor> input, const int32_t& width, const int32_t& height,
                                          const int32_t& channel, const std::shared_ptr<ITensor> means,
                                          const std::shared_ptr<ITensor> scales, const std::shared_ptr<ITensor> frac_lens) {
    ENN_UNUSED(input);
    this->width = width;
    this->height = height;
    this->channel = channel;
    this->means_ = std::static_pointer_cast<NEONTensor<double>>(means);
    this->scales_ = std::static_pointer_cast<NEONTensor<double>>(scales);
    this->frac_lens_ = std::static_pointer_cast<NEONTensor<int32_t>>(frac_lens);
    return Status::SUCCESS;
}

template <typename T>
T _clip(int32_t temp, T CONST_HEX_1, T CONST_HEX_2, T CONST_HEX_3) {
    const int32_t MAX_LIMIT = CONST_HEX_3;
    T result;
    if (temp < 0) {
        result = (temp & CONST_HEX_1) | CONST_HEX_2;
    } else {
        if (temp > MAX_LIMIT) {
            temp = MAX_LIMIT;
        }
        result = (temp & CONST_HEX_3);
    }
    return result;
}

template <typename T1, typename T2, typename T3>
Status NEONNormalQuantization::executeKernel(const std::shared_ptr<NEONTensor<T1>> input_tensor,
                                             std::shared_ptr<NEONTensor<T2>> output_tensor, T3 array_type) {
    UNUSED(array_type);

    T1* input_data = input_tensor->getBufferPtr();
    T2* output_data = output_tensor->getBufferPtr();
    double* means = means_->getDataPtr().get();
    double* scales = scales_->getDataPtr().get();
    int32_t* frac_lens = frac_lens_->getDataPtr().get();

    if ((!input_data) || (!output_data) || (width <= 0) || (height <= 0) || (channel <= 0) || (!means) || (!scales) ||
        (!frac_lens)) {
        ERROR_PRINT("Invalid parameter\n");
        return Status::INVALID_PARAMS;
    }

    T2 CONST_HEX_1 = 0xFF;
    T2 CONST_HEX_2 = 0x80;
    T2 CONST_HEX_3 = 0x7F;

    if (sizeof(T2) == sizeof(int16_t)) {
        CONST_HEX_1 = 0xFFFF;
        CONST_HEX_2 = 0x8000;
        CONST_HEX_3 = 0x7FFF;
    }

    int one_plane_size = width * height;

#ifdef NEON_OPT
    int in_index_ = one_plane_size / 8;
    int in_remain_start_ = in_index_ * 8;

    for (int c = 0; c < channel; c++) {
        float32x4_t vmeans_ = vdupq_n_f32(means[c]);

        float scale_mul_frac_;
        if (frac_lens[c] >= 0) {
            scale_mul_frac_ = (1 << frac_lens[c]) * scales[c];
        } else {
            scale_mul_frac_ = static_cast<float>(pow(2, frac_lens[c])) * scales[c];
        }

        float32x4_t scale_mul_fracs_ = vdupq_n_f32(scale_mul_frac_);

        for (int idx = 0; idx < in_index_; idx++) {
            float32x4x2_t float32_tmp = intermediate_get_float32x4x2_t(input_data + c * one_plane_size + idx * 8);

            float32_tmp.val[0] = (float32_tmp.val[0] - vmeans_) * scale_mul_fracs_;
            float32_tmp.val[1] = (float32_tmp.val[1] - vmeans_) * scale_mul_fracs_;

            T3 array_tmp_;
            int32_t val_in_;

            for (int j = 0; j < 8; j++) {
                val_in_ = round(float32_tmp.val[j / 4][j % 4]);
                array_tmp_[j] = _clip(val_in_, CONST_HEX_1, CONST_HEX_2, CONST_HEX_3);
            }
            intermediate_vst1(output_data + c * one_plane_size + idx * 8, array_tmp_);
        }

        for (int idx = in_remain_start_; idx < one_plane_size; idx++) {
            T2 result;
            T1 value = input_data[c * one_plane_size + idx];
            int32_t temp = round((static_cast<float>(value) - means[c]) * scale_mul_frac_);
            result = _clip(temp, CONST_HEX_1, CONST_HEX_2, CONST_HEX_3);

            output_data[c * one_plane_size + idx] = result;
        }
    }
#else
    for (int c = 0; c < channel; c++) {
        float scale_mul_frac_;
        if (frac_lens[c] >= 0) {
            scale_mul_frac_ = (1 << frac_lens[c]) * scales[c];
        } else {
            scale_mul_frac_ = static_cast<float>(pow(2, frac_lens[c])) * scales[c];
        }
        for (int idx = 0; idx < one_plane_size; idx++) {
            T2 result;
            T1 value = input_data[c * one_plane_size + idx];
            int32_t temp = round((static_cast<float>(value) - means[c]) * scale_mul_frac_);
            result = _clip(temp, CONST_HEX_1, CONST_HEX_2, CONST_HEX_3);
            output_data[c * one_plane_size + idx] = result;
        }
    }
#endif

    return Status::SUCCESS;
}

#ifdef NEON_OPT
float32x4x2_t NEONNormalQuantization::intermediate_get_float32x4x2_t(float* input) {
    float32x4x2_t output;

    output.val[0] = vld1q_f32(input);
    output.val[1] = vld1q_f32(input + 4);

    return output;
}

float32x4x2_t NEONNormalQuantization::intermediate_get_float32x4x2_t(uint8_t* input) {
    float32x4x2_t output;

    uint8x8_t uint8_tmp_ = vld1_u8(input);
    uint16x8_t uint16_tmp_ = vmovl_u8(uint8_tmp_);
    uint16x4_t uint16_tmp_0_ = vget_low_u16(uint16_tmp_);
    uint16x4_t uint16_tmp_1_ = vget_high_u16(uint16_tmp_);
    uint32x4_t uint32_tmp_0_ = vmovl_u16(uint16_tmp_0_);
    uint32x4_t uint32_tmp_1_ = vmovl_u16(uint16_tmp_1_);
    output.val[0] = vcvtq_f32_u32(uint32_tmp_0_);
    output.val[1] = vcvtq_f32_u32(uint32_tmp_1_);

    return output;
}

void NEONNormalQuantization::intermediate_vst1(int16_t* output, int16x8_t array) {
    vst1q_s16(output, array);
}

void NEONNormalQuantization::intermediate_vst1(int8_t* output, int8x8_t array) {
    vst1_s8(output, array);
}
#endif

Status NEONNormalQuantization::execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) {
    input_data_type = input->getDataType();
    output_data_type = output->getDataType();
    if (input_data_type == DataType::FLOAT && output_data_type == DataType::INT16) {
        auto input_tensor = std::static_pointer_cast<NEONTensor<float>>(input);
        auto output_tensor = std::static_pointer_cast<NEONTensor<int16_t>>(output);
#ifdef NEON_OPT
        int16x8_t array_type = vmovq_n_s16(0);
#else
        int16_t array_type = 0;
#endif
        return executeKernel(input_tensor, output_tensor, array_type);
    } else if (input_data_type == DataType::FLOAT &&
               (output_data_type == DataType::INT8 || output_data_type == DataType::UINT8)) {
        auto input_tensor = std::static_pointer_cast<NEONTensor<float>>(input);
        auto output_tensor = std::static_pointer_cast<NEONTensor<int8_t>>(output);
#ifdef NEON_OPT
        int8x8_t array_type = vmov_n_s8(0);
#else
        int8_t array_type = 0;
#endif
        return executeKernel(input_tensor, output_tensor, array_type);
    } else if (input_data_type == DataType::UINT8 && output_data_type == DataType::INT16) {
        auto input_tensor = std::static_pointer_cast<NEONTensor<uint8_t>>(input);
        auto output_tensor = std::static_pointer_cast<NEONTensor<int16_t>>(output);
#ifdef NEON_OPT
        int16x8_t array_type = vmovq_n_s16(0);
#else
        int16_t array_type = 0;
#endif
        return executeKernel(input_tensor, output_tensor, array_type);
    } else if (input_data_type == DataType::UINT8 &&
               (output_data_type == DataType::INT8 || output_data_type == DataType::UINT8)) {
        auto input_tensor = std::static_pointer_cast<NEONTensor<uint8_t>>(input);
        auto output_tensor = std::static_pointer_cast<NEONTensor<int8_t>>(output);
#ifdef NEON_OPT
        int8x8_t array_type = vmov_n_s8(0);
#else
        int8_t array_type = 0;
#endif
        return executeKernel(input_tensor, output_tensor, array_type);
    } else {
        ERROR_PRINT("Data type is not supported\n");
        return Status::FAILURE;
    }
}

Status NEONNormalQuantization::release() {
    return Status::SUCCESS;
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn