#include "NEONDequantization.hpp"

namespace enn {
namespace ud {
namespace cpu {

NEONDequantization::NEONDequantization(const PrecisionType& precision) {
    precision_ = precision;
    input_data_type = DataType::UNKNOWN;
    data_num = 0;
    img_size = 0;
}

Status NEONDequantization::initialize(const std::shared_ptr<ITensor> input, const int32_t& data_num,
                                      const std::shared_ptr<ITensor> frac_lens, const uint32_t& img_size) {
    ENN_UNUSED(input);
    this->data_num = data_num;
    this->img_size = img_size;
    this->frac_lens_ = std::static_pointer_cast<NEONTensor<int32_t>>(frac_lens);

#ifdef NEON_OPT
    check_mono_frac_len((data_num / img_size), frac_lens_->getDataPtr().get());
#endif

    return Status::SUCCESS;
}

#ifdef NEON_OPT
void NEONDequantization::check_mono_frac_len(int32_t channel, int32_t* frac_lens) {
    for (int32_t i = 0; i < channel - 1; ++i) {
        if (frac_lens[i] != frac_lens[i + 1]) {
            is_mono_frac_len = false;
            break;
        }
    }
    DEBUG_PRINT("is_mono_frac_len : %s\n", is_mono_frac_len ? "true" : "false");
}

inline int16x8_t intermediate_get_int16x8_t(int16_t* input_data, int index) {
    int16x8_t input_16x8 = vld1q_s16(input_data + index);
    return input_16x8;
}

inline int16x8_t intermediate_get_int16x8_t(int8_t* input_data, int index) {
    int8x8_t input_8x8 = vld1_s8(input_data + index);
    int16x8_t input_16x8 = vmovl_s8(input_8x8);
    return input_16x8;
}
#endif

template <typename T>
Status NEONDequantization::executeKernel(const std::shared_ptr<NEONTensor<T>> input_tensor,
                                         std::shared_ptr<NEONTensor<float>> output_tensor) {
    T* input_data = input_tensor->getBufferPtr();
    float* output_data = output_tensor->getBufferPtr();
    int32_t* frac_lens = frac_lens_->getDataPtr().get();

    if ((!input_data) || (!output_data) || (!frac_lens) || (!data_num)) {
        ERROR_PRINT("Invalid parameter\n");
        return Status::INVALID_PARAMS;
    }

    int32_t channel = data_num / img_size;
    int32_t index;
    float temp;

    DEBUG_PRINT("data_num=[%d], channel=[%d]\n", data_num, channel);

#ifndef ENN_BUILD_RELEASE
    std::string str_frac_len = "";
    for (int i = 0; i < channel; ++i) {
        str_frac_len += std::to_string(frac_lens[i]);
        str_frac_len += ", ";
        if (i > 10) {
            str_frac_len += ",,,,, ";
            str_frac_len += std::to_string(frac_lens[channel - 1]);
            break;
        }
    }
    DEBUG_PRINT("frac_lens(%d) = {%s}\n", channel, str_frac_len.c_str());
#endif

    const bool OPTIMIZED = true;
    // Optimization: copy input to output, and write output backward
    //               it is to improve cache write hit.
    // TODO(hj2020.kim): Add Memory Optimizer to use memory reuse
    if (OPTIMIZED) {
        DEBUG_PRINT("OPTIMIZED: enabled\n");
        memcpy(reinterpret_cast<T*>(output_data), input_data, sizeof(T) * data_num);
    }

#ifdef NEON_OPT
    DEBUG_PRINT("NEON_OPT: enabled\n");
    int32_t lane = 8;  // No. of data processing at a time.
    float32x4_t _temp32x4, res1_32x4, res2_32x4;
    int32x4_t input1_32x4, input2_32x4;
    int16x8_t input_16x8;

    // 1-D tensor (H*W = 1, channel = N) cannot benefit from NEON intrinsics because it is arranged in C-order.
    // If FracLens are identical in all channels, we can swap dims (H*W and channel) to make NEON loops workable.
    if (img_size == 1 && is_mono_frac_len) {
        img_size = channel;
        channel = 1;
    }

    int last_index = img_size % lane;
    for (int32_t c = channel - 1; c >= 0; c--) {
        int32_t fL = frac_lens[c];
        if (-63 <= fL && fL <= 63) {
            temp = (fL & (1 << 31)) ? (1 << (-fL)) : 1.0f / (1 << fL);
        } else {
            temp = 1.0f / (static_cast<float>(pow(2, fL)));
        }
        _temp32x4 = vmovq_n_f32(temp);
        index = c * img_size;

        for (int32_t idx = img_size - lane; idx >= last_index; idx -= lane) {
            if (OPTIMIZED) {
                input_16x8 = intermediate_get_int16x8_t(reinterpret_cast<T*>(output_data), index + idx);
            } else {
                input_16x8 = intermediate_get_int16x8_t(input_data, index + idx);
            }
            // convert into 32-bit signed integer
            input1_32x4 = vmovl_s16(vget_low_s16(input_16x8));
            input2_32x4 = vmovl_s16(vget_high_s16(input_16x8));
            // convert to 32-bit float
            res1_32x4 = vcvtq_f32_s32(input1_32x4);
            res2_32x4 = vcvtq_f32_s32(input2_32x4);

            res1_32x4 = vmulq_f32(res1_32x4, _temp32x4);
            res2_32x4 = vmulq_f32(res2_32x4, _temp32x4);

            vst1q_f32(output_data + index + idx, res1_32x4);
            vst1q_f32(output_data + index + idx + 4, res2_32x4);
        }

        for (int32_t idx = last_index - 1; idx >= 0; idx--) {
            if (OPTIMIZED) {
                output_data[index + idx] = static_cast<float>((reinterpret_cast<T*>(output_data))[index + idx]) * temp;
            } else {
                output_data[index + idx] = static_cast<float>(input_data[index + idx]) * temp;
            }
        }
    }
#else /* pow() function is called for channel times*/
    DEBUG_PRINT("NEON_OPT: disabled\n");
    for (int32_t c = channel - 1; c >= 0; c--) {
        int32_t fL = frac_lens[c];
        if (-63 <= fL && fL <= 63) {
            temp = (fL & (1 << 31)) ? (1 << (-fL)) : 1.0f / (1 << fL);
        } else {
            temp = 1.0f / (static_cast<float>(pow(2, fL)));
        }
        index = c * img_size;
        for (int32_t idx = img_size - 1; idx >= 0; idx--) {
            if (OPTIMIZED) {
                output_data[index + idx] = static_cast<float>((reinterpret_cast<T*>(output_data))[index + idx]) * temp;
            } else {
                output_data[index + idx] = static_cast<float>(input_data[index + idx]) * temp;
            }
        }
    }
#endif

    return Status::SUCCESS;
}

Status NEONDequantization::execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) {
    input_data_type = input->getDataType();

    switch (input_data_type) {
        case DataType::INT16: {
            DEBUG_PRINT("DataType::INT16\n");
            auto input_tensor = std::static_pointer_cast<NEONTensor<int16_t>>(input);
            auto output_tensor = std::static_pointer_cast<NEONTensor<float>>(output);
            return executeKernel(input_tensor, output_tensor);
        }
        case DataType::INT8:
        case DataType::UINT8: {
            DEBUG_PRINT("DataType::INT8 or DataType::UINT8\n");
            auto input_tensor = std::static_pointer_cast<NEONTensor<int8_t>>(input);
            auto output_tensor = std::static_pointer_cast<NEONTensor<float>>(output);
            return executeKernel(input_tensor, output_tensor);
        }
        default: {
            ERROR_PRINT("Data type is not supported\n");
            return Status::FAILURE;
        }
    }
}

Status NEONDequantization::release() {
    return Status::SUCCESS;
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
