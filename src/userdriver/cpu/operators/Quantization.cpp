#include "Quantization.hpp"

namespace enn {
namespace ud {
namespace cpu {

Quantization::Quantization(const PrecisionType& precision) {
    precision_ = precision;
    output_data_type = DataType::UNKNOWN;
    channel = 0;
    height = 0;
    width = 0;
}

Status Quantization::initialize(const std::shared_ptr<ITensor> input, const int32_t& channel, const int32_t& height,
                                const int32_t& width, const std::shared_ptr<ITensor> frac_lens) {
    ENN_UNUSED(input);
    this->channel = channel;
    this->height = height;
    this->width = width;
    this->frac_lens_ = std::static_pointer_cast<NEONTensor<int32_t>>(frac_lens);
    return Status::SUCCESS;
}

template <typename T1, typename T2>
Status Quantization::executeKernel(const std::shared_ptr<NEONTensor<T1>> input_tensor,
                                   std::shared_ptr<NEONTensor<T2>> output_tensor) {
    T1* input_data = input_tensor->getBufferPtr();
    T2* output_data = output_tensor->getBufferPtr();
    int32_t* frac_lens = frac_lens_->getDataPtr().get();

    if ((!input_data) || (!output_data) || (!frac_lens) || (channel <= 0) || (height <= 0) || (width <= 0)) {
        ERROR_PRINT("Invalid parameter\n");
        return Status::INVALID_PARAMS;
    }

    DEBUG_PRINT("channel=[%d], height=[%d], width=[%d]\n", channel, height, width);

#ifndef ENN_BUILD_RELEASE
    std::string str_frac_len = "";
    for (int i = 0; i < channel; i++) {
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

    T2 CONST_HEX_1 = 0xFF;
    T2 CONST_HEX_2 = 0x80;
    T2 CONST_HEX_3 = 0x7F;

    if (sizeof(T2) == sizeof(int16_t)) {
        CONST_HEX_1 = 0xFFFF;
        CONST_HEX_2 = 0x8000;
        CONST_HEX_3 = 0x7FFF;
    }

    for (int32_t c = 0; c < channel; c++) {
        for (int32_t h = 0; h < height; h++) {
            for (int32_t w = 0; w < width; w++) {
                int32_t index = (c * height * width) + (h * width) + w;
                float num = input_data[index];
                int32_t temp = round(num * static_cast<float>(pow(2, frac_lens[c])));
                T2 result = 0;
                // Set or clear sign bit
                if (num < 0) {
                    result = (temp & CONST_HEX_1) | CONST_HEX_2;
                } else {
                    result = (temp & CONST_HEX_1) & CONST_HEX_3;
                }
                output_data[index] = result;
            }
        }
    }

    return Status::SUCCESS;
}

Status Quantization::execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) {
    output_data_type = output->getDataType();

    switch (output_data_type) {
        case DataType::INT16: {
            DEBUG_PRINT("DataType::INT16\n");
            auto input_tensor = std::static_pointer_cast<NEONTensor<float>>(input);
            auto output_tensor = std::static_pointer_cast<NEONTensor<int16_t>>(output);
            return executeKernel(input_tensor, output_tensor);
        }
        case DataType::INT8:
        case DataType::UINT8: {
            DEBUG_PRINT("DataType::INT8 or DataType::UINT8\n");
            auto input_tensor = std::static_pointer_cast<NEONTensor<float>>(input);
            auto output_tensor = std::static_pointer_cast<NEONTensor<int8_t>>(output);
            return executeKernel(input_tensor, output_tensor);
        }
        default: {
            ERROR_PRINT("Data type is not supported\n");
            return Status::FAILURE;
        }
    }
}

Status Quantization::release() {
    return Status::SUCCESS;
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
