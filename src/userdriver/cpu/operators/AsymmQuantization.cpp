#include "AsymmQuantization.hpp"

namespace enn {
namespace ud {
namespace cpu {

AsymmQuantization::AsymmQuantization(const PrecisionType& precision) {
    precision_ = precision;
    output_data_type = DataType::UNKNOWN;
    channel = 0;
    width = 0;
    height = 0;
    scale = 0.f;
    zero_point = 0;
}

Status AsymmQuantization::initialize(const std::shared_ptr<ITensor> input, const int32_t& channel, const int32_t& width,
                                     const int32_t& height, const float& scale, const int32_t& zero_point) {
    ENN_UNUSED(input);
    this->channel = channel;
    this->width = width;
    this->height = height;
    this->scale = scale;
    this->zero_point = zero_point;
    return Status::SUCCESS;
}

template <typename T1, typename T2>
Status AsymmQuantization::executeKernel(const std::shared_ptr<NEONTensor<T1>> input_tensor,
                                        std::shared_ptr<NEONTensor<T2>> output_tensor) {
    T1* input_data = input_tensor->getBufferPtr();
    T2* output_data = output_tensor->getBufferPtr();

    if ((!input_data) || (!output_data) || (channel <= 0) || (height <= 0) || (width <= 0)) {
        ERROR_PRINT("Invalid parameter\n");
        return Status::INVALID_PARAMS;
    }

    DEBUG_PRINT("width=[%d], height=[%d], channel=[%d], scale=[%f], zero_point=[%d]\n", width, height, channel, scale,
                zero_point);

    int32_t data_num = width * height * channel;
    if (output_data_type == DataType::UINT8) {
        int val;
        for (int32_t idx = 0; idx < data_num; idx++) {
            val = round(input_data[idx] / scale + zero_point);
            output_data[idx] = (val & (~0xFF)) ? (-val) >> 31 : val;
        }
    } else {
        for (int32_t idx = 0; idx < data_num; idx++) {
            output_data[idx] = round(input_data[idx] / scale + zero_point);
        }
    }

    return Status::SUCCESS;
}

Status AsymmQuantization::execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) {
    output_data_type = output->getDataType();

    switch (output_data_type) {
        case DataType::INT16: {
            DEBUG_PRINT("DataType::INT16\n");
            auto input_tensor = std::static_pointer_cast<NEONTensor<float>>(input);
            auto output_tensor = std::static_pointer_cast<NEONTensor<int16_t>>(output);
            return executeKernel(input_tensor, output_tensor);
        }
        case DataType::INT8: {
            DEBUG_PRINT("DataType::INT8\n");
            auto input_tensor = std::static_pointer_cast<NEONTensor<float>>(input);
            auto output_tensor = std::static_pointer_cast<NEONTensor<int8_t>>(output);
            return executeKernel(input_tensor, output_tensor);
        }
        case DataType::UINT8: {
            DEBUG_PRINT("DataType::UINT8\n");
            auto input_tensor = std::static_pointer_cast<NEONTensor<float>>(input);
            auto output_tensor = std::static_pointer_cast<NEONTensor<uint8_t>>(output);
            return executeKernel(input_tensor, output_tensor);
        }
        default: {
            ERROR_PRINT("Data type is not supported\n");
            return Status::FAILURE;
        }
    }
}

Status AsymmQuantization::release() {
    return Status::SUCCESS;
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
