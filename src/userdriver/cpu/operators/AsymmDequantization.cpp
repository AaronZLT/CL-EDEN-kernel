#include "AsymmDequantization.hpp"

namespace enn {
namespace ud {
namespace cpu {

AsymmDequantization::AsymmDequantization(const PrecisionType& precision) {
    precision_ = precision;
    input_data_type = DataType::UNKNOWN;
    data_num = 0;
    scale = 0.f;
    zero_point = 0;
    img_size = 0;
}

Status AsymmDequantization::initialize(const std::shared_ptr<ITensor> input, const int32_t& data_num, const float& scale,
                                       const int32_t& zero_point, const uint32_t& img_size) {
    ENN_UNUSED(input);
    this->data_num = data_num;
    this->scale = scale;
    this->zero_point = zero_point;
    this->img_size = img_size;
    return Status::SUCCESS;
}

template <typename T1, typename T2>
Status AsymmDequantization::executeKernel(const std::shared_ptr<NEONTensor<T1>> input_tensor,
                                          std::shared_ptr<NEONTensor<T2>> output_tensor) {
    T1* input_data = input_tensor->getBufferPtr();
    T2* output_data = output_tensor->getBufferPtr();

    if ((!input_data) || (!output_data) || (!data_num)) {
        ERROR_PRINT("Invalid parameter\n");
        return Status::INVALID_PARAMS;
    }

    DEBUG_PRINT("data_num=[%d], channel=[%d], scale=[%f], zero_point=[%d]\n",
                data_num, data_num / img_size, scale, zero_point);

    if (input_data_type == DataType::UINT8) {
        for (int32_t idx = data_num - 1; idx >= 0; idx--) {
            output_data[idx] = (static_cast<float>(static_cast<uint8_t>(input_data[idx])) - zero_point) * scale;
        }
    } else {
        for (int32_t idx = data_num - 1; idx >= 0; idx--) {
            output_data[idx] = (static_cast<float>(input_data[idx]) - zero_point) * scale;
        }
    }

    return Status::SUCCESS;
}

Status AsymmDequantization::execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) {
    input_data_type = input->getDataType();

    switch (input_data_type) {
        case DataType::INT16: {
            DEBUG_PRINT("DataType::INT16\n");
            auto input_tensor = std::static_pointer_cast<NEONTensor<int16_t>>(input);
            auto output_tensor = std::static_pointer_cast<NEONTensor<float>>(output);
            return executeKernel(input_tensor, output_tensor);
        }
        case DataType::INT8: {
            DEBUG_PRINT("DataType::INT8\n");
            auto input_tensor = std::static_pointer_cast<NEONTensor<int8_t>>(input);
            auto output_tensor = std::static_pointer_cast<NEONTensor<float>>(output);
            return executeKernel(input_tensor, output_tensor);
        }
        case DataType::UINT8: {
            DEBUG_PRINT("DataType::UINT8\n");
            auto input_tensor = std::static_pointer_cast<NEONTensor<uint8_t>>(input);
            auto output_tensor = std::static_pointer_cast<NEONTensor<float>>(output);
            return executeKernel(input_tensor, output_tensor);
        }
        default: {
            ERROR_PRINT("Data type is not supported\n");
            return Status::FAILURE;
        }
    }
}

Status AsymmDequantization::release() {
    return Status::SUCCESS;
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
