#include "userdriver/cpu/common/NEONComputeLibrary.h"

namespace enn {
namespace ud {
namespace cpu {

std::shared_ptr<ITensor> NEONComputeLibrary::create_tensor(const TFlite::TensorType &type,
                                                           const PrecisionType &precision,
                                                           const NDims &ndim,
                                                           const int32_t &buffer_index,
                                                           const BufferType &buffer_type,
                                                           const bool &use_cpu_for_gpu,
                                                           const StorageType &storage_type,
                                                           const DataOrder &data_order,
                                                           const float &scale,
                                                           const int32_t &zero_point) {
    ENN_UNUSED(buffer_type);
    ENN_UNUSED(use_cpu_for_gpu);
    ENN_UNUSED(storage_type);
    ENN_UNUSED(data_order);
    ENN_UNUSED(scale);
    ENN_UNUSED(zero_point);
    switch (type) {
        case TFlite::TensorType::TensorType_FLOAT32:
            return std::make_shared<NEONTensor<float>>(ndim, precision, static_cast<uint32_t>(type), buffer_index);
	case TFlite::TensorType::TensorType_FLOAT16:
            return std::make_shared<NEONTensor<_Float16_t>>(ndim, precision, static_cast<uint32_t>(type), buffer_index);
        case TFlite::TensorType::TensorType_INT32:
            return std::make_shared<NEONTensor<int32_t>>(ndim, precision, static_cast<uint32_t>(type), buffer_index);
        case TFlite::TensorType::TensorType_UINT8:
            return std::make_shared<NEONTensor<uint8_t>>(ndim, precision, static_cast<uint32_t>(type), buffer_index);
        case TFlite::TensorType::TensorType_BOOL:
            return std::make_shared<NEONTensor<bool>>(ndim, precision, static_cast<uint32_t>(type), buffer_index);
        case TFlite::TensorType::TensorType_INT16:
            return std::make_shared<NEONTensor<int16_t>>(ndim, precision, static_cast<uint32_t>(type), buffer_index);
        case TFlite::TensorType::TensorType_INT8:
            return std::make_shared<NEONTensor<int8_t>>(ndim, precision, static_cast<uint32_t>(type), buffer_index);
        default:
            ENN_ERR_PRINT("Unsupported data type\n");
            return nullptr;
    }
}

std::shared_ptr<ITensor> NEONComputeLibrary::create_and_copy_tensor(const TFlite::TensorType &type,
                                                                    DataPtr &data,
                                                                    const PrecisionType &precision,
                                                                    const NDims &ndim,
                                                                    const int32_t &buffer_index,
                                                                    const BufferType &buffer_type,
                                                                    const bool &use_cpu_for_gpu,
                                                                    const StorageType &storage_type,
                                                                    const DataOrder &data_order,
                                                                    const float &scale,
                                                                    const int32_t &zero_point) {
    ENN_UNUSED(buffer_type);
    ENN_UNUSED(use_cpu_for_gpu);
    ENN_UNUSED(storage_type);
    ENN_UNUSED(data_order);
    ENN_UNUSED(scale);
    ENN_UNUSED(zero_point);
    switch (type) {
        case TFlite::TensorType::TensorType_FLOAT32: {
            float *cast_data = reinterpret_cast<float *>(data);
            return std::make_shared<NEONTensor<float>>(cast_data, ndim, precision, static_cast<uint32_t>(type),
                                                       buffer_index);
        }
        case TFlite::TensorType::TensorType_FLOAT16: {
            _Float16_t *cast_data = reinterpret_cast<_Float16_t *>(data);
            return std::make_shared<NEONTensor<_Float16_t>>(cast_data, ndim, precision, static_cast<uint32_t>(type),
                                                       buffer_index);
        }
        case TFlite::TensorType::TensorType_INT32: {
            int32_t *cast_data = reinterpret_cast<int32_t *>(data);
            return std::make_shared<NEONTensor<int32_t>>(cast_data, ndim, precision, static_cast<uint32_t>(type),
                                                         buffer_index);
        }
        case TFlite::TensorType::TensorType_UINT8: {
            uint8_t *cast_data = reinterpret_cast<uint8_t *>(data);
            return std::make_shared<NEONTensor<uint8_t>>(cast_data, ndim, precision, static_cast<uint32_t>(type),
                                                         buffer_index);
        }
        case TFlite::TensorType::TensorType_BOOL: {
            bool *cast_data = reinterpret_cast<bool *>(data);
            return std::make_shared<NEONTensor<bool>>(cast_data, ndim, precision, static_cast<uint32_t>(type), buffer_index);
        }
        case TFlite::TensorType::TensorType_INT16: {
            int16_t *cast_data = reinterpret_cast<int16_t *>(data);
            return std::make_shared<NEONTensor<int16_t>>(cast_data, ndim, precision, static_cast<uint32_t>(type),
                                                         buffer_index);
        }
        case TFlite::TensorType::TensorType_INT8: {
            int8_t *cast_data = reinterpret_cast<int8_t *>(data);
            return std::make_shared<NEONTensor<int8_t>>(cast_data, ndim, precision);
        }
        default:
            ENN_ERR_PRINT("Unsupported data type\n");
            return nullptr;
    }
}

std::shared_ptr<ITensor> NEONComputeLibrary::clone_tensor(const std::shared_ptr<ITensor> base) {
    return create_tensor((TFlite::TensorType)base->get_buffer_type(), base->getPrecisionType(), base->getDims(),
                         base->get_buffer_index());
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
