#include "common/enn_debug.h"
#include "userdriver/gpu/common/CLComputeLibrary.hpp"
#include "userdriver/gpu/common/CLOperators.hpp"
#include "userdriver/cpu/common/NEONTensor.hpp"

namespace enn {
namespace ud {
namespace gpu {

CLComputeLibrary::CLComputeLibrary(const uint32_t &device_id) {
    runtime_ = std::make_shared<CLRuntime>();
    runtime_->initialize(device_id);
}

void CLComputeLibrary::assignBuffers() {
    runtime_->assignBufferPool();
}

void CLComputeLibrary::flush() {
    clFlush(runtime_->getQueue());
}

void CLComputeLibrary::synchronize() {
    clFinish(runtime_->getQueue());
}

Status CLComputeLibrary::setPublicKernels(NnApiType nn_type, StorageType storage_type) {
    Status ret = Status::SUCCESS;
    ret = runtime_->setCommonKernels();
    CHECK_EXPR_RETURN_FAILURE(ret == Status::SUCCESS, "Failed to setCommonKernels.");
    if (nn_type == NnApiType::ANDROID_NN_API && storage_type == StorageType::TEXTURE) {
        ret = runtime_->setPublicTexture2dKernels();
        CHECK_EXPR_RETURN_FAILURE(ret == Status::SUCCESS, "Failed to setPublicTexture2dKernels.");
    }
    return ret;
}

Status CLComputeLibrary::initialize_queue() {
    Status ret = runtime_->initializeQueue();
    CHECK_EXPR_RETURN_FAILURE(ret == Status::SUCCESS, "Failed to initializeQueue.");
    return Status::SUCCESS;
}

std::shared_ptr<ITensor> CLComputeLibrary::create_tensor(const TFlite::TensorType &type,
                                                         const PrecisionType &precision,
                                                         const NDims &ndim,
                                                         const int32_t &buffer_index,
                                                         const BufferType &buffer_type,
                                                         const bool &use_cpu_for_gpu,
                                                         const StorageType &storage_type,
                                                         const DataOrder &data_order,
                                                         const float &scale,
                                                         const int32_t &zero_point) {
    if (use_cpu_for_gpu) {
        return std::make_shared<cpu::NEONTensor<float>>(ndim, precision, static_cast<uint32_t>(type), buffer_index);
    } else {
        DataType data_type = tensor_type_2_data_type(type);
        return std::make_shared<CLTensor>(runtime_,
                                          precision,
                                          data_type,
                                          ndim,
                                          data_order,
                                          scale,
                                          zero_point,
                                          buffer_type,
                                          storage_type,
                                          static_cast<uint32_t>(type),
                                          buffer_index);
    }
    return nullptr;
}

std::shared_ptr<ITensor> CLComputeLibrary::create_and_copy_tensor(const TFlite::TensorType &type,
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
    DEBUG_PRINT("CLComputeLibrary::create_and_copy_tensor() is called");
    switch (type) {
    case TFlite::TensorType::TensorType_FLOAT32: {
        float *cast_data = reinterpret_cast<float *>(data);
        return create_and_copy_tensor_impl(type,
                                           cast_data,
                                           precision,
                                           ndim,
                                           buffer_index,
                                           buffer_type,
                                           storage_type,
                                           data_order,
                                           scale,
                                           zero_point);
    }
    case TFlite::TensorType::TensorType_FLOAT16: {
        half_float::half *cast_data = reinterpret_cast<half_float::half *>(data);
        return create_and_copy_tensor_impl(type,
                                           cast_data,
                                           precision,
                                           ndim,
                                           buffer_index,
                                           buffer_type,
                                           storage_type,
                                           data_order,
                                           scale,
                                           zero_point);
    }
    case TFlite::TensorType::TensorType_INT8: {
        int8_t *cast_data = reinterpret_cast<int8_t *>(data);
        return create_and_copy_tensor_impl(type,
                                           cast_data,
                                           precision,
                                           ndim,
                                           buffer_index,
                                           buffer_type,
                                           storage_type,
                                           data_order,
                                           scale,
                                           zero_point);
    }
    case TFlite::TensorType::TensorType_INT32: {
        int32_t *cast_data = reinterpret_cast<int32_t *>(data);
        return create_and_copy_tensor_impl(type,
                                           cast_data,
                                           precision,
                                           ndim,
                                           buffer_index,
                                           buffer_type,
                                           storage_type,
                                           data_order,
                                           scale,
                                           zero_point);
    }
    case TFlite::TensorType::TensorType_UINT8: {
        uint8_t *cast_data = reinterpret_cast<uint8_t *>(data);
        return create_and_copy_tensor_impl(type,
                                           cast_data,
                                           precision,
                                           ndim,
                                           buffer_index,
                                           buffer_type,
                                           storage_type,
                                           data_order,
                                           scale,
                                           zero_point);
    }
    case TFlite::TensorType::TensorType_BOOL: {
        bool *cast_data = reinterpret_cast<bool *>(data);
        return create_and_copy_tensor_impl(type,
                                           cast_data,
                                           precision,
                                           ndim,
                                           buffer_index,
                                           buffer_type,
                                           storage_type,
                                           data_order,
                                           scale,
                                           zero_point);
    }
    case TFlite::TensorType::TensorType_INT16: {
        int16_t *cast_data = reinterpret_cast<int16_t *>(data);
        return create_and_copy_tensor_impl(type,
                                           cast_data,
                                           precision,
                                           ndim,
                                           buffer_index,
                                           buffer_type,
                                           storage_type,
                                           data_order,
                                           scale,
                                           zero_point);
    }
    default: ENN_ERR_PRINT("Unsupported data type\n"); return nullptr;
    }
    return nullptr;
}

// ToDo(SRCX): Change the logic as usage CLBuffer
std::shared_ptr<ITensor> CLComputeLibrary::clone_tensor(const std::shared_ptr<ITensor> base) {
    return create_tensor((TFlite::TensorType)base->get_buffer_type(), base->getPrecisionType(), base->getDims(),
                         base->get_buffer_index());
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
