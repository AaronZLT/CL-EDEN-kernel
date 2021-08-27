#pragma once

#include "model/schema/schema_nnc.h"
#include "userdriver/common/UserDriverTypes.h"
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/OperatorInterfaces.h"

namespace enn {
namespace ud {

class IComputeLibrary {
   public:
    virtual ~IComputeLibrary() = default;

    virtual void assignBuffers() {}  // for GPU
    virtual void flush() {}          // for GPU
    virtual void synchronize() {}    // for GPU

    virtual Status setPublicKernels(NnApiType nn_type, StorageType storage_type) {
        ENN_UNUSED(nn_type);
        ENN_UNUSED(storage_type);
        return Status::SUCCESS;
    }

    virtual std::shared_ptr<ITensor> create_tensor(const TFlite::TensorType &type,
                                                   const PrecisionType &precision,
                                                   const NDims &ndim,
                                                   const int32_t &buffer_index = UNDEFINED,
                                                   const BufferType &buffer_type = BufferType::DEDICATED,
                                                   const bool &use_cpu_for_gpu = false,
                                                   const StorageType &storage_type = StorageType::BUFFER,
                                                   const DataOrder &data_order = DataOrder::NCHW,
                                                   const float &scale = 1.0f,           // for quantized model
                                                   const int32_t &zero_point = 0) = 0;  // for quantized model

    virtual std::shared_ptr<ITensor> create_and_copy_tensor(const TFlite::TensorType &type,
                                                            DataPtr &data,
                                                            const PrecisionType &precision,
                                                            const NDims &ndim,
                                                            const int32_t &buffer_index = UNDEFINED,
                                                            const BufferType &buffer_type = BufferType::DEDICATED,
                                                            const bool &use_cpu_for_gpu = false,
                                                            const StorageType &storage_type = StorageType::BUFFER,
                                                            const DataOrder &data_order = DataOrder::NCHW,
                                                            const float &scale = 1.0f,
                                                            const int32_t &zero_point = 0) = 0;

    virtual std::shared_ptr<ITensor> clone_tensor(const std::shared_ptr<ITensor> tensor) = 0;
};  // class IComputeLibrary

}  // namespace ud
}  // namespace enn
