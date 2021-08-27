/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or
 * distributed, transmitted, transcribed, stored in a retrieval system or
 * translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed to third parties
 * without the express written permission of Samsung Electronics.
 */

#ifndef USERDRIVER_CPU_NEON_OPERATORS_NEON_COMPUTE_LIBRARY_HPP_
#define USERDRIVER_CPU_NEON_OPERATORS_NEON_COMPUTE_LIBRARY_HPP_

#include "userdriver/common/operator_interfaces/interfaces/IComputeLibrary.h"
#include "userdriver/common/operator_interfaces/OperatorInterfaces.h"
#include "userdriver/cpu/common/NEONOperators.hpp"

namespace enn {
namespace ud {
namespace cpu {

#define CPU_OP_CREATOR(_IF_, _CLASS_, _FUNC_)                              \
    std::shared_ptr<_IF_> create##_FUNC_(const PrecisionType &precision) { \
        ENN_DBG_PRINT("called\n");                                         \
        return std::make_shared<_CLASS_>(precision);                       \
    }

#define CPU_OP_CREATOR_SOC(_IF_, _CLASS_, _FUNC_)                                                      \
    std::shared_ptr<_IF_> create##_FUNC_(const PrecisionType &precision, const std::string soc_name) { \
        ENN_DBG_PRINT("called\n");                                                                     \
        return std::make_shared<_CLASS_>(precision, soc_name);                                         \
    }

class NEONComputeLibrary : public IComputeLibrary {
public:
    NEONComputeLibrary() = default;
    ~NEONComputeLibrary() = default;

    std::shared_ptr<ITensor> create_tensor(const TFlite::TensorType &type,
                                           const PrecisionType &precision,
                                           const NDims &ndim,
                                           const int32_t &buffer_index = UNDEFINED,
                                           const BufferType &buffer_type = BufferType::DEDICATED,
                                           const bool &use_cpu_for_gpu = false,
                                           const StorageType &storage_type = StorageType::BUFFER,
                                           const DataOrder &data_order = DataOrder::NCHW,
                                           const float &scale = 1.0f,
                                           const int32_t &zero_point = 0);

    std::shared_ptr<ITensor> create_and_copy_tensor(const TFlite::TensorType &type,
                                                    DataPtr &data,
                                                    const PrecisionType &precision,
                                                    const NDims &ndim,
                                                    const int32_t &buffer_index = UNDEFINED,
                                                    const BufferType &buffer_type = BufferType::DEDICATED,
                                                    const bool &use_cpu_for_gpu = false,
                                                    const StorageType &storage_type = StorageType::BUFFER,
                                                    const DataOrder &data_order = DataOrder::NCHW,
                                                    const float &scale = 1.0f,
                                                    const int32_t &zero_point = 0);

    std::shared_ptr<ITensor> clone_tensor(const std::shared_ptr<ITensor> tensor);

    CPU_OP_CREATOR(IAsymmDequantization, AsymmDequantization, AsymmDequantization);
    CPU_OP_CREATOR(IAsymmQuantization, AsymmQuantization, AsymmQuantization);
    CPU_OP_CREATOR_SOC(ICFUConverter, NEONCFUConverter, CFUConverter);
    CPU_OP_CREATOR_SOC(ICFUInverter, CFUInverter, CFUInverter);
    CPU_OP_CREATOR(IConcat, Concat, Concat);
    CPU_OP_CREATOR(IDequantization, NEONDequantization, Dequantization);
    CPU_OP_CREATOR(IDetection, Detection, Detection);
    CPU_OP_CREATOR(IDetection, DetectionBatchSingle, DetectionBatchSingle);
    CPU_OP_CREATOR(IFlatten, Flatten, Flatten);
    CPU_OP_CREATOR(INormalization, Normalization, Normalization);
    CPU_OP_CREATOR(INormalQuantization, NEONNormalQuantization, NormalQuantization);
    CPU_OP_CREATOR(IPad, Pad, Pad);
    CPU_OP_CREATOR(IQuantization, Quantization, Quantization);
    CPU_OP_CREATOR(ISigDet, NEONSigDet, SigDetection);
    CPU_OP_CREATOR(ISoftmax, Softmax, Softmax);

private:
};  // class NEONComputeLibrary

}  // namespace cpu
}  // namespace ud
}  // namespace enn

#endif  // USERDRIVER_CPU_NEON_OPERATORS_NEON_COMPUTE_LIBRARY_HPP_
