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

#ifndef USERDRIVER_GPU_CL_OPERATORS_CL_COMPUTE_LIBRARY_HPP_
#define USERDRIVER_GPU_CL_OPERATORS_CL_COMPUTE_LIBRARY_HPP_

#include "userdriver/common/operator_interfaces/interfaces/IComputeLibrary.h"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/common/CLOperators.hpp"

namespace enn {
namespace ud {
namespace gpu {
#define GPU_OP_CREATOR(CLASS_NAME, OP_NAME)                                                                                 \
    std::shared_ptr<CLASS_NAME> create##OP_NAME(const PrecisionType &precision) {                                           \
        ENN_DBG_PRINT("CLComputeLibrary::create" #OP_NAME " is called\n");                                                  \
        return std::make_shared<CLASS_NAME>(runtime_, precision);                                                           \
    }

class CLComputeLibrary : public IComputeLibrary {
public:
    void assignBuffers();
    void flush();
    void synchronize();
    Status setPublicKernels(NnApiType nn_type, StorageType storage_type) override;
    explicit CLComputeLibrary(const uint32_t &device_id);
    ~CLComputeLibrary() = default;

    Status initialize_queue();

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

    DataType tensor_type_2_data_type(const TFlite::TensorType &type) {
        const auto iter = tensorType_2_data_type_map_.find(type);
        if (iter == tensorType_2_data_type_map_.end()) {
            return DataType::SIZE;
        }
        return iter->second;
    }

    GPU_OP_CREATOR(CLAdd, Add);
    GPU_OP_CREATOR(CLAveragepool, Averagepool);
    GPU_OP_CREATOR(CLBidirectionalSequenceLstm, BidirectionalSequenceLstm);
    GPU_OP_CREATOR(CLCast, Cast);
    GPU_OP_CREATOR(CLConcat, Concat);
    GPU_OP_CREATOR(CLConvolution, Convolution);
    GPU_OP_CREATOR(CLDeconvolution, Deconvolution);
    GPU_OP_CREATOR(CLDepth2Space, Depth2Space);
    GPU_OP_CREATOR(CLDepthwiseConvolution, DepthwiseConvolution);
    GPU_OP_CREATOR(CLDiv, Div);
    GPU_OP_CREATOR(CLFullyConnected, FullyConnected);
    GPU_OP_CREATOR(CLGather, Gather);
    GPU_OP_CREATOR(CLMaxpool, Maxpool);
    GPU_OP_CREATOR(CLMean, Mean);
    GPU_OP_CREATOR(CLMul, Mul);
    GPU_OP_CREATOR(CLNormalization, Normalization);
    GPU_OP_CREATOR(CLPad, Pad);
    GPU_OP_CREATOR(CLReduce, Reduce);
    GPU_OP_CREATOR(CLRelu, Relu);
    GPU_OP_CREATOR(CLRelu1, Relu1);
    GPU_OP_CREATOR(CLRelu6, Relu6);
    GPU_OP_CREATOR(CLReshape, Reshape);
    GPU_OP_CREATOR(CLResizeBilinear, ResizeBilinear);
    GPU_OP_CREATOR(CLScale, Scale);
    GPU_OP_CREATOR(CLSlice, Slice);
    GPU_OP_CREATOR(CLSigmoid, Sigmoid);
    GPU_OP_CREATOR(CLSoftmax, Softmax);
    GPU_OP_CREATOR(CLSplit, Split);
    GPU_OP_CREATOR(CLSqueeze, Squeeze);
    GPU_OP_CREATOR(CLStridedSlice, StridedSlice);
    GPU_OP_CREATOR(CLSub, Sub);
    GPU_OP_CREATOR(CLTanh, Tanh);
    GPU_OP_CREATOR(CLTFSlice, TFSlice);
    GPU_OP_CREATOR(CLTranspose, Transpose);
    GPU_OP_CREATOR(CLUnpack, Unpack);

private:
    template <typename T>
    std::shared_ptr<ITensor> create_and_copy_tensor_impl(const TFlite::TensorType &type,
                                                         T *cast_data,
                                                         const PrecisionType &precision,
                                                         const NDims &ndim,
                                                         const int32_t buffer_index = UNDEFINED,
                                                         const BufferType &buffer_type = BufferType::DEDICATED,
                                                         const StorageType &storage_type = StorageType::BUFFER,
                                                         const DataOrder &data_order = DataOrder::NCHW,
                                                         const float &scale = 1.0,
                                                         const int32_t &offset = 0) {
        return std::make_shared<CLTensor>(runtime_,
                                          precision,
                                          cast_data,
                                          ndim,
                                          data_order,
                                          scale,
                                          offset,
                                          buffer_type,
                                          storage_type,
                                          static_cast<uint32_t>(type),
                                          buffer_index);
    }

private:
    std::shared_ptr<CLRuntime> runtime_;
    std::map<TFlite::TensorType, DataType> tensorType_2_data_type_map_ = {
        {TFlite::TensorType::TensorType_FLOAT64, DataType::FLOAT},
        {TFlite::TensorType::TensorType_FLOAT32, DataType::FLOAT},
        {TFlite::TensorType::TensorType_FLOAT16, DataType::HALF},
        {TFlite::TensorType::TensorType_INT32, DataType::INT32},
        {TFlite::TensorType::TensorType_UINT8, DataType::UINT8},
        {TFlite::TensorType::TensorType_INT8, DataType::INT8},
        {TFlite::TensorType::TensorType_BOOL, DataType::BOOL},
        {TFlite::TensorType::TensorType_INT16, DataType::INT16},
    };
};  // class CLComputeLibrary

}  // namespace gpu
}  // namespace ud
}  // namespace enn

#endif  // USERDRIVER_GPU_CL_OPERATORS_CL_COMPUTE_LIBRARY_HPP_
