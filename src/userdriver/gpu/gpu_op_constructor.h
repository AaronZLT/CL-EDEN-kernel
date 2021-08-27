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

/**
 * @file    gpu_op_constructor.h
 * @brief   This is common ENN GPU Userdriver API
 * @details This header defines ENN GPU Userdriver API.
 */
#ifndef USERDRIVER_GPU_GPU_OP_CONSTRUCTOR_H_
#define USERDRIVER_GPU_GPU_OP_CONSTRUCTOR_H_

#include <map>
#include <vector>
#include <unordered_set>

#include "model/schema/schema_nnc.h"
#include "userdriver/common/IOperationConstructor.h"
#include "userdriver/common/operator_interfaces/userdriver_operator.h"
#include "userdriver/common/operator_interfaces/custom_operator.h"
#include "userdriver/common/UserDriver.h"
#include "userdriver/gpu/common/CLParameter.hpp"
#include "userdriver/gpu/common/CLComputeLibrary.hpp"

namespace enn {
namespace ud {
namespace gpu {

class OperationConstructor : public IOperationConstructor {
public:
    OperationConstructor(std::shared_ptr<CLComputeLibrary> compute_library);
    ~OperationConstructor();

    EnnReturn create_ud_operator(const model::component::Operator::Ptr &operator_) override;

    template <TFlite::BuiltinOperator builtin_op>
    EnnReturn create_ud_operator(const model::component::Operator::Ptr &operator_);

    // Todo(all): Remove later if unnecessary
    template <CustomOperator custom_op> EnnReturn create_ud_operator(const model::component::Operator::Ptr &operator_);

    EnnReturn open_oplist(const model::component::OperatorList &operator_list);
    void close_oplist(const uint64_t &oplist_id);

private:
    using OperationCreateFunction = EnnReturn (OperationConstructor::*)(const model::component::Operator::Ptr &operator_);

    void convert_to_tensors(const model::component::Operator::Ptr &operator_,
                            const PrecisionType &precision_type,
                            std::vector<std::shared_ptr<ITensor>> &in_tensors,
                            std::vector<std::shared_ptr<ITensor>> &out_tensors,
                            const bool &use_fp32_for_fp16 = false,
                            const bool &use_cpu_for_fp16 = false);

    std::shared_ptr<ITensor> allocate_tensor(
        const std::shared_ptr<enn::model::component::Tensor> &edge,
        const PrecisionType &precision,
        const BufferType &buffer_type = BufferType::DEDICATED,
        const DataOrder &data_order = DataOrder::NCHW,
        const bool &use_fp32_for_fp16 = false,  //  used in NORMALIZATION, input is FP32 and output is FP16
        const bool &use_cpu_for_fp16 = false);  // used in TFLITE_DETECTION, input is GPU and output is CPU

    bool check_dim(const NDims &dim);
    bool is_nchw_layout(const TFlite::LegacyModel &legacy);

    PrecisionType get_precision(const TFlite::TensorType &dataType);
    void get_padding(TFlite::Padding paddingType,
                     Pad4 &padding,
                     NDims inputDim,
                     NDims outputDim,
                     NDims weightDim,
                     Dim2 stride,
                     Dim2 dilation,
                     bool deconv,
                     bool nchw);
    void get_edge_info(const std::shared_ptr<enn::model::component::Tensor> &edge,
                       NDims &dims,
                       TFlite::TensorType &data_type,
                       float &scale,
                       int32_t &zero_point);
    void get_perchannel_quant_info(const std::shared_ptr<enn::model::component::Tensor> &edge,
                                   bool &per_channel_quant,
                                   std::vector<float> &scale);
    void init_inter_buffer(const std::shared_ptr<model::component::Operator> &operator_);

    std::map<int32_t, OperationCreateFunction> builtin_op_map_;
    std::map<std::string, OperationCreateFunction> custom_op_map_;
    std::shared_ptr<CLComputeLibrary> compute_library_;

    std::mutex mutex_constructor_;
    uint64_t operator_list_id_;
    // map for (operator_list_id, (buffer_index, tensor))
    std::unordered_map<uint64_t, std::map<int32_t, std::shared_ptr<ITensor>>> alloc_tensors_map_;

    std::map<uint32_t, uint32_t> tensors_used_map_;  // Map for (buffer_index, operator_count)
    std::unordered_set<uint32_t> id_input_op_;       // set of input buffer_index
    std::unordered_set<uint32_t> id_output_op_;      // set of output buffer_index

    bool relax_computation_float32_to_float16_;
    TFlite::LegacyModel legacy_model_;

    StorageType storage_type_;
};

}  // namespace gpu
}  // namespace ud
}  // namespace enn

#endif  // USERDRIVER_GPU_GPU_OP_CONSTRUCTOR_H_
