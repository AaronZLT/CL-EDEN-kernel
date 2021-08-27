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
 * @file    cpu_op_constructor.h
 * @brief   This is common ENN CPU Userdriver API
 * @details This header defines ENN CPU Userdriver API.
 */
#ifndef USERDRIVER_CPU_CPU_OP_CONSTRUCTOR_H_
#define USERDRIVER_CPU_CPU_OP_CONSTRUCTOR_H_

#include <map>
#include <vector>

#include "userdriver/common/IOperationConstructor.h"
#include "userdriver/common/operator_interfaces/userdriver_operator.h"
#include "userdriver/common/operator_interfaces/custom_operator.h"
#include "userdriver/cpu/common/NEONComputeLibrary.h"
#include "model/schema/schema_nnc.h"

namespace enn {
namespace ud {
namespace cpu {

class OperationConstructor : public IOperationConstructor {
public:
    explicit OperationConstructor(std::shared_ptr<NEONComputeLibrary> compute_library_);
    ~OperationConstructor();

    EnnReturn create_ud_operator(const model::component::Operator::Ptr &operator_) override;

    template <TFlite::BuiltinOperator builtin_op>
    EnnReturn create_ud_operator(const model::component::Operator::Ptr &operator_);

    template <CustomOperator custom_op>
    EnnReturn create_ud_operator(const model::component::Operator::Ptr &operator_);

private:
    using OperationCreateFunction =
        EnnReturn (OperationConstructor::*)(const model::component::Operator::Ptr &operator_);

    void convert_to_tensors(const model::component::Operator::Ptr &operator_,
                            PrecisionType &precision_type,
                            std::vector<std::shared_ptr<ITensor>> &in_tensors,
                            std::vector<std::shared_ptr<ITensor>> &out_tensors,
                            std::vector<std::shared_ptr<ITensor>> &data_tensors);

    std::shared_ptr<ITensor> create_and_copy_tensor(const TFlite::TensorType &type, DataPtr &data, size_t size,
                                                    const PrecisionType &precision = PrecisionType::FP32);

    std::shared_ptr<ITensor> create_and_copy_tensor(const TFlite::TensorType &type, DataPtr &data, const NDims &ndim,
                                                    const PrecisionType &precision = PrecisionType::FP32);

    std::map<int32_t, OperationCreateFunction> builtin_op_map;
    std::map<std::string, OperationCreateFunction> custom_op_map;
    std::shared_ptr<NEONComputeLibrary> compute_library;
};

}  // namespace cpu
}  // namespace ud
}  // namespace enn

#endif  // USERDRIVER_CPU_CPU_OP_CONSTRUCTOR_H_
