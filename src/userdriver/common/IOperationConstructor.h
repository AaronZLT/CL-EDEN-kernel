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

#ifndef USERDRIVER_COMMON_IOPERATION_CONSTRUCTOR_H_
#define USERDRIVER_COMMON_IOPERATION_CONSTRUCTOR_H_

#include "client/enn_api-type.h"
#include "model/attribute.hpp"
#include "model/component/operator/operator.hpp"
#include "userdriver/common/operator_interfaces/userdriver_operator.h"
#include "userdriver/common/UserDriver.h"

namespace enn {
namespace ud {

class IOperationConstructor {
public:
    virtual ~IOperationConstructor() = default;

    EnnReturn initialize_ud_operators() {
        operators = std::make_shared<std::vector<std::shared_ptr<UDOperator>>>();
        in_tensors.clear();
        out_tensors.clear();
        return ENN_RET_SUCCESS;
    }

    const UDOperators &get_ud_operators() { return std::move(operators); }
    const UDTensors &get_in_tensors() const { return in_tensors; }
    const UDTensors &get_out_tensors() const { return out_tensors; }

    virtual EnnReturn create_ud_operator(const model::component::Operator::Ptr &operator_) = 0;

    bool is_builtin_operator(const TFlite::BuiltinOperator &code) {
        return (TFlite::BuiltinOperator_MIN <= (int)code && (int)code <= TFlite::BuiltinOperator_MAX);
    }

    inline uint32_t get_data_count(const TFlite::TensorType &type, size_t size) {
        return (uint32_t)(size / data_size_map[type]);
    }

    inline uint32_t get_data_size(const TFlite::TensorType &type, size_t count) {
        return (uint32_t)(count * data_size_map[type]);
    }

    // following functions are for GPU
    virtual EnnReturn open_oplist(const model::component::OperatorList &operator_list) {
        ENN_UNUSED(operator_list);
        return ENN_RET_SUCCESS;
    }
    virtual void close_oplist(const uint64_t &oplist_id) {ENN_UNUSED(oplist_id);}

    UDOperators operators;
    UDTensors in_tensors;
    UDTensors out_tensors;

private:
    std::map<TFlite::TensorType, uint32_t> data_size_map = {
        {TFlite::TensorType::TensorType_FLOAT32, sizeof(float)},
        {TFlite::TensorType::TensorType_INT32, sizeof(int32_t)},
        {TFlite::TensorType::TensorType_UINT8, sizeof(uint8_t)},
        {TFlite::TensorType::TensorType_BOOL, sizeof(bool)},
        {TFlite::TensorType::TensorType_INT16, sizeof(int16_t)},
        {TFlite::TensorType::TensorType_INT8, sizeof(int8_t)},
    };
};

}  // namespace ud
}  // namespace enn

#endif  // USERDRIVER_COMMON_IOPERATION_CONSTRUCTOR_H_
