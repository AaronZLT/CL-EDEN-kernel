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

#ifndef USERDRIVER_COMMON_OPERATOR_INTERFACES_USERDRIVER_OPERATOR_H_
#define USERDRIVER_COMMON_OPERATOR_INTERFACES_USERDRIVER_OPERATOR_H_

#include "client/enn_api-type.h"
#include "common/enn_debug.h"
#include "model/component/operator/operator.hpp"
#include "model/memory/buffer_table.hpp"
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/ITensor.hpp"
#include "common/identifier_chopper.hpp"

namespace enn {
namespace ud {

#define BUF_IN(_idx_) buffer->in[_idx_]
#define BUF_OUT(_idx_) buffer->out[_idx_]
#define BUF_DATA(_idx_) buffer->data[_idx_]
#define ARG_DATA(_idx_) data_[_idx_]

// DEFINE_EXECUTER macro function should be used with the above BUF_XXX() macros.
#define DEFINE_EXECUTOR(_library_name_, ...)                                                                \
    template <>                                                                                             \
    EnnReturn EnnUDOperator<_library_name_>::execute(const std::shared_ptr<UDBuffer> &buffer) {             \
        const std::string profile_label = name_ + "_#:" + std::to_string(id_);                              \
        if (buffer != nullptr)                                                                              \
            PROFILE_FROM(profile_label.c_str(), util::chop_into_model_id(buffer->get_id()));                \
        if (Status::SUCCESS != op_->execute(__VA_ARGS__)) {                                                 \
            return ENN_RET_FAILED;                                                                          \
        }                                                                                                   \
        if (buffer != nullptr)                                                                              \
            PROFILE_UNTIL(profile_label.c_str(), util::chop_into_model_id(buffer->get_id()));               \
        return ENN_RET_SUCCESS;                                                                             \
    }

class UDBuffer {
public:
    std::vector<std::shared_ptr<ITensor>> in;
    std::vector<std::shared_ptr<ITensor>> out;
    std::vector<std::shared_ptr<ITensor>> data;

    void set_id(uint64_t id) {
        execution_id = id;
    }

    uint64_t get_id() {
        return execution_id;
    }

private:
    uint64_t execution_id = 0;
};

class UDOperator {
public:
    virtual EnnReturn execute(const std::shared_ptr<UDBuffer> &buffer) = 0;
    virtual EnnReturn release() = 0;
    virtual ~UDOperator() {}

    UDOperator(const std::string name, const uint64_t id, const std::vector<std::shared_ptr<ITensor>> &in,
               const std::vector<std::shared_ptr<ITensor>> &out, const std::vector<std::shared_ptr<ITensor>> &data,
               const bool support_fp32_input_for_fp16 = false, const bool support_CPU_output_for_fp16 = false)
        : name_(name),
          id_(id),
          in_(in),
          out_(out),
          data_(data),
          support_FP32_input_for_FP16_(support_fp32_input_for_fp16),
          support_CPU_output_for_FP16_(support_CPU_output_for_fp16) {}

    std::string getName() {
        return name_;
    }

    uint64_t getId() {
        return id_;
    }

    std::vector<std::shared_ptr<ITensor>> getInTensors() {
        return in_;
    }

    std::vector<std::shared_ptr<ITensor>> getOutTensors() {
        return out_;
    }

    std::vector<std::shared_ptr<ITensor>> getDataTensors() {
        return data_;
    }

    bool isSupportFP32InputForFP16() {
        return support_FP32_input_for_FP16_;
    }

    bool isSupportCPUOutputForFP16() {
        return support_FP32_input_for_FP16_;
    }

protected:
    std::string name_;
    uint64_t id_;
    std::vector<std::shared_ptr<ITensor>> in_;
    std::vector<std::shared_ptr<ITensor>> out_;
    std::vector<std::shared_ptr<ITensor>> data_;
    bool support_FP32_input_for_FP16_;
    bool support_CPU_output_for_FP16_;
};  // class UDOperator

template <typename T>
class EnnUDOperator : public UDOperator {
public:
    EnnUDOperator(const std::string name, const uint64_t id, const std::vector<std::shared_ptr<ITensor>> &in,
                  const std::vector<std::shared_ptr<ITensor>> &out, const std::vector<std::shared_ptr<ITensor>> &data,
                  const std::shared_ptr<T> &op, const bool support_fp32_input_for_fp16_ = false,
                  const bool support_cpu_output_for_fp16_ = false)
        : UDOperator(name, id, in, out, data, support_fp32_input_for_fp16_, support_cpu_output_for_fp16_), op_(op) {}

    virtual EnnReturn execute(const std::shared_ptr<UDBuffer> &buffer);

    virtual EnnReturn release() {
        if (Status::SUCCESS != op_->release()) {
            return ENN_RET_FAILED;
        }
        return ENN_RET_SUCCESS;
    }

    std::shared_ptr<T> getOp() {
        return op_;
    }

private:
    std::shared_ptr<T> op_;
};

using UDOperators = std::shared_ptr<std::vector<std::shared_ptr<UDOperator>>>;

using UDBuffers = std::vector<std::shared_ptr<UDBuffer>>;

using UDTensors = std::vector<std::shared_ptr<ITensor>>;

}  // namespace ud
}  // namespace enn

#endif  // USERDRIVER_COMMON_OPERATOR_INTERFACES_USERDRIVER_OPERATOR_H_