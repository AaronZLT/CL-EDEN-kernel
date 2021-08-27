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
 * @file    npu_userdriver.h
 * @brief   This is common ENN NPU Userdriver API
 * @details This header defines ENN NPU Userdriver API.
 */
#ifndef USERDRIVER_NPU_NPU_USERDRIVER_H_
#define USERDRIVER_NPU_NPU_USERDRIVER_H_

#include <unordered_map>
#include <vector>
#include <mutex>
#include "userdriver/common/UserDriver.h"
#include "userdriver/common/operator_interfaces/userdriver_operator.h"
#include "userdriver/unified/link_vs4l.h"  // link

namespace enn {
namespace ud {
namespace npu {

class ExecutableNpuUDOperator {
public:
    EnnReturn init(uint32_t in_buf_cnt, uint32_t out_buf_cnt);
    EnnReturn set(model_info_t* op_info, const model::memory::BufferTable& buffer_table);
    req_info_t& get(void) { return executable_op_info; }
    std::shared_ptr<eden_memory_t> get_inputs(void) { return executable_op_info.inputs; }
    std::shared_ptr<eden_memory_t> get_outputs(void) { return executable_op_info.outputs; }
    EnnReturn deinit(void);

    void dump(void);
private:
    req_info_t executable_op_info;
};

// TODO(jungho7.kim, 6/30): add lock mechanism to this data structure
using ExecutableOperatorMap = std::unordered_map<uint64_t, std::shared_ptr<ExecutableNpuUDOperator>>;

class NpuUDOperator : public AccUDOperator {
public:
    explicit NpuUDOperator() : AccUDOperator(), op_info() {}
    ~NpuUDOperator() {}

    EnnReturn init(uint32_t in_buf_cnt, uint32_t out_buf_cnt);
    EnnReturn set(uint32_t in_buf_cnt, uint32_t out_buf_cnt, model::component::Operator::Ptr rt_opr_npu,
            uint64_t operator_list_id, uint64_t unified_op_id);
    model_info_t& get(void) { return op_info; }
    EnnReturn deinit(void);

    EnnReturn add_executable_op(uint64_t exec_op_id, const std::shared_ptr<ExecutableNpuUDOperator>& executable_op);
    std::vector<uint64_t> get_all_executable_op_id();
    std::shared_ptr<ExecutableNpuUDOperator> get_executable_op(uint64_t exec_op_id);
    EnnReturn remove_executable_op(uint64_t exec_op_id);
    uint64_t get_id(void);
    void dump(void);

private:
    model_info_t op_info;
    ExecutableOperatorMap executable_op_map;
    std::mutex mutex_executable_op_map;
};

using NpuUDOperators = std::vector<std::shared_ptr<NpuUDOperator>>;
using SubGraphMap = std::unordered_map<uint64_t, NpuUDOperators>;

class NpuUserDriver : public UserDriver {
public:
    enum class NpuUdStatus { NONE, INITIALIZED, SHUTDOWNED };

    static NpuUserDriver& get_instance(void);
    ~NpuUserDriver(void);

    EnnReturn Initialize(void) override;
    EnnReturn OpenSubGraph(const model::component::OperatorList& operator_list) override;
    // Temporarily, this function was added to support the unified UD
    // TODO(jungho7.kim): remove this method
    EnnReturn OpenSubGraph(const model::component::OperatorList& operator_list,
            uint64_t operator_list_id, uint64_t unified_op_id);
    EnnReturn PrepareSubGraph(const enn::runtime::ExecutableOperatorList& executable_operator_list) override;
    EnnReturn ExecuteSubGraph(const enn::runtime::OperatorListExecuteRequest& operator_list_execute_request) override;
    EnnReturn CloseSubGraph(const model::component::OperatorList& operator_list) override;
    // Temporarily, this function was added to support the unified UD
    // TODO(jungho7.kim): remove this method
    EnnReturn CloseSubGraph(const model::component::OperatorList& operator_list, uint64_t operator_list_id);
    EnnReturn Deinitialize(void) override;

    uint64_t get_operator_list_id();
    EnnReturn add_graph(uint64_t id, NpuUDOperators ud_operators);
    EnnReturn update_graph(uint64_t id, NpuUDOperators ud_operators);
    EnnReturn get_graph(uint64_t id, NpuUDOperators& out_graph);
    EnnReturn remove_graph(uint64_t id);
    EnnReturn set_npu_ud_status(NpuUdStatus npu_ud_status);
    NpuUdStatus get_npu_ud_status() const { return npu_ud_status_; }

private:
    NpuUserDriver(void) : UserDriver(NPU_UD), npu_ud_status_(NpuUdStatus::NONE) {
        ENN_DBG_PRINT("started\n");
    }

    SubGraphMap ud_operator_list_map;
    NpuUdStatus npu_ud_status_;
    accelerator_device acc_ = ACCELERATOR_NPU;
    std::mutex mutex_ud_operator_list_map;
    std::mutex mutex_npu_ud_status;
};

}  // namespace npu
}  // namespace ud
}  // namespace enn

#endif  // USERDRIVER_NPU_NPU_USERDRIVER_H_
