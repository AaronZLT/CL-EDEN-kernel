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
 * @file    cpu_userdriver.h
 * @brief   This is common ENN CPU Userdriver API
 * @details This header defines ENN CPU Userdriver API.
 */
#ifndef USERDRIVER_CPU_CPU_USERDRIVER_H_
#define USERDRIVER_CPU_CPU_USERDRIVER_H_

#include <unordered_map>
#include <vector>
#include <mutex>

#include "userdriver/common/UserDriver.h"
#include "userdriver/common/operator_interfaces/userdriver_operator.h"
#include "userdriver/cpu/cpu_op_constructor.h"
#include "userdriver/cpu/cpu_op_executor.h"

namespace enn {
namespace ud {
namespace cpu {

class CpuUserDriver : public UserDriver {
public:
    static CpuUserDriver& get_instance(void);

    ~CpuUserDriver();

    EnnReturn Initialize(void) override;
    EnnReturn OpenSubGraph(const model::component::OperatorList& operator_list) override;
    EnnReturn PrepareSubGraph(const enn::runtime::ExecutableOperatorList& executable_operator_list) override;
    EnnReturn ExecuteSubGraph(const enn::runtime::OperatorListExecuteRequest& operator_list_execute_reqeust) override;
    EnnReturn CloseSubGraph(const model::component::OperatorList& operator_list) override;
    EnnReturn Deinitialize(void) override;

    EnnReturn get_graph_for_TC(uint64_t id, UDOperators& out_graph) {
        return get_ud_operators(id, out_graph);
    }

private:
    CpuUserDriver(void) : UserDriver(CPU_UD) {
        ENN_DBG_PRINT("started\n");
    }

    std::unique_ptr<IOperationConstructor> op_constructor;
    std::unique_ptr<IOperationExecutor> op_executor;

    std::mutex mutex_operators_map;
    std::unordered_map<uint64_t, UDOperators> ud_operators_map;
    EnnReturn add_ud_operators(uint64_t id, UDOperators ud_operators);
    EnnReturn get_ud_operators(uint64_t id, UDOperators& out_ud_operators);
    EnnReturn remove_ud_operators(uint64_t id);

    std::mutex mutex_exec_buffers_map;
    std::unordered_map<uint64_t, UDBuffers> exec_buffers_map;
    EnnReturn add_executable_buffers(uint64_t id, const UDBuffers& executable_buffers);
    EnnReturn get_executable_buffers(uint64_t id, UDBuffers& out_executable_buffers);
    EnnReturn remove_executable_buffers(uint64_t id);

    std::mutex mutex_executable_id_map;
    std::unordered_map<uint64_t, std::vector<uint64_t>> executable_id_map;
};

}  // namespace cpu
}  // namespace ud
}  // namespace enn

#endif  // USERDRIVER_CPU_CPU_USERDRIVER_H_
