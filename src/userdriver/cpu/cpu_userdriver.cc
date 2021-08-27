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

#include "common/enn_debug.h"
#include "userdriver/cpu/cpu_userdriver.h"
#include "userdriver/cpu/common/NEONComputeLibrary.h"
#include "tool/profiler/include/ExynosNnProfilerApi.h"
#include "common/identifier_chopper.hpp"

namespace enn {
namespace ud {

namespace cpu {

CpuUserDriver& CpuUserDriver::get_instance(void) {
    static CpuUserDriver cpu_userdriver_instance;
    return cpu_userdriver_instance;
}

CpuUserDriver::~CpuUserDriver(void) {
    ENN_DBG_PRINT("started\n");

    op_constructor.reset();

    op_executor.reset();

    ENN_DBG_PRINT("ended successfully\n");
}

EnnReturn CpuUserDriver::Initialize(void) {
    ENN_DBG_PRINT("started\n");

    std::lock_guard<std::mutex> lock_guard_operators_map(mutex_operators_map);
    ud_operators_map.clear();

    std::lock_guard<std::mutex> lock_guard_exec_buffers_map(mutex_exec_buffers_map);
    exec_buffers_map.clear();

    auto compute_library = std::make_shared<NEONComputeLibrary>();

    op_constructor = std::unique_ptr<IOperationConstructor>(std::make_unique<OperationConstructor>(compute_library));

    op_executor = std::unique_ptr<IOperationExecutor>(std::make_unique<OperationExecutor>(compute_library));

    return ENN_RET_SUCCESS;
}

EnnReturn CpuUserDriver::OpenSubGraph(const model::component::OperatorList& operator_list) {
    ENN_DBG_PRINT("started\n");

    op_constructor->initialize_ud_operators();

    for (auto&& opr : operator_list) {
        EnnReturn ret = op_constructor->create_ud_operator(std::static_pointer_cast<enn::model::component::Operator>(opr));
        if (ret != ENN_RET_SUCCESS) {
            return ret;
        }
    }

    uint64_t operator_list_id = operator_list.get_id().get();
    ENN_DBG_PRINT("operator_list_id = 0x%" PRIx64 "\n", operator_list_id);

    return add_ud_operators(operator_list_id, op_constructor->get_ud_operators());
}

EnnReturn CpuUserDriver::PrepareSubGraph(const enn::runtime::ExecutableOperatorList& executable_operator_list) {
    ENN_DBG_PRINT("started\n");

    uint64_t operator_list_id = executable_operator_list.get_operator_list_id().get();
    ENN_DBG_PRINT("operator_list_id = 0x%" PRIx64 "\n", operator_list_id);

    uint64_t executable_id = executable_operator_list.get_id().get();
    ENN_DBG_PRINT("executable_id = 0x%" PRIx64 "\n", executable_id);

    UDOperators operators;

    if (get_ud_operators(operator_list_id, operators) != ENN_RET_SUCCESS) {
        return ENN_RET_FAILED;
    }

    UDBuffers buffers;

    if (op_executor->prepare(operators, buffers, executable_operator_list.get_buffer_table()) != ENN_RET_SUCCESS) {
        return ENN_RET_FAILED;
    }

    std::lock_guard<std::mutex> lock_guard_executable_id_map(mutex_executable_id_map);
    executable_id_map[operator_list_id].push_back(executable_id);

    return add_executable_buffers(executable_id, buffers);
}

EnnReturn CpuUserDriver::ExecuteSubGraph(const enn::runtime::OperatorListExecuteRequest& operator_list_execute_reqeust) {
    ENN_DBG_PRINT("started\n");

    uint64_t operator_list_id = operator_list_execute_reqeust.get_operator_list_id().get();
    ENN_DBG_PRINT("operator_list_id = 0x%" PRIx64 "\n", operator_list_id);

    uint64_t executable_id = operator_list_execute_reqeust.get_executable_operator_list_id().get();
    ENN_DBG_PRINT("executable_id = 0x%" PRIx64 "\n", executable_id);

    PROFILE_SCOPE("CPU_UD_Execution_#" + std::to_string(executable_id), util::chop_into_model_id(operator_list_id));

    UDOperators operators;

    if (get_ud_operators(operator_list_id, operators) != ENN_RET_SUCCESS) {
        return ENN_RET_FAILED;
    }

    UDBuffers buffers;
    auto& buffer_table = operator_list_execute_reqeust.get_buffer_table();

    if (get_executable_buffers(executable_id, buffers) != ENN_RET_SUCCESS) {
        if (op_executor->prepare(operators, buffers, buffer_table) != ENN_RET_SUCCESS) {
            return ENN_RET_FAILED;
        }

        std::lock_guard<std::mutex> lock_guard_executable_id_map(mutex_executable_id_map);
        executable_id_map[operator_list_id].push_back(executable_id);

        if (add_executable_buffers(executable_id, buffers) != ENN_RET_SUCCESS) {
            return ENN_RET_FAILED;
        }
    }

    return op_executor->execute(operators, buffers, buffer_table);
}

EnnReturn CpuUserDriver::CloseSubGraph(const model::component::OperatorList& operator_list) {
    ENN_DBG_PRINT("started\n");

    EnnReturn ret = ENN_RET_SUCCESS;

    uint64_t operator_list_id = operator_list.get_id().get();

    if (remove_ud_operators(operator_list_id) != ENN_RET_SUCCESS) {
        ret = ENN_RET_FAILED;
    }

    if (remove_executable_buffers(operator_list_id) != ENN_RET_SUCCESS) {
        ret = ENN_RET_FAILED;
    }

    return ret;
}

EnnReturn CpuUserDriver::Deinitialize(void) {
    ENN_DBG_PRINT("started\n");

    EnnReturn ret = ENN_RET_SUCCESS;

    std::lock_guard<std::mutex> lock_guard_operators_map(mutex_operators_map);
    ud_operators_map.clear();
    if (!ud_operators_map.empty()) {
        ENN_ERR_PRINT("ud_operators_map was not cleared.\n");
        ret = ENN_RET_FAILED;
    }

    std::lock_guard<std::mutex> lock_guard_exec_buffers_map(mutex_exec_buffers_map);
    exec_buffers_map.clear();
    if (!exec_buffers_map.empty()) {
        ENN_ERR_PRINT("exec_buffers_map was not cleared.\n");
        ret = ENN_RET_FAILED;
    }

    return ret;
}

EnnReturn CpuUserDriver::add_ud_operators(uint64_t id, UDOperators ud_operators) {
    ENN_DBG_PRINT("started, id = 0x%" PRIx64 "\n", id);

    EnnReturn ret = ENN_RET_SUCCESS;

    std::lock_guard<std::mutex> lock_guard(mutex_operators_map);
    if (ud_operators_map.find(id) != ud_operators_map.end()) {
        ENN_ERR_PRINT("ud_operators_map[%" PRIx64 "] was existed already.\n", id);
        ret = ENN_RET_FAILED;
    } else {
        ud_operators_map[id] = ud_operators;
    }

    return ret;
}

EnnReturn CpuUserDriver::get_ud_operators(uint64_t id, UDOperators& out_ud_operators) {
    ENN_DBG_PRINT("started, id = 0x%" PRIx64 "\n", id);

    EnnReturn ret = ENN_RET_SUCCESS;

    std::lock_guard<std::mutex> lock_guard(mutex_operators_map);
    if (ud_operators_map.find(id) == ud_operators_map.end()) {
        ENN_ERR_PRINT("ud_operators_map[%" PRIx64 "] was not found.\n", id);
        ret = ENN_RET_FAILED;
    } else {
        out_ud_operators = ud_operators_map[id];
    }

    return ret;
}

EnnReturn CpuUserDriver::remove_ud_operators(uint64_t id) {
    ENN_DBG_PRINT("started, id = 0x%" PRIx64 "\n", id);

    EnnReturn ret = ENN_RET_SUCCESS;

    std::lock_guard<std::mutex> lock_guard(mutex_operators_map);
    if (ud_operators_map.erase(id) != 1) {
        ENN_ERR_PRINT("remove ud_operators_map[%" PRIx64 "] failed.\n", id);
        ret = ENN_RET_FAILED;
    }

    return ret;
}

EnnReturn CpuUserDriver::add_executable_buffers(uint64_t id, const UDBuffers& executable_buffers) {
    ENN_DBG_PRINT("started, id = 0x%" PRIx64 "\n", id);

    EnnReturn ret = ENN_RET_SUCCESS;

    std::lock_guard<std::mutex> lock_guard(mutex_exec_buffers_map);
    if (exec_buffers_map.find(id) != exec_buffers_map.end()) {
        ENN_ERR_PRINT("exec_buffers_map[%" PRIx64 "] was existed already.\n", id);
        ret = ENN_RET_FAILED;
    } else {
        for (auto& buffer : executable_buffers) {
            buffer->set_id(id);
        }
        exec_buffers_map[id] = executable_buffers;
    }

    return ret;
}

EnnReturn CpuUserDriver::get_executable_buffers(uint64_t id, UDBuffers& out_executable_buffers) {
    ENN_DBG_PRINT("started, id = 0x%" PRIx64 "\n", id);

    EnnReturn ret = ENN_RET_SUCCESS;

    std::lock_guard<std::mutex> lock_guard(mutex_exec_buffers_map);
    if (exec_buffers_map.find(id) == exec_buffers_map.end()) {
        ENN_WARN_PRINT("exec_buffers_map[%" PRIx64 "] was not found.\n", id);
        ret = ENN_RET_FAILED;
    } else {
        out_executable_buffers = exec_buffers_map[id];
    }

    return ret;
}

EnnReturn CpuUserDriver::remove_executable_buffers(uint64_t id) {
    ENN_DBG_PRINT("started, id = 0x%" PRIx64 "\n", id);

    EnnReturn ret = ENN_RET_SUCCESS;

    // TODO(empire.jung, TBD): Make common for this code to remove the code repetition in CPU and GPU UD
    std::lock_guard<std::mutex> lock_guard_executable_id_map(mutex_executable_id_map);
    for (auto executable_id : executable_id_map[id]) {
        std::lock_guard<std::mutex> lock_guard(mutex_exec_buffers_map);
        auto exec_buffer = exec_buffers_map.find(executable_id);
        if (exec_buffer != exec_buffers_map.end()) {
            if (exec_buffers_map.erase(executable_id) == 0) {
                ENN_ERR_PRINT("remove exec_buffers_map[%" PRIx64 "] failed.\n", executable_id);
                ret = ENN_RET_FAILED;
            }
        }
    }

    executable_id_map[id].clear();
    executable_id_map.erase(id);

    return ret;
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
