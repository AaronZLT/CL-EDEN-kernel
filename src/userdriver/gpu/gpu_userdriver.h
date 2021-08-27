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
 * @file    gpu_userdriver.h
 * @brief   This is common ENN GPU Userdriver API
 * @details This header defines ENN GPU Userdriver API.
 */
#ifndef USERDRIVER_GPU_GPU_USERDRIVER_H_
#define USERDRIVER_GPU_GPU_USERDRIVER_H_

#include <unordered_map>
#include <vector>
#include <mutex>
#include <tuple>

#include "userdriver/common/UserDriver.h"
#include "userdriver/common/operator_interfaces/userdriver_operator.h"
#include "userdriver/gpu/common/CLComputeLibrary.hpp"
#include "userdriver/gpu/gpu_op_constructor.h"
#include "userdriver/gpu/gpu_op_executor.h"

namespace enn {
namespace ud {
namespace gpu {

class GpuUserDriver : public UserDriver {
public:
    static GpuUserDriver& get_instance(void);

    ~GpuUserDriver();

    EnnReturn Initialize(void) override;
    EnnReturn OpenSubGraph(const model::component::OperatorList& operator_list) override;
    EnnReturn PrepareSubGraph(const enn::runtime::ExecutableOperatorList& executable_operator_list) override;
    EnnReturn ExecuteSubGraph(const enn::runtime::OperatorListExecuteRequest& operator_list_execute_reqeust) override;
    EnnReturn CloseSubGraph(const model::component::OperatorList& operator_list) override;
    EnnReturn Deinitialize(void) override;

    EnnReturn get_graph_for_TC(uint64_t id, UDOperators& out_graph) {
        UDTensors in, out;
        return get_ud_operators(id, out_graph, in, out);
    }

    // (haizhu.shao) only used for gpu_ud_test, as the START_PROFILER is called in Engine::EngineImpl::open_model which will not
    // be executed in the TC, and will cause "not find match id" in PROFILE_SCOPE on EnnUDOperator<_library_name_>::execute.
    void disable_profile_for_TC() {
        profile_enable_ = false;
    }

private:
    GpuUserDriver(void) : UserDriver(GPU_UD) {
        ENN_DBG_PRINT("started\n");
    }

    EnnReturn set_input_data(UDTensors& in_tensors, const model::memory::BufferTable& buffer_table);
    EnnReturn set_output_data(UDTensors& out_tensors, const model::memory::BufferTable& buffer_table);

    void print_raw_input(const uint32_t& in_index,
                    const DataType& data_type,
                    const uint32_t& num,
                    const model::memory::BufferTable& buffer_table);
    template <typename T> void print_data(T* data, const uint32_t& num, const std::string& file_name);

    std::shared_ptr<CLComputeLibrary> compute_library;
    std::unique_ptr<IOperationConstructor> op_constructor;
    std::unique_ptr<IOperationExecutor> op_executor;

    std::mutex mutex_constructor;

    std::mutex mutex_operators_map;
    std::unordered_map<uint64_t, std::tuple<UDOperators, UDTensors, UDTensors>> ud_operators_map;
    EnnReturn add_ud_operators(uint64_t id, UDOperators ud_operators, UDTensors in_tensors, UDTensors out_tensors);
    EnnReturn get_ud_operators(uint64_t id, UDOperators& out_ud_operators, UDTensors &in_tensors, UDTensors &out_tensors);
    EnnReturn remove_ud_operators(uint64_t id);
    bool profile_enable_ = true;
};

}  // namespace gpu
}  // namespace ud
}  // namespace enn

#endif  // USERDRIVER_GPU_GPU_USERDRIVER_H_
