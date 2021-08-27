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
 * @file    gpu_op_executor.h
 * @brief   This is common ENN GPU Userdriver API
 * @details This header defines ENN GPU Userdriver API.
 */
#ifndef USERDRIVER_GPU_GPU_OP_EXECUTOR_H_
#define USERDRIVER_GPU_GPU_OP_EXECUTOR_H_

#include <unordered_map>
#include <vector>

#include "userdriver/common/IOperationExecutor.h"
#include "userdriver/gpu/common/CLComputeLibrary.hpp"
#include "userdriver/common/operator_interfaces/userdriver_operator.h"

namespace enn {
namespace ud {
namespace gpu {

class OperationExecutor : public IOperationExecutor {
public:
    explicit OperationExecutor(std::shared_ptr<CLComputeLibrary> compute_library) :
        IOperationExecutor("GPU", compute_library), compute_library_(compute_library) {}
    ~OperationExecutor() = default;
    EnnReturn execute(UDOperators& operators, UDBuffers& buffers, const model::memory::BufferTable& buffer_table) override;

private:
    void dump_operator_output_gpu(std::shared_ptr<enn::ud::UDOperator>& op);
    void dump_operator_input_gpu(std::shared_ptr<enn::ud::UDOperator>& op);
    std::shared_ptr<CLComputeLibrary> compute_library_;
};

}  // namespace gpu
}  // namespace ud
}  // namespace enn

#endif  // USERDRIVER_GPU_GPU_OP_EXECUTOR_H_
