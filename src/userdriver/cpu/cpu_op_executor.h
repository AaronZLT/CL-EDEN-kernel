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
 * @file    cpu_op_executor.h
 * @brief   This is common ENN CPU Userdriver API
 * @details This header defines ENN CPU Userdriver API.
 */
#ifndef USERDRIVER_CPU_CPU_OP_EXECUTOR_H_
#define USERDRIVER_CPU_CPU_OP_EXECUTOR_H_

#include <unordered_map>
#include <vector>

#include "userdriver/common/IOperationExecutor.h"
#include "userdriver/common/operator_interfaces/interfaces/IComputeLibrary.h"
#include "userdriver/common/operator_interfaces/userdriver_operator.h"

namespace enn {
namespace ud {
namespace cpu {

class OperationExecutor : public IOperationExecutor {
public:
    explicit OperationExecutor(std::shared_ptr<IComputeLibrary> compute_library_)
        : IOperationExecutor("CPU", compute_library_) {}
    ~OperationExecutor() = default;
};

}  // namespace cpu
}  // namespace ud
}  // namespace enn

#endif  // USERDRIVER_CPU_CPU_OP_EXECUTOR_H_
