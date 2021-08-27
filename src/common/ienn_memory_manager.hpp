/*
 * Copyright (C) 2020 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or
 * distributed, transmitted, transcribed, stored in a retrieval system or
 * translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed to third parties
 * without the express written permission of Samsung Electronics.
 */

#ifndef SRC_COMMON_IENN_MEMORY_MANAGER_HPP_
#define SRC_COMMON_IENN_MEMORY_MANAGER_HPP_

#include <memory>

#include "common/enn_memory_manager-type.h"
#include "client/enn_api-type.h"

namespace enn {

// TODO(yc18.cho, hoon98.choi, 7/10): It is temporary version for loosely decoupling
//  between MemoryManager and Model & ExecutableModel in runtime.
//  It shouild be redesigend with EnnBufferCore so that runtime and MemoryManger are decoupled.
//  In other words, MemoryManager creates and passes low-weight object that has methods that runtime
//  needs for creating a memory object and release it in the runtime space.
class IEnnMemoryManager {
 public:
    virtual ~IEnnMemoryManager() = default;
    virtual EnnReturn DeleteMemory(std::shared_ptr<EnnBufferCore> buffer) = 0;
};

};  // namespace enn

#endif  // SRC_COMMON_IENN_MEMORY_MANAGER_HPP_
