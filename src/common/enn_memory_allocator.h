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

#ifndef SRC_COMMON_INCLUDE_ENN_MEMORY_ALLOCATOR_H_
#define SRC_COMMON_INCLUDE_ENN_MEMORY_ALLOCATOR_H_

/**
 * @brief Memory management modules
 *
 * @file enn_memory_manager.h
 * @author Hoon Choi (hoon98.choi@samsung.com)
 * @date 2020-12-18
 */

#include <cstdint>
#include <vector>
#include <mutex>
#include <atomic>
#include <memory>
#include "common/enn_memory_manager-type.h"

namespace enn {
class EnnBufferCore;

/* EnnMemoryAllocator includes interface for enn memory manager */
class EnnMemoryAllocator {
public:
    EnnMemoryAllocator() {}
    virtual ~EnnMemoryAllocator() {}

    /* Should be implemented */
    virtual std::shared_ptr<EnnBufferCore> CreateMemory(uint32_t size, enn::EnnMmType type, uint32_t flag) = 0;
    virtual EnnReturn DeleteMemory(std::shared_ptr<EnnBufferCore> buffer) = 0;
    virtual EnnReturn EnnMemoryReset(std::shared_ptr<EnnBufferCore> buffer) = 0;
};

}  // namespace enn

#endif  // SRC_COMMON_INCLUDE_ENN_MEMORY_ALLOCATOR_H_
