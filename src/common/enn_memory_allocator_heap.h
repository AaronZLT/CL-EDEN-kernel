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

#ifndef SRC_COMMON_INCLUDE_ENN_MEMORY_ALLOCATOR_HEAP_H_
#define SRC_COMMON_INCLUDE_ENN_MEMORY_ALLOCATOR_HEAP_H_

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
#include <sys/mman.h>

#include "common/enn_debug.h"
#include "common/enn_utils.h"
#include "common/enn_memory_manager-type.h"
#include "common/enn_memory_allocator.h"

namespace enn {

/**
 * @brief Allocator for heap, which is used in x86 or similar machine that not use ION(dmabuf)
 * 
 */
class EnnMemoryAllocatorHeap : public EnnMemoryAllocator {
public:
    EnnMemoryAllocatorHeap();
    virtual ~EnnMemoryAllocatorHeap();

    std::shared_ptr<EnnBufferCore> CreateMemory(uint32_t size, enn::EnnMmType type, uint32_t flag);
    EnnReturn DeleteMemory(std::shared_ptr<EnnBufferCore> buffer);
    EnnReturn EnnMemoryReset(std::shared_ptr<EnnBufferCore> buffer);

private:
    std::mutex mma_mutex;
};

}  // namespace enn
#endif  // SRC_COMMON_INCLUDE_ENN_MEMORY_ALLOCATOR_ION_H_
