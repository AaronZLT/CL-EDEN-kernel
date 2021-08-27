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

#ifndef SRC_COMMON_INCLUDE_ENN_MEMORY_ALLOCATOR_ION_H_
#define SRC_COMMON_INCLUDE_ENN_MEMORY_ALLOCATOR_ION_H_

#ifdef ENN_ALLOCATE_DMABUFHEAP
    #define DMABUF_ALLOCATOR(v) v
    #define ION_ALLOCATOR(v)
#else
    #define DMABUF_ALLOCATOR(v)
    #define ION_ALLOCATOR(v) v
#endif

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

#ifdef ENN_ALLOCATE_DMABUFHEAP
#include <BufferAllocator/BufferAllocator.h>
#else
#include <ion/ion.h>
#include <sys/mman.h>
#endif

namespace enn {
/* NOTE(hoon98.choi): If multiple allocators are required, this class might be hierarchical */
class EnnMemoryAllocatorDeviceBuffer : public EnnMemoryAllocator {
public:
    EnnMemoryAllocatorDeviceBuffer();
    virtual ~EnnMemoryAllocatorDeviceBuffer();

    std::shared_ptr<EnnBufferCore> CreateMemory(uint32_t size, enn::EnnMmType type, uint32_t flag);
    EnnReturn DeleteMemory(std::shared_ptr<EnnBufferCore> buffer);

private:
    EnnReturn EnnMemoryDeviceBufferOpen();
    EnnReturn EnnMemoryDeviceBufferClose();
    EnnReturn EnnMemoryAllocateDeviceBuffer(uint32_t size, uint32_t flag, int32_t *out_fd);
    EnnReturn EnnMemoryReleaseDeviceBuffer(int32_t fd);
    EnnReturn EnnMemoryReset(std::shared_ptr<EnnBufferCore> buffer);
    EnnReturn SetNativeHandle(std::shared_ptr<EnnBufferCore> buffer, int32_t alloc_fd);
    bool CheckAndHandleValidity(void *va, int size, std::shared_ptr<EnnBufferCore> buf);
    bool CheckAndHandleValidity(void *va, int size, int alloc_fd);

    DMABUF_ALLOCATOR(bool IsAllocatorAvailable() { return (_dmabuf_allocator != nullptr); });

    ION_ALLOCATOR(int32_t _ion_client_fd);
    ION_ALLOCATOR(int32_t _heap_mask);
    DMABUF_ALLOCATOR(std::unique_ptr<BufferAllocator> _dmabuf_allocator = nullptr);

    std::mutex mma_mutex;
};

}  // namespace enn
#endif  // SRC_COMMON_INCLUDE_ENN_MEMORY_ALLOCATOR_ION_H_
