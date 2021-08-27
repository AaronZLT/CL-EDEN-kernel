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

#ifndef SRC_COMMON_INCLUDE_ENN_MEMORY_MANAGER_H_
#define SRC_COMMON_INCLUDE_ENN_MEMORY_MANAGER_H_

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
#include "common/enn_debug.h"
#include "common/enn_utils.h"
#include "common/ienn_memory_manager.hpp"

#ifdef __ANDROID__
#include "common/enn_memory_allocator_device_buffer.h"
#else
#include "common/enn_memory_allocator_heap.h"
#endif

/* this is only for ion allocator */
constexpr uint32_t ENN_ALLOC_FLAG_CACHED = ION_FLAG_CACHED;
namespace enn {
class EnnMemoryManager : public IEnnMemoryManager {
public:
    using fd_type = int32_t;

    EnnMemoryManager();
    ~EnnMemoryManager();

    EnnReturn init();
    EnnReturn deinit();

    std::shared_ptr<EnnBufferCore> CreateMemory(int32_t size, enn::EnnMmType type, uint32_t flag = ENN_ALLOC_FLAG_CACHED);
    EnnReturn DeleteMemory(std::shared_ptr<EnnBufferCore> buffer) override;
    EnnReturn DeleteMemory(EnnBuffer *raw_buffer);
    std::shared_ptr<EnnBufferCore> CreateMemoryObject(fd_type fd, uint32_t size, void *va, uint32_t offset = 0);
#ifdef ENN_ANDROID_BUILD
    std::shared_ptr<EnnBufferCore> CreateMemoryFromFd(fd_type fd, uint32_t size, const native_handle_t *_handle = nullptr);
    std::shared_ptr<EnnBufferCore> CreateMemoryFromFdWithOffset(fd_type fd, uint32_t size, uint32_t offset,
                                                                const native_handle_t *_handle = nullptr);
#else
    std::shared_ptr<EnnBufferCore> CreateMemoryFromFd(uint32_t, uint32_t) {
        ENN_ERR_PRINT("Not supported in non-Android system.\n");
        return nullptr;
    }
    std::shared_ptr<EnnBufferCore> CreateMemoryFromFdWithOffset(fd_type, uint32_t, uint32_t, bool) {
        ENN_ERR_PRINT("Not supported in non-Android system.\n");
        return nullptr;
    }
#endif

    /* Get options */
    EnnReturn GetMemoryMeta(const std::shared_ptr<EnnBufferCore> & buffer, enn::EnnBufferMetaType type, uint32_t *out);

    /* for debug */
    EnnReturn ShowMemoryPool(void);

private:
    EnnMemoryAllocator *GetAllocator() {
        return emm_allocator.get();
    }

    EnnReturn GenerateMagic(const void *va, const uint32_t size, const uint32_t offset, uint32_t *out);
    std::vector<std::shared_ptr<EnnBufferCore>> memory_pool;
    std::mutex mm_mutex;
    std::unique_ptr<EnnMemoryAllocator> emm_allocator;

    /* environment */
    const int TRY_DEINIT_POOL_MAX = 100;  // to avoid inifinity loop
};

}  // namespace enn
#endif  // SRC_COMMON_INCLUDE_ENN_MEMORY_MANAGER_H_
