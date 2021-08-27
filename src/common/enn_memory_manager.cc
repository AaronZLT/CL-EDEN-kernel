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

#include "common/enn_memory_manager.h"
#include "common/enn_debug.h"

#include <algorithm>

namespace enn {

using mem_obj = std::shared_ptr<EnnBufferCore>;

/* Implementation of EnnMemoryManager */
EnnMemoryManager::EnnMemoryManager() {
    ENN_DBG_PRINT("(+)\n");
}

EnnMemoryManager::~EnnMemoryManager() {
    ENN_DBG_PRINT("(-)\n");
    if (memory_pool.size() != 0) {
        ENN_WARN_PRINT("deinit() of Memory Manager is not called\n");
        deinit();
    }
}

EnnReturn EnnMemoryManager::init() {
    std::lock_guard<std::mutex> guard(mm_mutex);

    /* initialize member variables */
#ifdef __ANDROID__
    emm_allocator = std::make_unique<enn::EnnMemoryAllocatorDeviceBuffer>();  // Allocator can be changed in build time
#else
    emm_allocator = std::make_unique<enn::EnnMemoryAllocatorHeap>();
#endif
    return (emm_allocator != nullptr) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

EnnReturn EnnMemoryManager::deinit() {
    ENN_DBG_PRINT("(-)\n");
    /* NOTE(hoon98.choi): lock_guard should be set in API level */
    CHECK_AND_RETURN_ERR(GetAllocator() == nullptr, ENN_RET_FAILED, "Emm_allocator is not set\n");
    int try_max = TRY_DEINIT_POOL_MAX;
    while (memory_pool.size() > 0 && (--try_max) > 0) {
        ENN_MEM_PRINT("MemoryPool has %zu non-freed buffers(smart_ptr). Please check.\n", memory_pool.size());
        auto &buf = memory_pool[0];
        ENN_MEM_PRINT("Buffers to be removed: \n");
        buf->show();
        DeleteMemory(buf);
    }
    emm_allocator = nullptr;
    memory_pool.clear();

    return ENN_RET_SUCCESS;
}

std::shared_ptr<EnnBufferCore> EnnMemoryManager::CreateMemory(int32_t size, enn::EnnMmType type, uint32_t flag) {
    std::lock_guard<std::mutex> guard(mm_mutex);

    CHECK_AND_RETURN_ERR(GetAllocator() == nullptr, mem_obj(), "Emm_allocator is not set\n");
    auto ebc = GetAllocator()->CreateMemory(size, type, flag);

    CHECK_AND_RETURN_ERR(ebc == nullptr, mem_obj(), "Memory Allocation Err\n");

    /* generate magic and update magic */
    GenerateMagic(ebc->va, ebc->size, ebc->offset, &(ebc->magic));
    memory_pool.push_back(ebc);

    return ebc;
}

std::shared_ptr<EnnBufferCore> EnnMemoryManager::CreateMemoryObject(fd_type fd, uint32_t size, void *va, uint32_t offset) {
    std::lock_guard<std::mutex> guard(mm_mutex);
    ENN_MEM_PRINT("Create Memory Object with fd(%d), size(%d), va(%p)\n", fd, size, va);
    auto ebc = std::make_shared<EnnBufferCore>(reinterpret_cast<void *>(reinterpret_cast<char *>(va) + offset),
                                               EnnMmType::kEnnMmTypeCloned, size, fd, ION_FLAG_CACHED, va, offset);
    CHECK_AND_RETURN_ERR(ebc == nullptr, mem_obj(), "Memory Allocation Err\n");

    /* generate magic and update magic */
    GenerateMagic(ebc->va, ebc->size, ebc->offset, &(ebc->magic));
    memory_pool.push_back(ebc);

    return ebc;
}

EnnReturn EnnMemoryManager::DeleteMemory(EnnBuffer *raw_buffer) {
    CHECK_AND_RETURN_ERR(raw_buffer == nullptr, ENN_RET_FAILED, "Parameter Buffer is nullptr\n");
    std::shared_ptr<EnnBufferCore> buffer = nullptr;
    {
        std::lock_guard<std::mutex> guard(mm_mutex);
        for (auto &buf : memory_pool) {
            if (buf->va == raw_buffer->va && buf->size == raw_buffer->size && buf->offset == raw_buffer->offset) {
                buffer = buf;
            }
        }
    }
    CHECK_AND_RETURN_WARN(raw_buffer == nullptr, ENN_RET_FAILED, "There's no buffer in memory pool(%p, %d, %d)\n",
                          raw_buffer->va, raw_buffer->size, raw_buffer->offset);
    return DeleteMemory(buffer);
}

EnnReturn EnnMemoryManager::DeleteMemory(std::shared_ptr<EnnBufferCore> buffer) {
    CHECK_AND_RETURN_ERR(buffer == nullptr, ENN_RET_FAILED, "Parameter Buffer is nullptr\n");
    CHECK_AND_RETURN_ERR(GetAllocator() == nullptr, ENN_RET_FAILED, "Emm_allocator is not set\n");

    ENN_MEM_PRINT("Memory will be deleted: fd(%d), size(%d), va(%p), type(%d)\n", buffer->fd, buffer->size, buffer->va,
                  static_cast<int>(buffer->type));

    /* checking magic */
    uint32_t out;
    EnnReturn result = ENN_RET_SUCCESS;
    GenerateMagic(buffer->va, buffer->size, buffer->offset, &out);
    if (out != buffer->magic) {
        ENN_WARN_PRINT("Memory structure has currupted(magic failed, %p %d ++ %d)\n", buffer->va, buffer->size,
                       buffer->offset);
        result = ENN_RET_MEM_ERR;
    } else {
        ENN_MEM_PRINT("Magic: %X = %X, passed\n", out, buffer->magic);
    }

    {
        std::lock_guard<std::mutex> guard(mm_mutex);
        auto iter = std::find(memory_pool.begin(), memory_pool.end(), buffer);
        if (iter == memory_pool.end()) {
            ENN_WARN_PRINT("Buffer %p is not in memory pool!\n", buffer->va);
        } else {
            memory_pool.erase(iter);
        }
    }

    if (buffer->type == enn::EnnMmType::kEnnMmTypeCloned) {
        return result;
    }

    auto result_dm = GetAllocator()->DeleteMemory(buffer);

    return result == ENN_RET_SUCCESS ? result_dm : result;
}

#ifdef __ANDROID__
std::shared_ptr<EnnBufferCore> EnnMemoryManager::CreateMemoryFromFd(fd_type fd, uint32_t size, const native_handle_t *_handle) {
    return CreateMemoryFromFdWithOffset(fd, size, 0, _handle);
}

std::shared_ptr<EnnBufferCore> EnnMemoryManager::CreateMemoryFromFdWithOffset(fd_type fd, uint32_t size, uint32_t offset,
                                                                              const native_handle_t *_handle) {
    std::lock_guard<std::mutex> guard(mm_mutex);

    void *va = mmap(NULL, size + offset, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if ((void *)-1 == va) {
        ENN_ERR_PRINT(" MMAP(%d ++ %d) occurs an error (%d): pid(%d), fd(%d)\n", size, offset, errno, enn::util::get_pid(),
                      fd);
        return mem_obj();
    }

    /* generate an object of core parent */
    auto ebc = std::make_shared<EnnBufferCore>(reinterpret_cast<void *>(reinterpret_cast<char *>(va) + offset),
                                               enn::EnnMmType::kEnnMmTypeExternalIon, size, fd, ION_FLAG_CACHED, va,
                                               offset);  // cache enabled
    CHECK_AND_RETURN_ERR((ebc == nullptr), mem_obj(), "Failed allocate EnnBufferCoreInterface(fd: %d (sz: %d ++ %d)\n", fd,
                         size, offset);

    if (_handle) {
        ebc->SetNativeHandle(_handle);  // if _handled is passed as a parameter, _handle is cloned
        ENN_MEM_COUT << "fd generates: " << fd << " -> " << ebc->fd << std::endl;
    }

    /* generate magic and update magic */
    GenerateMagic(ebc->va, ebc->size, ebc->offset, &(ebc->magic));
    memory_pool.push_back(ebc);

    return ebc;
}
#endif

/* TODO(hoon98.choi, 6/11): implement layer after metatype is defined */
EnnReturn EnnMemoryManager::GetMemoryMeta(const std::shared_ptr<EnnBufferCore> &buffer, enn::EnnBufferMetaType type,
                                          uint32_t *out) {
    ENN_UNUSED(buffer);
    ENN_UNUSED(type);
    ENN_UNUSED(out);

    return ENN_RET_SUCCESS;
}

/* for debug */
EnnReturn EnnMemoryManager::ShowMemoryPool(void) {
    std::lock_guard<std::mutex> guard(mm_mutex);
    ENN_MEM_PRINT(" # memory pool has %zu element(s). \n", memory_pool.size());
    for (auto &buf : memory_pool) { buf->show(); }
    return ENN_RET_SUCCESS;
}

EnnReturn EnnMemoryManager::GenerateMagic(const void *va, const uint32_t size, const uint32_t offset, uint32_t *out) {
    CHECK_AND_RETURN_ERR(out == nullptr, ENN_RET_FAILED, "output ptr is nullptr\n");
    uint64_t par = ENN_MM_GEN_MAGIC(va, size, offset);
    *out = par & BIT_MASK(32);

    ENN_MEM_PRINT("Generate magic: %p %d %d --> 0x%08X\n", va, size, offset, *out);

    return ENN_RET_SUCCESS;
}

}  // namespace enn
