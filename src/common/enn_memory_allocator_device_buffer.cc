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

#include "common/enn_debug.h"
#include "common/enn_utils.h"
#include "common/enn_memory_allocator_device_buffer.h"

#include <algorithm>
#include <string>

#ifndef ENN_ANDROID_BUILD
#error "This file should compile with ENN_ANDROID_BUILD"
#endif


/**
 * NOTE(hoon98.choi): In Android S, libion cannot be called anymore.
 *                    Therefore we should use libdmabufheap alternatively, and build system
 *                    defines ENN_ALLOCATE_DMABUFHEAP if user tries to build with Android S option
 */
namespace enn {

ION_ALLOCATOR(constexpr int32_t BUF_CLIENT_FD_IDLE = -1);
ION_ALLOCATOR(constexpr int32_t BUF_HEAP_ID_IDLE = -1);
ION_ALLOCATOR(constexpr char BUF_HEAP_NAME[] = "ion_system_heap");
DMABUF_ALLOCATOR(constexpr char BUF_HEAP_NAME[] = "dmabuf_system_heap");
DMABUF_ALLOCATOR(constexpr char HEAP_NAME_CACHED[] = "system");
DMABUF_ALLOCATOR(constexpr char HEAP_NAME_UNCACHED[] = "system-uncached");

/* ION emm_allocator */
EnnMemoryAllocatorDeviceBuffer::EnnMemoryAllocatorDeviceBuffer() {
    std::lock_guard<std::mutex> guard(mma_mutex);
    CHECK_AND_RETURN_VOID((EnnMemoryDeviceBufferOpen()), "ION allocator initialization failed\n");
}

EnnMemoryAllocatorDeviceBuffer::~EnnMemoryAllocatorDeviceBuffer() {
    std::lock_guard<std::mutex> guard(mma_mutex);
    EnnMemoryDeviceBufferClose();
}

EnnReturn EnnMemoryAllocatorDeviceBuffer::EnnMemoryDeviceBufferClose() {
    DMABUF_ALLOCATOR(CHECK_AND_RETURN_ERR(!IsAllocatorAvailable(), ENN_RET_FAILED, "Allocator is not available.\n"));
    DMABUF_ALLOCATOR(_dmabuf_allocator.reset());  // explicitly delete unique_ptr

    ION_ALLOCATOR(auto ret = ion_close(_ion_client_fd));
    ION_ALLOCATOR(CHECK_AND_RETURN_ERR(ret < 0, ENN_RET_FAILED, "ion close error from libion(ret: %d, fd: %d)\n", ret,
                                       _ion_client_fd));
    ION_ALLOCATOR(_ion_client_fd = BUF_CLIENT_FD_IDLE);
    ENN_MEM_PRINT("DeviceBuffer successfully closed\n");

    return ENN_RET_SUCCESS;
}

EnnReturn EnnMemoryAllocatorDeviceBuffer::EnnMemoryDeviceBufferOpen() {
#ifdef ENN_ALLOCATE_DMABUFHEAP
    _dmabuf_allocator = std::make_unique<BufferAllocator>();
    CHECK_AND_RETURN_ERR(!IsAllocatorAvailable(), ENN_RET_FAILED, "Allocator is not available.\n");

    _dmabuf_allocator->MapNameToIonHeap(HEAP_NAME_UNCACHED, BUF_HEAP_NAME, 0, 1 << 0, 0);  // this is conventional
    _dmabuf_allocator->MapNameToIonHeap(HEAP_NAME_CACHED, BUF_HEAP_NAME, ION_FLAG_CACHED, 1 << 0, ION_FLAG_CACHED);
#else
    int client_fd = BUF_CLIENT_FD_IDLE, heap_id = BUF_HEAP_ID_IDLE;

    /* NOTE(hoon98.choi): ion_heap_data is taken from kernel source code */
    constexpr uint32_t MAX_HEAP_NAME = 32;
    struct ion_heap_data {
        char name[MAX_HEAP_NAME];
        uint32_t type;
        uint32_t heap_id;
        uint32_t reserved0;
        uint32_t reserved1;
        uint32_t reserved2;
    };

    client_fd = ion_open();
    if (!ion_is_legacy(client_fd)) {
        int cnt;
        ion_query_heap_cnt(client_fd, &cnt);
        struct ion_heap_data heaps[cnt];
        ion_query_get_heaps(client_fd, cnt, heaps);

        for (int i = 0; i < cnt; i++) {
            if (!strncmp(heaps[i].name, BUF_HEAP_NAME, 15)) {
                heap_id = heaps[i].heap_id;
                break;
            }
        }
        CHECK_AND_RETURN_ERR((heap_id == BUF_HEAP_ID_IDLE), ENN_RET_FAILED, "Couldn't find system_heap, Error\n");
    } else {
        heap_id = 0;
    }
    _heap_mask = 1 << heap_id;
    _ion_client_fd = client_fd;

    ENN_MEM_PRINT("Ion Open Success: _ion_client_fd(%d), HeapMask(%d)\n", _ion_client_fd, _heap_mask);
#endif  // ENN_ALLOCATE_DMABUFHEAP

    return ENN_RET_SUCCESS;
}

EnnReturn EnnMemoryAllocatorDeviceBuffer::EnnMemoryAllocateDeviceBuffer(uint32_t size, uint32_t flag, int32_t *out_fd) {
#ifdef ENN_ALLOCATE_DMABUFHEAP
    CHECK_AND_RETURN_ERR(!IsAllocatorAvailable(), ENN_RET_FAILED, "Allocator is not available.\n");
    if (flag) {
        *out_fd = _dmabuf_allocator->Alloc(HEAP_NAME_CACHED, size);
    } else {
        *out_fd = _dmabuf_allocator->Alloc(HEAP_NAME_UNCACHED, size);
    }
    return (*out_fd < 0) ? ENN_RET_FAILED : ENN_RET_SUCCESS;
#else
    int ret = ion_alloc_fd(_ion_client_fd, size, 0, _heap_mask, flag, out_fd);
    return (ret < 0) ? ENN_RET_FAILED : ENN_RET_SUCCESS;
#endif
}

EnnReturn EnnMemoryAllocatorDeviceBuffer::EnnMemoryReleaseDeviceBuffer(int32_t fd) {
    ENN_MEM_PRINT("try to close fd: %d\n", fd);
    close(fd);
    return ENN_RET_SUCCESS;
}

bool EnnMemoryAllocatorDeviceBuffer::CheckAndHandleValidity(void *va, int size, int alloc_fd) {
    ENN_UNUSED(size);
    if (va != nullptr)
        return true;
    close(alloc_fd);
    return false;
}

bool EnnMemoryAllocatorDeviceBuffer::CheckAndHandleValidity(void *va, int size, std::shared_ptr<EnnBufferCore> buf) {
    if (buf == nullptr && va != nullptr)  // allocation failed, mmap success
        munmap(va, size);
    else if (buf != nullptr && va != nullptr)  // success
        return true;
    if (buf->fd != ENN_MM_FD_NOT_DEFINED)  // mmap failed
        close(buf->fd);
    return false;
}

std::shared_ptr<EnnBufferCore> EnnMemoryAllocatorDeviceBuffer::CreateMemory(uint32_t size, enn::EnnMmType type,
                                                                            uint32_t flag) {
    std::lock_guard<std::mutex> guard(mma_mutex);
    int alloc_fd;
    CHECK_AND_RETURN_ERR(type != EnnMmType::kEnnMmTypeIon, nullptr, "MmType is not ION allocator\n");
    ENN_MEM_PRINT("Ion Memory creation Start: (size, flag) : (%d, %d)\n", size, flag);

    /* allocate space */
    auto ret = EnnMemoryAllocateDeviceBuffer(size, flag, &alloc_fd);
    CHECK_AND_RETURN_ERR(ret, nullptr, "Error from EnnMemoryAllocateDeviceBuffer: size(%d), flag(%d)\n", size, flag);

    /* map */
    void *va = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, alloc_fd, 0);
    CHECK_AND_RETURN_ERR(!CheckAndHandleValidity(va, size, alloc_fd), nullptr, "Failed to mmap(fd: %d, size: %d)\n",
                         alloc_fd, size);

    /* generate an object of core parent */
    // TODO(hoon98.choi, 6/11): swap location of variable flag, offset
    auto buf = std::make_shared<EnnBufferCore>(va, EnnMmType::kEnnMmTypeIon, size, alloc_fd, flag);
    CHECK_AND_RETURN_ERR(!CheckAndHandleValidity(va, size, buf), nullptr,
                         "Failed to allocate EnnBuffer: buf(%p), size(%d)\n", buf.get(), size);
    SetNativeHandle(buf, alloc_fd);

    ENN_MEM_PRINT("Ion Memory creation Completed: (size %d, flag %d, fd %d, va %p)\n", size, flag, alloc_fd, va);

    return buf;
}

EnnReturn EnnMemoryAllocatorDeviceBuffer::SetNativeHandle(std::shared_ptr<EnnBufferCore> buffer, int32_t alloc_fd) {
    return buffer->SetNativeHandle(alloc_fd);
}

EnnReturn EnnMemoryAllocatorDeviceBuffer::EnnMemoryReset(std::shared_ptr<EnnBufferCore> buffer) {
    if (buffer->base_va)
        munmap(buffer->base_va, buffer->size + buffer->offset);
    else
        ENN_WARN_COUT << "base address of buffer is null, couldn't unmap." << std::endl;
    buffer.reset();

    return ENN_RET_SUCCESS;
}

EnnReturn EnnMemoryAllocatorDeviceBuffer::DeleteMemory(std::shared_ptr<EnnBufferCore> buffer) {
    std::lock_guard<std::mutex> guard(mma_mutex);
    CHECK_AND_RETURN_ERR((buffer->fd == ENN_MM_FD_NOT_DEFINED), ENN_RET_FAILED, "fd is not allocated. Please check\n");
    CHECK_AND_RETURN_ERR((buffer->type != EnnMmType::kEnnMmTypeIon && buffer->type != enn::EnnMmType::kEnnMmTypeExternalIon),
                         ENN_RET_FAILED, "MmType is not Device allocator\n");

    ENN_MEM_PRINT("Memory deletion: (fd, size, flag) : (%d, %d, %d)\n", buffer->fd, buffer->size, buffer->cache_flag);
    if (buffer->ntv_handle != nullptr) {
        native_handle_delete(buffer->ntv_handle);
        buffer->ntv_handle = nullptr;
    }
    EnnMemoryReleaseDeviceBuffer(buffer->fd);
    EnnMemoryReset(buffer);

    return ENN_RET_SUCCESS;
}


// ReleaseMemoryBuffer
// AllocateMemoryBuffer

}  // namespace enn
