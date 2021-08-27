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

#include "common/enn_memory_allocator_heap.h"

#include <algorithm>
#include <string>

namespace enn {

EnnMemoryAllocatorHeap::EnnMemoryAllocatorHeap() {
}

EnnMemoryAllocatorHeap::~EnnMemoryAllocatorHeap() {
}

std::shared_ptr<EnnBufferCore> EnnMemoryAllocatorHeap::CreateMemory(uint32_t size, enn::EnnMmType type, uint32_t flag) {
    std::lock_guard<std::mutex> guard(mma_mutex);
    std::shared_ptr<char> shared_va(new char[size], std::default_delete<char[]>());
    auto buf = std::make_shared<EnnBufferCore>(reinterpret_cast<void *>(shared_va.get()), size, shared_va);
    CHECK_AND_RETURN_ERR(!buf.get() || !shared_va, nullptr, "Failed to allocate EnnBuffer: va(%p), size(%d)\n",
                         shared_va.get(), size);
    ENN_MEM_PRINT("Heap Memory creation: va(%p), size(%d)\n", shared_va.get(), size);

    return buf;
}

EnnReturn EnnMemoryAllocatorHeap::EnnMemoryReset(std::shared_ptr<EnnBufferCore> buffer) {
    buffer->va = 0;
    buffer->fd = 0;
    buffer->status = enn::EnnBufferStatus::kEnnMmStatusFreed;
    buffer.reset();

    return ENN_RET_SUCCESS;
}

EnnReturn EnnMemoryAllocatorHeap::DeleteMemory(std::shared_ptr<EnnBufferCore> buffer) {
    std::lock_guard<std::mutex> guard(mma_mutex);
    CHECK_AND_RETURN_ERR((buffer->type != EnnMmType::kEnnMmTypeHeap), ENN_RET_FAILED, "MmType is not Heap allocator\n");
    ENN_MEM_PRINT("Heap Memory Deletion: va(%p), size(%d)\n", buffer->va, buffer->size);
    EnnMemoryReset(buffer);

    return ENN_RET_SUCCESS;
}

// ReleaseMemoryBuffer
// AllocateMemoryBuffer

}  // namespace enn
