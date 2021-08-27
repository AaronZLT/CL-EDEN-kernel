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

#ifndef SRC_COMMON_INCLUDE_ENN_MEMORY_MANAGER_TYPE_H_
#define SRC_COMMON_INCLUDE_ENN_MEMORY_MANAGER_TYPE_H_

/**
 * @brief Memory management types
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

// TODO(hoon98.choi, TBD): remove ifdef for android

#ifdef ENN_ANDROID_BUILD
#include <ion/ion.h>
#include <sys/mman.h>
#include <cutils/native_handle.h>
#else
#define ION_FLAG_CACHED 0
#endif

/* This generates magic number with va, size, offset */
#define ENN_MM_GEN_MAGIC(va, size, offset) \
    reinterpret_cast<uint64_t>((BIT_MASK(32) & reinterpret_cast<uint64_t>(va)) + size * 0xFF + offset * 0x11)
#define ENN_MM_FD_NOT_DEFINED (0xFFFFFFF)

namespace enn {

enum class EnnBufferMetaType {
    kEnnMmMetaFd,
    kEnnMmMetaFlag,
    kEnnMmMetaMapped,
    kEnnMmMetaMax,
};

enum class EnnBufferStatus {
    kEnnMmStatusUnMapped = 0,
    kEnnMmStatusMapped,
    kEnnMmStatusFreed,
    kEnnMmStatusMax,
};

enum class EnnMmType {
    kEnnMmTypeNone,
    kEnnMmTypeIon,
    kEnnMmTypeExternalIon,
    kEnnMmTypeCloned,
    kEnnMmTypeHeap,
    kEnnMmTypeExternalMax,
};

class EnnMemoryAllocator;

/* data structures */
// TODO(hoon98.choi, TBD): Refactoring EnnBufferCore to make class, read only const member getters
// TODO(hoon98.choi, 6/11): Change EnnBufferCore to more clearer name
class EnnBufferCore {
 public:
    using Ptr = std::shared_ptr<EnnBufferCore>;

 public:
    /* NOTE(hoon98.choi): ION_FLAG_CACHED is default. */
    EnnBufferCore()
        : va(nullptr),
          size(0),
          offset(0),
          magic(0),
          type(EnnMmType::kEnnMmTypeNone),
          fd(ENN_MM_FD_NOT_DEFINED),
          base_va(nullptr),
          cache_flag(ION_FLAG_CACHED),
          status(EnnBufferStatus::kEnnMmStatusUnMapped) {}

    EnnBufferCore(void *va_, enn::EnnMmType type_, uint32_t size_, int32_t fd_, uint32_t cache_flag_ = ION_FLAG_CACHED,
                  void *base_va_ = nullptr, uint32_t offset_ = 0)
        : va(va_), size(size_), offset(offset_), magic(0), type(type_), fd(fd_), base_va(va_), cache_flag(cache_flag_),
          status(EnnBufferStatus::kEnnMmStatusUnMapped) {
        ENN_UNUSED(base_va_);
        ENN_MEM_PRINT("construct.. va(%p), size(%d)\n", va, size);
        show();
    }

    /* heap */
    EnnBufferCore(void *va_, uint32_t size_, std::shared_ptr<char> s_va_, uint32_t offset_ = 0)
        : va(va_),
          size(size_),
          offset(offset_),
          magic(0),
          type(enn::EnnMmType::kEnnMmTypeHeap),
          fd(ENN_MM_FD_NOT_DEFINED),
          s_va(s_va_),
          status(EnnBufferStatus::kEnnMmStatusMapped) {}

    ~EnnBufferCore() {
        va = nullptr,
        size = 0,
        offset = 0,
        base_va = nullptr,
        status = enn::EnnBufferStatus::kEnnMmStatusFreed;
#ifdef __ANDROID__
        if (ntv_handle) {
            native_handle_close(ntv_handle);
            native_handle_delete(ntv_handle);
        }
#endif
    }

    /* debug */
    void show() {
        ENN_MEM_PRINT(
            " - Memory (%p) : va(%p), sz(%d), off(%d), magic(%X), type(%d), fd(%d), c_flag(%d), status(%d), "
            "return_ptr(%p)\n",
            this, va, size, offset, magic, (int)type, fd, cache_flag, (int)status, return_ptr());
    }

    void* return_ptr() { return reinterpret_cast<void *>(&va); }

    void *va = nullptr;
    uint32_t size = 0;
    uint32_t offset;
    uint32_t magic;       // check integrity. size, offset included?
    enn::EnnMmType type;  // ION? heap? external? partial?
    int32_t fd;           // ion fd, not allowed ashmem fd or something like that
    void *base_va = nullptr;         // further using
    uint32_t cache_flag;  // if offset > 0, should use cache_disabled
    std::shared_ptr<char> s_va;  // to maintain ref_cnt for heap
    EnnBufferStatus status = EnnBufferStatus::kEnnMmStatusFreed;

#ifdef __ANDROID__
    native_handle_t* get_native_handle() { return ntv_handle; }
    native_handle_t *ntv_handle = nullptr;

    EnnReturn SetNativeHandle(int32_t alloc_fd) {
        this->ntv_handle = native_handle_create(1, 0);
        this->ntv_handle->numFds = 1;
        this->ntv_handle->data[0] = alloc_fd;

        return ENN_RET_SUCCESS;
    }

    EnnReturn SetNativeHandle(const native_handle_t *ntv_handle_) {
        this->ntv_handle = native_handle_clone(ntv_handle_);
        fd = this->ntv_handle->data[0];

        return ENN_RET_SUCCESS;
    }
#endif
};

};  // namespace enn

#endif  // SRC_COMMON_INCLUDE_ENN_MEMORY_MANAGER_TYPE_H_
