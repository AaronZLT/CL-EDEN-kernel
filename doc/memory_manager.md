---
Title: Memory Manager
Output: pdf_document
author: Hoon Choi
Date: 2021. 1. 6
Reference: http://pdf.plantuml.net/PlantUML_Language_Reference_Guide_ko.pdf
---

### Memory Manager
```plantUML

set namespaceSeparator ::

top to bottom direction

Enn::EnnMemoryManager -> Enn::EnnMemoryAllocator
Enn::EnnMemoryAllocator -- Enn::EnnBufferCore
Enn::EnnMemoryManager - Enn::EnnBufferCore


Class Enn::EnnMemoryManager {
    +EnnMemoryManager();
    +~EnnMemoryManager();

    +std::shared_ptr<EnnBufferCore> CreateMemory(int32_t size, enn::EnnMmType type, uint32_t flag);
    +EnnReturn DeleteMemory(std::shared_ptr<EnnBufferCore> buffer);
    +std::shared_ptr<EnnBufferCore> CreateMemoryFromFd(uint32_t fd, uint32_t size, uint32_t options = 0);
    +std::shared_ptr<EnnBufferCore> CreateMemoryFromFdWithOffset(uint32_t fd, uint32_t size

    +EnnReturn GetMemoryMeta(std::shared_ptr<EnnBufferCore> buffer, enn::EnnBufferMetaType type, uint32_t *out);
    +EnnReturn ShowMemoryPool(void);

    -EnnMemoryAllocator *GetAllocator()
    -EnnReturn GenerateMagic(const void *va, const uint32_t size, const uint32_t offset, uint32_t *out);
    -std::vector<std::shared_ptr<EnnBufferCore>> memory_pool;
    -std::mutex mm_mutex;
    -std::unique_ptr<EnnMemoryAllocator> emm_allocator;
}

note top of Enn::EnnMemoryManager:This class can be directly called from User API functions\nMemoryManager returns EnnBufferCore,\nAPI will return user-type (part of EnnBufferCore)


class Enn::EnnBufferCore {
    +EnnBufferCore();
    +EnnBufferCore(void *va, enn::EnnMmType type, uint32_t size, uint32_t fd, uint32_t offset uint32_t cache_flag)
    +~EnnBufferCore();
    +void show();
    +void* return_ptr();

    +void *va;
    +uint32_t size;
    +uint32_t offset;
    +uint32_t magic;
    +enn::EnnMmType type;
    +uint32_t fd;
    +uint32_t fd2;
    +uint32_t cache_flag;
    +EnnBufferStatus status;
}

Enn::EnnMemoryAllocator <|-- Enn::EnnMemoryAllocatorDeviceBuffer
Enn::EnnMemoryAllocator <|-- Enn::EnnMemoryAllocatorMalloc

Interface Enn::EnnMemoryAllocator {
    +EnnMemoryAllocator()
    +virtual ~EnnMemoryAllocator()
    +virtual std::shared_ptr<EnnBufferCore> CreateMemory(int32_t size, enn::EnnMmType type, uint32_t flag) = 0
    +virtual EnnReturn DeleteMemory(std::shared_ptr<EnnBufferCore> buffer) = 0
}

note bottom of Enn::EnnMemoryAllocatorMalloc: Not implemented yet.

class Enn::EnnMemoryAllocatorMalloc {
}

class Enn::EnnMemoryAllocatorDeviceBuffer {
    +EnnMemoryAllocatorDeviceBuffer();
    +virtual ~EnnMemoryAllocatorDeviceBuffer();
    +std::shared_ptr<EnnBufferCore> CreateMemory(int32_t size, enn::EnnMmType type, uint32_t flag);
    +EnnReturn DeleteMemory(std::shared_ptr<EnnBufferCore> buffer);
    -EnnReturn EnnMemoryDeviceBufferOpen();
    -EnnReturn EnnMemoryDeviceBufferClose();
    -int32_t _ion_client_fd;
    -int32_t _heap_mask;
    -std::mutex mma_mutex;
}

note bottom of EnnBuffer: public data structure, part of EnnBufferCore

Enn::EnnBufferCore <-- EnnBuffer

class EnnBuffer {
    +void *va;
    +uint32_t size;  // requested size
    +uint32_t offset;
}

```

### Tests

#### enn_memory_manager_test.cc
  - memory test with memory manager directly
  - ENN_GT_UNITTEST_MEMORY.
    * memory_test_1_emm_test
    * memory_test_2_emm_test
    * memory_test_3_emm_test_fd
    * memory_test_4_emm_test_fd_offset


#### enn_api_test.cc
  - memory test with user API
  - ENN_GT_UNIT_TEST_API.
    * mem_test_1_memory_allocation_test
    * mem_test_2_memory_allocation_test
    * mem_test_3_memory_import_test
    * mem_test_3_memory_import_test2