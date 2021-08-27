#ifndef SRC_MODEL_MEMORY_ALLOCATED_BUFFER_HPP_
#define SRC_MODEL_MEMORY_ALLOCATED_BUFFER_HPP_

#include <cstdint>
#include <memory>
#include <map>

#include "model/memory/buffer.hpp"

namespace enn {
namespace model {
namespace memory {


// AllocatedBuffer has address or fd of memory allocated.
class AllocatedBuffer : public Buffer {
 public:
    using Ptr = std::shared_ptr<AllocatedBuffer>;

 public:
    explicit AllocatedBuffer() = default;
    virtual ~AllocatedBuffer() = default;

    AllocatedBuffer(size_t size)
        : Buffer{size}, addr_{nullptr}, fd_{0}, offset_{0} {}

    AllocatedBuffer(const void* addr, size_t size)
        : Buffer{size}, addr_{addr}, fd_{0}, offset_{0} {}

    AllocatedBuffer(int32_t fd, size_t size)
        : Buffer{size}, addr_{nullptr}, fd_{fd}, offset_{0} {}

    AllocatedBuffer(int32_t fd, const void* addr, size_t size)
        : Buffer{size}, addr_{addr}, fd_{fd}, offset_{0} {}

    AllocatedBuffer(int32_t fd, const void* addr, size_t size, int32_t offset)
        : Buffer{size}, addr_{addr}, fd_{fd}, offset_{offset} {}

    void set_addr(const void* addr) { addr_ = addr; }
    void set_fd(int32_t fd) { fd_ = fd; }
    void set_offset(int32_t offset) { offset_ = offset; }

    const void* get_addr() const { return addr_; }
    int32_t get_fd() const { return fd_; }
    int32_t get_offset() const { return offset_; }


 private:
    const void* addr_;
    int32_t fd_;
    int32_t offset_;
};


};  // namespace memory
};  // namespaec model
};  // namespace enn

#endif  // SRC_MODEL_MEMORY_ALLOCATED_BUFFER_HPP_
