#ifndef SRC_MODEL_MEMORY_BUFFER_TABLE_HPP_
#define SRC_MODEL_MEMORY_BUFFER_TABLE_HPP_

#include <cstdint>
#include <memory>
#include <map>

#include "model/memory/indexed_buffer.hpp"
#include "model/memory/allocated_buffer.hpp"


namespace enn {
namespace model {
namespace memory {


// Every ExecutableModel has a BufferTable object.
//  return AllocatedBuffer object that real memory information as taking index of IndexedBuffer.
class BufferTable {
 private:
    // Key is index from IndexedBuffer and value is AllocatedBuffer.
    std::map<int32_t, AllocatedBuffer> buffer_map_;

 public:
    using Ptr = std::shared_ptr<BufferTable>;

    const auto& operator[](IndexedBuffer indexed_buffer) const{
        return buffer_map_.at(indexed_buffer.get_index());
    }

    const auto& operator[](int32_t index) const {
        return buffer_map_.at(index);
    }

    bool exist(int32_t index) const { return buffer_map_.find(index) != buffer_map_.end(); }

    void add(int32_t index, const void* addr, size_t size) {
        buffer_map_.insert(std::pair<int32_t, AllocatedBuffer>(index, AllocatedBuffer{addr, size}));
    }

    void add(int32_t index, int32_t fd, const void* addr, size_t size) {
        buffer_map_.insert(std::pair<int32_t, AllocatedBuffer>(index, AllocatedBuffer{fd, addr, size}));
    }
};


};  // namespace memory
};  // namespaec model
};  // namespace enn

#endif  // SRC_MODEL_MEMORY_BUFFER_TABLE_HPP_
