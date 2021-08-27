#ifndef SRC_MODEL_MEMORY_INDEXED_BUFFER_HPP_
#define SRC_MODEL_MEMORY_INDEXED_BUFFER_HPP_

#include <cstdint>
#include <memory>
#include <map>

#include "model/memory/buffer.hpp"

namespace enn {
namespace model {
namespace memory {


// IndexedBuffer has index with which can find AllocatedBuffer via BufferTable.
class IndexedBuffer : public Buffer {
 public:
    using Ptr = std::shared_ptr<IndexedBuffer>;

 public:
    explicit IndexedBuffer() : Buffer{0}, index_{0} {}
    virtual ~IndexedBuffer() = default;

    IndexedBuffer(size_t size)
        : Buffer{size}, index_{0} {}

    IndexedBuffer(int32_t index, size_t size)
        : Buffer{size}, index_{index} {}

    void set_index(int32_t index) { index_ = index; }

    int32_t get_index () const { return index_; }

 private:
    int32_t index_;
};


};  // namespace memory
};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_MEMORY_INDEXED_BUFFER_HPP_
