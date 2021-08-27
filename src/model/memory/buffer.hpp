#ifndef SRC_MODEL_MEMORY_BUFFER_HPP_
#define SRC_MODEL_MEMORY_BUFFER_HPP_

#include <cstdint>
#include <memory>
#include <map>

namespace enn {
namespace model {
namespace memory {


// Buffer class is for implementation inheritance into subclasses.
class Buffer {
 public:
    explicit Buffer() = default;
    virtual ~Buffer() = default;

    Buffer(size_t size)
        : size_(size) {}

    void set_size(size_t size) { size_ = size; }

    size_t get_size() const { return size_; }

 private:
    size_t size_;
};


};  // namespace memory
};  // namespaec model
};  // namespace enn

#endif  // SRC_MODEL_MEMORY_BUFFER_HPP_
