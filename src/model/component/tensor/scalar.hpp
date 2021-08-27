#ifndef SRC_MODE_COMPONENT_TENSOR_SCALAR_HPP_
#define SRC_MODE_COMPONENT_TENSOR_SCALAR_HPP_

#include <memory>

#include "model/component/tensor/tensor.hpp"
#include "model/memory/allocated_buffer.hpp"
#include "model/memory/indexed_buffer.hpp"

namespace enn {
namespace model {
namespace component {


class Scalar : public Tensor {
 public:
    using Ptr = std::shared_ptr<Scalar>;

 private:
    using AllocatedBuffer = enn::model::memory::AllocatedBuffer;
    using IndexedBuffer = enn::model::memory::IndexedBuffer;

 public:
    virtual ~Scalar() = default;

    bool is_const() const override { return true; }
    std::shared_ptr<IOperator> prev() const override { return nullptr; }

    size_t get_indexed_buffer_size() const { return indexed_buffer_->get_size(); }
    int32_t get_indexed_buffer_index() const { return indexed_buffer_->get_index(); }

    const void* get_default_buffer_addr() const { return default_buffer_->get_addr(); }
    size_t get_default_buffer_size() const { return default_buffer_->get_size(); }
    int32_t get_default_buffer_fd() const { return default_buffer_->get_fd(); }
    int32_t get_default_buffer_offset() const { return default_buffer_->get_offset(); }

 private:
    friend class ScalarBuilder;        // Declare Builder class, Scalar Builder only can create Scalar object.
    Scalar() : default_buffer_{std::make_shared<AllocatedBuffer>()}, indexed_buffer_{std::make_shared<IndexedBuffer>()} {}

    AllocatedBuffer::Ptr default_buffer_;       // has index pointing to certain Buffer in BufferTable.
    IndexedBuffer::Ptr indexed_buffer_;       // has index pointing to certain Buffer in BufferTable.
};


};  // namespace component
};  // namespace model
};  // namespace enn

#endif  // SRC_MODE_COMPONENT_TENSOR_SCALAR_HPP_
