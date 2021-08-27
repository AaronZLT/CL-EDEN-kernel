#ifndef SRC_MODE_COMPONENT_OPERAND_PARAMETER_PARAMETER_HPP_
#define SRC_MODE_COMPONENT_OPERAND_PARAMETER_PARAMETER_HPP_

#include <vector>
#include <memory>
#include <string>

#include "model/component/tensor/tensor.hpp"
#include "model/memory/allocated_buffer.hpp"

namespace enn {
namespace model {
namespace component {


class Parameter : public Tensor {
 public:
    using Ptr = std::shared_ptr<Parameter>;

 private:
    using AllocatedBuffer = enn::model::memory::AllocatedBuffer;

 public:
    virtual ~Parameter() = default;

    // Parameter's buffer is constant after being loaded.
    bool is_const() const override { return true; }
    std::shared_ptr<IOperator> prev() const override { return nullptr; }

    const void* get_buffer_addr() const { return buffer_->get_addr(); }
    size_t get_buffer_size() const { return buffer_->get_size(); }
    int32_t get_buffer_fd() const { return buffer_->get_fd(); }
    int32_t get_buffer_offset() const { return buffer_->get_offset(); }

 private:
    friend class ParameterBuilder;          // Declare Builder class
    Parameter()                             // ParameterBuilder only can create Parameter object.
        : buffer_{std::make_shared<AllocatedBuffer>()} {}

    AllocatedBuffer::Ptr buffer_;       // has index pointing to certain Buffer in BufferTable.
};


};  // namespace component
};  // namespace model
};  // namespace enn

#endif  // SRC_MODE_COMPONENT_OPERAND_PARAMETER_PARAMETER_HPP_