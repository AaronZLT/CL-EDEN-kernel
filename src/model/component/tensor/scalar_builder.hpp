#ifndef SRC_MODE_COMPONENT_TENSOR_SCALAR_BUILDER_HPP_
#define SRC_MODE_COMPONENT_TENSOR_SCALAR_BUILDER_HPP_

#include <memory>
#include <utility>

#include "model/component/tensor/scalar.hpp"
#include "model/component/tensor/tensor_builder.hpp"
#include "model/schema/schema_nnc.h"

namespace enn {
namespace model {
namespace component {


class ScalarBuilder : public TensorBuilder<ScalarBuilder, Scalar> {
 public:
    ScalarBuilder()
        : TensorBuilder{std::shared_ptr<Scalar>(new Scalar)} {}

    // It can accept argument as type of both rvalue and lvalue of shared_ptr or unique_ptr.
    template <typename T>
    explicit ScalarBuilder(T&& fm)
        : TensorBuilder<ScalarBuilder, Scalar>{std::forward<T>(fm)} {}

    ~ScalarBuilder() = default;

    auto create() {
        std::shared_ptr<Scalar> parameter_obj(new Scalar);
        this->tensor_.swap(parameter_obj);
        return std::move(parameter_obj);
    }

    ScalarBuilder& set_default_buffer_addr(const void* addr) {
        this->tensor_->default_buffer_->set_addr(addr);
        return *this;
    }

    ScalarBuilder& set_default_buffer_size(size_t size) {
        this->tensor_->default_buffer_->set_size(size);
        return *this;
    }

    ScalarBuilder& set_default_buffer_fd(int32_t fd) {
        this->tensor_->default_buffer_->set_fd(fd);
        return *this;
    }

    ScalarBuilder& set_default_buffer_offset(int32_t offset) {
        this->tensor_->default_buffer_->set_offset(offset);
        return *this;
    }

    ScalarBuilder& set_indexed_buffer_size(size_t size) {
        this->tensor_->indexed_buffer_->set_size(size);
        return *this;
    }

    ScalarBuilder& set_indexed_buffer_index(int32_t index) {
        this->tensor_->indexed_buffer_->set_index(index);
        return *this;
    }
};


};  // namespace component
};  // namespace model
};  // namespace enn

#endif  // SRC_MODE_COMPONENT_TENSOR_SCALAR_BUILDER_HPP_
