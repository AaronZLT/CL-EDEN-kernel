#ifndef SRC_MODEL_COMPONENT_TENSOR_PARAMETER_BUILDER_HPP_
#define SRC_MODEL_COMPONENT_TENSOR_PARAMETER_BUILDER_HPP_

#include <memory>
#include <utility>

#include "model/component/tensor/parameter.hpp"
#include "model/component/tensor/tensor_builder.hpp"
#include "model/schema/schema_nnc.h"

namespace enn {
namespace model {
namespace component {


class ParameterBuilder : public TensorBuilder<ParameterBuilder, Parameter> {

 public:
    ParameterBuilder()
        : TensorBuilder{std::shared_ptr<Parameter>(new Parameter)} {}

    // It can accept argument as type of both rvalue and lvalue of shared_ptr or unique_ptr.
    template <typename T>
    explicit ParameterBuilder(T&& fm)
        : TensorBuilder<ParameterBuilder, Parameter>{std::forward<T>(fm)} {}

    ~ParameterBuilder() = default;

    auto create() {
        std::shared_ptr<Parameter> parameter_obj(new Parameter);
        this->tensor_.swap(parameter_obj);
        return std::move(parameter_obj);
    }

    ParameterBuilder& set_buffer_addr(const void* addr) {
        this->tensor_->buffer_->set_addr(addr);
        return *this;
    }

    ParameterBuilder& set_buffer_size(size_t size) {
        this->tensor_->buffer_->set_size(size);
        return *this;
    }

    ParameterBuilder& set_buffer_fd(int32_t fd) {
        this->tensor_->buffer_->set_fd(fd);
        return *this;
    }

    ParameterBuilder& set_buffer_offset(int32_t offset) {
        this->tensor_->buffer_->set_offset(offset);
        return *this;
    }
};


};  // namespace component
};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_COMPONENT_TENSOR_PARAMETER_BUILDER_HPP_