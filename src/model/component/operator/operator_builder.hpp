#ifndef SRC_MODEL_COMPONENT_OPERATOR_OPERATOR_OPERATOR_BUILDER_HPP_
#define SRC_MODEL_COMPONENT_OPERATOR_OPERATOR_OPERATOR_BUILDER_HPP_

#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "model/component/operator/operator.hpp"
#include "model/component/operator/ioperator_builder.hpp"


namespace enn {
namespace model {
namespace component {


// Keep using CRTP idiom for reusability and fluent interface if you want to inherit another builder from this.
// Make this builder class template class like as TensorBuilder
class OperatorBuilder : public IOperatorBuilder<OperatorBuilder, Operator> {
 public:
    OperatorBuilder()
        : IOperatorBuilder{Operator::Ptr(new Operator)} {}

    // It can accept argument as type of both rvalue and lvalue of shared_ptr or unique_ptr.
    template <typename T>
    explicit OperatorBuilder(T&& opr)
        : IOperatorBuilder{std::forward<T>(opr)} {}

    ~OperatorBuilder() = default;

    auto create() {
        Operator::Ptr operator_obj(new Operator);
        this->operator_.swap(operator_obj);
        return std::move(operator_obj);
    }

    OperatorBuilder& set_id(uint64_t id) {
        operator_->id_ = id;
        return *this;
    }

    OperatorBuilder& set_code(TFlite::BuiltinOperator code) {
        this->operator_->code_ = code;
        return *this;
    }

    template<typename T>
    OperatorBuilder& set_option(T&& name, const void* addr, size_t size, TFlite::BuiltinOptions num) {
        this->operator_->option_ = {std::forward<T>(name), addr, size, num};
        return *this;
    }

    template<typename T>
    OperatorBuilder& add_binary(T&& name, const void* addr, size_t size, Accelerator accelerator = Accelerator::NONE) {
        this->operator_->binaries_.push_back({std::forward<T>(name), addr, size, accelerator});
        return *this;
    }

    template<typename T>
    OperatorBuilder& add_binary(T&& name, int32_t fd, const void* addr, size_t size,
        Accelerator accelerator = Accelerator::NONE) {
        this->operator_->binaries_.push_back({std::forward<T>(name), fd, addr, size, accelerator});
        return *this;
    }

    template<typename T>
    OperatorBuilder& add_binary(T&& name, int32_t fd, const void* addr, size_t size,
        int32_t offset, Accelerator accelerator = Accelerator::NONE) {
        this->operator_->binaries_.push_back({std::forward<T>(name), fd, addr, size, offset, accelerator});
        return *this;
    }

    OperatorBuilder& set_in_pixel_format(uint32_t in_pixel_format) {
        this->operator_->in_pixel_format_ = in_pixel_format;
        return *this;
    }

    OperatorBuilder& set_buffer_shared(bool buffer_shared) {
        this->operator_->buffer_shared_ = buffer_shared;
        return *this;
    }

    OperatorBuilder& set_ifm_bound(bool ifm_bound) {
        this->operator_->ifm_bound_ = ifm_bound;
        return *this;
    }

    OperatorBuilder& set_ofm_bound(bool ofm_bound) {
        this->operator_->ofm_bound_ = ofm_bound;
        return *this;
    }

    OperatorBuilder& set_dsp_async_exec(bool dsp_async_exec) {
        this->operator_->dsp_async_exec_ = dsp_async_exec;
        return *this;
    }

    //  This is for CGO, library name of CGO need to be delivered to UD,
    OperatorBuilder& set_lib_names(std::vector<std::string>& lib_names) {
        this->operator_->lib_names_ = lib_names;
        return *this;
    }
};


};  // namespace component
};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_BUILDER_GRAPH_VERTEX_BUILDER_HPP_