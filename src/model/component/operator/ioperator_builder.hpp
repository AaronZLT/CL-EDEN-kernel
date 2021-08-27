#ifndef SRC_MODEL_COMPONENT_OPERATOR_IOPERATOR_BUILDER_HPP_
#define SRC_MODEL_COMPONENT_OPERATOR_IOPERATOR_BUILDER_HPP_

#include <memory>
#include <string>
#include <utility>

#include "model/component/operator/ioperator.hpp"
#include "model/types.hpp"
#include "model/schema/schema_nnc.h"


namespace enn {
namespace model {
namespace component {


// Uses CRTP idioms to reuse base class builder's methods and enable fluent interface.
//  Certain builder that is responsible for building a object of subclass derived from  class
//  should inherit also TensorBuilder.
//  And sub class's builder should propagate its target tensor's type to it.
template <typename ConcreteBuilder, typename OperatorType>
class IOperatorBuilder {
 public:
    // It can accept argument as type of both rvalue and lvalue of shared_ptr or unique_ptr.
    template <typename Ot>
    explicit IOperatorBuilder(Ot&& opr)
        : operator_{std::forward<Ot>(opr)} {}

    virtual ~IOperatorBuilder() = default;

    template <typename Ot>
    ConcreteBuilder& operator=(Ot&& opr) {
        operator_ = std::forward<Ot>(opr);
        return static_cast<ConcreteBuilder&>(*this);
    }

    auto create() {
        return static_cast<ConcreteBuilder*>(this)->create();
    }

    auto get() {
        return operator_;
    }

    ConcreteBuilder& clear() {
        operator_.reset();
        return static_cast<ConcreteBuilder&>(*this);
    }

    template<typename T>
    ConcreteBuilder& set_name(T&& name) {
        operator_->name_ = std::forward<T>(name);
        return static_cast<ConcreteBuilder&>(*this);
    }

    ConcreteBuilder& set_accelerator(enn::model::Accelerator accelerator) {
        operator_->accelerator_ = accelerator;
        return static_cast<ConcreteBuilder&>(*this);
    }

    template <typename T>
    ConcreteBuilder& add_in_tensor(T&& in_tensor) {
        operator_->in_tensor_list_.push_back(std::forward<T>(in_tensor));
        return static_cast<ConcreteBuilder&>(*this);
    }

    template <typename T>
    ConcreteBuilder& add_out_tensor(T&& out_tensor) {
        operator_->out_tensor_list_.push_back(std::forward<T>(out_tensor));
        return static_cast<ConcreteBuilder&>(*this);
    }

 protected:
    typename OperatorType::Ptr operator_;
};


};  // namespace component
};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_BUILDER_GRAPH_VERTEX_BUILDER_HPP_