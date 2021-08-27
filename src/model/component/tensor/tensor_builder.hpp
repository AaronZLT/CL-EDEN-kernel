#ifndef SRC_MODEL_COMPONENT_OPERAND_TENSOR_TENSOR_BUILDER_HPP_
#define SRC_MODEL_COMPONENT_OPERAND_TENSOR_TENSOR_BUILDER_HPP_

#include <memory>
#include <utility>

#include "model/component/tensor/tensor.hpp"
#include "model/schema/schema_nnc.h"

namespace enn {
namespace model {
namespace component {


// Uses CRTP idioms to reuse base class builder's methods and enable fluent interface.
//  Certain builder that is responsible for building a object of subclass derived from Tensor class
//  should inherit also TensorBuilder.
//  And sub class's builder should propagate its target tensor's type to it.
template <typename ConcreteBuilder, typename TensorType>
class TensorBuilder {

 public:
    // It can accept argument as type of both rvalue and lvalue of shared_ptr or unique_ptr.
    template <typename Ts>
    explicit TensorBuilder(Ts&& tensor)
        : tensor_{std::forward<Ts>(tensor)} {}

    ~TensorBuilder() = default;

    template <typename Ts>
    ConcreteBuilder& operator=(Ts&& tensor) {
        tensor_ = std::forward<Ts>(tensor);
        return static_cast<ConcreteBuilder&>(*this);
    }

    auto create() {
        return static_cast<ConcreteBuilder*>(this)->create();
    }

    auto get() {
        return tensor_;
    }

    ConcreteBuilder& clear() {
        tensor_.reset();
        return static_cast<ConcreteBuilder&>(*this);
    }

    ConcreteBuilder& set_id(int32_t id) {
        tensor_->id_ = id;
        return static_cast<ConcreteBuilder&>(*this);
    }

    template<typename N>
    ConcreteBuilder& set_name(N&& name) {
        tensor_->name_ = std::forward<N>(name);
        return static_cast<ConcreteBuilder&>(*this);
    }

    ConcreteBuilder& set_data_type(TFlite::TensorType data_type) {
        tensor_->data_type_ = data_type;
        return static_cast<ConcreteBuilder&>(*this);
    }

    template<typename S>
    ConcreteBuilder& set_shape(S&& shape) {
        tensor_->shape_ = std::forward<S>(shape);
        return static_cast<ConcreteBuilder&>(*this);
    }

    template <typename Op>
    ConcreteBuilder& add_next_operator(Op&& next_operator) {
        tensor_->next_operators_.push_back(std::forward<Op>(next_operator));
        return static_cast<ConcreteBuilder&>(*this);
    }

    ConcreteBuilder& set_quantization_parameters(const TFlite::QuantizationParameters* quantization_parameters) {
        tensor_->quantization_parameters_ = quantization_parameters;
        return static_cast<ConcreteBuilder&>(*this);
    }

    ConcreteBuilder& set_symm_per_channel_quant_parameters(const TFlite::SymmPerChannelQuantParamters* param) {
        tensor_->symm_per_channel_quant_parameters_ = param;
        return static_cast<ConcreteBuilder&>(*this);
    }

 protected:
    typename TensorType::Ptr tensor_;
};


};  // namespace component
};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_COMPONENT_OPERAND_TENSOR_TENSOR_BUILDER_HPP_