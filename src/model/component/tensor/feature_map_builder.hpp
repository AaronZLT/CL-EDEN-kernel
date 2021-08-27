#ifndef SRC_MODEL_COMPONENT_OPERAND_FEATURE_MAP_FEATURE_MAP_BUILDER_HPP_
#define SRC_MODEL_COMPONENT_OPERAND_FEATURE_MAP_FEATURE_MAP_BUILDER_HPP_

#include <memory>
#include <utility>

#include "model/component/tensor/feature_map.hpp"
#include "model/component/tensor/tensor_builder.hpp"

namespace enn {
namespace model {
namespace component {


// Keep using CRTP idiom for reusability and fluent interface if you want to inherit another builder from this.
// Make this builder class template class like as TensorBuilder
class FeatureMapBuilder : public TensorBuilder<FeatureMapBuilder, FeatureMap> {

 public:
    FeatureMapBuilder()
        : TensorBuilder{FeatureMap::Ptr(new FeatureMap)} {}

    // It can accept argument as type of both rvalue and lvalue of shared_ptr or unique_ptr.
    template <typename T>
    explicit FeatureMapBuilder(T&& fm)
        : TensorBuilder{std::forward<T>(fm)} {}

    ~FeatureMapBuilder() = default;

    auto create() {
        FeatureMap::Ptr feature_map_obj(new FeatureMap);
        this->tensor_.swap(feature_map_obj);
        return std::move(feature_map_obj);
    }

    FeatureMapBuilder& set_buffer_size(size_t size) {
        this->tensor_->buffer_->set_size(size);
        return *this;
    }

    FeatureMapBuilder& set_buffer_index(int32_t index){
        this->tensor_->buffer_->set_index(index);
        return *this;
    }

    template <typename T>
    FeatureMapBuilder& set_prev_operator(T&& prev_operator) {
        this->tensor_->prev_operator_ = std::forward<T>(prev_operator);
        return *this;
    }

    FeatureMapBuilder& set_type(FeatureMap::Type type) {
        this->tensor_->type_ = type;
        return *this;
    }
};


};  // namespace component
};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_COMPONENT_OPERAND_FEATURE_MAP_FEATURE_MAP_BUILDER_HPP_