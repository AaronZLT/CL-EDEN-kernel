#ifndef SRC_MODE_COMPONENT_OPERAND_FEATURE_MAP_FEATURE_MAP_HPP_
#define SRC_MODE_COMPONENT_OPERAND_FEATURE_MAP_FEATURE_MAP_HPP_

#include <vector>
#include <memory>
#include <string>

#include "model/component/tensor/tensor.hpp"
#include "model/memory/indexed_buffer.hpp"

namespace enn {
namespace model {
namespace component {


// TODO(yc18.cho, TBD) : The name will be changed to PlaceHolder
class FeatureMap : public Tensor {
 private:
    using IndexedBuffer = enn::model::memory::IndexedBuffer;

 public:
    using Ptr = std::shared_ptr<FeatureMap>;

    enum class Type {
        UNDEFINED,        // Not set which would cause warning
        SUBGRAPH_INPUT,   // Tensor corresponding to an Input of SubGraph
        SUBGRAPH_OUTPUT,  // Tensor corresponding to an Output of SubGraph
        INTERMEDIATE,     // Tensor holding the intermediate computation result from certain Operator
    };

 public:
    virtual ~FeatureMap() = default;

    // FeatureMap's buffer is variable every execution.
    bool is_const() const override { return false; }
    std::shared_ptr<IOperator> prev() const override { return prev_operator_.lock(); }

    size_t get_buffer_size() const { return buffer_->get_size(); }
    int32_t get_buffer_index() const { return buffer_->get_index(); }
    Type get_type() const { return type_; }

 private:
    friend class FeatureMapBuilder;          // Declare Builder class
    FeatureMap()                             // FeatureMapBuilder only can create FeatureMap object.
        : buffer_{std::make_shared<IndexedBuffer>()} {}

    // Smart pointer to IOperator should be weak_ptr to avoid circular reference.
    std::weak_ptr<IOperator>       prev_operator_;  // previous operator
    IndexedBuffer::Ptr buffer_;       // has index pointing to certain Buffer in BufferTable.
    Type type_ = Type::UNDEFINED;
};


};  // namespace component
};  // namespace model
};  // namespace enn

#endif  // SRC_MODE_COMPONENT_OPERAND_FEATURE_MAP_FEATURE_MAP_HPP_