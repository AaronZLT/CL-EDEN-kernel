#ifndef SRC_MODEL_COMPONENT_OPERATOR_OPERATOR_OPERATOR_LIST_BUILDER_HPP_
#define SRC_MODEL_COMPONENT_OPERATOR_OPERATOR_OPERATOR_LIST_BUILDER_HPP_

#include <memory>
#include <string>
#include <utility>

#include "model/component/operator/operator_list.hpp"
#include "model/component/operator/ioperator_builder.hpp"
#include "common/enn_debug.h"


namespace enn {
namespace model {
namespace component {


// Keep using CRTP idiom for reusability and fluent interface if you want to inherit another builder from this.
// Make this builder class template class like as TensorBuilder
class OperatorListBuilder : public IOperatorBuilder<OperatorListBuilder, OperatorList> {
 public:
    OperatorListBuilder()
        : IOperatorBuilder{nullptr} {}

    // It can accept argument as type of both rvalue and lvalue of shared_ptr or unique_ptr.
    template <typename T>
    explicit OperatorListBuilder(T&& opr_list)
        : IOperatorBuilder{std::forward<T>(opr_list)} {}

    ~OperatorListBuilder() = default;

    auto create() {
        print();
        return std::move(this->operator_);
    }

    // Initialize method that should be called at once before setting object.
    // Pass the Model::UniqueID in order to create and set the OperatorList::UniqueID.
    OperatorListBuilder& build(const enn::identifier::IdentifierBase<identifier::FullIDType>& base_id) {
        OperatorList::Ptr new_operator_list(new OperatorList{base_id});
        this->operator_ = new_operator_list;
        return *this;
    }

    template<typename T>
    OperatorListBuilder& set_attribute(T&& attr) {
        this->operator_->attribute_ = std::forward<T>(attr);
        return *this;
    }

    template<typename T>
    OperatorListBuilder& add_operator(T&& op) {
        this->operator_->vertices_.push_back(std::forward<T>(op));
        return *this;
    }

    OperatorListBuilder& set_preset_id(uint32_t preset_id) {
        this->operator_->preset_id_ = preset_id;
        return *this;
    }

    OperatorListBuilder& set_pref_mode(uint32_t pref_mode) {
        this->operator_->pref_mode_ = pref_mode;
        return *this;
    }

    OperatorListBuilder& set_target_latency(uint32_t target_latency) {
        this->operator_->target_latency_ = target_latency;
        return *this;
    }

    OperatorListBuilder& set_tile_num(uint32_t tile_num) {
        this->operator_->tile_num_ = tile_num;
        return *this;
    }

    OperatorListBuilder& set_core_affinity(uint32_t core_affinity) {
        this->operator_->core_affinity_ = core_affinity;
        return *this;
    }

    OperatorListBuilder& set_priority(uint32_t priority) {
        this->operator_->priority_ = priority;
        return *this;
    }
    void print() {
        if (this->operator_->vertices_.size() != 0) {
            ENN_DBG_COUT << "=======================================" << std::endl;
            ENN_DBG_COUT << "Therer is/are " << this->operator_->vertices_.size() << " op in operator list" << std::endl;
            int count = 0;
            for (auto& op : this->operator_->vertices_) {
                ENN_DBG_COUT << "[" << count << "] : Name=" << op->get_name() << std::endl;
                count++;
            }
            ENN_DBG_COUT << "op list for hardware [ " << (int)this->operator_->get_accelerator()
                         << "] Created." << std::endl;
            ENN_DBG_COUT << "=======================================" << std::endl;
        }
    }
};


};  // namespace component
};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_BUILDER_GRAPH_VERTEX_BUILDER_HPP_