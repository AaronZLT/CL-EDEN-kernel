#ifndef SRC_MODEL_GRAPH_ELEMENTS_VERTEX_SUBGRAPH_HPP_
#define SRC_MODEL_GRAPH_ELEMENTS_VERTEX_SUBGRAPH_HPP_

#include <memory>
#include <vector>
#include <mutex>
#include <functional>

#include "model/component/operator/ioperator.hpp"
#include "runtime/dispatch/dispatchable.hpp"
#include "model/attribute.hpp"
#include "common/identifier_decorator.hpp"

namespace enn {
namespace model {
namespace component {

using namespace enn::identifier;

class OperatorList : public IOperator, public runtime::dispatch::Dispatchable {
 private:
    using UniqueIDDeco = IdentifierDecorator<FullIDType, uint16_t, 8>;

 public:
    using Ptr = std::shared_ptr<OperatorList>;
    using ID = IdentifierBase<FullIDType>;

 public:
    class UniqueID : public UniqueIDDeco {
     public:
        // Bring members from base temlate class for client to access directly them.
        using UniqueIDDeco::Type;
        using UniqueIDDeco::Max;
        using UniqueIDDeco::Offset;
        using UniqueIDDeco::Mask;

        explicit UniqueID(const IdentifierBase<FullIDType>& base_id)
            : UniqueIDDeco(base_id) {}

        // This constructor would be used for ExecutableOperatorList's ID of which all fields are set.
        UniqueID(IdentifierDecorator::Type operator_list_id,
                            const IdentifierBase<FullIDType>& base_id)
            : UniqueIDDeco(operator_list_id, base_id) {}
    };

 public:
    ~OperatorList() {
        ENN_DBG_COUT << "An OperatorList(ID: 0x" << *id_ << ") is released." << std::endl;
    }

    const ID& get_id() const {
        return *id_;
    }

    uint32_t get_size() const {
        return vertices_.size();
    }

    std::vector<IOperator::Ptr>::const_iterator begin() const {
        return vertices_.cbegin();
    }

    std::vector<IOperator::Ptr>::const_iterator end() const {
        return vertices_.cend();
    }

    const enn::model::Accelerator& get_accelerator() const override {
        return accelerator_;
    }

    const Attribute& get_attribute() const {
        return *attribute_;
    }

    uint32_t get_preset_id() const {
        return preset_id_;
    }

    uint32_t get_pref_mode() const {
        return pref_mode_;
    }

    uint32_t get_target_latency() const {
        return target_latency_;
    }

    uint32_t get_tile_num() const {
        return tile_num_;
    }

    uint32_t get_core_affinity() const {
        return core_affinity_;
    }

    uint32_t get_priority() const {
        return priority_;
    }

 private:
    friend class OperatorListBuilder;  // delcare Builder class as friend, which only has creation right.
    // OperatorListBuilder only can create OperatorList object
    explicit OperatorList(const IdentifierBase<FullIDType>& base_id)
        : id_(std::make_unique<UniqueID>(base_id)),
          preset_id_(0), pref_mode_(0), target_latency_(0), tile_num_(1), core_affinity_(0), priority_(0) {
            ENN_DBG_COUT << "An OperatorList(ID: 0x" << *id_ << ") is created." << std::endl;
    }

    // Prevent rvalue binding
    explicit OperatorList(const IdentifierBase<FullIDType>&& base_id) = delete;

 private:
    ID::UPtr id_;  // identifier for OperatorList
    std::vector<IOperator::Ptr> vertices_;  // vector for composition of Operator or OperatorList
    Attribute::Ptr attribute_;

    // Preferences From Client
    uint32_t preset_id_;
    uint32_t pref_mode_;         // for EDEN backward compatibility
    uint32_t target_latency_;    // for DVFS hint
    uint32_t tile_num_;         // for batch processing hint
    uint32_t core_affinity_;
    uint32_t priority_;
};


};  // namespace component
};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_GRAPH_ELEMENTS_VERTEX_SUBGRAPH_HPP_