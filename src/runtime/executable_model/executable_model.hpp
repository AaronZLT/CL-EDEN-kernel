#ifndef SRC_RUNTIME_EXECUTABLE_MODEL_EXECUTABLE_MODEL_HPP_
#define SRC_RUNTIME_EXECUTABLE_MODEL_EXECUTABLE_MODEL_HPP_

#include <memory>
#include <atomic>
#include <stdexcept>
#include <functional>
#include <vector>
#include <string>

#include "model/model.hpp"
#include "model/graph/iterator/methods/breadth_first_search.hpp"
// TODO(yc18.cho, TBD): Include only PrepareDispatcher, not all ConcreteDispatchers.
#include "runtime/dispatch/dispatcher_interface.hpp"
#include "runtime/executable_model/executable_operator_list.hpp"
#include "common/ref_hash_map.hpp"
// TODO(yc18.cho, hoon98.choi, TBD): remove all depedency on MemoryManager
//  after implementing lightweight class from this, which is loosely decoupled with runtime.
//  Also lightweight class from EnnBufferCore will be implemented with it.
//  It would be like as flyweight design pattern.
#include "common/ienn_memory_manager.hpp"
#include "runtime/pool/poolable.hpp"

namespace enn {
namespace runtime {

namespace execute {
class ExecuteRequest;
}

using namespace enn::identifier;

class ExecutableModel : public pool::Poolable<IdentifierBase<FullIDType>>,
                        public std::enable_shared_from_this<ExecutableModel> {
 private:
    using UniqueIDDeco = IdentifierDecorator<FullIDType, uint8_t, 0>;
    using OperatorList = enn::model::component::OperatorList;
    using BufferTable = enn::model::memory::BufferTable;
    using Model = enn::model::Model;
    using TableKey = OperatorList::ID;
    using ToDispatch = ExecutableOperatorList::Ptr;

 public:
    using Ptr = std::shared_ptr<ExecutableModel>;
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
    };

 public:
    // ExecutableModel should not be copied and created as default.
    ExecutableModel(const ExecutableModel&) = delete;
    ExecutableModel& operator=(const ExecutableModel&) = delete;
    ExecutableModel() = delete;

    // Factory method that can create ExecutableModel instance.
    template<typename... Args>
    static std::shared_ptr<ExecutableModel> create(Args&&... args) {
        // make shared_ptr creation of ExecutableModel with private constructor possible using inheritance.
        struct MakeSharedEnabler : public ExecutableModel {
            MakeSharedEnabler(Args&&...args)
                : ExecutableModel(std::forward<Args>(args)...) {}
        };
        return std::make_shared<MakeSharedEnabler>(std::forward<Args>(args)...);
    }

    ~ExecutableModel() {
        if (memory_manager_ == nullptr) {
            ENN_WARN_COUT << "The MemoryManager in a Model(ID: 0x"
            << *id_ << ") is null, cannot release a memory object" << std::endl;
        } else {
            for (auto memory_object : memory_objects_) {
                if (memory_manager_->DeleteMemory(memory_object) == ENN_RET_FAILED) {
                    ENN_WARN_COUT << "The MemoryManager in a Model(ID: 0x"
                    << *id_ << ") failed to release a memory object" << std::endl;
                }
            }
        }
        ENN_DBG_COUT << "An ExecutableModel(ID: 0x" << *id_ <<
            ") from a Model(ID: 0x" << model_->get_id() << ") is released." << std::endl;
    }

    // @override a pure virtual function from the Poolable.
    bool operator==(const Poolable& rp) const override { return rp == *id_; }
    bool operator==(const ID& rid) const override { return *id_ == rid; }
    bool is_ancestor_of(const Poolable& rp) const override { return rp.is_descendant_of(*id_); }
    bool is_ancestor_of(const ID& rid) const override { return id_->is_overlapped_by(rid); }
    bool is_descendant_of(const Poolable& rp) const override { return rp.is_ancestor_of(*id_); }
    bool is_descendant_of(const ID& rid) const override { return id_->is_overlapping(rid); }
    std::string to_string() const override {
        std::stringstream s;
        s << "ExecutableModel(ID: 0x" << id_->to_string() << ")";
        return s.str();
    }

    Ptr load(std::unique_ptr<dispatch::IDispatcher> dispatcher) {
        using namespace enn::model::graph::iterator;
        build_buffer_table();
        for (auto& opr_list : model_->get_scheduled_graph()->order<BreadthFirstSearch>()) {
            auto exec_op_list = std::make_shared<ExecutableOperatorList>(*id_, opr_list, buffer_table_);
            add_executable_operator_list(opr_list->get_id(), exec_op_list);
            try {
                dispatcher->dispatch(*exec_op_list);
            } catch (const std::runtime_error& re) {
                ENN_ERR_COUT << re.what() << std::endl;
                ENN_ERR_COUT << "Failed to Dispatch Prepare User Driver : "
                             << (int)(opr_list->get_accelerator()) << std::endl;
                throw std::runtime_error("Prepare Dispatch Failed");
            }
        }
        return shared_from_this();
    }

    const ID& get_id() {
        return *id_;
    }

    const Model::Ptr& get_model() {
        return model_;
    }

    Ptr set_memory_manager(IEnnMemoryManager* memory_manager) {
        memory_manager_ = memory_manager;
        return shared_from_this();
    }

    Ptr add_memory_object(const EnnBufferCore::Ptr& memory_object) {
        memory_objects_.push_back(memory_object);
        return shared_from_this();
    }

 private:
    // Prevent create ExecutableModel object via constructor, but use only create().
    explicit ExecutableModel(Model::Ptr model)
        : id_(std::make_unique<UniqueID>(model->get_id())),
          lock_{false},
          model_(model),
          memory_manager_{nullptr},
          buffer_table_(std::make_shared<BufferTable>()) {
        ENN_DBG_COUT << "An ExecutableModel(ID: 0x" << *id_ <<
            ") from a Model(ID: 0x" << model_->get_id() << ") is created." << std::endl;
    }

    void add_executable_operator_list(const OperatorList::ID& op_list_id,
                                      const ExecutableOperatorList::Ptr& exec_op_list) {
        if (dispatch_table_.find(op_list_id) != dispatch_table_.end()) {
            throw std::invalid_argument(
                "Invalid argument: The OperatorList::UniqueID already exists in the DispatchTable.");
        }
        dispatch_table_.insert({op_list_id, exec_op_list});
    }

    void add_buffer(int32_t index, const void* addr, size_t size) {
        buffer_table_->add(index, addr, size);
    }

    void add_buffer(int32_t index, int32_t fd, const void* addr, size_t size) {
        buffer_table_->add(index, fd, addr, size);
    }

    void build_buffer_table() {
        for (auto& buffer_meta_data : model_->get_buffer_meta_data()) {
            auto& buffer_core = memory_objects_[buffer_meta_data->get_region_index()];
            // add buffer information to BufferTable in ExecutableModel.
            ENN_DBG_PRINT("add_buffer -> index:%d, fd:%d, va:%p, size:%d\n",
                buffer_meta_data->get_index(), buffer_core->fd,
                reinterpret_cast<void *>(
                    reinterpret_cast<char *>(buffer_core->va) + buffer_meta_data->get_offset()),
                buffer_meta_data->get_size());

            add_buffer(
                buffer_meta_data->get_index(), buffer_core->fd,
                reinterpret_cast<void *>(
                    reinterpret_cast<char *>(buffer_core->va) + buffer_meta_data->get_offset()),
                buffer_meta_data->get_size());
        }
    }

 private:
    friend class execute::ExecuteRequest;  // ExecuteRequest can access private members directly.
    ID::UPtr id_;
    // Lock up whenever the ExecuteRequest is created, and then
    //  free this lock when the ExecuteRequest is released.
    std::atomic<bool> lock_;
    Model::Ptr model_;
    IEnnMemoryManager* memory_manager_;
    std::vector<EnnBufferCore::Ptr> memory_objects_;
    BufferTable::Ptr buffer_table_;
    adt::RefHashMap<TableKey, ToDispatch, TableKey::Hash> dispatch_table_;
    // @ structure of dispatch_table_
    //  ---------------------------------------------------------------------
    //  | Key                 | Value                                       |
    //  ---------------------------------------------------------------------
    //  | OperatorList ID     | ExecutableOperatorList { ID, BufferTable }  |
    //  ---------------------------------------------------------------------
    //  ExecutableOperatorList includes OperatorList which is decided by Model and BufferTable.
    //   In order to make the DispatchTable, creating ExecutableOperatorList with the BufferTable
    //   injected by client by traversing scheduled graph and getting each OperatorList.
};


};  // namespace runtime
};  // namespace enn

#endif  // SRC_RUNTIME_EXECUTABLE_MODEL_EXECUTABLE_MODEL_HPP_
