#ifndef SRC_MODEL_MODEL_HPP_
#define SRC_MODEL_MODEL_HPP_

#include <memory>
#include <map>
#include <vector>
#include <utility>
#include <functional>
#include <string>
#include <exception>

#include "model/graph/graph.hpp"
#include "model/graph/iterator/methods/breadth_first_search.hpp"
#include "model/meta_data/buffer_meta_data.hpp"
#include "model/attribute.hpp"
#include "model/component/operator/operator.hpp"
#include "model/component/operator/operator_list.hpp"
#include "model/component/tensor/feature_map.hpp"
#include "model/graph_types.hpp"
#include "runtime/dispatch/dispatcher_interface.hpp"
// TODO(yc18.cho, hoon98.choi, TBD): remove all depedency on MemoryManager
//  after implementing lightweight class from this, which is loosely decoupled with runtime.
//  Also lightweight class from EnnBufferCore will be implemented with it.
//  It would be like as flyweight design pattern.
#include "common/ienn_memory_manager.hpp"
#include "runtime/pool/poolable.hpp"
#include "runtime/client_process/client_process.hpp"

namespace enn {
namespace model {

using namespace component;
using namespace enn::identifier;

class Model : public runtime::pool::Poolable<IdentifierBase<FullIDType>>,
              public std::enable_shared_from_this<Model> {
 private:
    using UniqueIDDeco = IdentifierDecorator<FullIDType, uint8_t, 24>;

 public:
    using Ptr = std::shared_ptr<Model>;
    using ID = enn::identifier::IdentifierBase<identifier::FullIDType>;
    class UniqueID : public UniqueIDDeco {
     public:
        // bring data injected to base template class for client to access directly at compile-time.
        using UniqueIDDeco::Type;
        using UniqueIDDeco::Max;
        using UniqueIDDeco::Offset;
        using UniqueIDDeco::Mask;

        explicit UniqueID(const IdentifierBase<FullIDType>& base_id)
            : UniqueIDDeco(base_id) {}
    };

 public:
    // Disable copy & move sementics
    Model(Model&&) = delete;
    Model& operator=(Model&&) = delete;
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    ~Model() {
        try {
            unload();  // release all resources including ones of userdriver.
            if (memory_manager_ == nullptr) {
                ENN_WARN_COUT << "The MemoryManager in a Model(ID: 0x"
                << *id_ << ") is null, cannot release a memory object" << std::endl;
            } else {
                ENN_DBG_COUT << "Release a memory_object_pool in a Model(ID: 0x" << *id_ << ")..." << std::endl;
                for (auto &memory_object_ : memory_object_pool) {
                    if (memory_manager_->DeleteMemory(memory_object_) == ENN_RET_FAILED) {
                        ENN_WARN_COUT << "The MemoryManager in a Model(ID: 0x" << *id_
                                      << ") failed to release a memory object" << std::endl;
                    }
                }
            }
        } catch (const std::exception& ex) {
            ENN_ERR_COUT << ex.what() << std::endl;
        }
        ENN_DBG_COUT << "A Model(ID: 0x" << *id_ << ") is released." << std::endl;
    }

    explicit Model(const runtime::ClientProcess::Ptr& client_process)
        : id_(std::make_unique<UniqueID>(client_process->get_id())),
          client_process_{client_process},
          memory_manager_{nullptr} {
              ENN_DBG_COUT << "A Model(ID: 0x" << *id_ << ") is created." << std::endl;
          }

    Model(const runtime::ClientProcess::Ptr& client_process,
          OriginalGraph::Ptr origin_graph)
        : id_(std::make_unique<UniqueID>(client_process->get_id())),
          client_process_{client_process},
          origin_graph_(origin_graph),
          memory_manager_{nullptr} {
              ENN_DBG_COUT << "A Model(ID: 0x" << *id_ << ") is created." << std::endl;
          }

    Model(const runtime::ClientProcess::Ptr& client_process,
          OriginalGraph::Ptr origin_graph,
          const Attribute::Ptr& attribute)
        : id_(std::make_unique<UniqueID>(client_process->get_id())),
          client_process_{client_process},
          origin_graph_(origin_graph),
          memory_manager_{nullptr},
          attribute_(attribute) {
              ENN_DBG_COUT << "Model(ID: 0x" << *id_ << ") is created." << std::endl;
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
        s << "Model(ID: 0x" << id_->to_string() << ")";
        return s.str();
    }

    Ptr set_open_dispatcher(std::unique_ptr<runtime::dispatch::IDispatcher> dispatcher) {
        open_dispatcher_ = std::move(dispatcher);
        return shared_from_this();
    }

    Ptr set_close_dispatcher(std::unique_ptr<runtime::dispatch::IDispatcher> dispatcher) {
        close_dispatcher_ = std::move(dispatcher);
        return shared_from_this();
    }

    void load() {
        using namespace enn::model::graph::iterator;
        ENN_DBG_COUT << "Load the Model(ID: 0x" << *id_ << ")." << std::endl;
        for (auto& opr_list : scheduled_graph_->order<BreadthFirstSearch>()) {
            try {
                open_dispatcher_->dispatch(*opr_list);
            } catch (const std::runtime_error& re) {
                ENN_ERR_COUT << re.what() << std::endl;
                ENN_ERR_COUT << "Failed to Dispatch Open User Driver : " << (int)(opr_list->get_accelerator()) << std::endl;
                throw std::runtime_error("Open Dispatch Failed");
            }
        }
    }

    Ptr set_origin_graph(OriginalGraph::Ptr origin_graph) {
        origin_graph_ = std::move(origin_graph);
        return shared_from_this();
    }

    Ptr set_scheduled_graph(ScheduledGraph::Ptr op_lists) {
        scheduled_graph_ = std::move(op_lists);
        return shared_from_this();
    }

    const OriginalGraph::Ptr& get_origin_graph() {
        return origin_graph_;
    }

    const ScheduledGraph::Ptr& get_scheduled_graph() {
        return scheduled_graph_;
    }

    const runtime::ClientProcess::Ptr& get_client_process() {
        return client_process_;
    }

    void add_buffer_meta_data(metadata::BufferMetaData::Ptr buf_meta_data) {
        buffer_meta_data_.push_back(buf_meta_data);
    }

    const std::vector<metadata::BufferMetaData::Ptr>& get_buffer_meta_data() {
        return buffer_meta_data_;
    }

    const ID& get_id() const {
        return *id_;
    }

    template <typename T>
    Ptr set_attribute(T&& attribute) {
        attribute_ = std::forward<T>(attribute);
        return shared_from_this();
    }

    const Attribute::Ptr& get_attribute() const {
        return attribute_;
    }

    Ptr set_memory_manager(IEnnMemoryManager* memory_manager) {
        memory_manager_ = memory_manager;
        return shared_from_this();
    }

    Ptr add_memory_object(const EnnBufferCore::Ptr& memory_object) {
        memory_object_pool.push_back(memory_object);
        return shared_from_this();
    }

 private:
    void unload() {
        using namespace enn::model::graph::iterator;
        ENN_DBG_COUT << "Unload the Model(ID: 0x" << *id_ << ")." << std::endl;
        if (close_dispatcher_ == nullptr) throw std::runtime_error("Error: close_dispatcher is nullptr");
        for (auto& opr_list : scheduled_graph_->order<BreadthFirstSearch>()) {
            try {
                close_dispatcher_->dispatch(*opr_list);
            } catch (const std::runtime_error& re) {
                ENN_ERR_COUT << re.what() << std::endl;
                ENN_ERR_COUT << "Failed to Dispatch Close User Driver : "
                             << (int)(opr_list->get_accelerator()) << std::endl;
                throw std::runtime_error("Close Dispatch Failed");
            }
        }
    }

 private:
    ID::UPtr id_;  // model id generated to distinguish it from other models
    runtime::ClientProcess::Ptr client_process_;   // Client process that creates me.
    OriginalGraph::Ptr origin_graph_;  // graph composed of individual Operator and FeatureMap
    ScheduledGraph::Ptr scheduled_graph_;  // graph composed of OperatorList and FeatureMap
    std::vector<metadata::BufferMetaData::Ptr> buffer_meta_data_;  // memory info to be allocated
    IEnnMemoryManager* memory_manager_;
    std::vector<EnnBufferCore::Ptr> memory_object_pool;
    Attribute::Ptr attribute_;  // attribute defined by GraphGen
    std::unique_ptr<runtime::dispatch::IDispatcher> open_dispatcher_;
    std::unique_ptr<runtime::dispatch::IDispatcher> close_dispatcher_;
};


};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_MODEL_HPP_
