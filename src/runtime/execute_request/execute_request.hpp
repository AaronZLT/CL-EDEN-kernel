#ifndef SRC_RUNTIME_EXECUTE_EXECUTE_REQUEST_HPP_
#define SRC_RUNTIME_EXECUTE_EXECUTE_REQUEST_HPP_

#include <memory>
#include <string>
#include <unordered_map>
#include <functional>
#include <map>

#include "runtime/dispatch/dispatcher_interface.hpp"
#include "runtime/executable_model/executable_model.hpp"
#include "model/graph/iterator/methods/linear_search.hpp"
#include "runtime/execute_request/operator_list_execute_request.hpp"
#include "runtime/pool/poolable.hpp"
#include "common/ref_hash_map.hpp"
#include "tool/profiler/include/ExynosNnProfilerApi.h"

namespace enn {
namespace runtime {
namespace execute {

class ExecuteRequest : public pool::Poolable<IdentifierBase<FullIDType>> {
 private:
    using TableKey = OperatorList::ID;
    using ToDispatch = OperatorListExecuteRequest::Ptr;

 public:
    using Ptr = std::shared_ptr<ExecuteRequest>;
    using ID = ExecutableModel::ID;

 public:
    // Factory method that can create ExecuteRequest instance.
    template<typename... Args>
    static std::unique_ptr<ExecuteRequest> create(Args&&... args) {
        // make unique_ptr creation of ExecuteRequest with private constructor possible using inheritance.
        struct MakeUniqueEnabler : public ExecuteRequest {
            MakeUniqueEnabler(Args&&...args)
                : ExecuteRequest(std::forward<Args>(args)...) {}
        };
        return std::make_unique<MakeUniqueEnabler>(std::forward<Args>(args)...);
    }

    ~ExecuteRequest() {
        // release lock so that ExecuteRequest can be created again from a ExecutaleModel.
        executable_model_->lock_ = false;
        ENN_DBG_COUT << "An ExecuteRequest from an ExecutableModel(ID: 0x"
            << executable_model_->get_id() << ") is released." << std::endl;
    }

    // @override a pure virtual function from the Poolable.
    bool operator==(const Poolable& rp) const override { return rp == *executable_model_->id_; }
    bool operator==(const ID& rid) const override { return *executable_model_->id_ == rid; }
    bool is_ancestor_of(const Poolable& rp) const override { return rp.is_descendant_of(*executable_model_->id_); }
    bool is_ancestor_of(const ID& rid) const override { return executable_model_->id_->is_overlapped_by(rid); }
    bool is_descendant_of(const Poolable& rp) const override { return rp.is_ancestor_of(*executable_model_->id_); }
    bool is_descendant_of(const ID& rid) const override { return executable_model_->id_->is_overlapping(rid); }
    std::string to_string() const override {
        std::stringstream s;
        s << "ExecuteRequest from ExecutableModel(ID: 0x" << executable_model_->get_id().to_string() << ")";
        return s.str();
    }

    void execute(std::unique_ptr<dispatch::IDispatcher> dispatcher) {
        using namespace enn::model::graph::iterator;
        ENN_DBG_COUT << "Start to execute with a ExecutableModel(ID: " <<
                        executable_model_->get_id() << ")" << std::endl;
        auto& scheduled_graph = executable_model_->model_->get_scheduled_graph();
        // TODO(yc18.cho): Make LinearSearch be used only on the linear graph!!
        for (auto& opr_list : scheduled_graph->order<LinearSearch>()) {
            // create OperatorListExecuteRequest and pass it as rvalue to the Dispatcher on the fly.
            //  Dispatch can prolong this temporary object's lifetime as taking by constant reference.
            try {
                dispatcher->dispatch(*dispatch_table_[opr_list->get_id()]);
            } catch (const std::runtime_error& ia) {
                ENN_ERR_COUT << "Failed to dispatch Execute user driver : "
                             << (int)(opr_list->get_accelerator()) << std::endl;
                throw std::runtime_error("Execute Dispatch Failed");
            }
        }
        ENN_DBG_COUT << "Finish to execute with a ExecutableModel(ID: " <<
                        executable_model_->get_id() << ")" << std::endl;
    }

    // TODO : implement asynchronous execution by using ThreadPool.
    void execute_async(std::unique_ptr<dispatch::IDispatcher> dispatcher) {ENN_UNUSED(dispatcher);}

    const ExecutableModel::Ptr& get_executable_model() {
        return executable_model_;
    }

 private:
    // Prevent create ExecuteRequest object via constructor, but use only create().
    ExecuteRequest(ExecutableModel::Ptr executable_model)
        : executable_model_{executable_model} {
        // Only one ExecuteRequest can be created from one ExecutableModel which is constaint.
        if (executable_model_->lock_) {
            throw std::logic_error("Can't create an ExecuteRequest from an ExecutableModel(ID: 0x" +
                                    executable_model_->get_id().to_string() + "). It was already created before!");
        }
        // lock up such that more ExecuteRequest cannot be created from an ExecutableModel.
        executable_model_->lock_ = true;
        for (const auto& [op_list_id, exec_op_list] : executable_model_->dispatch_table_)
            dispatch_table_.insert({op_list_id, std::make_shared<OperatorListExecuteRequest>(exec_op_list)});
        ENN_DBG_COUT << "An ExecuteRequest from an ExecutableModel(ID: 0x"
                << executable_model_->get_id() << ") is created." << std::endl;
    }

 private:
    ExecutableModel::Ptr executable_model_;
    adt::RefHashMap<TableKey, ToDispatch, TableKey::Hash> dispatch_table_;
};

};  // namespace execute
};  // namespace runtime
};  // namespace enn

#endif  // SRC_RUNTIME_EXECUTE_EXECUTE_REQUEST_HPP_
