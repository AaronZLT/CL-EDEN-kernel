#ifndef SRC_RUNTIME_EXECUTE_REQUEST__OPERATOR_LIST_EXECUTE_REQUEST_HPP_
#define SRC_RUNTIME_EXECUTE_REQUEST__OPERATOR_LIST_EXECUTE_REQUEST_HPP_

#include <memory>

#include "runtime/dispatch/dispatchable.hpp"
#include "runtime/executable_model/executable_operator_list.hpp"
#include "common/identifier.hpp"

namespace enn {
namespace runtime {

class OperatorListExecuteRequest : public dispatch::Dispatchable {
 public:
    using BufferTable = enn::model::memory::BufferTable;

 public:
    using Ptr = std::shared_ptr<OperatorListExecuteRequest>;

    OperatorListExecuteRequest(const ExecutableOperatorList::Ptr& exec_op_list)
        : exec_op_list_{exec_op_list}, buffer_table_{nullptr} {
            ENN_DBG_COUT << "An OperatorListExecuteRequest from an ExecutableOperatorList(ID: 0x"
                << get_executable_operator_list_id() << ") is created." << std::endl;
        }

    OperatorListExecuteRequest(const ExecutableOperatorList::Ptr& exec_op_list,
                               const BufferTable::Ptr& buffer_table)
        : exec_op_list_{exec_op_list}, buffer_table_{buffer_table} {
            ENN_DBG_COUT << "An OperatorListExecuteRequest from an ExecutableOperatorList(ID: 0x"
                << get_executable_operator_list_id() << ") is created." << std::endl;
        }

    ~OperatorListExecuteRequest() {
        ENN_DBG_COUT << "An OperatorListExecuteRequest from an ExecutableOperatorList(ID: 0x"
            << get_executable_operator_list_id() << ") is released." << std::endl;
    }

    const enn::identifier::IdentifierBase<identifier::FullIDType>& get_executable_operator_list_id() const {
        return exec_op_list_->get_id();
    }

    const enn::identifier::IdentifierBase<identifier::FullIDType>& get_operator_list_id() const {
        return exec_op_list_->get_operator_list_id();
    }

    const BufferTable& get_buffer_table() const {
        if (buffer_table_) return *buffer_table_;
        return exec_op_list_->get_buffer_table();
    }

    const enn::model::Accelerator& get_accelerator() const override {
        return exec_op_list_->get_accelerator();
    }

 private:
    ExecutableOperatorList::Ptr exec_op_list_;
    BufferTable::Ptr buffer_table_;
    // @ Insert here preferences or options for execution...
};

};  // namespace model
};  // namespace enn

#endif  // SRC_RUNTIME_EXECUTE_REQUEST__OPERATOR_LIST_EXECUTE_REQUEST_HPP_
