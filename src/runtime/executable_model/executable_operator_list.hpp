#ifndef SRC_RUNTIME_EXECUTABLE_OPERATOR_LIST_WRAP_HPP_
#define SRC_RUNTIME_EXECUTABLE_OPERATOR_LIST_WRAP_HPP_

#include <unordered_map>

#include "model/component/operator/operator_list.hpp"
#include "model/memory/buffer_table.hpp"

namespace enn {
namespace runtime {

using namespace enn::model::component;

class ExecutableOperatorList : public dispatch::Dispatchable {
 private:
    using BufferTable = enn::model::memory::BufferTable;

 public:
    using Ptr = std::shared_ptr<ExecutableOperatorList>;
    using ID = enn::identifier::IdentifierBase<identifier::FullIDType>;

 public:
    ExecutableOperatorList(const enn::identifier::IdentifierBase<identifier::FullIDType>& base_id,
                           const OperatorList::Ptr& operator_list,
                           const BufferTable::Ptr& buffer_table)
        : id_(std::make_unique<OperatorList::UniqueID>(
            (operator_list->get_id() & OperatorList::UniqueID::Mask) >> OperatorList::UniqueID::Offset, base_id)),
          operator_list_{operator_list},
          buffer_table_{buffer_table} {
              ENN_DBG_COUT << "An ExecutableOperatorList(ID: 0x" << *id_ <<
                  ") from an OperatorList(ID: 0x" << operator_list_->get_id() << ") is created." << std::endl;
          }

    ExecutableOperatorList() {
        ENN_DBG_COUT << "An ExecutableOperatorList(ID: 0x" << *id_ <<
            ") from an OperatorList(ID: 0x" << operator_list_->get_id() << ") is created." << std::endl;
    }

    ~ExecutableOperatorList() {
        ENN_DBG_COUT << "An ExecutableOperatorList(ID: 0x" << *id_ <<
            ") from an OperatorList(ID: 0x" << operator_list_->get_id() << ") is released." << std::endl;
    }

    const ID& get_id() const {
        return *id_;
    }

    const BufferTable& get_buffer_table() const {
        return *buffer_table_;
    }

    const OperatorList::ID& get_operator_list_id() const {
        return operator_list_->get_id();
    }

    const enn::model::Accelerator& get_accelerator() const override {
        return operator_list_->get_accelerator();
    }

 private:
    ID::UPtr id_;
    OperatorList::Ptr operator_list_;
    BufferTable::Ptr buffer_table_;
};


};  // namespace runtime
};  // namespace enn

#endif  // SRC_RUNTIME_EXECUTABLE_OPERATOR_LIST_WRAP_HPP_
