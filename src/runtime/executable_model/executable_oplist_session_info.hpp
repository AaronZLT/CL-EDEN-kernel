#ifndef SRC_RUNTIME_EXECUTABLE_OP_LIST_SESSION_INFO_HPP_
#define SRC_RUNTIME_EXECUTABLE_OP_LIST_SESSION_INFO_HPP_

#include "model/component/operator/operator_list.hpp"

namespace enn {
namespace runtime {

using namespace enn::model::component;

// Dispatchable Class for DSP user driver to support (Cmera) DD to (DSP) DD communication.
// ExecutableOpListSessionInfo Class has ID for opened model's oplist, and by this id,
// dsp user-driver will fill the device session id from dsp device driver.
class ExecutableOpListSessionInfo : public dispatch::Dispatchable {
 public:
    using Ptr = std::shared_ptr<ExecutableOpListSessionInfo>;
    using ID = enn::identifier::IdentifierBase<identifier::FullIDType>;
    using DeviceSessionID = int32_t;

    ExecutableOpListSessionInfo(const OperatorList::Ptr& operator_list)
        : id_(operator_list->get_id()), hw_(operator_list->get_accelerator()), device_session_id_(0) {
            ENN_DBG_COUT << "Op List ID : " << id_ << std::endl;
          }

    ~ExecutableOpListSessionInfo() = default;

    const ID& get_id() const {
        return id_;
    }

    DeviceSessionID get_device_session_id() const {
        return device_session_id_;
    }

    void set_session_id(DeviceSessionID s_id) {
        this->device_session_id_ = s_id;
    }

    const enn::model::Accelerator& get_accelerator() const override {
        return hw_;
    }

 private:
    const ID& id_;
    enn::model::Accelerator hw_;
    DeviceSessionID device_session_id_;
};


};  // namespace runtime
};  // namespace enn

#endif  // SRC_RUNTIME_EXECUTABLE_OP_LIST_SESSION_INFO_HPP_
