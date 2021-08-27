#ifndef SRC_RUNTIME_DISPATCH_SESSION_ID_QUERY_DISPATCHER_HPP_
#define SRC_RUNTIME_DISPATCH_SESSION_ID_QUERY_DISPATCHER_HPP_

#include <memory>

#include "runtime/dispatch/dispatcher_interface.hpp"

namespace enn {
namespace runtime {
namespace dispatch {

// Dispatcher Class for DSP user driver to support (Cmera) DD to (DSP) DD communication.
// Session ID Querry Dispatcher Class will get the opened device session id from Device Driver
// through DSP user driver, and deliver it to Client Layer.
class SessionIdQueryDispatcher : public IDispatcher {
 public:
    SessionIdQueryDispatcher(ud::dsp::DspUserDriver& dsp_ud) : dsp_user_driver_(dsp_ud) {}

    void dispatch(const Dispatchable& dispatchable) override {
        auto& op_list_session_info = static_cast<const ExecutableOpListSessionInfo&>(dispatchable);

        auto target_hw = op_list_session_info.get_accelerator();
        ENN_DBG_COUT << "Dispatch a OperatorList(ID: " <<
            op_list_session_info.get_id() << ") to query session id of DD." << std::endl;
        if (available_accelerator(target_hw, model::Accelerator::DSP)) {
            if (dsp_user_driver_.get_dsp_session_id(
                const_cast<ExecutableOpListSessionInfo&>(op_list_session_info)) != ENN_RET_SUCCESS) {
                throw std::runtime_error("Failed Get DSP session ID");
            }
        } else {
            ENN_ERR_PRINT("Not Supported Target Device : %d\n", (int)target_hw);
            throw std::runtime_error("Not Supported Target to query session id");
        }
    }

 private:
    ud::dsp::DspUserDriver& dsp_user_driver_;
};


};  // namespace dispatch
};  // namespace runtime
};  // namespace enn

#endif  // SRC_RUNTIME_DISPATCH_SESSION_ID_QUERY_DISPATCHER_HPP_
