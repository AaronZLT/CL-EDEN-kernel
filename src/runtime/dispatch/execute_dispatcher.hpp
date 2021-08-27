#ifndef SRC_RUNTIME_DISPATCH_EXECUTE_DISPATCHER_HPP_
#define SRC_RUNTIME_DISPATCH_EXECUTE_DISPATCHER_HPP_

#include <memory>

#include "runtime/dispatch/dispatcher_interface.hpp"

namespace enn {
namespace runtime {
namespace dispatch {


class ExecuteDispatcher : public IDispatcher {
 public:
    ExecuteDispatcher(ud::UserDriver& cpu_ud,
                      ud::UserDriver& gpu_ud,
                      ud::UserDriver& npu_ud,
                      ud::UserDriver& dsp_ud,
                      ud::UserDriver& unified_ud)
        : cpu_user_driver_(cpu_ud), gpu_user_driver_(gpu_ud), npu_user_driver_(npu_ud),
          dsp_user_driver_(dsp_ud), unified_user_driver_(unified_ud) {}

    void dispatch(const Dispatchable& dispatchable) override {
        auto& operator_list_execute_request = static_cast<const OperatorListExecuteRequest&>(dispatchable);
        auto target_hw = operator_list_execute_request.get_accelerator();
        ENN_DBG_COUT << "Dispatch a OperatorListExecuteRequest from the ExecutableOperatorList(ID: "
            << operator_list_execute_request.get_executable_operator_list_id() << ") to execute." << std::endl;
        if (available_accelerator(target_hw, model::Accelerator::NPU)) {
            if (npu_user_driver_.ExecuteSubGraph(operator_list_execute_request) != ENN_RET_SUCCESS) {
                throw std::runtime_error("Failed NPU Execute SubGraph");
            }
        } else if (available_accelerator(target_hw, model::Accelerator::DSP)) {
            if (dsp_user_driver_.ExecuteSubGraph(operator_list_execute_request) != ENN_RET_SUCCESS) {
                throw std::runtime_error("Failed DSP Execute SubGraph");
            }
        } else if (available_accelerator(target_hw, model::Accelerator::CPU)) {
            if (cpu_user_driver_.ExecuteSubGraph(operator_list_execute_request) != ENN_RET_SUCCESS) {
                throw std::runtime_error("Failed CPU Execute SubGraph");
            }
        } else if (available_accelerator(target_hw, model::Accelerator::GPU)) {
            if (gpu_user_driver_.ExecuteSubGraph(operator_list_execute_request) != ENN_RET_SUCCESS) {
                throw std::runtime_error("Failed GPU Execute SubGraph");
            }
        } else if (available_accelerator(target_hw, model::Accelerator::UNIFIED)) {
            if (unified_user_driver_.ExecuteSubGraph(operator_list_execute_request) != ENN_RET_SUCCESS) {
                throw std::runtime_error("Fail Unified UD Execute SubGraph");
            }
        } else {
            ENN_ERR_PRINT("[Error] Unsuppoerted hardware : %d", (int)target_hw);
            throw std::runtime_error("Not Supported Hardware to Execute");
        }
    }

private:
    ud::UserDriver& cpu_user_driver_;
    ud::UserDriver& gpu_user_driver_;
    ud::UserDriver& npu_user_driver_;
    ud::UserDriver& dsp_user_driver_;
    ud::UserDriver& unified_user_driver_;
};


};  // namespace dispatch
};  // namespace runtime
};  // namespace enn

#endif  // SRC_RUNTIME_DISPATCH_EXECUTE_DISPATCHER_HPP_
