#ifndef SRC_RUNTIME_DISPATCH_OPEN_DISPATCHER_HPP_
#define SRC_RUNTIME_DISPATCH_OPEN_DISPATCHER_HPP_

#include <memory>

#include "runtime/dispatch/dispatcher_interface.hpp"

namespace enn {
namespace runtime {
namespace dispatch {

class OpenDispatcher : public IDispatcher {
 public:
    OpenDispatcher(ud::UserDriver& cpu_ud,
                   ud::UserDriver& gpu_ud,
                   ud::UserDriver& npu_ud,
                   ud::UserDriver& dsp_ud,
                   ud::UserDriver& unified_ud)
        : cpu_user_driver_(cpu_ud), gpu_user_driver_(gpu_ud), npu_user_driver_(npu_ud),
          dsp_user_driver_(dsp_ud), unified_user_driver_(unified_ud) {}

    void dispatch(const Dispatchable& dispatchable) override {
        auto& operator_list = static_cast<const OperatorList&>(dispatchable);

        auto target_hw = operator_list.get_accelerator();
        ENN_DBG_COUT << "Dispatch a OperatorList(ID: " <<
            operator_list.get_id() << ") to open." << std::endl;
        if (available_accelerator(target_hw, model::Accelerator::NPU)) {
            if (npu_user_driver_.OpenSubGraph(operator_list) != ENN_RET_SUCCESS) {
                throw std::runtime_error("Failed NPU Open SubGraph");
            }
        } else if (available_accelerator(target_hw, model::Accelerator::CPU)) {
            if (cpu_user_driver_.OpenSubGraph(operator_list) != ENN_RET_SUCCESS) {
                throw std::runtime_error("Failed CPU Open SubGraph");
            }
        } else if (available_accelerator(target_hw, model::Accelerator::GPU)) {
            if (gpu_user_driver_.OpenSubGraph(operator_list) != ENN_RET_SUCCESS) {
                throw std::runtime_error("Failed GPU Open SubGraph");
            }
        } else if (available_accelerator(target_hw, model::Accelerator::DSP)) {
            if (dsp_user_driver_.OpenSubGraph(operator_list) != ENN_RET_SUCCESS) {
                throw std::runtime_error("Failed DSP Open SubGraph");
            }
        } else if (available_accelerator(target_hw, model::Accelerator::UNIFIED)) {
            if (unified_user_driver_.OpenSubGraph(operator_list) != ENN_RET_SUCCESS) {
                throw std::runtime_error("Failed Unified UD Open SubGraph");
            }
        } else {
            ENN_ERR_PRINT("Not Supported Target Device : %d\n", (int)target_hw);
            throw std::runtime_error("Not supported Target Device to Open");
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

#endif  // SRC_RUNTIME_DISPATCH_OPEN_DISPATCHER_HPP_