#ifndef SRC_RUNTIME_DISPATCH_PREPARE_DISPATCHER_HPP_
#define SRC_RUNTIME_DISPATCH_PREPARE_DISPATCHER_HPP_

#include <memory>

#include "runtime/dispatch/dispatcher_interface.hpp"

namespace enn {
namespace runtime {
namespace dispatch {


class PrepareDispatcher : public IDispatcher {
 public:
    PrepareDispatcher(ud::UserDriver& cpu_ud,
                      ud::UserDriver& gpu_ud,
                      ud::UserDriver& npu_ud,
                      ud::UserDriver& dsp_ud,
                      ud::UserDriver& unified_ud)
        : cpu_user_driver_(cpu_ud), gpu_user_driver_(gpu_ud), npu_user_driver_(npu_ud),
          dsp_user_driver_(dsp_ud), unified_user_driver_(unified_ud) {}

    void dispatch(const Dispatchable& dispatchable) override {
        auto& executable_operator_list = static_cast<const ExecutableOperatorList&>(dispatchable);

        auto target_hw = executable_operator_list.get_accelerator();
        ENN_DBG_COUT << "Dispatch a ExecutableOperatorList(ID: "
            << executable_operator_list.get_id() << ") to prepare." << std::endl;
        if (available_accelerator(target_hw, model::Accelerator::NPU)) {
            if (npu_user_driver_.PrepareSubGraph(executable_operator_list) != ENN_RET_SUCCESS) {
                throw std::runtime_error("Failed NPU Prepare SubGraph");
            }
        } else if (available_accelerator(target_hw, model::Accelerator::DSP)) {
            if (dsp_user_driver_.PrepareSubGraph(executable_operator_list) != ENN_RET_SUCCESS) {
                throw std::runtime_error("Failed DSP Prepare SubGraph");
            }
        } else if (available_accelerator(target_hw, model::Accelerator::CPU)) {
            if (cpu_user_driver_.PrepareSubGraph(executable_operator_list) != ENN_RET_SUCCESS) {
                throw std::runtime_error("Failed CPU Prepare SubGraph");
            }
        } else if (available_accelerator(target_hw, model::Accelerator::GPU)) {
            if (gpu_user_driver_.PrepareSubGraph(executable_operator_list) != ENN_RET_SUCCESS) {
                throw std::runtime_error("Failed GPU Prepare SubGraph");
            }
        } else if (available_accelerator(target_hw, model::Accelerator::UNIFIED)) {
            if (unified_user_driver_.PrepareSubGraph(executable_operator_list) != ENN_RET_SUCCESS) {
                throw std::runtime_error("Failed Unified UD Prepare SubGrap");
            }
        } else {
            ENN_DBG_PRINT("Not Supported Target Device : %d", (int)target_hw);
            throw std::runtime_error("Not Supported Target to Prepare");
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

#endif  // SRC_RUNTIME_DISPATCH_PREPARE_DISPATCHER_HPP_
