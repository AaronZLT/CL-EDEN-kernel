#ifndef SRC_RUNTIME_USERDRIVER_MANAGER_HPP_
#define SRC_RUNTIME_USERDRIVER_MANAGER_HPP_

#include <memory>

#include "userdriver/cpu/cpu_userdriver.h"
#include "userdriver/gpu/gpu_userdriver.h"
#include "userdriver/unified/npu_userdriver.h"
#include "userdriver/unified/dsp_userdriver.h"
#include "userdriver/unified/unified_userdriver.h"

#include "runtime/dispatch/open_dispatcher.hpp"
#include "runtime/dispatch/prepare_dispatcher.hpp"
#include "runtime/dispatch/execute_dispatcher.hpp"
#include "runtime/dispatch/close_dispatcher.hpp"
#include "runtime/dispatch/session_id_query_dispatcher.hpp"

namespace enn {
namespace runtime {

// TODO(yc18.cho, TBD): add unit test for dispatcher with mock classes of userdrivers.

class UserdriverManager {
 public:
    using UPtr = std::unique_ptr<UserdriverManager>;

    // Ctr to be called when UserdriverManager is first created.
    UserdriverManager()
        : cpu_user_driver_{ud::cpu::CpuUserDriver::get_instance()},
          gpu_user_driver_{ud::gpu::GpuUserDriver::get_instance()},
          npu_user_driver_{ud::npu::NpuUserDriver::get_instance()},
          dsp_user_driver_{ud::dsp::DspUserDriver::get_instance()},
          unified_user_driver_{ud::unified::UnifiedUserDriver::get_instance()} {
        cpu_user_driver_.Initialize();
        gpu_user_driver_.Initialize();
        npu_user_driver_.Initialize();
        dsp_user_driver_.Initialize();
        unified_user_driver_.Initialize();
        ENN_DBG_COUT << "UserdriverManager is constructed" << std::endl;
    }

    virtual ~UserdriverManager() {
        cpu_user_driver_.Deinitialize();
        gpu_user_driver_.Deinitialize();
        npu_user_driver_.Deinitialize();
        dsp_user_driver_.Deinitialize();
        unified_user_driver_.Deinitialize();
        ENN_DBG_COUT << "UserdriverManager is destructed" << std::endl;
    }

    std::unique_ptr<dispatch::OpenDispatcher> create_open_dispatcher() {
        return std::make_unique<dispatch::OpenDispatcher>(cpu_user_driver_,
                                                          gpu_user_driver_,
                                                          npu_user_driver_,
                                                          dsp_user_driver_,
                                                          unified_user_driver_);
    }

    std::unique_ptr<dispatch::PrepareDispatcher> create_prepare_dispatcher() {
        return std::make_unique<dispatch::PrepareDispatcher>(cpu_user_driver_,
                                                             gpu_user_driver_,
                                                             npu_user_driver_,
                                                             dsp_user_driver_,
                                                             unified_user_driver_);
    }

    std::unique_ptr<dispatch::ExecuteDispatcher> create_execute_dispatcher() {
        return std::make_unique<dispatch::ExecuteDispatcher>(cpu_user_driver_,
                                                             gpu_user_driver_,
                                                             npu_user_driver_,
                                                             dsp_user_driver_,
                                                             unified_user_driver_);
    }

    std::unique_ptr<dispatch::CloseDispatcher> create_close_dispatcher() {
        return std::make_unique<dispatch::CloseDispatcher>(cpu_user_driver_,
                                                           gpu_user_driver_,
                                                           npu_user_driver_,
                                                           dsp_user_driver_,
                                                           unified_user_driver_);
    }

    std::unique_ptr<dispatch::SessionIdQueryDispatcher> create_session_id_query_dispatcher() {
        return std::make_unique<dispatch::SessionIdQueryDispatcher>(dsp_user_driver_);
    }


 protected:
    // TODO(yc18.cho, TBD): change reference to value after singleton pattern is not used.
    ud::cpu::CpuUserDriver& cpu_user_driver_;
    ud::gpu::GpuUserDriver& gpu_user_driver_;
    ud::npu::NpuUserDriver& npu_user_driver_;
    ud::dsp::DspUserDriver& dsp_user_driver_;
    ud::unified::UnifiedUserDriver& unified_user_driver_;
};

};  // namespace runtime
};  // namespace enn

#endif  // SRC_RUNTIME_USERDRIVER_MANAGER_HPP_
