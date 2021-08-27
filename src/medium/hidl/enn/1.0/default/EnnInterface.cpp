/**
 * @file EnnInterface.cpp
 * @author Hoon Choi (hoon98.choi@samsung.com)
 * @brief
 * @version 0.1
 * @date 2020-12-28
 *
 * @copyright Copyright (c) 2020 Samsung Electronics
 *
 */

#include "EnnInterface.h"

#include "client/enn_api-type.h"
#include "common/enn_debug.h"
#include "medium/enn_medium_interface.h"
#include "medium/enn_medium_utils.hpp"
#include "runtime/engine.hpp"

#include <hwbinder/IPCThreadState.h>

// flush the coverage report (used in service mode)
#ifdef COVERAGE_ENABLED
extern "C" void __gcov_dump();
extern "C" void __gcov_reset();
#endif

namespace vendor::samsung_slsi::hardware::enn::implementation {

void EnnClientRecipient::serviceDied(uint64_t cookie, const android::wp<::android::hidl::base::V1_0::IBase>&) {
    ENN_WARN_PRINT("Service: detect client is died! 0x%" PRIX64 "\n", cookie);
    if (!cm.PopClient(cookie)) {
        ENN_WARN_PRINT("Client 0x%" PRIX64 " is not existed in Service\n", cookie);
    }

    if (::enn::runtime::Engine::get_instance()->shutdown_client_process(cookie) != ENN_RET_SUCCESS) {
        ENN_WARN_PRINT("There is No Model to Close\n");
    }
    return;
}

/* Thread to monitor clients */
void EnnInterface::ClientMonitorThreadFunction(void) {
    ENN_INFO_PRINT("Client Monitor is initialized.\n");
    status = EnnInterfaceStatus::StatusActive;
    while (true) {
        std::unique_lock<std::mutex> guard(serviceInterfaceMutex);
        if (status == EnnInterfaceStatus::StatusOff) {
            ENN_INFO_PRINT("Monitor will be terminated because the service is aborted.\n");
            return;
        }

        if (status != EnnInterfaceStatus::StatusAbortBefore) {
            ENN_DBG_PRINT("Wait for %d second(s)\n", check_interval);
            monitor_thread_cv.wait_for(guard, std::chrono::seconds(check_interval));
        }

        status = EnnInterfaceStatus::StatusActive;
        client_manager.show();
    }
}

EnnInterface::EnnInterface(): status(EnnInterfaceStatus::StatusInit) {
    // NOTE(hoon98.choi): current data structure has thread-safe design, so I remove this
    //                    temporally. But any problems are occured, we'll uncomment the below and
    //                    put some thread-safe machanism.
    //   clientMonitorThread = std::thread([&] { ClientMonitorThreadFunction(); });
}

EnnInterface::~EnnInterface() {
    std::unique_lock<std::mutex> guard(serviceInterfaceMutex);
    ENN_INFO_PRINT("-\n");
    status = EnnInterfaceStatus::StatusOff;

    return;
}


Return<int32_t> EnnInterface::init(
    const sp<::vendor::samsung_slsi::hardware::enn::V1_0::IEnnCallback>& cb_sp) {
    std::unique_lock<std::mutex> guard(serviceInterfaceMutex);
    auto client_recipient = new EnnClientRecipient(client_manager);
    auto pid = ::enn::util::get_caller_pid();
    client_recipient_map[pid] = client_recipient;

    /* ping test to client if connected */
    ENN_INFO_PRINT("Alive call and returned (ping test): %s\n", cb_sp->isAlive() == true ? "True" : "False");

    /* set linkToDeath */
    if (cb_sp != nullptr && !cb_sp->linkToDeath(client_recipient, pid))
        ENN_WARN_PRINT("Link monitor is not set\n");
    else
        ENN_INFO_PRINT("Link monitor is set(%d)\n", pid);

    client_manager.PutClient(pid, cb_sp);

    return ::enn::runtime::Engine::get_instance()->init();
}

Return<int32_t> EnnInterface::deinit() {
    std::unique_lock<std::mutex> guard(serviceInterfaceMutex);
    auto pid = ::enn::util::get_caller_pid();
    auto pid_ele = client_manager.GetClient(pid);

    /* ping test to client if connected */
    ENN_INFO_PRINT("Alive call and returned (ping test): %s\n", pid_ele->cb_sp->isAlive() == true ? "True" : "False");

    if (pid_ele->cb_sp != nullptr) {
        if (client_recipient_map.find(pid) != client_recipient_map.end()) {
            pid_ele->cb_sp->unlinkToDeath(client_recipient_map[pid]);
            client_recipient_map.erase(pid);
            pid_ele->cb_sp.clear();
            ENN_INFO_PRINT("Unlink to Death Completed\n");
        }
    }

    client_manager.PopClient(pid);

    // NOTE(hoon98.choi): If anything is need to be disabled, please put code here.
    ENN_INFO_PRINT_FORCE("Now Available clients: %d\n", client_manager.GetClientNum());

    int32_t ret = ::enn::runtime::Engine::get_instance()->deinit();
#ifdef COVERAGE_ENABLED
    __gcov_dump();
    __gcov_reset();
#endif
    return ret;
}

Return<void> EnnInterface::open_model(
        const ::vendor::samsung_slsi::hardware::enn::V1_0::LoadParameter& load_param, open_model_cb _hidl_cb) {
    ::vendor::samsung_slsi::hardware::enn::V1_0::SessionBufInfo test;
    ::enn::runtime::Engine::get_instance()->open_model(load_param, &test);

    _hidl_cb(test);

    return Void();
}

Return<uint64_t> EnnInterface::commit_execution_data(
        const uint64_t model_id, const ::vendor::samsung_slsi::hardware::enn::V1_0::InferenceData& exec_data) {
    auto ret = ::enn::runtime::Engine::get_instance()->commit_execution_data(model_id, exec_data);
    return ret;
}

Return<int32_t> EnnInterface::execute_model(const hidl_vec<uint64_t>& exec_id_list)  {
    return ::enn::runtime::Engine::get_instance()->execute_model(exec_id_list);
}

Return<int32_t> EnnInterface::close_model(uint64_t model_id) {
    return ::enn::runtime::Engine::get_instance()->close_model(model_id);
}

// @NOTE(hoon98.choi): This function need to synchronize with medium interface in NON-HIDL mode.
Return<void> EnnInterface::custom_interface(const uint32_t identifier,
                             const ::vendor::samsung_slsi::hardware::enn::V1_0::GeneralParameter &parameter, custom_interface_cb _hidl_cb) {
    ::vendor::samsung_slsi::hardware::enn::V1_0::GeneralParameterReturn rettype;
    if (identifier == static_cast<uint32_t>(CustomFunctionTypeId::GET_DSP_SESSION_ID)) {
        uint64_t id = static_cast<uint64_t>(parameter.u32_v[0]) << 32 | static_cast<uint64_t>(parameter.u32_v[1]);
        auto ret = ::enn::runtime::Engine::get_instance()->get_dsp_session_id(id);
        rettype.i32_v.resize(1);
        rettype.i32_v[0] = ret;
        _hidl_cb(rettype);
    }
    return Void();
}


}  // namespace vendor::samsung_slsi::hardware::enn::implementation
