/*
 * Copyright (C) 2020 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or
 * distributed, transmitted, transcribed, stored in a retrieval system or
 * translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung
 * Electronics.
 */

#pragma once

#include <hidl/MQDescriptor.h>
#include <hidl/Status.h>
#include <hwbinder/IPCThreadState.h>

#include <vendor/samsung_slsi/hardware/enn/1.0/IEnnInterface.h>
#include <vendor/samsung_slsi/hardware/enn/1.0/IEnnCallback.h>
#include "medium/enn_medium_interface.h"
#include "medium/enn_client_manager.hpp"
#include <memory>
#include <map>
#include <mutex>
#include <chrono>
#include <thread>


// TODO(hoon98.choi): Change 0x70 to target dependent value
constexpr uint32_t AFFINITY_MID_CORE = 0x70;
namespace vendor::samsung_slsi::hardware::enn::implementation {

using ::android::sp;
using ::android::hardware::hidl_array;
using ::android::hardware::hidl_memory;
using ::android::hardware::hidl_string;
using ::android::hardware::hidl_handle;
using ::android::hardware::hidl_vec;
using ::android::hardware::Return;
using ::android::hardware::Void;
using ClientManagerType =
    ::enn::interface::ClientManager<int32_t, sp<::vendor::samsung_slsi::hardware::enn::V1_0::IEnnCallback>>;

class EnnClientRecipient : public android::hardware::hidl_death_recipient {
public:
    explicit EnnClientRecipient(ClientManagerType & _cm): cm(_cm) {}
    virtual void serviceDied(uint64_t, const android::wp<::android::hidl::base::V1_0::IBase>&);
    ClientManagerType & cm;
};

enum class EnnInterfaceStatus {
    StatusInit,
    StatusActive,
    StatusAbortBefore,
    StatusOff,
};

struct EnnInterface : public V1_0::IEnnInterface {
    // Methods from ::vendor::samsung_slsi::hardware::enn::V1_0::IEnnInterface follow.
    EnnInterface();
    virtual ~EnnInterface();

    /* model based interface */
    Return<int32_t> init(const sp<::vendor::samsung_slsi::hardware::enn::V1_0::IEnnCallback>& cb_sp) override;
    Return<int32_t> deinit() override;
    Return<void> open_model(const ::vendor::samsung_slsi::hardware::enn::V1_0::LoadParameter& load_param,
                            open_model_cb _hidl_cb) override;

    Return<int32_t> close_model(uint64_t model_id) override;
    Return<uint64_t> commit_execution_data(
        const uint64_t model_id, const ::vendor::samsung_slsi::hardware::enn::V1_0::InferenceData& exec_data) override;
    Return<int32_t> execute_model(const hidl_vec<uint64_t>& exec_id_list) override;

    Return<void> custom_interface(const uint32_t identifier,
                                    const ::vendor::samsung_slsi::hardware::enn::V1_0::GeneralParameter &parameter,
                                    custom_interface_cb _hidl_cb) override;

  private:
    void ClientMonitorThreadFunction(void);
    std::thread clientMonitorThread;
    std::condition_variable monitor_thread_cv;
    const int check_interval = 30;  // monitor client status every 2 seconds

    ClientManagerType client_manager;
    std::map<int32_t, sp<EnnClientRecipient>> client_recipient_map;
    std::mutex serviceInterfaceMutex;
    EnnInterfaceStatus status;
};

// FIXME: most likely delete, this is only for passthrough implementations
// extern "C" IEnnInterface* HIDL_FETCH_IEnnInterface(const char* name);

}  // namespace vendor::samsung_slsi::hardware::enn::implementation
