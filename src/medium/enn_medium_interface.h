/**
 * @file enn_medium_interface.h
 * @author Hoon Choi (hoon98.choi@samsung.com)
 * @brief Define interface for medium. This file tries to abstract service call
 * @version 0.1
 * @date 2020-12-29
 *
 * @copyright
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

#ifndef SRC_MEDIUM_ENN_MEDIUM_INTERFACE_H_
#define SRC_MEDIUM_ENN_MEDIUM_INTERFACE_H_

#include <iostream>
#include <unordered_map>
#include <vector>

#include <unistd.h>
#include <memory>

#include "common/enn_common_type.h"
#include "common/enn_debug.h"
#include "common/enn_utils.h"
#include "common/enn_memory_manager-type.h"

#ifdef ENN_MEDIUM_IF_HIDL
#define HIDL_IF(st) st
#define LIB_IF(st)
#else
#define HIDL_IF(st)
#define LIB_IF(st) st
#endif

#ifdef ENN_MEDIUM_IF_HIDL
#include "medium/types.hal.hidl.h"

struct EnnCallback : public IEnnCallback {
    // Methods from ::vendor::samsung_slsi::hardware::enn::V1_0::IEnnCallback follow.
    android::hardware::Return<uint32_t> executeCallback(uint64_t cbAddr, uint64_t value, const hidl_handle& hd) {
        ENN_UNUSED(cbAddr);
        ENN_UNUSED(value);
        ENN_UNUSED(hd);
        return uint32_t{0};
    }
    android::hardware::Return<bool> isAlive() {
        ENN_TST_PRINT("This is alive.\n");
        return true;
    }
};

#else
#include "types.hal.h"
#include "runtime/engine.hpp"
#endif

namespace enn {
namespace interface {

constexpr uint32_t IF_MAGIC = 0x1A2B3C4D;

/* global enum for hidl parameter */
enum InterfaceType : uint32_t {
    kEnnIfTypeInit = 101,
    kEnnIfTypeDeinit,
};

enum class emif_testcase : uint32_t {
    emif_test_none = 0,
    emif_test_gen = 1,
    emif_test_scls = 2,
    emif_test_max,
};

class EnnMediumInterface {
    using DeviceSessionID = int32_t;   // NOTE: sync with engine.hpp, enn::runtime::Engine::SessionID
public:
    EnnMediumInterface();
    ~EnnMediumInterface();

    void GetServiceInterface();
    void PutServiceInterface();

    EnnReturn init();
    EnnReturn deinit();
    EnnReturn open_model(std::vector<std::shared_ptr<EnnBufferCore>> buf_list, uint32_t model_type, std::vector<uint32_t> &,
                         std::shared_ptr<SessionBufInfo> ret_info);
    EnnReturn close_model(const EnnModelId model_id);

    EnnExecuteModelId commit_execution_data(const EnnModelId model_id, const InferenceData &);
    EnnReturn execute_model(const std::vector<EnnModelId> &);

    DeviceSessionID get_dsp_session_id(const EnnModelId model_id);

  private:
    HIDL_IF(android::sp<IEnnInterface> service);
    HIDL_IF(android::sp<EnnCallback> enn_callback);
    HIDL_IF(android::sp<EnnCallback> cb_sp);
    LIB_IF(::enn::runtime::Engine *service);
};

}  // namespace interface
}  // namespace enn

#endif  // SRC_MEDIUM_ENN_MEDIUM_INTERFACE_H_
