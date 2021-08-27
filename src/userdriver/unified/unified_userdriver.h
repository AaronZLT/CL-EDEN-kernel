/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or
 * distributed, transmitted, transcribed, stored in a retrieval system or
 * translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed to third parties
 * without the express written permission of Samsung Electronics.
 */

/**
 * @file    unified_userdriver.h
 * @brief   This is common ENN Unified Userdriver API
 * @details This header defines ENN Unified Userdriver API.
 */
#ifndef USERDRIVER_UNIFIED_UNIFIED_USERDRIVER_H_
#define USERDRIVER_UNIFIED_UNIFIED_USERDRIVER_H_

#include <unordered_map>
#include <vector>
#include "userdriver/common/UserDriver.h"
#include "userdriver/common/operator_interfaces/userdriver_operator.h"
#include "userdriver/unified/link_vs4l.h"  // link
#include "userdriver/unified/npu_userdriver.h"
#include "userdriver/unified/dsp_userdriver.h"

namespace enn {
namespace ud {
namespace unified {

using namespace enn::ud::npu;
using namespace enn::ud::dsp;

class UnifiedUserDriver : public UserDriver {
public:
    enum class UnifiedUdStatus { NONE, INITIALIZED, SHUTDOWNED };

    static UnifiedUserDriver& get_instance(void);
    ~UnifiedUserDriver(void);

    EnnReturn Initialize(void) override;
    EnnReturn OpenSubGraph(const model::component::OperatorList& operator_list) override;
    EnnReturn PrepareSubGraph(const enn::runtime::ExecutableOperatorList& executable_operator_list) override;
    EnnReturn ExecuteSubGraph(const enn::runtime::OperatorListExecuteRequest& operator_list_execute_request) override;
    EnnReturn CloseSubGraph(const model::component::OperatorList& operator_list) override;
    EnnReturn Deinitialize(void) override;
    EnnReturn set_unified_ud_status(UnifiedUdStatus unified_ud_status);
    UnifiedUdStatus get_unified_ud_status() const { return unified_ud_status_; }
    EnnReturn check_validity_op_list(const model::component::OperatorList& op_list);
    EnnReturn check_validity_op(std::shared_ptr<model::component::Operator> op);
    EnnReturn check_validity_op_option(const tflite::v2::ENN_UNIFIED_DEVICE_BinaryOptions* option);

private:
    UnifiedUserDriver(void) : UserDriver(Unified_UD),
                        npu_ud(nullptr), dsp_ud(nullptr), unified_ud_status_(UnifiedUdStatus::NONE) {
        ENN_DBG_PRINT("started\n");
    }

    NpuUserDriver* npu_ud;
    DspUserDriver* dsp_ud;
    UnifiedUdStatus unified_ud_status_;
    std::mutex mutex_unified_ud_status;
    // accelerator_device acc_ = ACCELERATOR_Unified;
};

}  // namespace unified
}  // namespace ud
}  // namespace enn

#endif  // USERDRIVER_UNIFIED_UNIFIED_USERDRIVER_H_
