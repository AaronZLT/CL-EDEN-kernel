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

#ifndef SRC_MEDIUM_ENN_MEDIUM_UTILS_H_
#define SRC_MEDIUM_ENN_MEDIUM_UTILS_H_

#ifdef ENN_MEDIUM_IF_HIDL
#include <hidl/HidlTransportSupport.h>
#include <hidlmemory/mapping.h>
#include <hwbinder/IPCThreadState.h>
#endif

#include "common/enn_utils.h"

/**
 * @brief Get the caller pid object in HIDL
 *        If current process is not callee or non-HIDL, returns current PID
 * @return pid, negative value if failed
 */

namespace enn {
namespace util {

static inline int get_caller_pid() {
#ifdef ENN_MEDIUM_IF_HIDL
    return ::android::hardware::IPCThreadState::self()->getCallingPid();
#else
    return get_pid();  // if lib mode, caller and callee are in a same process
#endif
}

}  // namespace util
}  // namespace enn

#endif  //  SRC_MEDIUM_ENN_MEDIUM_UTILS_H_
