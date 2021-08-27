/*
 * Copyright (C) 2020 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or
 * distributed, transmitted, transcribed, stored in a retrieval system or
 * translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed to third parties
 * without the express written permission of Samsung Electronics.
 */

#ifndef SRC_COMMON_INCLUDE_ENN_DEBUG_ZONE_H_
#define SRC_COMMON_INCLUDE_ENN_DEBUG_ZONE_H_

/**
 * @brief define debug zone enums. this file allows to share with external tools
 *
 * @file enn_debug.zone.h
 * @author Hoon Choi (hoon98.choi@samsung.com)
 * @date 2020-12-14
 */

#undef LOG_TAG
#define LOG_TAG "ENN_FRAMEWORK"

#ifdef __DSP_NDK_BUILD__
#include <android/log.h>
#elif defined(__ANDROID__)
#include <log/log.h>
#else
#include "test/internal/enn_test_driver_debug_utils.h"
#endif /* BUILD_X86 */

namespace enn {
namespace debug {

enum class DbgPriority {
    kDebug = ANDROID_LOG_DEBUG,
    kInfo = ANDROID_LOG_INFO,
    kWarn = ANDROID_LOG_WARN,
    kError = ANDROID_LOG_ERROR,
    kSIZE,
};

enum class DbgPrintOption {
    kEnnPrintFalse = 0,
    kEnnPrintTrue = 1,
    kEnnPrintRelease = 2,
};

// sync with DbgPrintManager::init() in enn_debug_utils.cc, enn_print in enn_debug_utils.h
// if wants to append debug flags, please insert it between kUser and kFileDumpSession
enum class DbgPartition {
    kError = 0,
    kWarning,
    kInfo,
    kDebug,
    kTest,
    kMemory,
    kUser,
    kFileDumpSession = 20,  // FileDump flag for debug
    kSIZE,
};

}
}

#endif  // SRC_COMMON_INCLUDE_ENN_DEBUG_ZONE_H_
