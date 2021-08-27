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

#ifndef SRC_COMMON_INCLUDE_ENN_COMMON_TYPE_H_
#define SRC_COMMON_INCLUDE_ENN_COMMON_TYPE_H_

/**
 * @file enn_common_type.h
 * @author Hoon Choi (hoon98.choi@samsung.com)
 * @brief define types commonly used, but not expose to user
 * @version 0.1
 * @date 2020-12-14
 */

#include <cstdint>

#include "client/enn_api-type.h"


#ifdef __ANDROID__
#define PREFIX_RED_STR    ""
#define PREFIX_YELLOW_STR ""
#define PREFIX_CYAN_STR   ""
#define PREFIX_STR        ""
#define POSTFIX_STR       ""
#else
/* define color prefix / postfix */
#define PREFIX_RED_STR    "[31m"
#define PREFIX_YELLOW_STR "[33m"
#define PREFIX_CYAN_STR   "[36m"
#define PREFIX_STR        ""
#define POSTFIX_STR       "[m"
#endif

/* Log tag */
#define LOG_TAG "ENN_FRAMEWORK"

/* header */
#define BIT_MASK(x)    (((x) >= sizeof(unsigned) * CHAR_BIT) ? (unsigned)-1 : (1U << (x)) - 1)
#define ALIGN_UP(x, y) (((x) + ((y)-1)) & ~((y)-1))
#define ZONE_BIT(zone) (1U << (zone))
#define DBG_PREFIX ""
#define ENN_UNUSED(x)  (void)(x)

using EnnModelId = EnnModelId;  // from api-type.h
using EnnRet = EnnReturn;
using EnnExecuteModelId = EnnModelId;

/* Preferences to stream */
struct EnnPreference {
    uint32_t preset_id;
    uint32_t pref_mode;         // for EDEN backward compatibility
    uint32_t target_latency;    // for DVFS hint
    uint32_t tile_num;          // for batch processing hint
    uint32_t core_affinity;
    uint32_t priority;
    uint32_t custom[2];
};

enum class CustomFunctionTypeId: uint32_t {
    NOT_DEFINE = 0,
    GET_DSP_SESSION_ID = 10,
    SECURE_INITIALIZE = 11,
    SECURE_DEINITIALIZE = 12,
    GET_DEVICE_SW_VERSION = 13,
};

#endif  // SRC_COMMON_INCLUDE_ENN_COMMON_TYPE_H_
