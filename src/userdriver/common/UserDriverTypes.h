/**
 * Copyright (C) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed,
 * transmitted, transcribed, stored in a retrieval system or translated into any human or
 * computer language in any form by any means, electronic, mechanical, manual or
 * otherwise or disclosed to third parties without the express written permission of
 * Samsung Electronics.
 */

/**
 * @file    UserDriverTypes.h
 * @brief   This is common ENN Userdriver types
 * @details This header defines ENN Userdriver types.
 *          They are compatible with C-language.
 */

#ifndef USERDRIVER_COMMON_USERDRIVERTYPES_H_
#define USERDRIVER_COMMON_USERDRIVERTYPES_H_

#include "client/enn_api-type.h"

#ifdef __cplusplus
extern "C" {
#endif

static const char* const CPU_UD = "CpuUserDriver";
static const char* const GPU_UD = "GpuUserDriver";
static const char* const NPU_UD = "NpuUserDriver";
static const char* const DSP_UD = "DspUserDriver";
static const char* const Unified_UD = "UnifiedUserDriver";

typedef struct _drv_info {
    int32_t NCP_version;
    int32_t Mailbox_version;
    int32_t CMD_version;
    int32_t API_version;
    int32_t SOC_version;
    int32_t CpuOfMidCluster;
} drv_info_t;

typedef enum _request_state {
    REQ_UNKNOWN = 0,
    REQ_FAILED,
    REQ_QUEUED,
    REQ_DISPATCHED,
    REQ_DONE,
    REQ_SIZE
} req_state_t;

typedef struct _shape {
    uint32_t number;
    uint32_t channel;
    uint32_t height;
    uint32_t width;
    uint32_t type_size;
    uint32_t get_size() { return number * channel * height * width * type_size; }
    std::string get_string() {
        std::string shape_string;
        shape_string.append(std::to_string(number)).append(",")
                .append(std::to_string(channel)).append(",")
                .append(std::to_string(height)).append(",")
                .append(std::to_string(width)).append(",")
                .append(std::to_string(type_size)).append(",");
        return shape_string;
    }
} shape_t;

typedef enum _dev_state {
    DEVICE_UNKNOWN = 0,
    DEVICE_POWER_OFF,
    DEVICE_POWER_ON,
    DEVICE_EMERGENCY_RECOVERY,
    DEVICE_INITIALIZED,
    DEVICE_RUNNING,
    DEVICE_SUSPENDED,
    DEVICE_RESUMED,
    DEVICE_SHUTDOWNS,
    DEVICE_SIZE
} dev_state_t;

typedef enum {
    EDEN_NN_API = 0,
    ANDROID_NN_API = 1,
    CAFFE2_NN_API = 2,
    APICOUNT,
} NnApiType;

#ifdef __cplusplus
}
#endif  // extern C

#endif  // USERDRIVER_COMMON_USERDRIVERTYPES_H_
