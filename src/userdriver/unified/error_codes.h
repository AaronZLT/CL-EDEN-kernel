/*
 * Copyright 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file error_codes.h
 * @brief UD error codes definition
 */
#ifndef USERDRIVER_UNIFIED_ERROR_CODES_H_
#define USERDRIVER_UNIFIED_ERROR_CODES_H_

typedef enum {
    MODEL_ALREADY_ADDED = 1,
} cb_ret_t;

/**
 * error codes.
 */
typedef enum _eden_error_code {
    //* common
    EDEN_SUCCESS = 0,

    //* npu user driver related
    UD_FAILED,
    UD_INVALID_ARG,
    UD_UNKNOWN_ARG,
    UD_INC_COUNT_FAIL,
    UD_DEC_COUNT_FAIL,

    UD_INIT_FAILED,
    UD_THREAD_CREATE_FAILED,
    UD_MALLOC_FAILED,
    UD_ION_FAILED,
    UD_STRING_COPY_FAILED,

    UD_NCP_ADD_FAILED,
    UD_NCP_UNLOAD_FAILED,
    UD_WAKEUP_SIGNALING_FAILED,
    UD_NOT_FOUND_MODEL,
    UD_NOT_FOUND_REQUEST,

    UD_NOT_FOUND_MATCHED_EMA,
    UD_FEATUREMAP_NOT_MATCHED,
    UD_GET_DRV_INFO_FAILED,
    UD_EMERGENCY_RECOVERY_FAILED,
    UD_OPEN_MODEL_FAILED,

    UD_CLOSE_MODEL_FAILED,
    UD_SRAM_FULL,
    UD_MODEL_ALREADY_OPENED,

    //* link-vs4l related
    LINK_FAILED,
    LINK_OPEN_FAILED,

    LINK_SRAM_FULL,
    LINK_VS4L_BUF_FULL,
    LINK_NO_MORE_REQ,
    LINK_EMERGENCY_DETECTED,
    LINK_APPLY_PREFERENCE_FAILED,
    LINK_APPLY_PERFORMANCE_FAILED,
    LINK_APPLY_PRESET_FAILED,

    LINK_NO_WORK_MID_BIG_CORES,
    LINK_NOT_SUPPORT_FILEPATH_FOR_NCP_EMBEDDING,
    LINK_KERNEL_NOT_MATCHED,
    LINK_NO_VS4L_CONTAINER,
    LINK_STREAM_OFF_FAILED,

    LINK_FD_CLOSE_FAILED,

    // Critical error!
    LINK_NPU_HW_TIMEOUT,

    LINK_ALLOCATE_PROFILER_FAILED,

    TOOL_FAILED,

    //* Android HAL related
    //* UD_HAL_xxx

    EDEN_MAX_ERR,
    EDEN_LIMIT_ERR = 0xffffffff,
} eden_ret_t;
// Nice to have: TODO(jungho7.kim, TBD): replace eden_ret_t with EnnReturn

static const char *acc_err_str[] = {
    //* common
    "EDEN_SUCCESS",

    //* npu user driver related
    "UD_FAILED",
    "UD_INVALID_ARG",
    "UD_UNKNOWN_ARG",
    "UD_INC_COUNT_FAIL",
    "UD_DEC_COUNT_FAIL",

    "UD_INIT_FAILED",
    "UD_THREAD_CREATE_FAILED",
    "UD_MALLOC_FAILED",
    "UD_ION_FAILED",
    "UD_STRING_COPY_FAILED",

    "UD_NCP_ADD_FAILED",
    "UD_NCP_UNLOAD_FAILED",
    "UD_WAKEUP_SIGNALING_FAILED",
    "UD_NOT_FOUND_MODEL",
    "UD_NOT_FOUND_REQUEST",

    "UD_NOT_FOUND_MATCHED_EMA",
    "UD_FEATUREMAP_NOT_MATCHED",
    "UD_GET_DRV_INFO_FAILED",
    "UD_EMERGENCY_RECOVERY_FAILED",
    "UD_OPEN_MODEL_FAILED",

    "UD_CLOSE_MODEL_FAILED",
    "UD_SRAM_FULL",
    "UD_MODEL_ALREADY_OPENED",

    //* link-vs4l related
    "LINK_FAILED",
    "LINK_OPEN_FAILED",

    "LINK_SRAM_FULL",
    "LINK_VS4L_BUF_FULL",
    "LINK_NO_MORE_REQ",
    "LINK_EMERGENCY_DETECTED",
    "LINK_APPLY_PREFERENCE_FAILED",
    "LINK_APPLY_PERFORMANCE_FAILED",
    "LINK_APPLY_PRESET_FAILED",

    "LINK_NO_WORK_MID_BIG_CORES",
    "LINK_NOT_SUPPORT_FILEPATH_FOR_NCP_EMBEDDING",
    "LINK_KERNEL_NOT_MATCHED",
    "LINK_NO_VS4L_CONTAINER",
    "LINK_STREAM_OFF_FAILED",

    "LINK_FD_CLOSE_FAILED",

    // Critical error!
    "LINK_NPU_HW_TIMEOUT",

    "LINK_ALLOCATE_PROFILER_FAILED",

    "TOOL_FAILED",

    //* Android HAL related
    //* UD_HAL_xxx

    "EDEN_MAX_ERR",
};

#endif  // USERDRIVER_UNIFIED_ERROR_CODES_H_
