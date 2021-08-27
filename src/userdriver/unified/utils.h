/*
 * Copyright (C) 2017 The Android Open Source Project
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

/** @file utils.h
    @brief utils header
*/

#ifndef COMMON_UTILS_H_
#define COMMON_UTILS_H_

#include "common/compiler.h"
#include "userdriver/unified/error_codes.h"
#include "userdriver/common/eden_osal/log.h"

#define SIZE_4K 0x1000

// Macro to check if the input parameter for operation are valid or not.
#define _CHK_TRUE_RET_MSG(expected_cond, msg)                  \
    ({                                                                  \
    int ret = (expected_cond);                                          \
    if (likely(ret)) {                                                          \
        ENN_INFO_PRINT("true: [%s]\n", msg);            \
    } else {                                                            \
        ENN_ERR_PRINT_FORCE("false: [%s]\n", msg);        \
    }                                                                   \
    ret;})

#define _CHK_RET_MSG(v, msg)                                  \
    ({                                                                  \
    int ret = (v);                                                      \
    if (likely(!ret)) {                                                         \
        ENN_INFO_PRINT("ok: %s\n", msg);              \
    } else {                                                            \
        ENN_ERR_PRINT_FORCE("failed: %s [%s]\n",           \
                msg, (ret < EDEN_MAX_ERR && ret > 0) ?                   \
                acc_err_str[ret] : "unknown");                          \
    }                                                                   \
    ret;})

#define _CHK_RET(v)                                           \
    ({                                                                  \
    int ret = (v);                                                      \
    if (likely(!ret)) {                                                         \
        ENN_INFO_PRINT("ok\n");                       \
    } else {                                                            \
        ENN_ERR_PRINT_FORCE("failed: [%s]\n",              \
                (ret < EDEN_MAX_ERR && ret > 0) ?                        \
                acc_err_str[ret] : "unknown");                          \
    }                                                                   \
    ret;})

#define _CHK_MGS(v, msg)                                     \
    do {                                                               \
        int ret = (v);                                                 \
        if (likely(!ret)) {                                                    \
            ENN_INFO_PRINT("ok: %s\n", msg);         \
        } else {                                                       \
            ENN_ERR_PRINT_FORCE("failed: %s [%s]\n",      \
                    msg, (ret < EDEN_MAX_ERR && ret > 0) ?              \
                    acc_err_str[ret] : "unknown");                     \
        }                                                              \
    } while (0);

#define _CHK(v) _CHK_MGS(v, "")

#define _MUTEX_LOCK(mutex)                                        \
    do {                                                                    \
        if (unlikely(_CHK_RET_MSG(os_mutex_lock(&mutex), "os_mutex_lock")))   \
            return UD_FAILED;                                              \
    } while (0);

#define _MUTEX_UNLOCK(mutex)                                         \
    do {                                                                       \
        if (unlikely(_CHK_RET_MSG(os_mutex_unlock(&mutex), "os_mutex_unlock")))  \
            return UD_FAILED;                                                 \
    } while (0);
#define _MUTEX_LOCK_NULL(mutex)                                        \
    do {                                                                    \
        if (unlikely(_CHK_RET_MSG(os_mutex_lock(&mutex), "os_mutex_lock")))   \
            return NULL;                                              \
    } while (0);

#define _MUTEX_UNLOCK_NULL(mutex)                                         \
    do {                                                                       \
        if (unlikely(_CHK_RET_MSG(os_mutex_unlock(&mutex), "os_mutex_unlock")))  \
            return NULL;                                                 \
    } while (0);

#define _MUTEX_LOCK_LINK(mutex)                                    \
    do {                                                                     \
        if (unlikely(_CHK_RET_MSG(os_mutex_lock(&mutex), "os_mutex_lock in link layer")))  \
        return LINK_FAILED;                                              \
    } while (0);

#define _MUTEX_UNLOCK_LINK(mutex)                                      \
    do {                                                                         \
        if (unlikely(_CHK_RET_MSG(os_mutex_unlock(&mutex), "os_mutex_unlock in link layer")))  \
        return LINK_FAILED;                                                  \
    } while (0);


inline int32_t get_cell_format_size(int32_t size, int cell_size) {
    return ((size + (cell_size - 1)) / cell_size) * cell_size;
}
inline int32_t get_aligned_buffer_size(int32_t alignment, int32_t size) {
    return size % alignment ? (((size / alignment) + 1) * alignment) : size;
}

#endif  // COMMON_UTILS_H_
