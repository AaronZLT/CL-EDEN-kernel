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

#ifndef SRC_COMMON_ENN_UTILS_H_
#define SRC_COMMON_ENN_UTILS_H_

#include "common/enn_common_type.h"
#include "common/enn_debug_zone.h"
#include "common/enn_debug.h"

#include <cstdlib>
#include <climits>
#include <string>
#include <memory>
#include <map>
#include <mutex>
#include <dlfcn.h>
#include <unistd.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <iostream>
#include <cmath>

#ifdef __ANDROID__
#include <cutils/properties.h>
#else
#include <cstdarg>
#endif

namespace enn {
namespace util {

constexpr float BUFFER_COMPARE_THRESHOLD = 0.000100;
constexpr float GPU_INCEPTIONV3_FP16_THRESHOLD = 0.017000;
constexpr size_t FIRST = 0;

/* file related utilities */

// @param result if *result == null, the function tries to allocate buffer. In this case, caller should free the buffer
extern EnnReturn import_file_to_mem(const char *filename, char **result, uint32_t *out_size, uint32_t size = 0,
                                    uint32_t offset = 0);
extern EnnReturn get_file_size(const char *filename, uint32_t *out_size);
extern EnnReturn memory_compare(const char *mem1, const char *mem2, size_t num);
extern EnnReturn get_environment_property(const std::string & prop_name, uint64_t *val);
extern EnnReturn export_mem_to_file(const char *filename, const void *va, uint32_t size);
extern void      show_raw_memory_to_hex(uint8_t *va, uint32_t size, const int line_max, const int32_t size_max = 0);

extern EnnReturn sys_util_set_sched_affinity(uint32_t mask);
extern EnnReturn ofi_adjust_thread_priority_urgent(void);

/*** evaluation tools ***/
/**
 * @brief compare two memory buffers
 *
 * @tparam CompareUnitType : float, int, uint8,..
 * @param goldenBuffer source buffer addr.
 * @param TargetBufferAddr target buffer addr
 * @param buffer_size size of buffer
 * @param compare_buffer_out if not nullptr, save comparison map
 * @param threshold threshold (0, 0.001)
 * @return int32_t
 */

template <typename T, std::enable_if_t<std::is_integral<T>::value, bool> = true>
inline static void show_diff(const char *title, int idx, T s, T t, T diff, T threshold) {
    printf("[%d] %s: golden %d vs. target %d = %d < threshold %d\n", idx, title, s, t, diff, threshold);
}

template <typename T, std::enable_if_t<std::is_floating_point<T>::value, bool> = true>
inline static void show_diff(const char *title, int idx, T s, T t, T diff, T threshold) {
    printf("[%d] %s: golden %f vs. target %f = %f < threshold %f\n", idx, title, s, t, diff, threshold);
}

inline static void show_diff(const char *title, int idx, uint8_t s, uint8_t t, uint8_t diff, uint8_t threshold) {
    printf("[%d] %s: golden %d vs. target %d = %d < threshold %d\n", idx, title, s, t, diff, threshold);
}

/**
 * @brief Comparer: compare two buffers between original buffers and target buffers that has void* type.
 *
 * @tparam CompareUnitType element type of original/target buffers, type of threshold
 * @param goldenBuffer reference buffer
 * @param TargetBufferAddr buffer to compare
 * @param buffer_size physical buffer size (in bytes)
 * @param compare_buffer_out default nullptr, if a user put uint8_t type array, the function fills each pixels result (0 or 0xFF)
 * @param threshold threshold. if a user wants to compare two pixels are same or not, set threshold to zero
 * @param is_debug if true, the functions shows a progress of comparison
 * @return int32_t number of over-threshold-pixels
 */
template <typename CompareUnitType>
int32_t CompareBuffersWithThreshold(void *goldenBuffer, void *TargetBufferAddr, int32_t buffer_size,
                                    uint8_t *compare_buffer_out = nullptr, CompareUnitType threshold = 0,
                                    bool is_debug = false) {
    CompareUnitType *target_p = reinterpret_cast<CompareUnitType *>(TargetBufferAddr);
    CompareUnitType *source_p = reinterpret_cast<CompareUnitType *>(goldenBuffer);
    bool is_save = (compare_buffer_out != nullptr);
    int diff_cnt = 0;

    if (is_save) {
        for (size_t i = 0; i < buffer_size / sizeof(CompareUnitType); i++) compare_buffer_out[i] = 0;
    }

    for (size_t i = 0; i < buffer_size / sizeof(CompareUnitType); i++) {
        auto diff = std::abs(source_p[i] - target_p[i]);
        if (diff > threshold) {
            diff_cnt++;
            if (is_save) {
                compare_buffer_out[i] = 0xFF;
            } else {
                show_diff("Different!", i, source_p[i], target_p[i], diff, threshold);
            }
        }
        if (is_debug) {
            show_diff("Check", i, source_p[i], target_p[i], diff, threshold);
        }
    }

    return diff_cnt;
}

/**
 * @brief Get the tid or pid from current process
 *
 * @return pid_t
 */
extern pid_t get_tid(void);
extern pid_t get_pid(void);

}  // namespace util
}  // namespace enn

// macros for checking condition + msg + return
#define CHECK_AND_RETURN_ERR(cond, ret, message, ...)         \
    do {                                                      \
        if (cond) {                                           \
            ENN_ERR_PRINT(DBG_PREFIX message, ##__VA_ARGS__); \
            return ret;                                       \
        }                                                     \
    } while (0)

#define CHECK_AND_RETURN_WARN(cond, ret, message, ...)         \
    do {                                                       \
        if (cond) {                                            \
            ENN_WARN_PRINT(DBG_PREFIX message, ##__VA_ARGS__); \
            return ret;                                        \
        }                                                      \
    } while (0)

#define CHECK_AND_RETURN_VOID(cond, message, ...)              \
    do {                                                       \
        if (cond) {                                            \
            ENN_WARN_PRINT(DBG_PREFIX message, ##__VA_ARGS__); \
            return;                                            \
        }                                                      \
    } while (0)

#endif  // SRC_COMMON_ENN_UTILS_H_

