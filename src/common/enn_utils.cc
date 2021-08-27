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

#include "common/enn_utils.h"
#include "common/enn_utils_buffer.hpp"
#include <cstring>
#include <sys/syscall.h>

namespace enn {
namespace util {

/* global filter */
constexpr const char ANSI_FGCOLOR_RED[] = "[31m";
constexpr const char ANSI_FGCOLOR_YELLOW[] = "[33m";
constexpr const char ANSI_FGCOLOR_CYAN[] = "[36m";
constexpr const char POSTFIX_PRINT[] = "[m";

EnnReturn get_file_size(const char *filename, uint32_t *out_size) {
    CHECK_AND_RETURN_ERR(out_size == nullptr, ENN_RET_INVAL, "2nd parameter is nullptr\n");
    auto size = enn::util::FileBufferReader(std::string(filename)).get_size();
    CHECK_AND_RETURN_ERR(size == 0, ENN_RET_INVAL, "File Loading Error\n");

    *out_size = size;

    return ENN_RET_SUCCESS;
}

EnnReturn import_file_to_mem(const char *filename, char **result, uint32_t *out_size, uint32_t size, uint32_t offset) {
    CHECK_AND_RETURN_ERR(result == nullptr, ENN_RET_INVAL, "2nd parameter is nullptr\n");
    ENN_INFO_PRINT("Get filename(%s), result(%p), size(%u), offset(%u)\n", filename, *result, size, offset);

    uint32_t file_size;
    auto ret = enn::util::get_file_size(filename, &file_size);
    CHECK_AND_RETURN_ERR(ret, ret, "File open error: %s\n", filename);
    CHECK_AND_RETURN_ERR(size + offset > static_cast<uint32_t>(file_size), ENN_RET_FAILED,
                         "Size(%d), Offset(%d) is bigger than file size(%d)\n", size, offset, file_size);
    CHECK_AND_RETURN_ERR(size == 0 && offset > 0, ENN_RET_FAILED, "Offset should be zero if size is full(default)\n");

    int load_size = size == 0 ? file_size : size;
    if (*result == nullptr)
        *result = reinterpret_cast<char *>(malloc(load_size + 1));
    if (out_size)
        *out_size = load_size;

    return enn::util::FileBufferReader(std::string(filename)).copy_buffer(*result, load_size, offset);
}

/**
 * @param va memory address to display
 * @param size size of va
 * @param line_max max number of hex to display in a line
 * @param size_max max number of hex to display in whole values
 */
void show_raw_memory_to_hex(uint8_t *va, uint32_t size, const int line_max, const uint32_t size_max) {
#ifndef ENN_BUILD_RELEASE
    char line_tmp[100] = {0,};
    int int_size = static_cast<int>(size);
    int max = (size_max == 0 ? int_size : (int_size < size_max ? int_size : size_max));
    int idx = sprintf(line_tmp, "[%p] ", va);  // prefix of line
    int i = 0;  // idx records current location of print line
    for (; i < max; ++i) {
        idx += sprintf(&(line_tmp[idx]), "%02X ", va[i]);
        if (i % line_max == (line_max - 1)) {
            // if new line is required, flush print --> and record prefix print
            line_tmp[idx] = 0;
            ENN_TST_COUT << line_tmp << std::endl;
            idx = 0;
            idx = sprintf(line_tmp, "[%p] ", &(va[i]));
        }
    }
    if (i % line_max != 0) {
        ENN_TST_COUT << line_tmp << std::endl;
    }
#endif
}

EnnReturn export_mem_to_file(const char *filename, const void *va, uint32_t size) {
    size_t ret_cnt;

    ENN_DBG_PRINT("DEBUG:: Export memory to file: name(%s) va(%p), size(%d)\n", filename, va, size);
    CHECK_AND_RETURN_ERR(filename == NULL, ENN_RET_INVAL, "Null Filename!\n");
    CHECK_AND_RETURN_ERR(va == NULL, ENN_RET_INVAL, "Null address!\n");
    CHECK_AND_RETURN_ERR(size <= 0, ENN_RET_INVAL, "Size is incorrect(%d)!\n", size);

    FILE *fp = fopen(filename, "wb");
    CHECK_AND_RETURN_ERR(fp == NULL, ENN_RET_INVAL, "File Open Failed(%s)!!\n", filename);

    ret_cnt = fwrite(va, size, 1, fp);
    if (ret_cnt <= 0) {
        ENN_ERR_PRINT("FileWrite Failed!!(%zu)\n", ret_cnt);
        fclose(fp);
        return ENN_RET_INVAL;
    }

    ENN_DBG_PRINT("DEBUG:: File Save Completed.\n");
    fclose(fp);

    return ENN_RET_SUCCESS;
}

EnnReturn memory_compare(const char *mem1, const char *mem2, size_t num) {
    return (memcmp(mem1, mem2, num) == 0 ? ENN_RET_SUCCESS : ENN_RET_FAILED);
}

EnnReturn get_environment_property(const std::string &prop_name, uint64_t *val) {
#ifdef __ANDROID__
    int64_t ret = property_get_int64(prop_name.c_str(), -1);
    if (ret >= 0) {
        *val = ret;
        return ENN_RET_SUCCESS;
    }
#else  // TODO(hoon98.choi, TBD): get environment variable in linux
    ENN_UNUSED(prop_name);
    ENN_UNUSED(val);
#endif
    return ENN_RET_IO;
}


/* device dependent codes */
pid_t get_tid(void) {
#ifdef __ANDROID__
    return gettid();
#else
    return (pid_t)syscall(SYS_gettid);
#endif
}

/* system info related APIs */
pid_t get_pid(void) {
#ifdef __ANDROID__
    return getpid();
#else
    return (pid_t)syscall(SYS_getpid);
#endif
}


EnnReturn sys_util_set_sched_affinity(uint32_t mask) {
    auto tid = get_tid();
    int32_t syscallres = syscall(__NR_sched_setaffinity, tid, sizeof(mask), &mask);
    ENN_INFO_PRINT_FORCE("Sched Affinity: set tid %d to 0x%X (ret: %d)\n", tid, mask, syscallres);
    if (syscallres)    return ENN_RET_FAILED;
    return ENN_RET_SUCCESS;
}


#define PRIORITY_URGENT (-20)
EnnReturn ofi_adjust_thread_priority_urgent(void) {
    int prio;
    errno = 0;
    prio = syscall(__NR_getpriority, PRIO_PROCESS, 0);

    if (prio == -1 && errno)
        return ENN_RET_FAILED;

    if (prio != PRIORITY_URGENT) {
        int32_t syscallres = syscall(__NR_setpriority, PRIO_PROCESS, 0, PRIORITY_URGENT);
        if (syscallres) {
            return ENN_RET_FAILED;
        }
    }

    ENN_INFO_PRINT_FORCE("Priority set to 0x%X\n", prio);

    return ENN_RET_SUCCESS;
}


}  // namespace util
}  // namespace enn

