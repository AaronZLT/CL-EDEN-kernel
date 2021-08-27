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

#include "common/enn_debug.h"
#include "common/enn_utils.h"
#include <string>
#include <cstring>
#include <memory>

namespace enn {
namespace debug {

::std::unique_ptr<DbgPrintManager> DbgPrintManager::_instance = nullptr;
::std::mutex DbgPrintManager::_mutex;

void DbgPrintManager::set_mask(MaskType mask) {
    print_mask = mask;
}

const std::map<DbgPartition, DebugPartitionInfo> &DbgPrintManager::get_debug_zone_info() {
    return debug_zone_info;
}

const MaskType &DbgPrintManager::get_print_mask() {
    return print_mask;
}

EnnReturn DbgPrintManager::enn_set_debug_zone(enum DbgPartition zone) {
    MaskType zone_value = static_cast<MaskType>(zone);
    if (zone_value < static_cast<MaskType>(DbgPartition::kSIZE)) {
        print_mask |= ZONE_BIT_MASK(zone_value);
        return ENN_RET_SUCCESS;
    }

    return ENN_RET_INVAL;
}

EnnReturn DbgPrintManager::enn_clr_debug_zone(enum DbgPartition zone) {
    MaskType zone_value = static_cast<MaskType>(zone);
    if (zone_value < static_cast<MaskType>(DbgPartition::kSIZE)) {
        print_mask &= ~ZONE_BIT_MASK(zone_value);
        return ENN_RET_SUCCESS;
    }

    return ENN_RET_INVAL;
}

/* members for singleton */
DbgPrintManager::DbgPrintManager() {
    init();
}

void DbgPrintManager::init() {
    if (::enn::util::get_environment_property(DEBUG_PROPERTY_NAME_ENN, &print_mask))
        print_mask = ENN_DEFAULT_DEBUG_ZONE;

    debug_zone_info[DbgPartition::kError] = {DbgPriority::kError, "ERR", PREFIX_RED_STR, POSTFIX_STR};
    debug_zone_info[DbgPartition::kWarning] = {DbgPriority::kWarn, "WAR", PREFIX_YELLOW_STR,
                                                    POSTFIX_STR};
    debug_zone_info[DbgPartition::kInfo] = {DbgPriority::kInfo, "INF", PREFIX_CYAN_STR, POSTFIX_STR};
    debug_zone_info[DbgPartition::kDebug] = {DbgPriority::kDebug, "DBG", PREFIX_STR, PREFIX_STR};
    debug_zone_info[DbgPartition::kTest] = {DbgPriority::kDebug, "TST", PREFIX_STR, PREFIX_STR};
    debug_zone_info[DbgPartition::kMemory] = {DbgPriority::kDebug, "MEM", PREFIX_STR, PREFIX_STR};
    debug_zone_info[DbgPartition::kUser] = {DbgPriority::kDebug, "USR", PREFIX_STR, PREFIX_STR};
}

static char *CutParameter(char *pf, int cut_buffer_size = 4) {
    size_t i;
    if (cut_buffer_size < 2) return pf;
    for (i = 0; i < strlen(pf); i++) {
        if (pf[i] == '(') {
            if (pf[i + 1] != ')') {
                for (int buf_cnt = 1; buf_cnt <= cut_buffer_size - 2; ++buf_cnt)
                    pf[i + buf_cnt] = '.';
                pf[i + cut_buffer_size - 1] = ')';
                pf[i + cut_buffer_size] = 0;
            }
            break;
        }
    }
    return reinterpret_cast<char *>(pf);
}

int __attribute__((format(printf, 6, 7)))
enn_print_with_check(const char *log_tag, DbgPartition zone, DbgPrintOption check, const char *caller, const int caller_num,
                         const char *format, ...) try {
#ifdef ENN_BUILD_RELEASE
    if (check == DbgPrintOption::kEnnPrintRelease) return ENN_RET_FILTERED;
#endif
    auto dbg_info = DbgPrintManager::GetInstance().get_debug_zone_info();
    auto print_mask = DbgPrintManager::GetInstance().get_print_mask();
    char string[MAX_PRINT_LINE];

    if (dbg_info.find(zone) == dbg_info.end()) return ENN_RET_FILTERED;
    if (check != DbgPrintOption::kEnnPrintFalse && !(print_mask & ZONE_BIT(static_cast<uint64_t>(zone))))
        return ENN_RET_FILTERED; /* filtered */

    va_list ap;
    va_start(ap, format);

#ifdef VERSION_INFO
    std::string prefix(VERSION_INFO);
#else
    std::string prefix("");
#endif

#ifndef ENN_BUILD_RELEASE
    /* assembly message with caller function, line, message */
    #define MAX_CALLER_N   100
    #define TO_STR(x)      #x
    const int add_buffer_size = 4;
    char tmp_char_line[MAX_PRINT_LINE + add_buffer_size];  // buffer for CutParameter()
    strncpy(tmp_char_line, caller, MAX_PRINT_LINE);
    tmp_char_line[MAX_PRINT_LINE] = 0;
    char *caller_reg = CutParameter(tmp_char_line, add_buffer_size);
    if (strlen(caller_reg) > MAX_CALLER_N)
        caller_reg = &(caller_reg[strlen(caller_reg) - MAX_CALLER_N]);
    snprintf(string, sizeof(string), "[%s:%s%s%s(%5d) %30s:%4d]%s", prefix.c_str(), dbg_info[zone].prefix.c_str(),
             dbg_info[zone].symbol.c_str(), dbg_info[zone].postfix.c_str(), enn::util::get_tid(), caller_reg, caller_num,
             format);
#else
    snprintf(string, sizeof(string), "%s: %s", prefix.c_str(), format);
#endif

#if defined(__ANDROID__) || defined(__DSP_NDK_BUILD__)
    if (log_tag == nullptr)
        __android_log_vprint(static_cast<int>(dbg_info[zone].priority), LOG_TAG, string, ap);
    else
        __android_log_vprint(static_cast<int>(dbg_info[zone].priority), log_tag, string, ap);
#else
    vprintf(string, ap);
#endif
    va_end(ap);

    return 0;
} catch (const std::out_of_range & oor) {
    /* not work */
    return -1;
} catch (const std::length_error & le) {
    /* not work */
    return -1;
}

}  // namespace dbg
}  // namespace enn

