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

#ifndef SRC_COMMON_ENN_DEBUG_H_
#define SRC_COMMON_ENN_DEBUG_H_

#include "common/enn_common_type.h"
#include "common/enn_debug_zone.h"
#include "common/enn_log.hpp"

#include <cinttypes>
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

namespace enn {
namespace debug {

/* define type */
using MaskType = uint64_t;
class DbgPrintManager;

constexpr uint32_t MAX_PRINT_LINE = 1024;

/* header */
#define ZONE_BIT_MASK(zone) (1U << static_cast<enn::debug::MaskType>(zone))

struct DebugPartitionInfo {
    DbgPriority priority;
    std::string symbol;
    std::string prefix;
    std::string postfix;
};

/* class to control msgs */
class DbgPrintManager {
public:
    ~DbgPrintManager() = default;

    DbgPrintManager(const DbgPrintManager &) = delete;
    DbgPrintManager(DbgPrintManager &&) = delete;
    DbgPrintManager &operator= (const DbgPrintManager &) = delete;
    DbgPrintManager &operator= (DbgPrintManager &&) = delete;

    /* setter for mask (for test) */
    void set_mask(MaskType mask);
    EnnReturn enn_set_debug_zone(enum DbgPartition zone);
    EnnReturn enn_clr_debug_zone(enum DbgPartition zone);

    /* getter for singleton */
    static DbgPrintManager& GetInstance() {
        std::lock_guard<std::mutex> lock(_mutex);
        //  std::call_once(DbgPrintManager::flag, []() { DbgPrintManager::_instance.reset(new DbgPrintManager); });
        if (_instance == nullptr)
            _instance.reset(new DbgPrintManager);
        return *(_instance.get());
    }

    const std::map<DbgPartition, DebugPartitionInfo> & get_debug_zone_info();
    const MaskType & get_print_mask();
    const std::string & get_debug_property_name() { return DEBUG_PROPERTY_NAME_ENN; }

private:
    /* default debug flags */
#ifdef ENN_BUILD_RELEASE
    const MaskType ENN_DEFAULT_DEBUG_ZONE =
        (ZONE_BIT_MASK(DbgPartition::kError) | ZONE_BIT_MASK(DbgPartition::kWarning));
#else
    const MaskType ENN_DEFAULT_DEBUG_ZONE =
        (ZONE_BIT_MASK(DbgPartition::kError) | ZONE_BIT_MASK(DbgPartition::kWarning) |
         ZONE_BIT_MASK(DbgPartition::kInfo) | ZONE_BIT_MASK(DbgPartition::kTest) |
         ZONE_BIT_MASK(DbgPartition::kDebug) | ZONE_BIT_MASK(DbgPartition::kMemory) |
         ZONE_BIT_MASK(DbgPartition::kUser));
#endif

    /* singleton */
    DbgPrintManager();
    static std::unique_ptr<DbgPrintManager> _instance;
    static std::mutex _mutex;

    /* mask */
    MaskType print_mask = ENN_DEFAULT_DEBUG_ZONE;

    /* zone information */
    std::map<DbgPartition, DebugPartitionInfo> debug_zone_info;

    void init();
    /* default property name */
    const std::string DEBUG_PROPERTY_NAME_ENN {"dbg.property.for.exynos.nn"};
};

/* Debug features */
extern int __attribute__((format(printf, 5, 6)))
enn_print_with_check(enum DbgPartition, enum DbgPrintOption, const char *, const int, const char *, ...);

}  // namespace debug
}  // namespace enn


// mask for filtering
#define enn_print(log_tag, zone, message, ...) \
    ::enn::debug::enn_print_with_check(log_tag, zone, ::enn::debug::DbgPrintOption::kEnnPrintTrue, __PRETTY_FUNCTION__, __LINE__, "  " message, ##__VA_ARGS__)
#define enn_print_release(log_tag, zone, message, ...) \
    ::enn::debug::enn_print_with_check(log_tag, zone, ::enn::debug::DbgPrintOption::kEnnPrintRelease, __PRETTY_FUNCTION__, __LINE__, "  " message, ##__VA_ARGS__)
#define enn_print_nocheck(log_tag, zone, message, ...) \
    ::enn::debug::enn_print_with_check(log_tag, zone, ::enn::debug::DbgPrintOption::kEnnPrintFalse, __PRETTY_FUNCTION__, __LINE__, "* " message, ##__VA_ARGS__)

// print related macros I : stream style support
// MEMO(hoon98.choi) : std::endl should be printed
#define ENN_ERR_COUT ::enn::debug::EnnMsgHandler(__PRETTY_FUNCTION__, __LINE__, ::enn::debug::DbgPartition::kError)
#define ENN_WARN_COUT ::enn::debug::EnnMsgHandler(__PRETTY_FUNCTION__, __LINE__, ::enn::debug::DbgPartition::kWarning)
#define ENN_INFO_COUT ::enn::debug::EnnMsgHandler(__PRETTY_FUNCTION__, __LINE__, ::enn::debug::DbgPartition::kInfo)
#define ENN_DBG_COUT ::enn::debug::EnnMsgHandler(__PRETTY_FUNCTION__, __LINE__, ::enn::debug::DbgPartition::kDebug)
#define ENN_TST_COUT ::enn::debug::EnnMsgHandler(__PRETTY_FUNCTION__, __LINE__, ::enn::debug::DbgPartition::kTest)
#define ENN_MEM_COUT ::enn::debug::EnnMsgHandler(__PRETTY_FUNCTION__, __LINE__, ::enn::debug::DbgPartition::kMemory)
#define ENN_USER_COUT ::enn::debug::EnnMsgHandler(__PRETTY_FUNCTION__, __LINE__, ::enn::debug::DbgPartition::kUser)

// print related macros II : err, warn and info are shown in release build (info msgs are hide)
#define ENN_ERR_PRINT(message, ...) enn_print(NULL, ::enn::debug::DbgPartition::kError, message, ##__VA_ARGS__)
#define ENN_ERR_PRINT_FORCE(message, ...) enn_print_nocheck(NULL, ::enn::debug::DbgPartition::kError, message, ##__VA_ARGS__)
#define ENN_WARN_PRINT(message, ...) enn_print(NULL, ::enn::debug::DbgPartition::kWarning, message, ##__VA_ARGS__)
#define ENN_WARN_PRINT_FORCE(message, ...) enn_print_nocheck(NULL, ::enn::debug::DbgPartition::kWarning, message, ##__VA_ARGS__)
#define ENN_INFO_PRINT(message, ...) enn_print_release(NULL, ::enn::debug::DbgPartition::kInfo, message, ##__VA_ARGS__)
#define ENN_INFO_PRINT_FORCE(message, ...) enn_print_nocheck(NULL, ::enn::debug::DbgPartition::kInfo, message, ##__VA_ARGS__)
#define ENN_DBG_PRINT(message, ...) enn_print_release(NULL, ::enn::debug::DbgPartition::kDebug, message, ##__VA_ARGS__)
#define ENN_TST_PRINT(message, ...) enn_print_release(NULL, ::enn::debug::DbgPartition::kTest, message, ##__VA_ARGS__)
#define ENN_MEM_PRINT(message, ...) enn_print_release(NULL, ::enn::debug::DbgPartition::kMemory, message, ##__VA_ARGS__)
#define ENN_USER_PRINT(message, ...) enn_print_release(NULL, ::enn::debug::DbgPartition::kUser, message, ##__VA_ARGS__)

#define ENN_ERR_PRINT_TAG(log_tag, message, ...) enn_print(log_tag, ::enn::debug::DbgPartition::kError, message, ##__VA_ARGS__)
#define ENN_WARN_PRINT_TAG(log_tag, message, ...) enn_print(log_tag, ::enn::debug::DbgPartition::kWarning, message, ##__VA_ARGS__)
#define ENN_INFO_PRINT_TAG(log_tag, message, ...) enn_print_release(log_tag, ::enn::debug::DbgPartition::kInfo, message, ##__VA_ARGS__)
#define ENN_INFO_PRINT_FORCE_TAG(log_tag, message, ...) enn_print_nocheck(log_tag, ::enn::debug::DbgPartition::kInfo, message, ##__VA_ARGS__)
#define ENN_DBG_PRINT_TAG(log_tag, message, ...) enn_print_release(log_tag, ::enn::debug::DbgPartition::kDebug, message, ##__VA_ARGS__)


#ifndef VELOCE_SOC
    #define TRY try
    #define CATCH(expr)               \
        catch (std::exception & ex) { \
            const char *expr = ex.what();
    #define END_TRY }
#else
    #define TRY
    #define CATCH(expr) const char expr[] = "Exception occurred";
    #define END_TRY
#endif

#endif  // SRC_COMMON_ENN_DEBUG_H_
