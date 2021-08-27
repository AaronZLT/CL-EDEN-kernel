/*
 * Copyright (C) 2020 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

/**
 * @file    ExynosNnProfiler.hpp
 * @brief   ExynosNnProfiler is interface class to users.
 * @details ExynosNnProfiler which is singleton can be instantiate and released by using each function.
 * @version 1
 */

#ifndef TOOLS_PROFILER_INCLUDE_EXYNOSNNPROFILER_HPP_
#define TOOLS_PROFILER_INCLUDE_EXYNOSNNPROFILER_HPP_
#ifdef __cplusplus

#include <map>
#include <memory>
#include <mutex>

class ProfileWatcher;

// The singleton instance of ExynosNnProfiler is instantiated only if it is actually used.
class ExynosNnProfiler {
 public:
    // Create a instance for the ProfileWatcher if there is no id defined.
    static ExynosNnProfiler* get_instance();

    // Create a instance for the ProfileWatcher according to id.
    static ExynosNnProfiler* get_instance(uint64_t id);

    // The singleton instance of ExynosNnProfiler is released,
    // when all of ProfileWather instances are released.
    static void release_instance(uint64_t id);

    // It returns the singleton instance without other job.
    static ExynosNnProfiler* only_get_instance();

    static ProfileWatcher* get_profile_watcher(uint64_t id = 0);

    ExynosNnProfiler(const ExynosNnProfiler&) = delete;
    ExynosNnProfiler() = default;
    ~ExynosNnProfiler() = default;
    ExynosNnProfiler& operator=(const ExynosNnProfiler&) = delete;

 private:
    static std::unique_ptr<ExynosNnProfiler> _instance;
    static std::once_flag _once_flag;
    static std::mutex _mutex;
    static std::map<uint32_t, ProfileWatcher*> profile_watchers;
};


class StartProfiler
{
 public:
    StartProfiler();
    StartProfiler(uint64_t id);
};


class FinishProfiler
{
 public:
    FinishProfiler(uint64_t id);
    ~FinishProfiler();

 private:
    uint64_t id;
};


class ScopedProfiling {
 public:
    ScopedProfiling(std::string file, int line, std::string func, uint64_t id);
    ScopedProfiling(std::string custom_label, uint64_t id);
    ScopedProfiling(std::string custom_label, uint64_t id, int32_t op_num);
    ~ScopedProfiling();

 private:
    std::string file;
    std::string custom_label;
    int line = 0;
    std::string func;
    uint64_t id;
    int32_t op_num;
};

#endif  // __cplusplus


#endif // TOOLS_PROFILER_INCLUDE_EXYNOSNNPROFILER_HPP_
