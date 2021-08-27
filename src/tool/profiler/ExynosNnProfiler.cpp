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
 * @file    ExynosNnProfiler.cpp
 * @brief   It defines interface functions to use ExynosNN Profiler.
 * @details The ExynosNnProfiler is creaetd and used in many places as singleton pattern.
 * @version 1
 */

#include "tool/profiler/include/ExynosNnProfiler.hpp"

#include "tool/profiler/include/ExynosNnProfiler.h"
#include "tool/profiler/include/ExynosNnProfilerConfig.h"
#include "tool/profiler/include/ProfileWatcher.hpp"
#include "tool/profiler/include/ProfilerLog.h"


static inline bool validate_range_of_profile_level() {
    if (*profile_level_is() > EXYNOS_NN_PROFILER_DISABLED &&
        *profile_level_is() < EXYNOS_NN_PROFILER_LEVEL_BOUND) {
        return true;
    } else {
        return false;
    }
}

uint8_t* profile_level_is(void) {
    static uint8_t profile_level = 0;
    return &profile_level;
}


std::unique_ptr<ExynosNnProfiler> ExynosNnProfiler::_instance;
std::once_flag ExynosNnProfiler::_once_flag;
std::mutex ExynosNnProfiler::_mutex;
std::map<uint32_t, ProfileWatcher*> ExynosNnProfiler::profile_watchers;

ExynosNnProfiler* ExynosNnProfiler::get_instance() {
    call_once(ExynosNnProfiler::_once_flag, []() {
        _instance.reset(new ExynosNnProfiler);
    });
    profile_watchers.insert(std::make_pair(0, new ProfileWatcher(0)));
    return _instance.get();
}

ExynosNnProfiler* ExynosNnProfiler::get_instance(uint64_t id) {
    call_once(ExynosNnProfiler::_once_flag, []() {
        _instance.reset(new ExynosNnProfiler);
    });
    std::lock_guard<std::mutex> lock_guard(_mutex);
    profile_watchers.insert(std::make_pair(id, new ProfileWatcher(id)));
    return _instance.get();
}

void ExynosNnProfiler::release_instance(uint64_t id) {
    std::lock_guard<std::mutex> lock_guard(_mutex);
    auto pw_iter = profile_watchers.find(id);
    if(pw_iter == profile_watchers.end()) {
        profile::log::error("Try to release the profiler_watcher but there is no id matched!\n");
        return;
    }
    delete pw_iter->second;
    profile_watchers.erase(pw_iter);
    if (_instance && profile_watchers.size() == 0) {
        // delete the singleton instance.
        _instance.reset();
        _instance = nullptr;
    }
}

ExynosNnProfiler* ExynosNnProfiler::only_get_instance() {
    return _instance.get();
}

ProfileWatcher* ExynosNnProfiler::get_profile_watcher(uint64_t id) {
    auto pw_iter = profile_watchers.find(id);
    if(pw_iter == profile_watchers.end()) {
        profile::log::error("There is no id matched!\n");
        return nullptr;
    }
    else {
        return pw_iter->second;
    }
}


StartProfiler::StartProfiler() {
    ExynosNnProfiler::get_instance();
}

StartProfiler::StartProfiler(uint64_t id) {
    ExynosNnProfiler::get_instance(id);
}

FinishProfiler::FinishProfiler(uint64_t id) : id(id) {
}

FinishProfiler::~FinishProfiler() {
    ExynosNnProfiler::release_instance(id);
}


ScopedProfiling::ScopedProfiling(std::string custom_label, uint64_t id)
: custom_label(custom_label), id(id), op_num(-1) {
    ExynosNnProfiler::only_get_instance()->get_profile_watcher(id)->push_profile_data(
        new EntryProfileData(custom_label));
}

ScopedProfiling::ScopedProfiling(std::string custom_label, uint64_t id, int32_t op_num)
: custom_label(custom_label), id(id), op_num(op_num) {
    ExynosNnProfiler::only_get_instance()->get_profile_watcher(id)->push_profile_data(
        new EntryProfileData(custom_label, op_num));
}

ScopedProfiling::~ScopedProfiling() {
    if(custom_label.empty())
        ExynosNnProfiler::only_get_instance()->get_profile_watcher(id)->push_profile_data(
            new ExitProfileData(file, std::to_string(line), func));
    else
        ExynosNnProfiler::only_get_instance()->get_profile_watcher(id)->push_profile_data(
            new ExitProfileData(custom_label, op_num));
}


void profile_from(const char* custom_label, uint64_t id) {
    ExynosNnProfiler::only_get_instance()->get_profile_watcher(id)->push_profile_data(
        new EntryProfileData(custom_label));
    return;
}

void profile_from_with_op_num(const char* custom_label, uint64_t id, int32_t op_num) {
    ExynosNnProfiler::only_get_instance()->get_profile_watcher(id)->push_profile_data(
        new EntryProfileData(custom_label, op_num));
    return;
}

void profile_until(const char* custom_label, uint64_t id) {
    ExynosNnProfiler::only_get_instance()->get_profile_watcher(id)->push_profile_data(
        new ExitProfileData(custom_label));
    return;
}

void profile_until_with_op_num(const char* custom_label, uint64_t id, int32_t op_num) {
    ExynosNnProfiler::only_get_instance()->get_profile_watcher(id)->push_profile_data(
        new ExitProfileData(custom_label, op_num));
    return;
}

void profile_append(struct CalculatedProfileNode* calculated_profile_node, uint64_t id) {
    ExynosNnProfiler::only_get_instance()->get_profile_watcher(id)->push_profile_data(
        new CalculatedProfileData(calculated_profile_node));
    return;
}

void profile_exclude_from(const char* custom_label, uint64_t id) {
    ExynosNnProfiler::only_get_instance()->get_profile_watcher(id)->push_profile_data(
        new EntryProfileData(custom_label, true));
    return;
}

void profile_exclude_until(const char* custom_label, uint64_t id) {
    ExynosNnProfiler::only_get_instance()->get_profile_watcher(id)->push_profile_data(
        new ExitProfileData(custom_label, true));
    return;
}
