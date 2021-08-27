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
 * @file    ProfileWatcher.hpp
 * @brief   It declares the class to generate the profile tree after dequeuing .
 * @details A thread created by method in this class works to create profile tree from ProfileDataQueue.
 * @version 1
 */

#ifndef TOOLS_PROFILER_INCLUDE_PROFILEWATCHER_HPP_
#define TOOLS_PROFILER_INCLUDE_PROFILEWATCHER_HPP_

#ifdef __cplusplus

#include <map>
#include <condition_variable>
#include <inttypes.h>

#include "tool/profiler/include/ExynosNnProfiler.h"
#include "tool/profiler/include/ExynosNnProfilerConfig.h"
#include "tool/profiler/include/ProfileTreeNode.hpp"
#include "tool/profiler/include/ProfileData.hpp"
#include "tool/profiler/include/ProfileDataQueue.hpp"

// Duration that Watcher thread should wait (us).
constexpr int WATCHER_THREAD_WAIT_TIME = 100;

// Messages to print out the result of profile.
constexpr char MSG_START[] =
    "Start printing the profiled data result...\n";
constexpr char MSG_TABLE_BORDER_TOP[] =
    "===========================================Result(Model:%#" PRIX64 ")============================================\n";
constexpr char MSG_TABLE_COLUMN_TITLE[] =
    "     <Avg>     <Min>     <Max>    <Self>    <Ratio>      <90%%>    <Count>          <Name>\n";
constexpr char MSG_CONTENT_ROW[] =
    " %9" PRId64 " %9" PRId64 " %9" PRId64 " %9" PRId64 " %9.2f%%  %9" PRId64 " %9" PRIu32 "       %s%-80s\n";
constexpr char MSG_TABLE_BORDER_BOTTOM[] =
    "===================================================================================================================\n";
constexpr char MSG_FINISH[] =
    "Finish printing the profiled data result...\n";

constexpr char DUMP_PATH[] = "/data/vendor/enn/dump/latency/exynos_nn_profiler_dump.csv";

class ProfileTreeNode;
class ProfileWatcher {
 public:
    ProfileWatcher(const ProfileWatcher&) = delete;
    ProfileWatcher(uint64_t id);
    ~ProfileWatcher();
    ProfileWatcher& operator=(const ProfileWatcher&) = delete;

    std::condition_variable& get_condition_variable();
    void push_profile_data(ProfileData* profile_data);
    void print_profile_tree();
    void print_node(ProfileTreeNode* sub_tree_root, ProfileTreeNode* node, int level);
    void print_profile_data_error();
    void calculate_self_duration(ProfileTreeNode* node);
    void trim_profile_tree();
    void thread_func();
    bool& profile_thread_should_finish();
    bool& profile_data_is_normal();
    void create_profile_tree();
    void create_profile_data_q();
    std::string parse_label(ProfileData* profile_data);
    void update_tree_foundation(const std::string& label, ProfileTreeNode* node);
    ProfileTreeNode* search_tree_foundation(const std::string& label);
    void propagate_excluded_duration(ProfileTreeNode* node);
    bool process_entry_profile_data(ProfileData* profile_data);
    bool process_exit_profile_data(ProfileData* profile_data);
    bool process_calculated_profile_data(ProfileData* profile_data);
    ProfileTreeNode* append_calculated_profile_node(CalculatedProfileNode* calculated_profile_node, ProfileTreeNode* head);
    void update_calculated_profile_node(CalculatedProfileNode* calculated_profile_node, ProfileTreeNode* head);
    bool supervise_profile_tree();
    void dump_node(ProfileTreeNode* node, std::string label);
    void dump_profile_tree();

 private:
    ProfileDataQueue<ProfileData>* profile_data_q;
    ProfileTreeNode* profile_tree_root;
    ProfileTreeNode* profile_tree_head;
    std::map<std::string, ProfileTreeNode*> map_info_oftree_foundation;
    std::map<std::thread::id, ProfileTreeNode*> tree_heads_of_threads;
    std::thread watcher_thread;
    std::condition_variable cv_;
    std::mutex m_;
    uint64_t id;
};

#endif // __cplusplus

#endif // TOOLS_PROFILER_INCLUDE_PROFILEWATCHER_HPP_
