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
 * @file    ProfileTreeNode.hpp
 * @brief   It declares the class to repesent the tree node to contain profiled data.
 * @details These nodes are linked in the form of a linked list, eventually forming a tree.
 * @version 1
 */

#ifndef TOOLS_PROFILER_INCLUDE_PROFILETREENODE_HPP_
#define TOOLS_PROFILER_INCLUDE_PROFILETREENODE_HPP_

#ifdef __cplusplus

#include <chrono>
#include <string>
#include <queue>
#include <thread>
#include "tool/profiler/include/ProfilerLog.h"

class ProfileTreeNode {
 public:
    ProfileTreeNode(const std::string& label);
    ProfileTreeNode(const std::string& label, const std::chrono::system_clock::time_point& entry_time);
    ProfileTreeNode(const std::string& label, const std::chrono::system_clock::time_point& entry_time,
                        const std::thread::id& thread_id);
    ProfileTreeNode(const std::string& label, const std::chrono::system_clock::time_point& entry_time,
                        const std::thread::id& thread_id, const bool& is_excluded);
    ProfileTreeNode(const std::string label, const uint32_t& duration);

    ProfileTreeNode* operator=(const std::chrono::system_clock::time_point& entry_time);
    ProfileTreeNode* operator=(const std::chrono::microseconds& duration);
    bool operator!=(const std::string& label_) const;

    void calc_avg_duration();
    void calc_ninetieth_latency();
    std::string get_label() const;
    ProfileTreeNode* push_child_node(ProfileTreeNode* child);
    ProfileTreeNode* pop_child();
    void calc_proportion(ProfileTreeNode* root);
    void add_entry_time(const std::chrono::system_clock::time_point& entry_time);

    void set_parent(ProfileTreeNode* parent);
    ProfileTreeNode* set_duration(std::chrono::system_clock::time_point exit_time);
    void update_exclude_duration(const std::chrono::microseconds& duration);
    void set_self_duration(const std::chrono::microseconds& duration);

    ProfileTreeNode* get_parent();
    std::chrono::microseconds& get_cur_duration();
    std::chrono::microseconds& get_avg_duration();
    std::chrono::microseconds& get_min_duration();
    std::chrono::microseconds& get_max_duration();
    std::chrono::microseconds& get_self_duration();
    double get_proportion_of_total();
    double get_proportion_of_caller();
    int get_iteration();
    std::chrono::system_clock::time_point& get_entry_time();
    std::queue<std::chrono::system_clock::time_point>& get_entry_times();
    std::vector<ProfileTreeNode*>& get_children();
    std::thread::id get_thread_id();
    std::chrono::microseconds& get_ninetieth_latency();
    std::vector<std::chrono::microseconds>& get_durations();
    bool is_established;
    bool is_excluded;


 private:
    std::string label;
    std::queue<std::chrono::system_clock::time_point> entry_times;
    std::chrono::system_clock::time_point entry_time;
    std::vector<std::chrono::microseconds> durations;
    std::chrono::microseconds ninetieth_latency;
    std::chrono::microseconds cur_duration;
    std::chrono::microseconds min_duration;
    std::chrono::microseconds max_duration;
    std::chrono::microseconds avg_duration;
    std::chrono::microseconds sum_duration;
    std::chrono::microseconds self_duration;
    std::chrono::microseconds exclude_duration;
    double proportion_of_total;
    double proportion_of_caller;
    std::thread::id thread_id;
    int iteration;
    ProfileTreeNode* parent;
    std::vector<ProfileTreeNode*> children;
};

#endif // __cplusplus

#endif // TOOLS_PROFILER_INCLUDE_PROFILETREENODE_HPP_
