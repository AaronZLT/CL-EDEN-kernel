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
 * @file    ProfileTreeNode.cpp
 * @brief   It defines class of ProfileTreeNode.
 * @details A node in profile tree represents one scope which user tries to profile.
 * @version 1
 */

#include "tool/profiler/include/ProfileTreeNode.hpp"

class ProfileTreeNode;

ProfileTreeNode::ProfileTreeNode(const std::string& label)
: iteration(1), proportion_of_total(0), proportion_of_caller(0), parent(nullptr),
thread_id(std::this_thread::get_id()), is_established(false),
is_excluded(false), label(label), entry_time(std::chrono::system_clock::now()),
ninetieth_latency(std::chrono::microseconds::zero()),
cur_duration(std::chrono::microseconds::zero()), self_duration(std::chrono::microseconds::zero()),
sum_duration(std::chrono::microseconds::zero()), min_duration(std::chrono::microseconds::max()),
avg_duration(std::chrono::microseconds::zero()), max_duration(std::chrono::microseconds::min()),
exclude_duration(std::chrono::microseconds::zero()) {
}

ProfileTreeNode::ProfileTreeNode(const std::string& label,
                                    const std::chrono::system_clock::time_point& entry_time)
: iteration(0), proportion_of_total(0), proportion_of_caller(0), parent(nullptr),
is_established(false), is_excluded(false), label(label), entry_time(entry_time),
ninetieth_latency(std::chrono::microseconds::zero()),
cur_duration(std::chrono::microseconds::zero()), self_duration(std::chrono::microseconds::zero()),
sum_duration(std::chrono::microseconds::zero()), min_duration(std::chrono::microseconds::max()),
avg_duration(std::chrono::microseconds::zero()), max_duration(std::chrono::microseconds::min()),
exclude_duration(std::chrono::microseconds::zero()) {
add_entry_time(entry_time);
}

ProfileTreeNode::ProfileTreeNode(const std::string& label,
                                    const std::chrono::system_clock::time_point& entry_time,
                                    const std::thread::id& thread_id)
: iteration(0), proportion_of_total(0), proportion_of_caller(0), parent(nullptr),
is_established(false), is_excluded(false), thread_id(thread_id), label(label),
ninetieth_latency(std::chrono::microseconds::zero()),
cur_duration(std::chrono::microseconds::zero()), self_duration(std::chrono::microseconds::zero()),
sum_duration(std::chrono::microseconds::zero()), min_duration(std::chrono::microseconds::max()),
avg_duration(std::chrono::microseconds::zero()), max_duration(std::chrono::microseconds::min()),
exclude_duration(std::chrono::microseconds::zero()) {
    add_entry_time(entry_time);
}

ProfileTreeNode::ProfileTreeNode(const std::string& label,
                                    const std::chrono::system_clock::time_point& entry_time,
                                    const std::thread::id& thread_id, const bool& is_excluded)
: iteration(0), proportion_of_total(0), proportion_of_caller(0), parent(nullptr),
is_established(false), is_excluded(is_excluded), thread_id(thread_id), label(label),
ninetieth_latency(std::chrono::microseconds::zero()),
cur_duration(std::chrono::microseconds::zero()), self_duration(std::chrono::microseconds::zero()),
sum_duration(std::chrono::microseconds::zero()), min_duration(std::chrono::microseconds::max()),
avg_duration(std::chrono::microseconds::zero()), max_duration(std::chrono::microseconds::min()),
exclude_duration(std::chrono::microseconds::zero()) {
    add_entry_time(entry_time);
}

ProfileTreeNode::ProfileTreeNode(std::string label, const uint32_t& duration)
: iteration(0),  proportion_of_total(0), proportion_of_caller(0), parent(nullptr),
is_established(false), is_excluded(false), thread_id(std::this_thread::get_id()), label(label),
ninetieth_latency(std::chrono::microseconds::zero()),
cur_duration(std::chrono::microseconds::zero()), self_duration(std::chrono::microseconds::zero()),
sum_duration(std::chrono::microseconds::zero()), min_duration(std::chrono::microseconds::max()),
avg_duration(std::chrono::microseconds::zero()), max_duration(std::chrono::microseconds::min()),
exclude_duration(std::chrono::microseconds::zero()) {
    *this = std::chrono::microseconds(duration);
}

ProfileTreeNode* ProfileTreeNode::operator=(const std::chrono::system_clock::time_point& entry_time) {
    this->entry_time = entry_time;
    return this;
}

ProfileTreeNode* ProfileTreeNode::operator=(const std::chrono::microseconds& duration) {
    cur_duration = duration - exclude_duration;
    exclude_duration = std::chrono::microseconds::zero();
    if (min_duration > cur_duration) {
        min_duration = cur_duration;
    }
    if (max_duration < cur_duration) {
        max_duration = cur_duration;
    }
    sum_duration += cur_duration;
    durations.push_back(cur_duration);
    iteration++;
    return this;
}

bool ProfileTreeNode::operator!=(const std::string& label_) const {
    if (label != label_) {
        profile::log::error("The start and end points of the profiling range do not coincide with each other.\n");
        profile::log::error("The start is : %s, but the end is %s\n", label.c_str(), label_.c_str());
        profile::log::error("Please profile in the same scope or with the same label\n");
        return true;
    }
    return false;
}

ProfileTreeNode* ProfileTreeNode::set_duration(std::chrono::system_clock::time_point exit_time) {
    std::chrono::microseconds duration =
            std::chrono::duration_cast<std::chrono::microseconds>(exit_time - entry_times.front());
    entry_times.pop();
    return *this = duration;
}

void ProfileTreeNode::update_exclude_duration(const std::chrono::microseconds& duration) {
    this->exclude_duration += duration;
    return;
}

void ProfileTreeNode::set_self_duration(const std::chrono::microseconds& duration) {
    this->self_duration = duration;
    return;
}

void ProfileTreeNode::add_entry_time(const std::chrono::system_clock::time_point& entry_time) {
    entry_times.push(entry_time);
    return;
}

void ProfileTreeNode::calc_avg_duration() {
    if (iteration > 3) {
        avg_duration = sum_duration - min_duration - max_duration;
        avg_duration /= iteration - 2;
    }
    else {
        avg_duration = sum_duration / iteration;
    }
    return;
}

void ProfileTreeNode::calc_ninetieth_latency() {
    std::sort(durations.begin(), durations.end());
    if(this->iteration >= 10) {
        ninetieth_latency = durations[this->iteration * 0.9f];
    } else {
        ninetieth_latency = std::chrono::microseconds::zero();
    }
}

void ProfileTreeNode::set_parent(ProfileTreeNode* parent) {
    this->parent = parent;
    return;
}

ProfileTreeNode* ProfileTreeNode::push_child_node(ProfileTreeNode* child) {
    child->set_parent(this);
    children.push_back(child);
    return child;
}

ProfileTreeNode* ProfileTreeNode::pop_child() {
    return parent;

}
void ProfileTreeNode::calc_proportion(ProfileTreeNode* root) {
    proportion_of_total =  static_cast<double>(self_duration.count()) /
                    static_cast<double>(root->get_avg_duration().count()) * 100;
    return;
}

std::string ProfileTreeNode::get_label() const {
    return label;
}

ProfileTreeNode* ProfileTreeNode::get_parent() {
    return parent;
}

std::chrono::microseconds& ProfileTreeNode::get_cur_duration() {
    return cur_duration;
}

std::chrono::microseconds& ProfileTreeNode::get_avg_duration() {
    return avg_duration;
}

std::chrono::microseconds& ProfileTreeNode::get_min_duration() {
    return min_duration;
}

std::chrono::microseconds& ProfileTreeNode::get_max_duration() {
    return max_duration;
}

std::chrono::microseconds& ProfileTreeNode::get_self_duration() {
    return self_duration;
}

double ProfileTreeNode::get_proportion_of_total() {
    return proportion_of_total;
}

double ProfileTreeNode::get_proportion_of_caller() {
    return proportion_of_caller;
}

int ProfileTreeNode::get_iteration() {
    return iteration;
}

std::chrono::system_clock::time_point& ProfileTreeNode::get_entry_time() {
    return entry_time;
}

std::queue<std::chrono::system_clock::time_point>& ProfileTreeNode::get_entry_times() {
    return entry_times;
}

std::vector<ProfileTreeNode*>& ProfileTreeNode::get_children() {
    return children;
}

std::thread::id ProfileTreeNode::get_thread_id() {
    return thread_id;
}

std::chrono::microseconds& ProfileTreeNode::get_ninetieth_latency() {
    return ninetieth_latency;
}

std::vector<std::chrono::microseconds>& ProfileTreeNode::get_durations() {
    return durations;
}

