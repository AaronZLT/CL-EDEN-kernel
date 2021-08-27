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
 * @file    ProfileData.cpp
 * @brief   It defines class of ProfileData.
 * @details ProfileData instantiated would be enqueued in ProfileDataQueue.
 * @version 1
 */

#include "tool/profiler/include/ProfileData.hpp"

ProfileData::ProfileData()
: thread_id(std::this_thread::get_id()), is_excluded(false), op_num(-1) {
}

ProfileData::ProfileData(const std::string& file, const std::string& line, const std::string& func)
: file(file), line(line), func(func), thread_id(std::this_thread::get_id()),
is_excluded(false), time(std::chrono::system_clock::now()), op_num(-1) {
}

ProfileData::ProfileData(const std::string& custom_label)
: custom_label(custom_label), thread_id(std::this_thread::get_id()),
is_excluded(false), time(std::chrono::system_clock::now()), op_num(-1) {
}

ProfileData::ProfileData(const std::string& custom_label, const bool& is_excluded)
: custom_label(custom_label), thread_id(std::this_thread::get_id()),
is_excluded(is_excluded), time(std::chrono::system_clock::now()), op_num(-1) {
}

ProfileData::ProfileData(const std::string& custom_label, const int32_t& op_num)
: custom_label(custom_label), thread_id(std::this_thread::get_id()),
is_excluded(false), time(std::chrono::system_clock::now()), op_num(op_num) {
}

void ProfileData::set_custom_label(const std::string& custom_label) {
    this->custom_label = custom_label;
    return;
}


const std::chrono::system_clock::time_point& ProfileData::get_time() {
    return time;
}

const std::string& ProfileData::get_file() {
    return file;
}

const std::string& ProfileData::get_line() {
    return line;
}

const std::string& ProfileData::get_func() {
    return func;
}

const std::thread::id& ProfileData::get_thread_id() {
    return thread_id;
}

const std::string& ProfileData::get_custom_label() {
    return custom_label;
}

const std::int32_t& ProfileData::get_op_num() {
    return op_num;
}


EntryProfileData::EntryProfileData(const std::string& file, const std::string& line, const std::string& func)
: ProfileData(file, line, func) {
}

EntryProfileData::EntryProfileData(const std::string& custom_label)
: ProfileData(custom_label) {
}

EntryProfileData::EntryProfileData(const std::string& custom_label, const bool& is_excluded)
: ProfileData(custom_label, is_excluded) {
}

EntryProfileData::EntryProfileData(const std::string& custom_label, const int32_t& op_num)
: ProfileData(custom_label, op_num) {
}

ExitProfileData::ExitProfileData(const std::string& file, const std::string& line, const std::string& func)
: ProfileData(file, line, func) {
}

ExitProfileData::ExitProfileData(const std::string& custom_label)
: ProfileData(custom_label) {
}

ExitProfileData::ExitProfileData(const std::string& custom_label, const bool& is_excluded)
: ProfileData(custom_label, is_excluded) {
}

ExitProfileData::ExitProfileData(const std::string& custom_label, const int32_t& op_num)
: ProfileData(custom_label, op_num) {
}

CalculatedProfileData::CalculatedProfileData(struct CalculatedProfileNode* calculated_profile_node)
: calculated_profile_node(calculated_profile_node) {
}

struct CalculatedProfileNode* CalculatedProfileData::get_calculated_profile_node() {
    return calculated_profile_node;
}
