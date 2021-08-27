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
 * @file    ExynosNnProfilerApi.h
 * @brief   It is collection of API to be called by users.
 * @details The functions are called for starting the profier or gathering time point data.
 * @version 1
 */

#ifndef TOOLS_PROFILER_INCLUDE_PROFILEDATA_HPP_
#define TOOLS_PROFILER_INCLUDE_PROFILEDATA_HPP_

#ifdef __cplusplus

#include <string>
#include <chrono>
#include <thread>

class ProfileData {
 public:
    ProfileData();
    ProfileData(const std::string& file, const std::string& line, const std::string& func);
    ProfileData(const std::string& custom_label);
    ProfileData(const std::string& custom_label, const bool& is_excluded);
    ProfileData(const std::string& custom_label, const int32_t& op_num);
    virtual ~ProfileData() = default;

    void set_custom_label(const std::string& custom_label);

    const std::chrono::system_clock::time_point& get_time();
    const std::string& get_file();
    const std::string& get_line();
    const std::string& get_func();
    const std::thread::id& get_thread_id();
    const std::string& get_custom_label();
    const int32_t& get_op_num();

    bool is_excluded;

 protected:
    std::chrono::system_clock::time_point time;

 private:
    std::string file;
    std::string line;
    std::string func;
    std::string custom_label;
    std::thread::id thread_id;
    int32_t op_num;
};


class EntryProfileData : public ProfileData {
 public:
    EntryProfileData(const std::string& file, const std::string& line, const std::string& func);
    EntryProfileData(const std::string& custom_label);
    EntryProfileData(const std::string& custom_label, const bool& is_excluded);
    EntryProfileData(const std::string& custom_label, const int32_t& op_num);
    virtual ~EntryProfileData() = default;
};


class ExitProfileData : public ProfileData {
 public:
    ExitProfileData(const std::string& file, const std::string& line, const std::string& func);
    ExitProfileData(const std::string& custom_label);
    ExitProfileData(const std::string& custom_label, const bool& is_excluded);
    ExitProfileData(const std::string& custom_label, const int32_t& op_num);
    virtual ~ExitProfileData() = default;
};

class CalculatedProfileData : public ProfileData {
 public:
    CalculatedProfileData(struct CalculatedProfileNode* calculated_profile_node);
    virtual ~CalculatedProfileData() = default;
    struct CalculatedProfileNode* get_calculated_profile_node();

 private:
    struct CalculatedProfileNode* calculated_profile_node;
};

#endif // __cplusplus

#endif // TOOLS_PROFILER_INCLUDE_PROFILEDATA_HPP_
