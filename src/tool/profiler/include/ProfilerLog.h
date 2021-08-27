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
 * @file    ProfileDataQueue.hpp
 * @brief   It declares the classes to enqueue or dequeue the instance from ProfileData.
 * @details ProfileWatcher object holds a ProfileDataQueue as class variable.
 * @version 1
 */

#ifndef TOOLS_PROFILER_INCLUDE_PROFILERLOG_H_
#define TOOLS_PROFILER_INCLUDE_PROFILERLOG_H_

#include <utility>
#include <string>
#include <android/log.h>

namespace profile {
namespace log {

static const char* TAG = "EXYNOS_NN_PROFILER";

template <typename... Args>
inline void result(std::string format, Args&&... args) noexcept {
    __android_log_print(ANDROID_LOG_INFO, TAG, format.c_str(), std::forward<Args>(args)...);
}

template <typename... Args>
inline void error(std::string format, Args&&... args) noexcept {
    __android_log_print(ANDROID_LOG_ERROR, TAG, format.c_str(), std::forward<Args>(args)...);
}


};  // namesapce log
};  // namespace profile

#endif // TOOLS_PROFILER_INCLUDE_PROFILERLOG_H_
