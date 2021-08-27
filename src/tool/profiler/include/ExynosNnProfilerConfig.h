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
 * @file    ExynosNnProfilerConfig.h
 * @brief   It is collection of configuration for ExynosNN Profiler
 * @details The enum indicating profile level
 * @version 1
 */

#ifndef TOOLS_PROFILER_INCLUDE_EXYNOSNNPROFILERCONFIG_H_
#define TOOLS_PROFILER_INCLUDE_EXYNOSNNPROFILERCONFIG_H_

/*
 * The level of ExynosNN Profiler set by "setprop vendor.enn.profile [value]" implies scope and output level.
 */

/*
 * The profiler determines the scope to attempt the profile by this scope level.
 * By executing commandline, "setprop vendor.enn.profile [value]"
 * EXYNOS_NN_PROFILER_SCOPE_DISABLED : Disable Profiler
 * EXYNOS_NN_PROFILER_SCOPE_DEFAULT  : Enable with default scopes to be profiled
 */
enum ExynosNnProfilerLevel {
    EXYNOS_NN_PROFILER_DISABLED         = 0,
    EXYNOS_NN_PROFILER_SCOPE_DEFAULT    = 1,
    EXYNOS_NN_PROFILER_ENN_ONLY         = 2,
    EXYNOS_NN_PROFILER_LEVEL_BOUND      = 3
};

/*
 * The profiler determines which profile data to output by this output level.
 * By executing commandline, "setprop vendor.enn.profile.out [value]"
 * EXYNOS_NN_PROFILER_OUTPUT_DEFAULT  : Print out to output stream defined such as logcat
 * EXYNOS_NN_PROFILER_OUTPUT_STDOUT   : Print out to stand output stream using printf function
 * EXYNOS_NN_PROFILER_OUTPUT_EXCLUDED : Print profiled data execluded
 * They can be combined using bitwise "OR" operator.
 */
enum ExynosNnProfilerOutputLevel {
    EXYNOS_NN_PROFILER_OUTPUT_DEFAULT   = 0x01,  // 0b00000001
    EXYNOS_NN_PROFILER_OUTPUT_STDOUT    = 0x02,  // 0b00000010
    EXYNOS_NN_PROFILER_OUTPUT_EXCLUDED  = 0x04   // 0b00000100
};

#endif  // TOOLS_PROFILER_INCLUDE_EXYNOSNNPROFILERCONFIG_H_
