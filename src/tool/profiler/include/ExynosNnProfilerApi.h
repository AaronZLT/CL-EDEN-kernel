/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
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

#ifndef TOOL_PROFILER_INCLUDE_EXYNOSNNPROFILERAPI_H_
#define TOOL_PROFILER_INCLUDE_EXYNOSNNPROFILERAPI_H_

#include "tool/profiler/include/ExynosNnProfiler.hpp"
#include "tool/profiler/include/ExynosNnProfiler.h"

/*
 * @ Description of API
 *
 * #START_PROFILER
 *  - Turn on ExynosNN Profiler depending on model id
 *  - Calling it, ExynosNN Profiler starts
 *
 * #FINISH_PROFILER
 *  - Turn off ExynosNN Profiler depending on model id
 *  - Calling it, ExynosNN Profiler prints result and finishes
 *
 * #PROFIE_SCOPE_HERE
 *  - Record start point and end point using RAII pattern depending on model id
 *  - Label with function name, line, and file name without user-defined label
 *
 * #PROFILE_SCOPE
 *  - Record start point and end point using RAII pattern depending on model id
 *  - Label with user-defined label
 *
 * #PROFILE_FROM
 *  - Record a start point depending on model id
 *  - Label with user-defined label which should coincide with one of PROFILE_UNTIL
 *
 * #PROFILE_UNTIL
 *  - Record a end point depending on model id
 *  - Label with user-defined label which should coincide with one of PROFILE_FROM
 *
 * #PROFILE_APPEND
 *  - Append a subtree to profile_tree as struct of CalculatedProfileNode depending on model id
 *  - The parameter, SUB_TREE should be allocated dynamically using malloc function, not new operator.
 *
 * #PROFILE_EXCLUDE_FROM
 *  - Record a start point of scope to be excluded from duration profiled
 *  - Label with user-defined label which should coincide with one of PROFILE_EXCLUDE_UNTIL
 *
 * #PROFILE_EXCLUDE_UNTIL
 *  - Record a end point of scope to be excluded from duration profiled
 *  - Label with user-defined label which should coincide with one of PROFILE_EXCLUDE_FROM
 *
 * #PROFILE_LEVEL_IS
 *  - Check the profile level which is set with "setprop vendor.enn.profile [value]"
 */

#ifdef EXYNOS_NN_PROFILER

#define GET_4TH(arg1, arg2, arg3, FUNC, ...) FUNC
#define START_PROFILER(ID)\
                                StartProfiler start_profiler(ID)
#define FINISH_PROFILER(ID)\
                                FinishProfiler finish_profiler(ID)
#define PROFILE_SCOPE_HERE(ID)\
                                ScopedProfiling scoped_profiling(__FILE__, __LINE__, __func__, ID)
#define PROFILE_SCOPE_WITH_LABEL(LABEL, ID)\
                                ScopedProfiling scoped_profiling(LABEL, ID)
#define PROFILE_SCOPE_WITH_LABEL_AND_OP_NUM(LABEL, ID, OP_NUM)\
                                ScopedProfiling scoped_profiling(LABEL, ID, OP_NUM)
#define PROFILE_SCOPE(...)\
                        GET_4TH(__VA_ARGS__,\
                                PROFILE_SCOPE_WITH_LABEL_AND_OP_NUM(__VA_ARGS__),\
                                PROFILE_SCOPE_WITH_LABEL(__VA_ARGS__))
#define PROFILE_FROM_WITH_LABEL(LABEL, ID)\
                                profile_from(LABEL, ID)
#define PROFILE_FROM_WITH_LABEL_AND_OP_NUM(LABEL, ID, OP_NUM)\
                                profile_from_with_op_num(LABEL, ID, OP_NUM)
#define PROFILE_FROM(...)\
                        GET_4TH(__VA_ARGS__,\
                                PROFILE_FROM_WITH_LABEL_AND_OP_NUM(__VA_ARGS__),\
                                PROFILE_FROM_WITH_LABEL(__VA_ARGS__))
#define PROFILE_UNTIL_WITH_LABEL(LABEL, ID)\
                                profile_until(LABEL, ID)
#define PROFILE_UNTIL_WITH_LABEL_AND_OP_NUM(LABEL, ID, OP_NUM)\
                                profile_until_with_op_num(LABEL, ID, OP_NUM)
#define PROFILE_UNTIL(...)\
                        GET_4TH(__VA_ARGS__,\
                                PROFILE_UNTIL_WITH_LABEL_AND_OP_NUM(__VA_ARGS__),\
                                PROFILE_UNTIL_WITH_LABEL(__VA_ARGS__))
#define PROFILE_APPEND(SUB_TREE, ID)\
                                profile_append(SUB_TREE, ID)
#define PROFILE_EXCLUDE_FROM(LABEL, ID)\
                                profile_exclude_from(LABEL, ID)
#define PROFILE_EXCLUDE_UNTIL(LABEL, ID)\
                                profile_exclude_until(LABEL, ID)
#define PROFILE_LEVEL_IS()\
                                *profile_level_is()

#else  // !EXYNOS_NN_PROFILER

#define START_PROFILER(ID)
#define FINISH_PROFILER(ID)
#define PROFILE_SCOPE_HERE(ID)
#define PROFILE_SCOPE(...)
#define PROFILE_FROM(...)
#define PROFILE_UNTIL(...)
#define PROFILE_APPEND(SUB_TREE, ID)
#define PROFILE_EXCLUDE_FROM(LABEL, ID)
#define PROFILE_EXCLUDE_UNTIL(LABEL, ID)
#define PROFILE_LEVEL_IS()

#endif // EXYNOS_NN_PROFILER

#endif // TOOL_PROFILER_INCLUDE_EXYNOSNNPROFILERAPI_H_
