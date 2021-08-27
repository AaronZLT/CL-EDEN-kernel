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
 * @file    ExynosNnProfiler.h
 * @brief   There are functions for profiling any scopes by users.
 * @details These functions can be used on C code also.
 * @version 1
 */

#ifndef TOOLS_PROFILER_INCLUDE_EXYNOSNNPROFILER_H_
#define TOOLS_PROFILER_INCLUDE_EXYNOSNNPROFILER_H_


#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

struct CalculatedProfileNode {
            char* label;
            unsigned int duration;
            struct CalculatedProfileNode** child;
};


uint8_t*    profile_level_is(void);
void        profile_from(const char* custom_label, uint64_t id);
void        profile_from2(const char* custom_label, uint64_t id, int32_t op_num);
void        profile_until(const char* custom_label, uint64_t id);
void        profile_until2(const char* custom_label, uint64_t id, int32_t op_num);
void        profile_append(struct CalculatedProfileNode* calculated_profile_node, uint64_t id);
void        profile_exclude_from(const char* custom_label, uint64_t id);
void        profile_exclude_until(const char* custom_label, uint64_t id);

#ifdef __cplusplus
}
#endif  // __cplusplus


#endif // TOOLS_PROFILER_INCLUDE_EXYNOSNNPROFILER_H_
