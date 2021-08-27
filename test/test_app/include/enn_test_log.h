/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */


#ifndef ENN_TEST_LOG_H_
#define ENN_TEST_LOG_H_

#if defined(ANDROID_LOG)
#include <android/log.h>

#define ENN_TEST_DEBUG(fmt, ...) \
    __android_log_print(ANDROID_LOG_DEBUG, "ENN_TEST", "%s:%d: " fmt, __FUNCTION__, __LINE__,\
            ##__VA_ARGS__);

#define ENN_TEST_INFO(fmt, ...) \
    __android_log_print(ANDROID_LOG_INFO, "ENN_TEST", "%s:%d: " fmt, __FUNCTION__, __LINE__,\
            ##__VA_ARGS__);

#define ENN_TEST_WARN(fmt, ...) \
    __android_log_print(ANDROID_LOG_WARN, "ENN_TEST", "%s:%d: " fmt, __FUNCTION__, __LINE__,\
            ##__VA_ARGS__);

#define ENN_TEST_ERR(fmt, ...) \
    __android_log_print(ANDROID_LOG_ERROR, "ENN_TEST", "%s:%d: " fmt, __FUNCTION__, __LINE__,\
            ##__VA_ARGS__);

#else

#define ENN_TEST_DEBUG(...)
#define ENN_TEST_INFO(...)
#define ENN_TEST_WARN(...)
#define ENN_TEST_ERR(...)

#endif


#endif  // ENN_TEST_LOG_H_
