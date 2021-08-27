#pragma once

#include "common/enn_debug.h"
#include "common/enn_utils.h"

#ifndef ENN_BUILD_RELEASE
#define DEBUG_PRINT(message, ...) ENN_DBG_PRINT(message, ##__VA_ARGS__)
#else
#define DEBUG_PRINT(message, ...)
#endif
#define ERROR_PRINT(message, ...) ENN_ERR_PRINT(message, ##__VA_ARGS__)

#ifndef UNUSED
#define UNUSED ENN_UNUSED
#endif
