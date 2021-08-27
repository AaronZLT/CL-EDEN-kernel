#pragma once

#include "Includes.hpp"
#include "Common.hpp"
#include "Debug.hpp"

#define ASSERT(condition) assert(condition)

#define COMPUTE_LIBRARY_SUCCESS 0
#define COMPUTE_LIBRARY_UNKNOWN_ERROR -1

#define CHECK_EXPR_NO_RETURN(expr, errorMsgFormat, ...) \
    do {                                                \
        if (!(expr)) {                                  \
            ERROR_PRINT(errorMsgFormat, ##__VA_ARGS__); \
            ERROR_PRINT("\n");                          \
        }                                               \
    } while (0)

#define CHECK_EXPR_RETURN_FAILURE(expr, errorMsgFormat, ...) \
    do {                                                     \
        if (!(expr)) {                                       \
            ERROR_PRINT(errorMsgFormat, ##__VA_ARGS__);      \
            ERROR_PRINT("\n");                               \
            return Status::FAILURE;                          \
        }                                                    \
    } while (0)

#define CHECK_EXPR_RETURN_NULL(expr, errorMsgFormat, ...) \
    do {                                                  \
        if (!(expr)) {                                    \
            ERROR_PRINT(errorMsgFormat, ##__VA_ARGS__);   \
            ERROR_PRINT("\n");                            \
            return NULL;                                  \
        }                                                 \
    } while (0)

#define CHECK_EXPR_TERMINATE(expr, errorMsgFormat, ...) \
    do {                                                \
        if (!(expr)) {                                  \
            ERROR_PRINT(errorMsgFormat, ##__VA_ARGS__); \
            ERROR_PRINT("\n");                          \
            exit(-1);                                   \
        }                                               \
    } while (0)

#define ERROR_PRINT_RETURN_FAILURE(message, ...) \
    ERROR_PRINT(message, ##__VA_ARGS__);         \
    return Status::FAILURE;

#define ERROR_PRINT_RETURN_CL_FAILURE(message, ...) \
    ERROR_PRINT(message, ##__VA_ARGS__);            \
    return Status::CL_FAILURE;

#define ERROR_PRINT_RETURN_NULL(message, ...) \
    ERROR_PRINT(message, ##__VA_ARGS__);      \
    return NULL;
