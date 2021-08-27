#ifndef SRC_COMMON_COMPILER_H_
#define SRC_COMMON_COMPILER_H_

#define likely(x)       __builtin_expect(!!(x), 1)
#define unlikely(x)     __builtin_expect(!!(x), 0)

#endif  // SRC_COMMON_COMPILER_H_
