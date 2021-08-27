#include <iostream>
#include <climits>
#include <cstdlib>
#include <map>
#include <mutex>
#include <memory>
#include <vector>
#include <cstring>
#include <cstdarg>
#include <fstream>
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>

#include "gtest/gtest.h"
#include "common/enn_debug.h"
#include "common/enn_utils.h"

#ifndef __ANDROID__
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}
#endif

namespace enn {
namespace test {
namespace internal {

class ENN_GT_API_UTIL_TEST:  public testing::Test {};

TEST_F(ENN_GT_API_UTIL_TEST, comparer_int_success) {
    int source[10] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    EXPECT_EQ(0, enn::util::CompareBuffersWithThreshold<int>(source, source, sizeof(source)));
}

TEST_F(ENN_GT_API_UTIL_TEST, comparer_int_failed) {
    int source[10] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    int target[10] = { 1, 2, 3, 4, 2, 6, 7, 8, 9, 10 };
    EXPECT_EQ(1, enn::util::CompareBuffersWithThreshold<int>(source, target, sizeof(source)));
}

TEST_F(ENN_GT_API_UTIL_TEST, comparer_float_failed) {
    float sourcef[5] = { 1, 2, 3, 4, 5 };
    float targetf[5] = { 1, 2.0, 3.0, 4.2, 5 };
    EXPECT_EQ(1, enn::util::CompareBuffersWithThreshold<float>(sourcef, targetf, sizeof(sourcef), nullptr, 0.1));
}

TEST_F(ENN_GT_API_UTIL_TEST, comparer_double_failed) {
    double sourced[5] = { 1, 2, 3, 4, 5 };
    double targetd[5] = { 1, 2.0, 3.0, 4.2, 5 };
    EXPECT_EQ(1, enn::util::CompareBuffersWithThreshold<double>(sourced, targetd, sizeof(sourced), nullptr, 0.1));
}

TEST_F(ENN_GT_API_UTIL_TEST, comparer_double_pass) {
    double sourced[5] = { 1, 2, 3, 4, 5 };
    double targetd[5] = { 1, 2.0, 3.0, 4.1, 5 };
    EXPECT_EQ(0, enn::util::CompareBuffersWithThreshold<double>(sourced, targetd, sizeof(sourced), nullptr, 0.2));
}

TEST_F(ENN_GT_API_UTIL_TEST, comparer_double_get_result_map) {
    uint8_t result_map[5];
    double sourced[5] = { 1, 2, 3, 4, 5 };
    double targetd[5] = { 1, 2.0, 3.0, 4.1, 5 };
    EXPECT_EQ(0, enn::util::CompareBuffersWithThreshold<double>(sourced, targetd, sizeof(sourced), result_map, 0.1));
}

}  // namespace internal
}  // namespace test
}  // namespace enn
