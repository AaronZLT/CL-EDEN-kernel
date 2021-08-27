/*
 * Copyright (C) 2020 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or
 * distributed, transmitted, transcribed, stored in a retrieval system or
 * translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed to third parties
 * without the express written permission of Samsung Electronics.
 *
 */

/**
 * @brief gtest main for unit test
 * @file enn_gtest_internal_unittest_main.cc
 * @author Hoon Choi
 * @date 2020_12_10
 */

#include "gtest/gtest.h"
namespace enn {
namespace test {
namespace internal {

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}

}  // namespace internal
}  // namespace test
}  // namespace enn
