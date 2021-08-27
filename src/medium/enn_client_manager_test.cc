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
 * @file enn_gtest_internal_unittest_temp.cc
 * @author Hoon Choi
 * @date 2020_12_10
 */

#include "client/enn_api-public.h"
#include "medium/enn_client_manager.hpp"
#include "gtest/gtest.h"

namespace enn {
namespace test {
namespace internal {

class ENN_GT_CLIENT_MANAGER_TEST : public testing::Test {};

TEST_F(ENN_GT_CLIENT_MANAGER_TEST, client_creation) {
    enn::interface::ClientManager<int32_t, char *> pool;
    char test_cb = 'a';
    pool.show();
    pool.GetClient(3);
    EXPECT_EQ(pool.GetClient(2123), nullptr);
    EXPECT_NE(pool.PutClient(1), false);
    pool.show();
    EXPECT_NE(pool.PutClient(2123, &test_cb), false);
    EXPECT_EQ(pool.PutClient(1), true);
    EXPECT_EQ(pool.PutClient(1, &test_cb), true);

    for (int i = 0; i < 10; i++) EXPECT_EQ(pool.PutClient(i), true);
    pool.show();
    EXPECT_EQ(pool.PopClient(2123), true);
    EXPECT_EQ(pool.PopClient(1), true);
    EXPECT_EQ(pool.PopClient(1), true);
    EXPECT_EQ(pool.PopClient(1), true);
    pool.show();
    for (int i = 0; i < 10; i++) EXPECT_EQ(pool.PopClient(i), true);
    pool.show();
    for (int i = 0; i < 10; i++) EXPECT_EQ(pool.PopClient(i), false);
}



}  // namespace internal
}  // namespace test
}  // namespace enn
