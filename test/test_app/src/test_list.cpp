/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include <string>
#include <gtest/gtest.h>

#include "enn_test.h"

extern enn_test::TestParams test_param;

TEST(INSTANCE_TEST, RUN) {
    test_param.print_param();
    enn_test::EnnTest test(test_param);

    for (int i = 0; i < test_param.repeat; ++i) {
        enn_test::EnnTestReturn ret = test.run();
        EXPECT_EQ(ret, enn_test::RET_SUCCESS);
        enn_test::print_result(ret);
    }
}
