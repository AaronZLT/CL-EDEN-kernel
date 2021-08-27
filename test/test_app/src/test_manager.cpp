/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include "test_manager.h"
#include "enn_test_type.hpp"

enn_test::TestParams test_param;

TestManager::TestManager() {
}

bool TestManager::fill_test_param(int argc, char** argv) {
    int32_t ret = cli_parser.parse_commandline(argc, argv, test_param);
    if (ret != enn_test::RET_SUCCESS) {
        return false;
    }
    for (int i = 0; i < argc; ++i) {
        // skip to print google test usage
        // other gtest options will be caused invalid argument by cli_parser
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            return false;
        }
    }
    ::testing::InitGoogleTest(&argc, argv);

    return true;
}

int TestManager::run_tests() {
    ::testing::GTEST_FLAG(filter) = test_param.gtest_filter;
    int ret = RUN_ALL_TESTS();

    return ret;
}
