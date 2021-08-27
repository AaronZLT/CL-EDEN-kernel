/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _TEST_MANAGER_H
#define _TEST_MANAGER_H

#include <gtest/gtest.h>
#include "cli_parser.h"

/**
 * @class   TestManager
 * @brief   This is TestManager Class
 * @details This class deals with detecting type of test
 *          Filling of test parameters and running the test
 */

class TestManager {
 private:
    CliParser cli_parser;

 public:
    TestManager();
    bool fill_test_param(int argc, char** argv);
    int run_tests();
};
#endif
