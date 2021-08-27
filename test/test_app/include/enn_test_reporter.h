/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */


#ifndef ENN_TEST_REPORTER_H_
#define ENN_TEST_REPORTER_H_

#include <fstream>
#include <string>
#include <vector>

#include "enn_test_type.hpp"

namespace enn_test {

class TestReporter {
    ReportType report_type;

    const std::string default_path = "/data/vendor/enn/results/";
    std::string file_name;
    std::ofstream wf;

    int32_t reporter_id;
    int32_t total_repeat;
    int32_t total_iteration;
    int64_t pass_num = 0;
    bool is_off = false;

    void CreateDefaultPath();
    std::string GetNewFileName(std::string name);
    void ReportToConsole();
    void ReportToFile();

  public:
    TestReporter() {
        is_off = true;
    }

    TestReporter(int32_t id, int32_t repeat, int32_t iter)
     : report_type(REPORT_TO_CONSOLE), reporter_id(id), total_repeat(repeat), total_iteration(iter) {
    }

    TestReporter(int32_t id, int32_t repeat, int32_t iter, std::string name)
     : report_type(REPORT_TO_FILE), reporter_id(id), total_repeat(repeat), total_iteration(iter) {
        file_name = GetNewFileName(name);
        wf.open(file_name, std::ios::app);
    }

    ~TestReporter() {
        Report();
        if (report_type == REPORT_TO_FILE && wf.is_open()) {
            wf.close();
        }
    }

    void Report();
    void IncreasePass();
};

}  // namespace enn_test
#endif  // ENN_TEST_REPORTER_H_
