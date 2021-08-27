/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include <sys/stat.h>   // mkdir
#include <unistd.h>     // access

#include "enn_test_reporter.h"
#include "enn_test_log.h"

namespace enn_test {

void TestReporter::CreateDefaultPath() {
    if (access(default_path.c_str(), F_OK) != 0) {
        ENN_TEST_DEBUG("Create default path : %s", default_path.c_str());
        mkdir(default_path.c_str(), 0755);

        if (access(default_path.c_str(), F_OK) != 0) {
            ENN_TEST_ERR("Create default path failed");
            throw RET_FILE_IO_ERROR;
        }
    }
}

std::string TestReporter::GetNewFileName(std::string name) {
    CreateDefaultPath();
    return default_path + name;
}

void TestReporter::ReportToFile() {
    int64_t total_execution = (int64_t)total_repeat * total_iteration;
    std::string total_execution_str = std::to_string(total_execution);

    if (wf.is_open()) {
        wf << "-----------------------------------------------------------\n";
        wf << " ID : " + std::to_string(reporter_id) + "\n";
        wf << " Total : " + total_execution_str + "\n";
        wf << " Pass : " + std::to_string(pass_num) + "\n";
        wf << " Fail : " + std::to_string(total_execution - pass_num) + "\n";
        wf << "-----------------------------------------------------------\n";
    }
}

void TestReporter::ReportToConsole() {
    int64_t total_execution = (int64_t)total_repeat * total_iteration;

    PRINT_FORCE("-----------------------[ Summary : %d ]-----------------------\n", reporter_id);
    PRINT_FORCE(" Total : %ld  (Repeats : %d,  Iterations : %d)\n",
            total_execution, total_repeat, total_iteration);
    if (pass_num == total_execution) {
        PRINT_GREEN_FORCE(" Pass : [ %ld / %ld ]\n", pass_num, total_execution);
    } else {
        PRINT_FORCE(" Pass : [ %ld / %ld ]\t", pass_num, total_execution);
        PRINT_RED_FORCE(" Fail : [ %ld / %ld ]\n", total_execution - pass_num, total_execution);
    }

    PRINT_FORCE("--------------------------------------------------------------\n");
}

void TestReporter::Report() {
    if (is_off) {
        return;
    }

    if (report_type == REPORT_TO_FILE) {
        ReportToFile();
    } else {
        ReportToConsole();
    }
}

void TestReporter::IncreasePass() {
    pass_num++;
}

}  // namespace enn_test
