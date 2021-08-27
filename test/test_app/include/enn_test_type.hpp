/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */


#ifndef TEST_PARAMS_H_
#define TEST_PARAMS_H_

#include <string>
#include <vector>
#include <stdio.h>
#include "client/enn_api-type.h"

extern bool should_skip_print;

#define PRINT_FORCE(fmt, ...) printf(fmt, ## __VA_ARGS__)
#define PRINT_RED_FORCE(fmt, ...) printf("\e[0;31m" fmt "\e[0m", ## __VA_ARGS__)
#define PRINT_GREEN_FORCE(fmt, ...) printf("\e[0;32m" fmt "\e[0m", ## __VA_ARGS__)
#define PRINT(fmt, ...) should_skip_print ? 0 : printf(fmt, ## __VA_ARGS__)
#define PRINT_RED(fmt, ...) should_skip_print ? 0 : printf("\e[0;31m" fmt "\e[0m", ## __VA_ARGS__)
#define PRINT_GREEN(fmt, ...) should_skip_print ? 0 : printf("\e[0;32m" fmt "\e[0m", ## __VA_ARGS__)

#define PRINT_RET_FORMAT "======================\n %s\n======================\n"

#define THRESHOLD_MIN 0.000001
const std::string default_path = "/data/vendor/enn/";

namespace enn_test {
typedef enum _EnnPerfMode {
    NORMAL = 0,
    BOOST,
    BOE,
    BB,
    PERF_MODE_SIZE,
} EnnPerfMode;

typedef enum _EnnTestReturn {
    RET_SUCCESS = 0,
    RET_INIT_FAILED,
    RET_OPEN_FAILED,
    RET_ALLOCATE_FAILED,
    RET_LOAD_FAILED,
    RET_COMMIT_FAILED,
    RET_EXECUTE_FAILED,
    RET_CLOSE_FAILED,
    RET_DEINIT_FAILED,
    RET_GOLDEN_MISMATCH,
    RET_FILE_IO_ERROR,
    RET_INVALID_PARAM,
    RET_DUMP_FAILED,
    RET_SET_PREFERENCE_FAILED,
    RET_NUM_OF_BUFFERS_FAILED,
} EnnTestReturn;

inline void print_result(EnnTestReturn ret) {
    switch (ret) {
        case RET_SUCCESS:
            PRINT_GREEN(PRINT_RET_FORMAT, "TEST SUCCESS");
            break;
        case RET_INIT_FAILED:
            PRINT_RED_FORCE(PRINT_RET_FORMAT, "INITIALIZE FAILED");
            break;
        case RET_OPEN_FAILED:
            PRINT_RED_FORCE(PRINT_RET_FORMAT, "OPEN MODEL FAILED");
            break;
        case RET_ALLOCATE_FAILED:
            PRINT_RED_FORCE(PRINT_RET_FORMAT, "ALLOCATION FAILED");
            break;
        case RET_LOAD_FAILED:
            PRINT_RED_FORCE(PRINT_RET_FORMAT, "LOAD FILE FAILED");
            break;
        case RET_EXECUTE_FAILED:
            PRINT_RED_FORCE(PRINT_RET_FORMAT, "EXECUTION FAILED");
            break;
        case RET_CLOSE_FAILED:
            PRINT_RED_FORCE(PRINT_RET_FORMAT, "CLOSE MODEL FAILED");
            break;
        case RET_DEINIT_FAILED:
            PRINT_RED_FORCE(PRINT_RET_FORMAT, "DEINITIALIZE FAILED");
            break;
        case RET_GOLDEN_MISMATCH:
            PRINT_RED_FORCE(PRINT_RET_FORMAT, "GOLDEN MISMATCH");
            break;
        case RET_FILE_IO_ERROR:
            PRINT_RED_FORCE(PRINT_RET_FORMAT, "FILE IO ERROR");
            break;
        case RET_INVALID_PARAM:
            PRINT_RED_FORCE(PRINT_RET_FORMAT, "INVALID PARAMETER");
            break;
        case RET_SET_PREFERENCE_FAILED:
            PRINT_RED_FORCE(PRINT_RET_FORMAT, "RET_SET_PREFERENCE_FAILED");
            break;
        case RET_NUM_OF_BUFFERS_FAILED:
            PRINT_RED_FORCE(PRINT_RET_FORMAT, "GIVEN BUFFER NUMBER IS INVALID");
            break;
        default:
            PRINT_RED_FORCE(PRINT_RET_FORMAT, "TEST FAILED");
    }
}

typedef enum _ReportType {
    REPORT_TO_CONSOLE,
    REPORT_TO_FILE,
} ReportType;

typedef struct _TestParams {
    std::string modelPath;
    std::vector<std::string> inputPath;
    std::vector<std::string> goldenPath;

    std::string gtest_filter;
    uint32_t executionMode;
    uint32_t iter;
    uint32_t duration;
    uint32_t repeat;
    float threshold;
    bool skipMatch;
    std::string reportPath;
    bool isAsync;
    uint32_t session_num;
    uint32_t thread_num;
    bool dump_output;
    uint32_t preset_id;
    uint32_t target_latency;
    uint32_t priority;
    uint32_t tile_num;
    uint32_t core_affinity;

    uint32_t delay;
    int32_t error;

    _TestParams() : modelPath(""), gtest_filter("INSTANCE_TEST.RUN"), executionMode(PERF_MODE_SIZE),
                    iter(1), duration(0), repeat(1), threshold(0), skipMatch(false), reportPath(""),
                    isAsync(false), session_num(1), thread_num(1), dump_output(false),
                    preset_id(0), target_latency(0), priority(0), tile_num(0), core_affinity(0),
                    delay(0), error(0) {
        inputPath.clear();
        goldenPath.clear();
    }

    void print_param() {
        PRINT("/**********************************\n");
        PRINT(" * model : %s\n", modelPath.c_str());
        for (int i = 0; i < inputPath.size(); i++) {
            PRINT(" * input %d : %s\n", i+1, inputPath[i].c_str());
        }
        for (int i = 0; i < goldenPath.size(); i++) {
            PRINT(" * golden %d : %s\n", i+1, goldenPath[i].c_str());
        }
        if (preset_id > 0) {
            PRINT(" * preset ID : %d\n", preset_id);
        } else if (executionMode < PERF_MODE_SIZE) {
            PRINT(" * mode : %d\n", executionMode);
        }
        if (duration == 0) {
            PRINT(" * iter : %d\n", iter);
        } else {
            PRINT(" * duration : %d\n", duration);
        }
        PRINT(" * repeat : %d\n", repeat);

        if (threshold > THRESHOLD_MIN) {
            PRINT(" * Float matching (threshold : %f)\n", threshold);
        } else {
            PRINT(" * Binary matching  (threshold : 0)\n");
        }

        if (skipMatch) {
            PRINT(" * Skip golden matching\n");
        }
        if (isAsync) {
            PRINT(" * Async Execution\n");
        }

        if (target_latency > 0) {
            PRINT(" * target_latency : %d\n", target_latency);
        }
        if (priority > 0) {
            PRINT(" * priority : %d\n", priority);
        }
        if (tile_num > 0) {
            PRINT(" * tile_num : %d\n", tile_num);
        }
        if (core_affinity > 0) {
            PRINT(" * core_affinity : %d\n", core_affinity);
        }

        if (!reportPath.empty()) {
            PRINT(" * reportPath : %s\n", reportPath.c_str());
        }
        PRINT(" **********************************/\n");
    }
} TestParams;

typedef struct _TestBuffers {
    std::vector<std::vector<EnnBufferPtr>> input_buffers;     // [session][idx]
    std::vector<std::vector<EnnBufferPtr>> output_buffers;    // [session][idx]
    std::vector<char*> golden_outputs;
    std::vector<int> golden_sizes;

    uint32_t input_num;
    uint32_t output_num;

    _TestBuffers() : input_num(0), output_num(0) {
        input_buffers.clear();
        output_buffers.clear();
        golden_outputs.clear();
        golden_sizes.clear();
    }

    ~_TestBuffers() {
        for (auto& golden : golden_outputs) {
            if (golden != nullptr) {
                free(golden);
            }
        }
    }
} TestBuffers;
}  // namespace enn_test
#endif  // TEST_PARAMS_H_
