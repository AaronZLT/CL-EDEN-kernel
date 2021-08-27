/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include <map>
#include <algorithm>    // transform
#include "cli_parser.h"

#define DEFAULT_PATH "/data/vendor/enn/"

bool should_skip_print = false;

bool CliParser::has_minimum_info() {
    bool ret = false;

    if ((bool)*cli_options.model) {
        if ((bool)*cli_options.input && (bool)*cli_options.golden) {
            ret = true;
        } else {
            std::cout << "Error :: input and golden both should be enterd with model \n";
        }
    } else {
        std::cout << "Error :: one of the following argument required:--model/--input/--golden\n";
    }

    return ret;
}

int32_t CliParser::fill_other_params(enn_test::TestParams& test_param, std::string executionMode) {
    if (!has_minimum_info()) {
        return enn_test::RET_INVALID_PARAM;
    }

    if (test_param.modelPath.find("/") == std::string::npos) {
        test_param.modelPath.insert(0, DEFAULT_PATH);
    }

    for (auto& in_path : test_param.inputPath) {
        if (in_path.find("/") == std::string::npos) {
            in_path.insert(0, DEFAULT_PATH);
        }
    }

    for (auto& out_path : test_param.goldenPath) {
        if (out_path.find("/") == std::string::npos) {
            out_path.insert(0, DEFAULT_PATH);
        }
    }

    std::map<std::string, enn_test::EnnPerfMode> execmode = { {"normal", enn_test::NORMAL},
                                                              {"boost", enn_test::BOOST},
                                                              {"boe", enn_test::BOE},
                                                              {"bb", enn_test::BB} };

    transform(executionMode.begin(), executionMode.end(), executionMode.begin(), ::tolower);
    if (execmode.find(executionMode) != execmode.end()) {
        test_param.executionMode = execmode[executionMode];
    } else if (executionMode != "") {
        std::cout << "Error :: invalid mode : " << executionMode << std::endl;
        return enn_test::RET_INVALID_PARAM;
    }

    return enn_test::RET_SUCCESS;
}

int32_t CliParser::parse_commandline(int argc, char** argv, enn_test::TestParams& test_param) {
    int32_t ret = enn_test::RET_SUCCESS;
    std::string executionMode = "";

    app.add_option_group("User Defined Test");
    app.add_option_group("Optional");

    cli_options.model = app.add_option("--model", test_param.modelPath, "--model model_path");
    cli_options.model->group("User Defined Test");

    cli_options.input = app.add_option("--input", test_param.inputPath,
                    "--input input1.bin input2.bin ...");
    cli_options.input->group("User Defined Test");

    cli_options.golden = app.add_option("--golden", test_param.goldenPath,
                    "--golden golden1.bin golden2.bin ...");
    cli_options.golden->group("User Defined Test");

    cli_options.filter = app.add_option("--filter", test_param.gtest_filter, "Test cases\n"
                    " - INSTANCE_TEST.RUN : Normal end-to-end test (default)");
    cli_options.filter->group("Optional");

    cli_options.mode = app.add_option("--mode", executionMode, "Execution performance mode\n"
                    " - NORMAL : no clock boosting\n"
                    " - BOOST : boost clock from open ~ close\n"
                    " - BOE : boost clock only in execution (BOOST_ON_EXECUTE)\n"
                    " - BB : internal use only (BOOST_BLOCK)\n");
    cli_options.mode->group("Optional");

    cli_options.iter = app.add_option("--iter", test_param.iter, "Execution iteration count");
    cli_options.iter->group("Optional")->check(CLI::PositiveNumber);

    cli_options.duration = app.add_option("--duration", test_param.duration, "Execte for input seconds");
    cli_options.duration->group("Optional")->check(CLI::PositiveNumber);

    cli_options.repeat = app.add_option("--repeat", test_param.repeat, "Test repeat count");
    cli_options.repeat->group("Optional")->check(CLI::PositiveNumber);

    cli_options.threshold = app.add_option("--threshold", test_param.threshold, "Threshold used "
                    "while Golden comparison\t(Default threshold : 0)\n"
                    "[Commonly used threshold]\n"
                    " - NPU models (float output) : 0.0001\n"
                    " - GPU models : 0.017\n"
                    " - SSD(Single Shot Detection) models : 0.03\n");
    cli_options.threshold->group("Optional")->check(CLI::Range(0.0f, FLT_MAX));

    cli_options.skip_match = app.add_flag("--skip_match", test_param.skipMatch, "Skip golden matching "
                    "when this flag is set");
    cli_options.skip_match->group("Optional");

    cli_options.report = app.add_option("--report", test_param.reportPath, "report file path");
    cli_options.report->group("Optional");

    cli_options.async = app.add_flag("--async", test_param.isAsync, "Use async execution API "
                    "when this flag is set");
    cli_options.async->group("Optional");

    cli_options.thread = app.add_option("--thread", test_param.thread_num, "Number of Execute threads");
    cli_options.thread->group("Optional")->check(CLI::PositiveNumber);

    cli_options.no_log = app.add_flag("--no_log", should_skip_print, "Print no log");
    cli_options.no_log->group("Optional");

    cli_options.dump_output = app.add_flag("--dump_output", test_param.dump_output, "Dump last out buffer to "
                    +default_path+"output_{n}.bin (n = output number)");
    cli_options.dump_output->group("Optional");

    cli_options.preset_id = app.add_option("--preset_id", test_param.preset_id, "Apply given preset Id. "
                    "(default : 0)\n'--preset_id' always take precedence over '--mode'");
    cli_options.preset_id->group("Optional")->check(CLI::PositiveNumber);

    cli_options.target_latency = app.add_option("--target_latency", test_param.target_latency, "Apply given target latency. "
                    "(default : 0)");
    cli_options.target_latency->group("Optional")->check(CLI::PositiveNumber);

    cli_options.priority = app.add_option("--priority", test_param.priority, "Apply given priority. (default : 0)");
    cli_options.priority->group("Optional")->check(CLI::NonNegativeNumber);

    cli_options.tile_num = app.add_option("--tile_num", test_param.tile_num, "Apply given tile num. (default : 0)");
    cli_options.tile_num->group("Optional")->check(CLI::NonNegativeNumber);

    cli_options.core_affinity = app.add_option("--core_affinity", test_param.core_affinity, "Apply given core affinity.");
    cli_options.core_affinity->group("Optional")->check(CLI::NonNegativeNumber);

    cli_options.delay = app.add_option("--delay", test_param.delay, "");
    cli_options.delay->group("Optional")->check(CLI::PositiveNumber);

    cli_options.error = app.add_option("--error", test_param.error, "");
    cli_options.error->group("Optional");

    CLI11_PARSE(app, argc, argv);

    ret = fill_other_params(test_param, executionMode);

    return ret;
}

