/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#ifndef _CLI_PARSER_H
#define _CLI_PARSER_H

#include "CLI11.hpp"
#include "enn_test_type.hpp"

struct CliOptions {
    CLI :: Option* model;
    CLI :: Option* input;
    CLI :: Option* golden;

    CLI :: Option* filter;
    CLI :: Option* mode;
    CLI :: Option* iter;
    CLI :: Option* duration;
    CLI :: Option* repeat;
    CLI :: Option* threshold;
    CLI :: Option* skip_match;
    CLI :: Option* report;
    CLI :: Option* async;
    CLI :: Option* thread;
    CLI :: Option* no_log;
    CLI :: Option* dump_output;
    CLI :: Option* preset_id;
    CLI :: Option* target_latency;
    CLI :: Option* priority;
    CLI :: Option* tile_num;
    CLI :: Option* core_affinity;

    CLI :: Option* delay;
    CLI :: Option* error;
};

class CliParser {
 private:
    CLI::App app{"ENN TEST FRAMEWORK"};

    bool has_minimum_info();
    int32_t fill_other_params(enn_test::TestParams& test_param, std::string executionMode);

 public:
    CliOptions cli_options;
    int32_t parse_commandline(int argc, char** argv, enn_test::TestParams& test_param);
};

#endif
