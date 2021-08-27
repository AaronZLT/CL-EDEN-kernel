/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */


#pragma once

#ifdef __cplusplus

#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <array>
#include <sys/time.h>

#include "common/enn_debug.h"

/**
* @brief Using UtilizationTable's constructor dump utilization table with delta as csv file
* @details write csv file in UtilizationTable::dir
*          use RAII Pattern for pairs of both start and end points
**/

namespace enn {
namespace dump {

class UtilizationTable {
    /* csv file path to save the utilization */
    std::string dir = "/data/vendor/enn/dump/utilization/";
    std::string file = "/data/vendor/enn/dump/utilization/exynos_nn_utilization_dump.csv";
    /* idle time sysfs list for each CPU core */
    std::map<std::string, std::string> idleStateConfigs = {
        {"CPU_CORE_0", "/sys/devices/system/cpu/cpu0/cpuidle"},
        {"CPU_CORE_1", "/sys/devices/system/cpu/cpu1/cpuidle"},
        {"CPU_CORE_2", "/sys/devices/system/cpu/cpu2/cpuidle"},
        {"CPU_CORE_3", "/sys/devices/system/cpu/cpu3/cpuidle"},
        {"CPU_CORE_4", "/sys/devices/system/cpu/cpu4/cpuidle"},
        {"CPU_CORE_5", "/sys/devices/system/cpu/cpu5/cpuidle"},
        {"CPU_CORE_6", "/sys/devices/system/cpu/cpu6/cpuidle"},
        {"CPU_CORE_7", "/sys/devices/system/cpu/cpu7/cpuidle"}
    };

    /* map has a pair of core and its idle time*/
    std::map<std::string, uint64_t> idleTimeMap;
    /* map has a pair of core and its total time*/
    uint64_t totalTime;
    std::ofstream dumpCsvFile;

 public:
    UtilizationTable() {
        collect(true);
    }

    ~UtilizationTable() {
        try {
            /* Create directory if dir doesn't exist in this device */
            if (stoi(executeCommand("[ -d '" + dir + "' ] && echo 0 || echo 1"))) {
                executeCommand("mkdir -p " + dir);
            }
            dumpCsvFile.open(file, std::ofstream::app | std::ofstream::out);
            collect(false);
            dumpCsvFile.close();
        } catch(std::exception& e) {
            std::cout << "Exception : " << e.what() << std::endl;
        }
    }

 private:
    /* Execute commandline and then return the output as string */
    std::string executeCommand(const std::string& cmd) {
        auto pPipe = ::popen(cmd.c_str(), "r");
        if (pPipe == nullptr) {
            ENN_ERR_PRINT("Cannot open pipe\n");
            return "ERROR";
        }

        std::array<char, 256> buffer;
        std::string result;
        while (!std::feof(pPipe)) {
            auto bytes = std::fread(buffer.data(), 1, buffer.size(), pPipe);
            result.append(buffer.data(), bytes);
        }
        ::pclose(pPipe);
        return result;
    }

    void collect(bool trigger) {
        /* Get current time in microseconds */
        struct timeval tv;
        gettimeofday(&tv,NULL);
        uint64_t now = 1000000 * tv.tv_sec + tv.tv_usec;

        if (trigger) {
            totalTime = now;
        } else {
            totalTime = now - totalTime;
        }

        for (auto& config : idleStateConfigs) {
            /* Get idle time for each CPU core */
            std::stringstream idleState0Time(executeCommand("cat " + config.second + "/state0/time"));
            std::stringstream idleState1Time(executeCommand("cat " + config.second + "/state1/time"));

            std::string idleState0TimeString, idleState1TimeString;

            getline(idleState0Time, idleState0TimeString);
            getline(idleState1Time, idleState1TimeString);

            uint64_t idleTime, utilization;
            idleTime = stoull(idleState0TimeString) + stoull(idleState1TimeString);

            /* trigger is true : Start point to dump utilization
             * trigger is false: End point to dump utilization */
            if (trigger) {
                idleTimeMap.insert(std::make_pair(config.first, idleTime));
            } else {
                idleTimeMap[config.first] = idleTime - idleTimeMap[config.first];

                if (idleTimeMap[config.first] > totalTime) {
                    utilization = 0;
                } else {
                    utilization = 10000 - 10000 * idleTimeMap[config.first] / totalTime;
                }

                dumpCsvFile << config.first + " utilization," << utilization / 100 << "." << utilization % 100 << "%\n";
            }
        }
        if (!trigger) {
            dumpCsvFile<< "\n";
            /* Clear up idleTimeMap */
            idleTimeMap.clear();
        }
    }
};

};  // namespace dump
};  // namespace enn

#endif // __cplusplus
