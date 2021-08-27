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

#include "common/enn_debug.h"

/**
* @brief Using FrequencyTable's constructor dump frequency table with delta as csv file
* @details write csv file in FrequencyTable::dir
*          use RAII Pattern for pairs of both start and end points
**/

namespace enn {
namespace dump {

class FrequencyTable {
    // TODO(yc18.cho, TBD) : Extract strings as configure file for scailability
    /* Path in device to save freq tables as csv file */
    std::string dir = "/data/vendor/enn/dump/frequency/";
    std::string file = "/data/vendor/enn/dump/frequency/exynos_nn_frequency_dump.csv";
    /* Constant map including Hareware - table frequency for time in a state */
    std::map<std::string, std::string> freqTableConfigs = {
        {"NPU", "/sys/class/devfreq/17000060.devfreq_npu/time_in_state"},
        {"DSP", "/sys/class/devfreq/170000d0.devfreq_dsp/time_in_state"},
        {"DNC", "/sys/class/devfreq/17000080.devfreq_dnc/time_in_state"},
        {"MIF", "/sys/class/devfreq/17000010.devfreq_mif/time_in_state"},
        {"INT", "/sys/class/devfreq/17000020.devfreq_int/time_in_state"},
        {"LITTLE_CPU", "/sys/devices/system/cpu/cpufreq/policy0/stats/time_in_state"},
        {"MIDDLE_CPU", "/sys/devices/system/cpu/cpufreq/policy4/stats/time_in_state"},
        {"BIG_CPU", "/sys/devices/system/cpu/cpufreq/policy7/stats/time_in_state"}
    };
    /* The freqTable at the time when this function is called for the first time is
    * saved, and when the function is called next, the diff of the count value of
    * the map below is calculated with the freqTable of next and saved.
    * Format of first call is { HW's Name : { Frequency : Count } }
    * Foramat of second call in { HW's Name : { Frequency : Delta from previous count } } */
    std::map<std::string, std::map<int32_t, int32_t>> dataMap;
    std::ofstream dumpCsvFile;

 public:
    FrequencyTable() {
        collect(true);
    }

    ~FrequencyTable() {
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
    /*
    * Execute commandline and then return the output as string
    */
    std::string executeCommand(const std::string& cmd) {
        auto pPipe = ::popen(cmd.c_str(), "r");
        if (pPipe == nullptr) {
            ENN_ERR_PRINT("Cannot open pipe\n");
            return "ERROR";
        }
        // std::cout << cmd << std::endl;
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
        /* For readabiltiy in csv file*/
        constexpr int32_t kHz = 1000;

        for (auto& config : freqTableConfigs) {
            // cout << config.first << endl;
            std::stringstream stateTable(executeCommand("cat " + config.second));
            // cout << stateTable.str() << endl;
            // cout << " " << endl;

            /* Parsing Key(frequency) - Value(count) */
            std::string tuple, frequency, count;
            int32_t frequencyKHz, countMs;
            bool isFirstTulple = true;
            while (getline(stateTable, tuple)) {
                std::stringstream stream(tuple);
                stream >> frequency >> count;
                frequencyKHz = stoi(frequency) / kHz;
                countMs = stoi(count);

                /* trigger is true : Start point for dump frequecy's tables
                * trigger is false : End point for dump frequency's tables */
                if (trigger) {
                    dataMap[config.first].insert(std::make_pair(frequencyKHz, countMs));
                } else {
                    dataMap[config.first][frequencyKHz] = countMs - dataMap[config.first][frequencyKHz];
                    if (isFirstTulple) {
                        dumpCsvFile << "<" + config.first + ">\n";
                        dumpCsvFile << "Frequency,Delta Of Count\n";
                        isFirstTulple = false;
                    }
                    dumpCsvFile << std::to_string(frequencyKHz) + ","
                                 + std::to_string(dataMap[config.first][frequencyKHz]) + "\n";
                }
            }
            if (!trigger) dumpCsvFile<< "\n";
        }
        /* Clear up dataMap using is done */
        if (!trigger) dataMap.clear();
    }
};

};  // namespace dump
};  // namespace enn

#endif // __cplusplus
