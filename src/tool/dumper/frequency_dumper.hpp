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

#include <map>
#include <memory>
#include <mutex>
#include <string>

#include "tool/dumper/frequency_table.hpp"

namespace enn {
namespace dump {

class FrequencyDumper {
    static std::unique_ptr<FrequencyDumper> instance_;
    static std::once_flag once_flag_;

    FrequencyDumper() = default;

 public:
    using ID = uint64_t;

    FrequencyDumper(const FrequencyDumper&) = delete;
    FrequencyDumper& operator=(const FrequencyDumper&) = delete;
    FrequencyDumper(FrequencyDumper&&) = delete;
    FrequencyDumper& operator=(FrequencyDumper&&) = delete;

    static FrequencyDumper* get_instance() {
        std::call_once(once_flag_, []() {
            instance_.reset(new FrequencyDumper);
        });
        return instance_.get();
    }

    void start_dump() {
        std::lock_guard<std::mutex> lock_guard(m_freq_table_map_);
        freq_table_ = std::make_unique<FrequencyTable>();
    }

    void finish_dump() {
        std::lock_guard<std::mutex> lock_guard(m_freq_table_map_);
        freq_table_.reset();
    }

 private:
    std::unique_ptr<FrequencyTable> freq_table_;
    std::mutex m_freq_table_map_;
};

std::unique_ptr<FrequencyDumper> FrequencyDumper::instance_;
std::once_flag FrequencyDumper::once_flag_;

};  // namespace dump
};  // namespace enn