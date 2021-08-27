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

#include "tool/dumper/utilization_table.hpp"

namespace enn {
namespace dump {

class UtilizationDumper {
    static std::unique_ptr<UtilizationDumper> instance_;
    static std::once_flag once_flag_;

    UtilizationDumper() = default;

 public:
    using ID = uint64_t;

    UtilizationDumper(const UtilizationDumper&) = delete;
    UtilizationDumper& operator=(const UtilizationDumper&) = delete;
    UtilizationDumper(UtilizationDumper&&) = delete;
    UtilizationDumper& operator=(UtilizationDumper&&) = delete;

    static UtilizationDumper* get_instance() {
        std::call_once(once_flag_, []() {
            instance_.reset(new UtilizationDumper);
        });
        return instance_.get();
    }

    void start_dump() {
        std::lock_guard<std::mutex> lock_guard(m_util_table_map_);
        util_table_ = std::make_unique<UtilizationTable>();
    }

    void finish_dump() {
        std::lock_guard<std::mutex> lock_guard(m_util_table_map_);
        util_table_.reset();
    }

 private:
    std::unique_ptr<UtilizationTable> util_table_;
    std::mutex m_util_table_map_;
};

std::unique_ptr<UtilizationDumper> UtilizationDumper::instance_;
std::once_flag UtilizationDumper::once_flag_;

};  // namespace dump
};  // namespace enn
