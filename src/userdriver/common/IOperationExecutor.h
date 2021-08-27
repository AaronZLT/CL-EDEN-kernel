/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or
 * distributed, transmitted, transcribed, stored in a retrieval system or
 * translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed to third parties
 * without the express written permission of Samsung Electronics.
 */

#ifndef USERDRIVER_COMMON_IOPERATION_EXECUTOR_H_
#define USERDRIVER_COMMON_IOPERATION_EXECUTOR_H_

#include <sys/stat.h>

#include "client/enn_api-type.h"
#include "model/memory/buffer_table.hpp"
#include "userdriver/common/operator_interfaces/interfaces/IComputeLibrary.h"
#include "userdriver/common/operator_interfaces/userdriver_operator.h"
#include "tool/profiler/include/ExynosNnProfilerApi.h"
#include "common/enn_debug.h"
#include "common/enn_utils.h"

#define DUMP_PATH "/data/vendor/enn/dump/"

namespace enn {
namespace ud {

class IOperationExecutor {
public:
    explicit IOperationExecutor(std::string accelator_, std::shared_ptr<IComputeLibrary> compute_library_)
        : accelator(accelator_), compute_library(compute_library_) {}

    virtual ~IOperationExecutor() = default;

    EnnReturn prepare(UDOperators& operators, UDBuffers& buffers, const model::memory::BufferTable& buffer_table) {
        for (int i = 0; i < operators->size(); i++) {
            auto& op = operators->at(i);
            ENN_DBG_PRINT("[%s]: op->getName() = %s\n", accelator.c_str(), op->getName().c_str());

            std::shared_ptr<UDBuffer> buffer = std::make_shared<UDBuffer>();

            for (auto in_ : op->getInTensors()) {
                std::shared_ptr<ITensor> tensor = compute_library->clone_tensor(in_);
                tensor->set_buffer_ptr(get_buffer_ptr(buffer_table, in_->get_buffer_index()));
                buffer->in.push_back(tensor);
            }
            for (auto out_ : op->getOutTensors()) {
                std::shared_ptr<ITensor> tensor = compute_library->clone_tensor(out_);
                tensor->set_buffer_ptr(get_buffer_ptr(buffer_table, out_->get_buffer_index()));
                buffer->out.push_back(tensor);
            }
            for (auto data_ : op->getDataTensors()) {
                std::shared_ptr<ITensor> tensor = compute_library->clone_tensor(data_);
                tensor->set_buffer_ptr(get_buffer_ptr(buffer_table, data_->get_buffer_index()));
                buffer->data.push_back(tensor);
            }

            buffers.push_back(buffer);
        }

        return ENN_RET_SUCCESS;
    }

    virtual EnnReturn execute(UDOperators& operators, UDBuffers& buffers, const model::memory::BufferTable& buffer_table) {
#ifndef ENN_BUILD_RELEASE
        bool dump_available = is_dump_available();
#endif
        for (int i = 0; i < operators->size(); i++) {
            auto& op = operators->at(i);
            auto& buffer = buffers.at(i);
            ENN_DBG_PRINT("[%s]: op->getName() = %s\n", accelator.c_str(), op->getName().c_str());

            if (op->execute(buffer) != ENN_RET_SUCCESS) {
                ENN_DBG_PRINT("[%s]: %s execute() failed.\n", accelator.c_str(), op->getName().c_str());
                return ENN_RET_FAILED;
            }
#ifndef ENN_BUILD_RELEASE
            if (dump_available) {
                dump_operator_output(op, buffer, buffer_table);
            }
#endif
        }

        return ENN_RET_SUCCESS;
    }

#ifndef ENN_BUILD_RELEASE
    bool is_dump_available() {
        bool dump_available = false;

#ifdef __ANDROID__
        enn::debug::MaskType env_val = 0;
        if (!util::get_environment_property(debug::DbgPrintManager::GetInstance().get_debug_property_name().c_str(),
                                            &env_val)) {
            if (env_val & ZONE_BIT_MASK(debug::DbgPartition::kFileDumpSession)) {
                if (mkdir(DUMP_PATH, S_IRWXU | S_IRWXG | S_IROTH) == 0 || errno == EEXIST) {
                    dump_available = true;
                } else {
                    ENN_WARN_COUT << "Can not mkdir " << DUMP_PATH << std::endl;
                }
            }
        }
        ENN_DBG_PRINT("get property: 0x%" PRIx64 ", dump_available: %d\n", env_val, dump_available);
#elif __FORCE_DUMP__
        dump_available = true;
#endif

        return dump_available;
    }

    void dump_operator_output(std::shared_ptr<enn::ud::UDOperator>& op, std::shared_ptr<UDBuffer>& buffer,
                              const model::memory::BufferTable& buffer_table) {
        std::string dump_path = "";
#ifdef __ANDROID__
        dump_path = DUMP_PATH;
#endif
        for (auto tensor : op->getOutTensors()) {
            int index = tensor->get_buffer_index();
            if (index >= 0) {
                auto buf = buffer_table[index];
                std::string dump_file = dump_path + accelator + "_" + std::to_string(buffer->get_id()) + "_" +
                                        std::to_string(op->getId()) + "_" + op->getName() + ".dump";
                util::export_mem_to_file(dump_file.c_str(), const_cast<void*>(buf.get_addr()), buf.get_size());
                ENN_DBG_COUT << "[DUMP] " << dump_file.c_str() << std::endl;
            }
        }
    }
#endif

    DataPtr get_buffer_ptr(const model::memory::BufferTable& buffer_table, int32_t index) {
        if (index >= 0) {
            auto buffer_addr = buffer_table[index].get_addr();
            ENN_DBG_PRINT("[%s]: buffer_table[%d].get_addr() = %p\n", accelator.c_str(), index, buffer_addr);
            return const_cast<DataPtr>(buffer_addr);
        } else {
            ENN_WARN_PRINT("This buffer address will be ignored, not used.\n");
            return nullptr;
        }
    }

private:
    std::string accelator;
    std::shared_ptr<IComputeLibrary> compute_library;
};

}  // namespace ud
}  // namespace enn

#endif  // USERDRIVER_COMMON_IOPERATION_EXECUTOR_H_
