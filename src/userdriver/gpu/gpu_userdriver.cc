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

#include "common/enn_debug.h"
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/gpu/gpu_userdriver.h"
#include "tool/profiler/include/ExynosNnProfilerApi.h"
#include "common/identifier_chopper.hpp"

namespace enn {
namespace ud {
namespace gpu {

GpuUserDriver& GpuUserDriver::get_instance(void) {
    static GpuUserDriver gpu_userdriver_instance;
    return gpu_userdriver_instance;
}

GpuUserDriver::~GpuUserDriver(void) {
    ENN_DBG_PRINT("started\n");
    compute_library.reset();
    op_constructor.reset();
    op_executor.reset();

    ENN_DBG_PRINT("ended successfully\n");
}

EnnReturn GpuUserDriver::Initialize(void) {
    ENN_DBG_PRINT("started\n");

    std::lock_guard<std::mutex> lock_guard_operators_map(mutex_operators_map);
    ud_operators_map.clear();

    compute_library = std::make_shared<CLComputeLibrary>(0);

    op_constructor = std::unique_ptr<IOperationConstructor>(std::make_unique<OperationConstructor>(compute_library));

    op_executor = std::unique_ptr<IOperationExecutor>(std::make_unique<OperationExecutor>(compute_library));

    return ENN_RET_SUCCESS;
}

EnnReturn GpuUserDriver::OpenSubGraph(const model::component::OperatorList& operator_list) {
    ENN_DBG_PRINT("started\n");
    std::lock_guard<std::mutex> lock_guard_operators_map(mutex_constructor);
    compute_library->initialize_queue();

    uint64_t operator_list_id = operator_list.get_id().get();
    ENN_DBG_PRINT("OpenSubGraph operator_list_id = 0x%" PRIx64 "\n", operator_list_id);

    op_constructor->initialize_ud_operators();

    EnnReturn ret = op_constructor->open_oplist(operator_list);
    if (ret != ENN_RET_SUCCESS) {
        return ret;
    }

    return add_ud_operators(operator_list_id,
                            op_constructor->get_ud_operators(),
                            op_constructor->get_in_tensors(),
                            op_constructor->get_out_tensors());
}

EnnReturn GpuUserDriver::PrepareSubGraph(const enn::runtime::ExecutableOperatorList& executable_operator_list) {
    ENN_UNUSED(executable_operator_list);
    ENN_DBG_PRINT("started\n");
    return ENN_RET_SUCCESS;
}

EnnReturn GpuUserDriver::ExecuteSubGraph(const enn::runtime::OperatorListExecuteRequest& operator_list_execute_reqeust) {
    ENN_DBG_PRINT("started\n");
    std::lock_guard<std::mutex> lock_guard_operators_map(mutex_constructor);

    uint64_t operator_list_id = operator_list_execute_reqeust.get_operator_list_id().get();
    ENN_DBG_PRINT("operator_list_id = 0x%" PRIx64 "\n", operator_list_id);

    uint64_t executable_id = operator_list_execute_reqeust.get_executable_operator_list_id().get();
    ENN_DBG_PRINT("executable_id = 0x%" PRIx64 "\n", executable_id);

    const std::string profile_label = std::string("GPU_UD_Execution_#") + std::to_string(executable_id);
    if (profile_enable_)
        PROFILE_FROM(profile_label.c_str(), util::chop_into_model_id(operator_list_id));

    UDOperators operators;
    UDTensors in_tensors;
    UDTensors out_tensors;

    if (get_ud_operators(operator_list_id, operators, in_tensors, out_tensors) != ENN_RET_SUCCESS) {
        return ENN_RET_FAILED;
    }

    UDBuffers buffer;
    auto& buffer_table = operator_list_execute_reqeust.get_buffer_table();
    set_input_data(in_tensors, buffer_table);
    op_executor->execute(operators, buffer, buffer_table);

    compute_library->flush();
    set_output_data(out_tensors, buffer_table);

    if (profile_enable_)
        PROFILE_UNTIL(profile_label.c_str(), util::chop_into_model_id(operator_list_id));

    return ENN_RET_SUCCESS;
}

EnnReturn GpuUserDriver::CloseSubGraph(const model::component::OperatorList& operator_list) {
    ENN_DBG_PRINT("started\n");
    EnnReturn ret = ENN_RET_SUCCESS;
    std::lock_guard<std::mutex> lock_guard_operators_map(mutex_constructor);
    uint64_t operator_list_id = operator_list.get_id().get();
    ENN_DBG_PRINT("operator_list_id = 0x%" PRIx64 "\n", operator_list_id);

    if (remove_ud_operators(operator_list_id) != ENN_RET_SUCCESS) {
        ret = ENN_RET_FAILED;
    }

    op_constructor->close_oplist(operator_list_id);

    return ret;
}

EnnReturn GpuUserDriver::Deinitialize(void) {
    ENN_DBG_PRINT("started\n");
    EnnReturn ret = ENN_RET_SUCCESS;

    std::lock_guard<std::mutex> lock_guard_operators_map(mutex_operators_map);
    ud_operators_map.clear();
    if (!ud_operators_map.empty()) {
        ENN_ERR_PRINT("ud_operators_map was not cleared.\n");
        ret = ENN_RET_FAILED;
    }

    return ret;
}

EnnReturn GpuUserDriver::add_ud_operators(uint64_t id, UDOperators ud_operators, UDTensors in_tensors, UDTensors out_tensors) {
    ENN_DBG_PRINT("started, id = 0x%" PRIx64 "\n", id);
    EnnReturn ret = ENN_RET_SUCCESS;

    std::lock_guard<std::mutex> lock_guard(mutex_operators_map);
    if (ud_operators_map.find(id) != ud_operators_map.end()) {
        ENN_ERR_PRINT("ud_operators_map[%" PRIx64 "] was existed already.\n", id);
        ret = ENN_RET_FAILED;
    } else {
        auto ud_operator = std::make_tuple(ud_operators, in_tensors, out_tensors);
        ud_operators_map[id] = ud_operator;
    }

    return ret;
}

EnnReturn GpuUserDriver::get_ud_operators(uint64_t id, UDOperators& out_ud_operators, UDTensors &in_tensors, UDTensors &out_tensors) {
    ENN_DBG_PRINT("started, id = 0x%" PRIx64 "\n", id);
    EnnReturn ret = ENN_RET_SUCCESS;

    std::lock_guard<std::mutex> lock_guard(mutex_operators_map);
    if (ud_operators_map.find(id) == ud_operators_map.end()) {
        ENN_ERR_PRINT("ud_operators_map[%" PRIx64 "] was not found.\n", id);
        ret = ENN_RET_FAILED;
    } else {
        std::tie(out_ud_operators, in_tensors, out_tensors) = ud_operators_map[id];
    }

    return ret;
}

EnnReturn GpuUserDriver::remove_ud_operators(uint64_t id) {
    ENN_DBG_PRINT("started, id = 0x%" PRIx64 "\n", id);
    EnnReturn ret = ENN_RET_SUCCESS;

    std::lock_guard<std::mutex> lock_guard(mutex_operators_map);
    if (ud_operators_map.erase(id) != 1) {
        ENN_ERR_PRINT("remove ud_operators_map[%" PRIx64 "] failed.\n", id);
        ret = ENN_RET_FAILED;
    }

    return ret;
}

EnnReturn GpuUserDriver::set_input_data(UDTensors& in_tensors, const model::memory::BufferTable& buffer_table) {
    ENN_DBG_PRINT("started\n");
    // set input data
    DataOrderChangeType order_type = DataOrderChangeType::OTHER;
    StorageType storage_type_ = StorageType::BUFFER;
    // TODO(yc18.cho & xin.lu): set the real index when OPList contains the inputIndex
    for (auto in : in_tensors) {
        const uint32_t in_index = in->get_buffer_index();
        ENN_DBG_PRINT("set input %d\n", in_index);
        if (!buffer_table.exist(in_index))
            continue;
        if (storage_type_ == StorageType::TEXTURE) {
            //  The DataOrder of input is NCHW, but the data order of input is NHWC
            order_type = DataOrderChangeType::NHWC2DHWC4;
        } else {
            order_type =
                in->getDataOrder() == DataOrder::NHWC ? DataOrderChangeType::NHWC2NCHW : DataOrderChangeType::OTHER;
        }
        ENN_DBG_PRINT("in->get_buffer_index(): %d, addr %p, size: %zu, offset: %d\n",
                      in_index,
                      buffer_table[in_index].get_addr(),
                      buffer_table[in_index].get_size(),
                      buffer_table[in_index].get_offset());
#ifndef ENN_BUILD_RELEASE
        print_raw_input(in_index, in->getDataType(), in->getTotalSizeFromDims(), buffer_table);
#endif
        CHECK_AND_RETURN_ERR(
            Status::SUCCESS != in->writeData(const_cast<DataPtr>(buffer_table[in_index].get_addr()), false, order_type),
            ENN_RET_FAILED,
            "in->writeData() failed\n");
    }
    return ENN_RET_SUCCESS;
}

EnnReturn GpuUserDriver::set_output_data(UDTensors& out_tensors, const model::memory::BufferTable& buffer_table) {
    ENN_DBG_PRINT("started\n");
    // set output data
    DataOrderChangeType order_type = DataOrderChangeType::OTHER;
    StorageType storage_type_ = StorageType::BUFFER;
    // TODO(yc18.cho & xin.lu): set the real index when OPList contains the outputIndex
    for (auto& out : out_tensors) {
        const uint32_t out_index = out->get_buffer_index();
        ENN_DBG_PRINT("set output %d\n", out_index);
        if (!buffer_table.exist(out_index))
            continue;
        if (storage_type_ == StorageType::TEXTURE) {
            // The DataOrder of input is NCHW, but the data order of input is NHWC
            order_type = DataOrderChangeType::NHWC2DHWC4;
        } else {
            order_type = out->getDataOrder() == DataOrder::NHWC ? DataOrderChangeType::NHWC2NCHW
                                                                : DataOrderChangeType::OTHER;
        }

        ENN_DBG_PRINT("out->get_buffer_index(): %d, addr %p\n", out_index, buffer_table[out_index].get_addr());
        CHECK_AND_RETURN_ERR(Status::SUCCESS !=
                                 out->readData(const_cast<DataPtr>(buffer_table[out_index].get_addr()), true, order_type),
                             ENN_RET_FAILED,
                             "out->writeData() failed\n");
    }
    return ENN_RET_SUCCESS;
}

void GpuUserDriver::print_raw_input(const uint32_t& in_index,
                                    const DataType& data_type,
                                    const uint32_t& num,
                                    const model::memory::BufferTable& buffer_table) {
#if 0  // set 1 only when want to check raw data
    printf("print_raw_input, index: %d, data_type: %d, size: %d\n", in_index, data_type, buffer_table[in_index].get_size());
    std::string file_name = DUMP_PATH + std::to_string(in_index) + "_input" + +".txt";
    switch (data_type) {
    case DataType::FLOAT: print_data((float*)buffer_table[in_index].get_addr(), num, file_name); break;
    case DataType::HALF: print_data((half_float::half*)buffer_table[in_index].get_addr(), num, file_name); break;
    case DataType::INT32: print_data((int*)buffer_table[in_index].get_addr(), num, file_name); break;
    case DataType::INT8: print_data((int8_t*)buffer_table[in_index].get_addr(), num, file_name); break;
    case DataType::UINT8: print_data((uint8_t*)buffer_table[in_index].get_addr(), num, file_name); break;
    case DataType::BOOL: print_data((bool*)buffer_table[in_index].get_addr(), num, file_name); break;
    case DataType::INT16: print_data((int16_t*)buffer_table[in_index].get_addr(), num, file_name); break;
    case DataType::UINT16: print_data((uint16_t*)buffer_table[in_index].get_addr(), num, file_name); break;
    default: break;
    }
#else
    ENN_UNUSED(in_index), ENN_UNUSED(data_type), ENN_UNUSED(num), ENN_UNUSED(buffer_table);
#endif
}

template <typename T> void GpuUserDriver::print_data(T* data, const uint32_t& num, const std::string& file_name) {
#ifndef ENN_BUILD_RELEASE
    std::ofstream out_file;
    bool save = false;  // set true when want save data as file
    if (op_executor->is_dump_available() && save) {
        std::cout << "saving " << file_name.c_str() << std::endl;
        out_file.open(file_name);
    } else {
        out_file.basic_ios<char>::rdbuf(std::cout.rdbuf());
    }

    for (int i = 0; i < num; i++) {
        if (typeid(T) == typeid(float) || typeid(T) == typeid(half_float::half)) {
            out_file << i << ": " << (float)(data[i]) << std::endl;
        } else {
            out_file << i << ": " << (int)(data[i]) << std::endl;
        }
    }

    if (op_executor->is_dump_available() && save) {
        out_file.close();
    }
#endif
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
