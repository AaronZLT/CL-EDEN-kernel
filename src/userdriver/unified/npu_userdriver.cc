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

#include <sys/mman.h>               // mmap(), munmap()
#include "common/enn_debug.h"
#include "common/compiler.h"
#include "userdriver/unified/npu_userdriver.h"
#include "userdriver/unified/utils.h"
#include "model/component/operator/operator.hpp"
#include "model/component/tensor/tensor.hpp"
#include "model/component/tensor/feature_map.hpp"
#include "tool/profiler/include/ExynosNnProfilerApi.h"

namespace enn {
namespace ud {
namespace npu {

static bool is_valid_ion_buffer(int fd, size_t size, const void* addr) {
    int ret;
    void *addr_tmp;

    ENN_DBG_PRINT("input parameters, fd:%d, size:%d, addr:%p\n", fd, (int) size, addr);
    if (fd < 0 || size <= 0 || addr == nullptr) {
        ENN_ERR_PRINT_FORCE("invalid parameters, fd:%d, size:%d, addr:%p\n", fd, (int) size, addr);
        return false;
    }

    addr_tmp = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr_tmp == nullptr || addr_tmp == MAP_FAILED) {
        ENN_ERR_PRINT_FORCE("invalid fd, fail to mmap(), addr:%p, addr_tmp:%p\n", addr, addr_tmp);
        return false;
    }

    ENN_DBG_PRINT("addr:%p, addr_tmp:%p\n", addr, addr_tmp);
    ret = memcmp(addr_tmp, addr, 1);
    if (ret) {
        ENN_ERR_PRINT_FORCE("invalid fd, fail to memcmp() ret:%d\n", ret);
        if (munmap(addr_tmp, size)) {
            ENN_ERR_PRINT_FORCE("fail to munmap() ret:%d\n", ret);
            return false;
        }
        return ENN_RET_FAILED;
    }

    if (munmap(addr_tmp, size)) {
        ENN_ERR_PRINT_FORCE("fail to munmap() ret:%d\n", ret);
        return false;
    }
    return true;
}

EnnReturn NpuUDOperator::init(uint32_t in_buf_cnt, uint32_t out_buf_cnt) {
    if ((in_buf_cnt <= 0) || (out_buf_cnt <= 0)) {
        ENN_ERR_PRINT_FORCE("invalid parameter, in_buf_cnt:%d, out_buf_cnt:%d\n", in_buf_cnt, out_buf_cnt);
        return ENN_RET_FAILED;
    }

    op_info.bin_in_shape = new shape_t[in_buf_cnt];
    op_info.bin_in_index = new uint32_t[in_buf_cnt];
    op_info.bin_in_bpp = new uint32_t[in_buf_cnt];
    op_info.bin_out_shape = new shape_t[out_buf_cnt];
    op_info.bin_out_index = new uint32_t[out_buf_cnt];

    if ((op_info.bin_in_shape == nullptr) || (op_info.bin_in_index == nullptr) || (op_info.bin_in_bpp == nullptr)
            || (op_info.bin_out_shape == nullptr) || (op_info.bin_out_index == nullptr)) {
        ENN_ERR_PRINT_FORCE("fail to malloc(), bin_in_shape:%p, bin_in_index:%p, bin_in_bpp:%p\n",
                op_info.bin_in_shape, op_info.bin_in_index, op_info.bin_in_bpp);
        ENN_ERR_PRINT_FORCE("fail to malloc(), bin_out_shape:%p, bin_out_index:%p\n",
                op_info.bin_out_shape, op_info.bin_out_index);

        return ENN_RET_MEM_ERR;
    }

    return ENN_RET_SUCCESS;
}

EnnReturn NpuUDOperator::set(uint32_t in_buf_cnt, uint32_t out_buf_cnt,
        model::component::Operator::Ptr rt_opr_npu,
        uint64_t operator_list_id, uint64_t unified_op_id) {
    int i;

    if (in_buf_cnt == 0 || out_buf_cnt == 0) {
        ENN_ERR_PRINT_FORCE("Invalid the number of buffers, in_buf_cnt:%u, out_buf_cnt:%u\n",
                in_buf_cnt, out_buf_cnt);
        return ENN_RET_FAILED;
    }

    // TODO(jungho7.kim, TBD): get cell_align_shape from RT layer
    shape_t cell_align_shape;
    cell_align_shape.number = 3;
    cell_align_shape.channel = 1;
    cell_align_shape.height = 1;
    cell_align_shape.width = 1;
    cell_align_shape.type_size = 3;

    if ((op_info.bin_in_shape == nullptr) || (op_info.bin_in_index == nullptr) || (op_info.bin_in_bpp == nullptr)
            || (op_info.bin_out_shape == nullptr) || (op_info.bin_out_index == nullptr)) {
        ENN_ERR_PRINT_FORCE("NULL pointer error, bin_in_shape:%p, bin_in_index:%p, bin_in_bpp:%p\n",
                op_info.bin_in_shape, op_info.bin_in_index, op_info.bin_in_bpp);
        ENN_ERR_PRINT_FORCE("NULL pointer error, bin_out_shape:%p, bin_out_index:%p\n",
                op_info.bin_out_shape, op_info.bin_out_index);

        return ENN_RET_MEM_ERR;
    }

    op_info.model_addr = (void*) rt_opr_npu->get_binaries().at(util::FIRST).get_addr();
    op_info.model_size = rt_opr_npu->get_binaries().at(util::FIRST).get_size();
    op_info.model_name = (uint8_t*) rt_opr_npu->get_binaries().at(util::FIRST).get_name().c_str();
    op_info.id = generate_op_id(operator_list_id, rt_opr_npu->get_id());
    ENN_DBG_PRINT("op_list_id:%lu, op_uid:%lu, op_id:%lx\n", (unsigned long) operator_list_id,
            (unsigned long) rt_opr_npu->get_id(), (unsigned long) op_info.id);
    if (op_info.id == 0) {
        ENN_ERR_PRINT_FORCE("Invalid op ID:%lx", (unsigned long) op_info.id);
        return ENN_RET_FAILED;
    }

    // TODO(jungho7.kim, TBD): modify uint8_t to uint32_t
    op_info.input_count = (uint8_t) in_buf_cnt;
    op_info.operator_list_id = operator_list_id;
    if (unified_op_id > 0)
        op_info.unified_op_id = unified_op_id;
    else
        op_info.unified_op_id = op_info.id;

    // TODO(jungho7.kim, TBD): get the cell_align_shape from RT layer.
    op_info.cell_align_shape = cell_align_shape;
    // TODO(jungho7.kim, TBD): remove shared_buffer.
    op_info.shared_buffer = -1;
    op_info.binding_ofm = rt_opr_npu->is_ofm_bound() ? 1 : 0;
    // TODO(jungho7.kim, 6/30): get tile_size from RT layer.
    op_info.tile_size = 1;

    i = 0;
    for (auto& tensor : rt_opr_npu->in_tensors) {
        if (!tensor->is_const()) {  // if tensor is not const, the tensor is feature map.
            model::component::FeatureMap::Ptr ifm =
                std::static_pointer_cast<model::component::FeatureMap>(tensor);
            // TODO(jungho7.kim, TBD): handle the case of NHWC
            op_info.bin_in_shape[i].number = ifm->get_shape()[0];
            op_info.bin_in_shape[i].channel = ifm->get_shape()[1];
            op_info.bin_in_shape[i].height = ifm->get_shape()[2];
            op_info.bin_in_shape[i].width = ifm->get_shape()[3];
            op_info.bin_in_shape[i].type_size = model::pixel_bit_format_size[
                static_cast<TFlite::TensorType>(ifm->get_data_type())];
            op_info.bin_in_index[i] = (uint32_t) ifm->get_buffer_index();
            op_info.bin_in_bpp[i] = 8 * model::pixel_bit_format_size[
                static_cast<TFlite::TensorType>(ifm->get_data_type())];
            i++;
        }
    }

    int idx_n, idx_c, idx_h, idx_w;
    // TODO(jungho7.kim, TBD): handle the case of NHWC
    idx_n = 0;
    idx_c = 1;
    idx_h = 2;
    idx_w = 3;

    if (!rt_opr_npu->is_ofm_bound()) {
        op_info.output_count = (uint8_t) out_buf_cnt;
        i = 0;
        for (auto& tensor : rt_opr_npu->out_tensors) {
            if (!tensor->is_const()) {  // if tensor is not const, the tensor is feature map.
                model::component::FeatureMap::Ptr ofm =
                    std::static_pointer_cast<model::component::FeatureMap>(tensor);
                op_info.bin_out_shape[i].number = ofm->get_shape()[idx_n];
                op_info.bin_out_shape[i].channel = ofm->get_shape()[idx_c];
                op_info.bin_out_shape[i].height = ofm->get_shape()[idx_h];
                op_info.bin_out_shape[i].width = ofm->get_shape()[idx_w];
                op_info.bin_out_shape[i].type_size = model::pixel_bit_format_size[
                    static_cast<TFlite::TensorType>(ofm->get_data_type())];
                op_info.bin_out_index[i] = (uint32_t) ofm->get_buffer_index();
                i++;
            }
        }
    } else {
        uint32_t total_size = 0;
        uint32_t ofm_buf_size = 0;

        op_info.output_count = 1;
        i = 0;
        for (auto& tensor : rt_opr_npu->out_tensors) {
            if (!tensor->is_const()) {  // if tensor is not const, the tensor is feature map.
                model::component::FeatureMap::Ptr ofm =
                    std::static_pointer_cast<model::component::FeatureMap>(tensor);
                ofm_buf_size = ofm->get_shape()[idx_n] * ofm->get_shape()[idx_c] *\
                               ofm->get_shape()[idx_h] * ofm->get_shape()[idx_w] *\
                               model::pixel_bit_format_size[
                               static_cast<TFlite::TensorType>(ofm->get_data_type())];
                total_size += ofm_buf_size;
                op_info.bin_out_index[i] = (uint32_t) ofm->get_buffer_index();
                i++;
            }
        }
        op_info.bin_out_shape[0].number = 1;
        op_info.bin_out_shape[0].channel = 1;
        op_info.bin_out_shape[0].height = 1;
        op_info.bin_out_shape[0].width = total_size;
        op_info.bin_out_shape[0].type_size = 1;
    }

    return ENN_RET_SUCCESS;
}

EnnReturn NpuUDOperator::deinit(void) {
    if (op_info.bin_in_shape)
        free(op_info.bin_in_shape);
    if (op_info.bin_in_index)
        free(op_info.bin_in_index);
    if (op_info.bin_in_bpp)
        free(op_info.bin_in_bpp);
    if (op_info.bin_out_shape)
        free(op_info.bin_out_shape);
    if (op_info.bin_out_index)
        free(op_info.bin_out_index);

    return ENN_RET_SUCCESS;
}

EnnReturn NpuUDOperator::add_executable_op(uint64_t exec_op_id,
        const std::shared_ptr<ExecutableNpuUDOperator>& executable_op) {

    ENN_INFO_PRINT("exec_op_id : %lu executable_op:%p\n", (unsigned long) exec_op_id, executable_op.get());
    std::lock_guard<std::mutex> lock_guard(mutex_executable_op_map);
    if (executable_op_map.find(exec_op_id) != executable_op_map.end()) {
        ENN_WARN_PRINT("executable_op_map[%lu] was existed already.\n", (unsigned long) exec_op_id);
        return ENN_RET_FAILED;
    }
    executable_op_map[exec_op_id] = executable_op;
    return ENN_RET_SUCCESS;
}

std::vector<uint64_t> NpuUDOperator::get_all_executable_op_id() {
    std::vector<uint64_t> exec_op_ids;
    for (auto const& element : executable_op_map) {
        exec_op_ids.push_back(element.first);
    }
    return exec_op_ids;
}

std::shared_ptr<ExecutableNpuUDOperator> NpuUDOperator::get_executable_op(uint64_t exec_op_id) {
    ENN_INFO_PRINT("exec_op_id : %lu\n", (unsigned long) exec_op_id);
    std::lock_guard<std::mutex> lock_guard(mutex_executable_op_map);
    if (executable_op_map.find(exec_op_id) == executable_op_map.end()) {
        ENN_WARN_PRINT("executable_op_map[%lu] was not found.\n", (unsigned long) exec_op_id);
        return nullptr;
    }
    return executable_op_map[exec_op_id];
}

EnnReturn NpuUDOperator::remove_executable_op(uint64_t exec_op_id) {
    std::lock_guard<std::mutex> lock_guard(mutex_executable_op_map);
    if (executable_op_map.erase(exec_op_id) != 1) {
        ENN_ERR_PRINT("remove executable__map[%lu] failed.\n", (unsigned long) exec_op_id);
        return ENN_RET_FAILED;
    }
    return ENN_RET_SUCCESS;
}

uint64_t NpuUDOperator::get_id(void) {
    return op_info.id;
}

void NpuUDOperator::dump(void) {
    ENN_DBG_PRINT("model_addr(%p), model_size(%d), model_name(%s), id(%lu)",
            op_info.model_addr, op_info.model_size, op_info.model_name, (unsigned long) op_info.id);
    ENN_DBG_PRINT("in_buf_cnt(%d), bin_in_shape(%d,%d,%d,%d,%d)",
            (int32_t) op_info.input_count, op_info.bin_in_shape->number,
            op_info.bin_in_shape->channel, op_info.bin_in_shape->height,
            op_info.bin_in_shape->width, op_info.bin_in_shape->type_size);
    ENN_DBG_PRINT("out_buf_cnt(%d), bin_out_shape(%d,%d,%d,%d,%d)",
            (int32_t) op_info.output_count, op_info.bin_out_shape->number,
            op_info.bin_out_shape->channel, op_info.bin_out_shape->height,
            op_info.bin_out_shape->width, op_info.bin_out_shape->type_size);
    ENN_DBG_PRINT("cell_align_shape(%d,%d,%d,%d,%d)",
            op_info.cell_align_shape.number, op_info.cell_align_shape.channel,
            op_info.cell_align_shape.height, op_info.cell_align_shape.width,
            op_info.cell_align_shape.type_size);
    ENN_DBG_PRINT("operator_list_id(0x%" PRIX64 ") priority(%d) in_pixel_format(%d)\n",
            op_info.operator_list_id, op_info.priority, op_info.bin_in_bpp[0]);
    ENN_DBG_PRINT("shared_buffer(%d) binding_ofm(%u) tile_size(%d) reserved[0](%u)\n",
            op_info.shared_buffer, op_info.binding_ofm, op_info.tile_size, op_info.reserved[1]);
}

NpuUserDriver::~NpuUserDriver(void) {
    ENN_DBG_PRINT("started\n");

    ENN_DBG_PRINT("ended successfully\n");
}

NpuUserDriver& NpuUserDriver::get_instance(void) {
    static NpuUserDriver npu_userdriver_instance;
    return npu_userdriver_instance;
}

EnnReturn NpuUserDriver::set_npu_ud_status(NpuUdStatus npu_ud_status) {
    std::lock_guard<std::mutex> lock_guard(mutex_npu_ud_status);
    npu_ud_status_ = npu_ud_status;
    return ENN_RET_SUCCESS;
}

EnnReturn NpuUserDriver::Initialize(void) {
    ENN_DBG_PRINT("NPU UD Initialize() start\n");

    // TODO(jungho7.kim, TBD): refactor max_request_size
    uint32_t max_request_size = 16;
    // TODO(jungho7.kim, TBD): replace eden_ret to EnnReturn
    EnnReturn ret;

    // Check if NPU UD is already initialized
    if (get_npu_ud_status() != NpuUdStatus::INITIALIZED) {
        ret = UdLink::get_instance().link_init(acc_, max_request_size);
        if (ret != ENN_RET_SUCCESS) {
            ENN_ERR_PRINT_FORCE("(-) init_acc error: %x\n", ret);
            return ENN_RET_FAILED;
        }
        std::lock_guard<std::mutex> lock_guard(mutex_ud_operator_list_map);
        ud_operator_list_map.clear();
        set_npu_ud_status(NpuUdStatus::INITIALIZED);
    } else {
        ENN_DBG_PRINT("(-) NPU is already INITIALIZED\n");
    }
    ENN_DBG_PRINT("NPU UD Initialize() end\n");
    return ENN_RET_SUCCESS;
}

EnnReturn NpuUserDriver::OpenSubGraph(const model::component::OperatorList& operator_list) {
    uint64_t operator_list_id = operator_list.get_id().get();
    ENN_DBG_PRINT("operator_list_id:%lu\n", (unsigned long) operator_list_id);

    return OpenSubGraph(operator_list, operator_list_id, 0);
}

EnnReturn NpuUserDriver::OpenSubGraph(const model::component::OperatorList& operator_list,
        uint64_t operator_list_id, uint64_t unified_op_id) {
    ENN_DBG_PRINT("NPU UD OpenSubGraph() start\n");

    uint32_t in_buf_cnt;
    uint32_t out_buf_cnt;

    // Nice to have: TODO(mj.kim010, TBD): Use preference of OperatorList
    if (get_npu_ud_status() != NpuUdStatus::INITIALIZED) {
        ENN_ERR_PRINT_FORCE("NPU UD is not initialized\n");
        return ENN_RET_FAILED;
    }

    NpuUDOperators ud_operators;
    for (auto& rt_opr : operator_list) {
        auto rt_opr_npu = std::static_pointer_cast<model::component::Operator>(rt_opr);

        std::shared_ptr<NpuUDOperator> ud_op = std::make_shared<NpuUDOperator>();

        in_buf_cnt = rt_opr->in_tensors.count();
        out_buf_cnt = rt_opr->out_tensors.count();

        EnnReturn ret_val;

        ENN_DBG_PRINT("in_buf_cnt:%u, out_buf_cnt:%u\n", in_buf_cnt, out_buf_cnt);
        ret_val = ud_op->init(in_buf_cnt, out_buf_cnt);
        if (ret_val) {
            ENN_ERR_PRINT_FORCE("fail to init(), ret:%d\n", ret_val);
            ud_op->deinit();
            return ret_val;
        }

        ret_val = ud_op->set(in_buf_cnt, out_buf_cnt, rt_opr_npu, operator_list_id, unified_op_id);
        if (ret_val) {
            ENN_ERR_PRINT("fail to set(), ret:%d\n", ret_val);
            ud_op->deinit();
            return ret_val;
        }

        ud_operators.push_back(ud_op);

        // TODO(jungho7.kim, TBD): modify EdenModelOptions to UdSubGraphPreference
        EdenModelOptions options;
        uint32_t perf_mode = operator_list.get_pref_mode();
        options.modelPreference.userPreference.hw = NPU_ONLY;
        // TODO(jungho7.kim): Create a table map and use as below
        // options.modelPreference.userPreference.mode = TABLE[perf_mode];
        if (perf_mode == ENN_PREF_MODE_NORMAL) {
            options.modelPreference.userPreference.mode = NORMAL_MODE;
        } else if (perf_mode == ENN_PREF_MODE_BOOST) {
            options.modelPreference.userPreference.mode = BOOST_MODE;
        } else if (perf_mode == ENN_PREF_MODE_BOOST_ON_EXE) {
            options.modelPreference.userPreference.mode = BOOST_ON_EXECUTE_MODE;
        } else if (perf_mode == ENN_PREF_MODE_BOOST_BLOCKING) {
            options.modelPreference.userPreference.mode = BOOST_BLOCKING_MODE;
        } else {
            ENN_WARN_PRINT_FORCE("Invalid perf_mode value:%u\n", operator_list.get_pref_mode());
            options.modelPreference.userPreference.mode = BOOST_ON_EXECUTE_MODE;
        }
        options.modelPreference.nnApiType = EDEN_NN_API;
        options.priority = operator_list.get_priority();
        options.latency = operator_list.get_target_latency();
        options.boundCore = operator_list.get_core_affinity();
        options.tileSize = operator_list.get_tile_num();
        options.presetScenarioId = operator_list.get_preset_id();

        ud_op->dump();
        ENN_DBG_PRINT("hw:%u mode:%u priority:%u latency:%u boundCore:%u tileSize:%u presetScenarioId:%d\n",
                (uint32_t) options.modelPreference.userPreference.hw,
                (uint32_t) options.modelPreference.userPreference.mode,
                (uint32_t) options.priority, (uint32_t) options.latency, (uint32_t) options.boundCore,
                (uint32_t) options.tileSize, options.presetScenarioId);

        // TODO(jungho7.kim, TBD): Deprecate second parameter of open_model()
        EnnReturn ret = UdLink::get_instance().link_open_model(acc_, &ud_op->get(), &options);

        if (ret != ENN_RET_SUCCESS) {
            ENN_ERR_PRINT_FORCE("(-) open_model error: %d\n", ret);
            ud_op->deinit();
            return ENN_RET_FAILED;
        }
    }

    ENN_INFO_PRINT("set_id --> id : %lu\n", (unsigned long) operator_list_id);

    add_graph(operator_list_id, ud_operators);
    ENN_DBG_PRINT("NPU UD OpenSubGraph() end\n");
    return ENN_RET_SUCCESS;
}

EnnReturn ExecutableNpuUDOperator::init(uint32_t in_buf_cnt, uint32_t out_buf_cnt) {
    executable_op_info.inputs = std::shared_ptr<eden_memory_t>(
                                    new eden_memory_t[in_buf_cnt],
                                    std::default_delete<eden_memory_t[]>());
    executable_op_info.outputs = std::shared_ptr<eden_memory_t>(
                                    new eden_memory_t[out_buf_cnt],
                                    std::default_delete<eden_memory_t[]>());
    return ENN_RET_SUCCESS;
}

EnnReturn ExecutableNpuUDOperator::deinit(void) {
    return ENN_RET_SUCCESS;
}

EnnReturn ExecutableNpuUDOperator::set(model_info_t* op_info, const model::memory::BufferTable& buffer_table) {
    uint32_t in_buf_cnt, out_buf_cnt;

    if (op_info == NULL) {
        ENN_ERR_PRINT_FORCE("NULL pointer op_info\n");
        return ENN_RET_FAILED;
    }

    if (op_info->input_count == 0 || op_info->output_count == 0
            || executable_op_info.inputs.get() == nullptr || executable_op_info.outputs.get() == nullptr) {
        ENN_ERR_PRINT_FORCE("Invalid arguments, in_buf_cnt:%u out_buf_cnt:%u inputs:%p outputs:%p\n",
                op_info->input_count, op_info->output_count,
                executable_op_info.inputs.get(), executable_op_info.outputs.get());
        return ENN_RET_FAILED;
    }

    in_buf_cnt = op_info->input_count;
    out_buf_cnt = op_info->output_count;

    executable_op_info.model_info = op_info;
    // TODO(jungho7.kim, TBD): remove requestId because this is not used
    executable_op_info.requestId = 1;
    executable_op_info.operator_list_id = op_info->operator_list_id;

    for (int i = 0; i < in_buf_cnt; i++) {
        auto& buffer = buffer_table[op_info->bin_in_index[i]];
        eden_memory_t &em = executable_op_info.inputs.get()[i];
        // TODO(jungho7.kim, TBD): get this type from RT layer
        em.type = ION;
        em.size = buffer.get_size();
        em.ref.ion.fd = buffer.get_fd();
        em.ref.ion.buf = (uint64_t) buffer.get_addr();
        em.alloc_size = buffer.get_size();
        if (!is_valid_ion_buffer(buffer.get_fd(),
                    buffer.get_size(), buffer.get_addr())) {
            ENN_ERR_PRINT_FORCE("invalid input ION buffer\n");
            return ENN_RET_FAILED;
        }
    }

    for (int i = 0; i < out_buf_cnt; i++) {
        auto& buffer = buffer_table[op_info->bin_out_index[i]];
        eden_memory_t &em = executable_op_info.outputs.get()[i];
        // TODO(jungho7.kim, TBD): get type from RT layer
        em.type = ION;
        em.size = buffer.get_size();
        em.ref.ion.fd = buffer.get_fd();
        em.ref.ion.buf = (uint64_t) buffer.get_addr();
        em.alloc_size = buffer.get_size();
        if (!is_valid_ion_buffer(buffer.get_fd(),
                    buffer.get_size(), buffer.get_addr())) {
            ENN_ERR_PRINT_FORCE("invalid output ION buffer\n");
            return ENN_RET_FAILED;
        }
    }

    return ENN_RET_SUCCESS;
}

void ExecutableNpuUDOperator::dump(void) {
    for (int i = 0; i < executable_op_info.model_info->input_count; i++) {
        auto &mem = executable_op_info.inputs.get()[i];
        ENN_DBG_PRINT("npu_ud inputs[%d] type:%u size:%u alloc_size:%u ion_fd:%d ion_buf:%lu\n",
                i, (uint32_t) mem.type, (uint32_t) mem.size, (uint32_t) mem.alloc_size,
                (uint32_t) mem.ref.ion.fd, (unsigned long) mem.ref.ion.buf);
    }
    for (int i = 0; i < executable_op_info.model_info->output_count; i++) {
        auto &mem = executable_op_info.outputs.get()[i];
        ENN_DBG_PRINT("npu_ud outputs[%d] type:%u size:%u alloc_size:%u ion_fd:%d ion_buf:%lu\n",
                i, (uint32_t) mem.type, (uint32_t) mem.size, (uint32_t) mem.alloc_size,
                (uint32_t) mem.ref.ion.fd, (unsigned long) mem.ref.ion.buf);
    }
    ENN_DBG_PRINT("npu_ud model_info:%p requestId:%u operator_list_id:0x%" PRIX64 "\n",
            executable_op_info.model_info, (uint32_t) executable_op_info.requestId,
            executable_op_info.operator_list_id);
}

EnnReturn NpuUserDriver::PrepareSubGraph(const enn::runtime::ExecutableOperatorList& executable_operator_list) {
    ENN_DBG_PRINT("NPU UD PrepareSubGraph() start\n");

    NpuUDOperators operators;
    uint64_t operator_list_id = executable_operator_list.get_operator_list_id().get();

    if (get_npu_ud_status() != NpuUdStatus::INITIALIZED) {
        ENN_ERR_PRINT_FORCE("NPU UD is not initialized\n");
        return ENN_RET_FAILED;
    }

    if (get_graph(operator_list_id, operators) != ENN_RET_SUCCESS) {
        return ENN_RET_FAILED;
    }

    for (auto& op : operators) {
        model_info_t* op_info;
        std::shared_ptr<ExecutableNpuUDOperator> executable_op;
        uint64_t exec_op_id = executable_operator_list.get_id().get();

        op_info = &op->get();
        if (op_info == NULL) {
            ENN_ERR_PRINT("NULL pointer op_info\n");
            return ENN_RET_FAILED;
        }

        executable_op = std::make_shared<ExecutableNpuUDOperator>();

        if (executable_op->init(op_info->input_count, op_info->output_count) != ENN_RET_SUCCESS) {
            ENN_ERR_PRINT("fail to init() of executable_op\n");
            return ENN_RET_FAILED;
        }

        if (executable_op->set(op_info, executable_operator_list.get_buffer_table()) != ENN_RET_SUCCESS) {
            ENN_ERR_PRINT("fail to set() of executable_op\n");
            return ENN_RET_FAILED;
        }

        if (op->add_executable_op(exec_op_id, executable_op) != ENN_RET_SUCCESS) {
            ENN_ERR_PRINT_FORCE("fail to add executable_op\n");
            return ENN_RET_FAILED;
        }

        EnnReturn ret = UdLink::get_instance().link_prepare_req(acc_, op_info, executable_op->get_inputs(), executable_op->get_outputs(), nullptr);

        if (ret != ENN_RET_SUCCESS) {
            ENN_ERR_PRINT_FORCE("(-) failed Err[%d] prepare_req() \n", ret);
            return ENN_RET_FAILED;
        }
    }
    ENN_DBG_PRINT("NPU UD PrepareSubGraph() end\n");
    return ENN_RET_SUCCESS;
}

EnnReturn NpuUserDriver::ExecuteSubGraph(const enn::runtime::OperatorListExecuteRequest& operator_list_execute_request) {
    ENN_DBG_PRINT("NPU UD ExecuteSubGraph() start\n");

    if (unlikely(get_npu_ud_status() != NpuUdStatus::INITIALIZED)) {
        ENN_ERR_PRINT_FORCE("NPU UD is not initialized\n");
        return ENN_RET_FAILED;
    }
    uint64_t operator_list_id = operator_list_execute_request.get_operator_list_id().get();
    PROFILE_SCOPE("NPU_UD_Execution", util::chop_into_model_id(operator_list_id));

    NpuUDOperators operators;

    if (unlikely(get_graph(operator_list_id, operators) != ENN_RET_SUCCESS)) {
        ENN_ERR_PRINT_FORCE("fail to get_graph()\n");
        return ENN_RET_FAILED;
    }

    for (auto& op : operators) {
        model_info_t* op_info;
        std::shared_ptr<ExecutableNpuUDOperator> executable_op;
        uint64_t exec_op_id = operator_list_execute_request.get_executable_operator_list_id().get();

        op_info = &op->get();
        if (unlikely(op_info == NULL)) {
            ENN_ERR_PRINT_FORCE("NULL pointer error, op_info\n");
            return ENN_RET_FAILED;
        }

        executable_op = op->get_executable_op(exec_op_id);
        if (unlikely(executable_op == NULL)) {  // if the buffer is not pre-allocated by prepare().
            executable_op = std::make_shared<ExecutableNpuUDOperator>();

            if (unlikely(executable_op->init(op_info->input_count, op_info->output_count) != ENN_RET_SUCCESS)) {
                ENN_ERR_PRINT("fail to init() of executable_op\n");
                return ENN_RET_FAILED;
            }

            if (unlikely(executable_op->set(op_info, operator_list_execute_request.get_buffer_table()) != ENN_RET_SUCCESS)) {
                ENN_ERR_PRINT("fail to set() of executable_op\n");
                return ENN_RET_FAILED;
            }

            if (unlikely(op->add_executable_op(exec_op_id, executable_op) != ENN_RET_SUCCESS)) {
                ENN_ERR_PRINT_FORCE("fail to add executable_op\n");
                return ENN_RET_FAILED;
            }
        }

        // TODO(jungho7.kim, TBD): remove EdenRequestOptions because it will be deprecated
        EdenRequestOptions options;
        options.userPreference.hw = NPU_ONLY;
        options.userPreference.mode = BOOST_MODE;
        options.requestMode = BLOCK;

        executable_op->dump();
        EnnReturn ret = UdLink::get_instance().link_execute_req(acc_, &executable_op->get(), &options);

        if (unlikely(ret != ENN_RET_SUCCESS)) {
            ENN_ERR_PRINT_FORCE("(-) failed Err[%d] execute_req() \n", ret);
            return ENN_RET_FAILED;
        }
    }
    ENN_DBG_PRINT("NPU UD ExecuteSubGraph() end\n");

    return ENN_RET_SUCCESS;
}

EnnReturn NpuUserDriver::CloseSubGraph(const model::component::OperatorList& operator_list) {
    uint64_t operator_list_id = operator_list.get_id().get();
    ENN_DBG_PRINT("operator_list_id:%lu\n", (unsigned long) operator_list_id);

    return CloseSubGraph(operator_list, operator_list_id);
}

EnnReturn NpuUserDriver::CloseSubGraph(const model::component::OperatorList& operator_list, uint64_t operator_list_id) {
    ENN_UNUSED(operator_list);
    ENN_DBG_PRINT("NPU UD CloseSubGraph() start\n");

    if (get_npu_ud_status() != NpuUdStatus::INITIALIZED) {
        ENN_ERR_PRINT_FORCE("NPU UD is not initialized\n");
        return ENN_RET_FAILED;
    }

    NpuUDOperators operators;

    if (get_graph(operator_list_id, operators) != ENN_RET_SUCCESS) {
        return ENN_RET_FAILED;
    }

    for (auto& op : operators) {
        std::shared_ptr<ExecutableNpuUDOperator> executable_op;
        const std::vector<uint64_t> &exec_op_ids = op->get_all_executable_op_id();
        for (auto &exec_op_id : exec_op_ids) {
            executable_op = op->get_executable_op(exec_op_id);
            if (executable_op != NULL) {
                executable_op->deinit();
                op->remove_executable_op(exec_op_id);
            }
        }
    }

    // TODO(jungho7.kim, TBD): rename? close_operator_list()?
    if (UdLink::get_instance().link_close_model(acc_, operator_list_id) != ENN_RET_SUCCESS) {  // legacy func()
        ENN_ERR_PRINT_FORCE("(-) fail close_model()\n");
        return ENN_RET_FAILED;
    }

    if (remove_graph(operator_list_id) != ENN_RET_SUCCESS) {
        ENN_ERR_PRINT_FORCE("(-) fail remove_graph()\n");
        return ENN_RET_FAILED;
    }
    ENN_DBG_PRINT("NPU UD CloseSubGraph() end\n");
    return ENN_RET_SUCCESS;
}

EnnReturn NpuUserDriver::Deinitialize(void) {
    ENN_DBG_PRINT("NPU UD Deinitialize() start\n");

    if (get_npu_ud_status() == NpuUdStatus::SHUTDOWNED) {
        ENN_DBG_PRINT("(-) NPU is already SHUTDOWNED\n");
        return ENN_RET_SUCCESS;
    }

    EnnReturn ret = UdLink::get_instance().link_shutdown(acc_);
    if (ret != ENN_RET_SUCCESS) {
        ENN_ERR_PRINT_FORCE("(-) failed Err[%d] shutdown()\n", ret);
        return ENN_RET_FAILED;
    }

    set_npu_ud_status(NpuUdStatus::SHUTDOWNED);

    // ToDo: clear open_model_map
    std::lock_guard<std::mutex> lock_guard(mutex_ud_operator_list_map);
    ud_operator_list_map.clear();

    if (!ud_operator_list_map.empty()) {
        return ENN_RET_FAILED;
    }
    ENN_DBG_PRINT("NPU UD Deinitialize() end\n");

    return ENN_RET_SUCCESS;
}

EnnReturn NpuUserDriver::add_graph(uint64_t id, NpuUDOperators ud_operators) {
    std::lock_guard<std::mutex> lock_guard(mutex_ud_operator_list_map);
    if (ud_operator_list_map.find(id) != ud_operator_list_map.end()) {
        ENN_ERR_PRINT("ud_operator_list_map[%lu] was existed already.\n", (unsigned long) id);
        return ENN_RET_FAILED;
    }
    ud_operator_list_map[id] = ud_operators;
    return ENN_RET_SUCCESS;
}

EnnReturn NpuUserDriver::update_graph(uint64_t id, NpuUDOperators ud_operators) {
    std::lock_guard<std::mutex> lock_guard(mutex_ud_operator_list_map);
    if (ud_operator_list_map.find(id) != ud_operator_list_map.end()) {
        ENN_WARN_PRINT("ud_operator_list_map[%lu] was not existed. So just add it.\n", (unsigned long) id);
    }
    ud_operator_list_map[id] = ud_operators;
    return ENN_RET_SUCCESS;
}

EnnReturn NpuUserDriver::get_graph(uint64_t id, NpuUDOperators& ud_operators) {
    ENN_INFO_PRINT("get_graph --> id : %lu\n", (unsigned long) id);
    std::lock_guard<std::mutex> lock_guard(mutex_ud_operator_list_map);
    if (ud_operator_list_map.find(id) == ud_operator_list_map.end()) {
        ENN_ERR_PRINT("ud_operator_list_map[%lu] was not found.\n", (unsigned long) id);
        return ENN_RET_FAILED;
    }
    ud_operators = ud_operator_list_map[id];
    return ENN_RET_SUCCESS;
}

EnnReturn NpuUserDriver::remove_graph(uint64_t id) {
    NpuUDOperators operators;

    if (get_graph(id, operators) != ENN_RET_SUCCESS) {
        return ENN_RET_FAILED;
    }

    for (auto& op : operators) {
        op->deinit();
    }

    std::lock_guard<std::mutex> lock_guard(mutex_ud_operator_list_map);
    if (ud_operator_list_map.erase(id) != 1) {
        ENN_ERR_PRINT("remove ud_operator_list_map[%lu] failed.\n", (unsigned long) id);
        return ENN_RET_FAILED;
    }
    return ENN_RET_SUCCESS;
}

}  // namespace npu
}  // namespace ud
}  // namespace enn
