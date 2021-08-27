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
#include <inttypes.h>      // PRIx64
#include "common/enn_debug.h"
#include "common/compiler.h"
#include "userdriver/unified/dsp_userdriver.h"
#include "model/component/operator/operator.hpp"
#include "model/component/tensor/tensor.hpp"
#include "model/component/tensor/feature_map.hpp"
#include "model/component/tensor/scalar.hpp"
#include "tool/profiler/include/ExynosNnProfilerApi.h"
#include "common/identifier_chopper.hpp"

namespace enn {
namespace ud {
namespace dsp {
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

EnnReturn DspUDOperator::init(uint32_t in_buf_cnt, uint32_t out_buf_cnt, bool isAsync) {
    ENN_DBG_PRINT("in count(%d) out count(%d) for DSP OP.\n", in_buf_cnt, out_buf_cnt);

    if (in_buf_cnt) {
        op_info.bin_in_shape = new shape_t[in_buf_cnt];
        op_info.bin_in_index = new uint32_t[in_buf_cnt];
        op_info.bin_in_bpp = new uint32_t[in_buf_cnt];
    }

    if (out_buf_cnt) {
        op_info.bin_out_shape = new shape_t[out_buf_cnt];
        op_info.bin_out_index = new uint32_t[out_buf_cnt];
    }

    op_info.is_async_execute = is_async_execute_flag_ = isAsync;

    return ENN_RET_SUCCESS;
}

EnnReturn DspUDOperator::set(uint32_t in_buf_cnt, uint32_t out_buf_cnt,
                        model::component::Operator::Ptr rt_opr_dsp,
                        uint64_t operator_list_id, uint64_t unified_op_id) {
    int i, i_reverse;

    ENN_DBG_PRINT("in count(%d) out count(%d) for DSP OP.\n", in_buf_cnt, out_buf_cnt);
    ENN_DBG_PRINT("OP num_in_tensor(%zu) num_out_tensor(%zu)",
                        rt_opr_dsp->in_tensors.count(), rt_opr_dsp->out_tensors.count());

    // Nice to have: TODO(mj.kim010, TBD): get cell_align_shape from RT layer
    shape_t cell_align_shape;
    cell_align_shape.number = 3;
    cell_align_shape.channel = 1;
    cell_align_shape.height = 1;
    cell_align_shape.width = 1;
    cell_align_shape.type_size = 3;

    is_cgo_ = (rt_opr_dsp->get_binaries().at(util::FIRST).get_name().find("TSGD") != std::string::npos);

    if (is_cgo_) {
        auto lib_names = rt_opr_dsp->get_lib_names();
        cgo_info.parse_tsgd(rt_opr_dsp->get_binaries().at(util::FIRST).get_addr(),
                                  rt_opr_dsp->get_binaries().at(util::FIRST).get_size(),
                                  rt_opr_dsp->get_binaries().at(util::FIRST).get_fd(),
                                  rt_opr_dsp->get_binaries().at(util::FIRST).get_offset(),
                                  lib_names, operator_list_id);
        cgo_info.parse_kernel_bin(lib_names);
        cgo_info.create_base_info((in_buf_cnt + out_buf_cnt), &op_info);
        i = 0;
        i_reverse = in_buf_cnt - 1;
        for (auto& tensor : rt_opr_dsp->in_tensors) {
            ENN_DBG_PRINT("Check CGO Tensor name: %s", tensor->get_name().c_str());
//            if (!tensor->is_const()) {  // if tensor is not const, the tensor is feature map.
                int i_temp = i;
                uint32_t buffer_size = 0;
                if (tensor->get_name().find("input_") != std::string::npos) {
                    i++;
                    model::component::FeatureMap::Ptr ifm =
                        std::static_pointer_cast<model::component::FeatureMap>(tensor);
                    op_info.bin_in_index[i_temp] = (uint32_t) ifm->get_buffer_index();
                    buffer_size = ifm->get_buffer_size();
                } else {
                    i_temp = i_reverse;
                    i_reverse--;
                    std::shared_ptr<model::component::Scalar> scalar =
                        std::static_pointer_cast<model::component::Scalar>(tensor);
                    op_info.bin_in_index[i_temp] = (uint32_t) scalar->get_indexed_buffer_index();
                    buffer_size = scalar->get_indexed_buffer_size();
                }
                // Nice to have: TODO(mj.kim010, TBD): handle the case of NHWC
                op_info.bin_in_shape[i_temp].number = tensor->get_shape()[0];
                op_info.bin_in_shape[i_temp].channel = tensor->get_shape()[1];
                op_info.bin_in_shape[i_temp].height = tensor->get_shape()[2];
                op_info.bin_in_shape[i_temp].width = tensor->get_shape()[3];
                op_info.bin_in_shape[i_temp].type_size =
                        model::pixel_bit_format_size[static_cast<TFlite::TensorType>(tensor->get_data_type())];
                op_info.bin_in_bpp[i_temp] = 8 * model::pixel_bit_format_size[
                    static_cast<TFlite::TensorType>(tensor->get_data_type())];
                ENN_DBG_PRINT("in tensor[%d] index(%d) nwhc(%d,%d,%d,%d,{%d})",
                        i_temp, op_info.bin_in_index[i_temp],
                        op_info.bin_in_shape[i_temp].number,
                        op_info.bin_in_shape[i_temp].width,
                        op_info.bin_in_shape[i_temp].height,
                        op_info.bin_in_shape[i_temp].channel,
                        op_info.bin_in_shape[i_temp].type_size);
                cgo_info.update_load_param(tensor->get_name(), op_info.bin_in_index[i_temp], op_info.bin_in_shape[i_temp]);
//            }
        }
        op_info.input_count = (uint8_t) i;
    } else {
        i = 0;
        for (auto& tensor : rt_opr_dsp->in_tensors) {
            if (!tensor->is_const()) {  // if tensor is not const, the tensor is feature map.
                model::component::FeatureMap::Ptr ifm =
                    std::static_pointer_cast<model::component::FeatureMap>(tensor);
                // Nice to have: TODO(mj.kim010, TBD): handle the case of NHWC
                op_info.bin_in_shape[i].number = ifm->get_shape()[0];
                op_info.bin_in_shape[i].channel = ifm->get_shape()[1];
                op_info.bin_in_shape[i].height = ifm->get_shape()[2];
                op_info.bin_in_shape[i].width = ifm->get_shape()[3];
                op_info.bin_in_shape[i].type_size =
                        model::pixel_bit_format_size[static_cast<TFlite::TensorType>(ifm->get_data_type())];
                op_info.bin_in_index[i] = (uint32_t) ifm->get_buffer_index();
                op_info.bin_in_bpp[i] = 8 * model::pixel_bit_format_size[
                    static_cast<TFlite::TensorType>(ifm->get_data_type())];
                ENN_DBG_PRINT("in tensor[%d] index(%d) nwhc(%d,%d,%d,%d,{%d})",
                        i, op_info.bin_in_index[i],
                        op_info.bin_in_shape[i].number,
                        op_info.bin_in_shape[i].width,
                        op_info.bin_in_shape[i].height,
                        op_info.bin_in_shape[i].channel,
                        op_info.bin_in_shape[i].type_size);
                i++;
            }
        }
        op_info.input_count = (uint8_t) i;
    }

    i = 0;
    for (auto& tensor : rt_opr_dsp->out_tensors) {
        if (!tensor->is_const()) {  // if tensor is not const, the tensor is feature map.
            model::component::FeatureMap::Ptr ofm =
                std::static_pointer_cast<model::component::FeatureMap>(tensor);
            // Nice to have: TODO(mj.kim010, TBD): handle the case of NHWC
            op_info.bin_out_shape[i].number = ofm->get_shape()[0];
            op_info.bin_out_shape[i].channel = ofm->get_shape()[1];
            op_info.bin_out_shape[i].height = ofm->get_shape()[2];
            op_info.bin_out_shape[i].width = ofm->get_shape()[3];
            op_info.bin_out_shape[i].type_size =
                        model::pixel_bit_format_size[static_cast<TFlite::TensorType>(ofm->get_data_type())];
            op_info.bin_out_index[i] = (uint32_t) ofm->get_buffer_index();
            ENN_DBG_PRINT("out tensor[%d] index(%d) nwhc(%d,%d,%d,%d,{%d])",
                    i, op_info.bin_out_index[i],
                    op_info.bin_out_shape[i].number,
                    op_info.bin_out_shape[i].width,
                    op_info.bin_out_shape[i].height,
                    op_info.bin_out_shape[i].channel,
                    op_info.bin_out_shape[i].type_size);
            if(is_cgo_) {
                cgo_info.update_load_param(tensor->get_name(), op_info.bin_out_index[i], op_info.bin_out_shape[i]);
            }
            i++;
        }
    }
    op_info.output_count = (uint8_t) i;

    // TODO: Make this iterative with for-loop or function call
    //       This code should cover unified op, NPU, and DSP op.
    op_info.model_name = (uint8_t*) rt_opr_dsp->get_binaries().at(util::FIRST).get_name().c_str();
    op_info.id = generate_op_id(operator_list_id ,rt_opr_dsp->get_id());
    ENN_DBG_PRINT("op_list_id:%lu, op_uid:%lu, op_id:%lx\n", (unsigned long) operator_list_id,
            (unsigned long) rt_opr_dsp->get_id(), (unsigned long) op_info.id);
    if (op_info.id == 0) {
        ENN_ERR_PRINT_FORCE("Invalid op ID:%lx", (unsigned long) op_info.id);
        return ENN_RET_FAILED;
    }

    op_info.operator_list_id = operator_list_id;
    if (unified_op_id > 0)
        op_info.unified_op_id = unified_op_id;
    else
        op_info.unified_op_id = op_info.id;

    // Nice to have: TODO(mj.kim010, TBD): get the cell_align_shape from RT layer.
    op_info.cell_align_shape = cell_align_shape;
    // Nice to have: TODO(mj.kim010, TBD): remove shared_buffer.
    op_info.shared_buffer = -1;
    // Required: TODO(mj.kim010, 6/30): Mirror NPU UD code.
    op_info.binding_ofm = 0;
    // Required: TODO(mj.kim010, 6/30): get tile_size from RT layer.
    op_info.tile_size = 1;

    if (!is_cgo_) {
        ucgo_info.parse_ucgo(rt_opr_dsp->get_binaries().at(util::FIRST).get_addr(),
                             rt_opr_dsp->get_binaries().at(util::FIRST).get_size(),
                             &op_info,
                             rt_opr_dsp->get_binaries().at(util::FIRST).get_fd(),
                             rt_opr_dsp->get_binaries().at(util::FIRST).get_offset());
    }

    return ENN_RET_SUCCESS;
}

EnnReturn DspUDOperator::deinit(void) {
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

EnnReturn DspUDOperator::add_executable_op(uint64_t exec_op_id,
        const std::shared_ptr<ExecutableDspUDOperator>& executable_op) {

    ENN_INFO_PRINT("exec_op_id : %lu executable_op:%p\n", (unsigned long) exec_op_id, executable_op.get());
    std::lock_guard<std::mutex> lock_guard(mutex_executable_op_map);
    if (executable_op_map.find(exec_op_id) != executable_op_map.end()) {
        ENN_WARN_PRINT("executable_op_map[%lu] was existed already.\n", (unsigned long) exec_op_id);
        return ENN_RET_FAILED;
    }
    executable_op_map[exec_op_id] = executable_op;
    return ENN_RET_SUCCESS;
}


std::vector<uint64_t> DspUDOperator::get_all_executable_op_id() {
  std::vector<uint64_t> exec_op_ids;
  std::lock_guard<std::mutex> lock_guard(mutex_executable_op_map);
  for (auto const& element : executable_op_map) {
    exec_op_ids.push_back(element.first);
  }
  return exec_op_ids;
}


std::shared_ptr<ExecutableDspUDOperator> DspUDOperator::get_executable_op(uint64_t exec_op_id) {
    ENN_INFO_PRINT("exec_op_id : %lu\n",  (unsigned long) exec_op_id);
    std::lock_guard<std::mutex> lock_guard(mutex_executable_op_map);
    if (executable_op_map.find(exec_op_id) == executable_op_map.end()) {
        ENN_WARN_PRINT("executable_op_map[%lu] was not found.\n",  (unsigned long) exec_op_id);
        return nullptr;
    }
    return executable_op_map[exec_op_id];
}

EnnReturn DspUDOperator::remove_executable_op(uint64_t exec_op_id) {
    std::lock_guard<std::mutex> lock_guard(mutex_executable_op_map);
    if (executable_op_map.erase(exec_op_id) != 1) {
        ENN_ERR_PRINT("remove executable+__map[%lu] failed.\n",  (unsigned long) exec_op_id);
        return ENN_RET_FAILED;
    }
    return ENN_RET_SUCCESS;
}

DspBinInfo* DspUDOperator::get_bin_info() {
    if (is_cgo())
        return reinterpret_cast<DspBinInfo*>(&(get_cgo_info()));
    else
        return reinterpret_cast<DspBinInfo*>(&(get_ucgo_info()));
}

uint64_t DspUDOperator::get_id(void) {
        return op_info.id;
}

DspUserDriver& DspUserDriver::get_instance(void) {
    static DspUserDriver dsp_userdriver_instance;
    return dsp_userdriver_instance;
}

EnnReturn DspUserDriver::Initialize(void) {
    ENN_DBG_PRINT("DSP UD Initialize start. asyncThread(%p), modelCnt(%d)\n", asyncExecuteThread_, asyncModelCount_.load());

    // Nice to have: TODO(mj.kim010, TBD): refactor max_request_size
    uint32_t max_request_size = 16;
     // Nice to have: TODO(mj.kim010, TBD): replace eden_ret to EnnReturn
    EnnReturn ret;

    // Check if DSP UD is already initialized
    if (get_dsp_ud_status() != DspUdStatus::INITIALIZED) {
        ret = UdLink::get_instance().link_init(acc_, max_request_size);
        if (ret != ENN_RET_SUCCESS) {
            ENN_ERR_PRINT_FORCE("(-) init_acc error: %x\n", ret);
            return ENN_RET_FAILED;
        }

        ud_operator_list_map.clear();
        set_dsp_ud_status(DspUdStatus::INITIALIZED);
    } else {
        ENN_DBG_PRINT("(-) DSP is already INITIALIZED\n");
    }
    ENN_DBG_PRINT("DSP UD Initialize end.n");
    return ENN_RET_SUCCESS;
}

int DspUserDriver::AsyncExecuteLoop() {
    for(;;) {
        std::unique_lock<std::mutex> guard(asyncMutex_);
        asyncCondVar_.wait(guard);
        /* Wake up and handle mail. */
        if (asyncJobQueue_.empty()) {
            ENN_WARN_PRINT_FORCE("Wrong processing : No message but wake.\n");
            continue;
        }

        std::shared_ptr<AsyncJob> jobInstance = asyncJobQueue_.front();
        asyncJobQueue_.pop();
        if (jobInstance->getJobType() == DspAsyncThreadJob::DESTRUCT) {
            ENN_INFO_PRINT_FORCE("Destruct Async thread.\n");
            break;
        }

        ENN_DBG_PRINT("Async execute start\n");
        UdLink::get_instance().link_execute_req(acc_, jobInstance->getReqInfo(), jobInstance->getReqOption());
        ENN_DBG_PRINT("Async execute end. Release asyncJob datas. \n");
    }
    return 0;
}

EnnReturn DspUserDriver::FinishAsyncThread(void) {
    ENN_DBG_PRINT("+\n");
    if (asyncExecuteThread_) {
        {
            std::unique_lock<std::mutex> guard(asyncMutex_);
            asyncJobQueue_.push(std::make_shared<AsyncJob>(DspAsyncThreadJob::DESTRUCT));
        }
        asyncCondVar_.notify_all();
        ENN_DBG_PRINT("Send close message for DSP async execution thread.\n");
        asyncExecuteThread_->join();
        ENN_DBG_PRINT("DSP async execution thread closed.\n");
        delete asyncExecuteThread_;
        asyncExecuteThread_ = nullptr;
        ENN_DBG_PRINT("DSP async execution thread deinitialized.\n");
    }
    ENN_DBG_PRINT("-\n");
    return ENN_RET_SUCCESS;
}

bool DspUserDriver::CheckOpAsyncExecution(const model::component::OperatorList& operator_list) {
    bool hasAsyncOp = false;
    /* Nice to have : TODO(mj.kim010, TBD) : Support multi operater in op_list. */
    for (auto& rt_opr : operator_list) {
        auto rt_opr_dsp = std::static_pointer_cast<model::component::Operator>(rt_opr);
        if (rt_opr_dsp->is_dsp_async_exec()) {
            hasAsyncOp = true;
            break;
        }
    }
    return hasAsyncOp;
}

void DspUserDriver::AddAsyncTriggerInfo() {
    if (asyncModelCount_ == 0) {
        asyncExecuteThread_ = new std::thread(&DspUserDriver::AsyncExecuteLoop, this);
        ENN_DBG_PRINT("Run DSP async execution thread.\n");
    }
    asyncModelCount_++;
    ENN_DBG_PRINT("DSP async execution model count(%d)\n", asyncModelCount_.load());
    return;
}

void DspUserDriver::RemoveAsyncTriggerInfo() {
    if (asyncModelCount_ <= 0) {
        ENN_WARN_PRINT("Wrong count handled for DSP async execution.(%d)\n", asyncModelCount_.load());
        return;
    }
    asyncModelCount_--;
    ENN_DBG_PRINT("DSP async execution remaining model count(%d)\n", asyncModelCount_.load());
    if(asyncModelCount_ == 0) {
        FinishAsyncThread();
    }
    return;
}

EnnReturn DspUserDriver::OpenSubGraph(const model::component::OperatorList& operator_list) {
    uint64_t operator_list_id = operator_list.get_id().get();
    ENN_DBG_PRINT("operator_list_id:%lu\n", (unsigned long) operator_list_id);

    return OpenSubGraph(operator_list, operator_list_id, 0);
}

EnnReturn DspUserDriver::OpenSubGraph(const model::component::OperatorList& operator_list,
        uint64_t operator_list_id, uint64_t unified_op_id) {
    ENN_DBG_PRINT("DSP UD OpenSubGraph() start\n");

    uint32_t in_buf_cnt = 0;
    uint32_t out_buf_cnt = 0;

    // Nice to have: TODO(mj.kim010, TBD): Use preference of OperatorList
    if (get_dsp_ud_status() != DspUdStatus::INITIALIZED) {
        ENN_ERR_PRINT_FORCE("DSP UD is not initialized\n");
        return ENN_RET_FAILED;
    }

    ENN_DBG_PRINT("operator_list_id:%lu\n", (unsigned long) operator_list_id);

    bool hasAsyncOp = CheckOpAsyncExecution(operator_list);

    DspUDOperators ud_operators;
    for (auto& rt_opr : operator_list) {
        auto rt_opr_dsp = std::static_pointer_cast<model::component::Operator>(rt_opr);
        std::shared_ptr<DspUDOperator> ud_op = std::make_shared<DspUDOperator>();

        in_buf_cnt = rt_opr->in_tensors.count();
        out_buf_cnt = rt_opr->out_tensors.count();

        EnnReturn ret_val;

        ENN_DBG_PRINT("in_buf_cnt:%u, out_buf_cnt:%u\n", in_buf_cnt, out_buf_cnt);
        /* Nice to have : TODO(mj.kim010, TBD) : Support different async_flag for each OP. */
        ret_val = ud_op->init(in_buf_cnt, out_buf_cnt, hasAsyncOp);
        if (ret_val) {
            ENN_ERR_PRINT_FORCE("fail to init(), ret:%d\n", ret_val);
            ud_op->deinit();
            return ret_val;
        }

        ret_val = ud_op->set(in_buf_cnt, out_buf_cnt, rt_opr_dsp, operator_list_id, unified_op_id);
        if (ret_val) {
            ENN_ERR_PRINT("fail to set(), ret:%d\n", ret_val);
            ud_op->deinit();
            return ret_val;
        }

        ud_operators.push_back(ud_op);

        // Nice to have: TODO(mj.kim010, TBD): modify EdenModelOptions to UdSubGraphPreference
        EdenModelOptions options;
        uint32_t perf_mode = operator_list.get_pref_mode();
        options.modelPreference.userPreference.hw = DSP_ONLY;
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

        // Nice to have: TODO(mj.kim010, TBD): Deprecate second parameter of open_model()
        EnnReturn ret = UdLink::get_instance().link_open_model(acc_, &ud_op->get(), &options);

        if (ret != ENN_RET_SUCCESS) {
            ENN_ERR_PRINT_FORCE("(-) open_model error: %d\n", ret);
            ud_op->deinit();
            return ENN_RET_FAILED;
        }
    }
    ENN_INFO_PRINT("set_id --> id : %lu\n",  (unsigned long) operator_list_id);

    if (hasAsyncOp)
        AddAsyncTriggerInfo();

    add_graph(operator_list_id, ud_operators);
    ENN_DBG_PRINT("DSP UD OpenSubGraph() end\n");
    return ENN_RET_SUCCESS;
}

EnnReturn ExecutableDspUDOperator::init(uint32_t in_buf_cnt, uint32_t out_buf_cnt) {

    if (in_buf_cnt) {
        executable_op_info.inputs = std::shared_ptr<eden_memory_t>(
                                        new eden_memory_t[in_buf_cnt],
                                        std::default_delete<eden_memory_t[]>());
    }

    if (out_buf_cnt) {
        executable_op_info.outputs = std::shared_ptr<eden_memory_t>(
                                         new eden_memory_t[out_buf_cnt],
                                         std::default_delete<eden_memory_t[]>());
    }
	return ENN_RET_SUCCESS;
}

EnnReturn ExecutableDspUDOperator::deinit(void) {
    return ENN_RET_SUCCESS;
}

EnnReturn ExecutableDspUDOperator::set(model_info_t* op_info, const model::memory::BufferTable& buffer_table) {
    if (op_info == NULL) {
        ENN_ERR_PRINT_FORCE("NULL pointer op_info\n");
        return ENN_RET_FAILED;
    }

    executable_op_info.model_info = op_info;
    // Nice to have: TODO(mj.kim010, TBD): remove requestId because this is not used
    executable_op_info.requestId = 1;
    executable_op_info.operator_list_id = op_info->operator_list_id;

     // Nice to have: TODO(mj.kim010, TBD): remove redundant code for in/out
    for (int i = 0; i < op_info->input_count; i++) {
        ENN_DBG_PRINT("in_cnt[%d] : bin_in_idx(%d)", i, op_info->bin_in_index[i]);
        auto& buffer = buffer_table[op_info->bin_in_index[i]];
        eden_memory_t &em = executable_op_info.inputs.get()[i];
        // Nice to have: TODO(mj.kim010, TBD): get this type from RT layer
        em.type = ION;
        em.size = buffer.get_size();
        em.ref.ion.fd = buffer.get_fd();
        em.ref.ion.buf = (uint64_t) buffer.get_addr();
        em.alloc_size = buffer.get_size();
        ENN_DBG_PRINT("executable_op_info in[%d] size(%zu,%zu) fd(%d)",
                        i, em.size, em.alloc_size, em.ref.ion.fd);
        if (!is_valid_ion_buffer(buffer.get_fd(),
                    buffer.get_size(), buffer.get_addr())) {
            ENN_ERR_PRINT_FORCE("invalid input ION buffer\n");
            return ENN_RET_FAILED;
        }
    }

    for (int i = 0; i < op_info->output_count; i++) {
        ENN_DBG_PRINT("out_cnt[%d] : bin_out_idx(%d)", i, op_info->bin_out_index[i]);
        auto& buffer = buffer_table[op_info->bin_out_index[i]];
        eden_memory_t &em = executable_op_info.outputs.get()[i];
        // Nice to have: TODO(mj.kim010, TBD): get type from RT layer
        em.type = ION;
        em.size = buffer.get_size();
        em.ref.ion.fd = buffer.get_fd();
        em.ref.ion.buf = (uint64_t) buffer.get_addr();
        em.alloc_size = buffer.get_size();
        ENN_DBG_PRINT("executable_op_info out[%d] size(%zu,%zu) fd(%d)",
                        i, em.size, em.alloc_size, em.ref.ion.fd);
        if (!is_valid_ion_buffer(buffer.get_fd(),
                    buffer.get_size(), buffer.get_addr())) {
            ENN_ERR_PRINT_FORCE("invalid output ION buffer\n");
            return ENN_RET_FAILED;
        }
    }

    return ENN_RET_SUCCESS;
}

EnnReturn ExecutableDspUDOperator::set_dsp_exec_info(DspBinInfo *binInfo,
                        const model::memory::BufferTable &buffer_table) {
    const model_info_t* op_info = executable_op_info.model_info;

    ofi_v4_execute_msg_info_t *newExecInfo = nullptr;


    if (executable_op_info.get_exec_info() == nullptr) {
        eden_memory_t *exec_mem = new eden_memory_t;
        /* Nice to have : TODO(mj.kim010, TODO) : Use ENN memory allocator. */
        exec_mem->type = ION;
        exec_mem->size = op_info->get_exec_msg_size();
        uint32_t ema_ret = eden_mem_allocate_with_ion_flag(exec_mem, DSP_MEM_ION_FLAG);
        if (ema_ret != PASS) {
            ENN_ERR_PRINT_FORCE("ion buffer for packed exec info failed");
            return ENN_RET_FAILED;
        }
        executable_op_info.set_exec_info(exec_mem);

        void *srcExecInfo = reinterpret_cast<void*>(binInfo->get_exec_info()->ref.ion.buf);
        ofi_v4_execute_msg_info_t *dstExecInfo =
                        reinterpret_cast<ofi_v4_execute_msg_info_t*>(exec_mem->ref.ion.buf);
        memcpy(dstExecInfo, srcExecInfo, exec_mem->size);
        newExecInfo = dstExecInfo;
    } else { // if exist, use it
        newExecInfo = reinterpret_cast<ofi_v4_execute_msg_info_t*>(executable_op_info.get_exec_info()->ref.ion.buf);
    }

    /* Update params. */
    if (binInfo->is_cgo()) {
        DspCgoInfo *cgo_info = reinterpret_cast<DspCgoInfo*>(binInfo);
        for (int i = 0; i < cgo_info->get_total_param_count(); i++) {
            auto buffer = buffer_table[i];
            cgo_info->update_exec_param(*newExecInfo, i, buffer.get_size(), buffer.get_fd());
        }
    } else {
        DspUcgoInfo *ucgo_info = reinterpret_cast<DspUcgoInfo*>(binInfo);
        ucgo_info->update_exec_info(*newExecInfo, &get());
    }

    return ENN_RET_SUCCESS;
}

ExecutableDspUDOperator::~ExecutableDspUDOperator() {
    /* Nice to have : TODO(mj.kim010, TODO) : Use ENN memory allocator. */
    eden_memory_t *em = executable_op_info.get_exec_info();
        if (em != nullptr)
            eden_mem_free(em);
}

EnnReturn DspUserDriver::UpdateExecutableOp(uint64_t exec_op_id, std::shared_ptr<DspUDOperator> op,
                            std::shared_ptr<ExecutableDspUDOperator> executable_op,
                            const model::memory::BufferTable &buffer_table) {
    model_info_t* op_info = &op->get();
    if (unlikely(executable_op->init(op_info->input_count, op_info->output_count) != ENN_RET_SUCCESS)) {
        ENN_ERR_PRINT("fail to init() of executable_op\n");
        return ENN_RET_FAILED;
    }

    if (unlikely(executable_op->set(op_info, buffer_table) != ENN_RET_SUCCESS)) {
        ENN_ERR_PRINT("fail to set() of executable_op\n");
        return ENN_RET_FAILED;
    }

    if (unlikely(executable_op->set_dsp_exec_info(op->get_bin_info(), buffer_table) != ENN_RET_SUCCESS)) {
        ENN_ERR_PRINT("fail to set_dsp_exec_info() of executable_op\n");
        return ENN_RET_FAILED;
    }

    if (unlikely(op->add_executable_op(exec_op_id, executable_op) != ENN_RET_SUCCESS)) {
        ENN_ERR_PRINT_FORCE("fail to add executable_op\n");
        return ENN_RET_FAILED;
    }

    return ENN_RET_SUCCESS;
}

EnnReturn DspUserDriver::PrepareSubGraph(const enn::runtime::ExecutableOperatorList& executable_operator_list) {
    ENN_DBG_PRINT("DSP UD prepare start.\n");

    DspUDOperators operators;

    uint64_t operator_list_id = executable_operator_list.get_operator_list_id().get();
    if (get_dsp_ud_status() != DspUdStatus::INITIALIZED) {
        ENN_ERR_PRINT_FORCE("DSP UD is not initialized\n");
        return ENN_RET_FAILED;
    }

    if (get_graph(operator_list_id, operators) != ENN_RET_SUCCESS) {
        return ENN_RET_FAILED;
    }

    for (auto& op : operators) {
        model_info_t* op_info;
        std::shared_ptr<ExecutableDspUDOperator> executable_op;
        uint64_t exec_op_id = executable_operator_list.get_id().get();

        op_info = &op->get();
        if (op_info == NULL) {
            ENN_ERR_PRINT("NULL pointer op_info\n");
            return ENN_RET_FAILED;
        }

        executable_op = std::make_shared<ExecutableDspUDOperator>();
        UpdateExecutableOp(exec_op_id, op, executable_op, executable_operator_list.get_buffer_table());

        EnnReturn ret = UdLink::get_instance().link_prepare_req(acc_, op_info,
                                            executable_op->get_inputs(), executable_op->get_outputs(),
                                            executable_op->get().execute_info);
        if (ret != ENN_RET_SUCCESS) {
            ENN_ERR_PRINT_FORCE("(-) failed Err[%d] prepare_req() \n", ret);
            return ENN_RET_FAILED;
        }
    }
    ENN_DBG_PRINT("DSP UD PrepareSubGraph().\n");
    return ENN_RET_SUCCESS;
}

EnnReturn DspUserDriver::ExecuteSubGraph(const enn::runtime::OperatorListExecuteRequest& operator_list_execute_request) {
    ENN_DBG_PRINT("DSP UD ExecuteSubGraph() start\n");

    if (unlikely(get_dsp_ud_status() != DspUdStatus::INITIALIZED)) {
        ENN_ERR_PRINT_FORCE("DSP UD is not initialized\n");
        return ENN_RET_FAILED;
    }

    DspUDOperators operators;

    uint64_t operator_list_id = operator_list_execute_request.get_operator_list_id().get();
    PROFILE_SCOPE("DSP_UD_Execution", util::chop_into_model_id(operator_list_id));

    if (unlikely(get_graph(operator_list_id, operators) != ENN_RET_SUCCESS)) {
        ENN_ERR_PRINT("Fail to get dsp op. id(0x%" PRIX64 "\n", operator_list_id);
        return ENN_RET_FAILED;
    }

    for (auto& op : operators) {
        model_info_t* op_info;
        std::shared_ptr<ExecutableDspUDOperator> executable_op;
        uint64_t exec_op_id = operator_list_execute_request.get_executable_operator_list_id().get();

        op_info = &op->get();
        if (unlikely(op_info == NULL)) {
            ENN_ERR_PRINT_FORCE("NULL pointer error, op_info\n");
            return ENN_RET_FAILED;
        }

        executable_op = op->get_executable_op(exec_op_id);
        if (unlikely(executable_op == nullptr)) {  // if the buffer is not pre-allocated by prepare().
            executable_op = std::make_shared<ExecutableDspUDOperator>();
            UpdateExecutableOp(exec_op_id, op, executable_op, operator_list_execute_request.get_buffer_table());
        }

        // Nice to have: TODO(mj.kim010, TBD): remove EdenRequestOptions because it will be deprecated
        EdenRequestOptions options;
        options.userPreference.hw = DSP_ONLY;
        options.userPreference.mode = BOOST_MODE;
        options.requestMode = BLOCK;

        /* Nice to have : TODO(mj.kim010, TODO) : Link layer should return EnnReturn type. */
        if (op->get_async_execute_flag()) {
            std::unique_lock<std::mutex> guard(asyncMutex_);
            asyncJobQueue_.push(std::make_shared<AsyncJob>(DspAsyncThreadJob::EXECUTE, &executable_op->get(), options));
            asyncCondVar_.notify_all();
        }
        else {
            EnnReturn ret = UdLink::get_instance().link_execute_req(acc_, &executable_op->get(), &options);
            if (unlikely(ret != ENN_RET_SUCCESS)) {
                ENN_ERR_PRINT_FORCE("(-) failed Err[%d] execute_req() \n", ret);
                return ENN_RET_FAILED;
            }
            if (!op->is_cgo()) {
                DspUcgoInfo *ucgo_info = reinterpret_cast<DspUcgoInfo*>(op->get_bin_info());
                ucgo_info->dump_lbl_intermediate_buffer();
            }
        }

    /* Nice to have : TODO(mj.kim010, TODO) Check if we need this for non-prepare case. */
#if 0
        executable_op->deinit();
        op->remove_executable_op(exec_op_id);
#endif
    }

    ENN_DBG_PRINT("DSP UD ExecuteSubGraph() end\n");
    return ENN_RET_SUCCESS;
}

EnnReturn DspUserDriver::CloseSubGraph(const model::component::OperatorList& operator_list) {
    uint64_t operator_list_id = operator_list.get_id().get();
    ENN_DBG_PRINT("operator_list_id:%lu\n", (unsigned long) operator_list_id);

    return CloseSubGraph(operator_list, operator_list_id);
}

EnnReturn DspUserDriver::CloseSubGraph(const model::component::OperatorList& operator_list, uint64_t operator_list_id) {
    ENN_DBG_PRINT("DSP UD CloseSubGraph() start.\n");
    bool hasAsyncOp = false;

    if (get_dsp_ud_status() != DspUdStatus::INITIALIZED) {
        ENN_ERR_PRINT_FORCE("DSP UD is not initialized\n");
        return ENN_RET_FAILED;
    }

    DspUDOperators operators;

    if (get_graph(operator_list_id, operators) != ENN_RET_SUCCESS) {
        return ENN_RET_FAILED;
    }

    for (auto& op : operators) {
        std::shared_ptr<ExecutableDspUDOperator> executable_op;
        if (op->get_async_execute_flag()) {
            hasAsyncOp = true;
        }
        const std::vector<uint64_t> &exec_op_ids = op->get_all_executable_op_id();
        for (auto &exec_op_id : exec_op_ids) {
            executable_op = op->get_executable_op(exec_op_id);
            // TODO(mj.kim10): if (executable_op) { // nullptr is false
            if (executable_op != nullptr) {
                executable_op->deinit();
                op->remove_executable_op(exec_op_id);
            }
        }
    }

     // Nice to have: TODO(mj.kim010, TBD): rename? close_operator_list()?
    if (UdLink::get_instance().link_close_model(acc_, operator_list_id) != ENN_RET_SUCCESS) {  // legacy func()
        ENN_ERR_PRINT_FORCE("(-) fail close_model()\n");
        return ENN_RET_FAILED;
    }

    if (remove_graph(operator_list_id) != ENN_RET_SUCCESS) {
        ENN_ERR_PRINT_FORCE("(-) fail remove_graph()\n");
        return ENN_RET_FAILED;
    }

    if (hasAsyncOp)
        RemoveAsyncTriggerInfo();

    ENN_DBG_PRINT("DSP UD CloseSubGraph() end.\n");
    ENN_UNUSED(operator_list);
    return ENN_RET_SUCCESS;
}

EnnReturn DspUserDriver::Deinitialize(void) {
    ENN_DBG_PRINT("DSP UD Deinitialize() start.\n");

    FinishAsyncThread();

    if (get_dsp_ud_status() == DspUdStatus::SHUTDOWNED) {
        ENN_DBG_PRINT("(-) DSP is already SHUTDOWNED\n");
        return ENN_RET_SUCCESS;
    }

    EnnReturn ret = UdLink::get_instance().link_shutdown(acc_);
    if (ret != ENN_RET_SUCCESS) {
        ENN_ERR_PRINT_FORCE("(-) failed Err[%d] shutdown() \n", ret);
        return ENN_RET_FAILED;
    }

    set_dsp_ud_status(DspUdStatus::SHUTDOWNED);

    ud_operator_list_map.clear();

    if (!ud_operator_list_map.empty()) {
        return ENN_RET_FAILED;
    }

    ENN_DBG_PRINT("DSP UD Deinitialize() end.\n");
    return ENN_RET_SUCCESS;
}

EnnReturn DspUserDriver::add_graph(uint64_t id, DspUDOperators ud_operators) {
    std::lock_guard<std::mutex> lock_guard(mutex_ud_operator_list_map);
    if (ud_operator_list_map.find(id) != ud_operator_list_map.end()) {
        ENN_ERR_PRINT("ud_operator_list_map[%lu] was existed already.\n", (unsigned long) id);
        return ENN_RET_FAILED;
    }
    ud_operator_list_map[id] = ud_operators;
    return ENN_RET_SUCCESS;
}

EnnReturn DspUserDriver::update_graph(uint64_t id, DspUDOperators ud_operators) {
    std::lock_guard<std::mutex> lock_guard(mutex_ud_operator_list_map);
    if (ud_operator_list_map.find(id) != ud_operator_list_map.end()) {
        ENN_WARN_PRINT("ud_operator_list_map[%lu] was not existed. So just add it.\n",  (unsigned long) id);
    }
    ud_operator_list_map[id] = ud_operators;
    return ENN_RET_SUCCESS;
}

EnnReturn DspUserDriver::get_graph(uint64_t id, DspUDOperators& ud_operators) {
    std::lock_guard<std::mutex> lock_guard(mutex_ud_operator_list_map);
    ENN_INFO_PRINT("get_graph --> id : %lu\n", (unsigned long)id);
    if (ud_operator_list_map.find(id) == ud_operator_list_map.end()) {
        ENN_ERR_PRINT("ud_operator_list_map[%lu] was not found.\n", (unsigned long) id);
        return ENN_RET_FAILED;
    }
    ud_operators = ud_operator_list_map[id];
    return ENN_RET_SUCCESS;
}

EnnReturn DspUserDriver::remove_graph(uint64_t id) {
    DspUDOperators operators;
    ENN_DBG_PRINT("+");
    if (get_graph(id, operators) != ENN_RET_SUCCESS) {
        ENN_ERR_PRINT("Fail to get graph(0x%" PRIx64 ").\n", id);
        return ENN_RET_FAILED;
    }

    for (auto& op : operators) {
        op->deinit();
    }

    {
        std::lock_guard<std::mutex> lock_guard(mutex_ud_operator_list_map);
        if (ud_operator_list_map.erase(id) != 1) {
            ENN_ERR_PRINT("remove ud_operator_list_map[%lu] failed.\n", (unsigned long) id);
            return ENN_RET_FAILED;
        }
    }
    ENN_DBG_PRINT("-");
    return ENN_RET_SUCCESS;
}

EnnReturn DspUserDriver::get_dsp_session_id(enn::runtime::ExecutableOpListSessionInfo& op_list_session_info) {
    int32_t session_id = LINK_ID_INVALID;
    uint64_t model_id = op_list_session_info.get_id().get();

    if (UdLink::get_instance().link_get_dd_session_id(acc_, model_id, session_id)) {
        ENN_ERR_PRINT_FORCE("Fail to get DD sessionID from Link. model_id(0x%" PRIx64 ")\n", model_id);
        return ENN_RET_FAILED;
    }

    ENN_INFO_PRINT_FORCE("DSP UD set dd_session_id(%d) for model(0x%" PRIx64 ")\n",
                                                            session_id, model_id);
    op_list_session_info.set_session_id(session_id);
    return ENN_RET_SUCCESS;
}

}  // namespace dsp
}  // namespace ud
}  // namespace enn

