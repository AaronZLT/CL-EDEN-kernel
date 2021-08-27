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
#include "userdriver/unified/unified_userdriver.h"
#include "userdriver/unified/utils.h"
#include "model/component/operator/operator.hpp"
#include "model/component/tensor/tensor.hpp"
#include "model/component/tensor/feature_map.hpp"
#include "model/component/operator/operator_builder.hpp"
#include "model/component/operator/operator_list_builder.hpp"
#include "tool/profiler/include/ExynosNnProfilerApi.h"

namespace enn {
namespace ud {
namespace unified {

UnifiedUserDriver::~UnifiedUserDriver(void) {
    ENN_DBG_PRINT("started\n");

    ENN_DBG_PRINT("ended successfully\n");
}

UnifiedUserDriver& UnifiedUserDriver::get_instance(void) {
    static UnifiedUserDriver unified_userdriver_instance;
    return unified_userdriver_instance;
}

EnnReturn UnifiedUserDriver::set_unified_ud_status(UnifiedUdStatus unified_ud_status) {
    std::lock_guard<std::mutex> lock_guard(mutex_unified_ud_status);
    unified_ud_status_ = unified_ud_status;
    return ENN_RET_SUCCESS;
}

EnnReturn UnifiedUserDriver::Initialize(void) {
    EnnReturn ret;

    ENN_DBG_PRINT("Unified UD Initialize() start\n");

    npu_ud = &ud::npu::NpuUserDriver::get_instance();
    dsp_ud = &ud::dsp::DspUserDriver::get_instance();

    // Check if Unified UD is already initialized
    if (get_unified_ud_status() != UnifiedUdStatus::INITIALIZED) {
        ret = dsp_ud->Initialize();
        if (ret != ENN_RET_SUCCESS) {
            ENN_ERR_PRINT_FORCE("(-) DSP UD initialization error: %x\n",ret);
            return ENN_RET_FAILED;
        }
        ret = npu_ud->Initialize();
        if (ret != ENN_RET_SUCCESS) {
            ENN_ERR_PRINT_FORCE("(-) NPU UD initialization error: %x\n", ret);
            return ENN_RET_FAILED;
        }
        set_unified_ud_status(UnifiedUdStatus::INITIALIZED);
    } else {
        ENN_DBG_PRINT("(-) Unified is already INITIALIZED\n");
    }

    ENN_DBG_PRINT("Unified UD Initialize() end\n");
    return ENN_RET_SUCCESS;
}

EnnReturn UnifiedUserDriver::check_validity_op_list(const model::component::OperatorList& operator_list) {
    if (operator_list.get_size() > 1) {
        ENN_ERR_PRINT_FORCE("multiple ops(%d) isn't supported in UUD\n", (int) operator_list.get_size());
        return ENN_RET_FAILED;
    }
    return ENN_RET_SUCCESS;
}

EnnReturn UnifiedUserDriver::check_validity_op(std::shared_ptr<model::component::Operator> op) {
    if (op->get_option().get_addr() != nullptr &&
            op->get_option().get_enum() == TFlite::BuiltinOptions_ENN_UNIFIED_DEVICEOptions) {
        auto uni_op_options = reinterpret_cast<const TFlite::ENN_UNIFIED_DEVICEOptions*>(op->get_option().get_addr());

        if (!uni_op_options) {
            ENN_ERR_PRINT_FORCE("uni_op_options NULL pointer error!\n");
            return ENN_RET_FAILED;
        }

        for (auto option : *uni_op_options->options()) {
            if (check_validity_op_option(option) != ENN_RET_SUCCESS) {
                ENN_ERR_PRINT_FORCE("fail to check validity of an unified operator option.\n");
                return ENN_RET_FAILED;
            }
        }
        return ENN_RET_SUCCESS;
    } else {
        ENN_ERR_PRINT_FORCE("Invalid NNC! It does not include unified options.\n");
        return ENN_RET_FAILED;
    }
}

EnnReturn UnifiedUserDriver::check_validity_op_option(const tflite::v2::ENN_UNIFIED_DEVICE_BinaryOptions* option) {

    ENN_DBG_PRINT("Unified op option, target_hw:%d start_bin:%d end_bin:%d binary_name:%s tensor_num:%d\n",
            (int) option->target_hw(), (int) option->start(), (int) option->end(),
            option->tensor_name()->c_str(), option->tensor_connections()->size());

    if (option->target_hw() == TFlite::TargetHw::TargetHw_NPU) {
        if ((option->start() != true) || (option->end() != true)) {
            ENN_ERR_PRINT_FORCE("Invalid NNC! start_bin(%d) and end_bin(%d) of a NPU binary must be true\n",
                    (int) option->start(), (int) option->end());
            return ENN_RET_FAILED;
        }
        if (option->tensor_connections()->size() > 0) {
            for (size_t i = 0; i < option->tensor_connections()->size(); ++i) {
                int32_t tensor_id = option->tensor_connections()->Get(i);
                ENN_DBG_PRINT("tensor_id:%d\n", tensor_id);
            }
        } else {
            ENN_ERR_PRINT_FORCE("Invalid NNC! NPU binary must have tensors in a NN model.\n");
            return ENN_RET_FAILED;
        }
    } else if (option->target_hw() == TFlite::TargetHw::TargetHw_DSP) {
        if (option->tensor_connections()->size() > 0) {
            ENN_ERR_PRINT_FORCE("Invalid NNC! DSP binary must not have a tensor(%d).\n",
                    (int) option->tensor_connections()->size());
            return ENN_RET_FAILED;
        }
    } else {
        ENN_ERR_PRINT_FORCE("Invalid NNC! It includes invalid binary type(%d).\n", (int) option->target_hw());
        return ENN_RET_FAILED;
    }
    return ENN_RET_SUCCESS;
}

EnnReturn UnifiedUserDriver::OpenSubGraph(const model::component::OperatorList& operator_list) {
    ENN_DBG_PRINT("Unified UD OpenSubGraph() start\n");

    if (get_unified_ud_status() != UnifiedUdStatus::INITIALIZED) {
        ENN_ERR_PRINT_FORCE("Unified UD is not initialized\n");
        return ENN_RET_FAILED;
    }

    if (check_validity_op_list(operator_list) != ENN_RET_SUCCESS) {
        ENN_ERR_PRINT_FORCE("fail to pass the validity check of an operator list\n");
        return ENN_RET_FAILED;
    }

    enn::model::component::OperatorListBuilder dsp_opr_list_builder;
    enn::model::component::OperatorListBuilder npu_opr_list_builder;

    std::shared_ptr<enn::model::component::OperatorList> dsp_opr_list;
    std::shared_ptr<enn::model::component::OperatorList> npu_opr_list;

    dsp_opr_list_builder.build(operator_list.get_id());
    npu_opr_list_builder.build(operator_list.get_id());

    ENN_DBG_PRINT("operator_list_id:0x%" PRIX64 "\n", operator_list.get_id().get());
    for (auto& rt_op : operator_list) {
        // For each opeartor, create a NPU operator list and a DSP operator list.
        auto rt_opr_unified = std::static_pointer_cast<model::component::Operator>(rt_op);
        if (check_validity_op(rt_opr_unified) != ENN_RET_SUCCESS) {
            ENN_ERR_PRINT_FORCE("fail to pass the validity check of an operator\n");
            return ENN_RET_FAILED;
        }

        int op_idx_npu = 0;
        int op_idx_dsp = 0;
        for (auto& binary : rt_opr_unified->get_binaries()) {
            std::shared_ptr<enn::model::component::Operator> opr;
            enn::model::component::OperatorBuilder opr_builder;

            TFlite::TargetHw target_hw = static_cast<TFlite::TargetHw>(binary.get_accelerator());
            std::vector<std::string> lib_names;
            lib_names = rt_opr_unified->get_lib_names();

            // Set the attributes of an operator
            opr_builder.set_name(binary.get_name())
                .set_buffer_shared(rt_opr_unified->is_buffer_shared())
                .set_ofm_bound(rt_opr_unified->is_ofm_bound())
                .set_lib_names(lib_names);

            // Add an operator into the operator list with setting the binary type of operator.
            if (target_hw == TFlite::TargetHw::TargetHw_NPU) {
                // TODO(jungho7.kim): tensors should be added according to the tensor_connections
                for (auto& tensor : rt_opr_unified->in_tensors) {
                    opr_builder.add_in_tensor(tensor);
                }
                for (auto& tensor : rt_opr_unified->out_tensors) {
                    opr_builder.add_out_tensor(tensor);
                }
                opr = opr_builder.set_accelerator(model::Accelerator::NPU)
                    .set_id(op_idx_npu++)
                    .add_binary(binary.get_name(), binary.get_fd(),
                            static_cast<const void*>(static_cast<const uint8_t*>(binary.get_addr()) + binary.get_offset()),
                            binary.get_size(), binary.get_offset(), binary.get_accelerator()).create();
                npu_opr_list_builder.add_operator(opr);
            } else if (target_hw == TFlite::TargetHw::TargetHw_DSP) {
                opr = opr_builder.set_accelerator(model::Accelerator::DSP)
                    .set_id(op_idx_dsp++)
                    .add_binary(binary.get_name(), binary.get_fd(), binary.get_addr(),
                            binary.get_size(), binary.get_offset(), binary.get_accelerator()).create();
                dsp_opr_list_builder.add_operator(opr);
            } else {
                ENN_ERR_PRINT_FORCE("Invalid unified operator target_hw:%d\n", target_hw);
                return ENN_RET_FAILED;
            }
        }
    }

    dsp_opr_list = dsp_opr_list_builder
        .set_tile_num(operator_list.get_tile_num())
        .set_priority(operator_list.get_priority())
        .set_preset_id(operator_list.get_preset_id())
        .set_pref_mode(operator_list.get_pref_mode())
        .set_target_latency(operator_list.get_target_latency())
        .set_core_affinity(operator_list.get_core_affinity())
        .create();
    npu_opr_list = npu_opr_list_builder
        .set_tile_num(operator_list.get_tile_num())
        .set_priority(operator_list.get_priority())
        .set_preset_id(operator_list.get_preset_id())
        .set_pref_mode(operator_list.get_pref_mode())
        .set_target_latency(operator_list.get_target_latency())
        .set_core_affinity(operator_list.get_core_affinity())
        .create();

    // Call DSP UD API
    if (dsp_ud->OpenSubGraph(*dsp_opr_list, operator_list.get_id().get(),
                operator_list.get_id().get()) != ENN_RET_SUCCESS) {
        ENN_ERR_PRINT_FORCE("fail to OpenSubGraph() in DSP UD\n");
        return ENN_RET_FAILED;
    }

    // Call NPU UD API
    if (npu_ud->OpenSubGraph(*npu_opr_list, operator_list.get_id().get(),
                operator_list.get_id().get()) != ENN_RET_SUCCESS) {
        ENN_ERR_PRINT_FORCE("fail to OpenSubGraph() in NPU UD\n");
        return ENN_RET_FAILED;
    }

    ENN_DBG_PRINT("Unified UD OpenSubGraph() end\n");
    return ENN_RET_SUCCESS;
}

EnnReturn UnifiedUserDriver::PrepareSubGraph(const enn::runtime::ExecutableOperatorList& executable_operator_list) {
    ENN_DBG_PRINT("Unified UD PrepareSubGraph() start\n");

    if (get_unified_ud_status() != UnifiedUdStatus::INITIALIZED) {
        ENN_ERR_PRINT_FORCE("Unified UD is not initialized\n");
        return ENN_RET_FAILED;
    }

    ENN_DBG_PRINT("operator_list_id:0x%" PRIX64 "\n", executable_operator_list.get_operator_list_id().get());

    // Create an executable_operator_list for NPU UD
    // Call NPU UD API
    if (npu_ud->PrepareSubGraph(executable_operator_list) != ENN_RET_SUCCESS) {
        ENN_ERR_PRINT_FORCE("fail to PrepareSubGraph() in NPU UD\n");
        return ENN_RET_FAILED;
    }

    ENN_DBG_PRINT("Unified UD PrepareSubGraph() end\n");
    return ENN_RET_SUCCESS;
}

EnnReturn UnifiedUserDriver::ExecuteSubGraph(const enn::runtime::OperatorListExecuteRequest& operator_list_execute_request) {
    ENN_DBG_PRINT("Unified UD ExecuteSubGraph() start\n");

    if (unlikely(get_unified_ud_status() != UnifiedUdStatus::INITIALIZED)) {
        ENN_ERR_PRINT_FORCE("Unified UD is not initialized\n");
        return ENN_RET_FAILED;
    }

    ENN_DBG_PRINT("operator_list_id:0x%" PRIX64 "\n", operator_list_execute_request.get_operator_list_id().get());

    // Call NPU UD API
    if (npu_ud->ExecuteSubGraph(operator_list_execute_request) != ENN_RET_SUCCESS) {
        ENN_ERR_PRINT_FORCE("fail to ExecuteSubGraph() in NPU UD\n");
        return ENN_RET_FAILED;
    }

    ENN_DBG_PRINT("Unified UD ExecuteSubGraph() end\n");
    return ENN_RET_SUCCESS;
}

EnnReturn UnifiedUserDriver::CloseSubGraph(const model::component::OperatorList& operator_list) {
    ENN_DBG_PRINT("Unified UD CloseSubGraph() start\n");

    if (get_unified_ud_status() != UnifiedUdStatus::INITIALIZED) {
        ENN_ERR_PRINT_FORCE("Unified UD is not initialized\n");
        return ENN_RET_FAILED;
    }

    ENN_DBG_PRINT("operator_list_id:0x%" PRIX64 "\n", operator_list.get_id().get());

    // Call DSP UD API
    if (dsp_ud->CloseSubGraph(operator_list, operator_list.get_id().get()) != ENN_RET_SUCCESS) {
        ENN_ERR_PRINT_FORCE("fail to CloseSubGraph() in DSP UD\n");
        return ENN_RET_FAILED;
    }

    // Call NPU UD API
    if (npu_ud->CloseSubGraph(operator_list, operator_list.get_id().get()) != ENN_RET_SUCCESS) {
        ENN_ERR_PRINT_FORCE("fail to CloseSubGraph() in NPU UD\n");
        return ENN_RET_FAILED;
    }

    ENN_DBG_PRINT("Unified UD CloseSubGraph() end\n");
    return ENN_RET_SUCCESS;
}

EnnReturn UnifiedUserDriver::Deinitialize(void) {
    EnnReturn ret;

    ENN_DBG_PRINT("Unified UD Deinitialize() start\n");

    if (get_unified_ud_status() == UnifiedUdStatus::SHUTDOWNED) {
        ENN_DBG_PRINT("(-) Unified is already SHUTDOWNED\n");
        return ENN_RET_SUCCESS;
    }

    ret = dsp_ud->Deinitialize();
    if (ret != ENN_RET_SUCCESS) {
        ENN_ERR_PRINT_FORCE("(-) DSP UD initialization error: %x\n", ret);
        return ENN_RET_FAILED;
    }

    ret = npu_ud->Deinitialize();
    if (ret != ENN_RET_SUCCESS) {
        ENN_ERR_PRINT_FORCE("(-) NPU UD initialization error: %x\n", ret);
        return ENN_RET_FAILED;
    }

    set_unified_ud_status(UnifiedUdStatus::SHUTDOWNED);

    ENN_DBG_PRINT("Unified UD Deinitialize() end\n");

    return ENN_RET_SUCCESS;
}

}  // namespace unified
}  // namespace ud
}  // namespace enn
