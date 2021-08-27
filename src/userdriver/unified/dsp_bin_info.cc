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

#include "dsp_bin_info.h"
#include "common/enn_utils.h"
#include "ofi_orca_fw_graph_parser.h"

#define DSP_UD_CGO_RT_PARSER

/* Common part of UCGO/CGO. */
uint32_t DspBinInfo::create_dsp_global_id(uint64_t op_list_id, uint16_t ucgo_uid) {
    /* Hard-coded because ID class is not visible to UD.
     * Required : TODO: (mj.kim010, TBD) : Not use this ID field.
     *                                     Need to implement with DD/FW. */
    const int shift_value = 16;
    const uint16_t mask_value = 0xFFFF;

    union global_id gidUnion;
    gidUnion.gid.enn_id = (uint16_t)(op_list_id >> shift_value & mask_value);
    gidUnion.gid.ucgo_uid = ucgo_uid;

    ENN_DBG_PRINT("DSP ID : mdl_id(0x%" PRIx16 " ucgo_uid(0x%" PRIx16 ")\n",
                        gidUnion.gid.enn_id, ucgo_uid);

    return gidUnion.gid_num;
}

EnnReturn DspBinInfo::parse_kernel_bin(std::vector<std::string>& lib_names) {
    uint32_t totalKernelNameLen = 0;
    for (auto & lib_name : lib_names) {
        totalKernelNameLen += (lib_name.length() + 1);
    }
    kernel_name_count_ = lib_names.size();
    kernel_name_size_ = kernel_name_count_ * sizeof(uint32_t) + totalKernelNameLen;
    kernel_name_ = new uint8_t[kernel_name_size_];
    uint32_t* length = reinterpret_cast<uint32_t*>(kernel_name_);
    uint8_t* kernelBuffer = kernel_name_ + (sizeof(uint32_t)*kernel_name_count_);

    ENN_DBG_PRINT("# of kernel bin for cgo: %d\n", kernel_name_count_);
    for (int i = 0; i < kernel_name_count_; i++) {
        length[i] = lib_names[i].length() + 1;
        memcpy(kernelBuffer, lib_names[i].c_str(), length[i]);
        kernelBuffer += length[i];
        ENN_DBG_PRINT("lib(%d): %s\n", i, lib_names[i].c_str());
    }

    return ENN_RET_SUCCESS;
}

void DspBinInfo::export_param_to_file(std::string idx_str, const void *dump_va, int size) {
    std::string filename = "/data/vendor/enn/";
    filename.append("dspParam_").append(idx_str).append(".bin");
    ENN_DBG_PRINT("Dump param[%s] to %s\n", idx_str.c_str(), filename.c_str());
    enn::util::export_mem_to_file(filename.c_str(), dump_va, size);
    sleep(3); // Wait so that can dump before kernel panic
    return;
}

/* Debug show purpose */
const std::string DspBinInfo::get_paramtype_str(enum DspMemType_e mem_type) {
    static std::map<enum DspMemType_e, std::string> string_table = {
        {DSP_GRAPH_BIN, "DSP_GRAPH_BIN"},
        {CMDQ_BIN, "CMDQ_BIN"},
        {KERNEL_BIN_STR, "KERNEL_BIN_STR"},
        {INPUT, "INPUT"},
        {OUTPUT, "OUTPUT"},
        {TEMP, "TEMP"},
        {WEIGHT, "WEIGHT"},
        {BIAS, "BIAS"},
        {SCALAR, "SCALAR"},
        {CUSTOM, "CUSTOM"},
        {EMPTY, "EMPTY"},
    };
    auto itr = string_table.find(mem_type);
    if (itr == string_table.end()) {
        ENN_WARN_PRINT_FORCE("No typename for memtype(%d)\n", mem_type);
        return std::string("");
    }
    return itr->second;
}

bool DspUcgoInfo::is_lbl_propery_set() {
    /* TODO: Check property */
    enn::debug::MaskType env_val = 0;
    if (enn::util::get_environment_property(DEBUG_PROPERTY_DSP_LBL.c_str(), &env_val)) {
        return false;
    }
    ENN_INFO_PRINT("Layer-by-layer intermediate dump for DSP : (0x%" PRIx64 ").\n", env_val);
    if (env_val) {
        ENN_INFO_PRINT_FORCE("Layer-by-layer DSP IM dump enabled\n");
        return true;
    }
    else {
        return false;
    }
}


/* DspUcgoInfo */

/* Allocate new memory and copy UCGO's data. */
EnnReturn DspUcgoInfo::reserve_lbl_intermediate_buffer(const void *src_va, int32_t size, int32_t &fd_im) {
    eden_memory_t intermediate_mem;
    intermediate_mem.type = ION;
    intermediate_mem.size = size;
    if (eden_mem_allocate_with_ion_flag(&intermediate_mem, DSP_MEM_ION_FLAG)) {
        ENN_ERR_PRINT_FORCE("ION Alloc fail for inter\n");
        return ENN_RET_FAILED;
    }
    /* Copy UCGO embedded data to new memory. */
    memcpy(reinterpret_cast<void*>(intermediate_mem.ref.ion.buf), src_va, size);
    /* Append to FD maps to find and to release. */
    intermediate_buffer_key_ = fd_im = intermediate_mem.ref.ion.fd;
    ucgo_buffer_map_[intermediate_buffer_key_] = intermediate_mem;
    return ENN_RET_SUCCESS;
}

void DspUcgoInfo::dump_lbl_intermediate_buffer(std::string file_prefix) {
    // skip if intermediate_buffer_key_ is init_val(-1)
    if (intermediate_buffer_key_ > 0) {
        auto itr = ucgo_buffer_map_.find(intermediate_buffer_key_);
        if (itr == ucgo_buffer_map_.end()) {
            ENN_WARN_PRINT_FORCE("No memory for lbl intermediate. (%d)\n", intermediate_buffer_key_);
            return;
        }
        eden_memory_t &im_mem = itr->second;

        std::string dump_file_name;
        dump_file_name.append(SAVE_FILEPATH_DSP_LBL)\
                        .append(file_prefix)\
                        .append(SAVE_FILENAME_DSP_LBL);
        ENN_INFO_PRINT_FORCE("Dump intermediate buffer. size(%zu)(%s)\n",
                                im_mem.size, dump_file_name.c_str());
        enn::util::export_mem_to_file(dump_file_name.c_str(),
                    reinterpret_cast<void*>(im_mem.ref.ion.buf), im_mem.size);
    }
    return;
}

/* Debug show purpose */
std::string DspUcgoInfo::get_loadtype_str(enum DspLoadType_e load_type) {
    static std::map<enum DspLoadType_e, std::string> string_table = {
        {ALLOC, "ALLOC"},
        {ALLOC_ZERO, "ALLOC_ZERO"},
        {ALLOC_LOAD, "ALLOC_LOAD"},
        {IMPORT, "IMPORT"},
        {BYPASS, "BYPASS"},
    };
    auto itr = string_table.find(load_type);
    if (itr == string_table.end()) {
        ENN_WARN_PRINT_FORCE("No typename for load_type(%d)\n", load_type);
        return std::string("");
    }
    return itr->second;
}

bool DspUcgoInfo::is_mapping_index_populated(ofi_v4_param_t &prm) {
    uint32_t p_idx = prm.idx.param_index;
    uint32_t p_type = prm.param_type;
    if (p_idx == MAPPING_INDEX_INVALID || p_type == KERNEL_BIN_STR)
        return false;
    return true;
}

void DspUcgoInfo::dump_raw_ucgo(const DspUcgoHeader *ucgo) {
    ENN_DBG_PRINT("------------------------------ UCGO raw -----------------------------");
    ENN_DBG_PRINT("magic(0x%x) unique_id(0x%x) total_size(%d) numList(%d)\n",
                    ucgo->magic, ucgo->unique_id, ucgo->totalSize, ucgo->numList);
    for (int i = 0; i < ucgo->numList; i++) {
        const DspMemInfo &info = ucgo->info[i];
        ENN_DBG_PRINT("----");
        ENN_DBG_PRINT("memtype(%d:%s) load_type(%d:%s), mappingIndex(%d)\n",
                        info.mem_type, get_paramtype_str(info.mem_type).c_str(),
                        info.load_type, get_loadtype_str(info.load_type).c_str(),
                        info.mappingIndex);
        ENN_DBG_PRINT("tensor_order_featuremap(%d), size(%d), dataOffset(%d), offset(%d)\n",
                        info.index, info.size, info.dataOffset, info.offset);
    }
    ENN_DBG_PRINT("---------------------------------------------------------------------");
}

/* bin_addr : nnc addr
 * bin_size : ucgo size
 * fd : nnc fd
 * fd_offset : offset for ucgo on nnc */
EnnReturn DspUcgoInfo::parse_ucgo(const void* bin_addr, int32_t bin_size,
                                model_info_t *model_info,
                                int32_t fd, int32_t fd_offset) {
    const void *va_ucgo = static_cast<const char*>(bin_addr) + fd_offset;
    uint32_t size_ucgo = bin_size;
    if (va_ucgo == nullptr || size_ucgo <= sizeof(DspUcgoHeader)) {
        ENN_ERR_PRINT("invalid ucgo data!\n");
        return ENN_RET_FAILED;
    }

    copied_ucgo_.type = ION;
    copied_ucgo_.size = size_ucgo;
    uint32_t ema_ret = eden_mem_allocate_with_ion_flag(&copied_ucgo_, DSP_MEM_ION_FLAG);
    if (ema_ret != PASS) {
        ENN_ERR_PRINT_FORCE("ion buffer for packed ucgo alloc failed");
        return ENN_RET_FAILED;
    }

    /* Allocate memory for UCGO and copy. */
    fd = copied_ucgo_.ref.ion.fd;
    fd_offset = 0;
    memcpy(reinterpret_cast<void*>(copied_ucgo_.ref.ion.buf), va_ucgo, size_ucgo);

    // Parse ucgo header
    const UnifiedDspCgo_t *base_ucgo = reinterpret_cast<const UnifiedDspCgo_t *>(va_ucgo);
    const DspUcgoHeader *ptr_ucgo_header = reinterpret_cast<const DspUcgoHeader *>(&base_ucgo->header);
    uint32_t size_total_meta = sizeof(UnifiedDspCgo_t) + sizeof(DspMemInfo) * (ptr_ucgo_header->numList);
    const uint8_t *base_load_data = reinterpret_cast<const uint8_t *>(base_ucgo) + size_total_meta;

    int num_param = 0;
    int idx_tsgd = -1;

    dump_raw_ucgo(ptr_ucgo_header);

    std::vector<std::string> dsp_kernel_list;
    DspMemType_e tsgd_type = EMPTY;
    for (int i = 0; i < ptr_ucgo_header->numList; i++) {
        const DspMemInfo *ptrInfo = reinterpret_cast<const DspMemInfo *>(&ptr_ucgo_header->info[i]);

        /* Nice to have : TODO(mj.kim010, TBD) Verify types */
        DspMemType_e param_type = ptrInfo->mem_type;
        DspLoadType_e load_type = ptrInfo->load_type;

        if (param_type == DSP_GRAPH_BIN || param_type == CMDQ_BIN) {
            idx_tsgd = i;  // save tsgd index
            tsgd_type = param_type;  // Used for KERNEL_BIN parsing
            continue;
        }
        // push kernelbin to vector
        if (param_type == KERNEL_BIN_STR) {
            const char *addr_kernelbin = reinterpret_cast<const char *>(base_load_data) + ptrInfo->dataOffset;
            std::string kpath_str = addr_kernelbin;
            dsp_kernel_list.push_back(kpath_str);
            if (tsgd_type == CMDQ_BIN) {
                ENN_INFO_PRINT_FORCE("CMDQ_BIN needs KERNEL_BIN_STR as parameter.\n");
            }
            else if (tsgd_type == DSP_GRAPH_BIN) {
                ENN_INFO_PRINT_FORCE("DSP_GRAPH_BIN remove KERNEL_BIN_STR from param table.\n");
                continue;
            }
            else {
                ENN_ERR_PRINT_FORCE("TSGD type is not identified.(%d)\n", tsgd_type);
                return ENN_RET_FAILED;
            }
        }

        int32_t ionFd = VALUE_FD_INIT;
        uint32_t offset = 0;
        if (load_type == ALLOC_LOAD) {
            ionFd = fd;
            offset = fd_offset + size_total_meta + ptrInfo->dataOffset;

            if (param_type == TEMP && is_lbl_propery_set()) {
                int32_t fd_im_buffer = VALUE_FD_INIT;
                const uint8_t *src_va = base_load_data + ptrInfo->dataOffset; // data from UCGO
                /* Allocate new buffer and replace existing buffer. Output: allocated fd. */
                if (reserve_lbl_intermediate_buffer(src_va, ptrInfo->size, fd_im_buffer)) {
                    ENN_ERR_PRINT_FORCE("Fail to initialize intermediated buffer\n");
                    return ENN_RET_FAILED;
                }
                ionFd = fd_im_buffer;
                offset = 0;
                /* First dump will be used as reference data. */
                dump_lbl_intermediate_buffer("ref_");
            }

            ENN_DBG_PRINT("param[%d] param_type(0x%x:%s) load_type(%s) fd(%d) load_offset:%d\n", i, param_type,
                            get_paramtype_str(param_type).c_str(),
                            get_loadtype_str(load_type).c_str(),
                            ionFd, offset);
            /* nice to have : TODO(mj.kim010, TBD) : Use propery for dump */
#ifdef DSP_PARAM_DUMP
            export_param_to_file(std::to_string(i), (uint8_t*)base_ucgo + offset, ptrInfo->size);
#endif
        } else if (load_type == ALLOC || load_type == ALLOC_ZERO) {
            // allocate new mem
            eden_memory_t paramBuffer;
            paramBuffer.type = ION;
            paramBuffer.size = ptrInfo->size;

            uint32_t ema_ret = eden_mem_allocate_with_ion_flag(&paramBuffer, DSP_MEM_ION_FLAG);
            if (ema_ret != PASS) {
                ENN_ERR_PRINT_FORCE("ion buffer alloc failed");
                return ENN_RET_FAILED;
            }
            if (load_type == ALLOC_ZERO) {
                memset(reinterpret_cast<void*>(paramBuffer.ref.ion.buf), 0, paramBuffer.size);
            }
            ionFd = paramBuffer.ref.ion.fd;
            ucgo_buffer_map_[ionFd] = paramBuffer;
        } else if (load_type == IMPORT) {
            // find whc data and set
            int32_t idx_nnc_tensor = -1;
            shape_t *shape = nullptr;
            std::map<uint32_t, uint32_t> *tensor_mapping_idx_map = nullptr;
            // int *featuremap_count;
            if (param_type == INPUT) {
                idx_nnc_tensor = model_info->bin_in_index[ptrInfo->index];
                shape = &model_info->bin_in_shape[ptrInfo->index];
                tensor_mapping_idx_map = &nnc_input_map_;
                ENN_DBG_PRINT("[%d] get input_shape. index(%d) shape(%s)\n",
                                            i, ptrInfo->index,
                                            shape->get_string().c_str());
            } else if (param_type == OUTPUT) {
                idx_nnc_tensor = model_info->bin_out_index[ptrInfo->index];
                shape = &model_info->bin_out_shape[ptrInfo->index];
                tensor_mapping_idx_map = &nnc_output_map_;
                ENN_DBG_PRINT("[%d] get output_shape. index(%d) shape(%s)\n",
                                            i, ptrInfo->index,
                                            shape->get_string().c_str());
            } else {
                ENN_ERR_PRINT_FORCE("ERR: Wrong type(0x%x:%s)(0x%x:%s)\n",
                            param_type, get_paramtype_str(param_type).c_str(),
                            load_type, get_loadtype_str(load_type).c_str());
                return ENN_RET_FAILED;
            }

            if (shape->get_size() != ptrInfo->size) {
                ENN_ERR_PRINT_FORCE("no matched nnc param! [%d] shape(%d=%s) vs size(%d)\n",
                                            i, shape->get_size(),
                                            shape->get_string().c_str(), ptrInfo->size);
                return ENN_RET_FAILED;
            }
            ENN_DBG_PRINT("[0x%x:%s] nnc_map[%d]=mappingIndex(%d)",
                            param_type, get_paramtype_str(param_type).c_str(),
                            idx_nnc_tensor, ptrInfo->mappingIndex);
            /* Nice to have : TODO(mj.kim010, TBD) : Not use tensor index. order is enough. */
            (*tensor_mapping_idx_map)[idx_nnc_tensor] = ptrInfo->mappingIndex;
        } else { // BYPASS
            /* Nothing to do for now. */
        }

        ofi_v4_param_t v4_param;
        ofi_common_mem_type_e mem_type = OFI_MEM_ION;
        /* BYPASS means no memory will be passed through UD,DD.
         * NPU FW will give mem info to DSP FW. */
        if (load_type == BYPASS) {
            mem_type = OFI_MEM_NONE;
            if (param_type == INPUT) {
                model_info->input_count = 0;
                ENN_INFO_PRINT_FORCE("Remove input for DSP OP since UCGO has input BYPASS.\n");
            }
            else if (param_type == OUTPUT) {
                model_info->output_count = 0;
                ENN_INFO_PRINT_FORCE("Remove output for DSP OP since UCGO has output BYPASS.\n");
            }
        }

        DspDdParam::fill_param(v4_param, ptrInfo->mappingIndex, mem_type, param_type, ptrInfo->size, ionFd, offset);

        ENN_DBG_PRINT("----- UCGO parsed data : param_idx[%d] ------\n", v4_param.idx.param_index);
        ENN_DBG_PRINT("param_type(0x%x:%s) size(%d) offset(%d) fd(%d) iova(%d)\n",
            v4_param.param_type, get_paramtype_str((DspMemType_e)v4_param.param_type).c_str(),
            v4_param.param_mem.size, v4_param.param_mem.offset,
            v4_param.param_mem.get_addr.mem.fd, v4_param.param_mem.get_addr.mem.iova);
        ENN_DBG_PRINT("----------------------------------------------\n");

        ucgo_param_vector_.push_back(v4_param);
        num_param++;
    }

    /* Fill TSGD param. */
    if (idx_tsgd == -1) {
        ENN_ERR_PRINT_FORCE("cannot find tsgd data in UCGO.\n");
        return ENN_RET_FAILED;
    }
    const DspMemInfo *tsgd_info = reinterpret_cast<const DspMemInfo *>(&ptr_ucgo_header->info[idx_tsgd]);
    tsgd_info_.type = ION;
    int size_tsgd = tsgd_info_.size = tsgd_info->size;
    ema_ret = eden_mem_allocate_with_ion_flag(&tsgd_info_, DSP_MEM_ION_FLAG);
    if (ema_ret != PASS) {
        ENN_ERR_PRINT_FORCE("ion buffer for tsgd alloc failed");
        return ENN_RET_FAILED;
    }
    memcpy(reinterpret_cast<void*>(tsgd_info_.ref.ion.buf),
           reinterpret_cast<const void*>(base_load_data + tsgd_info->dataOffset),
           size_tsgd);
    DspDdParam::fill_param(tsgd_param_, 0, OFI_MEM_ION, tsgd_info->mem_type, size_tsgd, tsgd_info_.ref.ion.fd, 0);
#ifdef DSP_PARAM_DUMP
    export_param_to_file(std::string("tsgd"), reinterpret_cast<void*>(tsgd_info_.ref.ion.buf), size_tsgd);
#endif

    /* Parse DSP kernel info to use as S_PARAM */
    parse_kernel_bin(dsp_kernel_list);

    /* Fill all DSP params entries. */
    if (fill_param_table(model_info, num_param, (uint16_t)ptr_ucgo_header->unique_id)) {
        return ENN_RET_FAILED;
    }
    return ENN_RET_SUCCESS;
}

EnnReturn DspUcgoInfo::fill_param_table(model_info_t *model_info, int num_param, uint16_t ucgo_uid) {
    // create load struct format
    int size_load_graph_info = sizeof(ofi_v4_load_graph_info) +
                                sizeof(ofi_v4_param_t) * (num_param + 1);

    loadgraph_info_.type = ION;
    loadgraph_info_.size = size_load_graph_info;
    if (eden_mem_allocate_with_ion_flag(&loadgraph_info_, DSP_MEM_ION_FLAG)) {
        ENN_ERR_PRINT_FORCE("ion buffer for load ucgo alloc failed");
        return ENN_RET_FAILED;
    }

    /* param table in here includes TSGD as [0] */
    ofi_v4_load_graph_info *v4_graph_info =
        reinterpret_cast<ofi_v4_load_graph_info*>(loadgraph_info_.ref.ion.buf);

    /* [0] is always TSGD : CMDQ_BIN or DSP_GRAPH_BIN */
    int i_not_populated = 1;
    v4_graph_info->param_list[0] = tsgd_param_;

    // add mem data
    for (int i = 0; i < ucgo_param_vector_.size(); i++) {
        /* Filter out BaaW memories and KernelStr */
        if (!is_mapping_index_populated(ucgo_param_vector_[i]))
            continue;
        uint32_t p_idx = ucgo_param_vector_[i].idx.param_index;
        v4_graph_info->param_list[p_idx+1] = ucgo_param_vector_[i];
        i_not_populated++;
        ENN_DBG_PRINT("Init Normal execInfo[%2d+1] with ucgo_param_map[%2d]\n", p_idx, i);
    }

    /* BaaW params and KERNEL_BIN are stacked behind normal params.
       Baaw param : mappingIndex==-1 and !TSGD */
    for (int i = 0; i < ucgo_param_vector_.size(); i++) {
        if (!is_mapping_index_populated(ucgo_param_vector_[i])) {
            ENN_DBG_PRINT("Init non-populated execInfo[%2d] with ucgo_param_map[%2d]\n", i_not_populated, i);
            v4_graph_info->param_list[i_not_populated++] = ucgo_param_vector_[i];
        }
    }

    /* Can't use -1 itself because -1 is for CV. so set it 0. means not using uid. */
    if (ucgo_uid == UCGO_UID_INVALID)
        ucgo_uid = 0;

    uint32_t gid = create_dsp_global_id(model_info->id, ucgo_uid);
    set_global_id(gid);

    ofi_v4_load_graph_info &graphInfo = *v4_graph_info;
    DspDdLoadGraphInfo::set_global_id(graphInfo, gid);
    DspDdLoadGraphInfo::set_num_tsgd(graphInfo, NUM_TSGD);
    DspDdLoadGraphInfo::set_num_param(graphInfo, num_param);
    DspDdLoadGraphInfo::set_num_kernel(graphInfo, kernel_name_count_);

    /* update model_info (dsp specific datas) */
    model_info->set_model_addr(reinterpret_cast<void*>(loadgraph_info_.ref.ion.buf));
    model_info->set_model_size(loadgraph_info_.size);
    model_info->set_kernel_name(kernel_name_);
    model_info->set_kernel_name_size(kernel_name_size_);
    model_info->set_kernel_name_count(kernel_name_count_);

    /* Create execute info struct
     * Nice to have: TODO(geunwon.lee, TBD): only update param will be used */
    exec_info_.type = ION;
    exec_info_.size = sizeof(ofi_v4_execute_msg_info_t) + sizeof(ofi_v4_param_t) * num_param;
    if (eden_mem_allocate_with_ion_flag(&exec_info_, DSP_MEM_ION_FLAG)) {
        ENN_ERR_PRINT_FORCE("exec_info alloc failed. num_param(%d)\n", num_param);
        return ENN_RET_FAILED;
    }

    model_info->set_exec_msg_size(exec_info_.size);

    ofi_v4_execute_msg_info_t *v4_exec_info =
         reinterpret_cast<ofi_v4_execute_msg_info_t *>(exec_info_.ref.ion.buf);
    for (int i = 0; i < num_param; i++) {
        ofi_v4_param_t* executeParam = &v4_exec_info->param_list[i];
        /* Copy loadGraph info to execute_info. */
        memcpy(executeParam, &v4_graph_info->param_list[i+1], sizeof(ofi_v4_param_t));
        /* Reset FD to make param empty. */
        DspDdParam::fill_param_fd_info(*executeParam, VALUE_FD_INIT);
    }

    ofi_v4_execute_msg_info_t &execInfo = *v4_exec_info;
    DspDdExecInfo::set_num_updated_param(execInfo, num_param);
    DspDdExecInfo::set_global_id(execInfo, global_id_);

    for (auto input : nnc_input_map_) {
        auto mapping_idx = input.second;
        DspDdParam::fill_param(v4_exec_info->param_list[mapping_idx],
                                mapping_idx, OFI_MEM_ION, INPUT, -1, VALUE_FD_INIT, 0);
    }
    for (auto output : nnc_output_map_) {
        auto mapping_idx = output.second;
        DspDdParam::fill_param(v4_exec_info->param_list[mapping_idx],
                                mapping_idx, OFI_MEM_ION, OUTPUT, -1, VALUE_FD_INIT, 0);
    }

    return ENN_RET_SUCCESS;
}

EnnReturn DspUcgoInfo::update_inout_param(ofi_v4_execute_msg_info *ddExecInfo,
                                            const std::shared_ptr<eden_memory_t> ema,
                                            const uint32_t *bin_idx_array,
                                            uint32_t count,
                                            std::map<uint32_t, uint32_t> &tensor_mapping_idx_map) {
    ENN_DBG_PRINT("n_param: %d, count_inout:%d", ddExecInfo->n_update_param, count);

    for (int i = 0; i < count; i++) {
        uint32_t nnc_map_idx = bin_idx_array[i];
        ENN_DBG_PRINT("Try to find by in_cnt_idx(%d),nnc_map_idx(%d)\n", i, nnc_map_idx);
        auto itr = tensor_mapping_idx_map.find(nnc_map_idx);
        if (itr == tensor_mapping_idx_map.end()) {
            ENN_ERR_PRINT_FORCE("ERR: No matching operand. in_cnt_idx(%d) nnc_map_idx(%d)\n", i, nnc_map_idx);
            return ENN_RET_FAILED;
        }
        uint32_t mapping_idx = itr->second;
        ddExecInfo->param_list[mapping_idx].param_mem.size = ema.get()[i].size;
        ddExecInfo->param_list[mapping_idx].param_mem.get_addr.mem.fd = ema.get()[i].ref.ion.fd;
        ENN_DBG_PRINT("in map[%d] dspIdx[%d] size(%d), fd(%d)\n",
                        i, mapping_idx,
                        ddExecInfo->param_list[mapping_idx].param_mem.size,
                        ddExecInfo->param_list[mapping_idx].param_mem.get_addr.mem.fd);
    }
    return ENN_RET_SUCCESS;
}

/* Per Execute(or Prepare) */
EnnReturn DspUcgoInfo::update_exec_info(ofi_v4_execute_msg_info &ddExecInfo, req_info_t *req_info) {
    EnnReturn ret = update_inout_param(&ddExecInfo,
                            req_info->inputs,
                            req_info->model_info->bin_in_index,
                            req_info->model_info->input_count,
                            nnc_input_map_);
    if (ret)
        return ret;

    ret = update_inout_param(&ddExecInfo,
                            req_info->outputs,
                            req_info->model_info->bin_out_index,
                            req_info->model_info->output_count,
                            nnc_output_map_);
    return ret;
};


/* DspCgoInfo */


int32_t DspCgoInfo::paramNameToParamType(std::string name) {
    int32_t ret = EMPTY;
    if(name.find("input_") != std::string::npos) {
        ret = INPUT;
    } else if (name.find("output_") != std::string::npos) {
        ret = OUTPUT;
    } else if (name.find("Shape Infos Buffer") != std::string::npos) {
        ret = CUSTOM;
    } else if (name.find("TEMP") != std::string::npos) {
        ret = TEMP;
    } else {
        ret = SCALAR;
    }

    return ret;
}

EnnReturn DspCgoInfo::parse_tsgd(const void *bin_addr, int32_t bin_size,
                        int32_t fd, int32_t fd_offset, std::vector<std::string> libnames,
                        uint64_t op_list_id) {
    global_id_ =  create_dsp_global_id(op_list_id);
  /* Nice to have : TODO(mj.kim010, TBD) : Use android property to turn on/off RT parser. */
#ifdef DSP_UD_CGO_RT_PARSER
    int new_tsgd_size = get_size_of_jib_pool(const_cast<void *>(bin_addr));
    ENN_DBG_PRINT("TSGD(DSP_GRAPH) size:%d --> Parsed Size:%d\n", bin_size, new_tsgd_size);

    std::vector<struct KernelBinInfo> kernelList_;
    for (auto &dsplib : libnames) {
        std::string kPath = dsplib.c_str();
        struct KernelBinInfo kbinInfo(kPath);
        kernelList_.push_back(kbinInfo);
    }

    uint8_t heap_new_tsgd[new_tsgd_size];
    memset(&heap_new_tsgd, 0, new_tsgd_size);
    int ret = parse_dsp_fbs(reinterpret_cast<char*>(&heap_new_tsgd),
                const_cast<void *>(bin_addr), kernelList_, global_id_);
    if (ret) {
        ENN_ERR_PRINT_FORCE("Fail to parse jib\n");
        return ENN_RET_FAILED;
    }
    /* Update src memory info. */
    const void *src_addr = &heap_new_tsgd;
    bin_size = new_tsgd_size;
#else
  const void *src_addr = bin_addr;
#endif
    // Objective: Parse TSGD and update tsgd_info_
    // Currently fw parser is used
    tsgd_info_.type = ION;
    tsgd_info_.size = bin_size;
    uint32_t emaRet = eden_mem_allocate_with_ion_flag(&tsgd_info_, DSP_MEM_ION_FLAG);
    if (emaRet != PASS) {
        ENN_ERR_PRINT_FORCE("ion buffer for tsgd alloc failed");
        return ENN_RET_FAILED;
    }

    memcpy(reinterpret_cast<void*>(tsgd_info_.ref.ion.buf), src_addr, bin_size);
    ENN_DBG_PRINT("copied tsgd from cgo binary format(addr %p size %d fd %d offset %d)",
                    bin_addr, bin_size, fd, fd_offset);
    /* debug file save */
#ifdef DSP_PARAM_DUMP
    export_param_to_file(std::string("tsgd"), src_addr, bin_size);
#endif
    return ENN_RET_SUCCESS;
}

EnnReturn DspCgoInfo::create_base_info(uint32_t buf_cnt, model_info_t* model_info) {
    // create load struct format
    int sizeLoadInfo = sizeof(ofi_v4_load_graph_info) + sizeof(ofi_v4_param_t) * (buf_cnt + 1);

    loadgraph_info_.size = sizeLoadInfo;
    loadgraph_info_.type = ION;
    uint32_t emaRet = eden_mem_allocate_with_ion_flag(&loadgraph_info_, DSP_MEM_ION_FLAG);
    if (emaRet != PASS) {
        ENN_ERR_PRINT_FORCE("ion buffer for load ucgo alloc failed");
        return ENN_RET_FAILED;
    }

    ENN_DBG_PRINT("create base load info struct(size: %zu)", loadgraph_info_.size);

    ofi_v4_load_graph_info *ucgoInfoHeader =
        reinterpret_cast<ofi_v4_load_graph_info*>(loadgraph_info_.ref.ion.buf);

    /* tsgd */
    DspDdParam::fill_param(tsgd_param_, 0, OFI_MEM_ION, DSP_GRAPH_BIN, tsgd_info_.size, tsgd_info_.ref.ion.fd, 0);

    ucgoInfoHeader->param_list[0] = tsgd_param_;

    ENN_DBG_PRINT("tsgd for cgo is set (type: %d size: %d fd: %d)", tsgd_param_.param_type,
                  tsgd_param_.param_mem.size, tsgd_param_.param_mem.get_addr.mem.fd);

    /* parameter */
    /* Nice to have: TODO(mj.kim010, TBD) : CV is not using offset. Check if OK. */
    for (int i = 0; i < buf_cnt; i++) {
        ENN_DBG_PRINT("Create dsp param[%d+1]\n", i + 1);
        DspDdParam::fill_param(ucgoInfoHeader->param_list[i + 1], 0, OFI_MEM_ION, EMPTY, 0, VALUE_FD_INIT, 0);
    }

    ofi_v4_load_graph_info &graphInfo = *ucgoInfoHeader;
    DspDdLoadGraphInfo::set_global_id(graphInfo, global_id_);
    DspDdLoadGraphInfo::set_num_tsgd(graphInfo, NUM_TSGD);
    DspDdLoadGraphInfo::set_num_param(graphInfo, buf_cnt);
    DspDdLoadGraphInfo::set_num_kernel(graphInfo, kernel_name_count_);

    ENN_DBG_PRINT("load param is initialized(param count: %d kernel count: %d)",
                  DspDdLoadGraphInfo::get_num_param(graphInfo),
                  DspDdLoadGraphInfo::get_num_kernel(graphInfo));

    // update model_info
    model_info->set_model_addr(reinterpret_cast<void*>(loadgraph_info_.ref.ion.buf));
    model_info->set_model_size(loadgraph_info_.size);
    model_info->set_kernel_name(kernel_name_);
    model_info->set_kernel_name_size(kernel_name_size_);
    model_info->set_kernel_name_count(kernel_name_count_);

    ENN_DBG_PRINT("cgo model info updated(addr: %p size: %d kernel name size: %d",
                  model_info->model_addr, model_info->model_size, model_info->kernel_name_size);

    // create execute info struct
    // Nice to have: TODO(geunwon.lee, TBD): only update param will be used
    exec_info_.size = sizeof(ofi_v4_execute_msg_info_t) + sizeof(ofi_v4_param_t) * buf_cnt;
    exec_info_.type = ION;

    emaRet = eden_mem_allocate_with_ion_flag(&exec_info_, DSP_MEM_ION_FLAG);
    if (emaRet != PASS) {
        ENN_ERR_PRINT_FORCE("ion buffer for tsgd alloc failed");
        return ENN_RET_FAILED;
    }

    ENN_DBG_PRINT("create base exec info struct(size: %zu)", exec_info_.size);

    ofi_v4_execute_msg_info_t &ddExecInfo =
         *(reinterpret_cast<ofi_v4_execute_msg_info_t *>(exec_info_.ref.ion.buf));
    for (int i = 0; i < buf_cnt; i++) {
        DspDdParam::initialize_param(ddExecInfo.param_list[i]);
    }

    model_info->set_exec_msg_size(exec_info_.size);

    DspDdExecInfo::set_num_updated_param(ddExecInfo, buf_cnt);
    DspDdExecInfo::set_global_id(ddExecInfo, global_id_);

    ENN_DBG_PRINT("exec param is initialized(param count: %d)",
                  DspDdExecInfo::get_num_updated_param(ddExecInfo));

    total_param_count_ = buf_cnt;

    return ENN_RET_SUCCESS;
}

/* OpenModel(loadGraph) time */
EnnReturn DspCgoInfo::update_load_param(std::string name, uint32_t index, shape_t & shape) {
    ofi_v4_load_graph_info *ucgoInfoHeader =
        reinterpret_cast<ofi_v4_load_graph_info*>(loadgraph_info_.ref.ion.buf);
    if (index > ucgoInfoHeader->n_param) {
        ENN_ERR_PRINT("load_param index is invalid");
        return ENN_RET_FAILED;
    }

    uint32_t param_type = paramNameToParamType(name);
    ENN_DBG_PRINT("update load param(name: %s, idx: %d, type: %d)", name.c_str(), index, param_type);
    ENN_DBG_PRINT("shape : (%s)\n", shape.get_string().c_str());

    /* Fill load_graph param info. */
    /* Nice to have: TODO(mj.kim010, TBD) : CV is not using offset. Check if OK. */
    ofi_v4_param *loadParamPtr = &ucgoInfoHeader->param_list[index + 1];
    DspDdParam::fill_param(*loadParamPtr, index, OFI_MEM_ION, param_type, shape.get_size(), VALUE_FD_INIT, 0);

    /* Fill exec_info with load_graph values. */
    ofi_v4_execute_msg_info_t *dalExecuteInfo =
         reinterpret_cast<ofi_v4_execute_msg_info_t *>(exec_info_.ref.ion.buf);
    dalExecuteInfo->param_list[index] = *loadParamPtr;

    return ENN_RET_SUCCESS;
}

/* Prepare/Execute time : Update only FD and size. */
EnnReturn DspCgoInfo::update_exec_param(ofi_v4_execute_msg_info_t &ddExecuteInfo,
                                            uint32_t index, uint32_t size, int32_t fd) {
    DspDdParam::fill_param_memory_info(ddExecuteInfo.param_list[index], fd, size);
    ENN_DBG_PRINT("Update cgo execute param[%d] size(%d), fd(%d)\n", index, size, fd);

    return ENN_RET_SUCCESS;
};
