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

/**
 * @file    dsp_bin_info.h
 * @brief   This is class for unpack ucgo data
 * @details Supports ucgo data in NNC
            Need to update for ucgo data from CGO
 */
#ifndef USERDRIVER_DSP_DSP_UCGO_INFO_H_
#define USERDRIVER_DSP_DSP_UCGO_INFO_H_

#include <unordered_map>
#include <vector>
#include <string.h>

#include "userdriver/common/eden_osal/eden_memory.h"      // eden_memory_t
#include "common/enn_debug.h"
#include "dsp_common_struct.h"                              // ofi_v4_param_t
#include "link_vs4l.h"


/* Use non-cache ION memory */
constexpr int32_t DSP_MEM_ION_FLAG = 0;
/* Always 1. synced with DD/FW */
constexpr int32_t NUM_TSGD = 1;

class DspDdExecInfo {
  public:
    /* static func used to fill ofi_v4_param_t which are passed to D/D directly. */
    static inline void set_num_updated_param(ofi_v4_execute_msg_info_t &info, int32_t count) {
        info.n_update_param = count;
    }
    static inline void set_global_id(ofi_v4_execute_msg_info_t &info, uint32_t gid) {
        info.global_id = gid;
    }
    static inline int32_t get_num_updated_param(ofi_v4_execute_msg_info_t &info) {
        return info.n_update_param;
    }
};

class DspDdLoadGraphInfo {
  public:
    /* static func used to fill ofi_v4_param_t which are passed to D/D directly. */
    static inline void set_num_tsgd(ofi_v4_load_graph_info &info, int32_t count) {
        info.n_tsgd = count;
    }
    static inline void set_num_param(ofi_v4_load_graph_info &info, int32_t count) {
        info.n_param = count;
    }
    static inline void set_num_kernel(ofi_v4_load_graph_info &info, int32_t count) {
        info.n_kernel = count;
    }
    static inline void set_global_id(ofi_v4_load_graph_info &info, uint32_t gid) {
        info.global_id = gid;
    }
    static inline int32_t get_num_param(ofi_v4_load_graph_info &info) {
        return info.n_param;
    }
    static inline int32_t get_num_kernel(ofi_v4_load_graph_info &info) {
        return info.n_kernel;
    }
};

class DspDdParam {
  public:
    /* static func used to fill ofi_v4_param_t which are passed to D/D directly. */
    static inline void fill_param_fd_info(ofi_v4_param_t &param, int32_t fd) {
        param.param_mem.get_addr.mem.fd = fd;
        param.param_mem.get_addr.mem.iova = VALUE_IOVA_INIT;
    }
    static inline void fill_param_fd_info(ofi_v4_param_t &param, int32_t fd, int32_t offset) {
        fill_param_fd_info(param, fd);
        param.param_mem.offset = offset;
    }
    static inline void fill_param_memory_info(ofi_v4_param_t &param, int32_t fd, uint32_t size) {
        fill_param_fd_info(param, fd);
        param.param_mem.size = size;
    }
    static inline void fill_param_memory_info(ofi_v4_param_t &param, int32_t fd, uint32_t size, int32_t offset) {
        fill_param_fd_info(param, fd, offset);
        param.param_mem.size = size;
    }

    static inline void initialize_param(ofi_v4_param_t &param) {
        param.param_type = EMPTY;
        param.idx.param_index = MAPPING_INDEX_INVALID;
    }

    static inline void fill_param(ofi_v4_param_t &param, uint32_t idx,
                    ofi_common_mem_type_e mem_type,
                    uint32_t param_type, uint32_t size, int32_t fd, int32_t offset) {
        param.idx.param_index = idx;
        param.param_type = param_type;
        param.param_mem.addr_type = OFI_FD;
        param.param_mem.mem_type = mem_type;
        param.param_mem.mem_attr = OFI_UNKNOWN_CACHEABLE;
        param.param_mem.is_mandatory = true;
        fill_param_memory_info(param, fd, size, offset);
        ENN_DBG_PRINT("Fill dsp_param[%u] type[%u] size[%u] fd[0x%x]\n",
                            param.idx.param_index, param.param_type,
                            param.param_mem.size, param.param_mem.get_addr.mem.fd);
        return;
    }
    ofi_v4_param_t dspParam_;
};

/* Common part of UCGO/CGO. */
class DspBinInfo {
public:
    DspBinInfo(bool cgo_flag) :
                kernel_name_(nullptr), kernel_name_size_(0), kernel_name_count_(0), global_id_(0), is_cgo_(cgo_flag) {
        memset(&loadgraph_info_, 0, sizeof(loadgraph_info_));
        memset(&exec_info_, 0, sizeof(exec_info_));
        memset(&tsgd_info_, 0, sizeof(tsgd_info_));
        memset(&tsgd_param_, 0, sizeof(tsgd_param_));
        loadgraph_info_.type = ION;
        exec_info_.type = ION;
        tsgd_info_.type = ION;
    }
    ~DspBinInfo() {
        if (loadgraph_info_.ref.ion.fd != 0) {
            eden_mem_free(&loadgraph_info_);
        }
        if (exec_info_.ref.ion.fd != 0) {
            eden_mem_free(&exec_info_);
        }
        if (tsgd_info_.ref.ion.fd != 0) {
            eden_mem_free(&tsgd_info_);
        }
        delete[] kernel_name_;
        kernel_name_size_ = 0;
        kernel_name_ = nullptr;
    }
    DspBinInfo(DspBinInfo const&) = delete;
    DspBinInfo& operator=(DspBinInfo const&) = delete;

    EnnReturn parse_kernel_bin(std::vector<std::string>& lib_names);
    const eden_memory_t* get_load_info(void) const { return &loadgraph_info_; }
    const eden_memory_t* get_exec_info(void) const { return &exec_info_; }
    const uint8_t* get_kernel_name(void) const { return kernel_name_; }
    int32_t get_kernel_name_size(void) const { return kernel_name_size_; }
    int32_t get_kernel_name_count(void) const { return kernel_name_count_; }
    const std::string get_paramtype_str(enum DspMemType_e mem_type);
    void set_global_id(uint32_t gid) { global_id_ = gid; }
    bool is_cgo(void) { return is_cgo_;}
protected:
    void export_param_to_file(std::string idx_str, const void *dump_va, int size);
    uint32_t create_dsp_global_id(uint64_t op_list_id, uint16_t ucgo_uid = UCGO_UID_INVALID);
    eden_memory_t loadgraph_info_;
    eden_memory_t exec_info_;
    eden_memory_t tsgd_info_;
    ofi_v4_param_t tsgd_param_;
    uint8_t* kernel_name_;
    int32_t kernel_name_size_;
    int32_t kernel_name_count_;
    uint32_t global_id_;
    bool is_cgo_;
};


class DspUcgoInfo : public DspBinInfo {
public:
    DspUcgoInfo() : DspBinInfo(false), // derived
                intermediate_buffer_key_(VALUE_FD_INIT) { // member
        memset(&copied_ucgo_, 0, sizeof(copied_ucgo_));
        copied_ucgo_.type = ION;
    }
    ~DspUcgoInfo() {
        if (copied_ucgo_.ref.ion.fd != 0)
            eden_mem_free(&copied_ucgo_);
        for (auto &buffer : ucgo_buffer_map_) {
            if (buffer.second.ref.ion.fd != 0) {
                eden_mem_free(&buffer.second);
            }
        }
    }

    EnnReturn parse_ucgo(const void *bin_addr, int32_t bin_size,
                        model_info_t *model_info,
                        int32_t fd, int32_t fd_offset);
    EnnReturn update_exec_info(ofi_v4_execute_msg_info &ddExecInfo, req_info_t *req_info);
    EnnReturn generate(int32_t id, void* ucgo_addr, int32_t ucgo_size, int32_t shared = -1);
    /* layer-by-layer support. */
    void dump_lbl_intermediate_buffer(std::string file_prefix = "");

private:
    EnnReturn fill_param_table(model_info_t *model_info, int num_param, uint16_t ucgo_uid);
    void dump_raw_ucgo(const DspUcgoHeader *ptr_ucgo_header);
    bool is_mapping_index_populated(ofi_v4_param_t &prm);
    EnnReturn update_inout_param(ofi_v4_execute_msg_info *ddExecInfo,
                const std::shared_ptr<eden_memory_t> ema,
                const uint32_t *bin_idx_array,
                uint32_t count,
                std::map<uint32_t, uint32_t> &tensor_mapping_idx_map);
    std::string get_loadtype_str(enum DspLoadType_e load_type);
    /* layer-by-layer support. */
    bool is_lbl_propery_set();
    EnnReturn reserve_lbl_intermediate_buffer(const void *va, int32_t size, int32_t &fd_im);

    eden_memory_t copied_ucgo_;
    std::map<uint32_t, eden_memory_t> ucgo_buffer_map_;
    std::vector<ofi_v4_param_t> ucgo_param_vector_;
    std::map<uint32_t, uint32_t> nnc_input_map_;
    std::map<uint32_t, uint32_t> nnc_output_map_;
    /* layer-by-layer support. */
    int32_t intermediate_buffer_key_; // key for ucgo_buffer_map_
    const std::string DEBUG_PROPERTY_DSP_LBL {"vendor.enn.dsp.lbl"};
    const std::string SAVE_FILEPATH_DSP_LBL {"/data/vendor/enn/"};
    const std::string SAVE_FILENAME_DSP_LBL {"dsp_intermediate.bin"};
};


class DspCgoInfo : public DspBinInfo {
public:
    DspCgoInfo() : DspBinInfo(true), // derived
                    total_param_count_(0) { // member
    }
    ~DspCgoInfo() {}

    EnnReturn parse_tsgd(const void *bin_addr, int32_t bin_size,
                        int32_t fd, int32_t fd_offset, std::vector<std::string> lib_names,
                        uint64_t op_list_id);
    EnnReturn create_base_info(uint32_t buf_cnt, model_info_t* model_info);
    EnnReturn update_load_param(std::string name, uint32_t index, shape_t & shape);
    EnnReturn update_exec_param(ofi_v4_execute_msg_info_t &ddExecuteInfo,
                                                uint32_t index, uint32_t size, int32_t fd);
    uint32_t get_total_param_count(void) { return total_param_count_; }
private:
    int32_t paramNameToParamType(std::string name);
    void fill_param(ofi_v4_param_t &param,
                uint32_t idx, uint32_t param_type, uint32_t size, uint32_t fd);
    uint32_t total_param_count_;
};

#endif  // USERDRIVER_DSP_DSP_UCGO_INFO_H_
