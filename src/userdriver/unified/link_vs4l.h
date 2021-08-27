/**
 * Copyright (C) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed,
 * transmitted, transcribed, stored in a retrieval system or translated into any human or
 * computer language in any form by any means, electronic, mechanical, manual or
 * otherwise or disclosed to third parties without the express written permission of
 * Samsung Electronics.
 */

/**
 * @file    link_vs4l.h
 * @brief   This is vs4l link.
 * @details This header defines vs4l link.
 * @version 0.3 Basic scenario support.
 */

#ifndef USERDRIVER_UNIFIED_LINK_VS4L_H__
#define USERDRIVER_UNIFIED_LINK_VS4L_H__

#include <map>
#include <atomic>
#include <setjmp.h>
#include <stdbool.h>
#include <pthread.h>
#include <sys/queue.h>

// userdriver
#include "userdriver/unified/vs4l.h"  // struct vs4l_xxx
#include "userdriver/unified/link_vs4l_config.h"
#include "userdriver/unified/drv_usr_if.h"
#include "userdriver/unified/dsp_common_struct.h"  // dsp v4 data struct
#include "userdriver/unified/error_codes.h"  // EnnReturn
#include "userdriver/common/UserDriverTypes.h"  // drv_info_t
#include "userdriver/common/eden_osal/osal_types.h"   // eden_addr_t
#include "userdriver/common/eden_osal/eden_memory.h"  // eden_memory_t
#include "common/enn_debug.h"

#define MAX_LEN_VS4L_TYPE 30

typedef uint32_t (*openFuncPtr)(void*, int32_t, int32_t);
typedef uint32_t (*executeFuncPtr)(void*, int32_t, int32_t);

using acc_req_p = struct acc_req*;
/*
 * Start of "eden_types.h"
 * These declarations replace the #include "eden_types.h"
 * These will be refactored.
 */

typedef struct __InputBufferPreference {
    int8_t enable;
    int8_t setInputAsFloat;
    __InputBufferPreference() : enable(0), setInputAsFloat(0) {}
} InputBufferPreference;

typedef struct __EdenPreference {
    HwPreference hw;
    ModePreference mode;
    InputBufferPreference inputBufferMode;  // Used by OpenModel
    __EdenPreference() : hw(ALL_HW), mode(NORMAL_MODE), inputBufferMode() {}
} EdenPreference;

typedef struct __ModelPreference {
    EdenPreference userPreference;
    NnApiType nnApiType;
    __ModelPreference() : userPreference(), nnApiType(APICOUNT) {}
} ModelPreference;

struct EdenModelOptions {
    ModelPreference modelPreference;
    uint32_t priority;
    uint32_t latency;
    uint32_t boundCore;
    // experimental. indicates number of tiles.
    // it works for a model without pre/post processing and injected allocated memory.
    uint32_t tileSize;
    int32_t presetScenarioId;
    int32_t reserved[30];
    EdenModelOptions() : modelPreference{}, priority(0), latency(0), boundCore(BOUND_NA),
        tileSize(1), presetScenarioId(PRESET_DISABLE_UPPER_BOUND), reserved{} {}
};

typedef struct __RequestPreference {
    EdenPreference userPreference;
} RequestPreference;

typedef struct __EdenRequestOptions {
    EdenPreference userPreference;
    RequestMode requestMode;
    int32_t reserved[32];
    __EdenRequestOptions() : userPreference(), requestMode(NONE), reserved{} {}
} EdenRequestOptions;

typedef struct __UpdatedOperations {
    int32_t numOfOperations;
    int32_t* operations;
} UpdatedOperations;

typedef struct __RequestOptions {
    RequestPreference requestPreference;
    UpdatedOperations updatedOperations;
    int32_t reserved[32];
} RequestOptions;
/* End of "eden_types.h" */

typedef struct _acc_model_info {
    const void* model_addr;
    int32_t model_size;
    const uint8_t* model_name;
    uint64_t id;
    uint8_t input_count;
    uint8_t output_count;
    /* Required: TODO (mj.kim010, 7/16) : apply shared_ptr */
    shape_t* bin_in_shape;
    uint32_t* bin_in_index;
    uint32_t* bin_in_bpp;
    shape_t* bin_out_shape;
    uint32_t* bin_out_index;
    uint64_t operator_list_id;
    uint64_t unified_op_id;
    int32_t priority;
    shape_t cell_align_shape;
    int32_t shared_buffer;
    uint32_t binding_ofm;
    int32_t tile_size;
    uint32_t exec_msg_size; // dsp only
    const uint8_t* kernel_name; // dsp only
    uint32_t kernel_name_size; // dsp only
    uint32_t kernel_name_count; // dsp only
    bool is_async_execute;
    uint32_t reserved[1];
    _acc_model_info() { memset(this, 0, sizeof(struct _acc_model_info)); }
    void set_model_addr(void *mdlAddr) { model_addr = mdlAddr; }
    void set_model_size(int32_t size) { model_size = size; }
    void set_kernel_name(const uint8_t* kName) { kernel_name = kName; }
    void set_kernel_name_size(uint32_t size) { kernel_name_size = size; }
    void set_kernel_name_count(uint32_t size) { kernel_name_count = size; }
    void set_exec_msg_size(uint32_t size) { exec_msg_size = size; }
    uint32_t get_exec_msg_size() const { return exec_msg_size; }
} model_info_t;

typedef struct _acc_req_info {
    const model_info_t* model_info;
    std::shared_ptr<eden_memory_t> inputs;
    std::shared_ptr<eden_memory_t> outputs;
    eden_addr_t requestId;
    uint64_t operator_list_id;
    eden_memory_t* execute_info;  // dsp only
    void set_exec_info(eden_memory_t *execInfo) { execute_info = execInfo; };
    eden_memory_t* get_exec_info() { return execute_info; }
} req_info_t;

struct acc_req {
    uint32_t bin_id;
    int state;
    uint32_t time_check;
    int32_t ret_code;
    const req_info_t* req_info;
    EdenRequestOptions options;
};

struct bin_data {
    /* Required : TODO(mj.kim010, 7/16) : Use destructor
     *              to free heap in vs4l data struct. ex)in_vs4l_ctl_array */
    bin_data() { memset(this, 0, sizeof(struct bin_data));}
    ~bin_data() {}
    uint64_t model_info_id;
    eden_memory_t bin_mem;
    uint32_t unique_id;
    int32_t device_fd;
    uint32_t vs4l_index;  //*< Using this VS4L buffer index when requests enqueued
    uint32_t in_fmap_count;
    uint32_t out_fmap_count;
    std::shared_ptr<acc_req_p> req;
    std::shared_ptr<vs4l_container_list> in_vs4l_ctl_array;  // vs4l_container_list array
    std::shared_ptr<vs4l_container_list> out_vs4l_ctl_array;  // vs4l_container_list array
    uint32_t bound_core;
    ModePreference mode;
    acc_perf_mode link_mode;
    uint32_t priority;
    bool prepared;
    std::shared_ptr<int32_t> fd_to_vs4l_index;
    int32_t tile_size;
    uint64_t operator_list_id;
    uint32_t get_unique_id() { return unique_id; };
};

struct link_perf_option {
    uint32_t priority;
    uint32_t mode;
    uint32_t latency;
    uint32_t bound;
    uint32_t preset_id;
    link_perf_option() : priority(0), mode(0), latency(0), bound(0), preset_id(0) {}
};

typedef enum _vs4l_syscall_t {
    VS4L_OPEN = 0,
    VS4L_IOCTL,
    VS4L_CLOSE,
} vs4l_syscall_t;

struct vs4l_timer_arg {
    jmp_buf env;
    timer_t timer_id;
    uint32_t cur_timeout;
    uint32_t target_timeout;
    uint32_t interval;
    vs4l_syscall_t type;
    unsigned long request;
};

/* Link layer */
/* Nice to have : TODO(mj.kim010, TBD) : devide class into 2 for SAM score. */
class UdLink {
    public:
        UdLink() : frame_id_(1), bin_node_name_("/dev/vertex10"), mutex_bin_instance_(), soc_idx_(0),
                   max_cluster_(0), acc_hw_error_(0), execute_done_log_count_(0), execute_req_log_count_(0) {
            std::fill_n(max_request_size_, NUM_ACCELERATOR, 0);
            std::fill_n(dev_state_, NUM_ACCELERATOR, DEVICE_UNKNOWN);
            std::fill_n(flag_sram_full_, NUM_ACCELERATOR, 0);
        }

        static UdLink& get_instance() {
            static UdLink instance;
            return instance;
        }
        EnnReturn link_init(accelerator_device acc, uint32_t max_request_size);
        EnnReturn link_open_model(accelerator_device acc, model_info_t* model_info, const EdenModelOptions* options);
        EnnReturn link_close_model(accelerator_device acc, uint64_t model_id);
        EnnReturn link_prepare_req(accelerator_device acc, model_info_t* model_info,
                                std::shared_ptr<eden_memory_t> em_inputs, std::shared_ptr<eden_memory_t> em_outputs,
                                const eden_memory_t* execute_info);
        EnnReturn link_execute_req(accelerator_device acc, req_info_t* req_info, const EdenRequestOptions* options);
        EnnReturn link_shutdown(accelerator_device acc);
        EnnReturn link_get_dd_session_id(accelerator_device acc, const uint64_t model_id, int32_t &session_id);

    private:
        inline int _acc_ioctl(int fd, unsigned long request, void* params);
        inline EnnReturn _acc_prepare(std::shared_ptr<bin_data> bin_instance, struct vs4l_container_list* c);
        inline EnnReturn _acc_qbuf(std::shared_ptr<bin_data> bin_instance, struct vs4l_container_list* c);
        inline EnnReturn _acc_dqbuf(std::shared_ptr<bin_data> bin_instance, struct vs4l_container_list* c);
        EnnReturn link_acc_get_state(accelerator_device acc, dev_state_t* state);
        EnnReturn validate_memory_set(const std::shared_ptr<eden_memory_t> mems, uint32_t count);
        EnnReturn validate_in_out(const std::shared_ptr<bin_data> bin_instance,
                                            const std::shared_ptr<eden_memory_t> em_inputs,
                                            const std::shared_ptr<eden_memory_t> em_outputs);
        EnnReturn allocate_container_list(int cidx, enum vs4l_direction dir, int count, std::shared_ptr<bin_data> bin_instance);
        EnnReturn allocate_sformat_list(struct vs4l_format_list *format_struct,
                                                    enum vs4l_direction dir, int count);
        void release_container(std::shared_ptr<vs4l_container_list> vs4l_container,
                                                int32_t fmap_count, int32_t idx);
        int32_t _cell_align(int32_t size, int cell_size);

        std::shared_ptr<bin_data> get_bin_by_model_info_id(accelerator_device acc, uint32_t minfo_id);
        std::shared_ptr<bin_data> get_bin_by_unique_id(accelerator_device acc, uint32_t unique_id);
        EnnReturn boost_open_close(std::shared_ptr<bin_data> bin, struct link_perf_option *link_option);
        EnnReturn boost_execution(std::shared_ptr<bin_data> bin, struct link_perf_option *link_option);
        EnnReturn _apply_acc_boundness(std::shared_ptr<bin_data> bin, uint32_t bound, uint32_t priority);
        EnnReturn __link_req_done(accelerator_device acc, int32_t device_fd, struct vs4l_timer_arg* timer_arg);
        EnnReturn _acc_init_graph(struct vs4l_graph* graph, eden_memory_t *bin_mem,
                                    struct drv_usr_share* device_bin_data, uint64_t unified_op_id);
        void show_ucgo_model_info(const model_info_t *mdl);
        void show_dsp_loadgraph_info(const ofi_v4_load_graph_info_t *info);
        void show_dsp_exec_info(const ofi_v4_execute_msg_info_t *info);
        void show_sformat_datas(const enum vs4l_direction dir, const int idx,
                                        const shape_t &shape, const struct vs4l_format &format);
        inline void _acc_print_execute_log(uint32_t id, int32_t fd, bool req_done);
        void _dump_vs4l_buffer(struct vs4l_buffer* buffer);
        void _dump_vs4l_container(struct vs4l_container* container);
        void _dump_vs4l_container_list(struct vs4l_container_list* container_list);
        void _dump_ioctl_params(unsigned long ioctl_cmd, int32_t device_fd,
                                struct vs4l_container_list* container_list, struct vs4l_graph* graph,
                                struct vs4l_format_list* format);
        inline EnnReturn init_vs4l_timer_arg(struct vs4l_timer_arg *timer_arg,
                vs4l_syscall_t type, unsigned long request,  uint32_t timeout);
        inline EnnReturn deinit_vs4l_timer_arg(struct vs4l_timer_arg *timer_arg);
        inline void init_itimerspec(struct itimerspec* its, time_t interval_sec);
        inline EnnReturn start_vs4l_timer(struct vs4l_timer_arg *timer_arg,
                vs4l_syscall_t type, unsigned long request, uint32_t timeout);
        inline EnnReturn stop_vs4l_timer(struct vs4l_timer_arg *timer_arg);
        inline int call_vs4l(vs4l_syscall_t type, int fd, unsigned long request, void* params, uint32_t timeout);
        inline uint32_t link_generate_frame_id(void);
        inline EnnReturn set_link_perf_option(struct link_perf_option* link_option, ModePreference mode, uint32_t priority,
                uint32_t latency, uint32_t bound, uint32_t preset_id);

        uint32_t frame_id_;
        const char *bin_node_name_;
        /* Nice to have: TODO(mj.kim, TBD): DO we need npu dsp individual mutex? */
        std::mutex mutex_bin_instance_;
        std::mutex mutex_frame_id_;
        uint32_t soc_idx_;
        uint32_t max_cluster_;
        int32_t acc_hw_error_;
        uint32_t execute_done_log_count_;
        uint32_t execute_req_log_count_;
        uint32_t max_request_size_[NUM_ACCELERATOR];
        dev_state_t dev_state_[NUM_ACCELERATOR];
        std::atomic<int32_t> flag_sram_full_[NUM_ACCELERATOR];
        /* bin ion fd is unique id. */
        std::map <uint32_t, std::shared_ptr<bin_data>> map_bin_instance_[NUM_ACCELERATOR];
        static constexpr uint32_t REQ_PRIORITY_DEFAULT = 0;
        static constexpr uint32_t REQ_PRIORITY_MIN = 0;
        static constexpr uint32_t REQ_PRIORITY_MAX = 256;
};

#endif  // USERDRIVER_UNIFIED_LINK_VS4L_H__
