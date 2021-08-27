/**
 * Copyright (C) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed,
 * transmitted, transcribed, stored in a retrieval system or translated into any human or
 * computer language in any form by any means, electronic, mechanical, manual or
 * otherwise or disclosed to third parties without the express written permission of
 * Samsung Electronics.
 */

#define CONFIG_NPU_MEM_ION
/**
 * Nice to have: TODO(jungho7.kim, TBD): modify *.cc to *.c
 * @file link-vs4l.c
 * @brief This file do A bridge function job between user_driver and device_driver
 * @details Manage the requests received from user_driver and request job to device_driver
 */

#include <errno.h>         // ioctl return check
#include <fcntl.h>         // fd open
#include <malloc.h>        // malloc
#include <sys/ioctl.h>     // ioctl
#include <unistd.h>        // fd close ,sysconf
#include <inttypes.h>      // PRIx64

#include <stdbool.h>       // bool
#include <time.h>          // for timeval
#include <string.h>        // strrchr
#include <vector>

#include <signal.h>

// userdriver/unified/link
#include "link_vs4l.h"
#include "common/compiler.h"

// common
#include "utils.h"         // _CHK_XXX
// osal
#include "userdriver/common/eden_osal/os_mutex.h"      // os_mutex_init, os_mutex_lock, os_mutex_unlock, os_mutex_destroy
// profile
#include "userdriver/unified/profile_util.h"
#include "common/identifier_chopper.hpp"
// Preset
#include "preset/preset_config.hpp"

static EnnReturn get_vs4l_type(char* vs4l_type, vs4l_syscall_t type, unsigned long request) {
    if (unlikely((vs4l_type == NULL))) {
        ENN_WARN_PRINT_FORCE("NULL pointer error!\n");
        return ENN_RET_FAILED;
    }

    // TODO(jungho7.kim): check if VS4L_VERTEXIOC_* can be replaced with 'enum' values
    switch (type) {
        case VS4L_OPEN:
            strcpy(vs4l_type, "open()");
            break;
        case VS4L_IOCTL:
            switch (request) {
                case VS4L_VERTEXIOC_S_GRAPH:
                    strcpy(vs4l_type, "ioctl(S_GRAPH)");
                    break;
                case VS4L_VERTEXIOC_S_FORMAT:
                    strcpy(vs4l_type, "ioctl(S_FORMAT)");
                    break;
                case VS4L_VERTEXIOC_S_PARAM:
                    strcpy(vs4l_type, "ioctl(S_PARAM)");
                    break;
                case VS4L_VERTEXIOC_S_CTRL:
                    strcpy(vs4l_type, "ioctl(S_CTRL)");
                    break;
                case VS4L_VERTEXIOC_STREAM_ON:
                    strcpy(vs4l_type, "ioctl(STREAM_ON)");
                    break;
                case VS4L_VERTEXIOC_STREAM_OFF:
                    strcpy(vs4l_type, "ioctl(STREAM_OFF)");
                    break;
                case VS4L_VERTEXIOC_QBUF:
                    strcpy(vs4l_type, "ioctl(QBUF or DQBUF)");
                    break;
                case VS4L_VERTEXIOC_DQBUF:
                    strcpy(vs4l_type, "ioctl(DQBUF)");
                    break;
                case VS4L_VERTEXIOC_PREPARE:
                    strcpy(vs4l_type, "ioctl(PREPARE)");
                    break;
                case VS4L_VERTEXIOC_UNPREPARE:
                    strcpy(vs4l_type, "ioctl(UNPREPARE)");
                    break;
                case VS4L_VERTEXIOC_SCHED_PARAM:
                    strcpy(vs4l_type, "ioctl(SCHED_PARAM)");
                    break;
#ifdef EXYNOS_NN_PROFILER
                case VS4L_VERTEXIOC_PROFILE_ON:
                    strcpy(vs4l_type, "ioctl(PROFILE_ON)");
                    break;
                case VS4L_VERTEXIOC_PROFILE_OFF:
                    strcpy(vs4l_type, "ioctl(PROFILE_OFF)");
                    break;
#endif
                case VS4L_VERTEXIOC_BOOTUP:
                    strcpy(vs4l_type, "ioctl(BOOTUP)");
                    break;
                default:
                    strcpy(vs4l_type, "Unknown");
                    break;
            }
            break;
        case VS4L_CLOSE:
            strcpy(vs4l_type, "close()");
            break;
        default:
            strcpy(vs4l_type, "Unknown");
            break;
    }
    return ENN_RET_SUCCESS;
}

static void vs4l_timer_handler(int sig_num, siginfo_t *si, void *uc) {
    ENN_UNUSED(sig_num);
    ENN_UNUSED(uc);
    char vs4l_type[MAX_LEN_VS4L_TYPE];

    vs4l_timer_arg* timer_arg;
    timer_arg = (vs4l_timer_arg *) si->si_value.sival_ptr;
    timer_arg->cur_timeout += timer_arg->interval;

    if (unlikely(get_vs4l_type(vs4l_type, timer_arg->type, timer_arg->request) != ENN_RET_SUCCESS)) {
        ENN_WARN_PRINT_FORCE("VS4L timeout warning! target:%u sec, cur:%u sec\n",
                timer_arg->target_timeout, timer_arg->cur_timeout);
    } else {
        ENN_WARN_PRINT_FORCE("VS4L %s timeout warning! target:%u sec, cur:%u sec\n",
                vs4l_type, timer_arg->target_timeout, timer_arg->cur_timeout);
    }

    if ((timer_arg->target_timeout > 0) &&
            (timer_arg->cur_timeout >= timer_arg->target_timeout)) {
        longjmp(timer_arg->env, 1);
    }
}

inline EnnReturn UdLink::init_vs4l_timer_arg(struct vs4l_timer_arg *timer_arg,
        vs4l_syscall_t type, unsigned long request, uint32_t timeout) {
    struct sigevent se;

    se.sigev_notify = SIGEV_SIGNAL;
    se.sigev_signo = SIGRTMIN;
    se.sigev_value.sival_ptr = timer_arg;

    if (unlikely(timer_create(CLOCK_REALTIME, &se, &timer_arg->timer_id) != 0)) {
        ENN_ERR_PRINT_FORCE("fail to timer_create()\n");
        return ENN_RET_FAILED;
    }

    timer_arg->cur_timeout = 0;
    timer_arg->target_timeout = timeout;
    timer_arg->interval = VS4L_TIMER_INTERVAL_SEC;
    timer_arg->type = type;
    timer_arg->request = request;

    return ENN_RET_SUCCESS;
}

inline void update_vs4l_arg_request(struct vs4l_timer_arg *timer_arg, unsigned long request) {
    if (likely(timer_arg != NULL)) {
        timer_arg->request = request;
    } else {
        ENN_WARN_PRINT_FORCE("vs4l_timer_arg NULL pointer error\n");
    }
}

inline EnnReturn UdLink::deinit_vs4l_timer_arg(struct vs4l_timer_arg *timer_arg) {
    if (unlikely(timer_delete(timer_arg->timer_id) != 0)) {
        ENN_ERR_PRINT_FORCE("fail to timer_delete()\n");
        return ENN_RET_FAILED;
    }
    return ENN_RET_SUCCESS;
}

inline void UdLink::init_itimerspec(struct itimerspec* its, time_t interval_sec) {
    its->it_value.tv_sec = interval_sec;
    its->it_value.tv_nsec = 0;
    its->it_interval.tv_sec = interval_sec;
    its->it_interval.tv_nsec = 0;
}

inline EnnReturn UdLink::start_vs4l_timer(struct vs4l_timer_arg *timer_arg,
        vs4l_syscall_t type, unsigned long request, uint32_t timeout) {
    struct itimerspec its;
    struct sigaction sa;

    sa.sa_flags = SA_SIGINFO;
    sa.sa_sigaction = vs4l_timer_handler;
    sigemptyset(&sa.sa_mask);

    if (unlikely(sigaction(SIGRTMIN, &sa, NULL) == -1)) {
        ENN_ERR_PRINT_FORCE("Error, fail to sigaction()\n");
        return ENN_RET_FAILED;
    }

    if (unlikely(init_vs4l_timer_arg(timer_arg, type, request, timeout)) != ENN_RET_SUCCESS) {
        ENN_ERR_PRINT_FORCE("fail to init_vs4l_timer_arg()\n");
        return ENN_RET_FAILED;
    }

    init_itimerspec(&its, timer_arg->interval);

    if(unlikely(timer_settime(timer_arg->timer_id, 0, &its, NULL) != 0)) {
        ENN_ERR_PRINT_FORCE("fail to timer_settime()\n");
        return ENN_RET_FAILED;
    }

    return ENN_RET_SUCCESS;
}

inline EnnReturn UdLink::stop_vs4l_timer(struct vs4l_timer_arg *timer_arg) {
    struct itimerspec its;

    init_itimerspec(&its, 0);
    if (unlikely(timer_settime(timer_arg->timer_id, 0, &its, NULL) != 0)) {
        ENN_ERR_PRINT_FORCE("fail to timer_settime()\n");
        return ENN_RET_FAILED;
    }

    if (unlikely(deinit_vs4l_timer_arg(timer_arg) != ENN_RET_SUCCESS)) {
        ENN_ERR_PRINT_FORCE("fail to deinit_vs4l_timer_arg()\n");
        return ENN_RET_FAILED;
    }

    return ENN_RET_SUCCESS;
}

inline int UdLink::call_vs4l(vs4l_syscall_t type, int fd,
        unsigned long request, void* params, uint32_t timeout) {
    int ret;
    struct vs4l_timer_arg timer_arg;

    if (unlikely(timeout > 0)) {
        if (unlikely(start_vs4l_timer(&timer_arg, type, request, timeout) != ENN_RET_SUCCESS)) {
            ENN_ERR_PRINT_FORCE("fail to start_vs4l_timer()\n");
            return ENN_RET_FAILED;
        }

        if (unlikely(setjmp(timer_arg.env) != 0)) {
            char vs4l_type[MAX_LEN_VS4L_TYPE];

            if (unlikely(get_vs4l_type(vs4l_type, timer_arg.type, timer_arg.request) != ENN_RET_SUCCESS)) {
               ENN_ERR_PRINT_FORCE("VS4L Timeout Error! timeout:%u, fd:%d, request:%lu, parms:%p\n",
                        timer_arg.target_timeout, fd, request, params);
            } else {
               ENN_ERR_PRINT_FORCE("VS4L %s Timeout Error! timeout:%u, fd:%d, request:%lu, parms:%p\n",
                        vs4l_type, timer_arg.target_timeout, fd, request, params);
            }
            stop_vs4l_timer(&timer_arg);
            return ENN_RET_FAILED;
        }
    }

    switch (type) {
        case VS4L_OPEN:
            ret = open(bin_node_name_, O_RDONLY, 0);
            break;
        case VS4L_IOCTL:
            ret = ioctl(fd, request, params);
            break;
        case VS4L_CLOSE:
            ret = close(fd);
            break;
    }

    if (unlikely(timeout > 0))
        stop_vs4l_timer(&timer_arg);

    return ret;
}

void UdLink::show_ucgo_model_info(const model_info_t *mdl) {
    ENN_DBG_PRINT("-----------model_info_t -------------");
    ENN_DBG_PRINT("dsp_model_addr: %p\n", mdl->model_addr);
    ENN_DBG_PRINT("dsp_model_size: %d\n", mdl->model_size);
    ENN_DBG_PRINT("mdl_name: %s\n", (const char *)mdl->model_name);
    ENN_DBG_PRINT("mdl_id: 0x%" PRIX64 "\n", mdl->id);
    ENN_DBG_PRINT("num_in:%d num_out:%d\n", mdl->input_count, mdl->output_count);
    for (int i=0; i< mdl->input_count; i++) {
        shape_t &sh = mdl->bin_in_shape[i];
        ENN_DBG_PRINT("\t in[%d] shape_num(%d) shape_ch(%d) shape_hgt(%d)"
                        "shape_wid(%d) shape_type_size(%d)\n",
                        i, sh.number, sh.channel,
                        sh.height, sh.width, sh.type_size);
        ENN_DBG_PRINT("\t bin_in_idx:(%d) bin_in_bpp(%d)\n",
                        mdl->bin_in_index[i], mdl->bin_in_bpp[i]);

    }
    for (int i=0; i< mdl->output_count; i++) {
        shape_t &sh = mdl->bin_out_shape[i];
        ENN_DBG_PRINT("\t out[%d] shape_num(%d) shape_ch(%d) shape_hgt(%d)"
                        "shape_wid(%d) shape_type_size(%d)\n",
                        i, sh.number, sh.channel,
                        sh.height, sh.width, sh.type_size);
        ENN_DBG_PRINT("\t  bin_out_idx:(%d)\n", mdl->bin_out_index[i]);

    }
    ENN_DBG_PRINT("sub_task_id(0x%" PRIX64 ") priority(%d)\n", mdl->operator_list_id, mdl->priority);
    const shape_t &sh = mdl->cell_align_shape;
    ENN_DBG_PRINT("number(%d) ch(%d) hgh(%d) wid(%d) type_size(%d)\n",
                        sh.number, sh.channel, sh.height, sh.width, sh.type_size);
    ENN_DBG_PRINT("shared(%d) binding_ofm(%d) tile_size(%d) exec_msg_size(%d)\n",
                    mdl->shared_buffer, mdl->binding_ofm, mdl->tile_size, mdl->get_exec_msg_size());
    ENN_DBG_PRINT("kernel_name(%s)\n", (const char *)mdl->kernel_name);
    ENN_DBG_PRINT("kernel_name_size(%d) kernel_name_count(%d) "
                    "reserved(%d)\n", mdl->kernel_name_size,
                    mdl->kernel_name_count, mdl->reserved[0]);
    ENN_DBG_PRINT("----------------------------------------");
}


void UdLink::show_dsp_loadgraph_info(const ofi_v4_load_graph_info_t *info) {
    #define MEM_PARAM_WIDTH(param_mem) (param_mem.param[0])
    #define MEM_PARAM_HEIGHT(param_mem) (param_mem.param[1])
    #define MEM_PARAM_CHANNEL(param_mem) (param_mem.param[2])

    ENN_DBG_PRINT("-------------- load_graph_info_t --------------\n");
    ENN_DBG_PRINT("graph_id=0x%x, n_tsgd=%d, n_param=%d, n_kernel:%d\n",
                    info->global_id, info->n_tsgd, info->n_param, info->n_kernel);

    for (uint32_t i = 0; i < (info->n_param + 1); i++) { // +1 for TSGD.
        auto &prm = info->param_list[i];
        auto &mem = prm.param_mem;
        ENN_DBG_PRINT("----------------------------------\n");
        ENN_DBG_PRINT("OrcaParam[%d] type(0x%x) idx##(%d)##\n", i, prm.param_type,
                        prm.idx.param_index);
        ENN_DBG_PRINT("--          mem_info       ----\n");
        ENN_DBG_PRINT("\taddr_type(%d) mem_attr(%d) mem_type(%d)\n", mem.addr_type,
                        mem.mem_attr, mem.mem_type);
        if (mem.mem_type == OFI_MEM_ION) {
            bool empty = (mem.get_addr.mem.fd == VALUE_FD_INIT) &
                                  (mem.get_addr.mem.iova == VALUE_IOVA_INIT);
            ENN_DBG_PRINT("\tmandatory(%d) size(%d) fd(%d) EMPTY(%d)\n", mem.is_mandatory,
                            mem.size, mem.get_addr.mem.fd, empty);
        } else {
            ENN_DBG_PRINT("\tmandatory(%d) size(%d) addr(%p)\n", mem.is_mandatory,
                            mem.size, mem.get_addr.addr);
        }
        ENN_DBG_PRINT("\tmem_offset(%d) w(%d) h(%d) c(%d)\n",
                        mem.offset, MEM_PARAM_WIDTH(mem),
                        MEM_PARAM_HEIGHT(mem), MEM_PARAM_CHANNEL(mem));
    }
    ENN_DBG_PRINT("-----------------------------------------------\n");
}


void UdLink::show_dsp_exec_info(const ofi_v4_execute_msg_info_t *info) {
    ENN_DBG_PRINT("--------------- execute_msg_info ---------------\n");
    ENN_DBG_PRINT("MSG_ID=0x%x, n_update_param=%d\n", info->global_id,
                    info->n_update_param);

    for (uint32_t i = 0; i < info->n_update_param; i++) {
        auto &prm = info->param_list[i];
        auto &mem = prm.param_mem;
        ENN_DBG_PRINT("-------------------------------------\n");
        ENN_DBG_PRINT("UpdateParam[%d] type(0x%x) idx##(%d)##\n", i, prm.param_type,
                        prm.idx.param_index);
        ENN_DBG_PRINT("--          mem_info       ----\n");
        ENN_DBG_PRINT("\taddr_type(%d) mem_attr(%d) mem_type(%d)\n", mem.addr_type,
                        mem.mem_attr, mem.mem_type);
        if (mem.mem_type == OFI_MEM_ION) {
            bool empty = (mem.get_addr.mem.fd == VALUE_FD_INIT) &
                                  (mem.get_addr.mem.iova == VALUE_IOVA_INIT);
            ENN_DBG_PRINT("\tmandatory(%d) size(%d) fd(%d) EMPTY(%d)\n", mem.is_mandatory,
                            mem.size, mem.get_addr.mem.fd, empty);
        } else {
            ENN_DBG_PRINT("\tmandatory(%d) size(%d) addr(%p)\n", mem.is_mandatory,
                            mem.size, mem.get_addr.addr);
        }
        ENN_DBG_PRINT("\tmem_offset(%d) w(%d) h(%d) c(%d)\n",
                        mem.offset, MEM_PARAM_WIDTH(mem),
                        MEM_PARAM_HEIGHT(mem), MEM_PARAM_CHANNEL(mem));
    }
    ENN_DBG_PRINT("-------------------------------------------------\n");
}


inline void UdLink::_acc_print_execute_log(uint32_t id, int32_t fd, bool req_done) {
    if (req_done) {
        if (++execute_done_log_count_ == LINK_LOG_PRINT_COUNT) {
            ENN_INFO_PRINT("interval[%d], link polling dqbuf done | unique id : [%d], fd : [%d]", LINK_LOG_PRINT_COUNT, id, fd);
            execute_done_log_count_ = 0;
        }
    } else {
        if (++execute_req_log_count_ == LINK_LOG_PRINT_COUNT) {
            ENN_INFO_PRINT("interval[%d], link execute req | unique id : [%d], fd : [%d]", LINK_LOG_PRINT_COUNT, id, fd);
            execute_req_log_count_ = 0;
        }
    }
}


void UdLink::_dump_vs4l_buffer(struct vs4l_buffer* buffer) {
    if (buffer == NULL) {
        ENN_DBG_PRINT("     - Oops! buffer is NULL!!!\n");
        return;
    }

    ENN_DBG_PRINT("     - roi = [%d,%d,%d,%d], userptr = [0x%lx], fd = [%d]\n",
         buffer->roi.x, buffer->roi.y, buffer->roi.w, buffer->roi.h,
         buffer->m.userptr, buffer->m.fd);
}


void UdLink::_dump_vs4l_container(struct vs4l_container* container) {
    if (container == NULL) {
        ENN_DBG_PRINT("   - Oops! container is NULL!!!\n");
        return;
    }

    ENN_DBG_PRINT("   - type = [%d], target = [%d], memory = [%d], count = [%d], buffers = [%p]\n",
         container->type, container->target, container->memory, container->count, container->buffers);

    for (int i = 0; i < container->count; i++) {
        ENN_DBG_PRINT("   - Dump vs4l_buffer[%d]...\n", i);
        _dump_vs4l_buffer(&(container->buffers[i]));
    }
}


void UdLink::_dump_vs4l_container_list(struct vs4l_container_list* container_list) {
    if (container_list == NULL) {
        ENN_DBG_PRINT(" - Oops! container_list is NULL!!!\n");
        return;
    }

    ENN_DBG_PRINT(" - Dump vs4l_container_list...\n");
    ENN_DBG_PRINT("   - direction = [%d], id = [%d], index = [%d], flags = [%d], "
                    "timestamp = [X], count = [%d], containers = [%p]\n",
                    container_list->direction, container_list->id, container_list->index, container_list->flags,
                    container_list->count, container_list->containers);
    for (int i = 0; i < container_list->count; i++) {
        ENN_DBG_PRINT("   - Dump vs4l_container[%d]...\n", i);
        _dump_vs4l_container(&(container_list->containers[i]));
    }
}


void UdLink::_dump_ioctl_params(unsigned long ioctl_cmd, int32_t device_fd,
            struct vs4l_container_list* container_list, struct vs4l_graph* graph,
            struct vs4l_format_list* format) {
    if (ioctl_cmd == VS4L_VERTEXIOC_S_GRAPH) {
        ENN_DBG_PRINT("Dump ioctl params for VS4L_VERTEXIOC_S_GRAPH...\n");
        ENN_DBG_PRINT(" - device_fd = [%d]\n", device_fd);
        ENN_DBG_PRINT(" - Dump vs4l_graph...\n");
        if (graph == NULL) {
            ENN_DBG_PRINT(" - Oops! graph is NULL!!!\n");
        } else {
            ENN_DBG_PRINT("   - id = [%d], priority = [%d], time = [%d], flags = [%d], size = [%d], addr = [0x%lx]\n",
                 graph->id, graph->priority, graph->time, graph->flags, graph->size, graph->addr);
        }
        ENN_DBG_PRINT("Dump ioctl params for VS4L_VERTEXIOC_S_GRAPH, Done\n");
    } else if (ioctl_cmd == VS4L_VERTEXIOC_STREAM_ON) {
        ENN_DBG_PRINT("Dump ioctl params for VS4L_VERTEXIOC_STREAM_ON...\n");
        ENN_DBG_PRINT(" - device_fd = [%d]\n", device_fd);
        ENN_DBG_PRINT("Dump ioctl params for VS4L_VERTEXIOC_STREAM_ON, Done\n");
    } else if (ioctl_cmd == VS4L_VERTEXIOC_STREAM_OFF) {
        ENN_DBG_PRINT("Dump ioctl params for VS4L_VERTEXIOC_STREAM_OFF...\n");
        ENN_DBG_PRINT(" - device_fd = [%d]\n", device_fd);
        ENN_DBG_PRINT("Dump ioctl params for VS4L_VERTEXIOC_STREAM_OFF, Done\n");
    } else if (ioctl_cmd == VS4L_VERTEXIOC_QBUF || ioctl_cmd == VS4L_VERTEXIOC_PREPARE) {
        ENN_DBG_PRINT("Dump ioctl params for VS4L_VERTEXIOC_QBUF...\n");
        ENN_DBG_PRINT(" - device_fd = [%d]\n", device_fd);
        _dump_vs4l_container_list(container_list);
        ENN_DBG_PRINT("Dump ioctl params for VS4L_VERTEXIOC_QBUF, Done\n");
    } else if (ioctl_cmd == VS4L_VERTEXIOC_DQBUF || ioctl_cmd == VS4L_VERTEXIOC_PREPARE) {
        ENN_DBG_PRINT("Dump ioctl params for VS4L_VERTEXIOC_DQBUF...\n");
        ENN_DBG_PRINT(" - device_fd = [%d]\n", device_fd);
        _dump_vs4l_container_list(container_list);
        ENN_DBG_PRINT("Dump ioctl params for VS4L_VERTEXIOC_DQBUF, Done\n");
    } else if (ioctl_cmd == VS4L_VERTEXIOC_S_FORMAT) {
        ENN_DBG_PRINT("Dump ioctl params for VS4L_VERTEXIOC_S_FORMAT...\n");
        for (int i = 0; i < format->count; i++) {
            ENN_DBG_PRINT("---------------------------------------\n");
            ENN_DBG_PRINT(" - target = [%d]\n", format->formats[i].target);
            ENN_DBG_PRINT(" - format = [0x%08x]\n", format->formats[i].format);
            ENN_DBG_PRINT(" - plane = [%d]\n", format->formats[i].plane);
            ENN_DBG_PRINT(" - width = [%d]\n", format->formats[i].width);
            ENN_DBG_PRINT(" - height = [%d]\n", format->formats[i].height);
            ENN_DBG_PRINT(" - stride = [%d]\n", format->formats[i].stride);
            ENN_DBG_PRINT(" - cstride = [%d]\n", format->formats[i].cstride);
            ENN_DBG_PRINT(" - channels = [%d]\n", format->formats[i].channels);
            ENN_DBG_PRINT(" - pixel_format = [%d]\n", format->formats[i].pixel_format);
        }
        ENN_DBG_PRINT("---------------------------------------\n");
        ENN_DBG_PRINT("Dump ioctl params for VS4L_VERTEXIOC_S_FORMAT...Done\n");
    }
}


int32_t UdLink::_cell_align(int32_t size, int cell_size) {
    return ((size + (cell_size - 1)) / cell_size) * cell_size;
}


/**
 * @brief find bin instance from map
 * @details guarded w/ mutex
 * @param[in] unique_id as a key of map
 * @returns std::shared_ptr<bin_data>
 * @returns nullptr
 */
std::shared_ptr<bin_data> UdLink::get_bin_by_unique_id(accelerator_device acc, uint32_t unique_id) {
    std::lock_guard<std::mutex> guard(mutex_bin_instance_);
    auto found = map_bin_instance_[acc].find(unique_id);
    if (found == map_bin_instance_[acc].end() || found->second == nullptr) {
        ENN_ERR_PRINT_FORCE("Fail to find bin_instance. unique_id(%d)\n", unique_id);
        return nullptr;
    }
    return found->second;
}

std::shared_ptr<bin_data> UdLink::get_bin_by_model_info_id(accelerator_device acc, uint32_t minfo_id) {
    std::lock_guard<std::mutex> guard(mutex_bin_instance_);
    for (auto &itr : map_bin_instance_[acc]) {
        if (itr.second) {
            int32_t bin_minfo_id = itr.second->model_info_id;
            if (minfo_id == bin_minfo_id) {
                return itr.second;
            }
        }
    }
    ENN_ERR_PRINT_FORCE("Fail to find bin_instance. model_info_id(%d)\n", minfo_id);
    return nullptr;
}

// TODO(jungho7.kim): replace this function with Preset function
EnnReturn UdLink::boost_execution(std::shared_ptr<bin_data> bin, struct link_perf_option *link_option) {
    if ((bin == nullptr) || (link_option == nullptr)) {
        ENN_ERR_PRINT_FORCE("Null pointer error!\n");
        return ENN_RET_FAILED;
    }

    ENN_DBG_PRINT("bin->mode:%d, link_option->mode:%d\n", (int) bin->mode, (int) link_option->mode);
    if (bin->mode == BOOST_ON_EXECUTE_MODE) {
        struct vs4l_param mode_val[2];
        struct vs4l_param_list mode_param;

        mode_val[0].target = NPU_S_PARAM_PERF_MODE;
        mode_val[0].offset = link_option->mode;

        if (link_option->mode == NPU_S_PARAM_PERF_MODE_NONE) {
            mode_val[1].target = NPU_S_PARAM_QOS_CL1;
            mode_val[1].offset = 0;
        } else if (link_option->mode == NPU_S_PARAM_PERF_MODE_NPU_BOOST_ON_EXECUTE) {
            mode_val[1].target = NPU_S_PARAM_QOS_CL1;
            mode_val[1].offset = target_tuned_freq[soc_idx_][PM_CPU_CL1_MAX];
        } else {
            ENN_ERR_PRINT_FORCE("Invalid link_option->mode:%u\n", link_option->mode);
            return ENN_RET_FAILED;
        }

        mode_param.params = &mode_val[0];
        mode_param.count = 2;

        int32_t drv_ret = ENN_RET_SUCCESS;
        ENN_DBG_PRINT("[IOCTL] VS4L_VERTEXIOC_S_PARAM, FD : %d\n", bin->device_fd);
        drv_ret = call_vs4l(VS4L_IOCTL, bin->device_fd,
                VS4L_VERTEXIOC_S_PARAM, &mode_param, VS4L_TIMER_TIMEOUT_SEC);
        if (drv_ret == ENN_RET_SUCCESS) {
            ENN_DBG_PRINT("success to boost_execute(), mode num  : %u\n", link_option->mode);
        } else if (drv_ret == EMERGENCY_RECOVERY) {
            ENN_WARN_PRINT("EMERGENCY_RECOVERY!! performance isn't applied, skip.\n");
        } else {
            ENN_ERR_PRINT_FORCE("fail to boost_execute(), mode:%u\n", link_option->mode);
            return ENN_RET_FAILED;
        }
        ENN_DBG_PRINT("(-)\n");
        return ENN_RET_SUCCESS;
    }
    return ENN_RET_SUCCESS;
}

/**
 * @brief apply performance mode
 * @details apply priority, boost and TPF
 * @param[in] *bin bin data for ioctl
 * @returns ENN_RET_SUCCESS: ioctl success
 * @returns LINK_APPLY_PREFERENCE_FAILED: ioctl failure
 */
EnnReturn UdLink::boost_open_close(std::shared_ptr<bin_data> bin, struct link_perf_option *link_option) {
    ENN_DBG_PRINT("(+)\n");
    ENN_INFO_PRINT("NPU_S_PARAM_PERF_MODE : %u", link_option->mode);
    ENN_INFO_PRINT("NPU_S_PARAM_LATENCY : %u", link_option->latency);

    const uint32_t PARAMS = 50;
    int32_t vs4l_param_index = 0;
    struct vs4l_param mode_val[PARAMS];
    for (int i = 0; i < PARAMS; i++) {
        mode_val[i].target = 0;
        mode_val[i].offset = 0;
    }

    // TODO(jungho7.kim): move Preset-related code to Preset driver
    std::shared_ptr<enn::preset::PresetConfig> preset_config;
    if (link_option->preset_id > 0) {
        auto preset_config_manager = enn::preset::PresetConfigManager::get_instance();
        const std::string json_file = "/vendor/etc/enn/uenn_preset.json";

        ENN_DBG_PRINT("preset_id:%u, json file:%s\n", link_option->preset_id, json_file.c_str());

        if (preset_config_manager->parse_scenario_from_ext_json_v2(json_file) == ENN_RET_SUCCESS) {
            preset_config = preset_config_manager->get_preset_config_by_id(link_option->preset_id);
            if (preset_config == nullptr)
                ENN_DBG_PRINT("preset_config nullptr error!\n");
        }
    }

    // Target latency control
    if (link_option->latency != 0) {
        mode_val[vs4l_param_index].target = NPU_S_PARAM_TPF;
        mode_val[vs4l_param_index].offset = link_option->latency; /* us */
        vs4l_param_index++;
    }

    if (link_option->mode != NPU_S_PARAM_PERF_MODE_NPU_BOOST_ON_EXECUTE) {
        if (link_option->mode == NPU_S_PARAM_PERF_MODE_NONE) {
            mode_val[vs4l_param_index].target = NPU_S_PARAM_PERF_MODE;  // 0x900000
            mode_val[vs4l_param_index].offset = link_option->mode;
            vs4l_param_index++;
        } else {
            // TODO(jungho7.kim): move Preset-related code to Preset driver
            if (preset_config == nullptr) {
                mode_val[vs4l_param_index].target = NPU_S_PARAM_PERF_MODE;  // 0x900000
                mode_val[vs4l_param_index].offset = link_option->mode;
                vs4l_param_index++;
                // TODO(jungho7.kim): replace it with Preset extension
                mode_val[vs4l_param_index].target = NPU_S_PARAM_QOS_CL1;  // 0x890001
                mode_val[vs4l_param_index].offset = target_max_freq[soc_idx_][PM_CPU_CL1_MAX];
                ENN_DBG_PRINT("CPU CL1 freq, target:%u offset:%u\n",
                        mode_val[vs4l_param_index].target, mode_val[vs4l_param_index].offset);
                vs4l_param_index++;
                // TODO(jungho7.kim): manage the target and offset by using some container
                // TODO(jungho7.kim): replace it with Preset extension
                mode_val[vs4l_param_index].target = NPU_S_PARAM_QOS_DSP;
                mode_val[vs4l_param_index].offset = 1066000;
                ENN_DBG_PRINT("DSP freq, target:%u offset:%u\n",
                        mode_val[vs4l_param_index].target, mode_val[vs4l_param_index].offset);
                vs4l_param_index++;
            } else {
                // TODO(jungho7.kim): move Preset-related code to Preset driver
                int32_t cpu_lit_freq, cpu_mid_freq, cpu_big_freq, dd_kpi_mode;
                int32_t npu_freq, dnc_freq, dsp_freq, mif_freq, int_freq;
                cpu_lit_freq = preset_config->get_target_cpu_lit_freq();
                cpu_mid_freq = preset_config->get_target_cpu_mid_freq();
                cpu_big_freq = preset_config->get_target_cpu_big_freq();
                npu_freq = preset_config->get_target_npu_freq();
                dnc_freq = preset_config->get_target_dnc_freq();
                dsp_freq = preset_config->get_target_dsp_freq();
                mif_freq = preset_config->get_target_mif_freq();
                int_freq = preset_config->get_target_int_freq();
                dd_kpi_mode = preset_config->get_target_dd_kpi_mode();
                ENN_DBG_PRINT("Preset configuration\n");
                ENN_DBG_PRINT("CPU(lit):%d CPU(mid):%d CPU(big):%d\n",
                        cpu_lit_freq, cpu_mid_freq, cpu_big_freq);
                ENN_DBG_PRINT("NPU:%d DNC:%d DSP:%d\n", npu_freq, dnc_freq, dsp_freq);
                ENN_DBG_PRINT("MIF:%d INT:%d\n", mif_freq, int_freq);
                ENN_DBG_PRINT("DD KPI mode:%d\n", dd_kpi_mode);

                // TODO(jungho7.kim): remove duplicate codes
                if (dd_kpi_mode == 1) {
                    mode_val[vs4l_param_index].target = NPU_S_PARAM_PERF_MODE;  // 0x900000
                    mode_val[vs4l_param_index].offset = NPU_S_PARAM_PERF_MODE_NPU_BOOST_BLOCKING;
                    vs4l_param_index++;
                } else {
                    mode_val[vs4l_param_index].target = NPU_S_PARAM_PERF_MODE;  // 0x900000
                    mode_val[vs4l_param_index].offset = link_option->mode;
                    vs4l_param_index++;
                }
                if (npu_freq > 0) {
                    mode_val[vs4l_param_index].target = NPU_S_PARAM_QOS_NPU_MAX;
                    mode_val[vs4l_param_index].offset = npu_freq;
                    vs4l_param_index++;
                    mode_val[vs4l_param_index].target = NPU_S_PARAM_QOS_NPU;
                    mode_val[vs4l_param_index].offset = npu_freq;
                    vs4l_param_index++;
                }
                if (dnc_freq > 0) {
                    mode_val[vs4l_param_index].target = NPU_S_PARAM_QOS_DNC;
                    mode_val[vs4l_param_index].offset = dnc_freq;
                    vs4l_param_index++;
                    mode_val[vs4l_param_index].target = NPU_S_PARAM_QOS_DNC_MAX;
                    mode_val[vs4l_param_index].offset = dnc_freq;
                    vs4l_param_index++;
                }
                if (dsp_freq > 0) {
                    mode_val[vs4l_param_index].target = NPU_S_PARAM_QOS_DSP;
                    mode_val[vs4l_param_index].offset = dsp_freq;
                    vs4l_param_index++;
                    mode_val[vs4l_param_index].target = NPU_S_PARAM_QOS_DSP_MAX;
                    mode_val[vs4l_param_index].offset = dsp_freq;
                    vs4l_param_index++;
                }
                if (cpu_lit_freq > 0) {
                    mode_val[vs4l_param_index].target = NPU_S_PARAM_QOS_CL0;
                    mode_val[vs4l_param_index].offset = cpu_lit_freq;
                    vs4l_param_index++;
                    mode_val[vs4l_param_index].target = NPU_S_PARAM_QOS_CL0_MAX;
                    mode_val[vs4l_param_index].offset = cpu_lit_freq;
                    vs4l_param_index++;
                }
                if (cpu_mid_freq > 0) {
                    mode_val[vs4l_param_index].target = NPU_S_PARAM_QOS_CL1;
                    mode_val[vs4l_param_index].offset = cpu_mid_freq;
                    vs4l_param_index++;
                    mode_val[vs4l_param_index].target = NPU_S_PARAM_QOS_CL1_MAX;
                    mode_val[vs4l_param_index].offset = cpu_mid_freq;
                    vs4l_param_index++;
                }
                if (cpu_big_freq > 0) {
                    mode_val[vs4l_param_index].target = NPU_S_PARAM_QOS_CL2;
                    mode_val[vs4l_param_index].offset = cpu_big_freq;
                    vs4l_param_index++;
                    mode_val[vs4l_param_index].target = NPU_S_PARAM_QOS_CL2_MAX;
                    mode_val[vs4l_param_index].offset = cpu_big_freq;
                    vs4l_param_index++;
                }
                if (mif_freq > 0) {
                    mode_val[vs4l_param_index].target = NPU_S_PARAM_QOS_MIF;
                    mode_val[vs4l_param_index].offset = mif_freq;
                    vs4l_param_index++;
                    mode_val[vs4l_param_index].target = NPU_S_PARAM_QOS_MIF_MAX;
                    mode_val[vs4l_param_index].offset = mif_freq;
                    vs4l_param_index++;
                }
                if (int_freq > 0) {
                    mode_val[vs4l_param_index].target = NPU_S_PARAM_QOS_INT;
                    mode_val[vs4l_param_index].offset = int_freq;
                    vs4l_param_index++;
                    mode_val[vs4l_param_index].target = NPU_S_PARAM_QOS_INT_MAX;
                    mode_val[vs4l_param_index].offset = int_freq;
                    vs4l_param_index++;
                }
            } // if (preset_config == nullptr)
        } // if (link_option->mode == NPU_S_PARAM_PERF_MODE_NONE)
    } // if (link_option->mode != NPU_S_PARAM_PERF_MODE_NPU_BOOST_ON_EXECUTE)

    struct vs4l_param_list mode_param;
    mode_param.params = &mode_val[0];
    mode_param.count = vs4l_param_index;

    int32_t drv_ret = ENN_RET_SUCCESS;
    ENN_DBG_PRINT("[IOCTL] VS4L_VERTEXIOC_S_PARAM, FD : %d\n", bin->device_fd);
    drv_ret = call_vs4l(VS4L_IOCTL, bin->device_fd,
            VS4L_VERTEXIOC_S_PARAM, &mode_param, VS4L_TIMER_TIMEOUT_SEC);
    if (drv_ret == ENN_RET_SUCCESS) {
        ENN_DBG_PRINT("succeeded applying the performance, mode num  : %u\n", link_option->mode);
    } else if (drv_ret == EMERGENCY_RECOVERY) {
        ENN_WARN_PRINT("EMERGENCY_RECOVERY!! performance isn't applied, skip.\n");
    } else {
        ENN_ERR_PRINT_FORCE("failed applying the performance, mode:%u, vs4l_param_index:%d\n",
                link_option->mode, vs4l_param_index);
        return ENN_RET_FAILED;
    }

    ENN_DBG_PRINT("(-)\n");
    return ENN_RET_SUCCESS;
}

EnnReturn UdLink::_apply_acc_boundness(std::shared_ptr<bin_data> bin, uint32_t bound, uint32_t priority) {
    ENN_DBG_PRINT("(+) fd : %d, bound : %u, priority : %u\n", bin->device_fd, bound, priority);

    struct vs4l_sched_param sched_val;
    sched_val.bound_id = bound;
    sched_val.priority = priority;

    ENN_DBG_PRINT("[IOCTL] VS4L_VERTEXIOC_SCHED_PARAM, FD : %d\n", bin->device_fd);
    int32_t drv_ret = call_vs4l(VS4L_IOCTL, bin->device_fd,
            VS4L_VERTEXIOC_SCHED_PARAM, &sched_val, VS4L_TIMER_TIMEOUT_SEC);
    if (drv_ret == ENN_RET_SUCCESS) {
        ENN_DBG_PRINT("succeeded applying the boundness, bound num  : %u\n", bound);
    } else if (drv_ret == EMERGENCY_RECOVERY) {
        ENN_WARN_PRINT("EMERGENCY_RECOVERY!! boundness isn't applied, skip.\n");
    } else {
        ENN_ERR_PRINT_FORCE("failed applying the boundness, bound num  : %u\n", bound);
        return ENN_RET_FAILED;
    }
    ENN_DBG_PRINT("(-)\n");
    return ENN_RET_SUCCESS;
}

/**
 * @brief request prepare to the NPU driver
 * @details call ioctl for enqueuing request.
 * @param[in] *bin_instance bin data for qbuf
 * @param[in] *c vs4l_container_list as input data
 * @returns ENN_RET_SUCCESS: ioctl success
 * @returns ENN_RET_FAILED: ioctl failure
 */
inline EnnReturn UdLink::_acc_prepare(std::shared_ptr<bin_data> bin_instance, struct vs4l_container_list* c) {
    _dump_ioctl_params(VS4L_VERTEXIOC_PREPARE, bin_instance->device_fd, c, NULL, NULL);
#if !defined(NPU_DD_EMULATOR)
    return (EnnReturn) call_vs4l(VS4L_IOCTL, bin_instance->device_fd,
            VS4L_VERTEXIOC_PREPARE, c, VS4L_TIMER_TIMEOUT_SEC);
#endif
}

/**
 * @brief request enqueue to the NPU driver
 * @details call ioctl for enqueuing request.
 * @param[in] *bin_instance bin data for qbuf
 * @param[in] *c vs4l_container_list as input data
 * @returns ENN_RET_SUCCESS: ioctl success
 * @returns ENN_RET_FAILED: ioctl failure
 */
inline EnnReturn UdLink::_acc_qbuf(std::shared_ptr<bin_data> bin_instance, struct vs4l_container_list* c) {

    _dump_ioctl_params(VS4L_VERTEXIOC_QBUF, bin_instance->device_fd, c, NULL, NULL);

#if !defined(NPU_DD_EMULATOR)
    ENN_DBG_PRINT("[IOCTL] VS4L_VERTEXIOC_QBUF, FD : %d\n", bin_instance->device_fd);
    return (EnnReturn) call_vs4l(VS4L_IOCTL, bin_instance->device_fd, VS4L_VERTEXIOC_QBUF, c, 0);
#endif
}

/**
 * @brief request dequeue to the NPU driver
 * @details call ioctl for dequeuing request.
 * @param[in] *bin_instance bin data for dqbuf
 * @param[out] *c - vs4l_container_list returned as output data
 * @returns ENN_RET_SUCCESS: ioctl success
 * @returns EWOULDBLOCK: ioctl success, no more request
 * @returns POSITIVE_VAL: ioctl failure
 */
inline EnnReturn UdLink::_acc_dqbuf(std::shared_ptr<bin_data> bin_instance, struct vs4l_container_list* c) {
    _dump_ioctl_params(VS4L_VERTEXIOC_DQBUF, bin_instance->device_fd, c, NULL, NULL);
#if !defined(NPU_DD_EMULATOR)
    int32_t drv_ret = ENN_RET_SUCCESS;
    ENN_DBG_PRINT("[IOCTL] VS4L_VERTEXIOC_DQBUF, FD : %d\n", bin_instance->device_fd);
    if (call_vs4l(VS4L_IOCTL, bin_instance->device_fd, VS4L_VERTEXIOC_DQBUF, c, 0)) {
        drv_ret = errno;
    }
#endif
    return (EnnReturn) drv_ret;
}

/**
 * @brief try dqbuf and return output to the user driver
 * @details try dqbuf until dqbuf fail and return output container to the user driver
 * @param[in] fd dequeue target fd
 * @returns ENN_RET_SUCCESS: dqbuf and return output container success
 * @returns ENN_RET_FAILED: dqbuf for target fd failure
 */
EnnReturn UdLink::__link_req_done(accelerator_device acc, int32_t device_fd, struct vs4l_timer_arg* timer_arg) {
    ENN_UNUSED(timer_arg);
    std::shared_ptr<bin_data> bin_instance = nullptr;

    bool bin_exist = false;

    {
        std::lock_guard<std::mutex> guard(mutex_bin_instance_);
        for (auto &itr : map_bin_instance_[acc]) {
            if (itr.second) {
                int32_t dev_fd = itr.second->device_fd;
                if (dev_fd == device_fd) {
                    bin_exist = true;
                    bin_instance = itr.second;
                    break;
                }
            }
        }
    }
    if (bin_exist == false) {
        ENN_ERR_PRINT_FORCE("ERR: Fail to find bin_instance : %d\n", device_fd);
        return ENN_RET_FAILED;
    }

    ENN_DBG_PRINT("done request device_fd : %d\n", device_fd);

    // TODO(jungho7.kim): change to in_container() or set_func()
    struct vs4l_container_list in_container;
    in_container.direction = VS4L_DIRECTION_IN;
    in_container.id = 0;
    in_container.index = 0;
    in_container.flags = 0;
    in_container.count = 0;
    in_container.containers = NULL;
    in_container.timestamp[0].tv_usec = 0;
    in_container.timestamp[0].tv_sec = 0;

    int32_t drv_ret = _acc_dqbuf(bin_instance, &in_container);
    if (drv_ret == EWOULDBLOCK) {
        ENN_DBG_PRINT("_acc_dqbuf : there is no more request\n");
        return ENN_RET_SUCCESS;
    } else if (drv_ret) {
        ENN_ERR_PRINT_FORCE("_acc_dqbuf : failed dqbuf!\n");
        return ENN_RET_FAILED;
    } else {
        if (unlikely(in_container.flags & (1 << VS4L_CL_FLAG_INVALID))) {
            // TODO(jungho7.kim): change to bin_instance.str() in_container.str()
            ENN_ERR_PRINT_FORCE("_acc_dqbuf : input_buf value is invalid!!\n");
            ENN_ERR_PRINT_FORCE("=========================================\n");
            ENN_ERR_PRINT_FORCE("bin_instance->unique_id : %d\n", bin_instance->unique_id);
            ENN_ERR_PRINT_FORCE("bin_instance->device_fd : %d\n", bin_instance->device_fd);
            ENN_ERR_PRINT_FORCE("=========================================\n");
            ENN_ERR_PRINT_FORCE("in_container.direction : %d\n", in_container.direction);
            ENN_ERR_PRINT_FORCE("in_container.id : %d\n", in_container.id);
            ENN_ERR_PRINT_FORCE("in_container.index : %d\n", in_container.index);
            ENN_ERR_PRINT_FORCE("in_container.flags : %d\n", in_container.flags);
            ENN_ERR_PRINT_FORCE("in_container.count : %d\n", in_container.count);
            ENN_ERR_PRINT_FORCE("=========================================\n");
            return ENN_RET_FAILED;
        } else {
            ENN_DBG_PRINT("_acc_dqbuf : in_container dqbuf succeeded\n");
        }

        // TODO(jungho7.kim): change to out_container.str()
        struct vs4l_container_list out_container;
        out_container.direction = VS4L_DIRECTION_OT;
        out_container.id = 0;
        out_container.index = 0;
        out_container.flags = 0;
        out_container.count = 0;
        out_container.containers = NULL;
        gettimeofday(&out_container.timestamp[0], NULL);

        if (!_acc_dqbuf(bin_instance, &out_container)) {
#ifdef EXYNOS_NN_PROFILER
            if (acc == ACCELERATOR_NPU) {
                PROFILE_EXCLUDE_FROM("VS4L_VERTEXIOC_PROFILE_OFF", enn::util::chop_into_model_id(bin_instance->operator_list_id));

                struct vs4l_profiler* profiler;
                _allocate_profile_tree(&profiler);

                ENN_DBG_PRINT("[IOCTL] VS4L_VERTEXIOC_PROFILE_OFF, FD : %d\n", device_fd);
                if (call_vs4l(VS4L_IOCTL, bin_instance->device_fd,
                            VS4L_VERTEXIOC_PROFILE_OFF, profiler, VS4L_TIMER_TIMEOUT_SEC))
                    ENN_ERR_PRINT_FORCE("Failed to turn off profiler!\n");

                PROFILE_EXCLUDE_UNTIL("VS4L_VERTEXIOC_PROFILE_OFF", enn::util::chop_into_model_id(bin_instance->operator_list_id));

                _append_profiler_node(profiler->node, enn::util::chop_into_model_id(bin_instance->operator_list_id));
                _release_profiler_struct(&profiler);
                PROFILE_UNTIL("Device_Driver_NPU_Execution", enn::util::chop_into_model_id(bin_instance->operator_list_id));
            }
            else if (acc == ACCELERATOR_DSP) {
                //PROFILE_UNTIL("Device_Driver_DSP_Execution", enn::util::chop_into_model_id(bin_instance->operator_list_id));
            }
#endif  // EXYNOS_NN_PROFILER
            struct acc_req* done_req = nullptr;
            {
                std::lock_guard<std::mutex> guard(mutex_bin_instance_);
                done_req = bin_instance->req.get()[out_container.index];
                bin_instance->req.get()[in_container.index]  = nullptr;
                bin_instance->req.get()[out_container.index] = nullptr;
            }

            if (done_req == LINK_REQUEST_DEL) {
                ENN_WARN_PRINT("_acc_dqbuf : request deleted!!\n");
                return ENN_RET_SUCCESS;
            } else if (done_req == NULL) {
                ENN_WARN_PRINT("_acc_dqbuf : can not found this request!! index : %d\n", out_container.index);
                return ENN_RET_SUCCESS;
            } else if (out_container.flags & (1 << VS4L_CL_FLAG_INVALID)) {
                ENN_ERR_PRINT_FORCE("_acc_dqbuf : output_buf value is invalid!!\n");
                ENN_ERR_PRINT_FORCE("=========================================\n");
                ENN_ERR_PRINT_FORCE("out_container.direction : %d\n", out_container.direction);
                ENN_ERR_PRINT_FORCE("out_container.id : %d\n", out_container.id);
                ENN_ERR_PRINT_FORCE("out_container.index : %d\n", out_container.index);
                ENN_ERR_PRINT_FORCE("out_container.flags : %d\n", out_container.flags);
                ENN_ERR_PRINT_FORCE("out_container.count : %d\n", out_container.count);
                ENN_ERR_PRINT_FORCE("=========================================\n");
                done_req->ret_code = ENN_RET_FAILED;
            } else {
                ENN_DBG_PRINT("done_req->ret_code : success\n");
                _acc_print_execute_log(bin_instance->unique_id, bin_instance->device_fd, true);
                done_req->ret_code = ENN_RET_SUCCESS;
                return ENN_RET_SUCCESS;
            }
        } else {
            ENN_ERR_PRINT_FORCE("output_buf dequeue error occured!!\n");
            return ENN_RET_FAILED;
        }
    }
    return ENN_RET_SUCCESS;
}

/**
 * @brief init graph
 * @details init graph with NCP header's data.
 * @param[in] *graph - vs4l graph data struct
 * @param[in] *user_bin_data - NCP data
 * @param[in] *device_bin_data - additinal datas that needed by just d/d & link
 * @return ENN_RET_SUCCESS: init graph success
 * @return ENN_RET_FAILED: init graph failure
 */
EnnReturn UdLink::_acc_init_graph(struct vs4l_graph* graph, eden_memory_t *bin_mem,
                                  struct drv_usr_share* device_bin_data, uint64_t unified_op_id) {
    if (_CHK_RET_MSG((graph == NULL || bin_mem == NULL || device_bin_data == NULL),
                    "parameter check")) {
        return ENN_RET_FAILED;
    }

    if (bin_mem->ref.ion.fd == 0 && bin_mem->ref.ion.buf == 0) {
        ENN_ERR_PRINT_FORCE("not support filepath anymore when loading NCP\n");
        return ENN_RET_FAILED;
    } else {
        device_bin_data->bin_fd = bin_mem->ref.ion.fd;
        device_bin_data->bin_size = bin_mem->size;
        device_bin_data->bin_mmap = bin_mem->ref.ion.buf;
        device_bin_data->unified_op_id = unified_op_id;

        ENN_DBG_PRINT("bin_fd=[%d], bin_size=[%d], bin_mmap=[%p], unified_op_id=[%lu]\n",
                device_bin_data->bin_fd, device_bin_data->bin_size,
                (eden_addr_t*)(device_bin_data->bin_mmap), (unsigned long) unified_op_id);
    }

    graph->addr = (eden_addr_t)device_bin_data;

    return ENN_RET_SUCCESS;
}

EnnReturn UdLink::link_init(accelerator_device acc, uint32_t max_request_size) {
    uint8_t soc_target;

    ENN_DBG_PRINT("link_init(%d)\n", max_request_size);

    if (dev_state_[acc] >= DEVICE_INITIALIZED) {
        ENN_DBG_PRINT("acc[%d] device already initialized\n", acc);
        return ENN_RET_SUCCESS;
    }

    max_request_size_[acc] = max_request_size;
    flag_sram_full_[acc].store(0);

    /* Initialize */
    map_bin_instance_[acc].clear();

    for (soc_target = 0; soc_target < EXYNOS_MAX; soc_target++) {
        // Nice to have: TODO(jungho7.kim, TBD): remove frequency control logic
        if (target_max_freq[soc_target][PM_SOC_NUM] == 9925) {
            soc_idx_ = soc_target;
        }
    }

    max_cluster_ = SEL_CLUSTER(target_max_freq[soc_idx_][PM_CPU_CL2_MAX]);

    // Nice to have: TODO(mj.kim010, TBD): remove this if possible
    eden_mem_init();

    dev_state_[acc] = DEVICE_INITIALIZED;

    return ENN_RET_SUCCESS;
}


/* Nice to have : TODO(mj.kim010, TBD) : check why we have this func. */
EnnReturn UdLink::link_acc_get_state(accelerator_device acc, dev_state_t* state) {
    ENN_UNUSED(acc);
    *state = DEVICE_INITIALIZED;
    return ENN_RET_SUCCESS;
}


EnnReturn UdLink::allocate_container_list(int cidx, enum vs4l_direction dir, int count, std::shared_ptr<bin_data> bin_instance) {

    vs4l_container_list* clist = nullptr;

    if (dir == VS4L_DIRECTION_IN) {
        clist = bin_instance->in_vs4l_ctl_array.get();
    } else {
        clist = bin_instance->out_vs4l_ctl_array.get();
    }

    if (count == 0) {
        /* We support zero in/out case. */
        ENN_INFO_PRINT("[%d] dir(%d) count is 0. Not allocate user mem for container.\n", cidx, dir);
        clist[cidx].containers = nullptr;
        return ENN_RET_SUCCESS;
    }

    /* Allocate user mem for container. */
    clist[cidx].containers = new struct vs4l_container[count];

    size_t size_buffers = sizeof(struct vs4l_buffer) * bin_instance->tile_size;
    for (int buf_idx = 0; buf_idx < count; buf_idx++) {
        /* Required: TODO(mj.kim010, 6/31) : Not use malloc */
        clist[cidx].containers[buf_idx].buffers = new(std::nothrow) struct vs4l_buffer[bin_instance->tile_size];
        if (clist[cidx].containers[buf_idx].buffers == nullptr) {
            ENN_ERR_PRINT_FORCE("Fail on allocate dir[%d] containers[%d][%d] size(%d)\n",
                                                dir, cidx, buf_idx, size_buffers);
            delete[] clist[cidx].containers;
            return ENN_RET_FAILED;
        }
    }
    return ENN_RET_SUCCESS;
}


EnnReturn UdLink::allocate_sformat_list(struct vs4l_format_list *format_struct,
                                            enum vs4l_direction dir, int count) {
    format_struct->direction = dir;
    format_struct->count = count;
    if (format_struct->count == 0) {
        /* Support zero in/out case. */
        ENN_INFO_PRINT("dir(%d) : SFORMAT count is 0. Not allocate formats mem.\n", dir);
        format_struct->formats = nullptr;
        return ENN_RET_SUCCESS;
    }

    format_struct->formats = new struct vs4l_format[count];
    return ENN_RET_SUCCESS;
}


void UdLink::show_sformat_datas(const enum vs4l_direction dir, const int idx,
                                const shape_t &shape, const struct vs4l_format &format){

    ENN_DBG_PRINT("----------- dir[%d] ----------", dir);
    /* S_FORMAT ingredient */
    ENN_DBG_PRINT("model_info bin_shape[%d].channel=[%d]", idx, shape.channel);
    ENN_DBG_PRINT("model_info bin_shape[%d].number=[%d]", idx, shape.number);
    ENN_DBG_PRINT("model_info bin_shape[%d].width=[%d]", idx, shape.width);
    ENN_DBG_PRINT("model_info bin_shape[%d].height=[%d]", idx, shape.height);
    ENN_DBG_PRINT("model_info bin_shape[%d].type_size=[%d]", idx, shape.type_size);
    /* Generated S_FORMAT data */
    ENN_DBG_PRINT("S_FORMAT data formats[%d].width=[%d]", idx, format.width);
    ENN_DBG_PRINT("S_FORMAT data formats[%d].height=[%d]", idx, format.height);
    ENN_DBG_PRINT("S_FORMAT data formats[%d].channels=[%d]", idx, format.channels);
    ENN_DBG_PRINT("S_FORMAT data formats[%d].pixel_format=[%d]", idx, format.pixel_format);
    return;
}


EnnReturn UdLink::link_open_model(accelerator_device acc, model_info_t* model_info, const EdenModelOptions* options) {
    ENN_DBG_PRINT("(+) acc[%d]\n", acc);
    // Required: TODO(mj.kim010, 6/30): Check acc num is valid

    // check argument: model
    if (model_info == NULL || options == NULL) {
        ENN_ERR_PRINT_FORCE("model_info || options is null\n");
        return ENN_RET_FAILED;
    }

    // check argument: model
    if (model_info->model_addr == NULL) {
        ENN_ERR_PRINT_FORCE("Error, model and model_addr are null! acc(%d)\n", acc);
        return ENN_RET_FAILED;
    }

    if (dev_state_[acc] >= DEVICE_INITIALIZED) {
        ENN_DBG_PRINT("UD Status : DEVICE_INITIALIZED\n");
    } else {
        ENN_ERR_PRINT_FORCE("UD Status : %d\n", dev_state_[acc]);
    }

    // check whether Accelerator sram is full or not
    if (flag_sram_full_[acc].load() == 1) {
        ENN_ERR_PRINT_FORCE("Error, Accelerator SRAM is full!\n");
        return ENN_RET_FAILED;
    }

    /* Nice to have : TODO(mj.kim010, TBD): Check duplicated bin */

    // Load NCP BINARY on ION buffer
    eden_memory_t bin_mem;

    /* Required: TODO(mj.kim010, 6/30) : free when fail.*/
    bin_mem.type = ION;
    bin_mem.size = model_info->model_size;
    uint32_t emaRet = eden_mem_allocate(&bin_mem);
    if (emaRet != PASS) {
        ENN_ERR_PRINT_FORCE("ion buffer alloc failed");
        return ENN_RET_FAILED;
    }
    memcpy((void*)bin_mem.ref.ion.buf, model_info->model_addr, bin_mem.size);


    if (model_info->model_name == 0) {
        ENN_ERR_PRINT_FORCE("model_name is null!");
        return ENN_RET_FAILED;
    }

    // from shutdown
    if (dev_state_[acc] == DEVICE_SHUTDOWNS) {
        ENN_ERR_PRINT_FORCE("The prep thread exit by shutdown\n");
        return ENN_RET_FAILED;
    }

    ENN_INFO_PRINT("acc[%d] model_info->id=0x%" PRIX64 ", name=%s\n",
            acc,  model_info->id, model_info->model_name);

    if (acc == ACCELERATOR_DSP)
        show_ucgo_model_info(model_info);

    ENN_DBG_PRINT("acc[%d] model_info->model_name: %s\n", acc, model_info->model_name);


    ENN_DBG_PRINT("(+)\n");

    ModePreference mode = options->modelPreference.userPreference.mode;

#if !defined(NPU_DD_EMULATOR)
    int32_t device_fd = call_vs4l(VS4L_OPEN, 0, 0, NULL, VS4L_TIMER_TIMEOUT_SEC);
#else
    int32_t device_fd = sizeof(bin_node_name_);
#endif

    ENN_DBG_PRINT("opened device_fd: %d, mode: %d", device_fd, mode);

    if (!_CHK_TRUE_RET_MSG(device_fd > 0, "device_fd open")) {
        ENN_ERR_PRINT_FORCE("Failed to open device_fd : %d %d, %s\n", device_fd, errno, strerror(errno));
        return ENN_RET_FAILED;
    }

    int32_t inbuf_idx = 0;
    int32_t outbuf_idx = 0;
    int32_t cidx = 0;
    int in_fmap_count_final = 0;

    struct vs4l_graph    graph;
    struct drv_usr_share device_bin_data;
    struct vs4l_format_list in_format_struct;
    struct vs4l_format_list out_format_struct;
    struct link_perf_option link_option;

    uint32_t max_req_size = max_request_size_[acc];

    std::shared_ptr<bin_data> bin_instance;
    std::shared_ptr<vs4l_container_list> vs4l_container_in_tmp(
                            new vs4l_container_list[max_req_size],
                            std::default_delete<vs4l_container_list[]>());
    std::shared_ptr<vs4l_container_list> vs4l_container_out_tmp(
                            new vs4l_container_list[max_req_size],
                            std::default_delete<vs4l_container_list[]>());
    std::shared_ptr<int32_t> fd_to_vs4l_index_tmp(
                            new int32_t[max_req_size],
                            std::default_delete<int32_t[]>());
    std::shared_ptr<acc_req_p> req_ptr_array_tmp(
                            new acc_req_p[max_req_size],
                            std::default_delete<acc_req_p[]>());

    uint8_t correction_value;
    int32_t drv_ret = 0;
    EnnReturn ret = ENN_RET_FAILED;
#ifndef ENN_NPU_DSP_DD
    struct vs4l_ctrl ctrl;
    ctrl.ctrl = 0; // not used
    ctrl.value = NPU_HWDEV_NPU;
    if (acc == ACCELERATOR_DSP) {
        ctrl.value = NPU_HWDEV_DSP;
    }

    // Nice to have: TODO(jungho7.kim, TBD): remove goto statement
    if (_CHK_RET_MSG(call_vs4l(VS4L_IOCTL, device_fd,
                    VS4L_VERTEXIOC_BOOTUP, &ctrl, VS4L_TIMER_TIMEOUT_SEC), "bootup")) {
        ENN_ERR_PRINT_FORCE("Fail to BOOTUP. device_ctrl_value(%d)\n", ctrl.value);
        goto _err_link_open_model_with_init_graph;
    }
#endif

    if (_CHK_RET_MSG(_acc_init_graph(&graph, &bin_mem, &device_bin_data,
                    model_info->unified_op_id), "graph init")) {
        goto _err_link_open_model_with_init_graph;
    }

    if (acc == ACCELERATOR_DSP) {
        show_dsp_loadgraph_info((ofi_v4_load_graph_info_t *)bin_mem.ref.ion.buf);
    }

    _dump_ioctl_params(VS4L_VERTEXIOC_S_GRAPH, device_fd, NULL, &graph, NULL);

#if !defined(NPU_DD_EMULATOR)
    if (acc == ACCELERATOR_DSP) {
        struct vs4l_param param;
        struct vs4l_param_list param_list;

        param.addr = (unsigned long)(model_info->kernel_name); // need to find proper casting
        param.size = model_info->kernel_name_size;
        param.offset = model_info->kernel_name_count;
        param.target = NPU_S_PARAM_DSP_KERNEL;

        param_list.count = 1;
        param_list.params = &param;
        // s_param for kernel will be used when vs4l for dsp is ready
        ENN_DBG_PRINT("[IOCTL] VS4L_VERTEXIOC_S_PARAM, FD : %d\n", device_fd);
        ENN_DBG_PRINT("[IOCTL] first kernel string %s\n",
            (model_info->kernel_name + 4*model_info->kernel_name_count));
        drv_ret = call_vs4l(VS4L_IOCTL, device_fd, VS4L_VERTEXIOC_S_PARAM, &param_list, VS4L_TIMER_TIMEOUT_SEC);
        if (drv_ret == ENN_RET_SUCCESS) {
            ENN_DBG_PRINT("succeeded S_PARAM - fd : %d\n", device_fd);
        } else if (drv_ret == EMERGENCY_RECOVERY) {
            ENN_ERR_PRINT_FORCE("EMERGENCY_RECOVERY!! - fd : %d\n", device_fd);
            _CHK_RET_MSG(call_vs4l(VS4L_CLOSE, device_fd, 0, NULL, VS4L_TIMER_TIMEOUT_SEC), "close device_fd");
            return ENN_RET_FAILED;
        } else {  /** in case of fail */
            _CHK_RET_MSG(call_vs4l(VS4L_CLOSE, device_fd, 0, NULL, VS4L_TIMER_TIMEOUT_SEC), "close device_fd");
            if (errno == ERR_LOAD_CANT_ALLOC_CMD_LENGTH || errno == ERR_LOAD_SEQ_ALLOC) {
                ENN_ERR_PRINT_FORCE("DSP SRAM is full! - fd : %d, error code : %d\n", device_fd, errno);
                flag_sram_full_[acc].store(1);
                return ENN_RET_FAILED;
            } else {
                ENN_ERR_PRINT_FORCE("failed S_PARAM - fd : %d, error_code : %d\n", device_fd, errno);
                return ENN_RET_FAILED;
            }
        }
    }

    ENN_DBG_PRINT("[IOCTL] VS4L_VERTEXIOC_S_GRAPH, FD : %d\n", device_fd);
    drv_ret = call_vs4l(VS4L_IOCTL, device_fd, VS4L_VERTEXIOC_S_GRAPH, &graph, VS4L_TIMER_TIMEOUT_SEC);
    if (drv_ret == ENN_RET_SUCCESS) {
        ENN_DBG_PRINT("succeeded S_GRAPH - fd : %d\n", device_fd);
    } else if (drv_ret == EMERGENCY_RECOVERY) {
        ENN_ERR_PRINT_FORCE("EMERGENCY_RECOVERY!! - fd : %d\n", device_fd);
        _CHK_RET_MSG(call_vs4l(VS4L_CLOSE, device_fd, 0, NULL, VS4L_TIMER_TIMEOUT_SEC), "close device_fd");
        return ENN_RET_FAILED;
    } else {  /** in case of fail */
        _CHK_RET_MSG(call_vs4l(VS4L_CLOSE, device_fd, 0, NULL, VS4L_TIMER_TIMEOUT_SEC), "close device_fd");
        if (errno == ERR_LOAD_CANT_ALLOC_CMD_LENGTH || errno == ERR_LOAD_SEQ_ALLOC) {
            ENN_ERR_PRINT_FORCE("NPU SRAM is full! - fd : %d, error code : %d\n", device_fd, errno);
            flag_sram_full_[acc].store(1);
            return ENN_RET_FAILED;
        } else {
            ENN_ERR_PRINT_FORCE("failed S_GRAPH - fd : %d, error_code : %d\n", device_fd, errno);
            return ENN_RET_FAILED;
        }
    }
#endif

    // TODO(jungho7.kim): use set_func()
    bin_instance = std::make_shared<bin_data>();
    bin_instance->req = req_ptr_array_tmp;
    bin_instance->in_vs4l_ctl_array = vs4l_container_in_tmp;
    bin_instance->out_vs4l_ctl_array = vs4l_container_out_tmp;
    bin_instance->fd_to_vs4l_index = fd_to_vs4l_index_tmp;

    bin_instance->device_fd = device_fd;
    bin_instance->vs4l_index = 0;
    bin_instance->bin_mem = bin_mem;
    bin_instance->model_info_id = model_info->id;
    bin_instance->in_fmap_count = model_info->input_count;
    bin_instance->out_fmap_count = model_info->output_count;
    bin_instance->unique_id  = device_bin_data.id;
    bin_instance->bound_core = NPU_BOUND_UNBOUND;
    bin_instance->priority = REQ_PRIORITY_DEFAULT;
    bin_instance->mode = options->modelPreference.userPreference.mode;
    bin_instance->link_mode = NPU_S_PARAM_PERF_MODE_NONE;
    bin_instance->prepared = false;
    bin_instance->tile_size = model_info->tile_size;
    bin_instance->operator_list_id = model_info->operator_list_id;

    for (int32_t i = 0; i < max_req_size; i++) {
        bin_instance->fd_to_vs4l_index.get()[i] = -1;
    }
#if defined(NPU_DD_EMULATOR)
    bin_instance->unique_id  = device_bin_data.id = 1;
#endif
    /* Update in_fmap count for DSP case. */
    in_fmap_count_final = bin_instance->in_fmap_count;
    if (acc == ACCELERATOR_DSP) {
        in_fmap_count_final++;
        ENN_DBG_PRINT("Add IFM count +1 for DSP data struct. (%d)\n", in_fmap_count_final);
    }

    /* Generate contaimner lists for whole executions. */
    ENN_DBG_PRINT("unique_id : %d\n", bin_instance->unique_id);
    for (cidx = 0; cidx < max_req_size; cidx++) {
        EnnReturn ret = allocate_container_list(cidx, VS4L_DIRECTION_IN, in_fmap_count_final, bin_instance);
        if (ret != ENN_RET_SUCCESS) {
            goto _err_link_open_model_with_vs4l_container;
        }

        ret = allocate_container_list(cidx, VS4L_DIRECTION_OT, bin_instance->out_fmap_count, bin_instance);
        if (ret != ENN_RET_SUCCESS) {
            goto _err_link_open_model_with_vs4l_container;
        }
    }

    /* Generate in S_FORMAT datas. */
    ret = allocate_sformat_list(&in_format_struct, VS4L_DIRECTION_IN, in_fmap_count_final);
    if (ret != ENN_RET_SUCCESS) {
        goto _err_link_open_model_set_format;
    }

    for (int i = 0; i < bin_instance->in_fmap_count; i++) {
        in_format_struct.formats[i].target = i;
        if (acc == ACCELERATOR_NPU) {
            in_format_struct.formats[i].format = VS4L_DF_IMAGE_NPU;
        } else if (acc == ACCELERATOR_DSP) {
            in_format_struct.formats[i].format = VS4L_DF_IMAGE_DSP;
        }
        /** plane : do not use */
        in_format_struct.formats[i].plane = 0;

        // Do not cell align if last of the input is Shared Buffer.
        if ((model_info->shared_buffer != NO_SHARED_BUFFER) &&
            (i == (bin_instance->in_fmap_count - 1))) {
            in_format_struct.formats[i].width = model_info->bin_in_shape[i].width;
            in_format_struct.formats[i].height = model_info->bin_in_shape[i].height;
            in_format_struct.formats[i].channels = model_info->bin_in_shape[i].channel;
        } else {
            in_format_struct.formats[i].width =
                _cell_align(model_info->bin_in_shape[i].width,
                model_info->cell_align_shape.width);
            in_format_struct.formats[i].height =
                _cell_align(model_info->bin_in_shape[i].height,
                model_info->cell_align_shape.height);
            in_format_struct.formats[i].channels =
                (_cell_align(model_info->bin_in_shape[i].channel,
                model_info->cell_align_shape.channel) *
                model_info->bin_in_shape[i].number);
        }

        /** stride, cstride, pixel_format : not implement yet */
        in_format_struct.formats[i].stride = 0;
        in_format_struct.formats[i].cstride = 0;
        in_format_struct.formats[i].pixel_format = model_info->bin_in_bpp[i];

        show_sformat_datas(VS4L_DIRECTION_IN, i, model_info->bin_in_shape[i], in_format_struct.formats[i]);
    }
    if (acc == ACCELERATOR_DSP) {
        /* For DSP execute info. */
        int idx_dsp_exec_info = bin_instance->in_fmap_count;
        in_format_struct.formats[idx_dsp_exec_info].target = bin_instance->in_fmap_count;
        in_format_struct.formats[idx_dsp_exec_info].format = VS4L_DF_IMAGE_DSP;
        in_format_struct.formats[idx_dsp_exec_info].plane = 0;
        in_format_struct.formats[idx_dsp_exec_info].width = model_info->get_exec_msg_size();
        in_format_struct.formats[idx_dsp_exec_info].height = 1;
        in_format_struct.formats[idx_dsp_exec_info].channels = 1;
        in_format_struct.formats[idx_dsp_exec_info].stride = 0;
        in_format_struct.formats[idx_dsp_exec_info].cstride = 0;
        in_format_struct.formats[idx_dsp_exec_info].pixel_format = 8;
    }

    /* Generate out S_FORMAT datas. */
    ret = allocate_sformat_list(&out_format_struct, VS4L_DIRECTION_OT, bin_instance->out_fmap_count);
    if (ret != ENN_RET_SUCCESS) {
        goto _err_link_open_model_set_format;
    }
    /*
     * In the Makalu, the in_size in NCP is not correct which should be same as cal_size.
     * That's why the size of channel is multiplied by 4 in the case of the Makalu in order to pass checking size.
     */
    correction_value = 1;
#ifdef VELOCE_SOC
    correction_value = 1;
#endif
    ENN_DBG_PRINT("correction_value=[%d]", correction_value);

    for (int i = 0; i < bin_instance->out_fmap_count; i++) {
        out_format_struct.formats[i].target = i;
        if (acc == ACCELERATOR_NPU) {
            out_format_struct.formats[i].format = VS4L_DF_IMAGE_NPU;
        } else if (acc == ACCELERATOR_DSP) {
            out_format_struct.formats[i].format = VS4L_DF_IMAGE_DSP;
        }
        /** plane : do not use */
        out_format_struct.formats[i].plane = 0;
        out_format_struct.formats[i].width = model_info->bin_out_shape[i].width;
        out_format_struct.formats[i].height = model_info->bin_out_shape[i].height;
        /** stride, cstride, pixel_format : not implement yet */
        out_format_struct.formats[i].stride = 0;
        out_format_struct.formats[i].cstride = 0;
        /* Nice to have: TODO(mj.kim010, TBD) : Need to check if this needed. */
        out_format_struct.formats[i].pixel_format = 8;

        if (model_info->binding_ofm == 0) {
            /** channel align should come from tflite/nnc model for optimized behavior */
           out_format_struct.formats[i].channels =
                _cell_align(model_info->bin_out_shape[i].channel,
                    model_info->cell_align_shape.channel) *
                model_info->bin_out_shape[i].number *
                model_info->bin_out_shape[i].type_size * correction_value;
        } else {
            out_format_struct.formats[i].channels =
                model_info->bin_out_shape[i].channel *
                model_info->bin_out_shape[i].number *
                model_info->bin_out_shape[i].type_size * correction_value;
        }
        show_sformat_datas(VS4L_DIRECTION_OT, i, model_info->bin_out_shape[i], out_format_struct.formats[i]);
    }

#ifdef DUMP_LINK_IOCTL_PARAMS
    _dump_ioctl_params(VS4L_VERTEXIOC_S_FORMAT, 0, NULL, NULL, &in_format_struct);
    _dump_ioctl_params(VS4L_VERTEXIOC_S_FORMAT, 0, NULL, NULL, &out_format_struct);
    _dump_ioctl_params(VS4L_VERTEXIOC_STREAM_ON, device_fd, NULL, NULL, NULL);
#endif

    ENN_DBG_PRINT("[IOCTL] VS4L_VERTEXIOC_S_FORMAT, in_format_struct, FD : %d\n", device_fd);
    drv_ret = call_vs4l(VS4L_IOCTL, device_fd, VS4L_VERTEXIOC_S_FORMAT, &in_format_struct, VS4L_TIMER_TIMEOUT_SEC);
    if (drv_ret == ENN_RET_SUCCESS) {
        ENN_DBG_PRINT("succeeded S_FORMAT - fd : %d\n", device_fd);
    } else {
        ENN_ERR_PRINT_FORCE("failed S_FORMAT - fd : %d\n", device_fd);
        goto _err_link_open_model_set_format;
    }
    ENN_DBG_PRINT("[IOCTL] VS4L_VERTEXIOC_S_FORMAT, out_format_struct, FD : %d\n", device_fd);
    drv_ret = call_vs4l(VS4L_IOCTL, device_fd, VS4L_VERTEXIOC_S_FORMAT, &out_format_struct, VS4L_TIMER_TIMEOUT_SEC);
    if (drv_ret == ENN_RET_SUCCESS) {
        ENN_DBG_PRINT("succeeded S_FORMAT - fd : %d\n", device_fd);
    } else {
        ENN_ERR_PRINT_FORCE("failed S_FORMAT - fd : %d\n", device_fd);
        goto _err_link_open_model_set_format;
    }
    ENN_DBG_PRINT("Call S_FORMAT ioctl...Done\n");

    //  Apply boost and priority
    /**
     * apply performance mode when do open_model.
     */
    set_link_perf_option(&link_option, options->modelPreference.userPreference.mode, options->priority,
            options->latency, options->boundCore, options->presetScenarioId);
    if (boost_open_close(bin_instance, &link_option) == ENN_RET_SUCCESS)
        bin_instance->link_mode = (acc_perf_mode) link_option.mode;

    if (_apply_acc_boundness(bin_instance, link_option.bound, link_option.priority)) {
        bin_instance->bound_core = link_option.bound;
    }

#if !defined(NPU_DD_EMULATOR)
    ENN_DBG_PRINT("[IOCTL] VS4L_VERTEXIOC_STREAM_ON, FD : %d\n", device_fd);
    drv_ret = call_vs4l(VS4L_IOCTL, device_fd, VS4L_VERTEXIOC_STREAM_ON, NULL, VS4L_TIMER_TIMEOUT_SEC);
    if (drv_ret == ENN_RET_SUCCESS) {
        ENN_DBG_PRINT("succeeded STREAM_ON - fd : %d\n", device_fd);
    } else {
        ENN_ERR_PRINT_FORCE("failed STREAM_ON - fd : %d\n", device_fd);
        goto _err_link_open_model_with_stream_on;
    }
    ENN_DBG_PRINT("Call STREAM_ON ioctl...Done\n");
#endif

    delete[] in_format_struct.formats;
    delete[] out_format_struct.formats;

    {
        std::lock_guard<std::mutex> guard(mutex_bin_instance_);
        map_bin_instance_[acc].insert(std::make_pair(bin_instance->unique_id, bin_instance));
    }

    ENN_DBG_PRINT("(-)\n");
    return ENN_RET_SUCCESS;

_err_link_open_model_set_format:
    ENN_ERR_PRINT_FORCE("free in/out format struct | fd : %d\n", device_fd);
    /* Required : TODO(mj.kim010, 6/30) : Not use free(). Use shared_ptr if possible. */
    delete[] in_format_struct.formats;
    delete[] out_format_struct.formats;
_err_link_open_model_with_vs4l_container:
_err_link_open_model_with_stream_on:
    ENN_ERR_PRINT_FORCE("free in/out vs4l_container buffer | fd : %d\n", device_fd);
    for (cidx = 0; cidx < max_req_size; cidx++) {
        // Release input buffer resources
        for (inbuf_idx = 0; inbuf_idx < bin_instance->in_fmap_count; inbuf_idx++) {
            delete[] bin_instance->in_vs4l_ctl_array.get()[cidx].containers[inbuf_idx].buffers;
        }
        /* Required : TODO(mj.kim010, 6/30) : Not use free(). Use shared_ptr if possible. */
        delete[] bin_instance->in_vs4l_ctl_array.get()[cidx].containers;
        // Release output buffer resources
#if !defined(NPU_DD_EMULATOR)
        for (outbuf_idx = 0; outbuf_idx < bin_instance->out_fmap_count; outbuf_idx++) {
            delete[] bin_instance->out_vs4l_ctl_array.get()[cidx].containers[outbuf_idx].buffers;
        }
#endif
        delete[] bin_instance->out_vs4l_ctl_array.get()[cidx].containers;
    }
    ENN_ERR_PRINT_FORCE("free bin_instance | fd : %d\n", device_fd);

_err_link_open_model_with_init_graph:
#if !defined(NPU_DD_EMULATOR)
    ENN_ERR_PRINT_FORCE("try to close device_fd | fd : %d\n", device_fd);
    if (_CHK_RET_MSG(call_vs4l(VS4L_CLOSE, device_fd, 0, NULL, VS4L_TIMER_TIMEOUT_SEC), "close device_fd")) {
        ENN_ERR_PRINT_FORCE("open model failed\n");
    }
#endif

    ENN_ERR_PRINT_FORCE("Release allocated resources for bin_data, Done.\n");

    if (drv_ret == EMERGENCY_RECOVERY) {
        ENN_ERR_PRINT_FORCE("EMERGENCY_RECOVERY!! - fd : %d\n", device_fd);
        /* Required : TODO(mj.kim010, 6/30): Handle emergency recovery */
        return ENN_RET_FAILED;
    } else {  /** in case of fail */
        if (errno == ERR_LOAD_CANT_ALLOC_CMD_LENGTH || errno == ERR_LOAD_SEQ_ALLOC) {
            ENN_ERR_PRINT_FORCE("NPU SRAM is full! - fd : %d, error code : %d\n", device_fd, errno);
            flag_sram_full_[acc].store(0);
            return ENN_RET_FAILED;
        } else {
            ENN_ERR_PRINT_FORCE("failed STREAM_ON - fd : %d, error code : %d\n", device_fd, errno);
            return ENN_RET_FAILED;
        }
    }
}


EnnReturn UdLink::validate_memory_set(const std::shared_ptr<eden_memory_t> mems_shared, uint32_t count) {
    /* Support zero in/out case.*/
    if (count == 0) {
        ENN_INFO_PRINT("No memory. Skip validation.\n");
        return ENN_RET_SUCCESS;
    }

    const eden_memory_t *mems = mems_shared.get();
    mem_t type = mems[0].type;
    // It is enough that checking first input/output data memory type
    if (((type == ION) || (type == EXTERNAL_ION))) {
#if defined(CONFIG_NPU_MEM_ION)
        for (int32_t i = 0; i < count; i++) {
            if (!_CHK_TRUE_RET_MSG((mems[i].ref.ion.fd > 0 && mems[i].ref.ion.buf != 0),
                        "em_input check")) {
                return ENN_RET_FAILED;
            }
        }
#else
        ENN_ERR_PRINT_FORCE("not available ION\n");
        return ENN_RET_FAILED;
#endif
    // It is enough that checking first input/output data memory type
    } else if (type == USER_HEAP) {
        for (int32_t i = 0; i < count; i++) {
            if (!_CHK_TRUE_RET_MSG(mems[i].ref.user_ptr != NULL,
                        "em_input heap check")) {
                return ENN_RET_FAILED;
            }
        }
    } else {
        ENN_ERR_PRINT_FORCE("enn memory type error\n");
        return ENN_RET_FAILED;
    }
    return ENN_RET_SUCCESS;
}


EnnReturn UdLink::validate_in_out(const std::shared_ptr<bin_data> bin_instance,
                            const std::shared_ptr<eden_memory_t> em_inputs,
                            const std::shared_ptr<eden_memory_t> em_outputs) {
        ENN_DBG_PRINT("Validate in memory. cnt(%d)\n", bin_instance->in_fmap_count);
        EnnReturn status = validate_memory_set(em_inputs, bin_instance->in_fmap_count);
        if (status != ENN_RET_SUCCESS) {
            return status;
        }
        ENN_DBG_PRINT("Validate out memory. cnt(%d)\n", bin_instance->in_fmap_count);
        status = validate_memory_set(em_outputs, bin_instance->out_fmap_count);
        if (status != ENN_RET_SUCCESS) {
            return status;
        }
        return status;
}



EnnReturn UdLink::link_prepare_req(accelerator_device acc, model_info_t* model_info,
                            std::shared_ptr<eden_memory_t> em_inputs,
                            std::shared_ptr<eden_memory_t> em_outputs,
                            const eden_memory_t* execute_info) {
    ENN_DBG_PRINT("acc[%d] prepare started\n", acc);
    if (em_inputs.get() == nullptr || em_outputs.get() == nullptr) {
        ENN_WARN_PRINT("Support no in/out case. (%p, %p)\n",
            em_inputs.get(), em_outputs.get());
    }

    EnnReturn status = ENN_RET_SUCCESS;
    int32_t idx = 0;

    std::shared_ptr<bin_data> bin_instance = get_bin_by_model_info_id(acc, model_info->id);
    if (bin_instance == nullptr) {
        ENN_ERR_PRINT_FORCE("Fail to find bin_instance. model_info_id(%d)\n", model_info->id);
        return ENN_RET_FAILED;
    }

    for (idx = 0; idx < max_request_size_[acc]; idx++) {
        int cur_idx = ((bin_instance->vs4l_index) + idx) % max_request_size_[acc];
        if (bin_instance->req.get()[cur_idx] == NULL) {
            {
                std::lock_guard<std::mutex> guard(mutex_bin_instance_);
                bin_instance->vs4l_index = cur_idx;
                ENN_DBG_PRINT("bin_instance->vs4l_index : %d\n", bin_instance->vs4l_index);
            }
            break;
        }
    }
    if (idx == max_request_size_[acc]) {
        ENN_ERR_PRINT_FORCE("vs4l buffer full\n");
        return ENN_RET_FAILED;
    }

    status = validate_in_out(bin_instance, em_inputs, em_outputs);
    if (!_CHK_TRUE_RET_MSG(status == ENN_RET_SUCCESS, "in,out validation at prepare")) {
        return ENN_RET_FAILED;
    }

    /* Nice to have : TODO(mj.kim010, TBD) : Remove redundant code in prepare,execute. */
    vs4l_container_list *in_container_list = &bin_instance->in_vs4l_ctl_array.get()[bin_instance->vs4l_index];
    vs4l_container_list *out_container_list = &bin_instance->out_vs4l_ctl_array.get()[bin_instance->vs4l_index];

    int in_fmap_count_final = bin_instance->in_fmap_count;
    if (acc == ACCELERATOR_DSP) {
        in_fmap_count_final++;
        ENN_DBG_PRINT("Prepare in data increased for DSP execute_info. (%d)\n", in_fmap_count_final);
    }

    in_container_list->direction = VS4L_DIRECTION_IN;
    in_container_list->id        = 0;
    in_container_list->index     = bin_instance->vs4l_index;
    in_container_list->count     = in_fmap_count_final;

    for (int32_t i = 0; i < bin_instance->in_fmap_count; i++) {
        in_container_list->containers[i].type = VS4L_BUFFER_LIST;
        /** @todo target can use as buffer size. consider it later */
        in_container_list->containers[i].target = i;
        in_container_list->containers[i].count  = 1;
        in_container_list->containers[i].memory = VS4L_MEMORY_DMABUF;
#if defined(CONFIG_NPU_MEM_ION)
        in_container_list->containers[i].buffers->m.fd = em_inputs.get()[i].ref.ion.fd;
        in_container_list->containers[i].buffers->reserved = em_inputs.get()[i].ref.ion.buf;
#else
        in_container_list->containers[i].buffers->m.userptr = em_inputs.get()[i].ref.user_ptr;
#endif
    }

    /* For no input case of DSP, execute info container will be 1st in_container. */
    if (acc == ACCELERATOR_DSP) {
        // add execute info for DSP
        int idx_dsp_exec_info = bin_instance->in_fmap_count;
        in_container_list->containers[idx_dsp_exec_info].type   = VS4L_BUFFER_LIST;
        in_container_list->containers[idx_dsp_exec_info].target = bin_instance->in_fmap_count;
        in_container_list->containers[idx_dsp_exec_info].count  = 1;
        in_container_list->containers[idx_dsp_exec_info].memory = VS4L_MEMORY_DMABUF;
        in_container_list->containers[idx_dsp_exec_info].buffers->m.fd = execute_info->ref.ion.fd;
        in_container_list->containers[idx_dsp_exec_info].buffers->reserved = execute_info->ref.ion.buf;
        in_container_list->containers[idx_dsp_exec_info].buffers->roi = (struct vs4l_roi) {0, 0, 0, 0};
    }

    out_container_list->direction = VS4L_DIRECTION_OT;
    out_container_list->id        = 0;
    out_container_list->index     = bin_instance->vs4l_index;
    out_container_list->count     = bin_instance->out_fmap_count;

    for (int32_t i = 0; i < bin_instance->out_fmap_count; i++) {
        out_container_list->containers[i].type   = VS4L_BUFFER_LIST;
        /** @todo target can use as buffer size. consider it later */
        out_container_list->containers[i].target = i;
        out_container_list->containers[i].count  = 1;
        out_container_list->containers[i].memory = VS4L_MEMORY_DMABUF;

#if defined(CONFIG_NPU_MEM_ION)
        out_container_list->containers[i].buffers->m.fd = em_outputs.get()[i].ref.ion.fd;
        out_container_list->containers[i].buffers->reserved = em_outputs.get()[i].ref.ion.buf;
#else
        out_container_list->containers[i].buffers->m.userptr = em_outputs.get()[i]->ref.user_ptr;
#endif
    }

    ENN_DBG_PRINT("link_prepare_req - in out setup done\n");


    int32_t drv_ret = _acc_prepare(bin_instance, in_container_list);
    if (drv_ret == ENN_RET_SUCCESS) {
        ENN_DBG_PRINT("in_container prepare succeeded - unique id : %d\n", bin_instance->unique_id);
    } else if (drv_ret == EMERGENCY_RECOVERY) {
        /* Required : TODO(mj.kim010, 6/30): Handle emergency recovery */
        ENN_ERR_PRINT_FORCE("EMERGENCY_RECOVERY!! %d\n", bin_instance->unique_id);
        return ENN_RET_FAILED;
    } else {
        ENN_ERR_PRINT_FORCE("in_container prepare failed - unique id : %d\n", bin_instance->unique_id);
        return ENN_RET_FAILED;
    }

    drv_ret = _acc_prepare(bin_instance, out_container_list);
    if (drv_ret == ENN_RET_SUCCESS) {
        ENN_DBG_PRINT("out_container prepare succeeded - unique id : %d\n", bin_instance->unique_id);
    } else if (drv_ret == EMERGENCY_RECOVERY) {
        /* Required : TODO(mj.kim010, 6/30): Handle emergency recovery */
        ENN_ERR_PRINT_FORCE("EMERGENCY_RECOVERY!! %d\n", bin_instance->unique_id);
        return ENN_RET_FAILED;
    } else {
        ENN_ERR_PRINT_FORCE("out_container prepare failed - unique id : %d\n", bin_instance->unique_id);
        return ENN_RET_FAILED;
    }


    {
        std::lock_guard<std::mutex> guard(mutex_bin_instance_);
        bin_instance->prepared = true;
        /* Normally 1st IFM is used, but
        * for no input case of DSP, execute info ionFD will be 1st in-ionFD as a key here. */
        bin_instance->fd_to_vs4l_index.get()[bin_instance->vs4l_index] = in_container_list->containers[0].buffers->m.fd;
        bin_instance->vs4l_index = ((bin_instance->vs4l_index) + 1) % max_request_size_[acc];
    }

    ENN_DBG_PRINT("(-)\n");
    return ENN_RET_SUCCESS;
}

inline uint32_t UdLink::link_generate_frame_id(void) {
    std::lock_guard<std::mutex> guard(mutex_frame_id_);
    return frame_id_++;
}

EnnReturn UdLink::link_execute_req(accelerator_device acc, req_info_t* req_info, const EdenRequestOptions* options) {
// EnnReturn UdLink::link_execute_req(accelerator_device acc, struct acc_req* req_data) {
    ENN_DBG_PRINT("acc[%d] started\n", acc);

    // argument check
    if (unlikely(req_info == NULL || options == NULL)) {
        ENN_ERR_PRINT_FORCE("req_info || options is null\n");
        return ENN_RET_FAILED;
    }

    if (likely(dev_state_[acc] >= DEVICE_INITIALIZED)) {
        ENN_DBG_PRINT("UD Status : DEVICE_INITIALIZED\n");
    } else {
        ENN_ERR_PRINT_FORCE("UD Status : %d\n", dev_state_[acc]);
        return ENN_RET_FAILED;
    }

    std::shared_ptr<bin_data> bin_instance = get_bin_by_model_info_id(acc, req_info->model_info->id);
    if (bin_instance == nullptr) {
        ENN_ERR_PRINT_FORCE("Fail to find bin_instance. model_info_id(%d)\n", req_info->model_info->id);
        return ENN_RET_FAILED;
    }

    struct link_perf_option link_option;
    set_link_perf_option(&link_option, bin_instance->mode, REQ_PRIORITY_DEFAULT, 0, NPU_UNBOUND, 0);
    if (boost_execution(bin_instance, &link_option) == ENN_RET_SUCCESS)
        bin_instance->link_mode = (acc_perf_mode) link_option.mode;

    /* Nice to have : TODO(mj.kim, TBD) : Use req_info_t directly */
    struct acc_req req_local;
    // mapping model - bin_id
    // TODO(jungho7.kim): use req_local() or set_func()
    req_local.bin_id = bin_instance->unique_id;
    req_local.req_info = req_info;
    req_local.state = REQ_QUEUED;
    req_local.time_check = 0;
    req_local.ret_code = 0;
    req_local.options = *options;

    uint32_t frame_id_local = link_generate_frame_id();
    ENN_INFO_PRINT("link_frame_id:%u\n", frame_id_local);

    EnnReturn status = ENN_RET_SUCCESS;
    bool check_fd = false;
    vs4l_container_list* in_container_list = nullptr;
    vs4l_container_list* out_container_list = nullptr;

    if (likely(bin_instance->prepared == true)) {
        int32_t idx;
        for (idx = 0; idx < max_request_size_[acc]; idx++) {
            int32_t key_fd = -1;
            if (bin_instance->in_fmap_count > 0) {
                key_fd = req_local.req_info->inputs.get()[0].ref.ion.fd;
            }
            else if (acc == ACCELERATOR_DSP) { /* input==0, DSP */
                /* Memory for DSP execute info is the only input container. */
                key_fd = req_local.req_info->execute_info->ref.ion.fd;
            }
            else {
                ENN_ERR_PRINT_FORCE("Wrong input count(%d). acc[%d]\n",
                                        bin_instance->in_fmap_count, acc);
                return ENN_RET_FAILED;
            }

            if (bin_instance->fd_to_vs4l_index.get()[idx] == key_fd) {
                check_fd = true;
                {
                    std::lock_guard<std::mutex> guard(mutex_bin_instance_);
                    bin_instance->vs4l_index = idx;
                    /* This will be used at __link_req_done() which called in this function. */
                    bin_instance->req.get()[idx] = &req_local;

                    in_container_list = &bin_instance->in_vs4l_ctl_array.get()[idx];
                    out_container_list = &bin_instance->out_vs4l_ctl_array.get()[idx];

                    ENN_DBG_PRINT("key_fd for searching prepared container: %d\n", key_fd);
                    ENN_DBG_PRINT("bin_instance->vs4l_index : %d\n", bin_instance->vs4l_index);
                }
                break;
            }
        }
        if (idx == max_request_size_[acc]) {
            ENN_ERR_PRINT_FORCE("vs4l buffer full\n");
            return ENN_RET_FAILED;
        }
        ENN_DBG_PRINT("Success to find matched containers! (in_container_list=%p, out_container_list=%p)\n",
                        in_container_list, out_container_list);
    } else { /* Non-prepared case */
        check_fd = true;
        int32_t idx;
        for (idx = 0; idx < max_request_size_[acc]; idx++) {
            int idx_calculated = ((bin_instance->vs4l_index) + idx) % max_request_size_[acc];
            if (bin_instance->req.get()[idx_calculated] == NULL) {
                {
                    std::lock_guard<std::mutex> guard(mutex_bin_instance_);
                    bin_instance->vs4l_index = idx_calculated;
                    ENN_DBG_PRINT("bin_instance->vs4l_index : %d\n", bin_instance->vs4l_index);
                    bin_instance->req.get()[bin_instance->vs4l_index] = &req_local;
                }
                break;
            }
        }
        if (idx == max_request_size_[acc]) {
            ENN_ERR_PRINT_FORCE("vs4l buffer full\n");
            return ENN_RET_FAILED;
        }
        ENN_DBG_PRINT("Success to revise vs4l_index(%d)!\n", bin_instance->vs4l_index);
    }
    if (check_fd == false) {
        ENN_ERR_PRINT_FORCE("Fail to validate fd. (%d)\n", bin_instance->unique_id);
        return ENN_RET_FAILED;
    }

    /* Handle non-prepare case */
    if (unlikely(bin_instance->prepared != true)) {
        ENN_DBG_PRINT("[%d] Not prepared yet. Start to set container.\n", acc);
        const std::shared_ptr<eden_memory_t> em_inputs = req_local.req_info->inputs;
        const std::shared_ptr<eden_memory_t> em_outputs = req_local.req_info->outputs;

        status = validate_in_out(bin_instance, em_inputs, em_outputs);
        if (!_CHK_TRUE_RET_MSG(status == ENN_RET_SUCCESS, "in,out validation at execute")) {
            return ENN_RET_FAILED;
        }

        in_container_list = &bin_instance->in_vs4l_ctl_array.get()[bin_instance->vs4l_index];
        out_container_list = &bin_instance->out_vs4l_ctl_array.get()[bin_instance->vs4l_index];

        /* non-prepare : Setup in container */
        int in_fmap_count_final = bin_instance->in_fmap_count;
        if (acc == ACCELERATOR_DSP) {
            in_fmap_count_final++;
            ENN_DBG_PRINT("Execute in data increased for DSP execute_info. (%d)\n", in_fmap_count_final);
        }
        in_container_list->direction = VS4L_DIRECTION_IN;
        in_container_list->id        = frame_id_local;
        in_container_list->index     = bin_instance->vs4l_index;
        in_container_list->count     = in_fmap_count_final;
        ENN_DBG_PRINT("in fmap : %d, out fmap: %d, tile : %d\n",bin_instance->in_fmap_count,
                                            bin_instance->out_fmap_count, bin_instance->tile_size);

        for (int32_t i = 0; i < bin_instance->in_fmap_count; i++) {
            in_container_list->containers[i].type   = VS4L_BUFFER_LIST;
            /** @todo target can use as buffer size. consider it later */
            in_container_list->containers[i].target = i;
            in_container_list->containers[i].count  = bin_instance->tile_size;
            in_container_list->containers[i].memory = VS4L_MEMORY_DMABUF;

#if defined(CONFIG_NPU_MEM_ION)
            const eden_memory_t &em_in = em_inputs.get()[i];
            if (1 < bin_instance->tile_size) {
                in_container_list->containers[i].type = VS4L_BUFFER_ROI;
                int fd = em_in.ref.ion.fd;
                int size = em_in.size;
                uint64_t buf = em_in.ref.ion.buf;
                ENN_INFO_PRINT("ROI IN container[%d].type=%d, fd=%d, size*tile=%d * %d",
                    i, in_container_list->containers[i].type, fd, size, bin_instance->tile_size);

                for (int32_t j = 0; j < bin_instance->tile_size; j++) {
                    in_container_list->containers[i].buffers[j].m.fd = fd;
                    in_container_list->containers[i].buffers[j].reserved = buf;
                    in_container_list->containers[i].buffers[j].roi = (struct vs4l_roi) {0, (unsigned int) (j * size), 1, (unsigned int) size};

                    ENN_DBG_PRINT("ROI IN containers[%d].buffers[%d]={fd=%d, roi={%d, %d, %d, %d}}\n", i, j,
                                    in_container_list->containers[i].buffers[j].m.fd,
                                    in_container_list->containers[i].buffers[j].roi.x,
                                    in_container_list->containers[i].buffers[j].roi.y,
                                    in_container_list->containers[i].buffers[j].roi.w,
                                    in_container_list->containers[i].buffers[j].roi.h);
                }
            } else { /* !(1 < bin_instance->tile_size) */
                in_container_list->containers[i].buffers->m.fd = em_in.ref.ion.fd;
                in_container_list->containers[i].buffers->reserved = em_in.ref.ion.buf;
                in_container_list->containers[i].buffers->roi = (struct vs4l_roi) {0, 0, 0, 0};
            }
#else
            in_container_list->containers[i].buffers->m.userptr = em_in.ref.user_ptr;
            in_container_list->containers[i].buffers->roi = (struct vs4l_roi) {0, 0, 0, 0};
#endif
        }

        if (acc == ACCELERATOR_DSP) {
            int idx_dsp_exec_info = bin_instance->in_fmap_count;
            in_container_list->containers[idx_dsp_exec_info].type   = VS4L_BUFFER_LIST;
            in_container_list->containers[idx_dsp_exec_info].target = bin_instance->in_fmap_count;
            in_container_list->containers[idx_dsp_exec_info].count  = bin_instance->tile_size;
            in_container_list->containers[idx_dsp_exec_info].memory = VS4L_MEMORY_DMABUF;
            if (1 < bin_instance->tile_size) {
                in_container_list->containers[idx_dsp_exec_info].type = VS4L_BUFFER_ROI;
                int fd = req_local.req_info->execute_info->ref.ion.fd;
                int size = req_local.req_info->execute_info->size;
                uint64_t buf = req_local.req_info->execute_info->ref.ion.buf;
                for (int32_t j = 0; j < bin_instance->tile_size; j++) {
                    in_container_list->containers[idx_dsp_exec_info].buffers[j].m.fd = fd;
                    in_container_list->containers[idx_dsp_exec_info].buffers[j].reserved = buf;
                    in_container_list->containers[idx_dsp_exec_info].buffers[j].roi = (struct vs4l_roi) {0, (unsigned int) (j * size), 1, (unsigned int) size};
                }
            } else {
                in_container_list->containers[idx_dsp_exec_info].buffers->m.fd = req_local.req_info->execute_info->ref.ion.fd;
                in_container_list->containers[idx_dsp_exec_info].buffers->reserved = req_local.req_info->execute_info->ref.ion.buf;
                in_container_list->containers[idx_dsp_exec_info].buffers->roi = (struct vs4l_roi) {0, 0, 0, 0};
            }
        }

        /* non-prepare : Setup out container */
        out_container_list->direction = VS4L_DIRECTION_OT;
        out_container_list->id        = frame_id_local;
        out_container_list->index     = bin_instance->vs4l_index;
        out_container_list->count     = bin_instance->out_fmap_count;
        // when we need to deliver one buffer even model has several outputs,
        // buffer is allocated in 1 buffer with sum of several outputs' sizes.
        // so we should change "out_fmap_count" to "1" , which represent Combined all output buffers at 1 buffer.
        // and "req->req_info->inputs[0].ref.ion.fd" should be checked !
        for (int32_t i = 0; i < bin_instance->out_fmap_count; i++) {
            out_container_list->containers[i].type   = VS4L_BUFFER_LIST;
            /** @todo target can use as buffer size. consider it later */
            out_container_list->containers[i].target = i;
            out_container_list->containers[i].count  = bin_instance->tile_size;
            out_container_list->containers[i].memory = VS4L_MEMORY_DMABUF;

#if defined(CONFIG_NPU_MEM_ION)
            const eden_memory_t &em_out = em_outputs.get()[i];
            if (1 < bin_instance->tile_size) {
                out_container_list->containers[i].type = VS4L_BUFFER_ROI;
                int fd = em_out.ref.ion.fd;
                int size = em_out.size;
                uint64_t buf = em_out.ref.ion.buf;
                ENN_INFO_PRINT("ROI OUT container[%d].type=%d, fd=%d, size*tile=%d * %d",
                    i, out_container_list->containers[i].type, fd, size, bin_instance->tile_size);

                for (int32_t j = 0; j < bin_instance->tile_size; j++) {
                    out_container_list->containers[i].buffers[j].m.fd = fd;
                    out_container_list->containers[i].buffers[j].reserved = buf;
                    out_container_list->containers[i].buffers[j].roi = (struct vs4l_roi) {0, (unsigned int) (j * size), 1, (unsigned int) size};

                    ENN_DBG_PRINT("ROI OUT containers[%d].buffers[%d]={fd=%d, roi={%d, %d, %d, %d}}\n", i, j,
                                    out_container_list->containers[i].buffers[j].m.fd,
                                    out_container_list->containers[i].buffers[j].roi.x,
                                    out_container_list->containers[i].buffers[j].roi.y,
                                    out_container_list->containers[i].buffers[j].roi.w,
                                    out_container_list->containers[i].buffers[j].roi.h);
                }
            } else {
                out_container_list->containers[i].buffers->m.fd = em_out.ref.ion.fd;
                out_container_list->containers[i].buffers->reserved = em_out.ref.ion.buf;
                out_container_list->containers[i].buffers->roi = (struct vs4l_roi) {0, 0, 0, 0};
            }
#else
            out_container_list->containers[i].buffers->m.userptr = em_out.ref.user_ptr;
            out_container_list->containers[i].buffers->roi = (struct vs4l_roi) {0, 0, 0, 0};
#endif
        }
    } else { /* prepared case */
        ENN_DBG_PRINT("acc[%d] UD prepared. Use prepared one for execute.\n", acc);
        in_container_list->id = frame_id_local;
        out_container_list->id = frame_id_local;
        gettimeofday(&in_container_list->timestamp[0], NULL);
        out_container_list->timestamp[0].tv_usec = 0;
        out_container_list->timestamp[0].tv_sec = 0;
    }

    ENN_DBG_PRINT("link_execute_req - in out setup done\n");
    ENN_DBG_PRINT("in_ctn_addr id: %d, out_ctn_addr id : %d", in_container_list->id, out_container_list->id);

    if (acc == ACCELERATOR_DSP) {
        const struct vs4l_container &exec_msg_container = in_container_list->containers[bin_instance->in_fmap_count];
        /* debug show function */
        show_dsp_exec_info((ofi_v4_execute_msg_info_t*)exec_msg_container.buffers->reserved);
    }

    /* Check timeout while executing ioctl(QBUF) and ioctl(DQBUF) calls */
    struct vs4l_timer_arg timer_arg;

    if (unlikely(start_vs4l_timer(&timer_arg, VS4L_IOCTL, VS4L_VERTEXIOC_QBUF, VS4L_TIMER_TIMEOUT_SEC)) != ENN_RET_SUCCESS) {
        ENN_ERR_PRINT_FORCE("fail to start_vs4l_timer()\n");
        return ENN_RET_FAILED;
    }

    if (unlikely(setjmp(timer_arg.env) != 0)) {
        char vs4l_type[MAX_LEN_VS4L_TYPE];

        if (unlikely(get_vs4l_type(vs4l_type, timer_arg.type, timer_arg.request) != ENN_RET_SUCCESS)) {
            ENN_ERR_PRINT_FORCE("VS4L ioctl(QBUF) or ioctl(DQBUF) Timeout Error! timeout:%u\n",
                    timer_arg.target_timeout);
        } else {
            ENN_ERR_PRINT_FORCE("VS4L %s Timeout Error! timeout:%u\n",
                    vs4l_type, timer_arg.target_timeout);
        }
        stop_vs4l_timer(&timer_arg);
        return ENN_RET_FAILED;
    }

#ifdef EXYNOS_NN_PROFILER
    if (acc == ACCELERATOR_NPU) {
        // Nice to have: TODO(sahil.sharma, TBD): dynamically accept id to identify a sequence to profile.
        PROFILE_FROM("Device_Driver_NPU_Execution", enn::util::chop_into_model_id(req_info->operator_list_id));

        PROFILE_EXCLUDE_FROM("VS4L_VERTEXIOC_PROFILE_ON", enn::util::chop_into_model_id(req_info->operator_list_id));
        struct vs4l_profiler* profiler;
        if (_allocate_profiler_struct(&profiler) != ENN_RET_SUCCESS) {
            ENN_ERR_PRINT_FORCE("Failed to allocate the profiler struct!\n");
        }
        ENN_DBG_PRINT("[IOCTL] VS4L_VERTEXIOC_PROFILE_ON, FD : %d\n", bin_instance->device_fd);
        if (call_vs4l(VS4L_IOCTL, bin_instance->device_fd, VS4L_VERTEXIOC_PROFILE_ON, profiler, VS4L_TIMER_TIMEOUT_SEC)) {
            ENN_ERR_PRINT_FORCE("Failed to turn on the profiler!\n");
    }
    // free the memory of profiler sturct allocated
    _release_profiler_struct(&profiler);
    PROFILE_EXCLUDE_UNTIL("VS4L_VERTEXIOC_PROFILE_ON", enn::util::chop_into_model_id(req_info->operator_list_id));
    } else if (acc == ACCELERATOR_DSP) {
        //PROFILE_FROM("Device_Driver_DSP_Execution", enn::util::chop_into_model_id(req_info->operator_list_id));
    }
#endif  // EXYNOS_NN_PROFILER

    int32_t drv_ret = _acc_qbuf(bin_instance, in_container_list);
    if (likely(drv_ret == ENN_RET_SUCCESS)) {
        ENN_DBG_PRINT("in_container qbuf succeeded - unique id : %d\n", bin_instance->unique_id);
    } else if (drv_ret == EMERGENCY_RECOVERY) {
        /* Required : TODO(mj.kim010, 6/30): Handle emergency recovery */
        ENN_ERR_PRINT_FORCE("EMERGENCY_RECOVERY!! %d\n", bin_instance->unique_id);
        stop_vs4l_timer(&timer_arg);
        return ENN_RET_FAILED;
    } else {
        ENN_ERR_PRINT_FORCE("in_container qbuf failed - unique id : %d\n", bin_instance->unique_id);
        stop_vs4l_timer(&timer_arg);
        return ENN_RET_FAILED;
    }

    drv_ret = _acc_qbuf(bin_instance, out_container_list);
    if (likely(drv_ret == ENN_RET_SUCCESS)) {
        ENN_DBG_PRINT("out_container qbuf succeeded - unique id : %d\n", bin_instance->unique_id);
    } else if (drv_ret == EMERGENCY_RECOVERY) {
        /* Required : TODO(mj.kim010, 6/30): Handle emergency recovery */
        ENN_ERR_PRINT_FORCE("EMERGENCY_RECOVERY!! %d\n", bin_instance->unique_id);
        stop_vs4l_timer(&timer_arg);
        return ENN_RET_FAILED;
    } else {
        ENN_ERR_PRINT_FORCE("out_container qbuf failed - unique id : %d\n", bin_instance->unique_id);
        stop_vs4l_timer(&timer_arg);
        return ENN_RET_FAILED;
    }

    update_vs4l_arg_request(&timer_arg, VS4L_VERTEXIOC_DQBUF);

    {
        std::lock_guard<std::mutex> guard(mutex_bin_instance_);
        if (unlikely(bin_instance->prepared != true)) {
            bin_instance->vs4l_index = ((bin_instance->vs4l_index) + 1) % max_request_size_[acc];
        }
        ENN_DBG_PRINT("link_execute_req done\n");
        _acc_print_execute_log(bin_instance->unique_id, bin_instance->device_fd, false);
    }
    if (unlikely(_CHK_RET_MSG(__link_req_done(acc, bin_instance->device_fd, &timer_arg), "dequeue for target fd"))) {
        stop_vs4l_timer(&timer_arg);
        return ENN_RET_FAILED;
    }

#if defined(NPU_DD_EMULATOR)
    uint32_t done_check = true;
    while (done_check) {
        if (_CHK_RET_MSG(__link_req_done(acc, bin_instance->device_fd, &timer_arg),
                        "dequeue for target fd")) {
            ENN_ERR_PRINT_FORCE("wait polling - fd : %d\n", bin_instance->device_fd);
        } else {
            ENN_DBG_PRINT("polling done. fd : %d\n", bin_instance->device_fd);
            done_check = false;
        }
    }
#endif
    stop_vs4l_timer(&timer_arg);

    set_link_perf_option(&link_option, NORMAL_MODE, REQ_PRIORITY_DEFAULT, 0, NPU_UNBOUND, 0);
    if (boost_execution(bin_instance, &link_option) == ENN_RET_SUCCESS)
        bin_instance->link_mode = (acc_perf_mode) link_option.mode;

    ENN_DBG_PRINT("(-)\n");
    return ENN_RET_SUCCESS;
}

EnnReturn UdLink::link_shutdown(accelerator_device acc) {
    ENN_DBG_PRINT("acc[%d] started\n", acc);

    if (_CHK_TRUE_RET_MSG(
                          dev_state_[acc] >= DEVICE_INITIALIZED,
                          "g_acc_dev_state[acc] >= DEVICE_INITIALIZED")) {
    } else {
        ENN_ERR_PRINT_FORCE("acc[%d] Incorrect shutdown() call to state: %d\n", dev_state_[acc], acc);
        return ENN_RET_SUCCESS;
    }
    dev_state_[acc] = DEVICE_SHUTDOWNS;

    std::vector<uint64_t> bin_id_to_close;
    {
        std::lock_guard<std::mutex> guard(mutex_bin_instance_);
        for (auto &iter: map_bin_instance_[acc]) {
                std::shared_ptr<bin_data> bin_instance = iter.second;
                if (bin_instance) {
                    bin_id_to_close.push_back(bin_instance->model_info_id);
                }
        }
    }

    for (auto &iter: bin_id_to_close) {
        link_close_model(acc, iter);
    }

    dev_state_[acc] = DEVICE_UNKNOWN;

    // Nice to have: TODO(mj.kim010, TBD): remove this if possible
    eden_mem_shutdown();

    ENN_DBG_PRINT("(-)\n");
    return ENN_RET_SUCCESS;
}

EnnReturn UdLink::link_get_dd_session_id(accelerator_device acc,
                                            const uint64_t model_id, int32_t &session_id) {
    if (acc != ACCELERATOR_DSP) {
        ENN_ERR_PRINT_FORCE("Session ID is currently used only by DSP.\n");
        return ENN_RET_FAILED;
    }

    if (dev_state_[acc] < DEVICE_INITIALIZED) {
        ENN_ERR_PRINT_FORCE("Accelerator[%d] not initialized.\n", acc);
        return ENN_RET_FAILED;
    }

    std::shared_ptr<bin_data> bin_instance = get_bin_by_model_info_id(acc, model_id);
    if (bin_instance.get() == nullptr) {
        ENN_ERR_PRINT_FORCE("Fail to find bin_instance. model_id(0x%" PRIx64 ")\n", model_id);
        return ENN_RET_FAILED;
    }

    session_id = (int32_t)bin_instance->get_unique_id();
    ENN_INFO_PRINT_FORCE("DD returns session_id(%d) for model_id(0x%" PRIx64 ")\n",
                            session_id, model_id);

    return ENN_RET_SUCCESS;
}


// EnnReturn UdLink::link_close_model(accelerator_device acc, uint32_t bin_id) {
EnnReturn UdLink::link_close_model(accelerator_device acc, uint64_t model_id) {
    ENN_DBG_PRINT("acc[%d] started model_id=[%lu]\n", acc, model_id);

    if (model_id == 0) {
        ENN_ERR_PRINT_FORCE("model id is zero\n");
        return ENN_RET_FAILED;
    }

    if (dev_state_[acc] < DEVICE_INITIALIZED) {
        ENN_ERR_PRINT_FORCE("Accelerator not initialized\n");
        return ENN_RET_FAILED;
    }

    EnnReturn status = ENN_RET_FAILED;

    std::shared_ptr<bin_data> bin_instance = get_bin_by_model_info_id(acc, model_id);
    if (bin_instance == nullptr) {
        ENN_ERR_PRINT_FORCE("Fail to find bin_instance. model_info_id(%d)\n", model_id);
        return ENN_RET_FAILED;
    }

    // TODO(Jungho): Make it a func() and apply this to Open() as well
    struct link_perf_option link_option;
    set_link_perf_option(&link_option, NORMAL_MODE, REQ_PRIORITY_DEFAULT, 0, NPU_UNBOUND, 0);
    if (boost_open_close(bin_instance, &link_option) == ENN_RET_SUCCESS)
        bin_instance->link_mode = (acc_perf_mode) link_option.mode;

    /** ioctl for stream off and close */
    _dump_ioctl_params(VS4L_VERTEXIOC_STREAM_OFF, bin_instance->device_fd, NULL, NULL, NULL);

#if !defined(NPU_DD_EMULATOR)
    ENN_DBG_PRINT("[IOCTL] VS4L_VERTEXIOC_STREAM_OFF, FD : %d\n", bin_instance->device_fd);
    status = (EnnReturn) call_vs4l(VS4L_IOCTL, bin_instance->device_fd,
            VS4L_VERTEXIOC_STREAM_OFF, NULL, VS4L_TIMER_TIMEOUT_SEC);
    if (status == ENN_RET_SUCCESS) {
        ENN_DBG_PRINT("succeeded stream off ioctl - fd : %d\n", bin_instance->device_fd);
    } else if (status == EMERGENCY_RECOVERY) {
        ENN_DBG_PRINT("succeeded stream off ioctl(EMERGENCY_RECOVERY) - fd : %d\n",
                    bin_instance->device_fd);
    } else {
        ENN_ERR_PRINT_FORCE("failed stream off ioctl - fd : %d, going to close fd\n", bin_instance->device_fd);
    }

    /** when closing fd, DeviceDriver is going to do stream_off */
    status = ENN_RET_SUCCESS;
    if (_CHK_RET_MSG(call_vs4l(VS4L_CLOSE, bin_instance->device_fd, 0, NULL, VS4L_TIMER_TIMEOUT_SEC), "device_fd closed")) {
        ENN_ERR_PRINT_FORCE("failed close ioctl - fd : %d\n", bin_instance->device_fd);
        status = ENN_RET_FAILED;
    }
#endif

    ENN_INFO_PRINT("stream_off and close device_fd=[%d] is done.\n", bin_instance->device_fd);

    /* Now release bin_instance and its related resources */
    {
        std::lock_guard<std::mutex> guard(mutex_bin_instance_);
        map_bin_instance_[acc].erase(bin_instance->unique_id);
    }

    uint32_t in_fmap_count_final = bin_instance->in_fmap_count;
    if (acc == ACCELERATOR_DSP)
        in_fmap_count_final++;
    /* Required : TODO(mj.kim010, 6/30) : Not use free(). Use shared_ptr if possible.. */
    for (int32_t idx = 0; idx < max_request_size_[acc]; idx++) {
        /* in container */
        release_container(bin_instance->in_vs4l_ctl_array, in_fmap_count_final, idx);
        /* out container */
        release_container(bin_instance->out_vs4l_ctl_array, bin_instance->out_fmap_count, idx);
    }
    eden_mem_free(&bin_instance->bin_mem);

    ENN_DBG_PRINT("(-)\n");
    return status;
}

EnnReturn UdLink::set_link_perf_option(struct link_perf_option* link_option, ModePreference mode, uint32_t priority,
        uint32_t latency, uint32_t bound, uint32_t preset_id) {
    if (unlikely(link_option == nullptr)) {
        ENN_ERR_PRINT_FORCE("link_perf_option nullptr error\n");
        return ENN_RET_FAILED;
    }

    if (unlikely(priority >= REQ_PRIORITY_MAX)) {
        ENN_ERR_PRINT_FORCE("invalid priority(%u)\n", priority);
        return ENN_RET_FAILED;
    }

    ENN_DBG_PRINT("mode:%d, priority:%u, latency:%u, bound:%x, preset_id:%u\n",
            (int) mode, priority, latency, bound, preset_id);

    switch (mode) {
        case NORMAL_MODE:
            link_option->mode = NPU_S_PARAM_PERF_MODE_NONE;
            break;
        case BOOST_MODE:
            link_option->mode = NPU_S_PARAM_PERF_MODE_NPU_BOOST;
            break;
        case BOOST_ON_EXECUTE_MODE:
            link_option->mode = NPU_S_PARAM_PERF_MODE_NPU_BOOST_ON_EXECUTE;
            break;
        case BENCHMARK_MODE: // deprecated mode
            link_option->mode = NPU_S_PARAM_PERF_MODE_NONE;
            break;
        case RESERVED: // deprecated mode
            link_option->mode = NPU_S_PARAM_PERF_MODE_NONE;
            break;
        case BOOST_AUX: // deprecated mode
            link_option->mode = NPU_S_PARAM_PERF_MODE_NONE;
            break;
        case BOOST_BLOCKING_MODE:
            link_option->mode = NPU_S_PARAM_PERF_MODE_NPU_BOOST_BLOCKING;
            break;
        default:
            ENN_ERR_PRINT_FORCE("invalid mode(%d)\n", (int) mode);
            return ENN_RET_FAILED;
    }

    link_option->priority = priority;
    link_option->latency = latency;
    link_option->bound = bound;
    link_option->preset_id = preset_id;

    return ENN_RET_SUCCESS;
}

void UdLink::release_container(std::shared_ptr<vs4l_container_list> vs4l_container,
                                            int32_t fmap_count, int32_t idx) {
    for (int32_t buf_idx = 0; buf_idx < fmap_count; buf_idx++) {
        delete[] vs4l_container.get()[idx].containers[buf_idx].buffers;
    }
    delete[] vs4l_container.get()[idx].containers;
}

/* Nice to have : TODO(mj.kim010, TBD) : Enable emergency recovery with D/D */
#if 0
static EnnReturn _emergency_recovery(accelerator_device acc) {
    ENN_DBG_PRINT("acc[%d] wait for finishing polling\n", acc);
    usleep(TIME_SEC);  // wait for 1 sec

    ENN_DBG_PRINT("reload all bin\n");

    { // RAII mutex
        std::lock_guard<std::mutex> guard(g_mutex_acc_req_list[acc]);
        struct acc_bin* bin = nullptr;
        { // RAII mutex
            std::lock_guard<std::mutex> guard(g_mutex_binary_list[acc]);
            TAILQ_FOREACH(bin, &g_list_acc_binary[acc], bin_entries) {
                bin->state = NCP_UNLOADED;
                memcpy((void*)bin->bin_data.ref.ion.buf, bin->model_info->model_addr,
                            bin->model_info->model_size);
                ENN_DBG_PRINT("bin memcpy done\n");
            }
            if (_CHK_RET_MSG(udlink.link_shutdown(acc), "udlink.link_shutdown(EMERGENCY)")) {
                return ENN_RET_FAILED;
            }
            if (_CHK_RET_MSG(udlink.link_init(acc, g_max_request_size[acc]), "udlink.link_acc_init(EMERGENCY)")) {
                return ENN_RET_FAILED;
            }
        } // g_mutex_binary_list

        uint32_t timeout_cnt;
        for (timeout_cnt = 0; timeout_cnt < UD_TIMEOUT; timeout_cnt++) {
            int8_t state_check = ENN_RET_SUCCESS;
            TAILQ_FOREACH(bin, &g_list_acc_binary[acc], bin_entries) {
                if (bin->state != NCP_LOADED) {
                    state_check = ENN_RET_FAILED;
                }
            }
            if (state_check == ENN_RET_SUCCESS) {
                ENN_DBG_PRINT("succeeded load bin\n");
                break;
            } else {
                ENN_DBG_PRINT("loading bin...\n");
            }
            usleep(TIME_MSEC * 10);  // 10ms
        }

        if (timeout_cnt >= UD_TIMEOUT) {
            ENN_ERR_PRINT_FORCE("acc[%d] failed emergency recovery\n", acc);
            return ENN_RET_FAILED;
        }

        struct acc_req* req = g_list_acc_req[acc].tqh_first;
        while(req) {
            struct acc_req* next_req = TAILQ_NEXT(req, req_entries);
            request_done(acc, req, CALLED_BY_UD);
            req = next_req;
        }
    } // g_mutex_acc_req_list
    return ENN_RET_SUCCESS;
}
#endif
