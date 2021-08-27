/*
 * Copyright 2017 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#define CONFIG_NPU_MEM_ION
/**
 * @file eden_memory.c
 * @brief implementation of eden_memory api
 */
#include <errno.h>
#include <malloc.h>
#include <string.h>
#include <unistd.h> // for close

#include "eden_memory.h"
#include "log.h"
#if defined(CONFIG_NPU_MEM_ION)
#include "dmabuf.h"
#include "ion.h"
#include <sys/mman.h> // mmap
#endif

#ifdef LOG_TAG
#undef LOG_TAG
#endif
#define LOG_TAG "EdenMemory"
#define PRE_ASSIGNED_SIZE 4096
typedef int (*allocator)(int, size_t, unsigned int, unsigned int);
typedef int (*freer)(int);
typedef int (*closer)(int);
typedef struct _eden_mem_manager {
#if defined(CONFIG_NPU_MEM_ION)
    int client;
    allocator allocate;
    freer free;
    closer close;
#endif
} eden_mem_manager_t;

static eden_mem_manager_t g_eden_mem_manager = { -1, NULL, NULL, NULL };

osal_ret_t eden_mem_init(void)
{
    LOGD(EDEN_EMA, "eden_mem_init started\n");
#if defined(CONFIG_NPU_MEM_ION)
    if (g_eden_mem_manager.client < 0) {
        g_eden_mem_manager.client = dmabuf_open();
        LOGD(EDEN_EMA, "DMABUF client=%d", g_eden_mem_manager.client);
        if (g_eden_mem_manager.client >= 0) {
            LOGI(EDEN_EMA, "use DMBUF client=%d", g_eden_mem_manager.client);
            g_eden_mem_manager.allocate = dmabuf_alloc;
            g_eden_mem_manager.free = dmabuf_free;
            g_eden_mem_manager.close = dmabuf_close;
        } else {
            g_eden_mem_manager.client = exynos_ion_open();
            if (g_eden_mem_manager.client < 0) {
                LOGE(EDEN_EMA, "ion client create fail. ion.client: %d, errno: %d\n",
                        g_eden_mem_manager.client, errno);
                return FAIL;
            }
            LOGI(EDEN_EMA, "use ION client=%d", g_eden_mem_manager.client);
            g_eden_mem_manager.allocate = exynos_ion_alloc;
            g_eden_mem_manager.free = exynos_ion_free;
            g_eden_mem_manager.close = exynos_ion_close;
        }
        LOGD(EDEN_EMA, "ion client initialized. client=[%d]\n", g_eden_mem_manager.client);
    } else {
        LOGD(EDEN_EMA, "ion client already initialized\n");
    }
#endif
    return PASS;
}

osal_ret_t eden_mem_allocate(eden_memory_t *eden_mem) {
    return eden_mem_allocate_with_ion_flag(eden_mem, ION_FLAG_CACHED);
}

// DSP_USERDRIVER: Support Configurable ION_CACHE flag for DSP ION Memory
osal_ret_t eden_mem_allocate_with_ion_flag(eden_memory_t* eden_mem,
                                           uint32_t ion_flag) {
    LOGD(EDEN_EMA, "eden_mem_allocate started\n");
    if (!eden_mem) {
        LOGE(EDEN_EMA, "eden_mem is null\n");
        return ERR_INVALID;
    }
    int preserved_wrong_size = 1;
    if (eden_mem->size == 0) {
        LOGE(EDEN_NN,
            "error alloc memory size : %d, change to pre_assigned size 4096\n",
            (int) eden_mem->size);
        preserved_wrong_size = eden_mem->size;
        eden_mem->size = PRE_ASSIGNED_SIZE;
    }
    eden_mem->alloc_size = eden_mem->size;
    switch (eden_mem->type) {
    case USER_HEAP:
        eden_mem->ref.user_ptr = malloc(eden_mem->size);
        if (!eden_mem->ref.user_ptr) {
            LOGE(EDEN_EMA, "eden_mem->ref.user_ptr malloc failed. errno: %d\n", errno);
            return FAIL;
        }
        break;
    case ION:
        if (!g_eden_mem_manager.allocate) {
            LOGE(EDEN_EMA, "ema not initialized, client=%d, allocate=%p\n",
                g_eden_mem_manager.client, g_eden_mem_manager.allocate);
            return FAIL;
        }

        eden_mem->ref.ion.fd = g_eden_mem_manager.allocate(g_eden_mem_manager.client,
                eden_mem->size, EXYNOS_ION_HEAP_SYSTEM_MASK, ion_flag);
        if (eden_mem->ref.ion.fd <= 0) {
            LOGE(EDEN_EMA, "eden_mem->ref.ion.fd ion alloc failed.\n");
            return FAIL;
        }

        eden_mem->ref.ion.buf = (uint64_t)mmap(NULL, eden_mem->size,
                PROT_READ | PROT_WRITE, MAP_SHARED, eden_mem->ref.ion.fd, 0);
        if (!eden_mem->ref.ion.buf || (void*)eden_mem->ref.ion.buf == MAP_FAILED) {
            LOGE(EDEN_EMA, "eden_mem->ref.ion.buf:%p mmap failed.\n",
                    (void*)eden_mem->ref.ion.buf);
            return FAIL;
        }
        break;
    case MMAP_FD:
    default:
        LOGE(EDEN_EMA, "eden_mem type enum[%d] has not yet supported\n", eden_mem->type);
        return ERR_INVALID;
    }
    if (preserved_wrong_size != 1) {
        eden_mem->size = preserved_wrong_size;
    }
    return PASS;
}

osal_ret_t eden_mem_free(eden_memory_t* eden_mem)
{
    LOGD(EDEN_EMA, "eden_mem_free started\n");
    if (!eden_mem) {
        LOGE(EDEN_EMA, "eden_mem is null\n");
        return ERR_INVALID;
    }

    switch (eden_mem->type) {
    case USER_HEAP:
        free(eden_mem->ref.user_ptr);
        eden_mem->size = 0;
        LOGD(EDEN_EMA, "free(eden_mem->ref.userptr)\n");
        break;
    case ION:
#if defined(CONFIG_NPU_MEM_ION)
        if (!g_eden_mem_manager.free) {
            LOGE(EDEN_EMA, "ema not initialized, client=%d, free=%p\n",
                g_eden_mem_manager.client, g_eden_mem_manager.free);
            return FAIL;
        }

        if ((eden_mem->ref.ion.buf > 0) && (eden_mem->size > 0) && (eden_mem->ref.ion.fd != 0)) {
            LOGD(EDEN_EMA, "eden_mem buf/size check success\n");
            int ret = munmap((void*)eden_mem->ref.ion.buf, eden_mem->alloc_size);
            if (ret < 0) {
                // try munmap with alloc_size then once again with size
                ret = munmap((void*)eden_mem->ref.ion.buf, eden_mem->size);
            }
            if (ret < 0) {
                LOGE(EDEN_EMA, "eden_mem munmap failed: %d\n", errno);
                return FAIL;
            }
            LOGD(EDEN_EMA, "close fd: %d\n", eden_mem->ref.ion.fd);
            ret = g_eden_mem_manager.free(eden_mem->ref.ion.fd);
            if (ret < 0) {
                LOGE(EDEN_EMA, "eden_mem close failed: %d\n", errno);
                return FAIL;
            }
            eden_mem->ref.ion.fd = 0;
            LOGD(EDEN_EMA, "free(eden_mem->ref.ion)\n");
        }
        break;
#endif
    case MMAP_FD:
    default:
        LOGE(EDEN_EMA, "eden_mem type enum [%d] has not yet supported\n", eden_mem->type);
        return ERR_INVALID;
    }
    eden_mem->alloc_size = 0;
    return PASS;
}

osal_ret_t eden_mem_convert(eden_memory_t* from, eden_memory_t* to)
{
    LOGD(EDEN_EMA, "eden_mem_convert started\n");
    if ((!from) || (!to)) {
        return ERR_INVALID;
    }

    LOGI(EDEN_EMA, "not yet supported\n");

    return FAIL;
}

osal_ret_t eden_mem_shutdown(void)
{
    LOGD(EDEN_EMA, "started\n");

#if defined(CONFIG_NPU_MEM_ION)
    if (g_eden_mem_manager.close) {
        int ret = g_eden_mem_manager.close(g_eden_mem_manager.client);
        if (ret != 0) {
            LOGE(EDEN_EMA, "ion close(free) failed: %d\n", ret);
            return FAIL;
        }
    } else {
        LOGW(EDEN_EMA, "ion fd is already closed\n");
    }
#endif

    memset(&g_eden_mem_manager, 0, sizeof(eden_mem_manager_t));
#if defined(CONFIG_NPU_MEM_ION)
    g_eden_mem_manager.client = -1;
#endif
    return PASS;
}
