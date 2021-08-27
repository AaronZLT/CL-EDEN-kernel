/*
 *  ion.c
 *
 *   Copyright 2018 Samsung Electronics Co., Ltd.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <assert.h>

// As suggested by:
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=60932
#ifdef __cplusplus
#include <atomic>
using namespace std;
#else
#include <stdatomic.h>
#endif
#include <unistd.h>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <stdbool.h>
#include <limits.h>
#include "log.h"
#include "ion.h"

#include "ion_uapi.h"

#define ION_MAX_HEAP_COUNT 12

/*
 * ION heap names
 * The array index is the legacy heap id
 */
static const struct {
    char *name;
    unsigned int namelen;
} ion_heap_name[ION_MAX_HEAP_COUNT] = {
    {"ion_system_heap", 15},
    {"crypto_heap",     11},
    {"vfw_heap",        8 },
    {"vstream_heap",    12},
    {"[reserved]",       0}, /* reserved heap id. never use */
    {"vframe_heap",     11},
    {"vscaler_heap",    12},
    {"vnfw_heap",       9 },
    {"gpu_crc",         7 },
    {"gpu_buffer",      10},
    {"camera_heap",     11},
    {"secure_camera_heap", 18}
};

#define ION_NUM_HEAP_NAMES (unsigned int)(sizeof(ion_heap_name)/sizeof(ion_heap_name[0]))

static struct {
    char name[MAX_HEAP_NAME];
    unsigned int namelen;
    unsigned int type;
} ion_heap_list[ION_NUM_HEAP_IDS]; /* No more than 16 heap ids */

#define ION_HEAP_TYPE_NONE INT_MAX

const char *exynos_ion_get_heap_name(unsigned int legacy_heap_id) {
    if (legacy_heap_id >= ION_NUM_HEAP_NAMES)
        return NULL;

    return ion_heap_name[legacy_heap_id].name;
}

static unsigned int ion_get_matched_heapmask(unsigned int legacy_heap_id) {
    unsigned int heap_id;

    for (heap_id = 0; heap_id < ION_NUM_HEAP_IDS; heap_id++) {
        if (ion_heap_list[heap_id].type == ION_HEAP_TYPE_NONE)
            continue;

        if ((ion_heap_list[heap_id].namelen == ion_heap_name[legacy_heap_id].namelen) &&
                !strcmp(ion_heap_list[heap_id].name, ion_heap_name[legacy_heap_id].name))
            return 1 << heap_id;
    }

    LOGE(EDEN_EMA, "%s: unable to find heap '%s'(id %u)",
          __func__, ion_heap_name[legacy_heap_id].name, legacy_heap_id);
    LOGD(EDEN_EMA, "ION HEAP LIST");
    for (heap_id = 0; heap_id < ION_NUM_HEAP_IDS; heap_id++)
        if (ion_heap_list[heap_id].type != ION_HEAP_TYPE_NONE)
                LOGD(EDEN_EMA, "ID %d, TYPE %d, NAME %s", heap_id,
                      ion_heap_list[heap_id].type, ion_heap_list[heap_id].name);

    return 0;
}

static unsigned int ion_get_modern_heapmask(unsigned int legacy_heap_mask) {
    unsigned int heap_mask = 0;
    unsigned int legacy_heap_id;

    for (legacy_heap_id = 0; legacy_heap_id < ION_NUM_HEAP_NAMES; legacy_heap_id++) {
        if ((1 << legacy_heap_id) & legacy_heap_mask)
            heap_mask |= ion_get_matched_heapmask(legacy_heap_id);
    }

    return heap_mask;
}

static int ion_free_handle(int fd, int handle) {
    struct ion_handle_data data = { .handle = handle, };
    return ioctl(fd, ION_IOC_FREE, &data);
}

static int ion_alloc_legacy(int ion_fd, size_t len,
                            unsigned int heap_mask, unsigned int flags) {
    int ret;
    struct ion_fd_data fd_data;
    struct ion_allocation_data_legacy alloc_data = {
        .len = len,
        .align = 0,
        .heap_id_mask = heap_mask,
        .flags = flags,
    };

    ret = ioctl(ion_fd, ION_IOC_ALLOC_LEGACY, &alloc_data);
    if (ret < 0) {
        LOGE(EDEN_EMA, "%s(%d, %zu, %#x, %#x) ION_IOC_ALLOC failed: %s", __func__,
              ion_fd, len, heap_mask, flags, strerror(errno));
        return -1;
    }

    fd_data.handle = alloc_data.handle;

    ret = ioctl(ion_fd, ION_IOC_SHARE, &fd_data);
    ion_free_handle(ion_fd, alloc_data.handle);
    if (ret < 0) {
        LOGE(EDEN_EMA, "%s(%d, %zu, %#x, %#x) ION_IOC_SHARE failed: %s", __func__,
              ion_fd, len, heap_mask, flags, strerror(errno));
        return -1;
    }

    return fd_data.fd;
}

static int find_heap_id(struct ion_heap_data *data, int size, const char* name, unsigned int *id) {
    for (int i = 0; i < size; i++) {
        LOGD(EDEN_EMA, "[%d]={id=[%d],name=[%s]}\n", i, data[i].heap_id, data[i].name);
        if (!strncmp(data[i].name, name, MAX_HEAP_NAME)) {
            *id = data[i].heap_id;
            LOGI(EDEN_EMA, "ion_heap_id=[%d], heap_name=[%s]\n", *id, data[i].name);
            return true;
        }
    }
    LOGW(EDEN_EMA, "not found [%s]\n", name);
    return false;
}

static unsigned int probe_ion_heap_id(int fd) {
    static int has_ion_heap_id = false;
    static unsigned int ion_heap_id;
    LOGD(EDEN_EMA, "ion_heap_id=[%d]\n", ion_heap_id);
    if (has_ion_heap_id) {
        return ion_heap_id;
    }

    struct ion_heap_data data[ION_NUM_MAX_HEAPS];
    int cnt = 0;
    ion_query_heap_cnt(fd, &cnt);
    ion_query_get_heaps(fd, cnt, (struct ion_heap_data *)data);

    if (!find_heap_id(data, cnt, ION_VENDOR_HEAP_NAME, &ion_heap_id)
        && !find_heap_id(data, cnt, ION_SYSTEM_HEAP_NAME, &ion_heap_id)) {
            LOGE(EDEN_EMA, "failed to find heap");
            ion_heap_id = 0;
    }

    has_ion_heap_id = true;
    LOGI(EDEN_EMA, "set ion_heap_id=[%d]\n", ion_heap_id);
    return ion_heap_id;
}

static int ion_alloc_modern(int ion_fd, size_t len,
                            unsigned int legacy_heap_mask,
                            unsigned int flags) {
    int ret;
    struct ion_allocation_data_modern data = {
        .len = len,
        .flags = flags,
    };

    unsigned int heap_id = probe_ion_heap_id(ion_fd);
    data.heap_id_mask = 1 << heap_id;
    LOGD(EDEN_EMA, "%s: legacy_heap_mask=%d, but overriden =%x (heap_id=%d)", __func__,
                    legacy_heap_mask, data.heap_id_mask, heap_id);
    if (!data.heap_id_mask) {
        LOGE(EDEN_EMA, "%s: unable to find heaps of heap_mask %#x", __func__, legacy_heap_mask);
        return -1;
    }

    ret = ioctl(ion_fd, ION_IOC_ALLOC_MODERN, &data);
    if (ret < 0) {
        LOGE(EDEN_EMA, "%s(%d, %zu, %#x(%#x), %#x) failed: %d %s", __func__,
              ion_fd, len, legacy_heap_mask, data.heap_id_mask, flags, errno, strerror(errno));
        return -1;
    }

    return (int)data.fd;
}

static void ion_query_heaps(int ion_fd) {
    int ret;
    unsigned int i;
    struct ion_heap_query query;
    struct ion_heap_data data[ION_NUM_HEAP_IDS];

    for (i = 0; i < ION_NUM_HEAP_IDS; i++)
        ion_heap_list[i].type = ION_HEAP_TYPE_NONE;

    memset(&data, 0, sizeof(data));
    memset(&query, 0, sizeof(query));

    query.cnt = ION_NUM_HEAP_IDS;
    query.heaps = (__u64)data;

    ret = ioctl(ion_fd, ION_IOC_HEAP_QUERY, &query);
    if (ret < 0) {
        LOGE(EDEN_EMA, "%s: failed query heaps with ion_fd %d: %s",
              __func__, ion_fd, strerror(errno));
        return;
    }

    if (query.cnt > ION_NUM_HEAP_IDS)
        query.cnt = ION_NUM_HEAP_IDS;

    for (i = 0; i < query.cnt; i++) {
        if (data[i].heap_id < ION_NUM_HEAP_IDS) {
            strncpy(ion_heap_list[data[i].heap_id].name, data[i].name, MAX_HEAP_NAME);
            ion_heap_list[data[i].heap_id].name[MAX_HEAP_NAME - 1] = '\0';
            ion_heap_list[data[i].heap_id].namelen = strlen(ion_heap_list[data[i].heap_id].name);
            ion_heap_list[data[i].heap_id].type = data[i].type;
        }
    }
}

enum ion_version { ION_VERSION_UNKNOWN, ION_VERSION_MODERN, ION_VERSION_LEGACY };

static atomic_int g_ion_version = ATOMIC_VAR_INIT(ION_VERSION_UNKNOWN);

static int ion_is_legacy(int ion_fd) {
    int version = atomic_load_explicit(&g_ion_version, memory_order_acquire);
    if (version == ION_VERSION_UNKNOWN) {
        ion_free_handle(ion_fd, 0);

        /**
          * Check for FREE IOCTL here; it is available only in the old
          * kernels, not the new ones.
          */
        version = (errno == ENOTTY) ? ION_VERSION_MODERN : ION_VERSION_LEGACY;
        atomic_store_explicit(&g_ion_version, version, memory_order_release);

        if (version == ION_VERSION_MODERN)
            ion_query_heaps(ion_fd);
    }
    return (version == ION_VERSION_LEGACY) ? 1 : 0;
}

int exynos_ion_open() {
    int fd = open("/dev/ion", O_RDONLY | O_CLOEXEC);
    if (fd < 0)
        LOGE(EDEN_EMA, "open /dev/ion failed: %d %s", errno, strerror(errno));
    return fd;
}

int exynos_ion_close(int fd) {
    int ret = close(fd);
    if (ret < 0)
        LOGE(EDEN_EMA, "closing fd %d of /dev/ion failed: %s", fd, strerror(errno));
    return ret;
}

int exynos_ion_alloc(int ion_fd, size_t len,
                      unsigned int heap_mask, unsigned int flags) {
    return ion_is_legacy(ion_fd) ? ion_alloc_legacy(ion_fd, len, heap_mask, flags)
                                 : ion_alloc_modern(ion_fd, len, heap_mask, flags);
}

int exynos_ion_free(int ion_fd) {
    return close(ion_fd);
}

#define DMA_BUF_IOCTL_TRACK    _IO('b', 8)
#define DMA_BUF_IOCTL_UNTRACK  _IO('b', 9)
/*
 * It is okay if dma_buf_trace_supported is not accessed atomically.
 * It is updated only once and there is no transient state.
 * Moreover, accessing dma_buf_trace_supported in an inaccurate state
 * harms nothing but unneccessary error messages.
 */
static bool dma_buf_trace_supported = true;
int exynos_ion_dma_buf_track(int fd)
{
    if (!dma_buf_trace_supported)
        return 0;

    if (ioctl(fd, DMA_BUF_IOCTL_TRACK) < 0) {
        if (errno == ENOTTY) {
            dma_buf_trace_supported = false;
            return 0;
        }
        LOGE(EDEN_EMA, "%s(%d) failed: %s", __func__, fd, strerror(errno));
        return -1;
    }

    return 0;
}

int exynos_ion_dma_buf_untrack(int fd)
{
    if (!dma_buf_trace_supported)
        return 0;

    if (ioctl(fd, DMA_BUF_IOCTL_UNTRACK) < 0) {
        if (errno == ENOTTY) {
            dma_buf_trace_supported = false;
            return 0;
        }
        LOGE(EDEN_EMA, "%s(%d) failed: %s", __func__, fd, strerror(errno));
        return -1;
    }

    return 0;
}

int exynos_ion_import_handle(int ion_fd, int fd, int* handle) {
    int ret;
    struct ion_fd_data data = {
        .fd = fd,
    };

    assert(handle == NULL);

    if (!ion_is_legacy(ion_fd)) {
        if (exynos_ion_dma_buf_track(fd))
            return -1;
        /*
         * buffer fd is not a handle and they are maintained seperately.
         * But we provide buffer fd as the buffer handle to keep the libion
         * api compatible with the legacy users including gralloc.
         * We should gradually change all the legacy users in the near future.
         */
        *handle = fd;
        return 0;
    }

    ret = ioctl(ion_fd, ION_IOC_IMPORT, &data);
    if (ret < 0) {
        LOGE(EDEN_EMA, "%s(%d, %d) failed: %s", __func__, ion_fd, fd, strerror(errno));
        return -1;
    }

    *handle = data.handle;

    return 0;
}

int exynos_ion_free_handle(int ion_fd, int handle) {
    int ret;

    if (!ion_is_legacy(ion_fd)) {
        if (exynos_ion_dma_buf_untrack(handle))
            return -1;
        return 0;
    }

    ret = ion_free_handle(ion_fd, handle);
    if (ret < 0) {
        LOGE(EDEN_EMA, "%s(%d, %d) failed: %s", __func__, ion_fd, handle, strerror(errno));
        return -1;
    }

    return 0;
}

int exynos_ion_sync_fd(int ion_fd, int fd) {
    struct ion_fd_data data = {
        .fd = fd,
    };

    if (!ion_is_legacy(ion_fd))
        return 0;

    if (ioctl(ion_fd, ION_IOC_SYNC, &data) < 0) {
        LOGE(EDEN_EMA, "%s(%d, %d) failed: %s", __func__, ion_fd, fd, strerror(errno));
        return -1;
    }

    return 0;
}

int exynos_ion_sync_fd_partial(int ion_fd, int fd, off_t offset, size_t len) {
    struct ion_fd_partial_data data = {
        .fd = fd,
        .offset = offset,
        .len = len
    };

    if (!ion_is_legacy(ion_fd)) {
        return 0;
    }

    if (ioctl(ion_fd, ION_IOC_SYNC_PARTIAL, &data) < 0) {
        LOGE(EDEN_EMA, "%s(%d, %d, %lu, %zu) failed: %s", __func__, ion_fd, fd, offset, len, strerror(errno));
        return -1;
    }

    return 0;
}

int ion_query_heap_cnt(int fd, int* cnt) {
    int ret;
    struct ion_heap_query query;

    if (!cnt) return -EINVAL;
    memset(&query, 0, sizeof(query));

    ret = ioctl(fd, ION_IOC_HEAP_QUERY, &query);
    if (ret < 0) return ret;

    *cnt = query.cnt;
    return ret;
}

int ion_query_get_heaps(int fd, int cnt, void* buffers) {
    int ret;
    struct ion_heap_query query = {
        .cnt = cnt, .heaps = (uintptr_t)buffers,
    };

    ret = ioctl(fd, ION_IOC_HEAP_QUERY, &query);
    return ret;
}
