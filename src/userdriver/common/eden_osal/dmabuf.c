/*
 *  osal/include/dmabuf.h
 *
 *   Copyright 2021 Samsung Electronics Co., Ltd.
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
#include <errno.h>
#include <fcntl.h>
#include <linux/ioctl.h>
#include <stdbool.h>
#include <stdint.h>
#include <sys/types.h>
#include <asm/types.h>
#include <unistd.h>

#include "log.h"

#include "ion.h"
#include "dmabuf.h"

#define INVALID_FD              (-1)
#define print_err()             _print_err(__FUNCTION__)

#define FD_FLAGS                O_RDONLY | O_CLOEXEC
#define FD_FLAGS_ALLOC          O_RDWR

#define DMA_HEAP_IOC_MAGIC      'H'
#define DMA_HEAP_IOCTL_ALLOC    _IOWR(DMA_HEAP_IOC_MAGIC, 0x0, struct dma_heap_allocation_data)

struct dma_heap_allocation_data
{
    __u64 len;
    __u32 fd;
    __u32 fd_flags;
    __u64 heap_flags;
};

static int dmabuf_uncached_fd = -1;

void _print_err(const char* func_name)
{
    LOGE(EDEN_EMA, "%s: err=[%d]", func_name, errno);
}

bool dmabuf_uncached_open()
{
    dmabuf_uncached_fd = open("/dev/dma_heap/system-uncached", FD_FLAGS);
    if (dmabuf_uncached_fd < 0)
    {
        print_err();
        LOGE(EDEN_EMA, "file to open system-uncached, ret=%d", dmabuf_uncached_fd);
    }
    return (0 <= dmabuf_uncached_fd);
}

void dmabuf_uncached_close()
{
    if (0 <= dmabuf_uncached_fd)
    {
        int ret = close(dmabuf_uncached_fd);
        if (ret < 0) print_err();
    }
    dmabuf_uncached_fd = -1;
}

int dmabuf_open()
{
    if (!dmabuf_uncached_open())
    {
        LOGE(EDEN_EMA, "");
        return INVALID_FD;
    }

    int dmabuf_fd = open("/dev/dma_heap/system", FD_FLAGS);
    if (dmabuf_fd < 0)
    {
        print_err();
        dmabuf_uncached_close();
    }

    LOGD(EDEN_EMA, "dmabuf_fd=[%d]", dmabuf_fd);
    return dmabuf_fd;
}


int dmabuf_close(int dmabuf_fd)
{
    LOGD(EDEN_EMA, "dmabuf_fd=[%d]", dmabuf_fd);

    dmabuf_uncached_close();

    int ret = close(dmabuf_fd);
    if (ret < 0) print_err();
    return ret;
}

int dmabuf_alloc(int dmabuf_fd, size_t len, unsigned int heap_mask, unsigned int flags)
{
    if (!(flags & ION_FLAG_CACHED))
    {
        LOGD(EDEN_EMA, "uncached, change fd %d -> %d", dmabuf_fd, dmabuf_uncached_fd);
        dmabuf_fd = dmabuf_uncached_fd;
    }

    struct dma_heap_allocation_data data =
    {
        len, 0, FD_FLAGS_ALLOC, 0
    };
    int ret = ioctl(dmabuf_fd, DMA_HEAP_IOCTL_ALLOC, &data);
    if (ret < 0) print_err();
    return (int)data.fd;
}

int dmabuf_free(int dmabuf_fd)
{
    return close(dmabuf_fd);
}
