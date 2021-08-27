/*
 * Copyright (C) 2018 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed,
 * transmitted, transcribed, stored in a retrieval system or translated into any human or
 * computer language in any form by any means, electronic, mechanical, manual or
 * otherwise or disclosed to third parties without the express written permission of
 * Samsung Electronics.
 */

/**
 * @file : vs4l.h
 * @brief : This file define A struct of vs4l
 * @details : vs4l data rule between user_driver and device_driver
 */

#ifndef USERDRIVER_UNIFIED_VS4L_H_
#define USERDRIVER_UNIFIED_VS4L_H_

#include <linux/types.h>
#include <sys/time.h>

struct vs4l_sched_param {
    __u32 priority;
    __u32 bound_id;
};

enum vs4l_acc_nw_priority_val {
    ACC_PRIORITY_MIN_VAL = 0,
    ACC_PRIORITY_MAX_VAL = 255,
};

/* TODO: Check if we need this for DSP. */
enum vs4l_acc_nw_boundness {
    NPU_BOUND_CORE0 = 0,
    NPU_BOUND_UNBOUND = 0xFFFFFFFF,
};

struct vs4l_param {
    __u32 target;
    unsigned long addr;
    __u32 offset;
    __u32 size;
};

struct vs4l_param_list {
    __u32 count;
    struct vs4l_param *params;
};

struct vs4l_roi {
    unsigned int x;
    unsigned int y;
    unsigned int w;
    unsigned int h;
};

enum acc_hardware_e {
    NPU_HWDEV_DNC = 0x1,
    NPU_HWDEV_NPU = 0x2,
    NPU_HWDEV_DSP = 0x4,
};

struct vs4l_ctrl {
    __u32 ctrl;
    __u32 value;
};

struct vs4l_graph {
    unsigned int  id;
    unsigned int  priority;
    unsigned int  time; /* in millisecond */
    unsigned int  flags;
    unsigned int  size;
    unsigned long addr;
};

struct vs4l_format {
    __u32 target;
    __u32 format;
    __u32 plane;
    __u32 width;
    __u32 height;
    __u32 stride;
    __u32 cstride;
    __u32 channels;
    __u32 pixel_format;
};

struct vs4l_format_list {
    __u32               direction;
    __u32               count;
    struct vs4l_format* formats;
};

struct vs4l_buffer {
    struct vs4l_roi roi;
    union {
        unsigned long userptr;
        int           fd;
    } m;
    unsigned long reserved;
};

enum vs4l_buffer_type {
    VS4L_BUFFER_LIST,
    VS4L_BUFFER_ROI,
    VS4L_BUFFER_PYRAMID };

enum vs4l_memory {
    VS4L_MEMORY_USERPTR = 1,
    VS4L_MEMORY_VIRTPTR,
    VS4L_MEMORY_DMABUF };

enum vs4l_cl_flag {
    VS4L_CL_FLAG_TIMESTAMP,
    VS4L_CL_FLAG_PREPARE = 8,
    VS4L_CL_FLAG_INVALID,
    VS4L_CL_FLAG_DONE
};

#ifdef EXYNOS_NN_PROFILER
struct vs4l_profiler_node {
    char*           label;
    unsigned int    duration;
    struct          vs4l_profiler_node** child;
};

struct vs4l_profiler {
    __u8                        level;
    struct vs4l_profiler_node*  node;
};
#endif

struct vs4l_container {
    unsigned int        type;
    unsigned int        target;
    unsigned int        memory;
    unsigned int        reserved[4];
    unsigned int        count;
    struct vs4l_buffer* buffers;
};

struct vs4l_container_list {
    unsigned int           direction;
    unsigned int           id;
    unsigned int           index;
    unsigned int           flags;
    struct timeval         timestamp[6];
    unsigned int           count;
    struct vs4l_container* containers;
};

enum vs4l_direction {
    VS4L_DIRECTION_IN = 1,
    VS4L_DIRECTION_OT };

#define VS4L_DF_IMAGE(a, b, c, d) ((a) | (b << 8) | (c << 16) | (d << 24))
#define VS4L_DF_IMAGE_RGB VS4L_DF_IMAGE('R', 'G', 'B', '2')
#define VS4L_DF_IMAGE_RGBX VS4L_DF_IMAGE('R', 'G', 'B', 'A')
#define VS4L_DF_IMAGE_NV12 VS4L_DF_IMAGE('N', 'V', '1', '2')
#define VS4L_DF_IMAGE_NV21 VS4L_DF_IMAGE('N', 'V', '2', '1')
#define VS4L_DF_IMAGE_YUYV VS4L_DF_IMAGE('Y', 'U', 'Y', 'V')
#define VS4L_DF_IMAGE_YUV4 VS4L_DF_IMAGE('Y', 'U', 'V', '4')
#define VS4L_DF_IMAGE_U8 VS4L_DF_IMAGE('U', '0', '0', '8')
#define VS4L_DF_IMAGE_U16 VS4L_DF_IMAGE('U', '0', '1', '6')
#define VS4L_DF_IMAGE_U32 VS4L_DF_IMAGE('U', '0', '3', '2')
#define VS4L_DF_IMAGE_S16 VS4L_DF_IMAGE('S', '0', '1', '6')
#define VS4L_DF_IMAGE_S32 VS4L_DF_IMAGE('S', '0', '3', '2')
#define VS4L_DF_IMAGE_NPU VS4L_DF_IMAGE('N', 'P', 'U', '0')
#define VS4L_DF_IMAGE_DSP VS4L_DF_IMAGE('D', 'S', 'P', '0')
#define VS4L_VERTEXIOC_S_GRAPH _IOW('V', 0, struct vs4l_graph)
#define VS4L_VERTEXIOC_S_FORMAT _IOW('V', 1, struct vs4l_format_list)
#define VS4L_VERTEXIOC_S_PARAM _IOW('V', 2, struct vs4l_param_list)
#define VS4L_VERTEXIOC_S_CTRL _IOW('V', 3, struct vs4l_ctrl)
#define VS4L_VERTEXIOC_STREAM_ON _IO('V', 4)
#define VS4L_VERTEXIOC_STREAM_OFF _IO('V', 5)
#define VS4L_VERTEXIOC_QBUF _IOW('V', 6, struct vs4l_container_list)
#define VS4L_VERTEXIOC_DQBUF _IOW('V', 7, struct vs4l_container_list)
#define VS4L_VERTEXIOC_PREPARE _IOW('V', 8, struct vs4l_container_list)
#define VS4L_VERTEXIOC_UNPREPARE _IOW('V', 9, struct vs4l_container_list)

#define VS4L_VERTEXIOC_SCHED_PARAM _IOW('V', 10, struct vs4l_sched_param)

#ifdef EXYNOS_NN_PROFILER
#define VS4L_VERTEXIOC_PROFILE_ON _IOW('V', 11, struct vs4l_profiler)
#define VS4L_VERTEXIOC_PROFILE_OFF _IOW('V', 12, struct vs4l_profiler)
#endif
#define VS4L_VERTEXIOC_BOOTUP _IO('V', 13)
#endif  // USERDRIVER_UNIFIED_VS4L_H_
