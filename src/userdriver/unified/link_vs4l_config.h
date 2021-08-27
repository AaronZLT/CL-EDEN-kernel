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
 * @file    link_vs4l_config.h
 * @brief   This is vs4l link config values.
 * @details This header defines vs4l link config values.
 * @version 0.3 Basic scenario support.
 */

#ifndef USERDRIVER_UNIFIED_LINK_VS4L_CONFIG_H__
#define USERDRIVER_UNIFIED_LINK_VS4L_CONFIG_H__

#define CPU_MID_CLUSTER (48)
#define CPU_BIG_CLUSTER (192)
#define MAX_MODEL_NAME_LENGTH (512)

#define NPU_UNBOUND 0xFFFFFFFF
#define BOUND_NA 0xFFFFFFFF

#define PRESET_DISABLE_UPPER_BOUND 0

#define LINK_REQUEST_DEL ((void*)-1)
#define EMERGENCY_RECOVERY (0xDC000003)
#define ERR_LOAD_CANT_ALLOC_CMD_LENGTH (0x10C)  // case of SRAM ERROR
#define ERR_LOAD_SEQ_ALLOC (0x112)              // case of SRAM ERROR
#define MAX_LITTLE_CORES (4)
#define MAX_MID_CORES (6)
#define LINK_LOG_PRINT_COUNT (10)
#define NO_SHARED_BUFFER (-1)
#define VS4L_TIMER_TIMEOUT_SEC 20
#define VS4L_TIMER_INTERVAL_SEC 1

// Nice to have: TODO(jungho7.kim, TBD): remove SEL_CLUSTER()
#define SEL_CLUSTER(cl) (cl != 0)?NPU_S_PARAM_QOS_CL2:NPU_S_PARAM_QOS_CL1


/*! User Preference */
typedef enum {
    ALL_HW = 0,
    NPU_ONLY,
    GPU_ONLY,
    CPU_ONLY,
    DSP_ONLY,
    HWCOUNT,
} HwPreference;

typedef enum {
    NORMAL_MODE = 0,
    BOOST_MODE,
    // DIRECT_MODE, // NOT SUPPORTED YET
    // LOW_POWER_MODE, // NOT SUPPORTED YET
    // HIGH_ACCURACY, // NOT SUPPORTED YET

    /**
     * BOOST_ON_EXECUTE_MODE
     * boost device clock when execute request.
     * just execute!, exclude open/close
     * deprecated
     */
    BOOST_ON_EXECUTE_MODE,
    BENCHMARK_MODE,
    RESERVED,
    BOOST_AUX,
    BOOST_BLOCKING_MODE,
    MODECOUNT, /* set max limit for uninitialized preference */
} ModePreference;


typedef enum {
    ACCELERATOR_NPU = 0,
    ACCELERATOR_DSP = 1,
    ACCELERATOR_Unified = 2,
    NUM_ACCELERATOR,
} accelerator_device;

typedef enum {
    NONE,
    BLOCK,
    NONBLOCK,
} RequestMode;

typedef enum _acc_health_state {
    ACC_HEALTH_NORMAL = 0,
    ACC_HEALTH_NO_MORE_MODEL_OPEN = 1,
} health_state_t;

typedef enum _acc_done_caller {
    CALLED_BY_LINK = 0,
    CALLED_BY_UD,
} done_caller_t;

typedef enum _bin_state {
    NCP_UNLOADED = 0,
    NCP_LOADED,
} bin_state_t;

// Nice to have: TODO(jungho7.kim, TBD): remove _target_dev
/** device target */
typedef enum {
    EXYNOS9820,
    EXYNOS9825,
    EXYNOS9630,
    EXYNOS980,
    EXYNOS880,
    EXYNOS9830,
    EXYNOS990,
    EXYNOS2100,
    EXYNOS9815,
    EXYNOS9925,
    EXYNOS_MAX
} _target_dev;

// Nice to have: TODO(jungho7.kim, TBD): remove _target_dev
/** target frequency*/
typedef enum {
    PM_SOC_NUM,
    PM_NPU_MAX,
    PM_MIF_MAX,
    PM_INT_MAX,
    PM_CPU_CL0_MAX,
    PM_CPU_CL1_MAX,
    PM_CPU_CL2_MAX,
    PM_FREQ_MAX
} _target_freq;

const int32_t LINK_ID_INVALID = -1;

static __u32 target_max_freq[EXYNOS_MAX][PM_FREQ_MAX] = {
    /*
     * SOC_NUM, NPU_MAX, MIF_MAX, INT_MAX, L_CPU_MAX, M_CPU_MAX, B_CPU_MAX
     */
    { 9925, 935000, 3172000, 800000, 2112000, 2304000, 2304000 }   //  9925
};

static __u32 target_tuned_freq[EXYNOS_MAX][PM_FREQ_MAX] = {
    /*
     * SOC_NUM, NPU_MAX, MIF_MAX, INT_MAX, L_CPU_MAX, M_CPU_MAX, B_CPU_MAX
     */
    { 9925, 935000, 3172000, 800000, 2112000, 2304000, 2304000 }   //  9925
};

#endif  // USERDRIVER_UNIFIED_LINK_VS4L_CONFIG_H__
