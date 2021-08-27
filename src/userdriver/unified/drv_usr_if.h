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

/**
 * @file drv_usr_if.h
 * @brief driver side struct
 * @details driver side struct
 */

#ifndef USERDRIVER_UNIFIED_DRV_USR_IF_H_
#define USERDRIVER_UNIFIED_DRV_USR_IF_H_

#define MAX_PATH_SIZE 100
#define MAX_FILE_SIZE 100

/** device freq */
#define DEV_TARGET_NUM (4)

/** Current supporting user API version */
#define USER_API_VERSION 1 // TODO: CHECK! DSP must be 3?

/** offset to undo preset scenario setting : INT32_MAX */
#define PRESET_SCENARIO_UNDO      0x7fffffff

struct drv_usr_share {
    unsigned int id;
    int bin_fd;                     // fd acquired by exynos_ion_open
    unsigned int bin_size;          // bin size
    unsigned long bin_mmap;         // virtual address acquired by mmap
    /*
     * Unified op ID is used by the unified firmware to identify which binaries belong to single unified op.
     * Unified op ID becames necessary to support an unified op from NCP v25.
     *
     * Steps
     *  1. UENN forwards an 64 bits unified op ID to UDD
     *  2. UDD converts the 64 bits ID to 32 bits ID
     *  3. UDD forwards the converted 32 bits ID to UFW
     *  4. UFW uses the 32 bits ID, finally.
     */
    uint64_t unified_op_id;         // unified op ID
};

/** device target for applying user preference */
/* TODO(mj.kim) : Check if we can rename this for unified */
typedef enum {
    NPU_S_PARAM_FW_UTC_LOAD = 0x780000,
    NPU_S_PARAM_FW_UTC_EXECUTE,
    NPU_S_PARAM_QOS_NPU = 0x880000,
    NPU_S_PARAM_QOS_DSP,
    NPU_S_PARAM_QOS_MIF,
    NPU_S_PARAM_QOS_INT,
    NPU_S_PARAM_QOS_DNC,
    NPU_S_PARAM_QOS_NPU_MAX,
    NPU_S_PARAM_QOS_DSP_MAX,
    NPU_S_PARAM_QOS_MIF_MAX,
    NPU_S_PARAM_QOS_INT_MAX,
    NPU_S_PARAM_QOS_DNC_MAX,
    NPU_S_PARAM_QOS_CL0 = 0x890000,
    NPU_S_PARAM_QOS_CL1,
    NPU_S_PARAM_QOS_CL2,
    NPU_S_PARAM_QOS_CL0_MAX,
    NPU_S_PARAM_QOS_CL1_MAX,
    NPU_S_PARAM_QOS_CL2_MAX,
    NPU_S_PARAM_CPU_AFF,

    /** --- User API version: 4 --- **/
    /** Targets for Preset Scenario **/
    NPU_S_PARAM_IS_PRESET = 0x891000,
    NPU_S_PARAM_QOS_NPU_PRESET,
    NPU_S_PARAM_QOS_DSP_PRESET,
    NPU_S_PARAM_QOS_MIF_PRESET,
    NPU_S_PARAM_QOS_INT_PRESET,
    NPU_S_PARAM_QOS_CL0_PRESET,
    NPU_S_PARAM_QOS_CL1_PRESET,
    NPU_S_PARAM_QOS_CL2_PRESET,
    NPU_S_PARAM_QOS_MO_SCEN_PRESET,
    NPU_S_PARAM_QOS_CPU_AFF_PRESET,

    /* Added in Exynos9925 */
    NPU_S_PARAM_QOS_APP_ID,
    NPU_S_PARAM_QOS_MODEL_ID,
    NPU_S_PARAM_QOS_SUBGRAPH_ID,
    NPU_S_PARAM_QOS_MODEL_NAME,
    NPU_S_PARAM_QOS_GPU_PRESET,     // Input the target frequency into the 'offset'
    NPU_S_PARAM_QOS_CL0_IDLE_PRESET,    // Input the target 0 or 1 into the 'offset'
    NPU_S_PARAM_QOS_CL1_IDLE_PRESET,    // Input the target 0 or 1 into the 'offset'
    NPU_S_PARAM_QOS_CL2_IDLE_PRESET,    // Input the target 0 or 1 into the 'offset'
    NPU_S_PARAM_QOS_FW_HINT_PRESET, // Input the target 0 or 1 into the 'offset'
    NPU_S_PARAM_QOS_LLC_PRESET,     // Input the LLC size for each region (DNC, NPU core, DSP, FLC) into the 'addr', 'size', 'offset'
    NPU_S_PARAM_QOS_LLC_SCEN_PRESET,    // Input the LLC scenario ID into the 'offset'

    NPU_S_PARAM_QOS_RST,

    NPU_S_PARAM_PERF_MODE = 0x900000,
    NPU_S_PARAM_PRIORITY,
    NPU_S_PARAM_TPF,
    /* DSP PARAM */
    NPU_S_PARAM_DSP_KERNEL = 0xA00000,
} prefer_target;



typedef enum {
    /** --- User API version: 4 --- **/
    NPU_S_PARAM_PERF_MODE_NONE = 0,
    NPU_S_PARAM_PERF_MODE_NPU_BOOST_ON_EXECUTE,
    NPU_S_PARAM_PERF_MODE_NPU_BOOST,
    NPU_S_PARAM_PERF_MODE_CPU_BOOST,
    NPU_S_PARAM_PERF_MODE_NPU_DN,
    NPU_S_PARAM_PERF_MODE_MO_BOOST_ON_EXECUTE,
    NPU_S_PARAM_PERF_MODE_DLV3,
    NPU_S_PARAM_PERF_MODE_NPU_BOOST_BLOCKING,
} acc_perf_mode;

#endif  // USERDRIVER_UNIFIED_DRV_USR_IF_H_
