/*
 * Copyright (C) 2020 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or
 * distributed, transmitted, transcribed, stored in a retrieval system or
 * translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung
 * Electronics.
 */

/**
 * @file enn_test_driver_model_container.h
 * @author Hoon Choi (hoon98.choi@)
 * @brief driver codes for enn test of model container
 * @version 0.1
 * @date 2021-03-10
 */

#ifndef SRC_TEST_INTERNAL_ENN_TEST_DRIVER_MODEL_CONTAINER_H_
#define SRC_TEST_INTERNAL_ENN_TEST_DRIVER_MODEL_CONTAINER_H_

#include <string>
#include <vector>

#pragma message("if medium/types.hal.h is not existed, please execute tools/hal_converter/types_to_header.py")
#include "medium/types.hal.h"  // if this eror
using namespace enn::hal;

/* test */
struct EnnBufferCore {
    void *va;
    uint32_t size;
    uint32_t offset;
    uint32_t magic;       // check integrity. size, offset included?
    int type;  // ION? heap? external? partial?
    uint32_t fd;          // ion fd, not allowed ashmem fd or something like that
    uint32_t fd_size;         // further using
    uint32_t cache_flag;  // if offset > 0, should use cache_disabled
    void *ntv_handle;
    int status;

    void *get_native_handle() { return nullptr; }
};

#endif  // SRC_TEST_INTERNAL_ENN_TEST_DRIVER_MODEL_CONTAINER_H_

