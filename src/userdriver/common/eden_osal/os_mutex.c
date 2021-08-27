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
 * @file os_mutex.h
 * @brief Mutex Function Body
 */
#include <pthread.h>
#include <errno.h>
#include "osal.h"

/**
 *  @brief os_mutex_init
 *  @details To initialize a mutex instance according to pvAttr.
 *  @param[in] pvAttr A pointer of attributes for mutex
 *  @param[out] pvMutex A pointer of mutex instance
 *  @returns
 *          PASS: If success
 *          FAIL: If call fail
 */
osal_ret_t os_mutex_init(void* pvMutex, void* pvAttr)
{
    int32_t iStatus = pthread_mutex_init((pthread_mutex_t *)pvMutex, NULL);

    if (iStatus == 0) {
        /* need to design. it just re-use for build */
        void* temp;
        temp = pvAttr;

        return PASS;
    } else {
        return FAIL;
    }
}

/**
 *  @brief os_mutex_destroy
 *  @details To destroy the mutex pointed by pvMutex.
 *  @param[in] pvAttr A pointer of mutex instance
 *  @returns
 *          PASS: If success
 *          FAIL: If call fail
 *          ERR_BUSY : If the mutex is being used.
 */
osal_ret_t os_mutex_destroy(void* pvMutex)
{
    int32_t iStatus = pthread_mutex_destroy((pthread_mutex_t *)pvMutex);

    if (iStatus == 0) {
        return PASS;
    } else if (iStatus == EBUSY) {
        return ERR_BUSY;
    } else {
        return FAIL;
    }
}

/**
 *  @brief os_mutex_lock
 *  @details To lock the mutex instance pointed by pvMutex.
 *  @param[in] pvMutex A pointer of mutex instance
 *  @returns
 *          PASS: If success
 *          FAIL: If call fail
 *          ERR_DEADLK : If already locked
 */
osal_ret_t os_mutex_lock(void* pvMutex)
{
    int32_t iStatus = pthread_mutex_lock((pthread_mutex_t *)pvMutex);

    if (iStatus == 0) {
        return PASS;
    } else if (iStatus == EBUSY) {
        return ERR_DEADLK;
    } else {
        return FAIL;
    }
}

/**
 *  @brief os_mutex_unlock
 *  @details To unlock the mutex instance pointed by pvMutex.
 *  @param[in] pvMutex A pointer of mutex instance
 *  @returns
 *          PASS: If success
 *          FAIL: If call fail
 *          ERR_PERM : If the thread is not the owner of the input mutex
 */
osal_ret_t os_mutex_unlock(void* pvMutex)
{
    int32_t iStatus = pthread_mutex_unlock((pthread_mutex_t *)pvMutex);

    if (iStatus == 0) {
        return PASS;
    } else if (iStatus == EPERM) {
        return ERR_PERM;
    } else {
        return FAIL;
    }
}
