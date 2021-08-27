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
 * @brief Mutex Function Header
 */
#ifndef OSAL_INCLUDE_OS_MUTEX_H_
#define OSAL_INCLUDE_OS_MUTEX_H_

#ifdef __cplusplus
extern "C" {
#endif

/**
 *  @brief os_mutex_init
 *  @details To initialize a mutex instance according to pvAttr.
 *  @param[in] pvAttr A pointer of attributes for mutex
 *  @param[out] pvMutex A pointer of mutex instance
 *  @returns
 *          PASS: If success
 *          FAIL: If call fail
 */
osal_ret_t os_mutex_init(void* pvMutex, void* pvAttr);

/**
 *  @brief os_mutex_destroy
 *  @details To destroy the mutex pointed by pvMutex.
 *  @param[in] pvAttr A pointer of mutex instance
 *  @returns
 *          PASS: If success
 *          FAIL: If call fail
 *          ERR_BUSY : If the mutex is being used.
 */
osal_ret_t os_mutex_destroy(void* pvMutex);

/**
 *  @brief os_mutex_lock
 *  @details To lock the mutex instance pointed by pvMutex.
 *  @param[in] pvMutex A pointer of mutex instance
 *  @returns
 *          PASS: If success
 *          FAIL: If call fail
 *          ERR_DEADLK : If already locked
 */
osal_ret_t os_mutex_lock(void* pvMutex);

/**
 *  @brief os_mutex_unlock
 *  @details To unlock the mutex instance pointed by pvMutex.
 *  @param[in] pvMutex A pointer of mutex instance
 *  @returns
 *          PASS: If success
 *          FAIL: If call fail
 *          ERR_PERM : If the thread is not the owner of the input mutex
 */
osal_ret_t os_mutex_unlock(void* pvMutex);

#ifdef __cplusplus
}
#endif

#endif // OSAL_INCLUDE_OS_MUTEX_H_
