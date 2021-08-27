/**
 * @file service_interface.cc
 * @author Hoon Choi (hoon98.choi@samsung.com)
 * @brief service interface for hidl, Android
 * @version 0.1
 * @date 2020-12-28
 *
 * @copyright
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

#include <android-base/logging.h>
#include <hidl/HidlTransportSupport.h>
#include <hidl/HidlLazyUtils.h>
#include "common/enn_utils.h"

#include "EnnInterface.h"

// #define LAZY_SERVICE

using ::vendor::samsung_slsi::hardware::enn::implementation::EnnInterface;
using ::vendor::samsung_slsi::hardware::enn::V1_0::IEnnInterface;
using android::hardware::LazyServiceRegistrar;

#include <stdio.h>
int main(int /* argc */, char * /* argv */[]) {
    android::hardware::configureRpcThreadpool(16 /* maxThreads */, true /* callerWillJoin */);

    enn::util::sys_util_set_sched_affinity(AFFINITY_MID_CORE);

    EnnInterface *service_ptr = new (std::nothrow) EnnInterface;
    if (!service_ptr) {
        return 1;
    }

    android::sp<EnnInterface> service(service_ptr);

    // policy   - scheduler policy as defined in linux UAPI
    // priority - priority. [-20..19] for SCHED_NORMAL, [1..99] for RT
    android::hardware::setMinSchedulerPolicy(service, SCHED_FIFO, 15);

#ifdef LAZY_SERVICE
    auto registrar = LazyServiceRegistrar::getInstance();//std::make_shared<android::hardware::LazyServiceRegistrar>();
    if (registrar.registerService(service) == android::OK) {
        ALOGI(" # ENN Service(lazy) is Started \n");
        android::hardware::joinRpcThreadpool();
    }
#else
    if (service->registerAsService() == android::OK) {
        ALOGI(" # ENN Service is Started \n");
        android::hardware::joinRpcThreadpool();
    }
#endif
    ALOGE("Could not register ENN service to Android. Please check.\n");
    return -1;
}
