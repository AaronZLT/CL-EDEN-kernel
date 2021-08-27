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

#include "client/enn_context_manager.h"

namespace enn {
namespace client {

int EnnContextManager::get_ref_cnt() {
    std::lock_guard<std::mutex> guard(mutex_client_context);
    return ref_cnt.load();
}

EnnReturn EnnContextManager::init() {
    std::lock_guard<std::mutex> guard(mutex_client_context);
    if (ref_cnt == 0) {
        /* check if medium interface is already constructed */
        CHECK_AND_RETURN_ERR((ccMediumInterface != nullptr), ENN_RET_INVAL,
                             "Critical Error: interface(%p) is initalized\n", ccMediumInterface.get());
        /* construct managers / containers */
        ccMediumInterface = std::make_shared<enn::interface::EnnMediumInterface>();
        CHECK_AND_RETURN_ERR((ccMediumInterface == nullptr), ENN_RET_MEM_ERR,
                             "Initialize Error: ccMediumInterface creation failed\n");
        ccMemoryManager = std::make_unique<enn::EnnMemoryManager>();
        CHECK_AND_RETURN_ERR((ccMemoryManager == nullptr), ENN_RET_MEM_ERR,
                             "Initialize Error: EnnMemoryManager creation failed\n");
        ccModelContainer = std::make_unique<EnnModelContainer<EnnModelId, SessionBufInfo, InferenceSet, InferenceData>>();
        CHECK_AND_RETURN_ERR((ccModelContainer == nullptr), ENN_RET_MEM_ERR,
                             "Initialize Error: EnnModelContainer creation failed\n");
        /* init managers */
        CHECK_AND_RETURN_ERR((ccMemoryManager->init()), ENN_RET_FAILED,
                             "Initialize Error: MemoryManager initialization failed\n");
        CHECK_AND_RETURN_ERR((ccMediumInterface->init()), ENN_RET_FAILED,
                             "Initialize Error: MediumInterface initialization failed\n");
    }
    // update flag for dumping memories only in debug build
#ifndef ENN_BUILD_RELEASE
    enn::debug::MaskType env_val = 0;
    // update dump_session_memory from environment variable every context's initialization
    if (!util::get_environment_property(debug::DbgPrintManager::GetInstance().get_debug_property_name().c_str(), &env_val)) {
        if (env_val & ZONE_BIT_MASK(debug::DbgPartition::kFileDumpSession)) {
            dump_session_memory = true;
            ENN_DBG_PRINT("Context init: set dump_session_memory\n");
        }
    }

    ENN_DBG_PRINT("Context init: get property: 0x%" PRIX64 "\n", env_val);
#endif

    ref_cnt++;
    std::string context_show = std::string("Context ref_cnt") + std::to_string(ref_cnt.load());
    context_show += std::string(", Commit [") + commit_info;
    context_show += std::string("], Version [") + version_info + std::string("]");
    ENN_INFO_PRINT_FORCE("%s %s\n", context_show.c_str(), (ref_cnt == 1 ? ", created context." : ""));
    return ENN_RET_SUCCESS;
}

EnnReturn EnnContextManager::deinit(bool force_deinit) {
    std::lock_guard<std::mutex> guard(mutex_client_context);
    --ref_cnt;
    ENN_INFO_PRINT("Context ref_cnt(%d) %s\n", ref_cnt.load(), (ref_cnt == 0 ? ", will delete context." : ""));
    if (ref_cnt == 0 || (ref_cnt > 0 && force_deinit)) {
        // try to wait std::future in the map ::asyncFutures which is missed by client
        flushAsyncFutures();

        ccMemoryManager->deinit();
        ccMemoryManager = nullptr;
        CHECK_AND_RETURN_ERR((ccMediumInterface->deinit()), ENN_RET_FAILED,
                             "Deinitialize Error: MediumInterface deinitialization failed\n");
        ccMediumInterface = nullptr;
        ref_cnt = 0;
    }
    return ENN_RET_SUCCESS;
}

bool EnnContextManager::hasAsyncFuture(const EnnExecuteModelId exec_model_id) {
    std::lock_guard<std::mutex> guard(mutex_async_context);
    return asyncFutures.find(exec_model_id) != asyncFutures.end();
}

bool EnnContextManager::putAysncFuture(const EnnExecuteModelId exec_model_id, std::future<EnnReturn>&& future) {
    std::lock_guard<std::mutex> guard(mutex_async_context);
    auto it = asyncFutures.find(exec_model_id);
    CHECK_AND_RETURN_ERR(it != asyncFutures.end(),
                            ENN_RET_INVAL,
                            "EnnExecuteModelId 0x%" PRIX64 " already exists.\n", exec_model_id);
    //ENN_DBG_PRINT("EnnExecuteModelId 0x%" PRIX64 ", F %p\n", exec_model_id, &future);
    asyncFutures.insert(std::make_pair(exec_model_id, std::move(future)));
    ENN_DBG_PRINT("EnnExecuteModelId 0x%" PRIX64 "\n", exec_model_id);
    return true;
}

template<class T>
std::future<std::decay_t<T>> make_ready_future(T&& value) {
    std::promise<std::decay_t<T>> promise;
    auto future = promise.get_future();
    promise.set_value(std::forward<T>(value));
    return future;
}

std::future<EnnReturn> EnnContextManager::popAsyncFuture(const EnnExecuteModelId exec_model_id) {
    std::lock_guard<std::mutex> guard(mutex_async_context);
    auto it = asyncFutures.find(exec_model_id);
    CHECK_AND_RETURN_ERR(it == asyncFutures.end(),
                            make_ready_future(ENN_RET_INVAL),
                            "EnnExecuteModelId 0x%" PRIX64 " is not found.\n", exec_model_id);
    auto future = std::move(it->second);
    ENN_DBG_PRINT("EnnExecuteModelId 0x%" PRIX64 "\n", it->first);
    asyncFutures.erase(it);
    return future;
}

void EnnContextManager::flushAsyncFutureFor(const EnnModelId model_id) {
    std::lock_guard<std::mutex> guard(mutex_async_context);
    // TODO: change to use 0x1FFFFFFF000000 as mask value after applying new UID scheme
    const uint64_t MODEL_ID_MASK = 0xFFFFFFFFFF000000;
    for (auto& it : asyncFutures) {
        if ((it.first & MODEL_ID_MASK) == model_id) {
            ENN_WARN_PRINT_FORCE("Flush Missed Waiting AsyncExecution for EnnExecuteModelId 0x%" PRIX64 "...\n", it.first);
            auto ret = it.second.get();
            ENN_DBG_PRINT("Flush Missed Waiting AsyncExecution for EnnExecuteModelId 0x%" PRIX64 ", R %d\n", it.first, ret);
            asyncFutures.erase(it.first);
        }
    }
}

void EnnContextManager::flushAsyncFutures() {
    std::lock_guard<std::mutex> guard(mutex_async_context);
    for (auto& it : asyncFutures) {
        ENN_WARN_PRINT_FORCE("Flush Missed Waiting AsyncExecution for EnnExecuteModelId 0x%" PRIX64 "...\n", it.first);
        auto ret = it.second.get();
        ENN_DBG_PRINT("Flush Missed Waiting AsyncExecution for EnnExecuteModelId 0x%" PRIX64 ", R %d\n", it.first, ret);
    }
    asyncFutures.clear();
}

}  // namespace client
}  // namespace enn
