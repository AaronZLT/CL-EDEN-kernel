
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
 * @file enn_client_context.h
 * @author Hoon Choi (hoon98.choi@)
 * @brief Manages client contexts
 * @version 0.1
 * @date 2020-12-28
 */

#include <atomic>
#include <future>
#include <mutex>
#include <memory>

#ifndef SRC_CLIENT_ENN_CONTEXT_MANAGER_H_
#define SRC_CLIENT_ENN_CONTEXT_MANAGER_H_

#include "common/enn_debug.h"
#include "common/enn_memory_manager.h"
#include "medium/enn_medium_interface.h"
#include "client/enn_model_container.hpp"
#include "common/enn_preference_generator.hpp"


namespace enn {
namespace client {

class EnnContextManager {
#ifndef ENN_MEDIUM_IF_HIDL
    using InferenceSet = enn::hal::InferenceSet;
    using InferenceData = enn::hal::InferenceData;
#endif
public:
    EnnContextManager() : ref_cnt(0), ccMediumInterface(nullptr), dump_session_memory(false) {
        preference_generator = std::make_unique<enn::preference::EnnPreferenceGenerator>();
    }
    ~EnnContextManager() { deinit(true); }

    int get_ref_cnt();

    EnnReturn init();
    EnnReturn deinit(bool force_deinit = false);
    bool ShouldDumpSessionMemory() { return dump_session_memory; }
    std::shared_ptr<enn::interface::EnnMediumInterface> GetMediumInterface() { return ccMediumInterface; }
    std::unique_ptr<enn::client::EnnModelContainer<EnnModelId, SessionBufInfo, InferenceSet, InferenceData>>
        ccModelContainer;
    std::unique_ptr<enn::EnnMemoryManager> ccMemoryManager;
    std::string GetVersionInfo() { return version_info; }
    std::string GetCommitInfo() { return commit_info; }

    bool hasAsyncFuture(const EnnExecuteModelId exec_model_id);
    bool putAysncFuture(const EnnExecuteModelId exec_model_id, std::future<EnnReturn>&& future);
    std::future<EnnReturn> popAsyncFuture(const EnnExecuteModelId exec_model_id);
    void flushAsyncFutureFor(const EnnModelId model_id);
    const std::unique_ptr<enn::preference::EnnPreferenceGenerator> & get_preference_generator() { return preference_generator; }

private:
    std::atomic_int ref_cnt;
    std::mutex mutex_client_context;
    std::shared_ptr<enn::interface::EnnMediumInterface> ccMediumInterface;
    std::mutex mutex_async_context;
    std::map<const EnnExecuteModelId, std::future<EnnReturn>> asyncFutures;
    std::unique_ptr<enn::preference::EnnPreferenceGenerator> preference_generator;
#if defined(COMMIT_INFO) && defined(VERSION_INFO)  // comes from build flag
    std::string version_info = std::string(COMMIT_INFO);
    std::string commit_info = std::string(VERSION_INFO);
#else
    std::string version_info = std::string();
    std::string commit_info = std::string("NOT DEFINED");
#endif
    bool dump_session_memory;
    void flushAsyncFutures();
    // EnnMemoryManager *ccMemoryManager;
    // EnnParser *ccEnnParser;
};

}  // namespace client
}  // namespace enn

#endif  // SRC_CLIENT_ENN_CONTEXT_MANAGER_H_
