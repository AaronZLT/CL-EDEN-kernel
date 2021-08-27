/**
 * @file enn_medium_interface.hpp
 * @author Hoon Choi (hoon98.choi@samsung.com)
 * @brief Define client manager to manage client pid in service manager
 * @version 0.1
 * @date 2020-12-29
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

#ifndef SRC_MEDIUM_ENN_CLIENT_MANAGER_HPP_
#define SRC_MEDIUM_ENN_CLIENT_MANAGER_HPP_

#include <iostream>
#include <memory>
#include <map>
#include <mutex>
#include "common/enn_debug.h"

namespace enn {
namespace interface {

enum class EnnClientStatus {
    client_alive,
    client_dead,
    client_wait,
    client_max,
};

template <typename IdType, typename SpMonitorType>
struct ClientPoolElement {
    explicit ClientPoolElement(IdType id) : reference_count(1), pid(id) {}
    EnnClientStatus status = EnnClientStatus::client_wait;
    SpMonitorType cb_sp;
    int reference_count;
    IdType pid;
};

template <typename IdType, typename SpMonitor>
class ClientManager {
public:
    ClientManager() {
        pool.clear();
    }

    std::shared_ptr<ClientPoolElement<IdType, SpMonitor>> GetClient(IdType id) {
        ENN_INFO_PRINT("id: %d, %s\n", id, pool.find(id) == pool.end() ? "No" : "Yes");
        if (pool.find(id) == pool.end())
            return nullptr;
        return pool[id];
    }

    bool PutClient(IdType id) {  // simple implementation to test
        auto ele = GetClient(id);
        if (ele == nullptr) {
            pool[id] = std::make_shared<ClientPoolElement<IdType, SpMonitor>>(id);
            pool[id]->status = EnnClientStatus::client_alive;
        } else {
            if (ele->status == EnnClientStatus::client_alive) {
                ele->reference_count++;
            } else {
                ENN_ERR_PRINT("The client is not alive, please check\n");
                return false;
            }
        }
        return true;
    }

    bool PutClient(IdType id, SpMonitor cb) {
        std::lock_guard<std::mutex> guard(mutex_client_manager);
        bool result = this->PutClient(id);
        if (result) pool[id]->cb_sp = cb;
        return result;
    }

    bool PopClient(IdType id) {
        std::lock_guard<std::mutex> guard(mutex_client_manager);

        auto ele = GetClient(id);
        if (ele == nullptr) {
            ENN_ERR_PRINT("# No id in pool \n");
            return false;
        }
        if (--(ele->reference_count) == 0)
            pool.erase(id);
        return true;
    }

    int32_t GetClientNum() { return pool.size(); }

    void show() {
        std::lock_guard<std::mutex> guard(mutex_client_manager);
        // if (pool.size() == 0) ENN_INFO_PRINT("[] No elements included");
        for (auto &t : pool) {
            ENN_INFO_PRINT("[] pid(%d, %d) --> ref_cnt(%d), status(%d)\n", t.first, t.second->pid, t.second->reference_count,
                           t.second->status);
        }
    }

private:
    std::mutex mutex_client_manager;
    std::map<IdType, std::shared_ptr<ClientPoolElement<IdType, SpMonitor>>> pool;
};

}  // namespace interface
}  // namespace enn

#endif  // SRC_MEDIUM_ENN_CLIENT_MANAGER_HPP_
