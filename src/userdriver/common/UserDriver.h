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
 * @file    UserDriver.h
 * @brief   This is common ENN Userdriver API
 * @details This header defines ENN Userdriver API.
 */

#ifndef USERDRIVER_COMMON_USERDRIVER_H_
#define USERDRIVER_COMMON_USERDRIVER_H_

#include <atomic>
#include <map>
#include <string>
#include <vector>
#include <inttypes.h>

#include "userdriver/common/UserDriverTypes.h"
#include "model/component/operator/operator_list.hpp"
#include "runtime/executable_model/executable_operator_list.hpp"
#include "runtime/execute_request/operator_list_execute_request.hpp"
#include "runtime/executable_model/executable_oplist_session_info.hpp"
#include "medium/enn_medium_utils.hpp"

namespace enn {
namespace ud {

constexpr uint64_t OP_UID_MAX = 255;
constexpr uint32_t OP_UID_BITS = 56;

class AccUDOperator {
public:
    uint64_t generate_op_id(uint64_t op_list_id, uint64_t op_uid) {
        ENN_DBG_PRINT("op_list_id:%lx op_uid:0x%lx\n", (unsigned long) op_list_id, (unsigned long) op_uid);
        if ((op_list_id == 0) || (op_list_id & ((uint64_t)0xff << OP_UID_BITS)) != 0 || (op_uid > OP_UID_MAX)) {
            ENN_ERR_PRINT_FORCE("invalid parameter! op_list_id:%lx op_uid:0x%" PRIX64 "\n", (unsigned long) op_list_id, op_uid);
            return 0;
        }
        return (op_list_id | (op_uid << OP_UID_BITS));
    }
};

class UserDriver {
public:
    /**
     * @brief UserDriver constructor
     * @details Initialize internal variables and resources
     * @param void
     */
    explicit UserDriver(const std::string& name) {
        this->name.assign(name);
    }

    /**
     * @brief UserDriver destructor
     * @details Release internal resourses
     * @param void
     */
    virtual ~UserDriver(void) {}

    /**
     * @brief Initialize NPU UD
     * @return enn_ret_t Zero if succussful
     */
    virtual EnnReturn Initialize(void) = 0;

    /**
     * @brief Open a subgraph with preference
     *
     * @param subGraph Subgraph
     * @param preference Preference to open a subgraph
     * @return ud_subgraph_id Subgraph ID
     */
    virtual EnnReturn OpenSubGraph(const model::component::OperatorList& operator_list) = 0;

    /**
     * @brief Prepare a subgraph (optional): Implement real one in the target UD if necessary.
     *
     * @param subGraphId Subgraph ID
     * @param bufferSet Full buffer set for execution
     * @return enn_ret_t Zero if succussful
     */
    virtual EnnReturn PrepareSubGraph(const enn::runtime::ExecutableOperatorList& executable_operator_list) {
        ENN_UNUSED(executable_operator_list);
        return ENN_RET_SUCCESS;
    }

    /**
     * @brief Execute a subgraph
     *
     * @param subGraphId Subgraph ID
     * @param bufferSet Full buffer set for execution
     * @return enn_ret_t Zero if succussful
     */
    virtual EnnReturn ExecuteSubGraph(const enn::runtime::OperatorListExecuteRequest& operator_list_execute_reqeust) = 0;

    /**
     * @brief close a subgraph
     *
     * @param subGraphId Subgraph ID
     * @return enn_ret_t Zero if succussful
     */
    virtual EnnReturn CloseSubGraph(const model::component::OperatorList& operator_list) = 0;

    /**
     * @brief Deinitialize NPU UD
     * @return enn_ret_t Zero if succussful
     */
    virtual EnnReturn Deinitialize(void) = 0;

private:
    std::string name;
};

}  // namespace ud
}  // namespace enn

#endif  // USERDRIVER_COMMON_USERDRIVER_H_
