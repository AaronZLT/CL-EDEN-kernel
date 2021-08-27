/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or
 * distributed, transmitted, transcribed, stored in a retrieval system or
 * translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed to third parties
 * without the express written permission of Samsung Electronics.
 */

/**
 * @file    dsp_userdriver.h
 * @brief   This is common ENN DSP Userdriver API
 * @details This header defines ENN DSP Userdriver API.
 */
#ifndef USERDRIVER_DSP_DSP_USERDRIVER_H_
#define USERDRIVER_DSP_DSP_USERDRIVER_H_

#include <unordered_map>
#include <vector>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <thread>
#include "userdriver/common/UserDriver.h"
#include "userdriver/common/operator_interfaces/userdriver_operator.h"
#include "userdriver/unified/dsp_bin_info.h"  // UCGO,CGO
#include "userdriver/unified/link_vs4l.h"    // link

namespace enn {
namespace ud {
namespace dsp {

class ExecutableDspUDOperator {
public:
    explicit ExecutableDspUDOperator() {
        memset(&executable_op_info, 0, sizeof(req_info_t));
    }
    ~ExecutableDspUDOperator();
    EnnReturn init(uint32_t in_buf_cnt, uint32_t out_buf_cnt);
    EnnReturn set(model_info_t* op_info, const model::memory::BufferTable& buffer_table);
    EnnReturn set_dsp_exec_info(DspBinInfo *binInfo,
                        const model::memory::BufferTable &buffer_table);
    req_info_t& get(void) { return executable_op_info; }
    std::shared_ptr<eden_memory_t>  get_inputs(void) { return executable_op_info.inputs; }
    std::shared_ptr<eden_memory_t>  get_outputs(void) { return executable_op_info.outputs; }
    EnnReturn deinit(void);
private:
    req_info_t executable_op_info;
};

// Required: TODO(mj.kim010, 6/30): add lock mechanism to this data structure
using ExecutableOperatorMap = std::unordered_map<uint64_t, std::shared_ptr<ExecutableDspUDOperator>>;

// Required: TODO(mj.kim010, 7/31): Camel case refactoring
class DspUDOperator : AccUDOperator {
public:
    explicit DspUDOperator() : AccUDOperator(), op_info() {
        is_cgo_ = false;
        is_async_execute_flag_ = false;
        memset(&op_info, 0, sizeof(model_info_t));
    }
    ~DspUDOperator() {}

    EnnReturn init(uint32_t in_buf_cnt, uint32_t out_buf_cnt, bool isAsync);
    EnnReturn set(uint32_t in_buf_cnt, uint32_t out_buf_cnt, model::component::Operator::Ptr rt_opr_dsp,
            uint64_t operator_list_id, uint64_t unified_op_id);
    model_info_t& get(void) { return op_info; }
    DspUcgoInfo& get_ucgo_info(void) { return ucgo_info; }
    DspCgoInfo& get_cgo_info(void) { return cgo_info; }
    DspBinInfo* get_bin_info(void);
    EnnReturn deinit(void);
    bool is_cgo(void) { return is_cgo_; }
    bool get_async_execute_flag(void) { return is_async_execute_flag_; }

    EnnReturn add_executable_op(uint64_t exec_op_id, const std::shared_ptr<ExecutableDspUDOperator>& executable_op);
    std::shared_ptr<ExecutableDspUDOperator> get_executable_op(uint64_t exec_op_id);
    std::vector<uint64_t> get_all_executable_op_id();
    EnnReturn remove_executable_op(uint64_t exec_op_id);
    uint64_t get_id(void);

private:
    model_info_t op_info;
    ExecutableOperatorMap executable_op_map;
    std::mutex mutex_executable_op_map;
    /* Nice to have: TODO(mj.kim010, TBD) : Unify ucgo/cgo member using parent class. */
    DspUcgoInfo ucgo_info;
    DspCgoInfo cgo_info;
    bool is_cgo_;
    bool is_async_execute_flag_;
};

using DspUDOperators = std::vector<std::shared_ptr<DspUDOperator>>;
using DspSubGraphMap = std::unordered_map<uint64_t, DspUDOperators>;

class DspUserDriver : public UserDriver {
public:
    enum class DspUdStatus { NONE, INITIALIZED, SHUTDOWNED };
    enum class DspAsyncThreadJob { EXECUTE, DESTRUCT };

    static DspUserDriver& get_instance(void);
    ~DspUserDriver(void){}

    EnnReturn Initialize(void) override;
    EnnReturn OpenSubGraph(const model::component::OperatorList& operator_list) override;
    // Temporarily, this function was added to support the unified UD
    // TODO(jungho7.kim): remove this method
    EnnReturn OpenSubGraph(const model::component::OperatorList& operator_list,
            uint64_t operator_list_id, uint64_t unified_op_id);
    EnnReturn PrepareSubGraph(const enn::runtime::ExecutableOperatorList& executable_operator_list) override;
    EnnReturn ExecuteSubGraph(const enn::runtime::OperatorListExecuteRequest& operator_list_execute_request) override;
    EnnReturn CloseSubGraph(const model::component::OperatorList& operator_list) override;
    // Temporarily, this function was added to support the unified UD
    // TODO(jungho7.kim): remove this method
    EnnReturn CloseSubGraph(const model::component::OperatorList& operator_list, uint64_t operator_list_id);
    EnnReturn Deinitialize(void) override;

    uint64_t get_operator_list_id();
    EnnReturn add_graph(uint64_t id, DspUDOperators ud_operators);
    EnnReturn update_graph(uint64_t id, DspUDOperators ud_operators);
    EnnReturn get_graph(uint64_t id, DspUDOperators& out_graph);
    EnnReturn remove_graph(uint64_t id);
    void set_dsp_ud_status(DspUdStatus dsp_ud_status) { dsp_ud_status_ = dsp_ud_status; }
    DspUdStatus get_dsp_ud_status() const { return dsp_ud_status_; }
    EnnReturn get_dsp_session_id(enn::runtime::ExecutableOpListSessionInfo& op_list_session_info);
    int AsyncExecuteLoop();
    EnnReturn FinishAsyncThread(void);

private:
    DspUserDriver(void) : UserDriver(DSP_UD), asyncExecuteThread_(nullptr), asyncModelCount_(0), dsp_ud_status_(DspUdStatus::NONE) {
        ENN_DBG_PRINT("started\n");
    }

    /* 7/27: Currently Async usage is only allowed for DLV3 KPI seperated OP. */
    class AsyncJob {
        public:
            explicit AsyncJob(DspAsyncThreadJob jobType, req_info_t* asyncReq, EdenRequestOptions asyncOpt)
                : jobType_(jobType), asyncReqInfo_(asyncReq), asyncOptions_(asyncOpt) {}
            explicit AsyncJob(DspAsyncThreadJob jobType) : jobType_(jobType) {
                asyncReqInfo_ = nullptr;
                memset(&asyncOptions_, 0, sizeof(EdenRequestOptions));
            }
            DspAsyncThreadJob getJobType() { return jobType_; }
            req_info_t* getReqInfo() { return asyncReqInfo_; }
            const EdenRequestOptions* getReqOption() { return &asyncOptions_; }
        private:
            DspAsyncThreadJob jobType_;
            req_info_t* asyncReqInfo_;
            EdenRequestOptions asyncOptions_;
    };
    bool CheckOpAsyncExecution(const model::component::OperatorList& operator_list);
    void AddAsyncTriggerInfo();
    void RemoveAsyncTriggerInfo();
    EnnReturn UpdateExecutableOp(uint64_t exec_op_id, std::shared_ptr<DspUDOperator> op,
                            std::shared_ptr<ExecutableDspUDOperator> executable_op,
                            const model::memory::BufferTable &buffer_table);
    std::thread *asyncExecuteThread_;
    std::mutex asyncMutex_;  // Guared for job queue
    std::condition_variable asyncCondVar_;
    std::queue<std::shared_ptr<AsyncJob>> asyncJobQueue_;
    std::atomic<int> asyncModelCount_;

    std::mutex mutex_ud_operator_list_map;
    DspSubGraphMap ud_operator_list_map;
    DspUdStatus dsp_ud_status_;
    accelerator_device acc_ = ACCELERATOR_DSP;
};

}  // namespace dsp
}  // namespace ud
}  // namespace enn

#endif  // USERDRIVER_DSP_DSP_USERDRIVER_H_
