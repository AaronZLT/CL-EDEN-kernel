/*
 * Copyright (C) 2020 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or
 * distributed, transmitted, transcribed, stored in a retrieval system or
 * translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed to third parties
 * without the express written permission of Samsung Electronics.
 */

#include "medium/enn_medium_interface.h"

#include "client/enn_api-type.h"
#include "client/enn_api.h"
#include "common/enn_debug.h"

#ifdef ENN_MEDIUM_IF_HIDL
#include <hwbinder/IPCThreadState.h>
#endif

#include <sys/mman.h>
#include <cinttypes>

namespace enn {
namespace interface {

EnnMediumInterface::EnnMediumInterface() {
    HIDL_IF(service = IEnnInterface::getService());
    LIB_IF(service = ::enn::runtime::Engine::get_instance());
    CHECK_AND_RETURN_ERR(service == nullptr, , "Service is empty\n");
}

EnnMediumInterface::~EnnMediumInterface() {
    HIDL_IF(IPCThreadState::self()->flushCommands());
    HIDL_IF(service.clear());
    HIDL_IF(IPCThreadState::self()->flushCommands());
}

#ifdef ENN_MEDIUM_IF_HIDL
// NOTE(hoon98.choi) Get service everytime. This is simply implemented with macro,
//                   for easy-using of instant variable.
#define __START_SERVICE() \
    auto service = IEnnInterface::getService(); \
    CHECK_AND_RETURN_ERR(service == nullptr, ENN_RET_FAILED, "Service is empty, please initialize again\n")

#define __FINISH_SERVICE() \
    IPCThreadState::self()->flushCommands(); \
    service.clear()
#else
#define __START_SERVICE()
#define __FINISH_SERVICE()
#endif

/****** Client side *******/
EnnReturn EnnMediumInterface::init() {
    __START_SERVICE();

    // convert to general type
    ENN_INFO_PRINT("pid: %d, tid: %d\n", enn::util::get_pid(), enn::util::get_tid());

    HIDL_IF(cb_sp = new EnnCallback());
    HIDL_IF(uint32_t ret = service->init(cb_sp));
    LIB_IF(uint32_t ret = service->init());
    __FINISH_SERVICE();

    return static_cast<EnnReturn>(ret);
}

EnnReturn EnnMediumInterface::deinit() {
    __START_SERVICE();

    // convert to general type
    ENN_INFO_PRINT("pid: %d, tid: %d\n", enn::util::get_pid(), enn::util::get_tid());

    int32_t ret = service->deinit();
    __FINISH_SERVICE();
    return static_cast<EnnReturn>(ret);
}

EnnReturn EnnMediumInterface::open_model(std::vector<std::shared_ptr<EnnBufferCore>> buf_list, uint32_t model_type,
                                         std::vector<uint32_t> & preference_stream, std::shared_ptr<SessionBufInfo> ret_info) {
    __START_SERVICE();
    auto& buf = buf_list[0];
    BufferCore open_model_buf;
    std::vector<BufferCore> open_model_params;
    open_model_params.reserve(buf_list.size());

#ifdef ENN_MEDIUM_IF_HIDL
    open_model_buf = {buf->get_native_handle(),
                      buf->size,
                      0,
                      reinterpret_cast<uint64_t>(buf->va)};

    for (int buf_idx = 1; buf_idx < buf_list.size(); ++buf_idx) {
        auto& buf_ele = buf_list[buf_idx];
        open_model_params.push_back({buf_ele->get_native_handle(),
                                     buf_ele->size,
                                     0,
                                     reinterpret_cast<uint64_t>(buf_ele->va)});
    }

    service->open_model({open_model_buf, open_model_params, model_type, {preference_stream, {}}}, [&](SessionBufInfo info) { *ret_info = info; });
#else
    open_model_buf.data = {1, {buf->fd}, buf->va};
    open_model_buf.size = buf->size,
    open_model_buf.attr = 0,
    open_model_buf.va = reinterpret_cast<uint64_t>(buf->va);

    for (int buf_idx = 1; buf_idx < buf_list.size(); ++buf_idx) {
        auto& buf_ele = buf_list[buf_idx];
        BufferCore param_ele;
        param_ele.data.numFds = buf_ele->size == 0 ? 0 : 1;
        param_ele.data.data[0] = buf_ele->fd;
        param_ele.data.va = buf_ele->va;
        param_ele.fd = &(param_ele.data);
        param_ele.size = buf_ele->size,
        param_ele.attr = 0,
        param_ele.va = reinterpret_cast<uint64_t>(buf_ele->va);
        open_model_params.push_back(param_ele);
    }

    service->open_model({open_model_buf, open_model_params, model_type, {preference_stream, {}}}, ret_info.get());
#endif
    __FINISH_SERVICE();

    return ENN_RET_SUCCESS;
}

EnnMediumInterface::DeviceSessionID EnnMediumInterface::get_dsp_session_id(const EnnModelId model_id) {
    int32_t ret = 0;
    __START_SERVICE();
#ifdef ENN_MEDIUM_IF_HIDL
    uint32_t modelid_low = static_cast<uint32_t>(model_id >> 32);
    uint32_t modelid_high = static_cast<uint32_t>(model_id & 0xFFFFFFFFF);
    service->custom_interface(static_cast<uint32_t>(CustomFunctionTypeId::GET_DSP_SESSION_ID),
                                            {{modelid_low, modelid_high}, {}},
                                            [&](GeneralParameterReturn ret_service) { ret = ret_service.i32_v[0]; });
#else
    ret = service->get_dsp_session_id(model_id); // not merged yet
#endif
    CHECK_AND_RETURN_ERR(ret, -1, "Get DSP Session Id Failed(model_id: 0x%" PRIX64 "\n", model_id);
    __FINISH_SERVICE();
    return ret;
}

EnnReturn EnnMediumInterface::close_model(const EnnModelId model_id) {
    __START_SERVICE();
    int32_t ret = service->close_model(model_id);
    __FINISH_SERVICE();
    return static_cast<EnnReturn>(ret);
}

EnnExecuteModelId EnnMediumInterface::commit_execution_data(const EnnModelId model_id, const InferenceData& exec_data) {
    __START_SERVICE();
    uint64_t ret = service->commit_execution_data(model_id, exec_data);
    __FINISH_SERVICE();
    return static_cast<EnnExecuteModelId>(ret);
}

EnnReturn EnnMediumInterface::execute_model(const std::vector<EnnModelId> & exec_id_list) {
    int32_t ret = service->execute_model(exec_id_list);
    HIDL_IF(IPCThreadState::self()->flushCommands());
    return static_cast<EnnReturn>(ret);
}

}  // namespace interface
}  // namespace enn
