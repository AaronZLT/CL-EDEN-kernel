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
 * @file enn_api-public.cc
 * @author Hoon Choi (hoon98.choi@)
 * @brief public header file for enn API supported by C++ syntax
 * @version 0.1
 * @date 2021-06-03
 * @note This file wraps enn_api-public.h
 */

#include "enn_api-public.h"
#include "enn_api-public.hpp"
#include "enn_api.h"
#include <cstdint>

#ifdef _DEBUG_CALL
#include <cstdio>
#define DEBUG_PRINT_API(message, ...) printf("[DEBUG_CALLED] %s(%d): " message, __func__, __LINE__, ##__VA_ARGS__)
#else
#define DEBUG_PRINT_API(...)
#endif

namespace enn {
namespace api {

/* Context initialize / deinitialize */
EnnReturn EnnInitialize(void) {
    return ::EnnInitialize();
}

EnnReturn EnnDeinitialize(void) {
    return ::EnnDeinitialize();
}

/* OpenModel */
EnnReturn EnnOpenModel(const char* model_file, EnnModelId *model_id) {
    DEBUG_PRINT_API("model_file(%s)\n", model_file);
    return ::EnnOpenModel(model_file, model_id);
}

EnnReturn EnnOpenModelFromMemory(const char* va, const uint32_t size, EnnModelId *model_id) {
    DEBUG_PRINT_API("model_file_from_memory(%p, %d)\n", va, size);
    return ::EnnOpenModelFromMemory(va, size, model_id);
}

EnnReturn EnnCloseModel(const EnnModelId model_id) {
    DEBUG_PRINT_API("model_id : 0x%llu\n", model_id);
    return ::EnnCloseModel(model_id);
}

/* Memory Handling */
EnnReturn EnnCreateBuffer(EnnBufferPtr *out, const uint32_t req_size, const bool is_cached) {
    DEBUG_PRINT_API("out: %p, reqsize: %d, is-cached: %d\n", out, req_size, is_cached);
    if (is_cached)        return ::EnnCreateBufferCache(req_size, out);
    return ::EnnCreateBuffer(req_size, 0, out);
}

EnnReturn EnnCreateBufferFromFd(EnnBufferPtr *out, const uint32_t fd, const uint32_t size, const uint32_t offset) {
    DEBUG_PRINT_API("out: %p, fd: %u, size: %u, offset: %u\n", out, fd, size, offset);
    return ::EnnCreateBufferFromFdWithOffset(fd, size, offset, out);
}

EnnReturn EnnAllocateAllBuffers(const EnnModelId model_id, EnnBufferPtr **out_buffers,
                                             NumberOfBuffersInfo *buf_info, const int session_id,
                                             const bool do_commit) {
    return ::EnnAllocateAllBuffersWithSessionId(model_id, out_buffers, buf_info, session_id, do_commit);
}

EnnReturn EnnReleaseBuffers(EnnBufferPtr *buffers, const int32_t numOfBuffers) {
    return ::EnnReleaseBuffers(buffers, numOfBuffers);
}

EnnReturn EnnReleaseBuffer(EnnBufferPtr buffer) {
    return ::EnnReleaseBuffer(buffer);
}

#if 0  // note: Because NDK<-->clang doesn't support vector type eachother, we change this to double-pointer
EnnReturn EnnAllocateAllBuffers(std::vector<EnnBufferPtr> &buf_v, const EnnModelId model_id, const uint32_t session_id,
                                const bool do_commit) {
    EnnBufferSet buffer_set;
    NumberOfBuffersInfo buf_info;

    DEBUG_PRINT_API("# of buf_v: %zu, model id: 0x%llu, session_id: %lu, do_commit: %d\n", buf_v.size(), model_id,
                    session_id, do_commit);

    auto ret = ::EnnAllocateAllBuffersWithSessionId(model_id, &buffer_set, &buf_info, session_id, do_commit);
    if (ret)
        return ret;

    std::vector<EnnBufferPtr> ret_vec;
    for (int i = 0; i < buf_info.n_in_buf + buf_info.n_out_buf; i++)
        ret_vec.push_back(buffer_set[i]);
    buf_v = ret_vec;

    return ret;
}

EnnReturn EnnReleaseBuffers(std::vector<EnnBufferPtr> buffers) {
    EnnReturn ret = ENN_RET_SUCCESS, ret_tmp;

    DEBUG_PRINT_API("# of buf_v: %zu\n", buffers.size());

    for (auto buf : buffers) {
        ret_tmp = EnnReleaseBuffer(buf);
        if(ret_tmp)
            ret = ENN_RET_FAILED;
    }
    return ENN_RET_SUCCESS;
}
#endif

/* set/get for model */
EnnReturn EnnGetBuffersInfo(NumberOfBuffersInfo *buffers_info, const EnnModelId model_id) {
    DEBUG_PRINT_API("buffers_info: %p Model ID: 0x%llu\n", buffers_info, model_id);

    return ::EnnGetBuffersInfo(model_id, buffers_info);
}

EnnReturn EnnGetBufferInfoByIndex(EnnBufferInfo *out_buf_info, const EnnModelId model_id, const enn_buf_dir_e direction,
                                  const uint32_t index) {
    DEBUG_PRINT_API("out_buf_info: %p Model ID: 0x%llu, direction: %d, index: %lu\n", out_buf_info, model_id, direction, index);
    return ::EnnGetBufferInfoByIndex(model_id, direction, index, out_buf_info);
}
EnnReturn EnnGetBufferInfoByLabel(EnnBufferInfo *out_buf_info, const EnnModelId model_id, const char *label) {
    DEBUG_PRINT_API("out_buf_info: %p Model ID: 0x%llu, label: %s\n", out_buf_info, model_id, label);

    return ::EnnGetBufferInfoByLabel(model_id, label, out_buf_info);
}

EnnReturn EnnSetBufferByIndex(const EnnModelId model_id, const enn_buf_dir_e direction, const uint32_t index, EnnBufferPtr buf, const int session_id) {
    DEBUG_PRINT_API("");
    return ::EnnSetBufferByIndexWithSessionId(model_id, direction, index, buf, session_id);
}

EnnReturn EnnSetBufferByLabel(const EnnModelId model_id, const char* label, EnnBufferPtr buf, const int session_id) {
    DEBUG_PRINT_API("");
    return ::EnnSetBufferByLabelWithSessionId(model_id, label, buf, session_id);
}

EnnReturn EnnSetBuffers(const EnnModelId model_id, EnnBuffer** bufs, const int32_t sum_io, const int session_id) {
    DEBUG_PRINT_API("");
    return ::EnnSetBuffersWithSessionId(model_id, bufs, sum_io, session_id);
}

/* Commit Buffers */
EnnReturn EnnGenerateBufferSpace(const EnnModelId model_id, const int n_set) {
    DEBUG_PRINT_API("");
    return ::EnnGenerateBufferSpace(model_id, n_set);
}

EnnReturn EnnBufferCommit(const EnnModelId model_id, const int session_id) {
    DEBUG_PRINT_API("");
    return ::EnnBufferCommitWithSessionId(model_id, session_id);
}


/* Execute Model */
EnnReturn EnnExecuteModel(const EnnModelId model_id, const int session_id) {
    DEBUG_PRINT_API("");
    return ::EnnExecuteModelWithSessionId(model_id, session_id);
}

EnnReturn EnnExecuteModelAsync(const EnnModelId model_id, const int session_id) {
    DEBUG_PRINT_API("");
    return ::EnnExecuteModelWithSessionIdAsync(model_id, session_id);
}

EnnReturn EnnExecuteModelWait(const EnnModelId model_id, const int session_id) {
    DEBUG_PRINT_API("");
    return ::EnnExecuteModelWithSessionIdWait(model_id, session_id);
}

/* Secure related */
EnnReturn EnnSecureOpen(const uint32_t heap_size, uint64_t* secure_heap_addr) {
    DEBUG_PRINT_API("");
    return ::EnnSecureOpen(heap_size, secure_heap_addr);
}

EnnReturn EnnSecureClose(void) {
    DEBUG_PRINT_API("");
    return ::EnnSecureClose();
}

EnnReturn EnnSetPreferencePresetId(const uint32_t val) {
    DEBUG_PRINT_API("");
    return ::EnnSetPreferencePresetId(val);
}

EnnReturn EnnSetPreferencePerfMode(const uint32_t val) {
    DEBUG_PRINT_API("");
    return ::EnnSetPreferencePerfMode(val);
}

EnnReturn EnnGetPreferencePresetId(uint32_t *val_ptr) {
    DEBUG_PRINT_API("");
    return ::EnnGetPreferencePresetId(val_ptr);
}

EnnReturn EnnGetPreferencePerfMode(uint32_t *val_ptr) {
    DEBUG_PRINT_API("");
    return ::EnnGetPreferencePerfMode(val_ptr);
}

EnnReturn EnnSetPreferenceTargetLatency(const uint32_t val) {
    DEBUG_PRINT_API("");
    return ::EnnSetPreferenceTargetLatency(val);
}

EnnReturn EnnGetPreferenceTargetLatency(uint32_t *val_ptr) {
    DEBUG_PRINT_API("");
    return ::EnnGetPreferenceTargetLatency(val_ptr);
}

#if 0  // These APIs are not opened

extern EnnReturn ::EnnSetPreferenceTargetLatency(const uint32_t val);
extern EnnReturn ::EnnSetPreferenceTileNum(const uint32_t val);
extern EnnReturn ::EnnSetPreferenceCoreAffinity(const uint32_t val);
extern EnnReturn ::EnnSetPreferencePriority(const uint32_t val);

/* getter */
extern EnnReturn ::EnnGetPreferenceTargetLatency(uint32_t *val_ptr);
extern EnnReturn ::EnnGetPreferenceTileNum(uint32_t *val_ptr);
extern EnnReturn ::EnnGetPreferenceCoreAffinity(uint32_t *val_ptr);
extern EnnReturn ::EnnGetPreferencePriority(uint32_t *val_ptr);

/* Reset as default */
extern EnnReturn ::EnnResetPreferenceAsDefault();

EnnReturn EnnResetPreferenceAsDefault() {
    DEBUG_PRINT_API("");
    return ::EnnResetPreferenceAsDefault();
}

EnnReturn EnnSetPreferencePresetId(const uint32_t val) {
    DEBUG_PRINT_API("");
    return ::EnnSetPreferencePresetId(val);
}

EnnReturn EnnSetPreferencePerfMode(const uint32_t val) {
    DEBUG_PRINT_API("");
    return ::EnnSetPreferencePerfMode(val);
}

EnnReturn EnnSetPreferenceTileNum(const uint32_t val) {
    DEBUG_PRINT_API("");
    return ::EnnSetPreferenceTileNum(val);
}

EnnReturn EnnSetPreferenceCoreAffinity(const uint32_t val) {
    DEBUG_PRINT_API("");
    return ::EnnSetPreferenceCoreAffinity(val);
}

EnnReturn EnnSetPreferencePriority(const uint32_t val) {
    DEBUG_PRINT_API("");
    return ::EnnSetPreferencePriority(val);
}

EnnReturn EnnSetPreferenceCustom_0(const uint32_t val) {
    DEBUG_PRINT_API("");
    return ::EnnSetPreferenceCustom_0(val);
}

EnnReturn EnnSetPreferenceCustom_1(const uint32_t val) {
    DEBUG_PRINT_API("");
    return ::EnnSetPreferenceCustom_1(val);
}

EnnReturn EnnGetPreferenceTileNum(uint32_t *val_ptr) {
    DEBUG_PRINT_API("");
    return ::EnnGetPreferenceTileNum(val_ptr);
}

EnnReturn EnnGetPreferenceCoreAffinity(uint32_t *val_ptr) {
    DEBUG_PRINT_API("");
    return ::EnnGetPreferenceCoreAffinity(val_ptr);
}

EnnReturn EnnGetPreferencePriority(uint32_t *val_ptr) {
    DEBUG_PRINT_API("");
    return ::EnnGetPreferencePriority(val_ptr);
}

EnnReturn EnnGetPreferencePresetId(uint32_t *val_ptr) {
    DEBUG_PRINT_API("");
    return ::EnnGetPreferencePresetId(val_ptr);
}

EnnReturn EnnGetPreferencePerfMode(uint32_t *val_ptr) {
    DEBUG_PRINT_API("");
    return ::EnnGetPreferencePrefMode(val_ptr);
}

EnnReturn EnnGetPreferenceCustom_0(uint32_t *val_ptr) {
    DEBUG_PRINT_API("");
    return ::EnnGetPreferenceCustom_0(val_ptr);
}

EnnReturn EnnGetPreferenceCustom_1(uint32_t *val_ptr) {
    DEBUG_PRINT_API("");
    return ::EnnGetPreferenceCustom_1(val_ptr);
}

#endif  // not opened yet.

EnnReturn EnnDspGetSessionId(const EnnModelId model_id, int32_t *out) {
    return ::EnnDspGetSessionId(model_id, out);
}

EnnReturn EnnGetMetaInfo(const EnnInfoId info_id, char *output_str) {
    return ::EnnGetMetaInfo(info_id, output_str);
}

}
}