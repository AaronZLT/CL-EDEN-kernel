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
 * @file enn_api-public.hpp
 * @author Hoon Choi (hoon98.choi@)
 * @brief public header file for enn API supported by C++ syntax
 * @version 0.1
 * @date 2021-06-03
 * @note This file wraps enn_api-public.h
 */

#pragma once

#include "enn_api-type.h"
#include <vector>

namespace enn {
namespace api {

/* Context initialize / deinitialize */
EnnReturn EnnInitialize(void);
EnnReturn EnnDeinitialize(void);

/* OpenModel */
EnnReturn EnnOpenModel(const char* model_file, EnnModelId *model_id);
EnnReturn EnnOpenModelFromMemory(const char* va, const uint32_t size, EnnModelId *model_id);
EnnReturn EnnCloseModel(const EnnModelId model_id);

/* Memory Handling */
EnnReturn EnnCreateBuffer(EnnBufferPtr *out, const uint32_t req_size, const bool is_cached = true);
EnnReturn EnnCreateBufferFromFd(EnnBufferPtr *out, const uint32_t fd, const uint32_t size, const uint32_t offset = 0);
#if 0 // disabled
EnnReturn EnnAllocateAllBuffers(std::vector<EnnBufferPtr> & buf_v, const EnnModelId model_id, const uint32_t session_id = 0, const bool do_commit = true);
EnnReturn EnnReleaseBuffers(std::vector<EnnBufferPtr> buffers);
#else
EnnReturn EnnAllocateAllBuffers(const EnnModelId model_id, EnnBufferPtr **out_buffers,
                                             NumberOfBuffersInfo *buf_info, const int session_id = 0,
                                             const bool do_commit = true);
EnnReturn EnnReleaseBuffers(EnnBufferPtr *buffers, const int32_t numOfBuffers);
EnnReturn EnnReleaseBuffer(EnnBufferPtr buffer);
#endif

/* set/get for model */
EnnReturn EnnGetBuffersInfo(NumberOfBuffersInfo *buffers_info, const EnnModelId model_id);
EnnReturn EnnGetBufferInfoByIndex(EnnBufferInfo* out_buf_info, const EnnModelId model_id, const enn_buf_dir_e direction, const uint32_t index);
EnnReturn EnnGetBufferInfoByLabel(EnnBufferInfo* out_buf_info, const EnnModelId model_id, const char* label);
EnnReturn EnnSetBufferByIndex(const EnnModelId model_id, const enn_buf_dir_e direction, const uint32_t index, EnnBufferPtr buf, const int session_id = 0);
EnnReturn EnnSetBufferByLabel(const EnnModelId model_id, const char* label, EnnBufferPtr buf, const int session_id = 0);
EnnReturn EnnSetBuffers(const EnnModelId model_id, EnnBuffer** bufs, const int32_t sum_io, const int session_id = 0);

/* Commit Buffers */
// NOTE(hoon98.choi): Generate Buffer Space will be hided, OpenModel Generates Space with MAX_SET (16)
// NOTE(hoon98.choi): User can commit multiple sessions
EnnReturn EnnGenerateBufferSpace(const EnnModelId model_id, const int n_set = 1);
EnnReturn EnnBufferCommit(const EnnModelId model_id, const int session_id = 0);

/* Execute Model */
EnnReturn EnnExecuteModel(const EnnModelId model_id, const int session_id = 0);
EnnReturn EnnExecuteModelAsync(const EnnModelId model_id, const int session_id = 0);
EnnReturn EnnExecuteModelWait(const EnnModelId model_id, const int session_id = 0);

/* Security */
// TODO(hoon98.choi, 6/25): Secure Open / close
EnnReturn EnnSecureOpen(const uint32_t heap_size, uint64_t* secure_heap_addr);
EnnReturn EnnSecureClose(void);

/* Prefrences */
/* setter */
EnnReturn EnnSetPreferencePresetId(const uint32_t val);
EnnReturn EnnSetPreferencePerfMode(const uint32_t val);

/* getter */
EnnReturn EnnGetPreferencePresetId(uint32_t *val_ptr);
EnnReturn EnnGetPreferencePerfMode(uint32_t *val_ptr);

EnnReturn EnnDspGetSessionId(const EnnModelId model_id, int32_t *out);

/* Get meta */
EnnReturn EnnGetMetaInfo(const EnnInfoId info_id, char *output_str);

}
}

