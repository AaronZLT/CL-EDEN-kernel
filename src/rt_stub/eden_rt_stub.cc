/*
 * Copyright 2018 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file eden_rt_stub.cpp
 * @brief implementation of eden_rt_stub api
 */

#include "client/enn_api-public.h"
#include "EdenRuntime.h"
//#include "common/enn_debug.h"
#include <stdio.h>
#include<unistd.h>
#include <iostream>
#include<vector>
#include<unordered_map>
#include<future>

#ifdef LOG_TAG
#undef LOG_TAG
#endif

namespace eden {
namespace rt {


#define MAX_ID 2000000000

std::unordered_map<uint32_t, EnnModelId> eden_modelid_to_enn_modelid;
std::unordered_map<EdenBuffer*, std::vector<EnnBufferPtr>> eden_buffer_to_enn_buffer;
std::unordered_map<EdenBuffer*, int> eden_buffer_to_session_id;
std::unordered_map<uint32_t, uint32_t> eden_model_id_to_curr_session_id;

uint32_t generate_eden_model_id(EnnModelId m_id) {
    static uint32_t eden_model_id = 0;
    while(eden_modelid_to_enn_modelid.find(eden_model_id) != eden_modelid_to_enn_modelid.end()) {
        eden_model_id = (eden_model_id + 1)%MAX_ID;
    }
    eden_modelid_to_enn_modelid.insert({eden_model_id, m_id});
    return eden_model_id;
}

/**
 *  @brief Init EDEN Runtime
 *  @details This API function initializes the CPU/GPU/NPU/DSP handler.
 *  @param void
 *  @returns return code
 */
RtRet Init(void) {
    std::cout<<"INIT called \n";
    ////ENN_INFO_PRINT("EDEN_RT_STUB : Init Started \n");
    // Get current process id
    pid_t stub_pid = getpid();
    // Print out pid
    //ENN_DBG_PRINT("EDEN_RT_STUB : stub_pid=[%d]\n", stub_pid);

    EnnInitialize();

    return RT_SUCCESS;
}

RtRet Init(uint32_t target) {
    //ENN_DBG_PRINT("EDEN_RT_STUB :Not support init with sepecific target device in stub layer\n");
    return Init();
}

/**
 *  @brief Open a model file and generates an in-memory model structure
 *  @details This function reads a model file and construct an in-memory model structure.
 *           The model file should be one of the supported model file format.
 *           Once it successes to parse a given model file,
 *           unique model id is returned via modelId.
 *  @param[in] modelFile It is representing for EDEN model file such as file path.
 *  @param[out] modelId It is representing for constructed EdenModel with a unique id.
 *  @param[in] preference It is representing for a model preference.
 *  @returns return code
 */
RtRet OpenModelFromFile(EdenModelFile* modelFile, uint32_t* modelId, ModelPreference preference) {
    //ENN_DBG_PRINT("EDEN_RT_STUB : Deprecated function\n");
    EdenModelOptions options;
    return OpenModelFromFile(modelFile, modelId, options);
}

RtRet OpenModelFromFile(EdenModelFile* modelFile, uint32_t* modelId, const EdenModelOptions& options) {
    //ENN_INFO_PRINT("EDEN_RT_STUB : OpenModelFromFile with modelFile: %s\n", modelFile->pathToModelFile);
    // Get current process id
    std::cout<<"OpenModelFromFile called \n";
    pid_t stub_pid = getpid();
    // Print out pid
    //ENN_DBG_PRINT("EDEN_RT_STUB : stub_pid=[%d]\n", stub_pid);

    RtRet result = RT_SUCCESS;

    char * model_file_name = (char*)modelFile->pathToModelFile;
    EnnModelId model_id;
    EnnReturn ret = EnnOpenModel(model_file_name, &model_id);
    if(ret) {
        return RT_FAILED;
    }
    *modelId = generate_eden_model_id(model_id);
    eden_model_id_to_curr_session_id[*modelId] = 0;
    EnnGenerateBufferSpace(model_id, 16);
    //ENN_INFO_PRINT("EDEN_RT_STUB : modelId: %d, result: %d (-)\n", *modelId, result);
    return result;
}

/**
 *  @brief Read a in-memory model on address and open it as a EdenModel
 *  @details This function reads a in-memory model on a given address and convert it to EdenModel.
 *           The in-memory model should be one of the supported model type in memory.
 *           Once it successes to parse a given in-memory model,
 *           unique model id is returned via modelId.
 *  @param[in] modelTypeInMemory it is representing for in-memory model such as Android NN Model.
 *  @param[in] addr address of in-memory model
 *  @param[in] size size of in-memory model
 *  @param[in] encrypted data on addr is encrypted
 *  @param[out] modelId It is representing for constructed EdenModel with a unique id.
 *  @param[in] preference It is representing for a model preference.
 *  @returns return code
 */
RtRet OpenModelFromMemory(ModelTypeInMemory modelTypeInMemory, int8_t* addr, int32_t size,
                          bool encrypted, uint32_t* modelId, ModelPreference preference) {
    //ENN_DBG_PRINT(EDEN_RT_STUB, "Deprecated function\n");
    EdenModelOptions options;
    return OpenModelFromMemory(modelTypeInMemory, addr, size, encrypted, modelId, options);
}

RtRet OpenModelFromMemory(ModelTypeInMemory modelTypeInMemory, int8_t* addr, int32_t size, bool encrypted,
                          uint32_t* modelId, const EdenModelOptions& options) {
    //ENN_INFO_PRINT("EDEN_RT_STUB : OpenModelFromMemory\n");
    std::cout<<"Openmodel from memory \n";
    RtRet result = RT_SUCCESS;
    const char* va = (const char*)addr;
    EnnModelId model_id;
    EnnReturn ret = EnnOpenModelFromMemory(va, size, &model_id);
    if(ret) {
        return RT_FAILED;
    }
    *modelId = generate_eden_model_id(model_id);
    eden_model_id_to_curr_session_id[*modelId] = 0;
    EnnGenerateBufferSpace(model_id, 16);
    return result;
}

/**
 *  @brief Allocate a buffer for input to execute a model
 *  @details This function allocates an efficient buffer to execute a model.
 *  @param[in] modelId The model id to be applied by.
 *  @param[out] buffers Array of EdenBuffers for input
 *  @param[out] numOfBuffers # of buffers
 *  @returns return code
 */
RtRet AllocateInputBuffers(uint32_t modelId,
                           EdenBuffer** buffers,
                           int32_t* numOfBuffers) {
    //ENN_INFO_PRINT("EDEN_RT_STUB : AllocateInputBuffers \n");
    std::cout<<"AllocateInputBuffers called \n";
    RtRet result = RT_SUCCESS;

    // Get current process id
    pid_t stub_pid = getpid();
    // Print out pid
    //ENN_DBG_PRINT("EDEN_RT_STUB : stub_pid=[%d]\n", stub_pid);
    if(eden_modelid_to_enn_modelid.find(modelId) == eden_modelid_to_enn_modelid.end()) {
        return RT_FAILED;
    }
    EnnModelId model_id = eden_modelid_to_enn_modelid[modelId];
    NumberOfBuffersInfo buf_info;
    EnnGetBuffersInfo(model_id, &buf_info);
    *numOfBuffers = buf_info.n_in_buf;
    std::vector<EnnBufferPtr> bufs(*numOfBuffers);
    EdenBuffer* eden_buffers  = new EdenBuffer[*numOfBuffers];
    EnnBufferInfo tmp_buf_info;
    for (int idx = 0; idx < buf_info.n_in_buf; idx++) {
        EnnGetBufferInfoByIndex(model_id, ENN_DIR_IN, idx, &tmp_buf_info);
        EnnCreateBufferCache(tmp_buf_info.size, &(bufs[idx]));
        EnnSetBufferByIndexWithSessionId(model_id, ENN_DIR_IN, idx, bufs[idx], eden_model_id_to_curr_session_id[modelId]);
        eden_buffers[idx].addr = bufs[idx]->va;
        eden_buffers[idx].size = bufs[idx]->size;
    }
    *buffers = eden_buffers;
    eden_buffer_to_enn_buffer[eden_buffers] = bufs;
    //ENN_INFO_PRINT("EDEN_RT_STUB : AllocateInputBuffers Done \n");
    return result;
}

/**
 *  @brief Allocate a buffer for output to execute a model
 *  @details This function allocates an efficient buffer to execute a model.
 *  @param[in] modelId The model id to be applied by.
 *  @param[out] buffers Array of EdenBuffers for input
 *  @param[out] numOfBuffers # of buffers
 *  @returns return code
 */
RtRet AllocateOutputBuffers(uint32_t modelId,
                            EdenBuffer** buffers,
                            int32_t* numOfBuffers) {
    //ENN_INFO_PRINT("EDEN_RT_STUB : AllocateOutputBuffers (+) \n");
    std::cout<<"AllocateOutputBuffers called \n";
    RtRet result = RT_SUCCESS;

    // Get current process id
    pid_t stub_pid = getpid();
    // Print out pid
    //ENN_DBG_PRINT("EDEN_RT_STUB stub_pid=[%d]\n", stub_pid);
    if(eden_modelid_to_enn_modelid.find(modelId) == eden_modelid_to_enn_modelid.end()) {
        return RT_FAILED;
    }
    EnnModelId model_id = eden_modelid_to_enn_modelid[modelId];
    NumberOfBuffersInfo buf_info;
    EnnGetBuffersInfo(model_id, &buf_info);
    *numOfBuffers = buf_info.n_out_buf;
    std::vector<EnnBufferPtr> bufs(*numOfBuffers);
    EdenBuffer* eden_buffers  = new EdenBuffer[*numOfBuffers];
    EnnBufferInfo tmp_buf_info;
    for (int idx = 0; idx < buf_info.n_out_buf; idx++) {
        EnnGetBufferInfoByIndex(model_id, ENN_DIR_OUT, idx, &tmp_buf_info);
        EnnCreateBufferCache(tmp_buf_info.size, &(bufs[idx]));
        EnnSetBufferByIndexWithSessionId(model_id, ENN_DIR_OUT, idx, bufs[idx], eden_model_id_to_curr_session_id[modelId]);
        eden_buffers[idx].addr = bufs[idx]->va;
        eden_buffers[idx].size = bufs[idx]->size;
    }
    eden_buffer_to_session_id[eden_buffers] = eden_model_id_to_curr_session_id[modelId];
    *buffers = eden_buffers;
    eden_buffer_to_enn_buffer[eden_buffers] = bufs;
    EnnBufferCommitWithSessionId(model_id, eden_model_id_to_curr_session_id[modelId]++);
    //ENN_INFO_PRINT("EDEN_RT_STUB : AllocateOutputBuffers (-) \n");
    return (RtRet)result;
}

/**
 * @brief Load a buffer of external to execute a model
 * @details This function loads to execute a buffer.
 * @param[in] modelId The model id to be applied by.
 * @param[in] Array of Userbuffers for input
 * @param[in] numOfBuffers # of buffers.
 * @param[out] buffers Array of EdenBuffers for input
 * @returns return code
 */
RtRet LoadInputBuffers(uint32_t modelId, UserBuffer* userBuffers, int32_t numOfBuffers, EdenBuffer** edenBuffers) {
    //ENN_INFO_PRINT("EDEN_RT_STUB : LoadInputBuffers (+)\n");

    RtRet result = RT_SUCCESS;

    // Get current process id
    pid_t stub_pid = getpid();
    // Print out pid
    //ENN_DBG_PRINT("EDEN_RT_STUB : stub_pid=[%d]\n", stub_pid);

    std::vector<EnnBufferPtr> enn_buffers;
    EdenBuffer* eden_buffers = new EdenBuffer[numOfBuffers];
    EnnModelId model_id = eden_modelid_to_enn_modelid[modelId];
    for (int32_t idx = 0; idx < numOfBuffers; idx++) {
        EnnBufferPtr enn_buffer;
        EnnCreateBufferFromFd(userBuffers[idx].elem.fd, userBuffers[idx].size, &enn_buffer);
        EnnSetBufferByIndexWithSessionId(model_id, ENN_DIR_IN, idx, enn_buffer, eden_model_id_to_curr_session_id[modelId]);
        enn_buffers.push_back(enn_buffer);
        eden_buffers[idx].addr = enn_buffer->va;
        eden_buffers[idx].size = enn_buffer->size;
    }
    eden_buffer_to_enn_buffer[eden_buffers] = enn_buffers;
    eden_buffer_to_session_id[eden_buffers] = eden_model_id_to_curr_session_id[modelId];
    *edenBuffers = eden_buffers;
    //ENN_INFO_PRINT("EDEN_RT_STUB : LoadInptuBuffers (-) \n");
    return (RtRet)result;
}

/**
 * @brief Load a buffer of external to execute a model
 * @details This function loads to execute a buffer.
 * @param[in] modelId The model id to be applied by.
 * @param[in] Array of Userbuffers for input
 * @param[in] numOfBuffers # of buffers.
 * @param[out] buffers Array of EdenBuffers for output
 * @returns return code
 */
RtRet LoadOutputBuffers(uint32_t modelId, UserBuffer* userBuffers, int32_t numOfBuffers, EdenBuffer** edenBuffers) {
    //ENN_INFO_PRINT("EDEN_RT_STUB : LoadOutputBuffers (+) \n")_;

    RtRet result = RT_SUCCESS;

    // Get current process id
    pid_t stub_pid = getpid();
    // Print out pid
    //ENN_DBG_PRINT("EDEN_RT_STUB : stub_pid=[%d]\n", stub_pid);

    std::vector<EnnBufferPtr> enn_buffers;
    EdenBuffer* eden_buffers = new EdenBuffer[numOfBuffers];
    EnnModelId model_id = eden_modelid_to_enn_modelid[modelId];
    for (int32_t idx = 0; idx < numOfBuffers; idx++) {
        EnnBufferPtr enn_buffer;
        EnnCreateBufferFromFd(userBuffers[idx].elem.fd, userBuffers[idx].size, &enn_buffer);
        EnnSetBufferByIndexWithSessionId(model_id, ENN_DIR_OUT, idx, enn_buffer, eden_model_id_to_curr_session_id[modelId]);
        enn_buffers.push_back(enn_buffer);
        eden_buffers[idx].addr = enn_buffer->va;
        eden_buffers[idx].size = enn_buffer->size;
    }
    eden_buffer_to_session_id[eden_buffers] = eden_model_id_to_curr_session_id[modelId];
    eden_buffer_to_enn_buffer[eden_buffers] = enn_buffers;
    *edenBuffers = eden_buffers;
    EnnBufferCommitWithSessionId(model_id, eden_model_id_to_curr_session_id[modelId]++);
    //ENN_INFO_PRINT("EDEN_RT_STUB : LoadOutputBuffers (-)\n");
    return (RtRet)result;
}

/**
 *  @brief Execute EDEN Req
 *  @details This API function executes EdenRequest with preference.
 *  @param[in] req It consists of EDEN Model ID, input/output buffers and callback.
 *  @param[in] evt Callback function defined by User.
 *  @param[in] preference it determines how to run EdenModel with preference.
 *  @returns return code
 */
RtRet ExecuteReq(EdenRequest* req, EdenEvent** evt, RequestPreference preference) {
    //ENN_INFO_PRINT("EDEN_RT_STUB : Deprecated function\n");
    RtRet result = RT_SUCCESS;
    EdenRequestOptions options;
    return ExecuteReq(req, evt, options);
}

RtRet ExecuteModelUtil(EdenRequest* req, EnnModelId model_id, const int session_id) {
    EnnReturn ret = EnnExecuteModelWithSessionId(model_id, session_id);
    if (req->callback->notify == nullptr) {
        std::cout<<"callback function is a null ptr \n";
        return RT_FAILED;
    }
    if (ret != ENN_RET_SUCCESS) {
        std::cout<<" Enn Execute Model Failed \n";
        req->callback->executionResult.inference.retCode = RT_FAILED;
    } else {
        req->callback->executionResult.inference.retCode = RT_SUCCESS;
    }
    req->callback->notify(&(req->callback->requestId), reinterpret_cast<addr_t>(req));
    return RT_SUCCESS;
}

RtRet ExecuteReq(EdenRequest* req, EdenEvent** evt, const EdenRequestOptions& options) {
    //ENN_INFO_PRINT("EDEN_RT_STUB : ExecuteReq (+)\n");
    std::cout<<"ExecuteReq called \n";
    RtRet result = RT_SUCCESS;
    pid_t stub_pid = getpid();
    // Print out pid
    //ENN_DBG_PRINT("EDEN_RT_STUB : stub_pid=[%d]\n", stub_pid);
    uint32_t modelId = req->modelId;
    if(eden_modelid_to_enn_modelid.find(modelId) == eden_modelid_to_enn_modelid.end()) {
        return RT_FAILED;
    }
    if(eden_buffer_to_session_id.find(req->outputBuffers) == eden_buffer_to_session_id.end()) {
        return RT_FAILED;
    }
    EnnModelId model_id = eden_modelid_to_enn_modelid[model_id];
    int session_id = eden_buffer_to_session_id[req->outputBuffers];
    auto future = std::async(std::launch::async, ExecuteModelUtil, req, model_id, session_id);
    //ENN_INFO_PRINT("EDEN_RT_STUB : ExecuteReq (-)\n");
    return (RtRet)result;
}

/**
 *  @brief Release a buffer allocated by Eden framework
 *  @details This function releases a buffer returned by AllocateXXXBuffers.
 *  @param[in] modelId The model id to be applied by.
 *  @param[in] buffers Buffer pointer allocated by AllocateXXXBuffers
 *  @returns return code
 */
RtRet FreeBuffers(uint32_t modelId, EdenBuffer* buffers) {
    //ENN_INFO_PRINT("EDEN_RT_STUB : FreeBuffers (+)\n");
    std::cout<<"Freebuffers called \n";
    pid_t stub_pid = getpid();
    // Print out pid
    //ENN_DBG_PRINT("EDEN_RT_STUB : stub_pid=[%d]\n", stub_pid);
    if(eden_buffer_to_enn_buffer.find(buffers) == eden_buffer_to_enn_buffer.end()) {
        return RT_FAILED;
    }
    std::vector<EnnBufferPtr> enn_buffers = eden_buffer_to_enn_buffer[buffers];
    EnnReturn ret_tmp;

    for (auto buf : enn_buffers) {
        ret_tmp = EnnReleaseBuffer(buf);
        if(ret_tmp)
            return RT_FAILED;
    }
    eden_buffer_to_enn_buffer.erase(buffers);
    eden_buffer_to_session_id.erase(buffers);
    //ENN_INFO_PRINT("EDEN_RT_STUB : FreeBuffers (-)\n");
    return RT_SUCCESS;
}

/**
 *  @brief Close EDEN Model
 *  @details This API function releases resources related with the EDEN Model.
 *  @param[in] modelId It is a unique id for EDEN Model.
 *  @returns return code
 */
RtRet CloseModel(uint32_t modelId) {
    //ENN_INFO_PRINT("EDEN_RT_STUB : CloseModel (+)\n");
    std::cout<<"CloseModel called \n";
    // Get current process id
    pid_t stub_pid = getpid();
    // Print out pid
    //ENN_DBG_PRINT("EDEN_RT_STUB : stub_pid=[%d]\n", stub_pid);

    uint32_t result = RT_SUCCESS;
    if(eden_modelid_to_enn_modelid.find(modelId) == eden_modelid_to_enn_modelid.end()) {
        return RT_FAILED;
    }
    EnnModelId model_id = eden_modelid_to_enn_modelid[modelId];
    eden_modelid_to_enn_modelid.erase(modelId);
    eden_model_id_to_curr_session_id.erase(modelId);
    EnnCloseModel(model_id);
    //ENN_INFO_PRINT("EDEN_RT_STUB : CloseModel(-)\n");
    return RT_SUCCESS;
}

/**
 *  @brief Shutdown EDEN Runtime
 *  @details This API function close all EDEN Models with related resources for shutdown EDEN Framework.
 *  @param void
 *  @returns return code
 */
RtRet Shutdown(void) {
    //ENN_INFO_PRINT("EDEN_RT_STUB : Shutdown (+)\n");

    // Get current process id
    pid_t stub_pid = getpid();
    // Print out pid
    //ENN_DBG_PRINT("EDEN_RT_STUB : stub_pid=[%d]\n", stub_pid);

    EnnDeinitialize();
    //ENN_INFO_PRINT("EDEN_RT_STUB : Shutdown (-)\n");
    return RT_SUCCESS;
}

/**
 *  @brief Get the input buffer information
 *  @details This function gets buffer shape for input buffer of a specified model.
 *  @param[in] modelId The model id to be applied by.
 *  @param[in] inputIndex Input index starting 0.
 *  @param[out] width Width
 *  @param[out] height Height
 *  @param[out] channel Channel
 *  @param[out] number Number
 *  @returns return code
 */
RtRet GetInputBufferShape(uint32_t modelId, int32_t inputIndex, int32_t* width, int32_t* height, int32_t* channel, int32_t* number) {
    //ENN_INFO_PRINT("EDEN_RT_STUB : GetInputBufferShape (+)\n");

    // Get current process id
    pid_t stub_pid = getpid();
    // Print out pid
    //ENN_DBG_PRINT("EDEN_RT_STUB : stub_pid=[%d]\n", stub_pid);
    RtRet result = RT_SUCCESS;
    if(eden_modelid_to_enn_modelid.find(modelId) == eden_modelid_to_enn_modelid.end()) {
        return RT_FAILED;
    }
    EnnModelId model_id = eden_modelid_to_enn_modelid[modelId];
    EnnBufferInfo tmp_buf_info;
    EnnGetBufferInfoByIndex(model_id, ENN_DIR_IN, inputIndex, &tmp_buf_info);
    *width = tmp_buf_info.width;
    *height = tmp_buf_info.height;
    *channel = tmp_buf_info.channel;
    *number = tmp_buf_info.n;
    //ENN_INFO_PRINT("EDEN_RT_STUB : GetInputBufferShape (-)\n");
    return result;
}

/**
 *  @brief Get the output buffer information
 *  @details This function gets buffer shape for output buffers of a specified model.
 *  @param[in] modelId The model id to be applied by.
 *  @param[in] outputIndex Output index starting 0.
 *  @param[out] width Width
 *  @param[out] height Height
 *  @param[out] channel Channel
 *  @param[out] number Number
 *  @returns return code
 */
RtRet GetOutputBufferShape(uint32_t modelId, int32_t outputIndex, int32_t* width, int32_t* height, int32_t* channel, int32_t* number) {
    //ENN_INFO_PRINT("EDEN_RT_STUB : GetOutputBufferShape (+)\n");

    // Get current process id
    pid_t stub_pid = getpid();
    // Print out pid
    //ENN_DBG_PRINT("EDEN_RT_STUB : stub_pid=[%d]\n", stub_pid);

    RtRet result = RT_SUCCESS;
    if(eden_modelid_to_enn_modelid.find(modelId) == eden_modelid_to_enn_modelid.end()) {
        return RT_FAILED;
    }
    EnnModelId model_id = eden_modelid_to_enn_modelid[modelId];
    EnnBufferInfo tmp_buf_info;
    EnnGetBufferInfoByIndex(model_id, ENN_DIR_OUT, outputIndex, &tmp_buf_info);
    *width = tmp_buf_info.width;
    * height = tmp_buf_info.height;
    *channel = tmp_buf_info.channel;
    *number = tmp_buf_info.n;


    //ENN_INFO_PRINT("EDEN_RT_STUB : GetOutputBufferShape (-)\n");
    return (RtRet)result;
}

RtRet GetEdenVersion(uint32_t modelId, int32_t* versions) {
    //ENN_INFO_PRINT("EDEN_RT_STUB : GetEdenVersion Not Supported by ENN Framework \n");
    return RT_SUCCESS;
}

RtRet GetCompileVersion(uint32_t modelId, EdenModelFile* modelFile, char versions[][256]) {
    //ENN_INFO_PRINT("EDEN_RT_STUB : GetCompileVersion  Not Supported by ENN framework \n");
    return RT_SUCCESS;
}

RtRet GetCompileVersionFromMemory(ModelTypeInMemory typeInMemory, int8_t* addr, int32_t size, bool encrypted,
                                    char versions[][VERSION_LENGTH_MAX]) {
    //ENN_INFO_PRINT("EDEN_RT_STUB : GetCompileVersionFromMemory Not Supported by ENN Framework \n");
    return RT_SUCCESS;
}

#if 0
RtRet GetState(EdenState* state) {
    //ENN_DBG_PRINT(EDEN_RT_STUB, "%s started\n", __func__);

    if (service == NULL) {
        //ENN_DBG_PRINT(EDEN_RT_STUB, "service is null\n");
        return RT_FAILED;
    }
}
#endif

}  // rt
}  // eden
