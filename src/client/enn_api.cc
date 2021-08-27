/**
 * @file enn_api.cc
 * @author SWAT
 * @brief API implementation for ENN
 * @version 0.1
 * @date 2020-12-30
 *
 * @copyright Copyright (c) 2020
 *
 */
#include "client/enn_api.h"

#include <atomic>
#include <future>
#include <memory>
#include <string>
#include <tuple>
#include <cinttypes>
#include <unistd.h>

#include "client/enn_context_manager.h"
#include "client/enn_client_parser.hpp"
#include "common/enn_memory_manager.h"
#include "medium/enn_medium_interface.h"
#include "common/enn_utils_buffer.hpp"
#include "common/enn_preference_generator.hpp"
#include "tool/profiler/include/ExynosNnProfilerApi.h"

enn::client::EnnContextManager enn_context;

EnnModelId EnnOpenModelExtension(const char *model_file, const EnnModelExtendedPreference &preference) {
    ENN_UNUSED(model_file);
    ENN_UNUSED(preference);
    return 0;
}

EnnReturn EnnExecuteModelExtended(const EnnModelId model_id, const EdenExecPreference &preference) {
    ENN_UNUSED(model_id);
    ENN_UNUSED(preference);
    ENN_INFO_PRINT("\n");
    return ENN_RET_SUCCESS;
}

EnnReturn EnnExecuteModelWithMemorySet(const EnnModelId model_id, EnnBufferSet &buffer_set) {
    ENN_UNUSED(model_id);
    ENN_UNUSED(buffer_set);
    ENN_INFO_PRINT("\n");
    return ENN_RET_SUCCESS;
}

EnnReturn EnnExecuteModelExtendedSetWithMemory(const EnnModelId model_id, EnnBufferSet &buffer_set,
                                               const EdenExecPreference &preference) {
    ENN_UNUSED(model_id);
    ENN_UNUSED(buffer_set);
    ENN_UNUSED(preference);
    ENN_INFO_PRINT("\n");
    return ENN_RET_SUCCESS;
}

EnnReturn EnnGetMetaInfo(const EnnInfoId info_id, char output_str[ENN_INFO_GRAPH_STR_LENGTH_MAX]) {
    CHECK_AND_RETURN_ERR(output_str == nullptr, ENN_RET_INVAL, "output_str should be char[256]\n");
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");

    if (info_id == ENN_META_VERSION_COMMIT)
        enn_context.GetCommitInfo().copy(output_str, enn_context.GetCommitInfo().size());
    else if (info_id == ENN_META_VERSION_FRAMEWORK)
        enn_context.GetCommitInfo().copy(output_str, enn_context.GetVersionInfo().size());
    else strcpy(output_str, "NOT DEFINED YET");

    return ENN_RET_SUCCESS;
}

EnnReturn EnnInitialize(void) {
    auto ret = enn_context.init();
    CHECK_AND_RETURN_ERR(ret, ret, "Error from context initialize\n");
    ENN_INFO_PRINT_FORCE("Initialize\n");
    //  enn_context.GetMediumInterface()->Initialize();
    return ENN_RET_SUCCESS;
}

EnnReturn EnnSecureOpen(const uint32_t heap_size, uint64_t *secure_heap_addr) {
    ENN_UNUSED(heap_size);
    ENN_UNUSED(secure_heap_addr);

    return ENN_RET_SUCCESS;
}

EnnReturn EnnSecureClose(void) {
    return ENN_RET_SUCCESS;
}

using BufferPtr = std::shared_ptr<enn::EnnBufferCore>;

/**
 * @brief Exception for open file error
 *
 * @param message   error handling message
 */
class OpenFileException : public std::exception {
    std::string message_;

public:
    OpenFileException(std::string message) : message_(std::string("Error: ") + message) {}

    const char *what() const noexcept override {
        return message_.c_str();
    }
};

#ifndef VELOCE_SOC
#define THROW_OPEN_FILE_ERR(msg) throw OpenFileException(msg)
#else
#define THROW_OPEN_FILE_ERR(msg)                       \
    ENN_ERR_COUT << "Error: " << msg << std::endl;     \
    for (auto &mm : result) {                          \
        enn_context.ccMemoryManager->DeleteMemory(mm); \
    }                                                  \
    return std::vector<BufferPtr>();
#endif

/**
 * @brief EnnOpenFile is static, which is called by another API internally.
 *        This function doesn't care abnormal input cases.
 * @param model_file filename string
 * @param size file size
 * @return std::shared_ptr<EnnBufferCore> memory object by memory manager
 *         empty std::shared_ptr<EnnBufferCore> if failed
 */
static std::vector<std::shared_ptr<enn::EnnBufferCore>> EnnOpenFile(enn::util::BufferReader::UPtrType &modelbuf,
                                                                    uint32_t *out_model_type) {
    std::vector<BufferPtr> result;

    TRY {
        CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, {}, "Context is not initialized\n");

        // Identify model
        enn::model::ModelType model_type;
        auto section = enn::model::ParserUtils::identify_model(modelbuf, &model_type);
        CHECK_AND_RETURN_ERR(section.first < 0, {}, "Model open error\n");

        uint32_t size = section.second - section.first;
        uint32_t offset = section.first;

        // Create ION Memory and load model to the space
        auto loaded_mem = enn_context.ccMemoryManager->CreateMemory(size, enn::EnnMmType::kEnnMmTypeIon);

        if (loaded_mem) {
            result.push_back(loaded_mem);
        } else {
            THROW_OPEN_FILE_ERR("CreateMemory error");
        }

        if (modelbuf->copy_buffer(reinterpret_cast<char *>(loaded_mem->va), size, offset)) {
            THROW_OPEN_FILE_ERR("File Loading Error");   
        }

        if (model_type == enn::model::ModelType::CGO) {
            if (!EnnClientCgoParse(enn_context.ccMemoryManager, modelbuf, loaded_mem->va, size, offset, &result)) {
                THROW_OPEN_FILE_ERR("Cgo client parsing error");
            }
        }

        *out_model_type = static_cast<uint32_t>(model_type);

        return result;
    }
    CATCH(what) {
        ENN_ERR_COUT << what << std::endl;
        for (auto &mm : result) {
            enn_context.ccMemoryManager->DeleteMemory(mm);
        }
        return std::vector<BufferPtr>();
    }
    END_TRY
}

static void EnnShowReturnValues(std::shared_ptr<SessionBufInfo> ret_loadModel) {
    ENN_INFO_PRINT("Returned SessionBufInfo: 0x%" PRIX64 "\n", ret_loadModel->model_id);
    ENN_TST_PRINT("         Buffers: %zu:\n", ret_loadModel->buffers.size());
    for (int i = 0; i < ret_loadModel->buffers.size(); i++) {
        auto &&buf = ret_loadModel->buffers[i];
        ENN_TST_PRINT(
            "    [%d] region_idx(%d), dir(%d), buf_idx(%d) size(%d), offset(%d), shape nwhc(%d %d %d %d), buffertype(%d), "
            "name(%s)\n",
            i, buf.region_idx, buf.dir, buf.buf_index, buf.size, buf.offset, buf.shape.n, buf.shape.w, buf.shape.h,
            buf.shape.c, buf.buffer_type, buf.name.c_str());
    }
    ENN_TST_PRINT("         Regions: %zu:\n", ret_loadModel->regions.size());
    for (int i = 0; i < ret_loadModel->regions.size(); i++) {
        auto &&reg = ret_loadModel->regions[i];
        ENN_TST_PRINT("    [%d] attr(%d), req_size(%d), name(%s)\n", i, reg.attr, reg.req_size, reg.name.c_str());
    }
}

static EnnReturn EnnOpenModelCore(enn::util::BufferReader::UPtrType & model, EnnModelId *model_id) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");
    CHECK_AND_RETURN_ERR(model_id == nullptr, ENN_RET_INVAL, "model_id ptr is null\n");

    uint32_t model_type = 0;
    auto pref_stream_vector = enn_context.get_preference_generator()->export_preference_to_vector();
    enn_context.get_preference_generator()->show();
    auto&& mem_list = EnnOpenFile(model, &model_type);

    CHECK_AND_RETURN_ERR(mem_list.size() == 0, ENN_RET_MEM_ERR, "Memory Allocation Err\n");

    // send it to load_model via medium interface
    auto session_buf_info = std::make_shared<SessionBufInfo>();
    auto ret = enn_context.GetMediumInterface()->open_model(mem_list, model_type, pref_stream_vector, session_buf_info);

    // NOTE(hoon98.choi) return err means explicit error in open, model_id 0 means logical problem in opened model
    if (session_buf_info->model_id == 0 || ret != ENN_RET_SUCCESS) {
        for (auto& ele : mem_list) {
            enn_context.ccMemoryManager->DeleteMemory(ele);
        }
        return ENN_RET_FAILED;
    }

    // show return values
    EnnShowReturnValues(session_buf_info);

    enn_context.ccModelContainer->SetSessionData(session_buf_info, mem_list[0]);  // ToDo(Hoon|empire.jung) : change to vector
    enn_context.ccModelContainer->ShowSessionData();  // optional

    *model_id = session_buf_info->model_id;

#ifdef ENN_MEDIUM_IF_HIDL
    // When service HIDL is used, this would be enabled so that overhead from HIDL can be measured.
    START_PROFILER(*model_id);
#endif
    ENN_INFO_PRINT_FORCE("EnnOpenModel Success(0x%" PRIX64 ")\n", *model_id);

    return ENN_RET_SUCCESS;
}

EnnReturn EnnOpenModel(const char *model_file, EnnModelId *model_id) {
    // generate buffer_reader object = model
    enn::util::BufferReader::UPtrType model = std::make_unique<enn::util::FileBufferReader>(std::string(model_file));
    CHECK_AND_RETURN_ERR(!model->is_valid(), ENN_RET_INVAL, "Model loading failed(file invalid: %s)\n", model_file);
    return EnnOpenModelCore(model, model_id);
}

EnnReturn EnnOpenModelFromMemory(const char *va, const uint32_t size, EnnModelId *model_id) {
    // generate buffer_reader object = model
    enn::util::BufferReader::UPtrType model = std::make_unique<enn::util::MemoryBufferReader>(va, size);
    CHECK_AND_RETURN_ERR(!model->is_valid(), ENN_RET_INVAL, "Model loading failed(va is nullptr)\n");
    return EnnOpenModelCore(model, model_id);
}

EnnReturn EnnCloseModel(const EnnModelId model_id) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");
    CHECK_AND_RETURN_ERR(model_id == 0, ENN_RET_FAILED, "Invalid Model ID(0).\n");

#ifdef ENN_MEDIUM_IF_HIDL
    // When service HIDL is used, this would be enabled so that overhead from HIDL can be measured.
    FINISH_PROFILER(model_id);
#endif

    // handle any async execution if client didn't wait
    enn_context.flushAsyncFutureFor(model_id);

    auto ret = enn_context.GetMediumInterface()->close_model(model_id);
    CHECK_AND_RETURN_ERR(ret, ret, "Close Model Failed\n");

    auto loaded_buf = enn_context.ccModelContainer->GetModelLoadedBuf(model_id);
    if (loaded_buf)
        enn_context.ccMemoryManager->DeleteMemory(loaded_buf);

    // hara:TODO: get ext_buffers and remove all
    auto &inf_set = enn_context.ccModelContainer->GetInferenceSet(model_id);
    if (inf_set != nullptr) {
        auto max_session_id = inf_set->n_inference;
        for (int i = 0; i < max_session_id; i++) {
            auto exist_ext_buf_lists = enn_context.ccModelContainer->GetAutoAllocatedExtBuffersFromSession(model_id, i);
            for (auto &exist_buf : exist_ext_buf_lists)
                enn_context.ccMemoryManager->DeleteMemory(exist_buf);
        }
    }
    if (enn_context.ccModelContainer->ClearModelData(model_id))
        ENN_WARN_PRINT("model_id(0x%" PRIX64 ") closeModel Failed in model container\n", model_id);
    else
        ENN_INFO_PRINT_FORCE("EnnCloseModel Success(0x%" PRIX64 ")\n", model_id);

    return ENN_RET_SUCCESS;
}

EnnReturn EnnExecuteModelWithSessionId(const EnnModelId model_id, const int session_id) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");

    ENN_INFO_PRINT_FORCE("ExecuteModel: 0x%" PRIX64 "\n", model_id);

    EnnExecuteModelId exec_model_id;
    if (enn_context.ccModelContainer->GetExecuteModelId(model_id, session_id, &exec_model_id)) {
        ENN_WARN_PRINT("Model ID(0x%" PRIX64 "): Memory is not committed. Execute commit now\n", model_id);
        CHECK_AND_RETURN_ERR(EnnBufferCommitWithSessionId(model_id, session_id), ENN_RET_FAILED, "Commit failed\n");
    }
    ENN_INFO_PRINT("Execute a model with exec_model_id(0x%jX)\n", exec_model_id);
    auto ret_mi = enn_context.GetMediumInterface()->execute_model({exec_model_id});

    if (ret_mi == ENN_RET_SUCCESS && enn_context.ShouldDumpSessionMemory()) {
        ENN_DBG_PRINT(" ## Start to dump after Execution..\n");
        enn_context.ccModelContainer->DumpSessionToFile(model_id, "dump", session_id);
    }

    return ret_mi;
}

EnnReturn EnnExecuteModel(const EnnModelId model_id) {
#ifdef ENN_MEDIUM_IF_HIDL
    // When service HIDL is used, this would be enabled so that overhead from HIDL can be measured.
    PROFILE_SCOPE("API_Execution", model_id);
#endif
    return EnnExecuteModelWithSessionId(model_id, 0);
}

EnnReturn EnnExecuteModelWithSessionIdAsync(const EnnModelId model_id, const int session_id) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_INVAL, "Context is not initialized\n");

    EnnExecuteModelId exec_model_id;
    CHECK_AND_RETURN_ERR(enn_context.ccModelContainer->GetExecuteModelId(model_id, session_id, &exec_model_id),
                         ENN_RET_INVAL, "Model ID(0x%" PRIX64 "): Memory is not committed. Execute commit now\n", model_id);

    CHECK_AND_RETURN_ERR(enn_context.hasAsyncFuture(exec_model_id), ENN_RET_INVAL,
                         "Execute Model ID(0x%" PRIX64 "): already exists.\n", exec_model_id);
#if 0
    // It should start immediately if std::launch::async is set,
    // but it doesn't conform to the C++11 standard.
    auto future = std::async(std::launch::async,
                            &EnnExecuteModelWithSessionId, model_id, session_id);
#else
    std::promise<EnnReturn> promise;
    auto future = promise.get_future();
    std::thread thread(
        [model_id, session_id](std::promise<EnnReturn> &&promise) {
            promise.set_value(EnnExecuteModelWithSessionId(model_id, session_id));
        },
        std::move(promise));
    thread.detach();
#endif
    ENN_DBG_PRINT("EnnExecuteModelId 0x%" PRIX64 "\n", exec_model_id);

    CHECK_AND_RETURN_ERR(!enn_context.putAysncFuture(exec_model_id, std::move(future)), ENN_RET_FAILED,
                         "Execute Model ID(0x%" PRIX64 "): Can't save future.\n", exec_model_id);

    return ENN_RET_SUCCESS;
}

EnnReturn EnnExecuteModelAsync(const EnnModelId model_id) {
    return EnnExecuteModelWithSessionIdAsync(model_id, 0);
}

EnnReturn EnnExecuteModelWithSessionIdWait(const EnnModelId model_id, const int session_id) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");

    EnnExecuteModelId exec_model_id;
    CHECK_AND_RETURN_ERR(enn_context.ccModelContainer->GetExecuteModelId(model_id, session_id, &exec_model_id),
                         ENN_RET_FAILED, "Model ID(0x%" PRIX64 "): Memory is not committed. Execute commit now\n", model_id);

    auto future = enn_context.popAsyncFuture(exec_model_id);
    auto ret = future.get();
    ENN_DBG_PRINT("EnnExecuteModelId 0x%" PRIX64 ", R %d\n", exec_model_id, ret);

    return ret;
}

EnnReturn EnnExecuteModelWait(const EnnModelId model_id) {
    return EnnExecuteModelWithSessionIdWait(model_id, 0);
}

EnnReturn EnnDeinitialize(void) {
    auto ret = enn_context.deinit();
    CHECK_AND_RETURN_ERR(ret, ret, "Error from context deinitialize\n");
    ENN_INFO_PRINT_FORCE("EnnDeinitialize\n");
    return ENN_RET_SUCCESS;
}

EnnReturn EnnGenerateBufferSpace(const EnnModelId model_id, const int n_set = 1) {
    ENN_UNUSED(n_set);
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");

    return enn_context.ccModelContainer->GenerateInferenceData(model_id, n_set);  // generate buf pool for single exec.
}

template <typename T> static EnnReturn EnnAssignBufferInfo(EnnBufferInfo *out_buf_info, const T &buf_ele) {
    // TODO(hoon.choi): Use copy constructor or assignment operator
    out_buf_info->n = buf_ele.shape.n;
    out_buf_info->channel = buf_ele.shape.c;
    out_buf_info->height = buf_ele.shape.h;
    out_buf_info->width = buf_ele.shape.w;
    out_buf_info->label = buf_ele.name.c_str();
    out_buf_info->size = buf_ele.size;
    return ENN_RET_SUCCESS;
}

static EnnReturn EnnGetBufferInfo(const EnnModelId model_id, const char *label, const enn_buf_dir_e direction,
                                  const uint32_t index, EnnBufferInfo *out_buf_info) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");
    CHECK_AND_RETURN_ERR(out_buf_info == nullptr, ENN_RET_FAILED, "out buffer is nullptr\n");
    auto &session_buf = enn_context.ccModelContainer->GetSession(model_id);
    CHECK_AND_RETURN_ERR(session_buf == nullptr, ENN_RET_INVAL, "Doesn't have any session data of ModelID(0x%" PRIX64 ")\n",
                         model_id);

    for (auto &buf_ele : session_buf->buffers) {
        if ((label == nullptr && (static_cast<enn_buf_dir_e>(buf_ele.dir) == direction && buf_ele.buf_index == index)) ||
            (label != nullptr && (std::string(label) == std::string(buf_ele.name.c_str())))) {
            return EnnAssignBufferInfo(out_buf_info, buf_ele);
        }
    }
    return ENN_RET_FAILED;
}

EnnReturn EnnGetBufferInfoByIndex(const EnnModelId model_id, const enn_buf_dir_e direction, const uint32_t index,
                                  EnnBufferInfo *out_buf_info) {
    return EnnGetBufferInfo(model_id, nullptr, direction, index, out_buf_info);
}

EnnReturn EnnGetBufferInfoByLabel(const EnnModelId model_id, const char *label, EnnBufferInfo *out_buf_info) {
    return EnnGetBufferInfo(model_id, label, ENN_DIR_SIZE, -1, out_buf_info);
}

EnnReturn EnnGetBuffersInfo(const EnnModelId model_id, NumberOfBuffersInfo *buffers_info) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");
    auto ret = enn_context.ccModelContainer->GetBufferInfo(model_id, &(buffers_info->n_in_buf), &(buffers_info->n_out_buf));
    CHECK_AND_RETURN_ERR(ret, ret, "Get buffer error(mId: 0x%" PRIX64 ")\n", model_id);
    ENN_DBG_PRINT("Model_ID(0x%" PRIX64 ") has %d Input, %d Output\n", model_id, buffers_info->n_in_buf,
                  buffers_info->n_out_buf);
    return ENN_RET_SUCCESS;
}

EnnReturn EnnBufferCommitWithSessionId(const EnnModelId model_id, const int session_id) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");

    /* check if session_id is already set */
    EnnExecuteModelId ret_execution_id;
    auto ret_exec = enn_context.ccModelContainer->GetExecuteModelId(model_id, session_id, &ret_execution_id);
    if (ret_exec)
        return ENN_RET_SUCCESS;
    CHECK_AND_RETURN_WARN(ret_execution_id != EXEC_MODEL_NOT_ASSIGNED, ENN_RET_INVAL,
                          "Session_id(%d, 0x%" PRIX64 ") is already commited!\n", session_id, ret_execution_id);

    auto exist_ext_buf_lists = enn_context.ccModelContainer->GetAutoAllocatedExtBuffersFromSession(model_id, session_id);
    if (exist_ext_buf_lists.size() > 0)
        ENN_WARN_PRINT("%zu Ext buffers are already set. Framework will clean all ext buffers of [mid: 0x%" PRIX64
                       ", sid: 0x%d]. please check!\n",
                       exist_ext_buf_lists.size(), model_id, session_id);

    // clear buf  // not have to.
    for (auto &exist_buf : exist_ext_buf_lists)
        enn_context.ccMemoryManager->DeleteMemory(exist_buf);

    auto ext_region_idx_set = enn_context.ccModelContainer->GetExtRegionIndexes(model_id, session_id);
    for (auto &idx : ext_region_idx_set) {
        int req_size = enn_context.ccModelContainer->GetRegionSizes(model_id, idx);
        ENN_MEM_PRINT("Idx(%d)'s memory: %d will be allocated\n", idx, req_size);
        auto buf = enn_context.ccMemoryManager->CreateMemory(req_size, enn::EnnMmType::kEnnMmTypeIon, ION_FLAG_CACHED);
        enn_context.ccModelContainer->SetAutoAllocatedExtBuffersToSession(model_id, session_id, buf);
        enn_context.ccModelContainer->SetInferenceData(model_id, session_id, idx, buf.get());
    }

    auto ret = enn_context.ccModelContainer->VerifyInferenceData(model_id, session_id);

    if (ret != ENN_RET_SUCCESS) {
        auto exist_ext_buf_lists = enn_context.ccModelContainer->GetAutoAllocatedExtBuffersFromSession(model_id, session_id);
        for (auto &exist_buf : exist_ext_buf_lists)
            enn_context.ccMemoryManager->DeleteMemory(exist_buf);
        ENN_ERR_PRINT("Verification error: ext-buffer set[%d]\n", session_id);
        return ENN_RET_FAILED;
    }

    CHECK_AND_RETURN_ERR(ret, ret, "Verification error: ext-buffer set[%d]\n", session_id);
    auto inf_set = enn_context.ccModelContainer->GetInferenceSet(model_id);
    CHECK_AND_RETURN_ERR(inf_set == nullptr, ENN_RET_FAILED, "There's no inference data\n");

    ret_execution_id = enn_context.GetMediumInterface()->commit_execution_data(model_id, inf_set->inference_set[session_id]);
    if (ret_execution_id != EXEC_MODEL_NOT_ASSIGNED)  // model id 0 means error
        enn_context.ccModelContainer->SetExecuteModelId(model_id, session_id, ret_execution_id);

    ENN_INFO_PRINT_FORCE("EnnBufferCommit: Returns ExecutionModelId: 0x%" PRIX64 "\n", ret_execution_id);

    return ENN_RET_SUCCESS;
}

EnnReturn EnnBufferCommit(const EnnModelId model_id) {
    return EnnBufferCommitWithSessionId(model_id, 0);
}

template <typename T> static T *enn_allocate_buffers(int n) {
    return new T[n]();
}

template <typename T> static void enn_delete_buffers(T *arr) {
    delete[] arr;
    arr = nullptr;
}

/* NOTE(hoon98.choi) : This function can be extended to "EnnAllocateAllBuffersMultiple" */
EnnReturn EnnAllocateAllBuffersWithSessionId(const EnnModelId model_id, EnnBufferPtr **out_buffers,
                                             NumberOfBuffersInfo *out_buffers_info, const int session_id,
                                             const bool do_commit) {
    /* TODO(hoon98.choi, TBD): Error handling. use exception? */
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");

    // NOTE(hoon98.choi): duplicate code with EnnSetBufferByIndexWithSessionId()
    auto n_inf = enn_context.ccModelContainer->GetNumInferenceData(model_id);
    CHECK_AND_RETURN_ERR((n_inf == 0 && session_id != 0), ENN_RET_FAILED, "Inference Data space is not enough(%d)", n_inf);
    CHECK_AND_RETURN_ERR((n_inf < session_id), ENN_RET_FAILED, "Inference Data space is not enough(%d, commit: %d)", n_inf,
                         session_id);

    NumberOfBuffersInfo buf_info;
    CHECK_AND_RETURN_ERR(EnnGetBuffersInfo(model_id, &buf_info), ENN_RET_FAILED, "Error to get buf info\n");

    uint32_t n_allocated = 0;

    auto bufs = enn_allocate_buffers<EnnBufferPtr>(buf_info.n_in_buf + buf_info.n_out_buf);
    CHECK_AND_RETURN_ERR(bufs == nullptr, ENN_RET_FAILED, "Memory allocation Error: i(%d), o(%d)\n", buf_info.n_in_buf,
                         buf_info.n_out_buf);

    EnnBufferInfo tmp_buf_info;
    for (int idx = 0; idx < buf_info.n_in_buf; idx++) {
        EnnGetBufferInfoByIndex(model_id, ENN_DIR_IN, idx, &tmp_buf_info);
        EnnCreateBufferCache(tmp_buf_info.size, &(bufs[n_allocated++]));
    }
    for (int idx = 0; idx < buf_info.n_out_buf; idx++) {
        EnnGetBufferInfoByIndex(model_id, ENN_DIR_OUT, idx, &tmp_buf_info);
        EnnCreateBufferCache(tmp_buf_info.size, &(bufs[n_allocated++]));
    }

    // TODO(hoon98.choi): Error handling for return buffers info
    if (EnnSetBuffersWithSessionId(model_id, bufs, (buf_info.n_in_buf + buf_info.n_out_buf), session_id)) {
        ENN_WARN_PRINT("Set Buffer Failed: Please check the graph\n");
        EnnReleaseBuffers(bufs, n_allocated);
        out_buffers_info->n_in_buf = 0;
        out_buffers_info->n_out_buf = 0;
        return ENN_RET_FAILED;
    }

    if (do_commit) {
        if (EnnBufferCommitWithSessionId(model_id, session_id)) {  // return zero means error (returns exe-model id)
            ENN_WARN_PRINT("Commit Buffer Failed: Pleace check your sequence\n");
            EnnReleaseBuffers(bufs, n_allocated);
            out_buffers_info->n_in_buf = 0;
            out_buffers_info->n_out_buf = 0;
            return ENN_RET_FAILED;
        }
    }

    // NOTE: ext buffers are allocated at commit
    *out_buffers = bufs;

    out_buffers_info->n_in_buf = buf_info.n_in_buf;
    out_buffers_info->n_out_buf = buf_info.n_out_buf;

    return ENN_RET_SUCCESS;
}

EnnReturn EnnAllocateAllBuffers(const EnnModelId model_id, EnnBufferPtr **out_buffers,
                                NumberOfBuffersInfo *out_buffers_info) {
    return EnnAllocateAllBuffersWithSessionId(model_id, out_buffers, out_buffers_info, 0, true);
}

EnnReturn EnnAllocateAllBuffersWithoutCommit(const EnnModelId model_id, EnnBufferPtr **out_buffers,
                                             NumberOfBuffersInfo *out_buffers_info) {
    return EnnAllocateAllBuffersWithSessionId(model_id, out_buffers, out_buffers_info, 0, false);
}

EnnReturn EnnCreateBuffer(const uint32_t req_size, const uint32_t flag, EnnBufferPtr *out) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");
    CHECK_AND_RETURN_ERR(!out, ENN_RET_FAILED, "Output parameter(*out) is NULL\n");
    auto mem = enn_context.ccMemoryManager->CreateMemory(req_size, enn::EnnMmType::kEnnMmTypeIon, flag);
    if (mem == nullptr)
        return ENN_RET_MEM_ERR;

    *out = reinterpret_cast<EnnBufferPtr>(mem->return_ptr());
    return ENN_RET_SUCCESS;
}

EnnReturn EnnCreateBufferCache(const uint32_t req_size, EnnBufferPtr *out) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");
    CHECK_AND_RETURN_ERR(!out, ENN_RET_FAILED, "Output parameter(*out) is NULL\n");

    auto mem = enn_context.ccMemoryManager->CreateMemory(req_size, enn::EnnMmType::kEnnMmTypeIon, ION_FLAG_CACHED);
    if (mem == nullptr)
        return ENN_RET_MEM_ERR;

    *out = reinterpret_cast<EnnBufferPtr>(mem->return_ptr());
    return ENN_RET_SUCCESS;
}

EnnReturn EnnCreateBufferFromFd(const uint32_t fd, const uint32_t size, EnnBufferPtr *out) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");
    CHECK_AND_RETURN_ERR(!out, ENN_RET_FAILED, "Output parameter(*out) is NULL\n");
    auto mem = enn_context.ccMemoryManager->CreateMemoryFromFd(fd, size);
    if (mem == nullptr)
        return ENN_RET_MEM_ERR;

    *out = reinterpret_cast<EnnBufferPtr>(mem->return_ptr());
    return ENN_RET_SUCCESS;
}

EnnReturn EnnCreateBufferFromFdWithOffset(const uint32_t fd, const uint32_t size, const uint32_t offset, EnnBufferPtr *out) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");
    CHECK_AND_RETURN_ERR(!out, ENN_RET_FAILED, "Output parameter(*out) is NULL\n");
    auto mem = enn_context.ccMemoryManager->CreateMemoryFromFdWithOffset(fd, size, offset);
    if (mem == nullptr)
        return ENN_RET_MEM_ERR;

    *out = reinterpret_cast<EnnBufferPtr>(mem->return_ptr());
    return ENN_RET_SUCCESS;
}

EnnReturn EnnReleaseBuffer(EnnBufferPtr buffer) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");
    return enn_context.ccMemoryManager->DeleteMemory(buffer);
}

EnnReturn EnnReleaseBuffers(EnnBufferPtr *buffers, const int32_t numOfBuffers) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");
    ENN_TST_PRINT("Buffers to be released: %p, num: %d\n", buffers, numOfBuffers);
    EnnReturn ret = ENN_RET_SUCCESS, ret_tmp;
    for (int i = 0; i < numOfBuffers; i++) {
        ret_tmp = enn_context.ccMemoryManager->DeleteMemory(buffers[i]);
        if (ret_tmp)
            ret = ENN_RET_FAILED;
    }
    enn_delete_buffers(buffers);
    return ret;
}

EnnReturn EnnSetBufferByIndexWithSessionId(const EnnModelId model_id, const enn_buf_dir_e direction, const uint32_t index,
                                           EnnBufferPtr buf, const int session_id) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");
    CHECK_AND_RETURN_WARN(direction == ENN_DIR_EXT, ENN_RET_FAILED, "Ext buffer cann be set with this function\n");

    auto n_inf = enn_context.ccModelContainer->GetNumInferenceData(model_id);

    // n_inf == 0 && session_id == 0? -> create one.
    // n_inf == 0 && session_id != 0? -> return error
    // n_inf < session_id? -> return error
    if (n_inf == 0 && session_id == 0) {
        enn_context.ccModelContainer->GenerateInferenceData(model_id);
    } else {
        CHECK_AND_RETURN_ERR((n_inf == 0 && session_id != 0), ENN_RET_FAILED, "Inference Data space is not enough(%d)",
                             n_inf);
        CHECK_AND_RETURN_ERR((n_inf < session_id), ENN_RET_FAILED, "Inference Data space is not enough(%d, commit: %d)",
                             n_inf, session_id);
    }
    auto sendBuf = reinterpret_cast<enn::EnnBufferCore *>(buf);
    return enn_context.ccModelContainer->SetInferenceData(model_id, session_id,
                                                          std::tuple<uint32_t, int32_t>(direction, index), sendBuf);
}

EnnReturn EnnSetBufferByIndex(const EnnModelId model_id, enn_buf_dir_e direction, const uint32_t index, EnnBufferPtr buf) {
    return EnnSetBufferByIndexWithSessionId(model_id, direction, index, buf, 0);
}

EnnReturn EnnSetBufferByLabelWithSessionId(const EnnModelId model_id, const char *label, EnnBufferPtr buf,
                                           const int session_id) {
    // all named buffer can be set, even if it is ext buffer
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");

    auto n_inf = enn_context.ccModelContainer->GetNumInferenceData(model_id);
    if (n_inf == 0 && session_id == 0) {
        enn_context.ccModelContainer->GenerateInferenceData(model_id);
    } else {
        CHECK_AND_RETURN_ERR((n_inf == 0 && session_id != 0), ENN_RET_FAILED, "Inference Data space is not enough(%d)",
                             n_inf);
        CHECK_AND_RETURN_ERR((n_inf < session_id), ENN_RET_FAILED, "Inference Data space is not enough(%d, commit: %d)",
                             n_inf, session_id);
    }

    auto sendBuf = reinterpret_cast<enn::EnnBufferCore *>(buf);
    return enn_context.ccModelContainer->SetInferenceData(model_id, session_id, std::string(label), sendBuf);
}

EnnReturn EnnSetBufferByLabel(const EnnModelId model_id, const char *label, EnnBufferPtr buf) {
    return EnnSetBufferByLabelWithSessionId(model_id, label, buf, 0);
}

EnnReturn EnnSetBuffersWithSessionId(const EnnModelId model_id, EnnBufferPtr *bufs, const int32_t sum_io,
                                     const int session_id) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");

    NumberOfBuffersInfo buf_info;
    CHECK_AND_RETURN_ERR(EnnGetBuffersInfo(model_id, &buf_info), ENN_RET_FAILED, "Error to get buf info\n");

    uint32_t n_buf_index = 0;

    CHECK_AND_RETURN_ERR((sum_io != buf_info.n_in_buf + buf_info.n_out_buf), ENN_RET_FAILED,
                         "Parameter is different! (%d + %d != %d)\n", buf_info.n_in_buf, buf_info.n_out_buf, sum_io);
    EnnReturn ret_set;

    for (int idx = 0; idx < buf_info.n_in_buf; idx++) {
        ret_set = EnnSetBufferByIndexWithSessionId(model_id, ENN_DIR_IN, idx, bufs[n_buf_index++], session_id);
        if (ret_set)
            return ret_set;
    }

    for (int idx = 0; idx < buf_info.n_out_buf; idx++) {
        ret_set = EnnSetBufferByIndexWithSessionId(model_id, ENN_DIR_OUT, idx, bufs[n_buf_index++], session_id);
        if (ret_set)
            return ret_set;
    }

    // NOTE: ext buffers are set at commit()
    return ENN_RET_SUCCESS;
}

EnnReturn EnnSetBuffers(const EnnModelId model_id, EnnBufferPtr *bufs, const int32_t sum_io) {
    return EnnSetBuffersWithSessionId(model_id, bufs, sum_io, 0);
}


/* setter */
EnnReturn EnnResetPreferenceAsDefault() {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");
    enn_context.get_preference_generator()->reset_as_default();
    return ENN_RET_SUCCESS;
}

EnnReturn EnnSetPreferencePresetId(const uint32_t val) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");
    return enn_context.get_preference_generator()->set_preset_id(val);
}

EnnReturn EnnSetPreferencePerfMode(const uint32_t val) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");
    return enn_context.get_preference_generator()->set_pref_mode(val);
}

EnnReturn EnnSetPreferenceTargetLatency(const uint32_t val) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");
    return enn_context.get_preference_generator()->set_target_latency(val);
}

EnnReturn EnnSetPreferenceTileNum(const uint32_t val) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");
    return enn_context.get_preference_generator()->set_tile_num(val);
}

EnnReturn EnnSetPreferenceCoreAffinity(const uint32_t val) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");
    return enn_context.get_preference_generator()->set_core_affinity(val);
}

EnnReturn EnnSetPreferencePriority(const uint32_t val) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");
    return enn_context.get_preference_generator()->set_priority(val);
}

EnnReturn EnnSetPreferenceCustom_0(const uint32_t val) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");
    return enn_context.get_preference_generator()->set_custom_0(val);
}

EnnReturn EnnSetPreferenceCustom_1(const uint32_t val) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");
    return enn_context.get_preference_generator()->set_custom_1(val);
}

/* getter */
EnnReturn EnnGetPreferenceTargetLatency(uint32_t *val_ptr) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");
    CHECK_AND_RETURN_ERR(val_ptr == nullptr, ENN_RET_INVAL, "Parameter invalid (nullptr)\n");

    *val_ptr = enn_context.get_preference_generator()->get_target_latency();
    return ENN_RET_SUCCESS;
}

EnnReturn EnnGetPreferenceTileNum(uint32_t *val_ptr) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");
    CHECK_AND_RETURN_ERR(val_ptr == nullptr, ENN_RET_INVAL, "Parameter invalid (nullptr)\n");
    *val_ptr = enn_context.get_preference_generator()->get_tile_num();
    return ENN_RET_SUCCESS;
}

EnnReturn EnnGetPreferenceCoreAffinity(uint32_t *val_ptr) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");
    CHECK_AND_RETURN_ERR(val_ptr == nullptr, ENN_RET_INVAL, "Parameter invalid (nullptr)\n");
    *val_ptr = enn_context.get_preference_generator()->get_core_affinity();
    return ENN_RET_SUCCESS;
}

EnnReturn EnnGetPreferencePriority(uint32_t *val_ptr) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");
    CHECK_AND_RETURN_ERR(val_ptr == nullptr, ENN_RET_INVAL, "Parameter invalid (nullptr)\n");
    *val_ptr = enn_context.get_preference_generator()->get_priority();
    return ENN_RET_SUCCESS;
}

EnnReturn EnnGetPreferencePresetId(uint32_t *val_ptr) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");
    CHECK_AND_RETURN_ERR(val_ptr == nullptr, ENN_RET_INVAL, "Parameter invalid (nullptr)\n");
    *val_ptr = enn_context.get_preference_generator()->get_preset_id();
    return ENN_RET_SUCCESS;
}

EnnReturn EnnGetPreferencePerfMode(uint32_t *val_ptr) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");
    CHECK_AND_RETURN_ERR(val_ptr == nullptr, ENN_RET_INVAL, "Parameter invalid (nullptr)\n");
    *val_ptr = enn_context.get_preference_generator()->get_pref_mode();
    return ENN_RET_SUCCESS;
}

EnnReturn EnnGetPreferenceCustom_0(uint32_t *val_ptr) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");
    CHECK_AND_RETURN_ERR(val_ptr == nullptr, ENN_RET_INVAL, "Parameter invalid (nullptr)\n");
    *val_ptr = enn_context.get_preference_generator()->get_custom_0();
    return ENN_RET_SUCCESS;
}

EnnReturn EnnGetPreferenceCustom_1(uint32_t *val_ptr) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");
    CHECK_AND_RETURN_ERR(val_ptr == nullptr, ENN_RET_INVAL, "Parameter invalid (nullptr)\n");
    *val_ptr = enn_context.get_preference_generator()->get_custom_1();
    return ENN_RET_SUCCESS;
}

/* Custom functions */
EnnReturn EnnDspGetSessionId(const EnnModelId model_id, int32_t *out) {
    CHECK_AND_RETURN_ERR(enn_context.get_ref_cnt() < 1, ENN_RET_FAILED, "Context is not initialized\n");
    CHECK_AND_RETURN_ERR(out == nullptr, ENN_RET_INVAL, "Parameter invalid (nullptr)\n");

    auto ret = enn_context.GetMediumInterface()->get_dsp_session_id(model_id);
    CHECK_AND_RETURN_ERR(ret < 0, ENN_RET_INVAL, "Failed to get dsp session_id\n");

    *out = ret;
    return ENN_RET_SUCCESS;
}

