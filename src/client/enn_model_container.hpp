
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
 * @file enn_model_container.hpp
 * @author Hoon Choi (hoon98.choi@)
 * @brief Manages model from loaded model
 * @version 0.1
 * @date 2021-03-10
 */

#include <memory>
#include <map>
#include <mutex>
#include <utility>
#include <tuple>
#include <string>
#include <cinttypes>
#include <tuple>
#include <vector>
#include <set>

#include <sys/mman.h>

#include "common/enn_debug.h"
#include "common/enn_utils.h"

#ifndef SRC_CLIENT_ENN_MODEL_CONTAINER_H_
#define SRC_CLIENT_ENN_MODEL_CONTAINER_H_

namespace enn {
namespace client {

template <typename T1, typename T2> constexpr auto ENN_STL_ITER_IS_IN(T1 obj, T2 ele) {
    return obj.find(ele) != obj.end();
}

template <typename idType, typename sessionType, typename execType, typename InferenceData> class EnnModelContainer {
  public:
    EnnModelContainer() {
        session_map.clear();
        exec_map.clear();
    }

    /* TODO: requires to return reference or pointer? */
    const std::shared_ptr<sessionType> GetSession(idType model_id) {
        auto iter = session_map.find(model_id);
        if (iter == session_map.end())
            return nullptr;
        else
            return iter->second;
    }

    EnnReturn SetSessionData(std::shared_ptr<sessionType> session, std::shared_ptr<EnnBufferCore> buf = nullptr) {
        std::lock_guard<std::mutex> guard(mdl_ctrl_mutex);
        auto model_id = session->model_id;
        CHECK_AND_RETURN_ERR((model_id == 0), ENN_RET_INVAL, "model ID (0x%" PRIX64 ") is incorrent\n", model_id);
        if (ENN_STL_ITER_IS_IN(session_map, model_id))
            ENN_WARN_PRINT(" Session is already set. overwrite it\n");

        session_map[model_id] = session;
        loaded_buf_map[model_id] = buf;
        ENN_INFO_PRINT("Set session with model_id(0x%" PRIX64 ")\n", model_id);

        return ENN_RET_SUCCESS;
    }

    EnnReturn SetAutoAllocatedExtBuffersToSession(idType model_id, int32_t session_id,
                                                  std::shared_ptr<EnnBufferCore> mem_obj) {
        std::lock_guard<std::mutex> guard(mdl_ctrl_mutex);
        auto &ext_buffer_map = auto_allocated_ext_buffer_lists[model_id];
        auto &ext_buffer_session_vec = ext_buffer_map[session_id];
        ext_buffer_session_vec.push_back(mem_obj);

        return ENN_RET_SUCCESS;
    }

    std::vector<std::shared_ptr<EnnBufferCore>> GetAutoAllocatedExtBuffersFromSession(idType model_id, int32_t session_id) {
        std::lock_guard<std::mutex> guard(mdl_ctrl_mutex);
        auto &ext_buffer_map = auto_allocated_ext_buffer_lists[model_id];
        return ext_buffer_map[session_id];
    }

    void ShowAutoAllocatedExtBuffers() {
        ENN_TST_PRINT("Show Ext Buffers...\n");
        for (auto &buf_map : auto_allocated_ext_buffer_lists) {
            ENN_TST_PRINT("model ID (0x%" PRIX64 ")... \n", buf_map.first);
            for (auto &buf_session_map : buf_map.second) {
                ENN_TST_PRINT("  session ID (0x%" PRIX64 ")... \n", buf_session_map.first);
                for (auto &buffs : buf_session_map.second) {
                    ENN_TST_PRINT("    buffer: size(%d), offset(%d), va(%p)\n", buffs->size, buffs->offset, buffs->va);
                }
            }
        }
    }

    void ShowSessionData() {
        std::lock_guard<std::mutex> guard(mdl_ctrl_mutex);
        for (auto &session_ele : session_map) {
            auto &m_id = session_ele.first;
            auto &session_buf = session_ele.second;
            ENN_DBG_PRINT("# Session.Model_ID: 0x%" PRIX64 " / 0x%" PRIX64 "(buf: %zu, reg: %zu):::: \n", m_id,
                          session_buf->model_id, session_buf->buffers.size(), session_buf->regions.size());
            for (auto &buf_ele : session_buf->buffers) {
                ENN_DBG_PRINT("# [Buffer] Region_idx(%d), dir(%d), buf_index(%d), size(%d), offset(%d),\n",
                              buf_ele.region_idx, buf_ele.dir, buf_ele.buf_index, buf_ele.size, buf_ele.offset);
                ENN_DBG_PRINT("           shape nwhc(%d x %d x %d x %d), buffer_type: %d, name: %s\n", buf_ele.shape.n,
                              buf_ele.shape.w, buf_ele.shape.h, buf_ele.shape.c, buf_ele.buffer_type, buf_ele.name.c_str());
            }
            ENN_DBG_PRINT("\n");
            for (auto &reg_ele : session_buf->regions) {
                ENN_DBG_PRINT("# [Region] attr(%d), req_size(%d), name(%s)\n", reg_ele.attr, reg_ele.req_size,
                              reg_ele.name.c_str());
            }
        }
    }

    // This function returns 'empty' ext buffers.
    std::set<int> GetExtRegionIndexes(idType model_id, int32_t session_id) {
        std::lock_guard<std::mutex> guard(mdl_ctrl_mutex);
        std::set<int> return_index_set;
        auto session_buf = GetSession(model_id);
        auto &inf_data = exec_map[model_id]->inference_set[session_id];

        CHECK_AND_RETURN_ERR(session_buf == nullptr, {}, "Doesn't have any session data of ModelID(0x%" PRIX64 ")\n",
                             model_id);
        for (auto &buf_ele : session_buf->buffers) {
            if (buf_ele.dir == DirType::ENN_BUF_DIR_EXT) {
                ENN_DBG_COUT << "ridx: " << buf_ele.region_idx << ", Ext [" << buf_ele.buf_index
                             << "], size: " << inf_data.inference_data[buf_ele.region_idx].size << std::endl;
                if (inf_data.inference_data[buf_ele.region_idx].size != 0) {
                    ENN_DBG_PRINT("model_id(0x%" PRIX64 "), ext(%d), region idx(%d) is already set by user. skip\n",
                                  model_id, buf_ele.buf_index, buf_ele.region_idx);
                    continue;
                } else {
                    ENN_DBG_PRINT("model_id(0x%" PRIX64 ") has ext(%d) --> region idx(%d)\n", model_id, buf_ele.buf_index,
                                  buf_ele.region_idx);
                }
                return_index_set.insert(buf_ele.region_idx);
            }
        }
        return return_index_set;
    }

    uint32_t GetRegionSizes(idType model_id, int region_idx) {
        std::lock_guard<std::mutex> guard(mdl_ctrl_mutex);
        auto session_buf = GetSession(model_id);
        CHECK_AND_RETURN_ERR(session_buf == nullptr, 0, "Doesn't have any session data of ModelID(0x%" PRIX64 ")\n",
                             model_id);
        CHECK_AND_RETURN_ERR(session_buf->regions.size() <= region_idx, 0, "Region_idx(%d) is invalid (max:%d)\n",
                             region_idx, (int)session_buf->regions.size() - 1);
        return session_buf->regions[region_idx].req_size;  // size = 0 means an error
    }

    EnnReturn GenerateInferenceData(idType model_id, int32_t num = 1) {
        std::lock_guard<std::mutex> guard(mdl_ctrl_mutex);
        auto session_buf = GetSession(model_id);
        CHECK_AND_RETURN_ERR(session_buf == nullptr, ENN_RET_INVAL,
                             "Doesn't have any session data of ModelID(0x%" PRIX64 ")\n", model_id);
        CHECK_AND_RETURN_ERR(num <= 0, ENN_RET_INVAL, "Number of inference data should be over zero\n");

        if (!ENN_STL_ITER_IS_IN(exec_map, model_id))
            ENN_INFO_PRINT("No inference data. generate it.\n");
        else
            ENN_INFO_PRINT("There's inference data on Model ID(0x%" PRIX64 "), clear and overwrite it.\n", model_id);

        GenerateInferenceSetAtExecMap(model_id, num, session_buf);

        ENN_INFO_PRINT("Generate Inference Data: ModelID(0x%" PRIX64 "), num(%d)\n", model_id, num);

        return ENN_RET_SUCCESS;
    }

    uint32_t GetNumInferenceData(idType model_id) {
        std::lock_guard<std::mutex> guard(mdl_ctrl_mutex);
        auto session_buf = GetSession(model_id);
        CHECK_AND_RETURN_ERR(session_buf == nullptr, ENN_RET_INVAL,
                             "Doesn't have any session data of ModelID(0x%" PRIX64 ")\n", model_id);
        if (!ENN_STL_ITER_IS_IN(exec_map, model_id))
            return 0;
        else
            return exec_map[model_id]->n_inference;
    }

    /**
     * @brief Set the Inference Data object: Core function to process super-set
     *
     * @param model_id    model id
     * @param session_id  session id
     * @param buffer_name if buffer_name == "", try to find buffer with {dir, index}
     * @param dir         IN or OUT
     * @param index       index number
     * @param mem_object  memory object to insert
     * @return EnnReturn  ENN_RET_SUCCESS(0), otheres are failed
     */
    template <typename KeyType>
    EnnReturn SetInferenceData(idType model_id, int32_t session_id, KeyType key, EnnBufferCore *mem_object) {
        CHECK_AND_RETURN_ERR(mem_object == nullptr, ENN_RET_INVAL, "Invalid memory object\n");
        auto ret = CheckReadyToSetExec(model_id, session_id);
        CHECK_AND_RETURN_ERR(ret, ret, "Failed\n");

        // 1. check model_id have session
        auto session_buf = GetSession(model_id);
        auto inf_data = &(exec_map[model_id]->inference_set[session_id]);

        // *. If data is already committed, cannot set inference before release commit
        CHECK_AND_RETURN_ERR((inf_data->is_commit == true), ENN_RET_FAILED,
                             "Inference Data[%d] is already commited. cannot support yet.", session_id);

        // 2. get buffer: take buffer in inference data
        //                For indexing, user can put label(string), or {direction, index}
        // TODO(hoon98.choi): make function to get_buffer()
        Buffer *select_buffer = nullptr;
        if constexpr (std::is_same<KeyType, std::string>::value) {
            for (auto &buf : session_buf->buffers) {
                if (buf.name == key) {
                    select_buffer = &buf;
                    break;
                }
            }
        } else if constexpr (std::is_same<KeyType, std::tuple<uint32_t, int32_t>>::value) {
            for (auto &buf : session_buf->buffers) {
                if (static_cast<uint32_t>(buf.dir) == std::get<0>(key) && buf.buf_index == std::get<1>(key)) {
                    select_buffer = &buf;
                    break;
                }
            }
        } else {
            ENN_WARN_PRINT("Cannot process with the parameter type\n");
        }

        // 3. check found buffer can insert to session list or not.
        CHECK_AND_RETURN_ERR(select_buffer == nullptr, ENN_RET_INVAL,
                             "Couldn't find buffer, or key type is not supported\n");
        CHECK_AND_RETURN_ERR((select_buffer->offset != 0), ENN_RET_INVAL,
                             "Currently partial update is not supported: buf offset(%d)\n", select_buffer->offset);
        CHECK_AND_RETURN_ERR((select_buffer->size != session_buf->regions[select_buffer->region_idx].req_size),
                             ENN_RET_INVAL, "Currently partial update is not supported: buf size(%d), inf_buf size(%d) \n",
                             select_buffer->size, session_buf->regions[select_buffer->region_idx].req_size);
        CHECK_AND_RETURN_ERR(select_buffer->size != mem_object->size, ENN_RET_INVAL,
                             "buffer size(%s[%d], input[%d]) is different\n", select_buffer->name.c_str(),
                             select_buffer->size, mem_object->size);

        // 4. Set inference data
        ret = setInferenceData(select_buffer, inf_data, mem_object);
        inf_data->exec_model_id = EXEC_MODEL_ID_NOT_DEFINED;

        return ret;
    }

    /**
     * @brief Please use internally
     *
     * @param model_id    model id
     * @param session_id  session id
     * @param buffer_name if buffer_name == "", try to find buffer with {dir, index}
     * @param dir         IN or OUT
     * @param index       index number
     * @param mem_object  memory object to insert
     * @return EnnReturn  ENN_RET_SUCCESS(0), otheres are failed
     */
    EnnReturn SetInferenceData(idType model_id, int32_t session_id, int region_idx, EnnBufferCore *mem_object) {
        CHECK_AND_RETURN_ERR(mem_object == nullptr, ENN_RET_INVAL, "Invalid memory object\n");
        auto ret = CheckReadyToSetExec(model_id, session_id);
        CHECK_AND_RETURN_ERR(ret, ret, "Failed\n");

        // 1. check model_id have session
        auto session_buf = GetSession(model_id);
        auto inf_data = &(exec_map[model_id]->inference_set[session_id]);

        // 2. Set inference data
        ret = setInferenceData(inf_data, region_idx, mem_object);
        inf_data->exec_model_id = EXEC_MODEL_ID_NOT_DEFINED;

        return ret;
    }

    EnnReturn VerifyInferenceData(idType model_id, int32_t session_id) {
        std::lock_guard<std::mutex> guard(mdl_ctrl_mutex);

        auto ret = CheckReadyToSetExec(model_id, session_id);
        CHECK_AND_RETURN_ERR(ret, ret, "Failed\n");

        // 1. check model_id have session
        auto regions = GetSession(model_id)->regions;
        const auto &inference_data = exec_map[model_id]->inference_set[session_id].inference_data;

        // 2. Check region size and inference data vector size
        CHECK_AND_RETURN_ERR(regions.size() != inference_data.size(), ENN_RET_INVAL,
                             "Inference Size[%d] incorrect(%zu %zu)\n", session_id, regions.size(), inference_data.size());

        // 3. check size in each inference_data size
        for (uint32_t i = 0; i < regions.size(); i++) {
            if (regions[i].req_size != inference_data[i].size) {
                ENN_WARN_PRINT("Region[%d] size is different (region %u, inf data %u)\n", i, regions[i].req_size,
                               inference_data[i].size);
                return ENN_RET_INVAL;
            }
        }

        ENN_INFO_PRINT("Verify Success: model ID(0x%" PRIX64 "), session ID(%d)\n", model_id, session_id);
        return ENN_RET_SUCCESS;
    }

    // SetCommitFlag -> SetExecuteModelId
    EnnReturn SetExecuteModelId(idType model_id, idType session_id, EnnExecuteModelId exec_id) {
        std::lock_guard<std::mutex> guard(mdl_ctrl_mutex);
        if (!ENN_STL_ITER_IS_IN(exec_map, model_id))
            return ENN_RET_FAILED;
        CHECK_AND_RETURN_ERR(session_id >= exec_map[model_id]->inference_set.size(), ENN_RET_FAILED,
                             "session_id(%ju) should be in {0, %zu}\n", session_id, exec_map[model_id]->inference_set.size());
        auto &&exec_model_id_ref = exec_map[model_id]->inference_set[session_id].exec_model_id;
        CHECK_AND_RETURN_ERR((exec_id != 0 && exec_model_id_ref != 0), ENN_RET_FAILED, "exec_id(%ju) is already set!\n",
                             exec_model_id_ref);
        exec_map[model_id]->inference_set[session_id].is_commit = true;
        exec_model_id_ref = exec_id;

        return ENN_RET_SUCCESS;
    }

    // GetCommitFlag
    bool GetExecuteModelId(idType model_id, idType session_id, EnnExecuteModelId *exec_id) {
        std::lock_guard<std::mutex> guard(mdl_ctrl_mutex);
        if (!ENN_STL_ITER_IS_IN(exec_map, model_id))
            return ENN_RET_FAILED;
        CHECK_AND_RETURN_ERR(session_id >= exec_map[model_id]->inference_set.size(), ENN_RET_FAILED,
                             "session_id(%ju) should be in {0, %zu}\n", session_id, exec_map[model_id]->inference_set.size());
        *exec_id = exec_map[model_id]->inference_set[session_id].exec_model_id;
        return ENN_RET_SUCCESS;
    }

    const InferenceData *GetInferenceData(idType model_id, int session_id = 0) {
        std::lock_guard<std::mutex> guard(mdl_ctrl_mutex);
        if (CheckReadyToSetExec(model_id, session_id))
            return nullptr;
        CHECK_AND_RETURN_ERR(session_id >= exec_map[model_id]->inference_set.size(), nullptr,
                             "session_id(%u) should be in {0, %zu}\n", session_id, exec_map[model_id]->inference_set.size());

        return &(exec_map[model_id]->inference_set[session_id]);
    }

    const std::shared_ptr<execType> GetInferenceSet(idType model_id) {
        std::lock_guard<std::mutex> guard(mdl_ctrl_mutex);
        auto session_buf = GetSession(model_id);
        CHECK_AND_RETURN_ERR(session_buf == nullptr, nullptr, "Doesn't have any session data of ModelID(0x%" PRIX64 ")\n",
                             model_id);
        if (!ENN_STL_ITER_IS_IN(exec_map, model_id))
            return nullptr;
        else
            return exec_map[model_id];
    }

    EnnReturn GetBufferInfo(idType model_id, uint32_t *n_in, uint32_t *n_out) {
        std::lock_guard<std::mutex> guard(mdl_ctrl_mutex);
        auto session_buf = GetSession(model_id);
        CHECK_AND_RETURN_ERR(session_buf == nullptr, ENN_RET_INVAL,
                             "Doesn't have any session data of ModelID(0x%" PRIX64 ")\n", model_id);
        /* NOTE(hoon98.choi) : Assume that all buffer indexes are numbered sequentially */
        uint32_t i = 0, o = 0;
        for (auto &buf_ele : session_buf->buffers) {
            if (static_cast<enn_buf_dir_e>(buf_ele.dir) == ENN_DIR_IN)
                i++;
            if (static_cast<enn_buf_dir_e>(buf_ele.dir) == ENN_DIR_OUT)
                o++;
        }
        *n_in = i;
        *n_out = o;

        return ENN_RET_SUCCESS;
    }

    EnnReturn ClearInferenceData(idType model_id) {
        if (!ENN_STL_ITER_IS_IN(exec_map, model_id)) {
            ENN_INFO_PRINT("Model ID(0x%" PRIX64 ") has no inference data\n", model_id);
            return ENN_RET_FAILED;
        }
        exec_map.erase(model_id);
        return ENN_RET_SUCCESS;
    }

    std::shared_ptr<EnnBufferCore> GetModelLoadedBuf(idType model_id) {
        if (loaded_buf_map.find(model_id) != loaded_buf_map.end())
            return loaded_buf_map[model_id];
        return nullptr;
    }

    EnnReturn ClearModelData(idType model_id) {
        ClearInferenceData(model_id);
        CHECK_AND_RETURN_ERR((!ENN_STL_ITER_IS_IN(session_map, model_id)), ENN_RET_INVAL,
                             "No session data of model ID(0x%" PRIX64 ").\n", model_id);
        session_map.erase(model_id);
        auto_allocated_ext_buffer_lists.erase(model_id);
        // hara:TODO: clear ext_buffer_map
        return ENN_RET_SUCCESS;
    }

    EnnReturn ClearModelAll(void) {
        session_map.clear();
        exec_map.clear();
        auto_allocated_ext_buffer_lists.clear();

        // hara:TODO: ext_buffer_map clear
        return ENN_RET_SUCCESS;
    }

    EnnReturn DumpSessionToFile(idType model_id, std::string prefix_str, int session_id = 0) {
#ifdef ENN_BUILD_RELEASE
        ENN_UNUSED(model_id);
        ENN_UNUSED(session_id);
        ENN_UNUSED(prefix_str);
#else
        std::vector<std::tuple<void *, int>> bkup_to_munmap;
        ENN_DBG_PRINT("# Prepare dump: model_id(0x%" PRIX64 "), sess_id: %d\n", model_id, session_id);
        auto inf_data = GetInferenceData(model_id, session_id);

        for (uint32_t region_idx = 0; region_idx < inf_data->n_region; region_idx++) {
            /* NOTE(hoon98.choi): should contains time for dump of multiple sessions ? */
            std::string filename = prefix_str + std::string("_exec") + std::to_string(model_id) + std::string("_");
            auto &region = inf_data->inference_data[region_idx];
            void *region_addr = reinterpret_cast<void *>(region.addr);
            ENN_DBG_PRINT("# Region to dump: [%d] addr(%p), size(%d), offset(%d)\n", region_idx, region_addr, region.size,
                          region.offset);
            if (region_addr == nullptr) {
                if (region.fd->data[0] < 0) {
                    ENN_DBG_PRINT("Region %d is not allocated\n", region_idx);
                    continue;
                } else {  // if fd is not mmaped
                    region_addr = mmap(NULL, region.size, PROT_READ | PROT_WRITE, MAP_SHARED, region.fd->data[0], 0);
                    if (region_addr == nullptr) {
                        ENN_DBG_PRINT("Region %d has invalid fd\n", region_idx);
                        continue;
                    }
                    bkup_to_munmap.push_back({region_addr, region.size});
                }
            }
            filename = filename + std::string("index") + std::to_string(region_idx) + std::string("_offset") +
                       std::to_string(region.offset) + std::string(".dump");
            // offset is just a reference. not partially dumped.
            if (::enn::util::export_mem_to_file(filename.c_str(), region_addr, region.size)) {
                ENN_WARN_PRINT("Cannot export in the current directory!\n");
                return ENN_RET_IO;
            }
        }

        // unmap, if anything in backup vector
        for (auto &unmap_element : bkup_to_munmap)
            munmap(std::get<0>(unmap_element), std::get<1>(unmap_element));
#endif
        return ENN_RET_SUCCESS;
    }

  private:
    std::map<idType, std::shared_ptr<sessionType>> session_map;
    std::map<idType, std::shared_ptr<execType>> exec_map;
    std::map<idType, std::shared_ptr<EnnBufferCore>> loaded_buf_map;
    std::map<idType, std::map<idType, std::vector<std::shared_ptr<EnnBufferCore>>>> auto_allocated_ext_buffer_lists;
    std::mutex mdl_ctrl_mutex;

    static const EnnExecuteModelId EXEC_MODEL_ID_NOT_DEFINED = 0;

    EnnReturn GenerateInferenceSetAtExecMap(idType model_id, int32_t num, const std::shared_ptr<sessionType> session_buf) {
        exec_map[model_id] = std::make_shared<execType>();
        exec_map[model_id]->inference_set.resize(0);  // equals to clear()
        exec_map[model_id]->inference_set.resize(num);
        exec_map[model_id]->n_inference = num;

        ENN_INFO_PRINT("Model_id: 0x%" PRIX64 ", num: %d\n", model_id, num);

        // NOTE(hoon98.choi): Because inference_set class is auto-generated by HIDL,
        //                    This object should be initialized here.
        for (int i = 0; i < num; i++) {
            auto & inf_set = (exec_map[model_id]->inference_set)[i];
            inf_set.n_region = session_buf->regions.size();
            inf_set.is_commit = false;
            inf_set.inference_data.resize(session_buf->regions.size());
            for (int k = 0; k < session_buf->regions.size(); ++k) {
                // explicitly initialized: because this type based on HIDL, we cannot put intiailize code to constructor.
                inf_set.inference_data[k].size = 0;
            }
        }

        return ENN_RET_SUCCESS;
    }

    // NOTE(hoon98.choi): This function does not check validation of mem_object to put inf_data. use internally.
    EnnReturn setInferenceData(InferenceData *inf_data, int region_idx, EnnBufferCore *mem_object) {
        auto inf_data_ele = &inf_data->inference_data[region_idx];

        if constexpr (!(std::is_same<decltype(mem_object->get_native_handle()), void *>::value))
#ifdef ENN_MEDIUM_IF_HIDL
            inf_data_ele->fd = mem_object->get_native_handle();
#else
            inf_data_ele->data.data[0] = mem_object->fd;
#endif
        inf_data_ele->addr = reinterpret_cast<uint64_t>(mem_object->va);
        inf_data_ele->size = mem_object->size;
        inf_data_ele->offset = mem_object->offset;

        ENN_INFO_PRINT("Set inference Data: Done. (size: %d)\n", inf_data_ele->size);

        return ENN_RET_SUCCESS;
    }

    EnnReturn setInferenceData(const Buffer *select_buffer, InferenceData *inf_data, EnnBufferCore *mem_object) {
        return setInferenceData(inf_data, select_buffer->region_idx, mem_object);
    }

    EnnReturn CheckReadyToSetExec(idType model_id, int32_t session_id) {
        CHECK_AND_RETURN_ERR(GetSession(model_id) == nullptr, ENN_RET_INVAL,
                             "Doesn't have any session data. openModel First.\n");
        CHECK_AND_RETURN_ERR(!ENN_STL_ITER_IS_IN(session_map, model_id), ENN_RET_INVAL,
                             "model_id(%" PRIX64 " doesn't have session map\n", model_id);
        CHECK_AND_RETURN_ERR(!ENN_STL_ITER_IS_IN(exec_map, model_id), ENN_RET_INVAL,
                             "Inference data of model_id(0x%" PRIX64 ") is not generated!\n", model_id);
        CHECK_AND_RETURN_ERR((exec_map[model_id]->n_inference) <= session_id, ENN_RET_INVAL,
                             "session_id(%d) should be in [0, %d]\n", session_id, exec_map[model_id]->n_inference - 1);

        return ENN_RET_SUCCESS;
    }
};
}  // namespace client
}  // namespace enn

#endif  // SRC_CLIENT_ENN_MODEL_CONTAINER_H_
