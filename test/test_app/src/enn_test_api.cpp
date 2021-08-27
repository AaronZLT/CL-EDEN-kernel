/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#include <vector>
#include <chrono>

#include "enn_test.h"
#include "enn_test_log.h"

#include "common/enn_utils.h"
#include "common/enn_debug.h"

namespace enn_test {
void EnnTest::Initialize() {
    ENN_TEST_DEBUG("(+)");
    if (test_params.error == 1) throw RET_INIT_FAILED;
    EnnReturn ret;
    ret = EnnInitialize();
    if (ret != ENN_RET_SUCCESS) {
        ENN_TEST_ERR("EnnInitialize failed : %d\n", ret);
        throw RET_INIT_FAILED;
    }

    ENN_TEST_DEBUG("(-)");
}

void EnnTest::SetPreference() {
    ENN_TEST_DEBUG("(+)");
    EnnReturn ret;
    if (test_params.preset_id > 0) {
        ENN_TEST_DEBUG("set preference preset id as %d\n", test_params.preset_id);
        ret = EnnSetPreferencePresetId(test_params.preset_id);
        if (ret != ENN_RET_SUCCESS) {
            ENN_TEST_ERR("EnnSetPreferencePresetId failed : %d\n", ret);
            throw RET_SET_PREFERENCE_FAILED;
        }
    } else if (test_params.executionMode < PERF_MODE_SIZE) {
        ENN_TEST_DEBUG("set preference perf mode to %d\n", test_params.executionMode);
        ret = EnnSetPreferencePerfMode(test_params.executionMode);
        if (ret != ENN_RET_SUCCESS) {
            ENN_TEST_ERR("EnnSetPreferencePerfMode failed : %d\n", ret);
            throw RET_SET_PREFERENCE_FAILED;
        }
    }

    if (test_params.target_latency > 0) {
        ENN_TEST_DEBUG("set preference target latency as %d\n", test_params.target_latency);
        ret = EnnSetPreferenceTargetLatency(test_params.target_latency);
        if (ret != ENN_RET_SUCCESS) {
            ENN_TEST_ERR("EnnSetPreferenceTargetLatency failed : %d\n", ret);
            throw RET_SET_PREFERENCE_FAILED;
        }
    }

    if (test_params.priority > 0) {
        ENN_TEST_DEBUG("set preference priority as %d\n", test_params.priority);
        ret = EnnSetPreferencePriority(test_params.priority);
        if (ret != ENN_RET_SUCCESS) {
            ENN_TEST_ERR("EnnSetPreferencePriority failed : %d\n", ret);
            throw RET_SET_PREFERENCE_FAILED;
        }
    }

    if (test_params.tile_num > 0) {
        ENN_TEST_DEBUG("set preference tile_num as %d\n", test_params.tile_num);
        ret = EnnSetPreferenceTileNum(test_params.tile_num);
        if (ret != ENN_RET_SUCCESS) {
            ENN_TEST_ERR("EnnSetPreferenceTileNum failed : %d\n", ret);
            throw RET_SET_PREFERENCE_FAILED;
        }
    }

    if (test_params.core_affinity > 0) {
        ENN_TEST_DEBUG("set preference core_affinity as %d\n", test_params.core_affinity);
        ret = EnnSetPreferenceCoreAffinity(test_params.core_affinity);
        if (ret != ENN_RET_SUCCESS) {
            ENN_TEST_ERR("EnnSetPreferenceCoreAffinity failed : %d\n", ret);
            throw RET_SET_PREFERENCE_FAILED;
        }
    }

    ENN_TEST_DEBUG("(-)");
}

void EnnTest::OpenModel(EnnModelId &model_id) {
    ENN_TEST_DEBUG("(+)");
    if (test_params.error == 2) throw RET_OPEN_FAILED;
    EnnReturn ret;
    ret = EnnOpenModel(test_params.modelPath.c_str(), &model_id);
    if (ret != ENN_RET_SUCCESS) {
        ENN_TEST_ERR("EnnOpenModel failed : %d\n", ret);
        throw RET_OPEN_FAILED;
    }
    ENN_TEST_INFO("model_id : %ld\n", model_id);

    ENN_TEST_DEBUG("(-)");
}

void EnnTest::AllocateBuffers(const EnnModelId model_id, TestBuffers& test_buffer) {
    ENN_TEST_DEBUG("(+)");
    if (test_params.error == 3) throw RET_ALLOCATE_FAILED;
    EnnReturn ret;

    NumberOfBuffersInfo n_buf_info;
    std::vector<EnnBufferInfo> in_buf_info;
    std::vector<EnnBufferInfo> out_buf_info;
    EnnBufferInfo tmp_buf_info;

    ENN_TEST_DEBUG("Get buffer info\n");
    ret = EnnGetBuffersInfo(model_id, &n_buf_info);
    if (ret != ENN_RET_SUCCESS) {
        ENN_TEST_ERR("EnnGetBuffersInfo failed : %d\n", ret);
        throw RET_ALLOCATE_FAILED;
    }
    test_buffer.input_num = n_buf_info.n_in_buf;
    test_buffer.output_num = n_buf_info.n_out_buf;
    ENN_TEST_INFO("buffer info : input=%d, output=%d\n", test_buffer.input_num, test_buffer.output_num);

    ValidateNumOfBuffers(test_buffer);

    ret = EnnGenerateBufferSpace(model_id, test_params.session_num);
    if (ret != ENN_RET_SUCCESS) {
        ENN_TEST_ERR("EnnGenerateBufferSpace failed : %d\n", ret);
        throw RET_ALLOCATE_FAILED;
    }

    test_buffer.input_buffers.resize(test_buffer.input_num);
    test_buffer.output_buffers.resize(test_buffer.output_num);

    for (int idx = 0; idx < test_buffer.input_num; ++idx) {
        ret = EnnGetBufferInfoByIndex(model_id, ENN_DIR_IN, idx, &tmp_buf_info);
        if (ret != ENN_RET_SUCCESS) {
            ENN_TEST_ERR("EnnGetBufferInfoByIndex failed : %d\n", ret);
            throw RET_ALLOCATE_FAILED;
        }
        ENN_TEST_INFO("input[%d] tmp_buf_info.size : %d\n", idx, tmp_buf_info.size);
        in_buf_info.push_back(tmp_buf_info);
    }

    for (int idx = 0; idx < test_buffer.output_num; ++idx) {
        ret = EnnGetBufferInfoByIndex(model_id, ENN_DIR_OUT, idx, &tmp_buf_info);
        if (ret != ENN_RET_SUCCESS) {
            ENN_TEST_ERR("import_file_to_mem failed : %d\n", ret);
            throw RET_ALLOCATE_FAILED;
        }
        ENN_TEST_INFO("output[%d] tmp_buf_info.size : %d\n", idx, tmp_buf_info.size);
        out_buf_info.push_back(tmp_buf_info);
    }


    ENN_TEST_DEBUG("allocate in buffers\n");
    EnnBufferPtr in_buf, out_buf;
    for (int session = 0; session < test_params.session_num; ++session) {
        for (int idx = 0; idx < test_buffer.input_num; ++idx) {
            ret = EnnCreateBufferCache(in_buf_info[idx].size, &in_buf);
            if (ret != ENN_RET_SUCCESS) {
                ENN_TEST_ERR("EnnCreateBufferCache failed : %d\n", ret);
                throw RET_ALLOCATE_FAILED;
            }

            ret = EnnSetBufferByIndexWithSessionId(model_id, ENN_DIR_IN, idx, in_buf, session);
            if (ret != ENN_RET_SUCCESS) {
                ENN_TEST_ERR("EnnSetBufferByIndexWithSessionId failed : %d\n", ret);
                throw RET_ALLOCATE_FAILED;
            }
            ENN_TEST_INFO("input[%d] in_buf->size : %d\n", idx, in_buf->size);
            test_buffer.input_buffers[session].push_back(in_buf);
        }
    }

    ENN_TEST_DEBUG("allocate out buffers\n");
    for (int session = 0; session < test_params.session_num; ++session) {
        for (int idx = 0; idx < test_buffer.output_num; ++idx) {
            ret = EnnCreateBufferCache(out_buf_info[idx].size, &out_buf);
            if (ret != ENN_RET_SUCCESS) {
                ENN_TEST_ERR("EnnCreateBufferCache failed : %d\n", ret);
                throw RET_ALLOCATE_FAILED;
            }

            ret = EnnSetBufferByIndexWithSessionId(model_id, ENN_DIR_OUT, idx, out_buf, session);
            if (ret != ENN_RET_SUCCESS) {
                ENN_TEST_ERR("EnnSetBufferByIndexWithSessionId failed : %d\n", ret);
                throw RET_ALLOCATE_FAILED;
            }
            ENN_TEST_INFO("output[%d] out_buf->size : %d\n", idx, out_buf->size);
            test_buffer.output_buffers[session].push_back(out_buf);
        }
    }

    ENN_TEST_DEBUG("(-)");
}

void EnnTest::LoadInput(TestBuffers& test_buffer) {
    ENN_TEST_DEBUG("(+)");
    if (test_params.error == 4) throw RET_LOAD_FAILED;
    EnnReturn ret;
    uint32_t file_size;
    for (int idx = 0; idx < test_buffer.input_num; ++idx) {
        char* input_buf;
        ret = enn::util::import_file_to_mem(test_params.inputPath[idx].c_str(),
                                            &input_buf,
                                            &file_size);
        if (ret != ENN_RET_SUCCESS) {
            ENN_TEST_ERR("import_file_to_mem failed : %d\n", ret);
            throw RET_LOAD_FAILED;
        }
        for (int session = 0; session < test_params.session_num; ++session) {
            std::memcpy(test_buffer.input_buffers[session][idx]->va,
                        reinterpret_cast<void*>(input_buf),
                        file_size);
        }
    }
    ENN_TEST_DEBUG("(-)");
}

void EnnTest::LoadOutput(TestBuffers& test_buffer) {
    ENN_TEST_DEBUG("(+)");
    if (test_params.error == 4) throw RET_LOAD_FAILED;
    EnnReturn ret;
    uint32_t file_size;
    for (int idx = 0; idx < test_buffer.output_num; ++idx) {
        char* golden_output;
        ret = enn::util::import_file_to_mem(test_params.goldenPath[idx].c_str(),
                                            &golden_output,
                                            &file_size);
        if (ret != ENN_RET_SUCCESS) {
            ENN_TEST_ERR("import_file_to_mem failed : %d\n", ret);
            throw RET_LOAD_FAILED;
        }
        test_buffer.golden_outputs.push_back(golden_output);
        test_buffer.golden_sizes.push_back(file_size);
    }
    ENN_TEST_DEBUG("(-)");
}

void EnnTest::PrepareToExecute(const EnnModelId model_id) {
    ENN_TEST_DEBUG("(+)");
    if (test_params.error == 5) throw RET_COMMIT_FAILED;
    EnnReturn ret;
    for (int session = 0; session < test_params.session_num; ++session) {
        ret = EnnBufferCommitWithSessionId(model_id, session);
        if (ret != ENN_RET_SUCCESS) {
            ENN_TEST_ERR("EnnBufferCommitWithSessionId failed : %d\n", ret);
            throw RET_COMMIT_FAILED;
        }
    }
    ENN_TEST_DEBUG("(-)");
}

void EnnTest::Execute(const EnnModelId model_id, const int32_t session_id) {
    ENN_TEST_DEBUG("(+)");
    PROFILE_SCOPE("Application_Execution", 123456788);
    if (test_params.error == 6) throw RET_EXECUTE_FAILED;
    EnnReturn ret;
    ret = EnnExecuteModelWithSessionId(model_id, session_id);
    if (ret != ENN_RET_SUCCESS) {
        ENN_TEST_ERR("EnnExecuteModel failed : %d", ret);
        throw RET_EXECUTE_FAILED;
    }
    ENN_TEST_DEBUG("(-)");
}

void EnnTest::Execute_async(const EnnModelId model_id, const int32_t session_id) {
    ENN_TEST_DEBUG("(+)");
    PROFILE_SCOPE("Application_Execution", 123456788);
    if (test_params.error == 6) throw RET_EXECUTE_FAILED;
    EnnReturn ret;
    ret = EnnExecuteModelWithSessionIdAsync(model_id, session_id);
    if (ret != ENN_RET_SUCCESS) {
        ENN_TEST_ERR("EnnExecuteModel failed : %d", ret);
        throw RET_EXECUTE_FAILED;
    }
    ret = EnnExecuteModelWithSessionIdWait(model_id, session_id);
    if (ret != ENN_RET_SUCCESS) {
        ENN_TEST_ERR("EnnExecuteModel failed : %d", ret);
        throw RET_EXECUTE_FAILED;
    }
    ENN_TEST_DEBUG("(-)");
}

#define PRINT_DIFF_CNT 5
template <typename T>
bool CompareBuffersWithThreshold(void *golden_addr, void *out_addr, int32_t size, T threshold) {
    T *out_buf = reinterpret_cast<T *>(out_addr);
    T *golden_buf = reinterpret_cast<T *>(golden_addr);
    int32_t buf_size = size / sizeof(T);
    int32_t diff_cnt = 0;

    for (int idx = 0; idx < buf_size; ++idx) {
        auto diff = std::abs(out_buf[idx] - golden_buf[idx]);
        if (diff > threshold) {
            ++diff_cnt;
            if (diff_cnt < PRINT_DIFF_CNT) {
                std::string str = " - Difference [" + std::to_string(idx) + "] : golden "
                                  + std::to_string(golden_buf[idx]) + " vs output "
                                  + std::to_string(out_buf[idx]) + "\n";
                PRINT("%s", str.c_str());
            }
            if (diff_cnt == PRINT_DIFF_CNT) {
                PRINT(" - Difference is more than %d\n", PRINT_DIFF_CNT);
            }
        }
    }
    return (diff_cnt > 0) ? false : true;
}

EnnTestReturn EnnTest::CompareGolden(TestBuffers& test_buffer, const int32_t iter,
                                     const int32_t session_id) {
    ENN_TEST_DEBUG("(+)");
    if (test_params.error == 9) return RET_GOLDEN_MISMATCH;
    EnnTestReturn ret = RET_SUCCESS;
    for (int idx = 0; idx < test_buffer.output_num; ++idx) {
        bool matching = true;
        if (test_params.threshold > THRESHOLD_MIN) {
            matching = CompareBuffersWithThreshold<float>(test_buffer.golden_outputs[idx],
                                reinterpret_cast<char *>(test_buffer.output_buffers[session_id][idx]->va),
                                test_buffer.golden_sizes[idx],
                                static_cast<float>(test_params.threshold));
        } else {
            matching = CompareBuffersWithThreshold<uint8_t>(test_buffer.golden_outputs[idx],
                                reinterpret_cast<char *>(test_buffer.output_buffers[session_id][idx]->va),
                                test_buffer.golden_sizes[idx],
                                static_cast<uint8_t>(test_params.threshold));
        }
        if (matching) {
            PRINT("[Iter: %d]\tGolden Matched\n", iter);
        } else {
            PRINT_RED("[Iter: %d]\tFailed : Golden Mismatch\n", iter);
            ret = RET_GOLDEN_MISMATCH;
        }
    }
    ENN_TEST_DEBUG("(-)");
    return ret;
}

EnnTestReturn EnnTest::Execute_iter(const EnnModelId model_id, TestBuffers& test_buffer,
                                    const int32_t thread_id) {
    ENN_TEST_DEBUG("(+)");
    EnnTestReturn ret = RET_SUCCESS;
    const int32_t session_id = 0;   // Todo : support multi session execution
    for (int iter = 0; iter < test_params.iter; ++iter) {
        if (test_params.isAsync) {
            EnnTest::Execute_async(model_id, session_id);
        } else {
            EnnTest::Execute(model_id, session_id);
        }

        if (test_params.skipMatch) {
            PRINT("[Iter: %d]\tskip golden match\n", iter + 1);
            test_reporter[thread_id][session_id]->IncreasePass();
        } else {
            ret = EnnTest::CompareGolden(test_buffer, iter + 1, session_id);
            if (ret == RET_SUCCESS) {
                test_reporter[thread_id][session_id]->IncreasePass();
            }
        }
    }

    ENN_TEST_DEBUG("(-)");
    return ret;
}

EnnTestReturn EnnTest::Execute_duration(const EnnModelId model_id, TestBuffers& test_buffer) {
    ENN_TEST_DEBUG("(+)");
    EnnTestReturn ret = RET_SUCCESS;
    const int32_t session_id = 0;
    int32_t iter = 0;

    auto start = std::chrono::steady_clock::now();
    auto timeout = std::chrono::milliseconds(test_params.duration * 1000);

    while (std::chrono::steady_clock::now() - start < timeout) {
        ++iter;
        if (test_params.isAsync) {
            EnnTest::Execute_async(model_id, session_id);
        } else {
            EnnTest::Execute(model_id, session_id);
        }

        if (test_params.skipMatch) {
            PRINT("[Iter: %d]\tskip golden match\n", iter);
        } else {
            ret = EnnTest::CompareGolden(test_buffer, iter, session_id);
        }
    }
    ENN_TEST_DEBUG("(-)");
    return ret;
}

void EnnTest::DumpOutput(TestBuffers& test_buffer) {
    ENN_TEST_DEBUG("(+)");
    for (int idx = 0; idx < test_buffer.output_num; ++idx) {
        std::string filename = default_path + "output_" + std::to_string(idx) + ".bin";
        char* data = reinterpret_cast<char *>(test_buffer.output_buffers[0][idx]->va);
        int32_t size = test_buffer.golden_sizes[idx];

        if (size <= 0) {
            PRINT_RED("Golden size error : %d", size);
            throw RET_DUMP_FAILED;
        }
        if (data == nullptr) {
            PRINT_RED("Addr is nullptr");
            throw RET_DUMP_FAILED;
        }
        std::ofstream dumpFile(filename, std::ofstream::binary);
        dumpFile.write(data, size);
        dumpFile.close();
        PRINT("Dump file is created : %s", filename.c_str());
    }
    ENN_TEST_DEBUG("(-)");
}

void EnnTest::ValidateNumOfBuffers(const TestBuffers& test_buffer) const {
    ENN_TEST_DEBUG("(+)");
    if (test_params.inputPath.size() != test_buffer.input_num) {
        PRINT_RED("Invalid number of input buffers (%zu/%d)",
            test_params.inputPath.size(), test_buffer.input_num);
        throw EnnTestReturn::RET_NUM_OF_BUFFERS_FAILED;
    }

    if (test_params.goldenPath.size() != test_buffer.output_num) {
        PRINT_RED("Invalid number of output buffers (%d/%d)",
            test_params.goldenPath.size(), test_buffer.output_num);
        throw _EnnTestReturn::RET_NUM_OF_BUFFERS_FAILED;
    }
    ENN_TEST_DEBUG("(-)");
}

void EnnTest::ReleaseBuffer(TestBuffers& test_buffer) {
    ENN_TEST_DEBUG("(+)");
    ENN_TEST_DEBUG("Release input buffers");
    for (auto& in_bufs : test_buffer.input_buffers) {
        for (auto& in_buf : in_bufs) {
            EnnReleaseBuffer(in_buf);
        }
    }
    ENN_TEST_DEBUG("Release output buffers");
    for (auto& out_bufs : test_buffer.output_buffers) {
        for (auto& out_buf : out_bufs) {
            EnnReleaseBuffer(out_buf);
        }
    }
    ENN_TEST_DEBUG("Release other buffers");
    for (auto& golden_buf : test_buffer.golden_outputs) {
        if (golden_buf != nullptr) {
            free(golden_buf);
        }
    }
    test_buffer.golden_outputs.clear();
    ENN_TEST_DEBUG("(-)");
}

void EnnTest::CloseModel(const EnnModelId model_id) {
    ENN_TEST_DEBUG("(+)");
    if (test_params.error == 7) throw RET_CLOSE_FAILED;
    EnnReturn ret;
    ret = EnnCloseModel(model_id);
    if (ret != ENN_RET_SUCCESS) {
        ENN_TEST_ERR("EnnCloseModel failed : %d\n", ret);
        throw RET_CLOSE_FAILED;
    }
    ENN_TEST_DEBUG("(-)");
}

void EnnTest::Deinit() {
    ENN_TEST_DEBUG("(+)");
    if (test_params.error == 8) throw RET_DEINIT_FAILED;
    EnnReturn ret;
    ret = EnnDeinitialize();
    if (ret != ENN_RET_SUCCESS) {
        ENN_TEST_ERR("EnnDeinitialize failed : %d\n", ret);
        throw RET_DEINIT_FAILED;
    }
    ENN_TEST_DEBUG("(-)");
}

}  // namespace enn_test
