/*
 * Copyright (C) 2020 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or
 * distributed, transmitted, transcribed, stored in a retrieval system or
 * translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed to third parties
 * without the express written permission of Samsung Electronics.
 *
 */

/**
 * @brief gtest main for unit test
 * @file enn_gtest_internal_unittest_temp.cc
 * @author Hoon Choi
 * @date 2020_12_10
 */

#include "client/enn_api-public.h"
#include "gtest/gtest.h"
#include "client/enn_api.h"
#include "client/enn_context_manager.h"
#include "tool/profiler/include/ExynosNnProfilerApi.h"
#include "common/helper_templates.hpp"
#include "test/materials.h"

#include <atomic>
#include <cstring>
#include <future>
#include <chrono>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>
#include <unistd.h>

extern enn::client::EnnContextManager enn_context;

namespace enn {
namespace test {
namespace internal {

#ifdef SCHEMA_NNC_V1
#define SAMPLE_MODEL_FILENAME OLYMPUS::NPU::IV3::NNC;
#else
#define SAMPLE_MODEL_FILENAME PAMIR::NPU::IV3::NNC;
#endif

class ENN_GT_API_UNIT_TEST: public testing::Test {};

TEST_F(ENN_GT_API_UNIT_TEST, init_deinit) {
    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnDeinitialize(), ENN_RET_SUCCESS);
}

TEST_F(ENN_GT_API_UNIT_TEST, init_deinit_3_times) {
    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnDeinitialize(), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnDeinitialize(), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnDeinitialize(), ENN_RET_SUCCESS);
}

/**
 * @brief Test golden match which model has 1 input and 1 output, and threshold type is float
 *
 * @param model_file target model file
 * @param golden_in_file golden in
 * @param golden_out_file golden out
 * @param golden_threshold threshold in float type
 * @return true pass
 * @return false failed
 */

#define ITERATION_N_ENV_NAME "ENN_ITER"
#define DEFAULT_ITER (1)
#define INITIAL_ITER (-1)


static int getenv_or_default(const char* prop_name, int def_value) {
    char *env_str = getenv(prop_name);
    if (env_str == nullptr) {
        return def_value;
    }
    return atoi(env_str);
}

static int getenv_or_default_ge_1(const char* prop_name, int def_value) {
    int val = getenv_or_default(prop_name, def_value);
    return (0 < val ? val : def_value);
}

template <typename UnitType>
static bool test_repeat_iv3_for_60sec(const std::string &model_file, const std::string golden_in_file,
                                  const std::string golden_out_file, float golden_threshold, bool check_with_golden_size,
                                  int num_exec = INITIAL_ITER) {
    ENN_UNUSED(num_exec);
    ENN_UNUSED(golden_threshold);
    ENN_UNUSED(check_with_golden_size);
    int64_t timeout_ms = getenv_or_default("ENN_TIMEOUT_MS", 60 * 1000);

    std::cout << " # Timeout (ms): " << timeout_ms << std::endl
              << "   You may override this by 'export ENN_TIMEOUT_MS=timeout_ms'." << std::endl;

    char *env_str;
    std::string model_file_path;
    env_str = getenv("ENN_MODEL");
    model_file_path = env_str == nullptr ? model_file : env_str;

    std::string in_file_path;
    env_str = getenv("ENN_INPUT");
    in_file_path = env_str == nullptr ? golden_in_file : env_str;

    std::string golden_file_path;
    env_str = getenv("ENN_GOLDEN");
    golden_file_path = env_str == nullptr ? golden_out_file : env_str;

    std::cout << " # model: " << model_file_path << std::endl
              << " # input: " << in_file_path << std::endl
              << " # golden: " << golden_file_path << std::endl
              << "   You may override them by setting env variables ENN_MODEL, ENN_INPUT and ENN_GOLDEN with full file paths." << std::endl;
    EnnModelId model_id = 0;
    NumberOfBuffersInfo n_buf_info;
    uint32_t i_, o_;
    EnnBufferInfo enn_inbuf_info;
    EnnBufferInfo enn_outbuf_info;
    EnnBufferPtr enn_inbuf;
    EnnBufferPtr enn_outbuf;

    uint32_t golden_in_file_size;
    char* golden_out_buf = nullptr;
    uint32_t golden_out_file_size;
    uint32_t size_to_check;

    ENN_TST_PRINT("1. Initialize \n");
    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);

    ENN_TST_PRINT("2. Open Model (test) : ModelID(0x%lX)\n", model_id);
    EXPECT_EQ(EnnOpenModel(model_file_path.c_str(), &model_id), ENN_RET_SUCCESS);

    ENN_TST_PRINT("3. Prepare Input & Output Buffers\n");
    EXPECT_EQ(EnnGetBuffersInfo(model_id, &n_buf_info), ENN_RET_SUCCESS);
    i_ = n_buf_info.n_in_buf;
    o_ = n_buf_info.n_out_buf;
    EXPECT_EQ(i_, 1);
    EXPECT_EQ(o_, 1);
    EXPECT_EQ(EnnGetBufferInfoByIndex(model_id, ENN_DIR_IN, 0, &enn_inbuf_info), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnCreateBufferCache(enn_inbuf_info.size, &enn_inbuf), ENN_RET_SUCCESS);
    EXPECT_EQ(enn::util::import_file_to_mem(in_file_path.c_str(), reinterpret_cast<char**>(&enn_inbuf->va), &golden_in_file_size, 0), ENN_RET_SUCCESS);
    EXPECT_EQ(golden_in_file_size, enn_inbuf_info.size);

    EXPECT_EQ(EnnGetBufferInfoByIndex(model_id, ENN_DIR_OUT, 0, &enn_outbuf_info), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnCreateBufferCache(enn_outbuf_info.size, &enn_outbuf), ENN_RET_SUCCESS);

    EXPECT_EQ(EnnSetBufferByIndex(model_id, ENN_DIR_IN, 0, enn_inbuf), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnSetBufferByIndex(model_id, ENN_DIR_OUT, 0, enn_outbuf), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnBufferCommit(model_id), ENN_RET_SUCCESS);

    EXPECT_EQ(enn::util::import_file_to_mem(golden_file_path.c_str(), &golden_out_buf, &golden_out_file_size, 0), ENN_RET_SUCCESS);
    size_to_check = check_with_golden_size ? golden_out_file_size : enn_outbuf_info.size;

    ENN_TST_PRINT("4-1. Test for %zd ms\n", timeout_ms);

    std::atomic_bool flag{true};
    auto start_time = std::chrono::high_resolution_clock::now();
    std::thread t1([&flag, model_id]() {
        while (flag.load()) {
            EXPECT_EQ(EnnExecuteModel(model_id), ENN_RET_SUCCESS);
            //EXPECT_EQ(enn::util::CompareBuffersWithThreshold<UnitType>(reinterpret_cast<char*>(enn_outbuf->va), golden_out_buf, size_to_check, nullptr, golden_threshold), 0);
        }
    });
    std::thread t2([&flag, timeout_ms]() {
        auto sleep_for_ns = (timeout_ms - 10 /*adjust*/) * 1000;
        usleep(sleep_for_ns);
        flag.store(false);
    });

    t2.join();
    t1.join();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_ms = end_time - start_time;
    std::cout << " stopped after " << duration_ms.count() << " ms" << std::endl;
    // wia

    ENN_TST_PRINT("5. Release Buffers\n");
    free(golden_out_buf);
    EnnReleaseBuffer(enn_outbuf);
    EnnReleaseBuffer(enn_inbuf);

    ENN_TST_PRINT("6. Close Model\n");
    EXPECT_EQ(EnnCloseModel(model_id), ENN_RET_SUCCESS);
    ENN_TST_PRINT("7. Deinitialize\n");
    EXPECT_EQ(EnnDeinitialize(), ENN_RET_SUCCESS);

    return true;
}

template <typename UnitType>
static bool test_golden_match(const std::string &model_file, const std::string &golden_in_file,
                              const std::vector<std::string> &golden_out_files, float golden_threshold,
                              bool check_with_golden_size = false, int num_exec = INITIAL_ITER) {
    int num_execution = getenv_or_default_ge_1(ITERATION_N_ENV_NAME, 0 < num_exec ? num_exec : DEFAULT_ITER);
    const int num_output = (int)golden_out_files.size();

    std::cout << " # Total Iteration: " << num_execution << ", You can control # of execution to set \"ENN_ITER=n\""
              << std::endl;

    ENN_TST_PRINT("1. Initialize \n");
    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);

    EnnModelId model_id = 0;
    ENN_TST_PRINT("2. Open Model (test) : ModelID(0x%lX)\n", model_id);
    EXPECT_EQ(EnnOpenModel(model_file.c_str(), &model_id), ENN_RET_SUCCESS);

    ENN_TST_PRINT("3. get buffer info: number of input, outputs \n");
    uint32_t i, o = 0;
    NumberOfBuffersInfo n_buf_info;

    ENN_TST_PRINT("3. get buffer info: number of input, outputs \n");
    EXPECT_EQ(EnnGetBuffersInfo(model_id, &n_buf_info), ENN_RET_SUCCESS);
    i = n_buf_info.n_in_buf;
    o = n_buf_info.n_out_buf;

    // check if input and output size is num_output
    EXPECT_EQ(i, 1);
    EXPECT_EQ(o, num_output);

    ENN_TST_PRINT("4. Buffers: Manually generates buffers\n");
    std::vector<EnnBufferPtr > allocated_buffer_v;
    EnnBufferInfo tmp_buf_info;

    // allocate input and load golden-data
    uint32_t file_size;
    EnnGetBufferInfoByIndex(model_id, ENN_DIR_IN, 0, &tmp_buf_info);  // IFM
    EnnBufferPtr buf_in;
    EXPECT_EQ(ENN_RET_SUCCESS, EnnCreateBufferCache(tmp_buf_info.size, &buf_in));
    EXPECT_EQ(ENN_RET_SUCCESS,
              enn::util::import_file_to_mem(golden_in_file.c_str(), reinterpret_cast<char **>(&(buf_in->va)), &file_size));
    EXPECT_EQ(static_cast<uint32_t>(file_size), buf_in->size);
    ENN_TST_PRINT("golden in  file size: %d\n", file_size);
    allocated_buffer_v.push_back(buf_in);
    EXPECT_EQ(ENN_RET_SUCCESS, EnnSetBufferByIndex(model_id, ENN_DIR_IN, 0, buf_in));

    // allocate other buffers
    EnnBufferPtr buf_out[num_output];
    for (int out_idx = 0; out_idx < num_output; ++out_idx) {
        EnnGetBufferInfoByIndex(model_id, ENN_DIR_OUT, out_idx, &tmp_buf_info);  // OFM
        EXPECT_EQ(ENN_RET_SUCCESS, EnnCreateBufferCache(tmp_buf_info.size, &(buf_out[out_idx])));
        EXPECT_EQ(ENN_RET_SUCCESS, EnnSetBufferByIndex(model_id, ENN_DIR_OUT, out_idx, buf_out[out_idx]));
        allocated_buffer_v.push_back(buf_out[out_idx]);
    }

    ENN_TST_PRINT("5. Send buffer information to Service \n");
    EXPECT_EQ(EnnBufferCommit(model_id), ENN_RET_SUCCESS);  // buffer commit to apply swapped buffer
    // allocate golden-out
    // internally create buffer to golden_out_heap_buf (this utility returns raw pointer, not smart pointer)
    std::vector<char *> golden_out_heap_buf(num_output);
    std::vector<int> golden_compare_size(num_output);
    for (int out_idx = 0; out_idx < num_output; ++out_idx) {
        EXPECT_EQ(ENN_RET_SUCCESS, enn::util::import_file_to_mem(golden_out_files[out_idx].c_str(),
                                                                 &golden_out_heap_buf[out_idx], &file_size));
        ENN_TST_PRINT("golden out file size: %d, ptr: %p\n", file_size, golden_out_heap_buf[out_idx]);
        golden_compare_size[out_idx] = check_with_golden_size ? file_size : buf_out[out_idx]->size;
    }

    {
        ENN_TST_PRINT("6. Execute Model, %d times\n", num_exec);
        for (int iter = 0; iter < num_execution; ++iter) {
            std::cout << "Iteration: " << (iter + 1) << std::endl;
            EXPECT_EQ(EnnExecuteModel(model_id), ENN_RET_SUCCESS);

            ENN_TST_PRINT("7. Compare to golden data, num_output: %d\n", num_output);
            for (int out_idx = 0; out_idx < num_output; ++out_idx) {
                EXPECT_EQ(0, enn::util::CompareBuffersWithThreshold<UnitType>(
                                 golden_out_heap_buf[out_idx], reinterpret_cast<char *>(buf_out[out_idx]->va),
                                 golden_compare_size[out_idx], nullptr, golden_threshold));
            }
        }
    }
    ENN_TST_PRINT("9. Release Buffers\n");
    for (auto &ve : allocated_buffer_v) {
        EnnReleaseBuffer(ve);
    }

    ENN_TST_PRINT("10. Close Model & Deinitialize\n");
    EXPECT_EQ(EnnCloseModel(model_id), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnDeinitialize(), ENN_RET_SUCCESS);

    for (int out_idx = 0; out_idx < num_output; ++out_idx) {
        free(golden_out_heap_buf[out_idx]);
    }

    return true;
}

template <typename UnitType>
static bool test_golden_match_1_1(const std::string &model_file, const std::string golden_in_file,
                                  const std::string golden_out_file, float golden_threshold, bool check_with_golden_size,
                                  int num_exec = INITIAL_ITER, bool load_from_memory = false) {
    int num_execution = getenv_or_default_ge_1(ITERATION_N_ENV_NAME, 0 < num_exec ? num_exec : DEFAULT_ITER);
    char *buf_to_load = nullptr;
    uint32_t loaded_size;

    std::cout << " # Total Iteration: " << num_execution << ", You can control # of execution to set \"ENN_ITER=n\"" << std::endl;

    ENN_TST_PRINT("1. Initialize \n");
    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);

    EnnModelId model_id = 0;
    ENN_TST_PRINT("2. Open Model (test) : ModelID(0x%lX), %s\n", model_id, (load_from_memory ? "Load From Memory" : "Load from File"));
    if (load_from_memory) {
        EXPECT_EQ(enn::util::import_file_to_mem(model_file.c_str(), &buf_to_load, &loaded_size), ENN_RET_SUCCESS);
        EXPECT_EQ(EnnOpenModelFromMemory(buf_to_load, loaded_size, &model_id), ENN_RET_SUCCESS);
    } else {
        EXPECT_EQ(EnnOpenModel(model_file.c_str(), &model_id), ENN_RET_SUCCESS);
    }

    ENN_TST_PRINT("3. get buffer info: number of input, outputs \n");
    uint32_t i, o = 0;
    NumberOfBuffersInfo n_buf_info;

    ENN_TST_PRINT("3. get buffer info: number of input, outputs \n");
    EXPECT_EQ(EnnGetBuffersInfo(model_id, &n_buf_info), ENN_RET_SUCCESS);
    i = n_buf_info.n_in_buf;
    o = n_buf_info.n_out_buf;

    // check if input and output size is 1
    EXPECT_EQ(i, 1);
    EXPECT_EQ(o, 1);

    ENN_TST_PRINT("4. Buffers: Manually generates buffers\n");
    std::vector<EnnBufferPtr > allocated_buffer_v;
    EnnBufferInfo tmp_buf_info;

    // allocate input and load golden-data
    uint32_t file_size;
    EnnGetBufferInfoByIndex(model_id, ENN_DIR_IN, 0, &tmp_buf_info);  // IFM
    EnnBufferPtr buf_in;
    EXPECT_EQ(ENN_RET_SUCCESS, EnnCreateBufferCache(tmp_buf_info.size, &buf_in));
    EXPECT_EQ(ENN_RET_SUCCESS,
              enn::util::import_file_to_mem(golden_in_file.c_str(), reinterpret_cast<char **>(&(buf_in->va)), &file_size));
    EXPECT_EQ(static_cast<uint32_t>(file_size), buf_in->size);
    ENN_TST_PRINT("golden in  file size: %d\n", file_size);
    allocated_buffer_v.push_back(buf_in);
    EXPECT_EQ(ENN_RET_SUCCESS, EnnSetBufferByIndex(model_id, ENN_DIR_IN, 0, buf_in));

    // allocate other buffers
    EnnGetBufferInfoByIndex(model_id, ENN_DIR_OUT, 0, &tmp_buf_info);  // OFM
    EnnBufferPtr buf_out;
    EXPECT_EQ(ENN_RET_SUCCESS, EnnCreateBufferCache(tmp_buf_info.size, &buf_out));
    EXPECT_EQ(ENN_RET_SUCCESS, EnnSetBufferByIndex(model_id, ENN_DIR_OUT, 0, buf_out));
    allocated_buffer_v.push_back(buf_out);

    ENN_TST_PRINT("5. Send buffer information to Service \n");
    EXPECT_EQ(EnnBufferCommit(model_id), ENN_RET_SUCCESS);  // buffer commit to apply swapped buffer
    // allocate golden-out
    // internally create buffer to golden_out_heap_buf (this utility returns raw pointer, not smart pointer)
    char *golden_out_heap_buf = nullptr;
    EXPECT_EQ(ENN_RET_SUCCESS, enn::util::import_file_to_mem(golden_out_file.c_str(), &golden_out_heap_buf, &file_size));
    ENN_TST_PRINT("golden out file size: %d, ptr: %p\n", file_size, golden_out_heap_buf);
    int golden_compare_size = check_with_golden_size ? file_size : buf_out->size;

    {
        ENN_TST_PRINT("6. Execute Model, %d times\n", num_exec);
        for (int iter = 0; iter < num_execution; ++iter) {
            std::cout << "Iteration: " << (iter + 1) << std::endl;
            EXPECT_EQ(EnnExecuteModel(model_id), ENN_RET_SUCCESS);

            ENN_TST_PRINT("7. Compare to golden data\n");
            EXPECT_EQ(0, enn::util::CompareBuffersWithThreshold<UnitType>(golden_out_heap_buf,
                                                                          reinterpret_cast<char *>(buf_out->va),
                                                                          golden_compare_size, nullptr, golden_threshold));
        }
    }
    ENN_TST_PRINT("9. Release Buffers\n");
    for (auto &ve : allocated_buffer_v) {
        EnnReleaseBuffer(ve);
    }

    ENN_TST_PRINT("10. Close Model & Deinitialize\n");
    EXPECT_EQ(EnnCloseModel(model_id), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnDeinitialize(), ENN_RET_SUCCESS);

    free(golden_out_heap_buf);
    if (load_from_memory)
        free(buf_to_load);

    return true;
}

template <typename UnitType>
static bool test_golden_match_1_1(const std::string &model_file, const std::string golden_in_file,
                                  const std::string golden_out_file, float golden_threshold, int num_exec = INITIAL_ITER,
                                  bool load_from_memory = false) {
    return test_golden_match_1_1<UnitType>(model_file, golden_in_file, golden_out_file, golden_threshold, false, num_exec,
                                           load_from_memory);
}

/**
 * @brief api_test_load_2_return_test
 * @test  Check file is correctly transferred to service
 *        Call EnnOpenModelTest() instead of calling EnnOpenModel()
 * @test  Scenario
 *        IN1  filebuffer pattern : { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };
 *        IN2  model_id : 0x1001
 *        IN3  preset : 0x1001
 *        OUT1 ret_loadModel.model_id : 0x2001
 *        OUT2 ret_loadModel.regions[0].shape : { 1, 2, 3, 4 }
 *        OUT3 ret_loadModel.buffers.size() : size of pattern
 * @result if test scenario is correct, returns ENN_RET_SUCCESS
 *
 */

TEST_F(ENN_GT_API_UNIT_TEST, open_model_test_with_test_model_nnc) {
    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);
    // EnnModelPreference test;
    EnnModelId model_id = 0x1001;
    const std::string filename = SAMPLE_MODEL_FILENAME;
    auto ret1 = EnnOpenModel(filename.c_str(), &model_id);
    EXPECT_EQ(ENN_RET_SUCCESS, ret1);
    EnnCloseModel(model_id);
    EXPECT_EQ(EnnDeinitialize(), ENN_RET_SUCCESS);
}

TEST_F(ENN_GT_API_UNIT_TEST, open_model_test_with_test_model_cgo_gaussan3x3) {
    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);
    // EnnModelPreference test;
    EnnModelId model_id = 0x1001;
    const std::string filename = PAMIR::DSP::CGO::Gaussian3x3;
    auto ret1 = EnnOpenModel(filename.c_str(), &model_id);
    ASSERT_EQ(ENN_RET_SUCCESS, ret1);
    EnnCloseModel(model_id);
    ASSERT_EQ(EnnDeinitialize(), ENN_RET_SUCCESS);
}

TEST_F(ENN_GT_API_UNIT_TEST, open_model_test_with_test_model_cgo_gaussan3x3_from_memory) {
    const std::string filename = PAMIR::DSP::CGO::Gaussian3x3;
    char *tmp_load = nullptr;
    uint32_t size = 0;

    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);
    // EnnModelPreference test;
    EnnModelId model_id = 0x1001;
    enn::util::import_file_to_mem(filename.c_str(), &tmp_load, &size);
    auto ret1 = EnnOpenModelFromMemory(tmp_load, size, &model_id);
    ASSERT_EQ(ENN_RET_SUCCESS, ret1);
    EnnCloseModel(model_id);

    free(tmp_load);
    ASSERT_EQ(EnnDeinitialize(), ENN_RET_SUCCESS);
}

TEST_F(ENN_GT_API_UNIT_TEST, open_model_test_with_incorrect_model_nnc) {
    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);
    EnnModelId model_id;
    // set incorrect file name ( but exists )
    const std::string filename = "/data/vendor/enn/models/pamir/NPU_IV3/NPU_InceptionV3_golden_data.bin";
    auto ret1 = EnnOpenModel(filename.c_str(), &model_id);
    EXPECT_NE(ENN_RET_SUCCESS, ret1);
    EXPECT_NE(EnnCloseModel(model_id), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnDeinitialize(), ENN_RET_SUCCESS);
}

#if 0 // Because DSP custom kernel is not imeplemented yet.
TEST_F(ENN_GT_API_UNIT_TEST, open_model_test_with_test_model_cgo_IV3) {
    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);
    // EnnModelPreference test;
    EnnModelId model_id = 0x1001;
    const std::string filename = PAMIR::DSP::CGO::IV3;
    auto ret1 = EnnOpenModel(filename.c_str(), &model_id);
    ASSERT_EQ(ENN_RET_SUCCESS, ret1);
    EnnCloseModel(model_id);
    ASSERT_EQ(EnnDeinitialize(), ENN_RET_SUCCESS);
}

TEST_F(ENN_GT_API_UNIT_TEST, open_model_test_with_test_model_cgo_IV3_from_memory) {
    const std::string filename = PAMIR::DSP::CGO::IV3;
    char *tmp_load = nullptr;
    uint32_t size = 0;

    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);
    // EnnModelPreference test;
    EnnModelId model_id = 0x1001;
    enn::util::import_file_to_mem(filename.c_str(), &tmp_load, &size);
    auto ret1 = EnnOpenModelFromMemory(tmp_load, size, &model_id);
    ASSERT_EQ(ENN_RET_SUCCESS, ret1);
    EnnCloseModel(model_id);

    free(tmp_load);
    ASSERT_EQ(EnnDeinitialize(), ENN_RET_SUCCESS);
}
#endif

/**
 * @brief api_test_load_1_return_test:
 * @test  check if a user put a wrong filename as a param to EnnOpenModel()
 */
TEST_F(ENN_GT_API_UNIT_TEST, open_test_with_invalid_file_name) {
    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);
    EnnModelId model_id;
    auto ret = EnnOpenModel("there_s_no_file_there", &model_id);
    EXPECT_NE(ENN_RET_SUCCESS, ret);
    EXPECT_EQ(EnnDeinitialize(), ENN_RET_SUCCESS);
}

TEST_F(ENN_GT_API_UNIT_TEST, multiple_load_test_4models) {
    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);
    // EnnModelPreference test;
    EnnModelId model_id1, model_id2, model_id3, model_id4;
    const std::string filename = SAMPLE_MODEL_FILENAME;
    auto ret1 = EnnOpenModel(filename.c_str(), &model_id1);
    auto ret2 = EnnOpenModel(filename.c_str(), &model_id2);
    auto ret3 = EnnOpenModel(filename.c_str(), &model_id3);
    auto ret4 = EnnOpenModel(filename.c_str(), &model_id4);
    EXPECT_EQ(ENN_RET_SUCCESS, ret1);
    EXPECT_EQ(ENN_RET_SUCCESS, ret2);
    EXPECT_EQ(ENN_RET_SUCCESS, ret3);
    EXPECT_EQ(ENN_RET_SUCCESS, ret4);
    EnnCloseModel(model_id1);
    EnnCloseModel(model_id2);
    EnnCloseModel(model_id3);
    EnnCloseModel(model_id4);
    EXPECT_EQ(EnnDeinitialize(), ENN_RET_SUCCESS);
}

TEST_F(ENN_GT_API_UNIT_TEST, open_and_get_buffers_info_test) {
    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);
    EnnModelId model_id;
    const std::string filename = SAMPLE_MODEL_FILENAME;
    auto ret = EnnOpenModel(filename.c_str(), &model_id);

    // EnnModelPreference test;
    // EnnModelId model_id = 0x1001;
    // auto ret = EnnOpenModelTest(&model_id, test);
    // auto ret = EnnOpenModel(filename, &model_id);
    EXPECT_EQ(ENN_RET_SUCCESS, ret);

    NumberOfBuffersInfo bi;
    EnnBufferInfo test_info, test_info2;
    EXPECT_EQ(ENN_RET_SUCCESS, ret);

    /* Test 1: get in, out, ext buffers num of model */
    EXPECT_EQ(EnnGetBuffersInfo(model_id, &bi), ENN_RET_SUCCESS);

    /* Get buffer from model, using {dir, index} == {label} */
    EXPECT_EQ(EnnGetBufferInfoByIndex(model_id, ENN_DIR_IN, 0, &test_info), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnGetBufferInfoByLabel(model_id, test_info.label, &test_info2), ENN_RET_SUCCESS);

    /* Check if two info data structures are same */
    EXPECT_EQ(test_info.n == test_info2.n, true);
    EXPECT_EQ(test_info.width == test_info2.width, true);
    EXPECT_EQ(test_info.height == test_info2.height, true);
    EXPECT_EQ(test_info.size == test_info2.size, true);
    EnnCloseModel(model_id);
    EXPECT_EQ(EnnDeinitialize(), ENN_RET_SUCCESS);
}


// Create API-complex test
class ENN_GT_API_MODEL_TEST: public testing::Test {};

/**
 * @brief api test sample 1
 *        example code with model life-cycle
 *        Init > Open > get buffer information > create buffers > set buffers
 *           > commit buffers > Execute > Close > Deinit
 */

TEST_F(ENN_GT_API_MODEL_TEST, completed_test_allocate_each_memory_buffer) {
    ENN_TST_PRINT("1. Initialize \n");
    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);

    ENN_TST_PRINT("2. Open Model (test) \n");
    // EnnModelPreference test;
    EnnModelId model_id;
    const std::string filename = SAMPLE_MODEL_FILENAME;
    auto ret = EnnOpenModel(filename.c_str(), &model_id);

    // EnnModelPreference test;
    // EnnModelId model_id = 0x1001;
    // auto ret = EnnOpenModelTest(&model_id, test);
    // auto ret = EnnOpenModel(filename, &model_id);
    EXPECT_EQ(ENN_RET_SUCCESS, ret);

    EnnBuffer** in_buf;
    EnnBuffer** out_buf;
    NumberOfBuffersInfo n_buf_info;
    EnnBufferInfo tmp_buf_info;

    ENN_TST_PRINT("3. get buffer info: number of input, outputs \n");
    EXPECT_EQ(EnnGetBuffersInfo(model_id, &n_buf_info), ENN_RET_SUCCESS);

    in_buf = (EnnBufferPtr *)malloc(sizeof(EnnBufferPtr ) * n_buf_info.n_in_buf);
    out_buf = (EnnBufferPtr *)malloc(sizeof(EnnBufferPtr ) * n_buf_info.n_out_buf);

    ENN_TST_PRINT("4-1. In_buffer: get information and create buffers\n");
    for (int idx = 0; idx < n_buf_info.n_in_buf; idx ++) {
        ASSERT_EQ(EnnGetBufferInfoByIndex(model_id, ENN_DIR_IN, idx, &tmp_buf_info), ENN_RET_SUCCESS);
        EnnCreateBufferCache(tmp_buf_info.size, &(in_buf[idx]));
    }

    /* set input buffer here */
    // load buffer like: memcpy(in_buf[0].va, source, in_buf[0].size);

    ENN_TST_PRINT("4-2. Out_buffer: get information and create buffers\n");
    for (int idx = 0; idx < n_buf_info.n_out_buf; idx ++) {
        ASSERT_EQ(EnnGetBufferInfoByIndex(model_id, ENN_DIR_OUT, idx, &tmp_buf_info), ENN_RET_SUCCESS);
        EnnCreateBufferCache(tmp_buf_info.size, &(out_buf[idx]));  /* ion flag: cache enabled */
    }

    ENN_TST_PRINT("5. Set Buffer by index from getting buffer\n");
    // Setbuffer can be failed, if buffer is part of region
    for (int idx = 0; idx < n_buf_info.n_in_buf; idx ++) {
        EXPECT_EQ(EnnSetBufferByIndex(model_id, ENN_DIR_IN, idx, in_buf[idx]), ENN_RET_SUCCESS);
    }
    for (int idx = 0; idx < n_buf_info.n_out_buf; idx ++) {
        EXPECT_EQ(EnnSetBufferByIndex(model_id, ENN_DIR_OUT, idx, out_buf[idx]), ENN_RET_SUCCESS);
    }

    ENN_TST_PRINT("6. Send buffer information to Service \n");
    EXPECT_EQ(EnnBufferCommit(model_id), ENN_RET_SUCCESS);

    ENN_TST_PRINT("7. Execute Model\n");
    EXPECT_EQ(EnnExecuteModel(model_id), ENN_RET_SUCCESS);


    ENN_TST_PRINT("in_buf[idx] = va; %p, size: %d, offset: %d\n", in_buf[0]->va, in_buf[0]->size, in_buf[0]->offset);
    ENN_TST_PRINT("out_buf[idx] = va; %p, size: %d, offset: %d\n", out_buf[0]->va, out_buf[0]->size, out_buf[0]->offset);


    ENN_TST_PRINT("8. Release Buffers\n");

    for (int idx = 0; idx < n_buf_info.n_in_buf; idx++)        EnnReleaseBuffer(in_buf[idx]);
    for (int idx = 0; idx < n_buf_info.n_out_buf; idx++)        EnnReleaseBuffer(out_buf[idx]);

    free(in_buf);
    free(out_buf);

    /* check output buffers */
    // do something

    ENN_TST_PRINT("9. Close Model\n");
    EnnCloseModel(model_id);

    ENN_TST_PRINT("10. Deinitialize\n");
    EXPECT_EQ(EnnDeinitialize(), ENN_RET_SUCCESS);
}

TEST_F(ENN_GT_API_MODEL_TEST, completed_test_allocate_all_buffers) {
    // EnnModelPreference test;
    EnnModelId model_id;
    EnnBufferSet buffers;

    ENN_TST_PRINT("1. Initialize & Open (for test)\n");
    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);
    const std::string filename = SAMPLE_MODEL_FILENAME;
    auto ret = EnnOpenModel(filename.c_str(), &model_id);

    // EnnModelPreference test;
    // EnnModelId model_id = 0x1001;
    // auto ret = EnnOpenModelTest(&model_id, test);
    // auto ret = EnnOpenModel(filename, &model_id);
    EXPECT_EQ(ENN_RET_SUCCESS, ret);

    /* Commit is included */
    ENN_TST_PRINT("3. Allocate & commit All buffers for single execution\n");
    NumberOfBuffersInfo _bi;
    EnnAllocateAllBuffers(model_id, &buffers, &_bi);

    /* Execute */
    ENN_TST_PRINT("4. Execute Model \n");
    EXPECT_EQ(EnnExecuteModel(model_id), ENN_RET_SUCCESS);

    /* Release all allocated buffers */
    ENN_TST_PRINT("5. Release & Close & Deinitialize \n");
    EXPECT_EQ(EnnReleaseBuffers(buffers, (_bi.n_in_buf + _bi.n_out_buf)), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnCloseModel(model_id), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnDeinitialize(), ENN_RET_SUCCESS);
}

TEST_F(ENN_GT_API_MODEL_TEST, completed_test_allocate_all_buffers_and_set_ext_buffer) {
    /* TEST: Label "692", "693", "691", "689" is ext buffer */
    /*       Size is 2860832, 1024, 1000, 4000 each */
    // EnnModelPreference test;
    EnnModelId model_id;
    EnnBufferSet buffers;

    ENN_TST_PRINT("1. Initialize & Open (for test)\n");
    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);
    const std::string filename = SAMPLE_MODEL_FILENAME;
    auto ret = EnnOpenModel(filename.c_str(), &model_id);

    // EnnModelPreference test;
    // EnnModelId model_id = 0x1001;
    // auto ret = EnnOpenModelTest(&model_id, test);
    // auto ret = EnnOpenModel(filename, &model_id);
    EXPECT_EQ(ENN_RET_SUCCESS, ret);

    /* Commit is included */
    ENN_TST_PRINT("3. Allocate & commit All buffers for single execution\n");
    NumberOfBuffersInfo _bi;
    EnnAllocateAllBuffersWithSessionId(model_id, &buffers, &_bi, 0, false);

    EnnBufferPtr tmp_buf;
    EXPECT_EQ(EnnCreateBufferCache(2860832, &tmp_buf), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnSetBufferByLabel(model_id, "692", tmp_buf), ENN_RET_SUCCESS);

    EnnBufferCommit(model_id);

    /* Execute */
    ENN_TST_PRINT("4. Execute Model \n");
    EXPECT_EQ(EnnExecuteModel(model_id), ENN_RET_SUCCESS);

    /* Release all allocated buffers */
    ENN_TST_PRINT("5. Release & Close & Deinitialize \n");
    EXPECT_EQ(EnnReleaseBuffer(tmp_buf), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnReleaseBuffers(buffers, (_bi.n_in_buf + _bi.n_out_buf)), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnCloseModel(model_id), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnDeinitialize(), ENN_RET_SUCCESS);
}

TEST_F(ENN_GT_API_MODEL_TEST, completed_test_allocate_all_buffers_with_2_sessions) {
    // EnnModelPreference test;
    EnnModelId model_id;
    EnnBufferSet buffers[2];

    ENN_TST_PRINT("1. Initialize & Open (for test)\n");
    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);
    const std::string filename = SAMPLE_MODEL_FILENAME;
    auto ret = EnnOpenModel(filename.c_str(), &model_id);

    // EnnModelPreference test;
    // EnnModelId model_id = 0x1001;
    // auto ret = EnnOpenModelTest(&model_id, test);
    // auto ret = EnnOpenModel(filename, &model_id);
    EXPECT_EQ(ENN_RET_SUCCESS, ret);

    /* Commit is included */
    ENN_TST_PRINT("3. Allocate & commit All buffers for single execution\n");
    EnnGenerateBufferSpace(model_id, 3);
    NumberOfBuffersInfo _bi;
    EnnAllocateAllBuffers(model_id, &(buffers[0]), &_bi);
    EnnAllocateAllBuffersWithSessionId(model_id, &(buffers[1]), &_bi, 1, true);

    /* Execute */
    ENN_TST_PRINT("4. Execute Model \n");
    EXPECT_EQ(EnnExecuteModelWithSessionId(model_id, 0), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnExecuteModelWithSessionId(model_id, 1), ENN_RET_SUCCESS);
    /* Release all allocated buffers */
    ENN_TST_PRINT("5. Release & Close & Deinitialize \n");
    EXPECT_EQ(EnnReleaseBuffers(buffers[0], (_bi.n_in_buf + _bi.n_out_buf)), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnReleaseBuffers(buffers[1], (_bi.n_in_buf + _bi.n_out_buf)), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnCloseModel(model_id), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnDeinitialize(), ENN_RET_SUCCESS);
}

static uint64_t getTimeNSec() {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (uint64_t)now.tv_sec * 1000000000LL + now.tv_nsec;
}

enum class op_option {
    SYNC_200,
    THREAD_2_SYNC_100,
    ASYNC_2_100,
};

static void completed_test_allocate_all_buffers_with_models_test(op_option option) {
    int num_execution;
    if (getenv(ITERATION_N_ENV_NAME)) {
        auto n = atoi(getenv(ITERATION_N_ENV_NAME));
        num_execution = n > 0 ? n : 1;
    } else {
        num_execution = DEFAULT_ITER;
    }

    ENN_INFO_PRINT_FORCE(" # of Test Iteration: %d (%s)\n", num_execution, option == op_option::SYNC_200 ? "SYNC_200" : option == op_option::ASYNC_2_100 ? "ASYNC_2_100" : "THREAD_2_SYNC_100");

    const int MAX_MODEL = 2;


    EnnModelId model_id[MAX_MODEL];
    EnnBufferSet buffers[MAX_MODEL];

    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);

    const std::string filename = SAMPLE_MODEL_FILENAME;
    NumberOfBuffersInfo _bi;
    for (int i = 0; i < MAX_MODEL; i++) {
        EXPECT_EQ(EnnOpenModel(filename.c_str(), &(model_id[i])), ENN_RET_SUCCESS);
        EXPECT_EQ(EnnGenerateBufferSpace(model_id[i], 1), ENN_RET_SUCCESS);
        EXPECT_EQ(EnnAllocateAllBuffers(model_id[i], &(buffers[i]), &_bi), ENN_RET_SUCCESS);
    }

    uint64_t start_time, accum_time = 0;

    if (option == op_option::THREAD_2_SYNC_100) {
        start_time = getTimeNSec();
        auto f1 = std::async([&]() {
            for (int iter = 1; iter <= num_execution; ++iter) {
                EnnExecuteModel(model_id[0]);
            }
        });
        auto f2 = std::async([&]() {
            for (int iter = 1; iter <= num_execution; ++iter) {
                EnnExecuteModel(model_id[1]);
            }
        });
        f1.get();
        f2.get();
        accum_time = (getTimeNSec() - start_time);
    } else {
        start_time = getTimeNSec();
        for (int iter = 1; iter <= num_execution; ++iter) {
            if (option == op_option::SYNC_200) {
                for (int i = 0; i < MAX_MODEL; i++) {
                    EXPECT_EQ(EnnExecuteModel(model_id[i]), ENN_RET_SUCCESS);
                }
            } else {
                for (int i = 0; i < MAX_MODEL; i++)
                    EXPECT_EQ(EnnExecuteModelAsync(model_id[i]), ENN_RET_SUCCESS);
                for (int i = 0; i < MAX_MODEL; i++)
                    EXPECT_EQ(EnnExecuteModelWait(model_id[i]), ENN_RET_SUCCESS);
            }
        }
        accum_time = (getTimeNSec() - start_time);
    }
    printf("# Avarage execution time per model %zu us\n", (accum_time) / 1000);


    for (int i = 0; i < MAX_MODEL; i++) {
        EXPECT_EQ(EnnReleaseBuffers(buffers[i], (_bi.n_in_buf + _bi.n_out_buf)), ENN_RET_SUCCESS);
        EXPECT_EQ(EnnCloseModel(model_id[i]), ENN_RET_SUCCESS);
    }

    EXPECT_EQ(EnnDeinitialize(), ENN_RET_SUCCESS);
}


TEST_F(ENN_GT_API_MODEL_TEST, completed_test_allocate_all_buffers_with_2_models_sync) {
    completed_test_allocate_all_buffers_with_models_test(op_option::SYNC_200);
}

TEST_F(ENN_GT_API_MODEL_TEST, completed_test_allocate_all_buffers_with_2_models_async) {
    completed_test_allocate_all_buffers_with_models_test(op_option::ASYNC_2_100);
}

TEST_F(ENN_GT_API_MODEL_TEST, completed_test_allocate_all_buffers_with_2_models_2_thread) {
    completed_test_allocate_all_buffers_with_models_test(op_option::THREAD_2_SYNC_100);
}

static void run_sync_with_n_models_n_threads(const std::string &model_file, const std::string input_file,
                                             const std::string golden_file, int count_n, bool try_to_match, bool is_async = false) {
    int num_execution;
    if (getenv(ITERATION_N_ENV_NAME)) {
        auto n = atoi(getenv(ITERATION_N_ENV_NAME));
        num_execution = n > 0 ? n : 1;
    } else {
        num_execution = DEFAULT_ITER;
    }

    ENN_INFO_PRINT_FORCE(" # of Test Iteration: %d (#Models : %d, #Threads : %d), %s\n", num_execution, count_n, count_n, (is_async ? "ASYNC" : "SYNC"));
    printf("*****************************************************************\n");
    printf("  The number of Model and Thread : %d. (%s)\n", count_n, (is_async ? "ASYNC" : "SYNC"));
    printf("   Note) Each thread repeats a execution %d times with each model.\n", num_execution);
    printf("*****************************************************************\n\n");

    std::vector<EnnModelId> model_id_vec(count_n);
    std::vector<EnnBufferSet> buffer_set_vec(count_n);
    // uint32_t in, out, e;

    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);

    NumberOfBuffersInfo _bi;
    uint32_t file_size;
    for (int i = 0; i < count_n; i++) {
        EXPECT_EQ(EnnOpenModel(model_file.c_str(), &(model_id_vec[i])), ENN_RET_SUCCESS);
        EXPECT_EQ(EnnGenerateBufferSpace(model_id_vec[i], 1), ENN_RET_SUCCESS);
        EXPECT_EQ(EnnAllocateAllBuffers(model_id_vec[i], &(buffer_set_vec[i]), &_bi), ENN_RET_SUCCESS);
        EXPECT_EQ(ENN_RET_SUCCESS,
            enn::util::import_file_to_mem(input_file.c_str(), reinterpret_cast<char **>(&(buffer_set_vec[i][0]->va)), &file_size));
        EXPECT_EQ(static_cast<uint32_t>(file_size), buffer_set_vec[i][0]->size);
    }

    std::vector<std::future<void>> future_list;

    uint64_t start_time, accum_time = 0;
    start_time = getTimeNSec();
    for(int i = 0; i < count_n; i++) {
        future_list.push_back(
            enn::util::RunAsync([&, i]() {
                for (int iter = 1; iter <= num_execution; ++iter) {
                    {
                        if (is_async) {
                            EnnExecuteModelAsync(model_id_vec[i]);
                            EnnExecuteModelWait(model_id_vec[i]);
                        } else {
                            EnnExecuteModel(model_id_vec[i]);
                        }
                    }
                    if (try_to_match) {
                        // Verify the output with golden file.
                        uint32_t file_size;
                        char *golden_out_heap_buf = nullptr;
                        EXPECT_EQ(ENN_RET_SUCCESS,
                                enn::util::import_file_to_mem(golden_file.c_str(), &golden_out_heap_buf,
                                                                &file_size));  // internally create buffer to golden_out_heap_buf (this utility
                                                                               // returns raw pointer, not smart pointer)
                        ENN_TST_PRINT("golden out file size: %d, ptr: %p\n", file_size, golden_out_heap_buf);
                        EXPECT_EQ(0, enn::util::CompareBuffersWithThreshold<float>(golden_out_heap_buf,
                                                                                   reinterpret_cast<char *>(buffer_set_vec[i][1]->va),
                                                                                   buffer_set_vec[i][1]->size,
                                                                                   nullptr,
                                                                                   enn::util::BUFFER_COMPARE_THRESHOLD));
                    }
                }
            })
        );
    }

    for(auto& future : future_list) {
        future.get();
    }

    accum_time = (getTimeNSec() - start_time);

    int total_execution = num_execution * count_n;
    double duration_us = accum_time / 1000;
    printf("=========================================================\n");
    printf(" * End-to-end duration for %d-execution : %.0lf us\n", total_execution, duration_us);
    printf(" * FPS : %.2lf\n", 1 / ((duration_us / total_execution) / (1000 * 1000)));
    printf("=========================================================\n");

    for (int i = 0; i < count_n; i++) {
        EXPECT_EQ(EnnReleaseBuffers(buffer_set_vec[i], (_bi.n_in_buf + _bi.n_out_buf)), ENN_RET_SUCCESS);
        EXPECT_EQ(EnnCloseModel(model_id_vec[i]), ENN_RET_SUCCESS);
    }

    EXPECT_EQ(EnnDeinitialize(), ENN_RET_SUCCESS);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////// Test with N Models & N Threads with trying to match with golden ///////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

TEST_F(ENN_GT_API_MODEL_TEST, pamir_mobilenet_edge_tpu_1_model_1_thread_sync_with_match) {
    const std::string model_file = PAMIR::NPU::EdgeTPU::NNC;
    const std::string input_file = PAMIR::NPU::EdgeTPU::INPUT;
    const std::string golden_file = PAMIR::NPU::EdgeTPU::GOLDEN;
    run_sync_with_n_models_n_threads(model_file, input_file, golden_file, 1, true);
}

TEST_F(ENN_GT_API_MODEL_TEST, pamir_mobilenet_edge_tpu_1_model_1_thread_async_with_match) {
    const std::string model_file = PAMIR::NPU::EdgeTPU::NNC;
    const std::string input_file = PAMIR::NPU::EdgeTPU::INPUT;
    const std::string golden_file = PAMIR::NPU::EdgeTPU::GOLDEN;
    run_sync_with_n_models_n_threads(model_file, input_file, golden_file, 1, true, true);
}

TEST_F(ENN_GT_API_MODEL_TEST, pamir_mobilenet_edge_tpu_2_models_2_threads_sync_with_match) {
    const std::string model_file = PAMIR::NPU::EdgeTPU::NNC;
    const std::string input_file = PAMIR::NPU::EdgeTPU::INPUT;
    const std::string golden_file = PAMIR::NPU::EdgeTPU::GOLDEN;
    run_sync_with_n_models_n_threads(model_file, input_file, golden_file, 2, true);
}

TEST_F(ENN_GT_API_MODEL_TEST, pamir_mobilenet_edge_tpu_4_models_4_threads_sync_with_match) {
    const std::string model_file = PAMIR::NPU::EdgeTPU::NNC;
    const std::string input_file = PAMIR::NPU::EdgeTPU::INPUT;
    const std::string golden_file = PAMIR::NPU::EdgeTPU::GOLDEN;
    run_sync_with_n_models_n_threads(model_file, input_file, golden_file, 4, true);
}

TEST_F(ENN_GT_API_MODEL_TEST, pamir_mobilenet_edge_tpu_6_models_6_threads_sync_with_match) {
    const std::string model_file = PAMIR::NPU::EdgeTPU::NNC;
    const std::string input_file = PAMIR::NPU::EdgeTPU::INPUT;
    const std::string golden_file = PAMIR::NPU::EdgeTPU::GOLDEN;
    run_sync_with_n_models_n_threads(model_file, input_file, golden_file, 6, true);
}

TEST_F(ENN_GT_API_MODEL_TEST, pamir_mobilenet_edge_tpu_8_models_8_threads_sync_with_match) {
    const std::string model_file = PAMIR::NPU::EdgeTPU::NNC;
    const std::string input_file = PAMIR::NPU::EdgeTPU::INPUT;
    const std::string golden_file = PAMIR::NPU::EdgeTPU::GOLDEN;
    run_sync_with_n_models_n_threads(model_file, input_file, golden_file, 8, true);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////// Test with N Models & N Threads without trying to match with golden //////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////

TEST_F(ENN_GT_API_MODEL_TEST, pamir_mobilenet_edge_tpu_1_model_1_thread_sync_without_match) {
    const std::string model_file = PAMIR::NPU::EdgeTPU::NNC;
    const std::string input_file = PAMIR::NPU::EdgeTPU::INPUT;
    const std::string golden_file = PAMIR::NPU::EdgeTPU::GOLDEN;
    run_sync_with_n_models_n_threads(model_file, input_file, golden_file, 1, false);
}

TEST_F(ENN_GT_API_MODEL_TEST, pamir_mobilenet_edge_tpu_2_models_2_threads_sync_without_match) {
    const std::string model_file = PAMIR::NPU::EdgeTPU::NNC;
    const std::string input_file = PAMIR::NPU::EdgeTPU::INPUT;
    const std::string golden_file = PAMIR::NPU::EdgeTPU::GOLDEN;
    run_sync_with_n_models_n_threads(model_file, input_file, golden_file, 2, false);
}

TEST_F(ENN_GT_API_MODEL_TEST, pamir_mobilenet_edge_tpu_4_models_4_threads_sync_without_match) {
    const std::string model_file = PAMIR::NPU::EdgeTPU::NNC;
    const std::string input_file = PAMIR::NPU::EdgeTPU::INPUT;
    const std::string golden_file = PAMIR::NPU::EdgeTPU::GOLDEN;
    run_sync_with_n_models_n_threads(model_file, input_file, golden_file, 4, false);
}

TEST_F(ENN_GT_API_MODEL_TEST, pamir_mobilenet_edge_tpu_6_models_6_threads_sync_without_match) {
    const std::string model_file = PAMIR::NPU::EdgeTPU::NNC;
    const std::string input_file = PAMIR::NPU::EdgeTPU::INPUT;
    const std::string golden_file = PAMIR::NPU::EdgeTPU::GOLDEN;
    run_sync_with_n_models_n_threads(model_file, input_file, golden_file, 6, false);
}

TEST_F(ENN_GT_API_MODEL_TEST, pamir_mobilenet_edge_tpu_8_models_8_threads_sync_without_match) {
    const std::string model_file = PAMIR::NPU::EdgeTPU::NNC;
    const std::string input_file = PAMIR::NPU::EdgeTPU::INPUT;
    const std::string golden_file = PAMIR::NPU::EdgeTPU::GOLDEN;
    run_sync_with_n_models_n_threads(model_file, input_file, golden_file, 8, false);
}

TEST_F(ENN_GT_API_MODEL_TEST, completed_test_allocate_3_sessions_duplicated_commit_test) {
    // EnnModelPreference test;
    EnnModelId model_id;
    EnnBufferSet buffers[3];

    ENN_TST_PRINT("1. Initialize & Open (for test)\n");
    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);
    const std::string filename = SAMPLE_MODEL_FILENAME;
    auto ret = EnnOpenModel(filename.c_str(), &model_id);

    // EnnModelPreference test;
    // EnnModelId model_id = 0x1001;
    // auto ret = EnnOpenModelTest(&model_id, test);
    // auto ret = EnnOpenModel(filename, &model_id);
    EXPECT_EQ(ENN_RET_SUCCESS, ret);

    /* Commit is included */
    ENN_TST_PRINT("3. Allocate & commit All buffers for single execution\n");
    EnnGenerateBufferSpace(model_id, 3);
    NumberOfBuffersInfo _bi[3];
    EXPECT_EQ(EnnAllocateAllBuffers(model_id, &(buffers[0]), &(_bi[0])), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnAllocateAllBuffersWithSessionId(model_id, &(buffers[1]), &(_bi[1]), 1, true), ENN_RET_SUCCESS);
    EXPECT_NE(EnnAllocateAllBuffersWithSessionId(model_id, &(buffers[2]), &(_bi[2]), 1, true), ENN_RET_SUCCESS);  // duplicated commit

    /* Release all allocated buffers */
    ENN_TST_PRINT("5. Release & Close & Deinitialize \n");
    EXPECT_EQ(EnnReleaseBuffers(buffers[0], (_bi[0].n_in_buf + _bi[0].n_out_buf)), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnReleaseBuffers(buffers[1], (_bi[1].n_in_buf + _bi[1].n_out_buf)), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnReleaseBuffers(buffers[2], (_bi[2].n_in_buf + _bi[2].n_out_buf)), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnCloseModel(model_id), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnDeinitialize(), ENN_RET_SUCCESS);
}

TEST_F(ENN_GT_API_MODEL_TEST, completed_test_allocate_all_buffers_and_swap_1buffer) {
    ENN_TST_PRINT("1. Initialize \n");
    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);

    ENN_TST_PRINT("2. Open Model (test) \n");
    EnnModelId model_id;
    const std::string filename = SAMPLE_MODEL_FILENAME;
    auto ret = EnnOpenModel(filename.c_str(), &model_id);

    // EnnModelPreference test;
    // EnnModelId model_id = 0x1001;
    // auto ret = EnnOpenModelTest(&model_id, test);
    // auto ret = EnnOpenModel(filename, &model_id);
    EXPECT_EQ(ENN_RET_SUCCESS, ret);
    EnnBufferInfo tmp_buf_info;

    EnnBufferSet buffers;
    EnnBufferPtr buffer_to_change;
    NumberOfBuffersInfo _bi;

    ENN_TST_PRINT("3-1. Allocate All buffers for execution \n");
    EnnAllocateAllBuffersWithoutCommit(model_id, &buffers, &_bi);
    // buffers are arranged by "IN[0], IN[1],.. IN[i-1], OUT[0], OUT[1]... OUT[o-1], EXT[0]... EXT[e-1]"

    ENN_TST_PRINT("3-2. Get Buffer info and create buffer to swap\n");
    /* Get buf info [OUT, 0] of model */
    ASSERT_EQ(EnnGetBufferInfoByIndex(model_id, ENN_DIR_OUT, 0, &tmp_buf_info), ENN_RET_SUCCESS);
    EnnCreateBufferCache(tmp_buf_info.size, &buffer_to_change);

    ENN_TST_PRINT("3-3. Swap buffer with buffer[OUT, 0] \n");
    auto tmp_buffer = buffers[_bi.n_in_buf + 0];
    buffers[_bi.n_in_buf + 0] = buffer_to_change;

    ENN_TST_PRINT("3-4. Set buffers to buffer pool \n");
    EXPECT_EQ(EnnSetBuffers(model_id, buffers, (_bi.n_in_buf + _bi.n_out_buf)), ENN_RET_SUCCESS);

    ENN_TST_PRINT("6. Send buffer information to Service \n");
    EXPECT_EQ(EnnBufferCommit(model_id), ENN_RET_SUCCESS);  // buffer commit to apply swapped buffer

    /* Execute */
    ENN_TST_PRINT("7. Execute Model\n");
    EXPECT_EQ(EnnExecuteModel(model_id), ENN_RET_SUCCESS);

    ENN_TST_PRINT("8. Release Buffers\n");
    EXPECT_EQ(EnnReleaseBuffers(buffers, (_bi.n_in_buf + _bi.n_out_buf)), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnReleaseBuffer(tmp_buffer), ENN_RET_SUCCESS);

    ENN_TST_PRINT("9. Close Model\n");
    EXPECT_EQ(EnnCloseModel(model_id), ENN_RET_SUCCESS);

    ENN_TST_PRINT("10. Deinitialize\n");
    EXPECT_EQ(EnnDeinitialize(), ENN_RET_SUCCESS);
}


TEST_F(ENN_GT_API_MODEL_TEST, completed_test_EnnSetBuffers_and_commit) {
    const std::string filename = SAMPLE_MODEL_FILENAME;

    ENN_TST_PRINT("1. Initialize \n");
    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);

    EnnModelId model_id;
    auto ret = EnnOpenModel(filename.c_str(), &model_id);

    EXPECT_EQ(ENN_RET_SUCCESS, ret);
    ENN_TST_PRINT("2. Open Model (test) : ModelID(0x%lX)\n", model_id);

    uint32_t n_allocated = 0;
    NumberOfBuffersInfo n_buf_info;
    EnnBufferInfo tmp_buf_info;

    ENN_TST_PRINT("3. get buffer info: number of input, outputs \n");
    EXPECT_EQ(EnnGetBuffersInfo(model_id, &n_buf_info), ENN_RET_SUCCESS);

    ENN_TST_PRINT("4. Buffers: Manually generates ext-buffers\n");
    auto bufs = (EnnBufferPtr *)malloc(sizeof(EnnBufferPtr ) * (n_buf_info.n_in_buf + n_buf_info.n_out_buf));
    ASSERT_NE(bufs, nullptr);

    for (int idx = 0; idx < n_buf_info.n_in_buf; idx++) {
        EnnGetBufferInfoByIndex(model_id, ENN_DIR_IN, idx, &tmp_buf_info);
        EnnCreateBufferCache(tmp_buf_info.size, &(bufs[n_allocated++]));
    }
    for (int idx = 0; idx < n_buf_info.n_out_buf; idx++) {
        EnnGetBufferInfoByIndex(model_id, ENN_DIR_OUT, idx, &tmp_buf_info);
        EnnCreateBufferCache(tmp_buf_info.size, &(bufs[n_allocated++]));

    }

    ENN_TST_PRINT("5. Set buffers using EnnSetBuffers, this function doesn't commit\n");
    EXPECT_EQ(EnnSetBuffers(model_id, bufs, (n_buf_info.n_in_buf + n_buf_info.n_out_buf)), ENN_RET_SUCCESS);

    ENN_TST_PRINT("6. Send buffer information to Service \n");
    EXPECT_EQ(EnnBufferCommit(model_id), ENN_RET_SUCCESS);  // buffer commit to apply swapped buffer

    ENN_TST_PRINT("7. Execute Model\n");
    EXPECT_EQ(EnnExecuteModel(model_id), ENN_RET_SUCCESS);

    ENN_TST_PRINT("8. Release Buffers\n");
    EXPECT_EQ(EnnReleaseBuffers(bufs, (n_buf_info.n_in_buf + n_buf_info.n_out_buf)), ENN_RET_SUCCESS);

    /* check output buffers */
    // do something

    ENN_TST_PRINT("9. Close Model\n");
    EXPECT_EQ(EnnCloseModel(model_id), ENN_RET_SUCCESS);

    ENN_TST_PRINT("10. Deinitialize\n");
    EXPECT_EQ(EnnDeinitialize(), ENN_RET_SUCCESS);

    free(bufs);
}

class ENN_GT_API_APPLICATION_TEST: public testing::Test {};

TEST_F(ENN_GT_API_APPLICATION_TEST, model_get_meta_test) {
    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);
    char output[ENN_INFO_GRAPH_STR_LENGTH_MAX];
    EXPECT_EQ(EnnGetMetaInfo(ENN_META_VERSION_FRAMEWORK, output), ENN_RET_SUCCESS);
    ENN_TST_PRINT("ENN_META_VERSION_FRAMEWORK: %s\n", output);

    EXPECT_EQ(EnnGetMetaInfo(ENN_META_VERSION_COMMIT, output), ENN_RET_SUCCESS);
    ENN_TST_PRINT("ENN_META_VERSION_COMMIT: %s\n", output);

    EXPECT_EQ(EnnGetMetaInfo(ENN_META_VERSION_MODEL, output), ENN_RET_SUCCESS);
    ENN_TST_PRINT("ENN_META_VERSION_MODEL: %s\n", output);

    EXPECT_EQ(EnnGetMetaInfo(ENN_META_VERSION_COMPILER, output), ENN_RET_SUCCESS);
    ENN_TST_PRINT("ENN_META_VERSION_COMPILER: %s\n", output);

    EXPECT_EQ(EnnGetMetaInfo(ENN_META_VERSION_DEVICEDRIVER, output), ENN_RET_SUCCESS);
    ENN_TST_PRINT("ENN_META_VERSION_DEVICEDRIVER: %s\n", output);

    EXPECT_EQ(EnnGetMetaInfo(ENN_META_DESCRPTION_MODEL, output), ENN_RET_SUCCESS);
    ENN_TST_PRINT("ENN_META_DESCRPTION_MODEL: %s\n", output);

    EXPECT_EQ(EnnDeinitialize(), ENN_RET_SUCCESS);
}


TEST_F(ENN_GT_API_APPLICATION_TEST, model_test_npu_inception_v3) {
#ifdef SCHEMA_NNC_V1
    const std::string filename = OLYMPUS::NPU::IV3::NNC;
#else
    const std::string filename = PAMIR::NPU::IV3::NNC;
#endif

    ENN_TST_PRINT("1. Initialize & Open \n");
    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);

    EnnModelId model_id;
    //  auto ret = EnnOpenModelTest(&model_id, test);
    auto ret = EnnOpenModel(filename.c_str(), &model_id);
    EXPECT_EQ(ENN_RET_SUCCESS, ret);

    /* Commit is included */
    EnnBufferSet buffers;

    ENN_TST_PRINT("3. Allocate & commit All buffers for single execution\n");
    NumberOfBuffersInfo _bi;
    EnnAllocateAllBuffers(model_id, &buffers, &_bi);

    /* Execute */
    ENN_TST_PRINT("4. Execute Model \n");
    EXPECT_EQ(EnnExecuteModel(model_id), ENN_RET_SUCCESS);

    /* Release all allocated buffers */
    ENN_TST_PRINT("5. Release & Close & Deinitialize \n");
    EXPECT_EQ(EnnReleaseBuffers(buffers, (_bi.n_in_buf + _bi.n_out_buf)), ENN_RET_SUCCESS);

    EXPECT_EQ(EnnCloseModel(model_id), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnDeinitialize(), ENN_RET_SUCCESS);
}

#ifdef SCHEMA_NNC_V1
TEST_F(ENN_GT_API_APPLICATION_TEST, model_test_olympus_npu_inception_v3_golden_match) {
    const std::string filename = OLYMPUS::NPU::IV3::NNC;
    const std::string input_file = OLYMPUS::NPU::IV3::INPUT;
    const std::string golden_file = OLYMPUS::NPU::IV3::GOLDEN;

    EXPECT_EQ(test_golden_match_1_1<float>(filename, input_file, golden_file, enn::util::BUFFER_COMPARE_THRESHOLD), true);
}
#else
TEST_F(ENN_GT_API_APPLICATION_TEST, model_test_pamir_npu_repeat_iv3_for_60s) {
    const std::string filename = PAMIR::NPU::IV3::NNC;
    const std::string input_file = PAMIR::NPU::IV3::INPUT;
    const std::string golden_file = PAMIR::NPU::IV3::GOLDEN;

    EXPECT_EQ(test_repeat_iv3_for_60sec<float>(filename, input_file, golden_file, enn::util::BUFFER_COMPARE_THRESHOLD, false, INITIAL_ITER), true);
}

TEST_F(ENN_GT_API_APPLICATION_TEST, model_test_pamir_npu_inception_v3_golden_match) {
    const std::string filename = PAMIR::NPU::IV3::NNC;
    const std::string input_file = PAMIR::NPU::IV3::INPUT;
    const std::string golden_file = PAMIR::NPU::IV3::GOLDEN;

    EXPECT_EQ(test_golden_match_1_1<float>(filename, input_file, golden_file, enn::util::BUFFER_COMPARE_THRESHOLD), true);
}

TEST_F(ENN_GT_API_APPLICATION_TEST, model_test_pamir_npu_inception_v3_golden_match_from_memory) {
    const std::string filename = PAMIR::NPU::IV3::NNC;
    const std::string input_file = PAMIR::NPU::IV3::INPUT;
    const std::string golden_file = PAMIR::NPU::IV3::GOLDEN;

    EXPECT_EQ(test_golden_match_1_1<float>(filename, input_file, golden_file, enn::util::BUFFER_COMPARE_THRESHOLD, INITIAL_ITER, true), true);
}

TEST_F(ENN_GT_API_APPLICATION_TEST, model_test_pamir_dsp_inception_v3_golden_match) {
    const std::string filename = PAMIR::DSP::IV3::NNC;
    const std::string input_file = PAMIR::DSP::IV3::INPUT;
    const std::string golden_file = PAMIR::DSP::IV3::GOLDEN;

    EXPECT_EQ(test_golden_match_1_1<float>(filename, input_file, golden_file, enn::util::BUFFER_COMPARE_THRESHOLD), true);
}

TEST_F(ENN_GT_API_APPLICATION_TEST, model_test_pamir_npu_deeplab_v3_golden_match) {
    const std::string filename = PAMIR::NPU::DLV3::NNC;
    const std::string input_file = PAMIR::NPU::DLV3::INPUT;
    const std::string golden_file = PAMIR::NPU::DLV3::GOLDEN;

    EXPECT_EQ(test_golden_match_1_1<int8_t>(filename, input_file, golden_file, enn::util::BUFFER_COMPARE_THRESHOLD), true);
}

TEST_F(ENN_GT_API_APPLICATION_TEST, model_test_pamir_npu_deeplab_v3_golden_match_from_memory) {
    const std::string filename = PAMIR::NPU::DLV3::NNC;
    const std::string input_file = PAMIR::NPU::DLV3::INPUT;
    const std::string golden_file = PAMIR::NPU::DLV3::GOLDEN;

    EXPECT_EQ(test_golden_match_1_1<int8_t>(filename, input_file, golden_file, enn::util::BUFFER_COMPARE_THRESHOLD, INITIAL_ITER, true), true);
}

TEST_F(ENN_GT_API_APPLICATION_TEST, model_test_pamir_npu_mobilenet_edge_tpu_golden_match) {
    const std::string filename = PAMIR::NPU::EdgeTPU::NNC;
    const std::string input_file = PAMIR::NPU::EdgeTPU::INPUT;
    const std::string golden_file = PAMIR::NPU::EdgeTPU::GOLDEN;

    EXPECT_EQ(test_golden_match_1_1<float>(filename, input_file, golden_file, enn::util::BUFFER_COMPARE_THRESHOLD), true);
}

#if 0  // Required, ToDo(empire.jung, 8/31): Legacy SSD, remove after deciding to use priorbox
TEST_F(ENN_GT_API_APPLICATION_TEST, model_test_pamir_npu_mobiledet_SSD_O0_hw_cfu_golden_match) {
    const std::string filename = PAMIR::NPU::SSD::LEGACY::NNC_O0_HW_CFU;
    const std::string input_file = PAMIR::NPU::SSD::INPUT;
    const std::string golden_file = PAMIR::NPU::SSD::GOLDEN;

    EXPECT_EQ(test_golden_match_1_1<float>(filename, input_file, golden_file, enn::util::BUFFER_COMPARE_THRESHOLD, true), true);
}

TEST_F(ENN_GT_API_APPLICATION_TEST, model_test_pamir_npu_mobiledet_SSD_O1_hw_cfu_golden_match) {
    const std::string filename = PAMIR::NPU::SSD::LEGACY::NNC_O1_HW_CFU;
    const std::string input_file = PAMIR::NPU::SSD::INPUT;
    const std::string golden_file = PAMIR::NPU::SSD::GOLDEN;

    EXPECT_EQ(test_golden_match_1_1<float>(filename, input_file, golden_file, enn::util::BUFFER_COMPARE_THRESHOLD, true), true);
}
#else
TEST_F(ENN_GT_API_APPLICATION_TEST, model_test_pamir_npu_mobiledet_SSD_golden_match) {
    const std::string filename = PAMIR::NPU::SSD::NNC;
    const std::string input_file = PAMIR::NPU::SSD::INPUT;
    const std::string golden_file = PAMIR::NPU::SSD::GOLDEN;

    EXPECT_EQ(test_golden_match_1_1<float>(filename, input_file, golden_file, enn::util::BUFFER_COMPARE_THRESHOLD, true), true);
}
#endif

TEST_F(ENN_GT_API_APPLICATION_TEST, model_test_pamir_npu_OD_VGA_golden_match) {
    const std::string filename = PAMIR::NPU::OD::VGA::NNC;
    const std::string input_file = PAMIR::NPU::OD::VGA::INPUT;
    const std::vector<std::string> golden_files = PAMIR::NPU::OD::VGA::GOLDENS;

    EXPECT_EQ(test_golden_match<float>(filename, input_file, golden_files, enn::util::BUFFER_COMPARE_THRESHOLD), true);
}

TEST_F(ENN_GT_API_APPLICATION_TEST, model_test_pamir_npu_OD_QVGA_golden_match) {
    const std::string filename = PAMIR::NPU::OD::QVGA::NNC;
    const std::string input_file = PAMIR::NPU::OD::QVGA::INPUT;
    const std::vector<std::string> golden_files = PAMIR::NPU::OD::QVGA::GOLDENS;

    EXPECT_EQ(test_golden_match<float>(filename, input_file, golden_files, enn::util::BUFFER_COMPARE_THRESHOLD), true);
}

// TODO(xin.lu): uncomment it when INV supported by GPU UD
TEST_F(ENN_GT_API_APPLICATION_TEST, model_test_pamir_gpu_inception_v3_golden_match) {
    const std::string filename = PAMIR::GPU::IV3::NNC;
    const std::string input_file = PAMIR::GPU::IV3::INPUT;
    const std::string golden_file = PAMIR::GPU::IV3::GOLDEN;

    EXPECT_EQ(test_golden_match_1_1<float>(filename, input_file, golden_file, enn::util::GPU_INCEPTIONV3_FP16_THRESHOLD),
              true);
}

#endif

class ENN_GT_API_APPLICATION_ITERATION: public ::testing::TestWithParam<int> {
public:
    ENN_GT_API_APPLICATION_ITERATION() {
        std::cout << " # Total Iteration: " << GetParam() << std::endl;
    }
};

TEST_P(ENN_GT_API_APPLICATION_ITERATION, npu_inception_v3_golden_match) {
#ifdef SCHEMA_NNC_V1
    const std::string filename = OLYMPUS::NPU::IV3::NNC;
    const std::string input_file = OLYMPUS::NPU::IV3::INPUT;
    const std::string golden_file = OLYMPUS::NPU::IV3::GOLDEN;
#else
    const std::string filename = PAMIR::NPU::IV3::NNC;
    const std::string input_file = PAMIR::NPU::IV3::INPUT;
    const std::string golden_file = PAMIR::NPU::IV3::GOLDEN;
#endif

    EXPECT_EQ(test_golden_match_1_1<float>(filename, input_file, golden_file, enn::util::BUFFER_COMPARE_THRESHOLD, GetParam()), true);
}

INSTANTIATE_TEST_SUITE_P(Iteration, ENN_GT_API_APPLICATION_ITERATION, testing::Values(1, 10, 100));

/* Create test_api- memory */
class ENN_GT_API_MEMORY_TEST: public testing::Test {};

/* Test if user doesn't call initialize before calling ennCreateBuffer */
TEST_F(ENN_GT_API_MEMORY_TEST, memory_allocation_test_without_context_init) {
    EnnBufferPtr tst;
    EXPECT_NE(ENN_RET_SUCCESS, EnnCreateBuffer(100, 0, &tst));
}

/* Test if user tries to allocate 2 buffers but free only 1 buffer before deinitialize */
TEST_F(ENN_GT_API_MEMORY_TEST, memory_auto_release_in_deinit_test) {
    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);

    EnnBufferPtr test_mem1, test_mem2;
    EnnCreateBuffer(100, 0, &test_mem1);
    EnnCreateBuffer(200, 0, &test_mem2);
    EnnReleaseBuffer(test_mem1);

    // check if test_mem2 is deleted in deinitialize()
    EXPECT_EQ(EnnDeinitialize(), ENN_RET_SUCCESS);

    // check test_mem2 is already released
    EXPECT_NE(EnnReleaseBuffer(test_mem2), ENN_RET_SUCCESS);
}

/* test if user modify parameter of ofi_memory */
TEST_F(ENN_GT_API_MEMORY_TEST, memory_check_test_if_memory_element_is_modified) {
    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);

    EnnBufferPtr test_mem1, test_mem2;
    EnnCreateBuffer(100, 0, &test_mem1);
    EnnCreateBuffer(200, 0, &test_mem2);
    test_mem2->offset = 10;  // change test_mem2. (not allowed)

    EnnReleaseBuffer(test_mem1);
    EXPECT_NE(EnnReleaseBuffer(test_mem2), ENN_RET_SUCCESS);  // error, i guess.
    EXPECT_EQ(EnnDeinitialize(), ENN_RET_SUCCESS);
}

/* test if user modify parameter of ofi_memory */
TEST_F(ENN_GT_API_MEMORY_TEST, import_memory_test) {
    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);
    EnnBufferPtr test_mem1, test_mem2;
    EnnCreateBuffer(100, 0, &test_mem1);

    /* hack: import fd from core structure */
    int fd = (reinterpret_cast<uint32_t *>(&(test_mem1->offset)))[3];
    // std::cout << "hacked fd: " << fd << std::endl;

    EXPECT_EQ(EnnCreateBufferFromFd(fd, 100, &test_mem2), ENN_RET_SUCCESS);
    char *t1 = reinterpret_cast<char *>(test_mem1->va);
    char *t2 = reinterpret_cast<char *>(test_mem2->va);

    for (int i = 0; i < 100; i++) t1[i] = i * 2;
    for (int i = 0; i < 100; i++) EXPECT_EQ(t2[i], i * 2);  // compare with import memory

    EXPECT_EQ(0, EnnReleaseBuffer(test_mem2));
    EXPECT_EQ(0, EnnReleaseBuffer(test_mem1));

    EXPECT_EQ(EnnDeinitialize(), ENN_RET_SUCCESS);
}

TEST_F(ENN_GT_API_MEMORY_TEST, import_partial_memory_test) {
    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);

    const int offset = 30;
    EnnBufferPtr test_mem1, test_mem2;

    EnnCreateBuffer(100, 0, &test_mem1);
    /* hack: import fd from core structure */
    int fd = (reinterpret_cast<uint32_t *>(&(test_mem1->offset)))[3];

    EXPECT_EQ(EnnCreateBufferFromFdWithOffset(fd, offset, offset, &test_mem2), ENN_RET_SUCCESS);
    char *t1 = reinterpret_cast<char *>(test_mem1->va);
    char *t2 = reinterpret_cast<char *>(test_mem2->va);

    for (int i = 0; i < 100; i++) t1[i] = i * 2;
    for (int i = 0; i < offset; i++) EXPECT_EQ(t1[i + offset], t2[i]);

    EXPECT_EQ(0, EnnReleaseBuffer(test_mem2));
    EXPECT_EQ(0, EnnReleaseBuffer(test_mem1));

    EXPECT_EQ(EnnDeinitialize(), ENN_RET_SUCCESS);
}

/* Create test_api- memory */
class ENN_GT_API_PERFERENCE_TEST: public testing::Test {};

TEST_F(ENN_GT_API_PERFERENCE_TEST, import_partial_memory_test) {
    uint32_t default_pref_mode, default_preset_id;
    uint32_t perf_mode, preset_id;

    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);

    EnnGetPreferencePerfMode(&default_pref_mode);
    EnnGetPreferencePresetId(&default_preset_id);

    EXPECT_EQ(EnnSetPreferencePresetId(0xFF), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnSetPreferencePerfMode(0xFF), ENN_RET_SUCCESS);

    EnnGetPreferencePresetId(&preset_id);
    EnnGetPreferencePerfMode(&perf_mode);

    EXPECT_EQ(preset_id, 0xFF);
    EXPECT_EQ(perf_mode, 0xFF);

    EXPECT_EQ(EnnResetPreferenceAsDefault(), ENN_RET_SUCCESS);

    EXPECT_EQ(EnnGetPreferencePresetId(&preset_id), ENN_RET_SUCCESS);
    EXPECT_EQ(EnnGetPreferencePerfMode(&perf_mode), ENN_RET_SUCCESS);

    EXPECT_EQ(preset_id, default_preset_id);
    EXPECT_EQ(perf_mode, default_pref_mode);

    EXPECT_EQ(EnnDeinitialize(), ENN_RET_SUCCESS);
}

class ENN_GT_API_CUSTOM_PATH_TEST: public testing::Test {};

TEST_F(ENN_GT_API_CUSTOM_PATH_TEST, test_dsp_get_session_id_api) {
    ENN_TST_PRINT("1. Initialize \n");

    int32_t dsp_session_id;
    uint64_t model_id;

    // Failed that context is not initialized
    EXPECT_EQ(EnnInitialize(), ENN_RET_SUCCESS);

    const std::string filename = PAMIR::DSP::CGO::Gaussian3x3;
    auto ret = EnnOpenModel(filename.c_str(), &model_id);
    EXPECT_NE(EnnDspGetSessionId(model_id, &dsp_session_id), -1);  // get something, not failed

    ENN_TST_PRINT("Received: %d (ret : %d)\n", dsp_session_id, ret);
}


}  // namespace internal
}  // namespace test
}  // namespace enn
