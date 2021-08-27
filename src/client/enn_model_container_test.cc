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
 * @brief test to verity enn_context_manager.hpp
 * @file enn_gtest_internal_unittest_temp.cc
 * @author Hoon Choi
 * @date 2021-03-12
 */

#include "client/enn_api-public.h"
#include "gtest/gtest.h"

#include "test/internal/enn_test_driver_model_container.h"
#include "client/enn_model_container.hpp"

#include <tuple>
#include <string>
#include <stdio.h>

#ifndef __ANDROID__
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}
#endif

/**
 *    NOTE(hoon98.choi) : How to USE container manager?
 *
 *    container.SetSessionData(sessBufInfo);   // LoadModel()로 부터 return 받은 inference Buffer info 를 저장
 *    auto ret = container.GenerateInferenceData(1, 3);  // Model ID 1에 inference data 를 3 set 만듬 ( Execution에 쓰일 수 있는 region vector 가 3개 set 만들어짐. )
 *    auto ret1 = container.SetInferenceData(1, 0, std::string("buffer0"), mem_object.get());  // 유저가 만든 mem_object를 buffer0 가 가리키는 region위치 - inference [0] 에 저장 
 *    auto ret2 = container.SetInferenceData(1, 1, std::tuple<DirType, int32_t>(ENN_BUF_DIR_IN, 0), mem_object.get());  // IN[0] 가 가리키는 region위치 - inference [1] 에 저장
 *    auto ret5 = container.VerifyInferenceData(1, 0);                           // ModelID 1, inference ID[0]이  Execution할 수 있는 상태인지 verification
 *    auto ret6 = container.GetInferenceData(1);  // get inference data vector
 *    auto ret7 = container.GetInferenceData(1, 1);  // get inference data[1]
 *    auto ret8 = container.ClearInferenceData(1);   // clear inference data of Model ID 1
 *    auto ret9 = container.ClearModelData(1);  // clear Model
 *    auto ret10 =container.ClearModelAll();  // clear all model data
 */


class ENN_GT_MODEL_CONTAINER_TEST : public testing::Test  {
public:
    /* global variable for test */
    enn::client::EnnModelContainer<uint64_t, SessionBufInfo, InferenceSet, InferenceData> container;
    std::shared_ptr<SessionBufInfo> sessBufInfo;
    std::shared_ptr<EnnBufferCore> mem_object, mem_object2;

    ENN_GT_MODEL_CONTAINER_TEST() {}
    ~ENN_GT_MODEL_CONTAINER_TEST() {}

    void SetUp();
    char tmp_char[10000] = {77, };
};

void ENN_GT_MODEL_CONTAINER_TEST::SetUp() {
    sessBufInfo = std::make_shared<SessionBufInfo>();
    sessBufInfo->model_id = 1;
    sessBufInfo->buffers.resize(4);
    sessBufInfo->regions.resize(2);
    sessBufInfo->buffers[0] = {
        .region_idx = 0,
        .dir = ENN_BUF_DIR_NONE,
        .buf_index = 0,
        .size = 10,
        .offset = 0,
        .shape = { 1, 5, 2, 1 },
        .buffer_type = 0,
        .name = "buffer0",
        .reserved = {0, },
    };
    sessBufInfo->buffers[1] = {
        .region_idx = 0,
        .dir = ENN_BUF_DIR_NONE,
        .buf_index = 1,
        .size = 20,
        .offset = 10,
        .shape = { 1, 2, 5, 1 },
        .buffer_type = 0,
        .name = "buffer1",
        .reserved = {0, },
    };
    sessBufInfo->buffers[2] = {
        .region_idx = 0,
        .dir = ENN_BUF_DIR_IN,
        .buf_index = 0,
        .size = 20,
        .offset = 0,
        .shape = { 1, 2, 5, 1 },
        .buffer_type = 0,
        .name = "correctBuffer0",
        .reserved = {0, },
    };
    sessBufInfo->buffers[3] = {
        .region_idx = 1,
        .dir = ENN_BUF_DIR_OUT,
        .buf_index = 0,
        .size = 40,
        .offset = 0,
        .shape = { 1, 2, 5, 1 },
        .buffer_type = 0,
        .name = "correctBuffer1",
        .reserved = {0, },
    };
    sessBufInfo->regions[0] = {
        .attr = 0,
        .req_size = 20,
        .name = "region0",
        .reserved = {0, },
    };
    sessBufInfo->regions[1] = {
        .attr = 0,
        .req_size = 40,
        .name = "region1",
        .reserved = {0, },
    };
    container.SetSessionData(sessBufInfo);

    mem_object = std::make_shared<EnnBufferCore>();
    mem_object->va = reinterpret_cast<void *>(tmp_char);
    mem_object->size = 20;
    mem_object->offset = 200;

    mem_object2 = std::make_shared<EnnBufferCore>();
    mem_object2->va = reinterpret_cast<void *>(tmp_char);
    mem_object2->size = 40;
    mem_object2->offset = 0;
}

TEST_F(ENN_GT_MODEL_CONTAINER_TEST, container_generate_inference_data_test_make1) {
    EXPECT_EQ(container.GenerateInferenceData(1), ENN_RET_SUCCESS);
}
TEST_F(ENN_GT_MODEL_CONTAINER_TEST, container_generate_inference_data_bigger_resize) {
    EXPECT_EQ(container.GenerateInferenceData(1), ENN_RET_SUCCESS);
    EXPECT_EQ(container.GenerateInferenceData(1, 3), ENN_RET_SUCCESS);
}

TEST_F(ENN_GT_MODEL_CONTAINER_TEST, container_generate_inference_data_smaller_resize) {
    EXPECT_EQ(container.GenerateInferenceData(1), ENN_RET_SUCCESS);
    EXPECT_EQ(container.GenerateInferenceData(1, 3), ENN_RET_SUCCESS);
    EXPECT_EQ(container.GenerateInferenceData(1, 1), ENN_RET_SUCCESS);
}

TEST_F(ENN_GT_MODEL_CONTAINER_TEST, container_generate_inference_data_resize0) {
    EXPECT_NE(container.GenerateInferenceData(1, 0), ENN_RET_SUCCESS);  // not allowed
}

TEST_F(ENN_GT_MODEL_CONTAINER_TEST, container_generate_inference_data_invalid_model_id) {
    EXPECT_NE(container.GenerateInferenceData(2, 3), ENN_RET_SUCCESS);
}

TEST_F(ENN_GT_MODEL_CONTAINER_TEST, container_set_inference_data_failure_test_size_mismatch) {
    // Default: Generate 2 inf_data at ModelID(1), clear and resize: Success
    EXPECT_EQ(container.GenerateInferenceData(1, 2), ENN_RET_SUCCESS);

    // 2-1. Set inference data[0] with "buffer0" of mem_object : Failed (size mismatch)
    EXPECT_NE(container.SetInferenceData(1, 0, std::string("buffer0"), mem_object.get()), ENN_RET_SUCCESS);
    // 2-2. Set inference data[0] with "buffer0" of mem_object2 : Failed (size mismatch)
    EXPECT_NE(container.SetInferenceData(1, 0, std::string("buffer0"), mem_object2.get()), ENN_RET_SUCCESS);
}

TEST_F(ENN_GT_MODEL_CONTAINER_TEST, container_set_inference_data_failure_test_partial_buffer_update) {
    // Default: Generate 2 inf_data at ModelID(1), clear and resize: Success
    EXPECT_EQ(container.GenerateInferenceData(1, 2), ENN_RET_SUCCESS);

    // 2-3. Set inference data[0] with "buffer1" of mem_object : Failed (partial buffer update)
    EXPECT_NE(container.SetInferenceData(1, 0, std::string("buffer1"), mem_object.get()), ENN_RET_SUCCESS);
    // 2-4. Set inference data[0] with "buffer1" of mem_object2 : Failed (partial buffer update)
    EXPECT_NE(container.SetInferenceData(1, 0, std::string("buffer1"), mem_object2.get()), ENN_RET_SUCCESS);
    // 2-5. Set inference data[0] with (IN, 0) = "buffer0" of mem_object2 : Failed (partial buffer update)
    EXPECT_NE(
        container.SetInferenceData(1, 0, std::tuple<uint32_t, int32_t>(DirType::ENN_BUF_DIR_IN, 0), mem_object2.get()),
        ENN_RET_SUCCESS);
}

TEST_F(ENN_GT_MODEL_CONTAINER_TEST, container_set_inference_data_set_by_label) {
    // Default: Generate 2 inf_data at ModelID(1), clear and resize: Success
    EXPECT_EQ(container.GenerateInferenceData(1, 2), ENN_RET_SUCCESS);
    // 2-6. Set inference data[0] with "correctBuffer0" of mem_object(size) : Success
    EXPECT_EQ(container.SetInferenceData(1, 0, std::string("correctBuffer0"), mem_object.get()), ENN_RET_SUCCESS);
}

TEST_F(ENN_GT_MODEL_CONTAINER_TEST, container_set_inference_data_set_by_dir_and_size) {
    // Default: Generate 2 inf_data at ModelID(1), clear and resize: Success
    EXPECT_EQ(container.GenerateInferenceData(1, 2), ENN_RET_SUCCESS);
    // 2-6. Set inference data[0] with (OUT, 0) = "correctBuffer1" of mem_object2(size) : Success
    EXPECT_EQ(
        container.SetInferenceData(1, 0, std::tuple<uint32_t, int32_t>(DirType::ENN_BUF_DIR_OUT, 0), mem_object2.get()),
        ENN_RET_SUCCESS);
    // 2-7. Set inference data[0] with "correctBuffer0" of mem_object(size) : Success
    EXPECT_EQ(container.SetInferenceData(1, 1, std::string("correctBuffer0"), mem_object.get()), ENN_RET_SUCCESS);
}

TEST_F(ENN_GT_MODEL_CONTAINER_TEST, container_set_inference_data_verify_failure_test) {
    // Default: Generate 2 inf_data at ModelID(1), clear and resize: Success
    EXPECT_EQ(container.GenerateInferenceData(1, 2), ENN_RET_SUCCESS);
    // 2-7. Set inference data[0] with "correctBuffer0" of mem_object(size) : Success
    EXPECT_EQ(container.SetInferenceData(1, 1, std::string("correctBuffer0"), mem_object.get()), ENN_RET_SUCCESS);
    // 3-1. inference data[1]: Failure
    EXPECT_NE(container.VerifyInferenceData(1, 1), ENN_RET_SUCCESS);
}

TEST_F(ENN_GT_MODEL_CONTAINER_TEST, container_set_inference_data_verify_test) {
    // Default: Generate 2 inf_data at ModelID(1), clear and resize: Success
    EXPECT_EQ(container.GenerateInferenceData(1, 1), ENN_RET_SUCCESS);
    // 2-6. Set inference data[0] with (OUT, 0) = "correctBuffer1" of mem_object2(size) : Success
    EXPECT_EQ(
        container.SetInferenceData(1, 0, std::tuple<uint32_t, int32_t>(DirType::ENN_BUF_DIR_OUT, 0), mem_object2.get()),
        ENN_RET_SUCCESS);
    // 2-7. Set inference data[0] with "correctBuffer0" of mem_object(size) : Success
    EXPECT_EQ(container.SetInferenceData(1, 0, std::string("correctBuffer0"), mem_object.get()), ENN_RET_SUCCESS);
    // 3-1. inference data[0]: Success
    EXPECT_EQ(container.VerifyInferenceData(1, 0), ENN_RET_SUCCESS);
}

TEST_F(ENN_GT_MODEL_CONTAINER_TEST, container_set_inference_data_and_dump) {
    char *result = nullptr;
    uint32_t out_size;

    // Default: Generate 2 inf_data at ModelID(1), clear and resize: Success
    EXPECT_EQ(container.GenerateInferenceData(1, 1), ENN_RET_SUCCESS);
    // 2-6. Set inference data[0] with (OUT, 0) = "correctBuffer1" of mem_object2(size) : Success
    EXPECT_EQ(
        container.SetInferenceData(1, 0, std::tuple<uint32_t, int32_t>(DirType::ENN_BUF_DIR_OUT, 0), mem_object2.get()),
        ENN_RET_SUCCESS);
    // 2-7. Set inference data[0] with "correctBuffer0" of mem_object(size) : Success
    EXPECT_EQ(container.SetInferenceData(1, 0, std::string("correctBuffer0"), mem_object.get()), ENN_RET_SUCCESS);
    // 3-1. inference data[0]: Success
    EXPECT_EQ(container.VerifyInferenceData(1, 0), ENN_RET_SUCCESS);
    if (!container.DumpSessionToFile(1, "test", 0)) {
        // check file is okay
        ::enn::util::import_file_to_mem("test_exec1_index0_offset200.dump", &result, &out_size);
        remove("test_exec1_index0_offset200.dump");  // remove file
        remove("test_exec1_index1_offset0.dump");    // remove file
        EXPECT_EQ(result[0], 77);                    // 77 is initialized value of buffer
    }

    free(result);
}

TEST_F(ENN_GT_MODEL_CONTAINER_TEST, container_all_test) {
    // =========================
    //  Environment
    // =========================
    //  - ModelID(1) is registered, has 4 buffers
    //  - buffer0, buffer1 has offset or mismatched size -> cannot set as a session data
    //    Correctbuffer0, Correctbuffer1 has no offset, matched size -> can set as a session data
    //  - region0, region1 can be set by correctBuffer0, correctBuffer1
    //
    // =========================
    //  Test
    // =========================
    // 1-1. Generate 1(default) inf_data at ModelID(1) : Success
    EXPECT_EQ(container.GenerateInferenceData(1), ENN_RET_SUCCESS);
    // 1-2. Generate 3 inf_data at ModelID(1), clear and resize: : Success
    EXPECT_EQ(container.GenerateInferenceData(1, 3), ENN_RET_SUCCESS);
    // 1-3. Generate 1 inf_data at ModelID(1), clear and resize: Success
    EXPECT_EQ(container.GenerateInferenceData(1, 1), ENN_RET_SUCCESS);
    // Finished: Container has 2 session data in Model ID(1)

    // check 1: Buffer -> Region, size are same?, not partial update?
    // check 2: MemoryObj -> buffer, size are same?
    //
    // 2-6. Set inference data[0] with "correctBuffer0" of mem_object(size) : Success
    EXPECT_EQ(container.SetInferenceData(1, 0, std::string("correctBuffer0"), mem_object.get()), ENN_RET_SUCCESS);
    // 2-6. Set inference data[0] with (OUT, 0) = "correctBuffer1" of mem_object2(size) : Success
    EXPECT_EQ(
        container.SetInferenceData(1, 0, std::tuple<uint32_t, int32_t>(DirType::ENN_BUF_DIR_OUT, 0), mem_object2.get()),
        ENN_RET_SUCCESS);

    // check: All inference data are filled?
    // 3-1. inference data[0]: Success
    EXPECT_EQ(container.VerifyInferenceData(1, 0), ENN_RET_SUCCESS);

    // Check: Get inference data all
    // 4-1. GetInferenceData: we expect [0] {.addr = , .size=20, .offset=200}
    //                      : we expect [1] {.addr = , .size=40, .offset=0}
    auto ret = container.GetInferenceData(1);
    EXPECT_EQ(ret[0].n_region, 2);
    EXPECT_EQ(ret[0].inference_data[0].size, 20);
    EXPECT_EQ(ret[0].inference_data[0].offset, 200);
    EXPECT_EQ(ret[0].inference_data[1].size, 40);
    EXPECT_EQ(ret[0].inference_data[1].offset, 0);

    // 5-0. Set inference data : Success
    EXPECT_EQ(container.SetInferenceData(1, 0, std::string("correctBuffer0"), mem_object.get()), ENN_RET_SUCCESS);
    // 5-1. Clear data
    EXPECT_EQ(container.ClearInferenceData(1), ENN_RET_SUCCESS);
    // 5-2. Not working because InferenceData is cleared.
    EXPECT_NE(container.SetInferenceData(1, 0, std::string("correctBuffer0"), mem_object.get()), ENN_RET_SUCCESS);

    EXPECT_EQ(container.ClearModelData(1), ENN_RET_SUCCESS);
    EXPECT_EQ(container.ClearModelAll(), ENN_RET_SUCCESS);
}



/* Scenario test 2 */
class ENN_GT_MODEL_CONTAINER_TEST2 : public testing::Test  {
public:
    /* global variable for test */
    enn::client::EnnModelContainer<uint64_t, SessionBufInfo, InferenceSet, InferenceData> container;
    std::shared_ptr<SessionBufInfo> sessBufInfo;
    std::shared_ptr<EnnBufferCore> mem_object[5];

    ENN_GT_MODEL_CONTAINER_TEST2() {}
    ~ENN_GT_MODEL_CONTAINER_TEST2() {}

    void SetUp();
    char tmp_char[10000] = {77, };
};

void ENN_GT_MODEL_CONTAINER_TEST2::SetUp() {
    sessBufInfo = std::make_shared<SessionBufInfo>();
    sessBufInfo->model_id = 2;
    sessBufInfo->buffers.resize(7);
    sessBufInfo->regions.resize(5);
    sessBufInfo->buffers[0] = {
        .region_idx = 0,
        .dir = ENN_BUF_DIR_IN,
        .buf_index = 0,
        .size = 307200,
        .offset = 0,
        .shape = { 1, 3, 320, 320 },
        .buffer_type = 0,
        .name = "IFM",
        .reserved = {0, },
    };
    sessBufInfo->buffers[1] = {
        .region_idx = 1,
        .dir = ENN_BUF_DIR_EXT,
        .buf_index = 0,
        .size = 8136,
        .offset = 10,
        .shape = { 1, 8136, 1, 1 },
        .buffer_type = 0,
        .name = "765",
        .reserved = {0, },
    };
    sessBufInfo->buffers[2] = {
        .region_idx = 1,
        .dir = ENN_BUF_DIR_EXT,
        .buf_index = 1,
        .size = 185094,
        .offset = 8136,
        .shape = { 1, 185094, 1, 1 },
        .buffer_type = 0,
        .name = "766",
        .reserved = {0, },
    };
    sessBufInfo->buffers[3] = {
        .region_idx = 1,
        .dir = ENN_BUF_DIR_EXT,
        .buf_index = 2,
        .size = 65088,
        .offset = 193230,
        .shape = { 1, 2, 8136, 1 },
        .buffer_type = 0,
        .name = "763",
        .reserved = {0, },
    };
    sessBufInfo->buffers[4] = {
        .region_idx = 2,
        .dir = ENN_BUF_DIR_EXT,
        .buf_index = 3,
        .size = 32544,
        .offset = 0,
        .shape = { 1, 1, 8136, 1 },
        .buffer_type = 0,
        .name = "754",
        .reserved = {0, },
    };
    sessBufInfo->buffers[5] = {
        .region_idx = 3,
        .dir = ENN_BUF_DIR_EXT,
        .buf_index = 4,
        .size = 740376,
        .offset = 0,
        .shape = { 1, 185094, 1, 1 },
        .buffer_type = 0,
        .name = "756",
        .reserved = {0, },
    };
    sessBufInfo->buffers[6] = {
        .region_idx = 4,
        .dir = ENN_BUF_DIR_OUT,
        .buf_index = 0,
        .size = 308,
        .offset = 0,
        .shape = { 1, 1, 11, 7 },
        .buffer_type = 0,
        .name = "308",
        .reserved = {0, },
    };


    sessBufInfo->regions[0] = {
        .attr = 0,
        .req_size = 307200,
        .name = "region0",
        .reserved = {0, },
    };
    sessBufInfo->regions[1] = {
        .attr = 0,
        .req_size = 258318,
        .name = "region1",
        .reserved = {0, },
    };
    sessBufInfo->regions[2] = {
        .attr = 0,
        .req_size = 32544,
        .name = "region2",
        .reserved = {0, },
    };
    sessBufInfo->regions[3] = {
        .attr = 0,
        .req_size = 740376,
        .name = "region3",
        .reserved = {0, },
    };
    sessBufInfo->regions[4] = {
        .attr = 0,
        .req_size = 308,
        .name = "region4",
        .reserved = {0, },
    };
    container.SetSessionData(sessBufInfo);

    mem_object[0] = std::make_shared<EnnBufferCore>();
    mem_object[0]->va = reinterpret_cast<void *>(tmp_char);
    mem_object[0]->size = 307200;
    mem_object[0]->offset = 0;

    mem_object[4] = std::make_shared<EnnBufferCore>();
    mem_object[4]->va = reinterpret_cast<void *>(tmp_char);
    mem_object[4]->size = 308;
    mem_object[4]->offset = 0;
}


TEST_F(ENN_GT_MODEL_CONTAINER_TEST2, container_scenario_test) {
    // 1-1. Generate 1(default) inf_data at ModelID(1) : Success
    EXPECT_EQ(container.GenerateInferenceData(2), ENN_RET_SUCCESS);
    // Finished: Container has 2 session data in Model ID(1)

    // check 1: Buffer -> Region, size are same?, not partial update?
    // check 2: MemoryObj -> buffer, size are same?
    //
    // 2-6. Set inference data[0] with "correctBuffer0" of mem_object(size) : Success
    EXPECT_EQ(
        container.SetInferenceData(2, 0, std::tuple<uint32_t, int32_t>(DirType::ENN_BUF_DIR_IN, 0), mem_object[0].get()),
        ENN_RET_SUCCESS);

    EXPECT_EQ(
        container.SetInferenceData(2, 0, std::tuple<uint32_t, int32_t>(DirType::ENN_BUF_DIR_OUT, 0), mem_object[4].get()),
        ENN_RET_SUCCESS);

    auto idx_set = container.GetExtRegionIndexes(2, 0);
    EXPECT_EQ(idx_set.size(), 3);

    mem_object[1] = std::make_shared<EnnBufferCore>();
    mem_object[1]->va = reinterpret_cast<void *>(tmp_char);
    mem_object[1]->size = 258318;
    mem_object[1]->offset = 0;

    mem_object[2] = std::make_shared<EnnBufferCore>();
    mem_object[2]->va = reinterpret_cast<void *>(tmp_char);
    mem_object[2]->size = 32544;
    mem_object[2]->offset = 0;

    mem_object[3] = std::make_shared<EnnBufferCore>();
    mem_object[3]->va = reinterpret_cast<void *>(tmp_char);
    mem_object[3]->size = 740376;
    mem_object[3]->offset = 0;

    for (auto &idx: idx_set) {
        ENN_TST_PRINT("get idx: %d\n", idx);
        container.SetInferenceData(2, 0, idx, mem_object[idx].get());
    }

    // check: All inference data are filled?
    // 3-1. inference data[0]: Success
    EXPECT_EQ(container.VerifyInferenceData(2, 0), ENN_RET_SUCCESS);
}


TEST_F(ENN_GT_MODEL_CONTAINER_TEST2, ext_buffer_management_test) {
    EXPECT_EQ(container.SetAutoAllocatedExtBuffersToSession(2, 0, mem_object[0]), ENN_RET_SUCCESS);
    container.SetAutoAllocatedExtBuffersToSession(2, 0, mem_object[4]);
    container.ShowAutoAllocatedExtBuffers();
    auto ext_buf_lists = container.GetAutoAllocatedExtBuffersFromSession(2, 0);
    EXPECT_EQ(ext_buf_lists.size(), 2);
    EXPECT_EQ(ext_buf_lists[0]->size, mem_object[0]->size);
    EXPECT_EQ(ext_buf_lists[0]->va, mem_object[0]->va);
    EXPECT_EQ(ext_buf_lists[0]->offset, mem_object[0]->offset);
    EXPECT_EQ(ext_buf_lists[1]->size, mem_object[4]->size);
    EXPECT_EQ(ext_buf_lists[1]->va, mem_object[4]->va);
    EXPECT_EQ(ext_buf_lists[1]->offset, mem_object[4]->offset);
}


TEST_F(ENN_GT_MODEL_CONTAINER_TEST2, one_ext_buffer_is_set_by_user_auto_allocation_test) {
    EXPECT_EQ(container.SetAutoAllocatedExtBuffersToSession(2, 0, mem_object[0]), ENN_RET_SUCCESS);
    container.SetAutoAllocatedExtBuffersToSession(2, 0, mem_object[4]);
    container.ShowAutoAllocatedExtBuffers();
    auto ext_buf_lists = container.GetAutoAllocatedExtBuffersFromSession(2, 0);
    EXPECT_EQ(ext_buf_lists.size(), 2);
    EXPECT_EQ(ext_buf_lists[0]->size, mem_object[0]->size);
    EXPECT_EQ(ext_buf_lists[0]->va, mem_object[0]->va);
    EXPECT_EQ(ext_buf_lists[0]->offset, mem_object[0]->offset);
    EXPECT_EQ(ext_buf_lists[1]->size, mem_object[4]->size);
    EXPECT_EQ(ext_buf_lists[1]->va, mem_object[4]->va);
    EXPECT_EQ(ext_buf_lists[1]->offset, mem_object[4]->offset);
}



TEST_F(ENN_GT_MODEL_CONTAINER_TEST2, multi_session_id_test) {
    // 1-1. Generate 1(default) inf_data at ModelID(1) : Success
    EXPECT_EQ(container.GenerateInferenceData(2, 2), ENN_RET_SUCCESS);
    // Finished: Container has 2 session data in Model ID(1)

    // check 1: Buffer -> Region, size are same?, not partial update?
    // check 2: MemoryObj -> buffer, size are same?
    //
    // 2-6. Set inference data[0] with "correctBuffer0" of mem_object(size) : Success
    EXPECT_EQ(
        container.SetInferenceData(2, 0, std::tuple<uint32_t, int32_t>(DirType::ENN_BUF_DIR_IN, 0), mem_object[0].get()),
        ENN_RET_SUCCESS);
    EXPECT_EQ(
        container.SetInferenceData(2, 1, std::tuple<uint32_t, int32_t>(DirType::ENN_BUF_DIR_IN, 0), mem_object[0].get()),
        ENN_RET_SUCCESS);

    EXPECT_EQ(
        container.SetInferenceData(2, 0, std::tuple<uint32_t, int32_t>(DirType::ENN_BUF_DIR_OUT, 0), mem_object[4].get()),
        ENN_RET_SUCCESS);
    EXPECT_EQ(
        container.SetInferenceData(2, 1, std::tuple<uint32_t, int32_t>(DirType::ENN_BUF_DIR_OUT, 0), mem_object[4].get()),
        ENN_RET_SUCCESS);

    auto idx_set = container.GetExtRegionIndexes(2, 0);
    EXPECT_EQ(idx_set.size(), 3);

    mem_object[1] = std::make_shared<EnnBufferCore>();
    mem_object[1]->va = reinterpret_cast<void *>(tmp_char);
    mem_object[1]->size = 258318;
    mem_object[1]->offset = 0;

    mem_object[2] = std::make_shared<EnnBufferCore>();
    mem_object[2]->va = reinterpret_cast<void *>(tmp_char);
    mem_object[2]->size = 32544;
    mem_object[2]->offset = 0;

    mem_object[3] = std::make_shared<EnnBufferCore>();
    mem_object[3]->va = reinterpret_cast<void *>(tmp_char);
    mem_object[3]->size = 740376;
    mem_object[3]->offset = 0;

    for (auto &idx: idx_set) {
        ENN_TST_PRINT("get idx: %d\n", idx);
        container.SetInferenceData(2, 0, idx, mem_object[idx].get());
        container.SetInferenceData(2, 1, idx, mem_object[idx].get());
    }

    // check: All inference data are filled?
    // 3-1. inference data[0]: Success
    EXPECT_EQ(container.VerifyInferenceData(2, 0), ENN_RET_SUCCESS);
    EXPECT_EQ(container.VerifyInferenceData(2, 1), ENN_RET_SUCCESS);
}

