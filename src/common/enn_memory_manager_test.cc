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

#include "gtest/gtest.h"
#include "common/enn_debug.h"
#include "common/enn_utils.h"
#include "common/enn_memory_manager.h"

namespace enn {
namespace test {
namespace internal {

class ENN_GT_UNITTEST_MEMORY : public testing::Test {};

TEST_F(ENN_GT_UNITTEST_MEMORY, memory_alloc_dealloc_test) {
    /* Init -> alloc(2) -> show() -> delete(2) -> deinit() */
    /* Generate Memory Manager instantly */
    enn::EnnMemoryManager *emm = new enn::EnnMemoryManager();
    emm->init();

    /* TODO(hoon98.choi, TBD): we will decide Android have multiple allocators (heap, dmabuf) */
    auto test_mem1 = emm->CreateMemory(10, enn::EnnMmType::kEnnMmTypeIon, 0);
    auto test_mem2 = emm->CreateMemory(283722, enn::EnnMmType::kEnnMmTypeIon, 0);

    EXPECT_NE(nullptr, test_mem1);
    EXPECT_NE(nullptr, test_mem2);

    emm->ShowMemoryPool();

    EXPECT_EQ(0, emm->DeleteMemory(test_mem1));
    EXPECT_EQ(0, emm->DeleteMemory(test_mem2));

    delete emm;
}

TEST_F(ENN_GT_UNITTEST_MEMORY, memory_alloc_dealloc_one_missing_test) {
    /* Init -> alloc(2) -> show() -> delete(1) -> deinit() */
    /* Generate Memory Manager instantly */
    enn::EnnMemoryManager *emm = new enn::EnnMemoryManager();
    emm->init();

    auto test_mem1 = emm->CreateMemory(454, enn::EnnMmType::kEnnMmTypeIon, 0);
    auto test_mem2 = emm->CreateMemory(64331, enn::EnnMmType::kEnnMmTypeIon, 0);

    EXPECT_NE(nullptr, test_mem1);
    EXPECT_NE(nullptr, test_mem2);

    emm->ShowMemoryPool();

    EXPECT_EQ(0, emm->DeleteMemory(test_mem1));

    /* Memory clean in pool is expected */
    // should be free 283722 (test_mem2) after the below line is executed.
    delete emm;
}

/* CreateMemoryFromFd() is used in Android */
#ifdef __ANDROID__
TEST_F(ENN_GT_UNITTEST_MEMORY, android_memory_fd_import_test) {
    /* Generate Memory Manager instantly */
    enn::EnnMemoryManager *emm = new enn::EnnMemoryManager();
    emm->init();

    auto test_mem1 = emm->CreateMemory(100, enn::EnnMmType::kEnnMmTypeIon, 0);
    EXPECT_NE(nullptr, test_mem1);
    test_mem1->show();
    auto test_mem_import = emm->CreateMemoryFromFd(test_mem1->fd, 100);
    emm->ShowMemoryPool();

    char *source = reinterpret_cast<char *>(test_mem1->va);
    char *target = reinterpret_cast<char *>(test_mem_import->va);

    /* Set source -> check target and source are same */
    for (int i = 0; i < 100; i++) source[i] = i*2;
    for (int i = 0; i < 100; i++) EXPECT_EQ(target[i], i*2);

    EXPECT_EQ(0, emm->DeleteMemory(test_mem1));
    EXPECT_EQ(0, emm->DeleteMemory(test_mem_import));

    delete emm;
}

TEST_F(ENN_GT_UNITTEST_MEMORY, android_memory_fd_offset_test) {
    const int offset = 50;
    enn::EnnMemoryManager *emm = new enn::EnnMemoryManager();
    emm->init();

    auto test_mem1 = emm->CreateMemory(100, enn::EnnMmType::kEnnMmTypeIon, 0);
    EXPECT_NE(nullptr, test_mem1);
    test_mem1->show();
    auto test_mem_import = emm->CreateMemoryFromFdWithOffset(test_mem1->fd, 50, offset);
    EXPECT_NE(test_mem_import, nullptr);
    emm->ShowMemoryPool();

    char *source = reinterpret_cast<char *>(test_mem1->va);
    char *target = reinterpret_cast<char *>(test_mem_import->va);

    /* Set source -> check target and source are same */
    for (int i = 0; i < 100; i++) source[i] = i;
    for (int i = 0; i < 50; i++) EXPECT_EQ(target[i], i + offset);

    EXPECT_EQ(0, emm->DeleteMemory(test_mem1));
    EXPECT_EQ(0, emm->DeleteMemory(test_mem_import));

    delete emm;
}
#endif

}  // namespace internal
}  // namespace test
}  // namespace enn
