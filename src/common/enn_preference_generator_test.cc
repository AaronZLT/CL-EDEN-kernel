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
#include "common/enn_preference_generator.hpp"

namespace enn {
namespace test {
namespace internal {
class ENN_PREFERENCE_GENERATOR_TEST : public testing::Test {
};

TEST_F(ENN_PREFERENCE_GENERATOR_TEST, stream_pointer_test) {
    enn::preference::EnnPreferenceGenerator instance;
    uint32_t *ptr = instance.get_stream_pointer();

    // Test default values
    EXPECT_EQ(ptr[0], 0);
    EXPECT_EQ(ptr[1], ENN_PREF_MODE_BOOST_ON_EXE);
    EXPECT_EQ(ptr[2], 0);
    EXPECT_EQ(ptr[3], 1);
    EXPECT_EQ(ptr[4], 0xFFFFFFFF);
    EXPECT_EQ(ptr[5], 0);
    EXPECT_EQ(ptr[6], 0);
}

TEST_F(ENN_PREFERENCE_GENERATOR_TEST, set_get_test) {
    enn::preference::EnnPreferenceGenerator instance;
    instance.set_core_affinity(0xF);
    EXPECT_EQ(instance.get_core_affinity(), 0xF);
    instance.reset_as_default();
    EXPECT_NE(instance.get_core_affinity(), 0xF);
}

TEST_F(ENN_PREFERENCE_GENERATOR_TEST, import_from_vector) {
    enn::preference::EnnPreferenceGenerator instance;

    // export vector from preference_generator
    auto ref_vec = instance.export_preference_to_vector();
    uint32_t test_val_base = 0;
    EXPECT_EQ(ref_vec[0], 0);
    EXPECT_EQ(ref_vec[1], ENN_PREF_MODE_BOOST_ON_EXE);
    EXPECT_EQ(ref_vec[2], 0);
    EXPECT_EQ(ref_vec[3], 1);
    EXPECT_EQ(ref_vec[4], 0xFFFFFFFF);
    EXPECT_EQ(ref_vec[5], 0);
    EXPECT_EQ(ref_vec[6], 0);

    // set 1 ~ 7
    for (auto & vec : ref_vec)
        vec = ++test_val_base;

    // import
    instance.import_preference_from_vector(ref_vec);

    // export and test
    auto ref_vec2 = instance.export_preference_to_vector();
    EXPECT_EQ(ref_vec2[0], 1);
    EXPECT_EQ(ref_vec2[1], 2);
    EXPECT_EQ(ref_vec2[2], 3);
    EXPECT_EQ(ref_vec2[3], 4);
    EXPECT_EQ(ref_vec2[4], 5);
    EXPECT_EQ(ref_vec2[5], 6);
    EXPECT_EQ(ref_vec2[6], 7);
}



};
};
};
