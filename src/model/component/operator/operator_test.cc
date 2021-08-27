#include <gtest/gtest.h>

#include <string>
#include <iostream>
#include <cstdlib>
#include <ctime>

#include "model/component/operator/operator_builder.hpp"
#include "model/component/tensor/feature_map_builder.hpp"
#include "model/types.hpp"  // enum class Accelerator
#include "common/enn_utils.h"

class OperatorTest : public testing::Test {
 protected:
    void SetUp() override {
        std::srand(std::time(nullptr));  // generate seed for rand
    }

    // add in/out edges to opr
    void add_edges(enn::model::component::Operator::Ptr opr) {
        enn::model::component::FeatureMapBuilder feature_map_builder;
        const int edge_cnt = 5;
        // add in edges as many as edge_cnt
        for (int i = 0; i < edge_cnt; i++) {
            enn::model::component::FeatureMap::Ptr feature_map = feature_map_builder.set_id(i).create();
            enn::model::component::OperatorBuilder opr_builder(opr);
            opr_builder.add_in_tensor(feature_map);
        }
        // add out edges as many as edge_cnt
        for (int i = 0; i < edge_cnt; i++) {
            enn::model::component::FeatureMap::Ptr feature_map = feature_map_builder.set_id(i).create();
            enn::model::component::OperatorBuilder opr_builder(opr);
            opr_builder.add_out_tensor(feature_map);
        }
    }

    enn::model::component::OperatorBuilder operator_builder;
};


TEST_F(OperatorTest, test_set_and_get_id_by_random) {
    auto id = std::rand();
    auto opr = operator_builder.set_id(id).create();
    EXPECT_EQ(opr->get_id(), id);
}

TEST_F(OperatorTest, test_set_and_get_name_by_random) {
    auto name = std::to_string(std::rand());
    auto opr = operator_builder.set_name(name).create();
    EXPECT_STREQ(opr->get_name().c_str(), name.c_str());
}

TEST_F(OperatorTest, test_set_and_get_code_with_softmax_code) {
    auto code = TFlite::BuiltinOperator::BuiltinOperator_SOFTMAX;
    auto opr = operator_builder.set_code(code).create();
    EXPECT_EQ(opr->get_code(), code);
}

TEST_F(OperatorTest, test_set_and_get_npu_accelerator) {
    auto opr = operator_builder.set_accelerator(enn::model::Accelerator::NPU).create();
    EXPECT_EQ(opr->get_accelerator(), enn::model::Accelerator::NPU);
}

TEST_F(OperatorTest, test_set_and_get_binary_by_dummy_class) {
    // regard dummy class Binary as virtual option object.
    class Binary {
        int value = 1;
     public:
        int get() const { return value; }
    };
    std::string binary_name = "Binary";
    int32_t binary_fd = 100;
    auto binary = new Binary;
    auto opr = operator_builder.add_binary(binary_name, binary_fd, binary, sizeof(Binary), enn::model::Accelerator::NPU)
                               .create();
    EXPECT_EQ(opr->get_binaries().at(enn::util::FIRST).get_addr(), binary);
    EXPECT_EQ((static_cast<const Binary*>(opr->get_binaries().at(enn::util::FIRST).get_addr()))->get(), binary->get());
    EXPECT_EQ(opr->get_binaries().at(enn::util::FIRST).get_size(), sizeof(Binary));
    EXPECT_STREQ(opr->get_binaries().at(enn::util::FIRST).get_name().c_str(), binary_name.c_str());
    EXPECT_EQ(opr->get_binaries().at(enn::util::FIRST).get_fd(), binary_fd);
    EXPECT_EQ(opr->get_binaries().at(enn::util::FIRST).get_accelerator(), enn::model::Accelerator::NPU);
    delete binary;
}

TEST_F(OperatorTest, test_set_and_get_option_by_dummy_class) {
    // regard dummy class Option as virtual option object.
    class Option {
        int value = 1;
     public:
        TFlite::BuiltinOptions num = TFlite::BuiltinOptions_Conv2DOptions;
        int get() const { return value; }
    };
    std::string option_name = "Option";
    auto option = new Option;
    auto opr = operator_builder.set_option(option_name, option, sizeof(Option), option->num)
                               .create();
    EXPECT_EQ(opr->get_option().get_addr(), option);
    EXPECT_EQ((static_cast<const Option*>(opr->get_option().get_addr()))->get(), option->get());
    EXPECT_EQ(opr->get_option().get_size(), sizeof(Option));
    EXPECT_EQ(opr->get_option().get_enum(), option->num);
    EXPECT_STREQ(opr->get_option().get_name().c_str(), option_name.c_str());
    delete option;
}

TEST_F(OperatorTest, test_set_and_get_in_pixel_format_by_random) {
    auto in_pixel_format = std::rand();
    auto opr = operator_builder.set_in_pixel_format(in_pixel_format).create();
    EXPECT_EQ(opr->get_in_pixel_format(), in_pixel_format);
}

TEST_F(OperatorTest, test_set_and_get_buffer_shared_by_random) {
    auto buffer_shared = std::rand() % 2;
    auto opr = operator_builder.set_buffer_shared(buffer_shared).create();
    EXPECT_EQ(opr->is_buffer_shared(), buffer_shared);
}

TEST_F(OperatorTest, test_set_and_get_ofm_bound_by_random) {
    auto ofm_bound = std::rand() % 2;
    auto opr = operator_builder.set_ofm_bound(ofm_bound).create();
    EXPECT_EQ(opr->is_ofm_bound(), ofm_bound);
}

TEST_F(OperatorTest, test_add_and_iterate_edges) {
    enn::model::component::Operator::Ptr opr = operator_builder.create();
    add_edges(opr);
    int expected_id = 0;
    for (auto& in_tensor : opr->in_tensors) {
        EXPECT_EQ(in_tensor->get_id(), expected_id++);
    }
    expected_id = 0;
    for (auto& out_tensor : opr->out_tensors) {
        EXPECT_EQ(out_tensor->get_id(), expected_id++);
    }
}

TEST_F(OperatorTest, test_set_and_get_by_fluent_interface) {
    auto id = std::rand();
    auto name = std::to_string(std::rand());
    auto code = TFlite::BuiltinOperator::BuiltinOperator_SOFTMAX;
    auto accelerator = enn::model::Accelerator::NPU;
    auto opr = operator_builder.set_id(id)
                               .set_name(name)
                               .set_code(code)
                               .set_accelerator(accelerator)
                               .create();
    EXPECT_EQ(opr->get_id(), id);
    EXPECT_STREQ(opr->get_name().c_str(), name.c_str());
    EXPECT_EQ(opr->get_code(), code);
    EXPECT_EQ(opr->get_accelerator(), accelerator);
}
