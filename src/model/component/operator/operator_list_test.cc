#include <gtest/gtest.h>

#include <string>
#include <iostream>
#include <cstdlib>
#include <ctime>

#include "model/component/operator/operator_builder.hpp"
#include "model/component/operator/operator_list_builder.hpp"
#include "model/model.hpp"
#include "runtime/client_process/client_process.hpp"
#include "common/identifier.hpp"

using namespace enn::model;
using namespace enn::model::component;
using namespace enn::identifier;
using namespace enn::runtime;

class OperatorListTest : public testing::Test {
 protected:
    void SetUp() override {
        std::srand(std::time(nullptr));  // generate seed for rand
        process = std::make_unique<ClientProcess>();
    }

    auto create_model() {
        return std::make_unique<Model>(process);
    }

    ClientProcess::Ptr process;
    enn::model::component::OperatorListBuilder operator_list_builder;
};


TEST_F(OperatorListTest, check_setting_random_id) {
    auto model = create_model();

    auto opr_list = operator_list_builder.build(model->get_id())
                                         .create();
    EXPECT_TRUE(opr_list->get_id().equal_in_common(model->get_id()));
}

TEST_F(OperatorListTest, check_setting_npu_accelerator) {
    auto model = create_model();
    auto opr_list = operator_list_builder.build(model->get_id())
                                         .set_accelerator(enn::model::Accelerator::NPU)
                                         .create();
    EXPECT_EQ((opr_list->get_accelerator()), enn::model::Accelerator::NPU);
}

TEST_F(OperatorListTest, check_setting_with_fluent_interface) {
    auto model = create_model();

    auto name = std::to_string(std::rand());
    auto accelerator = enn::model::Accelerator::NPU;

    auto opr_list = operator_list_builder.build(model->get_id())
                                         .set_name(name)
                                         .set_accelerator(accelerator)
                                         .create();

    EXPECT_TRUE(opr_list->get_id().equal_in_common(model->get_id()));
    EXPECT_STREQ(opr_list->get_name().c_str(), name.c_str());
    EXPECT_EQ((opr_list->get_accelerator()), accelerator);
}

TEST_F(OperatorListTest, check_adding_operators_with_id) {
    // get OperatorBuilder for creating each operator
    OperatorBuilder operator_builder;
    auto model = create_model();

    auto id = std::rand();
    Operator::Ptr opr1 = operator_builder.set_id(id + 1).create();
    Operator::Ptr opr2 = operator_builder.set_id(id + 2).create();
    Operator::Ptr opr3 = operator_builder.set_id(id + 3).create();
    Operator::Ptr opr4 = operator_builder.set_id(id + 4).create();
    Operator::Ptr opr5 = operator_builder.set_id(id + 5).create();
    Operator::Ptr opr6 = operator_builder.set_id(id + 6).create();

    auto opr_list = operator_list_builder.build(model->get_id())
                                         .add_operator(opr1)
                                         .add_operator(opr2)
                                         .add_operator(opr3)
                                         .add_operator(opr4)
                                         .add_operator(opr5)
                                         .add_operator(opr6)
                                         .create();

    for (auto opr : *opr_list) {
        EXPECT_EQ(std::static_pointer_cast<Operator>(opr)->get_id(), ++id);
    }
}

TEST_F(OperatorListTest, check_setting_attribute) {
    // get OperatorBuilder for creating each operator
    OperatorBuilder operator_builder;
    auto model= create_model();

    auto opr_list = operator_list_builder.build(model->get_id())
                                         .set_attribute(std::make_shared<Attribute>())
                                         .create();

    // check values with default values
    EXPECT_EQ(opr_list->get_attribute().get_relax_computation_float32_to_float16(), true);
}

TEST_F(OperatorListTest, create_and_release_10000_times) {
    constexpr int iter = 10000;
    auto model = create_model();
    OperatorBuilder operator_builder;
    for (int i = 0; i < iter; i++) {
        // OperatorList is releasd on getting out of scope here.
        EXPECT_NO_THROW(operator_list_builder.build(model->get_id()).create());
    }
}

TEST_F(OperatorListTest, throws_on_create_with_id_over) {
    constexpr int max = OperatorList::UniqueID::Max;

    // add objects created to prevent from being released
    std::vector<OperatorList::Ptr> em_list;
    auto model = create_model();
    OperatorBuilder operator_builder;
    // create OperatorList objects by upper bound
    for (int i = 1; i <= max; i++) {
        em_list.push_back(operator_list_builder.build(model->get_id()).create());
    }
    // create one more OperatorList object with id that exceeds the max limit.
    EXPECT_THROW(operator_list_builder.build(model->get_id()).create(), std::runtime_error);
}

TEST_F(OperatorListTest, test_set_and_get_preset_id) {
    auto model = create_model();
    uint32_t preset_id = std::rand();
    auto op_list = operator_list_builder.build(model->get_id())
                                        .set_preset_id(preset_id)
                                        .create();
    EXPECT_EQ(op_list->get_preset_id(), preset_id);
}

TEST_F(OperatorListTest, test_set_and_get_pref_mode) {
    auto model = create_model();
    uint32_t pref_mode = std::rand();
    auto op_list = operator_list_builder.build(model->get_id())
                                        .set_pref_mode(pref_mode)
                                        .create();
    EXPECT_EQ(op_list->get_pref_mode(), pref_mode);
}

TEST_F(OperatorListTest, test_set_and_get_target_latency) {
    auto model = create_model();
    uint32_t target_latency = std::rand();
    auto op_list = operator_list_builder.build(model->get_id())
                                        .set_target_latency(target_latency)
                                        .create();
    EXPECT_EQ(op_list->get_target_latency(), target_latency);
}

TEST_F(OperatorListTest, test_set_and_get_tile_num) {
    auto model = create_model();
    uint32_t tile_num = std::rand();
    auto op_list = operator_list_builder.build(model->get_id())
                                        .set_tile_num(tile_num)
                                        .create();
    EXPECT_EQ(op_list->get_tile_num(), tile_num);
}

TEST_F(OperatorListTest, test_set_and_get_core_affinity) {
    auto model = create_model();
    uint32_t core_affinity = std::rand();
    auto op_list = operator_list_builder.build(model->get_id())
                                        .set_core_affinity(core_affinity)
                                        .create();
    EXPECT_EQ(op_list->get_core_affinity(), core_affinity);
}

TEST_F(OperatorListTest, test_set_and_get_priority) {
    auto model = create_model();
    uint32_t priority = std::rand();
    auto op_list = operator_list_builder.build(model->get_id())
                                        .set_priority(priority)
                                        .create();
    EXPECT_EQ(op_list->get_priority(), priority);
}
