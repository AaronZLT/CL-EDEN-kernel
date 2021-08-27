#include <gtest/gtest.h>

#include <string>
#include <iostream>
#include <cstdlib>
#include <ctime>

#include "model/component/tensor/feature_map_builder.hpp"
#include "model/schema/schema_nnc.h"


// TODO(yc18.cho) : implement buffer setter/getter test after buffer is defined

class FeatureMapTest : public testing::Test {
 protected:
    void SetUp() override {
        std::srand(std::time(nullptr));  // generate seed for rand
    }

    enn::model::component::FeatureMapBuilder feature_map_builder;
};

namespace enn {
namespace model {
namespace component {
// Define dummy IOperator class for test
class IOperator {
 public:
    explicit IOperator(int id) : id(id) {}
    int id;
};
};
};
};

TEST_F(FeatureMapTest, test_set_id_by_random) {
    auto id = std::rand();
    auto feature_map = feature_map_builder.set_id(id).create();
    EXPECT_EQ(feature_map->get_id(), id);
}

TEST_F(FeatureMapTest, test_set_name_by_random) {
    auto name = std::to_string(std::rand());
    auto feature_map = feature_map_builder.set_name(name).create();
    EXPECT_STREQ((feature_map->get_name()).c_str(), name.c_str());
}

TEST_F(FeatureMapTest, test_set_data_type_of_int16) {
    auto feature_map = feature_map_builder.set_data_type(TFlite::TensorType_INT16).create();
    EXPECT_EQ(feature_map->get_data_type(), TFlite::TensorType_INT16);
}

TEST_F(FeatureMapTest, test_set_shape_by_random) {
    std::vector<uint32_t> shape = { (uint32_t)std::rand(), (uint32_t)std::rand(),
                                    (uint32_t)std::rand(), (uint32_t)std::rand()};
    auto feature_map = feature_map_builder.set_shape(shape).create();
    EXPECT_EQ(feature_map->get_shape(), shape);
}

TEST_F(FeatureMapTest, test_set_buffer_size_by_random) {
    auto size = std::rand();
    auto feature_map = feature_map_builder.set_buffer_size(size).create();
    EXPECT_EQ(feature_map->get_buffer_size(), size);
}

TEST_F(FeatureMapTest, test_set_buffer_index_by_random) {
    auto index = std::rand();
    auto feature_map = feature_map_builder.set_buffer_index(index).create();
    EXPECT_EQ(feature_map->get_buffer_index(), index);
}

TEST_F(FeatureMapTest, test_set_quantization_parameters_with_dummy) {
    // Use dummy class define because TFlite::QuantizationParameters's ctr is private.
    struct A {};
    const TFlite::QuantizationParameters* quantization_parameters =
                                    reinterpret_cast<TFlite::QuantizationParameters*>(new A());
    auto feature_map = feature_map_builder.set_quantization_parameters(quantization_parameters)
                                          .create();
    EXPECT_EQ(feature_map->get_quantization_parameters(), quantization_parameters);
    delete quantization_parameters;
}

TEST_F(FeatureMapTest, test_set_symm_per_channel_quant_parameters_with_dummy) {
    // Use dummy class definding because TFlite::QuantizationParameters's ctr is private.
    struct A {};
    const TFlite::SymmPerChannelQuantParamters* per_channel_quant_parameters =
                                    reinterpret_cast<TFlite::SymmPerChannelQuantParamters*>(new A());
    auto feature_map = feature_map_builder.set_symm_per_channel_quant_parameters(per_channel_quant_parameters)
                                      .create();
    EXPECT_EQ(feature_map->get_symm_per_channel_quant_parameters(), per_channel_quant_parameters);
    delete per_channel_quant_parameters;
}

TEST_F(FeatureMapTest, test_set_prev_operator) {
    const auto expected_id = 10;
    auto op = std::make_shared<enn::model::component::IOperator>(expected_id);
    auto feature_map = feature_map_builder.set_prev_operator(op)
                                          .create();
    EXPECT_EQ(feature_map->prev()->id, expected_id);
}

TEST_F(FeatureMapTest, test_set_next_operators) {
    constexpr auto next_op_num = 100;
    std::vector<std::shared_ptr<enn::model::component::IOperator>> op_vector;
    for (int i = 0; i < next_op_num; i++) {
        auto op = std::make_shared<enn::model::component::IOperator>(i);
        op_vector.push_back(op);
        feature_map_builder.add_next_operator(op);
    }
    auto feature_map = feature_map_builder.create();
    int check_index = 0;
    for(auto& next_op : feature_map->next()) {
        EXPECT_EQ(next_op->id, op_vector[check_index++]->id);
    }
}

TEST_F(FeatureMapTest, test_set_type) {
    using namespace enn::model::component;
    auto type = FeatureMap::Type::INTERMEDIATE;
    auto feature_map = feature_map_builder.set_type(type)
                                          .create();
    EXPECT_EQ(FeatureMap::Type::INTERMEDIATE, feature_map->get_type());
}

TEST_F(FeatureMapTest, test_set_by_fluent_interface) {
    auto id = std::rand();
    auto name = std::to_string(std::rand());
    auto data_type = TFlite::TensorType_INT16;
    std::vector<uint32_t> shape = { (uint32_t)std::rand(), (uint32_t)std::rand(),
                                    (uint32_t)std::rand(), (uint32_t)std::rand()};

    auto feature_map = feature_map_builder.set_id(id)
                                          .set_name(name)
                                          .set_data_type(data_type)
                                          .set_shape(shape)
                                          .create();

    EXPECT_EQ(feature_map->get_id(), id);
    EXPECT_STREQ((feature_map->get_name()).c_str(), name.c_str());
    EXPECT_EQ(feature_map->get_data_type(), data_type);
    EXPECT_EQ(feature_map->get_shape(), shape);
}
