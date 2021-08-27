#include <gtest/gtest.h>

#include <string>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <memory>

#include "model/component/tensor/parameter_builder.hpp"
#include "model/schema/schema_nnc.h"


// TODO(yc18.cho) : implement buffer setter/getter test after buffer is defined

class ParameterTest : public testing::Test {
 protected:
    void SetUp() override {
        std::srand(std::time(nullptr));  // generate seed for rand
    }

    enn::model::component::ParameterBuilder parameter_builder;
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

TEST_F(ParameterTest, test_set_id_by_random) {
    auto id = std::rand();
    auto parameter = parameter_builder.set_id(id).create();
    EXPECT_EQ(parameter->get_id(), id);
}

TEST_F(ParameterTest, test_set_name_by_random) {
    auto name = std::to_string(std::rand());
    auto parameter = parameter_builder.set_name(name).create();
    EXPECT_STREQ((parameter->get_name()).c_str(), name.c_str());
}

TEST_F(ParameterTest, test_set_data_type_of_int16) {
    auto parameter = parameter_builder.set_data_type(TFlite::TensorType_INT16).create();
    EXPECT_EQ(parameter->get_data_type(), TFlite::TensorType_INT16);
}

TEST_F(ParameterTest, test_set_shape_by_random) {
    std::vector<uint32_t> shape = { (uint32_t)std::rand(), (uint32_t)std::rand(),
                                    (uint32_t)std::rand(), (uint32_t)std::rand()};
    auto parameter = parameter_builder.set_shape(shape).create();
    EXPECT_EQ(parameter->get_shape(), shape);
}

TEST_F(ParameterTest, test_set_buffer_with_dummy) {
    auto fd = std::rand();
    auto offset = std::rand();
    struct Dummy {
        int check = 10;
    };
    Dummy* dummy = new Dummy{};
    auto parameter = parameter_builder.set_buffer_addr(dummy)
                                      .set_buffer_size(sizeof(Dummy))
                                      .set_buffer_fd(fd)
                                      .set_buffer_offset(offset)
                                      .create();
    EXPECT_EQ(parameter->get_buffer_addr(), dummy);
    EXPECT_EQ(parameter->get_buffer_size(), sizeof(Dummy));
    EXPECT_EQ(parameter->get_buffer_fd(), fd);
    EXPECT_EQ(parameter->get_buffer_offset(), offset);
    delete dummy;
}

TEST_F(ParameterTest, test_set_quantization_parameters_with_dummy) {
    // Use dummy class definding because TFlite::QuantizationParameters's ctr is private.
    struct A {};
    const TFlite::QuantizationParameters* quantization_parameters =
                                    reinterpret_cast<TFlite::QuantizationParameters*>(new A());
    auto parameter = parameter_builder.set_quantization_parameters(quantization_parameters)
                                      .create();
    EXPECT_EQ(parameter->get_quantization_parameters(), quantization_parameters);
    delete quantization_parameters;
}

TEST_F(ParameterTest, test_set_symm_per_channel_quant_parameters_with_dummy) {
    // Use dummy class definding because TFlite::QuantizationParameters's ctr is private.
    struct A {};
    const TFlite::SymmPerChannelQuantParamters* per_channel_quant_parameters =
                                    reinterpret_cast<TFlite::SymmPerChannelQuantParamters*>(new A());
    auto parameter = parameter_builder.set_symm_per_channel_quant_parameters(per_channel_quant_parameters)
                                      .create();
    EXPECT_EQ(parameter->get_symm_per_channel_quant_parameters(), per_channel_quant_parameters);
    delete per_channel_quant_parameters;
}

TEST_F(ParameterTest, test_set_next_operators) {
    constexpr auto next_op_num = 100;
    std::vector<std::shared_ptr<enn::model::component::IOperator>> op_vector;
    for (int i = 0; i < next_op_num; i++) {
        auto op = std::make_shared<enn::model::component::IOperator>(i);
        op_vector.push_back(op);
        parameter_builder.add_next_operator(op);
    }
    auto parameter = parameter_builder.create();
    int check_index = 0;
    for(auto& next_op : parameter->next()) {
        EXPECT_EQ(next_op->id, op_vector[check_index++]->id);
    }
}

TEST_F(ParameterTest, test_set_by_fluent_interface) {
    auto id = std::rand();
    auto name = std::to_string(std::rand());
    auto data_type = TFlite::TensorType_INT16;
    std::vector<uint32_t> shape = { (uint32_t)std::rand(), (uint32_t)std::rand(),
                                    (uint32_t)std::rand(), (uint32_t)std::rand()};

    auto parameter = parameter_builder.set_id(id)
                                          .set_name(name)
                                          .set_data_type(data_type)
                                          .set_shape(shape)
                                          .create();

    EXPECT_EQ(parameter->get_id(), id);
    EXPECT_STREQ((parameter->get_name()).c_str(), name.c_str());
    EXPECT_EQ(parameter->get_data_type(), data_type);
    EXPECT_EQ(parameter->get_shape(), shape);
}
