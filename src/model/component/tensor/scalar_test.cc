#include <gtest/gtest.h>

#include <string>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <memory>

#include "model/component/tensor/scalar_builder.hpp"
#include "model/schema/schema_nnc.h"
#include "model/component/tensor/parameter_builder.hpp"
class ScalarTest : public testing::Test {
 protected:
    void SetUp() override {
        std::srand(std::time(nullptr));
    }
    enn::model::component::ScalarBuilder scalar_builder;
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
};  // namespace component
};  // namespace model
};  // namespace enn

TEST_F(ScalarTest, test_set_id_by_random) {
    auto id = std::rand();
    auto scalar = scalar_builder.set_id(id).create();
    EXPECT_EQ(scalar->get_id(), id);
}

TEST_F(ScalarTest, test_set_name_by_random) {
    auto name = std::to_string(std::rand());
    auto scalar = scalar_builder.set_name(name).create();
    EXPECT_STREQ((scalar->get_name()).c_str(), name.c_str());
}

TEST_F(ScalarTest, test_set_data_type_of_int16) {
    auto scalar = scalar_builder.set_data_type(TFlite::TensorType_INT16).create();
    EXPECT_EQ(scalar->get_data_type(), TFlite::TensorType_INT16);
}

TEST_F(ScalarTest, test_set_shape_by_random) {
    std::vector<uint32_t> shape = { (uint32_t)std::rand(), (uint32_t)std::rand(),
                                    (uint32_t)std::rand(), (uint32_t)std::rand()};
    auto scalar = scalar_builder.set_shape(shape).create();
    EXPECT_EQ(scalar->get_shape(), shape);
}

TEST_F(ScalarTest, test_set_default_buffer_with_dummy) {
    auto fd = std::rand();
    auto offset = std::rand();
    struct Dummy {
        int check = 10;
    };
    Dummy* dummy = new Dummy{};
    auto scalar = scalar_builder.set_default_buffer_addr(dummy)
                                .set_default_buffer_size(sizeof(Dummy))
                                .set_default_buffer_fd(fd)
                                .set_default_buffer_offset(offset)
                                .create();
    EXPECT_EQ(scalar->get_default_buffer_addr(), dummy);
    EXPECT_EQ(scalar->get_default_buffer_size(), sizeof(Dummy));
    EXPECT_EQ(scalar->get_default_buffer_fd(), fd);
    EXPECT_EQ(scalar->get_default_buffer_offset(), offset);
    delete dummy;
}

TEST_F(ScalarTest, test_set_indexed_buffer_size_by_random) {
    auto size = std::rand();
    auto scalar = scalar_builder.set_indexed_buffer_size(size).create();
    EXPECT_EQ(scalar->get_indexed_buffer_size(), size);
}

TEST_F(ScalarTest, test_set_indexed_buffer_index_by_random) {
    auto index = std::rand();
    auto scalar = scalar_builder.set_indexed_buffer_index(index).create();
    EXPECT_EQ(scalar->get_indexed_buffer_index(), index);
}

TEST_F(ScalarTest, test_set_quantization_parameters_with_dummy) {
    // Use dummy class definding because TFlite::QuantizationParameters's ctr is private.
    struct A {};
    const TFlite::QuantizationParameters* quantization_parameters =
                                    reinterpret_cast<TFlite::QuantizationParameters*>(new A());
    auto scalar = scalar_builder.set_quantization_parameters(quantization_parameters)
                                      .create();
    EXPECT_EQ(scalar->get_quantization_parameters(), quantization_parameters);
    delete quantization_parameters;
}

TEST_F(ScalarTest, test_set_next_operators) {
    constexpr auto next_op_num = 100;
    std::vector<std::shared_ptr<enn::model::component::IOperator>> op_vector;
    for (int i = 0; i < next_op_num; i++) {
        auto op = std::make_shared<enn::model::component::IOperator>(i);
        op_vector.push_back(op);
        scalar_builder.add_next_operator(op);
    }
    auto scalar = scalar_builder.create();
    int check_index = 0;
    for (auto& next_op : scalar->next()) {
        EXPECT_EQ(next_op->id, op_vector[check_index++]->id);
    }
}

TEST_F(ScalarTest, test_set_by_fluent_interface) {
    auto id = std::rand();
    auto name = std::to_string(std::rand());
    auto data_type = TFlite::TensorType_INT16;
    std::vector<uint32_t> shape = { (uint32_t)std::rand(), (uint32_t)std::rand(),
                                    (uint32_t)std::rand(), (uint32_t)std::rand()};

    auto scalar = scalar_builder.set_id(id)
                                .set_name(name)
                                .set_data_type(data_type)
                                .set_shape(shape)
                                .create();

    EXPECT_EQ(scalar->get_id(), id);
    EXPECT_STREQ((scalar->get_name()).c_str(), name.c_str());
    EXPECT_EQ(scalar->get_data_type(), data_type);
    EXPECT_EQ(scalar->get_shape(), shape);
}

TEST_F(ScalarTest, test_dynamic_pointer_cast) {
    enn::model::component::Tensor::Ptr tensor_container_for_scalar;
    tensor_container_for_scalar = scalar_builder.set_id(111)
                                .set_name("test_for_dynamic_cast")
                                .set_data_type(TFlite::TensorType_FLOAT16)
                                .set_indexed_buffer_size(100)
                                .set_default_buffer_size(200)
                                .create();
    auto scalar = std::dynamic_pointer_cast<enn::model::component::Scalar>(tensor_container_for_scalar);

    EXPECT_EQ(tensor_container_for_scalar->get_id(), 111);
    EXPECT_EQ(scalar->get_default_buffer_size(), 200);
    EXPECT_EQ(scalar->get_indexed_buffer_size(), 100);
}

