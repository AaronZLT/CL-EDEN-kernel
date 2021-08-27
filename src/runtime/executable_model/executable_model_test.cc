#include <gtest/gtest.h>

#include <tuple>
#include <numeric>
#include <ctime>
#include <cstdlib>
#include <vector>

#include "runtime/executable_model/executable_model.hpp"
#include "runtime/client_process/client_process.hpp"

using namespace enn::model;
using namespace enn::runtime;
using namespace enn::identifier;

class ExecutableModelTest : public testing::Test {
 protected:
    void SetUp() override {
        std::srand(std::time(nullptr));  // generate seed for rand
        client_process = std::make_shared<ClientProcess>();
        model = std::make_shared<Model>(client_process);
    }

    ClientProcess::Ptr client_process;
    Model::Ptr model;
};


TEST_F(ExecutableModelTest, test_set_and_get_model_with_object) {
    auto executable_model = ExecutableModel::create(model);
    EXPECT_EQ(model->get_id(), executable_model->get_model()->get_id());
}

TEST_F(ExecutableModelTest, test_set_and_get_model_with_id) {
    auto executable_model = ExecutableModel::create(model);
    // Masking with Model Unique ID
    auto& model_uid = model->get_id();
    auto& model_uid_from_exec_model = executable_model->get_id();
    EXPECT_TRUE(model_uid.equal_in_common(model_uid_from_exec_model));
}

TEST_F(ExecutableModelTest, create_and_release_10000_times) {
    constexpr int iter = 10000;

    for (int i = 0; i < iter; i++) {
        // ExecutableModel is releasd on getting out of scope here.
        EXPECT_NO_THROW(ExecutableModel::create(model));
    }
}

TEST_F(ExecutableModelTest, throws_on_create_with_id_over) {
    constexpr int max = ExecutableModel::UniqueID::Max;

    // add objects created to prevent from being released
    std::vector<ExecutableModel::Ptr> em_list;

    // create ExecutableModel objects by upper bound
    for (int i = 1; i <= max; i++) {
        em_list.push_back(ExecutableModel::create(model));
    }
    // create one more ExecutableModel object with id that exceeds the max limit.
    EXPECT_THROW(ExecutableModel::create(model), std::runtime_error);
}

// The unit test for building buffer table woudl be enabled after
//  BufferCore and BufferMetadata are refined.

// TEST_F(ExecutableModelTest, test_add_buffers) {
//     // Dummmy class it is responded to buffer's data.
//     struct Data {
//         int index;
//         std::string content;
//         Data(int index)
//             : index{index}, content{std::to_string(index)} {}
//     };

//     // struct that keeps buffer's information that is allocated by struct Data.
//     struct BufferForData {
//         int32_t index;
//         const void* addr;
//         size_t size;
//         BufferForData(int32_t index, const void* addr, size_t size)
//             : index{index}, addr{addr}, size{size} {}
//     };

//     constexpr auto buffer_count = 100;
//     std::vector<BufferForData> buffer_data_vec;

//     for (int index = 0; index < buffer_count; index++) {
//         const void* data_addr = new Data(std::rand());
//         buffer_data_vec.push_back({index, data_addr, sizeof(Data)});
//     }
//     auto executable_model = ExecutableModel::create(create_model());

//     // Build ExecutableModel with multiple Buffers.
//     for (auto& buffer_data : buffer_data_vec) {
//         executable_model->add_buffer(buffer_data.index, buffer_data.addr, buffer_data.size);
//     }

//     // verification
//     auto buffer_table = executable_model->get_buffer_table();
//     for (int index = 0; index < buffer_count; index++) {
//         auto addr = (*buffer_table)[index].get_addr();
//         auto size = (*buffer_table)[index].get_size();
//         EXPECT_EQ(addr, buffer_data_vec[index].addr);
//         EXPECT_EQ(size, buffer_data_vec[index].size);
//     }
// }