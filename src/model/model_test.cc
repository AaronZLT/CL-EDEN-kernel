#include <gtest/gtest.h>

#include <tuple>
#include <numeric>
#include <ctime>
#include <cstdlib>
#include <vector>

#include "model/model.hpp"
#include "runtime/client_process/client_process.hpp"

using namespace enn::model;
using namespace enn::runtime;
using namespace enn::identifier;

class ModelTest : public testing::Test {
 protected:
    void SetUp() override {
        std::srand(std::time(nullptr));  // generate seed for rand
        client_process = std::make_shared<ClientProcess>();
    }

    ClientProcess::Ptr client_process;
};


TEST_F(ModelTest, create_model) {
    auto model = std::make_shared<Model>(client_process);
    EXPECT_TRUE(model->get_id().equal_in_common(client_process->get_id()));
}

TEST_F(ModelTest, create_and_release_10000_times) {
    constexpr int iter = 10000;
    for (int i = 0; i < iter; i++) {
        // Model is releasd on getting out of scope here.
        EXPECT_NO_THROW(std::make_shared<Model>(client_process));
    }
}

TEST_F(ModelTest, throws_on_create_with_id_over) {
    constexpr int max = Model::UniqueID::Max;

    // add objects created to prevent from being released
    std::vector<Model::Ptr> em_list;

    // create Model objects by upper bound
    for (int i = 1; i <= max; i++) {
        em_list.push_back(std::make_shared<Model>(client_process));
    }
    // create one more Model object with id that exceeds the max limit.
    EXPECT_THROW(std::make_shared<Model>(client_process), std::runtime_error);
}
