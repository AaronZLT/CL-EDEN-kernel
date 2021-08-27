#include <gtest/gtest.h>

#include <tuple>
#include <numeric>
#include <ctime>
#include <cstdlib>
#include <vector>

#include "runtime/executable_model/executable_operator_list.hpp"
#include "runtime/executable_model/executable_model.hpp"
#include "runtime/client_process/client_process.hpp"
#include "model/model.hpp"
#include "model/component/operator/operator_list_builder.hpp"

using namespace enn::model;
using namespace enn::runtime;
using namespace enn::identifier;
using namespace enn::model::component;
using namespace enn::model::memory;

class ExecutableOperatorListTest : public testing::Test {
 protected:
    void SetUp() override {
        auto model = std::make_shared<Model>(std::make_shared<ClientProcess>());
        OperatorListBuilder operator_list_builder;
        operator_list = operator_list_builder.build(model->get_id())
                                             .create();
        executable_model = ExecutableModel::create(model);
    }

    OperatorList::Ptr operator_list;
    ExecutableModel::Ptr executable_model;
};


TEST_F(ExecutableOperatorListTest, create_executable_operator_list) {
    auto executable_operator_list = std::make_shared<ExecutableOperatorList>(executable_model->get_id(),
                                                                             operator_list,
                                                                             std::make_shared<BufferTable>());
    EXPECT_EQ(executable_operator_list->get_operator_list_id(), operator_list->get_id());
    EXPECT_TRUE(executable_operator_list->get_id().equal_in_common(operator_list->get_id()));
}

TEST_F(ExecutableOperatorListTest, create_and_release_10000_times) {
    constexpr int iter = 10000;
    for (int i = 0; i < iter; i++) {
        // ExecutableOperatorList is releasd on getting out of scope here.
        EXPECT_NO_THROW(std::make_shared<ExecutableOperatorList>(executable_model->get_id(),
                                                                 operator_list,
                                                                 std::make_shared<BufferTable>()));
    }
}
