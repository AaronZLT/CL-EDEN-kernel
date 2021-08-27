#include <gtest/gtest.h>

#include "runtime/execute_request/execute_request.hpp"

#include "model/model.hpp"
#include "runtime/executable_model/executable_model.hpp"
#include "runtime/client_process/client_process.hpp"

using namespace enn::model;
using namespace enn::runtime;
using namespace enn::runtime::execute;
using namespace enn::identifier;

class ExecuteRequestTest : public testing::Test {
 protected:
    void SetUp() override {
        std::srand(std::time(nullptr));  // generate seed for rand
        client_process = std::make_shared<ClientProcess>();
    }

    auto create_model() {
        return std::make_unique<Model>(client_process);
    }

    auto create_executable_model(Model::Ptr model) {
        return ExecutableModel::create(model);
    }

    ClientProcess::Ptr client_process;
};

TEST_F(ExecuteRequestTest, check_sanity) {
    Model::Ptr model = create_model();
    std::cout << model->get_id() << std::endl;
    ExecutableModel::Ptr executable_model = create_executable_model(model);
    std::cout << executable_model->get_id() << std::endl;
    ExecuteRequest::Ptr execute_request = ExecuteRequest::create(executable_model);
    EXPECT_EQ(execute_request->get_executable_model()->get_id(), executable_model->get_id());
}
