#include <gtest/gtest.h>
#include <cstdlib>
#include <ctime>
#include <numeric>
#include <map>
#include <algorithm>

#include "runtime/pool/manager.hpp"
#include "common/helper_templates.hpp"
#include "runtime/client_process/client_process.hpp"

using namespace enn::runtime::pool;
using namespace enn::model;
using namespace enn::runtime;
using namespace enn::identifier;
using ModelMap = std::map<std::reference_wrapper<const Model::ID>,
                          Model::Ptr,
                          std::less<const Model::ID>>;
using ExecutableModelMap = std::map<ExecutableModel::ID, Model::Ptr>;

class ModelPoolManagerTest : public testing::Test {
 protected:
    void SetUp() override {
        client_process = std::make_shared<ClientProcess>();
    }

    void create_and_set_models(uint32_t model_count,
                       ModelMap& id_to_model_map) {
        for (uint32_t i = 0; i < model_count; i++) {
            auto model = std::make_shared<Model>(client_process);
            id_to_model_map[model->get_id()] = model;
        }
    }

    void add_model_to_pool(const ModelMap& id_to_model_map) {
        // add model object with id to model pool via model pool manager.
        for (const auto& it : id_to_model_map) {
            pool_manager.add<Model>(it.second);
        }
    }

    ClientProcess::Ptr client_process;
    Manager pool_manager;
};

TEST_F(ModelPoolManagerTest, throws_on_add_model_without_adding_client_process) {
    auto model = std::make_shared<Model>(client_process);
    // It should throw an exception not to add the ClientProcess object into the pool
    //  before adding Model from that.
    EXPECT_THROW(pool_manager.add<Model>(model), std::runtime_error);
}

TEST_F(ModelPoolManagerTest, throws_on_add_executable_model_without_adding_model) {
    pool_manager.add<ClientProcess>(client_process);
    auto model = std::make_shared<Model>(client_process);
    auto executable_model = ExecutableModel::create(model);
    // It should throw an exception not to add the Model object into the pool
    //  before adding ExecutableModel from that.
    EXPECT_THROW(pool_manager.add<ExecutableModel>(executable_model), std::runtime_error);
}

TEST_F(ModelPoolManagerTest, throws_on_add_execute_request_without_adding_executable_model) {
    pool_manager.add<ClientProcess>(client_process);
    auto model = std::make_shared<Model>(client_process);
    pool_manager.add<Model>(model);
    auto executable_model = ExecutableModel::create(model);
    ExecuteRequest::Ptr execute_request = ExecuteRequest::create(executable_model);
    // It should throw an exception not to add the ExecutableModel object into the pool
    //  before adding ExecuteRequest from that.
    EXPECT_THROW(pool_manager.add<ExecuteRequest>(execute_request), std::runtime_error);
}

TEST_F(ModelPoolManagerTest, throws_on_release_model_by_invalid_id) {
    pool_manager.add<ClientProcess>(client_process);
    auto model = std::make_shared<Model>(client_process);
    EXPECT_THROW(pool_manager.release<Model>(model->get_id().get()), std::runtime_error);
}

TEST_F(ModelPoolManagerTest, throws_on_release_executable_model_by_invalid_id) {
    pool_manager.add<ClientProcess>(client_process);
    auto model = std::make_shared<Model>(client_process);
    pool_manager.add<Model>(model);
    auto executable_model = ExecutableModel::create(model);
    EXPECT_THROW(pool_manager.release<ExecutableModel>(executable_model->get_id().get()), std::runtime_error);
}

TEST_F(ModelPoolManagerTest, throws_on_release_execute_request_by_invalid_id) {
    pool_manager.add<ClientProcess>(client_process);
    auto model = std::make_shared<Model>(client_process);
    pool_manager.add<Model>(model);
    auto executable_model = ExecutableModel::create(model);
    pool_manager.add<ExecutableModel>(executable_model);
    auto execute_request = ExecuteRequest::create(executable_model);
    // Use the ID of ExecutableModel becaues they are same.
    EXPECT_THROW(pool_manager.release<ExecuteRequest>(executable_model->get_id().get()), std::runtime_error);
}

TEST_F(ModelPoolManagerTest, throws_on_get_model_by_invalid_id) {
    pool_manager.add<ClientProcess>(client_process);
    uint64_t invalid_model_id = 0xAAAA;
    EXPECT_THROW(pool_manager.get<Model>(invalid_model_id), std::runtime_error);
}

TEST_F(ModelPoolManagerTest, throws_on_get_executable_model_by_invalid_id) {
    pool_manager.add<ClientProcess>(client_process);
    pool_manager.add<Model>(std::make_shared<Model>(client_process));
    uint64_t invalid_executable_model_id = 0xAAAA;
    EXPECT_THROW(pool_manager.get<ExecutableModel>(invalid_executable_model_id), std::runtime_error);
}

TEST_F(ModelPoolManagerTest, throws_on_get_execute_request_by_invalid_id) {
    pool_manager.add<ClientProcess>(client_process);
    auto model = std::make_shared<Model>(client_process);
    pool_manager.add<Model>(model);
    pool_manager.add<ExecutableModel>(ExecutableModel::create(model));
    uint64_t invalid_execute_request_id = 0xAAAA;
    EXPECT_THROW(pool_manager.get<ExecuteRequest>(invalid_execute_request_id), std::runtime_error);
}

TEST_F(ModelPoolManagerTest, test_add_random_models_to_pool) {
    auto pid = client_process->get_id().get();
    ModelMap id_to_model_map;
    create_and_set_models(10, id_to_model_map);
    pool_manager.add<ClientProcess>(client_process);
    add_model_to_pool(id_to_model_map);
    // get model from model pool manager and check equality
    for (const auto& it : id_to_model_map) {
        Model::Ptr model = pool_manager.get<Model>(static_cast<uint64_t>(it.first.get()));
        EXPECT_EQ(model->get_id(), it.first);
    }
}

TEST_F(ModelPoolManagerTest, test_release_models_from_pool) {
    ModelMap id_to_model_map;
    auto cnt = 10;
    create_and_set_models(cnt, id_to_model_map);
    pool_manager.add<ClientProcess>(client_process);
    // add model object with id to model pool via model pool manager.
    add_model_to_pool(id_to_model_map);
    // get model from model pool manager and check equality
    for (const auto& it : id_to_model_map) {
        auto model_id = it.first;
        pool_manager.release<Model>(static_cast<uint64_t>(model_id.get()));
        EXPECT_THROW(pool_manager.get<Model>(static_cast<uint64_t>(model_id.get())), std::runtime_error);
        EXPECT_EQ(pool_manager.count<Model>(), --cnt);
    }
}

TEST_F(ModelPoolManagerTest, test_get_model_with_invalid_argument) {
    ModelMap id_to_model_map;
    create_and_set_models(10, id_to_model_map);
    pool_manager.add<ClientProcess>(client_process);
    add_model_to_pool(id_to_model_map);
    for (const auto& it : id_to_model_map) {
        auto invalid_id = Model::UniqueID(client_process->get_id());
        EXPECT_NE(it.first.get(), invalid_id);
        EXPECT_THROW(pool_manager.get<Model>(static_cast<uint64_t>(invalid_id)), std::runtime_error);
    }
}

TEST_F(ModelPoolManagerTest, test_add_and_release_models_with_multi_threads) {
    pool_manager.add<ClientProcess>(client_process);
    auto add_and_release_func = [&](uint64_t start_id, uint64_t end_id) {
        std::vector<std::reference_wrapper<const Model::ID>> model_id_list;
        // create models with id of the range start_id ~ end_id.
        for (uint64_t model_id = start_id ; model_id <= end_id; model_id++) {
            auto model = std::make_shared<Model>(client_process);
            // add models to pool in model::pool::Manager
            model_id_list.push_back(model->get_id());
            pool_manager.add<Model>(model);
        }

        // release models from pool in model::pool::Manager
        for (const auto model_id : model_id_list) {
            pool_manager.release<Model>(static_cast<uint64_t>(model_id.get()));
        }
    };

    std::vector<std::future<void>> future_list;
    static const int thread_count = 5;
    static const int models_per_thread = 10;
    // create models per thread.
    for (int i = 0; i < thread_count; i++) {
        future_list.push_back(enn::util::RunAsync(add_and_release_func, 1 + i * models_per_thread,
                                                  (1 + i) * models_per_thread));
    }

    // wait for thread exit with future.get()
    for (auto& future : future_list) {
        future.get();
    }
    // The pool has nothing if threads work without data race.
    EXPECT_EQ(pool_manager.count<Model>(), 0);
}

TEST_F(ModelPoolManagerTest, test_add_executable_models_to_pool) {
    pool_manager.add<ClientProcess>(client_process);
    std::vector<Model::Ptr> model_list;
    for (uint64_t model_id = 1; model_id <= 10; model_id++) {
        auto model = std::make_shared<Model>(client_process);
        pool_manager.add<Model>(model);  // add to pool
        model_list.push_back(model);
    }

    std::vector<std::reference_wrapper<const ExecutableModel::ID>> exec_model_id_vec;
    // Create 5 ExecutableModels for each Model.
    for (auto& model : model_list)
        for (int exec_model_cnt = 1; exec_model_cnt <= 5; exec_model_cnt++) {
            ExecutableModel::Ptr executable_model = ExecutableModel::create(model);
            exec_model_id_vec.push_back(executable_model->get_id());
            pool_manager.add<ExecutableModel>(std::move(executable_model));  // add to pool
        }

    for (auto exec_model_id : exec_model_id_vec) {
        auto exec_model = pool_manager.get<ExecutableModel>(static_cast<uint64_t>(exec_model_id.get()));
        EXPECT_EQ(exec_model->get_id(), exec_model_id);
    }
}

TEST_F(ModelPoolManagerTest, test_release_executable_models_from_pool) {
    pool_manager.add<ClientProcess>(client_process);
    // Create Models with id is 1~10;
    std::vector<Model::Ptr> model_list;
    for (uint64_t model_id = 1; model_id <= 10; model_id++) {
        auto model = std::make_shared<Model>(client_process);
        pool_manager.add<Model>(model);  // add to pool
        model_list.push_back(model);
    }

    std::vector<std::reference_wrapper<const ExecutableModel::ID>> exec_model_id_vec;
    // Create 5 ExecutableModels for each Model.
    for (auto& model : model_list)
        for (int exec_model_cnt = 1; exec_model_cnt <= 5; exec_model_cnt++) {
            ExecutableModel::Ptr executable_model = ExecutableModel::create(model);
            exec_model_id_vec.push_back(executable_model->get_id());
            pool_manager.add<ExecutableModel>(std::move(executable_model));  // add to pool
        }
    // release ExecutableModels
    for (auto exec_model_id : exec_model_id_vec) {
        pool_manager.release<ExecutableModel>(static_cast<uint64_t>(exec_model_id.get()));
    }
    EXPECT_EQ(pool_manager.count<ExecutableModel>(), 0);
}

TEST_F(ModelPoolManagerTest, test_cascade_release_by_process_id) {
    pool_manager.add<ClientProcess>(client_process);

    std::vector<Model::Ptr> model_vec;
    auto model_cnt = 2;
    for (int i = 0; i < model_cnt; i++) {
        model_vec.push_back(std::make_shared<Model>(client_process));
        pool_manager.add<Model>(model_vec[i]);
    }
    auto exec_model_cnt = 5;  // the number of ExecutableModel to create per a Model.
    for (auto& model : model_vec) {
        for (int i = 0; i < exec_model_cnt; i++) {
            ExecutableModel::Ptr exec_model = ExecutableModel::create(model);
            pool_manager.add<ExecutableModel>(exec_model);
        }
    }
    EXPECT_EQ(pool_manager.count<ClientProcess>(), 1);
    EXPECT_EQ(pool_manager.count<Model>(), 2);
    EXPECT_EQ(pool_manager.count<ExecutableModel>(), 10);

    pool_manager.release<ClientProcess>(static_cast<uint64_t>(client_process->get_id()));
    EXPECT_EQ(pool_manager.count<ClientProcess>(), 0);
    EXPECT_EQ(pool_manager.count<Model>(), 0);
    EXPECT_EQ(pool_manager.count<ExecutableModel>(), 0);
}

TEST_F(ModelPoolManagerTest, test_cascade_release_by_model_id) {
    pool_manager.add<ClientProcess>(client_process);
    std::vector<Model::Ptr> model_vec;
    auto model_cnt = 2;
    for (int i = 0; i < model_cnt; i++) {
        model_vec.push_back(std::make_shared<Model>(client_process));
        pool_manager.add<Model>(model_vec[i]);
    }
    auto exec_model_cnt = 5;  // the number of ExecutableModel to create per a Model.
    for (auto& model : model_vec) {
        for (int i = 0; i < exec_model_cnt; i++) {
            ExecutableModel::Ptr exec_model = ExecutableModel::create(model);
            pool_manager.add<ExecutableModel>(exec_model);
        }
    }

    EXPECT_EQ(pool_manager.count<ClientProcess>(), 1);
    EXPECT_EQ(pool_manager.count<Model>(), 2);
    EXPECT_EQ(pool_manager.count<ExecutableModel>(), 10);

    pool_manager.release<Model>(static_cast<uint64_t>(model_vec[0]->get_id()));
    EXPECT_EQ(pool_manager.count<ClientProcess>(), 1);
    EXPECT_EQ(pool_manager.count<Model>(), 1);
    EXPECT_EQ(pool_manager.count<ExecutableModel>(), 5);

    pool_manager.release<Model>(static_cast<uint64_t>(model_vec[1]->get_id()));
    EXPECT_EQ(pool_manager.count<ClientProcess>(), 1);
    EXPECT_EQ(pool_manager.count<Model>(), 0);
    EXPECT_EQ(pool_manager.count<ExecutableModel>(), 0);
}

TEST_F(ModelPoolManagerTest, test_cascade_release_by_client_process_id) {
    pool_manager.add<ClientProcess>(client_process);
    std::vector<Model::Ptr> model_vec;
    auto model_cnt = 2;
    for (int i = 0; i < model_cnt; i++) {
        model_vec.push_back(std::make_shared<Model>(client_process));
        pool_manager.add<Model>(model_vec[i]);
    }
    auto exec_model_cnt = 5;  // the number of ExecutableModel to create per a Model.
    for (auto& model : model_vec) {
        for (int i = 0; i < exec_model_cnt; i++) {
            ExecutableModel::Ptr exec_model = ExecutableModel::create(model);
            pool_manager.add<ExecutableModel>(exec_model);
        }
    }

    EXPECT_EQ(pool_manager.count<ClientProcess>(), 1);
    EXPECT_EQ(pool_manager.count<Model>(), 2);
    EXPECT_EQ(pool_manager.count<ExecutableModel>(), 10);

    pool_manager.release<ClientProcess>(static_cast<uint64_t>(client_process->get_id()));
    EXPECT_EQ(pool_manager.count<ClientProcess>(), 0);
    EXPECT_EQ(pool_manager.count<Model>(), 0);
    EXPECT_EQ(pool_manager.count<ExecutableModel>(), 0);
}

