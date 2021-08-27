#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <string>

#include "runtime/scheduler/static_scheduler.hpp"

#include "model/types.hpp"
#include "model/model.hpp"
#include "model/graph/graph.hpp"
#include "model/parser/parser.hpp"
#include "model/generator/generator.hpp"
#include "model/graph/iterator/methods/topological_sort.hpp"

#include "common/enn_debug.h"
#include "common/enn_utils.h"
#include "common/identifier.hpp"
#include "test/materials.h"

using Model = enn::model::Model;
using RawModel = enn::model::raw::Model;
using Parser = enn::model::Parser;

#ifdef ENN_ANDROID_BUILD
#include "common/enn_memory_manager.h"
#endif  // ENN_ANDROID_BUILD

using namespace enn::identifier;

class StaticSchedulerTest : public testing::Test {
protected:
    void SetUp() override {
        client_process = std::make_shared<enn::runtime::ClientProcess>();
    }

    void model_setting(const enn::model::ModelType& model_type, const char* model_file) {
        // Pre-Setting
        // - Create Raw Model
        uint32_t file_size;
        enn::util::get_file_size(model_file, &file_size);
        std::unique_ptr<char[]> buffer{new char[file_size]};
        {
            std::ifstream is;
            is.open(model_file, std::ios::binary);
            is.read(buffer.get(), file_size);
            is.close();
        }
        std::unique_ptr<Parser> parser = std::make_unique<Parser>();
        std::shared_ptr<enn::model::ModelMemInfo> model_mem_info =
            std::make_shared<enn::model::ModelMemInfo>(buffer.get(), 0, file_size);
        parser->Set(model_type, model_mem_info, nullptr);
        raw_model = parser->Parse();

        std::unique_ptr<enn::model::Generator> model_generator = std::make_unique<enn::model::Generator>();
        enn_model = model_generator->generate_model(raw_model, client_process);
    }

    enn::runtime::ClientProcess::Ptr client_process;
    Model::Ptr enn_model;
    std::shared_ptr<RawModel> raw_model;
};

void check_operator_lists(Model::Ptr& model, std::vector<enn::model::Accelerator> expected_result) {
    int count = 0;
    for (auto& opr_list_ptr : model->get_scheduled_graph()->order<enn::model::graph::iterator::TopologicalSort>()) {
        EXPECT_TRUE(available_accelerator(opr_list_ptr->get_accelerator(), expected_result.at(count)));
        ++count;
    }
}

// TODO(daewhan.kim): Add more TCs
TEST_F(StaticSchedulerTest, static_shcedule_iv3_nnc_v2) {
    std::string model_file = PAMIR::NPU::IV3::NNC;
    model_setting(enn::model::ModelType::NNC, model_file.c_str());

    enn::runtime::schedule::StaticScheduler static_scheduler;
    static_scheduler.set_model(enn_model);
    static_scheduler.run();

    std::vector<enn::model::Accelerator> expected_result = {enn::model::Accelerator::CPU, enn::model::Accelerator::NPU,
                                                            enn::model::Accelerator::CPU};
    check_operator_lists(enn_model, expected_result);
}


TEST_F(StaticSchedulerTest, static_shcedule_mobile_bert_nnc) {
    std::string model_file = PAMIR::GPU::MobileBERT::NNC;
    model_setting(enn::model::ModelType::NNC, model_file.c_str());

    enn::runtime::schedule::StaticScheduler static_scheduler;
    static_scheduler.set_model(enn_model);
    static_scheduler.run();

    std::vector<enn::model::Accelerator> expected_result = {enn::model::Accelerator::GPU};
    check_operator_lists(enn_model, expected_result);
}
