#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <string>

#include "model/generator/generator.hpp"
#include "model/types.hpp"
#include "model/model.hpp"
#include "model/graph/graph.hpp"
#include "model/parser/parser.hpp"
#include "model/parser/parser_utils.hpp"
#include "model/raw/model.hpp"
#include "model/component/operator/ioperator.hpp"
#include "model/component/operator/operator.hpp"
#include "model/component/tensor/feature_map.hpp"
#include "model/graph/iterator/methods/topological_sort.hpp"

#include "common/enn_debug.h"
#include "common/enn_utils.h"
#include "common/enn_memory_manager.h"
#include "client/enn_client_parser.hpp"
#include "medium/enn_medium_utils.hpp"
#include "test/materials.h"

using Model = enn::model::Model;
using RawModel = enn::model::raw::Model;
using Parser = enn::model::Parser;
using IOperator = enn::model::component::IOperator;
using FeatureMap = enn::model::component::FeatureMap;

#ifdef ENN_ANDROID_BUILD
#include "common/enn_memory_manager.h"
#endif  // ENN_ANDROID_BUILD

using namespace enn::identifier;

class ModelGeneratorTest : public testing::Test {
    using MemInfos = std::shared_ptr<std::vector<std::shared_ptr<enn::model::ModelMemInfo>>>;

public:
    ModelGeneratorTest() {
        emm = std::make_unique<enn::EnnMemoryManager>();
        emm->init();
    }
    ~ModelGeneratorTest() {
        emm->deinit();
    }

protected:
    void SetUp() override {
        client_process = std::make_shared<enn::runtime::ClientProcess>();
    }

    void raw_model_setting(const enn::model::ModelType& model_type, const std::string model_file,
                           const int real_file_size = 0, const uint32_t model_offset = 0) {
        // Pre-Setting
        // - Create Raw Model
        uint32_t file_size;
        enn::util::get_file_size(model_file.c_str(), &file_size);

        if (model_type == enn::model::ModelType::CGO) {
            file_size = real_file_size;
        }
        std::unique_ptr<char[]> buffer{new char[file_size]};
        {
            std::ifstream is;
            is.open(model_file.c_str(), std::ios::binary);
            is.read(buffer.get(), file_size);
            is.close();
        }
        std::unique_ptr<Parser> parser = std::make_unique<Parser>();

        if (model_type == enn::model::ModelType::NNC) {
            std::shared_ptr<enn::model::ModelMemInfo> model_mem_info =
                std::make_shared<enn::model::ModelMemInfo>(buffer.get(), 0, file_size);
            parser->Set(model_type, model_mem_info, nullptr);
        } else if (model_type == enn::model::ModelType::CGO) {
            MemInfos param_mem_infos = parse_cgo_params_delegate(model_file.c_str(), file_size, model_offset);

            auto mem = emm->CreateMemory(file_size, enn::EnnMmType::kEnnMmTypeIon);
            uint32_t loaded_size;
            if (enn::util::import_file_to_mem(model_file.c_str(), reinterpret_cast<char**>(&(mem->va)), &loaded_size,
                                              file_size, model_offset)) {
                ENN_ERR_COUT << "Error: import_file_to_mem" << std::endl;
                return;
            }

            parser->Set(model_type, std::make_shared<enn::model::ModelMemInfo>(mem->va, 0, loaded_size), param_mem_infos);
        }
        raw_model = parser->Parse();
    }

    MemInfos parse_cgo_params_delegate(const char* model_file, uint32_t size, uint32_t offset) {
        uint32_t loaded_size;
        auto mem = emm->CreateMemory(size, enn::EnnMmType::kEnnMmTypeIon);

        if (enn::util::import_file_to_mem(model_file, reinterpret_cast<char**>(&(mem->va)), &loaded_size, size, offset)) {
            ENN_ERR_COUT << "Error: import_file_to_mem" << std::endl;
            return nullptr;
        }

        EXPECT_EQ(loaded_size, size);

        auto param_mem_infos = std::make_shared<std::vector<std::shared_ptr<enn::model::ModelMemInfo>>>();
        std::vector<std::shared_ptr<enn::EnnBufferCore>> buf_list;
        enn::util::BufferReader::UPtrType buffer_reader_ptr = std::make_unique<enn::util::FileBufferReader>(model_file);
        if (!EnnClientCgoParse(emm, buffer_reader_ptr, mem->va, size, offset, &buf_list)) {
            ENN_ERR_COUT << "Error: Cgo client parsing error";
            for (auto& buf_ele : buf_list) {
                emm->DeleteMemory(buf_ele);
            }
        }
        for (int buf_idx = 0; buf_idx < buf_list.size(); ++buf_idx) {
            auto& buf_ele = buf_list[buf_idx];
            param_mem_infos->push_back(
                std::make_shared<enn::model::ModelMemInfo>(buf_ele->va, buf_ele->fd, buf_ele->size, buf_ele->offset));
        }
        emm->DeleteMemory(mem);

        for (auto& buf_ele : buf_list) {
            emm->DeleteMemory(buf_ele);
        }

        return param_mem_infos;
    }

    enn::runtime::ClientProcess::Ptr client_process;
    std::shared_ptr<RawModel> raw_model;
    enn::model::Generator generator;
    std::unique_ptr<enn::EnnMemoryManager> emm;
};

void check_created_graph(Model::Ptr model, std::vector<std::string> expected_result) {
    int count = 0;
    for (auto& opr_ptr : model->get_origin_graph()->order<enn::model::graph::iterator::TopologicalSort>()) {
        EXPECT_STREQ(opr_ptr->get_name().c_str(), expected_result.at(count).c_str());
        ++count;
    }
}

#ifdef SCHEMA_NNC_V1
TEST_F(ModelGeneratorTest, check_generated_origin_graph_nnc_v1) {
    std::string model_file = OLYMPUS::NPU::IV3::NNC;
    raw_model_setting(enn::model::ModelType::NNC, model_file);

    auto&& model = generator.generate_model(raw_model, client_process);
    std::vector<std::string> expected_result = {"virtual_input", "NCP", "Dequantization", "SOFTMAX", "virtual_output"};
    check_created_graph(model, expected_result);
}

TEST_F(ModelGeneratorTest, check_generated_buffer_meta_data_nnc_v1) {
    std::string model_file = OLYMPUS::NPU::IV3::NNC;
    raw_model_setting(enn::model::ModelType::NNC, model_file);

    std::unique_ptr<Model> model = generator.generate_model(raw_model, client_process);

    // IV3 NNC has 4 Tensors
    // ID=1. Input                          : 1 x 3 x 299 x 299
    // ID=2. NCP Output                     : 1 x 1000 x 1 x 1
    // ID=3. Dequant Output / Softmax Input : 1 x 1000 x 1 x 1
    // ID=4. Output                         : 1 x 1000 x 1 x 1

    std::vector<std::vector<int>> expected_result;
    expected_result.push_back({1, 3, 299, 299});
    expected_result.push_back({1, 1000, 1, 1});
    expected_result.push_back({1, 1000, 1, 1});
    expected_result.push_back({1, 1000, 1, 1});

    int expected_count = 0;

    for (auto& buffer_meta_data : (model->get_buffer_meta_data())) {
        EXPECT_EQ(buffer_meta_data->get_index(), expected_count);

        EXPECT_EQ(buffer_meta_data->get_shape()[0], expected_result.at(expected_count)[0]);
        EXPECT_EQ(buffer_meta_data->get_shape()[1], expected_result.at(expected_count)[1]);
        EXPECT_EQ(buffer_meta_data->get_shape()[2], expected_result.at(expected_count)[2]);
        EXPECT_EQ(buffer_meta_data->get_shape()[3], expected_result.at(expected_count)[3]);

        expected_count++;
    }
}

TEST_F(ModelGeneratorTest, check_generated_model_id_nnc_v1) {
    std::string model_file = OLYMPUS::NPU::IV3::NNC;
    raw_model_setting(enn::model::ModelType::NNC, model_file);

    int expected_result = (enn::util::get_pid() & enn::model::MAX_PID);
    auto&& model = generator.generate_model(raw_model, client_process);

    // Check model id's upper 16bit is same with caller pid.
    EXPECT_EQ(expected_result, (model->get_id() >> 36));
}

// TODO(yc18.cho): Test real data from NNC after parser is enabled to parse attributes from NNC.
TEST_F(ModelGeneratorTest, check_set_attribute_nnc_v1) {
    std::string model_file = OLYMPUS::NPU::IV3::NNC;
    raw_model_setting(enn::model::ModelType::NNC, model_file);

    // For now, default values are set by generator
    auto expected_legacy_model = TFlite::LegacyModel_TENSORFLOW_NCHW;
    auto expected_relax_computation_float32_to_float16 = true;

    auto&& model = generator.generate_model(raw_model, client_process);

    EXPECT_EQ(expected_legacy_model, model->get_attribute().get_legacy_model());
    EXPECT_EQ(expected_relax_computation_float32_to_float16,
              model->get_attribute().get_relax_computation_float32_to_float16());
}
#else

TEST_F(ModelGeneratorTest, check_generated_IV3_model_id_nnc_v2_with_process_id) {
    std::string model_file = PAMIR::NPU::IV3::NNC;
    raw_model_setting(enn::model::ModelType::NNC, model_file);

    auto model = generator.generate_model(raw_model, client_process);

    EXPECT_TRUE(client_process->get_id().equal_in_common(model->get_id()));
}

TEST_F(ModelGeneratorTest, check_generated_DLV3_model_id_nnc_v2_with_process_id) {
    std::string model_file = PAMIR::NPU::DLV3::NNC;
    raw_model_setting(enn::model::ModelType::NNC, model_file);

    auto&& model = generator.generate_model(raw_model, client_process);

    EXPECT_TRUE(client_process->get_id().equal_in_common(model->get_id()));
}

TEST_F(ModelGeneratorTest, check_generated_SSD_model_id_nnc_v2) {
    std::string model_file = PAMIR::NPU::SSD::LEGACY::NNC_O0_HW_CFU;
    raw_model_setting(enn::model::ModelType::NNC, model_file);

    auto&& model = generator.generate_model(raw_model, client_process);

    EXPECT_TRUE(client_process->get_id().equal_in_common(model->get_id()));
}

TEST_F(ModelGeneratorTest, check_generated_nfd_model_for_dsp) {
    std::string model_file = PAMIR::DSP::NFD::VGA::NNC;
    raw_model_setting(enn::model::ModelType::NNC, model_file);

    auto&& model = generator.generate_model(raw_model, client_process);

    EXPECT_TRUE(client_process->get_id().equal_in_common(model->get_id()));
}

TEST_F(ModelGeneratorTest, test_null_raw_model_exception) {
    try {
        auto&& model = generator.generate_model(raw_model, client_process);
    } catch (std::exception& e) { EXPECT_STREQ(e.what(), "Raw Model is Null"); }
}

TEST_F(ModelGeneratorTest, test_cgo_model_generator) {
    std::string model_file = PAMIR::DSP::CGO::Gaussian3x3;

    enn::model::ModelType model_type;
    enn::util::BufferReader::UPtrType buffer_reader_ptr = std::make_unique<enn::util::FileBufferReader>(model_file);
    auto section = enn::model::ParserUtils::identify_model(buffer_reader_ptr, &model_type);
    ASSERT_GT(section.first, 0);
    ASSERT_GT(section.second, 0);
    ASSERT_EQ(enn::model::ModelType::CGO, model_type);

    const int real_file_size = section.second - section.first;
    const uint32_t model_offset = section.first;

    raw_model_setting(model_type, model_file, real_file_size, model_offset);

    auto model = generator.generate_model(raw_model, client_process);

    EXPECT_TRUE(client_process->get_id().equal_in_common(model->get_id()));

    std::vector<std::string> expected_op_result = {"virtual_input", "DSP", "virtual_output"};
    check_created_graph(model, expected_op_result);
}

TEST_F(ModelGeneratorTest, test_binding_ifm) {
    std::shared_ptr<enn::model::raw::Model> raw_model;
    enn::model::raw::ModelBuilder model_builder = enn::model::raw::ModelBuilder::build();

    // Create 1 Example Operator
    model_builder.build_operator()
        .add_operator()
        .set_op_index(0)
        .set_op_name("NPU_Operator")
        .set_op_code(0)
        .set_input_indexes(std::vector<int32_t>{0, 1, 2})
        .set_output_indexes(std::vector<int32_t>{3})
        .set_accelerator(enn::model::Accelerator::NPU)
        .build();

    // Create 3 input Tensor for operator
    for (int i = 0; i < 3; i++) {
        model_builder.build_tensor()
            .add_tensor()
            .set_index(i)
            .set_name("in_" + std::to_string(i))
            .set_type(TFlite::TensorType::TensorType_INT8)
            .set_prev_operator_index(-1)
            .add_next_operator_index(0)
            .set_shape(std::vector<uint32_t>{1, 1, 1, 1})
            .set_quantization_parameters(0)
            .set_address(0)
            .set_size(1)
            .build();
    }

    // Create 1 output Tensor for operator
    for (int i = 3; i < 4; i++) {
        model_builder.build_tensor()
            .add_tensor()
            .set_index(i)
            .set_name("out_" + std::to_string(i))
            .set_type(TFlite::TensorType::TensorType_INT8)
            .set_prev_operator_index(0)
            .add_next_operator_index(-1)
            .set_shape(std::vector<uint32_t>{1, 1, 1, 1})
            .set_quantization_parameters(0)
            .set_address(0)
            .set_size(1)
            .build();
    }

    model_builder.build_graph_info()
        .add_graph_info()
        .set_name("nnc")
        .set_inputs(std::vector<int32_t>{0, 1, 2})
        .set_outputs(std::vector<int32_t>{3})
        .build();

    model_builder.build_npu_options().add_npu_options().set_binding_ofm(false).set_binding_ifm(true).build();

    raw_model = model_builder.create();

    // Test
    auto model = generator.generate_model(raw_model, client_process);

    // Check Result
    auto buffer_meta_data_vec = model->get_buffer_meta_data();
    int expected_region_index = 0;  // Region should be set to 0 for all input.
    for (auto& bmd : buffer_meta_data_vec) {
        if (bmd->get_direction() == enn::model::Direction::Input) {
            EXPECT_EQ(expected_region_index, bmd->get_region_index());
        }
    }
}

TEST_F(ModelGeneratorTest, test_operator_tensor_index_order) {
    std::shared_ptr<enn::model::raw::Model> raw_model;
    enn::model::raw::ModelBuilder model_builder = enn::model::raw::ModelBuilder::build();

    // Create 1 Example Operator
    model_builder.build_operator()
        .add_operator()
        .set_op_index(0)
        .set_op_name("GPU_Operator")
        .set_op_code(0)
        .set_input_indexes(std::vector<int32_t>{2, 1, 0})  // Need to Keep this index Order for GPU.
        .set_output_indexes(std::vector<int32_t>{3})
        .set_accelerator(enn::model::Accelerator::NPU)
        .build();

    // Create 3 input Tensor for operator
    for (int i = 0; i < 3; i++) {
        model_builder.build_tensor()
            .add_tensor()
            .set_index(i)
            .set_name("in_" + std::to_string(i))
            .set_type(TFlite::TensorType::TensorType_INT8)
            .set_prev_operator_index(-1)
            .add_next_operator_index(0)
            .set_shape(std::vector<uint32_t>{1, 1, 1, 1})
            .set_quantization_parameters(0)
            .set_address(0)
            .set_size(1)
            .build();
    }

    // Create 1 output Tensor for operator
    for (int i = 3; i < 4; i++) {
        model_builder.build_tensor()
            .add_tensor()
            .set_index(i)
            .set_name("out_" + std::to_string(i))
            .set_type(TFlite::TensorType::TensorType_INT8)
            .set_prev_operator_index(0)
            .add_next_operator_index(-1)
            .set_shape(std::vector<uint32_t>{1, 1, 1, 1})
            .set_quantization_parameters(0)
            .set_address(0)
            .set_size(1)
            .build();
    }

    model_builder.build_graph_info()
        .add_graph_info()
        .set_name("nnc")
        .set_inputs(std::vector<int32_t>{0, 1, 2})
        .set_outputs(std::vector<int32_t>{3})
        .build();

    model_builder.build_npu_options().add_npu_options().set_binding_ofm(false).set_binding_ifm(false).build();

    raw_model = model_builder.create();

    // Test
    auto enn_model = generator.generate_model(raw_model, client_process);

    // Check Result
    std::vector<int> expected_tensor_index = {2, 1, 0};  // tensor index should be same with setting = {2, 1, 0};
    int input_count = 0;
    for (auto& opr_ptr : enn_model->get_origin_graph()->order<enn::model::graph::iterator::TopologicalSort>()) {
        if (opr_ptr->get_name() == "GPU_Operator") {
            EXPECT_EQ(opr_ptr->in_tensors[input_count]->get_id(), expected_tensor_index[input_count]);
            input_count++;
        }
    }
}

TEST_F(ModelGeneratorTest, test_mobile_bert_model_buffer_meta_data) {
    std::string model_file = PAMIR::GPU::MobileBERT::NNC;

    enn::model::ModelType model_type;
    enn::util::BufferReader::UPtrType buffer_reader_ptr = std::make_unique<enn::util::FileBufferReader>(model_file);
    auto section = enn::model::ParserUtils::identify_model(buffer_reader_ptr, &model_type);
    ASSERT_EQ(enn::model::ModelType::NNC, model_type);

    const int real_file_size = section.second - section.first;
    const uint32_t model_offset = section.first;

    raw_model_setting(model_type, model_file, model_offset);

    auto model = generator.generate_model(raw_model, client_process);

    EXPECT_TRUE(client_process->get_id().equal_in_common(model->get_id()));

    auto& buffer_meta_vector = model->get_buffer_meta_data();

    int total_buffer_num = PAMIR::GPU::MobileBERT::INPUTS.size() + PAMIR::GPU::MobileBERT::GOLDENS.size();
    EXPECT_EQ(buffer_meta_vector.size(), total_buffer_num);

    std::vector<int> expected_tensor_index = {0, 8, 16, 2744, 2745};
    for (int i = 0; i < total_buffer_num; i++) {
        EXPECT_EQ(buffer_meta_vector.at(i)->get_index(), expected_tensor_index.at(i));
    }
}

#endif
