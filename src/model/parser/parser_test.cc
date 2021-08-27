#include <fstream>
#include <tuple>
#include <gtest/gtest.h>

#include "common/enn_debug.h"
#include "common/enn_utils.h"
#include "common/enn_memory_manager.h"
#include "client/enn_client_parser.hpp"
#include "model/parser/parser.hpp"
#include "model/parser/parser_utils.hpp"
#include "model/parser/cgo/cgo_utils.hpp"
#include "model/raw/data/binary.hpp"
#include "model/raw/data/graph_info.hpp"
#include "model/raw/data/npu_options.hpp"
#include "model/raw/data/operator.hpp"
#include "model/raw/data/operator_options.hpp"
#include "model/raw/data/tensor.hpp"
#include "test/materials.h"

namespace enn {
namespace model {

using RawModel = std::shared_ptr<raw::Model>;
using MemInfos = std::shared_ptr<std::vector<std::shared_ptr<enn::model::ModelMemInfo>>>;

class ENN_GT_UNIT_TEST_PARSER : public testing::Test {
private:
    std::unique_ptr<enn::EnnMemoryManager> emm;

public:
    ENN_GT_UNIT_TEST_PARSER() {
        emm = std::make_unique<enn::EnnMemoryManager>();
        emm->init();
    }

    ~ENN_GT_UNIT_TEST_PARSER() {
        emm->deinit();
    }

    RawModel parse_model(const ModelType& model_type, void* va, int32_t fd, int32_t size = 0,
                         MemInfos param_mem_infos = nullptr) {
        Parser parser;
        parser.Set(model_type, std::make_shared<ModelMemInfo>(va, fd, size), param_mem_infos);
        return parser.Parse();
    }

    RawModel parse_model_delegate(const ModelType& model_type, const std::string model_file, uint32_t size = 0,
                                  uint32_t offset = 0, MemInfos param_mem_infos = nullptr) {
        uint32_t file_size;
        enn::util::get_file_size(model_file.c_str(), &file_size);
        EXPECT_NE(0, file_size);

        uint32_t loaded_size;
        auto mem = emm->CreateMemory(file_size, enn::EnnMmType::kEnnMmTypeIon);

        if (enn::util::import_file_to_mem(model_file.c_str(), reinterpret_cast<char**>(&(mem->va)), &loaded_size, size,
                                          offset)) {
            ENN_ERR_COUT << "Error: import_file_to_mem" << std::endl;
            return nullptr;
        }

        EXPECT_EQ(loaded_size, size ? size : file_size);

        RawModel raw_model = parse_model(model_type, mem->va, 0, loaded_size, param_mem_infos);

        emm->DeleteMemory(mem);

        return raw_model;
    }

    MemInfos parse_cgo_params_delegate(const std::string model_file, uint32_t size, uint32_t offset) {
        uint32_t loaded_size;
        auto mem = emm->CreateMemory(size, enn::EnnMmType::kEnnMmTypeIon);

        if (enn::util::import_file_to_mem(model_file.c_str(), reinterpret_cast<char**>(&(mem->va)), &loaded_size, size,
                                          offset)) {
            ENN_ERR_COUT << "Error: import_file_to_mem" << std::endl;
            return nullptr;
        }

        EXPECT_EQ(loaded_size, size);

        auto param_mem_infos = std::make_shared<std::vector<std::shared_ptr<ModelMemInfo>>>();

        std::vector<std::shared_ptr<enn::EnnBufferCore>> buf_list;
        enn::util::BufferReader::UPtrType fbr = std::make_unique<enn::util::FileBufferReader>(model_file);

        if (!EnnClientCgoParse(emm, fbr, mem->va, size, offset, &buf_list)) {
            ENN_ERR_COUT << "Error: Cgo client parsing error";
            for (auto& buf_ele : buf_list) {
                emm->DeleteMemory(buf_ele);
            }
        }
        for (int buf_idx = 0; buf_idx < buf_list.size(); ++buf_idx) {
            auto& buf_ele = buf_list[buf_idx];
            param_mem_infos->push_back(
                std::make_shared<ModelMemInfo>(buf_ele->va, buf_ele->fd, buf_ele->size, buf_ele->offset));
        }

        emm->DeleteMemory(mem);

        for (auto& buf_ele : buf_list) {
            emm->DeleteMemory(buf_ele);
        }

        return param_mem_infos;
    }

    void CHECK_GRAPH_INFO(RawModel raw_model, int num_input, int num_output) {
        for (size_t i = 0; i < raw_model->get_graph_infos().size(); ++i) {
            auto graph_info = raw_model->get_graph_infos().at(i);
            EXPECT_EQ(num_input, graph_info->get_inputs().size());
            EXPECT_EQ(num_output, graph_info->get_outputs().size());
        }
    }

    void CHECK_OPERATOR(RawModel raw_model, std::string reference[]) {
        for (size_t i = 0; i < raw_model->get_operators().size(); ++i) {
            auto op = raw_model->get_operators().at(i);
            EXPECT_EQ(reference[i], op->get_name());
            EXPECT_NE(static_cast<int>(Accelerator::NONE), static_cast<int>(op->get_accelerator()));
        }
    }

    void CHECK_OPERATOR_OPTIONS(RawModel raw_model, std::tuple<uint32_t, uint32_t> reference[]) {
        for (size_t i = 0; i < raw_model->get_operator_options().size(); ++i) {
            auto options = raw_model->get_operator_options().at(i);
            EXPECT_EQ(std::get<0>(reference[i]), options->get_operator_index());
            EXPECT_EQ(std::get<1>(reference[i]), options->get_number());
            EXPECT_NE(nullptr, options->get_options());
        }
    }

    void CHECK_TENSOR(RawModel raw_model, std::string reference[], size_t num_fm = 0) {
        for (size_t i = 0; i < raw_model->get_tensors().size(); ++i) {
            auto tensor = raw_model->get_tensors().at(i);
            EXPECT_EQ(reference[i], tensor->get_name());
            EXPECT_LE(0, tensor->get_type());
            EXPECT_LE(0, tensor->get_size());
            if (i < num_fm || num_fm == 0) {  // Featuremap
                EXPECT_EQ(4, tensor->get_shape().size());
            } else {  // Parameter
                EXPECT_NE(nullptr, tensor->get_address());
            }
        }
    }

    void CHECK_BINARY(RawModel raw_model, std::string reference[]) {
        for (size_t i = 0; i < raw_model->get_binaries().size(); ++i) {
            auto binary = raw_model->get_binaries().at(i);
            EXPECT_EQ(reference[i], binary->get_name());
            EXPECT_NE(nullptr, binary->get_address());
            EXPECT_LE(0, binary->get_offset());
            EXPECT_LE(0, binary->get_fd());
            EXPECT_LE(0, binary->get_size());
        }
    }
};

#ifdef SCHEMA_NNC_V1
TEST_F(ENN_GT_UNIT_TEST_PARSER, test_parse_invalid_schema_version_V1) {
    std::string model_file = PAMIR::NPU::IV3::NNC;

    ModelType model_type;
    enn::util::BufferReader::UPtrType fbr = std::make_unique<enn::util::FileBufferReader>(model_file);
    auto section = ParserUtils::identify_model(fbr, &model_type);
    EXPECT_EQ(section.first, 0);
    EXPECT_GT(section.second, 0);
    EXPECT_EQ(ModelType::NNC, model_type);

    RawModel raw_model = parse_model_delegate(model_type, model_file);
    EXPECT_EQ(nullptr, raw_model);
}

TEST_F(ENN_GT_UNIT_TEST_PARSER, test_parse_Olympus_NPU_IV3) {
    std::string model_file = OLYMPUS::NPU::IV3::NNC;

    RawModel raw_model = parse_model_delegate(ModelType::NNC, model_file);
    ASSERT_NE(nullptr, raw_model);

    CHECK_GRAPH_INFO(raw_model, 1, 1);

    std::string op_names[] = {"NCP", "Dequantization", "SOFTMAX"};
    CHECK_OPERATOR(raw_model, op_names);

    std::tuple<uint32_t, uint32_t> options_op_nums[] = {{2, static_cast<uint32_t>(TFlite::BuiltinOptions_SoftmaxOptions)}};
    CHECK_OPERATOR_OPTIONS(raw_model, options_op_nums);

    std::string tensor_names[] = {"IFM", "691", "689", "690", "FRAC_LEN"};
    CHECK_TENSOR(raw_model, tensor_names, 4);

    std::string binary_names[] = {"NPU_INCEPTIONV3_ncp.bin"};
    CHECK_BINARY(raw_model, binary_names);
}

#else

TEST_F(ENN_GT_UNIT_TEST_PARSER, test_parse_invalid_schema_version_V2) {
    std::string model_file = OLYMPUS::NPU::IV3::NNC;

    ModelType model_type;
    enn::util::BufferReader::UPtrType fbr = std::make_unique<enn::util::FileBufferReader>(model_file);
    auto section = ParserUtils::identify_model(fbr, &model_type);
    EXPECT_EQ(section.first, 0);
    EXPECT_GT(section.second, 0);
    EXPECT_EQ(ModelType::NNC, model_type);

    RawModel raw_model = parse_model_delegate(model_type, model_file);
    EXPECT_EQ(nullptr, raw_model);
}

TEST_F(ENN_GT_UNIT_TEST_PARSER, test_parse_Pamir_NPU_IV3) {
    std::string model_file = PAMIR::NPU::IV3::NNC;

    RawModel raw_model = parse_model_delegate(ModelType::NNC, model_file);
    ASSERT_NE(nullptr, raw_model);

    CHECK_GRAPH_INFO(raw_model, 1, 1);

    std::string op_names[] = {"ENN_CFU", "ENN_NPU", "ENN_INVERSE_CFU", "DEQUANTIZE", "SOFTMAX"};
    CHECK_OPERATOR(raw_model, op_names);

    std::tuple<uint32_t, uint32_t> options_op_nums[] = {
        {0, static_cast<uint32_t>(TFlite::BuiltinOptions_ENN_CFUOptions)},
        {1, static_cast<uint32_t>(TFlite::BuiltinOptions_ENN_NPUOptions)},
        {2, static_cast<uint32_t>(TFlite::BuiltinOptions_ENN_InverseCFUOptions)},
        {3, static_cast<uint32_t>(TFlite::BuiltinOptions_DequantizeOptions)},
        {4, static_cast<uint32_t>(TFlite::BuiltinOptions_SoftmaxOptions)}};
    CHECK_OPERATOR_OPTIONS(raw_model, options_op_nums);
}

TEST_F(ENN_GT_UNIT_TEST_PARSER, test_parse_Pamir_DSP_IV3) {
    std::string model_file = PAMIR::DSP::IV3::NNC;

    RawModel raw_model = parse_model_delegate(ModelType::NNC, model_file);
    ASSERT_NE(nullptr, raw_model);

    CHECK_GRAPH_INFO(raw_model, 1, 1);

    std::string op_names[] = {"ENN_NORMALIZATION", "QUANTIZE",   "ENN_CFU", "ENN_DSP",
                              "ENN_INVERSE_CFU",   "DEQUANTIZE", "SOFTMAX"};
    CHECK_OPERATOR(raw_model, op_names);

    std::tuple<uint32_t, uint32_t> options_op_nums[] = {
        {0, static_cast<uint32_t>(TFlite::BuiltinOptions_ENN_NormalizationOptions)},
        {1, static_cast<uint32_t>(TFlite::BuiltinOptions_QuantizeOptions)},
        {2, static_cast<uint32_t>(TFlite::BuiltinOptions_ENN_CFUOptions)},
        {3, static_cast<uint32_t>(TFlite::BuiltinOptions_ENN_DSPOptions)},
        {4, static_cast<uint32_t>(TFlite::BuiltinOptions_ENN_InverseCFUOptions)},
        {5, static_cast<uint32_t>(TFlite::BuiltinOptions_DequantizeOptions)},
        {6, static_cast<uint32_t>(TFlite::BuiltinOptions_SoftmaxOptions)}};
    CHECK_OPERATOR_OPTIONS(raw_model, options_op_nums);
}

TEST_F(ENN_GT_UNIT_TEST_PARSER, test_parse_Pamir_MV2_DLV3) {
    std::string model_file = PAMIR::NPU::DLV3::NNC;

    RawModel raw_model = parse_model_delegate(ModelType::NNC, model_file);
    ASSERT_NE(nullptr, raw_model);

    CHECK_GRAPH_INFO(raw_model, 1, 1);

    std::string op_names[] = {"ENN_UNIFIED_DEVICE"};
    CHECK_OPERATOR(raw_model, op_names);

    std::tuple<uint32_t, uint32_t> options_op_nums[] = {
        {0, static_cast<uint32_t>(TFlite::BuiltinOptions_ENN_UNIFIED_DEVICEOptions)},
    };
    CHECK_OPERATOR_OPTIONS(raw_model, options_op_nums);

    std::string binary_names[] = {"MV2_Deeplab_V3_plus_MLPerf_tflite_ncp.bin", "MV2_Deeplab_V3_plus_MLPerf_tflite_UCGO_0"};
    CHECK_BINARY(raw_model, binary_names);
}

TEST_F(ENN_GT_UNIT_TEST_PARSER, test_parse_Pamir_MV2_DLV3_4_UCGO) {
    const char* model_file = "pamir/DLV3/MV2_Deeplab_V3_plus_MLPerf_tflite_4_UCGO.nnc";

    RawModel raw_model = parse_model_delegate(ModelType::NNC, TEST_FILE(model_file));
    ASSERT_NE(nullptr, raw_model);

    CHECK_GRAPH_INFO(raw_model, 1, 1);

    std::string op_names[] = {"ENN_UNIFIED_DEVICE"};
    CHECK_OPERATOR(raw_model, op_names);

    std::tuple<uint32_t, uint32_t> options_op_nums[] = {
        {0, static_cast<uint32_t>(TFlite::BuiltinOptions_ENN_UNIFIED_DEVICEOptions)},
    };
    CHECK_OPERATOR_OPTIONS(raw_model, options_op_nums);

    std::string binary_names[] = {"NPU",
                                  "MV2_Deeplab_V3_plus_MLPerf_tflite_UCGO_0",
                                  "MV2_Deeplab_V3_plus_MLPerf_tflite_UCGO_1",
                                  "MV2_Deeplab_V3_plus_MLPerf_tflite_UCGO_2",
                                  "MV2_Deeplab_V3_plus_MLPerf_tflite_UCGO_3"};
    CHECK_BINARY(raw_model, binary_names);
}

TEST_F(ENN_GT_UNIT_TEST_PARSER, test_parse_Pamir_Mobilenet_EdgeTPU) {
    std::string model_file = PAMIR::NPU::EdgeTPU::NNC;

    RawModel raw_model = parse_model_delegate(ModelType::NNC, model_file);
    ASSERT_NE(nullptr, raw_model);

    CHECK_GRAPH_INFO(raw_model, 1, 1);

    std::string op_names[] = {"ENN_CFU", "ENN_NPU", "ENN_INVERSE_CFU", "DEQUANTIZE", "SOFTMAX"};
    CHECK_OPERATOR(raw_model, op_names);

    std::tuple<uint32_t, uint32_t> options_op_nums[] = {
        {0, static_cast<uint32_t>(TFlite::BuiltinOptions_ENN_CFUOptions)},
        {1, static_cast<uint32_t>(TFlite::BuiltinOptions_ENN_NPUOptions)},
        {2, static_cast<uint32_t>(TFlite::BuiltinOptions_ENN_InverseCFUOptions)},
        {3, static_cast<uint32_t>(TFlite::BuiltinOptions_DequantizeOptions)},
        {4, static_cast<uint32_t>(TFlite::BuiltinOptions_SoftmaxOptions)}};
    CHECK_OPERATOR_OPTIONS(raw_model, options_op_nums);
}

TEST_F(ENN_GT_UNIT_TEST_PARSER, test_parse_Pamir_Mobiledet_SSD_O0_hw_cfu) {
    std::string model_file = PAMIR::NPU::SSD::LEGACY::NNC_O0_HW_CFU;

    RawModel raw_model = parse_model_delegate(ModelType::NNC, model_file);
    ASSERT_NE(nullptr, raw_model);

    CHECK_GRAPH_INFO(raw_model, 1, 1);

    std::string op_names[] = {"ENN_NPU", "DEQUANTIZE", "DEQUANTIZE", "ENN_DETECTION"};
    CHECK_OPERATOR(raw_model, op_names);

    std::tuple<uint32_t, uint32_t> options_op_nums[] = {
        {0, static_cast<uint32_t>(TFlite::BuiltinOptions_ENN_NPUOptions)},
        {1, static_cast<uint32_t>(TFlite::BuiltinOptions_DequantizeOptions)},
        {2, static_cast<uint32_t>(TFlite::BuiltinOptions_DequantizeOptions)},
        {3, static_cast<uint32_t>(TFlite::BuiltinOptions_ENN_DetectionOptions)}};
    CHECK_OPERATOR_OPTIONS(raw_model, options_op_nums);
}

TEST_F(ENN_GT_UNIT_TEST_PARSER, test_parse_Pamir_Mobiledet_SSD_O0_sw_cfu) {
    std::string model_file = PAMIR::NPU::SSD::LEGACY::NNC_O0_SW_CFU;

    RawModel raw_model = parse_model_delegate(ModelType::NNC, model_file);
    ASSERT_NE(nullptr, raw_model);

    CHECK_GRAPH_INFO(raw_model, 1, 1);
}

TEST_F(ENN_GT_UNIT_TEST_PARSER, test_parse_Pamir_Mobiledet_SSD_O1_hw_cfu) {
    std::string model_file = PAMIR::NPU::SSD::LEGACY::NNC_O1_HW_CFU;

    RawModel raw_model = parse_model_delegate(ModelType::NNC, model_file);
    ASSERT_NE(nullptr, raw_model);

    CHECK_GRAPH_INFO(raw_model, 1, 1);
}

TEST_F(ENN_GT_UNIT_TEST_PARSER, test_parse_Pamir_Mobiledet_SSD_O1_sw_cfu) {
    std::string model_file = PAMIR::NPU::SSD::LEGACY::NNC_O1_SW_CFU;

    RawModel raw_model = parse_model_delegate(ModelType::NNC, model_file);
    ASSERT_NE(nullptr, raw_model);

    CHECK_GRAPH_INFO(raw_model, 1, 1);

    std::string op_names[] = {"ENN_CFU", "ENN_NPU", "DEQUANTIZE", "DEQUANTIZE", "ENN_DETECTION"};
    CHECK_OPERATOR(raw_model, op_names);

    std::tuple<uint32_t, uint32_t> options_op_nums[] = {
        {0, static_cast<uint32_t>(TFlite::BuiltinOptions_ENN_CFUOptions)},
        {1, static_cast<uint32_t>(TFlite::BuiltinOptions_ENN_NPUOptions)},
        {2, static_cast<uint32_t>(TFlite::BuiltinOptions_DequantizeOptions)},
        {3, static_cast<uint32_t>(TFlite::BuiltinOptions_DequantizeOptions)},
        {4, static_cast<uint32_t>(TFlite::BuiltinOptions_ENN_DetectionOptions)}};
    CHECK_OPERATOR_OPTIONS(raw_model, options_op_nums);
}

TEST_F(ENN_GT_UNIT_TEST_PARSER, test_parse_Pamir_Mobiledet_SSD) {
    std::string model_file = PAMIR::NPU::SSD::NNC;

    RawModel raw_model = parse_model_delegate(ModelType::NNC, model_file);
    ASSERT_NE(nullptr, raw_model);

    CHECK_GRAPH_INFO(raw_model, 1, 1);

    std::string op_names[] = {"ENN_NPU", "DEQUANTIZE", "DEQUANTIZE", "ENN_DETECTION"};
    CHECK_OPERATOR(raw_model, op_names);

    std::tuple<uint32_t, uint32_t> options_op_nums[] = {
        {0, static_cast<uint32_t>(TFlite::BuiltinOptions_ENN_NPUOptions)},
        {1, static_cast<uint32_t>(TFlite::BuiltinOptions_DequantizeOptions)},
        {2, static_cast<uint32_t>(TFlite::BuiltinOptions_DequantizeOptions)},
        {3, static_cast<uint32_t>(TFlite::BuiltinOptions_ENN_DetectionOptions)}};
    CHECK_OPERATOR_OPTIONS(raw_model, options_op_nums);
}

TEST_F(ENN_GT_UNIT_TEST_PARSER, test_parse_Pamir_OD_VGA) {
    std::string model_file = PAMIR::NPU::OD::VGA::NNC;

    RawModel raw_model = parse_model_delegate(ModelType::NNC, model_file);
    ASSERT_NE(nullptr, raw_model);

    CHECK_GRAPH_INFO(raw_model, 1, 12);
}

TEST_F(ENN_GT_UNIT_TEST_PARSER, test_parse_Pamir_OD_QVGA) {
    std::string model_file = PAMIR::NPU::OD::QVGA::NNC;

    RawModel raw_model = parse_model_delegate(ModelType::NNC, model_file);
    ASSERT_NE(nullptr, raw_model);

    CHECK_GRAPH_INFO(raw_model, 1, 10);
}

TEST_F(ENN_GT_UNIT_TEST_PARSER, test_parse_Schema_V2_GPU_operators_conv) {
    std::string model_file = PAMIR::GPU::OPERATOR::CONV_FLOAT;

    ModelType model_type;
    enn::util::BufferReader::UPtrType fbr = std::make_unique<enn::util::FileBufferReader>(model_file);
    auto section = ParserUtils::identify_model(fbr, &model_type);
    EXPECT_EQ(section.first, 0);
    EXPECT_GT(section.second, 0);
    EXPECT_EQ(ModelType::NNC, model_type);

    RawModel raw_model = parse_model_delegate(model_type, model_file);
    ASSERT_NE(nullptr, raw_model);

    CHECK_GRAPH_INFO(raw_model, 1, 1);

    std::string op_names[] = {"CONV_2D"};
    CHECK_OPERATOR(raw_model, op_names);

    std::tuple<uint32_t, uint32_t> options_op_nums[] = {
        {0, static_cast<uint32_t>(TFlite::BuiltinOptions_Conv2DOptions)},
    };
    CHECK_OPERATOR_OPTIONS(raw_model, options_op_nums);
}

TEST_F(ENN_GT_UNIT_TEST_PARSER, test_parse_Schema_V2_GPU_operators_add) {
    std::string model_file = PAMIR::GPU::OPERATOR::ADD;

    RawModel raw_model = parse_model_delegate(ModelType::NNC, model_file);
    ASSERT_NE(nullptr, raw_model);

    CHECK_GRAPH_INFO(raw_model, 2, 1);

    std::string op_names[] = {"ADD"};
    CHECK_OPERATOR(raw_model, op_names);

    std::tuple<uint32_t, uint32_t> options_op_nums[] = {
        {0, static_cast<uint32_t>(TFlite::BuiltinOptions_AddOptions)},
    };
    CHECK_OPERATOR_OPTIONS(raw_model, options_op_nums);
}
#endif

TEST_F(ENN_GT_UNIT_TEST_PARSER, test_parse_CGO_identify_model_type) {
    std::string model_file = PAMIR::DSP::CGO::Gaussian3x3;

    ModelType model_type;
    enn::util::BufferReader::UPtrType fbr = std::make_unique<enn::util::FileBufferReader>(model_file);
    auto section = ParserUtils::identify_model(fbr, &model_type);
    ASSERT_GT(section.first, 0);
    ASSERT_GT(section.second, 0);
    ASSERT_EQ(ModelType::CGO, model_type);
}

TEST_F(ENN_GT_UNIT_TEST_PARSER, test_parse_CGO_parse_paramters) {
    std::string model_file = PAMIR::DSP::CGO::Gaussian3x3;

    ModelType model_type;
    enn::util::BufferReader::UPtrType fbr = std::make_unique<enn::util::FileBufferReader>(model_file);
    auto section = ParserUtils::identify_model(fbr, &model_type);
    ASSERT_GT(section.first, 0);
    ASSERT_GT(section.second, 0);
    ASSERT_EQ(ModelType::CGO, model_type);

    uint32_t model_size = section.second - section.first;
    uint32_t model_offset = section.first;

    auto param_mem_infos = parse_cgo_params_delegate(model_file, model_size, model_offset);
    EXPECT_NE(nullptr, param_mem_infos);
}

TEST_F(ENN_GT_UNIT_TEST_PARSER, test_parse_CGO_Pamir_CV_Gaussian) {
    std::string model_file = PAMIR::DSP::CGO::Gaussian3x3;

    ModelType model_type;
    enn::util::BufferReader::UPtrType fbr = std::make_unique<enn::util::FileBufferReader>(model_file);
    auto section = ParserUtils::identify_model(fbr, &model_type);
    ASSERT_GT(section.first, 0);
    ASSERT_GT(section.second, 0);
    ASSERT_EQ(ModelType::CGO, model_type);

    uint32_t model_size = section.second - section.first;
    uint32_t model_offset = section.first;

    auto param_mem_infos = parse_cgo_params_delegate(model_file, model_size, model_offset);
    EXPECT_NE(nullptr, param_mem_infos);

    RawModel raw_model = parse_model_delegate(model_type, model_file, model_size, model_offset, param_mem_infos);
    ASSERT_NE(nullptr, raw_model);
}

TEST_F(ENN_GT_UNIT_TEST_PARSER, test_parse_CGO_Pamir_DSP_IV3) {
    std::string model_file = PAMIR::DSP::CGO::IV3;

    ModelType model_type;
    enn::util::BufferReader::UPtrType fbr = std::make_unique<enn::util::FileBufferReader>(model_file);
    auto section = ParserUtils::identify_model(fbr, &model_type);
    ASSERT_GT(section.first, 0);
    ASSERT_GT(section.second, 0);
    ASSERT_EQ(ModelType::CGO, model_type);

    uint32_t model_size = section.second - section.first;
    uint32_t model_offset = section.first;

    auto param_mem_infos = parse_cgo_params_delegate(model_file, model_size, model_offset);
    EXPECT_NE(nullptr, param_mem_infos);

    RawModel raw_model = parse_model_delegate(model_type, model_file, model_size, model_offset, param_mem_infos);
    ASSERT_NE(nullptr, raw_model);
}

TEST_F(ENN_GT_UNIT_TEST_PARSER, test_load_coded) {
    std::string dummy_buffer = "CODEDFLATBUFFER";
    int length = dummy_buffer.length();

    RawModel raw_model =
        parse_model(enn::model::ModelType::CODED, static_cast<void*>(const_cast<char*>(dummy_buffer.data())), 0, length);
    EXPECT_EQ(nullptr, raw_model) << "Not supported yet";
}

TEST_F(ENN_GT_UNIT_TEST_PARSER, test_load_unknown_type) {
    std::string dummy_buffer = "UNKNOWN_TYPE_FOR_COVERAGE";
    int length = dummy_buffer.length();

    RawModel raw_model =
        parse_model(enn::model::ModelType::NONE, static_cast<void*>(const_cast<char*>(dummy_buffer.data())), 0, length);
    EXPECT_EQ(nullptr, raw_model) << "Not supported";
}

};  // namespace model
};  // namespace enn
