#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <string>

#include "engine.hpp"
#include "model/parser/parser.hpp"
#include "model/parser/parser_utils.hpp"
#include "common/enn_memory_manager.h"
#include "client/enn_client_parser.hpp"
#include "medium/enn_medium_utils.hpp"
#include "test/materials.h"

class ENN_ENGINE_TEST : public testing::Test {
protected:
    void SetUp() override {}
    ENN_ENGINE_TEST() {
        emm = std::make_unique<enn::EnnMemoryManager>();
        emm->init();
    }

    ~ENN_ENGINE_TEST() {
        emm->deinit();
    }

    void load_param_setting(LoadParameter& load_param, const std::string model_file) {
        enn::model::ModelType model_type;
        uint32_t file_size;
        enn::util::get_file_size(model_file.c_str(), &file_size);

        load_param.buf_load_model.size = file_size;
        enn::util::BufferReader::UPtrType buffer_reader_ptr = std::make_unique<enn::util::FileBufferReader>(model_file);
        auto section = enn::model::ParserUtils::identify_model(buffer_reader_ptr, &model_type);
        uint32_t size = section.second - section.first;
        uint32_t offset = section.first;
#ifndef __ANDROID__
        std::unique_ptr<char[]> buffer{new char[file_size]};
        {
            std::ifstream is;
            is.open(model_file.c_str(), std::ios::binary);
            is.read(buffer.get(), file_size);
            is.close();
        }
        load_param.buf_load_model.va = (int64_t)(buffer.get());
        load_param.model_type = (uint32_t)model_type;
        enn::EnnBufferCore::Ptr mem = std::make_shared<enn::EnnBufferCore>();
        mem->va = (buffer.get());
#else
        auto mem = emm->CreateMemory(file_size, enn::EnnMmType::kEnnMmTypeIon);
        enn::util::import_file_to_mem(model_file.c_str(), reinterpret_cast<char**>(&(mem->va)), nullptr, size, offset);
#ifdef ENN_MEDIUM_IF_HIDL
        load_param.buf_load_model.fd = mem->get_native_handle();
#else
        load_param.buf_load_model.va = (int64_t)(mem->va);
        load_param.buf_load_model.fd->data[0] = (mem->fd);
#endif
        load_param.model_type = (uint32_t)model_type;
#endif
        std::vector<enn::EnnBufferCore::Ptr> buf_list;
        if (model_type == enn::model::ModelType::CGO) {
            enn::util::BufferReader::UPtrType buffer_reader_ptr = std::make_unique<enn::util::FileBufferReader>(model_file);
            if (!EnnClientCgoParse(emm, buffer_reader_ptr, mem->va, size, offset, &buf_list)) {
                ENN_ERR_COUT << "Cgo client parsing error";
                for (auto& buf_ele : buf_list) {
                    emm->DeleteMemory(buf_ele);
                }
            }
#ifdef ENN_MEDIUM_IF_HIDL
            std::vector<BufferCore> open_model_params;
            for (int buf_idx = 0; buf_idx < buf_list.size(); ++buf_idx) {
                auto& buf_ele = buf_list[buf_idx];
                open_model_params.push_back(
                    {buf_ele->get_native_handle(), buf_ele->size, 0, reinterpret_cast<uint64_t>(buf_ele->va)});
            }
            load_param.buf_load_params = open_model_params;
#endif
        }
    }

    void check_session_info_buffers(std::vector<Buffer> test_result, std::vector<Buffer> answer) {
        EXPECT_EQ(test_result.size(), answer.size());

        int count = 0;
        for (auto& buffer : test_result) {
            EXPECT_EQ(buffer.region_idx, answer[count].region_idx);
            EXPECT_EQ(buffer.dir, answer[count].dir);
            EXPECT_EQ(buffer.buf_index, answer[count].buf_index);
            EXPECT_EQ(buffer.size, answer[count].size);
            EXPECT_EQ(buffer.offset, answer[count].offset);
            EXPECT_EQ(buffer.shape, answer[count].shape);
            EXPECT_EQ(buffer.buffer_type, answer[count].buffer_type);
            EXPECT_STREQ(buffer.name.c_str(), answer[count].name.c_str());
            count++;
        }
    }

    void check_session_info_regions(std::vector<Region> test_result, std::vector<Region> answer) {
        int count = 0;
        for (auto& region : test_result) {
            EXPECT_EQ(region.req_size, answer[count++].req_size);
        }
    }

    std::unique_ptr<enn::EnnMemoryManager> emm;
};

TEST_F(ENN_ENGINE_TEST, test_open_model) {
    auto engine = enn::runtime::Engine::get_instance();
    engine->init();

#ifdef SCHEMA_NNC_V1
    std::string model_file = OLYMPUS::NPU::IV3::NNC;
#else
    std::string model_file = PAMIR::NPU::IV3::NNC;
#endif

    LoadParameter load_param;
    SessionBufInfo session_info;
    load_param_setting(load_param, model_file);

    // Test
    engine->open_model(load_param, &session_info);

    // CHECK Test Result
    ENN_TST_PRINT("model id : 0x%" PRIX64 "\n", session_info.model_id);
    std::vector<Buffer> expected_buffer_result;
    expected_buffer_result.push_back(Buffer{ 0, DirType::ENN_BUF_DIR_IN,  0,  268203, 0, {1, 299, 299,      3}, 3, "IFM"});
    expected_buffer_result.push_back(Buffer{ 1, DirType::ENN_BUF_DIR_EXT, 0, 2860832, 0, {1, 299, 299,     32}, 3, "692"});
    expected_buffer_result.push_back(Buffer{ 2, DirType::ENN_BUF_DIR_EXT, 1,    1024, 0, {1,   1,   1,   1024}, 9, "693"});
    expected_buffer_result.push_back(Buffer{ 3, DirType::ENN_BUF_DIR_EXT, 2,    1000, 0, {1,   1,   1,   1000}, 9, "691"});
    expected_buffer_result.push_back(Buffer{ 4, DirType::ENN_BUF_DIR_EXT, 3,    4000, 0, {1,   1,   1,   1000}, 0, "689"});
    expected_buffer_result.push_back(Buffer{ 5, DirType::ENN_BUF_DIR_OUT, 0,    4000, 0, {1,   1,   1,   1000}, 0, "690"});
    check_session_info_buffers(session_info.buffers, expected_buffer_result);

    std::vector<Region> expected_region_result;
    expected_region_result.push_back(Region{0, 268203});
    expected_region_result.push_back(Region{0, 2860832});
    expected_region_result.push_back(Region{0, 1024});
    expected_region_result.push_back(Region{0, 1000});
    expected_region_result.push_back(Region{0, 4000});
    expected_region_result.push_back(Region{0, 4000});
    check_session_info_regions(session_info.regions, expected_region_result);

    engine->deinit();
}

#ifndef SCHEMA_NNC_V1
TEST_F(ENN_ENGINE_TEST, test_open_model_SSD_nnc_v2) {
    auto engine = enn::runtime::Engine::get_instance();
    engine->init();

    std::string model_file = PAMIR::NPU::SSD::LEGACY::NNC_O0_HW_CFU;
    LoadParameter load_param;
    SessionBufInfo session_info;

    load_param_setting(load_param, model_file);

    // Test
    engine->open_model(load_param, &session_info);

    // CHECK Test Result
    ENN_TST_PRINT("model id : 0x%" PRIX64 "\n", session_info.model_id);
    std::vector<Buffer> expected_buffer_result;
    expected_buffer_result.push_back(Buffer{ 0, DirType::ENN_BUF_DIR_IN,  0, 307200,      0, {1, 320, 320, 3}, 3, "IFM"});
    expected_buffer_result.push_back(Buffer{ 1, DirType::ENN_BUF_DIR_EXT, 0,   8136,      0, {1, 1, 1, 8136}, 9, "765"});
    expected_buffer_result.push_back(Buffer{ 1, DirType::ENN_BUF_DIR_EXT, 1, 185094,   8136, {1, 1, 1, 185094}, 9, "766"});
    expected_buffer_result.push_back(Buffer{ 1, DirType::ENN_BUF_DIR_EXT, 2,  65088, 193230, {1, 1, 8136, 2}, 0, "763"});
    expected_buffer_result.push_back(Buffer{ 2, DirType::ENN_BUF_DIR_EXT, 3,  32544,      0, {1, 1, 1, 8136}, 0, "754"});
    expected_buffer_result.push_back(Buffer{ 3, DirType::ENN_BUF_DIR_EXT, 4, 740376,      0, {1, 1, 1, 185094}, 0, "756"});
    expected_buffer_result.push_back(Buffer{ 4, DirType::ENN_BUF_DIR_OUT, 0,    308,      0, {1, 7, 11, 1}, 0, "764"});
    check_session_info_buffers(session_info.buffers, expected_buffer_result);

    std::vector<Region> expected_region_result;
    expected_region_result.push_back(Region{0, 307200});
    expected_region_result.push_back(Region{0, 258318});
    expected_region_result.push_back(Region{0, 32544});
    expected_region_result.push_back(Region{0, 740376});
    expected_region_result.push_back(Region{0, 308});
    check_session_info_regions(session_info.regions, expected_region_result);

    engine->deinit();
}

TEST_F(ENN_ENGINE_TEST, shutdown_client_process) {
    auto engine = enn::runtime::Engine::get_instance();
    engine->init();
    std::string model_file = PAMIR::NPU::IV3::NNC;
    LoadParameter load_param;
    SessionBufInfo session_info;

    load_param_setting(load_param, model_file);

    engine->open_model(load_param, &session_info);

    engine->shutdown_client_process(enn::util::get_caller_pid());

    engine->deinit();
}

TEST_F(ENN_ENGINE_TEST, test_open_model_gaussian_cgo) {
    auto engine = enn::runtime::Engine::get_instance();
    engine->init();

    std::string model_file = PAMIR::DSP::CGO::Gaussian3x3;
    LoadParameter load_param;
    SessionBufInfo session_info;

    load_param_setting(load_param, model_file);

    // Test
    engine->open_model(load_param, &session_info);

    // CHECK Test Result
    ENN_TST_PRINT("model id : 0x%" PRIX64 "\n", session_info.model_id);
    std::vector<Buffer> expected_buffer_result;
    expected_buffer_result.push_back(
        Buffer{ 1, DirType::ENN_BUF_DIR_EXT, 0,     13, 0, {1, 13, 1, 1}, 3, "gaussian3x3_0"});
    expected_buffer_result.push_back(
        Buffer{ 2, DirType::ENN_BUF_DIR_EXT, 1,     92, 0, {1, 92, 1, 1}, 3, "Shape Infos Buffer"});
    expected_buffer_result.push_back(
        Buffer{ 3, DirType::ENN_BUF_DIR_OUT, 0, 102400, 0, {1, 102400, 1, 1}, 3, "output_gaussian3x3_0_0"});
    expected_buffer_result.push_back(
        Buffer{ 0, DirType::ENN_BUF_DIR_IN,  0, 102400, 0, {1, 102400, 1, 1}, 3, "input_gaussian3x3_0_0"});
    check_session_info_buffers(session_info.buffers, expected_buffer_result);

    std::vector<Region> expected_region_result;
    expected_region_result.push_back(Region{0, 102400});
    expected_region_result.push_back(Region{0, 13});
    expected_region_result.push_back(Region{0, 92});
    expected_region_result.push_back(Region{0, 102400});
    check_session_info_regions(session_info.regions, expected_region_result);

    engine->deinit();
}

#endif
