/*
 * Copyright (C) 2020 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or
 * distributed, transmitted, transcribed, stored in a retrieval system or
 * translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed to third parties
 * without the express written permission of Samsung Electronics.
 */

/**
 * @brief gtest main for unit test
 * @file dsp_userdriver_test.cc
 * @author Jungho Kim
 * @date 2021_03_11
 */

#include "gtest/gtest.h"
#include "client/enn_api-type.h"
#include "userdriver/unified/dsp_userdriver.h"
#include "userdriver/common/eden_osal/eden_memory.h"
#include "model/component/tensor/feature_map_builder.hpp"
#include "model/component/operator/operator_builder.hpp"
#include "model/component/operator/operator_list_builder.hpp"
#include "test/materials.h"

namespace enn {
namespace test {
namespace internal {

#define SAMPLE_MODEL_ID (0x10000000)
static auto MODEL_ID = identifier::Identifier<identifier::FullIDType, 0x7FFF, 49>(SAMPLE_MODEL_ID);

static auto MODEL_EXEC_ID(uint8_t offset) {
    return identifier::Identifier<identifier::FullIDType, 0x7FFF, 49>(SAMPLE_MODEL_ID + offset);
}

class ENN_GT_UNIT_TEST_DSP_UD : public testing::Test {
protected:
    void set_filename_ucgo(const std::string ucgo) {
        filename_ucgo = ucgo;
    }

    void set_filename_in(const std::string in) {
        filename_in = in;
    }

    void set_filename_out_golden(const std::string out_golden) {
        filename_out_golden = out_golden;
    }

    void set_filename_out_result(const std::string out_result) {
        filename_out_result = out_result;
    }

    void set_op_name(const std::string name) {
        op_name = name;
    }

    // add in/out edges to opr
    void add_edges(enn::model::component::Operator::Ptr opr,
            int in_num, int out_num) {
        enn::model::component::FeatureMapBuilder feature_map_builder;
        enn::model::component::OperatorBuilder operator_builder{opr};
        // add in edges as many as in_num
        for (int i = 0; i < in_num; i++) {
            char name[5];
            sprintf(name, "IFM%d", i);
            enn::model::component::FeatureMap::Ptr feature_map =
                feature_map_builder.set_id(i)
                                   .set_name(std::string(name))
                                   .set_buffer_index(i)
                                   .set_shape(std::vector<uint32_t>{1, 16, 299, 299})
                                   .set_data_type(TFlite::TensorType_UINT8)
                                   .create();
            operator_builder.add_in_tensor(feature_map);
        }
        // add out edges as many as edge_cnt
        for (int i = 0; i < out_num; i++) {
            char name[5];
            sprintf(name, "OFM%d", i);
            enn::model::component::FeatureMap::Ptr feature_map =
                feature_map_builder.set_id(i)
                                   .set_name(std::string(name))
                                   .set_buffer_index(i + in_num)
                                   .set_shape(std::vector<uint32_t>{1, 1008, 1, 1})
                                   .set_data_type(TFlite::TensorType_UINT8)
                                   .create();
            operator_builder.add_out_tensor(feature_map);
        }
    }

    bool is_correct_result() {
        uint32_t diff = 0;
        uint32_t* addr_out_result;

        addr_out_result = (uint32_t*) out_enn_mem[0].ref.ion.buf;
        fwrite(addr_out_result, 1, size_out_golden, fp_out_result);
        for (int i = 0; i < size_out_golden / 4; i++) {
            diff = addr_out_golden[i] - addr_out_result[i];
            if (diff != 0)
                return false;
        }
        return true;
    }

    void SetUp() override {
        size_t ret;

        enn::model::component::OperatorBuilder operator_builder;

        fp_ucgo = fopen(filename_ucgo.c_str(), "rb");
        fp_in = fopen(filename_in.c_str(), "rb");
        fp_out_result = fopen(filename_out_result.c_str(), "wb");
        fp_out_golden = fopen(filename_out_golden.c_str(), "rb");

        ASSERT_TRUE(fp_ucgo != NULL);
        ASSERT_TRUE(fp_in != NULL);
        ASSERT_TRUE(fp_out_result != NULL);
        ASSERT_TRUE(fp_out_golden != NULL);

        fseek(fp_ucgo, 0, SEEK_END);
        fseek(fp_in, 0, SEEK_END);
        fseek(fp_out_golden, 0, SEEK_END);
        size_ucgo = ftell(fp_ucgo) + 1;
        size_in = ftell(fp_in) + 1;
        size_out_golden = ftell(fp_out_golden) + 1;
        rewind(fp_ucgo);
        rewind(fp_in);
        rewind(fp_out_golden);

        addr_ucgo = malloc(sizeof(char) * size_ucgo);
        addr_in = malloc(sizeof(char) * size_in);
        addr_out_golden = (uint32_t*) malloc(sizeof(char) * size_out_golden);

        ret = fread(addr_ucgo, 1, size_ucgo, fp_ucgo);
        ret = fread(addr_in, 1, size_in, fp_in);
        ret = fread(addr_out_golden, 1, size_out_golden, fp_out_golden);

        enn::model::component::Operator::Ptr opr =
            operator_builder.set_name(op_name)
            .set_accelerator(model::Accelerator::DSP)
            .add_binary(op_name, fileno(fp_ucgo), addr_ucgo, size_ucgo, 0)
            .set_buffer_shared(buffer_shared)
            .set_ofm_bound(ofm_bound)
            .create();

        add_edges(opr, in_buf_cnt, out_buf_cnt);

        enn::model::component::OperatorListBuilder operator_list_builder;
        opr_list = operator_list_builder.build(MODEL_ID)
                                        .add_operator(opr)
                                        .set_tile_num(tile_size)
                                        .set_priority(priority)
                                        .create();

        in_enn_mem = new eden_memory_t[in_buf_cnt];
        out_enn_mem = new eden_memory_t[out_buf_cnt];

        ASSERT_EQ(PASS, eden_mem_init());
        for (int i = 0; i < in_buf_cnt; i++) {
            in_enn_mem[i].type = ION;
            in_enn_mem[i].size = size_in;
            ASSERT_EQ(PASS, eden_mem_allocate_with_ion_flag(&in_enn_mem[i], 0));
            memcpy((void*)in_enn_mem[i].ref.ion.buf, addr_in, size_in);
            buffer_table.add(i, in_enn_mem[i].ref.ion.fd,
                (void *) in_enn_mem[i].ref.ion.buf, in_enn_mem[i].size);
        }

        for (int i = 0; i < out_buf_cnt; i++) {
            out_enn_mem[i].type = ION;
            out_enn_mem[i].size = size_out_golden;
            ASSERT_EQ(PASS, eden_mem_allocate_with_ion_flag(&out_enn_mem[i], 0));
            buffer_table.add(i + in_buf_cnt, out_enn_mem[i].ref.ion.fd,
                    (void *) out_enn_mem[i].ref.ion.buf, out_enn_mem[i].size);
        }

        executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
            MODEL_EXEC_ID(0), opr_list, std::make_shared<model::memory::BufferTable>(buffer_table));

        operator_list_execute_request = std::make_shared<runtime::OperatorListExecuteRequest>(
            executable_operator_list, std::make_shared<model::memory::BufferTable>(buffer_table));

        ASSERT_EQ(PASS, eden_mem_shutdown());
    }

    void TearDown() override {
        if (addr_ucgo)
            free(addr_ucgo);
        if (addr_in)
            free(addr_in);
        if (addr_out_golden)
            free(addr_out_golden);
        if (fp_ucgo)
            fclose(fp_ucgo);
        if (fp_in)
            fclose(fp_in);
        if (fp_out_golden)
            fclose(fp_out_golden);
        if (fp_out_result)
            fclose(fp_out_result);

        ASSERT_EQ(PASS, eden_mem_init());
        if (in_enn_mem) {
            for (int i = 0; i < in_buf_cnt; i++) {
                ASSERT_EQ(PASS, eden_mem_free(&in_enn_mem[i]));
            }
            delete[] in_enn_mem;
        }
        if (out_enn_mem) {
            for (int i = 0; i < out_buf_cnt; i++) {
                ASSERT_EQ(PASS, eden_mem_free(&out_enn_mem[i]));
            }
            delete[] out_enn_mem;
        }
        ASSERT_EQ(PASS, eden_mem_shutdown());
    }

    std::string filename_ucgo;
    std::string filename_in;
    std::string filename_out_golden;
    std::string filename_out_result;
    std::string op_name;

    FILE* fp_ucgo;
    FILE* fp_in;
    FILE* fp_out_result;
    FILE* fp_out_golden;

    int32_t size_ucgo;
    int32_t size_in;
    int32_t size_out_golden;

    void* addr_ucgo;
    void* addr_in;
    uint32_t* addr_out_golden;

    int32_t in_buf_cnt;
    int32_t out_buf_cnt;

    eden_memory_t* in_enn_mem;
    eden_memory_t* out_enn_mem;

    enn::model::component::OperatorList::Ptr opr_list;
    std::shared_ptr<runtime::ExecutableOperatorList> executable_operator_list;
    std::shared_ptr<runtime::OperatorListExecuteRequest> operator_list_execute_request;
    model::memory::BufferTable buffer_table;

    int32_t priority;
    uint32_t tile_size;
    uint32_t latency;
    uint32_t bound_core;
    bool ofm_bound;
    bool buffer_shared;
};

class ENN_GT_UNIT_TEST_DSP_UD_PAMIR : public ENN_GT_UNIT_TEST_DSP_UD {

    void SetUp() override {
        // Parameters of input files
        const std::string filename_ucgo_pamir = PAMIR::DSP::IV3::UCGO::BIN;
        const std::string filename_in_pamir = PAMIR::DSP::IV3::UCGO::INPUT;
        const std::string filename_out_golden_pamir = PAMIR::DSP::IV3::UCGO::GOLDEN;
        const std::string filename_out_result_pamir = TEST_FILE("pamir/DSP_IV3/ucgo/dsp_iv3_pamir_result.bin");

        set_filename_ucgo(filename_ucgo_pamir);
        set_filename_in(filename_in_pamir);
        set_filename_out_golden(filename_out_golden_pamir);
        set_filename_out_result(filename_out_result_pamir);

        // Parameters for each operator
        const std::string op_name_pamir = "DSP_INCEPTIONV3_bin.bin";
        set_op_name(op_name_pamir);

        // Parameters for each tensor
        in_buf_cnt = 1;
        out_buf_cnt = 1;

        // Prameters of preference
        priority = 5;
        tile_size = 1;
        latency = 0;
        bound_core = 0xffffffff;
        ofm_bound = false;
        buffer_shared = false;

        ENN_GT_UNIT_TEST_DSP_UD::SetUp();
    }

    void TearDown() override {
        ENN_GT_UNIT_TEST_DSP_UD::TearDown();
    }

};

TEST_F(ENN_GT_UNIT_TEST_DSP_UD_PAMIR, dsp_ud_test_get_instance) {
    ud::dsp::DspUserDriver* dsp_ud = &ud::dsp::DspUserDriver::get_instance();
    EXPECT_NE(nullptr, dsp_ud);
}

TEST_F(ENN_GT_UNIT_TEST_DSP_UD_PAMIR, dsp_ud_test_init_deinit_once) {
    ud::dsp::DspUserDriver* dsp_ud = &ud::dsp::DspUserDriver::get_instance();
    ASSERT_EQ(ENN_RET_SUCCESS, dsp_ud->Initialize());
    ASSERT_EQ(ENN_RET_SUCCESS, dsp_ud->Deinitialize());
}

TEST_F(ENN_GT_UNIT_TEST_DSP_UD_PAMIR, dsp_ud_test_IV3_p0_open_close_once) {
    ud::dsp::DspUserDriver* dsp_ud = &ud::dsp::DspUserDriver::get_instance();

    ASSERT_EQ(ENN_RET_SUCCESS, dsp_ud->Initialize());
    ASSERT_EQ(ENN_RET_SUCCESS, dsp_ud->OpenSubGraph(*opr_list));
    ASSERT_EQ(ENN_RET_SUCCESS, dsp_ud->CloseSubGraph(*opr_list));
    ASSERT_EQ(ENN_RET_SUCCESS, dsp_ud->Deinitialize());
}

const std::string ITERATION_N_ENV_NAME = "ENN_ITER";
constexpr int32_t DEFAULT_ITER = 1;
constexpr int32_t MAX_ITER = 10000000;
TEST_F(ENN_GT_UNIT_TEST_DSP_UD_PAMIR, dsp_ud_test_IV3_p0_execute_once) {
    int num_execution;
    if (getenv(ITERATION_N_ENV_NAME.c_str())) {
        auto n = atoi(getenv(ITERATION_N_ENV_NAME.c_str()));
        num_execution = n > 0 ? n : 1;
    } else {
        num_execution = DEFAULT_ITER;
    }
    ENN_INFO_PRINT_FORCE("num_exe:%d", num_execution);
    ud::dsp::DspUserDriver* dsp_ud = &ud::dsp::DspUserDriver::get_instance();

    ASSERT_EQ(ENN_RET_SUCCESS, dsp_ud->Initialize());
    ASSERT_EQ(ENN_RET_SUCCESS, dsp_ud->OpenSubGraph(*opr_list));

    ASSERT_EQ(false, num_execution > MAX_ITER);

    for (int i = 0; i < num_execution; i++) {
        ASSERT_EQ(ENN_RET_SUCCESS, dsp_ud->ExecuteSubGraph(*operator_list_execute_request));
        ASSERT_EQ(true, is_correct_result());
    }
    ASSERT_EQ(ENN_RET_SUCCESS, dsp_ud->CloseSubGraph(*opr_list));
    ASSERT_EQ(ENN_RET_SUCCESS, dsp_ud->Deinitialize());
}

TEST_F(ENN_GT_UNIT_TEST_DSP_UD_PAMIR, dsp_ud_test_IV3_p0_prepare_execute_once) {
    ud::dsp::DspUserDriver* dsp_ud = &ud::dsp::DspUserDriver::get_instance();

    ASSERT_EQ(ENN_RET_SUCCESS, dsp_ud->Initialize());
    ASSERT_EQ(ENN_RET_SUCCESS, dsp_ud->OpenSubGraph(*opr_list));
    ASSERT_EQ(ENN_RET_SUCCESS, dsp_ud->PrepareSubGraph(*executable_operator_list));
    ASSERT_EQ(ENN_RET_SUCCESS, dsp_ud->ExecuteSubGraph(*operator_list_execute_request));

    ASSERT_EQ(true, is_correct_result());

    ASSERT_EQ(ENN_RET_SUCCESS, dsp_ud->CloseSubGraph(*opr_list));
    ASSERT_EQ(ENN_RET_SUCCESS, dsp_ud->Deinitialize());
}


}  // namespace internal
}  // namespace test
}  // namespace enn
