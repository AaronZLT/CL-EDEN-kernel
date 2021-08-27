/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
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
 * @file cpu_userdriver_test.cc
 * @author Byungjin Jung
 * @date 2021_03_11
 */

#include <random>

#include "gtest/gtest.h"

#include "client/enn_api-type.h"
#include "common/enn_memory_manager.h"
#include "userdriver/cpu/cpu_userdriver.h"
#include "userdriver/common/operator_interfaces/custom_operator.h"
#include "model/schema/schema_nnc.h"
#include "model/component/tensor/feature_map_builder.hpp"
#include "model/component/tensor/parameter_builder.hpp"
#include "model/component/operator/operator_builder.hpp"
#include "model/component/operator/operator_list_builder.hpp"
#include "model/parser/parser.hpp"
#include "model/raw/data/operator.hpp"
#include "model/raw/data/operator_options.hpp"
#include "model/raw/data/tensor.hpp"
#include "model/raw/model.hpp"
#include "test/materials.h"

namespace enn {
namespace test {
namespace internal {

auto MODEL_ID = identifier::Identifier<identifier::FullIDType, 0x7FFF, 49>(0x10000000);

auto EXEC_MODEL_ID(uint8_t offset) {
    return identifier::Identifier<identifier::FullIDType, 0x7FFF, 49>(0x10000000 + offset);
}

class ENN_GT_UNIT_TEST_CPU_UD : public testing::Test {
protected:
    ENN_GT_UNIT_TEST_CPU_UD() {
        emm = std::make_unique<enn::EnnMemoryManager>();
        emm->init();
    }

    ~ENN_GT_UNIT_TEST_CPU_UD() {
        emm->deinit();
    }

    std::unique_ptr<enn::EnnMemoryManager> emm;
    enn::model::component::OperatorListBuilder operator_list_builder;

    std::shared_ptr<model::raw::Model> parse_model(const model::ModelType& model_type, std::string model_file) {
        uint32_t file_size;
        enn::util::get_file_size(model_file.c_str(), &file_size);
        EXPECT_NE(0, file_size);

        auto mem = emm->CreateMemory(file_size, enn::EnnMmType::kEnnMmTypeIon);
        enn::util::import_file_to_mem(model_file.c_str(), reinterpret_cast<char**>(&(mem->va)), nullptr);

        auto param_mem_infos = std::make_shared<std::vector<std::shared_ptr<enn::model::ModelMemInfo>>>();
        model::Parser parser;
        parser.Set(model_type, std::make_shared<model::ModelMemInfo>(mem->va, 0, file_size), param_mem_infos);
        return parser.Parse();
    }

    bool is_invalid_memory(ud::cpu::CpuUserDriver& cpu_ud, const model::component::OperatorList& operator_list,
                           std::vector<EnnBufferCore::Ptr> buffers) {
        bool is_invalid_memory = false;
        for (auto mem : buffers) {
            if (mem == nullptr) {
                for (auto mem : buffers) {
                    emm->DeleteMemory(mem);
                }
                cpu_ud.CloseSubGraph(operator_list);
                cpu_ud.Deinitialize();
                is_invalid_memory = true;
                break;
            }
        }
        EXPECT_FALSE(is_invalid_memory);
        return is_invalid_memory;
    }

    std::unordered_map<TFlite::TensorType, uint32_t> data_size = {
        {TFlite::TensorType::TensorType_FLOAT32, sizeof(float)}, {TFlite::TensorType::TensorType_INT32, sizeof(int32_t)},
        {TFlite::TensorType::TensorType_UINT8, sizeof(uint8_t)}, {TFlite::TensorType::TensorType_BOOL, sizeof(bool)},
        {TFlite::TensorType::TensorType_INT16, sizeof(int16_t)}, {TFlite::TensorType::TensorType_INT8, sizeof(int8_t)},
    };

    // add in/out edges to opr
    void add_edges(enn::model::component::Operator::Ptr opr, int in_num, int out_num, TFlite::TensorType in_type,
                   TFlite::TensorType out_type) {
        enn::model::component::FeatureMapBuilder feature_map_builder;
        enn::model::component::OperatorBuilder operator_builder{opr};
        int32_t buffer_index = 0;

        // add in edges as many as in_num
        for (int i = 0; i < in_num; i++) {
            char name[5];
            sprintf(name, "IFM%d", i);
            uint32_t h = (i + 1) * 2, w = h;
            std::vector<uint32_t> shape = {1, 1, h, w};
            enn::model::component::FeatureMap::Ptr feature_map =
                feature_map_builder.set_id(i)
                    .set_name(std::string(name))
                    .set_data_type(in_type)
                    .set_buffer_index(buffer_index++)
                    .set_buffer_size(h * w * data_size[in_type])
                    .set_shape(shape)
                    .create();
            operator_builder.add_in_tensor(feature_map);
        }
        // add out edges as many as edge_cnt
        for (int i = 0; i < out_num; i++) {
            char name[5];
            sprintf(name, "OFM%d", i);
            uint32_t h = (i + 1) * 2, w = h;
            std::vector<uint32_t> shape = {1, 1, h, w};
            enn::model::component::FeatureMap::Ptr feature_map =
                feature_map_builder.set_id(i)
                    .set_name(std::string(name))
                    .set_data_type(out_type)
                    .set_buffer_index(buffer_index++)
                    .set_buffer_size(h * w * data_size[out_type])
                    .set_shape(shape)
                    .create();
            operator_builder.add_out_tensor(feature_map);
        }
    }

    template <typename T1, typename T2>
    inline void fill_buffer(T1& data, int size, T2 value) {
        for (int i = 0; i < size; i++) {
            data[i] = i + value;
        }
    }

    // Referenced from test_utils.h {
    template <typename T1, typename T2, typename T3>
    inline bool GenerateRandom(T1* ptr, const uint32_t& size, const T2& min, const T3& max) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(min, max);
        for (uint32_t idx = 0; idx < size; idx++) {
            *(ptr + idx) = dis(gen);
        }
        return true;
    }

    inline uint32_t GetDimSize(const Dim4& dims) {
        return dims.n * dims.c * dims.h * dims.w;
    }

    template <typename T1, typename T2>
    inline void compare(const T1* input1, const T2* input2, size_t size, float error_threshold_ = (1e-5)) {
        for (size_t i = 0; i < size; i++) {
            EXPECT_NEAR(input1[i], input2[i], error_threshold_) << i;
        }
    }
    // } Referenced from test_utils.h

    // { Referenced from Normalization_test
    void NormalizationRef(uint8_t* input, float* output, int channel, int height, int width, float* mean, float* scale,
                          uint8_t bgr_transpose) {
        int i, j, k;
        uint8_t num;
        uint8_t c_type_order;  //*< channel type order
        uint32_t idx;
        if (bgr_transpose == 1) {
            //* normalization & bgr_transpose
            uint8_t b, g, r;
            float normalized;
            int b_idx, g_idx, r_idx;
            b = 0;
            g = 1;
            r = 2;

            b_idx = 0;
            g_idx = height * width;
            r_idx = height * width * 2;

            for (i = 0; i < channel; i++) {
                for (j = 0; j < height; j++) {
                    for (k = 0; k < width; k++) {
                        idx = (i * height * width) + (j * width) + k;
                        num = input[idx];
                        c_type_order = i % channel;
                        normalized = ((float)num - mean[c_type_order]) * scale[c_type_order];
                        if (c_type_order == b) {
                            output[b_idx++] = normalized;
                        } else if (c_type_order == g) {
                            output[g_idx++] = normalized;
                        } else if (c_type_order == r) {
                            output[r_idx++] = normalized;
                        } else {
                        }
                    }
                }
            }

        } else {
            // just normalization
            for (i = 0; i < channel; i++) {
                for (j = 0; j < height; j++) {
                    for (k = 0; k < width; k++) {
                        idx = (i * height * width) + (j * width) + k;
                        num = input[idx];
                        c_type_order = i % channel;
                        output[idx] = (static_cast<float>(num) - mean[i]) * scale[i];
                    }
                }
            }
        }
    }
    // } Referenced from Normalization_test

    // { Referenced from Dequantization_test
    template <typename T>
    void DeQuantizationRef(const T* input, float* output, const uint32_t& input_size, int32_t* frac_len, int32_t channel) {
        uint32_t i;
        T num;
        float ret_val;
        for (i = 0; i < input_size; i++) {
            num = input[i];
            ret_val = (float)num / std::pow(2, frac_len[i / (input_size / channel)]);
            output[i] = ret_val;
        }
    }
    // } Referenced from Dequantization_test

    //                           [ Full ID Format]
    // |  reserved  | caller pid | model uid | op_list uid | executable model uid |
    // |   15 bit   |   17 bit   |   8  bit  |    16 bit   |        8 bit         |
    void CHECK_OP_LIST_UID(uint64_t op_list_id) {
        uint64_t uid = (op_list_id & 0x0000000000FFFF00) >> 8;
        ENN_DBG_PRINT("op_list_id: 0x%" PRIx64 ", uid: 0x%" PRIx64 "\n", op_list_id, uid);
        EXPECT_NE(0, uid);
    }

    void CHECK_OPERATOR(ud::UDOperators operators, std::string reference[]) {
        for (size_t i = 0; i < operators->size(); i++) {
            EXPECT_EQ(reference[i], operators->at(i)->getName());
        }
    }

    class UT_Tensor {
    public:
        Dim4 shapes;
        TFlite::TensorType tensor_type;
        std::string param_name;
        const void* param_addr;
        bool is_param;

        UT_Tensor(Dim4 shapes, TFlite::TensorType tensor_type) {
            this->shapes = shapes;
            this->tensor_type = tensor_type;
            is_param = false;
        }

        UT_Tensor(Dim4 shapes, TFlite::TensorType tensor_type, std::string param_name, const void* param_addr) {
            this->shapes = shapes;
            this->tensor_type = tensor_type;
            this->param_name = param_name;
            this->param_addr = param_addr;
            is_param = true;
        }

        static bool is_featuremap(const std::vector<uint32_t>& shape) {
            return ((shape[0] * shape[1] * shape[2] * shape[3]) > 0);
        }

        static Dim4 get_dims(const std::vector<uint32_t>& shape) {
            return {shape[0], shape[1], shape[2], shape[3]};
        }

        static void add_tensor(std::vector<std::shared_ptr<model::raw::data::Tensor>>& tensors,
                               std::vector<int32_t>& indexes, std::vector<UT_Tensor*>& result) {
            for (auto ii : indexes) {
                for (auto& tensor : tensors) {
                    if (ii == tensor->get_index() && UT_Tensor::is_featuremap(tensor->get_shape())) {
                        result.push_back(new UT_Tensor(get_dims(tensor->get_shape()),
                                                       static_cast<TFlite::TensorType>(tensor->get_type())));
                    }
                }
            }
        }
    };

    class UT_Operator {
    public:
        TFlite::BuiltinOperator builtin_operator;
        TFlite::BuiltinOptions builtin_options;
        std::string custom_operator;
        std::vector<UT_Tensor*> in_tensors;
        std::vector<UT_Tensor*> out_tensors;
        const void* options;
        bool is_builtin;

        UT_Operator(std::shared_ptr<model::raw::data::Operator>& op,
                    std::vector<std::shared_ptr<model::raw::data::OperatorOptions>>& operator_options) {
            if (op->get_op_code() == UNDEFINED) {
                this->custom_operator = op->get_name();
                this->is_builtin = false;
            } else {
                auto options = operator_options.at(op->get_operator_options_index());
                EXPECT_NE(nullptr, options);
                this->builtin_operator = static_cast<TFlite::BuiltinOperator>(op->get_op_code());
                this->builtin_options = static_cast<TFlite::BuiltinOptions>(options->get_number());
                this->options = options->get_options();
                this->is_builtin = true;
            }
        }
        ~UT_Operator() {
            for (UT_Tensor* ut_tensor : in_tensors) {
                delete ut_tensor;
            }
            for (UT_Tensor* ut_tensor : out_tensors) {
                delete ut_tensor;
            }
        }
    };

    template <typename UnitType>
    void TEST_OPERATOR(std::vector<std::shared_ptr<UT_Operator>>& operators, std::string input_file,
                       std::string output_file) {
        ud::cpu::CpuUserDriver& cpu_ud = ud::cpu::CpuUserDriver::get_instance();

        cpu_ud.Initialize();

        model::component::OperatorBuilder operator_builder;
        model::component::ParameterBuilder parameter_builder;
        model::component::FeatureMapBuilder feature_map_builder;
        model::memory::BufferTable buffer_table;
        std::vector<EnnBufferCore::Ptr> buffers;
        size_t op_size = operators.size();

        auto op_list_builder = operator_list_builder.build(MODEL_ID);

        for (size_t i = 0; i < op_size; ++i) {
            auto op = operators.at(i);

            if (op->is_builtin) {
                operator_builder.set_name(TFlite::EnumNameBuiltinOperator(op->builtin_operator)).set_code(op->builtin_operator);
            } else {
                operator_builder.set_name(op->custom_operator).set_code(static_cast<TFlite::BuiltinOperator>(UNDEFINED));
            }

            if (op->options != nullptr) {
                operator_builder.set_option(TFlite::EnumNameBuiltinOptions(op->builtin_options), op->options, 0,
                                            op->builtin_options);
            }

            for (size_t it = 0; it < op->in_tensors.size(); ++it) {
                auto in = op->in_tensors[it];
                size_t input_size = GetDimSize(in->shapes);

                if (in->is_param) {
                    std::shared_ptr<enn::model::component::Parameter> param =
                        parameter_builder.set_id(i)
                            .set_name(in->param_name)
                            .set_data_type(in->tensor_type)
                            .set_buffer_addr(in->param_addr)
                            .set_buffer_size(data_size[in->tensor_type] * input_size)
                            .create();

                    operator_builder.add_in_tensor(param);
                } else {
                    enn::model::component::FeatureMap::Ptr ifm =
                        feature_map_builder.set_id(i)
                            .set_name(std::string("ifm:" + std::to_string(i)))
                            .set_data_type(in->tensor_type)
                            .set_buffer_index(i)
                            .set_buffer_size(input_size)
                            .set_shape(std::vector<uint32_t>{in->shapes.n, in->shapes.c, in->shapes.h, in->shapes.w})
                            .create();

                    operator_builder.add_in_tensor(ifm);

                    auto in_buff = emm->CreateMemory(data_size[in->tensor_type] * input_size, enn::EnnMmType::kEnnMmTypeIon);
                    buffer_table.add(i, in_buff->va, in_buff->size);
                    buffers.push_back(in_buff);
                }
            }

            for (size_t ot = 0; ot < op->out_tensors.size(); ++ot) {
                auto out = op->out_tensors[ot];
                size_t output_size = GetDimSize(out->shapes);

                enn::model::component::FeatureMap::Ptr ofm =
                    feature_map_builder.set_id(i + 1)
                        .set_name(std::string("ofm:" + std::to_string(i)))
                        .set_data_type(out->tensor_type)
                        .set_buffer_index(i + 1)
                        .set_buffer_size(output_size)
                        .set_shape(std::vector<uint32_t>{out->shapes.n, out->shapes.c, out->shapes.h, out->shapes.w})
                        .create();

                operator_builder.add_out_tensor(ofm);

                if (i == (op_size - 1)) {
                    auto out_buff = emm->CreateMemory(data_size[out->tensor_type] * output_size, enn::EnnMmType::kEnnMmTypeIon);
                    buffer_table.add(i + 1, out_buff->va, out_buff->size);
                    buffers.push_back(out_buff);
                }
            }

            op_list_builder.add_operator(operator_builder.set_id(i).set_accelerator(model::Accelerator::CPU).create());
        }

        auto opr_list = op_list_builder.create();

        cpu_ud.OpenSubGraph(*opr_list);

        CHECK_AND_RETURN_VOID(is_invalid_memory(cpu_ud, *opr_list, buffers), "CreateMemory() failed");

        auto ibuf = buffers[0];
        auto obuf = buffers[op_size];

        uint32_t file_size;
        EXPECT_EQ(ENN_RET_SUCCESS,
                  enn::util::import_file_to_mem(input_file.c_str(), reinterpret_cast<char**>(&(ibuf->va)), &file_size));
        ENN_TST_PRINT("golden in file size: %d\n", file_size);

        auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
            MODEL_ID, opr_list, std::make_shared<model::memory::BufferTable>(buffer_table));
        EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.PrepareSubGraph(*executable_operator_list));

        auto operator_list_execute_request = runtime::OperatorListExecuteRequest(
            executable_operator_list, std::make_shared<model::memory::BufferTable>(buffer_table));
        EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.ExecuteSubGraph(operator_list_execute_request));

        // Check with golden data
        char* golden_out_buf = nullptr;
        EXPECT_EQ(ENN_RET_SUCCESS, enn::util::import_file_to_mem(output_file.c_str(), &golden_out_buf, &file_size));
        ENN_TST_PRINT("golden out file size: %d, ptr: %p\n", file_size, golden_out_buf);

        int32_t diff = enn::util::CompareBuffersWithThreshold<UnitType>(
            golden_out_buf, reinterpret_cast<char*>(obuf->va), obuf->size, nullptr, enn::util::BUFFER_COMPARE_THRESHOLD);

        delete[] golden_out_buf;

        for (auto buffer : buffers) {
            emm->DeleteMemory(buffer);
        }

        cpu_ud.CloseSubGraph(*opr_list);

        cpu_ud.Deinitialize();

        EXPECT_EQ(0, diff);
        if (diff == 0) {
            ENN_INFO_PRINT("Golden match : SUCCESS\n");
        }
    }

    enn::model::component::Operator::Ptr create_custom_softmax(const float& beta, const int32_t& axis) {
        enn::model::component::OperatorBuilder operator_builder;
        enn::model::component::ParameterBuilder parameter_builder;

        std::string op_name = "SOFTMAX";

        std::shared_ptr<enn::model::component::Parameter> param_beta = parameter_builder.set_name("BETA")
                                                                           .set_buffer_addr(&beta)
                                                                           .set_buffer_size(sizeof(beta))
                                                                           .set_data_type(TFlite::TensorType_FLOAT32)
                                                                           .create();

        std::shared_ptr<enn::model::component::Parameter> param_axis = parameter_builder.set_name("AXIS")
                                                                           .set_buffer_addr(&axis)
                                                                           .set_buffer_size(sizeof(axis))
                                                                           .set_data_type(TFlite::TensorType_INT32)
                                                                           .create();

        enn::model::component::Operator::Ptr opr = operator_builder.set_id(op_name.length())
                                                                   .set_name(op_name)
                                                                   .set_code((TFlite::BuiltinOperator)UNDEFINED)
                                                                   .set_accelerator(model::Accelerator::CPU)
                                                                   .add_in_tensor(param_beta)
                                                                   .add_in_tensor(param_axis)
                                                                   .create();

        add_edges(opr, 1, 1, TFlite::TensorType_FLOAT32, TFlite::TensorType_FLOAT32);

        return opr;
    }

    enn::model::component::Operator::Ptr create_builtin_softmax(float beta, int32_t axis) {
        enn::model::component::OperatorBuilder operator_builder;

        std::string op_name = "SOFTMAX";
        std::string options_name = "SoftmaxOptions";
        std::shared_ptr<ud::TC_SoftmaxOptions> options = std::make_shared<ud::TC_SoftmaxOptions>();
        options->beta_ = beta;
        options->axis_ = axis;

        enn::model::component::Operator::Ptr opr =
            operator_builder.set_id(op_name.length())
                .set_name(op_name)
                .set_accelerator(model::Accelerator::CPU)
                .set_code(TFlite::BuiltinOperator::BuiltinOperator_SOFTMAX)
                .set_option(options_name, options.get(), sizeof(ud::TC_SoftmaxOptions),
                            TFlite::BuiltinOptions_SoftmaxOptions)
                .create();

        add_edges(opr, 1, 1, TFlite::TensorType_FLOAT32, TFlite::TensorType_FLOAT32);

        return opr;
    }

    template <typename T1, typename T2>
    std::shared_ptr<model::memory::BufferTable> create_buffer_table(T1 in[], int32_t in_num, T2 out[], int32_t out_num) {
        model::memory::BufferTable buffer_table;
        fill_buffer(in, in_num, 1.f);
        fill_buffer(out, out_num, 1.f);
        buffer_table.add(0, in, sizeof(T1) * in_num);
        buffer_table.add(1, out, sizeof(T2) * out_num);
        return std::make_shared<model::memory::BufferTable>(buffer_table);
    }
};

TEST_F(ENN_GT_UNIT_TEST_CPU_UD, cpu_ud_test_initialize) {
    ud::cpu::CpuUserDriver& cpu_ud = ud::cpu::CpuUserDriver::get_instance();

    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.Initialize());
}

TEST_F(ENN_GT_UNIT_TEST_CPU_UD, cpu_ud_test_initialize_deinitialize) {
    ud::cpu::CpuUserDriver& cpu_ud = ud::cpu::CpuUserDriver::get_instance();

    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.Initialize());

    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.Deinitialize());
}

TEST_F(ENN_GT_UNIT_TEST_CPU_UD, cpu_ud_test_custom_operator_open) {
    ud::cpu::CpuUserDriver& cpu_ud = ud::cpu::CpuUserDriver::get_instance();

    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.Initialize());

    auto opr = create_custom_softmax(2.5f, 7);
    auto opr_list = operator_list_builder.build(MODEL_ID).add_operator(opr).create();

    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.OpenSubGraph(*opr_list));

    CHECK_OP_LIST_UID(opr_list->get_id());
}

TEST_F(ENN_GT_UNIT_TEST_CPU_UD, cpu_ud_test_custom_operator_open_prepare) {
    ud::cpu::CpuUserDriver& cpu_ud = ud::cpu::CpuUserDriver::get_instance();

    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.Initialize());

    auto opr = create_custom_softmax(2.5f, 7);
    auto opr_list = operator_list_builder.build(MODEL_ID).add_operator(opr).create();

    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.OpenSubGraph(*opr_list));

    CHECK_OP_LIST_UID(opr_list->get_id());

    float in[10], out[10];
    auto buffer_table = create_buffer_table(in, 10, out, 10);

    auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(MODEL_ID, opr_list, buffer_table);
    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.PrepareSubGraph(*executable_operator_list));
}

TEST_F(ENN_GT_UNIT_TEST_CPU_UD, cpu_ud_test_custom_operator_open_prepare_exec) {
    ud::cpu::CpuUserDriver& cpu_ud = ud::cpu::CpuUserDriver::get_instance();

    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.Initialize());

    auto opr = create_custom_softmax(2.5f, 7);
    auto opr_list = operator_list_builder.build(MODEL_ID).add_operator(opr).create();

    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.OpenSubGraph(*opr_list));

    CHECK_OP_LIST_UID(opr_list->get_id());

    float in[10], out[10];
    auto buffer_table = create_buffer_table(in, 10, out, 10);

    auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(MODEL_ID, opr_list, buffer_table);
    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.PrepareSubGraph(*executable_operator_list));

    auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list, buffer_table);
    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.ExecuteSubGraph(operator_list_execute_request));
}

TEST_F(ENN_GT_UNIT_TEST_CPU_UD, cpu_ud_test_custom_operator_open_prepare_exec_close) {
    ud::cpu::CpuUserDriver& cpu_ud = ud::cpu::CpuUserDriver::get_instance();

    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.Initialize());

    auto opr = create_custom_softmax(2.5f, 7);
    auto opr_list = operator_list_builder.build(MODEL_ID).add_operator(opr).create();

    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.OpenSubGraph(*opr_list));

    CHECK_OP_LIST_UID(opr_list->get_id());

    float in[10], out[10];
    auto buffer_table = create_buffer_table(in, 10, out, 10);

    auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(MODEL_ID, opr_list, buffer_table);
    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.PrepareSubGraph(*executable_operator_list));

    auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list, buffer_table);
    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.ExecuteSubGraph(operator_list_execute_request));

    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.CloseSubGraph(*opr_list));

    ud::UDOperators dummy;

    EXPECT_NE(ENN_RET_SUCCESS, cpu_ud.get_graph_for_TC(opr_list->get_id(), dummy));

    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.Deinitialize());
}

TEST_F(ENN_GT_UNIT_TEST_CPU_UD, cpu_ud_test_custom_operators_Normalization_Dequantization) {
    ud::cpu::CpuUserDriver& cpu_ud = ud::cpu::CpuUserDriver::get_instance();

    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.Initialize());

    enn::model::component::OperatorBuilder operator_builder;
    enn::model::component::ParameterBuilder parameter_builder;

    // { Dummy operator1
    std::string op1 = "Normalization";

    float arr_mean[3] = {128.1f, 128.2f, 128.3f};
    std::shared_ptr<enn::model::component::Parameter> param_mean = parameter_builder.set_name("MEAN")
                                                                       .set_buffer_addr(arr_mean)
                                                                       .set_buffer_size(sizeof(arr_mean))
                                                                       .set_data_type(TFlite::TensorType_FLOAT32)
                                                                       .create();

    float arr_scale[3] = {0.0078125f, 0.0078125f, 0.0078125f};
    std::shared_ptr<enn::model::component::Parameter> param_scale = parameter_builder.set_name("SCALE")
                                                                        .set_buffer_addr(arr_scale)
                                                                        .set_buffer_size(sizeof(arr_scale))
                                                                        .set_data_type(TFlite::TensorType_FLOAT32)
                                                                        .create();

    enn::model::component::Operator::Ptr opr1 = operator_builder.set_id(op1.length())
                                                                .set_name(op1)
                                                                .set_code((TFlite::BuiltinOperator)-1)
                                                                .set_accelerator(model::Accelerator::CPU)
                                                                .add_in_tensor(param_mean)
                                                                .add_in_tensor(param_scale)
                                                                .create();
    add_edges(opr1, 1, 1, TFlite::TensorType_UINT8, TFlite::TensorType_FLOAT32);
    // } Dummy operator1

    // { Dummy operator2
    std::string op2 = "Dequantization";

    int32_t arr_frac[3] = {1, 3, 5};
    std::shared_ptr<enn::model::component::Parameter> param_frac_lens = parameter_builder.set_name("FRAC_LEN")
                                                                            .set_buffer_addr(arr_frac)
                                                                            .set_buffer_size(sizeof(arr_frac))
                                                                            .set_data_type(TFlite::TensorType_INT32)
                                                                            .create();

    enn::model::component::Operator::Ptr opr2 = operator_builder.set_id(op2.length())
                                                                .set_name(op2)
                                                                .set_code((TFlite::BuiltinOperator)-1)
                                                                .set_accelerator(model::Accelerator::CPU)
                                                                .add_in_tensor(param_frac_lens)
                                                                .create();
    add_edges(opr2, 1, 1, TFlite::TensorType_INT16, TFlite::TensorType_FLOAT32);
    // } Dummy operator2

    auto opr_list = operator_list_builder.build(MODEL_ID).add_operator(opr1).add_operator(opr2).create();

    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.OpenSubGraph(*opr_list));

    CHECK_OP_LIST_UID(opr_list->get_id());

    ud::UDOperators operators;

    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.get_graph_for_TC(opr_list->get_id(), operators));

    EXPECT_EQ(2, operators->size());

    std::string op_names[] = {"Normalization", "Dequantization"};
    CHECK_OPERATOR(operators, op_names);

    model::memory::BufferTable buffer_table;
    uint8_t in[10][2];
    float out[10][2];
    fill_buffer(in[0], 10, 1);
    fill_buffer(in[1], 10, 10);
    fill_buffer(out[0], 10, 1.f);
    fill_buffer(out[1], 10, 10.f);
    buffer_table.add(0, in[0], sizeof(uint8_t) * 10);
    buffer_table.add(1, in[1], sizeof(uint8_t) * 10);
    buffer_table.add(2, out[0], sizeof(float) * 10);
    buffer_table.add(3, out[1], sizeof(float) * 10);

    auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
        MODEL_ID, opr_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.PrepareSubGraph(*executable_operator_list));

    auto operator_list_execute_request = runtime::OperatorListExecuteRequest(
        executable_operator_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.ExecuteSubGraph(operator_list_execute_request));

    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.CloseSubGraph(*opr_list));

    ud::UDOperators dummy;

    EXPECT_NE(ENN_RET_SUCCESS, cpu_ud.get_graph_for_TC(opr_list->get_id(), dummy));

    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.Deinitialize());
}

TEST_F(ENN_GT_UNIT_TEST_CPU_UD, cpu_ud_test_builtin_operator_with_prepare) {
    ud::cpu::CpuUserDriver& cpu_ud = ud::cpu::CpuUserDriver::get_instance();

    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.Initialize());

    auto opr_list = operator_list_builder.build(MODEL_ID).add_operator(create_builtin_softmax(0.5f, 2)).create();

    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.OpenSubGraph(*opr_list));

    CHECK_OP_LIST_UID(opr_list->get_id());

    ud::UDOperators operators;

    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.get_graph_for_TC(opr_list->get_id(), operators));

    EXPECT_EQ(1, operators->size());

    std::string op_names[] = {"SOFTMAX"};
    CHECK_OPERATOR(operators, op_names);

    float in[10], out[10];
    auto buffer_table = create_buffer_table(in, 10, out, 10);

    auto exec_model_id = EXEC_MODEL_ID(0);
    auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(exec_model_id, opr_list, buffer_table);
    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.PrepareSubGraph(*executable_operator_list));

    auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list);
    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.ExecuteSubGraph(operator_list_execute_request));

    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.CloseSubGraph(*opr_list));

    ud::UDOperators dummy;

    EXPECT_NE(ENN_RET_SUCCESS, cpu_ud.get_graph_for_TC(opr_list->get_id(), dummy));

    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.Deinitialize());
}

TEST_F(ENN_GT_UNIT_TEST_CPU_UD, cpu_ud_test_builtin_operator_without_prepare) {
    ud::cpu::CpuUserDriver& cpu_ud = ud::cpu::CpuUserDriver::get_instance();

    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.Initialize());

    auto opr_list = operator_list_builder.build(MODEL_ID).add_operator(create_builtin_softmax(0.5f, 2)).create();

    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.OpenSubGraph(*opr_list));

    CHECK_OP_LIST_UID(opr_list->get_id());

    ud::UDOperators operators;

    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.get_graph_for_TC(opr_list->get_id(), operators));

    EXPECT_EQ(1, operators->size());

    std::string op_names[] = {"SOFTMAX"};
    CHECK_OPERATOR(operators, op_names);

    float in[10], out[10];
    auto buffer_table = create_buffer_table(in, 10, out, 10);

    auto exec_model_id = EXEC_MODEL_ID(1);
    auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(exec_model_id, opr_list, buffer_table);

    auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list, buffer_table);
    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.ExecuteSubGraph(operator_list_execute_request));

    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.CloseSubGraph(*opr_list));

    ud::UDOperators dummy;

    EXPECT_NE(ENN_RET_SUCCESS, cpu_ud.get_graph_for_TC(opr_list->get_id(), dummy));

    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.Deinitialize());
}

TEST_F(ENN_GT_UNIT_TEST_CPU_UD, cpu_ud_test_builtin_operator_prepare_execute_multiple) {
    ud::cpu::CpuUserDriver& cpu_ud = ud::cpu::CpuUserDriver::get_instance();

    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.Initialize());

    auto opr_list = operator_list_builder.build(MODEL_ID).add_operator(create_builtin_softmax(0.5f, 2)).create();

    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.OpenSubGraph(*opr_list));

    CHECK_OP_LIST_UID(opr_list->get_id());

    float in[10], out[10];
    auto buffer_table_1 = create_buffer_table(in, 10, out, 10);

    auto exec_model_id_1 = EXEC_MODEL_ID(2);
    auto executable_operator_list_1 =
        std::make_shared<runtime::ExecutableOperatorList>(exec_model_id_1, opr_list, buffer_table_1);
    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.PrepareSubGraph(*executable_operator_list_1));

    auto operator_list_execute_request_1 = runtime::OperatorListExecuteRequest(executable_operator_list_1, buffer_table_1);
    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.ExecuteSubGraph(operator_list_execute_request_1));

    auto buffer_table_2 = create_buffer_table(in, 10, out, 10);

    auto exec_model_id_2 = EXEC_MODEL_ID(3);
    auto executable_operator_list_2 =
        std::make_shared<runtime::ExecutableOperatorList>(exec_model_id_2, opr_list, buffer_table_2);
    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.PrepareSubGraph(*executable_operator_list_2));

    auto operator_list_execute_request_2 = runtime::OperatorListExecuteRequest(executable_operator_list_2, buffer_table_2);
    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.ExecuteSubGraph(operator_list_execute_request_2));

    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.CloseSubGraph(*opr_list));

    ud::UDOperators dummy;

    EXPECT_NE(ENN_RET_SUCCESS, cpu_ud.get_graph_for_TC(opr_list->get_id(), dummy));

    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.Deinitialize());
}

TEST_F(ENN_GT_UNIT_TEST_CPU_UD, cpu_ud_test_full_Normalization) {
    ud::cpu::CpuUserDriver& cpu_ud = ud::cpu::CpuUserDriver::get_instance();

    cpu_ud.Initialize();

    enn::model::component::OperatorBuilder operator_builder;

    std::string op1 = "Normalization";

    Dim4 input_dims_ = {1, 3, 32, 32};
    Dim4 output_dims_ = input_dims_;
    Dim4 mean_dims_ = {1, 1, 1, input_dims_.c};
    Dim4 scale_dims_ = {1, 1, 1, input_dims_.c};
    size_t input_size = GetDimSize(input_dims_);
    size_t output_size = GetDimSize(output_dims_);
    size_t mean_size = GetDimSize(mean_dims_);
    size_t scale_size = GetDimSize(scale_dims_);

    // Make parameter
    std::shared_ptr<float> mean;
    std::shared_ptr<float> scale;
    scale.reset(new float[scale_size], std::default_delete<float[]>());
    GenerateRandom(scale.get(), scale_size, 0, 3);
    mean.reset(new float[mean_size], std::default_delete<float[]>());
    GenerateRandom(scale.get(), mean_size, 0, 3);

    for (size_t i = 0; i < scale_size; i++) {
        ENN_TST_PRINT("mean[%ld] = %f, scale[%ld] = %f\n", i, mean.get()[i], i, scale.get()[i]);
    }

    enn::model::component::ParameterBuilder parameter_builder;
    std::shared_ptr<enn::model::component::Parameter> param_mean = parameter_builder.set_name("MEAN")
                                                                       .set_buffer_addr(mean.get())
                                                                       .set_buffer_size(sizeof(float) * mean_size)
                                                                       .set_data_type(TFlite::TensorType_FLOAT32)
                                                                       .create();

    std::shared_ptr<enn::model::component::Parameter> param_scale = parameter_builder.set_name("SCALE")
                                                                        .set_buffer_addr(scale.get())
                                                                        .set_buffer_size(sizeof(float) * scale_size)
                                                                        .set_data_type(TFlite::TensorType_FLOAT32)
                                                                        .create();

    // Make IFM
    enn::model::component::FeatureMapBuilder feature_map_builder;
    enn::model::component::FeatureMap::Ptr ifm =
        feature_map_builder.set_id(0)
            .set_name(std::string("Tensor_000[1]"))
            .set_data_type(TFlite::TensorType_UINT8)
            .set_buffer_index(0)
            .set_buffer_size(input_size)
            .set_shape(std::vector<uint32_t>{input_dims_.n, input_dims_.c, input_dims_.h, input_dims_.w})
            .create();

    // Make OFM
    enn::model::component::FeatureMap::Ptr ofm =
        feature_map_builder.set_id(1)
            .set_name(std::string("Tensor_000[2]"))
            .set_data_type(TFlite::TensorType_FLOAT32)
            .set_buffer_index(1)
            .set_buffer_size(output_size)
            .set_shape(std::vector<uint32_t>{input_dims_.n, input_dims_.c, input_dims_.h, input_dims_.w})
            .create();

    // Make Operator
    enn::model::component::Operator::Ptr opr = operator_builder.set_id(op1.length())
                                                               .set_name(op1)
                                                               .set_code((TFlite::BuiltinOperator)-1)
                                                               .set_accelerator(model::Accelerator::CPU)
                                                               .add_in_tensor(ifm)
                                                               .add_in_tensor(param_mean)
                                                               .add_in_tensor(param_scale)
                                                               .add_out_tensor(ofm)
                                                               .create();

    // Make OperatorList
    auto opr_list = operator_list_builder.build(MODEL_ID).add_operator(opr).create();

    // Open
    cpu_ud.OpenSubGraph(*opr_list);

    // Make BufferTable
    model::memory::BufferTable buffer_table;

    auto in_mem = emm->CreateMemory(data_size[TFlite::TensorType_UINT8] * input_size, enn::EnnMmType::kEnnMmTypeIon);
    auto out_mem = emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * output_size, enn::EnnMmType::kEnnMmTypeIon);

    CHECK_AND_RETURN_VOID(is_invalid_memory(cpu_ud, *opr_list, {in_mem, out_mem}), "CreateMemory() failed");

    GenerateRandom((uint8_t*)in_mem->va, input_size, 0, 5);

    buffer_table.add(0, in_mem->va, in_mem->size);
    buffer_table.add(1, out_mem->va, out_mem->size);

    // Prepare & Execute
    auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
        MODEL_ID, opr_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.PrepareSubGraph(*executable_operator_list));

    auto operator_list_execute_request = runtime::OperatorListExecuteRequest(
        executable_operator_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.ExecuteSubGraph(operator_list_execute_request));

    // Check with reference
    float* refer_output = new float[output_size];
    NormalizationRef((uint8_t*)in_mem->va, refer_output, input_dims_.c, input_dims_.h, input_dims_.w, mean.get(),
                     scale.get(), 0);
    compare((float*)out_mem->va, refer_output, output_size);
    delete[] refer_output;

    emm->DeleteMemory(in_mem);
    emm->DeleteMemory(out_mem);

    cpu_ud.CloseSubGraph(*opr_list);

    cpu_ud.Deinitialize();
}

TEST_F(ENN_GT_UNIT_TEST_CPU_UD, cpu_ud_test_full_Dequantization) {
    ud::cpu::CpuUserDriver& cpu_ud = ud::cpu::CpuUserDriver::get_instance();

    cpu_ud.Initialize();

    enn::model::component::OperatorBuilder operator_builder;

    std::string op_name = "Dequantization";

    Dim4 input_dims_ = {1, 1000, 1, 1};
    Dim4 output_dims_ = input_dims_;
    Dim4 frac_dims_ = input_dims_;
    size_t input_size = GetDimSize(input_dims_);
    size_t output_size = GetDimSize(output_dims_);
    size_t frac_len_size = GetDimSize(frac_dims_);

    // Make parameter
    std::shared_ptr<int32_t> frac_len;
    frac_len.reset(new int32_t[frac_len_size], std::default_delete<int32_t[]>());
    GenerateRandom(frac_len.get(), frac_len_size, 1, 5);

    enn::model::component::ParameterBuilder parameter_builder;
    std::shared_ptr<enn::model::component::Parameter> param_dq = parameter_builder.set_name("FRAC_LEN")
                                                                     .set_buffer_addr(frac_len.get())
                                                                     .set_buffer_size(sizeof(int32_t) * frac_len_size)
                                                                     .set_data_type(TFlite::TensorType_INT32)
                                                                     .create();

    // Make IFM
    enn::model::component::FeatureMapBuilder feature_map_builder;
    enn::model::component::FeatureMap::Ptr ifm =
        feature_map_builder.set_id(0)
            .set_name(std::string("Tensor_000[1]"))
            .set_data_type(TFlite::TensorType_INT8)
            .set_buffer_index(0)
            .set_buffer_size(input_size)
            .set_shape(std::vector<uint32_t>{input_dims_.n, input_dims_.c, input_dims_.h, input_dims_.w})
            .create();

    // Make OFM
    enn::model::component::FeatureMap::Ptr ofm =
        feature_map_builder.set_id(1)
            .set_name(std::string("Tensor_000[2]"))
            .set_data_type(TFlite::TensorType_FLOAT32)
            .set_buffer_index(1)
            .set_buffer_size(output_size)
            .set_shape(std::vector<uint32_t>{input_dims_.n, input_dims_.c, input_dims_.h, input_dims_.w})
            .create();

    // Make Operator
    enn::model::component::Operator::Ptr opr = operator_builder.set_id(op_name.length())
                                                               .set_name(op_name)
                                                               .set_code((TFlite::BuiltinOperator)-1)
                                                               .set_accelerator(model::Accelerator::CPU)
                                                               .add_in_tensor(ifm)
                                                               .add_in_tensor(param_dq)
                                                               .add_out_tensor(ofm)
                                                               .create();

    // Make OperatorList
    auto opr_list = operator_list_builder.build(MODEL_ID).add_operator(opr).create();

    // Open
    cpu_ud.OpenSubGraph(*opr_list);

    // Make BufferTable
    model::memory::BufferTable buffer_table;

    auto in_mem = emm->CreateMemory(data_size[TFlite::TensorType_INT8] * input_size, enn::EnnMmType::kEnnMmTypeIon);
    auto out_mem = emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * output_size, enn::EnnMmType::kEnnMmTypeIon);

    CHECK_AND_RETURN_VOID(is_invalid_memory(cpu_ud, *opr_list, {in_mem, out_mem}), "CreateMemory() failed");

    GenerateRandom((int8_t*)in_mem->va, input_size, 0, 5);

    buffer_table.add(0, in_mem->va, in_mem->size);
    buffer_table.add(1, out_mem->va, out_mem->size);

    // Prepare & Execute
    auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
        MODEL_ID, opr_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.PrepareSubGraph(*executable_operator_list));

    auto operator_list_execute_request = runtime::OperatorListExecuteRequest(
        executable_operator_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.ExecuteSubGraph(operator_list_execute_request));

    // Check with reference
    float* refer_output = new float[output_size];
    DeQuantizationRef((int8_t*)in_mem->va, refer_output, input_size, frac_len.get(), input_dims_.c);
    compare((float*)out_mem->va, refer_output, output_size);
    delete[] refer_output;

    emm->DeleteMemory(in_mem);
    emm->DeleteMemory(out_mem);

    cpu_ud.CloseSubGraph(*opr_list);

    cpu_ud.Deinitialize();
}

TEST_F(ENN_GT_UNIT_TEST_CPU_UD, cpu_ud_test_full_Softmax) {
    ud::cpu::CpuUserDriver& cpu_ud = ud::cpu::CpuUserDriver::get_instance();

    cpu_ud.Initialize();

    enn::model::component::OperatorBuilder operator_builder;

    std::string op1 = "SOFTMAX";
    auto options = new ud::TC_SoftmaxOptions;
    options->beta_ = 1.0f;
    options->axis_ = 1;

    std::string options_name = "SoftmaxOptions";

    Dim4 input_dims_ = {2, 2, 3, 2};
    Dim4 output_dims_ = input_dims_;
    size_t input_size = GetDimSize(input_dims_);
    size_t output_size = GetDimSize(output_dims_);

    // Make IFM
    enn::model::component::FeatureMapBuilder feature_map_builder;
    enn::model::component::FeatureMap::Ptr ifm =
        feature_map_builder.set_id(0)
            .set_name(std::string("Tensor_000[1]"))
            .set_data_type(TFlite::TensorType_FLOAT32)
            .set_buffer_index(0)
            .set_buffer_size(input_size)
            .set_shape(std::vector<uint32_t>{input_dims_.n, input_dims_.c, input_dims_.h, input_dims_.w})
            .create();

    // Make OFM
    enn::model::component::FeatureMap::Ptr ofm =
        feature_map_builder.set_id(1)
            .set_name(std::string("Tensor_000[2]"))
            .set_data_type(TFlite::TensorType_FLOAT32)
            .set_buffer_index(1)
            .set_buffer_size(output_size)
            .set_shape(std::vector<uint32_t>{input_dims_.n, input_dims_.c, input_dims_.h, input_dims_.w})
            .create();

    // Make Operator
    enn::model::component::Operator::Ptr opr =
        operator_builder.set_id(op1.length())
            .set_name(op1)
            .set_accelerator(model::Accelerator::CPU)
            .set_code(TFlite::BuiltinOperator::BuiltinOperator_SOFTMAX)
            .set_option(options_name, options, sizeof(ud::TC_SoftmaxOptions), TFlite::BuiltinOptions_SoftmaxOptions)
            .add_in_tensor(ifm)
            .add_out_tensor(ofm)
            .create();

    // Make OperatorList
    auto opr_list = operator_list_builder.build(MODEL_ID).add_operator(opr).create();

    // Open
    cpu_ud.OpenSubGraph(*opr_list);

    float input_data[24] = {2.1, 1.3,  4.5,  0.2,   0, -1.2, 3.5, -0.1, 2.4, -9.1, 3.6, 2.7,
                            5.3, -8.1, 12.5, -16.9, 0, 0.5,  2.6, 8.1,  7.3, -5.2, 2.6, 8.1};

    float refer_output[24] = {0.19782, 0.80218, 0.89090, 0.99991, 0.02660, 0.01984, 0.80218, 0.19782,
                              0.10910, 0.00009, 0.97340, 0.98016, 0.93703, 0.00000, 0.99451, 0.00001,
                              0.06914, 0.00050, 0.06297, 1.00000, 0.00549, 0.99999, 0.93086, 0.99950};

    // Make BufferTable
    model::memory::BufferTable buffer_table;

    auto in_mem = emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * input_size, enn::EnnMmType::kEnnMmTypeIon);
    auto out_mem = emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * output_size, enn::EnnMmType::kEnnMmTypeIon);

    CHECK_AND_RETURN_VOID(is_invalid_memory(cpu_ud, *opr_list, {in_mem, out_mem}), "CreateMemory() failed");

    // Copy data to ion memory
    memcpy(in_mem->va, input_data, data_size[TFlite::TensorType_FLOAT32] * input_size);

    buffer_table.add(0, in_mem->va, in_mem->size);
    buffer_table.add(1, out_mem->va, out_mem->size);

    // Prepare & Execute
    auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
        MODEL_ID, opr_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.PrepareSubGraph(*executable_operator_list));

    auto operator_list_execute_request = runtime::OperatorListExecuteRequest(
        executable_operator_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    EXPECT_EQ(ENN_RET_SUCCESS, cpu_ud.ExecuteSubGraph(operator_list_execute_request));

    // Check with reference
    compare((float*)out_mem->va, refer_output, output_size);

    emm->DeleteMemory(in_mem);
    emm->DeleteMemory(out_mem);

    cpu_ud.CloseSubGraph(*opr_list);

    cpu_ud.Deinitialize();
    delete options;
}

#ifdef SCHEMA_NNC_V1
TEST_F(ENN_GT_UNIT_TEST_CPU_UD, model_test_olympus_npu_inception_v3_golden_match) {
    const std::string model_file = OLYMPUS::NPU::IV3::NNC;
    auto raw_model = parse_model(model::ModelType::NNC, model_file);
    EXPECT_NE(nullptr, raw_model);

    std::shared_ptr<int32_t> frac_len(new int32_t[1000], std::default_delete<int32_t[]>());
    GenerateRandom(frac_len.get(), 1000, 3, 3);

    std::vector<std::shared_ptr<UT_Operator>> post_operators;

    for (size_t i = 0; i < raw_model->get_operators().size(); ++i) {
        auto op = raw_model->get_operators().at(i);

        if (op->get_name() == "NCP") {
            continue;
        }

        std::shared_ptr<UT_Operator> ut_op = std::make_shared<UT_Operator>(op, raw_model->get_operator_options());
        UT_Tensor::add_tensor(raw_model->get_tensors(), op->get_input_indexes(), ut_op->in_tensors);
        UT_Tensor::add_tensor(raw_model->get_tensors(), op->get_output_indexes(), ut_op->out_tensors);

        // Custom parameter
        if (ut_op->is_builtin == false && ut_op->custom_operator == "Dequantization") {
            ut_op->in_tensors.push_back(
                new UT_Tensor({1, 1000, 1, 1}, TFlite::TensorType_INT32, "FRAC_LEN", frac_len.get()));
        }

        post_operators.push_back(ut_op);
    }

    const std::string input_file = OLYMPUS::NPU::IV3::NCP::GOLDEN;
    const std::string output_file = OLYMPUS::NPU::IV3::GOLDEN;

    TEST_OPERATOR<float>(post_operators, input_file, output_file);
}

#else

TEST_F(ENN_GT_UNIT_TEST_CPU_UD, model_test_pamir_npu_inception_v3_golden_match) {
    const std::string model_file = PAMIR::NPU::IV3::NNC;
    auto raw_model = parse_model(model::ModelType::NNC, model_file);
    EXPECT_NE(nullptr, raw_model);

    bool prev_npu = true;
    std::vector<std::shared_ptr<UT_Operator>> pre_operators, post_operators;

    for (size_t i = 0; i < raw_model->get_operators().size(); ++i) {
        auto op = raw_model->get_operators().at(i);

        if (static_cast<TFlite::BuiltinOperator>(op->get_op_code() == TFlite::BuiltinOperator_ENN_NPU)) {
            prev_npu = false;
            continue;
        }

        std::shared_ptr<UT_Operator> ut_op = std::make_shared<UT_Operator>(op, raw_model->get_operator_options());
        UT_Tensor::add_tensor(raw_model->get_tensors(), op->get_input_indexes(), ut_op->in_tensors);
        UT_Tensor::add_tensor(raw_model->get_tensors(), op->get_output_indexes(), ut_op->out_tensors);

        prev_npu ? pre_operators.push_back(ut_op) : post_operators.push_back(ut_op);
    }

    const std::string input_file_pre = PAMIR::NPU::IV3::INPUT;
    const std::string output_file_pre = PAMIR::NPU::IV3::NCP::INPUT;

    TEST_OPERATOR<float>(pre_operators, input_file_pre, output_file_pre);

    const std::string input_file_post = PAMIR::NPU::IV3::NCP::GOLDEN;
    const std::string output_file_post = PAMIR::NPU::IV3::GOLDEN;

    TEST_OPERATOR<float>(post_operators, input_file_post, output_file_post);
}
#endif

}  // namespace internal
}  // namespace test
}  // namespace enn
