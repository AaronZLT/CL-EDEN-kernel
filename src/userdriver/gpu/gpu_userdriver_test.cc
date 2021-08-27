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
 * @file gpu_userdriver_test.cc
 * @author Byungjin Jung
 * @date 2021_03_11
 */

#include <random>

#include "gtest/gtest.h"

#include "client/enn_api-type.h"
#include "common/enn_memory_manager.h"
#include "model/component/operator/operator_builder.hpp"
#include "model/component/operator/operator_list_builder.hpp"
#include "model/component/tensor/feature_map_builder.hpp"
#include "model/component/tensor/parameter_builder.hpp"
#include "model/schema/flatbuffers/flatbuffers.h"
#include "model/schema/schema_nnc.h"
#include "userdriver/gpu/gpu_userdriver.h"
#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/op_test/test_function.hpp"

#ifndef ENN_ANDROID_BUILD
#define TEST_FILE_PATH "./test_data/"
#else
#define TEST_FILE_PATH "/data/vendor/enn/models/"
#endif  // ENN_ANDROID_BUILD

#define TEST_FILE(M) (TEST_FILE_PATH + std::string(M))

template <typename T> class _Parameter {
public:
    T value;
    std::string name;
    TFlite::TensorType tensor_type;

    _Parameter(std::string name_, T value_, TFlite::TensorType tensor_type_) :
        name(name_), value(value_), tensor_type(tensor_type_) {}
};

std::unordered_map<std::string, uint64_t> operator_id_map_gpu;

namespace enn {
namespace test {
namespace internal {

namespace {
auto MODEL_ID = identifier::Identifier<identifier::FullIDType, 0x7FFF, 49>(0x10000000);
auto EXEC_MODEL_ID(uint8_t offset) {
    return identifier::Identifier<identifier::FullIDType, 0x7FFF, 49>(0x10000000 + offset);
}
}  // namespace

class ENN_GT_UNIT_TEST_GPU_UD : public testing::Test {
public:
    ENN_GT_UNIT_TEST_GPU_UD() {
        emm = std::make_unique<enn::EnnMemoryManager>();
        emm->init();
    }

    ~ENN_GT_UNIT_TEST_GPU_UD() { emm->deinit(); }

    std::unique_ptr<enn::EnnMemoryManager> emm;
    flatbuffers::FlatBufferBuilder build_flatbuffer;

    bool is_invalid_memory(ud::gpu::GpuUserDriver &gpu_ud,
                           model::component::OperatorList &operator_list,
                           std::vector<EnnBufferCore::Ptr> buffers) {
        bool is_invalid_memory = false;
        for (auto mem : buffers) {
            if (mem == nullptr) {
                for (auto mem : buffers) {
                    emm->DeleteMemory(mem);
                }
                gpu_ud.CloseSubGraph(operator_list);
                gpu_ud.Deinitialize();
                is_invalid_memory = true;
                break;
            }
        }
        EXPECT_FALSE(is_invalid_memory);
        return is_invalid_memory;
    }

    std::unordered_map<TFlite::TensorType, uint32_t> data_size = {
        {TFlite::TensorType::TensorType_FLOAT32, sizeof(float)},
        {TFlite::TensorType::TensorType_INT32, sizeof(int32_t)},
        {TFlite::TensorType::TensorType_UINT8, sizeof(uint8_t)},
        {TFlite::TensorType::TensorType_BOOL, sizeof(bool)},
        {TFlite::TensorType::TensorType_INT16, sizeof(int16_t)},
        {TFlite::TensorType::TensorType_INT8, sizeof(int8_t)},
    };

    // add in/out edges to opr
    void add_edges(enn::model::component::Operator::Ptr opr,
                   int in_num,
                   int out_num,
                   TFlite::TensorType in_type,
                   TFlite::TensorType out_type) {
        enn::model::component::FeatureMapBuilder feature_map_builder;
        enn::model::component::OperatorBuilder operator_builder{opr};

        // add in edges as many as in_num
        for (uint32_t i = 0; i < in_num; i++) {
            char name[5];
            sprintf(name, "IFM%d", i);
            uint32_t h = (i + 1) * 2, w = h;
            std::vector<uint32_t> shape = {1, 1, h, w};
            enn::model::component::FeatureMap::Ptr feature_map = feature_map_builder.set_id(i)
                                                                     .set_name(std::string(name))
                                                                     .set_data_type(in_type)
                                                                     .set_buffer_index(i)
                                                                     .set_buffer_size(h * w * data_size[in_type])
                                                                     .set_shape(shape)
                                                                     .create();
            operator_builder.add_in_tensor(feature_map);
        }
        // add out edges as many as edge_cnt
        for (uint32_t i = 0; i < out_num; i++) {
            char name[5];
            sprintf(name, "OFM%d", i);
            uint32_t h = (i + 1) * 2, w = h;
            std::vector<uint32_t> shape = {1, 1, h, w};
            enn::model::component::FeatureMap::Ptr feature_map = feature_map_builder.set_id(i)
                                                                     .set_name(std::string(name))
                                                                     .set_data_type(out_type)
                                                                     .set_buffer_index(i)
                                                                     .set_buffer_size(h * w * data_size[out_type])
                                                                     .set_shape(shape)
                                                                     .create();
            operator_builder.add_out_tensor(feature_map);
        }
    }

    template <typename T1, typename T2> inline void fill_buffer(T1 &data, int size, T2 value) {
        for (int i = 0; i < size; i++) {
            data[i] = i + value;
        }
    }

    ud::gpu::GpuUserDriver &create_gpu_ud() const {
        ud::gpu::GpuUserDriver &gpu_ud = ud::gpu::GpuUserDriver::get_instance();
        EXPECT_EQ(gpu_ud.Initialize(), ENN_RET_SUCCESS);
        gpu_ud.disable_profile_for_TC();
        return gpu_ud;
    }

    TFlite::QuantizationParameters *create_quantization(const float &scale = 1, const int64_t &zero_point = 0) {
        TFlite::QuantizationParameters *quant_param = nullptr;
        auto quant_info = TFlite::CreateQuantizationParameters(build_flatbuffer,
                                                               build_flatbuffer.CreateVector<float>({0}),
                                                               build_flatbuffer.CreateVector<float>({0}),
                                                               build_flatbuffer.CreateVector<float>({scale}),
                                                               build_flatbuffer.CreateVector<int64_t>({zero_point}));

        build_flatbuffer.Finish(quant_info);
        quant_param =
            (TFlite::QuantizationParameters *)build_flatbuffer.GetBufferPointer() +
            (flatbuffers::EndianScalar(*reinterpret_cast<flatbuffers::uoffset_t *>(build_flatbuffer.GetBufferPointer())));
        return quant_param;
    }

#ifndef SCHEMA_NNC_V1
    TFlite::SymmPerChannelQuantParamters *create_perchannel_quant(const std::vector<float> &scales,
                                                                  const uint32_t &channel_dim) {
        TFlite::SymmPerChannelQuantParamters *quant_perchannel = nullptr;
        auto perchannel_info = TFlite::CreateSymmPerChannelQuantParamters(
            build_flatbuffer, build_flatbuffer.CreateVector<float>(scales), channel_dim);

        build_flatbuffer.Finish(perchannel_info);
        quant_perchannel =
            (TFlite::SymmPerChannelQuantParamters *)build_flatbuffer.GetBufferPointer() +
            (flatbuffers::EndianScalar(*reinterpret_cast<flatbuffers::uoffset_t *>(build_flatbuffer.GetBufferPointer())));
        return quant_perchannel;
    }
#endif

    template <typename T>
    std::shared_ptr<enn::model::component::Parameter> create_parameter(
        enn::model::component::ParameterBuilder &parameter_builder,
        char name[],
        T *addr,
        const NDims &shape,
        TFlite::TensorType tensor_type,
        uint32_t id = 0,
        const float &scale = 1.0f,
        const int64_t &zero_point = 0,
        std::vector<float> scales = {},
        const uint32_t &channel_dim = 0) {
        TFlite::QuantizationParameters *quant_param = nullptr;
#ifndef SCHEMA_NNC_V1
        TFlite::SymmPerChannelQuantParamters *quant_perchannel = nullptr;
#endif
        if (tensor_type == TFlite::TensorType_INT8 || tensor_type == TFlite::TensorType_UINT8) {
            quant_param = create_quantization(scale, zero_point);
#ifndef SCHEMA_NNC_V1
            if (scales.size()) {
                quant_perchannel = create_perchannel_quant(scales, channel_dim);
            }
#endif
        }

        return parameter_builder.set_id(id)
            .set_name(name)
            .set_buffer_addr(addr)
            .set_buffer_size(sizeof(T) * GetDimSize(shape))
            .set_data_type(tensor_type)
            .set_shape(shape)
            .set_quantization_parameters(quant_param)
#ifdef SCHEMA_NNC_V1
            .set_symm_per_channel_quant_parameters(nullptr)
#else
            .set_symm_per_channel_quant_parameters(quant_perchannel)
#endif
            .create();
    }

    enn::model::component::FeatureMap::Ptr create_feature_map(enn::model::component::FeatureMapBuilder &feature_map_builder,
                                                              std::string name,
                                                              uint32_t buffer_index,
                                                              const NDims &shape,
                                                              TFlite::TensorType tensor_type,
                                                              const bool &is_subgraph_input = true,
                                                              uint32_t id = 0,
                                                              const float &scale = 1.0f,
                                                              const int64_t &zero_point = 0,
                                                              const std::vector<float>& scales = {},
                                                              const uint32_t& channel_dim = 0) {
        TFlite::QuantizationParameters *quant_param = nullptr;
#ifndef SCHEMA_NNC_V1
        TFlite::SymmPerChannelQuantParamters *quant_perchannel = nullptr;
#endif

        if (tensor_type == TFlite::TensorType_INT8 || tensor_type == TFlite::TensorType_UINT8) {
            quant_param = create_quantization(scale, zero_point);
#ifndef SCHEMA_NNC_V1
            if (scales.size()) {
                quant_perchannel = create_perchannel_quant(scales, channel_dim);
            }
#endif
        }

        const auto feature_map_type = is_subgraph_input ? enn::model::component::FeatureMap::Type::SUBGRAPH_INPUT
                                                        : enn::model::component::FeatureMap::Type::SUBGRAPH_OUTPUT;

        return feature_map_builder.set_id(id)
            .set_name(name)
            .set_buffer_index(buffer_index)
            .set_buffer_size(data_size[tensor_type] * GetDimSize(shape))
            .set_data_type(tensor_type)
            .set_shape(shape)
            .set_quantization_parameters(quant_param)
            .set_type(feature_map_type)
#ifdef SCHEMA_NNC_V1
            .set_symm_per_channel_quant_parameters(nullptr)
#else
            .set_symm_per_channel_quant_parameters(quant_perchannel)
#endif
            .create();
    }

    enn::model::component::Operator::Ptr create_operator(
        enn::model::component::OperatorBuilder &operator_builder,
        std::string name,
        TFlite::BuiltinOperator op_code,
        std::vector<enn::model::component::FeatureMap::Ptr> in_fms,
        std::vector<std::shared_ptr<enn::model::component::Parameter>> params,
        std::vector<enn::model::component::FeatureMap::Ptr> out_fms,
        std::string option_name,
        void *option_data,
        uint32_t option_data_size,
        TFlite::BuiltinOptions option_code,
        uint32_t id = 0) {
        operator_builder.set_id(id)
            .set_name(name)
            .set_code(op_code)
            .set_accelerator(model::Accelerator::GPU)
            .set_option(option_name, option_data, option_data_size, option_code);
        for (auto ifm : in_fms) {
            operator_builder.add_in_tensor(ifm);
        }
        for (auto param : params) {
            operator_builder.add_in_tensor(param);
        }
        for (auto ofm : out_fms) {
            operator_builder.add_out_tensor(ofm);
        }

        return operator_builder.create();
    }

    // Referenced from test_utils.h {
    template <typename T1, typename T2, typename T3>
    inline bool GenerateRandom(T1 *ptr, const uint32_t &size, const T2 &min, const T3 &max) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(min, max);
        for (uint32_t idx = 0; idx < size; idx++) {
            *(ptr + idx) = dis(gen);
        }
        return true;
    }

    template <typename T1, typename T2>
    inline void compare(const T1 *real, const T2 *expected, size_t size, float error_threshold_ = (1e-5)) {
        for (size_t i = 0; i < size; i++) {
            EXPECT_NEAR(real[i], expected[i], error_threshold_) << i;
        }
    }
    // } Referenced from test_utils.h

    void CHECK_OP_LIST_UID(uint64_t op_list_id) {
        uint64_t uid = (op_list_id & 0x0000000000FFFF00) >> 8;
        ENN_DBG_PRINT("op_list_id: 0x%" PRIx64 ", uid: 0x%" PRIx64 "\n", op_list_id, uid);
        EXPECT_NE(0, uid);
    }

    enn::model::component::OperatorListBuilder operator_list_builder;
    enn::model::component::OperatorBuilder operator_builder;
    enn::model::component::FeatureMapBuilder feature_map_builder;
    enn::model::component::ParameterBuilder parameter_builder;
};

TEST_F(ENN_GT_UNIT_TEST_GPU_UD, gpu_ud_test_initialize) {
    ud::gpu::GpuUserDriver &gpu_ud = ud::gpu::GpuUserDriver::get_instance();
    EXPECT_EQ(ENN_RET_SUCCESS, gpu_ud.Initialize());
}

TEST_F(ENN_GT_UNIT_TEST_GPU_UD, gpu_ud_test_open_builtin_operator) {
    ud::gpu::GpuUserDriver &gpu_ud = ud::gpu::GpuUserDriver::get_instance();
    enn::model::component::OperatorBuilder operator_builder;

    // { Dummy operator1
    std::string op1 = "SOFTMAX";
    auto options = new ud::TC_SoftmaxOptions;
    options->beta_ = 2.0f;
    options->axis_ = 2;

    enn::model::component::Operator::Ptr opr = operator_builder.set_id(op1.length())
                                                   .set_name(op1)
                                                   .set_accelerator(model::Accelerator::GPU)
                                                   .set_code(TFlite::BuiltinOperator::BuiltinOperator_SOFTMAX)
                                                   .set_option(EnumNameBuiltinOptions(TFlite::BuiltinOptions_SoftmaxOptions),
                                                               options,
                                                               sizeof(ud::TC_SoftmaxOptions),
                                                               TFlite::BuiltinOptions_SoftmaxOptions)
                                                   .create();
    add_edges(opr, 1, 1, TFlite::TensorType_FLOAT32, TFlite::TensorType_FLOAT32);
    // } Dummy operator1

    // GPU op_constructor->create_ud_operator is not prepared yet. So, operator
    // size is zero.
    auto attribute = std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_CAFFE, true);
    auto op_list = operator_list_builder.build(MODEL_ID).add_operator(opr).set_attribute(attribute).create();
    EXPECT_EQ(ENN_RET_SUCCESS, gpu_ud.OpenSubGraph(*op_list));
    CHECK_OP_LIST_UID(op_list->get_id());

    operator_id_map_gpu["SOFTMAX_builtin"] = op_list->get_id();
}

TEST_F(ENN_GT_UNIT_TEST_GPU_UD, gpu_ud_test_check_count_after_open_builtin_operator) {
    ud::gpu::GpuUserDriver &gpu_ud = ud::gpu::GpuUserDriver::get_instance();

    uint64_t operator_list_id = operator_id_map_gpu["SOFTMAX_builtin"];
    ENN_DBG_COUT << "operator_list_id = " << operator_list_id << std::endl;
    ud::UDOperators operators;
    EXPECT_EQ(ENN_RET_SUCCESS, gpu_ud.get_graph_for_TC(operator_list_id, operators));

    // GPU op_constructor->create_ud_operator is not prepared yet. So, operator size is zero.
    EXPECT_EQ(1, operators->size());
}

TEST_F(ENN_GT_UNIT_TEST_GPU_UD, gpu_ud_test_quant_scale_zp) {
    ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();

    // Define test case
    std::string op_name = "RELU";
    const std::string option_name = "ReluOptions";
    TFlite::BuiltinOperator op_code = TFlite::BuiltinOperator::BuiltinOperator_RELU;
#ifdef SCHEMA_NNC_V1
    TFlite::BuiltinOptions op_option = TFlite::BuiltinOptions::BuiltinOptions_ReluOptions;
#else
    TFlite::BuiltinOptions op_option = TFlite::BuiltinOptions::BuiltinOptions_ENN_ReluOptions;
#endif

    NDims dims = {2, 2};
    float scale = 1.0f;
    int64_t zero_point = 128;
    std::vector<uint8_t> input_data = {0, 1, 126, 127};
    std::vector<uint8_t> inference_output_data = {128, 128, 128, 128};
    float error_threshold = 0;

    // Make IFM & OFM
    auto ifm = create_feature_map(
        feature_map_builder, std::string("Tensor_000[1]"), 0, dims, TFlite::TensorType_UINT8, true, 0, scale, zero_point);
    auto ofm = create_feature_map(
        feature_map_builder, std::string("Tensor_000[2]"), 1, dims, TFlite::TensorType_UINT8, false, 1, scale, zero_point);

   // Make option
    flatbuffers::FlatBufferBuilder build_option;
#ifdef SCHEMA_NNC_V1
    auto tmp = TFlite::CreateReluOptions(build_option, 0);
#else
    auto tmp = TFlite::CreateENN_ReluOptions(build_option, 0);
#endif
    build_option.Finish(tmp);
    void *option_addr =
        build_option.GetBufferPointer() +
        (flatbuffers::EndianScalar(*reinterpret_cast<flatbuffers::uoffset_t *>(build_option.GetBufferPointer())));

    // Make operator
    auto op = create_operator(operator_builder,
                              op_name,
                              op_code,
                              {ifm},
                              {},
                              {ofm},
                              option_name,
                              option_addr,
                              build_option.GetSize(),
                              op_option);

    // Make OperatorList
    auto attribute = std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_CAFFE, true);
    auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();

    // OpenSubGraph
    EXPECT_EQ(ENN_RET_SUCCESS, gpu_ud.OpenSubGraph(*op_list));
    CHECK_OP_LIST_UID(op_list->get_id());

    // Make BufferTable
    model::memory::BufferTable buffer_table;

    auto in_mem =
        emm->CreateMemory(data_size[TFlite::TensorType_UINT8] * GetDimSize(dims), enn::EnnMmType::kEnnMmTypeIon);
    auto out_mem =
        emm->CreateMemory(data_size[TFlite::TensorType_UINT8] * GetDimSize(dims), enn::EnnMmType::kEnnMmTypeIon);

    CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem, out_mem}), "CreateMemory() failed");

    in_mem->va = static_cast<void *>(input_data.data());
    buffer_table.add(0, in_mem->va, in_mem->size);
    buffer_table.add(1, out_mem->va, out_mem->size);

    // Prepare & Execute
    auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
        MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    EXPECT_EQ(gpu_ud.PrepareSubGraph(*executable_operator_list), ENN_RET_SUCCESS);

    auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list);
    EXPECT_EQ(gpu_ud.ExecuteSubGraph(operator_list_execute_request), ENN_RET_SUCCESS);
    compare((uint8_t *)out_mem->va, inference_output_data.data(), GetDimSize(dims), error_threshold);

    emm->DeleteMemory(in_mem);
    emm->DeleteMemory(out_mem);
    gpu_ud.CloseSubGraph(*op_list);

    gpu_ud.Deinitialize();
}

#ifndef SCHEMA_NNC_V1
TEST_F(ENN_GT_UNIT_TEST_GPU_UD, gpu_ud_test_quant_perchannel) {
    ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();
    enn::model::component::OperatorBuilder operator_builder;

    NDims input_dims = {1, 2, 3, 1};
    NDims weight_dims = {3, 1, 1, 2};
    NDims bias_dims = {3, 1, 1, 1};
    NDims output_dims = {1, 3, 3, 1};
    float input_scale = 0.5;
    float output_scale = 1.0;
    int64_t zero_point = 128;

    Dim2 stride = {1, 1};
    Dim2 dilation = {1, 1};

    // Make parameter
    uint8_t input_data[] = {138, 138, 138, 108, 108, 108};
    int8_t weight_data[] = {1, 2, 1, 2, 1, 2};
    int bias_data[] = {4, 4, 4};
    uint8_t expect_data[] = {121, 121, 121, 118, 118, 118, 115, 115, 115};
    std::vector<float> scales_data = {0.5f, 0.75f, 1.0f};
    float error_threshold = 0;

    auto ifm = create_feature_map(
        feature_map_builder, std::string("Tensor_000[1]"), 0, input_dims, TFlite::TensorType_UINT8, true, 0, input_scale, zero_point);
    auto ofm = create_feature_map(feature_map_builder,
                                  std::string("Tensor_000[2]"),
                                  1,
                                  output_dims,
                                  TFlite::TensorType_UINT8, false, 1,
                                  output_scale,
                                  zero_point);

    auto param_kernel = create_parameter(
        parameter_builder, const_cast<char *>("Kernel"), weight_data, weight_dims, TFlite::TensorType_UINT8, 0, 0.0f, 0, scales_data, 0);
    auto param_bias = create_parameter(parameter_builder, const_cast<char *>("Bias"), bias_data, bias_dims, TFlite::TensorType_INT32);

    flatbuffers::FlatBufferBuilder build_data;
    TFlite::ActivationFunctionType activation = TFlite::ActivationFunctionType::ActivationFunctionType_NONE;
    TFlite::Padding padding = TFlite::Padding::Padding_SAME;

    auto tmp = TFlite::CreateConv2DOptions(build_data,
                                           padding,
                                           stride.h,
                                           stride.w,
                                           activation,
                                           dilation.h,
                                           dilation.w,
                                           build_data.CreateVector<int32_t>({0, 0, 0, 0}),
                                           true);
    build_data.Finish(tmp);
    void *base_addr =
        build_data.GetBufferPointer() +
        (flatbuffers::EndianScalar(*reinterpret_cast<flatbuffers::uoffset_t *>(build_data.GetBufferPointer())));
    enn::model::component::Operator::Ptr op =
        operator_builder.set_id(0)
            .set_name(TFlite::EnumNameBuiltinOperator(TFlite::BuiltinOperator_CONV_2D))
            .set_code(TFlite::BuiltinOperator_CONV_2D)
            .set_accelerator(model::Accelerator::GPU)
            .add_in_tensor(ifm)
            .add_in_tensor(param_kernel)
            .add_in_tensor(param_bias)
            .add_out_tensor(ofm)
            .set_option(TFlite::EnumNameBuiltinOptions(TFlite::BuiltinOptions_Conv2DOptions),
                        base_addr,
                        build_data.GetSize(),
                        TFlite::BuiltinOptions_Conv2DOptions)
            .create();
    // Make OperatorList
    auto attribute = std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_ANDROID_NN, false);
    auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();
    gpu_ud.OpenSubGraph(*op_list);

    // Make BufferTable
    model::memory::BufferTable buffer_table;

    auto in_mem =
        emm->CreateMemory(data_size[TFlite::TensorType_UINT8] * GetDimSize(input_dims), enn::EnnMmType::kEnnMmTypeIon);
    auto out_mem =
        emm->CreateMemory(data_size[TFlite::TensorType_UINT8] * GetDimSize(output_dims), enn::EnnMmType::kEnnMmTypeIon);

    CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem, out_mem}), "CreateMemory() failed");

    in_mem->va = static_cast<void *>(input_data);
    buffer_table.add(0, in_mem->va, in_mem->size);
    buffer_table.add(1, out_mem->va, out_mem->size);

    // Prepare & Execute
    auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
        MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    EXPECT_EQ(gpu_ud.PrepareSubGraph(*executable_operator_list), ENN_RET_SUCCESS);

    auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list);
    EXPECT_EQ(gpu_ud.ExecuteSubGraph(operator_list_execute_request), ENN_RET_SUCCESS);
    compare((uint8_t *)out_mem->va, expect_data, GetDimSize(output_dims), error_threshold);

    emm->DeleteMemory(in_mem);
    emm->DeleteMemory(out_mem);
    gpu_ud.CloseSubGraph(*op_list);

    gpu_ud.Deinitialize();
}
#endif

// TODO(all): move operators ud test into single test file
TEST_F(ENN_GT_UNIT_TEST_GPU_UD, gpu_ud_test_normalization) {
    ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();
    enn::model::component::OperatorBuilder operator_builder;
    NDims input_dims = {1, 1, 2, 2};
    NDims output_dims = {1, 1, 2, 2};
    NDims mean_dims = {1, 1, 1, 1};
    NDims scale_dims = {1, 1, 1, 1};
    size_t input_size = GetDimSize(input_dims);
    size_t output_size = GetDimSize(output_dims);
    size_t mean_size = GetDimSize(mean_dims);
    size_t scale_size = GetDimSize(scale_dims);

    // Make parameter
    std::shared_ptr<float> mean;
    std::shared_ptr<float> scale;
    scale.reset(new float[scale_size], std::default_delete<float[]>());
    GenerateRandom(scale.get(), scale_size, 0, 3);
    mean.reset(new float[mean_size], std::default_delete<float[]>());
    GenerateRandom(mean.get(), mean_size, 0, 3);
    float error_threshold = 0.015;

    auto ifm =
        create_feature_map(feature_map_builder, std::string("Tensor_000[1]"), 0, input_dims, TFlite::TensorType_UINT8, true);
    auto ofm =
        create_feature_map(feature_map_builder, std::string("Tensor_000[2]"), 1, output_dims, TFlite::TensorType_FLOAT32, false);

    auto param_mean = create_parameter(parameter_builder, const_cast<char *>("MEAN"), mean.get(), mean_dims, TFlite::TensorType_FLOAT32);
    auto param_scale = create_parameter(parameter_builder, const_cast<char *>("SCALE"), scale.get(), scale_dims, TFlite::TensorType_FLOAT32);

    // Make Operator
    enn::model::component::Operator::Ptr op = operator_builder
                                                  .set_id(0)
#ifdef SCHEMA_NNC_V1
                                                  .set_name("Normalization")
                                                  .set_code((TFlite::BuiltinOperator)-1)
#else
                                                  .set_name(TFlite::EnumNameBuiltinOperator(
                                                      TFlite::BuiltinOperator_ENN_NORMALIZATION))
                                                  .set_code(TFlite::BuiltinOperator_ENN_NORMALIZATION)
#endif

                                                  .set_accelerator(model::Accelerator::GPU)
                                                  .add_in_tensor(ifm)
                                                  .add_in_tensor(param_mean)
                                                  .add_in_tensor(param_scale)
                                                  .add_out_tensor(ofm)
                                                  .create();
    // Make OperatorList
    auto attribute = std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_CAFFE, true);
    auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();

    // Open
    gpu_ud.OpenSubGraph(*op_list);

    // Make BufferTable
    model::memory::BufferTable buffer_table;

    auto in_mem = emm->CreateMemory(data_size[TFlite::TensorType_UINT8] * input_size, enn::EnnMmType::kEnnMmTypeIon);
    auto out_mem = emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * output_size, enn::EnnMmType::kEnnMmTypeIon);

    CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem, out_mem}), "CreateMemory() failed");

    GenerateRandom((uint8_t *)in_mem->va, input_size, 0, 5);
    buffer_table.add(0, in_mem->va, in_mem->size);
    buffer_table.add(1, out_mem->va, out_mem->size);

    // Prepare & Execute
    auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
        MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    EXPECT_EQ(gpu_ud.PrepareSubGraph(*executable_operator_list), ENN_RET_SUCCESS);

    auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list);
    EXPECT_EQ(gpu_ud.ExecuteSubGraph(operator_list_execute_request), ENN_RET_SUCCESS);

    // Check with reference
    float *refer_output = new float[output_size];
    float max = -65532;
    NormalizationRef((uint8_t *)in_mem->va,
                     refer_output,
                     input_dims.at(1),
                     input_dims.at(2),
                     input_dims.at(3),
                     mean.get(),
                     scale.get(),
                     0,
                     max);
    // use FP16 as default precision,
    error_threshold = max / 500 > 0.01 ? max / 500 : 0.01;

    compare((float *)out_mem->va, refer_output, output_size, error_threshold);
    delete[] refer_output;

    emm->DeleteMemory(in_mem);
    emm->DeleteMemory(out_mem);
    gpu_ud.CloseSubGraph(*op_list);

    gpu_ud.Deinitialize();
}

TEST_F(ENN_GT_UNIT_TEST_GPU_UD, gpu_ud_test_convlution) {
    ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();
    enn::model::component::OperatorBuilder operator_builder;

    NDims input_dims = {1, 1, 4, 4};
    NDims output_dims = {1, 1, 4, 4};
    NDims weight_dims = {1, 1, 3, 2};
    NDims bias_dims = {1, 1, 1, 1};

    // Make parameter
    float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16};
    float weight_data[] = {1, 2, 3, 4, 5, 6};
    float bias_data[] = {0};
    float expect_data[] = {72, 90, 108, 52, 149, 170, 191, 88, 244, 265, 184, 44, 131, 141, 83, 12};
    float error_threshold = 0.17;
    Dim2 stride = {1, 1};
    Dim2 dilation = {1, 1};

    auto ifm =
        create_feature_map(feature_map_builder, std::string("Tensor_000[1]"), 0, input_dims, TFlite::TensorType_FLOAT32, true);
    auto ofm =
        create_feature_map(feature_map_builder, std::string("Tensor_000[2]"), 1, output_dims, TFlite::TensorType_FLOAT32, false);

    auto param_kernel = create_parameter(parameter_builder, const_cast<char *>("Kernel"), weight_data, weight_dims, TFlite::TensorType_FLOAT32);
    auto param_bias = create_parameter(parameter_builder, const_cast<char *>("Bias"), bias_data, bias_dims, TFlite::TensorType_FLOAT32);

    flatbuffers::FlatBufferBuilder build_data;
    TFlite::ActivationFunctionType activation = TFlite::ActivationFunctionType::ActivationFunctionType_RELU;
    TFlite::Padding padding = TFlite::Padding::Padding_SAME;

    auto tmp = TFlite::CreateConv2DOptions(
        build_data, padding, static_cast<uint32_t>(stride.h), stride.w, activation, dilation.h, dilation.w);
    build_data.Finish(tmp);
    void *base_addr =
        build_data.GetBufferPointer() +
        (flatbuffers::EndianScalar(*reinterpret_cast<flatbuffers::uoffset_t *>(build_data.GetBufferPointer())));
    enn::model::component::Operator::Ptr op =
        operator_builder.set_id(0)
            .set_name(TFlite::EnumNameBuiltinOperator(TFlite::BuiltinOperator_CONV_2D))
            .set_code(TFlite::BuiltinOperator_CONV_2D)
            .set_accelerator(model::Accelerator::GPU)
            .add_in_tensor(ifm)
            .add_in_tensor(param_kernel)
            .add_in_tensor(param_bias)
            .add_out_tensor(ofm)
            .set_option(TFlite::EnumNameBuiltinOptions(TFlite::BuiltinOptions_Conv2DOptions),
                        base_addr,
                        build_data.GetSize(),
                        TFlite::BuiltinOptions_Conv2DOptions)
            .create();
    // Make OperatorList
    auto attribute = std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_TENSORFLOW, false);
    auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();
    gpu_ud.OpenSubGraph(*op_list);

    // Make BufferTable
    model::memory::BufferTable buffer_table;

    auto in_mem =
        emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * GetDimSize(input_dims), enn::EnnMmType::kEnnMmTypeIon);
    auto out_mem =
        emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * GetDimSize(output_dims), enn::EnnMmType::kEnnMmTypeIon);

    CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem, out_mem}), "CreateMemory() failed");

    in_mem->va = static_cast<void *>(input_data);
    buffer_table.add(0, in_mem->va, in_mem->size);
    buffer_table.add(1, out_mem->va, out_mem->size);

    // Prepare & Execute
    auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
        MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    EXPECT_EQ(gpu_ud.PrepareSubGraph(*executable_operator_list), ENN_RET_SUCCESS);

    auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list);
    EXPECT_EQ(gpu_ud.ExecuteSubGraph(operator_list_execute_request), ENN_RET_SUCCESS);
    compare((float *)out_mem->va, expect_data, GetDimSize(output_dims), error_threshold);

    emm->DeleteMemory(in_mem);
    emm->DeleteMemory(out_mem);
    gpu_ud.CloseSubGraph(*op_list);

    gpu_ud.Deinitialize();
}

TEST_F(ENN_GT_UNIT_TEST_GPU_UD, gpu_ud_test_deconvlution) {
    ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();
    enn::model::component::OperatorBuilder operator_builder;

    float input_data[] = {1.86387,   0.823307,  0.172173,  -1.31165, 1.62649,    0.67635,  0.0448795, -0.286381,  0.583082,
                          0.884495,  -0.291902, -0.465355, -1.01752, 2.90559,    0.98394,  0.627009,  -1.42522,   1.11258,
                          1.37396,   -0.753714, 0.191197,  0.403568, -0.0168294, 0.250767, 0.942337,  -0.682471,  -1.78194,
                          -0.018691, -1.93659,  -1.06201,  0.481817, -0.283937,  0.445748, 2.0653,    0.00736003, 0.193207};
    float weight_data[] = {-0.188587,
                           0.707552,
                           -0.760529,
                           0.681911,
                           0.696376,
                           -0.277342,
                           -0.251224,
                           -0.284218,
                           0.497616,
                           1.84801,
                           0.444945,
                           0.176324};
    float bias_data[] = {0.936025, -0.29215, -2.92477};
    float reference_output[] = {2.20702,   0.309876,  1.49745,    0.805082,  0.00796517, 0.629291,  2.08685,   0.808474,
                                -0.304905, -0.220204, -0.210755,  -0.438634, -0.537457,  -0.495423, -0.211193, -0.616211,
                                -3.10419,  -1.63195,  -2.41245,   -2.48698,  -1.76606,   -3.63399,  -5.55859,  -2.37113,
                                1.87294,   1.50925,   0.42206,    0.790614,  1.22157,    0.939199,  0.924118,  0.888734,
                                -0.559979, -0.120697, -0.0981791, 0.155517,  -0.286966,  -1.64074,  0.244946,  -1.0317,
                                -2.83982,  -3.05111,  -2.97484,   -2.72644,  0.891926,   -2.92111,  -2.91117,  -2.82863};

    NDims input_dims = {2, 3, 2, 3};
    NDims output_dims = {2, 3, 2, 4};
    NDims weight_dims = {3, 1, 2, 2};
    NDims bias_dims = {3, 1, 1, 1};
    Dim2 stride = {2, 2};
    int32_t group_size = 3;
    float error_threshold = 0.17;

    auto ifm =
        create_feature_map(feature_map_builder, std::string("Tensor_000[1]"), 0, input_dims, TFlite::TensorType_FLOAT32, true);
    auto ofm =
        create_feature_map(feature_map_builder, std::string("Tensor_000[2]"), 1, output_dims, TFlite::TensorType_FLOAT32, false);

    auto param_kernel = create_parameter(parameter_builder, const_cast<char *>("Kernel"), weight_data, weight_dims, TFlite::TensorType_FLOAT32);
    auto param_bias = create_parameter(parameter_builder, const_cast<char *>("Bias"), bias_data, bias_dims, TFlite::TensorType_FLOAT32);

    flatbuffers::FlatBufferBuilder build_data;
    TFlite::ActivationFunctionType activation = TFlite::ActivationFunctionType::ActivationFunctionType_NONE;
    TFlite::Padding padding = TFlite::Padding::Padding_SAME;

    auto tmp = TFlite::CreateTransposeConvOptions(build_data, padding, stride.w, stride.h, group_size, activation);
    build_data.Finish(tmp);
    void *base_addr =
        build_data.GetBufferPointer() +
        (flatbuffers::EndianScalar(*reinterpret_cast<flatbuffers::uoffset_t *>(build_data.GetBufferPointer())));

    enn::model::component::Operator::Ptr op =
        operator_builder.set_id(0)
            .set_name(TFlite::EnumNameBuiltinOperator(TFlite::BuiltinOperator_TRANSPOSE_CONV))
            .set_code(TFlite::BuiltinOperator_TRANSPOSE_CONV)
            .set_accelerator(model::Accelerator::GPU)
            .add_in_tensor(ifm)
            .add_in_tensor(param_kernel)
            .add_in_tensor(param_bias)
            .add_out_tensor(ofm)
            .set_option(TFlite::EnumNameBuiltinOptions(TFlite::BuiltinOptions_TransposeConvOptions),
                        base_addr,
                        build_data.GetSize(),
                        TFlite::BuiltinOptions_TransposeConvOptions)
            .create();

    // Make OperatorList
    auto attribute = std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_CAFFE, false);
    auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();
    gpu_ud.OpenSubGraph(*op_list);

    // Make BufferTable
    model::memory::BufferTable buffer_table;

    auto in_mem =
        emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * GetDimSize(input_dims), enn::EnnMmType::kEnnMmTypeIon);
    auto out_mem =
        emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * GetDimSize(output_dims), enn::EnnMmType::kEnnMmTypeIon);

    CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem, out_mem}), "CreateMemory() failed");

    in_mem->va = static_cast<void *>(input_data);
    buffer_table.add(0, in_mem->va, in_mem->size);
    buffer_table.add(1, out_mem->va, out_mem->size);

    // Prepare & Execute
    auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
        MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    EXPECT_EQ(gpu_ud.PrepareSubGraph(*executable_operator_list), ENN_RET_SUCCESS);

    auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list);
    EXPECT_EQ(gpu_ud.ExecuteSubGraph(operator_list_execute_request), ENN_RET_SUCCESS);
    compare((float *)out_mem->va, reference_output, GetDimSize(output_dims), error_threshold);

    emm->DeleteMemory(in_mem);
    emm->DeleteMemory(out_mem);
    gpu_ud.CloseSubGraph(*op_list);

    gpu_ud.Deinitialize();
}

TEST_F(ENN_GT_UNIT_TEST_GPU_UD, gpu_ud_test_depth2space) {
    ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();
    enn::model::component::OperatorBuilder operator_builder;

    NDims input_dims = {1, 8, 3, 3};
    NDims output_dims = {1, 2, 6, 6};
    int32_t block_size = 2;

    float input_data[] = {0,  4,  8,  24, 28, 32, 48, 52, 56, 1,  5,  9,  25, 29, 33, 49, 53, 57, 2,  6,  10, 26, 30, 34,
                          50, 54, 58, 3,  7,  11, 27, 31, 35, 51, 55, 59, 12, 16, 20, 36, 40, 44, 60, 64, 68, 13, 17, 21,
                          37, 41, 45, 61, 65, 69, 14, 18, 22, 38, 42, 46, 62, 66, 70, 15, 19, 23, 39, 43, 47, 63, 67, 71};
    float expect_data[] = {0,  2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46,
                           48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 1,  3,  5,  7,  9,  11, 13, 15, 17, 19, 21, 23,
                           25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71};
    float error_threshold = 0.17;

    auto ifm =
        create_feature_map(feature_map_builder, std::string("Tensor_000[1]"), 0, input_dims, TFlite::TensorType_FLOAT32, true);
    auto ofm =
        create_feature_map(feature_map_builder, std::string("Tensor_000[2]"), 1, output_dims, TFlite::TensorType_FLOAT32, false);

    flatbuffers::FlatBufferBuilder build_data;

    auto tmp = TFlite::CreateDepthToSpaceOptions(build_data, block_size);
    build_data.Finish(tmp);
    void *base_addr =
        build_data.GetBufferPointer() +
        (flatbuffers::EndianScalar(*reinterpret_cast<flatbuffers::uoffset_t *>(build_data.GetBufferPointer())));
    enn::model::component::Operator::Ptr op =
        operator_builder.set_id(0)
            .set_name(TFlite::EnumNameBuiltinOperator(TFlite::BuiltinOperator_DEPTH_TO_SPACE))
            .set_code(TFlite::BuiltinOperator_DEPTH_TO_SPACE)
            .set_accelerator(model::Accelerator::GPU)
            .add_in_tensor(ifm)
            .add_out_tensor(ofm)
            .set_option(TFlite::EnumNameBuiltinOptions(TFlite::BuiltinOptions_DepthToSpaceOptions),
                        base_addr,
                        build_data.GetSize(),
                        TFlite::BuiltinOptions_DepthToSpaceOptions)
            .create();
    // Make OperatorList
    auto attribute = std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_TENSORFLOW, false);
    auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();
    gpu_ud.OpenSubGraph(*op_list);

    // Make BufferTable
    model::memory::BufferTable buffer_table;

    auto in_mem =
        emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * GetDimSize(input_dims), enn::EnnMmType::kEnnMmTypeIon);
    auto out_mem =
        emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * GetDimSize(output_dims), enn::EnnMmType::kEnnMmTypeIon);

    CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem, out_mem}), "CreateMemory() failed");

    in_mem->va = static_cast<void *>(input_data);
    buffer_table.add(0, in_mem->va, in_mem->size);
    buffer_table.add(1, out_mem->va, out_mem->size);

    // Prepare & Execute
    auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
        MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    EXPECT_EQ(gpu_ud.PrepareSubGraph(*executable_operator_list), ENN_RET_SUCCESS);

    auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list);
    EXPECT_EQ(gpu_ud.ExecuteSubGraph(operator_list_execute_request), ENN_RET_SUCCESS);
    compare((float *)out_mem->va, expect_data, GetDimSize(output_dims), error_threshold);

    emm->DeleteMemory(in_mem);
    emm->DeleteMemory(out_mem);
    gpu_ud.CloseSubGraph(*op_list);

    gpu_ud.Deinitialize();
}

TEST_F(ENN_GT_UNIT_TEST_GPU_UD, gpu_ud_test_depthwise_convlution) {
    ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();
    enn::model::component::OperatorBuilder operator_builder;

    float input_data[] = {1, 7, 3, 9, 5, 11, 2, 8, 4, 10, 6, 12};
    float weight_data[] = {
        1,
        -9,
        5,
        13,
        2,
        10,
        6,
        -14,
        3,
        -11,
        7,
        15,
        4,
        12,
        8,
        -16,
    };
    float bias_data[] = {1, 2, 3, 4};
    float reference_output[] = {
        71,
        91,
        -34,
        -26,
        99,
        127,
        -20,
        -4,
    };

    NDims input_dims = {1, 2, 3, 2};
    NDims output_dims = {1, 4, 2, 1};
    NDims weight_dims = {4, 1, 2, 2};
    NDims bias_dims = {4, 1, 1, 1};
    float error_threshold = 0.17;
    Dim2 stride = {1, 1};
    Dim2 dilation = {1, 1};
    uint32_t depth_multiplier = 2;

    auto ifm =
        create_feature_map(feature_map_builder, std::string("Tensor_000[1]"), 0, input_dims, TFlite::TensorType_FLOAT32, true);
    auto ofm =
        create_feature_map(feature_map_builder, std::string("Tensor_000[2]"), 1, output_dims, TFlite::TensorType_FLOAT32, false);

    auto param_kernel = create_parameter(parameter_builder, const_cast<char *>("Kernel"), weight_data, weight_dims, TFlite::TensorType_FLOAT32);
    auto param_bias = create_parameter(parameter_builder, const_cast<char *>("Bias"), bias_data, bias_dims, TFlite::TensorType_FLOAT32);

    flatbuffers::FlatBufferBuilder build_data;
    TFlite::ActivationFunctionType activation = TFlite::ActivationFunctionType::ActivationFunctionType_NONE;
    TFlite::Padding padding = TFlite::Padding::Padding_VALID;

    auto tmp = TFlite::CreateDepthwiseConv2DOptions(
        build_data, padding, stride.w, stride.h, depth_multiplier, activation, dilation.h, dilation.w);
    build_data.Finish(tmp);
    void *base_addr =
        build_data.GetBufferPointer() +
        (flatbuffers::EndianScalar(*reinterpret_cast<flatbuffers::uoffset_t *>(build_data.GetBufferPointer())));
    enn::model::component::Operator::Ptr op =
        operator_builder.set_id(0)
            .set_name(TFlite::EnumNameBuiltinOperator(TFlite::BuiltinOperator_DEPTHWISE_CONV_2D))
            .set_code(TFlite::BuiltinOperator_DEPTHWISE_CONV_2D)
            .set_accelerator(model::Accelerator::GPU)
            .add_in_tensor(ifm)
            .add_in_tensor(param_kernel)
            .add_in_tensor(param_bias)
            .add_out_tensor(ofm)
            .set_option(TFlite::EnumNameBuiltinOptions(TFlite::BuiltinOptions_DepthwiseConv2DOptions),
                        base_addr,
                        build_data.GetSize(),
                        TFlite::BuiltinOptions_DepthwiseConv2DOptions)
            .create();

    // Make OperatorList
    auto attribute = std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_TENSORFLOW, false);
    auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();
    gpu_ud.OpenSubGraph(*op_list);

    // Make BufferTable
    model::memory::BufferTable buffer_table;

    auto in_mem =
        emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * GetDimSize(input_dims), enn::EnnMmType::kEnnMmTypeIon);
    auto out_mem =
        emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * GetDimSize(output_dims), enn::EnnMmType::kEnnMmTypeIon);

    CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem, out_mem}), "CreateMemory() failed");

    in_mem->va = static_cast<void *>(input_data);
    buffer_table.add(0, in_mem->va, in_mem->size);
    buffer_table.add(1, out_mem->va, out_mem->size);

    // Prepare & Execute
    auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
        MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    EXPECT_EQ(gpu_ud.PrepareSubGraph(*executable_operator_list), ENN_RET_SUCCESS);

    auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list);
    EXPECT_EQ(gpu_ud.ExecuteSubGraph(operator_list_execute_request), ENN_RET_SUCCESS);
    compare((float *)out_mem->va, reference_output, GetDimSize(output_dims), error_threshold);

    emm->DeleteMemory(in_mem);
    emm->DeleteMemory(out_mem);
    gpu_ud.CloseSubGraph(*op_list);

    gpu_ud.Deinitialize();
}

TEST_F(ENN_GT_UNIT_TEST_GPU_UD, gpu_ud_test_fully_connected) {
    ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();
    enn::model::component::OperatorBuilder operator_builder;

    NDims input_dims = {2, 10, 1, 1};
    NDims output_dims = {2, 3, 1, 1};
    NDims weight_dims = {3, 10, 1, 1};
    NDims bias_dims = {3, 1, 1, 1};

    // Make parameter
    float input_data[] = {
        1, 2, 3, 4, 5, 6, 7, 8, -9, -10, 1, 2, 3, 4, 5, 6, 7, -8, 9, -10,
    };
    float weight_data[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    };
    float bias_data[] = {1, 2, 3};
    float expect_data[] = {24, 25, 26, 58, 59, 60};
    float error_threshold = 0.17;
    // Dim2 stride = {1, 1};
    // Dim2 dilation = {1, 1};

    auto ifm =
        create_feature_map(feature_map_builder, std::string("Tensor_000[1]"), 0, input_dims, TFlite::TensorType_FLOAT32, true);
    auto ofm =
        create_feature_map(feature_map_builder, std::string("Tensor_000[2]"), 1, output_dims, TFlite::TensorType_FLOAT32, false);

    auto param_kernel = create_parameter(parameter_builder, const_cast<char *>("Kernel"), weight_data, weight_dims, TFlite::TensorType_FLOAT32);
    auto param_bias = create_parameter(parameter_builder, const_cast<char *>("Bias"), bias_data, bias_dims, TFlite::TensorType_FLOAT32);

    flatbuffers::FlatBufferBuilder build_data;
    TFlite::ActivationFunctionType activation = TFlite::ActivationFunctionType::ActivationFunctionType_RELU;
    //TFlite::Padding padding = TFlite::Padding::Padding_SAME;

    auto tmp = TFlite::CreateFullyConnectedOptions(build_data, activation);
    build_data.Finish(tmp);
    void *base_addr =
        build_data.GetBufferPointer() +
        (flatbuffers::EndianScalar(*reinterpret_cast<flatbuffers::uoffset_t *>(build_data.GetBufferPointer())));
    enn::model::component::Operator::Ptr op =
        operator_builder.set_id(0)
            .set_name(TFlite::EnumNameBuiltinOperator(TFlite::BuiltinOperator_FULLY_CONNECTED))
            .set_code(TFlite::BuiltinOperator_FULLY_CONNECTED)
            .set_accelerator(model::Accelerator::GPU)
            .add_in_tensor(ifm)
            .add_in_tensor(param_kernel)
            .add_in_tensor(param_bias)
            .add_out_tensor(ofm)
            .set_option(TFlite::EnumNameBuiltinOptions(TFlite::BuiltinOptions_FullyConnectedOptions),
                        base_addr,
                        build_data.GetSize(),
                        TFlite::BuiltinOptions_FullyConnectedOptions)
            .create();
    // Make OperatorList
    auto attribute = std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_TENSORFLOW, false);
    auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();
    gpu_ud.OpenSubGraph(*op_list);

    // Make BufferTable
    model::memory::BufferTable buffer_table;

    auto in_mem =
        emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * GetDimSize(input_dims), enn::EnnMmType::kEnnMmTypeIon);
    auto out_mem =
        emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * GetDimSize(output_dims), enn::EnnMmType::kEnnMmTypeIon);

    CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem, out_mem}), "CreateMemory() failed");

    in_mem->va = static_cast<void *>(input_data);
    buffer_table.add(0, in_mem->va, in_mem->size);
    buffer_table.add(1, out_mem->va, out_mem->size);

    // Prepare & Execute
    auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
        MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    EXPECT_EQ(gpu_ud.PrepareSubGraph(*executable_operator_list), ENN_RET_SUCCESS);

    auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list);
    EXPECT_EQ(gpu_ud.ExecuteSubGraph(operator_list_execute_request), ENN_RET_SUCCESS);
    compare((float *)out_mem->va, expect_data, GetDimSize(output_dims), error_threshold);

    emm->DeleteMemory(in_mem);
    emm->DeleteMemory(out_mem);
    gpu_ud.CloseSubGraph(*op_list);

    gpu_ud.Deinitialize();
}

#ifndef SCHEMA_NNC_V1
TEST_F(ENN_GT_UNIT_TEST_GPU_UD, gpu_ud_test_split) {
    ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();
    enn::model::component::OperatorBuilder operator_builder;

    NDims input_dims = {6, 1, 1, 1};
    NDims output_dims = {2, 1, 1, 1};
    int32_t axis = 0;
    int32_t num_outputs = 3;

    // Make parameter
    float input_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    float output1[] = {1.0, 2.0};
    float output2[] = {3.0, 4.0};
    float output3[] = {5.0, 6.0};
    float error_threshold = 0.17;

    auto ifm =
        create_feature_map(feature_map_builder, std::string("Tensor_000[1]"), 0, input_dims, TFlite::TensorType_FLOAT32, true);
    auto ofm_0 =
        create_feature_map(feature_map_builder, std::string("Tensor_000[2]"), 1, output_dims, TFlite::TensorType_FLOAT32, false);
    auto ofm_1 =
        create_feature_map(feature_map_builder, std::string("Tensor_000[2]"), 2, output_dims, TFlite::TensorType_FLOAT32, false);
    auto ofm_2 =
        create_feature_map(feature_map_builder, std::string("Tensor_000[2]"), 3, output_dims, TFlite::TensorType_FLOAT32, false);

    flatbuffers::FlatBufferBuilder build_data;
    auto tmp = TFlite::CreateSplitOptions(build_data, num_outputs, axis);
    build_data.Finish(tmp);
    void *base_addr =
        build_data.GetBufferPointer() +
        (flatbuffers::EndianScalar(*reinterpret_cast<flatbuffers::uoffset_t *>(build_data.GetBufferPointer())));
    enn::model::component::Operator::Ptr op =
        operator_builder.set_id(0)
            .set_name(TFlite::EnumNameBuiltinOperator(TFlite::BuiltinOperator_SPLIT))
            .set_code(TFlite::BuiltinOperator_SPLIT)
            .set_accelerator(model::Accelerator::GPU)
            .add_in_tensor(ifm)
            .add_out_tensor(ofm_0)
            .add_out_tensor(ofm_1)
            .add_out_tensor(ofm_2)
            .set_option(TFlite::EnumNameBuiltinOptions(TFlite::BuiltinOptions_SplitOptions),
                        base_addr,
                        build_data.GetSize(),
                        TFlite::BuiltinOptions_SplitOptions)
            .create();
    // Make OperatorList
    auto attribute = std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_TENSORFLOW, false);
    auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();
    gpu_ud.OpenSubGraph(*op_list);

    // Make BufferTable
    model::memory::BufferTable buffer_table;

    auto in_mem =
        emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * GetDimSize(input_dims), enn::EnnMmType::kEnnMmTypeIon);

    auto out_mem_0 =
        emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * GetDimSize(output_dims), enn::EnnMmType::kEnnMmTypeIon);
    auto out_mem_1 =
        emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * GetDimSize(output_dims), enn::EnnMmType::kEnnMmTypeIon);
    auto out_mem_2 =
        emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * GetDimSize(output_dims), enn::EnnMmType::kEnnMmTypeIon);

    CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem, out_mem_0, out_mem_1, out_mem_2}),
                          "CreateMemory() failed");

    in_mem->va = static_cast<void *>(input_data);
    buffer_table.add(0, in_mem->va, in_mem->size);
    buffer_table.add(1, out_mem_0->va, out_mem_0->size);
    buffer_table.add(2, out_mem_1->va, out_mem_1->size);
    buffer_table.add(3, out_mem_2->va, out_mem_2->size);

    // Prepare & Execute
    auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
        MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    EXPECT_EQ(gpu_ud.PrepareSubGraph(*executable_operator_list), ENN_RET_SUCCESS);

    auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list);
    EXPECT_EQ(gpu_ud.ExecuteSubGraph(operator_list_execute_request), ENN_RET_SUCCESS);
    compare((float *)out_mem_0->va, output1, GetDimSize(output_dims), error_threshold);
    compare((float *)out_mem_1->va, output2, GetDimSize(output_dims), error_threshold);
    compare((float *)out_mem_2->va, output3, GetDimSize(output_dims), error_threshold);

    emm->DeleteMemory(in_mem);
    emm->DeleteMemory(out_mem_0);
    emm->DeleteMemory(out_mem_1);
    emm->DeleteMemory(out_mem_2);
    gpu_ud.CloseSubGraph(*op_list);

    gpu_ud.Deinitialize();
}
#endif

TEST_F(ENN_GT_UNIT_TEST_GPU_UD, gpu_ud_test_reduce_min) {
    ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();
    enn::model::component::OperatorBuilder operator_builder;

    NDims input_dims = {1, 2, 3, 4};
    NDims output_dims = {1, 2, 3, 1};
    float error_threshold = 0.01;

    float input_data[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                          13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    float expect_data[] = {1, 5, 9, 13, 17, 21};
    std::vector<int32_t> axis = {3};

    auto ifm =
        create_feature_map(feature_map_builder, std::string("Tensor_000[1]"), 0, input_dims, TFlite::TensorType_FLOAT32, true);
    auto ofm =
        create_feature_map(feature_map_builder, std::string("Tensor_000[2]"), 1, output_dims, TFlite::TensorType_FLOAT32, false);

    NDims axis_dims = {(uint32_t)axis.size(), 1, 1, 1};
    auto param_axis = create_parameter(parameter_builder, const_cast<char *>("Axis"), axis.data(), axis_dims, TFlite::TensorType_INT32);

    flatbuffers::FlatBufferBuilder build_data;
    auto tmp = TFlite::CreateReducerOptions(build_data, false);
    build_data.Finish(tmp);
    void *base_addr =
        build_data.GetBufferPointer() +
        (flatbuffers::EndianScalar(*reinterpret_cast<flatbuffers::uoffset_t *>(build_data.GetBufferPointer())));

    std::shared_ptr<enn::model::component::Operator> op =
        operator_builder.set_id(0)
            .set_name(TFlite::EnumNameBuiltinOperator(TFlite::BuiltinOperator_REDUCE_MIN))
            .set_code(TFlite::BuiltinOperator_REDUCE_MIN)
            .set_accelerator(model::Accelerator::GPU)
            .add_in_tensor(ifm)
            .add_in_tensor(param_axis)
            .add_out_tensor(ofm)
            .set_option(TFlite::EnumNameBuiltinOptions(TFlite::BuiltinOptions_ReducerOptions),
                        base_addr,
                        build_data.GetSize(),
                        TFlite::BuiltinOptions_ReducerOptions)
            .create();
    // Make OperatorList
    auto attribute = std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_TENSORFLOW, false);
    auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();
    gpu_ud.OpenSubGraph(*op_list);

    // Make BufferTable
    model::memory::BufferTable buffer_table;

    auto in_mem =
        emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * GetDimSize(input_dims), enn::EnnMmType::kEnnMmTypeIon);
    auto out_mem =
        emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * GetDimSize(output_dims), enn::EnnMmType::kEnnMmTypeIon);

    CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem, out_mem}), "CreateMemory() failed");

    in_mem->va = static_cast<void *>(input_data);
    buffer_table.add(0, in_mem->va, in_mem->size);
    buffer_table.add(1, out_mem->va, out_mem->size);

    // Prepare & Execute
    auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
        MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    EXPECT_EQ(gpu_ud.PrepareSubGraph(*executable_operator_list), ENN_RET_SUCCESS);

    auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list);
    EXPECT_EQ(gpu_ud.ExecuteSubGraph(operator_list_execute_request), ENN_RET_SUCCESS);
    compare((float *)out_mem->va, expect_data, GetDimSize(output_dims), error_threshold);

    emm->DeleteMemory(in_mem);
    emm->DeleteMemory(out_mem);
    gpu_ud.CloseSubGraph(*op_list);

    gpu_ud.Deinitialize();
}

TEST_F(ENN_GT_UNIT_TEST_GPU_UD, gpu_ud_test_bidirectional_sequence_lstm) {
    ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();
    enn::model::component::OperatorBuilder operator_builder;

    uint32_t n_batch = 1;
    uint32_t n_input = 2;
    uint32_t n_cell = 4;
    uint32_t n_output = 4;
    uint32_t sequence_length = 3;
    bool use_cifg = false;
    bool use_peephole = false;
    bool use_projection_weights = false;
    bool use_projection_bias = false;
    bool merge_outputs = true;
    float cell_clip = 0.0f;
    float proj_clip = 0.0f;
    float error_threshold = 0.17;
    ActivationInfo activation_info(ActivationInfo::ActivationType::TANH, true);
    std::vector<float> input_to_input_weights = {
        -0.45018822, -0.02338299, -0.0870589, -0.34550029, 0.04266912, -0.15680569, -0.34856534, 0.43890524};
    std::vector<float> input_to_forget_weights = {
        0.09701663, 0.20334584, -0.50592935, -0.31343272, -0.40032279, 0.44781327, 0.01387155, -0.35593212};
    std::vector<float> input_to_cell_weights = {
        -0.50013041, 0.1370284, 0.11810488, 0.2013163, -0.20583314, 0.44344562, 0.22077113, -0.29909778};
    std::vector<float> input_to_output_weights = {
        -0.25065863, -0.28290087, 0.04613829, 0.40525138, 0.44272184, 0.03897077, -0.1556896, 0.19487578};
    std::vector<float> input_gate_bias = {0., 0., 0., 0.};
    std::vector<float> forget_gate_bias = {1., 1., 1., 1.};
    std::vector<float> cell_gate_bias = {0., 0., 0., 0.};
    std::vector<float> output_gate_bias = {0., 0., 0., 0.};
    std::vector<float> recurrent_to_input_weights = {-0.0063535,
                                                     -0.2042388,
                                                     0.31454784,
                                                     -0.35746509,
                                                     0.28902304,
                                                     0.08183324,
                                                     -0.16555229,
                                                     0.02286911,
                                                     -0.13566875,
                                                     0.03034258,
                                                     0.48091322,
                                                     -0.12528998,
                                                     0.24077177,
                                                     -0.51332325,
                                                     -0.33502164,
                                                     0.10629296};
    std::vector<float> recurrent_to_forget_weights = {-0.48684245,
                                                      -0.06655136,
                                                      0.42224967,
                                                      0.2112639,
                                                      0.27654213,
                                                      0.20864892,
                                                      -0.07646349,
                                                      0.45877004,
                                                      0.00141793,
                                                      -0.14609534,
                                                      0.36447752,
                                                      0.09196436,
                                                      0.28053468,
                                                      0.01560611,
                                                      -0.20127171,
                                                      -0.01140004};
    std::vector<float> recurrent_to_cell_weights = {-0.3407414,
                                                    0.24443203,
                                                    -0.2078532,
                                                    0.26320225,
                                                    0.05695659,
                                                    -0.00123841,
                                                    -0.4744786,
                                                    -0.35869038,
                                                    -0.06418842,
                                                    -0.13502428,
                                                    -0.501764,
                                                    0.22830659,
                                                    -0.46367589,
                                                    0.26016325,
                                                    -0.03894562,
                                                    -0.16368064};
    std::vector<float> recurrent_to_output_weights = {0.43385774,
                                                      -0.17194885,
                                                      0.2718237,
                                                      0.09215671,
                                                      0.24107647,
                                                      -0.39835793,
                                                      0.18212086,
                                                      0.01301402,
                                                      0.48572797,
                                                      -0.50656658,
                                                      0.20047462,
                                                      -0.20607421,
                                                      -0.51818722,
                                                      -0.15390486,
                                                      0.0468148,
                                                      0.39922136};
    std::vector<std::vector<float>> input_data = {{2., 3., 3., 4., 1., 1.}};
    std::vector<std::vector<float>> fw_reference_output_data = {{-0.02973187,
                                                                 0.1229473,
                                                                 0.20885126,
                                                                 -0.15358765,
                                                                 -0.03716109,
                                                                 0.12507336,
                                                                 0.41193449,
                                                                 -0.20860538,
                                                                 -0.15053082,
                                                                 0.09120187,
                                                                 0.24278517,
                                                                 -0.12222792}};
    std::vector<std::vector<float>> bw_reference_output_data = {{-0.0806187,
                                                                 0.139077,
                                                                 0.400476,
                                                                 -0.197842,
                                                                 -0.0332076,
                                                                 0.123838,
                                                                 0.309777,
                                                                 -0.17621,
                                                                 -0.0490733,
                                                                 0.0739237,
                                                                 0.067706,
                                                                 -0.0208124}};
    std::vector<float> fw_cell_to_input_weights;
    std::vector<float> fw_cell_to_forget_weights;
    std::vector<float> fw_cell_to_output_weights;
    std::vector<float> fw_projection_weights;
    std::vector<float> fw_projection_bias;
    std::vector<float> bw_cell_to_input_weights;
    std::vector<float> bw_cell_to_forget_weights;
    std::vector<float> bw_cell_to_output_weights;
    std::vector<float> bw_projection_weights;
    std::vector<float> bw_projection_bias;

    // set input dim
    NDims input_dim_ = {sequence_length, n_batch, n_input, 1};
    NDims aux_input_dim_ = {sequence_length, n_batch, 0, 1};
    NDims fw_output_dim_;
    NDims bw_output_dim_;
    if (merge_outputs) {
        fw_output_dim_ = {sequence_length, n_batch, 2 * n_output, 1};
        bw_output_dim_ = {sequence_length, n_batch, 0, 1};
    } else {
        fw_output_dim_ = {sequence_length, n_batch, n_output, 1};
        bw_output_dim_ = {sequence_length, n_batch, n_output, 1};
    }
    NDims fw_input_to_input_weights_dim_ = {n_cell, n_input, 1, 1};
    NDims fw_input_to_forget_weights_dim_ = {n_cell, n_input, 1, 1};
    NDims fw_input_to_cell_weights_dim_ = {n_cell, n_input, 1, 1};
    NDims fw_input_to_output_weights_dim_ = {n_cell, n_input, 1, 1};
    NDims fw_recurrent_to_input_weights_dim_ = {n_cell, n_output, 1, 1};
    NDims fw_recurrent_to_forget_weights_dim_ = {n_cell, n_output, 1, 1};
    NDims fw_recurrent_to_cell_weights_dim_ = {n_cell, n_output, 1, 1};
    NDims fw_recurrent_to_output_weights_dim_ = {n_cell, n_output, 1, 1};
    NDims fw_cell_to_input_weights_dim_ = {n_cell, 1, 1, 1};
    NDims fw_cell_to_forget_weights_dim_ = {n_cell, 1, 1, 1};
    NDims fw_cell_to_output_weights_dim_ = {n_cell, 1, 1, 1};
    NDims fw_input_gate_bias_dim_ = {n_cell, 1, 1, 1};
    NDims fw_forget_gate_bias_dim_ = {n_cell, 1, 1, 1};
    NDims fw_cell_gate_bias_dim_ = {n_cell, 1, 1, 1};
    NDims fw_output_gate_bias_dim_ = {n_cell, 1, 1, 1};
    NDims fw_projection_weights_dim_ = {n_output, n_cell, 1, 1};
    NDims fw_projection_bias_dim_ = {n_output, 1, 1, 1};
    NDims fw_input_activation_state_dim_ = {n_batch, n_output, 1, 1};
    NDims fw_input_cell_state_dim_ = {n_batch, n_cell, 1, 1};

    NDims bw_input_to_input_weights_dim_ = {n_cell, n_input, 1, 1};
    NDims bw_input_to_forget_weights_dim_ = {n_cell, n_input, 1, 1};
    NDims bw_input_to_cell_weights_dim_ = {n_cell, n_input, 1, 1};
    NDims bw_input_to_output_weights_dim_ = {n_cell, n_input, 1, 1};
    NDims bw_recurrent_to_input_weights_dim_ = {n_cell, n_output, 1, 1};
    NDims bw_recurrent_to_forget_weights_dim_ = {n_cell, n_output, 1, 1};
    NDims bw_recurrent_to_cell_weights_dim_ = {n_cell, n_output, 1, 1};
    NDims bw_recurrent_to_output_weights_dim_ = {n_cell, n_output, 1, 1};
    NDims bw_cell_to_input_weights_dim_ = {n_cell, 1, 1, 1};
    NDims bw_cell_to_forget_weights_dim_ = {n_cell, 1, 1, 1};
    NDims bw_cell_to_output_weights_dim_ = {n_cell, 1, 1, 1};
    NDims bw_input_gate_bias_dim_ = {n_cell, 1, 1, 1};
    NDims bw_forget_gate_bias_dim_ = {n_cell, 1, 1, 1};
    NDims bw_cell_gate_bias_dim_ = {n_cell, 1, 1, 1};
    NDims bw_output_gate_bias_dim_ = {n_cell, 1, 1, 1};
    NDims bw_projection_weights_dim_ = {n_output, n_cell, 1, 1};
    NDims bw_projection_bias_dim_ = {n_output, 1, 1, 1};
    NDims bw_input_activation_state_dim_ = {n_batch, n_output, 1, 1};
    NDims bw_input_cell_state_dim_ = {n_batch, n_cell, 1, 1};

    // Make IFM
    enn::model::component::FeatureMapBuilder feature_map_builder;

    auto ifm =
        create_feature_map(feature_map_builder, std::string("Tensor_000[1]"), 0, input_dim_, TFlite::TensorType_FLOAT32, true);
    auto fw_output_ofm =
        create_feature_map(feature_map_builder, std::string("Tensor_000[2]"), 1, fw_output_dim_, TFlite::TensorType_FLOAT32, false);

    enn::model::component::ParameterBuilder parameter_builder;
    std::shared_ptr<enn::model::component::Parameter> fw_input_to_input_weights_parameter;

    NDims zero_dim = {0};
    // fw
    if (use_cifg) {
        fw_input_to_input_weights_parameter = parameter_builder.set_id(0)
                                                  .set_name("")
                                                  .set_buffer_addr(nullptr)
                                                  .set_buffer_size(0)
                                                  .set_data_type(TFlite::TensorType_FLOAT32)
                                                  .set_shape(zero_dim)
                                                  .create();
    } else {
        fw_input_to_input_weights_parameter = create_parameter(parameter_builder,
                                                               const_cast<char *>(""),
                                                               input_to_input_weights.data(),
                                                               fw_input_to_input_weights_dim_,
                                                               TFlite::TensorType_FLOAT32);
    }

    auto fw_input_to_forget_weights_parameter = create_parameter(
        parameter_builder, const_cast<char *>(""), input_to_forget_weights.data(), fw_input_to_forget_weights_dim_, TFlite::TensorType_FLOAT32);
    auto fw_input_to_cell_weights_parameter = create_parameter(
        parameter_builder, const_cast<char *>(""), input_to_cell_weights.data(), fw_input_to_cell_weights_dim_, TFlite::TensorType_FLOAT32);
    auto fw_input_to_output_weights_parameter = create_parameter(
        parameter_builder, const_cast<char *>(""), input_to_output_weights.data(), fw_input_to_output_weights_dim_, TFlite::TensorType_FLOAT32);

    std::shared_ptr<enn::model::component::Parameter> fw_recurrent_to_input_weights_parameter;
    if (use_cifg) {
        fw_recurrent_to_input_weights_parameter = parameter_builder.set_id(0)
                                                      .set_name("")
                                                      .set_buffer_addr(nullptr)
                                                      .set_buffer_size(0)
                                                      .set_data_type(TFlite::TensorType_FLOAT32)
                                                      .set_shape(zero_dim)
                                                      .create();
    } else {
        fw_recurrent_to_input_weights_parameter = create_parameter(parameter_builder,
                                                                   const_cast<char *>(""),
                                                                   recurrent_to_input_weights.data(),
                                                                   fw_recurrent_to_input_weights_dim_,
                                                                   TFlite::TensorType_FLOAT32);
    }

    auto fw_recurrent_to_forget_weights_parameter = create_parameter(parameter_builder,
                                                                     const_cast<char *>(""),
                                                                     recurrent_to_forget_weights.data(),
                                                                     fw_recurrent_to_forget_weights_dim_,
                                                                     TFlite::TensorType_FLOAT32);
    auto fw_recurrent_to_cell_weights_parameter = create_parameter(parameter_builder,
                                                                   const_cast<char *>(""),
                                                                   recurrent_to_cell_weights.data(),
                                                                   fw_recurrent_to_cell_weights_dim_,
                                                                   TFlite::TensorType_FLOAT32);
    auto fw_recurrent_to_output_weights_parameter = create_parameter(parameter_builder,
                                                                     const_cast<char *>(""),
                                                                     recurrent_to_output_weights.data(),
                                                                     fw_recurrent_to_output_weights_dim_,
                                                                     TFlite::TensorType_FLOAT32);

    std::shared_ptr<enn::model::component::Parameter> fw_cell_to_input_weights_parameter;
    std::shared_ptr<enn::model::component::Parameter> fw_cell_to_forget_weights_parameter;
    std::shared_ptr<enn::model::component::Parameter> fw_cell_to_output_weights_parameter;
    if (use_peephole) {
        if (use_cifg) {
            fw_cell_to_input_weights_parameter = parameter_builder.set_id(0)
                                                     .set_name("")
                                                     .set_buffer_addr(nullptr)
                                                     .set_buffer_size(0)
                                                     .set_data_type(TFlite::TensorType_FLOAT32)
                                                     .set_shape(zero_dim)
                                                     .create();
        } else {
            fw_cell_to_input_weights_parameter = create_parameter(parameter_builder,
                                                                  const_cast<char *>(""),
                                                                  fw_cell_to_input_weights.data(),
                                                                  fw_cell_to_input_weights_dim_,
                                                                  TFlite::TensorType_FLOAT32);
        }
        fw_cell_to_forget_weights_parameter = create_parameter(parameter_builder,
                                                               const_cast<char *>(""),
                                                               fw_cell_to_forget_weights.data(),
                                                               fw_cell_to_forget_weights_dim_,
                                                               TFlite::TensorType_FLOAT32);
        fw_cell_to_output_weights_parameter = create_parameter(parameter_builder,
                                                               const_cast<char *>(""),
                                                               fw_cell_to_output_weights.data(),
                                                               fw_cell_to_output_weights_dim_,
                                                               TFlite::TensorType_FLOAT32);
    } else {
        fw_cell_to_input_weights_parameter = parameter_builder.set_id(0)
                                                 .set_name("")
                                                 .set_buffer_addr(nullptr)
                                                 .set_buffer_size(0)
                                                 .set_data_type(TFlite::TensorType_FLOAT32)
                                                 .set_shape(zero_dim)
                                                 .create();
        fw_cell_to_forget_weights_parameter = parameter_builder.set_id(0)
                                                  .set_name("")
                                                  .set_buffer_addr(nullptr)
                                                  .set_buffer_size(0)
                                                  .set_data_type(TFlite::TensorType_FLOAT32)
                                                  .set_shape(zero_dim)
                                                  .create();
        fw_cell_to_output_weights_parameter = parameter_builder.set_id(0)
                                                  .set_name("")
                                                  .set_buffer_addr(nullptr)
                                                  .set_buffer_size(0)
                                                  .set_data_type(TFlite::TensorType_FLOAT32)
                                                  .set_shape(zero_dim)
                                                  .create();
    }

    std::shared_ptr<enn::model::component::Parameter> fw_input_gate_bias_parameter;
    if (use_cifg) {
        fw_input_gate_bias_parameter = parameter_builder.set_id(0)
                                           .set_name("")
                                           .set_buffer_addr(nullptr)
                                           .set_buffer_size(0)
                                           .set_data_type(TFlite::TensorType_FLOAT32)
                                           .set_shape(zero_dim)
                                           .create();
    } else {
        fw_input_gate_bias_parameter = create_parameter(
            parameter_builder, const_cast<char *>(""), input_gate_bias.data(), fw_input_gate_bias_dim_, TFlite::TensorType_FLOAT32);
    }

    auto fw_forget_gate_bias_parameter = create_parameter(
        parameter_builder, const_cast<char *>(""), forget_gate_bias.data(), fw_forget_gate_bias_dim_, TFlite::TensorType_FLOAT32);
    auto fw_cell_gate_bias_parameter =
        create_parameter(parameter_builder, const_cast<char *>(""), cell_gate_bias.data(), fw_cell_gate_bias_dim_, TFlite::TensorType_FLOAT32);
    auto fw_output_gate_bias_dim_parameter = create_parameter(
        parameter_builder, const_cast<char *>(""), output_gate_bias.data(), fw_output_gate_bias_dim_, TFlite::TensorType_FLOAT32);

    std::shared_ptr<enn::model::component::Parameter> fw_projection_weights_parameter;
    std::shared_ptr<enn::model::component::Parameter> fw_projection_bias_parameter;
    if (use_projection_weights) {
        fw_projection_weights_parameter = create_parameter(
            parameter_builder, const_cast<char *>(""), fw_projection_weights.data(), fw_projection_weights_dim_, TFlite::TensorType_FLOAT32);

        if (use_projection_bias) {
            fw_projection_bias_parameter = create_parameter(
                parameter_builder, const_cast<char *>(""), fw_projection_bias.data(), fw_projection_bias_dim_, TFlite::TensorType_FLOAT32);

        } else {
            fw_projection_bias_parameter = parameter_builder.set_id(0)
                                               .set_name("")
                                               .set_buffer_addr(nullptr)
                                               .set_buffer_size(0)
                                               .set_data_type(TFlite::TensorType_FLOAT32)
                                               .set_shape(zero_dim)
                                               .create();
        }
    } else {
        fw_projection_weights_parameter = parameter_builder.set_id(0)
                                              .set_name("")
                                              .set_buffer_addr(nullptr)
                                              .set_buffer_size(0)
                                              .set_data_type(TFlite::TensorType_FLOAT32)
                                              .set_shape(zero_dim)
                                              .create();
        fw_projection_bias_parameter = parameter_builder.set_id(0)
                                           .set_name("")
                                           .set_buffer_addr(nullptr)
                                           .set_buffer_size(0)
                                           .set_data_type(TFlite::TensorType_FLOAT32)
                                           .set_shape(zero_dim)
                                           .create();
    }

    // bw
    std::shared_ptr<enn::model::component::Parameter> bw_input_to_input_weights_parameter;
    if (use_cifg) {
        bw_input_to_input_weights_parameter = parameter_builder.set_id(0)
                                                  .set_name("")
                                                  .set_buffer_addr(nullptr)
                                                  .set_buffer_size(0)
                                                  .set_data_type(TFlite::TensorType_FLOAT32)
                                                  .set_shape(zero_dim)
                                                  .create();

    } else {
        bw_input_to_input_weights_parameter = create_parameter(parameter_builder,
                                                               const_cast<char *>(""),
                                                               input_to_input_weights.data(),
                                                               bw_input_to_input_weights_dim_,
                                                               TFlite::TensorType_FLOAT32);
    }

    auto bw_input_to_forget_weights_parameter = create_parameter(
        parameter_builder, const_cast<char *>(""), input_to_forget_weights.data(), bw_input_to_forget_weights_dim_, TFlite::TensorType_FLOAT32);
    auto bw_input_to_cell_weights_parameter = create_parameter(
        parameter_builder, const_cast<char *>(""), input_to_cell_weights.data(), bw_input_to_cell_weights_dim_, TFlite::TensorType_FLOAT32);
    auto bw_input_to_output_weights_parameter = create_parameter(
        parameter_builder, const_cast<char *>(""), input_to_output_weights.data(), bw_input_to_output_weights_dim_, TFlite::TensorType_FLOAT32);

    std::shared_ptr<enn::model::component::Parameter> bw_recurrent_to_input_weights_parameter;
    if (use_cifg) {
        bw_recurrent_to_input_weights_parameter = parameter_builder.set_id(0)
                                                      .set_name("")
                                                      .set_buffer_addr(nullptr)
                                                      .set_buffer_size(0)
                                                      .set_data_type(TFlite::TensorType_FLOAT32)
                                                      .set_shape(zero_dim)
                                                      .create();
    } else {
        bw_recurrent_to_input_weights_parameter = create_parameter(parameter_builder,
                                                                   const_cast<char *>(""),
                                                                   recurrent_to_input_weights.data(),
                                                                   bw_recurrent_to_input_weights_dim_,
                                                                   TFlite::TensorType_FLOAT32);
    }
    auto bw_recurrent_to_forget_weights_parameter = create_parameter(parameter_builder,
                                                                     const_cast<char *>(""),
                                                                     recurrent_to_forget_weights.data(),
                                                                     bw_recurrent_to_forget_weights_dim_,
                                                                     TFlite::TensorType_FLOAT32);
    auto bw_recurrent_to_cell_weights_parameter = create_parameter(parameter_builder,
                                                                   const_cast<char *>(""),
                                                                   recurrent_to_cell_weights.data(),
                                                                   bw_recurrent_to_cell_weights_dim_,
                                                                   TFlite::TensorType_FLOAT32);
    auto bw_recurrent_to_output_weights_parameter = create_parameter(parameter_builder,
                                                                     const_cast<char *>(""),
                                                                     recurrent_to_output_weights.data(),
                                                                     bw_recurrent_to_output_weights_dim_,
                                                                     TFlite::TensorType_FLOAT32);

    std::shared_ptr<enn::model::component::Parameter> bw_cell_to_input_weights_parameter;
    std::shared_ptr<enn::model::component::Parameter> bw_cell_to_forget_weights_parameter;
    std::shared_ptr<enn::model::component::Parameter> bw_cell_to_output_weights_parameter;
    if (use_peephole) {
        if (use_cifg) {
            bw_cell_to_input_weights_parameter = parameter_builder.set_id(0)
                                                     .set_name("")
                                                     .set_buffer_addr(nullptr)
                                                     .set_buffer_size(0)
                                                     .set_data_type(TFlite::TensorType_FLOAT32)
                                                     .set_shape(zero_dim)
                                                     .create();
        } else {
            bw_cell_to_input_weights_parameter = create_parameter(parameter_builder,
                                                                  const_cast<char *>(""),
                                                                  bw_cell_to_input_weights.data(),
                                                                  bw_cell_to_input_weights_dim_,
                                                                  TFlite::TensorType_FLOAT32);
        }
        bw_cell_to_forget_weights_parameter = create_parameter(parameter_builder,
                                                               const_cast<char *>(""),
                                                               bw_cell_to_forget_weights.data(),
                                                               bw_cell_to_forget_weights_dim_,
                                                               TFlite::TensorType_FLOAT32);
        bw_cell_to_output_weights_parameter = create_parameter(parameter_builder,
                                                               const_cast<char *>(""),
                                                               bw_cell_to_output_weights.data(),
                                                               bw_cell_to_output_weights_dim_,
                                                               TFlite::TensorType_FLOAT32);
    } else {
        bw_cell_to_input_weights_parameter = parameter_builder.set_id(0)
                                                 .set_name("")
                                                 .set_buffer_addr(nullptr)
                                                 .set_buffer_size(0)
                                                 .set_data_type(TFlite::TensorType_FLOAT32)
                                                 .set_shape(zero_dim)
                                                 .create();
        bw_cell_to_forget_weights_parameter = parameter_builder.set_id(0)
                                                  .set_name("")
                                                  .set_buffer_addr(nullptr)
                                                  .set_buffer_size(0)
                                                  .set_data_type(TFlite::TensorType_FLOAT32)
                                                  .set_shape(zero_dim)
                                                  .create();
        bw_cell_to_output_weights_parameter = parameter_builder.set_id(0)
                                                  .set_name("")
                                                  .set_buffer_addr(nullptr)
                                                  .set_buffer_size(0)
                                                  .set_data_type(TFlite::TensorType_FLOAT32)
                                                  .set_shape(zero_dim)
                                                  .create();
    }

    std::shared_ptr<enn::model::component::Parameter> bw_input_gate_bias_parameter;
    if (use_cifg) {
        bw_input_gate_bias_parameter = parameter_builder.set_id(0)
                                           .set_name("")
                                           .set_buffer_addr(nullptr)
                                           .set_buffer_size(0)
                                           .set_data_type(TFlite::TensorType_FLOAT32)
                                           .set_shape(zero_dim)
                                           .create();
    } else {
        bw_input_gate_bias_parameter = create_parameter(
            parameter_builder, const_cast<char *>(""), input_gate_bias.data(), bw_input_gate_bias_dim_, TFlite::TensorType_FLOAT32);
    }

    auto bw_forget_gate_bias_parameter = create_parameter(
        parameter_builder, const_cast<char *>(""), forget_gate_bias.data(), bw_forget_gate_bias_dim_, TFlite::TensorType_FLOAT32);
    auto bw_cell_gate_bias_parameter =
        create_parameter(parameter_builder, const_cast<char *>(""), cell_gate_bias.data(), bw_cell_gate_bias_dim_, TFlite::TensorType_FLOAT32);
    auto bw_output_gate_bias_parameter = create_parameter(
        parameter_builder, const_cast<char *>(""), output_gate_bias.data(), bw_output_gate_bias_dim_, TFlite::TensorType_FLOAT32);

    std::shared_ptr<enn::model::component::Parameter> bw_projection_weights_parameter;
    std::shared_ptr<enn::model::component::Parameter> bw_projection_bias_parameter;
    if (use_projection_weights) {
        bw_projection_weights_parameter = create_parameter(
            parameter_builder, const_cast<char *>(""), bw_projection_weights.data(), bw_projection_weights_dim_, TFlite::TensorType_FLOAT32);

        if (use_projection_bias) {
            bw_projection_bias_parameter = create_parameter(
                parameter_builder, const_cast<char *>(""), bw_projection_bias.data(), bw_projection_bias_dim_, TFlite::TensorType_FLOAT32);
        } else {
            bw_projection_bias_parameter = parameter_builder.set_id(0)
                                               .set_name("")
                                               .set_buffer_addr(nullptr)
                                               .set_buffer_size(0)
                                               .set_data_type(TFlite::TensorType_FLOAT32)
                                               .set_shape(zero_dim)
                                               .create();
        }
    } else {
        bw_projection_weights_parameter = parameter_builder.set_id(0)
                                              .set_name("")
                                              .set_buffer_addr(nullptr)
                                              .set_buffer_size(0)
                                              .set_data_type(TFlite::TensorType_FLOAT32)
                                              .set_shape(zero_dim)
                                              .create();
        bw_projection_bias_parameter = parameter_builder.set_id(0)
                                           .set_name("")
                                           .set_buffer_addr(nullptr)
                                           .set_buffer_size(0)
                                           .set_data_type(TFlite::TensorType_FLOAT32)
                                           .set_shape(zero_dim)
                                           .create();
    }

    std::shared_ptr<float> fw_input_activation_state;
    std::shared_ptr<float> fw_input_cell_state;
    fw_input_activation_state.reset(new float[GetDimSize(fw_input_activation_state_dim_)](), std::default_delete<float[]>());
    fw_input_cell_state.reset(new float[GetDimSize(fw_input_cell_state_dim_)](), std::default_delete<float[]>());

    auto fw_input_activation_state_parameter = create_parameter(
        parameter_builder, const_cast<char *>(""), fw_input_activation_state.get(), fw_input_activation_state_dim_, TFlite::TensorType_FLOAT32);
    auto fw_input_cell_state_parameter = create_parameter(
        parameter_builder, const_cast<char *>(""), fw_input_cell_state.get(), fw_input_cell_state_dim_, TFlite::TensorType_FLOAT32);

    std::shared_ptr<float> bw_input_activation_state;
    std::shared_ptr<float> bw_input_cell_state;
    bw_input_activation_state.reset(new float[GetDimSize(bw_input_activation_state_dim_)](), std::default_delete<float[]>());
    bw_input_cell_state.reset(new float[GetDimSize(bw_input_cell_state_dim_)](), std::default_delete<float[]>());

    auto bw_input_activation_state_parameter = create_parameter(
        parameter_builder, const_cast<char *>(""), bw_input_activation_state.get(), bw_input_activation_state_dim_, TFlite::TensorType_FLOAT32);
    auto bw_input_cell_state_parameter = create_parameter(
        parameter_builder, const_cast<char *>(""), bw_input_cell_state.get(), bw_input_cell_state_dim_, TFlite::TensorType_FLOAT32);

    auto optional_tensor_parameter = parameter_builder.set_id(0)
                                         .set_name("")
                                         .set_buffer_addr(nullptr)
                                         .set_buffer_size(0)
                                         .set_data_type(TFlite::TensorType_FLOAT32)
                                         .set_shape(zero_dim)
                                         .create();

    flatbuffers::FlatBufferBuilder build_data;
    TFlite::ActivationFunctionType activation = TFlite::ActivationFunctionType::ActivationFunctionType_TANH;

    auto tmp = TFlite::CreateBidirectionalSequenceLSTMOptions(build_data, activation, cell_clip, proj_clip, merge_outputs);
    build_data.Finish(tmp);
    auto tflOptions = (flatbuffers::GetRoot<TFlite::BidirectionalSequenceLSTMOptions>(build_data.GetBufferPointer()));
    void *base_addr =
        build_data.GetBufferPointer() +
        (flatbuffers::EndianScalar(*reinterpret_cast<flatbuffers::uoffset_t *>(build_data.GetBufferPointer())));
    enn::model::component::Operator::Ptr op =
        operator_builder.set_id(0)
            .set_name(TFlite::EnumNameBuiltinOperator(TFlite::BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM))
            .set_code(TFlite::BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM)
            .set_accelerator(model::Accelerator::GPU)
            .add_in_tensor(ifm)
            .add_in_tensor(fw_input_to_input_weights_parameter)
            .add_in_tensor(fw_input_to_forget_weights_parameter)
            .add_in_tensor(fw_input_to_cell_weights_parameter)
            .add_in_tensor(fw_input_to_output_weights_parameter)
            .add_in_tensor(fw_recurrent_to_input_weights_parameter)
            .add_in_tensor(fw_recurrent_to_forget_weights_parameter)
            .add_in_tensor(fw_recurrent_to_cell_weights_parameter)
            .add_in_tensor(fw_recurrent_to_output_weights_parameter)
            .add_in_tensor(fw_cell_to_input_weights_parameter)
            .add_in_tensor(fw_cell_to_forget_weights_parameter)
            .add_in_tensor(fw_cell_to_output_weights_parameter)
            .add_in_tensor(fw_input_gate_bias_parameter)
            .add_in_tensor(fw_forget_gate_bias_parameter)
            .add_in_tensor(fw_cell_gate_bias_parameter)
            .add_in_tensor(fw_output_gate_bias_dim_parameter)
            .add_in_tensor(fw_projection_weights_parameter)
            .add_in_tensor(fw_projection_bias_parameter)
            .add_in_tensor(bw_input_to_input_weights_parameter)
            .add_in_tensor(bw_input_to_forget_weights_parameter)
            .add_in_tensor(bw_input_to_cell_weights_parameter)
            .add_in_tensor(bw_input_to_output_weights_parameter)
            .add_in_tensor(bw_recurrent_to_input_weights_parameter)
            .add_in_tensor(bw_recurrent_to_forget_weights_parameter)
            .add_in_tensor(bw_recurrent_to_cell_weights_parameter)
            .add_in_tensor(bw_recurrent_to_output_weights_parameter)
            .add_in_tensor(bw_cell_to_input_weights_parameter)
            .add_in_tensor(bw_cell_to_forget_weights_parameter)
            .add_in_tensor(bw_cell_to_output_weights_parameter)
            .add_in_tensor(bw_input_gate_bias_parameter)
            .add_in_tensor(bw_forget_gate_bias_parameter)
            .add_in_tensor(bw_cell_gate_bias_parameter)
            .add_in_tensor(bw_output_gate_bias_parameter)
            .add_in_tensor(bw_projection_weights_parameter)
            .add_in_tensor(bw_projection_bias_parameter)
            .add_in_tensor(fw_input_activation_state_parameter)
            .add_in_tensor(fw_input_cell_state_parameter)
            .add_in_tensor(bw_input_activation_state_parameter)
            .add_in_tensor(bw_input_cell_state_parameter)
            .add_in_tensor(optional_tensor_parameter)
            .add_in_tensor(optional_tensor_parameter)
            .add_in_tensor(optional_tensor_parameter)
            .add_in_tensor(optional_tensor_parameter)
            .add_in_tensor(optional_tensor_parameter)
            .add_in_tensor(optional_tensor_parameter)
            .add_in_tensor(optional_tensor_parameter)
            .add_in_tensor(optional_tensor_parameter)
            .add_in_tensor(optional_tensor_parameter)
            .add_in_tensor(optional_tensor_parameter)
            .add_in_tensor(optional_tensor_parameter)
            .add_in_tensor(optional_tensor_parameter)
            .add_in_tensor(optional_tensor_parameter)
            .add_in_tensor(optional_tensor_parameter)
            .add_in_tensor(optional_tensor_parameter)
            .add_in_tensor(optional_tensor_parameter)
            .add_in_tensor(optional_tensor_parameter)
            .add_out_tensor(fw_output_ofm)
            .set_option(TFlite::EnumNameBuiltinOptions(TFlite::BuiltinOptions_BidirectionalSequenceLSTMOptions),
                        base_addr,
                        build_data.GetSize(),
                        TFlite::BuiltinOptions_BidirectionalSequenceLSTMOptions)
            .create();

    // Make OperatorList
    auto attribute = std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_TENSORFLOW, false);
    auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();
    gpu_ud.OpenSubGraph(*op_list);

    const int num_batchs = n_batch;
    const int num_inputs = n_input;
    const int fw_num_outputs = fw_output_dim_.at(2);
    const int bw_num_outputs = bw_output_dim_.at(2);

    std::vector<float> in;
    if (num_batchs == 2 && num_inputs == 5 && fw_num_outputs == 16 && bw_num_outputs == 16) {
        in.resize(GetDimSize(input_dim_));
        in.clear();
        for (int i = 0; i < sequence_length; i++) {
            memcpy(in.data() + 2 * i * num_inputs, input_data[0].data() + i * num_inputs, num_inputs * sizeof(float));
            memcpy(in.data() + (2 * i + 1) * num_inputs, input_data[1].data() + i * num_inputs, num_inputs * sizeof(float));
        }
    } else {
        // Set input
        in.resize(GetDimSize(input_dim_));
        in.clear();
        memcpy(in.data(), input_data[0].data(), num_inputs * num_batchs * sequence_length * sizeof(float));
    }
    auto in_mem =
        emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * GetDimSize(input_dim_), enn::EnnMmType::kEnnMmTypeIon);
    auto fw_out_mem =
        emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * GetDimSize(fw_output_dim_), enn::EnnMmType::kEnnMmTypeIon);

    CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem, fw_out_mem}), "CreateMemory() failed");

    // Make BufferTable
    model::memory::BufferTable buffer_table;

    in_mem->va = static_cast<void *>(in.data());

    buffer_table.add(0, in_mem->va, in_mem->size);
    buffer_table.add(1, fw_out_mem->va, fw_out_mem->size);

    // Prepare & Execute
    auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
        MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    EXPECT_EQ(gpu_ud.PrepareSubGraph(*executable_operator_list), ENN_RET_SUCCESS);

    auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list);

    EXPECT_EQ(gpu_ud.ExecuteSubGraph(operator_list_execute_request), ENN_RET_SUCCESS);

    std::vector<float> merged_expected;
    for (int k = 0; k < sequence_length * num_batchs; k++) {
        merged_expected.insert(merged_expected.end(),
                               fw_reference_output_data[0].data() + k * fw_num_outputs / 2,
                               fw_reference_output_data[0].data() + (k + 1) * fw_num_outputs / 2);
        merged_expected.insert(merged_expected.end(),
                               bw_reference_output_data[0].data() + k * fw_num_outputs / 2,
                               bw_reference_output_data[0].data() + (k + 1) * fw_num_outputs / 2);
    }
    compare((float *)fw_out_mem->va, merged_expected.data(), fw_num_outputs * num_batchs * sequence_length, error_threshold);

    emm->DeleteMemory(in_mem);
    emm->DeleteMemory(fw_out_mem);
    gpu_ud.CloseSubGraph(*op_list);

    gpu_ud.Deinitialize();
}

TEST_F(ENN_GT_UNIT_TEST_GPU_UD, gpu_ud_test_gather) {
    ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();

    // Define test case
    std::string op_name = "GATHER";
    const std::string option_name = "GatherOptions";
    TFlite::BuiltinOperator op_code = TFlite::BuiltinOperator::BuiltinOperator_GATHER;
    TFlite::BuiltinOptions op_option = TFlite::BuiltinOptions::BuiltinOptions_GatherOptions;
    NDims input_dims = {2, 2};
    NDims indices_dims = {2};
    NDims output_dims = {2, 2};
    int32_t axis = 0;
    float error_threshold = 0.17;
    std::unique_ptr<float[]> input(new float[GetDimSize(input_dims)]{-2.0, 0.2, 0.7, 0.8});
    std::unique_ptr<int32_t[]> indices_data(new int32_t[GetDimSize(indices_dims)]{1, 0});
    std::unique_ptr<float[]> expect_out(new float[GetDimSize(output_dims)]{0.7, 0.8, -2, 0.2});

    // Make IFM & OFM
    auto ifm =
        create_feature_map(feature_map_builder, std::string("Tensor_000[1]"), 0, input_dims, TFlite::TensorType_FLOAT32, true);
    auto ofm =
        create_feature_map(feature_map_builder, std::string("Tensor_000[2]"), 1, output_dims, TFlite::TensorType_FLOAT32, false);
    // Make parameter
    auto indices =
        create_parameter(parameter_builder, const_cast<char *>("Indices"), indices_data.get(), indices_dims, TFlite::TensorType_INT32);

    // Make option
    flatbuffers::FlatBufferBuilder build_data;
#ifdef SCHEMA_NNC_V1
    auto tmp = TFlite::CreateGatherOptions(build_data, axis);
#else
    auto tmp = TFlite::CreateGatherOptions(build_data, axis, 0);
#endif
    build_data.Finish(tmp);
    void *base_addr =
        build_data.GetBufferPointer() +
        (flatbuffers::EndianScalar(*reinterpret_cast<flatbuffers::uoffset_t *>(build_data.GetBufferPointer())));

    // Make operator
    auto op = create_operator(operator_builder,
                              op_name,
                              op_code,
                              {ifm},
                              {indices},
                              {ofm},
                              option_name,
                              base_addr,
                              build_data.GetSize(),
                              op_option);

    // Make OperatorList
    auto attribute = std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_CAFFE, true);
    auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();

    // OpenSubGraph
    EXPECT_EQ(ENN_RET_SUCCESS, gpu_ud.OpenSubGraph(*op_list));
    CHECK_OP_LIST_UID(op_list->get_id());

    // Make BufferTable
    model::memory::BufferTable buffer_table;

    auto in_mem =
        emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * GetDimSize(input_dims), enn::EnnMmType::kEnnMmTypeIon);
    auto out_mem =
        emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * GetDimSize(output_dims), enn::EnnMmType::kEnnMmTypeIon);

    CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem, out_mem}), "CreateMemory() failed");

    in_mem->va = static_cast<void *>(input.get());
    buffer_table.add(0, in_mem->va, in_mem->size);
    buffer_table.add(1, out_mem->va, out_mem->size);

    // Prepare & Execute
    auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
        MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    EXPECT_EQ(gpu_ud.PrepareSubGraph(*executable_operator_list), ENN_RET_SUCCESS);

    auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list);
    EXPECT_EQ(gpu_ud.ExecuteSubGraph(operator_list_execute_request), ENN_RET_SUCCESS);
    compare((float *)out_mem->va, expect_out.get(), GetDimSize(output_dims), error_threshold);

    emm->DeleteMemory(in_mem);
    emm->DeleteMemory(out_mem);
    gpu_ud.CloseSubGraph(*op_list);

    gpu_ud.Deinitialize();
}

TEST_F(ENN_GT_UNIT_TEST_GPU_UD, gpu_ud_test_mean) {
    ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();
    enn::model::component::OperatorBuilder operator_builder;

    NDims input_dims = {4, 3, 2, 1};
    NDims output_dims = {2, 1, 1, 1};
    // Make parameter
    float input_data[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                          13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};

    float expect_output[] = {12, 13};

    int32_t axis[] = {1, 0, -3, -3};
    NDims axis_dims = {1, 1, (int)(sizeof(axis) / sizeof(int32_t)), 1};


    float error_threshold = 0.17;

    enn::model::component::ParameterBuilder parameter_builder;
    auto param_axis = create_parameter(parameter_builder, const_cast<char *>("axis"), axis, axis_dims, TFlite::TensorType_FLOAT32);
    auto ifm =
        create_feature_map(feature_map_builder, std::string("Tensor_000[1]"), 0, input_dims, TFlite::TensorType_FLOAT32, true);
    auto ofm =
        create_feature_map(feature_map_builder, std::string("Tensor_000[2]"), 1, output_dims, TFlite::TensorType_FLOAT32, false);
    flatbuffers::FlatBufferBuilder build_data;
#ifdef SCHEMA_NNC_V1
    auto tmp = tflite::CreateMeanOptions(build_data);
#else
    auto tmp = TFlite::CreateENN_MeanOptions(build_data);
#endif
    build_data.Finish(tmp);
    void *base_addr =
        build_data.GetBufferPointer() +
        (flatbuffers::EndianScalar(*reinterpret_cast<flatbuffers::uoffset_t *>(build_data.GetBufferPointer())));

    std::shared_ptr<enn::model::component::Operator> op =
        operator_builder.set_id(0)
            .set_name(TFlite::EnumNameBuiltinOperator(TFlite::BuiltinOperator_MEAN))
            .set_code(TFlite::BuiltinOperator_MEAN)
            .set_accelerator(model::Accelerator::GPU)
            .add_in_tensor(ifm)
            .add_in_tensor(param_axis)
            .add_out_tensor(ofm)
#ifdef SCHEMA_NNC_V1
            .set_option(TFlite::EnumNameBuiltinOptions(tflite::BuiltinOptions_MeanOptions),
                        base_addr,
                        build_data.GetSize(),
                        tflite::BuiltinOptions_MeanOptions)
#else
            .set_option(TFlite::EnumNameBuiltinOptions(TFlite::BuiltinOptions_ENN_MeanOptions),
                        base_addr,
                        build_data.GetSize(),
                        TFlite::BuiltinOptions_ENN_MeanOptions)
#endif
            .create();
    auto attribute = std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_TENSORFLOW, false);
    auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();
    gpu_ud.OpenSubGraph(*op_list);

    // Make BufferTable
    model::memory::BufferTable buffer_table;

    auto in_mem =
        emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * GetDimSize(input_dims), enn::EnnMmType::kEnnMmTypeIon);
    auto out_mem =
        emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * GetDimSize(output_dims), enn::EnnMmType::kEnnMmTypeIon);

    CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem, out_mem}), "CreateMemory() failed");

    in_mem->va = static_cast<void *>(input_data);
    buffer_table.add(0, in_mem->va, in_mem->size);
    buffer_table.add(1, out_mem->va, out_mem->size);

    // Prepare & Execute
    auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
        MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    EXPECT_EQ(gpu_ud.PrepareSubGraph(*executable_operator_list), ENN_RET_SUCCESS);

    auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list);
    EXPECT_EQ(gpu_ud.ExecuteSubGraph(operator_list_execute_request), ENN_RET_SUCCESS);
    compare((float *)out_mem->va, expect_output, GetDimSize(output_dims), error_threshold);

    emm->DeleteMemory(in_mem);
    emm->DeleteMemory(out_mem);
    gpu_ud.CloseSubGraph(*op_list);

    gpu_ud.Deinitialize();
}

TEST_F(ENN_GT_UNIT_TEST_GPU_UD, gpu_ud_test_scale) {
    ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();

    // Define test case
    NDims input_dims = {2, 2, 1, 2};
    NDims scale_dims = {1, 2, 1, 1};
    NDims bias_dims = {1, 2, 1, 1};
    NDims output_dims = {2, 2, 1, 2};
    float error_threshold = 1e-2;
    std::unique_ptr<float[]> input(new float[GetDimSize(input_dims)]{2.4, -0.6, 1.2, 6.3, -6.2, 7.1, 9.0, 0});
    std::unique_ptr<float[]> scale_data(new float[GetDimSize(scale_dims)]{0.3, 0.2});
    std::unique_ptr<float[]> bias_data(new float[GetDimSize(bias_dims)]{0.2, 3.5});
    std::unique_ptr<float[]> expect_out(new float[GetDimSize(output_dims)]{0.92, 0.02, 3.74, 4.76, -1.66, 2.33, 5.3, 3.5});

    // Make IFM & OFM
    auto ifm = create_feature_map(feature_map_builder, std::string("Tensor_000[1]"), 0, input_dims,
        TFlite::TensorType_FLOAT32, true);
    auto ofm = create_feature_map(feature_map_builder, std::string("Tensor_000[2]"), 1, output_dims,
        TFlite::TensorType_FLOAT32, false);
    // Make parameter
    auto scale = create_parameter(parameter_builder, const_cast<char *>("Scale_Data"), scale_data.get(), scale_dims,
        TFlite::TensorType_FLOAT32);
    auto bias = create_parameter(parameter_builder, const_cast<char *>("Bias_Data"), bias_data.get(), bias_dims,
        TFlite::TensorType_FLOAT32);

    // Make operator

    auto op = operator_builder.set_id(0)
#ifdef SCHEMA_NNC_V1
        .set_name(TFlite::EnumNameBuiltinOperator(TFlite::BuiltinOperator_SCALE))
        .set_code(TFlite::BuiltinOperator_SCALE)
#else
        .set_name(TFlite::EnumNameBuiltinOperator(TFlite::BuiltinOperator_ENN_SCALE))
        .set_code(TFlite::BuiltinOperator_ENN_SCALE)
#endif
        .set_accelerator(model::Accelerator::GPU)
        .add_in_tensor(ifm)
        .add_in_tensor(scale)
        .add_in_tensor(bias)
        .add_out_tensor(ofm)
        .create();

    // Make OperatorList
    auto attribute = std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_CAFFE, true);
    auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();

    // OpenSubGraph
    gpu_ud.OpenSubGraph(*op_list);

    // Make BufferTable
    model::memory::BufferTable buffer_table;

    auto in_mem = emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * GetDimSize(input_dims),
        enn::EnnMmType::kEnnMmTypeIon);
    auto out_mem = emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * GetDimSize(output_dims),
        enn::EnnMmType::kEnnMmTypeIon);
    CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem, out_mem}), "CreateMemory() failed");

    in_mem->va = static_cast<void *>(input.get());
    buffer_table.add(0, in_mem->va, in_mem->size);
    buffer_table.add(1, out_mem->va, out_mem->size);

    // Prepare & Execute
    auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
        MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    EXPECT_EQ(gpu_ud.PrepareSubGraph(*executable_operator_list), ENN_RET_SUCCESS);

    auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list);
    EXPECT_EQ(gpu_ud.ExecuteSubGraph(operator_list_execute_request), ENN_RET_SUCCESS);
    compare((float *)out_mem->va, expect_out.get(), GetDimSize(output_dims), error_threshold);

    emm->DeleteMemory(in_mem);
    emm->DeleteMemory(out_mem);
    gpu_ud.CloseSubGraph(*op_list);

    gpu_ud.Deinitialize();
}

TEST_F(ENN_GT_UNIT_TEST_GPU_UD, gpu_ud_test_slice) {
    ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();

    // Define test case
    NDims input_dims = {6, 2, 1, 2};
    NDims output_dims = {6, 2, 1, 2};
    float error_threshold = 1e-2;
    int32_t axis = 0;
    std::vector<int32_t> slicePoint;
    std::unique_ptr<float[]> input(new float[GetDimSize(input_dims)]{0, 0.1, -0.2, 0.3,
                                                                     -0.4, 0.5, 0.6, -0.7,
                                                                     0.8, 0.9, 1, -1.1,
                                                                     1.2, 1.3, -1.4, 1.5,
                                                                     1.6, 1.7, -1.8, 1.9,
                                                                     2, 2.1, -2.2, 2.3});
    std::unique_ptr<float[]> expect_out(new float[GetDimSize(output_dims)]{0, 0.1, -0.2, 0.3,
                                                                           -0.4, 0.5, 0.6, -0.7,
                                                                           0.8, 0.9, 1, -1.1,
                                                                           1.2, 1.3, -1.4, 1.5,
                                                                           1.6, 1.7, -1.8, 1.9,
                                                                           2, 2.1, -2.2, 2.3});

    // Make IFM & OFM
    auto ifm = create_feature_map(feature_map_builder, std::string("Tensor_000[1]"), 0, input_dims,
        TFlite::TensorType_FLOAT32, true);
    auto ofm = create_feature_map(feature_map_builder, std::string("Tensor_000[2]"), 1, output_dims,
        TFlite::TensorType_FLOAT32, false);

    // Make parameter
    flatbuffers::FlatBufferBuilder build_data;
    auto tmp = TFlite::CreateSliceOptions(build_data, axis, build_data.CreateVector<int32_t>({}));
    build_data.Finish(tmp);
    void *base_addr = build_data.GetBufferPointer() +
        (flatbuffers::EndianScalar(*reinterpret_cast<flatbuffers::uoffset_t *>(build_data.GetBufferPointer())));

    // Make operator
    auto op = operator_builder.set_id(0)
        .set_name(TFlite::EnumNameBuiltinOperator(TFlite::BuiltinOperator_SLICE))
        .set_code(TFlite::BuiltinOperator_SLICE)
        .set_accelerator(model::Accelerator::GPU)
        .add_in_tensor(ifm)
        .add_out_tensor(ofm)
        .set_option(TFlite::EnumNameBuiltinOptions(TFlite::BuiltinOptions_SliceOptions),
                    base_addr,
                    build_data.GetSize(),
                    TFlite::BuiltinOptions_SliceOptions)
        .create();

    // Make OperatorList
    auto attribute = std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_CAFFE, true);
    auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();

    // OpenSubGraph
    gpu_ud.OpenSubGraph(*op_list);

    // Make BufferTable
    model::memory::BufferTable buffer_table;

    auto in_mem = emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * GetDimSize(input_dims),
        enn::EnnMmType::kEnnMmTypeIon);
    auto out_mem = emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * GetDimSize(output_dims),
        enn::EnnMmType::kEnnMmTypeIon);
    CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem, out_mem}), "CreateMemory() failed");

    in_mem->va = static_cast<void *>(input.get());
    buffer_table.add(0, in_mem->va, in_mem->size);
    buffer_table.add(1, out_mem->va, out_mem->size);

    // Prepare & Execute
    auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
        MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    EXPECT_EQ(gpu_ud.PrepareSubGraph(*executable_operator_list), ENN_RET_SUCCESS);

    auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list);
    EXPECT_EQ(gpu_ud.ExecuteSubGraph(operator_list_execute_request), ENN_RET_SUCCESS);
    compare((float *)out_mem->va, expect_out.get(), GetDimSize(output_dims), error_threshold);

    emm->DeleteMemory(in_mem);
    emm->DeleteMemory(out_mem);
    gpu_ud.CloseSubGraph(*op_list);

    gpu_ud.Deinitialize();
}

template <typename PRECISION> class CLReluTester : public ENN_GT_UNIT_TEST_GPU_UD {
public:
    CLReluTester() {
        precision_ = PRECISION::precision;
        switch (precision_) {
        case PrecisionType::FP32:
            data_type_ = DataType::FLOAT;
            error_threshold_ = 1e-5;
            break;
        case PrecisionType::FP16:
            data_type_ = DataType::FLOAT;
            error_threshold_ = 1e-2;
            break;
        default: break;
        }
    }

    CLReluTester &TestPrepare(const Dim4 &input_dim) {
        input_dim_ = input_dim;
        output_dim_ = input_dim;
        input_size_ = GetDimSize(input_dim_);
        output_size_ = GetDimSize(output_dim_);
        parameters_ = std::make_shared<ud::gpu::ReluParameters>();
        parameters_->negative_slope = 0.0f;

        return *this;
    }

    void TestRun(float *input_data = nullptr, float *golden_data = nullptr) {
        std::shared_ptr<float> input;
        std::shared_ptr<float> output;
        if (input_data == nullptr || golden_data == nullptr) {
            input = make_shared_array<float>(input_size_);
            output = make_shared_array<float>(output_size_);
            GenerateRandom(input.get(), input_size_, -1, 1);
            memset(output.get(), 0, output_size_ * sizeof(float));

            ReluGuard relu_guard;
            relu_guard.GuardPrepare(input_size_);
            relu_guard.GuardRun(input.get(), output.get());
            input_data = input.get();
            golden_data = output.get();
        }

        enn::model::component::FeatureMapBuilder feature_map_builder;
        // 1. Make IFM
        enn::model::component::FeatureMap::Ptr ifm =
            feature_map_builder.set_id(0)
                .set_name(std::string("Tensor_000[0]"))
                .set_data_type(TFlite::TensorType_FLOAT32)
                .set_buffer_index(0)
                .set_buffer_size(input_size_)
                .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_INPUT)
                .set_shape(std::vector<uint32_t>{input_dim_.n, input_dim_.c, input_dim_.h, input_dim_.w})
                .create();

        // 2. Make OFM
        enn::model::component::FeatureMap::Ptr ofm =
            feature_map_builder.set_id(1)
                .set_name(std::string("Tensor_000[1]"))
                .set_data_type(TFlite::TensorType_FLOAT32)
                .set_buffer_index(1)
                .set_buffer_size(output_size_)
                .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_OUTPUT)
                .set_shape(std::vector<uint32_t>{output_dim_.n, output_dim_.c, output_dim_.h, output_dim_.w})
                .create();

        // 3. Make Options
        ud::TC_ReluOptions options;
        options.negative_slope = 0.0f;

        enn::model::component::OperatorBuilder operator_builder;
        // 4. Make Operator
        const std::string op_name = "RELU";
        enn::model::component::Operator::Ptr op =
            operator_builder.set_id(op_name.length())
                .set_name(op_name)
                .set_code(TFlite::BuiltinOperator::BuiltinOperator_RELU)
                .set_accelerator(model::Accelerator::GPU)
#ifdef SCHEMA_NNC_V1
                .set_option(EnumNameBuiltinOptions(TFlite::BuiltinOptions::BuiltinOptions_ReluOptions),
                            &options,
                            sizeof(ud::TC_ReluOptions),
                            TFlite::BuiltinOptions::BuiltinOptions_ReluOptions)
#else
                .set_option(EnumNameBuiltinOptions(TFlite::BuiltinOptions::BuiltinOptions_ENN_ReluOptions),
                            &options,
                            sizeof(ud::TC_ReluOptions),
                            TFlite::BuiltinOptions::BuiltinOptions_ENN_ReluOptions)
#endif
                .add_in_tensor(ifm)
                .add_out_tensor(ofm)
                .create();

        // 5. Make OpList
        auto attribute = std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_CAFFE,
                                                                 precision_ == PrecisionType::FP32 ? false : true);
        auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();

        doRun(input_data, golden_data, op_list);
    }

private:
    void doRun(float *input_data, float *golden_data, model::component::OperatorList::Ptr op_list) {
        ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();

        EXPECT_EQ(gpu_ud.OpenSubGraph(*op_list), ENN_RET_SUCCESS);

        auto in_mem = emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * input_size_, enn::EnnMmType::kEnnMmTypeIon);
        auto out_mem =
            emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * output_size_, enn::EnnMmType::kEnnMmTypeIon);
        CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem, out_mem}), "CreateMemory() failed");

        // Make BufferTable
        memcpy(in_mem->va, input_data, data_size[TFlite::TensorType_FLOAT32] * input_size_);
        model::memory::BufferTable buffer_table;
        buffer_table.add(0, in_mem->va, in_mem->size);
        buffer_table.add(1, out_mem->va, out_mem->size);

        // Prepare & Execute
        auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
            MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
        EXPECT_EQ(gpu_ud.PrepareSubGraph(*executable_operator_list), ENN_RET_SUCCESS);

        auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list);
        EXPECT_EQ(gpu_ud.ExecuteSubGraph(operator_list_execute_request), ENN_RET_SUCCESS);

        compare((float *)out_mem->va, golden_data, output_size_, error_threshold_);

        EXPECT_EQ(emm->DeleteMemory(in_mem), ENN_RET_SUCCESS);
        EXPECT_EQ(emm->DeleteMemory(out_mem), ENN_RET_SUCCESS);
        gpu_ud.CloseSubGraph(*op_list);
        EXPECT_EQ(gpu_ud.Deinitialize(), ENN_RET_SUCCESS);
    }

private:
    PrecisionType precision_;
    DataType data_type_;
    float error_threshold_;

    Dim4 input_dim_;
    Dim4 output_dim_;
    size_t input_size_;
    size_t output_size_;
    std::shared_ptr<ud::gpu::ReluParameters> parameters_;
};
TYPED_TEST_CASE(CLReluTester, TestFP32AndFP16Type);
TYPED_TEST(CLReluTester, Simple4D) {
    float input_data[] = {-0.5, 0.2, 0.3, -0.1, -0.4, 0.9};
    float golden_data[] = {0, 0.2, 0.3, 0, 0, 0.9};
    this->TestPrepare({2, 1, 3, 1}).TestRun(input_data, golden_data);
}

TYPED_TEST(CLReluTester, DynamicOutput_Random4D_4) { this->TestPrepare({2, 3, 4, 5}).TestRun(); }

template <typename PRECISION> class CLRelu1Tester : public ENN_GT_UNIT_TEST_GPU_UD {
public:
    CLRelu1Tester() {
        precision_ = PRECISION::precision;
        switch (precision_) {
        case PrecisionType::FP32:
            data_type_ = DataType::FLOAT;
            error_threshold_ = 1e-5;
            break;
        case PrecisionType::FP16:
            data_type_ = DataType::FLOAT;
            error_threshold_ = 1e-2;
            break;
        default: break;
        }
    }

    CLRelu1Tester &TestPrepare(const Dim4 &input_dim) {
        input_dim_ = input_dim;
        output_dim_ = input_dim;
        input_size_ = GetDimSize(input_dim_);
        output_size_ = GetDimSize(output_dim_);

        return *this;
    }

    void TestRun(float *input_data = nullptr, float *golden_data = nullptr) {
        std::shared_ptr<float> input;
        std::shared_ptr<float> output;
        if (input_data == nullptr || golden_data == nullptr) {
            input = make_shared_array<float>(input_size_);
            output = make_shared_array<float>(output_size_);
            GenerateRandom(input.get(), input_size_, -1, 1);
            memset(output.get(), 0, output_size_ * sizeof(float));

            Relu1Guard relu1_guard;
            relu1_guard.GuardPrepare(input_size_);
            relu1_guard.GuardRun(input.get(), output.get());
            input_data = input.get();
            golden_data = output.get();
        }

        enn::model::component::FeatureMapBuilder feature_map_builder;
        // 1. Make IFM
        enn::model::component::FeatureMap::Ptr ifm =
            feature_map_builder.set_id(0)
                .set_name(std::string("Tensor_000[0]"))
                .set_data_type(TFlite::TensorType_FLOAT32)
                .set_buffer_index(0)
                .set_buffer_size(input_size_)
                .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_INPUT)
                .set_shape(std::vector<uint32_t>{input_dim_.n, input_dim_.c, input_dim_.h, input_dim_.w})
                .create();

        // 2. Make OFM
        enn::model::component::FeatureMap::Ptr ofm =
            feature_map_builder.set_id(1)
                .set_name(std::string("Tensor_000[1]"))
                .set_data_type(TFlite::TensorType_FLOAT32)
                .set_buffer_index(1)
                .set_buffer_size(output_size_)
                .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_OUTPUT)
                .set_shape(std::vector<uint32_t>{output_dim_.n, output_dim_.c, output_dim_.h, output_dim_.w})
                .create();

        enn::model::component::OperatorBuilder operator_builder;
        // 3. Make Operator
        const std::string op_name = "RELU1";
        enn::model::component::Operator::Ptr op = operator_builder.set_id(op_name.length())
                                                      .set_name(op_name)
                                                      .set_code(TFlite::BuiltinOperator::BuiltinOperator_RELU_N1_TO_1)
                                                      .set_accelerator(model::Accelerator::GPU)
                                                      .add_in_tensor(ifm)
                                                      .add_out_tensor(ofm)
                                                      .create();

        // 5. Make OpList
        auto attribute = std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_CAFFE,
                                                                 precision_ == PrecisionType::FP32 ? false : true);
        auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();

        doRun(input_data, golden_data, op_list);
    }

private:
    void doRun(float *input_data, float *golden_data, enn::model::component::OperatorList::Ptr op_list) {
        ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();
        EXPECT_EQ(gpu_ud.OpenSubGraph(*op_list), ENN_RET_SUCCESS);

        auto in_mem = emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * input_size_, enn::EnnMmType::kEnnMmTypeIon);
        auto out_mem =
            emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * output_size_, enn::EnnMmType::kEnnMmTypeIon);
        CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem, out_mem}), "CreateMemory() failed");

        // Make BufferTable
        memcpy(in_mem->va, input_data, data_size[TFlite::TensorType_FLOAT32] * input_size_);
        model::memory::BufferTable buffer_table;
        buffer_table.add(0, in_mem->va, in_mem->size);
        buffer_table.add(1, out_mem->va, out_mem->size);

        // Prepare & Execute
        auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
            MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
        EXPECT_EQ(gpu_ud.PrepareSubGraph(*executable_operator_list), ENN_RET_SUCCESS);

        auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list);
        EXPECT_EQ(gpu_ud.ExecuteSubGraph(operator_list_execute_request), ENN_RET_SUCCESS);

        compare((float *)out_mem->va, golden_data, output_size_, error_threshold_);

        EXPECT_EQ(emm->DeleteMemory(in_mem), ENN_RET_SUCCESS);
        EXPECT_EQ(emm->DeleteMemory(out_mem), ENN_RET_SUCCESS);

        gpu_ud.CloseSubGraph(*op_list);
        EXPECT_EQ(gpu_ud.Deinitialize(), ENN_RET_SUCCESS);
    }

private:
    PrecisionType precision_;

    DataType data_type_;
    float error_threshold_;

    Dim4 input_dim_;
    Dim4 output_dim_;
    size_t input_size_;
    size_t output_size_;
};
TYPED_TEST_CASE(CLRelu1Tester, TestFP32AndFP16Type);
TYPED_TEST(CLRelu1Tester, Simple4D) {
    float input_data[] = {-3.5, 0.2, 1.3, -0.1, -2.4, 0.9, 1.2, 3.5, 0.2};
    float golden_data[] = {-1, 0.2, 1, -0.1, -1, 0.9, 1, 1, 0.2};
    this->TestPrepare({1, 3, 1, 3}).TestRun(input_data, golden_data);
}

TYPED_TEST(CLRelu1Tester, Dynamic_Random4D) { this->TestPrepare({2, 3, 13, 13}).TestRun(); }

template <typename PRECISION> class CLRelu6Tester : public ENN_GT_UNIT_TEST_GPU_UD {
public:
    CLRelu6Tester() {
        precision_ = PRECISION::precision;
        switch (precision_) {
        case PrecisionType::FP32:
            data_type_ = DataType::FLOAT;
            error_threshold_ = 1e-5;
            break;
        case PrecisionType::FP16:
            data_type_ = DataType::FLOAT;
            error_threshold_ = 1e-2;
            break;
        default: break;
        }
    }

    CLRelu6Tester &TestPrepare(const Dim4 &input_dim) {
        input_dim_ = input_dim;
        output_dim_ = input_dim;
        input_size_ = GetDimSize(input_dim_);
        output_size_ = GetDimSize(output_dim_);

        return *this;
    }

    void TestRun(float *input_data = nullptr, float *golden_data = nullptr) {
        std::shared_ptr<float> input;
        std::shared_ptr<float> output;
        if (input_data == nullptr || golden_data == nullptr) {
            input = make_shared_array<float>(input_size_);
            output = make_shared_array<float>(output_size_);
            GenerateRandom(input.get(), input_size_, -1, 1);
            memset(output.get(), 0, output_size_ * sizeof(float));

            Relu6Guard relu6_guard;
            relu6_guard.GuardPrepare(input_size_);
            relu6_guard.GuardRun(input.get(), output.get());
            input_data = input.get();
            golden_data = output.get();
        }

        enn::model::component::FeatureMapBuilder feature_map_builder;
        // 1. Make IFM
        enn::model::component::FeatureMap::Ptr ifm =
            feature_map_builder.set_id(0)
                .set_name(std::string("Tensor_000[0]"))
                .set_data_type(TFlite::TensorType_FLOAT32)
                .set_buffer_index(0)
                .set_buffer_size(input_size_)
                .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_INPUT)
                .set_shape(std::vector<uint32_t>{input_dim_.n, input_dim_.c, input_dim_.h, input_dim_.w})
                .create();

        // 2. Make OFM
        enn::model::component::FeatureMap::Ptr ofm =
            feature_map_builder.set_id(1)
                .set_name(std::string("Tensor_000[1]"))
                .set_data_type(TFlite::TensorType_FLOAT32)
                .set_buffer_index(1)
                .set_buffer_size(output_size_)
                .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_OUTPUT)
                .set_shape(std::vector<uint32_t>{output_dim_.n, output_dim_.c, output_dim_.h, output_dim_.w})
                .create();

        enn::model::component::OperatorBuilder operator_builder;
        // 3. Make Operator
        const std::string op_name = "RELU6";
        enn::model::component::Operator::Ptr op = operator_builder.set_id(op_name.length())
                                                      .set_name(op_name)
                                                      .set_code(TFlite::BuiltinOperator::BuiltinOperator_RELU6)
                                                      .set_accelerator(model::Accelerator::GPU)
                                                      .add_in_tensor(ifm)
                                                      .add_out_tensor(ofm)
                                                      .create();

        // 5. Make OpList
        auto attribute = std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_CAFFE,
                                                                 precision_ == PrecisionType::FP32 ? false : true);
        auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();

        doRun(input_data, golden_data, op_list);
    }

private:
    void doRun(float *input_data, float *golden_data, enn::model::component::OperatorList::Ptr op_list) {
        ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();
        EXPECT_EQ(gpu_ud.OpenSubGraph(*op_list), ENN_RET_SUCCESS);

        auto in_mem = emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * input_size_, enn::EnnMmType::kEnnMmTypeIon);
        auto out_mem =
            emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * output_size_, enn::EnnMmType::kEnnMmTypeIon);
        CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem, out_mem}), "CreateMemory() failed");

        // Make BufferTable
        memcpy(in_mem->va, input_data, data_size[TFlite::TensorType_FLOAT32] * input_size_);
        model::memory::BufferTable buffer_table;
        buffer_table.add(0, in_mem->va, in_mem->size);
        buffer_table.add(1, out_mem->va, out_mem->size);

        // Prepare & Execute
        auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
            MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
        EXPECT_EQ(gpu_ud.PrepareSubGraph(*executable_operator_list), ENN_RET_SUCCESS);

        auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list);
        EXPECT_EQ(gpu_ud.ExecuteSubGraph(operator_list_execute_request), ENN_RET_SUCCESS);

        compare((float *)out_mem->va, golden_data, output_size_, error_threshold_);

        EXPECT_EQ(emm->DeleteMemory(in_mem), ENN_RET_SUCCESS);
        EXPECT_EQ(emm->DeleteMemory(out_mem), ENN_RET_SUCCESS);

        gpu_ud.CloseSubGraph(*op_list);
        EXPECT_EQ(gpu_ud.Deinitialize(), ENN_RET_SUCCESS);
    }

private:
    PrecisionType precision_;

    DataType data_type_;
    float error_threshold_;

    Dim4 input_dim_;
    Dim4 output_dim_;
    size_t input_size_;
    size_t output_size_;
};
TYPED_TEST_CASE(CLRelu6Tester, TestFP32AndFP16Type);
TYPED_TEST(CLRelu6Tester, Simple4D) {
    float input_data[] = {-3.5, 7.2, 4.3, -0.1, -2.4, 0.9, 3.5, 10.0, 5.9};
    float golden_data[] = {0, 6, 4.3, 0, 0, 0.9, 3.5, 6, 5.9};
    this->TestPrepare({3, 3, 1, 1}).TestRun(input_data, golden_data);
}

TYPED_TEST(CLRelu6Tester, Dynamic_Random4D) { this->TestPrepare({2, 3, 13, 13}).TestRun(); }

template <typename PRECISION> class CLSigmoidTester : public ENN_GT_UNIT_TEST_GPU_UD {
public:
    CLSigmoidTester() {
        precision_ = PRECISION::precision;
        switch (precision_) {
        case PrecisionType::FP32:
            data_type_ = DataType::FLOAT;
            error_threshold_ = 1e-5;
            break;
        case PrecisionType::FP16:
            data_type_ = DataType::FLOAT;
            error_threshold_ = 1e-2;
            break;
        default: break;
        }
    }

    CLSigmoidTester &TestPrepare(const Dim4 &input_dim) {
        input_dim_ = input_dim;
        output_dim_ = input_dim;
        input_size_ = GetDimSize(input_dim_);
        output_size_ = GetDimSize(output_dim_);

        return *this;
    }

    void TestRun(float *input_data = nullptr) {
        std::shared_ptr<float> output = make_shared_array<float>(output_size_);
        memset(output.get(), 0, output_size_ * sizeof(float));
        float *golden_data = output.get();

        std::shared_ptr<float> input;
        if (input_data == nullptr) {
            input = make_shared_array<float>(input_size_);
            GenerateRandom(input.get(), input_size_, -1, 1);
            input_data = input.get();
        }
        SigmoidGuard sigmoid_guard;
        sigmoid_guard.GuardPrepare(input_size_);
        sigmoid_guard.GuardRun(input_data, golden_data);

        enn::model::component::FeatureMapBuilder feature_map_builder;
        // 1. Make IFM
        enn::model::component::FeatureMap::Ptr ifm =
            feature_map_builder.set_id(0)
                .set_name(std::string("Tensor_000[0]"))
                .set_data_type(TFlite::TensorType_FLOAT32)
                .set_buffer_index(0)
                .set_buffer_size(input_size_)
                .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_INPUT)
                .set_shape(std::vector<uint32_t>{input_dim_.n, input_dim_.c, input_dim_.h, input_dim_.w})
                .create();

        // 2. Make OFM
        enn::model::component::FeatureMap::Ptr ofm =
            feature_map_builder.set_id(1)
                .set_name(std::string("Tensor_000[1]"))
                .set_data_type(TFlite::TensorType_FLOAT32)
                .set_buffer_index(1)
                .set_buffer_size(output_size_)
                .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_OUTPUT)
                .set_shape(std::vector<uint32_t>{output_dim_.n, output_dim_.c, output_dim_.h, output_dim_.w})
                .create();

        enn::model::component::OperatorBuilder operator_builder;
        // 3. Make Operator
        const std::string op_name = "SIGMOID";
        enn::model::component::Operator::Ptr op = operator_builder.set_id(op_name.length())
                                                      .set_name(op_name)
                                                      .set_code(TFlite::BuiltinOperator::BuiltinOperator_LOGISTIC)
                                                      .set_accelerator(model::Accelerator::GPU)
                                                      .add_in_tensor(ifm)
                                                      .add_out_tensor(ofm)
                                                      .create();

        // 5. Make OpList
        auto attribute = std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_CAFFE,
                                                                 precision_ == PrecisionType::FP32 ? false : true);
        auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();

        doRun(input_data, golden_data, op_list);
    }

private:
    void doRun(float *input_data, float *golden_data, enn::model::component::OperatorList::Ptr op_list) {
        ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();
        EXPECT_EQ(gpu_ud.OpenSubGraph(*op_list), ENN_RET_SUCCESS);

        auto in_mem = emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * input_size_, enn::EnnMmType::kEnnMmTypeIon);
        auto out_mem =
            emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * output_size_, enn::EnnMmType::kEnnMmTypeIon);
        CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem, out_mem}), "CreateMemory() failed");

        // Make BufferTable
        memcpy(in_mem->va, input_data, data_size[TFlite::TensorType_FLOAT32] * input_size_);
        model::memory::BufferTable buffer_table;
        buffer_table.add(0, in_mem->va, in_mem->size);
        buffer_table.add(1, out_mem->va, out_mem->size);

        // Prepare & Execute
        auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
            MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
        EXPECT_EQ(gpu_ud.PrepareSubGraph(*executable_operator_list), ENN_RET_SUCCESS);

        auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list);
        EXPECT_EQ(gpu_ud.ExecuteSubGraph(operator_list_execute_request), ENN_RET_SUCCESS);

        compare((float *)out_mem->va, golden_data, output_size_, error_threshold_);

        EXPECT_EQ(emm->DeleteMemory(in_mem), ENN_RET_SUCCESS);
        EXPECT_EQ(emm->DeleteMemory(out_mem), ENN_RET_SUCCESS);

        gpu_ud.CloseSubGraph(*op_list);
        EXPECT_EQ(gpu_ud.Deinitialize(), ENN_RET_SUCCESS);
    }

private:
    PrecisionType precision_;
    DataType data_type_;
    float error_threshold_;

    Dim4 input_dim_;
    Dim4 output_dim_;
    size_t input_size_;
    size_t output_size_;
};
TYPED_TEST_CASE(CLSigmoidTester, TestFP32AndFP16Type);
TYPED_TEST(CLSigmoidTester, Simple4D) {
    float input_data[] = {-3.5, 7.2, 4.3, -0.1, -2.4, 0.9, -3.5, 7.2, 0.3, -0.1, 2.4, 0.9, -3.5, 2.2, 4.3, 0.1, -2.4, 0.9};
    this->TestPrepare({3, 1, 2, 3}).TestRun(input_data);
}

template <typename PRECISION> class CLTanhTester : public ENN_GT_UNIT_TEST_GPU_UD {
public:
    CLTanhTester() {
        precision_ = PRECISION::precision;
        switch (precision_) {
        case PrecisionType::FP32:
            data_type_ = DataType::FLOAT;
            error_threshold_ = 1e-5;
            break;
        case PrecisionType::FP16:
            data_type_ = DataType::FLOAT;
            error_threshold_ = 1e-2;
            break;
        default: break;
        }
    }

    CLTanhTester &TestPrepare(const Dim4 &input_dim) {
        input_dim_ = input_dim;
        output_dim_ = input_dim;
        input_size_ = GetDimSize(input_dim_);
        output_size_ = GetDimSize(output_dim_);

        return *this;
    }

    void TestRun(float *input_data, float *golden_data) {
        enn::model::component::FeatureMapBuilder feature_map_builder;
        // 1. Make IFM
        enn::model::component::FeatureMap::Ptr ifm =
            feature_map_builder.set_id(0)
                .set_name(std::string("Tensor_000[0]"))
                .set_data_type(TFlite::TensorType_FLOAT32)
                .set_buffer_index(0)
                .set_buffer_size(input_size_)
                .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_INPUT)
                .set_shape(std::vector<uint32_t>{input_dim_.n, input_dim_.c, input_dim_.h, input_dim_.w})
                .create();

        // 2. Make OFM
        enn::model::component::FeatureMap::Ptr ofm =
            feature_map_builder.set_id(1)
                .set_name(std::string("Tensor_000[1]"))
                .set_data_type(TFlite::TensorType_FLOAT32)
                .set_buffer_index(1)
                .set_buffer_size(output_size_)
                .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_OUTPUT)
                .set_shape(std::vector<uint32_t>{output_dim_.n, output_dim_.c, output_dim_.h, output_dim_.w})
                .create();

        enn::model::component::OperatorBuilder operator_builder;
        // 3. Make Operator
        const std::string op_name = "TANH";
        enn::model::component::Operator::Ptr op = operator_builder.set_id(op_name.length())
                                                      .set_name(op_name)
                                                      .set_code(TFlite::BuiltinOperator::BuiltinOperator_TANH)
                                                      .set_accelerator(model::Accelerator::GPU)
                                                      .add_in_tensor(ifm)
                                                      .add_out_tensor(ofm)
                                                      .create();

        // 5. Make OpList
        auto attribute = std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_CAFFE,
                                                                 precision_ == PrecisionType::FP32 ? false : true);
        auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();

        doRun(input_data, golden_data, op_list);
    }

private:
    void doRun(float *input_data, float *golden_data, enn::model::component::OperatorList::Ptr op_list) {
        ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();
        EXPECT_EQ(gpu_ud.OpenSubGraph(*op_list), ENN_RET_SUCCESS);

        auto in_mem = emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * input_size_, enn::EnnMmType::kEnnMmTypeIon);
        auto out_mem =
            emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * output_size_, enn::EnnMmType::kEnnMmTypeIon);
        CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem, out_mem}), "CreateMemory() failed");

        // Make BufferTable
        memcpy(in_mem->va, input_data, data_size[TFlite::TensorType_FLOAT32] * input_size_);
        model::memory::BufferTable buffer_table;
        buffer_table.add(0, in_mem->va, in_mem->size);
        buffer_table.add(1, out_mem->va, out_mem->size);

        // Prepare & Execute
        auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
            MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
        EXPECT_EQ(gpu_ud.PrepareSubGraph(*executable_operator_list), ENN_RET_SUCCESS);

        auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list);
        EXPECT_EQ(gpu_ud.ExecuteSubGraph(operator_list_execute_request), ENN_RET_SUCCESS);

        compare((float *)out_mem->va, golden_data, output_size_, error_threshold_);

        EXPECT_EQ(emm->DeleteMemory(in_mem), ENN_RET_SUCCESS);
        EXPECT_EQ(emm->DeleteMemory(out_mem), ENN_RET_SUCCESS);

        gpu_ud.CloseSubGraph(*op_list);
        EXPECT_EQ(gpu_ud.Deinitialize(), ENN_RET_SUCCESS);
    }

private:
    PrecisionType precision_;

    DataType data_type_;
    float error_threshold_;

    Dim4 input_dim_;
    Dim4 output_dim_;
    size_t input_size_;
    size_t output_size_;
};
TYPED_TEST_CASE(CLTanhTester, TestFP32AndFP16Type);
TYPED_TEST(CLTanhTester, allpositive) {
    float input_data[] = {1.2, 0.1, 3.2, 8.1, 0.4, 5.7};
    float golden_data[] = {0.83365, 0.09967, 0.99668, 1.00000, 0.37995, 0.99998};
    this->TestPrepare({1, 2, 3, 1}).TestRun(input_data, golden_data);
}

TEST_F(ENN_GT_UNIT_TEST_GPU_UD, gpu_ud_test_avg_pool) {
    ud::gpu::GpuUserDriver &gpu_ud = ud::gpu::GpuUserDriver::get_instance();

    gpu_ud.Initialize();
    enn::model::component::OperatorBuilder operator_builder;
    NDims input_dims = {1, 1, 2, 2};
    Dim2 stride = {1, 1};
    Dim2 filter = {1, 1};
    NDims output_dims = {1, 1, 2, 2};
    float error_threshold = 0.17;

    float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float reference_output[] = {1.0f, 2.0f, 3.0f, 4.0f};

    auto ifm = create_feature_map(
        feature_map_builder, std::string("Tensor_000[1]"), 0, input_dims, TFlite::TensorType_FLOAT32, true);
    auto ofm = create_feature_map(
        feature_map_builder, std::string("Tensor_000[2]"), 1, output_dims, TFlite::TensorType_FLOAT32, false);

    flatbuffers::FlatBufferBuilder build_data;
    TFlite::ActivationFunctionType activation = TFlite::ActivationFunctionType::ActivationFunctionType_NONE;
    TFlite::Padding padding = TFlite::Padding::Padding_SAME;
    auto tmp = TFlite::CreatePool2DOptions(build_data, padding, stride.w, stride.h, filter.w, filter.h, activation);
    build_data.Finish(tmp);
    void *base_addr =
        build_data.GetBufferPointer() +
        (flatbuffers::EndianScalar(*reinterpret_cast<flatbuffers::uoffset_t *>(build_data.GetBufferPointer())));

    enn::model::component::Operator::Ptr op =
        operator_builder.set_id(0)
            .set_name(TFlite::EnumNameBuiltinOperator(TFlite::BuiltinOperator_AVERAGE_POOL_2D))
            .set_code(TFlite::BuiltinOperator_AVERAGE_POOL_2D)
            .set_accelerator(model::Accelerator::GPU)
            .add_in_tensor(ifm)
            .add_out_tensor(ofm)
            .set_option(TFlite::EnumNameBuiltinOptions(TFlite::BuiltinOptions_Pool2DOptions),
                        base_addr,
                        build_data.GetSize(),
                        TFlite::BuiltinOptions_Pool2DOptions)
            .create();

    // Make OperatorList
    auto attribute = std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_CAFFE, false);
    auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();
    gpu_ud.OpenSubGraph(*op_list);

    // Make BufferTable
    model::memory::BufferTable buffer_table;

    auto in_mem =
        emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * GetDimSize(input_dims), enn::EnnMmType::kEnnMmTypeIon);
    auto out_mem =
        emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * GetDimSize(output_dims), enn::EnnMmType::kEnnMmTypeIon);

    CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem, out_mem}), "CreateMemory() failed");

    in_mem->va = static_cast<void *>(input_data);
    buffer_table.add(0, in_mem->va, in_mem->size);
    buffer_table.add(1, out_mem->va, out_mem->size);

    // Prepare & Execute
    auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
        MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    EXPECT_EQ(gpu_ud.PrepareSubGraph(*executable_operator_list), ENN_RET_SUCCESS);

    auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list);
    EXPECT_EQ(gpu_ud.ExecuteSubGraph(operator_list_execute_request), ENN_RET_SUCCESS);
    compare((float *)out_mem->va, reference_output, GetDimSize(output_dims), error_threshold);

    emm->DeleteMemory(in_mem);
    emm->DeleteMemory(out_mem);
    gpu_ud.CloseSubGraph(*op_list);

    gpu_ud.Deinitialize();
}

TEST_F(ENN_GT_UNIT_TEST_GPU_UD, gpu_ud_test_max_pool) {
    ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();
    enn::model::component::OperatorBuilder operator_builder;
    NDims input_dims = {1, 1, 2, 2};
    Dim2 stride = {1, 1};
    Dim2 filter = {1, 1};
    NDims output_dims = {1, 1, 2, 2};
    float error_threshold = 0.17;

    float input_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float reference_output[] = {1.0f, 2.0f, 3.0f, 4.0f};

    auto ifm = create_feature_map(
        feature_map_builder, std::string("Tensor_000[1]"), 0, input_dims, TFlite::TensorType_FLOAT32, true);
    auto ofm = create_feature_map(
        feature_map_builder, std::string("Tensor_000[2]"), 1, output_dims, TFlite::TensorType_FLOAT32, false);

    flatbuffers::FlatBufferBuilder build_data;
    TFlite::ActivationFunctionType activation = TFlite::ActivationFunctionType::ActivationFunctionType_NONE;
    TFlite::Padding padding = TFlite::Padding::Padding_SAME;
    auto tmp = TFlite::CreatePool2DOptions(build_data, padding, stride.w, stride.h, filter.w, filter.h, activation);
    build_data.Finish(tmp);
    void *base_addr =
        build_data.GetBufferPointer() +
        (flatbuffers::EndianScalar(*reinterpret_cast<flatbuffers::uoffset_t *>(build_data.GetBufferPointer())));

    enn::model::component::Operator::Ptr op =
        operator_builder.set_id(0)
            .set_name(TFlite::EnumNameBuiltinOperator(TFlite::BuiltinOperator_MAX_POOL_2D))
            .set_code(TFlite::BuiltinOperator_MAX_POOL_2D)
            .set_accelerator(model::Accelerator::GPU)
            .add_in_tensor(ifm)
            .add_out_tensor(ofm)
            .set_option(TFlite::EnumNameBuiltinOptions(TFlite::BuiltinOptions_Pool2DOptions),
                        base_addr,
                        build_data.GetSize(),
                        TFlite::BuiltinOptions_Pool2DOptions)
            .create();

    // Make OperatorList
    auto attribute = std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_CAFFE, false);
    auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();
    gpu_ud.OpenSubGraph(*op_list);

    // Make BufferTable
    model::memory::BufferTable buffer_table;

    auto in_mem =
        emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * GetDimSize(input_dims), enn::EnnMmType::kEnnMmTypeIon);
    auto out_mem =
        emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * GetDimSize(output_dims), enn::EnnMmType::kEnnMmTypeIon);

    CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem, out_mem}), "CreateMemory() failed");

    in_mem->va = static_cast<void *>(input_data);
    buffer_table.add(0, in_mem->va, in_mem->size);
    buffer_table.add(1, out_mem->va, out_mem->size);

    // Prepare & Execute
    auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
        MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    EXPECT_EQ(gpu_ud.PrepareSubGraph(*executable_operator_list), ENN_RET_SUCCESS);

    auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list);
    EXPECT_EQ(gpu_ud.ExecuteSubGraph(operator_list_execute_request), ENN_RET_SUCCESS);
    compare((float *)out_mem->va, reference_output, GetDimSize(output_dims), error_threshold);

    emm->DeleteMemory(in_mem);
    emm->DeleteMemory(out_mem);
    gpu_ud.CloseSubGraph(*op_list);

    gpu_ud.Deinitialize();
}

template <typename PRECISION> class CLConcatTester : public ENN_GT_UNIT_TEST_GPU_UD {
public:
    CLConcatTester() {
        precision_ = PRECISION::precision;
        switch (precision_) {
        case PrecisionType::FP32:
            data_type_ = DataType::FLOAT;
            error_threshold_ = 1e-5;
            break;
        case PrecisionType::FP16:
            data_type_ = DataType::FLOAT;
            error_threshold_ = 1e-2;
            break;
        default: break;
        }
    }

    CLConcatTester &TestPrepare(const std::vector<Dim4> &input_dims, const Dim4 &output_dim, const int32_t &axis) {
        input_dims_ = input_dims;
        output_dim_ = output_dim;
        for (auto input_dim : input_dims_) {
            input_sizes_.push_back(GetDimSize(input_dim));
        }
        output_size_ = GetDimSize(output_dim_);
        parameters_ = std::make_shared<ud::gpu::ConcatParameters>();
        parameters_->axis = axis;

        return *this;
    }

    void TestRun(std::vector<float *> input_data = {}, float *golden_data = nullptr) {
        std::vector<std::shared_ptr<float>> input(input_sizes_.size());
        std::shared_ptr<float> output;
        if (golden_data == nullptr) {
            input_data.clear();
            for (size_t i = 0; i < input_sizes_.size(); ++i) {
                input[i] = make_shared_array<float>(input_sizes_[i]);
                GenerateRandom(input[i].get(), input_sizes_[i], -2, 2);
                input_data.push_back(input[i].get());
            }
            output = make_shared_array<float>(output_size_);
            memset(output.get(), 0, output_size_ * sizeof(float));
            golden_data = output.get();

            ConcatGuard concat_guard;

            std::vector<std::vector<uint32_t>> input_shape(input_dims_.size());
            for (size_t i = 0; i < input_dims_.size(); ++i) {
                input_shape[i].push_back(input_dims_[i].n);
                input_shape[i].push_back(input_dims_[i].c);
                input_shape[i].push_back(input_dims_[i].h);
                input_shape[i].push_back(input_dims_[i].w);
            }

            concat_guard.GuardPrepare(input_shape, parameters_->axis);
            concat_guard.GuardRun(input_data, golden_data);
        }

        enn::model::component::FeatureMapBuilder feature_map_builder;
        // 1. Make IFM
        std::vector<enn::model::component::FeatureMap::Ptr> ifms;
        for (size_t i = 0; i < input_dims_.size(); ++i) {
            std::string name = std::string("Tensor_000[") + std::to_string(i) + std::string("]");
            enn::model::component::FeatureMap::Ptr ifm =
                feature_map_builder.set_id(i)
                    .set_name(name)
                    .set_data_type(TFlite::TensorType_FLOAT32)
                    .set_buffer_index(i)
                    .set_buffer_size(input_sizes_[i])
                    .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_INPUT)
                    .set_shape(std::vector<uint32_t>{input_dims_[i].n, input_dims_[i].c, input_dims_[i].h, input_dims_[i].w})
                    .create();
            ifms.push_back(ifm);
        }

        // 2. Make OFM
        std::string name = std::string("Tensor_000[") + std::to_string(input_dims_.size()) + std::string("]");
        enn::model::component::FeatureMap::Ptr ofm =
            feature_map_builder.set_id(input_dims_.size())
                .set_name(name)
                .set_data_type(TFlite::TensorType_FLOAT32)
                .set_buffer_index(input_dims_.size())
                .set_buffer_size(output_size_)
                .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_OUTPUT)
                .set_shape(std::vector<uint32_t>{output_dim_.n, output_dim_.c, output_dim_.h, output_dim_.w})
                .create();

        // 3. Make Options
        ud::TC_ConcatDOptions options;
        options.axis = parameters_->axis;
        options.activation_info = parameters_->activation_info;

        enn::model::component::OperatorBuilder operator_builder;
        // 4. Make Operator
        const std::string op_name = "CONCATENATION";
        operator_builder.set_id(op_name.length())
            .set_name(op_name)
            .set_code(TFlite::BuiltinOperator::BuiltinOperator_CONCATENATION)
            .set_accelerator(model::Accelerator::GPU)
            .set_option(EnumNameBuiltinOptions(TFlite::BuiltinOptions::BuiltinOptions_ConcatenationOptions),
                        &options,
                        sizeof(ud::TC_ConcatDOptions),
                        TFlite::BuiltinOptions::BuiltinOptions_ConcatenationOptions)
            .add_out_tensor(ofm);
        for (auto ifm : ifms) {
            operator_builder.add_in_tensor(ifm);
        }
        enn::model::component::Operator::Ptr op = operator_builder.create();

        // 5. Make OpList
        auto attribute = std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_CAFFE,
                                                                 precision_ == PrecisionType::FP32 ? false : true);
        auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();

        doRun(input_data, golden_data, op_list);
    }

private:
    void doRun(std::vector<float *> input_data, float *golden_data, enn::model::component::OperatorList::Ptr op_list) {
        ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();
        EXPECT_EQ(gpu_ud.OpenSubGraph(*op_list), ENN_RET_SUCCESS);

        std::vector<EnnBufferCore::Ptr> in_mems;
        std::vector<EnnBufferCore::Ptr> all_mems;
        for (auto input_size : input_sizes_) {
            auto in_mem =
                emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * input_size, enn::EnnMmType::kEnnMmTypeIon);
            in_mems.push_back(in_mem);
            all_mems.push_back(in_mem);
        }
        auto out_mem =
            emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * output_size_, enn::EnnMmType::kEnnMmTypeIon);
        all_mems.push_back(out_mem);
        CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, all_mems), "CreateMemory() failed");

        // Make BufferTable
        model::memory::BufferTable buffer_table;
        for (size_t i = 0; i < in_mems.size(); ++i) {
            memcpy(in_mems[i]->va, input_data[i], data_size[TFlite::TensorType_FLOAT32] * input_sizes_[i]);
            buffer_table.add(i, in_mems[i]->va, in_mems[i]->size);
        }
        buffer_table.add(in_mems.size(), out_mem->va, out_mem->size);

        // Prepare & Execute
        auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
            MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
        EXPECT_EQ(gpu_ud.PrepareSubGraph(*executable_operator_list), ENN_RET_SUCCESS);

        auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list);
        EXPECT_EQ(gpu_ud.ExecuteSubGraph(operator_list_execute_request), ENN_RET_SUCCESS);

        compare((float *)out_mem->va, golden_data, output_size_, error_threshold_);

        for (auto in_mem : in_mems) {
            EXPECT_EQ(emm->DeleteMemory(in_mem), ENN_RET_SUCCESS);
        }
        EXPECT_EQ(emm->DeleteMemory(out_mem), ENN_RET_SUCCESS);

        gpu_ud.CloseSubGraph(*op_list);
        EXPECT_EQ(gpu_ud.Deinitialize(), ENN_RET_SUCCESS);
    }

private:
    PrecisionType precision_;

    DataType data_type_;
    float error_threshold_;

    std::vector<Dim4> input_dims_;
    Dim4 output_dim_;
    std::vector<size_t> input_sizes_;
    size_t output_size_;
    std::shared_ptr<ud::gpu::ConcatParameters> parameters_;
};
TYPED_TEST_CASE(CLConcatTester, TestFP32AndFP16Type);
TYPED_TEST(CLConcatTester, random_Axis0) {
    std::vector<Dim4> input_dims = {{2, 3, 4, 1}, {1, 3, 4, 1}, {3, 3, 4, 1}};
    Dim4 output_dim = {6, 3, 4, 1};
    int32_t axis = 0;
    this->TestPrepare(input_dims, output_dim, axis).TestRun();
}

TYPED_TEST(CLConcatTester, Axis1) {
    std::vector<Dim4> input_dims = {{1, 1, 2, 2}, {1, 2, 2, 2}, {1, 3, 2, 2}};
    Dim4 output_dim = {1, 6, 2, 2};
    int32_t axis = 1;

    float tensor1[] = {2.4, 3.3, 4.1, 5.2};
    float tensor2[] = {5.1, 2.6, 8.2, 5.9, -0.2, 0, 3.6, 3.5};
    float tensor3[] = {3.4, 0.1, 6.2, 5.2, 6.1, 0.5, -9.2, -7.4, -3.6, 4.6, 12.6, -9.3};
    std::vector<float *> input_data = {tensor1, tensor2, tensor3};
    float golden_data[] = {2.4, 3.3, 4.1, 5.2, 5.1, 2.6, 8.2,  5.9,  -0.2, 0,   3.6,  3.5,
                           3.4, 0.1, 6.2, 5.2, 6.1, 0.5, -9.2, -7.4, -3.6, 4.6, 12.6, -9.3};

    this->TestPrepare(input_dims, output_dim, axis).TestRun(input_data, golden_data);
}

template <typename PRECISION> class CLSoftmaxTester : public ENN_GT_UNIT_TEST_GPU_UD {
public:
    CLSoftmaxTester() {
        precision_ = PRECISION::precision;
        switch (precision_) {
        case PrecisionType::FP32:
            data_type_ = DataType::FLOAT;
            error_threshold_ = 1e-5;
            break;
        case PrecisionType::FP16:
            data_type_ = DataType::FLOAT;
            error_threshold_ = 1e-2;
            break;
        default: break;
        }
    }

    CLSoftmaxTester &TestPrepare(const Dim4 &input_dim, const int32_t &axis, const float &beta) {
        input_dim_ = input_dim;
        output_dim_ = input_dim;
        input_size_ = GetDimSize(input_dim_);
        output_size_ = GetDimSize(output_dim_);
        parameters_ = std::make_shared<ud::gpu::SoftmaxParameters>();
        parameters_->axis = axis;
        parameters_->beta = beta;
        return *this;
    }

    void TestRun(float *input_data = nullptr, float *golden_data = nullptr) {
        std::shared_ptr<float> input;
        std::shared_ptr<float> output;
        if (input_data == nullptr || golden_data == nullptr) {
            input = make_shared_array<float>(input_size_);
            output = make_shared_array<float>(output_size_);
            GenerateRandom(input.get(), input_size_, 0, 1);
            memset(output.get(), 0, output_size_ * sizeof(float));

            SoftmaxGuard softmax_guard;
            softmax_guard.GuardPrepare(input_dim_, parameters_->axis, parameters_->beta);
            softmax_guard.GuardRun(input.get(), output.get());
            input_data = input.get();
            golden_data = output.get();
        }

        enn::model::component::FeatureMapBuilder feature_map_builder;
        // 1. Make IFM
        enn::model::component::FeatureMap::Ptr ifm =
            feature_map_builder.set_id(0)
                .set_name(std::string("Tensor_000[0]"))
                .set_data_type(TFlite::TensorType_FLOAT32)
                .set_buffer_index(0)
                .set_buffer_size(input_size_)
                .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_INPUT)
                .set_shape(std::vector<uint32_t>{input_dim_.n, input_dim_.c, input_dim_.h, input_dim_.w})
                .create();

        // 2. Make OFM
        enn::model::component::FeatureMap::Ptr ofm =
            feature_map_builder.set_id(1)
                .set_name(std::string("Tensor_000[1]"))
                .set_data_type(TFlite::TensorType_FLOAT32)
                .set_buffer_index(1)
                .set_buffer_size(output_size_)
                .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_OUTPUT)
                .set_shape(std::vector<uint32_t>{output_dim_.n, output_dim_.c, output_dim_.h, output_dim_.w})
                .create();

        // 3. Make Options
        ud::TC_SoftmaxOptions options;
        options.axis_ = parameters_->axis;
        options.beta_ = parameters_->beta;

        enn::model::component::OperatorBuilder operator_builder;
        // 4. Make Operator
        const std::string op_name = "SOFTMAX";
        enn::model::component::Operator::Ptr op =
            operator_builder.set_id(op_name.length())
                .set_name(op_name)
                .set_code(TFlite::BuiltinOperator::BuiltinOperator_SOFTMAX)
                .set_accelerator(model::Accelerator::GPU)
                .set_option(EnumNameBuiltinOptions(TFlite::BuiltinOptions::BuiltinOptions_SoftmaxOptions),
                            &options,
                            sizeof(ud::TC_SoftmaxOptions),
                            TFlite::BuiltinOptions::BuiltinOptions_SoftmaxOptions)
                .add_in_tensor(ifm)
                .add_out_tensor(ofm)
                .create();

        // 5. Make OpList
        auto attribute = std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_CAFFE,
                                                                 precision_ == PrecisionType::FP32 ? false : true);
        auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();

        doRun(input_data, golden_data, op_list);
    }

private:
    void doRun(float *input_data, float *golden_data, enn::model::component::OperatorList::Ptr op_list) {
        ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();
        EXPECT_EQ(gpu_ud.OpenSubGraph(*op_list), ENN_RET_SUCCESS);

        auto in_mem = emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * input_size_, enn::EnnMmType::kEnnMmTypeIon);
        auto out_mem =
            emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * output_size_, enn::EnnMmType::kEnnMmTypeIon);
        CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem, out_mem}), "CreateMemory() failed");

        // Make BufferTable
        memcpy(in_mem->va, input_data, data_size[TFlite::TensorType_FLOAT32] * input_size_);
        model::memory::BufferTable buffer_table;
        buffer_table.add(0, in_mem->va, in_mem->size);
        buffer_table.add(1, out_mem->va, out_mem->size);

        // Prepare & Execute
        auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
            MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
        EXPECT_EQ(gpu_ud.PrepareSubGraph(*executable_operator_list), ENN_RET_SUCCESS);

        auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list);
        EXPECT_EQ(gpu_ud.ExecuteSubGraph(operator_list_execute_request), ENN_RET_SUCCESS);

        compare((float *)out_mem->va, golden_data, output_size_, error_threshold_);

        EXPECT_EQ(emm->DeleteMemory(in_mem), ENN_RET_SUCCESS);
        EXPECT_EQ(emm->DeleteMemory(out_mem), ENN_RET_SUCCESS);

        gpu_ud.CloseSubGraph(*op_list);
        EXPECT_EQ(gpu_ud.Deinitialize(), ENN_RET_SUCCESS);
    }

private:
    PrecisionType precision_;

    DataType data_type_;
    float error_threshold_;

    Dim4 input_dim_;
    Dim4 output_dim_;
    size_t input_size_;
    size_t output_size_;
    std::shared_ptr<ud::gpu::SoftmaxParameters> parameters_;
};
TYPED_TEST_CASE(CLSoftmaxTester, TestFP32AndFP16Type);
TYPED_TEST(CLSoftmaxTester, test_axis0_batch8xsize1) {
    const int32_t axis = 0;
    const float beta = 1.0;
    this->TestPrepare({64, 1, 1, 1}, axis, beta).TestRun();
}

TYPED_TEST(CLSoftmaxTester, vts2) {
    const int32_t axis = 1;
    const float beta = 1.0;
    float input_data[10] = {1., 2., 3., 4., 5., -1., -2., -3., -4., -5.};
    float golden_data[10] = {0.011656231,
                             0.031684921,
                             0.086128544,
                             0.234121657,
                             0.636408647,
                             0.636408647,
                             0.234121657,
                             0.086128544,
                             0.031684921,
                             0.011656231};
    this->TestPrepare({2, 5, 1, 1}, axis, beta).TestRun(input_data, golden_data);
}

template <typename PRECISION> class CLSubTester : public ENN_GT_UNIT_TEST_GPU_UD {
public:
    CLSubTester() {
        precision_ = PRECISION::precision;
        switch (precision_) {
        case PrecisionType::FP32:
            data_type_ = DataType::FLOAT;
            error_threshold_ = 1e-5;
            break;
        case PrecisionType::FP16:
            data_type_ = DataType::FLOAT;
            error_threshold_ = 1e-2;
            break;
        default: break;
        }
    }

    CLSubTester &TestPrepare(const Dim4 &input_dim_0, const Dim4 &input_dim_1, const Dim4 &output_dim) {
        input_dim_0_ = input_dim_0;
        input_dim_1_ = input_dim_1;
        output_dim_ = output_dim;
        input_size_0_ = GetDimSize(input_dim_0_);
        input_size_1_ = GetDimSize(input_dim_1_);
        output_size_ = GetDimSize(output_dim_);
        parameters_ = std::make_shared<ud::gpu::SubParameters>();

        return *this;
    }

    void TestRun(float *input_data_0, float *input_data_1, float *golden_data) {
        enn::model::component::FeatureMapBuilder feature_map_builder;
        // 1. Make IFM
        enn::model::component::FeatureMap::Ptr ifm_0 =
            feature_map_builder.set_id(0)
                .set_name(std::string("Tensor_000[0]"))
                .set_data_type(TFlite::TensorType_FLOAT32)
                .set_buffer_index(0)
                .set_buffer_size(input_size_0_)
                .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_INPUT)
                .set_shape(std::vector<uint32_t>{input_dim_0_.n, input_dim_0_.c, input_dim_0_.h, input_dim_0_.w})
                .create();

        enn::model::component::FeatureMap::Ptr ifm_1 =
            feature_map_builder.set_id(1)
                .set_name(std::string("Tensor_000[1]"))
                .set_data_type(TFlite::TensorType_FLOAT32)
                .set_buffer_index(1)
                .set_buffer_size(input_size_1_)
                .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_INPUT)
                .set_shape(std::vector<uint32_t>{input_dim_1_.n, input_dim_1_.c, input_dim_1_.h, input_dim_1_.w})
                .create();

        // 2. Make OFM
        enn::model::component::FeatureMap::Ptr ofm =
            feature_map_builder.set_id(1)
                .set_name(std::string("Tensor_000[2]"))
                .set_data_type(TFlite::TensorType_FLOAT32)
                .set_buffer_index(2)
                .set_buffer_size(output_size_)
                .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_OUTPUT)
                .set_shape(std::vector<uint32_t>{output_dim_.n, output_dim_.c, output_dim_.h, output_dim_.w})
                .create();

        // 3. Make Options
        // TODO(all): create TFlite::SubOptions
        ud::TC_SubOptions options;

        enn::model::component::OperatorBuilder operator_builder;
        // 4. Make Operator
        const std::string op_name = EnumNameBuiltinOperator(TFlite::BuiltinOperator::BuiltinOperator_SUB);
        enn::model::component::Operator::Ptr op =
            operator_builder.set_id(op_name.length())
                .set_name(op_name)
                .set_code(TFlite::BuiltinOperator::BuiltinOperator_SUB)
                .set_accelerator(model::Accelerator::GPU)
                // TODO(all): set TFlite::SubOptions
                .set_option(EnumNameBuiltinOptions(TFlite::BuiltinOptions::BuiltinOptions_SubOptions),
                            &options,
                            sizeof(ud::TC_SubOptions),
                            TFlite::BuiltinOptions::BuiltinOptions_SubOptions)
                .add_in_tensor(ifm_0)
                .add_in_tensor(ifm_1)
                .add_out_tensor(ofm)
                .create();

        // 5. Make OpList
        auto attribute = std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_CAFFE,
                                                                 precision_ == PrecisionType::FP32 ? false : true);
        auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();

        doRun(input_data_0, input_data_1, golden_data, op_list);
    }

private:
    void doRun(float *input_data_0,
               float *input_data_1,
               float *golden_data,
               enn::model::component::OperatorList::Ptr op_list) {
        ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();
        EXPECT_EQ(gpu_ud.OpenSubGraph(*op_list), ENN_RET_SUCCESS);

        auto in_mem_0 =
            emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * input_size_0_, enn::EnnMmType::kEnnMmTypeIon);
        auto in_mem_1 =
            emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * input_size_1_, enn::EnnMmType::kEnnMmTypeIon);
        auto out_mem =
            emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * output_size_, enn::EnnMmType::kEnnMmTypeIon);
        CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem_0, in_mem_1, out_mem}), "CreateMemory() failed");

        // Make BufferTable
        memcpy(in_mem_0->va, input_data_0, data_size[TFlite::TensorType_FLOAT32] * input_size_0_);
        memcpy(in_mem_1->va, input_data_1, data_size[TFlite::TensorType_FLOAT32] * input_size_1_);
        model::memory::BufferTable buffer_table;
        buffer_table.add(0, in_mem_0->va, in_mem_0->size);
        buffer_table.add(1, in_mem_1->va, in_mem_1->size);
        buffer_table.add(2, out_mem->va, out_mem->size);

        // Prepare & Execute
        auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
            MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
        EXPECT_EQ(gpu_ud.PrepareSubGraph(*executable_operator_list), ENN_RET_SUCCESS);

        auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list);
        EXPECT_EQ(gpu_ud.ExecuteSubGraph(operator_list_execute_request), ENN_RET_SUCCESS);

        compare((float *)out_mem->va, golden_data, output_size_, error_threshold_);

        EXPECT_EQ(emm->DeleteMemory(in_mem_0), ENN_RET_SUCCESS);
        EXPECT_EQ(emm->DeleteMemory(in_mem_1), ENN_RET_SUCCESS);
        EXPECT_EQ(emm->DeleteMemory(out_mem), ENN_RET_SUCCESS);
        // EXPECT_EQ(gpu_ud.CloseSubGraph(op_list->get_id()), ENN_RET_SUCCESS);
        gpu_ud.CloseSubGraph(*op_list);
        EXPECT_EQ(gpu_ud.Deinitialize(), ENN_RET_SUCCESS);
    }

private:
    PrecisionType precision_;
    DataType data_type_;
    float error_threshold_;

    Dim4 input_dim_0_;
    Dim4 input_dim_1_;
    Dim4 output_dim_;
    size_t input_size_0_;
    size_t input_size_1_;
    size_t output_size_;
    std::shared_ptr<ud::gpu::SubParameters> parameters_;
};
TYPED_TEST_CASE(CLSubTester, TestFP32AndFP16Type);
TYPED_TEST(CLSubTester, test_sub_no_broadcast) {
    float input_0[] = {-2.0, 0.2, 1.7, 0.5};
    float input_1[] = {0.1, 0.2, 0.3, 0.8};
    float golden_data[] = {-2.1, 0.0, 1.4, -0.3};
    this->TestPrepare({1, 2, 2, 1}, {1, 2, 2, 1}, {1, 2, 2, 1}).TestRun(input_0, input_1, golden_data);
}

TYPED_TEST(CLSubTester, test_sub_with_broadcast) {
    float input_0[] = {-2.0, 0.2, 1.7, 0.5, -1.1, 2.0};
    float input_1[] = {0.5};
    float golden_data[] = {-2.5, -0.3, 1.2, 0.0, -1.6, 1.5};
    this->TestPrepare({1, 2, 1, 3}, {1, 1, 1, 1}, {1, 2, 1, 3}).TestRun(input_0, input_1, golden_data);
}

template <typename PRECISION> class CLMulTester : public ENN_GT_UNIT_TEST_GPU_UD {
public:
    CLMulTester() {
        precision_ = PRECISION::precision;
        switch (precision_) {
        case PrecisionType::FP32:
            data_type_ = DataType::FLOAT;
            error_threshold_ = 1e-5;
            break;
        case PrecisionType::FP16:
            data_type_ = DataType::FLOAT;
            error_threshold_ = 1e-2;
            break;
        default: break;
        }
    }

    CLMulTester &TestPrepare(const Dim4 &input_dim_0, const Dim4 &input_dim_1, const Dim4 &output_dim) {
        input_dim_0_ = input_dim_0;
        input_dim_1_ = input_dim_1;
        output_dim_ = output_dim;
        input_size_0_ = GetDimSize(input_dim_0_);
        input_size_1_ = GetDimSize(input_dim_1_);
        output_size_ = GetDimSize(output_dim_);
        parameters_ = std::make_shared<ud::gpu::MulParameters>();

        return *this;
    }

    void TestRun(float *input_data_0, float *input_data_1, float *golden_data) {
        enn::model::component::FeatureMapBuilder feature_map_builder;
        // 1. Make IFM
        enn::model::component::FeatureMap::Ptr ifm_0 =
            feature_map_builder.set_id(0)
                .set_name(std::string("Tensor_000[0]"))
                .set_data_type(TFlite::TensorType_FLOAT32)
                .set_buffer_index(0)
                .set_buffer_size(input_size_0_)
                .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_INPUT)
                .set_shape(std::vector<uint32_t>{input_dim_0_.n, input_dim_0_.c, input_dim_0_.h, input_dim_0_.w})
                .create();

        enn::model::component::FeatureMap::Ptr ifm_1 =
            feature_map_builder.set_id(1)
                .set_name(std::string("Tensor_000[1]"))
                .set_data_type(TFlite::TensorType_FLOAT32)
                .set_buffer_index(1)
                .set_buffer_size(input_size_1_)
                .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_INPUT)
                .set_shape(std::vector<uint32_t>{input_dim_1_.n, input_dim_1_.c, input_dim_1_.h, input_dim_1_.w})
                .create();

        // 2. Make OFM
        enn::model::component::FeatureMap::Ptr ofm =
            feature_map_builder.set_id(1)
                .set_name(std::string("Tensor_000[2]"))
                .set_data_type(TFlite::TensorType_FLOAT32)
                .set_buffer_index(2)
                .set_buffer_size(output_size_)
                .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_OUTPUT)
                .set_shape(std::vector<uint32_t>{output_dim_.n, output_dim_.c, output_dim_.h, output_dim_.w})
                .create();

        // 3. Make Options
        // TODO(all): create TFlite::MulOptions
        ud::TC_MulOptions options;

        enn::model::component::OperatorBuilder operator_builder;
        // 4. Make Operator
        const std::string op_name = EnumNameBuiltinOperator(TFlite::BuiltinOperator::BuiltinOperator_MUL);
        enn::model::component::Operator::Ptr op =
            operator_builder.set_id(op_name.length())
                .set_name(op_name)
                .set_code(TFlite::BuiltinOperator::BuiltinOperator_MUL)
                .set_accelerator(model::Accelerator::GPU)
                // TODO(all): set TFlite::MulOptions
                .set_option(EnumNameBuiltinOptions(TFlite::BuiltinOptions::BuiltinOptions_MulOptions),
                            &options,
                            sizeof(ud::TC_MulOptions),
                            TFlite::BuiltinOptions::BuiltinOptions_MulOptions)
                .add_in_tensor(ifm_0)
                .add_in_tensor(ifm_1)
                .add_out_tensor(ofm)
                .create();

        // 5. Make OpList
        auto attribute = std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_CAFFE,
                                                                 precision_ == PrecisionType::FP32 ? false : true);
        auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();

        doRun(input_data_0, input_data_1, golden_data, op_list);
    }

private:
    void doRun(float *input_data_0,
               float *input_data_1,
               float *golden_data,
               enn::model::component::OperatorList::Ptr op_list) {
        ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();
        EXPECT_EQ(gpu_ud.OpenSubGraph(*op_list), ENN_RET_SUCCESS);

        auto in_mem_0 =
            emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * input_size_0_, enn::EnnMmType::kEnnMmTypeIon);
        auto in_mem_1 =
            emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * input_size_1_, enn::EnnMmType::kEnnMmTypeIon);
        auto out_mem =
            emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * output_size_, enn::EnnMmType::kEnnMmTypeIon);
        CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem_0, in_mem_1, out_mem}), "CreateMemory() failed");

        // Make BufferTable
        memcpy(in_mem_0->va, input_data_0, data_size[TFlite::TensorType_FLOAT32] * input_size_0_);
        memcpy(in_mem_1->va, input_data_1, data_size[TFlite::TensorType_FLOAT32] * input_size_1_);
        model::memory::BufferTable buffer_table;
        buffer_table.add(0, in_mem_0->va, in_mem_0->size);
        buffer_table.add(1, in_mem_1->va, in_mem_1->size);
        buffer_table.add(2, out_mem->va, out_mem->size);

        // Prepare & Execute
        auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
            MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
        EXPECT_EQ(gpu_ud.PrepareSubGraph(*executable_operator_list), ENN_RET_SUCCESS);

        auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list);
        EXPECT_EQ(gpu_ud.ExecuteSubGraph(operator_list_execute_request), ENN_RET_SUCCESS);

        compare((float *)out_mem->va, golden_data, output_size_, error_threshold_);

        EXPECT_EQ(emm->DeleteMemory(in_mem_0), ENN_RET_SUCCESS);
        EXPECT_EQ(emm->DeleteMemory(in_mem_1), ENN_RET_SUCCESS);
        EXPECT_EQ(emm->DeleteMemory(out_mem), ENN_RET_SUCCESS);
        // EXPECT_EQ(gpu_ud.CloseSubGraph(op_list->get_id()), ENN_RET_SUCCESS);
        gpu_ud.CloseSubGraph(*op_list);
        EXPECT_EQ(gpu_ud.Deinitialize(), ENN_RET_SUCCESS);
    }

private:
    PrecisionType precision_;
    DataType data_type_;
    float error_threshold_;

    Dim4 input_dim_0_;
    Dim4 input_dim_1_;
    Dim4 output_dim_;
    size_t input_size_0_;
    size_t input_size_1_;
    size_t output_size_;
    std::shared_ptr<ud::gpu::MulParameters> parameters_;
};
TYPED_TEST_CASE(CLMulTester, TestFP32AndFP16Type);
TYPED_TEST(CLMulTester, CoverEachBranch_Pass) {
    float input_data_1[] = {-0.5, 0.2, 0.3, -0.1, -0.4, 0.9};
    float input_data_2[] = {0.8, 0.7, 0.0, -1.5, -0.6, 2.5};
    float out_data[] = {-0.4, 0.14, 0.0, 0.15, 0.24, 2.25};
    this->TestPrepare({1, 2, 3, 1}, {1, 2, 3, 1}, {1, 2, 3, 1}).TestRun(input_data_1, input_data_2, out_data);
}

template <typename PRECISION> class CLDivTester : public ENN_GT_UNIT_TEST_GPU_UD {
public:
    CLDivTester() {
        precision_ = PRECISION::precision;
        switch (precision_) {
        case PrecisionType::FP32:
            data_type_ = DataType::FLOAT;
            error_threshold_ = 1e-5;
            break;
        case PrecisionType::FP16:
            data_type_ = DataType::FLOAT;
            error_threshold_ = 1e-2;
            break;
        default: break;
        }
    }

    CLDivTester &TestPrepare(const Dim4 &input_dim_0,
                             const Dim4 &input_dim_1,
                             const Dim4 &output_dim,
                             ActivationInfo::ActivationType activation_type = ActivationInfo::ActivationType::NONE) {
        input_dim_0_ = input_dim_0;
        input_dim_1_ = input_dim_1;
        output_dim_ = output_dim;
        input_size_0_ = GetDimSize(input_dim_0_);
        input_size_1_ = GetDimSize(input_dim_1_);
        output_size_ = GetDimSize(output_dim_);
        parameters_ = std::make_shared<ud::gpu::DivParameters>();
        parameters_->activation_info = ActivationInfo(activation_type, true);

        return *this;
    }

    void TestRun(float *input_data_0, float *input_data_1, float *golden_data) {
        enn::model::component::FeatureMapBuilder feature_map_builder;
        // 1. Make IFM
        enn::model::component::FeatureMap::Ptr ifm_0 =
            feature_map_builder.set_id(0)
                .set_name(std::string("Tensor_000[0]"))
                .set_data_type(TFlite::TensorType_FLOAT32)
                .set_buffer_index(0)
                .set_buffer_size(input_size_0_)
                .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_INPUT)
                .set_shape(std::vector<uint32_t>{input_dim_0_.n, input_dim_0_.c, input_dim_0_.h, input_dim_0_.w})
                .create();

        enn::model::component::FeatureMap::Ptr ifm_1 =
            feature_map_builder.set_id(1)
                .set_name(std::string("Tensor_000[1]"))
                .set_data_type(TFlite::TensorType_FLOAT32)
                .set_buffer_index(1)
                .set_buffer_size(input_size_1_)
                .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_INPUT)
                .set_shape(std::vector<uint32_t>{input_dim_1_.n, input_dim_1_.c, input_dim_1_.h, input_dim_1_.w})
                .create();

        // 2. Make OFM
        enn::model::component::FeatureMap::Ptr ofm =
            feature_map_builder.set_id(1)
                .set_name(std::string("Tensor_000[2]"))
                .set_data_type(TFlite::TensorType_FLOAT32)
                .set_buffer_index(2)
                .set_buffer_size(output_size_)
                .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_OUTPUT)
                .set_shape(std::vector<uint32_t>{output_dim_.n, output_dim_.c, output_dim_.h, output_dim_.w})
                .create();

        // 3. Make Options
        flatbuffers::FlatBufferBuilder fbb;
        auto activation_function_type =
            static_cast<TFlite::ActivationFunctionType>(parameters_->activation_info.activation());
        fbb.Finish(TFlite::CreateDivOptions(fbb, activation_function_type));
        void *base_addr = fbb.GetBufferPointer() +
                          (flatbuffers::EndianScalar(*reinterpret_cast<flatbuffers::uoffset_t *>(fbb.GetBufferPointer())));

        enn::model::component::OperatorBuilder operator_builder;
        // 4. Make Operator
        const std::string op_name = EnumNameBuiltinOperator(TFlite::BuiltinOperator::BuiltinOperator_DIV);
        enn::model::component::Operator::Ptr op =
            operator_builder.set_id(op_name.length())
                .set_name(op_name)
                .set_code(TFlite::BuiltinOperator::BuiltinOperator_DIV)
                .set_accelerator(model::Accelerator::GPU)
                .set_option(EnumNameBuiltinOptions(TFlite::BuiltinOptions::BuiltinOptions_DivOptions),
                            base_addr,
                            sizeof(TFlite::DivOptions),
                            TFlite::BuiltinOptions::BuiltinOptions_DivOptions)
                .add_in_tensor(ifm_0)
                .add_in_tensor(ifm_1)
                .add_out_tensor(ofm)
                .create();

        // 5. Make OpList
        auto attribute = std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_CAFFE,
                                                                 precision_ == PrecisionType::FP32 ? false : true);
        auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();

        doRun(input_data_0, input_data_1, golden_data, op_list);
    }

private:
    void doRun(float *input_data_0,
               float *input_data_1,
               float *golden_data,
               enn::model::component::OperatorList::Ptr op_list) {
        ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();
        EXPECT_EQ(gpu_ud.OpenSubGraph(*op_list), ENN_RET_SUCCESS);

        auto in_mem_0 =
            emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * input_size_0_, enn::EnnMmType::kEnnMmTypeIon);
        auto in_mem_1 =
            emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * input_size_1_, enn::EnnMmType::kEnnMmTypeIon);
        auto out_mem =
            emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * output_size_, enn::EnnMmType::kEnnMmTypeIon);
        CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem_0, in_mem_1, out_mem}), "CreateMemory() failed");

        // Make BufferTable
        memcpy(in_mem_0->va, input_data_0, data_size[TFlite::TensorType_FLOAT32] * input_size_0_);
        memcpy(in_mem_1->va, input_data_1, data_size[TFlite::TensorType_FLOAT32] * input_size_1_);
        model::memory::BufferTable buffer_table;
        buffer_table.add(0, in_mem_0->va, in_mem_0->size);
        buffer_table.add(1, in_mem_1->va, in_mem_1->size);
        buffer_table.add(2, out_mem->va, out_mem->size);

        // Prepare & Execute
        auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
            MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
        EXPECT_EQ(gpu_ud.PrepareSubGraph(*executable_operator_list), ENN_RET_SUCCESS);
        auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list);
        EXPECT_EQ(gpu_ud.ExecuteSubGraph(operator_list_execute_request), ENN_RET_SUCCESS);

        compare((float *)out_mem->va, golden_data, output_size_, error_threshold_);

        EXPECT_EQ(emm->DeleteMemory(in_mem_0), ENN_RET_SUCCESS);
        EXPECT_EQ(emm->DeleteMemory(in_mem_1), ENN_RET_SUCCESS);
        EXPECT_EQ(emm->DeleteMemory(out_mem), ENN_RET_SUCCESS);
        EXPECT_EQ(gpu_ud.CloseSubGraph(*op_list), ENN_RET_SUCCESS);
        EXPECT_EQ(gpu_ud.Deinitialize(), ENN_RET_SUCCESS);
    }

private:
    PrecisionType precision_;
    DataType data_type_;
    float error_threshold_;

    Dim4 input_dim_0_;
    Dim4 input_dim_1_;
    Dim4 output_dim_;
    size_t input_size_0_;
    size_t input_size_1_;
    size_t output_size_;
    std::shared_ptr<ud::gpu::DivParameters> parameters_;
};
TYPED_TEST_CASE(CLDivTester, TestFP32AndFP16Type);
TYPED_TEST(CLDivTester, test_1_shape) {
    float input1[] = {-0.2, 0.2, -1.2, 0.8};
    float input2[] = {0.5, 0.2, -1.5, 0.5};
    float expect_out[] = {-0.4, 1.0, 0.8, 1.6};
    this->TestPrepare({1, 2, 2, 1}, {1, 2, 2, 1}, {1, 2, 2, 1}).TestRun(input1, input2, expect_out);
}

TYPED_TEST(CLDivTester, test_2_shape) {
    float input1[] = {-0.2, 0.2, 0.07, 0.08, 0.11, -0.123};
    float input2[] = {0.1};
    float expect_out[] = {-2.0, 2.0, 0.7, 0.8, 1.1, -1.23};
    this->TestPrepare({1, 3, 1, 2}, {1, 1, 1, 1}, {1, 3, 1, 2}).TestRun(input1, input2, expect_out);
}

TYPED_TEST(CLDivTester, test_3_shape) {
    float input1[] = {-0.2, 0.2, -1.2, 0.8};
    float input2[] = {0.5, 0.2, -1.5, 0.5};
    float expect_out[] = {0, 1.0, 0.8, 1.6};
    this->TestPrepare({1, 2, 2, 1}, {1, 2, 2, 1}, {1, 2, 2, 1}, ActivationInfo::ActivationType::RELU)
        .TestRun(input1, input2, expect_out);
}

template <typename PRECISION> class CLAddTester : public ENN_GT_UNIT_TEST_GPU_UD {
public:
    CLAddTester() {
        precision_ = PRECISION::precision;
        switch (precision_) {
        case PrecisionType::FP32:
            data_type_ = DataType::FLOAT;
            error_threshold_ = 1e-5;
            break;
        case PrecisionType::FP16:
            data_type_ = DataType::FLOAT;
            error_threshold_ = 1e-2;
            break;
        default: break;
        }
    }

    CLAddTester &TestPrepare(const Dim4 &input_dim_0,
                             const Dim4 &input_dim_1,
                             const Dim4 &output_dim,
                             const std::vector<float> &coeff,
                             ActivationInfo::ActivationType activation_type = ActivationInfo::ActivationType::NONE) {
        input_dim_0_ = input_dim_0;
        input_dim_1_ = input_dim_1;
        output_dim_ = output_dim;
        input_size_0_ = GetDimSize(input_dim_0_);
        input_size_1_ = GetDimSize(input_dim_1_);
        output_size_ = GetDimSize(output_dim_);
        parameters_ = std::make_shared<ud::gpu::AddParameters>();
        parameters_->activation_info = ActivationInfo(activation_type, true);
        parameters_->coeff = coeff;

        return *this;
    }

    void TestRun(float *input_data_0, float *input_data_1, float *golden_data) {
        enn::model::component::FeatureMapBuilder feature_map_builder;
        // 1. Make IFM
        enn::model::component::FeatureMap::Ptr ifm_0 =
            feature_map_builder.set_id(0)
                .set_name(std::string("Tensor_000[0]"))
                .set_data_type(TFlite::TensorType_FLOAT32)
                .set_buffer_index(0)
                .set_buffer_size(input_size_0_)
                .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_INPUT)
                .set_shape(std::vector<uint32_t>{input_dim_0_.n, input_dim_0_.c, input_dim_0_.h, input_dim_0_.w})
                .create();

        enn::model::component::FeatureMap::Ptr ifm_1 =
            feature_map_builder.set_id(1)
                .set_name(std::string("Tensor_000[1]"))
                .set_data_type(TFlite::TensorType_FLOAT32)
                .set_buffer_index(1)
                .set_buffer_size(input_size_1_)
                .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_INPUT)
                .set_shape(std::vector<uint32_t>{input_dim_1_.n, input_dim_1_.c, input_dim_1_.h, input_dim_1_.w})
                .create();

        // 2. Make OFM
        enn::model::component::FeatureMap::Ptr ofm =
            feature_map_builder.set_id(2)
                .set_name(std::string("Tensor_000[2]"))
                .set_data_type(TFlite::TensorType_FLOAT32)
                .set_buffer_index(2)
                .set_buffer_size(output_size_)
                .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_OUTPUT)
                .set_shape(std::vector<uint32_t>{output_dim_.n, output_dim_.c, output_dim_.h, output_dim_.w})
                .create();

        // 3. Make Options
        flatbuffers::FlatBufferBuilder fbb;
        auto activation_function_type =
            static_cast<TFlite::ActivationFunctionType>(parameters_->activation_info.activation());
#ifndef SCHEMA_NNC_V1
        fbb.Finish(TFlite::CreateAddOptionsDirect(fbb, activation_function_type, false, &(parameters_->coeff)));
#else
        fbb.Finish(TFlite::CreateAddOptionsDirect(fbb, activation_function_type, &(parameters_->coeff)));
#endif
        void *base_addr = fbb.GetBufferPointer() +
                          (flatbuffers::EndianScalar(*reinterpret_cast<flatbuffers::uoffset_t *>(fbb.GetBufferPointer())));

        // 4. Make Operator
        const std::string op_name = EnumNameBuiltinOperator(TFlite::BuiltinOperator::BuiltinOperator_ADD);
        enn::model::component::OperatorBuilder operator_builder;
        enn::model::component::Operator::Ptr op =
            operator_builder.set_id(op_name.length())
                .set_name(op_name)
                .set_code(TFlite::BuiltinOperator::BuiltinOperator_ADD)
                .set_accelerator(model::Accelerator::GPU)
                .set_option(EnumNameBuiltinOptions(TFlite::BuiltinOptions::BuiltinOptions_AddOptions),
                            base_addr,
                            sizeof(TFlite::AddOptions),
                            TFlite::BuiltinOptions::BuiltinOptions_AddOptions)
                .add_in_tensor(ifm_0)
                .add_in_tensor(ifm_1)
                .add_out_tensor(ofm)
                .create();

        // 5. Make OpList
        auto attribute = std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_CAFFE,
                                                                 precision_ == PrecisionType::FP32 ? false : true);
        auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();

        doRun(input_data_0, input_data_1, golden_data, op_list);
    }

private:
    void doRun(float *input_data_0,
               float *input_data_1,
               float *golden_data,
               enn::model::component::OperatorList::Ptr op_list) {
        ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();
        EXPECT_EQ(gpu_ud.OpenSubGraph(*op_list), ENN_RET_SUCCESS);

        auto in_mem_0 =
            emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * input_size_0_, enn::EnnMmType::kEnnMmTypeIon);
        auto in_mem_1 =
            emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * input_size_1_, enn::EnnMmType::kEnnMmTypeIon);
        auto out_mem =
            emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * output_size_, enn::EnnMmType::kEnnMmTypeIon);
        CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem_0, in_mem_1, out_mem}), "CreateMemory() failed");

        // Make BufferTable
        memcpy(in_mem_0->va, input_data_0, data_size[TFlite::TensorType_FLOAT32] * input_size_0_);
        memcpy(in_mem_1->va, input_data_1, data_size[TFlite::TensorType_FLOAT32] * input_size_1_);
        model::memory::BufferTable buffer_table;
        buffer_table.add(0, in_mem_0->va, in_mem_0->size);
        buffer_table.add(1, in_mem_1->va, in_mem_1->size);
        buffer_table.add(2, out_mem->va, out_mem->size);

        // Prepare & Execute
        auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
            MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
        EXPECT_EQ(gpu_ud.PrepareSubGraph(*executable_operator_list), ENN_RET_SUCCESS);
        auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list);
        EXPECT_EQ(gpu_ud.ExecuteSubGraph(operator_list_execute_request), ENN_RET_SUCCESS);

        compare((float *)out_mem->va, golden_data, output_size_, error_threshold_);

        EXPECT_EQ(emm->DeleteMemory(in_mem_0), ENN_RET_SUCCESS);
        EXPECT_EQ(emm->DeleteMemory(in_mem_1), ENN_RET_SUCCESS);
        EXPECT_EQ(emm->DeleteMemory(out_mem), ENN_RET_SUCCESS);
        EXPECT_EQ(gpu_ud.CloseSubGraph(*op_list), ENN_RET_SUCCESS);
        EXPECT_EQ(gpu_ud.Deinitialize(), ENN_RET_SUCCESS);
    }

private:
    PrecisionType precision_;
    DataType data_type_;
    float error_threshold_;

    Dim4 input_dim_0_;
    Dim4 input_dim_1_;
    Dim4 output_dim_;
    size_t input_size_0_;
    size_t input_size_1_;
    size_t output_size_;
    std::shared_ptr<ud::gpu::AddParameters> parameters_;
};
TYPED_TEST_CASE(CLAddTester, TestFP32AndFP16Type);

TYPED_TEST(CLAddTester, EachAxisLargerOne) {
    float input_data_1[] = {-0.1, -0.2, 0.3, 0.0, -0.4, 0.9, 0.9, 1.0, -0.1, -0.2, 0.3, 0.0, -0.4, 0.9, 0.9, 1.0};
    float input_data_2[] = {0.1, -0.7, 0.0, 1.5, 3.1, 10.8, -0.9, 1.8, -0.1, -0.2, 0.6, 0.1, -0.04, 0.9, 0.9, 1.0};
    float out_data[] = {-0.2, 0.5, 0.3, -1.5, -3.5, -9.9, 1.8, -0.8, 0.0, 0.0, -0.3, -0.1, -0.36, 0.0, 0.0, 0.0};
    this->TestPrepare({2, 2, 2, 2}, {2, 2, 2, 2}, {2, 2, 2, 2}, {1.0, -1.0}).TestRun(input_data_1, input_data_2, out_data);
}

TYPED_TEST(CLAddTester, DifferentDimSize) {
    float input_data_1[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
    float input_data_2[] = {1, 2};
    float out_data[] = {1, 0, -1, -2, -3, -4, -5, -6, -7, -7, -8, -9, -10, -11, -12, -13, -14, -15};
    this->TestPrepare({1, 2, 3, 3}, {1, 2, 1, 1}, {1, 2, 3, 3}, {-1.0, 1.0}).TestRun(input_data_1, input_data_2, out_data);
}

TYPED_TEST(CLAddTester, DifferentDimSize_RELU) {
    float input_data_1[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17};
    float input_data_2[] = {1, 2};
    float out_data[] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    this->TestPrepare({1, 2, 3, 3}, {1, 2, 1, 1}, {1, 2, 3, 3}, {-1.0, 1.0}, ActivationInfo::ActivationType::RELU)
        .TestRun(input_data_1, input_data_2, out_data);
}

template <typename PRECISION> class CLSqueezeTester : public ENN_GT_UNIT_TEST_GPU_UD {
public:
    CLSqueezeTester() {
        precision_ = PRECISION::precision;
        switch (precision_) {
        case PrecisionType::FP32:
            data_type_ = DataType::FLOAT;
            error_threshold_ = 1e-5;
            break;
        case PrecisionType::FP16:
            data_type_ = DataType::FLOAT;
            error_threshold_ = 1e-2;
            break;
        default: break;
        }
    }

    CLSqueezeTester &TestPrepare(const Dim4 &input_dim,
                                 const Dim4 &output_dim,
                                 const std::vector<int32_t> &squeeze_dims = {}) {
        input_dim_ = input_dim;
        squeeze_dim_ = {static_cast<uint32_t>(squeeze_dims.size()), 1, 1, 1};
        output_dim_.n = output_dim.n > 0 ? output_dim.n : 1;
        output_dim_.c = output_dim.c > 0 ? output_dim.c : 1;
        output_dim_.h = output_dim.h > 0 ? output_dim.h : 1;
        output_dim_.w = output_dim.w > 0 ? output_dim.w : 1;
        input_size_ = GetDimSize(input_dim_);
        squeeze_size_ = GetDimSize(squeeze_dim_);
        output_size_ = GetDimSize(output_dim_);
        parameters_ = std::make_shared<ud::gpu::SqueezeParameters>();
        parameters_->squeeze_dims = squeeze_dims;
        if (output_size_ == 0) {
            parameters_->androidNN = true;
            output_size_ = input_size_;
        }

        return *this;
    }

    void TestRun(float *input_data, float *golden_data) {
        enn::model::component::FeatureMapBuilder feature_map_builder;
        // 1. Make IFM
        enn::model::component::FeatureMap::Ptr ifm_0 =
            feature_map_builder.set_id(0)
                .set_name(std::string("Tensor_000[0]"))
                .set_data_type(TFlite::TensorType_FLOAT32)
                .set_buffer_index(0)
                .set_buffer_size(input_size_)
                .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_INPUT)
                .set_shape(std::vector<uint32_t>{input_dim_.n, input_dim_.c, input_dim_.h, input_dim_.w})
                .create();

        enn::model::component::FeatureMap::Ptr ifm_1 =
            feature_map_builder.set_id(1)
                .set_name(std::string("Tensor_000[1]"))
                .set_data_type(TFlite::TensorType_INT32)
                .set_buffer_index(1)
                .set_buffer_size(squeeze_size_)
                .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_INPUT)
                .set_shape(std::vector<uint32_t>{squeeze_dim_.n, squeeze_dim_.c, squeeze_dim_.h, squeeze_dim_.w})
                .create();

        // 2. Make OFM
        enn::model::component::FeatureMap::Ptr ofm =
            feature_map_builder.set_id(2)
                .set_name(std::string("Tensor_000[2]"))
                .set_data_type(TFlite::TensorType_FLOAT32)
                .set_buffer_index(2)
                .set_buffer_size(output_size_)
                .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_OUTPUT)
                .set_shape(std::vector<uint32_t>{output_dim_.n, output_dim_.c, output_dim_.h, output_dim_.w})
                .create();

        // 3. Make Options
        flatbuffers::FlatBufferBuilder fbb;
        fbb.Finish(TFlite::CreateSqueezeOptionsDirect(fbb, &(parameters_->squeeze_dims)));
        void *base_addr = fbb.GetBufferPointer() +
                          (flatbuffers::EndianScalar(*reinterpret_cast<flatbuffers::uoffset_t *>(fbb.GetBufferPointer())));

        // 4. Make Operator
        const std::string op_name = EnumNameBuiltinOperator(TFlite::BuiltinOperator::BuiltinOperator_SQUEEZE);
        enn::model::component::OperatorBuilder operator_builder;
        enn::model::component::Operator::Ptr op =
            operator_builder.set_id(op_name.length())
                .set_name(op_name)
                .set_code(TFlite::BuiltinOperator::BuiltinOperator_SQUEEZE)
                .set_accelerator(model::Accelerator::GPU)
                .set_option(EnumNameBuiltinOptions(TFlite::BuiltinOptions::BuiltinOptions_SqueezeOptions),
                            base_addr,
                            sizeof(TFlite::SqueezeOptions),
                            TFlite::BuiltinOptions::BuiltinOptions_SqueezeOptions)
                .add_in_tensor(ifm_0)
                .add_in_tensor(ifm_1)
                .add_out_tensor(ofm)
                .create();

        // 5. Make OpList
        auto attribute = std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_CAFFE,
                                                                 precision_ == PrecisionType::FP32 ? false : true);
        auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();

        doRun(input_data, golden_data, op_list);
    }

private:
    void doRun(float *input_data, float *golden_data, enn::model::component::OperatorList::Ptr op_list) {
        ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();
        EXPECT_EQ(gpu_ud.OpenSubGraph(*op_list), ENN_RET_SUCCESS);

        auto in_mem_0 =
            emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * input_size_, enn::EnnMmType::kEnnMmTypeIon);
        auto in_mem_1 =
            emm->CreateMemory(data_size[TFlite::TensorType_INT32] * squeeze_size_, enn::EnnMmType::kEnnMmTypeIon);
        auto out_mem =
            emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * output_size_, enn::EnnMmType::kEnnMmTypeIon);
        CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem_0, in_mem_1, out_mem}), "CreateMemory() failed");

        // Make BufferTable
        memcpy(in_mem_0->va, input_data, data_size[TFlite::TensorType_FLOAT32] * input_size_);
        memcpy(in_mem_1->va, parameters_->squeeze_dims.data(), data_size[TFlite::TensorType_INT32] * squeeze_size_);
        model::memory::BufferTable buffer_table;
        buffer_table.add(0, in_mem_0->va, in_mem_0->size);
        buffer_table.add(1, in_mem_1->va, in_mem_1->size);
        buffer_table.add(2, out_mem->va, out_mem->size);

        // Prepare & Execute
        auto executable_operator_list = std::make_shared<runtime::ExecutableOperatorList>(
            MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
        EXPECT_EQ(gpu_ud.PrepareSubGraph(*executable_operator_list), ENN_RET_SUCCESS);
        auto operator_list_execute_request = runtime::OperatorListExecuteRequest(executable_operator_list);
        EXPECT_EQ(gpu_ud.ExecuteSubGraph(operator_list_execute_request), ENN_RET_SUCCESS);

        compare((float *)out_mem->va, golden_data, output_size_, error_threshold_);

        EXPECT_EQ(emm->DeleteMemory(in_mem_0), ENN_RET_SUCCESS);
        EXPECT_EQ(emm->DeleteMemory(in_mem_1), ENN_RET_SUCCESS);
        EXPECT_EQ(emm->DeleteMemory(out_mem), ENN_RET_SUCCESS);
        EXPECT_EQ(gpu_ud.CloseSubGraph(*op_list), ENN_RET_SUCCESS);
        EXPECT_EQ(gpu_ud.Deinitialize(), ENN_RET_SUCCESS);
    }

private:
    PrecisionType precision_;
    DataType data_type_;
    float error_threshold_;

    Dim4 input_dim_;
    Dim4 squeeze_dim_;
    Dim4 output_dim_;
    size_t input_size_;
    size_t squeeze_size_;
    size_t output_size_;
    std::shared_ptr<ud::gpu::SqueezeParameters> parameters_;
};
TYPED_TEST_CASE(CLSqueezeTester, TestFP32AndFP16Type);

TYPED_TEST(CLSqueezeTester, squeeze_negative_axis) {
    float input[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                     13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    float expect_out[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                          13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    this->TestPrepare({1, 24, 1, 1}, {24, 1}, {-1, 0}).TestRun(input, expect_out);
}

TEST_F(ENN_GT_UNIT_TEST_GPU_UD, gpu_ud_test_resize_bilinear) {
    PrecisionType precision = PrecisionType::FP16;
    // DataType data_type = DataType::FLOAT;
    float error_threshold = 1e-2;

    Dim4 input_dim = {1, 1, 2, 2};
    Dim4 output_dim = {1, 1, 4, 4};
    size_t input_size = GetDimSize(input_dim);
    size_t output_size = GetDimSize(output_dim);
    auto parameters = std::make_shared<ud::gpu::ResizeBilinearParameters>();
    parameters->new_height = output_dim.h;
    parameters->new_width = output_dim.w;

    float input_data[] = {1, 2, 3, 4, 1, 2, 3, 4};
    float golden_data[] = {1, 1.5, 2, 2, 2, 2.5, 3, 3, 3, 3.5, 4, 4, 3, 3.5, 4, 4};

    enn::model::component::FeatureMapBuilder feature_map_builder;
    // 1. Make IFM
    enn::model::component::FeatureMap::Ptr ifm =
        feature_map_builder.set_id(0)
            .set_name(std::string("Tensor_000[0]"))
            .set_data_type(TFlite::TensorType_FLOAT32)
            .set_buffer_index(0)
            .set_buffer_size(input_size)
            .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_INPUT)
            .set_shape(std::vector<uint32_t>{input_dim.n, input_dim.c, input_dim.h, input_dim.w})
            .create();

    // 2. Make OFM
    enn::model::component::FeatureMap::Ptr ofm =
        feature_map_builder.set_id(1)
            .set_name(std::string("Tensor_000[1]"))
            .set_data_type(TFlite::TensorType_FLOAT32)
            .set_buffer_index(1)
            .set_buffer_size(output_size)
            .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_OUTPUT)
            .set_shape(std::vector<uint32_t>{output_dim.n, output_dim.c, output_dim.h, output_dim.w})
            .create();

    // 3. Make Options
    flatbuffers::FlatBufferBuilder fbb;
#ifdef SCHEMA_NNC_V1
    fbb.Finish(TFlite::CreateResizeBilinearOptions(fbb, parameters->new_height, parameters->new_width));
#else
    fbb.Finish(TFlite::CreateResizeBilinearOptions(fbb));
#endif
    void *base_addr = fbb.GetBufferPointer() +
                      (flatbuffers::EndianScalar(*reinterpret_cast<flatbuffers::uoffset_t *>(fbb.GetBufferPointer())));

    // 4. Make Operator
    const std::string op_name = EnumNameBuiltinOperator(TFlite::BuiltinOperator::BuiltinOperator_RESIZE_BILINEAR);
    enn::model::component::OperatorBuilder operator_builder;
    enn::model::component::Operator::Ptr op =
        operator_builder.set_id(op_name.length())
            .set_name(op_name)
            .set_code(TFlite::BuiltinOperator::BuiltinOperator_RESIZE_BILINEAR)
            .set_accelerator(model::Accelerator::GPU)
            .set_option(EnumNameBuiltinOptions(TFlite::BuiltinOptions::BuiltinOptions_ResizeBilinearOptions),
                        base_addr,
                        sizeof(TFlite::ResizeBilinearOptions),
                        TFlite::BuiltinOptions::BuiltinOptions_ResizeBilinearOptions)
            .add_in_tensor(ifm)
            .add_out_tensor(ofm)
            .create();

    // 5. Make OpList
    auto attribute =
        std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_CAFFE, precision == PrecisionType::FP32 ? false : true);
    auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();

    ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();
    EXPECT_EQ(gpu_ud.OpenSubGraph(*op_list), ENN_RET_SUCCESS);

    auto in_mem = emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * input_size, enn::EnnMmType::kEnnMmTypeIon);
    auto out_mem = emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * output_size, enn::EnnMmType::kEnnMmTypeIon);
    CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem, out_mem}), "CreateMemory() failed");

    // Make BufferTable
    memcpy(in_mem->va, input_data, data_size[TFlite::TensorType_FLOAT32] * input_size);
    model::memory::BufferTable buffer_table;
    buffer_table.add(0, in_mem->va, in_mem->size);
    buffer_table.add(1, out_mem->va, out_mem->size);

    // Prepare & Execute
    auto exec_op_list = std::make_shared<runtime::ExecutableOperatorList>(
        MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    auto req_op_list = runtime::OperatorListExecuteRequest(exec_op_list);
    EXPECT_EQ(gpu_ud.PrepareSubGraph(*exec_op_list), ENN_RET_SUCCESS);
    EXPECT_EQ(gpu_ud.ExecuteSubGraph(req_op_list), ENN_RET_SUCCESS);

    compare((float *)out_mem->va, golden_data, output_size, error_threshold);

    EXPECT_EQ(emm->DeleteMemory(in_mem), ENN_RET_SUCCESS);
    EXPECT_EQ(emm->DeleteMemory(out_mem), ENN_RET_SUCCESS);
    EXPECT_EQ(gpu_ud.CloseSubGraph(*op_list), ENN_RET_SUCCESS);
    EXPECT_EQ(gpu_ud.Deinitialize(), ENN_RET_SUCCESS);
};

TEST_F(ENN_GT_UNIT_TEST_GPU_UD, gpu_ud_test_reshape) {
    const PrecisionType precision = PrecisionType::FP16;
    // const DataType data_type = DataType::FLOAT;
    const float error_threshold = 1e-2;

    const Dim4 input_dim = {2, 3, 6, 5};
    const Dim4 output_dim = {2, 3, 6, 5};
    const size_t input_size = GetDimSize(input_dim);
    const size_t output_size = GetDimSize(output_dim);

    std::vector<int32_t> shape_data = {3, 10, -1};
    size_t shape_size = shape_data.size();
    Dim4 shape_dim = {static_cast<uint32_t>(shape_size), 1, 1, 1};

    std::shared_ptr<float> input = make_shared_array<float>(input_size);
    std::shared_ptr<float> output = make_shared_array<float>(output_size);
    GenerateRandom(input.get(), input_size, 0, 1);
    memset(output.get(), 0, output_size * sizeof(float));

    ReshapeGuard reshape_guard;
    reshape_guard.GuardPrepare(input_size);
    reshape_guard.GuardRun(input.get(), output.get());

    enn::model::component::ParameterBuilder parameter_builder;
    std::shared_ptr<enn::model::component::Parameter> param_new_shape =
        parameter_builder.set_id(0)
            .set_name(std::string("NEW_SHAPE"))
            .set_data_type(TFlite::TensorType_INT32)
            .set_buffer_addr(shape_data.data())
            .set_buffer_size(shape_size)
            .set_shape(std::vector<uint32_t>{shape_dim.n, shape_dim.c, shape_dim.h, shape_dim.w})
            .create();

    enn::model::component::FeatureMapBuilder feature_map_builder;
    // 1. Make IFM
    enn::model::component::FeatureMap::Ptr ifm =
        feature_map_builder.set_id(0)
            .set_name(std::string("Tensor_000[0]"))
            .set_data_type(TFlite::TensorType_FLOAT32)
            .set_buffer_index(0)
            .set_buffer_size(input_size)
            .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_INPUT)
            .set_shape(std::vector<uint32_t>{input_dim.n, input_dim.c, input_dim.h, input_dim.w})
            .create();

    // 2. Make OFM
    enn::model::component::FeatureMap::Ptr ofm =
        feature_map_builder.set_id(1)
            .set_name(std::string("Tensor_000[1]"))
            .set_data_type(TFlite::TensorType_FLOAT32)
            .set_buffer_index(1)
            .set_buffer_size(output_size)
            .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_OUTPUT)
            .set_shape(std::vector<uint32_t>{output_dim.n, output_dim.c, output_dim.h, output_dim.w})
            .create();

    // 3. Make Options
    flatbuffers::FlatBufferBuilder fbb;
    fbb.Finish(TFlite::CreateReshapeOptions(fbb));
    void *base_addr = fbb.GetBufferPointer() +
                      (flatbuffers::EndianScalar(*reinterpret_cast<flatbuffers::uoffset_t *>(fbb.GetBufferPointer())));

    // 4. Make Operator
    const std::string op_name = EnumNameBuiltinOperator(TFlite::BuiltinOperator::BuiltinOperator_RESHAPE);
    enn::model::component::OperatorBuilder operator_builder;
    enn::model::component::Operator::Ptr op =
        operator_builder.set_id(op_name.length())
            .set_name(op_name)
            .set_code(TFlite::BuiltinOperator::BuiltinOperator_RESHAPE)
            .set_accelerator(model::Accelerator::GPU)
            .set_option(EnumNameBuiltinOptions(TFlite::BuiltinOptions::BuiltinOptions_ReshapeOptions),
                        base_addr,
                        sizeof(TFlite::ReshapeOptions),
                        TFlite::BuiltinOptions::BuiltinOptions_ReshapeOptions)
            .add_in_tensor(ifm)
            .add_in_tensor(param_new_shape)
            .add_out_tensor(ofm)
            .create();

    // 5. Make OpList
    auto attribute =
        std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_CAFFE, precision == PrecisionType::FP32 ? false : true);
    auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();

    ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();
    EXPECT_EQ(gpu_ud.OpenSubGraph(*op_list), ENN_RET_SUCCESS);

    auto in_mem = emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * input_size, enn::EnnMmType::kEnnMmTypeIon);
    auto out_mem = emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * output_size, enn::EnnMmType::kEnnMmTypeIon);
    CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem, out_mem}), "CreateMemory() failed");

    // Make BufferTable
    model::memory::BufferTable buffer_table;
    buffer_table.add(0, in_mem->va, in_mem->size);
    buffer_table.add(1, out_mem->va, out_mem->size);

    // Prepare & Execute
    auto exec_op_list = std::make_shared<runtime::ExecutableOperatorList>(
        MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    EXPECT_EQ(gpu_ud.PrepareSubGraph(*exec_op_list), ENN_RET_SUCCESS);

    memcpy(in_mem->va, input.get(), data_size[TFlite::TensorType_FLOAT32] * input_size);

    auto req_op_list = runtime::OperatorListExecuteRequest(exec_op_list);
    EXPECT_EQ(gpu_ud.ExecuteSubGraph(req_op_list), ENN_RET_SUCCESS);

    compare((float *)out_mem->va, output.get(), output_size, error_threshold);

    EXPECT_EQ(emm->DeleteMemory(in_mem), ENN_RET_SUCCESS);
    EXPECT_EQ(emm->DeleteMemory(out_mem), ENN_RET_SUCCESS);
    EXPECT_EQ(gpu_ud.CloseSubGraph(*op_list), ENN_RET_SUCCESS);
    EXPECT_EQ(gpu_ud.Deinitialize(), ENN_RET_SUCCESS);
};

TEST_F(ENN_GT_UNIT_TEST_GPU_UD, gpu_ud_test_strided_slice) {
    const PrecisionType precision = PrecisionType::FP16;
    // const DataType data_type = DataType::FLOAT;
    const float error_threshold = 1e-2;

    const Dim4 input_dim = {4, 1, 1, 1};
    const Dim4 output_dim = {2, 1, 1, 1};
    const size_t input_size = GetDimSize(input_dim);
    const size_t output_size = GetDimSize(output_dim);
    auto parameters = std::make_shared<ud::gpu::StridedSliceParameters>();
    parameters->begin.clear();
    parameters->end.clear();
    parameters->strides.clear();
    parameters->begin_mask = 0;
    parameters->end_mask = 0;
    parameters->shrink_axis_mask = 0;
    parameters->androidNN = false;

    float input_data[] = {1, 2, 3, 4};
    std::vector<int32_t> begin = {-3};
    std::vector<int32_t> end = {-5};
    std::vector<int32_t> strides = {-1};
    float golden_data[] = {2, 1};

    enn::model::component::ParameterBuilder parameter_builder;
    std::shared_ptr<enn::model::component::Parameter> param_begin =
        parameter_builder.set_name("Begin")
            .set_data_type(TFlite::TensorType_INT32)
            .set_buffer_addr(begin.data())
            .set_buffer_size(sizeof(int32_t) * begin.size())
            .set_shape(std::vector<uint32_t>{static_cast<uint32_t>(begin.size()), 1, 1, 1})
            .create();

    std::shared_ptr<enn::model::component::Parameter> param_end =
        parameter_builder.set_name("End")
            .set_data_type(TFlite::TensorType_INT32)
            .set_buffer_addr(end.data())
            .set_buffer_size(sizeof(int32_t) * end.size())
            .set_shape(std::vector<uint32_t>{static_cast<uint32_t>(end.size()), 1, 1, 1})
            .create();

    std::shared_ptr<enn::model::component::Parameter> param_strides =
        parameter_builder.set_name("Strides")
            .set_data_type(TFlite::TensorType_INT32)
            .set_buffer_addr(strides.data())
            .set_buffer_size(sizeof(int32_t) * strides.size())
            .set_shape(std::vector<uint32_t>{static_cast<uint32_t>(strides.size()), 1, 1, 1})
            .create();

    enn::model::component::FeatureMapBuilder feature_map_builder;
    // 1. Make IFM
    enn::model::component::FeatureMap::Ptr ifm =
        feature_map_builder.set_id(0)
            .set_name(std::string("Tensor_000[0]"))
            .set_data_type(TFlite::TensorType_FLOAT32)
            .set_buffer_index(0)
            .set_buffer_size(input_size)
            .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_INPUT)
            .set_shape(std::vector<uint32_t>{input_dim.n, input_dim.c, input_dim.h, input_dim.w})
            .create();

    // 2. Make OFM
    enn::model::component::FeatureMap::Ptr ofm =
        feature_map_builder.set_id(1)
            .set_name(std::string("Tensor_000[1]"))
            .set_data_type(TFlite::TensorType_FLOAT32)
            .set_buffer_index(1)
            .set_buffer_size(output_size)
            .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_OUTPUT)
            .set_shape(std::vector<uint32_t>{output_dim.n, output_dim.c, output_dim.h, output_dim.w})
            .create();

    // 3. Make Options
    flatbuffers::FlatBufferBuilder fbb;
    fbb.Finish(
        TFlite::CreateStridedSliceOptions(fbb, parameters->begin_mask, parameters->end_mask, parameters->ellipsis_mask));
    void *base_addr = fbb.GetBufferPointer() +
                      (flatbuffers::EndianScalar(*reinterpret_cast<flatbuffers::uoffset_t *>(fbb.GetBufferPointer())));

    // 4. Make Operator
    const std::string op_name = EnumNameBuiltinOperator(TFlite::BuiltinOperator::BuiltinOperator_STRIDED_SLICE);
    enn::model::component::OperatorBuilder operator_builder;
    enn::model::component::Operator::Ptr op =
        operator_builder.set_id(op_name.length())
            .set_name(op_name)
            .set_code(TFlite::BuiltinOperator::BuiltinOperator_STRIDED_SLICE)
            .set_accelerator(model::Accelerator::GPU)
            .set_option(EnumNameBuiltinOptions(TFlite::BuiltinOptions::BuiltinOptions_StridedSliceOptions),
                        base_addr,
                        sizeof(TFlite::StridedSliceOptions),
                        TFlite::BuiltinOptions::BuiltinOptions_StridedSliceOptions)
            .add_in_tensor(ifm)
            .add_in_tensor(param_begin)
            .add_in_tensor(param_end)
            .add_in_tensor(param_strides)
            .add_out_tensor(ofm)
            .create();

    // 5. Make OpList
    auto attribute =
        std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_CAFFE, precision == PrecisionType::FP32 ? false : true);
    auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();

    ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();
    EXPECT_EQ(gpu_ud.OpenSubGraph(*op_list), ENN_RET_SUCCESS);

    auto in_mem = emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * input_size, enn::EnnMmType::kEnnMmTypeIon);
    auto out_mem = emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * output_size, enn::EnnMmType::kEnnMmTypeIon);
    CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem, out_mem}), "CreateMemory() failed");

    // Make BufferTable
    model::memory::BufferTable buffer_table;
    buffer_table.add(0, in_mem->va, in_mem->size);
    buffer_table.add(1, out_mem->va, out_mem->size);

    // Prepare & Execute
    auto exec_op_list = std::make_shared<runtime::ExecutableOperatorList>(
        MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    EXPECT_EQ(gpu_ud.PrepareSubGraph(*exec_op_list), ENN_RET_SUCCESS);

    memcpy(in_mem->va, input_data, data_size[TFlite::TensorType_FLOAT32] * input_size);

    auto req_op_list = runtime::OperatorListExecuteRequest(exec_op_list);
    EXPECT_EQ(gpu_ud.ExecuteSubGraph(req_op_list), ENN_RET_SUCCESS);

    compare((float *)out_mem->va, golden_data, output_size, error_threshold);

    EXPECT_EQ(emm->DeleteMemory(in_mem), ENN_RET_SUCCESS);
    EXPECT_EQ(emm->DeleteMemory(out_mem), ENN_RET_SUCCESS);
    EXPECT_EQ(gpu_ud.CloseSubGraph(*op_list), ENN_RET_SUCCESS);
    EXPECT_EQ(gpu_ud.Deinitialize(), ENN_RET_SUCCESS);
};

TEST_F(ENN_GT_UNIT_TEST_GPU_UD, gpu_ud_test_transpose) {
    const PrecisionType precision = PrecisionType::FP16;
    // const DataType data_type = DataType::FLOAT;
    const float error_threshold = 1e-2;

    const Dim4 input_dim = {3, 2, 1, 1};
    const Dim4 output_dim = {3, 2, 1, 1};
    const size_t input_size = GetDimSize(input_dim);
    const size_t output_size = GetDimSize(output_dim);

    float input_data[6] = {0, 1, 2, 3, 4, 5};
    std::vector<int32_t> perm = {0, 1};
    float golden_data[6] = {0, 1, 2, 3, 4, 5};

    enn::model::component::ParameterBuilder parameter_builder;
    std::shared_ptr<enn::model::component::Parameter> param_perm =
        parameter_builder.set_name("Perm")
            .set_data_type(TFlite::TensorType_INT32)
            .set_buffer_addr(perm.data())
            .set_buffer_size(sizeof(int32_t) * perm.size())
            .set_shape(std::vector<uint32_t>{static_cast<uint32_t>(perm.size()), 1, 1, 1})
            .create();

    enn::model::component::FeatureMapBuilder feature_map_builder;
    // 1. Make IFM
    enn::model::component::FeatureMap::Ptr ifm =
        feature_map_builder.set_id(0)
            .set_name(std::string("Tensor_000[0]"))
            .set_data_type(TFlite::TensorType_FLOAT32)
            .set_buffer_index(0)
            .set_buffer_size(input_size)
            .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_INPUT)
            .set_shape(std::vector<uint32_t>{input_dim.n, input_dim.c, input_dim.h, input_dim.w})
            .create();

    // 2. Make OFM
    enn::model::component::FeatureMap::Ptr ofm =
        feature_map_builder.set_id(1)
            .set_name(std::string("Tensor_000[2]"))
            .set_data_type(TFlite::TensorType_FLOAT32)
            .set_buffer_index(1)
            .set_buffer_size(output_size)
            .set_type(enn::model::component::FeatureMap::Type::SUBGRAPH_OUTPUT)
            .set_shape(std::vector<uint32_t>{output_dim.n, output_dim.c, output_dim.h, output_dim.w})
            .create();

    // 3. Make Options
    flatbuffers::FlatBufferBuilder fbb;
    fbb.Finish(TFlite::CreateTransposeOptions(fbb));
    void *base_addr = fbb.GetBufferPointer() +
                      (flatbuffers::EndianScalar(*reinterpret_cast<flatbuffers::uoffset_t *>(fbb.GetBufferPointer())));

    // 4. Make Operator
    const std::string op_name = EnumNameBuiltinOperator(TFlite::BuiltinOperator::BuiltinOperator_TRANSPOSE);
    enn::model::component::OperatorBuilder operator_builder;
    enn::model::component::Operator::Ptr op =
        operator_builder.set_id(op_name.length())
            .set_name(op_name)
            .set_code(TFlite::BuiltinOperator::BuiltinOperator_TRANSPOSE)
            .set_accelerator(model::Accelerator::GPU)
            .set_option(EnumNameBuiltinOptions(TFlite::BuiltinOptions::BuiltinOptions_TransposeOptions),
                        base_addr,
                        sizeof(TFlite::TransposeOptions),
                        TFlite::BuiltinOptions::BuiltinOptions_TransposeOptions)
            .add_in_tensor(ifm)
            .add_in_tensor(param_perm)
            .add_out_tensor(ofm)
            .create();

    // 5. Make OpList
    auto attribute =
        std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_CAFFE, precision == PrecisionType::FP32 ? false : true);
    auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();

    ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();
    EXPECT_EQ(gpu_ud.OpenSubGraph(*op_list), ENN_RET_SUCCESS);

    auto in_mem = emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * input_size, enn::EnnMmType::kEnnMmTypeIon);
    auto out_mem = emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * output_size, enn::EnnMmType::kEnnMmTypeIon);
    CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem, out_mem}), "CreateMemory() failed");

    // Make BufferTable
    model::memory::BufferTable buffer_table;
    buffer_table.add(0, in_mem->va, in_mem->size);
    buffer_table.add(1, out_mem->va, out_mem->size);

    // Prepare & Execute
    auto exec_op_list = std::make_shared<runtime::ExecutableOperatorList>(
        MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    EXPECT_EQ(gpu_ud.PrepareSubGraph(*exec_op_list), ENN_RET_SUCCESS);

    memcpy(in_mem->va, input_data, data_size[TFlite::TensorType_FLOAT32] * input_size);

    auto req_op_list = runtime::OperatorListExecuteRequest(exec_op_list);
    EXPECT_EQ(gpu_ud.ExecuteSubGraph(req_op_list), ENN_RET_SUCCESS);

    compare((float *)out_mem->va, golden_data, output_size, error_threshold);

    EXPECT_EQ(emm->DeleteMemory(in_mem), ENN_RET_SUCCESS);
    EXPECT_EQ(emm->DeleteMemory(out_mem), ENN_RET_SUCCESS);
    EXPECT_EQ(gpu_ud.CloseSubGraph(*op_list), ENN_RET_SUCCESS);
    EXPECT_EQ(gpu_ud.Deinitialize(), ENN_RET_SUCCESS);
};

TEST_F(ENN_GT_UNIT_TEST_GPU_UD, gpu_ud_test_pad) {
    const PrecisionType precision = PrecisionType::FP16;
    // const DataType data_type = DataType::FLOAT;
    const float error_threshold = 1e-2;

    const Dim4 input_dim = {1, 1, 2, 2};
    const Dim4 output_dim = {1, 1, 4, 4};
    const size_t input_size = GetDimSize(input_dim);
    const size_t output_size = GetDimSize(output_dim);
    auto parameters = std::make_shared<ud::gpu::PadParameters>();
    parameters->padding.clear();
    parameters->pad_value = 0;

    float input_data[] = {1, 2, 3, 4};
    std::vector<int32_t> padding = {0, 0, 0, 0, 1, 1, 1, 1};
    float golden_data[] = {0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0};

    enn::model::component::ParameterBuilder parameter_builder;
    auto param_padding = create_parameter(parameter_builder,
                                          const_cast<char *>("Padding"),
                                          padding.data(),
                                          std::vector<uint32_t>{static_cast<uint32_t>(padding.size()), 1, 1, 1},
                                          TFlite::TensorType_INT32);

    auto param_pad_value = create_parameter(parameter_builder,
                                            const_cast<char *>("PaddingValue"),
                                            &(parameters->pad_value),
                                            std::vector<uint32_t>{1, 1, 1, 1},
                                            TFlite::TensorType_INT32);

    enn::model::component::FeatureMapBuilder feature_map_builder;
    // 1. Make IFM
    auto ifm = create_feature_map(feature_map_builder,
                                  std::string("Tensor_000[0]"),
                                  0,
                                  std::vector<uint32_t>{input_dim.n, input_dim.c, input_dim.h, input_dim.w},
                                  TFlite::TensorType_FLOAT32, true);

    // 2. Make OFM
    auto ofm = create_feature_map(feature_map_builder,
                                  std::string("Tensor_000[1]"),
                                  1,
                                  std::vector<uint32_t>{output_dim.n, output_dim.c, output_dim.h, output_dim.w},
                                  TFlite::TensorType_FLOAT32, false);

    // 3. Make Options
    flatbuffers::FlatBufferBuilder fbb;
    fbb.Finish(TFlite::CreatePadOptions(fbb));
    void *base_addr = fbb.GetBufferPointer() +
                      (flatbuffers::EndianScalar(*reinterpret_cast<flatbuffers::uoffset_t *>(fbb.GetBufferPointer())));

    // 4. Make Operator
    const std::string op_name = EnumNameBuiltinOperator(TFlite::BuiltinOperator::BuiltinOperator_PAD);
    enn::model::component::OperatorBuilder operator_builder;
    enn::model::component::Operator::Ptr op =
        operator_builder.set_id(op_name.length())
            .set_name(op_name)
            .set_code(TFlite::BuiltinOperator::BuiltinOperator_PAD)
            .set_accelerator(model::Accelerator::GPU)
            .set_option(EnumNameBuiltinOptions(TFlite::BuiltinOptions::BuiltinOptions_PadOptions),
                        base_addr,
                        sizeof(TFlite::PadOptions),
                        TFlite::BuiltinOptions::BuiltinOptions_PadOptions)
            .add_in_tensor(ifm)
            .add_in_tensor(param_padding)
            .add_in_tensor(param_pad_value)
            .add_out_tensor(ofm)
            .create();

    // 5. Make OpList
    auto attribute =
        std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_CAFFE, precision == PrecisionType::FP32 ? false : true);
    auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();

    ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();
    EXPECT_EQ(gpu_ud.OpenSubGraph(*op_list), ENN_RET_SUCCESS);

    auto in_mem = emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * input_size, enn::EnnMmType::kEnnMmTypeIon);
    auto out_mem = emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * output_size, enn::EnnMmType::kEnnMmTypeIon);
    CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem, out_mem}), "CreateMemory() failed");

    // Make BufferTable
    model::memory::BufferTable buffer_table;
    buffer_table.add(0, in_mem->va, in_mem->size);
    buffer_table.add(1, out_mem->va, out_mem->size);

    // Prepare & Execute
    auto exec_op_list = std::make_shared<runtime::ExecutableOperatorList>(
        MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    EXPECT_EQ(gpu_ud.PrepareSubGraph(*exec_op_list), ENN_RET_SUCCESS);

    memcpy(in_mem->va, input_data, data_size[TFlite::TensorType_FLOAT32] * input_size);

    auto req_op_list = runtime::OperatorListExecuteRequest(exec_op_list);
    EXPECT_EQ(gpu_ud.ExecuteSubGraph(req_op_list), ENN_RET_SUCCESS);

    compare((float *)out_mem->va, golden_data, output_size, error_threshold);

    EXPECT_EQ(emm->DeleteMemory(in_mem), ENN_RET_SUCCESS);
    EXPECT_EQ(emm->DeleteMemory(out_mem), ENN_RET_SUCCESS);
    EXPECT_EQ(gpu_ud.CloseSubGraph(*op_list), ENN_RET_SUCCESS);
    EXPECT_EQ(gpu_ud.Deinitialize(), ENN_RET_SUCCESS);
};

TEST_F(ENN_GT_UNIT_TEST_GPU_UD, gpu_ud_test_cast) {
    const PrecisionType precision = PrecisionType::FP16;
    // const DataType data_type = DataType::FLOAT;
    const float error_threshold = 1e-2;

    const Dim4 input_dim = {2, 3, 1, 1};
    const Dim4 output_dim = {2, 3, 1, 1};
    const size_t input_size = GetDimSize(input_dim);
    const size_t output_size = GetDimSize(output_dim);
    auto parameters = std::make_shared<ud::gpu::CastParameters>();
    parameters->in_data_type = DataType::INT32;
    parameters->out_data_type = DataType::FLOAT;

    int32_t input_data[] = {100, 200, 300, 400, 500, 600};
    float golden_data[] = {100.f, 200.f, 300.f, 400.f, 500.f, 600.f};

    enn::model::component::FeatureMapBuilder feature_map_builder;
    // 1. Make IFM
    auto ifm = create_feature_map(feature_map_builder,
                                  std::string("Tensor_000[0]"),
                                  0,
                                  std::vector<uint32_t>{input_dim.n, input_dim.c, input_dim.h, input_dim.w},
                                  TFlite::TensorType_INT32, true);

    // 2. Make OFM
    auto ofm = create_feature_map(feature_map_builder,
                                  std::string("Tensor_000[1]"),
                                  1,
                                  std::vector<uint32_t>{output_dim.n, output_dim.c, output_dim.h, output_dim.w},
                                  TFlite::TensorType_FLOAT32, false);

    // 3. Make Options
    flatbuffers::FlatBufferBuilder fbb;
    fbb.Finish(TFlite::CreateCastOptions(fbb, TFlite::TensorType_INT32, TFlite::TensorType_FLOAT32));
    void *base_addr = fbb.GetBufferPointer() +
                      (flatbuffers::EndianScalar(*reinterpret_cast<flatbuffers::uoffset_t *>(fbb.GetBufferPointer())));

    // 4. Make Operator
    const std::string op_name = EnumNameBuiltinOperator(TFlite::BuiltinOperator::BuiltinOperator_CAST);
    enn::model::component::OperatorBuilder operator_builder;
    enn::model::component::Operator::Ptr op =
        operator_builder.set_id(op_name.length())
            .set_name(op_name)
            .set_code(TFlite::BuiltinOperator::BuiltinOperator_CAST)
            .set_accelerator(model::Accelerator::GPU)
            .set_option(EnumNameBuiltinOptions(TFlite::BuiltinOptions::BuiltinOptions_CastOptions),
                        base_addr,
                        sizeof(TFlite::CastOptions),
                        TFlite::BuiltinOptions::BuiltinOptions_CastOptions)
            .add_in_tensor(ifm)
            .add_out_tensor(ofm)
            .create();

    // 5. Make OpList
    auto attribute =
        std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_CAFFE, precision == PrecisionType::FP32 ? false : true);
    auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();

    ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();
    EXPECT_EQ(gpu_ud.OpenSubGraph(*op_list), ENN_RET_SUCCESS);

    auto in_mem = emm->CreateMemory(data_size[TFlite::TensorType_INT32] * input_size, enn::EnnMmType::kEnnMmTypeIon);
    auto out_mem = emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * output_size, enn::EnnMmType::kEnnMmTypeIon);
    CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem, out_mem}), "CreateMemory() failed");

    // Make BufferTable

    model::memory::BufferTable buffer_table;
    buffer_table.add(0, in_mem->va, in_mem->size);
    buffer_table.add(1, out_mem->va, out_mem->size);

    // Prepare & Execute
    auto exec_op_list = std::make_shared<runtime::ExecutableOperatorList>(
        MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    EXPECT_EQ(gpu_ud.PrepareSubGraph(*exec_op_list), ENN_RET_SUCCESS);

    memcpy(in_mem->va, input_data, data_size[TFlite::TensorType_INT32] * input_size);

    auto req_op_list = runtime::OperatorListExecuteRequest(exec_op_list);
    EXPECT_EQ(gpu_ud.ExecuteSubGraph(req_op_list), ENN_RET_SUCCESS);

    compare((float *)out_mem->va, golden_data, output_size, error_threshold);

    EXPECT_EQ(emm->DeleteMemory(in_mem), ENN_RET_SUCCESS);
    EXPECT_EQ(emm->DeleteMemory(out_mem), ENN_RET_SUCCESS);
    EXPECT_EQ(gpu_ud.CloseSubGraph(*op_list), ENN_RET_SUCCESS);
    EXPECT_EQ(gpu_ud.Deinitialize(), ENN_RET_SUCCESS);
};

TEST_F(ENN_GT_UNIT_TEST_GPU_UD, gpu_ud_test_unpack) {
    const PrecisionType precision = PrecisionType::FP16;
    // const DataType data_type = DataType::FLOAT;
    const float error_threshold = 1e-2;

    const NDims input_dim = {3, 2};
    const NDims output_dim = {2};
    const size_t input_size = getTensorSizeFromDims(input_dim);
    const size_t output_size = getTensorSizeFromDims(output_dim);
    auto parameters = std::make_shared<ud::gpu::UnpackParameters>();
    parameters->axis = 0;
    parameters->num = 3;

    float input_data[6] = {1, 2, 3, 4, 5, 6};
    float output1[2] = {1, 2};
    float output2[2] = {3, 4};
    float output3[2] = {5, 6};
    std::vector<float *> golden_data = {output1, output2, output3};

    enn::model::component::FeatureMapBuilder feature_map_builder;
    // 1. Make IFM
    auto ifm = create_feature_map(feature_map_builder,
                                  std::string("Tensor_000[0]"),
                                  0,
                                  std::vector<uint32_t>{input_dim.begin(), input_dim.end()},
                                  TFlite::TensorType_FLOAT32, true);

    // 2. Make OFM
    std::vector<enn::model::component::FeatureMap::Ptr> ofms;
    for (int32_t i = 0; i < parameters->num; ++i) {
        std::string name = std::string("Tensor_000[") + std::to_string(i + 1) + std::string("]");
        auto ofm = create_feature_map(feature_map_builder,
                                      name,
                                      i + 1,
                                      std::vector<uint32_t>{output_dim.begin(), output_dim.end()},
                                      TFlite::TensorType_FLOAT32, false);
        ofms.push_back(ofm);
    }

    // 3. Make Options
    flatbuffers::FlatBufferBuilder fbb;
    fbb.Finish(TFlite::CreateUnpackOptions(fbb, parameters->num, parameters->axis));
    void *base_addr = fbb.GetBufferPointer() +
                      (flatbuffers::EndianScalar(*reinterpret_cast<flatbuffers::uoffset_t *>(fbb.GetBufferPointer())));

    // 4. Make Operator
    const std::string op_name = EnumNameBuiltinOperator(TFlite::BuiltinOperator::BuiltinOperator_UNPACK);
    enn::model::component::OperatorBuilder operator_builder;
    operator_builder.set_id(op_name.length())
        .set_name(op_name)
        .set_code(TFlite::BuiltinOperator::BuiltinOperator_UNPACK)
        .set_accelerator(model::Accelerator::GPU)
        .set_option(EnumNameBuiltinOptions(TFlite::BuiltinOptions::BuiltinOptions_TransposeOptions),
                    base_addr,
                    sizeof(TFlite::TransposeOptions),
                    TFlite::BuiltinOptions::BuiltinOptions_TransposeOptions)
        .add_in_tensor(ifm);
    for (auto ofm : ofms) {
        operator_builder.add_out_tensor(ofm);
    }
    enn::model::component::Operator::Ptr op = operator_builder.create();

    // 5. Make OpList
    auto attribute =
        std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_CAFFE, precision == PrecisionType::FP32 ? false : true);
    auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();

    ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();
    EXPECT_EQ(gpu_ud.OpenSubGraph(*op_list), ENN_RET_SUCCESS);

    std::vector<EnnBufferCore::Ptr> mems;
    auto in_mem = emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * input_size, enn::EnnMmType::kEnnMmTypeIon);
    mems.push_back(in_mem);
    for (int32_t i = 0; i < parameters->num; ++i) {
        auto out_mem = emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * output_size, enn::EnnMmType::kEnnMmTypeIon);
        mems.push_back(out_mem);
    }
    CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, mems), "CreateMemory() failed");

    // Make BufferTable
    model::memory::BufferTable buffer_table;
    buffer_table.add(0, mems[0]->va, mems[0]->size);
    for (int32_t i = 0; i < parameters->num; ++i) {
        buffer_table.add(i + 1, mems[i + 1]->va, mems[i + 1]->size);
    }

    // Prepare & Execute
    auto exec_op_list = std::make_shared<runtime::ExecutableOperatorList>(
        MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    EXPECT_EQ(gpu_ud.PrepareSubGraph(*exec_op_list), ENN_RET_SUCCESS);

    memcpy(mems[0]->va, input_data, data_size[TFlite::TensorType_FLOAT32] * input_size);

    auto req_op_list = runtime::OperatorListExecuteRequest(exec_op_list);
    EXPECT_EQ(gpu_ud.ExecuteSubGraph(req_op_list), ENN_RET_SUCCESS);

    for (int32_t i = 0; i < parameters->num; ++i) {
        compare((float *)mems[i + 1]->va, golden_data[i], output_size, error_threshold);
    }

    for (int32_t i = 0; i < parameters->num; ++i) {
        EXPECT_EQ(emm->DeleteMemory(mems[i]), ENN_RET_SUCCESS);
    }
    EXPECT_EQ(gpu_ud.CloseSubGraph(*op_list), ENN_RET_SUCCESS);
    EXPECT_EQ(gpu_ud.Deinitialize(), ENN_RET_SUCCESS);
};

TEST_F(ENN_GT_UNIT_TEST_GPU_UD, gpu_ud_test_tfslice) {
    const PrecisionType precision = PrecisionType::FP16;
    // const DataType data_type = DataType::FLOAT;
    const float error_threshold = 1e-2;

    const NDims input_dim = {4, 1, 1, 1};
    const NDims output_dim = {3, 1, 1, 1};
    const size_t input_size = getTensorSizeFromDims(input_dim);
    const size_t output_size = getTensorSizeFromDims(output_dim);
    auto parameters = std::make_shared<ud::gpu::TFSliceParameters>();
    parameters->begin.clear();
    parameters->size.clear();
    parameters->androidNN = false;

    float input_data[] = {1, 2, 3, 4};
    float golden_data[] = {2, 3, 4};
    std::vector<int32_t> begin = {1, 0, 0, 0};
    std::vector<int32_t> size = {3, 1, 1, 1};

    enn::model::component::ParameterBuilder parameter_builder;
    auto param_begin = create_parameter(parameter_builder,
                                        const_cast<char *>("Begin"),
                                        begin.data(),
                                        std::vector<uint32_t>{static_cast<uint32_t>(begin.size()), 1, 1, 1},
                                        TFlite::TensorType_INT32);

    auto param_size = create_parameter(parameter_builder,
                                       const_cast<char *>("Size"),
                                       size.data(),
                                       std::vector<uint32_t>{static_cast<uint32_t>(size.size()), 1, 1, 1},
                                       TFlite::TensorType_INT32);

    enn::model::component::FeatureMapBuilder feature_map_builder;
    // 1. Make IFM
    auto ifm = create_feature_map(feature_map_builder,
                                  std::string("Tensor_000[0]"),
                                  0,
                                  std::vector<uint32_t>{input_dim.begin(), input_dim.end()},
                                  TFlite::TensorType_FLOAT32, true);

    // 2. Make OFM
    auto ofm = create_feature_map(feature_map_builder,
                                  std::string("Tensor_000[1]"),
                                  1,
                                  std::vector<uint32_t>{output_dim.begin(), output_dim.end()},
                                  TFlite::TensorType_FLOAT32, false);

    // 3. Make Options
    flatbuffers::FlatBufferBuilder fbb;
#ifdef SCHEMA_NNC_V1
    fbb.Finish(TFlite::CreateTFliteSliceOptions(fbb));
#else
    fbb.Finish(TFlite::CreateENN_TFliteSliceOptions(fbb));
#endif
    void *base_addr = fbb.GetBufferPointer() +
                      (flatbuffers::EndianScalar(*reinterpret_cast<flatbuffers::uoffset_t *>(fbb.GetBufferPointer())));

// 4. Make Operator
#ifdef SCHEMA_NNC_V1
    const std::string op_name = EnumNameBuiltinOperator(TFlite::BuiltinOperator::BuiltinOperator_TFLITE_SLICE);
#else
    const std::string op_name = EnumNameBuiltinOperator(TFlite::BuiltinOperator::BuiltinOperator_ENN_TFLITE_SLICE);
#endif
    enn::model::component::OperatorBuilder operator_builder;
    enn::model::component::Operator::Ptr op =
        operator_builder.set_id(op_name.length())
            .set_name(op_name)
#ifdef SCHEMA_NNC_V1
            .set_code(TFlite::BuiltinOperator::BuiltinOperator_TFLITE_SLICE)
            .set_option(EnumNameBuiltinOptions(TFlite::BuiltinOptions::BuiltinOptions_TFliteSliceOptions),
                        base_addr,
                        sizeof(TFlite::TFliteSliceOptions),
                        TFlite::BuiltinOptions::BuiltinOptions_TFliteSliceOptions)
#else
            .set_code(TFlite::BuiltinOperator::BuiltinOperator_ENN_TFLITE_SLICE)
            .set_option(EnumNameBuiltinOptions(TFlite::BuiltinOptions::BuiltinOptions_ENN_TFliteSliceOptions),
                        base_addr,
                        sizeof(TFlite::ENN_TFliteSliceOptions),
                        TFlite::BuiltinOptions::BuiltinOptions_ENN_TFliteSliceOptions)
#endif
            .set_accelerator(model::Accelerator::GPU)
            .add_in_tensor(ifm)
            .add_in_tensor(param_begin)
            .add_in_tensor(param_size)
            .add_out_tensor(ofm)
            .create();

    // 5. Make OpList
    auto attribute =
        std::make_shared<enn::model::Attribute>(TFlite::LegacyModel_CAFFE, precision == PrecisionType::FP32 ? false : true);
    auto op_list = operator_list_builder.build(MODEL_ID).add_operator(op).set_attribute(attribute).create();

    ud::gpu::GpuUserDriver &gpu_ud = create_gpu_ud();
    EXPECT_EQ(gpu_ud.OpenSubGraph(*op_list), ENN_RET_SUCCESS);

    auto in_mem = emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * input_size, enn::EnnMmType::kEnnMmTypeIon);
    auto out_mem = emm->CreateMemory(data_size[TFlite::TensorType_FLOAT32] * output_size, enn::EnnMmType::kEnnMmTypeIon);
    CHECK_AND_RETURN_VOID(is_invalid_memory(gpu_ud, *op_list, {in_mem, out_mem}), "CreateMemory() failed");

    // Make BufferTable
    model::memory::BufferTable buffer_table;
    buffer_table.add(0, in_mem->va, in_mem->size);
    buffer_table.add(1, out_mem->va, out_mem->size);

    // Prepare & Execute
    auto exec_op_list = std::make_shared<runtime::ExecutableOperatorList>(
        MODEL_ID, op_list, std::make_shared<model::memory::BufferTable>(buffer_table));
    EXPECT_EQ(gpu_ud.PrepareSubGraph(*exec_op_list), ENN_RET_SUCCESS);

    memcpy(in_mem->va, input_data, data_size[TFlite::TensorType_FLOAT32] * input_size);

    auto req_op_list = runtime::OperatorListExecuteRequest(exec_op_list);
    EXPECT_EQ(gpu_ud.ExecuteSubGraph(req_op_list), ENN_RET_SUCCESS);

    compare((float *)out_mem->va, golden_data, output_size, error_threshold);

    EXPECT_EQ(emm->DeleteMemory(in_mem), ENN_RET_SUCCESS);
    EXPECT_EQ(emm->DeleteMemory(out_mem), ENN_RET_SUCCESS);
    EXPECT_EQ(gpu_ud.CloseSubGraph(*op_list), ENN_RET_SUCCESS);
    EXPECT_EQ(gpu_ud.Deinitialize(), ENN_RET_SUCCESS);
};

}  // namespace internal
}  // namespace test
}  // namespace enn
