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

#include "userdriver/gpu/gpu_op_constructor.h"
#include "common/enn_debug.h"
#include "common/enn_utils.h"
#include "common/helper_templates.hpp"
#include "model/component/tensor/feature_map.hpp"
#include "model/component/tensor/parameter.hpp"
#include "userdriver/common/op_test/test_capabilities.h"
#include "userdriver/common/operator_interfaces/common/ActivationInfo.hpp"
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/common/Debug.hpp"

namespace enn {
namespace ud {
namespace gpu {

/***************************************************************************************************************************
 * create_ud_operator()
 ***************************************************************************************************************************/
#ifdef SCHEMA_NNC_V1
template <>
EnnReturn OperationConstructor::create_ud_operator<CustomOperator::CustomOperator_Normalization>(
    const model::component::Operator::Ptr &operator_) {
#else
template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_ENN_NORMALIZATION>(
    const model::component::Operator::Ptr &operator_) {
#endif
    std::shared_ptr<NormalizationrParameters> parameters;
    std::vector<std::shared_ptr<ITensor>> in_tensors, out_tensors, data_tensors;
    parameters.reset(new NormalizationrParameters);
    parameters->use_FP32_input_for_fp16 = false;

    CHECK_AND_RETURN_ERR(0 >= operator_->out_tensors.count(), ENN_RET_FAILED, "1<= operator_->out_tensors.count()\n");
    PrecisionType precision = PrecisionType::FP16;
    model::component::FeatureMap::Ptr ofm =
        std::static_pointer_cast<model::component::FeatureMap>(operator_->out_tensors[0]);
    precision = get_precision((TFlite::TensorType)ofm->get_data_type());

    convert_to_tensors(operator_, precision, in_tensors, out_tensors);

    std::shared_ptr<CLNormalization> normalization = compute_library_->createNormalization(precision);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != normalization->initialize(in_tensors, out_tensors, parameters),
                         ENN_RET_FAILED,
                         "CLNormalization initialize is failed\n");

    operators->push_back(std::make_shared<EnnUDOperator<CLNormalization>>(
        operator_->get_name(), operator_->get_id(), in_tensors, out_tensors, data_tensors, normalization));

    return ENN_RET_SUCCESS;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_CONV_2D>(
    const model::component::Operator::Ptr &operator_) {
    ENN_DBG_PRINT("TFlite::BuiltinOperator::BuiltinOperator_CONV_2D opr->code = %d is called\n", operator_->get_code());
    std::string builtin_op_name = TFlite::EnumNameBuiltinOperator(operator_->get_code());
    CHECK_AND_RETURN_ERR(0 != builtin_op_name.compare(operator_->get_name()),
                         ENN_RET_FAILED,
                         "Invalid BuiltinOperator : %s\n",
                         builtin_op_name.c_str());

    std::shared_ptr<ConvolutionParameters> parameters = std::make_shared<ConvolutionParameters>();
    std::vector<std::shared_ptr<ITensor>> in_tensors, out_tensors, data_tensors;

    CHECK_AND_RETURN_ERR(0 >= operator_->out_tensors.count(), ENN_RET_FAILED, "1<= operator_->out_tensors.count()\n");
    PrecisionType precision = get_precision((TFlite::TensorType)operator_->out_tensors[0]->get_data_type());

    convert_to_tensors(operator_, precision, in_tensors, out_tensors);

    CHECK_AND_RETURN_ERR(
        nullptr == operator_->get_option().get_addr(), ENN_RET_FAILED, "operator_->get_option().get_addr() == nullptr\n");

    const TFlite::Conv2DOptions *tflOptions =
        reinterpret_cast<const TFlite::Conv2DOptions *>(operator_->get_option().get_addr());
    parameters->padding = {0, 0, 0, 0};
    parameters->dilation = {static_cast<uint32_t>(tflOptions->dilation_h_factor()),
                            static_cast<uint32_t>(tflOptions->dilation_w_factor())};
    parameters->stride = {static_cast<uint32_t>(tflOptions->stride_h()), static_cast<uint32_t>(tflOptions->stride_w())};
    parameters->activation_info = std::make_shared<ActivationInfo>(
        static_cast<ActivationInfo::ActivationType>(tflOptions->fused_activation_function()),
        tflOptions->fused_activation_function() != TFlite::ActivationFunctionType::ActivationFunctionType_NONE);
    parameters->group_size = 1;
    parameters->axis = 0;
    parameters->per_channel_quant = false;
    parameters->androidNN = false;
    parameters->isNCHW = is_nchw_layout(legacy_model_);

#ifndef SCHEMA_NNC_V1
    // TODO(xin.lu): parse the padding_value
    parameters->androidNN = legacy_model_ == TFlite::LegacyModel::LegacyModel_ANDROID_NN;
    if (parameters->androidNN)
        parameters->isNCHW = tflOptions->use_nchw();
#endif
    parameters->storage_type = StorageType::BUFFER;
    parameters->openAibWino = false;
    if (legacy_model_ == TFlite::LegacyModel_ANDROID_NN && tflOptions->padding_value()->size() == 4) {
        parameters->padding.l = static_cast<uint32_t>(tflOptions->padding_value()->data()[0]);
        parameters->padding.r = static_cast<uint32_t>(tflOptions->padding_value()->data()[1]);
        parameters->padding.t = static_cast<uint32_t>(tflOptions->padding_value()->data()[2]);
        parameters->padding.b = static_cast<uint32_t>(tflOptions->padding_value()->data()[3]);
    } else {
        get_padding(tflOptions->padding(),
                    parameters->padding,
                    in_tensors[0]->getDims(),
                    out_tensors[0]->getDims(),
                    in_tensors[1]->getDims(),
                    parameters->stride,
                    parameters->dilation,
                    false,
                    parameters->isNCHW);
    }

    get_perchannel_quant_info(operator_->in_tensors[1], parameters->per_channel_quant, parameters->scales);

    std::shared_ptr<CLConvolution> conv = compute_library_->createConvolution(precision);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != conv->initialize(in_tensors, out_tensors, parameters),
                         ENN_RET_FAILED,
                         "CLConvolution initialize is failed\n");

    operators->push_back(std::make_shared<EnnUDOperator<CLConvolution>>(
        operator_->get_name(), operator_->get_id(), in_tensors, out_tensors, data_tensors, conv));

    return ENN_RET_SUCCESS;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_DEPTH_TO_SPACE>(
    const model::component::Operator::Ptr &operator_) {
    ENN_DBG_PRINT("TFlite::BuiltinOperator::BuiltinOperator_DEPTH_TO_SPACE opr->code = %d is called\n",
                  operator_->get_code());
    std::string builtin_op_name = TFlite::EnumNameBuiltinOperator(operator_->get_code());
    CHECK_AND_RETURN_ERR(0 != builtin_op_name.compare(operator_->get_name()),
                         ENN_RET_FAILED,
                         "Invalid BuiltinOperator : %s\n",
                         builtin_op_name.c_str());

    std::shared_ptr<Depth2SpaceParameters> parameters = std::make_shared<Depth2SpaceParameters>();
    std::vector<std::shared_ptr<ITensor>> in_tensors, out_tensors, data_tensors;

    CHECK_AND_RETURN_ERR(0 >= operator_->out_tensors.count(), ENN_RET_FAILED, "1<= operator_->out_tensors.count()\n");
    PrecisionType precision = PrecisionType::FP16;
    model::component::FeatureMap::Ptr ofm =
        std::static_pointer_cast<model::component::FeatureMap>(operator_->out_tensors[0]);
    precision = get_precision((TFlite::TensorType)ofm->get_data_type());

    convert_to_tensors(operator_, precision, in_tensors, out_tensors);

    CHECK_AND_RETURN_ERR(
        nullptr == operator_->get_option().get_addr(), ENN_RET_FAILED, "operator_->get_option().get_addr() == nullptr\n");

    const TFlite::DepthToSpaceOptions *tflOptions =
        reinterpret_cast<const TFlite::DepthToSpaceOptions *>(operator_->get_option().get_addr());
    parameters->block_size = tflOptions->block_size();
    parameters->androidNN = false;
    parameters->isNCHW = is_nchw_layout(legacy_model_);

#ifndef SCHEMA_NNC_V1
    parameters->androidNN = legacy_model_ == TFlite::LegacyModel::LegacyModel_ANDROID_NN;
    if (parameters->androidNN)
        parameters->isNCHW = tflOptions->use_nchw();
#endif

    std::shared_ptr<CLDepth2Space> depth2space = compute_library_->createDepth2Space(precision);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != depth2space->initialize(in_tensors, out_tensors, parameters),
                         ENN_RET_FAILED,
                         "CLDepth2Space initialize is failed\n");

    operators->push_back(std::make_shared<EnnUDOperator<CLDepth2Space>>(
        operator_->get_name(), operator_->get_id(), in_tensors, out_tensors, data_tensors, depth2space));

    return ENN_RET_SUCCESS;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_DEPTHWISE_CONV_2D>(
    const model::component::Operator::Ptr &operator_) {
    ENN_DBG_PRINT("TFlite::BuiltinOperator::BuiltinOperator_DEPTHWISE_CONV_2D opr->code = %d is called\n",
                  operator_->get_code());
    std::string builtin_op_name = TFlite::EnumNameBuiltinOperator(operator_->get_code());
    CHECK_AND_RETURN_ERR(0 != builtin_op_name.compare(operator_->get_name()),
                         ENN_RET_FAILED,
                         "Invalid BuiltinOperator : %s\n",
                         builtin_op_name.c_str());

    std::shared_ptr<DepthwiseConvolutionParameters> parameters = std::make_shared<DepthwiseConvolutionParameters>();
    std::vector<std::shared_ptr<ITensor>> in_tensors, out_tensors, data_tensors;

    CHECK_AND_RETURN_ERR(0 >= operator_->out_tensors.count(), ENN_RET_FAILED, "1<= operator_->out_tensors.count()\n");
    PrecisionType precision = PrecisionType::FP16;
    model::component::FeatureMap::Ptr ofm =
        std::static_pointer_cast<model::component::FeatureMap>(operator_->out_tensors[0]);
    precision = get_precision((TFlite::TensorType)ofm->get_data_type());

    convert_to_tensors(operator_, precision, in_tensors, out_tensors);

    CHECK_AND_RETURN_ERR(
        nullptr == operator_->get_option().get_addr(), ENN_RET_FAILED, "operator_->get_option().get_addr() == nullptr\n");

    const TFlite::DepthwiseConv2DOptions *tflOptions =
        reinterpret_cast<const TFlite::DepthwiseConv2DOptions *>(operator_->get_option().get_addr());
    parameters->padding = {0, 0, 0, 0};
    parameters->dilation = {static_cast<uint32_t>(tflOptions->dilation_h_factor()),
                            static_cast<uint32_t>(tflOptions->dilation_w_factor())};
    parameters->stride = {static_cast<uint32_t>(tflOptions->stride_h()), static_cast<uint32_t>(tflOptions->stride_w())};
    parameters->depth_multiplier = tflOptions->depth_multiplier();
    parameters->per_channel_quant = false;
    parameters->androidNN = false;
    parameters->isNCHW = is_nchw_layout(legacy_model_);

#ifndef SCHEMA_NNC_V1
    // TODO(xin.lu): parse the padding_value
    parameters->androidNN = legacy_model_ == TFlite::LegacyModel::LegacyModel_ANDROID_NN;
    if (parameters->androidNN)
        parameters->isNCHW = tflOptions->use_nchw();
#endif
    parameters->storage_type = StorageType::BUFFER;
    parameters->activation_info = std::make_shared<ActivationInfo>(
        static_cast<ActivationInfo::ActivationType>(tflOptions->fused_activation_function()),
        tflOptions->fused_activation_function() != TFlite::ActivationFunctionType::ActivationFunctionType_NONE);
    if (legacy_model_ == TFlite::LegacyModel_ANDROID_NN && tflOptions->padding_value()->size() == 4) {
        parameters->padding.l = static_cast<uint32_t>(tflOptions->padding_value()->data()[0]);
        parameters->padding.r = static_cast<uint32_t>(tflOptions->padding_value()->data()[1]);
        parameters->padding.t = static_cast<uint32_t>(tflOptions->padding_value()->data()[2]);
        parameters->padding.b = static_cast<uint32_t>(tflOptions->padding_value()->data()[3]);
    } else {
        get_padding(tflOptions->padding(),
                    parameters->padding,
                    in_tensors[0]->getDims(),
                    out_tensors[0]->getDims(),
                    in_tensors[1]->getDims(),
                    parameters->stride,
                    parameters->dilation,
                    false,
                    parameters->isNCHW);
    }

    get_perchannel_quant_info(operator_->in_tensors[1], parameters->per_channel_quant, parameters->scales);

    std::shared_ptr<CLDepthwiseConvolution> depthwise_conv = compute_library_->createDepthwiseConvolution(precision);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != depthwise_conv->initialize(in_tensors, out_tensors, parameters),
                         ENN_RET_FAILED,
                         "CLDepthwiseConvolution initialize is failed\n");

    operators->push_back(std::make_shared<EnnUDOperator<CLDepthwiseConvolution>>(
        operator_->get_name(), operator_->get_id(), in_tensors, out_tensors, data_tensors, depthwise_conv));

    return ENN_RET_SUCCESS;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_TRANSPOSE_CONV>(
    const model::component::Operator::Ptr &operator_) {
    ENN_DBG_PRINT("TFlite::BuiltinOperator::BuiltinOperator_TRANSPOSE_CONV opr->code = %d is called\n",
                  operator_->get_code());
    std::string builtin_op_name = TFlite::EnumNameBuiltinOperator(operator_->get_code());
    CHECK_AND_RETURN_ERR(0 != builtin_op_name.compare(operator_->get_name()),
                         ENN_RET_FAILED,
                         "Invalid BuiltinOperator : %s\n",
                         builtin_op_name.c_str());

    std::shared_ptr<DeconvolutionParameters> parameters = std::make_shared<DeconvolutionParameters>();
    std::vector<std::shared_ptr<ITensor>> in_tensors, out_tensors, data_tensors;

    CHECK_AND_RETURN_ERR(0 >= operator_->out_tensors.count(), ENN_RET_FAILED, "1<= operator_->out_tensors.count()\n");
    CHECK_AND_RETURN_ERR(3 != operator_->in_tensors.count(), ENN_RET_FAILED, "3 == operator_->in_tensors.count()\n");

    PrecisionType precision = PrecisionType::FP16;
    model::component::FeatureMap::Ptr ofm =
        std::static_pointer_cast<model::component::FeatureMap>(operator_->out_tensors[0]);
    precision = get_precision((TFlite::TensorType)ofm->get_data_type());

    convert_to_tensors(operator_, precision, in_tensors, out_tensors);

    CHECK_AND_RETURN_ERR(
        nullptr == operator_->get_option().get_addr(), ENN_RET_FAILED, "operator_->get_option().get_addr() == nullptr\n");

    const TFlite::TransposeConvOptions *tflOptions =
        reinterpret_cast<const TFlite::TransposeConvOptions *>(operator_->get_option().get_addr());
    parameters->padding = {0, 0, 0, 0};
    parameters->stride = {static_cast<uint32_t>(tflOptions->stride_h()), static_cast<uint32_t>(tflOptions->stride_w())};
    parameters->group_size = tflOptions->group();
    parameters->weights_as_input = !in_tensors[1]->is_const();
    parameters->per_channel_quant = false;
    parameters->androidNN = false;
    parameters->isNCHW = is_nchw_layout(legacy_model_);

#ifndef SCHEMA_NNC_V1
    parameters->androidNN = legacy_model_ == TFlite::LegacyModel::LegacyModel_ANDROID_NN;
    if (parameters->androidNN)
        parameters->isNCHW = tflOptions->use_nchw();
#endif
    parameters->openAibWino = false;
    parameters->activation_info = std::make_shared<ActivationInfo>(
        static_cast<ActivationInfo::ActivationType>(tflOptions->fused_activation_function()),
        tflOptions->fused_activation_function() != TFlite::ActivationFunctionType::ActivationFunctionType_NONE);
    if (legacy_model_ == TFlite::LegacyModel_ANDROID_NN && tflOptions->padding_value()->size() == 4) {
        parameters->padding.l = static_cast<uint32_t>(tflOptions->padding_value()->data()[0]);
        parameters->padding.r = static_cast<uint32_t>(tflOptions->padding_value()->data()[1]);
        parameters->padding.t = static_cast<uint32_t>(tflOptions->padding_value()->data()[2]);
        parameters->padding.b = static_cast<uint32_t>(tflOptions->padding_value()->data()[3]);
    } else {
        get_padding(tflOptions->padding(),
                    parameters->padding,
                    in_tensors[0]->getDims(),
                    out_tensors[0]->getDims(),
                    in_tensors[1]->getDims(),
                    parameters->stride,
                    {1, 1},
                    true,
                    parameters->isNCHW);
    }

    get_perchannel_quant_info(operator_->in_tensors[1], parameters->per_channel_quant, parameters->scales);

    std::shared_ptr<CLDeconvolution> de_conv = compute_library_->createDeconvolution(precision);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != de_conv->initialize(in_tensors, out_tensors, parameters),
                         ENN_RET_FAILED,
                         "CLDeconvolution initialize is failed\n");

    operators->push_back(std::make_shared<EnnUDOperator<CLDeconvolution>>(
        operator_->get_name(), operator_->get_id(), in_tensors, out_tensors, data_tensors, de_conv));

    return ENN_RET_SUCCESS;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_FULLY_CONNECTED>(
    const model::component::Operator::Ptr &operator_) {
    ENN_DBG_PRINT("TFlite::BuiltinOperator::BuiltinOperator_FULLY_CONNECTED opr->code = %d is called\n",
                  operator_->get_code());
    std::string builtin_op_name = TFlite::EnumNameBuiltinOperator(operator_->get_code());
    CHECK_AND_RETURN_ERR(0 != builtin_op_name.compare(operator_->get_name()),
                         ENN_RET_FAILED,
                         "Invalid BuiltinOperator : %s\n",
                         builtin_op_name.c_str());

    std::shared_ptr<FullyConnectedParameters> parameters = std::make_shared<FullyConnectedParameters>();
    std::vector<std::shared_ptr<ITensor>> in_tensors, out_tensors, data_tensors;

    CHECK_AND_RETURN_ERR(0 >= operator_->out_tensors.count(), ENN_RET_FAILED, "1<= operator_->out_tensors.count()\n");
    PrecisionType precision = PrecisionType::FP16;
    model::component::FeatureMap::Ptr ofm =
        std::static_pointer_cast<model::component::FeatureMap>(operator_->out_tensors[0]);
    precision = get_precision((TFlite::TensorType)ofm->get_data_type());

    convert_to_tensors(operator_, precision, in_tensors, out_tensors);

    CHECK_AND_RETURN_ERR(
        nullptr == operator_->get_option().get_addr(), ENN_RET_FAILED, "operator_->get_option().get_addr() == nullptr\n");

    const TFlite::FullyConnectedOptions *tflOptions =
        reinterpret_cast<const TFlite::FullyConnectedOptions *>(operator_->get_option().get_addr());
#ifdef SCHEMA_NNC_V1
    parameters->androidNN = false;
#else
    parameters->androidNN = legacy_model_ == TFlite::LegacyModel::LegacyModel_ANDROID_NN;
#endif
    parameters->storage_type = StorageType::BUFFER;
    parameters->activation_info = std::make_shared<ActivationInfo>(
        static_cast<ActivationInfo::ActivationType>(tflOptions->fused_activation_function()),
        tflOptions->fused_activation_function() != TFlite::ActivationFunctionType::ActivationFunctionType_NONE);

    std::shared_ptr<CLFullyConnected> fully_connected = compute_library_->createFullyConnected(precision);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != fully_connected->initialize(in_tensors, out_tensors, parameters),
                         ENN_RET_FAILED,
                         "CLFullyConnected initialize is failed\n");

    operators->push_back(std::make_shared<EnnUDOperator<CLFullyConnected>>(
        operator_->get_name(), operator_->get_id(), in_tensors, out_tensors, data_tensors, fully_connected));

    return ENN_RET_SUCCESS;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM>(
    const model::component::Operator::Ptr &operator_) {
    ENN_DBG_PRINT("TFlite::BuiltinOperator::BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM opr->code = %d is called\n",
                  operator_->get_code());
    std::string builtin_op_name = TFlite::EnumNameBuiltinOperator(operator_->get_code());
    CHECK_AND_RETURN_ERR(0 != builtin_op_name.compare(operator_->get_name()),
                         ENN_RET_FAILED,
                         "Invalid BuiltinOperator : %s\n",
                         builtin_op_name.c_str());

    std::shared_ptr<BidirectionalSequenceLstmParameters> parameters =
        std::make_shared<BidirectionalSequenceLstmParameters>();
    std::vector<std::shared_ptr<ITensor>> in_tensors, out_tensors, data_tensors;

    CHECK_AND_RETURN_ERR(0 >= operator_->out_tensors.count(), ENN_RET_FAILED, "1<= operator_->out_tensors.count()\n");
    PrecisionType precision = PrecisionType::FP16;
    model::component::FeatureMap::Ptr ofm =
        std::static_pointer_cast<model::component::FeatureMap>(operator_->out_tensors[0]);
    precision = get_precision((TFlite::TensorType)ofm->get_data_type());

    convert_to_tensors(operator_, precision, in_tensors, out_tensors);

    CHECK_AND_RETURN_ERR(
        nullptr == operator_->get_option().get_addr(), ENN_RET_FAILED, "operator_->get_option().get_addr() == nullptr\n");

    const TFlite::BidirectionalSequenceLSTMOptions *tflOptions =
        reinterpret_cast<const TFlite::BidirectionalSequenceLSTMOptions *>(operator_->get_option().get_addr());
    parameters->cell_clip = tflOptions->cell_clip();
    parameters->proj_clip = tflOptions->proj_clip();
    parameters->merge_outputs = tflOptions->merge_outputs();
    parameters->time_major = tflOptions->time_major();
    parameters->androidNN = false;

#ifndef SCHEMA_NNC_V1
    parameters->androidNN = TFlite::LegacyModel::LegacyModel_ANDROID_NN == legacy_model_;
#endif

    // TODO(all): Set true when run CRNN float model
    parameters->force_alloc_state_at_init = false;
    parameters->weights_as_input = false;
    parameters->activation_info = std::make_shared<ActivationInfo>(
        static_cast<ActivationInfo::ActivationType>(tflOptions->fused_activation_function()),
        tflOptions->fused_activation_function() != TFlite::ActivationFunctionType::ActivationFunctionType_NONE);

    std::shared_ptr<CLBidirectionalSequenceLstm> bidirectional_sequence_lstm =
        compute_library_->createBidirectionalSequenceLstm(precision);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != bidirectional_sequence_lstm->initialize(in_tensors, out_tensors, parameters),
                         ENN_RET_FAILED,
                         "CLBidirectionalSequenceLstm initialize is failed\n");

    operators->push_back(std::make_shared<EnnUDOperator<CLBidirectionalSequenceLstm>>(
        operator_->get_name(), operator_->get_id(), in_tensors, out_tensors, data_tensors, bidirectional_sequence_lstm));

    return ENN_RET_SUCCESS;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_REDUCE_MIN>(
    const std::shared_ptr<model::component::Operator> &operator_) {
    ENN_DBG_PRINT("TFlite::BuiltinOperator::BuiltinOperator_REDUCE_MIN opr->code = %d is called\n", operator_->get_code());
    std::string builtin_op_name = TFlite::EnumNameBuiltinOperator(operator_->get_code());
    CHECK_AND_RETURN_ERR(0 != builtin_op_name.compare(operator_->get_name()),
                         ENN_RET_FAILED,
                         "Invalid BuiltinOperator : %s\n",
                         builtin_op_name.c_str());

    std::shared_ptr<ReduceParameters> parameters = std::make_shared<ReduceParameters>();
    std::vector<std::shared_ptr<ITensor>> in_tensors, out_tensors, data_tensors;

    CHECK_AND_RETURN_ERR(0 >= operator_->out_tensors.count(), ENN_RET_FAILED, "1<= operator_->out_tensors.count()\n");
    PrecisionType precision = PrecisionType::FP16;
    std::shared_ptr<model::component::FeatureMap> ofm =
        std::static_pointer_cast<model::component::FeatureMap>(operator_->out_tensors[0]);
    precision = get_precision((TFlite::TensorType)ofm->get_data_type());

    convert_to_tensors(operator_, precision, in_tensors, out_tensors);

    CHECK_AND_RETURN_ERR(
        nullptr == operator_->get_option().get_addr(), ENN_RET_FAILED, "operator_->get_option().get_addr() == nullptr\n");
    parameters->keep_dims = false;
    parameters->reducer = Reducer::MIN;

    std::shared_ptr<CLReduce> reduce = compute_library_->createReduce(precision);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != reduce->initialize(in_tensors, out_tensors, parameters),
                         ENN_RET_FAILED,
                         "CLReduce initialize is failed\n");

    operators->push_back(std::make_shared<EnnUDOperator<CLReduce>>(
        operator_->get_name(), operator_->get_id(), in_tensors, out_tensors, data_tensors, reduce));

    return ENN_RET_SUCCESS;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_SPLIT>(
    const model::component::Operator::Ptr &operator_) {
    ENN_DBG_PRINT("TFlite::BuiltinOperator::BuiltinOperator_SPLIT opr->code = %d is called\n", operator_->get_code());
    std::string builtin_op_name = TFlite::EnumNameBuiltinOperator(operator_->get_code());
    CHECK_AND_RETURN_ERR(0 != builtin_op_name.compare(operator_->get_name()),
                         ENN_RET_FAILED,
                         "Invalid BuiltinOperator : %s\n",
                         builtin_op_name.c_str());

    std::shared_ptr<SplitParameters> parameters = std::make_shared<SplitParameters>();
    std::vector<std::shared_ptr<ITensor>> in_tensors, out_tensors, data_tensors;

    CHECK_AND_RETURN_ERR(0 >= operator_->out_tensors.count(), ENN_RET_FAILED, "1<= operator_->out_tensors.count()\n");
    PrecisionType precision = PrecisionType::FP16;
    model::component::FeatureMap::Ptr ofm =
        std::static_pointer_cast<model::component::FeatureMap>(operator_->out_tensors[0]);
    precision = get_precision((TFlite::TensorType)ofm->get_data_type());

    convert_to_tensors(operator_, precision, in_tensors, out_tensors);

    CHECK_AND_RETURN_ERR(
        nullptr == operator_->get_option().get_addr(), ENN_RET_FAILED, "operator_->get_option().get_addr() == nullptr\n");

    const TFlite::SplitOptions *tflOptions =
        reinterpret_cast<const TFlite::SplitOptions *>(operator_->get_option().get_addr());
    // TODO(all): Change v1 scheam file to add axis info (SplitOptions)
    parameters->axis = -1;
    parameters->num_outputs = tflOptions->num_splits();
    parameters->androidNN = false;

#ifndef SCHEMA_NNC_V1
    parameters->axis = tflOptions->axis();
    parameters->androidNN = TFlite::LegacyModel::LegacyModel_ANDROID_NN == legacy_model_;
#endif

    std::shared_ptr<CLSplit> split = compute_library_->createSplit(precision);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != split->initialize(in_tensors, out_tensors, parameters),
                         ENN_RET_FAILED,
                         "CLSplit initialize is failed\n");

    operators->push_back(std::make_shared<EnnUDOperator<CLSplit>>(
        operator_->get_name(), operator_->get_id(), in_tensors, out_tensors, data_tensors, split));

    return ENN_RET_SUCCESS;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_RELU>(
    const model::component::Operator::Ptr &operation) {
    ENN_DBG_PRINT("OpInfo: op_code: %d, name = %s, builtin_op_name = %s\n",
                  operation->get_code(),
                  operation->get_name().c_str(),
                  TFlite::EnumNameBuiltinOperator(operation->get_code()));

    if (operation->get_code() != TFlite::BuiltinOperator::BuiltinOperator_RELU) {
        ENN_ERR_PRINT("Invalid BuiltinOperator: %d\n", operation->get_code());
        return ENN_RET_INVAL;
    }

#ifdef SCHEMA_NNC_V1
    if (operation->get_option().get_enum() != TFlite::BuiltinOptions_ReluOptions) {
#else
    if (operation->get_option().get_enum() != TFlite::BuiltinOptions_ENN_ReluOptions) {
#endif
        ENN_ERR_PRINT("Invalid BuiltinOptions : %s\n", TFlite::EnumNameBuiltinOptions(operation->get_option().get_enum()));
        return ENN_RET_INVAL;
    }

    if (operation->get_option().get_addr() == nullptr) {
        ENN_ERR_PRINT("Invalid BuiltinOptions (null)\n");
        return ENN_RET_INVAL;
    }

    auto ofm = std::static_pointer_cast<model::component::FeatureMap>(operation->out_tensors[0]);
    PrecisionType precision_type = get_precision((TFlite::TensorType)ofm->get_data_type());

    std::vector<std::shared_ptr<ITensor>> input_tensors, output_tensors, data_tensors;
    convert_to_tensors(operation, precision_type, input_tensors, output_tensors);

    std::shared_ptr<ReluParameters> parameters = std::make_shared<ReluParameters>();
    if (operation->get_option().get_size() == sizeof(TC_ReluOptions)) {
        auto options = (TC_ReluOptions *)(operation->get_option().get_addr());
        parameters->negative_slope = options->negative_slope;
    } else {
#ifdef SCHEMA_NNC_V1
        auto options = reinterpret_cast<const TFlite::ReluOptions *>(operation->get_option().get_addr());
#else
        auto options = reinterpret_cast<const TFlite::ENN_ReluOptions *>(operation->get_option().get_addr());
#endif
        parameters->negative_slope = options->negative_slope();
    }

    const auto &relu = compute_library_->createRelu(precision_type);

    Status status = relu->initialize(input_tensors, output_tensors, parameters);
    if (Status::SUCCESS != status) {
        ENN_ERR_PRINT("CLRelu initialize is failed\n");
        return ENN_RET_INVAL;
    }

    operators->push_back(std::make_shared<EnnUDOperator<CLRelu>>(
        operation->get_name(), operation->get_id(), input_tensors, output_tensors, data_tensors, relu));

    return (status == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_RELU_N1_TO_1>(
    const model::component::Operator::Ptr &operation) {
    ENN_DBG_PRINT("OpInfo: op_code: %d, name = %s, builtin_op_name = %s\n",
                  operation->get_code(),
                  operation->get_name().c_str(),
                  TFlite::EnumNameBuiltinOperator(operation->get_code()));

    if (operation->get_code() != TFlite::BuiltinOperator::BuiltinOperator_RELU_N1_TO_1) {
        ENN_ERR_PRINT("Invalid BuiltinOperator: %d\n", operation->get_code());
        return ENN_RET_INVAL;
    }

    auto ofm = std::static_pointer_cast<model::component::FeatureMap>(operation->out_tensors[0]);
    PrecisionType precision_type = get_precision((TFlite::TensorType)ofm->get_data_type());

    std::vector<std::shared_ptr<ITensor>> input_tensors, output_tensors, data_tensors;
    convert_to_tensors(operation, precision_type, input_tensors, output_tensors);

    const auto &relu1 = compute_library_->createRelu1(precision_type);

    Status status = relu1->initialize(input_tensors, output_tensors, nullptr);
    if (Status::SUCCESS != status) {
        ENN_ERR_PRINT("CLRelu1 initialize is failed\n");
        return ENN_RET_INVAL;
    }

    operators->push_back(std::make_shared<EnnUDOperator<CLRelu1>>(
        operation->get_name(), operation->get_id(), input_tensors, output_tensors, data_tensors, relu1));

    return (status == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_RELU6>(
    const model::component::Operator::Ptr &operation) {
    ENN_DBG_PRINT("OpInfo: op_code: %d, name = %s, builtin_op_name = %s\n",
                  operation->get_code(),
                  operation->get_name().c_str(),
                  TFlite::EnumNameBuiltinOperator(operation->get_code()));

    if (operation->get_code() != TFlite::BuiltinOperator::BuiltinOperator_RELU6) {
        ENN_ERR_PRINT("Invalid BuiltinOperator: %d\n", operation->get_code());
        return ENN_RET_INVAL;
    }

    auto ofm = std::static_pointer_cast<model::component::FeatureMap>(operation->out_tensors[0]);
    PrecisionType precision_type = get_precision((TFlite::TensorType)ofm->get_data_type());

    std::vector<std::shared_ptr<ITensor>> input_tensors, output_tensors, data_tensors;
    convert_to_tensors(operation, precision_type, input_tensors, output_tensors);

    const auto &relu6 = compute_library_->createRelu6(precision_type);

    Status status = relu6->initialize(input_tensors, output_tensors, nullptr);
    if (Status::SUCCESS != status) {
        ENN_ERR_PRINT("CLRelu6 initialize is failed\n");
        return ENN_RET_INVAL;
    }

    operators->push_back(std::make_shared<EnnUDOperator<CLRelu6>>(
        operation->get_name(), operation->get_id(), input_tensors, output_tensors, data_tensors, relu6));

    return (status == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_LOGISTIC>(
    const model::component::Operator::Ptr &operation) {
    ENN_DBG_PRINT("OpInfo: op_code: %d, name = %s, builtin_op_name = %s\n",
                  operation->get_code(),
                  operation->get_name().c_str(),
                  TFlite::EnumNameBuiltinOperator(operation->get_code()));

    if (operation->get_code() != TFlite::BuiltinOperator::BuiltinOperator_LOGISTIC) {
        ENN_ERR_PRINT("Invalid BuiltinOperator: %d\n", operation->get_code());
        return ENN_RET_INVAL;
    }

    auto ofm = std::static_pointer_cast<model::component::FeatureMap>(operation->out_tensors[0]);
    PrecisionType precision_type = get_precision((TFlite::TensorType)ofm->get_data_type());

    std::vector<std::shared_ptr<ITensor>> input_tensors, output_tensors, data_tensors;
    convert_to_tensors(operation, precision_type, input_tensors, output_tensors);

    const auto &sigmoid_op = compute_library_->createSigmoid(precision_type);

    Status status = sigmoid_op->initialize(input_tensors, output_tensors, nullptr);
    if (Status::SUCCESS != status) {
        ENN_ERR_PRINT("CLSigmoid initialize is failed\n");
        return ENN_RET_INVAL;
    }

    operators->push_back(std::make_shared<EnnUDOperator<CLSigmoid>>(
        operation->get_name(), operation->get_id(), input_tensors, output_tensors, data_tensors, sigmoid_op));

    return (status == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_TANH>(
    const model::component::Operator::Ptr &operation) {
    ENN_DBG_PRINT("OpInfo: op_code: %d, name = %s, builtin_op_name = %s\n",
                  operation->get_code(),
                  operation->get_name().c_str(),
                  TFlite::EnumNameBuiltinOperator(operation->get_code()));

    if (operation->get_code() != TFlite::BuiltinOperator::BuiltinOperator_TANH) {
        ENN_ERR_PRINT("Invalid BuiltinOperator: %d\n", operation->get_code());
        return ENN_RET_INVAL;
    }

    auto ofm = std::static_pointer_cast<model::component::FeatureMap>(operation->out_tensors[0]);
    PrecisionType precision_type = get_precision((TFlite::TensorType)ofm->get_data_type());

    std::vector<std::shared_ptr<ITensor>> input_tensors, output_tensors, data_tensors;
    convert_to_tensors(operation, precision_type, input_tensors, output_tensors);

    const auto &tanh_op = compute_library_->createTanh(precision_type);

    Status status = tanh_op->initialize(input_tensors, output_tensors, nullptr);
    if (Status::SUCCESS != status) {
        ENN_ERR_PRINT("CLTanh initialize is failed\n");
        return ENN_RET_INVAL;
    }

    operators->push_back(std::make_shared<EnnUDOperator<CLTanh>>(
        operation->get_name(), operation->get_id(), input_tensors, output_tensors, data_tensors, tanh_op));

    return (status == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_MAX_POOL_2D>(
    const model::component::Operator::Ptr &operation) {
    ENN_DBG_PRINT("OpInfo: op_code: %d, name = %s, builtin_op_name = %s\n",
                  operation->get_code(),
                  operation->get_name().c_str(),
                  TFlite::EnumNameBuiltinOperator(operation->get_code()));

    if (operation->get_code() != TFlite::BuiltinOperator::BuiltinOperator_MAX_POOL_2D) {
        ENN_ERR_PRINT("Invalid BuiltinOperator: %d\n", operation->get_code());
        return ENN_RET_INVAL;
    }

    auto ofm = std::static_pointer_cast<model::component::FeatureMap>(operation->out_tensors[0]);
    PrecisionType precision_type = get_precision((TFlite::TensorType)ofm->get_data_type());

    std::vector<std::shared_ptr<ITensor>> input_tensors, output_tensors, data_tensors;
    convert_to_tensors(operation, precision_type, input_tensors, output_tensors);

    std::shared_ptr<Pool2DParameters> parameters = std::make_shared<Pool2DParameters>();
    auto tfl_ptions = reinterpret_cast<const TFlite::Pool2DOptions *>(operation->get_option().get_addr());
    parameters->padding = {0, 0, 0, 0};
    parameters->stride.w = tfl_ptions->stride_w();
    parameters->stride.h = tfl_ptions->stride_h();
    parameters->filter.w = tfl_ptions->filter_width();
    parameters->filter.h = tfl_ptions->filter_height();
    parameters->activation_info = ActivationInfo(
        static_cast<ActivationInfo::ActivationType>(tfl_ptions->fused_activation_function()),
        tfl_ptions->fused_activation_function() != TFlite::ActivationFunctionType::ActivationFunctionType_NONE);

    parameters->androidNN = false;
    parameters->isNCHW = is_nchw_layout(legacy_model_);
#ifndef SCHEMA_NNC_V1
    parameters->androidNN = legacy_model_ == TFlite::LegacyModel::LegacyModel_ANDROID_NN;
    if (parameters->androidNN)
        parameters->isNCHW = tfl_ptions->use_nchw();

#endif

    if (tfl_ptions->padding_value()->size() == 4) {
        parameters->padding.l = static_cast<uint32_t>(tfl_ptions->padding_value()->data()[0]);
        parameters->padding.r = static_cast<uint32_t>(tfl_ptions->padding_value()->data()[1]);
        parameters->padding.t = static_cast<uint32_t>(tfl_ptions->padding_value()->data()[2]);
        parameters->padding.b = static_cast<uint32_t>(tfl_ptions->padding_value()->data()[3]);
    } else {
        get_padding(tfl_ptions->padding(),
                    parameters->padding,
                    input_tensors[0]->getDims(),
                    output_tensors[0]->getDims(),
                    {1, 1, parameters->filter.h, parameters->filter.w},
                    parameters->stride,
                    {1, 1},
                    false,
                    parameters->isNCHW);
    }

    const auto &max_pool = compute_library_->createMaxpool(precision_type);

    Status status = max_pool->initialize(input_tensors, output_tensors, parameters);
    if (Status::SUCCESS != status) {
        ENN_ERR_PRINT("CLMaxpool initialize is failed\n");
        return ENN_RET_INVAL;
    }

    operators->push_back(std::make_shared<EnnUDOperator<CLMaxpool>>(
        operation->get_name(), operation->get_id(), input_tensors, output_tensors, data_tensors, max_pool));

    return (status == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_AVERAGE_POOL_2D>(
    const model::component::Operator::Ptr &operation) {
    ENN_DBG_PRINT("OpInfo: op_code: %d, name = %s, builtin_op_name = %s\n",
                  operation->get_code(),
                  operation->get_name().c_str(),
                  TFlite::EnumNameBuiltinOperator(operation->get_code()));

    if (operation->get_code() != TFlite::BuiltinOperator::BuiltinOperator_AVERAGE_POOL_2D) {
        ENN_ERR_PRINT("Invalid BuiltinOperator: %d\n", operation->get_code());
        return ENN_RET_INVAL;
    }

    auto ofm = std::static_pointer_cast<model::component::FeatureMap>(operation->out_tensors[0]);
    PrecisionType precision_type = get_precision((TFlite::TensorType)ofm->get_data_type());

    std::vector<std::shared_ptr<ITensor>> input_tensors, output_tensors, data_tensors;
    convert_to_tensors(operation, precision_type, input_tensors, output_tensors);

    std::shared_ptr<Pool2DParameters> parameters = std::make_shared<Pool2DParameters>();

    auto tfl_ptions = reinterpret_cast<const TFlite::Pool2DOptions *>(operation->get_option().get_addr());
    parameters->padding = {0, 0, 0, 0};
    parameters->stride.w = tfl_ptions->stride_w();
    parameters->stride.h = tfl_ptions->stride_h();
    parameters->filter.w = tfl_ptions->filter_width();
    parameters->filter.h = tfl_ptions->filter_height();
    if (this->legacy_model_ == TFlite::LegacyModel::LegacyModel_CAFFE ||
        this->legacy_model_ == TFlite::LegacyModel::LegacyModel_CAFFE_NCHW ||
        this->legacy_model_ == TFlite::LegacyModel::LegacyModel_CAFFE_NHWC) {
        parameters->compute_type = ComputeType::Caffe;
    } else {
        parameters->compute_type = ComputeType::TFLite;
    }
    parameters->activation_info = ActivationInfo(
        static_cast<ActivationInfo::ActivationType>(tfl_ptions->fused_activation_function()),
        tfl_ptions->fused_activation_function() != TFlite::ActivationFunctionType::ActivationFunctionType_NONE);

    parameters->androidNN = false;
    parameters->isNCHW = is_nchw_layout(legacy_model_);
#ifndef SCHEMA_NNC_V1
    parameters->androidNN = legacy_model_ == TFlite::LegacyModel::LegacyModel_ANDROID_NN;
    if (parameters->androidNN)
        parameters->isNCHW = tfl_ptions->use_nchw();
#endif

    if (tfl_ptions->padding_value()->size() == 4) {
        parameters->padding.l = static_cast<uint32_t>(tfl_ptions->padding_value()->data()[0]);
        parameters->padding.r = static_cast<uint32_t>(tfl_ptions->padding_value()->data()[1]);
        parameters->padding.t = static_cast<uint32_t>(tfl_ptions->padding_value()->data()[2]);
        parameters->padding.b = static_cast<uint32_t>(tfl_ptions->padding_value()->data()[3]);
    } else {
        get_padding(tfl_ptions->padding(),
                    parameters->padding,
                    input_tensors[0]->getDims(),
                    output_tensors[0]->getDims(),
                    {1, 1, parameters->filter.h, parameters->filter.w},
                    parameters->stride,
                    {1, 1},
                    false,
                    parameters->isNCHW);
    }

    const auto &average_pool = compute_library_->createAveragepool(precision_type);

    Status status = average_pool->initialize(input_tensors, output_tensors, parameters);
    if (Status::SUCCESS != status) {
        ENN_ERR_PRINT("CLAveragepool initialize is failed\n");
        return ENN_RET_INVAL;
    }

    operators->push_back(std::make_shared<EnnUDOperator<CLAveragepool>>(
        operation->get_name(), operation->get_id(), input_tensors, output_tensors, data_tensors, average_pool));

    return (status == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_CONCATENATION>(
    const model::component::Operator::Ptr &operation) {
    ENN_DBG_PRINT("OpInfo: op_code: %d, name = %s, builtin_op_name = %s\n",
                  operation->get_code(),
                  operation->get_name().c_str(),
                  TFlite::EnumNameBuiltinOperator(operation->get_code()));

    if (operation->get_code() != TFlite::BuiltinOperator::BuiltinOperator_CONCATENATION) {
        ENN_ERR_PRINT("Invalid BuiltinOperator: %d\n", operation->get_code());
        return ENN_RET_INVAL;
    }

    auto ofm = std::static_pointer_cast<model::component::FeatureMap>(operation->out_tensors[0]);
    PrecisionType precision_type = get_precision((TFlite::TensorType)ofm->get_data_type());

    std::vector<std::shared_ptr<ITensor>> input_tensors, output_tensors, data_tensors;
    convert_to_tensors(operation, precision_type, input_tensors, output_tensors);

    std::shared_ptr<ConcatParameters> parameters = std::make_shared<ConcatParameters>();
    if (operation->get_option().get_size() == sizeof(TC_ConcatDOptions)) {
        auto options = (TC_ConcatDOptions *)(operation->get_option().get_addr());
        parameters->axis = options->axis;
        parameters->activation_info =
            ActivationInfo(options->activation_info.activation(), options->activation_info.isEnabled());
    } else {
        auto options = reinterpret_cast<const TFlite::ConcatenationOptions *>(operation->get_option().get_addr());
        parameters->axis = options->axis();
        parameters->activation_info = ActivationInfo(
            static_cast<ActivationInfo::ActivationType>(options->fused_activation_function()),
            options->fused_activation_function() != TFlite::ActivationFunctionType::ActivationFunctionType_NONE);
    }

    const auto &concat = compute_library_->createConcat(precision_type);

    Status status = concat->initialize(input_tensors, output_tensors, parameters);
    if (Status::SUCCESS != status) {
        ENN_ERR_PRINT("CLConcat initialize is failed\n");
        return ENN_RET_INVAL;
    }

    operators->push_back(std::make_shared<EnnUDOperator<CLConcat>>(
        operation->get_name(), operation->get_id(), input_tensors, output_tensors, data_tensors, concat));

    return (status == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_SOFTMAX>(
    const model::component::Operator::Ptr &operation) {
    ENN_DBG_PRINT("OpInfo: op_code: %d, name = %s, builtin_op_name = %s\n",
                  operation->get_code(),
                  operation->get_name().c_str(),
                  TFlite::EnumNameBuiltinOperator(operation->get_code()));

    if (operation->get_code() != TFlite::BuiltinOperator::BuiltinOperator_SOFTMAX) {
        ENN_ERR_PRINT("Invalid BuiltinOperator: %d\n", operation->get_code());
        return ENN_RET_INVAL;
    }

    auto ofm = std::static_pointer_cast<model::component::FeatureMap>(operation->out_tensors[0]);
    PrecisionType precision_type = get_precision((TFlite::TensorType)ofm->get_data_type());

    std::vector<std::shared_ptr<ITensor>> input_tensors, output_tensors, data_tensors;
    convert_to_tensors(operation, precision_type, input_tensors, output_tensors);

    std::shared_ptr<SoftmaxParameters> parameters = std::make_shared<SoftmaxParameters>();
    if (operation->get_option().get_size() == sizeof(TC_SoftmaxOptions)) {
        auto options = (TC_SoftmaxOptions *)(operation->get_option().get_addr());
        parameters->axis = options->axis();
        parameters->beta = options->beta();
    } else {
        auto options = reinterpret_cast<const TFlite::SoftmaxOptions *>(operation->get_option().get_addr());
        parameters->axis = options->axis();
        parameters->beta = options->beta();
    }
    const auto &softmax = compute_library_->createSoftmax(precision_type);

    Status status = softmax->initialize(input_tensors, output_tensors, parameters);
    if (Status::SUCCESS != status) {
        ENN_ERR_PRINT("CLSoftmax initialize is failed\n");
        return ENN_RET_INVAL;
    }

    operators->push_back(std::make_shared<EnnUDOperator<CLSoftmax>>(
        operation->get_name(), operation->get_id(), input_tensors, output_tensors, data_tensors, softmax));

    return (status == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_MEAN>(
    const std::shared_ptr<model::component::Operator> &operation) {
    ENN_DBG_PRINT("TFlite::BuiltinOperator::BuiltinOperator_MEAN opr->code = %d is called\n", operation->get_code());
    std::string builtin_op_name = TFlite::EnumNameBuiltinOperator(operation->get_code());
    CHECK_AND_RETURN_ERR(0 != builtin_op_name.compare(operation->get_name()),
                         ENN_RET_FAILED,
                         "Invalid BuiltinOperator : %s\n",
                         builtin_op_name.c_str());
    CHECK_AND_RETURN_ERR(0 >= operation->out_tensors.count(), ENN_RET_FAILED, "1<= operator_->out_tensors.count()\n");

    std::shared_ptr<MeanParameters> parameters = std::make_shared<MeanParameters>();
    PrecisionType precision_type = get_precision((TFlite::TensorType)operation->out_tensors[0]->get_data_type());
    std::vector<std::shared_ptr<ITensor>> input_tensors, output_tensors, data_tensors;
    convert_to_tensors(operation, precision_type, input_tensors, output_tensors);

    CHECK_AND_RETURN_ERR(
        nullptr == operation->get_option().get_addr(), ENN_RET_FAILED, "operator_->get_option().get_addr() == nullptr\n");
#ifdef SCHEMA_NNC_V1
    parameters->keep_dims = false;
    const tflite::MeanOptions *tflOptions =
        reinterpret_cast<const tflite::MeanOptions *>(operation->get_option().get_addr());
#else
    const TFlite::ENN_MeanOptions *tflOptions =
        reinterpret_cast<const TFlite::ENN_MeanOptions *>(operation->get_option().get_addr());
    parameters->keep_dims = tflOptions->keep_dims() == 0 ? false : true;
#endif
    std::shared_ptr<CLMean> mean = compute_library_->createMean(precision_type);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != mean->initialize(input_tensors, output_tensors, parameters),
                         ENN_RET_FAILED,
                         "CLMean initialize is failed\n");

    operators->push_back(std::make_shared<EnnUDOperator<CLMean>>(
        operation->get_name(), operation->get_id(), input_tensors, output_tensors, data_tensors, mean));

    return ENN_RET_SUCCESS;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_SUB>(
    const model::component::Operator::Ptr &operation) {
    ENN_DBG_PRINT("OpInfo: op_code: %d, name = %s, builtin_op_name = %s\n",
                  operation->get_code(),
                  operation->get_name().c_str(),
                  TFlite::EnumNameBuiltinOperator(operation->get_code()));

    if (operation->get_code() != TFlite::BuiltinOperator::BuiltinOperator_SUB) {
        ENN_ERR_PRINT("Invalid BuiltinOperator: %d\n", operation->get_code());
        return ENN_RET_INVAL;
    }

    auto ofm = std::static_pointer_cast<model::component::FeatureMap>(operation->out_tensors[0]);
    PrecisionType precision_type = get_precision((TFlite::TensorType)ofm->get_data_type());

    std::vector<std::shared_ptr<ITensor>> input_tensors, output_tensors, data_tensors;
    convert_to_tensors(operation, precision_type, input_tensors, output_tensors);

    std::shared_ptr<SubParameters> parameters = std::make_shared<SubParameters>();
    // TODO(all): remove TC_SubOptions in ud_test
    if (operation->get_option().get_size() == sizeof(TC_SubOptions)) {
        auto options = (TC_SubOptions *)(operation->get_option().get_addr());
        parameters->activation_info =
            ActivationInfo(options->activation_info.activation(), options->activation_info.isEnabled());
    } else {
        auto options = reinterpret_cast<const TFlite::SubOptions *>(operation->get_option().get_addr());
        parameters->activation_info = ActivationInfo(
            static_cast<ActivationInfo::ActivationType>(options->fused_activation_function()),
            options->fused_activation_function() != TFlite::ActivationFunctionType::ActivationFunctionType_NONE);
    }

    const auto &sub_op = compute_library_->createSub(precision_type);

    Status status = sub_op->initialize(input_tensors, output_tensors, parameters);
    if (Status::SUCCESS != status) {
        ENN_ERR_PRINT("CLSub initialize is failed\n");
        return ENN_RET_INVAL;
    }

    operators->push_back(std::make_shared<EnnUDOperator<CLSub>>(
        operation->get_name(), operation->get_id(), input_tensors, output_tensors, data_tensors, sub_op));

    return (status == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_MUL>(
    const model::component::Operator::Ptr &operation) {
    ENN_DBG_PRINT("OpInfo: op_code: %d, name = %s, builtin_op_name = %s\n",
                  operation->get_code(),
                  operation->get_name().c_str(),
                  TFlite::EnumNameBuiltinOperator(operation->get_code()));

    if (operation->get_code() != TFlite::BuiltinOperator::BuiltinOperator_MUL) {
        ENN_ERR_PRINT("Invalid BuiltinOperator: %d\n", operation->get_code());
        return ENN_RET_INVAL;
    }

    auto ofm = std::static_pointer_cast<model::component::FeatureMap>(operation->out_tensors[0]);
    PrecisionType precision_type = get_precision((TFlite::TensorType)ofm->get_data_type());

    std::vector<std::shared_ptr<ITensor>> input_tensors, output_tensors, data_tensors;
    convert_to_tensors(operation, precision_type, input_tensors, output_tensors);

    std::shared_ptr<MulParameters> parameters = std::make_shared<MulParameters>();
    // TODO(all): remove TC_MulOptions in ud_test
    if (operation->get_option().get_size() == sizeof(TC_MulOptions)) {
        auto options = (TC_MulOptions *)(operation->get_option().get_addr());
        parameters->activation_info =
            ActivationInfo(options->activation_info.activation(), options->activation_info.isEnabled());
    } else {
        auto options = reinterpret_cast<const TFlite::MulOptions *>(operation->get_option().get_addr());
        parameters->activation_info = ActivationInfo(
            static_cast<ActivationInfo::ActivationType>(options->fused_activation_function()),
            options->fused_activation_function() != TFlite::ActivationFunctionType::ActivationFunctionType_NONE);
        parameters->androidNN = legacy_model_ == TFlite::LegacyModel::LegacyModel_ANDROID_NN;
        parameters->isNCHW = false;  // TODO(xin.lu): set true when optimize for NCHW block
    }

    const auto &mul_op = compute_library_->createMul(precision_type);

    Status status = mul_op->initialize(input_tensors, output_tensors, parameters);
    if (Status::SUCCESS != status) {
        ENN_ERR_PRINT("CLMul initialize is failed\n");
        return ENN_RET_INVAL;
    }

    operators->push_back(std::make_shared<EnnUDOperator<CLMul>>(
        operation->get_name(), operation->get_id(), input_tensors, output_tensors, data_tensors, mul_op));

    return (status == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_GATHER>(
    const model::component::Operator::Ptr &operation) {
    ENN_DBG_PRINT("OpInfo: op_code: %d, name = %s, builtin_op_name = %s\n",
                  operation->get_code(),
                  operation->get_name().c_str(),
                  TFlite::EnumNameBuiltinOperator(operation->get_code()));

    if (operation->get_code() != TFlite::BuiltinOperator::BuiltinOperator_GATHER) {
        ENN_ERR_PRINT("Invalid BuiltinOperator: %d\n", operation->get_code());
        return ENN_RET_INVAL;
    }

    std::vector<std::shared_ptr<ITensor>> input_tensors, output_tensors, data_tensors;
    auto parameters = std::make_shared<GatherParameters>();

    // set precision and create tensors
    PrecisionType precision_type = get_precision((TFlite::TensorType)operation->out_tensors[0]->get_data_type());
    convert_to_tensors(operation, precision_type, input_tensors, output_tensors);

    // parse parameter
    auto options = reinterpret_cast<const TFlite::GatherOptions *>(operation->get_option().get_addr());
    parameters->axis = options->axis();
    parameters->androidNN = legacy_model_ == TFlite::LegacyModel::LegacyModel_ANDROID_NN;
    // create operator
    const auto &gather = compute_library_->createGather(precision_type);
    Status status = gather->initialize(input_tensors, output_tensors, parameters);
    if (Status::SUCCESS != status) {
        ENN_ERR_PRINT("CLGather initialize is failed\n");
        return ENN_RET_INVAL;
    }

    operators->push_back(std::make_shared<EnnUDOperator<CLGather>>(
        operation->get_name(), operation->get_id(), input_tensors, output_tensors, data_tensors, gather));

    return (status == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_DIV>(
    const model::component::Operator::Ptr &operation) {
    ENN_DBG_PRINT("OpInfo: op_code: %d, name = %s, builtin_op_name = %s\n",
                  operation->get_code(),
                  operation->get_name().c_str(),
                  TFlite::EnumNameBuiltinOperator(operation->get_code()));

    if (operation->get_code() != TFlite::BuiltinOperator::BuiltinOperator_DIV) {
        ENN_ERR_PRINT("Invalid BuiltinOperator: %d\n", operation->get_code());
        return ENN_RET_INVAL;
    }

    auto ofm = std::static_pointer_cast<model::component::FeatureMap>(operation->out_tensors[0]);
    PrecisionType precision_type = get_precision((TFlite::TensorType)ofm->get_data_type());

    std::vector<std::shared_ptr<ITensor>> input_tensors, output_tensors, data_tensors;
    convert_to_tensors(operation, precision_type, input_tensors, output_tensors);

    std::shared_ptr<DivParameters> parameters = std::make_shared<DivParameters>();
    auto options = reinterpret_cast<const TFlite::DivOptions *>(operation->get_option().get_addr());

    parameters->activation_info =
        ActivationInfo(static_cast<ActivationInfo::ActivationType>(options->fused_activation_function()),
                       options->fused_activation_function() != TFlite::ActivationFunctionType::ActivationFunctionType_NONE);

    const auto &div_op = compute_library_->createDiv(precision_type);

    Status status = div_op->initialize(input_tensors, output_tensors, parameters);
    if (Status::SUCCESS != status) {
        ENN_ERR_PRINT("CLDiv initialize is failed\n");
        return ENN_RET_INVAL;
    }

    operators->push_back(std::make_shared<EnnUDOperator<CLDiv>>(
        operation->get_name(), operation->get_id(), input_tensors, output_tensors, data_tensors, div_op));

    return (status == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_ADD>(
    const model::component::Operator::Ptr &operation) {
    ENN_DBG_PRINT("OpInfo: op_code: %d, name = %s, builtin_op_name = %s\n",
                  operation->get_code(),
                  operation->get_name().c_str(),
                  TFlite::EnumNameBuiltinOperator(operation->get_code()));

    if (operation->get_code() != TFlite::BuiltinOperator::BuiltinOperator_ADD) {
        ENN_ERR_PRINT("Invalid BuiltinOperator: %d\n", operation->get_code());
        return ENN_RET_INVAL;
    }

    auto ofm = std::static_pointer_cast<model::component::FeatureMap>(operation->out_tensors[0]);
    PrecisionType precision_type = get_precision((TFlite::TensorType)ofm->get_data_type());

    std::vector<std::shared_ptr<ITensor>> input_tensors, output_tensors, data_tensors;
    convert_to_tensors(operation, precision_type, input_tensors, output_tensors);

    std::shared_ptr<AddParameters> parameters = std::make_shared<AddParameters>();
    auto options = reinterpret_cast<const TFlite::AddOptions *>(operation->get_option().get_addr());
    parameters->activation_info =
        ActivationInfo(static_cast<ActivationInfo::ActivationType>(options->fused_activation_function()),
                       options->fused_activation_function() != TFlite::ActivationFunctionType::ActivationFunctionType_NONE);
    parameters->coeff = {options->coeff()->begin(), options->coeff()->end()};
    parameters->androidNN = legacy_model_ == TFlite::LegacyModel::LegacyModel_ANDROID_NN;
    parameters->isNCHW = false;  // TODO(xin.lu): set true when optimize for NCHW block
    if (parameters->coeff.empty()) {
        parameters->coeff = std::vector<float>(input_tensors.size(), 1.0f);
    }

    const auto &add_op = compute_library_->createAdd(precision_type);

    Status status = add_op->initialize(input_tensors, output_tensors, parameters);
    if (Status::SUCCESS != status) {
        ENN_ERR_PRINT("CLAdd initialize is failed\n");
        return ENN_RET_INVAL;
    }

    operators->push_back(std::make_shared<EnnUDOperator<CLAdd>>(
        operation->get_name(), operation->get_id(), input_tensors, output_tensors, data_tensors, add_op));

    return (status == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_SQUEEZE>(
    const model::component::Operator::Ptr &operation) {
    ENN_DBG_PRINT("OpInfo: op_code: %d, name = %s, builtin_op_name = %s\n",
                  operation->get_code(),
                  operation->get_name().c_str(),
                  TFlite::EnumNameBuiltinOperator(operation->get_code()));

    if (operation->get_code() != TFlite::BuiltinOperator::BuiltinOperator_SQUEEZE) {
        ENN_ERR_PRINT("Invalid BuiltinOperator: %d\n", operation->get_code());
        return ENN_RET_INVAL;
    }

    auto ofm = std::static_pointer_cast<model::component::FeatureMap>(operation->out_tensors[0]);
    PrecisionType precision_type = get_precision((TFlite::TensorType)ofm->get_data_type());

    std::vector<std::shared_ptr<ITensor>> input_tensors, output_tensors, data_tensors;
    convert_to_tensors(operation, precision_type, input_tensors, output_tensors);

    std::shared_ptr<SqueezeParameters> parameters = std::make_shared<SqueezeParameters>();
    auto options = reinterpret_cast<const TFlite::SqueezeOptions *>(operation->get_option().get_addr());
    if (options->squeeze_dims() != nullptr) {
        parameters->squeeze_dims = {options->squeeze_dims()->begin(), options->squeeze_dims()->end()};
    }
    const auto &squeeze = compute_library_->createSqueeze(precision_type);
    Status status = squeeze->initialize(input_tensors, output_tensors, parameters);
    if (Status::SUCCESS != status) {
        ENN_ERR_PRINT("CLSqueeze initialize is failed\n");
        return ENN_RET_INVAL;
    }

    operators->push_back(std::make_shared<EnnUDOperator<CLSqueeze>>(
        operation->get_name(), operation->get_id(), input_tensors, output_tensors, data_tensors, squeeze));

    return (status == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_RESIZE_BILINEAR>(
    const model::component::Operator::Ptr &operation) {
    ENN_DBG_PRINT("OpInfo: op_code: %d, name = %s, builtin_op_name = %s\n",
                  operation->get_code(),
                  operation->get_name().c_str(),
                  TFlite::EnumNameBuiltinOperator(operation->get_code()));
    CHECK_AND_RETURN_ERR(operation->get_code() != TFlite::BuiltinOperator::BuiltinOperator_RESIZE_BILINEAR,
                         ENN_RET_INVAL,
                         "Invalid BuiltinOperator: %d\n",
                         operation->get_code());

    auto ofm = std::static_pointer_cast<model::component::FeatureMap>(operation->out_tensors[0]);
    PrecisionType precision_type = get_precision((TFlite::TensorType)ofm->get_data_type());

    std::vector<std::shared_ptr<ITensor>> input_tensors, output_tensors, data_tensors;
    convert_to_tensors(operation, precision_type, input_tensors, output_tensors);

    std::shared_ptr<ResizeBilinearParameters> parameters = std::make_shared<ResizeBilinearParameters>();
    auto options = reinterpret_cast<const TFlite::ResizeBilinearOptions *>(operation->get_option().get_addr());
    parameters->align_corners = options->align_corners();
    parameters->half_pixel_centers = options->half_pixel_centers();
    parameters->androidNN = false;
    parameters->isNCHW = is_nchw_layout(legacy_model_);

#ifdef SCHEMA_NNC_V1
    parameters->new_height = options->new_height();
    parameters->new_width = options->new_width();
#else
    parameters->androidNN = legacy_model_ == TFlite::LegacyModel::LegacyModel_ANDROID_NN;
    if (parameters->androidNN)
        parameters->isNCHW = options->use_nchw();
    CHECK_AND_RETURN_ERR(output_tensors.size() < 1, ENN_RET_INVAL, "CLResizeBilinear have 1 output_tensor\n");
    if (parameters->isNCHW) {
        parameters->new_height = output_tensors[0]->getDims()[H_NCHW];
        parameters->new_width = output_tensors[0]->getDims()[W_NCHW];
    } else {
        parameters->new_height = output_tensors[0]->getDims()[H_NHWC];
        parameters->new_width = output_tensors[0]->getDims()[W_NHWC];
    }
#endif

    const auto &resize_bilinear = compute_library_->createResizeBilinear(precision_type);

    Status status = resize_bilinear->initialize(input_tensors, output_tensors, parameters);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != status, ENN_RET_INVAL, "CLResizeBilinear initialize is failed\n");

    operators->push_back(std::make_shared<EnnUDOperator<CLResizeBilinear>>(
        operation->get_name(), operation->get_id(), input_tensors, output_tensors, data_tensors, resize_bilinear));

    return ENN_RET_SUCCESS;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_RESHAPE>(
    const model::component::Operator::Ptr &operation) {
    ENN_DBG_PRINT("OpInfo: op_code: %d, name = %s, builtin_op_name = %s\n",
                  operation->get_code(),
                  operation->get_name().c_str(),
                  TFlite::EnumNameBuiltinOperator(operation->get_code()));
    CHECK_AND_RETURN_ERR(operation->get_code() != TFlite::BuiltinOperator::BuiltinOperator_RESHAPE,
                         ENN_RET_INVAL,
                         "Invalid BuiltinOperator: %d\n",
                         operation->get_code());

    auto ofm = std::static_pointer_cast<model::component::FeatureMap>(operation->out_tensors[0]);
    PrecisionType precision_type = get_precision((TFlite::TensorType)ofm->get_data_type());

    std::vector<std::shared_ptr<ITensor>> input_tensors, output_tensors, data_tensors;
    convert_to_tensors(operation, precision_type, input_tensors, output_tensors);

    auto options = reinterpret_cast<const TFlite::ReshapeOptions *>(operation->get_option().get_addr());
    std::shared_ptr<ReshapeParameters> parameters = std::make_shared<ReshapeParameters>();
    parameters->new_shape.clear();
    if (input_tensors.size() < 2) {
        if (options->new_shape()->size() == 0) {
            ENN_ERR_PRINT("CLReshape must have shape_tensor or shape_options\n");
            return ENN_RET_INVAL;
        } else {
            for (auto new_shape : *(options->new_shape())) {
                parameters->new_shape.push_back(new_shape);
            }
        }
    }

    if (this->legacy_model_ == TFlite::LegacyModel::LegacyModel_CAFFE ||
        this->legacy_model_ == TFlite::LegacyModel::LegacyModel_CAFFE_NCHW ||
        this->legacy_model_ == TFlite::LegacyModel::LegacyModel_CAFFE_NHWC) {
        parameters->compute_type = ComputeType::Caffe;
    } else {
        parameters->compute_type = ComputeType::TFLite;
    }
#ifdef SCHEMA_NNC_V1
    parameters->androidNN = false;
#else
    parameters->androidNN = legacy_model_ == TFlite::LegacyModel::LegacyModel_ANDROID_NN;
#endif
    parameters->isNCHW = is_nchw_layout(legacy_model_);

    const auto &reshape = compute_library_->createReshape(precision_type);

    Status status = reshape->initialize(input_tensors, output_tensors, parameters);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != status, ENN_RET_INVAL, "CLReshape initialize is failed\n");

    operators->push_back(std::make_shared<EnnUDOperator<CLReshape>>(
        operation->get_name(), operation->get_id(), input_tensors, output_tensors, data_tensors, reshape));

    return ENN_RET_SUCCESS;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_STRIDED_SLICE>(
    const model::component::Operator::Ptr &operation) {
    ENN_DBG_PRINT("OpInfo: op_code: %d, name = %s, builtin_op_name = %s\n",
                  operation->get_code(),
                  operation->get_name().c_str(),
                  TFlite::EnumNameBuiltinOperator(operation->get_code()));
    CHECK_AND_RETURN_ERR(operation->get_code() != TFlite::BuiltinOperator::BuiltinOperator_STRIDED_SLICE,
                         ENN_RET_INVAL,
                         "Invalid BuiltinOperator: %d\n",
                         operation->get_code());

    auto ofm = std::static_pointer_cast<model::component::FeatureMap>(operation->out_tensors[0]);
    PrecisionType precision_type = get_precision((TFlite::TensorType)ofm->get_data_type());

    std::vector<std::shared_ptr<ITensor>> input_tensors, output_tensors, data_tensors;
    convert_to_tensors(operation, precision_type, input_tensors, output_tensors);
    CHECK_AND_RETURN_ERR(
        input_tensors.size() < 4, ENN_RET_INVAL, "CLStridedSlice must have input, begin, end, strides tensor\n");
    CHECK_AND_RETURN_ERR(
        input_tensors[1]->is_const() != true || input_tensors[2]->is_const() != true || input_tensors[3]->is_const() != true,
        ENN_RET_INVAL,
        "CLStridedSlice's begin, end, strides tensor must be const tensor\n");

    std::shared_ptr<StridedSliceParameters> parameters = std::make_shared<StridedSliceParameters>();
    auto options = reinterpret_cast<const TFlite::StridedSliceOptions *>(operation->get_option().get_addr());
    parameters->begin_mask = options->begin_mask();
    parameters->end_mask = options->end_mask();
    parameters->ellipsis_mask = options->ellipsis_mask();
    parameters->new_axis_mask = options->new_axis_mask();
    parameters->shrink_axis_mask = options->shrink_axis_mask();
#ifdef SCHEMA_NNC_V1
    parameters->androidNN = false;
#else
    parameters->androidNN = legacy_model_ == TFlite::LegacyModel::LegacyModel_ANDROID_NN;
#endif

    const auto &strided_slice = compute_library_->createStridedSlice(precision_type);

    Status status = strided_slice->initialize(input_tensors, output_tensors, parameters);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != status, ENN_RET_INVAL, "CLStridedSlice initialize is failed\n");

    operators->push_back(std::make_shared<EnnUDOperator<CLStridedSlice>>(
        operation->get_name(), operation->get_id(), input_tensors, output_tensors, data_tensors, strided_slice));

    return ENN_RET_SUCCESS;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_TRANSPOSE>(
    const model::component::Operator::Ptr &operation) {
    ENN_DBG_PRINT("OpInfo: op_code: %d, name = %s, builtin_op_name = %s\n",
                  operation->get_code(),
                  operation->get_name().c_str(),
                  TFlite::EnumNameBuiltinOperator(operation->get_code()));
    CHECK_AND_RETURN_ERR(operation->get_code() != TFlite::BuiltinOperator::BuiltinOperator_TRANSPOSE,
                         ENN_RET_INVAL,
                         "Invalid BuiltinOperator: %d\n",
                         operation->get_code());

    auto ofm = std::static_pointer_cast<model::component::FeatureMap>(operation->out_tensors[0]);
    PrecisionType precision_type = get_precision((TFlite::TensorType)ofm->get_data_type());

    std::vector<std::shared_ptr<ITensor>> input_tensors, output_tensors, data_tensors;
    convert_to_tensors(operation, precision_type, input_tensors, output_tensors);

    auto options = reinterpret_cast<const TFlite::TransposeOptions *>(operation->get_option().get_addr());
    UNUSED(options);  // TODO(all) : Need to remove the definination of TransposeOptions in the schema file.
    std::shared_ptr<TransposeParameters> parameters = std::make_shared<TransposeParameters>();
    parameters->perm.clear();
#ifdef SCHEMA_NNC_V1
    parameters->androidNN = false;
#else
    parameters->androidNN = legacy_model_ == TFlite::LegacyModel::LegacyModel_ANDROID_NN;
#endif

    const auto &transpose = compute_library_->createTranspose(precision_type);

    Status status = transpose->initialize(input_tensors, output_tensors, parameters);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != status, ENN_RET_INVAL, "CLTranspose initialize is failed\n");

    operators->push_back(std::make_shared<EnnUDOperator<CLTranspose>>(
        operation->get_name(), operation->get_id(), input_tensors, output_tensors, data_tensors, transpose));

    return ENN_RET_SUCCESS;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_PAD>(
    const model::component::Operator::Ptr &operation) {
    ENN_DBG_PRINT("OpInfo: op_code: %d, name = %s, builtin_op_name = %s\n",
                  operation->get_code(),
                  operation->get_name().c_str(),
                  TFlite::EnumNameBuiltinOperator(operation->get_code()));
    CHECK_AND_RETURN_ERR(operation->get_code() != TFlite::BuiltinOperator::BuiltinOperator_PAD,
                         ENN_RET_INVAL,
                         "Invalid BuiltinOperator: %d\n",
                         operation->get_code());

    auto ofm = std::static_pointer_cast<model::component::FeatureMap>(operation->out_tensors[0]);
    PrecisionType precision_type = get_precision((TFlite::TensorType)ofm->get_data_type());

    std::vector<std::shared_ptr<ITensor>> input_tensors, output_tensors, data_tensors;
    convert_to_tensors(operation, precision_type, input_tensors, output_tensors);

    CHECK_AND_RETURN_ERR(input_tensors.size() < 2, ENN_RET_INVAL, "CLPad must have padding tensor\n");

    std::shared_ptr<PadParameters> parameters = std::make_shared<PadParameters>();
    parameters->padding.clear();
    parameters->pad_value = 0.0f;
    parameters->quant_pad_value = 0;

#ifdef SCHEMA_NNC_V1
    parameters->androidNN = false;
#else
    auto options = reinterpret_cast<const TFlite::PadOptions *>(operation->get_option().get_addr());
    UNUSED(options);  // TODO(all) : Need to remove the definination of PadOptions in the schema file.
    parameters->androidNN = legacy_model_ == TFlite::LegacyModel::LegacyModel_ANDROID_NN;
#endif
    parameters->isNCHW = is_nchw_layout(legacy_model_);

    const auto &pad = compute_library_->createPad(precision_type);

    Status status = pad->initialize(input_tensors, output_tensors, parameters);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != status, ENN_RET_INVAL, "CLPad initialize is failed\n");

    operators->push_back(std::make_shared<EnnUDOperator<CLPad>>(
        operation->get_name(), operation->get_id(), input_tensors, output_tensors, data_tensors, pad));

    return ENN_RET_SUCCESS;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_CAST>(
    const model::component::Operator::Ptr &operation) {
    ENN_DBG_PRINT("OpInfo: op_code: %d, name = %s, builtin_op_name = %s\n",
                  operation->get_code(),
                  operation->get_name().c_str(),
                  TFlite::EnumNameBuiltinOperator(operation->get_code()));
    CHECK_AND_RETURN_ERR(operation->get_code() != TFlite::BuiltinOperator::BuiltinOperator_CAST,
                         ENN_RET_INVAL,
                         "Invalid BuiltinOperator: %d\n",
                         operation->get_code());

    auto ofm = std::static_pointer_cast<model::component::FeatureMap>(operation->out_tensors[0]);
    PrecisionType precision_type = get_precision((TFlite::TensorType)ofm->get_data_type());

    std::vector<std::shared_ptr<ITensor>> input_tensors, output_tensors, data_tensors;
    convert_to_tensors(operation, precision_type, input_tensors, output_tensors);

    auto options = reinterpret_cast<const TFlite::CastOptions *>(operation->get_option().get_addr());
    std::shared_ptr<CastParameters> parameters = std::make_shared<CastParameters>();
    parameters->in_data_type = compute_library_->tensor_type_2_data_type(options->in_data_type());
    parameters->out_data_type = compute_library_->tensor_type_2_data_type(options->out_data_type());
#ifdef SCHEMA_NNC_V1
    parameters->androidNN = false;
#else
    parameters->androidNN = legacy_model_ == TFlite::LegacyModel::LegacyModel_ANDROID_NN;
#endif

    const auto &cast = compute_library_->createCast(precision_type);

    Status status = cast->initialize(input_tensors, output_tensors, parameters);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != status, ENN_RET_INVAL, "CLCast initialize is failed\n");

    operators->push_back(std::make_shared<EnnUDOperator<CLCast>>(
        operation->get_name(), operation->get_id(), input_tensors, output_tensors, data_tensors, cast));

    return ENN_RET_SUCCESS;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_UNPACK>(
    const model::component::Operator::Ptr &operation) {
    ENN_DBG_PRINT("OpInfo: op_code: %d, name = %s, builtin_op_name = %s\n",
                  operation->get_code(),
                  operation->get_name().c_str(),
                  TFlite::EnumNameBuiltinOperator(operation->get_code()));
    CHECK_AND_RETURN_ERR(operation->get_code() != TFlite::BuiltinOperator::BuiltinOperator_UNPACK,
                         ENN_RET_INVAL,
                         "Invalid BuiltinOperator: %d\n",
                         operation->get_code());

    auto ofm = std::static_pointer_cast<model::component::FeatureMap>(operation->out_tensors[0]);
    PrecisionType precision_type = get_precision((TFlite::TensorType)ofm->get_data_type());

    std::vector<std::shared_ptr<ITensor>> input_tensors, output_tensors, data_tensors;
    convert_to_tensors(operation, precision_type, input_tensors, output_tensors);

    std::shared_ptr<UnpackParameters> parameters = std::make_shared<UnpackParameters>();
    auto options = reinterpret_cast<const TFlite::UnpackOptions *>(operation->get_option().get_addr());
    parameters->axis = options->axis();
#ifdef SCHEMA_NNC_V1
    parameters->num = options->num_split();
#else
    parameters->num = options->num();
#endif

    const auto &unpack = compute_library_->createUnpack(precision_type);

    Status status = unpack->initialize(input_tensors, output_tensors, parameters);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != status, ENN_RET_INVAL, "CLUnpack initialize is failed\n");

    operators->push_back(std::make_shared<EnnUDOperator<CLUnpack>>(
        operation->get_name(), operation->get_id(), input_tensors, output_tensors, data_tensors, unpack));

    return ENN_RET_SUCCESS;
}

#ifdef SCHEMA_NNC_V1
template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_TFLITE_SLICE>(
    const model::component::Operator::Ptr &operation) {
#else
template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_ENN_TFLITE_SLICE>(
    const model::component::Operator::Ptr &operation) {
#endif
    ENN_DBG_PRINT("OpInfo: op_code: %d, name = %s, builtin_op_name = %s\n",
                  operation->get_code(),
                  operation->get_name().c_str(),
                  TFlite::EnumNameBuiltinOperator(operation->get_code()));
    CHECK_AND_RETURN_ERR(
#ifdef SCHEMA_NNC_V1
        operation->get_code() != TFlite::BuiltinOperator::BuiltinOperator_TFLITE_SLICE,
#else
        operation->get_code() != TFlite::BuiltinOperator::BuiltinOperator_ENN_TFLITE_SLICE,
#endif
        ENN_RET_INVAL,
        "Invalid BuiltinOperator: %d\n",
        operation->get_code());

    auto ofm = std::static_pointer_cast<model::component::FeatureMap>(operation->out_tensors[0]);
    PrecisionType precision_type = get_precision((TFlite::TensorType)ofm->get_data_type());

    std::vector<std::shared_ptr<ITensor>> input_tensors, output_tensors, data_tensors;
    convert_to_tensors(operation, precision_type, input_tensors, output_tensors);
    CHECK_AND_RETURN_ERR(input_tensors.size() < 3, ENN_RET_INVAL, "CLTFSlice must have input/begin/size tensor\n");

    std::shared_ptr<TFSliceParameters> parameters = std::make_shared<TFSliceParameters>();
#ifdef SCHEMA_NNC_V1
    parameters->androidNN = false;
    auto options = reinterpret_cast<const TFlite::TFliteSliceOptions *>(operation->get_option().get_addr());
#else
    parameters->androidNN = legacy_model_ == TFlite::LegacyModel::LegacyModel_ANDROID_NN;
    auto options = reinterpret_cast<const TFlite::ENN_TFliteSliceOptions *>(operation->get_option().get_addr());
#endif
    UNUSED(options);  // TODO(all) : Need to remove the definination of TFliteSliceOptions/ENN_TFliteSliceOptions in the
                      // schema file.

    const auto &tfslice = compute_library_->createTFSlice(precision_type);
    Status status = tfslice->initialize(input_tensors, output_tensors, parameters);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != status, ENN_RET_INVAL, "CLTFSlice initialize is failed\n");

    operators->push_back(std::make_shared<EnnUDOperator<CLTFSlice>>(
        operation->get_name(), operation->get_id(), input_tensors, output_tensors, data_tensors, tfslice));

    return ENN_RET_SUCCESS;
}

#ifdef SCHEMA_NNC_V1
template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_SCALE>(
    const model::component::Operator::Ptr &operation) {
#else
template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_ENN_SCALE>(
    const model::component::Operator::Ptr &operation) {
#endif
    ENN_DBG_PRINT("OpInfo: op_code: %d, name = %s, builtin_op_name = %s\n",
                  operation->get_code(),
                  operation->get_name().c_str(),
                  TFlite::EnumNameBuiltinOperator(operation->get_code()));
    CHECK_AND_RETURN_ERR(
#ifdef SCHEMA_NNC_V1
        operation->get_code() != TFlite::BuiltinOperator::BuiltinOperator_SCALE,
#else
        operation->get_code() != TFlite::BuiltinOperator::BuiltinOperator_ENN_SCALE,
#endif
        ENN_RET_INVAL,
        "Invalid BuiltinOperator: %d\n",
        operation->get_code());

    PrecisionType precision_type = get_precision((TFlite::TensorType)operation->out_tensors[0]->get_data_type());

    std::vector<std::shared_ptr<ITensor>> input_tensors, output_tensors, data_tensors;
    convert_to_tensors(operation, precision_type, input_tensors, output_tensors);
    CHECK_AND_RETURN_ERR(!(input_tensors.size() == 3 || input_tensors.size() == 2),
                         ENN_RET_INVAL,
                         "CLScale must have these input tensors: input, scale and bias(optional)\n");

    const auto &scale = compute_library_->createScale(precision_type);
    Status status = scale->initialize(input_tensors, output_tensors, nullptr);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != status, ENN_RET_INVAL, "CLScale initialize is failed\n");

    operators->push_back(std::make_shared<EnnUDOperator<CLScale>>(
        operation->get_name(), operation->get_id(), input_tensors, output_tensors, data_tensors, scale));

    return ENN_RET_SUCCESS;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_SLICE>(
    const model::component::Operator::Ptr &operation) {
    ENN_DBG_PRINT("OpInfo: op_code: %d, name = %s, builtin_op_name = %s\n",
                  operation->get_code(),
                  operation->get_name().c_str(),
                  TFlite::EnumNameBuiltinOperator(operation->get_code()));
    CHECK_AND_RETURN_ERR(operation->get_code() != TFlite::BuiltinOperator::BuiltinOperator_SLICE,
                         ENN_RET_INVAL,
                         "Invalid BuiltinOperator: %d\n",
                         operation->get_code());

    PrecisionType precision_type = get_precision((TFlite::TensorType)operation->out_tensors[0]->get_data_type());

    std::vector<std::shared_ptr<ITensor>> input_tensors, output_tensors, data_tensors;
    convert_to_tensors(operation, precision_type, input_tensors, output_tensors);

    std::shared_ptr<SliceParameters> parameters = std::make_shared<SliceParameters>();
    auto options = reinterpret_cast<const TFlite::SliceOptions *>(operation->get_option().get_addr());
    parameters->axis = options->axis();
    for (int i = 0; i < options->slice_point()->size(); i++) {
        parameters->slice_point.emplace_back(options->slice_point()->data()[i]);
    }

    const auto &slice = compute_library_->createSlice(precision_type);
    Status status = slice->initialize(input_tensors, output_tensors, parameters);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != status, ENN_RET_INVAL, "CLSlice initialize is failed\n");

    operators->push_back(std::make_shared<EnnUDOperator<CLSlice>>(
        operation->get_name(), operation->get_id(), input_tensors, output_tensors, data_tensors, slice));

    return ENN_RET_SUCCESS;
}

/***************************************************************************************************************************
 * public function                                                                           *
 ***************************************************************************************************************************/
#define CREATE_UD_OPERATION(type) {type, &OperationConstructor::create_ud_operator<type>}
OperationConstructor::OperationConstructor(std::shared_ptr<CLComputeLibrary> compute_library) :
    compute_library_(compute_library), operator_list_id_(0), relax_computation_float32_to_float16_(true),
    legacy_model_(TFlite::LegacyModel::LegacyModel_CAFFE), storage_type_(StorageType::BUFFER) {
    builtin_op_map_ = {
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_ADD),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_AVERAGE_POOL_2D),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_BIDIRECTIONAL_SEQUENCE_LSTM),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_CAST),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_CONCATENATION),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_CONV_2D),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_DEPTH_TO_SPACE),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_DEPTHWISE_CONV_2D),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_DIV),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_FULLY_CONNECTED),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_GATHER),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_LOGISTIC),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_MAX_POOL_2D),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_MEAN),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_MUL),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_PAD),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_REDUCE_MIN),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_RELU),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_RELU_N1_TO_1),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_RELU6),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_RESHAPE),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_RESIZE_BILINEAR),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_SLICE),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_SOFTMAX),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_SPLIT),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_SQUEEZE),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_STRIDED_SLICE),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_SUB),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_TANH),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_TRANSPOSE),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_TRANSPOSE_CONV),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_UNPACK),
#ifdef SCHEMA_NNC_V1
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_TFLITE_SLICE),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_SCALE),
#else
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_ENN_NORMALIZATION),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_ENN_TFLITE_SLICE),
        CREATE_UD_OPERATION(TFlite::BuiltinOperator::BuiltinOperator_ENN_SCALE),
#endif
    };

    custom_op_map_ = {
        {"Normalization", &OperationConstructor::create_ud_operator<CustomOperator::CustomOperator_Normalization>},
    };
}

OperationConstructor::~OperationConstructor() {
    builtin_op_map_.clear();
    custom_op_map_.clear();

    for (auto &operator_list_tensors_map : alloc_tensors_map_) {
        for (auto &tensor : operator_list_tensors_map.second) {
            tensor.second.reset();
        }
        operator_list_tensors_map.second.clear();
    }
    alloc_tensors_map_.clear();
    tensors_used_map_.clear();
    id_input_op_.clear();
    id_output_op_.clear();
}

template <CustomOperator custom_op>
EnnReturn OperationConstructor::create_ud_operator(const model::component::Operator::Ptr &operator_) {
    ENN_DBG_PRINT("opr->name = %s (not support yet)\n", operator_->get_name().c_str());
    return ENN_RET_SUCCESS;
}

EnnReturn OperationConstructor::create_ud_operator(const model::component::Operator::Ptr &operator_) {
    ENN_DBG_PRINT("OperationConstructor::create_ud_operator opr->code = %d\n", operator_->get_code());
    if (is_builtin_operator(operator_->get_code())) {
        const auto &it = builtin_op_map_.find(operator_->get_code());
        if (it == builtin_op_map_.end()) {
            ENN_ERR_PRINT("Not found operation code : %d\n", operator_->get_code());
            return ENN_RET_INVAL;
        }

        auto &func = it->second;
        return (this->*func)(operator_);
    } else if (!operator_->get_name().empty()) {
        // Custom Operation has name
        const auto &it = custom_op_map_.find(operator_->get_name());

        if (it == custom_op_map_.end()) {
            ENN_ERR_PRINT("Not found operation name : %s\n", operator_->get_name().c_str());
            return ENN_RET_INVAL;
        }

        auto &func = it->second;

        return (this->*func)(operator_);
    }

    return ENN_RET_FAILED;
}

EnnReturn OperationConstructor::open_oplist(const model::component::OperatorList &operator_list) {
    std::lock_guard<std::mutex> lock_guard(mutex_constructor_);

    // set oplist_id
    operator_list_id_ = operator_list.get_id().get();
    ENN_DBG_PRINT("open_oplist_id = 0x%" PRIx64 "\n", operator_list_id_);

    // set attribute
    auto attribute = operator_list.get_attribute();
    relax_computation_float32_to_float16_ = attribute.get_relax_computation_float32_to_float16();
    legacy_model_ = attribute.get_legacy_model();
    ENN_DBG_PRINT("relax_computation_float32_to_float16_: %d, legacy_model_: %d \n",
                  relax_computation_float32_to_float16_,
                  legacy_model_);

    // initialize inter buffer and buffer index set
    for (auto &&iop : operator_list) {
        const auto op = std::static_pointer_cast<enn::model::component::Operator>(iop);
        init_inter_buffer(op);
        for (auto &in_tensor : op->in_tensors) {
            if (!in_tensor->is_const()) {
                auto ifm = std::static_pointer_cast<model::component::FeatureMap>(in_tensor);
                int32_t buffer_index = ifm->get_buffer_index();
                id_input_op_.emplace(buffer_index);
            }
        }
        for (auto &out_tensor : op->out_tensors) {
            auto ifm = std::static_pointer_cast<model::component::FeatureMap>(out_tensor);
            int32_t buffer_index = ifm->get_buffer_index();
            id_output_op_.emplace(buffer_index);
        }
    }

    // create ud_operators
    for (auto &&iop : operator_list) {
        const auto op = std::static_pointer_cast<enn::model::component::Operator>(iop);
        EnnReturn ret = create_ud_operator(std::static_pointer_cast<enn::model::component::Operator>(op));
        if (ret != ENN_RET_SUCCESS) {
            return ret;
        }
    }

    // allocate inter&intra buffer
    compute_library_->assignBuffers();

    // clFinish
    compute_library_->synchronize();

    // clear intermediate data
    tensors_used_map_.clear();
    id_input_op_.clear();
    id_output_op_.clear();
    return ENN_RET_SUCCESS;
}

void OperationConstructor::close_oplist(const uint64_t &oplist_id) {
    ENN_DBG_PRINT("close_oplist = 0x%" PRIx64 "\n", oplist_id);
    ENN_DBG_PRINT("close_oplist size before: %zd \n", alloc_tensors_map_.size());

    std::lock_guard<std::mutex> lock_guard(mutex_constructor_);
    if (alloc_tensors_map_.find(oplist_id) != alloc_tensors_map_.end()) {
        for (auto &tensor : alloc_tensors_map_.at(oplist_id)) {
            tensor.second.reset();
        }
        alloc_tensors_map_.erase(oplist_id);
    }
    ENN_DBG_PRINT("close_oplist end: %zd \n", alloc_tensors_map_.size());
}

/***************************************************************************************************************************
 * privite function                                                                           *
 ***************************************************************************************************************************/
bool OperationConstructor::check_dim(const NDims &dims) {
    if (dims.size() == 0 || dims.size() > 4) {
        ENN_ERR_PRINT("Not supported dims : %zd\n", dims.size());
        return false;
    }

    for (uint32_t i = 0; i < dims.size(); ++i) {
        if (dims[i] < 0) {
            ENN_ERR_PRINT("The %dth is invalid dim : %d\n", i, dims[i]);
            return false;
        }
    }
    return true;
}

bool OperationConstructor::is_nchw_layout(const TFlite::LegacyModel &legacy) {
    switch (legacy) {
    case TFlite::LegacyModel_TENSORFLOW_NHWC:
    case TFlite::LegacyModel_CAFFE_NHWC: return false;

    default: return true;
    }
}

PrecisionType OperationConstructor::get_precision(const TFlite::TensorType &dataType) {
    ENN_DBG_PRINT("get_precision\n");
    PrecisionType precision = PrecisionType::FP16;
    switch (dataType) {
    case TFlite::TensorType::TensorType_INT8: precision = PrecisionType::INT8; break;
    case TFlite::TensorType::TensorType_UINT8: precision = PrecisionType::UINT8; break;
    case TFlite::TensorType::TensorType_FLOAT32:
        precision = relax_computation_float32_to_float16_ ? PrecisionType::FP16 : PrecisionType::FP32;
        break;
    default: break;
    }

    ENN_DBG_PRINT("precision is: %d\n", precision);
    return precision;
}

void OperationConstructor::get_padding(TFlite::Padding paddingType,
                                       Pad4 &padding,
                                       NDims inputDim,
                                       NDims outputDim,
                                       NDims weightDim,
                                       Dim2 stride,
                                       Dim2 dilation,
                                       bool deconv,
                                       bool nchw) {
    ENN_DBG_PRINT("OperationConstructor::get_padding() is called\n");
    padding.l = 0;
    padding.r = 0;
    padding.t = 0;
    padding.b = 0;
    if (paddingType == TFlite::Padding_SAME) {
        int32_t input_width = 0, input_height = 0;
        int32_t output_width = 0, output_height = 0;
        int32_t filter_width = 0, filter_height = 0;
        if (!nchw) {
            input_height = inputDim[H_NHWC];
            input_width = inputDim[W_NHWC];
            output_height = outputDim[H_NHWC];
            output_width = outputDim[W_NHWC];
            filter_height = weightDim[H_NHWC];
            filter_width = weightDim[W_NHWC];
        } else {
            input_height = inputDim[H_NCHW];
            input_width = inputDim[W_NCHW];
            output_height = outputDim[H_NCHW];
            output_width = outputDim[W_NCHW];
            filter_height = weightDim[H_NCHW];
            filter_width = weightDim[W_NCHW];
            if (legacy_model_ ==
                TFlite::LegacyModel_ANDROID_NN) {  // kernel shape is slways [Cout,Kh,Kw,Cin] in androidnn model
                filter_height = weightDim[C_NCHW];
                filter_width = weightDim[H_NCHW];
            }
        }

        int32_t out_size_width, out_size_height;
        int32_t needed_input_width, needed_input_height;
        int32_t total_padding_width, total_padding_height;
        if (deconv) {  // for deconvolution
            auto tempWidth = output_width;
            output_width = input_width;
            input_width = tempWidth;

            auto tempHeight = output_height;
            output_height = input_height;
            input_height = tempHeight;
        }
        out_size_width = output_width;
        out_size_height = output_height;

        // added for dilation
        filter_width = dilation.h * (filter_width - 1) + 1;
        filter_height = dilation.w * (filter_height - 1) + 1;

        ENN_DBG_PRINT("input: %d %d, output: %d %d, filter: %d %d, dilation: %d %d, stride: %d %d, deconv: %d nchw :%d\n",
                      input_height,
                      input_width,
                      output_height,
                      output_width,
                      filter_height,
                      filter_width,
                      dilation.h,
                      dilation.w,
                      stride.h,
                      stride.w,
                      deconv,
                      nchw);

        needed_input_width = (out_size_width - 1) * stride.w + filter_width;
        needed_input_height = (out_size_height - 1) * stride.h + filter_height;
        if (legacy_model_ == TFlite::LegacyModel::LegacyModel_ANDROID_NN && deconv && needed_input_height < input_height) {
            padding.r = (needed_input_width - input_width) - padding.l;
            padding.b = (needed_input_height - input_height) - padding.t;
        } else {
            total_padding_width = std::max(0, (needed_input_width - input_width));
            total_padding_height = std::max(0, (needed_input_height - input_height));
            if (legacy_model_ == TFlite::LegacyModel::LegacyModel_CAFFE ||
                legacy_model_ == TFlite::LegacyModel::LegacyModel_CAFFE_NHWC ||
                legacy_model_ == TFlite::LegacyModel::LegacyModel_CAFFE_NCHW) {
                // To match golden result that of caffe, ceiling is applied on padding.
                padding.l = (total_padding_width + 1) / 2;
                padding.r = padding.l;
                padding.t = (total_padding_height + 1) / 2;
                padding.b = padding.t;
            } else {
                // Below calculation comes from https://developer.android.com/ndk/reference/group/neural-networks
                // But, our tflite is generated from caffe model which uses same left/right and top/bottom.
                // So this won't be used to match golden result.
                padding.l = total_padding_width / 2;
                padding.t = total_padding_height / 2;
                padding.r = (total_padding_width + 1) / 2;
                padding.b = (total_padding_height + 1) / 2;
            }
        }
    }
    ENN_DBG_PRINT("padding.l padding.r padding.t padding.b: %d %d %d %d\n", padding.l, padding.r, padding.t, padding.b);
}

void OperationConstructor::get_perchannel_quant_info(const std::shared_ptr<enn::model::component::Tensor> &edge,
                                                     bool &per_channel_quant,
                                                     std::vector<float> &scale) {
    per_channel_quant = false;
    auto quant_info = edge->get_symm_per_channel_quant_parameters();
    if (quant_info) {
        per_channel_quant = true;
        for (size_t i = 0; i < quant_info->scales()->size(); ++i) {
            scale.push_back(static_cast<float>(quant_info->scales()->Get(i)));
        }
        ENN_DBG_PRINT("perchannel_quant model\n");
    }
}

void OperationConstructor::get_edge_info(const std::shared_ptr<enn::model::component::Tensor> &edge,
                                         NDims &dims,
                                         TFlite::TensorType &data_type,
                                         float &scale,
                                         int32_t &zero_point) {
    dims = edge->get_shape();
    data_type = (TFlite::TensorType)edge->get_data_type();
    auto quant_info = edge->get_quantization_parameters();
    if (quant_info) {
        scale = quant_info->scale()->Get(0);
        zero_point = quant_info->zero_point()->Get(0);
    } else {
        scale = 1.0f;
        zero_point = 0;
    }
    ENN_DBG_PRINT("edge: %s(id:%d), is_const:%d, shape: \n", edge->get_name().c_str(), edge->get_id(), edge->is_const());
    for (auto shape : dims) {
        ENN_DBG_PRINT("%d ", shape);
    }
    ENN_DBG_PRINT(" \n ");
    ENN_DBG_PRINT("data_type:%d, scale:%f, zero_point: %d\n", data_type, scale, zero_point);
}

void OperationConstructor::init_inter_buffer(const std::shared_ptr<model::component::Operator> &operator_) {
    ENN_DBG_PRINT("op_id: 0x%" PRIX64 " op_name %s\n", operator_->get_id(), operator_->get_name().c_str());
    for (auto in_tensor : operator_->in_tensors) {
        if (!in_tensor->is_const()) {  // map buffer_index of input to operator
            auto ifm = std::static_pointer_cast<model::component::FeatureMap>(in_tensor);
            uint32_t buffer_index = ifm->get_buffer_index();
            if (tensors_used_map_.find(buffer_index) == tensors_used_map_.end()) {
                tensors_used_map_[buffer_index] = 1;
            } else {
                tensors_used_map_[buffer_index]++;
            }
        }
    }
}

std::shared_ptr<ITensor> OperationConstructor::allocate_tensor(const std::shared_ptr<enn::model::component::Tensor> &edge,
                                                               const PrecisionType &precision,
                                                               const BufferType &buffer_type,
                                                               const DataOrder &data_order,
                                                               const bool &use_fp32_for_fp16,
                                                               const bool &use_cpu_for_fp16) {
    ENN_UNUSED(use_cpu_for_fp16);
    NDims dims;
    TFlite::TensorType data_type;
    float scale;
    int32_t zero_point;
    get_edge_info(edge, dims, data_type, scale, zero_point);

    if (!check_dim(dims)) {
        ENN_ERR_PRINT("allocate feature map tensor failed \n");
        return nullptr;
    }

    std::shared_ptr<ITensor> tensor = nullptr;
    PrecisionType precision_type = precision;
    if (use_fp32_for_fp16 &&
        (data_type == TFlite::TensorType::TensorType_FLOAT32 && precision_type == PrecisionType::FP16)) {
        ENN_ERR_PRINT(" use_FP32_for_fp16 \n");
        precision_type = PrecisionType::FP32;
    }

    if (!edge->is_const()) {  // if tensor is not const, the tensor is feature map.
        auto ifm = std::static_pointer_cast<model::component::FeatureMap>(edge);
        ENN_DBG_PRINT("index_buffer: %d, type: %d", ifm->get_buffer_index(), ifm->get_type());
        int32_t buffer_index = ifm->get_buffer_index();
        if (alloc_tensors_map_.find(operator_list_id_) == alloc_tensors_map_.end() ||
            alloc_tensors_map_.at(operator_list_id_).find(buffer_index) == alloc_tensors_map_.at(operator_list_id_).end()) {
            tensor = compute_library_->create_tensor(data_type,
                                                     precision_type,
                                                     dims,
                                                     buffer_index,
                                                     buffer_type,
                                                     use_fp32_for_fp16,
                                                     storage_type_,
                                                     data_order,
                                                     scale,
                                                     zero_point);

            if (tensor == nullptr) {
                ENN_ERR_PRINT(" allocate fm tensor failed \n");
                return nullptr;
            }

            if (alloc_tensors_map_.find(operator_list_id_) == alloc_tensors_map_.end()) {  // initialize tensor map
                alloc_tensors_map_[operator_list_id_][buffer_index] = tensor;
            } else {
                alloc_tensors_map_.at(operator_list_id_)[buffer_index] = tensor;
            }

            if (id_output_op_.find(buffer_index) == id_output_op_.end()) {
                ENN_DBG_PRINT("oplist input %d \n", buffer_index);
                in_tensors.push_back(tensor);
            } else if (id_input_op_.find(buffer_index) == id_input_op_.end()) {
                ENN_DBG_PRINT("oplist output %d \n", buffer_index);
                out_tensors.push_back(tensor);
            }
            return tensor;
        } else {
            return alloc_tensors_map_.at(operator_list_id_).at(buffer_index);
        }

    } else {
        auto param = std::static_pointer_cast<model::component::Parameter>(edge);
        if ((0 == param->get_buffer_size()) || (nullptr == param->get_buffer_addr())) {
            ENN_ERR_PRINT("allocate tensor = nullptr : %s\n", edge->get_name().c_str());
            return nullptr;
        }

        auto data = const_cast<DataPtr>(param->get_buffer_addr());
        tensor = compute_library_->create_and_copy_tensor(data_type,
                                                          data,
                                                          precision_type,
                                                          dims,
                                                          UNDEFINED,
                                                          buffer_type,
                                                          use_fp32_for_fp16,
                                                          storage_type_,
                                                          data_order,
                                                          scale,
                                                          zero_point);

        if (tensor == nullptr) {
            ENN_ERR_PRINT(" allocate const tensor failed \n");
        }
    }

    return tensor;
}

void OperationConstructor::convert_to_tensors(const model::component::Operator::Ptr &operator_,
                                              const PrecisionType &precision_type,
                                              std::vector<std::shared_ptr<ITensor>> &in_tensors,
                                              std::vector<std::shared_ptr<ITensor>> &out_tensors,
                                              const bool &use_fp32_for_fp16,
                                              const bool &use_cpu_for_fp16) {
    for (auto &out_tensor : operator_->out_tensors) {
        ENN_DBG_PRINT("create output tensor\n");
        std::shared_ptr<ITensor> tensor = allocate_tensor(
            out_tensor, precision_type, use_cpu_for_fp16 ? BufferType::DEDICATED : BufferType::INTER_SHARED_REUSE);
        if (tensor != nullptr) {
            out_tensors.push_back(tensor);
        } else {
            ENN_ERR_PRINT("allocate tensor failed : %s\n", out_tensor->get_name().c_str());
        }
    }
    for (auto &in_tensor : operator_->in_tensors) {
        ENN_DBG_PRINT("create input tensor\n");
        std::shared_ptr<ITensor> tensor = allocate_tensor(
            in_tensor, precision_type, use_fp32_for_fp16 ? BufferType::DEDICATED : BufferType::INTER_SHARED_NEW);
        in_tensors.push_back(tensor);

        if (!in_tensor->is_const() && tensor != nullptr) {  // InterBuffer reuse
            auto ifm = std::static_pointer_cast<model::component::FeatureMap>(in_tensor);
            int32_t buffer_index = ifm->get_buffer_index();
            if (tensors_used_map_.find(buffer_index) != tensors_used_map_.end() && tensors_used_map_.at(buffer_index) > 0) {
                tensors_used_map_.at(buffer_index)--;
                if (tensors_used_map_.at(buffer_index) == 0) {
                    ENN_DBG_PRINT("inter buffer reuse %d \n", buffer_index);
                    tensor->resetInterBuffer();
                }
            }
        }
    }
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
