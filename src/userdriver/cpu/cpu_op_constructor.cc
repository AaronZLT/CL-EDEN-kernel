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

#include "common/enn_debug.h"
#include "common/helper_templates.hpp"
#include "userdriver/cpu/cpu_op_constructor.h"
#include "model/component/tensor/feature_map.hpp"
#include "model/component/tensor/parameter.hpp"
#include "userdriver/cpu/common/NEONComputeLibrary.h"
#include "userdriver/common/op_test/test_capabilities.h"

namespace enn {
namespace ud {
namespace cpu {

/***************************************************************************************************************************
 * create_ud_operator() of Builtin Operators                                                                               *
 ***************************************************************************************************************************/
template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_SOFTMAX>(
    const model::component::Operator::Ptr &operator_) {
    ENN_DBG_PRINT("opr->code = %d\n", operator_->get_code());

    std::string builtin_op_name = TFlite::EnumNameBuiltinOperator(operator_->get_code());
    if (builtin_op_name.compare(operator_->get_name()) != 0) {
        ENN_ERR_PRINT("Invalid BuiltinOperator : %s\n", builtin_op_name.c_str());
        return ENN_RET_INVAL;
    }

    if (operator_->get_option().get_enum() != TFlite::BuiltinOptions_SoftmaxOptions) {
        ENN_ERR_PRINT("Invalid BuiltinOptions : %s\n", TFlite::EnumNameBuiltinOptions(operator_->get_option().get_enum()));
        return ENN_RET_INVAL;
    }

    if (operator_->get_option().get_addr() == nullptr) {
        ENN_ERR_PRINT("Invalid BuiltinOptions (null)\n");
        return ENN_RET_INVAL;
    }

    PrecisionType precision_type = PrecisionType::FP32;

    std::vector<std::shared_ptr<ITensor>> in_tensors, out_tensors, data_tensors;

    convert_to_tensors(operator_, precision_type, in_tensors, out_tensors, data_tensors);

    float beta = 0.f;
    int32_t axis = 0;

    if (operator_->get_option().get_size() == sizeof(TC_SoftmaxOptions)) {
        auto options = (TC_SoftmaxOptions *)(operator_->get_option().get_addr());
        beta = options->beta();
        axis = options->axis();
    } else {
        auto options = reinterpret_cast<const TFlite::SoftmaxOptions *>(operator_->get_option().get_addr());
        beta = options->beta();
        axis = options->axis();
    }

    int32_t number = in_tensors[0]->getDim().n;
    int32_t channel = in_tensors[0]->getDim().c;
    int32_t height = in_tensors[0]->getDim().h;
    int32_t width = in_tensors[0]->getDim().w;

    const auto &softmax = compute_library->createSoftmax(precision_type);

    Status ret_value = softmax->initialize(in_tensors[0], width, height, channel, number, beta, axis);

    operators->push_back(std::make_shared<EnnUDOperator<ISoftmax>>(operator_->get_name(), operator_->get_id(), in_tensors,
                                                                   out_tensors, data_tensors, softmax));

    return (ret_value == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

#ifndef SCHEMA_NNC_V1
template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_DEQUANTIZE>(
    const model::component::Operator::Ptr &operator_) {
    ENN_DBG_PRINT("opr->code = %d\n", operator_->get_code());

    std::string builtin_op_name = TFlite::EnumNameBuiltinOperator(operator_->get_code());
    if (builtin_op_name.compare(operator_->get_name()) != 0) {
        ENN_ERR_PRINT("Invalid BuiltinOperator : %s\n", builtin_op_name.c_str());
        return ENN_RET_INVAL;
    }

    if (operator_->get_option().get_enum() != TFlite::BuiltinOptions_DequantizeOptions) {
        ENN_ERR_PRINT("Invalid BuiltinOptions : %s\n", TFlite::EnumNameBuiltinOptions(operator_->get_option().get_enum()));
        return ENN_RET_INVAL;
    }

    if (operator_->get_option().get_addr() == nullptr) {
        ENN_ERR_PRINT("Invalid BuiltinOptions (null)\n");
        return ENN_RET_INVAL;
    }

    Status ret_value = Status::SUCCESS;

    PrecisionType precision_type = PrecisionType::FP32;

    std::vector<std::shared_ptr<ITensor>> in_tensors, out_tensors, data_tensors;

    convert_to_tensors(operator_, precision_type, in_tensors, out_tensors, data_tensors);

    uint32_t type;
    std::vector<int32_t> fractional_length;
    float scale;
    int32_t zero_point;

    if (operator_->get_option().get_size() == sizeof(TC_DequantizeOptions)) {
        auto options = (TC_DequantizeOptions *)(operator_->get_option().get_addr());
        type = options->type();
        if (type == TFlite::QuantType_SYMM) {
            fractional_length = options->fractional_length();
        } else {
            scale = options->scale_out();
            zero_point = options->zero_point_output();
        }
    } else {
        auto options = reinterpret_cast<const TFlite::DequantizeOptions *>(operator_->get_option().get_addr());
        type = options->type();
        if (type == TFlite::QuantType_SYMM) {
            fractional_length = util::convert_vector<int32_t>(options->fractional_length());
        } else {
            scale = options->scale_out()->Get(0);
            zero_point = options->zero_point_output()->Get(0);
        }
    }

    if (type == TFlite::QuantType_SYMM) {
        uint32_t img_size = in_tensors[0]->getDim().h * in_tensors[0]->getDim().w;
        int32_t data_num = in_tensors[0]->getDim().c * img_size;
        auto data = static_cast<DataPtr>(&fractional_length[0]);
        std::shared_ptr<ITensor> frac_tensor = create_and_copy_tensor(
            TFlite::TensorType_INT32, data, get_data_size(TFlite::TensorType_INT32, fractional_length.size()));

        const auto &dequantization = compute_library->createDequantization(precision_type);

        ret_value = dequantization->initialize(in_tensors[0], data_num, frac_tensor, img_size);

        operators->push_back(std::make_shared<EnnUDOperator<IDequantization>>(
            operator_->get_name(), operator_->get_id(), in_tensors, out_tensors, data_tensors, dequantization));
    } else {
        int32_t channel = in_tensors[0]->getDim().c;
        int32_t height = in_tensors[0]->getDim().h;
        int32_t width = in_tensors[0]->getDim().w;
        int32_t num_data = channel * height * width;
        uint32_t img_size = height * width;

        const auto &asymm_dequantize = compute_library->createAsymmDequantization(precision_type);

        ret_value = asymm_dequantize->initialize(in_tensors[0], num_data, scale, zero_point, img_size);

        operators->push_back(std::make_shared<EnnUDOperator<IAsymmDequantization>>(
            operator_->get_name(), operator_->get_id(), in_tensors, out_tensors, data_tensors, asymm_dequantize));
    }

    return (ret_value == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_QUANTIZE>(
    const model::component::Operator::Ptr &operator_) {
    ENN_DBG_PRINT("opr->code = %d\n", operator_->get_code());

    std::string builtin_op_name = TFlite::EnumNameBuiltinOperator(operator_->get_code());
    if (builtin_op_name.compare(operator_->get_name()) != 0) {
        ENN_ERR_PRINT("Invalid BuiltinOperator : %s\n", builtin_op_name.c_str());
        return ENN_RET_INVAL;
    }

    if (operator_->get_option().get_enum() != TFlite::BuiltinOptions_QuantizeOptions) {
        ENN_ERR_PRINT("Invalid BuiltinOptions : %s\n", TFlite::EnumNameBuiltinOptions(operator_->get_option().get_enum()));
        return ENN_RET_INVAL;
    }

    if (operator_->get_option().get_addr() == nullptr) {
        ENN_ERR_PRINT("Invalid BuiltinOptions (null)\n");
        return ENN_RET_INVAL;
    }

    Status ret_value = Status::SUCCESS;

    PrecisionType precision_type = PrecisionType::FP32;

    std::vector<std::shared_ptr<ITensor>> in_tensors, out_tensors, data_tensors;

    convert_to_tensors(operator_, precision_type, in_tensors, out_tensors, data_tensors);

    int32_t channel = in_tensors[0]->getDim().c;
    int32_t height = in_tensors[0]->getDim().h;
    int32_t width = in_tensors[0]->getDim().w;

    uint32_t type;
    std::vector<int32_t> fractional_length;
    float scale;
    int32_t zero_point;

    if (operator_->get_option().get_size() == sizeof(TC_DequantizeOptions)) {
        auto options = (TC_DequantizeOptions *)(operator_->get_option().get_addr());
        type = options->type();
        if (type == TFlite::QuantType_SYMM) {
            fractional_length = options->fractional_length();
        } else {
            scale = options->scale_out();
            zero_point = options->zero_point_output();
        }
    } else {
        auto options = reinterpret_cast<const TFlite::QuantizeOptions *>(operator_->get_option().get_addr());
        type = options->type();
        if (type == TFlite::QuantType_SYMM) {
            fractional_length = util::convert_vector<int32_t>(options->fractional_length());
        } else {
            scale = options->scale_out()->Get(0);
            zero_point = options->zero_point_output()->Get(0);
        }
    }

    if (type == TFlite::QuantType_SYMM) {
        auto data = static_cast<DataPtr>(&fractional_length[0]);
        std::shared_ptr<ITensor> frac_tensor = create_and_copy_tensor(
            TFlite::TensorType_INT32, data, get_data_size(TFlite::TensorType_INT32, fractional_length.size()));

        const auto &quantization = compute_library->createQuantization(precision_type);

        ret_value = quantization->initialize(in_tensors[0], channel, height, width, frac_tensor);

        operators->push_back(std::make_shared<EnnUDOperator<IQuantization>>(
            operator_->get_name(), operator_->get_id(), in_tensors, out_tensors, data_tensors, quantization));
    } else {
        const auto &asymm_quantize = compute_library->createAsymmQuantization(precision_type);

        ret_value = asymm_quantize->initialize(in_tensors[0], channel, width, height, scale, zero_point);

        operators->push_back(std::make_shared<EnnUDOperator<IAsymmQuantization>>(
            operator_->get_name(), operator_->get_id(), in_tensors, out_tensors, data_tensors, asymm_quantize));
    }

    return (ret_value == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_ENN_CFU>(
    const model::component::Operator::Ptr &operator_) {
    ENN_DBG_PRINT("opr->code = %d\n", operator_->get_code());

    std::string builtin_op_name = TFlite::EnumNameBuiltinOperator(operator_->get_code());
    if (builtin_op_name.compare(operator_->get_name()) != 0) {
        ENN_ERR_PRINT("Invalid BuiltinOperator : %s\n", builtin_op_name.c_str());
        return ENN_RET_INVAL;
    }

    if (operator_->get_option().get_enum() != TFlite::BuiltinOptions_ENN_CFUOptions) {
        ENN_ERR_PRINT("Invalid BuiltinOptions : %s\n", TFlite::EnumNameBuiltinOptions(operator_->get_option().get_enum()));
        return ENN_RET_INVAL;
    }

    if (operator_->get_option().get_addr() == nullptr) {
        ENN_ERR_PRINT("Invalid BuiltinOptions (null)\n");
        return ENN_RET_INVAL;
    }

    PrecisionType precision_type = PrecisionType::FP32;

    std::vector<std::shared_ptr<ITensor>> in_tensors, out_tensors, data_tensors;

    convert_to_tensors(operator_, precision_type, in_tensors, out_tensors, data_tensors);

    std::string soc_name;
#ifdef VELOCE_SOC
    soc_name = enn::platform::VELOCE_EXYNOS9925;
#else
    soc_name = enn::platform::EXYNOS9925;
#endif
    int32_t channel = in_tensors[0]->getDim().c;
    int32_t height = in_tensors[0]->getDim().h;
    int32_t width = in_tensors[0]->getDim().w;

    auto options = reinterpret_cast<const TFlite::ENN_InverseCFUOptions *>(operator_->get_option().get_addr());
    int32_t cols_in_cell = options->cols_in_cell();
    int32_t lines_in_cell = options->lines_in_cell();
    int32_t interleaved_slices = options->interleaved_slices();

    const auto &cfu_converter = compute_library->createCFUConverter(precision_type, soc_name);

    Status ret_value = cfu_converter->initialize(in_tensors[0], width, height, channel, cols_in_cell, lines_in_cell,
                                                 interleaved_slices);

    operators->push_back(std::make_shared<EnnUDOperator<ICFUConverter>>(
        operator_->get_name(), operator_->get_id(), in_tensors, out_tensors, data_tensors, cfu_converter));

    return (ret_value == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_ENN_INVERSE_CFU>(
    const model::component::Operator::Ptr &operator_) {
    ENN_DBG_PRINT("opr->code = %d\n", operator_->get_code());

    std::string builtin_op_name = TFlite::EnumNameBuiltinOperator(operator_->get_code());
    if (builtin_op_name.compare(operator_->get_name()) != 0) {
        ENN_ERR_PRINT("Invalid BuiltinOperator : %s\n", builtin_op_name.c_str());
        return ENN_RET_INVAL;
    }

    if (operator_->get_option().get_enum() != TFlite::BuiltinOptions_ENN_InverseCFUOptions) {
        ENN_ERR_PRINT("Invalid BuiltinOptions : %s\n", TFlite::EnumNameBuiltinOptions(operator_->get_option().get_enum()));
        return ENN_RET_INVAL;
    }

    if (operator_->get_option().get_addr() == nullptr) {
        ENN_ERR_PRINT("Invalid BuiltinOptions (null)\n");
        return ENN_RET_INVAL;
    }

    PrecisionType precision_type = PrecisionType::FP32;

    std::vector<std::shared_ptr<ITensor>> in_tensors, out_tensors, data_tensors;

    convert_to_tensors(operator_, precision_type, in_tensors, out_tensors, data_tensors);

    std::string soc_name;
#ifdef VELOCE_SOC
    soc_name = enn::platform::VELOCE_EXYNOS9925;
#else
    soc_name = enn::platform::EXYNOS9925;
#endif

    int32_t channel = in_tensors[0]->getDim().c;
    int32_t height = in_tensors[0]->getDim().h;
    int32_t width = in_tensors[0]->getDim().w;
    int32_t cols_in_cell = 0;
    int32_t lines_in_cell = 0;
    int32_t interleaved_slices = 0;

    if (operator_->get_option().get_size() == sizeof(TC_InverseCFUOptions)) {
        auto options = (TC_InverseCFUOptions *)(operator_->get_option().get_addr());
        cols_in_cell = options->cols_in_cell();
        lines_in_cell = options->lines_in_cell();
        interleaved_slices = options->interleaved_slices();
    } else {
        auto options = reinterpret_cast<const TFlite::ENN_InverseCFUOptions *>(operator_->get_option().get_addr());
        cols_in_cell = options->cols_in_cell();
        lines_in_cell = options->lines_in_cell();
        interleaved_slices = options->interleaved_slices();
    }

    const auto &cfu_inverter = compute_library->createCFUInverter(precision_type, soc_name);

    Status ret_value = cfu_inverter->initialize(in_tensors[0], width, height, channel, cols_in_cell, lines_in_cell,
                                                interleaved_slices, 0, 0);

    operators->push_back(std::make_shared<EnnUDOperator<ICFUInverter>>(operator_->get_name(), operator_->get_id(),
                                                                       in_tensors, out_tensors, data_tensors, cfu_inverter));

    return (ret_value == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_ENN_NORMALIZATION>(
    const model::component::Operator::Ptr &operator_) {
    ENN_DBG_PRINT("opr->code = %d\n", operator_->get_code());

    std::string builtin_op_name = TFlite::EnumNameBuiltinOperator(operator_->get_code());
    if (builtin_op_name.compare(operator_->get_name()) != 0) {
        ENN_ERR_PRINT("Invalid BuiltinOperator : %s\n", builtin_op_name.c_str());
        return ENN_RET_INVAL;
    }

    if (operator_->get_option().get_enum() != TFlite::BuiltinOptions_ENN_NormalizationOptions) {
        ENN_ERR_PRINT("Invalid BuiltinOptions : %s\n", TFlite::EnumNameBuiltinOptions(operator_->get_option().get_enum()));
        return ENN_RET_INVAL;
    }

    if (operator_->get_option().get_addr() == nullptr) {
        ENN_ERR_PRINT("Invalid BuiltinOptions (null)\n");
        return ENN_RET_INVAL;
    }

    PrecisionType precision_type = PrecisionType::FP32;

    std::vector<std::shared_ptr<ITensor>> in_tensors, out_tensors, data_tensors;

    convert_to_tensors(operator_, precision_type, in_tensors, out_tensors, data_tensors);

    std::shared_ptr<ITensor> mean_tensor;
    std::shared_ptr<ITensor> scale_tensor;

    auto options = reinterpret_cast<const TFlite::ENN_NormalizationOptions *>(operator_->get_option().get_addr());
    std::vector<float> mean = util::convert_vector<float>(options->mean());
    auto mean_data = static_cast<DataPtr>(&mean[0]);
    mean_tensor = create_and_copy_tensor(TFlite::TensorType_FLOAT32, mean_data,
                                         get_data_size(TFlite::TensorType_FLOAT32, mean.size()));

    std::vector<float> scale = util::convert_vector<float>(options->scale());
    auto scale_data = static_cast<DataPtr>(&scale[0]);
    scale_tensor = create_and_copy_tensor(TFlite::TensorType_FLOAT32, scale_data,
                                          get_data_size(TFlite::TensorType_FLOAT32, scale.size()));

    const auto &normalization = compute_library->createNormalization(precision_type);

    Status ret_value = normalization->initialize(in_tensors[0], mean_tensor, scale_tensor);

    operators->push_back(std::make_shared<EnnUDOperator<INormalization>>(
        operator_->get_name(), operator_->get_id(), in_tensors, out_tensors, data_tensors, normalization));

    return (ret_value == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_ENN_DETECTION>(
    const model::component::Operator::Ptr &operator_) {
    ENN_DBG_PRINT("opr->code = %d\n", operator_->get_code());

    std::string builtin_op_name = TFlite::EnumNameBuiltinOperator(operator_->get_code());
    if (builtin_op_name.compare(operator_->get_name()) != 0) {
        ENN_ERR_PRINT("Invalid BuiltinOperator : %s\n", builtin_op_name.c_str());
        return ENN_RET_INVAL;
    }

    if (operator_->get_option().get_enum() != TFlite::BuiltinOptions_ENN_DetectionOptions) {
        ENN_ERR_PRINT("Invalid BuiltinOptions : %s\n", TFlite::EnumNameBuiltinOptions(operator_->get_option().get_enum()));
        return ENN_RET_INVAL;
    }

    if (operator_->get_option().get_addr() == nullptr) {
        ENN_ERR_PRINT("Invalid BuiltinOptions (null)\n");
        return ENN_RET_INVAL;
    }

    PrecisionType precision_type = PrecisionType::FP32;

    std::vector<std::shared_ptr<ITensor>> in_tensors, out_tensors, data_tensors;

    convert_to_tensors(operator_, precision_type, in_tensors, out_tensors, data_tensors);

    std::shared_ptr<ITensor> mean_tensor;
    std::shared_ptr<ITensor> scale_tensor;

    auto options = reinterpret_cast<const TFlite::ENN_DetectionOptions *>(operator_->get_option().get_addr());

    uint32_t num_classes_ = static_cast<int32_t>(options->num_classes());
    bool share_location_ = options->share_location();
    float nms_threshold_ = options->nms_threshold();
    int32_t background_label_id_ = options->background_label_id();
    int32_t nms_top_k_ = options->nms_top_k();
    int32_t keep_top_k_ = options->keep_top_k();
    uint32_t code_type_ = options->code_type();
    float confidence_threshold_ = options->confidence_threshold();
    float nms_eta_ = options->nms_eta();
    bool variance_encoded_in_target_ = options->variance_encoded_in_target();

#if 0  // Required, ToDo(empire.jung, 8/31): Legacy SSD, remove after deciding to use priorbox
    Dim4 loc_dim_ = in_tensors[1]->getDim();
    Dim4 conf_dim_ = in_tensors[2]->getDim();
    Dim4 prior_dim_ = in_tensors[0]->getDim();
#else
    Dim4 loc_dim_ = in_tensors[0]->getDim();
    Dim4 conf_dim_ = in_tensors[1]->getDim();
    Dim4 prior_dim_ = {1, 1, 1, 1};

    for (auto &tensor : operator_->in_tensors) {
        if (tensor->is_const()) {
            std::shared_ptr<model::component::Parameter> param =
                std::static_pointer_cast<model::component::Parameter>(tensor);
            ENN_DBG_PRINT("Param: %s, size: %zu (%p)\n", param->get_name().c_str(), param->get_buffer_size(),
                          param->get_buffer_addr());
            if (param->get_name().compare("PRIORBOX_OUTPUT") == 0) {
                auto data = const_cast<DataPtr>(param->get_buffer_addr());
                auto pbox_tensor =
                    create_and_copy_tensor((TFlite::TensorType)param->get_data_type(), data, param->get_shape());
                prior_dim_ = pbox_tensor->getDim();
                data_tensors.push_back(pbox_tensor);
            }
        }
    }
#endif

    Dim4 output_dim = out_tensors[0]->getDim();

    const auto &detection = output_dim.n > 1 ? compute_library->createDetection(precision_type)
                                             : compute_library->createDetectionBatchSingle(precision_type);

    Status ret_value = detection->initialize(loc_dim_, conf_dim_, prior_dim_, output_dim, num_classes_, share_location_,
                                             nms_threshold_, background_label_id_, nms_top_k_, keep_top_k_, code_type_,
                                             confidence_threshold_, nms_eta_, variance_encoded_in_target_);

    operators->push_back(std::make_shared<EnnUDOperator<IDetection>>(operator_->get_name(), operator_->get_id(), in_tensors,
                                                                     out_tensors, data_tensors, detection));

    return (ret_value == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_ENN_FLATTEN>(
    const model::component::Operator::Ptr &operator_) {
    ENN_DBG_PRINT("opr->code = %d\n", operator_->get_code());

    std::string builtin_op_name = TFlite::EnumNameBuiltinOperator(operator_->get_code());
    if (builtin_op_name.compare(operator_->get_name()) != 0) {
        ENN_ERR_PRINT("Invalid BuiltinOperator : %s\n", builtin_op_name.c_str());
        return ENN_RET_INVAL;
    }

    if (operator_->get_option().get_enum() != TFlite::BuiltinOptions_ENN_FlattenOptions) {
        ENN_ERR_PRINT("Invalid BuiltinOptions : %s\n", TFlite::EnumNameBuiltinOptions(operator_->get_option().get_enum()));
        return ENN_RET_INVAL;
    }

    if (operator_->get_option().get_addr() == nullptr) {
        ENN_ERR_PRINT("Invalid BuiltinOptions (null)\n");
        return ENN_RET_INVAL;
    }

    PrecisionType precision_type = PrecisionType::FP32;

    std::vector<std::shared_ptr<ITensor>> in_tensors, out_tensors, data_tensors;

    convert_to_tensors(operator_, precision_type, in_tensors, out_tensors, data_tensors);

    auto options = reinterpret_cast<const TFlite::ENN_FlattenOptions *>(operator_->get_option().get_addr());
    int32_t axis = options->axis();
    int32_t end_axis = options->end_axis();

    const auto &flatten = compute_library->createFlatten(precision_type);

    Status ret_value = flatten->initialize(axis, end_axis);

    operators->push_back(std::make_shared<EnnUDOperator<IFlatten>>(operator_->get_name(), operator_->get_id(), in_tensors,
                                                                   out_tensors, data_tensors, flatten));

    return (ret_value == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}
#endif

/***************************************************************************************************************************
 * create_ud_operator() of Custom Operators                                                                                *
 ***************************************************************************************************************************/
template <>
EnnReturn OperationConstructor::create_ud_operator<CustomOperator::CustomOperator_Normalization>(
    const model::component::Operator::Ptr &operator_) {
    ENN_DBG_PRINT("opr->name = %s\n", operator_->get_name().c_str());

    PrecisionType precision_type = PrecisionType::FP32;

    std::vector<std::shared_ptr<ITensor>> in_tensors, out_tensors, data_tensors;

    convert_to_tensors(operator_, precision_type, in_tensors, out_tensors, data_tensors);

    std::shared_ptr<ITensor> mean_tensor;
    std::shared_ptr<ITensor> scale_tensor;

    for (auto &tensor : operator_->in_tensors) {
        if (tensor->is_const()) {  // if tensor is const, the tensor is parameter.
            std::shared_ptr<model::component::Parameter> param =
                std::static_pointer_cast<model::component::Parameter>(tensor);
            ENN_DBG_PRINT("Param: %s, size: %zu (%p)\n", param->get_name().c_str(), param->get_buffer_size(),
                          param->get_buffer_addr());
            if (param->get_name().compare("MEAN") == 0) {
                auto data = const_cast<DataPtr>(param->get_buffer_addr());
                mean_tensor =
                    create_and_copy_tensor((TFlite::TensorType)param->get_data_type(), data, param->get_buffer_size());
            } else if (param->get_name().compare("SCALE") == 0) {
                auto data = const_cast<DataPtr>(param->get_buffer_addr());
                scale_tensor =
                    create_and_copy_tensor((TFlite::TensorType)param->get_data_type(), data, param->get_buffer_size());
            }
        }
    }

    const auto &normalization = compute_library->createNormalization(precision_type);

    Status ret_value = normalization->initialize(in_tensors[0], mean_tensor, scale_tensor);

    operators->push_back(std::make_shared<EnnUDOperator<INormalization>>(
        operator_->get_name(), operator_->get_id(), in_tensors, out_tensors, data_tensors, normalization));

    return (ret_value == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<CustomOperator::CustomOperator_AsymmQuantization>(
    const model::component::Operator::Ptr &operator_) {
    ENN_DBG_PRINT("opr->name = %s\n", operator_->get_name().c_str());

    PrecisionType precision_type = PrecisionType::FP32;

    std::vector<std::shared_ptr<ITensor>> in_tensors, out_tensors, data_tensors;

    convert_to_tensors(operator_, precision_type, in_tensors, out_tensors, data_tensors);

    int32_t channel = in_tensors[0]->getDim().c;
    int32_t height = in_tensors[0]->getDim().h;
    int32_t width = in_tensors[0]->getDim().w;
    float scale = 0.f;
    int32_t zero_point = 0;

    for (auto &tensor : operator_->in_tensors) {
        if (tensor->is_const()) {  // if tensor is const, the tensor is parameter.
            std::shared_ptr<model::component::Parameter> param =
                std::static_pointer_cast<model::component::Parameter>(tensor);
            ENN_DBG_PRINT("Param: %s, size: %zu (%p)\n", param->get_name().c_str(), param->get_buffer_size(),
                          param->get_buffer_addr());
            if (param->get_name().compare("SCALE") == 0) {
                scale = *((float *)(param->get_buffer_addr()));
            } else if (param->get_name().compare("ZERO_POINT") == 0) {
                zero_point = *((int32_t *)(param->get_buffer_addr()));
            }
        }
    }

    const auto &asymm_quantize = compute_library->createAsymmQuantization(precision_type);

    Status ret_value = asymm_quantize->initialize(in_tensors[0], channel, width, height, scale, zero_point);

    operators->push_back(std::make_shared<EnnUDOperator<IAsymmQuantization>>(
        operator_->get_name(), operator_->get_id(), in_tensors, out_tensors, data_tensors, asymm_quantize));

    return (ret_value == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<CustomOperator::CustomOperator_AsymmDequantization>(
    const model::component::Operator::Ptr &operator_) {
    ENN_DBG_PRINT("opr->name = %s\n", operator_->get_name().c_str());

    PrecisionType precision_type = PrecisionType::FP32;

    std::vector<std::shared_ptr<ITensor>> in_tensors, out_tensors, data_tensors;

    convert_to_tensors(operator_, precision_type, in_tensors, out_tensors, data_tensors);

    int32_t channel = in_tensors[0]->getDim().c;
    int32_t height = in_tensors[0]->getDim().h;
    int32_t width = in_tensors[0]->getDim().w;
    int32_t num_data = channel * height * width;
    uint32_t img_size = height * width;
    float scale = 0.f;
    int32_t zero_point = 0;

    for (auto &tensor : operator_->in_tensors) {
        if (tensor->is_const()) {  // if tensor is const, the tensor is parameter.
            std::shared_ptr<model::component::Parameter> param =
                std::static_pointer_cast<model::component::Parameter>(tensor);
            ENN_DBG_PRINT("Param: %s, size: %zu (%p)\n", param->get_name().c_str(), param->get_buffer_size(),
                          param->get_buffer_addr());
            if (param->get_name().compare("SCALE") == 0) {
                scale = *((float *)(param->get_buffer_addr()));
            } else if (param->get_name().compare("ZERO_POINT") == 0) {
                zero_point = *((int32_t *)(param->get_buffer_addr()));
            }
        }
    }

    const auto &asymm_dequantize = compute_library->createAsymmDequantization(precision_type);

    Status ret_value = asymm_dequantize->initialize(in_tensors[0], num_data, scale, zero_point, img_size);

    operators->push_back(std::make_shared<EnnUDOperator<IAsymmDequantization>>(
        operator_->get_name(), operator_->get_id(), in_tensors, out_tensors, data_tensors, asymm_dequantize));

    return (ret_value == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<CustomOperator::CustomOperator_ConvertCFU>(
    const model::component::Operator::Ptr &operator_) {
    ENN_DBG_PRINT("opr->name = %s\n", operator_->get_name().c_str());

    PrecisionType precision_type = PrecisionType::FP32;

    std::vector<std::shared_ptr<ITensor>> in_tensors, out_tensors, data_tensors;

    convert_to_tensors(operator_, precision_type, in_tensors, out_tensors, data_tensors);

    int32_t channel = in_tensors[0]->getDim().c;
    int32_t height = in_tensors[0]->getDim().h;
    int32_t width = in_tensors[0]->getDim().w;
    int32_t cols_in_cell = 1;
    int32_t lines_in_cell = 1;
    int32_t interleaved_slices = 16;

    // Nice to have, ToDo(empire.jung, TBD): How to get soc platform name
    const std::string soc_name = enn::platform::EXYNOS2100;

    const auto &cfu_converter = compute_library->createCFUConverter(precision_type, soc_name);

    Status ret_value = cfu_converter->initialize(in_tensors[0], width, height, channel, cols_in_cell, lines_in_cell,
                                                 interleaved_slices);

    operators->push_back(std::make_shared<EnnUDOperator<ICFUConverter>>(
        operator_->get_name(), operator_->get_id(), in_tensors, out_tensors, data_tensors, cfu_converter));

    return (ret_value == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<CustomOperator::CustomOperator_InverseCFU>(
    const model::component::Operator::Ptr &operator_) {
    ENN_DBG_PRINT("opr->name = %s\n", operator_->get_name().c_str());

    PrecisionType precision_type = PrecisionType::FP32;

    std::vector<std::shared_ptr<ITensor>> in_tensors, out_tensors, data_tensors;

    convert_to_tensors(operator_, precision_type, in_tensors, out_tensors, data_tensors);

    std::unordered_map<std::string, std::vector<int32_t>> params_for_soc = {
        // {SOC_NAME, {cols_in_cell, lines_in_cell, interleaved_slices, idps, unit_size}}
        {enn::platform::EXYNOS2100, {1, 1, 16, 0, 0}},
        {enn::platform::EXYNOS9830, {8, 4, 2, 4, 2}},
        {enn::platform::EXYNOS9820, {4, 4, 1, 1, 1}},
    };

    // Nice to have, ToDo(empire.jung, TBD): How to get soc platform name
    const std::string soc_name = enn::platform::EXYNOS2100;

    int32_t channel = out_tensors[0]->getDim().c;
    int32_t height = out_tensors[0]->getDim().h;
    int32_t width = out_tensors[0]->getDim().w;
    std::vector<int32_t> params = params_for_soc[soc_name];

    const auto &cfu_inverter = compute_library->createCFUInverter(precision_type, soc_name);

    Status ret_value = cfu_inverter->initialize(in_tensors[0], width, height, channel, params[0], params[1], params[2],
                                                params[3], params[4]);

    operators->push_back(std::make_shared<EnnUDOperator<ICFUInverter>>(operator_->get_name(), operator_->get_id(),
                                                                       in_tensors, out_tensors, data_tensors, cfu_inverter));

    return (ret_value == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<CustomOperator::CustomOperator_Concat>(
    const model::component::Operator::Ptr &operator_) {
    ENN_DBG_PRINT("opr->name = %s\n", operator_->get_name().c_str());

    PrecisionType precision_type = PrecisionType::FP32;

    std::vector<std::shared_ptr<ITensor>> in_tensors, out_tensors, data_tensors;

    convert_to_tensors(operator_, precision_type, in_tensors, out_tensors, data_tensors);

    int32_t number = in_tensors[0]->getDim().n;
    int32_t channel = in_tensors[0]->getDim().c;
    int32_t height = in_tensors[0]->getDim().h;
    int32_t width = in_tensors[0]->getDim().w;
    int32_t *axis;

    for (auto &tensor : operator_->in_tensors) {
        if (tensor->is_const()) {  // if tensor is const, the tensor is parameter.
            std::shared_ptr<model::component::Parameter> param =
                std::static_pointer_cast<model::component::Parameter>(tensor);
            ENN_DBG_PRINT("Param: %s, size: %zu (%p)\n", param->get_name().c_str(), param->get_buffer_size(),
                          param->get_buffer_addr());
            if (param->get_name().compare("AXIS") == 0) {
                axis = (int32_t *)(param->get_buffer_addr());
            }
        }
    }

    const auto &concat = compute_library->createConcat(precision_type);

    Status ret_value = concat->initialize({in_tensors[0], in_tensors[1]}, 2, number, channel, height, width, *axis);

    operators->push_back(std::make_shared<EnnUDOperator<IConcat>>(operator_->get_name(), operator_->get_id(), in_tensors,
                                                                  out_tensors, data_tensors, concat));

    return (ret_value == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<CustomOperator::CustomOperator_Quantization>(
    const model::component::Operator::Ptr &operator_) {
    ENN_DBG_PRINT("opr->name = %s\n", operator_->get_name().c_str());

    PrecisionType precision_type = PrecisionType::FP32;

    std::vector<std::shared_ptr<ITensor>> in_tensors, out_tensors, data_tensors;

    convert_to_tensors(operator_, precision_type, in_tensors, out_tensors, data_tensors);

    int32_t channel = in_tensors[0]->getDim().c;
    int32_t height = in_tensors[0]->getDim().h;
    int32_t width = in_tensors[0]->getDim().w;
    std::shared_ptr<ITensor> frac_tensor;

    for (auto &tensor : operator_->in_tensors) {
        if (tensor->is_const()) {  // if tensor is const, the tensor is parameter.
            std::shared_ptr<model::component::Parameter> param =
                std::static_pointer_cast<model::component::Parameter>(tensor);
            ENN_DBG_PRINT("Param: %s, size: %zu (%p)\n", param->get_name().c_str(), param->get_buffer_size(),
                          param->get_buffer_addr());
            if (param->get_name().compare("FRAC_LEN") == 0) {
                auto data = const_cast<DataPtr>(param->get_buffer_addr());
                frac_tensor =
                    create_and_copy_tensor((TFlite::TensorType)param->get_data_type(), data, param->get_buffer_size());
            }
        }
    }

    const auto &quantization = compute_library->createQuantization(precision_type);

    Status ret_value = quantization->initialize(in_tensors[0], channel, height, width, frac_tensor);

    operators->push_back(std::make_shared<EnnUDOperator<IQuantization>>(
        operator_->get_name(), operator_->get_id(), in_tensors, out_tensors, data_tensors, quantization));

    return (ret_value == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<CustomOperator::CustomOperator_Dequantization>(
    const model::component::Operator::Ptr &operator_) {
    ENN_DBG_PRINT("opr->name = %s\n", operator_->get_name().c_str());

    PrecisionType precision_type = PrecisionType::FP32;

    std::vector<std::shared_ptr<ITensor>> in_tensors, out_tensors, data_tensors;

    convert_to_tensors(operator_, precision_type, in_tensors, out_tensors, data_tensors);

    uint32_t img_size = in_tensors[0]->getDim().h * in_tensors[0]->getDim().w;
    int32_t data_num = in_tensors[0]->getDim().c * img_size;
    std::shared_ptr<ITensor> frac_tensor;

    for (auto &tensor : operator_->in_tensors) {
        if (tensor->is_const()) {  // if tensor is const, the tensor is parameter.
            std::shared_ptr<model::component::Parameter> param =
                std::static_pointer_cast<model::component::Parameter>(tensor);
            ENN_DBG_PRINT("Param: %s, size: %zu (%p)\n", param->get_name().c_str(), param->get_buffer_size(),
                          param->get_buffer_addr());
            if (param->get_name().compare("FRAC_LEN") == 0) {
                auto data = const_cast<DataPtr>(param->get_buffer_addr());
                frac_tensor =
                    create_and_copy_tensor((TFlite::TensorType)param->get_data_type(), data, param->get_buffer_size());
            }
        }
    }

    const auto &dequantization = compute_library->createDequantization(precision_type);

    Status ret_value = dequantization->initialize(in_tensors[0], data_num, frac_tensor, img_size);

    operators->push_back(std::make_shared<EnnUDOperator<IDequantization>>(
        operator_->get_name(), operator_->get_id(), in_tensors, out_tensors, data_tensors, dequantization));

    return (ret_value == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<CustomOperator::CustomOperator_NormalizationQuantization>(
    const model::component::Operator::Ptr &operator_) {
    ENN_DBG_PRINT("opr->name = %s (not support yet)\n", operator_->get_name().c_str());

    return ENN_RET_SUCCESS;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<CustomOperator::CustomOperator_Pad>(
    const model::component::Operator::Ptr &operator_) {
    ENN_DBG_PRINT("opr->name = %s\n", operator_->get_name().c_str());

    PrecisionType precision_type = PrecisionType::FP32;

    std::vector<std::shared_ptr<ITensor>> in_tensors, out_tensors, data_tensors;
    // TODO(daewhan): check if convert_to_tensor can be merged with the following parameter for-loop
    convert_to_tensors(operator_, precision_type, in_tensors, out_tensors, data_tensors);

    int32_t channel = in_tensors[0]->getDim().c;
    int32_t height = in_tensors[0]->getDim().h;
    int32_t width = in_tensors[0]->getDim().w;
    std::vector<int32_t> pad_front;
    std::vector<int32_t> pad_end;
    std::vector<float> pad_val;

    for (auto &tensor : operator_->in_tensors) {
        if (tensor->is_const()) {  // if tensor is const, the tensor is parameter.
            std::shared_ptr<model::component::Parameter> param =
                std::static_pointer_cast<model::component::Parameter>(tensor);
            ENN_DBG_PRINT("Param: %s, size: %zu (%p)\n", param->get_name().c_str(), param->get_buffer_size(),
                          param->get_buffer_addr());
            if (param->get_name().compare("PAD_FRONT") == 0) {
                int32_t *data = (int32_t *)(param->get_buffer_addr());
                for (uint i = 0; i < get_data_count((TFlite::TensorType)param->get_data_type(), param->get_buffer_size());
                     i++) {
                    pad_front.push_back(*(data + i));
                }
            } else if (param->get_name().compare("PAD_END") == 0) {
                int32_t *data = (int32_t *)(param->get_buffer_addr());
                for (uint i = 0; i < get_data_count((TFlite::TensorType)param->get_data_type(), param->get_buffer_size());
                     i++) {
                    pad_end.push_back(*(data + i));
                }
            } else if (param->get_name().compare("PAD_VALUE") == 0) {
                float *data = (float *)(param->get_buffer_addr());
                for (uint i = 0; i < get_data_count((TFlite::TensorType)param->get_data_type(), param->get_buffer_size());
                     i++) {
                    pad_val.push_back(*(data + i));
                }
            }
        }
    }

    const auto &pad = compute_library->createPad(precision_type);

    Status ret_value = pad->initialize(in_tensors[0], width, height, channel, pad_front, pad_end, pad_val);

    operators->push_back(std::make_shared<EnnUDOperator<IPad>>(operator_->get_name(), operator_->get_id(), in_tensors,
                                                               out_tensors, data_tensors, pad));

    return (ret_value == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

template <>
EnnReturn OperationConstructor::create_ud_operator<CustomOperator::CustomOperator_Softmax>(
    const model::component::Operator::Ptr &operator_) {
    ENN_DBG_PRINT("opr->name = %s (custom)\n", operator_->get_name().c_str());

    PrecisionType precision_type = PrecisionType::FP32;

    std::vector<std::shared_ptr<ITensor>> in_tensors, out_tensors, data_tensors;

    convert_to_tensors(operator_, precision_type, in_tensors, out_tensors, data_tensors);

    int32_t number = in_tensors[0]->getDim().n;
    int32_t channel = in_tensors[0]->getDim().c;
    int32_t height = in_tensors[0]->getDim().h;
    int32_t width = in_tensors[0]->getDim().w;
    float beta = 0.f;
    int32_t axis = 0;

    for (auto &tensor : operator_->in_tensors) {
        if (tensor->is_const()) {  // if tensor is const, the tensor is parameter.
            std::shared_ptr<model::component::Parameter> param =
                std::static_pointer_cast<model::component::Parameter>(tensor);
            ENN_DBG_PRINT("Param: %s, size: %zu (%p)\n", param->get_name().c_str(), param->get_buffer_size(),
                          param->get_buffer_addr());
            if (param->get_name().compare("BETA") == 0) {
                beta = *((float *)(param->get_buffer_addr()));
            } else if (param->get_name().compare("AXIS") == 0) {
                axis = *((int32_t *)(param->get_buffer_addr()));
            }
        }
    }

    const auto &softmax = compute_library->createSoftmax(precision_type);

    Status ret_value = softmax->initialize(in_tensors[0], width, height, channel, number, beta, axis);

    operators->push_back(std::make_shared<EnnUDOperator<ISoftmax>>(operator_->get_name(), operator_->get_id(), in_tensors,
                                                                   out_tensors, data_tensors, softmax));

    return (ret_value == Status::SUCCESS) ? ENN_RET_SUCCESS : ENN_RET_FAILED;
}

/***************************************************************************************************************************
 * Implementation of OperationConstructor                                                                                  *
 ***************************************************************************************************************************/
OperationConstructor::OperationConstructor(std::shared_ptr<NEONComputeLibrary> compute_library_)
    : compute_library(compute_library_) {
    builtin_op_map = {
        {TFlite::BuiltinOperator::BuiltinOperator_SOFTMAX,
         &OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_SOFTMAX>},
#ifndef SCHEMA_NNC_V1
        {TFlite::BuiltinOperator::BuiltinOperator_DEQUANTIZE,
         &OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_DEQUANTIZE>},
        {TFlite::BuiltinOperator::BuiltinOperator_QUANTIZE,
         &OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_QUANTIZE>},
        {TFlite::BuiltinOperator::BuiltinOperator_ENN_CFU,
         &OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_ENN_CFU>},
        {TFlite::BuiltinOperator::BuiltinOperator_ENN_INVERSE_CFU,
         &OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_ENN_INVERSE_CFU>},
        {TFlite::BuiltinOperator::BuiltinOperator_ENN_NORMALIZATION,
         &OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_ENN_NORMALIZATION>},
        {TFlite::BuiltinOperator::BuiltinOperator_ENN_DETECTION,
         &OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_ENN_DETECTION>},
        {TFlite::BuiltinOperator::BuiltinOperator_ENN_FLATTEN,
         &OperationConstructor::create_ud_operator<TFlite::BuiltinOperator::BuiltinOperator_ENN_FLATTEN>},
#endif
    };

    custom_op_map = {
        {"Normalization", &OperationConstructor::create_ud_operator<CustomOperator::CustomOperator_Normalization>},
        {"AsymmQuantization", &OperationConstructor::create_ud_operator<CustomOperator::CustomOperator_AsymmQuantization>},
        {"AsymmDequantization", &OperationConstructor::create_ud_operator<CustomOperator::CustomOperator_AsymmDequantization>},
        {"ConvertCFU", &OperationConstructor::create_ud_operator<CustomOperator::CustomOperator_ConvertCFU>},
        {"InverseCFU", &OperationConstructor::create_ud_operator<CustomOperator::CustomOperator_InverseCFU>},
        {"Concat", &OperationConstructor::create_ud_operator<CustomOperator::CustomOperator_Concat>},
        {"Quantization", &OperationConstructor::create_ud_operator<CustomOperator::CustomOperator_Quantization>},
        {"Dequantization", &OperationConstructor::create_ud_operator<CustomOperator::CustomOperator_Dequantization>},
        {"NormalizationDequantization", &OperationConstructor::create_ud_operator<CustomOperator::CustomOperator_NormalizationQuantization>},
        {"Pad", &OperationConstructor::create_ud_operator<CustomOperator::CustomOperator_Pad>},
        {"SOFTMAX", &OperationConstructor::create_ud_operator<CustomOperator::CustomOperator_Softmax>},
    };
}

OperationConstructor::~OperationConstructor() {
    builtin_op_map.clear();
    custom_op_map.clear();
}

EnnReturn OperationConstructor::create_ud_operator(const model::component::Operator::Ptr &operator_) {
    if (is_builtin_operator(operator_->get_code())) {
        // Builtin Operation has code
        const auto &it = builtin_op_map.find(operator_->get_code());

        if (it == builtin_op_map.end()) {
            ENN_ERR_PRINT("Not found operation code : %d\n", operator_->get_code());
            return ENN_RET_INVAL;
        }

        const auto &func = it->second;

        return (this->*func)(operator_);
    } else if (!operator_->get_name().empty()) {
        // Custom Operation has name
        const auto &it = custom_op_map.find(operator_->get_name());

        if (it == custom_op_map.end()) {
            ENN_ERR_PRINT("Not found operation name : %s\n", operator_->get_name().c_str());
            return ENN_RET_INVAL;
        }

        const auto &func = it->second;

        return (this->*func)(operator_);
    }

    return ENN_RET_FAILED;
}

void OperationConstructor::convert_to_tensors(const model::component::Operator::Ptr &operator_,
                                              PrecisionType &precision_type,
                                              std::vector<std::shared_ptr<ITensor>> &in_tensors,
                                              std::vector<std::shared_ptr<ITensor>> &out_tensors,
                                              std::vector<std::shared_ptr<ITensor>> &data_tensors) {
    for (auto &in_tensor : operator_->in_tensors) {
        if (!in_tensor->is_const()) {  // if tensor is not const, the tensor is feature map.
            model::component::FeatureMap::Ptr ifm =
                std::static_pointer_cast<model::component::FeatureMap>(in_tensor);
            ENN_DBG_PRINT("%s(id:%d), index:%d, type:%d, NCHW(%d,%d,%d,%d)\n", ifm->get_name().c_str(), ifm->get_id(),
                          ifm->get_buffer_index(), ifm->get_data_type(), ifm->get_shape()[0], ifm->get_shape()[1],
                          ifm->get_shape()[2], ifm->get_shape()[3]);

            std::shared_ptr<ITensor> tensor = compute_library->create_tensor(
                (TFlite::TensorType)ifm->get_data_type(), precision_type, ifm->get_shape(), ifm->get_buffer_index());
            in_tensors.push_back(tensor);
        }
    }

    for (auto &out_tensor : operator_->out_tensors) {
        if (!out_tensor->is_const()) {
            model::component::FeatureMap::Ptr ofm =
                std::static_pointer_cast<model::component::FeatureMap>(out_tensor);
            ENN_DBG_PRINT("%s(id:%d), index:%d, type:%d, NCHW(%d,%d,%d,%d)\n", ofm->get_name().c_str(), ofm->get_id(),
                          ofm->get_buffer_index(), ofm->get_data_type(), ofm->get_shape()[0], ofm->get_shape()[1],
                          ofm->get_shape()[2], ofm->get_shape()[3]);

            std::shared_ptr<ITensor> tensor = compute_library->create_tensor(
                (TFlite::TensorType)ofm->get_data_type(), precision_type, ofm->get_shape(), ofm->get_buffer_index());
            out_tensors.push_back(tensor);
        }
    }

    ENN_UNUSED(data_tensors);
}

std::shared_ptr<ITensor> OperationConstructor::create_and_copy_tensor(const TFlite::TensorType &type, DataPtr &data,
                                                                      size_t size, const PrecisionType &precision) {
    return compute_library->create_and_copy_tensor(type, data, precision, {1, 1, 1, get_data_count(type, size)});
}

std::shared_ptr<ITensor> OperationConstructor::create_and_copy_tensor(const TFlite::TensorType &type, DataPtr &data,
                                                                      const NDims &ndim, const PrecisionType &precision) {
    return compute_library->create_and_copy_tensor(type, data, precision, ndim);
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn