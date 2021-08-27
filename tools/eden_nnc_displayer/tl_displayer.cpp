/*
 * Copyright (C) 2018 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

/**
 * @file    tfliteDisplayer.cpp
 */

#include <algorithm>  // max
#include <iostream>   // cout, endl
#include <string>     // string
#include <cstring>    // strncpy
#include <iomanip>    // setw, setfill, hex
#include <cstdio>     // fopen
#include <vector>     // vector
#include <memory>     // unique_ptr

// edenmodel
#include "tl_displayer.h"  // tflite::XXX

#ifdef LOG_TAG
#undef LOG_TAG
#endif
#define LOG_TAG "NN::TfLiteDisplayer"  // Change defined LOG_TAG on log.h for purpose.

#define ALIGN(a, b) (((a + (b - 1)) / b) * b)
#define NCP_INFO "NCP"
#define NORMALIZATION "Normalization"
#define QUANTIZATION "Quantization"
#define DEQUANTIZATION "Dequantization"
#define ASYMMQUANTIZATION "AsymmQuantization"
#define ASYMMDEQUANTIZATION "AsymmDequantization"
#define INVERSECFU "InverseCFU"

#define BIT_SIZE(type) (sizeof(type) * 8)

namespace eden {
namespace nn {

#define PrintPrefix "[NNC Displayer] "

struct opName {
    const char* edenOpName;
    const char* optionName;
};

const struct opName opNameTable[EDEN_OP_NUM_MAXIMUM] = {
    {"ADD", "AddOptions_000"},
    {"AVERAGE_POOL_2D", "AveragePool2DOptions_000"},
    {"CONCATENATION", "ConcatenationOptions_000"},
    {"CONV_2D", "Conv2DOptions_000"},
    {"DEPTHWISE_CONV_2D", "DepthwiseConv2DOptions_000"},

    {"DEPTH_TO_SPACE", "DepthToSpaceOptions_000"},
    {"DEQUANTIZE", "DequantizeOptions_000"},
    {"EMBEDDING_LOOKUP", "EmbeddingLookupOptions_000"},
    {"FLOOR", "FloorOptions_000"},
    {"FULLY_CONNECTED", "FullyConnectedOptions_000"},

    {"HASHTABLE_LOOKUP", "HashTableLookupOptions_000"},
    {"L2_NORMALIZATION", "L2NormalizationOptions_000"},
    {"L2_POOL_2D", "L2Pool2DOptions_000"},
    {"LOCAL_RESPONSE_NORMALIZATION", "LocalResponseNormalizationOptions_000"},
    {"LOGISTIC", "LogisticOptions_000"},

    {"LSH_PROJECTION", "LshProjectionOptions_000"},
    {"LSTM", "LSTMOptions_000"},
    {"MAX_POOL_2D", "MaxPool2DOptions_000"},
    {"MUL", "MulOptions_000"},
    {"RELU", "ReLUOptions_000"},

    {"RELU1", "ReLU1Options_000"},
    {"RELU6", "ReLU6Options_000"},
    {"RESHAPE", "ReshapeOptions_000"},
    {"RESIZE_BILINEAR", "ResizeBilinearOptions_000"},
    {"RNN", "RNNOptions_000"},

    {"SOFTMAX", "SoftmaxOptions_000"},
    {"SPACE_TO_DEPTH", "SpaceToDepthOptions_000"},
    {"SVDF", "SVDFOptions_000"},
    {"TANH", "TanhOptions_000"},
    {"BATCH_TO_SPACE_ND", "BatchToSpaceNDOptions_000"},

    {"DIV", "DivOptions_000"},
    {"MEAN", "MeanOptions_000"},
    {"CUSTOM", "CustomOptions_000"},
    {"SPACE_TO_BATCH_ND", "SpaceToBatchNDOptions_000"},
    {"SQUEEZE", "SqueezeOptions_000"},

    {"STRIDED_SLICE", "StridedSliceOptions_000"},
    {"SUB", "SubOptions_000"},
    {"TRANSPOSE", "TransposeOptions_000"},
    {"PRELU", "PReLUOptions_000"},
    {"ELEMENTWISE_MAX", "ElementwiseMaxOptions_000"},

    {"ARGMAX", "ArgmaxOptions_000"},
    {"SCALE", "ScaleOptions_000"},
    {"CROP", "CropOptions_000"},
    {"FLATTEN", "FlattenOptions_000"},
    {"PERMUTE", "PermuteOptions_000"},

    {"SLICE", "SliceOptions_000"},
    {"PRIORBOX", "PriorBoxOptions_000"},
    {"POWER", "PowerOptions_000"},
    {"PAD", "PadOptions_000"},
    {"DECONV_2D", "Deconv2DOptions_000"},

    {"DETECTION", "DetectionOptions_000"},
    {"ROIPOOL", "ROIPoolOptions_000"},
    {"BIDIRECTIONAL_SEQUENCE_LSTM", "BidirectionalSequenceLSTMOptions_000"},
    {"UNIDIRECTIONAL_SEQUENCE_LSTM", "UndirectionalSequenceLSTMOptions_000"},
    {"BIDIRECTIONAL_RNN", "BidirectionalRNNOptions_000"},

    {"UNIDIRECTIONAL_SEQUENCE_RNN", "UnidirectionalSequenceRNNOptions_000"},
    {"LAYER_NORM_LSTM", "LayerNormLSTMOptions_000"},
    {"TFLITE_DETECTION", "TFliteDetectionOptions_000"},
    {"LOGICAL_NOT", "LogicalNotOptions_000"},
    {"ROI_ALIGN", "ROIAlignOptions_000"},

    {"GENERATE_PROPOSALS", "GenerateProposalsOptions_000"},
    {"RESIZE_NEAREST_NEIGHBOR", "ResizeNearestNeighborOptions_000"},
    {"ABS", "ABSOptions_000"},
    {"GREATER", "GreaterOptions_000"},
    {"GREATER_EQUAL", "GreaterEqualOptions_000"},

    {"EQUAL", "EqualOptions_000"},
    {"NOT_EQUAL", "NotEqualOptions_000"},
    {"MINIMUM", "MinimumOptions_000"},
    {"NEG", "NegOptions_000"},
    {"EXPAND_DIMS", "ExpandDimsOptions_000"},

    {"GATHER", "GatherOptions_000"},
    {"SELECT", "SelectOptions_000"},
    {"SPLIT", "SplitOptions_000"},
    {"POW", "PowOptions_000"},
    {"LOG", "LogOptions_000"},

    {"SIN", "SinOptions_000"},
    {"RSQRT", "RSQRTOptions_000"},
    {"PAD_V2", "PADV2Options_000"},
    {"TOPK_V2", "TOPKV2Options_000"},
    {"TFLITEROIPOOL", "TfliteROIPoolOptions_000"},

    {"ARGMIN", "ARGMinOptions_000"},
    {"SQRT", "SQRTOptions_000"},
    {"BOX_WITH_NMS_LIMIT", "BoxWithNMSLIMITOptions_000"},
    {"EXP", "EXPOptions_000"},
    {"AXIS_ALIGNED_BBOX_TRANSFORM", "AxisAlignedBBoxTransformOptions_000"},

    {"INSTANCE_NORMALIZATION", "InstanceNormalizationOptions_000"},
    {"QUANTIZED_16BIT_LSTM", "Quantized16BitLSTMOptions_000"},
    {"QUANTIZE", "QuantizeOptions_000"},
    {"DETECTION_POSTPROCESSING", "DetectionPostprocessingOptions_000"},
    {"LOGICAL_AND", "LogicalAndOptions_000"},

    {"LOGICAL_OR", "LogicalOrOptions_000"},
    {"CAST", "CastOptions_000"},
    {"HEATMAP_MAX_KEYPOINT_OP", "HeatmapMaxKeypointOPOptions_000"},
    {"LOG_SOFTMAX", "LogSoftmaxOptions_000"},
    {"CHANNEL_SHUFFLE", "ChannelShuffleOptions_000"},

    {"RANDOM_MULTINOMIAL", "RandomMultinomialOptions_000"},
    {"LESS_EQUAL", "LessEqualOptions_000"},
    {"REDUCE_SUM", "ReduceSumOptions_000"},
    {"REDUCE_MIN", "ReduceMinOptions_000"},
    {"REDUCE_MAX", "ReduceMaxOptions_000"},

    {"REDUCE_PROD", "ReduceProdOptions_000"},
    {"REDUCE_ALL", "ReduceAllOptions_000"},
    {"REDUCE_ANY", "ReduceAnyOptions_000"},
    {"TILE", "TileOptions_000"},
    {"TF_SLICE", "TFSliceOptions_000"},

    {"LESS", "LessOptions_000"},
    {"MERGE_CONVOLUTION_RESHAPE_CONCAT", "MergeConvolutionReshapeConcatOptions_000"},
    {"FILL", "FillOptions_000"},
    {"ELU", "EluOptions_000"},
    {"QUANTIZED_LSTM", "QuantizedLSTMOptions_000"},

    {"HARDSWISH", "HardswishOptions_000"},
    {"RANK", "RankOptions_000"},
    {"CONVERT_NCHW2NHWC", "ConvertNCHW2NHWCOptions_000"},
    {"CONVERT_NHWC2NCHW", "ConvertNHWC2NCHWOptions_000"},
    {"UNPACK", "UnpackOptions_000"},

    {"CLIP", "ClipOptions_000"}
    // NOTICE: When you add new operation on EDEN_OP_NUM,
    //         you should add here too!!
};

inline const char* getEdenOpName(int edenOpNum) {
    static const char* strNull = "NULL";
    if (edenOpNum < 0 || edenOpNum >= EDEN_OP_NUM_MAXIMUM)
        return strNull;

    return opNameTable[edenOpNum].edenOpName;
}

inline const char* getOpOptionsName(int edenOpNum) {
    static const char* strNull = "NULL";
    if (edenOpNum < 0 || edenOpNum >= EDEN_OP_NUM_MAXIMUM)
        return strNull;

    return opNameTable[edenOpNum].optionName;
}

int TfLiteDisplayer::ModelLoading(const char* filename) {
    FILE* modelFp = std::fopen((const char*)filename, "rb");
    if (!modelFp) {
        printf(PrintPrefix "(-) Fail to open a given file!\n");
        return -1;
    }

    // Calculate a file size and load file into buffer
    std::fseek(modelFp, 0, SEEK_END);  // seek to end
    int32_t filesize = std::ftell(modelFp);
    std::fseek(modelFp, 0, SEEK_SET);  // seek to start
    printf(PrintPrefix " # Open File... (size: %d)\n", filesize);
    if (filesize < 0) {
        printf(PrintPrefix "(-) Fail to fread from a given file!\n");
        std::fclose(modelFp);
        return -1;
    }

    std::unique_ptr<char> spBuffer(new char[filesize]);
    loaded_model = std::move(spBuffer);
    char* buffer = loaded_model.get();

    size_t numOfBytes = std::fread(buffer, sizeof(char), filesize, modelFp);
    if (numOfBytes > 0) {
        filename_str = filename;
        model_size = numOfBytes;
    }
    std::fclose(modelFp);

    return 0;
}

void TfLiteDisplayer::ShowLoadedData() {
    std::cout << PrintPrefix << " # Filename: " << filename_str << ", FileSize: " << model_size << std::endl;
}

int TfLiteDisplayer::GetCompileDataFromMemory() {
    int ret = 0;

    ShowLoadedData();

    const tflite::Model* tfLiteModel = nullptr;
    char* buffer = loaded_model.get();
    int32_t alignedSize = ALIGN(model_size, 8);

    // Get tflite::Model from tflite::GetModel(buffer)
    tfLiteModel = tflite::GetModel(buffer);
    DumpTfLiteModel(tfLiteModel);

    return ret;
}

#define PREFIX(id) PrintPrefix << "[SG " << id << "] "

static void _draw_bar_big(std::string title, int val) {
    int nPrePost = 0;
    if (val - title.size() - 4 > 0)
        nPrePost = (val - title.size() - 4) / 2;
    std::cout << PrintPrefix << PREFIX_CYAN;
    for (int i = 0; i < nPrePost; i++) std::cout << "=";
    std::cout << "  " << title << "  ";
    for (int i = 0; i < nPrePost; i++) std::cout << "=";
    std::cout << POSTFIX << std::endl;
}

static void _draw_title_sub(std::string title, int sg_num = -1) {
    if (sg_num == -1)
        std::cout << PrintPrefix << PREFIX_WHITE << "[ " << title << " ]" << POSTFIX << std::endl;
    else
        std::cout << PREFIX(sg_num) << PREFIX_WHITE << "[ " << title << " ]" << POSTFIX << std::endl;

}

static void _draw_tensor_title(std::string title, int num, const char* name, int sgid = -1) {
    if (sgid == -1)
        std::cout << PrintPrefix << PREFIX_WHITE << " " << title << " #" << PREFIX_YELLOW << std::setw(2) << num
                  << PREFIX_WHITE << ": [" << std::setw(25) << name << "]" << POSTFIX;
    else
        std::cout << PREFIX(sgid) << PREFIX_WHITE << " " << title << " #" << PREFIX_YELLOW << std::setw(2) << num
                  << PREFIX_WHITE << ": [" << std::setw(25) << name << "]" << POSTFIX;
}

static void _draw_buffer_title(std::string title, int num) {
    std::cout << PrintPrefix << PREFIX_WHITE << " " << title << " #" << PREFIX_MAGENTA << std::setw(2) << num
                << PREFIX_WHITE << ":" << POSTFIX;
}

static void _draw_bar_none(const char *color, int size) {
    std::cout << PrintPrefix << color;
    for (int i = 0; i < size; i++) std::cout << "=";
    std::cout << POSTFIX << std::endl;
}
/**
 * @brief Dump TfLite Model contents
 * @details This function prints out a internal data values of TfLite Model.
 * @param void
 * @returns void
 */
void TfLiteDisplayer::DumpTfLiteModel(const tflite::Model* tfLiteModel) {
    _draw_bar_big("TFLite Displayer", 60);
    _draw_title_sub("Model Desciption");
    PrintVersion(tfLiteModel);
    PrintDescription(tfLiteModel);
    _draw_bar_none(PREFIX_CYAN, 60);
    _draw_title_sub("Operator Codes");
    PrintOperatorCodes(tfLiteModel);
    PrintSubgraphs(tfLiteModel);
    _draw_bar_big("Buffers", 60);
    PrintBuffers(tfLiteModel);
    _draw_bar_none(PREFIX_CYAN, 60);
}

int TfLiteDisplayer::PrintVersion(const tflite::Model* tfLiteModel) {
    std::cout << PrintPrefix << " # Version: " << tfLiteModel->version() << std::endl;
    return 0;
}

int TfLiteDisplayer::PrintDescription(const tflite::Model* tfLiteModel) {
    // description:string
    auto tflDescription = tfLiteModel->description();  // string description
    std::cout << PrintPrefix << " # Model Description: " << tflDescription->c_str() << std::endl;

    return 0;
}

const char* TfLiteDisplayer::GetTensorType(int tensorType) {
    if (tensorType == 0)
        return "FLOAT32";
    if (tensorType == 1)
        return "FLOAT16";
    if (tensorType == 2)
        return "INT32";
    if (tensorType == 3)
        return "UINT8";
    if (tensorType == 4)
        return "INT64";
    if (tensorType == 5)
        return "STRING";
    if (tensorType == 6)
        return "BOOL";
    return "INVALID";
}

//////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////// protected functions ////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////// private functions //////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

static void _draw_tensor_op(std::string title, int idx, int num, const char* name, int n_op, int subgraphId = -1) {
    if (subgraphId == -1)
        std::cout << PrintPrefix;
    else
        std::cout << PREFIX(subgraphId);

    std::cout << PREFIX_WHITE << "[" << idx << "] " << title << std::setw(2) << PREFIX_GREEN << num
              << PREFIX_WHITE << ": [" << std::setw(15) << name << ", " << n_op << "]";
    std::cout << POSTFIX;
    if (n_op == 32)  // custom
        std::cout << ": CUSTOM";
}

int TfLiteDisplayer::PrintOperatorCodes(const tflite::Model* tfLiteModel) {
//    _draw_bar_big("OPERATORS:");
    operators_data.clear();
    auto tflOperatorCodes = tfLiteModel->operator_codes();  // OperatorCode operator_codes[]
    for (uint32_t opIdx = 0; opIdx < tflOperatorCodes->size(); opIdx++) {
        int32_t builtinOp = tflOperatorCodes->Get(opIdx)->builtin_code();
        const char* builtinOpName = getEdenOpName(builtinOp);
        auto customOp = tflOperatorCodes->Get(opIdx)->custom_code();
        std::cout << PrintPrefix << " #" << PREFIX_GREEN << opIdx << POSTFIX << " : "
                  << (customOp ? customOp->str() : builtinOpName) << "(" << builtinOp << ")" << std::endl;
        // only save print operator codes
        operators_data.push_back({(customOp ? customOp->str() : builtinOpName), builtinOp});
    }
    return 0;
}

int TfLiteDisplayer::PrintSubgraphOperators(const tflite::Model* tfLiteModel, int subgraphId) {
    auto tflSubgraphs = tfLiteModel->subgraphs();
    auto tflSubgraph = tflSubgraphs->Get(subgraphId);
    auto tflOperators = tflSubgraph->operators();  // Operator operators[]
    _draw_title_sub("Operators", subgraphId);
    for (uint32_t oidx = 0; oidx < tflOperators->size(); oidx++) {
        auto ooidx = tflOperators->Get(oidx)->opcode_index();
        _draw_tensor_op("OPERATOR: Opcode_idx ", oidx, ooidx, operators_data[ooidx].first.c_str(),
                        operators_data[ooidx].second, subgraphId);
        std::cout << std::endl;

        auto tflOperator = tflOperators->Get(oidx);  // Operator operator = operators[oidx]
        auto tflOpcodeIndex = tflOperator->opcode_index();
        auto tflInputs = tflOperator->inputs();
        auto tflOutputs = tflOperator->outputs();

        std::cout << PREFIX(subgraphId) << "  # In/Out : [";
        for (uint32_t inidx = 0; inidx < tflInputs->size(); inidx++)
            std::cout << PREFIX_YELLOW << " " << tflInputs->Get(inidx) << POSTFIX;
        std::cout << " |";
        for (uint32_t outidx = 0; outidx < tflOutputs->size(); outidx++)
            std::cout << PREFIX_YELLOW << " " << tflOutputs->Get(outidx) << POSTFIX;
        std::cout << " ]";

        // builtin_options:BuiltinOptions
        auto tflBuiltinOptions = tflOperator->builtin_options();
        auto tflCustomOptions = tflOperator->custom_options();
        auto tflCustomOptionsFormat = tflOperator->custom_options_format();
        std::cout << ", builtin_opt(" << (tflBuiltinOptions == 0 ? "null" : "something") << ")";
        std::cout << ", custom_options(" << (tflCustomOptions == 0 ? "null" : "something") << ")";
        std::cout << ", tflCustomOptionsFormat(" << (tflCustomOptionsFormat == 0 ? "null" : "something") << ")"
                    << std::endl;
    }

    return 0;
}

int TfLiteDisplayer::PrintSubgraphInOut(const tflite::Model* tfLiteModel, int subgraphId) {
    auto tflSubgraphs = tfLiteModel->subgraphs();
    auto tflSubgraph = tflSubgraphs->Get(subgraphId);
    auto tflInputs = tflSubgraph->inputs();
    auto tflOutputs = tflSubgraph->outputs();
    std::cout << PREFIX(subgraphId) << "SubGraph #" << subgraphId << ": In / Out Tensors: [";
    for (uint32_t inidx = 0; inidx < tflInputs->size(); inidx++)
        std::cout << " " << PREFIX_YELLOW << tflInputs->Get(inidx) << POSTFIX;
    std::cout << " |";
    for (uint32_t outidx = 0; outidx < tflOutputs->size(); outidx++)
        std::cout << " " << PREFIX_YELLOW << tflOutputs->Get(outidx) << POSTFIX;
    std::cout << " ]" << std::endl;

    return 0;
}

int TfLiteDisplayer::PrintSubgraphTensors(const tflite::Model* tfLiteModel, int subgraphId) {
    auto tflSubgraphs = tfLiteModel->subgraphs();
    auto tflSubgraph = tflSubgraphs->Get(subgraphId);

    // Tensor:[Tensor]
    _draw_title_sub("TENSORS", subgraphId);
    auto tflTensors = tflSubgraph->tensors();  // tensors:[Tensor]
    for (uint32_t tidx = 0; tidx < tflTensors->size(); tidx++) {
        auto tflTensor = tflTensors->Get(tidx);  // Tensor tensor = tensors[tidx]
        _draw_tensor_title("Tensor", tidx, tflTensor->name()->c_str(), subgraphId);
        // shape:[int]
        auto tflShape = tflTensor->shape();
        std::cout << ", Type: " << std::setw(10) << GetTensorType(tflTensor->type()) << ", buf_idx: " << PREFIX_MAGENTA;
        if (tflTensor->buffer() == 0)
            std::cout << "X";
        else
            std::cout << tflTensor->buffer();
        std::cout << POSTFIX << ", Shape: { ";
        for (uint32_t sidx = 0; sidx < tflShape->size(); sidx++) {
            std::cout << tflShape->Get(sidx) << " ";
        }
        std::cout << "}" << std::endl;

        // quantization
        // PrintQuantization(tflTensor->quantization());
        auto tflQuantization = tflTensor->quantization();
        if (!tflQuantization)
            continue;

        std::cout << PREFIX(subgraphId) << " # QuantizationParameters: ";

        auto tflMin = tflQuantization->min();
        for (uint32_t midx = 0; midx < tflMin->size(); midx++)
            std::cout << " min[" << midx << "] = " << tflMin->Get(midx);

        auto tflMax = tflQuantization->max();
        for (uint32_t midx = 0; midx < tflMax->size(); midx++)
            std::cout << " max[" << midx << "] = " << tflMax->Get(midx);

        auto tflScale = tflQuantization->scale();
        for (uint32_t midx = 0; midx < tflScale->size(); midx++)
            std::cout << " scale[" << midx << "] = " << tflScale->Get(midx);

        auto tflZeroPoint = tflQuantization->zero_point();
        for (uint32_t midx = 0; midx < tflZeroPoint->size(); midx++)
            std::cout << " zero_point[" << midx << "] = " << tflZeroPoint->Get(midx);
        std::cout << std::endl;
    }

    return 0;
}

int TfLiteDisplayer::PrintSubgraphName(const tflite::Model* tfLiteModel, int subgraphId) {
    auto tflSubgraphs = tfLiteModel->subgraphs();
    auto tflSubgraph = tflSubgraphs->Get(subgraphId);

    if (tflSubgraph->name() == nullptr)
        _draw_title_sub("Name: Not defined", subgraphId);
    else
        _draw_title_sub(std::string("Name: ") + std::string(tflSubgraph->name()->c_str()), subgraphId);

    return 0;
}


int TfLiteDisplayer::PrintSubgraphs(const tflite::Model* tfLiteModel) {
//    _draw_bar_big("SubGraphs");

    // subgraphs:[SubGraph]
    auto tflSubgraphs = tfLiteModel->subgraphs();  // SubGraph subGraphs[]
    for (uint32_t gidx = 0; gidx < tflSubgraphs->size(); gidx++) {
        _draw_bar_big(std::string(" SubGraphs  #") + std::to_string(gidx), 60);
        PrintSubgraphInOut(tfLiteModel, gidx);
        PrintSubgraphName(tfLiteModel, gidx);
        PrintSubgraphOperators(tfLiteModel, gidx);
        PrintSubgraphTensors(tfLiteModel, gidx);

        auto tflSubgraph = tflSubgraphs->Get(gidx);
    }
    return 0;
}


int TfLiteDisplayer::PrintBuffers(const tflite::Model* tfLiteModel) {
    static const int32_t maxDataToShow = 20;
    _draw_title_sub("Buffers in File");
    auto tflBuffers = tfLiteModel->buffers();  // Buffer buffers[]
    for (uint32_t bidx = 0; bidx < tflBuffers->size(); bidx++) {
        auto tflBuffer = tflBuffers->Get(bidx);  // Operator operator = operators[oidx]
        auto tflData = tflBuffer->data();
        _draw_buffer_title("Buffer", bidx);
        int32_t lengthOfData = tflData->size();
        int32_t numOfLoop = (maxDataToShow < lengthOfData ? maxDataToShow : lengthOfData);
        std::cout << " " << std::setw(10) << lengthOfData << " Bytes: ";

        const uint8_t* addr = tflData->data();
        for (int32_t didx = 0; didx < numOfLoop; didx++)
            std::cout << " " << std::uppercase << std::hex << std::setfill('0') << std::setw(2) << (int32_t)addr[didx]
                      << std::dec;
        if (lengthOfData > maxDataToShow)
            std::cout << " ..";
        std::cout << std::setfill(' ') << std::endl;
    }
    return 0;
}

}  // namespace nn
}  // namespace eden

int main(int argc, char* argv[]) {
    const char* default_filename = "sample_nnc/NPU_AicClassifier.nnc";
    eden::nn::TfLiteDisplayer tfliteDisplayer;
    if (argc > 1)
        tfliteDisplayer.ModelLoading(argv[1]);
    else
        tfliteDisplayer.ModelLoading(default_filename);

    tfliteDisplayer.GetCompileDataFromMemory();
}
