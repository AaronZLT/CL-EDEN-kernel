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

#include "userdriver/gpu/gpu_op_executor.h"
#include "userdriver/gpu/common/CLComputeLibrary.hpp"
#include "userdriver/common/operator_interfaces/OperatorInterfaces.h"
#include "userdriver/common/operator_interfaces/userdriver_operator.h"

namespace enn {
namespace ud {
DEFINE_EXECUTOR(gpu::CLAdd);
DEFINE_EXECUTOR(gpu::CLAveragepool);
DEFINE_EXECUTOR(gpu::CLBidirectionalSequenceLstm);
DEFINE_EXECUTOR(gpu::CLCast);
DEFINE_EXECUTOR(gpu::CLConcat);
DEFINE_EXECUTOR(gpu::CLConvolution);
DEFINE_EXECUTOR(gpu::CLDeconvolution);
DEFINE_EXECUTOR(gpu::CLDepth2Space);
DEFINE_EXECUTOR(gpu::CLDepthwiseConvolution);
DEFINE_EXECUTOR(gpu::CLDiv);
DEFINE_EXECUTOR(gpu::CLFullyConnected);
DEFINE_EXECUTOR(gpu::CLGather);
DEFINE_EXECUTOR(gpu::CLMaxpool);
DEFINE_EXECUTOR(gpu::CLMean);
DEFINE_EXECUTOR(gpu::CLMul);
DEFINE_EXECUTOR(gpu::CLNormalization);
DEFINE_EXECUTOR(gpu::CLPad);
DEFINE_EXECUTOR(gpu::CLReduce);
DEFINE_EXECUTOR(gpu::CLRelu);
DEFINE_EXECUTOR(gpu::CLRelu1);
DEFINE_EXECUTOR(gpu::CLRelu6);
DEFINE_EXECUTOR(gpu::CLReshape);
DEFINE_EXECUTOR(gpu::CLResizeBilinear);
DEFINE_EXECUTOR(gpu::CLScale);
DEFINE_EXECUTOR(gpu::CLSlice);
DEFINE_EXECUTOR(gpu::CLSigmoid);
DEFINE_EXECUTOR(gpu::CLSoftmax);
DEFINE_EXECUTOR(gpu::CLSplit);
DEFINE_EXECUTOR(gpu::CLSqueeze);
DEFINE_EXECUTOR(gpu::CLStridedSlice);
DEFINE_EXECUTOR(gpu::CLSub);
DEFINE_EXECUTOR(gpu::CLTanh);
DEFINE_EXECUTOR(gpu::CLTFSlice);
DEFINE_EXECUTOR(gpu::CLTranspose);
DEFINE_EXECUTOR(gpu::CLUnpack);

namespace gpu {
EnnReturn OperationExecutor::execute(UDOperators& operators,
                                     UDBuffers& buffers,
                                     const model::memory::BufferTable& buffer_table) {
    UNUSED(buffers);
    UNUSED(buffer_table);

#ifndef ENN_BUILD_RELEASE
    bool dump_available = is_dump_available();
#endif
    ITensor::placeholder("OP EXECUTE STACK", ColorType::BOLDWHITE);
    for (int i = 0; i < operators->size(); i++) {
        auto& op = operators->at(i);
        std::cout << op->getName();
        if (i != operators->size() - 1)
            std::cout << " → ";
    }
    ITensor::placeholder("", ColorType::BOLDWHITE);
    for (int i = 0; i < operators->size(); i++) {
        auto& op = operators->at(i);
        ENN_DBG_PRINT("[%s]: op->getName() = %s\n", "GPU", op->getName().c_str());
        ITensor::placeholder("@ " + op->getName(), ColorType::BOLDWHITE);
        ITensor::placeholder("BEFORE EXECUTE", ColorType::YELLOW, true);
        for (const auto& tensor : op->getInTensors()) {
            CLTensor::checknull(std::static_pointer_cast<CLTensor>(tensor), "Intensor");
        }
        for (const auto& tensor : op->getOutTensors()) {
            CLTensor::checknull(std::static_pointer_cast<CLTensor>(tensor), "Outtensor");
        }
        if (op->execute(nullptr) != ENN_RET_SUCCESS) {
            ENN_DBG_PRINT("[%s]: %s execute() failed.\n", "GPU", op->getName().c_str());
            return ENN_RET_FAILED;
        }
        ITensor::placeholder("AFTER EXECUTE", ColorType::GREEN);
        for (const auto& tensor : op->getInTensors()) {
            CLTensor::checknull(std::static_pointer_cast<CLTensor>(tensor), "Intensor");
        }
        for (const auto& tensor : op->getOutTensors()) {
            CLTensor::checknull(std::static_pointer_cast<CLTensor>(tensor), "Outtensor");
        }
#ifndef ENN_BUILD_RELEASE
        if (dump_available) {
            dump_operator_output_gpu(op);
        }
#endif
    }

    return ENN_RET_SUCCESS;
}

void OperationExecutor::dump_operator_output_gpu(std::shared_ptr<enn::ud::UDOperator>& op) {
    std::string dump_path = "";
#ifdef __ANDROID__
    dump_path = DUMP_PATH;
#endif
    for (int m = 0; m < op->getOutTensors().size(); m++) {
        std::string shape;
        for (auto dim : op->getOutTensors()[m]->getDims()) {
            shape += "_" + std::to_string(dim);
        }

        std::string dump_file = dump_path + "GPU_" + "layer" + std::to_string(op->getId()) + "_" + op->getName() + "_out" +
                                std::to_string(m) + shape.c_str() + ".txt";
        op->getOutTensors()[m]->dumpTensorData(dump_file);
        ENN_DBG_COUT << "[DUMP] " << dump_file.c_str() << std::endl;
    }
}

void OperationExecutor::dump_operator_input_gpu(std::shared_ptr<enn::ud::UDOperator>& op) {
    std::string dump_path = "";
#ifdef __ANDROID__
    dump_path = DUMP_PATH;
#endif
    for (int m = 0; m < op->getInTensors().size(); m++) {
        std::string shape;
        for (auto dim : op->getInTensors()[m]->getDims()) {
            shape += "_" + std::to_string(dim);
        }

        std::string dump_file = dump_path + "GPU_" + "layer" + std::to_string(op->getId()) + "_" + op->getName() + "_in" +
                                std::to_string(m) + shape.c_str() + ".txt";
        op->getInTensors()[m]->dumpTensorData(dump_file);
        ENN_DBG_COUT << "[DUMP] " << dump_file.c_str() << std::endl;
    }
}
}  // namespace gpu

}  // namespace ud
}  // namespace enn
