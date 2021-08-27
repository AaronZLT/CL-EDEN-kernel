#include "Concat.hpp"

namespace enn {
namespace ud {
namespace cpu {

Concat::Concat(const PrecisionType& precision) {
    precision_ = precision;
    inputNum = 0;
    number = 0;
    channel = 0;
    height = 0;
    width = 0;
    axis = 0;
}

Status Concat::initialize(const std::vector<std::shared_ptr<ITensor>> input, const int32_t& inputNum, const int32_t& number,
                          const int32_t& channel, const int32_t& height, const int32_t& width, const int32_t& axis) {
    ENN_UNUSED(input);
    this->inputNum = inputNum;
    this->number = number;
    this->channel = channel;
    this->height = height;
    this->width = width;
    this->axis = axis;
    return Status::SUCCESS;
}

template <typename T>
Status Concat::executeKernel(const std::vector<std::shared_ptr<ITensor>> input_tensor,
                             std::shared_ptr<NEONTensor<T>> output_tensor) {
    T* output = output_tensor->getBufferPtr();
    T** inputs = new T*[input_tensor.size()];
    for (uint32_t i = 0; i < input_tensor.size(); i++) {
        auto input = std::static_pointer_cast<NEONTensor<T>>(input_tensor.at(i));
        inputs[i] = input->getBufferPtr();
    }

    // only 2 input is handled
    if ((!inputs) || (!inputs[0]) || (!inputs[1]) || (!output) || (inputNum != 2) || (channel <= 0) || (height <= 0) ||
        (width <= 0)) {
        ERROR_PRINT("Invalid parameter\n");
        delete[] inputs;
        return Status::INVALID_PARAMS;
    }

    uint32_t loopNum;
    uint32_t interval;
    if (axis == 0) {
        loopNum = 1;
        interval = number * channel * height * width;
    } else if (axis == 1) {
        loopNum = number;
        interval = channel * height * width;
    } else if (axis == 2) {
        loopNum = number * channel;
        interval = height * width;
    } else if (axis == 3) {
        loopNum = number * channel * height;
        interval = width;
    } else {
        ERROR_PRINT("Un-supported axis value : %d\n", axis);
        delete[] inputs;
        return Status::INVALID_PARAMS;
    }

    for (uint32_t i = 0; i < inputNum; ++i) {
        for (uint32_t j = 0; j < loopNum; ++j) {
            uint32_t inIdx = j * interval;
            uint32_t outIdx = (i + j * inputNum) * interval;
            memcpy(output + outIdx, inputs[i] + inIdx, sizeof(uint8_t) * interval);
        }
    }
    delete[] inputs;
    return Status::SUCCESS;
}

Status Concat::execute(const std::vector<std::shared_ptr<ITensor>> input, std::shared_ptr<ITensor> output) {
    switch (input[0]->getDataType()) {
        case DataType::UINT8: {
            DEBUG_PRINT("DataType::UINT8\n");
            auto output_tensor = std::static_pointer_cast<NEONTensor<uint8_t>>(output);
            return executeKernel(input, output_tensor);
        }
        default: {
            ERROR_PRINT("Data type is not supported\n");
            return Status::FAILURE;
        }
    }
}

Status Concat::release() {
    return Status::SUCCESS;
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
