#include "Flatten.hpp"

namespace enn {
namespace ud {
namespace cpu {

Flatten::Flatten(const PrecisionType& precision) {
    precision_ = precision;
    axis_ = 0;
    end_axis_ = 0;
}

Status Flatten::initialize(const int32_t& axis, const int32_t& end_axis) {
    this->axis_ = axis;
    this->end_axis_ = end_axis;
    DEBUG_PRINT("axis = %d, end_axis = %d\n", axis, end_axis);
    return Status::SUCCESS;
}

template <typename T>
Status Flatten::executeKernel(const std::shared_ptr<NEONTensor<T>> input_tensor,
                              std::shared_ptr<NEONTensor<T>> output_tensor, std::vector<uint32_t>* output_dim) {
    T* input_data = input_tensor->getBufferPtr();
    T* output_data = output_tensor->getBufferPtr();

    if (output_dim != nullptr) {
        int input_axes = 4;
        int flattened_dim = 1;
        int output_shape = 0;
        int start_axis_true = axis_ < 0 ? axis_ + input_axes : axis_;
        int end_axis_true = end_axis_ < 0 ? end_axis_ + input_axes : end_axis_;
        Dim4 in_dim = input_tensor->getDim();
        for (int i = 0; i < start_axis_true; i++) {
            output_shape = getDim(in_dim, i);
            output_dim->push_back(output_shape);
        }
        for (int i = start_axis_true; i <= end_axis_true; i++) {
            output_shape = getDim(in_dim, i);
            flattened_dim *= output_shape;
        }
        output_dim->push_back(flattened_dim);
        for (int i = end_axis_true + 1; i < input_axes; i++) {
            output_shape = getDim(in_dim, i);
            output_dim->push_back(output_shape);
        }
    }

    uint32_t size = input_tensor->getTotalSizeFromDims();
    memcpy(output_data, input_data, sizeof(T) * size);

    return Status::SUCCESS;
}

Status Flatten::execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output,
                        std::vector<uint32_t>* output_dim) {
    switch (input->getDataType()) {
        case DataType::FLOAT: {
            DEBUG_PRINT("DataType::FLOAT\n");
            auto input_tensor = std::static_pointer_cast<NEONTensor<float>>(input);
            auto output_tensor = std::static_pointer_cast<NEONTensor<float>>(output);
            return executeKernel(input_tensor, output_tensor, output_dim);
        }
        case DataType::INT16: {
            DEBUG_PRINT("DataType::INT16\n");
            auto input_tensor = std::static_pointer_cast<NEONTensor<int16_t>>(input);
            auto output_tensor = std::static_pointer_cast<NEONTensor<int16_t>>(output);
            return executeKernel(input_tensor, output_tensor, output_dim);
        }
        case DataType::INT8:
        case DataType::UINT8: {
            DEBUG_PRINT("DataType::INT8 or DataType::UINT8\n");
            auto input_tensor = std::static_pointer_cast<NEONTensor<int8_t>>(input);
            auto output_tensor = std::static_pointer_cast<NEONTensor<int8_t>>(output);
            return executeKernel(input_tensor, output_tensor, output_dim);
        }
        default: {
            ERROR_PRINT("Data type is not supported\n");
            return Status::FAILURE;
        }
    }
}

Status Flatten::release() {
    return Status::SUCCESS;
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
