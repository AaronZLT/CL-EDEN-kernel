#include "Softmax.hpp"

namespace enn {
namespace ud {
namespace cpu {

Softmax::Softmax(const PrecisionType& precision) {
    precision_ = precision;
    width = 0;
    height = 0;
    channel = 0;
    number = 0;
    beta = 0.f;
    axis = 0;
}

Status Softmax::initialize(const std::shared_ptr<ITensor> input, const int32_t& width, const int32_t& height,
                           const int32_t& channel, const int32_t& number, const float& beta, const int32_t& axis) {
    ENN_UNUSED(input);
    this->width = width;
    this->height = height;
    this->channel = channel;
    this->number = number;
    this->beta = beta;
    this->axis = axis;
    return Status::SUCCESS;
}

template <typename T>
Status Softmax::executeKernel(const std::shared_ptr<NEONTensor<T>> input_tensor,
                              std::shared_ptr<NEONTensor<T>> output_tensor) {
    T* input_data = input_tensor->getBufferPtr();
    T* output_data = output_tensor->getBufferPtr();

    if ((!input_data) || (!output_data)) {
        ERROR_PRINT("Invalid parameter\n");
        return Status::INVALID_PARAMS;
    }

    DEBUG_PRINT("beta=[%f], axis=[%d]\n", beta, axis);

    int32_t out_number = 0;
    int32_t channel_number = 0;
    int32_t inner_number = 0;
    int32_t dim = 0;

    uint32_t input_batch = number;
    uint32_t input_channel = channel;
    uint32_t input_height = height;
    uint32_t input_width = width;
    size_t bottom_size = input_batch * input_channel * input_height * input_width;

    // Nice to have, ToDo(empire.jung, TBD): make it member(). Use member variable for out_number, channel_number, ....
    int new_axis = axis;
    if (new_axis < 0) {
        new_axis = axis + 4;
    }
    if (new_axis == 0) {
        out_number = 1;
        channel_number = input_batch;
        inner_number = input_channel * input_height * input_width;
        dim = bottom_size / out_number;
    } else if (new_axis == 1) {
        out_number = input_batch;
        channel_number = input_channel;
        inner_number = input_height * input_width;
        dim = bottom_size / out_number;
    } else if (new_axis == 2) {
        out_number = input_batch * input_channel;
        channel_number = input_height;
        inner_number = input_width;
        dim = bottom_size / out_number;
    } else {
        out_number = input_batch * input_channel * input_height;
        channel_number = input_width;
        inner_number = 1;
        dim = bottom_size / out_number;
    }

    std::shared_ptr<float> channel_max_ptr(new float[inner_number], std::default_delete<float[]>());
    auto channel_max = channel_max_ptr.get();
    float channel_sum = 0.0;

    // Nice to have, ToDo(empire.jung, TBD): Make some comment for each inner for-loop. Or make it as a function.
    // Optional: Performance Opt.
    for (int j = 0; j < out_number; j++) {
        int offset = j * dim;
        for (int i = 0; i < inner_number; i++) {
            channel_max[i] = input_data[i + offset];
            for (int c = 1; c < channel_number; c++) {
                if (input_data[offset + i + inner_number * c] > channel_max[i]) {
                    channel_max[i] = input_data[offset + i + inner_number * c];
                }
            }
        }
        for (int i = 0; i < inner_number; i++) {
            channel_sum = 0.0;
            for (int c = 0; c < channel_number; c++) {
                int data_offset = offset + i + inner_number * c;
                output_data[data_offset] = exp((input_data[data_offset] - channel_max[i]) * beta);
                channel_sum += output_data[data_offset];
            }

            for (int c = 0; c < channel_number; c++) {
                int data_offset = offset + i + inner_number * c;
                output_data[data_offset] /= channel_sum;
            }
        }
    }
    return Status::SUCCESS;
}

Status Softmax::execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) {
    switch (input->getDataType()) {
        case DataType::FLOAT: {
            DEBUG_PRINT("DataType::FLOAT\n");
            auto input_tensor = std::static_pointer_cast<NEONTensor<float>>(input);
            auto output_tensor = std::static_pointer_cast<NEONTensor<float>>(output);
            return executeKernel(input_tensor, output_tensor);
        }
        default: {
            ERROR_PRINT("Data type is not supported\n");
            return Status::FAILURE;
        }
    }
}

Status Softmax::release() {
    return Status::SUCCESS;
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
