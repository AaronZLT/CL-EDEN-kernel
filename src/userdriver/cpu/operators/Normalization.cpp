#include "Normalization.hpp"

namespace enn {
namespace ud {
namespace cpu {

Normalization::Normalization(const PrecisionType& precision) {
    precision_ = precision;
    bgr_transpose_ = 0;
}

Status Normalization::initialize(const std::shared_ptr<ITensor> input, const std::shared_ptr<ITensor> mean,
                                 const std::shared_ptr<ITensor> scale, const uint8_t& bgr_transpose) {
    ENN_UNUSED(input);
    this->mean_ = std::static_pointer_cast<NEONTensor<float>>(mean);
    this->scale_ = std::static_pointer_cast<NEONTensor<float>>(scale);
    this->bgr_transpose_ = bgr_transpose;
    return Status::SUCCESS;
}

template <typename T>
Status Normalization::executeKernel(const std::shared_ptr<NEONTensor<T>> input_tensor,
                                    std::shared_ptr<NEONTensor<float>> output_tensor) {
    T* input_data = input_tensor->getBufferPtr();
    float* output_data = output_tensor->getBufferPtr();
    float* mean = mean_->getDataPtr().get();
    float* scale = scale_->getDataPtr().get();

    uint32_t i, j, k, n;
    T num;
    uint32_t idx;
    uint32_t batch = input_tensor->getDim().n;
    uint32_t channel = input_tensor->getDim().c;
    uint32_t height = input_tensor->getDim().h;
    uint32_t width = input_tensor->getDim().w;

    DEBUG_PRINT("NCHW = %d,%d,%d,%d\n", batch, channel, height, width);
    for (uint32_t c = 0; c < channel; c++) {
        DEBUG_PRINT("mean[%d] = %f, scale[%d] = %f\n", c, mean[c], c, scale[c]);
    }

    // Nice to have, ToDo(empire.jung, TBD): What is bgr_transpose_? After check its identity, refactoring this logic.
    if (bgr_transpose_ == 1) {
        // normalization & bgr_transpose
        enum { B = 0, G = 1, R = 2 };
        float normalized;
        int b_idx = B;
        int g_idx = height * width * G;
        int r_idx = height * width * R;
        // optimize the reminder
        for (n = 0; n < batch; n++) {
            for (i = 0; i < channel; i++) {
                for (j = 0; j < height; j++) {
                    for (k = 0; k < width; k++) {
                        idx = (n * channel * height * width) + (i * height * width) + (j * width) + k;
                        num = input_data[idx];
                        normalized = ((float)num - mean[i]) * scale[i];
                        if (i == B) {
                            output_data[b_idx++] = normalized;
                        } else if (i == G) {
                            output_data[g_idx++] = normalized;
                        } else if (i == R) {
                            output_data[r_idx++] = normalized;
                        } else {
                            DEBUG_PRINT("Channel greater > 3 is not supported for bgr transpose");
                            return Status::FAILURE;
                        }
                    }
                }
            }
        }
    } else {
        // just normalization
        for (n = 0; n < batch; n++) {
            for (i = 0; i < channel; i++) {
                for (j = 0; j < height; j++) {
                    for (k = 0; k < width; k++) {
                        idx = (n * channel * height * width) + (i * height * width) + (j * width) + k;
                        num = input_data[idx];
                        output_data[idx] = (static_cast<float>(num) - mean[i]) * scale[i];
                    }
                }
            }
        }
    }
    return Status::SUCCESS;
}

Status Normalization::execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) {
    auto output_tensor = std::static_pointer_cast<NEONTensor<float>>(output);
    switch (input->getDataType()) {
        case DataType::UINT8: {
            auto input_tensor = std::static_pointer_cast<NEONTensor<uint8_t>>(input);
            return executeKernel(input_tensor, output_tensor);
        }
        case DataType::FLOAT: {
            auto input_tensor = std::static_pointer_cast<NEONTensor<float>>(input);
            return executeKernel(input_tensor, output_tensor);
        }
        default: {
            ERROR_PRINT("Data type is not supported\n");
            return Status::FAILURE;
        }
    }
}

Status Normalization::release() {
    return Status::SUCCESS;
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
