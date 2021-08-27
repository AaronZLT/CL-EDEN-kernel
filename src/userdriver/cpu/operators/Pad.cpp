#include "Pad.hpp"

namespace enn {
namespace ud {
namespace cpu {
namespace {
enum { NUMBER = 0, CHANNEL, HEIGHT, WIDTH, SIZE };
}

#define PUT_PADDING(LOOP, OUTPUT, VALUE, SIZE)   \
    for (int32_t idx = 0; idx < (LOOP); idx++) { \
        *OUTPUT++ = VALUE;                       \
        SIZE++;                                  \
    }

Pad::Pad(const PrecisionType& precision) {
    precision_ = precision;
    width = 0;
    height = 0;
    channel = 0;
}

Status Pad::initialize(const std::shared_ptr<ITensor> input, const int32_t& width, const int32_t& height,
                       const int32_t& channel, const std::vector<int32_t>& padFront, const std::vector<int32_t>& padEnd,
                       const std::vector<float>& padValue) {
    ENN_UNUSED(input);
    this->width = width;
    this->height = height;
    this->channel = channel;
    this->padFront = padFront;
    this->padEnd = padEnd;
    this->padValue = padValue;
    return Status::SUCCESS;
}

template <typename T>
Status Pad::executeKernel(const std::shared_ptr<NEONTensor<T>> input_tensor, std::shared_ptr<NEONTensor<T>> output_tensor) {
    T* input_data = input_tensor->getBufferPtr();
    T* output_data = output_tensor->getBufferPtr();

    if ((!input_data) || (!output_data) || (padFront.size() != padEnd.size())) {
        ERROR_PRINT("Invalid parameter\n");
        return Status::INVALID_PARAMS;
    }

    for (uint32_t idx = 0; idx < padFront.size(); idx++) {
        DEBUG_PRINT("padFront[%d]=[%d]\n", idx, padFront[idx]);
        DEBUG_PRINT("padEnd[%d]=[%d]\n", idx, padEnd[idx]);
    }
    for (uint32_t idx = 0; idx < padValue.size(); idx++) {
        DEBUG_PRINT("padValue[%d]=[%f]\n", idx, padValue[idx]);
    }

    int32_t validLeft[3];
    int32_t validRight[3];

    int32_t padFrontWidth = padFront[WIDTH];
    int32_t padEndWidth = padEnd[WIDTH];
    int32_t padFrontHeight = padFront[HEIGHT];
    int32_t padEndHeight = padEnd[HEIGHT];
    int32_t padFrontChannel = padFront[CHANNEL];
    int32_t padEndChannel = padEnd[CHANNEL];

    validLeft[0] = padFrontWidth;
    validRight[0] = padFrontWidth + width;
    validLeft[1] = padFrontHeight;
    validRight[1] = padFrontHeight + height;
    validLeft[2] = padFrontChannel;
    validRight[2] = padFrontChannel + channel;

    int32_t outputWidth = padFrontWidth + width + padEndWidth;
    int32_t outputHeight = padFrontHeight + height + padEndHeight;

    // Padding value can be anything.
    // @todo Now only one padding value is supported.
    static const int32_t MAX_NUM_OF_PAD_VALUE = 1;
    float padValues[MAX_NUM_OF_PAD_VALUE];
    uint32_t numOfPadValue = padValue.size();
    if (numOfPadValue > MAX_NUM_OF_PAD_VALUE) {
        ERROR_PRINT("Oops, now Pad for NPU only supports for [%d] padValue...other values are ignored..\n",
                    MAX_NUM_OF_PAD_VALUE);
        numOfPadValue = MAX_NUM_OF_PAD_VALUE;
    }
    for (uint32_t idx = 0; idx < numOfPadValue; idx++) {
        padValues[idx] = padValue[idx];
    }
    for (uint32_t idx = numOfPadValue; idx < MAX_NUM_OF_PAD_VALUE; idx++) {
        padValues[idx] = 0;
    }
    float paddingValue = padValues[0];

    int _size_ = 0;
    int imgSize = height * width;

    // First, put front padding for channel
    PUT_PADDING(padFrontChannel * outputHeight * outputWidth, output_data, paddingValue, _size_);

    // Second, memcpy real data
    for (int32_t c = 0; c < channel; c++) {
        // First, put front padding for height
        PUT_PADDING(padFrontHeight * outputWidth, output_data, paddingValue, _size_);

        // Second, memcpy real data
        for (int32_t h = 0; h < height; h++) {
            // First, put front padding for width
            PUT_PADDING(padFrontWidth, output_data, paddingValue, _size_);

            // Second, memcpy real data
            for (int32_t w = 0; w < width; w++) {
                *output_data++ = input_data[c * imgSize + h * width + w];
            }
            _size_ += width;

            // Third, put end padding for width
            PUT_PADDING(padEndWidth, output_data, paddingValue, _size_);
        }

        // Third, put end padding for height
        PUT_PADDING(padEndHeight * outputWidth, output_data, paddingValue, _size_);
    }

    // Third, put end padding for channel
    PUT_PADDING(padEndChannel * outputHeight * outputWidth, output_data, paddingValue, _size_);

    return Status::SUCCESS;
}

Status Pad::execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) {
    switch (input->getDataType()) {
        case DataType::INT16: {
            DEBUG_PRINT("DataType::INT16\n");
            auto input_tensor = std::static_pointer_cast<NEONTensor<int16_t>>(input);
            auto output_tensor = std::static_pointer_cast<NEONTensor<int16_t>>(output);
            return executeKernel(input_tensor, output_tensor);
        }
        case DataType::INT8:
        case DataType::UINT8: {
            DEBUG_PRINT("DataType::INT8 or DataType::UINT8\n");
            auto input_tensor = std::static_pointer_cast<NEONTensor<int8_t>>(input);
            auto output_tensor = std::static_pointer_cast<NEONTensor<int8_t>>(output);
            return executeKernel(input_tensor, output_tensor);
        }
        default: {
            ERROR_PRINT("Data type is not supported\n");
            return Status::FAILURE;
        }
    }
}

Status Pad::release() {
    return Status::SUCCESS;
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
