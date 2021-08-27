#include <gtest/gtest.h>
#include "userdriver/cpu/operators/Pad.hpp"
#include "userdriver/common/op_test/test_utils.h"

namespace enn {
namespace ud {
namespace cpu {

class PadTester {
public:
    explicit PadTester(float threshold = ERROR_THRESHOLD) : error_threshold_(threshold) {}

    PadTester& InputDims(const Dim4& input_dims, const Dim4& output_dims) {
        input_dims_ = input_dims;
        output_dims_ = output_dims;
        return *this;
    }

    template <typename T>
    void TestRun(T* input_data, const T* reference_output_data, std::vector<int32_t>& padFront, std::vector<int32_t>& padEnd,
                 std::vector<float>& padValue, Status status = Status::SUCCESS) {
        PrecisionType precision = getPrecisionType(input_data);
        size_t output_size = GetDimSize(output_dims_);

        auto input_tensor = std::make_shared<NEONTensor<T>>(input_data, input_dims_, precision);
        auto output_tensor = std::make_shared<NEONTensor<T>>(output_dims_, precision);

        Pad _pad(precision);
        EXPECT_EQ(_pad.initialize(input_tensor, input_dims_.w, input_dims_.h, input_dims_.c,
                                         padFront, padEnd, padValue), Status::SUCCESS);
        EXPECT_EQ(_pad.execute(input_tensor, output_tensor), status);
        EXPECT_EQ(_pad.release(), Status::SUCCESS);

        if (status == Status::SUCCESS)
            Compare(output_tensor->getDataPtr().get(), reference_output_data, output_size, error_threshold_);
    }

private:
    float error_threshold_;
    Dim4 input_dims_;
    Dim4 output_dims_;
};

template <typename T>
T* GenerateInput(Dim4& input_dims) {
    size_t array_size = GetDimSize(input_dims);
    T* input = new T[array_size];
    for (int i = 0; i < array_size; i++) { input[i] = i + 1; }
    return input;
}

// Only one pad value is supported right now.
template <typename T>
T* GenerateRef(Dim4& input_dims, Dim4& output_dims, std::vector<int32_t>& PadFront, std::vector<int32_t>& PadEnd,
               std::vector<float>& PadValue, T* input_data) {
    size_t array_size = GetDimSize(output_dims);
    T* ref_data = new T[array_size];
    int index = 0;

    int32_t PadVal = (int32_t)(PadValue[0]);

    for (int i = 0; i < PadFront[1] * output_dims.h * output_dims.w; i++) { ref_data[index++] = PadVal; }

    for (int c = 0; c < input_dims.c; c++) {
        for (int h = 0; h < PadFront[2] * output_dims.w; h++) { ref_data[index++] = PadVal; }

        for (int h = 0; h < input_dims.h; h++) {
            for (int w = 0; w < PadFront[3]; w++) { ref_data[index++] = PadVal; }
            for (int w = 0; w < input_dims.w; w++) {
                ref_data[index++] = input_data[c * input_dims.w * input_dims.h + h * input_dims.w + w];
            }
            for (int w = PadFront[3] + input_dims.w; w < PadFront[3] + input_dims.w + PadEnd[3]; w++) {
                ref_data[index++] = PadVal;
            }
        }

        for (int h = (PadFront[2] + input_dims.h) * output_dims.w;
             h < (PadFront[2] + input_dims.h + PadEnd[2]) * output_dims.w; h++) {
            ref_data[index++] = PadVal;
        }
    }

    for (int i = (PadFront[1] + input_dims.c) * output_dims.h * output_dims.w;
         i < (PadFront[1] + input_dims.c + PadEnd[1]) * output_dims.h * output_dims.w; i++) {
        ref_data[index++] = PadVal;
    }
    return ref_data;
}

TEST(ENN_CPU_OP_UT_Pad, INT16_INPUT) {
    std::vector<int32_t> PadFront{0, 1, 2, 3};
    std::vector<int32_t> PadEnd{0, 2, 3, 4};
    std::vector<float> PadVal{1};
    Dim4 input_dims = {1, 2, 3, 4};
    Dim4 output_dims = {1, 5, 8, 11};

    int16_t* input_data = GenerateInput<int16_t>(input_dims);
    int16_t* reference_output_data = GenerateRef<int16_t>(input_dims, output_dims, PadFront, PadEnd, PadVal, input_data);

    PadTester().InputDims(input_dims, output_dims)
        .TestRun(input_data, reference_output_data, PadFront, PadEnd, PadVal);
    delete[] input_data;
    delete[] reference_output_data;
}

TEST(ENN_CPU_OP_UT_Pad, UINT8_INPUT) {
    std::vector<int32_t> PadFront{0, 1, 1, 2};
    std::vector<int32_t> PadEnd{0, 0, 2, 1};
    std::vector<float> PadVal{0};
    Dim4 input_dims = {1, 2, 3, 4};
    Dim4 output_dims = {1, 3, 6, 7};

    uint8_t* input_data = GenerateInput<uint8_t>(input_dims);
    uint8_t* reference_output_data = GenerateRef<uint8_t>(input_dims, output_dims, PadFront, PadEnd, PadVal, input_data);

    PadTester().InputDims(input_dims, output_dims)
        .TestRun(input_data, reference_output_data, PadFront, PadEnd, PadVal);
    delete[] input_data;
    delete[] reference_output_data;
}

TEST(ENN_CPU_OP_UT_Pad, INVALID_INPUT) {
    std::vector<int32_t> PadFront{0, 1, 2, 3};
    std::vector<int32_t> PadEnd{2, 3, 4};
    std::vector<float> PadVal{2};
    Dim4 input_dims = {1, 1, 0, 1};
    Dim4 output_dims = {1, 5, 8, 11};

    int16_t* input_data = GenerateInput<int16_t>(input_dims);
    int16_t* reference_output_data = GenerateRef<int16_t>(input_dims, output_dims, PadFront, PadEnd, PadVal, input_data);

    PadTester().InputDims(input_dims, output_dims)
        .TestRun(input_data, reference_output_data, PadFront, PadEnd, PadVal, Status::INVALID_PARAMS);
    delete[] input_data;
    delete[] reference_output_data;
}

TEST(ENN_CPU_OP_UT_Pad, UNSUPPORTED_DATA_TYPE) {
    std::vector<int32_t> PadFront{0, 1, 2, 3};
    std::vector<int32_t> PadEnd{0, 2, 3, 4};
    std::vector<float> PadVal{1};
    Dim4 input_dims = {1, 1, 0, 1};
    Dim4 output_dims = {1, 5, 8, 11};

    uint16_t* input_data = GenerateInput<uint16_t>(input_dims);
    uint16_t* reference_output_data = GenerateRef<uint16_t>(input_dims, output_dims, PadFront, PadEnd, PadVal, input_data);

    PadTester().InputDims(input_dims, output_dims)
        .TestRun(input_data, reference_output_data, PadFront, PadEnd, PadVal, Status::FAILURE);
    delete[] input_data;
    delete[] reference_output_data;
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
