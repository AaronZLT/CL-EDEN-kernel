#include <gtest/gtest.h>
#include "userdriver/cpu/operators/Quantization.hpp"
#include "userdriver/common/op_test/test_utils.h"

#define FRAC_DIM(x) \
    { 1, sizeof(x) / sizeof(int32_t), 1, 1 }

namespace enn {
namespace ud {
namespace cpu {

class QuantizationTester {
public:
    explicit QuantizationTester(float threshold = ERROR_THRESHOLD) : error_threshold_(threshold) {}

    QuantizationTester& InputDims(const Dim4& input_dims) {
        input_dims_ = input_dims;
        output_dims_ = input_dims;
        return *this;
    }

    template <typename T>
    void TestRun(float* input_data, int32_t* frac_len, Dim4& frac_dims, const T* reference_output_data,
                   Status status = Status::SUCCESS) {
        PrecisionType precision = getPrecisionType(reference_output_data);
        size_t output_size = GetDimSize(output_dims_);
        auto input_tensor = std::make_shared<NEONTensor<float>>(input_data, input_dims_, PrecisionType::FP32);
        auto output_tensor = std::make_shared<NEONTensor<T>>(output_dims_, precision);
        auto frac_tensor = std::make_shared<NEONTensor<int>>(frac_len, frac_dims, PrecisionType::INT32);

        Quantization _quantize(precision);
        EXPECT_EQ(_quantize.initialize(input_tensor, input_dims_.c, input_dims_.h, input_dims_.w, frac_tensor),
                  Status::SUCCESS);
        EXPECT_EQ(_quantize.execute(input_tensor, output_tensor), status);
        EXPECT_EQ(_quantize.release(), Status::SUCCESS);

        Compare(output_tensor->getDataPtr().get(), reference_output_data, output_size, error_threshold_);
    }

private:
    float error_threshold_;
    Dim4 input_dims_;
    Dim4 output_dims_;
};

TEST(ENN_CPU_OP_UT_Quantization, BASIC_INPUT) {
    float input_data[] = {2.3, -1.5};
    uint8_t reference_output_data[] = {9, 244};

    Dim4 input_dims_ = {1, 2, 1, 1};
    int32_t frac_len[] = {2, 3};
    Dim4 frac_dim = FRAC_DIM(frac_len);
    QuantizationTester().InputDims(input_dims_).TestRun(input_data, frac_len, frac_dim, reference_output_data);
}

TEST(ENN_CPU_OP_UT_Quantization, UINT8_INPUT_NEGATIVE) {
    float input_data[] = {-0.2, -2.1, -2.6, -3.2, -13.7, -15.1, -3.7, -1.8, -9.1};
    uint8_t reference_output_data[] = {255, 248, 246, 230, 146, 135, 197, 227, 238};

    Dim4 input_dims_ = {1, 3, 3, 1};
    int32_t frac_len[] = {2, 3, 4};
    Dim4 frac_dim = FRAC_DIM(frac_len);
    QuantizationTester().InputDims(input_dims_).TestRun(input_data, frac_len, frac_dim, reference_output_data);
}

TEST(ENN_CPU_OP_UT_Quantization, UINT8_INPUT_POSITIVE) {
    float input_data[] = {0.1, 0.6, 2.4, 2.7, 6.1, 6.8, 12.5, 16.1, 0.0};
    uint8_t reference_output_data[] = {0, 2, 10, 11, 24, 27, 50, 64, 0};

    Dim4 input_dims_ = {1, 3, 1, 3};
    int32_t frac_len[] = {2, 2, 2};
    Dim4 frac_dim = FRAC_DIM(frac_len);
    QuantizationTester().InputDims(input_dims_).TestRun(input_data, frac_len, frac_dim, reference_output_data);
}

TEST(ENN_CPU_OP_UT_Quantization, UINT8_INPUT_MIXED) {
    float input_data[] = {0.1, 0.6, -2.6, 2.7, 6.1, 6.8, -2.1, -0.2, 0.0};
    uint8_t reference_output_data[] = {0, 2, 246, 11, 24, 27, 248, 255, 0};

    Dim4 input_dims_ = {1, 3, 1, 3};
    int32_t frac_len[] = {2, 2, 2};
    Dim4 frac_dim = FRAC_DIM(frac_len);
    QuantizationTester().InputDims(input_dims_).TestRun(input_data, frac_len, frac_dim, reference_output_data);
}

TEST(ENN_CPU_OP_UT_Quantization, INT16_INPUT_POSITIVE) {
    float input_data[] = {0.1, 0.6, 2.4, 2.7, 6.1, 6.8, 12.5, 16.1, 0.0};
    int16_t reference_output_data[] = {0, 2, 10, 11, 24, 27, 50, 64, 0};

    Dim4 input_dims_ = {1, 3, 1, 3};
    int32_t frac_len[] = {2, 2, 2};
    Dim4 frac_dim = FRAC_DIM(frac_len);
    QuantizationTester().InputDims(input_dims_).TestRun(input_data, frac_len, frac_dim, reference_output_data);
}

TEST(ENN_CPU_OP_UT_Quantization, INVALID_INPUT) {  // channel = 0
    float input_data[] = {0.1, 0.6, 2.4, 2.7, 6.1, 6.8, 12.5, 16.1, 0.0};
    uint8_t reference_output_data[] = {0, 2, 10, 11, 24, 27, 50, 64, 0};

    Dim4 input_dims_ = {1, 0, 1, 3};
    int32_t frac_len[] = {2, 2, 2};
    Dim4 frac_dim = FRAC_DIM(frac_len);
    QuantizationTester()
        .InputDims(input_dims_)
        .TestRun(input_data, frac_len, frac_dim, reference_output_data, Status::INVALID_PARAMS);
}

TEST(ENN_CPU_OP_UT_Quantization, UNSUPPORTED_DATA_TYPE) {  // uint16_t is not supported data type
    float input_data[] = {0.1, 0.6, 2.4, 2.7, 6.1, 6.8, 12.5, 16.1, 0.0};
    uint16_t reference_output_data[] = {0, 2, 10, 11, 24, 27, 50, 64, 0};

    Dim4 input_dims_ = {1, 0, 1, 3};
    int32_t frac_len[] = {2, 2, 2};
    Dim4 frac_dim = FRAC_DIM(frac_len);
    QuantizationTester()
        .InputDims(input_dims_)
        .TestRun(input_data, frac_len, frac_dim, reference_output_data, Status::FAILURE);
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
