#include <gtest/gtest.h>
#include "userdriver/cpu/common/NEONTensor.hpp"
#include "userdriver/cpu/operators/AsymmDequantization.hpp"
#include "userdriver/common/op_test/test_utils.h"

namespace enn {
namespace ud {
namespace cpu {

class AsymmDequantizationTester {
public:
    explicit AsymmDequantizationTester(float threshold = ERROR_THRESHOLD) : error_threshold_(threshold) {}

    AsymmDequantizationTester& InputDims(const Dim4& input_dims) {
        input_dims_ = input_dims;
        output_dims_ = input_dims;
        return *this;
    }

    AsymmDequantizationTester& Scale(const float& scale) {
        scale_ = scale;
        return *this;
    }

    AsymmDequantizationTester& ZeroPoint(const int32_t& zero_point) {
        zero_point_ = zero_point;
        return *this;
    }

    template <typename T>
    void TestRun(T* input_data, const float* reference_output_data, Status status = Status::SUCCESS) {
        PrecisionType precision = getPrecisionType(input_data);
        size_t output_size = GetDimSize(output_dims_);

        auto input_tensor = std::make_shared<NEONTensor<T>>(input_data, input_dims_, precision, scale_, zero_point_);
        auto output_tensor = std::make_shared<NEONTensor<float>>(output_dims_, PrecisionType::FP32);

        AsymmDequantization _asymm_dequantize(precision);
        int32_t numOfData = input_dims_.c * input_dims_.w * input_dims_.h;
        uint32_t imgSize = input_dims_.w * input_dims_.h;
        EXPECT_EQ(_asymm_dequantize.initialize(input_tensor, numOfData, scale_, zero_point_, imgSize),
                  Status::SUCCESS);
        EXPECT_EQ(_asymm_dequantize.execute(input_tensor, output_tensor), status);
        EXPECT_EQ(_asymm_dequantize.release(), Status::SUCCESS);

        if (status == Status::SUCCESS)
            Compare(output_tensor->getDataPtr().get(), reference_output_data, output_size, error_threshold_);
    }

private:
    float error_threshold_;
    Dim4 input_dims_;
    Dim4 output_dims_;
    float scale_;
    int32_t zero_point_;
};

inline float calculate_scale(float min_value, float max_value) {
    return (max_value - min_value) / 255.0;
}

inline int32_t calculate_zero_point(float min_value, float scale) {
    return std::min(255, std::max(0, static_cast<int32_t>(round(0 - min_value / scale))));
}

TEST(ENN_CPU_OP_UT_AsymmDequantization, UINT8_INPUT_CASE0) {
    uint8_t input_data[] = {0, 1, 2, 3, 4, 251, 252, 253, 254, 255};
    float reference_output_data[] = {-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64};
    const float min_value = -63.5;
    const float max_value = 64.0;
    const float scale = calculate_scale(min_value, max_value);
    const int32_t zero_point = calculate_zero_point(min_value, scale);

    Dim4 input_dims_ = {1, 1, 2, 5};
    AsymmDequantizationTester()
        .InputDims(input_dims_)
        .Scale(scale)
        .ZeroPoint(zero_point)
        .TestRun(input_data, reference_output_data);
}

TEST(ENN_CPU_OP_UT_AsymmDequantization, UINT8_INPUT_CASE1) {
    uint8_t input_data[] = {0, 32, 128, 255};
    float reference_output_data[] = {0.0, 32.0, 128.0, 255.0};
    const float min_value = 0;
    const float max_value = 255;
    const float scale = calculate_scale(min_value, max_value);
    const int32_t zero_point = calculate_zero_point(min_value, scale);

    Dim4 input_dims_ = {1, 1, 2, 2};
    AsymmDequantizationTester()
        .InputDims(input_dims_)
        .Scale(scale)
        .ZeroPoint(zero_point)
        .TestRun(input_data, reference_output_data);
}

TEST(ENN_CPU_OP_UT_AsymmDequantization, INT8_INPUT_CASE0) {
    int8_t input_data[] = {-128, -96, 0, 127};
    float reference_output_data[] = {0.0, 32.0, 128.0, 255.0};

    Dim4 input_dims_ = {1, 2, 2, 1};
    AsymmDequantizationTester()
        .InputDims(input_dims_)
        .Scale(1.0)
        .ZeroPoint(-128)
        .TestRun(input_data, reference_output_data);
}

TEST(ENN_CPU_OP_UT_AsymmDequantization, INT8_INPUT_CASE1) {
    int8_t input_data[] = {-128, -127, -126, -125, -124, 123, 124, 125, 126, 127};
    float reference_output_data[] = {-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64};

    Dim4 input_dims_ = {1, 1, 2, 5};
    AsymmDequantizationTester()
        .InputDims(input_dims_)
        .Scale(0.5)
        .ZeroPoint(-1)
        .TestRun(input_data, reference_output_data);
}

TEST(ENN_CPU_OP_UT_AsymmDequantization, INVALID_INPUT) {
    int8_t* input_data = NULL;
    float* reference_output_data = NULL;

    Dim4 input_dims_ = {0, 0, 0, 0};
    AsymmDequantizationTester()
        .InputDims(input_dims_)
        .Scale(0.5)
        .ZeroPoint(-1)
        .TestRun(input_data, reference_output_data, Status::INVALID_PARAMS);
}

TEST(ENN_CPU_OP_UT_AsymmDequantization, INT16_INPUT) {
    int16_t input_data[] = {0, 32, 128, 255};
    float reference_output_data[] = {-63.0, -47.0, 1.0, 64.5};

    Dim4 input_dims_ = {1, 1, 2, 2};
    AsymmDequantizationTester()
        .InputDims(input_dims_)
        .Scale(0.5)
        .ZeroPoint(126)
        .TestRun(input_data, reference_output_data);
}

TEST(ENN_CPU_OP_UT_AsymmDequantization, UNSUPPORTED_DATA_TYPE) {
    uint16_t input_data[] = {0, 32, 128, 255};
    float reference_output_data[] = {0.0, 32.0, 128.0, 255.0};
    const float min_value = 0;
    const float max_value = 255;
    const float scale = calculate_scale(min_value, max_value);
    const int32_t zero_point = calculate_zero_point(min_value, scale);

    Dim4 input_dims_ = {1, 1, 2, 2};
    AsymmDequantizationTester()
        .InputDims(input_dims_)
        .Scale(scale)
        .ZeroPoint(zero_point)
        .TestRun(input_data, reference_output_data, Status::FAILURE);
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
