#include <gtest/gtest.h>
#include "userdriver/cpu/operators/AsymmQuantization.hpp"
#include "userdriver/common/op_test/test_utils.h"

namespace enn {
namespace ud {
namespace cpu {

class AsymmQuantizationTester {
public:
    explicit AsymmQuantizationTester(float threshold = ERROR_THRESHOLD) : error_threshold_(threshold) {}

    AsymmQuantizationTester& InputDims(const Dim4& input_dims) {
        input_dims_ = input_dims;
        output_dims_ = input_dims;
        return *this;
    }

    AsymmQuantizationTester& Scale(const float& scale) {
        scale_ = scale;
        return *this;
    }

    AsymmQuantizationTester& ZeroPoint(const int32_t& zero_point) {
        zero_point_ = zero_point;
        return *this;
    }

    template <typename T>
    void TestRun(float* input_data, const T* reference_output_data, Status status = Status::SUCCESS) {
        PrecisionType precision = getPrecisionType(reference_output_data);

        size_t output_size = GetDimSize(output_dims_);
        auto input_tensor = std::make_shared<NEONTensor<float>>(input_data, input_dims_, PrecisionType::FP32);
        auto output_tensor = std::make_shared<NEONTensor<T>>(output_dims_, precision);

        AsymmQuantization _asymm_quantize(precision);
        EXPECT_EQ(_asymm_quantize.initialize(input_tensor, input_dims_.c, input_dims_.w, input_dims_.h, scale_,
                                                    zero_point_), Status::SUCCESS);
        EXPECT_EQ(_asymm_quantize.execute(input_tensor, output_tensor), status);
        EXPECT_EQ(_asymm_quantize.release(), Status::SUCCESS);

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

TEST(ENN_CPU_OP_UT_AsymmQuantization, INT16_INPUT) {
    float input_data[] = {0.404169, 1.46263,  3.20529,  3.37802, 1.43899,  0.584112, 0.0264098, 1.37933,  0.364663, 3.98312,
                          1.92523,  0.640683, 1.27818,  3.81138, 0.198998, 4.90459,  3.04084,   0.59914,  4.3223,   1.76831,
                          3.32803,  2.30331,  0.880843, 3.36722, 4.74236,  3.48453,  3.23806,   1.03613,  3.21663,  0.39879,
                          4.039,    3.06301,  1.71988,  3.51697, 3.21619,  2.07977,  2.22677,   0.805753, 3.93991,  0.273998,
                          2.62675,  4.2729,   3.9976,   3.80347, 3.56157,  4.51265,  1.87787,   4.72435};
    int16_t reference_output_data[] = {0, 1, 3, 3, 1, 1, 0, 1, 0, 4, 2, 1, 1, 4, 0, 5, 3, 1, 4, 2, 3, 2, 1, 3,
                                       5, 3, 3, 1, 3, 0, 4, 3, 2, 4, 3, 2, 2, 1, 4, 0, 3, 4, 4, 4, 4, 5, 2, 5};

    Dim4 input_dims_ = {1, 3, 4, 4};
    AsymmQuantizationTester()
        .InputDims(input_dims_)
        .Scale(1.0)
        .ZeroPoint(0)
        .TestRun(input_data, reference_output_data);
}

TEST(ENN_CPU_OP_UT_AsymmQuantization, INT8_INPUT) {
    float input_data[] = {4.8651,   2.365,    0.593675, 0.469204, 1.96065,  2.05648, 0.0245581, 3.49778,
                          4.73254,  0.322086, 1.45791,  1.39092,  3.96151,  3.88495, 0.960615,  3.20117,
                          2.72509,  3.71358,  2.19144,  0.216669, 4.47418,  4.28047, 3.59324,   2.68954,
                          3.63474,  2.02869,  2.62357,  0.758686, 0.163207, 4.57463, 3.58757,   0.00715872,
                          0.993861, 0.148322, 3.96643,  2.1726,   2.73622,  1.85289, 2.74879,   0.207821,
                          4.21882,  4.9934,   0.89359,  2.13093,  1.88166,  2.07535, 4.19816,   1.46345};
    int8_t reference_output_data[] = {4, 3, 1, 1, 2, 2, 1, 3, 4, 1, 2, 2, 4, 4, 2, 3, 3, 3, 2, 1, 4, 4, 3, 3,
                                      3, 2, 3, 2, 1, 4, 3, 1, 2, 1, 4, 2, 3, 2, 3, 1, 4, 4, 2, 2, 2, 2, 4, 2};

    Dim4 input_dims_ = {1, 3, 4, 4};
    AsymmQuantizationTester()
        .InputDims(input_dims_)
        .Scale(1.5)
        .ZeroPoint(1)
        .TestRun(input_data, reference_output_data);
}

TEST(ENN_CPU_OP_UT_AsymmQuantization, UINT8_INPUT) {
    float input_data[] = {3.18677, 4.01314, 2.68667, 1.75165,  1.52554, 1.15767, 4.17756, 1.73734, 2.89103, 3.8086,
                          1.89301, 4.67053, 1.55751, 0.614763, 3.53305, 1.91049, 4.12768, 3.02662, 3.48029, 1.67558,
                          3.40606, 3.46261, 4.88316, 3.68227,  3.93727, 4.06219, 2.04688, 2.30952, 4.25192, 3.81366,
                          1.53624, 4.09275, 4.22483, 3.44326,  2.05071, 4.86572, 4.49895, 1.68737, 2.44259, 4.04028,
                          2.95715, 3.13349, 2.94501, 3.48166,  4.89992, 4.40967, 1.92522, 3.07479};
    uint8_t reference_output_data[] = {4, 4, 3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 3, 2, 4, 3, 4, 4, 4, 3, 4, 4, 4, 4,
                                       4, 4, 3, 3, 4, 4, 3, 4, 4, 4, 3, 4, 4, 3, 3, 4, 3, 4, 3, 4, 4, 4, 3, 4};

    Dim4 input_dims_ = {1, 3, 4, 4};
    AsymmQuantizationTester()
        .InputDims(input_dims_)
        .Scale(2.0)
        .ZeroPoint(2)
        .TestRun(input_data, reference_output_data);
}

TEST(ENN_CPU_OP_UT_AsymmQuantization, BASIC_INPUT) {
    float input_data[] = {-1.1, 0.6, 0.7, 1.2, -0.7, 0.1};
    uint8_t reference_output_data[] = {127, 128, 128, 129, 128, 128};

    Dim4 input_dims_ = {1, 1, 1, 6};
    AsymmQuantizationTester()
        .InputDims(input_dims_)
        .Scale(2.0)
        .ZeroPoint(128)
        .TestRun(input_data, reference_output_data);
}

TEST(ENN_CPU_OP_UT_AsymmQuantization, INVALID_DIMENSION) {
    float input_data[] = {-1.1, 0.6, 0.7, 1.2, -0.7, 0.1};
    uint8_t reference_output_data[] = {127, 128, 128, 129, 128, 128};

    Dim4 input_dims_ = {1, 1, 0, 6};
    AsymmQuantizationTester()
        .InputDims(input_dims_)
        .Scale(2.0)
        .ZeroPoint(128)
        .TestRun(input_data, reference_output_data, Status::INVALID_PARAMS);
}

TEST(ENN_CPU_OP_UT_AsymmQuantization, UNSUPPORTED_DATA_TYPE) {
    float input_data[] = {-1.1, 0.6, 0.7, 1.2, -0.7, 0.1};
    uint16_t reference_output_data[] = {127, 128, 128, 129, 128, 128};

    Dim4 input_dims_ = {1, 1, 1, 6};
    AsymmQuantizationTester()
        .InputDims(input_dims_)
        .Scale(2.0)
        .ZeroPoint(128)
        .TestRun(input_data, reference_output_data, Status::FAILURE);
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
