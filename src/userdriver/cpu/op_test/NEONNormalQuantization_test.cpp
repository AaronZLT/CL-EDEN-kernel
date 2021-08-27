#include <gtest/gtest.h>
#include "userdriver/cpu/operators/NEONNormalQuantization.hpp"
#include "userdriver/common/op_test/test_utils.h"

namespace enn {
namespace ud {
namespace cpu {

class NormalQuantizationTester {
public:
    explicit NormalQuantizationTester(float threshold = ERROR_THRESHOLD) : error_threshold_(threshold) {}

    NormalQuantizationTester& SetDims(const Dim4& input_dims) {
        input_dims_ = input_dims;
        output_dims_ = input_dims;
        return *this;
    }

    template <typename T1, typename T2>
    void TestRun(T1* input_data, const T2* reference_output_data, double* means, double* scales, int32_t* frac_lens,
                 Status status = Status::SUCCESS) {
        PrecisionType precision_input = getPrecisionType(input_data);
        PrecisionType precision_output = getPrecisionType(reference_output_data);
        size_t output_size = GetDimSize(output_dims_);

        auto input_tensor = std::make_shared<NEONTensor<T1>>(input_data, input_dims_, precision_input);
        auto output_tensor = std::make_shared<NEONTensor<T2>>(output_dims_, precision_output);
        auto means_ = std::make_shared<NEONTensor<double>>(means, (Dim4){1, input_dims_.c, 1, 1}, PrecisionType::FP32);
        auto scales_ = std::make_shared<NEONTensor<double>>(scales, (Dim4){1, input_dims_.c, 1, 1}, PrecisionType::FP32);
        auto frac_lens_ =
            std::make_shared<NEONTensor<int32_t>>(frac_lens, (Dim4){1, input_dims_.c, 1, 1}, PrecisionType::INT32);

        NEONNormalQuantization _neon_normquant(precision_input);
        EXPECT_EQ(_neon_normquant.initialize(input_tensor, input_dims_.w, input_dims_.h, input_dims_.c, means_, scales_,
                                               frac_lens_), Status::SUCCESS);
        EXPECT_EQ(_neon_normquant.execute(input_tensor, output_tensor), status);
        EXPECT_EQ(_neon_normquant.release(), Status::SUCCESS);

        if (status == Status::SUCCESS)
            Compare(output_tensor->getDataPtr().get(), reference_output_data, output_size, error_threshold_);
    }

private:
    float error_threshold_;
    Dim4 input_dims_;
    Dim4 output_dims_;
};

TEST(ENN_CPU_OP_UT_NormalQuantization, FLOAT_INPUT_INT8_OUTPUT) {
    Dim4 input_dims_ = {1, 2, 4, 4};
    double means[] = {1.0, 0.5};
    double scales[] = {0.5, 0.5};
    int32_t frac_lens[] = {2, -1};

    float input_data[] = {2.0, 3.2, 4.3, 66.0, 5.3,  -6.7, 1.0, 15.0, 0.0, 1.7, -1.0, 3.8, 11.0, 1.5, 0.5, -0.5,
                          3.0, 2.9, 1.9, 12.2, 10.1, -7.2, 3.5, 2.3,  2.2, 3.9, -1.0, 0,   3.0,  2.4, 6.2, -5.0};
    int8_t reference_output_data[] = {2, 4, 7, 127, 9, -15, 0, 28, -2, 1, -4, 6, 20, 1, -1, -3,
                                      1, 1, 0, 3,   2, -2,  1, 0,  0,  1, 0,  0, 1,  0, 1,  -1};

    NormalQuantizationTester()
        .SetDims(input_dims_)
        .TestRun(input_data, reference_output_data, means, scales, frac_lens);
}

TEST(ENN_CPU_OP_UT_NormalQuantization, FLOAT_INPUT_INT16_OUTPUT) {
    Dim4 input_dims_ = {1, 2, 4, 4};
    double means[] = {1.0, 0.5};
    double scales[] = {0.5, 1};
    int32_t frac_lens[] = {2, -1};

    float input_data[] = {2.0, 3.2, 4.3, 66.0, 5.3,  -6.7, 1.0, 15.0, 0.0, 1.7, -1.0, 3.8, 11.0, 1.5, 0.5, -0.5,
                          3.0, 2.9, 1.9, 12.2, 10.1, -7.2, 3.5, 2.3,  2.2, 3.9, -1.0, 0,   3.0,  2.4, 6.2, -5.0};
    int16_t reference_output_data[] = {2, 4, 7, 130, 9, -15, 0, 28, -2, 1, -4, 6, 20, 1, -1, -3,
                                       1, 1, 1, 6,   5, -4,  2, 1,  1,  2, -1, 0, 1,  1, 3,  -3};

    NormalQuantizationTester()
        .SetDims(input_dims_)
        .TestRun(input_data, reference_output_data, means, scales, frac_lens);
}

TEST(ENN_CPU_OP_UT_NormalQuantization, UINT8_INPUT_UINT8_OUTPUT) {
    Dim4 input_dims_ = {1, 2, 3, 4};
    double means[] = {0.2, 0.3};
    double scales[] = {0.5, 1.5};
    int32_t frac_lens[] = {-2, 3};

    uint8_t input_data[] = {127, 125, 132, 1,   0,  6,  78, 34, 23,  255, 245, 123,
                            67,  98,  30,  123, 12, 45, 34, 67, 189, 2,   15,  128};
    uint8_t reference_output_data[] = {16,  16,  16,  0,   0,   1,   10,  4,   3,   32, 31,  15,
                                       127, 127, 127, 127, 127, 127, 127, 127, 127, 20, 127, 127};

    NormalQuantizationTester()
        .SetDims(input_dims_)
        .TestRun(input_data, reference_output_data, means, scales, frac_lens);
}

TEST(ENN_CPU_OP_UT_NormalQuantization, UINT8_INPUT_INT16_OUTPUT) {
    Dim4 input_dims_ = {1, 2, 3, 2};
    double means[] = {0.2, 0.3};
    double scales[] = {0.5, 1.5};
    int32_t frac_lens[] = {-2, 2};

    uint8_t input_data[] = {127, 123, 87, 4, 130, 245, 89, 248, 1, 0, 45, 8};
    int16_t reference_output_data[] = {16, 15, 11, 0, 16, 31, 532, 1486, 4, -2, 268, 46};

    NormalQuantizationTester()
        .SetDims(input_dims_)
        .TestRun(input_data, reference_output_data, means, scales, frac_lens);
}

TEST(ENN_CPU_OP_UT_NormalQuantization, UNSUPPORTED_DATA_TYPE) {
    Dim4 input_dims_ = {1, 2, 3, 2};
    double means[] = {0.2, 0.3};
    double scales[] = {0.5, 1.5};
    int32_t frac_lens[] = {-2, 2};

    int8_t input_data[] = {127, 123, 87, 4, 6, 121, 89, 2, 1, 0, 45, 8};
    int16_t reference_output_data[] = {16, 15, 11, 0, 16, 31, 532, 1486, 4, -2, 268, 46};

    NormalQuantizationTester()
        .SetDims(input_dims_)
        .TestRun(input_data, reference_output_data, means, scales, frac_lens, Status::FAILURE);
}

TEST(ENN_CPU_OP_UT_NormalQuantization, INVALID_DIMENSION) {
    Dim4 input_dims_ = {1, 0, 1, 2};
    double* means = NULL;
    double* scales = NULL;
    int32_t* frac_lens = NULL;

    uint8_t* input_data = NULL;
    int16_t* reference_output_data = NULL;

    NormalQuantizationTester()
        .SetDims(input_dims_)
        .TestRun(input_data, reference_output_data, means, scales, frac_lens, Status::INVALID_PARAMS);
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
