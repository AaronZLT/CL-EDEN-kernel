#include <gtest/gtest.h>
#include "userdriver/cpu/operators/Softmax.hpp"
#include "userdriver/common/op_test/test_utils.h"

namespace enn {
namespace ud {
namespace cpu {

class SoftmaxTester {
public:
    explicit SoftmaxTester(float threshold = ERROR_THRESHOLD) : error_threshold_(threshold) {}
    SoftmaxTester& InputDims(const Dim4& input_dims) {
        input_dims_ = input_dims;
        output_dims_ = input_dims;
        return *this;
    }

    template <typename T>
    void TestRun(T* input_data, const T* reference_output_data, const float& beta, const int32_t& axis, Status status = Status::SUCCESS) {
        PrecisionType precision = getPrecisionType(input_data);
        size_t output_size = GetDimSize(output_dims_);

        auto input_tensor = std::make_shared<NEONTensor<T>>(input_data, input_dims_, precision);
        auto output_tensor = std::make_shared<NEONTensor<T>>(output_dims_, precision);

        Softmax _softmax(precision);
        EXPECT_EQ(_softmax.initialize(input_tensor, input_dims_.w, input_dims_.h, input_dims_.c, input_dims_.n, beta,
                                             axis),
                  Status::SUCCESS);
        EXPECT_EQ(_softmax.execute(input_tensor, output_tensor), status);
        EXPECT_EQ(_softmax.release(), Status::SUCCESS);

        if (status == Status::SUCCESS)
            Compare(output_tensor->getDataPtr().get(), reference_output_data, output_size, error_threshold_);
    }

private:
    float error_threshold_;
    Dim4 input_dims_;
    Dim4 output_dims_;
};

TEST(ENN_CPU_OP_UT_Softmax, POSITIVE_AXIS) {
    float input_data[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                          22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42};

    float reference_output_data_axis0[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    float reference_output_data_axis1[] = {0.014774, 0.014774, 0.014774, 0.014774, 0.014774, 0.014774, 0.014774,  // ch = 1
                                           0.014774, 0.014774, 0.014774, 0.014774, 0.014774, 0.014774, 0.014774,
                                           0.014774, 0.014774, 0.014774, 0.014774, 0.014774, 0.014774, 0.014774,

                                           0.985226, 0.985226, 0.985226, 0.985226, 0.985226, 0.985226, 0.985226,  // ch = 2
                                           0.985226, 0.985226, 0.985226, 0.985226, 0.985226, 0.985226, 0.985226,
                                           0.985226, 0.985226, 0.985226, 0.985226, 0.985226, 0.985226, 0.985226};

    float reference_output_data_axis2[] = {
        0.046512, 0.046512, 0.046512, 0.046512, 0.046512, 0.046512, 0.046512, 0.188615, 0.188615, 0.188615, 0.188615,
        0.188615, 0.188615, 0.188615, 0.764873, 0.764873, 0.764873, 0.764873, 0.764873, 0.764873, 0.764873, 0.046512,
        0.046512, 0.046512, 0.046512, 0.046512, 0.046512, 0.046512, 0.188615, 0.188615, 0.188615, 0.188615, 0.188615,
        0.188615, 0.188615, 0.764873, 0.764873, 0.764873, 0.764873, 0.764873, 0.764873, 0.764873};
    Dim4 input_dims_ = {1, 2, 3, 7};
    SoftmaxTester().InputDims(input_dims_).TestRun(input_data, reference_output_data_axis0, 0.2, 0);
    SoftmaxTester().InputDims(input_dims_).TestRun(input_data, reference_output_data_axis1, 0.2, 1);
    SoftmaxTester().InputDims(input_dims_).TestRun(input_data, reference_output_data_axis2, 0.2, 2);
}

TEST(ENN_CPU_OP_UT_Softmax, NEGATIVE_AXIS) {
    float input_data[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                          22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42};

    float reference_output_data[] = {
        0.0724675, 0.088512, 0.108109,  0.132044,  0.161279, 0.196987, 0.240601,  0.0724675, 0.088512, 0.108109, 0.132044,
        0.161279,  0.196987, 0.240601,  0.0724675, 0.088512, 0.108109, 0.132044,  0.161279,  0.196987, 0.240601, 0.0724675,
        0.088512,  0.108109, 0.132044,  0.161279,  0.196987, 0.240601, 0.0724675, 0.088512,  0.108109, 0.132044, 0.161279,
        0.196987,  0.240601, 0.0724675, 0.088512,  0.108109, 0.132044, 0.161279,  0.196987,  0.240601};

    Dim4 input_dims_ = {1, 2, 3, 7};
    SoftmaxTester().InputDims(input_dims_).TestRun(input_data, reference_output_data, 0.2, -1);
}
/*
TEST(ENN_CPU_OP_UT_Softmax, INVALID_INPUT) {

    float *input_data;
    float *reference_output_data;

    Dim4 input_dims_ = {0, 0, 0, 0};
    SoftmaxTester().InputDims(input_dims_).TestRun(input_data, reference_output_data,
        0.2, 0, 5, Status::INVALID_PARAMS);
}
*/
TEST(ENN_CPU_OP_UT_Softmax, INVALID_DATATYPE) {
    int16_t input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    int16_t reference_output_data[1] = {5};

    Dim4 input_dims_ = {1, 1, 2, 5};
    SoftmaxTester()
        .InputDims(input_dims_)
        .TestRun(input_data, reference_output_data, 0.2, 0, Status::FAILURE);
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
