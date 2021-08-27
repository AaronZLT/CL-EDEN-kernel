#include <gtest/gtest.h>
#include "userdriver/cpu/operators/ArgMin.hpp"
#include "userdriver/common/op_test/test_utils.h"

namespace enn {
namespace ud {
namespace cpu {

class ArgMinTester {
 public:
    explicit ArgMinTester(float threshold) : error_threshold_(threshold) {}

    inline ArgMinTester& InputDims(const Dim4& input_dims) {
        input_dims_ = input_dims;
        return *this;
    }

    ArgMinTester& OutputDims(const Dim4& output_dims) {
        output_dims_ = output_dims;
        return *this;
    }

    ArgMinTester& Axis(const uint32_t& axis) {
        axis_ = axis;
        return *this;
    }

    void TestRun(float* input_data, const float* reference_output_data) {
        size_t input_size = GetDimSize(input_dims_);
        int dim = getDim(input_dims_, axis_);
        size_t output_size = input_size / dim;
        std::shared_ptr<NEONTensor<float>> input_tensor = std::make_shared<NEONTensor<float>>(input_dims_, PrecisionType::FP32);
        memcpy(input_tensor->getDataPtr().get(), input_data, input_size * sizeof(float));
        std::shared_ptr<NEONTensor<int>> output_tensor = std::make_shared<NEONTensor<int>>(output_dims_, PrecisionType::INT32);
        ArgMin _argmin(PrecisionType::FP32);
        EXPECT_EQ(_argmin.initialize(input_tensor, axis_, output_tensor), Status::SUCCESS);
        EXPECT_EQ(_argmin.execute(input_tensor, output_tensor), Status::SUCCESS);
        EXPECT_EQ(_argmin.release(), Status::SUCCESS);
        Compare(output_tensor->getDataPtr().get(), reference_output_data, output_size, error_threshold_);
    }

 private:
    float error_threshold_;
    Dim4 input_dims_;
    Dim4 output_dims_;
    uint32_t axis_;
};

TEST(ENN_CPU_OP_UT_ArgMin, VTS1) {
    float input_data[] = {1.0, 2.0, 4.0, 3.0};
    float reference_output_data[] = {0, 1};
    ArgMinTester(1e-5)
    .InputDims({1, 1, 2, 2})
    .Axis(3)
    .OutputDims({1, 1, 2, 1})
    .TestRun(input_data, reference_output_data);
}

TEST(ENN_CPU_OP_UT_ArgMin, VTS2) {
    float input_data[] = {1.0, 2.0, 4.0, 3.0};
    float reference_output_data[] = {0, 0};
    ArgMinTester(1e-5)
    .InputDims({1, 1, 2, 2})
    .Axis(2)
    .OutputDims({1, 1, 1, 2})
    .TestRun(input_data, reference_output_data);
}

TEST(ENN_CPU_OP_UT_ArgMin, DynamicOutputShape) {
    float input_data[] = {1.0, 2.0, 4.0, 3.0};
    float reference_output_data[] = {0, 1};
    ArgMinTester(1e-5)
    .InputDims({1, 1, 2, 2})
    .Axis(3)
    .OutputDims({0, 0, 0, 0})
    .TestRun(input_data, reference_output_data);
}

}   // namespace cpu
}   // namespace ud
}  // namespace enn
