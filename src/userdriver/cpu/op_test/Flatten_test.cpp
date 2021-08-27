#include <gtest/gtest.h>
#include "userdriver/cpu/operators/Flatten.hpp"
#include "userdriver/common/op_test/test_utils.h"

namespace enn {
namespace ud {
namespace cpu {

class FlattenTester {
public:
    explicit FlattenTester(float threshold) : error_threshold_(threshold) {}

    FlattenTester &InputDim(const Dim4 &input_dims) {
        input_dims_ = input_dims;
        return *this;
    }

    FlattenTester &OutputDim(const Dim4 &output_dims) {
        output_dims_ = output_dims;
        return *this;
    }

    FlattenTester &NumAxis(const int &num_axis) {
        num_axis_ = num_axis;
        return *this;
    }

    void TestRun(const int start_axis, const int end_axis) {
        int size = GetDimSize(input_dims_);
        float *input_data = new float[size];
        GenerateRandom<float>(input_data, size, -10, 10);
        std::shared_ptr<NEONTensor<float>> input_tensor =
            std::make_shared<NEONTensor<float>>(input_data, input_dims_, PrecisionType::FP32);
        std::shared_ptr<NEONTensor<float>> output_tensor =
            std::make_shared<NEONTensor<float>>(output_dims_, PrecisionType::FP32);
        std::vector<uint32_t> output_dim;
        Flatten flatten(PrecisionType::FP32);
        flatten.initialize(start_axis, end_axis);
        flatten.execute(input_tensor, output_tensor, &output_dim);
        Compare(output_tensor->getDataPtr().get(), input_data, size, error_threshold_);
        EXPECT_EQ(num_axis_, output_dim.size());
        for (size_t i = 0; i < output_dim.size(); ++i) {
            if (i == 0) {
                EXPECT_EQ(output_dims_.n, output_dim[i]);
            } else if (i == 1) {
                EXPECT_EQ(output_dims_.c, output_dim[i]);
            } else if (i == 2) {
                EXPECT_EQ(output_dims_.h, output_dim[i]);
            } else if (i == 3) {
                EXPECT_EQ(output_dims_.w, output_dim[i]);
            }
        }
        delete[](input_data);
        flatten.release();
    }

private:
    std::shared_ptr<float> data_;
    float error_threshold_;
    uint32_t num_axis_;
    Dim4 input_dims_;
    Dim4 output_dims_;
};

// the value range of start and end is from -4 to 4,and |start+input_dim.size()| <= |end+input_dim.size()
TEST(FlattenTester, start_2_end_3) {
    int start_axis = 2;
    int end_axis = 3;
    FlattenTester(0).InputDim({1, 3, 6, 5}).OutputDim({1, 3, 6 * 5, 1}).NumAxis(3).TestRun(start_axis, end_axis);
}

TEST(FlattenTester, start_1_end_f2) {
    int start_axis = 1;
    int end_axis = -2;
    FlattenTester(0).InputDim({2, 1, 6, 5}).OutputDim({2, 1 * 6, 5, 1}).NumAxis(3).TestRun(start_axis, end_axis);
}

TEST(FlattenTester, start_2_end_2) {
    int start_axis = 2;
    int end_axis = 2;
    FlattenTester(0).InputDim({2, 3, 1, 5}).OutputDim({2, 3, 1, 5}).NumAxis(4).TestRun(start_axis, end_axis);
}

TEST(FlattenTester, start_f3_end_f2) {
    int start_axis = -3;
    int end_axis = -2;
    FlattenTester(0).InputDim({2, 3, 6, 1}).OutputDim({2, 3 * 6, 1, 1}).NumAxis(3).TestRun(start_axis, end_axis);
}

TEST(FlattenTester, start_f3_end_3) {
    int start_axis = -3;
    int end_axis = 3;
    FlattenTester(0).InputDim({4, 5, 6, 7}).OutputDim({4, 5 * 6 * 7, 1, 1}).NumAxis(2).TestRun(start_axis, end_axis);
}

TEST(FlattenTester, start_0_end_0) {
    int start_axis = 0;
    int end_axis = 0;
    FlattenTester(0).InputDim({6, 3, 2, 5}).OutputDim({6, 3, 2, 5}).NumAxis(4).TestRun(start_axis, end_axis);
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
