#include <gtest/gtest.h>
#include "userdriver/cpu/operators/ArgMax.hpp"
#include "userdriver/common/op_test/test_utils.h"

namespace enn {
namespace ud {
namespace cpu {

class ArgMaxTester {
 public:
    explicit ArgMaxTester(float threshold) : error_threshold_(threshold) {}

    inline ArgMaxTester& InputDims(const Dim4& input_dims) {
        input_dims_ = input_dims;
        return *this;
    }

    ArgMaxTester& OutputDims(const Dim4& output_dims) {
        output_dims_ = output_dims;
        return *this;
    }

    ArgMaxTester& Axis(const uint32_t& axis) {
        axis_ = axis;
        return *this;
    }

    void TestRun(float* input_data, const float* reference_output_data) {
        size_t input_size = GetDimSize(input_dims_);
        size_t output_size = GetDimSize(output_dims_);
        std::shared_ptr<NEONTensor<float>> input_tensor = std::make_shared<NEONTensor<float>>(input_dims_, PrecisionType::FP32);
        memcpy(input_tensor->getDataPtr().get(), input_data, input_size * sizeof(float));
        std::shared_ptr<NEONTensor<int>> output_tensor = std::make_shared<NEONTensor<int>>(output_dims_, PrecisionType::INT32);
        ArgMax _argmax(PrecisionType::FP32);
        EXPECT_EQ(_argmax.initialize(input_tensor, axis_, output_tensor), Status::SUCCESS);
        EXPECT_EQ(_argmax.execute(input_tensor, output_tensor), Status::SUCCESS);
        EXPECT_EQ(_argmax.release(), Status::SUCCESS);
        Compare(output_tensor->getDataPtr().get(), reference_output_data, output_size, error_threshold_);
    }

 private:
    float error_threshold_;
    Dim4 input_dims_;
    Dim4 output_dims_;
    uint32_t axis_;
};

TEST(ENN_CPU_OP_UT_ArgMax, SimpleAxis0) {
    float input_data[] = {0, 0.1, -0.2, 0.3, -0.4, 0.5,  //batch1 channel1
                          0.6, -0.7, 0.8, 0.9, 1, -1.1,  // channel2
                          1.2, 1.3, -1.4, 1.5, 1.6, 1.7,  //batch2
                          -1.8, 1.9, 2, 2.1, -2.2, 2.3
                         };
    float reference_output_data[] = {1, 1, 0, 1, 1, 1,
                                   0, 1, 1, 1, 0, 1
                                  };
    ArgMaxTester(1e-5)
    .InputDims({2, 2, 3, 2})
    .Axis(0)
    .OutputDims({1, 2, 3, 2})
    .TestRun(input_data, reference_output_data);
}

TEST(ENN_CPU_OP_UT_ArgMax, SimpleAxis1) {
    float input_data[] = {0, 0.1, -0.2, 0.3, -0.4, 0.5,  //batch1 channel1
                          0.6, -0.7, 0.8, 0.9, 1, -1.1,  // channel2
                          1.2, 1.3, -1.4, 1.5, 1.6, 1.7,  //batch2
                          -1.8, 1.9, 2, 2.1, -2.2, 2.3
                         };
    float reference_output_data[] = {1, 0, 1, 1, 1, 0,
                                   0, 1, 1, 1, 0, 1
                                  };
    ArgMaxTester(1e-5)
    .InputDims({2, 2, 3, 2})
    .Axis(1)
    .OutputDims({2, 1, 3, 2})
    .TestRun(input_data, reference_output_data);
}

TEST(ENN_CPU_OP_UT_ArgMax, SimpleAxis2) {
    float input_data[] = {0, 0.1, -0.2, 0.3, -0.4, 0.5,  //batch1 channel1
                          0.6, -0.7, 0.8, 0.9, 1, -1.1,  // channel2
                          1.2, 1.3, -1.4, 1.5, 1.6, 1.7,  //batch2
                          -1.8, 1.9, 2, 2.1, -2.2, 2.3
                         };
    float reference_output_data[] = {0, 2, 2, 1,
                                   2, 2, 1, 2
                                  };
    ArgMaxTester(1e-5)
    .InputDims({2, 2, 3, 2})
    .Axis(2)
    .OutputDims({2, 2, 1, 2})
    .TestRun(input_data, reference_output_data);
}

TEST(ENN_CPU_OP_UT_ArgMax, SimpleAxis3) {
    float input_data[] = {0, 0.1, -0.2, 0.3, -0.4, 0.5,  //batch1 channel1
                          0.6, -0.7, 0.8, 0.9, 1, -1.1,  // channel2
                          1.2, 1.3, -1.4, 1.5, 1.6, 1.7,  //batch2
                          -1.8, 1.9, 2, 2.1, -2.2, 2.3
                         };
    float reference_output_data[] = {1, 1, 1, 0, 1, 0,
                                   1, 1, 1, 1, 1, 1
                                  };
    ArgMaxTester(1e-5)
    .InputDims({2, 2, 3, 2})
    .Axis(3)
    .OutputDims({2, 2, 3, 1})
    .TestRun(input_data, reference_output_data);
}

}   // namespace cpu
}   // namespace ud
}  // namespace enn
