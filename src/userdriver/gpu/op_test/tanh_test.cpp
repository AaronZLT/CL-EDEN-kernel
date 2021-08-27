#include <gtest/gtest.h>
#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLTanh.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class CLTanhTester : public ::testing::Test {
  public:
    CLTanhTester() {
        precision_ = PRECISION::precision;
        switch (precision_) {
        case PrecisionType::FP32:
            data_type_ = DataType::FLOAT;
            error_threshold_ = 1e-5;
            break;
        case PrecisionType::FP16:
            data_type_ = DataType::FLOAT;
            error_threshold_ = 1e-2;
            break;
        default: break;
        }
        runtime_ = std::shared_ptr<CLRuntime>(new CLRuntime());
        runtime_->initialize(0);
        runtime_->initializeQueue();
    }

    CLTanhTester &TestPrepare(const Dim4 &input_dim) {
        input_dim_ = input_dim;
        output_dim_ = input_dim;
        input_size_ = GetDimSize(input_dim_);
        output_size_ = GetDimSize(output_dim_);
        return *this;
    }

    void TestRun(float *input_data = nullptr, float *golden_data = nullptr) {
        auto input_tensor = std::make_shared<CLTensor>(runtime_, precision_, input_data, input_dim_);
        auto output_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, output_dim_);
        CLTanh tanh_op(runtime_, precision_);
        EXPECT_EQ(tanh_op.initialize({input_tensor}, {output_tensor}, nullptr), Status::SUCCESS);
        EXPECT_EQ(tanh_op.execute(), Status::SUCCESS);
        EXPECT_EQ(tanh_op.release(), Status::SUCCESS);

        auto output_ptr = make_shared_array<float>(output_size_);
        output_tensor->readData(output_ptr.get());

        Compare(output_ptr.get(), golden_data, output_size_, error_threshold_);
    }

  private:
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;
    DataType data_type_;
    float error_threshold_;

    Dim4 input_dim_;
    Dim4 output_dim_;
    size_t input_size_;
    size_t output_size_;
};

TYPED_TEST_CASE(CLTanhTester, TestFP32AndFP16Type);
TYPED_TEST(CLTanhTester, allpositive) {
    float input_data[] = {1.2, 0.1, 3.2, 8.1, 0.4, 5.7};
    float golden_data[] = {0.83365, 0.09967, 0.99668, 1.00000, 0.37995, 0.99998};
    this->TestPrepare({1, 2, 3, 1}).TestRun(input_data, golden_data);
}

TYPED_TEST(CLTanhTester, allnegative) {
    float input_data[] = {-1.2, -0.1, -3.2, -8.1, -0.4, -5.7};
    float golden_data[] = {-0.83365, -0.09967, -0.99668, -1.00000, -0.37995, -0.99998};
    this->TestPrepare({1, 2, 3, 1}).TestRun(input_data, golden_data);
}

TYPED_TEST(CLTanhTester, allzero) {
    float input_data[] = {0, 0, 0, 0, 0, 0, 0, 0};
    float golden_data[] = {0, 0, 0, 0, 0, 0, 0, 0};
    this->TestPrepare({2, 2, 2, 1}).TestRun(input_data, golden_data);
}

TYPED_TEST(CLTanhTester, simple_non4) {
    float input_data[] = {3.5, -1.2, 0.1, 9.1, 0, -9.1, -3.5, 0, 0.2, 4.1, -3.6, 2.5, 7.1, -3.7, 2.6, -2.6, 5.2, 6.1};
    float golden_data[] = {0.99818,
                           -0.83365,
                           0.09967,
                           1.00000,
                           0.00000,
                           -1.00000,
                           -0.99818,
                           0.00000,
                           0.19738,
                           0.99945,
                           -0.99851,
                           0.98661,
                           1.00000,
                           -0.99878,
                           0.98903,
                           -0.98903,
                           0.99994,
                           0.99999};
    this->TestPrepare({2, 3, 3, 1}).TestRun(input_data, golden_data);
}

TYPED_TEST(CLTanhTester, simple_4) {
    float input_data[] = {4.2, -0.1, 3.5, -7.1, 0, 0.4, 5.1, 9.3};
    float golden_data[] = {0.99955, -0.09967, 0.99818, -1.00000, 0.00000, 0.37995, 0.99993, 1.00000};
    this->TestPrepare({1, 2, 2, 2}).TestRun(input_data, golden_data);
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
