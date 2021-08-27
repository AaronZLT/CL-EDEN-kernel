#include <gtest/gtest.h>
#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLRelu6.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class CLRelu6Tester : public ::testing::Test {
  public:
    CLRelu6Tester() {
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

    CLRelu6Tester &TestPrepare(const Dim4 &input_dim) {
        input_dim_ = input_dim;
        output_dim_ = input_dim;
        input_size_ = GetDimSize(input_dim_);
        output_size_ = GetDimSize(output_dim_);
        return *this;
    }

    void TestRun(float *input_data = nullptr, float *golden_data = nullptr) {
        std::shared_ptr<float> input;
        std::shared_ptr<float> output;
        if (input_data == nullptr || golden_data == nullptr) {
            input = make_shared_array<float>(input_size_);
            output = make_shared_array<float>(output_size_);
            GenerateRandom<float>(input.get(), input_size_, -1, 1);
            memset(output.get(), 0, output_size_ * sizeof(float));

            Relu6Guard relu6_guard;
            relu6_guard.GuardPrepare(input_size_);
            relu6_guard.GuardRun(input.get(), output.get());
            input_data = input.get();
            golden_data = output.get();
        }

        doRun(input_data, golden_data);
    }

  private:
    void doRun(float *input_data, float *golden_data) {
        auto input_tensor = std::make_shared<CLTensor>(runtime_, precision_, input_data, input_dim_);
        auto output_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, output_dim_);
        CLRelu6 relu1(runtime_, precision_);
        EXPECT_EQ(relu1.initialize({input_tensor}, {output_tensor}, nullptr), Status::SUCCESS);
        EXPECT_EQ(relu1.execute(), Status::SUCCESS);
        EXPECT_EQ(relu1.release(), Status::SUCCESS);

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

TYPED_TEST_CASE(CLRelu6Tester, TestFP32AndFP16Type);
TYPED_TEST(CLRelu6Tester, Simple4D) {
    float input_data[] = {-3.5, 7.2, 4.3, -0.1, -2.4, 0.9, 3.5, 10.0, 5.9};
    float golden_data[] = {0, 6, 4.3, 0, 0, 0.9, 3.5, 6, 5.9};
    this->TestPrepare({3, 3, 1, 1}).TestRun(input_data, golden_data);
}

TYPED_TEST(CLRelu6Tester, AllZero) {
    float input_data[] = {0, 0, 0, 0, 0, 0};
    float golden_data[] = {0, 0, 0, 0, 0, 0};
    this->TestPrepare({1, 2, 3, 1}).TestRun(input_data, golden_data);
}

TYPED_TEST(CLRelu6Tester, AllSix) {
    float input_data[] = {6, 6, 6, 6, 6, 6};
    float golden_data[] = {6, 6, 6, 6, 6, 6};
    this->TestPrepare({1, 1, 3, 2}).TestRun(input_data, golden_data);
}

TYPED_TEST(CLRelu6Tester, AllSmallerThanZero) {
    float input_data[] = {-3.5, -7.2, -4.3, -0.1, -2.4, -0.9};
    float golden_data[] = {0, 0, 0, 0, 0, 0};
    this->TestPrepare({6, 1, 1, 1}).TestRun(input_data, golden_data);
}

TYPED_TEST(CLRelu6Tester, AllLargerThanSix) {
    float input_data[] = {8.5, 7.2, 9.3, 7.1, 9.4, 8.9};
    float golden_data[] = {6, 6, 6, 6, 6, 6};
    this->TestPrepare({2, 1, 3, 1}).TestRun(input_data, golden_data);
}

TYPED_TEST(CLRelu6Tester, Random4D) { this->TestPrepare({2, 3, 13, 13}).TestRun(); }

}  // namespace gpu
}  // namespace ud
}  // namespace enn
