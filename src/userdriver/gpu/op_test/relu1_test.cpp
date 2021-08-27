#include <gtest/gtest.h>
#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLRelu1.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class CLRelu1Tester : public ::testing::Test {
  public:
    CLRelu1Tester() {
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

    CLRelu1Tester &TestPrepare(const Dim4 &input_dim) {
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

            Relu1Guard relu1_guard;
            relu1_guard.GuardPrepare(input_size_);
            relu1_guard.GuardRun(input.get(), output.get());
            input_data = input.get();
            golden_data = output.get();
        }

        doRun(input_data, golden_data);
    }

  private:
    void doRun(float *input_data, float *golden_data) {
        auto input_tensor = std::make_shared<CLTensor>(runtime_, precision_, input_data, input_dim_);
        auto output_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, output_dim_);
        CLRelu1 relu1(runtime_, precision_);
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

TYPED_TEST_CASE(CLRelu1Tester, TestFP32AndFP16Type);
TYPED_TEST(CLRelu1Tester, Simple4D) {
    float input_data[] = {-3.5, 0.2, 1.3, -0.1, -2.4, 0.9, 1.2, 3.5, 0.2};
    float golden_data[] = {-1, 0.2, 1, -0.1, -1, 0.9, 1, 1, 0.2};
    this->TestPrepare({1, 3, 1, 3}).TestRun(input_data, golden_data);
}

TYPED_TEST(CLRelu1Tester, AllOne) {
    float input_data[] = {1, 1, 1, 1, 1, 1};
    float golden_data[] = {1, 1, 1, 1, 1, 1};
    this->TestPrepare({1, 1, 1, 6}).TestRun(input_data, golden_data);
}

TYPED_TEST(CLRelu1Tester, AllMinusOne) {
    float input_data[] = {-1, -1, -1, -1, -1, -1};
    float golden_data[] = {-1, -1, -1, -1, -1, -1};
    this->TestPrepare({2, 1, 3, 1}).TestRun(input_data, golden_data);
}

TYPED_TEST(CLRelu1Tester, AllSmallerThanMinusOne) {
    float input_data[] = {-3.5, -1.2, -2.3, -3.1, -2.4, -5.9};
    float golden_data[] = {-1, -1, -1, -1, -1, -1};
    this->TestPrepare({1, 1, 3, 2}).TestRun(input_data, golden_data);
}

TYPED_TEST(CLRelu1Tester, AllLargerThanOne) {
    float input_data[] = {3.5, 1.2, 2.3, 3.1, 2.4, 5.9};
    float golden_data[] = {1, 1, 1, 1, 1, 1};
    this->TestPrepare({2, 1, 3, 1}).TestRun(input_data, golden_data);
}

TYPED_TEST(CLRelu1Tester, BetweenOneAndminusOne) {
    float input_data[] = {-0.5, 0.2, 0.3, -0.1, -0.4, 0.9};
    float golden_data[] = {-0.5, 0.2, 0.3, -0.1, -0.4, 0.9};
    this->TestPrepare({1, 2, 3, 1}).TestRun(input_data, golden_data);
}

TYPED_TEST(CLRelu1Tester, Random4D) { this->TestPrepare({2, 3, 32, 12}).TestRun(); }

}  // namespace gpu
}  // namespace ud
}  // namespace enn
