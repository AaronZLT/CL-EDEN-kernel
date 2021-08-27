#include <gtest/gtest.h>
#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLSigmoid.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class CLSigmoidTester : public ::testing::Test {
  public:
    CLSigmoidTester() {
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

    CLSigmoidTester &TestPrepare(const Dim4 &input_dim) {
        input_dim_ = input_dim;
        output_dim_ = input_dim;
        input_size_ = GetDimSize(input_dim_);
        output_size_ = GetDimSize(output_dim_);
        return *this;
    }

    void TestRun(float *input_data = nullptr) {
        std::shared_ptr<float> output = make_shared_array<float>(output_size_);
        memset(output.get(), 0, output_size_ * sizeof(float));

        std::shared_ptr<float> input;
        if (input_data == nullptr) {
            input = make_shared_array<float>(input_size_);
            GenerateRandom<float>(input.get(), input_size_, -1, 1);
            input_data = input.get();
        }

        SigmoidGuard sigmoid_guard;
        sigmoid_guard.GuardPrepare(input_size_);
        sigmoid_guard.GuardRun(input_data, output.get());

        doRun(input_data, output.get());
    }

  private:
    void doRun(float *input_data, float *golden_data) {
        auto input_tensor = std::make_shared<CLTensor>(runtime_, precision_, input_data, input_dim_);
        auto output_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, output_dim_);
        CLSigmoid sigmoid_op(runtime_, precision_);
        EXPECT_EQ(sigmoid_op.initialize({input_tensor}, {output_tensor}, nullptr), Status::SUCCESS);
        EXPECT_EQ(sigmoid_op.execute(), Status::SUCCESS);
        EXPECT_EQ(sigmoid_op.release(), Status::SUCCESS);

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

TYPED_TEST_CASE(CLSigmoidTester, TestFP32AndFP16Type);
TYPED_TEST(CLSigmoidTester, Simple4D) {
    float input_data[] = {-3.5, 7.2, 4.3, -0.1, -2.4, 0.9, -3.5, 7.2, 0.3, -0.1, 2.4, 0.9, -3.5, 2.2, 4.3, 0.1, -2.4, 0.9};
    this->TestPrepare({3, 1, 2, 3}).TestRun(input_data);
}

TYPED_TEST(CLSigmoidTester, CloseToNegInfinite) {
    float input_data[] = {-99999, -99999, -99999, -99999, -99999, -99999};
    this->TestPrepare({6, 1, 1, 1}).TestRun(input_data);
}

TYPED_TEST(CLSigmoidTester, CloseToPosInfinite) {
    float input_data[] = {99999, 99999, 99999, 99999, 99999, 99999};
    this->TestPrepare({1, 1, 3, 2}).TestRun(input_data);
}

TYPED_TEST(CLSigmoidTester, AllZero) {
    float input_data[] = {0, 0, 0, 0, 0, 0};
    this->TestPrepare({2, 1, 3, 1}).TestRun(input_data);
}

TYPED_TEST(CLSigmoidTester, Random4D) { this->TestPrepare({1, 3, 13, 13}).TestRun(); }

TYPED_TEST(CLSigmoidTester, Random4D1) { this->TestPrepare({1, 5, 3, 23}).TestRun(); }

TYPED_TEST(CLSigmoidTester, Random4D2) { this->TestPrepare({2, 3, 11, 17}).TestRun(); }

TYPED_TEST(CLSigmoidTester, Random4D3) { this->TestPrepare({2, 7, 3, 13}).TestRun(); }

}  // namespace gpu
}  // namespace ud
}  // namespace enn
