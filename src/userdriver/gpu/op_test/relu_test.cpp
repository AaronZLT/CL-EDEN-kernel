#include <gtest/gtest.h>
#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLRelu.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class CLReluTester : public ::testing::Test {
  public:
    CLReluTester() {
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

    CLReluTester &TestPrepare(const Dim4 &input_dim) {
        input_dim_ = input_dim;
        output_dim_ = input_dim;
        input_size_ = GetDimSize(input_dim_);
        output_size_ = GetDimSize(output_dim_);
        parameters_ = std::make_shared<ReluParameters>();
        parameters_->negative_slope = 0.0f;
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

            ReluGuard relu_guard;
            relu_guard.GuardPrepare(input_size_);
            relu_guard.GuardRun(input.get(), output.get());
            input_data = input.get();
            golden_data = output.get();
        }

        doRun(input_data, golden_data);
    }

  private:
    void doRun(float *input_data, float *golden_data) {
        auto input_tensor = std::make_shared<CLTensor>(runtime_, precision_, input_data, input_dim_);
        auto output_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, output_dim_);
        CLRelu relu(runtime_, precision_);
        EXPECT_EQ(relu.initialize({input_tensor}, {output_tensor}, parameters_), Status::SUCCESS);
        EXPECT_EQ(relu.execute(), Status::SUCCESS);
        EXPECT_EQ(relu.release(), Status::SUCCESS);

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
    std::shared_ptr<ReluParameters> parameters_;
};

TYPED_TEST_CASE(CLReluTester, TestFP32AndFP16Type);
TYPED_TEST(CLReluTester, Simple4D) {
    float input_data[] = {-0.5, 0.2, 0.3, -0.1, -0.4, 0.9};
    float golden_data[] = {0, 0.2, 0.3, 0, 0, 0.9};
    this->TestPrepare({2, 1, 3, 1}).TestRun(input_data, golden_data);
}

TYPED_TEST(CLReluTester, AllZero) {
    float input_data[] = {0, 0, 0, 0, 0, 0};
    float golden_data[] = {0, 0, 0, 0, 0, 0};
    this->TestPrepare({2, 3, 1, 1}).TestRun(input_data, golden_data);
}

TYPED_TEST(CLReluTester, AllNegative) {
    float input_data[] = {-0.5, -0.2, -0.3, -0.1, -0.4, -0.9};
    float golden_data[] = {0, 0, 0, 0, 0, 0};
    this->TestPrepare({2, 3, 1, 1}).TestRun(input_data, golden_data);
}

TYPED_TEST(CLReluTester, AllPositive) {
    float input_data[] = {0.5, 0.2, 0.3, 0.1, 0.4, 0.9};
    float golden_data[] = {0.5, 0.2, 0.3, 0.1, 0.4, 0.9};
    this->TestPrepare({2, 1, 3, 1}).TestRun(input_data, golden_data);
}

TYPED_TEST(CLReluTester, element_num_4) {
    float input_data[] = {1.6, 0.1, 0, -3.4, 5.7, -10.2, 0.5, 3.6};
    float golden_data[] = {1.6, 0.1, 0, 0, 5.7, 0, 0.5, 3.6};
    this->TestPrepare({2, 2, 2, 1}).TestRun(input_data, golden_data);
}

TYPED_TEST(CLReluTester, Random4D_non4) { this->TestPrepare({2, 3, 13, 13}).TestRun(); }

TYPED_TEST(CLReluTester, Random4D_4) { this->TestPrepare({2, 3, 4, 5}).TestRun(); }

}  // namespace gpu
}  // namespace ud
}  // namespace enn
