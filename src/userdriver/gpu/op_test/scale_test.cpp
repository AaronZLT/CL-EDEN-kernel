#include <gtest/gtest.h>
#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLScale.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class CLScaleTester : public ::testing::Test {
  public:
    CLScaleTester() {
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

    CLScaleTester &TestPrepare(const Dim4 &input_dim) {
        input_dim_ = input_dim;
        scale_dim_ = {1, input_dim_.c, 1, 1};
        bias_dim_ = scale_dim_;
        output_dim_ = input_dim;
        input_size_ = GetDimSize(input_dim_);
        output_size_ = GetDimSize(output_dim_);
        return *this;
    }

    void TestRun(float* scale, float* bias = nullptr, float *input_data = nullptr) {
        std::shared_ptr<float> output = make_shared_array<float>(output_size_);
        memset(output.get(), 0, output_size_ * sizeof(float));

        std::shared_ptr<float> input;
        if (input_data == nullptr) {
            input = make_shared_array<float>(input_size_);
            GenerateRandom<float>(input.get(), input_size_, -1, 1);
            input_data = input.get();
        }

        ScaleGuard scale_guard;
        scale_guard.GuardPrepare(scale, bias, input_dim_.n, input_dim_.c, input_dim_.h, input_dim_.w);
        scale_guard.GuardRun(input_data, output.get());

        doRun(input_data, output.get(), scale, bias);
    }

  private:
    void doRun(float *input_data, float *golden_data, float* scale_data, float* bias_data) {
        auto input_tensor = std::make_shared<CLTensor>(runtime_, precision_, input_data, input_dim_);
        auto scale_tensor = std::make_shared<CLTensor>(runtime_, precision_, scale_data, scale_dim_);
        auto bias_tensor = std::make_shared<CLTensor>(runtime_, precision_, bias_data, bias_dim_);
        auto output_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, output_dim_);

        CLScale scale(runtime_, precision_);
        EXPECT_EQ(scale.initialize({input_tensor, scale_tensor, bias_tensor}, {output_tensor}, nullptr), Status::SUCCESS);
        EXPECT_EQ(scale.execute(), Status::SUCCESS);
        EXPECT_EQ(scale.release(), Status::SUCCESS);

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
    Dim4 scale_dim_;
    Dim4 bias_dim_;
    Dim4 output_dim_;
    size_t input_size_;
    size_t output_size_;
};

TYPED_TEST_CASE(CLScaleTester, TestFP32AndFP16Type);
TYPED_TEST(CLScaleTester, Random) {
    float scale[] = {2.0, 2.5, 1.5};
    float bias[] = {0.1, 0.2, 0.3};
    this->TestPrepare({2, 3, 4, 7}).TestRun(scale, bias);
}

TYPED_TEST(CLScaleTester, batch_1) {
    float input[] = {1.2, 3.6, -9.2, 14.2, 6.3, 7.1, 6.3, 7.8, -7.1, -3.7, 5.3, 0};
    float scale[] = {-0.5, 1.4};
    float bias[] = {5.3, -1.2};
    this->TestPrepare({1, 2, 3, 2}).TestRun(scale, bias, input);
}

TYPED_TEST(CLScaleTester, batch_non1) {
    float input[] = {2.4, -0.6, 1.2, 6.3, -6.2, 7.1, 9.0, 0};
    float scale[] = {0.3, 0.2};
    float bias[] = {0.2, 3.5};
    this->TestPrepare({2, 2, 1, 2}).TestRun(scale, bias, input);
}

TYPED_TEST(CLScaleTester, channel_1) {
    float input[] = {1.2, 3.4, 5.6, 0.8, 0.2, -9.0, -0.4, -1.2, 4.6, 0, 3.2, 0.6, 4.3, -6.4, 10.3, -9.5, 5.3, 4.2};
    float scale[] = {1.3};
    float bias[] = {0.5};
    this->TestPrepare({2, 1, 3, 3}).TestRun(scale, bias, input);
}

TYPED_TEST(CLScaleTester, channel_non1) {
    float input[] = {1.2, 3.4, 5.6, 0.8, 0.2, -9.0, -0.4, -1.2};
    float scale[] = {2.3, 0.4, 3.4, -0.8};
    float bias[] = {0.5, -1.1, 0.1, 5.0};
    this->TestPrepare({2, 4, 1, 1}).TestRun(scale, bias, input);
}

TYPED_TEST(CLScaleTester, element_num_4) {
    float input[] = {4.6, 9.3, 5.9, -0.5, -6.1, -4.6, 3.7, 6.2, 9.4, 0, 3.7, 4.8};
    float scale[] = {1.1, 1.3, -0.5};
    float bias[] = {4.5, -6.1, 3.6};
    this->TestPrepare({1, 3, 2, 2}).TestRun(scale, bias, input);
}

TYPED_TEST(CLScaleTester, element_num_non4) {
    float input[] = {4.6, 9.3, 5.9, -0.5, -6.1, -4.6, 3.7, 6.2, 9.4, 0, 3.7, 4.8};
    float scale[] = {-0.2, 0.6};
    float bias[] = {3.4, -2.1};
    this->TestPrepare({1, 2, 3, 2}).TestRun(scale, bias, input);
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
