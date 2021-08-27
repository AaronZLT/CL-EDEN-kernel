#include <gtest/gtest.h>
#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLSoftmax.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class CLSoftmaxTester : public ::testing::Test {
  public:
    CLSoftmaxTester() {
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

    CLSoftmaxTester &TestPrepare(const Dim4 &input_dim, const int32_t &axis, const float &beta) {
        input_dim_ = input_dim;
        output_dim_ = input_dim;
        input_size_ = GetDimSize(input_dim_);
        output_size_ = GetDimSize(output_dim_);
        parameters_ = std::make_shared<SoftmaxParameters>();
        parameters_->axis = axis;
        parameters_->beta = beta;
        return *this;
    }

    void TestRun(float *input_data = nullptr, float *golden_data = nullptr) {
        std::shared_ptr<float> input;
        std::shared_ptr<float> output;
        if (input_data == nullptr || golden_data == nullptr) {
            input = make_shared_array<float>(input_size_);
            output = make_shared_array<float>(output_size_);
            GenerateRandom<float>(input.get(), input_size_, 0, 1);
            memset(output.get(), 0, output_size_ * sizeof(float));

            SoftmaxGuard softmax_guard;
            softmax_guard.GuardPrepare(input_dim_, parameters_->axis, parameters_->beta);
            softmax_guard.GuardRun(input.get(), output.get());
            input_data = input.get();
            golden_data = output.get();
        }

        doRun(input_data, golden_data);
    }

  private:
    void doRun(float *input_data, float *golden_data) {
        auto input_tensor = std::make_shared<CLTensor>(runtime_, precision_, input_data, input_dim_);
        auto output_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, output_dim_);
        CLSoftmax softmax(runtime_, precision_);
        EXPECT_EQ(softmax.initialize({input_tensor}, {output_tensor}, parameters_), Status::SUCCESS);
        EXPECT_EQ(softmax.execute(), Status::SUCCESS);
        EXPECT_EQ(softmax.release(), Status::SUCCESS);

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
    std::shared_ptr<SoftmaxParameters> parameters_;
};

TYPED_TEST_CASE(CLSoftmaxTester, TestFP32AndFP16Type);
TYPED_TEST(CLSoftmaxTester, test_axis0_batch8xsize1) {
    const int32_t axis = 0;
    const float beta = 1.0;
    this->TestPrepare({64, 1, 1, 1}, axis, beta).TestRun();
}

TYPED_TEST(CLSoftmaxTester, test_axis_0) {
    const int32_t axis = 0;
    const float beta = 1.0;
    this->TestPrepare({2, 3, 4, 7}, axis, beta).TestRun();
}

TYPED_TEST(CLSoftmaxTester, axis_batch_1_axis_0) {
    const int32_t axis = 0;
    const float beta = 1.0;
    float input_data[] = {2.1, 1.3, 4.5, 0.2, 0, -1.2, 3.5, -0.1, 2.4, -9.1, 3.6, 2.7};
    float golden_data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    this->TestPrepare({1, 2, 3, 2}, axis, beta).TestRun(input_data, golden_data);
}

TYPED_TEST(CLSoftmaxTester, axis_batch_non1_axis_0) {
    const int32_t axis = 0;
    const float beta = 1.0;
    float input_data[] = {9.1, 0.4, 0, -12.8, 4.6, -9.1, -7.3, 0.4, 4.7, 9.2, 5.3, 7.1, 6.2, 3.8, -6.3, 5.2};
    float golden_data[] = {0.98787,
                           0.00015,
                           0.00497,
                           0.00000,
                           0.16798,
                           0.00000,
                           0.26894,
                           0.00816,
                           0.01213,
                           0.99985,
                           0.99503,
                           1.00000,
                           0.83202,
                           1.00000,
                           0.73106,
                           0.99184};
    this->TestPrepare({2, 2, 2, 2}, axis, beta).TestRun(input_data, golden_data);
}

TYPED_TEST(CLSoftmaxTester, axis_channel_1_axis_1) {
    const int32_t axis = 1;
    const float beta = 1.0;
    float input_data[] = {2.1, 1.3, 4.5, 0.2, 0, -1.2, 3.5, -0.1, 2.4, -9.1, 3.6, 2.7};
    float golden_data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    this->TestPrepare({2, 1, 3, 2}, axis, beta).TestRun(input_data, golden_data);
}

TYPED_TEST(CLSoftmaxTester, axis_channel_non1_axis_1) {
    const int32_t axis = 1;
    const float beta = 1.0;
    float input_data[] = {2.1, 1.3,  4.5,  0.2,   0, -1.2, 3.5, -0.1, 2.4, -9.1, 3.6, 2.7,
                          5.3, -8.1, 12.5, -16.9, 0, 0.5,  2.6, 8.1,  7.3, -5.2, 2.6, 8.1};
    float golden_data[] = {0.19782, 0.80218, 0.89090, 0.99991, 0.02660, 0.01984, 0.80218, 0.19782,
                           0.10910, 0.00009, 0.97340, 0.98016, 0.93703, 0.00000, 0.99451, 0.00001,
                           0.06914, 0.00050, 0.06297, 1.00000, 0.00549, 0.99999, 0.93086, 0.99950};
    this->TestPrepare({2, 2, 3, 2}, axis, beta).TestRun(input_data, golden_data);
}

TYPED_TEST(CLSoftmaxTester, axis_height_1_axis_2) {
    const int32_t axis = 2;
    const float beta = 1.0;
    float input_data[] = {2.1, 1.3, 4.5, 0.2, 0, -1.2, 3.5, -0.1, 2.4, -9.1, 3.6, 2.7, -4.6, -10.3, 5.7, 2.8, 0};
    float golden_data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    this->TestPrepare({2, 3, 1, 3}, axis, beta).TestRun(input_data, golden_data);
}

TYPED_TEST(CLSoftmaxTester, axis_height_non1_axis_2) {
    const int32_t axis = 2;
    const float beta = 1.0;
    float input_data[] = {2.1, 1.3,  4.5,  0.2,   0, -1.2, 3.5, -0.1, 2.4, -9.1, 3.6, 2.7,
                          5.3, -8.1, 12.5, -16.9, 0, 0.5,  2.6, 8.1,  7.3, -5.2, 2.6, 8.1};
    float golden_data[] = {0.08317, 0.75026, 0.91683, 0.24974, 0.02931, 0.24974, 0.97069, 0.75026,
                           0.23148, 0.00001, 0.76852, 0.99999, 0.00075, 0.99985, 0.99925, 0.00015,
                           0.06914, 0.00050, 0.93086, 0.99950, 0.99099, 0.00000, 0.00901, 1.00000};
    this->TestPrepare({2, 3, 2, 2}, axis, beta).TestRun(input_data, golden_data);
}

TYPED_TEST(CLSoftmaxTester, axis_width_1_axis_3) {
    const int32_t axis = 3;
    const float beta = 1.0;
    float input_data[] = {2.1, 1.3, 4.5, 0.2, 0, -1.2, 3.5, -0.1, 2.4, -9.1, 3.6, 2.7, -4.6, -10.3, 5.7, 2.8, 0};
    float golden_data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    this->TestPrepare({2, 3, 3, 1}, axis, beta).TestRun(input_data, golden_data);
}

TYPED_TEST(CLSoftmaxTester, axis_width_non1_axis_1) {
    const int32_t axis = 3;
    const float beta = 1.0;
    float input_data[] = {-6.3, 9.2, 5.8, 0, 2.1, 1.3, 4.5, 0.2, 0, -1.2, 3.5, -0.1, 2.4, -9.1, 3.6, 2.7};
    float golden_data[] = {0.00000,
                           1.00000,
                           0.99698,
                           0.00302,
                           0.68997,
                           0.31003,
                           0.98661,
                           0.01339,
                           0.76852,
                           0.23148,
                           0.97340,
                           0.02660,
                           0.99999,
                           0.00001,
                           0.71095,
                           0.28905};
    this->TestPrepare({2, 2, 2, 2}, axis, beta).TestRun(input_data, golden_data);
}

TYPED_TEST(CLSoftmaxTester, axis_axis_6) {
    const int32_t axis = 6;
    const float beta = 1.0;
    float input_data[] = {-6.3, 9.2, 5.8, 0, 2.1, 1.3, 4.5, 0.2, 0, -1.2, 3.5, -0.1, 2.4, -9.1, 3.6, 2.7};
    float golden_data[] = {0.00000,
                           1.00000,
                           0.99698,
                           0.00302,
                           0.68997,
                           0.31003,
                           0.98661,
                           0.01339,
                           0.76852,
                           0.23148,
                           0.97340,
                           0.02660,
                           0.99999,
                           0.00001,
                           0.71095,
                           0.28905};
    this->TestPrepare({2, 2, 2, 2}, axis, beta).TestRun(input_data, golden_data);
}

TYPED_TEST(CLSoftmaxTester, axis_axis_f3) {
    const int32_t axis = -3;
    const float beta = 1.0;
    float input_data[] = {2.1, 1.3, 4.5, 0.2, 0, -1.2, 3.5, -0.1, 2.4, -9.1, 3.6, 2.7};
    float golden_data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    this->TestPrepare({2, 1, 3, 2}, axis, beta).TestRun(input_data, golden_data);
}

// The order of input data and output data is NCHW
TYPED_TEST(CLSoftmaxTester, vts1) {
    const int32_t axis = 1;
    const float beta = 0.000001;
    float input_data[4] = {1.0, 2.0, 10.0, 20.0};
    float golden_data[4] = {0.25, 0.25, 0.25, 0.25};
    this->TestPrepare({1, 4, 1, 1}, axis, beta).TestRun(input_data, golden_data);
}

TYPED_TEST(CLSoftmaxTester, vts2) {
    const int32_t axis = 1;
    const float beta = 1.0;
    float input_data[10] = {1., 2., 3., 4., 5., -1., -2., -3., -4., -5.};
    float golden_data[10] = {0.011656231,
                             0.031684921,
                             0.086128544,
                             0.234121657,
                             0.636408647,
                             0.636408647,
                             0.234121657,
                             0.086128544,
                             0.031684921,
                             0.011656231};
    this->TestPrepare({2, 5, 1, 1}, axis, beta).TestRun(input_data, golden_data);
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
