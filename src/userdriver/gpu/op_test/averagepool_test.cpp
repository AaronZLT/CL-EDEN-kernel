#include <gtest/gtest.h>
#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLAveragepool.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class CLAveragepoolTester : public ::testing::Test {
  public:
    CLAveragepoolTester() {
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

    CLAveragepoolTester &
    TestPrepare(const Dim4 &input_dim,
                const Pad4 &pad,
                const Dim2 &stride,
                const Dim2 &filter,
                const ComputeType &compute_type,
                const ActivationInfo::ActivationType &activation_type = ActivationInfo::ActivationType::NONE,
                const bool &activation_enabled = false,
                const Dim4 &output_dim = {0, 0, 0, 0}) {
        input_dim_ = input_dim;
        input_size_ = GetDimSize(input_dim_);
        parameters_ = std::make_shared<Pool2DParameters>();
        parameters_->padding = pad;
        parameters_->stride = stride;
        parameters_->filter = filter;
        parameters_->compute_type = compute_type;
        parameters_->activation_info = ActivationInfo(activation_type, activation_enabled);
        output_dim_ = output_dim;
        output_size_ = GetDimSize(output_dim_);
        return *this;
    }

    void TestRun(float *input_data = nullptr, float *golden_data = nullptr) {
        std::shared_ptr<float> input;
        std::shared_ptr<float> output;
        if (input_data == nullptr || golden_data == nullptr) {
            AveragepoolGuard averagepool_guard;
            averagepool_guard.GuardPrepare(
                input_dim_, parameters_->padding, parameters_->stride, parameters_->filter, output_dim_);
            output_size_ = GetDimSize(output_dim_);

            input = make_shared_array<float>(input_size_);
            output = make_shared_array<float>(output_size_);
            GenerateRandom<float>(input.get(), input_size_, 0, 1);
            memset(output.get(), 0, output_size_ * sizeof(float));

            averagepool_guard.GuardRun(input.get(), output.get());
            input_data = input.get();
            golden_data = output.get();
        }

        doRun(input_data, golden_data);
    }

  private:
    void doRun(float *input_data, float *golden_data) {
        auto input_tensor = std::make_shared<CLTensor>(runtime_, precision_, input_data, input_dim_);
        auto output_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, output_dim_);
        CLAveragepool averagepool_pool(runtime_, precision_);
        EXPECT_EQ(averagepool_pool.initialize({input_tensor}, {output_tensor}, parameters_), Status::SUCCESS);
        EXPECT_EQ(averagepool_pool.execute(), Status::SUCCESS);
        EXPECT_EQ(averagepool_pool.release(), Status::SUCCESS);

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
    std::shared_ptr<Pool2DParameters> parameters_;
};

TYPED_TEST_CASE(CLAveragepoolTester, TestFP32AndFP16Type);
TYPED_TEST(CLAveragepoolTester, SimpleTest) {
    this->TestPrepare({1, 18, 52, 52}, {3, 3, 3, 3}, {1, 1}, {7, 7}, ComputeType::Caffe).TestRun();
}

TYPED_TEST(CLAveragepoolTester, NoPadding) {
    this->TestPrepare({1, 64, 112, 112}, {0, 0, 0, 0}, {2, 2}, {3, 3}, ComputeType::Caffe).TestRun();
}

TYPED_TEST(CLAveragepoolTester, NoPadding_Batch_3) {
    this->TestPrepare({3, 64, 112, 112}, {0, 0, 0, 0}, {2, 2}, {3, 3}, ComputeType::Caffe).TestRun();
}

TYPED_TEST(CLAveragepoolTester, NoPadding_Batch_4) {
    this->TestPrepare({4, 64, 112, 112}, {0, 0, 0, 0}, {2, 2}, {3, 3}, ComputeType::Caffe).TestRun();
}

TYPED_TEST(CLAveragepoolTester, OddChannel) {
    this->TestPrepare({2, 64, 112, 112}, {0, 0, 0, 0}, {2, 2}, {3, 3}, ComputeType::Caffe).TestRun();
}

TYPED_TEST(CLAveragepoolTester, NotEqualHeightWidth) {
    this->TestPrepare({1, 64, 64, 112}, {0, 0, 0, 0}, {2, 2}, {3, 3}, ComputeType::Caffe).TestRun();
}

TYPED_TEST(CLAveragepoolTester, Kernel_5) {
    this->TestPrepare({1, 64, 112, 112}, {0, 0, 0, 0}, {2, 2}, {5, 5}, ComputeType::Caffe).TestRun();
}

TYPED_TEST(CLAveragepoolTester, Kernel_7) {
    this->TestPrepare({1, 64, 112, 112}, {0, 0, 0, 0}, {2, 2}, {7, 7}, ComputeType::Caffe).TestRun();
}

TYPED_TEST(CLAveragepoolTester, Padding_1) {
    this->TestPrepare({1, 64, 112, 112}, {1, 1, 1, 1}, {2, 2}, {3, 3}, ComputeType::Caffe).TestRun();
}

TYPED_TEST(CLAveragepoolTester, Padding_1_Kernel_3) {
    this->TestPrepare({1, 64, 56, 56}, {1, 1, 1, 1}, {1, 1}, {3, 3}, ComputeType::Caffe).TestRun();
}

TYPED_TEST(CLAveragepoolTester, Padding_2_Kernel_5) {
    this->TestPrepare({1, 64, 56, 56}, {2, 2, 2, 2}, {1, 1}, {5, 5}, ComputeType::Caffe).TestRun();
}

TYPED_TEST(CLAveragepoolTester, Padding_3_Kernel_7) {
    this->TestPrepare({1, 64, 56, 56}, {3, 3, 3, 3}, {1, 1}, {7, 7}, ComputeType::Caffe).TestRun();
}

TYPED_TEST(CLAveragepoolTester, VTS1) {
    float input_data[] = {1.0, 2.0, 3.0, 4.0};
    float golden_data[] = {1.0, 2.0, 3.0, 4.0};
    this->TestPrepare({1, 2, 2, 1},
                      {0, 0, 0, 0},
                      {1, 1},
                      {1, 1},
                      ComputeType::TFLite,
                      ActivationInfo::ActivationType::NONE,
                      false,
                      {1, 2, 2, 1})
        .TestRun(input_data, golden_data);
}
TYPED_TEST(CLAveragepoolTester, VTS2) {
    float input_data[] = {0.0, 6.0, 2.0, 4.0, 3.0, 2.0, 10.0, 7.0};
    float golden_data[] = {2.75, 5.75};
    this->TestPrepare({1, 1, 2, 4},
                      {0, 0, 0, 0},
                      {2, 2},
                      {2, 2},
                      ComputeType::TFLite,
                      ActivationInfo::ActivationType::NONE,
                      false,
                      {1, 1, 1, 2})
        .TestRun(input_data, golden_data);
}
TYPED_TEST(CLAveragepoolTester, VTS3) {
    float input_data[] = {0.0, 6.0, 2.0, 4.0, 3.0, 2.0, 10.0, 7.0};
    float golden_data[] = {2.75, 5.0, 5.75};
    this->TestPrepare({1, 1, 2, 4},
                      {0, 0, 0, 0},
                      {1, 1},
                      {2, 2},
                      ComputeType::TFLite,
                      ActivationInfo::ActivationType::NONE,
                      false,
                      {1, 1, 1, 3})
        .TestRun(input_data, golden_data);
}
TYPED_TEST(CLAveragepoolTester, VTS4) {
    float input_data[] = {0.0, 6.0, 2.0, 4.0, 3.0, 2.0, 10.0, 7.0};
    float golden_data[] = {0.0, 3.0, 4.0, 3.0, 4.0, 1.5, 2.75, 5.0, 5.75, 5.5, 3.0, 2.5, 6.0, 8.5, 7.0};
    this->TestPrepare({1, 1, 2, 4},
                      {1, 1, 1, 1},
                      {1, 1},
                      {2, 2},
                      ComputeType::TFLite,
                      ActivationInfo::ActivationType::NONE,
                      false,
                      {1, 1, 3, 5})
        .TestRun(input_data, golden_data);
}
TYPED_TEST(CLAveragepoolTester, VTS5) {
    float input_data[] = {-2.5, 2.4, 0.5, 4.0};
    float golden_data[] = {-1.0, 1.0, 0.5, 1.0};
    this->TestPrepare({1, 2, 2, 1},
                      {0, 0, 0, 0},
                      {1, 1},
                      {1, 1},
                      ComputeType::TFLite,
                      ActivationInfo::ActivationType::RELU1,
                      true,
                      {1, 2, 2, 1})
        .TestRun(input_data, golden_data);
}
TYPED_TEST(CLAveragepoolTester, VTS6) {
    float input_data[] = {0.0, 6.0, 2.0, 4.0, 3.0, 2.0, 10.0, 7.0};
    float golden_data[] = {1, 1, 1};
    this->TestPrepare({1, 1, 2, 4},
                      {0, 0, 0, 0},
                      {1, 1},
                      {2, 2},
                      ComputeType::TFLite,
                      ActivationInfo::ActivationType::RELU1,
                      true,
                      {1, 1, 1, 3})
        .TestRun(input_data, golden_data);
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
