#include <gtest/gtest.h>
#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLSub.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class CLSubTester : public ::testing::Test {
  public:
    CLSubTester() {
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

    CLSubTester &TestPrepare(const Dim4 &input_dim_0, const Dim4 &input_dim_1, const Dim4 &output_dim) {
        input_dim_0_ = input_dim_0;
        input_dim_1_ = input_dim_1;
        output_dim_ = output_dim;
        input_size_0_ = GetDimSize(input_dim_0_);
        input_size_1_ = GetDimSize(input_dim_1_);
        output_size_ = GetDimSize(output_dim_);
        parameters_ = std::make_shared<SubParameters>();
        return *this;
    }

    void TestRun(float *input_data_0, float *input_data_1, float *golden_data) {
        auto input_tensor_0 = std::make_shared<CLTensor>(runtime_, precision_, input_data_0, input_dim_0_);
        auto input_tensor_1 = std::make_shared<CLTensor>(runtime_, precision_, input_data_1, input_dim_1_);
        auto output_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, output_dim_);
        CLSub sub_op(runtime_, precision_);
        EXPECT_EQ(sub_op.initialize({input_tensor_0, input_tensor_1}, {output_tensor}, parameters_), Status::SUCCESS);
        EXPECT_EQ(sub_op.execute(), Status::SUCCESS);
        EXPECT_EQ(sub_op.release(), Status::SUCCESS);

        auto output_ptr = make_shared_array<float>(output_size_);
        output_tensor->readData(output_ptr.get());

        Compare(output_ptr.get(), golden_data, output_size_, error_threshold_);
    }

  private:
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;
    DataType data_type_;
    float error_threshold_;

    Dim4 input_dim_0_;
    Dim4 input_dim_1_;
    Dim4 output_dim_;
    size_t input_size_0_;
    size_t input_size_1_;
    size_t output_size_;
    std::shared_ptr<SubParameters> parameters_;
};

TYPED_TEST_CASE(CLSubTester, TestFP32AndFP16Type);
TYPED_TEST(CLSubTester, test_sub_1_no_broadcast) {
    float input_0[] = {-2.0, 0.2, 1.7, 0.5};
    float input_1[] = {0.1, 0.2, 0.3, 0.8};
    float golden_data[] = {-2.1, 0.0, 1.4, -0.3};
    this->TestPrepare({1, 2, 2, 1}, {1, 2, 2, 1}, {1, 2, 2, 1}).TestRun(input_0, input_1, golden_data);
}

TYPED_TEST(CLSubTester, test_sub_2_no_broadcast) {
    float input_0[] = {-2.0, 0.2, 1.7, 0.5, -1.1, 2.0};
    float input_1[] = {0.1, 0.2, 0.3, 0.8, -1.1, 0.1};
    float golden_data[] = {-2.1, 0.0, 1.4, -0.3, 0.0, 1.9};
    this->TestPrepare({1, 3, 1, 2}, {1, 3, 1, 2}, {1, 3, 1, 2}).TestRun(input_0, input_1, golden_data);
}

TYPED_TEST(CLSubTester, test_sub_3_with_broadcast) {
    float input_0[] = {-2.0, 0.2, 1.7, 0.5, -1.1, 2.0};
    float input_1[] = {0.5};
    float golden_data[] = {-2.5, -0.3, 1.2, 0.0, -1.6, 1.5};
    this->TestPrepare({1, 2, 1, 3}, {1, 1, 1, 1}, {1, 2, 1, 3}).TestRun(input_0, input_1, golden_data);
}

TYPED_TEST(CLSubTester, test_sub_4_with_broadcast) {
    float input_0[] = {-20, 2, 7, 8, 11, 20};
    float input_1[] = {1};
    float golden_data[] = {-21, 1, 6, 7, 10, 19};
    this->TestPrepare({1, 1, 1, 6}, {1, 1, 1, 1}, {1, 1, 1, 6}).TestRun(input_0, input_1, golden_data);
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
