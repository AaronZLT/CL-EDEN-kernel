#include <gtest/gtest.h>
#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLDiv.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class CLDivTester : public ::testing::Test {
  public:
    CLDivTester() {
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

    CLDivTester &TestPrepare(const Dim4 &input_dim_0, const Dim4 &input_dim_1, const Dim4 &output_dim) {
        input_dim_0_ = input_dim_0;
        input_dim_1_ = input_dim_1;
        output_dim_ = output_dim;
        input_size_0_ = GetDimSize(input_dim_0_);
        input_size_1_ = GetDimSize(input_dim_1_);
        output_size_ = GetDimSize(output_dim_);
        parameters_ = std::make_shared<DivParameters>();
        return *this;
    }

    void TestRun(float *input_data_0, float *input_data_1, float *golden_data) {
        auto input_tensor_0 = std::make_shared<CLTensor>(runtime_, precision_, input_data_0, input_dim_0_);
        auto input_tensor_1 = std::make_shared<CLTensor>(runtime_, precision_, input_data_1, input_dim_1_);
        auto output_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, output_dim_);
        CLDiv div_op(runtime_, precision_);
        EXPECT_EQ(div_op.initialize({input_tensor_0, input_tensor_1}, {output_tensor}, parameters_), Status::SUCCESS);
        EXPECT_EQ(div_op.execute(), Status::SUCCESS);
        EXPECT_EQ(div_op.release(), Status::SUCCESS);

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
    std::shared_ptr<DivParameters> parameters_;
};

TYPED_TEST_CASE(CLDivTester, TestFP32AndFP16Type);
TYPED_TEST(CLDivTester, test_1_shape) {
    float input1[] = {-0.2, 0.2, -1.2, 0.8};
    float input2[] = {0.5, 0.2, -1.5, 0.5};
    float expect_out[] = {-0.4, 1.0, 0.8, 1.6};
    this->TestPrepare({1, 2, 2, 1}, {1, 2, 2, 1}, {1, 2, 2, 1}).TestRun(input1, input2, expect_out);
}

TYPED_TEST(CLDivTester, test_2_shape) {
    float input1[] = {-0.2, 0.2, 0.07, 0.08, 0.11, -0.123};
    float input2[] = {0.1};
    float expect_out[] = {-2.0, 2.0, 0.7, 0.8, 1.1, -1.23};
    this->TestPrepare({1, 3, 1, 2}, {1, 1, 1, 1}, {1, 3, 1, 2}).TestRun(input1, input2, expect_out);
}

TYPED_TEST(CLDivTester, test_3_shape) {
    float input1[] = {-2.0, 0.2, 0.3, 0.8, 1.1, -2.0};
    float input2[] = {0.1, 0.2, 0.6, 0.5, -1.1, -0.1};
    float expect_out[] = {-20.0, 1.0, 0.5, 1.6, -1.0, 20.0};
    this->TestPrepare({1, 1, 1, 6}, {1, 1, 1, 6}, {1, 1, 1, 6}).TestRun(input1, input2, expect_out);
}

TYPED_TEST(CLDivTester, test_4_shape) {
    float input1[] = {-24.0, 21.0, 9.0, 18.0, 12.0, -123.0};
    float input2[] = {3.0};
    float expect_out[] = {-8.0, 7.0, 3.0, 6.0, 4.0, -41.0};
    this->TestPrepare({1, 2, 1, 3}, {1, 1, 1, 1}, {1, 2, 1, 3}).TestRun(input1, input2, expect_out);
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
