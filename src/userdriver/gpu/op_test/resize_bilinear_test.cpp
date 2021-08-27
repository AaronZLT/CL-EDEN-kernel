#include <gtest/gtest.h>
#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLResizeBilinear.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class CLResizeBilinearTester : public ::testing::Test {
  public:
    CLResizeBilinearTester() {
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

    CLResizeBilinearTester &TestPrepare(const Dim4 &input_dim, const Dim4 &output_dim) {
        input_dim_ = input_dim;
        output_dim_ = output_dim;
        input_size_ = GetDimSize(input_dim_);
        output_size_ = GetDimSize(output_dim_);
        parameters_ = std::make_shared<ResizeBilinearParameters>();
        parameters_->new_height = output_dim_.h;
        parameters_->new_width = output_dim_.w;
        return *this;
    }

    void TestRun(float *input_data, float *golden_data) {
        auto input_tensor = std::make_shared<CLTensor>(runtime_, precision_, input_data, input_dim_);
        auto output_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, output_dim_);
        CLResizeBilinear resize_bilinear(runtime_, precision_);
        EXPECT_EQ(resize_bilinear.initialize({input_tensor}, {output_tensor}, parameters_), Status::SUCCESS);
        EXPECT_EQ(resize_bilinear.execute(), Status::SUCCESS);
        EXPECT_EQ(resize_bilinear.release(), Status::SUCCESS);

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
    std::shared_ptr<ResizeBilinearParameters> parameters_;
};

TYPED_TEST_CASE(CLResizeBilinearTester, TestFP32AndFP16Type);
TYPED_TEST(CLResizeBilinearTester, Normal) {
    float input_data[] = {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22};
    float inference_output_data[] = {0,  1,  2,  2,  2,  3,  4,  4,  4,  5,  6,  6,  4,  5,  6,  6,
                                     8,  9,  10, 10, 10, 11, 12, 12, 12, 13, 14, 14, 12, 13, 14, 14,
                                     16, 17, 18, 18, 18, 19, 20, 20, 20, 21, 22, 22, 20, 21, 22, 22};
    this->TestPrepare({1, 3, 2, 2}, {1, 3, 4, 4}).TestRun(input_data, inference_output_data);
}

TYPED_TEST(CLResizeBilinearTester, TestBilinear2x2To1x1) {
    float input_data[] = {1, 2, 3, 4};
    float inference_output_data[] = {1};
    this->TestPrepare({1, 1, 2, 2}, {1, 1, 1, 1}).TestRun(input_data, inference_output_data);
}

TYPED_TEST(CLResizeBilinearTester, TestBilinear2x2To3x3) {
    float input_data[] = {1, 2, 3, 4};
    float inference_output_data[] = {1, 5.0f / 3, 2, 7.0f / 3, 3, 10.0f / 3, 3, 11.0f / 3, 4};
    this->TestPrepare({1, 1, 2, 2}, {1, 1, 3, 3}).TestRun(input_data, inference_output_data);
}

TYPED_TEST(CLResizeBilinearTester, TestBilinear3x3To2x2) {
    float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    float inference_output_data[] = {1, 2.5, 5.5, 7};
    this->TestPrepare({1, 1, 3, 3}, {1, 1, 2, 2}).TestRun(input_data, inference_output_data);
}

TYPED_TEST(CLResizeBilinearTester, TestBilinear4x4To3x3) {
    float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    float inference_output_data[] = {
        1, 7.0f / 3, 11.0f / 3, 19.0f / 3, 23.0f / 3, 27.0f / 3, 35.0f / 3, 39.0f / 3, 43.0f / 3};
    this->TestPrepare({1, 1, 4, 4}, {1, 1, 3, 3}).TestRun(input_data, inference_output_data);
}

TYPED_TEST(CLResizeBilinearTester, TestBilinear2x2To3x3Batch2) {
    float input_data[] = {1, 2, 3, 4, 1, 2, 3, 4};
    float inference_output_data[] = {
        1, 5.0f / 3, 2, 7.0f / 3, 3, 10.0f / 3, 3, 11.0f / 3, 4, 1, 5.0f / 3, 2, 7.0f / 3, 3, 10.0f / 3, 3, 11.0f / 3, 4};
    this->TestPrepare({2, 1, 2, 2}, {2, 1, 3, 3}).TestRun(input_data, inference_output_data);
}

TYPED_TEST(CLResizeBilinearTester, TestBilinear2x2To4x4) {
    float input_data[] = {1, 2, 3, 4, 1, 2, 3, 4};
    float inference_output_data[] = {1, 1.5, 2, 2, 2, 2.5, 3, 3, 3, 3.5, 4, 4, 3, 3.5, 4, 4};
    this->TestPrepare({1, 1, 2, 2}, {1, 1, 4, 4}).TestRun(input_data, inference_output_data);
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
