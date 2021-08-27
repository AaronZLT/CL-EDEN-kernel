#include <gtest/gtest.h>
#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLTranspose.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class CLTransposeTester : public ::testing::Test {
  public:
    CLTransposeTester() {
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

    CLTransposeTester &TestPrepare(const Dim4 &input_dim, const Dim4 &output_dim) {
        input_dim_ = input_dim;
        output_dim_ = output_dim;
        input_size_ = GetDimSize(input_dim_);
        output_size_ = GetDimSize(output_dim_);
        parameters_ = std::make_shared<TransposeParameters>();

        return *this;
    }

    void TestRun(float *input_data, std::vector<int32_t> &perm, float *golden_data) {
        Dim4 perm_dim = {static_cast<uint32_t>(perm.size()), 1, 1, 1};
        auto input_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, input_dim_);
        auto perm_tensor = std::make_shared<CLTensor>(runtime_, precision_, perm.data(), perm_dim);
        auto output_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, output_dim_);

        CLTranspose transpose(runtime_, precision_);
        EXPECT_EQ(transpose.initialize({input_tensor, perm_tensor}, {output_tensor}, parameters_), Status::SUCCESS);
        EXPECT_EQ(input_tensor->writeData(input_data), Status::SUCCESS);
        EXPECT_EQ(transpose.execute(), Status::SUCCESS);
        EXPECT_EQ(transpose.release(), Status::SUCCESS);

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
    std::shared_ptr<TransposeParameters> parameters_;
};

TYPED_TEST_CASE(CLTransposeTester, TestFP32AndFP16Type);
TYPED_TEST(CLTransposeTester, TestRefOps2D_0) {
    float input[6] = {0, 1, 2, 3, 4, 5};
    float output_expect[6] = {0, 1, 2, 3, 4, 5};
    std::vector<int32_t> perm = {0, 1};
    this->TestPrepare({3, 2, 1, 1}, {3, 2, 1, 1}).TestRun(input, perm, output_expect);
}

TYPED_TEST(CLTransposeTester, TestRefOps2D_1) {
    float input[6] = {0, 1, 2, 3, 4, 5};
    float output_expect[6] = {0, 2, 4, 1, 3, 5};
    std::vector<int32_t> perm = {1, 0};
    this->TestPrepare({3, 2, 1, 1}, {3, 2, 1, 1}).TestRun(input, perm, output_expect);
}

TYPED_TEST(CLTransposeTester, TestRefOps3D_0) {
    float input[24] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    float output_expect[24] = {0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 21, 2, 6, 10, 14, 18, 22, 3, 7, 11, 15, 19, 23};
    std::vector<int32_t> perm = {2, 0, 1};
    this->TestPrepare({2, 3, 4, 1}, {2, 3, 4, 1}).TestRun(input, perm, output_expect);
}

TYPED_TEST(CLTransposeTester, TestRefOps3D_1) {
    float input[24] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    float output_expect[24] = {0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 12, 16, 20, 13, 17, 21, 14, 18, 22, 15, 19, 23};
    std::vector<int32_t> perm = {0, 2, 1};
    this->TestPrepare({2, 3, 4, 1}, {2, 3, 4, 1}).TestRun(input, perm, output_expect);
}

TYPED_TEST(CLTransposeTester, TestRefOps3D_2) {
    float input[24] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    float output_expect[24] = {0, 12, 4, 16, 8, 20, 1, 13, 5, 17, 9, 21, 2, 14, 6, 18, 10, 22, 3, 15, 7, 19, 11, 23};
    std::vector<int32_t> perm = {2, 1, 0};
    this->TestPrepare({2, 3, 4, 1}, {2, 3, 4, 1}).TestRun(input, perm, output_expect);
}

TYPED_TEST(CLTransposeTester, TestRefOps3D_3) {
    float input[24] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    float output_expect[24] = {0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 16, 17, 18, 19, 8, 9, 10, 11, 20, 21, 22, 23};
    std::vector<int32_t> perm = {1, 0, 2};
    this->TestPrepare({2, 3, 4, 1}, {2, 3, 4, 1}).TestRun(input, perm, output_expect);
}

TYPED_TEST(CLTransposeTester, TestRefOps4D_0) {
    float input[24] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
    float output_expect[24] = {0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11, 12, 14, 16, 13, 15, 17, 18, 20, 22, 19, 21, 23};
    std::vector<int32_t> perm = {0, 1, 3, 2};
    this->TestPrepare({2, 2, 3, 2}, {2, 2, 3, 2}).TestRun(input, perm, output_expect);
}

TYPED_TEST(CLTransposeTester, TestRefOps4D_1) {
    float input[120];
    for (int k = 0; k < 120; k++) {
        input[k] = k;
    }
    float output_expect[120] = {0,  1,  2,  3,  4,  20,  21,  22,  23,  24,  40, 41, 42, 43, 44, 60,  61,  62,  63,  64,
                                80, 81, 82, 83, 84, 100, 101, 102, 103, 104, 5,  6,  7,  8,  9,  25,  26,  27,  28,  29,
                                45, 46, 47, 48, 49, 65,  66,  67,  68,  69,  85, 86, 87, 88, 89, 105, 106, 107, 108, 109,
                                10, 11, 12, 13, 14, 30,  31,  32,  33,  34,  50, 51, 52, 53, 54, 70,  71,  72,  73,  74,
                                90, 91, 92, 93, 94, 110, 111, 112, 113, 114, 15, 16, 17, 18, 19, 35,  36,  37,  38,  39,
                                55, 56, 57, 58, 59, 75,  76,  77,  78,  79,  95, 96, 97, 98, 99, 115, 116, 117, 118, 119};
    std::vector<int32_t> perm = {2, 0, 1, 3};
    this->TestPrepare({2, 3, 4, 5}, {2, 3, 4, 5}).TestRun(input, perm, output_expect);
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
