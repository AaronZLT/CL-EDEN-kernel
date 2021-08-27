#include <gtest/gtest.h>
#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLMaxpool.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class CLMaxpoolTester : public ::testing::Test {
  public:
    CLMaxpoolTester() {
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

    CLMaxpoolTester &TestPrepare(const Dim4 &input_dim, const Pad4 &pad, const Dim2 &stride, const Dim2 &filter) {
        input_dim_ = input_dim;
        input_size_ = GetDimSize(input_dim_);
        parameters_ = std::make_shared<Pool2DParameters>();
        parameters_->padding = pad;
        parameters_->stride = stride;
        parameters_->filter = filter;
        return *this;
    }

    void TestRun() {
        MaxpoolGuard maxpool_guard;
        maxpool_guard.GuardPrepare(input_dim_, parameters_->padding, parameters_->stride, parameters_->filter, output_dim_);
        output_size_ = GetDimSize(output_dim_);

        std::shared_ptr<float> input = make_shared_array<float>(input_size_);
        std::shared_ptr<float> output = make_shared_array<float>(output_size_);
        GenerateRandom<float>(input.get(), input_size_, 0, 1);
        memset(output.get(), 0, output_size_ * sizeof(float));

        maxpool_guard.GuardRun(input.get(), output.get());
        doRun(input.get(), output.get());
    }

  private:
    void doRun(float *input_data, float *golden_data) {
        auto input_tensor = std::make_shared<CLTensor>(runtime_, precision_, input_data, input_dim_);
        auto output_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, output_dim_);
        CLMaxpool max_pool(runtime_, precision_);
        EXPECT_EQ(max_pool.initialize({input_tensor}, {output_tensor}, parameters_), Status::SUCCESS);
        EXPECT_EQ(max_pool.execute(), Status::SUCCESS);
        EXPECT_EQ(max_pool.release(), Status::SUCCESS);

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

TYPED_TEST_CASE(CLMaxpoolTester, TestFP32AndFP16Type);
TYPED_TEST(CLMaxpoolTester, SimpleTest) { this->TestPrepare({1, 1, 8, 8}, {0, 0, 0, 0}, {2, 2}, {2, 2}).TestRun(); }

TYPED_TEST(CLMaxpoolTester, NoPadding) { this->TestPrepare({1, 64, 112, 112}, {0, 0, 0, 0}, {2, 2}, {3, 3}).TestRun(); }

TYPED_TEST(CLMaxpoolTester, NoPadding_Batch_3) {
    this->TestPrepare({3, 64, 112, 112}, {0, 0, 0, 0}, {2, 2}, {3, 3}).TestRun();
}

TYPED_TEST(CLMaxpoolTester, NoPadding_Batch_4) {
    this->TestPrepare({4, 64, 112, 112}, {0, 0, 0, 0}, {2, 2}, {3, 3}).TestRun();
}

TYPED_TEST(CLMaxpoolTester, NoPadding_Inconsistent_Height_Width_1) {
    this->TestPrepare({1, 2, 512, 154}, {0, 0, 0, 0}, {2, 2}, {3, 3}).TestRun();
}

TYPED_TEST(CLMaxpoolTester, NoPadding_Inconsistent_Height_Width_2) {
    this->TestPrepare({1, 7, 256, 7}, {0, 0, 0, 0}, {2, 2}, {3, 3}).TestRun();
}

TYPED_TEST(CLMaxpoolTester, Stride_1) { this->TestPrepare({1, 2, 512, 256}, {0, 0, 0, 0}, {1, 1}, {3, 3}).TestRun(); }

TYPED_TEST(CLMaxpoolTester, Stride_2_1) { this->TestPrepare({1, 1, 8, 4}, {0, 0, 0, 0}, {2, 2}, {3, 3}).TestRun(); }

TYPED_TEST(CLMaxpoolTester, Stride_2_2) { this->TestPrepare({1, 1, 128, 64}, {0, 0, 0, 0}, {2, 2}, {3, 3}).TestRun(); }

TYPED_TEST(CLMaxpoolTester, Stride_2_3) { this->TestPrepare({1, 33, 16, 8}, {0, 0, 0, 0}, {2, 2}, {3, 3}).TestRun(); }

TYPED_TEST(CLMaxpoolTester, Stride_3) { this->TestPrepare({1, 2, 512, 256}, {0, 0, 0, 0}, {3, 3}, {3, 3}).TestRun(); }

TYPED_TEST(CLMaxpoolTester, Stride_4) { this->TestPrepare({1, 2, 512, 256}, {0, 0, 0, 0}, {4, 4}, {3, 3}).TestRun(); }

TYPED_TEST(CLMaxpoolTester, Kernel_3x3) { this->TestPrepare({1, 2, 512, 256}, {1, 1, 1, 1}, {1, 1}, {3, 3}).TestRun(); }

TYPED_TEST(CLMaxpoolTester, Kernel_5x5) { this->TestPrepare({1, 2, 512, 256}, {2, 2, 2, 2}, {1, 1}, {5, 5}).TestRun(); }

TYPED_TEST(CLMaxpoolTester, Kernel_7x7) { this->TestPrepare({1, 2, 512, 256}, {3, 3, 3, 3}, {1, 1}, {7, 7}).TestRun(); }

TYPED_TEST(CLMaxpoolTester, unequal) { this->TestPrepare({2, 3, 40, 60}, {2, 2, 3, 3}, {2, 3}, {3, 5}).TestRun(); }

}  // namespace gpu
}  // namespace ud
}  // namespace enn
