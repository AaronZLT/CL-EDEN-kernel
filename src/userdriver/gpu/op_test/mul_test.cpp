#include <gtest/gtest.h>
#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLMul.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class CLMulTester : public ::testing::Test {
  public:
    CLMulTester() {
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
        input_dims_.clear();
    }

    CLMulTester &TestPrepare(const std::vector<Dim4> &input_dims, const Dim4 &output_dim) {
        for (auto input_dim : input_dims) {
            input_dims_.push_back(input_dim);
            input_sizes_.push_back(GetDimSize(input_dim));
        }
        output_dim_ = output_dim;
        output_size_ = GetDimSize(output_dim);
        parameters_ = std::make_shared<MulParameters>();
        return *this;
    }

    void TestRun(std::vector<float *> input_data = {}, float *golden_data = nullptr) {
        std::vector<std::shared_ptr<float>> inputs(input_sizes_.size());
        std::shared_ptr<float> output;
        if (golden_data == nullptr) {
            input_data.clear();
            for (size_t i = 0; i < input_sizes_.size(); ++i) {
                inputs[i] = make_shared_array<float>(input_sizes_[i]);
                GenerateRandom<float>(inputs[i].get(), input_sizes_[i], 0, 1);
                input_data.push_back(inputs[i].get());
            }
            output = make_shared_array<float>(output_size_);
            memset(output.get(), 0, output_size_ * sizeof(float));
            golden_data = output.get();

            MulGuard mul_guard;
            mul_guard.GuardPrepare(input_dims_[0]);
            mul_guard.GuardRun(input_data, golden_data);
        }

        doRun(input_data, golden_data);
    }

  private:
    void doRun(std::vector<float *> &input_data, float *golden_data) {
        std::vector<std::shared_ptr<ITensor>> input_tensors;
        for (size_t i = 0; i < input_data.size(); ++i) {
            auto input_tensor = std::make_shared<CLTensor>(runtime_, precision_, input_data[i], input_dims_[i]);
            input_tensors.push_back(input_tensor);
        }
        auto output_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, output_dim_);
        CLMul mul_op(runtime_, precision_);
        EXPECT_EQ(mul_op.initialize(input_tensors, {output_tensor}, parameters_), Status::SUCCESS);
        EXPECT_EQ(mul_op.execute(), Status::SUCCESS);
        EXPECT_EQ(mul_op.release(), Status::SUCCESS);

        auto output_ptr = make_shared_array<float>(output_size_);
        output_tensor->readData(output_ptr.get());

        Compare(output_ptr.get(), golden_data, output_size_, error_threshold_);
    }

  private:
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;
    DataType data_type_;
    float error_threshold_;

    std::vector<Dim4> input_dims_;
    Dim4 output_dim_;
    std::vector<size_t> input_sizes_;
    size_t output_size_;
    std::shared_ptr<MulParameters> parameters_;
};

TYPED_TEST_CASE(CLMulTester, TestFP32AndFP16Type);
TYPED_TEST(CLMulTester, eltwise_test_0) {
    this->TestPrepare({{2, 3, 3, 3}, {2, 3, 3, 3}}, {2, 3, 3, 3}).TestRun();
}

TYPED_TEST(CLMulTester, eltwise_test_1) {
    this->TestPrepare({{2, 1, 1, 1}, {2, 1, 1, 1}, {2, 1, 1, 1}}, {2, 1, 1, 1}).TestRun();
}

TYPED_TEST(CLMulTester, eltwise_test_2) {
    this->TestPrepare({{2, 3, 3, 3}, {2, 3, 3, 3}, {2, 3, 3, 3}}, {2, 3, 3, 3}).TestRun();
}

TYPED_TEST(CLMulTester, test_Random1) {
    this->TestPrepare({{2, 3, 4, 7}, {2, 3, 4, 7}}, {2, 3, 4, 7}).TestRun();
}

TYPED_TEST(CLMulTester, test_Random2) {
    this->TestPrepare({{1, 3, 3, 3}, {1, 3, 3, 3}}, {1, 3, 3, 3}).TestRun();
}

TYPED_TEST(CLMulTester, CoverEachBranch_Pass) {
    float input_data_1[] = {-0.5, 0.2, 0.3, -0.1, -0.4, 0.9};
    float input_data_2[] = {0.8, 0.7, 0.0, -1.5, -0.6, 2.5};
    float out_data[] = {-0.4, 0.14, 0.0, 0.15, 0.24, 2.25};
    this->TestPrepare({{1, 2, 3, 1}, {1, 2, 3, 1}}, {1, 2, 3, 1}).TestRun({input_data_1, input_data_2}, out_data);
}

TYPED_TEST(CLMulTester, EachAxisLargerOne) {

    float input_data_1[] = {-0.1, -0.2, 0.3, 0.0, -0.4, 0.9, 0.9, 1.0, -0.1, -0.2, 0.3, 0.0, -0.4, 0.9, 0.9, 1.0};
    float input_data_2[] = {0.1, -0.7, 0.0, 1.5, 3.1, 10.8, -0.9, 1.8, -0.1, -0.2, 0.6, 0.1, -0.04, 0.9, 0.9, 1.0};
    float out_data[] = {-0.01, 0.14, 0.0, 0.0, -1.24, 9.72, -0.81, 1.8, 0.01, 0.04, 0.18, 0.0, 0.016, 0.81, 0.81, 1.0};
    this->TestPrepare({{2, 2, 2, 2}, {2, 2, 2, 2}}, {2, 2, 2, 2}).TestRun({input_data_1, input_data_2}, out_data);
}

TYPED_TEST(CLMulTester, DifferentDimSize) {
    // Dim4 dimIn1 = {1, 2, 3, 3};
    // Dim4 dimIn2 = {1, 2, 1, 1};
    // Dim4 dimOut = {1, 2, 3, 3};
    float input_data_1[] = {
            0, 1, 2,
            3, 4, 5,
            6, 7, 8,
            9, 10, 11,
            12, 13, 14,
            15, 16, 17
    };
    float input_data_2[] = {
            1,
            2
    };
    float out_data[] = {
            0, 1, 2,
            3, 4, 5,
            6, 7, 8,
            18, 20, 22,
            24, 26, 28,
            30, 32, 34
    };
    this->TestPrepare({{1, 2, 3, 3}, {1, 2, 1, 1}}, {1, 2, 3, 3}).TestRun({input_data_1, input_data_2}, out_data);
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
