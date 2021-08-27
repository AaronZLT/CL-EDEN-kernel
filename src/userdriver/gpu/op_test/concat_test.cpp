#include <gtest/gtest.h>
#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLConcat.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class CLConcatTester : public ::testing::Test {
  public:
    CLConcatTester() {
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

    CLConcatTester &TestPrepare(const std::vector<Dim4> &input_dims, const Dim4 &output_dim, const int32_t &axis) {
        input_dims_ = input_dims;
        output_dim_ = output_dim;
        for (auto input_dim : input_dims_) {
            input_sizes_.push_back(GetDimSize(input_dim));
        }
        output_size_ = GetDimSize(output_dim_);
        parameters_ = std::make_shared<ConcatParameters>();
        parameters_->axis = axis;
        return *this;
    }

    void TestRun(std::vector<float *> input_data = {}, float *golden_data = nullptr) {
        std::vector<std::shared_ptr<float>> inputs(input_sizes_.size());
        std::shared_ptr<float> output;
        if (golden_data == nullptr) {
            input_data.clear();
            for (size_t i = 0; i < input_sizes_.size(); ++i) {
                inputs[i] = make_shared_array<float>(input_sizes_[i]);
                GenerateRandom<float>(inputs[i].get(), input_sizes_[i], -2, 2);
                input_data.push_back(inputs[i].get());
            }
            output = make_shared_array<float>(output_size_);
            memset(output.get(), 0, output_size_ * sizeof(float));
            golden_data = output.get();

            ConcatGuard concat_guard;

            std::vector<std::vector<uint32_t>> input_shape(input_dims_.size());
            for (size_t i = 0; i < input_dims_.size(); ++i) {
                input_shape[i].push_back(input_dims_[i].n);
                input_shape[i].push_back(input_dims_[i].c);
                input_shape[i].push_back(input_dims_[i].h);
                input_shape[i].push_back(input_dims_[i].w);
            }

            concat_guard.GuardPrepare(input_shape, parameters_->axis);
            concat_guard.GuardRun(input_data, golden_data);
        }

        doRun(input_data, golden_data);
    }

  private:
    void doRun(std::vector<float *> input_data, float *golden_data) {
        const size_t size = input_data.size();
        std::vector<std::shared_ptr<ITensor>> input_tensors(size);
        for (size_t i = 0; i < size; ++i) {
            auto input_tensor = std::make_shared<CLTensor>(runtime_, precision_, input_data.at(i), input_dims_.at(i));
            input_tensors[i] = input_tensor;
        }
        auto output_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, output_dim_);

        CLConcat concat(runtime_, precision_);
        EXPECT_EQ(concat.initialize(input_tensors, {output_tensor}, parameters_), Status::SUCCESS);
        EXPECT_EQ(concat.execute(), Status::SUCCESS);
        EXPECT_EQ(concat.release(), Status::SUCCESS);

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
    std::shared_ptr<ConcatParameters> parameters_;
};

TYPED_TEST_CASE(CLConcatTester, TestFP32AndFP16Type);
TYPED_TEST(CLConcatTester, random_Axis0) {
    std::vector<Dim4> input_dims = {{2, 3, 4, 1}, {1, 3, 4, 1}, {3, 3, 4, 1}};
    Dim4 output_dim = {6, 3, 4, 1};
    int32_t axis = 0;
    this->TestPrepare(input_dims, output_dim, axis).TestRun();
}

TYPED_TEST(CLConcatTester, random_Axis1) {
    std::vector<Dim4> input_dims = {{2, 2, 3, 4}, {2, 1, 3, 4}};
    Dim4 output_dim = {2, 3, 3, 4};
    int32_t axis = 1;
    this->TestPrepare(input_dims, output_dim, axis).TestRun();
}

TYPED_TEST(CLConcatTester, random_Axis1_morethan6) {
    std::vector<Dim4> input_dims;
    input_dims.push_back({2, 1, 3, 4});
    input_dims.push_back({2, 2, 3, 4});
    input_dims.push_back({2, 3, 3, 4});
    input_dims.push_back({2, 4, 3, 4});
    input_dims.push_back({2, 5, 3, 4});
    input_dims.push_back({2, 6, 3, 4});
    input_dims.push_back({2, 7, 3, 4});
    input_dims.push_back({2, 8, 3, 4});
    Dim4 output_dim = {2, 36, 3, 4};
    int32_t axis = 1;
    this->TestPrepare(input_dims, output_dim, axis).TestRun();
}

TYPED_TEST(CLConcatTester, random_Axis2) {
    std::vector<Dim4> input_dims = {{2, 3, 2, 1}, {2, 3, 4, 1}};
    Dim4 output_dim = {2, 3, 6, 1};
    int32_t axis = 2;
    this->TestPrepare(input_dims, output_dim, axis).TestRun();
}

TYPED_TEST(CLConcatTester, random_Axis3) {
    std::vector<Dim4> input_dims = {{2, 3, 2, 1}, {2, 3, 2, 3}, {2, 3, 2, 4}, {2, 3, 2, 5}};
    Dim4 output_dim = {2, 3, 2, 13};
    int32_t axis = 3;
    this->TestPrepare(input_dims, output_dim, axis).TestRun();
}

TYPED_TEST(CLConcatTester, Axis0) {
    std::vector<Dim4> input_dims = {{1, 1, 2, 2}, {2, 1, 2, 2}, {3, 1, 2, 2}};
    Dim4 output_dim = {6, 1, 2, 2};
    int32_t axis = 0;

    float tensor1[] = {2.4, 3.3, 4.1, 5.2};
    float tensor2[] = {5.1, 2.6, 8.2, 5.9, -0.2, 0, 3.6, 3.5};
    float tensor3[] = {3.4, 0.1, 6.2, 5.2, 6.1, 0.5, -9.2, -7.4, -3.6, 4.6, 12.6, -9.3};
    std::vector<float *> input_data = {tensor1, tensor2, tensor3};
    float golden_data[] = {2.4, 3.3, 4.1, 5.2, 5.1, 2.6, 8.2,  5.9,  -0.2, 0,   3.6,  3.5,
                           3.4, 0.1, 6.2, 5.2, 6.1, 0.5, -9.2, -7.4, -3.6, 4.6, 12.6, -9.3};

    this->TestPrepare(input_dims, output_dim, axis).TestRun(input_data, golden_data);
}

TYPED_TEST(CLConcatTester, Axis1) {
    std::vector<Dim4> input_dims = {{1, 1, 2, 2}, {1, 2, 2, 2}, {1, 3, 2, 2}};
    Dim4 output_dim = {1, 6, 2, 2};
    int32_t axis = 1;

    float tensor1[] = {2.4, 3.3, 4.1, 5.2};
    float tensor2[] = {5.1, 2.6, 8.2, 5.9, -0.2, 0, 3.6, 3.5};
    float tensor3[] = {3.4, 0.1, 6.2, 5.2, 6.1, 0.5, -9.2, -7.4, -3.6, 4.6, 12.6, -9.3};
    std::vector<float *> input_data = {tensor1, tensor2, tensor3};
    float golden_data[] = {2.4, 3.3, 4.1, 5.2, 5.1, 2.6, 8.2,  5.9,  -0.2, 0,   3.6,  3.5,
                           3.4, 0.1, 6.2, 5.2, 6.1, 0.5, -9.2, -7.4, -3.6, 4.6, 12.6, -9.3};

    this->TestPrepare(input_dims, output_dim, axis).TestRun(input_data, golden_data);
}

TYPED_TEST(CLConcatTester, Axis2) {
    std::vector<Dim4> input_dims = {{1, 2, 1, 2}, {1, 2, 2, 2}, {1, 2, 3, 2}};
    Dim4 output_dim = {1, 2, 6, 2};
    int32_t axis = 2;

    float tensor1[] = {2.4, 3.3, 4.1, 5.2};
    float tensor2[] = {5.1, 2.6, 8.2, 5.9, -0.2, 0, 3.6, 3.5};
    float tensor3[] = {3.4, 0.1, 6.2, 5.2, 6.1, 0.5, -9.2, -7.4, -3.6, 4.6, 12.6, -9.3};
    std::vector<float *> input_data = {tensor1, tensor2, tensor3};
    float golden_data[] = {2.4, 3.3, 5.1,  2.6, 8.2, 5.9, 3.4,  0.1,  6.2,  5.2, 6.1,  0.5,
                           4.1, 5.2, -0.2, 0,   3.6, 3.5, -9.2, -7.4, -3.6, 4.6, 12.6, -9.3};

    this->TestPrepare(input_dims, output_dim, axis).TestRun(input_data, golden_data);
}

TYPED_TEST(CLConcatTester, Axis3) {
    std::vector<Dim4> input_dims = {{1, 2, 2, 1}, {1, 2, 2, 2}, {1, 2, 2, 3}};
    Dim4 output_dim = {1, 2, 2, 6};
    int32_t axis = 3;

    float tensor1[] = {2.4, 3.3, 4.1, 5.2};
    float tensor2[] = {5.1, 2.6, 8.2, 5.9, -0.2, 0, 3.6, 3.5};
    float tensor3[] = {3.4, 0.1, 6.2, 5.2, 6.1, 0.5, -9.2, -7.4, -3.6, 4.6, 12.6, -9.3};
    std::vector<float *> input_data = {tensor1, tensor2, tensor3};
    float golden_data[] = {2.4, 5.1,  2.6, 3.4,  0.1,  6.2,  3.3, 8.2, 5.9, 5.2, 6.1,  0.5,
                           4.1, -0.2, 0,   -9.2, -7.4, -3.6, 5.2, 3.6, 3.5, 4.6, 12.6, -9.3};

    this->TestPrepare(input_dims, output_dim, axis).TestRun(input_data, golden_data);
}

TYPED_TEST(CLConcatTester, element_num_non4) {
    std::vector<Dim4> input_dims = {{1, 2, 1, 1}, {1, 2, 3, 1}};
    Dim4 output_dim = {1, 2, 4, 1};
    int32_t axis = 2;

    float tensor1[] = {3.5, -9.1};
    float tensor2[] = {5.1, 2.6, 8.2, 5.9, -0.2, 0, 3.6, 3.5};
    std::vector<float *> input_data = {tensor1, tensor2};
    float golden_data[] = {3.5, 5.1, 2.6, 8.2, -9.1, 5.9, -0.2, 0};

    this->TestPrepare(input_dims, output_dim, axis).TestRun(input_data, golden_data);
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
