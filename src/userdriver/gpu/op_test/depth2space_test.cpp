#include <gtest/gtest.h>

#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"
#include "userdriver/gpu/operators/CLDepth2Space.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class Depth2SpaceTester : public ::testing::Test {
public:
    Depth2SpaceTester() {
        runtime_ = std::shared_ptr<CLRuntime>(new CLRuntime());
        runtime_->initialize(0);
        runtime_->initializeQueue();
        precision_ = PRECISION::precision;
        data_type_ = DataType::FLOAT;
    }
    Depth2SpaceTester &TestPrepare(const Dim4 &input_dims, const int &block_size) {
        this->block_size_ = block_size;
        input_dims_ = input_dims;
        output_dims_ = {
            input_dims_.n, input_dims_.c / block_size / block_size, input_dims_.h * block_size, input_dims_.w * block_size};
        return *this;
    }
    inline Depth2SpaceTester &SetThreshold(float threshold) {
        error_threshold_ = threshold;
        return *this;
    }

    void TestRun(float *input_data, const float *inference_output_data) {
        in_size = GetDimSize(input_dims_);
        out_size = GetDimSize(output_dims_);

        auto input_tensor = std::make_shared<CLTensor>(runtime_, precision_, input_data, input_dims_);
        auto output_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, output_dims_);
        parameter_ = std::make_shared<Depth2SpaceParameters>();
        parameter_->block_size = block_size_;
        parameter_->androidNN = false;
        parameter_->isNCHW = true;

        CLDepth2Space cl_depth2space(runtime_, precision_);
        EXPECT_EQ(cl_depth2space.initialize({input_tensor}, {output_tensor}, parameter_), Status::SUCCESS);
        EXPECT_EQ(cl_depth2space.execute(), Status::SUCCESS);
        EXPECT_EQ(cl_depth2space.release(), Status::SUCCESS);

        std::shared_ptr<float> output = make_shared_array<float>(out_size);
        output_tensor->readData(output.get());
        Compare(output.get(), inference_output_data, out_size, error_threshold_);
    }

private:
    size_t in_size = 0;
    size_t out_size = 0;
    float error_threshold_;
    int block_size_ = 0;
    Dim4 input_dims_;
    Dim4 output_dims_;
    PrecisionType precision_;
    DataType data_type_;
    std::shared_ptr<CLRuntime> runtime_;
    std::shared_ptr<Depth2SpaceParameters> parameter_;
};

TYPED_TEST_CASE(Depth2SpaceTester, TestFP32AndFP16Type);
TYPED_TEST(Depth2SpaceTester, Normal) {
    float inference_output_data[] = {0,  2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34,
                                     36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70,
                                     1,  3,  5,  7,  9,  11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35,
                                     37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69, 71};
    float input_data[] = {0,  4,  8,  24, 28, 32, 48, 52, 56, 1,  5,  9,  25, 29, 33, 49, 53, 57, 2,  6,  10, 26, 30, 34,
                          50, 54, 58, 3,  7,  11, 27, 31, 35, 51, 55, 59, 12, 16, 20, 36, 40, 44, 60, 64, 68, 13, 17, 21,
                          37, 41, 45, 61, 65, 69, 14, 18, 22, 38, 42, 46, 62, 66, 70, 15, 19, 23, 39, 43, 47, 63, 67, 71};
    this->SetThreshold(1e-2).TestPrepare({1, 8, 3, 3}, 2).TestRun(input_data, inference_output_data);
}

TYPED_TEST(Depth2SpaceTester, BatchEqualsOne) {
    float inference_output_data[] = {1, 2, 3, 4};
    float input_data[] = {1, 2, 3, 4};
    this->SetThreshold(1e-2).TestPrepare({1, 4, 1, 1}, 2).TestRun(input_data, inference_output_data);
}

TYPED_TEST(Depth2SpaceTester, BatchEqualsThree) {
    float inference_output_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    float input_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    this->SetThreshold(1e-2).TestPrepare({3, 4, 1, 1}, 2).TestRun(input_data, inference_output_data);
}

TYPED_TEST(Depth2SpaceTester, BlockEqualsThree) {
    float input_data[] = {0,  3,  18, 21, 1,  4,  19, 22, 2,  5,  20, 23, 6,  9,  24, 27, 7,  10,
                          25, 28, 8,  11, 26, 29, 12, 15, 30, 33, 13, 16, 31, 34, 14, 17, 32, 35};
    float inference_output_data[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17,
                                     18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
    this->SetThreshold(1e-2).TestPrepare({1, 9, 2, 2}, 3).TestRun(input_data, inference_output_data);
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
