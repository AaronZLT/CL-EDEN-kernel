#include <gtest/gtest.h>
#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLUnpack.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class CLUnpackTester : public ::testing::Test {
public:
    CLUnpackTester() {
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

    CLUnpackTester &TestPrepare(const NDims &input_dim, const NDims &output_dim, int32_t axis, int32_t num) {
        input_dim_ = input_dim;
        output_dim_ = output_dim;
        input_size_ = getTensorSizeFromDims(input_dim_);
        output_size_ = getTensorSizeFromDims(output_dim_);
        parameters_ = std::make_shared<UnpackParameters>();
        parameters_->axis = axis;
        parameters_->num = num;

        return *this;
    }

    void TestRun(float *input_data, std::vector<float *> &golden_data) {
        EXPECT_EQ(parameters_->num, static_cast<int32_t>(golden_data.size()));

        auto input_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, input_dim_);
        std::vector<std::shared_ptr<ITensor>> output_tensors;
        for (int32_t i = 0; i < parameters_->num; ++i) {
            auto output_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, output_dim_);
            output_tensors.push_back(output_tensor);
        }

        CLUnpack unpack(runtime_, precision_);
        EXPECT_EQ(unpack.initialize({input_tensor}, output_tensors, parameters_), Status::SUCCESS);
        EXPECT_EQ(input_tensor->writeData(input_data), Status::SUCCESS);
        EXPECT_EQ(unpack.execute(), Status::SUCCESS);
        EXPECT_EQ(unpack.release(), Status::SUCCESS);

        for (int32_t i = 0; i < parameters_->num; ++i) {
            auto output_ptr = make_shared_array<float>(output_size_);
            output_tensors[i]->readData(output_ptr.get());

            Compare(output_ptr.get(), golden_data[i], output_size_, error_threshold_);
        }
    }

private:
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;
    DataType data_type_;
    float error_threshold_;

    NDims input_dim_;
    NDims output_dim_;
    size_t input_size_;
    size_t output_size_;
    std::shared_ptr<UnpackParameters> parameters_;
};

TYPED_TEST_CASE(CLUnpackTester, TestFP32AndFP16Type);
TYPED_TEST(CLUnpackTester, ThreeOutputs) {
    float input[6] = {1, 2, 3, 4, 5, 6};
    float output1[2] = {1, 2};
    float output2[2] = {3, 4};
    float output3[2] = {5, 6};
    std::vector<float *> golden = {output1, output2, output3};
    this->TestPrepare({3, 2}, {2}, 0, 3).TestRun(input, golden);
}

TYPED_TEST(CLUnpackTester, ThreeOutputsAxisOne) {
    float input[6] = {1, 2, 3, 4, 5, 6};
    float output1[3] = {1, 3, 5};
    float output2[3] = {2, 4, 6};
    std::vector<float *> golden = {output1, output2};
    this->TestPrepare({3, 2}, {3}, 1, 2).TestRun(input, golden);
}

TYPED_TEST(CLUnpackTester, ThreeOutputsNegativeAxisOne) {
    float input[6] = {1, 2, 3, 4, 5, 6};
    float output1[3] = {1, 3, 5};
    float output2[3] = {2, 4, 6};
    std::vector<float *> golden = {output1, output2};
    this->TestPrepare({3, 2}, {3}, -1, 2).TestRun(input, golden);
}

TYPED_TEST(CLUnpackTester, OneOutput) {
    float input[6] = {1, 2, 3, 4, 5, 6};
    float output1[6] = {1, 2, 3, 4, 5, 6};
    std::vector<float *> golden = {output1};
    this->TestPrepare({1, 6}, {6}, 0, 1).TestRun(input, golden);
}

TYPED_TEST(CLUnpackTester, ThreeDimensionsOutputs) {
    float input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float output1[4] = {1, 3, 5, 7};
    float output2[4] = {2, 4, 6, 8};
    std::vector<float *> golden = {output1, output2};
    this->TestPrepare({2, 2, 2}, {2, 2}, 2, 2).TestRun(input, golden);
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
