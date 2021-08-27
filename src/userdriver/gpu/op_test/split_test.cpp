#include <gtest/gtest.h>
#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLSplit.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class SplitTester : public ::testing::Test {
public:
    SplitTester() {
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

    SplitTester &TestPrepare(const Dim4 &input_dim,
                             const Dim4 &output_dim,
                             const int32_t &axis,
                             const int32_t &num_outputs) {
        input_dim_ = input_dim;
        output_dim_ = output_dim;
        axis_ = axis;
        num_outputs_ = num_outputs;
        input_size_ = GetDimSize(input_dim_);
        output_size_ = GetDimSize(output_dim_);

        return *this;
    }

    void TestRun(float *input_data, const std::vector<float *> expect_out) {
        auto input_tensor = std::make_shared<CLTensor>(runtime_, precision_, input_data, input_dim_);
        inputs_.push_back(input_tensor);

        for (int i = 0; i != num_outputs_; ++i) {
            std::shared_ptr<ITensor> tmp_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, output_dim_);
            outputs_.push_back(tmp_tensor);
        }

        parameters_ = std::make_shared<SplitParameters>();
        parameters_->axis = axis_;
        parameters_->num_outputs = num_outputs_;
        parameters_->androidNN = false;

        CLSplit split(runtime_, precision_);
        EXPECT_EQ(split.initialize({input_tensor}, outputs_, parameters_), Status::SUCCESS);
        EXPECT_EQ(split.execute(), Status::SUCCESS);
        EXPECT_EQ(split.release(), Status::SUCCESS);

        //  compare result
        for (int i = 0; i != num_outputs_; ++i) {
            auto output_ptr = make_shared_array<float>(output_size_);
            auto tmp_tensor = std::static_pointer_cast<CLTensor>(outputs_.at(i));
            tmp_tensor->readData(output_ptr.get());
            Compare(expect_out[i], output_ptr.get(), output_size_, error_threshold_);
        }
    }

private:
    std::shared_ptr<CLRuntime> runtime_;
    std::vector<std::shared_ptr<ITensor>> inputs_;
    std::vector<std::shared_ptr<ITensor>> outputs_;
    std::shared_ptr<SplitParameters> parameters_;
    PrecisionType precision_;
    DataType data_type_;
    float error_threshold_;

    Dim4 input_dim_;
    Dim4 output_dim_;
    int32_t axis_;
    int32_t num_outputs_;
    size_t input_size_;
    size_t output_size_;
};

TYPED_TEST_CASE(SplitTester, TestFP32AndFP16Type);
TYPED_TEST(SplitTester, Test1DInput2) {
    float input[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float output1[] = {1.0, 2.0};
    float output2[] = {3.0, 4.0};
    float output3[] = {5.0, 6.0};
    std::vector<float *> output;
    output.push_back(output1);
    output.push_back(output2);
    output.push_back(output3);

    this->TestPrepare({6, 1, 1, 1}, {2, 1, 1, 1}, 0, 3).TestRun(input, output);
}

TYPED_TEST(SplitTester, Test2DInput1) {
    float input[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float output1[] = {1.0, 2.0, 3.0};
    float output2[] = {4.0, 5.0, 6.0};
    std::vector<float *> output;
    output.push_back(output1);
    output.push_back(output2);

    this->TestPrepare({2, 3, 1, 1}, {1, 3, 1, 1}, 0, 2).TestRun(input, output);
}

TYPED_TEST(SplitTester, Test2DInput3) {
    float input[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    float output1[] = {1.0, 4.0};
    float output2[] = {2.0, 5.0};
    float output3[] = {3.0, 6.0};
    std::vector<float *> output;
    output.push_back(output1);
    output.push_back(output2);
    output.push_back(output3);

    this->TestPrepare({2, 3, 1, 1}, {2, 1, 1, 1}, 1, 3).TestRun(input, output);
}

TYPED_TEST(SplitTester, Test3DInput1) {
    float input[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    float output1[] = {1.0, 2.0, 5.0, 6.0};
    float output2[] = {3.0, 4.0, 7.0, 8.0};
    std::vector<float *> output;
    output.push_back(output1);
    output.push_back(output2);

    this->TestPrepare({2, 2, 2, 1}, {2, 1, 2, 1}, 1, 2).TestRun(input, output);
}

TYPED_TEST(SplitTester, Test3DInput3) {
    float input[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    float output1[] = {1.0, 2.0, 5.0, 6.0};
    float output2[] = {3.0, 4.0, 7.0, 8.0};
    std::vector<float *> output;
    output.push_back(output1);
    output.push_back(output2);

    this->TestPrepare({2, 2, 2, 1}, {2, 1, 2, 1}, -2, 2).TestRun(input, output);
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
