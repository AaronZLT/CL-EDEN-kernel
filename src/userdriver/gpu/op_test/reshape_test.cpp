#include <gtest/gtest.h>
#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLReshape.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class CLReshapeTester : public ::testing::Test {
  public:
    CLReshapeTester() {
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

    CLReshapeTester &TestPrepare(const Dim4 &input_dim) {
        input_dim_ = input_dim;
        output_dim_ = input_dim;
        input_size_ = GetDimSize(input_dim_);
        output_size_ = GetDimSize(output_dim_);
        parameters_ = std::make_shared<ReshapeParameters>();
        parameters_->androidNN = true;
        parameters_->compute_type = ComputeType::Caffe;
        return *this;
    }

    void TestRun(std::vector<int32_t> &shape_data) {
        shape_size_ = shape_data.size();
        shape_dim_ = {static_cast<uint32_t>(shape_size_), 1, 1, 1};
        std::shared_ptr<float> input = make_shared_array<float>(input_size_);
        std::shared_ptr<float> output = make_shared_array<float>(output_size_);
        GenerateRandom<float>(input.get(), input_size_, 0, 1);
        memset(output.get(), 0, output_size_ * sizeof(float));

        ReshapeGuard reshape_guard;
        reshape_guard.GuardPrepare(input_size_);
        reshape_guard.GuardRun(input.get(), output.get());
        doRun(input.get(), shape_data.data(), output.get());
    }

  private:
    void doRun(float *input_data, int32_t *shape_data, float *golden_data) {
        auto input_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, input_dim_);
        auto shape_tensor = std::make_shared<CLTensor>(runtime_, precision_, shape_data, shape_dim_);
        auto output_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, output_dim_);
        CLReshape reshape(runtime_, precision_);
        EXPECT_EQ(reshape.initialize({input_tensor, shape_tensor}, {output_tensor}, parameters_), Status::SUCCESS);
        EXPECT_EQ(input_tensor->writeData(input_data), Status::SUCCESS);
        EXPECT_EQ(reshape.execute(), Status::SUCCESS);
        EXPECT_EQ(reshape.release(), Status::SUCCESS);

        auto output_ptr = make_shared_array<float>(output_size_);
        EXPECT_EQ(output_tensor->readData(output_ptr.get()), Status::SUCCESS);

        Compare(output_ptr.get(), golden_data, output_size_, error_threshold_);
    }

  private:
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;
    DataType data_type_;
    float error_threshold_;

    Dim4 input_dim_;
    Dim4 shape_dim_;
    Dim4 output_dim_;
    size_t input_size_;
    size_t shape_size_;
    size_t output_size_;
    std::shared_ptr<ReshapeParameters> parameters_;
};

TYPED_TEST_CASE(CLReshapeTester, TestFP32AndFP16Type);

TYPED_TEST(CLReshapeTester, test_shape_1) {
    std::vector<int32_t> shape_data = {3, 10, -1};
    this->TestPrepare({2, 3, 6, 5}).TestRun(shape_data);
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
