#include <gtest/gtest.h>
#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLPad.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class CLPadTester : public ::testing::Test {
  public:
    CLPadTester() {
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

    CLPadTester &TestPrepare(const Dim4 &input_dim, const Dim4 &output_dim, const float &pad_value) {
        input_dim_ = input_dim;
        output_dim_ = output_dim;
        input_size_ = GetDimSize(input_dim_);
        output_size_ = GetDimSize(output_dim_);
        parameters_ = std::make_shared<PadParameters>();
        parameters_->pad_value = pad_value;

        return *this;
    }

    void TestRun(float *input_data, std::vector<int32_t> &padding, float *golden_data) {
        Dim4 padding_dim = {static_cast<uint32_t>(padding.size()), 1, 1, 1};
        Dim4 pad_value_dim = {1, 1, 1, 1};
        auto input_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, input_dim_);
        auto padding_tensor = std::make_shared<CLTensor>(runtime_, precision_, padding.data(), padding_dim);
        auto pad_value_tensor = std::make_shared<CLTensor>(runtime_, precision_, &parameters_->pad_value, pad_value_dim);
        auto output_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, output_dim_);

        CLPad pad(runtime_, precision_);
        EXPECT_EQ(pad.initialize({input_tensor, padding_tensor, pad_value_tensor}, {output_tensor}, parameters_),
                  Status::SUCCESS);
        EXPECT_EQ(input_tensor->writeData(input_data), Status::SUCCESS);
        EXPECT_EQ(pad.execute(), Status::SUCCESS);
        EXPECT_EQ(pad.release(), Status::SUCCESS);

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
    std::shared_ptr<PadParameters> parameters_;
};

TYPED_TEST_CASE(CLPadTester, TestFP32AndFP16Type);
TYPED_TEST(CLPadTester, SimpleConstTest) {
    float input_data[] = {1, 2, 3, 4};
    std::vector<int32_t> paddings = {0, 0, 0, 0, 1, 1, 1, 1};
    Dim4 input_dims = {1, 1, 2, 2};  // n,c,h,w
    Dim4 output_dims = {1, 1, 4, 4};
    float pad_value = 0;
    float expect_out[] = {0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0, 0, 0};
    this->TestPrepare(input_dims, output_dims, pad_value).TestRun(input_data, paddings, expect_out);
}

TYPED_TEST(CLPadTester, SimpleConstFloat32ValuedTest) {
    float input_data[] = {1, 2, 3, 4};
    std::vector<int32_t> paddings = {0, 0, 0, 0, 1, 1, 1, 1};
    Dim4 input_dims = {1, 1, 2, 2};  // n,c,h,w
    Dim4 output_dims = {1, 1, 4, 4};
    float pad_value = 5;
    float expect_out[] = {5, 5, 5, 5, 5, 1, 2, 5, 5, 3, 4, 5, 5, 5, 5, 5};
    this->TestPrepare(input_dims, output_dims, pad_value).TestRun(input_data, paddings, expect_out);
}

TYPED_TEST(CLPadTester, Simple4DConstFloat32ValuedTest) {
    float input_data[] = {3, 3};
    std::vector<int32_t> paddings = {0, 0, 1, 1, 0, 0, 0, 0};
    Dim4 input_dims = {1, 1, 1, 2};  // n,c,h,w
    Dim4 output_dims = {1, 3, 1, 2};
    float pad_value = 5;
    float expect_out[] = {5, 5, 3, 3, 5, 5};
    this->TestPrepare(input_dims, output_dims, pad_value).TestRun(input_data, paddings, expect_out);
}
TYPED_TEST(CLPadTester, AdvancedConstTest) {
    float input_data[] = {1, 2, 3, 4, 5, 6};
    std::vector<int32_t> paddings = {0, 0, 1, 1, 0, 0, 1, 1};
    Dim4 input_dims = {1, 1, 2, 3};  // n,c,h,w
    Dim4 output_dims = {1, 3, 2, 5};
    float pad_value = 0;
    float expect_out[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 4, 5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    this->TestPrepare(input_dims, output_dims, pad_value).TestRun(input_data, paddings, expect_out);
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
