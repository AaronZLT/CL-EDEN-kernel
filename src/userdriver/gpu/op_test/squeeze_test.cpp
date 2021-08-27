#include <gtest/gtest.h>
#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLSqueeze.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class CLSqueezeTester : public ::testing::Test {
  public:
    CLSqueezeTester() {
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

    CLSqueezeTester &
    TestPrepare(const Dim4 &input_dim, const Dim4 &output_dim, const std::vector<int32_t> &squeeze_dims = {}) {
        input_dim_ = input_dim;
        squeeze_dim_ = {static_cast<uint32_t>(squeeze_dims.size()), 1, 1, 1};
        output_dim_.n = output_dim.n > 0 ? output_dim.n : 1;
        output_dim_.c = output_dim.c > 0 ? output_dim.c : 1;
        output_dim_.h = output_dim.h > 0 ? output_dim.h : 1;
        output_dim_.w = output_dim.w > 0 ? output_dim.w : 1;
        input_size_ = GetDimSize(input_dim_);
        output_size_ = GetDimSize(output_dim_);
        parameters_ = std::make_shared<SqueezeParameters>();
        parameters_->squeeze_dims = squeeze_dims;
        if (output_size_ == 0) {
            parameters_->androidNN = true;
            output_size_ = input_size_;
        }

        return *this;
    }

    void TestRun(float *input_data, float *golden_data) {
        auto input_tensor = std::make_shared<CLTensor>(runtime_, precision_, input_data, input_dim_);
        std::shared_ptr<CLTensor> squeeze_tensor;
        if (parameters_->squeeze_dims.size() > 0) {
            squeeze_tensor =
                std::make_shared<CLTensor>(runtime_, precision_, parameters_->squeeze_dims.data(), squeeze_dim_);
        } else {
            squeeze_tensor = std::make_shared<CLTensor>(runtime_, precision_, DataType::INT32, squeeze_dim_);
        }
        auto output_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, output_dim_);

        CLSqueeze squeeze(runtime_, precision_);
        EXPECT_EQ(squeeze.initialize({input_tensor, squeeze_tensor}, {output_tensor}, parameters_), Status::SUCCESS);
        EXPECT_EQ(squeeze.execute(), Status::SUCCESS);
        EXPECT_EQ(squeeze.release(), Status::SUCCESS);

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
    Dim4 squeeze_dim_;
    Dim4 output_dim_;
    size_t input_size_;
    size_t output_size_;
    std::shared_ptr<SqueezeParameters> parameters_;
};

TYPED_TEST_CASE(CLSqueezeTester, TestFP32AndFP16Type);

TYPED_TEST(CLSqueezeTester, squeeze_all) {
    float input[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                     13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    float expect_out[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                          13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};

    this->TestPrepare({1, 24, 1, 1}, {24}).TestRun(input, expect_out);
}

TYPED_TEST(CLSqueezeTester, squeeze_selected_axis) {
    float input[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                     13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    float expect_out[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                          13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    this->TestPrepare({1, 24, 1, 1}, {1, 24, 1}, {2}).TestRun(input, expect_out);
}

TYPED_TEST(CLSqueezeTester, squeeze_negative_axis) {
    float input[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                     13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    float expect_out[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                          13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    this->TestPrepare({1, 24, 1, 1}, {24, 1}, {-1, 0}).TestRun(input, expect_out);
}

TYPED_TEST(CLSqueezeTester, squeeze_all_another) {
    float input[] = {3.85};
    float expect_out[] = {3.85};
    this->TestPrepare({1, 1, 1, 1}, {0, 0, 0, 0}, {}).TestRun(input, expect_out);
}

TYPED_TEST(CLSqueezeTester, squeeze_selected_axis_another) {
    float input[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                     13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    float expect_out[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                          13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    this->TestPrepare({1, 24, 1, 1}, {0, 0, 0, 0}, {2}).TestRun(input, expect_out);
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
