#include <gtest/gtest.h>

#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"
#include "userdriver/gpu/operators/CLMean.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename T> class MeanTester : public ::testing::Test {
public:
    MeanTester() {
        runtime_ = std::shared_ptr<CLRuntime>(new CLRuntime());
        runtime_->initialize(0);
        runtime_->initializeQueue();
        precision_ = T::precision;
        data_type_ = DataType::FLOAT;
        parameters_ = std::make_shared<MeanParameters>();
    }

    inline MeanTester &SetThreshold() {
        if (T::precision == PrecisionType ::FP32) {
            error_threshold_ = 1e-5;
        } else if (T::precision == PrecisionType ::FP16) {
            error_threshold_ = 1e-2;
        }
        return *this;
    }

    inline MeanTester &InputDims(const Dim4 &input_dims) {
        input_dims_ = input_dims;
        return *this;
    }
    inline MeanTester &OutputDims(const Dim4 &output_dims) {
        output_dims_ = output_dims;
        return *this;
    }

    inline MeanTester &KeepDims(const bool &keep_dims) {
        keep_dims_ = keep_dims;
        return *this;
    }

    void TestRun(float *input_data, int32_t *axis, const float *reference_output_data, Dim4 axis_dim) {
        size_t input_size = GetDimSize(input_dims_);
        size_t output_size = GetDimSize(output_dims_);
        auto in_ptr = make_shared_array<float>(input_size);
        memcpy(in_ptr.get(), input_data, input_size * sizeof(float));

        CLMean cl_mean(runtime_, precision_);
        std::shared_ptr<CLTensor> input_tensor_ =
            std::make_shared<CLTensor>(runtime_, precision_, in_ptr.get(), input_dims_);
        std::shared_ptr<CLTensor> axis_tensor_ = std::make_shared<CLTensor>(runtime_, precision_, axis, axis_dim);
        std::shared_ptr<CLTensor> output_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, output_dims_);
        parameters_->keep_dims = keep_dims_;
        cl_mean.initialize({input_tensor_, axis_tensor_}, {output_tensor}, parameters_);
        cl_mean.execute();
        cl_mean.release();
        auto out_ptr = make_shared_array<float>(output_size);
        output_tensor->readData(out_ptr.get());
        Compare(out_ptr.get(), reference_output_data, output_size, error_threshold_);
    }

private:
    std::shared_ptr<CLRuntime> runtime_;
    float error_threshold_;
    PrecisionType precision_;
    DataType data_type_;

    Dim4 input_dims_;
    Dim4 output_dims_;
    bool keep_dims_;
    std::shared_ptr<MeanParameters> parameters_;
};

TYPED_TEST_CASE(MeanTester, TestFP32AndFP16Type);
TYPED_TEST(MeanTester, Basic) {
    float input_data[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                          13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    float expect_output[] = {12, 13};
    int32_t axis[] = {1, 0, -3, -3};
    Dim4 axis_dim = {1, 1, (int)(sizeof(axis) / sizeof(int32_t)), 1};
    this->SetThreshold()
        .InputDims({4, 3, 2, 1})
        .OutputDims({2, 1, 1, 1})
        .KeepDims(false)
        .TestRun(input_data, axis, expect_output, axis_dim);
}

TYPED_TEST(MeanTester, KeepDims) {
    float input_data[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                          13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    float expect_output[] = {10.5, 12.5, 14.5};
    int32_t axis[] = {0, 2};
    Dim4 axis_dim = {1, 1, 1, (int)(sizeof(axis) / sizeof(int32_t))};
    this->SetThreshold()
        .InputDims({4, 3, 2, 1})
        .OutputDims({1, 3, 1, 1})
        .KeepDims(true)
        .TestRun(input_data, axis, expect_output, axis_dim);
}

TYPED_TEST(MeanTester, Scalar) {
    float input_data[] = {3.27};
    float expect_output[] = {3.27};
    int32_t axis[] = {0};
    Dim4 axis_dim = {1, 1, 1, (int)(sizeof(axis) / sizeof(int32_t))};
    this->SetThreshold()
        .InputDims({1, 1, 1, 1})
        .OutputDims({1, 1, 1, 1})
        .KeepDims(true)
        .TestRun(input_data, axis, expect_output, axis_dim);
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
