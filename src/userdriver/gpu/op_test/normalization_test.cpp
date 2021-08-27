#include <gtest/gtest.h>
#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/operators/CLNormalization.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"

namespace enn {
namespace ud {
namespace gpu {

class CLNormalizationTester {
  public:
    CLNormalizationTester(const PrecisionType &precision, float &max) : precision_(precision) {
        switch (precision_) {
        case PrecisionType::FP32:
            data_type_ = DataType::FLOAT;
            error_threshold_ = 1e-5;
            break;
        case PrecisionType::FP16:
            data_type_ = DataType::FLOAT;
            error_threshold_ = max / 500 > 0.01 ? max / 500 : 0.01;
            break;
        default: break;
        }
    }

    CLNormalizationTester &InputDims(const Dim4 &input_dims, const uint8_t &bgr_transpose) {
        input_dims_ = input_dims;
        output_dims_ = input_dims;
        mean_dims_ = {1, 1, 1, input_dims_.c};
        scale_dims_ = {1, 1, 1, input_dims_.c};
        parameters_.reset(new NormalizationrParameters);
        parameters_->use_FP32_input_for_fp16 = false;

        runtime_ = std::shared_ptr<CLRuntime>(new CLRuntime());
        runtime_->initialize(0);
        runtime_->initializeQueue();
        return *this;
    }

    template <typename T> void TestRun(T *input_data, float *scale, float *mean, const float *reference_output_data) {
        size_t output_size = GetDimSize(output_dims_);

        std::shared_ptr<ITensor> input_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_map_[typeid(T).name()], input_dims_);
        inputs_.push_back(input_tensor);
        std::shared_ptr<ITensor> mean_tensor = std::make_shared<CLTensor>(runtime_, precision_, mean, mean_dims_);
        std::shared_ptr<ITensor> scale_tensor = std::make_shared<CLTensor>(runtime_, precision_, scale, scale_dims_);

        inputs_.push_back(mean_tensor);
        inputs_.push_back(scale_tensor);

        std::shared_ptr<ITensor> output_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, output_dims_);
        outputs_.push_back(output_tensor);

        CLNormalization process_clnorm(runtime_, precision_);
        EXPECT_EQ(process_clnorm.initialize(inputs_, outputs_, parameters_), Status::SUCCESS);
        EXPECT_EQ(input_tensor->writeData(input_data), Status::SUCCESS);
        EXPECT_EQ(process_clnorm.execute(), Status::SUCCESS);
        EXPECT_EQ(process_clnorm.release(), Status::SUCCESS);

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
    Dim4 mean_dims_;
    Dim4 scale_dims_;
    Dim4 output_dims_;
    std::vector<std::shared_ptr<ITensor>> inputs_;
    std::vector<std::shared_ptr<ITensor>> outputs_;
    std::shared_ptr<NormalizationrParameters> parameters_;
};

template <typename T>
void TEST_COMMON(const PrecisionType &precision,
                 T type_dummy,
                 Dim4 &input_dims,
                 int8_t bgr_transpose,
                 Status status = Status::SUCCESS) {
    size_t array_size = GetDimSize(input_dims);
    std::shared_ptr<T> in;
    in.reset(new T[array_size], std::default_delete<T[]>());
    GenerateRandom<T>(in.get(), array_size, 0, 5);

    std::shared_ptr<float> scale;
    scale.reset(new float[input_dims.c], std::default_delete<float[]>());
    GenerateRandom<float>(scale.get(), input_dims.c, 0, 3);

    std::shared_ptr<float> mean;
    mean.reset(new float[input_dims.c], std::default_delete<float[]>());
    GenerateRandom<float>(mean.get(), input_dims.c, 0, 3);

    float *refer_output = new float[array_size];
    float max = -65532;
    NormalizationRef(
        in.get(), refer_output, input_dims.c, input_dims.h, input_dims.w, mean.get(), scale.get(), bgr_transpose, max);
    CLNormalizationTester(precision, max)
        .InputDims(input_dims, bgr_transpose)
        .TestRun(in.get(), scale.get(), mean.get(), refer_output);

    delete[] refer_output;
}

TEST(CLNormalizationTester, BigHeightBGRZero_FP32) {
    float type_dummy = 0;
    Dim4 input_dims = {1, 3, 299, 299};
    TEST_COMMON(PrecisionType::FP32, type_dummy, input_dims, 0);
}

TEST(CLNormalizationTester, SingleChannelBGR_FP32) {
    float type_dummy = 0;
    Dim4 input_dims = {1, 1, 7, 7};
    TEST_COMMON(PrecisionType::FP32, type_dummy, input_dims, 0);
}

TEST(CLNormalizationTester, SingleChannelBGRZero_FP32) {
    float type_dummy = 0;
    Dim4 input_dims = {1, 1, 7, 7};
    TEST_COMMON(PrecisionType::FP32, type_dummy, input_dims, 0);
}

TEST(CLNormalizationTester, BigHeightBGRZero_FP16) {
    float type_dummy = 0;
    Dim4 input_dims = {1, 3, 299, 299};
    TEST_COMMON(PrecisionType::FP16, type_dummy, input_dims, 0);
}

TEST(CLNormalizationTester, SingleChannelBGR_FP16) {
    float type_dummy = 0;
    Dim4 input_dims = {1, 1, 7, 7};
    TEST_COMMON(PrecisionType::FP16, type_dummy, input_dims, 0);
}

TEST(CLNormalizationTester, SingleChannelBGRZero_FP16) {
    float type_dummy = 0;
    Dim4 input_dims = {1, 1, 7, 7};
    TEST_COMMON(PrecisionType::FP16, type_dummy, input_dims, 0);
}

TEST(CLNormalizationTester, SmallHeightBG_FP32) {
    uint8_t type_dummy = 0;
    Dim4 input_dims = {1, 3, 32, 32};
    TEST_COMMON(PrecisionType::FP32, type_dummy, input_dims, 0);
}
TEST(CLNormalizationTester, SmallHeightBGR_16) {
    uint8_t type_dummy = 0;
    Dim4 input_dims = {1, 3, 32, 32};
    TEST_COMMON(PrecisionType::FP16, type_dummy, input_dims, 0);
}

TEST(CLNormalizationTester, SmallHeightBGRZero_32) {
    uint8_t type_dummy = 0;
    Dim4 input_dims = {1, 3, 32, 32};
    TEST_COMMON(PrecisionType::FP32, type_dummy, input_dims, 0);
}
TEST(CLNormalizationTester, SmallHeightBGRZero_16) {
    uint8_t type_dummy = 0;
    Dim4 input_dims = {1, 3, 32, 32};
    TEST_COMMON(PrecisionType::FP16, type_dummy, input_dims, 0);
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
