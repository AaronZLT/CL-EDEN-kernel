#include <gtest/gtest.h>
#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/operators/CLDeQuantization.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class DeQuantizationTester : public ::testing::Test {
  public:
    DeQuantizationTester() {
        precision_ = PRECISION::precision;
        switch (precision_) {
        case PrecisionType::FP32: error_threshold_ = 1e-5; break;
        case PrecisionType::FP16: error_threshold_ = 0.01; break;
        default: break;
        }

        runtime_ = std::shared_ptr<CLRuntime>(new CLRuntime());
        runtime_->initialize(0);
        runtime_->initializeQueue();
    }

    DeQuantizationTester &TestPrepare(const Dim4 &input_dims) {
        input_dims_ = input_dims;
        output_dims_ = input_dims;

        parameters_.reset(new DeQuantizationParameters);
        parameters_->per_channel_quant = false;
        parameters_->channel_dim = 0;
        return *this;
    }
    template <typename T>
    void TestRun(T *input_data, const float *reference_output_data, const float scale, const int32_t zero_point) {
        size_t output_size = GetDimSize(output_dims_);

        std::shared_ptr<ITensor> input_tensor = std::make_shared<CLTensor>(runtime_,
                                                                           precision_,
                                                                           input_data,
                                                                           input_dims_,
                                                                           DataOrder::NCHW,
                                                                           scale,
                                                                           zero_point,
                                                                           BufferType::DEDICATED,
                                                                           StorageType::BUFFER,
                                                                           0,
                                                                           UNDEFINED);
        inputs_.push_back(input_tensor);
        std::shared_ptr<ITensor> output_tensor =
            std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, output_dims_);
        outputs_.push_back(output_tensor);

        CLDeQuantization process_cl(runtime_, precision_);
        EXPECT_EQ(process_cl.initialize(inputs_, outputs_, parameters_), Status::SUCCESS);
        EXPECT_EQ(process_cl.execute(), Status::SUCCESS);
        EXPECT_EQ(process_cl.release(), Status::SUCCESS);

        auto out_ptr = make_shared_array<float>(output_size);
        output_tensor->readData(out_ptr.get());
        Compare(out_ptr.get(), reference_output_data, output_size, error_threshold_);
    }

  private:
    std::shared_ptr<CLRuntime> runtime_;
    float error_threshold_;
    Dim4 input_dims_;
    Dim4 output_dims_;
    PrecisionType precision_;
    DataType data_type_;

    std::vector<std::shared_ptr<ITensor>> inputs_;
    std::vector<std::shared_ptr<ITensor>> outputs_;
    std::shared_ptr<DeQuantizationParameters> parameters_;
};

TYPED_TEST_CASE(DeQuantizationTester, TestFP32AndFP16Type);
TYPED_TEST(DeQuantizationTester, test_case_one) {
    const float min_value = -63.5;
    const float max_value = 64.0;
    uint8_t input_data[] = {0, 1, 2, 3, 4, 251, 252, 253, 254, 255};
    float expect_out[] = {-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64};
    float scale_ = (max_value - min_value) / 255.0;
    int32_t zero_point_ = std::min(255, std::max(0, static_cast<int32_t>(round(0 - min_value / scale_))));
    this->TestPrepare({1, 1, 2, 5}).TestRun(input_data, expect_out, scale_, zero_point_);
}

TYPED_TEST(DeQuantizationTester, test_signed) {
    int8_t input_data[] = {-128, -96, 0, 127};
    float expect_out[] = {0.0, 32.0, 128.0, 255.0};
    this->TestPrepare({1, 2, 2, 1}).TestRun(input_data, expect_out, 1.0, -128);
}

TYPED_TEST(DeQuantizationTester, test_signed_1) {
    int8_t input_data[] = {-128, -127, -126, -125, -124, 123, 124, 125, 126, 127};
    float expect_out[] = {-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64};
    this->TestPrepare({1, 1, 2, 5}).TestRun(input_data, expect_out, 0.5, -1);
}

TYPED_TEST(DeQuantizationTester, VTS1) {
    const float min_value = 0;
    const float max_value = 255;
    uint8_t input_data[] = {0, 32, 128, 255};
    float expect_out[] = {0.0, 32.0, 128.0, 255.0};
    float scale_ = (max_value - min_value) / 255.0;
    int32_t zero_point_ = std::min(255, std::max(0, static_cast<int32_t>(round(0 - min_value / scale_))));
    this->TestPrepare({1, 1, 2, 2}).TestRun(input_data, expect_out, scale_, zero_point_);
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
