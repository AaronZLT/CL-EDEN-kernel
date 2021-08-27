#include <gtest/gtest.h>
#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/operators/CLQuantization.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class QuantizationTester : public ::testing::Test {
  public:
    QuantizationTester() {
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

    QuantizationTester &TestPrepare(const Dim4 &input_dims) {
        input_dims_ = input_dims;
        output_dims_ = input_dims;
        return *this;
    }
    template <typename T>
    void TestRun(float *input_data, const T *reference_output_data, const float scale, const int32_t zero_point) {
        size_t input_size = GetDimSize(input_dims_);
        size_t output_size = GetDimSize(output_dims_);

        data_type_ = data_type_map_[typeid(T).name()];

        std::shared_ptr<ITensor> input_tensor = std::make_shared<CLTensor>(runtime_, precision_, input_data, input_dims_);
        inputs_.push_back(input_tensor);
        std::shared_ptr<ITensor> output_tensor = std::make_shared<CLTensor>(runtime_,
                                                                            precision_,
                                                                            data_type_,
                                                                            output_dims_,
                                                                            DataOrder::NCHW,
                                                                            scale,
                                                                            zero_point,
                                                                            BufferType::DEDICATED,
                                                                            StorageType::BUFFER,
                                                                            0,
                                                                            UNDEFINED);
        outputs_.push_back(output_tensor);

        CLQuantization process_cl(runtime_, precision_);
        EXPECT_EQ(process_cl.initialize(inputs_, outputs_), Status::SUCCESS);
        EXPECT_EQ(process_cl.execute(), Status::SUCCESS);
        EXPECT_EQ(process_cl.release(), Status::SUCCESS);

        auto out_ptr = make_shared_array<T>(output_size);
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
};

TYPED_TEST_CASE(QuantizationTester, TestFP32AndFP16Type);
TYPED_TEST(QuantizationTester, simple) {
    const float scale = 2.0;
    const int32_t zero_point = 128;
    float input_data[] = {-1.1, 0.6, 0.7, 1.2, -0.7, 0.1};
    uint8_t expect_out[] = {127, 128, 128, 129, 128, 128};
    this->TestPrepare({1, 1, 1, 6}).TestRun(input_data, expect_out, scale, zero_point);
}

TYPED_TEST(QuantizationTester, V1_3_signed) {
    const float scale = 1.0;
    const int32_t zero_point = -128;
    float input_data[] = {-10.0f,
                          -9.933110367892976f,
                          -8.060200668896321f,
                          -7.993311036789297f,
                          -6.989966555183947f,
                          -5.050167224080268f,
                          -4.983277591973244f,
                          -3.0434782608695654f,
                          -2.976588628762542f,
                          -1.0367892976588635f,
                          -0.9698996655518393f,
                          1.1705685618729085f,
                          2.909698996655518f,
                          3.913043478260869f,
                          4.983277591973243f,
                          5.050167224080267f,
                          6.454849498327757f,
                          7.45819397993311f,
                          8.060200668896321f,
                          9.933110367892976f,
                          10.0f};
    int8_t expect_out[] = {-128, -128, -128, -128, -128, -128, -128, -128, -128, -128, -128,
                           -127, -125, -124, -123, -123, -122, -121, -120, -118, -118};
    this->TestPrepare({1, 1, 1, 21}).TestRun(input_data, expect_out, scale, zero_point);
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
