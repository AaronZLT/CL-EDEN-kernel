#include <gtest/gtest.h>
#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLCast.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class CLCastTester : public ::testing::Test {
public:
    CLCastTester() {
        precision_ = PRECISION::precision;
        switch (precision_) {
        case PrecisionType::FP32: error_threshold_ = 1e-5; break;
        case PrecisionType::FP16: error_threshold_ = 1e-2; break;
        default: break;
        }
        runtime_ = std::shared_ptr<CLRuntime>(new CLRuntime());
        runtime_->initialize(0);
        runtime_->initializeQueue();
    }

    CLCastTester &
    TestPrepare(const NDims &input_dim, const NDims &output_dim, DataType input_data_type, DataType output_data_type) {
        input_dim_ = input_dim;
        output_dim_ = output_dim;
        input_size_ = GetDimSize(input_dim_);
        output_size_ = GetDimSize(output_dim_);
        parameters_ = std::make_shared<CastParameters>();
        parameters_->in_data_type = input_data_type;
        parameters_->out_data_type = output_data_type;

        return *this;
    }

    template <typename T1, typename T2> void TestRun(T1 *input_data, T2 *golden_data) {
        auto input_tensor = std::make_shared<CLTensor>(runtime_, precision_, parameters_->in_data_type, input_dim_);
        if (parameters_->in_data_type != DataType::FLOAT && parameters_->in_data_type != DataType::HALF &&
            parameters_->out_data_type != DataType::FLOAT && parameters_->out_data_type != DataType::HALF) {
            precision_ = PrecisionType::FP32;
        } else if (parameters_->in_data_type == DataType::UINT8 && parameters_->out_data_type == DataType::HALF) {
            precision_ = PrecisionType::FP16;
        } else if (parameters_->in_data_type == DataType::UINT8 && parameters_->out_data_type == DataType::FLOAT) {
            precision_ = PrecisionType::FP32;
        }
        auto output_tensor = std::make_shared<CLTensor>(runtime_, precision_, parameters_->out_data_type, output_dim_);

        CLCast cast(runtime_, precision_);
        EXPECT_EQ(cast.initialize({input_tensor}, {output_tensor}, parameters_), Status::SUCCESS);
        EXPECT_EQ(input_tensor->writeData(input_data), Status::SUCCESS);
        EXPECT_EQ(cast.execute(), Status::SUCCESS);
        EXPECT_EQ(cast.release(), Status::SUCCESS);

        auto output_ptr = make_shared_array<T2>(output_size_);
        output_tensor->readData(output_ptr.get());

        Compare(output_ptr.get(), golden_data, output_size_, error_threshold_);
    }

private:
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;
    float error_threshold_;

    NDims input_dim_;
    NDims output_dim_;
    size_t input_size_;
    size_t output_size_;
    std::shared_ptr<CastParameters> parameters_;
};

TYPED_TEST_CASE(CLCastTester, TestFP32AndFP16Type);

TYPED_TEST(CLCastTester, CastInt32ToFloat) {
    int32_t input_data[] = {100, 200, 300, 400, 500, 600};
    float expect_out[] = {100.f, 200.f, 300.f, 400.f, 500.f, 600.f};
    this->TestPrepare({2, 3}, {2, 3}, DataType::INT32, DataType::FLOAT).TestRun(input_data, expect_out);
}
TYPED_TEST(CLCastTester, CastFloatToInt32) {
    float input_data[] = {100.f, 20.f, 3.f, 0.4f, 0.999f, 1.1f};
    int32_t expect_out[] = {100, 20, 3, 0, 0, 1};
    this->TestPrepare({3, 2}, {3, 2}, DataType::FLOAT, DataType::INT32).TestRun(input_data, expect_out);
}
TYPED_TEST(CLCastTester, CastFloatToUInt8) {
    float input_data[] = {100.f, 1.0f, 0.f, 0.4f, 1.999f, 1.1f};
    uint8_t expect_out[] = {100, 1, 0, 0, 1, 1};
    this->TestPrepare({3, 2}, {3, 2}, DataType::FLOAT, DataType::UINT8).TestRun(input_data, expect_out);
}
TYPED_TEST(CLCastTester, CastUInt8ToFloat) {
    uint8_t input_data[] = {123, 0, 1, 2, 3, 4};
    float expect_out[] = {123.f, 0.f, 1.f, 2.f, 3.f, 4.f};
    this->TestPrepare({3, 2}, {3, 2}, DataType::UINT8, DataType::FLOAT).TestRun(input_data, expect_out);
}
TYPED_TEST(CLCastTester, CastInt32ToUInt8) {
    int32_t input_data[] = {100, 1, 200, 2, 255, 3};
    uint8_t expect_out[] = {100, 1, 200, 2, 255, 3};
    this->TestPrepare({3, 2}, {3, 2}, DataType::INT32, DataType::UINT8).TestRun(input_data, expect_out);
}
TYPED_TEST(CLCastTester, CastUInt8ToInt32) {
    uint8_t input_data[] = {100, 1, 200, 2, 255, 3};
    int32_t expect_out[] = {100, 1, 200, 2, 255, 3};
    this->TestPrepare({3, 2}, {3, 2}, DataType::UINT8, DataType::INT32).TestRun(input_data, expect_out);
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
