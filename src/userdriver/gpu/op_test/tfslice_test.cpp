#include <gtest/gtest.h>
#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLTFSlice.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class CLTFSliceTester : public ::testing::Test {
public:
    CLTFSliceTester() {
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

    CLTFSliceTester &TestPrepare(const NDims &input_dim, const NDims &output_dim) {
        input_dim_ = input_dim;
        output_dim_ = output_dim;
        input_size_ = getTensorSizeFromDims(input_dim_);
        output_size_ = getTensorSizeFromDims(output_dim_);
        parameters_ = std::make_shared<TFSliceParameters>();
        return *this;
    }

    void TestRun(float *input_data, std::vector<int32_t> &begin, std::vector<int32_t> &size, float *golden_data) {
        NDims begin_dim = {static_cast<uint32_t>(begin.size())};
        NDims size_dim = {static_cast<uint32_t>(size.size())};
        auto input_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, input_dim_);
        auto begin_tensor = std::make_shared<CLTensor>(runtime_, precision_, begin.data(), begin_dim);
        auto size_tensor = std::make_shared<CLTensor>(runtime_, precision_, size.data(), size_dim);
        auto output_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, output_dim_);

        CLTFSlice tfslice(runtime_, precision_);
        EXPECT_EQ(tfslice.initialize({input_tensor, begin_tensor, size_tensor}, {output_tensor}, parameters_),
                  Status::SUCCESS);
        EXPECT_EQ(input_tensor->writeData(input_data), Status::SUCCESS);
        EXPECT_EQ(tfslice.execute(), Status::SUCCESS);
        EXPECT_EQ(tfslice.release(), Status::SUCCESS);

        auto output_ptr = make_shared_array<float>(output_size_);
        output_tensor->readData(output_ptr.get());

        Compare(output_ptr.get(), golden_data, output_size_, error_threshold_);
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
    std::shared_ptr<TFSliceParameters> parameters_;
};

TYPED_TEST_CASE(CLTFSliceTester, TestFP32AndFP16Type);

TYPED_TEST(CLTFSliceTester, test_1) {
    float input_data[] = {1, 2, 3, 4};
    float output_data[] = {2, 3, 4};
    std::vector<int32_t> begin = {1, 0, 0, 0};
    std::vector<int32_t> size = {3, 1, 1, 1};
    this->TestPrepare({4, 1, 1, 1}, {3, 1, 1, 1}).TestRun(input_data, begin, size, output_data);
}

TYPED_TEST(CLTFSliceTester, test_2) {
    float input_data[] = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6};
    float output_data[] = {3, 3, 3};
    std::vector<int32_t> begin = {1, 0, 0, 0};
    std::vector<int32_t> size = {1, 1, 3, 1};
    this->TestPrepare({3, 2, 3, 1}, {1, 1, 3, 1}).TestRun(input_data, begin, size, output_data);
}

TYPED_TEST(CLTFSliceTester, test_3) {
    float input_data[] = {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6};
    float output_data[] = {3, 3, 3, 5, 5, 5};
    std::vector<int32_t> begin = {1, 0, 0, 0};
    std::vector<int32_t> size = {2, 1, -1, 1};
    this->TestPrepare({3, 2, 3, 1}, {2, 1, 3, 1}).TestRun(input_data, begin, size, output_data);
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
