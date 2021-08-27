#include <gtest/gtest.h>
#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLStridedSlice.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class CLStridedSliceTester : public ::testing::Test {
  public:
    CLStridedSliceTester() {
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

    CLStridedSliceTester &TestPrepare(const Dim4 &input_dim,
                                      const Dim4 &output_dim,
                                      const int32_t begin_mask,
                                      const int32_t end_mask,
                                      const int32_t shrink_axis_mask) {
        input_dim_ = input_dim;
        output_dim_ = output_dim;
        input_size_ = GetDimSize(input_dim_);
        output_size_ = GetDimSize(output_dim_);
        parameters_ = std::make_shared<StridedSliceParameters>();
        parameters_->begin_mask = begin_mask;
        parameters_->end_mask = end_mask;
        parameters_->shrink_axis_mask = shrink_axis_mask;

        return *this;
    }

    void TestRun(float *input_data,
                 std::vector<int32_t> &begin,
                 std::vector<int32_t> &end,
                 std::vector<int32_t> &strides,
                 float *golden_data) {
        NDims begin_dim = {static_cast<uint32_t>(begin.size()), 1, 1, 1};
        NDims end_dim = {static_cast<uint32_t>(end.size()), 1, 1, 1};
        NDims strides_dim = {static_cast<uint32_t>(strides.size()), 1, 1, 1};
        auto input_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, input_dim_);
        auto begin_tensor = std::make_shared<CLTensor>(runtime_, precision_, begin.data(), begin_dim);
        auto end_tensor = std::make_shared<CLTensor>(runtime_, precision_, end.data(), end_dim);
        auto strides_tensor = std::make_shared<CLTensor>(runtime_, precision_, strides.data(), strides_dim);
        auto output_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, output_dim_);

        CLStridedSlice strided_slice(runtime_, precision_);
        EXPECT_EQ(
            strided_slice.initialize({input_tensor, begin_tensor, end_tensor, strides_tensor}, {output_tensor}, parameters_),
            Status::SUCCESS);
        EXPECT_EQ(input_tensor->writeData(input_data), Status::SUCCESS);
        EXPECT_EQ(strided_slice.execute(), Status::SUCCESS);
        EXPECT_EQ(strided_slice.release(), Status::SUCCESS);

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
    Dim4 output_dim_;
    size_t input_size_;
    size_t output_size_;
    std::shared_ptr<StridedSliceParameters> parameters_;
};

TYPED_TEST_CASE(CLStridedSliceTester, TestFP32AndFP16Type);
TYPED_TEST(CLStridedSliceTester, In1D_OutOfRangeEndNegativeStride) {
    float input_data[] = {1, 2, 3, 4};
    std::vector<int32_t> begin_data = {-3};
    std::vector<int32_t> end_data = {-5};
    std::vector<int32_t> strides_data = {-1};
    float expect_data[] = {2, 1};
    this->TestPrepare({4, 1, 1, 1}, {2, 1, 1, 1}, 0, 0, 0)
        .TestRun(input_data, begin_data, end_data, strides_data, expect_data);
}

TYPED_TEST(CLStridedSliceTester, In3D_Identity) {
    float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<int32_t> begin_data = {0, 0, 0};
    std::vector<int32_t> end_data = {2, 3, 2};
    std::vector<int32_t> strides_data = {1, 1, 1};
    float expect_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    this->TestPrepare({2, 3, 2, 1}, {2, 3, 2, 1}, 0, 0, 0)
        .TestRun(input_data, begin_data, end_data, strides_data, expect_data);
}

TYPED_TEST(CLStridedSliceTester, In3D_NegStride) {
    float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<int32_t> begin_data = {-1, -1, -1};
    std::vector<int32_t> end_data = {-3, -4, -3};
    std::vector<int32_t> strides_data = {-1, -1, -1};
    float expect_data[] = {12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
    this->TestPrepare({2, 3, 2, 1}, {2, 3, 2, 1}, 0, 0, 0)
        .TestRun(input_data, begin_data, end_data, strides_data, expect_data);
}

TYPED_TEST(CLStridedSliceTester, In3D_Strided2) {
    float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<int32_t> begin_data = {0, 0, 0};
    std::vector<int32_t> end_data = {2, 3, 2};
    std::vector<int32_t> strides_data = {2, 2, 2};
    float expect_data[] = {1, 5};
    this->TestPrepare({2, 3, 2, 1}, {1, 2, 1, 1}, 0, 0, 0)
        .TestRun(input_data, begin_data, end_data, strides_data, expect_data);
}

TYPED_TEST(CLStridedSliceTester, In3D_IdentityShrinkAxis1) {
    float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<int32_t> begin_data = {0, 0, 0};
    std::vector<int32_t> end_data = {1, 3, 2};
    std::vector<int32_t> strides_data = {1, 1, 1};
    float expect_data[] = {1, 2, 3, 4, 5, 6};
    this->TestPrepare({2, 3, 2, 1}, {3, 2, 1, 1}, 0, 0, 1)
        .TestRun(input_data, begin_data, end_data, strides_data, expect_data);
}

TYPED_TEST(CLStridedSliceTester, In3D_IdentityShrinkAxis2) {
    float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<int32_t> begin_data = {0, 0, 0};
    std::vector<int32_t> end_data = {1, 1, 2};
    std::vector<int32_t> strides_data = {1, 1, 1};
    float expect_data[] = {1, 2};
    this->TestPrepare({2, 3, 2, 1}, {2, 1, 1, 1}, 0, 0, 3)
        .TestRun(input_data, begin_data, end_data, strides_data, expect_data);
}

TYPED_TEST(CLStridedSliceTester, In3D_IdentityShrinkAxis4) {
    float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<int32_t> begin_data = {0, 0, 0};
    std::vector<int32_t> end_data = {2, 3, 1};
    std::vector<int32_t> strides_data = {1, 1, 1};
    float expect_data[] = {1, 3, 5, 7, 9, 11};
    this->TestPrepare({2, 3, 2, 1}, {2, 3, 1, 1}, 0, 0, 4)
        .TestRun(input_data, begin_data, end_data, strides_data, expect_data);
}

TYPED_TEST(CLStridedSliceTester, In3D_IdentityShrinkAxis5) {
    float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<int32_t> begin_data = {0, 0, 0};
    std::vector<int32_t> end_data = {1, 3, 1};
    std::vector<int32_t> strides_data = {1, 1, 1};
    float expect_data[] = {1, 3, 5};
    this->TestPrepare({2, 3, 2, 1}, {3, 1, 1, 1}, 0, 0, 5)
        .TestRun(input_data, begin_data, end_data, strides_data, expect_data);
}

TYPED_TEST(CLStridedSliceTester, In3D_IdentityShrinkAxis6) {
    float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    std::vector<int32_t> begin_data = {0, 0, 0};
    std::vector<int32_t> end_data = {2, 1, 1};
    std::vector<int32_t> strides_data = {1, 1, 1};
    float expect_data[] = {1, 7};
    this->TestPrepare({2, 3, 2, 1}, {2, 1, 1, 1}, 0, 0, 6)
        .TestRun(input_data, begin_data, end_data, strides_data, expect_data);
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
