#include <gtest/gtest.h>
#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"
#include "userdriver/gpu/operators/CLGather.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class CLGatherTester : public ::testing::Test {
  public:
    CLGatherTester() {
        precision_ = PRECISION::precision;
        switch (precision_) {
        case PrecisionType::FP32:
            data_type_ = DataType::FLOAT;
            error_threshold_ = 1e-5;
            break;
        case PrecisionType::FP16:
            data_type_ = DataType::FLOAT;
            error_threshold_ = 1e-3;
            break;
        default: break;
        }
        runtime_ = std::shared_ptr<CLRuntime>(new CLRuntime());
        runtime_->initialize(0);
        runtime_->initializeQueue();
    }

    CLGatherTester &TestPrepare(const NDims &input_dims, const NDims &indices_dims, const int32_t &axis) {
        input_dims_ = input_dims;
        indices_dims_ = indices_dims;
        output_dims_.clear();  //  compute in operation
        parameters_ = std::make_shared<GatherParameters>();
        parameters_->axis = axis;
        parameters_->androidNN = true;

        if (parameters_->axis < 0)
            parameters_->axis += input_dims.size();
        setOutSize();
        return *this;
    }

    void TestRun(float *input_data = nullptr, int32_t *indices_data = nullptr, const float *golden_data = nullptr) {
        auto input_tensor = std::make_shared<CLTensor>(runtime_, precision_, input_data, input_dims_);
        auto indices_tensor = std::make_shared<CLTensor>(runtime_, precision_, indices_data, indices_dims_);
        auto output_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, output_dims_);

        CLGather gather(runtime_, precision_);
        EXPECT_EQ(gather.initialize({input_tensor, indices_tensor}, {output_tensor}, parameters_), Status::SUCCESS);
        EXPECT_EQ(gather.execute(), Status::SUCCESS);
        EXPECT_EQ(gather.release(), Status::SUCCESS);

        auto output_ptr = make_shared_array<float>(output_size_);
        output_tensor->readData(output_ptr.get());
        Compare(output_ptr.get(), golden_data, output_size_, error_threshold_);
    }

  private:
    CLGatherTester &setOutSize() {
        uint32_t inner_size = 1;
        uint32_t outer_size = 1;
        uint32_t axis_size = GetDimSize(indices_dims_);
        for (int i = 0; i < parameters_->axis; ++i)
            outer_size *= input_dims_[i];
        for (size_t i = parameters_->axis + 1; i < input_dims_.size(); ++i)
            inner_size *= input_dims_[i];
        output_size_ = outer_size * axis_size * inner_size;
        return *this;
    }
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;
    DataType data_type_;
    float error_threshold_;

    NDims input_dims_;
    NDims output_dims_;
    NDims indices_dims_;
    uint32_t output_size_;
    std::shared_ptr<GatherParameters> parameters_;
};

TYPED_TEST_CASE(CLGatherTester, TestFP32AndFP16Type);
TYPED_TEST(CLGatherTester, test_Gather_1) {
    float input[] = {-2.0, 0.2, 0.7, 0.8};
    int32_t indices[] = {1, 0};
    const float expect_out[] = {0.7, 0.8, -2, 0.2};
    this->TestPrepare({2, 2}, {2}, 0).TestRun(input, indices, expect_out);
}
TYPED_TEST(CLGatherTester, test_Gather_2) {
    float input[] = {-2.0, 0.2, 0.7, 0.8};
    int32_t indices[] = {1};
    const float expect_out[] = {0.7, 0.8};
    this->TestPrepare({2, 2}, {1}, 0).TestRun(input, indices, expect_out);
}
TYPED_TEST(CLGatherTester, test_Gather_3) {
    float input[] = {1, 2, 3};
    int32_t indices[] = {1};
    const float expect_out[] = {2};
    this->TestPrepare({3}, {1}, 0).TestRun(input, indices, expect_out);
}
TYPED_TEST(CLGatherTester, test_Gather_4) {
    float input[] = {1, 2, 3};
    int32_t indices[] = {1, 0};
    const float expect_out[] = {2, 1};
    this->TestPrepare({3}, {2}, 0).TestRun(input, indices, expect_out);
}
TYPED_TEST(CLGatherTester, test_Gather_5) {
    float input[] = {-2.0, 0.2, 0.7, 0.8};
    int32_t indices[] = {0, 0};
    const float expect_out[] = {-2.0, 0.2, 0.7, 0.8, -2.0, 0.2, 0.7, 0.8};
    this->TestPrepare({1, 2, 2}, {2}, 0).TestRun(input, indices, expect_out);
}
TYPED_TEST(CLGatherTester, test_Gather_6) {
    float input[] = {-2.0, 0.2, 0.7, 0.8};
    int32_t indices[] = {1, 3};
    const float expect_out[] = {0.2, 0.8};
    this->TestPrepare({4, 1}, {2}, 0).TestRun(input, indices, expect_out);
}
TYPED_TEST(CLGatherTester, test_Gather_7) {
    float input[] = {1, 2, 3, 4, 5, 6};
    int32_t indices[] = {1, 0};
    const float expect_out[] = {4, 5, 6, 1, 2, 3};
    this->TestPrepare({1, 2, 3}, {2}, 1).TestRun(input, indices, expect_out);
}
TYPED_TEST(CLGatherTester, test_Gather_8) {
    float input[] = {1, 2, 3, 4, 5, 6};
    int32_t indices[] = {2, 0};
    const float expect_out[] = {3, 1, 6, 4};
    this->TestPrepare({1, 2, 3}, {2}, -1).TestRun(input, indices, expect_out);
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
