#include <gtest/gtest.h>
#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLSlice.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class CLSliceTester : public ::testing::Test {
  public:
    CLSliceTester() {
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

    CLSliceTester &TestPrepare(const Dim4 &input_dim, std::vector<Dim4>& out_dims) {
        input_dim_ = input_dim;
        out_dims_ = out_dims;
        return *this;
    }

    void TestRun(float *input_data, int32_t &axis, std::vector<int32_t> &slicePoint,
        std::vector<float*> &expected_outputs) {
        std::shared_ptr<ITensor> input_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, input_dim_);
        std::vector<std::shared_ptr<ITensor>> output_tensors;
        for (int i = 0; i < expected_outputs.size(); i++) {
            output_tensors.emplace_back(std::make_shared<CLTensor>(runtime_, precision_, data_type_, out_dims_[i]));
        }
        parameters_ = std::make_shared<SliceParameters>();
        parameters_->axis = axis;
        parameters_->slice_point = slicePoint;

        CLSlice slice(runtime_, precision_);
        EXPECT_EQ(slice.initialize({input_tensor}, output_tensors, parameters_), Status::SUCCESS);
        EXPECT_EQ(input_tensor->writeData(input_data), Status::SUCCESS);
        EXPECT_EQ(slice.execute(), Status::SUCCESS);
        EXPECT_EQ(slice.release(), Status::SUCCESS);

        int32_t output_size = 0;
        for (int i = 0; i < output_tensors.size(); i++) {
            auto out_tensor = output_tensors[i];
            output_size = out_tensor->getTotalSizeFromDims();
            auto output_ptr = make_shared_array<float>(output_size);
            out_tensor->readData(output_ptr.get());
            Compare(output_ptr.get(), expected_outputs[i], output_size, error_threshold_);
        }
    }

  private:
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;
    DataType data_type_;
    float error_threshold_;

    Dim4 input_dim_ = {0, 0, 0, 0};
    std::vector<Dim4> out_dims_;
    std::shared_ptr<SliceParameters> parameters_ = nullptr;
};

TYPED_TEST_CASE(CLSliceTester, TestFP32AndFP16Type);
TYPED_TEST(CLSliceTester, Axis0_Point0) {
    int32_t axis = 0;
    std::vector<int32_t> slicePoint;
    Dim4 in_dim = {6, 2, 1, 2};
    Dim4 out_dim1 = {2, 2, 1, 2};
    Dim4 out_dim2 = {2, 2, 1, 2};
    Dim4 out_dim3 = {2, 2, 1, 2};
    float input_data[] = {0, 0.1, -0.2, 0.3,
                          -0.4, 0.5, 0.6, -0.7,
                          0.8, 0.9, 1, -1.1,
                          1.2, 1.3, -1.4, 1.5,
                          1.6, 1.7, -1.8, 1.9,
                          2, 2.1, -2.2, 2.3};
    float slice_data_1[] = {0, 0.1, -0.2, 0.3,
                            -0.4, 0.5, 0.6, -0.7};
    float slice_data_2[] = {0.8, 0.9, 1, -1.1,
                            1.2, 1.3, -1.4, 1.5};
    float slice_data_3[] = {1.6, 1.7, -1.8, 1.9,
                            2, 2.1, -2.2, 2.3};
    std::vector<float*> expected_output;
    expected_output.push_back(slice_data_1);
    expected_output.push_back(slice_data_2);
    expected_output.push_back(slice_data_3);
    std::vector<Dim4> out_dims = {out_dim1, out_dim2, out_dim3};
    this->TestPrepare(in_dim, out_dims).TestRun(input_data, axis, slicePoint, expected_output);
}

TYPED_TEST(CLSliceTester, Axis0) {
    int32_t axis = 0;
    std::vector<int32_t> slicePoint = {2, 5};
    Dim4 in_dim = {6, 2, 1, 2};
    Dim4 out_dim1 = {2, 2, 1, 2};
    Dim4 out_dim2 = {3, 2, 1, 2};
    Dim4 out_dim3 = {1, 2, 1, 2};
    float input_data[] = {0, 0.1, -0.2, 0.3,
                          -0.4, 0.5, 0.6, -0.7,
                          0.8, 0.9, 1, -1.1,
                          1.2, 1.3, -1.4, 1.5,
                          1.6, 1.7, -1.8, 1.9,
                          2, 2.1, -2.2, 2.3};
    float slice_data_1[] = {0, 0.1, -0.2, 0.3,
                            -0.4, 0.5, 0.6, -0.7};
    float slice_data_2[] = {0.8, 0.9, 1, -1.1,
                            1.2, 1.3, -1.4, 1.5,
                            1.6, 1.7, -1.8, 1.9};
    float slice_data_3[] = {2, 2.1, -2.2, 2.3};
    std::vector<float*> expected_output;
    expected_output.push_back(slice_data_1);
    expected_output.push_back(slice_data_2);
    expected_output.push_back(slice_data_3);
    std::vector<Dim4> out_dims = {out_dim1, out_dim2, out_dim3};
    this->TestPrepare(in_dim, out_dims).TestRun(input_data, axis, slicePoint, expected_output);
}

TYPED_TEST(CLSliceTester, Axis0_SameWithInput) {
    int32_t axis = 0;
    std::vector<int32_t> slicePoint;
    Dim4 in_dim = {6, 2, 1, 2};
    Dim4 out_dim1 = {6, 2, 1, 2};
    float input_data[] = {0, 0.1, -0.2, 0.3,
                          -0.4, 0.5, 0.6, -0.7,
                          0.8, 0.9, 1, -1.1,
                          1.2, 1.3, -1.4, 1.5,
                          1.6, 1.7, -1.8, 1.9,
                          2, 2.1, -2.2, 2.3};
    float slice_data[] = {0, 0.1, -0.2, 0.3,
                          -0.4, 0.5, 0.6, -0.7,
                          0.8, 0.9, 1, -1.1,
                          1.2, 1.3, -1.4, 1.5,
                          1.6, 1.7, -1.8, 1.9,
                          2, 2.1, -2.2, 2.3};
    std::vector<float*> expected_output;
    expected_output.push_back(slice_data);
    std::vector<Dim4> out_dims = {out_dim1};
    this->TestPrepare(in_dim, out_dims).TestRun(input_data, axis, slicePoint, expected_output);
}

TYPED_TEST(CLSliceTester, Axis1) {
    int32_t axis = 1;
    std::vector<int32_t> slicePoint = {2, 5};
    Dim4 in_dim = {2, 6, 1, 2};
    Dim4 out_dim1 = {2, 2, 1, 2};
    Dim4 out_dim2 = {2, 3, 1, 2};
    Dim4 out_dim3 = {2, 1, 1, 2};
    float input_data[] = {0, 0.1, -0.2, 0.3,
                          -0.4, 0.5, 0.6, -0.7,
                          0.8, 0.9, 1, -1.1,
                          1.2, 1.3, -1.4, 1.5,
                          1.6, 1.7, -1.8, 1.9,
                          2, 2.1, -2.2, 2.3};
    float slice_data_1[] = {0, 0.1, -0.2, 0.3,
                            1.2, 1.3, -1.4, 1.5};
    float slice_data_2[] = {-0.4, 0.5, 0.6, -0.7,
                            0.8, 0.9, 1.6, 1.7,
                            -1.8, 1.9, 2, 2.1};
    float slice_data_3[] = {1, -1.1, -2.2, 2.3};
    std::vector<float*> expected_output;
    expected_output.push_back(slice_data_1);
    expected_output.push_back(slice_data_2);
    expected_output.push_back(slice_data_3);
    std::vector<Dim4> out_dims = {out_dim1, out_dim2, out_dim3};
    this->TestPrepare(in_dim, out_dims).TestRun(input_data, axis, slicePoint, expected_output);
}

TYPED_TEST(CLSliceTester, Axis1_Point0) {
    int32_t axis = 1;
    std::vector<int32_t> slicePoint;
    Dim4 in_dim = {2, 6, 1, 2};
    Dim4 out_dim1 = {2, 3, 1, 2};
    Dim4 out_dim2 = {2, 3, 1, 2};
    float input_data[] = {0, 0.1, -0.2, 0.3,
                          -0.4, 0.5, 0.6, -0.7,
                          0.8, 0.9, 1, -1.1,
                          1.2, 1.3, -1.4, 1.5,
                          1.6, 1.7, -1.8, 1.9,
                          2, 2.1, -2.2, 2.3};
    float slice_data_1[] = {0, 0.1, -0.2, 0.3, -0.4, 0.5,
                            1.2, 1.3, -1.4, 1.5, 1.6, 1.7};
    float slice_data_2[] = {0.6, -0.7, 0.8, 0.9, 1, -1.1,
                            -1.8, 1.9, 2, 2.1, -2.2, 2.3};
    std::vector<float*> expected_output;
    expected_output.push_back(slice_data_1);
    expected_output.push_back(slice_data_2);
    std::vector<Dim4> out_dims = {out_dim1, out_dim2};
    this->TestPrepare(in_dim, out_dims).TestRun(input_data, axis, slicePoint, expected_output);
}

TYPED_TEST(CLSliceTester, Axis2) {
    int32_t axis = 2;
    std::vector<int32_t> slicePoint = {2, 5};
    Dim4 in_dim = {2, 1, 6, 2};
    Dim4 out_dim1 = {2, 1, 2, 2};
    Dim4 out_dim2 = {2, 1, 3, 2};
    Dim4 out_dim3 = {2, 1, 1, 2};
    float input_data[] = {0, 0.1, -0.2, 0.3,
                          -0.4, 0.5, 0.6, -0.7,
                          0.8, 0.9, 1, -1.1,
                          1.2, 1.3, -1.4, 1.5,
                          1.6, 1.7, -1.8, 1.9,
                          2, 2.1, -2.2, 2.3};
    float slice_data_1[] = {0, 0.1, -0.2, 0.3,
                            1.2, 1.3, -1.4, 1.5};
    float slice_data_2[] = {-0.4, 0.5, 0.6, -0.7, 0.8, 0.9,
                            1.6, 1.7, -1.8, 1.9, 2, 2.1};
    float slice_data_3[] = {1, -1.1, -2.2, 2.3};
    std::vector<float*> expected_output;
    expected_output.push_back(slice_data_1);
    expected_output.push_back(slice_data_2);
    expected_output.push_back(slice_data_3);
    std::vector<Dim4> out_dims = {out_dim1, out_dim2, out_dim3};
    this->TestPrepare(in_dim, out_dims).TestRun(input_data, axis, slicePoint, expected_output);
}

TYPED_TEST(CLSliceTester, Axis2_Point0) {
    int32_t axis = 2;
    std::vector<int32_t> slicePoint;
    Dim4 in_dim = {2, 1, 6, 2};
    Dim4 out_dim1 = {2, 1, 3, 2};
    Dim4 out_dim2 = {2, 1, 3, 2};
    float input_data[] = {0, 0.1, -0.2, 0.3,
                          -0.4, 0.5, 0.6, -0.7,
                          0.8, 0.9, 1, -1.1,
                          1.2, 1.3, -1.4, 1.5,
                          1.6, 1.7, -1.8, 1.9,
                          2, 2.1, -2.2, 2.3};
    float slice_data_1[] = {0, 0.1, -0.2, 0.3, -0.4, 0.5,
                            1.2, 1.3, -1.4, 1.5, 1.6, 1.7};
    float slice_data_2[] = {0.6, -0.7, 0.8, 0.9, 1, -1.1,
                            -1.8, 1.9, 2, 2.1, -2.2, 2.3};
    std::vector<float*> expected_output;
    expected_output.push_back(slice_data_1);
    expected_output.push_back(slice_data_2);
    std::vector<Dim4> out_dims = {out_dim1, out_dim2};
    this->TestPrepare(in_dim, out_dims).TestRun(input_data, axis, slicePoint, expected_output);
}

TYPED_TEST(CLSliceTester, Axis3) {
    int32_t axis = 3;
    std::vector<int32_t> slicePoint = {2, 5};
    Dim4 in_dim = {2, 1, 2, 6};
    Dim4 out_dim1 = {2, 1, 2, 2};
    Dim4 out_dim2 = {2, 1, 2, 3};
    Dim4 out_dim3 = {2, 1, 2, 1};
    float input_data[] = {0, 0.1, -0.2, 0.3,
                          -0.4, 0.5, 0.6, -0.7,
                          0.8, 0.9, 1, -1.1,
                          1.2, 1.3, -1.4, 1.5,
                          1.6, 1.7, -1.8, 1.9,
                          2, 2.1, -2.2, 2.3};
    float slice_data_1[] = {0, 0.1, 0.6, -0.7,
                            1.2, 1.3, -1.8, 1.9};
    float slice_data_2[] = {-0.2, 0.3, -0.4, 0.8, 0.9, 1,
                            -1.4, 1.5, 1.6, 2, 2.1, -2.2};
    float slice_data_3[] = {0.5, -1.1, 1.7, 2.3};
    std::vector<float*> expected_output;
    expected_output.push_back(slice_data_1);
    expected_output.push_back(slice_data_2);
    expected_output.push_back(slice_data_3);
    std::vector<Dim4> out_dims = {out_dim1, out_dim2, out_dim3};
    this->TestPrepare(in_dim, out_dims).TestRun(input_data, axis, slicePoint, expected_output);
}

TYPED_TEST(CLSliceTester, Axis3_Point0) {
    int32_t axis = 3;
    std::vector<int32_t> slicePoint;
    Dim4 in_dim = {2, 1, 2, 6};
    Dim4 out_dim1 = {2, 1, 2, 2};
    Dim4 out_dim2 = {2, 1, 2, 2};
    Dim4 out_dim3 = {2, 1, 2, 2};
    float input_data[] = {0, 0.1, -0.2, 0.3,
                          -0.4, 0.5, 0.6, -0.7,
                          0.8, 0.9, 1, -1.1,
                          1.2, 1.3, -1.4, 1.5,
                          1.6, 1.7, -1.8, 1.9,
                          2, 2.1, -2.2, 2.3};
    float slice_data_1[] = {0, 0.1, 0.6, -0.7,
                            1.2, 1.3, -1.8, 1.9};
    float slice_data_2[] = {-0.2, 0.3,  0.8, 0.9,
                            -1.4, 1.5, 2, 2.1};
    float slice_data_3[] = {-0.4, 0.5, 1, -1.1,
                            1.6, 1.7, -2.2, 2.3};
    std::vector<float*> expected_output;
    expected_output.push_back(slice_data_1);
    expected_output.push_back(slice_data_2);
    expected_output.push_back(slice_data_3);
    std::vector<Dim4> out_dims = {out_dim1, out_dim2, out_dim3};
    this->TestPrepare(in_dim, out_dims).TestRun(input_data, axis, slicePoint, expected_output);
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
