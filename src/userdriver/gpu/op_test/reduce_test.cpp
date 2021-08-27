#include <gtest/gtest.h>

#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"
#include "userdriver/gpu/operators/CLReduce.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class ReduceTester : public ::testing::Test {
public:
    ReduceTester() {
        runtime_ = std::shared_ptr<CLRuntime>(new CLRuntime());
        runtime_->initialize(0);
        runtime_->initializeQueue();
        precision_ = PRECISION::precision;
        data_type_ = DataType::FLOAT;
        if (PRECISION::precision == PrecisionType ::FP32) {
            error_threshold_ = 1e-4;
        } else if (PRECISION::precision == PrecisionType ::FP16) {
            error_threshold_ = 1e-1;
        }
    }

    ReduceTester &TestPrepare(const NDims &input_dims,
                              const NDims &output_dims,
                              const DataType &data_type,
                              const Reducer &op_type) {
        input_dims_ = input_dims;
        output_dims_ = output_dims;
        data_type_ = data_type;
        op_type_ = op_type;
        return *this;
    }

    template <typename T> void TestRun(T *input_data, std::vector<int32_t> axis, const T *reference_output_data) {
        in_size = GetDimSize(input_dims_);
        out_size = GetDimSize(output_dims_);

        axis_dim_ = {(uint32_t)axis.size(), 1, 1, 1};

        auto input_tensor = std::make_shared<CLTensor>(runtime_, precision_, input_data, input_dims_);
        auto output_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, output_dims_);
        auto axis_tensor = std::make_shared<CLTensor>(runtime_, precision_, axis.data(), axis_dim_);

        parameters_ = std::make_shared<ReduceParameters>();
        parameters_->keep_dims = false;
        parameters_->reducer = op_type_;

        CLReduce cl_reduce(runtime_, precision_);
        EXPECT_EQ(cl_reduce.initialize({input_tensor, axis_tensor}, {output_tensor}, parameters_), Status::SUCCESS);
        EXPECT_EQ(cl_reduce.execute(), Status::SUCCESS);
        EXPECT_EQ(cl_reduce.release(), Status::SUCCESS);

        std::shared_ptr<T> output = make_shared_array<T>(out_size);
        output_tensor->readData(output.get());
        Compare(output.get(), reference_output_data, out_size, error_threshold_);
    }

private:
    size_t in_size = 0;
    size_t out_size = 0;
    float error_threshold_;
    int block_size_ = 0;
    NDims input_dims_;
    NDims output_dims_;
    NDims axis_dim_;
    PrecisionType precision_;
    DataType data_type_;
    Reducer op_type_;
    std::shared_ptr<CLRuntime> runtime_;
    std::shared_ptr<ReduceParameters> parameters_;
};

TYPED_TEST_CASE(ReduceTester, TestFP32AndFP16Type);
TYPED_TEST(ReduceTester, ReduceProdWithOneAxis_W1) {
    float input_data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
                          1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    float expect_output[] = {24, 1680, 11880, 24, 1680, 11880};
    std::vector<int32_t> axis = {3};
    this->TestPrepare({1, 2, 3, 4}, {1, 2, 3, 1}, DataType::FLOAT, Reducer::PROD).TestRun(input_data, axis, expect_output);
}

TYPED_TEST(ReduceTester, ReduceMinWithOneAxis_W1) {
    float input_data[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                          13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    float expect_output[] = {1, 5, 9, 13, 17, 21};
    std::vector<int32_t> axis = {3};
    this->TestPrepare({1, 2, 3, 4}, {1, 2, 3, 1}, DataType::FLOAT, Reducer::MIN).TestRun(input_data, axis, expect_output);
}

TYPED_TEST(ReduceTester, ReduceMaxWithOneAxis_W1) {
    float input_data[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                          13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    float expect_output[] = {4, 8, 12, 16, 20, 24};
    std::vector<int32_t> axis = {3};
    this->TestPrepare({1, 2, 3, 4}, {1, 2, 3, 1}, DataType::FLOAT, Reducer::MAX).TestRun(input_data, axis, expect_output);
}

TYPED_TEST(ReduceTester, ReduceAnyWithOneAxis_W1) {
    bool input_data[] = {false, true,  false, false, false, false, false, false, false, false, true,  true,
                         false, false, false, false, true,  false, true,  false, false, false, false, false};
    bool expect_output[] = {true, false, true, false, true, false};
    std::vector<int32_t> axis = {3};
    this->TestPrepare({1, 2, 3, 4}, {1, 2, 3, 1}, DataType::BOOL, Reducer::ANY).TestRun(input_data, axis, expect_output);
}

TYPED_TEST(ReduceTester, ReduceAllWithOneAxis_W1) {
    bool input_data[] = {false, true, false, false, false, false, true, false, true,  true, true, true,
                         true,  true, true,  false, true,  true,  true, true,  false, true, true, false};
    bool expect_output[] = {false, false, true, false, true, false};
    std::vector<int32_t> axis = {3};
    this->TestPrepare({1, 2, 3, 4}, {1, 2, 3, 1}, DataType::BOOL, Reducer::ALL).TestRun(input_data, axis, expect_output);
}

TYPED_TEST(ReduceTester, ReduceSumWithOneAxis_W2) {
    float input_data[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                          13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    float expect_output[] = {6, 15, 24, 33, 42, 51, 60, 69};
    std::vector<int32_t> axis = {3};
    this->TestPrepare({1, 2, 4, 3}, {1, 2, 4, 1}, DataType::FLOAT, Reducer::SUM).TestRun(input_data, axis, expect_output);
}

TYPED_TEST(ReduceTester, ReduceSumWithOneAxis_W3) {
    float input_data[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                          13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                          25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0};
    float expect_output[] = {21, 57, 93, 129, 165, 201};
    std::vector<int32_t> axis = {3};
    this->TestPrepare({1, 2, 3, 6}, {1, 2, 3, 1}, DataType::FLOAT, Reducer::SUM).TestRun(input_data, axis, expect_output);
}

TYPED_TEST(ReduceTester, ReduceSumWithOneAxis_H1) {
    float input_data[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                          13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    float expect_output[] = {15, 18, 21, 24, 51, 54, 57, 60};
    std::vector<int32_t> axis = {2};
    this->TestPrepare({1, 2, 3, 4}, {1, 2, 1, 4}, DataType::FLOAT, Reducer::SUM).TestRun(input_data, axis, expect_output);
}

TYPED_TEST(ReduceTester, ReduceSumWithOneAxis_H2) {
    float input_data[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                          13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
                          25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0};
    float expect_output[] = {51, 57, 63, 159, 165, 171};
    std::vector<int32_t> axis = {2};
    this->TestPrepare({1, 2, 6, 3}, {1, 2, 1, 3}, DataType::FLOAT, Reducer::SUM).TestRun(input_data, axis, expect_output);
}

TYPED_TEST(ReduceTester, ReduceSumWithOneAxis_H3) {
    float input_data[] = {51, 26, 39, 51, 64, 77};
    float expect_output[] = {154, 154};
    std::vector<int32_t> axis = {2};
    this->TestPrepare({1, 1, 3, 2}, {1, 1, 1, 2}, DataType::FLOAT, Reducer::SUM).TestRun(input_data, axis, expect_output);
}

TYPED_TEST(ReduceTester, ReduceSumWithOneAxis_C) {
    float input_data[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                          13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    float expect_output[] = {66, 72, 78, 84};
    std::vector<int32_t> axis = {1};
    this->TestPrepare({1, 6, 2, 2}, {1, 1, 2, 2}, DataType::FLOAT, Reducer::SUM).TestRun(input_data, axis, expect_output);
}

TYPED_TEST(ReduceTester, ReduceSumWithOneAxis_N) {
    float input_data[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                          13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    float expect_output[] = {27, 30, 33, 36, 39, 42, 45, 48};
    std::vector<int32_t> axis = {0};
    this->TestPrepare({3, 4, 1, 2}, {1, 4, 1, 2}, DataType::FLOAT, Reducer::SUM).TestRun(input_data, axis, expect_output);
}

TYPED_TEST(ReduceTester, ReduceSumWithTwoAxis_NH) {
    float input_data[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                          13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    float expect_output[] = {40, 44, 48, 52, 56, 60};
    std::vector<int32_t> axis = {0, 2};
    this->TestPrepare({4, 3, 1, 2}, {1, 3, 1, 2}, DataType::FLOAT, Reducer::SUM).TestRun(input_data, axis, expect_output);
}

TYPED_TEST(ReduceTester, ReduceSumWithTwoAxis_NC) {
    float input_data[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                          13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    float expect_output[] = {144, 156};
    std::vector<int32_t> axis = {-4, -3};
    this->TestPrepare({4, 3, 1, 2}, {1, 1, 1, 2}, DataType::FLOAT, Reducer::SUM).TestRun(input_data, axis, expect_output);
}

TYPED_TEST(ReduceTester, ReduceSumWithTwoAxis_CW) {
    float input_data[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                          13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    float expect_output[] = {126, 174};
    std::vector<int32_t> axis = {1, -3, 3, -1};
    this->TestPrepare({1, 3, 2, 4}, {1, 1, 2, 1}, DataType::FLOAT, Reducer::SUM).TestRun(input_data, axis, expect_output);
}

TYPED_TEST(ReduceTester, ReduceSumWithTwoAxis_HW) {
    float input_data[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                          13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    float expect_output[] = {36, 100, 164};
    std::vector<int32_t> axis = {2, 3};
    this->TestPrepare({1, 3, 2, 4}, {1, 3, 1, 1}, DataType::FLOAT, Reducer::SUM).TestRun(input_data, axis, expect_output);
}

TYPED_TEST(ReduceTester, ReduceSumWithThreeAxis_CHW) {
    float input_data[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                          13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    float expect_output[] = {78, 222};
    std::vector<int32_t> axis = {1, 2, 3};
    this->TestPrepare({2, 3, 1, 4}, {2, 1, 1, 1}, DataType::FLOAT, Reducer::SUM).TestRun(input_data, axis, expect_output);
}

TYPED_TEST(ReduceTester, ReduceSumWithFourAxis_NCHW) {
    float input_data[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                          13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    float expect_output[] = {300};
    std::vector<int32_t> axis = {0, 1, 2, 3};
    this->TestPrepare({2, 3, 1, 4}, {1, 1, 1, 1}, DataType::FLOAT, Reducer::SUM).TestRun(input_data, axis, expect_output);
}

TYPED_TEST(ReduceTester, ReduceMaxWithFourAxis_NCHW) {
    float input_data[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                          13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    float expect_output[] = {24};
    std::vector<int32_t> axis = {0, 1, 2, 3};
    this->TestPrepare({2, 3, 1, 4}, {1, 1, 1, 1}, DataType::FLOAT, Reducer::MAX).TestRun(input_data, axis, expect_output);
}

TYPED_TEST(ReduceTester, ReduceMinWithFourAxis_NCHW) {
    float input_data[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
                          13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
    float expect_output[] = {1};
    std::vector<int32_t> axis = {0, 1, 2, 3};
    this->TestPrepare({2, 3, 1, 4}, {1}, DataType::FLOAT, Reducer::MIN).TestRun(input_data, axis, expect_output);
}

TYPED_TEST(ReduceTester, CtsReduceMax1) {
    float input_data[] = {-1, -2, 3, 4, 5, -6};
    float expect_output[] = {-1, 4, 5};
    std::vector<int32_t> axis = {-1};
    this->TestPrepare({3, 2}, {3}, DataType::FLOAT, Reducer::MAX).TestRun(input_data, axis, expect_output);
}

TYPED_TEST(ReduceTester, CtsReduceMax2) {
    float input_data[] = {-1, -2, 3, 4, 5, -6};
    float expect_output[] = {3, 5};
    std::vector<int32_t> axis = {-1};
    this->TestPrepare({2, 3}, {2}, DataType::FLOAT, Reducer::MAX).TestRun(input_data, axis, expect_output);
}

TYPED_TEST(ReduceTester, CtsReduceMAX3) {
    float input_data[100];
    for (int i = 0; i < 100; i++) {
        input_data[i] = i + 1;
    }
    float expect_output[] = {100};
    std::vector<int32_t> axis = {0, -2, 0, -2, -1, -1};
    this->TestPrepare({10, 10}, {1}, DataType::FLOAT, Reducer::MAX).TestRun(input_data, axis, expect_output);
}
TYPED_TEST(ReduceTester, CtsReduceMAX4) {
    float input_data[100];
    for (int i = 0; i < 100; i++) {
        input_data[i] = i + 1;
    }
    float expect_output[] = {100};
    std::vector<int32_t> axis = {-1};
    this->TestPrepare({100}, {1}, DataType::FLOAT, Reducer::MAX).TestRun(input_data, axis, expect_output);
}

TYPED_TEST(ReduceTester, CtsReduceMAX5) {
    float input_data[11];
    for (int i = 0; i < 11; i++) {
        input_data[i] = i + 1;
    }
    float expect_output[] = {11};
    std::vector<int32_t> axis = {-1};
    this->TestPrepare({11}, {1}, DataType::FLOAT, Reducer::MAX).TestRun(input_data, axis, expect_output);
}
TYPED_TEST(ReduceTester, CtsReduceMAX6) {
    float input_data[10];
    for (int i = 0; i < 10; i++) {
        input_data[i] = i + 1;
    }
    float expect_output[] = {10};
    std::vector<int32_t> axis = {-1};
    this->TestPrepare({10}, {1}, DataType::FLOAT, Reducer::MAX).TestRun(input_data, axis, expect_output);
}

TYPED_TEST(ReduceTester, CtsReduceMAX7) {
    float input_data[9];
    for (int i = 0; i < 9; i++) {
        input_data[i] = i + 1;
    }
    float expect_output[] = {9};
    std::vector<int32_t> axis = {-1};
    this->TestPrepare({9}, {1}, DataType::FLOAT, Reducer::MAX).TestRun(input_data, axis, expect_output);
}

TYPED_TEST(ReduceTester, CtsReduceMAX8) {
    float input_data[157];
    for (int i = 0; i < 157; i++) {
        input_data[i] = i + 1;
    }
    float expect_output[] = {157};
    std::vector<int32_t> axis = {-1};
    this->TestPrepare({157}, {1}, DataType::FLOAT, Reducer::MAX).TestRun(input_data, axis, expect_output);
}

TYPED_TEST(ReduceTester, CtsReduceMAX9) {
    float input_data[10000];
    for (int i = 0; i < 10000; i++) {
        input_data[i] = i + 1;
    }
    float expect_output[] = {10000};
    std::vector<int32_t> axis = {-1};
    this->TestPrepare({10000}, {1}, DataType::FLOAT, Reducer::MAX).TestRun(input_data, axis, expect_output);
}

TYPED_TEST(ReduceTester, CtsReduceMAX10) {
    float input_data[10000];
    for (int i = 0; i < 10000; i++) {
        input_data[i] = i + 1;
    }
    float expect_output[] = {10000};
    std::vector<int32_t> axis = {-2, -1};
    this->TestPrepare({100, 100}, {1}, DataType::FLOAT, Reducer::MAX).TestRun(input_data, axis, expect_output);
}

TYPED_TEST(ReduceTester, CtsReduceSUM1) {
    float input_data[100];
    for (int i = 0; i < 100; i++) {
        input_data[i] = 1;
    }
    float expect_output[] = {100};
    std::vector<int32_t> axis = {0, -2, 0, -2, -1, -1};
    this->TestPrepare({10, 10}, {1}, DataType::FLOAT, Reducer::SUM).TestRun(input_data, axis, expect_output);
}

TYPED_TEST(ReduceTester, CtsReduceSUM2) {
    float input_data[10000];
    for (int i = 0; i < 10000; i++) {
        input_data[i] = 1;
    }
    float expect_output[] = {10000};
    std::vector<int32_t> axis = {0, -2, 0, -2, -1, -1};
    this->TestPrepare({100, 100}, {1}, DataType::FLOAT, Reducer::SUM).TestRun(input_data, axis, expect_output);
}

TYPED_TEST(ReduceTester, CtsReduceSUM3) {
    float input_data[79];
    for (int i = 0; i < 79; i++) {
        input_data[i] = 1;
    }
    float expect_output[] = {79};
    std::vector<int32_t> axis = {-1};
    this->TestPrepare({79}, {1}, DataType::FLOAT, Reducer::SUM).TestRun(input_data, axis, expect_output);
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
