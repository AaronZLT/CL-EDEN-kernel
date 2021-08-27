#include <gtest/gtest.h>
#include "userdriver/cpu/operators/Concat.hpp"
#include "userdriver/common/op_test/test_utils.h"

namespace enn {
namespace ud {
namespace cpu {

#define INPUT_NUM 2

class ConcatTester {
 public:
    explicit ConcatTester(float threshold = ERROR_THRESHOLD) : error_threshold_(threshold) {}

    ConcatTester& SetDims(const Dim4& input_dims) {
        input_dims_ = input_dims;
        output_dims_ = {INPUT_NUM * input_dims.n, input_dims.c, input_dims.h, input_dims.w};
        return *this;
    }

    ConcatTester& SetAxis(const int32_t& axis) {
        axis_ = axis;
        return *this;
    }

    template <typename T>
    void TestRun(T** input_data, const T* reference_output_data, Status status = Status::SUCCESS) {
        PrecisionType precision = getPrecisionType(input_data);
        size_t output_size = GetDimSize(output_dims_);

        std::vector<std::shared_ptr<ITensor>> input_tensor;
        auto output_tensor = std::make_shared<NEONTensor<T>>(output_dims_, precision);

        for (int i = 0; i < INPUT_NUM; i++) {
            auto input = std::make_shared<NEONTensor<T>>(input_data[i], input_dims_, precision);
            input_tensor.push_back(input);
        }

        Concat _concat(precision);
        EXPECT_EQ(_concat.initialize(input_tensor, INPUT_NUM, input_dims_.n,
                            input_dims_.c, input_dims_.h, input_dims_.w, axis_), Status::SUCCESS);
        EXPECT_EQ(_concat.execute(input_tensor, output_tensor), status);
        EXPECT_EQ(_concat.release(), Status::SUCCESS);

        if (status == Status::SUCCESS)
            Compare(output_tensor->getDataPtr().get(), reference_output_data, output_size, error_threshold_);
    }

 private:
    float error_threshold_;
    Dim4 input_dims_;
    Dim4 output_dims_;
    int32_t axis_;
};

template <typename T>
T** Generate_Input(Dim4& input_dims) {
    size_t array_size = GetDimSize(input_dims);
    T** input_data = new T*[INPUT_NUM];

    for (int i = 0; i < INPUT_NUM; i++) {
        input_data[i] = new T[array_size];
        for (int j = 0; j < array_size; j++) input_data[i][j] = i * array_size + (j + 1);
    }

    return input_data;
}

template<typename T>
void Free_Input(T** input_data) {
    for(int idx = 0; idx < INPUT_NUM; idx++) {
        delete[] input_data[idx];
    }
    delete[] input_data;
}

TEST(ENN_CPU_OP_UT_Concat, UINT8_INPUT_AXIS0) {
    Dim4 input_dims_ = {2, 2, 2, 3};
    uint8_t** input_data = Generate_Input<uint8_t>(input_dims_);
    uint8_t reference_output_data[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                       33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48};

    ConcatTester().SetDims(input_dims_).SetAxis(0).TestRun(input_data, reference_output_data);
    Free_Input(input_data);
}

TEST(ENN_CPU_OP_UT_Concat, UINT8_INPUT_AXIS1) {
    Dim4 input_dims_ = {2, 2, 2, 3};
    uint8_t** input_data = Generate_Input<uint8_t>(input_dims_);
    uint8_t reference_output_data[] = {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 25, 26, 27, 28,
                                       29, 30, 31, 32, 33, 34, 35, 36, 13, 14, 15, 16, 17, 18, 19, 20,
                                       21, 22, 23, 24, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48};

    ConcatTester().SetDims(input_dims_).SetAxis(1).TestRun(input_data, reference_output_data);
    Free_Input(input_data);
}

TEST(ENN_CPU_OP_UT_Concat, UINT8_INPUT_AXIS2) {
    Dim4 input_dims_ = {2, 2, 2, 3};
    uint8_t** input_data = Generate_Input<uint8_t>(input_dims_);

    uint8_t reference_output_data[] = {1,  2,  3,  4,  5,  6,  25, 26, 27, 28, 29, 30, 7,  8,  9,  10,
                                       11, 12, 31, 32, 33, 34, 35, 36, 13, 14, 15, 16, 17, 18, 37, 38,
                                       39, 40, 41, 42, 19, 20, 21, 22, 23, 24, 43, 44, 45, 46, 47, 48};

    ConcatTester().SetDims(input_dims_).SetAxis(2).TestRun(input_data, reference_output_data);
    Free_Input(input_data);
}

TEST(ENN_CPU_OP_UT_Concat, UINT8_INPUT_AXIS3) {
    Dim4 input_dims_ = {2, 2, 2, 3};
    uint8_t** input_data = Generate_Input<uint8_t>(input_dims_);

    uint8_t reference_output_data[] = {1,  2,  3,  25, 26, 27, 4,  5,  6,  28, 29, 30, 7,  8,  9,  31,
                                       32, 33, 10, 11, 12, 34, 35, 36, 13, 14, 15, 37, 38, 39, 16, 17,
                                       18, 40, 41, 42, 19, 20, 21, 43, 44, 45, 22, 23, 24, 46, 47, 48};

    ConcatTester().SetDims(input_dims_).SetAxis(3).TestRun(input_data, reference_output_data);
    Free_Input(input_data);
}

TEST(ENN_CPU_OP_UT_Concat, INVALID_INPUT) {
    Dim4 input_dims_ = {2, 0, 2, 3};
    uint8_t** input_data = Generate_Input<uint8_t>(input_dims_);

    uint8_t reference_output_data[] = {1};

    ConcatTester()
        .SetDims(input_dims_)
        .SetAxis(0)
        .TestRun(input_data, reference_output_data, Status::INVALID_PARAMS);  // channel = 0
    Free_Input(input_data);
}

TEST(ENN_CPU_OP_UT_Concat, INVALID_AXIS) {
    Dim4 input_dims_ = {2, 1, 2, 3};
    uint8_t** input_data = Generate_Input<uint8_t>(input_dims_);

    uint8_t reference_output_data[] = {1};
    ConcatTester()
        .SetDims(input_dims_)
        .SetAxis(4)
        .TestRun(input_data, reference_output_data, Status::INVALID_PARAMS);  // axis > 3
    Free_Input(input_data);
}

TEST(ENN_CPU_OP_UT_Concat, UNSUPPORTED_DATATYPE) {  // Data type not supported
    Dim4 input_dims_ = {1, 1, 1, 1};
    uint16_t** input_data = Generate_Input<uint16_t>(input_dims_);
    uint16_t reference_output_data[] = {1, 2};

    ConcatTester().SetDims(input_dims_).SetAxis(0).TestRun(input_data, reference_output_data, Status::FAILURE);
    Free_Input(input_data);
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
