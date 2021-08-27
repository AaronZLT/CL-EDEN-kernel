#include <gtest/gtest.h>
#include "userdriver/cpu/operators/NEONDequantization.hpp"
#include "userdriver/common/op_test/test_utils.h"

#define FRAC_DIM(x) \
    { 1, sizeof(x) / sizeof(int32_t), 1, 1 }

namespace enn {
namespace ud {
namespace cpu {

class DequantizationTester {
public:
    explicit DequantizationTester(float threshold = ERROR_THRESHOLD) : error_threshold_(threshold) {}

    DequantizationTester& InputDims(const Dim4& input_dims) {
        input_dims_ = input_dims;
        output_dims_ = input_dims;
        frac_dims_ = {input_dims.n, input_dims.c, 1, 1};
        return *this;
    }

    template <typename T>
    void TestRun(T* input_data, int32_t* frac_len, const float* reference_output_data, Status status = Status::SUCCESS) {
        PrecisionType precision = getPrecisionType(input_data);
        size_t output_size = GetDimSize(output_dims_);

        auto input_tensor = std::make_shared<NEONTensor<T>>(input_data, input_dims_, precision);
        auto output_tensor = std::make_shared<NEONTensor<float>>(output_dims_, PrecisionType::FP32);
        auto frac_tensor = std::make_shared<NEONTensor<int32_t>>(frac_len, frac_dims_, PrecisionType::INT32);

        uint32_t img_size = input_dims_.h * input_dims_.w;
        int32_t data_num = input_dims_.c * img_size;
        NEONDequantization _neon_dequantize(precision);
        EXPECT_EQ(_neon_dequantize.initialize(input_tensor, data_num, frac_tensor, img_size), Status::SUCCESS);
        EXPECT_EQ(_neon_dequantize.execute(input_tensor, output_tensor), status);
        EXPECT_EQ(_neon_dequantize.release(), Status::SUCCESS);

        if (status == Status::SUCCESS)
            Compare(output_tensor->getDataPtr().get(), reference_output_data, output_size, error_threshold_);
    }

private:
    float error_threshold_;
    Dim4 input_dims_;
    Dim4 output_dims_;
    Dim4 frac_dims_;
};

// reference code was ported from S.LSI implement
template <typename T>
void DeQuantizationRef(const T* input, float* output, const uint32_t& input_size, int32_t* frac_len, int32_t channel) {
    uint32_t i;
    int num;
    float ret_val;
    for (i = 0; i < input_size; i++) {
        num = input[i];
        ret_val = (float)num / std::pow(2, frac_len[i / (input_size / channel)]);
        output[i] = ret_val;
    }
}

template <typename UnitType>
void TEST_COMMON(Dim4& input_dims, Status status = Status::SUCCESS) {
    size_t array_size = GetDimSize(input_dims);
    std::shared_ptr<UnitType> in;
    in.reset(new UnitType[array_size], std::default_delete<UnitType[]>());
    GenerateRandom<UnitType>(in.get(), array_size, 0, 5);

    std::shared_ptr<int32_t> frac_len;
    frac_len.reset(new int32_t[input_dims.c], std::default_delete<int32_t[]>());
    GenerateRandom(frac_len.get(), input_dims.c, 1, 5);

    float* refer_output = new float[array_size];
    DeQuantizationRef(in.get(), refer_output, array_size, frac_len.get(), input_dims.c);
    DequantizationTester().InputDims(input_dims).TestRun(in.get(), frac_len.get(), refer_output, status);

    delete[] refer_output;
}

TEST(ENN_CPU_OP_UT_Dequantization, SingleChannel) {
    Dim4 input_dims = {1, 1, 7, 7};
    TEST_COMMON<int8_t>(input_dims);
}

TEST(ENN_CPU_OP_UT_Dequantization, HFDInput) {
    Dim4 input_dims = {1, 1, 32, 32};
    TEST_COMMON<int8_t>(input_dims);
}

TEST(ENN_CPU_OP_UT_Dequantization, Simple) {
    Dim4 input_dims = {1, 3, 8, 8};
    TEST_COMMON<int8_t>(input_dims);
}

TEST(ENN_CPU_OP_UT_Dequantization, Simple_Int16) {
    Dim4 input_dims = {1, 3, 16, 16};
    TEST_COMMON<int16_t>(input_dims);
}

TEST(ENN_CPU_OP_UT_Dequantization, AlexnetInput) {
    Dim4 input_dims = {1, 3, 256, 256};
    TEST_COMMON<int8_t>(input_dims);
}

TEST(ENN_CPU_OP_UT_Dequantization, InceptionV3Input) {
    Dim4 input_dims = {1, 3, 299, 299};
    TEST_COMMON<int8_t>(input_dims);
}

TEST(ENN_CPU_OP_UT_Dequantization, INVALID_INPUT) {
    Dim4 input_dims = {0, 0, 0, 0};
    int8_t* input_data = NULL;
    float* reference_output_data = NULL;
    int32_t* frac_lens = NULL;
    DequantizationTester()
        .InputDims(input_dims)
        .TestRun(input_data, frac_lens, reference_output_data, Status::INVALID_PARAMS);
}

TEST(ENN_CPU_OP_UT_Dequantization, UNSUPPORTED_DATA_TYPE) {
    Dim4 input_dims = {1, 2, 1, 1};
    uint16_t input_data[] = {3, 2};
    float reference_output_data[] = {2.0, 3.0};
    int32_t frac_lens[] = {2, 2};
    DequantizationTester().InputDims(input_dims).TestRun(input_data, frac_lens, reference_output_data, Status::FAILURE);
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
