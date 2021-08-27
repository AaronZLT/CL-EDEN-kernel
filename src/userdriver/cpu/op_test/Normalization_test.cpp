#include <gtest/gtest.h>
#include "userdriver/cpu/operators/Normalization.hpp"
#include "userdriver/common/op_test/test_utils.h"

namespace enn {
namespace ud {
namespace cpu {

class NormalizationTester {
public:
    explicit NormalizationTester(float threshold = ERROR_THRESHOLD) : error_threshold_(threshold) {}

    NormalizationTester& InputDims(const Dim4& input_dims, const uint8_t& bgr_transpose) {
        bgr_transpose_ = bgr_transpose;
        input_dims_ = input_dims;
        output_dims_ = input_dims;
        mean_dims_ = {1, 1, 1, input_dims_.c};
        scale_dims_ = {1, 1, 1, input_dims_.c};
        return *this;
    }

    template <typename T>
    void TestRun(T* input_data, float* scale, float* mean, const float* reference_output_data,
                 Status status = Status::SUCCESS) {
        PrecisionType precision = getPrecisionType(input_data);
        size_t output_size = GetDimSize(output_dims_);

        auto input_tensor = std::make_shared<NEONTensor<T>>(input_data, input_dims_, precision);
        auto output_tensor = std::make_shared<NEONTensor<float>>(output_dims_, PrecisionType::FP32);
        auto scale_tensor = std::make_shared<NEONTensor<float>>(scale, scale_dims_, PrecisionType::FP32);
        auto mean_tensor = std::make_shared<NEONTensor<float>>(mean, mean_dims_, PrecisionType::FP32);

        Normalization _normalize(precision);
        EXPECT_EQ(_normalize.initialize(input_tensor, mean_tensor, scale_tensor, bgr_transpose_), Status::SUCCESS);
        EXPECT_EQ(_normalize.execute(input_tensor, output_tensor), status);
        EXPECT_EQ(_normalize.release(), Status::SUCCESS);

        if (status == Status::SUCCESS)
            Compare(output_tensor->getDataPtr().get(), reference_output_data, output_size, error_threshold_);
    }

private:
    float error_threshold_;
    Dim4 input_dims_;
    Dim4 mean_dims_;
    Dim4 scale_dims_;
    Dim4 output_dims_;
    std::shared_ptr<float> mean_;
    std::shared_ptr<float> scale_;
    uint8_t bgr_transpose_;
};

// reference code was ported from S.LSI implement
template <typename T>
void NormalizationRef(T* input, float* output, int channel, int height, int width, float* mean, float* scale,
                      uint8_t bgr_transpose) {
    int i, j, k;
    T num;
    uint8_t c_type_order;  //*< channel type order
    uint32_t idx;
    if (bgr_transpose == 1) {
        //* normalization & bgr_transpose
        uint8_t b, g, r;
        float normalized;
        int b_idx, g_idx, r_idx;
        b = 0;
        g = 1;
        r = 2;

        b_idx = 0;
        g_idx = height * width;
        r_idx = height * width * 2;

        for (i = 0; i < channel; i++) {
            for (j = 0; j < height; j++) {
                for (k = 0; k < width; k++) {
                    idx = (i * height * width) + (j * width) + k;
                    num = input[idx];
                    c_type_order = i % channel;
                    normalized = ((float)num - mean[c_type_order]) * scale[c_type_order];
                    if (c_type_order == b)
                        output[b_idx++] = normalized;
                    else if (c_type_order == g)
                        output[g_idx++] = normalized;
                    else if (c_type_order == r)
                        output[r_idx++] = normalized;
                }
            }
        }
    } else {
        // just normalization
        for (i = 0; i < channel; i++) {
            for (j = 0; j < height; j++) {
                for (k = 0; k < width; k++) {
                    idx = (i * height * width) + (j * width) + k;
                    num = input[idx];
                    c_type_order = i % channel;
                    output[idx] = (static_cast<float>(num) - mean[i]) * scale[i];
                }
            }
        }
    }
}

template <typename UnitType>
void TEST_COMMON(Dim4& input_dims, int8_t bgr_transpose, Status status = Status::SUCCESS) {
    size_t array_size = GetDimSize(input_dims);
    std::shared_ptr<UnitType> in;
    in.reset(new UnitType[array_size], std::default_delete<UnitType[]>());
    GenerateRandom<UnitType>(in.get(), array_size, 0, 5);

    std::shared_ptr<float> scale;
    scale.reset(new float[input_dims.c], std::default_delete<float[]>());
    GenerateRandom<float>(scale.get(), input_dims.c, 0, 3);

    std::shared_ptr<float> mean;
    mean.reset(new float[input_dims.c], std::default_delete<float[]>());
    GenerateRandom<float>(mean.get(), input_dims.c, 0, 3);

    float* refer_output = new float[array_size];
    NormalizationRef(in.get(), refer_output, input_dims.c, input_dims.h, input_dims.w, mean.get(), scale.get(),
                     bgr_transpose);

    NormalizationTester(1e-5)
        .InputDims(input_dims, bgr_transpose)
        .TestRun(in.get(), scale.get(), mean.get(), refer_output, status);

    delete[] refer_output;
}

TEST(ENN_CPU_OP_UT_Normalization, SmallHeightBGR) {
    Dim4 input_dims = {1, 3, 32, 32};
    TEST_COMMON<uint8_t>(input_dims, 1);
}

TEST(ENN_CPU_OP_UT_Normalization, SmallHeightBGRZero) {
    Dim4 input_dims = {1, 3, 32, 32};
    TEST_COMMON<uint8_t>(input_dims, 0);
}

TEST(ENN_CPU_OP_UT_Normalization, SingleChannelBGR) {
    Dim4 input_dims = {1, 1, 7, 7};
    TEST_COMMON<float>(input_dims, 1);
}

TEST(ENN_CPU_OP_UT_Normalization, SingleChannelBGRZero) {
    Dim4 input_dims = {1, 1, 7, 7};
    TEST_COMMON<float>(input_dims, 0);
}

TEST(ENN_CPU_OP_UT_Normalization, BigHeightBGR) {
    Dim4 input_dims = {1, 3, 299, 299};
    TEST_COMMON<uint8_t>(input_dims, 1);
}

TEST(ENN_CPU_OP_UT_Normalization, BigHeightBGRZero) {
    Dim4 input_dims = {1, 3, 299, 299};
    TEST_COMMON<float>(input_dims, 0);
}

TEST(Normalization, UNSUPPORTED_DATATYPE) {
    Dim4 input_dims = {1, 1, 3, 3};
    TEST_COMMON<int16_t>(input_dims, 0, Status::FAILURE);
}

TEST(Normalization, CHANNEL_GT_THAN_THREE) {
    Dim4 input_dims = {1, 5, 7, 7};
    TEST_COMMON<uint8_t>(input_dims, 1, Status::FAILURE);
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
