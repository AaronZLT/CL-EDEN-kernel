#include <gtest/gtest.h>
#include "userdriver/cpu/operators/CFUInverter.hpp"
#include "userdriver/common/op_test/test_utils.h"

namespace enn {
namespace ud {
namespace cpu {

class CFUInverterTester {
public:
    explicit CFUInverterTester(float threshold = ERROR_THRESHOLD) : error_threshold_(threshold) {}

    CFUInverterTester& SetDims(const Dim4& input_dims, const Dim4& output_dims) {
        input_dims_ = input_dims;
        output_dims_ = output_dims;
        return *this;
    }

    CFUInverterTester& SetFormatterVal(const int32_t& cols_in_cell, const int32_t& lines_in_cell,
                                              const int32_t& interleaved_slices, const int32_t& idps = 0,
                                              const int32_t& unit_size = 0) {
        cols_in_cell_ = cols_in_cell;
        lines_in_cell_ = lines_in_cell;
        interleaved_slices_ = interleaved_slices;
        idps_ = idps;
        unit_size_ = unit_size;
        return *this;
    }

    template <typename T>
    void TestRun(T* input_data, const T* reference_output_data, const std::string soc_name,
                 Status status = Status::SUCCESS) {
        PrecisionType precision = getPrecisionType(input_data);
        size_t output_size = GetDimSize(output_dims_);

        auto input_tensor = std::make_shared<NEONTensor<T>>(input_data, input_dims_, precision);
        auto output_tensor = std::make_shared<NEONTensor<T>>(output_dims_, precision);

        CFUInverter _icfu(precision, soc_name);
        EXPECT_EQ(_icfu.initialize(input_tensor, output_dims_.w, output_dims_.h, output_dims_.c, cols_in_cell_,
                                   lines_in_cell_, interleaved_slices_, idps_, unit_size_), Status::SUCCESS);
        EXPECT_EQ(_icfu.execute(input_tensor, output_tensor), status);
        EXPECT_EQ(_icfu.release(), Status::SUCCESS);

        if (status == Status::SUCCESS)
            Compare(output_tensor->getDataPtr().get(), reference_output_data, output_size, error_threshold_);
    }

private:
    float error_threshold_;
    int32_t cols_in_cell_;
    int32_t lines_in_cell_;
    int32_t interleaved_slices_;
    int32_t idps_;
    int32_t unit_size_;
    Dim4 input_dims_;
    Dim4 output_dims_;
};

TEST(ENN_CPU_OP_UT_CFUInverter, INT16_EXYNOS2100) {
    std::string soc_name = enn::platform::EXYNOS2100;

    int16_t input_data[] = {
        0x0101, 0, 0, 0, 0, 0, 0, 0, 0x0202, 0, 0, 0, 0, 0, 0, 0, 0x0303, 0, 0, 0, 0, 0, 0, 0, 0x0404, 0, 0, 0, 0, 0, 0, 0,
        0x0505, 0, 0, 0, 0, 0, 0, 0, 0x0606, 0, 0, 0, 0, 0, 0, 0, 0x0707, 0, 0, 0, 0, 0, 0, 0, 0x0808, 0, 0, 0, 0, 0, 0, 0,
        0x0909, 0, 0, 0, 0, 0, 0, 0, 0x0A0A, 0, 0, 0, 0, 0, 0, 0, 0x0B0B, 0, 0, 0, 0, 0, 0, 0, 0x0C0C, 0, 0, 0, 0, 0, 0, 0,
        0x0D0D, 0, 0, 0, 0, 0, 0, 0, 0x0E0E, 0, 0, 0, 0, 0, 0, 0, 0x0F0F, 0, 0, 0, 0, 0, 0, 0, 0x0000, 0, 0, 0, 0, 0, 0, 0,
        0x0000, 0, 0, 0, 0, 0, 0, 0, 0x0000, 0, 0, 0, 0, 0, 0, 0, 0x0000, 0, 0, 0, 0, 0, 0, 0, 0x0000, 0, 0, 0, 0, 0, 0, 0,
        0x0000, 0, 0, 0, 0, 0, 0, 0, 0x0000, 0, 0, 0, 0, 0, 0, 0, 0x0000, 0, 0, 0, 0, 0, 0, 0, 0x0000, 0, 0, 0, 0, 0, 0, 0,
        0x0000, 0, 0, 0, 0, 0, 0, 0, 0x0000, 0, 0, 0, 0, 0, 0, 0, 0x0000, 0, 0, 0, 0, 0, 0, 0, 0x0000, 0, 0, 0, 0, 0, 0, 0,
        0x0000, 0, 0, 0, 0, 0, 0, 0, 0x0000, 0, 0, 0, 0, 0, 0, 0};
    int16_t reference_output_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                       1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    Dim4 input_dims_ = {1, 16, 3, 5};
    Dim4 output_dims_ = {1, 2, 3, 5};
    CFUInverterTester()
        .SetDims(input_dims_, output_dims_)
        .SetFormatterVal(1, 1, 16)
        .TestRun(input_data, reference_output_data, soc_name);
}

TEST(ENN_CPU_OP_UT_CFUInverter, INT8_EXYNOS2100) {
    std::string soc_name = enn::platform::EXYNOS2100;

    int8_t input_data[] = {4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    int8_t reference_output_data[] = {4, 5, 0, 2, 2, 0, 4, 2, 0, 4, 1, 3, 5, 2, 4, 2,
                                      4, 5, 0, 2, 2, 0, 4, 2, 0, 4, 1, 3, 5, 2, 4, 2};

    Dim4 input_dims_ = {1, 16, 4, 4};
    Dim4 output_dims_ = {1, 2, 4, 4};
    CFUInverterTester()
        .SetDims(input_dims_, output_dims_)
        .SetFormatterVal(1, 1, 16)
        .TestRun(input_data, reference_output_data, soc_name);
}

// Failed on 97 Server but passed on Bungjin's Server
// TEST(CFUInverter, INT16_EXYNOS9830) {
//     std::string soc_name = enn::platform::EXYNOS9830;

//     int16_t input_data[] = {0x0201, 0x0403, 0x0605, 0x0000, 0x0807, 0x0A09, 0x0C0B, 0x0000, 0x0E0D, 0x100F, 0x1211,
//                             0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0,      0,      0,      0,      0,      0,
//                             0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
//                             0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
//                             0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
//                             0,      0,      0,      0,      0,      0,      0,      0,      0};
//     int16_t reference_output_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};

//     Dim4 input_dims_ = {1, 2, 4, 8};
//     Dim4 output_dims_ = {1, 1, 3, 6};
//     CFUInverterTester()
//         .SetDims(input_dims_, output_dims_)
//         .SetFormatterVal(8, 4, 2, 4, 2)
//         .TestRun(input_data, reference_output_data, soc_name);
// }

TEST(ENN_CPU_OP_UT_CFUInverter, INT8_EXYNOS9830) {
    std::string soc_name = enn::platform::EXYNOS9830;

    int8_t input_data[] = {4, 5, 0, 2, 0, 0, 0, 0, 2, 0, 4, 2, 0, 0, 0, 0, 0, 4, 1, 3, 0, 0, 0, 0, 5, 2, 4, 2, 0, 0, 0, 0,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int8_t reference_output_data[] = {4, 5, 0, 2, 2, 0, 4, 2, 0, 4, 1, 3, 5, 2, 4, 2};

    Dim4 input_dims_ = {1, 2, 4, 8};
    Dim4 output_dims_ = {1, 1, 4, 4};
    CFUInverterTester()
        .SetDims(input_dims_, output_dims_)
        .SetFormatterVal(8, 4, 2, 4, 2)
        .TestRun(input_data, reference_output_data, soc_name);
}

TEST(ENN_CPU_OP_UT_CFUInverter, INT8_EXYNOS9820) {
    std::string soc_name = enn::platform::EXYNOS9820;

    int8_t input_data[] = {4, 5, 0, 2, 4, 2, 0, 4, 5, 2, 4, 2, 0, 0, 0, 0, 2, 0, 0, 0, 1, 3, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0};
    int8_t reference_output_data[] = {4, 5, 0, 2, 2, 0, 4, 2, 0, 4, 1, 3, 5, 2, 4, 2, 3, 4};

    Dim4 input_dims_ = {1, 1, 4, 8};
    Dim4 output_dims_ = {1, 1, 3, 6};
    CFUInverterTester()
        .SetDims(input_dims_, output_dims_)
        .SetFormatterVal(4, 4, 1, 1, 1)
        .TestRun(input_data, reference_output_data, soc_name);
}

TEST(ENN_CPU_OP_UT_CFUInverter, FLOAT16_EXYNOS9925) {
    std::string soc_name = enn::platform::EXYNOS9925;

    _Float16_t input_data[] = {
        1.0,  1.6,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.5,  2.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        3.6,  3.2,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4.2,  9.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        5.0,  4.7,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6.1,  8.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        7.6,  6.0,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.6,  7.4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        9.8,  8.9,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0,  9.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1.1,  10.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12.3, 1.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        14.6, 2.5,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4.2,  3.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        15.1, 14.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    _Float16_t reference_output_data[] = {1.0, 2.5, 3.6, 4.2, 5.0, 6.1, 7.6, 3.6, 9.8, 1.0, 1.1, 12.3, 14.6, 4.2, 15.1,
                               1.6, 2.8, 3.2, 9.1, 4.7, 8.1, 6.0, 7.4, 8.9, 9.1, 10.0, 1.1, 2.5, 3.8, 14.2};

    Dim4 input_dims_  = {1, 16, 3, 5};
    Dim4 output_dims_ = {1, 2, 3, 5};
#ifndef UNIT_TEST
    CFUInverterTester().SetDims(input_dims_, output_dims_).SetFormatterVal(1, 1, 16).  // 1, 1, 16
        TestRun(input_data, reference_output_data, soc_name);
#endif
}

TEST(ENN_CPU_OP_UT_CFUInverter, INT16_EXYNOS9925) {
    std::string soc_name = enn::platform::EXYNOS9925;

    int16_t reference_output_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                       1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    int16_t input_data[] = {
        0x0101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x0202, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0x0303, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x0404, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0x0505, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x0606, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0x0707, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x0808, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0x0909, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x0A0A, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0x0B0B, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x0C0C, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0x0D0D, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x0E0E, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0x0F0F, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x0000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0x0000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x0000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0x0000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x0000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0x0000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x0000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0x0000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x0000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0x0000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x0000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0x0000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x0000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0x0000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x0000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    Dim4 output_dims_ = {1, 2, 3, 5};
    Dim4 input_dims_ = {1, 32, 3, 5};
    CFUInverterTester().SetDims(input_dims_, output_dims_).SetFormatterVal(1, 1, 32).TestRun(input_data, reference_output_data, soc_name);
}

TEST(ENN_CPU_OP_UT_CFUInverter, UNKNOWN_SOC) {
    std::string soc_name = "Unknown";  // Incase of invalid SOC, 9820 logic will get executed

    int8_t input_data[] = {4, 5, 0, 2, 4, 2, 0, 4, 5, 2, 4, 2, 0, 0, 0, 0, 2, 0, 0, 0, 1, 3, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0};

    int8_t reference_output_data[] = {4, 5, 0, 2, 2, 0, 4, 2, 0, 4, 1, 3, 5, 2, 4, 2, 3, 4};

    Dim4 input_dims_ = {1, 1, 4, 8};
    Dim4 output_dims_ = {1, 1, 3, 6};
    CFUInverterTester()
        .SetDims(input_dims_, output_dims_)
        .SetFormatterVal(4, 4, 1, 1, 1)
        .TestRun(input_data, reference_output_data, soc_name);
}

TEST(ENN_CPU_OP_UT_CFUInverter, UNSUPPORTED_DATATYPE) {  // Data type not supported
    std::string soc_name = enn::platform::EXYNOS9830;

    uint16_t input_data[] = {0x0201, 0x0403, 0x0605, 0x0000, 0x0807, 0x0A09, 0x0C0B, 0x0000, 0x0E0D, 0x100F, 0x1211,
                             0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0,      0,      0,      0,      0,      0,
                             0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                             0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                             0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,
                             0,      0,      0,      0,      0,      0,      0,      0,      0};
    uint16_t reference_output_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};

    Dim4 input_dims_ = {1, 2, 4, 8};
    Dim4 output_dims_ = {1, 1, 3, 6};
    CFUInverterTester()
        .SetDims(input_dims_, output_dims_)
        .SetFormatterVal(8, 4, 2, 4, 2)
        .TestRun(input_data, reference_output_data, soc_name, Status::FAILURE);
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
