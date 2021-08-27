#include <gtest/gtest.h>
#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"
#include "userdriver/gpu/operators/CLDepthwiseConvolution.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class DepthwiseConvolutionTester : public ::testing::Test {
public:
    DepthwiseConvolutionTester() {
        runtime_ = std::shared_ptr<CLRuntime>(new CLRuntime());
        runtime_->initialize(0);
        runtime_->initializeQueue();
        precision_ = PRECISION::precision;
        data_type_ = DataType::FLOAT;
        if (PRECISION::precision == PrecisionType::FP32) {
            error_threshold_ = 1e-5;
        } else if (PRECISION::precision == PrecisionType::FP16) {
            error_threshold_ = 1e-2;
        }
    }

    inline DepthwiseConvolutionTester &SetComputePrecision(PrecisionType &precision) {
        precision_ = precision;
        return *this;
    }

    DepthwiseConvolutionTester &TestPrepare(const Dim4 &input_dims,
                                            const Dim4 &filters_dims,
                                            const Dim4 &bias_dims,
                                            const Dim4 &output_dims,
                                            const Dim2 &stride,
                                            const Pad4 &pad,
                                            uint32_t depth_multiplier) {
        input_dims_ = input_dims;
        output_dims_ = output_dims;
        filters_dims_ = filters_dims;
        bias_dims_ = bias_dims;
        pad_ = pad;
        stride_ = stride;
        depth_multiplier_ = depth_multiplier;
        return *this;
    }

    void TestRun(float *input_data, float *weight_data, float *bias_data, const float *reference_output_data) {
        auto input_tensor = std::make_shared<CLTensor>(runtime_, precision_, input_data, input_dims_);
        auto weight_tensor = std::make_shared<CLTensor>(runtime_, precision_, weight_data, filters_dims_);
        std::shared_ptr<CLTensor> bias_tensor;
        size_t bias_size = GetDimSize(bias_dims_);
        if (bias_size == 0) {
            bias_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, bias_dims_);
        } else {
            bias_tensor = std::make_shared<CLTensor>(runtime_, precision_, bias_data, bias_dims_);
        }
        inputs_.push_back(input_tensor);
        inputs_.push_back(weight_tensor);
        inputs_.push_back(bias_tensor);

        auto output_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, output_dims_);
        outputs_.push_back(output_tensor);

        parameters_ = std::make_shared<DepthwiseConvolutionParameters>();
        parameters_->padding = pad_;
        parameters_->dilation = {1, 1};
        parameters_->stride = stride_;
        parameters_->depth_multiplier = depth_multiplier_;
        parameters_->per_channel_quant = false;
        parameters_->androidNN = false;
        parameters_->isNCHW = true;
        parameters_->storage_type = StorageType::BUFFER;
        parameters_->activation_info = std::make_shared<ActivationInfo>(ActivationInfo::ActivationType::NONE, false);

        CLDepthwiseConvolution cl_depthwise_conv(runtime_, precision_);

        EXPECT_EQ(cl_depthwise_conv.initialize(inputs_, outputs_, parameters_), Status::SUCCESS);
        EXPECT_EQ(cl_depthwise_conv.execute(), Status::SUCCESS);
        EXPECT_EQ(cl_depthwise_conv.release(), Status::SUCCESS);
        size_t output_size = GetDimSize(output_dims_);
        auto out_ptr = make_shared_array<float>(output_size);
        output_tensor->readData(out_ptr.get());
        Compare(out_ptr.get(), reference_output_data, output_size, error_threshold_);
    }

    void TestRunPumuteData(float *input_data, float *weight_data, float *bias_data, const float *reference_output_data) {
        size_t input_size = GetDimSize(input_dims_);
        size_t output_size = GetDimSize(output_dims_);
        size_t filters_size = GetDimSize(filters_dims_);
        std::shared_ptr<float> input_nchw;
        input_nchw.reset(new float[input_size]);

        PermuteData(input_data, input_dims_, DataOrder::NHWC, input_nchw.get(), input_dims_);
        std::shared_ptr<float> filter_nchw;
        filter_nchw.reset(new float[filters_size]);
        PermuteData(weight_data, filters_dims_, DataOrder::NHWC, filter_nchw.get(), filters_dims_);
        std::shared_ptr<float> output_nchw;
        output_nchw.reset(new float[output_size]);
        PermuteData(reference_output_data, output_dims_, DataOrder::NHWC, output_nchw.get(), output_dims_);
        TestRun(input_nchw.get(), filter_nchw.get(), bias_data, output_nchw.get());
    }

private:
    std::shared_ptr<CLRuntime> runtime_;
    std::shared_ptr<DepthwiseConvolutionParameters> parameters_;
    std::vector<std::shared_ptr<ITensor>> inputs_;
    std::vector<std::shared_ptr<ITensor>> outputs_;
    PrecisionType precision_;
    DataType data_type_;
    float error_threshold_;
    Dim4 input_dims_;
    Dim4 output_dims_;
    Dim4 filters_dims_;
    Dim4 bias_dims_;
    Pad4 pad_;
    Dim2 stride_;
    uint32_t depth_multiplier_;
};

TYPED_TEST_CASE(DepthwiseConvolutionTester, TestFP32AndFP16Type);
TYPED_TEST(DepthwiseConvolutionTester, Simple) {
    float input[] = {1,
                     7,  // channel1
                     3,
                     9,
                     5,
                     11,
                     2,
                     8,  // channel2
                     4,
                     10,
                     6,
                     12};
    float filter[] = {
        1,
        -9,
        5,
        13,
        2,
        10,
        6,
        -14,
        3,
        -11,
        7,
        15,
        4,
        12,
        8,
        -16,
    };
    float bias[] = {1, 2, 3, 4};
    float reference_output[] = {
        71,
        91,
        -34,
        -26,
        99,
        127,
        -20,
        -4,
    };
    this->TestPrepare({1, 2, 3, 2}, {4, 1, 2, 2}, {4, 1, 1, 1}, {1, 4, 2, 1}, {1, 1}, {0, 0, 0, 0}, 2)
        .TestRun(input, filter, bias, reference_output);
}

TYPED_TEST(DepthwiseConvolutionTester, NoBias) {
    float input[] = {1,
                     7,  // channel1
                     3,
                     9,
                     5,
                     11,
                     2,
                     8,  // channel2
                     4,
                     10,
                     6,
                     12};
    float filter[] = {
        1,
        -9,
        5,
        13,
        2,
        10,
        6,
        -14,
        3,
        -11,
        7,
        15,
        4,
        12,
        8,
        -16,
    };
    float reference_output[] = {70.0, 90.0, -36.0, -28.0, 96.0, 124.0, -24.0, -8.0};
    this->TestPrepare({1, 2, 3, 2}, {4, 1, 2, 2}, {0, 0, 0, 0}, {1, 4, 2, 1}, {1, 1}, {0, 0, 0, 0}, 2)
        .TestRun(input, filter, nullptr, reference_output);
}

TYPED_TEST(DepthwiseConvolutionTester, Batch) {
    float input[] = {1, 7,                // batch1, channel1
                     3, 9,  5, 11, 2, 8,  // channel2
                     4, 10, 6, 12, 1, 7,  // batch2, channel1
                     3, 9,  5, 11, 2, 8,  // channel2
                     4, 10, 6, 12};
    float filter[] = {
        1,
        -9,
        5,
        13,
        2,
        10,
        6,
        -14,
        3,
        -11,
        7,
        15,
        4,
        12,
        8,
        -16,
    };
    float bias[] = {1, 2, 3, 4};
    float reference_output[] = {
        71,
        91,
        -34,
        -26,
        99,
        127,
        -20,
        -4,
        71,
        91,
        -34,
        -26,
        99,
        127,
        -20,
        -4,
    };
    this->TestPrepare({2, 2, 3, 2}, {4, 1, 2, 2}, {4, 1, 1, 1}, {2, 4, 2, 1}, {1, 1}, {0, 0, 0, 0}, 2)
        .TestRun(input, filter, bias, reference_output);
}

float input_128[] = {0.0,   1.0,   2.0,   3.0,   4.0,   5.0,   6.0,   7.0,   8.0,   9.0,   10.0,  11.0,  12.0,  13.0,  14.0,
                     15.0,  16.0,  17.0,  18.0,  19.0,  20.0,  21.0,  22.0,  23.0,  24.0,  25.0,  26.0,  27.0,  28.0,  29.0,
                     30.0,  31.0,  32.0,  33.0,  34.0,  35.0,  36.0,  37.0,  38.0,  39.0,  40.0,  41.0,  42.0,  43.0,  44.0,
                     45.0,  46.0,  47.0,  48.0,  49.0,  50.0,  51.0,  52.0,  53.0,  54.0,  55.0,  56.0,  57.0,  58.0,  59.0,
                     60.0,  61.0,  62.0,  63.0,  64.0,  65.0,  66.0,  67.0,  68.0,  69.0,  70.0,  71.0,  72.0,  73.0,  74.0,
                     75.0,  76.0,  77.0,  78.0,  79.0,  80.0,  81.0,  82.0,  83.0,  84.0,  85.0,  86.0,  87.0,  88.0,  89.0,
                     90.0,  91.0,  92.0,  93.0,  94.0,  95.0,  96.0,  97.0,  98.0,  99.0,  100.0, 101.0, 102.0, 103.0, 104.0,
                     105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.0, 119.0,
                     120.0, 121.0, 122.0, 123.0, 124.0, 125.0, 126.0, 127.0};

TYPED_TEST(DepthwiseConvolutionTester, Kernel3Stride1) {
    float filter[] = {1, -9, 5, 13, 2, 7, 6, 20, -3, 2, 10, 6, -14, -2, -12, 4, 6, 13};
    float bias[] = {1, 2};
    float reference_output[] = {
        124.0, 192.0,  237.0,  282.0,  327.0,  372.0,  410.0,  311.0, 508.0,  550.0,  592.0,  634.0,  676.0,  648.0,
        465.0, 802.0,  844.0,  886.0,  928.0,  970.0,  879.0,  619.0, 1096.0, 1138.0, 1180.0, 1222.0, 1264.0, 1110.0,
        773.0, 1390.0, 1432.0, 1474.0, 1516.0, 1558.0, 1341.0, 927.0, 1684.0, 1726.0, 1768.0, 1810.0, 1852.0, 1572.0,
        251.0, 837.0,  856.0,  875.0,  894.0,  913.0,  379.0,  136.0, 169.0,  164.0,  159.0,  154.0,  149.0,  46.0,
        177.0, 156.0,  169.0,  182.0,  195.0,  208.0,  74.0,   324.0, 247.0,  260.0,  273.0,  286.0,  299.0,  116.0,
        471.0, 338.0,  351.0,  364.0,  377.0,  390.0,  158.0,  618.0, 429.0,  442.0,  455.0,  468.0,  481.0,  200.0,
        765.0, 520.0,  533.0,  546.0,  559.0,  572.0,  242.0,  -32.0, -548.0, -558.0, -568.0, -578.0, -588.0, -262.0};
    this->TestPrepare({1, 1, 7, 7}, {2, 1, 3, 3}, {2, 1, 1, 1}, {1, 2, 7, 7}, {1, 1}, {1, 1, 1, 1}, 2)
        .TestRun(input_128, filter, bias, reference_output);
}

TYPED_TEST(DepthwiseConvolutionTester, DepthMultiplier1) {
    float filter[] = {
        1,
        -9,
        5,
        13,
        2,
        10,
        6,
        -14,
    };
    float bias[] = {1, 2};
    float reference_output[] = {41.0, 61.0, 6.0, 14.0};
    this->TestPrepare({1, 2, 3, 2}, {2, 1, 2, 2}, {2, 1, 1, 1}, {1, 2, 2, 1}, {1, 1}, {0, 0, 0, 0}, 1)
        .TestRun(input_128, filter, bias, reference_output);
}

TYPED_TEST(DepthwiseConvolutionTester, NoPaddingEqualStrideEqualKernel) {
    float filter[] = {1, -2, 3, -4, -5, 6, 7, 8, -9, 10, -11, 12, -13, 14, 15, 16, -17, -18};
    float bias[] = {1, 2};
    float reference_output[] = {74.0,
                                84.0,
                                94.0,
                                154.0,
                                164.0,
                                174.0,
                                234.0,
                                244.0,
                                254.0,
                                342.0,
                                358.0,
                                374.0,
                                470.0,
                                486.0,
                                502.0,
                                598.0,
                                614.0,
                                630.0};
    this->TestPrepare({1, 2, 8, 8}, {2, 1, 3, 3}, {2, 1, 1, 1}, {1, 2, 3, 3}, {2, 2}, {0, 0, 0, 0}, 1)
        .TestRun(input_128, filter, bias, reference_output);
}

TYPED_TEST(DepthwiseConvolutionTester, NoPaddingEqualStrideUnequalKernel) {
    float filter[] = {1, -2, 3, -4, -5, 6, 7, 8, -9, 10, -11, 12};
    float bias[] = {1, 2};
    float reference_output[] = {9.0,    7.0,    5.0,    3.0,    -7.0,   -9.0,   -11.0,  -13.0,
                                -23.0,  -25.0,  -27.0,  -29.0,  1144.0, 1178.0, 1212.0, 1246.0,
                                1416.0, 1450.0, 1484.0, 1518.0, 1688.0, 1722.0, 1756.0, 1790.0};
    this->TestPrepare({1, 2, 8, 8}, {2, 1, 3, 2}, {2, 1, 1, 1}, {1, 2, 3, 4}, {2, 2}, {0, 0, 0, 0}, 1)
        .TestRun(input_128, filter, bias, reference_output);
}

TYPED_TEST(DepthwiseConvolutionTester, EqualPaddingEqualStrideEqualKernel) {
    float filter[] = {1, -2, 3, -4, -5, 6, 7, 8, -9, 10, -11, 12, -13, 14, 15, 16, -17, -18};
    float bias[] = {1, 2};
    float reference_output[] = {-9.0,  43.0,  49.0,   139.0, 1.0,    105.0,  115.0,  209.0,  15.0,   175.0,  185.0,
                                279.0, 87.0,  -45.0,  -47.0, -469.0, -540.0, -290.0, -296.0, -8.0,   -556.0, 308.0,
                                324.0, -80.0, -626.0, 420.0, 436.0,  -94.0,  2752.0, 2466.0, 2520.0, 12.0};
    this->TestPrepare({1, 2, 7, 7}, {2, 1, 3, 3}, {2, 1, 1, 1}, {1, 2, 4, 4}, {2, 2}, {1, 1, 1, 1}, 1)
        .TestRun(input_128, filter, bias, reference_output);
}

TYPED_TEST(DepthwiseConvolutionTester, UnEqualPaddingEqualStrideEqualKernel) {
    float filter[] = {1, -2, 3, -4, -5, 6, 7, 8, -9, 10, -11, 12, 13, -14, 15, -16, -17, 18, 19, -20, -21, -22, 23, 24, -25};
    float bias[] = {1};
    float reference_output[] = {23.0, -159.0, 295.0, -193.0, 575.0, 595.0};
    this->TestPrepare({1, 1, 5, 4}, {1, 1, 5, 5}, {1, 1, 1, 1}, {1, 1, 3, 2}, {2, 2}, {2, 1, 2, 1}, 1)
        .TestRun(input_128, filter, bias, reference_output);
}

TYPED_TEST(DepthwiseConvolutionTester, UnEqualPaddingEqualStrideUnequalKernel) {
    float filter[] = {1, -2, 3, -4, -5, 6, 7, 8, -9, 10, -11, 12, 13, -14, 15, -16, -17, 18};
    float bias[] = {1};
    float reference_output[] = {
        79.0, 13.0, 45.0, -847.0, 123.0, 226.0, 256.0, -1247.0, 179.0, 436.0, 466.0, -1667.0, 83.0, 773.0, 805.0, 249.0};
    this->TestPrepare({1, 1, 8, 7}, {1, 1, 6, 3}, {1, 1, 1, 1}, {1, 1, 4, 4}, {2, 2}, {2, 1, 2, 1}, 1)
        .TestRun(input_128, filter, bias, reference_output);
}

TYPED_TEST(DepthwiseConvolutionTester, VTS_conv2d_float) {
    float input[] = {10, 21, 10, 22, 10, 23, 10, 24, 10, 25, 10, 26, 10, 27, 10, 28, 10, 29};
    float filter[] = {.25, 0, .2, 0, .25, 0, 0, .3, .25, 0, 0, 0, .25, .1, 0, 0};
    float bias[] = {1, 2, 3, 4};
    float reference_output[] = {11, 3, 7.2, 10.6, 11, 3, 7.4, 10.9, 11, 3, 7.8, 11.5, 11, 3, 8.0, 11.8};
    this->TestPrepare({1, 3, 3, 2}, {1, 2, 2, 4}, {4, 1, 1, 1}, {1, 2, 2, 4}, {1, 1}, {0, 0, 0, 0}, 2)
        .TestRunPumuteData(input, filter, bias, reference_output);
}

TYPED_TEST(DepthwiseConvolutionTester, VTS_conv2d_float_weights_as_inputs) {
    float input[] = {10, 21, 10, 22, 10, 23, 10, 24, 10, 25, 10, 26, 10, 27, 10, 28, 10, 29};
    float filter[] = {.25, 0, .2, 0, .25, 0, 0, .3, .25, 0, 0, 0, .25, .1, 0, 0};
    float bias[] = {1, 2, 3, 4};
    float reference_output[] = {11, 3, 7.2, 10.6, 11, 3, 7.4, 10.9, 11, 3, 7.8, 11.5, 11, 3, 8.0, 11.8};
    this->TestPrepare({1, 3, 3, 2}, {1, 2, 2, 4}, {4, 1, 1, 1}, {1, 2, 2, 4}, {1, 1}, {0, 0, 0, 0}, 2)
        .TestRunPumuteData(input, filter, bias, reference_output);
}

TYPED_TEST(DepthwiseConvolutionTester, VTS_conv2d_float_large) {
    float input[] = {10, 21, 10, 22, 10, 23, 10, 24};
    float filter[] = {.25, 0, .25, 1, .25, 0, .25, 1};
    float bias[] = {100, 200};
    float reference_output[] = {110, 246};
    this->TestPrepare({1, 2, 2, 2}, {1, 2, 2, 2}, {2, 1, 1, 1}, {1, 1, 1, 2}, {1, 1}, {0, 0, 0, 0}, 1)
        .TestRunPumuteData(input, filter, bias, reference_output);
}

TYPED_TEST(DepthwiseConvolutionTester, VTS_conv2d_float_large_weights_as_inputs) {
    float input[] = {10, 21, 10, 22, 10, 23, 10, 24};
    float filter[] = {.25, 0, .25, 1, .25, 0, .25, 1};
    float bias[] = {100, 200};
    float reference_output[] = {110, 246};
    this->TestPrepare({1, 2, 2, 2}, {1, 2, 2, 2}, {2, 1, 1, 1}, {1, 1, 1, 2}, {1, 1}, {0, 0, 0, 0}, 1)
        .TestRunPumuteData(input, filter, bias, reference_output);
}

TYPED_TEST(DepthwiseConvolutionTester, VTS_depthwise_conv_0) {
    float input[] = {
        -0.869931, 0.644628,   -0.918393, 0.153672,  0.868562,   -0.358177, -0.134931,  -0.247565,  0.22174,   -0.259157,
        -0.284296, -0.538065,  0.765559,  0.41986,   -0.556241,  0.658494,  0.214355,   -0.850169,  -0.252893, -0.478935,
        0.530526,  -0.0700663, -0.988729, -0.303061, 0.150845,   0.829915,  0.476349,   0.406537,   -0.355343, 0.757145,
        -0.356362, 0.800482,   -0.713861, 0.210483,  -0.634303,  0.718236,  -0.752038,  0.457547,   -0.550769, -0.551178,
        0.446766,  -0.227462,  0.216348,  -0.852806, -0.351486,  0.55906,   -0.668493,  -0.303493,  -0.363763, -0.162837,
        0.0701012, 0.756097,   -0.142269, 0.329724,  -0.656317,  -0.998086, -0.652949,  -0.40316,   -0.893682, 0.432744,
        0.612362,  -0.869588,  -0.71327,  -0.398092, -0.0423559, 0.436576,  -0.925272,  0.176549,   0.822904,  0.096833,
        -0.296802, -0.427195,  0.031654,  -0.254479, 0.244905,   0.0948254, 0.643769,   -0.90391,   0.352665,  -0.901179,
        0.266159,  -0.968068,  -0.615401, -0.388975, 0.939052,   -0.116289, 0.107523,   -0.0582711, 0.435172,  0.334675,
        0.459711,  0.717436,   0.496627,  -0.680175, -0.415066,  0.339848,  0.506004,   -0.337808,  -0.107218, -0.172496,
        0.870638,  0.931872,   -0.953884, 0.903042,  0.760078,   0.209727,  -0.285384,  -0.45514,   0.113194,  0.0756611,
        0.0924435, -0.472863,  0.960609,  -0.160385, -0.839445,  0.457097,  0.163348,   0.344867,   -0.131619, 0.688715,
        -0.540827, 0.571259,   -0.95587,  0.506164,  -0.155839,  0.0789621, 0.756772,   -0.662069,  0.242908,  0.460821,
        0.177872,  -0.289839,  -0.640603, 0.702598,  -0.506406,  -0.568262, -0.0713716, 0.413792,   0.159673,  -0.305208,
        0.133816,  -0.160254,  0.787323,  -0.753244, 0.600721,   0.263186,  -0.162387,  0.477962,   -0.702951, -0.731036,
        -0.939481, -0.524519,  0.934072,  -0.511637, -0.503499,  0.106236,  -0.323684,  0.534444,   -0.843745, 0.364171,
        0.0370358, -0.168801,  -0.404559, -0.814178, 0.91745,    -0.334276, 0.66925,    -0.801201,  0.156511,  -0.427949,
        0.379153,  0.818597,   -0.649902, 0.427087,  -0.586015,  -0.559789, -0.833923,  0.0892409,  -0.621251, 0.213826,
        0.465509,  0.4704,     0.380261,  0.413067,  0.180822,   0.172866,  0.59614,    0.825575,   0.662916,  -0.704381,
        -0.297631, 0.697778};
    float filter[] = {-0.966213, -0.467474, -0.82203};
    float bias[] = {0, 0, 0};
    float reference_output[] = {
        0.840539,   -0.301347, 0.754947,   -0.14848,  -0.40603,   0.294432,   0.130372,   0.11573,    -0.182277, 0.2504,
        0.132901,   0.442306,  -0.739693,  -0.196274, 0.457246,   -0.636246,  -0.100205,  0.698864,   0.244348,  0.22389,
        -0.436108,  0.067699,  0.462205,   0.249125,  -0.145748,  -0.387964,  -0.391573,  -0.392801,  0.166114,  -0.622396,
        0.344322,   -0.374205, 0.586815,   -0.203372, 0.29652,    -0.590411,  0.726629,   -0.213891,  0.452749,  0.532555,
        -0.208851,  0.186981,  -0.209039,  0.398664,  0.288932,   -0.540171,  0.312503,   0.24948,    0.351473,  0.076122,
        -0.0576253, -0.73055,  0.0665069,  -0.271043, 0.634142,   0.466579,   0.536743,   0.389538,   0.417773,  -0.355728,
        -0.591672,  0.40651,   0.586329,   0.384641,  0.0198003,  -0.358878,  0.894009,   -0.0825318, -0.676451, -0.0935613,
        0.138747,   0.351167,  -0.0305845, 0.118962,  -0.201319,  -0.0916215, -0.300945,  0.743041,   -0.34075,  0.421278,
        -0.218791,  0.935359,  0.287684,   0.319749,  -0.907324,  0.054362,   -0.0883874, 0.0563023,  -0.203432, -0.275113,
        -0.444178,  -0.335382, -0.408242,  0.657194,  0.194033,   -0.279365,  -0.488907,  0.157917,   0.0881365, 0.166668,
        -0.407001,  -0.766027, 0.921655,   -0.422149, -0.624807,  -0.202641,  0.13341,    0.374139,   -0.109369, -0.0353696,
        -0.0759913, 0.456887,  -0.44906,   0.131841,  0.811082,   -0.213681,  -0.134277,  -0.333215,  0.0615286, -0.566144,
        0.522554,   -0.267049, 0.785754,   -0.489062, 0.0728509,  -0.0649092, -0.731203,  0.3095,     -0.199677, -0.445251,
        -0.0831503, 0.238257,  0.618959,   -0.328446, 0.416281,   0.549062,   0.0333644,  -0.340149,  -0.154278, 0.142677,
        -0.110001,  0.15484,   -0.368053,  0.619189,  -0.580424,  -0.123033,  0.133487,   -0.461813,  0.328611,  0.600933,
        0.907739,   0.245199,  -0.767835,  0.49435,   0.235373,   -0.0873295, 0.312748,   -0.249839,  0.693584,  -0.351866,
        -0.0173133, 0.13876,   0.39089,    0.380607,  -0.754171,  0.322982,   -0.312857,  0.658611,   -0.151223, 0.200055,
        -0.311675,  -0.790939, 0.303812,   -0.351079, 0.566216,   0.261687,   0.68551,    -0.0862257, 0.290419,  -0.175771,
        -0.449781,  -0.2199,   -0.312586,  -0.399111, -0.0845297, -0.142101,  -0.575998,  -0.385935,  -0.544937, 0.680582,
        0.139135,   -0.573594};
    this->TestPrepare({1, 8, 8, 3}, {1, 1, 1, 3}, {3, 1, 1, 1}, {1, 8, 8, 3}, {1, 1}, {0, 0, 0, 0}, 1)
        .TestRunPumuteData(input, filter, bias, reference_output);
}

TYPED_TEST(DepthwiseConvolutionTester, VTS_depthwise_conv_1) {
    float input[] = {
        -0.295335,   -0.00387601, -0.552251,  0.166084,   -0.28482,    -0.152143, -0.719885, -0.869386, -0.745598,
        0.823947,    0.473183,    -0.331337,  0.187631,   0.0426571,   -0.826897, -0.755085, -0.472453, -0.0233656,
        0.0483436,   0.933418,    -0.961974,  0.0125783,  0.219742,    0.342604,  -0.15166,  0.0934905, 0.783221,
        0.129664,    0.838844,    -0.271388,  0.924519,   0.342843,    0.274418,  0.350817,  0.841638,  -0.543993,
        -0.00283395, -0.128467,   -0.682943,  -0.319117,  0.84634,     0.283003,  0.32865,   0.0293755, -0.0335696,
        0.591266,    -0.0743476,  -0.741271,  0.462056,   -0.583625,   -0.590183, 0.6234,    0.535269,  -0.670818,
        -0.955642,   -0.770173,   0.479986,   0.664377,   0.399445,    -0.968874, -0.276263, -0.901951, 0.544104,
        -0.958981,   0.482658,    -0.807284,  0.305369,   -0.947818,   0.827498,  -0.382887, -0.805741, -0.796678,
        -0.299804,   -0.229828,   0.818783,   -0.103055,  -0.45568,    -0.227827, 0.543743,  -0.96073,  0.946747,
        -0.857182,   -0.96426,    -0.292411,  -0.715614,  0.765278,    -0.475043, -0.590142, -0.238507, 0.673002,
        -0.473357,   -0.319626,   0.936014,   0.486607,   0.580844,    0.425352,  -0.800994, 0.290763,  -0.494953,
        -0.441162,   0.718677,    -0.828427,  0.96965,    7.53637e-05, -0.699973, -0.526886, -0.352682, 0.799466,
        0.332789,    0.723389,    0.407659,   -0.934084,  -0.284705,   0.961484,  -0.700395, -0.985808, -0.595342,
        -0.691721,   0.49448,     -0.0842649, 0.0390966,  0.298938,    -0.128094, -0.97158,  0.86393,   0.270606,
        -0.468986,   -0.256605,   0.47215,    -0.273117,  -0.590343,   -0.826529, -0.725381, -0.194821, -0.259661,
        -0.0949207,  -0.180302,   0.0446834,  -0.222133,  -0.40393,    0.295772,  -0.92949,  0.580079,  -0.169856,
        0.330311,    0.0173551,   -0.635823,  0.475942,   0.907175,    0.242777,  -0.512208, 0.362463,  0.0496289,
        0.65171,     0.990057,    0.690733,   -0.469013,  -0.101311,   -0.68372,  -0.157841, -0.677711, -0.708224,
        -0.659437,   -0.407607,   0.677033,   0.89032,    0.228307,    -0.749514, 0.772958,  0.054701,  0.551705,
        0.917052,    -0.895022,   -0.702397,  0.484142,   0.108648,    0.833347,  0.478872,  -0.984112, 0.387176,
        -0.73299,    0.7526,      0.443312,   -0.0987856, 0.125415,    0.10876,   -0.498108, 0.43209,   0.344609,
        0.928941,    -0.130732,   -0.0569167};
    float filter[] = {-0.966213, -0.467474, -0.82203};
    float bias[] = {0, 0, 0};
    float reference_output[] = {
        0.285357,   0.00181194,  0.453967,   -0.160473,  0.133146,     0.125066,   0.695562,  0.406415,   0.612903,
        -0.796108,  -0.221201,   0.272369,   -0.181291,  -0.0199411,   0.679734,   0.729573,  0.22086,    0.0192072,
        -0.0467102, -0.436349,   0.790771,   -0.0121533, -0.102724,    -0.281631,  0.146536,  -0.0437044, -0.643831,
        -0.125283,  -0.392138,   0.223089,   -0.893282,  -0.16027,     -0.22558,   -0.338964, -0.393444,  0.447179,
        0.0027382,  0.0600548,   0.5614,     0.308335,   -0.395642,    -0.232637,  -0.317546, -0.0137323, 0.0275952,
        -0.571289,  0.0347555,   0.609347,   -0.446445,  0.27283,      0.485148,   -0.602337, -0.250224,  0.551432,
        0.923353,   0.360036,    -0.394563,  -0.64193,   -0.18673,     0.796443,   0.266929,  0.421638,   -0.44727,
        0.926579,   -0.22563,    0.663612,   -0.295051,  0.44308,      -0.680228,  0.36995,   0.376663,   0.654893,
        0.289675,   0.107439,    -0.673064,  0.0995729,  0.213019,     0.18728,    -0.525372, 0.449116,   -0.778254,
        0.82822,    0.450766,    0.24037,    0.691436,   -0.357748,    0.3905,     0.570203,  0.111496,   -0.553228,
        0.457363,   0.149417,    -0.769431,  -0.470166,  -0.271529,    -0.349652,  0.773931,  -0.135924,  0.406866,
        0.426256,   -0.335963,   0.680992,   -0.936889,  -3.52306e-05, 0.575398,   0.509084,  0.16487,    -0.657185,
        -0.321545,  -0.338165,   -0.335108,  0.902524,   0.133092,     -0.790369,  0.676731,  0.46084,    0.489389,
        0.66835,    -0.231156,   0.0692682,  -0.0377757, -0.139746,    0.105297,   0.938753,  -0.403865,  -0.222446,
        0.45314,    0.119956,    -0.388121,  0.26389,    0.27597,      0.679432,   0.700873,  0.0910737,  0.213449,
        0.0917136,  0.0842865,   -0.0367311, 0.214628,   0.188827,     -0.243133,  0.898085,  -0.271172,  0.139627,
        -0.319151,  -0.00811307, 0.522665,   -0.459861,  -0.424081,    -0.19957,   0.494902,  -0.169442,  -0.0407964,
        -0.629691,  -0.462826,   -0.567803,  0.453167,   0.0473601,    0.562038,   0.152508,  0.316812,   0.582181,
        0.637157,   0.190546,    -0.556541,  -0.860239,  -0.106728,    0.616123,   -0.746842, -0.0255713, -0.453518,
        -0.886067,  0.418399,    0.577391,   -0.467784,  -0.05079,     -0.685036,  -0.462692, 0.460047,   -0.318271,
        0.708224,   -0.351821,   -0.364416,  0.0954479,  -0.0586282,   -0.0894044, 0.481278,  -0.201991,  -0.283279,
        -0.897555,  0.0611137,   0.0467872};
    this->TestPrepare({1, 8, 8, 3}, {1, 1, 1, 3}, {3, 1, 1, 1}, {1, 8, 8, 3}, {1, 1}, {0, 0, 0, 0}, 1)
        .TestRunPumuteData(input, filter, bias, reference_output);
}

TYPED_TEST(DepthwiseConvolutionTester, VTS_conv2d_float_large_2) {
    float input[] = {10, 21, 100, 0, 10, 22, 200, 0, 10, 23, 300, 0, 10, 24, 400, 0};
    float filter[] = {0.25, 0, 10, 100, 0.25, 1, 20, 100, 0.25, 0, 30, 100, 0.25, 1, 40, 100};
    float bias[] = {600000, 700000, 800000, 900000};
    float reference_output[] = {600010, 700046, 830000, 900000};
    PrecisionType precision = PrecisionType::FP32;
    this->SetComputePrecision(precision)
        .TestPrepare({1, 2, 2, 4}, {1, 2, 2, 4}, {4, 1, 1, 1}, {1, 1, 1, 4}, {1, 1}, {0, 0, 0, 0}, 1)
        .TestRunPumuteData(input, filter, bias, reference_output);
}

TYPED_TEST(DepthwiseConvolutionTester, VTS_large_2_weights_as_inputs) {
    float input[] = {10, 21, 100, 10, 22, 200, 10, 23, 300, 10, 24, 400};
    float filter[] = {0.25, 0, 10, 100, 0.25, 1, 20, 100, 0.25, 0, 30, 100, 0.25, 1, 40, 100};
    float bias[] = {600000, 700000, 800000, 900000};
    float reference_output[] = {600010, 700046, 830000, 900000};
    PrecisionType precision = PrecisionType::FP32;
    this->SetComputePrecision(precision)
        .TestPrepare({1, 2, 2, 3}, {1, 2, 2, 4}, {4, 1, 1, 1}, {1, 1, 1, 4}, {1, 1}, {0, 0, 0, 0}, 1)
        .TestRunPumuteData(input, filter, bias, reference_output);
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
