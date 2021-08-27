#include <gtest/gtest.h>

#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/op_test/deconv_guard.h"
#include "userdriver/gpu/op_test/test_function.hpp"
#include "userdriver/gpu/operators/CLDeconvolution.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class DeconvolutionTester : public ::testing::Test {
public:
    DeconvolutionTester() {
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

    DeconvolutionTester &TestPrepare(const Dim4 &input_dims,
                                     const Dim4 &output_dims,
                                     const Dim4 &kernel_dims,
                                     const Dim4 &bias_dims,
                                     const Pad4 &pad_dims,
                                     const Dim2 &stride_dims,
                                     const int &group,
                                     bool act_enabled,
                                     ActivationInfo::ActivationType activation_type = ActivationInfo::ActivationType::NONE) {
        input_dims_ = input_dims;
        output_dims_ = output_dims;
        kernel_dims_ = kernel_dims;
        bias_dims_ = bias_dims;
        padding_ = pad_dims;
        stride_ = stride_dims;
        group_ = group;
        activation_enabled_ = act_enabled;
        activation_type_ = activation_type;

        return *this;
    }

    void TestRun(float *input_data, float *filter_data, float *bias_data, const float *reference_output_data) {
        uint32_t input_size = GetDimSize(input_dims_);
        auto in_ptr = make_shared_array<float>(input_size);
        memcpy(in_ptr.get(), input_data, input_size * sizeof(float));
        std::shared_ptr<CLTensor> input_tensor = std::make_shared<CLTensor>(runtime_, precision_, in_ptr.get(), input_dims_);

        uint32_t filter_size = GetDimSize(kernel_dims_);
        auto fil_ptr = make_shared_array<float>(filter_size);
        memcpy(fil_ptr.get(), filter_data, filter_size * sizeof(float));
        std::shared_ptr<CLTensor> filter_tensor =
            std::make_shared<CLTensor>(runtime_, precision_, fil_ptr.get(), kernel_dims_);

        uint32_t bias_size = GetDimSize(bias_dims_);
        auto bias_ptr = make_shared_array<float>(bias_size);
        memcpy(bias_ptr.get(), bias_data, bias_size * sizeof(float));
        std::shared_ptr<CLTensor> bias_tensor = std::make_shared<CLTensor>(runtime_, precision_, bias_ptr.get(), bias_dims_);
        auto output_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, output_dims_);

        parameters_ = std::make_shared<DeconvolutionParameters>();
        parameters_->padding = padding_;
        parameters_->stride = stride_;
        parameters_->group_size = group_;
        parameters_->weights_as_input = false;
        parameters_->per_channel_quant = false;
        parameters_->androidNN = false;
        parameters_->isNCHW = true;
        parameters_->openAibWino = false;
        parameters_->activation_info =
            std::make_shared<ActivationInfo>(activation_type_, activation_enabled_);

        CLDeconvolution deconv(runtime_, precision_);
        EXPECT_EQ(deconv.initialize({input_tensor, filter_tensor, bias_tensor}, {output_tensor}, parameters_),
                  Status::SUCCESS);
        runtime_->assignBufferPool();
        EXPECT_EQ(deconv.execute(), Status::SUCCESS);
        EXPECT_EQ(deconv.release(), Status::SUCCESS);

        size_t output_size = GetDimSize(output_dims_);
        auto output = make_shared_array<float>(output_size);
        output_tensor->readData(output.get());
        Compare(reference_output_data, output.get(), output_size, error_threshold_);
    }

    void TestRunWithRandom() {
        uint32_t input_size = GetDimSize(input_dims_);
        std::vector<float> input_data(input_size);
        GenerateRandom<float>(input_data.data(), input_size, -2, 2);
        uint32_t filter_size = GetDimSize(kernel_dims_);
        std::vector<float> filter_data(filter_size);
        GenerateRandom<float>(filter_data.data(), filter_size, -2, 2);
        uint32_t bias_size = GetDimSize(bias_dims_);
        std::vector<float> bias_data(bias_size);
        GenerateRandom<float>(bias_data.data(), bias_size, -1, 1);
        uint32_t out_size = GetDimSize(output_dims_);
        std::vector<float> expect_output(out_size);

        deconvGuard.PrepareDeconvGuard(input_dims_.n,
                                       input_dims_.c,
                                       input_dims_.h,
                                       input_dims_.w,
                                       padding_.b,
                                       padding_.r,
                                       stride_.h,
                                       stride_.w,
                                       kernel_dims_.h,
                                       kernel_dims_.w,
                                       group_,
                                       output_dims_.n,
                                       output_dims_.c,
                                       output_dims_.h,
                                       output_dims_.w);

        deconvGuard.DeconvGuardRun(input_data.data(),
                                   filter_data.data(),
                                   bias_data.data(),
                                   expect_output.data(),
                                   activation_type_,
                                   activation_enabled_);
        TestRun(input_data.data(), filter_data.data(), bias_data.data(), expect_output.data());
    }

private:
    std::shared_ptr<CLRuntime> runtime_;
    std::shared_ptr<DeconvolutionParameters> parameters_;
    float error_threshold_;
    Dim4 input_dims_;
    Dim4 output_dims_;
    Dim4 bias_dims_;
    Pad4 padding_;
    Dim2 stride_;
    Dim4 kernel_dims_;
    uint32_t group_;
    DeconvGuard deconvGuard;
    PrecisionType precision_;
    DataType data_type_;

    ActivationInfo::ActivationType activation_type_;
    bool activation_enabled_;
};

TYPED_TEST_CASE(DeconvolutionTester, TestFP32AndFP16Type);
TYPED_TEST(DeconvolutionTester, Stride2Kernel2pad1Group3) {
    float input_data[] = {1.86387,   0.823307,  0.172173,  -1.31165, 1.62649,    0.67635,  0.0448795, -0.286381,  0.583082,
                          0.884495,  -0.291902, -0.465355, -1.01752, 2.90559,    0.98394,  0.627009,  -1.42522,   1.11258,
                          1.37396,   -0.753714, 0.191197,  0.403568, -0.0168294, 0.250767, 0.942337,  -0.682471,  -1.78194,
                          -0.018691, -1.93659,  -1.06201,  0.481817, -0.283937,  0.445748, 2.0653,    0.00736003, 0.193207};
    float filter_data[] = {-0.188587,
                           0.707552,
                           -0.760529,
                           0.681911,
                           0.696376,
                           -0.277342,
                           -0.251224,
                           -0.284218,
                           0.497616,
                           1.84801,
                           0.444945,
                           0.176324};
    float bias_data[] = {0.936025, -0.29215, -2.92477};
    float expect_data[] = {2.20702,   0.309876,  1.49745,    0.805082,  0.00796517, 0.629291,  2.08685,   0.808474,
                           -0.304905, -0.220204, -0.210755,  -0.438634, -0.537457,  -0.495423, -0.211193, -0.616211,
                           -3.10419,  -1.63195,  -2.41245,   -2.48698,  -1.76606,   -3.63399,  -5.55859,  -2.37113,
                           1.87294,   1.50925,   0.42206,    0.790614,  1.22157,    0.939199,  0.924118,  0.888734,
                           -0.559979, -0.120697, -0.0981791, 0.155517,  -0.286966,  -1.64074,  0.244946,  -1.0317,
                           -2.83982,  -3.05111,  -2.97484,   -2.72644,  0.891926,   -2.92111,  -2.91117,  -2.82863};
    this->TestPrepare({2, 3, 2, 3}, {2, 3, 2, 4}, {3, 1, 2, 2}, {3, 1, 1, 1}, {1, 1, 1, 1}, {2, 2}, 3, false)
        .TestRun(input_data, filter_data, bias_data, expect_data);
}

TYPED_TEST(DeconvolutionTester, Stride3Kernel3pad1Group1) {
    float input_data[] = {1.86387,  0.823307,  0.172173,  -1.31165, 1.62649,    0.67635,  0.0448795, -0.286381, 0.583082,
                          0.884495, -0.291902, -0.465355, -1.01752, 2.90559,    0.98394,  0.627009,  -1.42522,  1.11258,
                          1.37396,  -0.753714, 0.191197,  0.403568, -0.0168294, 0.250767, 0.942337,  -0.682471, -1.79194};
    float filter_data[] = {-0.00881104, -0.912916,  -0.500634, 0.227131,  -0.133849, 0.210128, 0.973593,
                           0.00346955,  0.0957927,  -0.125725, 0.471701,  -0.507019, 0.454607, 0.464251,
                           -0.184894,   -0.167483,  -0.189479, 0.331744,  1.23201,   0.29663,  0.11755,
                           0.312008,    -0.0973832, -0.974925, -0.691566, 0.804975,  0.369564};
    float bias_data[] = {0.669277};
    float expect_data[] = {0.696627, -0.442116, 0.48841,   0.496962, 1.63106,  0.556485, 0.411571, 1.61415,   1.64901,
                           2.04098,  0.120722,  0.372762,  0.782617, 0.911958, 1.50645,  1.88927,  0.268908,  0.550009,
                           -1.62017, 0.848558,  0.590337,  0.333156, 0.188348, 2.35435,  1.80213,  0.490228,  1.34844,
                           1.01112,  1.18239,   0.355219,  1.77782,  0.110826, 1.78277,  0.989552, 0.687049,  1.20359,
                           0.439675, 0.0101773, 0.0560006, 1.45504,  -1.68342, 0.130236, 0.862591, -0.355931, -0.256619,
                           0.112412, 1.53797,   0.7484,    1.28225};
    this->TestPrepare({1, 3, 3, 3}, {1, 1, 7, 7}, {1, 3, 3, 3}, {1, 1, 1, 1}, {1, 1, 1, 1}, {3, 3}, 1, false)
        .TestRun(input_data, filter_data, bias_data, expect_data);
}

TYPED_TEST(DeconvolutionTester, Stride3Kernel3pad0Group1) {
    float input_data[] = {0.217164, -0.149747, -0.662173, 1.97792,   -0.461092, -0.883833, 0.0954008, -0.592692, -1.03494,
                          0.511883, -0.147536, 0.0809238, 0.17551,   0.367571,  0.0520366, -0.864747, 1.06924,   0.379317,
                          1.05455,  1.49745,   1.51393,   0.0274402, -0.610953, -0.494948, -0.729247, -2.99364,  -1.68334};
    float filter_data[] = {0.286498,  0.952974, 0.331824,  -1.08953,  0.0917794, 0.196174,  0.387382,  -0.598646, -0.363989,
                           0.0552936, 0.442186, 0.570585,  -0.509091, 0.104349,  -1.01063,  -0.631202, -0.465933, 0.446579,
                           -0.56836,  0.118115, -0.232822, 0.0287724, 0.40424,   -0.103365, -0.234958, 0.0427544, 0.092457};
    float bias_data[] = {0.115128};
    float expect_data[] = {
        -0.393717, 0.672986,  0.233737,  -0.78702,   0.0840563, -0.367382, -0.930567, -0.301303, -0.4109,
        -0.351732, 0.614766,  -0.468598, 0.396475,   0.691316,  0.0800715, 0.838945,  0.67479,   -0.253045,
        -0.371625, -0.208293, 0.362179,  -0.201594,  0.337537,  0.242197,  -0.548176, 0.538557,  0.532264,
        0.675906,  2.08088,   0.865203,  0.350592,   -0.233909, 0.3141,    0.146098,  -0.762593, -0.0332229,
        -2.12843,  0.326067,  0.322933,  0.412794,   -0.135807, -0.283653, 1.03736,   -0.160638, -0.059687,
        0.764109,  -1.14955,  -0.523897, -0.151954,  0.193774,  0.390622,  -0.143807, 0.598824,  0.41431,
        0.50912,   -0.262472, -0.176843, 1.70591,    -0.330483, 1.22554,   0.796338,  -0.902245, 0.380062,
        0.430439,  -0.261143, 1.08316,   0.130405,   -1.03784,  -0.772313, 1.00118,   -0.620752, -0.297251,
        0.869257,  0.429752,  -0.373199, -0.0859986, -0.156247, 0.531579,  -0.1297,   0.485984,  0.505593};
    this->TestPrepare({1, 3, 3, 3}, {1, 1, 9, 9}, {1, 3, 3, 3}, {1, 1, 1, 1}, {0, 0, 0, 0}, {3, 3}, 1, false)
        .TestRun(input_data, filter_data, bias_data, expect_data);
}

TYPED_TEST(DeconvolutionTester, Stride2kernel4pad1Group16Random) {
    this->TestPrepare({2, 16, 8, 10},
                      {2, 16, 16, 20},
                      {16, 1, 4, 4},
                      {16, 1, 1, 1},
                      {1, 1, 1, 1},
                      {2, 2},
                      16,
                      true,
                      ActivationInfo::ActivationType::RELU)
        .TestRunWithRandom();
}

TYPED_TEST(DeconvolutionTester, Stride23kernel34pad23Group2Random) {
    this->TestPrepare({2, 4, 8, 10},
                      {2, 4, 13, 25},
                      {4, 2, 3, 4},
                      {4, 1, 1, 1},
                      {2, 3, 2, 3},
                      {2, 3},
                      2,
                      true,
                      ActivationInfo::ActivationType::RELU)
        .TestRunWithRandom();
}

TYPED_TEST(DeconvolutionTester, Stride2Kernel4pad1Group1Random) {
    this->TestPrepare({2, 16, 26, 26},
                      {2, 19, 52, 52},
                      {19, 16, 4, 4},
                      {19, 1, 1, 1},
                      {1, 1, 1, 1},
                      {2, 2},
                      1,
                      true,
                      ActivationInfo::ActivationType::RELU)
        .TestRunWithRandom();
}

TYPED_TEST(DeconvolutionTester, Stride3Kernel7Pad0Group1Random) {
    this->TestPrepare({2, 16, 26, 26},
                      {2, 19, 52, 52},
                      {19, 16, 7, 7},
                      {19, 1, 1, 1},
                      {0, 0, 0, 0},
                      {3, 3},
                      1,
                      true,
                      ActivationInfo::ActivationType::RELU)
        .TestRunWithRandom();
}

TYPED_TEST(DeconvolutionTester, Stride4kernel11Pad2Group1Random) {
    this->TestPrepare({1, 16, 55, 55},
                      {1, 3, 223, 223},
                      {3, 16, 11, 11},
                      {3, 1, 1, 1},
                      {2, 2, 2, 2},
                      {4, 4},
                      1,
                      true,
                      ActivationInfo::ActivationType::RELU)
        .TestRunWithRandom();
}

TYPED_TEST(DeconvolutionTester, Stride2kernel4Pad1Group1GneralRandom) {
    this->TestPrepare({1, 32, 16, 20},
                      {1, 32, 32, 40},
                      {32, 32, 4, 4},
                      {32, 1, 1, 1},
                      {1, 1, 1, 1},
                      {2, 2},
                      1,
                      true,
                      ActivationInfo::ActivationType::RELU)
        .TestRunWithRandom();
}

TYPED_TEST(DeconvolutionTester, Stride2kernel4Pad1Group1Opt1x8Random) {
    this->TestPrepare({1, 32, 32, 40},
                      {1, 32, 64, 80},
                      {32, 32, 4, 4},
                      {32, 1, 1, 1},
                      {1, 1, 1, 1},
                      {2, 2},
                      1,
                      true,
                      ActivationInfo::ActivationType::RELU)
        .TestRunWithRandom();
}

TYPED_TEST(DeconvolutionTester, Deconv2TimesRandom) {
    this->TestPrepare({2, 32, 32, 40}, {2, 32, 64, 80}, {32, 32, 2, 2}, {32, 1, 1, 1}, {0, 0, 0, 0}, {2, 2}, 1, false)
        .TestRunWithRandom();
}

TYPED_TEST(DeconvolutionTester, DeconvMakaluRandom) {
    this->TestPrepare({2, 32, 32, 40}, {2, 32, 96, 120}, {32, 32, 3, 3}, {32, 1, 1, 1}, {0, 0, 0, 0}, {3, 3}, 1, false)
        .TestRunWithRandom();
}

TYPED_TEST(DeconvolutionTester, DeconvDensenetOne) {
    this->TestPrepare({1, 64, 28, 28}, {1, 64, 56, 56}, {64, 64, 4, 4}, {64, 1, 1, 1}, {1, 1, 1, 1}, {2, 2}, 64, false)
        .TestRunWithRandom();
}

TYPED_TEST(DeconvolutionTester, DeconvDensenetTwo) {
    this->TestPrepare({1, 64, 56, 56}, {1, 64, 112, 112}, {64, 64, 4, 4}, {64, 1, 1, 1}, {1, 1, 1, 1}, {2, 2}, 64, false)
        .TestRunWithRandom();
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
