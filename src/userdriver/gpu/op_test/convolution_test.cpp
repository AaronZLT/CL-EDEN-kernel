#include <memory>
#include <gtest/gtest.h>

#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/op_test/convolution_guard.h"
#include "userdriver/gpu/op_test/test_function.hpp"
#include "userdriver/gpu/operators/CLConvolution.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class ConvolutionTester : public ::testing::Test {
  public:
    ConvolutionTester() {
        runtime_ = std::shared_ptr<CLRuntime>(new CLRuntime());
        // here we passed FP16 as runtime precision, this method will be
        runtime_->initialize(0);
        runtime_->initializeQueue();
        precision_ = PRECISION::precision;
        data_type_ = DataType::FLOAT;
        parameters_.reset(new ConvolutionParameters);
    }
    inline ConvolutionTester &SetThreshold(float error_threshold = 0.0f) {
        if (PRECISION::precision == PrecisionType ::FP32) {
            error_threshold_ = 1e-4;
        } else if (PRECISION::precision == PrecisionType ::FP16) {
            //  For random ConvolutionTester/0.KernelThree TypeParam =(TestTypeFP16) test,
            //  the output is too bigger,need to set bigger error_threshold.
            error_threshold_ = error_threshold >= 0.17 ? error_threshold : 0.17;
        }
        return *this;
    }

    ConvolutionTester &TestPrepare(const Dim4 &input_dims,
                                   const Dim2 &padding,
                                   const Dim2 &stride,
                                   const Dim2 &kernel,
                                   const int &group,
                                   const Dim2 &dilation,
                                   ActivationInfo::ActivationType act_type,
                                   bool act_enabled) {
        parameters_->padding = {padding.h, padding.w, padding.h, padding.w};
        parameters_->dilation = dilation;
        parameters_->stride = stride;
        parameters_->group_size = group;
        parameters_->axis = 0;
        parameters_->per_channel_quant = false;
        parameters_->androidNN = false;
        parameters_->isNCHW = true;
        parameters_->storage_type = StorageType::BUFFER;
        parameters_->openAibWino = false;
        parameters_->activation_info = std::make_shared<ActivationInfo>(act_type, act_enabled);

        group_ = group;
        padding_ = padding;
        stride_ = stride;
        kernel_ = kernel;
        dilation_ = dilation;
        input_dims_ = input_dims;
        in_size = GetDimSize(input_dims);
        activation_type_ = act_type;
        activation_enabled_ = act_enabled;
        int output_batch, output_channel, output_height, output_width;
        convolutionGuard.PrepareConvolutionGuard(input_dims.n,
                                                 input_dims.c,
                                                 input_dims.h,
                                                 input_dims.w,
                                                 padding.h,
                                                 padding.w,
                                                 stride.h,
                                                 stride.w,
                                                 kernel.h,
                                                 kernel.w,
                                                 &output_batch,
                                                 &output_channel,
                                                 &output_height,
                                                 &output_width,
                                                 group,
                                                 dilation);
        output_dims_.n = output_batch;
        output_dims_.c = output_channel;
        output_dims_.h = output_height;
        output_dims_.w = output_width;
        out_size = GetDimSize(output_dims_);
        in.reset(new float[in_size], std::default_delete<float[]>());
        GenerateRandom<float>(in.get(), in_size, 0, 1);
        auto filter_dims = convolutionGuard.GetFilterShape();
        weight_dims_ = {(unsigned int)filter_dims[0],
                        (unsigned int)filter_dims[1],
                        (unsigned int)filter_dims[2],
                        (unsigned int)filter_dims[3]};
        auto filter_size = (unsigned int)filter_dims[0] * (unsigned int)filter_dims[1] * (unsigned int)filter_dims[2] *
                           (unsigned int)filter_dims[3];
        bias_dims_ = {1, (unsigned int)filter_dims[0], 1, 1};
        auto bias_size = (unsigned int)filter_dims[0];
        weight.reset(new float[filter_size], std::default_delete<float[]>());
        GenerateRandom<float>(weight.get(), filter_size, -1, 1);
        bias.reset(new float[bias_size], std::default_delete<float[]>());
        GenerateRandom<float>(bias.get(), bias_size, 0, 1);
        ground_truth_out.reset(new float[out_size], std::default_delete<float[]>());
        enn_out.reset(new float[out_size], std::default_delete<float[]>());
        return *this;
    }

    void TestRun() {
        std::shared_ptr<ITensor> input_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, input_dims_);
        std::shared_ptr<ITensor> weight_tensor =
            std::make_shared<CLTensor>(runtime_, precision_, weight.get(), weight_dims_);
        std::shared_ptr<ITensor> bias_tensor = std::make_shared<CLTensor>(runtime_, precision_, bias.get(), bias_dims_);

        inputs_.push_back(input_tensor);
        inputs_.push_back(weight_tensor);
        inputs_.push_back(bias_tensor);

        std::shared_ptr<ITensor> output_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, output_dims_);
        outputs_.push_back(output_tensor);

        // FIXME: need to support fused activation test case
        ActivationInfo activation_info(activation_type_, activation_enabled_);
        CLConvolution cl_convolution(runtime_, precision_);
        Status state = cl_convolution.initialize(inputs_, outputs_, parameters_);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "initilize failure\n");
        runtime_->assignBufferPool();
        EXPECT_EQ(input_tensor->writeData(in.get()), Status::SUCCESS);
        DEBUG_PRINT("after test init");
        state = cl_convolution.execute();
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "execute failure\n");
        convolutionGuard.ConvolutionGuardRun(
            in.get(), weight.get(), bias.get(), ground_truth_out.get(), activation_type_, activation_enabled_);
        auto data_bytes = output_tensor->getNumOfBytes();
        std::shared_ptr<float> output;
        int type_size = 0;
        if (precision_ == PrecisionType::FP32) {
            type_size = sizeof(cl_float);
        } else {
            type_size = sizeof(cl_half);
        }
        output.reset(new float[data_bytes / type_size]);
        output_tensor->readData(output.get());
        DEBUG_PRINT("before compare");
        Compare(ground_truth_out.get(), output.get(), out_size, error_threshold_);
        cl_convolution.release();
    }

    void TestRunWithExpect(float *input_data, float *filter_data, float *bias_data, const float *reference_output_data) {
        std::shared_ptr<ITensor> input_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, input_dims_);
        std::shared_ptr<ITensor> weight_tensor = std::make_shared<CLTensor>(runtime_, precision_, filter_data, weight_dims_);
        std::shared_ptr<ITensor> bias_tensor = std::make_shared<CLTensor>(runtime_, precision_, bias_data, bias_dims_);

        inputs_.push_back(input_tensor);
        inputs_.push_back(weight_tensor);
        inputs_.push_back(bias_tensor);

        std::shared_ptr<ITensor> output_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, output_dims_);
        outputs_.push_back(output_tensor);

        ActivationInfo activation_info(activation_type_, activation_enabled_);
        CLConvolution cl_convolution(runtime_, precision_);

        Pad4 padding = {padding_.h, padding_.w, padding_.h, padding_.w};
        int pad_h = (output_dims_.h - 1) * stride_.h + kernel_.h - input_dims_.h;
        int pad_w = (output_dims_.w - 1) * stride_.w + kernel_.w - input_dims_.w;
        if (padding_.h == 0 && padding_.w == 0 && (pad_h != 0 || pad_w != 0)) {
            padding.r = (pad_w + 1) / 2;
            padding.b = (pad_h + 1) / 2;
            padding.l = pad_w - padding.r;
            padding.t = pad_h - padding.b;
        }
        parameters_->padding = {padding.t, padding.r, padding.b, padding.l};
        Status state = cl_convolution.initialize(inputs_, outputs_, parameters_);
        runtime_->assignBufferPool();
        EXPECT_EQ(input_tensor->writeData(input_data), Status::SUCCESS);
        state = cl_convolution.execute();
        cl_convolution.release();
        uint32_t out_size = GetDimSize(output_dims_);
        std::shared_ptr<float> output;
        output.reset(new float[out_size]);
        output_tensor->readData(output.get());
        Compare(reference_output_data, output.get(), out_size, error_threshold_);
    }

    void TestRunPumuteData(float *input_data,
                           float *filters_data,
                           float *bias_data,
                           const float *output_data,
                           const Dim4 &input_dims,
                           const Dim4 &weight_dims,
                           const Dim4 &output_dims,
                           const Dim2 &padding,
                           const Dim2 &stride,
                           const Dim2 &kernel,
                           ActivationInfo::ActivationType act_type,
                           bool act_enabled) {
        input_dims_ = input_dims;
        weight_dims_ = weight_dims;
        output_dims_ = output_dims;
        padding_ = padding;
        stride_ = stride;
        kernel_ = kernel;
        group_ = 1;
        dilation_ = {1, 1};
        activation_type_ = act_type;
        activation_enabled_ = act_enabled;

        parameters_->padding = {padding.h, padding.w, padding.h, padding.w};
        parameters_->dilation = {1, 1};
        parameters_->stride = stride;
        parameters_->group_size = 1;
        parameters_->axis = 0;
        parameters_->per_channel_quant = false;
        parameters_->androidNN = false;
        parameters_->isNCHW = true;
        parameters_->storage_type = StorageType::BUFFER;
        parameters_->openAibWino = false;
        parameters_->activation_info = std::make_shared<ActivationInfo>(activation_type_, activation_enabled_);

        size_t input_size = GetDimSize(input_dims_);
        size_t output_size = GetDimSize(output_dims_);
        size_t filters_size = GetDimSize(weight_dims_);
        std::shared_ptr<float> input_nchw;
        input_nchw.reset(new float[input_size]);
        PermuteData(input_data, input_dims_, DataOrder::NHWC, input_nchw.get(), input_dims_);
        std::shared_ptr<float> filter_nchw;
        filter_nchw.reset(new float[filters_size]);
        PermuteData(filters_data, weight_dims_, DataOrder::NHWC, filter_nchw.get(), weight_dims_);
        std::shared_ptr<float> output_nchw;
        output_nchw.reset(new float[output_size]);
        PermuteData(output_data, output_dims_, DataOrder::NHWC, output_nchw.get(), output_dims_);
        bias_dims_ = {output_dims_.c, 1, 1, 1};
        TestRunWithExpect(input_nchw.get(), filter_nchw.get(), bias_data, output_nchw.get());
    }

  private:
    std::shared_ptr<CLRuntime> runtime_;
    float error_threshold_;

    size_t group_ = 0;
    size_t in_size = 0;
    size_t out_size = 0;
    std::shared_ptr<float> in, ground_truth_out, enn_out;
    std::shared_ptr<float> weight, bias;

    std::vector<std::shared_ptr<ITensor>> inputs_;
    std::vector<std::shared_ptr<ITensor>> outputs_;
    std::shared_ptr<ConvolutionParameters> parameters_;

    Dim4 input_dims_;
    Dim4 weight_dims_;
    Dim4 bias_dims_;
    Dim4 output_dims_;

    Dim2 padding_;
    Dim2 stride_;
    Dim2 kernel_;
    Dim2 dilation_;

    ConvolutionGuard convolutionGuard;
    PrecisionType precision_;
    DataType data_type_;

    ActivationInfo::ActivationType activation_type_;
    bool activation_enabled_;
};

TYPED_TEST_CASE(ConvolutionTester, TestFP32AndFP16Type);
TYPED_TEST(ConvolutionTester, SamePadKernelUnequal) {
    float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16};
    float weight_data[] = {1, 2, 3, 4, 5, 6};
    float bias_data[] = {0};
    float expect_data[] = {72, 90, 108, 52, 149, 170, 191, 88, 244, 265, 184, 44, 131, 141, 83, 12};
    this->SetThreshold().TestRunPumuteData(input_data,
                                           weight_data,
                                           bias_data,
                                           expect_data,
                                           {1, 4, 4, 1},
                                           {1, 3, 2, 1},
                                           {1, 4, 4, 1},
                                           {0, 0},
                                           {1, 1},
                                           {3, 2},
                                           ActivationInfo::ActivationType::RELU,
                                           false);
}

TYPED_TEST(ConvolutionTester, SamePadKernelUnequal_padh0) {
    float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16};
    float weight_data[] = {1, 2};
    float bias_data[] = {0};
    float expect_data[] = {5, 8, 11, 4, 17, 20, 23, 8, 29, 32, 35, 12, 44, 47, 16, 0};
    this->SetThreshold().TestRunPumuteData(input_data,
                                           weight_data,
                                           bias_data,
                                           expect_data,
                                           {1, 4, 4, 1},
                                           {1, 1, 2, 1},
                                           {1, 4, 4, 1},
                                           {0, 0},
                                           {1, 1},
                                           {1, 2},
                                           ActivationInfo::ActivationType::RELU,
                                           false);
}

TYPED_TEST(ConvolutionTester, SamePadKernelUnequal_padw0) {
    float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16};
    float weight_data[] = {1, 2};
    float bias_data[] = {0};
    float expect_data[] = {11, 14, 17, 20, 23, 26, 29, 32, 37, 40, 43, 12, 14, 15, 16, 0};
    this->SetThreshold().TestRunPumuteData(input_data,
                                           weight_data,
                                           bias_data,
                                           expect_data,
                                           {1, 4, 4, 1},
                                           {1, 2, 1, 1},
                                           {1, 4, 4, 1},
                                           {0, 0},
                                           {1, 1},
                                           {2, 1},
                                           ActivationInfo::ActivationType::RELU,
                                           false);
}

TYPED_TEST(ConvolutionTester, HandCalculatedWithBiasFloat32) {
    float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float weight_data[] = {1, 4, 7, 2, 5, 8, 3, 6, 9};
    float bias_data[] = {10};
    float expect_data[] = {115, 160, 193, 105, 245, 322, 367, 188, 197, 244, 271, 131};
    this->SetThreshold().TestRunPumuteData(input_data,
                                           weight_data,
                                           bias_data,
                                           expect_data,
                                           {1, 3, 4, 1},
                                           {1, 3, 3, 1},
                                           {1, 3, 4, 1},
                                           {0, 0},
                                           {1, 1},
                                           {3, 3},
                                           ActivationInfo::ActivationType::RELU,
                                           false);
}

TYPED_TEST(ConvolutionTester, VTS_conv_float) {
    float input_data[] = {1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0};
    float weight_data[] = {.25, .25, .25, .25};
    float bias_data[] = {0};
    float expect_data[] = {.875, .875, .875, .875};
    this->SetThreshold().TestRunPumuteData(input_data,
                                           weight_data,
                                           bias_data,
                                           expect_data,
                                           {1, 3, 3, 1},
                                           {1, 2, 2, 1},
                                           {1, 2, 2, 1},
                                           {0, 0},
                                           {1, 1},
                                           {2, 2},
                                           ActivationInfo::ActivationType::RELU,
                                           false);
}

TYPED_TEST(ConvolutionTester, VTS_conv_float_2) {
    float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    float weight_data[] = {1, 4, 7, 2, 5, 8, 3, 6, 9};
    float bias_data[] = {-200};
    float expect_data[] = {0, 0, 0, 0, 35, 112, 157, 0, 0, 34, 61, 0};
    this->SetThreshold().TestRunPumuteData(input_data,
                                           weight_data,
                                           bias_data,
                                           expect_data,
                                           {1, 3, 4, 1},
                                           {1, 3, 3, 1},
                                           {1, 3, 4, 1},
                                           {1, 1},
                                           {1, 1},
                                           {3, 3},
                                           ActivationInfo::ActivationType::RELU,
                                           true);
}

TYPED_TEST(ConvolutionTester, VTS_conv_float_channels) {
    float input_data[] = {99.0, 99.0, 99.0};
    float weight_data[] = {1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0};
    float bias_data[] = {0., 0., 0.};
    float expect_data[] = {297., 594., 891.};
    this->SetThreshold().TestRunPumuteData(input_data,
                                           weight_data,
                                           bias_data,
                                           expect_data,
                                           {1, 1, 1, 3},
                                           {3, 1, 1, 3},
                                           {1, 1, 1, 3},
                                           {0, 0},
                                           {1, 1},
                                           {1, 1},
                                           ActivationInfo::ActivationType::RELU,
                                           false);
}

TYPED_TEST(ConvolutionTester, VTS_conv_float_large) {
    float input_data[] = {1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.};
    float weight_data[] = {1., 4., 7., 2., 5., 8., 3., 6., 9.};
    float bias_data[] = {0., 0., 0.};
    float expect_data[] = {
        30., 36., 42., 66., 81., 96., 102., 126., 150., 138., 171., 204., 174., 216., 258., 210., 261., 312.};
    this->SetThreshold().TestRunPumuteData(input_data,
                                           weight_data,
                                           bias_data,
                                           expect_data,
                                           {1, 2, 3, 3},
                                           {3, 1, 1, 3},
                                           {1, 2, 3, 3},
                                           {0, 0},
                                           {1, 1},
                                           {1, 1},
                                           ActivationInfo::ActivationType::RELU,
                                           false);
}

TYPED_TEST(ConvolutionTester, VTS_conv1_h3_w2_same_0) {
    float input_data[] = {
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
    float weight_data[] = {-0.966213,
                           -0.467474,
                           -0.82203,
                           -0.579455,
                           0.0278809,
                           -0.79946,
                           -0.684259,
                           0.563238,
                           0.37289,
                           0.738216,
                           0.386045,
                           -0.917775,
                           0.184325,
                           -0.270568,
                           0.82236,
                           0.0973683,
                           -0.941308,
                           -0.144706};
    float bias_data[] = {0};
    float expect_data[] = {1.85284,   -0.0393656, -0.127353,  1.43115,   -0.302294, -1.0402,   0.655023,  -0.587614,
                           1.72003,   1.55816,    0.667546,   2.23663,   0.0661516, 0.290254,  0.770222,  -0.346357,
                           -1.58197,  -0.850595,  -0.484224,  0.949967,  -0.577263, -0.871949, 2.34132,   -0.104506,
                           -0.135965, -0.985713,  0.815147,   1.03114,   -1.41915,  -0.515534, -0.373639, 1.42026,
                           -1.50604,  0.673113,   3.06139,    -0.388578, -1.76707,  -0.315667, -1.03815,  -0.343435,
                           0.432787,  -1.41643,   1.12944,    -0.175806, -0.846415, 1.40095,   0.70832,   -1.46717,
                           2.19562,   -2.61266,   -0.705383,  1.26124,   1.46545,   -2.35761,  2.04494,   1.23741,
                           -0.527402, -0.39954,   -0.0128623, 1.3644,    0.985755,  -0.718118, -0.1008,   1.24327};
    this->SetThreshold().TestRunPumuteData(input_data,
                                           weight_data,
                                           bias_data,
                                           expect_data,
                                           {1, 8, 8, 3},
                                           {1, 3, 2, 3},
                                           {1, 8, 8, 1},
                                           {0, 0},
                                           {1, 1},
                                           {3, 2},
                                           ActivationInfo::ActivationType::RELU,
                                           false);
}

TYPED_TEST(ConvolutionTester, VTS_conv1_h3_w2_same_1) {
    float input_data[] = {
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
    float weight_data[] = {-0.966213,
                           -0.467474,
                           -0.82203,
                           -0.579455,
                           0.0278809,
                           -0.79946,
                           -0.684259,
                           0.563238,
                           0.37289,
                           0.738216,
                           0.386045,
                           -0.917775,
                           0.184325,
                           -0.270568,
                           0.82236,
                           0.0973683,
                           -0.941308,
                           -0.144706};
    float bias_data[] = {0};
    float expect_data[] = {-0.000614278, -1.21221,  0.443861,  0.102117,  -2.52714,  1.47489,   0.173474,  -0.237577,
                           1.28735,      1.91315,   2.51734,   0.375841,  0.637563,  2.653,     2.72959,   -1.6271,
                           1.17389,      -2.12119,  2.91417,   -2.24246,  0.0497045, -0.127107, -0.144473, -0.133762,
                           -0.393284,    -2.02346,  -0.239178, -0.246508, 1.29277,   1.32963,   0.117521,  1.22372,
                           0.0665713,    1.09438,   -1.31426,  2.52594,   -0.969211, 0.515478,  -1.60926,  -0.838905,
                           0.135211,     0.786415,  -1.14382,  -0.739102, -1.01731,  0.281615,  2.36311,   0.891823,
                           1.93872,      -0.150491, 3.45217,   2.28219,   1.18282,   -2.25086,  3.05468,   0.166228,
                           0.434554,     -2.57529,  -0.958662, -2.23978,  2.66776,   0.542601,  1.76107,   -1.08134};
    this->SetThreshold().TestRunPumuteData(input_data,
                                           weight_data,
                                           bias_data,
                                           expect_data,
                                           {1, 8, 8, 3},
                                           {1, 3, 2, 3},
                                           {1, 8, 8, 1},
                                           {0, 0},
                                           {1, 1},
                                           {3, 2},
                                           ActivationInfo::ActivationType::RELU,
                                           false);
}

TYPED_TEST(ConvolutionTester, VTS_conv1_h3_w2_valid_0) {
    float input_data[] = {
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
    float weight_data[] = {-0.966213,
                           -0.467474,
                           -0.82203,
                           -0.579455,
                           0.0278809,
                           -0.79946,
                           -0.684259,
                           0.563238,
                           0.37289,
                           0.738216,
                           0.386045,
                           -0.917775,
                           0.184325,
                           -0.270568,
                           0.82236,
                           0.0973683,
                           -0.941308,
                           -0.144706};
    float bias_data[] = {0};
    float expect_data[] = {1.72003,   1.55816,   0.667546,  2.23663,   0.0661516, 0.290254,  0.770222,  -1.58197, -0.850595,
                           -0.484224, 0.949967,  -0.577263, -0.871949, 2.34132,   -0.135965, -0.985713, 0.815147, 1.03114,
                           -1.41915,  -0.515534, -0.373639, -1.50604,  0.673113,  3.06139,   -0.388578, -1.76707, -0.315667,
                           -1.03815,  0.432787,  -1.41643,  1.12944,   -0.175806, -0.846415, 1.40095,   0.70832,  2.19562,
                           -2.61266,  -0.705383, 1.26124,   1.46545,   -2.35761,  2.04494};
    this->SetThreshold().TestRunPumuteData(input_data,
                                           weight_data,
                                           bias_data,
                                           expect_data,
                                           {1, 8, 8, 3},
                                           {1, 3, 2, 3},
                                           {1, 6, 7, 1},
                                           {0, 0},
                                           {1, 1},
                                           {3, 2},
                                           ActivationInfo::ActivationType::RELU,
                                           false);
}

TYPED_TEST(ConvolutionTester, VTS_conv1_h3_w2_valid_1) {
    float input_data[] = {
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
    float weight_data[] = {-0.966213,
                           -0.467474,
                           -0.82203,
                           -0.579455,
                           0.0278809,
                           -0.79946,
                           -0.684259,
                           0.563238,
                           0.37289,
                           0.738216,
                           0.386045,
                           -0.917775,
                           0.184325,
                           -0.270568,
                           0.82236,
                           0.0973683,
                           -0.941308,
                           -0.144706};
    float bias_data[] = {0};
    float expect_data[] = {1.28735,   1.91315,  2.51734,   0.375841,  0.637563,  2.653,     2.72959,  1.17389,   -2.12119,
                           2.91417,   -2.24246, 0.0497045, -0.127107, -0.144473, -0.393284, -2.02346, -0.239178, -0.246508,
                           1.29277,   1.32963,  0.117521,  0.0665713, 1.09438,   -1.31426,  2.52594,  -0.969211, 0.515478,
                           -1.60926,  0.135211, 0.786415,  -1.14382,  -0.739102, -1.01731,  0.281615, 2.36311,   1.93872,
                           -0.150491, 3.45217,  2.28219,   1.18282,   -2.25086,  3.05468};
    this->SetThreshold().TestRunPumuteData(input_data,
                                           weight_data,
                                           bias_data,
                                           expect_data,
                                           {1, 8, 8, 3},
                                           {1, 3, 2, 3},
                                           {1, 6, 7, 1},
                                           {0, 0},
                                           {1, 1},
                                           {3, 2},
                                           ActivationInfo::ActivationType::RELU,
                                           false);
}

TYPED_TEST(ConvolutionTester, VTS_conv3_h3_w2_same_0) {
    float input_data[] = {
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
    float weight_data[] = {-0.966213,    -0.579455, -0.684259, 0.738216,  0.184325,  0.0973683, -0.176863, -0.23936,
                           -0.000233404, 0.055546,  -0.232658, -0.316404, -0.012904, 0.320705,  -0.326657, -0.919674,
                           0.868081,     -0.824608, -0.467474, 0.0278809, 0.563238,  0.386045,  -0.270568, -0.941308,
                           -0.779227,    -0.261492, -0.774804, -0.79665,  0.22473,   -0.414312, 0.685897,  -0.327792,
                           0.77395,      -0.714578, -0.972365, 0.0696099, -0.82203,  -0.79946,  0.37289,   -0.917775,
                           0.82236,      -0.144706, -0.167188, 0.268062,  0.702641,  -0.412223, 0.755759,  0.721547,
                           -0.43637,     -0.274905, -0.269165, 0.16102,   0.819857,  -0.312008};
    float bias_data[] = {0, 0, 0};
    float expect_data[] = {
        -1.27853,   1.74987,   -0.876718,  0.989692,   0.298548,  0.522103,   -0.536896,  -0.179382,    -0.966914, 1.33708,
        1.37042,    -0.495494, 1.43859,    -1.548,     -0.430026, -0.662793,  -0.0867897, -0.900658,    -0.524396, 0.255731,
        -0.779081,  0.12666,   0.915651,   -0.444765,  -0.186842, -1.87308,   1.21135,    -0.385009,    1.72032,   -1.56036,
        -1.23059,   1.23694,   0.00200015, 0.359522,   1.60084,   0.434006,   -0.282945,  2.37292,      -1.28653,  0.0847837,
        -0.352093,  -2.39659,  0.149246,   0.920351,   -1.34346,  0.952311,   -0.35811,   0.403449,     0.484796,  -1.19989,
        -0.684298,  -1.41301,  0.103177,   -0.307039,  1.17741,   2.58936,    -2.76237,   -1.21565,     -1.09619,  1.17432,
        0.512143,   0.771379,  0.399879,   -0.0533093, 0.290864,  0.95563,    1.16328,    1.80768,      -1.52564,  -0.126476,
        -0.185224,  -0.114779, 1.2248,     0.237127,   -0.213297, -0.619941,  0.497944,   -1.68688,     1.59314,   -0.127337,
        0.111419,   1.13719,   1.68537,    -0.479644,  1.18608,   -2.52744,   1.34136,    0.548297,     -2.0838,   2.64585,
        -0.993354,  0.128238,  1.26092,    0.318668,   0.893795,  -0.0600559, -0.629126,  -0.949229,    2.25828,   -1.961,
        0.00589599, -0.187854, -1.02403,   0.396121,   1.3704,    3.99355,    0.434221,   0.274464,     -0.562438, -0.914871,
        0.539129,   -0.928687, 0.834954,   0.844178,   -0.566053, -0.957341,  0.933336,   1.13613,      -1.22109,  1.4649,
        -0.414666,  -0.452821, -0.706006,  -1.72657,   -0.726574, -0.0979362, -0.478669,  1.78703,      -0.639288, 1.48565,
        -0.179904,  1.01003,   -0.317118,  -0.675387,  1.90969,   -1.38343,   0.697255,   -0.292255,    1.81634,   0.717801,
        0.862479,   -0.407478, -0.343106,  -0.0353232, -0.481893, -0.135565,  -2.95941,   0.247846,     2.67757,   -2.23999,
        -0.519673,  0.254447,  0.415283,   -1.01065,   0.507911,  0.979926,   -0.184304,  -0.000950437, -0.734348, -0.196685,
        -0.713241,  0.594972,  0.0845042,  2.48496,    0.385019,  -0.201145,  0.533332,   -0.904872,    -0.333518, -0.581063,
        -2.07065,   0.118687,  -1.86708,   -0.601987,  0.432037,  1.73923,    0.590007,   0.419788,     0.314198,  2.12817,
        0.570793,   -1.15998,  -0.348587,  -1.10231,   -2.13091,  0.134467,   -0.460382,  0.138338,     3.455,     0.679068,
        -0.190282,  -0.0307461};
    this->SetThreshold().TestRunPumuteData(input_data,
                                           weight_data,
                                           bias_data,
                                           expect_data,
                                           {1, 8, 8, 3},
                                           {3, 3, 2, 3},
                                           {1, 8, 8, 3},
                                           {0, 0},
                                           {1, 1},
                                           {3, 2},
                                           ActivationInfo::ActivationType::RELU,
                                           false);
}

TYPED_TEST(ConvolutionTester, VTS_conv3_h3_w2_same_1) {
    float input_data[] = {
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
    float weight_data[] = {-0.966213,    -0.579455, -0.684259, 0.738216,  0.184325,  0.0973683, -0.176863, -0.23936,
                           -0.000233404, 0.055546,  -0.232658, -0.316404, -0.012904, 0.320705,  -0.326657, -0.919674,
                           0.868081,     -0.824608, -0.467474, 0.0278809, 0.563238,  0.386045,  -0.270568, -0.941308,
                           -0.779227,    -0.261492, -0.774804, -0.79665,  0.22473,   -0.414312, 0.685897,  -0.327792,
                           0.77395,      -0.714578, -0.972365, 0.0696099, -0.82203,  -0.79946,  0.37289,   -0.917775,
                           0.82236,      -0.144706, -0.167188, 0.268062,  0.702641,  -0.412223, 0.755759,  0.721547,
                           -0.43637,     -0.274905, -0.269165, 0.16102,   0.819857,  -0.312008};
    float bias_data[] = {0, 0, 0};
    float expect_data[] = {
        0.78574,   0.0700466, -0.110245, 0.0141003,  -0.621007, -0.979104, 1.24104,    0.580398,   -0.512997, 0.900559,
        -0.683229, -1.0162,   1.0089,    -0.0752488, 0.110969,  0.270558,  0.756819,   -0.10753,   -0.371484, 0.149005,
        0.0973829, 0.155766,  -0.476502, 0.259481,   1.06709,   -1.16534,  1.52694,    -0.797245,  0.802736,  -0.997109,
        2.2661,    -1.45548,  2.15506,   -1.33682,   1.15225,   -3.09324,  0.943457,   0.885211,   0.987944,  -0.345875,
        -0.114708, 1.7107,    0.104745,  0.828324,   -2.49964,  -0.453742, -0.288829,  -0.0948694, -0.489415, 1.74889,
        -0.378257, -2.10237,  0.613022,  -2.5225,    -0.746785, 3.63816,   -1.9287,    0.774279,   -0.613917, -0.650011,
        1.03753,   -0.177923, 0.891815,  -1.00373,   1.83859,   -1.59239,  -0.0662623, 0.218806,   -1.088,    0.280837,
        0.902901,  -1.90127,  3.04734,   -1.57302,   1.10881,   -0.980369, -3.85305,   -0.955859,  1.64909,   2.33573,
        0.31144,   -0.594375, 0.325747,  -0.952566,  -0.613449, 2.85073,   1.94692,    1.12977,    1.1351,    -0.449652,
        0.118765,  -0.199547, 2.873,     1.35182,    -1.85457,  1.22364,   1.38049,    2.38342,    0.882321,  1.03795,
        -0.321571, -2.60202,  -1.6372,   1.09302,    0.461768,  1.8485,    -0.158928,  4.28871,    -0.437375, -1.5794,
        1.59869,   0.0811864, 0.912054,  0.452176,   2.01812,   2.62907,   1.50304,    -0.840276,  -0.455854, -0.224913,
        0.609824,  -0.11105,  3.35635,   2.02386,    1.4687,    -0.708365, -0.508992,  -3.02602,   -0.75725,  1.85277,
        2.92817,   -0.172997, -1.13279,  -0.355636,  -0.337669, -0.588752, 2.05759,    1.0651,     0.884758,  -0.0712112,
        3.81319,   0.771629,  0.949634,  0.0838967,  -2.19264,  0.114521,  0.543556,   -1.63197,   -0.267442, 1.15701,
        -2.37862,  2.57646,   0.531208,  0.9499,     -0.231441, 1.51461,   1.58888,    0.895931,   -0.753084, 0.545251,
        0.746903,  0.012994,  -0.790398, -1.1055,    1.77789,   0.430923,  0.818241,   -0.731412,  0.979546,  -2.48707,
        -1.53658,  -1.66798,  -1.04585,  -0.667911,  1.00299,   -2.20339,  0.137826,   -2.31281,   0.755535,  0.495396,
        0.549629,  0.713128,  0.751369,  0.283996,   -0.814532, 1.4866,    1.12105,    0.927998,   0.517938,  -0.612661,
        -1.47756,  -1.42422};
    this->SetThreshold().TestRunPumuteData(input_data,
                                           weight_data,
                                           bias_data,
                                           expect_data,
                                           {1, 8, 8, 3},
                                           {3, 3, 2, 3},
                                           {1, 8, 8, 3},
                                           {0, 0},
                                           {1, 1},
                                           {3, 2},
                                           ActivationInfo::ActivationType::RELU,
                                           false);
}

TYPED_TEST(ConvolutionTester, VTS_conv3_h3_w2_valid_0) {
    float input_data[] = {
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
    float weight_data[] = {-0.966213,    -0.579455, -0.684259, 0.738216,  0.184325,  0.0973683, -0.176863, -0.23936,
                           -0.000233404, 0.055546,  -0.232658, -0.316404, -0.012904, 0.320705,  -0.326657, -0.919674,
                           0.868081,     -0.824608, -0.467474, 0.0278809, 0.563238,  0.386045,  -0.270568, -0.941308,
                           -0.779227,    -0.261492, -0.774804, -0.79665,  0.22473,   -0.414312, 0.685897,  -0.327792,
                           0.77395,      -0.714578, -0.972365, 0.0696099, -0.82203,  -0.79946,  0.37289,   -0.917775,
                           0.82236,      -0.144706, -0.167188, 0.268062,  0.702641,  -0.412223, 0.755759,  0.721547,
                           -0.43637,     -0.274905, -0.269165, 0.16102,   0.819857,  -0.312008};
    float bias_data[] = {0, 0, 0};
    float expect_data[] = {
        -0.186842, -1.87308,  1.21135,   -0.385009, 1.72032,   -1.56036,  -1.23059,   1.23694,    0.00200015,   0.359522,
        1.60084,   0.434006,  -0.282945, 2.37292,   -1.28653,  0.0847837, -0.352093,  -2.39659,   0.149246,     0.920351,
        -1.34346,  0.484796,  -1.19989,  -0.684298, -1.41301,  0.103177,  -0.307039,  1.17741,    2.58936,      -2.76237,
        -1.21565,  -1.09619,  1.17432,   0.512143,  0.771379,  0.399879,  -0.0533093, 0.290864,   0.95563,      1.16328,
        1.80768,   -1.52564,  1.2248,    0.237127,  -0.213297, -0.619941, 0.497944,   -1.68688,   1.59314,      -0.127337,
        0.111419,  1.13719,   1.68537,   -0.479644, 1.18608,   -2.52744,  1.34136,    0.548297,   -2.0838,      2.64585,
        -0.993354, 0.128238,  1.26092,   -0.629126, -0.949229, 2.25828,   -1.961,     0.00589599, -0.187854,    -1.02403,
        0.396121,  1.3704,    3.99355,   0.434221,  0.274464,  -0.562438, -0.914871,  0.539129,   -0.928687,    0.834954,
        0.844178,  -0.566053, -0.957341, 0.933336,  -0.414666, -0.452821, -0.706006,  -1.72657,   -0.726574,    -0.0979362,
        -0.478669, 1.78703,   -0.639288, 1.48565,   -0.179904, 1.01003,   -0.317118,  -0.675387,  1.90969,      -1.38343,
        0.697255,  -0.292255, 1.81634,   0.717801,  0.862479,  -0.481893, -0.135565,  -2.95941,   0.247846,     2.67757,
        -2.23999,  -0.519673, 0.254447,  0.415283,  -1.01065,  0.507911,  0.979926,   -0.184304,  -0.000950437, -0.734348,
        -0.196685, -0.713241, 0.594972,  0.0845044, 2.48496,   0.385019};
    this->SetThreshold().TestRunPumuteData(input_data,
                                           weight_data,
                                           bias_data,
                                           expect_data,
                                           {1, 8, 8, 3},
                                           {3, 3, 2, 3},
                                           {1, 6, 7, 3},
                                           {0, 0},
                                           {1, 1},
                                           {3, 2},
                                           ActivationInfo::ActivationType::RELU,
                                           false);
}

TYPED_TEST(ConvolutionTester, VTS_conv3_h3_w2_valid_1) {
    float input_data[] = {
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
    float weight_data[] = {-0.966213,    -0.579455, -0.684259, 0.738216,  0.184325,  0.0973683, -0.176863, -0.23936,
                           -0.000233404, 0.055546,  -0.232658, -0.316404, -0.012904, 0.320705,  -0.326657, -0.919674,
                           0.868081,     -0.824608, -0.467474, 0.0278809, 0.563238,  0.386045,  -0.270568, -0.941308,
                           -0.779227,    -0.261492, -0.774804, -0.79665,  0.22473,   -0.414312, 0.685897,  -0.327792,
                           0.77395,      -0.714578, -0.972365, 0.0696099, -0.82203,  -0.79946,  0.37289,   -0.917775,
                           0.82236,      -0.144706, -0.167188, 0.268062,  0.702641,  -0.412223, 0.755759,  0.721547,
                           -0.43637,     -0.274905, -0.269165, 0.16102,   0.819857,  -0.312008};
    float bias_data[] = {0, 0, 0};
    float expect_data[] = {
        1.06709,   -1.16534,  1.52694,   -0.797245,  0.802736,  -0.997109, 2.2661,    -1.45548,  2.15506,   -1.33682,
        1.15225,   -3.09324,  0.943457,  0.885211,   0.987944,  -0.345875, -0.114708, 1.7107,    0.104745,  0.828324,
        -2.49964,  -0.489415, 1.74889,   -0.378257,  -2.10237,  0.613022,  -2.5225,   -0.746785, 3.63816,   -1.9287,
        0.774279,  -0.613917, -0.650011, 1.03753,    -0.177923, 0.891815,  -1.00373,  1.83859,   -1.59239,  -0.0662623,
        0.218806,  -1.088,    3.04734,   -1.57302,   1.10881,   -0.980369, -3.85305,  -0.955859, 1.64909,   2.33573,
        0.31144,   -0.594375, 0.325747,  -0.952566,  -0.613449, 2.85073,   1.94692,   1.12977,   1.1351,    -0.449652,
        0.118765,  -0.199547, 2.873,     1.38049,    2.38342,   0.882321,  1.03795,   -0.321571, -2.60202,  -1.6372,
        1.09302,   0.461768,  1.8485,    -0.158928,  4.28871,   -0.437375, -1.5794,   1.59869,   0.0811864, 0.912054,
        0.452176,  2.01812,   2.62907,   1.50304,    0.609824,  -0.11105,  3.35635,   2.02386,   1.4687,    -0.708365,
        -0.508992, -3.02602,  -0.75725,  1.85277,    2.92817,   -0.172997, -1.13279,  -0.355636, -0.337669, -0.588752,
        2.05759,   1.0651,    0.884758,  -0.0712112, 3.81319,   -2.19264,  0.114521,  0.543556,  -1.63197,  -0.267442,
        1.15701,   -2.37862,  2.57646,   0.531208,   0.9499,    -0.231441, 1.51461,   1.58888,   0.895931,  -0.753084,
        0.545251,  0.746904,  0.0129939, -0.790398,  -1.1055,   1.77789};
    this->SetThreshold().TestRunPumuteData(input_data,
                                           weight_data,
                                           bias_data,
                                           expect_data,
                                           {1, 8, 8, 3},
                                           {3, 3, 2, 3},
                                           {1, 6, 7, 3},
                                           {0, 0},
                                           {1, 1},
                                           {3, 2},
                                           ActivationInfo::ActivationType::RELU,
                                           false);
}

TYPED_TEST(ConvolutionTester, TestDilationWithExpect) {
    float input_data[] = {1.31911,   -0.866488,  0.308415,  1.09341,   -0.384415,  1.34418,      0.493926,  0.0279442,
                          -1.69373,  -1.23856,   0.133287,  -0.467732, -0.931202,  -0.30099,     -2.01954,  -0.17786,
                          0.317349,  -1.07178,   -0.671862, -0.386432, 0.441684,   -1.20286,     0.944534,  2.06125,
                          -1.12709,  -2.56163,   0.648932,  -2.00772,  0.00850639, 0.434264,     0.119314,  0.246319,
                          -0.632024, -1.9752,    -0.459729, -1.0943,   -0.921944,  1.02151,      0.105924,  -1.77448,
                          0.903466,  -1.2146,    1.1624,    0.162934,  -0.824302,  0.589411,     -0.156408, 0.686487,
                          -2.4209,   -0.381299,  0.599984,  -0.627918, 0.680131,   0.731922,     0.599708,  0.383854,
                          1.5073,    0.00898065, 1.81667,   -0.441201, -1.7617,    -1.51393,     -0.772214, 1.49109,
                          2.09752,   1.41569,    -1.45942,  1.20186,   -0.64103,   -0.000832081, 2.01377,   -1.11404};
    float weight_data[] = {-0.279548, -0.34634,  0.0471391, 1.0397,    0.420015,  0.254657, -0.855998, -0.428272,
                           2.4481,    -0.432038, -0.580084, -1.24587,  -0.556019, 0.651616, 0.854585,  0.363918,
                           -1.33182,  -0.870305, -0.491281, 0.0379723, 0.615447,  -0.16647, -1.3608,   -0.32948,
                           1.59414,   -0.484352, -0.227221, -0.617096, 0.415827,  0.112331, 0.984269,  -0.0282865,
                           0.206713,  -0.417092, 0.320196,  -0.230233};
    float bias_data[] = {-0.150637, 1.12665, -0.144772};
    float expect_data[] = {1.16489,  0.46593,  -1.07674,  -0.127949, 1.43205,  2.12531,  2.86303,    2.38747,
                           -1.04393, -1.4291,  -1.09735,  1.48246,   -3.49068, -3.14799, -0.155215,  3.48414,
                           0.245992, -2.51949, -0.537462, 1.27051,   0.25811,  0.916805, -0.0647314, -2.02111};
    this->SetThreshold()
        .TestPrepare({2, 3, 3, 4}, {1, 1}, {2, 2}, {2, 2}, 1, {2, 2}, ActivationInfo::ActivationType::RELU, false)
        .TestRunWithExpect(input_data, weight_data, bias_data, expect_data);
}

TYPED_TEST(ConvolutionTester, Dilation2) {
    this->SetThreshold()
        .TestPrepare({2, 3, 56, 48}, {1, 1}, {2, 3}, {3, 4}, 1, {2, 2}, ActivationInfo::ActivationType::RELU, true)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, Dilation3) {
    this->SetThreshold()
        .TestPrepare({2, 3, 56, 48}, {1, 1}, {2, 3}, {3, 4}, 1, {3, 3}, ActivationInfo::ActivationType::RELU, true)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, WINOTest) {
    this->SetThreshold()
        .TestPrepare({2, 64, 50, 50}, {2, 3}, {1, 1}, {3, 3}, 1, {1, 1}, ActivationInfo::ActivationType::RELU1, true)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, WINOTestBirost2x8InputNot64) {
    this->SetThreshold()
        .TestPrepare({2, 63, 50, 50}, {1, 1}, {1, 1}, {3, 3}, 1, {1, 1}, ActivationInfo::ActivationType::RELU1, true)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, WINOTestBirost2x8InputNot8) {
    this->SetThreshold()
        .TestPrepare({2, 11, 100, 100}, {1, 1}, {1, 1}, {3, 3}, 1, {1, 1}, ActivationInfo::ActivationType::RELU6, true)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, WINOTestBirost2x8TopNot8InputNot64Test) {
    this->SetThreshold()
        .TestPrepare({2, 28, 51, 51}, {1, 1}, {1, 1}, {3, 3}, 1, {1, 1}, ActivationInfo::ActivationType::RELU6, true)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, WINOTestGeneral1x2Test) {
    this->SetThreshold()
        .TestPrepare({2, 4, 5, 5}, {1, 1}, {1, 1}, {3, 3}, 1, {1, 1}, ActivationInfo::ActivationType::TANH, true)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, WINOTestOutput8) {
    this->SetThreshold()
        .TestPrepare({3, 4, 43, 43}, {1, 1}, {1, 1}, {3, 3}, 1, {1, 1}, ActivationInfo::ActivationType::TANH, true)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, WINOTestOutput3) {
    this->SetThreshold()
        .TestPrepare({3, 4, 13, 13}, {0, 1}, {1, 1}, {3, 3}, 1, {1, 1}, ActivationInfo::ActivationType::SIGMOID, true)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, WINOTestOutput32) {
    this->SetThreshold()
        .TestPrepare({3, 12, 26, 26}, {1, 1}, {1, 1}, {3, 3}, 1, {1, 1}, ActivationInfo::ActivationType::SIGMOID, true)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, WINOTestOutput51) {
    this->SetThreshold()
        .TestPrepare({3, 4, 33, 33}, {1, 1}, {1, 1}, {3, 3}, 1, {1, 1}, ActivationInfo::ActivationType::RELU, true)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, WINOTestInput66Output8) {
    this->SetThreshold()
        .TestPrepare({3, 66, 25, 25}, {1, 1}, {1, 1}, {3, 3}, 1, {1, 1}, ActivationInfo::ActivationType::RELU, true)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, WINOTestInput66Output8_1) {
    this->SetThreshold()
        .TestPrepare({3, 66, 13, 13}, {1, 1}, {1, 1}, {3, 3}, 1, {1, 1}, ActivationInfo::ActivationType::RELU, true)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, WINOTestInput23Output51) {
    this->SetThreshold()
        .TestPrepare({3, 23, 51, 51}, {1, 1}, {1, 1}, {3, 3}, 1, {1, 1}, ActivationInfo::ActivationType::RELU, true)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, WINOTestInput66Output20) {
    this->SetThreshold()
        .TestPrepare({2, 66, 50, 50}, {1, 1}, {1, 1}, {3, 3}, 1, {1, 1}, ActivationInfo::ActivationType::RELU, true)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, WINOTestInput66Output20_group2) {
    this->SetThreshold()
        .TestPrepare({2, 66, 26, 26}, {1, 1}, {1, 1}, {3, 3}, 2, {1, 1}, ActivationInfo::ActivationType::RELU, true)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, WINOTestInput26Output24_group2) {
    this->SetThreshold()
        .TestPrepare({2, 66, 31, 31}, {1, 1}, {1, 1}, {3, 3}, 2, {1, 1}, ActivationInfo::ActivationType::RELU, true)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, WINOCommonTest0) {
    this->SetThreshold()
        .TestPrepare({3, 21, 31, 56}, {1, 1}, {1, 1}, {3, 3}, 1, {1, 1}, ActivationInfo::ActivationType::RELU, true)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, WINOCommonTest1) {
    this->SetThreshold()
        .TestPrepare({3, 22, 45, 32}, {1, 1}, {1, 1}, {3, 3}, 2, {1, 1}, ActivationInfo::ActivationType::RELU, true)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, WINOCommonTest2) {
    this->SetThreshold()
        .TestPrepare({3, 33, 28, 32}, {1, 1}, {1, 1}, {3, 3}, 1, {1, 1}, ActivationInfo::ActivationType::RELU, true)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, WINOCommonTest3) {
    this->SetThreshold()
        .TestPrepare({3, 33, 19, 18}, {0, 0}, {1, 1}, {3, 3}, 1, {1, 1}, ActivationInfo::ActivationType::RELU, true)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, WINOCommonTest4) {
    this->SetThreshold()
        .TestPrepare({3, 9, 19, 27}, {0, 1}, {1, 1}, {3, 3}, 1, {1, 1}, ActivationInfo::ActivationType::RELU, true)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, WINOCommonTest5) {
    this->SetThreshold()
        .TestPrepare({3, 9, 19, 27}, {0, 1}, {1, 1}, {3, 3}, 1, {1, 1}, ActivationInfo::ActivationType::RELU, true)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, WINOCommonTest6) {
    this->SetThreshold()
        .TestPrepare({3, 32, 101, 19}, {1, 0}, {1, 1}, {3, 3}, 1, {1, 1}, ActivationInfo::ActivationType::RELU, true)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, WINOCommonTest7) {
    this->SetThreshold()
        .TestPrepare({3, 32, 31, 31}, {0, 0}, {1, 1}, {3, 3}, 1, {1, 1}, ActivationInfo::ActivationType::RELU, true)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, WINOCommonTest9) {
    this->SetThreshold()
        .TestPrepare({3, 64, 26, 26}, {0, 0}, {1, 1}, {3, 3}, 1, {1, 1}, ActivationInfo::ActivationType::RELU, true)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, NoPaddingSimpleTest) {
    this->SetThreshold()
        .TestPrepare({1, 1, 8, 8}, {0, 0}, {2, 2}, {2, 2}, 1, {1, 1}, ActivationInfo::ActivationType::RELU, true)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, NoPaddingChannelIsOne) {
    this->SetThreshold()
        .TestPrepare({1, 1, 56, 56}, {0, 0}, {1, 1}, {5, 5}, 1, {1, 1}, ActivationInfo::ActivationType::RELU, true)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, NoPaddingChannelIs64) {
    this->SetThreshold()
        .TestPrepare({1, 64, 56, 56}, {0, 0}, {1, 1}, {1, 1}, 1, {1, 1}, ActivationInfo::ActivationType::RELU, true)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, PaddingChannelIsOne) {
    this->SetThreshold()
        .TestPrepare({1, 1, 56, 56}, {1, 1}, {1, 1}, {5, 5}, 1, {1, 1}, ActivationInfo::ActivationType::RELU, true)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, PaddingOneChannelIs64) {
    this->SetThreshold()
        .TestPrepare({1, 64, 56, 56}, {1, 1}, {1, 1}, {3, 3}, 1, {1, 1}, ActivationInfo::ActivationType::RELU, true)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, PaddingTwoChannelIs64) {
    this->SetThreshold()
        .TestPrepare({1, 16, 28, 28}, {2, 2}, {1, 1}, {5, 5}, 1, {1, 1}, ActivationInfo::ActivationType::RELU, false)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, KernelThree) {
    this->SetThreshold(0.4)
        .TestPrepare({1, 64, 256, 256}, {1, 1}, {1, 1}, {3, 3}, 1, {1, 1}, ActivationInfo::ActivationType::RELU, false)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, StrideThree) {
    this->SetThreshold()
        .TestPrepare({1, 3, 224, 224}, {3, 3}, {2, 2}, {7, 7}, 1, {1, 1}, ActivationInfo::ActivationType::RELU, false)
        .TestRun();
}

TYPED_TEST(ConvolutionTester, KernelEleven) {
    this->SetThreshold()
        .TestPrepare({1, 3, 224, 224}, {3, 3}, {4, 4}, {11, 11}, 1, {1, 1}, ActivationInfo::ActivationType::RELU, false)
        .TestRun();
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
