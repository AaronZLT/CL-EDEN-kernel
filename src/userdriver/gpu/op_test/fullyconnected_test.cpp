#include <gtest/gtest.h>
#include "userdriver/common/op_test/test_utils.h"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/operators/CLFullyConnected.hpp"
#include "userdriver/gpu/op_test/test_function.hpp"

namespace enn {
namespace ud {
namespace gpu {

template <typename PRECISION> class FullyConnectedTester : public ::testing::Test {
  public:
    FullyConnectedTester() {
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

    FullyConnectedTester &TestPrepare(const Dim4 &input_dims,
                                      const Dim4 &output_dims,
                                      const Dim4 &weights_dims,
                                      const Dim4 &bias_dims,
                                      double min = 0.0f,
                                      double max = 0.0f) {
        input_dims_ = input_dims;
        output_dims_ = output_dims;
        weights_dims_ = weights_dims;
        bias_dims_ = bias_dims;
        bias_data_min_ = min;
        bias_data_max_ = max;
        return *this;
    }

    void TestRun(float *input_data, float *weights_data, float *bias_data, const float *reference_output_data) {
        size_t input_size = GetDimSize(input_dims_);
        size_t output_size = GetDimSize(output_dims_);
        size_t weights_size = GetDimSize(weights_dims_);
        size_t bias_size = GetDimSize(bias_dims_);
        auto in_ptr = make_shared_array<float>(input_size);
        memcpy(in_ptr.get(), input_data, input_size * sizeof(float));
        std::shared_ptr<ITensor> input_tensor = std::make_shared<CLTensor>(runtime_, precision_, in_ptr.get(), input_dims_);
        inputs_.push_back(input_tensor);

        auto fil_ptr = make_shared_array<float>(weights_size);
        memcpy(fil_ptr.get(), weights_data, weights_size * sizeof(float));
        std::shared_ptr<ITensor> weight_tensor =
            std::make_shared<CLTensor>(runtime_, precision_, fil_ptr.get(), weights_dims_);
        inputs_.push_back(weight_tensor);

        auto bias_ptr = make_shared_array<float>(bias_size);
        memcpy(bias_ptr.get(), bias_data, bias_size * sizeof(float));
        std::shared_ptr<ITensor> bias_tensor = std::make_shared<CLTensor>(runtime_, precision_, bias_ptr.get(), bias_dims_);
        inputs_.push_back(bias_tensor);

        std::shared_ptr<ITensor> output_tensor = std::make_shared<CLTensor>(runtime_, precision_, data_type_, output_dims_);
        outputs_.push_back(output_tensor);

        parameters_ = std::make_shared<FullyConnectedParameters>();
        parameters_->activation_info = std::make_shared<ActivationInfo>(ActivationInfo::ActivationType::NONE, false);

        CLFullyConnected fully_connected(runtime_, precision_);

        EXPECT_EQ(fully_connected.initialize(inputs_, outputs_, parameters_), Status::SUCCESS);
        EXPECT_EQ(fully_connected.execute(), Status::SUCCESS);
        EXPECT_EQ(fully_connected.release(), Status::SUCCESS);
        auto out_ptr = make_shared_array<float>(output_size);
        output_tensor->readData(out_ptr.get());
        Compare(out_ptr.get(), reference_output_data, output_size, error_threshold_);
    }

    void TestWithRandomInput() {
        std::vector<float> input(GetDimSize(input_dims_));
        std::vector<float> weights(GetDimSize(weights_dims_));
        std::vector<float> bias(GetDimSize(bias_dims_));
        std::vector<float> reference_output(GetDimSize(output_dims_));
        GenerateRandom<float>(input.data(), input.size(), -1, 1);
        GenerateRandom<float>(weights.data(), weights.size(), -1, 1);
        GenerateRandom<float>(bias.data(), bias.size(), bias_data_min_, bias_data_max_);
        FullyConnectedReference(input.data(), weights.data(), bias.data(), reference_output.data());
        TestRun(input.data(), weights.data(), bias.data(), reference_output.data());
    }

  private:
    void FullyConnectedReference(const float *input, const float *weight, const float *bias, float *output) {
        uint32_t batch_size = GetDimSize(input_dims_) / weights_dims_.c;
        uint32_t num_units = weights_dims_.n;
        uint32_t input_count = weights_dims_.c;
        for (uint32_t b = 0; b < batch_size; b++) {
            for (uint32_t out_c = 0; out_c < num_units; out_c++) {
                float value = bias[out_c];
                for (uint32_t in = 0; in < input_count; in++) {
                    value += input[b * input_count + in] * weight[out_c * input_count + in];
                }
                output[b * num_units + out_c] = value;
            }
        }
    }
    std::shared_ptr<CLRuntime> runtime_;
    std::vector<std::shared_ptr<ITensor>> inputs_;
    std::vector<std::shared_ptr<ITensor>> outputs_;
    std::shared_ptr<FullyConnectedParameters> parameters_;
    float error_threshold_;
    Dim4 input_dims_;
    Dim4 output_dims_;
    Dim4 weights_dims_;
    Dim4 bias_dims_;
    double bias_data_min_ = -1;
    double bias_data_max_ = 1;
    PrecisionType precision_;
    DataType data_type_;
};

TYPED_TEST_CASE(FullyConnectedTester, TestFP32AndFP16Type);
TYPED_TEST(FullyConnectedTester, Test8x1gemv) {
    this->TestPrepare({128, 4, 5, 6}, {128, 5, 1, 1}, {5, 4 * 5 * 6, 1, 1}, {5, 1, 1, 1}, -0.2, 0.2).TestWithRandomInput();
}

TYPED_TEST(FullyConnectedTester, Random4D1) {
    this->TestPrepare({128, 4, 8, 9}, {128, 5, 1, 1}, {5, 4 * 8 * 9, 8, 9}, {5, 1, 1, 1}).TestWithRandomInput();
}

TYPED_TEST(FullyConnectedTester, Random4D2) {
    this->TestPrepare({117, 4, 8, 9}, {117, 5, 1, 1}, {5, 4 * 8 * 9, 8, 9}, {5, 1, 1, 1}).TestWithRandomInput();
}

TYPED_TEST(FullyConnectedTester, SimpleTest) {
    float input_data[] = {
        1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // b = 0
        1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // b = 1
    };
    float weights_data[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
    };
    float bias_data[] = {1, 2, 3};
    float reference_output_data[] = {24, 25, 26, 58, 59, 60};
    this->TestPrepare({2, 10, 1, 1}, {2, 3, 1, 1}, {3, 10, 1, 1}, {3, 1, 1, 1})
        .TestRun(input_data, weights_data, bias_data, reference_output_data);
}

TYPED_TEST(FullyConnectedTester, SimpleTest4DTest) {
    float input_data[] = {
        1, 2, 3, 4, 5, 6, 7, 8,  -9, -10,  // first batch
        1, 2, 3, 4, 5, 6, 7, -8, 9,  -10,  // second batch
    };
    float weights_data[] = {
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 0
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  // u = 1
    };
    float bias_data[] = {1, 2, 3};
    float reference_output_data[] = {
        24,
        25,
        26,  // first batch
        58,
        59,
        60,  // second batch
    };
    this->TestPrepare({2, 1, 10, 1}, {2, 3, 1, 1}, {3, 10, 1, 1}, {3, 1, 1, 1})
        .TestRun(input_data, weights_data, bias_data, reference_output_data);
}

TYPED_TEST(FullyConnectedTester, Random4D) {
    this->TestPrepare({3, 3, 13, 13}, {3, 8, 1, 1}, {8, 3 * 13 * 13, 1, 1}, {8, 1, 1, 1}).TestWithRandomInput();
}

TYPED_TEST(FullyConnectedTester, Bias0) {
    this->TestPrepare({1, 64, 1, 1}, {1, 10, 1, 1}, {10, 64, 1, 1}, {10, 1, 1, 1}).TestWithRandomInput();
}

TYPED_TEST(FullyConnectedTester, Bias0_2) {
    this->TestPrepare({1, 64, 1, 1}, {1, 10, 1, 1}, {10, 64, 1, 1}, {10, 1, 1, 1}, 0.2, 0.2).TestWithRandomInput();
}

TYPED_TEST(FullyConnectedTester, MultipleFolder) {
    this->TestPrepare({2, 10, 2, 2}, {2, 25088, 1, 1}, {7 * 3584, 10 * 2 * 2, 1, 1}, {25088, 1, 1, 1}).TestWithRandomInput();
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
