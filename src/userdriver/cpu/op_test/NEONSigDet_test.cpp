#include <gtest/gtest.h>
#include "userdriver/cpu/operators/NEONSigDet.hpp"
#include "userdriver/common/op_test/test_utils.h"

namespace enn {
namespace ud {
namespace cpu {

class SigDetectionTester {
public:
    explicit SigDetectionTester(float threshold) : error_threshold_(threshold) {}

    SigDetectionTester& TestPrepare(const uint32_t& num_classes, const bool& share_location, const float& nms_threshold,
                                    const int32_t& background_label_id, const int32_t& nms_top_k, const int32_t& keep_top_k,
                                    const uint32_t& code_type, const float& confidence_threshold, const float& nms_eta,
                                    const bool& variance_encoded_in_target) {
        num_classes_ = num_classes;
        share_location_ = share_location;
        nms_threshold_ = nms_threshold;
        background_label_id_ = background_label_id;
        nms_top_k_ = nms_top_k;
        keep_top_k_ = keep_top_k;
        code_type_ = code_type;
        confidence_threshold_ = confidence_threshold;
        nms_eta_ = nms_eta;
        variance_encoded_in_target_ = variance_encoded_in_target;
        // generate prior data
        in_prior_.reset(new float[num_ * 2 * num_priors_ * 4], std::default_delete<float[]>());
        const float step = 0.5;
        const float box_size = 0.3;
        int idx = 0;
        for (int h = 0; h < 2; ++h) {
            float center_y = (h + 0.5) * step;
            for (int w = 0; w < 2; ++w) {
                float center_x = (w + 0.5) * step;
                in_prior_.get()[idx++] = (center_x - box_size / 2);
                in_prior_.get()[idx++] = (center_y - box_size / 2);
                in_prior_.get()[idx++] = (center_x + box_size / 2);
                in_prior_.get()[idx++] = (center_y + box_size / 2);
            }
        }
        for (int i = 0; i < idx; ++i) {
            in_prior_.get()[idx + i] = 0.1;
        }

        // generate confidence data
        in_conf_.reset(new float[num_ * num_priors_ * num_classes_ * 1], std::default_delete<float[]>());
        idx = 0;
        for (uint32_t i = 0; i < num_; ++i) {
            for (uint32_t j = 0; j < num_priors_; ++j) {
                for (uint32_t c = 0; c < num_classes_; ++c) {
                    if (i % 2 == c % 2) {
                        in_conf_.get()[idx++] = j * 0.2;
                    } else {
                        in_conf_.get()[idx++] = 1 - j * 0.2;
                    }
                }
            }
        }

        // generate location data
        num_loc_classes_ = share_location_ ? 1 : num_classes_;
        in_loc_.reset(new float[num_ * num_priors_ * num_loc_classes_ * 4], std::default_delete<float[]>());
        idx = 0;
        for (uint32_t i = 0; i < num_; ++i) {
            for (int h = 0; h < 2; ++h) {
                for (int w = 0; w < 2; ++w) {
                    for (uint32_t c = 0; c < num_loc_classes_; ++c) {
                        in_loc_.get()[idx++] = (w % 2 ? -1 : 1) * (i * 1 + c / 2. + 0.5);
                        in_loc_.get()[idx++] = (h % 2 ? -1 : 1) * (i * 1 + c / 2. + 0.5);
                        in_loc_.get()[idx++] = (w % 2 ? -1 : 1) * (i * 1 + c / 2. + 0.5);
                        in_loc_.get()[idx++] = (h % 2 ? -1 : 1) * (i * 1 + c / 2. + 0.5);
                    }
                }
            }
        }
        return *this;
    }

    SigDetectionTester& SetBatch(const uint32_t& num) {
        num_ = num;
        return *this;
    }

    void TestRun(Dim4& output_dim, const float* output_expect, Status status = Status::SUCCESS) {
        loc_dim_ = {num_, num_priors_ * num_loc_classes_ * 4, 1, 1};
        auto input_tensor_loc = std::make_shared<NEONTensor<float>>(in_loc_.get(), loc_dim_, precision_);

        conf_dim_ = {num_, num_priors_ * num_classes_, 1, 1};
        auto input_tensor_conf = std::make_shared<NEONTensor<float>>(in_conf_.get(), conf_dim_, precision_);

        prior_dim_ = {num_, 2, num_priors_ * 4, 1};
        auto input_tensor_prior = std::make_shared<NEONTensor<float>>(in_prior_.get(), prior_dim_, precision_);

        output_dim_ = output_dim;
        out_size = GetDimSize(output_dim_);

        auto output_tensor = std::make_shared<NEONTensor<float>>(output_dim_, PrecisionType::FP32);

        NEONSigDet _sigdet(PrecisionType::FP32);

        EXPECT_EQ(_sigdet.initialize(loc_dim_, conf_dim_, prior_dim_, output_dim, num_classes_, share_location_,
                                     nms_threshold_, background_label_id_, nms_top_k_, keep_top_k_, code_type_,
                                     confidence_threshold_, nms_eta_, variance_encoded_in_target_),
                  Status::SUCCESS);
        EXPECT_EQ(_sigdet.execute(input_tensor_loc, input_tensor_conf, input_tensor_prior, output_tensor), Status::SUCCESS);
        EXPECT_EQ(_sigdet.release(), Status::SUCCESS);

        auto data_bytes = output_tensor->getNumOfBytes();
        auto output_ptr = make_shared_array<float>(data_bytes / sizeof(float));
        output_tensor->readData(output_ptr.get());

#ifdef NEON_OPT
        if (status == Status::SUCCESS) {
            Compare(output_expect, output_ptr.get(), out_size, error_threshold_);
        }
#endif
    }

private:
    PrecisionType precision_;
    float error_threshold_;
    uint32_t num_classes_;
    bool share_location_;
    float nms_threshold_;
    int32_t background_label_id_;
    int32_t nms_top_k_;
    int32_t keep_top_k_;
    uint32_t code_type_;
    float confidence_threshold_;
    float nms_eta_;
    bool variance_encoded_in_target_;

    uint32_t num_ = 2;
    uint32_t num_priors_ = 4;
    uint32_t num_loc_classes_;

    std::shared_ptr<float> in_conf_, in_loc_, in_prior_, eden_out_;
    Dim4 conf_dim_, loc_dim_, prior_dim_, output_dim_;
    size_t out_size = 0;
};

TEST(ENN_CPU_OP_UT_SigDetection, detection_test_1) {
    const uint32_t num_classes = 2;
    const bool share_location = true;
    const float nms_threshold = 0.45;
    const int32_t background_label_id = 0;
    const int32_t nms_top_k = 4;
    const int32_t keep_top_k = -1;
    const uint32_t code_type = 1;
    const float confidence_threshold = 0.23f;
    const float nms_eta = 1.0f;
    const bool variance_encoded_in_target = 0;
    uint32_t batch = 1;
    Dim4 output_dim = {batch, 1, nms_top_k, 7};
    const float output_expect[] = {
        0, 1, 1.0, 0.107309, 0.107309, 0.422691, 0.422691, 0, 1, 0.8, 0.592316, 0.107309, 0.877684, 0.422691,
        0, 1, 0.6, 0.107309, 0.592316, 0.422691, 0.877684, 0, 1, 0.4, 0.592316, 0.592316, 0.877684, 0.877684};

    SigDetectionTester(1e-5)
        .SetBatch(batch)
        .TestPrepare(num_classes, share_location, nms_threshold, background_label_id, nms_top_k, keep_top_k, code_type,
                     confidence_threshold, nms_eta, variance_encoded_in_target)
        .TestRun(output_dim, output_expect);
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
