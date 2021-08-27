#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/operators/IDetection.hpp"
#include "userdriver/cpu/operators/neon_impl/NormalizedBbox.hpp"
#include "userdriver/cpu/common/NEONTensor.hpp"

namespace enn {
namespace ud {
namespace cpu {

class Detection : public IDetection {
public:
    explicit Detection(const PrecisionType &precision);

    Status initialize(const Dim4 &, const Dim4 &, const Dim4 &, const Dim4 &, const uint32_t &num_classes,
                      const bool &share_location, const float &nms_threshold, const int32_t &background_label_id,
                      const int32_t &nms_top_k, const int32_t &keep_top_k, const uint32_t &code_type,
                      const float &confidence_threshold, const float &nms_eta, const bool &variance_encoded_in_target);

    Status execute(const std::shared_ptr<ITensor> input_location, const std::shared_ptr<ITensor> input_confidence,
                   const std::shared_ptr<ITensor> input_prior, std::shared_ptr<ITensor> output);

    Status release();

private:
    PrecisionType precision_;

    typedef struct {
        uint32_t num_classes_ = 0;
        bool share_location_ = 1;
        float nms_threshold_ = 0.3;
        int32_t background_label_id_ = 0;
        int32_t nms_top_k_ = 0;
        int32_t keep_top_k_ = -1;
        uint32_t code_type_ = 0;
        float confidence_threshold_ = 0.0f;
        float nms_eta_ = 1.0f;
        bool variance_encoded_in_target_ = 0;
    } DetectionDescriptor;
    DetectionDescriptor descriptor_;

    void Detect(const uint32_t &num_input, const uint32_t &num_classes, const bool &share_location,
                const float &nms_threshold, const int32_t &background_label_id, const int32_t &top_k,
                const int32_t &keep_top_k, const PriorBoxCodingType &code_type, const float &confidence_threshold,
                const float &eta, const bool &variance_encoded_in_target, const uint32_t &num_priors, const float *loc_data,
                const float *conf_data, const float *prior_data, float *output_data, uint32_t &output_channel,
                uint32_t &output_height, uint32_t &output_width);
};

}  // namespace cpu
}  // namespace ud
}  // namespace enn
