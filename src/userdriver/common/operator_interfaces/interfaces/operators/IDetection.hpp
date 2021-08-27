#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/ITensor.hpp"

namespace enn {
namespace ud {

class IDetection {
public:
    virtual Status initialize(const Dim4 &location_dim, const Dim4 &confidence_dim, const Dim4 &prior_dim,
                              const Dim4 &output_dim, const uint32_t &num_classes, const bool &share_location,
                              const float &nms_threshold, const int32_t &background_label_id, const int32_t &nms_top_k,
                              const int32_t &keep_top_k, const uint32_t &code_type, const float &confidence_threshold,
                              const float &nms_eta, const bool &variance_encoded_in_target) = 0;
    virtual Status execute(const std::shared_ptr<ITensor> input_location, const std::shared_ptr<ITensor> input_confidence,
                           const std::shared_ptr<ITensor> input_prior, std::shared_ptr<ITensor> output) = 0;
    virtual Status release() = 0;
    virtual ~IDetection() = default;

private:
};  // class IDetection

}  // namespace ud
}  // namespace enn
