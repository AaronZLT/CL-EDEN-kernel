#pragma once

#include "BboxUtil_batch_single.hpp"

namespace enn {
namespace ud {
namespace cpu {

void detection(const uint32_t &num_input, const uint32_t &num_classes, const bool &share_location,
               const float &nms_threshold, const int32_t &background_label_id, const int32_t &top_k,
               const int32_t &keep_top_k, const PriorBoxCodingType &code_type, float confidence_threshold, const float &eta,
               const bool &variance_encoded_in_target, const uint32_t &num_priors, const float *loc_data,
               const float *conf_data, const float *prior_data, float *output_data,
               uint32_t &output_channel, uint32_t &output_height, uint32_t &output_width,
               bool is_conf_thresh_unnormalized = false, float conf_thresh_unnormalized = .0f,
               std::vector<Label_bbox> *all_decode_bboxes = nullptr, std::vector<uint32_t> *class_in_use = nullptr);

}  // namespace cpu
}  // namespace ud
}  // namespace enn
