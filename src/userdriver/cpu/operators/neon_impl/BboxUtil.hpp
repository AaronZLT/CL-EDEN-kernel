#pragma once

#include "userdriver/common/operator_interfaces/common/Includes.hpp"
#include "NormalizedBbox.hpp"

namespace enn {
namespace ud {
namespace cpu {

template <typename T>
inline static bool SortScorePairDescend(const std::pair<float, T> &pair1, const std::pair<float, T> &pair2) {
    return pair1.first > pair2.first;
}

void GetLocPredictions(const float *loc_data, const uint32_t &num, const uint32_t &num_preds_perclass,
                       const uint32_t &num_loc_classes, const bool &share_location, std::vector<Label_bbox> *loc_preds);

void GetConfidenceScores(const float *conf_data, const uint32_t &num, const uint32_t &num_preds_perclass,
                         const uint32_t &num_classes, std::vector<std::map<int, std::vector<float>>> *conf_preds);

float BBoxSize(const NormalizedBBox &bbox, const bool normalized = true);

void GetPriorBBoxes(const float *prior_data, const uint32_t &num_priors, std::vector<NormalizedBBox> *prior_bboxes,
                    std::vector<std::vector<float>> *prior_variances);

void ClipBBox(const NormalizedBBox &bbox, NormalizedBBox *clip_bbox);

void DecodeBBox(const NormalizedBBox &prior_bbox, const std::vector<float> &prior_variance,
                const PriorBoxCodingType &code_type, const bool &variance_encoded_in_target, const bool &clip_bbox,
                const NormalizedBBox &bbox, NormalizedBBox *decode_bbox);

void DecodeBBoxes(const std::vector<NormalizedBBox> &prior_bboxes, const std::vector<std::vector<float>> &prior_variances,
                  const PriorBoxCodingType &code_type, const bool &variance_encoded_in_target,
                  const bool &clip_bbox, const std::vector<NormalizedBBox> &bboxes,
                  std::vector<NormalizedBBox> *decode_bboxes);

void DecodeBBoxesAll(const std::vector<Label_bbox> &all_loc_preds, const std::vector<NormalizedBBox> &prior_bboxes,
                     const std::vector<std::vector<float>> &prior_variances, const uint32_t &num, const bool &share_location,
                     const uint32_t &num_loc_classes, const int32_t &background_label_id,
                     const PriorBoxCodingType &code_type, const bool &variance_encoded_in_target, const bool &clip,
                     std::vector<Label_bbox> *all_decode_bboxes);

void GetMaxScoreIndex(const std::vector<float> &scores, const float &threshold, const int32_t &top_k,
                      std::vector<std::pair<float, int>> *score_index_vec);

void IntersectBBox(const NormalizedBBox &bbox1, const NormalizedBBox &bbox2, NormalizedBBox *intersect_bbox);

float JaccardOverlap(const NormalizedBBox &bbox1, const NormalizedBBox &bbox2, const bool &normalized = true);

void ApplyNMSFast(const std::vector<NormalizedBBox> &bboxes, const std::vector<float> &scores, const float &score_threshold,
                  const float &nms_threshold, const float &eta, const int32_t &top_k, std::vector<int> *indices);

}  // namespace cpu
}  // namespace ud
}  // namespace enn
