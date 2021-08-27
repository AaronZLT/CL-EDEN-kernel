#pragma once

#include "userdriver/common/operator_interfaces/common/Includes.hpp"
#include "NormalizedBbox.hpp"

namespace enn {
namespace ud {
namespace cpu {

template <typename T>
inline bool sortScorePairDescend(const std::pair<float, T> &pair1, const std::pair<float, T> &pair2) {
    return pair1.first > pair2.first;
}

void getLocPredictions(const float *loc_data, const uint32_t &num, const uint32_t &num_preds_perclass,
                       const uint32_t &num_loc_classes, const bool &share_location, std::vector<Label_bbox> *loc_preds);

void getConfidenceScores(const float *conf_data, const uint32_t &num, const uint32_t &num_preds_perclass,
                         const uint32_t &num_classes, std::vector<std::map<int, std::vector<float>>> *conf_preds);

float bboxSize(const NormalizedBBox &bbox, const bool &normalized = true);

void getPriorBBoxes(const float *prior_data, const uint32_t num_priors, std::vector<NormalizedBBox> *prior_bboxes,
                    std::vector<std::vector<float>> *prior_variances);

void clipBBox(const NormalizedBBox &bbox, NormalizedBBox *clip_bbox);

void decodeBBox(const uint32_t &bbox_i, const uint32_t &var_i, const float *prior_data, const PriorBoxCodingType &code_type,
                const bool &variance_encoded_in_target, const bool &clip_bbox, const float *loc_data,
                NormalizedBBox *decode_bbox);

void decodeBBoxes(const uint32_t &num_priors, const float *prior_data, const PriorBoxCodingType &code_type,
                  const bool &variance_encoded_in_target, const bool &clip_bbox, const float *loc_data,
                  std::vector<NormalizedBBox> *decode_bboxes);

void decodeBBoxesAll(const float *loc_data, const uint32_t &num_priors, const float *prior_data, const uint32_t &num,
                     const bool &share_location, const uint32_t &num_loc_classes, const int32_t &background_label_id,
                     const PriorBoxCodingType &code_type, const bool &variance_encoded_in_target, const bool &clip,
                     std::vector<Label_bbox> *all_decode_bboxes);

void getMaxScoreIndex(const float *conf_data, uint32_t num_priors, uint32_t num_classes, uint32_t c, const float &threshold,
                      const int32_t &top_k, std::vector<std::pair<float, int>> *score_index_vec);

void intersectBBox(const NormalizedBBox &bbox1, const NormalizedBBox &bbox2, NormalizedBBox *intersect_bbox);

float jaccardOverlap(const NormalizedBBox &bbox1, const NormalizedBBox &bbox2, const bool &normalized = true);

void applyNMSFast(const std::vector<NormalizedBBox> &bboxes, const float &score_threshold, const float &nms_threshold,
                  const float &eta, const int32_t &top_k, const float *conf_data, uint32_t num_priors, uint32_t num_classes,
                  uint32_t c, std::vector<int> *indices);

}  // namespace cpu
}  // namespace ud
}  // namespace enn
