#include "BboxUtil.hpp"

namespace enn {
namespace ud {
namespace cpu {

void GetLocPredictions(const float *loc_data, const uint32_t &num, const uint32_t &num_preds_perclass,
                       const uint32_t &num_loc_classes, const bool &share_location, std::vector<Label_bbox> *loc_preds) {
    loc_preds->clear();
    loc_preds->resize(num);
    for (uint32_t i = 0; i < num; ++i) {
        Label_bbox &label_bbox = (*loc_preds)[i];
        for (uint32_t p = 0; p < num_preds_perclass; ++p) {
            int start_idx = p * num_loc_classes * 4;
            for (uint32_t c = 0; c < num_loc_classes; ++c) {
                int label = share_location ? -1 : c;
                if (label_bbox.find(label) == label_bbox.end()) {
                    label_bbox[label].resize(num_preds_perclass);
                }
                label_bbox[label][p].set_xmin(loc_data[start_idx + c * 4]);
                label_bbox[label][p].set_ymin(loc_data[start_idx + c * 4 + 1]);
                label_bbox[label][p].set_xmax(loc_data[start_idx + c * 4 + 2]);
                label_bbox[label][p].set_ymax(loc_data[start_idx + c * 4 + 3]);
            }
        }
        loc_data += num_preds_perclass * num_loc_classes * 4;
    }
}

void GetConfidenceScores(const float *conf_data, const uint32_t &num, const uint32_t &num_preds_perclass,
                         const uint32_t &num_classes, std::vector<std::map<int, std::vector<float>>> *conf_preds) {
    conf_preds->clear();
    conf_preds->resize(num);
    for (uint32_t i = 0; i < num; ++i) {
        std::map<int, std::vector<float>> &label_scores = (*conf_preds)[i];
        for (uint32_t p = 0; p < num_preds_perclass; ++p) {
            int start_idx = p * num_classes;
            for (uint32_t c = 0; c < num_classes; ++c) {
                label_scores[c].push_back(conf_data[start_idx + c]);
            }
        }
        conf_data += num_preds_perclass * num_classes;
    }
}

float BBoxSize(const NormalizedBBox &bbox, const bool normalized) {
    if (bbox.xmax() < bbox.xmin() || bbox.ymax() < bbox.ymin()) {
        // If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
        return 0;
    } else {
        if (bbox.has_size()) {
            return bbox.size();
        } else {
            float width = bbox.xmax() - bbox.xmin();
            float height = bbox.ymax() - bbox.ymin();
            if (normalized) {
                return width * height;
            } else {
                // If bbox is not within range [0, 1].
                return (width + 1) * (height + 1);
            }
        }
    }
}

void GetPriorBBoxes(const float *prior_data, const uint32_t &num_priors, std::vector<NormalizedBBox> *prior_bboxes,
                    std::vector<std::vector<float>> *prior_variances) {
    prior_bboxes->clear();
    prior_variances->clear();
    for (uint32_t i = 0; i < num_priors; ++i) {
        int start_idx = i * 4;
        NormalizedBBox bbox;
        bbox.set_xmin(prior_data[start_idx]);
        bbox.set_ymin(prior_data[start_idx + 1]);
        bbox.set_xmax(prior_data[start_idx + 2]);
        bbox.set_ymax(prior_data[start_idx + 3]);
        float bbox_size = BBoxSize(bbox);
        bbox.set_size(bbox_size);
        prior_bboxes->push_back(bbox);
    }
    for (uint32_t i = 0; i < num_priors; ++i) {
        int start_idx = (num_priors + i) * 4;
        std::vector<float> var;
        for (int j = 0; j < 4; ++j) {
            var.push_back(prior_data[start_idx + j]);
        }
        prior_variances->push_back(var);
    }
}

void ClipBBox(const NormalizedBBox &bbox, NormalizedBBox *clip_bbox) {
    clip_bbox->set_xmin(std::max(std::min(bbox.xmin(), 1.f), 0.f));
    clip_bbox->set_ymin(std::max(std::min(bbox.ymin(), 1.f), 0.f));
    clip_bbox->set_xmax(std::max(std::min(bbox.xmax(), 1.f), 0.f));
    clip_bbox->set_ymax(std::max(std::min(bbox.ymax(), 1.f), 0.f));
    clip_bbox->clear_size();
    clip_bbox->set_size(BBoxSize(*clip_bbox));
    clip_bbox->set_difficult(bbox.difficult());
}

void DecodeBBox(const NormalizedBBox &prior_bbox, const std::vector<float> &prior_variance,
                const PriorBoxCodingType &code_type, const bool &variance_encoded_in_target, const bool &clip_bbox,
                const NormalizedBBox &bbox, NormalizedBBox *decode_bbox) {
    if (code_type == PRIORBOX_CODETYPE_CORNER) {
        std::cout << "Non-implement CodeType" << std::endl;
    } else if (code_type == PRIORBOX_CODETYPE_CENTERSIZE) {
        float prior_width = prior_bbox.xmax() - prior_bbox.xmin();
        float prior_height = prior_bbox.ymax() - prior_bbox.ymin();
        float prior_center_X = (prior_bbox.xmin() + prior_bbox.xmax()) / 2.;
        float prior_center_Y = (prior_bbox.ymin() + prior_bbox.ymax()) / 2.;
        float decode_bbox_center_X = 0.0f, decode_bbox_center_Y = 0.0f;
        float decode_bbox_width = 0.0f, decode_bbox_height = 0.0f;
        if (variance_encoded_in_target) {
            // variance is encoded in target, we simply need to retore the offset predictions.
            decode_bbox_center_X = bbox.xmin() * prior_width + prior_center_X;
            decode_bbox_center_Y = bbox.ymin() * prior_height + prior_center_Y;
            decode_bbox_width = exp(bbox.xmax()) * prior_width;
            decode_bbox_height = exp(bbox.ymax()) * prior_height;
        } else {
            // variance is encoded in bbox, we need to scale the offset accordingly.
            decode_bbox_center_X = prior_variance[0] * bbox.xmin() * prior_width + prior_center_X;
            decode_bbox_center_Y = prior_variance[1] * bbox.ymin() * prior_height + prior_center_Y;
            decode_bbox_width = exp(prior_variance[2] * bbox.xmax()) * prior_width;
            decode_bbox_height = exp(prior_variance[3] * bbox.ymax()) * prior_height;
        }
        decode_bbox->set_xmin(decode_bbox_center_X - decode_bbox_width / 2.);
        decode_bbox->set_ymin(decode_bbox_center_Y - decode_bbox_height / 2.);
        decode_bbox->set_xmax(decode_bbox_center_X + decode_bbox_width / 2.);
        decode_bbox->set_ymax(decode_bbox_center_Y + decode_bbox_height / 2.);
    } else if (code_type == PRIORBOX_CODETYPE_CORNERSIZE) {
        std::cout << "Non-implement CodeType" << std::endl;
    } else {
        std::cerr << "Unknown LocLossType." << std::endl;
    }
    float bbox_size = BBoxSize(*decode_bbox);
    decode_bbox->set_size(bbox_size);
    if (clip_bbox) {
        ClipBBox(*decode_bbox, decode_bbox);
    }
}

void DecodeBBoxes(const std::vector<NormalizedBBox> &prior_bboxes, const std::vector<std::vector<float>> &prior_variances,
                  const PriorBoxCodingType &code_type, const bool &variance_encoded_in_target,
                  const bool &clip_bbox, const std::vector<NormalizedBBox> &bboxes,
                  std::vector<NormalizedBBox> *decode_bboxes) {
    int num_bboxes = prior_bboxes.size();
    decode_bboxes->clear();
    for (int i = 0; i < num_bboxes; ++i) {
        NormalizedBBox decode_bbox;
        DecodeBBox(prior_bboxes[i], prior_variances[i], code_type, variance_encoded_in_target, clip_bbox, bboxes[i],
                   &decode_bbox);
        decode_bboxes->push_back(decode_bbox);
    }
}

void DecodeBBoxesAll(const std::vector<Label_bbox> &all_loc_preds, const std::vector<NormalizedBBox> &prior_bboxes,
                     const std::vector<std::vector<float>> &prior_variances, const uint32_t &num, const bool &share_location,
                     const uint32_t &num_loc_classes, const int32_t &background_label_id,
                     const PriorBoxCodingType &code_type, const bool &variance_encoded_in_target, const bool &clip,
                     std::vector<Label_bbox> *all_decode_bboxes) {
    all_decode_bboxes->clear();
    all_decode_bboxes->resize(num);
    for (uint32_t i = 0; i < num; ++i) {
        // Decode predictions into bboxes.
        Label_bbox &decode_bboxes = (*all_decode_bboxes)[i];
        for (uint32_t c = 0; c < num_loc_classes; ++c) {
            int label = share_location ? -1 : c;
            if (label == background_label_id) {
                // Ignore background class.
                continue;
            }
            if (all_loc_preds[i].find(label) == all_loc_preds[i].end()) {
                // Something bad happened if there are no predictions for current label.
                std::cout << "Could not find location predictions for label" << label << std::endl;
            }
            const std::vector<NormalizedBBox> &label_loc_preds = all_loc_preds[i].find(label)->second;
            DecodeBBoxes(prior_bboxes, prior_variances, code_type, variance_encoded_in_target, clip, label_loc_preds,
                         &(decode_bboxes[label]));
        }
    }
}

void GetMaxScoreIndex(const std::vector<float> &scores, const float &threshold, const int32_t &top_k,
                      std::vector<std::pair<float, int>> *score_index_vec) {
    // Generate index score pairs.
    for (size_t i = 0; i < scores.size(); ++i) {
        if (scores[i] > threshold) {
            score_index_vec->push_back(std::make_pair(scores[i], i));
        }
    }
    // Sort the score pair according to the scores in descending order
    std::stable_sort(score_index_vec->begin(), score_index_vec->end(), SortScorePairDescend<int>);
    // Keep top_k scores if needed.
    if (top_k < 0 || static_cast<uint32_t>(top_k) < score_index_vec->size()) {
        score_index_vec->resize(top_k);
    }
}

void IntersectBBox(const NormalizedBBox &bbox1, const NormalizedBBox &bbox2, NormalizedBBox *intersect_bbox) {
    if (bbox2.xmin() > bbox1.xmax() || bbox2.xmax() < bbox1.xmin() || bbox2.ymin() > bbox1.ymax() ||
        bbox2.ymax() < bbox1.ymin()) {
        // Return [0, 0, 0, 0] if there is no intersection.
        intersect_bbox->set_xmin(0);
        intersect_bbox->set_ymin(0);
        intersect_bbox->set_xmax(0);
        intersect_bbox->set_ymax(0);
    } else {
        intersect_bbox->set_xmin(std::max(bbox1.xmin(), bbox2.xmin()));
        intersect_bbox->set_ymin(std::max(bbox1.ymin(), bbox2.ymin()));
        intersect_bbox->set_xmax(std::min(bbox1.xmax(), bbox2.xmax()));
        intersect_bbox->set_ymax(std::min(bbox1.ymax(), bbox2.ymax()));
    }
}

float JaccardOverlap(const NormalizedBBox &bbox1, const NormalizedBBox &bbox2, const bool &normalized) {
    NormalizedBBox normalized_bbox;
    IntersectBBox(bbox1, bbox2, &normalized_bbox);
    float intersect_width = 0.0f, intersect_height = 0.0f;
    if (normalized) {
        intersect_width = normalized_bbox.xmax() - normalized_bbox.xmin();
        intersect_height = normalized_bbox.ymax() - normalized_bbox.ymin();
    } else {
        intersect_width = normalized_bbox.xmax() - normalized_bbox.xmin() + 1;
        intersect_height = normalized_bbox.ymax() - normalized_bbox.ymin() + 1;
    }
    if (intersect_width > 0 && intersect_height > 0) {
        float intersect_size = intersect_width * intersect_height;
        float b_box_size = BBoxSize(bbox1);
        float b_box_size1 = BBoxSize(bbox2);
        return intersect_size / (b_box_size + b_box_size1 - intersect_size);
    } else {
        return 0.;
    }
}

void ApplyNMSFast(const std::vector<NormalizedBBox> &bboxes, const std::vector<float> &scores, const float &score_threshold,
                  const float &nms_threshold, const float &eta, const int32_t &top_k, std::vector<int> *indices) {
    if (bboxes.size() != scores.size()) {
        std::cout << "Fatal Error" << std::endl;
        return;
    }
    // Get top_k scores (with corresponding indices).
    std::vector<std::pair<float, int>> score_index_vec;
    GetMaxScoreIndex(scores, score_threshold, top_k, &score_index_vec);
    // Do nms.
    float adaptive_threshold = nms_threshold;
    indices->clear();
    while (score_index_vec.size() != 0) {
        const int idx = score_index_vec.front().second;
        bool keep = true;
        for (size_t k = 0; k < indices->size(); ++k) {
            if (keep) {
                const int kept_idx = (*indices)[k];
                float overlap = JaccardOverlap(bboxes[idx], bboxes[kept_idx]);
                keep = overlap <= adaptive_threshold;
            } else {
                break;
            }
        }
        if (keep) {
            indices->push_back(idx);
        }
        score_index_vec.erase(score_index_vec.begin());
        if (keep && eta < 1 && adaptive_threshold > 0.5) {
            adaptive_threshold *= eta;
        }
    }
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
