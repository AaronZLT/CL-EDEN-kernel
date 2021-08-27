#include "BboxUtil_batch_single.hpp"

namespace enn {
namespace ud {
namespace cpu {

void getLocPredictions(const float *loc_data, const uint32_t &num, const uint32_t &num_preds_perclass,
                       const uint32_t &num_loc_classes, const bool &share_location, std::vector<Label_bbox> *loc_preds) {
    loc_preds->clear();
    loc_preds->resize(num);
    for (uint32_t i = 0; i < num; ++i) {
        Label_bbox &Label_bbox = (*loc_preds)[i];
        for (uint32_t p = 0; p < num_preds_perclass; ++p) {
            uint32_t start_idx = p * num_loc_classes * 4;
            for (uint32_t c = 0; c < num_loc_classes; ++c) {
                int label = share_location ? -1 : c;
                if (Label_bbox.find(label) == Label_bbox.end()) {
                    Label_bbox[label].resize(num_preds_perclass);
                }
                Label_bbox[label][p].set_xmin(loc_data[start_idx + c * 4]);
                Label_bbox[label][p].set_ymin(loc_data[start_idx + c * 4 + 1]);
                Label_bbox[label][p].set_xmax(loc_data[start_idx + c * 4 + 2]);
                Label_bbox[label][p].set_ymax(loc_data[start_idx + c * 4 + 3]);
            }
        }
        loc_data += num_preds_perclass * num_loc_classes * 4;
    }
}

void getConfidenceScores(const float *conf_data, const uint32_t &num, const uint32_t &num_preds_perclass,
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

float bboxSize(const NormalizedBBox &bbox, const bool &normalized) {
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

void getPriorBBoxes(const float *prior_data, const uint32_t num_priors, std::vector<NormalizedBBox> *prior_bboxes,
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
        float bbox_size = bboxSize(bbox);
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

void clipBBox(const NormalizedBBox &bbox, NormalizedBBox *clip_bbox) {
    clip_bbox->set_xmin(std::max(std::min(bbox.xmin(), 1.f), 0.f));
    clip_bbox->set_ymin(std::max(std::min(bbox.ymin(), 1.f), 0.f));
    clip_bbox->set_xmax(std::max(std::min(bbox.xmax(), 1.f), 0.f));
    clip_bbox->set_ymax(std::max(std::min(bbox.ymax(), 1.f), 0.f));
    clip_bbox->clear_size();
    clip_bbox->set_size(bboxSize(*clip_bbox));
    clip_bbox->set_difficult(bbox.difficult());
}

void decodeBBox(const uint32_t &bbox_i, const uint32_t &var_i, const float *prior_data, const PriorBoxCodingType &code_type,
                const bool &variance_encoded_in_target, const bool &clip_bbox, const float *loc_data,
                NormalizedBBox *decode_bbox) {
    if (code_type == PRIORBOX_CODETYPE_CORNER) {
        std::cout << "Non-implement CodeType" << std::endl;
    } else if (code_type == PRIORBOX_CODETYPE_CENTERSIZE) {
        float priorWidth = prior_data[bbox_i + 2] - prior_data[bbox_i];
        float priorHeight = prior_data[bbox_i + 3] - prior_data[bbox_i + 1];
        float priorCenterX = (prior_data[bbox_i] + prior_data[bbox_i + 2]) / 2.;
        float priorCenterY = (prior_data[bbox_i + 1] + prior_data[bbox_i + 3]) / 2.;
        float decode_bboxCenterX = 0.0f, decode_bboxCenterY = 0.0f;
        float decode_bboxWidth = 0.0f, decode_bboxHeight = 0.0f;
        if (variance_encoded_in_target) {
            // variance is encoded in target, we simply need to retore the offset predictions.
            decode_bboxCenterX = loc_data[bbox_i] * priorWidth + priorCenterX;
            decode_bboxCenterY = loc_data[bbox_i + 1] * priorHeight + priorCenterY;
            decode_bboxWidth = exp(loc_data[bbox_i + 2]) * priorWidth;
            decode_bboxHeight = exp(loc_data[bbox_i + 3]) * priorHeight;
        } else {
            // variance is encoded in bbox, we need to scale the offset accordingly.
            decode_bboxCenterX = prior_data[var_i] * loc_data[bbox_i] * priorWidth + priorCenterX;
            decode_bboxCenterY = prior_data[var_i + 1] * loc_data[bbox_i + 1] * priorHeight + priorCenterY;
            decode_bboxWidth = exp(prior_data[var_i + 2] * loc_data[bbox_i + 2]) * priorWidth;
            decode_bboxHeight = exp(prior_data[var_i + 3] * loc_data[bbox_i + 3]) * priorHeight;
        }
        decode_bbox->set_xmin(decode_bboxCenterX - decode_bboxWidth / 2.);
        decode_bbox->set_ymin(decode_bboxCenterY - decode_bboxHeight / 2.);
        decode_bbox->set_xmax(decode_bboxCenterX + decode_bboxWidth / 2.);
        decode_bbox->set_ymax(decode_bboxCenterY + decode_bboxHeight / 2.);
    } else if (code_type == PRIORBOX_CODETYPE_CORNERSIZE) {
        std::cout << "Non-implement CodeType" << std::endl;
    } else {
        std::cerr << "Unknown LocLossType." << std::endl;
    }
    float bbox_size = bboxSize(*decode_bbox);
    decode_bbox->set_size(bbox_size);
    if (clip_bbox) {
        clipBBox(*decode_bbox, decode_bbox);
    }
}

void decodeBBoxes(const uint32_t &num_priors, const float *prior_data, const PriorBoxCodingType &code_type,
                  const bool &variance_encoded_in_target, const bool &clip_bbox, const float *loc_data,
                  std::vector<NormalizedBBox> *decode_bboxes) {
    int num_bboxes = num_priors;
    decode_bboxes->resize(num_bboxes);
    for (int i = 0; i < num_bboxes; ++i) {
        NormalizedBBox decode_bbox;
        uint32_t bbox_i = i * 4;
        uint32_t var_i = (num_priors + i) * 4;
        decodeBBox(bbox_i, var_i, prior_data, code_type, variance_encoded_in_target, clip_bbox, loc_data, &decode_bbox);
        decode_bboxes->at(i) = decode_bbox;
    }
}

void decodeBBoxesAll(const float *loc_data, const uint32_t &num_priors, const float *prior_data, const uint32_t &num,
                     const bool &share_location, const uint32_t &num_loc_classes, const int32_t &background_label_id,
                     const PriorBoxCodingType &code_type, const bool &variance_encoded_in_target, const bool &clip,
                     std::vector<Label_bbox> *all_decode_bboxes) {
    for (uint32_t i = 0; i < num; ++i) {
        // Decode predictions into bboxes.
        Label_bbox &decode_bboxes = (*all_decode_bboxes)[i];
        for (uint32_t c = 0; c < num_loc_classes; ++c) {
            int label = share_location ? -1 : c;
            if (label == background_label_id) {
                // Ignore background class.
                continue;
            }
            decodeBBoxes(num_priors, prior_data, code_type, variance_encoded_in_target, clip, loc_data,
                         &(decode_bboxes[label]));
        }
    }
}

void getMaxScoreIndex(const float *conf_data, uint32_t num_priors, uint32_t num_classes, uint32_t c, const float &threshold,
                      const int32_t &top_k, std::vector<std::pair<float, int>> *score_index_vec) {
    // Generate index score pairs.
    for (size_t i = 0; i < num_priors; ++i) {
        float conf_i = conf_data[i * num_classes + c];
        if (conf_i > threshold) {
            score_index_vec->push_back(std::make_pair(conf_i, i));
        }
    }
    // Sort the score pair according to the scores in descending order
    std::stable_sort(score_index_vec->begin(), score_index_vec->end(), sortScorePairDescend<int>);
    // Keep top_k scores if needed.
    if (top_k > -1 && (top_k < 0 || static_cast<uint32_t>(top_k) < score_index_vec->size())) {
        score_index_vec->resize(top_k);
    }
}

void intersectBBox(const NormalizedBBox &bbox1, const NormalizedBBox &bbox2, NormalizedBBox *intersect_bbox) {
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

float jaccardOverlap(const NormalizedBBox &bbox1, const NormalizedBBox &bbox2, const bool &normalized) {
    NormalizedBBox normalizedBBox;
    intersectBBox(bbox1, bbox2, &normalizedBBox);
    float intersect_width = 0.0f, intersect_height = 0.0f;
    if (normalized) {
        intersect_width = normalizedBBox.xmax() - normalizedBBox.xmin();
        intersect_height = normalizedBBox.ymax() - normalizedBBox.ymin();
    } else {
        intersect_width = normalizedBBox.xmax() - normalizedBBox.xmin() + 1;
        intersect_height = normalizedBBox.ymax() - normalizedBBox.ymin() + 1;
    }
    if (intersect_width > 0 && intersect_height > 0) {
        float intersect_size = intersect_width * intersect_height;
        float bBoxSize = bboxSize(bbox1);
        float bBoxSize1 = bboxSize(bbox2);
        return intersect_size / (bBoxSize + bBoxSize1 - intersect_size);
    } else {
        return 0.;
    }
}

void applyNMSFast(const std::vector<NormalizedBBox> &bboxes, const float &score_threshold, const float &nms_threshold,
                  const float &eta, const int32_t &top_k, const float *conf_data, uint32_t num_priors, uint32_t num_classes,
                  uint32_t c, std::vector<int> *indices) {
    if (bboxes.size() != num_priors) {
        std::cout << "Fatal Error" << std::endl;
        return;
    }
    // Get top_k scores (with corresponding indices).
    std::vector<std::pair<float, int>> score_index_vec;
    getMaxScoreIndex(conf_data, num_priors, num_classes, c, score_threshold, static_cast<int>(top_k), &score_index_vec);
    // Do nms.
    float adaptive_threshold = nms_threshold;
    indices->clear();
    for (auto &score_index_pair : score_index_vec) {
        const int idx = score_index_pair.second;
        bool keep = true;
        for (size_t k = 0; k < indices->size(); ++k) {
            if (keep) {
                const int kept_idx = (*indices)[k];
                float overlap = jaccardOverlap(bboxes[idx], bboxes[kept_idx]);
                keep = overlap <= adaptive_threshold;
            } else {
                break;
            }
        }
        if (keep) {
            indices->push_back(idx);
        }
        if (keep && eta < 1 && adaptive_threshold > 0.5) {
            adaptive_threshold *= eta;
        }
    }
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
