#include "DetectionOutput.hpp"
#include "userdriver/cpu/common/NEONIncludes.hpp"
#include "userdriver/cpu/common/NEONTensor.hpp"

namespace enn {
namespace ud {
namespace cpu {

void detection(const uint32_t &num_input, const uint32_t &num_classes, const bool &share_location,
               const float &nms_threshold, const int32_t &background_label_id, const int32_t &top_k,
               const int32_t &keep_top_k, const PriorBoxCodingType &code_type, float confidence_threshold, const float &eta,
               const bool &variance_encoded_in_target, const uint32_t &num_priors, const float *loc_data,
               const float *conf_data, const float *prior_data, float *output_data,
               uint32_t &output_channel, uint32_t &output_height, uint32_t &output_width, bool is_conf_thresh_unnormalized,
               float conf_thresh_unnormalized, std::vector<Label_bbox> *all_decode_bboxes,
               std::vector<uint32_t> *class_in_use) {
    int num_loc_classes = share_location ? 1 : num_classes;
    if (is_conf_thresh_unnormalized) {
        confidence_threshold = conf_thresh_unnormalized;
    }
    std::vector<Label_bbox> all_decoded_bboxes_;
    if (all_decode_bboxes == nullptr) {
        all_decoded_bboxes_.resize(num_input);
        Label_bbox &decode_bboxes = all_decoded_bboxes_[0];
        if (share_location) {
            decode_bboxes[-1].resize(num_priors);
        }
        all_decode_bboxes = &all_decoded_bboxes_;
    }
    const bool clip_bbox = false;
    decodeBBoxesAll(loc_data, num_priors, prior_data, num_input, share_location, num_loc_classes, background_label_id,
                    code_type, variance_encoded_in_target, clip_bbox, all_decode_bboxes);

    // pre-save classes utilized in nms
    std::vector<uint32_t> vec_class_in_use;
    if (class_in_use == nullptr) {
        vec_class_in_use.resize(num_classes, 0);
        class_in_use = &vec_class_in_use;
#ifdef NEON_OPT
        const float32x4_t _conf_thresh = vdupq_n_f32(confidence_threshold);
        size_t npack4 = num_classes >> 2;
        size_t nremain = num_classes - (npack4 << 2);
        for (size_t p = 0; p < num_priors; ++p) {
            float *input_ptr = const_cast<float*>(conf_data) + p * num_classes;
            uint32_t *output_ptr = class_in_use->data();
            size_t nn = npack4;
            size_t remain = nremain;
            for (; nn > 0; nn--) {
                float32x4_t _p = vld1q_f32(input_ptr);
                uint32x4_t _q = vld1q_u32(output_ptr);
                uint32x4_t _conf_thresh_mask_u32 = vcgeq_f32(_p, _conf_thresh);
                _q = vqaddq_u32(_q, _conf_thresh_mask_u32);
                vst1q_u32(output_ptr, _q);
                input_ptr += 4;
                output_ptr += 4;
            }
            for (; remain > 0; remain--) {
                uint32_t conf_thresh_mask = (*input_ptr > confidence_threshold ? 1 : 0);
                *output_ptr |= conf_thresh_mask;
                input_ptr++;
                output_ptr++;
            }
        }
#else
        float conf_temp = confidence_threshold;
        for (int i = 0; i < num_classes * num_priors; ++i) {
            if (conf_data[i] >= conf_temp) {
                int c = i % num_classes;
                class_in_use->at(c) = 1;
            }
        }
#endif
    }
    int num_kept = 0;
    std::vector<std::map<int, std::vector<int>>> all_indices;
    all_indices.resize(num_input);
    for (uint32_t i = 0; i < num_input; ++i) {
        const Label_bbox &decode_bboxes = (*all_decode_bboxes)[i];
        std::map<int, std::vector<int>> &indices = all_indices[i];
        int num_det = 0;
        for (uint32_t c = 0; c < num_classes; ++c) {
            if (background_label_id >= 0 && c == static_cast<uint32_t>(background_label_id)) {
                // Ignore background class.
                continue;
            }
            int label = share_location ? -1 : c;
            if (decode_bboxes.find(label) == decode_bboxes.end()) {
                // Something bad happened if there are no predictions for current label.
                continue;
            }
            const std::vector<NormalizedBBox> &bboxes = decode_bboxes.find(label)->second;
            if (class_in_use->at(c)) {
                applyNMSFast(bboxes, confidence_threshold, nms_threshold, eta, top_k, conf_data, num_priors, num_classes, c,
                             &(indices[c]));
                num_det += indices[c].size();
            }
        }

        if (keep_top_k > -1 && num_det > keep_top_k) {
            std::vector<std::pair<float, std::pair<int, int>>> scoreIndexPairs;
            for (std::map<int, std::vector<int>>::iterator it = indices.begin(); it != indices.end(); ++it) {
                int label = it->first;
                const std::vector<int> &labelIndices = it->second;
                for (size_t j = 0; j < labelIndices.size(); ++j) {
                    int idx = labelIndices[j];
                    scoreIndexPairs.push_back(
                        std::make_pair(conf_data[idx * num_classes + label], std::make_pair(label, idx)));
                }
            }
            /*  Keep top k results per image.*/
            std::sort(scoreIndexPairs.begin(), scoreIndexPairs.end(), sortScorePairDescend<std::pair<int, int>>);
            scoreIndexPairs.resize(keep_top_k);
            /* Store the new indices. */
            std::map<int, std::vector<int>> new_indices;
            for (size_t j = 0; j < scoreIndexPairs.size(); ++j) {
                int label = scoreIndexPairs[j].second.first;
                int idx = scoreIndexPairs[j].second.second;
                new_indices[label].push_back(idx);
            }
            all_indices[i] = new_indices;
            num_kept += keep_top_k;
        } else {
            num_kept += num_det;
        }
    }

    std::vector<int> outputShape = {1, 1, 1, 7};
    if (num_kept > 0) {
        outputShape[2] = num_kept;
    }
    int outputSize = outputShape[0] * outputShape[1] * outputShape[2] * outputShape[3];
    if (num_kept == 0) {
        float *output_data_ptr = output_data;
        for (int idx = 0; idx < outputSize; idx++) {
            *(output_data + idx) = -1;
        }
        for (uint32_t i = 0; i < num_input; ++i) {
            output_data_ptr[0] = i;
            output_data_ptr += 7;
        }
        output_data_ptr = nullptr;
    }
    int count = 0;
    for (uint32_t i = 0; i < num_input; ++i) {
        const Label_bbox &decode_bboxes = (*all_decode_bboxes)[i];
        for (std::map<int, std::vector<int>>::iterator it = all_indices[i].begin(); it != all_indices[i].end(); ++it) {
            int label = it->first;
            int loc_label = share_location ? -1 : label;
            if (decode_bboxes.find(loc_label) == decode_bboxes.end()) {
                // Something bad happened if there are no predictions for current label.
                continue;
            }
            const std::vector<NormalizedBBox> &bboxes = decode_bboxes.find(loc_label)->second;
            std::vector<int> &indices = it->second;
            float *output_data_beg_ptr = output_data;
            for (size_t j = 0; j < indices.size(); ++j) {
                int idx = indices[j];
                *(output_data_beg_ptr + count * 7) = i;
                *(output_data_beg_ptr + count * 7 + 1) = label;
                *(output_data_beg_ptr + count * 7 + 2) = conf_data[idx * num_classes + label];
                if (is_conf_thresh_unnormalized) {
                    float unnormalized_conf = *(output_data_beg_ptr + count * 7 + 2);
                    *(output_data_beg_ptr + count * 7 + 2) =
                        1.0f / (1.0f + expf(-1.0f * unnormalized_conf));  // normalize by sigmoid: 1 / (1 + e^(-x))
                }
                const NormalizedBBox &bbox = bboxes[idx];
                *(output_data_beg_ptr + count * 7 + 3) = bbox.xmin();
                *(output_data_beg_ptr + count * 7 + 4) = bbox.ymin();
                *(output_data_beg_ptr + count * 7 + 5) = bbox.xmax();
                *(output_data_beg_ptr + count * 7 + 6) = bbox.ymax();
                ++count;
            }
            output_data_beg_ptr = nullptr;
        }
    }
    output_channel = outputShape[1];
    output_height = outputShape[2];
    output_width = outputShape[3];
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
