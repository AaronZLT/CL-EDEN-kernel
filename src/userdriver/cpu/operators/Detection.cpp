#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/cpu/operators/neon_impl/BboxUtil.hpp"
#include "userdriver/cpu/operators/Detection.hpp"

namespace enn {
namespace ud {
namespace cpu {

Detection::Detection(const PrecisionType &precision) {
    precision_ = precision;
}

Status Detection::initialize(const Dim4 &, const Dim4 &, const Dim4 &, const Dim4 &, const uint32_t &num_classes,
                             const bool &share_location, const float &nms_threshold, const int32_t &background_label_id,
                             const int32_t &nms_top_k, const int32_t &keep_top_k, const uint32_t &code_type,
                             const float &confidence_threshold, const float &nms_eta,
                             const bool &variance_encoded_in_target) {
    DEBUG_PRINT(
        "num_classes=%d, share_location=%d, nms_threshold=%f, background_label_id=%d, nms_top_k=%d, keep_top_k=%d, "
        "code_type=%d, confidence_threshold=%f, nms_eta=%f, variance_encoded_in_target=%d\n",
        num_classes, share_location, nms_threshold, background_label_id, nms_top_k, keep_top_k, code_type,
        confidence_threshold, nms_eta, variance_encoded_in_target);

    descriptor_.num_classes_ = num_classes;
    descriptor_.share_location_ = share_location;
    descriptor_.nms_threshold_ = nms_threshold;
    descriptor_.background_label_id_ = background_label_id;
    descriptor_.nms_top_k_ = nms_top_k;
    descriptor_.keep_top_k_ = keep_top_k;
    descriptor_.code_type_ = code_type;
    descriptor_.confidence_threshold_ = confidence_threshold;
    descriptor_.nms_eta_ = nms_eta;
    descriptor_.variance_encoded_in_target_ = variance_encoded_in_target;
    return Status::SUCCESS;
}

Status Detection::execute(const std::shared_ptr<ITensor> input_location, const std::shared_ptr<ITensor> input_confidence,
                          const std::shared_ptr<ITensor> input_prior, std::shared_ptr<ITensor> output) {
    uint32_t output_channel = 0;
    uint32_t output_height = 0;
    uint32_t output_width = 0;

    auto in_0 = std::static_pointer_cast<NEONTensor<float>>(input_location);
    auto in_1 = std::static_pointer_cast<NEONTensor<float>>(input_confidence);
    auto in_2 = std::static_pointer_cast<NEONTensor<float>>(input_prior);
    auto out_0 = std::static_pointer_cast<NEONTensor<float>>(output);

    auto in_data_0 = in_0->getBufferPtr();
    auto in_data_1 = in_1->getBufferPtr();
    auto in_data_2 = in_2->getBufferPtr();
    auto out_data_0 = out_0->getBufferPtr();

    DEBUG_PRINT("Detection buffer info:\n\t in[0] %p (%" PRIu64 ")\n\t in[1] %p (%" PRIu64 ")\n\t in[2] %p (%" PRIu64
                ")\n\tout[0] %p (%" PRIu64 ")\n",
                in_data_0, in_0->getNumOfBytes(), in_data_1, in_1->getNumOfBytes(), in_data_2, in_2->getNumOfBytes(),
                out_data_0, out_0->getNumOfBytes());

    uint32_t num_priors = in_2->getDim().h / 4;

    Detect(in_0->getDim().n, descriptor_.num_classes_, descriptor_.share_location_, descriptor_.nms_threshold_,
           descriptor_.background_label_id_, descriptor_.nms_top_k_, descriptor_.keep_top_k_,
           static_cast<PriorBoxCodingType>(descriptor_.code_type_), descriptor_.confidence_threshold_, descriptor_.nms_eta_,
           descriptor_.variance_encoded_in_target_, num_priors, in_data_0, in_data_1, in_data_2, out_data_0, output_channel,
           output_height, output_width);

    Dim4 out_dim = {1, output_channel, output_height, output_width};
    out_0->reconfigureDim(out_dim);

    return Status::SUCCESS;
}

Status Detection::release() {
    return Status::SUCCESS;
}

void Detection::Detect(const uint32_t &num_input, const uint32_t &num_classes, const bool &share_location,
                       const float &nms_threshold, const int32_t &background_label_id, const int32_t &top_k,
                       const int32_t &keep_top_k, const PriorBoxCodingType &code_type, const float &confidence_threshold,
                       const float &eta, const bool &variance_encoded_in_target, const uint32_t &num_priors,
                       const float *loc_data, const float *conf_data, const float *prior_data, float *output_data,
                       uint32_t &output_channel, uint32_t &output_height, uint32_t &output_width) {
    int num_loc_classes = share_location ? 1 : num_classes;
    // Retrieve all location predictions.
    std::vector<Label_bbox> all_loc_preds;
    GetLocPredictions(loc_data, num_input, num_priors, num_loc_classes, share_location, &all_loc_preds);
    // Retrieve all confidences.
    std::vector<std::map<int, std::vector<float>>> all_conf_scores;
    GetConfidenceScores(conf_data, num_input, num_priors, num_classes, &all_conf_scores);
    // Retrieve all prior bboxes. It is same within a batch since we assume all
    // images in a batch are of same dimension.
    std::vector<NormalizedBBox> prior_bboxes;
    std::vector<std::vector<float>> prior_variances;
    GetPriorBBoxes(prior_data, num_priors, &prior_bboxes, &prior_variances);
    // Decode all loc predictions to bboxes.
    std::vector<Label_bbox> all_decode_bboxes;
    const bool clip_bbox = false;
    DecodeBBoxesAll(all_loc_preds, prior_bboxes, prior_variances, num_input, share_location, num_loc_classes,
                    background_label_id, code_type, variance_encoded_in_target, clip_bbox, &all_decode_bboxes);
    int num_kept = 0;
    std::vector<std::map<int, std::vector<int>>> all_indices;
    for (uint32_t i = 0; i < num_input; ++i) {
        const Label_bbox &decode_bboxes = all_decode_bboxes[i];
        const std::map<int, std::vector<float>> &conf_scores = all_conf_scores[i];
        std::map<int, std::vector<int>> indices;
        int num_det = 0;
        for (uint32_t c = 0; c < num_classes; ++c) {
            if ((background_label_id >= 0) && (c == static_cast<uint32_t>(background_label_id))) {
                // Ignore background class.
                continue;
            }
            if (conf_scores.find(c) == conf_scores.end()) {
                // Something bad happened if there are no predictions for current label.
            }
            const std::vector<float> &scores = conf_scores.find(c)->second;
            int label = share_location ? -1 : c;
            if (decode_bboxes.find(label) == decode_bboxes.end()) {
                // Something bad happened if there are no predictions for current label.
                continue;
            }
            const std::vector<NormalizedBBox> &bboxes = decode_bboxes.find(label)->second;
            ApplyNMSFast(bboxes, scores, confidence_threshold, nms_threshold, eta, top_k, &(indices[c]));
            num_det += indices[c].size();
        }
        if (keep_top_k > -1 && num_det > keep_top_k) {
            std::vector<std::pair<float, std::pair<int, int>>> score_index_pairs;
            std::map<int, std::vector<int>>::iterator it;
            for (it = indices.begin(); it != indices.end(); it++) {
                int label = it->first;
                const std::vector<int> &label_indices = it->second;
                if (conf_scores.find(label) == conf_scores.end()) {
                    // Something bad happened for current label.
                    continue;
                }
                const std::vector<float> &scores = conf_scores.find(label)->second;
                for (size_t j = 0; j < label_indices.size(); ++j) {
                    int idx = label_indices[j];
                    score_index_pairs.push_back(std::make_pair(scores[idx], std::make_pair(label, idx)));
                }
            }
            // Keep top k results per image
            std::sort(score_index_pairs.begin(), score_index_pairs.end(), SortScorePairDescend<std::pair<int, int>>);
            score_index_pairs.resize(keep_top_k);
            // Store the new indices.
            std::map<int, std::vector<int>> new_indices;
            for (size_t j = 0; j < score_index_pairs.size(); ++j) {
                int label = score_index_pairs[j].second.first;
                int idx = score_index_pairs[j].second.second;
                new_indices[label].push_back(idx);
            }
            all_indices.push_back(new_indices);
            num_kept += keep_top_k;
        } else {
            all_indices.push_back(indices);
            num_kept += num_det;
        }
    }
    std::vector<int> output_shape(2, 1);
    if (num_kept == 0) {
        output_shape.push_back(num_input);
    } else {
        output_shape.push_back(num_kept);
    }
    output_shape.push_back(7);
    int output_size = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3];
    if (num_kept == 0) {
        float *output_data_ptr = output_data;
        for (int idx = 0; idx < output_size; idx++) {
            *(output_data + idx) = -1;
        }
        for (uint32_t i = 0; i < num_input; ++i) {
            output_data_ptr[0] = i;
            output_data_ptr += 7;
        }
        output_data_ptr = nullptr;
    }
    int count = 0;
    for (uint32_t i = 0; i < num_input; i++) {
        const std::map<int, std::vector<float>> &conf_scores = all_conf_scores[i];
        const Label_bbox &decode_bboxes = all_decode_bboxes[i];
        std::map<int, std::vector<int>>::iterator it;
        for (it = all_indices[i].begin(); it != all_indices[i].end(); it++) {
            int label = it->first;
            if (conf_scores.find(label) == conf_scores.end()) {
                // Something bad happened if there are no predictions for current label.
                continue;
            }
            const std::vector<float> &scores = conf_scores.find(label)->second;
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
                *(output_data_beg_ptr + count * 7 + 2) = scores[idx];
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
    output_channel = output_shape[1];
    output_height = output_shape[2];
    output_width = output_shape[3];
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
