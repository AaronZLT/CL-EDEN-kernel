#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/operators/ISigDet.hpp"
#include "userdriver/cpu/operators/neon_impl/NormalizedBbox.hpp"
#include "userdriver/cpu/common/NEONIncludes.hpp"
#include "userdriver/cpu/common/NEONTensor.hpp"

namespace enn {
namespace ud {
namespace cpu {

class NEONSigDet : public ISigDet {
public:
    NEONSigDet(const PrecisionType &precision);
    Status initialize(const Dim4 &location_dim, const Dim4 &confidence_dim, const Dim4 &prior_dim, const Dim4 &output_dim,
                      const uint32_t &num_classes, const bool &share_location, const float &nms_threshold,
                      const int32_t &background_label_id, const int32_t &nms_top_k, const int32_t &keep_top_k,
                      const uint32_t &code_type, const float &confidence_threshold, const float &nms_eta,
                      const bool &variance_encoded_in_target);
    Status execute(const std::shared_ptr<ITensor> &input_location, const std::shared_ptr<ITensor> &input_confidence,
                   const std::shared_ptr<ITensor> &input_prior, std::shared_ptr<ITensor> output);
    Status release();

private:
    PrecisionType precision_;
    uint32_t num_classes_;  // default = 0
    uint32_t num_priors_;
    bool share_location_;          // default = true
    float nms_threshold_;          // default = 0.3
    int32_t background_label_id_;  // default = 0
    int32_t nms_top_k_;            // default = 0
    int32_t keep_top_k_;           // default = -1
    uint32_t code_type_;
    float nms_eta_ = 1.0;         // default = 1.0
    float confidence_threshold_;  // default = 0
    float conf_thresh_unnormalized_;
    bool variance_encoded_in_target_;  // default = false
    Dim4 loc_dim_;
    Dim4 prior_dim_;
    std::vector<uint32_t> class_in_use_;
    std::vector<Label_bbox> all_decode_bboxes_;
};  // class NEONSigDet

/**
 * @brief Compare conf scores with confidence threshold, and pick classes with score greater than threshold.
 *
 * @param conf Raw conf scores
 * @param num_classes Number of classes
 * @param num_priors Number of prior boxes
 * @param conf_thesh_unnormalized unnormalized conf threshold computed with sigmoid's inverse function
 * @param class_in_use classes picked with score greater than threshold
 */
static inline void exploitClassInUse(float *conf, uint32_t num_classes, uint32_t num_priors, float conf_thesh_unnormalized,
                                     uint32_t *class_in_use) {
#ifdef NEON_OPT
    const float32x4_t _conf_thresh = vdupq_n_f32(conf_thesh_unnormalized);
    {  // compute class_in_use
        size_t npack4 = num_classes >> 2;
        size_t nremain = num_classes - (npack4 << 2);
        for (size_t p = 0; p < num_priors; ++p) {
            float *input_ptr = conf + p * num_classes;
            uint32_t *output_ptr = class_in_use;
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
                uint32_t conf_thresh_mask = (*input_ptr > conf_thesh_unnormalized ? 1 : 0);
                *output_ptr |= conf_thresh_mask;
                input_ptr++;
                output_ptr++;
            }
        }
    }
#else
    ENN_UNUSED(conf);
    ENN_UNUSED(num_classes);
    ENN_UNUSED(num_priors);
    ENN_UNUSED(conf_thesh_unnormalized);
    ENN_UNUSED(class_in_use);
#endif
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
