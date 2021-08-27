#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/cpu/operators/neon_impl/DetectionOutput.hpp"
#include "NEONSigDet.hpp"

namespace enn {
namespace ud {
namespace cpu {

NEONSigDet::NEONSigDet(const PrecisionType &precision) {
    DEBUG_PRINT("created\n");

    precision_ = precision;
    num_classes_ = 0;
    num_priors_ = 0;
    share_location_ = true;
    nms_threshold_ = 0.3;
    background_label_id_ = 0;
    nms_top_k_ = 0;
    keep_top_k_ = -1;
    code_type_ = 0;
    confidence_threshold_ = 0;
    variance_encoded_in_target_ = false;
    loc_dim_ = {0, 0, 0, 0};
    prior_dim_ = {0, 0, 0, 0};
    conf_thresh_unnormalized_ = 0.0f;
}

Status NEONSigDet::initialize(const Dim4 &location_dim, const Dim4 &confidence_dim, const Dim4 &prior_dim,
                              const Dim4 &output_dim, const uint32_t &num_classes, const bool &share_location,
                              const float &nms_threshold, const int32_t &background_label_id, const int32_t &nms_top_k,
                              const int32_t &keep_top_k, const uint32_t &code_type, const float &confidence_threshold,
                              const float &nms_eta, const bool &variance_encoded_in_target) {
    DEBUG_PRINT("called\n");

    DEBUG_PRINT(
        "num_classes=%d, share_location=%d, nms_threshold=%f, background_label_id=%d, nms_top_k=%d, keep_top_k=%d, "
        "code_type=%d, confidence_threshold=%f, nms_eta=%f, variance_encoded_in_target=%d\n",
        num_classes, share_location, nms_threshold, background_label_id, nms_top_k, keep_top_k, code_type,
        confidence_threshold, nms_eta, variance_encoded_in_target);

    ENN_UNUSED(confidence_dim);
    ENN_UNUSED(output_dim);
    loc_dim_ = location_dim;
    prior_dim_ = prior_dim;
    num_classes_ = num_classes;
    num_priors_ = prior_dim_.h / 4;
    share_location_ = share_location;
    background_label_id_ = background_label_id;
    nms_threshold_ = nms_threshold;
    nms_top_k_ = nms_top_k;
    nms_eta_ = nms_eta;
    code_type_ = code_type;
    variance_encoded_in_target_ = variance_encoded_in_target;
    keep_top_k_ = keep_top_k;
    confidence_threshold_ = confidence_threshold;
    conf_thresh_unnormalized_ =
        -1.0f * logf(1.0f / (confidence_threshold_ + 1e-7) - 1.0f);  // inverse function of sigmoid: -ln(1/y - 1)
    class_in_use_.resize(num_classes, 0);
    all_decode_bboxes_.resize(loc_dim_.n);
    Label_bbox &decode_bboxes = all_decode_bboxes_[0];
    if (share_location) {
        decode_bboxes[-1].resize(num_priors_);
    }

    return Status::SUCCESS;
}

Status NEONSigDet::execute(const std::shared_ptr<ITensor> &input_location, const std::shared_ptr<ITensor> &input_confidence,
                           const std::shared_ptr<ITensor> &input_prior, std::shared_ptr<ITensor> output) {
    DEBUG_PRINT("called\n");

    uint32_t output_channel = 0;
    uint32_t output_height = 0;
    uint32_t output_width = 0;

    auto in_0 = std::static_pointer_cast<NEONTensor<float>>(input_location);
    auto in_1 = std::static_pointer_cast<NEONTensor<float>>(input_confidence);
    auto in_2 = std::static_pointer_cast<NEONTensor<float>>(input_prior);
    auto out_0 = std::static_pointer_cast<NEONTensor<float>>(output);

    float *loc_data = in_0->getBufferPtr();
    float *conf_data = in_1->getBufferPtr();
    float *prior_data = in_2->getBufferPtr();

    exploitClassInUse(conf_data, num_classes_, num_priors_, conf_thresh_unnormalized_, class_in_use_.data());

    constexpr const bool IS_CONF_THRESH_UNNORMALIZED = true;
    detection(loc_dim_.n, num_classes_, share_location_, nms_threshold_, background_label_id_, nms_top_k_, keep_top_k_,
              static_cast<PriorBoxCodingType>(code_type_), confidence_threshold_, nms_eta_, variance_encoded_in_target_,
              num_priors_, loc_data, conf_data, prior_data, out_0->getBufferPtr(), output_channel, output_height,
              output_width, IS_CONF_THRESH_UNNORMALIZED, conf_thresh_unnormalized_, &all_decode_bboxes_, &class_in_use_);

    // detection output fixed 1*1*target_num*7, even though batch is bigger than one
    Dim4 expected_out_dim = {1, output_channel, output_height, output_width};
    if (!isDimsSame(output->getDim(), expected_out_dim)) {
        output->reconfigureDim(expected_out_dim);
    }

    return Status::SUCCESS;
}

Status NEONSigDet::release() {
    DEBUG_PRINT("called\n");

    return Status::SUCCESS;
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
