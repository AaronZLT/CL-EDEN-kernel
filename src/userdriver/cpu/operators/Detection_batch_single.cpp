#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/cpu/operators/neon_impl/DetectionOutput.hpp"
#include "userdriver/cpu/operators/Detection_batch_single.hpp"

namespace enn {
namespace ud {
namespace cpu {

DetectionBatchSingle::DetectionBatchSingle(const PrecisionType &precision) {
    precision_ = precision;
}

Status DetectionBatchSingle::initialize(const Dim4 &, const Dim4 &, const Dim4 &, const Dim4 &, const uint32_t &num_classes,
                                        const bool &share_location, const float &nms_threshold,
                                        const int32_t &background_label_id, const int32_t &nms_top_k,
                                        const int32_t &keep_top_k, const uint32_t &code_type,
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

Status DetectionBatchSingle::execute(const std::shared_ptr<ITensor> input_location,
                                     const std::shared_ptr<ITensor> input_confidence,
                                     const std::shared_ptr<ITensor> input_prior, std::shared_ptr<ITensor> output) {
    uint32_t output_channel = 0;
    uint32_t output_height = 0;
    uint32_t output_width = 0;

    auto in_0 = std::static_pointer_cast<NEONTensor<float>>(input_location);
    auto in_1 = std::static_pointer_cast<NEONTensor<float>>(input_confidence);
    auto in_2 = std::static_pointer_cast<NEONTensor<float>>(input_prior);
    auto out_0 = std::static_pointer_cast<NEONTensor<float>>(output);

    auto in_loca = in_0->getBufferPtr();
    auto in_conf = in_1->getBufferPtr();
    auto in_pbox = in_2->getBufferPtr();
    auto outdata = out_0->getBufferPtr();

    DEBUG_PRINT("DetectionBatchSingle buffer info:\n\t in[0] %p (%" PRIu64 ")\n\t in[1] %p (%" PRIu64
                ")\n\t in[2] %p (%" PRIu64 ")\n\tout[0] %p (%" PRIu64 ")\n",
                in_loca, in_0->getNumOfBytes(), in_conf, in_1->getNumOfBytes(), in_pbox, in_2->getNumOfBytes(), outdata,
                out_0->getNumOfBytes());

    uint32_t num_priors = in_2->getDim().h / 4;

    detection(in_0->getDim().n, descriptor_.num_classes_, descriptor_.share_location_, descriptor_.nms_threshold_,
              descriptor_.background_label_id_, descriptor_.nms_top_k_, descriptor_.keep_top_k_,
              static_cast<PriorBoxCodingType>(descriptor_.code_type_), descriptor_.confidence_threshold_,
              descriptor_.nms_eta_, descriptor_.variance_encoded_in_target_, num_priors, in_loca, in_conf, in_pbox, outdata,
              output_channel, output_height, output_width);

    // detection output fixed 1*1*target_num*7, even though batch is bigger than one
    Dim4 out_dim = {1, output_channel, output_height, output_width};
    out_0->reconfigureDim(out_dim);

    return Status::SUCCESS;
}

Status DetectionBatchSingle::release() {
    return Status::SUCCESS;
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
