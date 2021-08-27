#include "userdriver/gpu/operators/cl_optimized_impls/CLDeconvolutionPerChannelQuantized.hpp"

namespace enn {
namespace ud {
namespace gpu {

constexpr int MAX_GROUP_SIZE = 128;

CLDeconvolutionPerChannelQuantized::CLDeconvolutionPerChannelQuantized(
    const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    DEBUG_PRINT("CLDeconvolutionPerChannelQuantized is created");
    pad_ = {0, 0, 0, 0};
    stride_ = {0, 0};
    kernel_ = {0, 0};
}

Status CLDeconvolutionPerChannelQuantized::initialize(const Dim4 &input_dim,
                                                      const Dim4 &output_dim,
                                                      const int &input_zeropoint,
                                                      const std::shared_ptr<ITensor> filter,
                                                      const std::shared_ptr<ITensor> bias,
                                                      const std::vector<float> &scales,
                                                      const Pad4 &padding,
                                                      const Dim2 &stride,
                                                      const uint32_t &group,
                                                      const ActivationInfo &activate_info,
                                                      const bool &weights_as_input,
                                                      const bool &androidNN) {
    DEBUG_PRINT("CLDeconvolutionPerChannelQuantized::initialize() is called");
    filter_ = std::static_pointer_cast<CLTensor>(filter);
    bias_ = std::static_pointer_cast<CLTensor>(bias);
    scales_ = scales;
    pad_ = padding;
    stride_ = stride;
    group_ = group;
    activation_info_ = activate_info;
    weights_as_input_ = weights_as_input;
    androidNN_ = androidNN;

    Dim4 filter_dim = filter_->getDim();
    if (androidNN_) {
        filter_dim = convertDimToNCHW(filter_dim);
        weight_nchw_ = std::make_shared<CLTensor>(runtime_,
                                                  precision_,
                                                  filter_->getDataType(),
                                                  filter_dim,
                                                  filter_->getDataOrder(),
                                                  filter_->getScale(),
                                                  filter_->getZeroPoint());
        if (!weights_as_input) {
            Status state = filter_->convertToNCHW(weight_nchw_);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "convertToNCHW failure\n");
        }
    }
    kernel_.h = filter_dim.h;
    kernel_.w = filter_dim.w;

    uint32_t filter_align_width = alignTo(input_dim.c / group_, 8);  // group_==1
    uint32_t filter_align_height = alignTo(kernel_.h * kernel_.w * output_dim.c, 8);
    uint32_t filter_count = filter_align_width * filter_align_height;
    Dim4 filter_buffer_dim = {filter_count, 1, 1, 1};
    filter_buffer_ =
        std::make_shared<CLTensor>(runtime_, precision_, DataType::INT8, filter_buffer_dim);
    std::vector<char> filter_zp_arr(filter_count, filter_->getZeroPoint());
    filter_buffer_->writeData(filter_zp_arr.data());

    uint32_t trans_input_align_width = alignTo(input_dim.c / group_, 8) * group_;
    uint32_t size_trans_input =
        input_dim.n * 1 * (input_dim.h * input_dim.w) * trans_input_align_width;
    Dim4 input_trans_buffer_dim = {size_trans_input, 1, 1, 1};
    if (precision_ == PrecisionType::INT8) {
        input_trans_buffer_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::INT8,
                                                         input_trans_buffer_dim);
        std::vector<char> input_zp_arr(size_trans_input, input_zeropoint);
        input_trans_buffer_->writeData(input_zp_arr.data());
    } else {
        input_trans_buffer_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::UINT8,
                                                         input_trans_buffer_dim);
        std::vector<unsigned char> input_zp_arr(size_trans_input, input_zeropoint);
        input_trans_buffer_->writeData(input_zp_arr.data());
    }

    uint32_t output_convert_size =
        input_dim.n * input_dim.h * input_dim.w * output_dim.c * kernel_.h * kernel_.w;
    Dim4 output_convert_buffer_dim = {output_convert_size, 1, 1, 1};
    output_convert_buffer_ = std::make_shared<CLTensor>(
        runtime_, precision_, DataType::INT32, output_convert_buffer_dim);

    output_multiplier_ =
        std::make_shared<CLTensor>(runtime_, precision_, DataType::INT32, bias_->getDim());
    output_shift_ =
        std::make_shared<CLTensor>(runtime_, precision_, DataType::INT32, bias_->getDim());

    output_multiplier_data_.reset(new int32_t[output_dim.c], std::default_delete<int32_t[]>());
    output_shift_data_.reset(new int32_t[output_dim.c], std::default_delete<int32_t[]>());

    Status state;
    if (weights_as_input_) {
      state = runtime_->setKernel(&kernel_align_, "SIGNEDweightAlign", precision_);
      CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "weightAlign setKernel failure\n");
    } else {
      state = runtime_->setKernel(&kernel_trans_signed_, "SIGNEDmatrixTrans", precision_);
      CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "SIGNEDmatrixTrans setKernel failure\n");
    }

    if (precision_ == PrecisionType::INT8) {
        state = runtime_->setKernel(&kernel_trans_, "SIGNEDmatrixTrans", precision_);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "SIGNEDmatrixTrans setKernel failure\n");
        state = runtime_->setKernel(&kernel_gemm_, "SIGNEDconvBackGemmRXR", precision_);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "SIGNEDconvBackGemmRXR setKernel failure\n");
        state = runtime_->setKernel(&kernel_convert_pre_channel_,
                                    "SIGNEDconvertBottomDiff2_per_channel", precision_);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "convertBottomDiff2 setKernel failure\n");
    } else {
      state = runtime_->setKernel(&kernel_trans_, "matrixTrans", precision_);
      CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "matrixTrans setKernel failure\n");
      state = runtime_->setKernel(&kernel_gemm_, "PERCHANNELconvBackGemmRXR", precision_);
      CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "PERCHANNELconvBackGemmRXR setKernel failure\n");
      state = runtime_->setKernel(
          &kernel_convert_pre_channel_, "convertBottomDiff2_per_channel", precision_);
      CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "convertBottomDiff2 setKernel failure\n");
    }

    return Status::SUCCESS;
}

Status CLDeconvolutionPerChannelQuantized::execute(const std::shared_ptr<ITensor> input,
                                                   std::shared_ptr<ITensor> output) {
    DEBUG_PRINT("CLDeconvolutionPerChannelQuantized::execute() is called");
    if (weights_as_input_ && androidNN_) {
        Status state = filter_->convertToNCHW(weight_nchw_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "convertToNCHW failure\n");
    }

    auto input_tensor = std::static_pointer_cast<CLTensor>(input);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output);
    auto bias_tensor = std::static_pointer_cast<CLTensor>(bias_);
    auto input_data = input_tensor->getDataPtr();
    auto output_data = output_tensor->getDataPtr();
    auto filter_data = androidNN_ ? weight_nchw_->getDataPtr() : filter_->getDataPtr();
    auto bias_data = bias_tensor->getDataPtr();
    auto input_dim = input_tensor->getDim();
    auto output_dim = output_tensor->getDim();
    auto filter_dim = androidNN_ ? weight_nchw_->getDim() : filter_->getDim();

    for (int i = 0; i < output_dim.c; i++) {
        double real_multiplier = 0.0;
        float filter_scale_channel = scales_[i];
        float bias_scale_channel = scales_[i] * input_tensor->getScale();
        GetQuantizedConvolutionMultipler(input_tensor->getScale(),
                                         filter_scale_channel,
                                         bias_scale_channel,
                                         output_tensor->getScale(),
                                         &real_multiplier);
        QuantizeMultiplierSmallerThanOneExp(
            real_multiplier, &output_multiplier_data_.get()[i], &output_shift_data_.get()[i]);
    }
    output_multiplier_->writeData(output_multiplier_data_.get());
    output_shift_->writeData(output_shift_data_.get());

    int32_t act_min = 0, act_max = 0;
    if (precision_ == PrecisionType::INT8) {
        CalculateActivationRangeInt8(activation_info_.activation(),
                                     output_tensor->getScale(),
                                     output_tensor->getZeroPoint(),
                                     &act_min,
                                     &act_max);
    } else {
        CalculateActivationRangeUint8(activation_info_.activation(),
                                      output_tensor->getScale(),
                                      output_tensor->getZeroPoint(),
                                      &act_min,
                                      &act_max);
    }

    int32_t input_offset = -input_tensor->getZeroPoint();
    int32_t weight_offset = -filter_->getZeroPoint();
    int32_t output_offset = output_tensor->getZeroPoint();

    // filter trans
    uint32_t filter_align_width = alignTo(input_dim.c / group_, 8);
    uint32_t trans_width = input_dim.c / group_;
    uint32_t trans_height = output_dim.c / group_ * filter_dim.h * filter_dim.w;
    uint32_t filter_batch = 0;

    Status state = Status::FAILURE;
    if (weights_as_input_) {
        size_t global_weight_align[2] = {0, 0};
        global_weight_align[0] = trans_height;
        global_weight_align[1] = trans_width;

        state = runtime_->setKernelArg(kernel_align_.get(),
                                       filter_data,
                                       filter_buffer_->getDataPtr(),
                                       filter_align_width,
                                       trans_width);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "kernel_align_ setKernelArg failure\n");
        state = runtime_->enqueueKernel(kernel_align_.get(), 2, global_weight_align, NULL);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state,
                                  "kernel_align_ execute kernel failure\n");
    } else {
        size_t global_weight_trans[3] = {0, 0, 0};
        size_t local_weight_trans[3] = {0, 0, 0};
        global_weight_trans[0] = group_;
        global_weight_trans[1] = trans_height;
        global_weight_trans[2] = trans_width;
        local_weight_trans[2] = findMaxFactor(global_weight_trans[2], MAX_GROUP_SIZE);
        local_weight_trans[1] =
            findMaxFactor(global_weight_trans[1], MAX_GROUP_SIZE / local_weight_trans[2]);
        local_weight_trans[0] =
            findMaxFactor(global_weight_trans[0],
                          MAX_GROUP_SIZE / (local_weight_trans[2] * local_weight_trans[1]));

        state = runtime_->setKernelArg(kernel_trans_signed_.get(),
                                       filter_data,
                                       filter_buffer_->getDataPtr(),
                                       filter_align_width,
                                       filter_batch,
                                       group_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "kernel_trans_signed setKernelArg failure\n");
        state = runtime_->enqueueKernel(
                kernel_trans_signed_.get(), 3, global_weight_trans, local_weight_trans);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state,
                                  "kernel_trans_signed execute kernel failure\n");
    }

    // input trans
    trans_width = input_dim.c / group_;
    trans_height = input_dim.h * input_dim.w;
    uint32_t trans_width_align = alignTo(input_dim.c / group_, 8);
    size_t global_input_trans[3] = {0, 0, 0};
    size_t local_input_trans[3] = {0, 0, 0};
    global_input_trans[0] = group_;
    global_input_trans[1] = trans_height;
    global_input_trans[2] = trans_width;
    local_input_trans[2] = findMaxFactor(global_input_trans[2], MAX_GROUP_SIZE);
    local_input_trans[1] =
        findMaxFactor(global_input_trans[1], MAX_GROUP_SIZE / local_input_trans[2]);
    local_input_trans[0] = findMaxFactor(
        global_input_trans[0], MAX_GROUP_SIZE / (local_input_trans[2] * local_input_trans[1]));
    for (uint32_t i = 0; i < input_dim.n; i++) {
        state = runtime_->setKernelArg(kernel_trans_.get(),
                                       input_data,
                                       input_trans_buffer_->getDataPtr(),
                                       trans_width_align,
                                       i,
                                       group_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "kernel_trans setKernelArg failure\n");
        state =
            runtime_->enqueueKernel(kernel_trans_.get(), 3, global_input_trans, local_input_trans);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state,
                                  "kernel_trans execute kernel failure\n");
    }

    // matrix multiplication
    uint32_t col = alignTo(input_dim.c / group_, 8);
    uint32_t row_filter = filter_dim.h * filter_dim.w * output_dim.c / group_;
    uint32_t row_input = input_dim.h * input_dim.w;
    size_t global_gemm[3] = {0, 0, 0};
    size_t local_gemm[3] = {1, 4, 32};
    global_gemm[0] = group_;
    global_gemm[1] = alignTo(ceil(row_filter / 8.0), 4);  // Each thread process 8 elements
    global_gemm[2] = alignTo(row_input, 32);
    for (uint32_t i = 0; i < input_dim.n; i++) {
        state = runtime_->setKernelArg(kernel_gemm_.get(),
                                       filter_buffer_->getDataPtr(),
                                       input_trans_buffer_->getDataPtr(),
                                       output_convert_buffer_->getDataPtr(),
                                       row_filter,
                                       col,
                                       row_input,
                                       i,
                                       group_,
                                       input_offset,
                                       weight_offset);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "kernel_gemm setKernelArg failure\n");
        state = runtime_->enqueueKernel(kernel_gemm_.get(), 3, global_gemm, local_gemm);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state,
                                  "kernel_gemm execute kernel failure\n");
    }

    // convert output
    // "paddingRight and paddingBottom in transpose conv may be less than 0 to resolve the
    // ambiguous output shape issue in the case of stride > 1", so we use pad_.t, pad_.l here
    size_t global_convert_output[3] = {0, 0, 0};
    size_t local_convert_output[3] = {1, 4, 32};
    global_convert_output[0] = output_dim.n * output_dim.c;
    global_convert_output[1] = alignTo(output_dim.h, 4);
    global_convert_output[2] = alignTo(output_dim.w, 32);
    state = runtime_->setKernelArg(kernel_convert_pre_channel_.get(),
                                   output_convert_buffer_->getDataPtr(),
                                   bias_data,
                                   output_data,
                                   output_dim.h,
                                   output_dim.w,
                                   output_dim.c,
                                   filter_dim.h,
                                   filter_dim.w,
                                   pad_.t,
                                   pad_.l,
                                   stride_.h,
                                   stride_.w,
                                   input_dim.h,
                                   input_dim.w,
                                   output_offset,
                                   output_multiplier_->getDataPtr(),
                                   output_shift_->getDataPtr(),
                                   act_min,
                                   act_max);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "kernel_convert_ setKernelArg failure\n");
    state = runtime_->enqueueKernel(
        kernel_convert_pre_channel_.get(), 3, global_convert_output, local_convert_output);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "kernel_convert_ execute kernel failure\n");

    return Status::SUCCESS;
}

Status CLDeconvolutionPerChannelQuantized::release() {
    DEBUG_PRINT("CLDeconvolutionPerChannelQuantized::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
