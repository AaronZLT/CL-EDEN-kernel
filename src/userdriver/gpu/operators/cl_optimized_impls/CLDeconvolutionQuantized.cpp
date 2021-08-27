#include "userdriver/gpu/operators/cl_optimized_impls/CLDeconvolutionQuantized.hpp"

namespace enn {
namespace ud {
namespace gpu {

constexpr int MAX_GROUP_SIZE = 128;

CLDeconvolutionQuantized::CLDeconvolutionQuantized(
    const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    DEBUG_PRINT("CLDeconvolutionQuantized is created");
    pad_ = {0, 0, 0, 0};
    stride_ = {0, 0};
    kernel_ = {0, 0};
}

Status CLDeconvolutionQuantized::initialize(const Dim4 &input_dim,
                                            const Dim4 &output_dim,
                                            const int &input_zeropoint,
                                            const std::shared_ptr<ITensor> filter,
                                            const std::shared_ptr<ITensor> bias,
                                            const Pad4 &padding,
                                            const Dim2 &stride,
                                            const uint32_t &group,
                                            const ActivationInfo &activate_info,
                                            const bool &weights_as_input,
                                            const bool &androidNN) {
    DEBUG_PRINT("CLDeconvolutionQuantized::initialize() is called");
    filter_ = std::static_pointer_cast<CLTensor>(filter);
    bias_ = std::static_pointer_cast<CLTensor>(bias);
    pad_ = padding;
    stride_ = stride;
    group_ = group;
    activation_info_ = activate_info;
    weights_as_input_ = weights_as_input;
    androidNN_ = androidNN;

    Dim4 filter_dim = filter_->getDim();
    if (androidNN_) {
        kernel_.h = filter_dim.c;
        kernel_.w = filter_dim.h;
    } else {
        kernel_.h = filter_dim.h;
        kernel_.w = filter_dim.w;
    }

    Status state;
    if (precision_ == PrecisionType::UINT8) {
        if (androidNN_) {
          state = runtime_->setKernel(&kernel_align_, "weightAlign", precision_);
          CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "weightAlign setKernel failure\n");
        }
        state = runtime_->setKernel(&kernel_trans_, "matrixTrans", precision_);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "matrixTrans setKernel failure\n");
        state = runtime_->setKernel(&kernel_gemm_, "convBackGemmRXR", precision_);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "convBackGemmRXR setKernel failure\n");
        state = runtime_->setKernel(&kernel_convert_, "convertBottomDiff2", precision_);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "convertBottomDiff2 setKernel failure\n");
    } else {
        if (androidNN_) {
          state = runtime_->setKernel(&kernel_align_, "SIGNEDweightAlign", precision_);
          CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "weightAlign setKernel failure\n");
        }
        state = runtime_->setKernel(&kernel_trans_, "SIGNEDmatrixTrans", precision_);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "matrixTrans setKernel failure\n");
        state = runtime_->setKernel(&kernel_gemm_, "SIGNEDconvBackGemmRXR", precision_);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "convBackGemmRXR setKernel failure\n");
        state = runtime_->setKernel(&kernel_convert_, "SIGNEDconvertBottomDiff2", precision_);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "convertBottomDiff2 setKernel failure\n");
    }

    uint32_t filter_align_width = alignTo(input_dim.c / group_, 8);  // group_==1
    uint32_t filter_align_height = alignTo(kernel_.h * kernel_.w * output_dim.c, 8);
    uint32_t filter_count = filter_align_width * filter_align_height;
    Dim4 filter_buffer_dim = {filter_count, 1, 1, 1};
    filter_buffer_ = std::make_shared<CLTensor>(runtime_, precision_, filter_->getDataType(),
                                                filter_buffer_dim);

    uint32_t trans_input_align_width = alignTo(input_dim.c / group_, 8) * group_;
    uint32_t size_trans_input =
            input_dim.n * 1 * (input_dim.h * input_dim.w) * trans_input_align_width;
    Dim4 input_trans_buffer_dim = {size_trans_input, 1, 1, 1};
    input_trans_buffer_ = std::make_shared<CLTensor>(runtime_, precision_, filter_->getDataType(),
                                                     input_trans_buffer_dim);

    if (precision_ == PrecisionType::UINT8) {
        std::vector<unsigned char> filter_zp_arr(filter_count, filter_->getZeroPoint());
        std::vector<unsigned char> input_zp_arr(size_trans_input, input_zeropoint);
        filter_buffer_->writeData(filter_zp_arr.data());
        input_trans_buffer_->writeData(input_zp_arr.data());
    } else {
        std::vector<char> filter_zp_arr(filter_count, filter_->getZeroPoint());
        std::vector<char> input_zp_arr(size_trans_input, input_zeropoint);
        filter_buffer_->writeData(filter_zp_arr.data());
        input_trans_buffer_->writeData(input_zp_arr.data());
    }

    uint32_t output_convert_size =
        input_dim.n * input_dim.h * input_dim.w * output_dim.c * kernel_.h * kernel_.w;
    Dim4 output_convert_buffer_dim = {output_convert_size, 1, 1, 1};
    output_convert_buffer_ = std::make_shared<CLTensor>(
        runtime_, precision_, DataType::INT32, output_convert_buffer_dim);

    return Status::SUCCESS;
}

Status CLDeconvolutionQuantized::execute(const std::shared_ptr<ITensor> input,
                                         std::shared_ptr<ITensor> output) {
    DEBUG_PRINT("CLDeconvolutionQuantized::execute() is called");
    auto input_tensor = std::static_pointer_cast<CLTensor>(input);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output);
    auto bias_tensor = std::static_pointer_cast<CLTensor>(bias_);
    auto input_data = input_tensor->getDataPtr();
    auto output_data = output_tensor->getDataPtr();
    auto filter_data = filter_->getDataPtr();
    auto bias_data = bias_tensor->getDataPtr();
    auto input_dim = input_tensor->getDim();
    auto output_dim = output_tensor->getDim();

    double real_multiplier = 0.0;
    GetQuantizedConvolutionMultipler(input_tensor->getScale(),
                                     filter_->getScale(),
                                     bias_tensor->getScale(),
                                     output_tensor->getScale(),
                                     &real_multiplier);
    int32_t output_multiplier = 0;
    int output_shift = 0;
    QuantizeMultiplierSmallerThanOneExp(real_multiplier, &output_multiplier, &output_shift);
    output_shift *= -1;

    int32_t act_min;
    int32_t act_max;
    if (precision_ == PrecisionType::UINT8) {
        CalculateActivationRangeUint8(activation_info_.activation(),
                                      output_tensor->getScale(),
                                      output_tensor->getZeroPoint(),
                                      &act_min,
                                      &act_max);
    } else {  // precision_ == PrecisionType::INT8
        CalculateActivationRangeInt8(activation_info_.activation(),
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
    uint32_t trans_height = output_dim.c / group_ * kernel_.h * kernel_.w;
    uint32_t filter_batch = 0;

    Status state = Status::FAILURE;
    if (androidNN_) {  // TODO(wuke): weights_as_input_
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

        state = runtime_->setKernelArg(kernel_trans_.get(),
                                       filter_data,
                                       filter_buffer_->getDataPtr(),
                                       filter_align_width,
                                       filter_batch,
                                       group_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "kernel_trans_ setKernelArg failure\n");
        state = runtime_->enqueueKernel(
            kernel_trans_.get(), 3, global_weight_trans, local_weight_trans);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state,
                                  "kernel_trans_ execute kernel failure\n");
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
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "kernel_trans_ setKernelArg failure\n");
        state =
            runtime_->enqueueKernel(kernel_trans_.get(), 3, global_input_trans, local_input_trans);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state,
                                  "kernel_trans_ execute kernel failure\n");
    }

    // matrix multiplication
    uint32_t col = alignTo(input_dim.c / group_, 8);
    uint32_t row_filter = kernel_.h * kernel_.w * output_dim.c / group_;
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
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "kernel_gemm_ setKernelArg failure\n");
        state = runtime_->enqueueKernel(kernel_gemm_.get(), 3, global_gemm, local_gemm);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state,
                                  "kernel_gemm_ execute kernel failure\n");
    }

    // convert output
    // "paddingRight and paddingBottom in transpose conv may be less than 0 to resolve the
    // ambiguous output shape issue in the case of stride > 1", so we use pad_.t, pad_.l here
    size_t global_convert_output[3] = {0, 0, 0};
    size_t local_convert_output[3] = {1, 4, 32};
    global_convert_output[0] = output_dim.n * output_dim.c;
    global_convert_output[1] = alignTo(output_dim.h, 4);
    global_convert_output[2] = alignTo(output_dim.w, 32);
    state = runtime_->setKernelArg(kernel_convert_.get(),
                                   output_convert_buffer_->getDataPtr(),
                                   bias_data,
                                   output_data,
                                   output_dim.h,
                                   output_dim.w,
                                   output_dim.c,
                                   kernel_.h,
                                   kernel_.w,
                                   pad_.t,
                                   pad_.l,
                                   stride_.h,
                                   stride_.w,
                                   input_dim.h,
                                   input_dim.w,
                                   output_offset,
                                   output_multiplier,
                                   output_shift,
                                   act_min,
                                   act_max);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "kernel_convert_ setKernelArg failure\n");
    state = runtime_->enqueueKernel(
        kernel_convert_.get(), 3, global_convert_output, local_convert_output);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "kernel_convert_ execute kernel failure\n");

    return Status::SUCCESS;
}

Status CLDeconvolutionQuantized::release() {
    DEBUG_PRINT("CLDeconvolutionQuantized::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
