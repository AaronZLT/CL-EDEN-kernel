#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLGEMMConvolution.hpp"

namespace enn {
namespace ud {
namespace gpu {

CLGEMMConvolution::CLGEMMConvolution(const std::shared_ptr<CLRuntime> runtime,
                                     const PrecisionType &precision,
                                     const Dim4 &input_dim,
                                     const Dim4 &output_dim) :
    runtime_(runtime),
    precision_(precision), input_dim_(input_dim), output_dim_(output_dim) {
    ENN_DBG_PRINT("CLGEMMConvolution is created");
    convert_out_dim_ = {0, 0, 0, 0};
    stride_ = {0, 0};
    kernel_ = {0, 0};
    weights_as_input_ = false;
    androidNN_ = false;
    isNCHW_ = true;
}

Status CLGEMMConvolution::initialize(const Pad4 &padding,
                                     const Dim2 &stride,
                                     const uint32_t &group_size,
                                     const uint32_t &axis,
                                     const Dim2 &dilation,
                                     const std::shared_ptr<ITensor> weight,
                                     const std::shared_ptr<ITensor> bias,
                                     const ActivationInfo &activate_info,
                                     const bool &weights_as_input,
                                     const bool &androidNN,
                                     const bool &isNCHW) {
    ENN_DBG_PRINT("CLGEMMConvolution::initialize() is called");
    weights_as_input_ = weights_as_input;
    androidNN_ = androidNN;
    isNCHW_ = isNCHW;
    activation_info_ = activate_info;

    Status state;
    auto weight_tensor = std::static_pointer_cast<CLTensor>(weight);
    conv_descriptor_.filter_ = weight_tensor;
    Dim4 weight_size = weight_tensor->getDim();
    if (androidNN_ || !isNCHW_) {
        weight_size = convertDimToNCHW(weight_size);
        weight_nchw_ = std::make_shared<CLTensor>(runtime_,
                                                  precision_,
                                                  weight_tensor->getDataType(),
                                                  weight_size,
                                                  weight_tensor->getDataOrder(),
                                                  weight_tensor->getScale(),
                                                  weight_tensor->getZeroPoint());
        if (!weights_as_input_) {
            state = weight_tensor->convertToNCHW(weight_nchw_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "convertToNCHW failure\n");
        }
    }
    auto bias_tensor = std::static_pointer_cast<CLTensor>(bias);
    conv_descriptor_.bias_ = bias_tensor;

    stride_ = stride;
    kernel_ = {weight_size.h, weight_size.w};

    conv_descriptor_.num_output_ = weight_size.n;
    conv_descriptor_.pad_right_ = padding.r;
    conv_descriptor_.pad_left_ = padding.l;
    conv_descriptor_.pad_top_ = padding.t;
    conv_descriptor_.pad_bottom_ = padding.b;
    conv_descriptor_.kernel_height_ = weight_size.h;
    conv_descriptor_.kernel_width_ = weight_size.w;
    conv_descriptor_.stride_height_ = stride.h;
    conv_descriptor_.stride_width_ = stride.w;
    conv_descriptor_.group_ = group_size;
    conv_descriptor_.axis_ = axis;
    conv_descriptor_.dilation_ = dilation;

    if (weights_as_input_ == false) {
        Status state = alignWeight();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "alignWeight failure\n");
    }

    pad_convert_executor_ = std::make_shared<CLPadConvert>(runtime_, precision_);
    pad_convert_executor_->initialize(
        input_dim_, padding, kernel_, stride_, group_size, {1, 1, 1, 1}, true, CLPadConvert::PadConvertType::GEMM);

    convert_out_dim_.n = input_dim_.n;
    convert_out_dim_.c = 1;
    convert_out_dim_.h = (output_dim_.h * output_dim_.w + 11) / 12 * 12 * group_size;
    convert_out_dim_.w = (kernel_.h * kernel_.w * input_dim_.c / group_size + 15) / 16 * 16;
    convert_output_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, convert_out_dim_);

    if (activation_info_.isEnabled()) {
        if (activation_info_.activation() == ActivationInfo::ActivationType::RELU) {
            state = runtime_->setKernel(&gemm_kernel_, "RELUgemmBlocked", precision_);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel failure\n");
        } else if (activation_info_.activation() == ActivationInfo::ActivationType::RELU6 &&
                   precision_ == PrecisionType::FP16) {
            state = runtime_->setKernel(&gemm_kernel_, "RELU6gemmBlocked", precision_);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel failure\n");
        } else {
            state = runtime_->setKernel(&gemm_kernel_, "gemmBlocked", precision_);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel failure\n");
        }
    } else {
        state = runtime_->setKernel(&gemm_kernel_, "gemmBlocked", precision_);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel failure\n");
    }
    return state;
}

Status CLGEMMConvolution::execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) {
    ENN_DBG_PRINT("CLGEMMConvolution::execute() is called");
    auto input_tensor = std::static_pointer_cast<CLTensor>(input);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output);

    Status state = Status::FAILURE;
    if (weights_as_input_ == true) {
        if (androidNN_ || !isNCHW_) {
            state = conv_descriptor_.filter_->convertToNCHW(weight_nchw_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "convertToNCHW failure\n");
        }
        state = alignWeight();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "alignWeight failure\n");
    }

    return conv2DGPU(input_tensor, output_tensor);
}

Status CLGEMMConvolution::alignWeight() {
    ENN_DBG_PRINT("CLGEMMConvolution is in alignWeight");
    auto weight_size = (androidNN_ || !isNCHW_) ? weight_nchw_->getDim() : conv_descriptor_.filter_->getDim();
    uint32_t group_size = conv_descriptor_.group_;
    uint32_t srcUnitCount = kernel_.w * kernel_.h * input_dim_.c / group_size;
    uint32_t dstAlignCount = alignTo(srcUnitCount, 16);
    uint32_t groupTopChannel = weight_size.n / group_size;
    uint32_t alignedGroupTopChannel = alignTo(groupTopChannel, 16);
    uint32_t alignWeightHeight = (kernel_.h * kernel_.w * input_dim_.c / group_size + 15) / 16 * 16;
    uint32_t alignWeightWidth = (output_dim_.c / group_size + 15) / 16 * 16 * group_size;
    uint32_t weightCount = alignWeightHeight * alignWeightWidth;
    Dim4 converted_filter_dim = {weightCount, 1, 1, 1};
    converted_filter_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, converted_filter_dim);
    cl_mem dst = converted_filter_->getDataPtr();
    cl_mem src = (androidNN_ || !isNCHW_) ? weight_nchw_->getDataPtr() : conv_descriptor_.filter_->getDataPtr();
    Status state = runtime_->setKernel(&align_weight_kernel_, "alignWeight_gemm", precision_);
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel failure\n");
    state = alignWeightGEMM(src, dst, srcUnitCount, weight_size.n, dstAlignCount, groupTopChannel, alignedGroupTopChannel);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "align weight failure\n");
    return Status::SUCCESS;
}

Status CLGEMMConvolution::alignWeightGEMM(const cl_mem &src,
                                          cl_mem &dst,
                                          const uint32_t &srcWidth,
                                          const uint32_t &srcHeight,
                                          const uint32_t &dstWidth,
                                          const uint32_t &groupTopChannel,
                                          const uint32_t &alignedGroupTopChannel) {
    ENN_DBG_PRINT("CLGEMMConvolution is in alignWeightGEMM");
    auto unalignedSrc = src;
    auto alignedDst = dst;

    size_t local_align_weight[2];
    local_align_weight[0] = 1;
    local_align_weight[1] = 12;

    size_t global_align_weight[2];
    global_align_weight[0] = srcHeight;
    global_align_weight[1] = alignTo(ceil(srcWidth / 8.0), local_align_weight[1]);

    Status state = runtime_->setKernelArg(
        align_weight_kernel_.get(), unalignedSrc, alignedDst, srcWidth, dstWidth, groupTopChannel, alignedGroupTopChannel);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "set kernel failure\n");
    state = runtime_->enqueueKernel(align_weight_kernel_.get(), 2, global_align_weight, local_align_weight);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");

    return Status::SUCCESS;
}

Status CLGEMMConvolution::GEMMRun(const std::shared_ptr<CLTensor> input, std::shared_ptr<CLTensor> output) {
    auto input_data = input->getDataPtr();
    auto output_data = output->getDataPtr();

    auto input_dim = input->getDim();
    auto output_dim = output->getDim();
    int group = conv_descriptor_.group_;

    cl_mem weight_data = converted_filter_->getDataPtr();
    cl_mem bias_data = conv_descriptor_.bias_->getDataPtr();

    int align_input_width = input_dim.w;
    int align_input_height = input_dim.h;
    int align_weight_width = alignTo(output_dim.c / group, 16) * group;
    int temp_width = output_dim.h * output_dim.w;
    int out_channel = output_dim.c;

    size_t global_conv[3];
    global_conv[0] = output_dim.n;
    global_conv[1] = align_weight_width / group / 4;
    global_conv[2] = align_input_height / group;

    size_t local_conv[3];
    local_conv[0] = 1;
    local_conv[1] = 1;
    local_conv[2] = 12;

    Status state;
    for (int i = 0; i < group; i++) {
        state = runtime_->setKernelArg(gemm_kernel_.get(),
                                       input_data,
                                       weight_data,
                                       bias_data,
                                       output_data,
                                       align_input_width,
                                       out_channel,
                                       temp_width,
                                       group,
                                       i);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "set kernel failure\n");
        state = runtime_->enqueueKernel(gemm_kernel_.get(), 3, global_conv, local_conv);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
    }

    return Status::SUCCESS;
}

Status CLGEMMConvolution::conv2DGPU(const std::shared_ptr<CLTensor> input, std::shared_ptr<CLTensor> output) {
    // do pad and img2col
    Status state = pad_convert_executor_->execute(input, convert_output_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "pad convert failure\n");

    // gemm
    state = GEMMRun(convert_output_, output);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "gemm failure\n");

    ENN_DBG_PRINT("CLGEMMConvolution is finished");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
