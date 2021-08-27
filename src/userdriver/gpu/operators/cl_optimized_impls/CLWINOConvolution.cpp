#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLWINOConvolution.hpp"
#include "userdriver/common/operator_interfaces/common/Debug.hpp"

namespace enn {
namespace ud {
namespace gpu {

CLWINOConvolution::CLWINOConvolution(const std::shared_ptr<CLRuntime> runtime,
                                     const PrecisionType &precision,
                                     const Dim4 &input_dim,
                                     const Dim4 &output_dim) :
    runtime_(runtime),
    precision_(precision), input_dim_(input_dim), output_dim_(output_dim) {
    ENN_DBG_PRINT("CLWINOConvolution is created");
    top_channel_need_aligned_ = false;
    branch_number_ = 0;
    speed_up_ = 0;
    bifrost_speedup_ = false;
    coalescing_feature_height_ = 0;
    compute_output_number_ = 0;
    splite_number_ = 0;
    wgrad_M_ = 0;
    wgrad_stride_ = 0;
    wgrad_tile_size_ = 0;
    wgrad_align_height_ = 0;
    wgrad_align_width_ = 0;
    wgrad_tile_height_ = 0;
    wgrad_tile_width_ = 0;
    weights_as_input_ = false;
    androidNN_ = false;
    isNCHW_ = true;
}

Status CLWINOConvolution::initialize(const Pad4 &padding,
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
    ENN_DBG_PRINT("CLWINOConvolution::initialize() is called");
    runtime_->resetIntraBuffer();
    auto weight_tensor = std::static_pointer_cast<CLTensor>(weight);
    auto bias_tensor = std::static_pointer_cast<CLTensor>(bias);
    conv_descriptor_.filter_ = weight_tensor;
    conv_descriptor_.bias_ = bias_tensor;

    weights_as_input_ = weights_as_input;
    androidNN_ = androidNN;
    isNCHW_ = isNCHW;

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
            Status state = weight_tensor->convertToNCHW(weight_nchw_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "convertToNCHW failure\n");
        }
    }

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
    activation_info_ = activate_info;

    top_channel_need_aligned_ = false;
    splite_number_ = WINO_TILE_ORIGINAL;
    branch_number_ = WINO_TILE_ORIGINAL;
    bifrost_speedup_ = false;
    coalescing_feature_height_ = WINO_NO_COALESCING;
    if (weights_as_input == false) {
        return initKernelWeight();
    } else {
        return Status::SUCCESS;
    }
}

Status CLWINOConvolution::initKernelWeight() {
    ENN_DBG_PRINT("CLWINOConvolution::initKernelWeight is execute");
    // initial kernel
    int pad_height = conv_descriptor_.pad_bottom_ + conv_descriptor_.pad_top_;
    int pad_width = conv_descriptor_.pad_left_ + conv_descriptor_.pad_right_;
    wgrad_M_ = 2;
    wgrad_stride_ = 2;
    wgrad_tile_size_ = 4;
    wgrad_align_height_ = alignTo(input_dim_.h + pad_height, wgrad_M_);
    wgrad_align_width_ = alignTo(input_dim_.w + pad_width, wgrad_M_);
    wgrad_tile_height_ = (wgrad_align_height_ - wgrad_tile_size_) / wgrad_M_ + 1;
    wgrad_tile_width_ = (wgrad_align_width_ - wgrad_tile_size_) / wgrad_M_ + 1;
    int output_channel = output_dim_.c;
    int batch = output_dim_.n;
    int input_channel = input_dim_.c;
    int group = conv_descriptor_.group_;
    int group_input_channel = input_dim_.c / group;
    auto rt_bytes = runtime_->getRuntimeTypeBytes(precision_);
    int weight_count = output_channel * group_input_channel * wgrad_tile_size_ * wgrad_tile_size_;
    branch_number_ = WINO_TILE_ORIGINAL;
    if ((output_channel / group) % WINO_TILE_SIZE_2 == 0) {
        branch_number_ = WINO_TILE_1X2;
        coalescing_feature_height_ = WINO_NO_COALESCING;
        compute_output_number_ = WINO_TILE_SIZE_2;
    } else {
        branch_number_ = WINO_TILE_ORIGINAL;
        coalescing_feature_height_ = WINO_NO_COALESCING;
        compute_output_number_ = WINO_TILE_SIZE_1;
    }

    speed_up_ = DEFAULT_SPEED_UP;
    if (group == 1 && runtime_->isBifrost()) {
        if (output_channel % WINO_TILE_SIZE_8 == 0 || output_channel % WINO_TILE_SIZE_4 == 0) {
            bifrost_speedup_ = true;
            long long flops = (long long)output_dim_.h * output_dim_.w * output_dim_.c * input_dim_.c * 3 * 3;
            if (runtime_->isMakalu() && output_channel % WINO_TILE_SIZE_4 == 0 && precision_ == PrecisionType::FP16 &&
                flops < MAKALU_COMPUTING_THRESHOLD) {
                speed_up_ = MAKALU_SPEED_UP_1X2;
                branch_number_ = WINO_TILE_1X2_MAKALU;
                coalescing_feature_height_ = WINO_COALESCING_SIZE_MAKALU_SINGLE_LINE;
                compute_output_number_ = WINO_TILE_SIZE_2;

                Status state = runtime_->setKernel(&winograd_convert_kernel_bifrost_, "Fast3x3_1_makalu_1x2", precision_);
                CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_1_makalu_1x2 failure\n");

                if (activation_info_.isEnabled()) {
                    if (activation_info_.activation() == ActivationInfo::ActivationType::RELU) {
                        state = runtime_->setKernel(&winograd_multi_kernel_split_, "RELUFast3x3_2_makalu_1x2", precision_);
                        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel RELUFast3x3_2_makalu_1x2 failure\n");
                    } else if (activation_info_.activation() == ActivationInfo::ActivationType::RELU6 &&
                               precision_ == PrecisionType::FP16) {
                        state = runtime_->setKernel(&winograd_multi_kernel_split_, "RELU6Fast3x3_2_makalu_1x2", precision_);
                        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel RELU6Fast3x3_2_makalu_1x2 failure\n");
                    } else {
                        state = runtime_->setKernel(&winograd_multi_kernel_split_, "Fast3x3_2_makalu_1x2", precision_);
                        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2_makalu_1x2 failure\n");
                    }
                } else {
                    state = runtime_->setKernel(&winograd_multi_kernel_split_, "Fast3x3_2_makalu_1x2", precision_);
                    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2_makalu_1x2 failure\n");
                }
            } else if (runtime_->isMakalu() && output_channel % WINO_TILE_SIZE_8 == 0 && precision_ == PrecisionType::FP16) {
                speed_up_ = MAKALU_SPEED_UP;
                branch_number_ = WINO_TILE_4X4;
                coalescing_feature_height_ = WINO_COALESCING_SIZE_MAKALU;
                compute_output_number_ = WINO_TILE_SIZE_4;
                splite_number_ = WINO_SPLIT_SIZE_4;
                Status state = runtime_->setKernel(&winograd_convert_kernel_bifrost_, "Fast3x3_1_4x4", precision_);
                CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_1_4x4 failure\n");

                state = runtime_->setKernel(&winograd_multi_kernel_split_, "Fast3x3_2_optimized_4x4_splite", precision_);
                CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2_optimized_4x4_splite failure\n");

                if (activation_info_.isEnabled()) {
                    if (activation_info_.activation() == ActivationInfo::ActivationType::RELU) {
                        state = runtime_->setKernel(
                            &winograd_multi_kernel_merge_, "RELUFast3x3_2_optimized_4x4_merge", precision_);
                        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2_optimized_4x4_merge failure\n");
                    } else if (activation_info_.activation() == ActivationInfo::ActivationType::RELU6 &&
                               precision_ == PrecisionType::FP16) {
                        state = runtime_->setKernel(
                            &winograd_multi_kernel_merge_, "RELU6Fast3x3_2_optimized_4x4_merge", precision_);
                        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2_optimized_4x4_merge failure\n");
                    } else {
                        state =
                            runtime_->setKernel(&winograd_multi_kernel_merge_, "Fast3x3_2_optimized_4x4_merge", precision_);
                        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2_optimized_4x4_merge failure\n");
                    }
                } else {
                    state = runtime_->setKernel(&winograd_multi_kernel_merge_, "Fast3x3_2_optimized_4x4_merge", precision_);
                    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2_optimized_4x4_merge failure\n");
                }
            } else if (precision_ == PrecisionType::FP32 && output_channel % WINO_TILE_SIZE_8 == 0) {
                speed_up_ = BIFROST_SPEED_UP;
                branch_number_ = WINO_TILE_1X8;
                coalescing_feature_height_ = WINO_NO_COALESCING;
                compute_output_number_ = WINO_TILE_SIZE_8;
                splite_number_ = WINO_SPLIT_SIZE_4;
                Status state = runtime_->setKernel(&winograd_convert_kernel_bifrost_, "Fast3x3_1_1x8", precision_);
                CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_1_1x8 failure\n");
                state = runtime_->setKernel(&winograd_multi_kernel_split_, "Fast3x3_2_optimized_1x8_splite", precision_);
                CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2_optimized_1x8_splite failure\n");

                if (activation_info_.isEnabled()) {
                    if (activation_info_.activation() == ActivationInfo::ActivationType::RELU) {
                        state = runtime_->setKernel(
                            &winograd_multi_kernel_merge_, "RELUFast3x3_2_optimized_1x8_merge", precision_);
                        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2_optimized_1x8_merge failure\n");
                    } else if (activation_info_.activation() == ActivationInfo::ActivationType::RELU6 &&
                               precision_ == PrecisionType::FP16) {
                        state = runtime_->setKernel(
                            &winograd_multi_kernel_merge_, "RELU6Fast3x3_2_optimized_1x8_merge", precision_);
                        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2_optimized_1x8_merge failure\n");
                    } else {
                        state =
                            runtime_->setKernel(&winograd_multi_kernel_merge_, "Fast3x3_2_optimized_1x8_merge", precision_);
                        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2_optimized_1x8_merge failure\n");
                    }
                } else {
                    state = runtime_->setKernel(&winograd_multi_kernel_merge_, "Fast3x3_2_optimized_1x8_merge", precision_);
                    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2_optimized_1x8_merge failure\n");
                }
            } else if (static_cast<double>(
                           (output_channel / WINO_TILE_SIZE_8) * WINO_SPLIT_SIZE_4 *
                           alignTo(static_cast<int>(
                                       ceil(static_cast<double>(wgrad_tile_height_ * wgrad_tile_width_) / WINO_TILE_SIZE_2)),
                                   24) /
                           24) /
                               20.0 >
                           20.0 &&
                       input_dim_.c > 64 && output_channel % WINO_TILE_SIZE_8 == 0) {
                speed_up_ = BIFROST_SPEED_UP;
                branch_number_ = WINO_TILE_2X8;
                coalescing_feature_height_ = WINO_COALESCING_SIZE;
                compute_output_number_ = WINO_TILE_SIZE_8;
                splite_number_ = WINO_SPLIT_SIZE_4;
                Status state = runtime_->setKernel(&winograd_convert_kernel_bifrost_, "Fast3x3_1_2x8", precision_);
                CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_1_2x8 failure\n");
                state = runtime_->setKernel(&winograd_multi_kernel_split_, "Fast3x3_2_optimized_2x8_splite", precision_);
                CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2_optimized_2x8_splite failure\n");

                if (activation_info_.isEnabled()) {
                    if (activation_info_.activation() == ActivationInfo::ActivationType::RELU) {
                        state = runtime_->setKernel(
                            &winograd_multi_kernel_merge_, "RELUFast3x3_2_optimized_2x8_merge", precision_);
                        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2_optimized_2x8_merge failure\n");
                    } else if (activation_info_.activation() == ActivationInfo::ActivationType::RELU6 &&
                               precision_ == PrecisionType::FP16) {
                        state = runtime_->setKernel(
                            &winograd_multi_kernel_merge_, "RELU6Fast3x3_2_optimized_2x8_merge", precision_);
                        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2_optimized_2x8_merge failure\n");
                    } else {
                        state =
                            runtime_->setKernel(&winograd_multi_kernel_merge_, "Fast3x3_2_optimized_2x8_merge", precision_);
                        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2_optimized_2x8_merge failure\n");
                    }
                } else {
                    state = runtime_->setKernel(&winograd_multi_kernel_merge_, "Fast3x3_2_optimized_2x8_merge", precision_);
                    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2_optimized_2x8_merge failure\n");
                }
            } else {
                speed_up_ = BIFROST_SPEED_UP;
                branch_number_ = WINO_TILE_2X4;
                coalescing_feature_height_ = WINO_COALESCING_SIZE;
                compute_output_number_ = WINO_TILE_SIZE_4;
                splite_number_ = WINO_SPLIT_SIZE_2;
                Status state = runtime_->setKernel(&winograd_convert_kernel_bifrost_, "Fast3x3_1_2x4", precision_);
                CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_1_2x4 failure\n");
                state = runtime_->setKernel(&winograd_multi_kernel_split_, "Fast3x3_2_optimized_2x4_splite", precision_);
                CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2_optimized_2x4_splite failure\n");

                if (activation_info_.isEnabled()) {
                    if (activation_info_.activation() == ActivationInfo::ActivationType::RELU) {
                        state = runtime_->setKernel(
                            &winograd_multi_kernel_merge_, "RELUFast3x3_2_optimized_2x4_merge", precision_);
                        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2_optimized_2x4_merge failure\n");
                    } else if (activation_info_.activation() == ActivationInfo::ActivationType::RELU6 &&
                               precision_ == PrecisionType::FP16) {
                        state = runtime_->setKernel(
                            &winograd_multi_kernel_merge_, "RELU6Fast3x3_2_optimized_2x4_merge", precision_);
                        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2_optimized_2x4_merge failure\n");
                    } else {
                        state =
                            runtime_->setKernel(&winograd_multi_kernel_merge_, "Fast3x3_2_optimized_2x4_merge", precision_);
                        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2_optimized_2x4_merge failure\n");
                    }
                } else {
                    state = runtime_->setKernel(&winograd_multi_kernel_merge_, "Fast3x3_2_optimized_2x4_merge", precision_);
                    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2_optimized_2x4_merge failure\n");
                }
            }
        } else {
            bifrost_speedup_ = true;
            top_channel_need_aligned_ = true;
            coalescing_feature_height_ = WINO_COALESCING_SIZE_MAKALU;
            compute_output_number_ = WINO_TILE_SIZE_4;
            splite_number_ = WINO_SPLIT_SIZE_4;

            Status state = runtime_->setKernel(&winograd_convert_kernel_bifrost_, "Fast3x3_1_4x4", precision_);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_1_4x4 failure\n");

            state = runtime_->setKernel(&winograd_multi_kernel_split_, "Fast3x3_2_optimized_4x4_splite", precision_);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2_optimized_4x4_splite failure\n");

            if (activation_info_.isEnabled()) {
                if (activation_info_.activation() == ActivationInfo::ActivationType::RELU) {
                    state = runtime_->setKernel(
                        &winograd_multi_kernel_merge_, "RELUFast3x3_2_optimized_4x4_merge_unaligned", precision_);
                    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel RELUFast3x3_2_optimized_4x4_merge failure\n");
                } else if (activation_info_.activation() == ActivationInfo::ActivationType::RELU6 &&
                           precision_ == PrecisionType::FP16) {
                    state = runtime_->setKernel(
                        &winograd_multi_kernel_merge_, "RELU6Fast3x3_2_optimized_4x4_merge_unaligned", precision_);
                    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel RELU6Fast3x3_2_optimized_4x4_merge failure\n");
                } else {
                    state = runtime_->setKernel(
                        &winograd_multi_kernel_merge_, "Fast3x3_2_optimized_4x4_merge_unaligned", precision_);
                    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2_optimized_4x4_merge failure\n");
                }
            } else {
                state = runtime_->setKernel(
                    &winograd_multi_kernel_merge_, "Fast3x3_2_optimized_4x4_merge_unaligned", precision_);
                CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2_optimized_4x4_merge failure\n");
            }
        }
    } else {
        bifrost_speedup_ = false;
    }

    if (bifrost_speedup_ == false) {
        Status state = runtime_->setKernel(&winograd_convert_kernel_, "Fast3x3_1", precision_);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_1 failure\n");
        if (branch_number_ == WINO_TILE_1X2) {
            if (activation_info_.isEnabled()) {
                if (activation_info_.activation() == ActivationInfo::ActivationType::RELU) {
                    state = runtime_->setKernel(&winograd_multi_kernel_optimized_, "RELUFast3x3_2_optimized", precision_);
                    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2_optimized failure\n");
                } else if (activation_info_.activation() == ActivationInfo::ActivationType::RELU6 &&
                           precision_ == PrecisionType::FP16) {
                    state = runtime_->setKernel(&winograd_multi_kernel_optimized_, "RELU6Fast3x3_2_optimized", precision_);
                    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2_optimized failure\n");
                } else {
                    state = runtime_->setKernel(&winograd_multi_kernel_optimized_, "Fast3x3_2_optimized", precision_);
                    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2_optimized failure\n");
                }
            } else {
                state = runtime_->setKernel(&winograd_multi_kernel_optimized_, "Fast3x3_2_optimized", precision_);
                CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2_optimized failure\n");
            }
        } else {
            if (activation_info_.isEnabled()) {
                if (activation_info_.activation() == ActivationInfo::ActivationType::RELU) {
                    state = runtime_->setKernel(&winograd_multi_kernel_, "RELUFast3x3_2", precision_);
                    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2 failure\n");
                } else if (activation_info_.activation() == ActivationInfo::ActivationType::RELU6 &&
                           precision_ == PrecisionType::FP16) {
                    state = runtime_->setKernel(&winograd_multi_kernel_, "RELU6Fast3x3_2", precision_);
                    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2 failure\n");
                } else {
                    state = runtime_->setKernel(&winograd_multi_kernel_, "Fast3x3_2", precision_);
                    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2 failure\n");
                }
            } else {
                state = runtime_->setKernel(&winograd_multi_kernel_, "Fast3x3_2", precision_);
                CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel Fast3x3_2 failure\n");
            }
        }
    }

    // initial weight
    int size_convert = batch * input_channel * alignTo(wgrad_tile_height_ * wgrad_tile_width_, coalescing_feature_height_) *
                       wgrad_tile_size_ * wgrad_tile_size_;
    Dim4 convert_buffer_dim = {static_cast<uint32_t>(size_convert), 1, 1, 1};

    if (weights_as_input_ == false) {
        convert_buffer_ = std::make_shared<CLTensor>(
            runtime_, precision_, DataType::FLOAT, convert_buffer_dim, DataOrder::NCHW, 1.0, 0, BufferType::INTRA_SHARED);
    } else {
        convert_buffer_ = std::make_shared<CLTensor>(
            runtime_, precision_, DataType::FLOAT, convert_buffer_dim, DataOrder::NCHW, 1.0, 0, BufferType::DEDICATED);
    }

    if (top_channel_need_aligned_ == true) {
        int aligned_weight_count = alignTo(output_channel, 8) * group_input_channel * wgrad_tile_size_ * wgrad_tile_size_;
        Dim4 aligned_weight_buffer_dim = {static_cast<uint32_t>(aligned_weight_count), 1, 1, 1};
        weight_buffer_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, aligned_weight_buffer_dim);

        cl_mem model_weight_buffer =
            (androidNN_ || !isNCHW_) ? weight_nchw_->getDataPtr() : conv_descriptor_.filter_->getDataPtr();
        Dim4 dim_model_weight = (androidNN_ || !isNCHW_) ? weight_nchw_->getDim() : conv_descriptor_.filter_->getDim();
        int size_model_weight = dim_model_weight.n * dim_model_weight.c * dim_model_weight.h * dim_model_weight.w;
        uint32_t aligned_size_model_weight =
            alignTo(dim_model_weight.n, 8) * dim_model_weight.c * dim_model_weight.h * dim_model_weight.w;
        Dim4 dim_model_aligned_weight = {aligned_size_model_weight, 1, 1, 1};
        aligned_model_weight_ =
            std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, dim_model_aligned_weight);
        std::shared_ptr<struct _cl_kernel> copy_buffer_kernel;
        Status state = runtime_->setKernel(&copy_buffer_kernel, "copy_buffer", precision_);
        size_t global = size_model_weight;
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel copy_buffer failure\n");
        state =
            runtime_->setKernelArg(copy_buffer_kernel.get(), model_weight_buffer, aligned_model_weight_->getDataPtr(), 0, 0);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");
        state = runtime_->enqueueKernel(copy_buffer_kernel.get(), (cl_uint)1, &global, NULL);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");

        state = runtime_->setKernel(&winograd_convert_weight_, "wino_convert_weight", precision_);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel wino_convert_aligned_weight failure\n");
        auto weight_buffer_tmp = weight_buffer_->getDataPtr();
        auto aligned_model_weight_tmp = aligned_model_weight_->getDataPtr();
        state = convertWeight(aligned_model_weight_tmp,
                              weight_buffer_tmp,
                              aligned_weight_count,
                              compute_output_number_,
                              alignTo(output_channel, 8),
                              group_input_channel,
                              MAKALU_SPEED_UP,
                              splite_number_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "convert aligned weight failure\n");
    } else {
        Dim4 weight_buffer_dim = {static_cast<uint32_t>(weight_count), 1, 1, 1};
        weight_buffer_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, weight_buffer_dim);
        cl_mem model_weight_buffer =
            (androidNN_ || !isNCHW_) ? weight_nchw_->getDataPtr() : conv_descriptor_.filter_->getDataPtr();
        cl_mem converted_weight_buffer = weight_buffer_->getDataPtr();

        if (speed_up_ == MAKALU_SPEED_UP) {
            Status state =
                runtime_->setKernel(&winograd_convert_weight_makalu_opt_, "wino_convert_weight_makalu_opt", precision_);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel wino_convert_weight_makalu_opt failure!\n");
            size_t local_convert = 24;
            size_t global_convert = alignTo(output_channel * input_channel, local_convert);
            state = runtime_->setKernelArg(winograd_convert_weight_makalu_opt_.get(),
                                           model_weight_buffer,
                                           converted_weight_buffer,
                                           group_input_channel,
                                           output_channel,
                                           compute_output_number_,
                                           splite_number_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state,
                                      "setKernelArg winograd_convert_weight_makalu_opt_ failure!\n");
            state = runtime_->enqueueKernel(
                winograd_convert_weight_makalu_opt_.get(), (cl_uint)1, &global_convert, &local_convert);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state,
                                      "enqueue kernel winograd_convert_weight_makalu_opt_ failure!\n");
        } else {
            Dim4 dim_model_weight = (androidNN_ || !isNCHW_) ? weight_nchw_->getDim() : conv_descriptor_.filter_->getDim();
            int size_model_weight = dim_model_weight.n * dim_model_weight.c * dim_model_weight.h * dim_model_weight.w;

            Status state = runtime_->setKernel(&winograd_convert_weight_, "wino_convert_weight", precision_);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel wino_convert_weight failure\n");

            state = convertWeight(model_weight_buffer,
                                  converted_weight_buffer,
                                  weight_count,
                                  compute_output_number_,
                                  output_channel,
                                  group_input_channel,
                                  speed_up_,
                                  splite_number_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "convert weight failure\n");
        }
    }

    // split buffer for bifrost
    if (bifrost_speedup_) {
        Dim4 split_buffer_dim = {0, 0, 0, 0};

        if (branch_number_ == WINO_TILE_4X4) {
            int align_output_channel = output_channel / compute_output_number_ * compute_output_number_;
            int align_HW =
                ceil(static_cast<double>(wgrad_tile_height_ * wgrad_tile_width_) / WINO_TILE_SIZE_4) * WINO_TILE_SIZE_4;
            split_buffer_dim = {batch * align_output_channel * align_HW * splite_number_ * 2, 1, 1, 1};
        } else if (branch_number_ == WINO_TILE_2X8 || branch_number_ == WINO_TILE_2X4) {
            int align_output_channel = output_channel / compute_output_number_ * compute_output_number_;
            int align_HW =
                ceil(static_cast<double>(wgrad_tile_height_ * wgrad_tile_width_) / WINO_TILE_SIZE_2) * WINO_TILE_SIZE_2;

            split_buffer_dim = {batch * align_output_channel * align_HW * splite_number_ * 4, 1, 1, 1};
        } else if (branch_number_ == WINO_TILE_1X8) {
            split_buffer_dim = {batch * output_channel / compute_output_number_ * wgrad_tile_height_ * wgrad_tile_width_ *
                                    compute_output_number_ * splite_number_ * 4,
                                1,
                                1,
                                1};
        } else if (top_channel_need_aligned_ == true) {
            int align_output_channel = alignTo(output_channel, 4);
            int align_HW =
                ceil(static_cast<double>(wgrad_tile_height_ * wgrad_tile_width_) / WINO_TILE_SIZE_4) * WINO_TILE_SIZE_4;

            split_buffer_dim = {batch * align_output_channel * align_HW * splite_number_ * 2, 1, 1, 1};
        }

        if (weights_as_input_ == false) {
            // TODO(zhaonan.qin): Need to check BufferType::INTRA_SHARED
            split_buffer_ = std::make_shared<CLTensor>(
                runtime_, precision_, DataType::FLOAT, split_buffer_dim, DataOrder::NCHW, 1.0, 0, BufferType::INTRA_SHARED);
        } else {
            split_buffer_ = std::make_shared<CLTensor>(
                runtime_, precision_, DataType::FLOAT, split_buffer_dim, DataOrder::NCHW, 1.0, 0, BufferType::DEDICATED);
        }
    }
    return Status::SUCCESS;
}

Status CLWINOConvolution::execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) {
    ENN_DBG_PRINT("CLWINOConvolution::execute() is called");
    if (weights_as_input_) {
        Status state = Status::FAILURE;
        if (androidNN_ || !isNCHW_) {
            state = conv_descriptor_.filter_->convertToNCHW(weight_nchw_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "convertToNCHW failure\n");
        }
        state = initKernelWeight();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "initKernelWeight failure\n");
    }
    auto input_tensor = std::static_pointer_cast<CLTensor>(input);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output);
    // wino
    return winoRun(input_tensor, output_tensor);
}

Status CLWINOConvolution::convertWeightCPU(float *weight,
                                           float *convertedWeight,
                                           const uint32_t &weightCount,
                                           const uint32_t &computedTopNumber,
                                           const uint32_t &topChannel,
                                           const uint32_t &bottomChannel,
                                           const int speedUp,
                                           const uint32_t &spliteNumber) {
    float G[12] = {1, 0, 0, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0, 0, 1};
    float GT[12] = {1, 0.5, 0.5, 0, 0, 0.5, -0.5, 0, 0, 0.5, 0.5, 1};
    std::unique_ptr<float[]> weightGPtr(new float[weightCount]);
    cl_float *weightG = weightGPtr.get();
    CHECK_EXPR_RETURN_FAILURE(weightG, "convertWeightCPU weightG buffer creation failed!");
    memset(weightG, 0, sizeof(float) * weightCount);
    std::unique_ptr<float[]> tempPtr(new float[computedTopNumber]);
    float *temp = tempPtr.get();
    CHECK_EXPR_RETURN_FAILURE(temp, "convertWeightCPU temp buffer creation failed!");
    memset(temp, 0, sizeof(float) * computedTopNumber);
    std::unique_ptr<int[]> channelIndexPtr(new int[computedTopNumber]);
    int *channelIndex = channelIndexPtr.get();
    CHECK_EXPR_RETURN_FAILURE(channelIndex, "convertWeightCPU channelIndex buffer creation failed!");
    memset(channelIndex, 0, sizeof(int) * computedTopNumber);
    for (int tc = 0; tc < topChannel / computedTopNumber; tc++) {
        for (int bc = 0; bc < bottomChannel; bc++) {
            for (int i = 0; i < computedTopNumber; i++) {
                channelIndex[i] = (tc * computedTopNumber + i) * bottomChannel + bc;
            }
            for (int b = 0; b < 4; b++) {
                for (int c = 0; c < 3; c++) {
                    memset(temp, 0, sizeof(float) * computedTopNumber);
                    for (int d = 0; d < 3; d++) {
                        for (int i = 0; i < computedTopNumber; i++) {
                            temp[i] += G[b * 3 + d] * weight[channelIndex[i] * 3 * 3 + 3 * d + c];
                        }
                    }
                    for (int i = 0; i < computedTopNumber; i++) {
                        weightG[channelIndex[i] * 4 * 3 + b * 3 + c] = temp[i];
                    }
                }
            }
            for (int b = 0; b < 4; b++) {
                for (int c = 0; c < 4; c++) {
                    memset(temp, 0, sizeof(float) * computedTopNumber);
                    for (int d = 0; d < 3; d++) {
                        for (int i = 0; i < computedTopNumber; i++) {
                            temp[i] += weightG[channelIndex[i] * 4 * 3 + b * 3 + d] * GT[4 * d + c];
                        }
                    }
                    for (int i = 0; i < computedTopNumber; i++) {
                        if (speedUp == BIFROST_SPEED_UP) {
                            int blockSize = 4 * 4 / spliteNumber;
                            convertedWeight[tc * bottomChannel * computedTopNumber * 4 * 4 +
                                            bc * computedTopNumber * blockSize +
                                            (b * 4 + c) / blockSize * computedTopNumber * blockSize * bottomChannel +
                                            (b * 4 + c) % blockSize * computedTopNumber + i] = temp[i];
                        } else if (speedUp == MAKALU_SPEED_UP) {
                            int blockSize = 4 * 4 / spliteNumber;
                            convertedWeight[tc / 2 * bottomChannel * computedTopNumber * 4 * 4 * 2 +
                                            bc * computedTopNumber * blockSize * 2 +
                                            (b * 4 + c) / blockSize * computedTopNumber * blockSize * bottomChannel * 2 +
                                            (b * 4 + c) % blockSize * computedTopNumber +
                                            tc % 2 * computedTopNumber * blockSize + i] = temp[i];
                        } else if (speedUp == MAKALU_SPEED_UP_1X2) {
                            convertedWeight[tc / 2 * bottomChannel * computedTopNumber * 4 * 4 * 2 +
                                            bc * computedTopNumber * 4 * 4 * 2 + tc % 2 * 4 * 4 + b * 4 + c +
                                            i * 4 * 4 * 2] = temp[i];
                        } else {
                            convertedWeight[tc * bottomChannel * computedTopNumber * 4 * 4 + bc * computedTopNumber * 4 * 4 +
                                            b * 4 + c + i * 16] = temp[i];
                        }
                    }
                }
            }
        }
    }
    weightGPtr = nullptr;
    tempPtr = nullptr;
    channelIndexPtr = nullptr;

    return Status::SUCCESS;
}

Status CLWINOConvolution::convertWeight(cl_mem &weight,
                                        cl_mem &convert_weight,
                                        const uint32_t &weight_count,
                                        const uint32_t &cpt_output_number,
                                        const uint32_t &output_channel,
                                        const uint32_t &input_channel,
                                        const int speed_up,
                                        const uint32_t &splite_number) {
    ENN_DBG_PRINT("CLWINOConvolution convertWeight");

    size_t local_convert[2] = {1, 1};
    size_t global_convert[2] = {output_channel / cpt_output_number, input_channel};

    Dim4 converted_filter_dim = {2 * weight_count, 1, 1, 1};
    converted_filter_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, converted_filter_dim);
    auto weightG_buffer = converted_filter_->getDataPtr();

    Status state = runtime_->setKernelArg(winograd_convert_weight_.get(),
                                          weight,
                                          convert_weight,
                                          weightG_buffer,
                                          output_channel,
                                          cpt_output_number,
                                          speed_up,
                                          splite_number);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");
    state = runtime_->enqueueKernel(winograd_convert_weight_.get(), (cl_uint)2, global_convert, local_convert);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "enqueue kernel failure\n");

    return Status::SUCCESS;
}

Status CLWINOConvolution::winoRun(const std::shared_ptr<CLTensor> input, std::shared_ptr<CLTensor> output) {
    ENN_DBG_PRINT("CLWINOConvolution winoRun");
    auto input_data = input->getDataPtr();
    auto output_data = output->getDataPtr();
    auto input_dim = input->getDim();
    auto output_dim = output->getDim();
    int group = conv_descriptor_.group_;
    cl_mem bias_data = conv_descriptor_.bias_->getDataPtr();

    int pad_height = conv_descriptor_.pad_top_;   // conv_descriptor_.pad_bottom_ ==conv_descriptor_.pad_top_
    int pad_width = conv_descriptor_.pad_right_;  // conv_descriptor_.pad_left_==conv_descriptor_.pad_right_
    int out_channel = output_dim.c;
    // convert
    size_t global_convert[3] = {0, 0, 0};
    size_t local_convert[3] = {0, 0, 0};
    int gsize0 = input_dim.n * input_dim.c / group;
    int lsize0 = 1;
    global_convert[0] = gsize0;
    int gsize1 = wgrad_tile_height_;
    int lsize1 = 1;
    global_convert[1] = alignTo(gsize1, lsize1);
    int gsize2 = wgrad_tile_width_;
    int lsize2 = 12;
    global_convert[2] = alignTo(gsize2, lsize2);
    local_convert[0] = lsize0;
    local_convert[1] = lsize1;
    local_convert[2] = lsize2;
    uint32_t group_channel = input_dim.c / group;

    if (bifrost_speedup_) {
        if (branch_number_ == WINO_TILE_2X8 || branch_number_ == WINO_TILE_2X4 || branch_number_ == WINO_TILE_4X4 ||
            branch_number_ == WINO_TILE_1X2_MAKALU || top_channel_need_aligned_ == true) {
            if (branch_number_ == WINO_TILE_4X4 || top_channel_need_aligned_ == true) {
                gsize0 = wgrad_tile_height_ * wgrad_tile_width_;
                lsize0 = 24;
                global_convert[0] = alignTo(gsize0, lsize0);

                gsize1 = input_dim.c;
                lsize1 = 2;
                global_convert[1] = alignTo(gsize1, lsize1);

                gsize2 = input_dim.n;
                lsize2 = 1;
                global_convert[2] = alignTo(gsize2, lsize2);

                local_convert[0] = lsize0;
                local_convert[1] = lsize1;
                local_convert[2] = lsize2;
            } else {
                gsize0 = input_dim.n;
                lsize0 = 1;
                global_convert[0] = gsize0;
                gsize1 = input_dim.c;
                lsize1 = 1;
                global_convert[1] = alignTo(gsize1, lsize1);
                gsize2 = wgrad_tile_height_ * wgrad_tile_width_;
                lsize2 = 32;
                global_convert[2] = alignTo(gsize2, lsize2);
                local_convert[0] = lsize0;
                local_convert[1] = lsize1;
                local_convert[2] = lsize2;
            }

            Status state = runtime_->setKernelArg(winograd_convert_kernel_bifrost_.get(),
                                                  input_data,
                                                  convert_buffer_->getDataPtr(),
                                                  pad_height,
                                                  pad_width,
                                                  input_dim.h,
                                                  input_dim.w,
                                                  group_channel,
                                                  wgrad_tile_height_,
                                                  wgrad_tile_width_,
                                                  coalescing_feature_height_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");
        } else {
            Status state = runtime_->setKernelArg(winograd_convert_kernel_bifrost_.get(),
                                                  input_data,
                                                  convert_buffer_->getDataPtr(),
                                                  pad_height,
                                                  pad_width,
                                                  input_dim.h,
                                                  input_dim.w,
                                                  group_channel,
                                                  wgrad_tile_height_,
                                                  wgrad_tile_width_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");
        }
        Status state =
            runtime_->enqueueKernel(winograd_convert_kernel_bifrost_.get(), (cl_uint)3, global_convert, local_convert);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
    } else {
        for (cl_uint i = 0; i < (cl_uint)group; i++) {
            Status state = runtime_->setKernelArg(winograd_convert_kernel_.get(),
                                                  input_data,
                                                  convert_buffer_->getDataPtr(),
                                                  pad_height,
                                                  pad_width,
                                                  input_dim.h,
                                                  input_dim.w,
                                                  i,
                                                  group_channel,
                                                  group,
                                                  wgrad_tile_height_,
                                                  wgrad_tile_width_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");
            state = runtime_->enqueueKernel(winograd_convert_kernel_.get(), (cl_uint)3, global_convert, local_convert);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
        }
    }

    // multiply
    size_t global_multi[3] = {0, 0, 0};
    size_t local_multi[3] = {0, 0, 0};
    cl_uint input_number_per_out2x2 = wgrad_tile_size_ * wgrad_tile_size_ * input_dim.c / group;

    if (bifrost_speedup_) {
        if (branch_number_ == WINO_TILE_1X2_MAKALU) {
            // 1x2 for makalu: coalscing memory and shared data between 2D local, no splitting and
            // merging
            gsize0 = output_dim.n;
            gsize1 = out_channel / compute_output_number_;
            gsize2 = wgrad_tile_height_ * wgrad_tile_width_;

            lsize0 = 1;
            lsize1 = 2;   // the top channels can be divided by 4 for makalu 2x2 by default
            lsize2 = 12;  // 2D loocal to shared data between 24 threads, 3 groups of 8 threads,
                          // each with shape 2x4

            global_multi[0] = alignTo(gsize0, lsize0);
            global_multi[1] = alignTo(gsize1, lsize1);
            global_multi[2] = alignTo(gsize2, lsize2);

            local_multi[0] = lsize0;
            local_multi[1] = lsize1;
            local_multi[2] = lsize2;

            Status state = runtime_->setKernelArg(winograd_multi_kernel_split_.get(),
                                                  convert_buffer_->getDataPtr(),
                                                  weight_buffer_->getDataPtr(),
                                                  bias_data,
                                                  output_data,
                                                  input_number_per_out2x2,
                                                  output_dim.h,
                                                  output_dim.w,
                                                  out_channel,
                                                  wgrad_tile_height_,
                                                  wgrad_tile_width_,
                                                  coalescing_feature_height_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");

            runtime_->enqueueKernel(winograd_multi_kernel_split_.get(), (cl_uint)3, global_multi, local_multi);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");

        } else if (top_channel_need_aligned_ == true) {
            gsize0 = output_dim.n * alignTo(out_channel, 4) / 4;
            lsize0 = 1;
            global_multi[0] = alignTo(gsize0, lsize0);
            gsize1 = splite_number_;
            lsize1 = 1;
            global_multi[1] = alignTo(gsize1, lsize1);
            gsize2 = ceil(wgrad_tile_height_ * wgrad_tile_width_ / 4.0);
            if (gsize2 > 12) {
                lsize2 = 24;
            } else {
                lsize2 = 8;
            }
            global_multi[2] = alignTo(gsize2, lsize2);
            local_multi[0] = lsize0;
            local_multi[1] = lsize1;
            local_multi[2] = lsize2;

            uint32_t aligned_outc = alignTo(out_channel, 4);
            Status state = runtime_->setKernelArg(winograd_multi_kernel_split_.get(),
                                                  convert_buffer_->getDataPtr(),
                                                  weight_buffer_->getDataPtr(),
                                                  bias_data,
                                                  split_buffer_->getDataPtr(),
                                                  input_number_per_out2x2,
                                                  output_dim.h,
                                                  output_dim.w,
                                                  aligned_outc,
                                                  wgrad_tile_height_,
                                                  wgrad_tile_width_,
                                                  coalescing_feature_height_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");
            state = runtime_->enqueueKernel(winograd_multi_kernel_split_.get(), (cl_uint)3, global_multi, local_multi);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");

            gsize0 = output_dim.n;
            gsize1 = ceil(out_channel / 4.0);
            gsize2 = output_dim.h * output_dim.w;
            lsize0 = 1;
            lsize1 = 1;
            lsize2 = 24;
            global_multi[0] = alignTo(gsize0, lsize0);
            global_multi[1] = alignTo(gsize1, lsize1);
            global_multi[2] = alignTo(gsize2, lsize2);
            local_multi[0] = lsize0;
            local_multi[1] = lsize1;
            local_multi[2] = lsize2;

            state = runtime_->setKernelArg(winograd_multi_kernel_merge_.get(),
                                           split_buffer_->getDataPtr(),
                                           weight_buffer_->getDataPtr(),
                                           bias_data,
                                           output_data,
                                           input_number_per_out2x2,
                                           output_dim.h,
                                           output_dim.w,
                                           out_channel,
                                           wgrad_tile_height_,
                                           wgrad_tile_width_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");
            state = runtime_->enqueueKernel(winograd_multi_kernel_merge_.get(), (cl_uint)3, global_multi, local_multi);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");

        } else {
            gsize0 = output_dim.n * out_channel / compute_output_number_;
            if (speed_up_ == MAKALU_SPEED_UP) {
                lsize0 = 2;
            } else {
                lsize0 = 1;
            }
            global_multi[0] = alignTo(gsize0, lsize0);
            gsize1 = splite_number_;
            lsize1 = 1;
            global_multi[1] = alignTo(gsize1, lsize1);
            gsize2 = wgrad_tile_height_ * wgrad_tile_width_;
            if (speed_up_ == MAKALU_SPEED_UP) {
                gsize2 = ceil(wgrad_tile_height_ * wgrad_tile_width_ / 4.0);
            } else if (branch_number_ == WINO_TILE_2X8 || branch_number_ == WINO_TILE_2X4) {
                gsize2 = ceil(wgrad_tile_height_ * wgrad_tile_width_ / 2.0);
            }
            if (gsize2 > 12) {
                lsize2 = 24;
            } else {
                lsize2 = 8;
            }
            global_multi[2] = alignTo(gsize2, lsize2);
            local_multi[0] = lsize0;
            local_multi[1] = lsize1;
            local_multi[2] = lsize2;
            if (branch_number_ == WINO_TILE_2X8 || branch_number_ == WINO_TILE_2X4 || branch_number_ == WINO_TILE_4X4) {
                Status state = runtime_->setKernelArg(winograd_multi_kernel_split_.get(),
                                                      convert_buffer_->getDataPtr(),
                                                      weight_buffer_->getDataPtr(),
                                                      bias_data,
                                                      split_buffer_->getDataPtr(),
                                                      input_number_per_out2x2,
                                                      output_dim.h,
                                                      output_dim.w,
                                                      out_channel,
                                                      wgrad_tile_height_,
                                                      wgrad_tile_width_,
                                                      coalescing_feature_height_);
                CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");
            } else {
                Status state = runtime_->setKernelArg(winograd_multi_kernel_split_.get(),
                                                      convert_buffer_->getDataPtr(),
                                                      weight_buffer_->getDataPtr(),
                                                      bias_data,
                                                      split_buffer_->getDataPtr(),
                                                      input_number_per_out2x2,
                                                      output_dim.h,
                                                      output_dim.w,
                                                      out_channel,
                                                      wgrad_tile_height_,
                                                      wgrad_tile_width_);
                CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");
            }
            Status state =
                runtime_->enqueueKernel(winograd_multi_kernel_split_.get(), (cl_uint)3, global_multi, local_multi);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");

            if (speed_up_ == MAKALU_SPEED_UP) {
                gsize0 = output_dim.n;
                gsize1 = out_channel / compute_output_number_;
                lsize1 = 1;
                lsize2 = 24;
            } else if (branch_number_ == WINO_TILE_1X8) {
                gsize0 = output_dim.n * out_channel / compute_output_number_;
                gsize1 = 4;
                lsize1 = 1;
                lsize2 = 96;
            } else if (branch_number_ == WINO_TILE_2X8) {
                gsize0 = output_dim.n * out_channel / compute_output_number_;
                gsize1 = 4;
                lsize1 = 4;
                lsize2 = 24;
            } else {
                gsize0 = output_dim.n;
                gsize1 = out_channel / compute_output_number_;
                lsize1 = 1;
                lsize2 = 24;
            }
            lsize0 = 1;
            global_multi[0] = alignTo(gsize0, lsize0);
            global_multi[1] = alignTo(gsize1, lsize1);
            gsize2 = wgrad_tile_height_ * wgrad_tile_width_;

            if (speed_up_ == MAKALU_SPEED_UP) {
                gsize2 = output_dim.h * output_dim.w;
            } else if (branch_number_ == WINO_TILE_2X8 || branch_number_ == WINO_TILE_2X4) {
                gsize2 = ceil(wgrad_tile_height_ * wgrad_tile_width_ / 2.0);
            }
            global_multi[2] = alignTo(gsize2, lsize2);
            local_multi[0] = lsize0;
            local_multi[1] = lsize1;
            local_multi[2] = lsize2;
            state = runtime_->setKernelArg(winograd_multi_kernel_merge_.get(),
                                           split_buffer_->getDataPtr(),
                                           weight_buffer_->getDataPtr(),
                                           bias_data,
                                           output_data,
                                           input_number_per_out2x2,
                                           output_dim.h,
                                           output_dim.w,
                                           out_channel,
                                           wgrad_tile_height_,
                                           wgrad_tile_width_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");
            state = runtime_->enqueueKernel(winograd_multi_kernel_merge_.get(), (cl_uint)3, global_multi, local_multi);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
        }
    } else if (branch_number_ == WINO_TILE_1X2) {
        //  this is for the non-bifrost architecture
        //  since winograd needs a lot of register resources so that the performance is fragile for
        //  local variables each thread processes data for two output channels
        cl_uint output_group_channel = out_channel / group;
        gsize0 = output_dim.n * output_group_channel / compute_output_number_;
        if (output_group_channel % (compute_output_number_ * 2) == 0) {
            lsize0 = 2;
        } else {
            lsize0 = 1;
        }
        global_multi[0] = gsize0;
        gsize1 = wgrad_tile_height_;
        lsize1 = 2;
        global_multi[1] = alignTo(gsize1, lsize1);
        gsize2 = wgrad_tile_width_;
        lsize2 = 24;
        global_multi[2] = alignTo(gsize2, lsize2);
        local_multi[0] = lsize0;
        local_multi[1] = lsize1;
        local_multi[2] = lsize2;
        for (cl_uint i = 0; i < (cl_uint)group; i++) {
            Status state = runtime_->setKernelArg(winograd_multi_kernel_optimized_.get(),
                                                  convert_buffer_->getDataPtr(),
                                                  weight_buffer_->getDataPtr(),
                                                  bias_data,
                                                  output_data,
                                                  input_number_per_out2x2,
                                                  output_dim.h,
                                                  output_dim.w,
                                                  output_group_channel,
                                                  wgrad_tile_height_,
                                                  wgrad_tile_width_,
                                                  group,
                                                  i);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");
            state = runtime_->enqueueKernel(winograd_multi_kernel_optimized_.get(), (cl_uint)3, global_multi, local_multi);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
        }
    } else {
        cl_uint output_group_channel = out_channel / group;
        global_multi[0] = output_dim.n * output_group_channel;
        global_multi[1] = wgrad_tile_height_;
        global_multi[2] = wgrad_tile_width_;
        local_multi[2] = findMaxFactor(wgrad_tile_width_, MAX_WORK_GROUP_SIZE);
        local_multi[0] = findMaxFactor(global_multi[0], MAX_WORK_GROUP_SIZE / local_multi[2]);
        local_multi[1] = findMaxFactor(wgrad_tile_height_, MAX_WORK_GROUP_SIZE / local_multi[2] / local_multi[0]);

        for (cl_uint i = 0; i < (cl_uint)group; i++) {
            Status state = runtime_->setKernelArg(winograd_multi_kernel_.get(),
                                                  convert_buffer_->getDataPtr(),
                                                  weight_buffer_->getDataPtr(),
                                                  bias_data,
                                                  output_data,
                                                  input_number_per_out2x2,
                                                  output_dim.h,
                                                  output_dim.w,
                                                  i,
                                                  output_group_channel,
                                                  group);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");
            state = runtime_->enqueueKernel(winograd_multi_kernel_.get(), (cl_uint)3, global_multi, local_multi);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
        }
    }

    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
