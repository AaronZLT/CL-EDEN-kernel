#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLDilationConvolution.hpp"

namespace enn {
namespace ud {
namespace gpu {

CLDilationConvolution::CLDilationConvolution(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) {
    ENN_DBG_PRINT("CLDilationConvolution is created");
    runtime_ = runtime;
    precision_ = precision;
    dilation_conv_kernel_ = nullptr;
    computed_top_channel_numbers_ = DILATION_TOP_CHANNEL_4;
    computed_top_width_numbers_ = DILATION_TOP_WIDTH_8;
    computed_top_height_numbers_ = 1;
    input_dim_ = {0, 0, 0, 0};
    weight_dim_ = {0, 0, 0, 0};
    padding_ = {0, 0, 0, 0};
    stride_ = {0, 0};
    dilation_ = {0, 0};
    weights_as_input_ = false;
    androidNN_ = false;
    isNCHW_ = true;
}

Status CLDilationConvolution::initialize(const Dim4 &input_dim,
                                         const Dim4 &output_dim,
                                         const Dim4 &weight_dim,
                                         const Pad4 &padding,
                                         const Dim2 &stride,
                                         const Dim2 &dilation,
                                         const std::shared_ptr<ITensor> weight,
                                         const std::shared_ptr<ITensor> bias,
                                         const ActivationInfo &activate_info,
                                         const bool &weights_as_input,
                                         const bool &androidNN,
                                         const bool &isNCHW) {
    ENN_DBG_PRINT("CLDilationConvolution::initialize() is called");
    weight_dim_ = weight_dim;
    activation_info_ = activate_info;
    padding_ = padding;
    stride_ = stride;
    dilation_ = dilation;
    bias_ = std::static_pointer_cast<CLTensor>(bias);
    weight_ = std::static_pointer_cast<CLTensor>(weight);
    weights_as_input_ = weights_as_input;
    androidNN_ = androidNN;
    isNCHW_ = isNCHW;

    Status state = Status::FAILURE;
    if (androidNN_ || !isNCHW_) {
        Dim4 weight_dim_nchw = {weight_dim.n, weight_dim.w, weight_dim.c, weight_dim.h};
        weight_nchw_ = std::make_shared<CLTensor>(runtime_,
                                                  precision_,
                                                  weight_->getDataType(),
                                                  weight_dim_nchw,
                                                  weight_->getDataOrder(),
                                                  weight_->getScale(),
                                                  weight_->getZeroPoint());
        weight_dim_ = weight_dim_nchw;
        if (!weights_as_input_) {
            state = weight_->convertToNCHW(weight_nchw_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "convertToNCHW failure\n");
        }
    }

    computed_top_channel_numbers_ = DILATION_TOP_CHANNEL_4;
    computed_top_width_numbers_ = DILATION_TOP_WIDTH_8;
    computed_top_height_numbers_ = 1;

    std::string kernel_name = "";
    if (activate_info.isEnabled()) {
        switch (activate_info.activation()) {
        case ActivationInfo::ActivationType::RELU: kernel_name += "RELU"; break;
        case ActivationInfo::ActivationType::RELU6: kernel_name += "RELU6"; break;
        default: ERROR_PRINT("Non supported ativation type: %d\n", activate_info.activation()); return Status::FAILURE;
        }
    }

    if (weight_dim_.h == 3 && weight_dim_.w == 3 && dilation_.h == 8 && dilation_.w == 8 && padding_.t == 8 &&
        padding_.b == 8 && padding_.l == 8 && padding_.r == 8) {
        state = runtime_->setKernel(&dilation_conv_kernel_, kernel_name + "dilation_conv_k3d8p8_4x8", precision_);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel dilation_conv_k3d8_4x8 failed!\n");
    } else {
        ERROR_PRINT("Non supported dilation convolution config: kernel(%d, %d), dilation(%d, %d)\n",
                    weight_dim_.h,
                    weight_dim_.w,
                    dilation_.h,
                    dilation_.w);
        return Status::FAILURE;
    }

    if (!weights_as_input) {
        weightConvert();
    }

    return Status::SUCCESS;
}

Status CLDilationConvolution::weightConvert() {
    Status state = Status::FAILURE;
    int aligned_wight_w = weight_dim_.w * computed_top_channel_numbers_;
    int aligned_weight_h = weight_dim_.h;
    int aligned_weight_c = weight_dim_.c;
    int aligned_weight_n = ceil(static_cast<double>(weight_dim_.n) / computed_top_channel_numbers_);
    Dim4 alinged_weight_dim = {static_cast<uint32_t>(aligned_weight_n),
                               static_cast<uint32_t>(aligned_weight_c),
                               static_cast<uint32_t>(aligned_weight_h),
                               static_cast<uint32_t>(aligned_wight_w)};
    aligned_weight_ = std::make_shared<CLTensor>(runtime_, precision_, weight_->getDataType(), alinged_weight_dim);

    std::shared_ptr<struct _cl_kernel> align_weight_kernel;
    state = runtime_->setKernel(&align_weight_kernel, "align_weight_direct", precision_);
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernel align_weight_direct failed\n");

    auto weight_data = (androidNN_ || !isNCHW_) ? weight_nchw_->getDataPtr() : weight_->getDataPtr();

    state = runtime_->setKernelArg(align_weight_kernel.get(),
                                   weight_data,
                                   aligned_weight_->getDataPtr(),
                                   weight_dim_.n,
                                   weight_dim_.c,
                                   weight_dim_.h,
                                   weight_dim_.w,
                                   aligned_weight_n,
                                   aligned_weight_c,
                                   aligned_weight_h,
                                   aligned_wight_w,
                                   computed_top_channel_numbers_);
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "setKernelArg align_weight_direct failed\n");

    size_t local[3] = {1, 1, 1};
    size_t global[3] = {static_cast<size_t>(aligned_wight_w),
                        static_cast<size_t>(aligned_weight_h),
                        static_cast<size_t>(aligned_weight_c * aligned_weight_n)};
    state = runtime_->enqueueKernel(align_weight_kernel.get(), (cl_uint)3, global, local);
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "enqueue align_weight_direct failed\n");
    return Status::SUCCESS;
}

Status CLDilationConvolution::execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) {
    ENN_DBG_PRINT("CLDilationConvolution::execute is called\n");
    Status status = Status::SUCCESS;
    if (weights_as_input_) {
        if (androidNN_ || !isNCHW_) {
            status = weight_->convertToNCHW(weight_nchw_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "convertToNCHW failure\n");
        }
        weightConvert();
    }

    auto input_ = std::static_pointer_cast<CLTensor>(input);
    auto output_ = std::static_pointer_cast<CLTensor>(output);
    Dim4 output_dim = output_->getDim();
    Dim4 input_dim = input_->getDim();

    size_t local[3] = {16, 1, 1};
    size_t global[3] = {0, 0, 0};

    if (output_dim.w / computed_top_width_numbers_ < 16) {
        local[0] = 4;
        local[1] = 4;
    }
    global[0] = alignTo(ceil(static_cast<double>(output_dim.w) / computed_top_width_numbers_), local[0]);
    global[1] = alignTo(ceil(static_cast<double>(output_dim.h) / computed_top_height_numbers_), local[1]);
    global[2] = alignTo(ceil(static_cast<double>(output_dim.c) / computed_top_channel_numbers_) * output_dim.n, local[2]);

    status = runtime_->setKernelArg(dilation_conv_kernel_.get(),
                                    input_->getDataPtr(),
                                    aligned_weight_->getDataPtr(),
                                    bias_->getDataPtr(),
                                    output_->getDataPtr(),
                                    padding_.t,
                                    padding_.l,
                                    output_dim.n,
                                    output_dim.c,
                                    output_dim.h,
                                    output_dim.w,
                                    input_dim.n,
                                    input_dim.c,
                                    input_dim.h,
                                    input_dim.w);

    CHECK_EXPR_NO_RETURN(Status::SUCCESS == status, "setKernelArg dilation_conv_kernel_ failed\n");

    status = runtime_->enqueueKernel(dilation_conv_kernel_.get(), (cl_uint)3, global, local);
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == status, "enqueue dilation_conv_kernel_ failed\n");

    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
