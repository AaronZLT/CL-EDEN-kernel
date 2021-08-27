#include "CLConcat.hpp"

namespace enn {
namespace ud {
namespace gpu {
namespace {
constexpr int OUTPUT_INDEX = 0;

class CLConcatTextureImpl {
  public:
    CLConcatTextureImpl(CLConcat *base) : base_(base) {}
    Status initialize() {
        const auto output_dim = base_->output_tensor_->getDim();
        int grid[3] = {(int)(output_dim.w * output_dim.n), (int)output_dim.h, (int)1};
        int best_work_group[3] = {0, 0, 0};
        GetBestWorkGroup(grid, best_work_group, (int)base_->runtime_->getMaxWorkGroupSize()[2]);

        local_[0] = static_cast<size_t>(best_work_group[0]);
        local_[1] = static_cast<size_t>(best_work_group[1]);
        local_[2] = static_cast<size_t>(best_work_group[2]);

        global_[0] = static_cast<size_t>(AlignByN(grid[0], local_[0]));
        global_[1] = static_cast<size_t>(AlignByN(grid[1], local_[1]));
        global_[2] = static_cast<size_t>(AlignByN(grid[2], local_[2]));

        src_size_0_depth_ = IntegralDivideRoundUp(base_->input_tensors_.at(0)->getDim().c, 4);
        src_size_1_depth_ = IntegralDivideRoundUp(base_->input_tensors_.at(1)->getDim().c, 4);
        dst_size_x_ = output_dim.w * output_dim.n;
        dst_size_y_ = output_dim.h;
        dst_size_z_ = IntegralDivideRoundUp(output_dim.c, 4);

        Status status = base_->runtime_->setKernel(&kernel_, "concat_axis1_tflite", base_->precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel concat_axis1_tflite failure\n");

        return Status::SUCCESS;
    }
    Status execute() {
        if (!(base_->parameters_->axis == 1 && base_->input_tensors_.size() == 2)) {
            ENN_ERR_PRINT("CLConcatTextureImpl execute failure, axis = %d, input_tensors.size = %zu\n",
                        base_->parameters_->axis,
                        base_->input_tensors_.size());
            return Status::FAILURE;
        }

        auto input_tensor_0 = base_->input_tensors_.at(0);
        auto input_tensor_1 = base_->input_tensors_.at(1);
        auto output_tensor = base_->output_tensor_;

        Status status;
        status = base_->runtime_->setKernelArg(kernel_.get(),
                                               input_tensor_0->getDataPtr(),
                                               input_tensor_1->getDataPtr(),
                                               output_tensor->getDataPtr(),
                                               src_size_0_depth_,
                                               src_size_1_depth_,
                                               dst_size_x_,
                                               dst_size_y_,
                                               dst_size_z_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "concat_axis1_tflite setKernelArg failure\n");

        status = base_->runtime_->enqueueKernel(kernel_.get(), (cl_uint)3, global_, local_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "concat_axis1_tflite enqueueKernel failure\n");

        return status;
    }

  private:
    size_t local_[3] = {0, 0, 0};
    size_t global_[3] = {0, 0, 0};
    int src_size_0_depth_;
    int src_size_1_depth_;
    int dst_size_x_;
    int dst_size_y_;
    int dst_size_z_;

    float CalculateResizeScale(int32_t input_size, int32_t output_size, const bool align_corners) {
        return align_corners && input_size > 1 && output_size > 1 ? static_cast<float>(input_size - 1) / (output_size - 1)
                                                                  : static_cast<float>(input_size) / output_size;
    }

    void GetBestWorkGroup(const int *grid, int *best_group_size, int max_size) {
        int wg_z = GetBiggestDividerWithPriority(grid[2], 8);
        int wg_xy_size = max_size / wg_z;
        int wg_x = std::min(IntegralDivideRoundUp(grid[0], 2), wg_xy_size);
        int wg_y = std::min(wg_xy_size / wg_x, grid[1]);
        best_group_size[0] = wg_x;
        best_group_size[1] = wg_y;
        best_group_size[2] = wg_z;
    }

  private:
    // 1. Base Operator
    CLConcat *base_;

    // 2. Operator kernels
    std::shared_ptr<struct _cl_kernel> kernel_;
};
}  // namespace

CLConcat::CLConcat(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    ENN_DBG_PRINT("CLConcat is created\n");
    input_tensors_.clear();
    output_tensor_ = nullptr;
    parameters_ = nullptr;
    kernel_ = nullptr;
    texture_impl_ = nullptr;
}

Status CLConcat::initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                            const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                            const std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLConcat::initialize() is called\n");

    for (auto input_tensor : input_tensors) {
        input_tensors_.push_back(std::static_pointer_cast<CLTensor>(input_tensor));
    }
    output_tensor_ = std::static_pointer_cast<CLTensor>(output_tensors.at(OUTPUT_INDEX));
    parameters_ = std::static_pointer_cast<ConcatParameters>(parameters);
    CHECK_EXPR_RETURN_FAILURE(nullptr != parameters_, "CLConcat must have parameters\n");

    if (parameters_->activation_info.isEnabled() &&
        parameters_->activation_info.activation() != ActivationInfo::ActivationType::NONE) {
        ENN_ERR_PRINT("CLConcat NOT Support fuse Activation Function\n");
        return Status::FAILURE;
    }

    if ((parameters_->androidNN && parameters_->isNCHW) || (parameters_->storage_type == StorageType::TEXTURE)) {
        CHECK_EXPR_RETURN_FAILURE((parameters_->axis > 0) && (parameters_->axis <= 4), "Error axis < 0 or axis > 4\n");
        switch (parameters_->axis) {
        case N_NHWC: parameters_->axis = N_NCHW; break;
        case H_NHWC: parameters_->axis = H_NCHW; break;
        case W_NHWC: parameters_->axis = W_NCHW; break;
        case C_NHWC: parameters_->axis = C_NCHW; break;
        }

        if (parameters_->androidNN && parameters_->isNCHW) {
            for (size_t i = 0; i < input_tensors_.size(); ++i) {
                CHECK_EXPR_RETURN_FAILURE(input_tensors_[0]->getNumOfDims() == 4,
                                          "Invalid input_tensors_[%d]->getNumOfDims(): %u, which should be 4.",
                                          i,
                                          input_tensors_[0]->getNumOfDims());
            }
        }
    }

    if (parameters_->androidNN) {
        NDims out_dim = input_tensors_[0]->getDims();
        for (size_t i = 1; i < input_tensors_.size(); ++i) {
            out_dim[parameters_->axis] += input_tensors_[i]->getDims()[parameters_->axis];
        }
        if (!isDimsSame(out_dim, output_tensor_->getDims())) {
            output_tensor_->reconfigureDimsAndBuffer(out_dim);
        }
    }

    Status status = Status::SUCCESS;
    if (parameters_->storage_type == StorageType::TEXTURE) {
        texture_impl_ = std::make_shared<CLConcatTextureImpl>(this);
        status = texture_impl_->initialize();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "CLConcatTextureImpl initialize failure\n");
    } else {
        const std::string kernel_prefix = precision_ == PrecisionType::INT8 ? "SIGNED" : "";
        if (parameters_->axis == 0) {
            status = runtime_->setKernel(&kernel_, kernel_prefix + "concat_axis0", precision_);
            CHECK_EXPR_RETURN_FAILURE(
                Status::SUCCESS == status, "setKernel %sconcat_axis0 failure\n", kernel_prefix.c_str());
        } else if (parameters_->axis == 1) {
            const size_t input_size = input_tensors_.size();
            if ((precision_ == PrecisionType::FP32 || precision_ == PrecisionType::FP16) && input_size <= 6) {
                status = runtime_->setKernel(&kernel_, "concat_axis1_size6", precision_);
                CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel Concat failure\n");
            } else {
                status = runtime_->setKernel(&kernel_, kernel_prefix + "concat_axis1", precision_);
                CHECK_EXPR_RETURN_FAILURE(
                    Status::SUCCESS == status, "setKernel %sconcat_axis1 failure\n", kernel_prefix.c_str());
            }
        } else if (parameters_->axis == 2) {
            status = runtime_->setKernel(&kernel_, kernel_prefix + "concat_axis2", precision_);
            CHECK_EXPR_RETURN_FAILURE(
                Status::SUCCESS == status, "setKernel %sconcat_axis2 failure\n", kernel_prefix.c_str());
        } else if (parameters_->axis == 3) {
            status = runtime_->setKernel(&kernel_, kernel_prefix + "concat_axis3", precision_);
            CHECK_EXPR_RETURN_FAILURE(
                Status::SUCCESS == status, "setKernel %sconcat_axis3 failure\n", kernel_prefix.c_str());
        }
    }

    return Status::SUCCESS;
}

Status CLConcat::execute() {
    ENN_DBG_PRINT("CLConcat::execute() is called\n");

    bool is_empty = true;
    for (auto input_tensor : input_tensors_) {
        if (input_tensor->getTotalSizeFromDims() != 0) {
            is_empty = false;
            break;
        }
    }
    if (is_empty) {
        ENN_DBG_PRINT("CLConcat execute return for zero_sized input\n");
        return Status::SUCCESS;
    }

    Status status = Status::SUCCESS;
    if (parameters_->storage_type == StorageType::TEXTURE) {
        status = texture_impl_->execute();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "CLConcatTextureImpl execute failure\n");
    } else {
        if (precision_ == PrecisionType::UINT8 || precision_ == PrecisionType::INT8) {
            status = concatQuant();
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "concatQuant execute failure\n");
        } else {
            status = concatFloat();
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "concatFloat execute failure\n");
        }
    }

    return Status::SUCCESS;
}

Status CLConcat::concatQuant() {
    ENN_DBG_PRINT("CLConcat::concatQuant() is called\n");

    int32_t qmin = 0;
    int32_t qmax = 0;
    if (precision_ == PrecisionType::INT8) {
        qmin = std::numeric_limits<int8_t>::min();
        qmax = std::numeric_limits<int8_t>::max();
    } else {
        qmin = std::numeric_limits<uint8_t>::min();
        qmax = std::numeric_limits<uint8_t>::max();
    }

    Status status;
    if (parameters_->axis == 0) {
        status = concatAxis0(true, qmin, qmax);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "concatAxis0 for quant execute failure\n");
    } else if (parameters_->axis == 1) {
        status = concatAxis1(true, qmin, qmax);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "concatAxis1 for quant execute failure\n");
    } else if (parameters_->axis == 2) {
        status = concatAxis2(true, qmin, qmax);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "concatAxis2 for quant execute failure\n");
    } else if (parameters_->axis == 3) {
        status = concatAxis3(true, qmin, qmax);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "concatAxis3 for quant execute failure\n");
    }

    return Status::SUCCESS;
}

Status CLConcat::concatFloat() {
    ENN_DBG_PRINT("CLConcat::concatFloat() is called\n");

    Status status;
    if (parameters_->axis == 0) {
        status = concatAxis0(false, 0, 0);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "concatAxis0 for float execute failure\n");
    } else if (parameters_->axis == 1) {
        if (input_tensors_.size() > 6) {
            status = concatAxis1(false, 0, 0);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "concatAxis1 for float execute failure\n");
        } else {
            uint32_t output_offset = 0;
            uint32_t offset[6] = {0};
            uint32_t input_count[6] = {0};
            uint32_t max_input_count = 0;

            const size_t input_size = input_tensors_.size();
            Dim4 input_dim;
            for (size_t i = 0; i < input_size; ++i) {
                auto input_tensor = std::static_pointer_cast<CLTensor>(input_tensors_[i]);
                input_dim = input_tensor->getDim();
                offset[i] = output_offset;
                input_count[i] = input_dim.c * input_dim.h * input_dim.w;
                output_offset += input_count[i];
                if (input_count[i] > max_input_count) {
                    max_input_count = input_count[i];
                }
            }

            auto input_data_0 = input_size > 0 ? input_tensors_[0]->getDataPtr() : nullptr;
            auto input_data_1 = input_size > 1 ? input_tensors_[1]->getDataPtr() : nullptr;
            auto input_data_2 = input_size > 2 ? input_tensors_[2]->getDataPtr() : nullptr;
            auto input_data_3 = input_size > 3 ? input_tensors_[3]->getDataPtr() : nullptr;
            auto input_data_4 = input_size > 4 ? input_tensors_[4]->getDataPtr() : nullptr;
            auto input_data_5 = input_size > 5 ? input_tensors_[5]->getDataPtr() : nullptr;
            auto output_data = output_tensor_->getDataPtr();
            status = runtime_->setKernelArg(kernel_.get(),
                                            input_data_0,
                                            input_data_1,
                                            input_data_2,
                                            input_data_3,
                                            input_data_4,
                                            input_data_5,
                                            output_data,
                                            offset[0],
                                            offset[1],
                                            offset[2],
                                            offset[3],
                                            offset[4],
                                            offset[5],
                                            input_count[0],
                                            input_count[1],
                                            input_count[2],
                                            input_count[3],
                                            input_count[4],
                                            input_count[5],
                                            output_offset);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");

            size_t global[3] = {0, 0, 0};
            size_t local[3] = {1, 1, 64};
            global[0] = output_tensor_->getDim().n;
            global[1] = input_tensors_.size();
            global[2] = alignTo(ceil(max_input_count / 8.0), 64);

            status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)3, global, local);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "executeKernel failure\n");
        }
    } else if (parameters_->axis == 2) {
        status = concatAxis2(false, 0, 0);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "concatAxis2 for float execute failure\n");
    } else if (parameters_->axis == 3) {
        status = concatAxis3(false, 0, 0);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "concatAxis3 for float execute failure\n");
    }

    return Status::SUCCESS;
}

Status CLConcat::concatAxis0(const bool &is_quant, const int32_t &qmin, const int32_t &qmax) {
    uint32_t offset = 0;
    for (auto input_tensor : input_tensors_) {
        if (input_tensor->getTotalSizeFromDims() == 0) {
            continue;
        }

        auto input_data = input_tensor->getDataPtr();
        auto output_data = output_tensor_->getDataPtr();
        const Dim4 input_dim = input_tensor->getDim();

        Status status;
        if (is_quant) {
            status = runtime_->setKernelArg(kernel_.get(),
                                            input_data,
                                            output_data,
                                            offset,
                                            input_tensor->getScale(),
                                            input_tensor->getZeroPoint(),
                                            output_tensor_->getScale(),
                                            output_tensor_->getZeroPoint(),
                                            qmin,
                                            qmax);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg ConcatAxis0 for quant failure\n");
        } else {
            status = runtime_->setKernelArg(kernel_.get(), input_data, output_data, offset);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg ConcatAxis0 for float failure\n");
        }

        size_t global[3] = {0, 0, 0};
        size_t local[3] = {0, 0, 0};
        global[0] = input_dim.n;
        global[1] = input_dim.c;
        global[2] = input_dim.h * input_dim.w;
        local[0] = 1;
        local[1] = 1;
        local[2] = findMaxFactor(global[2], 128);

        status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)3, global, local);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "executeKernel ConcatAxis0 failure\n");
        offset += input_dim.n * input_dim.c * input_dim.h * input_dim.w;
    }

    return Status::SUCCESS;
}

Status CLConcat::concatAxis1(const bool &is_quant, const int32_t &qmin, const int32_t &qmax) {
    const Dim4 output_dim = output_tensor_->getDim();
    const uint32_t batch_offset = output_dim.c * output_dim.h * output_dim.w;
    uint32_t offset = 0;

    uint32_t pic_offset = 0;
    if (!is_quant) {
        Dim4 input_dim;
        for (auto input_tensor : input_tensors_) {
            input_dim = input_tensor->getDim();
            pic_offset += input_dim.c * input_dim.h * input_dim.w;
        }
    }

    for (auto input_tensor : input_tensors_) {
        if (input_tensor->getTotalSizeFromDims() == 0) {
            continue;
        }

        auto input_data = input_tensor->getDataPtr();
        auto output_data = output_tensor_->getDataPtr();
        const Dim4 input_dim = input_tensor->getDim();

        const uint32_t size_input = input_dim.c * input_dim.h * input_dim.w;
        const uint32_t channel_size = input_dim.h * input_dim.w;

        Status status;
        if (is_quant) {
            status = runtime_->setKernelArg(kernel_.get(),
                                            input_data,
                                            output_data,
                                            batch_offset,
                                            channel_size,
                                            offset,
                                            input_tensor->getScale(),
                                            input_tensor->getZeroPoint(),
                                            output_tensor_->getScale(),
                                            output_tensor_->getZeroPoint(),
                                            qmin,
                                            qmax);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg ConcatAxis1 for quant failure\n");

            size_t global[3] = {0, 0, 0};
            size_t local[3] = {0, 0, 0};
            global[0] = input_dim.n;
            global[1] = input_dim.c;
            global[2] = input_dim.h * input_dim.w;
            local[0] = 1;
            local[1] = 1;
            local[2] = findMaxFactor(global[2], 128);

            status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)3, global, local);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "executeKernel ConcatAxis1 for quant failure\n");
        } else {
            status = runtime_->setKernelArg(kernel_.get(), input_data, output_data, offset, pic_offset, size_input);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg ConcatAxis1 for float failure\n");

            size_t global[2] = {0, 0};
            size_t local[2] = {1, 64};
            global[0] = input_dim.n;
            global[1] = alignTo(ceil(size_input / 8.0), 64);

            status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)2, global, local);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "executeKernel ConcatAxis1 for quant failure\n");
        }

        offset += size_input;
    }

    return Status::SUCCESS;
}

Status CLConcat::concatAxis2(const bool &is_quant, const int32_t &qmin, const int32_t &qmax) {
    const Dim4 output_dim = output_tensor_->getDim();
    const uint32_t batch_offset = output_dim.c * output_dim.h * output_dim.w;
    const uint32_t channel_offset = output_dim.h * output_dim.w;
    const uint32_t input_width = output_dim.w;
    uint32_t height_offset = 0;
    for (auto input_tensor : input_tensors_) {
        if (input_tensor->getTotalSizeFromDims() == 0) {
            continue;
        }

        auto input_data = input_tensor->getDataPtr();
        auto output_data = output_tensor_->getDataPtr();
        const Dim4 input_dim = input_tensor->getDim();

        Status status;
        if (is_quant) {
            status = runtime_->setKernelArg(kernel_.get(),
                                            input_data,
                                            output_data,
                                            batch_offset,
                                            channel_offset,
                                            height_offset,
                                            input_width,
                                            input_tensor->getScale(),
                                            input_tensor->getZeroPoint(),
                                            output_tensor_->getScale(),
                                            output_tensor_->getZeroPoint(),
                                            qmin,
                                            qmax);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg ConcatAxis2 for quant failure\n");
        } else {
            status = runtime_->setKernelArg(
                kernel_.get(), input_data, output_data, batch_offset, channel_offset, height_offset, input_width);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg ConcatAxis2 for float failure\n");
        }

        size_t global[3] = {0, 0, 0};
        size_t local[3] = {0, 0, 0};
        global[0] = input_dim.n;
        global[1] = input_dim.c;
        global[2] = input_dim.h * input_dim.w;
        local[0] = 1;
        local[1] = 1;
        local[2] = findMaxFactor(global[2], 128);

        status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)3, global, local);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "executeKernel ConcatAxis2 failure\n");
        height_offset += input_dim.h;
    }

    return Status::SUCCESS;
}

Status CLConcat::concatAxis3(const bool &is_quant, const int32_t &qmin, const int32_t &qmax) {
    const Dim4 output_dim = output_tensor_->getDim();
    const uint32_t batch_offset = output_dim.c * output_dim.h * output_dim.w;
    const uint32_t channel_offset = output_dim.h * output_dim.w;
    const uint32_t width_offset = output_dim.w;
    uint32_t current_width = 0;

    for (auto input_tensor : input_tensors_) {
        if (input_tensor->getTotalSizeFromDims() == 0) {
            continue;
        }

        auto input_data = input_tensor->getDataPtr();
        auto output_data = output_tensor_->getDataPtr();
        const Dim4 input_dim = input_tensor->getDim();

        Status status;
        if (is_quant) {
            status = runtime_->setKernelArg(kernel_.get(),
                                            input_data,
                                            output_data,
                                            batch_offset,
                                            channel_offset,
                                            width_offset,
                                            current_width,
                                            input_dim.w,
                                            input_tensor->getScale(),
                                            input_tensor->getZeroPoint(),
                                            output_tensor_->getScale(),
                                            output_tensor_->getZeroPoint(),
                                            qmin,
                                            qmax);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg ConcatAxis3 for quant failure\n");
        } else {
            status = runtime_->setKernelArg(kernel_.get(),
                                            input_data,
                                            output_data,
                                            batch_offset,
                                            channel_offset,
                                            width_offset,
                                            current_width,
                                            input_dim.w);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg ConcatAxis3 for quant failure\n");
        }

        size_t global[3] = {0, 0, 0};
        size_t local[3] = {0, 0, 0};
        global[0] = input_dim.n;
        global[1] = input_dim.c;
        global[2] = input_dim.h * input_dim.w;
        local[0] = 1;
        local[1] = 1;
        local[2] = findMaxFactor(global[2], 128);

        status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)3, global, local);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "executeKernel ConcatAxis3 failure\n");
        current_width += input_dim.w;
    }

    return Status::SUCCESS;
}

Status CLConcat::release() {
    ENN_DBG_PRINT("CLConcat::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
