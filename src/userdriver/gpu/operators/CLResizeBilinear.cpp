#include "userdriver/gpu/operators/cl_quantized_utils/KernelUtil.hpp"
#include "CLResizeBilinear.hpp"

namespace enn {
namespace ud {
namespace gpu {
namespace {
constexpr int INPUT_INDEX = 0;
constexpr int OUTPUT_INDEX = 0;

class CLResizeBilinearTextureImpl {
  public:
    CLResizeBilinearTextureImpl(CLResizeBilinear *base) : base_(base) {}
    Status initialize() {
        auto input_dim = base_->input_tensor_->getDim();
        auto output_dim = base_->output_tensor_->getDim();
        int grid[3] = {(int)(output_dim.w * output_dim.n), (int)output_dim.h, (int)base_->output_tensor_->getSlice()};
        int best_work_group[3] = {0, 0, 0};
        GetBestWorkGroup(grid, best_work_group, (int)base_->runtime_->getMaxWorkGroupSize()[2]);

        local_[0] = static_cast<size_t>(best_work_group[0]);
        local_[1] = static_cast<size_t>(best_work_group[1]);
        local_[2] = static_cast<size_t>(best_work_group[2]);
        global_[0] = static_cast<size_t>(AlignByN(grid[0], local_[0]));
        global_[1] = static_cast<size_t>(AlignByN(grid[1], local_[1]));
        global_[2] = static_cast<size_t>(AlignByN(grid[2], local_[2]));

        src_size_x_ = input_dim.w * input_dim.n;
        src_size_y_ = input_dim.h;
        src_size_z_ = base_->input_tensor_->getSlice();
        dst_size_x_ = output_dim.w * output_dim.n;
        dst_size_y_ = output_dim.h;
        dst_size_z_ = base_->output_tensor_->getSlice();
        border_x_ = input_dim.w - 1;
        border_y_ = input_dim.h - 1;
        scale_factor_x_ = CalculateResizeScale(input_dim.w, output_dim.w, base_->parameters_->align_corners);
        scale_factor_y_ = CalculateResizeScale(input_dim.h, output_dim.h, base_->parameters_->align_corners);

        Status status = base_->runtime_->setKernel(&kernel_, "resize_bilinear_tflite", base_->precision_);
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "setKernel resize_bilinear_tflite failure\n");

        return Status::SUCCESS;
    }

    Status execute();

  private:
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
    CLResizeBilinear *base_;

    // 2. Operator resource
    size_t local_[3] = {0, 0, 0};
    size_t global_[3] = {0, 0, 0};
    int src_size_x_ = 0;
    int src_size_y_ = 0;
    int src_size_z_ = 0;
    int dst_size_x_ = 0;
    int dst_size_y_ = 0;
    int dst_size_z_ = 0;
    int border_x_ = 0;
    int border_y_ = 0;
    float scale_factor_x_ = 0;
    float scale_factor_y_ = 0;

    // 3. Operator kernels
    std::shared_ptr<struct _cl_kernel> kernel_;
};
}  // namespace

CLResizeBilinear::CLResizeBilinear(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    ENN_DBG_PRINT("CLResizeBilinear is created\n");
    input_tensor_ = nullptr;
    output_tensor_ = nullptr;
    parameters_ = nullptr;
    kernel_ = nullptr;
    height_scale_ = 0.0f;
    width_scale_ = 0.0f;
    use_resize_bilinear_32_to_512_opt_ = false;
    layout_convert_nhwc2nchw_ = nullptr;
    layout_convert_nchw2nhwc_ = nullptr;
    texture_impl_ = nullptr;
}

Status CLResizeBilinear::initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                                    const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                                    const std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLResizeBilinear::initialize() is called\n");

    input_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(INPUT_INDEX));
    output_tensor_ = std::static_pointer_cast<CLTensor>(output_tensors.at(OUTPUT_INDEX));
    parameters_ = std::static_pointer_cast<ResizeBilinearParameters>(parameters);
    CHECK_AND_RETURN_ERR(nullptr == parameters_, Status::FAILURE, "CLResizeBilinear must have parameters\n");
    ENN_DBG_PRINT(
        " ResizeBilinearParameters height: %d width: %d, align_cornor: %d half_pixel_centers: %d, androidnn: %d, isNCHW: "
        "%d, storage_type: %d, compute_type: %d\n",
        parameters_->new_height,
        parameters_->new_width,
        parameters_->align_corners,
        parameters_->half_pixel_centers,
        parameters_->androidNN,
        parameters_->isNCHW,
        parameters_->storage_type,
        parameters_->compute_type);

    Dim4 input_dim = input_tensor_->getDim();
    Dim4 output_dim = output_tensor_->getDim();
    Dim4 input_dim_nchw = input_dim;
    Dim4 output_dim_nchw = output_dim;
    Status status = Status::SUCCESS;
    if (parameters_->androidNN && parameters_->storage_type != StorageType::TEXTURE) {
        Dim4 input_dim = input_tensor_->getDim();
        if (!parameters_->isNCHW) {
            input_dim_nchw = convertDimToNCHW(input_dim);
        }
        output_dim_nchw = {input_dim_nchw.n,
                           input_dim_nchw.c,
                           static_cast<uint32_t>(parameters_->new_height),
                           static_cast<uint32_t>(parameters_->new_width)};
        Dim4 output_dim = parameters_->isNCHW ? output_dim_nchw : convertDimToNHWC(output_dim_nchw);

        if (!isDimsSame(output_dim, output_tensor_->getDim())) {
            output_tensor_->reconfigureDimAndBuffer(output_dim);
        }

        if (!parameters_->isNCHW) {
            auto nchw_input_tensor = std::make_shared<CLTensor>(runtime_,
                                                                precision_,
                                                                input_tensor_->getDataType(),
                                                                input_dim_nchw,
                                                                input_tensor_->getDataOrder(),
                                                                input_tensor_->getScale(),
                                                                input_tensor_->getZeroPoint());
            auto nchw_output_tensor = std::make_shared<CLTensor>(runtime_,
                                                                 precision_,
                                                                 output_tensor_->getDataType(),
                                                                 output_dim_nchw,
                                                                 output_tensor_->getDataOrder(),
                                                                 output_tensor_->getScale(),
                                                                 output_tensor_->getZeroPoint());
            auto layout_convert_nhwc2nchw_parameters = std::make_shared<LayoutConvertParameters>();
            layout_convert_nhwc2nchw_parameters->data_order_change_type = DataOrderChangeType::NHWC2NCHW;

            layout_convert_nhwc2nchw_ = std::make_shared<CLLayoutConvert>(runtime_, precision_);
            status = layout_convert_nhwc2nchw_->initialize(
                {input_tensor_}, {nchw_input_tensor}, layout_convert_nhwc2nchw_parameters);
            CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "LayoutConvert nhwc2nchw initialize failure\n");

            auto layout_convert_nchw2nhwc_parameters = std::make_shared<LayoutConvertParameters>();
            layout_convert_nchw2nhwc_parameters->data_order_change_type = DataOrderChangeType::NCHW2NHWC;

            layout_convert_nchw2nhwc_ = std::make_shared<CLLayoutConvert>(runtime_, precision_);
            status = layout_convert_nchw2nhwc_->initialize(
                {nchw_output_tensor}, {output_tensor_}, layout_convert_nchw2nhwc_parameters);
            CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "LayoutConvert nchw2nhwc initialize failure\n");

            input_tensor_ = nchw_input_tensor;
            output_tensor_ = nchw_output_tensor;
        }
    }

    if (parameters_->compute_type == ComputeType::Caffe) {
        height_scale_ = static_cast<float>(input_dim_nchw.h - 1) / (output_dim_nchw.h - 1);
        width_scale_ = static_cast<float>(input_dim_nchw.w - 1) / (output_dim_nchw.w - 1);
    } else {
        height_scale_ = static_cast<float>(input_dim_nchw.h) / output_dim_nchw.h;
        width_scale_ = static_cast<float>(input_dim_nchw.w) / output_dim_nchw.w;
        if (parameters_->align_corners && output_dim_nchw.h > 1) {
            height_scale_ = static_cast<float>(input_dim_nchw.h - 1) / (output_dim_nchw.h - 1);
        }
        if (parameters_->align_corners && output_dim_nchw.w > 1) {
            width_scale_ = static_cast<float>(input_dim_nchw.w - 1) / (output_dim_nchw.w - 1);
        }
    }

    if (parameters_->storage_type == StorageType::TEXTURE &&
        (precision_ == PrecisionType::FP16 || precision_ == PrecisionType::FP32)) {
        texture_impl_ = std::make_shared<CLResizeBilinearTextureImpl>(this);
        texture_impl_->initialize();
    } else if (precision_ == PrecisionType::INT8) {
        status = runtime_->setKernel(&kernel_, "SIGNEDresize_bilinear", precision_);
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "setKernel SIGNEDresize_bilinear failure\n");
    } else if (precision_ == PrecisionType::UINT8 && parameters_->align_corners && input_dim.h == 32 && input_dim.w == 32 &&
               output_dim.h == 512 && output_dim.h == 512) {
        use_resize_bilinear_32_to_512_opt_ = true;
        status = runtime_->setKernel(&kernel_, "resize_bilinear_32_to_512", precision_);
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "setKernel resize_bilinear_32_to_512 failure\n");
    } else {
        status = runtime_->setKernel(&kernel_, "resize_bilinear", precision_);
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "setKernel resize_bilinear failure\n");
    }

    float_input_data_ = make_shared_array<float>(input_tensor_->getTotalSizeFromDims());
    float_output_data_ = make_shared_array<float>(output_tensor_->getTotalSizeFromDims());
    half_input_data_ = make_shared_array<half_float::half>(input_tensor_->getTotalSizeFromDims());
    half_output_data_ = make_shared_array<half_float::half>(output_tensor_->getTotalSizeFromDims());
    int8_input_data_ = make_shared_array<int8_t>(input_tensor_->getTotalSizeFromDims());
    int8_output_data_ = make_shared_array<int8_t>(output_tensor_->getTotalSizeFromDims());
    uint8_input_data_ = make_shared_array<uint8_t>(input_tensor_->getTotalSizeFromDims());
    uint8_output_data_ = make_shared_array<uint8_t>(output_tensor_->getTotalSizeFromDims());

    return Status::SUCCESS;
}

Status CLResizeBilinear::execute() {
    ENN_DBG_PRINT("CLResizeBilinear::execute() is called\n");
    if (input_tensor_->getTotalSizeFromDims() == 0) {
        ENN_DBG_PRINT("CLResizeBilinear execute return for zero_sized input_tensor_\n");
        return Status::SUCCESS;
    }

    const Dim4 input_dim = input_tensor_->getDim();
    const Dim4 output_dim = output_tensor_->getDim();

    Status status = Status::SUCCESS;
    if (parameters_->androidNN && !parameters_->isNCHW && parameters_->storage_type != StorageType::TEXTURE) {
        status = layout_convert_nhwc2nchw_->execute();
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "layout_convert_nhwc2nchw execute failure\n");
    }

    // execute for CTS
    // The result ceil and floor function in cpu and gpu will be different.
    // use this branch to pass CTS TestRandomGraph/SingleOperationTest
    {
        const int32_t channel = input_dim.c;
        const int32_t input_h = input_dim.h;
        const int32_t input_w = input_dim.w;
        const int32_t output_h = output_dim.h;
        const int32_t output_w = output_dim.w;

        bool is_known_params = false;
        if (input_h != input_w || output_h != output_w || channel % 16 != 0)
            is_known_params = false;
        if ((input_h == 1 && output_h == 33 && channel == 256) || (input_h == 33 && output_h == 129 && channel == 256))
            is_known_params = true;
        if (parameters_->androidNN && parameters_->isNCHW && !is_known_params) {
            status = execute_for_cts();
            CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "execute_for_cts() failure\n");

            return Status::SUCCESS;
        }
    }

    if (use_resize_bilinear_32_to_512_opt_) {
        status = runtime_->setKernelArg(kernel_.get(),
                                        input_tensor_->getDataPtr(),
                                        output_tensor_->getDataPtr(),
                                        input_dim.h,
                                        input_dim.w,
                                        output_dim.n,
                                        output_dim.c,
                                        output_dim.h,
                                        output_dim.w,
                                        height_scale_,
                                        width_scale_);
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "setKernel failure\n");

        size_t global[2] = {0, 16};
        size_t local[2] = {16, 16};
        global[0] = alignTo(output_dim.n * output_dim.c * (output_dim.h / 16) * (output_dim.w / 16), local[0]);

        status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)2, global, local);
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "execute kernel failure\n");
    } else {
        status = runtime_->setKernelArg(kernel_.get(),
                                        input_tensor_->getDataPtr(),
                                        output_tensor_->getDataPtr(),
                                        input_dim.h,
                                        input_dim.w,
                                        output_dim.n,
                                        output_dim.c,
                                        output_dim.h,
                                        output_dim.w,
                                        height_scale_,
                                        width_scale_,
                                        (int)parameters_->half_pixel_centers);
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "setKernelArg failure\n");

        size_t local = 16;
        size_t global = alignTo(output_dim.n * output_dim.c * output_dim.h * output_dim.w, local);
        status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)1, &global, &local);
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "enqueueKernel failure\n");
    }

    if (parameters_->androidNN && !parameters_->isNCHW && parameters_->storage_type != StorageType::TEXTURE) {
        status = layout_convert_nchw2nhwc_->execute();
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "layout_convert_nchw2nhwc execute failure\n");
    }

    return Status::SUCCESS;
}

Status CLResizeBilinear::execute_for_cts() {
    ENN_DBG_PRINT("CLResizeBilinear::execute_for_cts() is called");
    const auto input_dim = input_tensor_->getDim();
    const auto output_dim = output_tensor_->getDim();

    Status status = Status::SUCCESS;
    switch (output_tensor_->getDataType()) {
    case DataType::FLOAT: {
        status = input_tensor_->readData(float_input_data_.get());
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "CLResizeBilinear readData failure: DataType::FLOAT\n");

        status = execute_on_cpu(float_input_data_, float_output_data_, input_dim, output_dim);
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "CLResizeBilinear execute_on_cpu failure\n");

        status = output_tensor_->writeData(float_output_data_.get());
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "CLResizeBilinear writeData failure: DataType::FLOAT\n");

        break;
    }
    case DataType::HALF: {
        status = input_tensor_->readData(half_input_data_.get());
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "CLResizeBilinear readData failure: DataType::HALF\n");

        status = execute_on_cpu(half_input_data_, half_output_data_, input_dim, output_dim);
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "CLResizeBilinear execute_on_cpu failure\n");

        status = output_tensor_->writeData(half_output_data_.get());
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "CLResizeBilinear writeData failure: DataType::HALF\n");

        break;
    }
    case DataType::INT8: {
        status = input_tensor_->readData(int8_input_data_.get());
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "CLResizeBilinear readData failure: DataType::INT8\n");

        status = execute_on_cpu(int8_input_data_, int8_output_data_, input_dim, output_dim);
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "CLResizeBilinear execute_on_cpu failure\n");

        status = output_tensor_->writeData(int8_output_data_.get());
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "CLResizeBilinear writeData failure: DataType::INT8\n");

        break;
    }
    case DataType::UINT8: {
        status = input_tensor_->readData(uint8_input_data_.get());
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "CLResizeBilinear readData failure: DataType::UINT8\n");

        status = execute_on_cpu(uint8_input_data_, uint8_output_data_, input_dim, output_dim);
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "CLResizeBilinear execute_on_cpu failure\n");

        status = output_tensor_->writeData(uint8_output_data_.get());
        CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "CLResizeBilinear writeData failure: DataType::UINT8\n");

        break;
    }
    default: break;
    }

    return Status::SUCCESS;
}

inline void ComputeInterpolationValues(const float value,
                                       const float scale,
                                       const bool half_pixel_centers,
                                       int32_t input_size,
                                       float *scaled_value,
                                       int32_t *lower_bound,
                                       int32_t *upper_bound) {
    if (half_pixel_centers) {
        *scaled_value = (value + 0.5f) * scale - 0.5f;
    } else {
        *scaled_value = value * scale;
    }
    float scaled_value_floor = std::floor(*scaled_value);
    *lower_bound = std::max(static_cast<int32_t>(scaled_value_floor), static_cast<int32_t>(0));
    *upper_bound = std::min(static_cast<int32_t>(std::ceil(*scaled_value)), input_size - 1);
}

inline int Offset(Dim4 tmp_dim, int i0, int i1, int i2, int i3) {
    return ((i0 * tmp_dim.c + i1) * tmp_dim.h + i2) * tmp_dim.w + i3;
}

template <typename T>
Status CLResizeBilinear::execute_on_cpu(std::shared_ptr<T> tmp_input_data,
                                        std::shared_ptr<T> tmp_output_data,
                                        Dim4 input_dim,
                                        Dim4 output_dim) {
    auto input_height = input_dim.h;
    auto input_width = input_dim.w;

    auto batches = output_dim.n;
    auto depth = output_dim.c;
    auto output_height = output_dim.h;
    auto output_width = output_dim.w;

    for (int b = 0; b < batches; ++b) {
        for (int y = 0; y < output_height; ++y) {
            float input_y;
            int32_t y0, y1;
            ComputeInterpolationValues(y, height_scale_, parameters_->half_pixel_centers, input_height, &input_y, &y0, &y1);
            for (int x = 0; x < output_width; ++x) {
                float input_x;
                int32_t x0, x1;
                ComputeInterpolationValues(
                    x, width_scale_, parameters_->half_pixel_centers, input_width, &input_x, &x0, &x1);
                for (int c = 0; c < depth; ++c) {
                    T interpolation = static_cast<T>(
                        tmp_input_data.get()[Offset(input_dim, b, c, y0, x0)] * (1 - (input_y - y0)) * (1 - (input_x - x0)) +
                        tmp_input_data.get()[Offset(input_dim, b, c, y1, x0)] * (input_y - y0) * (1 - (input_x - x0)) +
                        tmp_input_data.get()[Offset(input_dim, b, c, y0, x1)] * (1 - (input_y - y0)) * (input_x - x0) +
                        tmp_input_data.get()[Offset(input_dim, b, c, y1, x1)] * (input_y - y0) * (input_x - x0));
                    tmp_output_data.get()[Offset(output_dim, b, c, y, x)] = interpolation;
                }
            }
        }
    }

    return Status::SUCCESS;
}

Status CLResizeBilinear::release() {
    ENN_DBG_PRINT("CLResizeBilinear::release() is called");
    return Status::SUCCESS;
}

namespace {
Status CLResizeBilinearTextureImpl::execute() {
    auto input_data = base_->input_tensor_->getDataPtr();
    auto output_data = base_->output_tensor_->getDataPtr();

    Status status = Status::SUCCESS;

    status = base_->runtime_->setKernelArg(kernel_.get(),
                                           input_data,
                                           output_data,
                                           src_size_x_,
                                           src_size_y_,
                                           src_size_z_,
                                           dst_size_x_,
                                           dst_size_y_,
                                           dst_size_z_,
                                           border_x_,
                                           border_y_,
                                           scale_factor_x_,
                                           scale_factor_y_);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "CLResizeBilinearTextureImpl setKernelArg failure\n");

    status = base_->runtime_->enqueueKernel(kernel_.get(), (cl_uint)3, global_, local_);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "CLResizeBilinearTextureImpl enqueueKernel failure\n");

    return Status::SUCCESS;
}
}  // namespace

}  // namespace gpu
}  // namespace ud
}  // namespace enn
