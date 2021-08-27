#include "userdriver/gpu/operators/cl_quantized_utils/KernelUtil.hpp"
#include "CLAveragepool.hpp"

namespace enn {
namespace ud {
namespace gpu {
namespace {
constexpr int INPUT_INDEX = 0;
constexpr int OUTPUT_INDEX = 0;
}  // namespace

CLAveragepool::CLAveragepool(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision), kernel_max_work_group_size_(0), input_tensor_(nullptr),
    output_tensor_(nullptr), nchw_input_tensor_(nullptr), nchw_output_tensor_(nullptr), parameters_(nullptr),
    cl_activation_(nullptr), kernel_(nullptr) {
    ENN_DBG_PRINT("CLAveragepool is created\n");
}

Status CLAveragepool::initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                                 const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                                 const std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLAveragepool::initialize() is called\n");

    input_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(INPUT_INDEX));
    output_tensor_ = std::static_pointer_cast<CLTensor>(output_tensors.at(OUTPUT_INDEX));
    parameters_ = std::static_pointer_cast<Pool2DParameters>(parameters);
    CHECK_EXPR_RETURN_FAILURE(nullptr != parameters_, "CLAveragepool must have parameters\n");
    ENN_DBG_PRINT(
        "CLMaxpool: padding.t %d padding.r %d padding.b %d padding.l %d; stride.h %d stride.w %d; filter.h %d filter.w %d; "
        "activation_info_: isEnabled() %d, activation() %d; androidNN %d; isNCHW %d; storage_type %d; compute_type %d;\n",
        parameters_->padding.t,
        parameters_->padding.r,
        parameters_->padding.b,
        parameters_->padding.l,
        parameters_->stride.h,
        parameters_->stride.w,
        parameters_->filter.h,
        parameters_->filter.w,
        parameters_->activation_info.isEnabled(),
        parameters_->activation_info.activation(),
        parameters_->androidNN,
        parameters_->isNCHW,
        parameters_->storage_type,
        parameters_->compute_type);

    Status status = Status::SUCCESS;
    if (parameters_->androidNN && parameters_->storage_type != StorageType::TEXTURE) {
        Dim4 input_dim = input_tensor_->getDim();
        Dim4 output_dim = output_tensor_->getDim();
        if (!parameters_->isNCHW) {
            input_dim = convertDimToNCHW(input_dim);
        }

        output_dim.n = input_dim.n;
        output_dim.c = input_dim.c;
        output_dim.h =
            (input_dim.h - parameters_->filter.h + parameters_->stride.h + parameters_->padding.t + parameters_->padding.b) /
            parameters_->stride.h;
        output_dim.w =
            (input_dim.w - parameters_->filter.w + parameters_->stride.w + parameters_->padding.l + parameters_->padding.r) /
            parameters_->stride.w;
        if (!parameters_->isNCHW) {
            nchw_input_tensor_ = std::make_shared<CLTensor>(runtime_,
                                                            precision_,
                                                            input_tensor_->getDataType(),
                                                            input_dim,
                                                            input_tensor_->getDataOrder(),
                                                            input_tensor_->getScale(),
                                                            input_tensor_->getZeroPoint());
            nchw_output_tensor_ = std::make_shared<CLTensor>(runtime_,
                                                             precision_,
                                                             output_tensor_->getDataType(),
                                                             output_dim,
                                                             output_tensor_->getDataOrder(),
                                                             output_tensor_->getScale(),
                                                             output_tensor_->getZeroPoint());
        }

        if (!parameters_->isNCHW) {
            output_dim = convertDimToNHWC(output_dim);
        }

        if (!isDimsSame(output_dim, output_tensor_->getDim())) {
            output_tensor_->reconfigureDimAndBuffer(output_dim);
        }
    }

    status = set_kernel();
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "set_kernel() failure\n");

    auto activation_parameters = std::make_shared<ActivationParameters>();
    activation_parameters->activation_info = parameters_->activation_info;
    cl_activation_ = std::make_shared<CLActivation>(runtime_, precision_);
    status = cl_activation_->initialize({output_tensor_}, {output_tensor_}, activation_parameters);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "activation initialize failure\n");

    return Status::SUCCESS;
}

Status CLAveragepool::set_kernel() {
    Status status = Status::SUCCESS;
    if (parameters_->compute_type == ComputeType::Caffe) {
        if (parameters_->filter.h >= 15 && precision_ == PrecisionType::FP16) {
            status = runtime_->setKernel(&kernel_, "avepooling_caffe_big_kernelsize", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel avepooling_caffe_big_kernelsize failure\n");
        } else {
            status = runtime_->setKernel(&kernel_, "avepooling_caffe", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel avepooling_caffe failure\n");
        }
    } else if (parameters_->compute_type == ComputeType::TFLite) {
        if (parameters_->storage_type == StorageType::TEXTURE) {
            Status status = runtime_->setKernel(&kernel_, "avepooling_tflite_texture2d", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel avepooling_tflite_texture2d failure\n");
            status =
                runtime_->GetKernelMaxWorkGroupSize(kernel_.get(), runtime_->getDeviceID(), &kernel_max_work_group_size_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "GetKernelMaxWorkGroupSize failure\n");

        } else if (precision_ == PrecisionType::INT8) {
            status = runtime_->setKernel(&kernel_, "SIGNEDavepooling_tflite", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel SIGNEDavepooling_tflite failure\n");
        } else {
            status = runtime_->setKernel(&kernel_, "avepooling_tflite", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel avepooling_tflite failure\n");
        }
    }

    return Status::SUCCESS;
}

Status CLAveragepool::execute() {
    ENN_DBG_PRINT("CLAveragepool::execute() is called\n");
    Status status = Status::SUCCESS;

    if (input_tensor_->getTotalSizeFromDims() == 0) {
        ENN_DBG_PRINT("CLAveragepool execute return for zero_sized input\n");
        return Status::SUCCESS;
    }

    if (parameters_->androidNN && !parameters_->isNCHW && parameters_->storage_type != StorageType::TEXTURE) {
        status = input_tensor_->convertToNCHW(nchw_input_tensor_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "convertToNCHW failure\n");
        status = eval_nchw(nchw_input_tensor_, nchw_output_tensor_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "evalNCHW failure\n");
        status = nchw_output_tensor_->convertToNHWC(output_tensor_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "convertToNHWC failure\n");
    } else {
        status = eval_nchw(input_tensor_, output_tensor_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "evalNCHW failure\n");
    }

    status = cl_activation_->execute();
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "cl_activation_->execute() failure\n");

    return Status::SUCCESS;
}

Status CLAveragepool::eval_nchw(const std::shared_ptr<CLTensor> input_tensor, std::shared_ptr<CLTensor> output_tensor) {
    Status status = Status::FAILURE;
    if (precision_ == PrecisionType::UINT8 || precision_ == PrecisionType::INT8) {
        status = averagepool_quant(input_tensor, output_tensor);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "averagepool_quant execute failure\n");
    } else {
        if (parameters_->storage_type == StorageType::TEXTURE && parameters_->compute_type == ComputeType::TFLite) {
            status = execute_texture2d_float(input_tensor, output_tensor);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "CLAveragepoolTextureImpl execute failure\n");
        } else {
            status = averagepool_float(input_tensor, output_tensor);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "averagepool_float execute failure\n");
        }
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "activation execute failure\n");
    }
    return status;
}

Status CLAveragepool::averagepool_quant(const std::shared_ptr<CLTensor> input_tensor,
                                        std::shared_ptr<CLTensor> output_tensor) {
    ENN_DBG_PRINT("CLAveragepool::averagepool_quant() is called\n");
    auto input_data = input_tensor->getDataPtr();
    auto output_data = output_tensor->getDataPtr();
    auto input_dim = input_tensor->getDim();
    auto output_dim = output_tensor->getDim();

    int32_t activation_min = 0;
    int32_t activation_max = 0;
    if (precision_ == PrecisionType::INT8) {
        CalculateActivationRangeInt8(parameters_->activation_info.activation(),
                                     output_tensor->getScale(),
                                     output_tensor->getZeroPoint(),
                                     &activation_min,
                                     &activation_max);

    } else {
        CalculateActivationRangeUint8(parameters_->activation_info.activation(),
                                      output_tensor->getScale(),
                                      output_tensor->getZeroPoint(),
                                      &activation_min,
                                      &activation_max);
    }

    Status status = runtime_->setKernelArg(kernel_.get(),
                                           input_data,
                                           output_data,
                                           input_dim.h,
                                           input_dim.w,
                                           output_dim.h,
                                           output_dim.w,
                                           parameters_->filter.h,
                                           parameters_->filter.w,
                                           parameters_->stride.h,
                                           parameters_->stride.w,
                                           parameters_->padding.t,
                                           parameters_->padding.l,
                                           activation_max,
                                           activation_min);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel failure\n");

    size_t global[3] = {0, 0, 0};
    size_t local[3] = {1, 1, 32};
    global[0] = output_dim.n;
    global[1] = output_dim.c;
    global[2] = alignTo(ceil(output_dim.h * output_dim.w), local[2]);

    status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)3, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "execute kernel failure\n");

    return Status::SUCCESS;
}

Status CLAveragepool::averagepool_float(const std::shared_ptr<CLTensor> input_tensor,
                                        std::shared_ptr<CLTensor> output_tensor) {
    ENN_DBG_PRINT("CLAveragepool::averagepool_float() is called\n");
    auto input_data = input_tensor->getDataPtr();
    auto output_data = output_tensor->getDataPtr();
    auto input_dim = input_tensor->getDim();
    auto output_dim = output_tensor->getDim();
    Status status = runtime_->setKernelArg(kernel_.get(),
                                           input_data,
                                           output_data,
                                           input_dim.h,
                                           input_dim.w,
                                           output_dim.h,
                                           output_dim.w,
                                           parameters_->filter.h,
                                           parameters_->filter.w,
                                           parameters_->stride.h,
                                           parameters_->stride.w,
                                           parameters_->padding.t,
                                           parameters_->padding.l);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");

    size_t global[3] = {0, 0, 0};
    size_t local[3] = {1, 1, 32};
    global[0] = output_dim.n;
    global[1] = output_dim.c;
    global[2] = alignTo(ceil(output_dim.h * output_dim.w), local[2]);

    status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)3, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "enqueueKernel failure\n");

    return Status::SUCCESS;
}

Status CLAveragepool::execute_texture2d_float(const std::shared_ptr<CLTensor> input_tensor,
                                              std::shared_ptr<CLTensor> output_tensor) {
    auto input_data = input_tensor->getDataPtr();
    auto output_data = output_tensor->getDataPtr();
    auto input_dim = input_tensor->getDim();
    auto output_dim = output_tensor->getDim();

    const int src_x = input_dim.w * input_dim.n;
    const int src_y = input_dim.h;
    const int src_z = input_tensor->getSlice();
    const int src_w = input_dim.n;
    const int dst_x = output_dim.w * output_dim.n;
    const int dst_y = output_dim.h;
    const int dst_z = input_tensor->getSlice();
    const int dst_w = output_dim.n;

    int grid_size[3] = {dst_x, dst_y, dst_z};
    int work_group_size[3];
    get_workgroup(grid_size, kernel_max_work_group_size_, work_group_size);

    Status status = runtime_->setKernelArg(kernel_.get(),
                                           input_data,
                                           output_data,
                                           src_x,
                                           src_y,
                                           src_z,
                                           src_w,
                                           dst_x,
                                           dst_y,
                                           dst_z,
                                           dst_w,
                                           parameters_->filter.w,
                                           parameters_->filter.h,
                                           parameters_->stride.w,
                                           parameters_->stride.h,
                                           parameters_->padding.l * src_w,
                                           parameters_->padding.t);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");

    size_t local[3] = {static_cast<size_t>(work_group_size[0]),
                       static_cast<size_t>(work_group_size[1]),
                       static_cast<size_t>(work_group_size[2])};
    size_t global[3];
    global[0] = static_cast<size_t>(AlignByN(grid_size[0], work_group_size[0]));
    global[1] = static_cast<size_t>(AlignByN(grid_size[1], work_group_size[1]));
    global[2] = static_cast<size_t>(AlignByN(grid_size[2], work_group_size[2]));

    status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)3, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "executeKernel failure\n");

    return Status::SUCCESS;
}

Status CLAveragepool::release() {
    ENN_DBG_PRINT("CLAveragepool::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
