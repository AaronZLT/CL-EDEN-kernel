#include "userdriver/gpu/operators/cl_quantized_utils/KernelUtil.hpp"
#include "CLMaxpool.hpp"

namespace enn {
namespace ud {
namespace gpu {
namespace {
constexpr int INPUT_INDEX = 0;
constexpr int OUTPUT_INDEX = 0;
}  // namespace

CLMaxpool::CLMaxpool(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    ENN_DBG_PRINT("CLMaxpool is created\n");
    input_tensor_ = nullptr;
    output_tensor_ = nullptr;
    parameters_ = nullptr;
    cl_activation_ = nullptr;
    kernel_ = nullptr;
}

Status CLMaxpool::initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                             const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                             const std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLMaxpool::initialize() is called\n");

    input_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(INPUT_INDEX));
    output_tensor_ = std::static_pointer_cast<CLTensor>(output_tensors.at(OUTPUT_INDEX));
    parameters_ = std::static_pointer_cast<Pool2DParameters>(parameters);
    CHECK_EXPR_RETURN_FAILURE(nullptr != parameters_, "CLMaxpool must have parameters\n");

    ENN_DBG_PRINT(
        "CLMaxpool: padding.t %d padding.r %d padding.b %d padding.l %d; stride.h %d stride.w %d; filter.h %d filter.w %d;"
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
    if (parameters_->androidNN) {
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
            input_nchw_tensor_ = std::make_shared<CLTensor>(runtime_,
                                                            precision_,
                                                            input_tensor_->getDataType(),
                                                            input_dim,
                                                            input_tensor_->getDataOrder(),
                                                            input_tensor_->getScale(),
                                                            input_tensor_->getZeroPoint());
            output_nchw_tensor_ = std::make_shared<CLTensor>(runtime_,
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

    if (precision_ == PrecisionType::INT8) {
        status = runtime_->setKernel(&kernel_, "SIGNEDmaxpooling", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel SIGNEDmaxpooling failure\n");
    } else {
        status = runtime_->setKernel(&kernel_, "maxpooling", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel maxpooling failure\n");
    }

    auto activation_parameters = std::make_shared<ActivationParameters>();
    activation_parameters->activation_info = parameters_->activation_info;

    cl_activation_ = std::make_shared<CLActivation>(runtime_, precision_);
    status = cl_activation_->initialize({output_tensor_}, {output_tensor_}, activation_parameters);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "activation initialize failure\n");

    return Status::SUCCESS;
}

Status CLMaxpool::execute() {
    ENN_DBG_PRINT("CLMaxpool::execute() is called\n");
    Status status = Status::SUCCESS;

    if (input_tensor_->getTotalSizeFromDims() == 0) {
        ENN_DBG_PRINT("CLMaxpool execute return for zero_sized input\n");
        return Status::SUCCESS;
    }

    if (parameters_->androidNN && !parameters_->isNCHW) {
        status = input_tensor_->convertToNCHW(input_nchw_tensor_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "convertToNCHW failure\n");
        if (precision_ == PrecisionType::UINT8 || precision_ == PrecisionType::INT8) {
            status = maxpool_quant(input_nchw_tensor_, output_nchw_tensor_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "maxpool_quant execute failure\n");
        } else {
            status = maxpool_float(input_nchw_tensor_, output_nchw_tensor_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "maxpool_float execute failure\n");
        }
        status = output_nchw_tensor_->convertToNHWC(output_tensor_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "convertToNCHW failure\n");
    } else if (precision_ == PrecisionType::UINT8 || precision_ == PrecisionType::INT8) {
        status = maxpool_quant(input_tensor_, output_tensor_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "maxpool_quant execute failure\n");
    } else {
        status = maxpool_float(input_tensor_, output_tensor_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "maxpool_float execute failure\n");
    }

    status = cl_activation_->execute();
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "cl_activation_->execute() failure\n");

    return Status::SUCCESS;
}

Status CLMaxpool::maxpool_quant(const std::shared_ptr<CLTensor> input, std::shared_ptr<CLTensor> output) {
    ENN_DBG_PRINT("CLMaxpool::maxpool_quant() is called\n");
    auto input_data = input->getDataPtr();
    auto output_data = output->getDataPtr();
    auto input_dim = input->getDim();
    auto output_dim = output->getDim();

    int32_t activation_min = 0;
    int32_t activation_max = 0;
    if (precision_ == PrecisionType::INT8) {
        CalculateActivationRangeInt8(parameters_->activation_info.activation(),
                                     output->getScale(),
                                     output->getZeroPoint(),
                                     &activation_min,
                                     &activation_max);

    } else {
        CalculateActivationRangeUint8(parameters_->activation_info.activation(),
                                      output->getScale(),
                                      output->getZeroPoint(),
                                      &activation_min,
                                      &activation_max);
    }

    Status status = runtime_->setKernelArg(kernel_.get(),
                                           input_data,
                                           output_data,
                                           input_dim.h,
                                           input_dim.w,
                                           parameters_->filter.h,
                                           parameters_->filter.w,
                                           parameters_->stride.h,
                                           parameters_->stride.w,
                                           parameters_->padding.t,
                                           parameters_->padding.l,
                                           output_dim.w,
                                           output_dim.h,
                                           activation_max,
                                           activation_min);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel failure\n");

    size_t global[3] = {0, 0, 0};
    size_t local[3] = {1, 1, 16};
    global[0] = output_dim.n;
    global[1] = output_dim.c;
    global[2] = alignTo(ceil(output_dim.h * output_dim.w), local[2]);

    status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)3, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "enqueueKernel kernel failure\n");

    return Status::SUCCESS;
}

Status CLMaxpool::maxpool_float(const std::shared_ptr<CLTensor> input, std::shared_ptr<CLTensor> output) {
    ENN_DBG_PRINT("CLMaxpool::maxpool_float() is called\n");
    auto input_data = input->getDataPtr();
    auto output_data = output->getDataPtr();
    auto input_dim = input->getDim();
    auto output_dim = output->getDim();

    Status status = runtime_->setKernelArg(kernel_.get(),
                                           input_data,
                                           output_data,
                                           input_dim.h,
                                           input_dim.w,
                                           parameters_->filter.h,
                                           parameters_->filter.w,
                                           parameters_->stride.h,
                                           parameters_->stride.w,
                                           parameters_->padding.t,
                                           parameters_->padding.l,
                                           output_dim.w,
                                           output_dim.h);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernelArg failure\n");

    size_t global[3] = {0, 0, 0};
    size_t local[3] = {1, 1, 16};
    global[0] = output_dim.n;
    global[1] = output_dim.c;
    global[2] = alignTo(ceil(output_dim.h * output_dim.w), local[2]);

    status = runtime_->enqueueKernel(kernel_.get(), (cl_uint)3, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "enqueueKernel failure\n");

    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "activation execute failure\n");

    return Status::SUCCESS;
}

Status CLMaxpool::release() {
    ENN_DBG_PRINT("CLMaxpool::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
