#include "userdriver/gpu/operators/CLDepth2Space.hpp"

namespace enn {
namespace ud {
namespace gpu {

namespace {
constexpr int INPUT_INDEX = 0;
constexpr int OUTPUT_INDEX = 0;
}  // namespace

CLDepth2Space::CLDepth2Space(const std::shared_ptr<CLRuntime> runtime,
                             const PrecisionType &precision) :
    runtime_(runtime),
    precision_(precision), kernel_(nullptr), kernel_opt_(nullptr) {
    ENN_DBG_PRINT("CLDepth2Space is created");

}

Status CLDepth2Space::initialize(std::vector<std::shared_ptr<ITensor>> input_tensors,
                                 std::vector<std::shared_ptr<ITensor>> output_tensors,
                                 std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLDepth2Space::initialize() is called");

    input_ = std::static_pointer_cast<CLTensor>(input_tensors[INPUT_INDEX]);
    output_ = std::static_pointer_cast<CLTensor>(output_tensors[OUTPUT_INDEX]);

    parameters_ = std::static_pointer_cast<Depth2SpaceParameters>(parameters);
    ENN_DBG_PRINT("parameters_->block_size %d \n", parameters_->block_size);
    ENN_DBG_PRINT("parameters_->androidNN %d \n", parameters_->androidNN);
    ENN_DBG_PRINT("parameters_->isNCHW %d \n", parameters_->isNCHW);

    Dim4 input_dim = input_->getDim();
    Dim4 output_dim = output_->getDim();

    if (parameters_->androidNN) {
        if (!parameters_->isNCHW) {
            input_dim = convertDimToNCHW(input_dim);
            input_nchw_ = std::make_shared<CLTensor>(runtime_,
                                                     precision_,
                                                     input_->getDataType(),
                                                     input_dim,
                                                     input_->getDataOrder(),
                                                     input_->getScale(),
                                                     input_->getZeroPoint());
            output_dim.n = input_dim.n;
            output_dim.c = input_dim.c / parameters_->block_size / parameters_->block_size;
            output_dim.h = input_dim.h * parameters_->block_size;
            output_dim.w = input_dim.w * parameters_->block_size;
            output_nchw_ = std::make_shared<CLTensor>(runtime_,
                                                      precision_,
                                                      output_->getDataType(),
                                                      output_dim,
                                                      output_->getDataOrder(),
                                                      output_->getScale(),
                                                      output_->getZeroPoint());
            output_dim = convertDimToNHWC(output_dim);
        } else {
            output_dim.n = input_dim.n;
            output_dim.c = input_dim.c / parameters_->block_size / parameters_->block_size;
            output_dim.h = input_dim.h * parameters_->block_size;
            output_dim.w = input_dim.w * parameters_->block_size;
        }

        if (!isDimsSame(output_dim, output_->getDim())) {
            output_->reconfigureDimAndBuffer(output_dim);
        }
    }

    if (precision_ == PrecisionType::INT8) {
        return runtime_->setKernel(&kernel_, "SIGNEDdepth_to_space", precision_);
    }

    const uint32_t output_batch = output_->getDim().n;
    if (precision_ != PrecisionType::INT8 && parameters_->block_size == 2 && output_batch == 1) {
        Status state = runtime_->setKernel(&kernel_opt_, "depth_to_space_opt_vload", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
    } else {
        Status state = runtime_->setKernel(&kernel_, "depth_to_space", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
    }

    return Status::SUCCESS;
}

Status CLDepth2Space::execute() {
    ENN_DBG_PRINT("CLDepth2Space::execute() is called");
    auto input_tensor = std::static_pointer_cast<CLTensor>(input_);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output_);

    if (parameters_->androidNN && !parameters_->isNCHW) {
        Status status = input_->convertToNCHW(input_nchw_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "convertToNCHW failure\n");

        status = eval(input_nchw_, output_nchw_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "depth2space failure\n");
        return output_nchw_->convertToNHWC(output_tensor);
    } else {
        return eval(input_tensor, output_tensor);
    }
}
Status CLDepth2Space::eval(const std::shared_ptr<CLTensor> input_tensor,
                               std::shared_ptr<CLTensor> output_tensor) {
    auto in_data = input_tensor->getDataPtr();
    auto out_data = output_tensor->getDataPtr();

    auto input_height = input_tensor->getDim().h;
    auto input_width = input_tensor->getDim().w;
    auto input_channel = input_tensor->getDim().c;

    auto output_batch = output_tensor->getDim().n;
    auto output_height = output_tensor->getDim().h;
    auto output_width = output_tensor->getDim().w;
    auto output_channel = output_tensor->getDim().c;

    if (precision_ != PrecisionType::INT8 && parameters_->block_size == 2 && output_batch == 1) {
        if (kernel_opt_ == nullptr) {
            Status state = runtime_->setKernel(&kernel_opt_, "depth_to_space_opt_vload", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
        }
        size_t global[3] = {0, 0, 0};
        global[2] = input_channel;
        global[1] = input_height;
        global[0] = (input_width - 1) / 8 + 1;
        size_t local[3] = {16, 1, 1};
        global[0] = alignTo(global[0], local[0]);

        Status state = runtime_->setKernelArg(kernel_opt_.get(),
                                              in_data,
                                              out_data,
                                              parameters_->block_size,
                                              output_batch,
                                              output_channel,
                                              output_height,
                                              output_width,
                                              input_width);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");
        state = runtime_->enqueueKernel(kernel_opt_.get(), (cl_uint)3, global, local);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
    } else {
        if (kernel_ == nullptr) {
            Status state = runtime_->setKernel(&kernel_, "depth_to_space", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
        }
        size_t global[3] = {0, 0, 0};
        global[0] = output_channel;
        global[1] = output_height;
        global[2] = output_width;
        size_t local[3] = {1, 1, 1};

        Status state = runtime_->setKernelArg(kernel_.get(),
                                              in_data,
                                              out_data,
                                              parameters_->block_size,
                                              output_batch,
                                              input_channel,
                                              input_height,
                                              input_width);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");
        state = runtime_->enqueueKernel(kernel_.get(), (cl_uint)3, global, local);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
    }
    return Status::SUCCESS;
}

Status CLDepth2Space::release() {
    ENN_DBG_PRINT("CLDepth2Space::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
