#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CL8X1GEMVFullyConnected.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLMemAlign.hpp"

namespace enn {
namespace ud {
namespace gpu {

CL8X1GEMVFullyConnected::CL8X1GEMVFullyConnected(const std::shared_ptr<CLRuntime> runtime,
                                                 const PrecisionType &precision,
                                                 const Dim4 &input_dim,
                                                 const Dim4 &output_dim,
                                                 const std::shared_ptr<ITensor> weight,
                                                 const std::shared_ptr<ITensor> bias,
                                                 bool weights_as_input) {
    ENN_DBG_PRINT("CL8X1GEMVFullyConnected is called");
    runtime_ = runtime;
    precision_ = precision;
    weight_ = std::static_pointer_cast<CLTensor>(weight);
    bias_ = std::static_pointer_cast<CLTensor>(bias);
    weights_input = weights_as_input;
    // set kernel
    Status state = runtime_->setKernel(&kernel_interleave_, "fc_interleave_8x1_gemv", precision_);
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "fc_interleave_8x1_gemv setKernel failure\n");
    state = runtime_->setKernel(&kernel_gemv_, "fc_8x1_gemv", precision_);
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "fc_8x1_gemv setKernel failure\n");

    interleave_dim_ = {
        1, 1, (uint32_t)(alignTo(input_dim.n, 8) / 8), (uint32_t)(alignTo(input_dim.c * input_dim.h * input_dim.w, 8) * 8)};
    uint32_t interleave_size = interleave_dim_.h * interleave_dim_.w;

    Dim4 interleave_buffer_dim = {interleave_size, 1, 1, 1};
    interleave_buffer_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, interleave_buffer_dim);

    if (false == weights_input) {
        state = align_weight(weight_->getDim().n);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "mem_align failure\n");
    }
    ENN_DBG_PRINT("CL8X1GEMVFullyConnected is created");
}

Status CL8X1GEMVFullyConnected::align_weight(int output_dim_c) {
    Dim4 weight_dim = weight_->getDim();
    int src_count = weight_dim.n * weight_dim.c * weight_dim.h * weight_dim.w;
    int src_unit_count = src_count / output_dim_c;
    int dst_align_count = alignTo(src_unit_count, 8);
    uint32_t dst_total_count = dst_align_count * output_dim_c;

    Dim4 weight_buffer_dim = {dst_total_count, 1, 1, 1};
    weight_buffer_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, weight_buffer_dim);

    auto mem_align = std::make_shared<CLMemAlign>(runtime_);
    return mem_align->execute(weight_->getDataPtr(),
                              weight_buffer_->getDataPtr(),
                              precision_,
                              src_count,
                              src_unit_count,
                              dst_total_count,
                              dst_align_count,
                              1);
}

Status CL8X1GEMVFullyConnected::execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) {
    ENN_DBG_PRINT("CL8X1GEMVFullyConnected::execute() is called");
    auto input_tensor = std::static_pointer_cast<CLTensor>(input);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output);
    auto weight_tensor = std::static_pointer_cast<CLTensor>(weight_);
    auto bias_tensor = std::static_pointer_cast<CLTensor>(bias_);
    auto input_data = input_tensor->getDataPtr();
    auto output_data = output_tensor->getDataPtr();
    auto bias_data = bias_tensor->getDataPtr();
    auto input_dim = input_tensor->getDim();
    auto output_dim = output_tensor->getDim();

    if (true == weights_input) {
        Status state = align_weight(output_dim.c);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "mem_align failure\n");
    }

    // interleave
    size_t global_interleave[2] = {0, 0};
    size_t local_interleave[2] = {1, 16};
    int input_pic_size = input_dim.c * input_dim.h * input_dim.w;
    global_interleave[0] = interleave_dim_.h;
    global_interleave[1] = alignTo(input_pic_size, local_interleave[1]);

    Status state = runtime_->setKernelArg(kernel_interleave_.get(),
                                          input_data,
                                          interleave_buffer_->getDataPtr(),
                                          input_dim.n,
                                          input_pic_size,
                                          interleave_dim_.w);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "kernel_interleave_ setKernelArg failure\n");
    state = runtime_->enqueueKernel(kernel_interleave_.get(), 2, global_interleave, local_interleave);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel_interleave_ kernel failure\n");

    // gemv
    size_t global_gemv[2] = {0, 0};
    size_t local_gemv[2] = {1, 24};
    int align_batch = alignTo(input_dim.n, 8 * local_gemv[0]);
    global_gemv[0] = align_batch / 8;
    global_gemv[1] = alignTo(weight_->getDim().n, local_gemv[1]);
    // pictrueStep 8 multiply
    int pic_step = alignTo(input_dim.c * input_dim.h * input_dim.w, 8);
    state = runtime_->setKernelArg(kernel_gemv_.get(),
                                   interleave_buffer_->getDataPtr(),
                                   weight_buffer_->getDataPtr(),
                                   bias_data,
                                   output_data,
                                   pic_step,
                                   input_dim.n,
                                   weight_->getDim().n);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "kernel_gemv_ setKernelArg failure\n");
    state = runtime_->enqueueKernel(kernel_gemv_.get(), 2, global_gemv, local_gemv);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel_gemv_ kernel failure\n");

    return Status::SUCCESS;
}

Status CL8X1GEMVFullyConnected::release() { return Status::SUCCESS; }

}  // namespace gpu
}  // namespace ud
}  // namespace enn
