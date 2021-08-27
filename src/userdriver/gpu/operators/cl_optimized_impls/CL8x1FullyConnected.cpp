#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CL8x1FullyConnected.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLMemAlign.hpp"

namespace enn {
namespace ud {
namespace gpu {

CL8x1FullyConnected::CL8x1FullyConnected(const std::shared_ptr<CLRuntime> runtime,
                                         const PrecisionType &precision,
                                         const Dim4 &input_dim,
                                         const Dim4 &output_dim,
                                         const std::shared_ptr<ITensor> weight,
                                         const std::shared_ptr<ITensor> bias,
                                         bool weights_as_input) {
    runtime_ = runtime;
    precision_ = precision;
    weight_ = std::static_pointer_cast<CLTensor>(weight);
    bias_ = std::static_pointer_cast<CLTensor>(bias);
    weights_input = weights_as_input;
    // set kernel
    Status state;
    state = runtime_->setKernel(&kernel_split_, "fc_splitopt_8x1", precision_);
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "fc_splitopt_8x1 setKernel failure\n");
    state = runtime_->setKernel(&kernel_merge_, "fc_mergeopt_8x1", precision_);
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "fc_mergeopt_8x1 setKernel failure\n");
    state = runtime_->setKernel(&kernel_interleave_, "fc_interleave_8x1", precision_);
    CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "fc_interleave_8x1 setKernel failure\n");
    split_size_ = 256;
    int input_count = alignTo(input_dim.c * input_dim.h * input_dim.w, split_size_);
    split_number_ = input_count / split_size_;
    uint32_t h = alignTo(static_cast<int>(input_dim.n), 8) / 8;
    uint32_t w = alignTo(static_cast<int>(input_dim.c * input_dim.h * input_dim.w), split_size_) * 8;
    Dim4 inter_dims = {1, 1, h, w};
    interleave_input_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, inter_dims);
    Dim4 split_dims = {input_dim.n, static_cast<uint32_t>(split_number_), weight_->getDim().n, 1};
    split_output_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, split_dims);

    if (false == weights_input) {
        state = align_weight(input_dim.c);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "mem_align failure\n");
    }
}

Status CL8x1FullyConnected::align_weight(int input_dim_c) {
    int src_total_count = weight_->getDim().n * input_dim_c * weight_->getDim().h * weight_->getDim().w;
    int src_uint_count = src_total_count / weight_->getDim().n;
    int dst_align_count = alignTo(src_uint_count, split_size_);
    int dst_total_count = dst_align_count * weight_->getDim().n;

    Dim4 weight_buffer_dim = {static_cast<uint32_t>(dst_total_count), 1, 1, 1};
    weight_buffer_ = std::make_shared<CLTensor>(runtime_, precision_, DataType::FLOAT, weight_buffer_dim);

    auto mem_align = std::make_shared<CLMemAlign>(runtime_);
    return mem_align->execute(weight_->getDataPtr(),
                              weight_buffer_->getDataPtr(),
                              precision_,
                              src_total_count,
                              src_uint_count,
                              dst_total_count,
                              dst_align_count,
                              1);
}

Status CL8x1FullyConnected::execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) {
    auto input_tensor = std::static_pointer_cast<CLTensor>(input);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output);
    auto input_data = input_tensor->getDataPtr();
    auto output_data = output_tensor->getDataPtr();
    auto bias_data = bias_->getDataPtr();

    uint32_t input_batch = input_tensor->getDim().n;
    uint32_t input_channel = input_tensor->getDim().c;
    uint32_t input_height = input_tensor->getDim().h;
    uint32_t input_width = input_tensor->getDim().w;
    uint32_t outputChannel = weight_->getDim().n;
    // interleave
    size_t global_interleave[2] = {0}, local_interleave[2] = {1, 16};
    size_t global_split[3] = {0}, local_split[3] = {1, 1, 24};
    size_t global_merge[2] = {0}, local_merge[2] = {1, 24};
    int input_pic_size = input_channel * input_height * input_width;
    global_interleave[0] = interleave_input_->getDim().h;
    global_interleave[1] = alignTo(input_pic_size, local_interleave[1]);
    Status state = runtime_->setKernelArg(kernel_interleave_.get(),
                                          input_data,
                                          interleave_input_->getDataPtr(),
                                          input_batch,
                                          input_pic_size,
                                          interleave_input_->getDim().w);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
    state = runtime_->enqueueKernel(kernel_interleave_.get(), (cl_uint)2, global_interleave, local_interleave);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");

    if (true == weights_input) {
        state = align_weight(input_tensor->getDim().c);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "mem_align failure\n");
    }

    // Split
    int align_batch = alignTo(input_batch, 8 * local_split[0]);
    global_split[0] = align_batch / 8;
    global_split[1] = alignTo(split_number_, local_split[1]);
    global_split[2] = alignTo(outputChannel, local_split[2]);
    int split_picture_step = alignTo(input_channel * input_height * input_width, split_size_);
    state = runtime_->setKernelArg(kernel_split_.get(),
                                   interleave_input_->getDataPtr(),
                                   weight_buffer_->getDataPtr(),
                                   split_output_->getDataPtr(),
                                   split_picture_step,
                                   split_size_,
                                   input_batch,
                                   outputChannel,
                                   split_number_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
    state = runtime_->enqueueKernel(kernel_split_.get(), (cl_uint)3, global_split, local_split);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
    // Merge
    global_merge[0] = alignTo(input_batch, local_merge[0]);
    global_merge[1] = alignTo(outputChannel, local_merge[1]);
    int merge_picture_step = split_number_ * outputChannel;
    state = runtime_->setKernelArg(kernel_merge_.get(),
                                   split_output_->getDataPtr(),
                                   bias_data,
                                   output_data,
                                   merge_picture_step,
                                   split_number_,
                                   input_batch,
                                   outputChannel);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
    state = runtime_->enqueueKernel(kernel_merge_.get(), (cl_uint)2, global_merge, local_merge);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
    return Status::SUCCESS;
}

Status CL8x1FullyConnected::release() { return Status::SUCCESS; }

}  // namespace gpu
}  // namespace ud
}  // namespace enn
