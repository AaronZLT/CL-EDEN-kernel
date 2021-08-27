#include "userdriver/gpu/operators/cl_rnnlstm_utils/RNNLSTMUtils.hpp"
#include "userdriver/gpu/operators/CLTranspose.hpp"

namespace enn {
namespace ud {
namespace gpu {

Status concatInputAndOutput(const std::shared_ptr<CLRuntime> runtime,
                            const PrecisionType &precision,
                            const std::shared_ptr<CLTensor> input_vector,
                            const std::shared_ptr<CLTensor> output_vector,
                            const std::shared_ptr<CLTensor> concat_vector,
                            int32_t n_batch,
                            int32_t n_input,
                            int32_t n_output) {
    Status state = Status::FAILURE;

    auto src_input = input_vector->getDataPtr();
    auto src_output = output_vector->getDataPtr();
    auto dst_concat = concat_vector->getDataPtr();

    auto src_input_offset = input_vector->getOffset();
    auto src_output_offset = output_vector->getOffset();
    auto dst_concat_offset = concat_vector->getOffset();

    std::shared_ptr<struct _cl_kernel> kernel_;
    if (n_input % 8 == 0 && n_output % 8 == 0) {
        state = runtime->setKernel(&kernel_, "concat_input_output_opt", precision);
        CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernel failure\n");
        int n_concat = n_input + n_output;
        size_t local[2] = {static_cast<size_t>(1), static_cast<size_t>(32)};
        size_t global[2] = {static_cast<size_t>(n_batch), static_cast<size_t>(alignTo(n_concat / 8, local[1]))};

        state = runtime->setKernelArg(kernel_.get(),
                                      src_input,
                                      src_output,
                                      dst_concat,
                                      src_input_offset,
                                      src_output_offset,
                                      dst_concat_offset,
                                      n_input,
                                      n_output);
        CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernelArg failure\n");

        state = runtime->enqueueKernel(kernel_.get(), (cl_uint)2, global, local);
        CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "execute kernel failure\n");

    } else {
        state = runtime->setKernel(&kernel_, "concat_input_output", precision);
        CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernel failure\n");
        size_t local[1] = {static_cast<size_t>(1)};
        size_t global[1] = {static_cast<size_t>(n_batch)};

        state = runtime->setKernelArg(kernel_.get(),
                                      src_input,
                                      src_output,
                                      dst_concat,
                                      src_input_offset,
                                      src_output_offset,
                                      dst_concat_offset,
                                      n_input,
                                      n_output);
        CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernelArg failure\n");

        state = runtime->enqueueKernel(kernel_.get(), (cl_uint)1, global, local);
        CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "execute kernel failure\n");
    }

    return state;
}

Status matrixBatchVectorMultiplyWithBias(const std::shared_ptr<CLRuntime> runtime,
                                         const PrecisionType &precision,
                                         const std::shared_ptr<CLTensor> input_vector,
                                         const std::shared_ptr<CLTensor> matrix,
                                         const std::shared_ptr<CLTensor> bias,
                                         int32_t m_rows,
                                         int32_t m_cols,
                                         int32_t n_batch,
                                         std::shared_ptr<CLTensor> input_result,
                                         std::shared_ptr<CLTensor> forget_result,
                                         std::shared_ptr<CLTensor> cell_result,
                                         std::shared_ptr<CLTensor> output_result) {
    Status state = Status::FAILURE;
    auto input_data = input_vector->getDataPtr();
    auto matrix_data = matrix->getDataPtr();
    auto input_result_data = input_result->getDataPtr();
    auto forget_result_data = forget_result->getDataPtr();
    auto cell_result_data = cell_result->getDataPtr();
    auto output_result_data = output_result->getDataPtr();

    auto input_offset = input_vector->getOffset();
    auto result_offset = input_result->getOffset();

    std::shared_ptr<struct _cl_kernel> kernel_;

    // matrix has been arranged from row * col to col / 16 * (row * 16), but the parameter
    // "m_rows" and "m_cols" are original.
    state = runtime->setKernel(&kernel_, "matrix_batch_vector_mul_with_bias", precision);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernel failure\n");
    size_t local[2] = {2, 32};
    size_t global[2] = {static_cast<size_t>(alignTo(n_batch, local[0])), static_cast<size_t>(alignTo(m_rows, local[1]))};

    state = runtime->setKernelArg(kernel_.get(),
                                  input_data,
                                  matrix->getDataPtr(),
                                  bias->getDataPtr(),
                                  m_rows,
                                  m_cols,
                                  n_batch,
                                  input_result_data,
                                  forget_result_data,
                                  cell_result_data,
                                  output_result_data,
                                  input_offset,
                                  result_offset);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernelArg failure\n");

    state = runtime->enqueueKernel(kernel_.get(), (cl_uint)2, global, local);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "execute kernel failure\n");

    return state;
}

Status updateCellAndOutput(const std::shared_ptr<CLRuntime> runtime,
                           const PrecisionType &precision,
                           const std::shared_ptr<CLTensor> input_gate_scratch,
                           const std::shared_ptr<CLTensor> forget_gate_scratch,
                           const std::shared_ptr<CLTensor> cell_scratch,
                           const std::shared_ptr<CLTensor> output_gate_scratch,
                           const std::shared_ptr<CLTensor> cell_state,
                           const std::shared_ptr<CLTensor> output,
                           const std::shared_ptr<CLTensor> output_state,
                           int32_t v_size) {
    Status state = Status::FAILURE;
    auto input_gate_data = input_gate_scratch->getDataPtr();
    auto forget_gate_data = forget_gate_scratch->getDataPtr();
    auto cell_data = cell_scratch->getDataPtr();
    auto output_gate_data = output_gate_scratch->getDataPtr();
    auto cell_sate_data = cell_state->getDataPtr();
    auto output_data = output->getDataPtr();
    auto output_state_data = output_state->getDataPtr();

    auto output_offset = output->getOffset();

    std::shared_ptr<struct _cl_kernel> kernel_;
    state = runtime->setKernel(&kernel_, "update_cell_output", precision);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernel failure\n");
    size_t local[1]{32};
    size_t global[1] = {static_cast<size_t>(alignTo(ceil(v_size / 8.0f), local[0]))};

    state = runtime->setKernelArg(kernel_.get(),
                                  input_gate_data,
                                  forget_gate_data,
                                  cell_data,
                                  output_gate_data,
                                  cell_sate_data,
                                  output_data,
                                  output_state_data,
                                  v_size,
                                  output_offset);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernelArg failure\n");

    state = runtime->enqueueKernel(kernel_.get(), (cl_uint)1, global, local);

    return state;
}

Status vectorBatchVectorAssign(const std::shared_ptr<CLRuntime> runtime,
                               const PrecisionType &precision,
                               const std::shared_ptr<CLTensor> vector,
                               int32_t v_size,
                               int32_t n_batch,
                               std::shared_ptr<CLTensor> batch_vector) {
    auto src = vector->getDataPtr();
    auto src_offset = vector->getOffset();
    auto dst = batch_vector->getDataPtr();
    auto dst_offset = batch_vector->getOffset();
    for (int32_t b = 0; b < n_batch; b++) {
        auto dst_batch_offset = b * v_size;
        size_t type_size = runtime->getRuntimeTypeBytes(precision);
        size_t src_offset_bytes = type_size * (size_t)src_offset;
        size_t dst_offset_bytes = type_size * (size_t)(dst_batch_offset + dst_offset);
        size_t size_bytes = type_size * v_size;

        Status state = runtime->copyBuffer(dst, src, dst_offset_bytes, src_offset_bytes, size_bytes);
        CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "copy buffer failure.\n");
    }
    return Status::SUCCESS;
}

Status matrixBatchVectorMultiplyAccumulate(const std::shared_ptr<CLRuntime> runtime,
                                           const PrecisionType &precision,
                                           const std::shared_ptr<CLTensor> matrix,
                                           int32_t m_rows,
                                           int32_t m_cols,
                                           const std::shared_ptr<CLTensor> vector,
                                           int32_t n_batch,
                                           std::shared_ptr<CLTensor> result,
                                           int32_t result_stride) {
    auto matrix_data = matrix->getDataPtr();
    auto vector_data = vector->getDataPtr();
    auto result_data = result->getDataPtr();

    auto vector_offset = vector->getOffset();
    auto result_offset = result->getOffset();

    Status state = Status::FAILURE;
    std::shared_ptr<struct _cl_kernel> kernel_;
    state = runtime->setKernel(&kernel_, "matrix_batch_vector_mul_acc", precision);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernel failure\n");
    size_t global[2] = {static_cast<size_t>(n_batch), static_cast<size_t>(m_rows)};

    state =
        runtime->setKernelArg(kernel_.get(), matrix_data, vector_data, vector_offset, m_cols, result_data, result_offset);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernelArg failure\n");

    state = runtime->enqueueKernel(kernel_.get(), (cl_uint)2, global, NULL);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "execute kernel failure\n");
    return Status::SUCCESS;
}

Status applyActivationToVector(const std::shared_ptr<CLRuntime> runtime,
                               const PrecisionType &precision,
                               const std::shared_ptr<CLTensor> vector,
                               int32_t v_size,
                               std::shared_ptr<CLTensor> result,
                               const ActivationInfo &activate_info) {
    auto in = vector->getDataPtr();
    auto out = result->getDataPtr();

    auto in_offset = vector->getOffset();
    auto out_offset = result->getOffset();

    Status state = Status::FAILURE;
    std::shared_ptr<struct _cl_kernel> kernel_;
    switch (activate_info.activation()) {
    case ActivationInfo::ActivationType::RELU: state = runtime->setKernel(&kernel_, "special_relu", precision); break;
    case ActivationInfo::ActivationType::SIGMOID: state = runtime->setKernel(&kernel_, "special_sigmoid", precision); break;
    case ActivationInfo::ActivationType::TANH: state = runtime->setKernel(&kernel_, "special_tanh", precision); break;
    case ActivationInfo::ActivationType::RELU1:
    case ActivationInfo::ActivationType::RELU6:
    case ActivationInfo::ActivationType::NONE: state = runtime->setKernel(&kernel_, "special_none", precision); break;
    default: ERROR_PRINT("Non-Support Activation Type");
    }
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernel failure\n");
    size_t global[1] = {static_cast<size_t>(v_size)};
    state = runtime->setKernelArg(kernel_.get(), in, out, in_offset, out_offset);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernelArg failure\n");

    state = runtime->enqueueKernel(kernel_.get(), (cl_uint)1, global, NULL);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "execute kernel failure\n");
    return Status::SUCCESS;
}

Status vectorBatchVectorCwiseProductAccumulate(const std::shared_ptr<CLRuntime> runtime,
                                               const PrecisionType &precision,
                                               const std::shared_ptr<CLTensor> vector,
                                               int v_size,
                                               const std::shared_ptr<CLTensor> batch_vector,
                                               int n_batch,
                                               std::shared_ptr<CLTensor> result) {
    auto vec = vector->getDataPtr();
    auto batch_vec = batch_vector->getDataPtr();
    auto res = result->getDataPtr();

    auto vec_offset = vector->getOffset();
    auto batch_vec_offset = batch_vector->getOffset();
    auto res_offset = result->getOffset();

    Status state = Status::FAILURE;
    std::shared_ptr<struct _cl_kernel> kernel_;
    state = runtime->setKernel(&kernel_, "batch_vector_product_acc", precision);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernel failure\n");
    size_t global[2] = {static_cast<size_t>(n_batch), static_cast<size_t>(v_size)};

    state = runtime->setKernelArg(kernel_.get(), vec, batch_vec, res, vec_offset, batch_vec_offset, res_offset);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernelArg failure\n");

    state = runtime->enqueueKernel(kernel_.get(), (cl_uint)2, global, NULL);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "execute kernel failure\n");
    return Status::SUCCESS;
}

Status vectorVectorCwiseProduct(const std::shared_ptr<CLRuntime> runtime,
                                const PrecisionType &precision,
                                const std::shared_ptr<CLTensor> vector1,
                                const std::shared_ptr<CLTensor> vector2,
                                int v_size,
                                std::shared_ptr<CLTensor> result) {
    auto v1 = vector1->getDataPtr();
    auto v2 = vector2->getDataPtr();
    auto res = result->getDataPtr();

    auto v1_offset = vector1->getOffset();
    auto v2_offset = vector2->getOffset();
    auto res_offset = result->getOffset();

    Status state = Status::FAILURE;
    std::shared_ptr<struct _cl_kernel> kernel_;
    state = runtime->setKernel(&kernel_, "vector_vector_product", precision);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernel failure\n");
    size_t global[1] = {static_cast<size_t>(v_size)};

    state = runtime->setKernelArg(kernel_.get(), v1, v2, res, v1_offset, v2_offset, res_offset);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernelArg failure\n");

    state = runtime->enqueueKernel(kernel_.get(), (cl_uint)1, global, NULL);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "execute kernel failure\n");
    return Status::SUCCESS;
}

Status sub1Vector(const std::shared_ptr<CLRuntime> runtime,
                  const PrecisionType &precision,
                  const std::shared_ptr<CLTensor> vector,
                  int v_size,
                  std::shared_ptr<CLTensor> result) {
    auto vec = vector->getDataPtr();
    auto vec_offset = vector->getOffset();

    Status state = Status::FAILURE;
    std::shared_ptr<struct _cl_kernel> kernel_;
    state = runtime->setKernel(&kernel_, "sub_1_vector", precision);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernel failure\n");
    size_t global[1] = {static_cast<size_t>(v_size)};

    state = runtime->setKernelArg(kernel_.get(), vec, vec_offset);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernelArg failure\n");

    state = runtime->enqueueKernel(kernel_.get(), (cl_uint)1, global, NULL);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "execute kernel failure\n");
    return Status::SUCCESS;
}

Status vectorVectorCwiseProductAccumulate(const std::shared_ptr<CLRuntime> runtime,
                                          const PrecisionType &precision,
                                          const std::shared_ptr<CLTensor> vector1,
                                          const std::shared_ptr<CLTensor> vector2,
                                          int v_size,
                                          std::shared_ptr<CLTensor> result) {
    auto v1 = vector1->getDataPtr();
    auto v2 = vector2->getDataPtr();
    auto res = result->getDataPtr();

    auto v1_offset = vector1->getOffset();
    auto v2_offset = vector2->getOffset();
    auto res_offset = result->getOffset();

    Status state = Status::FAILURE;
    std::shared_ptr<struct _cl_kernel> kernel_;
    state = runtime->setKernel(&kernel_, "vector_vector_product_accumulate", precision);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernel failure\n");
    size_t global[1] = {static_cast<size_t>(v_size)};

    state = runtime->setKernelArg(kernel_.get(), v1, v2, res, v1_offset, v2_offset, res_offset);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernelArg failure\n");

    state = runtime->enqueueKernel(kernel_.get(), (cl_uint)1, global, NULL);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "execute kernel failure\n");
    return Status::SUCCESS;
}

Status clipVector(const std::shared_ptr<CLRuntime> runtime,
                  const PrecisionType &precision,
                  const std::shared_ptr<CLTensor> vector,
                  int v_size,
                  float abs_limit,
                  std::shared_ptr<CLTensor> result) {
    auto vec = vector->getDataPtr();
    auto res = result->getDataPtr();

    auto vec_offset = vector->getOffset();
    auto res_offset = result->getOffset();

    Status state = Status::FAILURE;
    std::shared_ptr<struct _cl_kernel> kernel_;
    state = runtime->setKernel(&kernel_, "clip_vector", precision);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernel failure\n");
    size_t global[1] = {static_cast<size_t>(v_size)};

    state = runtime->setKernelArg(kernel_.get(), vec, res, abs_limit, vec_offset, res_offset);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernelArg failure\n");

    state = runtime->enqueueKernel(kernel_.get(), (cl_uint)1, global, NULL);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "execute kernel failure\n");
    return Status::SUCCESS;
}

Status zeroVector(const std::shared_ptr<CLRuntime> runtime,
                  const PrecisionType &precision,
                  std::shared_ptr<CLTensor> vector,
                  int v_size) {
    auto vec = vector->getDataPtr();
    auto vec_offset = vector->getOffset();

    Status state = Status::FAILURE;
    std::shared_ptr<struct _cl_kernel> kernel_;
    state = runtime->setKernel(&kernel_, "zero_vector", precision);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernel failure\n");
    size_t global[1] = {static_cast<size_t>(v_size)};

    state = runtime->setKernelArg(kernel_.get(), vec, vec_offset);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernelArg failure\n");

    state = runtime->enqueueKernel(kernel_.get(), (cl_uint)1, global, NULL);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "execute kernel failure\n");
    return Status::SUCCESS;
}

Status copyVector(const std::shared_ptr<CLRuntime> runtime,
                  const PrecisionType &precision,
                  const std::shared_ptr<CLTensor> vector,
                  int v_size,
                  std::shared_ptr<CLTensor> result) {
    auto vec = vector->getDataPtr();
    auto res = result->getDataPtr();

    auto vec_offset = vector->getOffset();
    auto res_offset = result->getOffset();

    Status state = Status::FAILURE;
    std::shared_ptr<struct _cl_kernel> kernel_;
    state = runtime->setKernel(&kernel_, "copy_vector", precision);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernel failure\n");
    size_t global[1] = {static_cast<size_t>(v_size)};

    state = runtime->setKernelArg(kernel_.get(), vec, res, vec_offset, res_offset);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernelArg failure\n");

    state = runtime->enqueueKernel(kernel_.get(), (cl_uint)1, global, NULL);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "execute kernel failure\n");

    return Status::SUCCESS;
}

Status meanStddevNormalization(const std::shared_ptr<CLRuntime> runtime,
                               const PrecisionType &precision,
                               const std::shared_ptr<CLTensor> input_vector,
                               std::shared_ptr<CLTensor> output_vector,
                               int v_size,
                               int n_batch,
                               float normalization_epsilon) {
    auto in = input_vector->getDataPtr();
    auto out = output_vector->getDataPtr();

    auto in_offset = input_vector->getOffset();
    auto out_offset = output_vector->getOffset();

    Status state = Status::FAILURE;
    std::shared_ptr<struct _cl_kernel> kernel_;
    state = runtime->setKernel(&kernel_, "mean_stddev_norm", precision);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernel failure\n");
    size_t global[2] = {static_cast<size_t>(n_batch)};

    state = runtime->setKernelArg(kernel_.get(), in, out, in_offset, out_offset, v_size, normalization_epsilon);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernelArg failure\n");

    state = runtime->enqueueKernel(kernel_.get(), (cl_uint)1, global, NULL);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "execute kernel failure\n");
    return Status::SUCCESS;
}

Status vectorBatchVectorCwiseProduct(const std::shared_ptr<CLRuntime> runtime,
                                     const PrecisionType &precision,
                                     const std::shared_ptr<CLTensor> vector,
                                     int v_size,
                                     const std::shared_ptr<CLTensor> batch_vector,
                                     int n_batch,
                                     std::shared_ptr<CLTensor> result) {
    auto vec = vector->getDataPtr();
    auto batch_vec = batch_vector->getDataPtr();
    auto res = result->getDataPtr();

    auto vec_offset = vector->getOffset();
    auto batch_vec_offset = batch_vector->getOffset();
    auto res_offset = result->getOffset();

    Status state = Status::FAILURE;
    std::shared_ptr<struct _cl_kernel> kernel_;
    state = runtime->setKernel(&kernel_, "batch_vector_product", precision);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernel failure\n");
    size_t global[2] = {static_cast<size_t>(n_batch), static_cast<size_t>(v_size)};

    state = runtime->setKernelArg(kernel_.get(), vec, batch_vec, res, vec_offset, batch_vec_offset, res_offset);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernelArg failure\n");

    state = runtime->enqueueKernel(kernel_.get(), (cl_uint)2, global, NULL);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "execute kernel failure\n");
    return Status::SUCCESS;
}

Status vectorBatchVectorAdd(const std::shared_ptr<CLRuntime> runtime,
                            const PrecisionType &precision,
                            const std::shared_ptr<CLTensor> vector,
                            int v_size,
                            int n_batch,
                            std::shared_ptr<CLTensor> batch_vector) {
    auto vec = vector->getDataPtr();
    auto batch_vec = batch_vector->getDataPtr();

    auto vec_offset = vector->getOffset();
    auto batch_vec_offset = batch_vector->getOffset();

    Status state = Status::FAILURE;
    std::shared_ptr<struct _cl_kernel> kernel_;
    state = runtime->setKernel(&kernel_, "vector_batch_vector_add", precision);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernel failure\n");
    size_t global[2] = {static_cast<size_t>(n_batch)};

    state = runtime->setKernelArg(kernel_.get(), vec, batch_vec, v_size, vec_offset, batch_vec_offset);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "setKernelArg failure\n");

    state = runtime->enqueueKernel(kernel_.get(), (cl_uint)1, global, NULL);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "execute kernel failure\n");
    return Status::SUCCESS;
}

Status rnnBatchStep(const std::shared_ptr<CLRuntime> runtime,
                    const PrecisionType &precision,
                    const std::shared_ptr<CLTensor> input_batch,
                    const std::shared_ptr<CLTensor> aux_input_batch,
                    const std::shared_ptr<CLTensor> hidden_state_input,
                    const std::shared_ptr<CLTensor> bias,
                    const std::shared_ptr<CLTensor> input_weight,
                    const std::shared_ptr<CLTensor> aux_input_weight,
                    const std::shared_ptr<CLTensor> recurrent_weights,
                    const ActivationInfo &activate_info,
                    int32_t input_size,
                    int32_t num_units,
                    int32_t batch_size,
                    std::shared_ptr<CLTensor> output_batch,
                    std::shared_ptr<CLTensor> hidden_state_output,
                    bool use_aux_input) {
    // Output = bias
    Status state = vectorBatchVectorAssign(runtime, precision, bias, num_units, batch_size, output_batch);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "vectorBatchVectorAssign failure\n");

    // Output += input * input_weights
    state = matrixBatchVectorMultiplyAccumulate(runtime,
                                                precision,
                                                input_weight,
                                                num_units,
                                                input_size,
                                                input_batch,
                                                batch_size,
                                                output_batch,
                                                /*result_stride=*/1);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "matrixBatchVectorMultiplyAccumulate failure\n");

    // Output += aux_input * aux_input_weights
    if (use_aux_input)
        state = matrixBatchVectorMultiplyAccumulate(runtime,
                                                    precision,
                                                    aux_input_weight,
                                                    num_units,
                                                    aux_input_batch->getDim().h,
                                                    aux_input_batch,
                                                    batch_size,
                                                    output_batch,
                                                    /*result_stride=*/1);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "matrixBatchVectorMultiplyAccumulate failure\n");

    // Output += recurrent_weights * hidden_state
    state = matrixBatchVectorMultiplyAccumulate(runtime,
                                                precision,
                                                recurrent_weights,
                                                num_units,
                                                num_units,
                                                hidden_state_input,
                                                batch_size,
                                                output_batch,
                                                /*result_stride=*/1);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "matrixBatchVectorMultiplyAccumulate failure\n");

    // Output = activation(Output)
    state = applyActivationToVector(runtime, precision, output_batch, num_units * batch_size, output_batch, activate_info);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "applyActivationToVector failure\n");

    // update hidden state
    state = vectorBatchVectorAssign(runtime, precision, output_batch, num_units, batch_size, hidden_state_output);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "vectorBatchVectorAssign failure\n");
    state = vectorBatchVectorAssign(runtime, precision, output_batch, num_units, batch_size, hidden_state_input);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "vectorBatchVectorAssign failure\n");
    return Status::SUCCESS;
}

Status rnnBatchStep(const std::shared_ptr<CLRuntime> runtime,
                    const PrecisionType &precision,
                    const std::shared_ptr<CLTensor> input_batch,
                    const std::shared_ptr<CLTensor> aux_input_batch,
                    const std::shared_ptr<CLTensor> input_weight,
                    const std::shared_ptr<CLTensor> aux_input_weight,
                    const std::shared_ptr<CLTensor> recurrent_weights,
                    const std::shared_ptr<CLTensor> bias,
                    int32_t input_size,
                    int32_t num_units,
                    int32_t batch_size,
                    bool use_aux_input,
                    std::shared_ptr<CLTensor> hidden_state_batch,
                    std::shared_ptr<CLTensor> output_batch,
                    const ActivationInfo &activate_info) {
    Status state = Status::SUCCESS;

    // Output = bias
    state = vectorBatchVectorAssign(runtime, precision, bias, num_units, batch_size, output_batch);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "vectorBatchVectorAssign failure\n");

    // Output += input * input_weights
    state = matrixBatchVectorMultiplyAccumulate(runtime,
                                                precision,
                                                input_weight,
                                                num_units,
                                                input_size,
                                                input_batch,
                                                batch_size,
                                                output_batch,
                                                /*result_stride=*/1);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "matrixBatchVectorMultiplyAccumulate failure\n");

    // Output += aux_input * aux_input_weights
    if (use_aux_input)
        state = matrixBatchVectorMultiplyAccumulate(runtime,
                                                    precision,
                                                    aux_input_weight,
                                                    num_units,
                                                    aux_input_batch->getDim().h,
                                                    aux_input_batch,
                                                    batch_size,
                                                    output_batch,
                                                    /*result_stride=*/1);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "matrixBatchVectorMultiplyAccumulate failure\n");

    // Output += recurrent_weights * hidden_state
    state = matrixBatchVectorMultiplyAccumulate(runtime,
                                                precision,
                                                recurrent_weights,
                                                num_units,
                                                num_units,
                                                hidden_state_batch,
                                                batch_size,
                                                output_batch,
                                                /*result_stride=*/1);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "matrixBatchVectorMultiplyAccumulate failure\n");

    // Output = activation(Output)
    state = applyActivationToVector(runtime, precision, output_batch, num_units * batch_size, output_batch, activate_info);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "applyActivationToVector failure\n");

    // update hidden_state
    state = vectorBatchVectorAssign(runtime, precision, output_batch, num_units, batch_size, hidden_state_batch);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "vectorBatchVectorAssign failure\n");
    return Status::SUCCESS;
}

Status convertTimeMajor(std::shared_ptr<CLRuntime> runtime,
                        PrecisionType precision,
                        std::shared_ptr<CLTensor> in,
                        std::shared_ptr<CLTensor> out) {
    Status state = Status::SUCCESS;
    auto transpose = std::make_shared<CLTranspose>(runtime, precision);

    std::vector<int32_t> perm = {1, 0, 2, 3};
    NDims perm_dim = {static_cast<uint32_t>(perm.size()), 1, 1, 1};
    auto perm_tensor =
        std::make_shared<CLTensor>(runtime, precision, perm.data(), perm_dim);

    auto parameters = std::make_shared<TransposeParameters>();
    parameters->androidNN = false;

    state = transpose->initialize({in, perm_tensor}, {out}, parameters);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "CLTranspose initialize failure\n");
    state = transpose->execute();
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "CLTranspose execute failure\n");
    transpose->release();
    CHECK_AND_RETURN_ERR(Status::SUCCESS != state, Status::FAILURE, "CLTranspose release failure\n");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
