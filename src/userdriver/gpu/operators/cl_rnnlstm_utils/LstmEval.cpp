#include "userdriver/gpu/operators/cl_rnnlstm_utils/LstmEval.hpp"
#include "userdriver/gpu/operators/cl_rnnlstm_utils/RNNLSTMUtils.hpp"

namespace enn {
namespace ud {
namespace gpu {

void lstmStepNoAuxInputNoCifgNoPeepholeNoProjNoCellClip(const std::shared_ptr<CLRuntime> runtime,
                                                        const PrecisionType &precision,
                                                        const std::shared_ptr<CLTensor> input_ptr_batch,
                                                        const std::shared_ptr<CLTensor> input_output_concat,
                                                        const std::shared_ptr<CLTensor> input_recurrent_to_4gate_weights,
                                                        const std::shared_ptr<CLTensor> gate4_bias,
                                                        const float proj_clip,
                                                        int n_batch,
                                                        int n_cell,
                                                        int n_input,
                                                        int n_output,
                                                        int output_batch_leading_dim,
                                                        std::shared_ptr<CLTensor> output_state_ptr,
                                                        std::shared_ptr<CLTensor> cell_state_ptr,
                                                        std::shared_ptr<CLTensor> input_gate_scratch,
                                                        std::shared_ptr<CLTensor> forget_gate_scratch,
                                                        std::shared_ptr<CLTensor> cell_scratch,
                                                        std::shared_ptr<CLTensor> output_gate_scratch,
                                                        std::shared_ptr<CLTensor> output_ptr_batch,
                                                        int32_t out_offset) {
    // optimization
    // concat input and output
    concatInputAndOutput(
        runtime, precision, input_ptr_batch, output_state_ptr, input_output_concat, n_batch, n_input, n_output);

    // Multiply
    matrixBatchVectorMultiplyWithBias(runtime,
                                      precision,
                                      input_output_concat,
                                      input_recurrent_to_4gate_weights,
                                      gate4_bias,
                                      4 * n_cell,
                                      n_input + n_output,
                                      n_batch,
                                      input_gate_scratch,
                                      forget_gate_scratch,
                                      cell_scratch,
                                      output_gate_scratch);

    // Update the cell state and output
    output_ptr_batch->setOffset(out_offset);
    updateCellAndOutput(runtime,
                        precision,
                        input_gate_scratch,
                        forget_gate_scratch,
                        cell_scratch,
                        output_gate_scratch,
                        cell_state_ptr,
                        output_ptr_batch,
                        output_state_ptr,
                        n_batch * n_cell);
}

void lstmStepWithAuxInput(const std::shared_ptr<CLRuntime> runtime,
                          const PrecisionType &precision,
                          const std::shared_ptr<CLTensor> input_ptr_batch,
                          const std::shared_ptr<CLTensor> input_to_input_weights_ptr,
                          const std::shared_ptr<CLTensor> input_to_forget_weights_ptr,
                          const std::shared_ptr<CLTensor> input_to_cell_weights_ptr,
                          const std::shared_ptr<CLTensor> input_to_output_weights_ptr,
                          const std::shared_ptr<CLTensor> aux_input_ptr_batch,
                          const std::shared_ptr<CLTensor> aux_input_to_input_weights_ptr,
                          const std::shared_ptr<CLTensor> aux_input_to_forget_weights_ptr,
                          const std::shared_ptr<CLTensor> aux_input_to_cell_weights_ptr,
                          const std::shared_ptr<CLTensor> aux_input_to_output_weights_ptr,
                          const std::shared_ptr<CLTensor> recurrent_to_input_weights_ptr,
                          const std::shared_ptr<CLTensor> recurrent_to_forget_weights_ptr,
                          const std::shared_ptr<CLTensor> recurrent_to_cell_weights_ptr,
                          const std::shared_ptr<CLTensor> recurrent_to_output_weights_ptr,
                          const std::shared_ptr<CLTensor> cell_to_input_weights_ptr,
                          const std::shared_ptr<CLTensor> cell_to_forget_weights_ptr,
                          const std::shared_ptr<CLTensor> cell_to_output_weights_ptr,
                          const std::shared_ptr<CLTensor> input_gate_bias_ptr,
                          const std::shared_ptr<CLTensor> forget_gate_bias_ptr,
                          const std::shared_ptr<CLTensor> cell_bias_ptr,
                          const std::shared_ptr<CLTensor> output_gate_bias_ptr,
                          const std::shared_ptr<CLTensor> projection_weights_ptr,
                          const std::shared_ptr<CLTensor> projection_bias_ptr,
                          const ActivationInfo activate_info,
                          const float cell_clip,
                          const float proj_clip,
                          int n_batch,
                          int n_cell,
                          int n_input,
                          int n_aux_input,
                          int n_output,
                          int output_batch_leading_dim,
                          std::shared_ptr<CLTensor> output_state_ptr,
                          std::shared_ptr<CLTensor> cell_state_ptr,
                          std::shared_ptr<CLTensor> input_gate_scratch,
                          std::shared_ptr<CLTensor> forget_gate_scratch,
                          std::shared_ptr<CLTensor> cell_scratch,
                          std::shared_ptr<CLTensor> output_gate_scratch,
                          std::shared_ptr<CLTensor> output_ptr_batch,
                          int32_t out_offset,
                          std::shared_ptr<CLTensor> output_state_out_ptr = nullptr,
                          std::shared_ptr<CLTensor> cell_state_out_ptr = nullptr) {
    if (output_state_out_ptr == nullptr) {
        output_state_out_ptr = output_state_ptr;
        cell_state_out_ptr = cell_state_ptr;
    }

    ActivationInfo act_sigmoid(ActivationInfo::ActivationType::SIGMOID, true);

    const bool use_cifg = (input_to_input_weights_ptr == nullptr);
    const bool use_peephole = (cell_to_output_weights_ptr != nullptr);

    // Initialize scratch buffers with bias.
    if (!use_cifg) {
        vectorBatchVectorAssign(runtime, precision, input_gate_bias_ptr, n_cell, n_batch, input_gate_scratch);
    }
    vectorBatchVectorAssign(runtime, precision, forget_gate_bias_ptr, n_cell, n_batch, forget_gate_scratch);
    vectorBatchVectorAssign(runtime, precision, cell_bias_ptr, n_cell, n_batch, cell_scratch);
    vectorBatchVectorAssign(runtime, precision, output_gate_bias_ptr, n_cell, n_batch, output_gate_scratch);

    // For each batch and cell: compute input_weight * input.
    if (!use_cifg) {
        matrixBatchVectorMultiplyAccumulate(runtime,
                                            precision,
                                            input_to_input_weights_ptr,
                                            n_cell,
                                            n_input,
                                            input_ptr_batch,
                                            n_batch,
                                            input_gate_scratch,
                                            1);
    }
    matrixBatchVectorMultiplyAccumulate(
        runtime, precision, input_to_forget_weights_ptr, n_cell, n_input, input_ptr_batch, n_batch, forget_gate_scratch, 1);
    matrixBatchVectorMultiplyAccumulate(
        runtime, precision, input_to_cell_weights_ptr, n_cell, n_input, input_ptr_batch, n_batch, cell_scratch, 1);
    matrixBatchVectorMultiplyAccumulate(
        runtime, precision, input_to_output_weights_ptr, n_cell, n_input, input_ptr_batch, n_batch, output_gate_scratch, 1);

    // If auxiliary input is available then compute aux_input_weight * aux_input
    if (aux_input_ptr_batch != nullptr) {
        if (!use_cifg) {
            matrixBatchVectorMultiplyAccumulate(runtime,
                                                precision,
                                                aux_input_to_input_weights_ptr,
                                                n_cell,
                                                n_aux_input,
                                                aux_input_ptr_batch,
                                                n_batch,
                                                input_gate_scratch,
                                                1);
        }
        matrixBatchVectorMultiplyAccumulate(runtime,
                                            precision,
                                            aux_input_to_forget_weights_ptr,
                                            n_cell,
                                            n_aux_input,
                                            aux_input_ptr_batch,
                                            n_batch,
                                            forget_gate_scratch,
                                            1);
        matrixBatchVectorMultiplyAccumulate(runtime,
                                            precision,
                                            aux_input_to_cell_weights_ptr,
                                            n_cell,
                                            n_aux_input,
                                            aux_input_ptr_batch,
                                            n_batch,
                                            cell_scratch,
                                            1);
        matrixBatchVectorMultiplyAccumulate(runtime,
                                            precision,
                                            aux_input_to_output_weights_ptr,
                                            n_cell,
                                            n_aux_input,
                                            aux_input_ptr_batch,
                                            n_batch,
                                            output_gate_scratch,
                                            1);
    }

    // For each batch and cell: compute recurrent_weight * output_state.
    if (!use_cifg) {
        matrixBatchVectorMultiplyAccumulate(runtime,
                                            precision,
                                            recurrent_to_input_weights_ptr,
                                            n_cell,
                                            n_output,
                                            output_state_ptr,
                                            n_batch,
                                            input_gate_scratch,
                                            1);
    }
    matrixBatchVectorMultiplyAccumulate(runtime,
                                        precision,
                                        recurrent_to_forget_weights_ptr,
                                        n_cell,
                                        n_output,
                                        output_state_ptr,
                                        n_batch,
                                        forget_gate_scratch,
                                        1);
    matrixBatchVectorMultiplyAccumulate(
        runtime, precision, recurrent_to_cell_weights_ptr, n_cell, n_output, output_state_ptr, n_batch, cell_scratch, 1);

    matrixBatchVectorMultiplyAccumulate(runtime,
                                        precision,
                                        recurrent_to_output_weights_ptr,
                                        n_cell,
                                        n_output,
                                        output_state_ptr,
                                        n_batch,
                                        output_gate_scratch,
                                        1);

    // For each batch and cell: update input gate.
    if (!use_cifg) {
        if (use_peephole) {
            vectorBatchVectorCwiseProductAccumulate(
                runtime, precision, cell_to_input_weights_ptr, n_cell, cell_state_ptr, n_batch, input_gate_scratch);
        }
        applyActivationToVector(runtime, precision, input_gate_scratch, n_cell * n_batch, input_gate_scratch, act_sigmoid);
    }

    // For each batch and cell: update forget gate.
    if (use_peephole) {
        vectorBatchVectorCwiseProductAccumulate(
            runtime, precision, cell_to_forget_weights_ptr, n_cell, cell_state_ptr, n_batch, forget_gate_scratch);
    }
    applyActivationToVector(runtime, precision, forget_gate_scratch, n_cell * n_batch, forget_gate_scratch, act_sigmoid);

    // For each batch and cell: update the cell.
    vectorVectorCwiseProduct(runtime, precision, forget_gate_scratch, cell_state_ptr, n_batch * n_cell, cell_state_out_ptr);
    applyActivationToVector(runtime, precision, cell_scratch, n_batch * n_cell, cell_scratch, activate_info);
    if (use_cifg) {
        sub1Vector(runtime, precision, forget_gate_scratch, n_batch * n_cell, forget_gate_scratch);
        vectorVectorCwiseProductAccumulate(
            runtime, precision, cell_scratch, forget_gate_scratch, n_batch * n_cell, cell_state_out_ptr);
    } else {
        vectorVectorCwiseProductAccumulate(
            runtime, precision, cell_scratch, input_gate_scratch, n_batch * n_cell, cell_state_out_ptr);
    }
    if (cell_clip > 0.000000000001f) {
        clipVector(runtime, precision, cell_state_out_ptr, n_batch * n_cell, cell_clip, cell_state_out_ptr);
    }

    // For each batch and cell: update the output gate.
    if (use_peephole) {
        vectorBatchVectorCwiseProductAccumulate(
            runtime, precision, cell_to_output_weights_ptr, n_cell, cell_state_out_ptr, n_batch, output_gate_scratch);
    }
    applyActivationToVector(runtime, precision, output_gate_scratch, n_batch * n_cell, output_gate_scratch, act_sigmoid);
    applyActivationToVector(runtime, precision, cell_state_out_ptr, n_batch * n_cell, cell_scratch, activate_info);
    vectorVectorCwiseProduct(runtime, precision, output_gate_scratch, cell_scratch, n_batch * n_cell, output_gate_scratch);

    const bool use_projection_weight = (projection_weights_ptr != nullptr);
    const bool use_projection_bias = (projection_bias_ptr != nullptr);

    // For each batch: update the projection and output_state. Note that since
    // the output batch rows may not be contiguous (output_batch_leading_dim !=
    // n_output), we unroll the batched operations where this is the case.
    if (output_batch_leading_dim == n_output) {
        if (use_projection_weight) {
            if (use_projection_bias) {
                vectorBatchVectorAssign(runtime, precision, projection_bias_ptr, n_output, n_batch, output_ptr_batch);
            } else {
                zeroVector(runtime, precision, output_ptr_batch, n_batch * n_output);
            }
            matrixBatchVectorMultiplyAccumulate(runtime,
                                                precision,
                                                projection_weights_ptr,
                                                n_output,
                                                n_cell,
                                                output_gate_scratch,
                                                n_batch,
                                                output_ptr_batch,
                                                1);
            if (proj_clip > 0.00000000001f) {
                clipVector(runtime, precision, output_ptr_batch, n_batch * n_output, proj_clip, output_ptr_batch);
            }
        } else {
            copyVector(runtime, precision, output_gate_scratch, n_batch * n_output, output_ptr_batch);
        }
        copyVector(runtime, precision, output_ptr_batch, n_batch * n_output, output_state_out_ptr);
    } else {
        if (use_projection_weight) {
            if (use_projection_bias) {
                for (int k = 0; k < n_batch; k++) {
                    output_ptr_batch->setOffset(out_offset + k * output_batch_leading_dim);
                    copyVector(runtime, precision, projection_bias_ptr, n_output, output_ptr_batch);
                }
            } else {
                for (int k = 0; k < n_batch; k++) {
                    output_ptr_batch->setOffset(out_offset + k * output_batch_leading_dim);
                    zeroVector(runtime, precision, output_ptr_batch, n_output);
                }
            }
            for (int k = 0; k < n_batch; k++) {
                output_ptr_batch->setOffset(out_offset + k * output_batch_leading_dim);
                output_gate_scratch->setOffset(k * n_cell);
                matrixBatchVectorMultiplyAccumulate(runtime,
                                                    precision,
                                                    projection_weights_ptr,
                                                    n_output,
                                                    n_cell,
                                                    output_gate_scratch,
                                                    1,
                                                    output_ptr_batch,
                                                    1);
                if (proj_clip > 0.000000000001f) {
                    output_ptr_batch->setOffset(out_offset + k * output_batch_leading_dim);
                    clipVector(runtime, precision, output_ptr_batch, n_output, proj_clip, output_ptr_batch);
                }
            }
        } else {
            for (int k = 0; k < n_batch; k++) {
                output_ptr_batch->setOffset(out_offset + k * output_batch_leading_dim);
                output_gate_scratch->setOffset(k * n_output);
                copyVector(runtime, precision, output_gate_scratch, n_output, output_ptr_batch);
            }
        }
        for (int k = 0; k < n_batch; k++) {
            output_ptr_batch->setOffset(out_offset + k * output_batch_leading_dim);
            output_state_ptr->setOffset(k * n_output);
            copyVector(runtime, precision, output_ptr_batch, n_output, output_state_out_ptr);
        }
    }
}

void layerNormLstmStep(const std::shared_ptr<CLRuntime> runtime,
                       const PrecisionType &precision,
                       const std::shared_ptr<CLTensor> input_ptr_batch,
                       const std::shared_ptr<CLTensor> input_to_input_weights_ptr,
                       const std::shared_ptr<CLTensor> input_to_forget_weights_ptr,
                       const std::shared_ptr<CLTensor> input_to_cell_weights_ptr,
                       const std::shared_ptr<CLTensor> input_to_output_weights_ptr,
                       const std::shared_ptr<CLTensor> recurrent_to_input_weights_ptr,
                       const std::shared_ptr<CLTensor> recurrent_to_forget_weights_ptr,
                       const std::shared_ptr<CLTensor> recurrent_to_cell_weights_ptr,
                       const std::shared_ptr<CLTensor> recurrent_to_output_weights_ptr,
                       const std::shared_ptr<CLTensor> cell_to_input_weights_ptr,
                       const std::shared_ptr<CLTensor> cell_to_forget_weights_ptr,
                       const std::shared_ptr<CLTensor> cell_to_output_weights_ptr,
                       const std::shared_ptr<CLTensor> input_layer_norm_weight_ptr,
                       const std::shared_ptr<CLTensor> forget_layer_norm_weight_ptr,
                       const std::shared_ptr<CLTensor> cell_layer_norm_weight_ptr,
                       const std::shared_ptr<CLTensor> output_layer_norm_weight_ptr,
                       const std::shared_ptr<CLTensor> input_gate_bias_ptr,
                       const std::shared_ptr<CLTensor> forget_gate_bias_ptr,
                       const std::shared_ptr<CLTensor> cell_bias_ptr,
                       const std::shared_ptr<CLTensor> output_gate_bias_ptr,
                       const std::shared_ptr<CLTensor> projection_weights_ptr,
                       const std::shared_ptr<CLTensor> projection_bias_ptr,
                       float cell_clip,
                       float proj_clip,
                       float layer_norm_epsilon,
                       const ActivationInfo activate_info,
                       int n_batch,
                       int n_cell,
                       int n_input,
                       int n_output,
                       std::shared_ptr<CLTensor> output_state_ptr,
                       std::shared_ptr<CLTensor> cell_state_ptr,
                       std::shared_ptr<CLTensor> input_gate_scratch,
                       std::shared_ptr<CLTensor> forget_gate_scratch,
                       std::shared_ptr<CLTensor> cell_scratch,
                       std::shared_ptr<CLTensor> output_gate_scratch,
                       std::shared_ptr<CLTensor> output_ptr_batch,
                       std::shared_ptr<CLTensor> output_state_out_ptr = nullptr,
                       std::shared_ptr<CLTensor> cell_state_out_ptr = nullptr) {
    if (output_state_out_ptr == nullptr) {
        output_state_out_ptr = output_state_ptr;
        cell_state_out_ptr = cell_state_ptr;
    }

    ActivationInfo act_sigmoid(ActivationInfo::ActivationType::SIGMOID, true);
    const bool use_cifg = (input_to_input_weights_ptr == nullptr);
    const bool use_peephole = (cell_to_output_weights_ptr != nullptr);

    // Initialize scratch buffers with 0.
    if (!use_cifg) {
        zeroVector(runtime, precision, input_gate_scratch, n_cell * n_batch);
    }
    zeroVector(runtime, precision, forget_gate_scratch, n_cell * n_batch);
    zeroVector(runtime, precision, cell_scratch, n_cell * n_batch);
    zeroVector(runtime, precision, output_gate_scratch, n_cell * n_batch);

    // For each batch and cell: compute input_weight * input.
    if (!use_cifg) {
        matrixBatchVectorMultiplyAccumulate(runtime,
                                            precision,
                                            input_to_input_weights_ptr,
                                            n_cell,
                                            n_input,
                                            input_ptr_batch,
                                            n_batch,
                                            input_gate_scratch,
                                            /*result_stride=*/1);
    }

    matrixBatchVectorMultiplyAccumulate(runtime,
                                        precision,
                                        input_to_forget_weights_ptr,
                                        n_cell,
                                        n_input,
                                        input_ptr_batch,
                                        n_batch,
                                        forget_gate_scratch,
                                        /*result_stride=*/1);
    matrixBatchVectorMultiplyAccumulate(runtime,
                                        precision,
                                        input_to_cell_weights_ptr,
                                        n_cell,
                                        n_input,
                                        input_ptr_batch,
                                        n_batch,
                                        cell_scratch,
                                        /*result_stride=*/1);
    matrixBatchVectorMultiplyAccumulate(runtime,
                                        precision,
                                        input_to_output_weights_ptr,
                                        n_cell,
                                        n_input,
                                        input_ptr_batch,
                                        n_batch,
                                        output_gate_scratch,
                                        /*result_stride=*/1);

    // For each batch and cell: compute recurrent_weight * output_state.
    if (!use_cifg) {
        matrixBatchVectorMultiplyAccumulate(runtime,
                                            precision,
                                            recurrent_to_input_weights_ptr,
                                            n_cell,
                                            n_output,
                                            output_state_ptr,
                                            n_batch,
                                            input_gate_scratch,
                                            /*result_stride=*/1);
    }
    matrixBatchVectorMultiplyAccumulate(runtime,
                                        precision,
                                        recurrent_to_forget_weights_ptr,
                                        n_cell,
                                        n_output,
                                        output_state_ptr,
                                        n_batch,
                                        forget_gate_scratch,
                                        /*result_stride=*/1);
    matrixBatchVectorMultiplyAccumulate(runtime,
                                        precision,
                                        recurrent_to_cell_weights_ptr,
                                        n_cell,
                                        n_output,
                                        output_state_ptr,
                                        n_batch,
                                        cell_scratch,
                                        /*result_stride=*/1);
    matrixBatchVectorMultiplyAccumulate(runtime,
                                        precision,
                                        recurrent_to_output_weights_ptr,
                                        n_cell,
                                        n_output,
                                        output_state_ptr,
                                        n_batch,
                                        output_gate_scratch,
                                        /*result_stride=*/1);

    // For each batch and cell: update input gate.
    if (!use_cifg) {
        if (use_peephole) {
            vectorBatchVectorCwiseProductAccumulate(
                runtime, precision, cell_to_input_weights_ptr, n_cell, cell_state_ptr, n_batch, input_gate_scratch);
        }
        meanStddevNormalization(
            runtime, precision, input_gate_scratch, input_gate_scratch, n_cell, n_batch, layer_norm_epsilon);
        vectorBatchVectorCwiseProduct(
            runtime, precision, input_layer_norm_weight_ptr, n_cell, input_gate_scratch, n_batch, input_gate_scratch);
        vectorBatchVectorAdd(runtime, precision, input_gate_bias_ptr, n_cell, n_batch, input_gate_scratch);
        applyActivationToVector(runtime, precision, input_gate_scratch, n_cell * n_batch, input_gate_scratch, act_sigmoid);
    }

    // For each batch and cell: update forget gate.
    if (use_peephole) {
        vectorBatchVectorCwiseProductAccumulate(
            runtime, precision, cell_to_forget_weights_ptr, n_cell, cell_state_ptr, n_batch, forget_gate_scratch);
    }
    meanStddevNormalization(
        runtime, precision, forget_gate_scratch, forget_gate_scratch, n_cell, n_batch, layer_norm_epsilon);
    vectorBatchVectorCwiseProduct(
        runtime, precision, forget_layer_norm_weight_ptr, n_cell, forget_gate_scratch, n_batch, forget_gate_scratch);
    vectorBatchVectorAdd(runtime, precision, forget_gate_bias_ptr, n_cell, n_batch, forget_gate_scratch);
    applyActivationToVector(runtime, precision, forget_gate_scratch, n_cell * n_batch, forget_gate_scratch, act_sigmoid);

    // For each batch and cell: update the cell.
    meanStddevNormalization(runtime, precision, cell_scratch, cell_scratch, n_cell, n_batch, layer_norm_epsilon);
    vectorBatchVectorCwiseProduct(
        runtime, precision, cell_layer_norm_weight_ptr, n_cell, cell_scratch, n_batch, cell_scratch);
    vectorBatchVectorAdd(runtime, precision, cell_bias_ptr, n_cell, n_batch, cell_scratch);
    vectorVectorCwiseProduct(runtime, precision, forget_gate_scratch, cell_state_ptr, n_batch * n_cell, cell_state_out_ptr);
    applyActivationToVector(runtime, precision, cell_scratch, n_batch * n_cell, cell_scratch, activate_info);
    if (use_cifg) {
        sub1Vector(runtime, precision, forget_gate_scratch, n_batch * n_cell, forget_gate_scratch);
        vectorVectorCwiseProductAccumulate(
            runtime, precision, cell_scratch, forget_gate_scratch, n_batch * n_cell, cell_state_out_ptr);
    } else {
        vectorVectorCwiseProductAccumulate(
            runtime, precision, cell_scratch, input_gate_scratch, n_batch * n_cell, cell_state_out_ptr);
    }
    if (cell_clip > 0.00000000001f) {
        clipVector(runtime, precision, cell_state_out_ptr, n_batch * n_cell, cell_clip, cell_state_out_ptr);
    }

    // For each batch and cell: update the output gate.
    if (use_peephole) {
        vectorBatchVectorCwiseProductAccumulate(
            runtime, precision, cell_to_output_weights_ptr, n_cell, cell_state_out_ptr, n_batch, output_gate_scratch);
    }
    meanStddevNormalization(
        runtime, precision, output_gate_scratch, output_gate_scratch, n_cell, n_batch, layer_norm_epsilon);
    vectorBatchVectorCwiseProduct(
        runtime, precision, output_layer_norm_weight_ptr, n_cell, output_gate_scratch, n_batch, output_gate_scratch);
    vectorBatchVectorAdd(runtime, precision, output_gate_bias_ptr, n_cell, n_batch, output_gate_scratch);
    applyActivationToVector(runtime, precision, output_gate_scratch, n_cell * n_batch, output_gate_scratch, act_sigmoid);
    applyActivationToVector(runtime, precision, cell_state_out_ptr, n_batch * n_cell, cell_scratch, activate_info);
    vectorVectorCwiseProduct(runtime, precision, output_gate_scratch, cell_scratch, n_batch * n_cell, output_gate_scratch);

    // For each batch: update the projection and output_state.
    const bool use_projection_weight = (projection_weights_ptr != nullptr);
    const bool use_projection_bias = (projection_bias_ptr != nullptr);
    if (use_projection_weight) {
        if (use_projection_bias) {
            vectorBatchVectorAssign(runtime, precision, projection_bias_ptr, n_output, n_batch, output_ptr_batch);
        } else {
            zeroVector(runtime, precision, output_ptr_batch, n_batch * n_output);
        }
        matrixBatchVectorMultiplyAccumulate(runtime,
                                            precision,
                                            projection_weights_ptr,
                                            n_output,
                                            n_cell,
                                            output_gate_scratch,
                                            n_batch,
                                            output_ptr_batch,
                                            /*result_stride=*/1);
        if (proj_clip > 0.000000000001f) {
            clipVector(runtime, precision, output_ptr_batch, n_batch * n_output, proj_clip, output_ptr_batch);
        }
    } else {
        copyVector(runtime, precision, output_gate_scratch, n_batch * n_output, output_ptr_batch);
    }
    copyVector(runtime, precision, output_ptr_batch, n_batch * n_output, output_state_out_ptr);
}

Status evalFloat(const std::shared_ptr<CLRuntime> runtime,
                 const PrecisionType &precision,
                 const std::shared_ptr<CLTensor> input,
                 const std::shared_ptr<CLTensor> input_to_input_weights,
                 const std::shared_ptr<CLTensor> input_to_forget_weights,
                 const std::shared_ptr<CLTensor> input_to_cell_weights,
                 const std::shared_ptr<CLTensor> input_to_output_weights,
                 const std::shared_ptr<CLTensor> recurrent_to_input_weights,
                 const std::shared_ptr<CLTensor> recurrent_to_forget_weights,
                 const std::shared_ptr<CLTensor> recurrent_to_cell_weights,
                 const std::shared_ptr<CLTensor> recurrent_to_output_weights,
                 const std::shared_ptr<CLTensor> cell_to_input_weights,
                 const std::shared_ptr<CLTensor> cell_to_forget_weights,
                 const std::shared_ptr<CLTensor> cell_to_output_weights,
                 const std::shared_ptr<CLTensor> aux_input,
                 const std::shared_ptr<CLTensor> aux_input_to_input_weights,
                 const std::shared_ptr<CLTensor> aux_input_to_forget_weights,
                 const std::shared_ptr<CLTensor> aux_input_to_cell_weights,
                 const std::shared_ptr<CLTensor> aux_input_to_output_weights,
                 const std::shared_ptr<CLTensor> input_gate_bias,
                 const std::shared_ptr<CLTensor> forget_gate_bias,
                 const std::shared_ptr<CLTensor> cell_bias,
                 const std::shared_ptr<CLTensor> output_gate_bias,
                 const std::shared_ptr<CLTensor> projection_weights,
                 const std::shared_ptr<CLTensor> projection_bias,
                 const int max_time,
                 const int n_batch,
                 const int n_input,
                 const int aux_input_size,
                 const int n_cell,
                 const int n_output,
                 const int output_batch_leading_dim,
                 const ActivationInfo activate_info,
                 const float cell_clip,
                 const float proj_clip,
                 bool forward_sequence,
                 uint32_t output_offset,
                 std::shared_ptr<CLTensor> input_gate_scratch,
                 std::shared_ptr<CLTensor> forget_gate_scratch,
                 std::shared_ptr<CLTensor> cell_gate_scratch,
                 std::shared_ptr<CLTensor> output_gate_scratch,
                 std::shared_ptr<CLTensor> activation_state,
                 std::shared_ptr<CLTensor> cell_state,
                 std::shared_ptr<CLTensor> output,
                 bool use_layer_norm,
                 std::shared_ptr<CLTensor> output_state_out,
                 std::shared_ptr<CLTensor> cell_state_out) {
    // Loop through the sequence
    const int input_step = n_batch * n_input;
    const int output_step = n_batch * output_batch_leading_dim;

    for (int t = 0; t < max_time; t++) {
        const int t_rel = forward_sequence ? t : max_time - t - 1;
        input->setOffset(t_rel * input_step);
        if (aux_input) {
            aux_input->setOffset(t_rel * input_step);
        }
        int32_t out_offset = t_rel * output_step + output_offset;
        output->setOffset(out_offset);

        output_gate_scratch->setOffset(0);
        activation_state->setOffset(0);

        if (use_layer_norm == false) {
            lstmStepWithAuxInput(runtime,
                                 precision,
                                 input,
                                 input_to_input_weights,
                                 input_to_forget_weights,
                                 input_to_cell_weights,
                                 input_to_output_weights,
                                 aux_input,
                                 aux_input_to_input_weights,
                                 aux_input_to_forget_weights,
                                 aux_input_to_cell_weights,
                                 aux_input_to_output_weights,
                                 recurrent_to_input_weights,
                                 recurrent_to_forget_weights,
                                 recurrent_to_cell_weights,
                                 recurrent_to_output_weights,
                                 cell_to_input_weights,
                                 cell_to_forget_weights,
                                 cell_to_output_weights,
                                 input_gate_bias,
                                 forget_gate_bias,
                                 cell_bias,
                                 output_gate_bias,
                                 projection_weights,
                                 projection_bias,
                                 activate_info,
                                 cell_clip,
                                 proj_clip,
                                 n_batch,
                                 n_cell,
                                 n_input,
                                 aux_input_size,
                                 n_output,
                                 output_batch_leading_dim,
                                 activation_state,
                                 cell_state,
                                 input_gate_scratch,
                                 forget_gate_scratch,
                                 cell_gate_scratch,
                                 output_gate_scratch,
                                 output,
                                 out_offset,
                                 output_state_out,
                                 cell_state_out);

            if (output_state_out != nullptr) {
                copyVector(runtime, precision, output_state_out, n_batch * n_output, activation_state);
                copyVector(runtime, precision, cell_state_out, n_batch * n_cell, cell_state);
            }
        } else {
            float layer_norm_epsilon = 1e-8;
            layerNormLstmStep(runtime,
                              precision,
                              input,
                              input_to_input_weights,
                              input_to_forget_weights,
                              input_to_cell_weights,
                              input_to_output_weights,
                              recurrent_to_input_weights,
                              recurrent_to_forget_weights,
                              recurrent_to_cell_weights,
                              recurrent_to_output_weights,
                              cell_to_input_weights,
                              cell_to_forget_weights,
                              cell_to_output_weights,
                              aux_input_to_input_weights,
                              aux_input_to_forget_weights,
                              aux_input_to_cell_weights,
                              aux_input_to_output_weights,
                              input_gate_bias,
                              forget_gate_bias,
                              cell_bias,
                              output_gate_bias,
                              projection_weights,
                              projection_bias,
                              cell_clip,
                              proj_clip,
                              layer_norm_epsilon,
                              activate_info,
                              n_batch,
                              n_cell,
                              n_input,
                              n_output,
                              activation_state,
                              cell_state,
                              input_gate_scratch,
                              forget_gate_scratch,
                              cell_gate_scratch,
                              output_gate_scratch,
                              output);
        }
    }
    return Status::SUCCESS;
}

Status evalFloatOpt(const std::shared_ptr<CLRuntime> runtime,
                    const PrecisionType &precision,
                    const std::shared_ptr<CLTensor> input,
                    const std::shared_ptr<CLTensor> input_output_concat,
                    const std::shared_ptr<CLTensor> input_recurrent_to_4gate_weights,
                    const std::shared_ptr<CLTensor> gate4_bias,
                    const int max_time,
                    const int n_batch,
                    const int n_input,
                    const int n_cell,
                    const int n_output,
                    const int output_batch_leading_dim,
                    const float proj_clip,
                    bool forward_sequence,
                    uint32_t output_offset,
                    std::shared_ptr<CLTensor> input_gate_scratch,
                    std::shared_ptr<CLTensor> forget_gate_scratch,
                    std::shared_ptr<CLTensor> cell_gate_scratch,
                    std::shared_ptr<CLTensor> output_gate_scratch,
                    std::shared_ptr<CLTensor> activation_state,
                    std::shared_ptr<CLTensor> cell_state,
                    std::shared_ptr<CLTensor> output) {
    // Loop through the sequence
    const int input_step = n_batch * n_input;
    const int output_step = n_batch * output_batch_leading_dim;

    for (int t = 0; t < max_time; t++) {
        const int t_rel = forward_sequence ? t : max_time - t - 1;
        input->setOffset(t_rel * input_step);
        int32_t out_offset = t_rel * output_step + output_offset;
        output->setOffset(out_offset);

        output_gate_scratch->setOffset(0);
        activation_state->setOffset(0);

        lstmStepNoAuxInputNoCifgNoPeepholeNoProjNoCellClip(runtime,
                                                           precision,
                                                           input,
                                                           input_output_concat,
                                                           input_recurrent_to_4gate_weights,
                                                           gate4_bias,
                                                           proj_clip,
                                                           n_batch,
                                                           n_cell,
                                                           n_input,
                                                           n_output,
                                                           output_batch_leading_dim,
                                                           activation_state,
                                                           cell_state,
                                                           input_gate_scratch,
                                                           forget_gate_scratch,
                                                           cell_gate_scratch,
                                                           output_gate_scratch,
                                                           output,
                                                           out_offset);
    }
    return Status::SUCCESS;
}

Status layerNormLstmEvalFloat(const std::shared_ptr<CLRuntime> runtime,
                              const PrecisionType &precision,
                              const std::shared_ptr<CLTensor> input,
                              const std::shared_ptr<CLTensor> input_to_input_weights,
                              const std::shared_ptr<CLTensor> input_to_forget_weights,
                              const std::shared_ptr<CLTensor> input_to_cell_weights,
                              const std::shared_ptr<CLTensor> input_to_output_weights,
                              const std::shared_ptr<CLTensor> recurrent_to_input_weights,
                              const std::shared_ptr<CLTensor> recurrent_to_forget_weights,
                              const std::shared_ptr<CLTensor> recurrent_to_cell_weights,
                              const std::shared_ptr<CLTensor> recurrent_to_output_weights,
                              const std::shared_ptr<CLTensor> cell_to_input_weights,
                              const std::shared_ptr<CLTensor> cell_to_forget_weights,
                              const std::shared_ptr<CLTensor> cell_to_output_weights,
                              const std::shared_ptr<CLTensor> input_layer_norm_weights,
                              const std::shared_ptr<CLTensor> forget_layer_norm_weights,
                              const std::shared_ptr<CLTensor> cell_layer_norm_weights,
                              const std::shared_ptr<CLTensor> output_layer_norm_weights,
                              const std::shared_ptr<CLTensor> input_gate_bias,
                              const std::shared_ptr<CLTensor> forget_gate_bias,
                              const std::shared_ptr<CLTensor> cell_gate_bias,
                              const std::shared_ptr<CLTensor> output_gate_bias,
                              const std::shared_ptr<CLTensor> projection_weights,
                              const std::shared_ptr<CLTensor> projection_bias,
                              const int n_batch,
                              const int n_input,
                              const int n_cell,
                              const int n_output,
                              const ActivationInfo activate_info,
                              const float cell_clip,
                              const float proj_clip,
                              const float layer_norm_epsilon,
                              std::shared_ptr<CLTensor> input_gate_scratch_buffer,
                              std::shared_ptr<CLTensor> forget_gate_scratch_buffer,
                              std::shared_ptr<CLTensor> cell_gate_scratch_buffer,
                              std::shared_ptr<CLTensor> output_gate_scratch_buffer,
                              std::shared_ptr<CLTensor> activation_state,
                              std::shared_ptr<CLTensor> cell_state,
                              std::shared_ptr<CLTensor> output) {
    layerNormLstmStep(runtime,
                      precision,
                      input,
                      input_to_input_weights,
                      input_to_forget_weights,
                      input_to_cell_weights,
                      input_to_output_weights,
                      recurrent_to_input_weights,
                      recurrent_to_forget_weights,
                      recurrent_to_cell_weights,
                      recurrent_to_output_weights,
                      cell_to_input_weights,
                      cell_to_forget_weights,
                      cell_to_output_weights,
                      input_layer_norm_weights,
                      forget_layer_norm_weights,
                      cell_layer_norm_weights,
                      output_layer_norm_weights,
                      input_gate_bias,
                      forget_gate_bias,
                      cell_gate_bias,
                      output_gate_bias,
                      projection_weights,
                      projection_bias,
                      cell_clip,
                      proj_clip,
                      layer_norm_epsilon,
                      activate_info,
                      n_batch,
                      n_cell,
                      n_input,
                      n_output,
                      activation_state,
                      cell_state,
                      input_gate_scratch_buffer,
                      forget_gate_scratch_buffer,
                      cell_gate_scratch_buffer,
                      output_gate_scratch_buffer,
                      output);

    return Status::SUCCESS;
}

Status evalFloatLayerNorm(const std::shared_ptr<CLRuntime> runtime,
                          const PrecisionType &precision,
                          const std::shared_ptr<CLTensor> input,
                          const std::shared_ptr<CLTensor> input_to_input_weights,
                          const std::shared_ptr<CLTensor> input_to_forget_weights,
                          const std::shared_ptr<CLTensor> input_to_cell_weights,
                          const std::shared_ptr<CLTensor> input_to_output_weights,
                          const std::shared_ptr<CLTensor> recurrent_to_input_weights,
                          const std::shared_ptr<CLTensor> recurrent_to_forget_weights,
                          const std::shared_ptr<CLTensor> recurrent_to_cell_weights,
                          const std::shared_ptr<CLTensor> recurrent_to_output_weights,
                          const std::shared_ptr<CLTensor> cell_to_input_weights,
                          const std::shared_ptr<CLTensor> cell_to_forget_weights,
                          const std::shared_ptr<CLTensor> cell_to_output_weights,
                          const std::shared_ptr<CLTensor> input_layer_norm_weights,
                          const std::shared_ptr<CLTensor> forget_layer_norm_weights,
                          const std::shared_ptr<CLTensor> cell_layer_norm_weights,
                          const std::shared_ptr<CLTensor> output_layer_norm_weights,
                          const std::shared_ptr<CLTensor> input_gate_bias,
                          const std::shared_ptr<CLTensor> forget_gate_bias,
                          const std::shared_ptr<CLTensor> cell_bias,
                          const std::shared_ptr<CLTensor> output_gate_bias,
                          const std::shared_ptr<CLTensor> projection_weights,
                          const std::shared_ptr<CLTensor> projection_bias,
                          const int max_time,
                          const int n_batch,
                          const int n_input,
                          const int n_cell,
                          const int n_output,
                          const int output_batch_leading_dim,
                          const ActivationInfo activate_info,
                          const float cell_clip,
                          const float proj_clip,
                          const float layer_norm_epsilon,
                          bool forward_sequence,
                          uint32_t output_offset,
                          std::shared_ptr<CLTensor> input_gate_scratch,
                          std::shared_ptr<CLTensor> forget_gate_scratch,
                          std::shared_ptr<CLTensor> cell_gate_scratch,
                          std::shared_ptr<CLTensor> output_gate_scratch,
                          std::shared_ptr<CLTensor> activation_state,
                          std::shared_ptr<CLTensor> cell_state,
                          std::shared_ptr<CLTensor> output,
                          std::shared_ptr<CLTensor> output_state_out,
                          std::shared_ptr<CLTensor> cell_state_out) {
    // Loop through the sequence
    const int input_step = n_batch * n_input;
    const int output_step = n_batch * output_batch_leading_dim;

    for (int t = 0; t < max_time; t++) {
        const int t_rel = forward_sequence ? t : max_time - t - 1;
        input->setOffset(t_rel * input_step);
        int32_t out_offset = t_rel * output_step + output_offset;
        output->setOffset(out_offset);
        output_gate_scratch->setOffset(0);
        activation_state->setOffset(0);
        layerNormLstmStep(runtime,
                          precision,
                          input,
                          input_to_input_weights,
                          input_to_forget_weights,
                          input_to_cell_weights,
                          input_to_output_weights,
                          recurrent_to_input_weights,
                          recurrent_to_forget_weights,
                          recurrent_to_cell_weights,
                          recurrent_to_output_weights,
                          cell_to_input_weights,
                          cell_to_forget_weights,
                          cell_to_output_weights,
                          input_layer_norm_weights,
                          forget_layer_norm_weights,
                          cell_layer_norm_weights,
                          output_layer_norm_weights,
                          input_gate_bias,
                          forget_gate_bias,
                          cell_bias,
                          output_gate_bias,
                          projection_weights,
                          projection_bias,
                          cell_clip,
                          proj_clip,
                          layer_norm_epsilon,
                          activate_info,
                          n_batch,
                          n_cell,
                          n_input,
                          n_output,
                          activation_state,
                          cell_state,
                          input_gate_scratch,
                          forget_gate_scratch,
                          cell_gate_scratch,
                          output_gate_scratch,
                          output,
                          output_state_out,
                          cell_state_out);

        copyVector(runtime, precision, output_state_out, n_batch * n_output, activation_state);
        copyVector(runtime, precision, cell_state_out, n_batch * n_cell, cell_state);
    }
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
