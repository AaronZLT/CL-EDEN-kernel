#pragma once

#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/common/operator_interfaces/common/ActivationInfo.hpp"
#include "userdriver/common/operator_interfaces/common/Common.hpp"

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
                            int32_t n_output);

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
                                         std::shared_ptr<CLTensor> output_result);

Status updateCellAndOutput(const std::shared_ptr<CLRuntime> runtime,
                           const PrecisionType &precision,
                           const std::shared_ptr<CLTensor> input_gate_scratch,
                           const std::shared_ptr<CLTensor> forget_gate_scratch,
                           const std::shared_ptr<CLTensor> cell_scratch,
                           const std::shared_ptr<CLTensor> output_gate_scratch,
                           const std::shared_ptr<CLTensor> cell_state,
                           const std::shared_ptr<CLTensor> output,
                           const std::shared_ptr<CLTensor> output_state,
                           int32_t v_size);

Status vectorBatchVectorAssign(const std::shared_ptr<CLRuntime> runtime,
                               const PrecisionType &precision,
                               const std::shared_ptr<CLTensor> vector,
                               int32_t v_size,
                               int32_t n_batch,
                               std::shared_ptr<CLTensor> batch_vector);

Status matrixBatchVectorMultiplyAccumulate(const std::shared_ptr<CLRuntime> runtime,
                                           const PrecisionType &precision,
                                           const std::shared_ptr<CLTensor> matrix,
                                           int32_t m_rows,
                                           int32_t m_cols,
                                           const std::shared_ptr<CLTensor> vector,
                                           int32_t n_batch,
                                           std::shared_ptr<CLTensor> result,
                                           int32_t result_stride);

Status applyActivationToVector(const std::shared_ptr<CLRuntime> runtime,
                               const PrecisionType &precision,
                               const std::shared_ptr<CLTensor> vector,
                               int32_t v_size,
                               std::shared_ptr<CLTensor> result,
                               const ActivationInfo &activate_info);

Status vectorBatchVectorCwiseProductAccumulate(const std::shared_ptr<CLRuntime> runtime,
                                               const PrecisionType &precision,
                                               const std::shared_ptr<CLTensor> vector,
                                               int v_size,
                                               const std::shared_ptr<CLTensor> batch_vector,
                                               int n_batch,
                                               std::shared_ptr<CLTensor> result);

Status vectorVectorCwiseProduct(const std::shared_ptr<CLRuntime> runtime,
                                const PrecisionType &precision,
                                const std::shared_ptr<CLTensor> vector1,
                                const std::shared_ptr<CLTensor> vector2,
                                int v_size,
                                std::shared_ptr<CLTensor> result);

Status sub1Vector(const std::shared_ptr<CLRuntime> runtime,
                  const PrecisionType &precision,
                  const std::shared_ptr<CLTensor> vector,
                  int v_size,
                  std::shared_ptr<CLTensor> result);

Status vectorVectorCwiseProductAccumulate(const std::shared_ptr<CLRuntime> runtime,
                                          const PrecisionType &precision,
                                          const std::shared_ptr<CLTensor> vector1,
                                          const std::shared_ptr<CLTensor> vector2,
                                          int v_size,
                                          std::shared_ptr<CLTensor> result);

Status clipVector(const std::shared_ptr<CLRuntime> runtime,
                  const PrecisionType &precision,
                  const std::shared_ptr<CLTensor> vector,
                  int v_size,
                  float abs_limit,
                  std::shared_ptr<CLTensor> result);

Status zeroVector(const std::shared_ptr<CLRuntime> runtime,
                  const PrecisionType &precision,
                  std::shared_ptr<CLTensor> vector,
                  int v_size);

Status copyVector(const std::shared_ptr<CLRuntime> runtime,
                  const PrecisionType &precision,
                  const std::shared_ptr<CLTensor> vector,
                  int v_size,
                  std::shared_ptr<CLTensor> result);

Status meanStddevNormalization(const std::shared_ptr<CLRuntime> runtime,
                               const PrecisionType &precision,
                               const std::shared_ptr<CLTensor> input_vector,
                               std::shared_ptr<CLTensor> output_vector,
                               int v_size,
                               int n_batch,
                               float normalization_epsilon);

Status vectorBatchVectorCwiseProduct(const std::shared_ptr<CLRuntime> runtime,
                                     const PrecisionType &precision,
                                     const std::shared_ptr<CLTensor> vector,
                                     int v_size,
                                     const std::shared_ptr<CLTensor> batch_vector,
                                     int n_batch,
                                     std::shared_ptr<CLTensor> result);

Status vectorBatchVectorAdd(const std::shared_ptr<CLRuntime> runtime,
                            const PrecisionType &precision,
                            const std::shared_ptr<CLTensor> vector,
                            int v_size,
                            int n_batch,
                            std::shared_ptr<CLTensor> batch_vector);

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
                    bool use_aux_input);

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
                    const ActivationInfo &activate_info);

Status convertTimeMajor(std::shared_ptr<CLRuntime> runtime,
                        PrecisionType precision,
                        std::shared_ptr<CLTensor> in,
                        std::shared_ptr<CLTensor> out);

}  // namespace gpu
}  // namespace ud
}  // namespace enn
