#include "userdriver/gpu/common/CLKernels.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define MERGE_WEIGHTS(input_to_input_weights, input_to_forget_weights, input_to_cell_weights, input_to_output_weights, \
                      recurrent_to_input_weights, recurrent_to_forget_weights, recurrent_to_cell_weights, \
                      recurrent_to_output_weights, input_recurrent_to_4gate_weights, n_input, n_output, n_cell) \
        int g0 = get_global_id(0); \
        for (int i = 0; i < n_input; i++) { \
            input_recurrent_to_4gate_weights[g0 * (n_input + n_output) + i] = \
                input_to_input_weights[g0 * n_input + i]; \
            input_recurrent_to_4gate_weights[(g0 + n_cell) * (n_input + n_output) + i] = \
                input_to_forget_weights[g0 * n_input + i]; \
            input_recurrent_to_4gate_weights[(g0 + 2 * n_cell) * (n_input + n_output) + i] = \
                input_to_cell_weights[g0 * n_input + i]; \
            input_recurrent_to_4gate_weights[(g0 + 3 * n_cell) * (n_input + n_output) + i] = \
                input_to_output_weights[g0 * n_input + i]; \
        } \
        for (int i = 0; i < n_output; i++) { \
            input_recurrent_to_4gate_weights[g0 * (n_input + n_output) + i + n_input] = \
                recurrent_to_input_weights[g0 * n_output + i]; \
            input_recurrent_to_4gate_weights[(g0 + n_cell) * (n_input + n_output) + i + n_input] = \
                recurrent_to_forget_weights[g0 * n_output + i]; \
            input_recurrent_to_4gate_weights[(g0 + 2 * n_cell) * (n_input + n_output) + i + n_input] = \
                recurrent_to_cell_weights[g0 * n_output + i]; \
            input_recurrent_to_4gate_weights[(g0 + 3 * n_cell) * (n_input + n_output) + i + n_input] = \
                recurrent_to_output_weights[g0 * n_output + i]; \
        }

#define CONCAT_INPUT_OUTPUT_OPT(src_input, src_output, dst_concat, src_input_offset, src_output_offset, \
                                dst_concat_offset, n_input, n_output) \
        int g0 = get_global_id(0); \
        int g1 = get_global_id(1) * 8; \
        if (g1 < (n_input + n_output)) { \
            DATA_T8 in = (DATA_T8)0.0f; \
            if (g1 < n_input) { \
                in = vload8(0, src_input + g0 * n_input + g1 + src_input_offset); \
            } else { \
                in = vload8(0, src_output + g0 * n_output + (g1 - n_input) + src_output_offset); \
            } \
            vstore8(in, 0, dst_concat + g0 * (n_input + n_output) + g1 + dst_concat_offset); \
        }

#define CONCAT_INPUT_OUTPUT(src_input, src_output, dst_concat, src_input_offset, src_output_offset, dst_concat_offset, \
                            n_input, n_output) \
        int g0 = get_global_id(0); \
        DATA_T8 in = (DATA_T8)0.0f; \
        for (int i = 0; i < n_input / 8; i++) { \
            in = vload8(0, src_input + g0 * n_input + i * 8 + src_input_offset); \
            vstore8(in, 0, dst_concat + g0 * (n_input + n_output) + i * 8 + dst_concat_offset); \
        } \
        for (int i = n_input / 8 * 8; i < n_input; i++) { \
            dst_concat[g0 * (n_input + n_output) + i + dst_concat_offset] = \
                src_input[g0 * n_input + i + src_input_offset]; \
        } \
        for (int i = 0; i < n_output / 8; i++) { \
            in = vload8(0, src_output + g0 * n_output + i * 8 + src_output_offset); \
            vstore8(in, 0, dst_concat + g0 * (n_input + n_output) + i * 8 + dst_concat_offset + n_input); \
        } \
        for (int i = n_output / 8 * 8; i < n_output; i++) { \
            dst_concat[g0 * (n_input + n_output) + i + dst_concat_offset + n_input] = \
                src_output[g0 * n_output + i + src_output_offset]; \
        }

#define UPDATE_CELL_OUTPUT(input_gate_scratch, forget_gate_scratch, cell_scratch, output_gate_scratch, \
                           cell_state, output, output_state, v_size, output_offset) \
        int g0 = get_global_id(0) * 8; \
        if (g0 < v_size) { \
            DATA_T8 input_gate_data = vload8(0, input_gate_scratch + g0); \
            DATA_T8 forget_gate_data = vload8(0, forget_gate_scratch + g0); \
            DATA_T8 cell_data = vload8(0, cell_scratch + g0); \
            DATA_T8 output_gate_data = vload8(0, output_gate_scratch + g0); \
            DATA_T8 cell_sate_data = vload8(0, cell_state + g0); \
            cell_sate_data = cell_sate_data * forget_gate_data; \
            cell_data = cell_data * input_gate_data; \
            cell_sate_data += cell_data; \
            cell_data = tanh(cell_sate_data); \
            output_gate_data = output_gate_data * cell_data; \
            if (g0 + 7 < v_size) { \
                vstore8(cell_sate_data, 0, cell_state + g0); \
                vstore8(output_gate_data, 0, output_state + g0); \
                vstore8(output_gate_data, 0, output + g0 + output_offset); \
            } else if (g0 + 6 < v_size) { \
                vstore4(cell_sate_data.s0123, 0, cell_state + g0); \
                vstore3(cell_sate_data.s456, 0, cell_state + g0 + 4); \
                vstore4(output_gate_data.s0123, 0, output_state + g0); \
                vstore3(output_gate_data.s456, 0, output_state + g0 + 4); \
                vstore4(output_gate_data.s0123, 0, output + g0 + output_offset); \
                vstore3(output_gate_data.s456, 0, output + g0 + output_offset + 4); \
            } else if (g0 + 5 < v_size) { \
                vstore4(cell_sate_data.s0123, 0, cell_state + g0); \
                vstore2(cell_sate_data.s45, 0, cell_state + g0 + 4); \
                vstore4(output_gate_data.s0123, 0, output_state + g0); \
                vstore2(output_gate_data.s45, 0, output_state + g0 + 4); \
                vstore4(output_gate_data.s0123, 0, output + g0 + output_offset); \
                vstore2(output_gate_data.s45, 0, output + g0 + output_offset + 4); \
            } else if (g0 + 4 < v_size) { \
                vstore4(cell_sate_data.s0123, 0, cell_state + g0); \
                cell_state[g0 + 4] = cell_sate_data.s4; \
                vstore4(output_gate_data.s0123, 0, output_state + g0); \
                output_state[g0 + 4] = output_gate_data.s4; \
                vstore4(output_gate_data.s0123, 0, output + g0 + output_offset); \
                output[g0 + output_offset + 4] = output_gate_data.s4; \
            } else if (g0 + 3 < v_size) { \
                vstore4(cell_sate_data.s0123, 0, cell_state + g0); \
                vstore4(output_gate_data.s0123, 0, output_state + g0); \
                vstore4(output_gate_data.s0123, 0, output + g0 + output_offset); \
            } else if (g0 + 2 < v_size) { \
                vstore3(cell_sate_data.s012, 0, cell_state + g0); \
                vstore3(output_gate_data.s012, 0, output_state + g0); \
                vstore3(output_gate_data.s012, 0, output + g0 + output_offset); \
            } else if (g0 + 1 < v_size) { \
                vstore2(cell_sate_data.s01, 0, cell_state + g0); \
                vstore2(output_gate_data.s01, 0, output_state + g0); \
                vstore2(output_gate_data.s01, 0, output + g0 + output_offset); \
            } else if (g0 < v_size) { \
                cell_state[g0] = cell_sate_data.s0; \
                output_state[g0] = output_gate_data.s0; \
                output[g0 + output_offset] = output_gate_data.s0; \
            } \
        }

#define MATRIX_BATCH_VECTOR_MUL_ACC(matrix, vector, vector_offset, col, result, result_offset) \
    int matrix_start = get_global_id(1) * col; \
    int output_idx = get_global_id(0) * get_global_size(1) + get_global_id(1); \
    int input_start = get_global_id(0) * col; \
    DATA_T output = (DATA_T)0.0f; \
    for (int idx = 0; idx < col; idx++) { \
        output += matrix[matrix_start + idx] * vector[input_start + idx + vector_offset]; \
    } \
    result[output_idx + result_offset] = output + result[output_idx + result_offset];

#define MEAN_STDDEV_NORM(in, out, in_offset, out_offset, v_size, eps) \
        int batch = get_global_id(0); \
        DATA_T sum = (DATA_T)0.0f; \
        DATA_T sum_sq = (DATA_T)0.0f; \
        for (int i = 0; i < v_size; ++i) { \
            sum += in[batch * v_size + i + in_offset]; \
            sum_sq += \
                in[batch * v_size + i + in_offset] * in[batch * v_size + i + in_offset]; \
        } \
        DATA_T mean = sum / v_size; \
        DATA_T stddev_inv = (DATA_T)0.0f; \
        DATA_T variance = sum_sq / v_size - mean * mean; \
        if (variance == 0) { \
            stddev_inv = (DATA_T)1.0f / (DATA_T)sqrt(eps); \
        } else { \
            stddev_inv = (DATA_T)1.0f / sqrt(variance); \
        } \
        for (int i = 0; i < v_size; ++i) { \
            out[batch * v_size + i + out_offset] = \
                (in[batch * v_size + i + in_offset] - mean) * stddev_inv; \
        }

#define REORDER_4GATES_MATRIX(matrix, aligned_matrix, height, width) \
    int g0 = get_global_id(0); \
    int g1 = get_global_id(1); \
    if (g1 * 16 + 16 <= width) { \
        int input_index = g0 * width + g1 * 16; \
        int output_index = g1 * height * 16 + g0 * 16; \
        vstore16(vload16(0, matrix + input_index), 0, aligned_matrix + output_index); \
    } else { \
        int remain = width % 16; \
        for (int i = remain; i > 0; i--) { \
            int input_index = g0 * width + (width - i); \
            int output_index = g1 * height * 16 + g0 * 16 + (width - i); \
            aligned_matrix[output_index] = matrix[input_index]; \
        } \
    }


#define DATA_T half
#define DATA_T8 half8
ADD_SINGLE_KERNEL(merge_weights_FP16, (__global half *input_to_input_weights,
                                     __global half *input_to_forget_weights,
                                     __global half *input_to_cell_weights,
                                     __global half *input_to_output_weights,
                                     __global half *recurrent_to_input_weights,
                                     __global half *recurrent_to_forget_weights,
                                     __global half *recurrent_to_cell_weights,
                                     __global half *recurrent_to_output_weights,
                                     __global half *input_recurrent_to_4gate_weights,
                                     int n_input,
                                     int n_output,
                                     int n_cell) {
        MERGE_WEIGHTS(input_to_input_weights, input_to_forget_weights, input_to_cell_weights, input_to_output_weights, \
                      recurrent_to_input_weights, recurrent_to_forget_weights, recurrent_to_cell_weights, \
                      recurrent_to_output_weights, input_recurrent_to_4gate_weights, n_input, n_output, n_cell)
})

ADD_SINGLE_KERNEL(concat_input_output_opt_FP16, (__global half *src_input,
                                               __global half *src_output,
                                               __global half *dst_concat,
                                               int src_input_offset,
                                               int src_output_offset,
                                               int dst_concat_offset,
                                               int n_input,
                                               int n_output) {
        CONCAT_INPUT_OUTPUT_OPT(src_input, src_output, dst_concat, src_input_offset, src_output_offset, \
                                dst_concat_offset, n_input, n_output)
})

ADD_SINGLE_KERNEL(concat_input_output_FP16, (__global half *src_input,
                                           __global half *src_output,
                                           __global half *dst_concat,
                                           int src_input_offset,
                                           int src_output_offset,
                                           int dst_concat_offset,
                                           int n_input,
                                           int n_output) {
        CONCAT_INPUT_OUTPUT(src_input, src_output, dst_concat, src_input_offset, src_output_offset, dst_concat_offset, \
                            n_input, n_output)
})

ADD_SINGLE_KERNEL(matrix_batch_vector_mul_with_bias_FP16, (__global half *input_vector,
                                                         __global half *matrix,
                                                         __global half *bias,
                                                         unsigned int row,
                                                         unsigned int col,
                                                         unsigned int n_batch,
                                                         __global half *input_result,
                                                         __global half *forget_result,
                                                         __global half *cell_result,
                                                         __global half *output_result,
                                                         int input_offset,
                                                         int result_offset) {
    if (get_global_id(0) < n_batch && get_global_id(1) < row) {
        int matrix_start = get_global_id(1) * 16;
        int output_idx = get_global_id(0) * row / 4 + get_global_id(1);
        int input_start = get_global_id(0) * col;
        half16 output = (half16)0.0f;
        output.s0 = bias[get_global_id(1)];
        for (int idx = 0; idx < col / 16; idx++) {
            output += vload16(0, matrix + matrix_start + idx * 16 * row) *
                        vload16(0, input_vector + input_start + idx * 16 + input_offset);
        }

        for (int idx = col / 16 * 16; idx < col; idx++) {
            output.s0 += matrix[col / 16 * 16 * row + matrix_start + idx]
                            * input_vector[input_start + idx + input_offset];
        }

        output.s01234567 += output.s89abcdef;
        output.s0123 += output.s4567;
        output.s0 += output.s1 + output.s2 + output.s3;
        if (get_global_id(1) >= 0 && get_global_id(1) < row / 4) {
            input_result[output_idx + result_offset] = (half)1.0f / ((half)1.0f + exp(-(output.s0)));
        } else if (get_global_id(1) >= row / 4 && get_global_id(1) < 2 * row / 4) {
            forget_result[output_idx + result_offset - row / 4] = (half)1.0f / ((half)1.0f + exp(-(output.s0)));
        } else if (get_global_id(1) >= 2 * row / 4 && get_global_id(1) < 3 * row / 4) {
            cell_result[output_idx + result_offset - 2 * row / 4] = tanh(output.s0);
        } else if (get_global_id(1) >= 3 * row / 4 && get_global_id(1) < row) {
            output_result[output_idx + result_offset - 3 * row / 4] = (half)1.0f / ((half)1.0f + exp(-(output.s0)));
        }
    }
})

ADD_SINGLE_KERNEL(update_cell_output_FP16, (__global half *input_gate_scratch,
                                          __global half *forget_gate_scratch,
                                          __global half *cell_scratch,
                                          __global half *output_gate_scratch,
                                          __global half *cell_state,
                                          __global half *output,
                                          __global half *output_state,
                                          int v_size,
                                          int output_offset) {
        UPDATE_CELL_OUTPUT(input_gate_scratch, forget_gate_scratch, cell_scratch, output_gate_scratch, \
                           cell_state, output, output_state, v_size, output_offset)
})

ADD_SINGLE_KERNEL(matrix_batch_vector_mul_acc_FP16, (__global half *matrix,
                                                               __global half *vector,
                                                               int vector_offset,
                                                               unsigned int col,
                                                               __global half *result,
                                                               int result_offset) {
    MATRIX_BATCH_VECTOR_MUL_ACC(matrix, vector, vector_offset, col, result, result_offset)
})

ADD_SINGLE_KERNEL(special_relu_FP16, (__global half *in,
                                                __global half *out,
                                                int in_offset,
                                                int out_offset) {
    int idx = get_global_id(0);
    if (in[idx + in_offset] < 0) {
        out[idx + out_offset] = 0;
    } else {
        out[idx + out_offset] = in[idx + in_offset];
    }
})

ADD_SINGLE_KERNEL(special_tanh_FP16, (__global half *in,
                                                __global half *out,
                                                int in_offset,
                                                int out_offset) {
    int idx = get_global_id(0);
    out[idx + out_offset] = tanh(in[idx + in_offset]);
})

ADD_SINGLE_KERNEL(special_sigmoid_FP16, (__global half *in,
                                                   __global half *out,
                                                   int in_offset,
                                                   int out_offset) {
    int idx = get_global_id(0);
    out[idx + out_offset] = (half)1.0f / ((half)1.0f + exp(-in[idx + in_offset]));
})

ADD_SINGLE_KERNEL(special_none_FP16, (__global half *in,
                                                __global half *out,
                                                int in_offset,
                                                int out_offset) {
    int idx = get_global_id(0);
    out[idx + out_offset] = in[idx + in_offset];
})

ADD_SINGLE_KERNEL(vector_vector_product_accumulate_FP16, (__global half *v1,
                                                                    __global half *v2,
                                                                    __global half *result,
                                                                    int v1_offset,
                                                                    int v2_offset,
                                                                    int res_offset) {
    int idx = get_global_id(0);
    result[idx + res_offset] += v1[idx + v1_offset] * v2[idx + v2_offset];
})

ADD_SINGLE_KERNEL(vector_vector_product_FP16, (__global half *v1,
                                                         __global half *v2,
                                                         __global half *result,
                                                         int v1_offset,
                                                         int v2_offset,
                                                         int res_offset) {
    int idx = get_global_id(0);
    result[idx + res_offset] = v1[idx + v1_offset] * v2[idx + v2_offset];
})

ADD_SINGLE_KERNEL(batch_vector_product_acc_FP16, (__global half *vector,
                                                            __global half *batch_vector,
                                                            __global half *result,
                                                            int vec_offset,
                                                            int batch_offset,
                                                            int res_offset) {
    int res_idx = get_global_id(0) * get_global_size(1) + get_global_id(1);
    int batch_idx = get_global_id(0) * get_global_size(1) + get_global_id(1);
    int vec_idx = get_global_id(1);
    result[res_idx + res_offset] +=
        vector[vec_idx + vec_offset] * batch_vector[batch_idx + batch_offset];
})

ADD_SINGLE_KERNEL(batch_vector_product_FP16, (__global half *vector,
                                                        __global half *batch_vector,
                                                        __global half *result,
                                                        int vec_offset,
                                                        int batch_offset,
                                                        int res_offset) {
    int res_idx = get_global_id(0) * get_global_size(1) + get_global_id(1);
    int batch_idx = get_global_id(0) * get_global_size(1) + get_global_id(1);
    int vec_idx = get_global_id(1);
    result[res_idx + res_offset] =
        vector[vec_idx + vec_offset] * batch_vector[batch_idx + batch_offset];
})

ADD_SINGLE_KERNEL(vector_batch_vector_add_FP16, (__global half *vector,
                                                           __global half *batch_vector,
                                                           int v_size,
                                                           int vec_offset,
                                                           int batch_vec_offset) {
    int idx = get_global_id(0);
    for (int i = 0; i < v_size; i++) {
        batch_vector[idx * v_size + i + batch_vec_offset] += vector[i + vec_offset];
    }
})

ADD_SINGLE_KERNEL(clip_vector_FP16, (__global half *vec,
                                               __global half *result,
                                               float abs_limit,
                                               int vec_offset,
                                               int res_offset) {
    int idx = get_global_id(0);
    half abs_limit_half = (half)abs_limit;
    half in = vec[idx + vec_offset];
    half res = (abs_limit_half < in) ? abs_limit_half : in;
    res = (-abs_limit_half > res) ? -abs_limit_half : res;
    result[idx + res_offset] = res;
})

ADD_SINGLE_KERNEL(copy_vector_FP16, (__global half *vec,
                                               __global half *result,
                                               int vec_offset,
                                               int res_offset) {
    int idx = get_global_id(0);
    result[idx + res_offset] = vec[idx + vec_offset];
})

ADD_SINGLE_KERNEL(zero_vector_FP16, (__global half *vec, int vec_offset) {
    int idx = get_global_id(0);
    vec[idx + vec_offset] = (half)0.0f;
})

ADD_SINGLE_KERNEL(sub_1_vector_FP16, (__global half *vec, int vec_offset) {
    int idx = get_global_id(0);
    vec[idx + vec_offset] = (half)1.0f - vec[idx + vec_offset];
})

ADD_SINGLE_KERNEL(mean_stddev_norm_FP16, (__global half *in,
                                                    __global half *out,
                                                    int in_offset,
                                                    int out_offset,
                                                    int v_size,
                                                    float eps) {
    MEAN_STDDEV_NORM(in, out, in_offset, out_offset, v_size, eps)
})

ADD_SINGLE_KERNEL(reorder_4gates_matrix_FP16, (__global DATA_T *matrix,
                                               __global DATA_T *aligned_matrix,
                                               unsigned int height,
                                               unsigned int width) {
    REORDER_4GATES_MATRIX(matrix, aligned_matrix, height, width)
})
#undef DATA_T
#undef DATA_T8

#define DATA_T float
#define DATA_T8 float8
ADD_SINGLE_KERNEL(merge_weights_FP32, (__global float *input_to_input_weights,
                        __global float *input_to_forget_weights,
                        __global float *input_to_cell_weights,
                        __global float *input_to_output_weights,
                        __global float *recurrent_to_input_weights,
                        __global float *recurrent_to_forget_weights,
                        __global float *recurrent_to_cell_weights,
                        __global float *recurrent_to_output_weights,
                        __global float *input_recurrent_to_4gate_weights,
                        int n_input,
                        int n_output,
                        int n_cell) {
MERGE_WEIGHTS(input_to_input_weights, input_to_forget_weights, input_to_cell_weights, input_to_output_weights, \
        recurrent_to_input_weights, recurrent_to_forget_weights, recurrent_to_cell_weights, \
        recurrent_to_output_weights, input_recurrent_to_4gate_weights, n_input, n_output, n_cell)
})

ADD_SINGLE_KERNEL(concat_input_output_opt_FP32, (__global float *src_input,
                                __global float *src_output,
                                __global float *dst_concat,
                                int src_input_offset,
                                int src_output_offset,
                                int dst_concat_offset,
                                int n_input,
                                int n_output) {
CONCAT_INPUT_OUTPUT_OPT(src_input, src_output, dst_concat, src_input_offset, src_output_offset, \
                dst_concat_offset, n_input, n_output)
})

ADD_SINGLE_KERNEL(concat_input_output_FP32, (__global float *src_input,
                            __global float *src_output,
                            __global float *dst_concat,
                            int src_input_offset,
                            int src_output_offset,
                            int dst_concat_offset,
                            int n_input,
                            int n_output) {
CONCAT_INPUT_OUTPUT(src_input, src_output, dst_concat, src_input_offset, src_output_offset, dst_concat_offset, \
            n_input, n_output)
})

ADD_SINGLE_KERNEL(matrix_batch_vector_mul_with_bias_FP32, (__global float *input_vector,
                                                         __global float *matrix,
                                                         __global float *bias,
                                                         unsigned int row,
                                                         unsigned int col,
                                                         unsigned int n_batch,
                                                         __global float *input_result,
                                                         __global float *forget_result,
                                                         __global float *cell_result,
                                                         __global float *output_result,
                                                         int input_offset,
                                                         int result_offset) {
    if (get_global_id(0) < n_batch && get_global_id(1) < row) {
        int matrix_start = get_global_id(1) * 16;
        int output_idx = get_global_id(0) * row / 4 + get_global_id(1);
        int input_start = get_global_id(0) * col;
        float16 output = (float16)0.0f;
        output.s0 = bias[get_global_id(1)];
        for (int idx = 0; idx < col / 16; idx++) {
            output += vload16(0, matrix + matrix_start + idx * 16 * row) *
                        vload16(0, input_vector + input_start + idx * 16 + input_offset);
        }

        for (int idx = col / 16 * 16; idx < col; idx++) {
            output.s0 += matrix[col / 16 * 16 * row + matrix_start + idx]
                            * input_vector[input_start + idx + input_offset];
        }

        output.s01234567 += output.s89abcdef;
        output.s0123 += output.s4567;
        output.s0 += output.s1 + output.s2 + output.s3;
        if (get_global_id(1) >= 0 && get_global_id(1) < row / 4) {
            input_result[output_idx + result_offset] = 1.0f / (1.0f + exp(-(output.s0)));
        } else if (get_global_id(1) >= row / 4 && get_global_id(1) < 2 * row / 4) {
            forget_result[output_idx + result_offset - row / 4] = 1.0f / (1.0f + exp(-(output.s0)));
        } else if (get_global_id(1) >= 2 * row / 4 && get_global_id(1) < 3 * row / 4) {
            cell_result[output_idx + result_offset - 2 * row / 4] = tanh(output.s0);
        } else if (get_global_id(1) >= 3 * row / 4 && get_global_id(1) < row) {
            output_result[output_idx + result_offset - 3 * row / 4] = 1.0f / (1.0f + exp(-(output.s0)));
        }
    }
})

ADD_SINGLE_KERNEL(update_cell_output_FP32, (__global float *input_gate_scratch,
                            __global float *forget_gate_scratch,
                            __global float *cell_scratch,
                            __global float *output_gate_scratch,
                            __global float *cell_state,
                            __global float *output,
                            __global float *output_state,
                            int v_size,
                            int output_offset) {
UPDATE_CELL_OUTPUT(input_gate_scratch, forget_gate_scratch, cell_scratch, output_gate_scratch, \
            cell_state, output, output_state, v_size, output_offset)
})

ADD_SINGLE_KERNEL(matrix_batch_vector_mul_acc_FP32, (__global float *matrix,
                                                __global float *vector,
                                                int vector_offset,
                                                unsigned int col,
                                                __global float *result,
                                                int result_offset) {
MATRIX_BATCH_VECTOR_MUL_ACC(matrix, vector, vector_offset, col, result, result_offset)
})

ADD_SINGLE_KERNEL(special_relu_FP32, (__global float *in,
                                __global float *out,
                                int in_offset,
                                int out_offset) {
    int idx = get_global_id(0);
    if (in[idx + in_offset] < 0) {
        out[idx + out_offset] = 0;
    } else {
        out[idx + out_offset] = in[idx + in_offset];
    }
})

ADD_SINGLE_KERNEL(special_tanh_FP32, (__global float *in,
                                __global float *out,
                                int in_offset,
                                int out_offset) {
    int idx = get_global_id(0);
    out[idx + out_offset] = tanh(in[idx + in_offset]);
})

ADD_SINGLE_KERNEL(special_sigmoid_FP32, (__global float *in,
                                    __global float *out,
                                    int in_offset,
                                    int out_offset) {
    int idx = get_global_id(0);
    out[idx + out_offset] = 1.0f / (1.0f + exp(-in[idx + in_offset]));
})

ADD_SINGLE_KERNEL(special_none_FP32, (__global float *in,
                                __global float *out,
                                int in_offset,
                                int out_offset) {
    int idx = get_global_id(0);
    out[idx + out_offset] = in[idx + in_offset];
})

ADD_SINGLE_KERNEL(vector_vector_product_accumulate_FP32, (__global float *v1,
                                                    __global float *v2,
                                                    __global float *result,
                                                    int v1_offset,
                                                    int v2_offset,
                                                    int res_offset) {
    int idx = get_global_id(0);
    result[idx + res_offset] += v1[idx + v1_offset] * v2[idx + v2_offset];
})

ADD_SINGLE_KERNEL(vector_vector_product_FP32, (__global float *v1,
                                            __global float *v2,
                                            __global float *result,
                                            int v1_offset,
                                            int v2_offset,
                                            int res_offset) {
    int idx = get_global_id(0);
    result[idx + res_offset] = v1[idx + v1_offset] * v2[idx + v2_offset];
})

ADD_SINGLE_KERNEL(batch_vector_product_acc_FP32, (__global float *vector,
                                            __global float *batch_vector,
                                            __global float *result,
                                            int vec_offset,
                                            int batch_offset,
                                            int res_offset) {
    int res_idx = get_global_id(0) * get_global_size(1) + get_global_id(1);
    int batch_idx = get_global_id(0) * get_global_size(1) + get_global_id(1);
    int vec_idx = get_global_id(1);
    result[res_idx + res_offset] +=
        vector[vec_idx + vec_offset] * batch_vector[batch_idx + batch_offset];
})

ADD_SINGLE_KERNEL(batch_vector_product_FP32, (__global float *vector,
                                        __global float *batch_vector,
                                        __global float *result,
                                        int vec_offset,
                                        int batch_offset,
                                        int res_offset) {
    int res_idx = get_global_id(0) * get_global_size(1) + get_global_id(1);
    int batch_idx = get_global_id(0) * get_global_size(1) + get_global_id(1);
    int vec_idx = get_global_id(1);
    result[res_idx + res_offset] =
        vector[vec_idx + vec_offset] * batch_vector[batch_idx + batch_offset];
})

ADD_SINGLE_KERNEL(vector_batch_vector_add_FP32, (__global float *vector,
                                            __global float *batch_vector,
                                            int v_size,
                                            int vec_offset,
                                            int batch_vec_offset) {
    int idx = get_global_id(0);
    for (int i = 0; i < v_size; i++) {
        batch_vector[idx * v_size + i + batch_vec_offset] += vector[i + vec_offset];
    }
})

ADD_SINGLE_KERNEL(clip_vector_FP32, (__global float *vec,
                                __global float *result,
                                float abs_limit,
                                int vec_offset,
                                int res_offset) {
    int idx = get_global_id(0);
    float in = vec[idx + vec_offset];
    float res = (abs_limit < in) ? abs_limit : in;
    res = (-abs_limit > res) ? -abs_limit : res;
    result[idx + res_offset] = res;
})

ADD_SINGLE_KERNEL(copy_vector_FP32, (__global float *vec,
                                __global float *result,
                                int vec_offset,
                                int res_offset) {
    int idx = get_global_id(0);
    result[idx + res_offset] = vec[idx + vec_offset];
})

ADD_SINGLE_KERNEL(zero_vector_FP32, (__global float *vec, int vec_offset) {
    int idx = get_global_id(0);
    vec[idx + vec_offset] = 0.0f;
})

ADD_SINGLE_KERNEL(sub_1_vector_FP32, (__global float *vec, int vec_offset) {
    int idx = get_global_id(0);
    vec[idx + vec_offset] = 1.0f - vec[idx + vec_offset];
})

ADD_SINGLE_KERNEL(mean_stddev_norm_FP32, (__global float *in,
                                    __global float *out,
                                    int in_offset,
                                    int out_offset,
                                    int v_size,
                                    float eps) {
    MEAN_STDDEV_NORM(in, out, in_offset, out_offset, v_size, eps)
})

ADD_SINGLE_KERNEL(reorder_4gates_matrix_FP32, (__global DATA_T *matrix,
                                               __global DATA_T *aligned_matrix,
                                               unsigned int height,
                                               unsigned int width) {
    REORDER_4GATES_MATRIX(matrix, aligned_matrix, height, width)
})
#undef DATA_T
#undef DATA_T8

}  // namespace gpu
}  // namespace ud
}  // namespace enn
