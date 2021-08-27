#ifndef ENN_DECONV_GUARD_H
#define ENN_DECONV_GUARD_H

#include <array>
#include <cmath>
#include <cstdlib>
#include <vector>

#include "userdriver/common/operator_interfaces/common/ActivationInfo.hpp"

class DeconvGuard {
 public:
    DeconvGuard() = default;
    DeconvGuard(const DeconvGuard &deconvGuard) = default;

    void PrepareDeconvGuard(unsigned int input_batch, unsigned int input_channel,
                            unsigned int input_height, unsigned int input_width,
                            unsigned int pad_h, unsigned int pad_w, unsigned int stride_h,
                            unsigned int stride_w, unsigned int kernel_h, unsigned int kernel_w,
                            unsigned int group, unsigned int output_batch, unsigned int output_channel,
                            unsigned int output_height, unsigned int output_width) {
        this->input_batch = input_batch;
        this->input_channel = input_channel;
        this->input_height = input_height;
        this->input_width = input_width;
        this->pad_h = pad_h;
        this->pad_w = pad_w;
        this->stride_h = stride_h;
        this->stride_w = stride_w;
        this->kernel_h = kernel_h;
        this->kernel_w = kernel_w;
        this->output_batch = output_batch;
        this->output_channel = output_channel;
        this->output_height = output_height;
        this->output_width = output_width;
        this->group = group;
        is_init = true;
    }

    std::array<int, 4> GetOutputShape() {
        std::array<int, 4> output_shape = {{ output_batch, output_channel, output_height, output_width }};
        return output_shape;
    }

    void DeconvGuardRun(const float* input, const float* filter, const float* bias, float *output, ActivationInfo::ActivationType act_type, bool enable) {
        if (!is_init) {
            std::cerr << "You are calling un-initialized DeconvGuard, Program aborts !!!" << std::endl;
            abort();
        }
        //matrix multiplication
        int size = output_batch * input_height * kernel_h * kernel_w * output_channel * input_width;
        std::vector<float > convert_output(size);
        int row = input_channel / group;
        int col_filter = kernel_w * kernel_h * output_channel /
                         group;
        int col_input = input_height * input_width;
        for (int g = 0; g < group; g++) {
            int offset_filter = g * col_filter * row;
            int offset_input = g * col_input * row;
            int offset_output = g * col_filter * col_input;
            for (int b = 0; b < input_batch; b++) {
                for (int c_f = 0; c_f < col_filter; c_f++) {
                    for (int c_i = 0; c_i < col_input; c_i++) {
                        float temp = 0.0;
                        for (int r = 0; r < row; r++) {
                            temp += filter[r * col_filter + c_f + offset_filter] *
                                    input[b * row * col_input * group + r * col_input +
                                            c_i + offset_input];
                        }
                        convert_output[b * col_filter * col_input * group + c_f * col_input + c_i +
                                         offset_output] = temp;
                    }
                }
            }
        }
        //convert top
        for (int j = 0; j < output_batch * output_channel; j++) {
            int c_im = j;
            int bias_num = j % output_channel;
            for (int h = 0; h < output_height; h++) {
                int h_im = h + pad_h;
                for (int w = 0; w < output_width; w++) {
                    int w_im = w + pad_w;
                    int w_col_start = (w_im < kernel_w) ? 0 :
                                      (w_im - kernel_w) /
                                      stride_w +
                                      1;
                    int w_col_end = (w_im / stride_w + 1) < input_width ?
                                    (w_im / stride_w + 1) : input_width;
                    int h_col_start = (h_im < kernel_h) ? 0 :
                                      (h_im - kernel_h) /
                                      stride_h +
                                      1;
                    int h_col_end =
                        (h_im / stride_h + 1) < input_height ?
                        (h_im / stride_h + 1) : input_height;
                    float val = 0.0;
                    for (int h_col = h_col_start; h_col < h_col_end; ++h_col) {
                        for (int w_col = w_col_start; w_col < w_col_end; ++w_col) {
                            int c_col =
                                c_im * kernel_w * kernel_h +
                                (h_im - h_col * stride_h) *
                                kernel_w +
                                (w_im - w_col * stride_w);
                            val += convert_output[
                                       (c_col * input_height + h_col) *
                                       input_width +
                                       w_col];
                        }
                    }
                    float tmp_out = val + bias[bias_num];
                    if (enable) {
                        switch(act_type) {
                            case ActivationInfo::ActivationType::RELU:
                                tmp_out = tmp_out > 0 ? tmp_out : 0;
                                break;
                            case ActivationInfo::ActivationType::RELU1:
                                tmp_out = tmp_out > 1 ? 1 : tmp_out < -1 ? -1 : tmp_out;
                                break;
                            case ActivationInfo::ActivationType::RELU6:
                                tmp_out = tmp_out > 6 ? 6 : tmp_out < 0 ? 0 : tmp_out;
                                break;
                            case ActivationInfo::ActivationType::SIGMOID:
                                tmp_out = static_cast<float>(1.f / (1.f + exp(-tmp_out)));
                                break;
                            case ActivationInfo::ActivationType::TANH:
                                tmp_out = std::tanh(tmp_out);
                                break;
                            default:
                                break;
                        }
                    }
                    output[j * output_height * output_width +
                             h * output_width + w] = tmp_out;
                }
            }
        }
    }

 private:
    bool is_init = false;
    int input_batch = 0;
    int input_channel = 0;
    int input_height = 0;
    int input_width = 0;
    int pad_h = 0;
    int pad_w = 0;
    int stride_h = 0;
    int stride_w = 0;
    int kernel_h = 0;
    int kernel_w = 0;
    int output_batch = 0;
    int output_channel = 0;
    int output_height = 0;
    int output_width = 0;
    int group = 0;
};

#endif //ENN_DECONV_GUARD_H
