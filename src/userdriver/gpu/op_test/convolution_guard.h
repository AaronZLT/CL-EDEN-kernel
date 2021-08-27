#pragma once
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>

#include "userdriver/common/operator_interfaces/common/ActivationInfo.hpp"
#include "userdriver/common/operator_interfaces/common/Common.hpp"

class ConvolutionGuard {
  public:
    ConvolutionGuard() = default;
    ConvolutionGuard(const ConvolutionGuard &convolutionGuard) = default;
    ConvolutionGuard(const int &input_batch,
                     const int &input_channel,
                     const int &input_height,
                     const int &input_width,
                     const int &pad_h,
                     const int &pad_w,
                     const int &stride_h,
                     const int &stride_w,
                     const int &kernel_h,
                     const int &kernel_w,
                     int *output_batch,
                     int *output_channel,
                     int *output_height,
                     int *output_width,
                     const int &group,
                     const Dim2 &dilation) {
        PrepareConvolutionGuard(input_batch,
                                input_channel,
                                input_height,
                                input_width,
                                pad_h,
                                pad_w,
                                stride_h,
                                stride_w,
                                kernel_h,
                                kernel_w,
                                output_batch,
                                output_channel,
                                output_height,
                                output_width,
                                group,
                                dilation);
    }

    void PrepareConvolutionGuard(const int &input_batch,
                                 const int &input_channel,
                                 const int &input_height,
                                 const int &input_width,
                                 const int &pad_h,
                                 const int &pad_w,
                                 const int &stride_h,
                                 const int &stride_w,
                                 const int &kernel_h,
                                 const int &kernel_w,
                                 int *output_batch,
                                 int *output_channel,
                                 int *output_height,
                                 int *output_width,
                                 const int &group,
                                 const Dim2 &dilation) {
        this->group = group;
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
        this->output_batch = input_batch;
        this->output_channel = input_channel;
        int kernel_extent_h = dilation.h * (kernel_h - 1) + 1;
        int kernel_extent_w = dilation.w * (kernel_w - 1) + 1;
        this->output_height = (this->input_height + 2 * this->pad_h - kernel_extent_h + this->stride_h) / this->stride_h;
        this->output_width = (this->input_width + 2 * this->pad_w - kernel_extent_w + this->stride_w) / this->stride_w;
        *output_batch = this->output_batch;
        *output_channel = this->output_channel;
        *output_height = this->output_height;
        *output_width = this->output_width;

        filter_batch = *output_channel;
        filter_channel = this->input_channel / this->group;
        filter_height = this->kernel_h;
        filter_width = this->kernel_w;

        this->dilation = dilation;

        is_init = true;
    }

    std::array<int, 4> GetOutputShape() {
        std::array<int, 4> output_shape{{output_batch, output_channel, output_height, output_width}};
        return output_shape;
    }

    std::array<int, 4> GetFilterShape() {
        std::array<int, 4> filter_shape{{filter_batch, filter_channel, filter_height, filter_width}};
        return filter_shape;
    }
    void ConvolutionGuardRun(const float *input,
                             float *weight,
                             float *bias,
                             float *output,
                             ActivationInfo::ActivationType act_type,
                             bool enable) {
        if (!is_init) {
            std::cerr << "You are calling un-initialized ConvolutionGuard, Program aborts !!!" << std::endl;
            abort();
        }
        double temp = 0.0f;
        for (int out_n = 0; out_n < output_batch; out_n++) {
            for (int out_g = 0; out_g < group; out_g++) {
                for (int out_c = 0; out_c < output_channel / group; out_c++) {
                    for (int out_h = 0; out_h < output_height; out_h++) {
                        for (int out_w = 0; out_w < output_width; out_w++) {
                            temp = 0.0f;
                            for (int in_c = 0; in_c < input_channel / group; in_c++) {
                                for (int w_h = 0; w_h < filter_height; w_h++) {
                                    for (int w_w = 0; w_w < filter_width; w_w++) {
                                        if ((out_h * stride_h + w_h * dilation.h - pad_h) >= 0 &&
                                            (out_h * stride_h + w_h * dilation.h - pad_h) < input_height &&
                                            (out_w * stride_w + w_w * dilation.w - pad_w) >= 0 &&
                                            (out_w * stride_w + w_w * dilation.w - pad_w) < input_width) {
                                            temp += input[out_n * input_channel * input_height * input_width +
                                                          out_g * (input_channel / group) * input_height * input_width +
                                                          in_c * input_width * input_height +
                                                          (out_h * stride_h + w_h * dilation.h - pad_h) * input_width +
                                                          (out_w * stride_w + w_w * dilation.w - pad_w)] *
                                                    weight[out_g * (input_channel / group) * (output_channel / group) *
                                                               filter_height * filter_width +
                                                           out_c * (input_channel / group) * filter_height * filter_width +
                                                           in_c * filter_height * filter_width + w_h * filter_width + w_w];
                                        }
                                    }
                                }
                            }
                            temp += bias[out_g * (output_channel / group) + out_c];
                            if (enable) {
                                switch (act_type) {
                                case ActivationInfo::ActivationType::RELU: temp = temp > 0 ? temp : 0; break;
                                case ActivationInfo::ActivationType::RELU1:
                                    temp = temp > 1 ? 1 : temp < -1 ? -1 : temp;
                                    break;
                                case ActivationInfo::ActivationType::RELU6: temp = temp > 6 ? 6 : temp < 0 ? 0 : temp; break;
                                case ActivationInfo::ActivationType::SIGMOID:
                                    temp = static_cast<float>(1.f / (1.f + exp(-temp)));
                                    break;
                                case ActivationInfo::ActivationType::TANH: temp = std::tanh(temp); break;
                                default: break;
                                }
                            }
                            output[out_n * output_channel * output_height * output_width +
                                   out_g * (output_channel / group) * output_height * output_width +
                                   out_c * output_height * output_width + out_h * output_width + out_w] = temp;
                        }
                    }
                }
            }
        }
    }

  private:
    bool is_init = false;
    int group = 1;
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
    int filter_batch = 0;
    int filter_channel = 0;
    int filter_height = 0;
    int filter_width = 0;
    Dim2 dilation = {1, 1};
};
