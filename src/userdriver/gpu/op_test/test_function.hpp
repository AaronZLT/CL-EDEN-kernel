#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/gpu/common/CLIncludes.hpp"

class TestTypeFP16 {
  public:
    static const PrecisionType precision = PrecisionType::FP16;
};

class TestTypeFP32 {
  public:
    static const PrecisionType precision = PrecisionType::FP32;
};

typedef ::testing::Types<TestTypeFP16, TestTypeFP32> TestFP32AndFP16Type;
typedef ::testing::Types<TestTypeFP32> TestFP32Type;

// reference code was ported from S.LSI implement
template <typename T>
void NormalizationRef(T *input,
                      float *output,
                      int channel,
                      int height,
                      int width,
                      float *mean,
                      float *scale,
                      uint8_t bgr_transpose,
                      float &max) {
    int i, j, k;
    T num;
    uint8_t c_type_order;  //*< channel type order
    uint32_t idx;
    max = -65532;
    if (bgr_transpose == 1) {
        //* normalization & bgr_transpose
        uint8_t b, g, r;
        float normalized;
        int b_idx, g_idx, r_idx;
        b = 0;
        g = 1;
        r = 2;

        b_idx = 0;
        g_idx = height * width;
        r_idx = height * width * 2;

        for (i = 0; i < channel; i++) {
            for (j = 0; j < height; j++) {
                for (k = 0; k < width; k++) {
                    idx = (i * height * width) + (j * width) + k;
                    num = input[idx];
                    c_type_order = i % channel;
                    normalized = ((float)num - mean[c_type_order]) * scale[c_type_order];
                    max = max > normalized ? max : normalized;
                    if (c_type_order == b)
                        output[b_idx++] = normalized;
                    else if (c_type_order == g)
                        output[g_idx++] = normalized;
                    else if (c_type_order == r)
                        output[r_idx++] = normalized;
                }
            }
        }
    } else {
        // just normalization
        for (i = 0; i < channel; i++) {
            for (j = 0; j < height; j++) {
                for (k = 0; k < width; k++) {
                    idx = (i * height * width) + (j * width) + k;
                    num = input[idx];
                    c_type_order = i % channel;
                    output[idx] = (static_cast<float>(num) - mean[i]) * scale[i];
                    max = max > output[idx] ? max : output[idx];
                }
            }
        }
    }
}

class ReluGuard {
  public:
    ReluGuard() = default;

    void GuardPrepare(const size_t size) {
        this->size_ = size;
        this->is_init_ = true;
    }

    void GuardRun(const float *input, float *output) {
        if (!is_init_) {
            std::cerr << "You are calling un-initialized ReluGuard, Program aborts !!!" << std::endl;
            abort();
        }

        for (size_t i = 0; i < size_; i++) {
            output[i] = input[i] > 0 ? input[i] : 0;
        }
    }

  private:
    size_t size_;
    bool is_init_ = false;
};

class Relu1Guard {
  public:
    Relu1Guard() = default;

    void GuardPrepare(const size_t size) {
        this->size_ = size;
        this->is_init_ = true;
    }

    void GuardRun(const float *input, float *output) {
        if (!is_init_) {
            std::cerr << "You are calling un-initialized Relu1Guard, Program aborts !!!" << std::endl;
            abort();
        }

        for (size_t i = 0; i < size_; i++) {
            output[i] = input[i] > 1 ? 1 : (input[i] < -1 ? -1 : input[i]);
        }
    }

  private:
    size_t size_;
    bool is_init_ = false;
};

class Relu6Guard {
  public:
    Relu6Guard() = default;

    void GuardPrepare(const size_t size) {
        this->size_ = size;
        this->is_init_ = true;
    }

    void GuardRun(const float *input, float *output) {
        if (!is_init_) {
            std::cerr << "You are calling un-initialized Relu1Guard, Program aborts !!!" << std::endl;
            abort();
        }

        for (size_t i = 0; i < size_; i++) {
            output[i] = input[i] > 6 ? 6 : (input[i] < 0 ? 0 : input[i]);
        }
    }

  private:
    size_t size_;
    bool is_init_ = false;
};

class SigmoidGuard {
  public:
    SigmoidGuard() = default;

    void GuardPrepare(const size_t size) {
        this->size_ = size;
        this->is_init_ = true;
    }

    void GuardRun(const float *input, float *output) {
        if (!is_init_) {
            std::cerr << "You are calling un-initialized SigmoidGuard, Program aborts !!!" << std::endl;
            abort();
        }

        for (size_t i = 0; i < size_; i++) {
            output[i] = static_cast<float>(1.f / (1.f + exp(-input[i])));
        }
    }

  private:
    size_t size_;
    bool is_init_ = false;
};

class ScaleGuard {
  public:
    ScaleGuard() = default;

    void GuardPrepare(float* scale, float* bias, const uint32_t& input_batch,
                      const uint32_t& input_channel, const uint32_t& input_height,
                      const uint32_t& input_width) {
        this->scale = scale;
        this->bias = bias;
        this->input_batch = input_batch;
        this->input_channel = input_channel;
        this->input_height = input_height;
        this->input_width = input_width;
        is_init = true;
    }

    void GuardRun(const float* input, float *output) {
        if (!is_init) {
            std::cerr << "You are calling un-initialized ScaleGuard, Program aborts !!!" << std::endl;
            abort();
        }

        for (uint32_t n = 0; n < input_batch; n++) {
            for (uint32_t c = 0; c < input_channel; c++) {
                for (uint32_t h = 0; h < input_height; h++) {
                    for (uint32_t w = 0; w < input_width; w++) {
                        int index = n * input_channel * input_height * input_width +
                                    c * input_height * input_width + h * input_width + w;
                        output[index] = input[index] * scale[c] + bias[c];
                    }
                }
            }
        }
    }

  private:
    float* scale;
    float* bias;
    uint32_t input_batch = 0;
    uint32_t input_channel = 0;
    uint32_t input_height = 0;
    uint32_t input_width = 0;
    bool is_init = false;
};

class MaxpoolGuard {
  public:
    MaxpoolGuard() = default;
    void GuardPrepare(const Dim4 &input_dim, const Pad4 &padding, const Dim2 &stride, const Dim2 &filter, Dim4 &output_dim) {
        this->input_batch_ = input_dim.n;
        this->input_channel_ = input_dim.c;
        this->input_height_ = input_dim.h;
        this->input_width_ = input_dim.w;
        this->pad_h_ = padding.t;
        this->pad_w_ = padding.l;
        this->stride_h_ = stride.h;
        this->stride_w_ = stride.w;
        this->filter_h_ = filter.h;
        this->filter_w_ = filter.w;
        this->output_batch_ = input_batch_;
        this->output_channel_ = input_channel_;
        this->output_height_ = floor(static_cast<float>(input_height_ + 2 * pad_h_ - filter_h_ + stride_h_) / stride_h_);
        this->output_width_ = floor(static_cast<float>(input_width_ + 2 * pad_w_ - filter_w_ + stride_w_) / stride_w_);
        output_dim.n = this->output_batch_;
        output_dim.c = this->output_channel_;
        output_dim.h = this->output_height_;
        output_dim.w = this->output_width_;
        this->is_init_ = true;
    }

    void GuardRun(const float *input, float *output) {
        if (!is_init_) {
            std::cerr << "You are calling un-initialized MaxpoolGuard, Program aborts !!!" << std::endl;
            abort();
        }

        const int padded_size = input_batch_ * input_channel_ * (input_height_ + 2 * pad_h_) * (input_width_ + 2 * pad_w_);
        float *padded_input = new float[padded_size];
        for (int i = 0; i < padded_size; i++) {
            padded_input[i] = BOUNDARY_FLAG;
        }
        for (int b = 0; b < input_batch_; b++) {
            for (int c = 0; c < input_channel_; c++) {
                for (int h = 0; h < input_height_; h++) {
                    for (int w = 0; w < input_width_; w++) {
                        int block_size = (input_height_ + 2 * pad_h_) * (input_width_ + 2 * pad_w_);
                        int paddedIndex = b * input_channel_ * block_size + c * block_size +
                                          (h + pad_h_) * (input_width_ + 2 * pad_w_) + w + pad_w_;
                        int inputIndex = b * input_channel_ * input_height_ * input_width_ +
                                         c * input_height_ * input_width_ + h * input_width_ + w;
                        padded_input[paddedIndex] = input[inputIndex];
                    }
                }
            }
        }
        float max = std::numeric_limits<float>::lowest();
        for (int b = 0; b < output_batch_; b++) {
            for (int c = 0; c < output_channel_; c++) {
                for (int h = 0; h < output_height_; h++) {
                    for (int w = 0; w < output_width_; w++) {
                        int block_size = output_height_ * output_width_;
                        int output_index = b * output_channel_ * block_size + c * block_size + h * output_width_ + w;
                        int start_h = h * stride_h_;
                        int start_w = w * stride_w_;
                        int end_h = std::min(start_h + filter_h_, input_height_ + 2 * pad_h_);
                        int end_w = std::min(start_w + filter_w_, input_width_ + 2 * pad_w_);
                        max = std::numeric_limits<float>::lowest();
                        for (int row = start_h; row < end_h; row++) {
                            for (int col = start_w; col < end_w; col++) {
                                int padded_block_size = (input_height_ + 2 * pad_h_) * (input_width_ + 2 * pad_w_);
                                int padded_idx = b * input_channel_ * padded_block_size + c * padded_block_size +
                                                 row * (input_width_ + 2 * pad_w_) + col;
                                if (padded_input[padded_idx] > max) {
                                    max = padded_input[padded_idx];
                                }
                            }
                        }
                        output[output_index] = max;
                    }
                }
            }
        }
        delete[] padded_input;
    }

  private:
    const float BOUNDARY_FLAG = -3.40282346638528860e+36;
    uint32_t input_batch_ = 0;
    uint32_t input_channel_ = 0;
    uint32_t input_height_ = 0;
    uint32_t input_width_ = 0;
    uint32_t pad_h_ = 0;
    uint32_t pad_w_ = 0;
    uint32_t stride_h_ = 0;
    uint32_t stride_w_ = 0;
    uint32_t filter_h_ = 0;
    uint32_t filter_w_ = 0;
    uint32_t output_batch_ = 0;
    uint32_t output_channel_ = 0;
    uint32_t output_height_ = 0;
    uint32_t output_width_ = 0;
    bool is_init_ = false;
};

class AveragepoolGuard {
  public:
    AveragepoolGuard() = default;
    void GuardPrepare(const Dim4 &input_dim, const Pad4 &padding, const Dim2 &stride, const Dim2 &filter, Dim4 &output_dim) {
        this->input_batch_ = input_dim.n;
        this->input_channel_ = input_dim.c;
        this->input_height_ = input_dim.h;
        this->input_width_ = input_dim.w;
        this->pad_h_ = padding.t;
        this->pad_w_ = padding.l;
        this->stride_h_ = stride.h;
        this->stride_w_ = stride.w;
        this->filter_h_ = filter.h;
        this->filter_w_ = filter.w;
        this->output_batch_ = input_batch_;
        this->output_channel_ = input_channel_;
        this->output_height_ = floor(static_cast<float>(input_height_ + 2 * pad_h_ - filter_h_ + stride_h_) / stride_h_);
        this->output_width_ = floor(static_cast<float>(input_width_ + 2 * pad_w_ - filter_w_ + stride_w_) / stride_w_);
        output_dim.n = this->output_batch_;
        output_dim.c = this->output_channel_;
        output_dim.h = this->output_height_;
        output_dim.w = this->output_width_;
        this->is_init_ = true;
    }

    void GuardRun(const float *input, float *output) {
        if (!is_init_) {
            std::cerr << "You are calling un-initialized AveragepoolGuard, Program aborts !!!" << std::endl;
            abort();
        }

        const int padded_size = input_batch_ * input_channel_ * (input_height_ + 2 * pad_h_) * (input_width_ + 2 * pad_w_);
        float *padded_input = new float[padded_size];
        for (int i = 0; i < padded_size; i++) {
            padded_input[i] = BOUNDARY_FLAG;
        }
        for (int b = 0; b < input_batch_; b++) {
            for (int c = 0; c < input_channel_; c++) {
                for (int h = 0; h < input_height_; h++) {
                    for (int w = 0; w < input_width_; w++) {
                        int padded_index = b * input_channel_ * (input_height_ + 2 * pad_h_) * (input_width_ + 2 * pad_h_) +
                                           c * (input_height_ + 2 * pad_h_) * (input_width_ + 2 * pad_h_) +
                                           (h + pad_h_) * (input_width_ + 2 * pad_h_) + w + pad_h_;
                        int inputIndex = b * input_channel_ * input_height_ * input_width_ +
                                         c * input_height_ * input_width_ + h * input_width_ + w;
                        padded_input[padded_index] = input[inputIndex];
                    }
                }
            }
        }
        for (int b = 0; b < input_batch_; b++) {
            for (int c = 0; c < input_channel_; c++) {
                for (int h = 0; h < output_height_; h++) {
                    for (int w = 0; w < output_width_; w++) {
                        int output_index = b * input_channel_ * output_height_ * output_width_ +
                                           c * output_height_ * output_width_ + h * output_width_ + w;
                        int start_h = h * stride_h_;
                        int start_w = w * stride_w_;
                        int end_h = std::min(start_h + filter_h_, input_height_ + 2 * pad_h_);
                        int end_w = std::min(start_w + filter_w_, input_width_ + 2 * pad_h_);
                        float accumlation = 0.0f;
                        for (int row = start_h; row < end_h; row++) {
                            for (int col = start_w; col < end_w; col++) {
                                int padded_index =
                                    b * input_channel_ * (input_height_ + 2 * pad_h_) * (input_width_ + 2 * pad_h_) +
                                    c * (input_height_ + 2 * pad_h_) * (input_width_ + 2 * pad_h_) +
                                    row * (input_width_ + 2 * pad_h_) + col;
                                if (padded_input[padded_index] > BOUNDARY_FLAG) {
                                    accumlation += padded_input[padded_index];
                                }
                            }
                        }
                        output[output_index] = accumlation / ((end_h - start_h) * (end_w - start_w));
                    }
                }
            }
        }
        delete[] padded_input;
    }

  private:
    const float BOUNDARY_FLAG = -3.40282346638528860e+36;
    uint32_t input_batch_ = 0;
    uint32_t input_channel_ = 0;
    uint32_t input_height_ = 0;
    uint32_t input_width_ = 0;
    uint32_t pad_h_ = 0;
    uint32_t pad_w_ = 0;
    uint32_t stride_h_ = 0;
    uint32_t stride_w_ = 0;
    uint32_t filter_h_ = 0;
    uint32_t filter_w_ = 0;
    uint32_t output_batch_ = 0;
    uint32_t output_channel_ = 0;
    uint32_t output_height_ = 0;
    uint32_t output_width_ = 0;
    bool is_init_ = false;
};

class ConcatGuard {
  public:
    ConcatGuard() = default;

    void GuardPrepare(const std::vector<std::vector<uint32_t>> &input_shape, const uint32_t &axis) {
        this->input_shape_ = input_shape;
        this->axis_ = axis;
        this->is_init_ = true;
    }

    void GuardRun(const std::vector<float *> &input, float *output) {
        uint32_t output_axis_shape = input_shape_[0][axis_];
        for (uint32_t i = 1; i < input.size(); i++) {
            output_axis_shape += input_shape_[i][axis_];
        }
        uint32_t num_concat = 1;
        for (uint32_t a = 0; a < axis_; ++a) {
            num_concat *= input_shape_[0][a];
        }
        uint32_t concat_input_size = 1;
        for (uint32_t a = (axis_ + 1); a < input_shape_[0].size(); ++a) {
            concat_input_size *= input_shape_[0][a];
        }
        uint32_t offset_concat_axis = 0;
        uint32_t output_concat_axis = output_axis_shape;
        for (std::vector<float *>::size_type i = 0; i < input.size(); i++) {
            uint32_t input_concat_axis = input_shape_[i][axis_];
            for (uint32_t n = 0; n < num_concat; ++n) {
                uint32_t offset_output = (n * output_concat_axis + offset_concat_axis) * concat_input_size;
                uint32_t offset_input = n * input_concat_axis * concat_input_size;
                memcpy(
                    output + offset_output, input[i] + offset_input, sizeof(float) * input_concat_axis * concat_input_size);
            }
            offset_concat_axis += input_concat_axis;
        }
    }

  private:
    std::vector<std::vector<uint32_t>> input_shape_;
    uint32_t axis_;
    bool is_init_ = false;
};

class SoftmaxGuard {
  public:
    SoftmaxGuard() = default;

    void GuardPrepare(const Dim4 &input_dim, const int32_t &axis, const float &beta) {
        this->input_batch_ = input_dim.n;
        this->input_channel_ = input_dim.c;
        this->input_height_ = input_dim.h;
        this->input_width_ = input_dim.w;
        this->axis_ = axis;
        this->beta_ = beta;
        this->is_init_ = true;
    }

    void GuardRun(const float *input, float *output) {
        if (!is_init_) {
            std::cerr << "You are calling un-initialized SoftmaxGuard, Program aborts !!!" << std::endl;
            abort();
        }
        int tmp_axis = axis_;
        if (tmp_axis < 0) {
            tmp_axis = axis_ + 4;
        }

        int outer_num, inner_num, channels;
        if (tmp_axis == 0) {
            outer_num = 1;
            channels = input_batch_;
            inner_num = input_channel_ * input_height_ * input_width_;
        } else if (tmp_axis == 1) {
            outer_num = input_batch_;
            channels = input_channel_;
            inner_num = input_height_ * input_width_;
        } else if (tmp_axis == 2) {
            outer_num = input_batch_ * input_channel_;
            channels = input_height_;
            inner_num = input_width_;
        } else {
            outer_num = input_batch_ * input_channel_ * input_height_;
            channels = input_width_;
            inner_num = 1;
        }

        float *max = new float[inner_num];
        float *expSum = new float[inner_num];

        for (int n = 0; n < outer_num; n++) {
            for (int i = 0; i < inner_num; i++) {
                max[i] = std::numeric_limits<float>::min();
                expSum[i] = 0.0f;
            }
            for (int c = 0; c < channels; c++) {
                for (int i = 0; i < inner_num; i++) {
                    int inputIndex = n * channels * inner_num + c * inner_num + i;
                    int index = i;
                    if (input[inputIndex] > max[index]) {
                        max[index] = input[inputIndex];
                    }
                }
            }
            for (int c = 0; c < channels; c++) {
                for (int i = 0; i < inner_num; i++) {
                    int inputIndex = n * channels * inner_num + c * inner_num + i;
                    int index = i;
                    float expValue = exp((input[inputIndex] - max[index]) * beta_);
                    output[inputIndex] = expValue;
                    expSum[index] += expValue;
                }
            }
            for (int c = 0; c < channels; c++) {
                for (int i = 0; i < inner_num; i++) {
                    int inputIndex = n * channels * inner_num + c * inner_num + i;
                    int index = i;
                    output[inputIndex] = output[inputIndex] / expSum[index];
                }
            }
        }

        delete[] max;
        delete[] expSum;
    }

  private:
    uint32_t input_batch_ = 0;
    uint32_t input_channel_ = 0;
    uint32_t input_height_ = 0;
    uint32_t input_width_ = 0;
    int32_t axis_;
    float beta_;
    bool is_init_ = false;
};

class MulGuard {
 public:
    MulGuard() = default;

    void GuardPrepare(const Dim4 &input_dim) {
        this->input_batch = input_dim.n;
        this->input_channel = input_dim.c;
        this->input_height = input_dim.h;
        this->input_width = input_dim.w;
        is_init = true;
    }

    void GuardRun(const std::vector<float *> &input_data, float *output) {
        if (!is_init) {
            std::cerr << "You are calling un-initialized MulGuard, Program aborts !!!" << std::endl;
            abort();
        }
        int size = input_batch * input_channel * input_height * input_width;
        for (int i = 0; i < size; i++) {
            output[i] = 1.0f;
        }
        for (std::vector<float*>::size_type i = 0; i < input_data.size(); i++) {
            for (uint32_t b = 0; b < input_batch; b++) {
                for (uint32_t c = 0; c < input_channel; c++) {
                    for (uint32_t h = 0; h < input_height; h++) {
                        for (uint32_t w = 0; w < input_width; w++) {
                            int index = b * input_channel * input_height * input_width
                                        + c * input_height * input_width + h * input_width + w;
                            output[index] *= input_data[i][index];
                        }
                    }
                }
            }
        }
    }

 private:
    bool is_init = false;
    uint32_t input_batch = 0;
    uint32_t input_channel = 0;
    uint32_t input_height = 0;
    uint32_t input_width = 0;
};

class AddGuard {
 public:
   AddGuard() = default;

   void GuardPrepare(const Dim4 input_dim) {
       this->input_batch = input_dim.n;
       this->input_channel = input_dim.c;
       this->input_height = input_dim.h;
       this->input_width = input_dim.w;
       is_init = true;
   }

    void GuardRun(const std::vector<float *> &input_data, float *coeff, float *output) {
        if (!is_init) {
            std::cerr << "You are calling un-initialized AddGuard, Program aborts !!!" << std::endl;
            abort();
        }
        int size = input_batch * input_channel * input_height * input_width;
        for (int i = 0; i < size; i++) {
            output[i] = 0.0f;
        }
        for (uint32_t i = 0; i < input_data.size(); i++) {
            for (uint32_t b = 0; b < input_batch; b++) {
                for (uint32_t c = 0; c < input_channel; c++) {
                    for (uint32_t h = 0; h < input_height; h++) {
                        for (uint32_t w = 0; w < input_width; w++) {
                            int index = b * input_channel * input_height * input_width
                                        + c * input_height * input_width + h * input_width + w;
                            output[index] += coeff[i] * input_data[i][index];
                        }
                    }
                }
            }
        }
    }

 private:
    bool is_init = false;
    uint32_t input_batch = 0;
    uint32_t input_channel = 0;
    uint32_t input_height = 0;
    uint32_t input_width = 0;
};

class ReshapeGuard {
  public:
    ReshapeGuard() = default;

    void GuardPrepare(int32_t input_size) {
        this->intput_size_ = input_size;
        is_init = true;
    }

    void GuardRun(const float *input, float *output) {
        if (!is_init) {
            std::cerr << "You are calling un-initialized ReshapeGuard, Program aborts !!!" << std::endl;
            abort();
        }
        for (int i = 0; i < intput_size_; i++) {
            output[i] = input[i];
        }
    }

  private:
    int32_t intput_size_;
    bool is_init = false;
};
