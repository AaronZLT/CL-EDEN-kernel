#pragma once

#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLPadConvert.hpp"
#include "userdriver/common/operator_interfaces/common/ActivationInfo.hpp"
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/ITensor.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define MAX_WORK_GROUP_SIZE 128

#define WINO_TILE_ORIGINAL 0
#define WINO_TILE_1X2 1
#define WINO_TILE_2X4 2
#define WINO_TILE_2X8 3
#define WINO_TILE_1X8 4
#define WINO_TILE_4X4 5
#define WINO_TILE_1X2_MAKALU 6

#define WINO_NO_COALESCING 1
#define WINO_COALESCING_SIZE 24
#define WINO_COALESCING_SIZE_MAKALU 48
#define WINO_COALESCING_SIZE_MAKALU_SINGLE_LINE 12

#define WINO_TILE_SIZE_1 1
#define WINO_TILE_SIZE_2 2
#define WINO_TILE_SIZE_4 4
#define WINO_TILE_SIZE_8 8

#define WINO_SPLIT_SIZE_1 1
#define WINO_SPLIT_SIZE_2 2
#define WINO_SPLIT_SIZE_4 4

#define DEFAULT_SPEED_UP 0
#define BIFROST_SPEED_UP 1
#define MAKALU_SPEED_UP 2
#define MAKALU_SPEED_UP_1X2 3

#define MAKALU_COMPUTING_THRESHOLD 30000000

class CLWINOConvolution {
  public:
    CLWINOConvolution(const std::shared_ptr<CLRuntime> runtime,
                      const PrecisionType &precision,
                      const Dim4 &input_dim,
                      const Dim4 &output_dim);

    Status initialize(const Pad4 &padding,
                      const Dim2 &stride,
                      const uint32_t &group_size,
                      const uint32_t &axis,
                      const Dim2 &dilation,
                      const std::shared_ptr<ITensor> weight,
                      const std::shared_ptr<ITensor> bias,
                      const ActivationInfo &activate_info,
                      const bool &weights_as_input = false,
                      const bool &androidNN = false,
                      const bool &isNCHW = true);

    Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output);

  private:
    typedef struct {
        uint32_t num_output_;
        uint32_t pad_right_;  // currently, only padright = padleft and padtop = padbottom is supported
        uint32_t pad_left_;
        uint32_t pad_top_;
        uint32_t pad_bottom_;
        uint32_t kernel_height_;
        uint32_t kernel_width_;
        uint32_t stride_height_;
        uint32_t stride_width_;
        uint32_t group_;
        uint32_t axis_;
        Dim2 dilation_;
        std::shared_ptr<CLTensor> filter_;
        std::shared_ptr<CLTensor> bias_;
    } ConvDescriptor;

    bool weights_as_input_;
    bool androidNN_;
    bool isNCHW_;
    std::shared_ptr<CLTensor> weight_nchw_;

    ConvDescriptor conv_descriptor_;
    Status winoRun(const std::shared_ptr<CLTensor> input, std::shared_ptr<CLTensor> output);

    Status initKernelWeight();

    Status convertWeight(cl_mem &weight,
                         cl_mem &convert_weight,
                         const uint32_t &weight_count,
                         const uint32_t &cpt_output_number,
                         const uint32_t &output_channel,
                         const uint32_t &input_channel,
                         const int speed_up,
                         const uint32_t &splite_number);

    Status convertWeightCPU(float *weight,
                            float *convertedWeight,
                            const uint32_t &weightCount,
                            const uint32_t &computedTopNumber,
                            const uint32_t &topChannel,
                            const uint32_t &bottomChannel,
                            const int speedUp,
                            const uint32_t &spliteNumber);

    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;

    std::shared_ptr<CLTensor> convert_buffer_;
    std::shared_ptr<CLTensor> split_buffer_;
    std::shared_ptr<CLTensor> weight_buffer_;
    std::shared_ptr<CLTensor> converted_filter_;
    std::shared_ptr<CLTensor> aligned_model_weight_;
    /* this step does input padding, tiling, and internal transpose */
    std::shared_ptr<struct _cl_kernel> winograd_convert_kernel_;
    std::shared_ptr<struct _cl_kernel> winograd_convert_kernel_bifrost_;
    /* this step does matrix multipling, tile transposing and deppading */
    std::shared_ptr<struct _cl_kernel> winograd_multi_kernel_;
    /* optimized kernel, but only support non-group and squared feature */
    std::shared_ptr<struct _cl_kernel> winograd_multi_kernel_optimized_;
    std::shared_ptr<struct _cl_kernel> winograd_multi_kernel_split_;
    std::shared_ptr<struct _cl_kernel> winograd_multi_kernel_merge_;
    std::shared_ptr<struct _cl_kernel> winograd_convert_weight_;
    std::shared_ptr<struct _cl_kernel> winograd_convert_weight_makalu_opt_;

    bool top_channel_need_aligned_;
    int branch_number_;
    int speed_up_;
    bool bifrost_speedup_;
    uint32_t coalescing_feature_height_;
    uint32_t compute_output_number_;
    uint32_t splite_number_;
    uint32_t wgrad_M_;
    uint32_t wgrad_stride_;
    uint32_t wgrad_tile_size_;
    uint32_t wgrad_align_height_;
    uint32_t wgrad_align_width_;
    uint32_t wgrad_tile_height_;
    uint32_t wgrad_tile_width_;

    Dim4 input_dim_;
    Dim4 output_dim_;

    ActivationInfo activation_info_;
};  // class CLWINOConvolution

}  // namespace gpu
}  // namespace ud
}  // namespace enn
