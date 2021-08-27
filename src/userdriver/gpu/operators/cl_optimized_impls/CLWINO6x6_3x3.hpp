/*
 * Copyright (C) 2020 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

/**
 * @file    CLWINO6x6_3x3.hpp
 * @brief
 * @details
 * @version
 */
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

class CLWINO6x6_3x3 {
  public:
    CLWINO6x6_3x3(const std::shared_ptr<CLRuntime> runtime,
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
                      const bool &androidNN = false);

    Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output);

  private:
    typedef struct {
        uint32_t num_output_;
        uint32_t pad_right_;
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

    bool androidNN_;
    bool weights_as_input_;
    int tile_size_;
    int aligned_width_;

    ConvDescriptor conv_descriptor_;

    Status convertWeight();

    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;

    std::shared_ptr<CLTensor> input_tm_buffer_;
    std::shared_ptr<CLTensor> dot_buffer_;
    std::shared_ptr<CLTensor> weight_nchw_;
    std::shared_ptr<CLTensor> weight_tm_buffer_;
    std::shared_ptr<CLTensor> weight_tensor_;
    Dim4 weight_dim_;

    std::shared_ptr<CLTensor> bias_tensor_;
    std::shared_ptr<struct _cl_kernel> weight_tm_kernel_;
    /* this step does input padding, tiling, and internal transpose */
    std::shared_ptr<struct _cl_kernel> input_tm_kernel_;
    /* this step does matrix multipling, tile transposing and deppading */
    std::shared_ptr<struct _cl_kernel> dot_multiply_kernel_;
    /* optimized kernel, but only support non-group and squared feature */
    std::shared_ptr<struct _cl_kernel> output_tm_kernel_;

    bool top_channel_need_aligned_;
    int branch_number_;
    int speed_up_;
    bool bifrost_speedup_;
    uint32_t coalescing_feature_height_;
    uint32_t compute_output_number_;
    uint32_t splite_number_;
    uint32_t wgrad_stride_;
    uint32_t wgrad_tile_size_;
    uint32_t wgrad_tile_height_;
    uint32_t wgrad_tile_width_;
    uint32_t wgrad_tile_total_;
    uint32_t aligned_tiles_;

    int aligned_inch_;
    int aligned_outch_;
    int aligned_pixel_;

    Dim4 input_dim_;
    Dim4 output_dim_;

    ActivationInfo activation_info_;
};  // class CLWINO6x6_3x3

}  // namespace gpu
}  // namespace ud
}  // namespace enn
