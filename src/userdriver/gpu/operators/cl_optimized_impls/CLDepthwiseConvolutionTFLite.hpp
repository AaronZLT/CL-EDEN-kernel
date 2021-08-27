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
 * @file    CLDepthwiseConvolutionTFLite.hpp
 * @brief
 * @details
 * @version
 */

#pragma once

#include "userdriver/common/operator_interfaces/common/ActivationInfo.hpp"
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"

namespace enn {
namespace ud {
namespace gpu {

class CLDepthwiseConvolutionTFLite {
public:
    CLDepthwiseConvolutionTFLite(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);
    Status initialize(const Dim4 &input_dim,
                      const Dim4 &output_dim,
                      const std::shared_ptr<ITensor> filter,
                      const std::shared_ptr<ITensor> bias,
                      const Dim2 &stride,
                      const Pad4 &pad,
                      const Dim2 &dilation,
                      const ActivationInfo &activate_info);
    Status execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output);
    Status RearrangeWeightsAndBiasesData(const std::shared_ptr<CLTensor> &weights,
                                         std::shared_ptr<CLTensor> &weights_converted);
    Status UploadWeightsAndBiases();
    Status release();

private:
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;
    std::shared_ptr<struct _cl_kernel> kernel_depthwise_tflite_;

    Dim4 input_dim_;
    Dim4 output_dim_;
    Pad4 pad_;
    Dim2 stride_;
    Dim2 dilation_;
    std::shared_ptr<ITensor> filter_;
    std::shared_ptr<ITensor> bias_;
    std::shared_ptr<CLTensor> weight_converted_;
    ActivationInfo activation_info_;

    int kernel_size_x_;
    int kernel_size_y_;
    int stride_size_x_;
    int stride_size_y_;
    int pad_size_x_;
    int pad_size_y_;
    int dilation_size_x_;
    int dilation_size_y_;
    int src_size_x_;
    int src_size_y_;
    int src_size_z_;
    int dst_size_x_;
    int dst_size_y_;
    int dst_size_z_;
    int grid_[3];
    int best_work_group_[3];

    int get_biggest_divider_with_priority(int number, int max_divider) {
        if (number % 8 == 0 && 8 <= max_divider) {
            return 8;
        }
        if (number % 4 == 0 && 4 <= max_divider) {
            return 4;
        }
        if (number % 2 == 0 && 2 <= max_divider) {
            return 2;
        }
        for (int i = max_divider; i != 0; i--) {
            if (number % i == 0) {
                return i;
            }
        }
        return 1;
    }

    void get_best_workgroup(const int *grid, int *best_group_size, int max_size) {
        int wg_z = get_biggest_divider_with_priority(grid[2], 8);
        int wg_xy_size = max_size / wg_z;
        int wg_x = std::min(IntegralDivideRoundUp(grid[0], 2), wg_xy_size);
        int wg_y = std::min(wg_xy_size / wg_x, grid[1]);
        best_group_size[0] = wg_x;
        best_group_size[1] = wg_y;
        best_group_size[2] = wg_z;
    }
};  // class CLDepthwiseConvolutionTFLite

}  // namespace gpu
}  // namespace ud
}  // namespace enn
