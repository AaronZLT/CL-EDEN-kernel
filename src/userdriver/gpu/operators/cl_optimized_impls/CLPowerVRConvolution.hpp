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
 * @file    CLPowerVRConvolution.hpp
 * @brief
 * @details
 * @version
 */

#pragma once

#include <float.h>
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/common/operator_interfaces/common/ActivationInfo.hpp"
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/ITensor.hpp"

namespace enn {
namespace ud {
namespace gpu {

class CLPowerVRConvolution {
  public:
    CLPowerVRConvolution(const std::shared_ptr<CLRuntime> runtime,
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
                      bool weights_as_input = false,
                      bool mergeadd = false);

    Status execute(const std::vector<std::shared_ptr<ITensor>> input, std::shared_ptr<ITensor> output);

  private:
    Status alignWeight();

    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;

    int32_t kernel_max_work_group_size_;
    Dim4 input_dim_;
    Dim4 output_dim_;
    Dim4 convert_out_dim_;
    Dim2 stride_;
    Dim2 kernel_;
    std::shared_ptr<CLTensor> convert_output_;
    ActivationInfo activation_info_;

    typedef struct {
        uint32_t num_output_;
        uint32_t pad_right_;  // currently, only padright=padleft and padtop=padbottom is supported
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

    std::string _kernel_name_;
    void generateKernelName(ActivationInfo activation) {
        src_depth_loop_size_ = 1;
        bool x_kernel_is_1 = false;
        bool y_kernel_is_1 = false;

        if (conv_descriptor_.kernel_width_ == 1 && conv_descriptor_.stride_width_ == 1 &&
            conv_descriptor_.dilation_.w == 1 && conv_descriptor_.pad_left_ == 0 && conv_descriptor_.pad_right_ == 0) {
            x_kernel_is_1 = true;
        }
        if (conv_descriptor_.kernel_height_ == 1 && conv_descriptor_.stride_height_ == 1 &&
            conv_descriptor_.dilation_.h == 1 && conv_descriptor_.pad_bottom_ == 0 && conv_descriptor_.pad_top_ == 0) {
            y_kernel_is_1 = true;
        }
        is1x1_ = x_kernel_is_1 && y_kernel_is_1;

        const int32_t dst_depth = IntegralDivideRoundUp(conv_descriptor_.filter_->getDim().n, 4);
        const int32_t src_depth = IntegralDivideRoundUp(conv_descriptor_.filter_->getDim().c, 4);

        int32_t task_size = output_dim_.w * output_dim_.n * output_dim_.h * dst_depth;
        int32_t block_size = GetRecommendedBlockSizeForConv(task_size);

        if (!is1x1_) {
            block_size = std::min(block_size, 4);
            if (block_size == 4) {
                block_size_x_ = 2;
                block_size_y_ = 1;
                block_size_z_ = 2;
            } else if (block_size == 2) {
                block_size_x_ = 2;
                block_size_y_ = 1;
                block_size_z_ = 1;
            } else {
                block_size_x_ = 1;
                block_size_y_ = 1;
                block_size_z_ = 1;
            }
            if (src_depth % 2 == 0 && block_size <= 2) {
                src_depth_loop_size_ = 2;
            }
        } else {
            if (block_size == 4) {
                if (dst_depth == 1 || dst_depth == 3) {
                    block_size_x_ = 2;
                    block_size_y_ = 2;
                    block_size_z_ = 1;
                } else {
                    block_size_x_ = 2;
                    block_size_y_ = 1;
                    block_size_z_ = 2;
                }
            } else if (block_size == 2) {
                block_size_x_ = 2;
                block_size_y_ = 1;
                block_size_z_ = 1;
            } else {
                block_size_x_ = 1;
                block_size_y_ = 1;
                block_size_z_ = 1;
            }

            if (src_depth % 2 == 0 && block_size <= 2) {
                src_depth_loop_size_ = 2;
            }
            if (src_depth % 4 == 0 && block_size == 1 && precision_ == PrecisionType::FP16) {
                src_depth_loop_size_ = 4;
            }
        }

        _kernel_name_ = "powervr_is1x1" + std::to_string(is1x1_) + "_srcdepth" + std::to_string(src_depth_loop_size_) +
                        "_block" + std::to_string(block_size_x_) + std::to_string(block_size_y_) +
                        std::to_string(block_size_z_);
        if (mergeadd_)
            _kernel_name_ = "MERGEADD" + _kernel_name_;

        if (activation.isEnabled()) {
            switch (activation.activation()) {
            case ActivationInfo::ActivationType::RELU: _kernel_name_ = "RELU" + _kernel_name_; break;
            case ActivationInfo::ActivationType::RELU6:
                if (precision_ == PrecisionType::FP16) {
                    _kernel_name_ = "RELU6" + _kernel_name_;
                } else {
                    printf("activation fsue is not used for FP32\n");
                }
                break;
            default: break;
            }
        }
    }

    int GetRecommendedBlockSizeForConv(int task_size) {
        const float task_size_per_cu = (float)task_size / runtime_->getComputeUnitsCount();
        int block_size = 1;
        float threshold_1 = FLT_MAX;
        float threshold_2 = FLT_MAX;
        float threshold_4 = FLT_MAX;

        switch (precision_) {
        case PrecisionType::FP16:
            threshold_1 = 256.0f;
            threshold_2 = 256.0f * 6.0f;
            threshold_4 = 256.0f * 16.0f;
            break;
        case PrecisionType::FP32:
            threshold_1 = 256.0f;
            threshold_2 = 256.0f * 12.0f;
            break;
        default:
            break;
        }

        if (task_size_per_cu <= threshold_1) {
            block_size = 1;
        } else if (task_size_per_cu <= threshold_2) {
            block_size = 2;
        } else {
            block_size = 4;
        }
        return block_size;
    }

    void GetBestWorkGroupConv(const int *grid, int *best_work_group) {
        int max_z_size = grid[0] > 48 ? 8 : 4;
        max_z_size = std::min(max_z_size, (int)runtime_->getMaxWorkGroupSize()[2]);
        GetWorkGroupConv(grid, kernel_max_work_group_size_, max_z_size, best_work_group);
    }

    void GetWorkGroupConv(const int *grid, int max_size, int max_z_size, int *best_work_group) {
        int wg_z = GetBiggestDivider(grid[2], max_z_size);
        int wg_xy_size = std::min(256, max_size) / wg_z;
        int wg_x = std::min(grid[0], wg_xy_size);
        int wg_y = std::min(wg_xy_size / wg_x, grid[1]);
        if (wg_y == grid[1] && grid[1] % 2 == 0) {
            wg_y = grid[1] / 2;
        }
        best_work_group[0] = wg_x;
        best_work_group[1] = wg_y;
        best_work_group[2] = wg_z;
    }

    int GetBiggestDivider(int number, int max_divider) {
        for (int i = max_divider; i != 0; i--) {
            if (number % i == 0) {
                return i;
            }
        }
        return 1;
    }

    void GetGridSize(int *grid) const {
        const int grid_x = IntegralDivideRoundUp(output_dim_.w * output_dim_.n, block_size_x_);
        const int grid_y = IntegralDivideRoundUp(output_dim_.h, block_size_y_);
        int slices = IntegralDivideRoundUp(output_dim_.c, 4);
        const int grid_z = IntegralDivideRoundUp(slices, block_size_z_);

        grid[0] = IntegralDivideRoundUp(grid_x, work_group_size_[0]) * work_group_size_[0];
        grid[1] = IntegralDivideRoundUp(grid_y, work_group_size_[1]) * work_group_size_[1];
        grid[2] = IntegralDivideRoundUp(grid_z, work_group_size_[2]) * work_group_size_[2];
    }

    ConvDescriptor conv_descriptor_;
    bool weights_as_input_;

    std::shared_ptr<CLTensor> filter_nchw_;
    std::shared_ptr<CLTensor> converted_filter_;
    std::shared_ptr<struct _cl_kernel> align_weight_kernel_;
    std::shared_ptr<struct _cl_kernel> powervr_kernel_;

    bool mergeadd_;
    int32_t block_size_x_, block_size_y_, block_size_z_;
    bool is1x1_ = false;
    int32_t src_depth_loop_size_ = 0;
    int work_group_size_[3];
    int grid_size_[3];
    int src_x_;
    int src_y_;
    int src_z_;
    int src_w_;
    int dst_x_;
    int dst_y_;
    int dst_z_;
    int dst_w_;
};  // class CLPowerVRConvolution

}  // namespace gpu
}  // namespace ud
}  // namespace enn
