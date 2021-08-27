#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/common/operator_interfaces/common/Error.hpp"
#include "userdriver/gpu/common/CLParameter.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLActivation.hpp"

namespace enn {
namespace ud {
namespace gpu {

class CLAveragepool {
  public:
    CLAveragepool(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);
    Status initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                      const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                      const std::shared_ptr<Parameters> parameters);
    Status execute();
    Status release();

  private:
    // 1. Runtime context
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;
    int32_t kernel_max_work_group_size_;

    // 2. Operator resource
    std::shared_ptr<CLTensor> input_tensor_;
    std::shared_ptr<CLTensor> output_tensor_;
    std::shared_ptr<CLTensor> nchw_input_tensor_;
    std::shared_ptr<CLTensor> nchw_output_tensor_;
    std::shared_ptr<Pool2DParameters> parameters_;
    std::shared_ptr<CLActivation> cl_activation_;

    // 3. Operator kernels
    std::shared_ptr<struct _cl_kernel> kernel_;

    // 4. execute functions
    Status set_kernel();
    Status eval_nchw(const std::shared_ptr<CLTensor> input_tensor, std::shared_ptr<CLTensor> output_tensor);
    Status averagepool_float(const std::shared_ptr<CLTensor> input_tensor, std::shared_ptr<CLTensor> output_tensor);
    Status averagepool_quant(const std::shared_ptr<CLTensor> input_tensor, std::shared_ptr<CLTensor> output_tensor);
    Status execute_texture2d_float(const std::shared_ptr<CLTensor> input_tensor, std::shared_ptr<CLTensor> output_tensor);
    void get_workgroup(const int *grid, int max_size, int *best_work_group) {
        int wg_z = GetBiggestDividerWithPriority(grid[2], 8);
        int wg_xy_size = max_size / wg_z;
        int wg_x = std::min(IntegralDivideRoundUp(grid[0], 2), wg_xy_size);
        int wg_y = std::min(wg_xy_size / wg_x, grid[1]);
        best_work_group[0] = wg_x;
        best_work_group[1] = wg_y;
        best_work_group[2] = wg_z;
    }
};

}  // namespace gpu
}  // namespace ud
}  // namespace enn
