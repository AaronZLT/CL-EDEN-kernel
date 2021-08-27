#pragma once

#include <set>
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/common/operator_interfaces/common/Error.hpp"
#include "userdriver/gpu/common/CLParameter.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"

namespace enn {
namespace ud {
namespace gpu {

struct MeanParameters : public Parameters {
    bool androidNN = false;
    bool keep_dims = false;
};
class CLMean {
public:
    CLMean(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);
    Status initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                      const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                      const std::shared_ptr<Parameters> parameters);
    Status execute();
    Status release();

private:
    // 1. Runtime context
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;

    // 2. Operator resource
    std::shared_ptr<CLTensor> input_tensor_;
    std::shared_ptr<CLTensor> output_tensor_;
    std::shared_ptr<ITensor> axis_tensor_;
    std::shared_ptr<MeanParameters> parameters_;

    // 3. Operator kernels
    std::shared_ptr<struct _cl_kernel> kernel_sum_;
    std::shared_ptr<struct _cl_kernel> kernel_mean_;
    std::shared_ptr<struct _cl_kernel> kernel_mean_feature_opt_;
    std::shared_ptr<struct _cl_kernel> kernel_int8_to_int_;

    // 4. Operator variables
    std::vector<int32_t> axis_;
    std::set<uint32_t> resolved_axis_;
    std::shared_ptr<int32_t> tmp_axis_;
    NDims input_dim_;
    uint32_t num_axis_;
    uint32_t in_num_dim_;
    uint32_t out_size_;
    uint32_t num_resolved_axis_;
    bool is_global_ave_pool_;
    bool axis_as_input_;
    // 5. Operator executable fuctions
    void prepare_mean(const std::shared_ptr<ITensor> axis, std::shared_ptr<ITensor> output);
    bool isFitFeatureMeanOpt(int num_elements_in_axis);

    // 6. Else
    void print_parameter();
};

}  // namespace gpu
}  // namespace ud
}  // namespace enn