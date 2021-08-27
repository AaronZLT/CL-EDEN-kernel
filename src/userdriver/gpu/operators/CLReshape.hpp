#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/common/operator_interfaces/common/Error.hpp"
#include "userdriver/gpu/common/CLParameter.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLActivation.hpp"
#include "userdriver/gpu/operators/CLLayoutConvert.hpp"

namespace enn {
namespace ud {
namespace gpu {
namespace {
struct ReshapeParameters : public Parameters {
    std::vector<int32_t> new_shape;
    bool androidNN = false;
    bool isNCHW = true;
    ComputeType compute_type = ComputeType::TFLite;
};
}  // namespace

class CLReshape {
  public:
    CLReshape(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);
    Status initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                      const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                      const std::shared_ptr<Parameters> parameters);
    Status execute();
    Status release();

  private:
    Status compute_output_shape();

  private:
    // 1. Runtime context
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;

    // 2. Operator resource
    std::shared_ptr<CLTensor> input_tensor_;
    std::shared_ptr<CLTensor> shape_tensor_;
    std::shared_ptr<CLTensor> output_tensor_;
    std::shared_ptr<ReshapeParameters> parameters_;

    // 3. Operator kernels
    std::shared_ptr<struct _cl_kernel> kernel_;

    // 4. execute functions
    Status reshapeFloat();
    Status reshapeQuant();
};

}  // namespace gpu
}  // namespace ud
}  // namespace enn
