#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/common/operator_interfaces/common/Error.hpp"
#include "userdriver/gpu/common/CLParameter.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"

namespace enn {
namespace ud {
namespace gpu {
namespace {
class CLConcatTextureImpl;
}  // namespace

struct ConcatParameters : public Parameters {
    int32_t axis = 0;
    bool androidNN = false;
    bool isNCHW = true;
    StorageType storage_type = StorageType::BUFFER;
    ActivationInfo activation_info = ActivationInfo();
};

class CLConcat {
  public:
    CLConcat(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);
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
    std::vector<std::shared_ptr<CLTensor>> input_tensors_;
    std::shared_ptr<CLTensor> output_tensor_;
    std::shared_ptr<ConcatParameters> parameters_;

    // 3. Operator kernels
    std::shared_ptr<struct _cl_kernel> kernel_;

    // 4. execute functions
    Status concatFloat();
    Status concatQuant();

  private:
    Status concatAxis0(const bool &is_quant = false, const int32_t &qmin = 0, const int32_t &qmax = 0);
    Status concatAxis1(const bool &is_quant = false, const int32_t &qmin = 0, const int32_t &qmax = 0);
    Status concatAxis2(const bool &is_quant = false, const int32_t &qmin = 0, const int32_t &qmax = 0);
    Status concatAxis3(const bool &is_quant = false, const int32_t &qmin = 0, const int32_t &qmax = 0);

  private:
    friend class CLConcatTextureImpl;
    std::shared_ptr<CLConcatTextureImpl> texture_impl_;
};

}  // namespace gpu
}  // namespace ud
}  // namespace enn
