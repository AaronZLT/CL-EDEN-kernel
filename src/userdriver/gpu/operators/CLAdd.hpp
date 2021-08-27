#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/common/operator_interfaces/common/Error.hpp"
#include "userdriver/gpu/common/CLParameter.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLActivation.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLAddQuantized.hpp"

namespace enn {
namespace ud {
namespace gpu {
namespace {
class CLAddTextureImpl;
}

struct AddParameters : public Parameters {
    ActivationInfo activation_info = ActivationInfo();
    bool pot_scale_int16 = false;
    std::vector<float> coeff = {};
    bool androidNN = false;
    bool isNCHW = false;
    StorageType storage_type = StorageType::BUFFER;
};

class CLAdd {
  public:
    CLAdd(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);
    Status initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                      const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                      const std::shared_ptr<Parameters> parameters);
    Status execute();
    Status release();

  private:
    Status addFloat();

  private:
    // 1. Runtime context
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;

    // 2. Operator resource
    std::vector<std::shared_ptr<CLTensor>> input_tensors_;
    std::shared_ptr<CLTensor> output_tensor_;
    std::shared_ptr<AddParameters> parameters_;
    std::shared_ptr<CLActivation> cl_activation_;
    std::shared_ptr<CLAddQuantized> quantized_add_;

    // 3. Operator kernels
    std::shared_ptr<struct _cl_kernel> kernel_;
    std::shared_ptr<struct _cl_kernel> kernel_one_input_;

    // 4. for broadcast
    std::shared_ptr<CLTensor> input_broadcast_0_;
    std::shared_ptr<CLTensor> input_broadcast_1_;

    // 5. for select kernels
    bool is_vector_add_;

  private:
    friend class CLAddTextureImpl;
    std::shared_ptr<CLAddTextureImpl> texture_impl_;
};

}  // namespace gpu
}  // namespace ud
}  // namespace enn
