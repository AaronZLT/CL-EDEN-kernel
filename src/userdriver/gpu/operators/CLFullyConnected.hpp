#pragma once

#include "userdriver/gpu/common/CLParameter.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/common/operator_interfaces/common/ActivationInfo.hpp"
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/gpu/operators/CLActivation.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CL8x1FullyConnected.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CL8X1GEMVFullyConnected.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLBaseFullyConnected.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLDirectFullyConnected.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLFCTFLiteTexture2D.hpp"

namespace enn {
namespace ud {
namespace gpu {

#define FULLY_CONNECTED_OPT_BATCH (100)

struct FullyConnectedParameters : public Parameters {
    bool androidNN = false;
    StorageType storage_type = StorageType::BUFFER;
    std::shared_ptr<ActivationInfo> activation_info = std::make_shared<ActivationInfo>();
};

class CLFullyConnected {
  public:
    enum FullyConnectedKernelType { DIRECT, BASE, FC8X1, FC8X1GEMV, TFLITE_TEXTURE2D };
    CLFullyConnected(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);
    Status initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                      const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                      const std::shared_ptr<Parameters> parameters);
    Status execute();
    Status release();

  private:
    std::shared_ptr<CLRuntime> runtime_;
    std::shared_ptr<ITensor> input_;
    std::shared_ptr<ITensor> output_;
    std::shared_ptr<ITensor> weight_;
    std::shared_ptr<ITensor> bias_;
    FullyConnectedKernelType fc_kernel_type_;

    std::shared_ptr<CLBaseFullyConnected> base_fullyconnected_;
    std::shared_ptr<CL8x1FullyConnected> cl8x1_fullyconnected_;
    std::shared_ptr<CL8X1GEMVFullyConnected> gemv_fullyconnected_;
    std::shared_ptr<CLDirectFullyConnected> direct_fullyconnected_;
    std::shared_ptr<CLFCTFLiteTexture2D> tflite_texture2d_fc_;
    std::shared_ptr<FullyConnectedParameters> parameters_;
    PrecisionType precision_;
    ActivationInfo activation_info_;
    std::shared_ptr<CLActivation> cl_activation_;

    NDims fc_2d_input_dims_;
    NDims fc_nd_input_dims_;
    bool isAndroidNN_;
    StorageType storage_type_;
    bool weights_as_input_ = false;
};  // class CLFullyConnected

}  // namespace gpu
}  // namespace ud
}  // namespace enn
