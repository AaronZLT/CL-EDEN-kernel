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
class CLResizeBilinearTextureImpl;
struct ResizeBilinearParameters : public Parameters {
    int32_t new_height = 0;
    int32_t new_width = 0;
    bool align_corners = false;
    bool half_pixel_centers = false;
    bool androidNN = false;
    bool isNCHW = true;
    StorageType storage_type = StorageType::BUFFER;
    ComputeType compute_type = ComputeType::TFLite;
};
}  // namespace

class CLResizeBilinear {
  public:
    CLResizeBilinear(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);
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
    std::shared_ptr<ResizeBilinearParameters> parameters_;

    // 3. Operator kernels
    std::shared_ptr<struct _cl_kernel> kernel_;

    // 4. Kernel related parameters
    float height_scale_;
    float width_scale_;
    bool use_resize_bilinear_32_to_512_opt_;

    // 5. for androidNN
    std::shared_ptr<CLLayoutConvert> layout_convert_nhwc2nchw_;
    std::shared_ptr<CLLayoutConvert> layout_convert_nchw2nhwc_;

    // 6. for CTS
    std::shared_ptr<float> float_input_data_;
    std::shared_ptr<float> float_output_data_;
    std::shared_ptr<half_float::half> half_input_data_;
    std::shared_ptr<half_float::half> half_output_data_;
    std::shared_ptr<int8_t> int8_input_data_;
    std::shared_ptr<int8_t> int8_output_data_;
    std::shared_ptr<uint8_t> uint8_input_data_;
    std::shared_ptr<uint8_t> uint8_output_data_;

    Status execute_for_cts();
    template <typename T>
    Status
    execute_on_cpu(std::shared_ptr<T> tmp_input_data, std::shared_ptr<T> tmp_output_data, Dim4 input_dim, Dim4 output_dim);

  private:
    friend class CLResizeBilinearTextureImpl;
    std::shared_ptr<CLResizeBilinearTextureImpl> texture_impl_;
};

}  // namespace gpu
}  // namespace ud
}  // namespace enn
