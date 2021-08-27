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
struct StridedSliceParameters : public Parameters {
    std::vector<int32_t> begin;
    std::vector<int32_t> end;
    std::vector<int32_t> strides;
    int32_t begin_mask = 0;
    int32_t end_mask = 0;
    int32_t ellipsis_mask = 0;
    int32_t new_axis_mask = 0;
    int32_t shrink_axis_mask = 0;
    bool androidNN = false;
};

class CLStridedSlice {
  public:
    CLStridedSlice(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);
    Status initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                      const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                      const std::shared_ptr<Parameters> parameters);
    Status execute();
    Status release();

  private:
    float offset(const Dim4 *dim, const int32_t &in0, const int32_t &in1, const int32_t &in2, const int32_t &in3);
    template <typename T> void strideSlice(T *in_data, T *out_data, Dim4 &input_dim);
    int32_t startForAxis(const int32_t &begin_mask,
                         const std::vector<int32_t> &start_indices,
                         const std::vector<int32_t> &strides,
                         const Dim4 &input_shape,
                         const int32_t &axis);
    int32_t stopForAxis(const int32_t &end_mask,
                        const int32_t &shrink_axis_mask,
                        const std::vector<int32_t> &stop_indices,
                        const std::vector<int32_t> &strides,
                        const Dim4 &input_shape,
                        const int32_t axis,
                        const int32_t &start_for_axis);
    int32_t clamp(const int32_t &v, const int32_t &lo, const int32_t &hi);
    bool loopCondition(const int32_t &index, const int32_t &stop, const int32_t &stride);

  private:
    // 1. Runtime context
    std::shared_ptr<CLRuntime> runtime_;
    PrecisionType precision_;

    // 2. Operator resource
    std::shared_ptr<CLTensor> input_tensor_;
    std::shared_ptr<CLTensor> begin_tensor_;
    std::shared_ptr<CLTensor> end_tensor_;
    std::shared_ptr<CLTensor> strides_tensor_;
    std::shared_ptr<CLTensor> output_tensor_;
    std::shared_ptr<StridedSliceParameters> parameters_;

    // 3. Operator kernels
    std::shared_ptr<struct _cl_kernel> kernel_;

    // 4.
    int32_t start_b_, stop_b_;
    int32_t start_h_, stop_h_;
    int32_t start_w_, stop_w_;
    int32_t start_d_, stop_d_;

    std::shared_ptr<float> cpu_input_data_f_ = nullptr;
    std::shared_ptr<float> cpu_output_data_f_ = nullptr;
    std::shared_ptr<uint8_t> cpu_input_data_u_ = nullptr;
    std::shared_ptr<uint8_t> cpu_output_data_u_ = nullptr;
    std::shared_ptr<half_float::half> cpu_input_data_h_ = nullptr;
    std::shared_ptr<half_float::half> cpu_output_data_h_ = nullptr;
};

}  // namespace gpu
}  // namespace ud
}  // namespace enn
