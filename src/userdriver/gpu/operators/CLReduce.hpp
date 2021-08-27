#pragma once

#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/common/operator_interfaces/common/Error.hpp"
#include "userdriver/gpu/common/CLParameter.hpp"
#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/gpu/operators/CLDeQuantization.hpp"
#include "userdriver/gpu/operators/CLQuantization.hpp"

namespace enn {
namespace ud {
namespace gpu {

struct ReduceParameters : public Parameters {
    bool keep_dims = false;
    Reducer reducer = Reducer::SIZE;
};

class CLReduce {
public:
    CLReduce(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision);
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
    std::shared_ptr<CLTensor> axis_tensor_;
    std::shared_ptr<CLTensor> output_tensor_;
    std::shared_ptr<CLTensor> map_tensor_;
    std::shared_ptr<ReduceParameters> parameters_;

    // 3. Operator kernels
    std::shared_ptr<struct _cl_kernel> reduce_kernel_;
    std::shared_ptr<struct _cl_kernel> output_kernel_;

    // 4. Other operator and parameter
    std::shared_ptr<CLQuantization> quantization_;
    std::shared_ptr<CLDeQuantization> dequantization_;
    std::vector<int32_t> axis_;

    std::string kernel_str(Reducer reducer) {
        switch (reducer) {
        case Reducer::SUM: return "SUM_reduce";
        case Reducer::MIN: return "MIN_reduce";
        case Reducer::MAX: return "MAX_reduce";
        case Reducer::PROD: return "PROD_reduce";
        case Reducer::ALL: return "ALL_reduce";
        case Reducer::ANY: return "ANY_reduce";
        default: return "reduce";
        }
    }

    // 5. Execute functions
    Status eval(const std::shared_ptr<CLTensor> input, std::shared_ptr<CLTensor> output);

};  // class CLReduce

}  // namespace gpu
}  // namespace ud
}  // namespace enn
