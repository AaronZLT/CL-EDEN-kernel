#pragma once

#include "userdriver/gpu/common/CLRuntime.hpp"
#include "userdriver/gpu/common/CLTensor.hpp"
#include "userdriver/common/operator_interfaces/common/Common.hpp"
#include "userdriver/common/operator_interfaces/interfaces/ITensor.hpp"

namespace enn {
namespace ud {
namespace gpu {

class CLMemAlign {
  public:
    CLMemAlign() = default;

    explicit CLMemAlign(const std::shared_ptr<CLRuntime> runtime);

    ~CLMemAlign();

    Status execute(const cl_mem input,
                   cl_mem output,
                   const PrecisionType &precisionType,
                   const int &src_total_count,
                   const int &src_unit_count,
                   const int &dst_total_count,
                   const int &dst_align_count,
                   const int &group);

  private:
    std::shared_ptr<CLRuntime> runtime_;
};  // class CLMemAlign

}  // namespace gpu
}  // namespace ud
}  // namespace enn
