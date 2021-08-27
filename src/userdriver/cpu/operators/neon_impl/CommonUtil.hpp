#pragma once

#include "userdriver/common/operator_interfaces/common/Includes.hpp"
#include "userdriver/cpu/common/NEONTensor.hpp"

namespace enn {
namespace ud {
namespace cpu {

static Status evalFloat(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output, int32_t axis, bool max) {
    auto input_tensor = std::static_pointer_cast<NEONTensor<float>>(input);
    auto output_tensor = std::static_pointer_cast<NEONTensor<int>>(output);
    float* input_data = input_tensor->getBufferPtr();
    int* output_data = output_tensor->getBufferPtr();

    Dim4 input_dim = input_tensor->getDim();
    uint32_t dim = getDim(input_dim, axis);
    int axis_dist = 1;
    for (int i = axis + 1; i < 4; ++i) {
        axis_dist *= getDim(input_dim, i);
    }
    int num = 0;
    if (dim == 0) {
        return Status::FAILURE;
    } else {
        num = input_tensor->getTotalSizeFromDims() / dim;
    }
    std::vector<std::pair<float, int>> vec(dim);
    for (int i = 0; i < num; ++i) {
        for (uint32_t j = 0; j < dim; ++j) {
            vec[j] = std::make_pair(input_data[(i / axis_dist * dim + j) * axis_dist + i % axis_dist], j);
        }
        std::sort(vec.begin(), vec.end(), std::greater<std::pair<float, int>>());
        output_data[(i / axis_dist) * axis_dist + i % axis_dist] = max ? vec[0].second : vec.back().second;
    }

    return Status::SUCCESS;
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
