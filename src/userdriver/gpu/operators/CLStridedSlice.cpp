#include "CLStridedSlice.hpp"

namespace enn {
namespace ud {
namespace gpu {
namespace {
constexpr int INPUT_INDEX = 0;
constexpr int BEGIN_INDEX = 1;
constexpr int END_INDEX = 2;
constexpr int STRIDES_INDEX = 3;
constexpr int OUTPUT_INDEX = 0;
}  // namespace

CLStridedSlice::CLStridedSlice(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    ENN_DBG_PRINT("CLStridedSlice is created\n");
    input_tensor_ = nullptr;
    begin_tensor_ = nullptr;
    end_tensor_ = nullptr;
    strides_tensor_ = nullptr;
    output_tensor_ = nullptr;
    parameters_ = nullptr;
    kernel_ = nullptr;
}

Status CLStridedSlice::initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                                  const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                                  const std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLStridedSlice::initialize() is called\n");

    input_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(INPUT_INDEX));
    begin_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(BEGIN_INDEX));
    end_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(END_INDEX));
    strides_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(STRIDES_INDEX));
    output_tensor_ = std::static_pointer_cast<CLTensor>(output_tensors.at(OUTPUT_INDEX));
    parameters_ = std::static_pointer_cast<StridedSliceParameters>(parameters);
    CHECK_AND_RETURN_ERR(nullptr == parameters_, Status::FAILURE, "CLStridedSlice must have parameters\n");

    parameters_->begin.resize(begin_tensor_->getTotalSizeFromDims());
    begin_tensor_->readData(parameters_->begin.data());
    parameters_->end.resize(end_tensor_->getTotalSizeFromDims());
    end_tensor_->readData(parameters_->end.data());
    parameters_->strides.resize(strides_tensor_->getTotalSizeFromDims());
    strides_tensor_->readData(parameters_->strides.data());

    const int32_t kMaxDimensions = 4;
    for (int32_t i = parameters_->begin.size(); i < kMaxDimensions; ++i) {
        parameters_->begin.push_back(0);
        parameters_->end.push_back(1);
        parameters_->strides.push_back(1);
    }

    start_b_ = startForAxis(parameters_->begin_mask, parameters_->begin, parameters_->strides, input_tensor_->getDim(), 0);
    stop_b_ = stopForAxis(parameters_->end_mask,
                          parameters_->shrink_axis_mask,
                          parameters_->end,
                          parameters_->strides,
                          input_tensor_->getDim(),
                          0,
                          start_b_);
    start_d_ = startForAxis(parameters_->begin_mask, parameters_->begin, parameters_->strides, input_tensor_->getDim(), 1);
    stop_d_ = stopForAxis(parameters_->end_mask,
                          parameters_->shrink_axis_mask,
                          parameters_->end,
                          parameters_->strides,
                          input_tensor_->getDim(),
                          1,
                          start_d_);
    start_h_ = startForAxis(parameters_->begin_mask, parameters_->begin, parameters_->strides, input_tensor_->getDim(), 2);
    stop_h_ = stopForAxis(parameters_->end_mask,
                          parameters_->shrink_axis_mask,
                          parameters_->end,
                          parameters_->strides,
                          input_tensor_->getDim(),
                          2,
                          start_h_);
    start_w_ = startForAxis(parameters_->begin_mask, parameters_->begin, parameters_->strides, input_tensor_->getDim(), 3);
    stop_w_ = stopForAxis(parameters_->end_mask,
                          parameters_->shrink_axis_mask,
                          parameters_->end,
                          parameters_->strides,
                          input_tensor_->getDim(),
                          3,
                          start_w_);

    if (input_tensor_->getDataType() == DataType::UINT8) {
        cpu_input_data_u_ = make_shared_array<uint8_t>(input_tensor_->getTotalSizeFromDims());
        cpu_output_data_u_ = make_shared_array<uint8_t>(output_tensor_->getTotalSizeFromDims());
    } else if (input_tensor_->getDataType() == DataType::HALF) {
        cpu_input_data_h_ = make_shared_array<half_float::half>(input_tensor_->getTotalSizeFromDims());
        cpu_output_data_h_ = make_shared_array<half_float::half>(output_tensor_->getTotalSizeFromDims());
    } else {
        cpu_input_data_f_ = make_shared_array<float>(input_tensor_->getTotalSizeFromDims());
        cpu_output_data_f_ = make_shared_array<float>(output_tensor_->getTotalSizeFromDims());
    }

    if (parameters_->androidNN) {
        NDims input_dim = input_tensor_->getDims();
        NDims output_dim;
        for (int32_t idx = 0; idx < static_cast<int32_t>(input_dim.size()); idx++) {
            int32_t dim = input_dim[idx];
            int32_t stride = parameters_->strides[idx];

            bool positive_stride = stride > 0;
            int32_t begin_arg = parameters_->begin_mask & (1 << idx)
                                    ? positive_stride ? 0 : dim - 1
                                    : clampedIndex(parameters_->begin[idx], dim, positive_stride);
            int32_t end_arg = parameters_->end_mask & (1 << idx) ? positive_stride ? dim : -1
                                                                 : clampedIndex(parameters_->end[idx], dim, positive_stride);

            // This is valid for both positive and negative strides
            int32_t out_dim = ceil((end_arg - begin_arg) / static_cast<float>(stride));
            out_dim = out_dim < 0 ? 0 : static_cast<uint32_t>(out_dim);

            if (!(parameters_->shrink_axis_mask & (1 << idx))) {
                output_dim.push_back(out_dim);
            }
        }

        if (!isDimsSame(output_tensor_->getDims(), output_dim)) {
            output_tensor_->reconfigureDimsAndBuffer(output_dim);
        }
    }

    Status status = runtime_->setKernel(&kernel_, "stride_slice", precision_);
    CHECK_AND_RETURN_ERR(Status::SUCCESS != status, status, "setKernel stride_slice failure\n");

    return Status::SUCCESS;
}

Status CLStridedSlice::execute() {
    ENN_DBG_PRINT("CLStridedSlice::execute() is called\n");

    auto input_dim = input_tensor_->getDim();
    if (input_tensor_->getDataType() == DataType::UINT8) {
        input_tensor_->readData(cpu_input_data_u_.get());
        strideSlice<uint8_t>(cpu_input_data_u_.get(), cpu_output_data_u_.get(), input_dim);
        output_tensor_->writeData(cpu_output_data_u_.get());
    } else if (input_tensor_->getDataType() == DataType::HALF) {
        input_tensor_->readData(cpu_input_data_h_.get());
        strideSlice<half_float::half>(cpu_input_data_h_.get(), cpu_output_data_h_.get(), input_dim);
        output_tensor_->writeData(cpu_output_data_h_.get());
    } else {
        input_tensor_->readData(cpu_input_data_f_.get());
        strideSlice<float>(cpu_input_data_f_.get(), cpu_output_data_f_.get(), input_dim);
        output_tensor_->writeData(cpu_output_data_f_.get());
    }

    return Status::SUCCESS;
}

Status CLStridedSlice::release() {
    ENN_DBG_PRINT("CLStridedSlice::release() is called");
    return Status::SUCCESS;
}

template <typename T> void CLStridedSlice::strideSlice(T *in_data, T *out_data, Dim4 &input_dim) {
    for (int in_b = start_b_; !loopCondition(in_b, stop_b_, parameters_->strides[0]); in_b += parameters_->strides[0]) {
        for (int in_d = start_d_; !loopCondition(in_d, stop_d_, parameters_->strides[1]); in_d += parameters_->strides[1]) {
            for (int in_h = start_h_; !loopCondition(in_h, stop_h_, parameters_->strides[2]);
                 in_h += parameters_->strides[2]) {
                for (int in_w = start_w_; !loopCondition(in_w, stop_w_, parameters_->strides[3]);
                     in_w += parameters_->strides[3]) {
                    if (in_b >= 0 && in_d >= 0 && in_h >= 0 && in_w >= 0) {
                        int in_offset = in_b * input_dim.c * input_dim.h * input_dim.w + in_d * input_dim.h * input_dim.w +
                                        in_h * input_dim.w + in_w;
                        *out_data++ = in_data[in_offset];
                    }
                }
            }
        }
    }
}

int32_t CLStridedSlice::startForAxis(const int32_t &begin_mask,
                                     const std::vector<int32_t> &start_indices,
                                     const std::vector<int32_t> &strides,
                                     const Dim4 &input_shape,
                                     const int32_t &axis) {
    // Begin with the specified index
    int start = start_indices[axis];
    // begin_mask override
    if (begin_mask & 1 << axis) {
        if (strides[axis] > 0) {
            // Forward iteration - use the first element. These values will get
            // clamped below (Note: We could have set them to 0 and axis_size-1, but
            // use lowest() and max() to maintain symmetry with StopForAxis())
            start = std::numeric_limits<int>::lowest();
        } else {
            // Backward iteration - use the last element.
            start = std::numeric_limits<int>::max();
        }
    }
    // Handle negative indices
    int axis_size = getDim(input_shape, axis);
    if (start < 0) {
        start += axis_size;
    }
    // Clamping
    start = clamp(start, 0, axis_size - 1);
    return start;
}

int32_t CLStridedSlice::stopForAxis(const int32_t &end_mask,
                                    const int32_t &shrink_axis_mask,
                                    const std::vector<int32_t> &stop_indices,
                                    const std::vector<int32_t> &strides,
                                    const Dim4 &input_shape,
                                    const int32_t axis,
                                    const int32_t &start_for_axis) {
    // Begin with the specified index
    const bool shrink_axis = shrink_axis_mask & (1 << axis);
    int stop = stop_indices[axis];
    // When shrinking an axis, the end position does not matter. Always use
    // start_for_axis + 1 to generate a length 1 slice, since start_for_axis has
    // already been adjusted for negative indices.
    if (shrink_axis) {
        stop = start_for_axis + 1;
    }
    // end_mask override
    if (end_mask & (1 << axis)) {
        if (strides[axis] > 0) {
            // Forward iteration - use the last element. These values will get
            // clamped below
            stop = std::numeric_limits<int>::max();
        } else {
            // Backward iteration - use the first element.
            stop = std::numeric_limits<int>::lowest();
        }
    }
    // Handle negative indices
    const int axis_size = getDim(input_shape, axis);
    if (stop < 0) {
        stop += axis_size;
    }
    // Clamping
    // Because the end index points one past the last element, we need slightly
    // different clamping ranges depending on the direction.
    if (strides[axis] > 0) {
        // Forward iteration
        stop = clamp(stop, 0, axis_size);
    } else {
        // Backward iteration
        stop = clamp(stop, -1, axis_size - 1);
    }
    return stop;
}

int32_t CLStridedSlice::clamp(const int32_t &v, const int32_t &lo, const int32_t &hi) {
    if (hi < v) {
        return hi;
    }
    if (v < lo) {
        return lo;
    }
    return v;
}

bool CLStridedSlice::loopCondition(const int32_t &index, const int32_t &stop, const int32_t &stride) {
    // True when we have reached the end of an axis and should loop.
    return stride > 0 ? index >= stop : index <= stop;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
