
#include "CLMean.hpp"

namespace enn {
namespace ud {
namespace gpu {
namespace {
constexpr int INPUT_INDEX = 0;
constexpr int AXIS_INDEX = 1;
constexpr int OUTPUT_INDEX = 0;
}  // namespace

CLMean::CLMean(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    ENN_DBG_PRINT("CLMean is created");
    parameters_ = std::make_shared<MeanParameters>();
    input_tensor_ = nullptr;
    output_tensor_ = nullptr;
    axis_tensor_ = nullptr;
    kernel_sum_ = nullptr;
    kernel_mean_ = nullptr;
    kernel_mean_feature_opt_ = nullptr;
    kernel_int8_to_int_ = nullptr;
    tmp_axis_ = nullptr;
    num_axis_ = 0;
    in_num_dim_ = 4;
    out_size_ = 0;
    num_resolved_axis_ = 0;
    is_global_ave_pool_ = false;
    axis_as_input_ = false;
    axis_.clear();
    resolved_axis_.clear();
}

Status CLMean::initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                          const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                          const std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLMean::initialize() is called\n");
    CHECK_EXPR_RETURN_FAILURE(input_tensors.size() >= 2,
                              "CLMean at least has two input tensors: input tensor and axis tensor, here only %u\n",
                              input_tensors.size());
    CHECK_EXPR_RETURN_FAILURE(nullptr != parameters, "CLMean must have parameters\n");

    input_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(INPUT_INDEX));
    axis_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(AXIS_INDEX));
    int axisdims2 = axis_tensor_->getTotalSizeFromDims();
    int intensordims = input_tensor_->getTotalSizeFromDims();
    if (axisdims2 > intensordims) {
        ITensor::placeholder("ERROR #1: MEAN's inputs REVERSED", ColorType::RED);
        ITensor::placeholder("Please refe below", ColorType::RED);
        ITensor::placeholder("https://code.sec.samsung.net/jira/browse/ISSUEBOARD-10203", ColorType::RED);
        // input_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(AXIS_INDEX));
        // axis_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(INPUT_INDEX));
    }
    output_tensor_ = std::static_pointer_cast<CLTensor>(output_tensors.at(OUTPUT_INDEX));

    parameters_ = std::static_pointer_cast<MeanParameters>(parameters);
    axis_as_input_ = !axis_tensor_->is_const();

    in_num_dim_ = input_tensor_->getDims().size();
    input_dim_ = input_tensor_->getDims();
    out_size_ = output_tensor_->getTotalSizeFromDims();

    CLTensor::checknull(std::static_pointer_cast<CLTensor>(axis_tensor_), "axis_tensor_");
    CLTensor::checknull(output_tensor_, "output_tensor_");
    bool no_opt_set_kernel = true;
    bool keep_dims_ = parameters_->keep_dims;
    bool androidNN_ = parameters_->androidNN;
    if (!axis_as_input_) {
        prepare_mean(axis_tensor_, output_tensor_);
        const int num_elements_in_axis = input_tensor_->getTotalSizeFromDims() / out_size_;
        is_global_ave_pool_ = precision_ == PrecisionType::INT8 ? false : isFitFeatureMeanOpt(num_elements_in_axis);
        no_opt_set_kernel = false;
    }

    Status status;
    if (no_opt_set_kernel || is_global_ave_pool_) {
        if (precision_ == PrecisionType::INT8) {
            status = runtime_->setKernel(&kernel_mean_feature_opt_, "SIGNEDmean_feature_opt", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel mean_feature_opt failure\n");
        } else {
            status = runtime_->setKernel(&kernel_mean_feature_opt_, "mean_feature_opt", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel mean_feature_opt failure\n");
        }
    }
    if (no_opt_set_kernel || !is_global_ave_pool_) {
        status = runtime_->setKernel(&kernel_sum_, "sum_axis", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel kernel_sum_ failure\n");
        status = runtime_->setKernel(&kernel_mean_, "mean", precision_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel kernel_mean_ failure\n");
        if (precision_ == PrecisionType::UINT8) {
            status = runtime_->setKernel(&kernel_int8_to_int_, "quant8_to_int32", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel kernel_int8_to_int_ failure\n");
        } else if (precision_ == PrecisionType::INT8) {
            status = runtime_->setKernel(&kernel_int8_to_int_, "signed_quant8_to_int32", precision_);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel kernel_int8_to_int_ failure\n");
        }
    }
    print_parameter();

    return Status::SUCCESS;
}

Status CLMean::execute() {
    DEBUG_PRINT("CLMean::execute() is called");
    ITensor::placeholder("CLMean::execute");
    if (axis_as_input_) {
        CLTensor::checknull(std::static_pointer_cast<CLTensor>(axis_tensor_), "axis_tensor_");
        CLTensor::checknull(output_tensor_, "output_tensor_");
        prepare_mean(axis_tensor_, output_tensor_);
    }

    Status status = Status::FAILURE;
    auto in_tensor = std::static_pointer_cast<CLTensor>(input_tensor_);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output_tensor_);
    int num_elements_in_axis = input_tensor_->getTotalSizeFromDims() / out_size_;

    CLTensor::checknull(in_tensor, "in_tensor");
    CLTensor::checknull(output_tensor, "output_tensor");

    std::shared_ptr<CLTensor> input_tensor;
    if (!is_global_ave_pool_) {
        if (precision_ == PrecisionType::UINT8 || precision_ == PrecisionType::INT8) {
            input_tensor = std::make_shared<CLTensor>(runtime_, precision_, DataType::INT32, input_tensor_->getDim());
            status = runtime_->setKernelArg(kernel_int8_to_int_.get(), in_tensor->getDataPtr(), input_tensor->getDataPtr());
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel failure\n");
            size_t global[1] = {input_tensor_->getTotalSizeFromDims()};
            status = runtime_->enqueueKernel(kernel_int8_to_int_.get(), (cl_uint)1, global, NULL);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "execute kernel failure\n");
        } else {
            input_tensor =
                std::make_shared<CLTensor>(runtime_, precision_, input_tensor_->getDataType(), input_tensor_->getDim());
            size_t copy_bytes = in_tensor->getNumOfBytes();
            size_t offset = 0;
            status = runtime_->copyBuffer(input_tensor->getDataPtr(), in_tensor->getDataPtr(), offset, offset, copy_bytes);
            CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "copyBuffer failure\n");
        }
    }

    if (is_global_ave_pool_) {  // Global Average Pooling
        size_t global[1] = {out_size_};
        int feature_size = num_elements_in_axis;
        status = runtime_->setKernelArg(kernel_mean_feature_opt_.get(),
                                        in_tensor->getDataPtr(),
                                        output_tensor->getDataPtr(),
                                        feature_size,
                                        num_elements_in_axis);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel failure\n");
        status = runtime_->enqueueKernel(kernel_mean_feature_opt_.get(), (cl_uint)1, global, NULL);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "execute kernel failure\n");
        return Status::SUCCESS;
    } else {  // from CLReduce.cpp
        auto in_dim = input_tensor_->getDim();
        uint32_t dm_dim[4] = {in_dim.n, in_dim.c, in_dim.h, in_dim.w};  // dynamic dim

        std::set<uint32_t>::reverse_iterator rit;
        for (rit = resolved_axis_.rbegin(); rit != resolved_axis_.rend(); ++rit) {
            uint32_t k = 1, ax = *rit;
            for (uint32_t i = ax + 1; i < 4; i++) {
                k *= getDim(in_dim, i);
            }
            while (dm_dim[ax] > 1) {
                uint32_t offset = dm_dim[ax] - (dm_dim[ax] >> 1);
                if (offset * 2 > dm_dim[ax]) {
                    dm_dim[ax] = offset - 1;
                } else {
                    dm_dim[ax] = offset;
                }
                uint32_t offset_k = offset * k;
                size_t global = input_tensor_->getTotalSizeFromDims();
                Status status = runtime_->setKernelArg(kernel_sum_.get(),
                                                       input_tensor->getDataPtr(),
                                                       offset_k,
                                                       in_dim.n,
                                                       in_dim.c,
                                                       in_dim.h,
                                                       in_dim.w,
                                                       dm_dim[0],
                                                       dm_dim[1],
                                                       dm_dim[2],
                                                       dm_dim[3]);
                dm_dim[ax] = offset;
                CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "setKernel failure\n");
                status = runtime_->enqueueKernel(kernel_sum_.get(), (cl_uint)1, &global, NULL);
                CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == status, "execute kernel failure\n");
            }
        }

        // write result to output mem
        size_t global = input_tensor_->getTotalSizeFromDims();
        Status state = runtime_->setKernelArg(kernel_mean_.get(),
                                              input_tensor->getDataPtr(),
                                              output_tensor->getDataPtr(),
                                              num_elements_in_axis,
                                              in_dim.n,
                                              in_dim.c,
                                              in_dim.h,
                                              in_dim.w,
                                              dm_dim[0],
                                              dm_dim[1],
                                              dm_dim[2],
                                              dm_dim[3]);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");
        state = runtime_->enqueueKernel(kernel_mean_.get(), (cl_uint)1, &global, NULL);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");
        return Status::SUCCESS;
    }
}

Status CLMean::release() {
    ENN_DBG_PRINT("CLMean::release() is called");
    return Status::SUCCESS;
}
bool CLMean::isFitFeatureMeanOpt(int num_elements_in_axis) {
    DEBUG_PRINT("CLMean::isFitFeatureMeanOpt() is called");
    uint32_t min_axis = *(resolved_axis_.begin());
    uint32_t max_axis = *(resolved_axis_.rbegin());
    uint32_t size_between_axises = 1, size_after_axises = 1;
    for (uint32_t ax = min_axis; ax < max_axis + 1; ax++) {
        size_between_axises *= input_dim_[ax];
    }
    for (uint32_t ax = max_axis + 1; ax < input_dim_.size(); ax++) {
        size_after_axises *= input_dim_[ax];
    }
    if (size_between_axises == num_elements_in_axis && size_after_axises == 1) {
        return true;
    }
    return false;
}
void CLMean::prepare_mean(const std::shared_ptr<ITensor> axis, std::shared_ptr<ITensor> output) {
    DEBUG_PRINT("CLMean::prepare_mean() is called");

    num_axis_ = axis->getTotalSizeFromDims();
    tmp_axis_ = make_shared_array<int32_t>(num_axis_);
    axis->readData(tmp_axis_.get());
    for (int i = 0; i < num_axis_; i++) {
        axis_.push_back(tmp_axis_.get()[i]);
    }

    // Resolve axis. remove duplicated axises
    if (!resolved_axis_.empty()) {
        resolved_axis_.clear();
    }
    for (uint32_t idx = 0; idx < num_axis_; ++idx) {
        // Handle negative index.
        uint32_t current = axis_[idx] < 0 ? (axis_[idx] + in_num_dim_) : axis_[idx];
        resolved_axis_.insert(current);
    }
    num_resolved_axis_ = resolved_axis_.size();

    if (parameters_->androidNN) {
        NDims output_dim;
        if (parameters_->keep_dims) {
            for (uint32_t i = 0; i < input_dim_.size(); i++) {
                if (resolved_axis_.find(i) == resolved_axis_.end())
                    output_dim.push_back(input_dim_[i]);
                else
                    output_dim.push_back(1);
            }
        } else {
            for (uint32_t i = 0; i < input_dim_.size(); i++) {
                if (resolved_axis_.find(i) == resolved_axis_.end())
                    output_dim.push_back(input_dim_[i]);
            }
        }
        if (output_dim.empty()) {
            output_dim = {1};
        }
        if (!isDimsSame(output_dim, output->getDims())) {
            output->reconfigureDimsAndBuffer(output_dim);
        }
    }
}
void CLMean::print_parameter() {
    ENN_DBG_PRINT("\
    Mean parameters as below: \
    num_axis_: %d\n \
    in_num_dim_: %d\n \
    out_size_: %d\n \
    num_resolved_axis_: %d\n \
    axis_as_input_: %d\n \
    androidNN_: %d\n \
    keep_dims_: %d\n \
    is_global_ave_pool_: %d\n",
                  num_axis_,
                  in_num_dim_,
                  out_size_,
                  num_resolved_axis_,
                  axis_as_input_,
                  parameters_->androidNN,
                  parameters_->keep_dims,
                  is_global_ave_pool_);
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
