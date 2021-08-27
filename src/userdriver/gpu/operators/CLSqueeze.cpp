#include "CLSqueeze.hpp"

namespace enn {
namespace ud {
namespace gpu {
namespace {
constexpr int INPUT_INDEX = 0;
constexpr int SQUEEZE_DIMS_INDEX = 1;
constexpr int OUTPUT_INDEX = 0;
}  // namespace

CLSqueeze::CLSqueeze(const std::shared_ptr<CLRuntime> runtime, const PrecisionType &precision) :
    runtime_(runtime), precision_(precision) {
    ENN_DBG_PRINT("CLSqueeze is created\n");
    input_tensor_ = nullptr;
    squeeze_dims_ = nullptr;
    output_tensor_ = nullptr;
    parameters_ = nullptr;
    kernel_ = nullptr;
    squeeze_dims_as_input_ = false;
    squeeze_dims_arr_ = nullptr;
}

Status CLSqueeze::initialize(const std::vector<std::shared_ptr<ITensor>> &input_tensors,
                             const std::vector<std::shared_ptr<ITensor>> &output_tensors,
                             const std::shared_ptr<Parameters> parameters) {
    ENN_DBG_PRINT("CLSqueeze::initialize() is called\n");

    input_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(INPUT_INDEX));
    if (input_tensors.size() > 1)
        squeeze_dims_ = std::static_pointer_cast<CLTensor>(input_tensors.at(SQUEEZE_DIMS_INDEX));
    output_tensor_ = std::static_pointer_cast<CLTensor>(output_tensors.at(OUTPUT_INDEX));
    parameters_ = std::static_pointer_cast<SqueezeParameters>(parameters);
    //hardcode here
    if(squeeze_dims_!=nullptr){
        int axisdims1 = squeeze_dims_->getDim().c*squeeze_dims_->getDim().h*squeeze_dims_->getDim().w*squeeze_dims_->getDim().n;
        int axisdims2 = squeeze_dims_->getTotalSizeFromDims();
        int intensordims = input_tensor_->getTotalSizeFromDims();
        if(axisdims2>intensordims){
            ITensor::placeholder("ERROR #1: SQUEEZE's inputs REVERSED",ColorType::RED);
            ITensor::placeholder("Please refe below",ColorType::RED);
            ITensor::placeholder("https://code.sec.samsung.net/jira/browse/ISSUEBOARD-10203",ColorType::RED);
            //input_tensor_ = std::static_pointer_cast<CLTensor>(input_tensors.at(SQUEEZE_DIMS_INDEX));
            //squeeze_dims_ = std::static_pointer_cast<CLTensor>(input_tensors.at(INPUT_INDEX));
        }
    }
    //hardcode end
    CHECK_EXPR_RETURN_FAILURE(nullptr != parameters_, "CLSqueeze must have parameters\n");
    if (squeeze_dims_ != nullptr) {
        squeeze_dims_as_input_ = !squeeze_dims_->is_const();
    }

    Status state = Status::FAILURE;
    if (parameters_->androidNN && !squeeze_dims_as_input_) {
        state = reconfigure_output();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "CLSqueeze reconfigure_output failure\n");
    }

    state = runtime_->setKernel(&kernel_, "squeeze", precision_);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernel failure\n");

    return Status::SUCCESS;
}

Status CLSqueeze::execute() {
    ENN_DBG_PRINT("CLSqueeze::execute() is called\n");

    Status state;
    if (parameters_->androidNN && squeeze_dims_as_input_) {
        state = reconfigure_output();
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "CLSqueeze prepare failure\n");
    }

    auto input_tensor = std::static_pointer_cast<CLTensor>(input_tensor_);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output_tensor_);

    auto input_data = input_tensor->getDataPtr();
    auto output_data = output_tensor->getDataPtr();

    size_t global[1] = {input_tensor->getTotalSizeFromDims()};
    size_t local[1] = {1};

    state = runtime_->setKernelArg(kernel_.get(), input_data, output_data);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "setKernelArg failure\n");

    state = runtime_->enqueueKernel(kernel_.get(), (cl_uint)1, global, local);
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute kernel failure\n");

    return Status::SUCCESS;
}

Status CLSqueeze::reconfigure_output() {
    ENN_DBG_PRINT("CLSqueeze::reconfigure_output() is called\n");
    NDims output_shape;
    if (squeeze_dims_ == nullptr || squeeze_dims_->getTotalSizeFromDims() == 0) {
        for (uint32_t idx = 0; idx < input_tensor_->getNumOfDims(); ++idx) {
            if (input_tensor_->getDims(idx) != 1)
                output_shape.push_back(input_tensor_->getDims(idx));
        }
    } else {
        NDims squeezed_shape = input_tensor_->getDims();
        int32_t num_dims_squeezed = 0;
        int32_t num_input_dims = input_tensor_->getNumOfDims();
        int squeeze_dims_size = squeeze_dims_->getTotalSizeFromDims();
        squeeze_dims_arr_ = make_shared_array<int32_t>(squeeze_dims_size);
        squeeze_dims_->readData(squeeze_dims_arr_.get());
        for (int32_t idx = 0; idx < squeeze_dims_size; idx++) {
            int32_t current = squeeze_dims_arr_.get()[idx] < 0 ? squeeze_dims_arr_.get()[idx] + num_input_dims
                                                               : squeeze_dims_arr_.get()[idx];
            CHECK_EXPR_RETURN_FAILURE((current >= 0 && current < num_input_dims && input_tensor_->getDims(current) == 1),
                                      "The dim to be squeezed is not 1: %u.\n",
                                      input_tensor_->getDims(current));
            if (squeezed_shape[current] != 0) {
                squeezed_shape[current] = 0;  // mark as "should be squeezed"
                num_dims_squeezed++;
            }
        }
        if (num_dims_squeezed == num_input_dims)
            output_shape.push_back(1);
        else
            for (auto &shape : squeezed_shape)
                if (shape != 0)
                    output_shape.push_back(shape);
    }
    if (output_tensor_->getDims() != output_shape)
        output_tensor_->reconfigureDimsAndBuffer(output_shape);

    return Status::SUCCESS;
}

Status CLSqueeze::release() {
    ENN_DBG_PRINT("CLSqueeze::release() is called");
    return Status::SUCCESS;
}

}  // namespace gpu
}  // namespace ud
}  // namespace enn
