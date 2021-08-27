#include "userdriver/common/operator_interfaces/common/Debug.hpp"
#include "userdriver/gpu/operators/cl_optimized_impls/CLBaseFullyConnected.hpp"

namespace enn {
namespace ud {
namespace gpu {

constexpr float CACHE_SIZE = 16 * 1024 * 0.7;
constexpr int MAX_GROUP_SIZE = 128;

uint32_t CLBaseFullyConnected::computeSplitNum(const uint32_t &input_count) {
    uint32_t type_size = 0;
    if (precision_ == PrecisionType ::FP16) {
        type_size = sizeof(cl_half);
    } else if (precision_ == PrecisionType ::FP32) {
        type_size = sizeof(cl_float);
    } else if (precision_ == PrecisionType ::UINT8 || precision_ == PrecisionType ::INT8) {
        type_size = sizeof(cl_float);
    }
    size_t split_number = 1;
    size_t data_size = input_count * type_size * 2;
    if (data_size > CACHE_SIZE) {
        split_number = (size_t)(pow(2, ceil(log(data_size / CACHE_SIZE) / log(2.0))));
    }
    // When the size of data for one computing thread exceeds the GPU cache size,
    // split the data to fit in the cache. And set the thread numbers to the pow
    // of 2 to improve cache efficiency.
    if (input_count * type_size > 512) {
        if (input_count % (8 * 32) == 0) {
            // each thread processes 8 * 32 elements, 24 threads as a group
            // split number must be multiple of 24 in this case
            split_number = alignTo(input_count, 8 * 24 * 32) / (8 * 24 * 32) * 24;
            // there is no need that the one line data can be uniformly divided here
            return split_number;
        } else if (runtime_->isBifrost() && input_count % (input_count * type_size / 512) == 0) {
            // the cache line size in bifrost is 512
            size_t split_size = input_count * type_size / 512;
            if (split_size < 4 && input_count % 4 == 0) {
                // the splitNumber can not be lower than 4 because the optimal local size should be
                // multiple of 4
                split_number = 4;
            } else {
                split_number = split_size;
            }
        } else if (input_count * type_size > 512 * 4 && input_count % (input_count * type_size / (512 * 4)) == 0) {
            // non-bifrost uses vector compute unit of 4
            size_t split_size = input_count * type_size / (512 * 4);
            if (split_size < 4 && input_count % 4 == 0) {
                split_number = 4;
            } else {
                split_number = split_size;
            }
        }
    }
    if (input_count % split_number) {
        split_number = 1;
    }
    return split_number;
}

CLBaseFullyConnected::CLBaseFullyConnected(const std::shared_ptr<CLRuntime> runtime,
                                           const PrecisionType &precision,
                                           const std::shared_ptr<ITensor> input,
                                           const std::shared_ptr<ITensor> weight,
                                           const std::shared_ptr<ITensor> bias,
                                           std::shared_ptr<ITensor> output,
                                           const bool &weights_as_input) {
    ENN_DBG_PRINT("CLBaseFullyConnected is called");
    runtime_ = runtime;
    weight_ = std::static_pointer_cast<CLTensor>(weight);
    bias_ = std::static_pointer_cast<CLTensor>(bias);
    precision_ = precision;

    uint32_t input_batch = input->getDim().n;
    uint32_t matrix_width = input->getDim().c * input->getDim().h * input->getDim().w;
    uint32_t output_channel = output->getDim().c;
    Dim4 input_dim = input->getDim();
    Dim4 output_dim = output->getDim();

    split_number_ = computeSplitNum(matrix_width);
    uint32_t split_count = input_batch * output_channel * split_number_;

    // we transform the input data of unit8 into the float to get the output.
    Dim4 split_buffer_dim = {split_count, 1, 1, 1};
    split_buffer_ = std::make_shared<CLTensor>(runtime_, PrecisionType::FP32, DataType::FLOAT, split_buffer_dim);

    // set kernel
    if (precision_ == PrecisionType::UINT8 || precision_ == PrecisionType::INT8) {
        map_input_tensor_ = std::make_shared<CLTensor>(runtime_, PrecisionType::FP32, DataType::FLOAT, input_dim);
        map_weight_tensor_ = std::make_shared<CLTensor>(runtime_, PrecisionType::FP32, DataType::FLOAT, weight_->getDim());
        map_bias_tensor_ = std::make_shared<CLTensor>(runtime_, PrecisionType::FP32, DataType::INT32, bias_->getDim());
        map_output_tensor_ = std::make_shared<CLTensor>(runtime_, PrecisionType::FP32, DataType::FLOAT, output_dim);

        quantization_ = std::make_shared<CLQuantization>(runtime_, PrecisionType::FP32);
        Status state = quantization_->initialize({map_output_tensor_}, {output}, nullptr);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "quantization initialize failure\n");

        auto dequantization_parameters = std::make_shared<DeQuantizationParameters>();
        dequantization_input_ = std::make_shared<CLDeQuantization>(runtime_, PrecisionType::FP32);
        state = dequantization_input_->initialize({input}, {map_input_tensor_}, dequantization_parameters);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "dequantization_input_ initialize failure\n");

        dequantization_weight_ = std::make_shared<CLDeQuantization>(runtime_, PrecisionType::FP32);
        state = dequantization_weight_->initialize({weight}, {map_weight_tensor_}, dequantization_parameters);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "dequantization_weight_ initialize failure\n");

        dequantization_bias_ = std::make_shared<CLDeQuantization>(runtime_, PrecisionType::FP32);
        state = dequantization_bias_->initialize({bias}, {map_bias_tensor_}, dequantization_parameters);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "dequantization_bias_ initialize failure\n");

        state = runtime_->setKernel(&kernel_split_, "fc_split", PrecisionType::FP32);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "fc_split setKernel failure\n");
        state = runtime_->setKernel(&kernel_merge_, "fc_merge", PrecisionType::FP32);
        CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "fc_merge setKernel failure\n");

        if (matrix_width % (8 * 32) == 0) {
            state = runtime_->setKernel(&kernel_split_, "fc_split_coalesced", PrecisionType::FP32);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "fc_split_coalesced setKernel failure\n");

            state = runtime_->setKernel(&kernel_merge_, "fc_merge_coalesced", PrecisionType::FP32);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "fc_merge_coalesced setKernel failure\n");
        }
    } else {
        if (matrix_width % (8 * 32) == 0) {
            Status state = runtime_->setKernel(&kernel_split_, "fc_split_coalesced", precision_);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "fc_split_coalesced setKernel failure\n");

            state = runtime_->setKernel(&kernel_merge_, "fc_merge_coalesced", precision_);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "fc_merge_coalesced setKernel failure\n");
        } else {
            Status state = runtime_->setKernel(&kernel_split_, "fc_split", precision_);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "fc_split setKernel failure\n");
            state = runtime_->setKernel(&kernel_merge_, "fc_merge", precision_);
            CHECK_EXPR_NO_RETURN(Status::SUCCESS == state, "fc_merge setKernel failure\n");
        }
    }

    ENN_DBG_PRINT("CLBaseFullyConnected is created");
}

Status CLBaseFullyConnected::execute(const std::shared_ptr<ITensor> input, std::shared_ptr<ITensor> output) {
    auto input_tensor = std::static_pointer_cast<CLTensor>(input);
    auto output_tensor = std::static_pointer_cast<CLTensor>(output);
    auto weight_tensor = std::static_pointer_cast<CLTensor>(weight_);
    auto bias_tensor = std::static_pointer_cast<CLTensor>(bias_);

    if (precision_ == PrecisionType::UINT8 || precision_ == PrecisionType::INT8) {
        return fullyConnectedQuant(input_tensor, weight_tensor, bias_tensor, output_tensor);
    } else {
        return fullyConnectedFloat(input_tensor, weight_tensor, bias_tensor, output_tensor);
    }
}

Status CLBaseFullyConnected::fullyConnectedFloat(const std::shared_ptr<CLTensor> input_tensor,
                                                 const std::shared_ptr<CLTensor> weight_tensor,
                                                 const std::shared_ptr<CLTensor> bias_tensor,
                                                 std::shared_ptr<CLTensor> output_tensor) {
    ENN_DBG_PRINT("CLBaseFullyConnected::execute() is called");
    auto input_data = input_tensor->getDataPtr();
    auto output_data = output_tensor->getDataPtr();
    auto weight_data = weight_tensor->getDataPtr();
    auto bias_data = bias_tensor->getDataPtr();

    Status state = Status::FAILURE;
    size_t global[3] = {0, 0, 0};
    size_t local[3] = {0, 0, 0};
    uint32_t input_batch = input_tensor->getDim().n;
    uint32_t input_channel = input_tensor->getDim().c;
    uint32_t input_height = input_tensor->getDim().h;
    uint32_t input_width = input_tensor->getDim().w;
    uint32_t output_channel = weight_->getDim().n;

    int matrix_width = input_channel * input_height * input_width;
    int matrix_height = output_channel;

    if (matrix_width % (8 * 32) == 0) {
        local[0] = 1;
        local[1] = 1;
        local[2] = 24;
        global[0] = input_tensor->getDim().n;
        global[1] = matrix_height;
        // actually, the split number already be multiple of 24 when computing before for this case
        global[2] = alignTo(split_number_, local[2]);

        state = runtime_->setKernelArg(kernel_split_.get(),
                                       input_data,
                                       weight_data,
                                       split_buffer_->getDataPtr(),
                                       matrix_width,
                                       matrix_height,
                                       split_number_);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "fc_split setKernelArg failure\n");
        state = runtime_->enqueueKernel(kernel_split_.get(), 3, global, local);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute fc_split kernel failure\n");

        local[0] = 1;
        local[1] = 24;
        global[0] = input_tensor->getDim().n;
        global[1] = alignTo(matrix_height, local[1]);

        state = runtime_->setKernelArg(
            kernel_merge_.get(), split_buffer_->getDataPtr(), bias_data, matrix_height, split_number_, output_data);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "fc_merge setKernelArg failure\n");
        state = runtime_->enqueueKernel(kernel_merge_.get(), 2, global, local);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute fc_merge kernel failure\n");
    } else {
        // Split
        global[0] = input_batch;
        global[1] = split_number_;
        global[2] = output_channel;
        local[0] = 1;
        int lsize1 = 4;
        if (split_number_ % lsize1 == 0) {
            local[1] = lsize1;
        } else {
            local[1] = 1;
        }
        if (output_channel % 24 == 0) {
            local[2] = 24;
        } else if (output_channel % 16 == 0) {
            local[2] = 16;
        } else if (output_channel % 12 == 0) {
            local[2] = 12;
        } else if (output_channel % 8 == 0) {
            local[2] = 8;
        } else if (output_channel % 4 == 0) {
            local[2] = 4;
        } else {
            local[2] = 1;
        }

        int split_picture_step = matrix_width;
        int split_step = split_picture_step / split_number_;

        state = runtime_->setKernelArg(
            kernel_split_.get(), input_data, weight_data, split_buffer_->getDataPtr(), split_picture_step, split_step);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "fc_split setKernelArg failure\n");
        state = runtime_->enqueueKernel(kernel_split_.get(), 3, global, local);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute fc_split kernel failure\n");

        // Merge
        size_t global_merge[3] = {input_tensor->getDim().n, output_channel, 1};
        size_t local_merge[3] = {0, 0, 0};
        local_merge[0] = findMaxFactor(input_tensor->getDim().n, MAX_GROUP_SIZE);
        local_merge[1] = 1;
        local_merge[2] = 1;
        uint32_t merge_picture_step = split_number_ * output_channel;
        uint32_t merge_step = output_channel;

        state = runtime_->setKernelArg(
            kernel_merge_.get(), split_buffer_->getDataPtr(), bias_data, merge_picture_step, merge_step, output_data);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "fc_merge setKernelArg failure\n");
        state = runtime_->enqueueKernel(kernel_merge_.get(), 3, global_merge, local_merge);
        CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "execute fc_merge kernel failure\n");
    }

    return Status::SUCCESS;
}

Status CLBaseFullyConnected::fullyConnectedQuant(const std::shared_ptr<CLTensor> input_tensor,
                                                 const std::shared_ptr<CLTensor> weight_tensor,
                                                 const std::shared_ptr<CLTensor> bias_tensor,
                                                 std::shared_ptr<CLTensor> output_tensor) {
    ENN_DBG_PRINT("CLBaseFullyConnected::execute() is called");
    Status state = dequantization_input_->execute();
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "dequantization_ input failure\n");
    state = dequantization_weight_->execute();
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "dequantization_ weight failure\n");
    state = dequantization_bias_->execute();
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "dequantization_ bias failure\n");

    fullyConnectedFloat(map_input_tensor_, map_weight_tensor_, map_bias_tensor_, map_output_tensor_);
    state = quantization_->execute();
    CHECK_EXPR_RETURN_FAILURE(Status::SUCCESS == state, "quantization failure\n");

    return Status::SUCCESS;
}

Status CLBaseFullyConnected::release() { return Status::SUCCESS; }

}  // namespace gpu
}  // namespace ud
}  // namespace enn
