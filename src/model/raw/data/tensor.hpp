#ifndef SRC_MODEL_RAW_DATA_TENSOR_HPP_
#define SRC_MODEL_RAW_DATA_TENSOR_HPP_

#include "common/enn_debug.h"
#include "model/raw/model.hpp"
#include "model/types.hpp"

namespace enn {
namespace model {
namespace raw {
namespace data {

/*
 * Tensor is {input/output of each operator} derived from Tensor
 */
class Tensor {
protected:
    int32_t index;
    std::string name;  // "BIAS", "FLAC_LEN", ...
    int32_t type;      // TensorType_FLOAT32(0), ... in schema_generated.h
    int32_t prev_operator_index;
    std::vector<int32_t> next_operator_indexes;
    std::vector<uint32_t> shape;
    const uint8_t* data_addr;  // addr of value loaded on memory from file
    int32_t data_size;         // size of value loaded on memory from file
    const TFlite::QuantizationParameters* quantization_parameters;
    const TFlite::SymmPerChannelQuantParamters* symm_per_channel_quant_parameters;

    friend class TensorBuilder;

public:
    Tensor() : index(UNDEFINED), type(UNDEFINED), prev_operator_index(UNDEFINED),
               data_addr(nullptr), data_size(UNDEFINED), quantization_parameters(nullptr),
               symm_per_channel_quant_parameters(nullptr) {}

    virtual ~Tensor() = default;

    int32_t get_index() {
        return index;
    }

    std::string get_name() {
        return name;
    }

    int32_t get_type() {
        return type;
    }

    int32_t get_prev_operator_index() {
        return prev_operator_index;
    }

    const std::vector<int32_t>& get_next_operator_indexes() {
        return next_operator_indexes;
    }

    const std::vector<uint32_t>& get_shape() {
        return shape;
    }

    const uint8_t* get_address() {
        return data_addr;
    }

    int32_t get_size() {
        return data_size;
    }

    const TFlite::QuantizationParameters* get_quantization_parameters() {
        return quantization_parameters;
    }

    const TFlite::SymmPerChannelQuantParamters* get_symm_per_channel_quant_parameters() {
        return symm_per_channel_quant_parameters;
    }
};

class TensorBuilder : public ModelBuilder {
private:
    std::shared_ptr<Tensor> tensor_;

public:
    explicit TensorBuilder(std::shared_ptr<RawModel> raw_model) : ModelBuilder(raw_model){};

    TensorBuilder& add_tensor() {
        tensor_ = std::make_unique<Tensor>();
        return *this;
    }

    TensorBuilder& get_tensor(uint32_t index) {
        tensor_ = raw_model_->get_tensors().at(index);
        return *this;
    }

    // For set prev/next operation index to each tensor
    TensorBuilder& get_tensor_from_index(int32_t index) {
        for (auto tensor : raw_model_->get_tensors()) {
            if (tensor->get_index() == index) {
                tensor_ = tensor;
            }
        }
        return *this;
    }

    TensorBuilder& set_index(int32_t index) {
        tensor_->index = index;
        return *this;
    }

    TensorBuilder& set_name(std::string name) {
        tensor_->name = name;
        return *this;
    }

    TensorBuilder& set_type(int32_t type) {
        tensor_->type = type;
        return *this;
    }

    TensorBuilder& set_prev_operator_index(int32_t prev_operator_index) {
        tensor_->prev_operator_index = prev_operator_index;
        return *this;
    }

    TensorBuilder& add_next_operator_index(int32_t next_operator_index) {
        tensor_->next_operator_indexes.push_back(next_operator_index);
        return *this;
    }

    TensorBuilder& set_next_operator_indexes(std::vector<int32_t> next_operator_indexes) {
        tensor_->next_operator_indexes.swap(next_operator_indexes);
        return *this;
    }

    template <typename V>
    TensorBuilder& set_shape(V&& shape) {
        tensor_->shape = std::forward<V>(shape);
        return *this;
    }

    TensorBuilder& set_address(const uint8_t* address) {
        tensor_->data_addr = address;
        return *this;
    }

    TensorBuilder& set_size(int32_t size) {
        tensor_->data_size = size;
        return *this;
    }

    TensorBuilder& set_quantization_parameters(const TFlite::QuantizationParameters* quantization_parameters) {
        tensor_->quantization_parameters = quantization_parameters;
        return *this;
    }

    TensorBuilder& set_symm_per_channel_quant_parameters(
        const TFlite::SymmPerChannelQuantParamters* symm_per_channel_quant_parameters) {
        tensor_->symm_per_channel_quant_parameters = symm_per_channel_quant_parameters;
        return *this;
    }

    void build() {
        raw_model_->tensors_.push_back(std::move(tensor_));
    }
};

};  // namespace data
};  // namespace raw
};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_RAW_DATA_TENSOR_HPP_
