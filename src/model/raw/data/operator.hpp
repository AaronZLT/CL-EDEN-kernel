#ifndef SRC_MODEL_RAW_DATA_OPERATOR_HPP_
#define SRC_MODEL_RAW_DATA_OPERATOR_HPP_

#include <memory>

#include "model/raw/model.hpp"
#include "model/types.hpp"

namespace enn {
namespace model {
namespace raw {
namespace data {

/*
 * Operator is {"NCP", "Normalization", "SOFTMAX", ...}
 */
class Operator {
private:
    int32_t index;
    int32_t op_code;
    std::string name;
    std::vector<std::string> lib_names;
    std::vector<int32_t> input_indexes;
    std::vector<int32_t> output_indexes;
    int32_t operator_options_index;       // OperatorOptions[index]
    std::vector<int32_t> binary_indexes;  // Binary[index]
    Accelerator accelerator;

    friend class OperatorBuilder;

public:
    Operator() : index(UNDEFINED), op_code(UNDEFINED), operator_options_index(UNDEFINED), accelerator(Accelerator::NONE) {}

    int32_t get_index() {
        return index;
    }

    int32_t get_op_code() {
        return op_code;
    }

    std::string get_name() {
        return name;
    }

    std::vector<std::string>& get_lib_names() {
        return lib_names;
    }

    std::vector<int32_t>& get_input_indexes() {
        return input_indexes;
    }

    std::vector<int32_t>& get_output_indexes() {
        return output_indexes;
    }

    int32_t get_operator_options_index() {
        return operator_options_index;
    }

    std::vector<int32_t>& get_binary_indexes() {
        return binary_indexes;
    }

    Accelerator get_accelerator() {
        return accelerator;
    }
};

class OperatorBuilder : public ModelBuilder {
private:
    std::shared_ptr<Operator> operator_;

public:
    explicit OperatorBuilder(std::shared_ptr<RawModel> raw_model) : ModelBuilder(raw_model){};

    OperatorBuilder& add_operator() {
        operator_ = std::make_unique<Operator>();
        return *this;
    }

    OperatorBuilder& get_operator(uint32_t index) {
        operator_ = raw_model_->get_operators().at(index);
        return *this;
    }

    OperatorBuilder& set_op_index(int32_t index) {
        operator_->index = index;
        return *this;
    }

    OperatorBuilder& set_op_code(int32_t op_code) {
        operator_->op_code = op_code;
        return *this;
    }

    OperatorBuilder& set_op_name(std::string name) {
        operator_->name = name;
        return *this;
    }

    OperatorBuilder& add_lib_name(std::string lib_name) {
        operator_->lib_names.push_back(lib_name);
        return *this;
    }

    OperatorBuilder& set_lib_names(std::vector<std::string> lib_names) {
        operator_->lib_names = lib_names;
        return *this;
    }

    OperatorBuilder& set_input_indexes(std::vector<int32_t> indexes) {
        operator_->input_indexes = indexes;
        return *this;
    }

    OperatorBuilder& set_output_indexes(std::vector<int32_t> indexes) {
        operator_->output_indexes = indexes;
        return *this;
    }

    OperatorBuilder& set_operator_options_index(int32_t index) {
        operator_->operator_options_index = index;
        return *this;
    }

    OperatorBuilder& set_binary_indexes(std::vector<int32_t> indexes) {
        operator_->binary_indexes = indexes;
        return *this;
    }

    OperatorBuilder& set_accelerator(Accelerator accelerator) {
        operator_->accelerator = accelerator;
        return *this;
    }

    void build() {
        raw_model_->operators_.push_back(std::move(operator_));
    }
};

};  // namespace data
};  // namespace raw
};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_RAW_DATA_OPERATOR_HPP_
