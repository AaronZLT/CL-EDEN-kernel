#ifndef SRC_MODEL_RAW_DATA_OPERATOR_OPTIONS_HPP_
#define SRC_MODEL_RAW_DATA_OPERATOR_OPTIONS_HPP_

#include "model/raw/model.hpp"

namespace enn {
namespace model {
namespace raw {
namespace data {

/*
 * OperatorOptions is {Conv2DOptions, SoftmaxOptions, ...}
 */
class OperatorOptions {
private:
    int32_t op_index;
    uint32_t number;      // enum value (BuiltinOptions) defined in schema_generated.h
    std::string name;     // BuiltinOptions name or FeatureMap name("NPC_BINARY", ...)
    const void* options;  // addr getting from builtin_options() in schema_generated.h

    friend class OperatorOptionsBuilder;

public:
    OperatorOptions() : op_index(UNDEFINED), number(UNDEFINED), options(nullptr) {}

    int32_t get_operator_index() {
        return op_index;
    }

    uint32_t get_number() {
        return number;
    }

    std::string get_name() {
        return name;
    }

    const void* get_options() {
        return options;
    }
};

class OperatorOptionsBuilder : public ModelBuilder {
private:
    std::shared_ptr<OperatorOptions> operator_option_;

public:
    explicit OperatorOptionsBuilder(std::shared_ptr<RawModel> raw_model) : ModelBuilder(raw_model){};

    OperatorOptionsBuilder& add_operator_options() {
        operator_option_ = std::make_unique<OperatorOptions>();
        return *this;
    }

    OperatorOptionsBuilder& get_operator_options(uint32_t index) {
        operator_option_ = raw_model_->get_operator_options().at(index);
        return *this;
    }

    OperatorOptionsBuilder& set_operator_index(int32_t index) {
        operator_option_->op_index = index;
        return *this;
    }

    OperatorOptionsBuilder& set_num(uint32_t number) {
        operator_option_->number = number;
        return *this;
    }

    OperatorOptionsBuilder& set_name(std::string name) {
        operator_option_->name = name;
        return *this;
    }

    OperatorOptionsBuilder& set_options(const void* options) {
        operator_option_->options = options;
        return *this;
    }

    void build() {
        raw_model_->operator_options_.push_back(std::move(operator_option_));
    }
};

};  // namespace data
};  // namespace raw
};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_RAW_DATA_OPERATOR_OPTIONS_HPP_
