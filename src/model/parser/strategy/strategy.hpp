#ifndef SRC_MODEL_PARSER_STRATEGY_STRATEGY_HPP_
#define SRC_MODEL_PARSER_STRATEGY_STRATEGY_HPP_

#include "model/raw/model.hpp"

namespace enn {
namespace model {

class ParseStrategy {
public:
    // The parameter of ctr in subclasses should be object ressponding to each flatbuffer
    explicit ParseStrategy() = default;
    virtual ~ParseStrategy() = default;
    ParseStrategy(ParseStrategy&&) = delete;
    ParseStrategy& operator=(ParseStrategy&&) = delete;
    ParseStrategy(const ParseStrategy&) = delete;
    ParseStrategy& operator=(const ParseStrategy&) = delete;

    // Pure virtual methods should be overrided, which is common no matter what strategy.
    // Hook method can be overrided optionally, which depends on certain strategy.
    // validate functions are non-virtual to optionally disable validation in SubClasses.
    virtual void parse_operators() = 0;
    bool validate_operators() {
        return true;
    }

    virtual void parse_tensors() = 0;
    bool validate_tensors() {
        return true;
    }

    virtual void parse_operator_options() {}
    bool validate_operator_options() {
        return true;
    }

    virtual void parse_scalars() {}
    bool validate_scalars() {
        return true;
    }

    virtual void parse_regions() {}
    bool validate_regions() {
        return true;
    }

    virtual void parse_buffers() {}
    bool validate_buffers() {
        return true;
    }

    virtual void parse_attribute() {}
    bool validate_attribute() {
        return true;
    }

    virtual void parse_control_option() {}
    bool validate_control_option() {
        return true;
    }

    virtual void parse_binaries() {}
    bool validate_binaries() {
        return true;
    }

    virtual void parse_parameters() {}
    bool validate_parameters() {
        return true;
    }

    virtual void parse_graph_infos() {}
    bool validate_graph_infos() {
        return true;
    }

    // return raw::Model via raw_model_builder_
    virtual std::shared_ptr<raw::Model> result() = 0;

    virtual void print() = 0;

    /**
     * Check the model file is loaded normally
     */
    virtual bool is_verified() {
        return verified;
    };

    /**
     * Invoked on the main thread before the parsing thread is executed
     */
    virtual void pre_execute() {}

    /**
     * Invoked on the main thread after the background computation finishes
     */
    virtual void post_execute() {}

protected:
    std::shared_ptr<ModelMemInfo> model_mem_info;
    bool verified = true;
};

};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_PARSER_STRATEGY_STRATEGY_HPP_
