#ifndef SRC_MODEL_PARSER_STRATEGY_CODED_PARSE_STRATEGY_HPP_
#define SRC_MODEL_PARSER_STRATEGY_CODED_PARSE_STRATEGY_HPP_

#include "model/parser/strategy/strategy.hpp"

namespace enn {
namespace model {

// TODO: Add member variable able to store model object from Coded.
//       Also add parameter of it to ctr for initializing that member.
class CodedParseStrategy : public ParseStrategy {
public:
    explicit CodedParseStrategy();

    void parse_operators() override;
    void parse_tensors() override;

    std::shared_ptr<raw::Model> result() override;

    void print() override;

private:
    // TODO: Disable in release version
    bool validate_operators();
    bool validate_tensors();

    // TODO: Add model object here.
};

};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_PARSER_STRATEGY_CODED_PARSE_STRATEGY_HPP_
