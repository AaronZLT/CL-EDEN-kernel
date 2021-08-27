#ifndef SRC_MODEL_PARSER_PARSER_HPP_
#define SRC_MODEL_PARSER_PARSER_HPP_

#include "model/raw/model.hpp"

namespace enn {
namespace model {

class Parser {
public:
    // Convert given input model to RawModel and then store in Impl
    explicit Parser();
    virtual ~Parser();
    Parser(Parser&&) = delete;
    Parser& operator=(Parser&&) = delete;
    Parser(const Parser&) = delete;
    Parser& operator=(const Parser&) = delete;

    // ModelManager using Parser should let it know ModelType.
    // Because ModelManager should know type of each model opened for caching.
    void Set(const ModelType& model_type, const std::shared_ptr<ModelMemInfo> model,
             const std::shared_ptr<std::vector<std::shared_ptr<enn::model::ModelMemInfo>>> params);

    std::shared_ptr<raw::Model> Parse();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_PARSER_PARSER_HPP_
