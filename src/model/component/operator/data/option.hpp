#ifndef SRC_MODEL_COMPONENT_OPERATOR_OPERATOR_DATA_OPTION_HPP_
#define SRC_MODEL_COMPONENT_OPERATOR_OPERATOR_DATA_OPTION_HPP_

#include <memory>
#include <vector>
#include <string>

#include "model/component/operator/data/data.hpp"
#include "model/schema/schema_nnc.h"

namespace enn {
namespace model {
namespace component {
namespace data {

class Option : public Data {
 public:
    explicit Option() { enum_ = TFlite::BuiltinOptions::BuiltinOptions_NONE; }
    Option(const std::string& name, const void* addr, size_t size, TFlite::BuiltinOptions num)
        : Data{name, addr, size}, enum_{num} {}
    Option(const Option&) = default;
    Option& operator=(const Option&) = default;

    const TFlite::BuiltinOptions& get_enum() const { return enum_; }

 private:
    TFlite::BuiltinOptions enum_;
};


};  // namespace data
};  // namespace component
};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_COMPONENT_OPERATOR_OPERATOR_DATA_OPTION_HPP_
