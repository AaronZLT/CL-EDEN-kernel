#ifndef SRC_MODEL_ATTRIBUTE_HPP_
#define SRC_MODEL_ATTRIBUTE_HPP_

#include <memory>
#include "model/schema/schema_nnc.h"

namespace enn {
namespace model {

class Attribute : public std::enable_shared_from_this<Attribute> {
    TFlite::LegacyModel legacy_model_;           // configure from NNC
    bool relax_computation_float32_to_float16_;  // configure from NNC for AndroidNN.

public:
    using Ptr = std::shared_ptr<Attribute>;

    // TODO(yc18.cho & empire.jung): Set all of values from NNC after they are enabled in parser.
    Attribute()
        : legacy_model_(TFlite::LegacyModel_TENSORFLOW_NCHW),
          relax_computation_float32_to_float16_(true) {}

    Attribute(const TFlite::LegacyModel &legacy_model,
              const bool &relax_computation_float32_to_float16) :
        legacy_model_(legacy_model),
        relax_computation_float32_to_float16_(relax_computation_float32_to_float16) {}

    TFlite::LegacyModel get_legacy_model() const {
        return legacy_model_;
    }

    bool get_relax_computation_float32_to_float16() const {
        return relax_computation_float32_to_float16_;
    }

private:
    friend class Generator;

    auto set_lagacy_model(TFlite::LegacyModel legacy_model) {
        legacy_model_ = legacy_model;
        return shared_from_this();
    }

    auto set_relax_computation_float32_to_float16(bool relax_computation_float32_to_float16) {
        relax_computation_float32_to_float16_ = relax_computation_float32_to_float16;
        return shared_from_this();
    }

};

};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_ATTRIBUTE_HPP_