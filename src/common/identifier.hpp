#ifndef SRC_COMMON_IDENTIFIER_HPP_
#define SRC_COMMON_IDENTIFIER_HPP_

#include <memory>
#include <limits>
#include <iostream>
#include <string>

#include "common/identifier_base.hpp"

namespace enn {
namespace identifier {

template <typename Type, Type Max, size_t Offset>
class Identifier : public IdentifierBase<Type> {
 public:
    static constexpr Type Mask = static_cast<Type>(Max) << Offset;

    // No explicit keyword, as implicit conversion is intended.
    Identifier(Type id)
        : IdentifierBase<Type>{id} {}

    Identifier()
        : IdentifierBase<Type>{0} {}

    Type get() const override {
        return this->id_;
    }

    operator Type() const override {
        return get();
    }

    Type mask_on() const override {
        return Mask;
    }

    Type mask_off() const override {
        return ~Mask;
    }
};


};  // namespace identifier
};  // namespace enn

#endif  // SRC_COMMON_IDENTIFIER_BASE_HPP_