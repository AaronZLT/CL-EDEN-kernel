#ifndef SRC_COMMON_IDENTIFIER_BASE_HPP_
#define SRC_COMMON_IDENTIFIER_BASE_HPP_

#include <memory>
#include <iostream>
#include <string>
#include <functional>

#include "common/enn_utils.h"

namespace enn {
namespace identifier {

using FullIDType = uint64_t;

template <typename IDType>
class IdentifierBase {
 protected:
    //                           [ Full ID Format]
    // |  reserved  | caller pid | model uid | op_list uid | executable model uid |
    // |   15 bit   |   17 bit   |   8  bit  |    16 bit   |        8 bit         |
    mutable IDType id_;
    explicit IdentifierBase(IDType id) : id_(id) {}

 public:
    using UPtr = std::unique_ptr<IdentifierBase>;

    IdentifierBase() : id_(0) {}
    virtual ~IdentifierBase() = default;

    // Return true if id overlaps rid's id, otherwise false.
    bool is_overlapping(const IdentifierBase& rid) const {
        return (this->mask_on() | (rid.mask_on() == this->mask_on())) && this->equal_in_common(rid);
    }

    // Return true if id is overlapped by rid's id, otherwise false.
    bool is_overlapped_by(const IdentifierBase& rid) const {
        return (this->mask_on() | (rid.mask_on() == rid.mask_on())) && this->equal_in_common(rid);
    }

    // Compare the common field extracted with a mask to check if it is equal.
    bool equal_in_common(const IdentifierBase& rid) const {
        IDType mask = this->mask_on() & rid.mask_on();
        return (this->get() & mask) == (rid.get() & mask);
    }

    bool operator==(const IdentifierBase& rid) const {
        return this->id_ == rid.get();
    }

    bool operator!=(const IdentifierBase& rid) const {
        return this->id_ != rid.get();
    }

    bool operator<(const IdentifierBase& rid) const {
        return this->id_ < rid.get();
    }

    std::string to_string() const {
        std::stringstream s;
        s << std::hex << std::uppercase << this->get();
        return s.str();
    }

    // Hash functor by get() virtual function.
    struct Hash {
        size_t operator()(const IdentifierBase& id) const {
            return std::hash<IDType>{}(id.get());
        }
    };

    // return id calculted recursively by Derived classes.
    virtual IDType get() const = 0;
    // return mask-on that sets id fields required to 1.
    virtual IDType mask_on() const = 0;
     // return mask-off that set id fields required to 0.
    virtual IDType mask_off() const = 0;
    // return id on type-casting to IDType.
    virtual operator IDType() const = 0;

    friend std::ostream& operator<<(std::ostream& out, const IdentifierBase& id) {
        out << std::hex << std::uppercase << id.get();
        return out;
    }

    friend enn::debug::EnnMsgHandler& operator<<(enn::debug::EnnMsgHandler& out, const IdentifierBase& id) {
        out << std::hex << std::uppercase << id.get();
        return out;
    }
};


};  // namespace identifier
};  // namespace enn

#endif  // SRC_COMMON_IDENTIFIER_BASE_HPP_