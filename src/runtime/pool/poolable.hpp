#ifndef SRC_RUNTIME_POOL_POOLABLE_HPP_
#define SRC_RUNTIME_POOL_POOLABLE_HPP_

#include <string>

#include "common/identifier_base.hpp"

namespace enn {
namespace runtime {
namespace pool {

using namespace identifier;

// [Usage of Poolable]
//  Any class that needs to be handled in DependentyTreePool should
//   inherit this Poolable and override all pure virtual functions.
//  On inheriting this, an argument for defining a type of the Key
//   that is used for identifying each Poolable objects.
template <typename K>
class Poolable {
 public:
    using Key = K;
    using Ptr = std::shared_ptr<Poolable>;

    virtual ~Poolable() = default;

    // less comparison between same type's objects
    virtual bool operator==(const Poolable& rp) const = 0;
    // equal comparsion with Key of an object.
    virtual bool operator==(const Key& rk) const = 0;
    // return true if the rp(Poolable) depends on itself, otherwise false.
    virtual bool is_ancestor_of(const Poolable& rp) const = 0;
    // return true if the Poolable with rk(Key) depends on itself, otherwise false.
    virtual bool is_ancestor_of(const Key& rk) const = 0;
    // return true if itself depends on the rp(Poolable), otherwise false.
    virtual bool is_descendant_of(const Poolable& rp) const = 0;
    // return true if itself depends on the Poolable with rk(Key), otherwise false.
    virtual bool is_descendant_of(const Key& rk) const = 0;
    // return string containg information of an object.
    virtual std::string to_string() const = 0;
};


};  // namespace pool
};  // namespace runtime
};  // namespace enn

#endif  // SRC_RUNTIME_POOL_POOLABLE_HPP_