#ifndef SRC_RUNTIME_DISPATCH_DISPATCHABLE_HPP_
#define SRC_RUNTIME_DISPATCH_DISPATCHABLE_HPP_

#include "model/types.hpp"

namespace enn {
namespace runtime {
namespace dispatch {

// Dispatchable is an interface class that explicitly represents objects
//  that can be dispatched (passed) to the Userdriver via the dispather.
// That is, the classes implemented by inheriting this interface can be object that
//  can be delivered to the Userdirver via a Dispatcher.
// List of objects that inherit Dispatchable interface.
//  - OperatorList
//  - ExecutableOperatorList
//  - OperatorListExecuteRequest
class Dispatchable {
 public:
    virtual ~Dispatchable() = default;
    virtual const enn::model::Accelerator& get_accelerator() const = 0;
};

}  // namespace enn
}  // namespace runtime
}  // namespace dispatch

#endif  // SRC_RUNTIME_DISPATCH_DISPATCHABLE_HPP_