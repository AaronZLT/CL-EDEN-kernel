#ifndef SRC_RUNTIME_DISPATCH_DISPATCHER_INTERFACE_HPP_
#define SRC_RUNTIME_DISPATCH_DISPATCHER_INTERFACE_HPP_

#include <memory>

#include "runtime/dispatch/dispatchable.hpp"
#include "userdriver/common/UserDriver.h"


namespace enn {
namespace runtime {
namespace dispatch {


class IDispatcher {
 public:
    IDispatcher() = default;
    virtual ~IDispatcher() = default;
    virtual void dispatch(const Dispatchable& dispatchable) = 0;
};


};  // namespace dispatch
};  // namespace runtime
};  // namespace enn

#endif  // SRC_RUNTIME_DISPATCH_dispatcher_interface_HPP_
