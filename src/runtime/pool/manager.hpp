#ifndef SRC_RUNTIME_POOL_MANAGER_HPP_
#define SRC_RUNTIME_POOL_MANAGER_HPP_

#include <utility>
#include <memory>
#include <mutex>

#include "model/model.hpp"
#include "runtime/executable_model/executable_model.hpp"
#include "runtime/client_process/client_process.hpp"
#include "runtime/execute_request/execute_request.hpp"
#include "runtime/pool/dependency_tree_pool.hpp"
#include "common/identifier.hpp"
#include "common/helper_templates.hpp"

namespace enn {
namespace runtime {
namespace pool {

using namespace enn::model;
using namespace enn::identifier;
using namespace enn::runtime::execute;
using namespace enn::util;

class Manager {
 public:
    using UPtr = std::unique_ptr<Manager>;

    Manager() = default;
    // throw exeception of std::runtime_error if the object already exists in the pool.
    template <typename T> void add(const std::shared_ptr<T>& poolable);
    // throw exeception of std::runtime_error if the object is not found in the pool.
    template <typename T> std::shared_ptr<T> get(FullIDType id) const;
    // throw exeception of std::runtime_error if the object cannot be removed successfully.
    template <typename T> void release(FullIDType id);
    template <typename T> size_t count() const;

 private:
    using PoolableKey = IdentifierBase<FullIDType>;
    enum class TreePoolConfig : size_t {
        CLIENT_PROCESS_LEVEL = 1,
        MODEL_LEVEL = 2,
        EXECUTABLE_MODEL_LEVEL = 3,
        EXECUTE_REQUEST_LEVEL = 4,
        HEIGHT = EXECUTE_REQUEST_LEVEL
    };
    DependencyTreePool<Poolable<PoolableKey>::Ptr, underlying_cast(TreePoolConfig::HEIGHT)> tree_pool_;
};

};  // namespace pool
};  // namespace runtime
};  // namespace enn

#endif  // SRC_RUNTIME_POOL_MANAGER_HPP_
