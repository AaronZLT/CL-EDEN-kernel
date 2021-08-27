#ifndef SRC_MODEL_COMPONENT_OPERATOR_OPERATOR_DATA_BINARY_HPP_
#define SRC_MODEL_COMPONENT_OPERATOR_OPERATOR_DATA_BINARY_HPP_

#include <memory>
#include <vector>
#include <string>

#include "model/component/operator/data/data.hpp"

namespace enn {
namespace model {
namespace component {
namespace data {

class Binary : public Data {
 public:
    explicit Binary() {}
    Binary(const std::string& name, const void* addr, size_t size, Accelerator accelerator)
        : Data{name, addr, size}, accelerator_(accelerator) {}

    Binary(const std::string& name, int32_t fd, const void* addr, size_t size, Accelerator accelerator)
        : Data{name, fd, addr, size}, accelerator_(accelerator) {}

    Binary(const std::string& name, int32_t fd, const void* addr, size_t size, int32_t offset, Accelerator accelerator)
        : Data{name, fd, addr, size, offset}, accelerator_(accelerator) {}

    Accelerator get_accelerator() const {
        return accelerator_;
    }

 private:
    Accelerator accelerator_;
    // Add some members neccesary to describe the Binary
};


};  // namespace data
};  // namespace component
};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_COMPONENT_OPERATOR_OPERATOR_DATA_BINARY_HPP_
