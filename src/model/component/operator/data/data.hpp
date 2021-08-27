#ifndef SRC_MODEL_COMPONENT_OPERATOR_OPERATOR_DATA_DATA_HPP_
#define SRC_MODEL_COMPONENT_OPERATOR_OPERATOR_DATA_DATA_HPP_

#include <memory>
#include <vector>
#include <string>

#include "model/memory/allocated_buffer.hpp"
#include "model/schema/schema_nnc.h"

namespace enn {
namespace model {
namespace component {
namespace data {

using AllocatedBuffer = enn::model::memory::AllocatedBuffer;

class Data {
 public:
    explicit Data() {}
    explicit Data(const std::string& name, const void* addr, size_t size)
        : name{name}, buffer{std::make_shared<AllocatedBuffer>(addr, size)} {}

    explicit Data(const std::string& name, int32_t fd, const void* addr, size_t size)
        : name{name}, buffer{std::make_shared<AllocatedBuffer>(fd, addr, size)} {}

    explicit Data(const std::string& name, int32_t fd, const void* addr, size_t size, int32_t offset)
        : name{name}, buffer{std::make_shared<AllocatedBuffer>(fd, addr, size, offset)} {}

    Data(const Data&) = default;
    Data& operator=(const Data&) = default;

    virtual ~Data() = default;

    const std::string& get_name() const { return name; }
    const void* get_addr() const { return buffer->get_addr(); }
    size_t get_size() const { return buffer->get_size(); }
    int32_t get_fd() const { return buffer->get_fd(); }
    int32_t get_offset() const { return buffer->get_offset(); }

 private:
    std::string       name;
    AllocatedBuffer::Ptr buffer;
};


};  // namespace data
};  // namespace component
};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_COMPONENT_OPERATOR_OPERATOR_DATA_DATA_HPP_
