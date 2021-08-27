#ifndef SRC_MODEL_RAW_DATA_BUFFER_HPP_
#define SRC_MODEL_RAW_DATA_BUFFER_HPP_

#include "model/raw/model.hpp"

namespace enn {
namespace model {
namespace raw {
namespace data {

class Buffer {
private:
    int32_t index;
    std::string name;
    const uint8_t* va;
    int32_t fd;
    int32_t size;
    int32_t offset;

    friend class BufferBuilder;

public:
    Buffer() : index(UNDEFINED), va(nullptr), fd(UNDEFINED), size(UNDEFINED), offset(UNDEFINED) {}

    int32_t get_index() {
        return index;
    }

    std::string get_name() {
        return name;
    }

    const uint8_t* get_address() {
        return va;
    }

    int32_t get_fd() {
        return fd;
    }

    int32_t get_size() {
        return size;
    }

    int32_t get_offset() {
        return offset;
    }
};

class BufferBuilder : public ModelBuilder {
private:
    std::shared_ptr<Buffer> buffer_;

public:
    explicit BufferBuilder(std::shared_ptr<RawModel> raw_model) : ModelBuilder(raw_model){};

    BufferBuilder& add_buffer() {
        buffer_ = std::make_unique<Buffer>();
        return *this;
    }

    BufferBuilder& get_buffer(uint32_t index) {
        buffer_ = raw_model_->get_buffers().at(index);
        return *this;
    }

    BufferBuilder& set_index(uint32_t index) {
        buffer_->index = index;
        return *this;
    }

    BufferBuilder& set_name(std::string name) {
        buffer_->name = name;
        return *this;
    }

    BufferBuilder& set_address(const uint8_t* address) {
        buffer_->va = address;
        return *this;
    }

    BufferBuilder& set_fd(int32_t fd) {
        buffer_->fd = fd;
        return *this;
    }

    BufferBuilder& set_size(uint32_t size) {
        buffer_->size = size;
        return *this;
    }

    BufferBuilder& set_offset(uint32_t offset) {
        buffer_->offset = offset;
        return *this;
    }

    void build() {
        raw_model_->buffers_.push_back(std::move(buffer_));
    }
};

};  // namespace data
};  // namespace raw
};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_RAW_DATA_BUFFER_HPP_
