#ifndef SRC_MODEL_RAW_DATA_BINARY_HPP_
#define SRC_MODEL_RAW_DATA_BINARY_HPP_

#include "model/raw/model.hpp"
#include "model/types.hpp"

namespace enn {
namespace model {
namespace raw {
namespace data {

/*
 * Binary is for {"NPU/NCP" and "DSP"} derived from Tensor {"NPU/NCP" or "DSP" / "_BINARY" or "_NAME"}
 */
class Binary {
private:
    int32_t index;
    std::string name;          // .bin file name of "NCP", "DSP"
    const uint8_t* data_addr;  // addr of binary loaded on memory from file
    int32_t data_size;         // size of binary loaded on memory from file
    int32_t buffer_index;      // Buffer[index]
    int32_t fd;
    int32_t offset;
    Accelerator accelerator;

    friend class BinaryBuilder;

public:
    Binary()
        : index(UNDEFINED), data_addr(nullptr), data_size(UNDEFINED), buffer_index(UNDEFINED),
          fd(UNDEFINED), offset(UNDEFINED), accelerator(Accelerator::NONE) {}

    int32_t get_index() {
        return index;
    }

    std::string get_name() {
        return name;
    }

    const uint8_t* get_address() {
        return data_addr;
    }

    int32_t get_size() {
        return data_size;
    }

    int32_t get_buffer_index() {
        return buffer_index;
    }

    int32_t get_fd() {
        return fd;
    }

    int32_t get_offset() {
        return offset;
    }

    Accelerator get_accelerator() {
        return accelerator;
    }
};

class BinaryBuilder : public ModelBuilder {
private:
    std::shared_ptr<Binary> binary_;

public:
    explicit BinaryBuilder(std::shared_ptr<RawModel> raw_model) : ModelBuilder(raw_model){};

    BinaryBuilder& add_binary() {
        binary_ = std::make_unique<Binary>();
        return *this;
    }

    BinaryBuilder& get_binary(uint32_t index) {
        binary_ = raw_model_->get_binaries().at(index);
        return *this;
    }

    BinaryBuilder& set_index(int32_t index) {
        binary_->index = index;
        return *this;
    }

    BinaryBuilder& set_name(std::string name) {
        binary_->name = name;
        return *this;
    }

    BinaryBuilder& set_address(const uint8_t* address) {
        binary_->data_addr = address;
        return *this;
    }

    BinaryBuilder& set_size(int32_t size) {
        binary_->data_size = size;
        return *this;
    }

    BinaryBuilder& set_buffer_index(int32_t index) {
        binary_->buffer_index = index;
        return *this;
    }

    BinaryBuilder& set_fd(int32_t fd) {
        binary_->fd = fd;
        return *this;
    }

    BinaryBuilder& set_offset(int32_t offset) {
        binary_->offset = offset;
        return *this;
    }

    BinaryBuilder& set_accelerator(Accelerator accelerator) {
        binary_->accelerator = accelerator;
        return *this;
    }

    void build() {
        raw_model_->binaries_.push_back(std::move(binary_));
    }
};

};  // namespace data
};  // namespace raw
};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_RAW_DATA_BINARY_HPP_
