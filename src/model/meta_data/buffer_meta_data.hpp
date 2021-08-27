#ifndef MODEL_METADATA_BUFFERMETADATA_HPP_
#define MODEL_METADATA_BUFFERMETADATA_HPP_

#include <vector>
#include <string>
#include "model/types.hpp"
#include "model/memory/indexed_buffer.hpp"


namespace enn {
namespace model {
namespace metadata {


class BufferMetaData : public std::enable_shared_from_this<BufferMetaData> {
 public:
    using Ptr = std::shared_ptr<BufferMetaData>;

    explicit BufferMetaData() : indexed_buffer{std::make_shared<memory::IndexedBuffer>()} {
        shape.clear();
        data_type = -1;  // Non data type (data type >= 0)
        direction = Direction::None;
        direction_idx = 0;
        name.clear();
        region_idx = 0;
        offset = 0;
    }
    uint32_t get_size() {
        return indexed_buffer->get_size();
    }
    uint32_t get_index() {
        return indexed_buffer->get_index();
    }
    uint32_t get_region_index() {
        return this->region_idx;
    }
    const std::vector<uint32_t>& get_shape() {
        return this->shape;
    }
    Direction get_direction() {
        return this->direction;
    }
    uint32_t get_direction_idx() {
        return this->direction_idx;
    }
    uint32_t get_data_type() {
        return this->data_type;
    }
    const std::string& get_name() {
        return this->name;
    }
    uint32_t get_offset() {
        return this->offset;
    }

    Ptr set_size(uint32_t size) {
        indexed_buffer->set_size(size);
        return shared_from_this();
    }
    Ptr set_index(uint32_t idx) {
        indexed_buffer->set_index(idx);
        return shared_from_this();
    }
    template <typename V>
    Ptr set_shape(V&& shape) {
        this->shape = std::forward<V>(shape);
        return shared_from_this();
    }
    Ptr set_data_type(uint32_t data_type) {
        this->data_type = data_type;
        return shared_from_this();
    }
    Ptr set_direction(Direction direction) {
        this->direction = direction;
        return shared_from_this();
    }
    Ptr set_name(const std::string& str) {
        this->name = str;
        return shared_from_this();
    }
    Ptr set_direction_index(uint32_t dir_idx) {
        this->direction_idx = dir_idx;
        return shared_from_this();
    }
    Ptr set_region_index(uint32_t idx) {
        this->region_idx = idx;
        return shared_from_this();
    }
    Ptr set_offset(uint32_t offset_size) {
        this->offset = offset_size;
        return shared_from_this();
    }

    void print() {
        ENN_DBG_COUT << "- Buffer Meta Data : "
                     << " index(" << this->get_index() << ")"
                     << " region index(" << this->get_region_index() << ")"
                     << " name("  << this->get_name() << ")"
                     << " size("  << this->get_size() << ")"
                     << " direction(" << (int)this->get_direction() << ")"
                     << " direction index(" << this->get_direction_idx() << ")"
                     << std::endl;
    }

 private:
    memory::IndexedBuffer::Ptr indexed_buffer;
    std::vector<uint32_t> shape;
    uint32_t data_type;
    Direction direction;
    uint32_t direction_idx;
    std::string name;
    uint32_t region_idx;
    uint32_t offset;
};

};  // namespace metadata
};  // namespace model
};  // namespace enn

#endif  // MODEL_METADATA_BUFFERMETADATA_HPP_
