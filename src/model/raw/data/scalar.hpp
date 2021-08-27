#ifndef SRC_MODEL_RAW_DATA_SCALAR_HPP_
#define SRC_MODEL_RAW_DATA_SCALAR_HPP_

#include <memory>
#include "common/enn_debug.h"
#include "model/raw/model.hpp"
#include "model/raw/data/tensor.hpp"

namespace enn {
namespace model {
namespace raw {
namespace data {

/*
 * Used for DSP(CGO)
 */
class Scalar : public Tensor {
private:
    int32_t fd;
    int32_t offset;

    friend class ScalarBuilder;

public:
    Scalar() : fd(UNDEFINED), offset(UNDEFINED) {}

    int32_t get_fd() {
        return fd;
    }

    int32_t get_offset() {
        return offset;
    }
};

class ScalarBuilder : public ModelBuilder {
private:
    std::shared_ptr<Scalar> scalar_;

public:
    explicit ScalarBuilder(std::shared_ptr<RawModel> raw_model) : ModelBuilder(raw_model){};

    ~ScalarBuilder() = default;

    ScalarBuilder& add_scalar() {
        scalar_ = std::make_unique<Scalar>();
        return *this;
    }

    ScalarBuilder& get_scalar(uint32_t index) {
        auto tensor = raw_model_->get_tensors().at(index);
        scalar_ = std::dynamic_pointer_cast<Scalar>(tensor);
        if (scalar_ == nullptr) {
            ENN_DBG_PRINT("Tensor[%" PRIu32 "] is not scalar\n", index);
        }
        return *this;
    }

    ScalarBuilder& set_index(int32_t index) {
        scalar_->index = index;
        return *this;
    }

    ScalarBuilder& set_name(std::string name) {
        scalar_->name = name;
        return *this;
    }

    ScalarBuilder& set_type(int32_t type) {
        scalar_->type = type;
        return *this;
    }

    ScalarBuilder& set_prev_operator_index(int32_t prev_operator_index) {
        scalar_->prev_operator_index = prev_operator_index;
        return *this;
    }

    ScalarBuilder& add_next_operator_index(int32_t next_operator_index) {
        scalar_->next_operator_indexes.push_back(next_operator_index);
        return *this;
    }

    template <typename V>
    ScalarBuilder& set_shape(V&& shape) {
        scalar_->shape = std::forward<V>(shape);
        return *this;
    }

    ScalarBuilder& set_address(const uint8_t* address) {
        scalar_->data_addr = address;
        return *this;
    }

    ScalarBuilder& set_size(int32_t size) {
        scalar_->data_size = size;
        return *this;
    }

    ScalarBuilder& set_fd(int32_t fd) {
        scalar_->fd = fd;
        return *this;
    }

    ScalarBuilder& set_offset(uint32_t offset) {
        scalar_->offset = offset;
        return *this;
    }

    void build() {
        raw_model_->tensors_.push_back(std::move(scalar_));
    }
};

};  // namespace data
};  // namespace raw
};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_RAW_DATA_SCALAR_HPP_
