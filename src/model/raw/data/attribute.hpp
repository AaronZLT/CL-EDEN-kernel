#ifndef SRC_MODEL_RAW_DATA_ATTRIBUTE_HPP_
#define SRC_MODEL_RAW_DATA_ATTRIBUTE_HPP_

#include "model/raw/model.hpp"

namespace enn {
namespace model {
namespace raw {
namespace data {

/*
 * Attribute is the information for compiled graph {"tflite" and "cgo"}
 */
class Attribute {
private:
    int32_t version;
    int32_t model_type;   // NNC, CGO... in Future
    int32_t nn_api_type;  // ANN, ENN... in Future

    friend class AttributeBuilder;

public:
    int32_t get_version() {
        return version;
    }

    int32_t get_model_type() {
        return model_type;
    }

    int32_t get_nn_api_type() {
        return nn_api_type;
    }
};

class AttributeBuilder : public ModelBuilder {
private:
    std::shared_ptr<Attribute> attribute_;

public:
    explicit AttributeBuilder(std::shared_ptr<RawModel> raw_model) : ModelBuilder(raw_model){};

    AttributeBuilder& add_attribute() {
        attribute_ = std::make_unique<Attribute>();
        return *this;
    }

    AttributeBuilder& get_attribute() {
        attribute_ = raw_model_->get_attribute();
        return *this;
    }

    AttributeBuilder& set_version(int32_t version) {
        attribute_->version = version;
        return *this;
    }

    AttributeBuilder& set_model_type(int32_t model_type) {
        attribute_->model_type = model_type;
        return *this;
    }

    AttributeBuilder& set_nn_api_type(int32_t nn_api_type) {
        attribute_->nn_api_type = nn_api_type;
        return *this;
    }

    void build() {
        raw_model_->attribute_ = std::move(attribute_);
    }
};

};  // namespace data
};  // namespace raw
};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_RAW_DATA_ATTRIBUTE_HPP_
