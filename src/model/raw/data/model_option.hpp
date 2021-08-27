#ifndef SRC_MODEL_RAW_DATA_MODEL_OPTION_HPP_
#define SRC_MODEL_RAW_DATA_MODEL_OPTION_HPP_

#include "model/raw/model.hpp"

namespace enn {
namespace model {
namespace raw {
namespace data {

class ModelOption {
private:
    int32_t index;
    int32_t legacy_model;
    bool relax_computation_float32_to_float16;

    friend class ModelOptionBuilder;

public:
    ModelOption() : index(UNDEFINED), legacy_model(0), relax_computation_float32_to_float16(true) {}

    int32_t get_index() {
        return index;
    }

    int32_t get_legacy_model() {
        return legacy_model;
    }

    bool is_relax_computation_float32_to_float16() {
        return relax_computation_float32_to_float16;
    }
};

class ModelOptionBuilder : public ModelBuilder {
private:
    std::shared_ptr<ModelOption> model_option_;

public:
    explicit ModelOptionBuilder(std::shared_ptr<RawModel> raw_model) : ModelBuilder(raw_model){};

    ModelOptionBuilder& add_model_option() {
        model_option_ = std::make_unique<ModelOption>();
        return *this;
    }

    ModelOptionBuilder& get_model_option() {
        model_option_ = raw_model_->get_model_options();
        return *this;
    }

    ModelOptionBuilder& set_index(int32_t index) {
        model_option_->index = index;
        return *this;
    }

    ModelOptionBuilder& set_legacy_model(int32_t legacy_model) {
        model_option_->legacy_model = legacy_model;
        return *this;
    }

    ModelOptionBuilder& set_relax_computation_float32_to_float16(bool relax_computation_float32_to_float16) {
        model_option_->relax_computation_float32_to_float16 = relax_computation_float32_to_float16;
        return *this;
    }

    void build() {
        raw_model_->model_options_ = std::move(model_option_);
    }
};

};  // namespace data
};  // namespace raw
};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_RAW_DATA_MODEL_OPTION_HPP_
