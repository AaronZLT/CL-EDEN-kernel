#ifndef SRC_MODEL_RAW_DATA_CONTROL_OPTION_HPP_
#define SRC_MODEL_RAW_DATA_CONTROL_OPTION_HPP_

#include "model/raw/model.hpp"

namespace enn {
namespace model {
namespace raw {
namespace data {

/*
 * ControlOption is performance preset configure
 */
class ControlOption {
private:
    int32_t preset_id;
    int32_t latency;

    friend class ControlOptionBuilder;
};

class ControlOptionBuilder : public ModelBuilder {
public:
    explicit ControlOptionBuilder(std::shared_ptr<RawModel> raw_model) : ModelBuilder(raw_model){};
};

};  // namespace data
};  // namespace raw
};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_RAW_DATA_CONTROL_OPTION_HPP_
