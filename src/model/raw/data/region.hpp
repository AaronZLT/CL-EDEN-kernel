#ifndef SRC_MODEL_RAW_DATA_PERF_REGION_HPP_
#define SRC_MODEL_RAW_DATA_PERF_REGION_HPP_

#include "model/raw/model.hpp"

namespace enn {
namespace model {
namespace raw {
namespace data {

/*
 * Used for DSP(CGO)
 */
class Region {
private:
    uint32_t fd;
    uint32_t size;

    friend class RegionBuilder;
};

class RegionBuilder : public ModelBuilder {
public:
    explicit RegionBuilder(std::shared_ptr<RawModel> raw_model) : ModelBuilder(raw_model){};
};

};  // namespace data
};  // namespace raw
};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_RAW_DATA_PERF_REGION_HPP_
