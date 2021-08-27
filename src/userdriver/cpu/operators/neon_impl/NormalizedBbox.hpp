#pragma once

#include "model/schema/schema_nnc.h"
#include "userdriver/common/operator_interfaces/common/Includes.hpp"

namespace enn {
namespace ud {
namespace cpu {

enum PriorBoxCodingType {
    PRIORBOX_CODETYPE_CORNER = TFlite::PriorBoxCodingType_CORNER,
    PRIORBOX_CODETYPE_CENTERSIZE = TFlite::PriorBoxCodingType_CENTER_SIZE,
    PRIORBOX_CODETYPE_CORNERSIZE = TFlite::PriorBoxCodingType_CORNER_SIZE,
    SIZE
};

class NormalizedBBox {
public:
    NormalizedBBox();

    NormalizedBBox(const NormalizedBBox& normalized_bbox);

    void set_xmin(float value);
    float xmin() const;

    void set_ymin(float value);
    float ymin() const;

    void set_xmax(float value);
    float xmax() const;

    void set_ymax(float value);
    float ymax() const;

    void set_size(float value);
    float size() const;
    bool has_size() const;
    void clear_size();

    void set_difficult(bool value);
    bool difficult() const;

private:
    float xmin_ = 0.0f;
    float ymin_ = 0.0f;
    float xmax_ = 0.0f;
    float ymax_ = 0.0f;
    float size_ = 0.0f;
    bool difficult_ = false;
};

typedef std::map<int, std::vector<NormalizedBBox>> Label_bbox;

}  // namespace cpu
}  // namespace ud
}  // namespace enn
