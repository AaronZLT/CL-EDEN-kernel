#include "NormalizedBbox.hpp"

namespace enn {
namespace ud {
namespace cpu {

NormalizedBBox::NormalizedBBox() {
    xmin_ = 0.0f;
    ymin_ = 0.0f;
    xmax_ = 0.0f;
    ymax_ = 0.0f;
    size_ = 0.0f;
}

NormalizedBBox::NormalizedBBox(const NormalizedBBox& normalized_bbox) {
    xmin_ = normalized_bbox.xmin();
    ymin_ = normalized_bbox.ymin();
    xmax_ = normalized_bbox.xmax();
    ymax_ = normalized_bbox.ymax();
    size_ = normalized_bbox.size();
    difficult_ = normalized_bbox.difficult();
}

void NormalizedBBox::set_xmin(float value) {
    xmin_ = value;
}

float NormalizedBBox::xmin() const {
    return xmin_;
}

void NormalizedBBox::set_ymin(float value) {
    ymin_ = value;
}

float NormalizedBBox::ymin() const {
    return ymin_;
}

void NormalizedBBox::set_xmax(float value) {
    xmax_ = value;
}

float NormalizedBBox::xmax() const {
    return xmax_;
}

void NormalizedBBox::set_ymax(float value) {
    ymax_ = value;
}

float NormalizedBBox::ymax() const {
    return ymax_;
}

void NormalizedBBox::set_size(float value) {
    size_ = value;
}

float NormalizedBBox::size() const {
    return size_;
}

bool NormalizedBBox::has_size() const {
    return size_ != 0;
}

void NormalizedBBox::clear_size() {
    size_ = 0.0f;
}

void NormalizedBBox::set_difficult(bool value) {
    difficult_ = value;
}

bool NormalizedBBox::difficult() const {
    return difficult_;
}

}  // namespace cpu
}  // namespace ud
}  // namespace enn
