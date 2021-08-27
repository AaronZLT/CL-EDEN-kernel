#ifndef SRC_MODEL_TYPES_HPP_
#define SRC_MODEL_TYPES_HPP_
#include <map>
#include "model/schema/schema_nnc.h"

namespace enn {
namespace model {

constexpr int32_t UNDEFINED = -1;

enum class Accelerator {
#ifdef SCHEMA_NNC_V1
    NONE    = 0,
    CPU     = 1,
    GPU     = 2,
    NPU     = 4,
    DSP     = 8,
    UNIFIED = 16,
#else
    NONE    = TFlite::TargetHw_NONE,
    CPU     = TFlite::TargetHw_CPU,
    GPU     = TFlite::TargetHw_GPU,
    NPU     = TFlite::TargetHw_NPU,
    DSP     = TFlite::TargetHw_DSP,
    UNIFIED = TFlite::TargetHw_UNIFIED,
#endif
    CUSTOM_CPU_KERNEL = 0x100,
    SIZE
};

inline static bool available_accelerator(Accelerator collection, Accelerator target) {
    return ((static_cast<int>(collection) & static_cast<int>(target)) != 0);
}

enum class Direction {
    Input = 0,
    Output,
    EXT,
    None,
    SIZE
};

static std::map<TFlite::TensorType, uint32_t> pixel_bit_format_size = {
    {TFlite::TensorType::TensorType_FLOAT64, sizeof(int64_t)},
    {TFlite::TensorType::TensorType_FLOAT32, sizeof(float)},
    {TFlite::TensorType::TensorType_FLOAT16, sizeof(int16_t)},
    {TFlite::TensorType::TensorType_UINT64, sizeof(int64_t)},
#ifndef SCHEMA_NNC_V1
    {TFlite::TensorType::TensorType_UINT32, sizeof(uint32_t)},  // Not supported type in NNC_V1
#endif
    {TFlite::TensorType::TensorType_UINT8, sizeof(uint8_t)},
    {TFlite::TensorType::TensorType_INT64, sizeof(int64_t)},
    {TFlite::TensorType::TensorType_INT32, sizeof(int32_t)},
    {TFlite::TensorType::TensorType_INT16, sizeof(int16_t)},
    {TFlite::TensorType::TensorType_INT8, sizeof(int8_t)},
};

};  // namespace model
};  // namespace enn

#endif  // SRC_MODEL_TYPES_HPP_