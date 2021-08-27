#pragma once

namespace enn {
namespace util {

// Extract an Model's id field from id passed as parameter.
//  This is completely a function of the helper nature.
//  Classes(, such as userdriver) that don't know the existence of the Model class
//  cannot and should not be aware of the model id. It is used for for id into profiler.
inline uint64_t chop_into_model_id(uint64_t id) {
    return id & 0x1FFFFFF000000;
}

}  // namespace util
}  // namespace enn