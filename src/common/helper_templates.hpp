/*
 * Copyright (C) 2021 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or
 * distributed, transmitted, transcribed, stored in a retrieval system or
 * translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed to third parties
 * without the express written permission of Samsung Electronics.
 */

#ifndef SRC_COMMON_HELPER_TEMPLATES_HPP_
#define SRC_COMMON_HELPER_TEMPLATES_HPP_

#include <future>
#include <vector>
#include <type_traits>

#include "model/schema/flatbuffers/flatbuffers.h"

namespace enn {
namespace util {

// Template function that explicitly drives std::async with the policy of std::launch::async
template <typename F, typename... Ps>
inline auto RunAsync(F&& f, Ps&&... params) {
    return std::async(std::launch::async,
                      std::forward<F>(f),
                      std::forward<Ps>(params)...);
}

template <typename T1, typename T2>
inline bool is_map_contain(T1 map, T2 value) {
    return (map.find(value) != map.end());
}

template <typename T1, typename T2>
inline bool is_vector_contain(T1 vector, T2 value) {
    for (auto ele : vector) {
        if (ele == value) {
            return true;
        }
    }
    return false;
}

template <typename T1, typename T2>
inline std::vector<T1> convert_vector(const flatbuffers::Vector<T2>* fb_vector) {
    std::vector<T1> result;
    for (size_t i = 0; i < fb_vector->size(); ++i) {
        result.push_back(static_cast<T1>(fb_vector->Get(i)));
    }
    return result;
}

template<typename E>
constexpr auto underlying_cast(E enumerator) noexcept {
    return static_cast<std::underlying_type_t<E>>(enumerator);
}

};  // namespace util
};  // namespace enn

#endif  // SRC_COMMON_HELPER_TEMPLATES_HPP_