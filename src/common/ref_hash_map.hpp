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


#ifndef SRC_COMMON_REF_HASH_MAP_HPP_
#define SRC_COMMON_REF_HASH_MAP_HPP_

#include <functional>

namespace enn {
namespace adt {

// Hash functor for reference_wrapper.
template <typename K, typename H>
struct RefHash {
    public:
    constexpr size_t operator()(const std::reference_wrapper<const K>& key) const {
        return H{}(key.get());
    }
};

// Equal functor for reference_wrapper.
template <typename K>
struct RefEqual {
    constexpr bool operator()(const std::reference_wrapper<const K>& lhs,
                              const std::reference_wrapper<const K>& rhs) const {
        return lhs.get() == rhs.get();
    }
};


// "RefHashMap" is alias of std::unordered_map whose the Key is std::reference_wrapper.
//  As the key is referenec_wrapper, the Hash and the KeyEqual functor should be customized.
// T is an element type of std::reference_wrapper
template <typename Key, typename T, typename Hash>
using RefHashMap = std::unordered_map<std::reference_wrapper<const Key>,
                                            T,
                                            RefHash<Key, Hash>,
                                            RefEqual<Key>>;


};  // namespace util
};  // namespace enn

#endif  // SRC_COMMON_REF_HASH_MAP_HPP_