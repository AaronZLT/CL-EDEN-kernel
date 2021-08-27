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


#ifndef SRC_COMMON_EXTENDED_TYPE_TRAITS_HPP_
#define SRC_COMMON_EXTENDED_TYPE_TRAITS_HPP_

#include <type_traits>

namespace enn {
namespace util {


template<typename T>
struct is_shared_ptr : std::false_type
{};

template<typename T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type
{};


};  // namespace util
};  // namespace enn

#endif  // SRC_COMMON_EXTENDED_TYPE_TRAITS_HPP_