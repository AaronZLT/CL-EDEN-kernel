/*
 * Copyright (C) 2020 Samsung Electronics Co. LTD
 *
 * This software is proprietary of Samsung Electronics.
 * No part of this software, either material or conceptual may be copied or distributed, transmitted,
 * transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
 * electronic, mechanical, manual or otherwise, or disclosed
 * to third parties without the express written permission of Samsung Electronics.
 */

#pragma once

#include <string>
#include "userdriver/common/operator_interfaces/common/ActivationInfo.hpp"
#include "userdriver/common/operator_interfaces/common/Common.hpp"

namespace enn {
namespace ud {
namespace gpu {

struct Parameters {
    virtual ~Parameters() {}
};

struct Pool2DParameters : public Parameters {
    Pad4 padding = {0, 0, 0, 0};
    Dim2 stride = {1, 1};
    Dim2 filter = {1, 1};
    ActivationInfo activation_info = ActivationInfo();
    bool androidNN = false;
    bool isNCHW = true;
    StorageType storage_type = StorageType::BUFFER;
    ComputeType compute_type = ComputeType::TFLite;
};



}  // namespace gpu
}  // namespace ud
}  // namespace enn
