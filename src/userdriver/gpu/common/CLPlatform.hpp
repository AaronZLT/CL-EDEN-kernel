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

#ifndef USERDRIVER_GPU_CL_OPERATORS_CL_PLATFORM_HPP_
#define USERDRIVER_GPU_CL_OPERATORS_CL_PLATFORM_HPP_

#include "CLIncludes.hpp"
#include "userdriver/common/operator_interfaces/common/Common.hpp"

namespace enn {
namespace ud {
namespace gpu {

class CLPlatform {
   public:
    CLPlatform();
    ~CLPlatform();

    Status initialize();

    static std::shared_ptr<CLPlatform> getInstance();

    Status validate(const uint32_t &device_id);

    cl_uint getNumDevices();
    cl_uint getNumPlatforms();

    cl_platform_id *getPlatforms();
    cl_device_id *getDevices();

    cl_platform_id *getPlatform();
    PlatformType getPlatformType() { return platform_type_; }

  private:
    static std::shared_ptr<CLPlatform> instance_;

    cl_uint num_platforms_;
    cl_uint num_devices_;

    cl_platform_id *platforms_;
    cl_device_id *devices_;

    cl_platform_id *selected_platform_;
    PlatformType platform_type_;
};  // class CLPlatform

}  // namespace gpu
}  // namespace ud
}  // namespace enn

#endif  // USERDRIVER_GPU_CL_OPERATORS_CL_PLATFORM_HPP_
