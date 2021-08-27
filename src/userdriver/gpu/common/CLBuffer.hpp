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

#ifndef USERDRIVER_GPU_CL_OPERATORS_CL_BUFFER_HPP_
#define USERDRIVER_GPU_CL_OPERATORS_CL_BUFFER_HPP_

#include "userdriver/gpu/common/CLIncludes.hpp"

namespace enn {
namespace ud {
namespace gpu {

class CLBuffer {
   public:
    CLBuffer(size_t byte) : byte_(byte), buffer_(nullptr) {}

    void assignBuffer(cl_mem buffer) { buffer_ = buffer; }

    void setBytes(size_t byte) { byte_ = byte; }

    size_t getBytes() { return byte_; }

    cl_mem getDataPtr() { return buffer_; }

    ~CLBuffer() {}

   private:
    size_t byte_;
    cl_mem buffer_;

};  // class CLBuffer

}  // namespace gpu
}  // namespace ud
}  // namespace enn

#endif  // USERDRIVER_GPU_CL_OPERATORS_CL_BUFFER_HPP_
