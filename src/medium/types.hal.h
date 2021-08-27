//// This file is generates from ../../src/medium/hidl/enn/1.0/types.hal

#ifndef SRC_MEDIUM_TYPES_HAL_H_
#define SRC_MEDIUM_TYPES_HAL_H_

// Header Start
#include <string>
#include <vector>

namespace enn {
namespace hal {

struct Buffer;

struct handle_x86 {
    int numFds;
    int data[1];
    void *va;
};
/*
 * Copyright (C) 2018 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */



// types for General interfaces
enum bufferType : int32_t {
    ENN_BUF_TYPE_ION,
    ENN_BUF_TYPE_MALLOC,
    ENN_BUF_TYPE_ASHMEM,
    ENN_BUF_TYPE_UNKNOWN,
    ENN_BUF_TYPE_MAX,
};

struct BufferInfo {
    handle_x86 data;
    handle_x86 *hd = &data;
    bufferType type;
    uint64_t src_addr;
    uint64_t dst_addr;
    uint32_t offset;
    uint32_t size;
};

struct InterfaceBaseInfo {
    uint32_t if_type;
    uint32_t magic;
    uint32_t caller_pid;
    uint32_t caller_tid;
    uint32_t n_mem;   // These are for verification
    uint32_t n_u64;
    uint32_t n_str;
};

struct BufferCore {
    handle_x86 data;
    handle_x86 *fd = &data;
    uint32_t size;
    uint32_t attr;
    uint64_t va;
};

struct GeneralParameter {
    std::vector<uint32_t> u32_v;
    std::vector<std::string> str_v;
};

struct GeneralParameterReturn {
    std::vector<int32_t> i32_v;
    std::vector<std::string> str_v;
};

// Parameter for load
struct LoadParameter {
    BufferCore buf_load_model;
    std::vector<BufferCore> buf_load_params;
    uint32_t model_type;
    GeneralParameter preferences;
};

enum DirType : uint32_t {
    ENN_BUF_DIR_IN,
    ENN_BUF_DIR_OUT,
    ENN_BUF_DIR_EXT,
    ENN_BUF_DIR_NONE,
};

struct ShapeInfo {
    uint32_t n;
    uint32_t w;
    uint32_t h;
    uint32_t c;
};

struct Buffer {
    uint32_t region_idx;
    DirType dir;
    uint32_t buf_index;
    uint32_t size;
    uint32_t offset;
    ShapeInfo shape;   // n x w x h x c
    uint32_t buffer_type;
    std::string name;
    std::vector<uint32_t> reserved;
};

struct Region {
    uint32_t attr;   // MANDATORY, IS_FD, header will define this.
    uint32_t req_size;
    std::string name;
    std::vector<uint32_t> reserved;
};

struct SessionBufInfo {
    uint64_t model_id;   // model_id should be over 1. model_id = 0 means error from service
    std::vector<Buffer> buffers;
    std::vector<Region> regions;
};

// data structures for execution
struct InferenceRegion {  // 1 region = 1 physical memory buffer
    uint32_t exec_attr;   // BLANK, IS_FD, header will define this.
    handle_x86 data;
    handle_x86 *fd = &data;
    uint64_t addr;
    uint32_t size;
    uint32_t offset;
    std::vector<uint32_t> reserved;
};

struct InferenceData {  //  Set of region (session data)
    uint32_t n_region;
    uint64_t exec_model_id;
    bool   is_commit;
    std::vector<InferenceRegion> inference_data;
};

struct InferenceSet {  // Set of inferenceData
    uint32_t n_inference;
    std::vector<InferenceData> inference_set;
};



using handle = handle_x86;  // generator: to make same interface with HIDL types

}  // namespace hal
}  // namespace enn


#endif  // SRC_MEDIUM_TYPES_HAL_H_

// Footer over


